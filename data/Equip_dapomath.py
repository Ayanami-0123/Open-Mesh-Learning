import asyncio
import aiohttp
import pandas as pd
import json
import os
import re
from tqdm.asyncio import tqdm

# --- 1. 核心修复：正则表达式 (Regex) ---
def parse_methods(response_text):
    # 更加宽容的匹配：
    # - 兼容 Can use / Can Use / Can-use
    # - 兼容 [Method] / [思路] / Method:
    # - 兼容各种冒号和空格
    # - 允许 Method 后面有无点号
    # 这个正则去掉了对方括号的强制要求，并兼容了更多的关键词
    pattern = r"(?:Can\s*use|可以使用|Method)\s*(?:\d\.)?\s*\[?(.*?)\]?\.?\s*(?:\[Method\]|思路|Method)[:：]\s*(.*?)(?=(?:\n\s*(?:Can\s*use|可以使用|Method))|$)"    
    matches = re.findall(pattern, response_text, re.IGNORECASE | re.DOTALL)
    
    cleaned_matches = []
    for method_name, method_desc in matches:
        cleaned_matches.append((method_name.strip(), method_desc.strip()))
    return cleaned_matches

# --- 2. 基础配置 ---
API_KEY = "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" # 建议检查一下 Key 是否正确
# 注意：OpenRouter 目前对应的 ID 通常是 google/gemini-pro-1.5 或 google/gemini-pro-1.5-exp
MODEL = "deepseek/deepseek-v3.2-exp" 
URL = "https://openrouter.ai/api/v1/chat/completions"

# 路径设置
output_dir = '/data/home/huangqiyuan/mutant_grpo/data'
os.makedirs(output_dir, exist_ok=True)
CHECKPOINT_FILE = os.path.join(output_dir, 'processed_uids.json')
OUTPUT_FILE = os.path.join(output_dir, 'dapomath_7000_augmented.parquet')

# --- 3. 数据加载与断点 ---
df_original = pd.read_parquet('/data/home/huangqiyuan/mutant_grpo/data/dapomath_7000.parquet')

if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, 'r') as f:
        processed_uids = set(json.load(f))
else:
    processed_uids = set()

final_results = []
if os.path.exists(OUTPUT_FILE):
    try:
        final_results = pd.read_parquet(OUTPUT_FILE).to_dict('records')
    except:
        final_results = []

def save_data():
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(list(processed_uids), f)
    if final_results:
        df_save = pd.DataFrame(final_results)
        df_save.to_parquet(OUTPUT_FILE, index=False)
        print(f"\n💾 已存档: {len(processed_uids)} 题 | 总行数: {len(final_results)}")

# --- 4. 异步核心 (带详细报错) ---
async def async_generate(session, semaphore, row):
    uid = str(row.name)
    if uid in processed_uids:
        return None

    problem = row['prompt'][0]['content']
    prompt = f"""You are an expert mathematical competition coach. Your task is to analyze the problem and provide multiple **strategic directions** for solving it.

<TASK_RULES>
1. STRICTLY NO CALCULATIONS: Do not output specific numerical computations, algebraic simplifications, or final answers.
2. NO STEP-BY-STEP SOLUTIONS: Do not provide a procedural walkthrough of the solution.
3. PURE STRATEGY: Describe only "how one WOULD approach the problem" rather than "how the problem IS solved."
4. QUANTITY: You must provide EXACTLY 4 distinct approaches.
</TASK_RULES>

Please follow the output format strictly. Do not include any introductory or concluding remarks.

<OUTPUT_FORMAT>
Can use [Method Name].
[Method]:Specific strategic description (strictly no calculations).

Can use [Method Name].
[Method]:Specific strategic description (strictly no calculations).
... (Repeat until there are exactly 4 approaches)
</OUTPUT_FORMAT>

<EXAMPLE>
Problem: Given $x+y=1$ and $x^2+y^2=2$, find $x^3+y^3$.
Answer:
Can use [Algebraic Manipulation].
[Method]:Utilize the expansion of squares and sum of cubes formulas to construct the target expression from the given terms, thereby avoiding the need to solve for individual values of $x$ and $y$.

Can use [Equation Construction].
[Method]:Treat $x$ and $y$ as the two real roots of a quadratic equation. Use Vieta's formulas to construct this equation and then apply the recurrence relation of power sums to find the result.

Can use [Functional Substitution].
[Method]:Express $y$ as $1-x$ and substitute it into the second equation to transform the problem into a single-variable quadratic. Analyze the root distribution to determine the target value.

Can use [Symmetric Polynomials].
[Method]:Introduce elementary symmetric polynomials $e_1$ and $e_2$. Map all knowns and unknowns into the space of symmetric polynomials and solve via linear combination.
</EXAMPLE>

<PROBLEM_TO_ANALYZE>
{problem}
</PROBLEM_TO_ANALYZE>
"""
    
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "include_reasoning": True,
    }

    async with semaphore:
        for attempt in range(3):
            try:
                async with session.post(URL, 
                                        headers={"Authorization": f"Bearer {API_KEY}"}, 
                                        json=payload, 
                                        timeout=60) as response:
                    
                    # 关键修复：如果状态码不是 200，打印出来看看
                    if response.status != 200:
                        err_text = await response.text()
                        if attempt == 0: # 只打一次，避免刷屏
                            print(f"\n❌ API 错误 (UID {uid}): 状态码 {response.status} | 信息: {err_text}")
                        await asyncio.sleep(5)
                        continue

                    data = await response.json()
                    raw_text = data['choices'][0]['message']['content']
                    
                    methods = parse_methods(raw_text)
                    if len(methods) == 4:
                        return uid, methods, row.get('solution', '')
                    else:
                        if attempt == 2:
                            print(f"\n⚠️ 解析失败 (UID {uid}): 仅抓取到 {len(methods)} 条。")
                            # 调试用：打印前 100 个字符
                            # print(f"DEBUG OUTPUT: {raw_text[:100]}")
            except Exception as e:
                await asyncio.sleep(2)
        return None

async def run_pipeline(df):
    semaphore = asyncio.Semaphore(8)
    # 增加连接池限制，防止被服务器断开
    connector = aiohttp.TCPConnector(limit=50)
    async with aiohttp.ClientSession(connector=connector) as session:
        # 别一次性创建 7500 个任务，按批次处理更稳定
        tasks = [async_generate(session, semaphore, row) for _, row in df.iterrows()]
        
        for i, task in enumerate(tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing")):
            res = await task
            if res:
                uid, methods, solution = res
                processed_uids.add(uid)
                for idx, (m_name, m_idea) in enumerate(methods):
                    final_results.append({
                        "uid": uid,
                        "method_id": f"method_{idx+1}",
                        "method_name": m_name,
                        "method_idea": m_idea,
                        "ground_truth": solution
                    })
                
                if (i + 1) % 20 == 0:
                    save_data()
            else:
            # 如果失败了，至少打印个点或者提示，别让我们干等着
                print("×", end="", flush=True)
        save_data()

# 执行
if __name__ == "__main__":
    asyncio.run(run_pipeline(df_original))