import asyncio
import aiohttp
from openai import AsyncOpenAI
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
URL = "https://openrouter.ai/api/v1"
models = [
    "deepseek/deepseek-v3.2-exp",
    "openai/gpt-5.6-luna",
    "anthropic/claude-opus-4.8",
] 
client = AsyncOpenAI(
    base_url=URL,
    api_key=API_KEY,
)

class ModelPool:

    def __init__(self, models):
        self.queue = asyncio.Queue()

        for m in models:
            self.queue.put_nowait(m)


    async def acquire(self):
        return await self.queue.get()


    def release(self, model):
        self.queue.put_nowait(model)

# 路径设置
output_dir = '~/Project1/data'
os.makedirs(output_dir, exist_ok=True)
CHECKPOINT_FILE = os.path.join(output_dir, 'processed_uids.json')
OUTPUT_FILE = os.path.join(output_dir, 'WildSci_final_aligned.parquet')

# --- 3. 数据加载与断点 ---
df_original = pd.read_parquet('~/Project1/data/WildSci_not_enhanced.parquet')

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

async def query_model(
    target_model,
    prompt
):
    try:
        response = await client.chat.completions.create(
            model=target_model,
            messages=[
                {
                    "role":"user",
                    "content":prompt
                }
            ],
            temperature=0.3,
            timeout=120,
        )

        return response.choices[0].message.content

    except Exception as e:
        raise RuntimeError(f"""
            ⚠️Model failed:
            MODEL:{target_model}
            ERROR:{str(e)}
            """)

# --- 4. 异步核心 (带详细报错) ---
async def async_generate(
    session,
    semaphore,
    pool,
    row
):
    uid = str(row.name)
    if uid in processed_uids:
        return None

    problem = row['prompt'][1]['content']
    prompt = f"""You are an expert scientific competition coach. Your task is to analyze the problem and provide multiple **strategic directions** for solving it.

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
Problem:
A researcher observes that a newly discovered protein mutation affects cellular signaling pathways. Which mechanism best explains this observation?

Answer:
Can use [Mechanism-Based Analysis].
[Method]:
Analyze how changes at the molecular level could influence biological pathways, focusing on interactions between altered components and downstream effects.

Can use [Experimental Evidence Evaluation].
[Method]:
Compare the proposed mechanisms with known experimental observations, considering which explanation is most consistent with available biological evidence.

Can use [Causal Pathway Construction].
[Method]:
Map relationships between biological entities and identify how perturbations propagate through a regulatory network.

Can use [Alternative Hypothesis Elimination].
[Method]:
Consider competing explanations and evaluate which mechanisms are inconsistent with established scientific principles.

</EXAMPLE>

<PROBLEM_TO_ANALYZE>
{problem}
</PROBLEM_TO_ANALYZE>
"""
    MODEL = await pool.acquire()
    try:
        async with semaphore:
            for attempt in range(3):
                try:
                    raw_text = await query_model(MODEL, prompt)
                    methods = parse_methods(raw_text)
                    if len(methods) == 4:
                        return uid, methods, row.get('solution', ''), MODEL
                    else:
                        if attempt == 2:
                            print(f"\n⚠️ 解析失败 (UID {uid}): 模型 {MODEL} 仅抓取到 {len(methods)} 条。")
                            print(raw_text)

                except Exception as e:
                    print(MODEL,e)
                    await asyncio.sleep(2)
            return None

    finally:
        pool.release(MODEL)

async def run_pipeline(df):
    semaphore = asyncio.Semaphore(24)
    pool = ModelPool(models)
    # 增加连接池限制，防止被服务器断开
    connector = aiohttp.TCPConnector(limit=50)
    async with aiohttp.ClientSession(connector=connector) as session:
        # 别一次性创建 7500 个任务，按批次处理更稳定
        tasks = [async_generate(
            session,
            semaphore,
            pool,
            row
            )
            
            for _, row in df.iterrows()]
        
        for i, task in enumerate(tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing")):
            res=await task
            if res:
                uid,methods,solution,model=res
                print(f"✓ {uid} by {model}")
                processed_uids.add(uid)

                uid_int = int(uid)
                for idx, (m_name, m_idea) in enumerate(methods):
                    final_results.append({
                        "uid": uid,
                        "prompt":df_original.at[uid_int,"prompt"],
                        "raw_prompt":df_original.at[uid_int,"raw_prompt"],
                        "data_source":"lighteval/MATH",
                        "reward_model":df_original.at[uid_int,"reward_model"],
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