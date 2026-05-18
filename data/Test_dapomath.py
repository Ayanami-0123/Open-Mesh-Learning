import pandas as pd
import asyncio
import aiohttp
import vllm
from tqdm import tqdm
import re
import copy
from functools import wraps

# 加载数据集
df = pd.read_parquet('dapomath_7000_final_aligned.parquet')

# 加载模型：
API_KEY = "MyKey"
MODEL = "/data/home/huangqiyuan/mutant_grpo/models/qwen" 
URL = "http://localhost:8000/v1/chat/completions"

def extract_answers(text):
# 1. 预处理：只截取最后一个思考标签之后的内容，直接干掉干扰源
    clean_text = text.split("</thinking>")[-1].strip() if "</thinking>" in text else text.strip()

    # 2. 定义优先级正则（按可靠性排序）
    # p1: 匹配 \boxed{...}，支持嵌套一层花括号
    # p2: 匹配 $...$，排除包含 $ 的内容，防止跨配对
    # p3: 兜底匹配：提取最后出现的数字（含小数）
    patterns = [
        r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
        r"\$([^\$]+?)\$",
        r"(\d+(?:\.\d+)?)"
    ]

    # 3. 按优先级尝试提取
    for i, pattern in enumerate(patterns):
        # DOTALL 确保能匹配换行符；从后往前找
        matches = re.findall(pattern, clean_text, flags=re.DOTALL)
        if matches:
            # 取最后一个匹配项
            final_match = matches[-1].strip()
            
            # 针对 $...$ 模式的额外清洗：如果匹配项包含 "answer is" 等废话，跳过它
            if i == 1 and ("answer" in final_match.lower() or len(final_match) > 100):
                continue
                
            return final_match

    return None

def async_retry(retries=3, backoff_delay=5):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    return await func(*args, **kwargs)
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    if i == retries - 1: raise e
                    wait = backoff_delay * (2 ** i) # 指数退避: 5s, 10s, 20s
                    # print(f"⚠️ 网络错误, {wait}s 后重试 ({i+1}/{retries})...")
                    await asyncio.sleep(wait)
            return None
        return wrapper
    return decorator

@async_retry(retries=3)
async def fetch_completion(session,payload):
    async with session.post(URL, 
        headers={"Authorization": f"Bearer {API_KEY}"}, 
        json=payload, 
        timeout=300) as response:
        if response.status == 429: # Rate limit
            await asyncio.sleep(20)
            raise aiohttp.ClientError("Rate limited")
        if response.status != 200:
            return None
        data = await response.json()
        return data['choices'][0]['message']['content']

async def fetch_all_choices(session, payload):
    async with session.post(URL, headers={"Authorization": f"Bearer {API_KEY}"}, json=payload, timeout=600) as response:
        if response.status != 200:
            return []
        data = await response.json()
        # 返回所有 choices 的 content 文本
        return [choice['message']['content'] for choice in data['choices']]

# 测试模型：
async def async_test(session, semaphore, row):
    uid = row['uid']
    raw_prompt = row['prompt']
    problem = copy.deepcopy(list(raw_prompt)) if not hasattr(raw_prompt, 'tolist') else copy.deepcopy(raw_prompt.tolist())
    
    right_answer = str(row['reward_model']['ground_truth']).strip()
    
    # 构造 Pass@16 的 Payload
    # 注意：vLLM 的 n=16 会在一次推理中生成 16 个输出，非常高效
    payload_base = {
        "model": MODEL,
        "messages": problem,
        "temperature": 0.7, # 提高温度以增加多样性
        "max_tokens": 8192,
        "n": 16,            # 核心参数：一次生成16个结果
    }

    # 如果需要对比 Hinted 的 Pass@16
    method_name = row['method_name']
    method_description = row['method_idea']
    hint_message = f"<thinking> <Method Name> {method_name} </Method Name> <Method Description> {method_description} </Method Description>"
    problem_hinted = problem + [{"role": "assistant", "content": hint_message}]
    
    payload_hinted = copy.deepcopy(payload_base)
    payload_hinted["messages"] = problem_hinted

    async with semaphore:
        tasks = [
            fetch_all_choices(session, payload_base),
            fetch_all_choices(session, payload_hinted)
        ]
        responses = await asyncio.gather(*tasks)
        
        res_list_base = responses[0] or []
        res_list_hinted = responses[1] or []
        
        # 定义内部判断函数
        def check_any_correct(res_list, target):
            for text in res_list:
                ans = extract_answers(text)
                if str(ans).strip() == str(target).strip():
                    return True, ans, text # 返回是否正确及最后一个提取到的样本当作展示
            last_text = res_list[-1] if res_list else None
            return False, extract_answers(last_text) if last_text else None, last_text if last_text else None

        base_pass, last_ans_base, last_ans_base_text = check_any_correct(res_list_base, right_answer)
        hinted_pass, last_ans_hinted, last_ans_hinted_text = check_any_correct(res_list_hinted, right_answer)

        # 打印 DEBUG 信息（取每个列表的最后一个作为样本展示，避免报错）
        if res_list_base:
            sample_text = res_list_base[0].replace('\n', ' ')[:100] # 只取前100字
            print(f"\n[UID:{uid}] Base Pass: {base_pass} | Sample: {last_ans_base}")
            print(f"Sample Text: {last_ans_base_text}")
            
        if res_list_hinted:
            print(f"[UID:{uid}] Hinted Pass: {hinted_pass} | Sample: {last_ans_hinted}")
            print(f"Sample Text: {hint_message+last_ans_hinted_text}")
        
        return {
            "uid": uid, 
            "base_correct": base_pass, 
            "hinted_correct": hinted_pass
        }
    '''
    async with semaphore:
        results = {"uid": uid, "base_correct": False, "hinted_correct": False}
        
        # 并发请求两个版本，节省时间
        tasks = [
            fetch_completion(session, payload),
            fetch_completion(session, payload_hinted)
        ]
        
        responses = await asyncio.gather(*tasks)
        
        # 结果比对 (Base)
        if responses[0]:
            
        return results
    '''

async def run_benchmark(df):
    semaphore = asyncio.Semaphore(8) # OpenRouter 建议并发不要太猛
    connector = aiohttp.TCPConnector(limit=50)
    
    final_scores = []
    
    stats = {"total": 0, "base_correct_cnt": 0, "hinted_correct_cnt": 0}
    
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [async_test(session, semaphore, row) for _, row in df.iterrows()]
        
        pbar = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Benchmark")
        for task in pbar:
            res = await task
            if res:
                final_scores.append(res)
                
                # 更新累计状态
                stats["total"] += 1
                if res["base_correct"]: stats["base_correct_cnt"] += 1
                if res["hinted_correct"]: stats["hinted_correct_cnt"] += 1
                
                # 计算当前瞬时准确率
                curr_base_acc = stats["base_correct_cnt"] / stats["total"]
                curr_hint_acc = stats["hinted_correct_cnt"] / stats["total"]
                
                # 实时更新 tqdm 的后缀显示 (Postfix)
                pbar.set_postfix({
                    "Base": f"{curr_base_acc:.1%}",
                    "Hinted": f"{curr_hint_acc:.1%}",
                    "Diff": f"{(curr_hint_acc - curr_base_acc):+.1%}"
                })
    
    # 计算准确率
    score_df = pd.DataFrame(final_scores)
    base_acc = score_df['base_correct'].mean()
    hint_acc = score_df['hinted_correct'].mean()
    
    print(f"\n--- 测试报告 ---")
    print(f"总样本数: {len(score_df)}")
    print(f"原始准确率 (Base Accuracy): {base_acc:.2%}")
    print(f"带思路准确率 (Hinted Accuracy): {hint_acc:.2%}")
    print(f"提升幅度: {(hint_acc - base_acc):.2%}")
    
    return score_df

# 执行
if __name__ == "__main__":
    asyncio.run(run_benchmark(df.head(10)))