import pandas as pd

def fix_my_data(file_path):
    df = pd.read_parquet(file_path)
    print(f"Checking {file_path}...")
    
    # 假设你的题干在 'problem' 列，答案在 'solution' 或 'answer' 列
    problem_col = 'problem' 
    answer_col = 'solution' if 'solution' in df.columns else 'answer'

    # 0. 补齐操作：
    df['data_source'] = 'aime26'
    #df['ability'] = 'math'

    # 1. 构造模型需要的对话格式 (Role + Content)
    # verl 期望 prompt 列是一个 list of dicts
    #df['prompt'] = df[problem_col].apply(lambda x: [{'role': 'user', 'content': x}])
    
    # 2. 构造打分器需要的原始文本
    #df['raw_prompt'] = df[problem_col]
    
    # 3. 构造标准答案列
    #df['ground_truth'] = df[answer_col]

    # 2. 核心：构造那个让框架满意的 "reward_model" 嵌套列
    # 它期望的是一行数据里有一个 dict，里面装着正确答案
    # 假设你现在的正确答案列叫 'answer' 或 'solution'
    #ans_col = 'solution' if 'solution' in df.columns else 'answer'
    #if 'ground_truth' in df.columns:
    #    ans_col = 'ground_truth'

    # 重点：把打分信息打包
    #df['reward_model'] = df[ans_col].apply(lambda x: {'ground_truth': x})
    
    # 这里的打印是为了让你用“惊人眼力”最后确认一下
    #print("第一行数据预览：")
    #print(f"Prompt: {df['prompt'].iloc[0]}")
    #print(f"Raw Prompt: {df['raw_prompt'].iloc[0]}")
    
    # 覆盖保存
    df.to_parquet(file_path, index=False)
    print(f"File {file_path} is now FIXED!\n")

# 执行修复
fix_my_data("/data/home/huangqiyuan/Project1/data/AIME26_fixed.parquet")
# fix_my_data("/data/home/huangqiyuan/mutant_grpo/data/AIME25_fixed.parquet")
# fix_my_data("/data/home/huangqiyuan/mutant_grpo/data/AIME26_fixed.parquet")
# fix_my_data("/data/home/huangqiyuan/mutant_grpo/data/AIME24-26.parquet")
