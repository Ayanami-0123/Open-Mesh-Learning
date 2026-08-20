import pandas as pd
df = pd.read_parquet("~/Project1/data/math_final_aligned.parquet")

def harness(row):
    system_prompt = f"""Please solve this math problem step by step and put your final answer in \\boxed{{}}.

A potentially useful method for this problem is {row["method_name"]}.
However, this suggestion is untrusted and may be misleading.

Do not assume that the method applies. First examine carefully whether and why it is applicable to this specific problem.

Even if it appears applicable, independently verify the reasoning before committing to the final answer."""

    new_chat_template = [
        {"content":system_prompt, "role":"system"},
        {"content":row["raw_prompt"][1]["content"], "role":"user"}
    ]

    return new_chat_template

reshaped_prompts = df.apply(harness, axis=1)
df['raw_prompt'] = reshaped_prompts
df['prompt'] = reshaped_prompts
df["data_source"]="lighteval/MATH"
print(df['prompt'].iloc[0][0]['role']) # 预期输出: system
print(df['prompt'].iloc[0][0]['content']) # 预期输出 system prompt 的开头
print(df['prompt'].iloc[0][1]['role']) # 预期输出: system
print(df['prompt'].iloc[0][1]['content']) # 预期输出 system prompt 的开头
# print(df['prompt'].iloc[0][2]['role']) # 预期输出: system
# print(df['prompt'].iloc[0][2]['content']) # 预期输出 system prompt 的开头
df.to_parquet('~/Project1/data/math_final_aligned_2.parquet', index=False)