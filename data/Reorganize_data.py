import pandas as pd
import re
import regex

# 1. 加载原始数据集
df_original = pd.read_parquet('/data/home/huangqiyuan/mutant_grpo/data/math_final_aligned.parquet')

# 2. 设定数据来源和能力
df_original['data_source'] = 'lighteval/MATH'
df_original['ability'] = 'math'

# 3. 提取答案
def extract_answer(text):
    match = regex.search(r"\\boxed{(\s*(?:[^{}]+|\{(?1)\})*\s*)}", text)
    return match.group(1) if match else None
df_original['ground_truth'] = df_original['solution'].apply(extract_answer)
df_original['reward_model'] = df_original['ground_truth'].apply(lambda x: {'ground_truth': x})

# 4. 构造模型需要的对话格式 (Role + Content + Prompt)
sys_prompt = r"""You are an expert mathematical assistant specializing in logical reasoning and problem-solving. Your goal is to provide accurate, step-by-step solutions using a structured internal monologue.

You must strictly adhere to the following response sequence for every mathematical problem:

1.  **Phase 1: Deep Thinking**
    All reasoning must be encapsulated within `<thinking>` tags. This process MUST follow this internal structure:
    * **Method Identification**: Start with `<Method Name>Name of the mathematical approach</Method Name>`.
    * **Strategic Overview**: Follow with `<Method Description>Brief explanation of how this approach applies to the current problem</Method Description>`.
    * **Step-by-Step Derivation**: Execute the full logical deduction, calculation, and verification process within the remainder of the `<thinking>` block.

2.  **Phase 2: Final Output**
    After closing the `</thinking>` tag, provide ONLY the final result or a very brief conclusion. The core numerical or symbolic answer must be formatted as: $\boxed{YOUR_ANSWER}$.

- DO NOT skip the thinking phase.
- DO NOT use the thinking tags for the final answer.
- Ensure all LaTeX formatting is correct and readable.
- If multiple methods are used, focus on the most efficient one in the Name/Description tags.
"""

df_original['prompt'] = df_original.apply(lambda row: [
    {'role': 'system', 'content': sys_prompt},
    {'role': 'user', 'content': row['problem']},
    {'role': 'assistant', 'content': f"<thinking>\n<Method Name>{row['method_name']}</Method Name>\n<Method Description>{row['method_idea']}</Method Description>"}
], axis=1)
df_original['raw_prompt'] = df_original['prompt']

# 5. 保存数据集
test_str = df_original['prompt'].iloc[0][2]['content']
print(test_str)
print(df_original['raw_prompt'].iloc[0])

df_original.to_parquet('/data/home/huangqiyuan/mutant_grpo/data/math_final_aligned.parquet', index=False)