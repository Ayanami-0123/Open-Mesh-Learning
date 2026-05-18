import pandas as pd
import json

# 1. 加载这个数据集
df = pd.read_parquet('dapomath_7000_final_aligned.parquet')

def reshape_prompt(row):
    raw_prompt = row['prompt']
    if isinstance(raw_prompt, str):
        try:
            content_list = json.loads(raw_prompt.replace("'", '"')) # 简单处理单引号
            problem_text = content_list[0]['content']
        except:
            problem_text = str(raw_prompt)
    else:
        problem_text = raw_prompt[0]['content']

    system_prompt = r"""## Role
You are an expert mathematical assistant specializing in logical reasoning and problem-solving. Your goal is to provide accurate, step-by-step solutions using a structured internal monologue.

## Strict Operational Rules
You must strictly adhere to the following response sequence for every mathematical problem:

1.  **Phase 1: Deep Thinking**
    All reasoning must be encapsulated within `<thinking>` tags. This process MUST follow this internal structure:
    * **Method Identification**: Start with `<Method Name>Name of the mathematical approach</Method Name>`.
    * **Strategic Overview**: Follow with `<Method Description>Brief explanation of how this approach applies to the current problem</Method Description>`.
    * **Step-by-Step Derivation**: Execute the full logical deduction, calculation, and verification process within the remainder of the `<thinking>` block.

2.  **Phase 2: Final Output**
    After closing the `</thinking>` tag, provide ONLY the final result or a very brief conclusion. The core numerical or symbolic answer must be formatted as: $\boxed{YOUR_ANSWER}$.

## Constraints
- DO NOT skip the thinking phase.
- DO NOT use the thinking tags for the final answer.
- Ensure all LaTeX formatting is correct and readable.
- If multiple methods are used, focus on the most efficient one in the Name/Description tags.
    """

    new_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem_text}
    ]
    return new_prompt

# 5. 保存这个处理后的文件
df['raw_prompt'] = df.apply(reshape_prompt, axis=1)
df['data_source'] = "lighteval/MATH"
print(df['prompt'].iloc[0][0]['role']) # 预期输出: system
print(df['prompt'].iloc[0][0]['content'][:50]) # 预期输出 system prompt 的开头
df.to_parquet('dapomath_7000_final_aligned.parquet', index=False)