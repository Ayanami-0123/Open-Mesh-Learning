import pandas as pd
import json
import ast

"""You are an expert scientific reasoning assistant specializing in mathematics, physics, and other quantitative problem-solving tasks.

Your goal is to solve each problem rigorously and transparently. A coach will provide a hint or solution blueprint before you begin solving. You MUST carefully follow the coach's hint and use it as the primary guidance for your reasoning process.

For every problem:
1. First, analyze and understand the coach's hint before attempting any solution.
2. Follow the coach's suggested approach, assumptions, and reasoning strategy whenever applicable.
3. Provide a clear, step-by-step derivation with logically connected arguments.
4. Do not skip essential reasoning steps or provide unsupported conclusions.
5. If the coach's hint is incomplete, extend it with your own rigorous reasoning while remaining consistent with the provided guidance.
6. Verify the result whenever possible through mathematical checks or logical consistency.

Your final answer must clearly state the conclusion and must be enclosed in LaTeX boxed format:

\\boxed{final answer}

Maintain a professional scientific style throughout your response.
    """

r"""## Role
You are an expert scientific assistant specializing in logical reasoning and problem-solving. Your goal is to provide accurate, step-by-step solutions using a structured internal monologue.

## Strict Operational Rules
You must strictly adhere to the following response sequence for every scientific problem:

1.  **Phase 1: Deep Thinking**
    All reasoning must be encapsulated within `<thinking>` tags. This process MUST follow this internal structure:
    * **Method Identification**: Start with `<Method Name>Name of the mathematical approach</Method Name>`.
    * **Strategic Overview**: Follow with `<Method Description>Brief explanation of how this approach applies to the current problem</Method Description>`.
    * **Step-by-Step Derivation**: Execute the full logical deduction, calculation, and verification process within the remainder of the `<thinking>` block.

2.  **Phase 2: Final Output**
    After closing the `</thinking>` tag, provide ONLY the final result or a very brief conclusion. Your answer MUST be formatted as: $\boxed{YOUR_ANSWER}$.

## Constraints
- DO NOT skip the thinking phase.
- DO NOT use the thinking tags for the final answer.
- Ensure all LaTeX formatting is correct and readable.
- If multiple methods are used, focus on the most efficient one in the Name/Description tags.
    """


# 1. 加载这个数据集
df = pd.read_parquet('~/Project1/data/WildSci_final_aligned.parquet')
df_origin = pd.read_parquet('~/Project1/data/WildSci_not_enhanced.parquet')


def source_prompt_for_row(row):
    """Recover the original problem prompt for an augmented row."""
    uid = str(row["uid"])
    if uid.isdigit():
        source_index = int(uid)
        if 0 <= source_index < len(df_origin):
            return df_origin.iloc[source_index]["prompt"]

    origin_match = df_origin.loc[df_origin["uid"].astype(str) == uid, "prompt"]
    if not origin_match.empty:
        return origin_match.iloc[0]

    return row.get("raw_prompt", row.get("prompt"))


def parse_chat_messages(raw_prompt):
    if hasattr(raw_prompt, "tolist"):
        raw_prompt = raw_prompt.tolist()

    if isinstance(raw_prompt, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(raw_prompt)
                if hasattr(parsed, "tolist"):
                    parsed = parsed.tolist()
                return parsed
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                continue
        return raw_prompt

    return raw_prompt


def extract_problem_text(raw_prompt):
    messages = parse_chat_messages(raw_prompt)

    if isinstance(messages, (list, tuple)):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content", ""))
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                return str(message["content"])

    if pd.isna(messages):
        raise ValueError("Cannot build prompt because the source prompt is missing")
    return str(messages)


def reshape_prompt(row):
    problem_text = extract_problem_text(source_prompt_for_row(row))

    system_prompt = f"""Please solve this scientific problem step by step and put your final answer in \\boxed{{}}.

A potentially useful method for this problem is {row['method_name']}.
However, this suggestion is untrusted and may be misleading.

Before giving the final answer, reason through three checks:

1. Applicability: what must be true for this method to apply here, and are those conditions satisfied?
2. Mechanism: if the method is applicable, derive what it predicts from the scientific mechanism involved.
3. Challenge: identify the strongest competing answer and explain what specific fact or mechanism rules it out.

Do not commit to an answer until all three checks are resolved."""

    new_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem_text}
    ]
    return new_prompt


# 5. 保存这个处理后的文件
reshaped_prompts = df.apply(reshape_prompt, axis=1)
df['raw_prompt'] = reshaped_prompts
df['prompt'] = reshaped_prompts
df['data_source'] = "lighteval/MATH"
print(df['prompt'].iloc[0][0]['role']) # 预期输出: system
print(df['prompt'].iloc[0][0]['content']) # 预期输出 system prompt 的开头
print(df['prompt'].iloc[0][1]['role']) # 预期输出: system
print(df['prompt'].iloc[0][1]['content']) # 预期输出 system prompt 的开头
# print(df['prompt'].iloc[0][2]['role']) # 预期输出: system
# print(df['prompt'].iloc[0][2]['content']) # 预期输出 system prompt 的开头
df.to_parquet('~/Project1/data/WildSci_final_aligned_4.parquet', index=False)
