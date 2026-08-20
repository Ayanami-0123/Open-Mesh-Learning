import os
from openai import OpenAI

API_KEY = "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" # 建议检查一下 Key 是否正确
URL = "https://openrouter.ai/api/v1"
client=OpenAI(
    base_url=URL,
    api_key=API_KEY,
)

models = [
    "deepseek/deepseek-v3.2-exp",
    "openai/gpt-5.6-luna",
    "anthropic/claude-opus-4.8",
] 

for MODEL in models:
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "give me the direct answer of 2^10 + 1."
                }
            ],
            max_tokens=20,
            temperature=0,
        )

        print(f"\n===== {MODEL} SUCCESS =====")
        print(
            response.choices[0].message.content
        )

    except Exception as e:
        print(f"\n==== {MODEL} FAILURE ====")
        print(type(e).__name__)
        print(e)