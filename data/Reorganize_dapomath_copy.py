import pandas as pd

# 1. 加载两个数据集
df_original = pd.read_parquet('dapomath_7000.parquet')
df_aug = pd.read_parquet('dapomath_7000_augmented.parquet')

# 2. 给原始数据集显式地创建一个 'uid' 列
# 必须使用 str()，因为你的 aug 表里 uid 是字符串格式（'5350' 而不是 5350）
df_original['uid'] = df_original.index.astype(str)

# 3. 对齐！
# 我们只需要 original 里的 prompt, reward_model(包含答案), 和 extra_info
# 将它们合并到 augmented 数据集里
df_final = pd.merge(
    df_aug, 
    df_original[['uid','data_source', 'prompt', 'ability', 'reward_model', 'extra_info', 'raw_prompt']], 
    on='uid', 
    how='left'
)

# 4. 验证一下
print(f"合并后的样本量: {len(df_final)}")
print(df_final.head())

# 5. 保存这个救命的文件
df_final.to_parquet('dapomath.parquet', index=False)