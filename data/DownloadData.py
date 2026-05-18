import os
from datasets import load_dataset

def download_and_save_as_parquet(save_path="dapomath_7000.parquet", sample_size=7000):
    print("🚀 正在从 HuggingFace 获取 DapoMath...")
    
    try:
        # 1. 加载数据集 (默认加载 train 分片)
        dataset = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
        print(f"✅ 原始数据集加载成功，总计: {len(dataset)} 条")
        
        # 2. 随机采样
        # 使用 dataset 自身的 shuffle 和 select 效率更高
        if len(dataset) > sample_size:
            sampled_dataset = dataset.shuffle(seed=42).select(range(sample_size))
            print(f"🎯 已随机抽取 {sample_size} 条题目")
        else:
            sampled_dataset = dataset
            print("⚠️ 数据集总数不足采样值，将导出全部数据")
            
        # 3. 转换为 Parquet 格式并保存
        # HuggingFace 的 Dataset 对象原生支持导出 parquet
        sampled_dataset.to_parquet(save_path)
        
        print("-" * 30)
        print(f"🎉 导出成功！")
        print(f"📂 文件路径: {os.path.abspath(save_path)}")
        print(f"📊 文件大小: {os.path.getsize(save_path) / (1024*1024):.2f} MB")

    except Exception as e:
        print(f"❌ 发生错误: {e}")

def generate_test_data(
    input_path=None,
    save_path="math_500.parquet",
    n=500,
    seed=42,
):
    if input_path is None:
        input_path = os.path.join(os.path.dirname(__file__), "math_test_fixed.parquet")
    # 本地 parquet：用 parquet 构造器 + data_files
    ds_dict = load_dataset("parquet", data_files=input_path)
    dataset = ds_dict["train"]
    if len(dataset) < n:
        raise ValueError(f"文件只有 {len(dataset)} 条，不足 {n} 条")
    subset = dataset.shuffle(seed=seed).select(range(n))
    subset.to_parquet(save_path)
    print(f"已写入 {len(subset)} 条 -> {os.path.abspath(save_path)}")

if __name__ == "__main__":
    generate_test_data()