#!/bin/bash

# --- 配置区 ---
TARGET_SCRIPT="run_qwen2_5_7b_grpo.sh"
INTERVAL=1  # 每秒撞针一次

# 检查函数：输入两个 GPU ID，检查是否都空闲（显存占用 < 100MiB）
check_pair_free() {
    local gpu_a=$1
    local gpu_b=$2
    
    usage_a=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_a | xargs)
    usage_b=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_b | xargs)

    if [ "$usage_a" -lt 100 ] && [ "$usage_b" -lt 100 ]; then
        return 0 # 成功，两张都空闲
    else
        return 1 # 失败
    fi
}

echo "🎯 撞针启动：正在监控 (0,3) 和 (1,2) 组合..."
echo "等待空闲中..."

while true; do
    # 尝试组合 0,3
    if check_pair_free 0 3; then
        echo -e "\n🔥 [$(date +%H:%M:%S)] 抢到 0,3 卡！正在启动任务..."
        export CUDA_VISIBLE_DEVICES=0,3
        bash $TARGET_SCRIPT
        break
    fi

    # 尝试组合 1,2
    if check_pair_free 1 2; then
        echo -e "\n🔥 [$(date +%H:%M:%S)] 抢到 1,2 卡！正在启动任务..."
        export CUDA_VISIBLE_DEVICES=1,2
        bash $TARGET_SCRIPT
        break
    fi

    # 终端打印进度，不换行
    echo -ne "\r检测中... 当前时间: $(date +%H:%M:%S)"
    sleep $INTERVAL
done