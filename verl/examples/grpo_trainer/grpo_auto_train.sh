#!/bin/bash

# 定义你要运行的那个测试脚本
TARGET_SCRIPT="run_qwen2_5_7b_grpo.sh"

# 计数器，方便看重启了多少次
COUNT=0

echo "开始执行监控循环..."

while true; do
    # 记录时间并启动
    echo "[$(date '+%H:%M:%S')] 第 $COUNT 次启动脚本: $TARGET_SCRIPT"
    
    # 运行你的测试脚本
    bash "$TARGET_SCRIPT"
    
    # 获取退出状态码
    # 0 代表正常跑完退出，非 0 代表崩了（包括 OOM）
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "脚本已正常运行结束，退出监控。"
        break
    else
        COUNT=$((COUNT + 1))
        echo "检测到异常退出 (Code: $EXIT_CODE)，1秒后尝试自动续传..."
        
        # 强制清理：有时候 OOM 之后显存不会立刻释放干净
        # 这一步是保险，防止重启后瞬间再次 OOM
        sleep 1
    fi
done