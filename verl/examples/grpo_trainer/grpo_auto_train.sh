#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "Start 6 GRPO experiments on DAPO-math 7K."

QWEN3_MODEL_PATH="${QWEN3_MODEL_PATH:-${PROJECT_ROOT}/models/Qwen3-4B-Instruct-2507}"
QWEN25_MODEL_PATH="${QWEN25_MODEL_PATH:-${PROJECT_ROOT}/models/Qwen2.5-7B-Instruct}"

CUDA_VISIBLE_DEVICES=4,5 MODEL_PATH="$QWEN3_MODEL_PATH" TEMPERATURE=1.0 USE_KL_LOSS=False KL_LOSS_COEF=0.001 TRAINER_NAME="qwen3_4b_math500_t1_no_kl" bash "run_qwen2_5_7b_grpo.sh"
MODEL_PATH="$QWEN3_MODEL_PATH" TEMPERATURE=1.0 USE_KL_LOSS=True KL_LOSS_COEF=0.001 TRAINER_NAME="qwen3_4b_math500_t1_kl_beta0_001" bash "run_qwen2_5_7b_grpo.sh"
CUDA_VISIBLE_DEVICES=2,3 MODEL_PATH="$QWEN3_MODEL_PATH" TEMPERATURE=1.0 USE_KL_LOSS=True KL_LOSS_COEF=0.01 TRAINER_NAME="qwen3_4b_math500_t1_kl_beta0_01" bash "run_qwen2_5_7b_grpo.sh"

MODEL_PATH="$QWEN25_MODEL_PATH" TEMPERATURE=1.0 USE_KL_LOSS=False KL_LOSS_COEF=0.001 TRAINER_NAME="qwen2_5_7b_math500_t1_no_kl" bash "run_qwen2_5_7b_grpo.sh"
MODEL_PATH="$QWEN25_MODEL_PATH" TEMPERATURE=1.0 USE_KL_LOSS=True KL_LOSS_COEF=0.001 TRAINER_NAME="qwen2_5_7b_math500_t1_kl_beta0_001" bash "run_qwen2_5_7b_grpo.sh"
MODEL_PATH="$QWEN25_MODEL_PATH" TEMPERATURE=1.0 USE_KL_LOSS=True KL_LOSS_COEF=0.01 TRAINER_NAME="qwen2_5_7b_math500_t1_kl_beta0_01" bash "run_qwen2_5_7b_grpo.sh"

echo "All 6 GRPO experiments finished."
