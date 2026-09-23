#!/bin/bash
set -euo pipefail

: "${WANDB_API_KEY:?Set WANDB_API_KEY in the environment before running this script}"
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export HYDRA_FULL_ERROR=1

RUNTIME_TMPDIR="${RUNTIME_TMPDIR:-${PROJECT_ROOT}/verl/tmp}"
export TMPDIR="$RUNTIME_TMPDIR"
export TMP="$RUNTIME_TMPDIR"
export TEMP="$RUNTIME_TMPDIR"
mkdir -p "$TMPDIR"

# Ray 的 session / object spilling 目录可按机器情况覆盖。
export RAY_TMPDIR="${RAY_TMPDIR:-${PROJECT_ROOT}/ray_tmp}"
mkdir -p "$RAY_TMPDIR"

# 1. Profiler 变量
PROFILE_STEPS="[2,4]"
PROFILE_RANKS_ALL=False
DISCRETE=True
PROFILE_RANKS="[1,2]"

# 2. 数据路径
SAVE_PATH="${SAVE_PATH:-${PROJECT_ROOT}/profile_data}" # profiler 保存路径
TRAIN_PATH="${TRAIN_PATH:-${PROJECT_ROOT}/data/math_final_aligned.parquet}" # 训练数据路径
TEST_PATH="${TEST_PATH:-${PROJECT_ROOT}/data/MATH-500_fixed.parquet}" # 测试数据路径
TRAIN_FILES="${TRAIN_FILES:-$TRAIN_PATH}"
VAL_FILES="${VAL_FILES:-$TEST_PATH}"
MODEL_PATH="${MODEL_PATH:-/workspace/Qwen2.5-7B}" # 模型路径

LEVEL="level1"
CONTENTS="['npu','cpu']"
ANALYSIS=True

# 3. Batch Size 逻辑
# 对于 Online DPO，train_batch_size 表示每一轮从数据集中取出的 Prompt 数量；
# rollout.n 是每个 Prompt 生成的样本数（在线构造 chosen / rejected 配对，必须 >= 2）。
# ppo_mini_batch_size 用于 DPO 更新阶段的分批。
train_batch_size="${TRAIN_BATCH_SIZE:-8}"
ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-$train_batch_size}"
ppo_micro_batch_size="${PPO_MICRO_BATCH_SIZE:-4}"
ROLLOUT_N="${ROLLOUT_N:-4}"
TEMPERATURE="${TEMPERATURE:-1.0}"

# DPO 专属超参
DPO_BETA="${DPO_BETA:-0.1}"
DPO_LOSS_TYPE="${DPO_LOSS_TYPE:-sigmoid}"
REF_UPDATE_FREQ="${REF_UPDATE_FREQ:-1}"   # 参考模型周期性更新频率（Online DPO）

# group_kl（uniform group KL 辅助损失）接口
UNIFORM_GROUP_KL_ENABLE="${UNIFORM_GROUP_KL_ENABLE:-True}"
UNIFORM_GROUP_KL_TEMPERATURE="${UNIFORM_GROUP_KL_TEMPERATURE:-1.0}"
UNIFORM_GROUP_KL_COEF="${UNIFORM_GROUP_KL_COEF:-0.001}"

# mei 指标接口
COMPUTE_MEI_METRIC="${COMPUTE_MEI_METRIC:-True}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}"
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-8192}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
SAVE_FREQ="${SAVE_FREQ:-100}"
TEST_FREQ="${TEST_FREQ:-5}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"

# 4. 设备映射（确保 NPU/GPU 数量匹配）
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"

# 5. 项目名称
PROJECT_NAME="verl_m_dpo_math"
TRAINER_NAME="${TRAINER_NAME:-qwen2_5_7b_dapomath_m_dpo_sglang_t${TEMPERATURE}_beta${DPO_BETA}}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/verl_outputs/${TRAINER_NAME}}" # 断点保存路径
HYDRA_OUTPUT_DIR="${HYDRA_OUTPUT_DIR:-${PROJECT_ROOT}/hydra_outputs/${TRAINER_NAME}}"
mkdir -p "$SAVE_PATH" "$OUTPUT_PATH" "$HYDRA_OUTPUT_DIR"

cd "$PROJECT_ROOT/verl" || exit 1

# 入口：recipe.spin.main_spin（verl 中的 Online DPO 实现）
# 配置默认为 recipe/spin/config/spin_trainer.yaml，它 defaults: - ppo_trainer，
# 因此 actor 的 uniform_group_kl_* / compute_mei_metric 接口可照常传入。
python -m recipe.spin.main_spin \
    hydra.run.dir=$HYDRA_OUTPUT_DIR \
    algorithm.adv_estimator=null \
    +algorithm.dpo_beta=$DPO_BETA \
    +algorithm.dpo_loss_type=$DPO_LOSS_TYPE \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    ${VAL_BATCH_SIZE:+data.val_batch_size=$VAL_BATCH_SIZE} \
    ${VAL_MAX_SAMPLES:+data.val_max_samples=$VAL_MAX_SAMPLES} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    +model.torch_dtype=bfloat16 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.param=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer=bfloat16 \
    actor_rollout_ref.actor.dpo_beta=$DPO_BETA \
    actor_rollout_ref.actor.uniform_group_kl_enable=$UNIFORM_GROUP_KL_ENABLE \
    actor_rollout_ref.actor.uniform_group_kl_temperature=$UNIFORM_GROUP_KL_TEMPERATURE \
    actor_rollout_ref.actor.uniform_group_kl_coef=$UNIFORM_GROUP_KL_COEF \
    +actor_rollout_ref.actor.compute_mei_metric=$COMPUTE_MEI_METRIC \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ppo_micro_batch_size \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.val_kwargs.n=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$LOG_PROB_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$ppo_micro_batch_size \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$TRAINER_NAME \
    trainer.default_local_dir=$OUTPUT_PATH \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.resume_mode=auto \
    trainer.test_freq=$TEST_FREQ \
    trainer.ref_update_freq=$REF_UPDATE_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    ${TOTAL_TRAINING_STEPS:+trainer.total_training_steps=$TOTAL_TRAINING_STEPS} \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.device=cuda \
    +ray_kwargs.ray_init.include_dashboard=False \
    reward_model.strategy=naive \
    +reward_model.num_examine=2 \
    global_profiler.tool=None \
    global_profiler.steps=$PROFILE_STEPS \
    global_profiler.save_path=$SAVE_PATH
