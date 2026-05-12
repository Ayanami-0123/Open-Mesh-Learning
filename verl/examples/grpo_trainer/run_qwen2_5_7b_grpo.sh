set -x
export WANDB_API_KEY="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"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export HYDRA_FULL_ERROR=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export RAY_DISABLE_GPU_METRICS=1
export RAY_DISABLE_DASHBOARD=1
export RAY_DEDUP_LOGS=1
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=WARN
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export HSA_NO_SCRATCH_RECLAIM=1
export PYTHONPATH="${SCRIPT_DIR}/ray_ppu_sitecustomize:${PYTHONPATH:-}"
export TMPDIR="${RUNTIME_TMPDIR:-/tmp/xts-runtime-tmp}"
mkdir -p "$TMPDIR"

# 1. 修正变量定义区（删除了错误的 cat 管道命令）
PROFILE_STEPS="[2,4]"
PROFILE_RANKS_ALL=False
DISCRETE=True
PROFILE_RANKS="[1,2]"

# 2. 数据路径
SAVE_PATH="${SAVE_PATH:-${PROJECT_ROOT}/profile_data}" # profiler 保存路径
TRAIN_PATH="${TRAIN_PATH:-${PROJECT_ROOT}/data/math_train_fixed.parquet}" # 训练数据路径
TEST_PATH="${TEST_PATH:-${PROJECT_ROOT}/data/MATH-500_fixed.parquet}" # 测试数据路径
TRAIN_FILES="${TRAIN_FILES:-$TRAIN_PATH}"
VAL_FILES="${VAL_FILES:-$TEST_PATH}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/xts/models/Qwen2.5-7B-Instruct}" # 模型路径
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/verl_outputs}" # 断点保存路径
HYDRA_OUTPUT_DIR="${HYDRA_OUTPUT_DIR:-${PROJECT_ROOT}/hydra_outputs/${TRAINER_NAME:-manual_run}}"
mkdir -p "$SAVE_PATH" "$OUTPUT_PATH" "$HYDRA_OUTPUT_DIR"

LEVEL="level1"
CONTENTS="['npu','cpu']"
ANALYSIS=True

# 3. Batch Size 逻辑修正
# 对于 GRPO，通常 train_batch_size 表示每一轮从数据集中取出的 Prompt 数量
# 而 rollout.n (16) 是每个 Prompt 生成的数量
# 关键!!! ppo_mini_batch_size 在 GRPO 模式下必须等于 train_batch_size
train_batch_size="${TRAIN_BATCH_SIZE:-32}"
ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-$train_batch_size}"
ppo_micro_batch_size="${PPO_MICRO_BATCH_SIZE:-8}"
ROLLOUT_N="${ROLLOUT_N:-16}"
TEMPERATURE="${TEMPERATURE:-1.0}"
USE_KL_LOSS="${USE_KL_LOSS:-False}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
COMPUTE_MEI_METRIC="${COMPUTE_MEI_METRIC:-True}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}"
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-8192}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
SAVE_FREQ="${SAVE_FREQ:-100}"
TEST_FREQ="${TEST_FREQ:-20}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"

# 4. 设备映射（确保 NPU/GPU 数量匹配）
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"

# 5. 项目名称
PROJECT_NAME="verl_grpo_math_xts"
TRAINER_NAME="${TRAINER_NAME:-qwen2_5_7b_dapomath_grpo_vllm_t${TEMPERATURE}_kl${USE_KL_LOSS}_beta${KL_LOSS_COEF}}"

cd "$PROJECT_ROOT/verl" || exit 1

python -m verl.trainer.main_ppo \
    hydra.run.dir=$HYDRA_OUTPUT_DIR \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    ${VAL_BATCH_SIZE:+data.val_batch_size=$VAL_BATCH_SIZE} \
    ${VAL_MAX_SAMPLES:+data.val_max_samples=$VAL_MAX_SAMPLES} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +model.torch_dtype=bfloat16 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.param=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer=bfloat16 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.compute_mei_metric=$COMPUTE_MEI_METRIC \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ppo_micro_batch_size \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$LOG_PROB_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$ppo_micro_batch_size \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
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
    trainer.total_epochs=$TOTAL_EPOCHS \
    ${TOTAL_TRAINING_STEPS:+trainer.total_training_steps=$TOTAL_TRAINING_STEPS} \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.device=cuda \
    +ray_kwargs.ray_init.include_dashboard=False \
    reward_model.strategy=naive \
    +reward_model.num_examine=5 \
    global_profiler.tool=None \
    global_profiler.steps=$PROFILE_STEPS \
    global_profiler.save_path=$SAVE_PATH
