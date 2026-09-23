set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [ -z "${PERSIST_ROOT:-}" ]; then
    if [ "$PROJECT_ROOT" = "/workspace" ] && [ -d /workspace/verl ]; then
        PERSIST_ROOT="${PROJECT_ROOT}/verl"
    else
        PERSIST_ROOT="$PROJECT_ROOT"
    fi
fi

export HYDRA_FULL_ERROR=1
# 注意：不能开 expandable_segments，它与 SGLang 的 TorchMemorySaver 冲突
# （TorchMemorySaver 负责 colocate 时释放推理显存，比碎片整理更重要）。
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 选用的 GPU 1 和 3 之间没有 NVLink（只有 PCIe/PIX），容器内 PCIe P2P 会导致 NCCL 卡死。
# 禁用 P2P，让 NCCL 走共享内存通信（略慢但不会 hang）。
# export NCCL_P2P_DISABLE=1
# 如需排查 NCCL 卡在哪一步，临时打开下面这行看详细日志：
# export NCCL_DEBUG=INFO

RUNTIME_TMPDIR="${RUNTIME_TMPDIR:-${PERSIST_ROOT}/tmp}"
RAY_TMPDIR="${RAY_TMPDIR:-${RUNTIME_TMPDIR}/ray}"
export TMPDIR="$RUNTIME_TMPDIR"
export TMP="$RUNTIME_TMPDIR"
export TEMP="$RUNTIME_TMPDIR"
mkdir -p "$TMPDIR" "$RAY_TMPDIR"

# 1. 修正变量定义区（删除了错误的 cat 管道命令）
PROFILE_STEPS="[2,4]"
PROFILE_RANKS_ALL=False
DISCRETE=True
PROFILE_RANKS="[1,2]"

# 2. 数据路径
SAVE_PATH="${SAVE_PATH:-${PERSIST_ROOT}/profile_data}" # profiler 保存路径
TRAIN_PATH="${TRAIN_PATH:-${PROJECT_ROOT}/data/dapomath_7000_final_aligned.parquet}" # 训练数据路径
TEST_PATH="${TEST_PATH:-${PROJECT_ROOT}/data/AIME26_fixed.parquet}" # 测试数据路径
TEST_PATH_2="${TEST_PATH_2:-${PROJECT_ROOT}/data/AIME25_fixed.parquet}" # 第二个测试数据路径，可选
TRAIN_FILES="${TRAIN_FILES:-$TRAIN_PATH}"
if [ -z "${VAL_FILES:-}" ]; then
    if [ -n "$TEST_PATH_2" ]; then
        VAL_FILES="['$TEST_PATH', '$TEST_PATH_2']"
    else
        VAL_FILES="$TEST_PATH"
    fi
fi
MODEL_PATH="${MODEL_PATH:-/workspace/models/Qwen/Qwen2.5-7B-Instruct}" # 模型路径

LEVEL="level1"
CONTENTS="['npu','cpu']"
ANALYSIS=True

# 3. Batch Size 逻辑修正
# 对于 GRPO，通常 train_batch_size 表示每一轮从数据集中取出的 Prompt 数量
# 而 rollout.n (16) 是每个 Prompt 生成的数量
# 关键!!! ppo_mini_batch_size 在 GRPO 模式下必须等于 train_batch_size
train_batch_size="${TRAIN_BATCH_SIZE:-128}"
ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-128}"
ppo_micro_batch_size="${PPO_MICRO_BATCH_SIZE:-4}"
ROLLOUT_N="${ROLLOUT_N:-4}"
TEMPERATURE="${TEMPERATURE:-1.0}"
UNIFORM_GROUP_KL_ENABLE="${UNIFORM_GROUP_KL_ENABLE:-False}"
UNIFORM_GROUP_KL_GROUP_SIZE="${UNIFORM_GROUP_KL_GROUP_SIZE:-4}"
USE_KL_LOSS="${USE_KL_LOSS:-False}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.01}"
DIVERGENCE_TYPE="${DIVERGENCE_TYPE:-js}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-3000}"
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-3000}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-20}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
# 验证阶段采样配置：用于计算 pass@k (verl 里叫 best@N)。
# 默认贪心单样本只能得到 pass@1；这里打开采样、每个 prompt 采 VAL_N 个，
# wandb 会自动出现 val-core/.../best@2,4,8,16/mean（即 pass@2/4/8/16）。
VAL_N="${VAL_N:-4}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-1.0}"
VAL_TOP_P="${VAL_TOP_P:-0.95}"

# 4. 设备映射（确保 NPU/GPU 数量匹配）
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"

# 5. 项目名称
PROJECT_NAME="AIME"
TRAINER_NAME="Qwen2.5-7B-GRPO-Null+CP-AIME"
OUTPUT_PATH="${OUTPUT_PATH:-${PERSIST_ROOT}/verl_outputs/${TRAINER_NAME}}" # 断点保存路径
HYDRA_OUTPUT_DIR="${HYDRA_OUTPUT_DIR:-${PERSIST_ROOT}/hydra_outputs/${TRAINER_NAME}}"
mkdir -p "$SAVE_PATH" "$OUTPUT_PATH" "$HYDRA_OUTPUT_DIR"

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
    data.shuffle=False \
    +model.torch_dtype=bfloat16 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.param=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer=bfloat16 \
    algorithm.divergence_type=$DIVERGENCE_TYPE \
    actor_rollout_ref.actor.divergence_type=$DIVERGENCE_TYPE \
    actor_rollout_ref.actor.uniform_group_kl_enable=$UNIFORM_GROUP_KL_ENABLE \
    actor_rollout_ref.actor.uniform_group_kl_temperature=1.0 \
    actor_rollout_ref.actor.uniform_group_kl_group_size=$UNIFORM_GROUP_KL_GROUP_SIZE \
    actor_rollout_ref.actor.uniform_group_kl_coef=0.00005 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    +actor_rollout_ref.actor.compute_mei_metric=True \
    actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.data_parallel_size=1 \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ppo_micro_batch_size \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.n=$VAL_N \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE \
    actor_rollout_ref.rollout.val_kwargs.top_p=$VAL_TOP_P \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$LOG_PROB_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$ppo_micro_batch_size \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.logger='["console","wandb"]' \
    trainer.log_val_generations=10 \
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
    +ray_kwargs.ray_init._temp_dir=$RAY_TMPDIR \
    +ray_kwargs.ray_init.runtime_env.env_vars.TMPDIR=$TMPDIR \
    +ray_kwargs.ray_init.runtime_env.env_vars.TMP=$TMPDIR \
    +ray_kwargs.ray_init.runtime_env.env_vars.TEMP=$TMPDIR \
    reward_model.strategy=naive \
    +reward_model.num_examine=2 \
    global_profiler.tool=None \
    global_profiler.steps=$PROFILE_STEPS \
    global_profiler.save_path=$SAVE_PATH
