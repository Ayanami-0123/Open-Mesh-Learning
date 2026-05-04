set -x

# 1. 修正变量定义区（删除了错误的 cat 管道命令）
PROFILE_STEPS="[2,4]"
PROFILE_RANKS_ALL=False
DISCRETE=True
PROFILE_RANKS="[1,2]"
SAVE_PATH="/workspace/profile_data"
LEVEL="level1"
CONTENTS="['npu','cpu']"
ANALYSIS=True

# 2. Batch Size 逻辑修正
# 对于 GRPO，通常 train_batch_size 表示每一轮从数据集中取出的 Prompt 数量
# 而 rollout.n (16) 是每个 Prompt 生成的数量
# 关键：ppo_mini_batch_size 在 GRPO 模式下必须等于 train_batch_size
train_batch_size=32
ppo_mini_batch_size=32 
ppo_micro_batch_size=2

# 3. 设备映射（确保 NPU/GPU 数量匹配）
export CUDA_VISIBLE_DEVICES=1,2

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/workspace/data/dapomath_7000_final_aligned.parquet \
    data.val_files=/workspace/data/dapomath_7000_final_aligned.parquet \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +model.torch_dtype=bfloat16 \
    actor_rollout_ref.model.path=/workspace/Qwen2.5-7B \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.param=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer=bfloat16 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ppo_micro_batch_size \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$ppo_micro_batch_size \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.logger=wandb \
    trainer.project_name='verl_grpo_math' \
    trainer.experiment_name='qwen2_5_7b_grpo' \
    trainer.default_local_dir=/tmp/verl_outputs \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.device=cuda \
    reward_model.strategy=naive \
    +reward_model.num_examine=5 \
    global_profiler.tool=None \
    global_profiler.steps=$PROFILE_STEPS \
    global_profiler.save_path=$SAVE_PATH