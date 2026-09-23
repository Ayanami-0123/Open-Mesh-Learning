# Anonymous Code Release

This repository contains the code and data artifacts for an anonymous paper submission. The repository is organized to support reproducibility while avoiding author-identifying information.

## Repository Structure

- `verl/`: Training and evaluation code based on the verl framework, including the experiment launch scripts used in the submission.
- `data/`: Dataset files used by the experiments. Python preprocessing scripts have been removed from the public artifact to avoid exposing private information, the system prompt used to generate coach prompts are provided in Appendix D.2 of the essay.
- `LICENSE`: License information for the released code.

## Environment Setup

The recommended setup uses `uv` so that dependency resolution and installation are reproducible across machines. Run the commands from the repository root.

### 1. Create and activate an environment

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
```

### 2. Install the training package

```bash
uv pip install --upgrade pip setuptools wheel
uv pip install -e ./verl
uv pip install -r verl/requirements.txt
```

### 3. Install backend-specific requirements

Install only the backend needed for your machine. For CUDA-based experiments:

```bash
uv pip install -r verl/requirements-cuda.txt
uv pip install -r verl/requirements_sglang.txt
```

For NPU-based experiments:

```bash
uv pip install -r verl/requirements-npu.txt
```

### 4. Fetch data files tracked with Git LFS

Some dataset files are stored with Git LFS. After cloning the repository, fetch them with:

```bash
git lfs install
git lfs pull --include="data/*.parquet"
```

### 5. Sanity check the installation

```bash
python -c "import verl; print('verl import ok')"
```

Additional accelerator-specific dependencies may be required depending on the target hardware, driver version, and rollout backend.

## Reproducibility Notes

### Commands
The three fundamental launch scripts are provided under `verl/examples/`:

- `verl/examples/dapo_trainer/run_qwen2_5_7b_dapo.sh`
- `verl/examples/gspo_trainer/run_qwen2_5_7b_gspo.sh`
- `verl/examples/grpo_trainer/run_qwen2_5_7b_grpo.sh`

Before running an experiment, review the script and set local paths, cluster settings, logging configuration, and any required environment variables outside the repository. The general command template is provided below:

```bash
MODEL_PATH=/path/to/model \
TRAIN_PATH=data/<train_file>.parquet \
TEST_PATH=data/<eval_file>.parquet \
TEST_PATH_2=data/<optional_second_eval_file>.parquet \
PROJECT_NAME=<project_name> \
TRAINER_NAME=<run_name> \
N_GPUS_PER_NODE=<num_gpus> \
TOTAL_EPOCHS=<epochs> \
MAX_RESPONSE_LENGTH=<max_response_length> \
USE_KL_LOSS=<True_or_False> \
DIVERGENCE_TYPE=<kl_or_js> \
ROLLOUT_N=<num_rollouts_per_prompt> \
VAL_N=<num_validation_samples> \
UNIFORM_GROUP_KL_ENABLE=<True_or_False> \
UNIFORM_GROUP_KL_GROUP_SIZE=<num_methods_to_rebalance> \
bash verl/examples/<method>_trainer/<one_of_the_three_scripts>.sh
```

The main variables are:

| Variable | Meaning |
| --- | --- |
| `MODEL_PATH` | Local path to the base model checkpoint. |
| `TRAIN_PATH` | Training dataset in parquet format. |
| `TEST_PATH` | Primary evaluation dataset. |
| `TEST_PATH_2` | Optional secondary evaluation dataset. |
| `PROJECT_NAME` | Logger project name. |
| `TRAINER_NAME` | Run name used for logs/checkpoints. |
| `N_GPUS_PER_NODE` | Number of GPUs used on one node. |
| `TOTAL_EPOCHS` | Number of training epochs. |
| `TOTAL_TRAINING_STEPS` | Optional cap on training steps. |
| `MAX_RESPONSE_LENGTH` | Token budget for each rollout response. |
| `USE_KL_LOSS` | Whether to turn on actor KL loss; use `True` or `False`. |
| `DIVERGENCE_TYPE` | Divergence type used by the actor/reference penalty; use `kl` or `js`. |
| `ROLLOUT_N` | Number of rollout responses sampled per prompt during training. |
| `VAL_N` | Number of sampled responses per validation prompt. |
| `UNIFORM_GROUP_KL_ENABLE` | Whether to enable the uniform-group KL rebalancing auxiliary loss; use `True` or `False`. |
| `UNIFORM_GROUP_KL_GROUP_SIZE` | Number of same-prompt methods/responses randomly sampled without replacement by uniform-group KL. The default `4` matches the original setting; set `3`, `2`, or `1` to rebalance a random subset of fewer methods from each prompt uid. |

### Model
The three models used in the experiments are:
- `Qwen/Qwen3-4B`
- `Qwen/Qwen2.5-7B-Instruct`
- `unsloth/Phi-4-mini-reasoning`

All three models can be downloaded on `modelscope`, using the command below:
```bash
modelscope download --model <model_family>/<model_name> --local_dir path/to/your/model/storage
```

### Data

The main training and evaluation files are stored under `data/`. The expected files for different benchmarks include:

### Benchmark name: MATH-500
- Benchmark `MATH-500_fixed.parquet`
- training set (No Coach prompt) `math_train_fixed.parquet`
- training set (With Coach prompt) `math_final_aligned2.parquet`

### Benchmark name: AIME25/26
- Benchmark `AIME25_fixed.parquet` `AIME26_fixed.parquet`
- training set (No Coach prompt) `dapomath_7000_not_enhanced.parquet`
- training set (With Coach prompt) `dapomath_7000_final_aligned.parquet`

### Benchmark name: GPQA
- Benchmark `GPQA_diamond.parquet`
- training set (No Coach prompt) `WildSci_not_enhanced.parquet`
- training set (With Coach prompt) `WildSci_final_aligned_3.parquet`

### Benchmark name: LiveCodeBench
- Benchmark `livecodebench.parquet`
- training set (No Coach prompt) `Skywork_not_enhanced.parquet`
- training set (With Coach prompt) `Skywork_final_aligned2.parquet`

Update paths in the launch scripts if the data is stored in a different location on your system.

### Prompt Length
| Benchmark | Phi-4-mini-reasoning | Qwen2.5-7B-Instruct | Qwen3-4B |
| --- | --- | --- | --- |
| AIME25 | 8192 | 8192 | 1024 |
| AIME26 | 8192 | 8192 | 1024 |
| GPQA | 4096 | 4096 | 1024 |
| MATH-500 | 4096 | 4096 | 1024 |
| LiveCodeBench | 4096 | 4096 | \ |

### Ablation Experiments
You can turn off `UNIFORM_GROUP_KL_ENABLE` and use training sets with Coach prompts to ablate the effectiveness of Strategy-balancing regularization.

You can change `UNIFORM_GROUP_KL_GROUP_SIZE` to ablate the sensitivity to selected strategy capacity.

### Command example
The main Qwen3-4B AIME26 experiment can be reproduced with:

```bash
MODEL_PATH=/path/to/Qwen3-4B \
TRAIN_PATH=data/dapomath_7000_final_aligned.parquet \
TEST_PATH=data/AIME26_fixed.parquet \
N_GPUS_PER_NODE=... \
TOTAL_EPOCHS=... \
MAX_RESPONSE_LENGTH=8192 \
ROLLOUT_N=4 \
VAL_N=4 \
UNIFORM_GROUP_KL_ENABLE=True \
UNIFORM_GROUP_KL_GROUP_SIZE=4 \
bash verl/examples/grpo_trainer/run_qwen2_5_7b_grpo.sh
```
`ROLLOUT_N` is set 4 here as we split the 16 rollouts on 4 prompts with identical query but different system prompt.

### Reproducibility Disclaimer

- Large generated outputs, checkpoints, local logs, and Hydra output directories are intentionally excluded from the repository.
- wandb API keys and other service credentials should be prepared by the users.
- Exact hardware throughput and runtime may vary with GPU type, driver version, backend configuration, and cluster scheduling.
- Due to the stochasticity of RL training and inference, rerunning an experiment may not reproduce the reported metric exactly. The provided scripts and configuration can be used to reproduce the primary results reported.

## Anonymity

This artifact is prepared for anonymous review. Please do not infer authorship from repository metadata, local paths, or external service configuration.
