# Anonymous Code Release

This repository contains the code and data artifacts for an anonymous paper submission. The repository is organized to support reproducibility while avoiding author-identifying information.

## Repository Structure

- `verl/`: Training and evaluation code based on the verl framework, including the experiment launch scripts used in the submission.
- `data/`: Dataset files used by the experiments. Python preprocessing scripts have been removed from the public artifact to avoid exposing private credentials.
- `LICENSE`: License information for the released code.

## Environment Setup

The recommended setup uses `uv` so that dependency resolution and installation are reproducible across machines. Run the commands from the repository root.

### 1. Create and activate an environment

```bash
uv venv --python 3.10 .venv
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

## Data

The main training and evaluation files are stored under `data/`. The expected files include:

- `math_train_fixed.parquet`
- `math_test_fixed.parquet`

Update paths in the launch scripts if the data is stored in a different location on your system.

## Running Experiments

Example launch scripts are provided under `verl/examples/`:

- `verl/examples/dapo_trainer/run_qwen2_5_7b_dapo.sh`
- `verl/examples/gspo_trainer/run_qwen2_5_7b_gspo.sh`
- `verl/examples/grpo_trainer/run_qwen2_5_7b_grpo.sh`
- `verl/examples/grpo_trainer/run_qwen3_4b_grpo.sh`

Before running an experiment, review the script and set local paths, cluster settings, logging configuration, and any required environment variables outside the repository.

```bash
bash verl/examples/grpo_trainer/run_qwen3_4b_grpo.sh
```

## Reproducibility Notes

- Large generated outputs, checkpoints, local logs, and Hydra output directories are intentionally excluded from the repository.
- API keys and service credentials should be provided through local environment variables or a private secrets manager, not committed to version control.
- Exact hardware throughput and runtime may vary with GPU type, driver version, backend configuration, and cluster scheduling.

## Anonymity

This artifact is prepared for anonymous review. Please do not infer authorship from repository metadata, local paths, or external service configuration.
