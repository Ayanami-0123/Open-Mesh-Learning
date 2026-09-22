# Anonymous Code Release

This repository contains the code and data artifacts for an anonymous paper submission. The repository is organized to support reproducibility while avoiding author-identifying information.

## Repository Structure

- `verl/`: Training and evaluation code based on the verl framework, including the experiment launch scripts used in the submission.
- `data/`: Dataset files used by the experiments. Python preprocessing scripts have been removed from the public artifact to avoid exposing private credentials.
- `LICENSE`: License information for the released code.

## Environment Setup

Create a Python environment and install the project dependencies from the `verl/` directory:

```bash
cd verl
pip install -e .
pip install -r requirements.txt
```

Additional accelerator-specific dependencies may be required depending on the target hardware and backend.

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
