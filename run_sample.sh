#!/bin/bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate env_mit/
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzZjE4OGUwYS01NTg2LTRkMmMtOWQxMi1jZjZjYTlhMGRiMDUifQ=="
python train.py --config configs/train/sample.yaml --gpus 0
