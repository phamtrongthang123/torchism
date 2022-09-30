#!/bin/bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate env_mit/
python train.py --config configs/train/sample_cifar.yaml --gpus 0
