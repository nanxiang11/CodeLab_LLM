#!/bin/bash

TRAIN_SCRIPT="SFTtrain_LORA.py"

# 自动检测 GPU 数量
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

if [ "$NUM_GPUS" -le 1 ]; then
    echo "单卡训练..."
    python $TRAIN_SCRIPT
else
    echo "多卡 DDP 训练..."
    torchrun --nproc_per_node=$NUM_GPUS $TRAIN_SCRIPT
fi
