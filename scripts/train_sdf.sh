#!/bin/bash
# Training script for SDF prediction model on EM dataset
# Usage: bash scripts/train_sdf.sh

# First, prepare the data if not already done
if [ ! -f "datasets/EM/EM_train_inputs.h5" ]; then
    echo "Preparing EM dataset..."
    conda activate agq
    python scripts/prepare_em_data.py --input datasets/EM/em_volumes.h5 --output datasets/EM/ --mode split
fi

export CUDA_VISIBLE_DEVICES=0

python projects/AGQ/train_sdf.py \
    --config projects/AGQ/configs/SDF-EM.yaml \
    --output outputs/EM/SDF
