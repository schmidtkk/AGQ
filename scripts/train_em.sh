#!/bin/bash
# Training script for EM dataset with AGQ model
# Usage: bash scripts/train_em.sh

# First, prepare the data if not already done
if [ ! -f "datasets/EM/EM_train_inputs.h5" ]; then
    echo "Preparing EM dataset..."
    conda activate agq
    python scripts/prepare_em_data.py --input datasets/EM/em_volumes.h5 --output datasets/EM/ --mode split
fi

GPU_ID=0
NUM_GPUS=1

CUDA_VISIBLE_DEVICES=$GPU_ID python -u -m torch.distributed.run \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=6667 \
    projects/AGQ/main.py --distributed \
    --config-base projects/AGQ/configs/EM-Base.yaml \
    --config-file projects/AGQ/configs/AGQ-EM.yaml \
    SOLVER.SAMPLES_PER_BATCH 1 \
    SOLVER.ITERATION_TOTAL 200000 \
    MODEL.INFERENCE_WITHOUT_BG False \
    DATASET.OUTPUT_PATH outputs/EM/AGQ
