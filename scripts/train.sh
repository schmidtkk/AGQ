#!/bin/bash
GPU_ID=0,1
NUM_GPUS=2

CUDA_VISIBLE_DEVICES=$GPU_ID python -u -m torch.distributed.run \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=6666 \
    projects/AGQ/main.py --distributed \
    --config-base projects/AGQ/configs/SNEMI-Base.yaml \
    --config-file projects/AGQ/configs/AGQ.yaml \
    SOLVER.SAMPLES_PER_BATCH 8 \
    SOLVER.ITERATION_TOTAL 200000 \
    MODEL.INFERENCE_WITHOUT_BG False \
    DATASET.OUTPUT_PATH outputs/train
