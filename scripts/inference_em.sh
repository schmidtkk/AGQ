#!/bin/bash
# Inference script for EM dataset with AGQ model
# Usage: bash scripts/inference_em.sh

export CUDA_VISIBLE_DEVICES=0

# EM dataset paths (use validation set for inference)
IMAGE_FILE=datasets/EM/EM_val_inputs.h5
LABEL_FILE=datasets/EM/EM_val_labels.h5

OUTPUT_DIR=outputs/EM/inference
PKL_PATH=${OUTPUT_DIR}/pkl_results
CKPT_FILE=./checkpoints/em_checkpoint_200000.pth.tar  # Update with your trained checkpoint


if [ ! -d ${OUTPUT_DIR} ]; then
    mkdir -p ${OUTPUT_DIR}
fi


# ------------ Blockwise inference and evaluation ------------ #
python scripts/inference_mp.py \
    --config-base projects/AGQ/configs/EM-Base.yaml \
    --config-file projects/AGQ/configs/AGQ-EM.yaml \
    --inference \
    --checkpoint ${CKPT_FILE} \
    SYSTEM.PARALLEL None \
    INFERENCE.STRIDE '[16,256,256]' \
    INFERENCE.IMAGE_NAME ${IMAGE_FILE} \
    INFERENCE.LABEL_PATH ${LABEL_FILE} \
    INFERENCE.OUTPUT_PATH ${PKL_PATH} \
    MODEL.INIT_MASK_METHOD connected_components \
    MODEL.INFERENCE_WITHOUT_BG True 


# ------------ Full volume concat, merge and evaluation ------------ #
python scripts/concat_merge_eval.py \
    --pkl_path ${PKL_PATH} \
    --image_path ${IMAGE_FILE} \
    --label_path ${LABEL_FILE} \
    --out_path ${OUTPUT_DIR}
