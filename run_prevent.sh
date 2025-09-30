#!/bin/bash

set -e
export CUDA_VISIBLE_DEVICES=0 

export MODEL_NAME="llama3.1-8b"
# export CACHE_DIR=".cache/huggingface/hub"
export DATA_PATH="jailbreak_classification/classification_llama3.1.csv"
export SV_PATH="scaling_library/asguard_checkpoints_llama3.1/asguard_scales_llama3.1-8b.pt"
export TARGET_HEADS="L0H3,L10H19,L10H25,L13H18,L13H25,L13H30,L13H8,L14H14,L16H30,L19H11,L7H14"
export OUTPUT_DIR="prevent_output/${MODEL_NAME}_prevent"

mkdir -p $OUTPUT_DIR

############### FSDP ################
echo "Starting Preventative Steering Finetuning for model: $MODEL_NAME..."

accelerate launch --num_processes 1 train_pft.py \
    --model_name $MODEL_NAME \
    --cache_dir "$CACHE_DIR" \
    --target_heads "$TARGET_HEADS" \
    --data_path "$DATA_PATH" \
    --sv_path "$SV_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --seed 42 \
    --lr 9e-06 \
    --epochs 1 \
    --batch_size 1 \
    --grad_accum_steps 32

echo "Training finished. Final model saved to $OUTPUT_DIR"

##################################