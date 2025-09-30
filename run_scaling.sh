#!/bin/bash

set -e
export CUDA_VISIBLE_DEVICES=0

##################################

python train_sv.py \
    --model_name "llama3.1-8b" \
    --target_heads "L0H3,L10H19,L10H25,L13H18,L13H25,L13H30,L13H8,L14H14,L16H30,L19H11,L7H14" \
    --dataset_path "jailbreak_classification/classification_llama3.1.csv" \
    --learning_rate 5e-2 \
    --epochs 3 \
    --output_dir "scaling_library/asguard_checkpoints"

##################################

python analysis_heads.py \
    --sv_path "scaling_library/asguard_checkpoints" \
    --target_heads "L0H3,L10H19,L10H25,L13H18,L13H25,L13H30,L13H8,L14H14,L16H30,L19H11,L7H14" \
    --dataset_path "jailbreak_classification/classification_llama3.1.csv"

##################################

