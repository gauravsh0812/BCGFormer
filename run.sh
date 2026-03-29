#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=1

SAVE_PATH="./results_avg"
MODELS=("sslt_d1" "spectralmamba" "spectralformer" "swinhsi" "hit")
DATASETS=("houston") # "pavia" "salinas")

mkdir -p "$SAVE_PATH"

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo ""
        echo "========================================"
        echo "Running: model=$model  dataset=$dataset"
        echo "========================================"
        python main.py --model "$model" --dataset "$dataset" --save_path "$SAVE_PATH" \
            2>&1 | tee "${SAVE_PATH}/log_${model}_${dataset}.txt"
    done
done

echo ""
echo "All runs complete. Results saved in: $SAVE_PATH"
