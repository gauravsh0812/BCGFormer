#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=4

SAVE_PATH="./results_avg"
NUM_RUNS=5
MODELS=("sslt_d1" "spectralmamba" "spectralformer" "swinhsi" "hit" "hybridsn")
DATASETS=("houston" "pavia" "salinas" "indiana")

mkdir -p "$SAVE_PATH"

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo ""
        echo "========================================"
        echo "Running: model=$model  dataset=$dataset"
        echo "========================================"
        python main.py --model "$model" --dataset "$dataset" --save_path "$SAVE_PATH" --num_runs "$NUM_RUNS" \
            2>&1 | tee "${SAVE_PATH}/log_${model}_${dataset}.txt"
    done
done

echo ""
echo "All runs complete. Results saved in: $SAVE_PATH"
