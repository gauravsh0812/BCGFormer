#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=4

SAVE_PATH="./results_avg"
NUM_RUNS=5
MODELS=("sslt_d1" "spectralmamba" "swinhsi" "hit" "hybridsn") #"spectralformer" 
DATASETS=("salinas" "indiana") #"houston" "pavia" 

# mkdir -p "$SAVE_PATH"

TOTAL=$(( ${#MODELS[@]} * ${#DATASETS[@]} ))
COUNT=0

echo "Total runs: $TOTAL"

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do

        COUNT=$((COUNT + 1))

        # Progress bar
        PERCENT=$((COUNT * 100 / TOTAL))

        echo ""
        echo "[$COUNT/$TOTAL] ($PERCENT%) Running: model=$model | dataset=$dataset"
        echo "----------------------------------------"

        python main.py \
            --model "$model" \
            --dataset "$dataset" \
            --save_path "$SAVE_PATH" \
            --num_runs "$NUM_RUNS" \
            2>&1 | tee "${SAVE_PATH}/log_${model}_${dataset}.txt"

    done
done

echo ""
echo "All runs complete. Results saved in: $SAVE_PATH"