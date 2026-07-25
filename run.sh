#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=1

SAVE_PATH="./results_avg_rebuttal"
NUM_RUNS=10
EPOCHS=50
MODELS=("sslt_d1" "sslt_se" "sslt_eca")
DATASETS=("hanchuan")

mkdir -p "$SAVE_PATH"

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

        python main_for_jstars_rebuttal.py \
            --model "$model" \
            --dataset "$dataset" \
            --save_path "$SAVE_PATH" \
            --num_runs "$NUM_RUNS" \
            --epochs "$EPOCHS" \
            2>&1 | tee "${SAVE_PATH}/log_${model}_${dataset}.txt"

        # Wait for GPU context to fully release before next experiment
        sleep 10
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader \
            | grep -q '.' && echo "[WARN] GPU still has active processes" || echo "[OK] GPU clear"

    done
done

echo ""
echo "All runs complete. Results saved in: $SAVE_PATH"