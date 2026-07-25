#!/bin/bash

datasets=("pavia" "houston18" "honghu")

for samples in 50 100 200 500; do
    for window in 3 5 7 11; do

        save_dir="results_sensitivity/win${window}_sam${samples}"
        mkdir -p "${save_dir}"

        for dataset in "${datasets[@]}"; do

            echo "Dataset=${dataset}, Window=${window}, Samples=${samples}"

            CUDA_VISIBLE_DEVICES=0 python main_for_jstars_rebuttal.py \
                --model sslt_d1 \
                --train_sample ${samples} \
                --epochs 50 \
                --num_runs 3 \
                --window_size ${window} \
                --save_path "${save_dir}/" \
                --dataset ${dataset}

        done
    done
done