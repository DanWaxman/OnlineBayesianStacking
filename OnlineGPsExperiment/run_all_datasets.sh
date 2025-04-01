#!/bin/bash

# List of datasets
datasets=(
    "Friedman #1"
    "Friedman #2"
    "Elevators"
    "SARCOS"
    "Kuka #1"
    "CaData"
    "CPU Small"
)

# Random seed
for random_seed in {2..9}; do
    # Iterate over datasets and run the Python script
    for dataset in "${datasets[@]}"; do
        echo "Running for dataset: $dataset with random seed: $random_seed"
        /usr/local/bin/python3 main_save_pll_models.py -r "$random_seed" -d "$dataset"
    done
done
