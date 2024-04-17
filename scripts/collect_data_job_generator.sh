#!/bin/bash

# Array of seed values
seeds=(15 20 25 26 30 31 120 121 123 124 125 126 127) # Add or remove seed values as needed

for seed in "${seeds[@]}"; do
    sbatch scripts/collect_data.sh $seed
done
