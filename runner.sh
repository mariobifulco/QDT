#!/bin/bash

DATASETS=("iris" "wine" "breast_cancer")
ITERATIONS=9
BATCH_START=1
BATCH_END=20

mkdir -p results

for ds in "${DATASETS[@]}"; do
    for i in $(seq 0 $ITERATIONS); do
        for batch in $(seq $BATCH_START $BATCH_END); do
            echo "------------------------------------------------"
            echo "Running: DS=$ds, Iter=$i, Batch=$batch"
            conda run -n qdt python main.py --dataset "$ds" --iteration "$i" --batch_size "$batch"
        done
    done
done

echo "Done"

# 36068