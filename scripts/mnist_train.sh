#!/bin/bash

DEVICE=0
L_FINAL=2


SEEDS=(4 175)
ESTIMATORS=("s2s" "s2a")

for S in 1 2 3 4; do
    for n_A in $(seq 1 $S); do
        for SEED in "${SEEDS[@]}"; do
            for ESTIMATOR in "${ESTIMATORS[@]}"; do
                python mnist_train.py  --S $S --device $DEVICE \
                --seed $SEED --L_final $L_FINAL\
                --n_A $n_A --estimator $ESTIMATOR &
            done
        done
    done
done

# Wait for all background processes to finish
wait
