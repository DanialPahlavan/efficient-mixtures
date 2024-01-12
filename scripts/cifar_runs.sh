#!/bin/bash

# Run commands in parallel
python -m cifar_train --L_final 1 --n_A 2 --S 10 --estimator s2s --seed 123 --device 0 &
python -m cifar_train --L_final 1 --n_A 2 --S 10 --estimator s2s --seed 234 --device 0 &
python -m cifar_train --L_final 1 --n_A 2 --S 10 --estimator s2s --seed 345 --device 0 &
python -m cifar_train --L_final 1 --n_A 2  --S 20 --estimator s2s --seed 123 --device 0 &

python -m cifar_train --L_final 1 --n_A 2 --S 20 --estimator s2s --seed 234 --device 1 &
python -m cifar_train --L_final 1 --n_A 2 --S 20 --estimator s2s --seed 345 --device 1 &
python -m cifar_train --L_final 1 --n_A 2 --S 50 --estimator s2s --seed 123 --device 1 &
python -m cifar_train --L_final 1 --n_A 2  --S 50 --estimator s2s --seed 234 --device 1 &

# Wait for all background processes to finish
wait
