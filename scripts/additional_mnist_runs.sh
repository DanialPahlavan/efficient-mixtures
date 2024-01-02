#!/bin/bash

# Run commands in parallel
python -m mnist_train --L_final 1 --S 20 --estimator s2a --seed 333 --n_A 2 &
python -m mnist_train --L_final 1 --S 20 --estimator s2a --seed 444 --n_A 2 &
python -m mnist_train --L_final 1 --S 50 --estimator s2a --seed 333 --n_A 1 &
python -m mnist_train --L_final 1 --S 50 --estimator s2a --seed 444 --n_A 1 &
python -m mnist_train --L_final 1 --S 10 --estimator s2s --seed 333 --n_A 1 &
python -m mnist_train --L_final 1 --S 10 --estimator s2s --seed 444 --n_A 1 &
python -m mnist_train --L_final 1 --S 10 --estimator s2s --seed 333 --n_A 2 &
python -m mnist_train --L_final 1 --S 10 --estimator s2s --seed 444 --n_A 2 &
python -m mnist_train --L_final 1 --S 20 --estimator s2s --seed 333 --n_A 1 &
python -m mnist_train --L_final 1 --S 20 --estimator s2s --seed 444 --n_A 1 &
python -m mnist_train --L_final 1 --S 20 --estimator s2s --seed 333 --n_A 2 &
python -m mnist_train --L_final 1 --S 20 --estimator s2s --seed 444 --n_A 2 &
python -m mnist_train --L_final 1 --S 50 --estimator s2s --seed 333 --n_A 1 &
python -m mnist_train --L_final 1 --S 50 --estimator s2s --seed 444 --n_A 1 &
python -m mnist_train --L_final 1 --S 50 --estimator s2s --seed 333 --n_A 2 &
python -m mnist_train --L_final 1 --S 50 --estimator s2s --seed 444 --n_A 2 &

# Wait for all background processes to finish
wait
