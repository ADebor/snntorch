#!/bin/bash

# Define an array of Hydra overrides
overrides=(
    "logger.backend=null hypers.n_epochs=1 task=mnist"
    "logger.backend=null hypers.n_epochs=1 task=silence"
)

# Loop through each override and launch the script
script_dir="$(dirname "$(realpath "$0")")"
parent_dir="$(dirname "$script_dir")"
for override in "${overrides[@]}"; do
    python "$parent_dir/mnist_train.py" ${override}
done