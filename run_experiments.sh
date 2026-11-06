#!/bin/bash

# this is an EXAMPLE SCRIPT!!
# adapt to your purpose!

# this crashes the script in case any command doesn't work
set -e

# remove this stuff
echo "ERROR!"
echo "Please adapt this script to your desired setup before running it! Exiting..."
exit 1

CONFIG="config/vae/fashion.yaml"

# example to iterate over beta values for VAE
for beta in 0.25 0.5 1 2 4 8 16
do
    echo "Running experiment with beta=${beta}..."
    python3 train.py fit \
        --config "$CONFIG" \
        --model.init_args.beta "$beta" \
        --trainer.logger.init_args.name "experiment_beta_${beta}"
done
