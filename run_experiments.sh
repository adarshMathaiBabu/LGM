#!/bin/bash

# Train the four Fashion-MNIST GAN configs in order.
# On Colab, mount Drive first (see the comment at the bottom), then run:
#   bash run_experiments.sh
# Finished logs, checkpoints, configs, and the GAN code are copied to Drive
# after every run. Override the destination with:
#   DRIVE_DIR="/content/drive/MyDrive/some/other/folder" bash run_experiments.sh

set -euo pipefail

cd "$(dirname "$0")"

CONFIGS=(
    "config/gan/fashion.yaml"
    "config/gan/fashion_lsgan.yaml"
    "config/gan/fashion_featmatch.yaml"
    "config/gan/fashion_full.yaml"
)

DRIVE_DIR="${DRIVE_DIR:-/content/drive/MyDrive/lgm2026_gan}"

save_to_drive() {
    if [ ! -d "/content/drive/MyDrive" ]; then
        echo "Google Drive is not mounted. Skipping save."
        echo "Mount it with: from google.colab import drive; drive.mount('/content/drive')"
        return 0
    fi

    mkdir -p "$DRIVE_DIR/lightning_logs" "$DRIVE_DIR/config" "$DRIVE_DIR/lgm/models"

    if [ -d "lightning_logs/gan" ]; then
        cp -a lightning_logs/gan "$DRIVE_DIR/lightning_logs/"
    fi
    cp -a config/gan "$DRIVE_DIR/config/"
    cp -a lgm/models/gan "$DRIVE_DIR/lgm/models/"
    cp -a run_experiments.sh "$DRIVE_DIR/"
    echo "Saved logs and code to $DRIVE_DIR"
}

for config in "${CONFIGS[@]}"; do
    echo "============================================================"
    echo "Running $config"
    echo "============================================================"
    python3 train.py fit --config "$config"
    save_to_drive
done

echo "All four runs finished."

# Colab cells to run before this script:
#
# from google.colab import drive
# drive.mount("/content/drive")
#
# %cd /content/lgm2026
# !bash run_experiments.sh
