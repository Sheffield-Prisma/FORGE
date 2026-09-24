#!/bin/bash
#SBATCH --job-name=shihuahuaco_seg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=82G
#SBATCH --time=24:00:00

# Edit the two variables below for the selected ablation or deployment split.
DATA_YAML=/path/to/Shihuahuaco_v2/data_abl_both.yaml
RUN_NAME=abl_both_v2

set -euo pipefail
source ~/.bashrc
conda activate ldm
nvidia-smi
python -m forge_yolo.train --data "$DATA_YAML" --name "$RUN_NAME" --device 0

