#!/bin/bash

#SBATCH -N 1
#SBATCH -J c22a1
#SBATCH -o "c22a1.txt"
#SBATCH --time=20:00:00
#SBATCH --mem=32G
#SBATCH -c 6
#SBATCH --gres=gpu:v100:1

cd ~/diffusion_augment/
zsh
python diffusion_augment_seg.py \
    --max_elevation_angle 75 \
    --cache 22 \
    --experiment_name 'a1' \
    --use_sam_hq \
    --control_type 'canny' \
    --augment