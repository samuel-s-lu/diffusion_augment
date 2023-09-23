#!/bin/bash

#SBATCH -N 1
#SBATCH -J bbox
#SBATCH -o "bbox.txt"
#SBATCH --time=20:00:00
#SBATCH --mem=32G
#SBATCH -c 6
#SBATCH --gres=gpu:v100:1

cd ~/diffusion_augment/
zsh
python get_bbox.py --caches '20a,20o,22a'