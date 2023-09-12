#!/bin/bash

python diffusion_augment_seg.py \
    --max_elevation_angle 75 \
    --cache 17 \
    --experiment_name 'a1' \
    --use_sam_hq \
    --control_type 'canny' \
    --augment