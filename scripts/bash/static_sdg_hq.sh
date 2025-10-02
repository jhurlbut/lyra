#!/bin/bash
# High-Quality Static Scene Data Generation (SDG) Script
# Optimized for better Gaussian splatting quality with reduced artifacts

CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=1 cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py \
    --checkpoint_dir checkpoints \
    --num_gpus 1 \
    --input_image_path assets/demo/static/diffusion_input/images/00172.png \
    --video_save_folder assets/demo/static/diffusion_output_generated_hq \
    --foreground_masking \
    --multi_trajectory \
    --num_steps 50 \
    --guidance 7.5 \
    --filter_points_threshold 0.03 \
    --noise_aug_strength 0.0 \
    --total_movement_distance_factor 1.2 \
    --seed 42
