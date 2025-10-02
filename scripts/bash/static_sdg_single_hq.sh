#!/bin/bash
# High-Quality Single Trajectory SDG Script
# Use this to test quality on one trajectory before running full multi-trajectory generation

CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=1 cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py \
    --checkpoint_dir checkpoints \
    --num_gpus 1 \
    --input_image_path assets/demo/static/diffusion_input/images/00172.png \
    --video_save_folder assets/demo/static/diffusion_output_single_test \
    --foreground_masking \
    --trajectory left \
    --movement_distance 0.25 \
    --camera_rotation center_facing \
    --num_steps 50 \
    --guidance 7.5 \
    --filter_points_threshold 0.03 \
    --noise_aug_strength 0.0 \
    --seed 42
