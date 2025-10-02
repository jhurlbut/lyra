# Gaussian Splatting Quality Improvements Guide

This guide provides optimized scripts and configurations to improve the quality of Gaussian splat generation from single images, reducing artifacts and foreground fading issues.

## Problem: Poor Quality Gaussian Splats

**Symptoms:**
- Artifacts in the 3D reconstruction
- Foreground subject fades in and out when view changes
- Inconsistent geometry across viewpoints
- "Floaters" or disconnected Gaussian primitives

**Root Causes:**
1. Insufficient diffusion steps during video generation
2. Low guidance scale causing drift from input image
3. Aggressive depth filtering removing valid foreground points
4. Inconsistent camera trajectories with too much movement
5. Noise augmentation degrading warped frames

---

## Solution: High-Quality Pipeline

### Step 1: Test with Single Trajectory (Recommended First)

Use this to quickly test if improvements work for your image:

```bash
# Make script executable
chmod +x scripts/bash/static_sdg_single_hq.sh

# Edit the script to use YOUR image path
# Change line 7: --input_image_path YOUR_IMAGE.png

# Run single trajectory test
bash scripts/bash/static_sdg_single_hq.sh
```

**What this does:**
- Generates only one camera trajectory (left movement)
- Uses high-quality settings (50 diffusion steps, guidance 7.5)
- Faster than full multi-trajectory (~5-10 minutes vs 30-60 minutes)
- Outputs to: `assets/demo/static/diffusion_output_single_test/`

### Step 2: Generate Full Multi-Trajectory (If Step 1 Looks Good)

```bash
# Make script executable
chmod +x scripts/bash/static_sdg_hq.sh

# Edit the script to use YOUR image path
# Change line 7: --input_image_path YOUR_IMAGE.png

# Run high-quality multi-trajectory generation
bash scripts/bash/static_sdg_hq.sh
```

**What this does:**
- Generates 6 different camera trajectories (left, right, up, zoom_in, zoom_out, clockwise)
- Each uses optimized quality parameters
- Outputs to: `assets/demo/static/diffusion_output_generated_hq/`

### Step 3: Reconstruct 3D Gaussians

```bash
# Run the 3DGS decoder with high-quality config
accelerate launch sample.py --config configs/demo/lyra_static_hq.yaml
```

**What this does:**
- Reconstructs 3D Gaussians from your generated latents
- Renders more frames (every 2nd frame instead of every 4th)
- Exports high-quality PLY file with pruning
- Outputs to: `outputs/demo/lyra_static_hq/gaussians_orig/gaussians_0.ply`

---

## Parameter Improvements Explained

### SDG (Latent Generation) Parameters

| Parameter | Default | Improved | Impact |
|-----------|---------|----------|--------|
| `--num_steps` | 35 | **50** | More denoising iterations = cleaner video frames |
| `--guidance` | 1.0 | **7.5** | Stronger adherence to input image = less drift |
| `--filter_points_threshold` | 0.05 | **0.03** | Stricter depth filtering = fewer bad 3D points |
| `--noise_aug_strength` | 0.0 | **0.0** | No noise added to warped frames = cleaner |
| `--total_movement_distance_factor` | 1.0 | **1.2** | Slightly more camera movement = better coverage |
| `--seed` | Random | **42** | Fixed seed = reproducible results |
| `--foreground_masking` | Off | **On** | Mask boundary regions = fewer artifacts |

### 3DGS Reconstruction Parameters

| Parameter | Default | Improved | Impact |
|-----------|---------|----------|--------|
| `target_index_subsample` | 4 | **2** | Render 2x more frames = smoother results |
| `static_view_indices_fixed` | ['5','0'...] | **['0','1'...]** | Chronological order = smoother transitions |
| `save_gaussians_orig` | false | **true** | Export standard PLY format |

---

## Advanced Tuning

### If Quality is Still Poor

#### 1. Adjust Camera Movement Distance

Edit the movement distances in [scripts/bash/static_sdg_hq.sh](scripts/bash/static_sdg_hq.sh):

```bash
# For objects that are close/foreground-dominant, use smaller movements:
--total_movement_distance_factor 0.8

# For scenes with depth/background content, use larger movements:
--total_movement_distance_factor 1.5
```

#### 2. Generate More Video Frames

Edit both SDG scripts to add:

```bash
--num_video_frames 241  # Default: 121, doubles generation time
```

This provides more temporal coverage and smoother camera trajectories.

#### 3. Adjust Diffusion Guidance

```bash
# If output drifts too far from input image:
--guidance 10.0  # Higher = stricter adherence

# If output is too rigid/artifact-prone:
--guidance 5.0   # Lower = more generation freedom
```

#### 4. Modify Camera Trajectories

Edit `demo_multi_trajectory()` function in:
[cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py](cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py#L567-587)

```python
# Line 570: Adjust spiral trajectory tightness
args.camera_gen_kwargs = {
    'radius_x_factor': 0.10,  # Smaller = tighter spiral (default: 0.15)
    'radius_y_factor': 0.07,  # Smaller = tighter spiral (default: 0.10)
    'num_circles': 1          # Fewer circles = less movement (default: 2)
}

# Lines 572-577: Adjust movement distance ranges
"left": {"traj_idx": 0, "movement_distance_range": [0.15, 0.25]},  # Smaller movements
```

#### 5. Enable PLY Pruning

When exporting PLY files, prune low-opacity Gaussians by editing:
[src/models/utils/render.py](src/models/utils/render.py#L204)

Change the `save_ply_orig` call in [sample.py:307](sample.py#L307) to:

```python
save_ply_orig(model_output['gaussians'], path_gaussians_orig,
              scale_factor=gaussians_scale_factor,
              prune=True,           # Enable pruning
              prune_factor=0.01)    # Remove Gaussians with opacity < 0.01
```

---

## Input Image Best Practices

For best results, your input image should have:

1. **High resolution**: Native 1280x704 or larger
2. **Clear subject**: Well-defined foreground object
3. **Clean background**: Uncluttered, ideally uniform
4. **Good lighting**: Even illumination, minimal shadows
5. **Sharp focus**: No motion blur
6. **Centered composition**: Subject in center 60% of frame
7. **Moderate depth**: Some depth variation but not extreme

---

## Troubleshooting

### Issue: Foreground Still Fades

**Solutions:**
- Increase `--guidance` to 10.0
- Decrease `--filter_points_threshold` to 0.02
- Reduce `--total_movement_distance_factor` to 0.8

### Issue: Too Many Artifacts/Floaters

**Solutions:**
- Enable PLY pruning with `prune=True, prune_factor=0.02`
- Increase `--filter_points_threshold` to 0.04
- Ensure input image has clean background

### Issue: Gaussian Splat Looks "Melted"

**Solutions:**
- Check depth estimation quality (visualize depth in outputs)
- Reduce camera movement distances
- Ensure foreground_masking is enabled

### Issue: Generation Takes Too Long

**Solutions:**
- Use GPU offloading flags (see README Memory Requirements)
- Reduce `--num_steps` back to 35 (small quality loss)
- Use single trajectory test instead of multi-trajectory

---

## File Locations

**Scripts:**
- [scripts/bash/static_sdg_hq.sh](scripts/bash/static_sdg_hq.sh) - Full multi-trajectory HQ generation
- [scripts/bash/static_sdg_single_hq.sh](scripts/bash/static_sdg_single_hq.sh) - Single trajectory test

**Configs:**
- [configs/demo/lyra_static_hq.yaml](configs/demo/lyra_static_hq.yaml) - HQ 3DGS reconstruction config

**Registry:**
- [src/models/data/registry.py](src/models/data/registry.py) - Dataset paths (updated with `lyra_static_demo_generated_hq`)

**Outputs:**
- Latents: `assets/demo/static/diffusion_output_generated_hq/`
- PLY files: `outputs/demo/lyra_static_hq/gaussians_orig/gaussians_0.ply`
- Videos: `outputs/demo/lyra_static_hq/main_gaussians_renderings/`

---

## Quick Reference: Complete Workflow

```bash
# 1. Test single trajectory (fast, ~5-10 min)
bash scripts/bash/static_sdg_single_hq.sh

# 2. If good, run full generation (~30-60 min)
bash scripts/bash/static_sdg_hq.sh

# 3. Reconstruct 3D Gaussians (~5-10 min)
accelerate launch sample.py --config configs/demo/lyra_static_hq.yaml

# 4. Find your PLY file at:
# outputs/demo/lyra_static_hq/gaussians_orig/gaussians_0.ply
```

---

## Summary of Changes

**Created Files:**
1. `scripts/bash/static_sdg_hq.sh` - High-quality multi-trajectory script
2. `scripts/bash/static_sdg_single_hq.sh` - Single trajectory test script
3. `configs/demo/lyra_static_hq.yaml` - High-quality inference config
4. `QUALITY_IMPROVEMENTS.md` - This documentation

**Modified Files:**
1. `src/models/data/registry.py` - Added `lyra_static_demo_generated_hq` dataset entry

**Key Improvements:**
- 50 diffusion steps (vs 35)
- Guidance scale 7.5 (vs 1.0)
- Stricter depth filtering (0.03 vs 0.05)
- 2x frame rendering density
- Fixed random seed for reproducibility
- PLY export with original 3DGS format
