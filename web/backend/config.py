"""Configuration for Lyra Web Application Backend"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
WEB_DIR = BASE_DIR / "web"
UPLOAD_DIR = WEB_DIR / "uploads"
OUTPUT_DIR = WEB_DIR / "outputs"

# Lyra paths
LYRA_ROOT = BASE_DIR
CHECKPOINTS_DIR = LYRA_ROOT / "checkpoints"
CONFIGS_DIR = LYRA_ROOT / "configs"

# Pipeline scripts
SDG_SCRIPT = LYRA_ROOT / "cosmos_predict1" / "diffusion" / "inference" / "gen3c_single_image_sdg.py"
SAMPLE_SCRIPT = LYRA_ROOT / "sample.py"

# Pipeline settings
DEFAULT_SDG_PARAMS = {
    "num_steps": 50,
    "guidance": 7.5,
    "filter_points_threshold": 0.03,
    "noise_aug_strength": 0.0,
    "seed": 42,
    "trajectory": "left",  # For single trajectory
    "movement_distance": 0.25,
    "camera_rotation": "center_facing",
    "foreground_masking": True,
}

# Supported image formats
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB

# Server settings
HOST = "0.0.0.0"
PORT = 8000

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
