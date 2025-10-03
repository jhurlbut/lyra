"""TEST VERSION: Pipeline execution for Lyra Gaussian Splatting - Using pre-generated data"""
import asyncio
import os
import re
import shutil
from pathlib import Path
from typing import AsyncGenerator, Callable, Optional
import yaml
import time

from config import (
    BASE_DIR, LYRA_ROOT, SDG_SCRIPT, SAMPLE_SCRIPT,
    DEFAULT_SDG_PARAMS, CONFIGS_DIR
)
from job_manager import JobManager, JobStatus, PipelineStage


# Path to pre-generated test data
TEST_DATA_PATH = Path.home() / "Downloads" / "lyra_outputs" / "outputs" / "demo" / "lyra_static_single_test" / "static_view_indices_fixed_0" / "lyra_static_demo_single_test"


class PipelineRunner:
    def __init__(self, job_manager: JobManager):
        self.job_manager = job_manager

    async def run_pipeline(
        self,
        job_id: str,
        image_path: Path,
        output_dir: Path,
        log_callback: Optional[Callable[[str], None]] = None
    ):
        """SIMULATED pipeline: Copy pre-generated data instead of running scripts"""
        try:
            # Check if can start job
            if not await self.job_manager.can_start_job():
                self.job_manager.update_job_status(
                    job_id, JobStatus.FAILED,
                    "Another job is already running. Please wait."
                )
                return

            await self.job_manager.set_active_job(job_id)

            # Create output directories
            latent_output_dir = output_dir / "latents"
            reconstruction_output_dir = output_dir / "reconstruction"
            latent_output_dir.mkdir(parents=True, exist_ok=True)
            reconstruction_output_dir.mkdir(parents=True, exist_ok=True)

            # Step 1: SIMULATE SDG Latent Generation
            self.job_manager.update_job_stage(job_id, PipelineStage.SDG, 0)
            await self._log(job_id, "=== Starting SDG Latent Generation ===", log_callback)
            await self._log(job_id, "🚨 TEST MODE: Using pre-generated data", log_callback)

            # Simulate SDG progress with fake logs
            await self._simulate_sdg_logs(job_id, log_callback)

            await self._log(job_id, "=== SDG Latent Generation Complete ===", log_callback)

            # Step 2: SIMULATE 3DGS Reconstruction
            self.job_manager.update_job_stage(job_id, PipelineStage.RECONSTRUCTION, 50)
            await self._log(job_id, "=== Starting 3DGS Reconstruction ===", log_callback)

            # Simulate reconstruction progress
            await self._simulate_reconstruction_logs(job_id, log_callback)

            # Copy pre-generated files
            await self._copy_test_data(reconstruction_output_dir, job_id, log_callback)

            await self._log(job_id, "=== 3DGS Reconstruction Complete ===", log_callback)

            # Scan for output files
            await self._scan_outputs(job_id, reconstruction_output_dir)

            # Mark as completed
            self.job_manager.update_job_status(job_id, JobStatus.COMPLETED)
            self.job_manager.update_job_stage(job_id, PipelineStage.FINISHED, 100)
            await self._log(job_id, "=== Pipeline Complete! ===", log_callback)
            await self._log(job_id, "📁 Output files copied from test data", log_callback)

        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            await self._log(job_id, f"ERROR: {error_msg}", log_callback)
            self.job_manager.update_job_status(job_id, JobStatus.FAILED, error_msg)
        finally:
            await self.job_manager.clear_active_job()

    async def _simulate_sdg_logs(self, job_id: str, log_callback: Optional[Callable[[str], None]]):
        """Simulate SDG execution logs"""
        logs = [
            "Loading checkpoint from checkpoints/Lyra/cosmos_world_generation_video_v1.0.pt",
            "Input image shape: (1280, 704)",
            "Generating latent with trajectory: left",
            "Movement distance: 0.25",
            "Processing frame 1/121",
            "Processing frame 25/121",
            "Processing frame 50/121",
            "Processing frame 75/121",
            "Processing frame 100/121",
            "Processing frame 121/121",
            "Saving latent to disk...",
            "SDG generation complete!"
        ]

        for i, log in enumerate(logs):
            await self._log(job_id, log, log_callback)
            # Update progress
            progress = int((i / len(logs)) * 50)
            self.job_manager.update_job_stage(job_id, PipelineStage.SDG, progress)
            await asyncio.sleep(0.5)  # Simulate processing time

    async def _simulate_reconstruction_logs(self, job_id: str, log_callback: Optional[Callable[[str], None]]):
        """Simulate 3DGS reconstruction logs"""
        logs = [
            "Loading Lyra checkpoint: checkpoints/Lyra/lyra_static.pt",
            "Initializing 3D Gaussian model",
            "Loading latent data...",
            "Reconstructing 3D Gaussians:",
            "  Step 1/10: Initialize points",
            "  Step 2/10: Optimize positions",
            "  Step 3/10: Optimize scales",
            "  Step 4/10: Optimize rotations",
            "  Step 5/10: Optimize opacities",
            "  Step 6/10: Optimize SH coefficients",
            "  Step 7/10: Pruning low-opacity Gaussians",
            "  Step 8/10: Rendering validation views",
            "  Step 9/10: Exporting PLY file",
            "  Step 10/10: Generating videos",
            "Reconstruction complete!"
        ]

        for i, log in enumerate(logs):
            await self._log(job_id, log, log_callback)
            # Update progress (50-100%)
            progress = 50 + int((i / len(logs)) * 50)
            self.job_manager.update_job_stage(job_id, PipelineStage.RECONSTRUCTION, progress)
            await asyncio.sleep(0.3)

    async def _copy_test_data(self, output_dir: Path, job_id: str, log_callback: Optional[Callable[[str], None]]):
        """Copy pre-generated test data to output directory"""
        await self._log(job_id, "Copying test data files...", log_callback)

        # Create subdirectories
        gaussians_dir = output_dir / "gaussians_orig"
        videos_dir = output_dir / "main_gaussians_renderings"
        gaussians_dir.mkdir(parents=True, exist_ok=True)
        videos_dir.mkdir(parents=True, exist_ok=True)

        # Copy PLY file
        src_ply = TEST_DATA_PATH / "gaussians_orig" / "gaussians_0.ply"
        dst_ply = gaussians_dir / "gaussians_0.ply"
        if src_ply.exists():
            shutil.copy2(src_ply, dst_ply)
            await self._log(job_id, f"Copied PLY file: {dst_ply.name}", log_callback)

        # Copy video file
        src_video = TEST_DATA_PATH / "main_gaussians_renderings" / "rgb_0.mp4"
        dst_video = videos_dir / "rgb_0.mp4"
        if src_video.exists():
            shutil.copy2(src_video, dst_video)
            await self._log(job_id, f"Copied video file: {dst_video.name}", log_callback)

        # Copy any other relevant files
        for subdir in ["full_output", "raw", "meta"]:
            src_dir = TEST_DATA_PATH / subdir
            if src_dir.exists():
                dst_dir = output_dir / subdir
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
                await self._log(job_id, f"Copied {subdir} directory", log_callback)

    async def _log(
        self,
        job_id: str,
        message: str,
        callback: Optional[Callable[[str], None]] = None
    ):
        """Log message to job and callback"""
        self.job_manager.add_log(job_id, message)
        if callback:
            callback(message)

    async def _scan_outputs(self, job_id: str, output_dir: Path):
        """Scan output directory for generated files"""
        # Look for videos
        video_patterns = ["*.mp4", "*.avi"]
        for pattern in video_patterns:
            for video_file in output_dir.rglob(pattern):
                rel_path = video_file.relative_to(output_dir)
                self.job_manager.add_video_file(job_id, str(rel_path))

        # Look for PLY file
        for ply_file in output_dir.rglob("*.ply"):
            rel_path = ply_file.relative_to(output_dir)
            self.job_manager.set_ply_file(job_id, str(rel_path))
            break  # Use first PLY found