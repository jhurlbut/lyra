"""Pipeline execution for Lyra Gaussian Splatting"""
import asyncio
import os
import re
import shutil
from pathlib import Path
from typing import AsyncGenerator, Callable, Optional
import yaml

from config import (
    BASE_DIR, LYRA_ROOT, SDG_SCRIPT, SAMPLE_SCRIPT,
    DEFAULT_SDG_PARAMS, CONFIGS_DIR
)
from job_manager import JobManager, JobStatus, PipelineStage
from logger import logger as lyra_logger


class PipelineRunner:
    def __init__(self, job_manager: JobManager):
        self.job_manager = job_manager
        self.logger = lyra_logger

    async def run_pipeline(
        self,
        job_id: str,
        image_path: Path,
        output_dir: Path,
        log_callback: Optional[Callable[[str], None]] = None
    ):
        """Run the complete pipeline: SDG + 3DGS reconstruction"""
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

            # Step 1: SDG Latent Generation
            self.job_manager.update_job_stage(job_id, PipelineStage.SDG, 0)
            await self._log(job_id, "=== Starting SDG Latent Generation ===", log_callback)

            success = await self._run_sdg(
                job_id, image_path, latent_output_dir, log_callback
            )

            if not success:
                raise Exception("SDG latent generation failed")

            await self._log(job_id, "=== SDG Latent Generation Complete ===", log_callback)
            
            # Restructure the latent output to match expected format
            await self._restructure_latent_output(job_id, latent_output_dir, log_callback)
            
            # Scan for latent videos after SDG completes
            await self._scan_latent_outputs(job_id, latent_output_dir)

            # Step 2: 3DGS Reconstruction
            self.job_manager.update_job_stage(job_id, PipelineStage.RECONSTRUCTION, 50)
            await self._log(job_id, "=== Starting 3DGS Reconstruction ===", log_callback)

            # Create dynamic config for this job
            config_path = await self._create_job_config(
                job_id, latent_output_dir, reconstruction_output_dir
            )

            success = await self._run_reconstruction(
                job_id, config_path, log_callback
            )

            if not success:
                raise Exception("3DGS reconstruction failed")

            await self._log(job_id, "=== 3DGS Reconstruction Complete ===", log_callback)

            # Scan for output files
            await self._scan_outputs(job_id, reconstruction_output_dir)

            # Mark as completed
            self.job_manager.update_job_status(job_id, JobStatus.COMPLETED)
            self.job_manager.update_job_stage(job_id, PipelineStage.FINISHED, 100)
            await self._log(job_id, "=== Pipeline Complete! ===", log_callback)

        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            await self._log(job_id, f"ERROR: {error_msg}", log_callback)
            self.job_manager.update_job_status(job_id, JobStatus.FAILED, error_msg)
        finally:
            await self.job_manager.clear_active_job()

    async def _run_sdg(
        self,
        job_id: str,
        image_path: Path,
        output_dir: Path,
        log_callback: Optional[Callable[[str], None]]
    ) -> bool:
        """Run SDG latent generation"""
        # Build command with minimal default parameters
        cmd = [
            "torchrun",
            "--nproc_per_node=1",
            str(SDG_SCRIPT),
            "--checkpoint_dir", "checkpoints",
            "--num_gpus", "1",
            "--input_image_path", str(image_path),
            "--video_save_folder", str(output_dir),
        ]

        # Add optional flags from DEFAULT_SDG_PARAMS
        if DEFAULT_SDG_PARAMS.get("foreground_masking"):
            cmd.append("--foreground_masking")

        if DEFAULT_SDG_PARAMS.get("multi_trajectory"):
            cmd.append("--multi_trajectory")

        # Add total_movement_distance_factor (default 1.0 for balanced camera motion)
        movement_factor = DEFAULT_SDG_PARAMS.get("total_movement_distance_factor", 1.0)
        cmd.extend(["--total_movement_distance_factor", str(movement_factor)])

        # Execute directly on AIP job with periodic video scanning
        return await self._run_subprocess(
            job_id, cmd, log_callback, cwd=LYRA_ROOT,
            scan_videos_dir=output_dir  # Enable periodic video scanning during SDG
        )

    async def _run_reconstruction(
        self,
        job_id: str,
        config_path: Path,
        log_callback: Optional[Callable[[str], None]]
    ) -> bool:
        """Run 3DGS reconstruction"""
        cmd = [
            "accelerate", "launch",
            str(SAMPLE_SCRIPT),
            "--config", str(config_path)
        ]

        # Execute directly on AIP job
        return await self._run_subprocess(job_id, cmd, log_callback, cwd=LYRA_ROOT)

    async def _run_subprocess(
        self,
        job_id: str,
        cmd: list,
        log_callback: Optional[Callable[[str], None]],
        cwd: Optional[Path] = None,
        scan_videos_dir: Optional[Path] = None
    ) -> bool:
        """Execute subprocess directly with conda environment"""
        # Set up environment
        env = os.environ.copy()
        env['CUDA_HOME'] = env.get('CONDA_PREFIX', '/opt/conda/envs/lyra')
        env['PYTHONPATH'] = str(cwd) if cwd else str(LYRA_ROOT)

        # Build shell command with conda activation
        cmd_str = ' '.join(str(c) for c in cmd)
        shell_cmd = f"source /opt/conda/bin/activate lyra && cd {cwd or LYRA_ROOT} && {cmd_str}"

        await self._log(job_id, f"Executing: {shell_cmd}", log_callback)

        process = await asyncio.create_subprocess_shell(
            shell_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            executable='/bin/bash'
        )

        return await self._stream_output(job_id, process, log_callback, scan_videos_dir)

    async def _stream_output(
        self,
        job_id: str,
        process: asyncio.subprocess.Process,
        log_callback: Optional[Callable[[str], None]],
        scan_videos_dir: Optional[Path] = None
    ) -> bool:
        """Stream subprocess output to logs with optional periodic video scanning"""
        try:
            # Create a task for periodic video scanning if enabled
            scan_task = None
            if scan_videos_dir:
                scan_task = asyncio.create_task(
                    self._periodic_video_scan(job_id, scan_videos_dir)
                )

            while True:
                line = await process.stdout.readline()
                if not line:
                    break

                line_str = line.decode('utf-8').rstrip()
                await self._log(job_id, line_str, log_callback)
                self._parse_progress(job_id, line_str)

            # Wait for process to complete
            returncode = await process.wait()

            # Cancel periodic scanning task if it exists
            if scan_task:
                scan_task.cancel()
                try:
                    await scan_task
                except asyncio.CancelledError:
                    pass

            if returncode == 0:
                await self._log(job_id, "Command completed successfully", log_callback)
                return True
            else:
                await self._log(job_id, f"Command failed with exit code {returncode}", log_callback)
                return False

        except Exception as e:
            await self._log(job_id, f"Error executing command: {str(e)}", log_callback)
            return False

    async def _periodic_video_scan(self, job_id: str, latent_dir: Path):
        """Periodically scan for new videos during SDG execution"""
        last_trajectory_count = 0
        while True:
            try:
                await asyncio.sleep(10)  # Scan every 10 seconds

                # Count completed trajectories
                trajectory_count = 0
                for trajectory_dir in sorted(latent_dir.glob("*")):
                    if trajectory_dir.is_dir() and trajectory_dir.name.isdigit():
                        rgb_dir = trajectory_dir / "rgb"
                        if rgb_dir.exists():
                            # Check if this trajectory has a video
                            has_video = False
                            for video_file in rgb_dir.glob("*.mp4"):
                                job = self.job_manager.get_job(job_id)
                                if job:
                                    rel_path = Path("latents") / trajectory_dir.name / "rgb" / video_file.name
                                    rel_path_str = str(rel_path)
                                    if rel_path_str not in job.video_files:
                                        self.job_manager.add_video_file(job_id, rel_path_str)
                                    has_video = True

                            if has_video:
                                trajectory_count += 1

                # Update progress based on trajectory completion (assume 6 trajectories total)
                if trajectory_count > 0:
                    # Progress: 0-50% during SDG (6 trajectories)
                    progress = int((trajectory_count / 6.0) * 50)
                    self.job_manager.update_job_stage(job_id, PipelineStage.SDG, progress)

                # Log if new trajectories found
                if trajectory_count > last_trajectory_count:
                    self.logger.log_job(job_id, f"Completed trajectory {trajectory_count}/6")
                    last_trajectory_count = trajectory_count

            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Silently ignore errors in background scanning
                pass

    def _parse_progress(self, job_id: str, log_line: str):
        """Parse progress from log lines"""
        # Look for patterns like "Step 25/50" or "25/50"
        match = re.search(r'(\d+)/(\d+)', log_line)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            progress = int((current / total) * 100)

            job = self.job_manager.get_job(job_id)
            if job:
                # Scale progress based on stage
                if job.stage == PipelineStage.SDG:
                    progress = int(progress * 0.5)  # 0-50%
                elif job.stage == PipelineStage.RECONSTRUCTION:
                    progress = 50 + int(progress * 0.5)  # 50-100%

                self.job_manager.update_job_stage(job_id, job.stage, progress)

    async def _log(
        self,
        job_id: str,
        message: str,
        callback: Optional[Callable[[str], None]] = None
    ):
        """Log message to job and callback using centralized logger"""
        self.job_manager.add_log(job_id, message)
        # Also log to file using centralized logger
        self.logger.log_job(job_id, message, callback=callback)

    async def _create_job_config(
        self,
        job_id: str,
        latent_dir: Path,
        output_dir: Path
    ) -> Path:
        """Create a dynamic YAML config for this job"""
        # First, update the registry to add our web job dataset with the correct path
        await self._update_registry_for_job(latent_dir)
        
        # Create job-specific config using the web job dataset
        config = {
            "out_dir_inference": str(output_dir),
            "dataset_name": "lyra_web_job",
            "static_view_indices_fixed": ['0'],  # Single trajectory
            "target_index_subsample": 2,
            "set_manual_time_idx": True,
            "config_path": [
                "configs/training/default.yaml",
                "configs/training/3dgs_res_704_1280_views_121_multi_6_prune.yaml"
            ],
            "ckpt_path": "checkpoints/Lyra/lyra_static.pt",
            "save_gaussians_orig": True,
            "save_gt_input": True,
            "save_gt_depth": True,
            "save_video_input": False,
            "save_rgb_decoding": False,
            "skip_existing": False,  # Always regenerate for web jobs
            "out_fps": 24,
        }

        # Save config
        config_path = output_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        return config_path

    async def _update_registry_for_job(self, latent_dir: Path):
        """Update the registry.py file with web job dataset pointing to correct path"""
        registry_file = LYRA_ROOT / "src" / "models" / "data" / "registry.py"
        
        # Read the registry file
        with open(registry_file, 'r') as f:
            lines = f.readlines()
        
        # Check if lyra_web_job already exists
        has_web_job = any("'lyra_web_job'" in line or '"lyra_web_job"' in line for line in lines)
        
        if has_web_job:
            # Update the existing entry's root_path
            in_web_job = False
            for i, line in enumerate(lines):
                if "'lyra_web_job'" in line or '"lyra_web_job"' in line:
                    in_web_job = True
                elif in_web_job and '"root_path":' in line:
                    # Update this line with the new path
                    lines[i] = f'        "root_path": "{str(latent_dir)}",\n'
                    in_web_job = False
                    break
        else:
            # Add new entry at the end
            web_job_entry = f"""
# Web pipeline job (dynamically generated)
dataset_registry['lyra_web_job'] = {{
    'cls': RadymWrapper,
    'kwargs': {{
        "root_path": "{str(latent_dir)}",
        "is_static": True,
        "is_multi_view": True,
        "has_latents": True,
        "is_generated_cosmos_latent": True,
        "sampling_buckets": [['0']],
        "start_view_idx": 0,
    }},
    'scene_scale': 1.,
    'max_gap': 121,
    'min_gap': 45,
}}
"""
            lines.append(web_job_entry)
        
        # Write back the updated registry
        with open(registry_file, 'w') as f:
            f.writelines(lines)

    async def _restructure_latent_output(self, job_id: str, latent_dir: Path, log_callback: Optional[Callable[[str], None]]):
        """Restructure SDG output to match expected directory structure"""
        await self._log(job_id, "Restructuring latent output for Lyra compatibility...", log_callback)
        
        # Check if files are in the root of latent_dir (wrong structure)
        subdirs = ['latent', 'pose', 'rgb', 'intrinsics']
        needs_restructure = any((latent_dir / subdir).exists() for subdir in subdirs)
        
        if needs_restructure:
            # Create trajectory folder "0"
            trajectory_dir = latent_dir / "0"
            trajectory_dir.mkdir(exist_ok=True)
            
            # Move each subdirectory into the trajectory folder
            for subdir in subdirs:
                src = latent_dir / subdir
                if src.exists():
                    dst = trajectory_dir / subdir
                    if dst.exists():
                        # Remove destination if it exists
                        shutil.rmtree(dst)
                    shutil.move(str(src), str(dst))
                    await self._log(job_id, f"  Moved {subdir}/ to 0/{subdir}/", log_callback)
            
            await self._log(job_id, "Restructuring complete - files now in 0/ trajectory folder", log_callback)
        else:
            # Check if already in correct structure
            if (latent_dir / "0").exists():
                await self._log(job_id, "Latent output already in correct structure", log_callback)
    
    async def _scan_latent_outputs(self, job_id: str, latent_dir: Path):
        """Scan latent directory for generated videos"""
        # Scan for all trajectory folders (multi_trajectory creates 0-5)
        video_patterns = ["*.mp4", "*.avi"]
        found_videos = False

        # Look for trajectory folders (0, 1, 2, ...)
        for trajectory_dir in sorted(latent_dir.glob("*")):
            if trajectory_dir.is_dir() and trajectory_dir.name.isdigit():
                rgb_dir = trajectory_dir / "rgb"
                if rgb_dir.exists():
                    for pattern in video_patterns:
                        for video_file in rgb_dir.glob(pattern):
                            job = self.job_manager.get_job(job_id)
                            if job:
                                rel_path = Path("latents") / trajectory_dir.name / "rgb" / video_file.name
                                self.job_manager.add_video_file(job_id, str(rel_path))
                                found_videos = True

        # Fallback: check for direct rgb folder (single trajectory without restructuring)
        if not found_videos:
            rgb_dir = latent_dir / "rgb"
            if rgb_dir.exists():
                for pattern in video_patterns:
                    for video_file in rgb_dir.glob(pattern):
                        job = self.job_manager.get_job(job_id)
                        if job:
                            rel_path = Path("latents") / "rgb" / video_file.name
                            self.job_manager.add_video_file(job_id, str(rel_path))

    async def _scan_outputs(self, job_id: str, output_dir: Path):
        """Scan output directory for generated files"""
        # Look for videos
        video_patterns = ["*.mp4", "*.avi"]
        for pattern in video_patterns:
            for video_file in output_dir.rglob(pattern):
                rel_path = video_file.relative_to(output_dir)
                self.job_manager.add_video_file(job_id, str(rel_path))

        # Look for PLY file in reconstruction subdirectory
        reconstruction_dir = output_dir / "reconstruction"
        if reconstruction_dir.exists():
            for ply_file in reconstruction_dir.rglob("*.ply"):
                # Store path relative to reconstruction dir (for use with main.py prefix)
                rel_path = ply_file.relative_to(reconstruction_dir)
                self.job_manager.set_ply_file(job_id, str(rel_path))
                break  # Use first PLY found
