"""Job management for tracking pipeline executions"""
import asyncio
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
import json


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineStage(str, Enum):
    IDLE = "idle"
    SDG = "sdg"  # Latent generation
    RECONSTRUCTION = "reconstruction"  # 3DGS reconstruction
    FINISHED = "finished"


@dataclass
class Job:
    job_id: str
    image_path: Path
    output_dir: Path
    status: JobStatus = JobStatus.PENDING
    stage: PipelineStage = PipelineStage.IDLE
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    progress: int = 0  # 0-100

    # Output files
    video_files: List[str] = field(default_factory=list)
    ply_file: Optional[str] = None

    def to_dict(self):
        return {
            "job_id": self.job_id,
            "image_path": str(self.image_path),
            "output_dir": str(self.output_dir),
            "status": self.status.value,
            "stage": self.stage.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "logs": self.logs,
            "progress": self.progress,
            "video_files": self.video_files,
            "ply_file": self.ply_file,
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Restore a Job from dictionary"""
        return cls(
            job_id=data["job_id"],
            image_path=Path(data["image_path"]),
            output_dir=Path(data["output_dir"]),
            status=JobStatus(data["status"]),
            stage=PipelineStage(data["stage"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            error_message=data.get("error_message"),
            logs=data.get("logs", []),
            progress=data.get("progress", 0),
            video_files=data.get("video_files", []),
            ply_file=data.get("ply_file"),
        )


class JobManager:
    def __init__(self, state_file: Optional[Path] = None):
        self.jobs: Dict[str, Job] = {}
        self.active_job: Optional[str] = None
        self._lock = asyncio.Lock()
        self.state_file = state_file or Path(__file__).parent / "job_state.json"

        # Load existing jobs from disk
        self._load_state()

    def _save_state(self):
        """Save job state to disk"""
        state = {
            "jobs": {job_id: job.to_dict() for job_id, job in self.jobs.items()},
            "active_job": self.active_job,
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load job state from disk"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)

                # Restore jobs
                for job_id, job_data in state.get("jobs", {}).items():
                    self.jobs[job_id] = Job.from_dict(job_data)

                # Don't restore active_job - it was likely interrupted
                # active jobs will be marked as failed during startup

                print(f"Loaded {len(self.jobs)} jobs from state file")
            except Exception as e:
                print(f"Error loading job state: {e}")

    def create_job(self, image_path: Path, output_dir: Path) -> str:
        """Create a new job and return its ID"""
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            image_path=image_path,
            output_dir=output_dir,
        )
        self.jobs[job_id] = job
        self._save_state()
        return job_id

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        return self.jobs.get(job_id)

    def list_jobs(self) -> List[Dict]:
        """List all jobs"""
        return [job.to_dict() for job in self.jobs.values()]

    async def can_start_job(self) -> bool:
        """Check if a new job can be started (no active GPU job)"""
        async with self._lock:
            return self.active_job is None

    async def set_active_job(self, job_id: str):
        """Mark a job as active"""
        async with self._lock:
            self.active_job = job_id
            job = self.jobs[job_id]
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()

    async def clear_active_job(self):
        """Clear the active job"""
        async with self._lock:
            self.active_job = None

    def update_job_status(self, job_id: str, status: JobStatus, error: Optional[str] = None):
        """Update job status"""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job.status = status
            if status == JobStatus.FAILED:
                job.error_message = error
            if status in (JobStatus.COMPLETED, JobStatus.FAILED):
                job.completed_at = datetime.now()
            self._save_state()

    def update_job_stage(self, job_id: str, stage: PipelineStage, progress: int = 0):
        """Update job pipeline stage and progress"""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job.stage = stage
            job.progress = progress
            self._save_state()

    def add_log(self, job_id: str, message: str):
        """Add a log message to job"""
        if job_id in self.jobs:
            self.jobs[job_id].logs.append(message)
            # Don't save state for every log - too frequent

    def add_video_file(self, job_id: str, video_path: str):
        """Add a generated video file to job"""
        if job_id in self.jobs:
            if video_path not in self.jobs[job_id].video_files:
                self.jobs[job_id].video_files.append(video_path)
                self._save_state()

    def set_ply_file(self, job_id: str, ply_path: str):
        """Set the PLY file for job"""
        if job_id in self.jobs:
            self.jobs[job_id].ply_file = ply_path
            self._save_state()


# Global job manager instance
job_manager = JobManager()
