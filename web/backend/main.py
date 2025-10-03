"""FastAPI backend for Lyra Gaussian Splatting Web Application"""
import asyncio
from pathlib import Path
from typing import List
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from config import (
    UPLOAD_DIR, OUTPUT_DIR, ALLOWED_EXTENSIONS,
    MAX_UPLOAD_SIZE, HOST, PORT, WEB_DIR
)
from job_manager import job_manager, JobStatus
from pipeline import PipelineRunner  # REAL VERSION
# from pipeline_test import PipelineRunner  # TEST VERSION - using pre-generated data


app = FastAPI(title="Lyra Gaussian Splatting API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
app.mount("/static", StaticFiles(directory=WEB_DIR / "frontend"), name="static")

# Pipeline runner
pipeline_runner = PipelineRunner(job_manager)

# Store SSE clients for each job
sse_clients = {}


@app.get("/")
async def root():
    """Serve frontend"""
    return FileResponse(WEB_DIR / "frontend" / "index.html")


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image file"""
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Read file
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {MAX_UPLOAD_SIZE / 1024 / 1024}MB"
        )

    # Create job
    job_id = job_manager.create_job(
        image_path=UPLOAD_DIR / f"{job_manager.create_job.__self__.jobs.__len__()}{file_ext}",
        output_dir=OUTPUT_DIR / "temp"
    )

    # Save uploaded file
    image_path = UPLOAD_DIR / f"{job_id}{file_ext}"
    with open(image_path, "wb") as f:
        f.write(contents)

    # Update job with correct paths
    output_dir = OUTPUT_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    job = job_manager.get_job(job_id)
    job.image_path = image_path
    job.output_dir = output_dir

    return {
        "job_id": job_id,
        "filename": file.filename,
        "message": "File uploaded successfully"
    }


@app.post("/api/process/{job_id}")
async def process_job(job_id: str, background_tasks: BackgroundTasks):
    """Start processing a job"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != JobStatus.PENDING:
        raise HTTPException(status_code=400, detail="Job already processed")

    # Check if another job is running
    if not await job_manager.can_start_job():
        raise HTTPException(
            status_code=409,
            detail="Another job is currently running. Please wait."
        )

    # Start pipeline in background
    background_tasks.add_task(
        pipeline_runner.run_pipeline,
        job_id,
        job.image_path,
        job.output_dir,
        lambda msg: broadcast_log(job_id, msg)
    )

    return {
        "job_id": job_id,
        "message": "Processing started",
        "status": "running"
    }


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs"""
    return {"jobs": job_manager.list_jobs()}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job details"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return job.to_dict()


@app.get("/api/jobs/{job_id}/logs")
async def get_logs(job_id: str, tail: int = 100):
    """Get job logs"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    logs = job.logs[-tail:] if tail > 0 else job.logs
    return {"logs": logs}


@app.get("/api/jobs/{job_id}/stream")
async def stream_logs(job_id: str):
    """Stream job logs via Server-Sent Events"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        # Create queue for this client
        queue = asyncio.Queue()

        # Register client
        if job_id not in sse_clients:
            sse_clients[job_id] = []
        sse_clients[job_id].append(queue)

        try:
            # Send existing logs first
            for log in job.logs:
                yield f"data: {log}\n\n"

            # Stream new logs
            while True:
                log = await queue.get()
                if log is None:  # Job finished
                    break
                yield f"data: {log}\n\n"

                # Check if job is finished
                current_job = job_manager.get_job(job_id)
                if current_job and current_job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    break

        finally:
            # Unregister client
            if job_id in sse_clients:
                sse_clients[job_id].remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/api/outputs/{job_id}/videos")
async def list_videos(job_id: str):
    """List available videos for a job"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {"videos": job.video_files}


@app.get("/api/outputs/{job_id}/videos/{video_path:path}")
async def get_video(job_id: str, video_path: str):
    """Get a video file"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    video_file = job.output_dir / "reconstruction" / video_path
    if not video_file.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    return FileResponse(video_file)


@app.get("/api/outputs/{job_id}/ply")
async def get_ply(job_id: str):
    """Get the PLY file"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not job.ply_file:
        raise HTTPException(status_code=404, detail="PLY file not yet available")

    ply_file = job.output_dir / "reconstruction" / job.ply_file
    if not ply_file.exists():
        raise HTTPException(status_code=404, detail="PLY file not found")

    return FileResponse(ply_file, media_type="application/octet-stream")


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its files"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Don't delete if running
    if job.status == JobStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Cannot delete running job")

    # Delete files
    if job.image_path.exists():
        job.image_path.unlink()
    if job.output_dir.exists():
        shutil.rmtree(job.output_dir)

    # Remove from manager
    del job_manager.jobs[job_id]

    return {"message": "Job deleted successfully"}


def broadcast_log(job_id: str, message: str):
    """Broadcast log message to all SSE clients for a job"""
    if job_id in sse_clients:
        for queue in sse_clients[job_id]:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                pass


if __name__ == "__main__":
    print(f"Starting Lyra Web API on http://{HOST}:{PORT}")
    print(f"Frontend available at http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)
