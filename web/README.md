# Lyra Web Application

A modern web interface for generating 3D Gaussian splats from single images using NVIDIA's Lyra pipeline.

## Features

- **Image Upload**: Drag-and-drop or click to upload images
- **Real-time Processing**: Live console output streaming via Server-Sent Events
- **Video Gallery**: View generated multi-view videos as they're created
- **Interactive 3D Viewer**: Explore Gaussian splats using SparkJS/Three.js
- **Job Management**: Track multiple jobs and revisit past results
- **Progress Tracking**: Visual pipeline stages and progress bars

## Architecture

### Backend (FastAPI)
- `config.py` - Configuration and paths
- `job_manager.py` - Job queue and status tracking
- `pipeline.py` - Pipeline execution and logging
- `main.py` - FastAPI endpoints and SSE streaming

### Frontend (Vanilla JS + Three.js + SparkJS)
- `index.html` - UI structure
- `app.js` - Main application logic
- `viewer.js` - 3D viewer with SparkJS
- `styles.css` - Modern dark theme styling

## Installation

### Prerequisites

1. **Lyra Environment**: Ensure you have Lyra installed and configured (see main README.md)
2. **Python 3.8+**: Required for FastAPI backend
3. **CUDA-capable GPU**: Required for pipeline processing

### Setup

1. **Install Backend Dependencies** (using virtual environment):
   ```bash
   cd web/backend
   ./setup.sh
   ```

   This creates a separate Python venv that won't interfere with your conda environment.

   Or manually:
   ```bash
   cd web/backend
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Verify Lyra Checkpoints**:
   Ensure the Lyra checkpoints are downloaded:
   ```bash
   # From project root
   ls checkpoints/Lyra/lyra_static.pt
   ```

3. **Start the Server**:
   ```bash
   cd web/backend
   ./start.sh
   ```

   Or manually:
   ```bash
   cd web/backend
   source venv/bin/activate
   python main.py
   ```

4. **Access the Web App**:
   Open your browser to `http://localhost:8000`

## Usage

### Basic Workflow

1. **Upload Image**
   - Drag and drop an image (PNG/JPG, max 10MB)
   - Or click the upload area to browse files
   - Preview will appear with "Start Processing" button

2. **Start Processing**
   - Click "Start Processing" to begin the pipeline
   - Progress stages will appear:
     - Upload ✓
     - Latent Generation (SDG) - ~5-10 minutes
     - 3DGS Reconstruction - ~5-10 minutes
     - Complete ✓

3. **Monitor Progress**
   - Watch real-time console output
   - View generated videos as they appear
   - Track overall progress percentage

4. **Explore Results**
   - **Videos**: Play generated multi-view videos
   - **3D Viewer**: Interact with the Gaussian splat
     - Left-click + drag: Rotate
     - Right-click + drag: Pan
     - Scroll: Zoom
   - **Download**: Save the PLY file

5. **Job History**
   - View all past jobs in the history section
   - Click any job to reload its results

## API Endpoints

### Upload & Processing
- `POST /api/upload` - Upload image file
- `POST /api/process/{job_id}` - Start pipeline processing

### Job Management
- `GET /api/jobs` - List all jobs
- `GET /api/jobs/{job_id}` - Get job details
- `DELETE /api/jobs/{job_id}` - Delete job and files

### Logging
- `GET /api/jobs/{job_id}/logs` - Get job logs (paginated)
- `GET /api/jobs/{job_id}/stream` - Stream logs via SSE

### Output Files
- `GET /api/outputs/{job_id}/videos` - List available videos
- `GET /api/outputs/{job_id}/videos/{path}` - Get video file
- `GET /api/outputs/{job_id}/ply` - Get PLY file

## Configuration

Edit `backend/config.py` to customize:

### Pipeline Parameters
```python
DEFAULT_SDG_PARAMS = {
    "num_steps": 50,              # Diffusion steps (quality)
    "guidance": 7.5,              # Guidance scale
    "trajectory": "left",         # Camera trajectory
    "movement_distance": 0.25,    # Camera movement range
    # ... more parameters
}
```

### Server Settings
```python
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 8000       # Server port
```

### File Limits
```python
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
```

## Directory Structure

```
web/
├── backend/
│   ├── config.py           # Configuration
│   ├── job_manager.py      # Job tracking
│   ├── pipeline.py         # Pipeline runner
│   ├── main.py             # FastAPI app
│   └── requirements.txt    # Dependencies
├── frontend/
│   ├── index.html          # UI
│   ├── app.js              # Logic
│   ├── viewer.js           # 3D viewer
│   └── styles.css          # Styles
├── uploads/                # Uploaded images
├── outputs/                # Job outputs
│   └── {job_id}/
│       ├── latents/        # SDG outputs
│       └── reconstruction/ # 3DGS outputs
└── README.md               # This file
```

## Troubleshooting

### Pipeline Fails to Start
- **Check GPU**: Ensure CUDA is available
- **Verify Checkpoints**: Confirm `checkpoints/Lyra/lyra_static.pt` exists
- **Review Logs**: Check console output for specific errors

### "Another job is running"
- Only one GPU job can run at a time
- Wait for current job to complete
- Or restart the server to clear stuck jobs

### Videos Not Appearing
- Videos are generated during reconstruction
- Check console logs for generation progress
- Refresh the videos section manually

### PLY Viewer Not Loading
- Ensure PLY file was generated successfully
- Check browser console for errors
- SparkJS requires WebGL support

### Port Already in Use
```bash
# Change port in config.py or:
python main.py --port 8001
```

## Development

### Running in Development Mode
```bash
# Backend with auto-reload
cd web/backend
uvicorn main:app --reload --port 8000

# Frontend served via FastAPI
# No separate build step needed
```

### Testing Individual Components

**Test Upload**:
```bash
curl -X POST -F "file=@test.png" http://localhost:8000/api/upload
```

**Test Job Status**:
```bash
curl http://localhost:8000/api/jobs/{job_id}
```

**Stream Logs**:
```bash
curl -N http://localhost:8000/api/jobs/{job_id}/stream
```

## Performance Tips

1. **GPU Memory**: Close other GPU applications before processing
2. **Multiple Jobs**: Queue system prevents concurrent GPU jobs
3. **File Cleanup**: Delete old jobs to free disk space
4. **Browser Performance**: Close unused tabs while viewing large PLY files

## Advanced Usage

### Custom Pipeline Parameters

Modify `pipeline.py` to use different trajectories or quality settings:

```python
# In _run_sdg method, change parameters:
"--trajectory", "zoom_in",  # Try: left, right, up, zoom_in, zoom_out
"--num_steps", "100",       # Higher = better quality, slower
"--guidance", "10.0",       # Higher = stricter to input
```

### Multi-Trajectory Generation

To generate multiple camera trajectories (like the bash scripts):

1. Edit `pipeline.py` `_run_sdg` method
2. Add `--multi_trajectory` flag support
3. Update dataset registry to handle multiple trajectories

## Credits

- **Lyra**: NVIDIA Gaussian Splatting Pipeline
- **SparkJS**: 3D Gaussian Splat Viewer
- **FastAPI**: Modern Python web framework
- **Three.js**: 3D graphics library

## License

Same license as Lyra (Apache 2.0)
