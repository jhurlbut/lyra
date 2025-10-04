import { initViewer, loadPLY, resetCamera } from './viewer.js';

// State
let currentJobId = null;
let currentFile = null;
let eventSource = null;
let videoCheckInterval = null;

// DOM Elements
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const previewContainer = document.getElementById('preview-container');
const previewImage = document.getElementById('preview-image');
const removeImageBtn = document.getElementById('remove-image');
const startBtn = document.getElementById('start-btn');

const progressSection = document.getElementById('progress-section');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');

const consoleSection = document.getElementById('console-section');
const consoleOutput = document.getElementById('console-output');
const toggleConsoleBtn = document.getElementById('toggle-console');
const clearConsoleBtn = document.getElementById('clear-console');
const consoleContainer = document.getElementById('console-container');

const videosSection = document.getElementById('videos-section');
const videosGrid = document.getElementById('videos-grid');

const viewerSection = document.getElementById('viewer-section');
const resetCameraBtn = document.getElementById('reset-camera');
const downloadPlyBtn = document.getElementById('download-ply');

const errorModal = document.getElementById('error-modal');
const errorMessage = document.getElementById('error-message');
const closeErrorBtn = document.getElementById('close-error');

const jobsList = document.getElementById('jobs-list');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeUpload();
    initViewer();  // Fixed: should be initViewer from viewer.js
    loadJobHistory();
});

// Upload handling
function initializeUpload() {
    uploadArea.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    removeImageBtn.addEventListener('click', clearFile);
    startBtn.addEventListener('click', startProcessing);

    toggleConsoleBtn.addEventListener('click', toggleConsole);
    clearConsoleBtn.addEventListener('click', () => consoleOutput.innerHTML = '');

    resetCameraBtn.addEventListener('click', resetCamera);
    downloadPlyBtn.addEventListener('click', downloadPLY);

    // Reference point event handlers
    document.getElementById('set-origin-point').addEventListener('click', () => {
        if (window.setReferencePointMode) {
            window.setReferencePointMode(true);
            // Update button to show active state
            document.getElementById('set-origin-point').style.background = '#3b82f6';
            document.getElementById('set-origin-point').style.color = 'white';
            alert('Click on the splat where you want to set the new origin point.');
        }
    });

    document.getElementById('apply-origin').addEventListener('click', () => {
        if (window.applyReferencePoint) {
            window.applyReferencePoint();
            // Reset button style
            document.getElementById('set-origin-point').style.background = '';
            document.getElementById('set-origin-point').style.color = '';
        }
    });

    document.getElementById('clear-origin').addEventListener('click', () => {
        if (window.clearReferencePoint) {
            window.clearReferencePoint();
            // Reset button style
            document.getElementById('set-origin-point').style.background = '';
            document.getElementById('set-origin-point').style.color = '';
        }
    });

    closeErrorBtn.addEventListener('click', () => errorModal.style.display = 'none');
}

function handleFile(file) {
    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
        showError('Invalid file type. Please upload a PNG or JPG image.');
        return;
    }

    // Validate file size (10MB)
    if (file.size > 10 * 1024 * 1024) {
        showError('File too large. Maximum size is 10MB.');
        return;
    }

    currentFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadArea.style.display = 'none';
        previewContainer.style.display = 'block';
        startBtn.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

function clearFile() {
    currentFile = null;
    previewImage.src = '';
    uploadArea.style.display = 'flex';
    previewContainer.style.display = 'none';
    startBtn.style.display = 'none';
    fileInput.value = '';
}

async function startProcessing() {
    if (!currentFile) return;

    try {
        startBtn.disabled = true;
        startBtn.textContent = 'Uploading...';

        // Upload file
        const formData = new FormData();
        formData.append('file', currentFile);

        const uploadResponse = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        if (!uploadResponse.ok) {
            throw new Error('Upload failed');
        }

        const uploadData = await uploadResponse.json();
        currentJobId = uploadData.job_id;

        // Start processing
        const processResponse = await fetch(`/api/process/${currentJobId}`, {
            method: 'POST'
        });

        if (!processResponse.ok) {
            const error = await processResponse.json();
            throw new Error(error.detail || 'Processing failed to start');
        }

        // Show progress UI
        showProcessingUI();

        // Start streaming logs
        startLogStream();

        // Start checking for videos
        startVideoCheck();

        updateStage('upload', true);

    } catch (error) {
        showError(error.message);
        startBtn.disabled = false;
        startBtn.textContent = 'Start Processing';
    }
}

function showProcessingUI() {
    progressSection.style.display = 'block';
    consoleSection.style.display = 'block';

    // Scroll to progress section
    progressSection.scrollIntoView({ behavior: 'smooth' });
}

function startLogStream() {
    if (eventSource) {
        eventSource.close();
    }

    eventSource = new EventSource(`/api/jobs/${currentJobId}/stream`);

    eventSource.onmessage = (event) => {
        const logLine = event.data;
        appendLog(logLine);

        // Update stages based on log content
        if (logLine.includes('Starting SDG')) {
            updateStage('sdg', false);
        } else if (logLine.includes('SDG') && logLine.includes('Complete')) {
            updateStage('sdg', true);
        } else if (logLine.includes('Starting 3DGS')) {
            updateStage('recon', false);
        } else if (logLine.includes('3DGS') && logLine.includes('Complete')) {
            updateStage('recon', true);
        } else if (logLine.includes('Pipeline Complete')) {
            updateStage('done', true);
            onPipelineComplete();
        }
    };

    eventSource.onerror = () => {
        eventSource.close();
        checkJobStatus();
    };
}

function appendLog(message) {
    const line = document.createElement('div');
    line.className = 'log-line';
    line.textContent = message;
    consoleOutput.appendChild(line);

    // Auto-scroll to bottom
    consoleOutput.scrollTop = consoleOutput.scrollHeight;
}

function updateStage(stage, completed) {
    const stages = {
        'upload': 0,
        'sdg': 25,
        'recon': 50,
        'done': 100
    };

    const stageElement = document.getElementById(`stage-${stage}`);
    if (stageElement) {
        if (completed) {
            stageElement.classList.add('completed');
        } else {
            stageElement.classList.add('active');
        }
    }

    const progress = stages[stage] || 0;
    updateProgress(progress);
}

function updateProgress(percent) {
    progressFill.style.width = `${percent}%`;
    progressText.textContent = `${percent}%`;
}

function toggleConsole() {
    if (consoleContainer.style.display === 'none') {
        consoleContainer.style.display = 'block';
        toggleConsoleBtn.textContent = 'Hide Console';
    } else {
        consoleContainer.style.display = 'none';
        toggleConsoleBtn.textContent = 'Show Console';
    }
}

function startVideoCheck() {
    videoCheckInterval = setInterval(async () => {
        try {
            // Check for videos
            const videoResponse = await fetch(`/api/outputs/${currentJobId}/videos`);
            if (videoResponse.ok) {
                const videoData = await videoResponse.json();
                displayVideos(videoData.videos);
            }

            // Check job status and progress
            const jobResponse = await fetch(`/api/jobs/${currentJobId}`);
            if (jobResponse.ok) {
                const job = await jobResponse.json();

                // Update progress bar with actual backend progress
                if (job.progress !== undefined) {
                    updateProgress(job.progress);
                }

                // Check completion status
                if (job.status === 'completed') {
                    clearInterval(videoCheckInterval);
                    onPipelineComplete();
                } else if (job.status === 'failed') {
                    clearInterval(videoCheckInterval);
                    showError(job.error_message || 'Pipeline failed');
                }
            }
        } catch (error) {
            console.error('Error checking status:', error);
        }
    }, 5000); // Check every 5 seconds
}

function displayVideos(videos) {
    if (videos.length === 0) return;

    videosSection.style.display = 'block';
    videosGrid.innerHTML = '';

    videos.forEach(videoPath => {
        const videoCard = document.createElement('div');
        videoCard.className = 'video-card';

        const video = document.createElement('video');
        video.src = `/api/outputs/${currentJobId}/videos/${videoPath}`;
        video.controls = true;
        video.loop = true;

        const label = document.createElement('div');
        label.className = 'video-label';
        
        // Create user-friendly labels based on filename and path
        const filename = videoPath.split('/').pop();
        let friendlyName = filename;
        
        // Check if this is a latent video (from SDG phase)
        if (videoPath.includes('latents/0/rgb/') || videoPath.includes('latents/rgb/')) {
            friendlyName = '🎥 Generated Camera Trajectory (Latent)';
            // Add job ID or trajectory type if present in filename
            if (filename.includes('left')) {
                friendlyName = '🎥 Left Camera Trajectory (Latent)';
            } else if (filename.includes('right')) {
                friendlyName = '🎥 Right Camera Trajectory (Latent)';
            } else if (filename.includes('up')) {
                friendlyName = '🎥 Upward Camera Trajectory (Latent)';
            }
        } 
        // Map technical filenames to user-friendly descriptions for reconstruction videos
        else if (filename.includes('rgb_wave')) {
            friendlyName = '🌊 Gaussian Splat Wave Animation';
        } else if (filename.includes('rgb_0_view_idx')) {
            friendlyName = '📹 Multi-View Rendering';
        } else if (filename === 'rgb_0.mp4') {
            friendlyName = '🎬 Primary Reconstruction View';
        } else if (filename === 'sample_0.mp4') {
            friendlyName = '✨ Sample Output Visualization';
        } else if (filename.includes('left')) {
            friendlyName = '⬅️ Left Trajectory Video';
        } else if (filename.includes('right')) {
            friendlyName = '➡️ Right Trajectory Video';  
        } else if (filename.includes('up')) {
            friendlyName = '⬆️ Upward Trajectory Video';
        } else if (filename.includes('zoom_in')) {
            friendlyName = '🔍 Zoom In Trajectory';
        } else if (filename.includes('zoom_out')) {
            friendlyName = '🔎 Zoom Out Trajectory';
        } else if (filename.includes('clockwise')) {
            friendlyName = '🔄 Clockwise Rotation Video';
        } else if (filename.includes('depth')) {
            friendlyName = '🏔️ Depth Map Visualization';
        }
        
        label.textContent = friendlyName;

        videoCard.appendChild(video);
        videoCard.appendChild(label);
        videosGrid.appendChild(videoCard);
    });
}

async function onPipelineComplete() {
    // Stop checking for videos
    if (videoCheckInterval) {
        clearInterval(videoCheckInterval);
    }

    // Close event stream
    if (eventSource) {
        eventSource.close();
    }

    // Load final videos
    try {
        const response = await fetch(`/api/outputs/${currentJobId}/videos`);
        if (response.ok) {
            const data = await response.json();
            displayVideos(data.videos);
        }
    } catch (error) {
        console.error('Error loading videos:', error);
    }

    // Load PLY file
    await loadPLYFile();

    // Update job history
    await loadJobHistory();

    // Reset upload UI
    startBtn.disabled = false;
    startBtn.textContent = 'Start Processing';
    clearFile();
}

async function loadPLYFile() {
    try {
        const response = await fetch(`/api/outputs/${currentJobId}/ply`);
        if (response.ok) {
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);

            await loadPLY(url);
            viewerSection.style.display = 'block';

            // Scroll to viewer
            viewerSection.scrollIntoView({ behavior: 'smooth' });
        }
    } catch (error) {
        console.error('Error loading PLY:', error);
    }
}

async function downloadPLY() {
    if (!currentJobId) return;

    const link = document.createElement('a');
    link.href = `/api/outputs/${currentJobId}/ply`;
    link.download = `gaussians_${currentJobId}.ply`;
    link.click();
}

async function checkJobStatus() {
    try {
        const response = await fetch(`/api/jobs/${currentJobId}`);
        if (response.ok) {
            const job = await response.json();

            if (job.status === 'completed') {
                onPipelineComplete();
            } else if (job.status === 'failed') {
                showError(job.error_message || 'Pipeline failed');
            }
        }
    } catch (error) {
        console.error('Error checking job status:', error);
    }
}

async function loadJobHistory() {
    try {
        const response = await fetch('/api/jobs');
        if (response.ok) {
            const data = await response.json();
            displayJobHistory(data.jobs);
        }
    } catch (error) {
        console.error('Error loading job history:', error);
    }
}

function displayJobHistory(jobs) {
    if (jobs.length === 0) {
        jobsList.innerHTML = '<p class="no-jobs">No jobs yet. Upload an image to get started!</p>';
        return;
    }

    jobsList.innerHTML = '';

    // Sort by created_at descending
    jobs.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));

    jobs.forEach(job => {
        const jobCard = document.createElement('div');
        jobCard.className = `job-card job-${job.status}`;

        const date = new Date(job.created_at).toLocaleString();

        const cancelButton = job.status === 'running' ?
            `<button class="cancel-job-btn" data-job-id="${job.job_id}">Cancel</button>` : '';

        jobCard.innerHTML = `
            <div class="job-header">
                <span class="job-id">${job.job_id.substring(0, 8)}</span>
                <span class="job-status">${job.status}</span>
            </div>
            <div class="job-details">
                <div>Created: ${date}</div>
                <div>Stage: ${job.stage}</div>
                <div>Progress: ${job.progress}%</div>
            </div>
            ${cancelButton}
        `;

        // Add cancel button handler
        const cancelBtn = jobCard.querySelector('.cancel-job-btn');
        if (cancelBtn) {
            cancelBtn.addEventListener('click', async (e) => {
                e.stopPropagation(); // Prevent job card click
                if (confirm('Cancel this job?')) {
                    try {
                        const response = await fetch(`/api/jobs/${job.job_id}/cancel`, {
                            method: 'POST'
                        });
                        if (response.ok) {
                            refreshJobHistory();
                        }
                    } catch (error) {
                        console.error('Error cancelling job:', error);
                    }
                }
            });
        }

        jobCard.addEventListener('click', () => loadJob(job.job_id));
        jobsList.appendChild(jobCard);
    });
}

async function loadJob(jobId) {
    currentJobId = jobId;

    try {
        const response = await fetch(`/api/jobs/${jobId}`);
        if (response.ok) {
            const job = await response.json();

            // Show progress section
            showProcessingUI();

            // Load logs
            consoleOutput.innerHTML = '';
            job.logs?.forEach(log => appendLog(log));

            // Update progress
            updateProgress(job.progress);

            // Load videos
            if (job.video_files?.length > 0) {
                displayVideos(job.video_files);
            }

            // Load PLY if available
            if (job.ply_file && job.status === 'completed') {
                await loadPLYFile();
            }
        }
    } catch (error) {
        showError(`Error loading job: ${error.message}`);
    }
}

function showError(message) {
    errorMessage.textContent = message;
    errorModal.style.display = 'flex';
}
