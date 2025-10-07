import { initViewer, loadPLY, resetCamera, toggleCameraLimits, initDebugPanel, unloadViewer, reloadViewer, cachePLYUrl } from './viewer.js';

// State (exposed to window for progress simulation)
window.currentJobId = null;
let currentFile = null;
let eventSource = null;
let videoCheckInterval = null;

// Progress tracking (exposed to window for progress simulation)
window.sdgStartTime = null;
window.processingStartTime = null;
window.latentVideoCount = 0;
window.videoCompletionTimes = [];
window.EXPECTED_TRAJECTORIES = 6;
window.progressSimulationInterval = null;

// Time estimates (in minutes)
const EST_MODEL_LOAD = 3;
const EST_PER_TRAJECTORY = 10;
const EST_RECONSTRUCTION = 10;
const EST_FINALIZATION = 2;
const EST_TOTAL = EST_MODEL_LOAD + (EST_PER_TRAJECTORY * window.EXPECTED_TRAJECTORIES) + EST_RECONSTRUCTION + EST_FINALIZATION; // ~75 min

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
const downloadPlyBtn = document.getElementById('download-ply');
const toggleCameraBtn = document.getElementById('toggle-camera-limits');

const errorModal = document.getElementById('error-modal');
const errorMessage = document.getElementById('error-message');
const closeErrorBtn = document.getElementById('close-error');

const jobsList = document.getElementById('jobs-list');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeUpload();
    // Don't initialize viewer on page load - only when user clicks load button
    initDebugPanel();  // Initialize debug panel controls
    loadJobHistory();

    // Wire up the "Load 3D Viewer" button
    const loadViewerBtn = document.getElementById('load-viewer-btn');
    if (loadViewerBtn) {
        loadViewerBtn.addEventListener('click', async () => {
            console.log('[BTN] Load 3D Viewer button clicked');
            try {
                loadViewerBtn.disabled = true;
                loadViewerBtn.textContent = 'Loading...';
                console.log('[BTN] Button disabled, calling reloadViewer()...');

                // Load the viewer with cached PLY
                await reloadViewer();

                console.log('[BTN] reloadViewer() completed successfully');

                // Hide the button after successful load
                const placeholder = document.getElementById('viewer-placeholder');
                if (placeholder) placeholder.style.display = 'none';
                console.log('[BTN] Placeholder hidden, load complete');
            } catch (error) {
                console.error('[BTN] Error loading viewer:', error);
                console.error('[BTN] Error stack:', error.stack);
                loadViewerBtn.textContent = 'Load 3D Viewer';
                loadViewerBtn.disabled = false;
                console.log('[BTN] Button re-enabled after error');
            }
        });
    }
});

// Cleanup on page unload to prevent memory leaks
window.addEventListener('beforeunload', () => {
    // Close event source
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }

    // Clear all intervals
    if (videoCheckInterval) {
        clearInterval(videoCheckInterval);
        videoCheckInterval = null;
    }

    if (window.progressSimulationInterval) {
        clearInterval(window.progressSimulationInterval);
        window.progressSimulationInterval = null;
    }
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

    downloadPlyBtn.addEventListener('click', downloadPLY);

    toggleCameraBtn.addEventListener('click', () => {
        const isLocked = toggleCameraLimits();
        toggleCameraBtn.textContent = isLocked ? '🔓 Unlock Camera' : '🔒 Lock Camera';
    });

    closeErrorBtn.addEventListener('click', () => errorModal.style.display = 'none');
}

function handleFile(file) {
    // Clear any previous job state
    clearPreviousJob();

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

function clearPreviousJob() {
    // Stop active processes
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }

    if (videoCheckInterval) {
        clearInterval(videoCheckInterval);
        videoCheckInterval = null;
    }

    stopProgressSimulation();

    // Reset state variables
    window.currentJobId = null;
    window.sdgStartTime = null;
    window.latentVideoCount = 0;
    window.videoCompletionTimes = [];
    window.processingStartTime = null;

    // Clear and hide UI sections
    consoleOutput.innerHTML = '';
    consoleSection.style.display = 'none';

    videosGrid.innerHTML = '';
    videosSection.style.display = 'none';

    viewerSection.style.display = 'none';

    progressSection.style.display = 'none';

    // Reset progress UI
    progressFill.style.width = '0%';
    progressText.textContent = '0%';

    // Reset stage labels
    const sdgStageLabel = document.querySelector('#stage-sdg .stage-label');
    if (sdgStageLabel) {
        sdgStageLabel.textContent = 'Latent Gen';
    }
}

async function startProcessing() {
    if (!currentFile) return;

    try {
        startBtn.disabled = true;
        startBtn.textContent = 'Processing...';

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
        window.currentJobId = uploadData.job_id;

        // Clear video grid for new job
        clearVideoGrid();

        // Start processing
        const processResponse = await fetch(`/api/process/${window.currentJobId}`, {
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
}

function startLogStream() {
    if (eventSource) {
        eventSource.close();
    }

    eventSource = new EventSource(`/api/jobs/${window.currentJobId}/stream`);

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

function startProgressSimulation() {
    // Clear any existing simulation
    if (window.progressSimulationInterval) {
        clearInterval(window.progressSimulationInterval);
    }

    window.progressSimulationInterval = setInterval(() => {
        if (!window.processingStartTime) return;

        const elapsedMinutes = (Date.now() - window.processingStartTime) / (1000 * 60);
        let simulatedProgress = 0;

        if (window.latentVideoCount === 0) {
            // Model loading phase (0-3 min → 0-5%)
            simulatedProgress = Math.min(5, (elapsedMinutes / EST_MODEL_LOAD) * 5);
        } else {
            // After first video, use time-based estimation
            const sdgElapsed = (Date.now() - window.sdgStartTime) / (1000 * 60);
            const avgPerVideo = window.latentVideoCount > 0 ? sdgElapsed / window.latentVideoCount : EST_PER_TRAJECTORY;
            const estimatedSDGTotal = avgPerVideo * window.EXPECTED_TRAJECTORIES;
            const sdgProgress = Math.min(100, (sdgElapsed / estimatedSDGTotal) * 100);

            // Map to 5-70% range (SDG phase)
            simulatedProgress = 5 + (sdgProgress * 0.65);
        }

        // Cap at 95% until actual completion
        simulatedProgress = Math.min(95, simulatedProgress);

        // Only update if simulated is higher than current (never decrease)
        const currentProgress = parseInt(progressText.textContent) || 0;
        if (simulatedProgress > currentProgress) {
            updateProgress(Math.round(simulatedProgress));
        }
    }, 5000); // Update every 5 seconds
}

function stopProgressSimulation() {
    if (window.progressSimulationInterval) {
        clearInterval(window.progressSimulationInterval);
        window.progressSimulationInterval = null;
    }
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
    // Initialize timing
    if (!window.sdgStartTime) {
        window.sdgStartTime = Date.now();
    }
    if (!window.processingStartTime) {
        window.processingStartTime = Date.now();
    }

    // Start progress simulation (updates every 5 seconds)
    startProgressSimulation();

    videoCheckInterval = setInterval(async () => {
        try {
            // Check for videos
            const videoResponse = await fetch(`/api/outputs/${window.currentJobId}/videos`);
            if (videoResponse.ok) {
                const videoData = await videoResponse.json();
                displayVideos(videoData.videos);
            }

            // Refresh job history every 10 seconds during processing
            await refreshJobHistory();

            // Check job status and progress
            const jobResponse = await fetch(`/api/jobs/${window.currentJobId}`);
            if (jobResponse.ok) {
                const job = await jobResponse.json();

                // Count latent videos and update stage label
                const newLatentCount = job.video_files ?
                    job.video_files.filter(path => path.includes('latents/')).length : 0;

                if (newLatentCount !== window.latentVideoCount) {
                    // New video detected - record completion time
                    const now = Date.now();
                    window.videoCompletionTimes.push(now);
                    window.latentVideoCount = newLatentCount;
                }

                // Update "Latent Gen x/6" label if in SDG stage
                if (job.stage === 'sdg') {
                    const stageLabel = document.querySelector('#stage-sdg .stage-label');
                    if (stageLabel) {
                        stageLabel.textContent = `Latent Gen ${window.latentVideoCount}/${window.EXPECTED_TRAJECTORIES}`;
                    }

                    // Calculate adaptive time-based progress estimation
                    if (window.latentVideoCount > 0 && window.videoCompletionTimes.length > 0) {
                        const now = Date.now();
                        const elapsedMinutes = (now - window.sdgStartTime) / (1000 * 60);
                        const avgMinutesPerVideo = elapsedMinutes / window.latentVideoCount;
                        const remainingVideos = window.EXPECTED_TRAJECTORIES - window.latentVideoCount;
                        const estimatedRemainingMinutes = remainingVideos * avgMinutesPerVideo;

                        // Total pipeline estimate: SDG + reconstruction (10 min) + final (10 min)
                        const totalEstimatedMinutes = (window.EXPECTED_TRAJECTORIES * avgMinutesPerVideo) + 10 + 10;
                        const estimatedProgress = Math.min(95, (elapsedMinutes / totalEstimatedMinutes) * 100);

                        // Blend estimated progress with backend progress (favor whichever is higher)
                        const blendedProgress = Math.max(job.progress || 0, estimatedProgress);
                        updateProgress(Math.round(blendedProgress));

                        console.log(`Progress estimate: ${window.latentVideoCount}/${window.EXPECTED_TRAJECTORIES} videos, ` +
                                    `${avgMinutesPerVideo.toFixed(1)} min/video avg, ` +
                                    `~${estimatedRemainingMinutes.toFixed(0)} min remaining`);
                    } else {
                        // No videos yet, use backend progress
                        if (job.progress !== undefined) {
                            updateProgress(job.progress);
                        }
                    }
                } else {
                    // Not in SDG stage, use backend progress
                    if (job.progress !== undefined) {
                        updateProgress(job.progress);
                    }
                }

                // Check completion status
                if (job.status === 'completed') {
                    clearInterval(videoCheckInterval);
                    stopProgressSimulation();
                    onPipelineComplete();
                } else if (job.status === 'failed') {
                    clearInterval(videoCheckInterval);
                    stopProgressSimulation();
                    showError(job.error_message || 'Pipeline failed');
                }
            }
        } catch (error) {
            console.error('Error checking status:', error);
        }
    }, 5000); // Check every 5 seconds
}

// Track currently loaded video
let currentlyLoadedVideo = null;

function displayVideos(videos) {
    if (videos.length === 0) return;

    videosSection.style.display = 'block';
    clearVideoGrid();

    videos.forEach(videoPath => {
        const videoCard = document.createElement('div');
        videoCard.className = 'video-card';
        videoCard.style.position = 'relative';
        videoCard.style.cursor = 'pointer';

        // Create placeholder with play button
        const placeholder = document.createElement('div');
        placeholder.className = 'video-placeholder';
        placeholder.style.cssText = `
            width: 100%;
            height: 200px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 8px;
            position: relative;
        `;

        const playIcon = document.createElement('div');
        playIcon.innerHTML = '▶';
        playIcon.style.cssText = `
            font-size: 48px;
            color: white;
            background: rgba(0, 0, 0, 0.5);
            width: 80px;
            height: 80px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            padding-left: 8px;
        `;
        placeholder.appendChild(playIcon);

        const label = document.createElement('div');
        label.className = 'video-label';
        label.textContent = getVideoFriendlyName(videoPath);

        videoCard.appendChild(placeholder);
        videoCard.appendChild(label);
        videosGrid.appendChild(videoCard);

        // Click handler for lazy loading
        videoCard.addEventListener('click', () => {
            loadAndPlayVideo(videoPath, videoCard, placeholder);
        });
    });
}

function getVideoFriendlyName(videoPath) {
    const filename = videoPath.split('/').pop();

    // Check if this is a trajectory video (from SDG phase)
    const latentMatch = videoPath.match(/latents\/(\d+)\/rgb\//);
    if (latentMatch) {
        const trajectoryNum = parseInt(latentMatch[1]);
        const trajectoryNames = ['Left', 'Right', 'Up', 'Zoom Out', 'Zoom In', 'Clockwise'];
        const trajectoryName = trajectoryNames[trajectoryNum] || 'Unknown';
        return `${trajectoryName} Trajectory`;
    }

    // Map technical filenames to user-friendly descriptions for reconstruction videos
    if (filename.includes('rgb_wave')) {
        return '🌊 Gaussian Splat Wave Animation';
    } else if (filename.includes('rgb_0_view_idx')) {
        return 'Splat Preview Rendering';
    } else if (filename === 'rgb_0.mp4') {
        return '🎬 Primary Reconstruction View';
    } else if (filename === 'sample_0.mp4') {
        return 'Output Vis: Splat / Latent / Depth';
    } else if (filename.includes('depth')) {
        return '🏔️ Depth Map Visualization';
    }

    return filename;
}

function loadAndPlayVideo(videoPath, videoCard, placeholder) {
    // Unload currently playing video if exists
    if (currentlyLoadedVideo) {
        const oldVideo = currentlyLoadedVideo.video;
        oldVideo.pause();
        oldVideo.src = '';
        oldVideo.load();

        // Replace video with placeholder again
        const oldCard = currentlyLoadedVideo.card;
        oldCard.innerHTML = '';
        oldCard.appendChild(currentlyLoadedVideo.placeholder);
        oldCard.appendChild(currentlyLoadedVideo.label);
        oldCard.style.cursor = 'pointer';

        // Re-add click handler
        const oldVideoPath = currentlyLoadedVideo.videoPath;
        const oldPlaceholder = currentlyLoadedVideo.placeholder;
        oldCard.onclick = () => loadAndPlayVideo(oldVideoPath, oldCard, oldPlaceholder);
    }

    // Create and load video
    const video = document.createElement('video');
    video.src = `/api/outputs/${window.currentJobId}/videos/${videoPath}`;
    video.controls = true;
    video.loop = true;
    video.autoplay = true;
    video.style.width = '100%';
    video.style.borderRadius = '8px';

    const label = videoCard.querySelector('.video-label');

    // Replace placeholder with video
    videoCard.innerHTML = '';
    videoCard.appendChild(video);
    videoCard.appendChild(label);
    videoCard.style.cursor = 'default';
    videoCard.onclick = null;

    // Track current video
    currentlyLoadedVideo = {
        video: video,
        card: videoCard,
        placeholder: placeholder,
        label: label,
        videoPath: videoPath
    };

    // Play video
    video.play().catch(err => {
        console.error('Error playing video:', err);
    });
}

// Helper function to clear and unload all videos
function clearVideoGrid() {
    if (currentlyLoadedVideo) {
        const video = currentlyLoadedVideo.video;
        video.pause();
        video.src = '';
        video.load();
        currentlyLoadedVideo = null;
    }
    videosGrid.innerHTML = '';
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
        const response = await fetch(`/api/outputs/${window.currentJobId}/videos`);
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
        console.log('[LOAD] loadPLYFile() called for job:', window.currentJobId);

        // Fetch the PLY to cache the URL
        const response = await fetch(`/api/outputs/${window.currentJobId}/ply`);
        if (!response.ok) {
            throw new Error(`PLY file not found: ${response.status} ${response.statusText}`);
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        console.log('[LOAD] PLY blob URL created:', url);

        // Cache the URL for later loading (don't initialize viewer yet)
        cachePLYUrl(url);

        // Show viewer section with placeholder (not loaded yet)
        viewerSection.style.display = 'block';
        const placeholder = document.getElementById('viewer-placeholder');
        const viewerContainer = document.getElementById('viewer-container');

        console.log('[LOAD] Showing placeholder, hiding viewer container');
        if (placeholder) {
            placeholder.style.display = 'block';
            console.log('[LOAD] Placeholder display set to block');
        } else {
            console.error('[LOAD] Placeholder element not found!');
        }

        // Hide the actual viewer container
        if (viewerContainer) {
            viewerContainer.style.display = 'none';
            console.log('[LOAD] Viewer container hidden');
        }
    } catch (error) {
        console.error('Error preparing PLY:', error);
        showError(`Error loading PLY file: ${error.message}`);
    }
}

async function downloadPLY() {
    if (!window.currentJobId) return;

    const link = document.createElement('a');
    link.href = `/api/outputs/${window.currentJobId}/ply`;
    link.download = `gaussians_${window.currentJobId}.ply`;
    link.click();
}

async function checkJobStatus() {
    try {
        const response = await fetch(`/api/jobs/${window.currentJobId}`);
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

async function refreshJobHistory() {
    await loadJobHistory();
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

        const deleteButton = job.status !== 'running' ?
            `<button class="cancel-job-btn" data-job-id="${job.job_id}" style="background: var(--error-color);">Delete</button>` : '';

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
            ${cancelButton}${deleteButton}
        `;

        // Add cancel button handler
        const cancelBtn = jobCard.querySelector('.cancel-job-btn');
        if (cancelBtn && job.status === 'running') {
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

        // Add delete button handler
        if (deleteButton) {
            const deleteBtn = jobCard.querySelector('.cancel-job-btn');
            if (deleteBtn) {
                deleteBtn.addEventListener('click', async (e) => {
                    e.stopPropagation(); // Prevent job card click
                    if (confirm('Delete this job and all its files?')) {
                        try {
                            const response = await fetch(`/api/jobs/${job.job_id}`, {
                                method: 'DELETE'
                            });
                            if (response.ok) {
                                refreshJobHistory();
                            }
                        } catch (error) {
                            console.error('Error deleting job:', error);
                        }
                    }
                });
            }
        }

        jobCard.addEventListener('click', () => loadJob(job.job_id));
        jobsList.appendChild(jobCard);
    });
}

async function loadJob(jobId) {
    window.currentJobId = jobId;

    // Clear video grid for loaded job
    clearVideoGrid();

    try {
        const response = await fetch(`/api/jobs/${jobId}`);
        if (response.ok) {
            const job = await response.json();

            // Load and display the original uploaded image
            if (job.image_path) {
                // Extract filename from path (e.g., "/path/to/{job_id}.png" -> "{job_id}.png")
                const filename = job.image_path.split('/').pop();

                // Set preview image source
                previewImage.src = `/api/uploads/${filename}`;

                // Show preview container, hide upload area
                previewContainer.style.display = 'block';
                uploadArea.style.display = 'none';
                startBtn.style.display = 'none';
            }

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
