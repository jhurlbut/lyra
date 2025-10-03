// Simplified viewer for testing - just shows PLY info without SparkJS

export function initViewer() {
    const container = document.getElementById('viewer-container');
    if (!container) {
        console.log('Viewer container not found');
        return;
    }
    
    // Just create a simple message for now
    container.innerHTML = `
        <div style="padding: 40px; text-align: center; color: #94a3b8;">
            <h3>3D Viewer Placeholder</h3>
            <p>PLY file is ready for viewing</p>
            <p style="font-size: 0.9em; opacity: 0.7;">
                Note: SparkJS integration requires additional setup
            </p>
        </div>
    `;
    
    console.log('Simple viewer initialized');
}

export async function loadPLY(url) {
    console.log('PLY URL:', url);
    
    const container = document.getElementById('viewer-container');
    if (!container) return;
    
    // Show that we received the PLY
    container.innerHTML = `
        <div style="padding: 40px; text-align: center; color: #94a3b8;">
            <h3>✅ PLY File Loaded</h3>
            <p>File: gaussians_0.ply</p>
            <p>Ready for 3D viewing</p>
            <p style="font-size: 0.9em; opacity: 0.7; margin-top: 20px;">
                URL: ${url}
            </p>
        </div>
    `;
    
    // Show the viewer section
    const viewerSection = document.getElementById('viewer-section');
    if (viewerSection) {
        viewerSection.style.display = 'block';
        console.log('Viewer section shown');
    }
}

export function resetCamera() {
    console.log('Reset camera (placeholder)');
}

export function disposeViewer() {
    console.log('Dispose viewer (placeholder)');
}