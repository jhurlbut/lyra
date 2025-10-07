import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Viewer state
let scene, camera, renderer, controls;
let splatMesh = null;
let animationId = null;
let isAnimating = false;
let intersectionObserver = null;

// Reference point system
let raycaster = null;
let mouse = new THREE.Vector2();
let referencePointMode = false;
let referencePoint = null;
let referencePointIndicator = null;

// Camera limits toggle state
let cameraLimitsEnabled = true;

// Configurable camera settings (adjusted for 2x trajectory movement)
let cameraSettings = {
    fov: 55,                    // Field of view (was 45° for 1x)
    distance: 1.6,              // Camera distance (was 1.0 for 1x)
    cameraZ: 1.5,               // Camera Z position (was 1.0 for 1x)
    minAzimuth: -40,            // Min horizontal rotation in degrees (was -20° for 1x)
    maxAzimuth: 10,             // Max horizontal rotation in degrees (was +5° for 1x)
    minPolar: 78,               // Min vertical angle in degrees (was ~75° for 1x)
    maxPolar: 90,               // Max vertical angle in degrees (was ~105° for 1x)
    splatX: 0,                  // Splat X position
    splatY: 0,                  // Splat Y position
    splatZ: 1.5                 // Splat Z position (was 1.0 for 1x)
};

export function initViewer() {
    const container = document.getElementById('viewer-container');
    
    // Ensure container has dimensions
    if (!container) {
        console.error('Viewer container not found');
        return;
    }
    
    // Get actual dimensions (fallback to defaults if 0)
    const width = container.clientWidth || 800;
    const height = container.clientHeight || 600;
    console.log('Initializing viewer with dimensions:', width, height);

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);

    // Camera (FOV from settings)
    camera = new THREE.PerspectiveCamera(
        cameraSettings.fov,
        width / height,
        0.01,  // Near clipping plane (lowered from 0.1 to see closer objects)
        1000   // Far clipping plane
    );
    camera.position.set(0, 0, 5);

    // Renderer (antialias disabled for better SparkJS performance)
    renderer = new THREE.WebGLRenderer({
        antialias: false,
        powerPreference: 'high-performance',
        alpha: false,
        stencil: false
    });
    renderer.setSize(width, height);
    // Limit pixel ratio to max 2 for better performance
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    // Add CSS hardware acceleration hints
    renderer.domElement.style.transform = 'translateZ(0)';
    renderer.domElement.style.willChange = 'transform';

    container.appendChild(renderer.domElement);

    // Controls with rotation limits matching SDG trajectory bounds
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.25;  // Higher = less momentum (was 0.05)
    controls.screenSpacePanning = false;
    // Lock camera distance (from settings)
    controls.minDistance = cameraSettings.distance;
    controls.maxDistance = cameraSettings.distance;
    // Limit rotation to SDG trajectory angular bounds (from settings)
    controls.minAzimuthAngle = cameraSettings.minAzimuth * Math.PI / 180;
    controls.maxAzimuthAngle = cameraSettings.maxAzimuth * Math.PI / 180;
    controls.minPolarAngle = cameraSettings.minPolar * Math.PI / 180;
    controls.maxPolarAngle = cameraSettings.maxPolar * Math.PI / 180;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 5, 5);
    scene.add(directionalLight);

    // Handle window resize
    window.addEventListener('resize', onWindowResize);

    // Start animation loop
    startAnimation();

    // Initialize reference point system
    initReferencePointSystem();

    // Initialize visibility observer to pause animation when off-screen
    initVisibilityObserver();
}

function initReferencePointSystem() {
    try {
        // Initialize raycaster
        raycaster = new THREE.Raycaster();
        
        // Add mouse event listeners for reference point selection
        const container = document.getElementById('viewer-container');
        if (container && renderer && renderer.domElement) {
            renderer.domElement.addEventListener('click', onMouseClick);
            renderer.domElement.addEventListener('mousemove', onMouseMove);
        }
        
        // Make reference point functions available globally
        window.setReferencePointMode = setReferencePointMode;
        window.applyReferencePoint = applyReferencePoint;
        window.clearReferencePoint = clearReferencePoint;
        
        console.log('Reference point system initialized');
    } catch (error) {
        console.error('Error initializing reference point system:', error);
    }
}

function onMouseMove(event) {
    if (!referencePointMode) return;
    
    // Update mouse coordinates for raycasting
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
}

function onMouseClick(event) {
    if (!referencePointMode || !splatMesh || !raycaster) return;
    
    // Update mouse coordinates
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    
    console.log('Attempting raycasting at mouse position:', mouse.x, mouse.y);
    
    // Perform raycasting - try multiple approaches
    raycaster.setFromCamera(mouse, camera);
    
    // First try direct intersection with splatMesh
    let intersects = raycaster.intersectObject(splatMesh, true);
    console.log('Direct splat intersects:', intersects.length);
    
    // If no direct intersection, try intersecting with all scene objects
    if (intersects.length === 0) {
        intersects = raycaster.intersectObjects(scene.children, true);
        console.log('Scene intersects:', intersects.length);
    }
    
    // If still no intersection, create a point based on camera direction and distance
    if (intersects.length === 0) {
        console.log('No intersections found, creating point based on camera direction');
        
        // Calculate a point in front of the camera at a reasonable distance
        const direction = new THREE.Vector3();
        raycaster.ray.direction.clone().normalize();
        
        // Use the camera position and look towards the splat mesh
        const targetDistance = 10; // Approximate distance to the splat
        const point = new THREE.Vector3();
        point.copy(camera.position);
        camera.getWorldDirection(direction);
        point.add(direction.multiplyScalar(targetDistance));
        
        setReferencePoint(point);
        console.log('Reference point set at calculated position:', point);
        return;
    }
    
    // Use the first intersection found
    const intersection = intersects[0];
    setReferencePoint(intersection.point);
    console.log('Reference point set at intersection:', intersection.point);
}

function setReferencePointMode(enabled) {
    referencePointMode = enabled;
    
    // Update cursor style to indicate mode
    if (renderer && renderer.domElement) {
        renderer.domElement.style.cursor = enabled ? 'crosshair' : 'default';
    }
    
    // Update UI to show current mode
    console.log('Reference point mode:', enabled ? 'ENABLED' : 'DISABLED');
}

function setReferencePoint(point) {
    referencePoint = point.clone();
    
    // Remove existing indicator
    if (referencePointIndicator) {
        scene.remove(referencePointIndicator);
    }
    
    // Create visual indicator for reference point
    const geometry = new THREE.SphereGeometry(0.2, 16, 16);
    const material = new THREE.MeshBasicMaterial({ 
        color: 0xff0000, 
        transparent: true, 
        opacity: 0.8 
    });
    referencePointIndicator = new THREE.Mesh(geometry, material);
    referencePointIndicator.position.copy(referencePoint);
    scene.add(referencePointIndicator);
    
    // Exit reference point mode after selection
    setReferencePointMode(false);
    
    console.log('Reference point indicator created at:', referencePoint);
}

function applyReferencePoint() {
    if (!referencePoint || !splatMesh) {
        console.log('No reference point set or no mesh loaded');
        return;
    }
    
    // Calculate offset needed to move reference point to origin
    const offset = referencePoint.clone().negate();
    
    // Apply offset to mesh position
    splatMesh.position.add(offset);
    
    // Move the reference point indicator to origin
    if (referencePointIndicator) {
        referencePointIndicator.position.set(0, 0, 0);
    }
    
    // Update controls target to origin
    controls.target.set(0, 0, 0);
    controls.update();
    
    console.log('Applied reference point offset:', offset);
    console.log('Mesh repositioned so reference point is at origin');
}

function clearReferencePoint() {
    referencePoint = null;
    
    // Remove visual indicator
    if (referencePointIndicator) {
        scene.remove(referencePointIndicator);
        referencePointIndicator = null;
    }
    
    // Exit reference point mode
    setReferencePointMode(false);
    
    console.log('Reference point cleared');
}

function animate() {
    if (!isAnimating) return;

    animationId = requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function startAnimation() {
    if (isAnimating) return;
    isAnimating = true;

    // Render one frame immediately to avoid expensive first-frame cost during scroll
    if (renderer && scene && camera && controls) {
        controls.update();
        renderer.render(scene, camera);
    }

    animate();
    console.log('Animation started');
}

function stopAnimation() {
    if (!isAnimating) return;
    isAnimating = false;
    if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
    }
    console.log('Animation stopped');
}

function initVisibilityObserver() {
    const container = document.getElementById('viewer-section');
    if (!container) return;

    // Observe when the viewer section enters/exits the viewport
    intersectionObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                // Viewer is visible - start animation during browser idle time
                // This avoids blocking scroll/paint operations
                if ('requestIdleCallback' in window) {
                    requestIdleCallback(() => {
                        startAnimation();
                    }, { timeout: 200 });
                } else {
                    // Fallback for browsers without requestIdleCallback
                    setTimeout(() => startAnimation(), 100);
                }
            } else {
                // Viewer is off-screen - stop animation immediately
                stopAnimation();
            }
        });
    }, {
        // Use rootMargin to trigger earlier for smoother experience
        rootMargin: '100px 0px',
        // Trigger when any part is visible
        threshold: 0
    });

    intersectionObserver.observe(container);
    console.log('Visibility observer initialized for viewer section');
}

function onWindowResize() {
    const container = document.getElementById('viewer-container');
    
    if (!container || !camera || !renderer) return;
    
    const width = container.clientWidth || 800;
    const height = container.clientHeight || 600;

    camera.aspect = width / height;
    camera.updateProjectionMatrix();

    renderer.setSize(width, height);
}

export async function loadPLY(url, onProgress = null) {
    console.log('Loading Gaussian Splat PLY from:', url);

    // Remove existing mesh
    if (splatMesh) {
        scene.remove(splatMesh);
        splatMesh = null;
    }

    // Dynamically import SparkJS (ES module)
    const Spark = await import('@sparkjsdev/spark');
    console.log('SparkJS loaded:', Spark);

    // Store Spark globally for sorting functions
    window.Spark = Spark;

    // Use SplatLoader with progress tracking
    const loader = new Spark.SplatLoader();
    return loader.loadAsync(url, (event) => {
        if (event.type === "progress") {
            const progress = event.lengthComputable
                ? `${((event.loaded / event.total) * 100).toFixed(2)}%`
                : `${event.loaded} bytes`;
            console.log(`Background download progress: ${progress}`);

            // Call custom progress callback if provided
            if (onProgress && event.lengthComputable) {
                const percentage = ((event.loaded / event.total) * 100).toFixed(0);
                onProgress(percentage);
            }
        }
    })
    .then((packedSplats) => {
        console.log('Splat data loaded, creating mesh...');

        // Create SplatMesh from loaded data
        splatMesh = new Spark.SplatMesh({ packedSplats });

        // Re-orient from OpenCV to OpenGL coordinates
        splatMesh.quaternion.set(1, 0, 0, 0);
        //splatMesh.position.set(0, 0, -1);
       // splatMesh.scale.setScalar(0.5);

        console.log('Applied coordinate system transformation (OpenCV -> OpenGL)');

        // Add to scene
        scene.add(splatMesh);
        console.log('SplatMesh added to scene');

        // Get accurate bounding box using SparkJS method
        let box, center, size;
        try {
            // Use SparkJS getBoundingBox for accurate bounds
            box = splatMesh.getBoundingBox(false); // false = include splat scales for accuracy
            center = box.getCenter(new THREE.Vector3());
            size = box.getSize(new THREE.Vector3());
            console.log('Using SparkJS getBoundingBox');
        } catch (e) {
            // Fallback to Three.js method if SparkJS method fails
            box = new THREE.Box3().setFromObject(splatMesh);
            center = box.getCenter(new THREE.Vector3());
            size = box.getSize(new THREE.Vector3());
            console.log('Using Three.js Box3.setFromObject fallback');
        }

        console.log('Bounding box:', { center, size });

        // Calculate appropriate scale - make it 5x bigger than before
        const maxDim = Math.max(size.x, size.y, size.z);
        const targetSize = 20; // 5x bigger than the original 4 units
        const scaleFactor = maxDim > 0 ? targetSize / maxDim : 1;
        // Apply additional scaling on top of the 0.5 coordinate transform scale
        splatMesh.scale.setScalar(0.5 * scaleFactor);

        // Position the mesh (from settings)
        splatMesh.position.set(cameraSettings.splatX, cameraSettings.splatY, cameraSettings.splatZ);

        console.log(`Scaled by ${0.5 * scaleFactor} (0.5 coord transform * ${scaleFactor} fit) to target size of ${targetSize} units`);
        console.log(`Positioned at (${cameraSettings.splatX}, ${cameraSettings.splatY}, ${cameraSettings.splatZ})`);

        // Position camera (from settings)
        camera.position.set(0, 0, cameraSettings.cameraZ);
        camera.lookAt(0, 0, 0);  // Look at origin

        // Update controls target to origin
        controls.target.set(0, 0, 0);
        controls.update();

        // Pre-warm the renderer with an initial render to avoid first-frame freeze
        // This performs the expensive splat sorting/GPU buffer updates now
        // instead of when the viewer first scrolls into view
        console.log('Pre-warming renderer with initial frame...');
        renderer.render(scene, camera);
        console.log('Initial frame rendered');

        // Make viewer section visible
        const viewerSection = document.getElementById('viewer-section');
        if (viewerSection) {
            viewerSection.style.display = 'block';
        }

        console.log('Gaussian splat loaded and displayed with SparkJS!');
        return true;
    })
    .catch((error) => {
        console.error('Error loading Gaussian splat with SparkJS:', error);

        // Show error in viewer
        const container = document.getElementById('viewer-container');
        if (container) {
            container.innerHTML = `
                <div style="padding: 40px; text-align: center; color: #f59e0b;">
                    <h3>⚠️ Loading Issue</h3>
                    <p style="font-size: 0.9em; margin-top: 10px; color: #94a3b8;">
                        ${error.message}
                    </p>
                </div>
            `;
        }

        // Still show the viewer section
        const viewerSection = document.getElementById('viewer-section');
        if (viewerSection) {
            viewerSection.style.display = 'block';
        }

        throw error;
    });
}


// Splat sorting functions
export function sortSplatsByDepth() {
    if (!splatMesh || !camera) return;

    console.log('Sorting splats by camera depth...');

    const cameraPos = camera.position;
    const splats = [];

    // Extract all splat data with camera distance
    // Callback signature: (index, center, scales, quaternion, opacity, color)
    splatMesh.forEachSplat((index, center, scales, quaternion, opacity, color) => {
        const dx = center.x - cameraPos.x;
        const dy = center.y - cameraPos.y;
        const dz = center.z - cameraPos.z;
        const distance = Math.sqrt(dx*dx + dy*dy + dz*dz);

        splats.push({
            index,
            center: center.clone(),
            scales: scales.clone(),
            quaternion: quaternion.clone(),
            opacity,
            color: color.clone(),
            distance
        });
    });

    // Sort by distance (back to front)
    splats.sort((a, b) => b.distance - a.distance);

    // Rebuild packedSplats in new order
    rebuildSplatOrder(splats);
}

export function sortSplatsByOpacity() {
    if (!splatMesh) return;

    console.log('Sorting splats by opacity...');

    const splats = [];

    // Extract all splat data with opacity
    // Callback signature: (index, center, scales, quaternion, opacity, color)
    splatMesh.forEachSplat((index, center, scales, quaternion, opacity, color) => {
        splats.push({
            index,
            center: center.clone(),
            scales: scales.clone(),
            quaternion: quaternion.clone(),
            opacity,
            color: color.clone()
        });
    });

    // Sort by opacity (descending - highest first)
    splats.sort((a, b) => b.opacity - a.opacity);

    // Rebuild packedSplats in new order
    rebuildSplatOrder(splats);
}

export function reverseSplatOrder() {
    if (!splatMesh) return;

    console.log('Reversing splat order...');

    const splats = [];

    // Extract all splat data
    // Callback signature: (index, center, scales, quaternion, opacity, color)
    splatMesh.forEachSplat((index, center, scales, quaternion, opacity, color) => {
        splats.push({
            index,
            center: center.clone(),
            scales: scales.clone(),
            quaternion: quaternion.clone(),
            opacity,
            color: color.clone()
        });
    });

    // Reverse the array
    splats.reverse();

    // Rebuild packedSplats in new order
    rebuildSplatOrder(splats);
}

function rebuildSplatOrder(sortedSplats) {
    const Spark = window.Spark;
    if (!Spark) {
        console.error('Spark not available');
        return;
    }

    // Create new PackedSplats with same capacity
    const newPackedSplats = new Spark.PackedSplats(sortedSplats.length);

    // Rebuild in sorted order
    sortedSplats.forEach(({ center, scales, quaternion, opacity, color }) => {
        newPackedSplats.pushSplat(
            center,
            scales,
            quaternion,
            opacity,
            color
        );
    });

    // Replace the mesh's packed splats
    splatMesh.packedSplats = newPackedSplats;
    splatMesh.packedSplats.needsUpdate = true;

    console.log(`Reordered ${sortedSplats.length} splats`);
}

export function resetCamera() {
    if (splatMesh) {
        // Reset camera to view the scaled and positioned mesh with improved positioning
        const targetSize = 20; // Same as used in loadPLY (5x bigger)
        const viewDistance = targetSize * 2.0; // Closer than previous 2.5x
        const modelHeight = targetSize; // Approximate height based on scaling
        camera.position.set(0, modelHeight * 0.3, viewDistance);
        camera.lookAt(0, modelHeight * 0.4, 0); // Look at mid-height
        controls.target.set(0, modelHeight / 2, 0);
        controls.update();
    } else {
        camera.position.set(0, 0, 5);
        camera.lookAt(0, 0, 0);
        controls.target.set(0, 0, 0);
        controls.update();
    }
}

export function toggleCameraLimits() {
    if (!controls) return;

    cameraLimitsEnabled = !cameraLimitsEnabled;

    if (cameraLimitsEnabled) {
        // Enable limits - restore restricted rotation and zoom
        controls.minDistance = 1.0;
        controls.maxDistance = 1.0;
        controls.minAzimuthAngle = -Math.PI / 9;   // -20°
        controls.maxAzimuthAngle = Math.PI / 36;   // +5°
        controls.minPolarAngle = Math.PI / 2 - 0.26;  // ~75°
        controls.maxPolarAngle = Math.PI / 2 + 0.26;  // ~105°
    } else {
        // Disable limits - allow free rotation and zoom
        controls.minDistance = 0;
        controls.maxDistance = Infinity;
        controls.minAzimuthAngle = -Infinity;
        controls.maxAzimuthAngle = Infinity;
        controls.minPolarAngle = 0;
        controls.maxPolarAngle = Math.PI;
    }

    return cameraLimitsEnabled;
}

export function disposeViewer() {
    stopAnimation();

    if (intersectionObserver) {
        intersectionObserver.disconnect();
        intersectionObserver = null;
    }

    if (splatMesh) {
        scene.remove(splatMesh);
        splatMesh = null;
    }

    if (renderer) {
        renderer.dispose();
    }

    window.removeEventListener('resize', onWindowResize);
}

// ============================================================================
// Debug Panel Functions
// ============================================================================

// Setter functions for camera parameters
export function updateCameraFOV(fov) {
    if (camera) {
        camera.fov = fov;
        camera.updateProjectionMatrix();
    }
    cameraSettings.fov = fov;
}

export function updateCameraDistance(distance) {
    if (controls) {
        controls.minDistance = distance;
        controls.maxDistance = distance;
    }
    cameraSettings.distance = distance;
}

export function updateCameraZ(z) {
    if (camera) {
        camera.position.z = z;
    }
    cameraSettings.cameraZ = z;
}

export function updateMinAzimuth(degrees) {
    if (controls) {
        controls.minAzimuthAngle = degrees * Math.PI / 180;
    }
    cameraSettings.minAzimuth = degrees;
}

export function updateMaxAzimuth(degrees) {
    if (controls) {
        controls.maxAzimuthAngle = degrees * Math.PI / 180;
    }
    cameraSettings.maxAzimuth = degrees;
}

export function updateMinPolar(degrees) {
    if (controls) {
        controls.minPolarAngle = degrees * Math.PI / 180;
    }
    cameraSettings.minPolar = degrees;
}

export function updateMaxPolar(degrees) {
    if (controls) {
        controls.maxPolarAngle = degrees * Math.PI / 180;
    }
    cameraSettings.maxPolar = degrees;
}

export function updateSplatX(x) {
    if (splatMesh) {
        splatMesh.position.x = x;
    }
    cameraSettings.splatX = x;
}

export function updateSplatY(y) {
    if (splatMesh) {
        splatMesh.position.y = y;
    }
    cameraSettings.splatY = y;
}

export function updateSplatZ(z) {
    if (splatMesh) {
        splatMesh.position.z = z;
    }
    cameraSettings.splatZ = z;
}

export function resetDebugSettings() {
    // Reset to default values for 2x movement
    const defaults = {
        fov: 55,
        distance: 1.6,
        cameraZ: 1.5,
        minAzimuth: -40,
        maxAzimuth: 10,
        minPolar: 78,
        maxPolar: 90,
        splatX: 0,
        splatY: 0,
        splatZ: 1.5
    };

    updateCameraFOV(defaults.fov);
    updateCameraDistance(defaults.distance);
    updateCameraZ(defaults.cameraZ);
    updateMinAzimuth(defaults.minAzimuth);
    updateMaxAzimuth(defaults.maxAzimuth);
    updateMinPolar(defaults.minPolar);
    updateMaxPolar(defaults.maxPolar);
    updateSplatX(defaults.splatX);
    updateSplatY(defaults.splatY);
    updateSplatZ(defaults.splatZ);

    // Update UI
    document.getElementById('fov-slider').value = defaults.fov;
    document.getElementById('fov-value').textContent = defaults.fov;
    document.getElementById('distance-slider').value = defaults.distance;
    document.getElementById('distance-value').textContent = defaults.distance;
    document.getElementById('camera-z-slider').value = defaults.cameraZ;
    document.getElementById('camera-z-value').textContent = defaults.cameraZ;
    document.getElementById('min-azimuth-slider').value = defaults.minAzimuth;
    document.getElementById('min-azimuth-value').textContent = defaults.minAzimuth;
    document.getElementById('max-azimuth-slider').value = defaults.maxAzimuth;
    document.getElementById('max-azimuth-value').textContent = defaults.maxAzimuth;
    document.getElementById('min-polar-slider').value = defaults.minPolar;
    document.getElementById('min-polar-value').textContent = defaults.minPolar;
    document.getElementById('max-polar-slider').value = defaults.maxPolar;
    document.getElementById('max-polar-value').textContent = defaults.maxPolar;
    document.getElementById('splat-x-slider').value = defaults.splatX;
    document.getElementById('splat-x-value').textContent = defaults.splatX;
    document.getElementById('splat-y-slider').value = defaults.splatY;
    document.getElementById('splat-y-value').textContent = defaults.splatY;
    document.getElementById('splat-z-slider').value = defaults.splatZ;
    document.getElementById('splat-z-value').textContent = defaults.splatZ;
}

export function copyDebugSettings() {
    const settingsText = `Camera Settings (2x movement):
FOV: ${cameraSettings.fov}°
Distance: ${cameraSettings.distance}
Camera Z: ${cameraSettings.cameraZ}
Min Azimuth: ${cameraSettings.minAzimuth}°
Max Azimuth: ${cameraSettings.maxAzimuth}°
Min Polar: ${cameraSettings.minPolar}°
Max Polar: ${cameraSettings.maxPolar}°
Splat Position: (${cameraSettings.splatX}, ${cameraSettings.splatY}, ${cameraSettings.splatZ})`;

    navigator.clipboard.writeText(settingsText).then(() => {
        alert('Settings copied to clipboard!');
    }).catch(err => {
        console.error('Failed to copy settings:', err);
    });
}

// Initialize debug panel controls
export function initDebugPanel() {
    const debugPanel = document.getElementById('debug-panel');
    const toggleButton = document.getElementById('toggle-debug-panel');

    // Keyboard listener for 'D' key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'd' || e.key === 'D') {
            const isVisible = debugPanel.style.display !== 'none';
            debugPanel.style.display = isVisible ? 'none' : 'block';
        }
    });

    // Toggle button click
    if (toggleButton) {
        toggleButton.addEventListener('click', () => {
            const isVisible = debugPanel.style.display !== 'none';
            debugPanel.style.display = isVisible ? 'none' : 'block';
        });
    }

    // FOV slider
    const fovSlider = document.getElementById('fov-slider');
    const fovValue = document.getElementById('fov-value');
    fovSlider.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        fovValue.textContent = value;
        updateCameraFOV(value);
    });

    // Distance slider
    const distanceSlider = document.getElementById('distance-slider');
    const distanceValue = document.getElementById('distance-value');
    distanceSlider.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        distanceValue.textContent = value.toFixed(1);
        updateCameraDistance(value);
    });

    // Camera Z slider
    const cameraZSlider = document.getElementById('camera-z-slider');
    const cameraZValue = document.getElementById('camera-z-value');
    cameraZSlider.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        cameraZValue.textContent = value.toFixed(1);
        updateCameraZ(value);
    });

    // Min Azimuth slider
    const minAzimuthSlider = document.getElementById('min-azimuth-slider');
    const minAzimuthValue = document.getElementById('min-azimuth-value');
    minAzimuthSlider.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        minAzimuthValue.textContent = value;
        updateMinAzimuth(value);
    });

    // Max Azimuth slider
    const maxAzimuthSlider = document.getElementById('max-azimuth-slider');
    const maxAzimuthValue = document.getElementById('max-azimuth-value');
    maxAzimuthSlider.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        maxAzimuthValue.textContent = value;
        updateMaxAzimuth(value);
    });

    // Min Polar slider
    const minPolarSlider = document.getElementById('min-polar-slider');
    const minPolarValue = document.getElementById('min-polar-value');
    minPolarSlider.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        minPolarValue.textContent = value;
        updateMinPolar(value);
    });

    // Max Polar slider
    const maxPolarSlider = document.getElementById('max-polar-slider');
    const maxPolarValue = document.getElementById('max-polar-value');
    maxPolarSlider.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        maxPolarValue.textContent = value;
        updateMaxPolar(value);
    });

    // Splat X slider
    const splatXSlider = document.getElementById('splat-x-slider');
    const splatXValue = document.getElementById('splat-x-value');
    splatXSlider.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        splatXValue.textContent = value.toFixed(1);
        updateSplatX(value);
    });

    // Splat Y slider
    const splatYSlider = document.getElementById('splat-y-slider');
    const splatYValue = document.getElementById('splat-y-value');
    splatYSlider.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        splatYValue.textContent = value.toFixed(1);
        updateSplatY(value);
    });

    // Splat Z slider
    const splatZSlider = document.getElementById('splat-z-slider');
    const splatZValue = document.getElementById('splat-z-value');
    splatZSlider.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        splatZValue.textContent = value.toFixed(1);
        updateSplatZ(value);
    });

    // Reset button
    const resetButton = document.getElementById('reset-debug');
    if (resetButton) {
        resetButton.addEventListener('click', resetDebugSettings);
    }

    // Copy button
    const copyButton = document.getElementById('copy-debug');
    if (copyButton) {
        copyButton.addEventListener('click', copyDebugSettings);
    }

    console.log('Debug panel initialized - press D to toggle');
}
