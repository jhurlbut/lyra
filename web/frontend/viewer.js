import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';

// Viewer state
let scene, camera, renderer, controls;
let splatMesh = null;
let animationId = null;
let transformControls = null;

// Reference point system
let raycaster = null;
let mouse = new THREE.Vector2();
let referencePointMode = false;
let referencePoint = null;
let referencePointIndicator = null;

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

    // Camera
    camera = new THREE.PerspectiveCamera(
        75,
        width / height,
        0.1,
        1000
    );
    camera.position.set(0, 0, 5);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    // Controls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = false;
    controls.minDistance = 1;
    controls.maxDistance = 100;
    controls.maxPolarAngle = Math.PI;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 5, 5);
    scene.add(directionalLight);

    // Grid helper
    const gridHelper = new THREE.GridHelper(10, 10, 0x444444, 0x222222);
    scene.add(gridHelper);

    // Axes helper
    const axesHelper = new THREE.AxesHelper(5);
    scene.add(axesHelper);

    // Handle window resize
    window.addEventListener('resize', onWindowResize);

    // Start animation loop
    animate();
    
    // Initialize transform controls after everything is set up
    initTransformControls();
    
    // Initialize reference point system
    initReferencePointSystem();
}

function initTransformControls() {
    try {
        if (!camera || !renderer || !scene) {
            console.error('Cannot initialize TransformControls: missing camera, renderer, or scene');
            return;
        }
        
        // Based on official Three.js examples, TransformControls might extend Object3D
        // but in different versions or builds this can vary. Let's use a more robust approach.
        console.log('Creating TransformControls...');
        transformControls = new TransformControls(camera, renderer.domElement);
        
        // Modern Three.js TransformControls should extend Object3D, but let's handle edge cases
        console.log('TransformControls type:', typeof transformControls);
        console.log('TransformControls constructor:', transformControls.constructor.name);
        
        // Try direct addition first (this is the correct approach in most cases)
        scene.add(transformControls);
        console.log('TransformControls added to scene');
        
        // Set up event listeners
        transformControls.addEventListener('dragging-changed', (event) => {
            controls.enabled = !event.value;
        });
        
        transformControls.addEventListener('change', () => {
            renderer.render(scene, camera);
        });
        
        // Make transformControls available globally for UI controls
        window.transformControls = transformControls;
        
        console.log('TransformControls initialized successfully');
        
    } catch (error) {
        console.error('Error with TransformControls:', error);
        
        // Fallback: Try to work without TransformControls
        console.log('Continuing without TransformControls - manual transforms only');
        transformControls = null;
        window.transformControls = null;
    }
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
    animationId = requestAnimationFrame(animate);

    controls.update();
    renderer.render(scene, camera);
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

export async function loadPLY(url) {
    try {
        console.log('Loading Gaussian Splat PLY from:', url);
        
        // Remove existing mesh
        if (splatMesh) {
            scene.remove(splatMesh);
            splatMesh = null;
        }

        // Dynamically import SparkJS (ES module)
        const Spark = await import('@sparkjsdev/spark');
        console.log('SparkJS loaded:', Spark);

        // Create SplatMesh with the PLY URL and onLoad callback
        return new Promise((resolve, reject) => {
            splatMesh = new Spark.SplatMesh({ 
                url: url,
                onLoad: (mesh) => {
                    console.log('Splat mesh loaded successfully via onLoad callback!');
                    
                    // Add to scene first
                    scene.add(mesh);
                    console.log('SplatMesh added to scene');
                    
                    // Get accurate bounding box using SparkJS method
                    let box, center, size;
                    try {
                        // Use SparkJS getBoundingBox for accurate bounds
                        box = mesh.getBoundingBox(false); // false = include splat scales for accuracy
                        center = box.getCenter(new THREE.Vector3());
                        size = box.getSize(new THREE.Vector3());
                        console.log('Using SparkJS getBoundingBox');
                    } catch (e) {
                        // Fallback to Three.js method if SparkJS method fails
                        box = new THREE.Box3().setFromObject(mesh);
                        center = box.getCenter(new THREE.Vector3());
                        size = box.getSize(new THREE.Vector3());
                        console.log('Using Three.js Box3.setFromObject fallback');
                    }
                    
                    console.log('Bounding box:', { center, size });

                    // Note: Camera matrices now correctly generate OpenCV-convention data
                    // No rotation needed - data is correctly oriented at source

                    // Calculate appropriate scale - make it 5x bigger than before
                    const maxDim = Math.max(size.x, size.y, size.z);
                    const targetSize = 20; // 5x bigger than the original 4 units
                    const scaleFactor = maxDim > 0 ? targetSize / maxDim : 1;
                    mesh.scale.setScalar(scaleFactor);
                    
                    // Position the mesh so its bottom (min Y) is at Y=0 and back (max Z) is at Z=0
                    // First center it at origin
                    mesh.position.sub(center);
                    // Then shift it up by half the scaled height so bottom is at 0
                    const scaledHeight = size.y * scaleFactor;
                    mesh.position.y = scaledHeight / 2;
                    // And shift it forward by half the scaled depth so back is at 0
                    const scaledDepth = size.z * scaleFactor;
                    mesh.position.z = -scaledDepth / 2;
                    
                    console.log(`Scaled by ${scaleFactor} to fit target size of ${targetSize} units`);
                    console.log(`Positioned with bottom at Y=0, back at Z=0`);
                    console.log(`Shifted up by ${scaledHeight / 2}, forward by ${scaledDepth / 2}`);
                    
                    // Position camera for typical reconstruction viewpoint (closer and better angle)
                    const viewDistance = targetSize * 2.0;  // Closer than previous 2.5x
                    camera.position.set(0, scaledHeight * 0.3, viewDistance);
                    camera.lookAt(0, scaledHeight * 0.4, 0);  // Look at mid-height
                    
                    // Update controls target to center of model
                    controls.target.set(0, scaledHeight / 2, 0);
                    controls.update();
                    
                    // Attach transform controls to the mesh
                    if (transformControls) {
                        transformControls.attach(mesh);
                    }
                    
                    // Make viewer section visible
                    const viewerSection = document.getElementById('viewer-section');
                    if (viewerSection) {
                        viewerSection.style.display = 'block';
                    }
                    
                    console.log('Gaussian splat loaded and displayed with SparkJS!');
                    resolve(true);
                }
            });
            
            // Add error timeout
            setTimeout(() => {
                reject(new Error('Timeout loading splat mesh after 30 seconds'));
            }, 30000);
        });

    } catch (error) {
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
    }
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

export function disposeViewer() {
    if (animationId) {
        cancelAnimationFrame(animationId);
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
