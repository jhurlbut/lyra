import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { SplatMesh } from '@sparkjsdev/spark';

// Viewer state
let scene, camera, renderer, controls;
let splatMesh = null;
let animationId = null;

export function initViewer() {
    const container = document.getElementById('viewer-container');

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);

    // Camera
    camera = new THREE.PerspectiveCamera(
        75,
        container.clientWidth / container.clientHeight,
        0.1,
        1000
    );
    camera.position.set(0, 0, 5);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
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
}

function animate() {
    animationId = requestAnimationFrame(animate);

    controls.update();
    renderer.render(scene, camera);
}

function onWindowResize() {
    const container = document.getElementById('viewer-container');

    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();

    renderer.setSize(container.clientWidth, container.clientHeight);
}

export async function loadPLY(url) {
    try {
        // Remove existing splat mesh
        if (splatMesh) {
            scene.remove(splatMesh);
            splatMesh = null;
        }

        // Load new splat mesh
        splatMesh = new SplatMesh({ url });

        // Wait for splat to load
        await new Promise((resolve, reject) => {
            const checkLoaded = setInterval(() => {
                if (splatMesh.ready) {
                    clearInterval(checkLoaded);
                    resolve();
                }
            }, 100);

            // Timeout after 30 seconds
            setTimeout(() => {
                clearInterval(checkLoaded);
                reject(new Error('Timeout loading PLY file'));
            }, 30000);
        });

        // Add to scene
        scene.add(splatMesh);

        // Center the splat mesh
        centerMesh(splatMesh);

        // Reset camera to view the mesh
        fitCameraToMesh(splatMesh);

        console.log('PLY loaded successfully');

    } catch (error) {
        console.error('Error loading PLY:', error);
        throw error;
    }
}

function centerMesh(mesh) {
    // Compute bounding box
    const box = new THREE.Box3().setFromObject(mesh);
    const center = box.getCenter(new THREE.Vector3());

    // Center mesh
    mesh.position.sub(center);
}

function fitCameraToMesh(mesh) {
    // Compute bounding box
    const box = new THREE.Box3().setFromObject(mesh);
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);

    // Calculate camera distance
    const fov = camera.fov * (Math.PI / 180);
    const cameraDistance = Math.abs(maxDim / Math.sin(fov / 2)) * 1.5;

    // Position camera
    camera.position.set(cameraDistance, cameraDistance, cameraDistance);
    camera.lookAt(0, 0, 0);

    // Update controls
    controls.target.set(0, 0, 0);
    controls.update();
}

export function resetCamera() {
    if (splatMesh) {
        fitCameraToMesh(splatMesh);
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
