// Simple PLY Loader for Three.js
// Based on Three.js PLYLoader but simplified for our use case

import * as THREE from 'three';

export class PLYLoader {
    constructor() {}

    async load(url) {
        const response = await fetch(url);
        const text = await response.text();
        return this.parse(text);
    }

    parse(data) {
        const geometry = new THREE.BufferGeometry();
        
        // Parse PLY header
        const headerEnd = data.indexOf('end_header\n') + 11;
        const header = data.substring(0, headerEnd);
        const body = data.substring(headerEnd);
        
        // Extract vertex count
        const vertexMatch = header.match(/element vertex (\d+)/);
        if (!vertexMatch) {
            console.error('Invalid PLY file: no vertex count');
            return geometry;
        }
        
        const vertexCount = parseInt(vertexMatch[1]);
        console.log(`PLY file has ${vertexCount} vertices`);
        
        // Parse vertices (assuming ASCII format for simplicity)
        const lines = body.trim().split('\n');
        const positions = [];
        const colors = [];
        
        for (let i = 0; i < Math.min(vertexCount, lines.length); i++) {
            const parts = lines[i].trim().split(/\s+/);
            if (parts.length >= 3) {
                // Position (x, y, z)
                positions.push(
                    parseFloat(parts[0]),
                    parseFloat(parts[1]),
                    parseFloat(parts[2])
                );
                
                // Color if available (r, g, b)
                if (parts.length >= 6) {
                    colors.push(
                        parseInt(parts[3]) / 255,
                        parseInt(parts[4]) / 255,
                        parseInt(parts[5]) / 255
                    );
                } else {
                    // Default color
                    colors.push(0.5, 0.5, 1.0);
                }
            }
        }
        
        // Set geometry attributes
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        
        // Compute bounding sphere for proper camera positioning
        geometry.computeBoundingSphere();
        
        return geometry;
    }
}