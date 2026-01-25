/**
 * @fileoverview Lava Lamp visualization effect for beat-synced blob animation.
 * 
 * Renders rising blobs with radial gradients, drop shadows, and sinusoidal
 * wobble. Blobs spawn on each beat with size determined by energy and color
 * determined by spectral centroid.
 * 
 * @author Remixatron Team
 */

/**
 * Represents a single rising blob in the lava lamp visualization.
 */
class Blob {
    /**
     * Creates a new blob at the bottom of the screen.
     * @param {number} x - Initial X position (0.0-1.0 normalized).
     * @param {number} radius - Radius in pixels (scaled by energy).
     * @param {number[]} color - RGB color array [r, g, b], each 0.0-1.0.
     */
    constructor(x, radius, color) {
        this.x = x;                        // Normalized X (0-1)
        this.startX = x;                   // Original X for wobble calculation
        this.y = 1.1;                      // Start just below screen (normalized 0=top, 1=bottom)
        this.radius = radius;              // Base radius in px
        this.color = color;                // [r, g, b] floats
        this.phase = Math.random() * Math.PI * 2;  // Random phase for wobble
        this.breathPhase = Math.random() * Math.PI * 2;  // Random phase for breathing
        this.speed = 0.00015 + Math.random() * 0.0001;  // Rise speed (50% slower)
    }
}

/**
 * Manages the Lava Lamp background visualization.
 * Renders on a Canvas 2D context with radial gradients and shadows.
 */
export class LavaLampViz {
    /**
     * Creates a new LavaLampViz instance.
     * @param {HTMLCanvasElement} canvas - The canvas to render on.
     */
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.blobs = [];
        this.maxBlobs = 30;  // Performance cap

        // RGB color palette (same as viz.js moodPalette)
        this.palette = [
            [1.0, 0.0, 0.33],  // #FF0055
            [0.0, 1.0, 0.6],   // #00FF99
            [0.0, 0.8, 1.0],   // #00CCFF
            [1.0, 0.67, 0.0],  // #FFAA00
            [0.8, 0.0, 1.0],   // #CC00FF
            [1.0, 0.2, 0.0],   // #FF3300
            [0.67, 1.0, 0.0],  // #AAFF00
            [0.0, 0.33, 1.0],  // #0055FF
            [1.0, 0.0, 0.67],  // #FF00AA
            [0.0, 1.0, 0.33]   // #00FF55
        ];

        // Animation state
        this.lastUpdateTime = performance.now();
    }

    /**
     * Spawns a new blob at the bottom of the screen.
     * Called on each beat by the viz controller.
     * 
     * @param {number} energy - Normalized RMS energy (0.0-1.0), determines size.
     * @param {number[]} color - RGB color array [r, g, b], each 0.0-1.0.
     */
    spawnBlob(energy, color) {
        // Cap blob count
        if (this.blobs.length >= this.maxBlobs) {
            this.blobs.shift();  // Remove oldest
        }

        // Random X position with slight center bias
        const x = 0.2 + Math.random() * 0.6;

        // Radius scales with energy (10-40px range, smaller than original)
        const minRadius = 10;
        const maxRadius = 40;
        const radius = minRadius + energy * (maxRadius - minRadius);

        this.blobs.push(new Blob(x, radius, color));
    }

    /**
     * Updates blob positions and removes off-screen blobs.
     * Should be called at ~30fps.
     * 
     * @param {number} dt - Delta time in milliseconds since last update.
     */
    update(dt) {
        // Update each blob
        for (const blob of this.blobs) {
            // Rise upward (decrease y since 0 = top)
            blob.y -= blob.speed * dt;

            // Sinusoidal wobble
            const wobbleAmplitude = 0.05;  // 5% of screen width
            blob.x = blob.startX + Math.sin(blob.y * 8 + blob.phase) * wobbleAmplitude;
        }

        // Remove blobs that have floated off the top
        this.blobs = this.blobs.filter(b => b.y > -0.2);
    }

    /**
     * Renders all blobs to the canvas.
     * Uses radial gradients, drop shadows, and breathing animation.
     */
    render() {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;

        // Clear canvas (transparent or let shader show through)
        ctx.clearRect(0, 0, w, h);

        const now = performance.now();

        for (const blob of this.blobs) {
            // Convert normalized coords to pixels
            const px = blob.x * w;
            const py = blob.y * h;

            // Breathing: oscillate radius by ±10%
            const breathScale = 1.0 + 0.1 * Math.sin(now * 0.003 + blob.breathPhase);
            const r = blob.radius * breathScale;

            // Skip if off-screen
            if (py < -r || py > h + r) continue;

            // Convert color to CSS - brighter center, fading to transparent edge
            const colorBright = `rgba(${Math.floor(blob.color[0] * 255)}, ${Math.floor(blob.color[1] * 255)}, ${Math.floor(blob.color[2] * 255)}, 1)`;
            const colorMid = `rgba(${Math.floor(blob.color[0] * 230)}, ${Math.floor(blob.color[1] * 230)}, ${Math.floor(blob.color[2] * 230)}, 0.8)`;
            const colorEdge = `rgba(${Math.floor(blob.color[0] * 200)}, ${Math.floor(blob.color[1] * 200)}, ${Math.floor(blob.color[2] * 200)}, 0)`;

            // Subtle glow shadow (colored, not black)
            ctx.save();
            ctx.shadowBlur = 20;
            ctx.shadowColor = `rgba(${Math.floor(blob.color[0] * 255)}, ${Math.floor(blob.color[1] * 255)}, ${Math.floor(blob.color[2] * 255)}, 0.5)`;
            ctx.shadowOffsetX = 0;
            ctx.shadowOffsetY = 0;

            // Radial gradient: bright center -> base color -> transparent edge
            const gradient = ctx.createRadialGradient(
                px, py, 0,       // Center
                px, py, r        // Edge
            );
            gradient.addColorStop(0, colorBright);    // Center: bright saturated color
            gradient.addColorStop(0.5, colorMid);     // Mid: slightly less bright
            gradient.addColorStop(1, colorEdge);      // Edge: fade to transparent

            // Draw blob
            ctx.beginPath();
            ctx.arc(px, py, r, 0, Math.PI * 2);
            ctx.fillStyle = gradient;
            ctx.fill();

            ctx.restore();
        }

        ctx.globalAlpha = 1.0;
    }

    /**
     * Clears all blobs from the visualization.
     */
    clear() {
        this.blobs = [];
    }

    /**
     * Resizes the canvas for HiDPI rendering.
     * @param {number} width - CSS width.
     * @param {number} height - CSS height.
     * @param {number} dpr - Device pixel ratio.
     */
    resize(width, height, dpr) {
        this.canvas.width = Math.floor(width * dpr);
        this.canvas.height = Math.floor(height * dpr);
        this.ctx.scale(dpr, dpr);
    }
}
