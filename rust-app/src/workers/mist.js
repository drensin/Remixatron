/**
 * @fileoverview Mist visualization module for the Infinite Jukebox.
 *
 * Creates a soft, layered mist effect using particle-based clouds.
 * Each cloud retains its spawn color, creating a beautiful color history
 * as the music progresses through different segments.
 *
 * Unlike the procedural WebGL fog shader, this system:
 * - Maintains color history (old clouds keep their segment color)
 * - Runs entirely in the worker thread
 * - Uses Canvas 2D for soft radial gradients
 *
 * @author Remixatron Team
 */

// =============================================================================
// MIST CONFIGURATION
// =============================================================================

/**
 * Maximum number of mist clouds to render.
 * Higher values = denser mist, more overlap, richer color blending.
 */
const MAX_MIST_CLOUDS = 80;

/**
 * Cloud lifespan in milliseconds.
 * Longer lifespan = more color history visible at once.
 */
const CLOUD_LIFESPAN_MS = 12000;

/**
 * Base alpha for cloud center (increased for visibility).
 */
const CLOUD_CENTER_ALPHA = 0.25;

/**
 * Size range for clouds (as fraction of screen dimension).
 */
const MIN_CLOUD_SIZE = 0.15;
const MAX_CLOUD_SIZE = 0.35;

// =============================================================================
// MIST STATE
// =============================================================================

/**
 * Array of active mist cloud objects.
 * @type {Array<{x: number, y: number, radius: number, color: number[], birthTime: number}>}
 */
let mistClouds = [];

/**
 * Canvas context for mist rendering (set during init).
 */
let mistCtx = null;

/**
 * Current canvas dimensions.
 */
let mistWidth = 0;
let mistHeight = 0;

// =============================================================================
// PUBLIC API
// =============================================================================

/**
 * Initializes the mist module with a canvas context.
 * @param {CanvasRenderingContext2D} ctx - The 2D context to render to.
 * @param {number} width - Canvas width in logical pixels.
 * @param {number} height - Canvas height in logical pixels.
 */
export function initMist(ctx, width, height) {
    mistCtx = ctx;
    mistWidth = width;
    mistHeight = height;
    mistClouds = [];
    console.log('[Mist] Initialized');
}

/**
 * Updates mist dimensions on resize.
 * @param {number} width - New canvas width.
 * @param {number} height - New canvas height.
 */
export function resizeMist(width, height) {
    mistWidth = width;
    mistHeight = height;
}

/**
 * Spawns a new mist cloud with the given color.
 * Called on beat events to add new clouds matching the current segment.
 * @param {number[]} color - RGB color array [r, g, b], each 0.0-1.0.
 */
export function spawnMistCloud(color) {
    if (mistClouds.length >= MAX_MIST_CLOUDS) {
        mistClouds.shift();  // Remove oldest cloud
    }

    // Random position (full screen coverage including corners)
    const x = Math.random();  // 0% to 100% of width
    const y = Math.random();  // 0% to 100% of height

    // Random size within range
    const sizeRange = MAX_CLOUD_SIZE - MIN_CLOUD_SIZE;
    const size = MIN_CLOUD_SIZE + Math.random() * sizeRange;
    const baseSize = Math.min(mistWidth, mistHeight);
    const radius = baseSize * size;

    mistClouds.push({
        x: x * mistWidth,
        y: y * mistHeight,
        radius,
        color,
        birthTime: performance.now()
    });

    console.log(`[Mist] Spawned cloud: pos=(${(x * mistWidth).toFixed(0)}, ${(y * mistHeight).toFixed(0)}), radius=${radius.toFixed(0)}, clouds=${mistClouds.length}`);
}

/**
 * Clears all mist clouds.
 */
export function clearMist() {
    mistClouds = [];
    if (mistCtx) {
        mistCtx.clearRect(0, 0, mistWidth, mistHeight);
    }
}

/**
 * Renders all mist clouds with proper fading based on age.
 * Should be called every frame when mist mode is active.
 */
export function renderMist() {
    if (!mistCtx || mistClouds.length === 0) return;

    const ctx = mistCtx;
    const now = performance.now();

    // Clear canvas
    ctx.clearRect(0, 0, mistWidth, mistHeight);

    // Use 'lighter' for true additive blending (works on transparent canvas)
    ctx.globalCompositeOperation = 'lighter';

    // Render each cloud (oldest first for proper layering)
    for (let i = 0; i < mistClouds.length; i++) {
        const cloud = mistClouds[i];
        const age = now - cloud.birthTime;

        // Skip if cloud has expired
        if (age > CLOUD_LIFESPAN_MS) {
            continue;
        }

        // Calculate fade (1.0 at birth, 0.0 at death)
        const lifeFraction = age / CLOUD_LIFESPAN_MS;
        const fade = 1.0 - lifeFraction;

        // Ease out the fade for smoother disappearance
        const easedFade = fade * fade;

        renderCloud(ctx, cloud, easedFade);
    }

    // Remove expired clouds
    mistClouds = mistClouds.filter(c => (now - c.birthTime) < CLOUD_LIFESPAN_MS);

    // Reset composite operation
    ctx.globalCompositeOperation = 'source-over';
}

// =============================================================================
// PRIVATE HELPERS
// =============================================================================

/**
 * Renders a single mist cloud with soft radial gradient.
 * @param {CanvasRenderingContext2D} ctx - Canvas context.
 * @param {Object} cloud - Cloud object with position, radius, color.
 * @param {number} fade - Fade multiplier (0.0 = invisible, 1.0 = full).
 */
function renderCloud(ctx, cloud, fade) {
    const { x, y, radius, color } = cloud;

    // Convert RGB floats to 0-255
    const r = Math.round(color[0] * 255);
    const g = Math.round(color[1] * 255);
    const b = Math.round(color[2] * 255);

    // Create ultra-soft radial gradient (no visible core)
    const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);

    const centerAlpha = CLOUD_CENTER_ALPHA * fade;
    const midAlpha = centerAlpha * 0.6;
    const outerAlpha = centerAlpha * 0.2;

    gradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${centerAlpha.toFixed(3)})`);
    gradient.addColorStop(0.3, `rgba(${r}, ${g}, ${b}, ${midAlpha.toFixed(3)})`);
    gradient.addColorStop(0.7, `rgba(${r}, ${g}, ${b}, ${outerAlpha.toFixed(3)})`);
    gradient.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);

    ctx.fillStyle = gradient;
    ctx.fillRect(x - radius, y - radius, radius * 2, radius * 2);
}
