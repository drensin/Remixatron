/**
 * @fileoverview Web Worker for offscreen Canvas 2D rendering.
 *
 * This worker handles all computationally expensive 2D canvas operations:
 * - Beat map visualization (segments, waveform ring)
 * - Lava lamp physics and rendering
 * - Mist cloud particle system
 *
 * The worker owns two OffscreenCanvas instances (jukebox + background) and runs
 * an independent 60fps render loop, freeing the main thread for UI responsiveness.
 *
 * @author Remixatron Team
 */

import { initMist, resizeMist, spawnMistCloud, renderMist, clearMist } from './mist.js';

// =============================================================================
// BLOB CLASS (Lava Lamp)
// =============================================================================

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
        this.y = 1.1;                      // Start just below screen (0=top, 1=bottom)
        this.radius = radius;              // Base radius in px
        this.color = color;                // [r, g, b] floats
        this.phase = Math.random() * Math.PI * 2;  // Random phase for wobble
        this.breathPhase = Math.random() * Math.PI * 2;  // Random phase for breathing
        this.speed = 0.000075 + Math.random() * 0.00005;  // Rise speed (halved for relaxed feel)
    }
}

// =============================================================================
// WORKER STATE
// =============================================================================

// Canvases and contexts
let jukeboxCanvas = null;
let lavaCanvas = null;
let jukeboxCtx = null;
let lavaCtx = null;

// Dimensions (logical, before DPR scaling)
let width = 0;
let height = 0;
let dpr = 1;

// Jukebox geometry
let centerX = 0;
let centerY = 0;
let radius = 0;

// Visualization data (immutable after setData)
let beats = [];
let segments = [];
let duration = 0;
let waveformEnvelope = [];

// Color palettes
const colors = [
    '#FF0055', '#00FF99', '#00CCFF', '#FFAA00', '#CC00FF',
    '#FF3300', '#AAFF00', '#0055FF', '#FF00AA', '#00FF55'
];

const moodPalette = [
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

// Dynamic playback state
let activeBeatIndex = -1;
let activeSegmentIndex = -1;
let currentSeqPos = 0;
let currentSeqLen = 0;

// Animation timing for smooth cursor interpolation
let tickReceivedAt = 0;
let currentBeatStart = 0;
let currentBeatDuration = 0.5;

// Lava lamp state
let blobs = [];
const MAX_BLOBS = 30;

// Visualization mode: 'none' | 'fog' | 'lava' | 'mist'
let vizMode = 'none';

// Waveform ring configuration
const WAVEFORM_WIDTH_RATIO = 1.5;
const WAVEFORM_GAP = 3;
const WAVEFORM_MIN_AMPLITUDE = 0.1;

// =============================================================================
// MESSAGE HANDLERS
// =============================================================================

self.onmessage = (e) => {
    const { type } = e.data;

    switch (type) {
        case 'init':
            handleInit(e.data);
            break;
        case 'setData':
            handleSetData(e.data);
            break;
        case 'setWaveform':
            handleSetWaveform(e.data);
            break;
        case 'tick':
            handleTick(e.data);
            break;
        case 'spawnBlob':
            handleSpawnBlob(e.data);
            break;
        case 'spawnCloud':
            handleSpawnCloud(e.data);
            break;
        case 'resize':
            handleResize(e.data);
            break;
        case 'setVizMode':
            vizMode = e.data.mode;
            break;
        case 'clear':
            handleClear();
            break;
        default:
            console.warn('[Worker] Unknown message type:', type);
    }
};

/**
 * Initializes canvases and starts the render loop.
 * @param {Object} data - Init payload with canvases and dimensions.
 */
function handleInit(data) {
    jukeboxCanvas = data.jukeboxCanvas;
    lavaCanvas = data.lavaCanvas;

    jukeboxCtx = jukeboxCanvas.getContext('2d');
    lavaCtx = lavaCanvas.getContext('2d');

    handleResize(data);
    startRenderLoop();

    // Initialize mist module with the same canvas as lava (background layer)
    initMist(lavaCtx, width, height);

    console.log('[Worker] Initialized with dual canvases + mist');
}

/**
 * Stores visualization data for rendering.
 * @param {Object} data - Beats, segments, duration.
 */
function handleSetData(data) {
    beats = data.beats || [];
    segments = data.segments || [];
    duration = data.duration || 0;

    console.log(`[Worker] Loaded ${beats.length} beats, ${segments.length} segments`);
}

/**
 * Stores waveform envelope (Transferable ArrayBuffer).
 * @param {Object} data - Contains envelope ArrayBuffer.
 */
function handleSetWaveform(data) {
    if (data.envelope instanceof ArrayBuffer) {
        waveformEnvelope = new Float32Array(data.envelope);
    } else {
        waveformEnvelope = data.envelope || [];
    }
    console.log(`[Worker] Loaded waveform: ${waveformEnvelope.length} samples`);
}

/**
 * Updates playback position from main thread tick.
 * @param {Object} data - Beat index, segment index, sequence state, and timing.
 */
function handleTick(data) {
    activeBeatIndex = data.beatIndex;
    activeSegmentIndex = data.segmentIndex;
    currentSeqPos = data.seqPos;
    currentSeqLen = data.seqLen;

    // Use main thread's timestamp for reference
    tickReceivedAt = data.tickTime || performance.now();

    if (activeBeatIndex >= 0 && activeBeatIndex < beats.length) {
        const beat = beats[activeBeatIndex];
        currentBeatStart = beat.start;
        currentBeatDuration = beat.duration || 0.5;
    }

    // NOTE: No immediate render — RAF loop handles segments/waveform (non-sync-critical).
    // Cursor/countdown/arcs are rendered on main thread sync-overlay for tight sync.
}

/**
 * Spawns a new lava lamp blob.
 * @param {Object} data - Energy and color for the blob.
 */
function handleSpawnBlob(data) {
    if (blobs.length >= MAX_BLOBS) {
        blobs.shift();  // Remove oldest
    }

    const x = 0.2 + Math.random() * 0.6;  // Center-biased X
    // Larger radius range to match original visual appearance
    const minRadius = 50;
    const maxRadius = 150;
    // Use ?? to handle energy=0 correctly (|| would treat 0 as falsy)
    const energy = data.energy ?? 0.5;
    const r = minRadius + energy * (maxRadius - minRadius);

    console.log(`[Worker] Spawning blob with energy=${energy}, radius=${r}`);
    blobs.push(new Blob(x, r, data.color ?? [1, 1, 1]));
}

/**
 * Spawns a new mist cloud.
 * @param {Object} data - Color for the cloud.
 */
function handleSpawnCloud(data) {
    spawnMistCloud(data.color ?? [1, 1, 1]);
}

/**
 * Handles canvas resize with HiDPI support.
 * @param {Object} data - Width, height, device pixel ratio.
 */
function handleResize(data) {
    width = data.width;
    height = data.height;
    dpr = data.dpr || 1;

    // Resize jukebox canvas
    jukeboxCanvas.width = width * dpr;
    jukeboxCanvas.height = height * dpr;
    jukeboxCtx.setTransform(1, 0, 0, 1, 0, 0);  // Reset transform
    jukeboxCtx.scale(dpr, dpr);

    // Resize lava canvas (also used for mist)
    lavaCanvas.width = width * dpr;
    lavaCanvas.height = height * dpr;
    lavaCtx.setTransform(1, 0, 0, 1, 0, 0);
    lavaCtx.scale(dpr, dpr);

    // Recalculate jukebox geometry
    centerX = width / 2;
    centerY = height / 2;
    radius = Math.min(width, height) / 2.2;

    // Update mist module dimensions
    resizeMist(width, height);

    console.log(`[Worker] Resized to ${width}x${height} @${dpr}x`);
}

/**
 * Clears all state for new track.
 */
function handleClear() {
    beats = [];
    segments = [];
    duration = 0;
    waveformEnvelope = [];
    blobs = [];
    activeBeatIndex = -1;
    activeSegmentIndex = -1;
    currentSeqPos = 0;
    currentSeqLen = 0;

    // Clear mist
    clearMist();

    // Clear canvases
    jukeboxCtx.clearRect(0, 0, width, height);
    lavaCtx.clearRect(0, 0, width, height);
}

// =============================================================================
// RENDER LOOP
// =============================================================================

let lastFrameTime = 0;
const TARGET_FPS = 60;
const FRAME_INTERVAL = 1000 / TARGET_FPS;

/**
 * Starts the independent 60fps render loop.
 */
function startRenderLoop() {
    function frame(now) {
        const dt = now - lastFrameTime;

        if (dt >= FRAME_INTERVAL) {
            lastFrameTime = now;

            // 1. Background layer: lava or mist (mutually exclusive)
            if (vizMode === 'lava') {
                updateBlobs(dt);
                renderLavaLamp();
            } else if (vizMode === 'mist') {
                renderMist();
            } else {
                // Clear background canvas when not in lava/mist mode
                lavaCtx.clearRect(0, 0, width, height);
            }

            // 2. Jukebox: always render (core visualization)
            renderJukebox();
        }

        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
    console.log('[Worker] Render loop started');
}

// =============================================================================
// JUKEBOX RENDERING
// =============================================================================

/**
 * Main jukebox render function.
 * Draws segments and waveform only.
 * (Cursor, arcs, and countdown are now rendered on main thread sync-overlay)
 */
function renderJukebox() {
    if (beats.length === 0) return;

    const ctx = jukeboxCtx;
    ctx.clearRect(0, 0, width, height);
    ctx.lineCap = 'butt';

    // 1. Waveform ring (behind segments)
    drawWaveformRing(ctx);

    // 2. Segment arcs (neon glow - 3 pass)
    drawSegments(ctx);

    // NOTE: Cursor, jump arcs, and countdown are now rendered on main thread
    // sync-overlay canvas for tight audio synchronization.
}

/**
 * Converts time to radial angle (12 o'clock = start).
 * @param {number} time - Time in seconds.
 * @returns {number} Angle in radians.
 */
function getAngle(time) {
    return (time / duration) * 2 * Math.PI - (Math.PI / 2);
}

/**
 * Draws segment arcs with 3-pass neon glow effect.
 * @param {CanvasRenderingContext2D} ctx - Canvas context.
 */
function drawSegments(ctx) {
    for (let i = 0; i < segments.length; i++) {
        const seg = segments[i];
        const startAngle = getAngle(seg.start_time);
        const endAngle = getAngle(seg.end_time);
        const color = colors[seg.label % colors.length];
        const isActive = (activeSegmentIndex === -1 || activeSegmentIndex === i);

        const drawArc = () => {
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius, startAngle, endAngle);
        };

        // Pass 1: Large blur (background glow)
        ctx.lineWidth = 20;
        ctx.shadowBlur = 25;
        ctx.shadowColor = color;
        ctx.strokeStyle = color;
        ctx.globalAlpha = isActive ? 0.25 : 0.08;
        drawArc();
        ctx.stroke();

        // Pass 2: Medium blur
        ctx.lineWidth = 15;
        ctx.shadowBlur = 10;
        ctx.globalAlpha = isActive ? 0.5 : 0.15;
        drawArc();
        ctx.stroke();

        // Pass 3: Sharp core
        ctx.lineWidth = 8;
        ctx.shadowBlur = 0;
        ctx.globalAlpha = isActive ? 1.0 : 0.4;
        drawArc();
        ctx.stroke();

        ctx.globalAlpha = 1.0;
    }
}

/**
 * Draws waveform amplitude ring inside segment ring.
 * @param {CanvasRenderingContext2D} ctx - Canvas context.
 */
function drawWaveformRing(ctx) {
    if (waveformEnvelope.length === 0 || segments.length === 0) return;

    const segmentRingWidth = 15;
    const waveformRingWidth = segmentRingWidth * WAVEFORM_WIDTH_RATIO;
    const baselineRadius = radius - segmentRingWidth / 2 - WAVEFORM_GAP - waveformRingWidth / 2;

    const numBars = waveformEnvelope.length;
    const barWidth = 1.5;

    // Helper: find segment at time
    const getSegmentAtTime = (time) => {
        for (let i = 0; i < segments.length; i++) {
            if (time >= segments[i].start_time && time < segments[i].end_time) {
                return i;
            }
        }
        return -1;
    };

    for (let i = 0; i < numBars; i++) {
        const normalizedPos = i / numBars;
        const angle = normalizedPos * 2 * Math.PI - (Math.PI / 2);
        const time = normalizedPos * duration;

        let amp = waveformEnvelope[i] || 0;
        amp = Math.max(amp, WAVEFORM_MIN_AMPLITUDE);

        const barHalfHeight = (waveformRingWidth / 2) * amp;
        const innerRadius = baselineRadius - barHalfHeight;
        const outerRadius = baselineRadius + barHalfHeight;

        const segIdx = getSegmentAtTime(time);
        const isActive = (segIdx !== -1) && (activeSegmentIndex === -1 || activeSegmentIndex === segIdx);

        const innerX = centerX + Math.cos(angle) * innerRadius;
        const innerY = centerY + Math.sin(angle) * innerRadius;
        const outerX = centerX + Math.cos(angle) * outerRadius;
        const outerY = centerY + Math.sin(angle) * outerRadius;

        ctx.beginPath();
        ctx.moveTo(innerX, innerY);
        ctx.lineTo(outerX, outerY);
        ctx.lineWidth = barWidth;
        ctx.lineCap = 'round';

        if (isActive) {
            ctx.shadowBlur = 6;
            ctx.shadowColor = '#FFFFFF';
            ctx.globalAlpha = 1.0;
            ctx.strokeStyle = '#FFFFFF';
        } else {
            ctx.shadowBlur = 0;
            ctx.globalAlpha = 0.35;
            ctx.strokeStyle = '#AAAAAA';
        }

        ctx.stroke();
    }

    ctx.shadowBlur = 0;
    ctx.globalAlpha = 1.0;
    ctx.lineCap = 'butt';
}

/**
 * Draws the playback cursor (white glowing dot).
 * @param {CanvasRenderingContext2D} ctx - Canvas context.
 * @param {Object} beat - Current beat object.
 */
function drawCursor(ctx, beat) {
    // Smooth cursor interpolation
    const elapsedMs = performance.now() - tickReceivedAt;
    const elapsedSec = elapsedMs / 1000.0;
    const beatProgress = Math.min(elapsedSec / currentBeatDuration, 1.0);

    const currentAngle = getAngle(currentBeatStart);
    const nextAngle = getAngle(currentBeatStart + currentBeatDuration);
    const interpolatedAngle = currentAngle + (nextAngle - currentAngle) * beatProgress;

    const x = centerX + Math.cos(interpolatedAngle) * radius;
    const y = centerY + Math.sin(interpolatedAngle) * radius;

    ctx.shadowBlur = 15;
    ctx.shadowColor = '#FFFFFF';
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, 2 * Math.PI);
    ctx.fillStyle = '#FFFFFF';
    ctx.fill();
    ctx.shadowBlur = 0;
}

/**
 * Draws jump arc connections from current beat to candidates.
 * @param {CanvasRenderingContext2D} ctx - Canvas context.
 * @param {Object} beat - Current beat with jump_candidates.
 */
function drawJumpArcs(ctx, beat) {
    const angle = getAngle(beat.start);
    const sx = centerX + Math.cos(angle) * (radius - 20);
    const sy = centerY + Math.sin(angle) * (radius - 20);

    ctx.globalCompositeOperation = 'screen';

    for (const targetIdx of beat.jump_candidates) {
        if (targetIdx >= beats.length) continue;

        const targetBeat = beats[targetIdx];
        const targetAngle = getAngle(targetBeat.start);
        const tx = centerX + Math.cos(targetAngle) * (radius - 20);
        const ty = centerY + Math.sin(targetAngle) * (radius - 20);

        const targetSegIdx = targetBeat.segment;
        const targetSeg = segments[targetSegIdx];
        const targetColor = targetSeg
            ? colors[targetSeg.label % colors.length]
            : '#FFFFFF';

        const drawCurve = () => {
            ctx.beginPath();
            ctx.moveTo(sx, sy);
            ctx.quadraticCurveTo(centerX, centerY, tx, ty);
        };

        // Pass 1: Glow
        ctx.lineWidth = 4;
        ctx.shadowBlur = 10;
        ctx.shadowColor = targetColor;
        ctx.strokeStyle = targetColor;
        ctx.globalAlpha = 0.3;
        drawCurve();
        ctx.stroke();

        // Pass 2: Core
        ctx.lineWidth = 1.5;
        ctx.shadowBlur = 0;
        ctx.globalAlpha = 0.7;
        drawCurve();
        ctx.stroke();
    }

    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1.0;
}

/**
 * Draws countdown ring showing beats until branch.
 * @param {CanvasRenderingContext2D} ctx - Canvas context.
 */
function drawCountdown(ctx) {
    const cx = centerX;
    const cy = centerY;
    const remaining = currentSeqLen - currentSeqPos;

    // Color based on urgency
    let color = '#00CCFF';  // Safe cyan
    if (remaining <= 4) color = '#FFAA00';  // Warn orange
    if (remaining <= 1) color = '#FF0055';  // Imminent pink

    // Background ring
    ctx.beginPath();
    ctx.arc(cx, cy, 40, 0, Math.PI * 2);
    ctx.lineWidth = 4;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.stroke();

    // Progress ring (unwinding)
    const pct = currentSeqPos / currentSeqLen;
    const startAngle = -Math.PI / 2;
    const endAngle = startAngle + (Math.PI * 2 * (1.0 - pct));

    ctx.beginPath();
    ctx.arc(cx, cy, 40, startAngle, endAngle, false);
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Number
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = color;
    ctx.font = 'bold 24px sans-serif';
    ctx.fillText(remaining.toString(), cx, cy - 2);

    // Label
    ctx.font = '8px sans-serif';
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.fillText('BRANCH', cx, cy + 12);
}

// =============================================================================
// LAVA LAMP RENDERING
// =============================================================================

/**
 * Updates blob physics (rise, wobble).
 * @param {number} dt - Delta time in milliseconds.
 */
function updateBlobs(dt) {
    for (const blob of blobs) {
        blob.y -= blob.speed * dt;

        const wobbleAmplitude = 0.05;
        blob.x = blob.startX + Math.sin(blob.y * 8 + blob.phase) * wobbleAmplitude;
    }

    // Remove off-screen blobs
    blobs = blobs.filter(b => b.y > -0.2);
}

/**
 * Renders lava lamp blobs with radial gradients and shadows.
 */
function renderLavaLamp() {
    const ctx = lavaCtx;
    const w = width;
    const h = height;

    ctx.clearRect(0, 0, w, h);

    const now = performance.now();

    for (const blob of blobs) {
        const px = blob.x * w;
        const py = blob.y * h;

        // Breathing animation
        const breathScale = 1.0 + 0.1 * Math.sin(now * 0.003 + blob.breathPhase);
        const r = blob.radius * breathScale;

        if (py < -r || py > h + r) continue;

        // Color stops
        const c = blob.color;
        const colorBright = `rgba(${Math.floor(c[0] * 255)}, ${Math.floor(c[1] * 255)}, ${Math.floor(c[2] * 255)}, 1)`;
        const colorMid = `rgba(${Math.floor(c[0] * 230)}, ${Math.floor(c[1] * 230)}, ${Math.floor(c[2] * 230)}, 0.8)`;
        const colorEdge = `rgba(${Math.floor(c[0] * 200)}, ${Math.floor(c[1] * 200)}, ${Math.floor(c[2] * 200)}, 0)`;

        ctx.save();
        ctx.shadowBlur = 20;
        ctx.shadowColor = `rgba(${Math.floor(c[0] * 255)}, ${Math.floor(c[1] * 255)}, ${Math.floor(c[2] * 255)}, 0.5)`;
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 0;

        const gradient = ctx.createRadialGradient(px, py, 0, px, py, r);
        gradient.addColorStop(0, colorBright);
        gradient.addColorStop(0.5, colorMid);
        gradient.addColorStop(1, colorEdge);

        ctx.beginPath();
        ctx.arc(px, py, r, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.fill();
        ctx.restore();
    }

    ctx.globalAlpha = 1.0;
}
