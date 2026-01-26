/**
 * @fileoverview Hybrid visualization coordinator for the Infinite Jukebox.
 *
 * HYBRID ARCHITECTURE:
 * - Worker thread: Segments, waveform ring, lava lamp (expensive, not sync-critical)
 * - Main thread: Cursor, countdown, jump arcs (cheap, sync-critical)
 *
 * The worker renders to an OffscreenCanvas (jukebox-canvas), while the main
 * thread renders sync-critical elements to a separate overlay canvas (sync-overlay).
 *
 * @author Remixatron Team
 */

import { MoodShader } from './MoodShader.js';

/**
 * Coordinates hybrid visualization rendering.
 */
export class InfiniteJukeboxViz {
    /**
     * Creates a new InfiniteJukeboxViz instance.
     * @param {HTMLCanvasElement} canvas - The main jukebox canvas (will be transferred to worker).
     */
    constructor(canvas) {
        this.canvas = canvas;
        this.beats = [];
        this.segments = [];
        this.duration = 0;

        // Color palette
        this.colors = [
            '#FF0055', '#00FF99', '#00CCFF', '#FFAA00', '#CC00FF',
            '#FF3300', '#AAFF00', '#0055FF', '#FF00AA', '#00FF55'
        ];

        // RGB color palette for mood shader (floats)
        this.moodPalette = [
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

        // Playback state
        this.activeBeatIndex = -1;
        this.activeSegmentIndex = -1;
        this.currentSeqPos = 0;
        this.currentSeqLen = 0;
        this.smoothedEnergy = 0.5;
        this.smoothedCentroid = 0.5;
        this.currentMoodColor = [0.3, 0.3, 0.3];
        this.tickReceivedAt = 0;
        this.currentBeatStart = 0;
        this.currentBeatDuration = 0.5;

        // Visualization mode
        this.vizMode = 'none';
        this.lavaBeatCounter = 0;

        // ─────────────────────────────────────────────────────────────────────
        // SYNC OVERLAY CANVAS (Main Thread - cursor, countdown, arcs)
        // ─────────────────────────────────────────────────────────────────────
        this.syncOverlay = document.getElementById('sync-overlay');
        if (this.syncOverlay) {
            this.syncCtx = this.syncOverlay.getContext('2d');
        } else {
            console.error('[Viz] sync-overlay canvas not found!');
            this.syncCtx = null;
        }

        // Get dimensions
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.parentElement.getBoundingClientRect();
        this.width = rect.width;
        this.height = rect.height;
        this.dpr = dpr;
        this.centerX = rect.width / 2;
        this.centerY = rect.height / 2;
        this.radius = Math.min(rect.width, rect.height) / 2.2;

        // Initialize sync overlay canvas size
        if (this.syncOverlay) {
            this.syncOverlay.width = rect.width * dpr;
            this.syncOverlay.height = rect.height * dpr;
            this.syncCtx.scale(dpr, dpr);
        }

        // ─────────────────────────────────────────────────────────────────────
        // WORKER (segments, waveform, lava lamp)
        // ─────────────────────────────────────────────────────────────────────
        const lavaCanvas = document.getElementById('lava-lamp');

        // Transfer both canvases to worker
        const offscreenJukebox = canvas.transferControlToOffscreen();
        const offscreenLava = lavaCanvas ? lavaCanvas.transferControlToOffscreen() : null;

        this.worker = new Worker(
            new URL('./workers/renderer.worker.js', import.meta.url),
            { type: 'module' }
        );

        const transferables = [offscreenJukebox];
        if (offscreenLava) transferables.push(offscreenLava);

        this.worker.postMessage({
            type: 'init',
            jukeboxCanvas: offscreenJukebox,
            lavaCanvas: offscreenLava,
            width: rect.width,
            height: rect.height,
            dpr
        }, transferables);

        // Listen for resize events
        window.addEventListener('resize', () => this.resize());

        // ─────────────────────────────────────────────────────────────────────
        // WebGL MoodShader (stays on main thread)
        // ─────────────────────────────────────────────────────────────────────
        const bgCanvas = document.getElementById('bg-shader');
        if (bgCanvas) {
            this.moodShader = new MoodShader(bgCanvas);
            if (this.moodShader.initWebGL()) {
                this._startMoodAnimationLoop();
            } else {
                console.warn('[Viz] MoodShader WebGL init failed');
                this.moodShader = null;
            }
        } else {
            this.moodShader = null;
        }

        console.log('[Viz] Initialized with hybrid rendering (worker + main thread overlay)');
    }

    /**
     * Updates the sequence state for countdown display.
     * @param {number} seqPos - Current position in sequence.
     * @param {number} seqLen - Total sequence length.
     */
    updatePlaybackState(seqPos, seqLen) {
        this.currentSeqPos = seqPos;
        this.currentSeqLen = seqLen;
    }

    /**
     * Sets the background visualization mode.
     * @param {'none' | 'fog' | 'lava' | 'mist'} mode - The visualization mode.
     */
    setVizMode(mode) {
        this.vizMode = mode;
        console.log(`[Viz] Mode set to: ${mode}`);

        this.worker.postMessage({ type: 'setVizMode', mode });

        const lavaEl = document.getElementById('lava-lamp');
        const fogEl = document.getElementById('bg-shader');

        // Lava canvas is used for both lava and mist modes
        if (lavaEl) lavaEl.style.display = (mode === 'lava' || mode === 'mist') ? 'block' : 'none';
        if (fogEl) fogEl.style.display = (mode === 'fog') ? 'block' : 'none';
    }

    /**
     * Clears the visualization and resets state.
     */
    clear() {
        this.beats = [];
        this.segments = [];
        this.duration = 0;
        this.currentSeqPos = 0;
        this.currentSeqLen = 0;
        this.activeBeatIndex = -1;
        this.activeSegmentIndex = -1;

        // Clear sync overlay
        if (this.syncCtx) {
            this.syncCtx.clearRect(0, 0, this.width, this.height);
        }

        this.worker.postMessage({ type: 'clear' });
    }

    /**
     * Handles canvas resize with HiDPI support.
     */
    resize() {
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.parentElement.getBoundingClientRect();

        this.width = rect.width;
        this.height = rect.height;
        this.dpr = dpr;
        this.centerX = rect.width / 2;
        this.centerY = rect.height / 2;
        this.radius = Math.min(rect.width, rect.height) / 2.2;

        // Resize sync overlay
        if (this.syncOverlay) {
            this.syncOverlay.width = rect.width * dpr;
            this.syncOverlay.height = rect.height * dpr;
            this.syncCtx.setTransform(1, 0, 0, 1, 0, 0);
            this.syncCtx.scale(dpr, dpr);
        }

        // Forward to worker
        this.worker.postMessage({
            type: 'resize',
            width: rect.width,
            height: rect.height,
            dpr
        });

        // Resize MoodShader
        if (this.moodShader) {
            this.moodShader.resize(rect.width, rect.height, dpr);
        }
    }

    /**
     * Updates visualization data with new analysis result.
     * @param {Object} payload - Analysis result from backend.
     */
    setData(payload) {
        this.beats = payload.beats || [];
        this.segments = payload.segments || [];

        if (this.beats.length > 0) {
            const lastBeat = this.beats[this.beats.length - 1];
            this.duration = lastBeat.start + lastBeat.duration;
        } else {
            this.duration = 0;
        }

        console.log(`[Viz] Loaded ${this.beats.length} beats, ${this.duration.toFixed(2)}s`);

        this.worker.postMessage({
            type: 'setData',
            beats: payload.beats,
            segments: payload.segments,
            duration: this.duration,
            colors: this.colors,
            moodPalette: this.moodPalette
        });
    }

    /**
     * Sets the waveform amplitude envelope.
     * @param {Array<number>} envelope - ~720 normalized amplitude values.
     */
    setWaveformEnvelope(envelope) {
        const buffer = new Float32Array(envelope).buffer;
        this.worker.postMessage({ type: 'setWaveform', envelope: buffer }, [buffer]);
        console.log(`[Viz] Sent waveform: ${envelope.length} samples`);
    }

    /**
     * Converts time to radial angle (12 o'clock = start).
     * @param {number} time - Time in seconds.
     * @returns {number} Angle in radians.
     */
    getAngle(time) {
        return (time / this.duration) * 2 * Math.PI - (Math.PI / 2);
    }

    /**
     * Main draw call - renders sync-critical elements on main thread.
     * @param {number} activeBeatIndex - Current beat index.
     * @param {number} activeSegmentIndex - Current segment index.
     */
    draw(activeBeatIndex = -1, activeSegmentIndex = -1) {
        this.activeBeatIndex = activeBeatIndex;
        this.activeSegmentIndex = activeSegmentIndex;
        this.tickReceivedAt = performance.now();

        if (activeBeatIndex >= 0 && activeBeatIndex < this.beats.length) {
            const beat = this.beats[activeBeatIndex];
            this.currentBeatStart = beat.start;
            this.currentBeatDuration = beat.duration || 0.5;

            // Spawn lava blob every 4th beat
            if (this.vizMode === 'lava') {
                this.lavaBeatCounter++;
                if (this.lavaBeatCounter >= 4) {
                    this.lavaBeatCounter = 0;
                    const energy = beat.energy ?? 0.5;
                    const cluster = beat.cluster ?? 0;
                    const color = this.moodPalette[cluster % this.moodPalette.length];

                    this.worker.postMessage({
                        type: 'spawnBlob',
                        energy,
                        color
                    });
                }
            }

            // Spawn mist cloud every beat for dense mist effect
            if (this.vizMode === 'mist') {
                const cluster = beat.cluster ?? 0;
                const color = this.moodPalette[cluster % this.moodPalette.length];

                this.worker.postMessage({
                    type: 'spawnCloud',
                    color
                });
            }
        }

        // Forward segment/waveform state to worker
        this.worker.postMessage({
            type: 'tick',
            beatIndex: activeBeatIndex,
            segmentIndex: activeSegmentIndex,
            seqPos: this.currentSeqPos,
            seqLen: this.currentSeqLen,
            tickTime: this.tickReceivedAt
        });

        // SYNC-CRITICAL: Draw cursor, countdown, and arcs on main thread
        this._drawSyncOverlay();
    }

    /**
     * Draws sync-critical elements (cursor, countdown, arcs) on main thread overlay.
     * @private
     */
    _drawSyncOverlay() {
        if (!this.syncCtx || this.beats.length === 0) return;

        const ctx = this.syncCtx;
        ctx.clearRect(0, 0, this.width, this.height);

        if (this.activeBeatIndex < 0 || this.activeBeatIndex >= this.beats.length) return;

        const beat = this.beats[this.activeBeatIndex];

        // 1. Draw Cursor
        this._drawCursor(ctx, beat);

        // 2. Draw Jump Arcs
        this._drawJumpArcs(ctx, beat);

        // 3. Draw Countdown
        if (this.currentSeqLen > 0) {
            this._drawCountdown(ctx);
        }
    }

    /**
     * Draws the playback cursor.
     * @private
     */
    _drawCursor(ctx, beat) {
        const angle = this.getAngle(beat.start);
        const x = this.centerX + Math.cos(angle) * this.radius;
        const y = this.centerY + Math.sin(angle) * this.radius;

        ctx.shadowBlur = 15;
        ctx.shadowColor = '#FFFFFF';
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, 2 * Math.PI);
        ctx.fillStyle = '#FFFFFF';
        ctx.fill();
        ctx.shadowBlur = 0;
    }

    /**
     * Draws jump arc connections.
     * @private
     */
    _drawJumpArcs(ctx, beat) {
        if (!beat.jump_candidates || beat.jump_candidates.length === 0) return;

        const angle = this.getAngle(beat.start);
        const sx = this.centerX + Math.cos(angle) * (this.radius - 20);
        const sy = this.centerY + Math.sin(angle) * (this.radius - 20);

        ctx.globalCompositeOperation = 'screen';

        for (const targetIdx of beat.jump_candidates) {
            if (targetIdx >= this.beats.length) continue;

            const targetBeat = this.beats[targetIdx];
            const targetAngle = this.getAngle(targetBeat.start);
            const tx = this.centerX + Math.cos(targetAngle) * (this.radius - 20);
            const ty = this.centerY + Math.sin(targetAngle) * (this.radius - 20);

            const targetSegIdx = targetBeat.segment;
            const targetSeg = this.segments[targetSegIdx];
            const targetColor = targetSeg
                ? this.colors[targetSeg.label % this.colors.length]
                : '#FFFFFF';

            // Pass 1: Glow
            ctx.lineWidth = 4;
            ctx.shadowBlur = 10;
            ctx.shadowColor = targetColor;
            ctx.strokeStyle = targetColor;
            ctx.globalAlpha = 0.3;
            ctx.beginPath();
            ctx.moveTo(sx, sy);
            ctx.quadraticCurveTo(this.centerX, this.centerY, tx, ty);
            ctx.stroke();

            // Pass 2: Core
            ctx.lineWidth = 1.5;
            ctx.shadowBlur = 0;
            ctx.globalAlpha = 0.7;
            ctx.beginPath();
            ctx.moveTo(sx, sy);
            ctx.quadraticCurveTo(this.centerX, this.centerY, tx, ty);
            ctx.stroke();
        }

        ctx.globalCompositeOperation = 'source-over';
        ctx.globalAlpha = 1.0;
    }

    /**
     * Draws countdown ring.
     * @private
     */
    _drawCountdown(ctx) {
        const cx = this.centerX;
        const cy = this.centerY;
        const remaining = this.currentSeqLen - this.currentSeqPos;

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
        const pct = this.currentSeqPos / this.currentSeqLen;
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

    // ─────────────────────────────────────────────────────────────────────
    // WebGL MOOD SHADER ANIMATION LOOP
    // ─────────────────────────────────────────────────────────────────────

    /**
     * Starts the WebGL mood shader animation loop.
     * @private
     */
    _startMoodAnimationLoop() {
        const SMOOTHING_FACTOR = 0.85;
        const TARGET_FPS = 30;
        const FRAME_INTERVAL = 1000 / TARGET_FPS;
        let lastFrameTime = 0;

        const loop = () => {
            const now = performance.now();

            if (now - lastFrameTime < FRAME_INTERVAL) {
                requestAnimationFrame(loop);
                return;
            }
            lastFrameTime = now;

            if (this.vizMode !== 'fog' || !this.moodShader) {
                requestAnimationFrame(loop);
                return;
            }

            const elapsedMs = performance.now() - this.tickReceivedAt;
            const elapsedSec = elapsedMs / 1000.0;
            const beatProgress = Math.min(elapsedSec / this.currentBeatDuration, 1.0);

            let currentEnergy = 0.5, nextEnergy = 0.5;
            let currentCentroid = 0.5, nextCentroid = 0.5;
            let targetNovelty = 0.0;
            let currentColor = [0.3, 0.3, 0.3];
            let nextColor = [0.3, 0.3, 0.3];

            if (this.beats.length > 0 && this.activeBeatIndex >= 0 && this.activeBeatIndex < this.beats.length) {
                const beat = this.beats[this.activeBeatIndex];

                currentEnergy = beat.energy ?? 0.5;
                currentCentroid = beat.centroid ?? 0.5;
                targetNovelty = beat.novelty ?? 0.0;

                const cluster = beat.cluster ?? 0;
                currentColor = this.moodPalette[cluster % this.moodPalette.length];

                const nextBeatIdx = this.activeBeatIndex + 1;
                if (nextBeatIdx < this.beats.length) {
                    const nextBeat = this.beats[nextBeatIdx];
                    nextEnergy = nextBeat.energy ?? 0.5;
                    nextCentroid = nextBeat.centroid ?? 0.5;
                    const nextCluster = nextBeat.cluster ?? 0;
                    nextColor = this.moodPalette[nextCluster % this.moodPalette.length];
                } else {
                    nextEnergy = currentEnergy;
                    nextCentroid = currentCentroid;
                    nextColor = currentColor;
                }
            }

            const lerp = (a, b, t) => a + (b - a) * t;
            const targetEnergy = lerp(currentEnergy, nextEnergy, beatProgress);
            const targetCentroid = lerp(currentCentroid, nextCentroid, beatProgress);
            const targetColor = [
                lerp(currentColor[0], nextColor[0], beatProgress),
                lerp(currentColor[1], nextColor[1], beatProgress),
                lerp(currentColor[2], nextColor[2], beatProgress)
            ];

            this.smoothedEnergy = this.smoothedEnergy * SMOOTHING_FACTOR + targetEnergy * (1 - SMOOTHING_FACTOR);
            this.smoothedCentroid = this.smoothedCentroid * SMOOTHING_FACTOR + targetCentroid * (1 - SMOOTHING_FACTOR);

            for (let i = 0; i < 3; i++) {
                this.currentMoodColor[i] = this.currentMoodColor[i] * SMOOTHING_FACTOR + targetColor[i] * (1 - SMOOTHING_FACTOR);
            }

            const time = performance.now() / 1000.0;
            this.moodShader.render(
                time,
                this.smoothedEnergy,
                this.smoothedCentroid,
                this.currentMoodColor,
                targetNovelty
            );

            requestAnimationFrame(loop);
        };

        loop();
        console.log('[Viz] MoodShader loop started (main thread)');
    }
}
