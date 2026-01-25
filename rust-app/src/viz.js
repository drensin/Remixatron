/**
 * @fileoverview Visualization engine for the Infinite Jukebox.
 * 
 * Renders the circular "beat map" on an HTML5 Canvas.
 * Supports:
 * - High DPI rendering
 * - Dynamic segment coloring
 * - Beat cursor tracking
 * - Jump arc rendering
 * - Countdown pulse ring for branches
 * - Debug overlay for Novelty Curve
 * - WebGL Mood Shader background
 * 
 * @author Remixatron Team
 */

import { MoodShader } from './MoodShader.js';
import { LavaLampViz } from './LavaLampViz.js';

/**
 * Manages the HTML5 Canvas rendering context for the musical visualization.
 */
export class InfiniteJukeboxViz {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.beats = [];
        this.segments = [];
        this.duration = 0;
        this.resize();

        window.addEventListener('resize', () => this.resize());

        // Color Palette (Vibrant)
        this.colors = [
            '#FF0055', '#00FF99', '#00CCFF', '#FFAA00', '#CC00FF',
            '#FF3300', '#AAFF00', '#0055FF', '#FF00AA', '#00FF55'
        ];

        // Waveform Ring Configuration
        this.waveformEnvelope = [];       // 720 amplitude values (0.0-1.0)
        this.WAVEFORM_WIDTH_RATIO = 1.5;  // 150% of segment ring width
        this.WAVEFORM_GAP = 3;            // px between segment ring and waveform
        this.WAVEFORM_MIN_AMPLITUDE = 0.1; // Floor for quiet sections
        this.WAVEFORM_OPACITY = 0.4;       // Base opacity for non-active segments
        this.WAVEFORM_ACTIVE_OPACITY = 0.7; // Opacity for current segment

        // ─────────────────────────────────────────────────────────────────────
        // MOOD SHADER STATE (Phase 4: WebGL Background)
        // ─────────────────────────────────────────────────────────────────────

        // RGB color palette for mood shader (matches segment colors but as floats)
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

        // Lag smoothing state (for silky transitions)
        this.activeBeatIndex = -1;        // Current beat for animation loop access
        this.smoothedEnergy = 0.5;        // Lag-smoothed RMS energy
        this.smoothedCentroid = 0.5;      // Lag-smoothed spectral centroid
        this.currentMoodColor = [0.3, 0.3, 0.3];  // Current interpolated color

        // Tick timing state (for inter-beat interpolation)
        this.tickReceivedAt = 0;          // performance.now() when playback_tick arrived
        this.currentBeatStart = 0;        // beat.start time of the current beat
        this.currentBeatDuration = 0.5;   // Duration of current beat (fallback 0.5s)

        // Initialize WebGL Mood Shader
        const bgCanvas = document.getElementById('bg-shader');
        if (bgCanvas) {
            this.moodShader = new MoodShader(bgCanvas);
            if (this.moodShader.initWebGL()) {
                this._startMoodAnimationLoop();
            } else {
                console.warn('[Viz] MoodShader WebGL init failed, falling back to solid background');
                this.moodShader = null;
            }
        } else {
            this.moodShader = null;
        }

        // Initialize Lava Lamp Viz (separate 2D canvas)
        const lavaCanvas = document.getElementById('lava-lamp');
        if (lavaCanvas) {
            this.lavaLamp = new LavaLampViz(lavaCanvas);
            this.lavaCanvas = lavaCanvas;
        } else {
            this.lavaLamp = null;
            this.lavaCanvas = null;
        }

        // Visualization mode: 'none' | 'fog' | 'lava'
        this.vizMode = 'none';  // Default to none
        this.lavaBeatCounter = 0;  // Counter for lava lamp spawn rate (every 4th beat)
    }

    updatePlaybackState(seqPos, seqLen) {
        this.currentSeqPos = seqPos;
        this.currentSeqLen = seqLen;
    }

    /**
     * Sets the background visualization mode.
     * @param {'none' | 'fog' | 'lava'} mode - The visualization mode to switch to.
     */
    setVizMode(mode) {
        this.vizMode = mode;
        console.log(`[Viz] Background mode set to: ${mode}`);

        // Clear lava lamp blobs when switching away
        if (mode !== 'lava' && this.lavaLamp) {
            this.lavaLamp.clear();
        }
    }

    /**
     * Clears the visualization canvas and resets all state.
     * 
     * Should be called when starting a new track to ensure the previous
     * visualization doesn't persist during loading.
     */
    clear() {
        // Reset data
        this.beats = [];
        this.segments = [];
        this.novelty = null;
        this.peaks = [];
        this.duration = 0;
        this.currentSeqPos = 0;
        this.currentSeqLen = 0;

        // Clear canvas
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.width, this.height);
    }

    /**
     * Resizes the canvas to match the parent container (HiDPI aware).
     * 
     * Calculates the radius, center point, and scaling factor for the visualization.
     * Should be called on window resize.
     */
    resize() {
        // High DPI Support
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.parentElement.getBoundingClientRect();

        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;

        this.canvas.style.width = `${rect.width}px`;
        this.canvas.style.height = `${rect.height}px`;

        this.ctx.scale(dpr, dpr);
        this.width = rect.width;
        this.height = rect.height;
        this.radius = Math.min(this.width, this.height) / 2.2;
        this.centerX = this.width / 2;
        this.centerY = this.height / 2;

        // Resize MoodShader canvas to match
        if (this.moodShader) {
            this.moodShader.resize(rect.width, rect.height, dpr);
        }

        this.draw(); // Redraw on resize
    }

    /**
     * Updates the visualization data with a new Analysis Result.
     * 
     * @param {Object} payload - The structure payload from the backend.
     * @param {Array} payload.beats - List of Beat objects.
     * @param {Array} payload.segments - List of Segment objects.
     * @param {Array} payload.novelty_curve - Novelty score array.
     * @param {Array} payload.peaks - Indices of detected boundaries.
     */
    setData(payload) {
        this.beats = payload.beats;
        this.segments = payload.segments;
        this.novelty = payload.novelty_curve;
        this.peaks = payload.peaks;

        // Calculate Angles
        // Assuming last beat end timestamp is total duration approximately
        if (this.beats.length > 0) {
            const lastBeat = this.beats[this.beats.length - 1];
            this.duration = lastBeat.start + lastBeat.duration;
        }

        console.log(`Loaded Visualization: ${this.beats.length} beats, ${this.duration.toFixed(2)}s`);
        this.draw();
    }

    /**
     * Sets the waveform amplitude envelope for the inner ring visualization.
     * 
     * @param {Array<number>} envelope - Array of ~720 normalized amplitude values (0.0-1.0).
     */
    setWaveformEnvelope(envelope) {
        this.waveformEnvelope = envelope;
        console.log(`Loaded waveform envelope: ${envelope.length} samples`);
    }

    getAngle(time) {
        return (time / this.duration) * 2 * Math.PI - (Math.PI / 2); // Start at 12 o'clock
    }

    /**
     * Main render loop.
     * 
     * Draws the visualization frame based on the current active beat.
     * Layer Order:
     * 1. Clear Screen
     * 2. Segments Ring (Static base)
     * 3. Beating Cursor (Active position)
     * 4. Jump Arcs (Future possibilities)
     * 5. Countdown Ring (If near a branch)
     * 6. Debug Graphs (Novelty curve)
     * 
     * @param {number} activeBeatIndex - The global index of the beat currently playing.
     * @param {number} activeSegmentIndex - The segment index to highlight (optional).
     */
    draw(activeBeatIndex = -1, activeSegmentIndex = -1) {
        // Track active beat for animation loop to access asynchronously
        this.activeBeatIndex = activeBeatIndex;

        // Record timing for inter-beat interpolation
        // The animation loop uses this to estimate current audio time
        this.tickReceivedAt = performance.now();
        if (activeBeatIndex >= 0 && activeBeatIndex < this.beats.length) {
            const beat = this.beats[activeBeatIndex];
            this.currentBeatStart = beat.start;
            this.currentBeatDuration = beat.duration || 0.5;

            // Spawn lava lamp blob every 4th beat (if in lava mode)
            if (this.vizMode === 'lava' && this.lavaLamp) {
                this.lavaBeatCounter++;
                if (this.lavaBeatCounter >= 4) {
                    this.lavaBeatCounter = 0;
                    const energy = beat.energy ?? 0.5;
                    // Use segment color (from cluster) instead of centroid for visual consistency
                    const cluster = beat.cluster ?? 0;
                    const segmentColor = this.moodPalette[cluster % this.moodPalette.length];
                    this.lavaLamp.spawnBlob(energy, segmentColor);
                }
            }
        }

        if (!this.beats.length) return;

        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.width, this.height);

        // Reset Context State
        ctx.lineCap = "butt";

        // 0.5. Draw Waveform Ring (Inner, behind segment ring)
        this.drawWaveformRing(ctx, activeSegmentIndex);

        // 1. Draw Segments (Outer Ring) - Neon Glow Effect
        // Three-pass rendering: background glow → medium glow → sharp core
        this.segments.forEach((seg, i) => {
            const startAngle = this.getAngle(seg.start_time);
            const endAngle = this.getAngle(seg.end_time);
            const color = this.colors[seg.label % this.colors.length];
            const isActive = (activeSegmentIndex === -1 || activeSegmentIndex === i);

            // Path definition (reused for all passes)
            const drawArc = () => {
                ctx.beginPath();
                ctx.arc(this.centerX, this.centerY, this.radius, startAngle, endAngle);
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
        });

        // 2. Draw Beating Cursor
        if (activeBeatIndex !== -1 && activeBeatIndex < this.beats.length) {
            const beat = this.beats[activeBeatIndex];
            const angle = this.getAngle(beat.start); // Interpolate later for smoothness?

            const x = this.centerX + Math.cos(angle) * this.radius;
            const y = this.centerY + Math.sin(angle) * this.radius;

            // Glow
            ctx.shadowBlur = 15;
            ctx.shadowColor = "#FFFFFF";

            ctx.beginPath();
            ctx.arc(x, y, 8, 0, 2 * Math.PI);
            ctx.fillStyle = "#FFFFFF";
            ctx.fill();

            ctx.shadowBlur = 0;

            // 3. Draw Jump Connections (Arcs) - Neon Glow with Screen Blend
            // Screen blend makes arcs additive/glowing against the shader background
            ctx.globalCompositeOperation = 'screen';

            beat.jump_candidates.forEach(targetIdx => {
                // Skip invalid indices (may occur if beats were truncated)
                if (targetIdx >= this.beats.length) return;

                const targetBeat = this.beats[targetIdx];
                const targetAngle = this.getAngle(targetBeat.start);
                const tx = this.centerX + Math.cos(targetAngle) * (this.radius - 20);
                const ty = this.centerY + Math.sin(targetAngle) * (this.radius - 20);

                const sx = this.centerX + Math.cos(angle) * (this.radius - 20);
                const sy = this.centerY + Math.sin(angle) * (this.radius - 20);

                // Get target segment's color
                const targetSegIdx = targetBeat.segment;
                const targetSeg = this.segments[targetSegIdx];
                const targetColor = targetSeg
                    ? this.colors[targetSeg.label % this.colors.length]
                    : "#FFFFFF";

                // Path definition for arc
                const drawCurve = () => {
                    ctx.beginPath();
                    ctx.moveTo(sx, sy);
                    ctx.quadraticCurveTo(this.centerX, this.centerY, tx, ty);
                };

                // Pass 1: Glow background
                ctx.lineWidth = 4;
                ctx.shadowBlur = 10;
                ctx.shadowColor = targetColor;
                ctx.strokeStyle = targetColor;
                ctx.globalAlpha = 0.3;
                drawCurve();
                ctx.stroke();

                // Pass 2: Sharp core
                ctx.lineWidth = 1.5;
                ctx.shadowBlur = 0;
                ctx.globalAlpha = 0.7;
                drawCurve();
                ctx.stroke();
            });

            ctx.globalCompositeOperation = 'source-over';
            ctx.globalAlpha = 1.0;
        }

        // 4. Draw Countdown Pulse Ring
        if (activeBeatIndex !== -1 && this.currentSeqLen > 0) {
            this.drawCountdown(ctx, this.currentSeqPos, this.currentSeqLen);
        }

        // 5. Draw Novelty Curve (Debug View) - Bottom 100px
        // DISABLED: Data is still emitted from backend if needed later
        // this.drawNoveltyCurve(ctx);
    }

    /**
     * Renders the Novelty Curve debug graph at the bottom of the canvas.
     * 
     * Used to verify that the spectral clustering and boundary detection
     * align with the visual segments.
     * 
     * @param {CanvasRenderingContext2D} ctx - The canvas context.
     */
    drawNoveltyCurve(ctx) {
        if (!this.novelty || !this.novelty.length) return;

        const h = 100; // Height of graph
        const yBase = this.height - 10;
        const w = this.width;

        // Normalize curve
        const maxVal = Math.max(...this.novelty);

        // Draw Background
        ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
        ctx.fillRect(0, this.height - h - 20, w, h + 20);

        // Draw Curve
        ctx.beginPath();
        ctx.strokeStyle = "#FFFFFF";
        ctx.lineWidth = 2;

        for (let i = 0; i < this.novelty.length; i++) {
            const x = (i / this.novelty.length) * w;
            const val = this.novelty[i];
            const y = yBase - (val / maxVal) * h;

            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Draw Detected Peaks (Red Lines)
        ctx.lineWidth = 1;
        ctx.strokeStyle = "#FF0000";
        this.peaks.forEach(beatIdx => {
            const x = (beatIdx / this.novelty.length) * w;
            ctx.beginPath();
            ctx.moveTo(x, yBase);
            ctx.lineTo(x, yBase - h);
            ctx.stroke();
        });

        // Draw Actual Segment Boundaries (Blue Dots) from 'segments' to verify alignment
        ctx.fillStyle = "#00CCFF";
        this.segments.forEach(seg => {
            // Find beat index for start_time
            // Approx map: time -> beat index (linear scan or binary search, linear is fine for < 100 segments)
            const beatIdx = this.beats.findIndex(b => Math.abs(b.start - seg.start_time) < 0.1);
            if (beatIdx !== -1) {
                const x = (beatIdx / this.novelty.length) * w;
                ctx.beginPath();
                ctx.arc(x, yBase + 5, 3, 0, Math.PI * 2);
                ctx.fill();
            }
        });
    }

    /**
     * Draws the "Branch Countdown" ring.
     * 
     * Visualizes the time remaining until a potential jump event.
     * The ring "unwinds" counter-clockwise as the jump approaches.
     * 
     * @param {CanvasRenderingContext2D} ctx - The canvas context.
     * @param {number} pos - Current position in the sequence (0..total).
     * @param {number} total - Total length of the sequence.
     */
    drawCountdown(ctx, pos, total) {
        // Center of screen
        const cx = this.centerX;
        const cy = this.centerY;

        // Remaining beats
        const remaining = total - pos;

        // Color Logic
        let color = "#00CCFF"; // Safe Cyan
        if (remaining <= 4) color = "#FFAA00"; // Warn Orange
        if (remaining <= 1) color = "#FF0055"; // Jump Imminent (Hot Pink)

        // 1. Draw Ring Background (Dim)
        ctx.beginPath();
        ctx.arc(cx, cy, 40, 0, Math.PI * 2);
        ctx.lineWidth = 4;
        ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
        ctx.stroke();

        // 2. Draw Progress Ring (Counter-Clockwise Unwinding)
        // Start at top (-PI/2)
        // Percentage complete = pos / total
        const pct = pos / total;
        const startAngle = -Math.PI / 2;
        const endAngle = startAngle + (Math.PI * 2 * (1.0 - pct));

        ctx.beginPath();
        // Draw arc counter-clockwise? No, standard arc draws clockwise. 
        // To "unwind", we draw from Start to End where End decreases.
        ctx.arc(cx, cy, 40, startAngle, endAngle, false);
        ctx.strokeStyle = color;
        ctx.lineWidth = 4;
        ctx.lineCap = "round";
        ctx.stroke();

        // 3. Draw Number
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillStyle = color;
        ctx.font = "bold 24px sans-serif";
        ctx.fillText(remaining.toString(), cx, cy - 2);

        // 4. Draw Label
        ctx.font = "8px sans-serif";
        ctx.fillStyle = "rgba(255, 255, 255, 0.6)";
        ctx.fillText("BRANCH", cx, cy + 12);
    }

    /**
     * Draws the waveform amplitude ring inside the segment ring.
     * 
     * The waveform is drawn as radial bars (like a linear waveform bent into a circle):
     * - Each bar extends both inward and outward from a circular baseline
     * - Bar height is proportional to amplitude at that position
     * - Colors match the corresponding segment
     * - Current segment glows brighter than others
     * 
     * @param {CanvasRenderingContext2D} ctx - The canvas context.
     * @param {number} activeSegmentIndex - Index of current playing segment (-1 if none).
     */
    drawWaveformRing(ctx, activeSegmentIndex) {
        if (this.waveformEnvelope.length === 0 || this.segments.length === 0) {
            return;
        }

        // Calculate ring dimensions
        const segmentRingWidth = 15; // Must match segment ring lineWidth
        const waveformRingWidth = segmentRingWidth * this.WAVEFORM_WIDTH_RATIO;
        const baselineRadius = this.radius - segmentRingWidth / 2 - this.WAVEFORM_GAP - waveformRingWidth / 2;

        // Bar configuration
        const numBars = this.waveformEnvelope.length;
        const barWidth = 1.5; // Width of each radial bar in pixels

        // Pre-compute segment lookup for each bar position
        const getSegmentAtTime = (time) => {
            for (let i = 0; i < this.segments.length; i++) {
                if (time >= this.segments[i].start_time && time < this.segments[i].end_time) {
                    return i;
                }
            }
            return -1; // Return -1 if no segment matches (e.g. Intro gap)
        };

        // Draw each radial bar
        for (let i = 0; i < numBars; i++) {
            const normalizedPos = i / numBars;
            const angle = normalizedPos * 2 * Math.PI - (Math.PI / 2); // Start at 12 o'clock
            const time = normalizedPos * this.duration;

            // Get amplitude with floor
            let amp = this.waveformEnvelope[i] || 0;
            amp = Math.max(amp, this.WAVEFORM_MIN_AMPLITUDE);

            // Calculate bar extent (half goes in, half goes out)
            const barHalfHeight = (waveformRingWidth / 2) * amp;
            const innerRadius = baselineRadius - barHalfHeight;
            const outerRadius = baselineRadius + barHalfHeight;

            // Get segment for this position (for active detection)
            const segIdx = getSegmentAtTime(time);

            // Determine if this segment is active (for glow effect)
            // If segIdx is -1 (Intro/Gap), it is NEVER active.
            const isActive = (segIdx !== -1) && (activeSegmentIndex === -1 || activeSegmentIndex === segIdx);

            // Calculate start and end points of the radial bar
            const innerX = this.centerX + Math.cos(angle) * innerRadius;
            const innerY = this.centerY + Math.sin(angle) * innerRadius;
            const outerX = this.centerX + Math.cos(angle) * outerRadius;
            const outerY = this.centerY + Math.sin(angle) * outerRadius;

            // Draw the bar
            ctx.beginPath();
            ctx.moveTo(innerX, innerY);
            ctx.lineTo(outerX, outerY);

            ctx.lineWidth = barWidth;
            ctx.lineCap = "round";

            // Apply glow effect for active segment (white glow), otherwise dim gray
            if (isActive) {
                ctx.shadowBlur = 6;
                ctx.shadowColor = "#FFFFFF";
                ctx.globalAlpha = 1.0;
                ctx.strokeStyle = "#FFFFFF";
            } else {
                ctx.shadowBlur = 0;
                ctx.globalAlpha = 0.35;
                ctx.strokeStyle = "#AAAAAA";
            }

            ctx.stroke();
        }

        // Reset context state
        ctx.shadowBlur = 0;
        ctx.globalAlpha = 1.0;
        ctx.lineCap = "butt";
    }

    // ─────────────────────────────────────────────────────────────────────
    // MOOD SHADER ANIMATION LOOP
    // ─────────────────────────────────────────────────────────────────────

    /**
     * Starts the independent animation loop for the WebGL mood shader.
     * 
     * This loop runs at requestAnimationFrame rate (~60fps) independently of
     * the main `draw()` cycles. It reads the current beat state and applies
     * lag smoothing for silky visual transitions.
     * 
     * @private
     */
    _startMoodAnimationLoop() {
        const SMOOTHING_FACTOR = 0.85;  // Slightly lower for more responsive feel
        const TARGET_FPS = 30;
        const FRAME_INTERVAL = 1000 / TARGET_FPS;  // ~33ms between frames
        let lastFrameTime = 0;

        const loop = () => {
            const now = performance.now();

            // Throttle to 30fps to reduce CPU usage
            if (now - lastFrameTime < FRAME_INTERVAL) {
                requestAnimationFrame(loop);
                return;
            }
            lastFrameTime = now;

            if (!this.moodShader) return;

            // 1. Compute inter-beat progress using tick timing
            // This estimates where we are within the current beat based on elapsed time
            const elapsedMs = performance.now() - this.tickReceivedAt;
            const elapsedSec = elapsedMs / 1000.0;
            const beatProgress = Math.min(elapsedSec / this.currentBeatDuration, 1.0);

            // 2. Get current and next beat values for interpolation
            let currentEnergy = 0.5, nextEnergy = 0.5;
            let currentCentroid = 0.5, nextCentroid = 0.5;
            let targetNovelty = 0.0;
            let currentColor = [0.3, 0.3, 0.3];
            let nextColor = [0.3, 0.3, 0.3];

            if (this.beats.length > 0 && this.activeBeatIndex >= 0 && this.activeBeatIndex < this.beats.length) {
                const beat = this.beats[this.activeBeatIndex];

                // Current beat values
                currentEnergy = beat.energy ?? 0.5;
                currentCentroid = beat.centroid ?? 0.5;
                targetNovelty = beat.novelty ?? 0.0;

                const cluster = beat.cluster ?? 0;
                currentColor = this.moodPalette[cluster % this.moodPalette.length];

                // Next beat values (for lerp target)
                const nextBeatIdx = this.activeBeatIndex + 1;
                if (nextBeatIdx < this.beats.length) {
                    const nextBeat = this.beats[nextBeatIdx];
                    nextEnergy = nextBeat.energy ?? 0.5;
                    nextCentroid = nextBeat.centroid ?? 0.5;
                    const nextCluster = nextBeat.cluster ?? 0;
                    nextColor = this.moodPalette[nextCluster % this.moodPalette.length];
                } else {
                    // Last beat - no interpolation target
                    nextEnergy = currentEnergy;
                    nextCentroid = currentCentroid;
                    nextColor = currentColor;
                }
            }

            // 3. Lerp between current and next beat based on progress
            const lerp = (a, b, t) => a + (b - a) * t;
            const targetEnergy = lerp(currentEnergy, nextEnergy, beatProgress);
            const targetCentroid = lerp(currentCentroid, nextCentroid, beatProgress);
            const targetColor = [
                lerp(currentColor[0], nextColor[0], beatProgress),
                lerp(currentColor[1], nextColor[1], beatProgress),
                lerp(currentColor[2], nextColor[2], beatProgress)
            ];

            // 4. Apply lag smoothing on top of lerped values (optional extra smoothness)
            this.smoothedEnergy = this.smoothedEnergy * SMOOTHING_FACTOR + targetEnergy * (1 - SMOOTHING_FACTOR);
            this.smoothedCentroid = this.smoothedCentroid * SMOOTHING_FACTOR + targetCentroid * (1 - SMOOTHING_FACTOR);

            for (let i = 0; i < 3; i++) {
                this.currentMoodColor[i] = this.currentMoodColor[i] * SMOOTHING_FACTOR + targetColor[i] * (1 - SMOOTHING_FACTOR);
            }

            // Novelty is NOT smoothed (instant flash on section boundaries)

            // 5. Render based on current viz mode
            const time = performance.now() / 1000.0;
            const dt = FRAME_INTERVAL;  // Delta time for lava lamp update

            if (this.vizMode === 'fog' && this.moodShader) {
                // Fog mode: render WebGL shader
                const bgCanvas = document.getElementById('bg-shader');
                if (bgCanvas) bgCanvas.style.display = 'block';

                this.moodShader.render(
                    time,
                    this.smoothedEnergy,
                    this.smoothedCentroid,
                    this.currentMoodColor,
                    targetNovelty
                );
                // Hide lava canvas when showing fog
                if (this.lavaCanvas) this.lavaCanvas.style.display = 'none';
            } else if (this.vizMode === 'lava' && this.lavaLamp) {
                // Lava mode: update and render 2D blobs
                // Show lava canvas, hide WebGL canvas
                if (this.lavaCanvas) this.lavaCanvas.style.display = 'block';
                const bgCanvas = document.getElementById('bg-shader');
                if (bgCanvas) bgCanvas.style.display = 'none';

                this.lavaLamp.update(dt);
                this.lavaLamp.render();
            } else if (this.vizMode === 'none') {
                // None mode: hide both background canvases
                if (this.lavaCanvas) this.lavaCanvas.style.display = 'none';
                const bgCanvas = document.getElementById('bg-shader');
                if (bgCanvas) bgCanvas.style.display = 'none';
            }

            requestAnimationFrame(loop);
        };

        loop();
        console.log('[Viz] MoodShader animation loop started');
    }
}
