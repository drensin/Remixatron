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
 * 
 * @author Remixatron Team
 */

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
    }

    updatePlaybackState(seqPos, seqLen) {
        this.currentSeqPos = seqPos;
        this.currentSeqLen = seqLen;
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
        if (!this.beats.length) return;

        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.width, this.height);

        // Reset Context State
        ctx.lineCap = "butt";

        // 0.5. Draw Waveform Ring (Inner, behind segment ring)
        this.drawWaveformRing(ctx, activeSegmentIndex);

        // 1. Draw Segments (Outer Ring)
        ctx.lineWidth = 15;
        this.segments.forEach((seg, i) => {
            const startAngle = this.getAngle(seg.start_time);
            const endAngle = this.getAngle(seg.end_time);

            ctx.beginPath();
            ctx.arc(this.centerX, this.centerY, this.radius, startAngle, endAngle);
            ctx.strokeStyle = this.colors[seg.label % this.colors.length];

            // Dim non-active segments?
            ctx.globalAlpha = (activeSegmentIndex === -1 || activeSegmentIndex === i) ? 1.0 : 0.3;
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

            // 3. Draw Jump Connections (Arcs)
            // Draw lines to all possible jump candidates from this beat
            // Each arc is colored by its TARGET segment to show where it leads
            ctx.lineWidth = 1.5;
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

                ctx.beginPath();
                ctx.moveTo(sx, sy);
                // Bezier curve through center
                ctx.quadraticCurveTo(this.centerX, this.centerY, tx, ty);
                ctx.strokeStyle = targetColor;
                ctx.globalAlpha = 0.5;
                ctx.stroke();
                ctx.globalAlpha = 1.0;
            });
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
}
