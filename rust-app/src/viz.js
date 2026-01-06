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
    }

    updatePlaybackState(seqPos, seqLen) {
        this.currentSeqPos = seqPos;
        this.currentSeqLen = seqLen;
    }

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

    getAngle(time) {
        return (time / this.duration) * 2 * Math.PI - (Math.PI / 2); // Start at 12 o'clock
    }

    draw(activeBeatIndex = -1, activeSegmentIndex = -1) {
        if (!this.beats.length) return;

        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.width, this.height);

        // Reset Context State
        ctx.lineCap = "butt";

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
            ctx.lineWidth = 1;
            beat.jump_candidates.forEach(targetIdx => {
                const targetBeat = this.beats[targetIdx];
                const targetAngle = this.getAngle(targetBeat.start);
                const tx = this.centerX + Math.cos(targetAngle) * (this.radius - 20);
                const ty = this.centerY + Math.sin(targetAngle) * (this.radius - 20);

                const sx = this.centerX + Math.cos(angle) * (this.radius - 20);
                const sy = this.centerY + Math.sin(angle) * (this.radius - 20);

                ctx.beginPath();
                ctx.moveTo(sx, sy);
                // Bezier curve through center?
                ctx.quadraticCurveTo(this.centerX, this.centerY, tx, ty);
                ctx.strokeStyle = "rgba(0, 0, 0, 0.5)"; // Black for light background
                ctx.stroke();
            });
        }

        // 4. Draw Countdown Pulse Ring
        if (activeBeatIndex !== -1 && this.currentSeqLen > 0) {
            this.drawCountdown(ctx, this.currentSeqPos, this.currentSeqLen);
        }

        // 5. Draw Novelty Curve (Debug View) - Bottom 100px
        this.drawNoveltyCurve(ctx);
    }

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
}
