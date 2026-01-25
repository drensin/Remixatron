# Implementation Plan: Remixatron Mood Visualizer

## Objective
Enhance the Remixatron **desktop** visualization with a reactive WebGL "Mood Shader" background that reflects the musical structure (Energy, Segment Identity, Novelty) in real-time, while maintaining the readability of the existing Jump Graph.

## Scope
**Desktop Application Only** (`src/viz.js`, `src/main.js`).  
The Chromecast `/receiver/` endpoint is explicitly out of scope for this phase.

---

## Architecture

### Dual Canvas Approach
1. **Background (`MoodShader.js`):** A WebGL canvas rendering the atmospheric shader.
2. **Foreground (`viz.js`):** The existing 2D canvas rendering beats, segments, and jump arcs.

### Data Flow
```
[Rust Backend] ──(WebSocket init)──► [viz.js] ──(per-frame)──► [MoodShader]
     │                                    │
     │ Computes: energy, centroid,        │ Passes: u_energy, u_centroid,
     │ novelty per beat                   │ u_color, u_novelty to GLSL
     └────────────────────────────────────┘
```

---

## Data Source: Pre-Computed in Rust

### Per-Beat Data (Added to Existing `beats` Array)
| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `energy` | f32 | Normalized RMS energy (0.0–1.0) | `features.rs` → `workflow.rs` (normalized) |
| `centroid` | f32 | Spectral Centroid (0.0–1.0) | `features.rs` (built-in normalization) |
| `novelty` | f32 | Spectral flux (0.0–1.0) | `workflow.rs` (normalized copy) |
| `cluster` | usize | Structural cluster ID | Already exists |

### Normalization Requirements (workflow.rs)
Shader uniforms expect 0.0–1.0 range. Raw values require normalization:

| Data | Raw Range | Normalization | Impact on Existing UI |
|------|-----------|---------------|----------------------|
| **RMS** | ~0.0–0.3 | Divide by max | None — normalized after Fade Detection |
| **Novelty** | Variable | Create normalized copy | None — original `novelty_curve` unchanged |
| **Centroid** | 0.0–1.0 | Already normalized | N/A |

### Data Overhead Estimate
| Song Length | Beats @ 120 BPM | Extra Data (JSON) |
|-------------|-----------------|-------------------|
| 3 min | 360 | ~10 KB |
| 4 min | 480 | ~13 KB |
| 5 min | 600 | ~16 KB |

---

## Color Strategy: Segment-Anchored Palette

| Data Source | Visual Effect |
|-------------|---------------|
| **Segment Cluster ID** | Base hue. Each structural cluster maps to a distinct color. |
| **Energy** | Modulates brightness and saturation. High-energy = vivid; low-energy = muted. |
| **Centroid** | (Optional/Subtle) Modulates "Temperature" or "Sharpness". |
| **Novelty** | Triggers smooth color transitions or brief "flash" effects. |

---

## Key Components

### 1. `MoodShader.js`
- **Uniforms:** `u_time`, `u_energy`, `u_centroid`, `u_color` (vec3), `u_novelty`
- **Required Methods:** `initWebGL()`, `resize(width, height, dpr)`, `render()`, `destroy()`

### 2. `viz.js` Integration Points
| Hook | Location | Action |
|------|----------|--------|
| Constructor | `L19-43` | Add `this.moodShader = new MoodShader(bgCanvas)` |
| `resize()` | `L77-96` | Add `this.moodShader.resize(this.width, this.height, dpr)` |
| `setData()` | `L107-122` | Beat data with energy/centroid/novelty arrives in `payload.beats` |
| `draw()` | `L153-242` | Lerp values, call `this.moodShader.render()` |

### 3. Frontend Layout
**Current (`index.html:25`):** Single `<canvas id="jukebox-canvas">` inside `#viz-layer`

**Target:**
```html
<div id="viz-layer">
    <canvas id="bg-shader"></canvas>      <!-- z-index: 0, WebGL -->
    <canvas id="jukebox-canvas"></canvas> <!-- z-index: 1, 2D -->
</div>
```

---

## Data Strategy: Interpolated Smoothness & Jump Handling

The shader needs to feel "fluid," but the backend data is discrete (per-beat) and jumps are determined Just-In-Time (JIT).

### The "JIT Jump" Problem
Since Remixatron decides to jump at the very last moment, the frontend cannot know for certain what the *next* beat will be.
*   **Naive Interpolation:** Interpolating linearly to `Current + 1` works 95% of the time but causes a visual "snap" if a jump occurs to a different color.
*   **Solution: The "Follower" Lag.**
    *   The frontend always assumes the next beat is linear (`Current + 1`) for the purpose of "breathing."
    *   To handle jumps gracefully, the Shader Uniforms apply a **Low-Pass Filter (Lag Smoothing)**.
    *   If a jump occurs, the target color changes instantly, but the visual color slews to the new target over ~200ms. This turns a "hard snap" into a "fast fade."

### 1. The "Linear Magnitude" Rule (Backend)
- **CRITICAL:** Compute Spectral Centroid using `cqt_spectrogram_raw` (line 93) **before** dB conversion (line 96-111).
- Using log-scale data yields mathematically incorrect centroids.

### 2. High-DPI Synchronization (Frontend)
- `viz.js` handles DPR at `L79-88`.
- `MoodShader.resize(width, height, dpr)` must match exactly.
- Both canvases must be sized identically to prevent layer drift.

### 3. Rust Struct Propagation (Breaking Change)
- **Beat struct:** `playback_engine.rs:62-83` — add 3 fields.
- **Beat instantiation:** `workflow.rs:333-343` — populate new fields in same commit.
- `rms_vec` is available at this scope (line 131, truncated at 184).

### 4. Shader Complexity
- **Constraint:** No raymarching, no 3D geometry.
- **Target:** 2D generative gradients + Simplex Noise.
- **Performance:** 60fps on integrated GPU.

---

## Source File Reference

| File | Key Lines | Purpose |
|------|-----------|---------|
| `features.rs` | L93, L271 | CQT raw data, RMS return |
| `workflow.rs` | L131, L333-343, L424 | rms_vec, Beat instantiation, novelty_curve |
| `playback_engine.rs` | L62-83 | Beat struct definition |
| `viz.js` | L19-43, L77-96, L153-242 | Constructor, resize, draw |
| `index.html` | L24-26 | Canvas container |
| `styles.css` | L59-69 | #viz-layer positioning |
