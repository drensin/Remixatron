# Task List: Mood Visualizer Integration

## Phase 1: Backend Data Generation (Rust)

> ⚠️ **Breaking Change:** Modifying `Beat` struct requires simultaneous `workflow.rs` update.

- [ ] **Compute Spectral Centroid** (`features.rs`)
  - [ ] Locate `cqt_spectrogram_raw` at **line 93** (linear magnitude).
  - [ ] **Before** the dB conversion loop (lines 96-111), compute:
    ```rust
    // Centroid[t] = Sum(k * Mag[t][k]) / Sum(Mag[t][k])
    let mut centroid_vec = Vec::with_capacity(n_beats);
    ```
  - [ ] Normalize to `0.0 - 1.0` (divide by `n_bins`).
  - [ ] Apply median pooling to sync with beats (like `rms_sync`).
  - [ ] Return as 4th element: `(mfcc_sync, chroma_sync, cqt_sync, rms_sync, centroid_sync)`.

- [ ] **Update `Beat` Struct** (`playback_engine.rs:62-83`)
  - [ ] Add after line 82:
    ```rust
    pub energy: f32,
    pub centroid: f32,
    pub novelty: f32,
    ```

- [x] **Normalize Values for Shader** (`workflow.rs`, before Assembly)
  - [x] RMS: Divide by max RMS (preserves relative dynamics, yields 0.0–1.0)
  - [x] Novelty: Create `novelty_normalized` copy (original curve unchanged for debug UI)
  - [x] Centroid: Already 0.0–1.0 from `features.rs`
  - [x] **Safety:** Normalize RMS *after* Fade Detection (line 207) to avoid affecting that logic

- [x] **Populate Beat Fields** (`workflow.rs:352-365`)
  - [x] `energy`: from normalized `rms_vec`
  - [x] `centroid`: from `centroid_vec` (already 0–1)
  - [x] `novelty`: from `novelty_normalized` (not raw curve)

- [x] **Verify Compilation:** `cargo check` ✅ 0 warnings

---

## Phase 2: Frontend Infrastructure

- [ ] **Create `src/MoodShader.js`**
  ```javascript
  export class MoodShader {
    constructor(canvas) { this.canvas = canvas; this.gl = null; }
    initWebGL() { /* ... */ }
    resize(width, height, dpr) { /* ... */ }
    render(time, energy, centroid, color, novelty) { /* ... */ }
    destroy() { /* ... */ }
  }
  ```

- [ ] **Update `index.html` (line 24-26)**
  - [ ] Add `<canvas id="bg-shader"></canvas>` BEFORE `#jukebox-canvas`.

- [ ] **Update `styles.css`**
  - [ ] Add after line 69:
    ```css
    #bg-shader {
      position: absolute;
      top: 0; left: 0;
      width: 100%; height: 100%;
      z-index: 0;
    }
    #jukebox-canvas {
      position: relative;
      z-index: 1;
      background: transparent;
    }
    ```

---

## Phase 3: Shader Implementation (GLSL)

- [ ] **Vertex Shader:** Simple fullscreen quad (2 triangles, clip-space coords).

- [ ] **Fragment Shader:**
  ```glsl
  uniform float u_time;
  uniform float u_energy;
  uniform float u_centroid;
  uniform vec3 u_color;
  uniform float u_novelty;
  
  // Simplex noise (inline or import)
  // ...
  
  void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution.xy;
    
    // Base gradient from u_color
    vec3 baseColor = mix(u_color * 0.3, u_color, uv.y);
    
    // Noise layer (scale affected by u_centroid)
    float noiseScale = 2.0 + u_centroid * 2.0;
    float n = snoise(vec3(uv * noiseScale, u_time * 0.1));
    
    // Energy modulates brightness
    float brightness = 0.5 + u_energy * 0.5;
    
    // Novelty flash
    float flash = u_novelty * 0.3;
    
    gl_FragColor = vec4(baseColor * brightness + flash, 1.0);
  }
  ```

- [ ] **Implement `resize()`:** Update `gl.viewport()` and uniform `u_resolution`.

---

## Phase 4: Visualization Integration (`src/viz.js`)

- [ ] **Import MoodShader** (top of file):
  ```javascript
  import { MoodShader } from './MoodShader.js';
  ```

- [ ] **Define Color Palette & State Variables** (inside constructor):
  ```javascript
  this.moodPalette = [
    [0.8, 0.2, 0.4], [0.2, 0.6, 0.9], [0.6, 0.3, 0.8],
    [0.9, 0.6, 0.2], [0.2, 0.8, 0.5], [0.9, 0.9, 0.3],
  ];
  this.currentMoodColor = [0,0,0]; // For lag smoothing
  this.activeBeatIndex = -1;       // Tracks current beat for animation loop
  this.smoothedEnergy = 0.5;       // Lag-smoothed energy
  this.smoothedCentroid = 0.5;     // Lag-smoothed centroid
  ```

- [ ] **Update `activeBeatIndex` from `draw()`:**
  - [ ] At top of `draw(activeBeatIndex, ...)`: `this.activeBeatIndex = activeBeatIndex;`
  - [ ] This allows the animation loop to read current beat state independently.

- [ ] **Initialize MoodShader & Start Loop** (in constructor):
  ```javascript
  const bgCanvas = document.getElementById('bg-shader');
  this.moodShader = new MoodShader(bgCanvas);
  this.moodShader.initWebGL();
  this.startAnimationLoop();
  ```

- [ ] **Implement `startAnimationLoop()`**:
  ```javascript
  startAnimationLoop() {
    const loop = () => {
      if (this.moodShader && this.beats.length > 0) {
        // 1. Identify Target State from this.currentSeqPos / activeBeatIndex
        // 2. Apply Lag Smoothing (Low Pass Filter) to color/energy
        //    this.smoothedEnergy = this.smoothedEnergy * 0.9 + targetEnergy * 0.1;
        // 3. Render
        this.moodShader.render(performance.now()/1000, this.smoothedEnergy, ...);
      }
      requestAnimationFrame(loop);
    };
    loop();
  }
  ```

- [ ] **Update `resize()`:** Call `this.moodShader.resize(width, height, dpr)`.

- [ ] **Upgrade Arc Rendering:**
  - [ ] Add `ctx.globalCompositeOperation = 'screen'` before drawing arcs.
  - [ ] After arcs: `ctx.globalCompositeOperation = 'source-over'`.

---

## Phase 5: Polish & Verification

- [ ] **Add CSS Vignette** (`styles.css`):
  ```css
  #viz-layer::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at center, transparent 40%, black 100%);
    pointer-events: none;
    z-index: 2;
  }
  ```

- [ ] **Test Visual Balance:** Ensure foreground graph remains readable.
- [ ] **Performance Test:** Verify 60fps on integrated GPU.
- [ ] **Edge Cases:** Missing energy/centroid values (default to 0.5).

---

## Deferred (Fast-Follow)

- [ ] **User Controls UI:** Toggle, intensity slider, palette selector.
- [ ] **Receiver Port:** Evaluate for `src-receiver/index.html`.
- [ ] **Spectral Centroid Enhancement:** Subtle hue shift (±15°).
