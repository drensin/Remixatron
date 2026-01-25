# Remixatron Frontend (`src/`)

This directory contains the user interface and visualization logic for the Remixatron web application.

## Core Components
*   **`main.js`**: The application controller. Handles:
    *   Tauri Backend Communication (Invoke/Listen)
    *   UI State Management (Loading, Playback, Download)
    *   Audio Analysis Pipeline Orchestration
    *   Network Receiver Sync (beat/segment updates for `/receiver`)
*   **`viz.js`**: The visualization engine. Handles:
    *   HTML5 Canvas Rendering of the "Infinite Walk"
    *   Dynamic Beat Cursor & Segment Arcs
    *   Jump Prediction Arcs
*   **`LavaLampViz.js`**: Ambient background effect. Handles:
    *   Canvas-based floating blob animation
    *   Radial gradients with drop shadows
    *   Sinusoidal wobble and "breathing" movement
*   **`MoodShader.js`**: Reactive WebGL background. Handles:
    *   Fullscreen GLSL shader rendering behind the main UI
    *   Simplex noise modulated by spectral centroid and energy
    *   Reactive "Flashes" on segment changes (Novelty detection)
*   **`styles.css`**: Material Design 3 (M3) styling implementation.
*   **`index.html`**: App shell structure.

## Architecture
The frontend follows a **"Metadata Shielding"** pattern:
1.  **Immediate Feedback**: When a user inputs a URL/File, the frontend attempts to display metadata *instantly* via local probes or API calls, without waiting for the heavy audio analysis.
2.  **State Isolation**: Visualization logic is decoupled from playback logic via the `InfiniteJukeboxViz` class, which receives update ticks from the backend.
