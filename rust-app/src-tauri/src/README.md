# Remixatron Rust Backend (`src-tauri/src/`)

This directory contains the high-performance core of Remixatron, written in Rust. It handles all heavy lifting: Machine Learning inference, Audio Analysis, and Real-Time Playback orchestration.

## Module Structure

### Core Logic
*   **`lib.rs`**: The library entry point. Defines `AppState`, registers Tauri commands (`analyze_track`, `play_track`), and manages the global `JukeboxEngine` mutex.
*   **`main.rs`**: The application binary entry point. Bootstraps Tauri and starts the event loop.
*   **`workflow.rs`**: The "Brain" of the operation. Orchestrates the end-to-end analysis pipeline:
    1.  **Decoding** (`audio/`)
    2.  **Mel Spectrogram** (`beat_tracker/mel.rs`)
    3.  **Beat Tracking** (`beat_tracker/inference.rs`)
    4.  **Segmentation** (`analysis/structure.rs`)
    5.  **Graph Generation** (Jump Candidates)
*   **`downloader.rs`**: Universal Downloader logic. Wraps `yt-dlp` to fetch audio from HTTP sources.
*   **`favorites.rs`**: Favorites persistence. Loads/saves user's favorite tracks to local storage.

### Audio Engine
*   **`playback_engine.rs`**: The infinite playback scheduler.
    *   Manages a dedicated audio thread.
    *   Implements the "Ghost Pruning" strategy for gapless transitions.
    *   Handles low-latency Pause/Resume via a custom Clock implementation.
    *   Sends audio samples to the broadcast transcoder ("Pre-Kira Tap").

### Subdirectory Modules

Each subdirectory has its own README.md with detailed documentation.

*   **`audio/`**: Audio file loading and decoding (Symphonia). See [`audio/README.md`](audio/README.md).
*   **`audio_backend/`**: Kira audio engine wrapper for playback. See [`audio_backend/README.md`](audio_backend/README.md).
*   **`analysis/`**: Feature extraction and segmentation algorithms. See [`analysis/README.md`](analysis/README.md).
*   **`beat_tracker/`**: Neural network beat detection (BeatThis ONNX). See [`beat_tracker/README.md`](beat_tracker/README.md).
*   **`broadcasting/`**: Network streaming for remote visualization. See [`broadcasting/README.md`](broadcasting/README.md).

## Related Directories

*   **`../verification/`**: Standalone test binaries for pipeline stages. See [`../verification/README.md`](../verification/README.md).
