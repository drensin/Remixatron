# Remixatron Rust Backend (`src-tauri/src/`)

This directory contains the high-performance core of Remixatron, written in Rust. It handles all heavy lifting: Machine Learning inference, Audio Analysis, and Real-Time Playback orchestration.

## Module Structure

### Core Logic
*   **`lib.rs`**: The library entry point. Defines `AppState`, registers Tauri commands (`analyze_track`, `play_track`), and manages the global `JukeboxEngine` mutex.
*   **`workflow.rs`**: The "Brain" of the operation. Orchestrates the end-to-end analysis pipeline:
    1.  **Decoding** (`audio/`)
    2.  **Mel Spectrogram** (`analysis/mel.rs`)
    3.  **Beat Tracking** (`analysis/inference.rs`)
    4.  **Segmentation** (`analysis/structure.rs`)
    5.  **Graph Generation** (Jump Candidates)
*   **`downloader.rs`**: Universal Downloader logic. Wraps `yt-dlp` to fetch audio from HTTP sources.

### Audio Engine
*   **`playback_engine.rs`**: The infinite playback scheduler.
    *   Manages a dedicated audio thread.
    *   Implements the "Ghost Pruning" strategy for gapless transitions.
    *   Handles low-latency Pause/Resume via a custom Clock implementation.
*   **`audio/`**: Low-level audio decoding utilities (via `symphonia`).

### Analysis Modules (`analysis/`)
*   **`structure.rs`**: Implements Spectral Clustering (Laplacian Eigenmaps) to find song sections.
*   **`features.rs`**: Computes CQT/MFCC features for similarity comparison.
*   **`mel.rs`**: Pre-processing for the BeatNet ONNX model.
*   **`inference.rs`**: Runs the ONNX Runtime session for beat detection.

## Key binaries
*   **`main.rs`**: The application binary entry point.
*   **`bin/verify_audio.rs`**: A CLI tool for testing the playback engine without the GUI.
