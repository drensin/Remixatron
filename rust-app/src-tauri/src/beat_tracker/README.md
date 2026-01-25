# Beat Tracker (`src-tauri/src/beat_tracker/`)

This directory contains the neural network-based beat detection pipeline using the **BeatThis** ONNX model.

## Files

*   **`mod.rs`**: Module exports and shared types.
    *   Re-exports all beat tracking components
    *   Defines `BeatInfo` and related data structures

*   **`mel.rs`**: Mel spectrogram preprocessing.
    *   Converts raw audio to mel-frequency representation
    *   Normalizes input for ONNX model consumption
    *   Handles windowing and hop-length configuration

*   **`inference.rs`**: ONNX Runtime session management.
    *   Loads the BeatThis model (`BeatThis_small0.onnx`)
    *   Runs inference on mel spectrograms
    *   Returns raw activation curves for beats and downbeats

*   **`post_processor.rs`**: Peak picking and beat extraction.
    *   Converts activation curves to discrete beat times
    *   Applies threshold and minimum interval constraints
    *   Identifies downbeats for musical phrase detection

*   **`processing.rs`**: End-to-end beat tracking pipeline.
    *   Orchestrates mel → inference → post-processing
    *   Provides the main `track_beats()` API
    *   Handles chunked processing for long audio files

## Model

The **BeatThis** model is a State-of-the-Art beat tracker published at ISMIR 2024. It uses a neural network architecture trained on diverse music datasets to detect beats and downbeats with high accuracy.

## Usage

```rust
use beat_tracker::processing::track_beats;

let beat_times = track_beats(&audio_samples, sample_rate)?;
```
