# Audio Loader (`src-tauri/src/audio/`)

This directory contains the audio file loading and decoding utilities.

## Files

*   **`mod.rs`**: Module exports and shared audio types.
    *   Re-exports the `loader` module
    *   Defines common audio data structures

*   **`loader.rs`**: High-level audio file loading.
    *   Wraps Symphonia for multi-format decoding (MP3, FLAC, WAV, AAC)
    *   Handles resampling to target sample rate
    *   Provides normalized f32 sample output

## Usage

The loader is primarily used by `workflow.rs` during the analysis pipeline to decode incoming audio files before feature extraction.

```rust
use audio::loader::load_audio;

let samples = load_audio("/path/to/song.mp3", 22050)?;
```
