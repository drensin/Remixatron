# Audio Backend (`src-tauri/src/audio_backend/`)

This directory contains the low-level audio playback infrastructure using the Kira audio engine.

## Files

*   **`mod.rs`**: Module exports and shared backend types.
    *   Re-exports decoder and player modules
    *   Defines audio backend configuration

*   **`decoder.rs`**: Sample-level audio decoding.
    *   Converts raw audio bytes to Kira `Frame` format
    *   Handles stereo/mono conversion
    *   Provides sample-accurate timing information

*   **`player.rs`**: Kira audio manager wrapper.
    *   Initializes the Kira audio context
    *   Manages audio track handles
    *   Provides play/pause/seek operations
    *   Handles gapless beat-to-beat transitions

## Architecture

The audio backend is used by `playback_engine.rs` to achieve sample-accurate playback scheduling. The separation allows the playback engine to focus on jump logic while the backend handles low-level audio I/O.

```
playback_engine.rs → audio_backend/player.rs → Kira → System Audio
```
