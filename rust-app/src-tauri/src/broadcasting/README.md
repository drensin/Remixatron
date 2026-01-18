# Broadcasting (`src-tauri/src/broadcasting/`)

This directory contains the network streaming infrastructure for the "Infinite Broadcast" feature, enabling remote visualization on TVs, tablets, and phones.

## Files

*   **`mod.rs`**: Module orchestration and documentation.
    *   Re-exports transcoder and server modules
    *   Documents the overall broadcasting architecture

*   **`transcoder.rs`**: Real-time PCM→MP3 encoding.
    *   Uses LAME via `mp3lame-encoder` crate
    *   Receives raw audio samples from `crossbeam-channel`
    *   Broadcasts encoded MP3 chunks to connected clients
    *   Handles dynamic sample rate changes

*   **`server.rs`**: Axum HTTP/WebSocket server.
    *   **`/stream.mp3`**: Chunked MP3 audio stream (infinite HTTP response)
    *   **`/viz`**: WebSocket endpoint for beat/segment sync data
    *   **`/receiver`**: Static file server for `src-receiver/` HTML

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  JukeboxEngine  │───▶│   Transcoder    │───▶│   Axum Server   │
│  (PCM samples)  │    │   (LAME MP3)    │    │   (:3030)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                      │
                                              ┌───────┴───────┐
                                              ▼               ▼
                                        /stream.mp3      /viz (WS)
```

## Documentation

See **[BROADCAST.md](../../../BROADCAST.md)** for comprehensive architecture documentation including:
- Synchronization strategy
- Protocol specifications
- Performance characteristics
