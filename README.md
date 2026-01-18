# Web Receiver (`src-receiver/`)

This directory contains the self-hosted web receiver that displays Remixatron's visualization on remote devices (TVs, tablets, phones).

## Overview

The receiver is a lightweight, self-contained HTML application that:
- Connects to the Remixatron desktop app over the local network
- Plays synchronized MP3 audio via the `/stream.mp3` endpoint
- Receives real-time beat/segment updates via WebSocket (`/viz`)
- Renders the circular visualization using HTML5 Canvas

## Files

*   **`index.html`**: The complete receiver application (HTML + CSS + JavaScript bundled together).
    *   WebSocket client for visualization sync
    *   Audio element for MP3 streaming
    *   Canvas-based visualization renderer
    *   Play bar with track metadata (title, artist, thumbnail)
    *   Sync engine with offset calibration

## Usage

1. Start playback in the main Remixatron app
2. Open `http://<your-ip>:3030/receiver/` on any device on your network

## Architecture

See **[BROADCAST.md](../BROADCAST.md)** for detailed documentation of the streaming architecture.
