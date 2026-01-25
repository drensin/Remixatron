# The Infinite Broadcast

> **Stream Remixatron's visualization and audio to any device on your local network.**

---

## Table of Contents

1. [The Problem](#the-problem)
2. [The Solution](#the-solution)
3. [Alternatives Considered](#alternatives-considered)
4. [Technical Deep Dive](#technical-deep-dive)
   - [Architecture Overview](#architecture-overview)
   - [Component Details](#component-details)
   - [Data Flow](#data-flow)
   - [Synchronization Strategy](#synchronization-strategy)
   - [Protocol Specifications](#protocol-specifications)

---

## The Problem

### Why Can't I Just Cast This?

Remixatron is a desktop application that generates infinite remixes of songs by algorithmically jumping between musically similar beats. The visualization—a beautiful circular ring showing the song's structure with animated beat cursors and countdown indicators—runs in real-time on your laptop.

But what if you want to:

- **Watch on your TV** while the laptop controls playback?
- **Share the experience** with friends across the room without crowding around a laptop?
- **Cast to a Chromecast** for ambient music visualization while working?

The core challenge is that Remixatron isn't a video. It's a **live, algorithmic experience**. The engine makes probabilistic jumping decisions in real-time, so there's no pre-recorded video to cast. Every playback is unique.

Traditional screen mirroring has problems:
- **Latency**: 200-500ms delay makes the visualization feel disconnected from the audio
- **Quality**: Compression artifacts on a TV-sized display
- **Resource usage**: Encoding your screen in real-time taxes your CPU

We needed a way to replicate the live visualization on a remote device in perfect sync with the audio.

---

## The Solution

### The "Receiver" Architecture

Instead of streaming video, we stream **data and audio separately**:

1. **Audio**: Encode the raw audio to MP3 and stream it over HTTP
2. **Visualization Data**: Send beat timing, segment colors, and playback position over WebSocket
3. **Receiver UI**: A lightweight web page that reconstructs the visualization locally using Canvas

The receiver runs in any modern browser (or a Chromecast custom receiver app). It:
- Plays the MP3 audio stream in an `<audio>` element
- Receives real-time beat/segment updates
- Draws the visualization locally, perfectly synced to its own audio playback

This approach has significant advantages:
- **Zero video encoding CPU cost** on the desktop
- **High-quality visualization** at native resolution on the receiver
- **Sub-beat latency** after initial buffer
- **Works on any device** with a browser

### Quick Start

1. Launch Remixatron
2. On any device on your local network, open: `http://<your-laptop-ip>:3030/receiver/`
3. The receiver will connect. Start playback on the laptop to begin the party!

---

## Alternatives Considered

### 1. Screen Mirroring (Miracast/AirPlay)

**Tried**: Used macOS AirPlay to mirror to an Apple TV.

**Problems**:
- 300-400ms audio delay (configurable but still noticeable)
- Visualization looked compressed and slightly blurry
- CPU usage spiked 20-30% on the laptop
- Required specific hardware (Apple TV)

**Verdict**: Rejected due to latency and quality issues.

---

### 2. HLS/fMP4 Video Streaming

**Tried**: Implemented a full HLS encoder using `ffmpeg-next` in Rust:
- Captured visualization frames from an offscreen canvas in JavaScript
- Transferred 8MB RGBA frames to Rust via Tauri IPC
- Encoded frames to H.264 with `libx264`
- Muxed into HLS `.ts` segments

**Problems**:
- **IPC bottleneck**: Transferring 8MB per beat (at ~2 beats/sec) caused severe UI lag
- **Encoding latency**: Even with `ultrafast` preset, encoding added 100-200ms
- **Complexity**: Managing A/V sync in HLS is notoriously difficult
- **Segment latency**: HLS typically has 3-10 second latency due to segment buffering

**Verdict**: Rejected. The fundamental architecture of "render in JS → transfer to Rust → encode → mux → serve" was too slow for real-time use. Even with optimizations (reduced resolution, base64 encoding, non-blocking sends), the performance was unacceptable.

---

### 3. Rust-Native Rendering with `tiny-skia`

**Tried**: Planned to render visualization frames entirely in Rust:
- Use `tiny-skia` for 2D graphics (arcs, paths, fills)
- Use `ab_glyph` for text rendering
- Encode directly to HLS without any IPC

**Problems**:
- Added complexity: porting the entire visualization to Rust
- Still requires video encoding latency
- Would need to duplicate visualization logic between desktop and receiver

**Verdict**: Rejected. While this would solve the IPC problem, it still has encoding latency and significant implementation effort. The receiver architecture is simpler and provides lower latency.

---

### 4. WebRTC Video Streaming

**Considered**: Use WebRTC for real-time video delivery.

**Problems**:
- Requires STUN/TURN infrastructure for NAT traversal (overkill for local network)
- Still requires video encoding
- Significant implementation complexity

**Verdict**: Not attempted. The receiver architecture solves the problem more elegantly.

---

## Technical Deep Dive

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           REMIXATRON DESKTOP                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐     │
│  │  AUDIO DECODER   │     │  JUKEBOX ENGINE  │     │  KIRA AUDIO      │     │
│  │  (Symphonia)     │────►│  (Playback Loop) │────►│  (Local Output)  │     │
│  └──────────────────┘     └────────┬─────────┘     └──────────────────┘     │
│                                    │                                         │
│                          ┌─────────▼─────────┐                               │
│                          │   PRE-KIRA TAP    │                               │
│                          │ (Raw PCM Samples) │                               │
│                          └─────────┬─────────┘                               │
│                                    │                                         │
│         ┌──────────────────────────┼──────────────────────────┐              │
│         │                          │                          │              │
│         ▼                          ▼                          ▼              │
│  ┌──────────────┐          ┌──────────────┐          ┌──────────────┐       │
│  │  TRANSCODER  │          │  VIZ INIT    │          │  VIZ UPDATE  │       │
│  │  (PCM→MP3)   │          │  (watch::Tx) │          │  (broadcast) │       │
│  └──────┬───────┘          └──────┬───────┘          └──────┬───────┘       │
│         │                         │                         │                │
│         │                         ▼                         ▼                │
│         └────────────────────► ┌──────────────────────────────────┐          │
│                                │      WEBSOCKET DISPATCHER        │          │
│                                └────────────────┬─────────────────┘          │
│                                                 │                            │
│                                                 ▼                            │
└──────────────────────────────────────────────────────────────────────────────┘
                                                  │
                                                  │ WebSocket (Port 3030)
                                                  │ (Binary: Audio + Metadata)
                                                  │ (Text: Init + Control)
                                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           WEB RECEIVER (Browser/Cast)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐                               ┌──────────────────┐    │
│  │ Audio Context    │◄──────(Binary: MP3)───────────┤ WebSocket Client │    │
│  │ (Web Audio API)  │                               │ (Parser)         │    │
│  └──────────────────┘                               └───────┬──────────┘    │
│                                                             │               │
│                                                             │ (JSON: Meta)  │
│                                                             ▼               │
│  ┌──────────────────┐     ┌──────────────────┐      ┌──────────────────┐    │
│  │ Playback Sched.  │────►│   Sync Engine    │─────►│ Canvas Renderer  │    │
│  │ (Buffer/Time)    │     │   (Ref Clock)    │      │ (60fps Viz)      │    │
│  └──────────────────┘     └──────────────────┘      └──────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Component Details

#### 1. Pre-Kira Tap (playback_engine.rs)

The "Pre-Kira Tap" is a critical architectural decision. When audio is loaded:

```rust
// In load_track()
if let Some(tx) = &self.broadcast_tx {
    // Send ALL decoded samples to the transcoder upfront
    let _ = tx.send(samples.clone());
    
    // Also update the shared sample rate
    if let Some(sr) = &self.broadcast_sample_rate {
        sr.store(sample_rate, Ordering::SeqCst);
    }
}
```

**Why "Pre-Kira"?**

Kira is the audio engine that handles mixing and playback. Its internal buffers and timing are opaque. Trying to tap audio *after* Kira processes it would require complex synchronization.

Instead, we capture raw decoded samples *before* they enter Kira. The transcoder then streams these samples to clients at the rate dictated by `stream_time` updates from the playback engine.

---

#### 2. Transcoder (transcoder.rs)

The transcoder runs in a dedicated background thread:

```rust
pub struct Transcoder {
    audio_rx: crossbeam_channel::Receiver<Vec<f32>>,
    sample_rate: Arc<AtomicU32>,
    // This sender now sends to the WebSocket dispatcher
    ws_audio_tx: mpsc::Sender<Bytes>, 
}
```

**Encoding Loop**:

1. Receive PCM samples from `audio_rx`
2. Buffer samples until we have enough for an MP3 frame (1152 samples)
3. Encode using LAME (`mp3lame-encoder`)
4. Send encoded bytes to the `ws_audio_tx` for dispatch via WebSocket.

**Sample Rate Handling**:

Different audio files have different sample rates (44100, 48000, etc.). The transcoder dynamically reinitializes the LAME encoder when the sample rate changes:

```rust
if self.current_sample_rate != new_rate {
    self.encoder = Encoder::new(new_rate, 2, Quality::Best)?;
    self.current_sample_rate = new_rate;
}
```

---

#### 3. HTTP Server (server.rs)

The Axum server exposes two endpoints:

##### `/viz` - WebSocket Data & Audio

Clients connect via WebSocket and receive a mix of Text (JSON) and Binary frames.

**1. Init Message (Text/JSON)**
Sent once on connection. Contains the full remix graph structure.
```json
{
  "type": "init",
  "beats": [...],
  "segments": [...],
  "waveform": [...],
  "title": "...",
  "artist": "..."
}
```

**2. Update Frames (Binary)**
Sent continuously (approx. every beat). Contains 16 bytes of metadata followed by the MP3 audio chunk for that beat. This ensures metadata and audio are perfectly coupled.
```text
[Byte 0-3]   beat_id (u32, little-endian)
[Byte 4-7]   seg_id  (u32, little-endian)
[Byte 8-11]  seq_pos (u32, little-endian)
[Byte 12-15] seq_len (u32, little-endian)
[Byte 16+]   MP3 Audio Data...
```

**3. Control Messages (Text/JSON)**
```json
{ "type": "update", "paused": true, "stopped": false }
```

##### `/receiver/` - Static Files

Serves the `src-receiver/` directory containing `index.html`.

---

#### 4. Web Receiver (index.html)

The receiver is a self-contained HTML file with embedded CSS and JavaScript. Key components:

##### Audio Playback

The receiver uses the **Web Audio API** to decode and play MP3 audio chunks received via WebSocket. This allows for precise scheduling and synchronization.

```javascript
const audioContext = new (window.AudioContext || window.webkitAudioContext)();
let currentAudioTime = 0; // Tracks the playback time of the Web Audio API
```

##### Sync Table

The receiver maintains a "sync table" mapping `stream_time` to visualization state:

```javascript
const syncTable = [];

// On each WebSocket update (after parsing binary frame):
syncTable.push({
    streamTime: msg.stream_time, // This is derived from the scheduled audio buffer
    beatId: msg.beat_id,
    segId: msg.segment_id,
    seqPos: msg.seq_pos,
    seqLen: msg.seq_len
});
```

##### Offset Calibration

When audio first starts playing, we calculate the offset between `audioContext.currentTime` and `stream_time`. This `streamOffset` represents the `stream_time` at which the local `audioContext.currentTime` was 0.

```javascript
// When the first audio buffer is scheduled to play at a specific stream_time:
streamOffset = bufferStartStreamTime - audioContext.currentTime;
console.log(`Calibrated: offset=${streamOffset.toFixed(2)}s`);
```

##### Render Loop

```javascript
function renderLoop() {
    // Calculate which stream_time we're hearing right now
    const audioStreamPosition = audioContext.currentTime + streamOffset;
    
    // Find the matching sync entry
    const entry = syncTable.find(e => e.streamTime <= audioStreamPosition);
    
    if (entry) {
        activeBeatIndex = entry.beatId;
        activeSegmentIndex = entry.segId;
        // ... update visualization state
    }
    
    draw(); // Canvas rendering
    requestAnimationFrame(renderLoop);
}
```

---

### Data Flow

#### On Track Load

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Frontend   │───►│   analyze_   │───►│   Jukebox    │───►│  Transcoder  │
│   (main.js)  │    │   track()    │    │   Engine     │    │  (samples)   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  viz_init_tx │───► WebSocket Dispatcher → WebSocket clients receive init data
                    └──────────────┘
```

#### During Playback

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Playback   │───►│  Beat Event  │───►│ viz_update_  │───► WebSocket Dispatcher
│   Engine     │    │  Callback    │    │     tx       │    (metadata)
└──────────────┘    └──────────────┘    └──────────────┘
       │                                       ▲
       ▼                                       │
┌──────────────┐    ┌──────────────┐           │
│   Sample     │───►│  Transcoder  │───────────┘
│   Buffer     │    │  (LAME enc)  │           (MP3 audio)
└──────────────┘    └──────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           WEBSOCKET DISPATCHER                               │
│ (Combines metadata from viz_update_tx and MP3 audio from Transcoder)        │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           WEBSOCKET CLIENTS (RECEIVER)                       │
│ (Receive combined binary frames: metadata + MP3)                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Synchronization Strategy
 
 The core challenge is keeping the receiver's visualization in sync with its audio playback. The code implements a **Direct Scheduling** strategy using the Web Audio API.
 
 #### Solution: Unified Binary Stream + Direct Scheduling
 
 1.  **Bundled Data**: Every WebSocket message is a **Binary Frame** containing both metadata (16 bytes) and the specific audio chunk for that beat.
 2.  **Sequential Scheduling**: The receiver maintains a pointer `nextPlayTime` (AudioContext time). When a new chunk arrives, it is scheduled to play immediately after the previous chunk:
     ```javascript
     source.start(nextPlayTime);
     nextPlayTime += buffer.duration;
     ```
 3.  **Visualization Trigger**: Since we know *exactly* when the audio will start playing (at `nextPlayTime`), we set a standard JavaScript timeout to trigger the visual update at that precise moment:
     ```javascript
     const delay = (nextPlayTime - audioContext.currentTime) * 1000;
     setTimeout(() => {
         // Update UI for this beat
         activeBeatIndex = chunk.beatId;
     }, delay);
     ```
 
 This forces perfect AV sync. The visualization cannot "drift" because it is triggered by the same clock (AudioContext) that handles the audio playback.
 
 #### Why not a Sync Table?
 
 Earlier prototypes used a global `stream_time` and a lookup table. This was abandoned because:
 1.  It required complex clock synchronization between server and client.
 2.  It was vulnerable to "spiral of death" drifts if the network lagged.
 3.  Direct scheduling is simpler and self-correcting: if the network lags, both audio `nextPlayTime` and the visual timeout shift together effectively pausing until data arrives.
 
 ---
 
 ### Protocol Specifications

#### WebSocket Messages

##### Init Message (Server → Client)

```typescript
interface VizInitMessage {
  type: "init";
  beats: Beat[];
  segments: Segment[];
  waveform: number[];  // 720 normalized amplitudes
  title: string;
  artist: string;
  thumbnail: string;  // Data URL or http URL
}

interface Beat {
  id: number;
  start: number;       // Seconds from track start
  duration: number;    // Beat duration in seconds
  segment: number;     // Segment index
  jump_candidates: number[];  // Beat IDs that can be jumped to
}

interface Segment {
  label: number;       // Cluster label (for color)
  start_time: number;
  end_time: number;
}
```

##### Update Frame (Binary)

Sent as a WebSocket Binary Message. All integers are **Little-Endian u32** (4 bytes).

| Offset | Field | Type | Description |
|--------|-------|------|-------------|
| 0 | `beat_id` | u32 | Index of the active beat |
| 4 | `seg_id` | u32 | Index of the active segment |
| 8 | `seq_pos` | u32 | Position in current sequence |
| 12 | `seq_len` | u32 | Length of current sequence |
| 16 | `payload` | bytes | MP3 Audio Frame(s) |

##### Control Messages (Text/JSON)

```typescript
interface ControlMessage {
  type: "update";
  paused?: boolean;
  stopped?: boolean;
}
```

---

### Performance Characteristics

| Metric | Value |
|--------|-------|
| Audio latency (initial buffer) | ~3-5 seconds |
| Visualization sync accuracy | ±20ms |
| CPU overhead (desktop) | <5% |
| Network bandwidth | ~128 kbps (MP3) + ~1 kbps (WebSocket) |
| Memory (receiver) | ~10-15 MB |

---

### Future Work

1.  **Multi-room Audio**: Synchronize playback across multiple receivers (requires NTP-like clock sync).
2.  **Quality Settings**: Allow receiver to request different MP3 bitrates.

---

## Conclusion

The "Infinite Broadcast" feature enables Remixatron to share its live, algorithmic music visualization with any device on the local network. By separating audio streaming from visualization data, we achieve low latency, high quality, and minimal CPU overhead—a significant improvement over traditional screen mirroring or video streaming approaches.

Open `http://<your-ip>:3030/receiver/` and enjoy the infinite remix on your big screen! 🎵📺
