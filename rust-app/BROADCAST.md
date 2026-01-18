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

1. Start playback in Remixatron
2. On any device on your local network, open: `http://<your-laptop-ip>:3030/receiver/`
3. The visualization appears and audio plays in sync

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
└─────────┼─────────────────────────┼─────────────────────────┼────────────────┘
          │                         │                         │
          ▼                         ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AXUM HTTP SERVER (:3030)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  GET /stream.mp3          GET /viz (WebSocket)       GET /receiver/*        │
│  ┌──────────────┐         ┌──────────────────┐       ┌──────────────────┐   │
│  │ Chunked MP3  │         │ JSON Messages:   │       │ Static Files:    │   │
│  │ Audio Stream │         │ - init (beats,   │       │ - index.html     │   │
│  │ (infinite)   │         │   segments, etc) │       │ (self-contained) │   │
│  │              │         │ - update (beat,  │       │                  │   │
│  │              │         │   seq_pos, etc)  │       │                  │   │
│  └──────────────┘         └──────────────────┘       └──────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ HTTP / WebSocket
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           WEB RECEIVER (Browser)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐         ┌──────────────────┐       ┌──────────────────┐   │
│  │ <audio> Tag  │         │ WebSocket Client │       │ Canvas Renderer  │   │
│  │ (MP3 Stream) │         │ (Sync Table)     │       │ (60fps Viz)      │   │
│  └──────┬───────┘         └────────┬─────────┘       └────────▲─────────┘   │
│         │                          │                          │              │
│         │                          │                          │              │
│         ▼                          ▼                          │              │
│  ┌──────────────────────────────────────────────────────────┐ │              │
│  │                    SYNC ENGINE                            │ │              │
│  │  audio.currentTime + streamOffset = cumulative_audio_time ├─┘              │
│  │  Lookup beat in sync table → render correct state         │               │
│  └───────────────────────────────────────────────────────────┘               │
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
    broadcast_tx: broadcast::Sender<Bytes>,
}
```

**Encoding Loop**:

1. Receive PCM samples from `audio_rx`
2. Buffer samples until we have enough for an MP3 frame (1152 samples)
3. Encode using LAME (`mp3lame-encoder`)
4. Broadcast encoded bytes to all connected clients via `tokio::sync::broadcast`

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

The Axum server exposes three endpoints:

##### `/stream.mp3` - Chunked Audio Stream

```rust
async fn handle_audio_stream(State(state): State<AppState>) -> impl IntoResponse {
    let rx = state.audio_broadcast_rx.resubscribe();
    let stream = BroadcastStream::new(rx)
        .filter_map(|result| ready(result.ok()));
    
    Response::builder()
        .header("Content-Type", "audio/mpeg")
        .header("Cache-Control", "no-cache")
        .body(Body::from_stream(stream))
}
```

This creates an infinite HTTP response that streams MP3 chunks as they become available.

##### `/viz` - WebSocket Visualization Data

Clients connect via WebSocket and receive:

1. **Init Message** (once, on connect):
   ```json
   {
     "type": "init",
     "beats": [...],      // Array of beat timing/segment data
     "segments": [...],   // Array of segment boundaries
     "waveform": [...],   // Amplitude envelope for waveform ring
     "title": "...",
     "artist": "...",
     "thumbnail": "..."
   }
   ```

2. **Update Messages** (continuous):
   ```json
   {
     "type": "update",
     "beat_id": 42,
     "segment_id": 3,
     "seq_pos": 5,
     "seq_len": 8,
     "stream_time": 123.456
   }
   ```

##### `/receiver/` - Static Files

Serves the `src-receiver/` directory containing `index.html`.

---

#### 4. Web Receiver (index.html)

The receiver is a self-contained HTML file with embedded CSS and JavaScript. Key components:

##### Audio Playback

```javascript
const audio = new Audio(`http://${host}:3030/stream.mp3`);
audio.play();
```

##### Sync Table

The receiver maintains a "sync table" mapping `stream_time` to visualization state:

```javascript
const syncTable = [];

// On each WebSocket update:
syncTable.push({
    streamTime: msg.stream_time,
    beatId: msg.beat_id,
    segId: msg.segment_id,
    seqPos: msg.seq_pos,
    seqLen: msg.seq_len
});
```

##### Offset Calibration

When audio first starts playing, we calculate the offset between `audio.currentTime` and `stream_time`:

```javascript
audio.addEventListener('playing', () => {
    streamOffset = bufferStartStreamTime - audio.currentTime;
    console.log(`Calibrated: offset=${streamOffset.toFixed(2)}s`);
});
```

##### Render Loop

```javascript
function renderLoop() {
    // Calculate which stream_time we're hearing right now
    const audioStreamPosition = audio.currentTime + streamOffset;
    
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
                    │  viz_init_tx │───► WebSocket clients receive init data
                    └──────────────┘
```

#### During Playback

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Playback   │───►│  Beat Event  │───►│ viz_update_  │───► WebSocket clients
│   Engine     │    │  Callback    │    │     tx       │    receive updates
└──────────────┘    └──────────────┘    └──────────────┘
       │
       ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Sample     │───►│  Transcoder  │───►│  Broadcast   │───► HTTP clients
│   Buffer     │    │  (LAME enc)  │    │  Channel Rx  │    receive MP3
└──────────────┘    └──────────────┘    └──────────────┘
```

---

### Synchronization Strategy

The core challenge is keeping the receiver's visualization in sync with its audio playback. Here's how we solve it:

#### Problem: Network Latency

The receiver's `<audio>` element buffers several seconds of MP3 data before playback starts. The WebSocket messages, however, arrive in real-time. If we show beat 100 on the viz while the audio is still playing beat 95, it looks wrong.

#### Solution: Cumulative Stream Time

The playback engine tracks `cumulative_audio_time`—the total seconds of audio that have been "streamed" (sent to clients) since playback started:

```rust
// When a beat plays
stream_time += beat.duration;
viz_update_tx.send(VizUpdateData {
    beat_id: beat.id,
    stream_time: stream_time,
    // ...
});
```

The receiver's `<audio>` element also tracks `currentTime`—how many seconds of audio have been played locally.

#### Calibration

When audio starts playing, we know the head of the audio buffer corresponds to `bufferStartStreamTime` (the `stream_time` from when buffering began). We calculate:

```
streamOffset = bufferStartStreamTime - audio.currentTime
```

For example:
- `audio.currentTime = 0` (just started playing)
- `bufferStartStreamTime = 123.456` (we buffered starting at second 123.456)
- `streamOffset = 123.456`

Now, at any moment:
```
audioStreamPosition = audio.currentTime + streamOffset
```

This tells us "what stream_time is the user hearing right now?" We look up that time in the sync table to get the correct beat/segment to display.

#### Drift Handling

Over long playback sessions, small timing drifts can accumulate. We handle this by:

1. **Retaining a generous time window** of sync entries (e.g., the last 60 seconds of `stream_time`). This ensures we can always find a valid lookup even if the receiver's audio playback lags slightly behind the live stream.

2. **Using reverse iteration** to find the most recent entry with `streamTime <= audioStreamPosition`. This is faster than binary search for typical cases where we're looking for recent entries.

3. **Periodically pruning entries older than the retention window** to prevent unbounded memory growth during long playback sessions (which can run for hours). Entries far in the past will never be looked up again since `audio.currentTime` only moves forward.

> **Note**: We keep entries for a time *window*, not the *entire* sync history. This balances memory efficiency with the need to handle audio buffering delays.

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

##### Update Message (Server → Client)

```typescript
interface VizUpdateMessage {
  type: "update";
  beat_id: number;
  segment_id: number;
  seq_pos: number;     // Position in current jump sequence
  seq_len: number;     // Length of current jump sequence
  stream_time: number; // Cumulative audio seconds
}
```

##### Control Messages (Server → Client)

```typescript
interface PauseMessage {
  type: "update";
  paused: boolean;
}

interface StopMessage {
  type: "update";
  stopped: boolean;
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

1. **Chromecast Integration**: Use `rust_cast` to discover and launch a custom receiver on Chromecast devices
2. **Multi-room Audio**: Synchronize playback across multiple receivers
3. **Quality Settings**: Allow receiver to request different MP3 bitrates

---

## Conclusion

The "Infinite Broadcast" feature enables Remixatron to share its live, algorithmic music visualization with any device on the local network. By separating audio streaming from visualization data, we achieve low latency, high quality, and minimal CPU overhead—a significant improvement over traditional screen mirroring or video streaming approaches.

Open `http://<your-ip>:3030/receiver/` and enjoy the infinite remix on your big screen! 🎵📺
