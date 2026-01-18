# Remixatron User's Manual

**Version 0.9.1**

---

## Table of Contents
1.  [What is Remixatron?](#what-is-remixatron)
2.  [Installation](#installation)
3.  [Getting Started](#getting-started)
4.  [The Interface](#the-interface)
5.  [Core Features](#core-features)
6.  [Network Receiver (Cast to Any Screen)](#network-receiver-cast-to-any-screen)
7.  [Troubleshooting](#troubleshooting)
8.  [Frequently Asked Questions](#frequently-asked-questions)

---

## What is Remixatron?

Remixatron is an intelligent audio player that transforms any song into an **infinite, seamless remix**. It analyzes the musical structure of a track—detecting beats, timbre, and segmentation—to identify "Jump Points": moments in the song that are acoustically similar but occur at different times.

As the song plays, Remixatron intelligently jumps between these points in musical phrases, creating a never-ending version of the track that stays true to the original artist's style but never repeats the exact same pattern twice.

**Use Cases:**
*   **Focus Playlists**: Turn your favorite 3-minute track into a 3-hour ambient soundscape.
*   **DJ Sets**: Explore the hidden structure of songs before mixing.
*   **Music Study**: Visually see the Verse/Chorus/Bridge architecture.

---

## Installation

### Downloading a Release
1.  Navigate to the [Releases Page](https://github.com/drensin/Remixatron/releases) on GitHub.
2.  Download the installer for your operating system:
    *   **macOS**: `.dmg` file (Universal Binary for Intel & Apple Silicon)
    *   **Windows**: `.msi` installer
    *   **Linux**: `.deb` package (Ubuntu/Debian) or `.AppImage`
3.  Run the installer and follow the on-screen instructions.

### macOS Security Note
On macOS, you may see a warning that the app is from an "unidentified developer."
1.  **Right-click** (or Ctrl+Click) the `Remixatron.app` icon.
2.  Select **Open**.
3.  Click **Open** again in the dialog box that appears.

You only need to do this once.

---

## Getting Started

### Step 1: Launch the App
Double-click the Remixatron application. You will be greeted by a simple onboarding card asking for your audio source.

### Step 2: Provide Audio
You can provide audio in two ways:

#### Option A: Local File
1.  Click the text input field.
2.  Type the **full absolute path** to an audio file on your computer.
    *   Example (macOS): `/Users/yourname/Music/MySong.mp3`
    *   Example (Windows): `C:\Users\yourname\Music\MySong.mp3`
3.  Supported formats: **MP3**, **FLAC**, **WAV**, **AAC (M4A)**.

#### Option B: Paste a URL
1.  Copy a URL from **YouTube**, **SoundCloud**, or other supported platforms.
2.  Paste the URL directly into the input field.
    *   Example: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`
3.  Remixatron will automatically download the audio for you.

### Step 3: Analyze
Click the **Analyze** button. The application will:
1.  (If URL) Download the highest quality audio stream.
2.  Decode and resample the audio.
3.  Run a Neural Network (BeatThis) to detect every beat.
4.  Perform hybrid segmentation to find song structure.
5.  Build the Jump Graph.

This process takes approximately **15-45 seconds**, depending on track length and your hardware.

### Step 4: Enjoy the Infinite Walk
Once analysis completes, playback begins automatically. Watch the visualization light up as beats play, and observe the arcs connect similar sections of music. The song will now play forever!

---

## The Interface

### Onboarding Card
The initial screen with the text input and "Analyze" button. This is where you provide your audio source.

### Floating Player
Once a track is loaded, the "Floating Player" appears at the bottom of the screen. It contains:
*   **Track Title**: The name of the current song.
*   **Album Art**: If available, a thumbnail of the track's cover.
*   **Pause/Play Button**: Click to toggle pause. The icon changes between a Pause symbol and a Play symbol.
*   **Stop Button**: Stops the current playback and returns you to the Onboarding Card.

### The Visualization Canvas
The main area of the screen is a dynamic visualization.
*   **The Circle**: Represents the entire song laid out radially. The 12 o'clock position is the beginning, and it continues clockwise.
*   **Beat Dots**: Small dots around the circle, each representing a single beat.
*   **Segment Colors**: Beats are colored by their "Cluster" (e.g., Verse, Chorus). Similar sections share the same color.
*   **Jump Arcs**: Curved lines connecting beats that are valid jump targets.
*   **The Playhead**: A bright highlight indicating the currently playing beat.
*   **The Pulse Ring**: A central ring that counts down to the next possible jump point.

---

## Core Features

### Infinite Playback
The signature feature. Music continues indefinitely, intelligently looping between similar sections.

### Just-In-Time Decisions
Unlike simpler loopers, Remixatron doesn't pre-calculate a fixed path. It decides where to go *as the song plays*, meaning every listen is unique.

### Universal Downloader
Paste a URL from YouTube, SoundCloud, Vimeo, and many other sites. Remixatron uses `yt-dlp` under the hood to fetch the audio stream.

### Pause/Resume
Click the pause button to freeze the infinite walk. Click again to resume exactly where you left off.

### Stop & Reset
Click the stop button to end playback entirely and return to the onboarding screen to load a new track.

---

## Network Receiver (Cast to Any Screen)

Remixatron can stream its visualization and audio to **any device on your local network**. Watch the infinite remix on your TV, tablet, or phone—no special hardware required.

### How It Works

When Remixatron is running, it starts a local server on port 3030. Any device on your network can connect via a web browser to see a perfectly synchronized copy of the visualization with audio.

### Using the Receiver

1.  **Start playback** on your laptop (the main Remixatron app).
2.  **Find your laptop's IP address**:
    *   **macOS**: System Preferences → Network → Wi-Fi → IP Address
    *   **Windows**: Open CMD, type `ipconfig`, look for "IPv4 Address"
    *   **Linux**: Run `ip addr` or `hostname -I`
3.  **On your TV/tablet/phone**, open a web browser and go to:
    ```
    http://<your-laptop-ip>:3030/receiver/
    ```
    Example: `http://192.168.1.42:3030/receiver/`
4.  The visualization appears and audio plays in sync!

### Notes

*   Both devices must be on the **same Wi-Fi network**.
*   There's a **3-5 second initial buffer** while the audio loads.
*   The receiver shows the same visualization as the main app, including the play bar with track info and thumbnail.
*   You can open **multiple receivers** simultaneously (e.g., TV + phone).

---

## Troubleshooting

### "Download Failed" Error
*   **Cause**: The YouTube or streaming service may have blocked the request (e.g., CAPTCHA, geographic restriction, or a stale downloader).
*   **Solution**: Wait a few minutes and try again. Remixatron automatically updates its internal downloader on each launch to stay compatible.

### No Sound
1.  Check your system volume.
2.  Ensure no other application has exclusive audio control.
3.  On Linux, ensure `libasound2` is installed.

### App Crashes on macOS with "Library Not Loaded"
*   **Cause**: This is a known issue with older builds that dynamically linked against Homebrew libraries.
*   **Solution**: Update to the latest release (v0.2.1 or later), which bundles all dependencies statically.

### Analysis Takes Forever
*   The first run may be slower as components initialize.
*   Try with a shorter track (under 5 minutes) first.
*   Extremely long tracks (> 20 minutes) may take several minutes to analyze.

### Receiver Won't Connect
*   Ensure both devices are on the **same Wi-Fi network**.
*   Check that your firewall isn't blocking port **3030**.
*   Try `http://localhost:3030/receiver/` on the laptop itself to verify the server is running.
*   If using a VPN, try disconnecting it.

---

## Frequently Asked Questions

### Q: What platforms are supported?
A: Remixatron provides official builds for:
*   macOS (Apple Silicon & Intel)
*   Windows (x64)
*   Linux (Ubuntu/Debian x64)

### Q: Can I control *when* a jump happens?
A: Not yet. The playback engine makes real-time decisions using structured musical phrases. Manual "steering" is on the roadmap for a future version.

### Q: Is my music uploaded to the cloud?
A: **No**. All analysis and playback happen 100% locally on your machine. Your audio files never leave your device.

### Q: Why does the downloaded audio sound slightly different from YouTube?
A: Remixatron downloads the highest-quality audio stream available (usually AAC or Opus) without any video. This is often a higher bitrate than what YouTube's video player delivers, so it may even sound *better*.

### Q: How does it detect the beats?
A: Remixatron uses **BeatThis**, a State-of-the-Art neural network from ISMIR 2024 (a top Music Information Retrieval conference). It runs via the ONNX Runtime for fast, portable inference.

### Q: Can I watch on my TV?
A: **Yes!** Open `http://<your-laptop-ip>:3030/receiver/` in any browser on your smart TV or a device connected to your TV. See the [Network Receiver](#network-receiver-cast-to-any-screen) section for details.

### Q: Can I use my own ONNX models?
A: Not through the UI, but advanced users can replace the `.onnx` files in the application bundle's `models/` directory.

---

## Credits & Acknowledgements

### Inspiration
Remixatron is a love letter to **Paul Lamere's** original [Infinite Jukebox](http://infinitejuke.com). Paul created the concept while working at The Echo Nest (later acquired by Spotify). His vision of finding hidden loops in music and playing them forever inspired this entire project.

> *"This work is inspired by the Infinite Jukebox project created by Paul Lamere (paul@spotify.com)."*
> — Original Remixatron README (2017)

### Author
**Dave Rensin** (drensin@gmail.com) — Sole author and maintainer of Remixatron since 2017. V1 (Python CLI), V2 (Web UI), and V3 (Rust Native) are all his work.

### Research & Algorithms
Remixatron's "secret sauce" is built on the shoulders of giants in the **Music Information Retrieval (MIR)** community:

*   **BeatThis** — The neural beat tracker used by V3 is based on State-of-the-Art research published at **ISMIR 2024**. It uses modern neural network architecture for robust beat and downbeat detection.
*   **Novelty-Based Segmentation** — The core segmentation algorithm uses novelty detection on a self-similarity matrix to find structural boundaries (verse, chorus, bridge). Each segment is assigned a unique label.
*   **MFCC & Chroma Features** — Timbral and harmonic fingerprinting methods standardized in audio analysis research. The original implementation used **[Librosa](https://librosa.org/doc/latest/index.html)** by Brian McFee et al. (See: McFee, B. et al., *"librosa: Audio and Music Signal Analysis in Python,"* Proc. of the 14th Python in Science Conference, 2015. [Paper PDF](https://conference.scipy.org/proceedings/scipy2015/pdfs/proceedings.pdf#page=24)).

### Key Libraries & Frameworks

#### Current (V3 - Rust Native)
| Library | Purpose | Link |
|---|---|---|
| **Tauri** | Cross-platform app framework (Rust + WebView) | [tauri.app](https://tauri.app) |
| **Kira** | Low-latency, sample-accurate audio engine | [github.com/tesselode/kira](https://github.com/tesselode/kira) |
| **ONNX Runtime** | High-performance ML inference | [onnxruntime.ai](https://onnxruntime.ai) |
| **Symphonia** | Pure-Rust audio decoding (MP3, FLAC, WAV, AAC) | [github.com/pdeljanov/Symphonia](https://github.com/pdeljanov/Symphonia) |
| **Rubato** | High-fidelity audio resampling | [crates.io/crates/rubato](https://crates.io/crates/rubato) |
| **yt-dlp** | Universal video/audio downloader | [github.com/yt-dlp/yt-dlp](https://github.com/yt-dlp/yt-dlp) |
| **Linfa** | Pure-Rust machine learning toolkit | [github.com/rust-ml/linfa](https://github.com/rust-ml/linfa) |
| **Axum** | Async HTTP/WebSocket server | [github.com/tokio-rs/axum](https://github.com/tokio-rs/axum) |
| **mp3lame-encoder** | Real-time MP3 encoding (LAME) | [crates.io/crates/mp3lame-encoder](https://crates.io/crates/mp3lame-encoder) |

#### Legacy (V1/V2 - Python)
| Library | Purpose |
|---|---|
| **Librosa** | Audio analysis (MFCC, Chroma, Beat Tracking) |
| **scikit-learn** | Spectral Clustering & K-Means |
| **madmom** | RNN-based beat tracking (V2) |
| **PyGame** | Audio playback (V1) |
| **Flask** | Web server (V2) |
| **Socket.IO** | Real-time browser communication (V2) |

### Special Thanks
*   The **ISMIR community** for advancing the science of Music Information Retrieval.
*   The **Tauri team** for making cross-platform native apps accessible to solo developers.
*   The **yt-dlp maintainers** for keeping the Universal Downloader alive against all odds.
*   Everyone who filed issues, tested builds, and provided feedback over the years.

---

© 2017-2026 Dave Rensin. MIT License.

