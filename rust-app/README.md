# Remixatron: The Infinite Jukebox

> **Just want to run it?**
> You don't need to build from source!
> 📦 **[Download the Latest Release for Windows, macOS, and Linux](https://github.com/drensin/Remixatron/releases)**
>
> 📖 **[Read the User's Manual](USERS_MANUAL.md)** for a complete guide to using the app.

## What is Remixatron?

Remixatron is an intelligent audio player that extends any song into an infinite, seamless remix. By analyzing the musical structure of a track—detecting beats, timbre, and segmentation—it identifies "Jump Candidates": points in the song that are acoustically similar but occur at different times. 

As the song plays, Remixatron intelligently jumps between these points in musical phrases, creating a never-ending version of the track that stays true to the original artist's style but never repeats the exact same pattern twice.

## Why Does It Exist?
We've all had that one song we wish would never end. The original [Infinite Jukebox](https://musicmachinery.com/2012/11/12/the-infinite-jukebox/) (by Paul Lamere) solved this beautifully in the browser over a decade ago. Remixatron is a modern, privacy-focused, and high-performance reimagining of that concept.

It exists to:
1.  **Solve "Short Song Syndrome"**: Turn a 2-minute bop into a 2-hour workflow soundtrack.
2.  **Preserve Privacy**: Unlike web apps that require uploading your music to a third-party server, Remixatron runs 100% locally on your machine. Your audio files never leave your device.
3.  **Ensure Quality**: Built with Rust and the Kira audio engine, it offers sample-accurate scheduling and gapless transitions that web browsers often struggle to maintain.

## What Can I Do With It?
*   **Load Local Audio**: Drag and drop your favorite MP3, WAV, or FLAC files.
*   **Stream from the Web**: Paste a YouTube or SoundCloud URL, and Remixatron will automatically download the highest quality audio for analysis.
*   **Visualize the Walk**: Watch the dazzling real-time visualization that maps every beat, segment, and jump decision as it happens.
*   **Control the Vibes**: Play, Pause, and Resume the infinite walk with zero latency.

---

# Project History

## The Python Era (V1: 2017-2019)
Remixatron began life as a Python CLI tool created by Dave Rensin. It was a love letter to the original [Infinite Jukebox](http://www.infinitejuke.com) by Paul Lamere, designed to run locally on the command line.

### How It Worked
V1 was built on a heavy scientific stack: **Librosa** for audio features, **scikit-learn** for clustering, and **PyGame** for audio playback.

The workflow was linear and pre-calculated:
1.  **Decompose**: The song was broken into beats.
2.  **Cluster**: Beats were grouped by timbre and pitch using Laplacian Spectral Clustering.
3.  **Route**: A "Play Vector" was pre-computed—a list of millions of beats representing a randomized path through the song.
4.  **Play**: The app simply iterated through this pre-determined list.

### The Problem
While accurate, V1 suffered from "Python Limitations":
*   **Latency**: The Global Interpreter Lock (GIL) made true real-time audio scheduling difficult.
*   **Pre-computation**: You couldn't "steer" the walk. The path was set before playback began.
*   **Distribution**: Packaging a Python app with scientific dependencies (numpy, scipy) for end-users is notoriously painful.

## The Web UI Era (V2: 2019-2025)
To solve the distribution problem, V2 moved Remixatron to the browser. Hosted in `Remixatron/Web UI`, this version used a hybrid architecture:

*   **Backend**: A **Flask** server running the original V1 Python analysis code (Librosa/SciPy).
*   **Frontend**: A web interface communicating via **Socket.IO** to visualize the graph and play audio.
*   **Delivery**: Docker containers were used to tame the complex dependency tree.

### The Problem
While V2 made the *UI* better, the underlying engine remained "Heavy":
*   **Complexity**: Users still had to run a local server (or Docker container). It wasn't "just a website" or "just an app."
*   **Latency**: The backend still did all the heavy lifting.
*   **Maintenance**: Managing Python environments, ffmpeg versions, and `yt-dlp` updates inside Docker became a chore.

## The Rust Era (V3: Present)
Remixatron V3 (this project) represents the mature evolution of the concept—a complete rewrite focusing on **Performance** and **Portability**.

### Performance
By moving to **Rust**, we eliminated the Python Global Interpreter Lock (GIL). This unlocked:
*   **Sample-Accurate Audio**: Using the **Kira** audio engine, jumps happen with sub-millisecond precision. No more "hiccups" or slight drifts.
*   **Real-Time Inference**: Instead of pre-calculating a fixed path, V3 runs an **ONNX** Machine Learning model (BeatThis) in real-time. This allows the playback engine to make "JIT" (Just-In-Time) decisions, meaning you can pause, resume, and eventually "steer" the walk dynamically.

### Portability via Tauri
We adopted **Tauri** to build the application. This gives us the best of both worlds:
*   **Modern UI**: The interface is built with standard Web Technologies (HTML/JS/CSS), ensuring a beautiful, responsive experience.
*   **Native Backend**: The heavy lifting is done by a compiled Rust binary.
*   **Cross-Platform**: Remixatron compiles to a **single, native executable** for **macOS**, **Windows**, and **Linux**.

Gone are the days of `pip install`, broken Docker containers, and environment hell. You just download the app and double-click.

---

# Architecture & Code Structure

Remixatron is a hybrid application. The UI logic lives in JavaScript, while the heavy computation and audio processing live in Rust.

## Tech Stack
*   **Frontend**: Vanilla JavaScript (ES6+), CSS3 (Material Design), HTML5 Canvas (Visualization).
*   **Framework**: [Tauri v2](https://tauri.app) (Bridges JS and Rust).
*   **Backend Language**: Rust (Edition 2021).
*   **Audio Engine**: [Kira](https://github.com/tesselode/kira) (Low-latency scheduling).
*   **Machine Learning**: [ONNX Runtime](https://onnxruntime.ai/) (Running the **BeatThis** model for beat tracking).
*   **Decoding**: [Symphonia](https://github.com/pdeljanov/Symphonia) (MP3/WAV/FLAC/AAC support).

## Directory Layout
The project is split into three main zones.

### 1. The Frontend (`src/`)
This is the "Face" of the application. It runs inside a system WebView.
*   **`main.js`**: The controller. It handles user input (`startRemix`), manages UI state, and calls Rust commands (`invoke('analyze_track')`). Use this to understand the high-level application flow.
*   **`viz.js`**: The visualization engine. A focused class that draws the "Infinite Walk" graph on an HTML5 Canvas.
*   **`styles.css`**: Contains the Material Design 3 theme and layout logic.

### 2. The Backend (`src-tauri/src/`)
This is the "Brain" of the application. It runs natively on the OS.
*   **`workflow.rs`**: The implementation of the primary pipeline. It coordinates Decoding -> Mel Spectrogram -> Beat Tracking -> Segmentation -> Graph Generation.
*   **`playback_engine.rs`**: The "Heartbeat". It runs the infinite loop, deciding in real-time which beat to play next based on the graph generated by `workflow.rs`.
*   **`analysis/`**: Contains the hard math.
    *   **`inference.rs`**: Wraps the ONNX Runtime session.
    *   **`structure.rs`**: Implements novelty-based segmentation (spectral clustering disabled).
*   **`downloader.rs`**: Wraps `yt-dlp` to provide the "Universal Downloader" capability.

### 3. Verification (`scripts/`)
*   **`generate_gold_standard.py`**: The Source of Truth. This Python script runs the original V1 algorithm (Librosa/Sklearn) to generate reference data. The Rust test suite uses this data to ensure the new engine produces mathematically identical results to the original research code.

## How it Fits Together
1.  **User** drops a file or URL in the UI (`src/main.js`).
2.  **Frontend** calls `invoke('analyze_track')`.
3.  **Tauri** routes this to the Rust backend (`src-tauri/src/lib.rs`).
4.  **Workflow** runs the ML pipeline (`workflow.rs`) and returns a `StructurePayload` (Beats, Segments, Jumps).
5.  **Frontend** sends this payload to the Visualizer (`viz.js`).
6.  **Frontend** calls `invoke('play_track')`.
7.  **Playback Engine** (`playback_engine.rs`) takes over the audio thread, scheduling audio blocks and making jump decisions in real-time.

---

# Developer Setup

Ready to contribute? Here is how to get the infinite jukebox running on your machine.

## 1. Prerequisites
You need the standard "Rust + JS" starter pack.

*   **Rust**: Install via [rustup.rs](https://rustup.rs/).
*   **Node.js**: Version 18+ (LTS recommended).
*   **System Libraries**:
    *   **Linux**: `sudo apt install libasound2-dev libgtk-3-dev libwebkit2gtk-4.0-dev`
    *   **Mac/Win**: No special libraries needed usually.

## 2. Install Dependencies
Clone the repo and install the frontend dependencies.

```bash
cd rust-app
npm install
```

## 3. The ONNX Runtime (Critical!)
Remixatron relies on `libonnxruntime` for its ML capabilities. You do not need to install Python or PyTorch, but you **DO** need the shared library.

1.  Download `libonnxruntime` (v1.16+) for your platform.
2.  Place the library file (`libonnxruntime.so`, `.dylib`, or `.dll`) in the `rust-app/` root.
3.  Set the environment variable `ORT_DYLIB_PATH` to point to it.

```bash
export ORT_DYLIB_PATH=$(pwd)/libonnxruntime.so
```

*(Note: The `build.rs` script attempts to locate this automatically, but setting the env var is the safest bet).*

## 4. Run in Development Mode
This will launch the Tauri window with "Hot Reload" enabled for both Rust and JS.

```bash
npm run tauri dev
```

## 5. Build for Production
To create a strictly optimized, release-ready binary installer (deb, dmg, msi):

```bash
npm run tauri build
```
The output will be in `src-tauri/target/release/bundle/`.

---

## 🤓 A Special Note For the Math Nerds
Want to understand the **Spectral Clustering**, **Graph Laplacian**, or **BeatThis** architecture under the hood?
Read the **[Deep DSP Documentation](DSP.md)** for a line-by-line breakdown of the Digital Signal Processing pipeline.
