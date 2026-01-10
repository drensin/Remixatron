# Remixatron Android: Hybrid Rust/Kotlin Architecture v2.0

A comprehensive design and implementation guide for porting the Remixatron Infinite Jukebox to Android using a shared Rust DSP core with a native Kotlin UI and audio engine.

> [!IMPORTANT]
> **v2.3 Update**: This revision addresses all gaps identified in critical reviews (Rounds 1-5), including JIT state machine architecture, ONNX bundling, audio ownership, threading model, and Android media integration.

---

## 1. Executive Summary

This document outlines the architecture for a native Android app that reuses the verified Remixatron Rust DSP core while implementing platform-native audio playback (Oboe) and UI (Jetpack Compose). The hybrid approach preserves the "Golden Master" audio analysis pipeline while delivering a first-class Android experience.

### Goals
- **Code Reuse**: ~70% of Rust logic unchanged
- **Performance**: Low-latency audio via Oboe (AAudio/OpenSL ES)
- **UX**: Native Material Design 3 with Jetpack Compose
- **Maintainability**: Minimal JNI surface (~12 bridge functions)
- **Future-Proof**: iOS port requires only UI + audio engine (same Rust core)

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       ANDROID APPLICATION                            │
├─────────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                     PRESENTATION LAYER                          │ │
│  │  ┌─────────────────┐  ┌──────────────────────────────────────┐ │ │
│  │  │  Jetpack Compose │  │  Canvas Visualization                │ │ │
│  │  │  (M3 UI)         │  │  (Pre-calculated positions)          │ │ │
│  │  └────────┬────────┘  └────────────────┬─────────────────────┘ │ │
│  └───────────┼────────────────────────────┼───────────────────────┘ │
│              │                            │                          │
│  ┌───────────▼────────────────────────────▼───────────────────────┐ │
│  │                     APPLICATION LAYER                           │ │
│  │  ┌─────────────────┐  ┌──────────────────────────────────────┐ │ │
│  │  │   ViewModel     │  │  RemixatronRepository                │ │ │
│  │  │   (StateFlow)   │  │  (Coroutines + JNI Bridge)           │ │ │
│  │  └────────┬────────┘  └────────────────┬─────────────────────┘ │ │
│  └───────────┼────────────────────────────┼───────────────────────┘ │
│              │                            │                          │
│  ┌───────────▼────────────────────────────▼───────────────────────┐ │
│  │                     PLATFORM LAYER                              │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐  │ │
│  │  │  Oboe Engine    │  │  Download Mgr   │  │  MediaSession  │  │ │
│  │  │  (C++ native)   │  │  (yt-dlp)       │  │  (Foreground)  │  │ │
│  │  └────────┬────────┘  └────────┬────────┘  └───────┬────────┘  │ │
│  └───────────┼────────────────────┼───────────────────┼───────────┘ │
│              │                    │                   │              │
│              │ JNI                │                   │              │
│  ┌───────────▼────────────────────┴───────────────────┴───────────┐ │
│  │                     RUST NATIVE LAYER                           │ │
│  │  ┌───────────────────────────────────────────────────────────┐ │ │
│  │  │               remixatron-android (cdylib)                  │ │ │
│  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐  │ │ │
│  │  │  │ jni_bridge  │ │ jukebox_jit │ │ remixatron-core     │  │ │ │
│  │  │  │ (new)       │ │ (new)       │ │ (extracted crate)   │  │ │ │
│  │  │  └─────────────┘ └─────────────┘ └─────────────────────┘  │ │ │
│  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐  │ │ │
│  │  │  │ symphonia   │ │   rubato    │ │ ort (static link)   │  │ │ │
│  │  │  └─────────────┘ └─────────────┘ └─────────────────────┘  │ │ │
│  │  └───────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Design

### 3.1 Rust Native Layer

#### 3.1.0 Phase 0: Monorepo Structure & `remixatron-core` Extraction

Before Android development, restructure the repository into a **Cargo workspace** with shared code extracted into a standalone crate.

##### Complete Directory Structure

```
/home/rensin/Projects/Remixatron/
├── Cargo.toml                     # NEW: Workspace root
├── README.md                      # Update to document all platforms
├── LICENSE
│
├── models/                        # NEW: Shared ONNX models (single source)
│   ├── BeatThis_small0.onnx
│   ├── BeatThis_small1.onnx
│   ├── BeatThis_small2.onnx
│   ├── MelSpectrogram_Ultimate.onnx
│   └── MelSpectrogram_Ultimate.onnx.data
│
├── remixatron-core/               # NEW: Shared Rust library
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                 # Public API
│       ├── workflow.rs            # Analysis orchestration
│       ├── analysis/              # MFCC, Chroma, clustering
│       │   ├── mod.rs
│       │   ├── features.rs
│       │   └── structure.rs
│       ├── beat_tracker/          # ONNX inference
│       │   ├── mod.rs
│       │   └── madmom_rnn.rs
│       ├── audio/                 # Decoding + resampling
│       │   ├── mod.rs
│       │   └── loader.rs
│       └── jit/                   # JIT playback brain
│           ├── mod.rs
│           ├── engine.rs          # Stateful JukeboxEngine
│           └── types.rs           # Beat, Segment, PlayInstruction
│
├── rust-app/                      # Existing desktop app (Tauri)
│   ├── src-tauri/
│   │   ├── Cargo.toml             # Depends on remixatron-core
│   │   └── src/
│   │       ├── lib.rs             # Tauri commands
│   │       ├── playback_engine.rs # Kira-based (desktop only)
│   │       └── downloader.rs      # yt-dlp subprocess (desktop only)
│   ├── src/                       # Frontend (HTML/JS/CSS)
│   └── package.json
│
├── android-app/                   # NEW: Android application
│   ├── app/
│   │   ├── src/main/
│   │   │   ├── kotlin/com/remixatron/
│   │   │   ├── cpp/               # Oboe C++ code
│   │   │   ├── assets/            # Symlink → ../../models/
│   │   │   └── jniLibs/           # Built Rust .so files
│   │   ├── proguard-rules.pro
│   │   └── build.gradle.kts
│   ├── rust/
│   │   └── remixatron-android/    # JNI wrapper crate
│   │       ├── Cargo.toml         # Depends on remixatron-core
│   │       └── src/
│   ├── settings.gradle.kts
│   └── build.gradle.kts
│
└── (legacy Python files - can remain at root)
    ├── Remixatron.py
    ├── infinite_jukebox.py
    └── requirements.txt
```

##### Cargo Workspace Configuration

```toml
# /home/rensin/Projects/Remixatron/Cargo.toml (NEW: Workspace root)

[workspace]
members = [
    "remixatron-core",
    "rust-app/src-tauri",
    "android-app/rust/remixatron-android",
]
resolver = "2"

[workspace.package]
version = "0.4.0"
edition = "2021"
authors = ["Your Name"]
license = "Apache-2.0"

[workspace.dependencies]
# Shared dependencies with consistent versions
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
ndarray = "0.15"
rand = "0.8"
log = "0.4"
thiserror = "1.0"

# Audio processing
symphonia = { version = "0.5", features = ["all"] }
rubato = "0.14"

# ML Inference
ort = { version = "=2.0.0-rc.10", default-features = false }
```

```toml
# /home/rensin/Projects/Remixatron/remixatron-core/Cargo.toml

[package]
name = "remixatron-core"
version.workspace = true
edition.workspace = true

[dependencies]
serde = { workspace = true }
ndarray = { workspace = true }
rand = { workspace = true }
log = { workspace = true }
thiserror = { workspace = true }
symphonia = { workspace = true }
rubato = { workspace = true }
ort = { workspace = true }
```

```toml
# /home/rensin/Projects/Remixatron/rust-app/src-tauri/Cargo.toml (UPDATED)

[package]
name = "remixatron-desktop"
version.workspace = true
edition.workspace = true

[dependencies]
remixatron-core = { path = "../../remixatron-core" }

# Desktop-only dependencies
tauri = { version = "2.0", features = [...] }
kira = "0.10"
# ... rest of existing dependencies
```

```toml
# /home/rensin/Projects/Remixatron/android-app/rust/remixatron-android/Cargo.toml

[package]
name = "remixatron-android"
version.workspace = true
edition.workspace = true

[lib]
crate-type = ["cdylib"]

[dependencies]
remixatron-core = { path = "../../../remixatron-core" }
jni = "0.21"

[target.'cfg(target_os = "android")'.dependencies]
android_logger = "0.13"
log = { workspace = true }
```

##### ONNX Model Sharing

The `models/` directory is the single source of truth. Each platform references it differently:

| Platform | How Models Are Accessed |
|----------|------------------------|
| **Desktop** | Direct path: `../models/BeatThis_small0.onnx` |
| **Android** | Symlinked into assets, then copied to cache at runtime |

```bash
# Android asset symlink setup (run once)
cd android-app/app/src/main/assets
ln -s ../../../../../models/*.onnx .
```

> [!NOTE]
> Git will follow the symlinks and store the files. Alternatively, use a Gradle copy task.

**Why this structure?**
1. **Single Cargo workspace** — shared dependency versions, single `target/` directory
2. **Platform separation** — `rust-app/` and `android-app/` are clear siblings
3. **Shared core** — `remixatron-core` is referenced by both via relative path
4. **Model deduplication** — ONNX files exist once, not copied across platforms

#### 3.1.1 Project Structure (`remixatron-android`)

```
remixatron-android/
├── Cargo.toml
├── build.rs                    # ONNX static linking config
├── src/
│   ├── lib.rs                  # JNI entry points
│   ├── jni_bridge.rs           # Type conversions + error handling
│   ├── jni_exceptions.rs       # Exception throwing utilities
│   ├── handles.rs              # AnalysisHandle, JukeboxHandle
│   └── assets.rs               # Android asset path resolution
└── jniLibs/                    # Output directory for .so files
```

#### 3.1.2 Reused Modules (via `remixatron-core`)

| Module | Status | Notes |
|--------|--------|-------|
| `workflow.rs` | ✅ Extracted | Core analysis orchestration |
| `analysis/` | ✅ Extracted | MFCC, Chroma, spectral clustering |
| `beat_tracker/` | ✅ Extracted | ONNX inference pipeline |
| `audio/loader.rs` | ✅ Extracted | Symphonia + Rubato decoding |
| `jit/engine.rs` | ✅ **NEW** | JIT state machine (extracted from `playback_engine.rs`) |
| `favorites.rs` | ⚠️ Adapt | Use Android `app_data_dir` path |
| `playback_engine.rs` | ❌ Desktop only | Uses Kira (not Android-compatible) |
| `downloader.rs` | ❌ Desktop only | Uses subprocess (not Android-compatible) |

---

### 3.2 JIT State Machine Architecture

> [!IMPORTANT]
> The JIT playback brain is the core of the infinite jukebox. It maintains **mutable state** across beat decisions. This section clarifies the ownership model.

#### 3.2.1 State Lives in Rust

The `JukeboxEngine` struct holds all JIT state. A separate handle (`jlong`) references it from Kotlin.

```rust
// remixatron-core/src/jit/engine.rs

use rand::prelude::*;
use rand::rngs::StdRng;
use std::collections::VecDeque;

/// Complete JIT playback state machine.
/// 
/// This struct is **stateful** — each call to `get_next_beat()` mutates
/// internal state (cursor, sequence tracking, failed jumps, etc.).
/// 
/// The Kotlin side holds an opaque `jlong` pointer to this struct.
pub struct JukeboxEngine {
    // Read-only analysis data
    pub beats: Vec<Beat>,
    
    // JIT State (mutable across calls)
    cursor: usize,
    current_sequence: usize,
    min_sequence_len: isize,
    beats_since_jump: usize,
    failed_jumps: usize,
    recent_segments: VecDeque<usize>,
    rng: StdRng,
    
    // Pre-calculated constants
    last_chance_beat: usize,
    acceptable_jump_amounts: Vec<usize>,
    max_beats_between_jumps: usize,
}

impl JukeboxEngine {
    /// Creates a new engine from analysis results.
    pub fn new(beats: Vec<Beat>, clusters: usize) -> Self {
        // Pre-calculate constants (tempo-dependent phrase lengths, etc.)
        // See desktop playback_engine.rs lines 96-156 for full logic
        // ...
    }
    
    /// Returns the NEXT beat to play, mutating internal state.
    /// 
    /// This implements the full JIT decision tree:
    /// 1. Update recent segments buffer
    /// 2. Check jump trigger (sequence complete OR panic threshold)
    /// 3. Strategy 1: Jump to non-recent segment
    /// 4. Strategy 2: Quartile Busting (after 10% failed jumps)
    /// 5. Strategy 3: Nuclear Reset (after 30% failed jumps)
    /// 6. Update cursor for next call
    pub fn get_next_beat(&mut self) -> PlayInstruction {
        // Full implementation from desktop playback_engine.rs lines 427-566
    }
    
    /// Resets the engine to the beginning of the track.
    pub fn reset(&mut self) {
        self.cursor = 0;
        self.current_sequence = 0;
        self.beats_since_jump = 0;
        self.failed_jumps = 0;
        self.recent_segments.clear();
    }
    
    /// Seeks to a specific beat (for user-initiated scrubbing).
    pub fn seek_to_beat(&mut self, beat_id: usize) {
        self.cursor = beat_id.min(self.beats.len().saturating_sub(1));
        self.current_sequence = 0;
        // Don't reset recent_segments — preserve variety
    }
}
```

#### 3.2.2 Two Separate Handles

The JNI bridge exposes **two handles** to Kotlin:

| Handle | Contents | Lifecycle |
|--------|----------|-----------|
| `AnalysisHandle` | Beats, segments, metadata, decoded audio buffer pointer | Created by `analyzeTrack()`, freed by `freeAnalysisHandle()` |
| `JukeboxHandle` | `JukeboxEngine` instance (mutable JIT state) | Created by `createJukebox()`, freed by `freeJukeboxHandle()` |

```rust
// remixatron-android/src/handles.rs

/// Immutable analysis results + audio buffer.
pub struct AnalysisHandle {
    pub beats: Vec<Beat>,
    pub segments: Vec<Segment>,
    pub metadata: TrackMetadata,
    pub audio_buffer: AudioBuffer,  // Decoded samples for Oboe
}

/// Mutable JIT playback state.
pub struct JukeboxHandle {
    pub engine: JukeboxEngine,
}

/// Decoded audio ready for playback.
pub struct AudioBuffer {
    pub samples: Vec<f32>,       // Interleaved stereo
    pub sample_rate: u32,        // Original rate (44100/48000)
    pub channel_count: u16,      // Always 2 (stereo)
}
```

#### 3.2.3 JNI Bridge (Updated)

```rust
// remixatron-android/src/jni_bridge.rs

use jni::JNIEnv;
use jni::objects::{JClass, JString, JObject, JByteArray, JValue};
use jni::sys::{jlong, jint, jfloat, jobjectArray};

// ─────────────────────────────────────────────────────────────────
// Analysis Methods
// ─────────────────────────────────────────────────────────────────

/// Analyzes an audio file.
/// 
/// # Returns
/// - On success: Non-zero handle (store as Long in Kotlin)
/// - On failure: 0, and a Java exception is thrown
#[no_mangle]
pub extern "system" fn Java_com_remixatron_bridge_RemixatronBridge_analyzeTrack(
    mut env: JNIEnv,
    _class: JClass,
    audio_path: JString,
    model_dir: JString,
    progress_callback: JObject,
) -> jlong {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        analyze_track_impl(&mut env, audio_path, model_dir, progress_callback)
    }));
    
    match result {
        Ok(Ok(handle)) => handle,
        Ok(Err(msg)) => {
            throw_remixatron_exception(&mut env, &msg);
            0
        }
        Err(_) => {
            throw_remixatron_exception(&mut env, "Rust panic during analysis");
            0
        }
    }
}

/// Creates a JukeboxEngine for JIT playback decisions.
#[no_mangle]
pub extern "system" fn Java_com_remixatron_bridge_RemixatronBridge_createJukebox(
    mut env: JNIEnv,
    _class: JClass,
    analysis_handle: jlong,
) -> jlong {
    if analysis_handle == 0 {
        throw_remixatron_exception(&mut env, "Invalid analysis handle");
        return 0;
    }
    
    let analysis = unsafe { &*(analysis_handle as *const AnalysisHandle) };
    
    // Clone beats for the JukeboxEngine (it needs ownership)
    let engine = JukeboxEngine::new(
        analysis.beats.clone(),
        analysis.metadata.cluster_count,
    );
    
    let handle = Box::new(JukeboxHandle { engine });
    Box::into_raw(handle) as jlong
}

/// Gets the next beat to play. Mutates internal JIT state.
/// 
/// This is the primary integration point with the audio engine.
/// Call this at each beat boundary to get the next beat index.
#[no_mangle]
pub extern "system" fn Java_com_remixatron_bridge_RemixatronBridge_getNextBeat(
    _env: JNIEnv,
    _class: JClass,
    jukebox_handle: jlong,
) -> jint {
    let handle = unsafe { &mut *(jukebox_handle as *mut JukeboxHandle) };
    let instruction = handle.engine.get_next_beat();
    instruction.beat_id as jint
}

/// Gets extended next-beat info (for UI: seq_len, seq_pos).
#[no_mangle]
pub extern "system" fn Java_com_remixatron_bridge_RemixatronBridge_getNextBeatExtended<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    jukebox_handle: jlong,
) -> JObject<'local> {
    let handle = unsafe { &mut *(jukebox_handle as *mut JukeboxHandle) };
    let instruction = handle.engine.get_next_beat();
    
    // Find PlayInstruction class
    let class = match env.find_class("com/remixatron/bridge/PlayInstruction") {
        Ok(c) => c,
        Err(_) => return JObject::null(),
    };
    
    // Constructor: PlayInstruction(beatId: Int, seqLen: Int, seqPos: Int)
    env.new_object(&class, "(III)V", &[
        JValue::Int(instruction.beat_id as i32),
        JValue::Int(instruction.seq_len as i32),
        JValue::Int(instruction.seq_pos as i32),
    ]).unwrap_or(JObject::null())
}

/// Resets playback to beat 0.
#[no_mangle]
pub extern "system" fn Java_com_remixatron_bridge_RemixatronBridge_resetJukebox(
    _env: JNIEnv,
    _class: JClass,
    jukebox_handle: jlong,
) {
    let handle = unsafe { &mut *(jukebox_handle as *mut JukeboxHandle) };
    handle.engine.reset();
}

// ─────────────────────────────────────────────────────────────────
// Audio Buffer Access (for Oboe)
// ─────────────────────────────────────────────────────────────────

/// Returns a pointer to the decoded audio samples.
/// 
/// # Safety
/// The returned pointer is valid until `freeAnalysisHandle()` is called.
/// The Oboe C++ code must not outlive the Rust buffer.
#[no_mangle]
pub extern "system" fn Java_com_remixatron_bridge_RemixatronBridge_getAudioBufferPtr(
    _env: JNIEnv,
    _class: JClass,
    analysis_handle: jlong,
) -> jlong {
    let handle = unsafe { &*(analysis_handle as *const AnalysisHandle) };
    handle.audio_buffer.samples.as_ptr() as jlong
}

/// Returns the number of samples in the audio buffer.
#[no_mangle]
pub extern "system" fn Java_com_remixatron_bridge_RemixatronBridge_getAudioBufferLen(
    _env: JNIEnv,
    _class: JClass,
    analysis_handle: jlong,
) -> jint {
    let handle = unsafe { &*(analysis_handle as *const AnalysisHandle) };
    handle.audio_buffer.samples.len() as jint
}

/// Returns the sample rate of the decoded audio.
#[no_mangle]
pub extern "system" fn Java_com_remixatron_bridge_RemixatronBridge_getAudioSampleRate(
    _env: JNIEnv,
    _class: JClass,
    analysis_handle: jlong,
) -> jint {
    let handle = unsafe { &*(analysis_handle as *const AnalysisHandle) };
    handle.audio_buffer.sample_rate as jint
}

// ─────────────────────────────────────────────────────────────────
// Memory Management
// ─────────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "system" fn Java_com_remixatron_bridge_RemixatronBridge_freeAnalysisHandle(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    if handle != 0 {
        unsafe { drop(Box::from_raw(handle as *mut AnalysisHandle)) };
    }
}

#[no_mangle]
pub extern "system" fn Java_com_remixatron_bridge_RemixatronBridge_freeJukeboxHandle(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    if handle != 0 {
        unsafe { drop(Box::from_raw(handle as *mut JukeboxHandle)) };
    }
}
```

---

#### 3.2.4 JNI Array Construction (getBeats Implementation)

Converting Rust `Vec<Beat>` to a Java array requires creating individual JNI objects:

```rust
// remixatron-android/src/jni_arrays.rs

/// Returns an array of Beat objects to Kotlin.
#[no_mangle]
pub extern "system" fn Java_com_remixatron_bridge_RemixatronBridge_getBeats<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    analysis_handle: jlong,
) -> jobjectArray {
    let handle = unsafe { &*(analysis_handle as *const AnalysisHandle) };
    let beat_class = env.find_class("com/remixatron/bridge/Beat").unwrap();
    let array = env.new_object_array(handle.beats.len() as i32, &beat_class, JObject::null()).unwrap();
    
    for (i, beat) in handle.beats.iter().enumerate() {
        let jump_array = env.new_int_array(beat.jump_candidates.len() as i32).unwrap();
        let jump_ints: Vec<i32> = beat.jump_candidates.iter().map(|&x| x as i32).collect();
        env.set_int_array_region(&jump_array, 0, &jump_ints).unwrap();
        
        let beat_obj = env.new_object(&beat_class, "(IFFJIIIIII[I)V", &[
            JValue::Int(beat.id as i32), JValue::Float(beat.start), JValue::Float(beat.duration),
            JValue::Long(beat.start_sample as i64), JValue::Long(beat.end_sample as i64),
            JValue::Int(beat.bar_position as i32), JValue::Int(beat.segment as i32),
            JValue::Int(beat.cluster as i32), JValue::Int(beat.intra_segment_index as i32),
            JValue::Int(beat.quartile as i32), JValue::Object(&jump_array),
        ]).unwrap();
        env.set_object_array_element(&array, i as i32, beat_obj).unwrap();
    }
    array.into_raw()
}
```

#### 3.2.5 Progress Callback (Thread-Safe JNI)

```rust
// remixatron-android/src/jni_callback.rs

pub struct ProgressCallbackWrapper { jvm: JavaVM, callback: GlobalRef }

impl ProgressCallbackWrapper {
    pub fn new(env: &mut JNIEnv, cb: JObject) -> Result<Self, String> {
        Ok(Self { jvm: env.get_java_vm()?, callback: env.new_global_ref(cb)? })
    }
    
    pub fn on_progress(&self, message: &str, progress: f32) {
        if let Ok(mut env) = self.jvm.attach_current_thread() {
            if let Ok(msg) = env.new_string(message) {
                let _ = env.call_method(&self.callback, "onProgress", "(Ljava/lang/String;F)V",
                    &[JValue::Object(&msg), JValue::Float(progress)]);
            }
        }
    }
}
```

#### 3.2.6 Build Script

```bash
#!/bin/bash
# android-app/rust/remixatron-android/build.sh
set -e
cargo install cargo-ndk 2>/dev/null || true
rustup target add aarch64-linux-android armv7-linux-androideabi x86_64-linux-android
cargo ndk -t arm64-v8a -t armeabi-v7a -t x86_64 -o ../../app/src/main/jniLibs build --release
```

---

### 3.3 ONNX Model Loading Strategy

> [!WARNING]
> Android assets are not accessible via filesystem paths. ONNX Runtime requires either a file path or raw bytes.

#### 3.3.1 Solution: Copy Assets to Cache on First Launch

```kotlin
// ModelAssetManager.kt
object ModelAssetManager {
    // Model names must match files in assets/ directory
    private val MODEL_FILES = listOf(
        "MelSpectrogram_Ultimate.onnx",
        "MelSpectrogram_Ultimate.onnx.data",  // External data file
        "BeatThis_small0.onnx",
        "BeatThis_small1.onnx",
        "BeatThis_small2.onnx"
    )
    
    /**
     * Copies ONNX models from assets to cache directory.
     * Returns the cache directory path for Rust to use.
     */
    suspend fun ensureModelsReady(context: Context): String =
        withContext(Dispatchers.IO) {
            val modelDir = File(context.cacheDir, "models")
            modelDir.mkdirs()
            
            MODEL_FILES.forEach { modelName ->
                val destFile = File(modelDir, modelName)
                if (!destFile.exists()) {
                    context.assets.open(modelName).use { input ->
                        destFile.outputStream().use { output ->
                            input.copyTo(output)
                        }
                    }
                }
            }
            
            modelDir.absolutePath
        }
}
```

#### 3.3.2 ONNX Runtime Static Linking

To avoid bundling separate `.so` files, use static linking:

```toml
# remixatron-android/Cargo.toml

[dependencies]
# Use static linking to avoid .so bundling complexity
ort = { version = "=2.0.0-rc.10", default-features = false, features = [] }

# Bundle the ONNX Runtime library
onnxruntime-sys = { version = "0.0.14", features = ["static"] }
```

> [!NOTE]
> If static linking causes issues, bundle `libonnxruntime.so` in `jniLibs/arm64-v8a/`.

---

### 3.4 Complete Kotlin JNI Bridge

The `RemixatronBridge` object declares all native methods available from Rust:

```kotlin
// bridge/RemixatronBridge.kt
package com.remixatron.bridge

/**
 * JNI bridge to the Rust remixatron-android library.
 * All methods throw RemixatronException on native errors.
 */
object RemixatronBridge {
    
    init {
        System.loadLibrary("remixatron_android")
    }
    
    // Analysis
    @Throws(RemixatronException::class)
    external fun analyzeTrack(audioPath: String, modelDir: String, progressCallback: ProgressCallback): Long
    
    // Jukebox / JIT
    external fun createJukebox(analysisHandle: Long): Long
    external fun getNextBeat(jukeboxHandle: Long): Int
    external fun getNextBeatExtended(jukeboxHandle: Long): PlayInstruction
    external fun resetJukebox(jukeboxHandle: Long)
    
    // Data Accessors
    external fun getBeats(analysisHandle: Long): Array<Beat>
    external fun getSegments(analysisHandle: Long): Array<Segment>
    external fun getMetadata(analysisHandle: Long): TrackMetadata
    external fun getJumpCandidates(analysisHandle: Long, beatIndex: Int): IntArray
    
    // Audio Buffer (for Oboe)
    external fun getAudioBufferPtr(analysisHandle: Long): Long
    external fun getAudioBufferLen(analysisHandle: Long): Int
    external fun getAudioSampleRate(analysisHandle: Long): Int
    
    // Memory Management (Oboe MUST be stopped before freeAnalysisHandle!)
    external fun freeAnalysisHandle(handle: Long)
    external fun freeJukeboxHandle(handle: Long)
}

fun interface ProgressCallback {
    fun onProgress(message: String, progress: Float)
}
```

---

### 3.5 Complete Beat Data Structure

The `Beat` struct must include ALL fields from the desktop implementation:

```kotlin
// Beat.kt
data class Beat(
    val id: Int,                    // Unique index
    val startSeconds: Float,        // Start time in seconds
    val durationSeconds: Float,     // Duration in seconds
    val startSample: Long,          // Start sample at playback rate
    val endSample: Long,            // End sample at playback rate
    val barPosition: Int,           // 0-3 within a measure
    val segmentId: Int,             // Structural segment instance
    val clusterId: Int,             // Functional cluster (Verse, Chorus, etc.)
    val intraSegmentIndex: Int,     // Position within segment (for segment-local logic)
    val quartile: Int,              // 0-3 song quartile (for Quartile Busting)
    val jumpCandidates: IntArray    // Valid jump targets from this beat
) {
    // Sample conversion for different rates
    fun startSampleAt(sampleRate: Int): Long = 
        (startSeconds * sampleRate).toLong()
    
    fun endSampleAt(sampleRate: Int): Long = 
        ((startSeconds + durationSeconds) * sampleRate).toLong()
}

data class Segment(
    val id: Int,
    val startBeat: Int,
    val endBeat: Int,
    val clusterId: Int,
    val label: String
)

data class TrackMetadata(
    val title: String,
    val artist: String,
    val albumArtBase64: String?,
    val durationSeconds: Float,
    val beatCount: Int,
    val segmentCount: Int,
    val clusterCount: Int
)
```

---

### 3.6 Audio Buffer Ownership Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AUDIO DATA FLOW                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. DECODE (Rust)                                                    │
│     Symphonia reads file → Vec<f32> (interleaved stereo)             │
│     Rubato resamples for ML (22050 Hz) - analysis only               │
│     Original samples kept at native rate (44100/48000 Hz)            │
│                                                                      │
│  2. STORE (Rust)                                                     │
│     AudioBuffer { samples: Vec<f32>, sample_rate, channels }         │
│     Stored in AnalysisHandle on Rust heap                            │
│     Rust OWNS this memory                                            │
│                                                                      │
│  3. SHARE (JNI)                                                      │
│     Kotlin calls getAudioBufferPtr() → raw pointer (jlong)           │
│     Kotlin calls getAudioBufferLen() → sample count                  │
│     Kotlin calls getAudioSampleRate() → sample rate                  │
│     Kotlin passes these to C++ Oboe engine                           │
│                                                                      │
│  4. PLAYBACK (C++)                                                   │
│     Oboe reads from pointer (read-only)                              │
│     Does NOT copy the buffer (zero-copy approach)                    │
│     Seeks by adjusting playhead index                                │
│                                                                      │
│  5. FREE (Rust)                                                      │
│     Kotlin calls freeAnalysisHandle()                                │
│     Rust drops Box<AnalysisHandle>                                   │
│     C++ Oboe engine MUST be stopped before this call                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Critical Rule:** The Kotlin side must ensure Oboe playback is stopped BEFORE calling `freeAnalysisHandle()`.

```kotlin
// Safe cleanup pattern
fun cleanup() {
    // 1. Stop playback first
    oboeEngine.stop()
    
    // 2. Wait for Oboe to fully stop (audio thread exit)
    oboeEngine.awaitStop()
    
    // 3. Now safe to free Rust memory
    jukeboxHandle?.let { bridge.freeJukeboxHandle(it) }
    analysisHandle?.let { bridge.freeAnalysisHandle(it) }
}
```

---

### 3.7 Sample Rate Architecture

| Pipeline Stage | Sample Rate | Purpose |
|----------------|-------------|---------|
| Original Audio | 44100/48000 Hz | Playback quality |
| ML Analysis | 22050 Hz | ONNX model expectation |
| Beat Timestamps | Seconds (float) | Rate-independent |
| Playback Scheduling | Samples at original rate | Sample-accurate timing |

**Conversion:**
```kotlin
// Beat times are stored in seconds
val beatStartSample = (beat.startSeconds * audioSampleRate).toLong()
```

---

### 3.8 Threading Architecture

> [!CAUTION]
> Oboe audio callbacks run on a high-priority audio thread. JNI calls from this thread cause priority inversion and potential deadlocks.

#### 3.8.1 Event Queue Pattern

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   AUDIO THREAD      │     │   EVENT QUEUE       │     │   MAIN THREAD       │
│   (Oboe callback)   │     │   (Lock-free)       │     │   (Kotlin/Compose)  │
├─────────────────────┤     ├─────────────────────┤     ├─────────────────────┤
│                     │     │                     │     │                     │
│  onAudioReady() {   │     │                     │     │  LaunchedEffect {   │
│    // Play samples  │     │                     │     │    while(true) {    │
│    // Detect beat   │────▶│  beatQueue.push(42) │────▶│      val beat =     │
│    // NO JNI HERE!  │     │                     │     │        queue.poll() │
│  }                  │     │                     │     │      updateUI(beat) │
│                     │     │                     │     │    }                │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
```

#### 3.8.2 Implementation

```cpp
// oboe_engine.cpp - Lock-free queue for beat events

#include <atomic>
#include <array>

class BeatEventQueue {
public:
    static constexpr size_t QUEUE_SIZE = 64;
    
    void push(int beatIndex) {
        size_t next = (writePos_.load() + 1) % QUEUE_SIZE;
        if (next != readPos_.load()) {
            buffer_[writePos_.load()] = beatIndex;
            writePos_.store(next);
        }
    }
    
    bool pop(int& beatIndex) {
        if (readPos_.load() == writePos_.load()) return false;
        beatIndex = buffer_[readPos_.load()];
        readPos_.store((readPos_.load() + 1) % QUEUE_SIZE);
        return true;
    }
    
private:
    std::array<int, QUEUE_SIZE> buffer_;
    std::atomic<size_t> writePos_{0};
    std::atomic<size_t> readPos_{0};
};

// In audio callback:
void onAudioReady() {
    // ... play samples ...
    
    if (crossedBeatBoundary) {
        beatQueue_.push(currentBeatIndex);  // Lock-free, fast
        // DO NOT call JNI here!
    }
}
```

```kotlin
// Kotlin side polls the queue
class OboePlaybackEngine {
    @Volatile private var running = false
    
    fun startPolling(scope: CoroutineScope) {
        running = true
        scope.launch(Dispatchers.Default) {
            while (running) {
                val beat = nativePollBeatEvent()
                if (beat >= 0) {
                    // Safe to call JNI from this thread
                    val nextBeat = bridge.getNextBeat(jukeboxHandle)
                    if (nextBeat != beat + 1) {
                        nativeSeekToBeat(nextBeat)
                    }
                    _currentBeat.emit(beat)
                }
                delay(8) // ~120 Hz polling
            }
        }
    }
    
    private external fun nativePollBeatEvent(): Int  // Returns -1 if empty
}
```

#### 3.8.3 Complete Oboe C++ Implementation

```cpp
// app/src/main/cpp/oboe_engine.cpp

#include <oboe/Oboe.h>
#include <jni.h>
#include <atomic>
#include <array>
#include <vector>
#include <android/log.h>

#define LOG_TAG "OboeEngine"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

// Lock-free queue for beat events
class BeatEventQueue {
public:
    static constexpr size_t QUEUE_SIZE = 64;
    std::array<int, QUEUE_SIZE> buffer_;
    std::atomic<size_t> writePos_{0};
    std::atomic<size_t> readPos_{0};
    
    void push(int beatIndex) {
        size_t next = (writePos_.load() + 1) % QUEUE_SIZE;
        if (next != readPos_.load()) {
            buffer_[writePos_.load()] = beatIndex;
            writePos_.store(next);
        }
    }
    
    int pop() {
        if (readPos_.load() == writePos_.load()) return -1;
        int val = buffer_[readPos_.load()];
        readPos_.store((readPos_.load() + 1) % QUEUE_SIZE);
        return val;
    }
};

class RemixatronPlayer : public oboe::AudioStreamCallback {
public:
    RemixatronPlayer(const float* audioBuffer, int bufferLen, int sampleRate,
                     const int64_t* beatStartSamples, int beatCount)
        : audioBuffer_(audioBuffer)
        , bufferLen_(bufferLen)
        , sampleRate_(sampleRate)
        , beatCount_(beatCount)
    {
        beatStartSamples_.assign(beatStartSamples, beatStartSamples + beatCount);
        playheadSample_ = 0;
        currentBeatIndex_ = 0;
    }
    
    oboe::DataCallbackResult onAudioReady(
        oboe::AudioStream* stream,
        void* audioData,
        int32_t numFrames
    ) override {
        auto* output = static_cast<float*>(audioData);
        int channels = stream->getChannelCount();
        
        for (int frame = 0; frame < numFrames; frame++) {
            // Read stereo samples from buffer
            int64_t sampleIdx = playheadSample_ * 2; // Interleaved stereo
            if (sampleIdx + 1 < bufferLen_) {
                output[frame * channels] = audioBuffer_[sampleIdx];
                output[frame * channels + 1] = audioBuffer_[sampleIdx + 1];
            } else {
                output[frame * channels] = 0.0f;
                output[frame * channels + 1] = 0.0f;
            }
            
            playheadSample_++;
            
            // Check beat boundary crossing
            if (currentBeatIndex_ < beatCount_ - 1) {
                if (playheadSample_ >= beatStartSamples_[currentBeatIndex_ + 1]) {
                    beatQueue_.push(currentBeatIndex_);
                    currentBeatIndex_++;
                }
            }
        }
        
        return oboe::DataCallbackResult::Continue;
    }
    
    void seekToBeat(int beatIndex) {
        if (beatIndex >= 0 && beatIndex < beatCount_) {
            currentBeatIndex_ = beatIndex;
            playheadSample_ = beatStartSamples_[beatIndex];
            LOGI("Seek to beat %d (sample %lld)", beatIndex, playheadSample_);
        }
    }
    
    int pollBeatEvent() { return beatQueue_.pop(); }
    
private:
    const float* audioBuffer_;
    int bufferLen_;
    int sampleRate_;
    int beatCount_;
    std::vector<int64_t> beatStartSamples_;
    std::atomic<int64_t> playheadSample_;
    std::atomic<int> currentBeatIndex_;
    BeatEventQueue beatQueue_;
};

// Global player instance (one at a time)
static RemixatronPlayer* gPlayer = nullptr;
static oboe::AudioStream* gStream = nullptr;

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_remixatron_audio_OboePlaybackEngine_nativeCreateStream(
    JNIEnv* env, jobject thiz,
    jint sampleRate, jint channelCount,
    jlong audioBufferPtr, jint audioBufferLen,
    jlongArray beatStartSamplesArr
) {
    auto* audioBuffer = reinterpret_cast<const float*>(audioBufferPtr);
    
    // Get beat timestamps
    jsize beatCount = env->GetArrayLength(beatStartSamplesArr);
    jlong* beats = env->GetLongArrayElements(beatStartSamplesArr, nullptr);
    
    gPlayer = new RemixatronPlayer(audioBuffer, audioBufferLen, sampleRate,
                                    beats, beatCount);
    env->ReleaseLongArrayElements(beatStartSamplesArr, beats, 0);
    
    oboe::AudioStreamBuilder builder;
    builder.setDirection(oboe::Direction::Output)
           ->setPerformanceMode(oboe::PerformanceMode::LowLatency)
           ->setSharingMode(oboe::SharingMode::Exclusive)
           ->setSampleRate(sampleRate)
           ->setChannelCount(channelCount)
           ->setFormat(oboe::AudioFormat::Float)
           ->setCallback(gPlayer);
    
    oboe::Result result = builder.openStream(&gStream);
    if (result != oboe::Result::OK) {
        LOGI("Failed to open stream: %s", oboe::convertToText(result));
        delete gPlayer;
        gPlayer = nullptr;
        return 0;
    }
    
    return reinterpret_cast<jlong>(gStream);
}

JNIEXPORT void JNICALL
Java_com_remixatron_audio_OboePlaybackEngine_nativeStart(JNIEnv*, jobject, jlong) {
    if (gStream) gStream->requestStart();
}

JNIEXPORT void JNICALL
Java_com_remixatron_audio_OboePlaybackEngine_nativePause(JNIEnv*, jobject, jlong) {
    if (gStream) gStream->requestPause();
}

JNIEXPORT void JNICALL
Java_com_remixatron_audio_OboePlaybackEngine_nativeStop(JNIEnv*, jobject, jlong) {
    if (gStream) gStream->requestStop();
}

JNIEXPORT void JNICALL
Java_com_remixatron_audio_OboePlaybackEngine_nativeDestroy(JNIEnv*, jobject, jlong) {
    if (gStream) {
        gStream->close();
        delete gStream;
        gStream = nullptr;
    }
    if (gPlayer) {
        delete gPlayer;
        gPlayer = nullptr;
    }
}

JNIEXPORT void JNICALL
Java_com_remixatron_audio_OboePlaybackEngine_nativeSeekToBeat(JNIEnv*, jobject, jlong, jint beatIndex) {
    if (gPlayer) gPlayer->seekToBeat(beatIndex);
}

JNIEXPORT jint JNICALL
Java_com_remixatron_audio_OboePlaybackEngine_nativePollBeatEvent(JNIEnv*, jobject) {
    return gPlayer ? gPlayer->pollBeatEvent() : -1;
}

JNIEXPORT void JNICALL
Java_com_remixatron_audio_OboePlaybackEngine_nativeAwaitStreamClose(JNIEnv*, jobject, jlong) {
    // Oboe close() is synchronous - blocks until audio thread exits
    if (gStream && gStream->getState() != oboe::StreamState::Closed) {
        gStream->close();
    }
}

} // extern "C"
```

---

### 3.9 Android Media Integration

#### 3.9.1 Audio Focus

```kotlin
// AudioFocusManager.kt
class AudioFocusManager(private val context: Context) {
    private val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    
    private val focusRequest = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN)
        .setAudioAttributes(
            AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                .build()
        )
        .setOnAudioFocusChangeListener { focusChange ->
            when (focusChange) {
                AudioManager.AUDIOFOCUS_LOSS -> onFocusLost()
                AudioManager.AUDIOFOCUS_LOSS_TRANSIENT -> onFocusLostTransient()
                AudioManager.AUDIOFOCUS_LOSS_TRANSIENT_CAN_DUCK -> onDuck()
                AudioManager.AUDIOFOCUS_GAIN -> onFocusGained()
            }
        }
        .build()
    
    // Callback interface for focus change events
    var onFocusChange: ((FocusState) -> Unit)? = null
    
    enum class FocusState { FOCUSED, PAUSED, DUCKED, LOST }
    
    private fun onFocusLost() {
        onFocusChange?.invoke(FocusState.LOST)
    }
    
    private fun onFocusLostTransient() {
        onFocusChange?.invoke(FocusState.PAUSED)
    }
    
    private fun onDuck() {
        onFocusChange?.invoke(FocusState.DUCKED)
    }
    
    private fun onFocusGained() {
        onFocusChange?.invoke(FocusState.FOCUSED)
    }
    
    fun requestFocus(): Boolean {
        return audioManager.requestAudioFocus(focusRequest) == 
            AudioManager.AUDIOFOCUS_REQUEST_GRANTED
    }
    
    fun abandonFocus() {
        audioManager.abandonAudioFocusRequest(focusRequest)
    }
}
```

#### 3.9.2 Foreground Service + MediaSession

```kotlin
// RemixatronPlaybackService.kt
@AndroidEntryPoint
class RemixatronPlaybackService : Service() {
    
    @Inject lateinit var playbackEngine: OboePlaybackEngine
    
    private lateinit var mediaSession: MediaSessionCompat
    private lateinit var notificationManager: NotificationManager
    
    override fun onCreate() {
        super.onCreate()
        
        mediaSession = MediaSessionCompat(this, "Remixatron").apply {
            setCallback(MediaSessionCallback())
            isActive = true
        }
        
        startForeground(NOTIFICATION_ID, createNotification())
    }
    
    private inner class MediaSessionCallback : MediaSessionCompat.Callback() {
        override fun onPlay() { playbackEngine.play() }
        override fun onPause() { playbackEngine.pause() }
        override fun onStop() { stopSelf() }
    }
    
    private fun createNotification(): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(currentTrack?.title ?: "Remixatron")
            .setContentText(currentTrack?.artist ?: "Infinite Jukebox")
            .addAction(/* play/pause */)
            .setStyle(
                androidx.media.app.NotificationCompat.MediaStyle()
                    .setMediaSession(mediaSession.sessionToken)
            )
            .build()
    }
}
```

---

### 3.10 Error Handling Patterns

#### 3.10.1 Rust → Java Exception Bridge

```rust
// remixatron-android/src/jni_exceptions.rs

use jni::JNIEnv;

/// Custom exception class for Remixatron errors.
const EXCEPTION_CLASS: &str = "com/remixatron/bridge/RemixatronException";

/// Throws a Java exception from Rust.
pub fn throw_remixatron_exception(env: &mut JNIEnv, message: &str) {
    // Try custom exception first
    if env.find_class(EXCEPTION_CLASS).is_ok() {
        let _ = env.throw_new(EXCEPTION_CLASS, message);
    } else {
        // Fallback to RuntimeException
        let _ = env.throw_new("java/lang/RuntimeException", message);
    }
}

/// Wraps a fallible operation, converting errors to exceptions.
pub fn jni_try<T, E: std::fmt::Display>(
    env: &mut JNIEnv,
    result: Result<T, E>,
    default: T,
) -> T {
    match result {
        Ok(value) => value,
        Err(e) => {
            throw_remixatron_exception(env, &e.to_string());
            default
        }
    }
}
```

#### 3.10.2 Kotlin Exception Wrapper

```kotlin
// RemixatronException.kt
class RemixatronException(message: String) : Exception(message)

// Safe JNI call wrapper
inline fun <T> safeJniCall(crossinline block: () -> T): Result<T> {
    return try {
        Result.success(block())
    } catch (e: RemixatronException) {
        Result.failure(e)
    } catch (e: UnsatisfiedLinkError) {
        Result.failure(RemixatronException("Native library not loaded: ${e.message}"))
    } catch (e: Exception) {
        Result.failure(RemixatronException("JNI call failed: ${e.message}"))
    }
}
```

#### 3.10.3 User-Facing Error States

```kotlin
sealed class AnalysisError {
    data class FileNotFound(val path: String) : AnalysisError()
    data class UnsupportedFormat(val format: String) : AnalysisError()
    data class AudioTooShort(val durationSeconds: Float) : AnalysisError()
    data class ModelLoadFailed(val modelName: String) : AnalysisError()
    data class OutOfMemory(val requiredMb: Int) : AnalysisError()
    data class Unknown(val message: String) : AnalysisError()
}
```

---

### 3.11 Kotlin Audio Engine (Updated)

```kotlin
// OboePlaybackEngine.kt
package com.remixatron.audio

/**
 * Low-latency audio playback engine using Google Oboe.
 *
 * Key design decisions:
 * - Audio buffer lives in Rust (zero-copy via pointer)
 * - Beat events queued lock-free, polled from Kotlin coroutine
 * - JIT decisions made on Kotlin thread, not audio thread
 */
class OboePlaybackEngine(
    private val bridge: RemixatronBridge,
    private val audioFocusManager: AudioFocusManager
) {
    private var streamPtr: Long = 0
    private var analysisHandle: Long = 0
    private var jukeboxHandle: Long = 0
    
    private val _playbackState = MutableStateFlow(PlaybackState.STOPPED)
    val playbackState: StateFlow<PlaybackState> = _playbackState.asStateFlow()
    
    private val _currentBeat = MutableStateFlow(0)
    val currentBeat: StateFlow<Int> = _currentBeat.asStateFlow()
    
    private val _playInstruction = MutableStateFlow<PlayInstruction?>(null)
    val playInstruction: StateFlow<PlayInstruction?> = _playInstruction.asStateFlow()
    
    private var pollingJob: Job? = null
    
    companion object {
        init {
            try {
                System.loadLibrary("oboe_engine")
                System.loadLibrary("remixatron_android")
            } catch (e: UnsatisfiedLinkError) {
                Log.e("OboeEngine", "Failed to load native libraries", e)
            }
        }
    }
    
    fun initialize(analysisHandle: Long, scope: CoroutineScope): Result<Unit> {
        return safeJniCall {
            this.analysisHandle = analysisHandle
            this.jukeboxHandle = bridge.createJukebox(analysisHandle)
            
            val bufferPtr = bridge.getAudioBufferPtr(analysisHandle)
            val bufferLen = bridge.getAudioBufferLen(analysisHandle)
            val sampleRate = bridge.getAudioSampleRate(analysisHandle)
            
            // Build beat start samples array for C++ beat tracking
            val beats = bridge.getBeats(analysisHandle)
            val beatStartSamples = LongArray(beats.size) { i ->
                beats[i].startSampleAt(sampleRate)
            }
            
            streamPtr = nativeCreateStream(sampleRate, 2, bufferPtr, bufferLen, beatStartSamples)
            
            // Start beat event polling
            startPolling(scope)
        }
    }
    
    fun play(): Result<Unit> {
        if (!audioFocusManager.requestFocus()) {
            return Result.failure(RemixatronException("Could not obtain audio focus"))
        }
        
        return safeJniCall {
            nativeStart(streamPtr)
            _playbackState.value = PlaybackState.PLAYING
        }
    }
    
    fun pause(): Result<Unit> = safeJniCall {
        nativePause(streamPtr)
        _playbackState.value = PlaybackState.PAUSED
    }
    
    fun stop(): Result<Unit> = safeJniCall {
        nativeStop(streamPtr)
        audioFocusManager.abandonFocus()
        _playbackState.value = PlaybackState.STOPPED
    }
    
    fun cleanup() {
        pollingJob?.cancel()
        if (streamPtr != 0L) {
            nativeStop(streamPtr)
            nativeDestroy(streamPtr)
            streamPtr = 0
        }
        if (jukeboxHandle != 0L) {
            bridge.freeJukeboxHandle(jukeboxHandle)
            jukeboxHandle = 0
        }
        // Note: analysisHandle is NOT freed here — caller owns it
    }
    
    private fun startPolling(scope: CoroutineScope) {
        pollingJob = scope.launch(Dispatchers.Default) {
            while (isActive) {
                val beatIndex = nativePollBeatEvent()
                if (beatIndex >= 0) {
                    // JIT decision (safe from this thread)
                    val instruction = bridge.getNextBeatExtended(jukeboxHandle)
                    
                    // Check if jump needed
                    if (instruction.beatId != beatIndex + 1) {
                        nativeSeekToBeat(streamPtr, instruction.beatId)
                    }
                    
                    // Emit UI updates
                    _currentBeat.emit(beatIndex)
                    _playInstruction.emit(instruction)
                }
                delay(8) // ~120 Hz polling
            }
        }
    }
    
    /**
     * Blocks until the audio stream has fully stopped.
     * Must be called before freeing native handles.
     */
    suspend fun awaitStop() = withContext(Dispatchers.IO) {
        pollingJob?.join()
        nativeAwaitStreamClose(streamPtr)
    }
    
    // Native methods
    private external fun nativeCreateStream(
        sampleRate: Int,
        channelCount: Int,
        audioBufferPtr: Long,
        audioBufferLen: Int,
        beatStartSamples: LongArray  // Beat boundaries in samples
    ): Long
    
    private external fun nativeStart(streamPtr: Long)
    private external fun nativePause(streamPtr: Long)
    private external fun nativeStop(streamPtr: Long)
    private external fun nativeDestroy(streamPtr: Long)
    private external fun nativeSeekToBeat(streamPtr: Long, beatIndex: Int)
    private external fun nativePollBeatEvent(): Int
    private external fun nativeAwaitStreamClose(streamPtr: Long)  // Blocks until closed
}

enum class PlaybackState { STOPPED, PLAYING, PAUSED }

data class PlayInstruction(
    val beatId: Int,
    val seqLen: Int,
    val seqPos: Int
)
```

---

### 3.12 UI Layer with Performance Optimizations

#### 3.12.1 Pre-Calculated Visualization Data

```kotlin
// VisualizationState.kt

/**
 * Pre-calculated visualization data to avoid per-frame computation.
 * Calculate once after analysis, reuse every frame.
 */
class VisualizationState(
    beats: List<Beat>,
    private val canvasSize: IntSize
) {
    val center = Offset(canvasSize.width / 2f, canvasSize.height / 2f)
    val radius = minOf(canvasSize.width, canvasSize.height) / 2.2f
    
    // Pre-calculated beat positions (calculate once)
    val beatPositions: FloatArray = FloatArray(beats.size * 2).also { arr ->
        beats.forEachIndexed { index, _ ->
            val angle = (index.toFloat() / beats.size) * 360f - 90f
            val radians = Math.toRadians(angle.toDouble())
            arr[index * 2] = center.x + radius * cos(radians).toFloat()
            arr[index * 2 + 1] = center.y + radius * sin(radians).toFloat()
        }
    }
    
    // Pre-grouped beats by cluster (for efficient coloring)
    val beatsByCluster: Map<Int, List<Int>> = beats
        .mapIndexed { index, beat -> index to beat.clusterId }
        .groupBy({ it.second }, { it.first })
}
```

#### 3.12.2 Optimized Canvas Drawing

```kotlin
@Composable
fun VisualizationCanvas(
    vizState: VisualizationState,
    beats: List<Beat>,
    currentBeat: Int,
    jumpCandidates: IntArray,
    modifier: Modifier = Modifier
) {
    val currentCluster = remember(currentBeat) { beats.getOrNull(currentBeat)?.clusterId ?: 0 }
    
    Canvas(modifier = modifier.fillMaxSize()) {
        // Use drawPoints for batch rendering (much faster than individual drawCircle)
        val points = FloatArray(beats.size * 2)
        System.arraycopy(vizState.beatPositions, 0, points, 0, points.size)
        
        // Draw all non-active beats in one call
        drawPoints(
            points = points.toList().chunked(2).map { Offset(it[0], it[1]) },
            pointMode = PointMode.Points,
            color = Color.White.copy(alpha = 0.3f),
            strokeWidth = 4.dp.toPx(),
            cap = StrokeCap.Round
        )
        
        // Overdraw cluster-matched beats
        vizState.beatsByCluster[currentCluster]?.forEach { index ->
            val x = vizState.beatPositions[index * 2]
            val y = vizState.beatPositions[index * 2 + 1]
            drawCircle(
                color = Color.Magenta.copy(alpha = 0.7f),
                radius = 4.dp.toPx(),
                center = Offset(x, y)
            )
        }
        
        // Draw current beat (brightest)
        val cx = vizState.beatPositions[currentBeat * 2]
        val cy = vizState.beatPositions[currentBeat * 2 + 1]
        drawCircle(
            color = Color.Cyan,
            radius = 6.dp.toPx(),
            center = Offset(cx, cy)
        )
        
        // Draw jump arcs efficiently
        jumpCandidates.forEach { targetBeat ->
            drawJumpArc(vizState, currentBeat, targetBeat, beats.size)
        }
    }
}
```

---

## 4. Build System

### 4.1 Project Structure (Updated)

```
remixatron-mobile/
├── app/                              # Android application
│   ├── src/main/
│   │   ├── kotlin/com/remixatron/
│   │   │   ├── MainActivity.kt
│   │   │   ├── RemixatronApplication.kt
│   │   │   ├── ui/                   # Compose screens
│   │   │   │   ├── OnboardingScreen.kt
│   │   │   │   ├── PlayerScreen.kt
│   │   │   │   └── VisualizationCanvas.kt
│   │   │   ├── audio/                # Oboe engine
│   │   │   │   ├── OboePlaybackEngine.kt
│   │   │   │   └── AudioFocusManager.kt
│   │   │   ├── bridge/               # JNI interface
│   │   │   │   ├── RemixatronBridge.kt
│   │   │   │   ├── RemixatronException.kt
│   │   │   │   └── DataClasses.kt
│   │   │   ├── download/             # youtubedl-android
│   │   │   │   └── DownloadManager.kt
│   │   │   ├── media/                # MediaSession + Service
│   │   │   │   ├── RemixatronPlaybackService.kt
│   │   │   │   └── AudioFocusManager.kt
│   │   │   └── domain/               # ViewModels, Repository
│   │   │       ├── RemixatronViewModel.kt
│   │   │       └── RemixatronRepository.kt
│   │   ├── cpp/                      # Oboe C++ code
│   │   │   ├── oboe_engine.cpp
│   │   │   ├── beat_event_queue.h
│   │   │   └── CMakeLists.txt
│   │   ├── assets/                   # ONNX models
│   │   │   ├── MelSpectrogram_Ultimate.onnx
│   │   │   ├── MelSpectrogram_Ultimate.onnx.data
│   │   │   ├── BeatThis_small0.onnx
│   │   │   ├── BeatThis_small1.onnx
│   │   │   └── BeatThis_small2.onnx
│   │   └── jniLibs/                  # Pre-built Rust .so files
│   │       ├── arm64-v8a/
│   │       │   └── libremixatron_android.so
│   │       ├── armeabi-v7a/
│   │       └── x86_64/
│   ├── proguard-rules.pro
│   └── build.gradle.kts
│
├── rust/
│   ├── remixatron-core/              # Shared DSP (git submodule)
│   └── remixatron-android/           # Android JNI wrapper
│       ├── Cargo.toml
│       ├── src/
│       └── build.sh
│
├── settings.gradle.kts
└── build.gradle.kts
```

### 4.2 Gradle Configuration (Updated)

```kotlin
// settings.gradle.kts
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        maven("https://jitpack.io")  // Required for youtubedl-android
    }
}
```

```kotlin
// app/build.gradle.kts
plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.hilt)
    alias(libs.plugins.ksp)
}

android {
    namespace = "com.remixatron"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.remixatron"
        minSdk = 24  // Android 7.0 (Oboe AAudio support)
        targetSdk = 34
        versionCode = 1
        versionName = "0.4.0"

        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64")
        }

        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
                arguments += "-DANDROID_STL=c++_shared"
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildFeatures {
        compose = true
    }
    
    // Enable App Bundle for per-ABI splits (reduces download size)
    bundle {
        language { enableSplit = true }
        density { enableSplit = true }
        abi { enableSplit = true }  // Important: ~50% APK size reduction
    }
}

dependencies {
    // Compose
    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.material3)

    // Lifecycle + ViewModel
    implementation(libs.lifecycle.viewmodel.compose)
    implementation(libs.lifecycle.runtime.compose)
    implementation(libs.lifecycle.service)

    // Hilt
    implementation(libs.hilt.android)
    ksp(libs.hilt.compiler)

    // youtubedl-android (yt-dlp + Python + FFmpeg bundle)
    implementation("com.github.yausername.youtubedl-android:library:0.15.0")
    implementation("com.github.yausername.youtubedl-android:ffmpeg:0.15.0")

    // Media3 for MediaSession
    implementation(libs.media3.session)
    implementation(libs.media3.exoplayer)  // For MediaSession only, not playback

    // Coil (image loading)
    implementation(libs.coil.compose)

    // Coroutines
    implementation(libs.kotlinx.coroutines.android)
}
```

### 4.3 Version Catalog (`gradle/libs.versions.toml`)

```toml
[versions]
agp = "8.2.0"
kotlin = "1.9.21"
compose-bom = "2024.01.00"
compose-compiler = "1.5.7"
lifecycle = "2.7.0"
hilt = "2.50"
media3 = "1.2.0"
coil = "2.5.0"
coroutines = "1.7.3"

[libraries]
compose-bom = { group = "androidx.compose", name = "compose-bom", version.ref = "compose-bom" }
compose-ui = { group = "androidx.compose.ui", name = "ui" }
compose-material3 = { group = "androidx.compose.material3", name = "material3" }
lifecycle-viewmodel-compose = { group = "androidx.lifecycle", name = "lifecycle-viewmodel-compose", version.ref = "lifecycle" }
lifecycle-runtime-compose = { group = "androidx.lifecycle", name = "lifecycle-runtime-compose", version.ref = "lifecycle" }
lifecycle-service = { group = "androidx.lifecycle", name = "lifecycle-service", version.ref = "lifecycle" }
hilt-android = { group = "com.google.dagger", name = "hilt-android", version.ref = "hilt" }
hilt-compiler = { group = "com.google.dagger", name = "hilt-compiler", version.ref = "hilt" }
media3-session = { group = "androidx.media3", name = "media3-session", version.ref = "media3" }
media3-exoplayer = { group = "androidx.media3", name = "media3-exoplayer", version.ref = "media3" }
coil-compose = { group = "io.coil-kt", name = "coil-compose", version.ref = "coil" }
kotlinx-coroutines-android = { group = "org.jetbrains.kotlinx", name = "kotlinx-coroutines-android", version.ref = "coroutines" }

[plugins]
android-application = { id = "com.android.application", version.ref = "agp" }
kotlin-android = { id = "org.jetbrains.kotlin.android", version.ref = "kotlin" }
kotlin-compose = { id = "org.jetbrains.kotlin.plugin.compose", version.ref = "kotlin" }
hilt = { id = "com.google.dagger.hilt.android", version.ref = "hilt" }
ksp = { id = "com.google.devtools.ksp", version = "1.9.21-1.0.16" }
```

### 4.4 ProGuard Rules

```proguard
# proguard-rules.pro

# Keep all JNI bridge classes and methods
-keep class com.remixatron.bridge.** { *; }
-keep class com.remixatron.audio.** { native <methods>; }

# Keep @Keep annotated elements
-keep,allowobfuscation @interface androidx.annotation.Keep
-keep @androidx.annotation.Keep class * { *; }
-keepclassmembers class * {
    @androidx.annotation.Keep *;
}

# Keep data classes used in JNI (Kotlin serialization)
-keepclassmembers class com.remixatron.bridge.Beat { *; }
-keepclassmembers class com.remixatron.bridge.Segment { *; }
-keepclassmembers class com.remixatron.bridge.TrackMetadata { *; }
-keepclassmembers class com.remixatron.bridge.PlayInstruction { *; }

# youtubedl-android
-keep class com.yausername.youtubedl_android.** { *; }
-keep class com.yausername.ffmpeg.** { *; }
```

### 4.5 CMakeLists.txt (Updated)

```cmake
cmake_minimum_required(VERSION 3.22.1)
project("oboe_engine")

# Find Oboe package
find_package(oboe REQUIRED CONFIG)

# Add our native library
add_library(oboe_engine SHARED
    oboe_engine.cpp
)

# Include headers
target_include_directories(oboe_engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link libraries
target_link_libraries(oboe_engine
    oboe::oboe
    android
    log
)

# Enable LTO for release builds
set_property(TARGET oboe_engine PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
```

---

## 5. Implementation Phases (Updated)

### Phase 0: Core Extraction (Week 0-1) — NEW
- [ ] Create `remixatron-core` crate from desktop code
- [ ] Extract JIT logic from `playback_engine.rs` into `jit/engine.rs`
- [ ] Remove all Tauri/Kira dependencies from core
- [ ] Add `#[cfg(feature = "android")]` gates where needed
- [ ] Verify desktop still builds against new core

### Phase 1: Foundation (Week 1-2)
- [ ] Set up Android project (Gradle, Compose, Hilt)
- [ ] Configure JitPack repository for youtubedl-android
- [ ] Create `remixatron-android` Rust crate
- [ ] Implement JNI library loading with error handling
- [ ] Bundle ONNX models as assets
- [ ] Implement asset-to-cache copy for models

### Phase 2: Analysis Pipeline (Week 3-4)
- [ ] Implement full JNI bridge (AnalysisHandle, JukeboxHandle)
- [ ] Implement Rust exception → Java exception bridge
- [ ] Integrate Symphonia/Rubato loading
- [ ] Test ONNX inference on device (static linking)
- [ ] Implement progress callback via JNI GlobalRef
- [ ] Build basic Compose UI (file picker + progress)

### Phase 3: Audio Engine (Week 5-7)
- [ ] Implement Oboe audio stream (C++)
- [ ] Implement lock-free beat event queue
- [ ] Implement audio buffer pointer sharing
- [ ] Implement beat-boundary detection
- [ ] Wire up Kotlin polling loop for JIT decisions
- [ ] Implement Play/Pause/Stop with audio focus

### Phase 4: Media Integration (Week 8)
- [ ] Implement MediaSessionCompat
- [ ] Create Foreground Service for background playback
- [ ] Build notification with playback controls
- [ ] Handle audio focus changes (duck, pause)

### Phase 5: UI & Visualization (Week 9-10)
- [ ] Pre-calculate beat positions (VisualizationState)
- [ ] Implement optimized circular beat Canvas
- [ ] Implement batch point drawing
- [ ] Build Floating Player Card (M3)
- [ ] Implement jump arc rendering
- [ ] Add Pulse Ring countdown indicator

### Phase 6: Download & Favorites (Week 11-12)
- [ ] Integrate youtubedl-android library
- [ ] Implement yt-dlp auto-update on app launch
- [ ] Implement URL input detection with Early Metadata
- [ ] Build Favorites persistence (Room)
- [ ] Implement Favorites carousel UI

### Phase 7: Polish & Testing (Week 13-14)
- [ ] Device matrix testing (Pixel, Samsung, OnePlus)
- [ ] Performance profiling (Systrace, Perfetto)
- [ ] Memory leak auditing (LeakCanary)
- [ ] Audio latency tuning (buffer sizes)
- [ ] ProGuard testing
- [ ] Prepare Play Store submission

---

## 6. Risk Mitigation (Updated)

| Risk | Mitigation |
|------|------------|
| JIT state corruption | Two-handle architecture isolates mutable state; explicit lifecycle |
| ONNX load failure | Copy-to-cache pattern; static linking preferred |
| Audio buffer use-after-free | Strict cleanup ordering: stop Oboe → free handles |
| JNI exceptions crash app | Try/catch in all JNI calls; exception bridge |
| Oboe latency issues | Start with `PerformanceMode::LowLatency`; tune buffer sizes |
| Beat callback priority inversion | Lock-free queue; no JNI in audio thread |
| APK size (~120MB total) | App Bundle with ABI splits; model quantization if needed |
| yt-dlp 403 errors | Auto-update on first launch; network error handling |
| ProGuard breaks JNI | Explicit keep rules for all bridge classes |
| Visualization jank | Pre-calculate positions; batch point drawing |

---

## 7. Verification Plan (Updated)

### 7.1 Unit Tests

| Test | Location | Command |
|------|----------|---------|
| Rust DSP parity | `remixatron-core` | `cargo test` |
| JIT state machine | `remixatron-core/src/jit` | `cargo test jit` |
| JNI bridge compilation | `remixatron-android` | `cargo ndk build --release` |
| Kotlin ViewModel | `app/src/test` | `./gradlew testDebugUnitTest` |

### 7.2 Integration Tests

| Test | Method |
|------|--------|
| Full analysis pipeline | Instrumented test with sample MP3 |
| Oboe playback | Device test with known audio file |
| JIT decisions | Compare Rust output to desktop for same seed |
| Memory lifecycle | Allocate/free 100x, check for leaks |

### 7.3 Manual Verification

1. **Analysis Verification**
   - Analyze "Stay with Me" on device
   - Compare beat count, segment count, cluster count to desktop
   - Verify metadata extraction (title, artist, album art)

2. **Playback Verification**
   - Play for 10+ minutes
   - Confirm jumps occur at beat boundaries (no clicks)
   - Verify Pulse Ring countdown matches actual jump timing
   - Test pause/resume (no audio glitches)

3. **Background Playback**
   - Verify playback continues when app is minimized
   - Verify notification controls work
   - Verify audio focus (pause for phone call, resume after)

4. **Memory Verification**
   - Profile with Android Studio Profiler
   - Verify no native memory leaks (Rust handles freed)
   - Verify no Java memory leaks (LeakCanary)

---

## 8. Future: iOS Port

With this architecture, an iOS port requires:

| Component | Android | iOS (New) |
|-----------|---------|-----------|
| Rust Core | ✅ Shared (`remixatron-core`) | ✅ Shared |
| JIT Engine | ✅ Shared (`jit/engine.rs`) | ✅ Shared |
| Audio Engine | Oboe (C++) | AVAudioEngine (Swift) |
| UI | Jetpack Compose | SwiftUI |
| Download | youtubedl-android | yt-dlp via Pythonista or custom |

**Estimated iOS effort:** 4-6 weeks (UI + audio only, no DSP work)
