//! # Remixatron Library
//!
//! This is the core library for the Remixatron application. It bridges the gap between the
//! Tauri backend (Rust) and the Frontend UI (JavaScript).
//!
//! ## Responsibilities
//! *   **State Management**: Holds the global application state (`AppState`) including the `JukeboxEngine`.
//! *   **Command Exposure**: Defines `#[tauri::command]` functions callable from the frontend.
//! *   **Lifecycle Management**: Initializes the Tauri runtime, plugins, and threaded playback.
//!
//! ## Architecture
//! The library uses a `Mutex`-protected `Option<JukeboxEngine>` to manage the playback engine safely
//! across threads. Time-critical operations (like audio callbacks) run in a separate thread spawned
//! by the engine, but triggered via Tauri commands.

use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, State}; // Use Emitter trait for emitting events

pub mod analysis;
pub mod beat_tracker;
pub mod audio_backend;
pub mod workflow;
pub mod audio;
pub mod playback_engine;

use workflow::Remixatron;
use playback_engine::{JukeboxEngine, Beat};

/// Global Application State.
///
/// Wraps the `JukeboxEngine` in a `Mutex` to allow safe concurrent access from
/// multiple Tauri commands (e.g., UI updates vs. Playback control).
///
/// * `engine`: An `Arc<Mutex<Option<...>>>` allows us to share ownership of the lock
///   across threads (specifically, sharing it with the playback thread).
#[derive(Clone)]
struct AppState {
    engine: Arc<Mutex<Option<JukeboxEngine>>>,
}

/// The payload returned to the Frontend after a successful analysis.
///
/// This contains all the structural data needed to visualize the "Infinite Walk".
#[derive(serde::Serialize, Clone)]
struct StructurePayload {
    /// List of computed segments (Verse, Chorus, etc.)
    segments: Vec<workflow::Segment>,
    /// List of all beats with their connectivity info (Jump Candidates)
    beats: Vec<Beat>,
    /// The computed novelty curve (for debugging visualization)
    novelty_curve: Vec<f32>,
    /// Indices of detected structural peaks/boundaries
    peaks: Vec<usize>,
}

/// Event payload emitted on every playback tick.
///
/// Used to update the specific "cursor" position on the frontend visualization.
#[derive(serde::Serialize, Clone)]
struct PlaybackTick {
    /// The global index of the beat currently playing.
    beat_index: usize,
    /// The index of the segment this beat belongs to.
    segment_index: usize,
}

/// A simple greeting command for testing the Tauri connection.
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

/// Analyzes an audio file and initializes the Jukebox Engine.
///
/// This is a heavy operation that runs the entire ML pipeline:
/// 1.  **Loading**: Decodes MP3/WAV audio.
/// 2.  **Beat Tracking**: Runs ONNX model to find beats.
/// 3.  **Segmentation**: Performs Spectral Clustering to find structure.
/// 4.  **Graph Building**: Computes similarity and jump candidates.
///
/// # Arguments
/// * `path` - The absolute file path to the audio track.
/// * `state` - The tauri managed state object to store the resulting engine.
///
/// # Returns
/// * `Result<StructurePayload, String>` - The structural data for the UI, or an error description.
#[tauri::command]
async fn analyze_track(
    _app: AppHandle,
    path: String,
    state: State<'_, AppState>
) -> Result<StructurePayload, String> {
    println!("Analyzing: {}", path);
    // Hardcoded model paths for MVP - Ensure these exist in your bundle or are absolute
    // TODO: Move these to a configuration file or embedded assets for production.
    let mel_path = "/home/rensin/Projects/Remixatron/rust-app/MelSpectrogram_Ultimate.onnx";
    let beat_path = "/home/rensin/Projects/Remixatron/rust-app/BeatThis_small0.onnx";

    // 1. Run Analysis Pipeline
    // This delegates to the `workflow` module which orchestrates the complex logic.
    let engine_workflow = Remixatron::new(mel_path, beat_path);
    let analysis_result = engine_workflow.analyze(&path)
        .map_err(|e| format!("Analysis Failed: {}", e))?;

    // 2. Initialize Jukebox Engine
    // We create the engine with the analyzed beats. It is not yet playing.
    let mut jukebox = JukeboxEngine::new(analysis_result.beat_structs.clone(), analysis_result.k_optimal);

    // 3. Compute Jump Candidates
    // Note: This logic is now handled upstream in the Analysis Pipeline (workflow.rs).
    // The beats passed into the engine already contain their jump candidates.

    // 4. Load Audio for Playback
    // Decode the audio into memory so the engine can play it instantly.
    println!("Loading Audio into Engine...");
    jukebox.load_track(&path)
        .map_err(|e| format!("Audio Load Failed: {}", e))?;

    // 5. Store Engine in State
    // We clone the beats first for the return value because we're locking and modifying the state below.
    let beats_with_jumps = jukebox.beats.clone();

    // Acquire the lock on the global state and replace the engine.
    let mut engine_guard = state.engine.lock().map_err(|_| "Failed to lock state".to_string())?;
    *engine_guard = Some(jukebox);

    // 6. Return Structure to Frontend
    Ok(StructurePayload {
        segments: analysis_result.segments,
        beats: beats_with_jumps,
        novelty_curve: analysis_result.novelty_curve,
        peaks: analysis_result.peaks,
    })
}

/// Starts the infinite playback loop.
///
/// Spawns a dedicated background thread to handle the playback without blocking the Tauri
/// main thread or the UI.
///
/// # Callbacks
/// The playback engine invokes the closure provided here every time a beat occurs,
/// allowing us to emit `playback_tick` events back to the frontend.
#[tauri::command]
async fn play_track(
    app: AppHandle,
    state: State<'_, AppState>
) -> Result<(), String> {
    println!("Starting Playback...");

    let app_handle = app.clone();

    // Since `state` is a Tauri managed wrapper, we access the inner `AppState`.
    // Because `AppState` derives Clone (and wraps Arcs), this is a cheap operation
    // that creates a new reference to the same shared memory.
    let state_clone = state.inner().clone();

    // Spawn a blocking thread for the heavy audio loop.
    tauri::async_runtime::spawn_blocking(move || {
        // Lock the engine.
        // NOTE: This lock is held for the entire duration of playback (until the song ends or errors).
        // This is acceptable for the MVP as "Playing" is a blocking state, but it prevents
        // other commands from acquiring the engine lock concurrently.
        if let Ok(mut guard) = state_clone.engine.lock() {
            if let Some(engine) = guard.as_mut() {
                // Play with callback
                let _ = engine.play_with_callback(1000, move |beat_idx, segment_idx| {
                    // Emit event to Frontend
                    // We ignore errors here (e.g., if app is closing).
                    let _ = app_handle.emit("playback_tick", PlaybackTick {
                        beat_index: beat_idx,
                        segment_index: segment_idx
                    });
                });
            }
        }
    });

    Ok(())
}

/// The main entry point helper for the Tauri Library.
///
/// Configures the environment, registers plugins, manages state, and launches the app.
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Explicitly point to the bundled ONNX Runtime library to ensure it loads correctly on Linux
    std::env::set_var("ORT_DYLIB_PATH", "/home/rensin/Projects/Remixatron/rust-app/libonnxruntime.so");

    tauri::Builder::default()
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(AppState {
            engine: Arc::new(Mutex::new(None))
        })
        .invoke_handler(tauri::generate_handler![greet, analyze_track, play_track])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
