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
use tauri::{AppHandle, Emitter, State, Manager};
use std::sync::mpsc::{self, Sender};
use playback_engine::PlaybackCommand;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::probe::Hint;
use symphonia::core::meta::{MetadataOptions, Limit};
use symphonia::core::formats::FormatOptions;
use base64::prelude::*;
use std::fs::File;

pub mod analysis;
pub mod beat_tracker;
pub mod audio_backend;
pub mod workflow;
pub mod audio;
pub mod playback_engine;
pub mod downloader; 

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
    /// Channel Sender to control the active playback thread.
    /// Kept separate from `engine` lock to prevent deadlocks.
    playback_tx: Arc<Mutex<Option<Sender<PlaybackCommand>>>>,
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
    /// Metadata: Track Title
    title: String,
    /// Metadata: Track Artist
    artist: String,
    /// Metadata: Album Art (Base64 Data URI)
    album_art_base64: Option<String>,
}

/// Early metadata payload emitted before analysis completes.
#[derive(serde::Serialize, Clone)]
struct MetadataPayload {
    title: String,
    artist: String,
    album_art_base64: Option<String>,
}

/// Payload for explicit progress updates during analysis.
#[derive(serde::Serialize, Clone)]
struct ProgressPayload {
    message: String,
    progress: f32,
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
    /// Total length of the current sequence (for countdown).
    seq_len: usize,
    /// Current position in said sequence.
    seq_pos: usize,
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
    app: AppHandle,
    path: String,
    state: State<'_, AppState>
) -> Result<StructurePayload, String> {
    println!("Analyzing: {}", path);

    // 0. Extract Metadata (Early)
    let mut title = String::from("Unknown Title");
    let mut artist = String::from("Unknown Artist");
    let mut album_art_base64 = None;

    // Probe for Metadata (Title, Artist, Art)
    // We do this BEFORE the heavy analysis so the UI updates instantly.
    if let Ok(src) = File::open(&path) {
        let mss = MediaSourceStream::new(Box::new(src), Default::default());
        let mut hint = Hint::new();
        if let Some(ext) = std::path::Path::new(&path).extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }
        
        let meta_opts = MetadataOptions { 
            limit_visual_bytes: Limit::Maximum(10 * 1024 * 1024), 
            ..Default::default() 
        };
        let fmt_opts = FormatOptions::default();

        if let Ok(mut probed) = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts) {
            // Helper to extract data from a revision
            let mut extract_from_rev = |rev: &symphonia::core::meta::MetadataRevision| {
                 if let Some(tag) = rev.visuals().first() {
                     let b64 = BASE64_STANDARD.encode(&tag.data);
                     album_art_base64 = Some(format!("data:{};base64,{}", tag.media_type, b64));
                 }
                 if title == "Unknown Title" {
                     if let Some(tags) = rev.tags().iter().find(|t| t.std_key == Some(symphonia::core::meta::StandardTagKey::TrackTitle)) {
                         title = tags.value.to_string();
                     }
                 }
                 if artist == "Unknown Artist" {
                     if let Some(tags) = rev.tags().iter().find(|t| t.std_key == Some(symphonia::core::meta::StandardTagKey::Artist)) {
                         artist = tags.value.to_string();
                     }
                 }
            };

            // 1. Check Format Metadata (e.g. ID3v2)
            if let Some(rev) = probed.format.metadata().current() {
                 extract_from_rev(rev);
            }
            
            // 2. Check Container Metadata (e.g. MP4 Atoms / MOOV)
            if let Some(rev) = probed.metadata.get().as_ref().and_then(|m| m.current()) {
                 extract_from_rev(rev);
            }
        }
    }

    // Fallback: If title is still unknown, use filename
    if title == "Unknown Title" {
        let path_obj = std::path::Path::new(&path);
        if let Some(stem) = path_obj.file_stem() {
            title = stem.to_string_lossy().to_string();
        }
    }

    // EMIT EARLY EVENT
    let _ = app.emit("metadata_ready", MetadataPayload {
        title: title.clone(),
        artist: artist.clone(),
        album_art_base64: album_art_base64.clone(),
    });

    // Hardcoded model paths for MVP - Ensure these exist in your bundle or are absolute
    // TODO: Move these to a configuration file or embedded assets for production.
    let mel_path = "/home/rensin/Projects/Remixatron/rust-app/MelSpectrogram_Ultimate.onnx";
    let beat_path = "/home/rensin/Projects/Remixatron/rust-app/BeatThis_small0.onnx";

    // 1. Run Analysis Pipeline
    // This delegates to the `workflow` module which orchestrates the complex logic.
    let engine_workflow = Remixatron::new(mel_path, beat_path);
    
    // Create a callback that emits events to the frontend
    let app_handle = app.clone();
    let analysis_result = engine_workflow.analyze(&path, |message, progress| {
        let _ = app_handle.emit("analysis_progress", ProgressPayload {
            message: message.to_string(),
            progress,
        });
    }).map_err(|e| format!("Analysis Failed: {}", e))?;

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

    // 6. Metadata already extracted above!
    // We just return it now.
    
    // 7. Return Structure to Frontend
    Ok(StructurePayload {
        segments: analysis_result.segments,
        beats: beats_with_jumps,
        novelty_curve: analysis_result.novelty_curve,
        peaks: analysis_result.peaks,
        title,
        artist,
        album_art_base64,
    })
}

#[tauri::command]
async fn import_url(app: AppHandle, url: String) -> Result<downloader::VideoMetadata, String> {
    downloader::download_url(app, url).await.map_err(|e| e.to_string())
}

/// Signals the running playback thread to stop safely.
///
/// This resolves the "Mutex Deadlock" by allowing the playback thread to exit its loop
/// and release the `engine` lock, so that a new analysis can acquire it.
#[tauri::command]
async fn stop_playback(state: State<'_, AppState>) -> Result<(), String> {
    println!("Stopping Playback...");
    // 1. Acquire Sender Lock (This is fast, never held long)
    let tx_guard = state.playback_tx.lock().map_err(|_| "Failed to lock playback_tx".to_string())?;
    
    // 2. Send Stop Command
    if let Some(tx) = tx_guard.as_ref() {
        // We ignore send errors (e.g., if thread is already dead)
        let _ = tx.send(PlaybackCommand::Stop);
    }
    
    // We do NOT wait for join here. The Frontend handles the flow.
    Ok(())
}

/// Toggles the pause state of the playback engine.
#[tauri::command]
async fn set_paused(state: State<'_, AppState>, paused: bool) -> Result<(), String> {
    // 1. Acquire Sender Lock
    let tx_guard = state.playback_tx.lock().map_err(|_| "Failed to lock playback_tx".to_string())?;
    
    // 2. Send Command
    if let Some(tx) = tx_guard.as_ref() {
        let cmd = if paused { PlaybackCommand::Pause } else { PlaybackCommand::Resume };
        // Ignore errors (if thread dead, nothing to pause)
        let _ = tx.send(cmd);
    }
    
    Ok(())
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
    
    // 1. Create Control Channel
    let (tx, rx) = mpsc::channel();
    
    // 2. Store Sender in State (overwrite previous if any)
    {
        let mut tx_guard = state.playback_tx.lock().map_err(|_| "Failed to lock playback_tx".to_string())?;
        *tx_guard = Some(tx);
    } // Drop lock immediately

    // Spawn a blocking thread for the heavy audio loop.
    tauri::async_runtime::spawn_blocking(move || {
        // Lock the engine.
        // NOTE: This lock is held for the entire duration of playback (until the song ends or errors).
        if let Ok(mut guard) = state_clone.engine.lock() {
            if let Some(engine) = guard.as_mut() {
                // Play with callback and RECEIVER
                let _ = engine.play_with_callback(rx, move |instruction, segment_idx| {
                    // Emit event to Frontend
                    // We ignore errors here (e.g., if app is closing).
                    let beat_idx = instruction.beat_id;
                    
                    let _ = app_handle.emit("playback_tick", PlaybackTick {
                        beat_index: beat_idx,
                        segment_index: segment_idx,
                        seq_len: instruction.seq_len,
                        seq_pos: instruction.seq_pos,
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
        .setup(|app| {
            // Manage state
            app.manage(AppState {
                engine: Arc::new(Mutex::new(None)),
                playback_tx: Arc::new(Mutex::new(None)),
            });

            // Initialize Downloader in background
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                match downloader::init_downloader(handle.clone()).await {
                    Ok(path) => println!("Downloader Initialized at {:?}", path),
                    Err(e) => eprintln!("Downloader Init Failed: {}", e),
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            greet, 
            analyze_track, 
            play_track, 
            stop_playback, 
            import_url,
            set_paused
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
