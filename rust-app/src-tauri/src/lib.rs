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

pub mod favorites;
pub mod broadcasting;

use workflow::Remixatron;
use playback_engine::{JukeboxEngine, Beat};
use crossbeam_channel::Sender as CrossbeamSender;
use tokio::sync::{broadcast, watch};
use broadcasting::server::{VizInitData, VizUpdateData};

use std::sync::atomic::AtomicU32;

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
    playback_tx: Arc<Mutex<Option<Sender<PlaybackCommand>>>>,
    
    /// Broadcasting audio channel.
    broadcast_audio_tx: CrossbeamSender<Vec<f32>>,
    
    /// Shared sample rate for the transcoder.
    broadcast_sample_rate: Arc<AtomicU32>,
    
    /// Watch sender for static visualization data (beats, segments, waveform).
    viz_init_tx: watch::Sender<VizInitData>,
    
    /// Broadcast sender for dynamic viz updates (current beat, sequence state).
    viz_update_tx: broadcast::Sender<VizUpdateData>,
    
    /// Last download metadata (from YouTube). Used to provide metadata to receiver
    /// when the downloaded file doesn't have embedded metadata.
    last_download_metadata: Arc<Mutex<Option<downloader::VideoMetadata>>>,
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

    // --- MODEL PATH RESOLUTION ---
    // In Production: They are bundled in `models/` resource.
    // In Development: They are in `src-tauri/models/` (relative to CWD).
    
    // We resolve the resources directory using Tauri's path resolver.
    // Since we are in a Command, we can access the app handle directly.
    
    let resource_dir = app.path().resource_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    
    let mel_resource = resource_dir.join("models").join("MelSpectrogram_Ultimate.onnx");
    let beat_resource = resource_dir.join("models").join("BeatThis_small0.onnx");

    let (mel_path_buf, beat_path_buf) = if mel_resource.exists() && beat_resource.exists() {
         (mel_resource, beat_resource)
    } else {
         // Fallback for Dev Environment (cargo run from root)
         // Assuming CWD is project root
         let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
         let dev_mel = cwd.join("src-tauri").join("models").join("MelSpectrogram_Ultimate.onnx");
         let dev_beat = cwd.join("src-tauri").join("models").join("BeatThis_small0.onnx");
         
         if dev_mel.exists() {
             (dev_mel, dev_beat)
         } else {
             // Second Fallback: Maybe CWD is src-tauri?
             let dev_mel_2 = cwd.join("models").join("MelSpectrogram_Ultimate.onnx");
              let dev_beat_2 = cwd.join("models").join("BeatThis_small0.onnx");
             (dev_mel_2, dev_beat_2)
         }
    };
    
    let mel_path = mel_path_buf.to_string_lossy().to_string();
    let beat_path = beat_path_buf.to_string_lossy().to_string();
    
    println!("Using Models:\nMel: {}\nBeat: {}", mel_path, beat_path);

    // 1. Run Analysis Pipeline
    // This delegates to the `workflow` module which orchestrates the complex logic.
    let engine_workflow = Remixatron::new(&mel_path, &beat_path);
    
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
    
    // Enable Broadcasting (Always on for now)
    jukebox.enable_broadcasting(
        state.broadcast_audio_tx.clone(),
        state.broadcast_sample_rate.clone(),
    );

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
    let waveform_envelope = jukebox.waveform_envelope.clone();
    let mut engine_guard = state.engine.lock().map_err(|_| "Failed to lock state".to_string())?;
    *engine_guard = Some(jukebox);

    // 6. Broadcast viz init data to WebSocket clients.
    // Prefer YouTube metadata if we just downloaded, otherwise use file metadata.
    let (viz_title, viz_artist, viz_thumbnail) = {
        let download_meta = state.last_download_metadata.lock().ok().and_then(|g| g.clone());
        if let Some(meta) = download_meta {
            // Use YouTube metadata and clear it (consumed)
            if let Ok(mut guard) = state.last_download_metadata.lock() {
                *guard = None;
            }
            (
                meta.title,
                meta.artist,
                meta.thumbnail_url.unwrap_or_default(),
            )
        } else {
            // Use file-extracted metadata
            (
                title.clone(),
                artist.clone(),
                album_art_base64.clone().unwrap_or_default(),
            )
        }
    };
    
    // Converts beats and segments to JSON values for the receiver.
    let viz_init = VizInitData {
        beats: beats_with_jumps.iter().map(|b| serde_json::json!({
            "id": b.id,
            "start": b.start,
            "duration": b.duration,
            "segment": b.segment,
            "jump_candidates": b.jump_candidates,
        })).collect(),
        segments: analysis_result.segments.iter().map(|s| serde_json::json!({
            "label": s.label,
            "start_time": s.start_time,
            "end_time": s.end_time,
        })).collect(),
        waveform: waveform_envelope,
        title: viz_title,
        artist: viz_artist,
        thumbnail: viz_thumbnail,
    };
    let _ = state.viz_init_tx.send(viz_init);
    println!("[Broadcast] Sent viz init data to WebSocket clients.");

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
async fn import_url(
    app: AppHandle,
    url: String,
    state: State<'_, AppState>
) -> Result<downloader::VideoMetadata, String> {
    let metadata = downloader::download_url(app, url)
        .await
        .map_err(|e| e.to_string())?;
    
    // Store metadata for use by analyze_track when building VizInitData
    if let Ok(mut guard) = state.last_download_metadata.lock() {
        *guard = Some(metadata.clone());
    }
    
    Ok(metadata)
}

/// Checks if required external dependencies (yt-dlp and ffmpeg) are available.
///
/// This command should be called at startup. If any dependencies are missing,
/// the frontend should display an error dialog and exit the application.
///
/// # Returns
/// `DependencyStatus` with version strings for available tools, None for missing.
#[tauri::command]
fn check_dependencies() -> downloader::DependencyStatus {
    downloader::check_dependencies()
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
    
    // 3. Notify WebSocket clients that playback has stopped.
    let _ = state.viz_update_tx.send(VizUpdateData {
        stopped: true,
        ..Default::default()
    });
    
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

    // 3. Notify WebSocket clients about pause state.
    let _ = state.viz_update_tx.send(VizUpdateData {
        paused,
        ..Default::default()
    });
    
    Ok(())
}

/// Returns the pre-computed waveform amplitude envelope for visualization.
///
/// The envelope contains ~720 normalized values (0.0-1.0) representing the
/// RMS amplitude at each position around the 360° ring (2 samples per degree).
/// This is computed once during `load_track` and cached for the session.
///
/// # Returns
/// A `Vec<f32>` of amplitude values, or an error if no track is loaded.
#[tauri::command]
async fn get_waveform_envelope(state: State<'_, AppState>) -> Result<Vec<f32>, String> {
    let engine_guard = state.engine.lock()
        .map_err(|_| "Failed to lock engine".to_string())?;
    
    let engine = engine_guard.as_ref()
        .ok_or_else(|| "No track loaded".to_string())?;
    
    Ok(engine.waveform_envelope.clone())
}

// =============================================================================
// Favorites Commands
// =============================================================================

/// Returns all saved favorites, sorted by artist (ascending), then by title (ascending).
///
/// This command reads from the persistent JSON file in the app data directory.
/// If no favorites exist or the file is missing/corrupted, an empty list is returned.
///
/// # Returns
/// A vector of `Favorite` objects.
#[tauri::command]
async fn list_favorites(app: AppHandle) -> Vec<favorites::Favorite> {
    favorites::load_favorites(&app)
}

/// Adds a new favorite or updates an existing one (upsert behavior).
///
/// If a favorite with the same `source` already exists, its metadata is updated.
/// Otherwise, a new favorite is created and persisted.
///
/// # Arguments
/// * `source` - The file path or URL of the track.
/// * `artist` - The artist name for display.
/// * `title` - The track title for display.
///
/// # Returns
/// The `Favorite` object that was added or updated.
#[tauri::command]
async fn add_favorite(
    app: AppHandle,
    source: String,
    artist: String,
    title: String,
) -> Result<favorites::Favorite, String> {
    favorites::add_favorite(&app, source, artist, title)
}

/// Removes a favorite by its source identifier.
///
/// If no favorite with the given source exists, this is a silent no-op.
///
/// # Arguments
/// * `source` - The file path or URL of the track to remove.
#[tauri::command]
async fn remove_favorite(app: AppHandle, source: String) -> Result<(), String> {
    favorites::remove_favorite(&app, &source)
}

/// Checks whether a given source is already saved as a favorite.
///
/// # Arguments
/// * `source` - The file path or URL to check.
///
/// # Returns
/// `true` if the source exists in the favorites list, `false` otherwise.
#[tauri::command]
async fn check_is_favorite(app: AppHandle, source: String) -> bool {
    favorites::is_favorite(&app, &source)
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
                // Clone beats for duration lookup in the closure.
                let beats_for_sync = engine.beats.clone();

                // Track cumulative audio duration for sync.
                // This directly corresponds to audio.currentTime on the client.
                let mut cumulative_audio_time: f32 = 0.0;

                // Play with callback and RECEIVER
                let viz_tx = state_clone.viz_update_tx.clone();
                let _ = engine.play_with_callback(rx, move |instruction, segment_idx| {
                    let beat_idx = instruction.beat_id;

                    // Emit event to Frontend
                    let _ = app_handle.emit("playback_tick", PlaybackTick {
                        beat_index: beat_idx,
                        segment_index: segment_idx,
                        seq_len: instruction.seq_len,
                        seq_pos: instruction.seq_pos,
                    });

                    // Broadcast to WebSocket clients
                    // stream_time is the position at START of this beat.
                    let _ = viz_tx.send(VizUpdateData {
                        active_beat: beat_idx,
                        active_seg: segment_idx,
                        seq_pos: instruction.seq_pos,
                        seq_len: instruction.seq_len,
                        stream_time: cumulative_audio_time,
                        stopped: false,
                        paused: false,
                    });

                    // Update cumulative time AFTER sending (for next beat)
                    let beat_duration = if beat_idx < beats_for_sync.len() {
                        beats_for_sync[beat_idx].duration
                    } else {
                        0.5 // Fallback: typical beat duration
                    };
                    cumulative_audio_time += beat_duration as f32;
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

    tauri::Builder::default()
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            // --- ONNX Runtime Library Resolution ---
            #[cfg(desktop)]
            {
                use tauri::Manager;
                let resource_path = app.path().resource_dir().expect("failed to get resource dir");
                
                // Construct path based on platform
                let lib_path = if cfg!(target_os = "windows") {
                    resource_path.join("libs\\win\\onnxruntime.dll")
                } else if cfg!(target_os = "macos") {
                    resource_path.join("libs/macos/libonnxruntime.dylib")
                } else {
                    resource_path.join("libs/linux/libonnxruntime.so")
                };

                // In Development, resources aren't bundled the same way.
                // We fallback to a relative path check or allow the hardcoded path.
                if !lib_path.exists() {
                     // Dev fallback: Try local relative path
                     // This assumes running from `src-tauri` root usually?
                     // actually, cargo run runs from root.
                     println!("Bundled ORT not found at {:?}. Assuming Dev Environment.", lib_path);
                     
                     // Helper for dev environment path
                     let dev_path = if cfg!(target_os = "windows") {
                        "libs/win/onnxruntime.dll"
                     } else if cfg!(target_os = "macos") {
                        "libs/macos/libonnxruntime.dylib"
                     } else {
                        "libs/linux/libonnxruntime.so"
                     };
                     
                     // We just try to set it to absolute path of project root + dev_path
                     // Or just leave it if user set it externally.
                     if let Ok(cwd) = std::env::current_dir() {
                         let abs_dev_path = cwd.join("src-tauri").join(dev_path);
                         if abs_dev_path.exists() {
                             std::env::set_var("ORT_DYLIB_PATH", &abs_dev_path);
                             println!("Set ORT_DYLIB_PATH to Dev Path: {:?}", abs_dev_path);
                         }
                     }
                } else {
                    // Production Bundle
                    std::env::set_var("ORT_DYLIB_PATH", lib_path.clone());
                    println!("Set ORT_DYLIB_PATH to Bundled Resource: {:?}", lib_path);
                }
            }

            // =================================================================
            // BROADCASTING SERVICES INITIALIZATION
            // =================================================================
            // The broadcasting pipeline consists of three components that run
            // for the lifetime of the application:
            //
            // 1. Audio Tee Channel: A bounded crossbeam channel that receives
            //    raw PCM audio samples from the playback engine during track
            //    loading. This is the "Pre-Kira Tap" approach.
            //
            // 2. Transcoder: A dedicated OS thread that reads PCM samples,
            //    encodes them to MP3 using LAME, and publishes chunks to a
            //    tokio broadcast channel.
            //
            // 3. Web Server: An Axum HTTP server that streams MP3 audio to
            //    clients via chunked transfer encoding.
            //
            // The pipeline is always running but only transmits data when
            // a track is loaded/playing. This enables instant casting without
            // waiting for server initialization.
            // =================================================================

            // Channel for PCM samples (Playback Engine -> Transcoder).
            // REDUCED: 64 chunks (~1.5 seconds) to minimize stream latency.
            // Smaller buffer means viz and audio stay in sync.
            let (audio_tx, audio_rx) = crossbeam_channel::bounded::<Vec<f32>>(64);

            // Broadcast channel for MP3 bytes (Transcoder -> HTTP Clients).
            // REDUCED: 4 chunks to minimize latency; slow clients skip.
            let (mp3_tx, _) = broadcast::channel(4);

            // Sample rate handle: updated by engine when track loads.
            let sample_rate_handle = broadcasting::transcoder::create_sample_rate_handle();

            // Spawn the transcoder background thread with sample rate handle.
            broadcasting::transcoder::spawn_transcoder(
                audio_rx, 
                mp3_tx.clone(), 
                sample_rate_handle.clone()
            );

            // Visualization state channels.
            // - viz_init: watch channel for static data (beats, segments, waveform)
            // - viz_update: broadcast channel for dynamic playback state
            let (viz_init_tx, viz_init_rx) = watch::channel(VizInitData::default());
            let (viz_update_tx, _) = broadcast::channel(64);

            // Spawn the web server in Tauri's async runtime.
            let mp3_tx_clone = mp3_tx.clone();
            let viz_init_rx_clone = viz_init_rx.clone();
            let viz_update_tx_clone = viz_update_tx.clone();
            tauri::async_runtime::spawn(async move {
                broadcasting::server::start_server(
                    mp3_tx_clone,
                    viz_init_rx_clone,
                    viz_update_tx_clone,
                ).await;
            });

            app.manage(AppState {
                engine: Arc::new(Mutex::new(None)),
                playback_tx: Arc::new(Mutex::new(None)),
                broadcast_audio_tx: audio_tx,
                broadcast_sample_rate: sample_rate_handle,
                viz_init_tx,
                viz_update_tx,
                last_download_metadata: Arc::new(Mutex::new(None)),
            });

            // NOTE: No downloader init here - frontend checks dependencies on startup
            // and shows error dialog if yt-dlp/ffmpeg missing.

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            greet, 
            analyze_track, 
            play_track, 
            stop_playback, 
            import_url,
            check_dependencies,
            set_paused,
            get_waveform_envelope,
            list_favorites,
            add_favorite,
            remove_favorite,
            check_is_favorite])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
