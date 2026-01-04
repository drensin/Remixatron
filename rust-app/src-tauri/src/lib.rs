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

#[derive(Clone)]
struct AppState {
    engine: Arc<Mutex<Option<JukeboxEngine>>>,
}

#[derive(serde::Serialize, Clone)]

struct StructurePayload {
    segments: Vec<workflow::Segment>,
    beats: Vec<Beat>,
    novelty_curve: Vec<f32>,
    peaks: Vec<usize>,
}



#[derive(serde::Serialize, Clone)]
struct PlaybackTick {
    beat_index: usize,
    segment_index: usize,
}

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[tauri::command]
async fn analyze_track(
    _app: AppHandle, 
    path: String, 
    state: State<'_, AppState>
) -> Result<StructurePayload, String> {
    println!("Analyzing: {}", path);
    // Hardcoded model paths for MVP - Ensure these exist in your bundle or are absolute
    let mel_path = "/home/rensin/Projects/Remixatron/rust-app/MelSpectrogram_Ultimate.onnx";
    let beat_path = "/home/rensin/Projects/Remixatron/rust-app/BeatThis_small0.onnx";
    
    // 1. Run Analysis Pipeline
    let engine_workflow = Remixatron::new(mel_path, beat_path);
    let analysis_result = engine_workflow.analyze(&path)
        .map_err(|e| format!("Analysis Failed: {}", e))?;
        
    // 2. Initialize Jukebox Engine
    let mut jukebox = JukeboxEngine::new(analysis_result.beat_structs.clone(), analysis_result.k_optimal);
    
    // 3. Compute Jump Candidates (The Graph)
    println!("Identifying Jump Candidates...");
    jukebox.identify_jump_candidates();
    
    // 4. Load Audio for Playback
    println!("Loading Audio into Engine...");
    jukebox.load_track(&path)
        .map_err(|e| format!("Audio Load Failed: {}", e))?;

    // 5. Store Engine in State
    // We clone the beats first for the return value because we're moving the engine into the mutex
    let beats_with_jumps = jukebox.beats.clone(); 
    
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

#[tauri::command]
async fn play_track(
    app: AppHandle, 
    state: State<'_, AppState>
) -> Result<(), String> {
    println!("Starting Playback...");
    
    let app_handle = app.clone();
    
    // Since AppState wraps Arc<Mutex<...>>, and State implements Deref<Target=AppState> (sort of, or we can clone the inner content)
    // Actually, AppState is just a struct. State<T> is a wrapper around T.
    // We can get a reference to AppState via `state.inner()`.
    // Since we made `AppState` implement Clone (containing Arc), we can clone it!
    let state_clone = state.inner().clone();
    
    tauri::async_runtime::spawn_blocking(move || {
        // Lock the engine
        if let Ok(mut guard) = state_clone.engine.lock() {
            if let Some(engine) = guard.as_mut() {
                // Play with callback
                let _ = engine.play_with_callback(1000, move |beat_idx, segment_idx| { 
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Explicitly point to the bundled ONNX Runtime library
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
