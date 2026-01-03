// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

pub mod analysis;
pub mod beat_tracker;
pub mod audio_backend;
pub mod workflow;
pub mod audio;
pub mod playback_engine; 

use workflow::{Remixatron, AnalysisResult};

#[tauri::command]
async fn analyze_file(path: String) -> Result<AnalysisResult, String> {
    println!("Analyzing: {}", path);
    // Hardcoded model paths for MVP
    let mel_path = "/home/rensin/Projects/Remixatron/rust-app/MelSpectrogram_Ultimate.onnx";
    let beat_path = "/home/rensin/Projects/Remixatron/rust-app/BeatThis_small0.onnx";
    
    let engine = Remixatron::new(mel_path, beat_path);
    match engine.analyze(&path) {
        Ok(res) => Ok(res),
        Err(e) => Err(format!("Analysis Failed: {}", e))
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![greet, analyze_file])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
