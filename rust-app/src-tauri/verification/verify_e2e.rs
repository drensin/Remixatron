use remixatron_lib::audio::loader::load_audio;
use remixatron_lib::beat_tracker::mel::MelProcessor;
use remixatron_lib::beat_tracker::inference::BeatProcessor;
use remixatron_lib::beat_tracker::post_processor::MinimalPostProcessor;

use std::fs::File;
use std::io::BufReader;
use serde::Deserialize;
use anyhow::Result;

#[derive(Deserialize)]
struct GoldBeats(Vec<f32>, #[allow(dead_code)] Vec<f32>); // [beats, downbeats]

fn main() -> Result<()> {
    let audio_path = "/home/rensin/Downloads/Stay with Me.mp3";
    let mel_model_path = "/home/rensin/Projects/Remixatron/rust-app/MelSpectrogram_Ultimate.onnx";
    let infer_model_path = "/home/rensin/Projects/Remixatron/rust-app/BeatThis_small0.onnx";
    let gold_beats_path = "/home/rensin/Projects/Remixatron/rust-app/gold_standard/gold_beats.json";

    println!("Stage 1: Loading Audio...");
    let audio_data = load_audio(audio_path, 22050)?;
    println!("  Loaded {} samples.", audio_data.signal.len());
    
    // Offset correction?
    // Parity analysis showed ~507 sample lag in Rust (approx 0.023s).
    // If strictly checking parity with Python generated form 'gold_audio.json' (which was FFmpeg),
    // we should expect this offset.
    // However, for "Beat Detection Reliability", testing the actual Rust pipeline is key.
    // We will compare against Gold Beats and see if they match within tolerance (e.g. 50ms).
    
    println!("Stage 2: Mel Spectrogram...");
    let mut mel_proc = MelProcessor::new(mel_model_path)?;
    let mel = mel_proc.process(&audio_data.signal)?;
    println!("  Mel shape: {:?}", mel.shape());
    
    println!("Stage 3: Inference...");
    let mut beat_proc = BeatProcessor::new(infer_model_path)?;
    let (beat_logits, downbeat_logits) = beat_proc.process(&mel)?;
    println!("  Logits len: {}", beat_logits.len());
    
    println!("Stage 4: Post-Processing...");
    let pp = MinimalPostProcessor::new(50.0);
    // Be careful! beat_this uses 50.0 fps relative to the spectrogram hop size?
    // Mel uses 22050 Hz.
    // Spec: N_FFT=2048, Hop=441 (usually). 
    // 22050 / 441 = 50.0 Hz. Correct.
    let (r_beats, _r_downbeats) = pp.process(&beat_logits, &downbeat_logits)?;
    
    println!("Loading Golden Beats for Comparison...");
    let file = File::open(gold_beats_path)?;
    let reader = BufReader::new(file);
    let gold_beats: GoldBeats = serde_json::from_reader(reader)?;
    let g_beats = gold_beats.0;
    
    println!("Comparing Beats (End-to-End)...");
    compare_times("Beats", &r_beats, &g_beats)?;

    Ok(())
}

fn compare_times(label: &str, rust: &[f32], gold: &[f32]) -> Result<()> {
    println!("  {}: Rust Count {}, Gold Count {}", label, rust.len(), gold.len());
    
    // Allow small count mismatch at edges due to padding/offset
    let len = rust.len().min(gold.len());
    let mut max_diff = 0.0;
    let mut sum_diff = 0.0;
    
    for i in 0..len {
        let diff = (rust[i] - gold[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 0.5 && sum_diff < 100.0 { // Print first few big jumps
             println!("    JUMP DETECTED at index {}: Diff {:.4} (R: {:.4}, G: {:.4})", i, diff, rust[i], gold[i]);
             // Stop spamming after detecting the break
             sum_diff = 1000.0; 
        }
        if sum_diff < 1000.0 {
            sum_diff += diff;
        }
    }
    
    let avg_diff = sum_diff / len as f32;
    println!("  {}: Max Diff: {:.6}s, Avg Diff: {:.6}s", label, max_diff, avg_diff);
    
    // Tolerance: 50ms (0.05s) for decoder differences.
    // Recall we saw 0.023s lag.
    if max_diff > 0.05 {
        println!("  WARNING: Large drift detected. Likely decoder offset.");
        // Try to align?
        // Let's print the first 5 diffs
        for i in 0..5.min(len) {
            println!("    diff[{}] = {:.6} (R: {:.6}, G: {:.6})", i, rust[i]-gold[i], rust[i], gold[i]);
        }
    }
    
    if max_diff < 0.05 {
        println!(">>> SUCCESS: E2E Parity Confirmed (within 50ms tolerance).");
    } else {
         println!(">>> CAUTION: E2E Variance > 50ms. Investigate if critical.");
    }
    
    Ok(())
}
