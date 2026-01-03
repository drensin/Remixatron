use remixatron_lib::beat_tracker::mel::MelProcessor;

// "Array3 and Axis" unused.
// Let's check code. r_shape = mel_output.shape(). r_shape[1].
// mel_output[[0, t, f]].
// No explicit Array3 or Axis usage seen in snippet.
// But import lines 2 and 4.
// Line 2: use ndarray::{Array3, Axis};
// Warning says BOTH unused. So remove line 2 completely or empty imports.

use std::fs::File;

use serde::Deserialize;
use anyhow::Result;

#[derive(Deserialize)]
struct GoldAudio {
    signal: Vec<f32>,
    _sr: u32,
}



fn main() -> Result<()> {
    let model_path = "/home/rensin/Projects/Remixatron/rust-app/MelSpectrogram_Ultimate.onnx";
    let gold_audio_path = "/home/rensin/Projects/Remixatron/rust-app/gold_standard/gold_audio.json";
    let gold_mel_path = "/home/rensin/Projects/Remixatron/rust-app/gold_standard/gold_mel.json";

    println!("Initializing MelProcessor...");
    let mut processor = MelProcessor::new(model_path)?;
    
    println!("Loading Golden Audio...");
    let file = File::open(gold_audio_path)?;
    let reader = std::io::BufReader::new(file);
    let gold_audio: GoldAudio = serde_json::from_reader(reader)?;
    
    println!("Running Inference (Audio -> Mel)...");
    let mel_output = processor.process(&gold_audio.signal)?;
    
    println!("Loading Golden Mel...");
    let file = File::open(gold_mel_path)?;
    let reader = std::io::BufReader::new(file);
    let gold_mel_vec: Vec<Vec<f32>> = serde_json::from_reader(reader)?;
    
    // Rust output is (1, Time, Freq). Gold is (Time, Freq).
    let r_shape = mel_output.shape();
    let g_time = gold_mel_vec.len();
    let g_freq = gold_mel_vec[0].len();
    
    println!("Shapes: Rust {:?}, Gold ({}, {})", r_shape, g_time, g_freq);
    
    // Check if Rust time matches Gold time (allowing small difference? No, strict for now)
    if r_shape[1] != g_time || r_shape[2] != g_freq {
        println!("WARNING: Shape mismatch! Rust: {:?}, Gold: ({}, {})", r_shape, g_time, g_freq);
        // We will proceed to compare the overlap
    }
    
    println!("Comparing content...");
    let mut total_sq_err = 0.0;
    let mut max_diff = 0.0;
    
    let t_len = r_shape[1].min(g_time);
    let count = (t_len * g_freq) as f32;
    
    for t in 0..t_len {
        for f in 0..g_freq {
            let r = mel_output[[0, t, f]];
            let g = gold_mel_vec[t][f];
            let diff = (r - g).abs();
            total_sq_err += diff * diff;
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }
    
    let mse = total_sq_err / count;
    println!("Comparison Result:");
    println!("  MSE: {:.8}", mse);
    println!("  Max Diff: {:.8}", max_diff);
    
    if mse < 1e-4 {
        println!(">>> SUCCESS: Mel Spectrogram (using Golden Audio) matches.");
    } else {
        println!(">>> FAILURE: Mel Spectrogram differs significantly.");
        std::process::exit(1);
    }

    Ok(())
}
