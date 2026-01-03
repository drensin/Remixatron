use remixatron_lib::beat_tracker::inference::BeatProcessor;
use ndarray::Ix3;
use std::fs::File;

use serde::Deserialize;
use anyhow::Result;


#[derive(Deserialize)]
struct GoldLogits {
    beat: Vec<f32>,
    _downbeat: Vec<f32>,
}

fn main() -> Result<()> {
    let model_path = "/home/rensin/Projects/Remixatron/rust-app/BeatThis_small0.onnx";
    let gold_mel_path = "/home/rensin/Projects/Remixatron/rust-app/gold_standard/gold_mel.json";
    let gold_logits_path = "/home/rensin/Projects/Remixatron/rust-app/gold_standard/gold_logits.json";

    println!("Initializing BeatProcessor...");
    let mut processor = BeatProcessor::new(model_path)?;
    
    println!("Loading Golden Mel...");
    let file = File::open(gold_mel_path)?;
    let reader = std::io::BufReader::new(file);
    let gold_mel_vec: Vec<Vec<f32>> = serde_json::from_reader(reader)?;
    
    let time = gold_mel_vec.len();
    let freq = gold_mel_vec[0].len();
    
    // Convert to Array3 (Batch=1, Time, Freq)
    // Flatten data: row-major (Time, Freq)
    let mut flat_mel = Vec::with_capacity(time * freq);
    for row in &gold_mel_vec {
        flat_mel.extend_from_slice(row);
    }
    
    let mel_array = ndarray::Array::from_shape_vec((1, time, freq), flat_mel)?
        .into_dimensionality::<Ix3>()?;
    
    println!("Running Inference...");
    let (beat_out, _down_out) = processor.process(&mel_array)?;
    
    println!("Loading Golden Logits...");
    let file = File::open(gold_logits_path)?;
    let reader = std::io::BufReader::new(file);
    let gold_logits: GoldLogits = serde_json::from_reader(reader)?;
    
    println!("Sizes: Beat Rust {}, Gold {}", beat_out.len(), gold_logits.beat.len());
    
    if beat_out.len() != gold_logits.beat.len() {
        println!("WARNING: Length mismatch! Difference: {}", (beat_out.len() as i32 - gold_logits.beat.len() as i32));
    }
    
    // Compare
    let len = beat_out.len().min(gold_logits.beat.len());
    let mut total_sq_err = 0.0;
    let mut max_diff = 0.0;
    
    for i in 0..len {
        let diff = (beat_out[i] - gold_logits.beat[i]).abs();
        total_sq_err += diff * diff;
        if diff > max_diff {
            max_diff = diff;
        }
    }
    
    let mse = total_sq_err / len as f32;
    println!("Beat Logits Comparison Result:");
    println!("  MSE: {:.8}", mse);
    println!("  Max Diff: {:.8}", max_diff);
    
    if mse < 1e-4 {
        println!(">>> SUCCESS: Inference Parity Achieved (Beats).");
    } else {
        println!(">>> FAILURE: Inference differs significantly.");
        std::process::exit(1);
    }

    Ok(())
}
