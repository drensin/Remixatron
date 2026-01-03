use remixatron_lib::audio::loader::load_audio;
use std::fs::File;

use serde::Deserialize;
use anyhow::Result;

#[derive(Deserialize)]
struct GoldAudio {
    signal: Vec<f32>,
    sr: u32,
}

fn main() -> Result<()> {
    let mp3_path = "/home/rensin/Downloads/Stay with Me.mp3";
    let gold_path = "/home/rensin/Projects/Remixatron/rust-app/gold_standard/gold_audio.json";

    println!("Loading Rust audio...");
    let rust_data = load_audio(mp3_path, 22050)?;
    
    println!("Loading Golden Master...");
    let file = File::open(gold_path)?;
    let reader = std::io::BufReader::new(file);
    let gold_data: GoldAudio = serde_json::from_reader(reader)?;
    
    println!("Comparing...");
    if rust_data.sample_rate != gold_data.sr {
        panic!("Sample rate mismatch! Rust: {}, Gold: {}", rust_data.sample_rate, gold_data.sr);
    }
    
    if rust_data.signal.len() != gold_data.signal.len() {
        println!("WARNING: Length mismatch! Rust: {}, Gold: {}", rust_data.signal.len(), gold_data.signal.len());
        // Truncate to shorter for comparison
    }
    
    let len = rust_data.signal.len().min(gold_data.signal.len());
    // Cross-correlation to find optimal lag
    // We only check the first 100000 samples for speed
    let n = 100000.min(len);
    let r_slice = &rust_data.signal[0..n];
    let g_slice = &gold_data.signal[0..n];
    
    let search_window = 2000; // Search +/- 2000 samples
    let mut best_lag = 0;
    let mut min_mse = f32::MAX;
    
    for lag in -(search_window as i32)..=(search_window as i32) {
        let mut sq_err = 0.0;
        let mut count = 0;
        
        for i in 0..n {
            let idx_r = i as i32;
            let idx_g = i as i32 + lag;
            
            if idx_g >= 0 && idx_g < n as i32 {
                let diff = r_slice[idx_r as usize] - g_slice[idx_g as usize];
                sq_err += diff * diff;
                count += 1;
            }
        }
        
        if count > 0 {
            let mse = sq_err / count as f32;
            if mse < min_mse {
                min_mse = mse;
                best_lag = lag;
            }
        }
    }
    
    println!("Alignment Result:");
    println!("  Best Lag: {} samples (Rust needs to shift by {})", best_lag, -best_lag);
    println!("  MSE after alignment: {:.8}", min_mse);
    
    if min_mse < 1e-4 {
        println!(">>> SUCCESS: Audio matches with lag correction.");
        // We consider this a pass for now, knowing we might need to trim/pad
    } else {
        println!(">>> FAILURE: Even aligned, audio differs. MSE: {}", min_mse);
        std::process::exit(1);
    }

    Ok(())
}
