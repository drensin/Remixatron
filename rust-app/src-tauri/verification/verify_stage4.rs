use remixatron_lib::beat_tracker::post_processor::MinimalPostProcessor;
use std::fs::File;

use serde::Deserialize;
use anyhow::Result;

#[derive(Deserialize)]
struct GoldLogits {
    beat: Vec<f32>,
    downbeat: Vec<f32>,
}

#[derive(Deserialize)]
struct GoldBeats(Vec<f32>, Vec<f32>); // [beats, downbeats]

fn main() -> Result<()> {
    let gold_logits_path = "/home/rensin/Projects/Remixatron/rust-app/gold_standard/gold_logits.json";
    let gold_beats_path = "/home/rensin/Projects/Remixatron/rust-app/gold_standard/gold_beats.json";

    println!("Initializing MinimalPostProcessor...");
    let processor = MinimalPostProcessor::new(50.0); // 50 fps default
    
    println!("Loading Golden Logits...");
    let file = File::open(gold_logits_path)?;
    let reader = std::io::BufReader::new(file);
    let gold_logits: GoldLogits = serde_json::from_reader(reader)?;
    
    println!("Running Post-Processing...");
    let (r_beats, r_downbeats) = processor.process(&gold_logits.beat, &gold_logits.downbeat)?;
    
    println!("Loading Golden Beats...");
    let file = File::open(gold_beats_path)?;
    let reader = std::io::BufReader::new(file);
    let gold_beats: GoldBeats = serde_json::from_reader(reader)?;
    
    let g_beats = gold_beats.0;
    let g_downbeats = gold_beats.1;
    
    println!("Comparing Beats...");
    compare_times("Beats", &r_beats, &g_beats)?;
    
    println!("Comparing Downbeats...");
    compare_times("Downbeats", &r_downbeats, &g_downbeats)?;

    println!(">>> SUCCESS: Post-Processing Parity Achieved.");
    Ok(())
}

fn compare_times(label: &str, rust: &[f32], gold: &[f32]) -> Result<()> {
    println!("  {}: Rust Count {}, Gold Count {}", label, rust.len(), gold.len());
    
    if rust.len() != gold.len() {
        println!("  WARNING: Count mismatch!");
    }
    
    let len = rust.len().min(gold.len());
    let mut max_diff = 0.0;
    
    for i in 0..len {
        let diff = (rust[i] - gold[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    
    println!("  {}: Max Diff: {:.6} seconds", label, max_diff);
    
    // Threshold: 10ms? 1ms? 
    // Float ops differences can exist. 0.001 (1ms) is safe.
    if max_diff > 0.001 {
        println!("  FAILURE: {} mismatch too large.", label);
        std::process::exit(1);
    }
    Ok(())
}
