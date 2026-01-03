use remixatron_lib::beat_tracker::mel::MelProcessor;
use remixatron_lib::analysis::features::FeatureExtractor;
use remixatron_lib::analysis::structure::StructureAnalyzer;
use ndarray::Axis;
use std::f32::consts::PI;
use anyhow::{Result, anyhow};



fn main() -> Result<()> {
    println!(">>> Starting Segmentation Verification (Synthetic)");
    
    // 1. Generate Synthetic Audio
    // Structure: A (10s) - B (10s) - A (10s) - C (10s)
    // A: 440 Hz (Note A4)
    // B: 554 Hz (Note C#5) + Noise
    // C: 659 Hz (Note E5)
    // Tempo: 120 BPM (0.5s per beat).
    // Total 40s = 80 beats.
    
    let sr = 22050;
    println!("  Generating Audio...");
    let mut signal = Vec::new();
    
    // Section A (0-10s): 440Hz (A4)
    signal.extend(generate_tone(440.0, 10.0, sr));
    // Section B (10-20s): 554Hz (C#5)
    signal.extend(generate_tone(554.0, 10.0, sr));

    // Section A (20-30s): 440Hz
    signal.extend(generate_tone(440.0, 10.0, sr));

    // Section C (30-40s): 349Hz (F4 - Distinct pitch)
    signal.extend(generate_tone(349.0, 10.0, sr));



    
    // 2. Mel Spectrogram
    println!("  Computing Mel Spectrogram...");
    let model_path = "/home/rensin/Projects/Remixatron/rust-app/MelSpectrogram_Ultimate.onnx";
    if !std::path::Path::new(model_path).exists() {
         return Err(anyhow!("Model not found at {}", model_path));
    }
    
    let mut mel_proc = MelProcessor::new(model_path)?;
    let mel = mel_proc.process(&signal)?;
    println!("    Mel Shape: {:?}", mel.shape());
    
    // 3. Feature Extraction
    println!("  Extracting Features (MFCC + Chroma)...");
    let mut feature_ex = FeatureExtractor::new(128, sr as f32);
    
    // Hardcoded Beats: every 0.5s (120 BPM)
    // 40s / 0.5 = 80 beats.
    let mut beats = Vec::new();
    for i in 0..80 {
        beats.push(i as f32 * 0.5);
    }
    
    // Mel shape: [1, Time, 128]. Need [Time, 128]
    let mel_2d = mel.index_axis(Axis(0), 0).to_owned();
    
    let min = mel_2d.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = mel_2d.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    println!("    Mel Stats: Min={:.4}, Max={:.4}", min, max);

    let (mfcc, chroma) = feature_ex.compute_sync_features(&signal, &mel_2d, &beats, 50.0); // 50Hz FPS

    
    // DEBUG: Inspect Features
    // println!("    Chroma [Beat 10 (A)]: {:?}", chroma.slice(s![10, ..]));
    
    println!("    MFCC Shape: {:?}", mfcc.shape());
    println!("    Chroma Shape: {:?}", chroma.shape());

    
    println!("    MFCC Shape: {:?}", mfcc.shape());
    println!("    Chroma Shape: {:?}", chroma.shape());
    
    // 4. Segmentation (Spectral Clustering)
    println!("  Clustering (K=3)...");
    let analyzer = StructureAnalyzer::new();
    // Force K=3 for synthetic test
    let result = analyzer.compute_segments(&mfcc, &chroma, 3);
    let labels = result.labels;
    
    println!("    Labels: {:?}", labels);
    
    // 5. Assertions
    // Expected:
    // Beats 0-19 (A): Cluster X
    // Beats 20-39 (B): Cluster Y
    // Beats 40-59 (A): Cluster X
    // Beats 60-79 (C): Cluster Z
    
    let region_a1 = &labels[5..15];  // Middle of A1
    let region_b  = &labels[25..35]; // Middle of B
    let region_a2 = &labels[45..55]; // Middle of A2
    let region_c  = &labels[65..75]; // Middle of C
    
    let label_a = majority_vote(region_a1);
    let label_b = majority_vote(region_b);
    let label_c = majority_vote(region_c);
    let label_a2 = majority_vote(region_a2);
    
    println!("    Detected Clusters: A1={}, B={}, A2={}, C={}", label_a, label_b, label_a2, label_c);
    
    if label_a != label_a2 {
        println!("FAILURE: Section A did not re-cluster to same ID ({} vs {})", label_a, label_a2);
        return Err(anyhow!("Structure Mismatch A1 != A2"));
    }
    
    if label_a == label_b {
         println!("FAILURE: Section A and B collapsed ({} == {})", label_a, label_b);
         return Err(anyhow!("Structure Mismatch A == B"));
    }
    
    if label_b == label_c {
         println!("FAILURE: Section B and C collapsed ({} == {})", label_b, label_c);
         return Err(anyhow!("Structure Mismatch B == C"));
    }
    
    if label_a == label_c {
         println!("FAILURE: Section A and C collapsed ({} == {})", label_a, label_c);
         return Err(anyhow!("Structure Mismatch A == C"));
    }
    
    println!(">>> SUCCESS: Segmentation Verified. A-B-A-C structure recovered.");
    
    Ok(())
}

fn generate_tone(freq: f32, duration: f32, sr: u32) -> Vec<f32> {
    let samples = (duration * sr as f32) as usize;
    let mut vec = Vec::with_capacity(samples);
    for i in 0..samples {
        let t = i as f32 / sr as f32;
        vec.push((2.0 * PI * freq * t).sin());
    }
    vec
}



fn majority_vote(labels: &[usize]) -> usize {
    let mut counts = std::collections::HashMap::new();
    for &l in labels {
        *counts.entry(l).or_insert(0) += 1;
    }
    counts.into_iter().max_by_key(|&(_, count)| count).map(|(val, _)| val).unwrap()
}
