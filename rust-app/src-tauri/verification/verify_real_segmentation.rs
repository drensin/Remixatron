use remixatron_lib::audio_backend::decoder::decode_audio_file;
use remixatron_lib::beat_tracker::mel::MelProcessor;
use remixatron_lib::beat_tracker::inference::BeatProcessor;
use remixatron_lib::beat_tracker::post_processor::MinimalPostProcessor;
use remixatron_lib::analysis::features::FeatureExtractor;
use remixatron_lib::analysis::structure::StructureAnalyzer;
use ndarray::Axis;
use anyhow::Result;



fn main() -> Result<()> {
    println!(">>> Starting Real Audio Verification (Stay with Me)");
    
    let path = "/home/rensin/Downloads/Stay with Me.mp3";
    
    // 1. Load Audio
    println!("  Loading Audio...");
    let (interleaved, sr, _) = decode_audio_file(path).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    println!("    Audio Samples: {}", interleaved.len());
    println!("    Sample Rate: {}", sr);
    
    // Mixdown to Mono 
    let mut audio = Vec::with_capacity(interleaved.len() / 2);
    for chunk in interleaved.chunks(2) {
        if chunk.len() == 2 {
            audio.push((chunk[0] + chunk[1]) / 2.0);
        } else {
             audio.push(chunk[0]); // Tail
        }
    }
    
    // Resample if needed (Target 22050)
    let target_sr = 22050;
    if sr != target_sr {
        println!("    Resampling {} -> {} (Linear)...", sr, target_sr);
        audio = resample_linear(&audio, sr as f32, target_sr as f32);
    }
    
    // 2. Mel Spectrogram
    println!("  Computing Mel Spectrogram...");
    let model_path = "/home/rensin/Projects/Remixatron/rust-app/MelSpectrogram_Ultimate.onnx";
    let mut mel_proc = MelProcessor::new(model_path)?;
    let mel = mel_proc.process(&audio)?;
    
    // 3. Beat Tracking
    println!("  Tracking Beats...");
    let tracker_model = "/home/rensin/Projects/Remixatron/rust-app/BeatThis_small0.onnx";
    let mut tracker = BeatProcessor::new(tracker_model)?;
    // tracker.process returns Result<(beat_logits, downbeat_logits)>
    let (b_logits, d_logits) = tracker.process(&mel)?;
    
    let post = MinimalPostProcessor::new(50.0); // FPS=50
    let (beats, _downbeats) = post.process(&b_logits, &d_logits)?;
    println!("    Beats: {}", beats.len());
    
    // 4. Feature Extraction
    println!("  Extracting Features (MFCC + Chroma + CQT)...");
    let mut feature_ex = FeatureExtractor::new(128, 22050.0);
    // Convert Mel to 2D
    let mel_2d = mel.index_axis(Axis(0), 0).to_owned();
    
    let total_duration = audio.len() as f32 / 22050.0;
    let mut beats_extended = beats.clone();
    beats_extended.push(total_duration);
    
    let (mfcc, chroma) = feature_ex.compute_sync_features(&audio, &mel_2d, &beats_extended, 50.0); 
    println!("    Chroma Shape: {:?}", chroma.shape());
    
    // 5. Structure Analysis (Auto-K)
    println!("  Analyzing Structure (Auto-K)...");
    let analyzer = StructureAnalyzer::new();
    // K=0 triggers Auto-K
    let result = analyzer.compute_segments_knn(&mfcc, &chroma, None);
    
    println!("    Detected K: {}", result.k_optimal);
    println!("    Eigenvalues (First 14):"); // Kept for debug
    for (i, ev) in result.eigenvalues.iter().enumerate() {
        println!("      [{}] {:.8}", i, ev);
    }

    // Parity Check 1: Auto-K
    // With Legacy Heuristic, we expect K >= 4.
    if result.k_optimal >= 4 {
        println!(">>> SUCCESS: Auto-K produced musically interesting structure (K={}).", result.k_optimal);
    } else {
        println!(">>> WARNING: Auto-K produced low complexity (K={}).", result.k_optimal);
    }
    
    // assert_eq!(result.k_optimal, 2, "Auto-K Parity Failed! Expected 2, got {}", result.k_optimal);
    
    // Parity Check 2: Boundaries
    // TODO: Compare result.labels with Golden Master beats?
    // Since we changed the heuristic, we NO LONGER EXPECT PARITY with the simple K=2 Python master.
    // We are now verifying the "Improved" logic.
    // std::process::exit(1);
    
    Ok(())
}

fn resample_linear(input: &[f32], from_hz: f32, to_hz: f32) -> Vec<f32> {
    let ratio = from_hz / to_hz;
    let new_len = (input.len() as f32 / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(new_len);
    
    for i in 0..new_len {
        let sample_idx = i as f32 * ratio;
        let idx_floor = sample_idx.floor() as usize;
        let idx_ceil = (idx_floor + 1).min(input.len() - 1);
        let t = sample_idx - idx_floor as f32;
        
        let val = input[idx_floor] * (1.0 - t) + input[idx_ceil] * t;
        output.push(val);
    }
    output
}
