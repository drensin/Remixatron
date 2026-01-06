use remixatron_lib::workflow::Remixatron;
use remixatron_lib::playback_engine::JukeboxEngine;
use std::thread;
use std::time::Duration;
use std::path::Path;

fn main() {
    println!("I am the verification binary. I am running.");
    println!("Current Dir: {:?}", std::env::current_dir().unwrap());
    
    // Model Paths (Relative to src-tauri, assuming run from there)
    let mel_path = "../MelSpectrogram_Ultimate.onnx";
    let beat_path = "../BeatThis_small0.onnx";
    
    // Verify paths exist
    if !Path::new(mel_path).exists() {
        eprintln!("Error: Model not found at {}", mel_path);
        // Try current dir?
        if Path::new("MelSpectrogram_Ultimate.onnx").exists() {
            eprintln!("Found model in current dir, please update paths.");
        }
        return;
    }

    // Audio Path from CLI args or default
    let args: Vec<String> = std::env::args().collect();
    let default_path = "/home/rensin/Downloads/Two Timin' Man.mp3";
    let audio_path = if args.len() > 1 {
        &args[1]
    } else {
        default_path
    };

    if !Path::new(audio_path).exists() {
        eprintln!("Error: Audio file not found at {}", audio_path);
        eprintln!("Usage: verify_audio [path_to_mp3]");
        return;
    }
    
    let remixatron = Remixatron::new(mel_path, beat_path);
    println!("Analyzing {}... (This may take a moment)", audio_path);
    
    match remixatron.analyze(audio_path, |status| println!("[Status]: {}", status)) {
        Ok(analysis) => {
            println!("Analysis complete. Beats: {}", analysis.beat_structs.len());
            println!("K-Optimal: {}", analysis.k_optimal);
            
            let mut engine = JukeboxEngine::new(analysis.beat_structs, analysis.k_optimal);

            
            let cand_count: usize = engine.beats.iter().map(|b| b.jump_candidates.len()).sum();
            println!("Identified {} jump candidates.", cand_count);

            // DEBUG: Inspect Beat Data
            for i in 0..5.min(engine.beats.len()) {
                println!("Beat {}: BarPos: {}, Segment: {}", i, engine.beats[i].bar_position, engine.beats[i].segment);
            }
            if engine.beats.len() > 100 {
                 let i = 100;
                 println!("Beat {}: BarPos: {}, Segment: {}", i, engine.beats[i].bar_position, engine.beats[i].segment);
            }
            let max_bp = engine.beats.iter().map(|b| b.bar_position).max().unwrap_or(0);
            println!("Max Bar Position: {}", max_bp);
            
            if cand_count == 0 {
                eprintln!("WARNING: No jump candidates found! Infinite Walk will be a straight line.");
            }
            
            println!("Loading track into Kira...");
            if let Err(e) = engine.load_track(audio_path) {
                eprintln!("Failed to load track: {}", e);
                return;
            }
            
            println!("Generating Play Vector & Scheduling Audio...");
            // note: play() generates vector AND schedules playback
            // To inspect vector, we can compute it first!
            let mut instructions = Vec::new();
            // Create a clone of engine to simulate without disturbing state? 
            // Actually JIT changes state. So if we generate 100 here, the engine has advanced 100 steps.
            println!("Simulating 100 steps to verify logic...");
            for _ in 0..100 {
                instructions.push(engine.get_next_beat());
            }
             println!("DEBUG: Generated Vector (First 20): {:?}", instructions.iter().take(20).map(|i| i.beat_id).collect::<Vec<_>>());
             println!("DEBUG: Generated Vector (Last 20): {:?}", instructions.iter().skip(instructions.len().saturating_sub(20)).map(|i| i.beat_id).collect::<Vec<_>>());

            // Re-running play() will regenerate a slightly different random vector.
            // Be aware of this discrepancy. ideally we modify play to take vector (refactoring),
            // but for now let's just observe play() behavior by relying on the engine debug prints I just enabled.
            if let Err(e) = engine.play() {
                 eprintln!("Playback error: {}", e);
                 return;
            }
            
            println!("Playback scheduling complete (100 beats). Draining buffer for 10 seconds...");
            thread::sleep(Duration::from_secs(10));
            println!("Finished.");
        },
        Err(e) => {
            eprintln!("Analysis failed: {}", e);
        }
    }
}
