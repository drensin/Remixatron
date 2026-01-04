use remixatron_lib::playback_engine::{Beat, JukeboxEngine};

fn main() {
    println!("Verifying Playback Logic Parity...");

    // 1. Create Mock Beats
    // Scenario: Simple 4/4 loop with 2 segments
    // We manually inject a jump from Beat 3 -> Beat 0 to simulate a loop.
    
    let mut beats = Vec::new();
    for i in 0..8 {
        let mut candidates = Vec::new();
        // Inject Jump: Beat 3 can jump to Beat 0
        if i == 3 {
            candidates.push(0); // Jump to 0
        }
        
        beats.push(Beat {
            id: i,
            start: i as f32 * 0.5,
            duration: 0.5,
            bar_position: i % 4,
            cluster: 0,
            segment: if i < 4 { 0 } else { 1 },
            intra_segment_index: i % 4,
            quartile: 0, 
            jump_candidates: candidates,
        });
    }

    // 2. Initialize Engine
    let engine = JukeboxEngine::new(beats, 1);
    
    // 3. Compute Play Vector
    println!("Computing play vector...");
    let instructions = engine.compute_play_vector(100);
    
    let mut jumps = 0;
    for i in 0..instructions.len()-1 {
        let current = instructions[i].beat_id;
        let next = instructions[i+1].beat_id;
        let expected_next = if current + 1 < 8 { current + 1 } else { 0 };
        
        if next != expected_next {
            println!("Jump detected! {} -> {}", current, next);
            jumps += 1;
        }
    }
    
    println!("Total Jumps in 100 steps: {}", jumps);
    if jumps > 0 {
        println!("SUCCESS: Logic generated jumps.");
    } else {
        println!("WARNING: No jumps generated. Logic might be strict or RNG unlucky?");
    }
}
