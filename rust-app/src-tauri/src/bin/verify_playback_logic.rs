use remixatron_lib::playback_engine::{Beat, JukeboxEngine};

fn main() {
    println!("Verifying Playback Logic Parity...");

    // 1. Create Mock Beats
    // Scenario: Simple 4/4 loop with 2 segments
    // 0,1,2,3 (Seg 0) -> 4,5,6,7 (Seg 1)
    // We want to force a jump from 3 -> 0 (loop) logic.
    
    let mut beats = Vec::new();
    for i in 0..8 {
        beats.push(Beat {
            id: i,
            start: i as f32 * 0.5,
            duration: 0.5,
            bar_position: i % 4,
            cluster: if i < 4 { 0 } else { 0 }, // Same cluster to allow jumps
            segment: if i < 4 { 0 } else { 1 },
            intra_segment_index: i % 4,
            quartile: 0, // Will be computed
            jump_candidates: Vec::new(), // Will be computed
        });
    }

    // 2. Initialize Engine
    let mut engine = JukeboxEngine::new(beats, 1);
    
    // 3. Compute Jump Candidates
    println!("Identifying candidates...");
    engine.identify_jump_candidates();
    
    // Verify candidates manually
    // Beat 3 (Seg 0, is=3, bar=3) -> Next is Beat 4 (Seg 1, is=0, bar=0)
    // Candidate: Beat 7 (Seg 1, is=3, bar=3) -> Next is Beat 8 (Wrap->0, Seg 0, is=0, bar=0)
    // Check parity:
    // Beat 3 -> Next 4.
    // Candidate 7. 
    // 7.cluster == 4.cluster (0 == 0) -> OK
    // 7.is == 4.is? (3 != 0) -> NO.
    // Wait, Python 'is' check: `bx['is'] == beats[beat['next']]['is']`.    
    // Beat 3 next is 4. 4.is is 0.
    // Candidate 7. 7.is is 3.
    // 3 != 0. So 7 is NOT a candidate for 3?
    
    // Let's re-read parity logic I implemented:
    // `candidate.intra_segment_index == next_beat.intra_segment_index`
    
    // So if I want a jump from 3 -> 0.
    // Source: 3. Next: 4.
    // Target: 0. 
    // 0 is valid IF 0.is == 4.is.
    // 0.is = 0. 4.is = 0. -> MATCH!
    // 0.cluster == 4.cluster -> MATCH.
    // 0.bar == 4.bar -> MATCH.
    // 0.segment (0) != 3.segment (0)? FALSE. Same segment.
    // So 0 is NOT a candidate for 3 because same segment.
    
    // So to jump 3 -> 0, I need 0 to be in a DIFFERENT segment?
    // Correct! "Infinite Walk" jumps BETWEEN segments.
    
    // So if I have Seg 0 and Seg 1.
    // 3 (Seg 0) -> Next 4 (Seg 1).
    // Can I jump to ? (Seg X).
    // Candidates for 3 must look like 4.
    // If I want to jump to 0 (Seg 0), 0 must look like 4?
    // And 0 must be in diff segment than 3? No, 0 is Seg 0, 3 is Seg 0.
    // So Intra-Segment internal jumps are BANNED by `candidate.segment != current_segment`.
    
    // So I need at least 3 segments to have rich jumps? 
    // Or 2 segments where we jump Seg0->Seg1 and Seg1->Seg0.
    
    // Let's Inspect candidates for Beat 7 (Seg 1).
    // 7 -> Next 0 (Seg 0).
    // Candidates must match 0.
    // Target T.
    // T.cluster == 0.cluster.
    // T.is == 0.is.
    // T.bar == 0.bar.
    // T.segment != 7.segment (1).
    // T != 0 (Next).
    
    // Example Target: Beat 3?
    // 3.is = 3. 0.is = 0. NO.
    // Example Target: Beat 4? (Seg 1).
    // 4.segment == 7.segment. NO.
    
    // Example Target: Beat 0?
    // 0.segment (0) != 7.segment (1). OK.
    // 0.is (0) == 0.is (0). OK.
    // 0 != 0? FALSE. Target cannot be Next.
    
    // So with only 2 identical segments, logic is strict!
    // We need variation.
    
    // Let's create beats that ALLOW jumps.
    // Beat 3 (Seg 0) -> Next 4 (Seg 1). 
    // We want a candidate T (Seg 1) that looks like 4.
    // 4 is (is=0, bar=0).
    // If we have Beat 8 (Seg 2), is=0, bar=0.
    // Then 8 is valid candidate for 3!
    
    // So let's make 3 segments.
    // 0-3 (Seg 0), 4-7 (Seg 1), 8-11 (Seg 2).
    // All identical structure.
    
    // Beat 3 (Seg 0) -> Next 4 (Seg 1).
    // Candidate for 3: Beat 8 (Seg 2).
    // Checks:
    // 8.cluster == 4.cluster (0==0). OK.
    // 8.is == 4.is (0==0). OK.
    // 8.bar == 4.bar (0==0). OK.
    // 8.segment (2) != 3.segment (0). OK.
    // 8 != 4. OK.
    // Dist > 7? |8 - 4| = 4. 4 > 7? NO.
    // Wait, dist check: `(j - next)`. `candidate.id - next.id`.
    // |8 - 4| = 4.
    // Default dist threshold is > 7?
    // Line 347 in playback_engine.rs: `dist > 7`.
    // So indices must be far apart!
    
    // So I need MORE beats.
    // Let's make 4 segments of 4 beats = 16 beats.
    // Seg 0: 0-3
    // Seg 1: 4-7
    // Seg 2: 8-11
    // Seg 3: 12-15
    
    // Beat 3 (Seg 0) -> Next 4.
    // Candidate 12 (Seg 3).
    // |12 - 4| = 8. > 7? YES.
    // So 12 should be a candidate for 3.
    
    // Candidate 4? No (it is next).
    // Candidate 8 (Seg 2)? |8 - 4| = 4. Too close.
    
    println!("Regenerating beats (16 beats, 4 segments)...");
    beats = Vec::new();
    for i in 0..16 {
        beats.push(Beat {
            id: i,
            start: i as f32 * 0.5,
            duration: 0.5,
            bar_position: i % 4,
            cluster: 0,
            segment: i / 4,
            intra_segment_index: i % 4,
            quartile: 0,
            jump_candidates: Vec::new(),
        });
    }
    
    engine = JukeboxEngine::new(beats, 1);
    engine.identify_jump_candidates();
    
    // Inspect Beat 3
    // We cannot access fields of engine directly if private?
    // `beats` is private in JukeboxEngine!
    // We need to disable encapsulation for verification or use a getter.
    // `playback_engine.rs` defines `pub struct JukeboxEngine { beats: Vec<Beat> ... }`?
    // Let's check visibility. 
    // `beats: Vec<Beat>` is PRIVATE (no pub).
    // I cannot inspect `beats` from binary!
    
    // I need to add a getter or make `beats` pub for crate?
    // Or I rely on `compute_play_vector` output to verify jumps occurred.
    
    println!("Computing play vector...");
    let instructions = engine.compute_play_vector(100);
    
    let mut jumps = 0;
    for i in 0..instructions.len()-1 {
        let current = instructions[i].beat_id;
        let next = instructions[i+1].beat_id;
        let expected_next = if current + 1 < 16 { current + 1 } else { 0 };
        
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
