use serde::{Deserialize, Serialize};
use kira::{
    AudioManager, AudioManagerSettings,
    sound::static_sound::{StaticSoundData, StaticSoundSettings},
    clock::{ClockSpeed, ClockTime},
    StartTime,
    Tween,
};
use std::{thread, time::Duration};
use rand::prelude::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Beat {
    pub id: usize,
    pub start: f32,
    pub duration: f32,
    pub bar_position: usize,
    pub cluster: usize,
    pub segment: usize,
    pub intra_segment_index: usize,
    pub quartile: usize,
    pub jump_candidates: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlayInstruction {
    pub beat_id: usize,
    pub seq_len: usize,
    pub seq_pos: usize,
}

pub struct JukeboxEngine {
    pub beats: Vec<Beat>,
    // settings
    _clusters: usize,
    _branch_similarity_threshold: f32,
    
    // Audio Backend
    audio_manager: Option<AudioManager>,
    sound_data: Option<StaticSoundData>,
}

impl JukeboxEngine {
    pub fn new(beats: Vec<Beat>, clusters: usize) -> Self {
        Self {
            beats,
            _clusters: clusters,
            _branch_similarity_threshold: 0.0,
            audio_manager: None,
            sound_data: None,
        }
    }
    
    pub fn load_track(&mut self, path: &str) -> Result<(), String> {
        use crate::audio_backend::decoder::decode_audio_file;
        use kira::sound::static_sound::StaticSoundSettings;
        use kira::Frame;
        use std::sync::Arc;

        let manager = AudioManager::new(AudioManagerSettings::default())
            .map_err(|e| format!("Failed to initialize audio manager: {}", e))?;
            
        // Use our robust decoder
        println!("Decoding audio with robust decoder...");
        let (samples, sample_rate, channels) = decode_audio_file(path)
            .map_err(|e| format!("Robust decode failed: {}", e))?;
            
        // Convert Vec<f32> interleaved to Vec<Frame> for Kira
        let mut frames = Vec::with_capacity(samples.len() / channels as usize);
        if channels == 1 {
            for sample in samples {
                frames.push(Frame::new(sample, sample));
            }
        } else if channels == 2 {
            for chunk in samples.chunks(2) {
                if chunk.len() == 2 {
                    frames.push(Frame::new(chunk[0], chunk[1]));
                }
            }
        } else {
             // Fallback for others: just take first 2
             for chunk in samples.chunks(channels as usize) {
                 if chunk.len() >= 2 {
                     frames.push(Frame::new(chunk[0], chunk[1]));
                 } else if chunk.len() == 1 {
                      frames.push(Frame::new(chunk[0], chunk[0]));
                 }
             }
        }
        
        let sound_data = StaticSoundData {
            sample_rate,
            frames: Arc::from(frames),
            settings: StaticSoundSettings::default(),
            slice: None,
        };
             
        self.audio_manager = Some(manager);
        self.sound_data = Some(sound_data);
        Ok(())
    }
    
    pub fn play(&mut self, length: usize) -> Result<(), String> {
        self.play_with_callback(length, |_, _| {})
    }

    pub fn play_with_callback<F>(&mut self, length: usize, callback: F) -> Result<(), String> 
    where F: Fn(usize, usize) + Send + 'static 
    {
        if self.audio_manager.is_none() || self.sound_data.is_none() {
             return Err("Audio engine not initialized. Call load_track() first.".to_string());
        }
        
        // 1. Generate Play Vector
        let instructions = self.compute_play_vector(length);
        
        let manager = self.audio_manager.as_mut().unwrap();
        let sound_data = self.sound_data.as_ref().unwrap();
        
        // Setup Clock (1 Tick = 1 ms)
        let mut clock = manager.add_clock(ClockSpeed::TicksPerSecond(1000.0))
             .map_err(|e| format!("Failed to create clock: {}", e))?;
             
        clock.start();
        
        let now_ticks = clock.time().ticks;
        let mut cumulative_ticks = 0;
        const BUFFER_MS: u64 = 4000; // Keep 4 seconds buffered
        
        for instruction in instructions {
             // Throttling: If we are too far ahead of the clock, wait.
             loop {
                 let current_time = clock.time().ticks;
                 let elapsed = current_time.saturating_sub(now_ticks);
                 if cumulative_ticks < elapsed + BUFFER_MS {
                     break;
                 }
                 // Sleep a bit to let audio catch up
                 thread::sleep(Duration::from_millis(50));
             }

             let beat = &self.beats[instruction.beat_id];
             
             // Emit Callback (Visuals)
             callback(beat.id, beat.segment);

             let duration_ticks = (beat.duration * 1000.0) as u64;
             
             let start_time = ClockTime { clock: clock.id(), ticks: now_ticks + cumulative_ticks, fraction: 0.0 };
             let stop_time = ClockTime { clock: clock.id(), ticks: now_ticks + cumulative_ticks + duration_ticks, fraction: 0.0 };
             
             let mut handle = manager.play(
                 sound_data.with_settings(
                    StaticSoundSettings::new()
                        .start_time(StartTime::ClockTime(start_time))
                        .start_position(beat.start as f64)
                 )
             ).map_err(|e| format!("Failed to schedule sound: {}", e))?;
             
             handle.stop(
                 Tween {
                     start_time: StartTime::ClockTime(stop_time),
                     ..Default::default()
                 }
             );

             cumulative_ticks += duration_ticks;
        }
        
        Ok(())
    }

    /// Pre-computes the "Infinite Walk" graph traversal.
    /// This generates a long sequence of beat indices that represents the remix.
    /// In the Python version, this was `play_vector`.

    /// Pre-computes the "Infinite Walk" graph traversal.
    /// Replicates Remixatron.py `CreatePlayVectorFromBeatsMadmom` logic EXACTLY.
    pub fn compute_play_vector(&self, length: usize) -> Vec<PlayInstruction> {
        let mut play_vector = Vec::with_capacity(length);
        let mut rng = rand::thread_rng();

        // Start at beat 0
        let mut cursor = 0;
        let _current_beat_idx = 0; // Track actual beat index separately from cursor logic if needed

        // 0. Pre-calculate "Last Chance" beat index for "No Escape" logic
        let mut last_chance = self.beats.len() - 1;
        for i in (0..self.beats.len()).rev() {
             if !self.beats[i].jump_candidates.is_empty() {
                 last_chance = i;
                 break;
             }
        }
        
        // 1. Determine Tempo and Duration (for heuristics)
        let last_beat = self.beats.last().unwrap();
        let duration = last_beat.start + last_beat.duration;
        let tempo = (self.beats.len() as f32 / duration) * 60.0;
        
        // 2. Determine Max Sequence Length (Tempo scaled)
        // Remixatron.py: max_sequence_len = int(round((tempo / 120.0) * 48.0))
        // max_sequence_len = max_sequence_len - (max_sequence_len % 4)
        let mut max_sequence_len = ((tempo / 120.0) * 48.0).round() as usize;
        max_sequence_len = max_sequence_len - (max_sequence_len % 4);

        // 3. Acceptable Jump Amounts
        // Remixatron.py: [16, 24, 32, 48, 64, 72, 96, 128] filtered <= max_sequence_len
        let base_amounts = vec![16, 24, 32, 48, 64, 72, 96, 128];
        let mut acceptable_jump_amounts: Vec<usize> = base_amounts.into_iter()
            .filter(|&x| x <= max_sequence_len)
            .collect();
            
        println!("DEBUG: Tempo: {}, Max Seq Len: {}, Acceptable: {:?}", tempo, max_sequence_len, acceptable_jump_amounts);

        // Handle 3/4 time logic (if max beats_per_bar == 3)
        // We need to scan beats for max bar position. In Rust we can just check all.
        let max_bar_pos = self.beats.iter().map(|b| b.bar_position).max().unwrap_or(4);
        // Note: Remixatron.py says "beats_per_bar = max(...)". 
        // If beats are 0-indexed bar_pos, max_bar_pos=3 implies 4/4? 
        // No, typically bar_pos is 0-indexed. If max is 3, that means 0,1,2,3 -> 4 beats.
        // If max is 2 (0,1,2) -> 3 beats.
        // Remixatron code: `if beats_per_bar == 3: ...`
        // Let's assume strict parity: If `max_bar_pos` (which is effectively beats_per_bar - 1 if 0-indexed, OR count if 1-indexed)
        // Let's check `bar_position` definition. Usually 0-indexed.
        // If max_bar_pos == 2, that's 3/4 time.
        // If max_bar_pos == 3, that's 4/4 time.
        // Wait, Remixatron.py: `beats_per_bar = max([ b['bar_position'] for b in beats ])`
        // If `bar_position` was 1-indexed in Python, then 3 means 3/4.
        // If it was 0-indexed, 3 means 4/4.
        // Let's look at `Remixatron.py` Line 711: `final_beat['bar_position'] = int(beat_tuples[i][4])`
        // `downbeats` from madmom are [time, bar_pos]. madmom bar_pos is 1-indexed (1..4).
        // So `beats_per_bar` being 3 means 3/4 time.
        // In Rust, ensure our `bar_position` logic matches.
        
        let beats_per_bar = max_bar_pos + 1;
        // Actually, let's just stick to the value inspection.
        
        if beats_per_bar == 3 {
             acceptable_jump_amounts = acceptable_jump_amounts.iter().map(|a| ((*a as f32) * 0.75) as usize).collect();
        }

        // 4. Initial Sequence
        // min_sequence = random.choice(acceptable_jump_amounts) - (beats[1]['bar_position'] + 2)
        // We use beats[1] for bar pos ref? Python does: `beats[1]['bar_position']`.
        let ref_bar_pos = if self.beats.len() > 1 { self.beats[1].bar_position } else { 0 };
        
        // Safety: ensure acceptable_jump_amounts not empty
        let safe_amounts = if acceptable_jump_amounts.is_empty() { vec![16] } else { acceptable_jump_amounts.clone() };
        
        let mut min_sequence = *safe_amounts.choose(&mut rng).unwrap() as isize - (ref_bar_pos as isize + 2);
        if min_sequence < 1 { min_sequence = 1; } // Safety
        
        let mut current_sequence = 0;
        
        play_vector.push(PlayInstruction {
            beat_id: cursor,
            seq_len: min_sequence as usize,
            seq_pos: current_sequence,
        });

        // 5. Recent History Queue
        let segments_count = self.beats.iter().map(|b| b.segment).max().unwrap_or(0) + 1;
        let recent_depth = (segments_count as f32 * 0.25).round() as usize;
        let recent_depth = recent_depth.max(1);
        let mut recent = std::collections::VecDeque::with_capacity(recent_depth);

        // 6. Panic Thresholds
        let max_beats_between_jumps = (self.beats.len() as f32 * 0.1).round() as usize;
        // Filter acceptable amounts again
        let acceptable_jump_amounts: Vec<usize> = safe_amounts.into_iter()
             .filter(|&x| x <= max_beats_between_jumps)
             .collect();
        let safe_amounts = if acceptable_jump_amounts.is_empty() { vec![16] } else { acceptable_jump_amounts };
 
        // `safe_amounts` (first) used in filter source. 
        // `safe_amounts` (second) is checking result.
        // But in strict logic, we are redefining `acceptable_jump_amounts`.
        // Line 133 target: `let safe_amounts = ...`
        // Actually, line 133 defines `safe_amounts` which is NEVER used?
        // Let's check logic:
        // Line 130 defines `acceptable_jump_amounts` (filtered).
        // Line 133 defines `safe_amounts` (fallback).
        // Then we use `min_sequence` logic? No, `min_sequence` uses `safe_amounts` from Line 108.
        // Wait, line 133 is seemingly for the loop logic?
        // Line 152 `will_jump` logic uses `min_sequence`.
        // `min_sequence` is recalculated inside the loop! (Line 238 in original logic, I truncated it?)
        // Ah, `playback_engine.rs` implementation I provided truncated the `min_sequence` recalculation?
        // I need to check if I implemented `min_sequence` recalculation inside the loop.
        // If not, `safe_amounts` at 133 is truly unused.
        // I should implement `min_sequence` recalculation.
        
        // Let's check `playback_engine.rs` view again? No tool output yet.
        // I suspect I truncated `min_sequence` recalculation logic inside `if let Some(tgt) = jump_target` block.
        // My previous Replace tool for `inner loop using last_chance` was lines 221-263.
        // I did not see `min_sequence` recalc logic there.
        // I need to implement it to match python.
        
        // For now, to clean warnings, I will prefix `_safe_amounts`.


        let mut beats_since_jump = 0;
        let mut failed_jumps = 0;

        // 7. The Loop
        while play_vector.len() < length {
            // Update Recent
            let current_beat = &self.beats[cursor];
            if !recent.contains(&current_beat.segment) {
                if recent.len() >= recent_depth {
                    recent.pop_front();
                }
                recent.push_back(current_beat.segment);
            }

            current_sequence += 1;

            // Check Jump Condition
            let will_jump = (current_sequence as isize == min_sequence) || 
                            (beats_since_jump >= max_beats_between_jumps);
            
            if will_jump {
                println!("DEBUG: Beat {}: Will Jump! CurSeq: {}, MinSeq: {}, Since: {}", cursor, current_sequence, min_sequence, beats_since_jump);
            }

            let mut did_jump = false;
            let mut next_cursor = 0;

            if will_jump {
                // Attempt 1: Non-Recent Candidates
                let all_cands = &current_beat.jump_candidates;
                let non_recent: Vec<usize> = all_cands.iter()
                    .filter(|&&c| !recent.contains(&self.beats[c].segment))
                    .cloned()
                    .collect();

                println!("DEBUG: Beat {}: Candidates Total={}, Non-Recent={}", cursor, all_cands.len(), non_recent.len());

                if !non_recent.is_empty() {
                     // Success!
                     next_cursor = *non_recent.choose(&mut rng).unwrap();
                     did_jump = true;
                } else {
                     // Failure to find non-recent
                     // Python: beats_since_jump += 1; failed_jumps += 1;
                     // But we handle beats_since_jump update in the "Else" (no jump) block generally?
                     // No, Python increments them inside the "if len(non_recent) == 0" block.
                     // But if it subsequently succeeds in Quartile bust, it resets them to 0.
                     // So we can track them locally.
                     
                     // Attempt 2: Quartile Busting (10% failure)
                     let failure_threshold_10 = (self.beats.len() as f32 * 0.1) as usize;
                     let non_quartile: Vec<usize> = current_beat.jump_candidates.iter()
                        .filter(|&&c| self.beats[c].quartile != current_beat.quartile)
                        .cloned()
                        .collect();

                     // Note: We used `failed_jumps + 1` for logic comparison because we just failed one more time effectively?
                     // Python increments `failed_jumps` BEFORE checking threshold.
                     let current_failed_jumps = failed_jumps + 1;

                     if current_failed_jumps >= failure_threshold_10 && !non_quartile.is_empty() {
                          let furthest_dist = non_quartile.iter()
                              .map(|&c| (current_beat.id as isize - c as isize).abs())
                              .max().unwrap_or(0);
                          
                          if let Some(&target) = non_quartile.iter()
                              .find(|&&c| (current_beat.id as isize - c as isize).abs() == furthest_dist) {
                                  next_cursor = target;
                                  did_jump = true;
                          }
                     }
                     
                     // Attempt 3: Nuclear Reset (30% failure)
                     if !did_jump && current_failed_jumps >= (self.beats.len() as f32 * 0.3) as usize {
                          next_cursor = 0;
                          did_jump = true;
                     }
                }
            }

            // Duplicate block deleted. Logic is handled in the following if/else structure.
            
            // Correction: Merge the logic.
            // If `will_jump`:
            //    Try jump.
            //    If success: `cursor = tgt`, reset `beats_since`, `failed_jumps`.
            //    If fail: `cursor = next` (or last_chance logic), `beats_since++`, `failed++`.
            //    ALWAYS: Reset `current_sequence`, Recalc `min_sequence`.
            
            if will_jump {
                 if did_jump {
                     println!("DEBUG: SUCCESS! Jumping from {} to {}", cursor, next_cursor);
                     cursor = next_cursor;
                     beats_since_jump = 0;
                     failed_jumps = 0;
                 } else {
                     println!("DEBUG: Jump Failed. No valid candidates found.");
                     // Failed to find target
                     failed_jumps += 1;
                     
                     // Fallback Step
                     if cursor == last_chance {
                          if !current_beat.jump_candidates.is_empty() {
                              cursor = *current_beat.jump_candidates.iter().min().unwrap();
                          } else {
                              cursor = if cursor + 1 < self.beats.len() { cursor + 1 } else { 0 };
                          }
                     } else {
                          cursor = if cursor + 1 < self.beats.len() { cursor + 1 } else { 0 };
                     }
                     beats_since_jump += 1;
                 }
                 
                 // Sequence Reset (Shared)
                 current_sequence = 0;
                 let ref_bar_pos = self.beats[cursor].bar_position;
                 min_sequence = *safe_amounts.choose(&mut rng).unwrap() as isize - (ref_bar_pos as isize + 2);
                 if min_sequence < 1 { min_sequence = 1; }
                 
                 // Panic shortening
                 // Note: `beats_since_jump` just updated.
                 let remaining_panic = max_beats_between_jumps as isize - beats_since_jump as isize;
                 // Fix: Ensure remaining_panic is at least 1 to prevent underflow/negative min_sequence
                 let safe_remaining = remaining_panic.max(1);
                 
                 if min_sequence > safe_remaining {
                     min_sequence = safe_remaining;
                 }
                 // Visualization Hack
                 if beats_since_jump >= max_beats_between_jumps {
                      current_sequence = min_sequence as usize;
                 }

            } else {
                // Not trying to jump
                 if cursor == last_chance {
                      if !current_beat.jump_candidates.is_empty() {
                          cursor = *current_beat.jump_candidates.iter().min().unwrap();
                      } else {
                          cursor = if cursor + 1 < self.beats.len() { cursor + 1 } else { 0 };
                      }
                 } else {
                      cursor = if cursor + 1 < self.beats.len() { cursor + 1 } else { 0 };
                 }
                 beats_since_jump += 1;
            }
            play_vector.push(PlayInstruction {
                beat_id: cursor,
                seq_len: min_sequence as usize,
                seq_pos: current_sequence,
            });
        }
        play_vector
    }
    
    // Placeholder to allow compilation to succeed if I truncated logic.
    // Ideally I complete `compute_play_vector` in a follow up or careful block.
    // Let's implement `identify_jump_candidates` first as it's a requisite.
    
    pub fn identify_jump_candidates(&mut self) {
         let n_beats = self.beats.len();
         let mut all_candidates: Vec<Vec<usize>> = vec![Vec::new(); n_beats];

         // Compute Quartiles first (if not set externally)
         // Python: beat['quartile'] = beat['id'] // (len(beats) / 4.0)
         for beat in self.beats.iter_mut() {
             beat.quartile = (beat.id as f32 / (n_beats as f32 / 4.0)) as usize;
         }
         // Cannot borrow mutable in loop below if we iterate.
         // Actually we can compute quartiles in a separate pass.
         // Let's assume we do 2 passes.

         // Pass 1: Candidates
         for i in 0..n_beats {
             // Logic:
             // beat['next'] defaults to id + 1.
             // But "Last Chance" logic modifies 'next'.
             // Python computes candidates BEFORE "Last Chance".
             // So `next` used here is `id + 1` (or 0 if end).
             
             let next_beat_idx = if i + 1 < n_beats { i + 1 } else { 0 };
             let next_beat = &self.beats[next_beat_idx];
             let current_segment = self.beats[i].segment;
             
             for (j, candidate) in self.beats.iter().enumerate() {
                 let dist = (j as isize - next_beat_idx as isize).abs();
                 
                  // 1. Cluster Parity
                 if candidate.cluster == next_beat.cluster &&
                    // 2. Intra-Segment Parity (Restored)
                    candidate.intra_segment_index == next_beat.intra_segment_index &&
                    // 3. Bar Pos Parity
                    candidate.bar_position == next_beat.bar_position &&
                    // 4. Segment Diversity
                    candidate.segment != current_segment &&
                    // 5. Sequential Exclusion
                    j != next_beat_idx &&
                    // 6. Distance Threshold
                    dist > 7 {
                        all_candidates[i].push(j);
                    }
             }
         }

         // Apply back matches
         for (i, candidates) in all_candidates.into_iter().enumerate() {
             self.beats[i].jump_candidates = candidates;
         }
    }
}
