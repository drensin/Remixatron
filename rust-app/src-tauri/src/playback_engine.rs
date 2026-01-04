//! # Jukebox Playback Engine
//!
//! This module handles the "Infinite Walk" logic and the actual audio playback.
//! It uses `kira` for low-latency audio scheduling.
//!
//! ## Core Components
//! *   `JukeboxEngine`: The main controller.
//! *   `Beat`: A rich struct containing connectivity data (where can we jump to?).
//! *   `PlayInstruction`: A calculated step in the infinite walk.

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

/// A single musical beat with metadata for graph traversal.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Beat {
    /// Unique index of the beat in the song.
    pub id: usize,
    /// Start time in seconds.
    pub start: f32,
    /// Duration in seconds.
    pub duration: f32,
    /// Position in the bar (usually 0-3).
    pub bar_position: usize,
    /// The structural cluster ID this beat belongs to.
    pub cluster: usize,
    /// The specific segment instance ID.
    pub segment: usize,
    /// Index of this beat within its segment.
    pub intra_segment_index: usize,
    /// Quartile of the song (0-3), used for "Quartile Busting" logic.
    pub quartile: usize,
    /// List of valid beat IDs we can seamless jump to from here.
    pub jump_candidates: Vec<usize>,
}

/// A step in the computed playback plan.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlayInstruction {
    /// The beat to play.
    pub beat_id: usize,
    /// The target length of the current sequence (for debugging/visuals).
    pub seq_len: usize,
    /// Current position in that sequence.
    pub seq_pos: usize,
}

/// The Audio Engine and Graph Walker.
pub struct JukeboxEngine {
    pub beats: Vec<Beat>,
    // Settings
    _clusters: usize,
    _branch_similarity_threshold: f32,
    
    // Audio Backend (Kira)
    audio_manager: Option<AudioManager>,
    sound_data: Option<StaticSoundData>,
}

impl JukeboxEngine {
    /// Creates a new engine instance.
    pub fn new(beats: Vec<Beat>, clusters: usize) -> Self {
        Self {
            beats,
            _clusters: clusters,
            _branch_similarity_threshold: 0.0,
            audio_manager: None,
            sound_data: None,
        }
    }
    
    /// Loads and decodes the audio file into memory.
    ///
    /// This uses a robust custom decoder (`rodio` based) to handle various formats,
    /// then converts the raw samples into `Kira` frames.
    pub fn load_track(&mut self, path: &str) -> Result<(), String> {
        use crate::audio_backend::decoder::decode_audio_file;
        use kira::Frame;
        use std::sync::Arc;

        let manager = AudioManager::new(AudioManagerSettings::default())
            .map_err(|e| format!("Failed to initialize audio manager: {}", e))?;
            
        // Use our robust decoder
        println!("Decoding audio with robust decoder...");
        let (samples, sample_rate, channels) = decode_audio_file(path)
            .map_err(|e| format!("Robust decode failed: {}", e))?;
            
        // Convert Vec<f32> interleaved to Vec<Frame> for Kira
        // Kira expects stereo frames. We handle mono/stereo/surround manually.
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
             // Fallback for >2 channels: just take first 2 (Left/Right)
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
    
    /// Simple play command (no callback).
    pub fn play(&mut self, length: usize) -> Result<(), String> {
        self.play_with_callback(length, |_, _| {})
    }

    /// Starts playback with a callback for UI updates.
    ///
    /// # Arguments
    /// * `length` - How many beats to play (approximate, since it's infinite).
    /// * `callback` - Function called on every beat start (beat_id, segment_id).
    pub fn play_with_callback<F>(&mut self, length: usize, callback: F) -> Result<(), String> 
    where F: Fn(usize, usize) + Send + 'static 
    {
        if self.audio_manager.is_none() || self.sound_data.is_none() {
             return Err("Audio engine not initialized. Call load_track() first.".to_string());
        }
        
        // 1. Generate the "Infinite Walk" Plan
        // This pre-calculates the sequence of beats to play.
        let instructions = self.compute_play_vector(length);
        
        let manager = self.audio_manager.as_mut().unwrap();
        let sound_data = self.sound_data.as_ref().unwrap();
        
        // Setup Clock (1 Tick = 1 ms) to synchronize scheduling.
        let mut clock = manager.add_clock(ClockSpeed::TicksPerSecond(1000.0))
             .map_err(|e| format!("Failed to create clock: {}", e))?;
             
        clock.start();
        
        let now_ticks = clock.time().ticks;
        let mut cumulative_ticks = 0;
        const BUFFER_MS: u64 = 4000; // Keep 4 seconds buffered ahead
        
        for instruction in instructions {
             // Throttling: If we are too far ahead of the clock, wait.
             // This prevents calculating 10 hours of audio instantly and OOMing the scheduler.
             loop {
                 let current_time = clock.time().ticks;
                 let elapsed = current_time.saturating_sub(now_ticks);
                 if cumulative_ticks < elapsed + BUFFER_MS {
                     break;
                 }
                 // Backpressure: Wait for playback to catch up to the buffer.
                 // This yields the thread to prevent tight-looping while waiting.
                 thread::sleep(Duration::from_millis(50));
             }

             let beat = &self.beats[instruction.beat_id];
             
             // Emit Callback (Visuals)
             // CRITICAL BUG: This callback fires when the beat is *scheduled*, which is `BUFFER_MS` (4s) ahead of playback.
             // Indicates that the UI will display the "Future" state significantly earlier than the audio.
             //
             // TODO: FIX UI LATENCY (High Priority)
             // We must decouple the UI event from this scheduling loop.
             // Potential solutions:
             // 1. Send the `start_time` (clock ticks) to the frontend, let JS handle the wait.
             // 2. Spawn a separate thread that polls `clock.time()` and emits events at the correct moment.
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
    ///
    /// This mimics the probabilistic path finding from the original Python implementation.
    /// It balances "sticking to the song" (playing sequential beats) with "jumping"
    /// (moving to a similar beat elsewhere) to create an endless mix.
    pub fn compute_play_vector(&self, length: usize) -> Vec<PlayInstruction> {
        let mut play_vector = Vec::with_capacity(length);
        let mut rng = rand::thread_rng();

        // Cursor: The current beat we are playing.
        let mut cursor = 0;

        // 0. Pre-calculate "Last Chance" beat index for "No Escape" logic
        // If we reach the end of the song and there are no jumps, we must loop back
        // earlier. We find the last beat that HAS a jump candidate.
        let mut last_chance = self.beats.len() - 1;
        for i in (0..self.beats.len()).rev() {
             if !self.beats[i].jump_candidates.is_empty() {
                 last_chance = i;
                 break;
             }
        }
        
        // 1. Determine Tempo and Duration (simple heuristic)
        let last_beat = self.beats.last().unwrap();
        let duration = last_beat.start + last_beat.duration;
        let tempo = (self.beats.len() as f32 / duration) * 60.0;
        
        // 2. Determine Max Sequence Length (Tempo scaled)
        // We want to play longer uninterrupted sequences for faster songs.
        let mut max_sequence_len = ((tempo / 120.0) * 48.0).round() as usize;
        max_sequence_len = max_sequence_len - (max_sequence_len % 4);

        // 3. Define Acceptable Jump Points
        // We only jump after playing a phrase of these lengths (in beats).
        let base_amounts = vec![16, 24, 32, 48, 64, 72, 96, 128];
        let mut acceptable_jump_amounts: Vec<usize> = base_amounts.into_iter()
            .filter(|&x| x <= max_sequence_len)
            .collect();

        // Handle 3/4 time logic adjustment
        let max_bar_pos = self.beats.iter().map(|b| b.bar_position).max().unwrap_or(4);
        let beats_per_bar = max_bar_pos + 1;
        
        if beats_per_bar == 3 {
             acceptable_jump_amounts = acceptable_jump_amounts.iter().map(|a| ((*a as f32) * 0.75) as usize).collect();
        }

        // 4. Set Initial Sequence Target
        let ref_bar_pos = if self.beats.len() > 1 { self.beats[1].bar_position } else { 0 };
        let safe_amounts = if acceptable_jump_amounts.is_empty() { vec![16] } else { acceptable_jump_amounts.clone() };
        
        // Randomly pick how long we play before ATTEMPTING a jump.
        let mut min_sequence = *safe_amounts.choose(&mut rng).unwrap() as isize - (ref_bar_pos as isize + 2);
        if min_sequence < 1 { min_sequence = 1; }
        
        let mut current_sequence = 0;
        let mut beats_since_jump = 0;
        let mut failed_jumps = 0;
        
        // Add first instruction
        play_vector.push(PlayInstruction {
            beat_id: cursor,
            seq_len: min_sequence as usize,
            seq_pos: current_sequence,
        });

        // 5. Recent History Queue (Prevent immediate loops)
        let segments_count = self.beats.iter().map(|b| b.segment).max().unwrap_or(0) + 1;
        let recent_depth = (segments_count as f32 * 0.25).round() as usize;
        let recent_depth = recent_depth.max(1);
        let mut recent = std::collections::VecDeque::with_capacity(recent_depth);

        // 6. Panic Threshold
        // If we haven't jumped in 10% of the song, force a jump soon.
        let max_beats_between_jumps = (self.beats.len() as f32 * 0.1).round() as usize;

        // 7. The Main Loop
        while play_vector.len() < length {
            // Update Recent Segments
            let current_beat = &self.beats[cursor];
            if !recent.contains(&current_beat.segment) {
                if recent.len() >= recent_depth {
                    recent.pop_front();
                }
                recent.push_back(current_beat.segment);
            }

            current_sequence += 1;

            // Check if we should ATTEMPT a jump
            let will_jump = (current_sequence as isize == min_sequence) || 
                            (beats_since_jump >= max_beats_between_jumps);
            
            let mut did_jump = false;
            let mut next_cursor = 0;

            if will_jump {
                // Strategy 1: Standard Jump (Must not be to a recently visited segment)
                let all_cands = &current_beat.jump_candidates;
                let non_recent: Vec<usize> = all_cands.iter()
                    .filter(|&&c| !recent.contains(&self.beats[c].segment))
                    .cloned()
                    .collect();

                if !non_recent.is_empty() {
                     // Success! Pick a random valid candidate.
                     next_cursor = *non_recent.choose(&mut rng).unwrap();
                     did_jump = true;
                } else {
                     // Strategy 2: "Quartile Busting" (Relaxed constraints)
                     // If we failed too many times (10% of song length), allows jumping to same quartile.
                     // (Heuristic: usually we want to jump to a different part of the song, but desperation kicks in).
                     let failure_threshold_10 = (self.beats.len() as f32 * 0.1) as usize;
                     
                     // We count this as a failure before checking threshold
                     let current_failed_jumps = failed_jumps + 1;
                     
                     let non_quartile: Vec<usize> = current_beat.jump_candidates.iter()
                        .filter(|&&c| self.beats[c].quartile != current_beat.quartile)
                        .cloned()
                        .collect();

                     if current_failed_jumps >= failure_threshold_10 && !non_quartile.is_empty() {
                          // Find the furthest component to jump to (maximize change)
                          let furthest_dist = non_quartile.iter()
                              .map(|&c| (current_beat.id as isize - c as isize).abs())
                              .max().unwrap_or(0);
                          
                          if let Some(&target) = non_quartile.iter()
                              .find(|&&c| (current_beat.id as isize - c as isize).abs() == furthest_dist) {
                                  next_cursor = target;
                                  did_jump = true;
                          }
                     }
                     
                     // Strategy 3: "Nuclear Reset"
                     // If we are REALLY stuck (30% failures), jump to start.
                     if !did_jump && current_failed_jumps >= (self.beats.len() as f32 * 0.3) as usize {
                          next_cursor = 0;
                          did_jump = true;
                     }
                }
            }

            // Apply Outcome
            if will_jump {
                 if did_jump {
                     // Jump Succeeded
                     cursor = next_cursor;
                     beats_since_jump = 0;
                     failed_jumps = 0;
                 } else {
                     // Jump Failed (No candidates met criteria)
                     failed_jumps += 1;
                     
                     // Advance normally, unless we are at "Last Chance"
                     if cursor == last_chance {
                          // Forced jump if possible, else loop to start
                          if !current_beat.jump_candidates.is_empty() {
                              cursor = *current_beat.jump_candidates.iter().min().unwrap();
                          } else {
                              cursor = if cursor + 1 < self.beats.len() { cursor + 1 } else { 0 };
                          }
                     } else {
                          // Standard advance
                          cursor = if cursor + 1 < self.beats.len() { cursor + 1 } else { 0 };
                     }
                     beats_since_jump += 1;
                 }
                 
                 // Reset Sequence Counter (We attempted a jump, so a new "phrase" begins)
                 current_sequence = 0;
                 let ref_bar_pos = self.beats[cursor].bar_position;
                 min_sequence = *safe_amounts.choose(&mut rng).unwrap() as isize - (ref_bar_pos as isize + 2);
                 if min_sequence < 1 { min_sequence = 1; }
                 
                 // Panic Logic: If we haven't jumped in a long time, shorten the next phrase
                 // to try again sooner.
                 let remaining_panic = max_beats_between_jumps as isize - beats_since_jump as isize;
                 let safe_remaining = remaining_panic.max(1);
                 
                 if min_sequence > safe_remaining {
                     min_sequence = safe_remaining;
                 }
                 // Visual hack: force progress bar to full if panic
                 if beats_since_jump >= max_beats_between_jumps {
                      current_sequence = min_sequence as usize;
                 }

            } else {
                // Standard Playback (No Jump Attempt)
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
            
            // Record instruction
            play_vector.push(PlayInstruction {
                beat_id: cursor,
                seq_len: min_sequence as usize,
                seq_pos: current_sequence,
            });
        }
        play_vector
    }
}
