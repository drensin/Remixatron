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
    sound::static_sound::{StaticSoundData, StaticSoundSettings, StaticSoundHandle},
    clock::{ClockSpeed, ClockTime},
    StartTime,
    Tween,
};
use std::{thread, time::Duration};
use rand::prelude::*;
use rand::rngs::StdRng;
use std::collections::VecDeque;
use std::sync::mpsc::Receiver;

/// Commands to control the playback loop from another thread.
pub enum PlaybackCommand {
    /// Stop playback immediately and return from the loop.
    Stop,
    /// Pause the clock (freezes time and audio).
    Pause,
    /// Resume the clock.
    Resume,
}

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

    // JIT Playback State
    cursor: usize,
    current_sequence: usize,
    min_sequence_len: isize,
    beats_since_jump: usize,
    failed_jumps: usize,
    recent_segments: VecDeque<usize>,
    rng: StdRng,
    
    // Pre-calculated Metadata
    last_chance_beat: usize,
    acceptable_jump_amounts: Vec<usize>,
    max_beats_between_jumps: usize,
    
    /// RMS amplitude envelope for waveform visualization.
    /// Contains ~720 normalized values (0.0-1.0) representing amplitude at each angle.
    pub waveform_envelope: Vec<f32>,
}

impl JukeboxEngine {
    /// Creates a new engine instance.
    pub fn new(beats: Vec<Beat>, clusters: usize) -> Self {
        let mut rng = StdRng::from_entropy();
        
        // 1. Pre-calculate "Last Chance" beat
        let mut last_chance = if !beats.is_empty() { beats.len() - 1 } else { 0 };
        for i in (0..beats.len()).rev() {
             if !beats[i].jump_candidates.is_empty() {
                 last_chance = i;
                 break;
             }
        }

        // 2. Pre-calculate Jump Constants
        let last_beat_end = if let Some(b) = beats.last() { b.start + b.duration } else { 0.0 };
        let tempo = (beats.len() as f32 / last_beat_end) * 60.0;
        let mut max_sequence_len = ((tempo / 120.0) * 48.0).round() as usize;
        max_sequence_len = max_sequence_len - (max_sequence_len % 4);

        let base_amounts = vec![16, 24, 32, 48, 64, 72, 96, 128];
        let mut acceptable_jump_amounts: Vec<usize> = base_amounts.into_iter()
            .filter(|&x| x <= max_sequence_len)
            .collect();
            
        // 3/4 Time Adjustment
        let max_bar_pos = beats.iter().map(|b| b.bar_position).max().unwrap_or(4);
        if max_bar_pos + 1 == 3 {
             acceptable_jump_amounts = acceptable_jump_amounts.iter().map(|a| ((*a as f32) * 0.75) as usize).collect();
        }
        
        let safe_amounts = if acceptable_jump_amounts.is_empty() { vec![16] } else { acceptable_jump_amounts.clone() };
        let max_beats_between_jumps = (beats.len() as f32 * 0.1).round() as usize;

        // 3. Initialize First Phrase Length
        // Use beats[0] as ref, and offset 0 to target BarPos 3 (End of Bar)
        let start_bar_pos = if !beats.is_empty() { beats[0].bar_position } else { 0 };
        let mut min_seq = *safe_amounts.choose(&mut rng).unwrap() as isize - (start_bar_pos as isize);
        if min_seq < 1 { min_seq = 1; }

        Self {
            beats,
            _clusters: clusters,
            _branch_similarity_threshold: 0.0,
            audio_manager: None,
            sound_data: None,
            
            // State Init
            cursor: 0,
            current_sequence: 0,
            min_sequence_len: min_seq,
            beats_since_jump: 0,
            failed_jumps: 0,
            recent_segments: VecDeque::with_capacity(5), // Dynamic capacity
            rng,
            
            // Constants
            last_chance_beat: last_chance,
            acceptable_jump_amounts,
            max_beats_between_jumps,
            
            // Waveform envelope (computed in load_track)
            waveform_envelope: Vec::new(),
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
        
        // --- Compute Waveform Envelope for Visualization ---
        // We divide the audio into 720 chunks (2 per degree for smooth 360° ring).
        // For each chunk, compute RMS amplitude and normalize to 0.0-1.0.
        const ENVELOPE_SAMPLES: usize = 720;
        let total_samples = samples.len();
        let chunk_size = (total_samples / ENVELOPE_SAMPLES).max(1);
        
        let mut envelope: Vec<f32> = Vec::with_capacity(ENVELOPE_SAMPLES);
        let mut max_rms: f32 = 0.0;
        
        for i in 0..ENVELOPE_SAMPLES {
            let start = i * chunk_size;
            let end = ((i + 1) * chunk_size).min(total_samples);
            
            if start >= total_samples {
                envelope.push(0.0);
                continue;
            }
            
            // Calculate RMS for this chunk
            let chunk = &samples[start..end];
            let sum_sq: f32 = chunk.iter().map(|s| s * s).sum();
            let rms = (sum_sq / chunk.len() as f32).sqrt();
            
            if rms > max_rms {
                max_rms = rms;
            }
            envelope.push(rms);
        }
        
        // Normalize to 0.0-1.0 range
        if max_rms > 0.0 {
            for val in &mut envelope {
                *val /= max_rms;
            }
        }
        
        self.waveform_envelope = envelope;
        println!("Computed waveform envelope: {} samples", ENVELOPE_SAMPLES);
        // --- End Envelope Computation ---
            
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
    pub fn play(&mut self) -> Result<(), String> {
        let (_tx, rx) = std::sync::mpsc::channel();
        // Pass dummy segment index (0) since this play() is for verification only
        self.play_with_callback(rx, |_, _| {})
    }

    /// Starts playback with a callback for UI updates.
    /// Uses JIT (Just-In-Time) logic to decide the next beat in real-time.
    /// * `command_rx` - A channel receiver to listen for Stop/Pause commands.
    /// * `callback` - Closure called on every beat (instruction, segment_index).
    pub fn play_with_callback<F>(&mut self, command_rx: Receiver<PlaybackCommand>, mut callback: F) -> Result<(), String> 
    where F: FnMut(&PlayInstruction, usize) 
    {
        println!("Starting JIT Playback Loop...");

        // Ensure audio is loaded
        let mut manager = self.audio_manager.take()
            .ok_or("Audio Manager not initialized. Call load_track() first.")?;
        let sound_data = self.sound_data.take()
            .ok_or("Sound Data not loaded.")?;
            
        // Initial Play
        // let mut last_processed_cursor = usize::MAX; // Unused
        
        let mut clock = manager.add_clock(ClockSpeed::TicksPerSecond(1000.0))
            .map_err(|e| format!("Failed to create clock: {}", e))?;
        clock.start();
        
        // Track Clock Time locally for scheduling
        let now_ticks = clock.time().ticks;
        let mut cumulative_ticks = 0;
        // Reducing buffer from 4000ms to 200ms ensures "Pause" is instant.
        const BUFFER_MS: u64 = 200; 
        
        use std::collections::VecDeque;
        // Queue now stores: (fire_time, PlayInstruction)
        let mut pending_events: VecDeque<(u64, PlayInstruction)> = VecDeque::new();
        // Track active audio handles to explicitly pause them (Kira clock doesn't prevent "Runaway Beats")
        // Store (start_time, end_time, handle) to prune dead handles.
        let mut active_handles: VecDeque<(u64, u64, StaticSoundHandle)> = VecDeque::new();

        let mut paused = false;

        // Loop Indefinitely (until Stop command or error)
        loop {
            // 0. Check for Control Commands (Non-blocking)
            if let Ok(cmd) = command_rx.try_recv() {
                match cmd {
                    PlaybackCommand::Stop => {
                        println!("Playback Stopped by Command.");
                        drop(manager);
                        return Ok(());
                    },
                    PlaybackCommand::Pause => {
                        println!("CMD: Pause");
                        if !paused {
                            clock.pause();
                            let now = clock.time().ticks;
                            println!("State -> PAUSED. Clock frozen at: {}", now);
                            
                            // MANUAL STOP & PRUNE: Ensure finished sounds are actually stopped
                            let mut alive_handles = VecDeque::new();
                            while let Some((start, end, mut handle)) = active_handles.pop_front() {
                                if end <= now {
                                    // It should be done. Kill it to be sure.
                                    handle.stop(Tween::default());
                                } else {
                                    alive_handles.push_back((start, end, handle));
                                }
                            }
                            active_handles = alive_handles;

                            // Pause ONLY currently playing sounds
                            for (start_tick, end_tick, handle) in active_handles.iter_mut() {
                                if *start_tick <= now {
                                    println!("  [PAUSE] Handle {}-{} (now {}). Pausing.", start_tick, end_tick, now);
                                    handle.pause(Tween::default());
                                } else {
                                    println!("  [SKIP] Handle {}-{} > now {}. Future.", start_tick, end_tick, now);
                                }
                            }
                            paused = true;
                        }
                    },
                    PlaybackCommand::Resume => {
                        println!("CMD: Resume");
                        if paused {
                            clock.start();
                            let now = clock.time().ticks; 
                            println!("State -> RUNNING. Clock resumed at: {}", now);
                            
                            // MANUAL STOP & PRUNE
                            let mut alive_handles = VecDeque::new();
                            while let Some((start, end, mut handle)) = active_handles.pop_front() {
                                if end <= now {
                                    handle.stop(Tween::default());
                                } else {
                                    alive_handles.push_back((start, end, handle));
                                }
                            }
                            active_handles = alive_handles;

                            // Resume active sounds
                            for (start_tick, end_tick, handle) in active_handles.iter_mut() {
                                if *start_tick <= now {
                                    println!("  [RESUME] Handle {}-{} (now {}). Resuming.", start_tick, end_tick, now);
                                    handle.resume(Tween::default());
                                } else {
                                    println!("  [SKIP] Handle {}-{} > now {}. Future.", start_tick, end_tick, now);
                                }
                            }
                            paused = false;
                        }
                    }
                }
            }
        
            if paused {
                thread::sleep(Duration::from_millis(50));
                continue;
            }

            // 1. Throttling & Maintenance
            loop {
                let current_time = clock.time().ticks;
                
                // Active Maintenance: Stop expired handles periodically
                // This protects against Kira's scheduler failing during Pause cycles
                if !active_handles.is_empty() {
                    let mut i = 0;
                    while i < active_handles.len() {
                        let (_, end, _) = active_handles[i];
                        if end <= current_time {
                             // Stop and Remove
                             if let Some((_, _, mut handle)) = active_handles.remove(i) {
                                  // println!("  [AUTO-STOP] Enforcing stop at {}", current_time);
                                  handle.stop(Tween::default());
                             }
                        } else {
                             i += 1;
                        }
                    }
                }

                let elapsed = current_time.saturating_sub(now_ticks);
                
                // Service Events
                while let Some((fire_time, instruction)) = pending_events.front() {
                    if current_time >= *fire_time {
                        if instruction.beat_id < self.beats.len() {
                             let seg_id = self.beats[instruction.beat_id].segment;
                             callback(instruction, seg_id);
                        }
                        pending_events.pop_front();
                    } else {
                        break;
                    }
                }

                if cumulative_ticks < elapsed + BUFFER_MS {
                    break;
                }
                thread::sleep(Duration::from_millis(10));
            }

            // 2. JIT Decision: Get the next beat
            let instruction = self.get_next_beat();
            
            let beat = &self.beats[instruction.beat_id];
            let duration_ticks = (beat.duration * 1000.0) as u64;
            // Schedule relative to initial start + cumulative duration
            let scheduled_start_ticks = now_ticks + cumulative_ticks;
            
            // 3. Queue Notification
            pending_events.push_back((scheduled_start_ticks, instruction));
            
            // 4. Schedule Audio
            let start_time = ClockTime { clock: clock.id(), ticks: scheduled_start_ticks, fraction: 0.0 };
            let stop_time = ClockTime { clock: clock.id(), ticks: scheduled_start_ticks + duration_ticks, fraction: 0.0 };
            
            // println!("Scheduling Beat {} at {}", instruction.beat_id, scheduled_start_ticks);

            match manager.play(
                sound_data.with_settings(
                   StaticSoundSettings::new()
                       .start_time(StartTime::ClockTime(start_time))
                       .start_position(beat.start as f64)
                )
            ) {
                Ok(mut handle) => {
                     // We still ask Kira to schedule the stop as Plan A
                     handle.stop(
                        Tween {
                            start_time: StartTime::ClockTime(stop_time),
                            ..Default::default()
                        }
                    );
                    
                    // Track handle for Action & Manual Cleanup (Plan B)
                    active_handles.push_back((scheduled_start_ticks, scheduled_start_ticks + duration_ticks, handle));
                    // Note: No popping here, we let the Maintenance loop handle cleanup based on time.
                    // But we keep a safety max size just in case
                    if active_handles.len() > 10 {
                        active_handles.pop_front();
                    }
                },
                Err(e) => {
                    eprintln!("Failed to schedule sound: {}", e);
                }
            }

            cumulative_ticks += duration_ticks;
        }
    }

    /// Determines the next beat to play based on current state.
    ///
    /// This is the core "JIT Walk" logic that replaced the old `compute_play_vector`.
    /// It makes real-time decisions about whether to continue linearly or jump.
    pub fn get_next_beat(&mut self) -> PlayInstruction {
        // 1. Capture Current State (This is what we will return)
        let current_cursor = self.cursor;
        let current_beat = &self.beats[current_cursor];
        
        // 2. Update Recent Segments based on CURRENT beat
        // Ensure recent buffer is sized correctly
        let segments_count = self.beats.iter().map(|b| b.segment).max().unwrap_or(0) + 1;
        let recent_depth = (segments_count as f32 * 0.25).round() as usize;
        let recent_depth = recent_depth.max(1);
        
        if !self.recent_segments.contains(&current_beat.segment) {
            if self.recent_segments.len() >= recent_depth {
                self.recent_segments.pop_front();
            }
            self.recent_segments.push_back(current_beat.segment);
        }

        self.current_sequence += 1;
        
        // Capture State for Return (BEFORE any jump resets modifiers)
        let display_seq_len = self.min_sequence_len as usize;
        let display_seq_pos = self.current_sequence;

        // 3. Check Jump Trigger (Do we jump AFTER this beat?)
        // Note: logical offset is now handled in the min_seq calculation
        let will_jump = (self.current_sequence as isize >= self.min_sequence_len) || 
                        (self.beats_since_jump >= self.max_beats_between_jumps);
        
        let mut did_jump = false;
        let mut next_cursor = 0;

        if will_jump {
            // Strategy 1: Standard Jump
            let all_cands = &current_beat.jump_candidates;
            let non_recent: Vec<usize> = all_cands.iter()
                .filter(|&&c| !self.recent_segments.contains(&self.beats[c].segment))
                .cloned()
                .collect();

            if !non_recent.is_empty() {
                 next_cursor = *non_recent.choose(&mut self.rng).unwrap();
                 did_jump = true;
            } else {
                 // Strategy 2: Quartile Busting
                 let failure_threshold_10 = (self.beats.len() as f32 * 0.1) as usize;
                 let current_failed = self.failed_jumps + 1;
                 
                 let non_quartile: Vec<usize> = current_beat.jump_candidates.iter()
                    .filter(|&&c| self.beats[c].quartile != current_beat.quartile)
                    .cloned()
                    .collect();

                 if current_failed >= failure_threshold_10 && !non_quartile.is_empty() {
                      let furthest_dist = non_quartile.iter()
                          .map(|&c| (current_beat.id as isize - c as isize).abs())
                          .max().unwrap_or(0);
                      
                      if let Some(&target) = non_quartile.iter()
                          .find(|&&c| (current_beat.id as isize - c as isize).abs() == furthest_dist) {
                              next_cursor = target;
                              did_jump = true;
                      }
                 }
                 
                 // Strategy 3: Nuclear Reset
                 if !did_jump && current_failed >= (self.beats.len() as f32 * 0.3) as usize {
                      next_cursor = 0;
                      did_jump = true;
                 }
            }
        }

        // 4. Update Internal State for NEXT call
        if will_jump {
             if did_jump {
                 self.cursor = next_cursor;
                 self.beats_since_jump = 0;
                 self.failed_jumps = 0;
             } else {
                 self.failed_jumps += 1;
                 
                 // Play Next or Loop
                 if current_cursor == self.last_chance_beat {
                      if !current_beat.jump_candidates.is_empty() {
                          self.cursor = *current_beat.jump_candidates.iter().min().unwrap();
                      } else {
                          self.cursor = if current_cursor + 1 < self.beats.len() { current_cursor + 1 } else { 0 };
                      }
                 } else {
                      self.cursor = if current_cursor + 1 < self.beats.len() { current_cursor + 1 } else { 0 };
                 }
                 self.beats_since_jump += 1;
             }
             
             // Reset Sequence Target
             self.current_sequence = 0;
             let ref_bar_pos = self.beats[self.cursor].bar_position;
             let safe_amounts = if self.acceptable_jump_amounts.is_empty() { vec![16] } else { self.acceptable_jump_amounts.clone() };
             
             // Use new cursor for ref, and offset 0
             self.min_sequence_len = *safe_amounts.choose(&mut self.rng).unwrap() as isize - (ref_bar_pos as isize);
             if self.min_sequence_len < 1 { self.min_sequence_len = 1; }
             
             // Panic Logic
             let remaining_panic = self.max_beats_between_jumps as isize - self.beats_since_jump as isize;
             let safe_remaining = remaining_panic.max(1);
             
             if self.min_sequence_len > safe_remaining {
                 self.min_sequence_len = safe_remaining;
             }
             if self.beats_since_jump >= self.max_beats_between_jumps {
                  self.current_sequence = self.min_sequence_len as usize;
             }

        } else {
            // No Jump Attempt
             if current_cursor == self.last_chance_beat {
                  if !current_beat.jump_candidates.is_empty() {
                      self.cursor = *current_beat.jump_candidates.iter().min().unwrap();
                  } else {
                      self.cursor = if current_cursor + 1 < self.beats.len() { current_cursor + 1 } else { 0 };
                  }
             } else {
                  self.cursor = if current_cursor + 1 < self.beats.len() { current_cursor + 1 } else { 0 };
             }
             self.beats_since_jump += 1;
        }

        // 5. Return the CURRENT instruction (The one we started with)
        PlayInstruction {
            beat_id: current_cursor, // Return the beat we processed/played
            seq_len: display_seq_len,
            seq_pos: display_seq_pos,
        }
    }


}
