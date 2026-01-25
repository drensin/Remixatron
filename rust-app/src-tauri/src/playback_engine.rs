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
use crossbeam_channel::Sender;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// Minimum recency threshold for jump candidate selection (as a fraction of song length).
///
/// Candidates are filtered to only include beats that were played MORE THAN this
/// fraction of the song ago (lower position in play history = played longer ago).
///
/// - **Value: 0.25** means candidates must be in the oldest 75% of the play history
/// - **Example**: In a 500-beat song, won't jump to beats played in the last 125 beats
/// - **Bypassed in panic mode** (when stuck for 10%+ of song) to ensure forward progress
///
/// **Tuning Guide**:
/// - Lower (0.10): More permissive, allows some repetition
/// - Higher (0.50): Stricter, forces maximum variety but may feel too random
const MIN_RECENCY_THRESHOLD: f32 = 0.25;
/// Recency threshold used during Panic Mode.
///
/// When the engine is stuck (Panic), we lower the bar effectively allowing
/// repetition, but we still BAN extremely recent beats to prevent
/// immediate micro-loops.
///
/// **Value: 0.05** means valid candidates must be older than 5% of the song.
const PANIC_RECENCY_THRESHOLD: f32 = 0.05;

/// Commands to control the playback loop from another thread.
pub enum PlaybackCommand {
    /// Stop playback immediately and return from the loop.
    Stop,
    /// Pause the clock (freezes time and audio).
    Pause,
    /// Resume the clock.
    Resume,
    /// Set the master volume (0.0 to 1.0).
    SetVolume(f64),
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
    /// Normalized RMS energy (0.0-1.0) for Mood Shader brightness.
    pub energy: f32,
    /// Spectral centroid (0.0-1.0) for Mood Shader "sharpness".
    pub centroid: f32,
    /// Novelty score (0.0-1.0) for Mood Shader section transition effects.
    pub novelty: f32,
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
    /// FIFO queue tracking beat IDs in play order (most recent at back).
    /// Size = song length to enable recency-based jump selection.
    play_history: VecDeque<usize>,
    rng: StdRng,
    
    // Pre-calculated Metadata
    acceptable_jump_amounts: Vec<usize>,
    max_beats_between_jumps: usize,
    
    /// RMS amplitude envelope for waveform visualization.
    /// Contains ~720 normalized values (0.0-1.0) representing amplitude at each angle.
    pub waveform_envelope: Vec<f32>,

    /// Optional channel to broadcast raw audio frames (Stereo Interleaved F32).
    /// Used for synchronized streaming during playback.
    broadcast_tx: Option<Sender<Vec<f32>>>,

    /// Shared sample rate handle for the transcoder.
    /// Updated when a new track is loaded.
    broadcast_sample_rate: Option<Arc<AtomicU32>>,

    /// Raw decoded audio samples (Interleaved Stereo F32).
    /// Stored for beat-synchronized broadcast during playback.
    raw_samples: Vec<f32>,

    /// Sample rate of the loaded audio (e.g., 44100, 48000).
    /// Used by the transcoder to encode at the correct rate.
    audio_sample_rate: u32,

    /// Number of audio channels (typically 2 for stereo).
    audio_channels: u16,
}

impl JukeboxEngine {
    /// Creates a new engine instance.
    pub fn new(beats: Vec<Beat>, clusters: usize) -> Self {
        let mut rng = StdRng::from_entropy();
        
        // 1. Pre-calculate Jump Constants
        let last_beat_end = if let Some(b) = beats.last() { b.start + b.duration } else { 0.0 };
        let tempo = (beats.len() as f32 / last_beat_end) * 60.0;
        let mut max_sequence_len = ((tempo / 120.0) * 48.0).round() as usize;
        max_sequence_len = max_sequence_len - (max_sequence_len % 4);

        let base_amounts = vec![16, 24, 32, 48, 64, 72, 96, 128];
        let mut acceptable_jump_amounts: Vec<usize> = base_amounts.into_iter()
            .filter(|&x| x <= max_sequence_len)
            .collect();

        // 2. Calculate Dynamic Panic Threshold
        // 
        // Panic mode bypasses the recency filter when stuck playing linearly too long.
        // The threshold must scale with graph density:
        //
        // - SPARSE graphs (few jump candidates) → need MORE time for recency filter
        //   Example: avg=1.7 candidates → threshold = 81 beats (~14.6% of song)
        //
        // - DENSE graphs (many jump candidates) → baseline is fine
        //   Example: avg=2.5 candidates → threshold = 47 beats (10% of song)
        //
        // Formula: threshold = song_length × 0.10 × (2.5 / avg_candidates)
        //          capped at 20% of song length
        //
        // The constant 2.5 comes from empirical testing:
        // - A dense graph with avg=2.5 had 0% panic mode triggers
        // - This suggests avg=2.5 provides enough options for recency-based selection
        // - Sparser graphs scale UP from this baseline
        let song_length = beats.len();
        let total_candidates: usize = beats.iter().map(|b| b.jump_candidates.len()).sum();
        let avg_candidates = if song_length > 0 {
            total_candidates as f32 / song_length as f32
        } else {
            1.0
        };
        
        let baseline_threshold = song_length as f32 * 0.10;  // 10% baseline
        let sparsity_adjusted = baseline_threshold * (2.5 / avg_candidates);
        let max_threshold = song_length as f32 / 5.0;  // Cap at 20%
        let max_beats_between_jumps = sparsity_adjusted.min(max_threshold) as usize;

        // 3. Adjust for 3/4 time signature
        let max_bar_pos = beats.iter().map(|b| b.bar_position).max().unwrap_or(4);
        if max_bar_pos + 1 == 3 {
             acceptable_jump_amounts = acceptable_jump_amounts.iter().map(|a| ((*a as f32) * 0.75) as usize).collect();
        }
        
        let safe_amounts = if acceptable_jump_amounts.is_empty() { vec![16] } else { acceptable_jump_amounts.clone() };

        // 4. Initialize First Phrase Length
        // Use beats[0] as ref, and offset 0 to target BarPos 3 (End of Bar)
        let start_bar_pos = if !beats.is_empty() { beats[0].bar_position } else { 0 };
        let mut min_seq = *safe_amounts.choose(&mut rng).unwrap() as isize - (start_bar_pos as isize);
        if min_seq < 1 { min_seq = 1; }

        // 4. Initialize play history queue with capacity = song length
        let song_length = beats.len();

        // DEBUG LOGGING: Jump Graph Structure Analysis
        {
            use std::io::Write;
            use std::fs::OpenOptions;
            if let Ok(mut file) = OpenOptions::new()
                .create(true)
                .append(true)
                .open("remixatron_debug.log") 
            {
                let _ = writeln!(file, "\n\n=== JUMP GRAPH ANALYSIS ===");
                let _ = writeln!(file, "Song length: {} beats", song_length);
                let _ = writeln!(file, "Panic threshold: {} beats (10% of song)", max_beats_between_jumps);
                
                // Candidate count statistics
                let candidate_counts: Vec<usize> = beats.iter()
                    .map(|b| b.jump_candidates.len())
                    .collect();
                
                let total_candidates: usize = candidate_counts.iter().sum();
                let avg_candidates = if !candidate_counts.is_empty() {
                    total_candidates as f32 / candidate_counts.len() as f32
                } else {
                    0.0
                };
                let min_candidates = candidate_counts.iter().min().copied().unwrap_or(0);
                let max_candidates = candidate_counts.iter().max().copied().unwrap_or(0);
                
                // Count beats by candidate count buckets
                let zero_cands = candidate_counts.iter().filter(|&&c| c == 0).count();
                let one_cand = candidate_counts.iter().filter(|&&c| c == 1).count();
                let two_to_five = candidate_counts.iter().filter(|&&c| (2..=5).contains(&c)).count();
                let six_to_ten = candidate_counts.iter().filter(|&&c| (6..=10).contains(&c)).count();
                let over_ten = candidate_counts.iter().filter(|&&c| c > 10).count();
                
                let _ = writeln!(file, "\nCandidate Count Statistics:");
                let _ = writeln!(file, "  Total edges: {}", total_candidates);
                let _ = writeln!(file, "  Average candidates per beat: {:.1}", avg_candidates);
                let _ = writeln!(file, "  Min/Max candidates: {} / {}", min_candidates, max_candidates);
                let _ = writeln!(file, "\nBeats by candidate count:");
                let _ = writeln!(file, "  0 candidates: {} beats ({:.1}%)", zero_cands, 100.0 * zero_cands as f32 / song_length as f32);
                let _ = writeln!(file, "  1 candidate:  {} beats ({:.1}%)", one_cand, 100.0 * one_cand as f32 / song_length as f32);
                let _ = writeln!(file, "  2-5 candidates: {} beats ({:.1}%)", two_to_five, 100.0 * two_to_five as f32 / song_length as f32);
                let _ = writeln!(file, "  6-10 candidates: {} beats ({:.1}%)", six_to_ten, 100.0 * six_to_ten as f32 / song_length as f32);
                let _ = writeln!(file, "  >10 candidates: {} beats ({:.1}%)", over_ten, 100.0 * over_ten as f32 / song_length as f32);
                
                // Full graph structure (first 50 beats + any with <=2 candidates)
                let _ = writeln!(file, "\nJump Graph Structure (first 50 beats + sparse beats):");
                for (i, beat) in beats.iter().enumerate() {
                    if i < 50 || beat.jump_candidates.len() <= 2 {
                        let cands_str = if beat.jump_candidates.len() <= 10 {
                            format!("{:?}", beat.jump_candidates)
                        } else {
                            format!("[{}... ({} total)]", 
                                beat.jump_candidates.iter().take(5)
                                    .map(|c| c.to_string())
                                    .collect::<Vec<_>>()
                                    .join(", "),
                                beat.jump_candidates.len())
                        };
                        let _ = writeln!(file, "  Beat {}: {} candidates {}", 
                            i, beat.jump_candidates.len(), cands_str);
                    }
                }
                
                let _ = writeln!(file, "=== END JUMP GRAPH ANALYSIS ===\n");
            }
        }

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
            play_history: VecDeque::with_capacity(song_length),
            rng,
            
            // Constants
            acceptable_jump_amounts,
            max_beats_between_jumps,
            
            // Waveform envelope (computed in load_track)
            waveform_envelope: Vec::new(),
            
            // Broadcasting
            broadcast_tx: None,
            broadcast_sample_rate: None,
            
            // Raw audio storage (populated in load_track)
            raw_samples: Vec::new(),
            audio_sample_rate: 0,
            audio_channels: 0,
        }
    }

    /// Enables broadcasting mode.
    ///
    /// This method configures the engine to send audio samples to the broadcast
    /// transcoder during playback. Must be called BEFORE `load_track`.
    ///
    /// # Arguments
    ///
    /// * `tx` - Channel sender for PCM audio samples.
    /// * `sample_rate_handle` - Shared atomic for dynamic sample rate updates.
    pub fn enable_broadcasting(
        &mut self, 
        tx: Sender<Vec<f32>>,
        sample_rate_handle: Arc<AtomicU32>,
    ) {
        self.broadcast_tx = Some(tx);
        self.broadcast_sample_rate = Some(sample_rate_handle);
        println!("Broadcasting enabled for JukeboxEngine.");
    }
    
    /// Loads and decodes the audio file into memory.
    ///
    /// This method performs several critical tasks:
    /// 1. Decodes audio using a robust multi-format decoder (rodio-based).
    /// 2. Computes the waveform envelope for visualization.
    /// 3. Optionally sends decoded samples to the broadcast transcoder (Pre-Kira tap).
    /// 4. Converts samples to Kira `Frame` format for playback.
    ///
    /// # Arguments
    ///
    /// * `path` - Absolute path to the audio file (MP3, WAV, FLAC, etc.).
    ///
    /// # Errors
    ///
    /// Returns an error if audio decoding fails or the audio manager cannot be initialized.
    ///
    /// # Broadcasting (Pre-Kira Tap)
    ///
    /// If `enable_broadcasting()` was called before this method, the decoded PCM samples
    /// are sent to the transcoder channel *before* being converted to Kira frames. This
    /// "Pre-Kira" approach decouples the broadcast from Kira's internal mixer, improving
    /// stability and maintainability. The transcoder receives the full decoded audio
    /// upfront, then streams it in sync with playback.
    pub fn load_track(&mut self, path: &str) -> Result<(), String> {
        use crate::audio_backend::decoder::decode_audio_file;
        use kira::Frame;
        use std::sync::Arc;

        // Initialize the Kira audio manager.
        let manager = AudioManager::new(AudioManagerSettings::default())
            .map_err(|e| format!("Failed to initialize audio manager: {}", e))?;

        // Decode the audio file to raw PCM samples.
        println!("Decoding audio with robust decoder...");
        let (samples, sample_rate, channels) = decode_audio_file(path)
            .map_err(|e| format!("Robust decode failed: {}", e))?;

        // --- STORE RAW AUDIO FOR BROADCAST ---
        // Instead of sending samples now (which would stream the linear song),
        // we store them for beat-synchronized broadcast during playback.
        // This ensures the broadcast matches the remixed/jumped audio.
        self.raw_samples = samples.clone();
        self.audio_sample_rate = sample_rate;
        self.audio_channels = channels as u16;

        // Signal the transcoder about the actual sample rate.
        if let Some(rate_handle) = &self.broadcast_sample_rate {
            rate_handle.store(sample_rate, Ordering::Relaxed);
            println!("[Broadcast] Updated transcoder sample rate to {}Hz.", sample_rate);
        }

        println!(
            "[Broadcast] Stored {} samples ({}Hz, {} channels) for playback sync.", 
            samples.len(), sample_rate, channels
        );
        // --- END STORE RAW AUDIO ---

        // --- Compute Waveform Envelope for Visualization ---
        // We divide the audio into 720 chunks (2 per degree for smooth 360° ring).
        // For each chunk, compute RMS amplitude and normalize to 0.0-1.0.
        // RESPECT TRUNCATION: Only compute envelope up to the last valid beat.
        let effective_limit = if let Some(last_beat) = self.beats.last() {
            let limit_sec = last_beat.start + last_beat.duration;
            let limit_samples = (limit_sec * sample_rate as f32) as usize * channels as usize;
            limit_samples.min(samples.len())
        } else {
            samples.len()
        };

        const ENVELOPE_SAMPLES: usize = 720;
        let total_samples = effective_limit; // Use truncated length
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

        // Convert Vec<f32> interleaved to Vec<Frame> for Kira.
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

        // Create the static sound data with default settings.
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
        let mut current_volume = 1.0; // Default: Full Volume

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
                    },
                    PlaybackCommand::SetVolume(vol) => {
                        println!("CMD: SetVolume({:.2})", vol);
                        current_volume = vol;
                        
                        // Apply to ALL active handles immediately
                        let db = if current_volume <= 0.001 { -80.0 } else { 20.0 * current_volume.log10() };
                        for (_, _, handle) in active_handles.iter_mut() {
                            handle.set_volume(db as f32, Tween::default());
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

            // 3.5. BROADCAST: Send this beat's audio to the transcoder
            // This ensures the broadcast matches the remixed audio, not the linear file.
            if let Some(tx) = &self.broadcast_tx {
                // Calculate sample range for this beat.
                // samples are interleaved stereo, so multiply by channels.
                let channels = self.audio_channels as usize;
                let sample_rate = self.audio_sample_rate as f32;
                let start_sample = (beat.start * sample_rate) as usize * channels;
                let end_sample = ((beat.start + beat.duration) * sample_rate) as usize * channels;
                let end_sample = end_sample.min(self.raw_samples.len());

                if start_sample < self.raw_samples.len() {
                    let beat_samples = &self.raw_samples[start_sample..end_sample];
                    // Non-blocking send: dropped messages are acceptable for latency.
                    let _ = tx.try_send(beat_samples.to_vec());
                }
            }

            // 4. Schedule Audio
            let start_time = ClockTime { clock: clock.id(), ticks: scheduled_start_ticks, fraction: 0.0 };
            let stop_time = ClockTime { clock: clock.id(), ticks: scheduled_start_ticks + duration_ticks, fraction: 0.0 };

            // println!("Scheduling Beat {} at {}", instruction.beat_id, scheduled_start_ticks);

            match manager.play(
                sound_data.with_settings(
                   StaticSoundSettings::new()
                       .start_time(StartTime::ClockTime(start_time))
                       .start_position(beat.start as f64)
                       .volume({
                           if current_volume <= 0.001 { -80.0 } else { 20.0 * current_volume.log10() }
                       } as f32)
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
    /// Uses recency-based scoring: candidates are scored by their position in the
    /// play history queue (oldest = highest score). Avoids long local loops by
    /// preferring beats that haven't been played recently.
    pub fn get_next_beat(&mut self) -> PlayInstruction {
        // 1. Capture Current State (This is what we will return)
        let current_cursor = self.cursor;
        let current_beat = &self.beats[current_cursor];
        
        // 2. Add current beat to play history (FIFO - most recent at back)
        self.play_history.push_back(current_cursor);
        
        // Keep queue size = song length
        let song_length = self.beats.len();
        while self.play_history.len() > song_length {
            self.play_history.pop_front();
        }

        self.current_sequence += 1;
        
        // Capture State for Return (BEFORE any jump resets modifiers)
        let display_seq_len = self.min_sequence_len as usize;
        let display_seq_pos = self.current_sequence;

        // 3. Check Jump Trigger (Do we jump AFTER this beat?)
        // Standard Trigger: Phrase complete OR Panic mode
        let will_jump = (self.current_sequence as isize >= self.min_sequence_len) || 
                        (self.beats_since_jump >= self.max_beats_between_jumps);
        
        let mut did_jump = false;
        let mut next_cursor = 0;

        if will_jump {
            // Recency-Based Jump Selection
            let all_cands = &current_beat.jump_candidates;
            
            if !all_cands.is_empty() {
                // Score each candidate by recency
                let mut scored_candidates: Vec<(usize, usize, usize)> = Vec::new(); // (beat_id, recency_score, distance)
                
                for &candidate_id in all_cands {
                    // Find MOST RECENT position in play history (rposition searches from back)
                    let recency_score = if let Some(pos) = self.play_history.iter().rposition(|&id| id == candidate_id) {
                        // "How many beats ago was this?"
                        // pos is 0-indexed from start. history.len()-1 is the current beat.
                        // Example: Hist=[A,B,C] (len=3). Find B (pos=1). 
                        // Beats ago = 3 - 1 - 1 = 1.
                        self.play_history.len() - 1 - pos
                    } else {
                        song_length  // Never played = maximum score
                    };
                    
                    let distance = (current_cursor as isize - candidate_id as isize).unsigned_abs();
                    scored_candidates.push((candidate_id, recency_score, distance));
                }
                
                // Apply recency threshold filter using MIN_RECENCY_THRESHOLD constant
                let in_panic_mode = self.beats_since_jump >= self.max_beats_between_jumps;
                
                // STANDARD: 25% of song. PANIC: 5% of song (Soft Panic).
                let threshold_fraction = if in_panic_mode { PANIC_RECENCY_THRESHOLD } else { MIN_RECENCY_THRESHOLD };
                let min_recency_score = (song_length as f32 * threshold_fraction) as usize;
                
                let viable_candidates: Vec<(usize, usize, usize)> = scored_candidates
                    .into_iter()
                    .filter(|(_, score, _)| {
                        // Soft Panic: We enforce the threshold even in panic mode, just a lower one.
                        *score > min_recency_score
                    })
                    .collect();
                
                // Sort by recency_score (descending), then by distance (descending) as tiebreaker
                let mut sorted_candidates = viable_candidates;
                sorted_candidates.sort_by(|a, b| {
                    match b.1.cmp(&a.1) {  // b.1 vs a.1 for descending recency
                        std::cmp::Ordering::Equal => b.2.cmp(&a.2),  // b.2 vs a.2 for descending distance
                        other => other,
                    }
                });
                
                // Pick the best candidate
                if let Some(&(best_candidate, score, dist)) = sorted_candidates.first() {
                    next_cursor = best_candidate;
                    did_jump = true;

                    // --- TELEMETRY LOGGING (SURGICAL) ---
                    // Log the jump decision to diagnose "backward jump" behavior
                    {
                        use std::fs::OpenOptions;
                        use std::io::Write;
                        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open("remixatron_debug.log") {
                            let trigger = if in_panic_mode { "PANIC" } else { "PHRASE" };
                            let _ = writeln!(file, "[Beat {}] Trigger: {} ({}/{}). Candidates: Raw {} -> Viable {}. Selected: {} (Score: {}, Dist: {})",
                                current_cursor,
                                trigger,
                                self.beats_since_jump,
                                self.max_beats_between_jumps,
                                all_cands.len(),
                                sorted_candidates.len(),
                                best_candidate,
                                score,
                                dist
                            );
                            
                            // If we utilized the lower panic threshold, log it
                            if in_panic_mode && score <= (song_length as f32 * MIN_RECENCY_THRESHOLD) as usize {
                                let _ = writeln!(file, "  -> SOFT PANIC: Used Lower Recency Threshold (Score {} > {})", score, min_recency_score);
                            }
                        }
                    }
                    // --- END TELEMETRY ---
                }
            }
        }

        // 4. Update Internal State for NEXT call
        if will_jump {
             if did_jump {
                 self.cursor = next_cursor;
                 self.beats_since_jump = 0;
             } else {
                 // No candidates available - continue linearly
                 if current_cursor + 1 < self.beats.len() {
                     self.cursor = current_cursor + 1;
                 } else {
                     // WRAP AROUND (Smart Loop)
                     // Instead of jumping to 0 (which might be a silent intro),
                     // jump to the first beat that has any connectivity.
                     // This likely skips the unique/slow intro on the loop.
                     let smart_start = self.beats.iter()
                         .position(|b| !b.jump_candidates.is_empty())
                         .unwrap_or(0);
                     
                     println!("[SmartLoop] Wrapped! Found start index: {} (Beat 0 cands: {})", 
                        smart_start, 
                        self.beats.first().map(|b| b.jump_candidates.len()).unwrap_or(0)
                     );

                     self.cursor = smart_start;
                     // Reset panic counter on full loop
                     self.beats_since_jump = 0; 
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
            self.cursor = if current_cursor + 1 < self.beats.len() { current_cursor + 1 } else { 0 };
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
