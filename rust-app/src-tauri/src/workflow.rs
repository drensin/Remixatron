//! # Workflow Pipeline
//!
//! This module orchestrates the end-to-end analysis process. It acts as the "Director",
//! coordinating the sub-systems (Audio Loading, ML Inference, Structural Analysis)
//! to produce a playable `AnalysisResult`.
//!
//! ## Pipeline Stages
//! 1.  **Decode**: Audio is loaded and downmixed to mono.
//! 2.  **Resample**: Audio is converted to 22050Hz (required by models).
//! 3.  **Mel Spectrogram**: Computes the implementation-specific spectrogram for inputs.
//! 4.  **Beat Tracking**: Runs `BeatThis` ONNX model to find beat/downbeat instances.
//! 5.  **Feature Extraction**: synchronized MFCC/Chroma features are extracted per beat.
//! 6.  **Structural Analysis**: Hybrid novelty+recurrence segmentation (via `structure.rs`).
//! 7.  **Graph Construction**: The finalized `Beat` structs are assembled with connectivity data.

use anyhow::{Result, anyhow};

use crate::audio::loader::load_audio;
use crate::beat_tracker::mel::MelProcessor;
use crate::beat_tracker::inference::BeatProcessor;
use crate::beat_tracker::post_processor::MinimalPostProcessor;
use crate::analysis::features::FeatureExtractor;
use crate::analysis::structure::StructureAnalyzer;
use ndarray::{Axis, s};

use crate::playback_engine::Beat;

/// The complete, serializable result of the analysis pipeline.
///
/// This struct contains everything the frontend needs to render the visualization
/// and everything the backend needs to play the infinite mix.
#[derive(serde::Serialize)]
pub struct AnalysisResult {
    /// Total duration of the track in seconds.
    pub duration_seconds: f32,
    /// Timestamps of every detected beat.
    pub beats: Vec<f32>,
    /// Timestamps of every detected downbeat (bar start).
    pub downbeats: Vec<f32>,
    /// Rich beat objects containing clustering and jump data.
    pub beat_structs: Vec<Beat>,
    /// High-level structural segments (e.g., "Chorus 1").
    pub segments: Vec<Segment>,
    /// The number of clusters (K) chosen by the algorithm.
    pub k_optimal: usize,
    /// The novelty curve used for debugging segmentation.
    pub novelty_curve: Vec<f32>,
    /// Indices of the major structural boundaries.
    pub peaks: Vec<usize>,
    /// Optional error message if the pipeline failed gracefully.
    pub error: Option<String>,
}

/// A contiguous block of music belonging to a single structural cluster.
///
/// Used primarily for UI visualization (drawing colored arcs).
#[derive(serde::Serialize, Clone)]
pub struct Segment {
    pub start_time: f32,
    pub end_time: f32,
    /// The cluster ID (0..K) this segment belongs to.
    pub label: usize,
}

/// The main pipeline coordinator.
pub struct Remixatron {
    mel_path: String,
    beat_path: String,
}

impl Remixatron {
    /// Creates a new Pipeline instance with paths to the required ONNX models.
    pub fn new(mel_path: &str, beat_path: &str) -> Self {
        Self {
            mel_path: mel_path.to_string(),
            beat_path: beat_path.to_string(),
        }
    }

    /// Executes the full analysis pipeline on the given audio file.
    ///
    /// # Arguments
    /// * `audio_path`: Path to the audio file.
    /// * `progress_callback`: A closure that accepts a status string `&str`.
    ///
    /// # Errors
    /// Returns an error if:
    /// * File cannot be opened or decoded.
    /// * ONNX models fail to load or run.
    /// * Audio is too short or silent.
    pub fn analyze<F>(&self, audio_path: &str, progress_callback: F) -> Result<AnalysisResult> 
    where F: Fn(&str, f32)
    {
        // 1. Load & Resample Audio
        progress_callback("Decoding & Resampling Audio...", 0.10);
        // We use the High-Quality Loader (Rubato) to get mono audio at 22050 Hz.
        // This ensures maximum spectral accuracy for the downstream feature extractors.
        let target_sr = 22050;
        let audio_data = load_audio(audio_path, target_sr).map_err(|e| anyhow!(e.to_string()))?;
        
        let audio = audio_data.signal;
        let sr = audio_data.sample_rate; // Should be 22050
        
        let duration_seconds = audio.len() as f32 / sr as f32;
        
        // 2. Mel Spectrogram
        progress_callback("Generating Mel Spectrogram...", 0.30);
        // Compute the input tensor for the Beat Tracker.
        let mut mel_proc = MelProcessor::new(&self.mel_path)?;
        let mel = mel_proc.process(&audio)?;
        
        // 3. Beat Tracking
        progress_callback("Tracking Beats (AI Inference)...", 0.60);
        // Run inference to get beat/downbeat activation logits.
        let mut tracker = BeatProcessor::new(&self.beat_path)?;
        let (b_logits, d_logits) = tracker.process(&mel)?;
        
        // Post-process logits into discrete timestamps (FPS=50).
        let post = MinimalPostProcessor::new(50.0);
        let (mut beats, mut downbeats) = post.process(&b_logits, &d_logits)?;
        
        // 4. Feature Extraction & Structure
        progress_callback("Extracting Timbre & Harmony...", 0.80);
        // Extract Timbre (MFCC) and Harmony (Chroma) for every beat.
        let mut feature_ex = FeatureExtractor::new(128, target_sr as f32);
        let mel_2d = mel.index_axis(Axis(0), 0).to_owned();
        
        let mut beats_extended = beats.clone();
        beats_extended.push(duration_seconds); // Add end marker to define the last beat's duration
        
        let (mut mfcc, mut chroma, mut _cqt, mut rms_vec) = feature_ex.compute_sync_features(&audio, &mel_2d, &beats_extended, 50.0);
        
        // Remove the feature vector corresponding to the end marker
        // because it has 0 duration and is not a playble unit.
        if mfcc.nrows() > 0 {
             mfcc = mfcc.slice(s![0..mfcc.nrows()-1, ..]).to_owned();
             chroma = chroma.slice(s![0..chroma.nrows()-1, ..]).to_owned();
             _cqt = _cqt.slice(s![0.._cqt.nrows()-1, ..]).to_owned();
             if !rms_vec.is_empty() { rms_vec.pop(); }
        }

        // --- FADE TRUNCATION (Pre-Computation) ---
        // We detect fade-outs based on RMS amplitude and physically remove those beats
        // so the graph never sees them.
        
        // Skip for short clips (< 60s) to avoid damaging ringtones/samples
        if duration_seconds >= 60.0 {
            // 1. Calculate Median RMS
            let mut sorted_rms = rms_vec.clone();
            sorted_rms.sort_by(|a: &f32, b: &f32| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median_rms = if sorted_rms.is_empty() { 0.0 } else { sorted_rms[sorted_rms.len() / 2] };
            
            // 2. Determine Thresholds
            // Relative: 0.4 * Median (~ -8dB)
            // Absolute: 0.05 (Noise Floor)
            let relative_threshold: f32 = 0.4 * median_rms;
            let absolute_limit: f32 = 0.05;
            let fade_threshold = relative_threshold.max(absolute_limit);
            
            // 3. Scan Backwards
            let mut cutoff_index = beats.len(); // Default: keep all
            for i in (0..beats.len()).rev() {
                if rms_vec[i] > fade_threshold {
                    cutoff_index = i + 1; // Keep this beat, cut after
                    break;
                }
            }
            
            // 4. Truncate (if found and safe)
            // Safety: Ensure we don't delete *too* much (keep at least 32 beats)
            let min_safe_beats = 32;
            if cutoff_index < beats.len() && cutoff_index >= min_safe_beats {
                let trimmed_count = beats.len() - cutoff_index;
                let trimmed_duration = beats[beats.len() - 1] - beats[cutoff_index-1]; // Approx
                println!("[Fade Detection] Truncating {} beats ({:.2}s). Threshold: {:.4} (Med: {:.4})", 
                    trimmed_count, trimmed_duration, fade_threshold, median_rms);
                
                // Perform Truncation
                beats.truncate(cutoff_index);
                beats_extended.truncate(cutoff_index + 1); // +1 for end marker
                mfcc = mfcc.slice(s![0..cutoff_index, ..]).to_owned();
                chroma = chroma.slice(s![0..cutoff_index, ..]).to_owned();
                _cqt = _cqt.slice(s![0..cutoff_index, ..]).to_owned();
                rms_vec.truncate(cutoff_index);
                
                // Update Duration (Crucial for UI sync)
                // Use the End Time of the last valid beat
                 // beats_extended[cutoff_index] is logically the end time of beats[cutoff_index-1]
                let new_duration = beats_extended[cutoff_index];
                // duration_seconds is immutable in scope? No, it's a let binding.
                // Shadowing likely required or re-assignment if mutable.
                // Wait, duration_seconds is 'let' above (line 104). We need to shadow it? 
                // Wait, logic lower down uses `duration_seconds` for AnalysisResult.
                // I will shadow it here for the rest of the function? 
                // Rust Scoping: I cannot modify `duration_seconds` unless it was mut.
                // It was not declared mut. I should declare a new variable or I might need to make it mut generally.
                // Easier: `let duration_seconds = ...` shadows it for subsequent code in this block? 
                // No, the scope ends.
                // I need to change line 104 to `let mut duration_seconds`.
                // OR I can shadow it at module level? 
                // Let's assume I can't easily change line 104 without a huge diff.
                // I'll shadow it *outside* this block? No.
                
                // Prune Downbeats
                downbeats.retain(|&t| t < new_duration);
            }
        }
        
        // Re-declare duration_seconds to match beats (even if not truncated) for safety
        // beats_extended currently has `beats.len() + 1` elements.
        let duration_seconds = beats_extended.last().copied().unwrap_or(duration_seconds);
        
        // 5. Pre-compute Bar Positions
        // Maps each beat to its position within a bar (typically 0-3 for 4/4).
        // Logic:
        // - Resets to 0 whenever a strict Downbeat is detected (handling 3/4, 2/4, etc).
        // - Defaults to modulo-4 counting if downbeats are sparse or missing.
        // - LIMITATION: Hard-coded modulo 4 means 5/4 time will wrap incorrectly if downbeats are missed.
        let mut bar_positions = Vec::with_capacity(beats_extended.len() - 1);
        let mut bar_pos_counter = 0;
        
        for i in 0..mfcc.nrows() {
             let start_time = beats_extended[i];
             // Simple proximity check: is this beat within 50ms of a known downbeat?
             let is_downbeat = downbeats.iter().any(|&d| (d - start_time).abs() < 0.05);
             if is_downbeat {
                 bar_pos_counter = 0;
             }
             bar_positions.push(bar_pos_counter);
             bar_pos_counter = (bar_pos_counter + 1) % 4;
        }
        
        // 6. Structural Analysis (The Core Logic)
        progress_callback("Clustering Structure...", 0.95);
        // Perform Checkerboard Novelty Segmentation on the beat-synchronous features.
        // Uses novelty peaks to find boundaries, then clusters segments between boundaries.
        let analyzer = StructureAnalyzer::new();
        let result = analyzer.compute_segments_checkerboard(&mfcc, &chroma, &bar_positions, None); 
        
        // 7. Assembly
        // Convert the raw labels and jump graph into the final `Beat` and `Segment` structs.
        let mut beat_structs = Vec::with_capacity(result.labels.len());
        let mut segments = Vec::new();
        
        if !result.labels.is_empty() {
            let mut current_segment_id = 0;
            let mut intra_segment_index = 0;
            let mut segment_start_idx = 0;
            let mut current_label = result.labels[0];

            // 0. Pre-calculate Segment IDs for all beats
            // We need this map to filter "Intra-Segment" jumps efficiently in the main loop.
            // (i.e. we need to know the segment ID of a *future* target beat).
            let mut beat_to_segment_id = vec![0; result.labels.len()];
            {
                let mut temp_seg_id = 0;
                let mut temp_label = result.labels[0];
                for i in 0..result.labels.len() {
                    if result.labels[i] != temp_label {
                        temp_seg_id += 1;
                        temp_label = result.labels[i];
                    }
                    beat_to_segment_id[i] = temp_seg_id;
                }
            }

            for i in 0..result.labels.len() {
                let start_time = beats_extended[i];
                let label = result.labels[i];
                
                // Segment Change Detection
                if label != current_label {
                    // Finalize previous segment
                    segments.push(Segment {
                        start_time: beats_extended[segment_start_idx],
                        end_time: start_time,
                        label: current_label,
                    });
                    
                    current_label = label;
                    current_segment_id += 1;
                    intra_segment_index = 0;
                    segment_start_idx = i;
                }
                
                let duration = beats_extended[i+1] - start_time;
                
                // Retrieve Jumps for this beat
                // SOTA Logic: "Look Ahead"
                // To avoid stutters, we look up the neighbors of the NEXT beat (i+1).
                // If we are at beat X, and we want to jump, we need to land on a beat Y
                // such that Y is similar to X+1. This preserves the musical flow.
                let next_beat_idx = if i + 1 < result.labels.len() { i + 1 } else { 0 };
                
                let raw_candidates = if next_beat_idx < result.jumps.len() {
                    &result.jumps[next_beat_idx]
                } else {
                    &Vec::new()
                };

                // FEATURE: Phase-Locked Jumps
                // Logic:
                // 1. Phase Consistency: We check the Bar Position (0-3) of the *next* beat.
                //    Any jump target must have the SAME Bar Position.
                // 2. Safety: No dead ends (avoid jumping to last beat).
                //
                // NOTE: Segment diversity check removed - recency-based scoring in playback
                // engine handles micro-loop prevention more effectively.
                let mut candidates = Vec::new();

                // Determine the "Target Phase" (The Bar Position of the beat we are about to play)
                let target_phase = if next_beat_idx < bar_positions.len() { bar_positions[next_beat_idx] } else { 0 };

                for target_idx in raw_candidates {
                    // Check bounds just in case
                    if *target_idx < beat_to_segment_id.len() {
                        
                        // CHECK 1: Phase Consistency
                        let candidate_phase = bar_positions[*target_idx];
                        if candidate_phase != target_phase { continue; }

                        // CHECK 2: Minimum Distance
                        // Prevent micro-jumps that feel jarring (e.g., jumping just 2-3 beats away).
                        // Require at least 8 beats (2 bars) between source and target.
                        let distance = (*target_idx as isize - i as isize).unsigned_abs();
                        const MIN_JUMP_DISTANCE: usize = 8;
                        if distance < MIN_JUMP_DISTANCE { continue; }

                        candidates.push(*target_idx);
                    }
                }

                beat_structs.push(Beat {
                    id: i,
                    start: start_time,
                    duration,
                    bar_position: bar_positions[i],
                    cluster: label,
                    segment: current_segment_id,
                    intra_segment_index,
                    quartile: 0, // Calculated dynamically in playback_engine
                    jump_candidates: candidates,
                });
                
                intra_segment_index += 1;
            }
            
            // Push final Tail Segment
            segments.push(Segment {
                start_time: beats_extended[segment_start_idx],
                end_time: beats_extended[result.labels.len()], // Use result labels len in case of mismatch? No, should be sync.
                label: current_label,
            });
        }

        // --- POST-GRAPH PRUNING (Dead End Removal) ---
        // Remove beats from the end that have NO valid jump candidates.
        // This ensures the "Terminal Beat" is always a valid exit node.
        if !beat_structs.is_empty() {
             let mut pruned_count = 0;
             while let Some(last_beat) = beat_structs.last() {
                 if last_beat.jump_candidates.is_empty() {
                     // Safety limit: Don't delete entire song
                     if beat_structs.len() <= 32 { break; }
                     beat_structs.pop();
                     
                     // Also pop from scalar vectors to keep sync
                     beats.pop();
                     // We don't need to resize mfcc/chroma/segments because they aren't used downstream except for AnalysisResult
                     // But AnalysisResult expects `beats` and `beat_structs` to match.
                     pruned_count += 1;
                 } else {
                     break; // Found a beat with candidates
                 }
             }
             if pruned_count > 0 {
                 println!("[Graph Pruning] Removed {} dead-end beats from tail.", pruned_count);
                 
                 // Update duration again
                 if let Some(_last_beat) = beat_structs.last() {
                     // beat.start + beat.duration
                     // We can just use the beat struct's calculated duration
                      // But wait, beats_extended/duration_seconds used for AnalysisResult need update?
                      // AnalysisResult uses `beats` (which we popped) and `duration_seconds`.
                      // We should update duration_seconds.
                      // beat_structs[last].start + beat_structs[last].duration
                 }
                 
                 // Fix Segments (Tail segment might now exceed bounds)
                  if let Some(last_seg) = segments.last_mut() {
                       if let Some(last_beat) = beat_structs.last() {
                           let end_t = last_beat.start + last_beat.duration;
                           if last_seg.end_time > end_t {
                               last_seg.end_time = end_t;
                           }
                           // If segment became empty/inverted, pop it?
                           if last_seg.start_time >= last_seg.end_time {
                               segments.pop();
                           }
                       }
                  }

                  // Sanitize Jump Candidates (remove stale references to pruned beats)
                  let final_count = beat_structs.len();
                  for beat in &mut beat_structs {
                      beat.jump_candidates.retain(|&cand_id| cand_id < final_count);
                  }
             }
        }
        
        let final_duration = if let Some(last) = beat_structs.last() {
            last.start + last.duration
        } else {
            duration_seconds
        };
        
        Ok(AnalysisResult {
            duration_seconds: final_duration,
            beats,
            downbeats,
            beat_structs,
            segments,
            k_optimal: result.k_optimal,
            novelty_curve: result.novelty_curve,
            peaks: result.peaks,
            error: None,
        })
    }
}


