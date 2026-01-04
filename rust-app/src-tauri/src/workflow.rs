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
//! 6.  **Structural Analysis**: Spectral Clustering is performed (via `structure.rs`).
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
    /// # Errors
    /// Returns an error if:
    /// * File cannot be opened or decoded.
    /// * ONNX models fail to load or run.
    /// * Audio is too short or silent.
    pub fn analyze(&self, audio_path: &str) -> Result<AnalysisResult> {
        // 1. Load & Resample Audio
        // We use the High-Quality Loader (Rubato) to get mono audio at 22050 Hz.
        // This ensures maximum spectral accuracy for the downstream feature extractors.
        let target_sr = 22050;
        let audio_data = load_audio(audio_path, target_sr).map_err(|e| anyhow!(e.to_string()))?;
        
        let audio = audio_data.signal;
        let sr = audio_data.sample_rate; // Should be 22050
        
        let duration_seconds = audio.len() as f32 / sr as f32;
        
        // 2. Mel Spectrogram
        // Compute the input tensor for the Beat Tracker.
        let mut mel_proc = MelProcessor::new(&self.mel_path)?;
        let mel = mel_proc.process(&audio)?;
        
        // 3. Beat Tracking
        // Run inference to get beat/downbeat activation logits.
        let mut tracker = BeatProcessor::new(&self.beat_path)?;
        let (b_logits, d_logits) = tracker.process(&mel)?;
        
        // Post-process logits into discrete timestamps (FPS=50).
        let post = MinimalPostProcessor::new(50.0);
        let (beats, downbeats) = post.process(&b_logits, &d_logits)?;
        
        // 4. Feature Extraction & Structure
        // Extract Timbre (MFCC) and Harmony (Chroma) for every beat.
        let mut feature_ex = FeatureExtractor::new(128, target_sr as f32);
        let mel_2d = mel.index_axis(Axis(0), 0).to_owned();
        
        let mut beats_extended = beats.clone();
        beats_extended.push(duration_seconds); // Add end marker to define the last beat's duration
        
        let (mut mfcc, mut chroma) = feature_ex.compute_sync_features(&audio, &mel_2d, &beats_extended, 50.0);
        
        // Remove the feature vector corresponding to the end marker
        // because it has 0 duration and is not a playble unit.
        if mfcc.nrows() > 0 {
             mfcc = mfcc.slice(s![0..mfcc.nrows()-1, ..]).to_owned();
             chroma = chroma.slice(s![0..chroma.nrows()-1, ..]).to_owned();
        }
        
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
        // Perform Spectral Clustering on the beat-synchronous features.
        let analyzer = StructureAnalyzer::new();
        let result = analyzer.compute_segments_knn(&mfcc, &chroma, None); 
        
        // 7. Assembly
        // Convert the raw labels and jump graph into the final `Beat` and `Segment` structs.
        let mut beat_structs = Vec::with_capacity(result.labels.len());
        let mut segments = Vec::new();
        
        if !result.labels.is_empty() {
            let mut current_segment_id = 0;
            let mut intra_segment_index = 0;
            let mut segment_start_idx = 0;
            let mut current_label = result.labels[0];

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
                
                let candidates = if next_beat_idx < result.jumps.len() {
                    result.jumps[next_beat_idx].clone()
                } else {
                    Vec::new()
                };

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
                end_time: beats_extended[result.labels.len()],
                label: current_label,
            });
        }
        
        Ok(AnalysisResult {
            duration_seconds,
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


