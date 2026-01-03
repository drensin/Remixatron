use anyhow::{Result, anyhow};
use crate::audio_backend::decoder::decode_audio_file;
use crate::beat_tracker::mel::MelProcessor;
use crate::beat_tracker::inference::BeatProcessor;
use crate::beat_tracker::post_processor::MinimalPostProcessor;
use crate::analysis::features::FeatureExtractor;
use crate::analysis::structure::StructureAnalyzer;
use ndarray::Axis;


use crate::playback_engine::Beat;

/// High-level result of the track analysis
#[derive(serde::Serialize)]
pub struct AnalysisResult {
    pub duration_seconds: f32,
    pub beats: Vec<f32>,
    pub downbeats: Vec<f32>,
    pub beat_structs: Vec<Beat>,
    pub segments: Vec<Segment>,
    pub k_optimal: usize,
    pub error: Option<String>,
}

#[derive(serde::Serialize)]
pub struct Segment {
    pub start_time: f32,
    pub end_time: f32,
    pub label: usize,
}

pub struct Remixatron {
    mel_path: String,
    beat_path: String,
}

impl Remixatron {
    pub fn new(mel_path: &str, beat_path: &str) -> Self {
        Self {
            mel_path: mel_path.to_string(),
            beat_path: beat_path.to_string(),
        }
    }

    pub fn analyze(&self, audio_path: &str) -> Result<AnalysisResult> {
        // 1. Load & Resample Audio
        // We need 22050 Hz for models
        let (interleaved, sr, _) = decode_audio_file(audio_path).map_err(|e| anyhow!(e.to_string()))?;
        
        // Mono Mixdown
        let mut audio = Vec::with_capacity(interleaved.len() / 2);
        for chunk in interleaved.chunks(2) {
            if chunk.len() == 2 {
                audio.push((chunk[0] + chunk[1]) / 2.0);
            } else {
                 audio.push(chunk[0]); // Tail
            }
        }
        
        let target_sr = 22050;
        if sr != target_sr {
             audio = resample_linear(&audio, sr as f32, target_sr as f32);
        }
        
        let duration_seconds = audio.len() as f32 / target_sr as f32;
        
        // 2. Mel Spectrogram
        let mut mel_proc = MelProcessor::new(&self.mel_path)?;
        let mel = mel_proc.process(&audio)?;
        
        // 3. Beat Tracking
        let mut tracker = BeatProcessor::new(&self.beat_path)?;
        let (b_logits, d_logits) = tracker.process(&mel)?;
        
        let post = MinimalPostProcessor::new(50.0); // FPS=50
        let (beats, downbeats) = post.process(&b_logits, &d_logits)?;
        
        // 4. Feature Extraction & Structure
        let mut feature_ex = FeatureExtractor::new(128, target_sr as f32);
        let mel_2d = mel.index_axis(Axis(0), 0).to_owned();
        
        let mut beats_extended = beats.clone();
        beats_extended.push(duration_seconds);
        
        let (mfcc, chroma) = feature_ex.compute_sync_features(&audio, &mel_2d, &beats_extended, 50.0);
        
        let analyzer = StructureAnalyzer::new();
        let result = analyzer.compute_segments(&mfcc, &chroma, 0); // Auto-K
        
        // Convert labels to Segments

        
        // Create Beats and Segments
        let mut beat_structs = Vec::with_capacity(result.labels.len());
        let mut segments = Vec::new();
        
        if !result.labels.is_empty() {
            let mut current_segment_id = 0;
            let mut intra_segment_index = 0;
            let mut segment_start_idx = 0;
            let mut current_label = result.labels[0];
            
            // Bar position inference state
            let mut bar_pos_counter = 0;

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
                
                // Bar Position Inference
                // Check if this beat is close to a downbeat
                let is_downbeat = downbeats.iter().any(|&d| (d - start_time).abs() < 0.05);
                if is_downbeat {
                    bar_pos_counter = 0;
                }
                
                let duration = beats_extended[i+1] - start_time;
                
                beat_structs.push(Beat {
                    id: i,
                    start: start_time,
                    duration,
                    bar_position: bar_pos_counter,
                    cluster: label,
                    segment: current_segment_id,
                    intra_segment_index,
                    quartile: 0, // Calculated in playback_engine
                    jump_candidates: Vec::new(),
                });
                
                intra_segment_index += 1;
                bar_pos_counter = (bar_pos_counter + 1) % 4; // Assume 4/4 if no downbeat msg
            }
            
            // Tail Segment
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
            error: None,
        })
    }
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
