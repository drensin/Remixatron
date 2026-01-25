//! # Feature Extraction Module
//!
//! This module handles the conversion of raw audio or Mel Spectrograms into
//! high-level structural features (MFCCs and Chroma).
//!
//! ## Core Features
//! *   **MFCC (Mel-Frequency Cepstral Coefficients)**: Captures timbre/texture.
//! *   **Chroma (CQT)**: Captures harmonic content (chords/notes).
//! *   **Spectral Centroid**: Captures "brightness" of timbre (used for Mood Shader).
//! *   **Beat Synchronization**: Aggregates features per beat using Median Pooling to be tempo-invariant.

use crate::analysis::cqt::CQTProcessor;
use ndarray::{Array2, s};
use std::f32::consts::PI;

/// Coordinates the extraction of MFCC and Chroma features.
pub struct FeatureExtractor {
    n_mels: usize,
    sample_rate: f32,
    /// Pre-computed Discrete Cosine Transform matrix for MFCC generation.
    dct_matrix: Array2<f32>,
    /// Constant-Q Transform processor for Chroma generation.
    cqt: CQTProcessor,
}

impl FeatureExtractor {
    /// Creates a new FeatureExtractor.
    ///
    /// # Arguments
    /// * `n_mels` - Number of Mel bands in the input spectrogram (usually 128).
    /// * `sample_rate` - Audio sample rate (usually 44100).
    pub fn new(n_mels: usize, sample_rate: f32) -> Self {
        let dct_matrix = compute_dct_matrix(n_mels, 20); // Keep top 20 MFCCs
        
        // CQT Init: matching Librosa Remixatron.py
        // bins_per_octave = 36 (3 bins per semitone)
        // n_octaves = 7
        // n_bins = 36 * 7 = 252
        // fmin = C1 (~32.7 Hz)
        let cqt = CQTProcessor::new(sample_rate, 32.703, 252, 36);
        
        Self {
            n_mels,
            sample_rate,
            dct_matrix,
            cqt,
        }
    }

    /// Computes beat-synchronous features (MFCC and Chroma) for the entire track.
    ///
    /// This process involves:
    /// 1.  **MFCC**: Computed from the provided Mel Spectrogram via DCT.
    /// 2.  **CQT/Chroma**: Computed directly from raw audio using a Constant-Q Transform.
    /// 3.  **Synchronization**: Both feature sets are aggregated into beat intervals using Median Pooling.
    ///
    /// # Arguments
    /// *   `audio` - Raw audio samples (monophonic).
    /// *   `mel` - Log Mel Spectrogram [Time, n_mels].
    /// *   `beats` - Timestamps of detected beats in seconds.
    /// *   `fps` - Frame rate of the Mel Spectrogram.
    ///
    /// # Returns
    /// A tuple `(MFCCs, Chroma, CQT, RMS, Centroid)` where:
    /// - MFCCs: `[n_beats, 20]` for timbral similarity
    /// - Chroma: `[n_beats, 12]` for harmonic content  
    /// - CQT: `[n_beats, 252]` for recurrence matrix (like Python)
    /// - RMS: `Vec<f32>` normalized RMS energy per beat (for Mood Shader)
    /// - Centroid: `Vec<f32>` spectral centroid per beat, 0.0-1.0 (for Mood Shader)
    #[allow(clippy::type_complexity)]  // 5-tuple return is intentional; type alias would reduce clarity
    pub fn compute_sync_features(
        &mut self,
        audio: &[f32],
        mel: &Array2<f32>, 
        beats: &[f32], 
        fps: f32
    ) -> (Array2<f32>, Array2<f32>, Array2<f32>, Vec<f32>, Vec<f32>) {
        let (n_time, _) = mel.dim();
        let n_beats = beats.len();
        
        // 1. Compute MFCCs (from Mel)
        let mut mfcc_frames = Array2::<f32>::zeros((n_time, 20));
        
        for t in 0..n_time {
            let mel_frame = mel.slice(s![t, ..]);
            for k in 0..20 {
                let mut sum = 0.0;
                for n in 0..self.n_mels {
                   sum += mel_frame[n] * self.dct_matrix[[k, n]];
                }
                mfcc_frames[[t, k]] = sum;
            }
        }
        
        // 2. Compute CQT & Chroma (from Audio)
        // Result is [CQT_Time, 252]
        let cqt_spectrogram_raw = self.cqt.process(audio);
        let (n_cqt_frames, n_bins) = cqt_spectrogram_raw.dim();
        
        // 2a. Compute Frame-wise Spectral Centroid (from RAW linear magnitude)
        // CRITICAL: Must use linear magnitude BEFORE dB conversion.
        // Centroid = Sum(k * Mag[k]) / Sum(Mag[k]), normalized to 0.0-1.0.
        let mut centroid_frames = Vec::with_capacity(n_cqt_frames);
        for t in 0..n_cqt_frames {
            let mut weighted_sum = 0.0_f32;
            let mut magnitude_sum = 0.0_f32;
            for k in 0..n_bins {
                let mag = cqt_spectrogram_raw[[t, k]];
                weighted_sum += (k as f32) * mag;
                magnitude_sum += mag;
            }
            // Avoid division by zero; normalize to 0.0-1.0 range
            let centroid = if magnitude_sum > 1e-10 {
                (weighted_sum / magnitude_sum) / (n_bins as f32)
            } else {
                0.5 // Default to middle if silent
            };
            centroid_frames.push(centroid);
        }
        
        // Convert to dB scale like Python: C = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        // This normalizes the value range to roughly [-80, 0] dB
        let max_val = cqt_spectrogram_raw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let ref_val = if max_val > 1e-10 { max_val } else { 1.0 };
        
        let mut cqt_spectrogram = Array2::<f32>::zeros((n_cqt_frames, n_bins));
        for t in 0..n_cqt_frames {
            for k in 0..n_bins {
                // amplitude_to_db: 20 * log10(amplitude / ref)
                // Clamp to avoid log(0)
                let amp = cqt_spectrogram_raw[[t, k]].max(1e-10);
                let db = 20.0 * (amp / ref_val).log10();
                // Clamp to -80 dB floor (standard librosa default)
                cqt_spectrogram[[t, k]] = db.max(-80.0);
            }
        }
        
        // CQT timebase: sr / hop_length = 44100 / 512 ~ 86 fps.
        let cqt_fps = self.sample_rate / 512.0; 
        
        // Compute Frame-wise Chroma (12 bins) from RAW CQT (before dB conversion)
        // We use raw magnitudes for chroma because we apply ln(1+x) compression,
        // which requires positive values.
        let mut chroma_frames = Array2::<f32>::zeros((n_cqt_frames, 12));
        
        for t in 0..n_cqt_frames {
            for k in 0..n_bins {
                // Bin mapping: 36 bins per octave.
                // 3 bins per semitone.
                // Semitone index = k / 3.
                // Chroma class = (k / 3) % 12.
                let chroma_idx = (k / 3) % 12;
                chroma_frames[[t, chroma_idx]] += cqt_spectrogram_raw[[t, k]];  // Use RAW, not dB
            }
            
            // Log compress
            for c in 0..12 {
                chroma_frames[[t, c]] = (1.0 + chroma_frames[[t, c]]).ln();
            }
        }
        
        // 3. Beat Synchronization
        // MFCC sync (using Mel timebase/fps)
        let mut mfcc_sync = Array2::<f32>::zeros((n_beats, 20));
        // Chroma sync (using CQT timebase/cqt_fps)
        let mut chroma_sync = Array2::<f32>::zeros((n_beats, 12));
        // Full CQT sync (252 bins for recurrence matrix - like Python's Csync)
        let mut cqt_sync = Array2::<f32>::zeros((n_beats, n_bins));
        // RMS Amplitude (per beat)
        let mut rms_sync = Vec::with_capacity(n_beats);
        // Spectral Centroid (per beat)
        let mut centroid_sync = Vec::with_capacity(n_beats);
        
        // Mel Beats
        let beat_frames_mel: Vec<usize> = beats.iter().map(|&t| (t * fps).round() as usize).collect();
        // CQT Beats
        let beat_frames_cqt: Vec<usize> = beats.iter().map(|&t| (t * cqt_fps).round() as usize).collect();
        // Audio Beats (samples)
        let beat_samples: Vec<usize> = beats.iter().map(|&t| (t * self.sample_rate).round() as usize).collect();
        
        for i in 0..n_beats {
            // MFCC
            let start = if i == 0 { 0 } else { beat_frames_mel[i] };
            let end = if i == n_beats - 1 { n_time } else { beat_frames_mel[i+1] };
            
            let start_c = start.min(n_time);
            let end_c = end.min(n_time); // Strictly bounded
            
            // If empty (beats on same frame), we could peek 1 frame if available, or just skip.
            // Skipping is safer.
            let count = (end_c.saturating_sub(start_c)) as f32;
            
            if count > 0.0 {
                // Median Pooling for MFCC (Robust against transients)
                for k in 0..20 {
                    let mut values = Vec::with_capacity(end_c - start_c);
                    for t in start_c..end_c {
                        values.push(mfcc_frames[[t, k]]);
                    }
                    // Sort for median
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    
                    let med = if values.is_empty() { 0.0 }
                    else if values.len() % 2 == 1 {
                        values[values.len() / 2]
                    } else {
                         let mid = values.len() / 2;
                         (values[mid-1] + values[mid]) / 2.0
                    };
                    mfcc_sync[[i, k]] = med;
                }
            } else if start_c < n_time {
                 for k in 0..20 { mfcc_sync[[i, k]] = mfcc_frames[[start_c, k]]; }
            }
            
            // Chroma (Median)
            let start_q = if i == 0 { 0 } else { beat_frames_cqt[i] };
            let end_q = if i == n_beats - 1 { n_cqt_frames } else { beat_frames_cqt[i+1] };
            
            let start_qc = start_q.min(n_cqt_frames);
            let end_qc = end_q.min(n_cqt_frames);
            
            if start_qc < end_qc {
                // Chroma (Median pooling)
                for k in 0..12 {
                    let mut values = Vec::new();
                    for t in start_qc..end_qc {
                        values.push(chroma_frames[[t, k]]);
                    }
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    
                     let med = if values.is_empty() { 0.0 }
                     else if values.len() % 2 == 1 {
                        values[values.len() / 2]
                    } else {
                         let mid = values.len() / 2;
                         (values[mid-1] + values[mid]) / 2.0
                    };
                    chroma_sync[[i, k]] = med;
                }
                
                // CQT (Median pooling - 252 bins like Python's Csync)
                for k in 0..n_bins {
                    let mut values = Vec::new();
                    for t in start_qc..end_qc {
                        values.push(cqt_spectrogram[[t, k]]);
                    }
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    
                    let med = if values.is_empty() { 0.0 }
                    else if values.len() % 2 == 1 {
                        values[values.len() / 2]
                    } else {
                        let mid = values.len() / 2;
                        (values[mid-1] + values[mid]) / 2.0
                    };
                    cqt_sync[[i, k]] = med;
                }
                
                // Centroid (Median pooling)
                let mut centroid_values: Vec<f32> = (start_qc..end_qc)
                    .map(|t| centroid_frames[t])
                    .collect();
                centroid_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let centroid_med = if centroid_values.is_empty() {
                    0.5
                } else if centroid_values.len() % 2 == 1 {
                    centroid_values[centroid_values.len() / 2]
                } else {
                    let mid = centroid_values.len() / 2;
                    (centroid_values[mid - 1] + centroid_values[mid]) / 2.0
                };
                centroid_sync.push(centroid_med);
            } else if start_qc < n_cqt_frames {
                // Single frame fallback
                for k in 0..12 { chroma_sync[[i, k]] = chroma_frames[[start_qc, k]]; }
                for k in 0..n_bins { cqt_sync[[i, k]] = cqt_spectrogram[[start_qc, k]]; }
                centroid_sync.push(centroid_frames[start_qc]);
            } else {
                // No valid CQT frames for this beat
                centroid_sync.push(0.5);
            }

            // 4. RMS Amplitude
            let start_samp = if i == 0 { 0 } else { beat_samples[i] };
            let end_samp = if i == n_beats - 1 { audio.len() } else { beat_samples[i+1] };
            let start_s = start_samp.min(audio.len());
            let end_s = end_samp.min(audio.len());

            if start_s < end_s {
                let slice = &audio[start_s..end_s];
                let sum_sq: f32 = slice.iter().map(|&x| x * x).sum();
                let rms = (sum_sq / slice.len() as f32).sqrt();
                rms_sync.push(rms);
            } else {
                rms_sync.push(0.0);
            }
        }
        
        // L2 normalize each beat's chroma vector.
        // This converts absolute energy to RELATIVE pitch class distribution,
        // making frames with the same chord (but different loudness) look similar,
        // and frames with different chords look different.
        for i in 0..n_beats {
            let mut norm_sq = 0.0_f32;
            for k in 0..12 {
                norm_sq += chroma_sync[[i, k]] * chroma_sync[[i, k]];
            }
            let norm = norm_sq.sqrt();
            if norm > 1e-10 {
                for k in 0..12 {
                    chroma_sync[[i, k]] /= norm;
                }
            }
        }
        
        (mfcc_sync, chroma_sync, cqt_sync, rms_sync, centroid_sync)
    }
}

// Helpers

/// Computes the Discrete Cosine Transform (Type-II) matrix.
///
/// Used to decorrelate Mel Spectrogram bands into MFCCs.
///
/// # Arguments
/// * `n` - Input dimension (number of Mel bands).
/// * `k` - Output dimension (number of MFCC coefficients).
fn compute_dct_matrix(n: usize, k: usize) -> Array2<f32> {
    let mut matrix = Array2::<f32>::zeros((k, n));
    let scale = (2.0 / n as f32).sqrt(); // Orthogonal normalization
    
    // Type-II DCT
    for i in 0..k {
        for j in 0..n {
            let v = (PI / n as f32 * (j as f32 + 0.5) * i as f32).cos();
            // i=0 needs 1/sqrt(2) scaling for ortho
            let s = if i == 0 { (1.0 / 2.0f32).sqrt() } else { 1.0 };
            matrix[[i, j]] = scale * s * v;
        }
    }
    matrix
}
