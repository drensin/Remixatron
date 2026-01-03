use crate::analysis::cqt::CQTProcessor;
use ndarray::{Array2, s};
use std::f32::consts::PI;

pub struct FeatureExtractor {
    n_mels: usize,
    sample_rate: f32,
    dct_matrix: Array2<f32>,
    cqt: CQTProcessor,
}

impl FeatureExtractor {
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

    /// Compute beat-synchronous features.
    /// Input: 
    ///   audio: Raw audio samples (for CQT)
    ///   mel: Log Mel Spectrogram (Time, Freq=128) (for MFCC)
    ///   beats: List of beat times in seconds.
    ///   fps: Frames per second of the mel spectrogram (usually 50.0).
    /// Returns: (MFCC_Sync, Chroma_Sync)
    /// MFCC_Sync: (NumBeats, 20)
    /// Chroma_Sync: (NumBeats, 12)
    pub fn compute_sync_features(
        &mut self,
        audio: &[f32],
        mel: &Array2<f32>, 
        beats: &[f32], 
        fps: f32
    ) -> (Array2<f32>, Array2<f32>) {
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
        let cqt_spectrogram = self.cqt.process(audio);
        let (n_cqt_frames, n_bins) = cqt_spectrogram.dim();
        
        // CQT hop usually 512. Mel hop?
        // We need to align CQT frames to Mel frames (or beat frames).
        // Best approach: Sync CQT to beats directly, using CQT timebase.
        // CQT timebase: sr / hop_length = 44100 / 512 ~ 86 fps.
        // Mel fps usually ~50 defined by caller.
        // We use `self.cqt.hop_length` implicitly via `process`. 
        // We assume `process` returns frames at `sr/512` rate.
        let cqt_fps = self.sample_rate / 512.0; 
        
        // Compute Frame-wise Chroma (12 bins) from CQT
        let mut chroma_frames = Array2::<f32>::zeros((n_cqt_frames, 12));
        
        for t in 0..n_cqt_frames {
            for k in 0..n_bins {
                // Bin mapping: 36 bins per octave.
                // 3 bins per semitone.
                // Semitone index = k / 3.
                // Chroma class = (k / 3) % 12.
                let chroma_idx = (k / 3) % 12;
                chroma_frames[[t, chroma_idx]] += cqt_spectrogram[[t, k]];
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
        
        // Mel Beats
        let beat_frames_mel: Vec<usize> = beats.iter().map(|&t| (t * fps).round() as usize).collect();
        // CQT Beats
        let beat_frames_cqt: Vec<usize> = beats.iter().map(|&t| (t * cqt_fps).round() as usize).collect();
        
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
                for t in start_c..end_c {
                    for k in 0..20 { mfcc_sync[[i, k]] += mfcc_frames[[t, k]]; }
                }
                for k in 0..20 { mfcc_sync[[i, k]] /= count; }
            } else if start_c < n_time {
                // Determine value for single point (aliasing?)
                // Just use start_c frame
                 for k in 0..20 { mfcc_sync[[i, k]] = mfcc_frames[[start_c, k]]; }
            }
            
            // Chroma (Median)
            let start_q = if i == 0 { 0 } else { beat_frames_cqt[i] };
            let end_q = if i == n_beats - 1 { n_cqt_frames } else { beat_frames_cqt[i+1] };
            
            let start_qc = start_q.min(n_cqt_frames);
            let end_qc = end_q.min(n_cqt_frames);
            
            if start_qc < end_qc {
                for k in 0..12 {
                    let mut values = Vec::new();
                    for t in start_qc..end_qc {
                        values.push(chroma_frames[[t, k]]);
                    }
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    
                     let med = if values.len() == 0 { 0.0 }
                     else if values.len() % 2 == 1 {
                        values[values.len() / 2]
                    } else {
                         let mid = values.len() / 2;
                         (values[mid-1] + values[mid]) / 2.0
                    };
                    chroma_sync[[i, k]] = med;
                }
            } else if start_qc < n_cqt_frames {
                 // Single frame fallback
                 for k in 0..12 { chroma_sync[[i, k]] = chroma_frames[[start_qc, k]]; }
            }
        }
        
        (mfcc_sync, chroma_sync)
    }
}

// Helpers

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
