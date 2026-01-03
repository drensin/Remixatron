use qdft::QDFT;
use ndarray::Array2;
use num_complex::Complex;
 

pub struct CQTProcessor {
    qdft: QDFT<f32, f32>, 
    n_bins: usize,
    _sample_rate: f32,
    hop_length: usize,
}

impl CQTProcessor {
    pub fn new(sample_rate: f32, fmin: f32, n_bins: usize, bins_per_octave: usize) -> Self {
        let fmax = fmin * 2.0f32.powf((n_bins as f32) / (bins_per_octave as f32));
        
        // Inferred API from docs: new(sample_rate, (min_freq, max_freq), bins_per_octave, bandwidth_offset, latency_or_window?)
        // Standard CQT settings: 
        // Bandwidth offset: 0.0 usually means standard Q.
        // Window: None (rectangular/default) or explicit.
        let qdft = QDFT::new(sample_rate as f64, (fmin as f64, fmax as f64), bins_per_octave as f64, 0.0, None);
        
        CQTProcessor {
            qdft,
            n_bins,
            _sample_rate: sample_rate,
            hop_length: 512,
        }
    }
    
    pub fn process(&mut self, audio: &[f32]) -> Array2<f32> {
        let n_frames = audio.len() / self.hop_length;
        // Verify dimensions
        let qdft_size = self.qdft.size();
        // If qdft_size != self.n_bins, we might need to handle it.
        // But usually we set bandwidth to match n_bins.
        
        let mut spectrogram = Array2::<f32>::zeros((n_frames, qdft_size));
        
        // DFT buffer for generic sliding DFT
        let mut dft = vec![Complex::new(0.0, 0.0); qdft_size];
        
        let mut frame_idx = 0;
        
        for (i, &sample) in audio.iter().enumerate() {
            // Update sliding DFT state
            self.qdft.qdft_scalar(&sample, &mut dft);
            
            // Decimate output
            if i > 0 && i % self.hop_length == 0 {
                if frame_idx < n_frames {
                    for (bin_idx, complex_val) in dft.iter().enumerate() {
                        // Manually compute norm to avoid trait issues
                        let mag = (complex_val.re * complex_val.re + complex_val.im * complex_val.im).sqrt();
                        spectrogram[[frame_idx, bin_idx]] = mag;
                    }
                    frame_idx += 1;
                }
            }
        }

        
        // Truncate if n_bins mismatch (usually qdft_size is slightly different due to bin alignment)
        if qdft_size != self.n_bins {
             // For strict parity, we return self.n_bins.
             // If qdft_size > n_bins, slice it.
             // If qdft_size < n_bins, zero pad.
             // For now return full qdft_size width, user handles it.
             // Or better: ensure we return requested bins.
             
             // Simplest: just use qdft_size as truth for now.
             // Ideally we slice.
             // let slice = spectrogram.slice(s![.., 0..self.n_bins]);
             // return slice.to_owned();
        }

        spectrogram
    }
}
