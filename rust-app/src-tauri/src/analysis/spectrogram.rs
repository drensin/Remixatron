use ndarray::{Array2, Axis};
use rustfft::{FftPlanner, num_complex::Complex};
use std::sync::Arc;
use std::fs::File;
use std::io::BufReader;

pub struct LogSpect {
    pub sample_rate: u32,
    pub win_length: usize,
    pub hop_length: usize,
    pub filterbank: Array2<f32>,
    pub fft: Arc<dyn rustfft::Fft<f32>>,
}

impl LogSpect {
    pub fn new(filterbank_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Load filterbank from JSON
        let file = File::open(filterbank_path)?;
        let reader = BufReader::new(file);
        let fb_vec: Vec<Vec<f32>> = serde_json::from_reader(reader)?;
        
        let rows = fb_vec.len();
        let cols = fb_vec[0].len();
        
        // Convert to ndarray. JSON is likely (InputBins, OutputBands).
        // Flatten
        let flat_fb: Vec<f32> = fb_vec.into_iter().flatten().collect();
        let filterbank = Array2::from_shape_vec((rows, cols), flat_fb)?;
        
        // Params hardcoded from BeatNet
        let sample_rate = 22050;
        let win_length = 1411; 
        let hop_length = 441;
        let fft_size = 1411; // Matches Python default frame_size

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);

        Ok(Self {
            sample_rate,
            win_length,
            hop_length,
            filterbank,
            fft,
        })
    }

    pub fn process_audio(&self, audio: &[f32]) -> Array2<f32> {
        let num_samples = audio.len();
        // Frame processing
        // Number of frames?
        // madmom usually pads? 
        // Let's implement basic framing:
        // centered? BeatNet implies framing logic.
        // For verification, we assume `audio` is already padded or matched.
        // Or we implement explicit framing.
        
        let fft_size = self.fft.len();
        let num_frames = (num_samples - self.win_length) / self.hop_length + 1;
        
        // Prepare output buffer for Spectrogram: (NumFrames, NumBands)
        // Actually FilteredSpectrogram: (NumFrames, NumBands)
        // Then Log: (NumFrames, NumBands)
        // Then Diff: (NumFrames, NumBands)
        // Then Stack: (NumFrames, 2 * NumBands)
        
        // Wait, filterbank shape is (1025, 136)?
        // Check export script: "Captured filterbank shape: (705, 136)"
        // Why 705? 
        // 1411 samples window.
        // If FFT is done on 1411 samples (zero padded to 2048?), bins = 1025.
        // If FFT is done on 1411 samples *directly*? 
        // No, `madmom` STFT usually aligns to `fft_size`.
        // Wait. `LogarithmicFilterbank` uses `np.fft.fftfreq(fft_size)`.
        // If `fft_size` was inferred as 1411? (Not power of 2).
        // 1411 / 2 + 1 = 706.5? No.
        // 1411 is odd. 
        // `rfft` of 1411 size -> 706 bins. 
        // (0..705).
        // Ah, `madmom` might use `frame_size` as `fft_size` if not specified?
        // BeatNet log_spect.py: `stft = ShortTimeFourierTransformProcessor()`. 
        // It doesn't pass fft_size.
        // madmom STFT defaults to `frame_size` if `fft_size` is None.
        // So BeatNet uses 1411-point DFT? Slow, but possible.
        // 136 bands.
        
        let _num_bands = self.filterbank.ncols();
        let num_bins = self.filterbank.nrows(); // Should be 706? Or 705?
        
        // If 705, then nyquist is at bin 706?
        // Let's verify what `rustfft` does for 1411 size.
        // Note: 1411 is prime? 1411 = 17 * 83. Not prime.
        // `rustfft` handles arbitrary sizes.
        
        // We need to match STFT exactly.
        // Framing:
        // Hanning window? `madmom` SignalProcessor defaults?
        // `madmom.audio.signal.SignalProcessor`: defaults to `win_type=None`? No, usually Hanning.
        // `BeatNet/log_spect.py`: `sig = SignalProcessor(...)`.
        // `madmom` defaults: `frame_size=2048`.
        // BeatNet passes `win_length=1411`.
        // `FramedSignalProcessor` defaults `origin='center'`? 
        // We will stick to simple framing for now and see residuals.
        
        let mut frames: Vec<Vec<f32>> = Vec::new();
        // Naive framing
        for i in 0..num_frames {
            let start = i * self.hop_length;
            let end = start + self.win_length;
            if end > num_samples { break; }
            frames.push(audio[start..end].to_vec());
        }

        // Apply Window? 
        // `madmom` applies Hanning by default if not specified? 
        // Check `madmom` docs/source. `SignalProcessor` usually raw.
        // `FramedSignalProcessor` does framing.
        // `STFT` does windowing?
        // `ShortTimeFourierTransformProcessor(window=HanningWindow, ...)`?
        // We assume Hanning for now.
        
        let window: Vec<f32> = (0..self.win_length)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (self.win_length - 1) as f32).cos()))
            .collect();

        // 1. Compute Mag Spectrogram
        let mut spec_frames = Vec::new();
        
        for frame in frames {
            let mut buf_in: Vec<Complex<f32>> = Vec::with_capacity(fft_size);
            
            for (i, &s) in frame.iter().enumerate() {
                 buf_in.push(Complex::new(s * window[i], 0.0));
            }
            
            // No padding needed if fft_size == win_length = 1411
            
            self.fft.process(&mut buf_in);
            
            // Compute Magnitude (bins 0..706)
            // madmom `Spectrum` usually keeps unique bins.
            // (N/2 + 1). (1411/2 + 1) = 705.5 -> 706 bins. 
            // 0..705.
            let mag: Vec<f32> = buf_in.iter().take(num_bins)
                .map(|c| c.norm())
                .collect();
                
            spec_frames.push(mag);
        }
        
        // 2. Filterbank
        // (NumFrames, 706) * (706, 136) -> (NumFrames, 136)
        // Convert spec_frames to Array2
        let num_actual_frames = spec_frames.len();
        let flattened_spec: Vec<f32> = spec_frames.into_iter().flatten().collect();
        let spec_matrix = Array2::from_shape_vec((num_actual_frames, num_bins), flattened_spec).unwrap();
        
        let filtered = spec_matrix.dot(&self.filterbank);
        
        // 3. Log LogarithmicSpectrogramProcessor(mul=1, add=1)
        // log10(mul * x + add)
        let logged = filtered.mapv(|x| (1.0 * x + 1.0).log10());
        
        // 4. Diff (First Order Difference)
        // diff_ratio=0.5, positive=True
        // diff = x[t] - x[t-1]
        // If positive=True: max(0, diff)
        // Stack: hstack(spec, diff)
        
        // Compute diff
        let mut diff = Array2::zeros(logged.raw_dim());
        // For t=0, diff=0? Or replicate? madmom usually pad?
        // "Calculates the difference ... forward difference?"
        // Usually `diff[t] = x[t] - x[t-1]`.
        // t=0: x[0] - 0? Or x[0] - x[0]?
        // madmom default: 
        // We'll perform standard diff.
        
        for i in 1..num_actual_frames {
            let row = logged.row(i);
            let prev = logged.row(i-1);
            let d = &row - &prev;
            diff.row_mut(i).assign(&d);
        }
        
        // Apply positive and ratio?
        // diff_ratio=0.5 -> does it mean return `spec + 0.5 * diff`?
        // No, `stack_diffs=np.hstack`.
        // It returns [spec, diff].
        // But what about `diff_ratio`?
        // madmom docs: "Scale the difference by this factor."
        // diff = diff * 0.5.
        // positive_diffs=True -> diff = max(0, diff).
        
        let processed_diff = diff.mapv(|x| (x * 0.5).max(0.0));
        
        // 5. Stack
        // output = concatenation of logged and processed_diff along Axis(1)
        // Shape: (NumFrames, 136 + 136) = (NumFrames, 272).
        
        // ndarray concat
        let output = ndarray::concatenate(Axis(1), &[logged.view(), processed_diff.view()]).unwrap();
        
        output
    }
}
