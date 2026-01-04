//! # High-Quality Audio Loader
//!
//! This module provides a robust, high-fidelity audio loader designed for
//! the Analysis Pipeline. It differs from the playback loader in two key ways:
//! 1.  **Mono Mixdown**: It forces downmixing to mono (required for spectral analysis).
//! 2.  **High-Quality Resampling**: It uses `rubato` (Sinc Interpolation) instead of linear interpolation
//!     to preserve spectral characteristics during rate conversion.
//!
//! **Usage**:
//! Used by `AnalysisWorkflow` to prepare audio for Beat Tracking and Feature Extraction.

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use std::fs::File;
use std::path::Path;
use anyhow::{anyhow, Result};
use rubato::{Resampler, SincFixedIn, SincInterpolationType, SincInterpolationParameters, WindowFunction};

/// Container for decoded and processed audio.
pub struct AudioData {
    /// Interleaved audio samples (though usually Mono in this module).
    pub signal: Vec<f32>,
    /// Sample rate of the data (e.g., 22050 Hz).
    pub sample_rate: u32,
    /// Number of channels (locked to 1 for this loader).
    pub channels: u32,
}

/// Loads an audio file, mixes it to mono, and resamples it to the target rate.
///
/// # Arguments
/// * `path` - Path to the audio file.
/// * `target_sr` - Desired sample rate (e.g., 22050 for analysis).
///
/// # Returns
/// An `AudioData` struct containing the mono signal at the target rate.
///
/// # Notes
/// *   Uses `symphonia` for broad format support (MP3, WAV, FLAC, etc.).
/// *   Uses `rubato` for audiophile-quality Sinc resampling.
/// *   Always returns 1 channel (Mono).
pub fn load_audio<P: AsRef<Path>>(path: P, target_sr: u32) -> Result<AudioData> {
    let src = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());

    let hint = Hint::new();
    let format_opts = FormatOptions::default();
    let metadata_opts = MetadataOptions::default();
    let decoder_opts = DecoderOptions::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)?;
    let mut format = probed.format;
    
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow!("no supported audio tracks"))?;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &decoder_opts)?;

    let track_id = track.id;
    let mut samples: Vec<f32> = Vec::new();
    let mut source_sr = 0;
    
    // Decode all packets
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(Error::IoError(_)) => break,
            Err(Error::ResetRequired) => {
                // The track list has been changed. Re-examine it and create a new decoder
                // if necessary. For now, we'll just ignore this case.
                unimplemented!();
            }
            Err(err) => return Err(anyhow!(err)),
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                source_sr = decoded.spec().rate;
                // Just force stereo to mono via average if 2 channels, or simple copy if 1
                let spec = *decoded.spec();
                let mut sample_buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
                sample_buf.copy_interleaved_ref(decoded);
                
                let buf_samples = sample_buf.samples();
                
                if spec.channels.count() == 1 {
                    samples.extend_from_slice(buf_samples);
                } else {
                     // Interleaved stereo -> average to mono
                     for frame in buf_samples.chunks(spec.channels.count()) {
                         let sum: f32 = frame.iter().sum();
                         samples.push(sum / spec.channels.count() as f32);
                     }
                }
            }
            Err(Error::IoError(_)) => break,
            Err(Error::DecodeError(_)) => (), // ignore decode errors
            Err(err) => return Err(anyhow!(err)),
        }
    }
    
    // Resample if necessary
    if source_sr != target_sr {
        // High quality resampling with rubato
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };
        
        let mut resampler = SincFixedIn::<f32>::new(
            target_sr as f64 / source_sr as f64,
            2.0, // max resample ratio
            params,
            samples.len(),
            1, // channels
        )?;
        
        // Rubato expects Vec<Vec<f32>> for channels
        let waves_in = vec![samples];
        let waves_out = resampler.process(&waves_in, None)?;
        samples = waves_out[0].clone();
    }
    
    Ok(AudioData {
        signal: samples,
        sample_rate: target_sr,
        channels: 1
    })
}
