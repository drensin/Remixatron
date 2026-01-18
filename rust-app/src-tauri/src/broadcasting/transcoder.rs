//! # MP3 Transcoder
//!
//! This module provides real-time MP3 encoding of PCM audio samples. It runs
//! in a dedicated OS thread to ensure the audio encoding never blocks the
//! main application or Tokio runtime.
//!
//! ## Data Flow
//!
//! ```text
//! [Playback Engine] --> crossbeam::Sender<Vec<f32>> --> [Transcoder Thread]
//!                                                              |
//!                                                              v
//!                                                       LAME Encoder
//!                                                              |
//!                                                              v
//!                                    [tokio::broadcast::Sender<Bytes>] --> [HTTP Clients]
//! ```
//!
//! ## Thread Safety
//!
//! The transcoder uses `crossbeam_channel` for receiving PCM samples because it
//! provides a blocking API suitable for dedicated threads. The outgoing MP3 bytes
//! use `tokio::sync::broadcast` to efficiently multicast to all connected clients.
//!
//! ## Sample Rate Handling
//!
//! The transcoder accepts a shared `AtomicU32` for the sample rate, allowing the
//! playback engine to update the rate when a new track is loaded. The encoder
//! re-initializes when the sample rate changes.

use bytes::Bytes;
use crossbeam_channel::Receiver;
use mp3lame_encoder::{Builder, InterleavedPcm};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::thread;
use tokio::sync::broadcast;

/// Number of audio channels (stereo).
const CHANNELS: u8 = 2;

/// Default sample rate if none is set.
const DEFAULT_SAMPLE_RATE: u32 = 44_100;

/// Creates a new sample rate handle for sharing between engine and transcoder.
///
/// Returns an `Arc<AtomicU32>` initialized to the default sample rate (44.1kHz).
/// The playback engine should update this when a track is loaded.
pub fn create_sample_rate_handle() -> Arc<AtomicU32> {
    Arc::new(AtomicU32::new(DEFAULT_SAMPLE_RATE))
}

/// Spawns a dedicated background thread for MP3 encoding.
///
/// This function starts a long-running thread that continuously reads PCM
/// audio samples from `input_rx`, encodes them to MP3 format using the LAME
/// library, and broadcasts the resulting bytes to all subscribers of `output_tx`.
///
/// # Arguments
///
/// * `input_rx` - A `crossbeam_channel::Receiver` providing interleaved stereo
///   `f32` PCM samples. The receiver will block until samples are available.
/// * `output_tx` - A `tokio::sync::broadcast::Sender` for distributing encoded
///   MP3 bytes to all connected HTTP streaming clients.
/// * `sample_rate_handle` - A shared atomic handle that the playback engine
///   updates with the current track's sample rate.
///
/// # Panics
///
/// Panics if the LAME encoder fails to initialize. This is a critical error
/// that indicates a misconfiguration or resource exhaustion.
pub fn spawn_transcoder(
    input_rx: Receiver<Vec<f32>>,
    output_tx: broadcast::Sender<Bytes>,
    sample_rate_handle: Arc<AtomicU32>,
) {
    thread::spawn(move || {
        let mut current_sample_rate = sample_rate_handle.load(Ordering::Relaxed);
        let mut lame = create_encoder(current_sample_rate);

        // Main encoding loop: runs until the input channel is closed.
        loop {
            // Check if sample rate changed (new track loaded).
            let new_rate = sample_rate_handle.load(Ordering::Relaxed);
            if new_rate != current_sample_rate && new_rate > 0 {
                println!(
                    "[Transcoder] Sample rate changed: {} -> {}. Reinitializing encoder.",
                    current_sample_rate, new_rate
                );
                current_sample_rate = new_rate;
                lame = create_encoder(current_sample_rate);
            }

            match input_rx.recv() {
                Ok(samples) => {
                    encode_and_broadcast(&mut lame, &samples, &output_tx);
                }
                Err(_) => {
                    // Channel disconnected - application is shutting down.
                    println!("[Transcoder] Input channel closed. Exiting.");
                    break;
                }
            }
        }
    });
}

/// Creates and configures a new LAME encoder instance.
///
/// # Arguments
///
/// * `sample_rate` - The audio sample rate in Hz (e.g., 44100, 48000).
///
/// # Panics
///
/// Panics if encoder configuration fails.
fn create_encoder(sample_rate: u32) -> mp3lame_encoder::Encoder {
    let mut builder = Builder::new().expect("Failed to create LAME encoder builder");
    builder
        .set_num_channels(CHANNELS)
        .expect("Failed to set channel count");
    builder
        .set_sample_rate(sample_rate)
        .expect("Failed to set sample rate");
    builder
        .set_brate(mp3lame_encoder::Bitrate::Kbps320)
        .expect("Failed to set bitrate");
    builder
        .set_quality(mp3lame_encoder::Quality::Best)
        .expect("Failed to set encoding quality");

    builder.build().expect("Failed to build LAME encoder")
}

/// Encodes a buffer of PCM samples to MP3 and broadcasts the result.
///
/// This is an internal helper function that handles the LAME encoding and
/// error recovery. Failed encodings are logged but do not stop the transcoder.
///
/// # Arguments
///
/// * `lame` - A mutable reference to the initialized LAME encoder.
/// * `samples` - A slice of interleaved stereo `f32` PCM samples.
/// * `output_tx` - The broadcast sender for distributing encoded MP3 data.
fn encode_and_broadcast(
    lame: &mut mp3lame_encoder::Encoder,
    samples: &[f32],
    output_tx: &broadcast::Sender<Bytes>,
) {
    // Calculate required output buffer size.
    // LAME recommendation: 1.25 * num_samples + 7200 bytes.
    let num_frames = samples.len() / 2; // Stereo: 2 samples per frame
    let estimated_size = ((num_frames as f64) * 1.25 + 7200.0) as usize;

    let mut mp3_buffer: Vec<u8> = Vec::with_capacity(estimated_size);

    // Encode the PCM samples into the MP3 buffer.
    let input = InterleavedPcm(samples);
    match lame.encode(input, mp3_buffer.spare_capacity_mut()) {
        Ok(bytes_written) => {
            if bytes_written > 0 {
                // SAFETY: LAME guarantees it wrote exactly `bytes_written` bytes.
                unsafe {
                    mp3_buffer.set_len(bytes_written);
                }

                // Broadcast the MP3 chunk to all subscribers.
                // Errors here mean no clients are connected, which is fine.
                let _ = output_tx.send(Bytes::from(mp3_buffer));
            }
        }
        Err(e) => {
            eprintln!("[Transcoder] LAME encoding error: {:?}", e);
        }
    }
}
