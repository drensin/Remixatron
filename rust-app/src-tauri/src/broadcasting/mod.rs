//! # Broadcasting Module
//!
//! This module provides the infrastructure for streaming Remixatron's audio
//! and visualization data over the local network. It enables the "Infinite Broadcast"
//! feature, allowing Chromecast devices and web browsers to play the infinite remix.
//!
//! ## Architecture Overview
//!
//! The broadcasting system follows a producer-consumer pipeline:
//!
//! 1. **Audio Tap**: Raw PCM samples are captured *before* they are sent to the
//!    Kira audio engine. This Pre-Kira approach decouples the broadcast from
//!    Kira's internal mixer APIs, improving stability and maintainability.
//!
//! 2. **Transcoder**: A dedicated background thread reads PCM samples from a
//!    `crossbeam_channel`, encodes them to MP3 using LAME, and broadcasts the
//!    resulting bytes to all connected clients via a `tokio::sync::broadcast`.
//!
//! 3. **Web Server (Axum)**: An HTTP server provides endpoints for:
//!    - `/stream.mp3`: A live, chunked MP3 audio stream.
//!    - `/viz`: A WebSocket endpoint for real-time visualization data.
//!    - `/receiver`: Static files for the web-based Chromecast receiver.
//!
//! ## Usage
//!
//! The broadcasting pipeline is started during application initialization in
//! [`lib.rs`] and runs for the lifetime of the application. Audio is only
//! transmitted when a track is actively playing.

pub mod transcoder;
pub mod server;
// pub mod controller; // Phase 2: Chromecast Discovery and Control
