//! # Broadcast Web Server
//!
//! This module implements an HTTP server using Axum that provides endpoints for
//! streaming audio and visualization data to remote clients. It is designed to
//! support both Chromecast receivers and standard web browsers.
//!
//! ## Endpoints
//!
//! | Path | Description |
//! |------|-------------|
//! | `/stream.mp3` | Live MP3 audio stream (chunked transfer encoding) |
//! | `/viz` | WebSocket for real-time visualization updates |
//! | `/receiver` | Static files for the Chromecast web receiver |
//!
//! ## Network Binding
//!
//! The server binds to `0.0.0.0:3030`, making it accessible from any device on
//! the local network. This is intentional to support the Cast workflow where
//! the TV fetches audio from the desktop's IP address.

use axum::{
    body::Body,
    extract::{
        ws::{Message, WebSocket},
        State, WebSocketUpgrade,
    },
    response::{IntoResponse, Response},
    routing::get,
    Router,
};
use bytes::Bytes;
use futures::stream::StreamExt;
use serde::Serialize;
use std::net::SocketAddr;
use tokio::sync::{broadcast, watch};
use tower_http::services::ServeDir;

/// The port on which the broadcast server listens.
const SERVER_PORT: u16 = 3030;

/// Static visualization data sent once when a client connects.
///
/// This contains the full remix graph structure needed to render the visualization.
#[derive(Clone, Default, Serialize)]
pub struct VizInitData {
    /// All beats with their jump candidates.
    pub beats: Vec<serde_json::Value>,
    /// Segment boundaries and labels.
    pub segments: Vec<serde_json::Value>,
    /// Waveform amplitude envelope (720 samples).
    pub waveform: Vec<f32>,
    /// Track title for display in receiver UI.
    pub title: String,
    /// Artist name for display in receiver UI.
    pub artist: String,
    /// Thumbnail image (base64 data URI for local files, URL for YouTube).
    pub thumbnail: String,
}

/// Dynamic playback state sent on every beat.
#[derive(Clone, Default, Serialize)]
pub struct VizUpdateData {
    /// Index of the currently playing beat.
    pub active_beat: usize,
    /// Index of the current segment.
    pub active_seg: usize,
    /// Position in current sequence (for countdown).
    pub seq_pos: usize,
    /// Length of current sequence.
    pub seq_len: usize,
    /// Cumulative stream time in seconds since playback started.
    /// Client uses this with audio.currentTime for sync.
    pub stream_time: f32,
    /// True when playback has stopped.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub stopped: bool,
    /// True when playback is paused.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub paused: bool,
}

/// Shared state passed to all request handlers.
#[derive(Clone)]
struct AppState {
    /// Sender for the MP3 broadcast channel.
    mp3_tx: broadcast::Sender<Bytes>,
    /// Watch receiver for static viz data (init message).
    viz_init_rx: watch::Receiver<VizInitData>,
    /// Broadcast receiver for dynamic viz updates.
    viz_update_tx: broadcast::Sender<VizUpdateData>,
}

/// Starts the broadcast web server.
///
/// This function creates an Axum router with all broadcast endpoints and binds
/// it to `0.0.0.0:3030`. It runs indefinitely until the application terminates.
///
/// # Arguments
///
/// * `mp3_tx` - Broadcast sender for encoded MP3 audio chunks.
/// * `viz_init_rx` - Watch receiver for static visualization data.
/// * `viz_update_tx` - Broadcast sender for dynamic playback updates.
///
/// # Panics
///
/// Panics if the TCP listener fails to bind (e.g., port already in use).
pub async fn start_server(
    mp3_tx: broadcast::Sender<Bytes>,
    viz_init_rx: watch::Receiver<VizInitData>,
    viz_update_tx: broadcast::Sender<VizUpdateData>,
) {
    let state = AppState {
        mp3_tx,
        viz_init_rx,
        viz_update_tx,
    };

    // Build the router with all broadcast endpoints.
    let app = Router::new()
        .route("/stream.mp3", get(handle_audio_stream))
        .route("/viz", get(handle_websocket))
        .nest_service("/receiver", ServeDir::new("../src-receiver"))
        .with_state(state);

    // Bind to all interfaces on the configured port.
    let addr = SocketAddr::from(([0, 0, 0, 0], SERVER_PORT));
    println!("[Broadcast Server] Listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind broadcast server port");

    axum::serve(listener, app)
        .await
        .expect("Broadcast server terminated unexpectedly");
}

/// Handles requests to `/stream.mp3`.
///
/// Returns a chunked HTTP response that streams live MP3 audio.
async fn handle_audio_stream(State(state): State<AppState>) -> Response {
    let rx = state.mp3_tx.subscribe();

    let stream = tokio_stream::wrappers::BroadcastStream::new(rx).filter_map(|result| async move {
        match result {
            Ok(bytes) => Some(Ok::<_, std::io::Error>(bytes)),
            Err(_) => None,
        }
    });

    let body = Body::from_stream(stream);
    ([(axum::http::header::CONTENT_TYPE, "audio/mpeg")], body).into_response()
}

/// Handles WebSocket upgrade requests to `/viz`.
///
/// On connection:
/// 1. Sends an `init` message with full visualization structure.
/// 2. Streams `update` messages with current playback position.
async fn handle_websocket(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> Response {
    ws.on_upgrade(move |socket| handle_viz_socket(socket, state))
}

/// Manages a single WebSocket connection for visualization updates.
async fn handle_viz_socket(mut socket: WebSocket, state: AppState) {
    // Helper to build init message from VizInitData
    let build_init_msg = |data: &VizInitData| -> serde_json::Value {
        serde_json::json!({
            "type": "init",
            "beats": data.beats,
            "segments": data.segments,
            "waveform": data.waveform,
            "title": data.title,
            "artist": data.artist,
            "thumbnail": data.thumbnail,
        })
    };

    // 1. Send initial data
    let init_data = state.viz_init_rx.borrow().clone();
    if let Ok(text) = serde_json::to_string(&build_init_msg(&init_data)) {
        let _ = socket.send(Message::Text(text.into())).await;
    }

    // 2. Subscribe to updates and forward to client
    let mut update_rx = state.viz_update_tx.subscribe();
    let mut viz_init_rx = state.viz_init_rx.clone();

    loop {
        tokio::select! {
            // Forward beat updates to client
            Ok(update) = update_rx.recv() => {
                let msg = serde_json::json!({
                    "type": "update",
                    "active_beat": update.active_beat,
                    "active_seg": update.active_seg,
                    "seq_pos": update.seq_pos,
                    "seq_len": update.seq_len,
                    "stream_time": update.stream_time,
                });

                if let Ok(text) = serde_json::to_string(&msg) {
                    if socket.send(Message::Text(text.into())).await.is_err() {
                        break; // Client disconnected
                    }
                }
            }
            // Forward new track init data to client (when user loads a new track)
            Ok(_) = viz_init_rx.changed() => {
                let new_init = viz_init_rx.borrow().clone();
                // Only send if there's actual content (not empty default)
                if !new_init.beats.is_empty() {
                    if let Ok(text) = serde_json::to_string(&build_init_msg(&new_init)) {
                        if socket.send(Message::Text(text.into())).await.is_err() {
                            break; // Client disconnected
                        }
                    }
                }
            }
            // Handle client messages (ping/pong, close)
            Some(msg) = socket.recv() => {
                match msg {
                    Ok(Message::Close(_)) => break,
                    Err(_) => break,
                    _ => {} // Ignore other messages
                }
            }
        }
    }
}
