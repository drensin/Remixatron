//! YouTube Audio Downloader Module
//!
//! This module provides functionality to download audio from YouTube URLs using
//! system-installed `yt-dlp` and `ffmpeg` binaries. It does NOT bundle or manage
//! these binaries - users must install them separately.
//!
//! # Prerequisites
//! - `yt-dlp` must be in PATH (https://github.com/yt-dlp/yt-dlp)
//! - `ffmpeg` must be in PATH (https://ffmpeg.org)
//!
//! # Usage
//! 1. Call `check_dependencies()` at startup to verify tools are available.
//! 2. Call `download_url()` to download audio from a YouTube URL.

use std::process::Command;
use std::fs;
use anyhow::{Result, anyhow};
use tauri::AppHandle;
use tauri::Manager;
use tauri::Emitter;

// =============================================================================
// Dependency Checking
// =============================================================================

/// Status of required external dependencies (yt-dlp and ffmpeg).
///
/// This struct is returned to the frontend on startup so it can display
/// appropriate error dialogs if tools are missing.
#[derive(serde::Serialize, Clone, Debug)]
pub struct DependencyStatus {
    /// Version string if yt-dlp found, None if missing.
    pub ytdlp_version: Option<String>,
    /// Version string if ffmpeg found, None if missing.
    pub ffmpeg_version: Option<String>,
}

impl DependencyStatus {
    /// Returns true if all required dependencies are available.
    pub fn all_present(&self) -> bool {
        self.ytdlp_version.is_some() && self.ffmpeg_version.is_some()
    }
}

/// Checks if required external tools (yt-dlp and ffmpeg) are available in PATH.
///
/// This function runs each binary with its version flag and captures the output.
/// If the command succeeds, the tool is available; if it fails, it's missing.
///
/// # Returns
/// `DependencyStatus` with version strings for available tools, None for missing.
pub fn check_dependencies() -> DependencyStatus {
    DependencyStatus {
        ytdlp_version: get_binary_version("yt-dlp", "--version"),
        ffmpeg_version: get_binary_version("ffmpeg", "-version"),
    }
}

/// Runs a binary with a version flag and returns the version string if successful.
///
/// This is a cross-platform way to check if a binary is available in PATH.
/// Works on Windows, macOS, and Linux without needing `which` or `where`.
///
/// # Arguments
/// * `binary` - Name of the binary to check (e.g., "yt-dlp", "ffmpeg").
/// * `version_flag` - Flag to get version (e.g., "--version", "-version").
///
/// # Returns
/// * `Some(version)` - First line of stdout if command succeeded.
/// * `None` - If command failed (binary not found or not executable).
fn get_binary_version(binary: &str, version_flag: &str) -> Option<String> {
    // Use find_binary to locate the executable (handles AppImage PATH issues)
    let binary_path = find_binary(binary)?;
    
    // Clear AppImage's Python environment variables that break yt-dlp.
    // AppImages set PYTHONHOME/PYTHONPATH which causes Python to fail with
    // "No module named 'encodings'" when running external Python scripts.
    Command::new(&binary_path)
        .arg(version_flag)
        .env_remove("PYTHONHOME")
        .env_remove("PYTHONPATH")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| {
            String::from_utf8_lossy(&output.stdout)
                .lines()
                .next()
                .unwrap_or("")
                .trim()
                .to_string()
        })
        .filter(|s| !s.is_empty())
}

/// Finds a binary by checking PATH and common installation locations.
///
/// AppImages have a modified PATH that may not include user directories like
/// ~/.local/bin. This function checks common locations to ensure we can find
/// binaries installed via pip, brew, etc.
///
/// # Arguments
/// * `binary` - Name of the binary to find (e.g., "yt-dlp", "ffmpeg").
///
/// # Returns
/// * `Some(path)` - Full path to the binary if found.
/// * `None` - If binary not found in any location.
fn find_binary(binary: &str) -> Option<String> {
    use std::path::Path;
    
    // Build list of common binary locations to check.
    // AppImages have a modified PATH that may not include user directories.
    let mut paths_to_check: Vec<String> = Vec::new();
    
    // Add user-specific paths FIRST (pip installs here on Linux)
    // This is the most common location for yt-dlp
    if let Ok(home) = std::env::var("HOME") {
        paths_to_check.push(format!("{}/.local/bin", home));
    }
    
    // Standard system paths
    paths_to_check.push("/usr/local/bin".to_string());
    paths_to_check.push("/usr/bin".to_string());
    paths_to_check.push("/bin".to_string());
    
    // macOS Homebrew locations
    paths_to_check.push("/opt/homebrew/bin".to_string());
    
    // Check each path for the binary
    for dir in &paths_to_check {
        let full_path = format!("{}/{}", dir, binary);
        if Path::new(&full_path).exists() {
            return Some(full_path);
        }
    }
    
    // Last resort: try the binary name directly (uses current PATH)
    // This works in normal shells but may fail in AppImages
    if Command::new(binary)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok()
    {
        return Some(binary.to_string());
    }
    
    None
}

// =============================================================================
// Video Metadata and Download
// =============================================================================

/// Metadata returned from a successful download operation.
///
/// This struct is serialized and sent to the frontend via Tauri IPC.
/// It contains all information needed to display the track in the UI
/// and add it to favorites.
#[derive(serde::Serialize, Clone)]
pub struct VideoMetadata {
    /// Absolute path to the downloaded audio file.
    pub path: String,
    /// Cleaned video title (redundant artist prefix removed).
    pub title: String,
    /// Channel/artist name from YouTube metadata.
    pub artist: String,
    /// URL to the video thumbnail image for display in the UI.
    pub thumbnail_url: Option<String>,
}

/// Downloads audio from a YouTube URL and returns metadata.
///
/// This function:
/// 1. Cleans up previous downloads to prevent storage bloat.
/// 2. Fetches video metadata using `yt-dlp --dump-json`.
/// 3. Downloads the audio using `yt-dlp -x --audio-format m4a`.
/// 4. If bot detection is triggered, retries with browser cookies.
///
/// # Arguments
/// * `app` - The Tauri application handle for accessing paths and emitting events.
/// * `url` - The YouTube video URL to download.
///
/// # Returns
/// * `Ok(VideoMetadata)` - Contains path to audio file and extracted metadata.
/// * `Err` - If metadata fetch fails, binary is missing, or download fails.
///   The error message will indicate if this is likely a 403/outdated yt-dlp issue.
///
/// # Emits
/// * `downloader_status` - Progress events for UI feedback.
/// * `download_metadata` - Early metadata for immediate UI update.
pub async fn download_url(app: AppHandle, url: String) -> Result<VideoMetadata> {
    let app_data_dir = app.path().app_local_data_dir()
        .map_err(|e| anyhow!("Failed to get app data dir: {}", e))?;
    
    let dl_dir = app_data_dir.join("downloads");

    // Ensure downloads directory exists.
    if !dl_dir.exists() {
        fs::create_dir_all(&dl_dir)?;
    }

    let _ = app.emit("downloader_status", "Preparing download...");

    // 0. Cleanup Previous Downloads
    // Safe to do because playback engine loads file into RAM.
    let _ = cleanup_downloads(&app);

    // 1. Fetch Metadata using yt-dlp --dump-json
    let _ = app.emit("downloader_status", "Fetching metadata...");
    
    // Find yt-dlp binary (handles AppImage PATH issues)
    let ytdlp_path = find_binary("yt-dlp")
        .ok_or_else(|| anyhow!("yt-dlp not found. Please install it and ensure it's in your PATH."))?;

    // Find ffmpeg binary (needed for post-processing)
    let ffmpeg_path = find_binary("ffmpeg");
    
    // List of browsers to try for cookie extraction (in order of popularity).
    // yt-dlp will try to extract cookies from the first available browser.
    let browsers_to_try = ["chrome", "firefox", "edge", "brave", "chromium", "safari", "opera"];
    
    // Helper to configure command environment
    let configure_command = |cmd: &mut Command| {
        // If we found binaries in a non-standard location, add that directory to PATH
        // so yt-dlp can find other tools (node, deno, atomicparsley, etc).
        let ytdlp_dir = std::path::Path::new(&ytdlp_path).parent();
        if let Some(dir) = ytdlp_dir {
             if let Some(path_str) = dir.to_str() {
                 if let Ok(current_path) = std::env::var("PATH") {
                     let new_path = format!("{}:{}", path_str, current_path);
                     cmd.env("PATH", new_path);
                 } else {
                     cmd.env("PATH", path_str);
                 }
             }
        }
        
        // Remove interfering Python env vars
        cmd.env_remove("PYTHONHOME")
           .env_remove("PYTHONPATH");
    };

    // --- Helper to run metadata fetch (with optional browser cookies) ---
    let fetch_metadata = |browser: Option<&str>| -> Result<(String, String, Option<String>)> {
        let mut cmd = Command::new(&ytdlp_path);
        configure_command(&mut cmd);
        
        cmd.arg("--dump-json")
           .arg("--no-download");
        
        // Add browser cookies if specified
        if let Some(b) = browser {
            cmd.arg("--cookies-from-browser").arg(b);
        }
        
        cmd.arg(&url);
        
        let output = cmd.output()
            .map_err(|e| anyhow!("Failed to run yt-dlp: {}", e))?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(create_download_error(&stderr));
        }
        
        let json_str = String::from_utf8_lossy(&output.stdout);
        let metadata: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| anyhow!("Failed to parse yt-dlp metadata: {}", e))?;
        
        let raw_title = metadata["title"].as_str().unwrap_or("Unknown Title").to_string();
        let artist = metadata["channel"].as_str()
            .or_else(|| metadata["uploader"].as_str())
            .unwrap_or("Unknown Artist")
            .to_string();
        let thumbnail_url = metadata["thumbnail"].as_str().map(|s| s.to_string());
        
        Ok((raw_title, artist, thumbnail_url))
    };

    // --- 1. Try metadata fetch without cookies first ---
    let (raw_title, artist, thumbnail_url, working_browser) = match fetch_metadata(None) {
        Ok((title, artist, thumb)) => (title, artist, thumb, None),
        Err(e) => {
            let err_str = e.to_string();
            // Check if this is a bot detection error that cookies might fix
            if err_str.contains("Sign in to confirm") || err_str.contains("bot") {
                let _ = app.emit("downloader_status", "Bot detected, trying browser cookies...");
                
                // Try each browser until one works
                let mut success = None;
                for browser in &browsers_to_try {
                    let _ = app.emit("downloader_status", format!("Trying {} cookies...", browser));
                    match fetch_metadata(Some(browser)) {
                        Ok((title, artist, thumb)) => {
                            println!("Successfully fetched metadata using {} cookies", browser);
                            success = Some((title, artist, thumb, Some(browser.to_string())));
                            break;
                        }
                        Err(e) => {
                            // If this browser failed with a different error, note it
                            let err_str = e.to_string();
                            if !err_str.contains("could not find") && !err_str.contains("no cookies") {
                                // This browser exists but still failed - might be a different issue
                                eprintln!("Browser {} failed: {}", browser, err_str);
                            }
                            // Continue to next browser
                        }
                    }
                }
                
                match success {
                    Some((t, a, th, b)) => (t, a, th, b),
                    None => return Err(anyhow!("Bot detection triggered. No browser cookies found. \
                        Try logging into YouTube in Chrome or Firefox.")),
                }
            } else {
                // Not a bot detection error, propagate original error
                return Err(e);
            }
        }
    };

    // 2. Clean up title to remove redundant artist prefix.
    let title = clean_title(&raw_title, &artist);

    // 3. Emit metadata immediately so the UI can update while downloading.
    let early_metadata = serde_json::json!({
        "title": title,
        "artist": artist,
        "thumbnail_url": thumbnail_url
    });
    let _ = app.emit("download_metadata", early_metadata);

    // 4. Download Audio
    let _ = app.emit("downloader_status", "Downloading audio...");
    
    let filename = format!(
        "audio_{}.m4a",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_millis()
    );
    let output_path = dl_dir.join(&filename);

    // --- 5. Run Download Command (with same browser cookies if they worked) ---
    let mut download_cmd = Command::new(&ytdlp_path);
    configure_command(&mut download_cmd);

    if let Some(ff_path) = &ffmpeg_path {
        // Explicitly tell yt-dlp where ffmpeg is
        download_cmd.arg("--ffmpeg-location").arg(ff_path);
    }
    
    // Use the same browser cookies that worked for metadata
    if let Some(ref browser) = working_browser {
        download_cmd.arg("--cookies-from-browser").arg(browser);
    }

    let download_output = download_cmd
        .arg("-f")
        .arg("bestaudio")
        .arg("-x")
        .arg("--audio-format")
        .arg("m4a")
        .arg("-o")
        .arg(&output_path)
        .arg(&url)
        .output()
        .map_err(|e| anyhow!("Failed to run yt-dlp: {}", e))?;

    if !download_output.status.success() {
        let stderr = String::from_utf8_lossy(&download_output.stderr);
        eprintln!("yt-dlp download error: {}", stderr);
        return Err(create_download_error(&stderr));
    }

    let _ = app.emit("downloader_status", "Download complete!");

    Ok(VideoMetadata {
        path: output_path.to_string_lossy().to_string(),
        title,
        artist,
        thumbnail_url,
    })
}

/// Creates an appropriate error message based on yt-dlp stderr output.
///
/// Detects common error patterns (403, HTTP errors) and returns a user-friendly
/// message suggesting the issue may be an outdated yt-dlp.
fn create_download_error(stderr: &str) -> anyhow::Error {
    let is_likely_outdated = stderr.contains("403")
        || stderr.contains("HTTP Error")
        || stderr.contains("Unable to extract")
        || stderr.contains("Sign in to confirm");

    if is_likely_outdated {
        anyhow!("OUTDATED_YTDLP: {}", stderr.trim())
    } else {
        anyhow!("Download failed: {}", stderr.trim())
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Cleans up the downloads directory by removing all temporary files.
///
/// This function is called before each download to prevent storage bloat.
/// It is safe to call at any time because the playback engine loads files into RAM.
fn cleanup_downloads(app: &AppHandle) -> Result<()> {
    let app_data_dir = app.path().app_local_data_dir()
        .map_err(|e| anyhow!("Failed to get app data dir: {}", e))?;
    let dl_dir = app_data_dir.join("downloads");

    if !dl_dir.exists() {
        return Ok(());
    }

    for entry in fs::read_dir(dl_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Err(e) = fs::remove_file(&path) {
                eprintln!("Failed to remove temp file {:?}: {}", path, e);
            }
        }
    }

    Ok(())
}

/// Cleans a video title by removing a redundant artist prefix.
///
/// Many YouTube videos are titled in the format "Artist - Song Title" while the
/// channel name (artist) is also available separately. When displayed as
/// "Artist - Artist - Song Title" in favorites, this is redundant and confusing.
///
/// # Arguments
/// * `raw_title` - The original video title from YouTube metadata.
/// * `artist` - The channel/artist name.
///
/// # Returns
/// A cleaned title with the redundant artist prefix removed.
///
/// # Examples
/// - `clean_title("Peter Gabriel - Big Time", "Peter Gabriel")` → `"Big Time"`
/// - `clean_title("Big Time", "Peter Gabriel")` → `"Big Time"` (no change)
fn clean_title(raw_title: &str, artist: &str) -> String {
    let title_lower = raw_title.to_lowercase();
    let artist_lower = artist.to_lowercase();

    if !title_lower.starts_with(&artist_lower) {
        return raw_title.to_string();
    }

    let remainder = &raw_title[artist.len()..];

    let separators = [" - ", " – ", " — ", " | ", ": ", "-", "–", "—", "|", ":"];
    let mut cleaned = remainder;

    for sep in separators {
        if let Some(stripped) = cleaned.strip_prefix(sep) {
            cleaned = stripped;
            break;
        }
    }

    let cleaned = cleaned.trim();

    if cleaned.is_empty() || !cleaned.chars().any(|c| c.is_alphanumeric()) {
        return raw_title.to_string();
    }

    cleaned.to_string()
}
