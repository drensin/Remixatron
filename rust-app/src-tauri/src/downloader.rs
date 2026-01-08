use tauri::AppHandle;
use std::path::PathBuf;
use anyhow::{Result, anyhow};
use std::fs;
use tauri::Manager;
use tauri::Emitter; 
use yt_dlp::Youtube;

/// Result of a download operation.
///
/// Contains the local file path and extracted metadata from the downloaded video.
/// This struct is used internally and may differ from the API-facing `VideoMetadata`.
pub struct DownloadResult {
    /// Path to the downloaded audio file on disk.
    pub file_path: PathBuf,
    /// Cleaned video title (redundant artist prefix removed).
    pub title: String,
    /// Channel/artist name from YouTube metadata.
    pub artist: String,
    /// Optional path to the downloaded thumbnail image.
    pub thumbnail_path: Option<PathBuf>,
}

/// Initializes the downloader by ensuring binaries exist and are up-to-date.
///
/// This function:
/// 1. Resolves the app's local data directory paths.
/// 2. Creates the `bin` and `downloads` directories if they don't exist.
/// 3. Downloads yt-dlp and ffmpeg binaries if missing.
/// 4. Updates yt-dlp to the latest version (prevents stale binary 403 errors).
///
/// # Arguments
/// * `app` - The Tauri application handle.
///
/// # Returns
/// The path to the bin directory containing the executables.
pub async fn init_downloader(app: AppHandle) -> Result<PathBuf> {
    // 1. Resolve Local Data Paths
    let app_data_dir = app.path().app_local_data_dir()
        .map_err(|e| anyhow!("Failed to get app data dir: {}", e))?;
    
    let bin_dir = app_data_dir.join("bin");
    let dl_dir = app_data_dir.join("downloads");

    // 2. Report Status
    let _ = app.emit("downloader_status", "Checking external tools...");
    
    // 2a. Cleanup Old Downloads (Startup Hygiene)
    let _ = cleanup_downloads(&app);

    // 3. Initialize logic using crate's auto-setup
    // This will download yt-dlp and ffmpeg if missing.
    if !bin_dir.exists() { fs::create_dir_all(&bin_dir)?; }
    if !dl_dir.exists() { fs::create_dir_all(&dl_dir)?; }

    let fetcher = Youtube::with_new_binaries(bin_dir.clone(), dl_dir)
        .await
        .map_err(|e| anyhow!("Failed to setup yt-dlp binaries: {}", e))?;

    // 4. Update yt-dlp to the latest version.
    // YouTube frequently changes their API, so stale yt-dlp versions fail with 403 errors.
    let _ = app.emit("downloader_status", "Updating yt-dlp...");
    if let Err(e) = fetcher.update_downloader().await {
        eprintln!("Warning: Failed to update yt-dlp: {}. Continuing with existing version.", e);
    }

    let _ = app.emit("downloader_status", "Downloader Ready!");
    
    Ok(bin_dir)
}

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
/// 2. Fetches video metadata (title, channel, thumbnail) using the yt-dlp crate.
/// 3. Downloads the audio by invoking the yt-dlp binary as a subprocess.
///
/// The subprocess approach is used because the crate's built-in download methods
/// use direct HTTP requests to stream URLs, which YouTube blocks with 403 Forbidden.
/// The yt-dlp binary handles signatures, throttling, and retries internally.
///
/// # Arguments
/// * `app` - The Tauri application handle for accessing paths and emitting events.
/// * `url` - The YouTube video URL to download.
///
/// # Returns
/// * `Ok(VideoMetadata)` - Contains path to audio file and extracted metadata.
/// * `Err` - If metadata fetch fails, binary is missing, or download fails.
///
/// # Emits
/// * `downloader_status` - Progress events for UI feedback.
pub async fn download_url(app: AppHandle, url: String) -> Result<VideoMetadata> {
    let app_data_dir = app.path().app_local_data_dir()
        .map_err(|e| anyhow!("Failed to get app data dir: {}", e))?;
    
    let bin_dir = app_data_dir.join("bin");
    let dl_dir = app_data_dir.join("downloads");

    let _ = app.emit("downloader_status", "Init Downloader...");

    // 0. Cleanup Previous Downloads
    // Safe to do because playback engine loads file into RAM.
    let _ = cleanup_downloads(&app);

    // 1. Get Fetcher (for metadata only - we use subprocess for download).
    let fetcher = Youtube::with_new_binaries(bin_dir.clone(), dl_dir.clone())
        .await
        .map_err(|e| anyhow!("Failed to load downloader: {}", e))?;

    let _ = app.emit("downloader_status", "Fetching metadata...");

    // 2. Fetch Metadata FIRST
    let video = fetcher.fetch_video_infos(url.clone())
        .await
        .map_err(|e| anyhow!("Failed to fetch video info: {}", e))?;

    let raw_title = video.title.clone();
    let artist = video.channel.clone();
    let thumbnail_url = video.thumbnail.clone();

    // 2b. Clean up title to remove redundant artist prefix.
    // Many YouTube videos are titled "Artist - Song Title", which leads to
    // "Artist - Artist - Song Title" when displayed in favorites.
    let title = clean_title(&raw_title, &artist);

    let _ = app.emit("downloader_status", "Downloading Audio...");

    // 3. Download Audio by running yt-dlp directly as a subprocess.
    // NOTE: The crate's download methods (download_audio_stream_with_quality,
    // download_format) use direct HTTP requests to stream URLs, which YouTube
    // blocks with 403 Forbidden. Running yt-dlp directly as a subprocess works
    // because yt-dlp handles signatures, throttling, and retries internally.
    let filename = format!("audio_{}.m4a", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_millis());
    let output_path = dl_dir.join(&filename);

    // Locate the yt-dlp binary in our bin directory.
    let ytdlp_bin = bin_dir.join("yt-dlp");
    if !ytdlp_bin.exists() {
        return Err(anyhow!("yt-dlp binary not found at {:?}", ytdlp_bin));
    }

    // Run yt-dlp directly to download audio only.
    // -f bestaudio: Select best audio format
    // -x: Extract audio (convert to audio-only)
    // --audio-format m4a: Output format
    // -o: Output path template
    let output = std::process::Command::new(&ytdlp_bin)
        .arg("-f")
        .arg("bestaudio")
        .arg("-x")
        .arg("--audio-format")
        .arg("m4a")
        .arg("-o")
        .arg(&output_path)
        .arg(&url)
        .output()
        .map_err(|e| anyhow!("Failed to execute yt-dlp: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("yt-dlp error: {}", stderr);
        return Err(anyhow!("yt-dlp failed: {}", stderr));
    }
        
    let _ = app.emit("downloader_status", "Download complete!");
    
    Ok(VideoMetadata {
        path: output_path.to_string_lossy().to_string(),
        title,
        artist,
        thumbnail_url: Some(thumbnail_url),
    })
}

/// Cleans up the downloads directory by removing all temporary files.
///
/// This function is called at startup and before each download to prevent
/// storage bloat from accumulated temporary audio files. It is safe to call
/// at any time because the playback engine loads audio files into RAM.
///
/// # Arguments
/// * `app` - The Tauri application handle for accessing the app data directory.
///
/// # Returns
/// * `Ok(())` - Cleanup completed (files removed or directory was already empty).
/// * `Err` - If the app data directory could not be resolved.
fn cleanup_downloads(app: &AppHandle) -> Result<()> {
    let app_data_dir = app.path().app_local_data_dir()
        .map_err(|e| anyhow!("Failed to get app data dir: {}", e))?;
    let dl_dir = app_data_dir.join("downloads");

    // Early return if directory doesn't exist yet.
    if !dl_dir.exists() {
        return Ok(());
    }

    // Iterate through directory and remove each file.
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
/// This function detects if the title starts with the artist name (case-insensitive)
/// and removes it along with any common separator characters (" - ", " – ", " | ", ": ").
///
/// # Arguments
/// * `raw_title` - The original video title from YouTube metadata.
/// * `artist` - The channel/artist name.
///
/// # Returns
/// A cleaned title with the redundant artist prefix removed, or the original title
/// if no redundancy was detected.
///
/// # Examples
/// - `clean_title("Peter Gabriel - Big Time", "Peter Gabriel")` → `"Big Time"`
/// - `clean_title("Big Time", "Peter Gabriel")` → `"Big Time"` (no change)
/// - `clean_title("PETER GABRIEL: In Your Eyes", "Peter Gabriel")` → `"In Your Eyes"`
fn clean_title(raw_title: &str, artist: &str) -> String {
    // Perform case-insensitive comparison.
    let title_lower = raw_title.to_lowercase();
    let artist_lower = artist.to_lowercase();

    // Check if the title starts with the artist name.
    if !title_lower.starts_with(&artist_lower) {
        // No redundancy detected; return original title.
        return raw_title.to_string();
    }

    // Remove the artist prefix.
    let remainder = &raw_title[artist.len()..];

    // Define common separator patterns to strip.
    // These are sorted by length (longest first) to ensure proper matching.
    let separators = [" - ", " – ", " — ", " | ", ": ", "-", "–", "—", "|", ":"];

    let mut cleaned = remainder;

    // Attempt to strip each separator pattern.
    for sep in separators {
        if let Some(stripped) = cleaned.strip_prefix(sep) {
            cleaned = stripped;
            break;
        }
    }

    // Trim any remaining leading/trailing whitespace.
    let cleaned = cleaned.trim();

    // Safety check: if we stripped everything or only have punctuation left,
    // fall back to the original title.
    if cleaned.is_empty() || !cleaned.chars().any(|c| c.is_alphanumeric()) {
        return raw_title.to_string();
    }

    cleaned.to_string()
}
