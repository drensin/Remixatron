use tauri::AppHandle;
use std::path::PathBuf;
use anyhow::{Result, anyhow};
use std::fs;
use tauri::Manager;
use tauri::Emitter; 
use yt_dlp::Youtube;

pub struct DownloadResult {
    pub file_path: PathBuf,
    pub title: String,
    pub artist: String,
    pub thumbnail_path: Option<PathBuf>,
}

/// Initializes the downloader by ensuring binaries exist.
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

    let _fetcher = Youtube::with_new_binaries(bin_dir.clone(), dl_dir)
        .await
        .map_err(|e| anyhow!("Failed to setup yt-dlp binaries: {}", e))?;

    let _ = app.emit("downloader_status", "Downloader Ready!");
    
    Ok(bin_dir)
}

/// Downloads a URL and returns the local path to the audio file.
#[derive(serde::Serialize, Clone)]
pub struct VideoMetadata {
    pub path: String,
    pub title: String,
    pub artist: String,
    pub thumbnail_url: Option<String>,
}

pub async fn download_url(app: AppHandle, url: String) -> Result<VideoMetadata> {
    let app_data_dir = app.path().app_local_data_dir()
        .map_err(|e| anyhow!("Failed to get app data dir: {}", e))?;
    
    let bin_dir = app_data_dir.join("bin");
    let dl_dir = app_data_dir.join("downloads");
    
    let _ = app.emit("downloader_status", "Init Downloader...");

    let _ = app.emit("downloader_status", "Init Downloader...");

    // 0. Cleanup Previous Downloads
    // Safe to do because playback engine loads file into RAM.
    let _ = cleanup_downloads(&app);

    // 1. Get Fetcher
    let fetcher = Youtube::with_new_binaries(bin_dir, dl_dir.clone())
        .await
        .map_err(|e| anyhow!("Failed to load downloader: {}", e))?;

    let _ = app.emit("downloader_status", "Fetching metadata...");

    // 2. Fetch Metadata FIRST
    let video = fetcher.fetch_video_infos(url.clone())
        .await
        .map_err(|e| anyhow!("Failed to fetch video info: {}", e))?;

    let raw_title = video.title;
    let artist = video.channel; // Field is 'channel', not 'uploader', and it is a String (not Option)
    let thumbnail_url = video.thumbnail;

    // 2b. Clean up title to remove redundant artist prefix.
    // Many YouTube videos are titled "Artist - Song Title", which leads to
    // "Artist - Artist - Song Title" when displayed in favorites.
    let title = clean_title(&raw_title, &artist);

    let _ = app.emit("downloader_status", "Downloading Audio...");

    // 3. Download Audio
    let filename = format!("audio_{}.m4a", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_millis());
    
    use yt_dlp::model::selector::{AudioQuality, AudioCodecPreference};

    fetcher.download_audio_stream_with_quality(
        url,
        &filename,
        AudioQuality::Best,
        AudioCodecPreference::AAC
    ).await
    .map_err(|e| anyhow!("Download failed: {}", e))?;
        
    let _ = app.emit("downloader_status", "Download complete!");
    
    Ok(VideoMetadata {
        path: dl_dir.join(filename).to_string_lossy().to_string(),
        title,
        artist,
        thumbnail_url: Some(thumbnail_url),
    })
}

/// Cleans up the downloads directory by removing all files.
/// This prevents storage bloat from accumulated temporary files.
fn cleanup_downloads(app: &AppHandle) -> Result<()> {
    let app_data_dir = app.path().app_local_data_dir()
        .map_err(|e| anyhow!("Failed to get app data dir: {}", e))?;
    let dl_dir = app_data_dir.join("downloads");

    if !dl_dir.exists() {
        return Ok(());
    }

    let mut count = 0;
    for entry in fs::read_dir(dl_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Err(e) = fs::remove_file(&path) {
                eprintln!("Failed to remove temp file {:?}: {}", path, e);
            } else {
                count += 1;
            }
        }
    }
    
    if count > 0 {
        println!("Cleaned up {} temporary files from downloads.", count);
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
