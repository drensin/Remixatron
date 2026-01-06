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

    // 1. Get Fetcher
    let fetcher = Youtube::with_new_binaries(bin_dir, dl_dir.clone())
        .await
        .map_err(|e| anyhow!("Failed to load downloader: {}", e))?;

    let _ = app.emit("downloader_status", "Fetching metadata...");

    // 2. Fetch Metadata FIRST
    let video = fetcher.fetch_video_infos(url.clone())
        .await
        .map_err(|e| anyhow!("Failed to fetch video info: {}", e))?;

    let title = video.title;
    let artist = video.channel; // Field is 'channel', not 'uploader', and it is a String (not Option)
    let thumbnail_url = video.thumbnail;

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
