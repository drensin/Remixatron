//! # Favorites Module
//!
//! This module provides persistent storage and retrieval of user-saved favorite tracks.
//! Favorites are stored as a JSON file in the OS-standard application data directory.
//!
//! ## Data Model
//! Each favorite consists of:
//! - `source`: The file path (for local files) or URL (for streaming sources).
//! - `artist`: The artist name for display purposes.
//! - `title`: The track title for display purposes.
//!
//! ## Storage Location
//! - **macOS**: `~/Library/Application Support/com.remixatron.dev/favorites.json`
//! - **Windows**: `C:\Users\<User>\AppData\Roaming\com.remixatron.dev\favorites.json`
//! - **Linux**: `~/.local/share/com.remixatron.dev/favorites.json`
//!
//! ## Uniqueness
//! The `source` field serves as the primary key. Duplicate sources are not permitted;
//! adding a favorite with an existing source will update the existing entry (upsert).

use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;
use tauri::AppHandle;
use tauri::Manager;

/// The filename for the favorites JSON file within the app data directory.
const FAVORITES_FILENAME: &str = "favorites.json";

/// Represents a single favorite track saved by the user.
///
/// # Fields
/// - `source`: The file path (for local audio) or URL (for streaming sources like YouTube).
///   This field serves as the unique identifier for the favorite.
/// - `artist`: The artist name, used for display and sorting.
/// - `title`: The track title, used for display and sorting.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Favorite {
    /// The file path or URL of the audio source. Acts as the primary key.
    pub source: String,
    /// The artist name for display purposes.
    pub artist: String,
    /// The track title for display purposes.
    pub title: String,
}

/// Resolves the full path to the favorites JSON file.
///
/// # Arguments
/// * `app` - The Tauri application handle, used to locate the app data directory.
///
/// # Returns
/// The `PathBuf` pointing to `favorites.json` within the app data directory.
///
/// # Panics
/// Panics if the app data directory cannot be resolved (should not happen in normal operation).
fn get_favorites_path(app: &AppHandle) -> PathBuf {
    // Resolve the OS-standard application data directory.
    let app_data_dir = app
        .path()
        .app_data_dir()
        .expect("Failed to resolve app data directory");

    // Ensure the directory exists. Create it if necessary.
    if !app_data_dir.exists() {
        fs::create_dir_all(&app_data_dir).expect("Failed to create app data directory");
    }

    app_data_dir.join(FAVORITES_FILENAME)
}

/// Loads all favorites from the JSON file.
///
/// If the file does not exist or is corrupted, an empty vector is returned.
/// This ensures the application never crashes due to missing or invalid favorites data.
///
/// # Arguments
/// * `app` - The Tauri application handle.
///
/// # Returns
/// A vector of `Favorite` objects, sorted by artist (ascending), then by title (ascending).
pub fn load_favorites(app: &AppHandle) -> Vec<Favorite> {
    let path = get_favorites_path(app);

    // If the file doesn't exist, return an empty list.
    if !path.exists() {
        return Vec::new();
    }

    // Attempt to open and parse the JSON file.
    let file = match File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Warning: Failed to open favorites file: {}", e);
            return Vec::new();
        }
    };

    let reader = BufReader::new(file);
    let mut favorites: Vec<Favorite> = match serde_json::from_reader(reader) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Warning: Failed to parse favorites JSON: {}", e);
            return Vec::new();
        }
    };

    // Sort by artist (primary), then by title (secondary), both ascending.
    favorites.sort_by(|a, b| {
        let artist_cmp = a.artist.to_lowercase().cmp(&b.artist.to_lowercase());
        if artist_cmp == std::cmp::Ordering::Equal {
            a.title.to_lowercase().cmp(&b.title.to_lowercase())
        } else {
            artist_cmp
        }
    });

    favorites
}

/// Saves the given list of favorites to the JSON file.
///
/// This overwrites the existing file completely. The list should be the full set of favorites.
///
/// # Arguments
/// * `app` - The Tauri application handle.
/// * `favorites` - The complete list of favorites to persist.
///
/// # Errors
/// Returns an error if the file cannot be written.
fn save_favorites(app: &AppHandle, favorites: &[Favorite]) -> Result<(), String> {
    let path = get_favorites_path(app);

    // Open the file for writing (create or truncate).
    let file = File::create(&path).map_err(|e| format!("Failed to create favorites file: {}", e))?;

    let writer = BufWriter::new(file);

    // Serialize the favorites to JSON with pretty-printing for human readability.
    serde_json::to_writer_pretty(writer, favorites)
        .map_err(|e| format!("Failed to write favorites JSON: {}", e))?;

    Ok(())
}

/// Adds a new favorite or updates an existing one (upsert behavior).
///
/// If a favorite with the same `source` already exists, its `artist` and `title` are updated.
/// Otherwise, a new favorite is appended to the list.
///
/// # Arguments
/// * `app` - The Tauri application handle.
/// * `source` - The file path or URL of the track.
/// * `artist` - The artist name.
/// * `title` - The track title.
///
/// # Returns
/// The `Favorite` object that was added or updated.
///
/// # Errors
/// Returns an error if the favorites file cannot be saved.
pub fn add_favorite(
    app: &AppHandle,
    source: String,
    artist: String,
    title: String,
) -> Result<Favorite, String> {
    let mut favorites = load_favorites(app);

    let new_fav = Favorite {
        source: source.clone(),
        artist,
        title,
    };

    // Check if this source already exists (upsert logic).
    if let Some(existing) = favorites.iter_mut().find(|f| f.source == source) {
        // Update existing entry with new metadata.
        existing.artist = new_fav.artist.clone();
        existing.title = new_fav.title.clone();
    } else {
        // Add as a new favorite.
        favorites.push(new_fav.clone());
    }

    // Persist the updated list.
    save_favorites(app, &favorites)?;

    Ok(new_fav)
}

/// Removes a favorite by its source identifier.
///
/// If no favorite with the given source exists, this is a no-op.
///
/// # Arguments
/// * `app` - The Tauri application handle.
/// * `source` - The file path or URL of the track to remove.
///
/// # Errors
/// Returns an error if the favorites file cannot be saved.
pub fn remove_favorite(app: &AppHandle, source: &str) -> Result<(), String> {
    let mut favorites = load_favorites(app);

    // Filter out the favorite with the matching source.
    let original_len = favorites.len();
    favorites.retain(|f| f.source != source);

    // Only save if something was actually removed.
    if favorites.len() < original_len {
        save_favorites(app, &favorites)?;
    }

    Ok(())
}

/// Checks whether a given source is already saved as a favorite.
///
/// # Arguments
/// * `app` - The Tauri application handle.
/// * `source` - The file path or URL to check.
///
/// # Returns
/// `true` if the source exists in the favorites list, `false` otherwise.
pub fn is_favorite(app: &AppHandle, source: &str) -> bool {
    let favorites = load_favorites(app);
    favorites.iter().any(|f| f.source == source)
}
