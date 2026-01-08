/**
 * @fileoverview Main entry point for the Remixatron Frontend.
 * 
 * This file orchestrates the interaction between the UI (HTML), the visualization logic (Viz.js),
 * and the Rust backend (Tauri). It handles user input, manages the download/analysis pipeline,
 * and coordinates the infinite playback loop.
 * 
 * @author Remixatron Team
 */

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;
import { InfiniteJukeboxViz } from './viz.js';

// Global Vars
let viz;
let statusEl;
let analyzeBtn;
let pathInput;
let remoteMetadata = null; // Defined globally so listener can see it
let isPaused = false; // Track playback state

// UI Elements (New M3 Structure)
let onboardingCard;
let floatingPlayer;
let stopBtn;
let pauseBtn;
let trackTitleEl;
let albumArtImg;
let albumArtPlaceholder;

// Favorites State
let favoriteBtn;
let favoritesContainer;
let favoritesToggle;
let favoritesDropdown;
let favoritesSearch;
let favoritesList;
let allFavorites = [];          // Full list from backend
let currentTrackSource = null;  // Track source for the currently playing track
let currentTrackArtist = null;  // Artist for the currently playing track
let currentTrackTitle = null;   // Title for the currently playing track

// Undo Toast State
let undoToast;
let toastMessage;
let toastUndoBtn;
let pendingDelete = null;       // Favorite awaiting permanent deletion
let toastTimeout = null;        // Timer for auto-commit of delete

/**
 * Initiates the Remixatron workflow.
 * 
 * This async function handles the full lifecycle of a new track:
 * 1. detects input type (Local File vs URL).
 * 2. (Optional) Downloads audio via the backend.
 * 3. Extracts and displays metadata (Title, Art).
 * 4. Invokes the backend analysis pipeline.
 * 5. Initializes the visualization.
 * 6. Starts the audio playback.
 * 
 * @returns {Promise<void>}
 */
async function startRemix() {
    let path = pathInput.value;
    if (!path) return;

    try {
        // UI Transition: Zero -> Loading
        onboardingCard.classList.add("hidden");
        floatingPlayer.classList.add("visible");
        floatingPlayer.classList.add("loading"); // Start Progress Bar
        analyzeBtn.disabled = true;

        // Clear previous visualization state
        viz.clear();

        // UI: Reset Pause State
        isPaused = false;
        if (pauseBtn) {
            const icon = pauseBtn.querySelector(".material-symbols-outlined");
            if (icon) icon.textContent = "pause_circle";
        }

        // UI: Reset Metadata to Loading State initially
        const filename = path.split(/[/\\]/).pop();
        if (trackTitleEl) trackTitleEl.textContent = `Loading...`;
        if (albumArtImg) albumArtImg.classList.add("hidden");
        if (albumArtPlaceholder) albumArtPlaceholder.classList.remove("hidden");

        remoteMetadata = null;

        // --- Early Favorites State ---
        // Set the track source immediately so heart state is correct during loading.
        // Artist/Title will be updated once metadata is available.
        currentTrackSource = pathInput.value;
        currentTrackArtist = null;
        currentTrackTitle = null;
        updateFavoriteButtonState();

        // 0. Handle URLs (Download First)
        if (path.startsWith("http://") || path.startsWith("https://")) {
            statusEl.textContent = "Initializing Downloader...";
            try {
                console.log("Calling import_url...");
                const metadata = await invoke("import_url", { url: path });
                console.log("Metadata received!", metadata);

                remoteMetadata = metadata; // Save for later
                path = metadata.path;

                // UI: Update Metadata Immediately (Step 0b)
                if (trackTitleEl) {
                    console.log("Updating Title to:", metadata.title);
                    trackTitleEl.textContent = metadata.title;
                }

                // UI: Update Album Art from URL
                if (metadata.thumbnail_url && albumArtImg) {
                    console.log("Updating Art URL to:", metadata.thumbnail_url);
                    albumArtImg.src = metadata.thumbnail_url;
                    albumArtImg.classList.remove("hidden");
                    if (albumArtPlaceholder) albumArtPlaceholder.classList.add("hidden");
                }

                // Update artist/title for favorites now that we have metadata.
                currentTrackArtist = metadata.artist || "Unknown Artist";
                currentTrackTitle = metadata.title || "Unknown Track";
            } catch (e) {
                console.error("Download Step Failed:", e);
                throw "Download Failed: " + e;
            }
        } else {
            // Local file: Show filename
            if (trackTitleEl) trackTitleEl.textContent = `Loading ${filename}...`;
        }

        statusEl.textContent = "Stopping previous playback...";
        try {
            await invoke("stop_playback");
        } catch (e) {
            console.log("Stop playback harmless error:", e);
        }

        statusEl.textContent = "Analyzing Audio Structure...";
        analyzeBtn.disabled = true;

        // 1. Analyze
        const payload = await invoke("analyze_track", { path });
        console.log("Analysis Complete!", payload);
        statusEl.textContent = "Structure Decoded. Starting Walk...";

        // 1b. Update Metadata (Merge Remote with Local)
        // If we have remote metadata, use it. Otherwise use payload.
        if (remoteMetadata) {
            // Keep the title we already set
        } else {
            if (trackTitleEl) trackTitleEl.textContent = payload.title;

            // 1c. Update Album Art (Local Only)
            if (payload.album_art_base64 && albumArtImg) {
                albumArtImg.src = payload.album_art_base64;
                albumArtImg.classList.remove("hidden");
                if (albumArtPlaceholder) albumArtPlaceholder.classList.add("hidden");
            }
        }

        // 2. Setup Viz
        viz.setData(payload);

        // 3. Play
        await invoke("play_track", { path });

        // UI Transition: Loading -> Active
        statusEl.textContent = "Infinite Walk Active";
        floatingPlayer.classList.remove("loading"); // Hide Progress Bar
        analyzeBtn.disabled = false;

        // --- Favorites: Update artist/title for local files ---
        // For URLs, these were set after download. For local files, set now.
        if (!remoteMetadata) {
            currentTrackArtist = "Unknown Artist";
            currentTrackTitle = payload.title || "Unknown Track";
        }

    } catch (e) {
        console.error(e);
        statusEl.textContent = "Error: " + e;
        analyzeBtn.disabled = false;

        // UI Transition: Revert on fatal error?
        // Keep player visible so error can be seen
        floatingPlayer.classList.remove("loading");
    }
}

/**
 * Stops the current playback and resets the UI to the "Ready" state.
 * 
 * This function handles safety cleanup:
 * 1. Signals the backend to stop the audio engine.
 * 2. Hides the floating player.
 * 3. Resets internal playback state (Pause flags).
 * 4. Shows the Onboarding Card after a transition delay.
 * 
 * @returns {Promise<void>}
 */
async function stopRemix() {
    try {
        await invoke("stop_playback");

        // UI Transition: Active -> Zero
        floatingPlayer.classList.remove("visible");

        // Reset State
        isPaused = false;
        // Reset State
        isPaused = false;
        if (pauseBtn) {
            const icon = pauseBtn.querySelector(".material-symbols-outlined");
            if (icon) icon.textContent = "pause_circle";
        }

        // Wait for player to slide down before showing card
        setTimeout(() => {
            onboardingCard.classList.remove("hidden");
            statusEl.textContent = "Ready";
        }, 300);

        // Reset favorites state.
        currentTrackSource = null;
        currentTrackArtist = null;
        currentTrackTitle = null;

    } catch (e) {
        console.error(e);
    }
}

// =============================================================================
// Favorites Functions
// =============================================================================

/**
 * Loads all favorites from the backend and populates the dropdown.
 * 
 * This function is called on app initialization and after any add/remove operation.
 * If favorites exist, the container is shown; otherwise, it remains hidden.
 * 
 * @returns {Promise<void>}
 */
async function loadFavorites() {
    try {
        allFavorites = await invoke("list_favorites");
        console.log("Loaded favorites:", allFavorites.length);

        // Show or hide the favorites container based on whether any exist.
        if (allFavorites.length > 0) {
            favoritesContainer.classList.remove("hidden");
        } else {
            favoritesContainer.classList.add("hidden");
        }

        // Render the list (with no filter).
        renderFavoritesList(allFavorites);

    } catch (e) {
        console.error("Failed to load favorites:", e);
    }
}

/**
 * Renders the favorites list in the dropdown.
 * 
 * @param {Array} favorites - The filtered or full list of favorites to display.
 */
function renderFavoritesList(favorites) {
    if (!favoritesList) return;

    // Clear existing items.
    favoritesList.innerHTML = "";

    // Show empty state if no favorites.
    if (favorites.length === 0) {
        const emptyLi = document.createElement("li");
        emptyLi.className = "empty-state";
        emptyLi.textContent = "No favorites found.";
        favoritesList.appendChild(emptyLi);
        return;
    }

    // Create a list item for each favorite.
    favorites.forEach(fav => {
        const li = document.createElement("li");
        li.dataset.source = fav.source; // Store source for selection.

        // Music note icon.
        const icon = document.createElement("span");
        icon.className = "material-symbols-outlined fav-icon";
        icon.textContent = "music_note";
        li.appendChild(icon);

        // Label container (clips overflow) and label (can expand on hover).
        const labelContainer = document.createElement("div");
        labelContainer.className = "fav-label-container";

        const label = document.createElement("span");
        label.className = "fav-label";
        label.textContent = `${fav.artist} - ${fav.title}`;
        labelContainer.appendChild(label);
        li.appendChild(labelContainer);

        // Delete button (trash icon).
        const deleteBtn = document.createElement("button");
        deleteBtn.className = "fav-delete-btn";
        deleteBtn.title = "Remove from Favorites";
        deleteBtn.innerHTML = '<span class="material-symbols-outlined">delete_outline</span>';
        li.appendChild(deleteBtn);

        // Delete button click: Show undo toast instead of immediate delete.
        deleteBtn.addEventListener("click", (e) => {
            e.stopPropagation(); // Prevent triggering the play handler.
            initiateDelete(fav);
        });

        // Click handler: Select and play this favorite.
        li.addEventListener("click", () => {
            pathInput.value = fav.source;
            closeFavoritesDropdown();
            startRemix();
        });

        favoritesList.appendChild(li);

        // --- Teleprompter Setup ---
        // After appending, check if the label overflows its container.
        // If so, add the 'overflows' class and set the scroll distance and duration.
        requestAnimationFrame(() => {
            // Compare label's natural width to container's visible width.
            if (label.scrollWidth > labelContainer.clientWidth) {
                label.classList.add("overflows");

                // Calculate how far to scroll: negative because we move left.
                const overflowPx = label.scrollWidth - labelContainer.clientWidth;
                label.style.setProperty("--overflow-amount", `${-overflowPx}px`);

                // Calculate duration for consistent scroll speed.
                // Target: ~50 pixels/second for comfortable reading.
                // Add 30% for the pauses at start/end (15% + 15% in keyframes).
                const scrollTime = overflowPx / 50; // seconds for the actual scroll
                const totalDuration = Math.max(3, Math.min(15, scrollTime * 1.3));
                label.style.setProperty("--scroll-duration", `${totalDuration}s`);
            }
        });
    });
}

/**
 * Filters the favorites list based on the search input.
 * 
 * Performs a case-insensitive search against both artist and title.
 */
function filterFavorites() {
    const query = favoritesSearch.value.toLowerCase().trim();

    if (!query) {
        renderFavoritesList(allFavorites);
        return;
    }

    const filtered = allFavorites.filter(fav => {
        const combined = `${fav.artist} ${fav.title}`.toLowerCase();
        return combined.includes(query);
    });

    renderFavoritesList(filtered);
}

/**
 * Toggles the Favorite (heart) button state based on whether the current track is a favorite.
 */
async function updateFavoriteButtonState() {
    if (!favoriteBtn || !currentTrackSource) return;

    const isFav = await invoke("check_is_favorite", { source: currentTrackSource });
    const icon = favoriteBtn.querySelector(".material-symbols-outlined");

    if (isFav) {
        favoriteBtn.classList.add("active");
        icon.textContent = "favorite";
        favoriteBtn.title = "Remove from Favorites";
    } else {
        favoriteBtn.classList.remove("active");
        icon.textContent = "favorite_border";
        favoriteBtn.title = "Add to Favorites";
    }
}

/**
 * Handles the click on the favorite button: toggle add/remove.
 */
async function toggleFavorite() {
    if (!currentTrackSource) return;

    try {
        const isFav = await invoke("check_is_favorite", { source: currentTrackSource });

        if (isFav) {
            // Remove from favorites.
            await invoke("remove_favorite", { source: currentTrackSource });
            console.log("Removed from favorites:", currentTrackSource);
        } else {
            // Add to favorites.
            await invoke("add_favorite", {
                source: currentTrackSource,
                artist: currentTrackArtist,
                title: currentTrackTitle,
            });
            console.log("Added to favorites:", currentTrackSource);
        }

        // Refresh UI.
        await loadFavorites();
        updateFavoriteButtonState();

    } catch (e) {
        console.error("Failed to toggle favorite:", e);
    }
}

/**
 * Confirms deletion of a favorite via a simple confirmation (long-press).
 * 
 * @param {object} fav - The favorite object to delete.
 * @deprecated Replaced by initiateDelete() with undo toast.
 */
// async function confirmDeleteFavorite(fav) { ... } // REMOVED

// =============================================================================
// Undo Toast Delete System
// =============================================================================

/**
 * Initiates a delete operation with undo capability.
 * 
 * The favorite is immediately removed from the UI (optimistic update),
 * and a toast is shown with an Undo button. If the user does not click Undo
 * within 5 seconds, the delete is committed to the backend.
 * 
 * @param {object} fav - The favorite object to delete.
 */
function initiateDelete(fav) {
    // If there's already a pending delete, commit it first.
    if (pendingDelete) {
        commitDelete();
    }

    // Store the favorite for potential undo.
    pendingDelete = fav;

    // Optimistically remove from the local list and re-render.
    allFavorites = allFavorites.filter(f => f.source !== fav.source);
    renderFavoritesList(allFavorites);

    // Update favorites visibility (hide container if empty).
    if (allFavorites.length === 0) {
        favoritesContainer.classList.add("hidden");
        closeFavoritesDropdown();
    }

    // Update heart button if the deleted item is the current track.
    if (currentTrackSource === fav.source) {
        updateFavoriteButtonState();
    }

    // Show the undo toast.
    toastMessage.textContent = `Removed "${fav.title}"`;
    undoToast.classList.remove("hidden");

    // Auto-commit after 5 seconds.
    toastTimeout = setTimeout(() => {
        commitDelete();
    }, 5000);
}

/**
 * Undoes the pending delete operation.
 * 
 * Re-adds the favorite to the local list and hides the toast.
 * No backend call is needed because we never actually deleted it yet.
 */
async function undoDelete() {
    if (!pendingDelete) return;

    // Clear the auto-commit timer.
    clearTimeout(toastTimeout);

    // Re-add to local list.
    allFavorites.push(pendingDelete);

    // Re-sort by artist, then title.
    allFavorites.sort((a, b) => {
        const artistCmp = a.artist.toLowerCase().localeCompare(b.artist.toLowerCase());
        if (artistCmp !== 0) return artistCmp;
        return a.title.toLowerCase().localeCompare(b.title.toLowerCase());
    });

    // Re-render and show container.
    renderFavoritesList(allFavorites);
    favoritesContainer.classList.remove("hidden");

    // Update heart button.
    updateFavoriteButtonState();

    // Hide toast and clear pending state.
    hideUndoToast();
    console.log("Undo: Restored favorite:", pendingDelete.title);
    pendingDelete = null;
}

/**
 * Commits the pending delete to the backend.
 * 
 * Called either when the toast times out or when a new delete is initiated
 * while one is already pending.
 */
async function commitDelete() {
    if (!pendingDelete) return;

    try {
        await invoke("remove_favorite", { source: pendingDelete.source });
        console.log("Committed delete:", pendingDelete.title);
    } catch (e) {
        console.error("Failed to commit delete:", e);
    }

    // Hide toast and clear state.
    hideUndoToast();
    pendingDelete = null;
}

/**
 * Hides the undo toast.
 */
function hideUndoToast() {
    clearTimeout(toastTimeout);
    if (undoToast) undoToast.classList.add("hidden");
}

/**
 * Opens the favorites dropdown and marks the toggle button as open.
 */
function openFavoritesDropdown() {
    if (favoritesDropdown) favoritesDropdown.classList.remove("hidden");
    if (favoritesToggle) favoritesToggle.classList.add("open");
    if (favoritesSearch) favoritesSearch.focus();
}

/**
 * Closes the favorites dropdown and resets the toggle button state.
 */
function closeFavoritesDropdown() {
    if (favoritesDropdown) favoritesDropdown.classList.add("hidden");
    if (favoritesToggle) favoritesToggle.classList.remove("open");
    if (favoritesSearch) favoritesSearch.value = ""; // Reset search.
    renderFavoritesList(allFavorites); // Reset to full list.
}

window.addEventListener("DOMContentLoaded", () => {
    // Bind M3 Elements
    statusEl = document.querySelector("#status-msg");
    analyzeBtn = document.querySelector("#analyze-btn");
    pathInput = document.querySelector("#file-path");

    onboardingCard = document.querySelector("#onboarding-card");
    floatingPlayer = document.querySelector("#floating-player");
    stopBtn = document.querySelector("#stop-btn");
    pauseBtn = document.querySelector("#pause-btn");
    trackTitleEl = document.querySelector(".track-title");
    albumArtImg = document.querySelector("#album-art-img");
    albumArtPlaceholder = document.querySelector("#album-art-placeholder");

    const canvas = document.querySelector("#jukebox-canvas");
    viz = new InfiniteJukeboxViz(canvas);

    analyzeBtn.addEventListener("click", () => {
        startRemix();
    });

    // Bind Stop Button
    if (stopBtn) {
        stopBtn.addEventListener("click", () => {
            stopRemix();
        });
    }

    // Bind Pause Button
    if (pauseBtn) {
        pauseBtn.addEventListener("click", async () => {
            try {
                isPaused = !isPaused;
                await invoke("set_paused", { paused: isPaused });
                const icon = pauseBtn.querySelector(".material-symbols-outlined");
                if (icon) {
                    icon.textContent = isPaused ? "play_circle" : "pause_circle";
                }
                statusEl.textContent = isPaused ? "Paused" : "Infinite Walk Active";
            } catch (e) {
                console.error("Pause toggle failed:", e);
            }
        });
    }

    // --- Favorites UI Bindings ---
    favoriteBtn = document.querySelector("#favorite-btn");
    favoritesContainer = document.querySelector("#favorites-container");
    favoritesToggle = document.querySelector("#favorites-toggle");
    favoritesDropdown = document.querySelector("#favorites-dropdown");
    favoritesSearch = document.querySelector("#favorites-search");
    favoritesList = document.querySelector("#favorites-list");

    // Load favorites on startup.
    loadFavorites();

    // --- Undo Toast Bindings ---
    undoToast = document.querySelector("#undo-toast");
    toastMessage = document.querySelector("#toast-message");
    toastUndoBtn = document.querySelector("#toast-undo-btn");

    if (toastUndoBtn) {
        toastUndoBtn.addEventListener("click", undoDelete);
    }

    // Bind Favorite (Heart) Button.
    if (favoriteBtn) {
        favoriteBtn.addEventListener("click", toggleFavorite);
    }

    // Bind Favorites Toggle Button (dropdown open/close).
    if (favoritesToggle) {
        favoritesToggle.addEventListener("click", () => {
            const isHidden = favoritesDropdown.classList.contains("hidden");
            if (isHidden) {
                openFavoritesDropdown();
            } else {
                closeFavoritesDropdown();
            }
        });
    }

    // Bind Search Input for Filtering.
    if (favoritesSearch) {
        favoritesSearch.addEventListener("input", filterFavorites);
    }

    // Close dropdown when clicking outside.
    document.addEventListener("click", (e) => {
        if (favoritesContainer && !favoritesContainer.contains(e.target)) {
            closeFavoritesDropdown();
        }
    });

    // Listen for Playback Ticks
    listen('playback_tick', (event) => {
        // payload: { beat_index, segment_index, seq_len, seq_pos }
        const { beat_index, segment_index, seq_len, seq_pos } = event.payload;

        // Update Viz State & Draw
        viz.updatePlaybackState(seq_pos, seq_len);
        viz.draw(beat_index, segment_index);
    });

    // Listen for Early Metadata (Instant Update)
    listen('metadata_ready', (event) => {
        // If we have high-quality API metadata, ignore the local probe
        if (remoteMetadata) {
            console.log("Ignoring local metadata update (preferring remote API)");
            return;
        }

        const payload = event.payload;
        // Update Metadata
        if (trackTitleEl) trackTitleEl.textContent = payload.title;

        // Update Album Art
        if (payload.album_art_base64 && albumArtImg) {
            albumArtImg.src = payload.album_art_base64;
            albumArtImg.classList.remove("hidden");
            if (albumArtPlaceholder) albumArtPlaceholder.classList.add("hidden");
        }
    });

    // Listen for Detailed Status Updates
    listen('analysis_progress', (event) => {
        // payload: { message: string, progress: float }
        const { message, progress } = event.payload;
        if (statusEl) statusEl.textContent = message;

        // Update Progress Bar
        const fill = document.querySelector(".progress-bar-fill");
        if (fill) {
            fill.style.width = (progress * 100) + "%";
        }
    });

    // Listen for Downloader Status
    listen('downloader_status', (event) => {
        if (statusEl) statusEl.textContent = event.payload;
    });

    // Listen for Early Metadata (fires before download completes)
    listen('download_metadata', (event) => {
        const meta = event.payload;
        console.log("Early metadata received:", meta);

        // Update track title immediately
        if (trackTitleEl && meta.title) {
            trackTitleEl.textContent = meta.title;
        }

        // Update album art from thumbnail URL
        if (meta.thumbnail_url && albumArtImg) {
            albumArtImg.src = meta.thumbnail_url;
            albumArtImg.classList.remove("hidden");
            if (albumArtPlaceholder) albumArtPlaceholder.classList.add("hidden");
        }

        // Update favorites state with early metadata
        if (meta.artist) currentTrackArtist = meta.artist;
        if (meta.title) currentTrackTitle = meta.title;
    });

}, false); // End Event Listener

// Global Error Handler
window.addEventListener('error', (event) => {
    const status = document.getElementById("status-msg");
    if (status) status.textContent = "JS Error: " + event.message;
});
