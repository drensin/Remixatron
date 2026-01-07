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

async function startRemix() {
    let path = pathInput.value;
    if (!path) return;

    try {
        // UI Transition: Zero -> Loading
        onboardingCard.classList.add("hidden");
        floatingPlayer.classList.add("visible");
        floatingPlayer.classList.add("loading"); // Start Progress Bar
        analyzeBtn.disabled = true;

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

    } catch (e) {
        console.error(e);
        statusEl.textContent = "Error: " + e;
        analyzeBtn.disabled = false;

        // UI Transition: Revert on fatal error?
        // Keep player visible so error can be seen
        floatingPlayer.classList.remove("loading");
    }
}

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

    } catch (e) {
        console.error(e);
    }
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

}, false); // End Event Listener

// Global Error Handler
window.addEventListener('error', (event) => {
    const status = document.getElementById("status-msg");
    if (status) status.textContent = "JS Error: " + event.message;
});
