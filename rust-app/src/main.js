const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;
import { InfiniteJukeboxViz } from './viz.js';

let viz;
let statusEl;
let analyzeBtn;
let pathInput;

// UI Elements (New M3 Structure)
let onboardingCard;
let floatingPlayer;
let stopBtn;
let trackTitleEl;
let albumArtImg;
let albumArtPlaceholder;

async function startRemix() {
    const path = pathInput.value;
    if (!path) return;

    try {
        // UI Transition: Zero -> Loading
        onboardingCard.classList.add("hidden");
        floatingPlayer.classList.add("visible");
        floatingPlayer.classList.add("loading"); // Start Progress Bar

        statusEl.textContent = "Stopping previous playback...";
        try {
            await invoke("stop_playback");
        } catch (e) {
            console.log("Stop playback harmless error:", e);
        }

        statusEl.textContent = "Analyzing Audio Structure...";
        analyzeBtn.disabled = true;

        // UI: Show what we are loading immediately
        const filename = path.split(/[/\\]/).pop();
        if (trackTitleEl) trackTitleEl.textContent = `Loading ${filename}...`;

        // UI: Reset Art
        if (albumArtImg) albumArtImg.classList.add("hidden");
        if (albumArtPlaceholder) albumArtPlaceholder.classList.remove("hidden");

        // 1. Analyze
        const payload = await invoke("analyze_track", { path });
        console.log("Analysis Complete!", payload);
        statusEl.textContent = "Structure Decoded. Starting Walk...";

        // 1b. Update Metadata
        if (trackTitleEl) trackTitleEl.textContent = payload.title;

        // 1c. Update Album Art
        if (payload.album_art_base64 && albumArtImg) {
            albumArtImg.src = payload.album_art_base64;
            albumArtImg.classList.remove("hidden");
            if (albumArtPlaceholder) albumArtPlaceholder.classList.add("hidden");
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
        if (statusEl) statusEl.textContent = event.payload;
    });

}, false); // End Event Listener

// Global Error Handler
window.addEventListener('error', (event) => {
    const status = document.getElementById("status-msg");
    if (status) status.textContent = "JS Error: " + event.message;
});
