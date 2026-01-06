const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;
import { InfiniteJukeboxViz } from './viz.js';

let viz;
let statusEl;
let analyzeBtn;
let pathInput;

async function startRemix() {
    const path = pathInput.value;
    if (!path) return;

    try {
        statusEl.textContent = "Stopping previous playback...";
        try {
            await invoke("stop_playback");
        } catch (e) {
            console.log("Stop playback harmless error:", e);
        }

        statusEl.textContent = "Analyzing... (This may take a moment)";
        analyzeBtn.disabled = true;

        // 1. Analyze
        const payload = await invoke("analyze_track", { path });
        console.log("Analysis Complete!", payload);
        statusEl.textContent = "Analysis Complete! Starting Playback...";

        // 2. Setup Viz
        viz.setData(payload);

        // 3. Play
        await invoke("play_track", { path });
        statusEl.textContent = "Playing Infinite Walk...";
        analyzeBtn.disabled = false;

    } catch (e) {
        console.error(e);
        statusEl.textContent = "Error: " + e;
        analyzeBtn.disabled = false;
    }
}

window.addEventListener("DOMContentLoaded", () => {
    statusEl = document.querySelector("#status-msg");
    analyzeBtn = document.querySelector("#analyze-btn");
    pathInput = document.querySelector("#file-path");

    const canvas = document.querySelector("#jukebox-canvas");
    viz = new InfiniteJukeboxViz(canvas);

    analyzeBtn.addEventListener("click", () => {
        startRemix();
    });

    // Listen for Playback Ticks
    // Listen for Playback Ticks
    listen('playback_tick', (event) => {
        // payload: { beat_index, segment_index, seq_len, seq_pos }
        const { beat_index, segment_index, seq_len, seq_pos } = event.payload;

        // Update Viz State & Draw
        viz.updatePlaybackState(seq_pos, seq_len);
        viz.draw(beat_index, segment_index);
    });
}, false); // End Event Listener

// Global Error Handler
window.addEventListener('error', (event) => {
    const status = document.getElementById("status-msg");
    if (status) status.textContent = "JS Error: " + event.message;
});
