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
        statusEl.textContent = "Analyzing... (This may take a moment)";
        analyzeBtn.disabled = true;

        // 1. Analyze
        const payload = await invoke("analyze_track", { path });
        console.log("Analysis Complete!", payload);
        statusEl.textContent = "Analysis Complete! Starting Playback...";

        // 2. Setup Viz
        viz.setData(payload);

        // 3. Play
        // Note: play_track is async/blocking in backend logic but returns quickly if we spawned thread correctly?
        // Wait, my backend implementation blocked. `spawn_blocking` returns a task handle, but `invoke` awaits it?
        // `spawn_blocking` runs on a thread pool. The `invoke` returns the result of the closure.
        // My closure returns `Ok(())` immediately? No.
        // `spawn_blocking(closure)` returns a `JoinHandle`.
        // I did NOT await the join handle in Rust. I just disregarded it?
        // Ah, `tauri::async_runtime::spawn_blocking` returns a JoinHandle.
        // If I don't await it, the function returns immediately.
        // So `play_track` will return immediately, keeping the thread running. Perfect.

        await invoke("play_track", { path });
        statusEl.textContent = "Playing Infinite Walk...";

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
    listen('playback_tick', (event) => {
        // payload: { beat_index, segment_index }
        const { beat_index, segment_index } = event.payload;
        viz.draw(beat_index, segment_index);
    });
}, false); // End Event Listener

// Global Error Handler
window.addEventListener('error', (event) => {
    const status = document.getElementById("status-msg");
    if (status) status.textContent = "JS Error: " + event.message;
});
