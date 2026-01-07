# App Icons (`src-tauri/icons/`)

This directory contains the application icons used by Tauri to bundle the application for different operating systems (Windows, macOS, Linux).

## Contents
*   **`icon.png`**: The master high-resolution icon (typically 512x512 or larger).
*   **`icon.ico`**: Windows application icon (multiple sizes embedded).
*   **`icon.icns`**: macOS application icon bundle.
*   **`32x32.png`**, **`128x128.png`**: Standard sizes used for Linux desktop entries and window icons.

## Usage
These files are referenced in `tauri.conf.json` under `bundle > icon`. When running `tauri build`, these assets are baked into the final executable and installer.
