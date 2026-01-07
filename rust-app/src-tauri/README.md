# Tauri Backend Root (`src-tauri/`)

This directory contains the Rust environment and configuration for the Remixatron application. It serves as the bridge between the operating system and the web-based frontend.

## key Files
*   **`src/`**: The actual Rust source code for the application logic (Audio Engine, ML Pipeline, etc.).
*   **`tauri.conf.json`**: The main configuration file for the Tauri application. Defines:
    *   Window properties (size, resizability)
    *   Security capabilities (filesystem access, shell commands)
    *   Bundle settings (app name, version, identifier)
    *   Content Security Policy (CSP)
*   **`Cargo.toml`**: The Rust dependency manifest. Manages crates like `kira` (Audio), `ort` (ONNX), and `symphonia` (decoding).
*   **`build.rs`**: Build script used to handle native library linking (e.g., linking `libonnxruntime.so` on Linux).
*   **`capabilities/`**: Directory containing fine-grained permission files for Tauri v2's security model.
