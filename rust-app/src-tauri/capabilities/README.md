# Tauri Capabilities (`src-tauri/capabilities/`)

This directory contains the security configuration files for Tauri's permission system (v2).

## Purpose
Tauri v2 uses a "Capabilities" model to strictly control what the frontend can do. Instead of a global allowlist, specific sets of permissions are grouped into capability files (`.json`) and assigned to specific windows or contexts.

## Files
*   **`default.json`** (or similar): Defines the default set of permissions available to the main application window.
    *   `fs:allow-read-recursive`: Allows reading files selected by the user.
    *   `shell:allow-open`: Allows opening links in the external browser.
    *   `core:default`: Basic window and event operations.

## Usage
To add a new permission (e.g., allowing the app to read a specific config file), you must modify the relevant JSON file here and rebuild the application.
