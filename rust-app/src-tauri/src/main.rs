//! Remixatron: An Infinite Jukebox Application (Tauri Binary Entry Point).
//!
//! This file serves as the main entry point for the compiled binary.
//! Ideally, this file remains minimal, delegating all application logic,
//! setup, and lifecycle management to the library crate (`lib.rs`).

// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

/// The main entry point of the application.
///
/// This function simply delegates execution to `remixatron_lib::run()`,
/// which initializes the Tauri runtime and starts the application event loop.
fn main() {
    remixatron_lib::run()
}
