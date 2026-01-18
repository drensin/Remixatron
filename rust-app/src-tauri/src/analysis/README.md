# Analysis Modules (`src-tauri/src/analysis/`)

This directory contains the audio analysis and feature extraction modules for structural segmentation.

## Files

*   **`mod.rs`**: Module exports and shared types for the analysis pipeline.

*   **`structure.rs`**: Novelty-based segmentation algorithm.
    *   Computes self-similarity matrices from audio features
    *   Detects structural boundaries (verse, chorus, bridge) via novelty curves
    *   Assigns cluster labels to segments

*   **`features.rs`**: Audio feature extraction.
    *   Computes MFCC (Mel-Frequency Cepstral Coefficients)
    *   Computes Chroma features for harmonic analysis
    *   Provides similarity metrics for beat comparison

*   **`clustering.rs`**: Spectral clustering implementation.
    *   Graph Laplacian construction
    *   K-means on eigenvectors for segment labeling
    *   Used for grouping similar beats into clusters

*   **`cqt.rs`**: Constant-Q Transform implementation.
    *   Frequency-domain representation with logarithmic frequency bins
    *   Used for pitch and harmonic analysis

*   **`spectrogram.rs`**: Spectrogram computation utilities.
    *   Short-Time Fourier Transform (STFT)
    *   Magnitude and power spectrogram generation

## Architecture

These modules are orchestrated by `workflow.rs` in the parent directory. The typical flow is:

1. Audio → Mel Spectrogram (`beat_tracker/mel.rs`)
2. Beat Tracking → Beat times (`beat_tracker/inference.rs`)
3. Features → Similarity matrix (`features.rs`)
4. Segmentation → Cluster labels (`structure.rs`)
