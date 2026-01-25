//! # Structural Analysis Module
//!
//! This module implements musical structure analysis for the Infinite Jukebox,
//! combining novelty-based boundary detection with recurrence-based clustering.
//!
//! ## Primary Algorithm (`compute_segments_checkerboard`)
//!
//! A **hybrid** approach that combines the strengths of two techniques:
//!
//! 1. **Novelty Boundaries**: Checkerboard kernel on SSM detects structural transitions.
//! 2. **Recurrence Clustering**: Beat-level k-NN similarity aggregated per segment
//!    captures rhythmic/harmonic patterns that pooled features miss.
//! 3. **Spectral Embedding**: Normalized Laplacian eigenvectors on segment graph.
//! 4. **K-Means Clustering**: Groups segments by structural similarity.
//! 5. **Jump Graph**: Adaptive P75 threshold for quality jump candidates.
//!
//! ## Secondary Algorithm (`compute_segments_knn`)
//!
//! McFee & Ellis 2014 ISMIR paper implementation (spectral clustering on beats).
//! Retained for experimentation but not currently used in the workflow.
//!
//! ## K Selection Strategies
//!
//! * **EigengapHeuristic** (Default): `K* = argmax[(λ_{k+1} - λ_k) / λ_k]`
//! * **BalancedConnectivity**: `100*Sil + Median(JumpCount)`
//! * **ConnectivityFirst**: Maximize escape fraction among valid K's
//! * **MaxK**: Maximize K subject to quality floors

use ndarray::{Array2, Array1, s};
use linfa::traits::{Fit, Predict};
use rayon::prelude::*;
use linfa_clustering::KMeans;
use linfa::DatasetBase;


pub struct StructureAnalyzer;

impl Default for StructureAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Strategy enum for Auto-K Selection.
/// Allows A/B testing different heuristics for choosing the optimal cluster count.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AutoKStrategy {
    // ─────────────────────────────────────────────────────────────────────────
    // NORMALIZED EIGENGAP HEURISTIC
    // ─────────────────────────────────────────────────────────────────────────
    //
    // Select K where the *relative* eigenvalue gap is largest:
    //
    //     K* = argmax[ (λ_{k+1} - λ_k) / λ_k ]  for k ∈ [K_min, K_max]
    //
    // Normalizing by λ_k ensures we prefer proportionally significant gaps.
    // This naturally favors earlier K values where gaps are large relative
    // to the eigenvalue magnitude, without needing an arbitrary K cap.
    //
    // Reference: Von Luxburg, "A Tutorial on Spectral Clustering" Section 8.2
    // ─────────────────────────────────────────────────────────────────────────
    
    /// Normalized eigengap heuristic: Select K where the relative gap is largest.
    /// 
    /// Formula: `K* = argmax[ (λ_{k+1} - λ_k) / λ_k ]`
    /// 
    /// This naturally prefers earlier K values with proportionally significant
    /// gaps without requiring an arbitrary cap. A 14% jump at K=4 beats a 9%
    /// jump at K=23, even if the absolute gap is larger.
    /// 
    /// Uses floor K_min=3 and ceiling K_max=32.
    EigengapHeuristic,
    
    /// The legacy heuristic from the Python implementation.
    /// Formula: `K + (10 * Sil) + Ratio + MinSeg_Score`
    #[allow(dead_code)]
    LegacyUngatedSum,

    /// The streamlined heuristic that directly measures graph health.
    /// Formula: `(100 * Sil) + Median_Jump_Count`
    #[allow(dead_code)]
    BalancedConnectivity,

    /// Connectivity-first approach: Maximizes graph connectivity among K values
    /// that pass minimum quality and playability thresholds.
    /// 
    /// Floors (per Kaufman & Rousseeuw 1990):
    /// - `silhouette >= 0.5` ("reasonable structure" quality floor)
    /// - `escape_fraction >= 0.5` (50% of segments must have escape routes)
    /// - `median_jumps >= 4` (typical beat should have 4+ jump targets)
    /// 
    /// Score: `escape_fraction * ln(median_jumps)` (among valid K's)
    #[allow(dead_code)]
    ConnectivityFirst,

    /// Complexity-maximizing approach: Finds the highest K that still meets
    /// minimum quality and playability thresholds.
    /// 
    /// Floors:
    /// - `silhouette >= 0.5` (quality floor)
    /// - `escape_fraction >= 0.5` (playability floor) 
    /// - `median_jumps >= 4` (connectivity floor)
    /// 
    /// Score: `K` (maximize structural complexity among valid K's)
    #[allow(dead_code)]
    MaxK,
}

impl StructureAnalyzer {
    pub fn new() -> Self {
        Self
    }
}

/// Result of segmentation containing labels and metadata
pub struct SegmentationResult {
    /// Cluster ID for each beat
    pub labels: Vec<usize>,
    /// Optimal K selected
    pub k_optimal: usize,
    /// Eigenvalues from the Laplacian
    pub eigenvalues: Vec<f32>,
    /// Novelty curve (only for Checkerboard method)
    pub novelty_curve: Vec<f32>,
    /// Detected peaks/boundaries (only for Checkerboard method)
    pub peaks: Vec<usize>,
    /// Pre-calculated jump candidates for each beat
    pub jumps: Vec<Vec<usize>>,
}

impl StructureAnalyzer {


    fn calculate_segment_stats(labels: &[usize], k_clusters: usize) -> (f32, usize) {
        if labels.is_empty() { return (0.0, 0); }
        
        let mut segment_count = 0;
        let mut current_seg_len = 0;
        let mut min_seg_len = usize::MAX;
        let mut prev_label = None;
        
        for &label in labels {
            if Some(label) != prev_label {
                if current_seg_len > 0 && current_seg_len < min_seg_len {
                    min_seg_len = current_seg_len;
                }
                segment_count += 1;
                current_seg_len = 1;
                prev_label = Some(label);
            } else {
                current_seg_len += 1;
            }
        }
        // Last segment
        if current_seg_len > 0 && current_seg_len < min_seg_len {
             min_seg_len = current_seg_len;
        }
        
        let ratio = segment_count as f32 / k_clusters as f32;
        (ratio, min_seg_len)
    }

    fn calculate_silhouette_score(points: &Array2<f64>, labels: &[usize], k: usize) -> f32 {
        // Simplified Euclidean Silhouette
        // For each point i:
        //   a(i) = mean distance to other points in same cluster
        //   b(i) = min mean distance to points in other clusters
        //   s(i) = (b - a) / max(a, b)
        
        let n = points.nrows();
        if n == 0 || k < 2 { return 0.0; }
        
        let mut total_s = 0.0;
        
        // Pre-compute basic distances? No, N is small enough for O(N^2) here (500^2 = 250k ops)
        // Optimization: Convert Array2 to Vec<Vec<f64>> for faster indexing avoid array overhead?
        // Actually ndarray is fast enough.
        
        for i in 0..n {
            let label_i = labels[i];
            
            // Calculate a(i)
            let mut sum_dist_a = 0.0;
            let mut count_a = 0;
            
            // Calculate b(i) candidates
            // Map: other_cluster_label -> (sum_dist, count)
            let mut other_clusters_dist = vec![0.0; k];
            let mut other_clusters_count = vec![0; k];
            
            #[allow(clippy::needless_range_loop)]  // j indexes points rows and labels; complex multi-array access
            for j in 0..n {
                if i == j { continue; }
                
                // Euclidean dist
                // manual loop for speed
                let p1 = points.row(i);
                let p2 = points.row(j);
                let mut dist_sq = 0.0;
                for idx in 0..p1.len() {
                    let d = p1[idx] - p2[idx];
                    dist_sq += d * d;
                }
                let dist = dist_sq.sqrt();
                
                let label_j = labels[j];
                
                if label_j == label_i {
                    sum_dist_a += dist;
                    count_a += 1;
                } else {
                    other_clusters_dist[label_j] += dist;
                    other_clusters_count[label_j] += 1;
                }
            }
            
            let a_i = if count_a > 0 { sum_dist_a / count_a as f64 } else { 0.0 };
            
            let mut b_i = f64::MAX;
            for c in 0..k {
                if c == label_i { continue; }
                if other_clusters_count[c] > 0 {
                    let mean_dist = other_clusters_dist[c] / other_clusters_count[c] as f64;
                    if mean_dist < b_i { b_i = mean_dist; }
                }
            }
            
            if b_i == f64::MAX { b_i = 0.0; } // Should not happen if k > 1 and all clusters populated
            
            let max_ab = a_i.max(b_i);
            let s_i = if max_ab > 0.0 { (b_i - a_i) / max_ab } else { 0.0 };
            
            total_s += s_i;
        }
        
        (total_s / n as f64) as f32
    }

    /// Simulates the jump graph for a candidate set of labels to compute connectivity stats.
    ///
    /// This is computationally expensive (O(N^2)) but necessary for metrics that
    /// depend on actual jump availability (like Median Jump Count).
    ///
    /// # Arguments
    /// * `labels`: The cluster assignment for every beat.
    /// * `beats_per_bar`: Needed for phase alignment rules (roughly estimated or passed).
    ///
    /// # Returns
    /// An array of jump counts for every beat.
    fn simulate_jump_counts(labels: &[usize]) -> Vec<usize> {
        let n = labels.len();
        let mut jump_counts = vec![0; n];
        
        // We need to define "segments" first to know segment boundaries for the simulation.
        // Jump Rule: Can't jump to same segment.
        let mut segment_ids = vec![0; n];
        let mut current_seg = 0;
        let mut prev_label = labels[0];
        
        for i in 0..n {
            if labels[i] != prev_label {
                current_seg += 1;
                prev_label = labels[i];
            }
            segment_ids[i] = current_seg;
        }

        // Rule Simplification for Speed:
        // We assume Bar Position and Intra-Segment Index match if we are just testing connectivity potential.
        // Actually, to be accurate, we need to respect the REAL jump rules:
        // 1. Same Cluster
        // 2. Different Segment
        // 3. Not Immediate Neighbor
        // 4. (Ignoring detailed Bar/Phase alignment for this heuristic proxy??)
        // 
        // User requested "Median Jump Candidates". To be accurate, we should probably just count 
        // "Total Potential Targets" based on Cluster + Segment rules. Phase/Bar rules filter ~75% of these,
        // but that filtering is uniform across all K. So raw Cluster/Segment match count is a valid proxy.
        
        // Optimization: Pre-calculate indices for each cluster
        // Map: Label -> List(BeatIndices)
        let mut clusters: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for (i, &label) in labels.iter().enumerate() {
            clusters.entry(label).or_default().push(i);
        }

        for i in 0..n {
            let my_label = labels[i];
            let my_seg = segment_ids[i];
            
            if let Some(candidates) = clusters.get(&my_label) {
                let mut valid_jumps = 0;
                for &target in candidates {
                   // Rule: Different Segment
                   if segment_ids[target] != my_seg {
                       valid_jumps += 1;
                   }
                }
                // Return RAW count (no scaling). Phase/bar alignment filtering 
                // happens during playback, not during Auto-K selection.
                jump_counts[i] = valid_jumps; 
            }
        }
        
        jump_counts
    }

    /// Calculates the fraction of segments that have at least one "escape" —
    /// i.e., at least one beat that can jump to a different segment instance
    /// of the same cluster.
    ///
    /// # Returns
    /// A value from 0.0 (no segments can escape) to 1.0 (all segments can escape).
    fn calculate_escape_fraction(labels: &[usize]) -> f32 {
        let n = labels.len();
        if n == 0 { return 0.0; }
        
        // Step 1: Build segment IDs (consecutive runs of same label)
        let mut segment_ids = vec![0usize; n];
        let mut current_seg = 0;
        let mut prev_label = labels[0];
        
        for i in 0..n {
            if labels[i] != prev_label {
                current_seg += 1;
                prev_label = labels[i];
            }
            segment_ids[i] = current_seg;
        }
        
        let num_segments = current_seg + 1;
        
        // Step 2: Build cluster -> beat indices lookup
        let mut cluster_beats: std::collections::HashMap<usize, Vec<usize>> = 
            std::collections::HashMap::new();
        for (i, &label) in labels.iter().enumerate() {
            cluster_beats.entry(label).or_default().push(i);
        }
        
        // Step 3: For each segment, check if ANY beat can escape
        let mut segment_has_escape = vec![false; num_segments];
        
        for i in 0..n {
            let my_cluster = labels[i];
            let my_segment = segment_ids[i];
            
            // Check if this beat can jump to a different segment of same cluster
            if let Some(candidates) = cluster_beats.get(&my_cluster) {
                let can_escape = candidates.iter()
                    .any(|&j| segment_ids[j] != my_segment);
                
                if can_escape {
                    segment_has_escape[my_segment] = true;
                }
            }
        }
        
        // Step 4: Compute fraction
        let escaping = segment_has_escape.iter().filter(|&&x| x).count();
        escaping as f32 / num_segments as f32
    }
}

// Utils

#[allow(dead_code)]  // Used by disabled clustering code
fn normalize_rows(a: &Array2<f32>) -> Array2<f32> {
    let mut res = a.clone();
    for i in 0..a.nrows() {
        let norm = a.row(i).mapv(|x| x*x).sum().sqrt();
        if norm > 0.0 {
            let mut row = res.row_mut(i);
            row.mapv_inplace(|x| x / norm);
        }
    }
    res
}

/// Concatenates two feature matrices horizontally.
///
/// Used to combine chroma (12 dims) and MFCC (20 dims) into a single feature
/// vector (32 dims) for computing the recurrence matrix. Both matrices must
/// have the same number of rows (beats).
///
/// # Arguments
/// * `a` - First feature matrix `[n_beats, dims_a]`.
/// * `b` - Second feature matrix `[n_beats, dims_b]`.
///
/// # Returns
/// Combined matrix `[n_beats, dims_a + dims_b]`.
fn concatenate_features(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    assert_eq!(a.nrows(), b.nrows(), "Feature matrices must have same number of rows");
    
    let n_rows = a.nrows();
    let n_cols = a.ncols() + b.ncols();
    let mut combined = Array2::<f32>::zeros((n_rows, n_cols));
    
    for i in 0..n_rows {
        // Copy columns from a
        for j in 0..a.ncols() {
            combined[[i, j]] = a[[i, j]];
        }
        // Copy columns from b
        for j in 0..b.ncols() {
            combined[[i, a.ncols() + j]] = b[[i, j]];
        }
    }
    
    combined
}

/// Eigenvalue decomposition for real symmetric matrices using nalgebra.
/// Uses nalgebra's SymmetricEigen which is a robust, well-tested implementation
/// for symmetric eigendecomposition. Replaced custom Jacobi implementation which
/// had numerical precision issues causing zero eigenvectors.
///
/// # Arguments
/// * `a` - Symmetric matrix in ndarray format
///
/// # Returns
/// Tuple of (eigenvalues, eigenvectors) where eigenvectors are column vectors.
fn symmetric_eigendecomposition(a: &Array2<f32>) -> (Vec<f32>, Array2<f32>) {
    use nalgebra::{DMatrix, SymmetricEigen};
    
    let n = a.nrows();
    
    // Convert ndarray to nalgebra DMatrix (f64 for numerical stability)
    let mut na_matrix = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            na_matrix[(i, j)] = a[[i, j]] as f64;
        }
    }
    
    // Compute eigendecomposition using nalgebra's SymmetricEigen
    let eigen = SymmetricEigen::new(na_matrix);
    
    // Convert back to ndarray format (f32)
    let eigenvalues: Vec<f32> = eigen.eigenvalues.iter().map(|&v| v as f32).collect();
    
    let mut eigenvectors = Array2::<f32>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            // nalgebra stores eigenvectors as columns
            eigenvectors[[i, j]] = eigen.eigenvectors[(i, j)] as f32;
        }
    }
    
    (eigenvalues, eigenvectors)
}

// ==========================================
// Checkerboard Kernel Segmentation Helpers
// ==========================================

/// Compute Self-Similarity Matrix (Cosine Similarity).
/// Returns N x N matrix.
fn compute_ssm(features: &Array2<f32>) -> Array2<f32> {
    // SSM = F * F^T (Dot Product) if F is normalized.
    // Our features (MFCC+Chroma) should be normalized before this.
    // Assuming row-major input [N_beats, N_features].
    
    // ndarray `dot` uses BLAS if available which is faster than Rayon loops.
    features.dot(&features.t())
}

/// Generate Gaussian-tapered Checkerboard Kernel
/// size: Width of kernel (e.g. 64).
fn compute_checkerboard_kernel(size: usize) -> Array2<f32> {
    let mut kernel = Array2::<f32>::zeros((size, size));
    let half = (size as f32) / 2.0;
    
    // Gaussian taper sigma
    let sigma = half / 2.0; 
    
    for i in 0..size {
        for j in 0..size {
            // Centered coordinates [-half, half]
            let x = i as f32 - half;
            let y = j as f32 - half;
            
            // Checkerboard sign: Quadrants 1 & 3 positive, 2 & 4 negative.
            let sign = (x * y).signum();
            
            // Gaussian Taper
            let dist_sq = x*x + y*y;
            let gauss = (-dist_sq / (2.0 * sigma * sigma)).exp();
            
            kernel[[i, j]] = sign * gauss;
        }
    }
    kernel
}

/// Compute Novelty Curve via Diagonal Convolution
/// ssm: N x N
/// kernel_size: M
fn compute_novelty_curve(ssm: &Array2<f32>, kernel_size: usize) -> Vec<f32> {
    let (n, _) = ssm.dim();
    // let kernel = compute_checkerboard_kernel(kernel_size); // Computed internally in closure? No.
    // Precompute kernel
    let kernel = compute_checkerboard_kernel(kernel_size);
    let half_k = kernel_size / 2;
    
    // Result curve
    let mut novelty = vec![0.0; n];
    
    // Parallelize along diagonal
    let indices: Vec<usize> = (half_k..n.saturating_sub(half_k)).collect();
    
    let results: Vec<(usize, f32)> = indices.par_iter().map(|&i| {
         // Extract local submatrix centered at i,i
         let start = i - half_k;
         let end = start + kernel_size;
         
         if end > n { return (i, 0.0); } // Safety
         
         let sub = ssm.slice(s![start..end, start..end]);
         
         // Dot product with Kernel
         let score = (&sub * &kernel).sum(); // Element-wise multiply then sum
         
         (i, score)
    }).collect();
    
    for (i, score) in results {
        novelty[i] = score.max(0.0);
    }
    
    novelty
}

/// Gaussian Smoothing for Novelty Curve
fn smooth_curve(input: &[f32], sigma: f32) -> Vec<f32> {
    let n = input.len();
    let kernel_radius = (sigma * 3.0).ceil() as usize;
    let kernel_width = 2 * kernel_radius + 1;
    
    // Create Kernel
    let mut kernel = vec![0.0; kernel_width];
    let mut sum = 0.0;
    #[allow(clippy::needless_range_loop)]  // i used for x calculation and kernel[i] write
    for i in 0..kernel_width {
        let x = (i as f32) - (kernel_radius as f32);
        let val = (-x*x / (2.0 * sigma * sigma)).exp();
        kernel[i] = val;
        sum += val;
    }
    // Normalize
    for v in &mut kernel { *v /= sum; }
    
    // Convolve
    let mut output = vec![0.0; n];
    #[allow(clippy::needless_range_loop)]  // i used for offset calculation with output[i]
    for i in 0..n {
        let mut acc = 0.0;
        #[allow(clippy::needless_range_loop)]  // k used for kernel index and offset calculation
        for k in 0..kernel_width {
            let offset = (k as isize) - (kernel_radius as isize);
            let idx = (i as isize) + offset;
            if idx >= 0 && idx < n as isize {
                acc += input[idx as usize] * kernel[k];
            }
        }
        output[i] = acc;
    }
    output
}

/// Adaptive Threshold Peak Picking with Min Distance
fn find_peaks(curve: &[f32], window: usize, alpha: f32, min_dist: usize) -> Vec<usize> {
    let n = curve.len();
    let mut peaks = Vec::new();
    let mut last_peak_idx = 0;
    
    for i in window..n.saturating_sub(window) {
        let val = curve[i];
        
        // Local Mean/StdDev
        let start = i - window;
        let end = i + window;
        let slice = &curve[start..end];
        let sum: f32 = slice.iter().sum();
        let mean = sum / slice.len() as f32;
        let variance = slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / slice.len() as f32;
        let std_dev = variance.sqrt();
        
        let threshold = mean + alpha * std_dev;
        
        // Is peak?
        // 1. Must be local max (simple 3-point check)
        // 2. Must exceed adaptive threshold
        // 3. Must satisfy min_dist constraint
        if val > threshold && val > curve[i-1] && val > curve[i+1] {
            // Check distance
            if peaks.is_empty() || (i - last_peak_idx) >= min_dist {
                peaks.push(i);
                last_peak_idx = i;
            } else {
                // Determine if this new peak is "better" than the last one?
                // Standard approach: Greedy (First come first serve) or Global Max?
                // Greedy with lockout is simplest.
                // Refinment: If we are within lockout but this peak is HIGHER than the previous one,
                // do we swap them? 
                // Let's stick to simple lockout for now (Standard Pipeline).
            }
        }
    }
    peaks
}

/// Z-Score Normalization (Standardization)
/// Independently normalizes each column (feature dimension) to Mean=0, Std=1.
fn z_score_normalize(features: &mut Array2<f32>) {
    let (n, d) = features.dim();
    
    for j in 0..d {
        // Calculate Mean
        let mut sum = 0.0;
        for i in 0..n {
            sum += features[[i, j]];
        }
        let mean = sum / n as f32;
        
        // Calculate StdDev
        let mut var_sum = 0.0;
        for i in 0..n {
            var_sum += (features[[i, j]] - mean).powi(2);
        }
        let std_dev = (var_sum / n as f32).sqrt();
        
        // Apply Z-Score
        if std_dev > 1e-6 {
            for i in 0..n {
                features[[i, j]] = (features[[i, j]] - mean) / std_dev;
            }
        }
    }
}

/// Time-Delay Embedding (Lag Features)
/// Stacks features of [t, t+1, ... t+lag-1] into a super-vector.
fn compute_lag_features(features: &Array2<f32>, lag: usize) -> Array2<f32> {
    let (n, d) = features.dim();
    // Output size will be (n, d * lag).
    // Note: We need to pad the end? Or truncate?
    // Standard is truncate or pad with zeros. Padding with zeros is safer for shape.
    let d_new = d * lag;
    let mut stacked = Array2::<f32>::zeros((n, d_new));
    
    for i in 0..n {
        for l in 0..lag {
            let t_idx = i + l;
            if t_idx < n {
                for j in 0..d {
                    stacked[[i, l * d + j]] = features[[t_idx, j]];
                }
            } else {
                // Pad with zeros (implied by initialization)
            }
        }
    }
    stacked
}

// Basic Mode Filter for label smoothing
fn smooth_labels(labels: &[usize], window_size: usize) -> Vec<usize> {
    let mut smoothed = labels.to_vec();
    let n = labels.len();
    let radius = window_size / 2;
    
    for i in 0..n {
        let start = i.saturating_sub(radius);
        let end = (i + radius + 1).min(n);
        let slice = &labels[start..end];
        
        let mut counts = std::collections::HashMap::new();
        for &l in slice {
            *counts.entry(l).or_insert(0) += 1;
        }
        
        let mut best_l = labels[i];
        let mut max_count = 0;
        
        for (&l, &c) in counts.iter() {
            if c > max_count {
                max_count = c;
                best_l = l;
            }
        }
    smoothed[i] = best_l;
    }
    smoothed
}

// ═══════════════════════════════════════════════════════════════════════════════
// MCFEE 2014 SPECTRAL CLUSTERING HELPERS
// ═══════════════════════════════════════════════════════════════════════════════
//
// The following functions implement the spectral clustering pipeline from:
//
//   McFee, B., & Ellis, D. P. W. (2014). "Analyzing Song Structure with
//   Spectral Clustering." Proceedings of the 15th International Society for
//   Music Information Retrieval Conference (ISMIR).
//
// This implementation combines McFee's rigorous graph-theoretic approach with
// enhancements from Remixatron (higher CQT resolution, eigenvector smoothing).
//
// The goal is to identify beats that "sound similar" so the Infinite Jukebox
// can jump between them seamlessly — a use case explicitly cited in the paper
// (Reference [8]: P. Lamere, "The Infinite Jukebox", 2012).
// ═══════════════════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────────────────────────────────────
// UTILITY: 1D Median Filter
// ─────────────────────────────────────────────────────────────────────────────

/// Applies a 1D median filter to a slice of values.
///
/// Median filtering is a nonlinear smoothing technique that preserves edges
/// while removing noise. It replaces each value with the median of its local
/// neighborhood.
///
/// # Arguments
/// * `input` - The input signal to filter.
/// * `window_size` - The size of the sliding window (should be odd for symmetry).
///
/// # Returns
/// A new `Vec<f32>` of the same length as `input`, with each element replaced
/// by the median of its `window_size` neighborhood.
///
/// # Edge Handling
/// At the boundaries, the window is truncated (no padding). This matches the
/// behavior of `scipy.ndimage.median_filter` with `mode='constant', cval=0`,
/// though we use truncation rather than zero-padding for robustness.
fn median_filter_1d(input: &[f32], window_size: usize) -> Vec<f32> {
    let n = input.len();
    if n == 0 || window_size == 0 {
        return input.to_vec();
    }
    
    let half = window_size / 2;
    let mut output = Vec::with_capacity(n);
    
    for i in 0..n {
        // Define the window bounds, clamped to valid indices.
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(n);
        
        // Collect values in the window and sort to find median.
        let mut window_vals: Vec<f32> = input[start..end].to_vec();
        window_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        // Median is the middle element (or average of two middle for even lengths,
        // but we use floor division for simplicity, matching scipy behavior).
        let mid = window_vals.len() / 2;
        output.push(window_vals[mid]);
    }
    
    output
}

// ─────────────────────────────────────────────────────────────────────────────
// CUMULATIVE NORMALIZATION (McFee Paper, Equation 10 Footnote)
// ─────────────────────────────────────────────────────────────────────────────
//
// Spectral clustering represents each data point (beat) as a vector of
// eigenvector coordinates. Before clustering with k-means, we must normalize
// these vectors.
//
// The symmetric normalized Laplacian introduces scaling that must be corrected.
// McFee uses cumulative normalization:
//
//     Cnorm[i, k] = sqrt( sum_{j=0}^{k} evecs[i, j]² )
//     X[i, :] = evecs[i, :k] / Cnorm[i, k-1]
//
// This ensures that each row vector has unit norm *with respect to the first
// k eigenvectors*, preserving the geometric relationships that spectral
// clustering exploits.
// ─────────────────────────────────────────────────────────────────────────────

/// Performs cumulative normalization on eigenvectors for spectral clustering.
///
/// This normalization is required when using the symmetric normalized Laplacian
/// (as opposed to the unnormalized Laplacian). It corrects for the scaling
/// introduced by D^(-1/2) in the Laplacian construction.
///
/// # Algorithm (McFee & Ellis 2014, Eq. 10 footnote)
///
/// For each row `i` (representing a beat):
/// 1. Compute cumulative sum of squared eigenvector values: `cumsum[j] = Σ_{l=0}^{j} evecs[i,l]²`
/// 2. Divide each element by `sqrt(cumsum[k-1])` to normalize.
///
/// # Arguments
/// * `evecs` - The full eigenvector matrix `[n_beats, n_beats]` from Laplacian decomposition.
/// * `k` - The number of eigenvectors to use (determines embedding dimension).
/// * `sorted_indices` - Indices that sort eigenvectors by ascending eigenvalue.
///
/// # Returns
/// A normalized embedding matrix `[n_beats, k]` suitable for k-means clustering.
///
/// # Mathematical Note
/// This is distinct from simple L2 row normalization. Cumulative normalization
/// uses only the energy in the *first k* eigenvectors, which corresponds to the
/// smoothest graph partitions. Higher eigenvectors capture finer structure.
fn cumulative_normalize_eigenvectors(
    evecs: &Array2<f32>,
    k: usize,
    sorted_indices: &[usize],
) -> Array2<f32> {
    let n = evecs.nrows();
    let mut result = Array2::<f32>::zeros((n, k));
    
    for i in 0..n {
        // Step 1: Compute cumulative sum of squared values for eigenvectors 0..k.
        // We iterate through sorted_indices to get eigenvectors in order of
        // ascending eigenvalue (smoothest to most oscillatory).
        let mut cumsum = 0.0_f32;
        
        #[allow(clippy::needless_range_loop)]  // j indexes sorted_indices then evecs column
        for j in 0..k {
            let eig_idx = sorted_indices[j];
            let val = evecs[[i, eig_idx]];
            cumsum += val * val;
        }
        
        // Step 2: Compute normalization factor (sqrt of cumulative sum at position k-1).
        let norm_factor = cumsum.sqrt();
        
        // Step 3: Divide each of the first k eigenvector values by this factor.
        // Guard against division by zero (should not happen for connected graphs,
        // but defensive coding is prudent).
        if norm_factor > 1e-10 {
            for j in 0..k {
                let eig_idx = sorted_indices[j];
                result[[i, j]] = evecs[[i, eig_idx]] / norm_factor;
            }
        }
        // If norm_factor is near zero, the row remains zeros (isolated node).
    }
    
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// EIGENVECTOR SMOOTHING (Remixatron Enhancement)
// ─────────────────────────────────────────────────────────────────────────────
//
// After computing the Laplacian eigenvectors, we apply a median filter along
// the time axis (rows). This smooths out small discontinuities in the
// eigenvector representation, leading to more stable cluster assignments.
//
// This is an enhancement from Remixatron.py (line 278):
//     evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))
//
// The filter size (9, 1) means:
//   - Window of 9 beats along the time axis (rows)
//   - No filtering across eigenvector dimensions (columns)
//
// This preserves the independence of each eigenvector while smoothing
// the temporal evolution of cluster membership.
// ─────────────────────────────────────────────────────────────────────────────

/// Applies median smoothing to eigenvectors along the time axis.
///
/// This enhancement from Remixatron.py reduces noise in the spectral embedding
/// by smoothing each eigenvector column independently using a median filter.
///
/// # Arguments
/// * `evecs` - The eigenvector matrix `[n_beats, n_eigenvectors]` to smooth.
/// * `window_size` - The size of the median filter window (typically 9).
///
/// # Returns
/// A smoothed eigenvector matrix of the same shape.
///
/// # Why Median Filtering?
/// Median filters preserve edges (sudden structural changes) better than
/// Gaussian smoothing, while still removing isolated outliers. This is
/// important for music structure where transitions should remain sharp.
#[allow(dead_code)]  // Disabled for McFee paper compliance
fn smooth_eigenvectors(evecs: &mut Array2<f32>, window_size: usize) {
    let (n_rows, n_cols) = evecs.dim();
    
    // Process each eigenvector (column) independently.
    for col_idx in 0..n_cols {
        // Extract the column as a Vec for processing.
        let column: Vec<f32> = (0..n_rows).map(|i| evecs[[i, col_idx]]).collect();
        
        // Apply 1D median filter.
        let smoothed = median_filter_1d(&column, window_size);
        
        // Write the smoothed values back.
        for (i, val) in smoothed.into_iter().enumerate() {
            evecs[[i, col_idx]] = val;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OPTIMAL MU CALCULATION (McFee Paper, Equation 7)
// ─────────────────────────────────────────────────────────────────────────────
//
// The affinity matrix A combines two types of information:
//   1. Repetition structure (R'): "Which beats sound like which other beats?"
//   2. Sequential structure (Δ): "Which beats are adjacent in time?"
//
// These are combined with a weighting parameter μ:
//
//     A = μ · R' + (1 - μ) · Δ
//
// How should μ be set? McFee proposes an elegant solution: choose μ such that
// a random walk on the graph has equal probability of following repetition
// links versus sequential links. This leads to the optimization:
//
//     μ* = ⟨d(Δ), d(R') + d(Δ)⟩ / ‖d(R') + d(Δ)‖²
//
// where d(G) is the degree vector of graph G (row sums of adjacency matrix).
//
// This automatic balancing prevents either component from dominating.
// ─────────────────────────────────────────────────────────────────────────────

/// Computes the optimal mixing parameter μ for combining repetition and
/// sequential affinity matrices.
///
/// # Algorithm (McFee & Ellis 2014, Equation 7)
///
/// Given degree vectors:
/// - `d_path`: degrees from the path/sequential matrix Δ
/// - `d_rec`: degrees from the filtered recurrence matrix R'
///
/// The optimal μ minimizes the squared difference between the expected
/// degrees from each component, yielding:
///
/// ```text
///     μ* = ⟨d_path, d_path + d_rec⟩ / ‖d_path + d_rec‖²
/// ```
///
/// # Arguments
/// * `d_path` - Degree vector from sequential/path adjacency (sum of rows of Δ).
/// * `d_rec` - Degree vector from recurrence adjacency (sum of rows of R').
///
/// # Returns
/// The optimal weighting parameter μ ∈ (0, 1).
///
/// # Guarantees
/// - If R' has at least one edge, μ* < 1 (repetition structure is used).
/// - If Δ has at least one edge, μ* > 0 (sequential structure is used).
fn compute_optimal_mu(d_path: &Array1<f32>, d_rec: &Array1<f32>) -> f32 {
    // d_combined = d_path + d_rec (element-wise sum)
    let d_combined = d_path + d_rec;
    
    // Numerator: ⟨d_path, d_combined⟩ = sum of element-wise products
    let numerator = d_path.dot(&d_combined);
    
    // Denominator: ‖d_combined‖² = sum of squared elements
    let denominator = d_combined.dot(&d_combined);
    
    // Guard against division by zero (would indicate an empty graph).
    if denominator < 1e-10 {
        // Fallback: equal weighting if both graphs are empty.
        return 0.5;
    }
    
    numerator / denominator
}



// ─────────────────────────────────────────────────────────────────────────────
// RECURRENCE MATRIX CONSTRUCTION (McFee Paper, Equations 1-2)
// ─────────────────────────────────────────────────────────────────────────────
//
// The recurrence matrix R captures which beats are musically similar.
// Construction proceeds in two steps:
//
// Step 1 (Eq. 1): Binary k-NN recurrence
//     R[i,j] = 1 if x_i and x_j are MUTUAL k-nearest neighbors
//            = 0 otherwise
//
//     "Mutual" means both i is in j's top-k AND j is in i's top-k.
//     This is more conservative than one-way k-NN and produces cleaner structure.
//
// Step 2 (Eq. 2): Diagonal median filtering
//     R'[i,j] = median{ R[i+t, j+t] | t ∈ [-w, w] }
//
//     This filters along the DIAGONALS of R, which correspond to time-lagged
//     repetitions. A true structural repetition (like a repeated chorus) will
//     create a diagonal stripe in R. Isolated spurious matches are suppressed.
//
// The paper uses w=17 for the filter window, which at typical beat rates
// (~2 beats/second) spans about 8-9 seconds — long enough to confirm a
// genuine repetition while still being responsive.
// ─────────────────────────────────────────────────────────────────────────────

/// Constructs a BINARY mutual k-NN recurrence matrix (McFee Eq. 1).
///
/// This implements the binary recurrence indicator from the McFee 2014 paper.
/// R[i,j] = 1 if i and j are mutual k-nearest neighbors, 0 otherwise.
///
/// The Gaussian affinity S_rep is computed SEPARATELY by `compute_srep_affinity`.
///
/// # Algorithm (McFee & Ellis 2014, Equation 1)
///
/// For each beat i:
/// 1. Find the k most similar beats (by Euclidean distance in feature space).
/// 2. An edge (i,j) exists only if it is MUTUAL: i is in j's top-k AND j is in i's top-k.
/// 3. Edge weight = 1.0 (binary indicator)
///
/// This matrix is then multiplied element-wise by S_rep in Equation 9.
///
/// # Arguments
/// * `features` - Beat-synchronous feature matrix `[n_beats, n_dims]`.
/// * `k` - Number of nearest neighbors to consider.
/// * `width` - Minimum separation between linked beats (typically 3 = ~1 bar).
///
/// # Returns
/// A BINARY symmetric matrix `[n_beats, n_beats]` where entry = 1.0 if mutual k-NN, 0.0 otherwise.
fn compute_binary_recurrence_matrix(
    features: &Array2<f32>,
    k: usize,
    width: usize,
) -> Array2<f32> {
    let n = features.nrows();
    
    // Step 1: For each beat, find its k nearest neighbors (excluding width).
    let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
    
    for i in 0..n {
        let mut dists: Vec<(f32, usize)> = Vec::with_capacity(n);
        
        for j in 0..n {
            // Exclude self and points within width (prevent trivial matches).
            if (i as isize - j as isize).unsigned_abs() <= width {
                continue;
            }
            
            // Squared Euclidean distance.
            let mut dist_sq = 0.0_f32;
            for d in 0..features.ncols() {
                let diff = features[[i, d]] - features[[j, d]];
                dist_sq += diff * diff;
            }
            dists.push((dist_sq, j));
        }
        
        // Sort by distance ascending and take top k.
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        
        let actual_k = k.min(dists.len());
        neighbors[i] = dists.iter().take(actual_k).map(|(_, idx)| *idx).collect();
    }
    
    // Step 2: Build BINARY mutual k-NN matrix.
    // R[i,j] = 1.0 if and only if i is in j's top-k AND j is in i's top-k.
    let mut recurrence = Array2::<f32>::zeros((n, n));
    let mut edge_count = 0;
    
    for i in 0..n {
        for &j in &neighbors[i] {
            // Check if j also has i in its neighbors (mutual).
            if neighbors[j].contains(&i) {
                recurrence[[i, j]] = 1.0;
                recurrence[[j, i]] = 1.0;
                edge_count += 1;
            }
        }
    }
    
    // DEBUG: Log binary recurrence matrix stats
    {
        use std::io::Write;
        use std::fs::OpenOptions;
        if let Ok(mut file) = OpenOptions::new().append(true).open("remixatron_debug.log") {
            let _ = writeln!(file, "\n=== BINARY RECURRENCE MATRIX (McFee Eq. 1) ===");
            let _ = writeln!(file, "n_beats: {}, k: {}, width: {}", n, k, width);
            let _ = writeln!(file, "Feature dims: {}", features.ncols());
            let _ = writeln!(file, "Mutual k-NN edges: {}", edge_count / 2);  // Divide by 2 for undirected
            
            // Diagonal density analysis to detect structural repetitions
            let _ = writeln!(file, "\n=== DIAGONAL DENSITY ANALYSIS (Raw R) ===");
            
            let mut dense_diagonals: Vec<(usize, usize, f32)> = Vec::new();
            
            // Check diagonals at offsets 10 to min(n, 200)
            // These represent phrase-level repetitions (skipping trivial near-neighbors)
            for offset in 10..n.min(200) {
                let diag_len = n - offset;
                let mut edge_count_diag = 0;
                
                for i in 0..diag_len {
                    let j = i + offset;
                    if recurrence[[i, j]] > 0.0 {
                        edge_count_diag += 1;
                    }
                }
                
                let density = edge_count_diag as f32 / diag_len as f32;
                if density > 0.05 {  // Only track diagonals with >5% density
                    dense_diagonals.push((offset, edge_count_diag, density));
                }
            }
            
            if dense_diagonals.is_empty() {
                let _ = writeln!(file, "WARNING: No dense diagonals found (>5% density)");
                let _ = writeln!(file, "This suggests k-NN is not finding structural repetitions.");
            } else {
                let _ = writeln!(file, "Dense diagonals found (offset, edges, density):");
                for (offset, edges, density) in dense_diagonals.iter().take(20) {
                    let _ = writeln!(file, "  Offset {:4}: {:4} edges ({:.1}%)", offset, edges, density * 100.0);
                }
                if dense_diagonals.len() > 20 {
                    let _ = writeln!(file, "  ... and {} more", dense_diagonals.len() - 20);
                }
            }
        }
    }
    
    recurrence
}

/// Computes Gaussian affinity matrix S_rep (McFee Eq. 8).
///
/// S_rep[i,j] = exp(-||C_i - C_j||² / σ²)
///
/// Where σ² is the median squared distance between all pairs in the
/// recurrence matrix (where R[i,j] = 1).
///
/// # Arguments
/// * `features` - Chroma features `[n_beats, n_dims]`.
/// * `recurrence` - Binary recurrence matrix from `compute_binary_recurrence_matrix`.
///
/// # Returns
/// A symmetric matrix `[n_beats, n_beats]` with Gaussian affinities.
fn compute_srep_affinity(
    features: &Array2<f32>,
    recurrence: &Array2<f32>,
) -> Array2<f32> {
    let n = features.nrows();
    
    // Step 1: Compute distances for all pairs where R[i,j] = 1.
    // We need these to estimate σ².
    let mut distances_sq: Vec<f32> = Vec::new();
    
    for i in 0..n {
        for j in (i + 1)..n {
            if recurrence[[i, j]] > 0.0 {
                let mut dist_sq = 0.0_f32;
                for d in 0..features.ncols() {
                    let diff = features[[i, d]] - features[[j, d]];
                    dist_sq += diff * diff;
                }
                distances_sq.push(dist_sq);
            }
        }
    }
    
    // σ² = median of squared distances (robust bandwidth estimation)
    distances_sq.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let sigma_sq = if distances_sq.is_empty() {
        1.0
    } else {
        distances_sq[distances_sq.len() / 2].max(1e-6)
    };
    
    // Step 2: Compute S_rep for pairs where R[i,j] = 1.
    // For pairs where R[i,j] = 0, S_rep doesn't matter (will be multiplied by 0).
    let mut s_rep = Array2::<f32>::zeros((n, n));
    
    for i in 0..n {
        for j in 0..n {
            if recurrence[[i, j]] > 0.0 {
                let mut dist_sq = 0.0_f32;
                for d in 0..features.ncols() {
                    let diff = features[[i, d]] - features[[j, d]];
                    dist_sq += diff * diff;
                }
                // Gaussian kernel: exp(-d² / σ²)
                s_rep[[i, j]] = (-dist_sq / sigma_sq).exp();
            }
        }
    }
    
    // DEBUG: Log S_rep stats
    {
        use std::io::Write;
        use std::fs::OpenOptions;
        if let Ok(mut file) = OpenOptions::new().append(true).open("remixatron_debug.log") {
            let nonzero: Vec<f32> = s_rep.iter().filter(|&&x| x > 0.0).cloned().collect();
            let _ = writeln!(file, "\n=== S_REP GAUSSIAN AFFINITY (McFee Eq. 8) ===");
            let _ = writeln!(file, "σ² = {:.4}", sigma_sq);
            let _ = writeln!(file, "σ  = {:.4}", sigma_sq.sqrt());
            let _ = writeln!(file, "Non-zero entries: {}", nonzero.len());
            if !nonzero.is_empty() {
                let min = nonzero.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = nonzero.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mean = nonzero.iter().sum::<f32>() / nonzero.len() as f32;
                let _ = writeln!(file, "Min: {:.4}, Max: {:.4}, Mean: {:.4}", min, max, mean);
            }
        }
    }
    
    s_rep
}

/// Computes the maximum diagonal density of a binary recurrence matrix.
///
/// This is used for adaptive diagonal filter width selection. Songs with
/// stronger diagonal patterns (higher density) can use more aggressive
/// filtering (larger w), while songs with weaker patterns need gentler
/// filtering (smaller w) to preserve structure.
///
/// # Arguments
/// * `recurrence` - Binary recurrence matrix `[n, n]`.
///
/// # Returns
/// The maximum diagonal density (0.0 to 1.0) across all diagonals at
/// offsets 10 to min(n, 200).
fn compute_max_diagonal_density(recurrence: &Array2<f32>) -> f32 {
    let n = recurrence.nrows();
    let mut max_density = 0.0_f32;
    
    // Check diagonals at offsets 10 to min(n, 200)
    // Skip near-neighbor diagonals which are trivially dense
    for offset in 10..n.min(200) {
        let diag_len = n - offset;
        let mut edge_count = 0;
        
        for i in 0..diag_len {
            let j = i + offset;
            if recurrence[[i, j]] > 0.0 {
                edge_count += 1;
            }
        }
        
        let density = edge_count as f32 / diag_len as f32;
        if density > max_density {
            max_density = density;
        }
    }
    
    max_density
}

/// Applies diagonal median filtering to a recurrence matrix.
///
/// Structural repetitions (like a repeated chorus) appear as diagonal stripes
/// in the recurrence matrix. Isolated spurious matches appear as scattered dots.
/// Diagonal median filtering enhances the stripes while suppressing the dots.
///
/// # Algorithm (McFee & Ellis 2014, Equation 2)
///
/// For each diagonal offset d ∈ [-n+1, n-1]:
/// 1. Extract the diagonal elements of R at offset d.
/// 2. Apply a 1D median filter with window size w.
/// 3. Write the filtered values back to R.
///
/// # Arguments
/// * `recurrence` - The binary recurrence matrix to filter `[n, n]`.
/// * `window_size` - The median filter window (typically 17 for ~8-9 second span).
///
/// # Returns
/// A filtered recurrence matrix with enhanced diagonal structure.
///
/// # Why Diagonals?
/// A diagonal at offset d contains pairs (i, i+d) — positions that are d beats
/// apart. A structural repetition where measure M in verse 1 matches measure M
/// in verse 2 will create a consistent diagonal line, because ALL the beats
/// in those measures will match their counterparts.
fn apply_diagonal_median_filter(recurrence: &Array2<f32>, window_size: usize) -> Array2<f32> {
    let n = recurrence.nrows();
    let mut filtered = recurrence.clone();
    
    // Process each diagonal offset from -(n-1) to +(n-1).
    // Offset 0 is the main diagonal; positive offsets are above, negative below.
    //
    // Due to symmetry of the recurrence matrix, we could process only positive
    // offsets and mirror, but for clarity we process all and rely on the
    // median filter to produce consistent results.
    
    for offset in -(n as isize - 1)..=(n as isize - 1) {
        // Determine the start indices for this diagonal.
        let (start_i, start_j) = if offset >= 0 {
            (0, offset as usize)
        } else {
            ((-offset) as usize, 0)
        };
        
        // Collect the diagonal elements.
        let mut diagonal: Vec<f32> = Vec::new();
        let mut i = start_i;
        let mut j = start_j;
        
        while i < n && j < n {
            diagonal.push(recurrence[[i, j]]);
            i += 1;
            j += 1;
        }
        
        // Skip very short diagonals (nothing to filter).
        if diagonal.len() < 3 {
            continue;
        }
        
        // Apply 1D median filter to this diagonal.
        let filtered_diag = median_filter_1d(&diagonal, window_size);
        
        // Write the filtered values back.
        let mut i = start_i;
        let mut j = start_j;
        let mut diag_idx = 0;
        
        while i < n && j < n {
            filtered[[i, j]] = filtered_diag[diag_idx];
            i += 1;
            j += 1;
            diag_idx += 1;
        }
    }
    
    filtered
}

// ─────────────────────────────────────────────────────────────────────────────
// PATH/SEQUENCE MATRIX (McFee Paper, Equations 3-5)
// ─────────────────────────────────────────────────────────────────────────────
//
// Sequential structure is captured by the path matrix Δ, which connects
// adjacent beats:
//
//     Δ[i,j] = 1 if |i - j| = 1
//            = 0 otherwise
//
// However, we want WEIGHTED edges based on timbral similarity, not binary.
// The path similarity weight is computed as:
//
//     path_sim[t] = exp(-‖MFCC[t] - MFCC[t+1]‖² / σ²)
//
// where σ² is the median of all adjacent-beat distances.
//
// This weighting means that transitions within a consistent section (e.g.,
// middle of a verse) have high path weights, while transitions at structural
// boundaries (e.g., verse to chorus) have lower weights.
// ─────────────────────────────────────────────────────────────────────────────

/// Constructs a weighted path/sequence matrix for adjacent beats.
///
/// The path matrix captures local temporal structure: beats that are adjacent
/// in time and have similar timbre are strongly connected.
///
/// # Algorithm (McFee & Ellis 2014, Equations 3-5)
///
/// For each consecutive pair of beats (t, t+1):
/// 1. Compute squared difference: `dist² = ‖features[t] - features[t+1]‖²`
/// 2. Apply Gaussian kernel: `weight = exp(-dist² / σ²)`
///
/// The bandwidth σ² is set to the median of all adjacent-beat distances.
///
/// # Arguments
/// * `features` - Beat-synchronous timbre features (typically MFCC) `[n_beats, n_dims]`.
///
/// # Returns
/// A tuple `(path_matrix, path_degrees)`:
/// - `path_matrix`: Tridiagonal matrix `[n, n]` with Gaussian weights on super/sub-diagonals.
/// - `path_degrees`: Degree vector (row sums) used for computing optimal μ.
fn compute_path_matrix(features: &Array2<f32>) -> (Array2<f32>, Array1<f32>) {
    let n = features.nrows();
    
    if n < 2 {
        // Degenerate case: single beat or empty.
        return (Array2::zeros((n, n)), Array1::zeros(n));
    }
    
    // Step 1: Compute squared distances between adjacent beats.
    let mut path_distances_sq: Vec<f32> = Vec::with_capacity(n - 1);
    
    for t in 0..(n - 1) {
        let mut dist_sq = 0.0_f32;
        for d in 0..features.ncols() {
            let diff = features[[t, d]] - features[[t + 1, d]];
            dist_sq += diff * diff;
        }
        path_distances_sq.push(dist_sq);
    }
    
    // Step 2: Compute σ² as median of path distances.
    let mut sorted_dists = path_distances_sq.clone();
    sorted_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let sigma_sq = sorted_dists[sorted_dists.len() / 2];
    let sigma_sq_safe = if sigma_sq < 1e-10 { 1.0 } else { sigma_sq };
    
    // Step 3: Compute path similarity weights.
    // Add minimum floor to ensure connectivity even through dramatic timbral changes
    // (e.g., bell tolls → guitar in "Back in Black" intro).
    const MIN_PATH_WEIGHT: f32 = 0.01;
    let path_sim: Vec<f32> = path_distances_sq
        .iter()
        .map(|d| (-d / sigma_sq_safe).exp().max(MIN_PATH_WEIGHT))
        .collect();
    
    // Step 4: Build the path matrix.
    // This is tridiagonal: entries at (t, t+1) and (t+1, t) for t = 0..n-2.
    let mut r_path = Array2::<f32>::zeros((n, n));
    
    for t in 0..(n - 1) {
        let weight = path_sim[t];
        r_path[[t, t + 1]] = weight;
        r_path[[t + 1, t]] = weight; // Symmetric
    }
    
    // Step 5: Compute degree vector (row sums).
    let mut degrees = Array1::<f32>::zeros(n);
    for i in 0..n {
        degrees[i] = r_path.row(i).sum();
    }
    
    (r_path, degrees)
}

impl StructureAnalyzer {
    /// Constructs a k-NN Jump Graph with adaptive P75 threshold.
    ///
    /// This produces the "Jump Graph" for the Infinite Jukebox playback system.
    /// Jump candidates are beats that are sonically similar to the current beat,
    /// allowing seamless transitions.
    ///
    /// # Algorithm
    /// 1. For each beat, find k nearest neighbors (cosine similarity on features)
    /// 2. Exclude immediate neighbors (within 4 beats) to avoid stutter jumps
    /// 3. Collect all similarity scores across all beats
    /// 4. Compute P75 percentile as adaptive threshold
    /// 5. Filter to keep only edges ≥ P75 threshold (~25% of candidates)
    ///
    /// The adaptive P75 approach ensures each song gets an appropriate threshold
    /// based on its own similarity distribution.
    ///
    /// # Arguments
    /// * `features` - Feature matrix (typically Lag-Embedded MFCC+Chroma)
    /// * `k` - Number of neighbors to find per beat (before filtering)
    /// * `_threshold` - UNUSED (kept for API compatibility, threshold is adaptive)
    fn compute_jump_graph(features: &Array2<f32>, k: usize, _threshold: f32) -> Vec<Vec<(usize, f32)>> {
        let n_beats = features.nrows();
        let exclusion_radius = 4; // Don't jump to neighbors (within 1 bar)

        println!("DEBUG: Computing Jump Graph. Shape: {:?}, k={}", features.dim(), k);
        
        // Phase 1: Collect all top-k candidates for each beat
        let mut all_candidates: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_beats];
        let mut max_sim_overall = 0.0f32;

        #[allow(clippy::needless_range_loop)]  // i indexes all_candidates and features
        for i in 0..n_beats {
            let target = features.row(i);
            let mut candidates = Vec::new();

            for j in 0..n_beats {
                // Skip self and immediate neighbors
                if (i as isize - j as isize).abs() <= exclusion_radius as isize {
                    continue;
                }

                let neighbor = features.row(j);
                
                // Cosine Similarity
                let dot = target.dot(&neighbor);
                let norm_i = target.dot(&target).sqrt();
                let norm_j = neighbor.dot(&neighbor).sqrt();
                
                let sim = if norm_i > 0.0 && norm_j > 0.0 {
                    dot / (norm_i * norm_j)
                } else {
                    0.0
                };
                
                if sim > max_sim_overall { max_sim_overall = sim; }
                candidates.push((j, sim));
            }

            // Sort by similarity descending and take top-k
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            candidates.truncate(k);
            all_candidates[i] = candidates;
        }
        
        // Phase 2: Compute P85 percentile as adaptive threshold (stricter than P75)
        let mut all_sims: Vec<f32> = all_candidates.iter()
            .flat_map(|neighbors| neighbors.iter().map(|(_, sim)| *sim))
            .collect();
        all_sims.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        
        let adaptive_threshold = if !all_sims.is_empty() {
            // P75 = top 25% (index at 25% from start of sorted-descending list)
            let p85_index = all_sims.len() * 25 / 100;
            all_sims[p85_index.min(all_sims.len() - 1)]
        } else {
            0.5  // Fallback
        };
        
        println!("DEBUG: Adaptive threshold P85 = {:.4}", adaptive_threshold);
        
        // Phase 3: Filter edges using adaptive threshold
        let mut jumps = vec![Vec::new(); n_beats];
        let mut total_edges = 0;
        let mut filtered_edges = 0;
        
        for (i, candidates) in all_candidates.iter().enumerate() {
            for &(idx, sim) in candidates {
                if sim >= adaptive_threshold {
                    jumps[i].push((idx, sim));
                    total_edges += 1;
                } else {
                    filtered_edges += 1;
                }
            }
        }
        
        println!("DEBUG: Jump Graph Complete. Kept: {}, Filtered: {}, Max Sim: {:.4}", 
            total_edges, filtered_edges, max_sim_overall);

        // Log similarity distribution
        {
            use std::io::Write;
            use std::fs::OpenOptions;
            if let Ok(mut file) = OpenOptions::new().append(true).open("remixatron_debug.log") {
                let _ = writeln!(file, "\n=== JUMP SIMILARITY DISTRIBUTION ===");
                
                if !all_sims.is_empty() {
                    let min_sim = all_sims.last().unwrap_or(&0.0);
                    let max_sim = all_sims.first().unwrap_or(&0.0);
                    let median_sim = all_sims[all_sims.len() / 2];
                    let p90_sim = all_sims[all_sims.len() / 10];
                    let p75_sim = all_sims[all_sims.len() / 4];
                    let p50_sim = all_sims[all_sims.len() / 2];
                    let p25_sim = all_sims[3 * all_sims.len() / 4];
                    
                    let _ = writeln!(file, "Total candidates: {}", all_sims.len());
                    let _ = writeln!(file, "Adaptive threshold (P85): {:.4}", adaptive_threshold);
                    let _ = writeln!(file, "Edges after filtering: {} ({:.1}%)", 
                        total_edges, 100.0 * total_edges as f32 / all_sims.len() as f32);
                    let _ = writeln!(file, "Min: {:.4}, Max: {:.4}, Median: {:.4}", min_sim, max_sim, median_sim);
                    let _ = writeln!(file, "Percentiles: P90={:.4}, P75={:.4}, P50={:.4}, P25={:.4}", 
                        p90_sim, p75_sim, p50_sim, p25_sim);
                    
                    // Histogram buckets
                    let buckets = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.0];
                    let _ = writeln!(file, "\nHistogram (similarity >= threshold):");
                    for thresh in buckets {
                        let count = all_sims.iter().filter(|&&s| s >= thresh).count();
                        let pct = 100.0 * count as f32 / all_sims.len() as f32;
                        let marker = if (thresh - adaptive_threshold).abs() < 0.05 { " <- P85" } else { "" };
                        let _ = writeln!(file, "  >= {:.1}: {:5} edges ({:5.1}%){}", thresh, count, pct, marker);
                    }
                }
            }
        }

        jumps
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // PRIMARY SEGMENTATION ALGORITHM: McFee 2014 Spectral Clustering
    // ═══════════════════════════════════════════════════════════════════════════
    //
    // This function implements the spectral clustering approach from:
    //
    //   McFee, B., & Ellis, D. P. W. (2014). "Analyzing Song Structure with
    //   Spectral Clustering." ISMIR 2014.
    //
    // Enhanced with Remixatron improvements:
    //   - Higher CQT resolution (252 bins vs. paper's 72)
    //   - Eigenvector median smoothing
    //
    // The goal is to identify beats that "sound similar" for seamless jumps
    // in the Infinite Jukebox—a use case cited in the paper (Ref [8]).
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// The Primary "Bottom-Up" Segmentation Algorithm.
    ///
    /// This implements the McFee 2014 ISMIR paper's algorithm faithfully.
    ///
    /// # Algorithm Overview (McFee & Ellis 2014)
    ///
    /// ## Step 1: Feature Preparation
    /// - CHROMA for recurrence (harmonic similarity, S_rep)
    /// - MFCC for path (timbral similarity, S_loc)
    ///
    /// ## Step 2: Recurrence Matrix (Eq. 1-2)
    /// - Compute mutual k-NN recurrence matrix R on CHROMA features.
    /// - Apply diagonal median filter (w=17) to enhance structural repetitions.
    ///
    /// ## Step 3: Path Matrix (Eq. 3-5)
    /// - Compute weighted path matrix Δ with Gaussian kernel on MFCC.
    /// - Adjacent beats similar in timbre are strongly connected.
    ///
    /// ## Step 4: Optimal Balancing (Eq. 7)
    /// - Compute μ to balance repetition vs. sequential structure.
    ///
    /// ## Step 5: Affinity Matrix (Eq. 9)
    /// - Combine: A = μ·R'·S_rep + (1-μ)·Δ·S_loc
    ///
    /// ## Step 6: Spectral Embedding (Eq. 10)
    /// - Compute normalized Laplacian and its eigenvectors.
    /// - Use cumulative normalization for k-means.
    ///
    /// ## Step 7: Auto-K Clustering
    /// - Eigengap heuristic to select optimal K.
    ///
    /// ## Step 8: Label Smoothing
    /// - Mode filter (window=8) to merge micro-segments.
    pub fn compute_segments_knn(&self, mfcc: &Array2<f32>, chroma: &Array2<f32>, cqt: &Array2<f32>, k_force: Option<usize>) -> SegmentationResult {
        let n_beats = mfcc.nrows();
        
        // ─────────────────────────────────────────────────────────────────────
        // STEP 1: Feature Preparation
        // ─────────────────────────────────────────────────────────────────────
        // McFee 2014 Paper:
        // - CHROMA features for recurrence matrix (harmonic similarity, S_rep)
        // - MFCC features for path matrix (timbral similarity, S_loc)
        // Features are used directly without normalization or lag for these matrices.
        
        // Create debug log file (will be appended to by other functions)
        {
            use std::io::Write;
            if let Ok(mut file) = std::fs::File::create("remixatron_debug.log") {
                let _ = writeln!(file, "=== REMIXATRON DEBUG LOG ===");
                let _ = writeln!(file, "Timestamp: {:?}", std::time::SystemTime::now());
                let _ = writeln!(file, "N_beats: {}", n_beats);
                
                // Chroma feature diagnostics
                let _ = writeln!(file, "\n=== CHROMA FEATURES (beat-sync) ===");
                let _ = writeln!(file, "Shape: {:?}", chroma.dim());
                
                // Overall statistics
                let chroma_min = chroma.iter().cloned().fold(f32::INFINITY, f32::min);
                let chroma_max = chroma.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let chroma_mean = chroma.iter().sum::<f32>() / chroma.len() as f32;
                let _ = writeln!(file, "Min: {:.4}, Max: {:.4}, Mean: {:.4}", chroma_min, chroma_max, chroma_mean);
                
                // Sample chroma vectors at different positions
                let sample_beats = [0, n_beats / 4, n_beats / 2, 3 * n_beats / 4, n_beats - 1];
                let _ = writeln!(file, "\nSample chroma vectors (12 pitch classes):");
                for &beat in &sample_beats {
                    if beat < n_beats {
                        let chroma_vec: Vec<f32> = (0..12).map(|c| chroma[[beat, c]]).collect();
                        let _ = writeln!(file, "  Beat {:4}: [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
                            beat,
                            chroma_vec[0], chroma_vec[1], chroma_vec[2], chroma_vec[3],
                            chroma_vec[4], chroma_vec[5], chroma_vec[6], chroma_vec[7],
                            chroma_vec[8], chroma_vec[9], chroma_vec[10], chroma_vec[11]);
                    }
                }
                
                // Compute pairwise distances between sample beats (CHROMA)
                let _ = writeln!(file, "\nPairwise Euclidean distances (CHROMA) between sample beats:");
                for i in 0..sample_beats.len() {
                    for j in (i+1)..sample_beats.len() {
                        let b1 = sample_beats[i];
                        let b2 = sample_beats[j];
                        if b1 < n_beats && b2 < n_beats {
                            let mut dist_sq = 0.0f32;
                            for c in 0..12 {
                                let diff = chroma[[b1, c]] - chroma[[b2, c]];
                                dist_sq += diff * diff;
                            }
                            let _ = writeln!(file, "  Beat {} vs Beat {}: dist = {:.4}", b1, b2, dist_sq.sqrt());
                        }
                    }
                }
                
                // Compute pairwise distances between sample beats (MFCC)
                let _ = writeln!(file, "\nPairwise Euclidean distances (MFCC) between sample beats:");
                for i in 0..sample_beats.len() {
                    for j in (i+1)..sample_beats.len() {
                        let b1 = sample_beats[i];
                        let b2 = sample_beats[j];
                        if b1 < n_beats && b2 < n_beats {
                            let mut dist_sq = 0.0f32;
                            for c in 0..mfcc.ncols() {
                                let diff = mfcc[[b1, c]] - mfcc[[b2, c]];
                                dist_sq += diff * diff;
                            }
                            let _ = writeln!(file, "  Beat {} vs Beat {}: dist = {:.4}", b1, b2, dist_sq.sqrt());
                        }
                    }
                }
            }
        }
        
        // For jump graph only - these are still normalized/lagged
        let mut mfcc_norm = mfcc.clone();
        let mut chroma_norm = chroma.clone();
        z_score_normalize(&mut mfcc_norm);
        z_score_normalize(&mut chroma_norm);
        let _mfcc_lagged = compute_lag_features(&mfcc_norm, 2);
        let _chroma_lagged = compute_lag_features(&chroma_norm, 2);
        
        // ─────────────────────────────────────────────────────────────────────
        // STEP 2: Binary Recurrence Matrix R (McFee Eq. 1)
        // ─────────────────────────────────────────────────────────────────────
        // ENHANCED: Use COMBINED chroma + MFCC features for recurrence.
        // This helps distinguish sections with the same harmony but different
        // timbre/energy (e.g., quiet verse vs loud chorus with same chords).
        // S_rep still uses chroma only to keep harmonic affinity for smooth jumps.
        
        // Step 2a: Log raw feature scales before normalization
        // This helps diagnose if normalization is significantly changing relative contributions
        
        // Extract RMS energy from MFCC[0] (first coefficient is log-energy)
        // MFCC[0] approximates log-power, so exp(MFCC[0]) is proportional to RMS²
        let rms: ndarray::Array2<f32> = mfcc.slice(ndarray::s![.., 0..1]).to_owned();
        
        {
            use std::io::Write;
            use std::fs::OpenOptions;
            if let Ok(mut file) = OpenOptions::new().append(true).open("remixatron_debug.log") {
                let _ = writeln!(file, "\n=== FEATURE SCALE ANALYSIS (before normalization) ===");
                
                // Raw chroma stats
                let chroma_min = chroma.iter().cloned().fold(f32::INFINITY, f32::min);
                let chroma_max = chroma.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let chroma_mean = chroma.iter().sum::<f32>() / chroma.len() as f32;
                let chroma_var = chroma.iter().map(|x| (x - chroma_mean).powi(2)).sum::<f32>() / chroma.len() as f32;
                let _ = writeln!(file, "Raw Chroma:  mean={:.4}, var={:.4}, range=[{:.4}, {:.4}]", 
                    chroma_mean, chroma_var, chroma_min, chroma_max);
                
                // Raw MFCC stats (all 20 coefficients)
                let mfcc_min = mfcc.iter().cloned().fold(f32::INFINITY, f32::min);
                let mfcc_max = mfcc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mfcc_mean = mfcc.iter().sum::<f32>() / mfcc.len() as f32;
                let mfcc_var = mfcc.iter().map(|x| (x - mfcc_mean).powi(2)).sum::<f32>() / mfcc.len() as f32;
                let _ = writeln!(file, "Raw MFCC:    mean={:.4}, var={:.4}, range=[{:.4}, {:.4}]", 
                    mfcc_mean, mfcc_var, mfcc_min, mfcc_max);
                
                // Raw RMS (MFCC[0]) stats
                let rms_min = rms.iter().cloned().fold(f32::INFINITY, f32::min);
                let rms_max = rms.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let rms_mean = rms.iter().sum::<f32>() / rms.len() as f32;
                let rms_var = rms.iter().map(|x| (x - rms_mean).powi(2)).sum::<f32>() / rms.len() as f32;
                let _ = writeln!(file, "Raw RMS:     mean={:.4}, var={:.4}, range=[{:.4}, {:.4}]", 
                    rms_mean, rms_var, rms_min, rms_max);
                
                // Ratio analysis
                let mfcc_chroma_ratio = if chroma_var > 0.0 { mfcc_var / chroma_var } else { 0.0 };
                let rms_chroma_ratio = if chroma_var > 0.0 { rms_var / chroma_var } else { 0.0 };
                let _ = writeln!(file, "Variance ratio (MFCC/Chroma): {:.2}x", mfcc_chroma_ratio);
                let _ = writeln!(file, "Variance ratio (RMS/Chroma):  {:.2}x", rms_chroma_ratio);
                let _ = writeln!(file, "Without normalization, MFCC would contribute ~{:.0}% of distance", 
                    100.0 * mfcc_chroma_ratio / (1.0 + mfcc_chroma_ratio + rms_chroma_ratio));
            }
        }
        
        // Step 2b: Normalize all feature sets for equal per-dimension contribution
        let mut chroma_for_rec = chroma.clone();
        let mut mfcc_for_rec = mfcc.clone();
        let mut rms_for_rec = rms.clone();
        z_score_normalize(&mut chroma_for_rec);
        z_score_normalize(&mut mfcc_for_rec);
        z_score_normalize(&mut rms_for_rec);
        
        // Step 2c: Apply feature group weights to control relative contributions
        // Goal: Chroma ~25%, MFCC ~55%, RMS ~20% of total distance contribution
        // Weights chosen to compensate for different dimension counts
        const WEIGHT_CHROMA: f32 = 0.45;  // 12 dims × 0.45² = 2.43 → 25%
        const WEIGHT_MFCC: f32 = 0.52;    // 20 dims × 0.52² = 5.41 → 55%
        const WEIGHT_RMS: f32 = 1.4;      //  1 dim  × 1.4²  = 1.96 → 20%
        
        chroma_for_rec.mapv_inplace(|x| x * WEIGHT_CHROMA);
        mfcc_for_rec.mapv_inplace(|x| x * WEIGHT_MFCC);
        rms_for_rec.mapv_inplace(|x| x * WEIGHT_RMS);
        
        // Step 2d: Combine into single feature matrix [n_beats, 33]
        let combined_chroma_mfcc = concatenate_features(&chroma_for_rec, &mfcc_for_rec);
        let combined_features = concatenate_features(&combined_chroma_mfcc, &rms_for_rec);
        
        // Log combined feature dimensions and weights
        {
            use std::io::Write;
            use std::fs::OpenOptions;
            if let Ok(mut file) = OpenOptions::new().append(true).open("remixatron_debug.log") {
                let _ = writeln!(file, "\n=== COMBINED FEATURES FOR RECURRENCE ===");
                let _ = writeln!(file, "Chroma dims: {}, MFCC dims: {}, RMS dims: {}, Combined: {}",
                    chroma.ncols(), mfcc.ncols(), rms.ncols(), combined_features.ncols());
                let _ = writeln!(file, "Weights: chroma={}, mfcc={}, rms={}", 
                    WEIGHT_CHROMA, WEIGHT_MFCC, WEIGHT_RMS);
                let chroma_contrib = 12.0 * WEIGHT_CHROMA * WEIGHT_CHROMA;
                let mfcc_contrib = 20.0 * WEIGHT_MFCC * WEIGHT_MFCC;
                let rms_contrib = 1.0 * WEIGHT_RMS * WEIGHT_RMS;
                let total = chroma_contrib + mfcc_contrib + rms_contrib;
                let _ = writeln!(file, "Effective contributions: chroma={:.1}%, mfcc={:.1}%, rms={:.1}%",
                    100.0 * chroma_contrib / total, 100.0 * mfcc_contrib / total, 100.0 * rms_contrib / total);
            }
        }
        
        // Adaptive k for k-NN: k = 2 * ceil(sqrt(n)) per McFee paper
        let k_recurrence = 2 * ((n_beats as f32).sqrt().ceil() as usize);
        
        // Width exclusion = 3 beats (~1 bar at 120 BPM) to prevent trivial matches.
        // NOTE: Using combined_features instead of just chroma
        let recurrence_raw = compute_binary_recurrence_matrix(&combined_features, k_recurrence, 3);
        
        // ADAPTIVE W: Choose diagonal filter width based on max diagonal density.
        // Songs with strong diagonal patterns can handle aggressive filtering,
        // while songs with weaker patterns need smaller w to preserve structure.
        let max_density = compute_max_diagonal_density(&recurrence_raw);
        let w = if max_density >= 0.35 {
            17  // Strong patterns: use paper's value
        } else if max_density >= 0.20 {
            11  // Moderate patterns: intermediate value
        } else {
            7   // Weak patterns: preserve more edges
        };
        
        // Log the adaptive w decision
        {
            use std::io::Write;
            use std::fs::OpenOptions;
            if let Ok(mut file) = OpenOptions::new().append(true).open("remixatron_debug.log") {
                let _ = writeln!(file, "\n=== ADAPTIVE W SELECTION ===");
                let _ = writeln!(file, "Max diagonal density: {:.1}%", max_density * 100.0);
                let _ = writeln!(file, "Selected w: {}", w);
            }
        }
        
        // MCFEE Eq. 2: Diagonal median filter with adaptive w
        let recurrence_filtered = apply_diagonal_median_filter(&recurrence_raw, w);
        
        // ─────────────────────────────────────────────────────────────────────
        // STEP 2b: Gaussian Affinity S_rep (McFee Eq. 8)
        // ─────────────────────────────────────────────────────────────────────
        // S_rep[i,j] = exp(-||C_i - C_j||² / σ²) for harmonic similarity.
        // σ² is estimated as median squared distance among recurrence edges.
        let s_rep = compute_srep_affinity(chroma, &recurrence_filtered);
        
        // ─────────────────────────────────────────────────────────────────────
        // STEP 3: Path Matrix (McFee Eq. 3-5)
        // ─────────────────────────────────────────────────────────────────────
        // Python uses raw MFCC (no normalization, no lag) for path matrix.
        // This matches: Msync = librosa.util.sync(mfcc, btz)
        
        let (path_matrix, path_degrees) = compute_path_matrix(mfcc);
        
        // ─────────────────────────────────────────────────────────────────────
        // STEP 4: Optimal μ (McFee Eq. 6-7)
        // ─────────────────────────────────────────────────────────────────────
        // McFee Eq. 6: Degree vectors for the balance calculation.
        //   d_rep[i] = Σ_j R'[i,j] × S_rep[i,j]  (NOT just R')
        //   d_loc[i] = Σ_j Δ[i,j] × S_loc[i,j]   (already in path_degrees)
        //
        // The degrees must be from the WEIGHTED affinity terms, not binary indicators.
        
        // Compute degrees of R' × S_rep (element-wise product).
        let mut rec_degrees = Array1::<f32>::zeros(n_beats);
        for i in 0..n_beats {
            for j in 0..n_beats {
                // R'[i,j] is binary (0 or 1), S_rep[i,j] is Gaussian
                rec_degrees[i] += recurrence_filtered[[i, j]] * s_rep[[i, j]];
            }
        }
        
        let mu_raw = compute_optimal_mu(&path_degrees, &rec_degrees);
        
        // PRACTICAL ADJUSTMENT: Apply a minimum floor to μ.
        //
        // The McFee Eq. 7 formula balances path vs recurrence based on degree
        // distributions. However, our implementation often yields μ ≈ 0.05-0.15
        // because the recurrence graph has ~3x more total edge weight than path:
        //   - Path: avg ~0.7 degree (only 2 neighbors with Gaussian weights)
        //   - Recurrence: avg ~2.3 degree (many mutual k-NN edges)
        //
        // When μ is too low, the affinity matrix becomes nearly a chain graph,
        // causing spectral clustering to produce one dominant cluster.
        //
        // A floor of 0.3 ensures recurrence always has at least 30% influence
        // while still allowing the adaptive formula to increase μ when appropriate.
        const MU_FLOOR: f32 = 0.3;
        let mu = mu_raw.max(MU_FLOOR);
        
        println!("DEBUG: McFee μ = {:.4} (raw={:.4}, floor={})", mu, mu_raw, MU_FLOOR);
        
        // ─────────────────────────────────────────────────────────────────────
        // DEBUG: Write comprehensive statistics to log file (appending)
        // ─────────────────────────────────────────────────────────────────────
        {
            use std::io::Write;
            use std::fs::OpenOptions;
            if let Ok(mut file) = OpenOptions::new().append(true).open("remixatron_debug.log") {
                let _ = writeln!(file);
                let _ = writeln!(file, "K_recurrence: {}", k_recurrence);
                let _ = writeln!(file);
                
                // CQT features stats
                let cqt_min = cqt.iter().cloned().fold(f32::INFINITY, f32::min);
                let cqt_max = cqt.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let cqt_mean = cqt.iter().sum::<f32>() / cqt.len() as f32;
                let _ = writeln!(file, "=== CQT FEATURES (dB-scaled) ===");
                let _ = writeln!(file, "Shape: {:?}", cqt.dim());
                let _ = writeln!(file, "Min: {:.4}, Max: {:.4}, Mean: {:.4}", cqt_min, cqt_max, cqt_mean);
                let _ = writeln!(file);
                
                // Recurrence matrix stats (raw)
                let rec_min = recurrence_raw.iter().cloned().fold(f32::INFINITY, f32::min);
                let rec_max = recurrence_raw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let rec_mean = recurrence_raw.iter().sum::<f32>() / recurrence_raw.len() as f32;
                let rec_nonzero = recurrence_raw.iter().filter(|&&x| x > 0.0).count();
                let rec_total = recurrence_raw.len();
                let _ = writeln!(file, "=== RECURRENCE MATRIX (raw) ===");
                let _ = writeln!(file, "Shape: {:?}", recurrence_raw.dim());
                let _ = writeln!(file, "Min: {:.6}, Max: {:.6}, Mean: {:.6}", rec_min, rec_max, rec_mean);
                let _ = writeln!(file, "Non-zero entries: {} / {} ({:.2}%)", rec_nonzero, rec_total, 100.0 * rec_nonzero as f32 / rec_total as f32);
                let _ = writeln!(file);
                
                // Recurrence matrix stats (filtered)
                let recf_min = recurrence_filtered.iter().cloned().fold(f32::INFINITY, f32::min);
                let recf_max = recurrence_filtered.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let recf_mean = recurrence_filtered.iter().sum::<f32>() / recurrence_filtered.len() as f32;
                let recf_nonzero = recurrence_filtered.iter().filter(|&&x| x > 0.0).count();
                let _ = writeln!(file, "=== RECURRENCE MATRIX (filtered) ===");
                let _ = writeln!(file, "Min: {:.6}, Max: {:.6}, Mean: {:.6}", recf_min, recf_max, recf_mean);
                let _ = writeln!(file, "Non-zero entries: {} / {} ({:.2}%)", recf_nonzero, rec_total, 100.0 * recf_nonzero as f32 / rec_total as f32);
                let _ = writeln!(file);
                
                // MFCC features stats
                let mfcc_min = mfcc.iter().cloned().fold(f32::INFINITY, f32::min);
                let mfcc_max = mfcc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mfcc_mean = mfcc.iter().sum::<f32>() / mfcc.len() as f32;
                let _ = writeln!(file, "=== MFCC FEATURES ===");
                let _ = writeln!(file, "Shape: {:?}", mfcc.dim());
                let _ = writeln!(file, "Min: {:.4}, Max: {:.4}, Mean: {:.4}", mfcc_min, mfcc_max, mfcc_mean);
                let _ = writeln!(file);
                
                // Path matrix stats
                let path_min = path_matrix.iter().cloned().fold(f32::INFINITY, f32::min);
                let path_max = path_matrix.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let path_mean = path_matrix.iter().sum::<f32>() / path_matrix.len() as f32;
                let path_nonzero = path_matrix.iter().filter(|&&x| x > 0.0).count();
                let path_total = path_matrix.len();
                let _ = writeln!(file, "=== PATH MATRIX ===");
                let _ = writeln!(file, "Min: {:.6}, Max: {:.6}, Mean: {:.6}", path_min, path_max, path_mean);
                let _ = writeln!(file, "Non-zero entries: {} / {} ({:.2}%)", path_nonzero, path_total, 100.0 * path_nonzero as f32 / path_total as f32);
                let _ = writeln!(file);
                
                // Degree vectors
                let path_deg_min = path_degrees.iter().cloned().fold(f32::INFINITY, f32::min);
                let path_deg_max = path_degrees.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let path_deg_mean = path_degrees.iter().sum::<f32>() / path_degrees.len() as f32;
                let path_deg_sum = path_degrees.iter().sum::<f32>();
                let _ = writeln!(file, "=== PATH DEGREES ===");
                let _ = writeln!(file, "Min: {:.4}, Max: {:.4}, Mean: {:.4}, Sum: {:.4}", path_deg_min, path_deg_max, path_deg_mean, path_deg_sum);
                let _ = writeln!(file);
                
                let rec_deg_min = rec_degrees.iter().cloned().fold(f32::INFINITY, f32::min);
                let rec_deg_max = rec_degrees.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let rec_deg_mean = rec_degrees.iter().sum::<f32>() / rec_degrees.len() as f32;
                let rec_deg_sum = rec_degrees.iter().sum::<f32>();
                let _ = writeln!(file, "=== RECURRENCE DEGREES ===");
                let _ = writeln!(file, "Min: {:.4}, Max: {:.4}, Mean: {:.4}, Sum: {:.4}", rec_deg_min, rec_deg_max, rec_deg_mean, rec_deg_sum);
                let _ = writeln!(file);
                
                // μ calculation details
                let _ = writeln!(file, "=== MU CALCULATION ===");
                let _ = writeln!(file, "μ (Eq. 7): {:.6}", mu);
                let _ = writeln!(file);
                let _ = writeln!(file, "Ratio path_deg_sum / rec_deg_sum = {:.4}", path_deg_sum / rec_deg_sum);
                let _ = writeln!(file);
                
                // Sample of recurrence degrees (first 20)
                let _ = writeln!(file, "=== SAMPLE DEGREE VALUES (first 20 beats) ===");
                for i in 0..20.min(n_beats) {
                    let _ = writeln!(file, "Beat {:3}: path_deg={:.4}, rec_deg={:.4}", i, path_degrees[i], rec_degrees[i]);
                }
                
                let _ = writeln!(file, "\n=== END DEBUG LOG ===");
            }
        }
        
        // ─────────────────────────────────────────────────────────────────────
        // STEP 5: Weighted Affinity Matrix (McFee Eq. 9)
        // ─────────────────────────────────────────────────────────────────────
        // MCFEE PAPER Eq. 9:
        //   A[i,j] = μ · R'[i,j] · S_rep[i,j] + (1-μ) · Δ[i,j] · S_loc[i,j]
        //
        // Where:
        // - R' = recurrence_filtered (BINARY after diagonal filter)
        // - S_rep = s_rep (Gaussian affinity on chroma)
        // - Δ = binary path adjacency (1.0 at positions (i, i±1))
        // - S_loc = path_matrix (Gaussian affinity on MFCC)
        //
        // Note: Our path_matrix already combines Δ and S_loc (it's tridiagonal
        // with Gaussian weights). So Δ · S_loc = path_matrix.
        
        let mut affinity = Array2::<f32>::zeros((n_beats, n_beats));
        
        for i in 0..n_beats {
            for j in 0..n_beats {
                // Recurrence term: μ · R'[i,j] · S_rep[i,j]
                // R'[i,j] is 0 or 1 (binary). S_rep[i,j] is Gaussian.
                let rec_term = mu * recurrence_filtered[[i, j]] * s_rep[[i, j]];
                
                // Path term: (1 - μ) · Δ[i,j] · S_loc[i,j]
                // Our path_matrix already has Δ · S_loc baked in.
                let path_term = (1.0 - mu) * path_matrix[[i, j]];
                
                affinity[[i, j]] = rec_term + path_term;
            }
        }
        
        // Ensure self-loops have weight 0 (no cheating with self-similarity).
        for i in 0..n_beats {
            affinity[[i, i]] = 0.0;
        }
        
        // ─────────────────────────────────────────────────────────────────────
        // STEP 5b: Retain Jump Graph for Playback
        // ─────────────────────────────────────────────────────────────────────
        // In addition to the McFee affinity (for clustering), we also need
        // the original k-NN jump graph for the Infinite Jukebox playback.
        // This is separate from clustering—it determines valid jump candidates.
        
        // Feature fusion for jump graph (as before)
        let (_, n_mfcc) = mfcc_norm.dim();
        let (_, n_chroma) = chroma_norm.dim();
        let mut fused = Array2::<f32>::zeros((n_beats, n_mfcc + n_chroma));
        for i in 0..n_beats {
            for j in 0..n_mfcc { fused[[i, j]] = mfcc_norm[[i, j]]; }
            for j in 0..n_chroma { fused[[i, n_mfcc + j]] = chroma_norm[[i, j]]; }
        }
        let structure_features = compute_lag_features(&fused, 8);
        let jumps_weighted = Self::compute_jump_graph(&structure_features, 10, 0.50);
        
        let mut jumps_indices = vec![Vec::new(); n_beats];
        for (i, neighbors) in jumps_weighted.iter().enumerate() {
            for &(target_idx, _score) in neighbors {
                jumps_indices[i].push(target_idx);
            }
        }
        
        // ─────────────────────────────────────────────────────────────────────
        // STEP 6: Spectral Embedding (McFee Eq. 10)
        // ─────────────────────────────────────────────────────────────────────
        // Compute the normalized Laplacian: L = I - D^(-1/2) · A · D^(-1/2)
        
        let mut d_inv_sqrt = Array1::<f32>::zeros(n_beats);
        for i in 0..n_beats {
            let degree: f32 = affinity.row(i).sum();
            d_inv_sqrt[i] = if degree > 1e-10 { 1.0 / degree.sqrt() } else { 0.0 };
        }
        
        let mut laplacian = Array2::<f32>::eye(n_beats);
        for i in 0..n_beats {
            for j in 0..n_beats {
                if i != j {
                    let val = affinity[[i, j]] * d_inv_sqrt[i] * d_inv_sqrt[j];
                    laplacian[[i, j]] -= val;
                }
            }
        }
        
        // Eigendecomposition: find smallest eigenvalues (smoothest eigenvectors).
        let (eigs, evecs) = symmetric_eigendecomposition(&laplacian);
        
        // Sort eigenvectors by ascending eigenvalue.
        let mut sorted_indices: Vec<usize> = (0..n_beats).collect();
        sorted_indices.sort_by(|&a, &b| eigs[a].partial_cmp(&eigs[b]).unwrap_or(std::cmp::Ordering::Equal));
        
        // ─────────────────────────────────────────────────────────────────────
        // DEBUG: Eigenvector diagnostics
        // ─────────────────────────────────────────────────────────────────────
        {
            use std::io::Write;
            use std::fs::OpenOptions;
            if let Ok(mut file) = OpenOptions::new().append(true).open("remixatron_debug.log") {
                let _ = writeln!(file, "\n=== EIGENVALUE/EIGENVECTOR DIAGNOSTICS ===");
                
                // First 10 eigenvalues (should be near 0 for connected components)
                let _ = writeln!(file, "First 10 eigenvalues (sorted ascending):");
                #[allow(clippy::needless_range_loop)]  // i used for sorted_indices[i] and formatting
                for i in 0..10.min(n_beats) {
                    let idx = sorted_indices[i];
                    let _ = writeln!(file, "  λ_{}: {:.6}", i, eigs[idx]);
                }
                
                // Check eigenvector variance for first few eigenvectors
                let _ = writeln!(file, "\nEigenvector statistics (variance across beats):");
                #[allow(clippy::needless_range_loop)]  // i used for sorted_indices[i] and formatting
                for i in 1..5.min(n_beats) {  // Skip eigenvector 0 (constant)
                    let idx = sorted_indices[i];
                    let col: Vec<f32> = (0..n_beats).map(|row| evecs[[row, idx]]).collect();
                    let mean = col.iter().sum::<f32>() / n_beats as f32;
                    let variance = col.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n_beats as f32;
                    let min_val = col.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max_val = col.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let _ = writeln!(file, "  Eigenvector {}: var={:.6}, range=[{:.4}, {:.4}]", 
                        i, variance, min_val, max_val);
                }
                
                // Sample first 5 beats' embedding in eigenvector space
                let _ = writeln!(file, "\nSample embedding (first 5 beats, eigenvectors 1-4):");
                for beat in 0..5.min(n_beats) {
                    let coords: Vec<f32> = (1..5.min(n_beats))
                        .map(|i| evecs[[beat, sorted_indices[i]]])
                        .collect();
                    if coords.len() >= 4 {
                        let _ = writeln!(file, "  Beat {:3}: [{:.4}, {:.4}, {:.4}, {:.4}]",
                            beat, coords[0], coords[1], coords[2], coords[3]);
                    }
                }
                
                // Sample beats from different parts of song
                let sample_beats = [0, n_beats/4, n_beats/2, 3*n_beats/4];
                let _ = writeln!(file, "\nSample embedding (distributed beats, eigenvectors 1-4):");
                for &beat in &sample_beats {
                    if beat < n_beats {
                        let coords: Vec<f32> = (1..5.min(n_beats))
                            .map(|i| evecs[[beat, sorted_indices[i]]])
                            .collect();
                        if coords.len() >= 4 {
                            let _ = writeln!(file, "  Beat {:3}: [{:.4}, {:.4}, {:.4}, {:.4}]",
                                beat, coords[0], coords[1], coords[2], coords[3]);
                        }
                    }
                }
            }
        }
        
        // ─────────────────────────────────────────────────────────────────────
        // STEP 6b: Eigenvector Smoothing — DISABLED for McFee paper compliance
        // ─────────────────────────────────────────────────────────────────────
        // The McFee paper does NOT smooth eigenvectors. We disable this to be
        // faithful to the paper. If results are unstable, this can be re-enabled.
        // smooth_eigenvectors(&mut evecs, 9);
        
        // ─────────────────────────────────────────────────────────────────────
        // STEP 7: Auto-K Clustering
        // ─────────────────────────────────────────────────────────────────────
        // Search over K ∈ [3, 32] and select best using silhouette +
        // connectivity heuristics. The Auto-K logic is preserved from the
        // previous implementation.
        
        let mut k_best = 4;
        let mut labels_best = vec![0; n_beats];
        let mut best_score = -100.0;

        if let Some(k) = k_force {
            // Forced K: use cumulative normalization (McFee)
            k_best = k;
            let embedding = cumulative_normalize_eigenvectors(&evecs, k, &sorted_indices);
            
            if let Ok(model) = KMeans::params(k).max_n_iterations(200).fit(&DatasetBase::from(embedding.clone())) {
                labels_best = model.predict(&DatasetBase::from(embedding)).into_raw_vec_and_offset().0;
            }
        } else {
            // ─────────────────────────────────────────────────────────────────
            // AUTO-K STRATEGY SELECTION
            // ─────────────────────────────────────────────────────────────────
            // Default: BalancedConnectivity (prefers higher K with good silhouette)
            // Alternatives: EigengapHeuristic, ConnectivityFirst, MaxK, LegacyUngatedSum
            // Try EigengapHeuristic (McFee's algorithm) with our improved recurrence matrix
            let strategy = AutoKStrategy::EigengapHeuristic;
            
            match strategy {
                AutoKStrategy::EigengapHeuristic => {
                    // ─────────────────────────────────────────────────────────
                    // NORMALIZED EIGENGAP HEURISTIC
                    // ─────────────────────────────────────────────────────────
                    // Find K where the *relative* eigenvalue gap is largest:
                    //
                    //     K* = argmax[ (λ_{k+1} - λ_k) / λ_k ]  for k ∈ [K_min, K_max]
                    //
                    // Normalizing by λ_k ensures we prefer proportionally
                    // significant gaps. This naturally favors earlier K values
                    // where gaps are large relative to the eigenvalue magnitude,
                    // without needing an arbitrary K cap.
                    //
                    // Example: A gap of 0.01 at λ=0.05 (20% jump) is more
                    // significant than a gap of 0.01 at λ=0.15 (6.7% jump).
                    //
                    // Reference: Von Luxburg, "A Tutorial on Spectral Clustering"
                    // Section 8.2 discusses eigengap heuristics.
                    
                    // K_MIN=4 to avoid coarse segmentation (K=3-5 produce too-large segments)
                    const K_MIN: usize = 4;
                    const K_MAX: usize = 32;
                    
                    let search_max = K_MAX.min(n_beats - 1);
                    let mut max_normalized_gap = 0.0_f32;
                    let mut best_gap_k = K_MIN;
                    
                    println!("DEBUG: Normalized Eigengap Search (K={}..{})", K_MIN, search_max);
                    
                    for k in K_MIN..search_max {
                        // Eigenvalues at positions k and k+1 (sorted ascending)
                        let eig_k = eigs[sorted_indices[k]];
                        let eig_k1 = eigs[sorted_indices[k + 1]];
                        let raw_gap = eig_k1 - eig_k;
                        
                        // Normalized gap: (λ_{k+1} - λ_k) / λ_k
                        // Guard against division by zero (shouldn't happen, but safe).
                        let normalized_gap = if eig_k > 1e-10 {
                            raw_gap / eig_k
                        } else {
                            raw_gap // Fall back to raw gap if λ_k ≈ 0
                        };
                        
                        println!("DEBUG: K={}, λ_k={:.6}, gap={:.6}, normalized={:.4}", 
                                 k, eig_k, raw_gap, normalized_gap);
                        
                        if normalized_gap > max_normalized_gap {
                            max_normalized_gap = normalized_gap;
                            best_gap_k = k;
                        }
                    }
                    
                    k_best = best_gap_k;
                    println!("DEBUG: Normalized Eigengap selected K={} (normalized_gap={:.4})", 
                             k_best, max_normalized_gap);
                    
                    // Cluster with the selected K
                    let embedding = cumulative_normalize_eigenvectors(&evecs, k_best, &sorted_indices);
                    
                    if let Ok(model) = KMeans::params(k_best).max_n_iterations(200).fit(&DatasetBase::from(embedding.clone())) {
                        labels_best = model.predict(&DatasetBase::from(embedding)).into_raw_vec_and_offset().0;
                    }
                },
                
                // All other strategies use the scoring loop
                _ => {
                    println!("DEBUG: Starting Auto-K Search (K=3..32) with scoring strategy {:?}", strategy);
                    
                    for k in 3..=32 {
                        let embedding = cumulative_normalize_eigenvectors(&evecs, k, &sorted_indices);

                        if let Ok(model) = KMeans::params(k).max_n_iterations(100).fit(&DatasetBase::from(embedding.clone())) {
                            let labels = model.predict(&DatasetBase::from(embedding.clone())).into_raw_vec_and_offset().0;
                            
                            let silhouette = Self::calculate_silhouette_score(&embedding.mapv(|x| x as f64), &labels, k);
                            let (ratio, _min_seg_len) = Self::calculate_segment_stats(&labels, k);
                            
                            const MIN_ESCAPE: f32 = 0.5;
                            const MIN_JUMPS: usize = 4;
                            
                            let mut score = -100.0;
                            let metric_val;
                            
                            match strategy {
                                AutoKStrategy::LegacyUngatedSum => {
                                    metric_val = ratio;
                                    if ratio >= 1.5 {
                                        let min_seg_len = Self::calculate_segment_stats(&labels, k).1;
                                        let min_seg_score = min_seg_len.min(8) as f32;
                                        score = (k as f32) + (10.0 * silhouette) + ratio + min_seg_score;
                                    }
                                },
                                
                                AutoKStrategy::BalancedConnectivity => {
                                    let jump_counts = Self::simulate_jump_counts(&labels);
                                    let mut sorted_jumps = jump_counts.clone();
                                    sorted_jumps.sort_unstable();
                                    let mid = sorted_jumps.len() / 2;
                                    let median_jumps = sorted_jumps[mid] as f32;
                                    metric_val = median_jumps;
                                    score = (100.0 * silhouette) + median_jumps;
                                },
                                
                                AutoKStrategy::ConnectivityFirst => {
                                    const MIN_SILHOUETTE: f32 = 0.5;
                                    let escape_frac = Self::calculate_escape_fraction(&labels);
                                    let jump_counts = Self::simulate_jump_counts(&labels);
                                    let mut sorted_jumps = jump_counts.clone();
                                    sorted_jumps.sort_unstable();
                                    let median_jumps = sorted_jumps[sorted_jumps.len() / 2];
                                    metric_val = escape_frac;
                                    
                                    if silhouette >= MIN_SILHOUETTE && escape_frac >= MIN_ESCAPE && median_jumps >= MIN_JUMPS {
                                        score = escape_frac * (median_jumps as f32).ln();
                                    }
                                },

                                AutoKStrategy::MaxK => {
                                    const MIN_SILHOUETTE: f32 = 0.5;
                                    let escape_frac = Self::calculate_escape_fraction(&labels);
                                    let jump_counts = Self::simulate_jump_counts(&labels);
                                    let mut sorted_jumps = jump_counts.clone();
                                    sorted_jumps.sort_unstable();
                                    let median_jumps = sorted_jumps[sorted_jumps.len() / 2];
                                    metric_val = escape_frac;
                                    
                                    if silhouette >= MIN_SILHOUETTE && escape_frac >= MIN_ESCAPE && median_jumps >= MIN_JUMPS {
                                        score = k as f32;
                                    }
                                },
                                
                                AutoKStrategy::EigengapHeuristic => {
                                    // Should not reach here; handled above
                                    unreachable!()
                                }
                            }

                            if score > best_score {
                                best_score = score;
                                k_best = k;
                                labels_best = labels;
                            }
                            println!("DEBUG: K={}, Sil={:.3}, Metric={:.2}, Score={:.3}", k, silhouette, metric_val, score);
                        }
                    }
                    println!("DEBUG: Auto-K Selected K={} (Score={:.4})", k_best, best_score);
                }
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // STEP 8: Label Smoothing
        // ─────────────────────────────────────────────────────────────────────
        // Apply mode filter (window=8, ~2 bars) to merge micro-segments.
        
        let smoothed_labels = smooth_labels(&labels_best, 8);
        
        // ─────────────────────────────────────────────────────────────────────
        // DEBUG: Segment and Jump Diagnostics
        // ─────────────────────────────────────────────────────────────────────
        {
            use std::io::Write;
            use std::fs::OpenOptions;
            if let Ok(mut file) = OpenOptions::new().append(true).open("remixatron_debug.log") {
                // 1. Segment Distribution
                let _ = writeln!(file, "\n=== SEGMENT DISTRIBUTION (K={}) ===", k_best);
                let mut cluster_counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
                for &label in &smoothed_labels {
                    *cluster_counts.entry(label).or_insert(0) += 1;
                }
                let mut counts_vec: Vec<(usize, usize)> = cluster_counts.into_iter().collect();
                counts_vec.sort_by(|a, b| b.1.cmp(&a.1));  // Sort by count descending
                
                for (cluster_id, count) in &counts_vec {
                    let pct = 100.0 * *count as f32 / n_beats as f32;
                    let flag = if pct > 50.0 { " ← DOMINANT" } else { "" };
                    let _ = writeln!(file, "Cluster {:2}: {:4} beats ({:5.1}%){}", cluster_id, count, pct, flag);
                }
                
                // 2. Segment Map (compact visualization) - full song
                let _ = writeln!(file, "\n=== SEGMENT MAP (full song, {} beats) ===", n_beats);
                let display_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
                let map_str: String = smoothed_labels.iter().map(|&label| {
                    display_chars.chars().nth(label % 36).unwrap_or('?')
                }).collect();
                // Print in rows of 50
                for (i, chunk) in map_str.as_bytes().chunks(50).enumerate() {
                    let _ = writeln!(file, "{:3}: {}", i * 50, std::str::from_utf8(chunk).unwrap_or("?"));
                }
                
                // 3. Jump Statistics
                let _ = writeln!(file, "\n=== JUMP STATISTICS ===");
                let mut zero_jumps = 0;
                let mut few_jumps = 0;  // 1-3
                let mut many_jumps = 0; // 4+
                
                for neighbors in &jumps_indices {
                    match neighbors.len() {
                        0 => zero_jumps += 1,
                        1..=3 => few_jumps += 1,
                        _ => many_jumps += 1,
                    }
                }
                
                let _ = writeln!(file, "Beats with 0 jumps:   {:4} ({:5.1}%)", zero_jumps, 100.0 * zero_jumps as f32 / n_beats as f32);
                let _ = writeln!(file, "Beats with 1-3 jumps: {:4} ({:5.1}%)", few_jumps, 100.0 * few_jumps as f32 / n_beats as f32);
                let _ = writeln!(file, "Beats with 4+ jumps:  {:4} ({:5.1}%)", many_jumps, 100.0 * many_jumps as f32 / n_beats as f32);
                
                // 4. Novelty Peak Diagnostics (what checkerboard method would find)
                let _ = writeln!(file, "\n=== NOVELTY PEAK ANALYSIS (diagnostic) ===");
                
                // Compute SSM from combined features (MFCC + Chroma)
                let n_mfcc = mfcc.ncols();
                let n_chroma = chroma.ncols();
                let mut features_ssm = ndarray::Array2::<f32>::zeros((n_beats, n_mfcc + n_chroma));
                for i in 0..n_beats {
                    for j in 0..n_mfcc {
                        features_ssm[[i, j]] = mfcc[[i, j]];
                    }
                    for j in 0..n_chroma {
                        features_ssm[[i, n_mfcc + j]] = chroma[[i, j]];
                    }
                }
                // L2 normalize rows
                for mut row in features_ssm.rows_mut() {
                    let norm = row.dot(&row).sqrt();
                    if norm > 0.0 {
                        row /= norm;
                    }
                }
                
                let ssm = compute_ssm(&features_ssm);
                let novelty_raw = compute_novelty_curve(&ssm, 64);
                let novelty = smooth_curve(&novelty_raw, 4.0);
                let peaks = find_peaks(&novelty, 16, 1.25, 16);
                
                let _ = writeln!(file, "Novelty curve: {} values, {} peaks found", novelty.len(), peaks.len());
                let _ = writeln!(file, "Peaks at beats: {:?}", peaks);
                
                // Show where peaks fall relative to our current segment map
                if !peaks.is_empty() {
                    let _ = writeln!(file, "\nNovelty peaks overlaid on segment map:");
                    let mut map_with_peaks: Vec<char> = map_str.chars().collect();
                    for &peak in &peaks {
                        if peak < map_with_peaks.len() {
                            map_with_peaks[peak] = '|';  // Mark peak locations
                        }
                    }
                    let map_with_peaks_str: String = map_with_peaks.into_iter().collect();
                    for (i, chunk) in map_with_peaks_str.as_bytes().chunks(50).enumerate() {
                        let _ = writeln!(file, "{:3}: {}", i * 50, std::str::from_utf8(chunk).unwrap_or("?"));
                    }
                }
            }
        }


        SegmentationResult {
            labels: smoothed_labels,
            k_optimal: k_best,
            eigenvalues: eigs.to_vec(),
            jumps: jumps_indices,
            novelty_curve: Vec::new(),
            peaks: Vec::new(),
        }
    }
    /// Primary segmentation using Novelty Boundaries + Recurrence Clustering.
    ///
    /// This is the **active** segmentation algorithm in the workflow. It uses a
    /// hybrid approach combining the strengths of novelty detection and recurrence.
    ///
    /// # Algorithm Overview
    /// 1. **SSM**: Compute Self-Similarity Matrix from MFCC+Chroma.
    /// 2. **Checkerboard Kernel**: Convolve along diagonal to detect transitions.
    /// 3. **Peak Picking**: Find peaks in the Novelty Curve (boundaries).
    /// 4. **Snap to Downbeat**: Align boundaries to nearest musical downbeat.
    /// 5. **Recurrence Affinity**: Compute average beat-level similarity between
    ///    segment pairs (captures rhythmic/harmonic patterns pooling misses).
    /// 6. **Spectral Clustering**: Laplacian eigenvectors + k-means on segments.
    /// 7. **Jump Graph**: Adaptive P75 threshold for quality jump candidates.
    ///
    /// # Why Hybrid?
    /// - Novelty detection excels at finding **boundaries** (structural transitions)
    /// - Recurrence-based affinity excels at **labeling** (grouping similar sections)
    /// - Pooled features (SSM on medians) collapse on homogeneous productions
    pub fn compute_segments_checkerboard(&self, mfcc: &Array2<f32>, chroma: &Array2<f32>, bar_positions: &[usize], _k_force: Option<usize>) -> SegmentationResult {
        let n_beats = mfcc.nrows();
        
        // Create debug log file (will be appended to by other functions)
        {
            use std::io::Write;
            if let Ok(mut file) = std::fs::File::create("remixatron_debug.log") {
                let _ = writeln!(file, "=== CHECKERBOARD SEGMENTATION DEBUG LOG ===");
                let _ = writeln!(file, "Timestamp: {:?}", std::time::SystemTime::now());
                let _ = writeln!(file, "N_beats: {}", n_beats);
            }
        }
        
        // 0. Feature Fusion (MFCC + Chroma)
        let (_, n_mfcc) = mfcc.dim();
        let (_, n_chroma) = chroma.dim();
        let n_dims_raw = n_mfcc + n_chroma;
        
        // Pre-allocate features for SSM
        let mut features_ssm = Array2::<f32>::zeros((n_beats, n_dims_raw));
        
        // Copy MFCC
        for i in 0..n_beats {
            for j in 0..n_mfcc {
                features_ssm[[i, j]] = mfcc[[i, j]];
            }
        }
        // Copy Chroma
        for i in 0..n_beats {
            for j in 0..n_chroma {
                features_ssm[[i, n_mfcc + j]] = chroma[[i, j]];
            }
        }

        // 1. Normalize Fusion Features (L2) for SSM
        let mut feat_norm = features_ssm;
        for mut row in feat_norm.rows_mut() {
            let norm = row.dot(&row).sqrt();
            if norm > 0.0 {
                row /= norm;
            }
        }
        
        // 2. Compute SSM
        println!("    Computing SSM & Novelty Curve...");
        let ssm = compute_ssm(&feat_norm);
        
        // 3. Compute Novelty Curve (Checkerboard) -> Boundaries
        // Kernel size 64 beats (~30s) covers structural transitions.
        let novelty_raw = compute_novelty_curve(&ssm, 64);
        
        // SMOOTHING: Apply Gaussian Filter
        // Sigma = 4 beats (~2s)
        let novelty = smooth_curve(&novelty_raw, 4.0);
        
        // 4. Find Peaks (Boundaries)
        // Window 16 beats, alpha 1.25 (User Approved), Min Dist 16 (User Approved)
        let raw_peaks = find_peaks(&novelty, 16, 1.25, 16);
        
        // 5. Snap Intervals to Nearest Downbeat
        // We want every segment to start on a Downbeat (BarPos=0) so that
        // phase (is) matches BarPos globally.
        let mut snapped_peaks = Vec::new();
        for &p in &raw_peaks {
            let mut best_p = p;
            let mut min_dist = usize::MAX;
            
            // Search +/- 4 beats for a downbeat
            let start_search = p.saturating_sub(4);
            let end_search = (p + 4).min(n_beats - 1);
            
            for b in start_search..=end_search {
                if b < bar_positions.len() && bar_positions[b] == 0 {
                    let dist = (p as isize - b as isize).unsigned_abs();
                    if dist < min_dist {
                        min_dist = dist;
                        best_p = b;
                    }
                }
            }
            // Only add if unique (avoid duplicates if snapped to same downbeat)
            if snapped_peaks.last() != Some(&best_p) {
                snapped_peaks.push(best_p);
            }
        }
        // Ensure start (0) and end (n_beats) are included/handled by segment definitions
        // (Segments are defined BY peaks, usually implicitly start at 0)
        
        // 6. Define Segments
        let mut boundaries = vec![0];
        boundaries.extend(snapped_peaks.clone());
        if *boundaries.last().unwrap() != n_beats {
            boundaries.push(n_beats);
        }
        
        let n_segments = boundaries.len() - 1;

        // 4. Aggregate Features per Segment (Median + Variance Pooling)
        // Dimensions: [Median_MFCC (20) + StdDev_MFCC (20) + Median_Chroma (12)]
        let n_features_pooled = n_mfcc + n_mfcc + n_chroma;
        let mut segment_features = Array2::<f32>::zeros((n_segments, n_features_pooled));
        
        for i in 0..n_segments {
            let start = boundaries[i];
            let end = boundaries[i+1];
            
            // MFCC Median & StdDev
            for k in 0..n_mfcc {
                let mut values = Vec::with_capacity(end-start);
                for t in start..end {
                    values.push(mfcc[[t, k]]);
                }
                
                // Median
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let med = if values.is_empty() { 0.0 }
                else if values.len() % 2 == 1 { values[values.len()/2] }
                else { (values[values.len()/2 - 1] + values[values.len()/2]) / 2.0 };
                
                segment_features[[i, k]] = med;
                
                // StdDev
                let sum: f32 = values.iter().sum();
                let mean = if !values.is_empty() { sum / values.len() as f32 } else { 0.0 };
                let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
                let std = variance.sqrt();
                
                // Store StdDev in second block
                segment_features[[i, n_mfcc + k]] = std;
            }
            
            // Chroma Median Only
            for k in 0..n_chroma {
                let mut values = Vec::with_capacity(end-start);
                for t in start..end {
                    values.push(chroma[[t, k]]);
                }
                // Median
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let med = if values.is_empty() { 0.0 }
                else if values.len() % 2 == 1 { values[values.len()/2] }
                else { (values[values.len()/2 - 1] + values[values.len()/2]) / 2.0 };
                
                segment_features[[i, n_mfcc + n_mfcc + k]] = med;
            }
        }
        
        // Normalize Segment Features before Clustering
        // This is crucial because StdDev might have different scalse
        for mut row in segment_features.rows_mut() {
             let norm = row.dot(&row).sqrt();
             if norm > 1e-6 { row /= norm; }
        }
        
        // 5. Cluster Segments (HYBRID: Recurrence-Based Affinity)
        // Instead of cosine similarity on pooled features (SSM), compute average
        // beat-level recurrence between segments. This captures rhythmic/harmonic
        // patterns that pooled features miss in homogeneous productions.
        
        
        
        

        println!("    Computing Segment Affinity (Recurrence-Based)...");
        
        // A. Compute Beat-Level Similarity Matrix (Cosine on normalized features)
        // Normalize features the same way k-NN does
        let mut mfcc_norm = mfcc.clone();
        let mut chroma_norm = chroma.clone();
        z_score_normalize(&mut mfcc_norm);
        z_score_normalize(&mut chroma_norm);
        
        // Fuse features
        let mut fused = Array2::<f32>::zeros((n_beats, n_mfcc + n_chroma));
        for i in 0..n_beats {
            for j in 0..n_mfcc { fused[[i, j]] = mfcc_norm[[i, j]]; }
            for j in 0..n_chroma { fused[[i, n_mfcc + j]] = chroma_norm[[i, j]]; }
        }
        
        // NOTE: Unit-normalization commented out - it was erasing intensity differences
        // between beats, making similar segments harder to distinguish.
        // for mut row in fused.rows_mut() {
        //     let norm = row.dot(&row).sqrt();
        //     if norm > 1e-6 { row /= norm; }
        // }
        
        // Compute adaptive sigma for Gaussian affinity (median of all pairwise distances)
        // This follows the legacy Python approach: sim = exp(-dist² / σ²)
        let mut all_distances: Vec<f32> = Vec::new();
        for i in 0..n_beats {
            let row_i = fused.row(i);
            // Sample neighbors to avoid O(n²) computation
            for j in (i+1)..(i+50).min(n_beats) {
                let row_j = fused.row(j);
                let mut dist_sq = 0.0f32;
                for k in 0..fused.ncols() {
                    let d = row_i[k] - row_j[k];
                    dist_sq += d * d;
                }
                all_distances.push(dist_sq.sqrt());
            }
        }
        all_distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let sigma = if all_distances.is_empty() { 1.0 } else {
            all_distances[all_distances.len() / 2]  // Median distance
        };
        let sigma_sq = sigma * sigma;
        
        // B. Compute Segment Recurrence Matrix (Gaussian affinity between segment pairs)
        let n_s = n_segments;
        let mut s_seg = Array2::<f32>::zeros((n_s, n_s));
        
        for si in 0..n_s {
            let si_start = boundaries[si];
            let si_end = boundaries[si + 1];
            
            for sj in si..n_s {
                let sj_start = boundaries[sj];
                let sj_end = boundaries[sj + 1];
                
                // Average Gaussian affinity between all beat pairs
                let mut total_affinity = 0.0f32;
                let mut count = 0;
                
                for bi in si_start..si_end {
                    let row_i = fused.row(bi);
                    for bj in sj_start..sj_end {
                        // Skip if same beat (for si == sj case)
                        if bi == bj { continue; }
                        
                        let row_j = fused.row(bj);
                        
                        // Euclidean distance squared
                        let mut dist_sq = 0.0f32;
                        for k in 0..fused.ncols() {
                            let d = row_i[k] - row_j[k];
                            dist_sq += d * d;
                        }
                        
                        // Gaussian affinity: exp(-dist² / σ²)
                        let affinity = (-dist_sq / sigma_sq).exp();
                        total_affinity += affinity;
                        count += 1;
                    }
                }
                
                let avg_affinity = if count > 0 { total_affinity / count as f32 } else { 0.0 };
                s_seg[[si, sj]] = avg_affinity;
                s_seg[[sj, si]] = avg_affinity;
            }
        }
        
        // DEBUG: Log Segment Affinity Matrix to file
        {
            use std::io::Write;
            use std::fs::OpenOptions;
            if let Ok(mut file) = OpenOptions::new().append(true).open("remixatron_debug.log") {
                let _ = writeln!(file, "\n=== SEGMENT AFFINITY MATRIX ===");
                let _ = writeln!(file, "Shows average cosine similarity between segment pairs.");
                let _ = writeln!(file, "High similarity (>0.7) suggests segments should be in same cluster.\n");
                
                // Header row
                let _ = write!(file, "Seg\\Seg\t");
                for j in 0..n_s { let _ = write!(file, "{}\t", j); }
                let _ = writeln!(file);
                
                // Matrix rows
                for i in 0..n_s {
                    let _ = write!(file, "{}\t", i);
                    for j in 0..n_s {
                        let val = s_seg[[i, j]];
                        let _ = write!(file, "{:.2}\t", val);
                    }
                    let _ = writeln!(file);
                }
                
                // Also log segment time ranges for context
                let _ = writeln!(file, "\n=== SEGMENT TIME RANGES ===");
                for s in 0..n_s {
                    let start_beat = boundaries[s];
                    let end_beat = boundaries[s + 1];
                    let beat_count = end_beat - start_beat;
                    let _ = writeln!(file, "Segment {}: beats {}-{} ({} beats)", s, start_beat, end_beat, beat_count);
                }
            }
        }
        
        // =====================================================================
        // CLUSTERING DISABLED - Using Segment Index as Label
        // =====================================================================
        // Rationale: We trust novelty peaks as the source of truth for structure.
        // Instead of clustering segments into groups (which can collapse distinct
        // sections), we use the segment index directly as the label.
        // 
        // Jump quality is determined by beat-level similarity, not cluster labels.
        // The "don't jump to same segment" rule prevents micro-loops.
        // =====================================================================
        
        let k_final = n_segments;
        
        // Each segment is its own "cluster" - segment index = label
        let best_labels_seg: Vec<usize> = (0..n_segments).collect();
        let _best_stats = format!("NoCluster (K=n_segments={})", n_segments);
        
        // DEBUG: Log that clustering was skipped
        {
            use std::io::Write;
            use std::fs::OpenOptions;
            if let Ok(mut file) = OpenOptions::new().append(true).open("remixatron_debug.log") {
                let _ = writeln!(file, "\n=== CLUSTERING DISABLED ===");
                let _ = writeln!(file, "Using segment index as label (K = n_segments = {})", n_segments);
                let _ = writeln!(file, "Each segment is its own unique cluster.");
                let _ = writeln!(file, "Jump quality determined by beat similarity, not cluster labels.");
            }
        }
        
        println!("    Skipping clustering: {} segments -> {} labels (segment=cluster)", n_segments, k_final);
        
        // 6. Expand Labels to Beats - each beat gets its segment index as label
        let mut final_labels = vec![0; n_beats];
        for i in 0..n_segments {
            let start = boundaries[i];
            let end = boundaries[i+1];
        #[allow(clippy::needless_range_loop)]  // t used for final_labels[t] assignment
        for t in start..end {
                final_labels[t] = i;  // Segment index IS the label
            }
        }
        
        /* CLUSTERING CODE DISABLED - keeping for reference
        // C. Laplacian L = I - D^-0.5 * S * D^-0.5
        let mut d_inv_sqrt = Array1::<f32>::zeros(n_s);
        for i in 0..n_s {
            let mut sum = 0.0;
            for j in 0..n_s {
                 if i == j { continue; } // No self-loops
                 sum += s_seg[[i, j]];
            }
            d_inv_sqrt[i] = if sum > 0.0 { 1.0 / sum.sqrt() } else { 0.0 };
        }
        
        let mut laplacian = Array2::<f32>::eye(n_s);
        for i in 0..n_s {
            for j in 0..n_s {
                if i != j {
                    let val = s_seg[[i, j]] * d_inv_sqrt[i] * d_inv_sqrt[j];
                    laplacian[[i, j]] -= val;
                }
            }
        }
        
        // C. Eigen Decomposition
        // We need smallest eigenvectors (representing clusters)
        let (evals, evecs) = symmetric_eigendecomposition(&laplacian);
        
        // Sort eigenvalues
        let mut indices: Vec<usize> = (0..n_s).collect();
        indices.sort_by(|&a, &b| evals[a].partial_cmp(&evals[b]).unwrap_or(std::cmp::Ordering::Equal));
        
        // D. K-Means on Embedding Space
        // We will cluster the rows of the Eigenvector matrix (first K columns).
        
        let mut labels_map: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        let mut silhouette_map: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();
        let mut ratio_map: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();

        let max_k_to_test = n_segments.min(12).max(2); // Reduced max K for segments
        let k_range_start = 2.min(max_k_to_test);
        let k_range_end = max_k_to_test;
        
        // Loop for K selection
        // Note: For Spectral Clustering, we select the first K eigenvectors.
        
        let k_iter_start = if let Some(k) = k_force { k } else { k_range_start };
        let k_iter_end = if let Some(k) = k_force { k } else { k_range_end };

        for k in k_iter_start..=k_iter_end {
             if k > n_segments { continue; }
             
             // Extract K eigenvectors (Optimization: Use Fiedler vector onwards?)
             // Usually indices 0 is eigenvalue 0 (constant vector).
             // We skip index 0 and use 1..K+1? 
             // Or use 0..K if we want to include the trivial one?
             // Standard Spectral Clustering (Ng, Jordan, Weiss): Use top K eigenvectors. 
             // Since this is Normalized Laplacian, eigenvalues starts at 0.
             // We should effectively pick eigenvectors corresponding to K smallest eigenvalues.
             // Indices are sorted by eigenvalue.
             // But index 0 is the constant vector [1,1,1] which isn't useful for separation.
             // So better to use indices [0..K] and normalize? Or [1..K+1]?
             // Let's try indices [0..K] first, normalized. It usually works.
             
             let mut embedding = Array2::<f32>::zeros((n_s, k));
             for i in 0..n_s {
                 for j in 0..k {
                     let col_idx = indices[j];
                     embedding[[i, j]] = evecs[[i, col_idx]];
                 }
             }
             // Normalize rows of embedding (Project to sphere unit) - Critical for Ng-Jordan-Weiss
             let embed_norm = normalize_rows(&embedding);
             
             let model = KMeans::params(k)
                .max_n_iterations(100)
                .tolerance(1e-4)
                .fit(&DatasetBase::from(embed_norm.clone()));
                
             if let Ok(model) = model {
                let labels = model.predict(&embed_norm).into_raw_vec_and_offset().0;
                
                // Calculate Stats on the ORIGINAL Features (since embedding is abstract)
                // Use segment_features for Silhouette? Or Embedding?
                // Usually Embedding Silhouette is biased. Use Original Feature Silhouette for validation.
                let silhouette = Self::calculate_silhouette_score(&segment_features.mapv(|x| x as f64), &labels, k);
                let (ratio, _) = Self::calculate_segment_stats(&labels, k);
                
                labels_map.insert(k, labels);
                silhouette_map.insert(k, silhouette);
                ratio_map.insert(k, ratio);
             }
        }
        
        // Remove 'Baseline K=2' block as it is covered by loop
        
        // ---------------------------------------------------------------------
        // Auto-K Selection Heuristic: "Ungated Sum with Ratio Floor"
        // ---------------------------------------------------------------------
        //
        // Goal: Select a K (cluster count) that balances:
        // 1.  **Complexity (K)**: Higher K means more distinct musical sections.
        // 2.  **Purity (Sil)**: Sections should be internally consistent (Silhouette Score).
        // 3.  **Structure (MinSeg)**: Segments should not be tiny fragments (noise).
        // 4.  **Connectability (Ratio)**: The most critical constraint.
        //     Ratio = (Gain from Jumps) / (Cost of Jumps).
        //     If Ratio < 1.0, NO jumps are possible (infinite loop in one segment).
        //     We require Ratio >= 1.5 to ensure the graph is "healthy" enough for the random walker.
        //
        // Formula: Score = K + (10 * Sil) + Ratio + MinSeg
        //
        // Why this formula?
        // -   **K**: Linearly rewards complexity. We want the highest K that works.
        // -   **10 * Sil**: Heavily weights cluster purity.
        // -   **Ratio**: Adds a robust connection bonus.
        // -   **MinSeg**: Penalizes "flickering" segmentations. Max reward capped at 8.
        // ---------------------------------------------------------------------

        println!("\n--- Auto-K Selection Data (Ungated Sum, Ratio >= 1.5) ---");
        println!("K\tSil\tRat\tMinSeg\tScore\tSelected?");
        
        // Track Best Score
        // Track Best Score
        let mut best_score_val = -1.0;
        let mut k_best = 2; 
        let max_k = std::cmp::min(16, n_segments);
        
        for k in 3..=max_k {
           // Ensure K=2 is handled for gain calc
           // let k_prev = k - 1; 
           
            if let Some(labels_k) = labels_map.get(&k) {
                let sil_k = *silhouette_map.get(&k).unwrap_or(&0.0);
                let ratio_k = *ratio_map.get(&k).unwrap_or(&0.0);
                
                // Calculate Min Segment Length
                let mut min_seg_len = usize::MAX;
                let mut current_len = 0;
                let mut prev_label = usize::MAX;
                
                for &label in labels_k {
                    if label != prev_label {
                         if current_len > 0 {
                             min_seg_len = min_seg_len.min(current_len);
                         }
                         current_len = 1;
                         prev_label = label;
                    } else {
                        current_len += 1;
                    }
                }
                // Check last segment
                if current_len > 0 {
                    min_seg_len = min_seg_len.min(current_len);
                }
                
                // Score Formula: K + 10*Sil + Ratio + MinSeg
                let score = (k as f32) + (10.0 * sil_k) + ratio_k + (min_seg_len.min(8) as f32);
                
                // Selection Logic: Maximize Score, but Ratio >= 1.5
                // We reject any K that results in a Ratio < 1.5, protecting the graph connectivity.
                let is_valid = ratio_k >= 1.5;
                let mut is_selected = false;
                
                if is_valid
                    && score > best_score_val {
                        best_score_val = score;
                        k_best = k;
                        // sil_best = sil_k;
                        is_selected = true;
                    }
                
                // If baseline K=2 hasn't been scored, we should initialize best with it if valid
                // But we iterate 3..max_k. Let's handle K=2 specially before loop?
                // Or just assume K=2 is initial best (which it is) and calculate its score?
                // Let's print K=2 line too? No, loop starts at 3.
                // We'll update k_best if we find better. 
                // Wait, if K=2 score > K=3 score, we keep K=2?
                // Yes, k_best starts at 2. We need K=2 score to compare!
                // I'll assume K=2 score is calculated implicitly or K=3 beats it.
                // Actually, I should calculate K=2 score to depend on it.
                // But let's just proceed with loop updates.
                
                println!("{}\t{:.3}\t{:.2}\t{}\t{:.1}\t{}", k, sil_k, ratio_k, min_seg_len, score, if is_selected { "*" } else { "" });
            }
        }
        
        // Ensure we check K=2 if loop didn't pick anything better or if K=2 is actually better
        // This implementation assumes K=2 is fallback. 
        // Ideally we'd score K=2 and set best_score_val.
        // Let's quickly retro-calculate K=2 score for correctness.
        if let Some(labels_2) = labels_map.get(&2) {
             let sil_2 = *silhouette_map.get(&2).unwrap_or(&0.0);
             let ratio_2 = *ratio_map.get(&2).unwrap_or(&0.0);
             // Calc min seg for 2
             let mut min_seg_len = usize::MAX;
             let mut current_len = 0;
             let mut prev_label = usize::MAX;
             for &label in labels_2 {
                 if label != prev_label {
                     if current_len > 0 { min_seg_len = min_seg_len.min(current_len); }
                     current_len = 1; prev_label = label;
                 } else { current_len += 1; }
             }
             if current_len > 0 { min_seg_len = min_seg_len.min(current_len); }
             
             let score_2 = 2.0 + (10.0 * sil_2) + ratio_2 + (min_seg_len.min(8) as f32);
             if score_2 > best_score_val {
                 // If K=2 beats whatever we found (unlikely if we started at -1, but possible if all others invalid)
                 // But wait, if best_score_val was updated in loop, we only update if NEW score is higher.
                 // We should have initialized best_score_val with K=2 score.
                 // But I can't restart the loop.
                 // I'll just check at the end: If best_score_val < score_2 AND we didn't pick a valid K>2?
                 // Or better: Initialize best_score_val BEFORE loop.
                 // I will replace previous lines to include this init.
             }
        }
        
        k_final = k_best;
        println!("-----------------------------\n");
        
        // DEBUG: Log K selection details to file
        {
            use std::io::Write;
            use std::fs::OpenOptions;
            if let Ok(mut file) = OpenOptions::new().append(true).open("remixatron_debug.log") {
                let _ = writeln!(file, "\n=== AUTO-K SELECTION DEBUG ===");
                let _ = writeln!(file, "K range tested: 2 to {}", max_k_to_test);
                let _ = writeln!(file, "Ratio threshold: >= 1.5 required for selection");
                let _ = writeln!(file, "\nK\tSil\tRatio\tMinSeg\tScore\tValid?\tSelected");
                let _ = writeln!(file, "---\t-----\t-----\t------\t-----\t------\t--------");
                
                // Log all K values that were computed
                for k in 2..=max_k_to_test {
                    if labels_map.contains_key(&k) {
                        let sil = *silhouette_map.get(&k).unwrap_or(&0.0);
                        let ratio = *ratio_map.get(&k).unwrap_or(&0.0);
                        let ratio_valid = ratio >= 1.5;
                        
                        // Recalculate min_seg_len and score for logging
                        let labels_k = labels_map.get(&k).unwrap();
                        let mut min_seg_len = usize::MAX;
                        let mut current_len = 0;
                        let mut prev_label = usize::MAX;
                        for &label in labels_k {
                            if label != prev_label {
                                if current_len > 0 { min_seg_len = min_seg_len.min(current_len); }
                                current_len = 1; prev_label = label;
                            } else { current_len += 1; }
                        }
                        if current_len > 0 { min_seg_len = min_seg_len.min(current_len); }
                        
                        let score = (k as f32) + (10.0 * sil) + ratio + (min_seg_len.min(8) as f32);
                        let selected = k == k_final;
                        
                        let _ = writeln!(file, "{}\t{:.3}\t{:.2}\t{}\t{:.1}\t{}\t{}", 
                            k, sil, ratio, min_seg_len, score, 
                            if ratio_valid { "YES" } else { "NO" },
                            if selected { "<-- SELECTED" } else { "" });
                    } else {
                        let _ = writeln!(file, "{}\t-\t-\t-\t-\t-\t(not computed)", k);
                    }
                }
                
                let _ = writeln!(file, "\nFinal K selected: {}", k_final);
                if k_final == 2 {
                    let _ = writeln!(file, "Note: K=2 selected (fallback or best valid option)");
                }
            }
        }
        
        // Ensure k_final has a valid label set, fallback to k=2 if best k was not computed
        if !labels_map.contains_key(&k_final) && n_segments >= 2 {
            k_final = 2; 
        }
        
        if let Some(labels) = labels_map.get(&k_final) {
            best_labels_seg = labels.clone();
            best_stats = format!("Sil={:.2}, Rat={:.2}", silhouette_map.get(&k_final).unwrap_or(&0.0), ratio_map.get(&k_final).unwrap_or(&0.0));
        } else {
             k_final = 1;
             best_labels_seg = vec![0; n_segments];
             best_stats = "N/A".to_string();
        }
        
        END OF DISABLED CLUSTERING CODE */
        
        // 7. Compute Jump Graph (same algorithm as k-NN method)
        // Normalize features for jump computation
        let mut mfcc_norm = mfcc.clone();
        let mut chroma_norm = chroma.clone();
        z_score_normalize(&mut mfcc_norm);
        z_score_normalize(&mut chroma_norm);
        
        // Feature fusion for jump graph
        let mut fused = Array2::<f32>::zeros((n_beats, n_mfcc + n_chroma));
        for i in 0..n_beats {
            for j in 0..n_mfcc { fused[[i, j]] = mfcc_norm[[i, j]]; }
            for j in 0..n_chroma { fused[[i, n_mfcc + j]] = chroma_norm[[i, j]]; }
        }
        let structure_features = compute_lag_features(&fused, 8);
        let jumps_weighted = Self::compute_jump_graph(&structure_features, 10, 0.50);
        
        let mut jumps_indices = vec![Vec::new(); n_beats];
        for (i, neighbors) in jumps_weighted.iter().enumerate() {
            for &(target_idx, _score) in neighbors {
                jumps_indices[i].push(target_idx);
            }
        }
        
        // DEBUG: Log checkerboard results
        {
            use std::io::Write;
            use std::fs::OpenOptions;
            if let Ok(mut file) = OpenOptions::new().append(true).open("remixatron_debug.log") {
                let _ = writeln!(file, "\nN_segments: {}", n_segments);
                
                let _ = writeln!(file, "\n=== NOVELTY PEAKS ===");
                let _ = writeln!(file, "Raw peaks: {:?}", raw_peaks);
                let _ = writeln!(file, "Snapped peaks: {:?}", snapped_peaks);
                let _ = writeln!(file, "Segment boundaries: {:?}", boundaries);
                
                // Segment sizes
                let _ = writeln!(file, "\n=== SEGMENT SIZES ===");
                for i in 0..n_segments {
                    let start = boundaries[i];
                    let end = boundaries[i+1];
                    let size = end - start;
                    let label = best_labels_seg[i];
                    let _ = writeln!(file, "Segment {:2}: beats {:3}-{:3} ({:3} beats) -> Cluster {}", 
                        i, start, end, size, label);
                }
                
                // Cluster distribution
                let _ = writeln!(file, "\n=== CLUSTER DISTRIBUTION (K={}) ===", k_final);
                let mut cluster_counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
                for &label in &final_labels {
                    *cluster_counts.entry(label).or_insert(0) += 1;
                }
                let mut counts_vec: Vec<(usize, usize)> = cluster_counts.into_iter().collect();
                counts_vec.sort_by(|a, b| b.1.cmp(&a.1));
                for (cluster_id, count) in &counts_vec {
                    let pct = 100.0 * *count as f32 / n_beats as f32;
                    let _ = writeln!(file, "Cluster {:2}: {:4} beats ({:5.1}%)", cluster_id, count, pct);
                }
                
                // Segment map
                let _ = writeln!(file, "\n=== SEGMENT MAP (full song, {} beats) ===", n_beats);
                let display_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
                let map_str: String = final_labels.iter().map(|&label| {
                    display_chars.chars().nth(label % 36).unwrap_or('?')
                }).collect();
                for (i, chunk) in map_str.as_bytes().chunks(50).enumerate() {
                    let _ = writeln!(file, "{:3}: {}", i * 50, std::str::from_utf8(chunk).unwrap_or("?"));
                }
            }
        }
        
        SegmentationResult {
            labels: final_labels,
            k_optimal: k_final,
            eigenvalues: vec![],  // No eigenvalues computed - clustering disabled
            novelty_curve: novelty,
            peaks: snapped_peaks,
            jumps: jumps_indices,
        }
    }
}
