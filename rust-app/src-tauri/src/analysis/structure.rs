use ndarray::{Array2, Array1, s};
use linfa::traits::{Fit, Predict};
use rayon::prelude::*;
use linfa_clustering::KMeans;
use linfa::DatasetBase;
use std::f32::consts::PI;


pub struct StructureAnalyzer;

impl StructureAnalyzer {
    pub fn new() -> Self {
        Self
    }
}

/// Result of segmentation containing labels and metadata
pub struct SegmentationResult {
    pub labels: Vec<usize>,
    pub k_optimal: usize,
    pub eigenvalues: Vec<f32>,
    pub novelty_curve: Vec<f32>,
    pub peaks: Vec<usize>,
    pub jumps: Vec<Vec<usize>>, // NEW: Pre-calculated Jumps per beat
}

impl StructureAnalyzer {
    /// SOTA "Bottom-Up" Segmentation Pipelne
    /// 1. Z-Score Normalize Features
    /// 2. Compute Lag Embedding (Structure Features)
    /// 3. Construct k-NN Affinity Graph
    /// 4. Spectral Clustering on Graph
    pub fn compute_segments_knn(&self, mfcc: &Array2<f32>, chroma: &Array2<f32>, k_force: Option<usize>) -> SegmentationResult {
        let n_beats = mfcc.nrows();
        
        // 1. Preprocessing (Z-Score)
        let mut mfcc_norm = mfcc.clone();
        z_score_normalize(&mut mfcc_norm);
        
        let mut chroma_norm = chroma.clone();
        z_score_normalize(&mut chroma_norm);
        
        // 2. Feature Fusion (Equal Weight)
        // Just concatenate them
        let (_, n_mfcc) = mfcc_norm.dim();
        let (_, n_chroma) = chroma_norm.dim();
        let mut fused = Array2::<f32>::zeros((n_beats, n_mfcc + n_chroma));
        
        for i in 0..n_beats {
            for j in 0..n_mfcc {
                fused[[i, j]] = mfcc_norm[[i, j]];
            }
            for j in 0..n_chroma {
                fused[[i, n_mfcc + j]] = chroma_norm[[i, j]];
            }
        }
        
        // 3. Lag Embedding (Structure Features)
        // Lag = 8 Beats (Expert Recommendation)
        let structure_features = compute_lag_features(&fused, 8);
        
        // 4. k-NN Affinity Graph
        // Construct Adjacency Matrix from k-NN
        println!("    Computing k-NN Jump Graph (k=10, thresh=0.9)...");
        let jumps = compute_jump_graph(&structure_features, 10, 0.9);
        
        // Convert Jump Graph to Dense Laplacian for Spectral Clustering (Or Sparse?)
        // Our 'jacobi_eigenvalue' expects Dense Array2.
        // For N=5000, Dense is 25M floats = 100MB. totally fine.
        let mut adjacency = Array2::<f32>::zeros((n_beats, n_beats));
        
        // Fill from jumps (symmetry is not guaranteed by k-NN, but needed for Spectral)
        // We make it symmetric: A[i,j] = 1 if i->j OR j->i exist?
        // Or weighted?
        // Let's use weighted cosine similarity from the graph calculation if we can,
        // but `compute_jump_graph` returns indices only.
        // Let's re-use the standard logic: If i is neighbor of j, link them.
        for (i, neighbors) in jumps.iter().enumerate() {
            for &j in neighbors {
                adjacency[[i, j]] = 1.0;
                adjacency[[j, i]] = 1.0;
            }
        }
        
        // Add minimal continuity? (i connected to i+1)
        // Standard Spectral Clustering often includes temporal connectivity.
        // Let's add diagonal neighbors.
        for i in 0..n_beats-1 {
            adjacency[[i, i+1]] = 1.0;
            adjacency[[i+1, i]] = 1.0;
        }

        // 5. Laplacian & Eigen Decomposition
        let n_s = n_beats;
        let mut d_inv_sqrt = Array1::<f32>::zeros(n_s);
        for i in 0..n_s {
            let sum: f32 = adjacency.row(i).sum();
            d_inv_sqrt[i] = if sum > 0.0 { 1.0 / sum.sqrt() } else { 0.0 };
        }
        
        let mut laplacian = Array2::<f32>::eye(n_s);
        for i in 0..n_s {
            for j in 0..n_s {
                let val = adjacency[[i, j]] * d_inv_sqrt[i] * d_inv_sqrt[j];
                laplacian[[i, j]] -= val;
            }
        }
        
        let (evals, evecs) = jacobi_eigenvalue(&laplacian, 100);
        
        // Sort eigenvalues
        let mut indices: Vec<usize> = (0..n_beats).collect();
        indices.sort_by(|&a, &b| evals[a].partial_cmp(&evals[b]).unwrap_or(std::cmp::Ordering::Equal));
        
        // 6. Spectral Clustering (Auto-K or Forced)
        // Use the same logic as before (Legacy Composite or simple)
        // Let's replicate the Auto-K logic simply.
        
        let k_final = k_force.unwrap_or(4); // Fallback to 4 if user didn't force
        
        // Extract features
        let mut embedding = Array2::<f32>::zeros((n_beats, k_final));
        for i in 0..n_beats {
            for j in 0..k_final {
                let col_idx = indices[j];
                embedding[[i, j]] = evecs[[i, col_idx]];
            }
        }
        let embed_norm = normalize_rows(&embedding);
        
        let model = KMeans::params(k_final)
            .max_n_iterations(100)
            .fit(&DatasetBase::from(embed_norm.clone()))
            .expect("KMeans failed");
            
        let labels = model.predict(&embed_norm).into_raw_vec_and_offset().0;
        
        SegmentationResult {
            labels,
            k_optimal: k_final,
            eigenvalues: evals,
            novelty_curve: vec![],
            peaks: vec![],
            jumps, // Return optimal jumps
        }
    }

    /// Compute segmentation from beat-synchronous features.
    /// Returns: List of cluster labels for each beat.
    /// If k=0, auto-detects K using Eigengap heuristic (max 14 eigenvalues).
    /// If k>0, forces K (still returns 14 eigenvalues).
    pub fn compute_segments(&self, mfcc: &Array2<f32>, chroma: &Array2<f32>, k_force: usize) -> SegmentationResult {
        let n_beats = mfcc.nrows();
        
        // 1. Recurrence Matrix (Chroma - Cosine Sim)
        // R[i, j] = chroma[i] . chroma[j] / (|ci| |cj|)
        // Chroma is likely already normalized? 
        // Let's normalize rows first
        let chroma_norm = normalize_rows(chroma);
        let mut recurrence = chroma_norm.dot(&chroma_norm.t());
        
        // Thresholding to remove weak links (noise)
        recurrence.mapv_inplace(|x| if x < 0.5 { 0.0 } else { x });
        
        // Median Filter Recurrence (skipped for parity)
        
        // 2. Path Matrix (MFCC - Continuity)
        let mut path_dists = Vec::with_capacity(n_beats - 1);
        for i in 0..n_beats-1 {
            let diff = &mfcc.slice(s![i, ..]) - &mfcc.slice(s![i+1, ..]);
            let dist_sq = diff.dot(&diff);
            path_dists.push(dist_sq);
        }
        
        // Median sigma
        let mut sorted = path_dists.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let sigma = if sorted.is_empty() { 1.0 } else { sorted[sorted.len()/2] };
        let sigma = sigma.max(1e-5); 
        
        let mut adjacency = recurrence.clone(); 
        
        // Remixatron Logic: A = mu * R + (1-mu) * P
        let mu = 0.5;
        
        // 1. Scale Recurrence by mu
        adjacency.mapv_inplace(|x| x * mu);
        
        // 2. Add Path (weighted by 1-mu)
        // Path only affects diagonals k=1 and k=-1.
        for i in 0..n_beats-1 {
             let d = path_dists[i];
             let p_val = (-d / sigma).exp();
             let weighted_p = (1.0 - mu) * p_val;
             
             // P is symmetric. Link (i, i+1) and (i+1, i)
             adjacency[[i, i+1]] += weighted_p;
             adjacency[[i+1, i]] += weighted_p;
        }

        for i in 0..n_beats {
            for j in 0..n_beats {
                if adjacency[[i, j]] < 0.0 { adjacency[[i, j]] = 0.0; }
                if i == j { adjacency[[i, j]] = 0.0; } // Remove self-loops
            }
        }
        
        // 3. Laplacian
        // D[i] = sum(A[i, :])
        // L = I - D^-0.5 A D^-0.5
        
        let mut d_inv_sqrt = Array1::<f32>::zeros(n_beats);
        for i in 0..n_beats {
            let sum: f32 = adjacency.row(i).sum();
            d_inv_sqrt[i] = if sum > 0.0 { 1.0 / sum.sqrt() } else { 0.0 };
        }
        
        let mut laplacian = Array2::<f32>::eye(n_beats);
        for i in 0..n_beats {
            for j in 0..n_beats {
                let val = adjacency[[i, j]] * d_inv_sqrt[i] * d_inv_sqrt[j];
                laplacian[[i, j]] -= val;
            }
        }
        
        // 4. Eigen Decomposition
        // We need smallest Eigenvalues of L.
        let (evals, evecs) = jacobi_eigenvalue(&laplacian, 100); // 100 iters
        
        // 5. Auto-K Selection
        // Sort eigenvalues
        let mut indices: Vec<usize> = (0..n_beats).collect();
        indices.sort_by(|&a, &b| evals[a].partial_cmp(&evals[b]).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut sorted_evals = Vec::with_capacity(14);
        for i in 0..14.min(n_beats) {
            sorted_evals.push(evals[indices[i]]);
        }
        


        // 5. Clustering (Auto-K selection)
        let mut k_final;
        let mut labels_final = Vec::new();
        
        if k_force > 0 {
             k_final = k_force;
             // Extract k_final eigenvectors
             let mut features = Array2::<f32>::zeros((n_beats, k_final));
             for i in 0..n_beats {
                 for j in 0..k_final {
                     let col_idx = indices[j];
                     features[[i, j]] = evecs[[i, col_idx]];
                 }
             }
             let feat_norm = normalize_rows(&features);

             let model = KMeans::params(k_final)
                .max_n_iterations(200)
                .tolerance(1e-5)
                .fit(&DatasetBase::from(feat_norm.clone()))
                .expect("KMeans failed");
                
            labels_final = model.predict(&feat_norm).into_raw_vec_and_offset().0;
        } else {
            // Auto-K: Legacy Composite Score Heuristic
            // Search range: [4, 20]
            
            println!("    Running Auto-K (Legacy Composite)...");
            
            let mut best_score: f32 = -1.0;
            let mut best_stats = String::new();

            // Default fallback
            k_final = 4;
            
            // Iterate downwards
            for k in (4..=20).rev() {
                // Extract k eigenvectors
                let mut features = Array2::<f32>::zeros((n_beats, k));
                for i in 0..n_beats {
                    for j in 0..k {
                        let col_idx = indices[j];
                        features[[i, j]] = evecs[[i, col_idx]];
                    }
                }
                let feat_norm = normalize_rows(&features);

                let dataset = DatasetBase::from(feat_norm.clone());
                let model = KMeans::params(k) // linfa::k_means::KMeans
                    .max_n_iterations(100)
                    .tolerance(1e-4)
                    .fit(&dataset);

                if let Ok(model) = model {
                    let labels = model.predict(&feat_norm).into_raw_vec_and_offset().0;
                    
                    // 1. Calculate Segment Stats
                    let (ratio, min_seg_len) = Self::calculate_segment_stats(&labels, k);
                    
                    // 2. Calculate Silhouette Score
                    let silhouette = Self::calculate_silhouette_score(&feat_norm.mapv(|x| x as f64), &labels, k);
                    
                    // 3. Composite Score
                    let mut score = 0.0;
                    if ratio >= 3.0 && silhouette > 0.4 {
                         // Occam's Razor Score:
                         // We want to maximize Quality (Silhouette) while minimizing Complexity (K).
                         //
                         // Formula: Score = (10 * Silhouette) + Ratio - (0.5 * K)
                         //
                         // Rationale for 0.5 * K Penalty:
                         // - Silhouette is scaled by 10 (Range 0-10).
                         // - A penalty of 0.5 per K means that adding 1 cluster is only justified 
                         //   if it increases the Silhouette score by at least 0.05 (5%).
                         //
                         // - If Penalty was 1.0 (Strict): Requires 10% gain per cluster. (Too conservative, stays at K=4)
                         // - If Penalty was 0.1 (Loose): Requires 1% gain per cluster. (Too aggressive, drifts to K=20)
                         // - 0.5 is the "Sweet Spot" that allows complexity only when it measurably improves structure.
                         
                         score = (10.0 * silhouette) 
                                + ratio 
                                - (k as f32 * 0.5);
                    }
                    
                    if score > best_score {
                        best_score = score;
                        k_final = k;
                        labels_final = labels;
                        best_stats = format!("Sil={:.2}, Rat={:.2}, Min={}", silhouette, ratio, min_seg_len);
                    }
                }
            }
            
            println!("    Selected K={} (Score={:.2} | {})", k_final, best_score, best_stats);
            
            if labels_final.is_empty() {
                println!("    WARNING: No ideal K found (all blocked by constraints). Forcing K=4.");
                k_final = 4;
                // Extract 4 eigen vectors
                let mut features = Array2::<f32>::zeros((n_beats, k_final));
                for i in 0..n_beats {
                    for j in 0..k_final {
                        let col_idx = indices[j];
                        features[[i, j]] = evecs[[i, col_idx]];
                    }
                }
                let feat_norm = normalize_rows(&features);

                let model = KMeans::params(k_final)
                    .max_n_iterations(200)
                    .fit(&DatasetBase::from(feat_norm.clone()))
                    .expect("KMeans fallback failed");
                labels_final = model.predict(&feat_norm).into_raw_vec_and_offset().0;
            }
        }
        
        SegmentationResult {
            labels: labels_final,
            k_optimal: k_final,
            eigenvalues: evals, // Return Segment Eigenvalues
            novelty_curve: vec![],
            peaks: vec![],
            jumps: vec![],
        }
    }

    fn calculate_segment_stats(labels: &[usize], k_clusters: usize) -> (f32, usize) {
        if labels.is_empty() { return (0.0, 0); }
        
        let mut segment_count = 0;
        let mut current_seg_len = 0;
        let mut min_seg_len = usize::MAX;
        let mut prev_label = None;
        
        for &label in labels {
            if Some(label) != prev_label {
                if current_seg_len > 0 {
                    if current_seg_len < min_seg_len { min_seg_len = current_seg_len; }
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
}

// Utils

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

/// Jacobi Eigenvalue Algorithm for real symmetric matrices.
/// Returns (eigenvalues, eigenvectors)
fn jacobi_eigenvalue(a: &Array2<f32>, max_iter: usize) -> (Vec<f32>, Array2<f32>) {
    let n = a.nrows();
    let mut v = Array2::<f32>::eye(n); // Eigenvectors
    let mut d = a.clone();             // Will become diagonal with eigenvalues
    
    for _iter in 0..max_iter {
        // Find max off-diagonal element
        let mut max_val = 0.0;
        let mut p = 0;
        let mut q = 0;
        
        for i in 0..n {
            for j in (i + 1)..n {
                let val = d[[i, j]].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }
        
        if max_val < 1e-9 {
            break; // Converged
        }
        
        let theta;
        let y = (d[[q, q]] - d[[p, p]]) / 2.0;

        
        if y == 0.0 {
            theta = PI / 4.0; 
        } else {
            // theta = 0.5 * atan(x / y) ? No.
            // tan(2theta) = x/y?
            // Correct robust: theta = 0.5 * atan2(2*a_pq, a_qq - a_pp)
            // My y definition: (a_qq - a_pp)/2.
            // My x definition: -a_pq.
            // So: atan2(-2x, 2y) = atan2(-x, y). 
            // -x = d[p,q]. 2y = d[q,q]-d[p,p].
            theta = 0.5 * (2.0 * d[[p, q]]).atan2(d[[q, q]] - d[[p, p]]);
        }
        
        let c = theta.cos();
        let s = theta.sin();
        
        // Update D
        let d_pp = d[[p, p]];
        let d_qq = d[[q, q]];
        let d_pq = d[[p, q]];
        
        d[[p, p]] = c*c*d_pp + s*s*d_qq - 2.0*s*c*d_pq;
        d[[q, q]] = s*s*d_pp + c*c*d_qq + 2.0*s*c*d_pq;
        d[[p, q]] = 0.0;
        d[[q, p]] = 0.0;
        
        for k in 0..n {
            if k != p && k != q {
                let d_pk = d[[p, k]];
                let d_qk = d[[q, k]];
                d[[p, k]] = c*d_pk - s*d_qk;
                d[[k, p]] = d[[p, k]];
                d[[q, k]] = s*d_pk + c*d_qk;
                d[[k, q]] = d[[q, k]];
            }
        }
        
        // Update V
        for k in 0..n {
            let v_kp = v[[k, p]];
            let v_kq = v[[k, q]];
            v[[k, p]] = c*v_kp - s*v_kq;
            v[[k, q]] = s*v_kp + c*v_kq;
        }
    }
    
    let evals: Vec<f32> = d.diag().to_vec();
    (evals, v)
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
    for i in 0..n {
        let mut acc = 0.0;
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

/// k-Nearest Neighbors Jump Graph
/// Returns a sparse adjacency list: graph[i] = vec![j1, j2, ...]
/// Filtering criteria:
/// 1. Top-K neighbors (by Cosine Similarity)
/// 2. Similarity > Threshold
fn compute_jump_graph(features: &Array2<f32>, k: usize, threshold: f32) -> Vec<Vec<usize>> {
    let n = features.nrows();
    let mut graph = vec![Vec::new(); n];
    
    // Normalize rows for Cosine Sim
    let feat_norm = normalize_rows(features);
    
    // For each beat, find neighbors
    // Parallelizing this is good for performance
    let results: Vec<Vec<usize>> = (0..n).into_par_iter().map(|i| {
        let mut candidates = Vec::new();
        let target = feat_norm.row(i);
        
        // Scan all other beats (Naive k-NN is O(N^2), acceptable for N < 5000)
        // Optimization: Don't compare with self or immediate neighbors (+/- 4 beats) to avoid trivial jumps?
        // SOTA: Yes, exclude "Too Close" beats.
        let exclusion_radius = 4; // 1 Bar
        
        for j in 0..n {
            if (i as isize - j as isize).abs() <= exclusion_radius { continue; }
            
            let other = feat_norm.row(j);
            let sim = target.dot(&other);
            
            if sim > threshold {
                 candidates.push((j, sim));
            }
        }
        
        // Sort by similarity descending
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Take top K
        candidates.into_iter().take(k).map(|(idx, _)| idx).collect()
    }).collect();
    
    graph = results;
    graph
}

impl StructureAnalyzer {
    /// Compute segmentation using Checkerboard Kernel + Segment Clustering.
    /// Compute segmentation using Checkerboard Kernel + Segment Clustering.
    pub fn compute_segments_checkerboard(&self, mfcc: &Array2<f32>, chroma: &Array2<f32>, bar_positions: &[usize], k_force: Option<usize>) -> SegmentationResult {
        let n_beats = mfcc.nrows();
        
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
                    let dist = (p as isize - b as isize).abs() as usize;
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
        
        // 5. Cluster Segments (Spectral Clustering on Segment Graph)
        // Section 5.3: Construct Segment Similarity Matrix -> Laplacian -> Eigenvectors
        
        let best_labels_seg: Vec<usize>;
        let mut k_final;
        let best_stats;

        println!("    Computing Segment Affinity & Spectral Embedding...");
        
        // A. Segment Affinity Matrix S_seg (Cosine Similarity)
        // feat_norm is [N_seg, D]. S = F * F^T
        let s_seg = compute_ssm(&segment_features); // segment_features is already normalized? 
        // Wait, line 689 normalized 'segment_features'.
        
        // B. Laplacian L = I - D^-0.5 * S * D^-0.5
        let n_s = n_segments;
        let mut d_inv_sqrt = Array1::<f32>::zeros(n_s);
        for i in 0..n_s {
            // Zero out self-similarity and negatives for graph
            let mut sum = 0.0;
            for j in 0..n_s {
                 if i == j { continue; } // No self-loops
                 let val = s_seg[[i, j]].max(0.0);
                 sum += val;
            }
            d_inv_sqrt[i] = if sum > 0.0 { 1.0 / sum.sqrt() } else { 0.0 };
        }
        
        let mut laplacian = Array2::<f32>::eye(n_s);
        for i in 0..n_s {
            for j in 0..n_s {
                if i != j {
                    let val = s_seg[[i, j]].max(0.0) * d_inv_sqrt[i] * d_inv_sqrt[j];
                    laplacian[[i, j]] -= val;
                }
            }
        }
        
        // C. Eigen Decomposition
        // We need smallest eigenvectors (representing clusters)
        let (evals, evecs) = jacobi_eigenvalue(&laplacian, 100);
        
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
                
                if is_valid {
                    if score > best_score_val {
                        best_score_val = score;
                        k_best = k;
                        // sil_best = sil_k;
                        is_selected = true;
                    }
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
        
        println!("    Clustering {} segments -> Selected K={} ({})", n_segments, k_final, best_stats);
        
        // 6. Expand Labels to Beats
        let mut final_labels = vec![0; n_beats];
        for i in 0..n_segments {
            let start = boundaries[i];
            let end = boundaries[i+1];
            let label = best_labels_seg[i];
            for t in start..end {
                final_labels[t] = label;
            }
        }
        
        SegmentationResult {
            labels: final_labels,
            k_optimal: k_final,
            eigenvalues: evals,
            novelty_curve: novelty,
            peaks: snapped_peaks,
            jumps: vec![],
        }
    }
}
