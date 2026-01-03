use ndarray::{Array2, Array1, s};
use linfa::traits::{Fit, Predict};
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
}

impl StructureAnalyzer {
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
                         score = (k as f32) 
                                + (10.0 * silhouette) 
                                + (min_seg_len.min(8) as f32) 
                                + ratio;
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
            eigenvalues: sorted_evals,
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
