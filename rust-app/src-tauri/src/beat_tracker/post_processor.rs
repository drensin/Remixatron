use anyhow::Result;

pub struct MinimalPostProcessor {
    fps: f32,
}

impl MinimalPostProcessor {
    pub fn new(fps: f32) -> Self {
        Self { fps }
    }

    pub fn process(&self, beat_logits: &[f32], downbeat_logits: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        // 1. Peak Picking
        let beat_peaks = self.find_peaks(beat_logits);
        let downbeat_peaks = self.find_peaks(downbeat_logits);

        // 2. Convert to Time
        let beat_times: Vec<f32> = beat_peaks.iter().map(|&idx| idx as f32 / self.fps).collect();
        let mut downbeat_times: Vec<f32> = downbeat_peaks.iter().map(|&idx| idx as f32 / self.fps).collect();

        // 3. Align Downbeats to nearest Beat
        if !beat_times.is_empty() {
            for d_time in downbeat_times.iter_mut() {
                // Find nearest beat
                // Optimize: beat_times is sorted. Use binary search or straight scan? Scan is simple.
                let mut best_diff = f32::MAX;
                let mut best_val = *d_time;
                for &b_time in &beat_times {
                    let diff = (b_time - *d_time).abs();
                    if diff < best_diff {
                        best_diff = diff;
                        best_val = b_time;
                    }
                }
                *d_time = best_val;
            }
        }

        // 4. Deduplicate Downbeats (after alignment, some might merge)
        downbeat_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        downbeat_times.dedup();

        Ok((beat_times, downbeat_times))
    }

    fn find_peaks(&self, logits: &[f32]) -> Vec<usize> {
        let len = logits.len();
        let mut peaks = Vec::new();

        // Max Pool Window = 7 (+/- 3 frames)
        // Stride = 1
        
        // Python: pred_logits != F.max_pool1d(pred_logits, 7, 1, 3)
        // Python Padding = 3.
        
        for i in 0..len {
            // Check threshold (> 0.0 logit implies > 0.5 prob)
            if logits[i] <= 0.0 {
                continue;
            }

            // Check Max Pool condition
            let start = i.saturating_sub(3);
            let end = (i + 4).min(len);
            
            let mut is_max = true;
            for j in start..end {
                if logits[j] > logits[i] {
                    is_max = false;
                    break;
                }
            }
            
            if is_max {
                peaks.push(i);
            }
        }
        
        // Deduplicate adjacent
        self.deduplicate_peaks(peaks, 1)
    }

    fn deduplicate_peaks(&self, peaks: Vec<usize>, width: usize) -> Vec<usize> {
        if peaks.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::new();
        let mut p = peaks[0] as f32;
        let mut c = 1.0;

        #[allow(clippy::needless_range_loop)]  // i used for peaks[i] in running mean; refactoring reduces clarity
        for i in 1..peaks.len() {
            let p2 = peaks[i] as f32;
            if (p2 - p) <= width as f32 {
                c += 1.0;
                // update mean: p += (p2 - p) / c
                // wait, Python code: p += (p2 - p) / c. 
                // p starts as the running mean.
                p += (p2 - p) / c;
            } else {
                result.push(p.round() as usize);
                p = p2;
                c = 1.0;
            }
        }
        result.push(p.round() as usize);
        
        result
    }
}
