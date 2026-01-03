use anyhow::Result;
use ndarray::{Array3, Axis};
use ort::{session::{Session, builder::GraphOptimizationLevel}, value::Value};
use std::path::Path;

pub struct BeatProcessor {
    session: Session,
}

impl BeatProcessor {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(model_path)?;
        Ok(Self { session })
    }

    pub fn process(&mut self, mel: &Array3<f32>) -> Result<(Vec<f32>, Vec<f32>)> {
        // Input: mel spectrogram (Batch, Time, Freq)
        // We only support Batch=1 for now.
        let shape = mel.shape();
        let full_time = shape[1];
        let n_mels = shape[2];
        
        let chunk_size = 1500;
        let border_size = 6;
        let step_size = chunk_size - 2 * border_size;
        
        let mut starts = Vec::new();
        let mut curr = -(border_size as i32);
        while curr < (full_time as i32 - border_size as i32) {
            starts.push(curr);
            curr += step_size as i32;
        }
        
        // "Avoid short end": move last start to align end
        if full_time > (chunk_size - 2 * border_size) {
            if let Some(last) = starts.last_mut() {
                *last = full_time as i32 - (chunk_size as i32 - border_size as i32);
            }
        }
        
        // Output buffers
        // Initialize with a low value (-1000.0)
        let mut beat_logits = vec![-1000.0; full_time];
        let mut downbeat_logits = vec![-1000.0; full_time];
        
        // Python "keep_first" logic reverses the list so earlier overwrite later.
        // Wait, "process in reverse order, so predictions of earlier excerpts overwrite later ones".
        // This means earlier chunks (in time) are written LAST.
        // So the loop goes from end to start?
        // Let's implement Python loop: `starts = reversed(list(starts))`
        starts.reverse();
        
        // Extract 2D view (Time, Freq)
        let mel_view = mel.index_axis(Axis(0), 0); 
        
        for start in starts {
             let s_start = start.max(0) as usize;
             let s_end = (start + chunk_size as i32).min(full_time as i32) as usize;
             
             // Extract slice
             let slice = mel_view.slice(ndarray::s![s_start..s_end, ..]);
             
             // Compute padding
             let pad_left = (0 - start).max(0) as usize;
             // right pad: see python logic
             // right=max(0, min(border_size, start + chunk_size - len(spect)))
             // simplified: target size is chunk_size.
             let current_len = slice.nrows();
             let _pad_right = chunk_size - current_len - pad_left;
             
             // Construct padded chunk (1, chunk_size, 128)
             let mut chunk = Array3::<f32>::zeros((1, chunk_size, n_mels));
             
             // Assign slice to middle
             // We need to handle this carefully.
             // slice is (Time', Freq). Target is (1, Time, Freq).
             // Assign to chunk[0, pad_left..pad_left+len, :]
             chunk.slice_mut(ndarray::s![0, pad_left..pad_left+current_len, ..]).assign(&slice);
             
             // Run Inference
             let chunk_value = Value::from_array(chunk)?;
             let outputs = self.session.run(ort::inputs!["mel_spectrogram" => chunk_value])?;
             
             let (_, b_data) = outputs["beat_logits"].try_extract_tensor::<f32>()?;
             let (_, d_data) = outputs["downbeat_logits"].try_extract_tensor::<f32>()?;
             
             // Strip borders
             // Python: pred[border:-border]
             // Rust slice: [border..chunk_size-border]
             let valid_len = chunk_size - 2 * border_size;
             let b_valid = &b_data[border_size..chunk_size-border_size];
             let d_valid = &d_data[border_size..chunk_size-border_size];
             
             // Write to output
             // Python: piece[start+border : start+chunk-border] = valid
             let write_start = (start + border_size as i32) as usize;
             let write_end = (start + chunk_size as i32 - border_size as i32) as usize;
             
             // Bound check (vital for the last chunk shift logic)
             // The start/end logic guarantees alignment, but let's clamp just in case?
             // Actually, `starts` logic ensures bounds?
             // If start was negative, write_start = -border + border = 0. Correct.
             
             // Verify lengths match
             let target_len = write_end - write_start;
             if target_len != valid_len {
                 // Loop/logic mismatch?
                 // Usually valid_len is constant.
                 // write range depends on start.
             }
             
             // Copy
             for i in 0..valid_len {
                 let dest_idx = write_start + i;
                 if dest_idx < full_time {
                     beat_logits[dest_idx] = b_valid[i];
                     downbeat_logits[dest_idx] = d_valid[i];
                 }
             }
        }
        
        Ok((beat_logits, downbeat_logits))
    }
}
