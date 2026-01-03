use anyhow::Result;
use ndarray::{Array2, Array3, Ix3};
use ort::{session::{Session, builder::GraphOptimizationLevel}, value::Value};
use std::path::Path;

pub struct MelProcessor {
    session: Session,
}

impl MelProcessor {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(model_path)?;
        Ok(Self { session })
    }

    pub fn process(&mut self, samples: &[f32]) -> Result<Array3<f32>> {
        // Input: (Batch, Samples) -> (1, N)
        // Ensure input is f32
        
        let len = samples.len();
        let input_tensor = Array2::from_shape_vec((1, len), samples.to_vec())?;
        
        // Convert to ort Value explicitely
        let input_value = Value::from_array(input_tensor)?;
        
        // Input name: "audio_pcm"
        let outputs = self.session.run(ort::inputs!["audio_pcm" => input_value])?;
        
        // Output: (Batch, Time, Freq)
        // try_extract_tensor returns (shape, data_slice)
        let (shape, data) = outputs["mel_spectrogram"].try_extract_tensor::<f32>()?;
        let shape_usize: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        
        // Construct ArrayD then into dim 3
        let array = ndarray::Array::from_shape_vec(shape_usize, data.to_vec())?;
        let mel_spectrogram: Array3<f32> = array.into_dimensionality::<Ix3>()?;
        
        
        Ok(mel_spectrogram)
    }
}
