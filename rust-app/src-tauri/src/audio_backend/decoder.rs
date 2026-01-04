
use std::path::Path;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

// Consolidated implementation
pub fn decode_audio_file(path: &str) -> Result<(Vec<f32>, u32, u64), Box<dyn std::error::Error>> {
    let src = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = Path::new(path).extension() {
        if let Some(ext_str) = ext.to_str() {
            hint.with_extension(ext_str);
        }
    }

    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &fmt_opts, &meta_opts)?;

    let mut format = probed.format;
    let track = format.default_track().ok_or("No default track")?;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())?;

    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.ok_or("Unknown sample rate")?;
    let channels = track.codec_params.channels.ok_or("Unknown channel count")?.count() as u32;
    
    let mut samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
             Ok(packet) => packet,
            Err(Error::IoError(ref e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(Error::IoError(ref e)) if e.kind() == std::io::ErrorKind::InvalidData => {
                 continue; // Skip invalid packets 
            }
            Err(e) => return Err(Box::new(e)),
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                 let spec = *decoded.spec();
                 let duration = decoded.capacity() as u64;
                 let mut sample_buf = SampleBuffer::<f32>::new(duration, spec);
                 sample_buf.copy_interleaved_ref(decoded);
                 samples.extend_from_slice(sample_buf.samples());
            }
             Err(Error::IoError(ref e)) if e.kind() == std::io::ErrorKind::InvalidData => {
                 continue;
            }
            Err(Error::IoError(_)) => {
                continue;
            }
            Err(Error::DecodeError(_)) => {
                continue;
            }
            Err(e) => return Err(Box::new(e)),
        }
    }

    Ok((samples, sample_rate, channels as u64))
}


