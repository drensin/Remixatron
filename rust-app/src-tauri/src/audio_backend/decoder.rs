
use std::path::Path;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

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
                 
                 // Interleaved samples. If stereo, L, R, L, R...
                 // BeatNet expects MONO.
                 // We should convert to mono here or return interleaved and convert later.
                 // Let's return interleaved and handle mixdown later.
                 samples.extend_from_slice(sample_buf.samples());
            }
            Err(Error::IoError(_)) => {
                // The packet failed to decode due to an IO error, skip the packet.
                continue;
            }
            Err(Error::DecodeError(_)) => {
                // The packet failed to decode due to invalid data, skip the packet.
                continue;
            }
            Err(e) => return Err(Box::new(e)),
        }
    }
    
    // Total frames = samples.len() / channels
    // Since we don't know channels yet locally, we should return channels too?
    // But format knew it.
    // Let's assume for now we just return raw.
    // Wait, sample rate is u32.
    
    Ok((samples, sample_rate, 0)) // 0 dummy for now (duration?)
}
