//! Encode `f32` samples into WAV / raw PCM (in-process) or compressed formats
//! via an `ffmpeg` subprocess.

use std::io::{Cursor, Read, Write};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// Default wall-clock cap on one `ffmpeg` conversion, after which it is killed.
/// Without it a stuck ffmpeg holds its synthesis permit (released only when the
/// blocking work returns) forever. Override with `SUPERTONIC_FFMPEG_TIMEOUT_SECS`.
const DEFAULT_FFMPEG_TIMEOUT_SECS: usize = 120;

/// Per-conversion ffmpeg timeout, read once from the environment.
fn ffmpeg_timeout() -> Duration {
    static V: OnceLock<usize> = OnceLock::new();
    let secs = *V.get_or_init(|| {
        crate::validate::env_usize("SUPERTONIC_FFMPEG_TIMEOUT_SECS", DEFAULT_FFMPEG_TIMEOUT_SECS)
    });
    Duration::from_secs(secs as u64)
}

#[derive(Debug)]
pub enum AudioError {
    Unsupported(String),
    FfmpegMissing,
    Internal(String),
}

impl std::fmt::Display for AudioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AudioError::Unsupported(fmt) => write!(f, "unsupported format: {fmt}"),
            AudioError::FfmpegMissing => write!(f, "ffmpeg is required for this format"),
            AudioError::Internal(m) => write!(f, "{m}"),
        }
    }
}

/// Whether an `ffmpeg` binary is on PATH. Checked once at startup to advertise
/// which formats the server supports.
pub fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Formats the server can produce given ffmpeg availability.
pub fn available_formats(ffmpeg: bool) -> Vec<&'static str> {
    let mut formats = vec!["wav", "pcm"];
    if ffmpeg {
        formats.extend(["mp3", "opus", "aac", "flac", "ogg"]);
    }
    formats
}

fn to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
        .collect()
}

fn wav_bytes(samples: &[f32], sample_rate: i32) -> Result<Vec<u8>, AudioError> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sample_rate as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::<u8>::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec)
            .map_err(|e| AudioError::Internal(format!("wav init: {e}")))?;
        for s in to_i16(samples) {
            writer
                .write_sample(s)
                .map_err(|e| AudioError::Internal(format!("wav write: {e}")))?;
        }
        writer
            .finalize()
            .map_err(|e| AudioError::Internal(format!("wav finalize: {e}")))?;
    }
    Ok(cursor.into_inner())
}

fn pcm_bytes(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for s in to_i16(samples) {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// `(ffmpeg -f value, media type)` for compressed formats.
fn ffmpeg_format(fmt: &str) -> Option<(&'static str, &'static str)> {
    match fmt {
        "mp3" => Some(("mp3", "audio/mpeg")),
        "opus" => Some(("opus", "audio/opus")),
        "aac" => Some(("adts", "audio/aac")),
        "flac" => Some(("flac", "audio/flac")),
        "ogg" => Some(("ogg", "audio/ogg")),
        _ => None,
    }
}

fn ffmpeg_convert(wav: &[u8], ffmpeg_fmt: &str) -> Result<Vec<u8>, AudioError> {
    let mut child = Command::new("ffmpeg")
        .args([
            "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", ffmpeg_fmt, "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                AudioError::FfmpegMissing
            } else {
                AudioError::Internal(format!("ffmpeg spawn: {e}"))
            }
        })?;

    // Drain stdin/stdout/stderr on separate threads so a full pipe can never
    // deadlock the wait loop below (ffmpeg blocks on output if we stop reading),
    // and so we can poll for exit with a deadline rather than blocking forever.
    let mut stdin = child.stdin.take().expect("stdin piped");
    let mut stdout = child.stdout.take().expect("stdout piped");
    let mut stderr = child.stderr.take().expect("stderr piped");
    let input = wav.to_vec();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&input);
        // stdin dropped here -> EOF for ffmpeg
    });
    let out_reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stdout.read_to_end(&mut buf);
        buf
    });
    let err_reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stderr.read_to_end(&mut buf);
        buf
    });

    // Poll for exit until the deadline, killing a stuck ffmpeg.
    let deadline = Instant::now() + ffmpeg_timeout();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = writer.join();
                    let _ = out_reader.join();
                    let _ = err_reader.join();
                    return Err(AudioError::Internal("ffmpeg timed out".to_string()));
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = writer.join();
                let _ = out_reader.join();
                let _ = err_reader.join();
                return Err(AudioError::Internal(format!("ffmpeg wait: {e}")));
            }
        }
    };

    let _ = writer.join();
    let stdout = out_reader.join().unwrap_or_default();
    let stderr = err_reader.join().unwrap_or_default();

    if !status.success() {
        return Err(AudioError::Internal(format!(
            "ffmpeg failed: {}",
            String::from_utf8_lossy(&stderr).trim()
        )));
    }
    Ok(stdout)
}

/// Encode `samples` into `fmt`, returning `(bytes, media_type)`.
pub fn encode(
    samples: &[f32],
    sample_rate: i32,
    fmt: &str,
) -> Result<(Vec<u8>, &'static str), AudioError> {
    let fmt = fmt.to_ascii_lowercase();
    match fmt.as_str() {
        "wav" => Ok((wav_bytes(samples, sample_rate)?, "audio/wav")),
        // Raw little-endian 16-bit mono PCM. Not `audio/L16` — that MIME type is
        // defined as big-endian — so serve it as opaque bytes.
        "pcm" => Ok((pcm_bytes(samples), "application/octet-stream")),
        other => {
            let (ff, media) =
                ffmpeg_format(other).ok_or_else(|| AudioError::Unsupported(other.to_string()))?;
            let wav = wav_bytes(samples, sample_rate)?;
            Ok((ffmpeg_convert(&wav, ff)?, media))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_i16_clips_out_of_range() {
        assert_eq!(to_i16(&[2.0, -2.0, 0.0]), vec![32767, -32767, 0]);
    }

    #[test]
    fn pcm_is_little_endian_i16() {
        // 1.0 -> 32767 -> 0x7FFF -> [0xFF, 0x7F]
        assert_eq!(pcm_bytes(&[1.0]), vec![0xFF, 0x7F]);
        assert_eq!(pcm_bytes(&[0.0, 0.0]).len(), 4);
    }

    #[test]
    fn wav_has_riff_wave_header() {
        let bytes = wav_bytes(&[0.0; 8], 24_000).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
    }

    #[test]
    fn available_formats_gate_on_ffmpeg() {
        assert_eq!(available_formats(false), vec!["wav", "pcm"]);
        assert!(available_formats(true).contains(&"mp3"));
    }

    #[test]
    fn encode_rejects_unknown_format() {
        let err = encode(&[0.0], 24_000, "xyz").unwrap_err();
        assert!(matches!(err, AudioError::Unsupported(_)));
    }

    #[test]
    fn encode_media_types() {
        assert_eq!(encode(&[0.0], 24_000, "wav").unwrap().1, "audio/wav");
        // pcm is little-endian, so not the big-endian `audio/L16` MIME type.
        assert_eq!(encode(&[0.0], 24_000, "pcm").unwrap().1, "application/octet-stream");
    }

    #[test]
    fn ffmpeg_format_maps_known_only() {
        assert_eq!(ffmpeg_format("mp3"), Some(("mp3", "audio/mpeg")));
        assert_eq!(ffmpeg_format("wav"), None);
    }
}
