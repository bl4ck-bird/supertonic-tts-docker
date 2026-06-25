//! Wraps the upstream `helper.rs` inference code behind a small, server-friendly
//! API. `TextToSpeech::call` takes `&mut self`, so synthesis is serialized
//! behind a `Mutex`. Voice styles live in a separate [`VoiceStore`].

use std::path::Path;
use std::sync::{Mutex, MutexGuard};

use crate::helper::{chunk_text, load_text_to_speech, TextToSpeech};
use crate::voice_store::VoiceStore;

/// Asset root inside the container; `onnx/` and `voice_styles/` live under it.
pub const ASSETS_DIR: &str = "/assets";

/// Acquire a lock, recovering a poisoned guard instead of panicking so one
/// panicked request cannot take the endpoint down with it.
pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[derive(Debug)]
pub enum EngineError {
    UnknownVoice(String),
    BadRequest(String),
    Internal(String),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::UnknownVoice(v) => write!(f, "unknown voice: {v}"),
            EngineError::BadRequest(m) => write!(f, "{m}"),
            EngineError::Internal(m) => write!(f, "{m}"),
        }
    }
}

/// Parameters for one synthesis call (mirrors `TextToSpeech::call`).
pub struct SynthParams {
    pub text: String,
    pub lang: String,
    pub voice: String,
    pub total_steps: usize,
    pub speed: f32,
    pub silence_duration: f32,
    /// Optional outer chunk target, in bytes. `call` already chunks internally
    /// (≤300 bytes, ≤120 for ko/ja), so this only forces a smaller pre-split. It
    /// is best-effort: an unbreakable token (no spaces) can still exceed it.
    pub max_chunk_length: Option<usize>,
}

pub struct Engine {
    tts: Mutex<TextToSpeech>,
    voices: VoiceStore,
    sample_rate: i32,
}

impl Engine {
    /// Load the ONNX model from `<assets_dir>/onnx` and locate voice styles in
    /// `<assets_dir>/voice_styles`.
    pub fn load(assets_dir: &Path, use_gpu: bool) -> Result<Self, EngineError> {
        let onnx_dir = assets_dir.join("onnx");
        let onnx_dir_s = onnx_dir
            .to_str()
            .ok_or_else(|| EngineError::Internal("non-utf8 onnx dir".into()))?;
        let tts = load_text_to_speech(onnx_dir_s, use_gpu)
            .map_err(|e| EngineError::Internal(format!("failed to load model: {e}")))?;
        let sample_rate = tts.sample_rate;
        Ok(Self {
            tts: Mutex::new(tts),
            voices: VoiceStore::new(VoiceStore::dir(assets_dir)),
            sample_rate,
        })
    }

    pub fn sample_rate(&self) -> i32 {
        self.sample_rate
    }

    pub fn list_voices(&self) -> Vec<String> {
        self.voices.list()
    }

    /// Register a custom voice for this running engine; immediately usable.
    pub fn import_voice(&self, name: &str, json: &str) -> Result<(), EngineError> {
        self.voices.import(name, json)
    }

    /// Synthesize one utterance, returning `(samples, duration_seconds)`.
    ///
    /// With `max_chunk_length` set, the text is pre-split at that length and the
    /// pieces are synthesized and concatenated (see [`SynthParams::max_chunk_length`]).
    pub fn synthesize(&self, params: &SynthParams) -> Result<(Vec<f32>, f32), EngineError> {
        let c = crate::validate::validate(params)?;
        let style = self.voices.resolve(&params.voice)?;
        let mut tts = lock(&self.tts);

        let call_one = |tts: &mut TextToSpeech, text: &str| {
            tts.call(text, &params.lang, &style, c.total_steps, c.speed, c.silence_duration)
                .map_err(|e| EngineError::Internal(format!("synthesis failed: {e}")))
        };

        match c.max_chunk_length {
            None => call_one(&mut tts, &params.text),
            Some(max_len) => {
                // Separate adjacent outer pieces with the same `silence_duration`
                // gap `call` puts between its internal chunks. Stream piece by
                // piece (don't collect them all first) to keep peak memory bounded.
                let silence_len = (c.silence_duration * self.sample_rate as f32) as usize;
                let mut samples = Vec::new();
                let mut duration = 0.0_f32;
                let mut emitted = false;
                for chunk in chunk_text(&params.text, Some(max_len)) {
                    // Skip whitespace-only pieces; never feed the model an empty string.
                    if chunk.trim().is_empty() {
                        continue;
                    }
                    if emitted {
                        samples.resize(samples.len() + silence_len, 0.0);
                        duration += c.silence_duration;
                    }
                    let (wav, dur) = call_one(&mut tts, &chunk)?;
                    samples.extend_from_slice(&wav);
                    duration += dur;
                    emitted = true;
                }
                Ok((samples, duration))
            }
        }
    }
}
