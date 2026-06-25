//! Request/response payloads and their serde defaults. `#[schema(...)]` ranges
//! are documentation only; validation and clamping happen in [`crate::validate`]
//! (called by `Engine::synthesize`).

use serde::{Deserialize, Serialize};
use serde_json::Value;
use utoipa::ToSchema;

// ---- defaults ---------------------------------------------------------------

pub(crate) fn default_voice() -> String {
    "M1".to_string()
}
pub(crate) fn default_format() -> String {
    "wav".to_string()
}
pub(crate) fn default_speed() -> f32 {
    1.05
}
pub(crate) fn default_openai_speed() -> f32 {
    1.0
}
pub(crate) fn default_steps() -> usize {
    8
}
pub(crate) fn default_silence() -> f32 {
    0.3
}

// ---- requests ---------------------------------------------------------------

/// OpenAI-compatible speech request (`POST /v1/audio/speech`).
#[derive(Deserialize, ToSchema)]
pub(crate) struct OpenAISpeechRequest {
    /// Model id; accepted for OpenAI compatibility (ignored).
    #[allow(dead_code)]
    pub(crate) model: Option<String>,
    /// Text to synthesize.
    pub(crate) input: String,
    /// Preset voice name (`M1`–`M5`, `F1`–`F5`).
    #[serde(default = "default_voice")]
    #[schema(default = "M1")]
    pub(crate) voice: String,
    /// Output container format.
    #[serde(default = "default_format")]
    #[schema(default = "wav")]
    pub(crate) response_format: String,
    /// Playback speed; clamped to 0.7–2.0.
    #[serde(default = "default_openai_speed")]
    #[schema(default = 1.0)]
    pub(crate) speed: f32,
    /// ISO language code, or `na` for auto.
    #[serde(default)]
    pub(crate) lang: Option<String>,
}

/// Native synthesis request with the full parameter set (`POST /v1/tts`).
#[derive(Deserialize, ToSchema)]
pub(crate) struct TtsRequest {
    /// Text to synthesize.
    pub(crate) text: String,
    #[serde(default = "default_voice")]
    #[schema(default = "M1")]
    pub(crate) voice: String,
    /// ISO language code, or `na` for auto.
    #[serde(default)]
    pub(crate) lang: Option<String>,
    #[serde(default = "default_speed")]
    #[schema(default = 1.05, minimum = 0.7, maximum = 2.0)]
    pub(crate) speed: f32,
    /// Quality/iteration steps (useful range 5–12).
    #[serde(default = "default_steps")]
    #[schema(default = 8, minimum = 1, maximum = 100)]
    pub(crate) steps: usize,
    /// Silence inserted between chunks (sentences/pieces), in seconds; no effect
    /// on text short enough to fit a single chunk.
    #[serde(default = "default_silence")]
    #[schema(default = 0.3)]
    pub(crate) silence_duration: f32,
    /// Target chunk size in bytes, best-effort (multibyte scripts like Korean
    /// fit fewer characters per chunk; an unbreakable token can exceed it).
    #[serde(default)]
    #[schema(minimum = 1, maximum = 10000)]
    pub(crate) max_chunk_length: Option<usize>,
    #[serde(default = "default_format")]
    #[schema(default = "wav")]
    pub(crate) response_format: String,
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct BatchItem {
    pub(crate) text: String,
    #[serde(default = "default_voice")]
    #[schema(default = "M1")]
    pub(crate) voice: String,
    #[serde(default)]
    pub(crate) lang: Option<String>,
    #[serde(default = "default_speed")]
    #[schema(default = 1.05)]
    pub(crate) speed: f32,
    #[serde(default = "default_steps")]
    #[schema(default = 8)]
    pub(crate) steps: usize,
    #[serde(default = "default_silence")]
    #[schema(default = 0.3)]
    pub(crate) silence_duration: f32,
    #[serde(default)]
    pub(crate) max_chunk_length: Option<usize>,
}

/// Batch synthesis request (`POST /v1/tts/batch`).
#[derive(Deserialize, ToSchema)]
pub(crate) struct BatchRequest {
    pub(crate) items: Vec<BatchItem>,
    #[serde(default = "default_format")]
    #[schema(default = "wav")]
    pub(crate) response_format: String,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct BatchResultItem {
    pub(crate) index: usize,
    /// Base64-encoded audio for this item.
    pub(crate) audio_base64: String,
    pub(crate) duration_seconds: f32,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct BatchResponse {
    pub(crate) sample_rate: i32,
    pub(crate) response_format: String,
    pub(crate) items: Vec<BatchResultItem>,
}

/// Register a custom voice style (`POST /v1/voices/import`).
#[derive(Deserialize, ToSchema)]
pub(crate) struct StyleImportRequest {
    /// Name to register the voice under.
    pub(crate) name: String,
    /// Voice Builder TTL component: `{data, dims, type}`.
    #[schema(value_type = Object)]
    pub(crate) style_ttl: Value,
    /// Voice Builder DP component: `{data, dims, type}`.
    #[schema(value_type = Object)]
    pub(crate) style_dp: Value,
}
