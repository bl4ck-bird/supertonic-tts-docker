//! Request handlers and the synth→encode render pipeline. (Static web assets
//! are served from [`super::webui`].)
//!
//! Synthesis (ONNX inference) and ffmpeg encoding are blocking, CPU-bound work;
//! they run on `tokio::task::spawn_blocking` so they never stall the async
//! runtime's worker threads.

use std::sync::Arc;

use axum::{
    body::Body,
    extract::State,
    http::header,
    response::Response,
    Json,
};
use base64::Engine as _;
use serde_json::{json, Value};
use utoipa::OpenApi;

use crate::audio;
use crate::engine::{Engine, SynthParams};

use super::dto::*;
use super::error::AppError;
use super::{ApiDoc, AppState};

// ---- rendering --------------------------------------------------------------

struct Rendered {
    bytes: Vec<u8>,
    media: &'static str,
    duration: f32,
}

/// Blocking: synthesize then encode. Call via [`render_async`] from handlers.
fn render(engine: &Engine, params: SynthParams, response_format: &str) -> Result<Rendered, AppError> {
    let (samples, duration) = engine.synthesize(&params)?;
    let (bytes, media) = audio::encode(&samples, engine.sample_rate(), response_format)?;
    Ok(Rendered {
        bytes,
        media,
        duration,
    })
}

/// Run the blocking render pipeline off the async runtime. The permit moves into
/// the blocking closure, so it is held until the work finishes — not freed when
/// the handler future is dropped (e.g. by the request timeout).
async fn render_async(
    engine: Arc<Engine>,
    params: SynthParams,
    response_format: String,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Result<Rendered, AppError> {
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        render(&engine, params, &response_format)
    })
    .await
    .map_err(|e| AppError::from_join("synthesis task failed", e))?
}

fn audio_response(rendered: Rendered, with_duration: bool) -> Response {
    let mut builder = Response::builder().header(header::CONTENT_TYPE, rendered.media);
    if with_duration {
        builder = builder.header("X-Audio-Duration", format!("{:.3}", rendered.duration));
    }
    builder.body(Body::from(rendered.bytes)).unwrap()
}

fn lang_or_auto(lang: Option<String>) -> String {
    lang.unwrap_or_else(|| "na".to_string())
}

/// Take a synthesis permit, or fail fast with 503 if the server is saturated.
/// Hold the returned permit until the blocking work finishes (it releases on
/// drop), bounding how many requests pile up on the serial model.
fn acquire(state: &AppState) -> Result<tokio::sync::OwnedSemaphorePermit, AppError> {
    state.sem.clone().try_acquire_owned().map_err(|_| AppError::busy())
}

/// Maximum items in one batch request (each is a full serialized synthesis).
const MAX_BATCH_ITEMS: usize = 100;

/// Default aggregate cap on total input text across one batch. A batch holds a
/// single synthesis permit and accumulates every item's audio in memory before
/// responding, so this bounds the whole request's serial work and peak memory
/// (audio is ~10,000x the text). Override with `SUPERTONIC_MAX_BATCH_TEXT_BYTES`.
const DEFAULT_MAX_BATCH_TEXT_BYTES: usize = 50_000;

/// Aggregate batch input cap (bytes), read once from the environment.
fn max_batch_text_bytes() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::validate::env_usize("SUPERTONIC_MAX_BATCH_TEXT_BYTES", DEFAULT_MAX_BATCH_TEXT_BYTES))
}

fn check_batch_size(n: usize) -> Result<(), AppError> {
    if n > MAX_BATCH_ITEMS {
        return Err(AppError::bad_request(format!(
            "batch too large: {n} items (max {MAX_BATCH_ITEMS})"
        )));
    }
    Ok(())
}

fn check_batch_text(total_bytes: usize, max_bytes: usize) -> Result<(), AppError> {
    if total_bytes > max_bytes {
        return Err(AppError::bad_request(format!(
            "batch text too large: {total_bytes} bytes (max {max_bytes})"
        )));
    }
    Ok(())
}

// ---- handlers ---------------------------------------------------------------

#[utoipa::path(
    get, path = "/v1/health", tag = "system",
    responses((status = 200, description = "Service health and supported formats"))
)]
pub(crate) async fn health(State(state): State<AppState>) -> Json<Value> {
    // Reads the voice dir live, so a runtime import is reflected immediately.
    Json(json!({
        "status": "ok",
        "model_loaded": true,
        "sample_rate": state.engine.sample_rate(),
        "voices": state.engine.list_voices().len(),
        "ffmpeg": state.ffmpeg,
        "formats": audio::available_formats(state.ffmpeg),
    }))
}

#[utoipa::path(
    get, path = "/v1/voices", tag = "voices",
    responses((status = 200, description = "Available voice names"))
)]
pub(crate) async fn list_voices(State(state): State<AppState>) -> Json<Value> {
    Json(json!({ "voices": state.engine.list_voices() }))
}

#[utoipa::path(
    post, path = "/v1/voices/import", tag = "voices",
    request_body = StyleImportRequest,
    responses(
        (status = 200, description = "Voice registered; returns the updated voice list"),
        (status = 400, description = "Invalid name or style document"),
    )
)]
pub(crate) async fn import_voice(
    State(state): State<AppState>,
    Json(req): Json<StyleImportRequest>,
) -> Result<Json<Value>, AppError> {
    // Serializing the style doc and loading it writes to disk and parses a
    // possibly-large JSON, so run it off the async runtime's worker threads.
    let engine = state.engine.clone();
    let name = req.name.clone();
    let voices = tokio::task::spawn_blocking(move || -> Result<Vec<String>, AppError> {
        let doc = json!({ "style_ttl": req.style_ttl, "style_dp": req.style_dp });
        let json_str =
            serde_json::to_string(&doc).map_err(|e| AppError::bad_request(e.to_string()))?;
        engine.import_voice(&req.name, &json_str)?;
        Ok(engine.list_voices())
    })
    .await
    .map_err(|e| AppError::from_join("voice import task failed", e))??;
    Ok(Json(json!({ "name": name, "voices": voices })))
}

#[utoipa::path(
    post, path = "/v1/audio/speech", tag = "speech",
    request_body = OpenAISpeechRequest,
    responses(
        (status = 200, description = "Synthesized audio (format per response_format)"),
        (status = 400, description = "Unknown voice or unsupported format"),
        (status = 501, description = "Compressed format requested but ffmpeg unavailable"),
    )
)]
pub(crate) async fn openai_speech(
    State(state): State<AppState>,
    Json(req): Json<OpenAISpeechRequest>,
) -> Result<Response, AppError> {
    let permit = acquire(&state)?;
    let params = SynthParams {
        text: req.input,
        lang: lang_or_auto(req.lang),
        voice: req.voice,
        total_steps: default_steps(),
        speed: req.speed,
        silence_duration: default_silence(),
        max_chunk_length: None,
    };
    let rendered = render_async(state.engine.clone(), params, req.response_format, permit).await?;
    Ok(audio_response(rendered, false))
}

#[utoipa::path(
    post, path = "/v1/tts", tag = "speech",
    request_body = TtsRequest,
    responses(
        (status = 200, description = "Synthesized audio (format per response_format)"),
        (status = 400, description = "Unknown voice or unsupported format"),
        (status = 501, description = "Compressed format requested but ffmpeg unavailable"),
    )
)]
pub(crate) async fn native_tts(
    State(state): State<AppState>,
    Json(req): Json<TtsRequest>,
) -> Result<Response, AppError> {
    let permit = acquire(&state)?;
    let params = SynthParams {
        text: req.text,
        lang: lang_or_auto(req.lang),
        voice: req.voice,
        total_steps: req.steps,
        speed: req.speed,
        silence_duration: req.silence_duration,
        max_chunk_length: req.max_chunk_length,
    };
    let rendered = render_async(state.engine.clone(), params, req.response_format, permit).await?;
    Ok(audio_response(rendered, true))
}

#[utoipa::path(
    post, path = "/v1/tts/batch", tag = "speech",
    request_body = BatchRequest,
    responses((status = 200, description = "Batch results", body = BatchResponse))
)]
pub(crate) async fn batch_tts(
    State(state): State<AppState>,
    Json(req): Json<BatchRequest>,
) -> Result<Json<BatchResponse>, AppError> {
    check_batch_size(req.items.len())?;
    check_batch_text(
        req.items.iter().map(|i| i.text.len()).sum(),
        max_batch_text_bytes(),
    )?;
    let permit = acquire(&state)?;

    let engine = state.engine.clone();
    let response_format = req.response_format.clone();
    let items = req.items;

    // The batch loop is blocking (serialized inference + base64); run it once on
    // a blocking thread. The permit moves in so the slot stays held until the
    // whole batch finishes, even if the request times out.
    let result_items = tokio::task::spawn_blocking(move || -> Result<Vec<BatchResultItem>, AppError> {
        let _permit = permit;
        let b64 = base64::engine::general_purpose::STANDARD;
        let mut out = Vec::with_capacity(items.len());
        for (index, item) in items.into_iter().enumerate() {
            let params = SynthParams {
                text: item.text,
                lang: lang_or_auto(item.lang),
                voice: item.voice,
                total_steps: item.steps,
                speed: item.speed,
                silence_duration: item.silence_duration,
                max_chunk_length: item.max_chunk_length,
            };
            let rendered = render(&engine, params, &response_format)?;
            out.push(BatchResultItem {
                index,
                audio_base64: b64.encode(&rendered.bytes),
                duration_seconds: rendered.duration,
            });
        }
        Ok(out)
    })
    .await
    .map_err(|e| AppError::from_join("batch synthesis task failed", e))??;

    Ok(Json(BatchResponse {
        sample_rate: state.engine.sample_rate(),
        response_format: req.response_format,
        items: result_items,
    }))
}

pub(crate) async fn openapi_json() -> Json<utoipa::openapi::OpenApi> {
    Json(ApiDoc::openapi())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_size_within_limit_is_ok() {
        assert!(check_batch_size(0).is_ok());
        assert!(check_batch_size(MAX_BATCH_ITEMS).is_ok());
    }

    #[test]
    fn batch_size_over_limit_is_rejected() {
        assert!(check_batch_size(MAX_BATCH_ITEMS + 1).is_err());
    }

    #[test]
    fn batch_text_within_limit_is_ok() {
        assert!(check_batch_text(0, 1000).is_ok());
        assert!(check_batch_text(1000, 1000).is_ok());
    }

    #[test]
    fn batch_text_over_limit_is_rejected() {
        assert!(check_batch_text(1001, 1000).is_err());
    }
}
