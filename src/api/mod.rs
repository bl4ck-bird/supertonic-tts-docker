//! HTTP layer: OpenAI-compatible + native endpoints.
//!
//! Split into focused modules:
//! - [`dto`] — request/response payloads and serde defaults.
//! - [`error`] — `AppError`, status mapping, internal-detail masking.
//! - [`handlers`] — synthesis/diagnostics handlers and the (blocking) render pipeline.
//! - [`webui`] — static test-console and Swagger UI serving.
//!
//! The OpenAPI document is generated from the `#[utoipa::path]` / `ToSchema`
//! annotations (no hand-maintained JSON) and served with a Swagger UI page at
//! `/docs`.

mod dto;
mod error;
mod handlers;
mod webui;

use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::DefaultBodyLimit,
    http::StatusCode,
    routing::{get, post},
    Router,
};
use tokio::sync::Semaphore;
use tower_http::timeout::TimeoutLayer;
use utoipa::OpenApi;

use crate::engine::Engine;
use dto::*;
use handlers::*;

/// Max concurrent in-flight synthesis requests; past this the server sheds load
/// with 503. The model is serial (one runs at a time behind a mutex).
const MAX_INFLIGHT_SYNTH: usize = 16;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Engine>,
    /// Whether ffmpeg is available (controls compressed-format support).
    pub ffmpeg: bool,
    /// Backpressure permits for synthesis endpoints (see [`MAX_INFLIGHT_SYNTH`]).
    pub sem: Arc<Semaphore>,
}

impl AppState {
    pub fn new(engine: Arc<Engine>, ffmpeg: bool) -> Self {
        Self {
            engine,
            ffmpeg,
            sem: Arc::new(Semaphore::new(MAX_INFLIGHT_SYNTH)),
        }
    }
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(webui::index))
        .route("/v1/health", get(health))
        .route("/v1/voices", get(list_voices))
        .route("/v1/voices/import", post(import_voice))
        .route("/v1/audio/speech", post(openai_speech))
        .route("/v1/tts", post(native_tts))
        .route("/v1/tts/batch", post(batch_tts))
        .route("/openapi.json", get(openapi_json))
        .route("/docs", get(webui::docs))
        // Reject oversized bodies (413) and time out long requests (408).
        .layer(DefaultBodyLimit::max(16 * 1024 * 1024))
        .layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(300),
        ))
        .with_state(state)
}

// ---- OpenAPI document -------------------------------------------------------

#[derive(OpenApi)]
#[openapi(
    info(
        title = "SuperTonic 3 TTS (Rust)",
        description = "OpenAI-compatible Text-to-Speech for SuperTonic 3."
    ),
    paths(openai_speech, native_tts, batch_tts, list_voices, import_voice, health),
    components(schemas(
        OpenAISpeechRequest,
        TtsRequest,
        BatchItem,
        BatchRequest,
        BatchResultItem,
        BatchResponse,
        StyleImportRequest
    )),
    tags(
        (name = "speech", description = "Speech synthesis."),
        (name = "voices", description = "Voice listing."),
        (name = "system", description = "Health and diagnostics.")
    )
)]
pub(crate) struct ApiDoc;
