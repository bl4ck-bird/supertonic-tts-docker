//! HTTP error type. Maps domain/audio errors to status codes and keeps internal
//! failure detail out of client responses: `Internal` variants are logged to
//! stderr and returned to the client as a generic message, so paths and other
//! internals are not leaked.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

use crate::audio::AudioError;
use crate::engine::EngineError;

pub(crate) struct AppError {
    status: StatusCode,
    /// Client-facing message. Never contains internal detail for 5xx errors.
    detail: String,
}

impl AppError {
    pub(crate) fn bad_request(detail: impl Into<String>) -> Self {
        AppError {
            status: StatusCode::BAD_REQUEST,
            detail: detail.into(),
        }
    }

    /// 503 backpressure: too many synthesis requests are already in flight.
    pub(crate) fn busy() -> Self {
        AppError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            detail: "server busy: too many concurrent requests".to_string(),
        }
    }

    /// A masked 500: the real cause is logged to stderr, the client sees a
    /// generic message.
    fn internal(context: &str, detail: impl std::fmt::Display) -> Self {
        tracing::error!("{context}: {detail}");
        AppError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            detail: "internal error".to_string(),
        }
    }

    /// Build an error from a `spawn_blocking` join failure (worker panic). The
    /// `context` names the task so a panic is triaged to the right endpoint.
    pub(crate) fn from_join(context: &str, e: tokio::task::JoinError) -> Self {
        AppError::internal(context, e)
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (self.status, Json(json!({ "detail": self.detail }))).into_response()
    }
}

impl From<EngineError> for AppError {
    fn from(e: EngineError) -> Self {
        match e {
            EngineError::UnknownVoice(_) | EngineError::BadRequest(_) => AppError {
                status: StatusCode::BAD_REQUEST,
                detail: e.to_string(),
            },
            EngineError::Internal(m) => AppError::internal("engine error", m),
        }
    }
}

impl From<AudioError> for AppError {
    fn from(e: AudioError) -> Self {
        match e {
            AudioError::Unsupported(_) => AppError {
                status: StatusCode::BAD_REQUEST,
                detail: e.to_string(),
            },
            AudioError::FfmpegMissing => AppError {
                status: StatusCode::NOT_IMPLEMENTED,
                detail: e.to_string(),
            },
            AudioError::Internal(m) => AppError::internal("audio error", m),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_errors_map_to_status() {
        let e: AppError = EngineError::UnknownVoice("zz".into()).into();
        assert_eq!(e.status, StatusCode::BAD_REQUEST);
        let e: AppError = EngineError::BadRequest("bad".into()).into();
        assert_eq!(e.status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn audio_errors_map_to_status() {
        let e: AppError = AudioError::Unsupported("xyz".into()).into();
        assert_eq!(e.status, StatusCode::BAD_REQUEST);
        let e: AppError = AudioError::FfmpegMissing.into();
        assert_eq!(e.status, StatusCode::NOT_IMPLEMENTED);
    }

    #[test]
    fn internal_detail_is_masked_from_client() {
        let e: AppError = EngineError::Internal("/secret/path/model.onnx".into()).into();
        assert_eq!(e.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(e.detail, "internal error");
    }

    #[test]
    fn client_facing_detail_is_preserved() {
        let e: AppError = EngineError::UnknownVoice("nope".into()).into();
        assert_eq!(e.detail, "unknown voice: nope");
    }

    #[test]
    fn busy_maps_to_503() {
        let e = AppError::busy();
        assert_eq!(e.status, StatusCode::SERVICE_UNAVAILABLE);
    }
}
