//! Static web asset serving: the test console (`/`) and Swagger UI (`/docs`).
//! The only filesystem-reading endpoints; each HTML file is read from disk once
//! and then served from an in-memory cache.

use std::sync::OnceLock;

use axum::{
    http::StatusCode,
    response::{Html, IntoResponse, Response},
    Json,
};
use serde_json::json;

/// Static web assets directory.
const WEBUI_DIR: &str = "/app/webui";

/// Serve a static HTML asset, caching it in memory after the first successful
/// read so repeat requests don't hit the disk. A read failure is not cached, so
/// a file that appears later is still picked up.
fn serve_webui(cache: &OnceLock<String>, file: &str) -> Response {
    if let Some(html) = cache.get() {
        return Html(html.clone()).into_response();
    }
    match std::fs::read_to_string(std::path::Path::new(WEBUI_DIR).join(file)) {
        Ok(html) => {
            let html = cache.get_or_init(|| html);
            Html(html.clone()).into_response()
        }
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "detail": format!("{file} not found") })),
        )
            .into_response(),
    }
}

pub(crate) async fn index() -> Response {
    static INDEX_HTML: OnceLock<String> = OnceLock::new();
    serve_webui(&INDEX_HTML, "index.html")
}

pub(crate) async fn docs() -> Response {
    static DOCS_HTML: OnceLock<String> = OnceLock::new();
    serve_webui(&DOCS_HTML, "docs.html")
}
