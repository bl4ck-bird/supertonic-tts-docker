//! SuperTonic 3 OpenAI-compatible TTS server (Rust / Axum).
//!
//! One binary, two modes: with no subcommand it runs the HTTP server; the CLI
//! subcommands (see [`cli`]) reuse the same core. `helper.rs` is the vendored
//! upstream inference module. Assets (onnx + voice_styles) load from a fixed
//! `/assets` (Docker-only).

mod api;
mod audio;
mod cli;
mod download;
mod engine;
mod validate;
mod voice_store;
// Vendored verbatim from upstream; it carries example-only helpers we don't
// call, and we don't lint its style (clippy::all) — fixes belong upstream.
#[allow(dead_code, clippy::all)]
mod helper;

use std::net::SocketAddr;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;

use clap::Parser;

use cli::{Cli, Command};
use engine::Engine;

fn main() -> ExitCode {
    // Primary thread control is with_intra_threads on each session (see
    // helper::session_builder); the shipped load-dynamic onnxruntime has no
    // OpenMP so OMP_NUM_THREADS alone is ignored. Kept as a fallback for
    // OpenMP-linked runtimes. Must stay here: before any thread spawns or
    // onnxruntime loads.
    if let Ok(threads) = std::env::var("SUPERTONIC_THREADS") {
        if !threads.is_empty() {
            std::env::set_var("OMP_NUM_THREADS", threads);
        }
    }

    match Cli::parse().command {
        None | Some(Command::Serve) => run_server(),
        Some(Command::Synth(args)) => cli::run_synth(args),
        Some(Command::Voices) => cli::run_voices(),
        Some(Command::Import(args)) => cli::run_import(args),
        Some(Command::Healthcheck) => cli::run_healthcheck(),
    }
}

fn fatal(msg: impl std::fmt::Display) -> ExitCode {
    eprintln!("fatal: {msg}");
    ExitCode::FAILURE
}

/// Start the Tokio runtime and serve until terminated or a fatal startup error.
fn run_server() -> ExitCode {
    match tokio::runtime::Runtime::new() {
        Ok(rt) => rt.block_on(serve()),
        Err(e) => fatal(format!("tokio runtime: {e}")),
    }
}

async fn serve() -> ExitCode {
    tracing_subscriber::fmt::init();

    let port: u16 = std::env::var("SUPERTONIC_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    // The server needs both the onnx model and the voice styles present.
    if let Err(e) = download::ensure_assets(Path::new(engine::ASSETS_DIR), &["onnx", "voice_styles"])
    {
        return fatal(format!("asset download: {e}"));
    }

    tracing::info!("loading SuperTonic 3 assets from {}", engine::ASSETS_DIR);
    let engine = match Engine::load(Path::new(engine::ASSETS_DIR), false) {
        Ok(engine) => engine,
        Err(err) => return fatal(err),
    };
    tracing::info!(
        "model loaded: sample_rate={}, voices={:?}",
        engine.sample_rate(),
        engine.list_voices()
    );

    let ffmpeg = audio::ffmpeg_available();
    tracing::info!("ffmpeg available: {ffmpeg} (compressed formats {})",
        if ffmpeg { "enabled" } else { "disabled — wav/pcm only" });

    let state = api::AppState::new(Arc::new(engine), ffmpeg);
    let app = api::router(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) => return fatal(format!("bind {addr}: {e}")),
    };
    tracing::info!("listening on http://{addr}");
    if let Err(e) = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
    {
        return fatal(format!("server error: {e}"));
    }
    ExitCode::SUCCESS
}

/// Resolve when the process receives Ctrl-C or SIGTERM (Docker's stop signal),
/// letting `axum` drain in-flight requests before exit.
async fn shutdown_signal() {
    use tokio::signal;
    let ctrl_c = async {
        let _ = signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut sig) = signal::unix::signal(signal::unix::SignalKind::terminate()) {
            sig.recv().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
    tracing::info!("shutdown signal received; draining in-flight requests");
}
