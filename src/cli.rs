//! Command-line interface. The same binary runs as an HTTP server when no
//! subcommand is given (see `main`); the subcommands here reuse the synthesis
//! [`Engine`] (for `synth`) and the voice store (for `voices` / `import`).

use std::path::Path;
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};

use crate::audio;
use crate::download;
use crate::engine::{Engine, SynthParams, ASSETS_DIR};
use crate::voice_store::VoiceStore;

#[derive(Parser)]
#[command(
    name = "supertonic",
    version,
    about = "SuperTonic 3 TTS — HTTP server (default) and CLI"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand)]
pub enum Command {
    /// Run the HTTP server (the default when no subcommand is given).
    Serve,
    /// Synthesize one utterance to a file.
    Synth(SynthArgs),
    /// List available preset and custom voices.
    Voices,
    /// Register a custom Voice Builder style from a JSON file.
    Import(ImportArgs),
    /// Probe the local server's health endpoint; backs the Docker HEALTHCHECK.
    /// Exit 0 if healthy, 1 otherwise.
    Healthcheck,
}

#[derive(Args)]
pub struct SynthArgs {
    /// Text to synthesize.
    #[arg(short, long)]
    text: String,
    #[arg(long, default_value = "M1")]
    voice: String,
    /// ISO language code, or `na` for auto.
    #[arg(long, default_value = "na")]
    lang: String,
    #[arg(long, default_value_t = 1.05)]
    speed: f32,
    #[arg(long, default_value_t = 8)]
    steps: usize,
    #[arg(long, default_value_t = 0.3)]
    silence: f32,
    #[arg(long)]
    max_chunk_length: Option<usize>,
    #[arg(long, default_value = "wav")]
    format: String,
    /// Output file path.
    #[arg(short, long, default_value = "out.wav")]
    out: String,
}

#[derive(Args)]
pub struct ImportArgs {
    /// Name to register the voice under.
    #[arg(long)]
    name: String,
    /// Path to a Voice Builder JSON file: `{"style_ttl": {...}, "style_dp": {...}}`.
    #[arg(long)]
    file: String,
}

/// Download any missing assets, then load the full engine (ONNX model) from the
/// fixed assets dir. Only `synth` needs the model; `voices`/`import` operate on
/// the voice dir without loading it.
fn load_engine() -> Result<Engine, String> {
    download::ensure_assets(Path::new(ASSETS_DIR), &["onnx", "voice_styles"])
        .map_err(|e| e.to_string())?;
    Engine::load(Path::new(ASSETS_DIR), false).map_err(|e| e.to_string())
}

/// Ensure only the voice styles are present (no onnx model), for the
/// model-free `voices` / `import` subcommands.
fn ensure_voice_styles() -> Result<(), String> {
    download::ensure_assets(Path::new(ASSETS_DIR), &["voice_styles"]).map_err(|e| e.to_string())
}

fn fail(msg: impl std::fmt::Display) -> ExitCode {
    eprintln!("error: {msg}");
    ExitCode::FAILURE
}

pub fn run_synth(args: SynthArgs) -> ExitCode {
    let engine = match load_engine() {
        Ok(e) => e,
        Err(e) => return fail(e),
    };
    let params = SynthParams {
        text: args.text,
        lang: args.lang,
        voice: args.voice,
        total_steps: args.steps,
        speed: args.speed,
        silence_duration: args.silence,
        max_chunk_length: args.max_chunk_length,
    };
    let (samples, duration) = match engine.synthesize(&params) {
        Ok(out) => out,
        Err(e) => return fail(e),
    };
    let (bytes, _media) = match audio::encode(&samples, engine.sample_rate(), &args.format) {
        Ok(out) => out,
        Err(e) => return fail(e),
    };

    if let Err(e) = std::fs::write(&args.out, &bytes) {
        return fail(e);
    }
    eprintln!("wrote {} ({duration:.3}s audio)", args.out);
    ExitCode::SUCCESS
}

pub fn run_voices() -> ExitCode {
    if let Err(e) = ensure_voice_styles() {
        return fail(e);
    }
    let voice_dir = VoiceStore::dir(Path::new(ASSETS_DIR));
    for v in VoiceStore::list_in(&voice_dir) {
        println!("{v}");
    }
    ExitCode::SUCCESS
}

pub fn run_import(args: ImportArgs) -> ExitCode {
    if let Err(e) = ensure_voice_styles() {
        return fail(e);
    }
    let json = match std::fs::read_to_string(&args.file) {
        Ok(s) => s,
        Err(e) => return fail(e),
    };
    let voice_dir = VoiceStore::dir(Path::new(ASSETS_DIR));
    match VoiceStore::import_to(&voice_dir, &args.name, &json) {
        Ok(()) => {
            eprintln!("registered voice: {}", args.name);
            ExitCode::SUCCESS
        }
        Err(e) => fail(e),
    }
}

/// Probe `http://127.0.0.1:<port>/v1/health`, the Docker HEALTHCHECK command.
/// Exit 0 only on a 2xx response.
pub fn run_healthcheck() -> ExitCode {
    let port = crate::validate::env_port("SUPERTONIC_PORT", 8080);
    let url = format!("http://127.0.0.1:{port}/v1/health");
    match ureq::get(&url).call() {
        Ok(resp) if resp.status().is_success() => ExitCode::SUCCESS,
        _ => ExitCode::FAILURE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
        Cli::try_parse_from(args)
    }

    #[test]
    fn no_subcommand_means_server() {
        assert!(parse(&["supertonic"]).unwrap().command.is_none());
    }

    #[test]
    fn bare_subcommands_parse() {
        assert!(matches!(
            parse(&["supertonic", "serve"]).unwrap().command,
            Some(Command::Serve)
        ));
        assert!(matches!(
            parse(&["supertonic", "voices"]).unwrap().command,
            Some(Command::Voices)
        ));
        assert!(matches!(
            parse(&["supertonic", "healthcheck"]).unwrap().command,
            Some(Command::Healthcheck)
        ));
    }

    #[test]
    fn synth_applies_defaults() {
        let cli = parse(&["supertonic", "synth", "--text", "hi"]).unwrap();
        let Some(Command::Synth(a)) = cli.command else {
            panic!("expected synth");
        };
        assert_eq!(a.text, "hi");
        assert_eq!(a.voice, "M1");
        assert_eq!(a.lang, "na");
        assert_eq!(a.steps, 8);
        assert_eq!(a.silence, 0.3);
        assert_eq!(a.format, "wav");
        assert_eq!(a.out, "out.wav");
        assert_eq!(a.max_chunk_length, None);
    }

    #[test]
    fn synth_applies_overrides() {
        let cli = parse(&[
            "supertonic",
            "synth",
            "--text",
            "yo",
            "--voice",
            "F2",
            "--lang",
            "ko",
            "--speed",
            "1.2",
            "--steps",
            "12",
            "--silence",
            "0.5",
            "--format",
            "pcm",
            "--out",
            "x.pcm",
            "--max-chunk-length",
            "300",
        ])
        .unwrap();
        let Some(Command::Synth(a)) = cli.command else {
            panic!("expected synth");
        };
        assert_eq!(a.voice, "F2");
        assert_eq!(a.lang, "ko");
        assert_eq!(a.speed, 1.2);
        assert_eq!(a.steps, 12);
        assert_eq!(a.silence, 0.5);
        assert_eq!(a.format, "pcm");
        assert_eq!(a.out, "x.pcm");
        assert_eq!(a.max_chunk_length, Some(300));
    }

    #[test]
    fn synth_requires_text() {
        assert!(parse(&["supertonic", "synth"]).is_err());
    }

    #[test]
    fn import_parses_name_and_file() {
        let cli = parse(&["supertonic", "import", "--name", "foo", "--file", "f.json"]).unwrap();
        let Some(Command::Import(a)) = cli.command else {
            panic!("expected import");
        };
        assert_eq!(a.name, "foo");
        assert_eq!(a.file, "f.json");
    }

    #[test]
    fn import_requires_name_and_file() {
        assert!(parse(&["supertonic", "import", "--name", "foo"]).is_err());
        assert!(parse(&["supertonic", "import", "--file", "f.json"]).is_err());
    }
}
