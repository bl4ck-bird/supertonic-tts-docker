//! Input validation and clamping for one synthesis request, gathered in one
//! place. [`validate`] is the single entry point `Engine::synthesize` calls: it
//! rejects bad text and languages, then clamps the numeric parameters to range.

use std::sync::OnceLock;

use crate::engine::{EngineError, SynthParams};
use crate::helper::is_valid_lang;

/// Request values clamped to the documented ranges. The `#[schema(min/max)]`
/// annotations are documentation-only (utoipa), not enforced at deserialization,
/// so values are clamped here.
pub(crate) struct Clamped {
    pub(crate) total_steps: usize,
    pub(crate) speed: f32,
    pub(crate) silence_duration: f32,
    pub(crate) max_chunk_length: Option<usize>,
}

/// Validate and clamp a synthesis request in one pass: reject empty/oversized
/// text and unknown languages, then clamp the numeric parameters to range.
pub(crate) fn validate(params: &SynthParams) -> Result<Clamped, EngineError> {
    validate_text(&params.text, max_text_bytes())?;
    validate_lang(&params.lang)?;
    Ok(clamp(params))
}

/// Read a positive `usize` from the environment, falling back to `default` when
/// unset, empty, unparseable, or zero. Shared by the input caps.
pub(crate) fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default)
}

/// Default per-utterance input cap in bytes. Synthesized audio dwarfs the input
/// text, so this — not the request body limit — bounds a request's peak memory.
/// Override with `SUPERTONIC_MAX_TEXT_BYTES`.
const DEFAULT_MAX_TEXT_BYTES: usize = 4096;

/// Per-utterance input cap (bytes), read once from the environment.
fn max_text_bytes() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| env_usize("SUPERTONIC_MAX_TEXT_BYTES", DEFAULT_MAX_TEXT_BYTES))
}

fn validate_text(text: &str, max_bytes: usize) -> Result<(), EngineError> {
    if text.trim().is_empty() {
        return Err(EngineError::BadRequest("text must be non-empty".into()));
    }
    if text.len() > max_bytes {
        return Err(EngineError::BadRequest(format!(
            "text too long: {} bytes (max {max_bytes})",
            text.len()
        )));
    }
    Ok(())
}

/// The upstream loader rejects an unknown `lang` deep in inference, which would
/// surface as a masked 500; check it here so a bad code is a clean 400.
fn validate_lang(lang: &str) -> Result<(), EngineError> {
    if is_valid_lang(lang) {
        Ok(())
    } else {
        Err(EngineError::BadRequest(format!("invalid lang: {lang}")))
    }
}

fn clamp(p: &SynthParams) -> Clamped {
    Clamped {
        total_steps: p.total_steps.clamp(1, 100),
        speed: p.speed.clamp(0.7, 2.0),
        silence_duration: p.silence_duration.clamp(0.0, 10.0),
        max_chunk_length: p.max_chunk_length.map(|n| n.clamp(1, 10_000)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(steps: usize, speed: f32, silence: f32, chunk: Option<usize>) -> SynthParams {
        SynthParams {
            text: "x".into(),
            lang: "na".into(),
            voice: "M1".into(),
            total_steps: steps,
            speed,
            silence_duration: silence,
            max_chunk_length: chunk,
        }
    }

    #[test]
    fn clamps_to_documented_ranges() {
        let c = clamp(&params(99_999, 9.0, 99.0, Some(99_999)));
        assert_eq!(c.total_steps, 100);
        assert_eq!(c.speed, 2.0);
        assert_eq!(c.silence_duration, 10.0);
        assert_eq!(c.max_chunk_length, Some(10_000));

        let c = clamp(&params(0, 0.1, -1.0, Some(0)));
        assert_eq!(c.total_steps, 1);
        assert_eq!(c.speed, 0.7);
        assert_eq!(c.silence_duration, 0.0);
        assert_eq!(c.max_chunk_length, Some(1));
    }

    #[test]
    fn leaves_in_range_values_untouched() {
        let c = clamp(&params(8, 1.05, 0.3, None));
        assert_eq!(c.total_steps, 8);
        assert_eq!(c.speed, 1.05);
        assert_eq!(c.silence_duration, 0.3);
        assert_eq!(c.max_chunk_length, None);
    }

    #[test]
    fn validate_text_rejects_empty_and_oversized() {
        let max = 100;
        assert!(validate_text("", max).is_err());
        assert!(validate_text("   ", max).is_err());
        assert!(validate_text(&"x".repeat(max + 1), max).is_err());
        assert!(validate_text("hello", max).is_ok());
        assert!(validate_text(&"x".repeat(max), max).is_ok());
    }

    #[test]
    fn env_usize_falls_back_on_invalid() {
        // Unset -> default.
        assert_eq!(env_usize("SUPERTONIC_DEFINITELY_UNSET_VAR_XYZ", 42), 42);
        // Empty / unparseable / zero -> default; a positive value parses.
        std::env::set_var("SUPERTONIC_TEST_CAP", "");
        assert_eq!(env_usize("SUPERTONIC_TEST_CAP", 42), 42);
        std::env::set_var("SUPERTONIC_TEST_CAP", "abc");
        assert_eq!(env_usize("SUPERTONIC_TEST_CAP", 42), 42);
        std::env::set_var("SUPERTONIC_TEST_CAP", "0");
        assert_eq!(env_usize("SUPERTONIC_TEST_CAP", 42), 42);
        std::env::set_var("SUPERTONIC_TEST_CAP", "7");
        assert_eq!(env_usize("SUPERTONIC_TEST_CAP", 42), 7);
        std::env::remove_var("SUPERTONIC_TEST_CAP");
    }

    #[test]
    fn validate_lang_accepts_known_rejects_unknown() {
        assert!(validate_lang("na").is_ok());
        assert!(validate_lang("ko").is_ok());
        assert!(validate_lang("en").is_ok());
        assert!(validate_lang("xx").is_err());
        assert!(validate_lang("").is_err());
        assert!(validate_lang("EN").is_err()); // case-sensitive, matches upstream
    }

    #[test]
    fn validate_runs_text_then_lang_then_clamps() {
        let mut empty_text = params(8, 1.05, 0.3, None);
        empty_text.text = String::new();
        assert!(validate(&empty_text).is_err());

        let mut bad_lang = params(8, 1.05, 0.3, None);
        bad_lang.lang = "xx".into();
        assert!(validate(&bad_lang).is_err());

        let c = validate(&params(999, 9.0, 99.0, None)).unwrap();
        assert_eq!(c.total_steps, 100);
        assert_eq!(c.speed, 2.0);
    }
}
