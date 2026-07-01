//! Voice-style repository: lists, resolves (with caching), and imports custom
//! Voice Builder styles under a `voice_styles/` directory. Independent of the
//! synthesis model, so the CLI can list and import without loading ONNX.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::Value;

use crate::engine::{lock, EngineError};
use crate::helper::{load_voice_style, Style};

/// Per-process counter for unique import temp filenames.
static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Serializes the check-then-publish in [`VoiceStore::write_and_load`] so two
/// concurrent imports of the same new name cannot both create it.
static IMPORT_LOCK: Mutex<()> = Mutex::new(());

/// Per-tensor scalar cap for an imported style. The upstream loader allocates a
/// `dims[1] * dims[2]` buffer from the declared dimensions, so an unvalidated
/// `dims` like `[1, 100000, 100000]` would request tens of GB. The preset
/// styles are ~12.8k elements, so this is generous while bounding the alloc.
const MAX_STYLE_ELEMS: usize = 10_000_000;

/// A voice name maps to `<dir>/<name>.json`, so restrict it to a conservative
/// charset: ASCII alphanumerics plus `.`, `_`, `-`, no leading dot, and no `..`.
/// This blocks directory escape (no `/`, `\`, `..`) and also rejects odd inputs
/// (empty, whitespace, control chars, leading-dot hidden files).
fn is_safe_voice_name(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('.')
        && !name.contains("..")
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

/// Count the leaf numbers in a (possibly nested) JSON value.
fn count_scalars(v: &Value) -> usize {
    match v {
        Value::Array(a) => a.iter().map(count_scalars).sum(),
        Value::Number(_) => 1,
        _ => 0,
    }
}

/// Validate one `{dims, data, ...}` style component the way the upstream loader
/// uses it: it indexes `dims[1]`/`dims[2]` and writes one `data` scalar per
/// element into a `dims[0]*dims[1]*dims[2]` buffer. Without this guard a
/// malformed document reaches the loader and either panics (short `dims`, or
/// more `data` values than the buffer) or requests an enormous allocation.
fn validate_style_component(doc: &Value, which: &str) -> Result<(), EngineError> {
    let bad = |m: String| EngineError::BadRequest(m);
    let comp = doc
        .get(which)
        .ok_or_else(|| bad(format!("missing {which}")))?;
    let dims = comp
        .get("dims")
        .and_then(Value::as_array)
        .ok_or_else(|| bad(format!("{which}.dims must be an array")))?;
    if dims.len() != 3 {
        return Err(bad(format!(
            "{which}.dims must have 3 elements, got {}",
            dims.len()
        )));
    }
    let mut d = [0usize; 3];
    for (i, x) in dims.iter().enumerate() {
        d[i] = x
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| bad(format!("{which}.dims[{i}] must be a non-negative integer")))?;
    }
    if d[0] != 1 {
        return Err(bad(format!("{which}.dims[0] must be 1, got {}", d[0])));
    }
    let elems = d[0]
        .checked_mul(d[1])
        .and_then(|p| p.checked_mul(d[2]))
        .ok_or_else(|| bad(format!("{which}.dims product overflows")))?;
    if elems == 0 || elems > MAX_STYLE_ELEMS {
        return Err(bad(format!(
            "{which}.dims product {elems} out of range (1..={MAX_STYLE_ELEMS})"
        )));
    }
    // The loader writes one buffer slot per `data` scalar; more than `elems`
    // indexes out of bounds. Require an exact match for a well-formed tensor.
    let count = comp.get("data").map(count_scalars).unwrap_or(0);
    if count != elems {
        return Err(bad(format!(
            "{which}.data has {count} values but dims expect {elems}"
        )));
    }
    Ok(())
}

/// Validate a Voice Builder document (`{style_ttl, style_dp}`) before it reaches
/// the upstream loader.
fn validate_style_doc(json: &str) -> Result<(), EngineError> {
    let doc: Value = serde_json::from_str(json)
        .map_err(|e| EngineError::BadRequest(format!("invalid voice style json: {e}")))?;
    validate_style_component(&doc, "style_ttl")?;
    validate_style_component(&doc, "style_dp")?;
    Ok(())
}

/// Stores and caches voice styles loaded from `<dir>/<name>.json`.
pub struct VoiceStore {
    dir: PathBuf,
    cache: Mutex<HashMap<String, Arc<Style>>>,
}

impl VoiceStore {
    pub fn new(dir: PathBuf) -> Self {
        Self {
            dir,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// The `voice_styles/` directory under an assets root.
    pub fn dir(assets_dir: &Path) -> PathBuf {
        assets_dir.join("voice_styles")
    }

    /// Voice names from the `*.json` files in a `voice_styles/` dir. Associated
    /// (no cache), so the CLI can list without a running store.
    pub fn list_in(dir: &Path) -> Vec<String> {
        let mut names = Vec::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("json") {
                    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                        names.push(stem.to_string());
                    }
                }
            }
        }
        names.sort();
        names
    }

    pub fn list(&self) -> Vec<String> {
        Self::list_in(&self.dir)
    }

    /// Resolve a voice to its loaded style, caching the result.
    pub fn resolve(&self, voice: &str) -> Result<Arc<Style>, EngineError> {
        // The cache only ever holds names already validated by this function or
        // by `import`, so a hit can skip the safety check below.
        if let Some(style) = lock(&self.cache).get(voice) {
            return Ok(style.clone());
        }
        if !is_safe_voice_name(voice) {
            return Err(EngineError::UnknownVoice(voice.to_string()));
        }
        let path = self.dir.join(format!("{voice}.json"));
        if !path.is_file() {
            return Err(EngineError::UnknownVoice(voice.to_string()));
        }
        let style = load_voice_style(&[path.to_string_lossy().into_owned()], false)
            .map_err(|e| EngineError::Internal(format!("failed to load voice {voice}: {e}")))?;
        let arc = Arc::new(style);
        lock(&self.cache).insert(voice.to_string(), arc.clone());
        Ok(arc)
    }

    /// Register a custom voice and cache it so it is immediately usable.
    /// Refuses to overwrite an existing voice (see [`Self::write_and_load`]).
    pub fn import(&self, name: &str, json: &str) -> Result<(), EngineError> {
        let style = Self::write_and_load(&self.dir, name, json)?;
        lock(&self.cache).insert(name.to_string(), Arc::new(style));
        Ok(())
    }

    /// Register a custom voice into a dir without a cache (one-shot CLI use).
    /// The document is `{"style_ttl": {...}, "style_dp": {...}}`.
    pub fn import_to(dir: &Path, name: &str, json: &str) -> Result<(), EngineError> {
        Self::write_and_load(dir, name, json).map(|_| ())
    }

    /// Validate a Voice Builder JSON by loading it from a temp file, then
    /// atomically rename it to `<dir>/<name>.json`. Invalid input leaves any
    /// existing voice of that name untouched. Returns the loaded `Style`.
    fn write_and_load(dir: &Path, name: &str, json: &str) -> Result<Style, EngineError> {
        if !is_safe_voice_name(name) {
            return Err(EngineError::BadRequest(format!(
                "invalid voice name: {name}"
            )));
        }
        // In-process guard against a same-name import race; cross-process callers
        // rely on the single-instance assumption.
        let _guard = lock(&IMPORT_LOCK);
        // Never overwrite an existing voice (preset or prior import); delete its
        // file to replace it. Same rule for the server and the CLI.
        let path = dir.join(format!("{name}.json"));
        if path.exists() {
            return Err(EngineError::BadRequest(format!(
                "voice already exists: {name} (delete it to replace)"
            )));
        }
        // Reject malformed shapes before the upstream loader can panic or request
        // an enormous allocation from attacker-controlled `dims`.
        validate_style_doc(json)?;
        std::fs::create_dir_all(dir)
            .map_err(|e| EngineError::Internal(format!("failed to create voice dir: {e}")))?;
        // Unique temp per import so concurrent imports of the same name cannot
        // corrupt each other's in-flight file before the atomic rename.
        let n = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let tmp = dir.join(format!("{name}.json.part.{}.{n}", std::process::id()));
        std::fs::write(&tmp, json)
            .map_err(|e| EngineError::Internal(format!("failed to write voice file: {e}")))?;
        match load_voice_style(&[tmp.to_string_lossy().into_owned()], false) {
            Ok(style) => {
                std::fs::rename(&tmp, &path).map_err(|e| {
                    EngineError::Internal(format!("failed to save voice file: {e}"))
                })?;
                Ok(style)
            }
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                Err(EngineError::BadRequest(format!("invalid voice style: {e}")))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_path_traversal_voice_names() {
        for bad in ["", "..", "../etc/passwd", "a/b", "a\\b", "foo/..", "..\\x"] {
            assert!(!is_safe_voice_name(bad), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn rejects_odd_but_non_traversing_names() {
        for bad in [
            ".", ".hidden", " ", "a b", "a\nb", "voice!", "naïve", "a\0b",
        ] {
            assert!(!is_safe_voice_name(bad), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn accepts_plain_voice_names() {
        for ok in ["M1", "F5", "my_voice", "voice.2", "a-b"] {
            assert!(is_safe_voice_name(ok), "{ok:?} should be accepted");
        }
    }

    fn style_doc(ttl_dims: &str, ttl_data: &str, dp_dims: &str, dp_data: &str) -> String {
        format!(
            r#"{{"style_ttl":{{"dims":{ttl_dims},"data":{ttl_data}}},
                "style_dp":{{"dims":{dp_dims},"data":{dp_data}}}}}"#
        )
    }

    #[test]
    fn validate_style_doc_accepts_well_formed() {
        // dims [1,1,2] -> 2 scalars; dims [1,2,1] -> 2 scalars.
        let doc = style_doc("[1,1,2]", "[[[0.1,0.2]]]", "[1,2,1]", "[[[0.3],[0.4]]]");
        assert!(validate_style_doc(&doc).is_ok());
    }

    #[test]
    fn validate_style_doc_rejects_short_dims() {
        // dims length < 3 would panic the loader at dims[1]/dims[2].
        let doc = style_doc("[1]", "[0.1]", "[1,1,1]", "[[[0.1]]]");
        assert!(validate_style_doc(&doc).is_err());
    }

    #[test]
    fn validate_style_doc_rejects_oversized_dims() {
        // Would allocate ~10^10 f32 in the loader.
        let doc = style_doc("[1,100000,100000]", "[]", "[1,1,1]", "[[[0.1]]]");
        assert!(validate_style_doc(&doc).is_err());
    }

    #[test]
    fn validate_style_doc_rejects_data_dims_mismatch() {
        // dims expect 1 scalar but data carries 3 -> loader writes out of bounds.
        let doc = style_doc("[1,1,1]", "[[[0.1,0.2,0.3]]]", "[1,1,1]", "[[[0.1]]]");
        assert!(validate_style_doc(&doc).is_err());
    }

    #[test]
    fn validate_style_doc_rejects_nonunit_batch() {
        let doc = style_doc("[2,1,1]", "[[[0.1]],[[0.2]]]", "[1,1,1]", "[[[0.1]]]");
        assert!(validate_style_doc(&doc).is_err());
    }

    #[test]
    fn import_rejects_malformed_shape_without_writing() {
        let dir = std::env::temp_dir().join(format!("st-shape-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let doc = style_doc("[1]", "[0.1]", "[1,1,1]", "[[[0.1]]]");
        let res = VoiceStore::import_to(&dir, "bad", &doc);
        assert!(res.is_err(), "short dims must be rejected");
        assert!(!dir.join("bad.json").exists(), "no file should be written");
        // No temp left behind either.
        let leftovers: Vec<_> = std::fs::read_dir(&dir).unwrap().flatten().collect();
        assert!(
            leftovers.is_empty(),
            "no .part temp should leak: {leftovers:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A loadable `{style_ttl, style_dp}` doc with the `type` field upstream
    /// `load_voice_style` requires. dims [1,1,2]/[1,2,1] -> 2 scalars each.
    fn loadable_style_doc() -> String {
        r#"{"style_ttl":{"dims":[1,1,2],"data":[[[0.1,0.2]]],"type":"float32"},
            "style_dp":{"dims":[1,2,1],"data":[[[0.3],[0.4]]],"type":"float32"}}"#
            .to_string()
    }

    #[test]
    fn import_refuses_to_overwrite_existing_voice() {
        let dir = std::env::temp_dir().join(format!("st-existing-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // An existing voice — a preset, or one imported in a previous run.
        let existing = dir.join("M1.json");
        std::fs::write(&existing, "EXISTING").unwrap();

        let store = VoiceStore::new(dir.clone());
        let doc = loadable_style_doc();

        // Both the server (`import`) and the CLI (`import_to`) refuse to overwrite
        // it, leaving the file untouched.
        assert!(
            store.import("M1", &doc).is_err(),
            "server must not overwrite"
        );
        assert!(
            VoiceStore::import_to(&dir, "M1", &doc).is_err(),
            "CLI must not overwrite"
        );
        assert_eq!(std::fs::read_to_string(&existing).unwrap(), "EXISTING");

        // A new name imports; re-importing it (now existing) is then refused too —
        // every already-present voice is protected, not just presets.
        assert!(
            store.import("custom1", &doc).is_ok(),
            "new voice should import"
        );
        assert!(
            store.import("custom1", &doc).is_err(),
            "existing custom is protected"
        );
        assert!(
            VoiceStore::import_to(&dir, "custom1", &doc).is_err(),
            "CLI sees it too"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn concurrent_import_of_same_new_name_creates_once() {
        let dir = std::env::temp_dir().join(format!("st-race-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Many threads import the same brand-new name at once. Exactly one must
        // win; the rest must see it as already existing.
        let store = Arc::new(VoiceStore::new(dir.clone()));
        let doc = loadable_style_doc();
        let mut handles = Vec::new();
        for _ in 0..8 {
            let s = store.clone();
            let d = doc.clone();
            handles.push(std::thread::spawn(move || s.import("racer", &d).is_ok()));
        }
        let wins = handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .filter(|&ok| ok)
            .count();
        assert_eq!(wins, 1, "exactly one concurrent import should succeed");
        assert!(dir.join("racer.json").is_file());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn failed_import_does_not_clobber_existing_voice() {
        let dir = std::env::temp_dir().join(format!("st-voice-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let existing = dir.join("M1.json");
        std::fs::write(&existing, "ORIGINAL").unwrap();

        let res = VoiceStore::import_to(&dir, "M1", "{ not valid json");
        assert!(res.is_err(), "invalid import should fail");
        assert_eq!(
            std::fs::read_to_string(&existing).unwrap(),
            "ORIGINAL",
            "existing voice must survive a failed import"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
