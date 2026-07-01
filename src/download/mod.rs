//! In-process asset download from the Hugging Face Hub.
//!
//! Files are placed in the flat `<assets>/onnx` + `<assets>/voice_styles`
//! layout the engine loads from, discovered via the HF tree API ([`hf`]) so no
//! filenames are hard-coded. Guarantees:
//!
//! - Each file streams to a `*.part` temp and is renamed only after a complete
//!   download, so an interrupted fetch never leaves a file that later looks
//!   present.
//! - Every file we download is verified ([`verify`]) against the listing hash
//!   (SHA-256 for LFS files, the git blob SHA-1 otherwise); a mismatch is
//!   retried. A pre-existing file already at the expected size is reused without
//!   re-hashing (the assets dir is a trusted volume).
//! - The per-dir `.<subdir>-complete` marker is written only after a non-empty
//!   listing and every file succeeds, so a failed listing cannot mark a
//!   directory done.
//! - Listing paths are validated before use, since they come from the network.

mod hf;
mod verify;

use std::fs;
use std::io::{Read, Write};
use std::path::Path;
use std::time::Duration;

use hf::{is_safe_repo_path, list_dir, HfConfig, TreeFile, HF_HOST};
use verify::Verifier;

const RETRIES: usize = 3;
const CHUNK: usize = 64 * 1024;
/// Absolute ceiling for a file whose listing omits `size` (reported as 0), so an
/// unsized response cannot stream unbounded to disk before verification. Far
/// above any real asset; sized files use their exact length instead.
const MAX_UNSIZED_BYTES: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Debug)]
pub struct DownloadError(String);

impl std::fmt::Display for DownloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub(crate) fn err(msg: impl Into<String>) -> DownloadError {
    DownloadError(msg.into())
}

/// Whether a listing path lives under `subdir`, e.g. `onnx/vocoder.onnx` under
/// `onnx`. Anything else is a malformed listing and must not count toward the
/// `.<subdir>-complete` marker.
fn path_in_subdir(path: &str, subdir: &str) -> bool {
    match path.strip_prefix(subdir).and_then(|r| r.strip_prefix('/')) {
        Some(rest) => !rest.is_empty(),
        None => false,
    }
}

/// The byte ceiling to enforce while streaming: the listing size when known,
/// otherwise an absolute cap.
fn stream_cap(listed_size: u64) -> u64 {
    if listed_size != 0 {
        listed_size
    } else {
        MAX_UNSIZED_BYTES
    }
}

/// Whether an on-disk file is reused without re-downloading: only when the
/// listing gives a non-zero size and the existing file matches it exactly. The
/// reused bytes are trusted by size (the assets dir is a trusted volume), not
/// re-hashed.
fn reuse_existing(existing_len: u64, listed_size: u64) -> bool {
    listed_size != 0 && existing_len == listed_size
}

/// HTTP client with bounded connect/response timeouts, so a stalled endpoint
/// fails (and is retried) instead of hanging startup indefinitely. There is no
/// body-duration timeout — the model is hundreds of MB and streams a while — but
/// the byte count is capped while streaming (see `stream_cap` in `fetch`).
fn http_agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_recv_response(Some(Duration::from_secs(60)))
        .build()
        .into()
}

/// Ensure each `subdir` (e.g. `onnx`, `voice_styles`) is fully present under
/// `assets_dir`, downloading anything missing. Idempotent: a present completion
/// marker makes it a no-op, so it is safe to call on every startup.
pub fn ensure_assets(assets_dir: &Path, subdirs: &[&str]) -> Result<(), DownloadError> {
    let cfg = HfConfig::from_env();
    let agent = http_agent();
    fs::create_dir_all(assets_dir)
        .map_err(|e| err(format!("create {}: {e}", assets_dir.display())))?;
    for subdir in subdirs {
        ensure_dir(&agent, assets_dir, subdir, &cfg)?;
    }
    Ok(())
}

/// Confirm the assets dir is writable before a download begins. Read-only
/// pre-populated assets never reach this — their completion marker short-circuits
/// the download in `ensure_dir` before this is called.
fn ensure_writable(assets_dir: &Path) -> Result<(), DownloadError> {
    fs::create_dir_all(assets_dir)
        .map_err(|e| err(format!("create {}: {e}", assets_dir.display())))?;
    let probe = assets_dir.join(format!(".write-probe.{}", std::process::id()));
    fs::write(&probe, b"").map_err(|e| {
        err(format!(
            "assets dir {} is not writable by the current user: {e}. A download is \
             needed but cannot be saved. If you changed the container user (e.g. \
             SUPERTONIC_UID/GID), chown the mounted assets dir to that user; or \
             pre-populate it and mount it read-only.",
            assets_dir.display()
        ))
    })?;
    let _ = fs::remove_file(&probe);
    Ok(())
}

fn ensure_dir(
    agent: &ureq::Agent,
    assets_dir: &Path,
    subdir: &str,
    cfg: &HfConfig,
) -> Result<(), DownloadError> {
    let marker = assets_dir.join(format!(".{subdir}-complete"));
    if marker.exists() {
        return Ok(());
    }
    // A download is required; fail early and clearly if we cannot write to it.
    ensure_writable(assets_dir)?;
    eprintln!("downloading {subdir} from {}@{}...", cfg.repo, cfg.revision);

    let files = list_dir(agent, cfg, subdir)?;
    // An empty listing means the request failed or returned nothing; writing the
    // marker now would permanently mark the directory done with files missing.
    if files.is_empty() {
        return Err(err(format!("empty or failed listing for {subdir}")));
    }
    // Reject a listing pointing outside the subdir before downloading anything.
    for file in &files {
        if !path_in_subdir(&file.path, subdir) {
            return Err(err(format!(
                "listing entry {} is outside {subdir}/",
                file.path
            )));
        }
    }
    for file in &files {
        fetch_with_retry(agent, assets_dir, file, cfg)?;
    }
    fs::write(&marker, b"").map_err(|e| err(format!("write marker {subdir}: {e}")))?;
    Ok(())
}

fn fetch_with_retry(
    agent: &ureq::Agent,
    assets_dir: &Path,
    file: &TreeFile,
    cfg: &HfConfig,
) -> Result<(), DownloadError> {
    let mut last = None;
    for attempt in 1..=RETRIES {
        match fetch(agent, assets_dir, file, cfg) {
            Ok(()) => return Ok(()),
            Err(e) => {
                eprintln!("  {} (attempt {attempt}/{RETRIES}): {e}", file.path);
                last = Some(e);
            }
        }
    }
    Err(last.unwrap_or_else(|| err("download failed")))
}

fn fetch(
    agent: &ureq::Agent,
    assets_dir: &Path,
    file: &TreeFile,
    cfg: &HfConfig,
) -> Result<(), DownloadError> {
    // The path comes from the network listing, so validate it before joining.
    if !is_safe_repo_path(&file.path) {
        return Err(err(format!("unsafe path in listing: {}", file.path)));
    }
    let dest = assets_dir.join(&file.path);

    // Skip a file already on disk at the expected size. The dir marker gates
    // normal restarts; this only matters when resuming after a failed run, and
    // avoids re-hashing hundreds of MB.
    if let Ok(meta) = fs::metadata(&dest) {
        if reuse_existing(meta.len(), file.size) {
            return Ok(());
        }
    }

    let parent = dest
        .parent()
        .ok_or_else(|| err(format!("no parent for {}", file.path)))?;
    fs::create_dir_all(parent).map_err(|e| err(format!("mkdir {}: {e}", parent.display())))?;
    let name = dest
        .file_name()
        .ok_or_else(|| err(format!("no file name for {}", file.path)))?
        .to_string_lossy();
    let tmp = parent.join(format!("{name}.part"));

    let url = format!(
        "{HF_HOST}/{}/resolve/{}/{}",
        cfg.repo, cfg.revision, file.path
    );
    eprintln!("  downloading {}", file.path);
    let mut resp = agent
        .get(&url)
        .call()
        .map_err(|e| err(format!("get {}: {e}", file.path)))?;
    if !resp.status().is_success() {
        return Err(err(format!("get {}: HTTP {}", file.path, resp.status())));
    }

    // Every asset must carry an integrity hash (sha256 for LFS, git oid
    // otherwise); a listing entry with neither is malformed, so refuse it.
    let mut verifier = Verifier::for_file(file)
        .ok_or_else(|| err(format!("{}: listing has no integrity hash", file.path)))?;

    let mut reader = resp.body_mut().as_reader();
    let mut out =
        fs::File::create(&tmp).map_err(|e| err(format!("create {}: {e}", tmp.display())))?;
    let mut buf = vec![0u8; CHUNK];
    let mut written: u64 = 0;
    // Stop a runaway response (hostile/misconfigured repo) from filling the disk.
    // Use the listing size when known; otherwise an absolute ceiling so an
    // unsized stream cannot run unbounded before verification.
    let cap = stream_cap(file.size);
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| err(format!("read {}: {e}", file.path)))?;
        if n == 0 {
            break;
        }
        written += n as u64;
        if written > cap {
            let _ = fs::remove_file(&tmp);
            return Err(err(format!("{}: response exceeds {cap} bytes", file.path)));
        }
        verifier.update(&buf[..n]);
        out.write_all(&buf[..n])
            .map_err(|e| err(format!("write {}: {e}", tmp.display())))?;
    }
    out.flush()
        .map_err(|e| err(format!("flush {}: {e}", tmp.display())))?;
    // fsync before the rename publishes the file: a host crash could otherwise
    // leave a size-correct file with unwritten (zero) blocks that the size-only
    // reuse check would trust on the next start.
    out.sync_all()
        .map_err(|e| err(format!("sync {}: {e}", tmp.display())))?;
    drop(out);

    // A known byte count that doesn't match means a truncated download; never
    // publish it, so a short read can't be marked complete.
    if file.size != 0 && written != file.size {
        let _ = fs::remove_file(&tmp);
        return Err(err(format!(
            "{}: got {written} bytes, expected {}",
            file.path, file.size
        )));
    }
    if let Err(msg) = verifier.verify() {
        let _ = fs::remove_file(&tmp);
        return Err(err(format!("{}: {msg}", file.path)));
    }

    // Atomic publish: a reader only ever sees the fully-downloaded file.
    fs::rename(&tmp, &dest).map_err(|e| err(format!("rename {}: {e}", dest.display())))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_in_subdir_requires_a_file_under_the_dir() {
        assert!(path_in_subdir("onnx/vocoder.onnx", "onnx"));
        assert!(path_in_subdir("voice_styles/M1.json", "voice_styles"));
        assert!(!path_in_subdir("onnx", "onnx")); // the dir itself, no file
        assert!(!path_in_subdir("onnx/", "onnx")); // empty file name
        assert!(!path_in_subdir("other/x.json", "onnx")); // different dir
        assert!(!path_in_subdir("onnx2/x", "onnx")); // prefix is not a path segment
        assert!(!path_in_subdir("x/onnx/y", "onnx")); // nested elsewhere
    }

    #[test]
    fn stream_cap_uses_size_or_absolute_ceiling() {
        assert_eq!(stream_cap(123), 123);
        assert_eq!(stream_cap(0), MAX_UNSIZED_BYTES);
    }

    #[test]
    fn reuse_existing_only_on_exact_known_size() {
        assert!(reuse_existing(100, 100)); // match -> reuse
        assert!(!reuse_existing(99, 100)); // size mismatch -> re-download
        assert!(!reuse_existing(0, 0)); // unknown listing size -> never reuse
        assert!(!reuse_existing(100, 0)); // unknown listing size -> never reuse
    }

    #[cfg(unix)]
    #[test]
    fn ensure_writable_accepts_writable_rejects_readonly() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("st-writable-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        // A normal writable dir passes and leaves no probe behind.
        assert!(ensure_writable(&dir).is_ok());
        let leftovers: Vec<_> = fs::read_dir(&dir).unwrap().flatten().collect();
        assert!(
            leftovers.is_empty(),
            "probe must be cleaned up: {leftovers:?}"
        );

        // Make it read-only and assert the denial — but only when the OS actually
        // denies the write (root ignores DAC permissions, so skip the assert there).
        let mut perms = fs::metadata(&dir).unwrap().permissions();
        perms.set_mode(0o555);
        fs::set_permissions(&dir, perms).unwrap();
        if fs::write(dir.join(".probe-check"), b"").is_err() {
            assert!(
                ensure_writable(&dir).is_err(),
                "read-only dir must be rejected"
            );
        } else {
            let _ = fs::remove_file(dir.join(".probe-check"));
        }

        let mut perms = fs::metadata(&dir).unwrap().permissions();
        perms.set_mode(0o755);
        let _ = fs::set_permissions(&dir, perms);
        let _ = fs::remove_dir_all(&dir);
    }
}
