//! Hugging Face tree-API access: source config, listing, and the URL/path
//! validation applied to network-sourced data.

use super::{err, DownloadError};

pub(crate) const HF_HOST: &str = "https://huggingface.co";
const HF_HOSTNAME: &str = "huggingface.co";

/// Default model revision: an immutable commit SHA, not `main`, so a rebuild of
/// the same image pulls the same weights (the integrity hashes only attest to
/// whatever the listing currently points at). Override with
/// `SUPERTONIC_HF_REVISION` to track a moving ref or a newer release.
const DEFAULT_HF_REVISION: &str = "3cadd1ee6394adea1bd021217a0e650ede09a323";

/// Hugging Face source for the assets (repo + revision), from the environment.
pub(crate) struct HfConfig {
    pub(crate) repo: String,
    pub(crate) revision: String,
}

impl HfConfig {
    pub(crate) fn from_env() -> Self {
        Self {
            repo: env_or("SUPERTONIC_HF_REPO", "Supertone/supertonic-3"),
            revision: env_or("SUPERTONIC_HF_REVISION", DEFAULT_HF_REVISION),
        }
    }
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| default.to_string())
}

/// One downloadable file from the HF tree API.
pub(crate) struct TreeFile {
    /// Repo-relative path, e.g. `onnx/vocoder.onnx`.
    pub(crate) path: String,
    /// Content size in bytes, used for the cheap "already complete" skip.
    pub(crate) size: u64,
    /// SHA-256 hex for LFS files (`lfs.oid`).
    pub(crate) sha256: Option<String>,
    /// Git blob SHA-1 hex (`oid`) for non-LFS files, verified as the content hash.
    pub(crate) git_oid: Option<String>,
}

/// True only for an absolute `https://huggingface.co/...` URL, matching the host
/// exactly. A prefix check (`starts_with(HF_HOST)`) would also accept
/// `https://huggingface.co.evil.com/...`, so parse out the authority and compare
/// the host (ignoring any userinfo or port) to `HF_HOSTNAME`.
fn is_hf_url(url: &str) -> bool {
    let Some(rest) = url.strip_prefix("https://") else {
        return false;
    };
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    let host = authority.rsplit('@').next().unwrap_or(authority);
    let host = host.split(':').next().unwrap_or(host);
    host.eq_ignore_ascii_case(HF_HOSTNAME)
}

/// Reject listing paths that could escape the assets dir: absolute paths, any
/// `..` or empty segment (which also rejects `a//b` and leading/trailing `/`),
/// and backslashes.
pub(crate) fn is_safe_repo_path(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with('/')
        && !path.contains('\\')
        && path.split('/').all(|seg| !seg.is_empty() && seg != "..")
}

/// List the files under a repo subdir via the HF tree API, following the
/// `Link: …; rel="next"` cursor so large directories are not truncated.
pub(crate) fn list_dir(
    agent: &ureq::Agent,
    cfg: &HfConfig,
    subdir: &str,
) -> Result<Vec<TreeFile>, DownloadError> {
    let mut url = format!(
        "{HF_HOST}/api/models/{}/tree/{}/{subdir}",
        cfg.repo, cfg.revision
    );
    let mut files = Vec::new();
    loop {
        let mut resp = agent
            .get(&url)
            .call()
            .map_err(|e| err(format!("list {subdir}: {e}")))?;
        if !resp.status().is_success() {
            return Err(err(format!("list {subdir}: HTTP {}", resp.status())));
        }
        // Read the next-page cursor before consuming the body.
        let next = resp
            .headers()
            .get("link")
            .and_then(|h| h.to_str().ok())
            .and_then(parse_next_link);
        let body = resp
            .body_mut()
            .read_to_string()
            .map_err(|e| err(format!("read listing {subdir}: {e}")))?;
        files.extend(parse_tree(&body)?);
        match next {
            // Only follow a cursor that stays on the HF host (exact match, not a
            // prefix, so `huggingface.co.evil.com` is rejected).
            Some(n) if is_hf_url(&n) => url = n,
            Some(n) => return Err(err(format!("list {subdir}: off-host next link: {n}"))),
            None => break,
        }
    }
    Ok(files)
}

/// Parse one page of the HF tree API JSON array into files, skipping directory
/// entries. LFS files expose a content SHA-256 in `lfs.oid`.
fn parse_tree(json: &str) -> Result<Vec<TreeFile>, DownloadError> {
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|e| err(format!("tree json: {e}")))?;
    let array = value
        .as_array()
        .ok_or_else(|| err("tree is not an array"))?;
    let mut files = Vec::new();
    for item in array {
        if item.get("type").and_then(|t| t.as_str()) != Some("file") {
            continue;
        }
        let Some(path) = item.get("path").and_then(|p| p.as_str()) else {
            continue;
        };
        let lfs = item.get("lfs");
        let sha256 = lfs
            .and_then(|l| l.get("oid"))
            .and_then(|o| o.as_str())
            .map(str::to_string);
        // For non-LFS files the top-level `oid` is the git blob SHA-1 of the
        // content; for LFS files it is the pointer's sha, so ignore it there.
        let git_oid = if lfs.is_none() {
            item.get("oid").and_then(|o| o.as_str()).map(str::to_string)
        } else {
            None
        };
        let size = lfs
            .and_then(|l| l.get("size"))
            .and_then(serde_json::Value::as_u64)
            .or_else(|| item.get("size").and_then(serde_json::Value::as_u64))
            .unwrap_or(0);
        files.push(TreeFile {
            path: path.to_string(),
            size,
            sha256,
            git_oid,
        });
    }
    Ok(files)
}

/// Extract the `rel="next"` URL from an HTTP `Link` header, if present.
fn parse_next_link(link: &str) -> Option<String> {
    for part in link.split(',') {
        if part.contains("rel=\"next\"") {
            let start = part.find('<')?;
            let end = part[start + 1..].find('>')? + start + 1;
            return Some(part[start + 1..end].to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unsafe_repo_paths() {
        for bad in [
            "",
            "/etc/passwd",
            "..",
            "onnx/../../etc/passwd",
            "a//b",
            "a\\b",
            "/onnx/x",
            "onnx/",
        ] {
            assert!(!is_safe_repo_path(bad), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn accepts_plain_repo_paths() {
        for ok in ["onnx/vocoder.onnx", "voice_styles/M1.json", "model.onnx"] {
            assert!(is_safe_repo_path(ok), "{ok:?} should be accepted");
        }
    }

    #[test]
    fn parse_tree_keeps_files_with_lfs_sha_and_skips_dirs() {
        let json = r#"[
            {"type":"directory","path":"onnx/sub"},
            {"type":"file","path":"onnx/vocoder.onnx","oid":"pointer_sha","size":123,
             "lfs":{"oid":"abc123","size":456}},
            {"type":"file","path":"onnx/config.json","oid":"deadbeef","size":7}
        ]"#;
        let files = parse_tree(json).unwrap();
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].path, "onnx/vocoder.onnx");
        assert_eq!(files[0].sha256.as_deref(), Some("abc123"));
        assert_eq!(files[0].git_oid, None); // LFS: top-level oid is the pointer, ignored
        assert_eq!(files[0].size, 456); // lfs.size wins over the pointer size
        assert_eq!(files[1].path, "onnx/config.json");
        assert_eq!(files[1].sha256, None);
        assert_eq!(files[1].git_oid.as_deref(), Some("deadbeef")); // non-LFS: git blob sha-1
        assert_eq!(files[1].size, 7);
    }

    #[test]
    fn is_hf_url_requires_exact_host() {
        assert!(is_hf_url("https://huggingface.co/api/models/x/tree/main"));
        assert!(is_hf_url("https://huggingface.co/api?cursor=abc"));
        assert!(is_hf_url("https://huggingface.co:443/api"));
        // The prefix-match bypass a `starts_with(HF_HOST)` check would allow.
        assert!(!is_hf_url("https://huggingface.co.evil.com/api"));
        assert!(!is_hf_url("https://evil.com/huggingface.co"));
        assert!(!is_hf_url("https://user@evil.com/api"));
        assert!(!is_hf_url("http://huggingface.co/api")); // not https
        assert!(!is_hf_url("https://HUGGINGFACE.CO.evil.com/"));
    }

    #[test]
    fn parse_next_link_finds_cursor() {
        let header =
            "<https://huggingface.co/api/...?cursor=abc>; rel=\"next\", <https://x>; rel=\"prev\"";
        assert_eq!(
            parse_next_link(header).as_deref(),
            Some("https://huggingface.co/api/...?cursor=abc")
        );
        assert_eq!(parse_next_link("<https://x>; rel=\"prev\""), None);
    }
}
