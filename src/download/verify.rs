//! Per-file integrity verification: SHA-256 for LFS files, git blob SHA-1 for
//! non-LFS files. Exactly one applies per file, so the choice and the compare
//! live in one place.

use sha1::Sha1;
use sha2::{Digest, Sha256};

use super::hf::TreeFile;

/// The content check the listing offers for a file: SHA-256 (LFS) or the git
/// blob SHA-1 (`blob <len>\0` + content) for non-LFS files.
pub(crate) enum Verifier {
    Sha256 { hasher: Sha256, expected: String },
    GitBlob { hasher: Sha1, expected: String },
}

impl Verifier {
    pub(crate) fn for_file(file: &TreeFile) -> Option<Self> {
        if let Some(expected) = &file.sha256 {
            return Some(Verifier::Sha256 {
                hasher: Sha256::new(),
                expected: expected.clone(),
            });
        }
        if let Some(expected) = &file.git_oid {
            let mut hasher = Sha1::new();
            // git blob header uses the listing `size`. A wrong/missing size yields
            // a mismatch against the oid and rejects the file — fail-closed (never
            // publish unverified bytes). The HF tree API always reports size.
            hasher.update(format!("blob {}\0", file.size).as_bytes());
            return Some(Verifier::GitBlob {
                hasher,
                expected: expected.clone(),
            });
        }
        None
    }

    pub(crate) fn update(&mut self, bytes: &[u8]) {
        match self {
            Verifier::Sha256 { hasher, .. } => hasher.update(bytes),
            Verifier::GitBlob { hasher, .. } => hasher.update(bytes),
        }
    }

    /// Consume and compare against the expected hash. The `Err` message omits the
    /// file path, which the caller prefixes.
    pub(crate) fn verify(self) -> Result<(), String> {
        let (kind, got, expected) = match self {
            Verifier::Sha256 { hasher, expected } => ("sha256", hex(hasher.finalize()), expected),
            Verifier::GitBlob { hasher, expected } => ("git oid", hex(hasher.finalize()), expected),
        };
        if got == expected {
            Ok(())
        } else {
            Err(format!("{kind} mismatch: expected {expected}, got {got}"))
        }
    }
}

pub(crate) fn hex(bytes: impl AsRef<[u8]>) -> String {
    let mut s = String::with_capacity(bytes.as_ref().len() * 2);
    for b in bytes.as_ref() {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree_file(sha256: Option<&str>, git_oid: Option<&str>, size: u64) -> TreeFile {
        TreeFile {
            path: "x".into(),
            size,
            sha256: sha256.map(str::to_string),
            git_oid: git_oid.map(str::to_string),
        }
    }

    #[test]
    fn git_blob_oid_matches_canonical_git() {
        // Mirrors the framing fetch() uses for non-LFS files: sha1("blob <len>\0" + content).
        fn oid(content: &[u8]) -> String {
            let mut h = Sha1::new();
            h.update(format!("blob {}\0", content.len()).as_bytes());
            h.update(content);
            hex(h.finalize())
        }
        // Values from `git hash-object`.
        assert_eq!(oid(b""), "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391");
        assert_eq!(oid(b"hello\n"), "ce013625030ba8dba906f756967f9e9ca394464a");
    }

    #[test]
    fn selects_gitblob_for_non_lfs_and_checks_content() {
        let oid = "ce013625030ba8dba906f756967f9e9ca394464a"; // git oid of "hello\n"
        let file = tree_file(None, Some(oid), 6);

        let mut ok = Verifier::for_file(&file).unwrap();
        ok.update(b"hello\n");
        assert!(ok.verify().is_ok());

        let mut bad = Verifier::for_file(&file).unwrap();
        bad.update(b"HELLO\n");
        assert!(bad.verify().is_err());
    }

    #[test]
    fn selects_sha256_for_lfs() {
        // sha256("") — LFS takes precedence over any git_oid.
        let empty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        let mut v = Verifier::for_file(&tree_file(Some(empty), Some("ignored"), 0)).unwrap();
        v.update(b"");
        assert!(v.verify().is_ok());
    }

    #[test]
    fn absent_when_nothing_to_check() {
        assert!(Verifier::for_file(&tree_file(None, None, 0)).is_none());
    }

    #[test]
    fn sha256_hex_matches_known_vector() {
        // SHA-256("abc")
        let mut h = Sha256::new();
        h.update(b"abc");
        assert_eq!(
            hex(h.finalize()),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
