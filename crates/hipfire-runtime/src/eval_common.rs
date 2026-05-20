//! Shared helpers for the KLD-eval example binaries (`build_kld_ref`,
//! `eval_hipfire`, `eval_gguf`). Each of these examples re-verifies the
//! same three invariants before doing work; centralising the helpers
//! here so a fix to one applies to all three. See
//! `docs/plans/issue-113-quant-quality-eval.md` §4 (Reference format &
//! pipeline) for what these checks defend against.

use std::path::Path;
use std::process::Command;

/// Verify the reference file's sha256 against the in-tree
/// `manifest.json` index.
///
/// Layout assumption: ref lives at `.../refs/<name>.kldref.bin`, manifest
/// at `.../harness/manifest.json` (sibling to `refs/`).
///
/// Behaviour:
/// - if the manifest is absent OR has no entry for `<name>`, emit a
///   warning and return (developer pre-upload state);
/// - if sha256 disagrees, print a clear error and `std::process::exit(2)`.
///
/// `tool_name` is the binary's short name (e.g. `"eval_hipfire"`) and is
/// used only in log lines.
pub fn verify_ref_sha256(ref_path: &Path, tool_name: &str) {
    let manifest_path = match ref_path.parent().and_then(|p| p.parent()) {
        Some(p) => p.join("harness").join("manifest.json"),
        None => {
            eprintln!(
                "warning: cannot locate harness/manifest.json relative to {}; \
                 skipping ref sha256 check",
                ref_path.display()
            );
            return;
        }
    };
    if !manifest_path.exists() {
        eprintln!(
            "warning: {} missing; skipping ref sha256 check",
            manifest_path.display()
        );
        return;
    }
    let manifest_file = std::fs::File::open(&manifest_path).expect("open manifest.json");
    let manifest: serde_json::Value =
        serde_json::from_reader(manifest_file).expect("parse manifest.json");
    let ref_name = ref_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    let expected = manifest
        .get("references")
        .and_then(|r| r.get(ref_name))
        .and_then(|r| r.get("sha256"))
        .and_then(|s| s.as_str())
        .map(String::from);
    let expected = match expected {
        Some(s) => s,
        None => {
            eprintln!("warning: no manifest entry / sha256 for {ref_name}; skipping check");
            return;
        }
    };
    eprintln!("{tool_name}: computing sha256 of {} ...", ref_path.display());
    let out = Command::new("sha256sum")
        .arg(ref_path)
        .output()
        .expect("invoke sha256sum");
    let actual = String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .map(String::from)
        .expect("empty sha256sum output");
    if actual != expected {
        eprintln!("ERROR: ref sha256 mismatch for {}", ref_path.display());
        eprintln!("  expected: {expected}");
        eprintln!("  actual:   {actual}");
        std::process::exit(2);
    }
    eprintln!("{tool_name}: verified ref sha256 = {actual}");
}

/// Verify the slice file's md5 against the sibling `slice.md5` (one-line
/// `<md5>` or `md5sum`-format output).
///
/// Behaviour:
/// - if `<slice_dir>/slice.md5` is absent OR no recognisable hash on the
///   first line, emit a warning and return;
/// - if md5 disagrees, print a clear error and `std::process::exit(2)`.
pub fn verify_slice_md5(slice_path: &Path, tool_name: &str) {
    let md5_path = match slice_path.parent() {
        Some(p) => p.join("slice.md5"),
        None => return,
    };
    if !md5_path.exists() {
        eprintln!(
            "warning: {} missing; skipping slice md5 check",
            md5_path.display()
        );
        return;
    }
    let expected_line = match std::fs::read_to_string(&md5_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("warning: cannot read {}: {e}; skipping slice md5", md5_path.display());
            return;
        }
    };
    let expected = match expected_line.split_whitespace().next() {
        Some(s) => s.to_string(),
        None => {
            eprintln!("warning: {} empty; skipping", md5_path.display());
            return;
        }
    };
    let out = Command::new("md5sum")
        .arg(slice_path)
        .output()
        .expect("invoke md5sum");
    let actual = String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .map(String::from)
        .expect("empty md5sum output");
    if actual != expected {
        eprintln!("ERROR: slice md5 mismatch for {}", slice_path.display());
        eprintln!("  expected: {expected}");
        eprintln!("  actual:   {actual}");
        std::process::exit(2);
    }
    eprintln!("{tool_name}: verified slice md5 = {actual}");
}

/// Verify that the supplied `llama-perplexity` binary's reported commit
/// hash matches `pinned`.
///
/// Behaviour:
/// - parses `<bin> --version`'s "version: N (hash)" line;
/// - demands the binary's hash be ≥ 7 chars (collision floor on short
///   git hashes);
/// - compares an equal-length prefix of `pinned` to the binary's hash.
///   The prior implementation used `pinned.starts_with(hash)`, which
///   would accept arbitrarily short binary hashes (e.g. "9d" matching
///   "9dcf83552…"). Surfaced by glm-5 review finding 2.5.
///
/// On any mismatch, `std::process::exit(2)`.
pub fn verify_llama_commit(bin: &str, pinned: &str, tool_name: &str) {
    let out = Command::new(bin).arg("--version").output();
    let out = match out {
        Ok(o) => o,
        Err(e) => {
            eprintln!("ERROR: failed to invoke `{bin} --version`: {e}");
            std::process::exit(2);
        }
    };
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let needle = "version: ";
    let after_version = match combined.find(needle) {
        Some(i) => &combined[i + needle.len()..],
        None => {
            eprintln!("ERROR: could not find 'version: ' in `{bin} --version` output");
            std::process::exit(2);
        }
    };
    let open = match after_version.find('(') {
        Some(i) => i,
        None => {
            eprintln!("ERROR: malformed `--version` output: no '(' after version number");
            std::process::exit(2);
        }
    };
    let after_paren = &after_version[open + 1..];
    let close = match after_paren.find(')') {
        Some(i) => i,
        None => {
            eprintln!("ERROR: malformed `--version` output: no ')' after commit hash");
            std::process::exit(2);
        }
    };
    let hash = &after_paren[..close];
    if hash.len() < 7 {
        eprintln!(
            "ERROR: llama-perplexity reported a {}-char hash; want ≥ 7",
            hash.len()
        );
        eprintln!("  binary:    {bin}");
        eprintln!("  reported:  {hash}");
        std::process::exit(2);
    }
    if hash.len() > pinned.len() {
        eprintln!(
            "ERROR: llama-perplexity hash ({}) longer than pinned ({})",
            hash.len(),
            pinned.len()
        );
        std::process::exit(2);
    }
    let pinned_prefix = &pinned[..hash.len()];
    if hash != pinned_prefix {
        eprintln!("ERROR: llama-perplexity commit mismatch");
        eprintln!("  binary:             {bin}");
        eprintln!("  expected (pinned):  {pinned}");
        eprintln!("  actual (--version): {hash}");
        eprintln!("  Either rebuild llama.cpp at the pinned commit, or update");
        eprintln!("  PINNED_LLAMACPP_COMMIT in this binary AND in the PRD.");
        std::process::exit(2);
    }
    eprintln!("{tool_name}: verified llama.cpp commit prefix {hash}");
}
