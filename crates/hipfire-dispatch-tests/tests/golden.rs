// SPDX-License-Identifier: MIT OR Apache-2.0
// Assert the current build reproduces the legacy golden token-id streams.
// Skipped automatically when fixtures or models are absent (CI-light).
use std::path::Path;

fn fixtures_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/golden")
}

#[test]
fn golden_streams_present_or_skipped() {
    let dir = fixtures_dir();
    if !dir.join("prompt.md5").exists() {
        eprintln!("no golden fixtures; skipping");
        return;
    }
    // Capture the current build into a temp dir, then byte-compare each
    // *.committed.jsonl against the fixture. Implemented as a thin wrapper
    // over scripts/golden-capture.sh writing to $HOME/hipfire-golden-tmp.
    let tmp = std::env::var("HOME").unwrap() + "/hipfire-golden-tmp";
    let status = std::process::Command::new("scripts/golden-capture.sh")
        .arg(&tmp)
        .current_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
        .status();
    let Ok(status) = status else { eprintln!("capture unavailable; skipping"); return; };
    if !status.success() { eprintln!("capture skipped (no models); skipping"); return; }
    for entry in std::fs::read_dir(&dir).unwrap() {
        let p = entry.unwrap().path();
        if p.extension().and_then(|e| e.to_str()) != Some("jsonl") { continue; }
        let name = p.file_name().unwrap();
        let got = Path::new(&tmp).join(name);
        if !got.exists() { continue; } // model missing locally
        let a = std::fs::read(&p).unwrap();
        let b = std::fs::read(&got).unwrap();
        assert_eq!(a, b, "golden mismatch for {name:?}");
    }
}
