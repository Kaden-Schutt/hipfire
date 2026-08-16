// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Source guard: no per-token `decode(&[tok])` on a client-output path.
//!
//! `Tokenizer::decode()` reassembles UTF-8 byte fragments only within a single
//! call, so calling it once per streamed token runs `from_utf8_lossy` over a
//! partial sequence. Byte-level BPE and SentencePiece byte-fallback spread one
//! character across several tokens (`中` = `<0xE4> <0xB8> <0xAD>`, every emoji,
//! `∑∫`), so the client receives `���` where the model emitted an emoji.
//! [`crate::tokenizer::TokenTextStream`] is the fix; this module checks that it
//! is actually *used*.
//!
//! Every other test around this fix verifies the helper. This one verifies the
//! wiring — without it, an arch added next month reintroduces the bug on its
//! own decode loop and every other test still passes. That is exactly how the
//! bug survived on eight archs while the mainline qwen35 path was correct.
//!
//! Precedent for this style of guard in this repo: `scripts/check-fmt-bomb.sh`,
//! `scripts/test-gpu-lock.sh`.

#![cfg(test)]

use std::path::{Path, PathBuf};

/// Repository root, derived from this crate's manifest directory.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root must resolve from CARGO_MANIFEST_DIR")
}

/// Source files the guard covers.
///
/// `daemon.rs` and the per-arch crates are where every streamed decode loop
/// lives. `hipfire-cli/src/main.rs` is included because the multi-slot serve
/// backend does its own token → text loop there — that path was one of the
/// affected sites and nothing else would keep it fixed.
fn guarded_files(root: &Path) -> Vec<PathBuf> {
    let mut files = vec![
        root.join("crates/hipfire-runtime/examples/daemon.rs"),
        root.join("crates/hipfire-cli/src/main.rs"),
    ];
    let crates_dir = root.join("crates");
    let mut arch_dirs: Vec<PathBuf> = std::fs::read_dir(&crates_dir)
        .expect("crates/ must be readable")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("hipfire-arch-"))
        })
        .map(|p| p.join("src"))
        .filter(|p| p.is_dir())
        .collect();
    arch_dirs.sort();
    for dir in arch_dirs {
        collect_rs(&dir, &mut files);
    }
    files.sort();
    files
}

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            collect_rs(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// Line ranges (1-based, inclusive) covered by a `#[cfg(test)]` item.
///
/// An attribute at indent N marks the item that follows; that item's body ends
/// at the first line that is exactly `}` (or `};`) at the same indent. rustfmt
/// guarantees that shape for every item this repo compiles, and the guard
/// self-tests below fail loudly if the assumption ever stops holding.
fn cfg_test_regions(lines: &[&str]) -> Vec<(usize, usize)> {
    let mut regions = Vec::new();
    let mut i = 0usize;
    while i < lines.len() {
        if lines[i].trim() == "#[cfg(test)]" {
            let indent = lines[i].len() - lines[i].trim_start().len();
            let close = format!("{}}}", " ".repeat(indent));
            let close_semi = format!("{close};");
            let mut j = i + 1;
            while j < lines.len() && lines[j] != close && lines[j] != close_semi {
                j += 1;
            }
            regions.push((i + 1, j + 1));
            i = j;
        }
        i += 1;
    }
    regions
}

/// Why a `decode(&[` occurrence is permitted, or `None` if it is a violation.
///
/// Kept deliberately narrow. A bare pattern with no rationale invites the first
/// person who hits a failure to widen it until it stops complaining, and the
/// guard becomes decorative.
fn allowed_reason(lines: &[&str], idx: usize, in_test: bool) -> Option<&'static str> {
    let line = lines[idx];
    let trimmed = line.trim_start();
    if trimmed.starts_with("//") {
        return Some("comment, not code");
    }
    if in_test {
        return Some("#[cfg(test)] — test fixture, not client output");
    }
    // Grammar matchers drive a text state machine, not client output, and are
    // deliberately out of scope (they do still see U+FFFD for non-ASCII, which
    // makes grammar-constrained decoding over non-ASCII a separate open
    // question). The decode and the `advance` are sometimes split across lines,
    // so look at a small window.
    let window_end = (idx + 3).min(lines.len());
    let window = lines[idx..window_end].join(" ");
    if window.contains("matcher.advance") || window.contains("matcher.is_token_allowed") {
        return Some("grammar matcher — text state machine, not client output");
    }
    // Whole-vocab dumps decode every id independently on purpose; there is no
    // stream to reassemble across.
    if line.contains("(0..") && line.contains(".map(") {
        return Some("whole-vocab dump — per-id decode is the intent");
    }
    None
}

/// Scan one file, returning `file:line: source` for every violation.
fn violations_in(path: &Path, source: &str) -> Vec<String> {
    let lines: Vec<&str> = source.split('\n').collect();
    let regions = cfg_test_regions(&lines);
    let mut out = Vec::new();
    for (idx, line) in lines.iter().enumerate() {
        if !line.contains("decode(&[") {
            continue;
        }
        let lineno = idx + 1;
        let in_test = regions.iter().any(|&(a, b)| a <= lineno && lineno <= b);
        if allowed_reason(&lines, idx, in_test).is_none() {
            out.push(format!("{}:{}: {}", path.display(), lineno, line.trim()));
        }
    }
    out
}

/// The guard itself.
#[test]
fn no_per_token_decode_on_client_output_paths() {
    let root = repo_root();
    let files = guarded_files(&root);
    assert!(
        files.len() > 5,
        "guard scanned only {} files — the path set is wrong, not the code clean",
        files.len()
    );

    let mut scanned_matches = 0usize;
    let mut violations = Vec::new();
    for path in &files {
        let Ok(source) = std::fs::read_to_string(path) else {
            continue;
        };
        scanned_matches += source.matches("decode(&[").count();
        violations.extend(violations_in(path, &source));
    }

    // Guard against the guard silently matching nothing (a moved file, a
    // renamed API): the legitimate grammar/vocab/test uses are still there, so
    // a zero total means the scan itself broke.
    assert!(
        scanned_matches > 5,
        "guard found only {scanned_matches} `decode(&[` occurrences across {} files — \
         the scan is broken, not the tree clean",
        files.len()
    );

    assert!(
        violations.is_empty(),
        "per-token `decode(&[tok])` on a client-output path — a character whose UTF-8 \
         spans several tokens (emoji, byte-fallback CJK, ∑∫) reaches the client as U+FFFD.\n\
         Use `hipfire_runtime::tokenizer::TokenTextStream` (push per token, flush at end of \
         stream) instead.\n\n{}",
        violations.join("\n")
    );
}

/// The guard must actually reject the thing it exists to reject.
#[test]
fn guard_flags_a_reintroduced_per_token_decode() {
    let bad = "fn generate_newarch() {\n    let frag = tokenizer.decode(&[next_tok]);\n    emit(&frag);\n}\n";
    let found = violations_in(Path::new("synthetic.rs"), bad);
    assert_eq!(
        found.len(),
        1,
        "guard failed to flag a fresh per-token decode: {found:?}"
    );
    assert!(found[0].contains("synthetic.rs:2"));
}

/// The allowlist must stay narrow — and must still permit exactly the three
/// documented legitimate shapes.
#[test]
fn guard_allowlist_covers_only_the_documented_shapes() {
    let grammar = "fn f() {\n    grammar_matcher.advance(&tokenizer.decode(&[t]));\n}\n";
    assert!(violations_in(Path::new("g.rs"), grammar).is_empty());

    let grammar_split =
        "fn f() {\n    let text = tokenizer.decode(&[t]);\n    grammar_matcher.advance(&text);\n}\n";
    assert!(violations_in(Path::new("g.rs"), grammar_split).is_empty());

    let vocab_dump =
        "fn f() {\n    let v: Vec<String> = (0..n).map(|id| tokenizer.decode(&[id])).collect();\n}\n";
    assert!(violations_in(Path::new("v.rs"), vocab_dump).is_empty());

    let in_test = "#[cfg(test)]\nmod tests {\n    fn t() {\n        let s = tok.decode(&[id]);\n    }\n}\n";
    assert!(violations_in(Path::new("t.rs"), in_test).is_empty());

    // …but not a plain emit dressed up next to unrelated text.
    let sneaky = "fn f() {\n    let frag = tokenizer.decode(&[t]);\n    buf.push_str(&frag);\n}\n";
    assert_eq!(violations_in(Path::new("s.rs"), sneaky).len(), 1);
}

/// `#[cfg(test)]` region detection must not swallow the rest of the file — if
/// it did, every real violation after the first test module would be excused.
#[test]
fn cfg_test_region_ends_at_the_matching_close() {
    let src = "#[cfg(test)]\nmod tests {\n    fn t() {}\n}\n\nfn real() {\n    let f = tok.decode(&[t]);\n}\n";
    let lines: Vec<&str> = src.split('\n').collect();
    let regions = cfg_test_regions(&lines);
    assert_eq!(regions, vec![(1, 4)], "region must close at the top-level }}");
    assert_eq!(violations_in(Path::new("r.rs"), src).len(), 1);
}
