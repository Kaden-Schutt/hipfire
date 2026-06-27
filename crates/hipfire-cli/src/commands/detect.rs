// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `hipfire detect` — run the observational coherence detectors over a captured
//! token stream and emit a JSON verdict.
//!
//! This is the CLI front-end for the `hipfire-detect` `DetectorBank`. It exists
//! so shell gates stop re-implementing the Path-A token-attractor detector as
//! copy-pasted Python heredocs (the old `DETECT_PY` in
//! `coherence-gate-dflash.sh` / `path-c-smoke.sh`). The detector logic — and the
//! `self_check` anti-rot harness that guards it — now lives in one place: Rust.
//!
//! Input: the demo/daemon stdout on stdin. Token IDs are read from the
//! `DFlash tokens: [..]` / `AR tokens: [..]` line the runtime prints. Output: a
//! single JSON line carrying `ok` / `soft_warn` (the keys the gates already
//! parse) plus a per-detector breakdown.
//!
//! By default only the first-128-window attractor runs, matching the old
//! `DETECT_PY` semantics exactly. `--path-a` opts into the fuller token-id
//! attractor family (first-128 + last-128 + long-state collapse) that the Rust
//! port adds over the shell original.

use clap::Args;
use hipfire_detect::{
    attractor::{AttractorFirst128, AttractorLast128, LongStateCollapse},
    parity, rollback, DetectorBank, Event, Severity, Verdict,
};
use regex::Regex;
use serde_json::{json, Value};
use std::io::Read;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum RollbackKind {
    /// Parse the `rollback_parity:` replay-count line.
    Replay,
    /// Parse the `verify_graph:` count line.
    VerifyGraph,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Source {
    /// Prefer the `DFlash tokens:` line, fall back to `AR tokens:`.
    Auto,
    /// Read only the `DFlash tokens:` line.
    Dflash,
    /// Read only the `AR tokens:` line.
    Ar,
}

#[derive(Debug, Args)]
pub struct DetectArgs {
    /// Which `tokens: [..]` line to read when the stream contains both.
    #[arg(long, value_enum, default_value_t = Source::Auto)]
    source: Source,

    /// Run the full Path-A token-attractor family (first-128 + last-128 +
    /// long-state collapse) instead of just the first-128 window.
    #[arg(long)]
    path_a: bool,

    /// Exit non-zero when a hard fail is detected. Off by default so the
    /// command is a drop-in for the always-exit-0 Python it replaces.
    #[arg(long)]
    exit_code: bool,

    /// Parity mode: file holding the AR-baseline run output (its
    /// `AR tokens:` line). Requires --parity-dflash. When both are set the
    /// command compares the two streams instead of reading stdin.
    #[arg(long, requires = "parity_dflash")]
    parity_ar: Option<PathBuf>,

    /// Parity mode: file holding the DFlash run output (its `DFlash tokens:`
    /// line). Requires --parity-ar.
    #[arg(long, requires = "parity_ar")]
    parity_dflash: Option<PathBuf>,

    /// Rollback mode: parse a single-line replay/verify stat summary instead
    /// of running detectors. Reads `--file` if given, else stdin.
    #[arg(long, value_enum)]
    rollback: Option<RollbackKind>,

    /// Input file for `--rollback` mode (defaults to stdin when omitted).
    #[arg(long)]
    file: Option<PathBuf>,
}

pub fn run(args: DetectArgs) -> anyhow::Result<()> {
    if let (Some(ar_path), Some(df_path)) = (&args.parity_ar, &args.parity_dflash) {
        return run_parity(ar_path, df_path, args.exit_code);
    }
    if let Some(kind) = args.rollback {
        return run_rollback(kind, args.file.as_deref(), args.exit_code);
    }

    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;

    let toks = match parse_token_line(&input, args.source) {
        None => {
            emit(&json!({ "ok": false, "soft_warn": false, "reason": "no_tokens_line" }));
            return finish(false, args.exit_code);
        }
        Some(t) if t.is_empty() => {
            emit(&json!({ "ok": false, "soft_warn": false, "reason": "zero_tokens" }));
            return finish(false, args.exit_code);
        }
        Some(t) => t,
    };

    let mut bank = DetectorBank::new();
    bank.add(Box::new(AttractorFirst128::new()));
    if args.path_a {
        bank.add(Box::new(AttractorLast128::new()));
        bank.add(Box::new(LongStateCollapse::new()));
    }

    // The attractor detectors consume `Event::Committed` (token ids) and do
    // their own EOT trim + short-window suppression — feed the raw stream.
    for (pos, tok_id) in toks.iter().enumerate() {
        bank.observe(&Event::Committed {
            tok_id: *tok_id,
            pos,
            t_ms: pos as u64,
        });
    }
    bank.observe(&Event::Done {
        total_tokens: toks.len(),
        total_visible_bytes: 0,
        wall_ms: 0,
        ttft_ms: 0,
    });

    let mut hard_fails = 0usize;
    let mut soft_warns = 0usize;
    let detectors: Vec<Value> = bank
        .finalize()
        .into_iter()
        .map(|(name, verdict)| {
            let (status, detail) = match &verdict {
                Verdict::Ok => ("pass", None),
                Verdict::Skip { reason } => ("skip", Some(reason.clone())),
                Verdict::Fired {
                    severity: Severity::Warn,
                    detail,
                } => {
                    soft_warns += 1;
                    ("warn", Some(detail.clone()))
                }
                Verdict::Fired {
                    severity: Severity::Fail,
                    detail,
                } => {
                    hard_fails += 1;
                    ("fail", Some(detail.clone()))
                }
            };
            json!({ "detector": name, "status": status, "detail": detail })
        })
        .collect();

    let ok = hard_fails == 0;
    // soft_warn mirrors the old DETECT_PY: a soft signal only when not already
    // a hard fail.
    let soft_warn = ok && soft_warns > 0;
    emit(&json!({
        "ok": ok,
        "soft_warn": soft_warn,
        "hard_fails": hard_fails,
        "soft_warns": soft_warns,
        "total": toks.len(),
        "detectors": detectors,
    }));
    finish(ok, args.exit_code)
}

fn finish(ok: bool, exit_code: bool) -> anyhow::Result<()> {
    if exit_code && !ok {
        std::process::exit(1);
    }
    Ok(())
}

/// AR vs DFlash token-parity comparison (replaces the gate's PARITY_PY). Reads
/// the `AR tokens:` line from `ar_path` and the `DFlash tokens:` line from
/// `df_path`, then reports exact-equality with a first-mismatch window.
fn run_parity(ar_path: &std::path::Path, df_path: &std::path::Path, exit_code: bool) -> anyhow::Result<()> {
    // Match PARITY_PY: read bytes, lossy-decode, grep the labelled line.
    let ar_txt = String::from_utf8_lossy(&std::fs::read(ar_path)?).into_owned();
    let df_txt = String::from_utf8_lossy(&std::fs::read(df_path)?).into_owned();

    let ar = match parse_token_line(&ar_txt, Source::Ar) {
        Some(v) => v,
        None => {
            emit(&json!({ "ok": false, "reason": "missing_ar_tokens" }));
            return finish(false, exit_code);
        }
    };
    let df = match parse_token_line(&df_txt, Source::Dflash) {
        Some(v) => v,
        None => {
            emit(&json!({ "ok": false, "reason": "missing_dflash_tokens" }));
            return finish(false, exit_code);
        }
    };

    let report = parity::compare(&ar, &df);
    emit(&serde_json::to_value(&report).expect("serialize parity report"));
    finish(report.ok(), exit_code)
}

/// Rollback stat-line parsers (replaces the gate's ROLLBACK_REPLAY_PY /
/// VERIFY_GRAPH_PY). Reads `file` (lossy-decoded like the Python) or stdin.
fn run_rollback(kind: RollbackKind, file: Option<&std::path::Path>, exit_code: bool) -> anyhow::Result<()> {
    let text = match file {
        Some(path) => String::from_utf8_lossy(&std::fs::read(path)?).into_owned(),
        None => {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s)?;
            s
        }
    };
    let (value, ok) = match kind {
        RollbackKind::Replay => {
            let r = rollback::replay_parity(&text);
            (serde_json::to_value(&r).expect("serialize replay report"), r.ok())
        }
        RollbackKind::VerifyGraph => {
            let r = rollback::verify_graph(&text);
            (serde_json::to_value(&r).expect("serialize verify_graph report"), r.ok())
        }
    };
    emit(&value);
    finish(ok, exit_code)
}

fn emit(v: &Value) {
    println!("{}", serde_json::to_string(v).expect("serialize verdict"));
}

/// Extract the token-id list from a `DFlash tokens: [..]` / `AR tokens: [..]`
/// line. Mirrors the old `DETECT_PY` regex + `src = m or ar_m` preference.
fn parse_token_line(input: &str, source: Source) -> Option<Vec<u32>> {
    let dflash = Regex::new(r"DFlash tokens: \[([^\]]+)\]").expect("static regex");
    let ar = Regex::new(r"AR tokens: \[([^\]]+)\]").expect("static regex");
    let caps = match source {
        Source::Dflash => dflash.captures(input),
        Source::Ar => ar.captures(input),
        Source::Auto => dflash.captures(input).or_else(|| ar.captures(input)),
    }?;
    let list = caps.get(1)?.as_str();
    Some(
        list.split(',')
            .filter_map(|s| s.trim().parse::<u32>().ok())
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_dflash_line() {
        let s = "noise\nDFlash tokens: [1, 2, 3, 4]\nmore";
        assert_eq!(parse_token_line(s, Source::Auto), Some(vec![1, 2, 3, 4]));
    }

    #[test]
    fn auto_prefers_dflash_over_ar() {
        let s = "AR tokens: [9, 9]\nDFlash tokens: [1, 2, 3]";
        assert_eq!(parse_token_line(s, Source::Auto), Some(vec![1, 2, 3]));
    }

    #[test]
    fn ar_source_reads_ar_line() {
        let s = "AR tokens: [9, 8, 7]\nDFlash tokens: [1, 2, 3]";
        assert_eq!(parse_token_line(s, Source::Ar), Some(vec![9, 8, 7]));
    }

    #[test]
    fn missing_line_is_none() {
        assert_eq!(parse_token_line("no tokens here", Source::Auto), None);
    }
}
