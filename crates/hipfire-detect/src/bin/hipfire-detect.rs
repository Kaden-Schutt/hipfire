// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Thin CLI over `hipfire_detect::DetectorBank`.
//!
//! Reads generated text or a daemon JSONL event stream, runs the full
//! detector bank, and prints the existing `Report` JSON on stdout.
//! No detection logic lives here — thresholds stay in the library.
//!
//! Usage:
//!     hipfire-detect [--jsonl] [--input FILE]
//!     cat out.txt | hipfire-detect
//!     hipfire-detect --jsonl --input capture.jsonl
//!
//! Exit codes:
//!     0  OK or soft warnings only
//!     1  one or more hard fails
//!     2  I/O or usage error

use hipfire_detect::{
    attractor::{AttractorFirst128, AttractorLast128},
    eos_immediate::EosImmediate,
    ngram::{LoopGuardMirror, NgramDensity},
    report::{prompt_md5, Report, ReportHeader},
    self_check::{parse_jsonl_events, replay, OwnedEventPub},
    special_leak::SpecialLeak,
    think::ThinkEmpty,
    toolcall::ToolcallShape,
    whitespace_only::WhitespaceOnly,
    DetectorBank,
};
use std::env;
use std::io::{self, Read};
use std::process;

fn build_full_bank() -> DetectorBank {
    let mut bank = DetectorBank::new();
    bank.add(Box::new(AttractorFirst128::new()));
    bank.add(Box::new(AttractorLast128::new()));
    bank.add(Box::new(NgramDensity::new()));
    bank.add(Box::new(LoopGuardMirror::new()));
    bank.add(Box::new(ThinkEmpty::new()));
    bank.add(Box::new(SpecialLeak::new()));
    bank.add(Box::new(ToolcallShape::new()));
    bank.add(Box::new(EosImmediate::new()));
    bank.add(Box::new(WhitespaceOnly::new()));
    bank
}

fn text_to_events(text: &str) -> Vec<OwnedEventPub> {
    // Plain generated text has no token ids — feed one Token event so
    // text-level detectors (attractor windows, special leak, think,
    // toolcall, whitespace, eos-immediate) still see the payload.
    let mut events = Vec::with_capacity(2);
    events.push(OwnedEventPub::Token {
        text: text.to_string(),
        t_ms: 0,
        synthetic: false,
    });
    events.push(OwnedEventPub::Done {
        total_tokens: 0,
        total_visible_bytes: text.len(),
        wall_ms: 0,
        ttft_ms: 0,
    });
    events
}

fn read_input(path: Option<&str>) -> Result<String, String> {
    match path {
        Some("-") | None => {
            let mut buf = String::new();
            io::stdin()
                .read_to_string(&mut buf)
                .map_err(|e| format!("read stdin: {e}"))?;
            Ok(buf)
        }
        Some(p) => std::fs::read_to_string(p).map_err(|e| format!("read {p}: {e}")),
    }
}

fn usage() -> ! {
    eprintln!(
        "hipfire-detect — offline DetectorBank runner\n\n\
         Usage:\n  \
           hipfire-detect [--jsonl] [--input FILE]\n\n\
         Reads generated text (default) or daemon JSONL (--jsonl) from\n\
         --input FILE or stdin. Prints Report JSON on stdout.\n\
         Exit 0 = OK/WARN, 1 = hard FAIL, 2 = usage/I/O error.\n"
    );
    process::exit(2);
}

fn main() {
    let mut jsonl = false;
    let mut input: Option<String> = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--jsonl" => jsonl = true,
            "--input" => {
                input = Some(args.next().unwrap_or_else(|| {
                    eprintln!("--input requires a path");
                    process::exit(2);
                }));
            }
            "-h" | "--help" => usage(),
            other => {
                eprintln!("unknown arg: {other}");
                usage();
            }
        }
    }

    let body = match read_input(input.as_deref()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            process::exit(2);
        }
    };

    let events = if jsonl {
        parse_jsonl_events(&body)
    } else {
        text_to_events(&body)
    };

    let mut bank = build_full_bank();
    let finals = replay(&mut bank, &events);

    let total_tokens = events
        .iter()
        .find_map(|e| match e {
            OwnedEventPub::Done { total_tokens, .. } => Some(*total_tokens),
            _ => None,
        })
        .unwrap_or(0);
    let ttft_ms = events
        .iter()
        .find_map(|e| match e {
            OwnedEventPub::Done { ttft_ms, .. } => Some(*ttft_ms),
            _ => None,
        })
        .unwrap_or(0);

    let header = ReportHeader {
        prompt_md5: prompt_md5(b""),
        prompt_label: if jsonl {
            "jsonl-replay".into()
        } else {
            "plain-text".into()
        },
        model: String::new(),
        arch: String::new(),
        host: String::new(),
        total_tokens,
        tok_s: 0.0,
        gen_tok_s: 0.0,
        ttft_ms,
        daemon_prefill_ms: 0.0,
        daemon_prefill_tok_s: 0.0,
        daemon_decode_tok_s: 0.0,
        daemon_ttft_ms: 0.0,
        daemon_tok_s: 0.0,
    };

    let report = Report::new(header, finals);
    println!("{}", report.to_json());

    if report.hard_fails > 0 {
        process::exit(1);
    }
}
