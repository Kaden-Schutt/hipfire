// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! coherence_probe — user-facing model behavior debugger.
//!
//! Spawns the daemon as a child process, drives a prompt through it,
//! parses the JSONL output stream, runs the `hipfire-detect` detectors
//! live, and prints a structured report. Strictly observational:
//! detectors never block, mutate, or interfere with generation.
//!
//! The daemon emits `{"type":"committed",...}` events alongside the
//! existing `{"type":"token","text":"..."}` events when
//! `HIPFIRE_EMIT_TOKEN_IDS=1` is in its environment — the probe sets
//! that on the daemon child it spawns.
//!
//! Usage:
//!     coherence_probe --model PATH --prompt-file PATH \
//!         [--system PATH] [--max-tokens N] [--temperature F] \
//!         [--report-json OUT.json] [--agentic] [--stall-tokens N] \
//!         [--detect-timing] [--no-strip-think] \
//!         [--max-seq N]
//!     coherence_probe --self-check
//!
//! Exit codes:
//!     0  every detector OK or only soft warnings
//!     1  one or more hard fails (or self-check miss)
//!     2  build / env / I/O error
//!
//! Example end-to-end:
//!     coherence_probe \
//!         --model ~/.hipfire/models/qwen3.6-27b.mq4 \
//!         --prompt-file benchmarks/prompts/lru_cache_pep8_strict.txt \
//!         --max-tokens 200 --temperature 0.0
//!
//! Self-check (no GPU needed):
//!     coherence_probe --self-check

use hipfire_detect::{self_check, Severity, Verdict};

#[derive(Debug, Default)]
struct Args {
    model: Option<String>,
    prompt_file: Option<String>,
    system: Option<String>,
    max_tokens: Option<usize>,
    temperature: Option<f64>,
    repeat_penalty: Option<f64>,
    repeat_window: Option<usize>,
    max_seq: Option<usize>,
    state: Option<String>,
    report_json: Option<String>,
    agentic: bool,
    stall_tokens: Option<usize>,
    detect_timing: bool,
    no_strip_think: bool,
    self_check: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => args.model = it.next(),
            "--prompt-file" => args.prompt_file = it.next(),
            "--system" => args.system = it.next(),
            "--max-tokens" => {
                args.max_tokens = it.next().and_then(|v| v.parse().ok());
            }
            "--temperature" => {
                args.temperature = it.next().and_then(|v| v.parse().ok());
            }
            "--repeat-penalty" => {
                args.repeat_penalty = it.next().and_then(|v| v.parse().ok());
            }
            "--repeat-window" => {
                args.repeat_window = it.next().and_then(|v| v.parse().ok());
            }
            "--max-seq" => {
                args.max_seq = it.next().and_then(|v| v.parse().ok());
            }
            "--state" => args.state = it.next(),
            "--report-json" => args.report_json = it.next(),
            "--agentic" => args.agentic = true,
            "--stall-tokens" => {
                args.stall_tokens = it.next().and_then(|v| v.parse().ok());
            }
            "--detect-timing" => args.detect_timing = true,
            "--no-strip-think" => args.no_strip_think = true,
            "--self-check" => args.self_check = true,
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {}", other)),
        }
    }
    Ok(args)
}

fn print_help() {
    eprintln!(
        "coherence_probe - user-facing model behavior debugger\n\n\
        Usage:\n  \
          coherence_probe --model PATH --prompt-file PATH [flags...]\n  \
          coherence_probe --self-check\n\n\
        Flags:\n  \
          --model PATH          model file (.mq3/.mq4/.hfq, etc.)\n  \
          --prompt-file PATH    user prompt file\n  \
          --system PATH         optional system-prompt file\n  \
          --max-tokens N        max generated tokens (default 200)\n  \
          --temperature F       sampling temperature (default 0.0)\n  \
          --repeat-penalty F    repeat penalty passed to daemon\n  \
          --repeat-window N     repeat penalty window passed to daemon\n  \
          --max-seq N           daemon max_seq override (default 4096)\n  \
          --state q8|fp32|q4       DeltaNet state mode to request from daemon\n  \
          --report-json OUT     also write the report as JSON\n  \
          --agentic             auto-engage tool-call shape detector\n  \
          --stall-tokens N      enable think_stall detector with budget N\n  \
          --detect-timing       enable per-token step-time spike detector\n  \
          --no-strip-think      ask daemon to leave <think> bytes intact\n  \
          --self-check          run synthetic+replay self-check (no GPU needed)\n"
    );
}

fn read_text(path: &str) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("read {}: {}", path, e))
}

fn run_self_check() -> Result<(), Vec<String>> {
    let mut errors: Vec<String> = Vec::new();
    let r = self_check::run_full();

    // Phase A — synthetic payloads.
    let mut a_pass = 0;
    for (name, ok, detail) in &r.phase_a {
        if *ok {
            a_pass += 1;
        } else {
            errors.push(format!("Phase A miss: {} ({})", name, detail));
        }
    }
    eprintln!(
        "self-check Phase A: {} / {} detectors fired correctly",
        a_pass,
        r.phase_a.len()
    );

    // Phase B — captured-JSONL replay against shipped fixtures.
    let mut b_pass = 0;
    for (label, ok, detail) in &r.phase_b {
        if *ok {
            b_pass += 1;
            eprintln!("self-check Phase B: {} {}", label, detail);
        } else {
            errors.push(format!("Phase B miss: {} ({})", label, detail));
        }
    }
    eprintln!(
        "self-check Phase B: {} / {} fixtures replayed correctly",
        b_pass,
        r.phase_b.len()
    );

    if errors.is_empty() {
        eprintln!(
            "self-check passed: Phase A {} / {} synthetic + Phase B {} / {} replay",
            a_pass,
            r.phase_a.len(),
            b_pass,
            r.phase_b.len()
        );
        Ok(())
    } else {
        Err(errors)
    }
}

fn run() -> Result<i32, String> {
    let args = parse_args().map_err(|e| {
        print_help();
        e
    })?;

    if args.self_check {
        return match run_self_check() {
            Ok(()) => Ok(0),
            Err(errs) => {
                for e in errs {
                    eprintln!("{}", e);
                }
                Ok(2)
            }
        };
    }

    let model = args.model.clone().ok_or("--model required")?;
    let prompt_path = args.prompt_file.clone().ok_or("--prompt-file required")?;
    let prompt = read_text(&prompt_path)?;
    let system = match args.system.as_deref() {
        Some(p) => Some(read_text(p)?),
        None => None,
    };
    let prompt_label = format!(
        "{}{}",
        prompt_path,
        args.system
            .as_deref()
            .map(|s| format!(" + {}", s))
            .unwrap_or_default()
    );
    let mut profile = hipfire_runtime::coherence_runtime::DetectorProfile::default_for_prompt(
        &prompt,
        system.as_deref(),
    );
    profile.agentic |= args.agentic;
    profile.stall_tokens = args.stall_tokens;
    profile.detect_timing = args.detect_timing;

    let run_config = hipfire_runtime::coherence_runtime::CoherenceRunConfig {
        model,
        prompt,
        prompt_label,
        system,
        max_tokens: args.max_tokens.unwrap_or(200),
        temperature: args.temperature.unwrap_or(0.0),
        repeat_penalty: args.repeat_penalty,
        repeat_window: args.repeat_window,
        max_seq: args.max_seq.unwrap_or(4096),
        state: args.state.clone(),
        profile,
    };
    let output = hipfire_runtime::coherence_runtime::run_coherence(&run_config)?;

    // Markdown to stdout.
    println!("{}", output.report.to_markdown());

    // Optional JSON.
    if let Some(p) = &args.report_json {
        std::fs::write(
            p,
            serde_json::to_string_pretty(&output.artifact_value())
                .map_err(|e| format!("encode json: {}", e))?,
        )
        .map_err(|e| format!("write json: {}", e))?;
        eprintln!("[probe] json report: {}", p);
    }

    let exit = if output.report.hard_fails > 0 {
        1
    } else if output.report.rows.iter().any(|r| {
        matches!(
            r.verdict,
            Verdict::Fired {
                severity: Severity::Warn,
                ..
            }
        )
    }) {
        // Soft warns alone don't fail the exit code per plan.
        0
    } else {
        0
    };
    Ok(exit)
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("[probe] {}", e);
            std::process::exit(2);
        }
    }
}
