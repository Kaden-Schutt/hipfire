// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Emulated-EP2 parity CLI (STEP-002 Task 8, Phase 2B): thin argument
//! parsing + one call into the high-level qwen35 harness
//! (`hipfire_arch_qwen35::ep2_harness::run`).  No store / tensor / raw
//! ownership types are touched here; all GPU work lives in the arch crate.
//!
//! Usage:
//! ```text
//! cargo run --release --features deltanet,emulated-ep2-harness \
//!     --example ep_decode_parity -- MODEL.mq4 [options]
//!
//!   --prompt-file PATH        prompt file (default benchmarks/prompts/qwen35_moe_ep_parity.txt)
//!   --prompt "TEXT"           inline prompt (overrides --prompt-file)
//!   --steps N                 greedy decode steps compared lockstep (default 16)
//!   --probe                   PROBE mode: NON-ACCEPTANCE (default; deltas reported only)
//!   --max-logit-delta D       ACCEPTANCE mode: finite positive pinned max-abs-logit-delta
//!                             (mutually exclusive with --probe)
//!   --kv-mode MODE            KV cache mode (q8 required; any other resolution refused)
//!   --state-quant fp32        DeltaNet state quant (default fp32; other values refused)
//!   --suffix "TEXT"           deterministic second-turn suffix (default "\n\n...")
//!   --max-seq N               KV max sequence length (default 4096)
//! ```
//!
//! Determinism gates: `HIPFIRE_DETERMINISTIC=1` and `HIPFIRE_GRAPH=0` are
//! MANDATORY — there is no bypass.
//!
//! Naturally SINGLE-SHOT: this CLI calls `run` exactly once and exits.  The
//! harness itself is single-shot per process (see its STEP-002R debt note);
//! there is no repeated-load / lifecycle claim here.
//!
//! Exit codes: 0 = acceptance pass OR probe report (probe is visibly
//! NON-ACCEPTANCE and never counts as a pass); 1 = acceptance failure;
//! 2 = harness/config error.

use std::path::PathBuf;
use std::process::ExitCode;

fn usage() -> ! {
    eprintln!(
        "usage: ep_decode_parity MODEL.mq4 [--prompt-file PATH | --prompt TEXT] \
         [--steps N] [--probe | --max-logit-delta D] [--kv-mode MODE] \
         [--state-quant fp32] [--suffix TEXT] [--max-seq N]"
    );
    std::process::exit(2);
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        usage();
    }

    let mut options = hipfire_arch_qwen35::ep2_harness::Ep2HarnessOptions::default();
    let mut probe = false;
    let mut prompt_file: Option<PathBuf> = None;
    let mut i = 0usize;
    options.model_path = PathBuf::from(&args[0]);
    i += 1;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt-file" => {
                i += 1;
                prompt_file = Some(PathBuf::from(args.get(i).unwrap_or_else(|| usage())));
            }
            "--prompt" => {
                i += 1;
                options.prompt = args.get(i).unwrap_or_else(|| usage()).clone();
            }
            "--steps" => {
                i += 1;
                options.max_steps = args
                    .get(i)
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage());
            }
            "--probe" => {
                if options.max_logit_delta.is_some() {
                    eprintln!("--probe conflicts with --max-logit-delta");
                    usage();
                }
                probe = true;
            }
            "--max-logit-delta" => {
                if probe {
                    eprintln!("--max-logit-delta conflicts with --probe");
                    usage();
                }
                i += 1;
                options.max_logit_delta = Some(
                    args.get(i)
                        .unwrap_or_else(|| usage())
                        .parse()
                        .unwrap_or_else(|_| usage()),
                );
            }
            "--kv-mode" => {
                i += 1;
                options.kv_mode = args.get(i).unwrap_or_else(|| usage()).clone();
            }
            "--state-quant" => {
                i += 1;
                options.state_quant = args.get(i).unwrap_or_else(|| usage()).clone();
            }
            "--suffix" => {
                i += 1;
                options.second_turn_suffix = args.get(i).unwrap_or_else(|| usage()).clone();
            }
            "--max-seq" => {
                i += 1;
                options.max_seq = args
                    .get(i)
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage());
            }
            other => {
                eprintln!("unknown argument: {other}");
                usage();
            }
        }
        i += 1;
    }

    if options.prompt.is_empty() {
        let path = prompt_file
            .unwrap_or_else(|| PathBuf::from("benchmarks/prompts/qwen35_moe_ep_parity.txt"));
        options.prompt = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) => {
                eprintln!("cannot read prompt file {}: {e}", path.display());
                std::process::exit(2);
            }
        };
    }
    if options.prompt.trim().is_empty() {
        eprintln!("prompt must be non-empty");
        std::process::exit(2);
    }

    eprintln!("== emulated-EP2 parity ==");
    eprintln!(
        "model: {} | steps: {} | mode: {} | kv: {} | state_quant: {}",
        options.model_path.display(),
        options.max_steps,
        match options.max_logit_delta {
            Some(d) => format!("acceptance (pinned max-logit-delta {d})"),
            None => "PROBE (NON-ACCEPTANCE)".into(),
        },
        options.kv_mode,
        options.state_quant,
    );

    match hipfire_arch_qwen35::ep2_harness::run(&options) {
        Ok(report) => {
            println!("mode: {}", ep2_mode_label(report.mode));
            println!("passed: {}", report.passed);
            println!("finite_logits: {}", report.finite_logits);
            println!("first_token_match: {}", report.first_token_match);
            println!("generated_tokens_match: {}", report.generated_tokens_match);
            println!("second_turn_match: {}", report.second_turn_match);
            println!("reset_match: {}", report.reset_match);
            println!("max_abs_logit_delta: {}", report.max_abs_logit_delta);
            println!("resolved_kv_mode: {}", report.resolved_kv_mode);
            match report.first_delta_pos {
                Some(pos) => {
                    println!("first_delta_pos: {pos}");
                    println!("first_delta_index: {:?}", report.first_delta_index);
                    println!("baseline_logit: {:?}", report.baseline_logit);
                    println!("ep2_logit: {:?}", report.ep2_logit);
                }
                None => println!("first_delta_pos: none"),
            }
            println!("baseline_tokens: {:?}", report.baseline_tokens);
            println!("ep2_tokens: {:?}", report.ep2_tokens);
            let is_pass = report.mode == hipfire_arch_qwen35::ep2_harness::Ep2Mode::Acceptance
                && report.passed;
            if report.mode == hipfire_arch_qwen35::ep2_harness::Ep2Mode::Probe {
                println!("RESULT: PROBE (NON-ACCEPTANCE) — deltas reported, NOTHING accepted");
                ExitCode::SUCCESS
            } else if is_pass {
                println!("RESULT: ACCEPTANCE PASS");
                ExitCode::SUCCESS
            } else {
                println!("RESULT: ACCEPTANCE FAIL");
                ExitCode::from(1)
            }
        }
        Err(e) => {
            eprintln!("ep2 parity harness error: {e}");
            ExitCode::from(2)
        }

    }
}

fn ep2_mode_label(mode: hipfire_arch_qwen35::ep2_harness::Ep2Mode) -> &'static str {
    match mode {
        hipfire_arch_qwen35::ep2_harness::Ep2Mode::Probe => "probe",
        hipfire_arch_qwen35::ep2_harness::Ep2Mode::Acceptance => "acceptance",
    }
}
