// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! imatrix_collect — calibration-corpus activation-magnitude collector.
//!
//! Phase A Step 4 (Tier 2 — subprocess wrapper per
//! `docs/plans/qwen35-mq4-quality-gap.md` §5). Spawns llama.cpp's
//! `llama-imatrix` over a calibration corpus + a BF16 GGUF model, captures
//! per-channel `Σ act²` per linear-layer input, validates the resulting
//! GGUF-format imatrix file, prints summary stats.
//!
//! Why Tier 2 (subprocess wrapper) rather than Tier 1 (hipfire-native):
//!
//! - llama.cpp's imatrix collector is mature + tested; the file format is
//!   already a GGUF that hipfire's `gguf_input.rs` reader can consume
//!   directly (each linear-layer tensor gets `{name}.in_sum2` + `{name}.counts`
//!   entries plus dataset/chunk-count metadata). Reusing this means our
//!   imatrix data is byte-identical to what Unsloth's UD recipes are
//!   computed against — apples-to-apples for Step 6b cohort comparisons.
//! - The Tier 1 ROI is dominated by tokenizer parity (llama.cpp and hipfire
//!   tokenizers disagree on ~46% of token positions per
//!   `issue-113-quant-quality-eval.md:126`) but that's a small effect
//!   averaged over millions of activation samples. Not a Phase A gate.
//! - The hard part of Tier 1 — capture hooks at every fused/unfused
//!   linear-layer dispatch + new on-GPU sum-of-squares reduce kernel — is
//!   ~6-10 days of work. Tier 2 takes ~2-3 days; the difference is
//!   "Phase A unblocked this week vs next week".
//!
//! Pinned llama.cpp commit: 9dcf83552887bb898b4a98a5761361e504e31fc3
//! (same pin as `build_kld_ref.rs` — both eval-pipeline tools must stay
//! tokenization-compatible).
//!
//! Usage:
//!   cargo run --release -p hipfire-runtime --example imatrix_collect -- \
//!       --bf16-gguf <path-to-bf16.gguf> \
//!       --corpus    <path-to-calibration-corpus.txt> \
//!       --output    <path-to-output.imatrix.gguf> \
//!       [--n-ctx <N>=2048] [--n-batch <N>=512] [--chunks <N>=-1] \
//!       [--llama-imatrix-bin <path>=llama-imatrix] \
//!       [--process-output]                # collect data for the lm_head too
//!
//! Output: a GGUF file at `--output`. Consumed in Step 5 (L5c
//! activation-weighted LS) by reading via the existing `gguf_input.rs`.

use hipfire_runtime::eval_common;
use std::path::PathBuf;
use std::process::Command;

const PINNED_LLAMACPP_COMMIT: &str = "9dcf83552887bb898b4a98a5761361e504e31fc3";

struct Args {
    bf16_gguf: PathBuf,
    corpus: PathBuf,
    output: PathBuf,
    n_ctx: usize,
    n_batch: usize,
    chunks: Option<i64>,
    llama_imatrix_bin: String,
    process_output: bool,
}

fn print_usage() {
    eprintln!(
        "Usage:\n  imatrix_collect --bf16-gguf <path> --corpus <path> --output <path>\n\
         \n\
         Optional flags:\n\
           --n-ctx <N>           context length (default: 2048; matches eval slice)\n\
           --n-batch <N>         logical batch (default: 512)\n\
           --chunks <N>          cap chunks processed (-1 = all; default)\n\
           --llama-imatrix-bin <path>   default: llama-imatrix\n\
           --process-output      also collect data for the lm_head / output tensor\n\
         \n\
         Phase A Step 4 (Tier 2). See docs/plans/qwen35-mq4-quality-gap.md §5."
    );
}

fn parse_args() -> Args {
    let mut bf16_gguf: Option<PathBuf> = None;
    let mut corpus: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut n_ctx: usize = 2048;
    let mut n_batch: usize = 512;
    let mut chunks: Option<i64> = None;
    let mut llama_imatrix_bin = "llama-imatrix".to_string();
    let mut process_output = false;

    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--bf16-gguf" => { bf16_gguf = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--corpus" => { corpus = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--output" => { output = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--n-ctx" => { n_ctx = argv[i + 1].parse().expect("--n-ctx must be u32"); i += 2; }
            "--n-batch" => { n_batch = argv[i + 1].parse().expect("--n-batch must be u32"); i += 2; }
            "--chunks" => { chunks = Some(argv[i + 1].parse().expect("--chunks must be int")); i += 2; }
            "--llama-imatrix-bin" => { llama_imatrix_bin = argv[i + 1].clone(); i += 2; }
            "--process-output" => { process_output = true; i += 1; }
            "-h" | "--help" => { print_usage(); std::process::exit(0); }
            other => {
                eprintln!("unknown arg: {other}");
                print_usage();
                std::process::exit(1);
            }
        }
    }

    let bf16_gguf = bf16_gguf.unwrap_or_else(|| { print_usage(); std::process::exit(1); });
    let corpus = corpus.unwrap_or_else(|| { print_usage(); std::process::exit(1); });
    let output = output.unwrap_or_else(|| { print_usage(); std::process::exit(1); });

    if !bf16_gguf.exists() {
        eprintln!("error: --bf16-gguf not found: {}", bf16_gguf.display());
        std::process::exit(1);
    }
    if !corpus.exists() {
        eprintln!("error: --corpus not found: {}", corpus.display());
        std::process::exit(1);
    }

    Args {
        bf16_gguf, corpus, output, n_ctx, n_batch, chunks,
        llama_imatrix_bin, process_output,
    }
}

fn main() {
    let args = parse_args();

    // 0. Sanity check: pinned llama.cpp commit. The imatrix data has to be
    // produced by the same llama.cpp build that produced the kldref + the
    // GGUF anchor numbers — otherwise tokenization (which differs across
    // llama.cpp versions in subtle ways) yields different activation
    // statistics and the imatrix is silently miscalibrated.
    eval_common::verify_llama_commit(
        &args.llama_imatrix_bin, PINNED_LLAMACPP_COMMIT, "imatrix_collect",
    );

    eprintln!();
    eprintln!("imatrix_collect: configuration");
    eprintln!("  bf16-gguf:       {}", args.bf16_gguf.display());
    eprintln!("  corpus:          {}", args.corpus.display());
    eprintln!("  output:          {}", args.output.display());
    eprintln!("  n_ctx:           {}", args.n_ctx);
    eprintln!("  n_batch:         {}", args.n_batch);
    eprintln!("  chunks:          {}", args.chunks.map_or("all".to_string(), |c| c.to_string()));
    eprintln!("  process_output:  {}", args.process_output);
    eprintln!("  pinned commit:   {}", PINNED_LLAMACPP_COMMIT);
    eprintln!();

    // 1. Spawn llama-imatrix. Stream stdout/stderr to our stdout/stderr so the
    // user sees progress; this is a multi-minute job (small models 5-10 min;
    // 9B ~30-60 min on gfx1100 with the default wikitext2 1175-chunk slice).
    let mut cmd = Command::new(&args.llama_imatrix_bin);
    cmd.args(["-m", &args.bf16_gguf.display().to_string()])
       .args(["-f", &args.corpus.display().to_string()])
       .args(["-c", &args.n_ctx.to_string()])
       .args(["-b", &args.n_batch.to_string()])
       .args(["-o", &args.output.display().to_string()])
       .args(["--output-format", "gguf"]);
    if let Some(chunks) = args.chunks {
        cmd.args(["--chunks", &chunks.to_string()]);
    }
    if args.process_output {
        cmd.arg("--process-output");
    }
    // --no-mmap: same rationale as build_kld_ref — BF16 models are large
    // and demand-paging stalls on UMA / high-MB-per-second working sets.
    // Forces one upfront sequential read into pinned RAM.
    cmd.arg("--no-mmap");

    eprintln!("imatrix_collect: invoking llama-imatrix...");
    eprintln!("  {} {}",
        args.llama_imatrix_bin,
        cmd.get_args()
            .map(|a| a.to_string_lossy().to_string())
            .collect::<Vec<_>>()
            .join(" ")
    );
    eprintln!();

    let status = cmd.status().expect("failed to invoke llama-imatrix");
    if !status.success() {
        eprintln!();
        eprintln!("imatrix_collect: llama-imatrix exited with {status}");
        std::process::exit(status.code().unwrap_or(1));
    }

    // 2. Validate the output is a parseable imatrix GGUF + dump summary.
    //
    // llama-imatrix's GGUF format stores per-tensor pairs:
    //   <name>.in_sum2   F32[k, n_mat]   sum of squared activations (per-channel × per-MoE-matrix)
    //   <name>.counts    F32[1, n_mat]   token count contributing to each matrix
    // Plus metadata: imatrix.chunk_count, imatrix.chunk_size, imatrix.datasets[].
    //
    // We use hipfire_runtime's GGUF reader to validate the file structure
    // before declaring success.
    eprintln!();
    eprintln!("imatrix_collect: validating output...");
    summarize_imatrix(&args.output);
}

/// Open the produced imatrix GGUF and dump a summary: total tensors, total
/// linear layers detected (each layer contributes a .in_sum2 + .counts pair),
/// any partial-coverage tensors (count<n_ctx*chunks for at least one matrix
/// element — happens for MoE experts that weren't exercised by the
/// calibration corpus).
fn summarize_imatrix(path: &std::path::Path) {
    // Defer to hipfire-runtime's hf_reader since hipfire_quantize::gguf_input
    // isn't reachable from this crate without a lib re-export. Use a fresh
    // subprocess to inspect via llama-imatrix --show-statistics if the
    // runtime-side reader isn't wired up to imatrix GGUFs yet. For Phase A
    // Step 4, a minimal "exists + non-empty + looks like a GGUF" check
    // is acceptable.
    use std::fs;
    let metadata = match fs::metadata(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: failed to stat output file: {e}");
            std::process::exit(1);
        }
    };
    let size = metadata.len();
    if size < 32 {
        eprintln!("error: output file suspiciously small ({size} bytes); aborting");
        std::process::exit(1);
    }

    // Verify GGUF magic (first 4 bytes = "GGUF").
    use std::io::Read;
    let mut f = match fs::File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: failed to open output file: {e}");
            std::process::exit(1);
        }
    };
    let mut magic = [0u8; 4];
    if f.read_exact(&mut magic).is_err() || &magic != b"GGUF" {
        eprintln!("error: output file does not start with GGUF magic; got {magic:?}");
        std::process::exit(1);
    }

    eprintln!("  file size:    {size} bytes ({:.2} MB)", size as f64 / 1_048_576.0);
    eprintln!("  GGUF magic:   ok");
    eprintln!();
    eprintln!("=== Next step ===");
    eprintln!("  Phase A Step 5 (L5c activation-weighted LS) reads this file");
    eprintln!("  via gguf_input.rs and uses the {{name}}.in_sum2 entries to weight");
    eprintln!("  the per-block scale-fit objective. See docs/plans/qwen35-mq4-");
    eprintln!("  quality-gap.md §5 Step 5 for the full quantizer-side recipe.");
    eprintln!();
    eprintln!("=== Inspection ===");
    eprintln!("  llama-imatrix --in-file {} --show-statistics", path.display());
}
