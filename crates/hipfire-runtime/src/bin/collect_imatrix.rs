//! `collect_imatrix` — Tier 1 hipfire-native activation-magnitude collector.
//!
//! Foundation scaffold for the imatrix-collection binary that replaces
//! the Tier 2 subprocess wrapper at `examples/imatrix_collect.rs`
//! (which shells out to `llama-imatrix`). When complete, this binary
//! will:
//!
//! 1. Load a BF16 HuggingFace model directly from `<dir>/*.safetensors`
//!    via `engine::bf16_loader` (Task 4 scaffold in this same series).
//! 2. Run hipfire's MFMA-direct forward path on the calibration corpus
//!    through the new BF16 MFMA GEMM kernel
//!    (`kernels/src/gemm_bf16_mfma.gfx942.hip`).
//! 3. At each linear-layer dispatch site, fire the `ActivationCapture`
//!    hook (Task 3 trait in `rdna-compute::dispatch`) to accumulate
//!    per-channel `Σ act²` on-GPU via a dedicated reduction kernel
//!    (Phase 2, separate task).
//! 4. Write the per-tensor `in_sum2` / `counts` pairs as a GGUF imatrix
//!    file (Phase 2, separate task) that is byte-compatible with
//!    `llama-imatrix --output-format gguf` — so existing readers
//!    (`hipfire-quantize::gguf_input`) work without changes.
//!
//! Target speedup vs Tier 2: 20× (~8h → ~25min for a 27B-class model on
//! MI300x). See `docs/investigations/2026-05-19-tier1-bf16-mfma/README.md`
//! for the foundation-POC validating the BF16 MFMA GEMM building block.
//!
//! Usage (target CLI, mirrors `examples/imatrix_collect.rs` semantics where
//! possible — the `--hf-model <dir>` arg replaces `--bf16-gguf <file>` since
//! Tier 1 reads safetensors directly):
//!
//! ```ignore
//! cargo run --release -p hipfire-runtime --bin collect_imatrix -- \
//!     --hf-model    <path-to-bf16-hf-model-dir> \
//!     --corpus      <path-to-calibration-corpus.txt> \
//!     --output      <path-to-output.imatrix.gguf> \
//!     [--n-ctx 2048] [--n-sequences 128] [--process-output]
//! ```
//!
//! TODO: replace this stdlib-only arg parser with `clap` once the
//! crate accepts a clap workspace-dep. Matched the imatrix_collect.rs
//! example style for now to keep the workspace dep graph clean while
//! the rest of Phase 2 lands.

use std::path::PathBuf;

#[derive(Debug)]
#[allow(dead_code)]
struct Args {
    /// Path to a HuggingFace model directory containing
    /// `model.safetensors[.index.json]` + `config.json` + `tokenizer.json`.
    hf_model: PathBuf,
    /// Plain-text calibration corpus (one document or one file = one
    /// concatenated sequence; the binary chunks into `n_sequences ×
    /// n_ctx` tokens internally). Tier 2 used `wikitext-2-raw-v1` by
    /// default; Tier 1 will mirror that once corpus-loading lands.
    corpus: PathBuf,
    /// Output GGUF path (must end in `.imatrix.gguf` by convention).
    output: PathBuf,
    /// Tokens per calibration sequence.
    n_ctx: usize,
    /// Calibration sequences (matches GPTQ paper's 128-seq scale).
    n_sequences: usize,
    /// Also collect data for the `output` / `lm_head` tensor. Mirrors
    /// llama-imatrix's `--process-output` flag.
    process_output: bool,
}

fn print_usage() {
    eprintln!(
        "Usage:\n  collect_imatrix --hf-model <dir> --corpus <file> --output <gguf>\n\
         \n\
         Optional flags:\n\
           --n-ctx <N>           tokens per calibration sequence (default: 2048)\n\
           --n-sequences <N>     calibration sequences (default: 128)\n\
           --process-output      also collect data for lm_head / output tensor\n\
         \n\
         Tier 1 hipfire-native imatrix collector. See\n\
         docs/investigations/2026-05-19-tier1-bf16-mfma/ for the foundation POC."
    );
}

fn parse_args() -> Args {
    let mut hf_model: Option<PathBuf> = None;
    let mut corpus: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut n_ctx: usize = 2048;
    let mut n_sequences: usize = 128;
    let mut process_output = false;

    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--hf-model" => {
                hf_model = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--corpus" => {
                corpus = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--output" => {
                output = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--n-ctx" => {
                n_ctx = argv[i + 1].parse().expect("--n-ctx must be a positive integer");
                i += 2;
            }
            "--n-sequences" => {
                n_sequences = argv[i + 1]
                    .parse()
                    .expect("--n-sequences must be a positive integer");
                i += 2;
            }
            "--process-output" => {
                process_output = true;
                i += 1;
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                print_usage();
                std::process::exit(1);
            }
        }
    }

    let hf_model = hf_model.unwrap_or_else(|| {
        eprintln!("error: --hf-model is required");
        print_usage();
        std::process::exit(1);
    });
    let corpus = corpus.unwrap_or_else(|| {
        eprintln!("error: --corpus is required");
        print_usage();
        std::process::exit(1);
    });
    let output = output.unwrap_or_else(|| {
        eprintln!("error: --output is required");
        print_usage();
        std::process::exit(1);
    });

    Args {
        hf_model,
        corpus,
        output,
        n_ctx,
        n_sequences,
        process_output,
    }
}

fn main() {
    let args = parse_args();
    eprintln!("collect_imatrix (Tier 1 — foundation scaffold)");
    eprintln!("  hf-model:       {}", args.hf_model.display());
    eprintln!("  corpus:         {}", args.corpus.display());
    eprintln!("  output:         {}", args.output.display());
    eprintln!("  n-ctx:          {}", args.n_ctx);
    eprintln!("  n-sequences:    {}", args.n_sequences);
    eprintln!("  process-output: {}", args.process_output);
    eprintln!();
    eprintln!("Implementation pending — this is the Phase 1 scaffold.");
    eprintln!("Next deliverables (separate subagent tasks):");
    eprintln!("  - BF16 safetensors loader → device tensors");
    eprintln!("  - On-GPU Σx² reduction kernel + ActivationCapture wiring");
    eprintln!("  - GGUF imatrix writer (byte-compatible with llama-imatrix)");
    eprintln!();
    unimplemented!(
        "collect_imatrix Tier 1 forward pass + Σx² + GGUF write not yet \
         implemented. See `examples/imatrix_collect.rs` for the Tier 2 \
         subprocess fallback."
    );
}
