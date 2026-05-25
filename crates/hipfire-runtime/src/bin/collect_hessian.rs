//! `collect_hessian` — Tier 1 hipfire-native Hessian collector for GPTQ.
//!
//! Foundation scaffold for the Hessian-collection binary that replaces
//! the Tier 2 PyTorch wrapper at `scripts/collect_hessian.py`. When
//! complete, this binary will:
//!
//! 1. Load a BF16 HuggingFace model directly from `<dir>/*.safetensors`
//!    via `engine::bf16_loader` (Task 4 scaffold in this same series).
//! 2. Run hipfire's MFMA-direct forward path on the calibration corpus
//!    through the new BF16 MFMA GEMM kernel
//!    (`kernels/src/gemm_bf16_mfma.gfx942.hip`).
//! 3. At each `nn.Linear`-equivalent dispatch site that matches the
//!    GPTQ target list (mirrors `collect_hessian.py:80-92`), fire the
//!    `ActivationCapture` hook (Task 3 trait in `rdna-compute::dispatch`)
//!    to accumulate a per-tensor `H_t = (1/N) * Σ_t x_t · x_t^T`
//!    outer-product Hessian on-GPU via a dedicated K×K rank-1-update
//!    kernel (Phase 2, separate task).
//! 4. Write per-tensor `K×K F32` blocks to a HFHS-v1 binary file
//!    (Phase 2, separate task; byte-compatible with the existing format
//!    documented in `scripts/collect_hessian.py:25-43`) so that
//!    `crates/hipfire-quantize/src/hessian_io.rs` reads it unchanged.
//!
//! Target speedup vs Tier 2: 20× (~8h → ~25min for a 27B-class model on
//! MI300x). The Python path is currently bottlenecked on HF transformers
//! eager forward + CPU-side `x.T @ x` accumulator with the BF16→FP32
//! cast + per-token `.cpu()` PCIe transfer in
//! `scripts/collect_hessian.py:213-222`.
//!
//! Tier 1 keeps the outer-product accumulator on-GPU for the full
//! calibration pass; final HFHS dump is the only host-side step.
//!
//! Usage (target CLI):
//!
//! ```ignore
//! cargo run --release -p hipfire-runtime --bin collect_hessian -- \
//!     --hf-model    <path-to-bf16-hf-model-dir> \
//!     --corpus      <path-to-corpus.txt-or-hf-dataset-id> \
//!     --output      <path-to-out.hessian.bin> \
//!     [--n-sequences 128] [--ctx-len 2048] [--n-passes 1]
//! ```
//!
//! TODO: replace this stdlib-only arg parser with `clap` once the
//! workspace accepts the dep. Matched the imatrix_collect.rs example
//! style for now to keep the workspace dep graph clean.

use std::path::PathBuf;

#[derive(Debug)]
#[allow(dead_code)]
struct Args {
    /// HuggingFace BF16 model dir (`*.safetensors` + `config.json`).
    hf_model: PathBuf,
    /// Plain-text corpus file OR HuggingFace dataset id
    /// (e.g. `wikitext-2-raw-v1`). The Tier 2 Python path defaults to
    /// `wikitext`; we mirror that once corpus-loading lands.
    corpus: PathBuf,
    /// Output HFHS-v1 binary path (`*.hessian.bin` by convention).
    /// Consumed by `crates/hipfire-quantize/src/hessian_io.rs` at
    /// GPTQ quantize-time.
    output: PathBuf,
    /// Number of calibration sequences (GPTQ paper default: 128).
    n_sequences: usize,
    /// Tokens per calibration sequence (GPTQ paper default: 2048).
    ctx_len: usize,
    /// Number of full passes over the calibration corpus. Default 1.
    /// Multiple passes are useful for noisy GPTQ convergence on MoE
    /// models with sparse expert activations.
    n_passes: usize,
}

fn print_usage() {
    eprintln!(
        "Usage:\n  collect_hessian --hf-model <dir> --corpus <file-or-hf-id> --output <bin>\n\
         \n\
         Optional flags:\n\
           --n-sequences <N>     calibration sequences (default: 128)\n\
           --ctx-len <N>         tokens per calibration sequence (default: 2048)\n\
           --n-passes <N>        passes over the corpus (default: 1)\n\
         \n\
         Tier 1 hipfire-native Hessian collector. See\n\
         docs/investigations/2026-05-19-tier1-bf16-mfma/ for the foundation POC.\n\
         Output format: HFHS v1 (see scripts/collect_hessian.py:25-43)."
    );
}

fn parse_args() -> Args {
    let mut hf_model: Option<PathBuf> = None;
    let mut corpus: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut n_sequences: usize = 128;
    let mut ctx_len: usize = 2048;
    let mut n_passes: usize = 1;

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
            "--n-sequences" => {
                n_sequences = argv[i + 1]
                    .parse()
                    .expect("--n-sequences must be a positive integer");
                i += 2;
            }
            "--ctx-len" => {
                ctx_len = argv[i + 1]
                    .parse()
                    .expect("--ctx-len must be a positive integer");
                i += 2;
            }
            "--n-passes" => {
                n_passes = argv[i + 1]
                    .parse()
                    .expect("--n-passes must be a positive integer");
                i += 2;
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
        n_sequences,
        ctx_len,
        n_passes,
    }
}

fn main() {
    let args = parse_args();
    eprintln!("collect_hessian (Tier 1 — foundation scaffold)");
    eprintln!("  hf-model:    {}", args.hf_model.display());
    eprintln!("  corpus:      {}", args.corpus.display());
    eprintln!("  output:      {}", args.output.display());
    eprintln!("  n-sequences: {}", args.n_sequences);
    eprintln!("  ctx-len:     {}", args.ctx_len);
    eprintln!("  n-passes:    {}", args.n_passes);
    eprintln!();
    eprintln!("Implementation pending — this is the Phase 1 scaffold.");
    eprintln!("Next deliverables (separate subagent tasks):");
    eprintln!("  - BF16 safetensors loader → device tensors");
    eprintln!("  - On-GPU K×K outer-product Hessian rank-1-update kernel");
    eprintln!("  - ActivationCapture wiring at the GPTQ-target dispatch sites");
    eprintln!("  - HFHS-v1 binary writer (matches scripts/collect_hessian.py)");
    eprintln!();
    unimplemented!(
        "collect_hessian Tier 1 forward pass + outer-product Hessian + HFHS \
         write not yet implemented. See `scripts/collect_hessian.py` for \
         the Tier 2 PyTorch fallback."
    );
}
