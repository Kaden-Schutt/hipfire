//! Standalone forward-pass driver for `hipfire-arch-qwen2`.
//!
//! This binary is the bring-up validation harness called out in §6 R3
//! of `docs/plans/qwen_2.0_vlm_plus_dots_ocr.md`. It bypasses the
//! daemon entirely — no `arch_id`-based dispatch, no `LoadedModel`
//! plumbing — so the forward pass can be validated against the HF
//! reference at `benchmarks/references/qwen2_1p5b_instruct_smoke.json`
//! without phase-3 daemon wiring landing first.
//!
//! Pipeline (in implementation order):
//!
//! 1. Load HFQ → `Qwen2Config` + `Qwen2Weights` via the
//!    [`hipfire_runtime::arch::Architecture`] trait. **Done in rev 2.**
//! 2. Build [`hipfire_runtime::tokenizer::Tokenizer`] from the HFQ's
//!    embedded `tokenizer.json` blob. **Done in rev 2.**
//! 3. Encode the prompt and (optionally) compare its token-id sequence
//!    against the reference artifact — a tokenizer-parity check that
//!    can pass before any kernel work lands. **Done in rev 2.**
//! 4. Forward + greedy decode N tokens. **PENDING (forward port).**
//! 5. Compare the generated token IDs against
//!    `first_16_completion_token_ids` in the reference. **PENDING.**
//!
//! Usage:
//!
//! ```text
//! export PATH=/opt/rocm-7.12/bin:$PATH
//! export LD_LIBRARY_PATH=/opt/rocm-7.12/lib:$LD_LIBRARY_PATH
//!
//! cargo run --release --example infer_qwen2 -p hipfire-arch-qwen2 -- \
//!     --hfq /data/cache/hipfire/qwen2-1.5b.arch7.hfq4 \
//!     --prompt-file benchmarks/prompts/qwen2_smoke.txt \
//!     --reference benchmarks/references/qwen2_1p5b_instruct_smoke.json
//! ```
//!
//! Pass `--no-load` to skip GPU weight upload (only exercises config +
//! tokenizer; useful when iterating without a GPU lock).

use std::path::Path;

use hipfire_arch_qwen2::qwen2;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;

#[derive(Default)]
struct Args {
    hfq: Option<String>,
    prompt_file: Option<String>,
    reference: Option<String>,
    no_load: bool,
    max_new_tokens: usize,
}

fn parse_args() -> Args {
    let mut out = Args { max_new_tokens: 16, ..Default::default() };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--hfq" => out.hfq = it.next(),
            "--prompt-file" => out.prompt_file = it.next(),
            "--reference" => out.reference = it.next(),
            "--max-new-tokens" => out.max_new_tokens = it.next()
                .and_then(|s| s.parse().ok())
                .unwrap_or(16),
            "--no-load" => out.no_load = true,
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                print_help();
                std::process::exit(1);
            }
        }
    }
    out
}

fn print_help() {
    eprintln!(
        "usage: infer_qwen2 --hfq <path.hfq> [--prompt-file <path>] \
         [--reference <path.json>] [--max-new-tokens N] [--no-load]\n\
         \n\
         Without --prompt-file, runs config+weight-load smoke only.\n\
         With --prompt-file, also tokenizes and (if --reference given) \
         checks tokenizer parity against the HF reference.\n\
         \n\
         max-new-tokens controls how many continuation tokens the \
         forward pass will be asked for. Default 16 (matches the plan's \
         top-1 match acceptance criterion). Note: the forward pass is \
         not yet implemented — this binary will exit before generating \
         tokens until that lands."
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let hfq_path = args.hfq.as_deref()
        .ok_or("--hfq is required")?;

    eprintln!("[1/5] opening HFQ: {hfq_path}");
    let mut hfq = HfqFile::open(Path::new(hfq_path))?;
    eprintln!("      arch_id (header) = {}", hfq.arch_id);
    if hfq.arch_id != 7 {
        eprintln!(
            "      warning: arch_id={} but this binary targets the \
             hipfire-arch-qwen2 path (arch_id=7). Continuing — the \
             weight loader only reads the metadata + tensor manifest, \
             so a mis-tagged file will still load. Re-quantise with \
             `--arch-id 7` for daemon-compatible dispatch (see R1).",
            hfq.arch_id
        );
    }

    eprintln!("[2/5] parsing Qwen2Config");
    let cfg = qwen2::config_from_hfq(&hfq)
        .ok_or("qwen2: failed to parse config from HFQ metadata")?;
    eprintln!(
        "      hidden={}, layers={}, n_heads={}, n_kv_heads={}, \
         head_dim={}, vocab={}, attention_bias={}, tie_word_embeddings={}, \
         eos_ids={:?}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.vocab_size,
        cfg.attention_bias,
        cfg.tie_word_embeddings,
        cfg.eos_token_ids,
    );

    eprintln!("[3/5] building tokenizer from HFQ metadata");
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .ok_or("qwen2: tokenizer not found in HFQ metadata")?;
    eprintln!("      vocab_size={}", tok.vocab_size());

    // Steps 4/5 — prompt tokenize + parity check.
    let mut prompt_ids: Vec<u32> = Vec::new();
    if let Some(prompt_path) = args.prompt_file.as_deref() {
        let prompt_bytes = std::fs::read(prompt_path)?;
        let prompt_text = std::str::from_utf8(&prompt_bytes)?;
        eprintln!(
            "[4/5] encoding prompt ({} bytes) from {prompt_path}",
            prompt_bytes.len()
        );
        prompt_ids = tok.encode(prompt_text);
        eprintln!(
            "      {} prompt tokens; first 16 ids: {:?}",
            prompt_ids.len(),
            &prompt_ids[..prompt_ids.len().min(16)],
        );

        if let Some(ref_path) = args.reference.as_deref() {
            check_tokenizer_parity(ref_path, &prompt_ids)?;
        } else {
            eprintln!("      (no --reference; skipping parity check)");
        }
    } else {
        eprintln!("[4/5] no --prompt-file — skipping tokenize/parity");
    }

    if args.no_load {
        eprintln!("[5/5] --no-load → skipping GPU upload + forward");
        eprintln!("ok (no-load)");
        return Ok(());
    }

    eprintln!("[5/5] loading weights to GPU");
    let mut gpu = Gpu::init()?;
    let weights = qwen2::load_weights(&mut hfq, &cfg, &mut gpu)?;
    eprintln!(
        "      loaded: {} layers, tied_lm_head={}, embd_format={:?}",
        weights.layers.len(),
        weights.tied_lm_head,
        weights.embd_format,
    );

    eprintln!(
        "\nNOTE: forward pass not yet implemented in hipfire-arch-qwen2 \
         (rev 2). Once it lands, this binary will greedy-decode \
         max_new_tokens={} from prompt_ids[{}] and compare against the \
         reference's first_16_completion_token_ids field.\n\
         \n\
         For now, the validation work this binary performs is:\n\
         (a) HFQ header / config parse / weight load succeeds end-to-end\n\
         (b) tokenizer parity against HF for the smoke prompt\n\
         \n\
         Both of (a) and (b) are prerequisites for top-1 token match.",
        args.max_new_tokens,
        prompt_ids.len(),
    );

    Ok(())
}

fn check_tokenizer_parity(
    ref_path: &str,
    hipfire_ids: &[u32],
) -> Result<(), Box<dyn std::error::Error>> {
    let ref_bytes = std::fs::read(ref_path)?;
    let ref_json: serde_json::Value = serde_json::from_slice(&ref_bytes)?;
    let ref_ids: Vec<u32> = ref_json
        .get("prompt_token_ids")
        .and_then(|v| v.as_array())
        .ok_or("reference JSON missing prompt_token_ids array")?
        .iter()
        .filter_map(|v| v.as_u64().map(|n| n as u32))
        .collect();

    eprintln!(
        "      parity check: hipfire={} tokens, reference={} tokens",
        hipfire_ids.len(),
        ref_ids.len(),
    );

    if hipfire_ids == ref_ids.as_slice() {
        eprintln!("      ✓ tokenizer parity: token IDs match exactly");
        return Ok(());
    }

    eprintln!("      ✗ tokenizer parity FAILED");
    eprintln!("        reference: {:?}", ref_ids);
    eprintln!("        hipfire:   {:?}", hipfire_ids);
    let first_div = hipfire_ids.iter().zip(ref_ids.iter())
        .position(|(a, b)| a != b);
    if let Some(pos) = first_div {
        eprintln!(
            "        first divergence at position {pos}: \
             hipfire={}, reference={}",
            hipfire_ids[pos], ref_ids[pos]
        );
    } else {
        eprintln!(
            "        prefix matches up to common length; lengths differ"
        );
    }
    Err("tokenizer parity check failed — see above".into())
}
