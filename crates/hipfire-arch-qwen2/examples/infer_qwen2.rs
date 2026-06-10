//! Standalone forward-pass driver for `hipfire-arch-qwen2`.
//!
//! This binary is the bring-up validation harness called out in §6 R3
//! of `docs/plans/dots-ocr-prd.md`. It bypasses the
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
//!    embedded `tokenizer.json` blob.
//! 3. Encode the prompt and (optionally) compare its token-id sequence
//!    against the reference artifact — a tokenizer-parity check that
//!    catches BPE divergence before any kernel work runs.
//! 4. Forward + greedy decode N tokens via [`qwen2::forward_step`] +
//!    [`qwen2::forward_step_greedy`] (28-layer Qwen2 stack on GPU).
//! 5. Compare the generated token IDs against
//!    `first_16_completion_token_ids` in the reference; print
//!    per-position PASS/FAIL and exit non-zero on divergence.
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
    max_seq: usize,
}

fn parse_args() -> Args {
    let mut out = Args { max_new_tokens: 16, max_seq: 512, ..Default::default() };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--hfq" => out.hfq = it.next(),
            "--prompt-file" => out.prompt_file = it.next(),
            "--reference" => out.reference = it.next(),
            "--max-new-tokens" => out.max_new_tokens = it.next()
                .and_then(|s| s.parse().ok())
                .unwrap_or(16),
            "--max-seq" => out.max_seq = it.next()
                .and_then(|s| s.parse().ok())
                .unwrap_or(512),
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
         forward pass will generate. Default 16 (matches the plan's \
         top-1 match acceptance criterion). With both --prompt-file \
         and --reference, the binary exits non-zero if hipfire's \
         generated tokens diverge from the reference."
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
        .map_err(|e| format!("qwen2: tokenizer not found in HFQ metadata: {e}"))?;
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

    eprintln!("[5/5] loading weights + allocating Qwen2State");
    let mut gpu = Gpu::init()?;
    let weights = qwen2::load_weights(&mut hfq, &cfg, &mut gpu)?;
    eprintln!(
        "      loaded: {} layers, tied_lm_head={}, embd_format={:?}",
        weights.layers.len(),
        weights.tied_lm_head,
        weights.embd_format,
    );

    let mut state = if args.max_seq > 512 {
        qwen2::Qwen2State::new_with_max_seq(&mut gpu, &cfg, args.max_seq)
            .map_err(|e| format!("Qwen2State::new_with_max_seq({}) failed: {e}", args.max_seq))?
    } else {
        qwen2::Qwen2State::new(&mut gpu, &cfg)
            .map_err(|e| format!("Qwen2State::new failed: {e}"))?
    };
    eprintln!("      KV budget = {} positions", state.max_seq);

    if prompt_ids.is_empty() {
        eprintln!("ok (loaded; no --prompt-file → skipping forward)");
        return Ok(());
    }

    // Prefill: run forward_step for each prompt token. We keep only the
    // *last* logits — earlier positions are written to the KV cache but
    // their logits are discarded because we want the prediction
    // *after* the final prompt token.
    eprintln!("[forward] prefilling {} prompt tokens", prompt_ids.len());
    let prefill_start = std::time::Instant::now();
    for (i, &tok) in prompt_ids.iter().enumerate() {
        qwen2::forward_step(&mut gpu, &weights, &cfg, &mut state, tok)?;
        if i % 8 == 0 || i + 1 == prompt_ids.len() {
            eprintln!("  prompt pos {i:3}: token {tok} → pos_after={}", state.next_pos);
        }
    }
    let prefill_ms = prefill_start.elapsed().as_millis();

    // Greedy-decode max_new_tokens from the post-prefill logits.
    eprintln!(
        "[forward] greedy-decoding {} continuation tokens",
        args.max_new_tokens
    );
    let mut generated: Vec<u32> = Vec::with_capacity(args.max_new_tokens);
    // The first continuation token is argmax of the logits already in
    // state.logits (set by the last prefill forward_step).
    let mut next_tok = gpu.argmax_f32(&state.logits, cfg.vocab_size)?;
    generated.push(next_tok);
    eprintln!("  gen 0: {next_tok}");
    for i in 1..args.max_new_tokens {
        next_tok = qwen2::forward_step_greedy(&mut gpu, &weights, &cfg, &mut state, next_tok)?;
        generated.push(next_tok);
        eprintln!("  gen {i}: {next_tok}");
    }
    let total_ms = prefill_start.elapsed().as_millis();
    eprintln!(
        "[forward] done: {} prompt + {} gen tokens in {} ms (prefill {} ms)",
        prompt_ids.len(),
        generated.len(),
        total_ms,
        prefill_ms,
    );

    // Reference compare on the generated tokens.
    if let Some(ref_path) = args.reference.as_deref() {
        check_completion_parity(ref_path, &generated)?;
    } else {
        eprintln!("(no --reference; skipping completion parity check)");
    }

    eprintln!("ok");
    Ok(())
}

fn check_completion_parity(
    ref_path: &str,
    hipfire_ids: &[u32],
) -> Result<(), Box<dyn std::error::Error>> {
    let ref_bytes = std::fs::read(ref_path)?;
    let ref_json: serde_json::Value = serde_json::from_slice(&ref_bytes)?;
    let ref_first16: Vec<u32> = ref_json
        .get("first_16_completion_token_ids")
        .and_then(|v| v.as_array())
        .ok_or("reference JSON missing first_16_completion_token_ids array")?
        .iter()
        .filter_map(|v| v.as_u64().map(|n| n as u32))
        .collect();

    let n = ref_first16.len().min(hipfire_ids.len());
    eprintln!(
        "[validate] comparing first {n} generated tokens vs HF reference"
    );

    let mut matches = 0usize;
    for i in 0..n {
        let h = hipfire_ids[i];
        let r = ref_first16[i];
        let mark = if h == r { "✓" } else { "✗" };
        eprintln!("  {mark} pos {i:2}: hipfire={h:6}  ref={r:6}");
        if h == r {
            matches += 1;
        }
    }
    eprintln!("[validate] {matches} / {n} top-1 matches");
    if matches == n {
        eprintln!("[validate] PASS — top-1 match on all {n} positions");
        Ok(())
    } else {
        Err(format!(
            "top-1 token match FAILED: {matches}/{n} positions match \
             between hipfire and HF reference. First divergence at \
             position {}.",
            (0..n).find(|&i| hipfire_ids[i] != ref_first16[i]).unwrap_or(n)
        )
        .into())
    }
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
