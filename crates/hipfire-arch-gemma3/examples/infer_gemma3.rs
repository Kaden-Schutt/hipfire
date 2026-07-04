//! Standalone forward-pass driver for `hipfire-arch-gemma3` (E1 bring-up).
//!
//! Bypasses the daemon — no `arch_id` dispatch, no `LoadedModel` — so the
//! gemma3 forward can be validated before the seam wiring (E2) lands. Loads an
//! HFQ, tokenizes a prompt, prefills, greedy-decodes N tokens, and prints the
//! decoded continuation for a coherence eyeball (plus optional token-id parity
//! against an HF reference).
//!
//! ```text
//! cargo run --release --example infer_gemma3 -p hipfire-arch-gemma3 -- \
//!     --hfq ~/.hipfire/models/medgemma-27b-text-it-q8f16.hfq \
//!     --prompt-file benchmarks/prompts/gemma3_smoke.txt --max-new-tokens 48
//! ```
//!
//! `--no-load` exercises config + tokenizer only (no GPU).

use std::path::Path;

use hipfire_arch_gemma3 as gemma3;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_rdna::Gpu;

#[derive(Default)]
struct Args {
    hfq: Option<String>,
    prompt_file: Option<String>,
    prompt: Option<String>,
    no_load: bool,
    max_new_tokens: usize,
    max_seq: usize,
}

fn parse_args() -> Args {
    let mut out = Args {
        max_new_tokens: 48,
        max_seq: gemma3::forward::DEFAULT_MAX_SEQ,
        ..Default::default()
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--hfq" => out.hfq = it.next(),
            "--prompt-file" => out.prompt_file = it.next(),
            "--prompt" => out.prompt = it.next(),
            "--max-new-tokens" => {
                out.max_new_tokens = it.next().and_then(|s| s.parse().ok()).unwrap_or(48)
            }
            "--max-seq" => {
                out.max_seq = it
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(gemma3::forward::DEFAULT_MAX_SEQ)
            }
            "--no-load" => out.no_load = true,
            "-h" | "--help" => {
                eprintln!(
                    "usage: infer_gemma3 --hfq <path.hfq> [--prompt-file <p> | --prompt <text>] \
                     [--max-new-tokens N] [--max-seq N] [--no-load]"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let hfq_path = args.hfq.as_deref().ok_or("--hfq is required")?;

    eprintln!("[1/5] opening HFQ: {hfq_path}");
    let mut hfq = HfqFile::open(Path::new(hfq_path))?;
    eprintln!("      arch_id (header) = {}", hfq.arch_id);
    if hfq.arch_id != 12 {
        eprintln!("      warning: arch_id={} (gemma3 expects 12)", hfq.arch_id);
    }

    eprintln!("[2/5] parsing Gemma3Config");
    let cfg = gemma3::config_from_hfq(&hfq).ok_or("gemma3: failed to parse config")?;
    eprintln!(
        "      hidden={} layers={} n_heads={} n_kv={} head_dim={} vocab={}\n\
               sliding_window={} pattern={} qpas={} embed_scale={:.3} \
         attn_scale={:.5} q_prescale={:.5} norm_offset={} tie={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.vocab_size,
        cfg.sliding_window,
        cfg.sliding_window_pattern,
        cfg.query_pre_attn_scalar,
        cfg.embed_scale(),
        cfg.attn_scale(),
        cfg.q_prescale(),
        cfg.gemma_norm_offset,
        cfg.tie_word_embeddings,
    );
    if cfg.gemma_norm_offset == 0.0 {
        eprintln!(
            "      WARNING: gemma_norm_offset=0 — the (1+w) RMSNorm bake is \
             missing; re-ingest with the gemma3 quantizer or output will be wrong."
        );
    }

    eprintln!("[3/5] building tokenizer");
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("gemma3: tokenizer not found: {e}"))?;
    eprintln!("      vocab_size={}", tok.vocab_size());

    let prompt_text = if let Some(p) = args.prompt.as_deref() {
        p.to_string()
    } else if let Some(pf) = args.prompt_file.as_deref() {
        String::from_utf8(std::fs::read(pf)?)?
    } else {
        // Gemma chat framing for a quick coherence smoke.
        "<start_of_turn>user\nIn one sentence, what is a CT scan?<end_of_turn>\n\
         <start_of_turn>model\n"
            .to_string()
    };
    let prompt_ids = tok.encode(&prompt_text);
    eprintln!(
        "[4/5] {} prompt tokens; first 16: {:?}",
        prompt_ids.len(),
        &prompt_ids[..prompt_ids.len().min(16)]
    );

    if args.no_load {
        eprintln!("[5/5] --no-load → done (config + tokenizer only)");
        return Ok(());
    }

    eprintln!(
        "[5/5] loading weights + Gemma3State (max_seq={})",
        args.max_seq
    );
    let mut gpu = Gpu::init()?;
    let weights = gemma3::load_weights(&mut hfq, &cfg, &mut gpu)?;
    let mut state = gemma3::Gemma3State::new_with_max_seq(&mut gpu, &cfg, args.max_seq, false)
        .map_err(|e| format!("Gemma3State::new failed: {e:?}"))?;

    eprintln!("[forward] prefilling {} tokens", prompt_ids.len());
    let t0 = std::time::Instant::now();
    for &t in &prompt_ids {
        gemma3::forward_step(&mut gpu, &weights, &cfg, &mut state, t)?;
    }
    let prefill_ms = t0.elapsed().as_millis();

    eprintln!("[forward] greedy-decoding {} tokens", args.max_new_tokens);
    let mut generated: Vec<u32> = Vec::with_capacity(args.max_new_tokens);
    let mut next_tok = gpu.argmax_f32(&state.logits, cfg.vocab_size)?;
    generated.push(next_tok);
    for _ in 1..args.max_new_tokens {
        next_tok = gemma3::forward_step_greedy(&mut gpu, &weights, &cfg, &mut state, next_tok)?;
        generated.push(next_tok);
    }
    let total_ms = t0.elapsed().as_millis();

    let text = tok.decode(&generated);
    eprintln!(
        "[forward] {} prompt + {} gen in {} ms (prefill {} ms)",
        prompt_ids.len(),
        generated.len(),
        total_ms,
        prefill_ms
    );
    eprintln!("      gen ids: {:?}", generated);
    println!("\n=== gemma3 continuation ===\n{text}\n===========================");
    Ok(())
}
