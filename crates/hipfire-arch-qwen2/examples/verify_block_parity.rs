//! Byte-identical parity check: block-parallel batched verify vs the legacy
//! sequential verify for `hipfire-arch-qwen2` (arch_id=7) spec-decode.
//!
//! For the SAME (block, position, KV state), the batched
//! `qwen2::forward_verify_block_batched` must return the EXACT same per-slot
//! argmax vector as the sequential `forward_step`-per-token loop. This is the
//! non-negotiable correctness gate the perf win rides on — a fast wrong verify
//! is worthless.
//!
//! Method: load VibeThinker-3B, prefill a prompt, then at several mid-sequence
//! positions run both verify paths on the same candidate block off a freshly
//! re-prefilled KV state and assert equality. The block is built from the
//! model's own greedy continuation so it exercises realistic verify shapes
//! (high accept), plus a perturbed block to exercise the reject path.
//!
//! Usage:
//!   cargo run --release --example verify_block_parity -p hipfire-arch-qwen2 -- \
//!       --hfq ~/.hipfire/models/vibethinker-3b.mq4.hfq

use std::path::Path;

use hipfire_arch_qwen2::qwen2;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut hfq_path: Option<String> = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--hfq" => hfq_path = it.next(),
            other => return Err(format!("unknown arg: {other}").into()),
        }
    }
    let hfq_path = hfq_path.ok_or("--hfq is required")?;

    let mut hfq = HfqFile::open(Path::new(&hfq_path))?;
    let cfg = qwen2::config_from_hfq(&hfq)?;
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json)?;
    eprintln!(
        "loaded config: hidden={}, layers={}, n_heads={}, n_kv_heads={}, head_dim={}, vocab={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.vocab_size,
    );

    let mut gpu = Gpu::init()?;
    let weights = qwen2::load_weights(&mut hfq, &cfg, &mut gpu)?;
    let mut state = qwen2::Qwen2State::new_with_max_seq(&mut gpu, &cfg, 1024)?;

    // A LONG prompt (>128 tokens) so the sequential verify's per-step flash
    // attention runs MULTIPLE chunks (online-softmax across chunks), exercising
    // the case where the batched single-pass softmax could in principle diverge
    // in the low bits. We want parity to hold here, not just for short contexts.
    let para = "The history of computing spans many centuries and cultures. \
        Early mechanical calculators gave way to electromechanical relays, then \
        vacuum tubes, then transistors, and finally integrated circuits. Each \
        leap reduced size and cost while increasing speed and reliability. \
        Today's processors contain billions of transistors on a single die. ";
    let prompt = format!("{para}{para}{para}Summarize the key transitions:");
    let prompt_ids = tok.encode(&prompt);
    eprintln!("prompt: {} tokens", prompt_ids.len());

    // Helper: prefill prompt_ids into a fresh state and return greedy
    // continuation of `n` tokens (drives a realistic candidate block).
    let prefill =
        |gpu: &mut Gpu, state: &mut qwen2::Qwen2State, ids: &[u32]| -> Result<(), String> {
            state.reset();
            for &t in ids {
                qwen2::forward_step(gpu, &weights, &cfg, state, t).map_err(|e| format!("{e:?}"))?;
            }
            Ok(())
        };

    // Generate a greedy continuation of 8 tokens to use as the candidate block.
    prefill(&mut gpu, &mut state, &prompt_ids)?;
    let mut cont: Vec<u32> = Vec::new();
    let mut nxt = gpu.argmax_f32(&state.logits, cfg.vocab_size)?;
    cont.push(nxt);
    for _ in 1..8 {
        nxt = qwen2::forward_step_greedy(&mut gpu, &weights, &cfg, &mut state, nxt)?;
        cont.push(nxt);
    }
    eprintln!("greedy continuation block: {cont:?}");

    // The verify position is right after the prompt (where the block's slot 0
    // lives). Build candidate blocks of varying lengths and an off-distribution
    // perturbation to exercise the reject path.
    let position = prompt_ids.len();

    let mut all_ok = true;
    let mut case = |gpu: &mut Gpu,
                    state: &mut qwen2::Qwen2State,
                    label: &str,
                    block: &[u32]|
     -> Result<(), String> {
        // Sequential reference (re-prefill so both start from identical KV).
        prefill(gpu, state, &prompt_ids)?;
        let seq = {
            state.next_pos = position;
            let mut out = Vec::with_capacity(block.len());
            for &t in block {
                qwen2::forward_step(gpu, &weights, &cfg, state, t).map_err(|e| format!("{e:?}"))?;
                out.push(
                    gpu.argmax_f32(&state.logits, cfg.vocab_size)
                        .map_err(|e| format!("{e:?}"))?,
                );
            }
            out
        };

        // Batched (re-prefill again so KV history is identical to the seq run).
        prefill(gpu, state, &prompt_ids)?;
        let bat = qwen2::forward_verify_block_batched(gpu, &weights, &cfg, state, block, position)
            .map_err(|e| format!("{e:?}"))?;

        let ok = seq == bat;
        eprintln!(
            "[{label}] B={:2} seq={seq:?}\n         bat={bat:?}  -> {}",
            block.len(),
            if ok { "MATCH" } else { "MISMATCH" }
        );
        if !ok {
            all_ok = false;
            let first = seq.iter().zip(&bat).position(|(a, b)| a != b);
            eprintln!("         first divergence at slot {first:?}");
        }
        Ok(())
    };

    // Case 1: full greedy block (high accept, every slot self-consistent).
    case(&mut gpu, &mut state, "greedy-8", &cont)?;
    // Case 2: shorter blocks (B=1,2,3,4) — verify edge shapes.
    for b in [1usize, 2, 3, 4] {
        case(&mut gpu, &mut state, &format!("greedy-{b}"), &cont[..b])?;
    }
    // Case 3: perturbed block (reject path) — flip a couple of tokens.
    let mut perturbed = cont.clone();
    if perturbed.len() >= 4 {
        perturbed[2] = (perturbed[2] + 137) % cfg.vocab_size as u32;
        perturbed[5] = (perturbed[5] + 911) % cfg.vocab_size as u32;
    }
    case(&mut gpu, &mut state, "perturbed-8", &perturbed)?;

    if all_ok {
        eprintln!("\nPASS: batched verify is byte-identical (per-slot argmax) to sequential on all cases.");
        Ok(())
    } else {
        Err("FAIL: batched verify diverged from sequential — see slots above.".into())
    }
}
