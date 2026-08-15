//! dspark_forward_smoke: end-to-end smoke for the DSpark draft forward.
//!
//! Loads the DeepSeek-V4-Flash trunk (the `-dspark` sidecar auto-loads into
//! `weights.dspark`), prefills a short prompt through the batched trunk
//! forward WITH the target-hidden capture armed (layers
//! `dspark.cfg.target_layer_ids`, e.g. [40,41,42]), assembles the captured
//! per-layer hidden states for the last prompt position into the
//! `main_hidden` [n_target_layers*hidden] vector, and runs `dspark_forward`
//! to produce a block of draft tokens.
//!
//! It prints the raw draft token ids, the tokenizer-decoded draft text (so a
//! human can eyeball plausibility), the per-slot confidence array, and the
//! greedy AR next token (argmax of the trunk's last-position logits) for
//! comparison. No GPU validation is asserted — this is an eyeball smoke.
//!
//! Usage:
//!   dspark_forward_smoke
//!   HIPFIRE_DEEPSEEK4_MODEL=/path/to/deepseek-v4-flash.mq2lloyd dspark_forward_smoke
//!
//! ENV:
//!   HIPFIRE_DEEPSEEK4_MODEL    trunk HFQ path (default ~/.hipfire/models/deepseek-v4-flash.mq2lloyd)
//!   HIPFIRE_DEEPSEEK4_PROMPT   prompt text (default "The capital of France is")
//!   HIPFIRE_DEEPSEEK4_PP_BATCH prefill chunk size (default 1024)

use hipfire_arch_deepseek4::forward::{
    dspark_assemble_main_hidden, dspark_forward, forward_prefill_batch_chunked, PrefillBatchScratch,
};
use hipfire_arch_deepseek4::{DeepseekV4, DeepseekV4State};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;
use std::path::Path;

fn main() -> Result<(), String> {
    let path = std::env::var("HIPFIRE_DEEPSEEK4_MODEL").unwrap_or_else(|_| {
        format!(
            "{}/.hipfire/models/deepseek-v4-flash.mq2lloyd",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    let prompt = std::env::var("HIPFIRE_DEEPSEEK4_PROMPT")
        .unwrap_or_else(|_| "The capital of France is".to_string());
    let pbs_max_batch: usize = std::env::var("HIPFIRE_DEEPSEEK4_PP_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);

    eprintln!("Loading DeepSeek V4 trunk from {path} (dspark sidecar auto-loads)...");
    let mut hfq = HfqFile::open(Path::new(&path)).map_err(|e| format!("open: {e:?}"))?;
    let cfg = DeepseekV4::config_from_hfq(&hfq)?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer not found in HFQ metadata: {e:?}"))?;

    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    let weights = DeepseekV4::load_weights(&mut hfq, &cfg, &mut gpu)?;
    let mut state = DeepseekV4State::new(&cfg)?;

    let dspark = weights.dspark.as_ref().ok_or_else(|| {
        "weights.dspark is None — DSpark sidecar not found next to the trunk \
         (expected <stem>-dspark.<ext>)"
            .to_string()
    })?;
    eprintln!(
        "DSpark loaded: block_size={} target_layers={:?} stages={}",
        dspark.cfg.block_size,
        dspark.cfg.target_layer_ids,
        dspark.stages.len()
    );

    // Arm the target-hidden capture for exactly the DSpark target layers.
    state.dspark_capture_active = true;
    state.dspark_target_layers = dspark.cfg.target_layer_ids.clone();

    let mut pbs = PrefillBatchScratch::new(&mut gpu, &cfg, pbs_max_batch)?;

    // Tokenize the prompt (base completion — no chat framing; the smoke only
    // needs a real hidden-state to feed the drafter).
    let prompt_tokens = tokenizer.encode(&prompt);
    if prompt_tokens.is_empty() {
        return Err("prompt tokenized to zero tokens".to_string());
    }
    let prompt_len = prompt_tokens.len();
    eprintln!("Prompt: {prompt:?} -> {prompt_len} tokens");

    // Prefill through the batched trunk forward. This routes through
    // forward_prefill_batch_chunk, which (because capture is armed) stashes
    // the [40,41,42] HC-mean-pooled hidden states per position. Returns the
    // LAST position's trunk logits.
    let last_logits = forward_prefill_batch_chunked(
        &cfg,
        &weights,
        &mut state,
        &mut gpu,
        &prompt_tokens,
        /*start_pos=*/ 0,
        &mut pbs,
    )?;

    // The capture buffer holds one slot per prompt position within the LAST
    // chunk. With a single chunk (prompt_len <= max_batch) the last prompt
    // position lives at batch row prompt_len-1 of the final chunk.
    let last_chunk_len = if prompt_len % pbs_max_batch == 0 {
        pbs_max_batch
    } else {
        prompt_len % pbs_max_batch
    };
    let last_pos_in_chunk = last_chunk_len - 1;

    // Assemble the captured [40,41,42] hiddens for that position into the
    // [n_target_layers * hidden] main_hidden vector.
    let main_hidden =
        dspark_assemble_main_hidden(&mut state, &mut gpu, &cfg, last_pos_in_chunk)?.shallow_clone();

    let token_embd = weights
        .token_embd
        .as_ref()
        .ok_or("weights.token_embd is None")?
        .shallow_clone();
    let head = weights
        .head
        .as_ref()
        .ok_or("weights.head is None")?
        .shallow_clone();
    let output_norm = weights
        .output_norm
        .as_ref()
        .ok_or("weights.output_norm is None")?
        .shallow_clone();

    let last_prompt_token = prompt_tokens[prompt_len - 1];
    let position = (prompt_len - 1) as u32;

    eprintln!("Running dspark_forward (prev_token={last_prompt_token}, position={position})...");
    let dspark_w = weights.dspark.as_ref().unwrap();
    let draft = dspark_forward(
        &cfg,
        dspark_w,
        &mut state,
        &mut gpu,
        &main_hidden,
        &token_embd,
        &head,
        &output_norm,
        last_prompt_token,
        position,
    )?;

    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("post-forward sync: {e:?}"))?;

    // Greedy AR next token from the trunk's last-position logits, for
    // comparison against draft slot 0.
    let ar_next: u32 = last_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0);

    println!("=== DSpark draft smoke ===");
    println!("prompt: {prompt:?}");
    println!(
        "AR greedy next token: {ar_next} -> {:?}",
        tokenizer.decode(&[ar_next])
    );
    println!("draft tokens ({}): {:?}", draft.tokens.len(), draft.tokens);
    println!("draft decoded: {:?}", tokenizer.decode(&draft.tokens));
    for (i, &t) in draft.tokens.iter().enumerate() {
        let conf = draft.confidence.get(i).copied().unwrap_or(f32::NAN);
        println!(
            "  slot {i}: token={t} conf={conf:.4} text={:?}",
            tokenizer.decode(&[t])
        );
    }
    println!("confidence array: {:?}", draft.confidence);
    println!("SMOKE OK (eyeball the draft text + confidence for plausibility)");

    Ok(())
}
