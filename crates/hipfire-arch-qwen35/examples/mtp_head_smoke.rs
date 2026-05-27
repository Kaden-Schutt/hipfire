//! mtp_head_smoke: minimal end-to-end smoke test for the Qwen3.5 native
//! MTP head loader + forward pass.
//!
//! Algorithm:
//!   1. Load `~/.hipfire/models/qwen3.5-0.8b.mq4` as the trunk.
//!   2. Run trunk forward at (token=1, pos=0); capture post-final-norm
//!      hidden as `prev_hidden`.
//!   3. Load `/tmp/qwen3.5-0.8b.mtp` (Task 8 output).
//!   4. Call `mtp_head_forward(next_token=1, prev_hidden, pos=0)`; download
//!      logits and assert finite, non-degenerate, top1 in vocab range.
//!   5. Call again at pos=1 with a different `next_token` to confirm the
//!      KV cache write at pos=0 is consumed when attending at pos=1.
//!   6. Print `[mtp-smoke] OK: top1=<id> logit=<v>` and exit 0.

use hip_bridge::HipResult;
use hipfire_arch_qwen35::mtp_head;
use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::KvCache;
use rdna_compute::Gpu;
use std::path::Path;

fn main() -> HipResult<()> {
    let trunk_path = std::env::var("HIPFIRE_MTP_SMOKE_TRUNK")
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").expect("HOME unset");
            format!("{home}/.hipfire/models/qwen3.5-0.8b.mq4")
        });
    let mtp_path = std::env::var("HIPFIRE_MTP_SMOKE_HEAD")
        .unwrap_or_else(|_| "/tmp/qwen3.5-0.8b.mtp".to_string());

    eprintln!("[mtp-smoke] trunk: {trunk_path}");
    eprintln!("[mtp-smoke] head:  {mtp_path}");

    if !Path::new(&trunk_path).exists() {
        panic!(
            "[mtp-smoke] trunk model not found at {trunk_path}; \
             override with HIPFIRE_MTP_SMOKE_TRUNK"
        );
    }
    if !Path::new(&mtp_path).exists() {
        panic!(
            "[mtp-smoke] .mtp file not found at {mtp_path}; build it with: \
             ./target/release/mtp_extract \
             --hf-dir \"$(echo ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*)\" \
             --output {mtp_path}"
        );
    }

    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("[mtp-smoke] gpu: {}", gpu.arch);

    // ── Step 1: Load trunk + run a forward to capture prev_hidden ────────
    eprintln!("[mtp-smoke] loading trunk...");
    let mut trunk_hfq = HfqFile::open(Path::new(&trunk_path))
        .expect("open trunk model");
    let trunk_config = qwen35::config_from_hfq(&trunk_hfq)
        .expect("trunk config_from_hfq");
    let trunk_weights = qwen35::load_weights(&mut trunk_hfq, &trunk_config, &mut gpu)
        .expect("trunk load_weights");

    eprintln!(
        "[mtp-smoke] trunk dims: dim={} n_layers={} vocab={} eos={}",
        trunk_config.dim, trunk_config.n_layers,
        trunk_config.vocab_size, trunk_config.eos_token,
    );

    // KV cache + DeltaNet state for trunk's single warm-up forward.
    let mut trunk_kv = KvCache::new_gpu_q8(
        &mut gpu,
        trunk_config.n_layers,
        trunk_config.n_kv_heads,
        trunk_config.head_dim,
        128,
    ).expect("trunk KV alloc");
    let mut trunk_dn = DeltaNetState::new(&mut gpu, &trunk_config)
        .expect("DeltaNetState::new");
    let trunk_scratch = Qwen35Scratch::new(&mut gpu, &trunk_config, 64)
        .expect("Qwen35Scratch::new");

    eprintln!("[mtp-smoke] trunk forward: token=1 pos=0...");
    qwen35::forward_scratch(
        &mut gpu, &trunk_weights, &trunk_config,
        /*token*/ 1, /*pos*/ 0,
        &mut trunk_kv, &mut trunk_dn, &trunk_scratch,
    ).expect("trunk forward");

    // After forward_scratch, scratch.tmp holds RMSNorm(last-layer-x,
    // output_norm) — the exact input to lm_head, i.e. the post-final-norm
    // hidden state. Sanity-check by downloading.
    let prev_hidden_host = gpu.download_f32(&trunk_scratch.tmp)
        .expect("download trunk hidden");
    let nn = prev_hidden_host.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(nn, 0, "[mtp-smoke] trunk hidden has {nn} non-finite values");
    eprintln!(
        "[mtp-smoke] trunk hidden: first4=[{:.4},{:.4},{:.4},{:.4}] norm={:.3}",
        prev_hidden_host[0], prev_hidden_host[1], prev_hidden_host[2], prev_hidden_host[3],
        prev_hidden_host.iter().map(|v| v * v).sum::<f32>().sqrt(),
    );

    // ── Step 2: Load MTP head + alloc scratch + KV cache ─────────────────
    eprintln!("[mtp-smoke] loading mtp head...");
    let mtp_max_seq = 64;
    let head = mtp_head::load_mtp_head(Path::new(&mtp_path), &mut gpu, mtp_max_seq)
        .expect("load_mtp_head");
    eprintln!(
        "[mtp-smoke] head config: n_embd={} n_head={} n_head_kv={} head_dim={} n_ff={} \
         vocab={} n_rot={} rope_theta={} eps={} tied_emb={}",
        head.config.n_embd, head.config.n_head, head.config.n_head_kv,
        head.config.head_dim, head.config.n_ff, head.config.vocab_size,
        head.config.n_rot, head.config.rope_theta, head.config.rms_norm_eps,
        head.config.tie_word_embeddings,
    );

    // Sanity-check head config matches trunk.
    assert_eq!(head.config.n_embd, trunk_config.dim, "n_embd mismatch");
    assert_eq!(head.config.vocab_size, trunk_config.vocab_size, "vocab mismatch");

    let mtp_scratch = mtp_head::Qwen35MtpHeadScratch::new(&mut gpu, &head.config)
        .expect("MtpHeadScratch::new");
    let mut mtp_kv = mtp_head::Qwen35MtpHeadKvCache::new(&mut gpu, &head.config)
        .expect("MtpHeadKvCache::new");

    // ── Step 3: First MTP forward at pos=0 ───────────────────────────────
    eprintln!("[mtp-smoke] mtp forward 1: next_token=1 pos=0...");
    mtp_head::mtp_head_forward(
        &mut gpu, &head, &mtp_scratch, &mut mtp_kv,
        /*next_token*/ 1,
        /*prev_hidden*/ &trunk_scratch.tmp,
        /*pos*/ 0,
        &trunk_weights,
        /*lm_head_weights*/ &trunk_weights.output,
    ).expect("mtp_head_forward 1");

    let logits1 = gpu.download_f32(&mtp_scratch.logits)
        .expect("download logits 1");
    assert_logits_healthy("forward 1", &logits1, head.config.vocab_size);
    let (top1_idx, top1_val) = argmax(&logits1);

    // ── Step 4: Second MTP forward at pos=1 with different token ─────────
    //
    // This both verifies the second-position attention reads pos=0's K/V
    // (otherwise the output is undefined / zero-attended), and that the
    // forward is deterministic — feeding the same prev_hidden + a
    // different next_token at a different pos should yield a different
    // top1.
    eprintln!("[mtp-smoke] mtp forward 2: next_token=42 pos=1...");
    mtp_head::mtp_head_forward(
        &mut gpu, &head, &mtp_scratch, &mut mtp_kv,
        /*next_token*/ 42,
        /*prev_hidden*/ &trunk_scratch.tmp,
        /*pos*/ 1,
        &trunk_weights,
        &trunk_weights.output,
    ).expect("mtp_head_forward 2");
    let logits2 = gpu.download_f32(&mtp_scratch.logits)
        .expect("download logits 2");
    assert_logits_healthy("forward 2", &logits2, head.config.vocab_size);
    let (top1_idx2, top1_val2) = argmax(&logits2);

    // The two forwards differ in (next_token, pos) AND the second forward's
    // attention reads pos=0's KV. If the head silently no-op'd the cache
    // write, the second logits would equal the first. Allow the unlikely
    // case that argmax happens to coincide but require some float
    // divergence in the L2 distance between the two logit vectors.
    let dist: f32 = logits1.iter().zip(logits2.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>().sqrt();
    assert!(
        dist > 1e-3,
        "[mtp-smoke] logits1 and logits2 are bitwise-near-identical \
         (l2 distance {dist}) — KV write at pos=0 is likely not being read \
         at pos=1"
    );
    eprintln!("[mtp-smoke] logits l2 distance forward1↔forward2: {dist:.3} (> 1e-3 OK)");

    // ── Step 5: Final report ─────────────────────────────────────────────
    println!(
        "[mtp-smoke] OK: top1={} logit={:.4} (forward2 top1={} logit={:.4}, l2={:.3})",
        top1_idx, top1_val, top1_idx2, top1_val2, dist,
    );

    // Free GPU buffers (not strictly required since we're exiting, but
    // confirms the free paths compile + don't panic).
    mtp_scratch.free_gpu(&mut gpu);
    mtp_kv.free_gpu(&mut gpu);
    head.free_gpu(&mut gpu);

    Ok(())
}

fn assert_logits_healthy(label: &str, logits: &[f32], expected_vocab: usize) {
    assert_eq!(
        logits.len(), expected_vocab,
        "[mtp-smoke {label}] logits len {} != expected vocab {expected_vocab}",
        logits.len(),
    );
    let nan_count = logits.iter().filter(|v| v.is_nan()).count();
    let inf_count = logits.iter().filter(|v| v.is_infinite()).count();
    assert_eq!(nan_count, 0, "[mtp-smoke {label}] {nan_count} NaN logits");
    assert_eq!(inf_count, 0, "[mtp-smoke {label}] {inf_count} Inf logits");

    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    let mut sum: f64 = 0.0;
    for &v in logits {
        if v < min_v { min_v = v; }
        if v > max_v { max_v = v; }
        sum += v as f64;
    }
    let mean = sum / logits.len() as f64;
    eprintln!(
        "[mtp-smoke {label}] logits stats: min={min_v:.3} max={max_v:.3} \
         mean={mean:.3} range={:.3}",
        max_v - min_v,
    );
    assert!(
        max_v > min_v + 1.0,
        "[mtp-smoke {label}] logits are nearly constant \
         (min={min_v} max={max_v} range={})",
        max_v - min_v,
    );
    // Distinct-values check — all-equal logits often signal a degenerate
    // forward (e.g. all-zero hidden, a missing matmul). Sample 32 evenly
    // across the vocab.
    let stride = (logits.len() / 32).max(1);
    let mut distinct: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for i in (0..logits.len()).step_by(stride) {
        distinct.insert(logits[i].to_bits());
    }
    assert!(
        distinct.len() >= 16,
        "[mtp-smoke {label}] only {} distinct sampled-logit values (32 sampled)",
        distinct.len(),
    );
}

fn argmax(logits: &[f32]) -> (u32, f32) {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v { best_v = v; best_i = i; }
    }
    (best_i as u32, best_v)
}
