// SPDX-License-Identifier: Apache-2.0
//! Focused parity/lifecycle harness for Muse Glimmer continuous-batch decode.
//!
//! Gates:
//! 1. B=1 vs sequential 8-token greedy exact token equality
//! 2. Same prompt in physical lane 0 inside B=2 (lane 1 distinct); both match isolated refs
//! 3. Hole mask 0b101: no durable inactive-lane write (lane-1 isolated decode parity)
//! 4. Reset/reuse a longer lane with a shorter prompt; 8 tokens match short isolated ref
//! 5. free_gpu on every batch and sequential state on success paths
//!
//! Usage: batch_decode_parity <model.mq4|model.hfq>

use hipfire_arch_muse_glimmer::batch::{batch_weight_formats_supported, GlimmerDecodeBatchState};
use hipfire_arch_muse_glimmer::config::GlimmerConfig;
use hipfire_arch_muse_glimmer::forward::{decode_step, prefill_with_capture};
use hipfire_arch_muse_glimmer::forward_batch::forward_decode_batch_glimmer;
use hipfire_arch_muse_glimmer::glimmer::{GlimmerState, GlimmerWeights};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

const STEPS: usize = 8;
const LANE_CAPACITY: usize = 2048;

fn argmax_f32(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .expect("empty logits")
}

/// One-token-ahead sequential greedy reference.
/// Prefill once → pending = argmax(last logits). For each step: record pending,
/// feed it exactly once at absolute position prompt.len()+step, then argmax next.
fn sequential_greedy(
    gpu: &mut Gpu,
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    prompt: &[u32],
    steps: usize,
) -> Result<Vec<u32>, String> {
    let mut state = GlimmerState::new_with_max_seq(gpu, cfg, LANE_CAPACITY)
        .map_err(|e| format!("seq state: {e}"))?;
    let mut hidden = Vec::new();
    let logits = prefill_with_capture(cfg, weights, &mut state, gpu, prompt, 0, &[], &mut hidden)
        .map_err(|e| format!("seq prefill: {e}"))?;
    let mut pending = argmax_f32(&logits);
    let mut out = Vec::with_capacity(steps);
    for step in 0..steps {
        out.push(pending);
        let pos = (prompt.len() + step) as u32;
        let next_logits = decode_step(cfg, weights, &mut state, gpu, pending, pos)
            .map_err(|e| format!("seq decode step {step} pos {pos}: {e}"))?;
        pending = argmax_f32(&next_logits);
    }
    state.free_gpu(gpu);
    Ok(out)
}

/// Batch-lane greedy for a single active physical lane of width `physical_batch`.
fn batch_lane_greedy(
    gpu: &mut Gpu,
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &mut GlimmerDecodeBatchState,
    lane: usize,
    prompt: &[u32],
    steps: usize,
    physical_batch: usize,
) -> Result<Vec<u32>, String> {
    assert!(lane < physical_batch && physical_batch <= state.max_batch);

    let ok = state
        .prefill_lane_cancellable(gpu, weights, cfg, lane, prompt, &mut || false)
        .map_err(|e| format!("batch prefill lane {lane}: {e}"))?;
    if !ok {
        return Err(format!("batch prefill lane {lane} cancelled"));
    }

    let (mut pending, _) = state
        .sample_lane_product(gpu, cfg, lane, 0.0, 1.0, None, 1)
        .map_err(|e| format!("batch sample pending lane {lane}: {e}"))?;
    let mut pos = prompt.len();
    let mut out = Vec::with_capacity(steps);
    let mask = 1u64 << lane;

    for step in 0..steps {
        out.push(pending);
        let mut tokens = vec![0u32; physical_batch];
        let mut positions = vec![0usize; physical_batch];
        tokens[lane] = pending;
        positions[lane] = pos;
        forward_decode_batch_glimmer(gpu, weights, cfg, &tokens, &positions, mask, state)
            .map_err(|e| format!("batch forward step {step} lane {lane}: {e}"))?;
        let (next, _) = state
            .sample_lane_product(gpu, cfg, lane, 0.0, 1.0, None, 1)
            .map_err(|e| format!("batch sample step {step} lane {lane}: {e}"))?;
        pending = next;
        pos += 1;
    }
    Ok(out)
}

fn assert_tokens_eq(label: &str, expected: &[u32], got: &[u32]) {
    if expected == got {
        eprintln!("PASS {label}: {expected:?}");
        return;
    }
    eprintln!("FAIL {label}");
    eprintln!("  expected ({}) {:?}", expected.len(), expected);
    eprintln!("  got      ({}) {:?}", got.len(), got);
    for i in 0..expected.len().max(got.len()) {
        let e = expected.get(i);
        let g = got.get(i);
        if e != g {
            eprintln!("  mismatch at [{i}]: expected {e:?} got {g:?}");
        }
    }
    panic!("{label}: token mismatch");
}

fn main() {
    let model = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: batch_decode_parity <model.mq4|model.hfq>");
        std::process::exit(2);
    });

    let mut gpu = Gpu::init().expect("gpu init");
    let hfq = HfqFile::open(std::path::Path::new(&model)).expect("open model");
    let cfg = GlimmerConfig::from_hfq(&hfq).expect("cfg");
    let weights = GlimmerWeights::load(&hfq, &cfg, &mut gpu).expect("weights");
    eprintln!(
        "glimmer batch parity: dim={} layers={} vocab={} window={} lm_head={:?}",
        cfg.dim, cfg.n_layers, cfg.vocab_size, cfg.sliding_window, weights.lm_head.gpu_dtype
    );

    batch_weight_formats_supported(&weights).expect("batch_weight_formats_supported");
    eprintln!("batch gate: PASS");

    let prompt_a: Vec<u32> = {
        let mut v = vec![cfg.bos_token];
        v.extend((0..10).map(|i| ((100 + i * 7) % 202_040) as u32));
        v
    };
    let prompt_b: Vec<u32> = {
        let mut v = vec![cfg.bos_token];
        v.extend((0..10).map(|i| ((200 + i * 11) % 202_040) as u32));
        v
    };
    let short_prompt: Vec<u32> = vec![cfg.bos_token, 111, 222, 333];
    let long_prompt: Vec<u32> = (0..64).map(|i| ((500 + i * 3) % 202_040) as u32).collect();

    let seq_a =
        sequential_greedy(&mut gpu, &cfg, &weights, &prompt_a, STEPS).expect("sequential prompt_a");
    let seq_b =
        sequential_greedy(&mut gpu, &cfg, &weights, &prompt_b, STEPS).expect("sequential prompt_b");
    let seq_short = sequential_greedy(&mut gpu, &cfg, &weights, &short_prompt, STEPS)
        .expect("sequential short_prompt");
    eprintln!("seq_a     = {seq_a:?}");
    eprintln!("seq_b     = {seq_b:?}");
    eprintln!("seq_short = {seq_short:?}");

    // --- Gate 1: B=1 vs sequential ---
    {
        let mut bs = GlimmerDecodeBatchState::new(&mut gpu, &cfg, 2, LANE_CAPACITY)
            .expect("batch state B=1");
        let bat = batch_lane_greedy(&mut gpu, &cfg, &weights, &mut bs, 0, &prompt_a, STEPS, 1)
            .expect("batch B=1");
        assert_tokens_eq("gate1 B=1 vs sequential", &seq_a, &bat);
        bs.free_gpu(&mut gpu);
    }

    // --- Gate 2: B=2 isolation (lane0=prompt_a, lane1=prompt_b) ---
    {
        let mut bs = GlimmerDecodeBatchState::new(&mut gpu, &cfg, 2, LANE_CAPACITY)
            .expect("batch state B=2");

        let ok0 = bs
            .prefill_lane_cancellable(&mut gpu, &weights, &cfg, 0, &prompt_a, &mut || false)
            .expect("prefill lane0");
        let ok1 = bs
            .prefill_lane_cancellable(&mut gpu, &weights, &cfg, 1, &prompt_b, &mut || false)
            .expect("prefill lane1");
        assert!(ok0 && ok1, "prefill cancelled");

        let (mut t0, _) = bs
            .sample_lane_product(&mut gpu, &cfg, 0, 0.0, 1.0, None, 1)
            .expect("pending lane0");
        let (mut t1, _) = bs
            .sample_lane_product(&mut gpu, &cfg, 1, 0.0, 1.0, None, 1)
            .expect("pending lane1");
        let mut p0 = prompt_a.len();
        let mut p1 = prompt_b.len();
        let mut out0 = Vec::with_capacity(STEPS);
        let mut out1 = Vec::with_capacity(STEPS);

        for step in 0..STEPS {
            out0.push(t0);
            out1.push(t1);
            let tokens = vec![t0, t1];
            let positions = vec![p0, p1];
            forward_decode_batch_glimmer(
                &mut gpu, &weights, &cfg, &tokens, &positions, 0b11, &mut bs,
            )
            .unwrap_or_else(|e| panic!("B=2 forward step {step}: {e}"));
            let (n0, _) = bs
                .sample_lane_product(&mut gpu, &cfg, 0, 0.0, 1.0, None, 1)
                .expect("sample lane0");
            let (n1, _) = bs
                .sample_lane_product(&mut gpu, &cfg, 1, 0.0, 1.0, None, 1)
                .expect("sample lane1");
            t0 = n0;
            t1 = n1;
            p0 += 1;
            p1 += 1;
        }

        assert_tokens_eq("gate2 B=2 lane0 vs sequential A", &seq_a, &out0);
        assert_tokens_eq("gate2 B=2 lane1 vs sequential B", &seq_b, &out1);
        bs.free_gpu(&mut gpu);
    }

    // --- Gate 3: hole mask 0b101 (max_batch>=3) ---
    //
    // Contract: inactive KV/attention are masked. Scratch and the dense lm_head
    // prefix of length batch_size may be touched for hole rows (see
    // forward_decode_batch_glimmer docs). Observable durable check:
    // after a 0b101 tick that never activated lane 1, reset+prefill lane 1 and
    // prove its 8-token greedy stream equals the isolated sequential reference.
    {
        let mut bs = GlimmerDecodeBatchState::new(&mut gpu, &cfg, 4, LANE_CAPACITY)
            .expect("batch state hole");
        let ok0 = bs
            .prefill_lane_cancellable(&mut gpu, &weights, &cfg, 0, &prompt_a, &mut || false)
            .expect("hole prefill lane0");
        let ok2 = bs
            .prefill_lane_cancellable(&mut gpu, &weights, &cfg, 2, &prompt_b, &mut || false)
            .expect("hole prefill lane2");
        assert!(ok0 && ok2, "hole prefill cancelled");

        let (t0, _) = bs
            .sample_lane_product(&mut gpu, &cfg, 0, 0.0, 1.0, None, 1)
            .expect("hole pending lane0");
        let (t2, _) = bs
            .sample_lane_product(&mut gpu, &cfg, 2, 0.0, 1.0, None, 1)
            .expect("hole pending lane2");

        // Snapshot active-lane pending picks so we can prove they advanced,
        // and that sample_product leaves inactive RNG untouched.
        let mut rngs = [11u32, 22u32, 33u32, 44u32];
        let rngs_before = rngs;

        let tokens = vec![t0, 0, t2, 0];
        let positions = vec![prompt_a.len(), 0, prompt_b.len(), 0];
        let mask = 0b101u64;
        forward_decode_batch_glimmer(&mut gpu, &weights, &cfg, &tokens, &positions, mask, &mut bs)
            .expect("hole forward");

        let sampled = bs
            .sample_product(&mut gpu, &cfg, 0.0, 1.0, None, &mut rngs, mask)
            .expect("hole sample_product");
        assert!(sampled[0].is_some(), "lane0 should sample");
        assert!(sampled[1].is_none(), "lane1 hole must not sample");
        assert!(sampled[2].is_some(), "lane2 should sample");
        assert!(sampled[3].is_none(), "lane3 hole must not sample");
        assert_eq!(rngs[1], rngs_before[1], "inactive lane1 RNG advanced");
        assert_eq!(rngs[3], rngs_before[3], "inactive lane3 RNG advanced");
        // Greedy leaves RNG unchanged for active lanes too (argmax path).
        assert_eq!(rngs[0], rngs_before[0]);
        assert_eq!(rngs[2], rngs_before[2]);
        eprintln!(
            "PASS gate3 hole sample_product mask={mask:#b} active picks={:?}/{:?}",
            sampled[0].map(|(t, _)| t),
            sampled[2].map(|(t, _)| t)
        );

        // Subsequent lane-1 isolated decode parity: no durable inactive write.
        bs.reset_lane(&mut gpu, 1).expect("reset lane1");
        let out1 = batch_lane_greedy(&mut gpu, &cfg, &weights, &mut bs, 1, &prompt_b, STEPS, 4)
            .expect("post-hole lane1 isolated");
        assert_tokens_eq(
            "gate3 post-hole lane1 isolated vs sequential B",
            &seq_b,
            &out1,
        );
        bs.free_gpu(&mut gpu);
    }

    // --- Gate 4: reset/reuse longer lane with shorter prompt ---
    {
        let mut bs = GlimmerDecodeBatchState::new(&mut gpu, &cfg, 2, LANE_CAPACITY)
            .expect("batch state reset");
        let ok = bs
            .prefill_lane_cancellable(&mut gpu, &weights, &cfg, 0, &long_prompt, &mut || false)
            .expect("long prefill");
        assert!(ok, "long prefill cancelled");
        let (pending_long, _) = bs
            .sample_lane_product(&mut gpu, &cfg, 0, 0.0, 1.0, None, 1)
            .expect("long pending");
        forward_decode_batch_glimmer(
            &mut gpu,
            &weights,
            &cfg,
            &[pending_long, 0],
            &[long_prompt.len(), 0],
            0b01,
            &mut bs,
        )
        .expect("long decode tick");

        bs.reset_lane(&mut gpu, 0).expect("reset_lane 0");
        let short_bat = batch_lane_greedy(
            &mut gpu,
            &cfg,
            &weights,
            &mut bs,
            0,
            &short_prompt,
            STEPS,
            1,
        )
        .expect("short after reset");
        assert_tokens_eq(
            "gate4 reset/reuse short vs sequential short",
            &seq_short,
            &short_bat,
        );
        bs.free_gpu(&mut gpu);
    }

    // --- Gate 5: free lifecycle ---
    {
        let mut bs = GlimmerDecodeBatchState::new(&mut gpu, &cfg, 2, LANE_CAPACITY)
            .expect("lifecycle alloc1");
        bs.reset(&mut gpu).expect("lifecycle reset");
        bs.free_gpu(&mut gpu);
        let mut bs2 = GlimmerDecodeBatchState::new(&mut gpu, &cfg, 2, LANE_CAPACITY)
            .expect("lifecycle alloc2");
        bs2.free_gpu(&mut gpu);
        eprintln!("PASS gate5 free_gpu lifecycle");
    }

    eprintln!("All batch parity/lifecycle checks passed.");
}
