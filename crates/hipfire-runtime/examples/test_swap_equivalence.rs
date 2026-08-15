// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! SP6 gate #3: a session that was evicted and restored must produce exactly
//! the tokens it would have produced had it never moved.
//!
//! Structure: one slot, two sessions. Session A is prefilled and decodes a few
//! tokens; then B needs the slot, so A is captured and parked and B takes it;
//! then A is restored and continues. A's full token stream must match a control
//! run of A that was never disturbed.
//!
//! Both tiers are exercised: once with a host budget large enough to keep the
//! snapshot in memory, and once with a budget of zero so it is forced to disk.
//!
//! `assert!(evictions > 0)` is mandatory — a gate that silently never evicts
//! would pass forever while testing nothing.

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet,arch-qwen35");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::forward_slots::{
        forward_batch_slots_graphed, SlotDecodeGraph, SlotDescStaging,
    };
    use hipfire_arch_qwen35::qwen35::{
        self, DeltaNetState, LayerType, PrefillBatchScratch, Qwen35Scratch, Qwen35Weights,
    };
    use hipfire_arch_qwen35::scheduler::{PendingWork, Scheduler};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::swap::snapshot::{capture_slot, restore_slot, SnapshotStamp};
    use hipfire_runtime::swap::SwapManager;
    use hipfire_runtime::tokenizer::Tokenizer;
    use rdna_compute::kv_slots::{preflight_alloc, R9700_VRAM_BYTES};
    use rdna_compute::sampling::SlotSampleParams;
    use rdna_compute::slot_pool::{SlotId, SlotPool};
    use rdna_compute::{DType, Gpu, GpuTensor};
    use std::path::Path;

    const DECODE_BEFORE: usize = 6;
    const DECODE_AFTER: usize = 10;

    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        let home = std::env::var("HOME").expect("HOME not set");
        format!("{home}/.hipfire/models/qwen3.6-35b-a3b.mq4r")
    });
    println!("=== test_swap_equivalence ===\nmodel: {model_path}");

    let mut hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    let n_fa_layers = config
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::FullAttention)
        .count();
    let per_pos_bytes = config.n_kv_heads * (config.head_dim / 32) * 34;

    let prompt_a: Vec<u32> = tokenizer.encode("The capital of France is");
    let prompt_b: Vec<u32> = tokenizer.encode("In machine learning, gradient descent works by");
    let cap_tokens = prompt_a.len().max(prompt_b.len()) + DECODE_BEFORE + DECODE_AFTER + 8;
    let max_batch = cap_tokens;

    let weight_bytes = std::fs::metadata(&model_path).expect("stat").len();
    let cap_rounded = cap_tokens.div_ceil(128) * 128;
    let kv_bytes = (n_fa_layers as u64) * 2 * (cap_rounded as u64) * (per_pos_bytes as u64);
    let planned = weight_bytes + kv_bytes + 512 * 1024 * 1024;
    eprintln!(
        "preflight: weights={:.2} GiB, kv={:.1} MiB, planned={:.2} GiB",
        weight_bytes as f64 / 1073741824.0,
        kv_bytes as f64 / 1048576.0,
        planned as f64 / 1073741824.0
    );
    preflight_alloc(planned, R9700_VRAM_BYTES, "test_swap_equivalence").expect("preflight refused");

    let mut gpu = Gpu::init().expect("gpu init");
    let weights: Qwen35Weights = {
        let mut src = qwen35::HfqSource::new(&mut hfq, &config);
        let layout = qwen35::Layout::single(config.n_layers);
        qwen35::load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
    }
    .expect("load weights");

    let mut pool = SlotPool::new(1, cap_tokens, per_pos_bytes).expect("SlotPool");
    pool.acquire().expect("acquire");
    let arena_bytes = pool.arena_bytes();
    let mut k_arenas = Vec::new();
    let mut v_arenas = Vec::new();
    for _ in 0..n_fa_layers {
        k_arenas.push(gpu.zeros(&[arena_bytes], DType::Raw).expect("k"));
        v_arenas.push(gpu.zeros(&[arena_bytes], DType::Raw).expect("v"));
    }
    let mut dn_states = vec![DeltaNetState::new(&mut gpu, &config).expect("dn")];
    let mut desc_staging = SlotDescStaging::new(&mut gpu, 1, max_batch).expect("staging");
    let pbs = PrefillBatchScratch::new(&mut gpu, &config, max_batch).expect("pbs");
    let scratch =
        Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 64, cap_tokens).expect("scratch");
    let logits_out = gpu.zeros(&[config.vocab_size], DType::F32).expect("logits");
    let out_tokens = gpu.zeros(&[1], DType::F32).expect("out");
    let sample_params = vec![SlotSampleParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        seed: 0,
    }];

    fn dn_buffers(dn: &DeltaNetState) -> Vec<&GpuTensor> {
        let mut v: Vec<&GpuTensor> = Vec::new();
        v.extend(dn.s_matrices.iter());
        v.extend(dn.s_scales.iter());
        v.extend(dn.conv_states.iter());
        v.extend(dn.s_ef_residual.iter());
        v
    }
    let dn_bytes: u64 = dn_buffers(&dn_states[0])
        .iter()
        .map(|t| t.buf.size() as u64)
        .sum();
    let stamp = SnapshotStamp {
        model_hash: 0xFEED_FACE,
        kv_dtype_tag: 1,
        per_pos_bytes: per_pos_bytes as u32,
        n_fa_layers: n_fa_layers as u32,
        dn_layout_version: 1,
        cap: pool.descriptors()[0].cap as u32,
        dn_bytes,
    };

    // Run `feed` (a prompt, or a single token to continue from) and return the
    // tokens produced. Sampling only happens once the feed is fully consumed:
    // sampling mid-prefill and appending injects a generated token into the
    // middle of the prompt.
    #[allow(clippy::too_many_arguments)]
    fn step(
        gpu: &mut Gpu,
        weights: &Qwen35Weights,
        config: &qwen35::Qwen35Config,
        pool: &mut SlotPool,
        dn_states: &mut [DeltaNetState],
        k_arenas: &[GpuTensor],
        v_arenas: &[GpuTensor],
        desc_staging: &mut SlotDescStaging,
        pbs: &PrefillBatchScratch,
        scratch: &Qwen35Scratch,
        logits_out: &GpuTensor,
        out_tokens: &GpuTensor,
        sample_params: &[SlotSampleParams],
        feed: &[u32],
        start_pos: usize,
        n_decode: usize,
    ) -> Vec<u32> {
        let mut work = vec![PendingWork {
            slot: SlotId(0),
            remaining_prompt: feed.to_vec(),
            next_pos: start_pos,
            decoding: false,
        }];
        let mut sched = Scheduler {
            chunk_size: feed.len().max(1),
        };
        let mut graph = SlotDecodeGraph::new();
        let mut produced = Vec::new();
        for _ in 0..(1 + n_decode) {
            let batch = sched.next_batch(&mut work);
            if batch.is_empty() {
                break;
            }
            forward_batch_slots_graphed(
                gpu,
                weights,
                config,
                &batch,
                pool,
                dn_states,
                k_arenas,
                v_arenas,
                desc_staging,
                pbs,
                scratch,
                logits_out,
                &mut graph,
            )
            .expect("forward");
            gpu.hip.device_synchronize().expect("sync");
            gpu.sample_per_slot(logits_out, sample_params, 1, config.vocab_size, out_tokens)
                .expect("sample");
            gpu.hip.device_synchronize().expect("sync");
            let mut tok = [0i32; 1];
            {
                let bytes: &mut [u8] =
                    unsafe { std::slice::from_raw_parts_mut(tok.as_mut_ptr() as *mut u8, 4) };
                gpu.hip.memcpy_dtoh(bytes, &out_tokens.buf).expect("dtoh");
            }
            if !work[0].remaining_prompt.is_empty() {
                continue;
            }
            if produced.len() < n_decode {
                produced.push(tok[0] as u32);
                work[0].remaining_prompt.push(tok[0] as u32);
            } else {
                break;
            }
        }
        produced
    }

    macro_rules! run_step {
        ($feed:expr, $pos:expr, $n:expr) => {
            step(
                &mut gpu,
                &weights,
                &config,
                &mut pool,
                &mut dn_states,
                &k_arenas,
                &v_arenas,
                &mut desc_staging,
                &pbs,
                &scratch,
                &logits_out,
                &out_tokens,
                &sample_params,
                $feed,
                $pos,
                $n,
            )
        };
    }

    // ---- CONTROL: session A, never disturbed ----
    pool.reset(SlotId(0));
    dn_states[0].reset(&mut gpu).expect("dn reset");
    let mut control = run_step!(&prompt_a, 0, DECODE_BEFORE);
    let ctl_pos = prompt_a.len() + control.len();
    let ctl_last = *control.last().expect("some tokens");
    control.extend(run_step!(&[ctl_last], ctl_pos, DECODE_AFTER));
    println!("  control produced {} tokens", control.len());

    // ---- CANDIDATE: A is evicted for B, then restored ----
    for (tier_label, budget) in [("host", 1u64 << 30), ("disk", 0u64)] {
        let dir = std::env::temp_dir().join(format!("hipfire-swap-equiv-{tier_label}"));
        let mut mgr = SwapManager::new(dir, budget).expect("SwapManager");

        pool.reset(SlotId(0));
        dn_states[0].reset(&mut gpu).expect("dn reset");
        let mut candidate = run_step!(&prompt_a, 0, DECODE_BEFORE);
        let a_pos = prompt_a.len() + candidate.len();
        let a_last = *candidate.last().expect("some tokens");

        // A is evicted: capture its whole state, then park it.
        let dn_refs = dn_buffers(&dn_states[0]);
        let snap = capture_slot(
            &mut gpu,
            &pool,
            SlotId(0),
            &k_arenas,
            &v_arenas,
            &dn_refs,
            &prompt_a,
            stamp,
        )
        .expect("capture A");
        drop(dn_refs);
        mgr.park(1, snap).expect("park A");
        assert_eq!(
            mgr.store().tier_of(1),
            Some(tier_label),
            "snapshot did not land in the {tier_label} tier"
        );

        // B takes the slot, overwriting A's KV and recurrent state.
        pool.reset(SlotId(0));
        dn_states[0].reset(&mut gpu).expect("dn reset");
        let b_out = run_step!(&prompt_b, 0, DECODE_BEFORE);
        assert_eq!(b_out.len(), DECODE_BEFORE, "B must actually have run");

        // A is restored and continues where it left off.
        let restored = mgr.unpark(1).expect("unpark A");
        let dn_refs = dn_buffers(&dn_states[0]);
        restore_slot(
            &mut gpu,
            &mut pool,
            SlotId(0),
            &k_arenas,
            &v_arenas,
            &dn_refs,
            &restored,
            stamp,
        )
        .expect("restore A");
        drop(dn_refs);
        candidate.extend(run_step!(&[a_last], a_pos, DECODE_AFTER));

        let (evictions, restores, failures) = mgr.stats();
        println!("  [{tier_label}] evictions={evictions} restores={restores} failures={failures}");
        assert!(evictions > 0, "gate proved nothing: no eviction occurred");
        assert!(restores > 0, "gate proved nothing: no restore occurred");
        assert_eq!(failures, 0, "unexpected swap failure");
        assert_eq!(
            control, candidate,
            "[{tier_label}] eviction+restore changed the output"
        );
        println!(
            "  [{tier_label}] {} tokens identical to control",
            control.len()
        );
    }

    println!("ALL CHECKS PASS");

    for t in k_arenas {
        let _ = gpu.free_tensor(t);
    }
    for t in v_arenas {
        let _ = gpu.free_tensor(t);
    }
    for dn in dn_states {
        dn.free_gpu(&mut gpu);
    }
}
