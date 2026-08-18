// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! SP5 gate: reused KV must be output-equivalent to recomputed KV.
//!
//! Two arms, one slot each, greedy (argmax) sampling:
//!
//!   REFERENCE — prefill the whole turn-2 prompt cold, then decode N tokens.
//!   CANDIDATE — prefill turn 1, then `begin_turn()` against the turn-2 prompt
//!               so the shared prefix is reused, prefill only the suffix,
//!               then decode N.
//!
//! They must produce identical token ids. Anything else means reused KV is not
//! equivalent to recomputed KV, which would make the prefix cache silently
//! change model output — the one failure class that matters here, because it
//! surfaces as a subtly worse agent rather than as an error.
//!
//! The unit tests behind `plan_turn`/`begin_turn` prove the arithmetic. Only a
//! real model can prove the KV is actually equivalent.

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
    use hipfire_runtime::admission::{AdmissionController, ModelFootprint};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::session_table::SessionTable;
    use hipfire_runtime::tokenizer::Tokenizer;
    use rdna_compute::kv_slots::{preflight_alloc, R9700_VRAM_BYTES};
    use rdna_compute::sampling::SlotSampleParams;
    use rdna_compute::slot_pool::{SlotId, SlotPool};
    use rdna_compute::{DType, Gpu};
    use std::path::Path;

    const DECODE_N: usize = 24;

    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        let home = std::env::var("HOME").expect("HOME not set");
        format!("{home}/.hipfire/models/qwen3.6-35b-a3b.mq4r")
    });
    println!("=== test_prefix_cache_equivalence ===");
    println!("model: {model_path}");

    let mut hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("parse config");
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("load tokenizer");
    let n_fa_layers = config
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::FullAttention)
        .count();
    let n_delta_layers = config
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::LinearAttention)
        .count();
    let per_pos_bytes = config.n_kv_heads * (config.head_dim / 32) * 34;

    // Turn 2 strictly extends turn 1, which is what makes the prefix reusable.
    let turn1: Vec<u32> = tokenizer.encode("The capital of France is Paris. It is known for");
    let turn2: Vec<u32> = {
        let mut v = turn1.clone();
        v.extend(tokenizer.encode(" its museums and food. The capital of Italy is"));
        v
    };
    assert!(
        turn2.len() > turn1.len(),
        "turn 2 must extend turn 1 or there is no prefix to reuse"
    );
    println!(
        "  turn1 = {} tokens, turn2 = {} tokens, decode {DECODE_N}",
        turn1.len(),
        turn2.len()
    );

    let cap_tokens = turn2.len() + DECODE_N + 8;
    let max_batch = turn2.len().max(1);

    // ---- preflight: itemised, device AND host, mirroring the other harnesses ----
    let weight_bytes = std::fs::metadata(&model_path).expect("stat model").len();
    let cap_rounded = cap_tokens.div_ceil(128) * 128;
    let kv_bytes = (n_fa_layers as u64) * 2 * (cap_rounded as u64) * (per_pos_bytes as u64);
    let dn_s_dim = config.linear_key_head_dim;
    let dn_heads = config.linear_num_value_heads;
    let dn_s_size = dn_heads * dn_s_dim * dn_s_dim;
    let dn_conv_channels = config.linear_num_key_heads * config.linear_key_head_dim * 2
        + config.linear_num_value_heads * config.linear_value_head_dim;
    let dn_conv_state_size = dn_conv_channels * config.conv_kernel_dim.saturating_sub(1);
    let dn_bytes = (n_delta_layers as u64)
        * (dn_s_size as u64
            + (dn_heads * dn_s_dim) as u64 * 4
            + dn_s_size as u64 * 2
            + dn_conv_state_size as u64 * 4);
    let planned = weight_bytes + kv_bytes + dn_bytes + 256 * 1024 * 1024;
    eprintln!(
        "preflight: weights={:.2} GiB, kv={:.1} MiB, dn={:.1} MiB, planned={:.2} GiB",
        weight_bytes as f64 / 1073741824.0,
        kv_bytes as f64 / 1048576.0,
        dn_bytes as f64 / 1048576.0,
        planned as f64 / 1073741824.0,
    );
    preflight_alloc(planned, R9700_VRAM_BYTES, "test_prefix_cache_equivalence")
        .expect("preflight_alloc refused this configuration");

    let mut gpu = Gpu::init().expect("gpu init");
    let weights: Qwen35Weights = {
        let mut src = qwen35::HfqSource::new(&mut hfq, &config);
        let layout = qwen35::Layout::single(config.n_layers);
        qwen35::load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
    }
    .expect("load weights");
    println!(
        "model loaded: {} layers ({n_fa_layers} FA / {n_delta_layers} LA), vocab={}",
        config.n_layers, config.vocab_size
    );

    let mut pool = SlotPool::new(1, cap_tokens, per_pos_bytes).expect("SlotPool::new");
    let arena_bytes = pool.arena_bytes();
    let mut k_arenas = Vec::with_capacity(n_fa_layers);
    let mut v_arenas = Vec::with_capacity(n_fa_layers);
    for _ in 0..n_fa_layers {
        k_arenas.push(gpu.zeros(&[arena_bytes], DType::Raw).expect("k arena"));
        v_arenas.push(gpu.zeros(&[arena_bytes], DType::Raw).expect("v arena"));
    }
    let mut dn_states: Vec<DeltaNetState> =
        vec![DeltaNetState::new(&mut gpu, &config).expect("DeltaNetState::new")];
    let mut desc_staging = SlotDescStaging::new(&mut gpu, 1, max_batch).expect("SlotDescStaging");
    let pbs = PrefillBatchScratch::new(&mut gpu, &config, max_batch).expect("PrefillBatchScratch");
    let scratch =
        Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 64, cap_tokens).expect("Qwen35Scratch");
    let logits_out = gpu
        .zeros(&[config.vocab_size], DType::F32)
        .expect("logits_out");
    let out_tokens = gpu.zeros(&[1], DType::F32).expect("out_tokens");
    let sample_params = vec![SlotSampleParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        seed: 0,
    }];

    // One arm: optionally warm the slot with `warm`, then begin_turn against
    // `prompt`, prefill only the suffix, and greedily decode DECODE_N tokens.
    #[allow(clippy::too_many_arguments)]
    fn run_arm(
        gpu: &mut Gpu,
        weights: &Qwen35Weights,
        config: &qwen35::Qwen35Config,
        pool: &mut SlotPool,
        dn_states: &mut [DeltaNetState],
        k_arenas: &[rdna_compute::GpuTensor],
        v_arenas: &[rdna_compute::GpuTensor],
        desc_staging: &mut SlotDescStaging,
        pbs: &PrefillBatchScratch,
        scratch: &Qwen35Scratch,
        logits_out: &rdna_compute::GpuTensor,
        out_tokens: &rdna_compute::GpuTensor,
        sample_params: &[SlotSampleParams],
        warm: Option<&[u32]>,
        prompt: &[u32],
        decode_n: usize,
        chunk: usize,
        label: &str,
    ) -> Vec<u32> {
        // Fresh slot and fresh recurrent state, so the arms cannot leak into
        // each other. DeltaNet state is NOT in the KV arena; forgetting it
        // here would make the second arm start from the first arm's state.
        // `reset` only zeroes seq_len; it does not hand the slot back. The
        // previous arm's SessionTable was dropped still holding it, so release
        // first or the next `open` gets PoolFull.
        pool.release(SlotId(0));
        pool.reset(SlotId(0));
        dn_states[0].reset(gpu).expect("DeltaNetState::reset");

        let mut adm = AdmissionController::new(
            ModelFootprint {
                weights_bytes: 0,
                kv_bytes_per_token: 0,
            },
            u64::MAX,
        );
        let mut table = SessionTable::default();
        // The pool was reset above, so re-acquire slot 0 for this session.
        let sid = table.open(pool, &mut adm, 0).expect("open session");

        let mut graph = SlotDecodeGraph::new();
        // BOTH arms must chunk prefill identically, or this compares
        // chunked-vs-unchunked prefill rather than reuse-vs-recompute: a
        // 21-token batch and an 11+10 pair take different kernel paths and
        // differ numerically for reasons that have nothing to do with the
        // prefix cache. With `chunk` equal to the warm turn's length, the
        // reference computes 11 then 10 and so does the candidate; the only
        // remaining difference is whether the first chunk was computed in
        // this turn or the previous one.
        let mut sched = Scheduler {
            chunk_size: chunk.max(1),
        };

        // Prefill + decode driver shared by the warm turn and the real turn.
        let mut drive = |gpu: &mut Gpu,
                         pool: &mut SlotPool,
                         dn_states: &mut [DeltaNetState],
                         desc_staging: &mut SlotDescStaging,
                         graph: &mut SlotDecodeGraph,
                         feed: &[u32],
                         start_pos: usize,
                         n_decode: usize|
         -> Vec<u32> {
            let mut work = vec![PendingWork {
                slot: SlotId(0),
                remaining_prompt: feed.to_vec(),
                next_pos: start_pos,
                decoding: false,
            }];
            let mut produced = Vec::new();
            // Prefill may take several chunks. Sampling is only valid once the
            // whole prompt has been consumed -- sampling after an intermediate
            // chunk and appending the result injects a generated token into the
            // MIDDLE of the prompt, which silently corrupts the sequence.
            let steps = feed.len().div_ceil(sched.chunk_size.max(1)) + n_decode + 1;
            for _ in 0..steps {
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
                    graph,
                )
                .expect("forward_batch_slots_graphed");
                gpu.hip.device_synchronize().expect("sync");
                gpu.sample_per_slot(logits_out, sample_params, 1, config.vocab_size, out_tokens)
                    .expect("sample_per_slot");
                gpu.hip.device_synchronize().expect("sync");
                let mut tok = [0i32; 1];
                {
                    let bytes: &mut [u8] =
                        unsafe { std::slice::from_raw_parts_mut(tok.as_mut_ptr() as *mut u8, 4) };
                    gpu.hip
                        .memcpy_dtoh(bytes, &out_tokens.buf)
                        .expect("download token");
                }
                if !work[0].remaining_prompt.is_empty() {
                    // Still prefilling: these logits belong to a mid-prompt
                    // token, not to a position we may sample from.
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
        };

        // Warm turn: prefill only, no decode kept.
        if let Some(w) = warm {
            let _ = drive(gpu, pool, dn_states, desc_staging, &mut graph, w, 0, 0);
            let s = table.get_mut(sid).expect("session");
            s.tokens.extend_from_slice(w);
            s.next_pos = w.len();
        }

        let plan = table
            .begin_turn(pool, sid, prompt)
            .expect("begin_turn must succeed");
        println!(
            "  [{label}] reused {} of {} prompt tokens, prefilling {}",
            plan.reused,
            prompt.len(),
            plan.to_prefill
        );

        let suffix = &prompt[plan.reused..];
        drive(
            gpu,
            pool,
            dn_states,
            desc_staging,
            &mut graph,
            suffix,
            plan.reused,
            decode_n,
        )
    }

    let reference = run_arm(
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
        None,
        &turn2,
        DECODE_N,
        turn1.len(),
        "REFERENCE cold",
    );
    let candidate = run_arm(
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
        Some(&turn1),
        &turn2,
        DECODE_N,
        turn1.len(),
        "CANDIDATE reuse",
    );

    println!("  reference: {reference:?}");
    println!("  candidate: {candidate:?}");
    assert_eq!(
        reference.len(),
        DECODE_N,
        "reference arm did not produce {DECODE_N} tokens"
    );
    assert_eq!(
        reference, candidate,
        "prefix-cache reuse changed the output; reused KV is not equivalent to recomputed KV"
    );
    println!("{DECODE_N}/{DECODE_N} tokens identical — prefix reuse is output-equivalent");

    // Negative control: the comparison must be able to fail.
    let mut corrupted = candidate.clone();
    corrupted[0] = corrupted[0].wrapping_add(1);
    assert_ne!(
        reference, corrupted,
        "negative control: the comparison is not sensitive"
    );
    println!("negative control fired");
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
