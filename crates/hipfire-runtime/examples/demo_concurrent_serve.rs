// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// demo_concurrent_serve — SP4 Task 3, the concurrent-serve demo.
//
// `demo_multislot_generate` (SP3 Task 5) showed N sequences advancing
// together through `Scheduler` + `forward_batch_slots` directly against a
// `SlotPool`. This demo adds the layer SP4 built on top: real clients don't
// get a slot for free — every admission goes through `AdmissionController`
// (the production memory gate), and sessions are addressed through
// `SessionTable`, not a raw `SlotId`. The budget gate being OBSERVABLE —
// printed decisions, including rejections with their numbers — is half the
// value of this demo; the other half is that the generation loop
// underneath is exactly SP3's, unmodified, reached through SP4's plumbing.
//
// REAL NUMBERS, NOT THE 27B/35B-A3B TABLE. The programme's measured budget
// facts —
//
//   | model              | weights | KV/token | 4x128K            | 4x96K |
//   |--------------------|---------|----------|--------------------|-------|
//   | qwen3.6:27b        | 15.0 GiB| 34 KiB   | exactly 32 GiB —   | 28.7  |
//   |                    |         |          | rejected, 0 headrm | fits  |
//   | qwen3.6:35b-a3b    | ~20 GiB | 10.6 KiB | 25.8 GiB — fits    | fits  |
//
// — describe `qwen3.6:27b` and `qwen3.6:35b-a3b`. Neither is a dense Q8_0
// checkpoint: `forward_batch_slots` is Q8_0-only and refuses MoE layers
// (see its module doc), and the only confirmed dense-Q8_0 Qwen3.5
// checkpoint on this box is the same 4B model `test_forward_slots_golden`
// and `demo_multislot_generate` use. Wiring `AdmissionController` up with
// the 27B's numbers while the GPU underneath is actually running a 4B
// model would print a "production" budget decision about memory nothing
// in the process would ever actually touch — a plausible-looking
// simulation dressed as a real one. Instead, `AdmissionController` here is
// built from THIS run's own measured `ModelFootprint` (the loaded model's
// real weight_bytes and real per-token KV cost, computed exactly like
// `demo_multislot_generate`'s preflight arithmetic) against the same 32
// GiB R9700 deployment-target budget `preflight_alloc` uses elsewhere.
// Every number this demo prints is about the model actually resident on
// the GPU. The table above is reproduced solely as the programme's
// documented reference point, not fed into the live gate.
//
// SETUP REUSE. Model loading, arena/scratch allocation and preflight
// accounting are the same as `demo_multislot_generate.rs` (itself copied
// from `test_forward_slots_golden.rs`), not reinvented.
//
// Env vars:
//   N_SLOTS         number of concurrent, WITHIN-BUDGET sessions (default 3)
//   MAX_NEW_TOKENS  tokens generated per session (default 20)
//   MODEL_PATH      dense Q8_0 Qwen3.5 checkpoint (default
//                   ~/.hipfire/models/qwen3.5-4b-q8.hf4)
//
// Run only through `scripts/run-bounded.sh`, and only when no daemon holds
// a model resident and MemAvailable is comfortably above what the
// preflight computation below plans to use — see `test_forward_slots_golden.rs`'s
// header for why: on this box the cgroup does not contain amdgpu GTT, and
// an under-provisioned run has previously triggered a GLOBAL OOM that
// killed unrelated user processes.

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet,arch-qwen35");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::forward_slots::{forward_batch_slots, SlotDescStaging};
    use hipfire_arch_qwen35::qwen35::{
        self, DeltaNetState, LayerType, PrefillBatchScratch, Qwen35Scratch, Qwen35Weights,
    };
    use hipfire_arch_qwen35::scheduler::{PendingWork, Scheduler};
    use hipfire_runtime::admission::{AdmissionController, AdmitError, ModelFootprint};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::session_table::{SessionId, SessionTable};
    use hipfire_runtime::tokenizer::Tokenizer;
    use rdna_compute::kv_slots::{preflight_alloc, R9700_VRAM_BYTES};
    use rdna_compute::sampling::SlotSampleParams;
    use rdna_compute::slot_pool::{SlotId, SlotPool};
    use rdna_compute::{DType, Gpu};
    use std::path::Path;

    const GIB: u64 = 1024 * 1024 * 1024;

    const DEFAULT_PROMPTS: [&str; 4] = [
        "The capital of France is",
        "Once upon a time, in a distant galaxy,",
        "The recipe for a good cup of tea starts with",
        "In machine learning, gradient descent works by",
    ];

    let n_slots: usize = std::env::var("N_SLOTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    assert!(n_slots > 0, "N_SLOTS must be positive");
    let max_new_tokens: usize = std::env::var("MAX_NEW_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
        let home = std::env::var("HOME").expect("HOME not set and MODEL_PATH not given");
        format!("{home}/.hipfire/models/qwen3.5-4b-q8.hf4")
    });

    println!("=== demo_concurrent_serve: {n_slots} sessions via SessionTable + AdmissionController ===");
    println!("model: {model_path}");
    println!(
        "(programme reference numbers, NOT used below — the live gate uses this run's own \
         measured footprint: qwen3.6:27b 15.0 GiB/34 KiB-per-tok, 4x128K exactly 32 GiB \
         [rejected], 4x96K 28.7 GiB [fits]; qwen3.6:35b-a3b ~20 GiB/10.6 KiB-per-tok, both fit)"
    );

    // ---- host-only setup, identical to demo_multislot_generate ----
    let mut hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("parse Qwen3.5 config");
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
    let per_pos_bytes = config.n_kv_heads * (config.head_dim / 32) * 34; // Q8_0 K and V, same stride

    let prompts: Vec<String> = (0..n_slots)
        .map(|i| DEFAULT_PROMPTS[i % DEFAULT_PROMPTS.len()].to_string())
        .collect();
    let prompt_tokens: Vec<Vec<u32>> = prompts.iter().map(|p| tokenizer.encode(p)).collect();
    let prompt_lens: Vec<usize> = prompt_tokens.iter().map(|t| t.len()).collect();
    let max_prompt_len = prompt_lens.iter().copied().max().unwrap_or(1);
    let chunk_size = max_prompt_len;
    let cap_tokens = max_prompt_len + max_new_tokens + 8;

    // One extra pool slot beyond the n_slots real sessions, held in
    // reserve and deliberately left unacquired: the over-budget request
    // below must be refused by AdmissionController itself, not merely by
    // the pool being full. Without this spare slot, a PoolFull rejection
    // would look identical to a budget rejection in the printed output —
    // SessionTable.open() calls adm.admit() BEFORE pool.acquire(), so
    // giving it room to spare isolates which gate actually fired.
    let pool_capacity = n_slots + 1;
    let max_batch = prompt_lens.iter().sum::<usize>().max(pool_capacity);

    for (i, (p, toks)) in prompts.iter().zip(&prompt_tokens).enumerate() {
        println!("  session {i} (pending): \"{p}\" ({} prompt tokens)", toks.len());
    }

    // ---- preflight: same itemized accounting as demo_multislot_generate,
    // sized to pool_capacity (n_slots + 1) since that many slabs/DeltaNet
    // states are allocated regardless of how many are ever acquired. ----
    let weight_bytes = std::fs::metadata(&model_path).expect("stat model file").len();
    let cap_rounded = cap_tokens.div_ceil(128) * 128;

    let candidate_kv_bytes = (n_fa_layers as u64)
        * 2
        * (pool_capacity as u64)
        * (cap_rounded as u64)
        * (per_pos_bytes as u64);

    let dn_s_dim = config.linear_key_head_dim;
    let dn_heads = config.linear_num_value_heads;
    let dn_s_size = dn_heads * dn_s_dim * dn_s_dim;
    let dn_conv_channels = config.linear_num_key_heads * config.linear_key_head_dim * 2
        + config.linear_num_value_heads * config.linear_value_head_dim;
    let dn_conv_state_size = dn_conv_channels * config.conv_kernel_dim.saturating_sub(1);
    let per_slot_dn_bytes = (n_delta_layers as u64)
        * (dn_s_size as u64
            + (dn_heads * dn_s_dim) as u64 * 4
            + dn_s_size as u64 * 2
            + dn_conv_state_size as u64 * 4);
    let candidate_dn_bytes = (pool_capacity as u64) * per_slot_dn_bytes;

    let n_steps = 1 + max_new_tokens;
    let host_token_bytes = (n_steps * pool_capacity * 4) as u64;

    let planned = weight_bytes
        + candidate_kv_bytes
        + candidate_dn_bytes
        + host_token_bytes
        + 256 * 1024 * 1024; // PrefillBatchScratch / SlotDescStaging / Qwen35Scratch / logits_out misc, flat slop

    eprintln!(
        "preflight: weights={:.2} GiB, candidate_kv={:.1} MiB, candidate_dn={:.1} MiB, \
         host_tokens={:.3} MiB, planned={:.2} GiB (cap={cap_tokens} tokens/slot, pool={pool_capacity} slots)",
        weight_bytes as f64 / 1073741824.0,
        candidate_kv_bytes as f64 / 1048576.0,
        candidate_dn_bytes as f64 / 1048576.0,
        host_token_bytes as f64 / 1048576.0,
        planned as f64 / 1073741824.0,
    );
    preflight_alloc(planned, R9700_VRAM_BYTES, "demo_concurrent_serve")
        .expect("preflight_alloc refused this configuration");

    let mut gpu = Gpu::init().expect("gpu init");
    let weights: Qwen35Weights = {
        let mut src = qwen35::HfqSource::new(&mut hfq, &config);
        let layout = qwen35::Layout::single(config.n_layers);
        qwen35::load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
    }
    .expect("load weights");

    println!(
        "model loaded: {} layers ({} FullAttention / {} LinearAttention), vocab={}",
        config.n_layers, n_fa_layers, n_delta_layers, config.vocab_size
    );

    // ---- the production memory gate: THIS run's own measured footprint,
    // against the real 32 GiB R9700 deployment-target budget. ----
    let kv_bytes_per_token = (n_fa_layers as u64) * 2 * (per_pos_bytes as u64);
    let footprint = ModelFootprint {
        weights_bytes: weight_bytes,
        kv_bytes_per_token,
    };
    println!(
        "\nAdmissionController footprint (measured, this model): weights={:.2} GiB, \
         kv={:.2} KiB/token, budget={:.0} GiB",
        weight_bytes as f64 / GIB as f64,
        kv_bytes_per_token as f64 / 1024.0,
        R9700_VRAM_BYTES as f64 / GIB as f64,
    );

    let mut adm = AdmissionController::new(footprint, R9700_VRAM_BYTES);
    let mut pool = SlotPool::new(pool_capacity, cap_tokens, per_pos_bytes).expect("SlotPool::new");
    let mut sessions = SessionTable::default();

    // slot index -> the session admitted into it, for building this
    // step's PendingWork. `None` for the spare slot (and for any slot
    // whose session has since closed).
    let mut slot_session: Vec<Option<SessionId>> = vec![None; pool_capacity];

    println!("\n--- admission ---");
    for (i, toks) in prompt_tokens.iter().enumerate() {
        match sessions.open(&mut pool, &mut adm, cap_tokens) {
            Ok(id) => {
                let slot = sessions.get(id).expect("just-opened session must resolve").slot;
                slot_session[slot.0] = Some(id);
                println!(
                    "  ADMITTED session {}: slot={}, granted_ctx={cap_tokens} tokens, \
                     used={:.3} GiB / {:.0} GiB budget  (prompt {i}: {} tokens)",
                    id.0,
                    slot.0,
                    adm.used_bytes() as f64 / GIB as f64,
                    R9700_VRAM_BYTES as f64 / GIB as f64,
                    toks.len(),
                );
            }
            Err(e) => panic!(
                "session {i} unexpectedly rejected ({e}) — N_SLOTS or MAX_NEW_TOKENS is too \
                 large for this demo's budget; shrink one of them"
            ),
        }
    }

    // The deliberately over-budget request. Any realistic per-token KV
    // cost times this many tokens overwhelms a 32 GiB budget by orders of
    // magnitude — the point is not to tune this to exactly the edge (that
    // is what AdmissionController's own unit tests already do, see
    // `admission.rs`'s `the_27b_cannot_take_four_agents_at_128k`), it's to
    // make a REJECTION actually happen here rather than merely being
    // possible in principle.
    const RIDICULOUS_CTX: usize = 1_000_000_000_000; // 1e12 tokens
    println!("\n--- one deliberately over-budget request (RIDICULOUS_CTX={RIDICULOUS_CTX} tokens) ---");
    match sessions.open(&mut pool, &mut adm, RIDICULOUS_CTX) {
        Ok(id) => panic!(
            "session {} was admitted at {RIDICULOUS_CTX} tokens — the budget gate did not fire, \
             which means either the arithmetic regressed or this box's numbers changed enough \
             that RIDICULOUS_CTX is no longer ridiculous",
            id.0
        ),
        Err(AdmitError::WouldExceedBudget { need, available }) => {
            println!(
                "  REJECTED: needs {:.2} GiB but only {:.2} GiB of the budget remains \
                 (need={need} bytes, available={available} bytes) — {e}",
                need as f64 / GIB as f64,
                available as f64 / GIB as f64,
                e = AdmitError::WouldExceedBudget { need, available },
            );
        }
        Err(e @ AdmitError::PoolFull) => {
            panic!(
                "session rejected as PoolFull ({e}), not WouldExceedBudget — the spare pool \
                 slot should have kept this a pure budget rejection; pool_capacity={pool_capacity} \
                 vs n_slots={n_slots} looks wrong"
            );
        }
    }
    println!(
        "  (the spare pool slot stays unacquired: admission ran before pool.acquire(), so the \
     rejected request never touched slot capacity — {} of {pool_capacity} slots remain free)",
        pool_capacity - n_slots
    );

    // ---- arenas / per-slot state, sized to pool_capacity like the
    // preflight above ----
    let arena_bytes = pool.arena_bytes();
    let mut k_arenas = Vec::with_capacity(n_fa_layers);
    let mut v_arenas = Vec::with_capacity(n_fa_layers);
    for _ in 0..n_fa_layers {
        k_arenas.push(gpu.zeros(&[arena_bytes], DType::Raw).expect("alloc k_arena"));
        v_arenas.push(gpu.zeros(&[arena_bytes], DType::Raw).expect("alloc v_arena"));
    }
    let mut dn_states: Vec<DeltaNetState> = (0..pool_capacity)
        .map(|_| DeltaNetState::new(&mut gpu, &config).expect("DeltaNetState::new"))
        .collect();
    let mut desc_staging =
        SlotDescStaging::new(&mut gpu, pool_capacity, max_batch).expect("SlotDescStaging::new");
    let pbs =
        PrefillBatchScratch::new(&mut gpu, &config, max_batch).expect("PrefillBatchScratch::new");
    let scratch = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 64, cap_tokens)
        .expect("Qwen35Scratch::new_with_kv_max");
    let logits_out = gpu
        .zeros(&[pool_capacity * config.vocab_size], DType::F32)
        .expect("alloc logits_out");
    let out_tokens = gpu.zeros(&[pool_capacity], DType::F32).expect("alloc out_tokens");

    let sample_params: Vec<SlotSampleParams> = (0..pool_capacity)
        .map(|_| SlotSampleParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: 0,
        })
        .collect();

    // PendingWork for every pool slot, including the spare — an idle slot
    // contributes zero rows (see scheduler.rs's `an_idle_slot_contributes_nothing`)
    // but forward_batch_slots requires one SlotBatch entry per pool slot
    // regardless of occupancy.
    let mut work: Vec<PendingWork> = (0..pool_capacity)
        .map(|slot_ix| PendingWork {
            slot: SlotId(slot_ix),
            remaining_prompt: slot_session[slot_ix]
                .map(|id| prompt_tokens[id.0 as usize].clone())
                .unwrap_or_default(),
            next_pos: 0,
            decoding: false,
        })
        .collect();

    let mut scheduler = Scheduler { chunk_size };
    let mut n_generated = vec![0usize; pool_capacity];
    // Every slot holding an admitted session starts NOT finished; the
    // spare (unacquired) slot starts finished so it never runs.
    let mut finished: Vec<bool> = slot_session.iter().map(|s| s.is_none()).collect();
    let mut generated_tokens: Vec<Vec<u32>> = vec![Vec::new(); pool_capacity];

    println!("\n--- generating (each line: one step, one token per still-active session) ---");
    for step in 0..n_steps {
        let batch = scheduler.next_batch(&mut work);
        if batch.is_empty() {
            break;
        }

        forward_batch_slots(
            &mut gpu,
            &weights,
            &config,
            &batch,
            &mut pool,
            &mut dn_states,
            &k_arenas,
            &v_arenas,
            &mut desc_staging,
            &pbs,
            &scratch,
            &logits_out,
        )
        .expect("forward_batch_slots");
        gpu.hip.device_synchronize().expect("sync after forward");

        gpu.sample_per_slot(&logits_out, &sample_params, pool_capacity, config.vocab_size, &out_tokens)
            .expect("sample_per_slot");
        gpu.hip.device_synchronize().expect("sync after sample");

        let mut token_ids = vec![0i32; pool_capacity];
        {
            let bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(token_ids.as_mut_ptr() as *mut u8, pool_capacity * 4)
            };
            gpu.hip.memcpy_dtoh(bytes, &out_tokens.buf).expect("download out_tokens");
        }

        print!("  step {step:>3}:");
        for slot_ix in 0..pool_capacity {
            if batch.m_per_slot[slot_ix] == 0 || finished[slot_ix] {
                continue;
            }
            let Some(id) = slot_session[slot_ix] else {
                continue;
            };
            let token = token_ids[slot_ix] as u32;
            let text = tokenizer.decode(&[token]);
            print!("  [session {}] {text:?}", id.0);
            generated_tokens[slot_ix].push(token);
            n_generated[slot_ix] += 1;

            if tokenizer.is_terminator(token) || n_generated[slot_ix] >= max_new_tokens {
                finished[slot_ix] = true;
            } else {
                work[slot_ix].remaining_prompt.push(token);
            }
        }
        println!();

        if finished.iter().all(|&f| f) {
            break;
        }
    }

    println!("\n--- final text per session ---");
    for (slot_ix, s) in slot_session.iter().enumerate() {
        if let Some(id) = s {
            println!(
                "  session {}: \"{}\" + \"{}\"",
                id.0,
                prompts[id.0 as usize],
                tokenizer.decode(&generated_tokens[slot_ix])
            );
        }
    }

    println!("\n--- closing sessions (returns slot + budget together) ---");
    for s in slot_session.iter().flatten() {
        let before = adm.used_bytes();
        sessions.close(&mut pool, &mut adm, *s);
        println!(
            "  closed session {}: used_bytes {:.3} GiB -> {:.3} GiB",
            s.0,
            before as f64 / GIB as f64,
            adm.used_bytes() as f64 / GIB as f64,
        );
    }
    assert_eq!(sessions.active(), 0, "every opened session was closed above");

    // ---- free everything held live ----
    for t in k_arenas {
        let _ = gpu.free_tensor(t);
    }
    for t in v_arenas {
        let _ = gpu.free_tensor(t);
    }
    for dn in dn_states {
        dn.free_gpu(&mut gpu);
    }
    desc_staging.free_gpu(&mut gpu);
    pbs.free_gpu(&mut gpu);
    scratch.free_gpu(&mut gpu);
    let _ = gpu.free_tensor(logits_out);
    let _ = gpu.free_tensor(out_tokens);
    weights.free_gpu(&mut gpu);

    println!("\nALL SESSIONS DONE");
}
