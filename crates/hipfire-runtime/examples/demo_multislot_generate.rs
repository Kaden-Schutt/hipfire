// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// demo_multislot_generate — SP3 Task 5, the programme's first visible
// result.
//
// SP1 gave the attention kernels a slot dimension. SP2 built the per-slot
// state. SP3 built `SlotBatch`, `forward_batch_slots` and the round-robin
// `Scheduler`. Nothing before this file has actually shown several
// sequences advancing together on one GPU — every prior harness in this
// programme (SP3 Task 3's `test_forward_slots_golden`) proved the N-slot
// forward matches the single-sequence path NUMERICALLY, on synthetic token
// streams, never printing a token as text. This demo is the first thing
// that runs real prompts, through a real tokenizer, N at once, and prints
// the interleaving as it happens.
//
// SCOPE. `forward_batch_slots` is Q8_0-only and refuses MoE layers (see its
// module doc), so this demo requires a DENSE Q8_0 Qwen3.5 checkpoint —
// `qwen3.5-4b-q8.hf4` by default, the same file `test_forward_slots_golden`
// uses.
//
// SETUP REUSE. Model loading (`HfqFile` + `qwen35::config_from_hfq` +
// `qwen35::load_weights`), arena/scratch allocation
// (`SlotDescStaging`/`PrefillBatchScratch`/`Qwen35Scratch`) and the
// itemized `preflight_alloc` accounting are all copied from
// `test_forward_slots_golden.rs` rather than reinvented, per the brief.
//
// DESIGN: ONE RAGGED PREFILL STEP, THEN UNIFORM DECODE. Each slot's
// `chunk_size` is set to at least the longest prompt, so the FIRST
// `Scheduler::next_batch` call drains every slot's whole prompt in one
// step — a genuinely ragged batch (different row counts per slot) that
// exercises exactly the shape `SlotBatch` exists for. `final_logits_per_slot`
// (inside `forward_batch_slots`) always gathers only each active slot's
// LAST row, so the very same per-slot logits row is valid whether that
// slot just finished a 20-token prefill or a 1-token decode — no special
// casing is needed to know when a slot's row is "ready to sample". After
// step 0 every slot is decoding, one sampled token per slot per step,
// which is where the interleaving becomes visible: each step's printed
// line has one token per still-active slot, side by side.
//
// A slot that hits `MAX_NEW_TOKENS` or emits a terminator token before the
// others simply stops contributing rows (see `PendingWork`'s "idle slot"
// contract in `scheduler.rs`) while the rest keep decoding — this alone
// demonstrates that slots are independent, not lock-stepped.
//
// Env vars:
//   N_SLOTS         number of concurrent sequences (default 3)
//   MAX_NEW_TOKENS  tokens generated per slot (default 20)
//   MODEL_PATH      dense Q8_0 Qwen3.5 checkpoint (default
//                   ~/.hipfire/models/qwen3.5-4b-q8.hf4)
//
// Run only through `scripts/run-bounded.sh`, and only when no daemon holds
// a model resident and MemAvailable is comfortably above what the
// preflight computation below plans to use — see this file's header in
// `test_forward_slots_golden.rs` for why: on this box the cgroup does not
// contain amdgpu GTT, and an under-provisioned run has previously
// triggered a GLOBAL OOM that killed unrelated user processes.

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
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::tokenizer::Tokenizer;
    use rdna_compute::kv_slots::{preflight_alloc, R9700_VRAM_BYTES};
    use rdna_compute::sampling::SlotSampleParams;
    use rdna_compute::slot_pool::{SlotId, SlotPool};
    use rdna_compute::{DType, Gpu};
    use std::path::Path;

    /// Distinct, short prompts so N slots produce visibly different streams.
    /// Cycled with `% DEFAULT_PROMPTS.len()` when N_SLOTS exceeds 4.
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

    println!("=== demo_multislot_generate: {n_slots} sequences, {max_new_tokens} tokens each ===");
    println!("model: {model_path}");

    // ---- host-only setup: open the file, parse config and tokenizer
    // before any GPU allocation, exactly as test_forward_slots_golden does,
    // so the preflight check below is computed from real values. ----
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
    let max_batch = prompt_lens.iter().sum::<usize>().max(n_slots);

    // Chunk size at least as large as the longest prompt: every slot's
    // whole prompt drains in Scheduler's FIRST call, so step 0 is one
    // ragged prefill batch and every step after that is pure decode — see
    // the module doc for why this makes "is this slot's logits row ready
    // to sample" trivially always-yes.
    let chunk_size = max_prompt_len;
    // Cap far below the 96K/128K budget figures this program is built
    // toward — this demo's whole point is the interleaving, not context
    // depth, and a deliberately small cap keeps the run well inside the
    // 32 GiB deployment target with room to spare.
    let cap_tokens = max_prompt_len + max_new_tokens + 8;

    for (i, (p, toks)) in prompts.iter().zip(&prompt_tokens).enumerate() {
        println!("  slot {i}: \"{p}\" ({} prompt tokens)", toks.len());
    }

    // ---- preflight: itemized, not a magic number — mirrors
    // test_forward_slots_golden's accounting (device AND host, the TOTAL
    // held live at once). No "reference arm" here (this demo has only the
    // candidate path), so this is simpler than the golden harness's. ----
    let weight_bytes = std::fs::metadata(&model_path).expect("stat model file").len();
    let cap_rounded = cap_tokens.div_ceil(128) * 128;

    let candidate_kv_bytes =
        (n_fa_layers as u64) * 2 * (n_slots as u64) * (cap_rounded as u64) * (per_pos_bytes as u64);

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
    let candidate_dn_bytes = (n_slots as u64) * per_slot_dn_bytes;

    // Host-side downloads: only per-slot token ids (4 bytes/slot/step) —
    // unlike the golden harness this demo samples on-device via
    // `sample_per_slot` and never downloads full logits.
    let n_steps = 1 + max_new_tokens;
    let host_token_bytes = (n_steps * n_slots * 4) as u64;

    let planned = weight_bytes
        + candidate_kv_bytes
        + candidate_dn_bytes
        + host_token_bytes
        + 256 * 1024 * 1024; // PrefillBatchScratch / SlotDescStaging / Qwen35Scratch / logits_out misc, flat slop

    eprintln!(
        "preflight: weights={:.2} GiB, candidate_kv={:.1} MiB, candidate_dn={:.1} MiB, \
         host_tokens={:.3} MiB, planned={:.2} GiB (cap={cap_tokens} tokens/slot)",
        weight_bytes as f64 / 1073741824.0,
        candidate_kv_bytes as f64 / 1048576.0,
        candidate_dn_bytes as f64 / 1048576.0,
        host_token_bytes as f64 / 1048576.0,
        planned as f64 / 1073741824.0,
    );
    preflight_alloc(planned, R9700_VRAM_BYTES, "demo_multislot_generate")
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

    // ---- SlotPool + per-slot state ----
    let mut pool = SlotPool::new(n_slots, cap_tokens, per_pos_bytes).expect("SlotPool::new");
    for s in 0..n_slots {
        let id = pool.acquire().expect("SlotPool::acquire");
        assert_eq!(id.0, s, "SlotPool handed out slots out of order");
    }

    let arena_bytes = pool.arena_bytes();
    let mut k_arenas = Vec::with_capacity(n_fa_layers);
    let mut v_arenas = Vec::with_capacity(n_fa_layers);
    for _ in 0..n_fa_layers {
        k_arenas.push(gpu.zeros(&[arena_bytes], DType::Raw).expect("alloc k_arena"));
        v_arenas.push(gpu.zeros(&[arena_bytes], DType::Raw).expect("alloc v_arena"));
    }
    let mut dn_states: Vec<DeltaNetState> = (0..n_slots)
        .map(|_| DeltaNetState::new(&mut gpu, &config).expect("DeltaNetState::new"))
        .collect();
    let mut desc_staging =
        SlotDescStaging::new(&mut gpu, n_slots, max_batch).expect("SlotDescStaging::new");
    let pbs = PrefillBatchScratch::new(&mut gpu, &config, max_batch).expect("PrefillBatchScratch::new");
    let scratch =
        Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 64, cap_tokens).expect("Qwen35Scratch::new_with_kv_max");
    let logits_out = gpu
        .zeros(&[n_slots * config.vocab_size], DType::F32)
        .expect("alloc logits_out");
    // Argmax destination: i32 indices stored in F32-typed slots (the
    // convention `argmax_f32_batched`'s other call sites use throughout
    // this codebase — see e.g. speculative.rs's `VerifyScratch::argmax`).
    let out_tokens = gpu.zeros(&[n_slots], DType::F32).expect("alloc out_tokens");

    // Greedy for every slot: deterministic output, and it takes
    // `sample_per_slot`'s `argmax_f32_batched` fast path for the whole
    // batch every step rather than one kernel launch per slot.
    let sample_params: Vec<SlotSampleParams> = (0..n_slots)
        .map(|_| SlotSampleParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: 0,
        })
        .collect();

    let mut work: Vec<PendingWork> = (0..n_slots)
        .map(|s| PendingWork {
            slot: SlotId(s),
            remaining_prompt: prompt_tokens[s].clone(),
            next_pos: 0,
            decoding: false,
        })
        .collect();

    let mut scheduler = Scheduler { chunk_size };
    let mut n_generated = vec![0usize; n_slots];
    let mut finished = vec![false; n_slots];
    let mut generated_tokens: Vec<Vec<u32>> = vec![Vec::new(); n_slots];

    println!("\n--- generating (each line: one step, one token per still-active slot) ---");

    // Safety net against an infinite loop from a logic error above; a
    // correct run always terminates within n_steps.
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

        gpu.sample_per_slot(&logits_out, &sample_params, n_slots, config.vocab_size, &out_tokens)
            .expect("sample_per_slot");
        gpu.hip.device_synchronize().expect("sync after sample");

        let mut token_ids = vec![0i32; n_slots];
        {
            let bytes: &mut [u8] =
                unsafe { std::slice::from_raw_parts_mut(token_ids.as_mut_ptr() as *mut u8, n_slots * 4) };
            gpu.hip.memcpy_dtoh(bytes, &out_tokens.buf).expect("download out_tokens");
        }

        print!("  step {step:>3}:");
        for s in 0..n_slots {
            if batch.m_per_slot[s] == 0 || finished[s] {
                continue;
            }
            let token = token_ids[s] as u32;
            let text = tokenizer.decode(&[token]);
            print!("  [slot {s}] {text:?}");
            generated_tokens[s].push(token);
            n_generated[s] += 1;

            if tokenizer.is_terminator(token) || n_generated[s] >= max_new_tokens {
                finished[s] = true;
            } else {
                work[s].remaining_prompt.push(token);
            }
        }
        println!();

        if finished.iter().all(|&f| f) {
            break;
        }
    }

    println!("\n--- final text per slot ---");
    for s in 0..n_slots {
        println!(
            "  slot {s}: \"{}\" + \"{}\"",
            prompts[s],
            tokenizer.decode(&generated_tokens[s])
        );
    }

    // ---- free everything held live, per-loop-iteration allocations were
    // never accumulated (there were none — pbs/scratch/arenas are
    // allocated once, before the loop, and reused every step). ----
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

    println!("\nALL SLOTS DONE");
}
