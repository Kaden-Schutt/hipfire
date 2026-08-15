// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! SP6 gate #1/#2: a slot's KV **and DeltaNet state** round-trip bitwise, and
//! a bad snapshot is refused rather than restored.
//!
//! The unit tests prove the container is sound. They cannot prove the GPU
//! copies address the right bytes, which is what this does — and it is the
//! gate that catches a forgotten DeltaNet state, the failure mode that would
//! otherwise pass a short smoke test and corrupt long conversations silently.
//!
//! Three controls, all necessary:
//!
//!   * **scribble** — the slot is overwritten with junk between capture and
//!     restore, so a restore that copies nothing cannot pass.
//!   * **capture sees the scribble** — a capture taken while the slot is junk
//!     must differ from the original, which proves capture reads the region
//!     the scribble touched rather than some unrelated offset.
//!   * **corruption and stamp mismatch are refused**, not silently accepted.

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
    use hipfire_runtime::swap::snapshot::{
        capture_slot, restore_slot, SlotSnapshot, SnapshotStamp,
    };
    use hipfire_runtime::swap::SwapError;
    use hipfire_runtime::tokenizer::Tokenizer;
    use rdna_compute::kv_slots::{preflight_alloc, R9700_VRAM_BYTES};
    use rdna_compute::slot_pool::{SlotId, SlotPool};
    use rdna_compute::{DType, Gpu, GpuTensor};
    use std::path::Path;

    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        let home = std::env::var("HOME").expect("HOME not set");
        format!("{home}/.hipfire/models/qwen3.6-35b-a3b.mq4r")
    });
    println!("=== test_swap_roundtrip ===\nmodel: {model_path}");

    let mut hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("parse config");
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
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

    let prompt: Vec<u32> =
        tokenizer.encode("The capital of France is Paris, a city known for its museums.");
    let cap_tokens = prompt.len() + 8;
    let max_batch = prompt.len().max(1);

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
    preflight_alloc(planned, R9700_VRAM_BYTES, "test_swap_roundtrip")
        .expect("preflight_alloc refused this configuration");

    let mut gpu = Gpu::init().expect("gpu init");
    let weights: Qwen35Weights = {
        let mut src = qwen35::HfqSource::new(&mut hfq, &config);
        let layout = qwen35::Layout::single(config.n_layers);
        qwen35::load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
    }
    .expect("load weights");
    println!("model loaded: {n_fa_layers} FA / {n_delta_layers} LA layers");

    let mut pool = SlotPool::new(1, cap_tokens, per_pos_bytes).expect("SlotPool");
    pool.acquire().expect("acquire");
    let arena_bytes = pool.arena_bytes();
    let mut k_arenas = Vec::new();
    let mut v_arenas = Vec::new();
    for _ in 0..n_fa_layers {
        k_arenas.push(gpu.zeros(&[arena_bytes], DType::Raw).expect("k arena"));
        v_arenas.push(gpu.zeros(&[arena_bytes], DType::Raw).expect("v arena"));
    }
    let mut dn_states = vec![DeltaNetState::new(&mut gpu, &config).expect("dn state")];
    let mut desc_staging = SlotDescStaging::new(&mut gpu, 1, max_batch).expect("staging");
    let pbs = PrefillBatchScratch::new(&mut gpu, &config, max_batch).expect("pbs");
    let scratch =
        Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 64, cap_tokens).expect("scratch");
    let logits_out = gpu.zeros(&[config.vocab_size], DType::F32).expect("logits");

    // The DeltaNet buffers, in the order `DeltaNetState::reset` uses. This
    // ordering is the contract between capture and restore; both call sites
    // must derive it identically, so it is derived once here.
    fn dn_buffers(dn: &DeltaNetState) -> Vec<&GpuTensor> {
        let mut v: Vec<&GpuTensor> = Vec::new();
        v.extend(dn.s_matrices.iter());
        v.extend(dn.s_scales.iter());
        v.extend(dn.conv_states.iter());
        v.extend(dn.s_ef_residual.iter());
        v
    }

    // ---- populate the slot with real state ----
    let mut work = vec![PendingWork {
        slot: SlotId(0),
        remaining_prompt: prompt.clone(),
        next_pos: 0,
        decoding: false,
    }];
    let mut sched = Scheduler {
        chunk_size: prompt.len(),
    };
    let mut graph = SlotDecodeGraph::new();
    let batch = sched.next_batch(&mut work);
    forward_batch_slots_graphed(
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
        &mut graph,
    )
    .expect("prefill");
    gpu.hip.device_synchronize().expect("sync");
    let seq_len = pool.descriptors()[0].seq_len as usize;
    assert_eq!(seq_len, prompt.len(), "prefill did not advance seq_len");

    let dn_bytes: u64 = dn_buffers(&dn_states[0])
        .iter()
        .map(|t| t.buf.size() as u64)
        .sum();
    let stamp = SnapshotStamp {
        model_hash: 0xDEAD_BEEF,
        kv_dtype_tag: 1,
        per_pos_bytes: per_pos_bytes as u32,
        n_fa_layers: n_fa_layers as u32,
        dn_layout_version: 1,
        cap: pool.descriptors()[0].cap as u32,
        dn_bytes,
    };
    println!(
        "  slot holds {seq_len} tokens: KV {:.2} MiB + DN {:.2} MiB",
        (n_fa_layers * 2 * seq_len * per_pos_bytes) as f64 / 1048576.0,
        dn_bytes as f64 / 1048576.0
    );

    let dn_refs = dn_buffers(&dn_states[0]);
    let before = capture_slot(
        &mut gpu,
        &pool,
        SlotId(0),
        &k_arenas,
        &v_arenas,
        &dn_refs,
        &prompt,
        stamp,
    )
    .expect("capture");
    drop(dn_refs);

    // ---- scribble: a restore that copies nothing must not pass ----
    for a in k_arenas.iter().chain(v_arenas.iter()) {
        gpu.hip
            .memset(&a.buf, 0xAB, a.buf.size())
            .expect("scribble kv");
    }
    for t in dn_buffers(&dn_states[0]) {
        gpu.hip
            .memset(&t.buf, 0xCD, t.buf.size())
            .expect("scribble dn");
    }
    gpu.hip.device_synchronize().expect("sync");

    // Positive control for capture itself: it must SEE the scribble, which
    // proves it reads the region the scribble touched and not some other one.
    let dn_refs = dn_buffers(&dn_states[0]);
    let scribbled = capture_slot(
        &mut gpu,
        &pool,
        SlotId(0),
        &k_arenas,
        &v_arenas,
        &dn_refs,
        &prompt,
        stamp,
    )
    .expect("capture scribbled");
    drop(dn_refs);
    assert_ne!(
        before.payload, scribbled.payload,
        "capture did not observe the scribble; it is reading the wrong region"
    );
    println!("  capture observes the scribble (reads the right region)");

    // ---- restore and compare ----
    let dn_refs = dn_buffers(&dn_states[0]);
    restore_slot(
        &mut gpu,
        &mut pool,
        SlotId(0),
        &k_arenas,
        &v_arenas,
        &dn_refs,
        &before,
        stamp,
    )
    .expect("restore");
    drop(dn_refs);
    gpu.hip.device_synchronize().expect("sync");

    let dn_refs = dn_buffers(&dn_states[0]);
    let after = capture_slot(
        &mut gpu,
        &pool,
        SlotId(0),
        &k_arenas,
        &v_arenas,
        &dn_refs,
        &prompt,
        stamp,
    )
    .expect("capture after");
    drop(dn_refs);
    assert_eq!(
        before.payload.len(),
        after.payload.len(),
        "payload length changed across the round trip"
    );
    assert_eq!(
        before.payload, after.payload,
        "KV and/or DeltaNet state were not restored bitwise"
    );
    assert_eq!(pool.descriptors()[0].seq_len as usize, seq_len);
    println!(
        "  {} B restored BITWISE (KV + DeltaNet state)",
        after.payload.len()
    );

    // ---- cost breakdown: what a synchronous eviction actually blocks on ----
    // `park` runs on the ENGINE thread inside admit, so whatever it costs
    // stalls every in-flight slot, not just the evicting request.
    {
        let dn_refs = dn_buffers(&dn_states[0]);
        let t = std::time::Instant::now();
        let snap = capture_slot(
            &mut gpu,
            &pool,
            SlotId(0),
            &k_arenas,
            &v_arenas,
            &dn_refs,
            &prompt,
            stamp,
        )
        .expect("capture timing");
        let capture_ms = t.elapsed().as_secs_f64() * 1e3;
        drop(dn_refs);

        let t = std::time::Instant::now();
        let encoded = snap.to_bytes();
        let encode_ms = t.elapsed().as_secs_f64() * 1e3;

        let tmp = std::env::temp_dir().join("hipfire-park-timing.bin");
        let t = std::time::Instant::now();
        std::fs::write(&tmp, &encoded).expect("write timing");
        let write_ms = t.elapsed().as_secs_f64() * 1e3;

        let t = std::time::Instant::now();
        let read_back = std::fs::read(&tmp).expect("read timing");
        let read_ms = t.elapsed().as_secs_f64() * 1e3;
        let t = std::time::Instant::now();
        let parsed = SlotSnapshot::from_bytes(&read_back).expect("parse timing");
        let parse_ms = t.elapsed().as_secs_f64() * 1e3;
        let _ = std::fs::remove_file(&tmp);

        let dn_refs = dn_buffers(&dn_states[0]);
        let t = std::time::Instant::now();
        restore_slot(
            &mut gpu,
            &mut pool,
            SlotId(0),
            &k_arenas,
            &v_arenas,
            &dn_refs,
            &parsed,
            stamp,
        )
        .expect("restore timing");
        let restore_ms = t.elapsed().as_secs_f64() * 1e3;
        drop(dn_refs);

        let mb = snap.payload.len() as f64 / 1e6;
        println!("\n--- eviction cost breakdown ({mb:.1} MB snapshot) ---");
        println!("  capture (GPU->host)   {capture_ms:8.2} ms   <- synchronous either way");
        println!("  to_bytes (host copy)  {encode_ms:8.2} ms   <- only on the disk tier");
        println!("  fs::write             {write_ms:8.2} ms   <- only on the disk tier");
        println!("  ---- host tier park is a Vec move: ~0 ms ----");
        println!("  fs::read              {read_ms:8.2} ms");
        println!("  from_bytes            {parse_ms:8.2} ms");
        println!("  restore (host->GPU)   {restore_ms:8.2} ms");
        println!(
            "  => a disk eviction blocks the engine for ~{:.1} ms; host ~{:.1} ms",
            capture_ms + encode_ms + write_ms,
            capture_ms
        );
    }

    // ---- disk tier: the serialised form must survive a file round trip ----
    let bytes = before.to_bytes();
    let parsed = SlotSnapshot::from_bytes(&bytes).expect("parse");
    assert_eq!(parsed.payload, before.payload, "disk form lost bytes");
    assert!(parsed.validate(stamp).is_ok());
    println!("  disk form round-trips ({} B encoded)", bytes.len());

    // ---- negative controls: each must be REFUSED ----
    let mut corrupt = before.clone();
    corrupt.payload[0] ^= 0xFF;
    let dn_refs = dn_buffers(&dn_states[0]);
    let r = restore_slot(
        &mut gpu,
        &mut pool,
        SlotId(0),
        &k_arenas,
        &v_arenas,
        &dn_refs,
        &corrupt,
        stamp,
    );
    drop(dn_refs);
    assert!(
        matches!(r, Err(SwapError::Corrupt(_))),
        "a corrupted payload must be refused, got {r:?}"
    );

    let mut wrong = stamp;
    wrong.model_hash ^= 1;
    let dn_refs = dn_buffers(&dn_states[0]);
    let r = restore_slot(
        &mut gpu,
        &mut pool,
        SlotId(0),
        &k_arenas,
        &v_arenas,
        &dn_refs,
        &before,
        wrong,
    );
    drop(dn_refs);
    assert!(
        matches!(r, Err(SwapError::Stamp(_))),
        "a stamp mismatch must be refused, got {r:?}"
    );
    println!("  negative controls fired (corruption + stamp mismatch refused)");

    // A refused restore must have left the slot alone.
    let dn_refs = dn_buffers(&dn_states[0]);
    let untouched = capture_slot(
        &mut gpu,
        &pool,
        SlotId(0),
        &k_arenas,
        &v_arenas,
        &dn_refs,
        &prompt,
        stamp,
    )
    .expect("capture untouched");
    drop(dn_refs);
    assert_eq!(
        untouched.payload, after.payload,
        "a refused restore modified the slot; validation must precede any device write"
    );
    println!("  a refused restore left the slot untouched");
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
