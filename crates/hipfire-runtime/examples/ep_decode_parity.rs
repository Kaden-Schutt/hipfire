// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Ship 6 EP (expert-parallel) decode-parity validation for qwen3.x-A3B.
//!
//! Brings up `tp_size` replicated ranks via `Gpus::init_tp`, loads the model
//! **replicated** on every rank, shards each MoE layer's routed experts per
//! rank (`shard_all_moe_layers`), then greedy-decodes a fixed prompt through
//! the EP forward driver (`qwen35::forward_ep` → all-reduce-EP executor).
//!
//! Two gates:
//!   1. **In-process anchor (tp=1 only):** rank 0 owns all experts at tp=1, so
//!      the production single-GPU path (`forward_scratch`) is valid on it. The
//!      example runs BOTH the production path and the EP path over the same
//!      prompt+state and asserts per-step argmax parity + logs max-abs logit
//!      diff. This proves the EP machinery reproduces production on one rank.
//!   2. **Cross-process sharding gate (tp=N vs tp=1):** the example prints the
//!      generated token-id stream and an FNV-1a hash of it. Run at tp=1 and
//!      tp=N (e.g. on hiptrx devices 0+1) and diff the hash — identical means
//!      the expert sharding + all-reduce is argmax-exact.
//!
//! Run (hiptrx; TP=1 reference then TP=2 sharded):
//!   HIP_VISIBLE_DEVICES=0 cargo run --release --features deltanet \
//!       -p hipfire-runtime --example ep_decode_parity -- \
//!       ~/.hipfire/models/qwen3.6-35b-a3b.mq4 1 24 "The capital of France is"
//!   HIP_VISIBLE_DEVICES=0,1 cargo run --release --features deltanet \
//!       -p hipfire-runtime --example ep_decode_parity -- \
//!       ~/.hipfire/models/qwen3.6-35b-a3b.mq4 2 24 "The capital of France is"

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("ep_decode_parity requires --features deltanet");
}

#[cfg(feature = "deltanet")]
fn fnv1a(ids: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &id in ids {
        for b in id.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::{self, KvCache};
    use hipfire_runtime::multi_gpu::Gpus;
    use hipfire_runtime::tp_shard::{ExpertAssign, ShardConfig};
    use rdna_compute::{DType, GpuTensor};
    use std::path::Path;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: ep_decode_parity <model.mq4> [tp_size=2] [steps=24] [prompt]");
        std::process::exit(1);
    }
    let model_path = Path::new(&args[1]);
    let tp: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(2);
    let steps: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(24);
    let prompt = args
        .get(4)
        .cloned()
        .unwrap_or_else(|| "The capital of France is".to_string());
    let kv_seq: usize = std::env::var("HIPFIRE_EP_KV_SEQ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512usize);

    // ── config + tokenizer (one open; per-rank loads reopen below) ──────────
    let hfq0 = HfqFile::open(model_path).expect("open model");
    let config = qwen35::config_from_hfq(&hfq0).expect("read config");
    assert!(config.num_experts > 0, "ep_decode_parity expects a MoE (A3B) model");
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq0.metadata_json)
        .expect("tokenizer");
    drop(hfq0);
    eprintln!(
        "config: layers={} dim={} experts={} top_k={} n_kv_heads={} head_dim={}",
        config.n_layers, config.dim, config.num_experts, config.num_experts_per_tok,
        config.n_kv_heads, config.head_dim,
    );
    // Prefill mode: HIPFIRE_EP_PREFILL=batched → WMMA batched prefill EP (E6b)
    // via forward_prefill_batch_ep; default (sequential) → token-by-token via
    // forward_ep (E6a). Validation: both must yield the SAME gen FNV.
    let batched_prefill = std::env::var("HIPFIRE_EP_PREFILL").as_deref() == Ok("batched");
    eprintln!(
        "EP: tp_size={tp} steps={steps} kv_seq={kv_seq} prefill={} prompt={prompt:?}",
        if batched_prefill { "batched-WMMA" } else { "sequential" },
    );

    // ── tokenize (early — sizes the prefill batch + routed partials) ─────────
    let prompt_tokens: Vec<u32> = tokenizer.encode(&prompt);
    assert!(!prompt_tokens.is_empty(), "empty prompt tokenization");
    let max_batch = prompt_tokens.len().max(2);
    eprintln!("prompt tokenizes to {} tokens (max_batch={max_batch})", prompt_tokens.len());

    // ── bring up N ranks ────────────────────────────────────────────────────
    let mut gpus = Gpus::init_tp(tp, config.n_layers).expect("init_tp");
    let n = gpus.devices.len();
    assert_eq!(n, tp, "init_tp gave {n} devices, expected {tp} (check HIP_VISIBLE_DEVICES)");
    for (r, d) in gpus.devices.iter().enumerate() {
        eprintln!("  rank {r}: device_id={} arch={}", d.device_id, d.arch);
    }

    // ── replicated load + per-rank expert shard ─────────────────────────────
    let shard = ShardConfig::new(tp, /*tp_kv_replicate=*/ true, config.num_experts, ExpertAssign::Stride)
        .expect("ShardConfig");
    let mut weights_per_rank = Vec::with_capacity(n);
    for r in 0..n {
        gpus.devices[r].bind_thread().expect("bind rank");
        let mut hfq = HfqFile::open(model_path).expect("reopen model");
        eprintln!("  [rank {r}] loading replicated weights ...");
        let mut w = qwen35::load_weights(&mut hfq, &config, &mut gpus.devices[r]).expect("load_weights");
        qwen35::shard_all_moe_layers(&mut gpus.devices[r], &mut w, &shard, r, config.num_experts)
            .expect("shard_all_moe_layers");
        weights_per_rank.push(w);
    }
    eprintln!("  all ranks loaded + sharded (assign=stride: rank r owns experts e%{tp}==r)");

    // ── per-rank state + routed partials (+ prefill scratch when batched) ────
    use hipfire_arch_qwen35::qwen35::PrefillBatchScratch;
    let mut kv_per_rank: Vec<KvCache> = Vec::with_capacity(n);
    let mut dn_per_rank: Vec<DeltaNetState> = Vec::with_capacity(n);
    let mut scratch_per_rank: Vec<Qwen35Scratch> = Vec::with_capacity(n);
    let mut pbs_per_rank: Vec<PrefillBatchScratch> = Vec::with_capacity(n);
    // SEPARATE routed partials for decode vs prefill so each all-reduce's count
    // matches its buffer size exactly: decode partial is [dim] (count=dim), the
    // prefill partial is [max_batch·dim] (count=n·dim). Sharing one [max_batch·dim]
    // buffer for both made decode's all-reduce a count<buffer in-place RCCL
    // reduction, which page-faults on multi-rank (tp≥2). Keeping count==buffer
    // matches the validated decode-EP config.
    let mut partials: Vec<GpuTensor> = Vec::with_capacity(n); // decode: [dim]
    let mut prefill_partials: Vec<GpuTensor> = Vec::with_capacity(n); // prefill: [max_batch·dim]
    for r in 0..n {
        gpus.devices[r].bind_thread().expect("bind rank");
        let g = &mut gpus.devices[r];
        kv_per_rank.push(
            KvCache::new_gpu_q8(g, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq)
                .expect("kv"),
        );
        dn_per_rank.push(DeltaNetState::new(g, &config).expect("dn"));
        scratch_per_rank.push(Qwen35Scratch::new(g, &config, 64).expect("scratch"));
        partials.push(g.zeros(&[config.dim], DType::F32).expect("partial"));
        if batched_prefill {
            pbs_per_rank.push(PrefillBatchScratch::new(g, &config, max_batch).expect("pbs"));
            prefill_partials.push(g.zeros(&[max_batch * config.dim], DType::F32).expect("prefill partial"));
        }
    }
    if n > 1 {
        let peer = gpus.enable_peer_all().expect("enable_peer_all");
        eprintln!("  peer_access_enabled={peer}");
    }
    hipfire_runtime::ep::ensure_rank_streams(&mut gpus).expect("ensure_rank_streams");

    // ── EP prefill (batched-WMMA or sequential) — timed (TTFT ≈ prefill wall) ─
    use std::time::Instant;
    eprintln!("\n=== EP forward (prefill {} toks → decode {steps}) ===", prompt_tokens.len());
    let t_prefill = Instant::now();
    if batched_prefill {
        qwen35::forward_prefill_batch_ep(
            &mut gpus, &weights_per_rank, &config, &prompt_tokens, 0,
            &mut kv_per_rank, &mut dn_per_rank, &scratch_per_rank, &pbs_per_rank, &prefill_partials,
        )
        .expect("forward_prefill_batch_ep");
    } else {
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            qwen35::forward_ep(
                &mut gpus, &weights_per_rank, &config, tok, pos,
                &mut kv_per_rank, &dn_per_rank, &scratch_per_rank, &partials,
            )
            .expect("forward_ep prefill");
        }
    }
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
    gpus.devices[0].bind_thread().expect("bind 0");
    let mut logits = gpus.devices[0].download_f32(&scratch_per_rank[0].logits).expect("dl");
    assert!(!logits.iter().any(|v| v.is_nan() || v.is_infinite()), "NaN/Inf in EP prefill logits");
    let mut gen_ep: Vec<u32> = Vec::with_capacity(steps);
    let mut step_ms: Vec<f64> = Vec::with_capacity(steps);
    let mut next = llama::argmax(&logits);
    gen_ep.push(next);
    let base = prompt_tokens.len();
    for step in 1..steps {
        let t = Instant::now();
        qwen35::forward_ep(
            &mut gpus, &weights_per_rank, &config, next, base + step - 1,
            &mut kv_per_rank, &dn_per_rank, &scratch_per_rank, &partials,
        )
        .expect("forward_ep decode");
        gpus.devices[0].bind_thread().expect("bind 0");
        logits = gpus.devices[0].download_f32(&scratch_per_rank[0].logits).expect("dl");
        step_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        assert!(!logits.iter().any(|v| v.is_nan() || v.is_infinite()), "NaN/Inf at EP step {step}");
        next = llama::argmax(&logits);
        gen_ep.push(next);
    }
    let text_ep = tokenizer.decode(&gen_ep);
    eprintln!("EP gen ids : {gen_ep:?}");
    eprintln!("EP gen text: {:?}", text_ep);
    eprintln!("EP gen FNV : 0x{:016x}", fnv1a(&gen_ep));

    // ── perf summary (steady-state decode skips first 3 steps: JIT/cache/DPM warm) ─
    let pf_toks = prompt_tokens.len() as f64;
    let pf_tok_s = pf_toks * 1000.0 / prefill_ms;
    let settled: Vec<f64> = step_ms.iter().skip(3).copied().collect();
    let (dec_tok_s, dec_avg_ms, n_settled) = if settled.is_empty() {
        (f64::NAN, f64::NAN, 0usize)
    } else {
        let avg = settled.iter().sum::<f64>() / settled.len() as f64;
        (1000.0 / avg, avg, settled.len())
    };
    eprintln!(
        "\nPERF tp={tp} prefill={}({} tok): TTFT≈{:.1} ms, prefill {:.1} tok/s | decode {:.1} tok/s ({:.2} ms/tok, steady n={})",
        if batched_prefill { "batched-WMMA" } else { "sequential" },
        prompt_tokens.len(),
        prefill_ms, pf_tok_s, dec_tok_s, dec_avg_ms, n_settled,
    );

    // ── In-process anchor: tp=1 + sequential → production forward_scratch on the
    //    unsharded rank-0 replica. (At tp≥2 rank 0 is sharded → production invalid;
    //    the cross-process tp1-vs-tpN hash diff carries that gate. For batched
    //    prefill the gate is the cross-MODE FNV match vs the sequential run, since
    //    production prefill is itself token-by-token.) ──────
    if n == 1 && !batched_prefill {
        eprintln!("\n=== tp=1 anchor: production forward_scratch (unsharded rank 0) ===");
        let mut kv_ref = KvCache::new_gpu_q8(
            &mut gpus.devices[0], config.n_layers, config.n_kv_heads, config.head_dim, kv_seq,
        )
        .expect("kv ref");
        let mut dn_ref = DeltaNetState::new(&mut gpus.devices[0], &config).expect("dn ref");
        let scratch_ref = Qwen35Scratch::new(&mut gpus.devices[0], &config, 64).expect("scratch ref");
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            qwen35::forward_scratch(
                &mut gpus.devices[0], &weights_per_rank[0], &config, tok, pos,
                &mut kv_ref, &mut dn_ref, &scratch_ref,
            )
            .expect("forward_scratch prefill");
        }
        let ref_logits = gpus.devices[0].download_f32(&scratch_ref.logits).expect("dl ref");
        let ref_max_logit = ref_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut gen_ref: Vec<u32> = Vec::with_capacity(steps);
        let mut nref = llama::argmax(&ref_logits);
        gen_ref.push(nref);
        for step in 1..steps {
            qwen35::forward_scratch(
                &mut gpus.devices[0], &weights_per_rank[0], &config, nref, base + step - 1,
                &mut kv_ref, &mut dn_ref, &scratch_ref,
            )
            .expect("forward_scratch decode");
            let l = gpus.devices[0].download_f32(&scratch_ref.logits).expect("dl ref");
            nref = llama::argmax(&l);
            gen_ref.push(nref);
        }
        eprintln!("REF gen ids : {gen_ref:?}");
        eprintln!("REF gen text: {:?}", tokenizer.decode(&gen_ref));
        eprintln!("REF gen FNV : 0x{:016x}", fnv1a(&gen_ref));
        eprintln!("(ref prefill max logit ~{ref_max_logit:.3})");
        assert_eq!(
            gen_ref, gen_ep,
            "ANCHOR FAIL: EP path argmax stream != production forward_scratch at tp=1.\n\
             EP : {gen_ep:?}\nREF: {gen_ref:?}",
        );
        eprintln!("\n✅ tp=1 ANCHOR PASS: EP argmax stream == production forward_scratch.");
        dn_ref.free_gpu(&mut gpus.devices[0]);
        kv_ref.free_gpu(&mut gpus.devices[0]);
    }

    eprintln!(
        "\n=== ep_decode_parity DONE (tp={tp}) ===\n\
         To validate sharding: run this at tp=1 and tp={} on the same prompt and\n\
         confirm the EP gen FNV hashes are identical.",
        if tp == 1 { 2 } else { tp },
    );
}
