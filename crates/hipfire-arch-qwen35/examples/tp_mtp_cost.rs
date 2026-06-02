//! tp_mtp_cost — measure the TENSOR-PARALLEL trunk cost of one MTP decode cycle
//! (DN snapshot → batched verify[N] → DN rollback → batched replay[advance]) at
//! TP=1 (the single-GPU baseline) vs TP=N, to decide whether MTP-over-TP decode
//! can beat single-GPU MTP (measured 68.5 tok/s) / PP+MTP (64.9).
//!
//! The MTP DRAFT (small rank-0 head) + accept bookkeeping are TENSOR-PARALLEL
//! INVARIANT (they run on devices[0] identically regardless of TP), so this
//! bench measures ONLY the part that TP changes — the trunk verify + replay
//! forwards — and backs the draft+accept remainder out of the measured
//! single-GPU MTP cycle to give an implied end-to-end TP+MTP tok/s.
//!
//! forward_prefill_chunk_tp / DeltaNetSnapshot are the SAME functions the full
//! TP+MTP path would call; tp=1 routes through the identical kernels (Full
//! phase, no all-reduce), so the TP=1↔TP=N delta is the pure TP overhead.
//!
//! Run: HIP_VISIBLE_DEVICES=0,1,2,3 cargo run --release -p hipfire-arch-qwen35 \
//!   --features deltanet --example tp_mtp_cost -- <model.mq4> [verify_width=4] [advance=3] [cycles=40]

use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch, StateQuant};
use hipfire_arch_qwen35::speculative::DeltaNetSnapshot;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::tp_shard::{ExpertAssign, ShardConfig};
use rdna_compute::{DType, Gpu};
use std::path::Path;

const KV_MAX: usize = 4096;

fn make_kv(gpu: &mut Gpu, config: &qwen35::Qwen35Config) -> KvCache {
    let (nl, nk, hd) = (config.n_layers, config.n_kv_heads, config.head_dim);
    KvCache::new_gpu_q8(gpu, nl, nk, hd, KV_MAX).expect("kv q8")
}

/// Median per-cycle TRUNK wall (ms): snapshot → verify[vw] → rollback → replay[adv].
fn run_cost(path: &str, prompt: &[u32], tp: usize, vw: usize, adv: usize, cycles: usize) -> f64 {
    let mut hfq = HfqFile::open(Path::new(path)).expect("open hfq");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    assert_eq!(config.num_experts, 0, "dense models only (MoE hits unreachable in TP forward)");
    assert_eq!(config.n_kv_heads % tp, 0, "n_kv_heads {} not divisible by tp {}", config.n_kv_heads, tp);
    let dim = config.dim;
    // pbs must hold the largest single forward: the one-time prompt prefill
    // (prompt.len()) as well as the per-cycle verify[vw]/replay[adv].
    let pb_batch = prompt.len().max(vw).max(adv).max(16);

    let shard = ShardConfig::new(tp, false, config.num_experts, ExpertAssign::Stride).unwrap();
    shard.validate(config.n_heads, config.n_kv_heads).unwrap();
    let mut gpus = Gpus::init_tp(tp, config.n_layers).expect("init_tp");
    for dev in gpus.devices.iter_mut() {
        dev.bind_thread().expect("bind");
        let st = dev.hip.stream_create().expect("stream");
        dev.active_stream = Some(st);
    }
    let configs: Vec<qwen35::Qwen35Config> = (0..tp)
        .map(|_| if tp == 1 { config.clone() } else { qwen35::local_attn_config(&config, &shard) })
        .collect();

    let mut weights = Vec::with_capacity(tp);
    let mut scratches = Vec::with_capacity(tp);
    let mut kvs = Vec::with_capacity(tp);
    let mut dns = Vec::with_capacity(tp);
    let mut pbs_vec = Vec::with_capacity(tp);
    let mut partials = Vec::with_capacity(tp);
    let mut snaps = Vec::with_capacity(tp);
    for r in 0..tp {
        gpus.devices[r].bind_thread().unwrap();
        weights.push(if tp == 1 {
            qwen35::load_weights(&mut hfq, &config, &mut gpus.devices[r]).unwrap()
        } else {
            qwen35::load_weights_tp(&mut hfq, &config, &mut gpus.devices[r], &shard, r).unwrap()
        });
        scratches.push(Qwen35Scratch::new(&mut gpus.devices[r], &configs[r], 128).unwrap());
        kvs.push(make_kv(&mut gpus.devices[r], &configs[r]));
        dns.push(DeltaNetState::new_with_quant(&mut gpus.devices[r], &configs[r], StateQuant::Q8).unwrap());
        pbs_vec.push(qwen35::PrefillBatchScratch::new(&mut gpus.devices[r], &configs[r], pb_batch).unwrap());
        partials.push(gpus.devices[r].zeros(&[pb_batch * dim], DType::F32).unwrap());
        snaps.push(DeltaNetSnapshot::new_for(&mut gpus.devices[r], &dns[r]).unwrap());
    }

    // Prefill the prompt → trunk KV/DN advanced to ctx0 = prompt.len().
    qwen35::forward_prefill_chunk_tp(&mut gpus, &shard, &weights, &configs, prompt, 0,
        &mut kvs, &mut dns, &pbs_vec, &partials, &scratches, None).expect("prefill");
    let mut ctx = prompt.len();
    let last = prompt[prompt.len() - 1];
    let vtok = vec![last; vw];
    let rtok = vec![last; adv];

    macro_rules! one_cycle {
        () => {{
            for r in 0..tp {
                gpus.devices[r].bind_thread().unwrap();
                snaps[r].save_from(&dns[r], &mut gpus.devices[r]).unwrap();
            }
            qwen35::forward_prefill_chunk_tp(&mut gpus, &shard, &weights, &configs, &vtok, ctx,
                &mut kvs, &mut dns, &pbs_vec, &partials, &scratches, None).unwrap();
            for r in 0..tp {
                gpus.devices[r].bind_thread().unwrap();
                snaps[r].restore_to(&mut dns[r], &mut gpus.devices[r]).unwrap();
            }
            qwen35::forward_prefill_chunk_tp(&mut gpus, &shard, &weights, &configs, &rtok, ctx,
                &mut kvs, &mut dns, &pbs_vec, &partials, &scratches, None).unwrap();
            ctx += adv;
        }};
    }

    // Warm (5 cycles).
    for _ in 0..5 { one_cycle!(); }
    gpus.devices[0].bind_thread().unwrap();
    gpus.devices[0].hip.device_synchronize().unwrap();

    // Measure per-cycle wall (median).
    let mut samples = Vec::with_capacity(cycles);
    for _ in 0..cycles {
        if ctx + vw + 8 >= KV_MAX { break; }
        let t = std::time::Instant::now();
        one_cycle!();
        gpus.devices[0].bind_thread().unwrap();
        gpus.devices[0].hip.device_synchronize().unwrap();
        samples.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = samples[samples.len() / 2];

    // Teardown (free per-rank allocs + drain pool so the next run_cost loads clean).
    for r in (0..tp).rev() {
        gpus.devices[r].bind_thread().unwrap();
        if let Some(st) = gpus.devices[r].active_stream.take() {
            let _ = gpus.devices[r].hip.stream_destroy(st);
        }
        if let Some(p) = partials.pop() { let _ = gpus.devices[r].free_tensor(p); }
        if let Some(pb) = pbs_vec.pop() { pb.free_gpu(&mut gpus.devices[r]); }
        if let Some(s) = scratches.pop() { s.free_gpu(&mut gpus.devices[r]); }
        if let Some(d) = dns.pop() { d.free_gpu(&mut gpus.devices[r]); }
        if let Some(k) = kvs.pop() { k.free_gpu(&mut gpus.devices[r]); }
        if let Some(w) = weights.pop() { w.free_gpu(&mut gpus.devices[r]); }
        gpus.devices[r].drain_pool();
    }
    med
}

fn main() {
    std::env::set_var("HIPFIRE_DETERMINISTIC", "1");
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: tp_mtp_cost <model.mq4> [verify_width=4] [advance=3] [cycles=40]");
    let vw: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let adv: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    let cycles: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(40);

    let hfq = HfqFile::open(Path::new(path)).expect("open");
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    drop(hfq);
    let mut prompt = tok.encode("<|im_start|>");
    prompt.extend(tok.encode("user\n"));
    prompt.extend(tok.encode("Write a Python function to compute the sum of an array using a loop."));
    prompt.extend(tok.encode("<|im_end|>\n<|im_start|>assistant\n"));
    eprintln!("prompt tokens: {}", prompt.len());

    eprintln!("=== TP+MTP trunk cost: per-cycle wall = verify[N={vw}] + DN rollback + replay[{adv}] ===");
    let tp1 = run_cost(path, &prompt, 1, vw, adv, cycles);
    let tp4 = run_cost(path, &prompt, 4, vw, adv, cycles);
    eprintln!("TP=1 (this TP forward, 1 GPU) trunk/cycle: {tp1:.2} ms");
    eprintln!("TP=4 (sharded)               trunk/cycle: {tp4:.2} ms");
    eprintln!("TP=4 / TP=1 trunk ratio: {:.2}x  ({})", tp4 / tp1,
        if tp4 < tp1 { "TP FASTER" } else { "TP SLOWER" });

    // Implied end-to-end TP+MTP tok/s. Per MTP cycle: draft (MTP head on
    // devices[0], TP-INVARIANT) + the measured trunk (verify+rollback+replay).
    // The MTP head is ~1 transformer layer of the trunk's 64, run K=3 times +
    // a compressed-vocab lm_head, so the draft is a small TP-invariant constant;
    // sweep DRAFT_MS to bound it. Apples-to-apples (both use THIS TP forward):
    // tok/s = tau / (draft + trunk).
    let tau = 3.16_f64;
    for draft_ms in [2.0_f64, 5.0] {
        let i1 = tau / ((tp1 + draft_ms) / 1000.0);
        let i4 = tau / ((tp4 + draft_ms) / 1000.0);
        eprintln!(
            "  draft={draft_ms:.0}ms → implied TP=1+MTP {i1:.1} tok/s, TP=4+MTP {i4:.1} tok/s",
        );
    }
    eprintln!("REFERENCE (production fwd): single-GPU MTP 68.5 tok/s, PP=4+MTP 64.9 tok/s.");
    eprintln!("NOTE: this bench's TP forward (forward_prefill_chunk_tp) is the UNtuned TP path;");
    eprintln!("      it is ~1.8x slower at tp=1 than the production forward_prefill_batch, so the");
    eprintln!("      tp=1 row here is NOT the 68.5 baseline — the apples-to-apples signal is TP=4/TP=1.");
}
