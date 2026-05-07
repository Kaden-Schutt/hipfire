//! Extended pp parity: ChatML-wrapped prompt + multi-token prefill + top-K
//! logit dump at the first divergence. Stage 6 pp_parity used a synthetic
//! `tok=1` repeated input, which exercised decode-only steady state and
//! sailed through 100/100. Real daemon flow runs through 15+ ChatML
//! tokens of prefill — accumulating KV history that crosses the band
//! boundary via hipMemcpyPeer. This binary mirrors that flow and prints
//! both top-K's at the first argmax flip so we can tell roundoff from
//! structural bug.
//!
//! Run: HIP_VISIBLE_DEVICES=0,1 cargo run -p hipfire-runtime \
//!         --release --features deltanet --example pp_parity_chatml -- \
//!         ~/.hipfire/models/qwen3.5-0.8b.mq4

use hipfire_arch_qwen35::qwen35::{
    self, DeltaNetState, Qwen35Scratch, Qwen35ScratchSet, StateQuant,
};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;
use std::path::Path;

const N_DECODE: usize = 50;
const TOP_K: usize = 5;
const PROMPT: &str = "Write a one-sentence greeting.";

fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

fn top_k(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
    idx.into_iter().take(k).map(|i| (i as u32, logits[i])).collect()
}

fn build_prompt_tokens(tok: &Tokenizer) -> Vec<u32> {
    let im_start = tok.encode("<|im_start|>");
    let im_end = tok.encode("<|im_end|>");
    let nl = tok.encode("\n");
    let user = tok.encode("user");
    let asst = tok.encode("assistant");
    let q = tok.encode(PROMPT);
    let mut t = Vec::new();
    t.extend_from_slice(&im_start);
    t.extend_from_slice(&user);
    t.extend_from_slice(&nl);
    t.extend_from_slice(&q);
    t.extend_from_slice(&im_end);
    t.extend_from_slice(&nl);
    t.extend_from_slice(&im_start);
    t.extend_from_slice(&asst);
    t.extend_from_slice(&nl);
    t
}

fn run_single_gpu(path: &str, prompt_tokens: &[u32]) -> (Vec<u32>, Vec<Vec<f32>>) {
    let mut hfq = HfqFile::open(Path::new(path)).expect("open hfq");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let mut gpu = Gpu::init().expect("Gpu::init");
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load_weights");
    let mut kv = KvCache::new_gpu_asym3_capped(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, 4096, 4096,
    )
    .expect("kv");
    let mut dn = DeltaNetState::new_with_quant(&mut gpu, &config, StateQuant::Q8).expect("dn");
    let scratch = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 64, 4096).expect("scratch");

    // Per-token prefill (matches daemon pp=2 flow; pp=1 daemon uses
    // forward_prefill_batch but per-token gives a fair pp comparison
    // because pp=2 has no batched analogue yet — same kernel path).
    let mut all_logits: Vec<Vec<f32>> = Vec::new();
    for (i, &tok) in prompt_tokens.iter().enumerate() {
        qwen35::forward_scratch(&mut gpu, &weights, &config, tok, i, &mut kv, &mut dn, &scratch)
            .expect("forward_scratch prefill");
    }
    let mut tokens = Vec::with_capacity(N_DECODE);
    let mut tok = {
        let logits = gpu.download_f32(&scratch.logits).expect("download logits");
        let next = argmax(&logits);
        all_logits.push(logits);
        next
    };
    tokens.push(tok);
    for step in 1..N_DECODE {
        let pos = prompt_tokens.len() + step - 1;
        qwen35::forward_scratch(&mut gpu, &weights, &config, tok, pos, &mut kv, &mut dn, &scratch)
            .expect("forward_scratch decode");
        let logits = gpu.download_f32(&scratch.logits).expect("download logits");
        tok = argmax(&logits);
        tokens.push(tok);
        all_logits.push(logits);
    }
    scratch.free_gpu(&mut gpu);
    dn.free_gpu(&mut gpu);
    kv.free_gpu(&mut gpu);
    weights.free_gpu(&mut gpu);
    gpu.drain_pool();
    (tokens, all_logits)
}

fn run_multi_gpu(path: &str, prompt_tokens: &[u32]) -> (Vec<u32>, Vec<Vec<f32>>) {
    let hfq = HfqFile::open(Path::new(path)).expect("open hfq");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let mut gpus = Gpus::init_uniform(2, config.n_layers).expect("init_uniform");
    let weights =
        qwen35::load_weights_multi(&hfq, &config, &mut gpus).expect("load_weights_multi");
    let scratch_set = Qwen35ScratchSet::new_with_kv_max_multi(&mut gpus, &config, 64, 4096)
        .expect("scratch_set");
    let mut kv = KvCache::new_gpu_asym3_capped_multi(
        &mut gpus, config.n_layers, config.n_kv_heads, config.head_dim, 4096, 4096,
    )
    .expect("kv multi");
    let (mut dn, _la_to_device) =
        DeltaNetState::new_with_quant_multi(&mut gpus, &config, StateQuant::Q8)
            .expect("dn multi");
    let _ = gpus.enable_peer_all().expect("enable_peer_all");

    let dev_last = gpus.output_device;
    let mut all_logits: Vec<Vec<f32>> = Vec::new();
    for (i, &tok) in prompt_tokens.iter().enumerate() {
        qwen35::forward_scratch_multi(
            &mut gpus, &weights, &config, tok, i, &mut kv, &mut dn, &scratch_set,
        )
        .expect("forward_scratch_multi prefill");
    }
    let mut tokens = Vec::with_capacity(N_DECODE);
    let mut tok = {
        let s_last = &scratch_set.per_device[dev_last];
        let logits = gpus.devices[dev_last]
            .download_f32(&s_last.logits)
            .expect("download logits");
        let next = argmax(&logits);
        all_logits.push(logits);
        next
    };
    tokens.push(tok);
    for step in 1..N_DECODE {
        let pos = prompt_tokens.len() + step - 1;
        qwen35::forward_scratch_multi(
            &mut gpus, &weights, &config, tok, pos, &mut kv, &mut dn, &scratch_set,
        )
        .expect("forward_scratch_multi decode");
        let s_last = &scratch_set.per_device[dev_last];
        let logits = gpus.devices[dev_last]
            .download_f32(&s_last.logits)
            .expect("download logits");
        tok = argmax(&logits);
        tokens.push(tok);
        all_logits.push(logits);
    }
    (tokens, all_logits)
}

fn main() {
    // Force deterministic WMMA reduction so parity holds regardless of
    // whether the caller (pp-gate.sh) sets the var in the environment.
    // PP FP16 reduction order is non-deterministic without this flag.
    std::env::set_var("HIPFIRE_DETERMINISTIC", "1");
    let path = std::env::args().nth(1).expect("Usage: ... <model.mq4>");
    let hfq = HfqFile::open(Path::new(&path)).expect("open hfq");
    let tokenizer =
        Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    let prompt_tokens = build_prompt_tokens(&tokenizer);
    println!("prompt: {:?}  (len={})", PROMPT, prompt_tokens.len());

    println!("\n── PP=1 (per-token prefill + decode) ──");
    let (toks1, logits1) = run_single_gpu(&path, &prompt_tokens);
    println!("PP=1 first 20: {:?}", &toks1[..20.min(toks1.len())]);

    println!("\n── PP=2 (per-token prefill + decode) ──");
    let (toks2, logits2) = run_multi_gpu(&path, &prompt_tokens);
    println!("PP=2 first 20: {:?}", &toks2[..20.min(toks2.len())]);

    println!("\n── parity ──");
    let first_diff = toks1.iter().zip(toks2.iter()).position(|(a, b)| a != b);
    match first_diff {
        None => {
            println!("ALL {N_DECODE} tokens identical.");
            // Still report max abs logit delta and max top-1 margin.
            let mut max_abs = 0f32;
            let mut sum_abs = 0f32;
            for (a, b) in logits1.iter().zip(logits2.iter()) {
                for (x, y) in a.iter().zip(b.iter()) {
                    let d = (x - y).abs();
                    if d > max_abs { max_abs = d; }
                    sum_abs += d;
                }
            }
            let n = logits1.len() * logits1[0].len();
            println!("max |Δlogit| = {:.3e}, mean |Δlogit| = {:.3e}", max_abs, sum_abs / n as f32);
        }
        Some(i) => {
            println!("first diff at decode step {i}:");
            println!("  PP=1 picked token {} ({:?})", toks1[i], tokenizer.decode_bytes(&[toks1[i]]));
            println!("  PP=2 picked token {} ({:?})", toks2[i], tokenizer.decode_bytes(&[toks2[i]]));
            let l1 = &logits1[i];
            let l2 = &logits2[i];
            let t1 = top_k(l1, TOP_K);
            let t2 = top_k(l2, TOP_K);
            println!("  PP=1 top-{TOP_K}:");
            for (id, v) in &t1 {
                println!("    {:>6}  {:>12.6}  {:?}", id, v, tokenizer.decode_bytes(&[*id]));
            }
            println!("  PP=2 top-{TOP_K}:");
            for (id, v) in &t2 {
                println!("    {:>6}  {:>12.6}  {:?}", id, v, tokenizer.decode_bytes(&[*id]));
            }
            let delta_top1 = (l1[toks1[i] as usize] - l2[toks1[i] as usize]).abs();
            let delta_top2 = (l1[toks2[i] as usize] - l2[toks2[i] as usize]).abs();
            println!("  |Δlogit| at PP=1 winner ({}): {:.3e}", toks1[i], delta_top1);
            println!("  |Δlogit| at PP=2 winner ({}): {:.3e}", toks2[i], delta_top2);
            // Margin = winner − runner-up on each side. Tiny margin + tiny
            // |Δlogit| means we crossed an argmax razor's edge — diagnostic
            // signature of pure floating-point accumulation, not a bug.
            let margin1 = t1[0].1 - t1[1].1;
            let margin2 = t2[0].1 - t2[1].1;
            println!("  PP=1 top-1 margin: {:.3e}", margin1);
            println!("  PP=2 top-1 margin: {:.3e}", margin2);
        }
    }

    let common = toks1.iter().zip(toks2.iter()).take_while(|(a, b)| a == b).count();
    println!("\nmatched {common}/{N_DECODE} decode tokens before divergence");
}
