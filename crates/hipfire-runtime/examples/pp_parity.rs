//! Stage 6 smoke: PP=1 vs PP=2 token-stream parity on a real model.
//! Loads qwen3.5:0.8b twice — once via the single-GPU forward_scratch
//! and once via the multi-GPU forward_scratch_multi — and asserts the
//! greedy token sequence matches for ≥ 100 decoded tokens (temp=0,
//! same prompt token).
//!
//! Run: HIP_VISIBLE_DEVICES=0,1 cargo run -p hipfire-runtime \
//!         --release --features deltanet --example pp_parity -- \
//!         ~/.hipfire/models/qwen3.5-0.8b.mq4

use hipfire_arch_qwen35::qwen35::{
    self, DeltaNetState, Qwen35Scratch, Qwen35ScratchSet, StateQuant,
};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::multi_gpu::Gpus;
use rdna_compute::Gpu;
use std::path::Path;

const N_TOKENS: usize = 100;
const PROMPT_TOKEN: u32 = 1;

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

fn run_single_gpu(path: &str) -> Vec<u32> {
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

    let mut tokens = Vec::with_capacity(N_TOKENS);
    let mut tok = PROMPT_TOKEN;
    for pos in 0..N_TOKENS {
        qwen35::forward_scratch(&mut gpu, &weights, &config, tok, pos, &mut kv, &mut dn, &scratch)
            .expect("forward_scratch");
        let logits = gpu.download_f32(&scratch.logits).expect("download logits");
        tok = argmax(&logits);
        tokens.push(tok);
    }
    // F4 (review): explicitly free + drain pool before returning, otherwise
    // dev 0 stays VRAM-loaded into the next phase and the multi-GPU
    // `Gpus::init_uniform` preflight_vram trips on the asymmetry. Without
    // this, the parity smoke passes only on models small enough that the
    // leak fits inside the 2 GiB tolerance.
    scratch.free_gpu(&mut gpu);
    dn.free_gpu(&mut gpu);
    kv.free_gpu(&mut gpu);
    weights.free_gpu(&mut gpu);
    gpu.drain_pool();
    tokens
}

fn run_multi_gpu(path: &str) -> Vec<u32> {
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
    let mut tokens = Vec::with_capacity(N_TOKENS);
    let mut tok = PROMPT_TOKEN;
    for pos in 0..N_TOKENS {
        qwen35::forward_scratch_multi(
            &mut gpus, &weights, &config, tok, pos, &mut kv, &mut dn, &scratch_set,
        )
        .expect("forward_scratch_multi");
        let s_last = &scratch_set.per_device[dev_last];
        let logits = gpus.devices[dev_last]
            .download_f32(&s_last.logits)
            .expect("download logits");
        tok = argmax(&logits);
        tokens.push(tok);
    }
    tokens
}

fn main() {
    let path = std::env::args().nth(1).expect("Usage: ... <model.mq4>");

    println!("── PP=1 forward ──────────────────────────────────────────");
    let tokens_pp1 = run_single_gpu(&path);
    println!("PP=1 tokens (first 20): {:?}", &tokens_pp1[..20.min(tokens_pp1.len())]);

    println!("\n── PP=2 forward ──────────────────────────────────────────");
    let tokens_pp2 = run_multi_gpu(&path);
    println!("PP=2 tokens (first 20): {:?}", &tokens_pp2[..20.min(tokens_pp2.len())]);

    println!("\n── parity check ──────────────────────────────────────────");
    if tokens_pp1 == tokens_pp2 {
        println!("ALL {N_TOKENS} tokens identical between PP=1 and PP=2");
        println!("\npp_parity: PASS");
        return;
    }
    let first_diff = tokens_pp1
        .iter()
        .zip(tokens_pp2.iter())
        .position(|(a, b)| a != b);
    let n_match = first_diff.unwrap_or(N_TOKENS);
    println!("matched {n_match}/{N_TOKENS} tokens before first divergence");
    if let Some(i) = first_diff {
        println!(
            "first diff at idx {i}: PP=1={} PP=2={}",
            tokens_pp1[i], tokens_pp2[i]
        );
        println!("PP=1 around: {:?}", &tokens_pp1[i.saturating_sub(2)..(i + 5).min(N_TOKENS)]);
        println!("PP=2 around: {:?}", &tokens_pp2[i.saturating_sub(2)..(i + 5).min(N_TOKENS)]);
    }
    eprintln!("\npp_parity: FAIL");
    std::process::exit(1);
}
