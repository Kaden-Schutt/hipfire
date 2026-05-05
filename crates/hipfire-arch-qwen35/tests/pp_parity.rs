//! Stage 9 multi-GPU parity test (env-gated).
//!
//! Runs only when `HIPFIRE_HAVE_2_GPU=1` is set — single-GPU dev boxes
//! and CI without dual GPUs silently skip. Mirrors the
//! `pp_parity_chatml` example as a `cargo test`-driven gate so
//! `cargo test --workspace` exercises pp parity when the hardware
//! supports it.
//!
//! Asserts: per-token `forward_scratch_multi` ≡ `forward_scratch` bit-
//! exact across 50 decode tokens after a 15-token ChatML prefill on
//! qwen3.5-0.8b.mq4 + asym3 KV. This is the floor — if this regresses,
//! pp=2 is broken.
//!
//! Model location: `$HOME/.hipfire/models/qwen3.5-0.8b.mq4` (override
//! via `HIPFIRE_PP_PARITY_MODEL`).

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

fn run_pp1(path: &str, prompt: &[u32]) -> Vec<u32> {
    let hfq = HfqFile::open(Path::new(path)).expect("open hfq");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let mut gpu = Gpu::init().expect("Gpu::init");
    let weights = qwen35::load_weights(&hfq, &config, &mut gpu).expect("load_weights");
    let mut kv = KvCache::new_gpu_asym3_capped(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, 4096, 4096,
    ).expect("kv");
    let mut dn = DeltaNetState::new_with_quant(&mut gpu, &config, StateQuant::Q8).expect("dn");
    let scratch = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 64, 4096).expect("scratch");

    for (i, &tok) in prompt.iter().enumerate() {
        qwen35::forward_scratch(&mut gpu, &weights, &config, tok, i, &mut kv, &mut dn, &scratch)
            .expect("forward_scratch prefill");
    }
    let mut tokens = Vec::with_capacity(N_DECODE);
    let mut tok = {
        let logits = gpu.download_f32(&scratch.logits).expect("download logits");
        argmax(&logits)
    };
    tokens.push(tok);
    for step in 1..N_DECODE {
        let pos = prompt.len() + step - 1;
        qwen35::forward_scratch(&mut gpu, &weights, &config, tok, pos, &mut kv, &mut dn, &scratch)
            .expect("forward_scratch decode");
        let logits = gpu.download_f32(&scratch.logits).expect("download logits");
        tok = argmax(&logits);
        tokens.push(tok);
    }
    scratch.free_gpu(&mut gpu);
    dn.free_gpu(&mut gpu);
    kv.free_gpu(&mut gpu);
    weights.free_gpu(&mut gpu);
    gpu.drain_pool();
    tokens
}

fn run_pp2(path: &str, prompt: &[u32]) -> Vec<u32> {
    let hfq = HfqFile::open(Path::new(path)).expect("open hfq");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let mut gpus = Gpus::init_uniform(2, config.n_layers).expect("init_uniform");
    let weights =
        qwen35::load_weights_multi(&hfq, &config, &mut gpus).expect("load_weights_multi");
    let scratch_set = Qwen35ScratchSet::new_with_kv_max_multi(&mut gpus, &config, 64, 4096)
        .expect("scratch_set");
    let mut kv = KvCache::new_gpu_asym3_capped_multi(
        &mut gpus, config.n_layers, config.n_kv_heads, config.head_dim, 4096, 4096,
    ).expect("kv multi");
    let (mut dn, _la_to_device) =
        DeltaNetState::new_with_quant_multi(&mut gpus, &config, StateQuant::Q8)
            .expect("dn multi");
    let _ = gpus.enable_peer_all().expect("enable_peer_all");

    let dev_last = gpus.output_device;
    for (i, &tok) in prompt.iter().enumerate() {
        qwen35::forward_scratch_multi(
            &mut gpus, &weights, &config, tok, i, &mut kv, &mut dn, &scratch_set,
        ).expect("forward_scratch_multi prefill");
    }
    let mut tokens = Vec::with_capacity(N_DECODE);
    let mut tok = {
        let s_last = &scratch_set.per_device[dev_last];
        let logits = gpus.devices[dev_last]
            .download_f32(&s_last.logits)
            .expect("download logits");
        argmax(&logits)
    };
    tokens.push(tok);
    for step in 1..N_DECODE {
        let pos = prompt.len() + step - 1;
        qwen35::forward_scratch_multi(
            &mut gpus, &weights, &config, tok, pos, &mut kv, &mut dn, &scratch_set,
        ).expect("forward_scratch_multi decode");
        let s_last = &scratch_set.per_device[dev_last];
        let logits = gpus.devices[dev_last]
            .download_f32(&s_last.logits)
            .expect("download logits");
        tok = argmax(&logits);
        tokens.push(tok);
    }
    tokens
}

// `#[ignore]`d by default. The cargo-test environment doesn't pre-compile
// the full hot-path kernel set the daemon ships with — JIT via hipcc fails
// on NixOS (and any host without ROCm SDK on PATH). Run explicitly:
//
//   HIP_VISIBLE_DEVICES=0,1 HIPFIRE_HAVE_2_GPU=1 \
//       cargo test -p hipfire-arch-qwen35 --release --features deltanet \
//                  --test pp_parity -- --ignored
//
// The canonical hands-free regression is `scripts/pp-gate.sh`, which
// drives the daemon binary (carries pre-compiled kernels) end-to-end
// and is wired into the pre-commit hook.
#[test]
#[ignore]
fn pp_parity_chatml_50_decode() {
    if std::env::var("HIPFIRE_HAVE_2_GPU").ok().as_deref() != Some("1") {
        eprintln!("skipping: HIPFIRE_HAVE_2_GPU not set");
        return;
    }
    let model = std::env::var("HIPFIRE_PP_PARITY_MODEL")
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").expect("HOME");
            format!("{home}/.hipfire/models/qwen3.5-0.8b.mq4")
        });
    if !Path::new(&model).is_file() {
        eprintln!("skipping: model {model} not found");
        return;
    }

    let hfq = HfqFile::open(Path::new(&model)).expect("open hfq");
    let tokenizer =
        Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    let prompt = build_prompt_tokens(&tokenizer);
    drop(hfq);

    let toks_pp1 = run_pp1(&model, &prompt);
    let toks_pp2 = run_pp2(&model, &prompt);

    assert_eq!(
        toks_pp1.len(),
        N_DECODE,
        "pp=1 generated {} tokens, expected {N_DECODE}",
        toks_pp1.len(),
    );
    assert_eq!(
        toks_pp2.len(),
        N_DECODE,
        "pp=2 generated {} tokens, expected {N_DECODE}",
        toks_pp2.len(),
    );

    let common = toks_pp1
        .iter()
        .zip(toks_pp2.iter())
        .take_while(|(a, b)| a == b)
        .count();
    if common < N_DECODE {
        let i = common;
        panic!(
            "pp parity broke at decode step {i}: pp=1={}, pp=2={}\n\
             pp=1 head: {:?}\npp=2 head: {:?}",
            toks_pp1[i], toks_pp2[i],
            &toks_pp1[..i.min(toks_pp1.len())],
            &toks_pp2[..i.min(toks_pp2.len())],
        );
    }
}
