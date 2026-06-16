// SPDX-License-Identifier: Apache-2.0
// hipfire — golden-output runner for tiny gating fixtures.
//
// Loads a tiny arch-5/6 .hfq (from `hipfire-quantize --emit-fixture`), feeds a
// FIXED raw token-ID sequence (the fixture has no real tokenizer), runs the
// forward, and emits a deterministic golden: the per-position argmax token
// sequence + a hash of the logits. Run twice / across builds and diff — a
// drift in the argmax line is the tripwire (escalate to the 35B golden).
//
// Usage:
//   fixture_golden <model.hfq> [--len 32] [--warmup 2] [--seed 1]

use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::KvCache;
use std::path::Path;

/// splitmix64 — same generator as the fixture emitter, for a reproducible
/// fixed token stream independent of any tokenizer.
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .expect("usage: fixture_golden <model.hfq> [--len N] [--warmup N] [--seed N]");

    let mut len: usize = 32;
    let mut warmup: usize = 2;
    let mut seed: u64 = 1;
    while let Some(flag) = args.next() {
        let val = args.next().expect("flag missing value");
        match flag.as_str() {
            "--len" => len = val.parse().unwrap(),
            "--warmup" => warmup = val.parse().unwrap(),
            "--seed" => seed = val.parse().unwrap(),
            _ => panic!("unknown flag: {flag}"),
        }
    }
    assert!(len > warmup + 1, "len must exceed warmup");

    // Fixed token stream in a small range valid for any fixture vocab.
    let mut st = seed ^ 0x5DEE_CE66_D8A1_0001;
    let tokens: Vec<u32> = (0..len).map(|_| (splitmix(&mut st) % 100) as u32).collect();

    let mut hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("config");

    let mut gpu = rdna_compute::Gpu::init().expect("GPU init");
    eprintln!("GPU: {}", gpu.arch);
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load_weights");

    let kv_max = len + 16;
    let mut kv_cache = KvCache::new_gpu_q8(
        &mut gpu,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        kv_max,
    )
    .unwrap();
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 64).unwrap();

    // FNV-1a over the logit bits — sensitive (byte-exact) golden; the argmax
    // line is the robust golden. Print both.
    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    let mut argmax_seq: Vec<u32> = Vec::new();

    for (pos, &tok) in tokens.iter().enumerate() {
        qwen35::forward_scratch(
            &mut gpu,
            &weights,
            &config,
            tok,
            pos,
            &mut kv_cache,
            &mut dn_state,
            &scratch,
        )
        .expect("forward");

        if pos < warmup {
            continue;
        }
        let logits = gpu.download_f32(&scratch.logits).unwrap();
        let mut best = (f32::NEG_INFINITY, 0u32);
        for (i, &v) in logits.iter().enumerate() {
            for b in v.to_bits().to_le_bytes() {
                hash ^= b as u64;
                hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
            }
            if v > best.0 {
                best = (v, i as u32);
            }
        }
        argmax_seq.push(best.1);
    }

    println!("model:     {model_path}");
    println!("tokens:    len={len} warmup={warmup} seed={seed}");
    println!("argmax:    {argmax_seq:?}");
    println!("logit_hash: {hash:#018x}");
}
