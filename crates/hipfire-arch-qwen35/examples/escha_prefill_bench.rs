// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Escha-W2 prefill/decode timing harness — the measurement behind the
//! Phase-2 prefill work.
//!
//! `escha_model_smoke` prefills 8 tokens because it is a *gate*: it exists to
//! prove the escha executor ran for every (layer, token) and the logits are
//! finite. Eight tokens is far too short to say anything about prefill THROUGHPUT
//! — the load and the first-launch JIT dominate. This harness prefills a
//! realistic prompt (512 or 2048), reports tok/s, and reports the
//! H128 launch count so the route taken is never in doubt:
//!
//! * `4 * n_layers` launches **per token** => the per-token `forward_scratch`
//!   fallback (prefill doing decode's work once per token).
//! * `4 * n_layers` launches **per chunk** => a genuinely batched escha
//!   prefill body.
//!
//! Prompt token ids are arbitrary but FIXED (a deterministic LCG), because the
//! point is a stable timing comparison, not semantics. Routing depends on the
//! activations, so a fixed prompt also keeps the expert-selection work
//! comparable between runs.
//!
//! COST: loads the whole model, ~37.6 GB resident. See `escha_model_smoke`.
//!
//! Run:
//!   cargo run --release -p hipfire-arch-qwen35 \
//!     --example escha_prefill_bench -- /data/hipfire-models/escha-35b.hfq 512 [decode_tokens]
use hipfire_arch_qwen35::qwen35;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::loader_api::{CaskConfig, LoadCtx, ModelSource, SpecLoadCfg};
use rdna_compute::Gpu;
use std::path::Path;

/// Deterministic filler ids. A fixed sequence, not a constant one: a constant
/// token would route every position to the same experts and understate the
/// routed work.
fn prompt_of(n: usize, vocab: usize) -> Vec<u32> {
    let mut s: u64 = 0x2545_F491_4F6C_DD1D;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % vocab as u64) as u32
        })
        .collect()
}

fn main() -> Result<(), String> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/data/hipfire-models/escha-35b.hfq".to_string());
    // ONE length per process, on purpose. Prefill mutates the KV cache and the
    // DeltaNet recurrent state, and a second prefill in the same process would
    // either continue that context (so its attention cost is not comparable) or
    // need a state reset whose completeness is one more thing to be wrong
    // about. A fresh process is the cheap, unambiguous control.
    let n: usize = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "512".to_string())
        .trim()
        .parse()
        .expect("prefill length");
    let decode_tokens: usize = std::env::args()
        .nth(3)
        .map(|s| s.parse().expect("decode count"))
        .unwrap_or(6);

    let hfq = HfqFile::open(Path::new(&path)).map_err(|e| format!("open: {e:?}"))?;
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    let cask = CaskConfig::default();
    let src = ModelSource::Hfq(hfq);
    let mut ctx = LoadCtx {
        path: &path,
        // Room for the longest prefill plus the decode tail.
        max_seq: n + decode_tokens + 64,
        deepseek4_compute_placement: Default::default(),
        deepseek4_experts_per_token: None,
        draft_path: None,
        kv_mode_override: None,
        kv_backend: hipfire_runtime::kv_backend::KvBackend::Contiguous,
        kv_adaptive_override: None,
        state_quant_override: None,
        cask: &cask,
        pp: 1,
        spec: SpecLoadCfg::default(),
        gpu: &mut gpu,
        gemma4_drafter_path: None,
        gemma4_draft_len: 3,
    };
    let t0 = std::time::Instant::now();
    let mut b = hipfire_arch_qwen35::load_qwen35_bundle(src, &mut ctx)?;
    eprintln!("loaded in {:?}", t0.elapsed());

    let per_token_budget =
        hipfire_dispatch::pipeline::escha::escha_launches_per_token(b.config.n_layers);
    eprintln!(
        "n_layers={} vocab={} per-token H128 budget={per_token_budget}",
        b.config.n_layers, b.config.vocab_size
    );

    {
        let prompt = prompt_of(n, b.config.vocab_size);
        let before = rdna_compute::escha_h128_launches();
        let t = std::time::Instant::now();
        qwen35::forward_prefill_batch(
            ctx.gpu,
            &b.weights,
            &b.config,
            &prompt,
            0,
            &mut b.kv_cache,
            &mut b.dn_state,
            &b.scratch,
            None,
            None,
            None,
            None,
        )
        .map_err(|e| format!("prefill {n}: {e:?}"))?;
        ctx.gpu
            .hip
            .device_synchronize()
            .map_err(|e| format!("sync: {e:?}"))?;
        let wall = t.elapsed();
        let launches = rdna_compute::escha_h128_launches() - before;
        let logits = ctx
            .gpu
            .download_f32(&b.scratch.logits)
            .map_err(|e| format!("download logits: {e:?}"))?;
        let bad = logits
            .iter()
            .take(b.config.vocab_size)
            .filter(|v| !v.is_finite())
            .count();
        let (argmax, best) = logits.iter().take(b.config.vocab_size).enumerate().fold(
            (0usize, f32::NEG_INFINITY),
            |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
        );
        eprintln!(
            "PREFILL n={n}: {:.1} ms, {:.1} tok/s, {:.3} ms/token | H128 launches={launches} \
             ({:.1} per token) | non-finite={bad} argmax={argmax} ({best:.4})",
            wall.as_secs_f64() * 1e3,
            n as f64 / wall.as_secs_f64(),
            wall.as_secs_f64() * 1e3 / n as f64,
            launches as f64 / n as f64,
        );
        if bad != 0 {
            return Err(format!("non-finite logits after an {n}-token prefill"));
        }
        // Full final-token logits, for byte-level A/B between two runs of the
        // SAME binary (e.g. `HIPFIRE_ESCHA_INDEXED=0` vs not, to check the two
        // routed routes are bit-identical). An argmax and a 4-decimal top
        // logit cannot establish bit-identity; this can. Nothing else in the
        // harness depends on it.
        if let Ok(dump) = std::env::var("HIPFIRE_BENCH_LOGITS_OUT") {
            let mut bytes = Vec::with_capacity(b.config.vocab_size * 4);
            for v in logits.iter().take(b.config.vocab_size) {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(&dump, &bytes).map_err(|e| format!("write {dump}: {e}"))?;
            eprintln!("wrote {} logits to {dump}", b.config.vocab_size);
        }

        // Decode continuation, from the state this prefill just built. The
        // first token is discarded: it pays any first-shape JIT.
        let mut times = Vec::new();
        for i in 0..decode_tokens {
            let tok = prompt[i % prompt.len()];
            let t = std::time::Instant::now();
            let logits = qwen35::forward(
                ctx.gpu,
                &b.weights,
                &b.config,
                tok,
                n + i,
                &mut b.kv_cache,
                &mut b.dn_state,
            )
            .map_err(|e| format!("decode: {e:?}"))?;
            times.push(t.elapsed().as_secs_f64());
            if logits.iter().any(|v| !v.is_finite()) {
                return Err(format!("non-finite decode logits at pos {}", n + i));
            }
        }
        let warm = &times[1..];
        let mean = warm.iter().sum::<f64>() / warm.len() as f64;
        let lo = warm.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = warm.iter().cloned().fold(0.0f64, f64::max);
        eprintln!(
            "DECODE after n={n}: {:.2} ms/token mean ({:.2}-{:.2}), {:.1} tok/s, n={}",
            mean * 1e3,
            lo * 1e3,
            hi * 1e3,
            1.0 / mean,
            warm.len()
        );
    }
    eprintln!("escha_prefill_bench: done");
    Ok(())
}
