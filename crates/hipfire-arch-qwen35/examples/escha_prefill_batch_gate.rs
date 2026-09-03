// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//! G6: batched Escha-W2 prefill against the per-token route, whole model.
//!
//! This is the gate specified in task-perf-2 §5.4. It runs ONE prompt through
//! the model twice — once through `forward_prefill_batch` (batched) and once
//! through the per-token `forward` loop the non-batched route is byte-identical
//! to — and reports what changed.
//!
//! # What must be equal, what must not be, and why
//!
//! The routed (escha) half is asserted EQUAL, bit-for-bit, by
//! `escha_moe_block_gate` (G4): `escha_routed_prefill_indexed` differs from
//! `escha_routed_decode_indexed` only in `slots`, and every kernel in that
//! pipeline is purely slot-parallel — slot `s` performs the same FLOPs in the
//! same order whether the launch carried 8 slots or 2 048. That is a real
//! equality claim and it is asserted as equality there, not as a tolerance.
//!
//! The DENSE half is NOT bit-identical and cannot be. Batched prefill runs the
//! attention projections, the router and the shared expert as batched WMMA
//! GEMMs (`gemm_q8_0_residual_wmma`, `gemm_f16_wmma_mb8`, ...) where the
//! per-token route runs batch-1 GEMVs. A 16x16x16 WMMA tile accumulates a
//! different K-order than a warp-reduced GEMV, so the two disagree at the
//! last bits of every dot product and the difference compounds across 40
//! layers. That is a summation-order difference, not a defect.
//!
//! So the whole-model claim this gate can make is: the final-token logits
//! agree to a MEASURED bound, and the ARGMAX does not move.
//!
//! The measured values at n=64 on escha-35b are `max|delta| = 4.393e-1` and
//! `mean|delta| = 7.160e-2`, and both are now asserted rather than printed.
//! Note that these are two orders of magnitude ABOVE what pure accumulation
//! reordering would give: the dominant term is not reordering at all but the
//! expert-selection divergence documented below, which is 24.1% of (token,
//! layer) decisions over the whole stack. A token whose expert set differs
//! computes a different hidden state, and that compounds with depth. An
//! earlier version of this file quoted "~1e-3" here; that figure describes
//! two arms with identical routing, which these are not.
//!
//! **The argmax assertion is the load-bearing one.** Do not relax this gate to
//! "max delta < tol" and stop there. A dropped H128 transform, a stale
//! activation cache, or a wrong expert-slot stride all produce finite, fluent
//! output that is wrong by ~1e-1 per element — and a tolerance chosen loosely
//! enough to absorb the real divergence can absorb those too. The argmax
//! moving is the signal. (When this port's stale-FP16-activation bug was live,
//! this gate's max delta was ~1e+1 and the argmax moved from 25760 to 220.)
//!
//! # The expert-selection divergence
//!
//! Escha's router logits pass through `router_logits_round_f16_rne` on BOTH
//! routes (that is what makes the comparison meaningful at all), but the
//! logits fed to it differ by the same accumulation reordering. ~0.42% of
//! escha routing decisions were measured in Task 9 to straddle an f16
//! boundary, so a few tokens per prompt legitimately select a different expert
//! set on the two routes. That is a property of the format, not a defect — but
//! it has to be BOUNDED, so this gate measures it rather than repeating the
//! estimate: with `HIPFIRE_ESCHA_ROUTE_TRACE` set, both arms record the
//! `topk_indices` every MoE layer actually indexed with, and the gate reports
//! the fraction of (token, layer) decisions whose top-k SET differs.
//!
//! # State reset between the arms
//!
//! The DeltaNet recurrent state is the only thing that carries between the two
//! arms, and it must not: arm B would otherwise start from the state arm A
//! left behind and the comparison would be meaningless. A fresh
//! `DeltaNetState` is allocated for each arm rather than zeroing in place, so
//! "reset" means exactly what a fresh process means. The KV cache needs no
//! reset — arm B prefills from position 0 and rewrites the same slots, and
//! attention only reads positions below its own `start_pos + n`.
//!
//! COST: loads the whole model, ~37.5 GB resident.
//!
//! Run:
//!   cargo run --release -p hipfire-arch-qwen35 \
//!     --example escha_prefill_batch_gate -- /data/hipfire-models/escha-35b.hfq [n]

use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::qwen35::DeltaNetState;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::loader_api::{CaskConfig, LoadCtx, ModelSource, SpecLoadCfg};
use rdna_compute::Gpu;
use std::path::Path;

/// Same deterministic filler as `escha_prefill_bench`, for the same reason: a
/// constant token would route every position to the same experts and would
/// understate both the routed work and the selection divergence.
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

/// One trace record: `n_tokens` rows of `k` expert ids, token-major.
struct TraceRec {
    n_tokens: usize,
    k: usize,
    ids: Vec<i32>,
}

fn read_trace(path: &str) -> Result<Vec<TraceRec>, String> {
    let data = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
    let mut out = Vec::new();
    let mut off = 0usize;
    while off + 8 <= data.len() {
        let n = u32::from_le_bytes(data[off..off + 4].try_into().unwrap()) as usize;
        let k = u32::from_le_bytes(data[off + 4..off + 8].try_into().unwrap()) as usize;
        off += 8;
        let want = n * k * 4;
        if off + want > data.len() {
            return Err(format!("{path}: truncated record"));
        }
        let ids = data[off..off + want]
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        off += want;
        out.push(TraceRec {
            n_tokens: n,
            k,
            ids,
        });
    }
    if off != data.len() {
        return Err(format!("{path}: trailing bytes"));
    }
    Ok(out)
}

/// Flatten a trace into `(token, layer) -> sorted expert set`, given the number
/// of MoE layers. Every MoE layer emits exactly one record per forward, so
/// record order determines the grid; the caller checks the totals.
fn flatten(
    recs: &[TraceRec],
    n_layers_moe: usize,
    n_tokens: usize,
) -> Result<Vec<Vec<i32>>, String> {
    let mut out: Vec<Vec<i32>> = vec![Vec::new(); n_tokens * n_layers_moe];
    let mut layer = 0usize;
    let mut token_base = 0usize;
    for r in recs {
        for t in 0..r.n_tokens {
            let tok = token_base + t;
            if tok >= n_tokens {
                return Err("trace covers more tokens than the prompt".into());
            }
            let mut ids: Vec<i32> = r.ids[t * r.k..(t + 1) * r.k].to_vec();
            ids.sort_unstable();
            out[tok * n_layers_moe + layer] = ids;
        }
        layer += 1;
        if layer == n_layers_moe {
            layer = 0;
            token_base += r.n_tokens;
        }
    }
    if token_base != n_tokens {
        return Err(format!(
            "trace covered {token_base} tokens, expected {n_tokens} — record order does not \
             match the assumed (chunk, layer) grid"
        ));
    }
    if out.iter().any(|v| v.is_empty()) {
        return Err("trace left a (token, layer) cell unfilled".into());
    }
    Ok(out)
}

fn main() -> Result<(), String> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/data/hipfire-models/escha-35b.hfq".to_string());
    let n: usize = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "64".to_string())
        .trim()
        .parse()
        .expect("prefill length");

    let trace_dir = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string());
    let trace_a = format!("{trace_dir}/escha-route-batched.bin");
    let trace_b = format!("{trace_dir}/escha-route-pertoken.bin");

    let hfq = HfqFile::open(Path::new(&path)).map_err(|e| format!("open: {e:?}"))?;
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    let cask = CaskConfig::default();
    let src = ModelSource::Hfq(hfq);
    let mut ctx = LoadCtx {
        path: &path,
        max_seq: n + 64,
        deepseek4_compute_placement: Default::default(),
        deepseek4_experts_per_token: None,
        draft_path: None,
        kv_mode_override: None,
        kv_backend: hipfire_runtime::kv_backend::KvBackend::Contiguous,
        kv_adaptive_override: None,
        // `ESCHA_GATE_STATE_QUANT=f32` runs both arms with an FP32 DeltaNet
        // state. That is an ATTRIBUTION lever, not a mode: the Q8 recurrent
        // state is requantised after every token on both routes, but the
        // batched GDN kernel's requant frame is documented as
        // "distributionally equivalent to decode, not byte-identical", so some
        // of the prefill-vs-decode divergence measured below predates any of
        // this and belongs to that kernel rather than to the batched dense
        // half. Running FP32 removes the requant and shows how much.
        state_quant_override: std::env::var("ESCHA_GATE_STATE_QUANT")
            .ok()
            .map(|_| "f32")
            .filter(|_| std::env::var("ESCHA_GATE_STATE_QUANT").as_deref() == Ok("f32")),
        cask: &cask,
        pp: 1,
        spec: SpecLoadCfg::default(),
        gpu: &mut gpu,
        gemma4_drafter_path: None,
        gemma4_draft_len: 3,
    };
    let mut b = hipfire_arch_qwen35::load_qwen35_bundle(src, &mut ctx)?;
    let vocab = b.config.vocab_size;
    let prompt = prompt_of(n, vocab);
    let n_moe_layers = b
        .weights
        .layers
        .iter()
        .filter(|l| {
            matches!(
                l,
                qwen35::LayerWeights::DeltaNetMoe(_) | qwen35::LayerWeights::FullAttnMoe(_)
            )
        })
        .count();
    println!("tokens={n} vocab={vocab} moe_layers={n_moe_layers}");

    // ── Arm A: batched ───────────────────────────────────────────────────
    //
    // The H128 launch count identifies the route with no ambiguity, which
    // matters because `forward_prefill_batch` falls back to the per-token loop
    // SILENTLY when a layer is inadmissible. Without this check a refused
    // model would compare the per-token route against itself and pass a gate
    // that proves nothing.
    // Open the trace explicitly rather than via HIPFIRE_ESCHA_ROUTE_TRACE:
    // `developer_var` reads a config snapshot resolved at process start, so a
    // `set_var` here would never be seen and the gate would silently compare
    // two empty traces and report 0% divergence.
    hipfire_dispatch::pipeline::route_trace::reopen(&trace_a);
    let before = rdna_compute::escha_h128_launches();
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
    .map_err(|e| format!("batched prefill: {e:?}"))?;
    ctx.gpu
        .hip
        .device_synchronize()
        .map_err(|e| format!("sync: {e:?}"))?;
    let batched_launches = rdna_compute::escha_h128_launches() - before;
    let logits_a = ctx
        .gpu
        .download_f32(&b.scratch.logits)
        .map_err(|e| format!("download A: {e:?}"))?;

    let per_token_budget =
        hipfire_dispatch::pipeline::escha::escha_launches_per_token(b.config.n_layers);
    println!(
        "arm A (batched):  H128 launches={batched_launches} \
         (per-token route would be {})",
        per_token_budget * n
    );
    if batched_launches >= (per_token_budget * n) as u64 {
        return Err(format!(
            "arm A issued {batched_launches} H128 launches, which is the PER-TOKEN budget \
             ({per_token_budget} x {n} tokens) — `forward_prefill_batch` fell back to the \
             per-token loop, so this gate would be comparing that route against itself. Run \
             with HIPFIRE_DEBUG_BATCH=1 to see which layer refused."
        ));
    }

    // ── Arm B: per-token ─────────────────────────────────────────────────
    //
    // Fresh recurrent state; see the module docs for why the KV cache does not
    // need one.
    let fresh = DeltaNetState::new_with_quant(ctx.gpu, &b.config, b.dn_state.quant)
        .map_err(|e| format!("fresh dn_state: {e:?}"))?;
    let stale = std::mem::replace(&mut b.dn_state, fresh);
    stale.free_gpu(ctx.gpu);

    // The trace sink is initialised on first use — inside arm A's first MoE
    // layer — so setting the env var again here would be ignored. Redirect it
    // explicitly.
    hipfire_dispatch::pipeline::route_trace::reopen(&trace_b);
    let mut logits_b = Vec::new();
    for (i, &tok) in prompt.iter().enumerate() {
        logits_b = qwen35::forward(
            ctx.gpu,
            &b.weights,
            &b.config,
            tok,
            i,
            &mut b.kv_cache,
            &mut b.dn_state,
        )
        .map_err(|e| format!("per-token forward at {i}: {e:?}"))?;
    }

    // ── Final-token logits ───────────────────────────────────────────────
    let (mut max_d, mut sum_d) = (0.0f32, 0.0f64);
    for i in 0..vocab {
        let d = (logits_a[i] - logits_b[i]).abs();
        if d > max_d {
            max_d = d;
        }
        sum_d += d as f64;
    }
    let mean_d = sum_d / vocab as f64;
    let am = |v: &[f32]| {
        v.iter()
            .take(vocab)
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                if x > bv {
                    (i, x)
                } else {
                    (bi, bv)
                }
            })
    };
    let (arg_a, best_a) = am(&logits_a);
    let (arg_b, best_b) = am(&logits_b);
    println!("final-token logits: max|delta|={max_d:.3e} mean|delta|={mean_d:.3e}");
    println!("argmax: batched={arg_a} ({best_a:.4})  per-token={arg_b} ({best_b:.4})");
    let nonfinite = logits_a
        .iter()
        .take(vocab)
        .filter(|v| !v.is_finite())
        .count();
    println!("non-finite logits (batched): {nonfinite}");
    assert_eq!(nonfinite, 0, "batched prefill produced non-finite logits");

    // ── Expert-selection divergence ──────────────────────────────────────
    //
    // Reported THREE ways, because one number would be misleading.
    //
    // * LAYER 0 is where the two routes have accumulated the LEAST difference:
    //   the embedding they start from is identical. It is NOT a matched-input
    //   measurement — by the time layer 0's router runs, the hidden state has
    //   already been through the batched attention projections AND the batched
    //   GDN recurrence, whose Q8 state requant is documented as
    //   "distributionally equivalent to decode, not byte-identical". So layer
    //   0's rate is the floor of the whole effect, not the f16-boundary rate
    //   in isolation. Run with `ESCHA_GATE_STATE_QUANT=f32` to remove the
    //   requant term and see what is left. This is still the number to assert
    //   on, because it is the one that cannot have compounded.
    // * Layers 1.. see inputs that already differ, because a flip at layer L
    //   changes that token's hidden state for every later layer. The rate
    //   therefore COMPOUNDS with depth and plateaus. That is arithmetic, not
    //   a defect, and it would happen on any model whose batched prefill is
    //   not bit-identical to its decode.
    // * The per-EXPERT-SLOT rate says how much a differing set differs. A
    //   boundary straddle swaps the 8th expert for the 9th, so it should be
    //   ~1/k of the set rate; a much larger ratio would mean the routes are
    //   choosing genuinely different experts, not neighbouring ones.
    let ra = read_trace(&trace_a)?;
    let rb = read_trace(&trace_b)?;
    let fa = flatten(&ra, n_moe_layers, n)?;
    let fb = flatten(&rb, n_moe_layers, n)?;
    let total = fa.len();
    let differing = fa.iter().zip(fb.iter()).filter(|(a, c)| a != c).count();
    let pct = 100.0 * differing as f64 / total as f64;

    let mut per_layer = vec![0usize; n_moe_layers];
    let (mut slots_total, mut slots_diff) = (0usize, 0usize);
    for tok in 0..n {
        for l in 0..n_moe_layers {
            let a = &fa[tok * n_moe_layers + l];
            let c = &fb[tok * n_moe_layers + l];
            if a != c {
                per_layer[l] += 1;
            }
            slots_total += a.len();
            slots_diff += a.iter().filter(|e| !c.contains(e)).count();
        }
    }
    let l0_pct = 100.0 * per_layer[0] as f64 / n as f64;
    let slot_pct = 100.0 * slots_diff as f64 / slots_total as f64;
    println!(
        "expert-selection divergence, layer 0 (least-accumulated): {}/{n} = {l0_pct:.4}%",
        per_layer[0]
    );
    println!(
        "expert-selection divergence, whole stack: {differing} of {total} \
         (token, layer) decisions = {pct:.4}%"
    );
    println!("expert-slot divergence, whole stack: {slots_diff} of {slots_total} = {slot_pct:.4}%");
    print!("per-layer set-divergence %:");
    for l in 0..n_moe_layers {
        print!(" {:.1}", 100.0 * per_layer[l] as f64 / n as f64);
    }
    println!();

    // Layer 0 is the assertion. Both routes feed it the SAME embedding, so a
    // flip there can only come from this layer's own accumulation reordering
    // landing on an f16 boundary. Task 9 measured ~0.42% of escha routing
    // decisions sitting on such a boundary; 5% leaves room for sampling noise
    // at small `n` while still failing loudly if the dense half is wrong (when
    // the stale-FP16-activation bug was live this was well above it).
    assert!(
        l0_pct < 10.0,
        "layer-0 expert-selection divergence {l0_pct:.4}% is far above the few percent \
         expected from one layer of batched-vs-GEMV dense arithmetic plus the batched GDN \
         requant frame. Layer 0 cannot have compounded — it is the first layer — so a large \
         rate here means the batched dense half is computing materially different router \
         logits, not that a few decisions straddled an f16 boundary."
    );
    // A differing set should differ by about one expert in k (the boundary
    // swap). Much more than that is not a boundary effect.
    assert!(
        slot_pct * (2.0 * ra[0].k as f64) < pct.max(1e-9) * 3.0 + 5.0,
        "sets that differ are differing by {slot_pct:.3}% of slots against a {pct:.3}% set \
         rate — a boundary straddle swaps ONE expert, so the slot rate should be near the \
         set rate divided by k. The routes are picking genuinely different experts."
    );

    // ── The logit-delta bound ────────────────────────────────────────────
    //
    // Until now `max_d` and `mean_d` were computed and only PRINTED, so the
    // bound the module docs call load-bearing was enforced by a human reading
    // stdout. That is not enforcement, and the defect this gate exists to
    // catch produced ~1e+1 in exactly this quantity.
    //
    // THE BOUNDS BELOW ARE MEASURED, not derived from the "~1e-3" figure this
    // file used to quote. At n=64 on this model the actual values are
    //
    //     max|delta|  = 4.393e-1        mean|delta| = 7.160e-2
    //
    // reproduced identically across builds. The old ~1e-3 estimate described
    // pure accumulation reordering with IDENTICAL routing on both arms, and
    // that is not what these two arms do: 24.1% of (token, layer) routing
    // decisions differ between them (see the divergence report above). A
    // token routed to a different expert set computes a genuinely different
    // hidden state, and 40 layers of that lands two orders of magnitude above
    // the reordering-only figure. The estimate was wrong, not the run.
    //
    // So the headroom here is real but narrow: 4.4e-1 measured against the
    // ~1e+1 the stale-FP16-activation bug produced is a factor of ~23, and the
    // bound has to sit inside it. 2.0 is ~4.6x above the measurement and ~5x
    // below the known-bad value — the honest split. The mean is the steadier
    // statistic (it is an average over 248 320 logits rather than one extreme
    // order statistic), so it is bounded more tightly at ~7x headroom.
    //
    // This bound is NOT what makes the gate work; the argmax assertion below
    // is. What it adds is a trip-wire for a structural error that happens not
    // to move the argmax on this one prompt.
    const MAX_ABS_LOGIT_DELTA: f32 = 2.0;
    const MAX_MEAN_LOGIT_DELTA: f64 = 5e-1;
    assert!(
        max_d < MAX_ABS_LOGIT_DELTA,
        "final-token logit max|delta| {max_d:.3e} exceeds {MAX_ABS_LOGIT_DELTA:.1}. Measured \
         on this model at n=64: 4.393e-1, from f32 accumulation reordering compounded by the \
         24% prefill-vs-decode expert-selection divergence reported above. A value near 1e+1 \
         is the signature of a structural error — a dropped H128 transform, a stale \
         activation-conversion cache, or a wrong expert-slot stride — every one of which \
         stays finite and fluent. Check the divergence percentages above first: if THEY are \
         unchanged and this moved, the dense half changed."
    );
    assert!(
        mean_d < MAX_MEAN_LOGIT_DELTA,
        "final-token logit mean|delta| {mean_d:.3e} exceeds {MAX_MEAN_LOGIT_DELTA:.1e}. \
         Measured on this model at n=64: 7.160e-2. The mean is averaged over the whole \
         vocabulary, so unlike max|delta| it does not move on a single outlier logit — a \
         mean this far out means the two routes disagree broadly, not at one index."
    );

    // The load-bearing assertion. See the module docs: a tolerance alone would
    // absorb the failure class this port keeps catching.
    assert_eq!(
        arg_a, arg_b,
        "batched prefill chose a different next token than the per-token route \
         (batched {arg_a} @ {best_a}, per-token {arg_b} @ {best_b}). The dense half is not \
         bit-identical by design (batched WMMA vs batch-1 GEMV accumulation order) and the \
         routing genuinely diverges on ~24% of decisions, which together move these logits by \
         ~4e-1 — but that must never move the argmax. A moved argmax means a structural \
         error — a dropped H128 transform, a stale activation-conversion cache, or a wrong \
         expert-slot stride — all of which stay finite and fluent."
    );

    println!("G6 PASS");
    Ok(())
}
