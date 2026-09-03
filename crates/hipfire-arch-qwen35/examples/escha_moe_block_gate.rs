// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! G4 (Escha-W2 port, Task 10): arch-6 must reproduce EschaLabs' layer-0 MoE
//! block.
//!
//! Runs the shipped `moeblk_x.f16` through hipfire's layer-0 MoE block with
//! `moeblk_ids.i64` / `moeblk_scores.f32` **injected** — the fixture ships the
//! routing precisely because it does not gate the router; that was Task 9's
//! job (`examples/escha_router_contract.rs`). What this gates is the part
//! Task 10 built: expert loading (trellis decode -> transpose -> Q8_0) and the
//! H128-wrapped, batched-across-experts routed executor.
//!
//! The golden is `routed + shared expert`, with no residual add — verified by
//! decomposition, not assumed (the routed sum alone lands at cos 0.266 against
//! the golden and 22% of its magnitude; adding the shared expert takes it to
//! cos 1.00000).
//!
//! # This is a TOLERANCE gate, and it has two arms
//!
//! The golden came from EschaLabs' Metal path, not from `ref.py`, so exact
//! agreement is not available at any weight precision. The codec goldens
//! (G2 `test_escha_decode_gpu_vs_cpu`, G3 `test_escha_h128_gpu_vs_cpu`) ARE
//! bit-exact; do NOT generalise the bounds below to them.
//!
//! Two arms run, because the two error sources are independent and must not
//! be allowed to hide each other:
//!
//! * **F32 arm** — experts stored as the exactly-decoded fp16 widened to f32,
//!   no re-quantisation. This isolates the WIRING: transpose orientation,
//!   H128 placement, SwiGLU half order, the f16(score) combine. If the H128
//!   pair were missing, this arm lands near 1e-1, not 1e-4.
//! * **Q8_0 arm** — production storage. The delta between the arms IS the cost
//!   of the 8-bit re-quantisation, reported explicitly rather than buried in
//!   a single pass/fail number.
//!
//! Run:
//!   cargo run --release -p hipfire-arch-qwen35 \
//!     --example escha_moe_block_gate -- /data/hipfire-models/escha-35b.hfq

use hipfire_arch_qwen35::qwen35::escha::{load_escha_moe_experts, EschaWeightStore};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::escha::{
    escha_launches_per_token, escha_routed_decode, escha_routed_decode_indexed, EschaIndexedRouting,
};
use hipfire_quantize::float16::f16_to_f32;
use hipfire_runtime::hfq::{load_weight_tensor_pread, HfqFile};
use hipfire_runtime::llama::{weight_gemv, WeightTensor};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../hipfire-quantize/tests/data/escha")
        .join(name)
}

fn read_f16(name: &str) -> Vec<f32> {
    std::fs::read(fixture(name))
        .expect("run crates/hipfire-quantize/tests/data/escha/fetch-goldens.sh first")
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// The candidate-name expander the qwen35 loader uses. The escha `.hfq`
/// already carries fully-qualified `model.language_model.*` names, so this is
/// the identity for every name below; passing the real expander keeps the gate
/// on the same lookup path production takes.
fn exact_or_prefixed(name: &str) -> Vec<String> {
    if name.starts_with("model.") {
        vec![name.to_string()]
    } else {
        vec![
            format!("model.language_model.{name}"),
            format!("model.{name}"),
            name.to_string(),
        ]
    }
}

fn upload_f32(gpu: &Gpu, v: &[f32]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
    gpu.upload_raw(bytes, &[v.len()]).expect("upload")
}

struct SharedExpert {
    gate: WeightTensor,
    up: WeightTensor,
    down: WeightTensor,
    scalar_gate: WeightTensor,
}

/// The shared expert, run exactly as `run_moe_decode_cpu_fallback`'s generic
/// (non-MQ4) shared-down arm runs it — sigmoid(gate·x) scaling a SwiGLU MLP,
/// accumulated into the output. Unchanged arch-6 code; Task 10 does not touch
/// it, and it is here only because the golden includes it.
fn run_shared_expert(
    gpu: &mut Gpu,
    w: &SharedExpert,
    x: &GpuTensor,
    out: &GpuTensor,
    smi: usize,
    hidden: usize,
) {
    let scalar = gpu.alloc_tensor(&[1], DType::F32).unwrap();
    let g = gpu.alloc_tensor(&[smi], DType::F32).unwrap();
    let u = gpu.alloc_tensor(&[smi], DType::F32).unwrap();
    let h = gpu.alloc_tensor(&[smi], DType::F32).unwrap();
    let y = gpu.alloc_tensor(&[hidden], DType::F32).unwrap();

    weight_gemv(gpu, &w.scalar_gate, x, &scalar).unwrap();
    gpu.sigmoid_f32(&scalar).unwrap();
    weight_gemv(gpu, &w.gate, x, &g).unwrap();
    weight_gemv(gpu, &w.up, x, &u).unwrap();
    gpu.silu_mul_f32(&g, &u, &h).unwrap();
    weight_gemv(gpu, &w.down, &h, &y).unwrap();
    gpu.scaled_add_inplace_gpu_scalar_f32(out, &y, &scalar)
        .unwrap();

    for t in [scalar, g, u, h, y] {
        let _ = gpu.free_tensor(t);
    }
}

struct ArmResult {
    max_abs: f32,
    mean_abs: f32,
    got: Vec<f32>,
    /// `Some(n)` when the indexed (GPU-top-K) route was cross-checked on this
    /// arm: the number of ROUTED-ONLY output floats that did NOT match the
    /// CPU-top-K route bit-for-bit, across every token. Must be 0.
    indexed_mismatches: Option<usize>,
}

/// Device pointer table over the loaded experts, in the `[n_exp]` packed-u64
/// layout every indexed MoE GEMV consumes. Built here the same way
/// `qwen35::load` builds it for production, because the indexed executor
/// reaches the weights ONLY through this table — a gate that passed its own
/// hand-rolled table would not be gating the production addressing.
fn expert_ptr_table(gpu: &Gpu, ptrs: &[u64]) -> GpuTensor {
    let bytes: Vec<u8> = ptrs.iter().flat_map(|p| p.to_ne_bytes()).collect();
    gpu.upload_raw(&bytes, &[2 * ptrs.len()])
        .expect("ptr table")
}

#[allow(clippy::too_many_arguments)]
fn run_arm(
    gpu: &mut Gpu,
    hfq: &HfqFile,
    layer_prefix: &str,
    shared: &SharedExpert,
    store: EschaWeightStore,
    x: &[f32],
    want: &[f32],
    ids: &[i64],
    scores: &[f32],
    n_tok: usize,
    top_k: usize,
    n_exp: usize,
    hidden: usize,
    mi: usize,
    smi: usize,
) -> ArmResult {
    let all: Vec<usize> = (0..n_exp).collect();
    let (experts, tables, owners) = load_escha_moe_experts(
        hfq,
        gpu,
        layer_prefix,
        &all,
        n_exp,
        hidden,
        mi,
        top_k,
        store,
        exact_or_prefixed,
    )
    .expect("escha expert load");

    let refs = tables.refs();
    let routed: Vec<_> = experts
        .iter()
        .map(|e| (e.gate_up.dispatch_ref(), e.down.dispatch_ref()))
        .collect();
    let ctx = DispatchCtx::new(gpu);

    let out = gpu.alloc_tensor(&[hidden], DType::F32).unwrap();
    let zeros = vec![0.0f32; hidden];
    let mut got = vec![0.0f32; n_tok * hidden];

    // ── Indexed (GPU-top-K) route cross-check ────────────────────────────
    //
    // Production decode and prefill take `escha_routed_decode_indexed`, not
    // the host-routed executor this gate was originally written against. The
    // two must agree BIT-FOR-BIT: they are the same eight phases over the
    // same weights, differing only in where the routing lives, so any
    // difference at all is a defect (a changed GEMV accumulate order, a
    // device-vs-host disagreement in the f16 score rounding, a wrong slot
    // stride) — not a tolerance question. Asserting equality rather than a
    // bound is what lets the golden tolerances below keep meaning ONE thing
    // for both routes.
    //
    // Q8_0 only: the indexed GEMV decodes the Q8_0 34 B/32-element block
    // layout, so the F32 weight-exact control arm has no indexed counterpart
    // (and needs none — it exists to isolate wiring, which both routes share).
    let indexed = if matches!(store, EschaWeightStore::Q8_0) {
        let gu_ptrs: Vec<u64> = experts
            .iter()
            .map(|e| e.gate_up.buf.buf.as_ptr() as u64)
            .collect();
        let dn_ptrs: Vec<u64> = experts
            .iter()
            .map(|e| e.down.buf.buf.as_ptr() as u64)
            .collect();
        Some((
            expert_ptr_table(gpu, &gu_ptrs),
            expert_ptr_table(gpu, &dn_ptrs),
            gpu.alloc_tensor(&[top_k], DType::F32).unwrap(), // ids (i32 bits)
            gpu.alloc_tensor(&[top_k], DType::F32).unwrap(), // raw scores
            gpu.alloc_tensor(&[hidden], DType::F32).unwrap(), // routed-only out
        ))
    } else {
        None
    };
    let mut indexed_mismatches = indexed.as_ref().map(|_| 0usize);

    for t in 0..n_tok {
        let x_gpu = upload_f32(gpu, &x[t * hidden..(t + 1) * hidden]);
        gpu.hip
            .memcpy_htod(&out.buf, unsafe {
                std::slice::from_raw_parts(zeros.as_ptr() as *const u8, hidden * 4)
            })
            .unwrap();

        let slot_ids: Vec<usize> = ids[t * top_k..(t + 1) * top_k]
            .iter()
            .map(|&v| v as usize)
            .collect();
        let slot_w = &scores[t * top_k..(t + 1) * top_k];
        escha_routed_decode(
            &ctx, gpu, &refs, &routed, &slot_ids, slot_w, &x_gpu, &out, hidden, mi,
        )
        .expect("escha routed decode");

        if let (Some((gu_tbl, dn_tbl, ids_dev, wts_dev, out_idx)), Some(bad)) =
            (indexed.as_ref(), indexed_mismatches.as_mut())
        {
            // Routing goes up ONCE, as the device buffers the GPU top-K
            // kernel would have written: ids as i32 bits in an F32 tensor,
            // scores UNROUNDED (the executor's own kernel does the f16
            // round-trip — that is part of what is being gated).
            let id_bytes: Vec<u8> = slot_ids
                .iter()
                .flat_map(|&i| (i as i32).to_le_bytes())
                .collect();
            gpu.hip.memcpy_htod(&ids_dev.buf, &id_bytes).unwrap();
            let w_bytes: Vec<u8> = slot_w.iter().flat_map(|w| w.to_le_bytes()).collect();
            gpu.hip.memcpy_htod(&wts_dev.buf, &w_bytes).unwrap();
            gpu.hip
                .memcpy_htod(&out_idx.buf, unsafe {
                    std::slice::from_raw_parts(zeros.as_ptr() as *const u8, hidden * 4)
                })
                .unwrap();

            escha_routed_decode_indexed(
                gpu,
                &refs,
                &EschaIndexedRouting {
                    expert_gate_up_ptrs: gu_tbl,
                    expert_down_ptrs: dn_tbl,
                    topk_indices: ids_dev,
                    topk_weights: wts_dev,
                    gate_up_m: 2 * mi,
                    gate_up_k: hidden,
                    down_m: hidden,
                    down_k: mi,
                },
                out_idx,
                &x_gpu,
                hidden,
                mi,
                top_k,
            )
            .expect("escha routed decode (indexed)");

            gpu.hip.device_synchronize().unwrap();
            let host_route = gpu.download_f32(&out).unwrap();
            let idx_route = gpu.download_f32(out_idx).unwrap();
            *bad += host_route[..hidden]
                .iter()
                .zip(idx_route[..hidden].iter())
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
        }

        run_shared_expert(gpu, shared, &x_gpu, &out, smi, hidden);

        gpu.hip.device_synchronize().unwrap();
        let row = gpu.download_f32(&out).unwrap();
        got[t * hidden..(t + 1) * hidden].copy_from_slice(&row[..hidden]);
        let _ = gpu.free_tensor(x_gpu);
    }

    let diffs: Vec<f32> = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .collect();
    let max_abs = diffs.iter().cloned().fold(0.0f32, f32::max);
    let mean_abs = diffs.iter().sum::<f32>() / diffs.len() as f32;

    let _ = gpu.free_tensor(out);
    if let Some((gu_tbl, dn_tbl, ids_dev, wts_dev, out_idx)) = indexed {
        for t in [gu_tbl, dn_tbl, ids_dev, wts_dev, out_idx] {
            let _ = gpu.free_tensor(t);
        }
    }
    // Expert slots are non-owning views into `owners` (one blob per
    // projection), so return the two blobs and NOT the 512 views. `free_all`
    // on a view is refused by `free_tensor` and would leak the blob — this
    // gate runs the F32 arm at 256 experts, i.e. 2 GiB per projection, and
    // both arms run in one process.
    drop(experts);
    let _ = gpu.free_tensor(owners.gate_up);
    let _ = gpu.free_tensor(owners.down);
    tables.free_gpu(gpu);

    ArmResult {
        max_abs,
        mean_abs,
        got,
        indexed_mismatches,
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/data/hipfire-models/escha-35b.hfq".to_string());
    let hfq = HfqFile::open(std::path::Path::new(&path)).expect("open hfq");
    let config = hipfire_arch_qwen35::qwen35::config::config_from_hfq(&hfq).expect("config");
    let hidden = config.dim;
    let mi = config.moe_intermediate_size;
    let smi = config.shared_expert_intermediate_size;
    let n_exp = config.num_experts;
    let top_k = config.num_experts_per_tok;
    let layer_prefix = "model.language_model.layers.0";

    let x = read_f16("moeblk_x.f16");
    let want = read_f16("moeblk_out.f16");
    let n_tok = x.len() / hidden;
    assert_eq!(want.len(), n_tok * hidden, "fixture shape mismatch");
    let ids: Vec<i64> = std::fs::read(fixture("moeblk_ids.i64"))
        .unwrap()
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let scores: Vec<f32> = std::fs::read(fixture("moeblk_scores.f32"))
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(ids.len(), n_tok * top_k);
    assert_eq!(scores.len(), n_tok * top_k);

    let mut gpu = Gpu::init().expect("gpu");
    assert!(
        hipfire_arch_qwen35::qwen35::escha::layer_is_escha(&hfq, layer_prefix, exact_or_prefixed),
        "layer 0 of {path} does not carry Escha-W2 routed experts"
    );

    let shared = SharedExpert {
        gate: load_weight_tensor_pread(
            &hfq,
            &gpu,
            &format!("{layer_prefix}.mlp.shared_expert.gate_proj.weight"),
            smi,
            hidden,
            exact_or_prefixed,
        )
        .unwrap(),
        up: load_weight_tensor_pread(
            &hfq,
            &gpu,
            &format!("{layer_prefix}.mlp.shared_expert.up_proj.weight"),
            smi,
            hidden,
            exact_or_prefixed,
        )
        .unwrap(),
        down: load_weight_tensor_pread(
            &hfq,
            &gpu,
            &format!("{layer_prefix}.mlp.shared_expert.down_proj.weight"),
            hidden,
            smi,
            exact_or_prefixed,
        )
        .unwrap(),
        scalar_gate: load_weight_tensor_pread(
            &hfq,
            &gpu,
            &format!("{layer_prefix}.mlp.shared_expert_gate.weight"),
            1,
            hidden,
            exact_or_prefixed,
        )
        .unwrap(),
    };

    let mag = want.iter().map(|v| v.abs()).sum::<f32>() / want.len() as f32;
    println!("tokens={n_tok} hidden={hidden} top_k={top_k} experts={n_exp}");
    println!("golden mean magnitude: {mag:.4e}");

    // ── Arm 1: weight-exact (F32) — isolates the wiring ──────────────────
    let launches_before = rdna_compute::escha_h128_launches();
    let f32_arm = run_arm(
        &mut gpu,
        &hfq,
        layer_prefix,
        &shared,
        EschaWeightStore::F32,
        &x,
        &want,
        &ids,
        &scores,
        n_tok,
        top_k,
        n_exp,
        hidden,
        mi,
        smi,
    );
    let launches_one_layer_all_tokens = rdna_compute::escha_h128_launches() - launches_before;
    println!(
        "MoE block [F32 experts, weight-exact]: max|diff|={:.3e} mean|diff|={:.3e}",
        f32_arm.max_abs, f32_arm.mean_abs
    );

    // ── Arm 2: production (Q8_0) ─────────────────────────────────────────
    let q8_arm = run_arm(
        &mut gpu,
        &hfq,
        layer_prefix,
        &shared,
        EschaWeightStore::Q8_0,
        &x,
        &want,
        &ids,
        &scores,
        n_tok,
        top_k,
        n_exp,
        hidden,
        mi,
        smi,
    );
    println!(
        "MoE block [Q8_0 experts, production]:  max|diff|={:.3e} mean|diff|={:.3e}",
        q8_arm.max_abs, q8_arm.mean_abs
    );

    // ── Indexed route == host route, bit-for-bit ─────────────────────────
    // Production decode/prefill run `escha_routed_decode_indexed`. The
    // tolerances asserted below were measured on the host-routed executor, so
    // they only describe production if the two routes are the SAME numbers —
    // which they are by construction (identical phases, identical GEMV
    // accumulate order, the f16 score rounding moved from host to kernel) and
    // must therefore be provable exactly, not to a tolerance.
    let n_indexed = q8_arm
        .indexed_mismatches
        .expect("the Q8_0 arm must cross-check the indexed route");
    println!("indexed route vs host route: {n_indexed} differing floats (want 0)");
    assert_eq!(
        n_indexed, 0,
        "the indexed (GPU-top-K) escha route disagreed with the CPU-top-K route on \
         {n_indexed} routed-output floats. These must be BIT-identical: same phases, same \
         weights, same H128 pair, same GEMV accumulate order. Look at (1) the wide-vs-narrow \
         Q8_0 kernel choice in `escha_gemv_q8_0_moe_k8_indexed_batched` — it must reuse \
         `gemv_q8_0`'s `k <= 1536` rule, because the wide kernel folds four interleaved \
         accumulators and the narrow one a single sum; (2) `escha_round_weights_f16_rne` \
         against the host's `half::f16::from_f32`; (3) the per-slot x/y strides. Until this is \
         0 the tolerances below describe a route production does not take."
    );
    assert!(
        f32_arm.indexed_mismatches.is_none(),
        "the F32 control arm has no indexed counterpart (the indexed GEMV is Q8_0-only) — a \
         Some here means the cross-check silently ran against the wrong container"
    );

    let dq: Vec<f32> = q8_arm
        .got
        .iter()
        .zip(f32_arm.got.iter())
        .map(|(a, b)| (a - b).abs())
        .collect();
    let dq_max = dq.iter().cloned().fold(0.0f32, f32::max);
    let dq_mean = dq.iter().sum::<f32>() / dq.len() as f32;
    println!("Q8_0 re-quantisation cost (arm2 - arm1): max={dq_max:.3e} mean={dq_mean:.3e}");

    // ── Launch budget ────────────────────────────────────────────────────
    // Measured, not asserted from a comment: the counter in
    // `Gpu::escha_h128_batched` ticks once per batched transform launch.
    let per_layer_per_token = launches_one_layer_all_tokens as f64 / n_tok as f64;
    let per_token = per_layer_per_token * config.n_layers as f64;
    println!(
        "H128 launches: {per_layer_per_token} per (layer, token) -> {per_token} per token at \
         {} layers (a per-expert wiring would be {})",
        config.n_layers,
        4 * top_k * config.n_layers
    );
    assert_eq!(
        per_token as usize,
        escha_launches_per_token(config.n_layers),
        "H128 launch budget drifted from the batched contract"
    );

    // ── Bounds ───────────────────────────────────────────────────────────
    // Arm 1 (weight-exact) carries the brief's measured tolerance: this is
    // the arm that says "the wiring is right". A missing H128 pair lands at
    // ~1e-1 here, three orders of magnitude outside it.
    assert!(
        f32_arm.max_abs <= 2e-4,
        "F32 arm max|diff| {:.3e} exceeds 2e-4 — with weight-exact experts this can only be a \
         wiring defect (transpose orientation, H128 placement/side, SwiGLU half order, or the \
         f16(score) combine). If it is ~1e-1 the H128 pair is not being applied at all: check \
         that the escha dtypes did not reach a Plain GEMV.",
        f32_arm.max_abs
    );
    // The mean bound is 1.2e-5, NOT the brief's 1e-5. Derivation, because this
    // number is otherwise a knife-edge that would misdiagnose:
    //
    //   measured (deterministic, no sampling)                    9.673e-6
    //   sensitivity to the SHARED expert's rounding contract      7.32e-7
    //     (report 5.2 CPU sweep: 8.942e-6 -> 9.674e-6 mean when only the
    //      shared expert's rounding flips)
    //   bound = 9.673e-6 + 3 x 7.32e-7 = 1.186e-5, rounded to    1.2e-5
    //
    // The brief's 1e-5 left 3.3% headroom = 0.45x that single sensitivity. The
    // shared expert is arch-6 code Task 10 does not own: any benign change to
    // `silu_mul_f32`, `sigmoid_f32`, the shared-down GEMV selection, or the HIP
    // compiler could move the metric further than the entire remaining margin
    // and trip an assert whose message blames escha WIRING. That is a false
    // diagnosis sending the next person after a bug that does not exist.
    //
    // Option (a) from the review — exclude the shared expert from the
    // comparison — was considered and rejected as not practical: the shipped
    // golden `moeblk_out.f16` is `routed + shared` and no routed-only golden
    // exists. Subtracting hipfire's OWN shared output from both sides is
    // algebraically a no-op on the diff:
    //   (r_h + s_h) - (r_e + s_e) == (r_h + s_h - s_h) - (r_e + s_e - s_h)
    // so the shared expert's Metal-vs-hipfire divergence stays in the measured
    // quantity either way. Widening with the derivation written down is the
    // honest option; 1.2e-5 is still ~4 orders below the ~1e-1 a missing H128
    // pair produces, so the gate keeps all of its diagnostic power.
    assert!(
        f32_arm.mean_abs <= 1.2e-5,
        "F32 arm mean|diff| {:.3e} exceeds 1.2e-5. NOTE before you go hunting escha wiring: the \
         SHARED expert is inside this measured quantity (the golden is routed + shared, and no \
         routed-only golden ships), and it is arch-6 code Task 10 does not own. A move of order \
         1e-6 is consistent with a change to silu_mul_f32 / sigmoid_f32 / the shared-down GEMV \
         selection / the HIP compiler, NOT with an escha defect. Only a jump of 1e-4 or more — \
         and especially ~1e-1 — indicates the escha wiring (transpose orientation, H128 \
         placement/side, SwiGLU half order, the f16(score) combine, or the H128 pair not being \
         applied at all). Check the max|diff| assert above and the two bit-exact codec gates \
         (test_escha_decode_gpu_vs_cpu, test_escha_h128_gpu_vs_cpu) first.",
        f32_arm.mean_abs
    );

    // Arm 2 adds the 8-bit re-quantisation on top. That cost is real,
    // irreducible at this storage format, and MEASURED — see the report for
    // the CPU-side derivation that agrees with it. The bound below is set
    // from that measurement with headroom, and its job is to catch a
    // REGRESSION in the quantiser (a wrong block axis, a dropped clamp, a
    // truncating instead of RNE scale), not to re-prove the wiring.
    assert!(
        q8_arm.max_abs <= 4e-4,
        "Q8_0 arm max|diff| {:.3e} exceeds 4e-4",
        q8_arm.max_abs
    );
    assert!(
        q8_arm.mean_abs <= 6e-5,
        "Q8_0 arm mean|diff| {:.3e} exceeds 6e-5",
        q8_arm.mean_abs
    );
    assert!(
        dq_mean <= 5e-5,
        "Q8_0 re-quantisation cost {dq_mean:.3e} exceeds 5e-5 — the quantiser regressed"
    );

    println!("G4 PASS");
}
