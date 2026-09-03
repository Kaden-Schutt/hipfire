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
use hipfire_dispatch::pipeline::escha::{escha_launches_per_token, escha_routed_decode};
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
    let (experts, tables) = load_escha_moe_experts(
        hfq,
        gpu,
        layer_prefix,
        &all,
        n_exp,
        hidden,
        mi,
        top_k,
        store,
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
    for e in experts {
        e.gate_up.free_all(gpu);
        e.down.free_all(gpu);
    }
    tables.free_gpu(gpu);

    ArmResult {
        max_abs,
        mean_abs,
        got,
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
        hipfire_arch_qwen35::qwen35::escha::layer_is_escha(&hfq, layer_prefix),
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
    assert!(
        f32_arm.mean_abs <= 1e-5,
        "F32 arm mean|diff| {:.3e} exceeds 1e-5",
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
