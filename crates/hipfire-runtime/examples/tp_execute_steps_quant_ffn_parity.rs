// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! PB-TP4-quant proof: a **whole SwiGLU FFN block on REAL MQ4G256 weights**
//! through `execute_steps_tp` == single-device. PB-TP1..4a validated the TP
//! executor on synthetic **F32** weights (`RotationPlan::None`); this closes the
//! gap bjoern's native-quant decision opened — that the SAME executor handles a
//! rotated quant format (`MQ4G256` → `FwhtG256`) under column/row sharding with
//! NO executor change.
//!
//! Why it works with no new code: `launch_op` dispatches every dtype's rotation
//! (`run_auto` applies FWHT internally), and FWHT-G256 is **block-diagonal per
//! 256-element k-group**, so it commutes with a **group-aligned** k-split. In a
//! correct TP dataflow the row-parallel op receives its own on-rank k-slice
//! (produced by the preceding column-parallel gate/up + silu), so each rank
//! FWHT-rotates exactly its groups; the partials sum to the whole:
//! ```text
//!   g_r = W_gate_r · xn      [ColumnShard axis0 → None]   (xn replicated, k=D full)
//!   u_r = W_up_r   · xn      [ColumnShard axis0 → None]
//!   h_r = silu(g_r)*u_r      [SiluMul — on the on-rank inter/tp slice → None]
//!   o   = all_reduce_r(W_down_r · h_r)  [RowShard axis1 → AllReduceOut{D}]
//! ```
//! Weights are layer-0 of a real qwen3-0.6b-llama.mq4 (D=1024, INTER=3072 →
//! inter/tp=1536, both k-dims %256==0 so every shard stays MQ4G256-valid). The
//! reference runs the identical per-op quant kernels on the WHOLE weights (no
//! fusion — matching the TP path). Validated on emulated Tp-2 (gfx1151).
//!
//! Run: HIP_VISIBLE_DEVICES=0 cargo run -p hipfire-runtime --release \
//!          --example tp_execute_steps_quant_ffn_parity [model.mq4]

use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemv::{GemvFamily, WeightRef};
use hipfire_dispatch::pipeline::{execute_steps_tp, GemvInput, Step, TpCollective};
use hipfire_hardware::{CollectiveHint, DeviceMesh, DimKind};
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::weight_manifest::{collective_for_policy, ShardPolicy, WeightEntry};
use hipfire_runtime::weight_store::{fulfill_manifest, WeightHandle, WeightStore};
use rdna_compute::{DType, GpuTensor};

const TP: usize = 2;
const TOL: f32 = 1e-3;
const MQ4G256_QT: u8 = 13;

fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}
fn wref<'a>(t: &'a GpuTensor, m: usize, k: usize) -> WeightRef<'a> {
    // MQ4G256: rotation is derived from the dtype inside run_auto (FWHT-G256),
    // paro/awq absent, row_stride 0 (see WeightTensor::dispatch_ref).
    WeightRef {
        buf: t,
        dtype: DType::MQ4G256,
        m,
        k,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
    }
}
fn resident<'a>(store: &'a WeightStore, name: &str, dev: usize) -> &'a GpuTensor {
    match store.get(name, Some(0), dev) {
        Some(WeightHandle::Resident(t)) => t,
        _ => panic!("{name} not resident on device {dev}"),
    }
}
fn collective_for(policy: &ShardPolicy, out_dim: usize) -> TpCollective {
    match collective_for_policy(policy) {
        Some(CollectiveHint::AllReduce { kind: DimKind::Tp }) => {
            TpCollective::AllReduceOut { dim: out_dim }
        }
        _ => TpCollective::None,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).map(String::as_str).unwrap_or(concat!(
        env!("HOME"),
        "/.hipfire/models/qwen3-0.6b-llama.mq4"
    ));

    // Read real layer-0 FFN weights (raw MQ4G256 bytes) from the HFQ.
    let hfq =
        hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(model_path)).expect("open model");
    let config = hipfire_runtime::hfq::config_from_hfq(&hfq).expect("config");
    let d = config.dim;
    let inter = config.hidden_dim;
    let inter_r = inter / TP;
    assert_eq!(inter % TP, 0, "INTER {inter} not divisible by Tp {TP}");
    assert_eq!(
        inter_r % 256,
        0,
        "inter/tp {inter_r} must be MQ4G256 group-aligned"
    );
    assert_eq!(d % 256, 0, "D {d} must be MQ4G256 group-aligned");

    let read = |name: &str| -> Vec<u8> {
        let (info, bytes) = hfq
            .tensor_data(name)
            .unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(
            info.quant_type, MQ4G256_QT,
            "{name} quant_type {} != MQ4G256({MQ4G256_QT}) — this proof assumes an mq4 model",
            info.quant_type
        );
        bytes.to_vec()
    };
    let gate_b = read("model.layers.0.mlp.gate_proj.weight");
    let up_b = read("model.layers.0.mlp.up_proj.weight");
    let down_b = read("model.layers.0.mlp.down_proj.weight");
    eprintln!("layer-0 FFN MQ4G256: gate/up [{inter},{d}] down [{d},{inter}] (inter/tp={inter_r})");

    std::env::set_var("HIPFIRE_EMULATE_GPUS", "2");
    let mut gpus = match Gpus::init_uniform(TP, TP) {
        Ok(g) => g,
        Err(e) => {
            println!("tp_execute_steps_quant_ffn_parity: SKIPPED (no {TP}-rank Gpus: {e})");
            return;
        }
    };
    let _ = gpus.enable_peer_all().expect("enable_peer_all");
    for dev in gpus.devices.iter_mut() {
        dev.bind_thread().expect("bind");
        let s = dev.hip.stream_create().expect("stream_create");
        dev.active_stream = Some(s);
    }

    // Deterministic post-norm activation xn[D] (skip rmsnorm — this proof targets
    // the quant GEMV + FWHT-under-shard, not norm).
    let xn0: Vec<f32> = (0..d).map(|i| ((i % 17) as f32 - 8.0) * 0.03).collect();
    let xn_bytes: Vec<u8> = xn0.iter().flat_map(|f| f.to_ne_bytes()).collect();

    // ── Single-device reference: whole weights, per-op quant kernels (no fusion) ──
    let gemv = GemvFamily::new();
    let o_ref = {
        let g = &mut gpus.devices[0];
        g.bind_thread().unwrap();
        let ctx = DispatchCtx::new(g);
        let wg = g.upload_raw(&gate_b, &[gate_b.len()]).unwrap();
        let wu = g.upload_raw(&up_b, &[up_b.len()]).unwrap();
        let wd = g.upload_raw(&down_b, &[down_b.len()]).unwrap();
        let xn = g.upload_raw(&xn_bytes, &[d]).unwrap();
        let gt = g.alloc_tensor(&[inter], DType::F32).unwrap();
        let ut = g.alloc_tensor(&[inter], DType::F32).unwrap();
        let ht = g.alloc_tensor(&[inter], DType::F32).unwrap();
        let ot = g.alloc_tensor(&[d], DType::F32).unwrap();
        gemv.run_auto(&ctx, g, &wref(&wg, inter, d), &xn, &gt)
            .unwrap();
        gemv.run_auto(&ctx, g, &wref(&wu, inter, d), &xn, &ut)
            .unwrap();
        g.silu_mul_f32(&gt, &ut, &ht).unwrap();
        gemv.run_auto(&ctx, g, &wref(&wd, d, inter), &ht, &ot)
            .unwrap();
        g.hip
            .stream_synchronize(g.active_stream.as_ref().unwrap())
            .unwrap();
        let mut out = vec![0u8; d * 4];
        g.hip.memcpy_dtoh(&mut out, &ot.buf).unwrap();
        bytes_to_f32(&out)
    };

    // ── TP path: shard the same weights, run the FFN through execute_steps_tp ──
    let p_col = ShardPolicy::ColumnShard { axis: 0 };
    let p_row = ShardPolicy::RowShard { axis: 1 };
    let manifest = vec![
        WeightEntry::layer("w_gate", 0, vec![inter, d], DType::MQ4G256, p_col.clone()),
        WeightEntry::layer("w_up", 0, vec![inter, d], DType::MQ4G256, p_col.clone()),
        WeightEntry::layer("w_down", 0, vec![d, inter], DType::MQ4G256, p_row.clone()),
    ];
    let mesh = DeviceMesh::rect(&[(DimKind::Tp, TP)]);
    let store = fulfill_manifest(&manifest, &mesh, 1, &gpus, |e| {
        let bytes = match e.name.as_str() {
            "w_gate" => gate_b.clone(),
            "w_up" => up_b.clone(),
            _ => down_b.clone(),
        };
        Ok((bytes, DType::MQ4G256))
    })
    .expect("shard weights");

    let group = mesh.group_along(DimKind::Tp, &mesh.coord_of(0));
    let mut xn_r = Vec::with_capacity(TP);
    let mut g_r = Vec::with_capacity(TP);
    let mut u_r = Vec::with_capacity(TP);
    let mut h_r = Vec::with_capacity(TP);
    let mut o_r = Vec::with_capacity(TP);
    for &dev in &group {
        let g = &mut gpus.devices[dev];
        g.bind_thread().unwrap();
        xn_r.push(g.upload_raw(&xn_bytes, &[d]).unwrap());
        g_r.push(g.alloc_tensor(&[inter_r], DType::F32).unwrap());
        u_r.push(g.alloc_tensor(&[inter_r], DType::F32).unwrap());
        h_r.push(g.alloc_tensor(&[inter_r], DType::F32).unwrap());
        o_r.push(g.alloc_tensor(&[d], DType::F32).unwrap());
    }
    let wg_refs: Vec<WeightRef> = group
        .iter()
        .map(|&dev| wref(resident(&store, "w_gate", dev), inter_r, d))
        .collect();
    let wu_refs: Vec<WeightRef> = group
        .iter()
        .map(|&dev| wref(resident(&store, "w_up", dev), inter_r, d))
        .collect();
    let wd_refs: Vec<WeightRef> = group
        .iter()
        .map(|&dev| wref(resident(&store, "w_down", dev), d, inter_r))
        .collect();

    let per_rank_steps: Vec<Vec<Step>> = (0..TP)
        .map(|r| {
            vec![
                Step::Gemv {
                    w: &wg_refs[r],
                    input: GemvInput::Raw(&xn_r[r]),
                    out: &g_r[r],
                },
                Step::Gemv {
                    w: &wu_refs[r],
                    input: GemvInput::Raw(&xn_r[r]),
                    out: &u_r[r],
                },
                Step::SiluMul {
                    gate: &g_r[r],
                    up: &u_r[r],
                    out: &h_r[r],
                },
                Step::Gemv {
                    w: &wd_refs[r],
                    input: GemvInput::Raw(&h_r[r]),
                    out: &o_r[r],
                },
            ]
        })
        .collect();
    let collectives = vec![
        collective_for(&p_col, inter_r),
        collective_for(&p_col, inter_r),
        TpCollective::None,
        collective_for(&p_row, d),
    ];

    execute_steps_tp(&mesh, &mut gpus, &per_rank_steps, &collectives).expect("execute_steps_tp");

    let dev0 = group[0];
    gpus.devices[dev0].bind_thread().unwrap();
    let mut out = vec![0u8; d * 4];
    gpus.devices[dev0]
        .hip
        .memcpy_dtoh(&mut out, &o_r[0].buf)
        .unwrap();
    let o_tp = bytes_to_f32(&out);

    let diff = max_abs_diff(&o_tp, &o_ref);
    let ref_mag = o_ref.iter().map(|v| v.abs()).fold(0.0, f32::max);
    println!(
        "[tp-quant-ffn] real MQ4G256 SwiGLU FFN through execute_steps_tp vs single-device: \
         max|Δ|={diff:.3e} (ref max|o|={ref_mag:.3e})"
    );

    for dev in gpus.devices.iter_mut() {
        if let Some(s) = dev.active_stream.take() {
            dev.bind_thread().unwrap();
            let _ = dev.hip.stream_destroy(s);
        }
    }
    assert!(
        diff < TOL,
        "quant FFN TP diverges from single-device: max|Δ|={diff}"
    );
    println!(
        "tp_execute_steps_quant_ffn_parity: real MQ4G256 FFN through execute_steps_tp == \
         single-device — native-quant TP validated (no executor change)"
    );
}
