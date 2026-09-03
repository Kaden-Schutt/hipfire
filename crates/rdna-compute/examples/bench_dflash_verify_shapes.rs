// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! M0 "potential improvement" microbench for DFlash verify-phase GEMMs.
//!
//! ONE question: at the DFlash verify shape (batch N=16, real weight bytes at
//! real shapes from the actual qwen3.8:27b.mq4 file), how far is the current
//! gfx1100 dispatch from the bandwidth roofline (960 GB/s), per projection
//! and summed over one layer? No new kernels, no dispatch changes — measure
//! what runs.
//!
//! FILE REALITY (checked 2026-09-03): qwen3.8-27b.mq4 stores its dense
//! projections as qt=13 (MQ4G256 v1), NOT qt=44. The v1 and v2 layouts share
//! the identical 136 B/group stride, so every tensor's data_size EQUALS the
//! true MQ4G256V2 byte count for its shape (asserted per row), and the bench
//! uploads the exact file bytes into the production MQ4V2 entry points below.
//! Bandwidth timing is unaffected by header-value interpretation: the dequant
//! path has no data-dependent dispatch, and X is random-but-finite either way.
//!
//! What it does:
//!   1. Parses the HFQ container index directly (std File+Seek only; an
//!      rdna-compute example cannot depend on hipfire-runtime — that would be
//!      a dependency cycle — so the ~40-line index parse from
//!      hipfire-runtime/src/hfq.rs `HfqFile::open_at_offset` is replicated
//!      here: 32 B header, brace-scan for the metadata JSON end, then the
//!      tensor index. Byte layout comments cite hfq.rs line numbers.)
//!   2. Reads `layer_types` from the metadata JSON, takes layer 0 (must be a
//!      DeltaNet/LinearAttention layer) and the first FullAttention layer.
//!   3. Looks up the real tensors by suffix
//!      (`layers.{i}.linear_attn.in_proj_qkv.weight`, ..., `self_attn.q_proj`
//!      etc. — the bare names from
//!      hipfire-arch-qwen35/src/qwen35/load.rs `validate_*` preflight),
//!      asserts qt == 44 (MQ4G256V2), reads M/K from the header shape
//!      ([M, K] — enforced by `validate_mq4_proj_info`), and checks
//!      data_size == M*K/256*136 (`expected_mq4_bytes`).
//!   4. Uploads the REAL weight bytes, allocates finite-random F32 X [N*K]
//!      and zeroed F32 Y [N*M], and times the exact production entry points
//!      the verify path resolves to at batch N (the fused family's batched
//!      run-arms in hipfire-dispatch/src/families/fused_qkv.rs call these
//!      same `gpu.*` methods with `batch_size: Some(n)`, so calling them
//!      directly exercises the identical tier with no DispatchCtx needed):
//!        residual / down / wo / o_proj -> gpu.gemm_mq4g256v2_residual_wmma
//!        gate+up (fused)               -> gpu.gemm_gate_up_hfq4g256_mq4v2
//!        FA qkv (fused)                -> gpu.gemm_qkv_hfq4g256_mq4v2
//!        LA qkvza (fused)              -> gpu.gemm_qkvza_hfq4g256_mq4v2
//!      The kernel SYMBOL that actually fired is asserted per arm via
//!      `rdna_compute::profile::{start,stop}` (one profiled launch per arm).
//!
//! Measurement discipline (from bench_gemv_paired_throughput.rs):
//!   - >= 32 warmup launches per arm before the measured window.
//!   - device-side timing: device_synchronize around a batch of >= 200
//!     launches, report per-launch.
//!   - 3 samples, interleaved arm-by-arm (sample loop outside, arm loop
//!     inside), report MIN and MEDIAN.
//!   - bytes = weight bytes + staged fp16 X (N*K*2) + F32 Y traffic
//!     (N*M*4*2 for residual Y+= RMW; fused outputs counted the same way,
//!     shared X counted once). Achieved GB/s = bytes/us/1e3, % of 960 GB/s,
//!     roofline floor us = bytes/960e9.
//!
//! Prints the table to stdout and also writes it to
//! `$HOME/dflash-m0/verify-shapes.txt` (mkdir -p).
//!
//! Run (on hipx, RX 7900 XTX gfx1100):
//!   CARGO_TARGET_DIR=~/slice-target-dflash-kernels cargo build --release \
//!     -p rdna-compute --features lab --example bench_dflash_verify_shapes
//!   ./<target>/release/examples/bench_dflash_verify_shapes \
//!     [/path/to/qwen3.8-27b.mq4]

use rdna_compute::{DType, Gpu};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};

const ROOFLINE_GBS: f64 = 960.0;
const WARMUP: usize = 32;
const LAUNCHES: usize = 200;
const SAMPLES: usize = 3;
const NS: [usize; 3] = [1, 8, 16];

const MODEL_DEFAULT: &str = "/home/kaden/.hipfire/models/qwen3.8-27b.mq4";

struct HfqTensor {
    name: String,
    qt: u8,
    shape: Vec<u32>,
    data_off: usize,
    data_len: usize,
}

fn u32le(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}
fn u64le(b: &[u8]) -> u64 {
    u64::from_le_bytes([
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
    ])
}

/// Minimal HFQ index parse mirroring HfqFile::open_at_offset (hfq.rs:445+).
/// Returns (canonical_path, metadata_json, tensors).
fn parse_hfq_index(path: &std::path::Path) -> (String, String, Vec<HfqTensor>) {
    let canon = std::fs::canonicalize(path)
        .unwrap_or_else(|e| panic!("canonicalize {}: {e}", path.display()));
    let mut f = File::open(&canon).expect("open hfq");
    let mut hdr = [0u8; 32];
    f.read_exact(&mut hdr).expect("read hfq header");
    assert_eq!(&hdr[0..4], b"HFQM", "not an HFQ container");
    let n_tensors = u32le(&hdr[12..16]) as usize;
    let metadata_offset = u64le(&hdr[16..24]) as usize;
    let data_offset = u64le(&hdr[24..32]) as usize;
    assert!(metadata_offset <= data_offset, "bad meta/data offsets");
    // Region between metadata start and data start holds JSON + index.
    let region_len = data_offset - metadata_offset;
    let mut region = vec![0u8; region_len];
    f.seek(SeekFrom::Start(metadata_offset as u64)).unwrap();
    f.read_exact(&mut region).expect("read hfq meta+index");
    // Brace-scan for the metadata JSON end (hfq.rs:523-568).
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    let mut json_end = 0usize;
    for (i, &b) in region.iter().enumerate() {
        if esc {
            esc = false;
            continue;
        }
        if b == b'\\' && in_str {
            esc = true;
            continue;
        }
        if b == b'"' {
            in_str = !in_str;
            continue;
        }
        if !in_str {
            if b == b'{' {
                depth += 1;
            }
            if b == b'}' {
                depth -= 1;
                if depth == 0 {
                    json_end = i + 1;
                    break;
                }
            }
        }
    }
    assert!(json_end > 0, "metadata JSON not brace-terminated");
    let meta_json = String::from_utf8_lossy(&region[..json_end]).to_string();
    // Tensor index follows the JSON (hfq.rs:571+): u32 n, then per tensor
    // u16 name_len, name, u8 qt, u8 n_dims, n_dims*u32 shape, u32 group, u64 size.
    let mut pos = json_end;
    let idx_n = u32le(&region[pos..pos + 4]) as usize;
    assert_eq!(idx_n, n_tensors, "index count != header count");
    pos += 4;
    let mut tensors = Vec::with_capacity(n_tensors);
    let mut cum = data_offset;
    for _ in 0..n_tensors {
        let nl = u16::from_le_bytes([region[pos], region[pos + 1]]) as usize;
        pos += 2;
        let name = String::from_utf8_lossy(&region[pos..pos + nl]).to_string();
        pos += nl;
        let qt = region[pos];
        pos += 1;
        let nd = region[pos] as usize;
        pos += 1;
        let mut shape = Vec::with_capacity(nd);
        for _ in 0..nd {
            shape.push(u32le(&region[pos..pos + 4]));
            pos += 4;
        }
        pos += 4; // group_size
        let data_len = u64le(&region[pos..pos + 8]) as usize;
        pos += 8;
        tensors.push(HfqTensor {
            name,
            qt,
            shape,
            data_off: cum,
            data_len,
        });
        cum += data_len;
    }
    (canon.display().to_string(), meta_json, tensors)
}

/// Parse `"layer_types": [...]` string array from the metadata JSON config.
/// Handles both top-level and nested-under-"config" placement.
fn parse_layer_types(meta: &str) -> Vec<String> {
    let key = "\"layer_types\"";
    let kpos = meta.find(key).expect("metadata has no layer_types");
    let arr_start = meta[kpos..].find('[').expect("layer_types not an array") + kpos;
    let arr_end = meta[arr_start..]
        .find(']')
        .expect("layer_types array unterminated")
        + arr_start;
    let body = &meta[arr_start + 1..arr_end];
    body.split(',')
        .map(|s| s.trim().trim_matches('"').to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn find_tensor<'a>(tensors: &'a [HfqTensor], suffix: &str) -> &'a HfqTensor {
    tensors
        .iter()
        .find(|t| t.name.ends_with(suffix))
        .unwrap_or_else(|| panic!("tensor not found: *{suffix}"))
}

fn read_tensor_bytes(path: &str, t: &HfqTensor) -> Vec<u8> {
    let mut f = File::open(path).expect("reopen hfq for payload");
    f.seek(SeekFrom::Start(t.data_off as u64)).unwrap();
    let mut buf = vec![0u8; t.data_len];
    f.read_exact(&mut buf).expect("read tensor payload");
    buf
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn gbps(bytes: usize, us: f64) -> f64 {
    bytes as f64 / us / 1e3
}

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Finite-random F32 X in [-1, 1].
fn random_x(nk: usize, seed: u64) -> Vec<f32> {
    let mut st = seed | 1;
    (0..nk)
        .map(|_| {
            let r = (xorshift64(&mut st) >> 11) as f64 / (u64::MAX >> 11) as f64;
            (r as f32 * 2.0 - 1.0).clamp(-1.0, 1.0)
        })
        .collect()
}

fn sync(gpu: &Gpu) {
    gpu.hip.device_synchronize().unwrap();
}

/// Time `LAUNCHES` launches of `launch` (device-sync around, per-launch us).
/// `launch` takes `&mut Gpu` so the caller keeps sole ownership of `gpu`.
fn time_batch(gpu: &mut Gpu, launch: &mut dyn FnMut(&mut Gpu)) -> f64 {
    sync(gpu);
    let t0 = std::time::Instant::now();
    for _ in 0..LAUNCHES {
        launch(gpu);
    }
    sync(gpu);
    t0.elapsed().as_secs_f64() * 1e6 / LAUNCHES as f64
}

/// One profiled launch -> the kernel symbol that actually fired.
fn profile_symbol(gpu: &mut Gpu, launch: &mut dyn FnMut(&mut Gpu)) -> String {
    rdna_compute::profile::start();
    launch(gpu);
    let entries = rdna_compute::profile::stop().unwrap_or_default();
    sync(gpu);
    entries
        .last()
        .map(|e| e.kernel.to_string())
        .unwrap_or_else(|| "(no profile entry)".to_string())
}

struct Arm {
    /// Display label, e.g. "L0 qkvza (fused)".
    label: String,
    /// Entry point called, e.g. "gpu.gemm_qkvza_hfq4g256_mq4v2".
    entry: String,
    /// Weight byte count (real data_size from file).
    w_bytes: usize,
    /// K (shared input dim).
    k: usize,
    /// Output row counts (one per fused output; single for residual).
    ms: Vec<usize>,
    /// Which launch to run: 0=residual(m,k), 1=gate_up, 2=qkv, 3=qkvza.
    kind: u8,
    /// Uploaded weight blobs in launch order.
    w_names: Vec<String>,
}

fn main() {
    let mut out = String::new();
    let mut emit = |s: &str| {
        println!("{s}");
        out.push_str(s);
        out.push('\n');
    };

    let model_arg = std::env::args().nth(1);
    let model_path = std::path::PathBuf::from(model_arg.as_deref().unwrap_or(MODEL_DEFAULT));
    let (canon, meta, tensors) = parse_hfq_index(&model_path);
    emit(&format!("model: {canon}"));
    emit(&format!("tensors in index: {}", tensors.len()));

    let layer_types = parse_layer_types(&meta);
    let n_layers = layer_types.len();
    let n_la = layer_types
        .iter()
        .filter(|s| s.contains("linear"))
        .count();
    let n_fa = layer_types
        .iter()
        .filter(|s| s.contains("full"))
        .count();
    emit(&format!(
        "layers: {n_layers} ({n_la} linear_attention, {n_fa} full_attention)"
    ));
    assert_eq!(n_layers, n_la + n_fa, "unexpected layer type strings");
    assert!(
        layer_types[0].contains("linear"),
        "layer 0 must be a DeltaNet/LinearAttention layer, got {}",
        layer_types[0]
    );
    let fa_layer = layer_types
        .iter()
        .position(|s| s.contains("full"))
        .expect("no full_attention layer found");
    emit(&format!("bench layers: L0 (LA) + L{fa_layer} (FA)"));

    // ---- weight table -----------------------------------------------------
    struct Proj {
        label: String,
        suffix: String,
    }
    let la = 0usize;
    let projs = vec![
        Proj { label: "L0 in_proj_qkv".into(), suffix: format!("layers.{la}.linear_attn.in_proj_qkv.weight") },
        Proj { label: "L0 in_proj_z".into(), suffix: format!("layers.{la}.linear_attn.in_proj_z.weight") },
        Proj { label: "L0 in_proj_a".into(), suffix: format!("layers.{la}.linear_attn.in_proj_a.weight") },
        Proj { label: "L0 in_proj_b".into(), suffix: format!("layers.{la}.linear_attn.in_proj_b.weight") },
        Proj { label: "L0 out_proj".into(), suffix: format!("layers.{la}.linear_attn.out_proj.weight") },
        Proj { label: "L0 gate_proj".into(), suffix: format!("layers.{la}.mlp.gate_proj.weight") },
        Proj { label: "L0 up_proj".into(), suffix: format!("layers.{la}.mlp.up_proj.weight") },
        Proj { label: "L0 down_proj".into(), suffix: format!("layers.{la}.mlp.down_proj.weight") },
        Proj { label: format!("L{fa_layer} q_proj"), suffix: format!("layers.{fa_layer}.self_attn.q_proj.weight") },
        Proj { label: format!("L{fa_layer} k_proj"), suffix: format!("layers.{fa_layer}.self_attn.k_proj.weight") },
        Proj { label: format!("L{fa_layer} v_proj"), suffix: format!("layers.{fa_layer}.self_attn.v_proj.weight") },
        Proj { label: format!("L{fa_layer} o_proj"), suffix: format!("layers.{fa_layer}.self_attn.o_proj.weight") },
        Proj { label: format!("L{fa_layer} gate_proj"), suffix: format!("layers.{fa_layer}.mlp.gate_proj.weight") },
        Proj { label: format!("L{fa_layer} up_proj"), suffix: format!("layers.{fa_layer}.mlp.up_proj.weight") },
        Proj { label: format!("L{fa_layer} down_proj"), suffix: format!("layers.{fa_layer}.mlp.down_proj.weight") },
    ];
    emit(&format!(
        "\n{:>22} {:>4} {:>6} {:>6} {:>12} {:>12}  {}",
        "projection", "qt", "M", "K", "w_bytes", "M*K/256*136", "tensor"
    ));
    struct Dim {
        label: String,
        m: usize,
        k: usize,
        w_bytes: usize,
        payload: Vec<u8>,
    }
    let mut dims: Vec<Dim> = Vec::new();
    let mut skipped = 0usize;
    let mut saw_qt = std::collections::HashSet::new();
    for p in &projs {
        let t = find_tensor(&tensors, &p.suffix);
        saw_qt.insert(t.qt);
        let m = t.shape[0] as usize;
        let k = t.shape[1] as usize;
        let expect = m * (k / 256) * 136;
        let mark = if t.data_len != expect {
            skipped += 1;
            "SKIPPED (size != M*K/256*136)"
        } else {
            ""
        };
        emit(&format!(
            "{:>22} {:>4} {:>6} {:>6} {:>12} {:>12}  {} {mark}",
            p.label, t.qt, m, k, t.data_len, expect, t.name
        ));
        assert_eq!(
            t.data_len, expect,
            "{}: data_size {} != M*K/256*136 {expect}",
            p.label, t.data_len
        );
        assert_eq!(t.shape.len(), 2, "{}: expected 2D shape", p.label);
        let payload = read_tensor_bytes(&canon, t);
        dims.push(Dim { label: p.label.clone(), m, k, w_bytes: t.data_len, payload });
    }
    // FILE REALITY NOTE: qwen3.8-27b.mq4 stores its dense projections as qt=13
    // (MQ4G256 v1), not qt=44. The v1 and v2 layouts share the identical
    // 136 B/group stride, so every data_size here EQUALS the true MQ4G256V2
    // byte count for the same shape (asserted per row above), and the bench
    // uploads these exact file bytes into the production MQ4V2 entry points.
    // Bandwidth timing is unaffected by header-value interpretation (no
    // data-dependent dispatch in the dequant path; X is random either way).
    emit(&format!(
        "NOTE: file dense-projection quants seen on benched tensors: {saw_qt:?} (expected 44 per ticket; file holds v1 qt=13 — same 136 B/group stride, sizes asserted equal)"
    ));
    assert_eq!(skipped, 0, "some projections failed the v2 size check — see SKIPPED rows");
    // Shared-X consistency: the fused qkvza arm feeds ONE x [N x dim] to all
    // four weights, so qkv/z/a/b must share K (out_proj is a separate arm with
    // its own X, as are gate/up). Verify, don't assume.
    for (a, b) in [
        ("L0 in_proj_qkv", "L0 in_proj_z"),
        ("L0 in_proj_qkv", "L0 in_proj_a"),
        ("L0 in_proj_qkv", "L0 in_proj_b"),
        ("L0 gate_proj", "L0 up_proj"),
    ] {
        assert_eq!(d(a).k, d(b).k, "K mismatch {a} vs {b}");
    }

    // ---- build arms -------------------------------------------------------
    // ms = output row counts; w order matches launch arg order.
    let mut arms = vec![
        Arm {
            label: "L0 qkvza (fused qkv+z+a+b)".into(),
            entry: "gpu.gemm_qkvza_hfq4g256_mq4v2".into(),
            w_bytes: d("L0 in_proj_qkv").w_bytes
                + d("L0 in_proj_z").w_bytes
                + d("L0 in_proj_a").w_bytes
                + d("L0 in_proj_b").w_bytes,
            k: d("L0 in_proj_qkv").k,
            ms: vec![d("L0 in_proj_qkv").m, d("L0 in_proj_z").m, d("L0 in_proj_a").m, d("L0 in_proj_b").m],
            kind: 3,
            w_names: vec!["L0 in_proj_qkv".into(), "L0 in_proj_z".into(), "L0 in_proj_a".into(), "L0 in_proj_b".into()],
        },
        Arm {
            label: "L0 out_proj (residual)".into(),
            entry: "gpu.gemm_mq4g256v2_residual_wmma".into(),
            w_bytes: d("L0 out_proj").w_bytes,
            k: d("L0 out_proj").k,
            ms: vec![d("L0 out_proj").m],
            kind: 0,
            w_names: vec!["L0 out_proj".into()],
        },
        Arm {
            label: "L0 gate+up (fused)".into(),
            entry: "gpu.gemm_gate_up_hfq4g256_mq4v2".into(),
            w_bytes: d("L0 gate_proj").w_bytes + d("L0 up_proj").w_bytes,
            k: d("L0 gate_proj").k,
            ms: vec![d("L0 gate_proj").m, d("L0 up_proj").m],
            kind: 1,
            w_names: vec!["L0 gate_proj".into(), "L0 up_proj".into()],
        },
        Arm {
            label: "L0 down_proj (residual)".into(),
            entry: "gpu.gemm_mq4g256v2_residual_wmma".into(),
            w_bytes: d("L0 down_proj").w_bytes,
            k: d("L0 down_proj").k,
            ms: vec![d("L0 down_proj").m],
            kind: 0,
            w_names: vec!["L0 down_proj".into()],
        },
        Arm {
            label: format!("L{fa_layer} qkv (fused q+k+v)"),
            entry: "gpu.gemm_qkv_hfq4g256_mq4v2".into(),
            w_bytes: d(&format!("L{fa_layer} q_proj")).w_bytes
                + d(&format!("L{fa_layer} k_proj")).w_bytes
                + d(&format!("L{fa_layer} v_proj")).w_bytes,
            k: d(&format!("L{fa_layer} q_proj")).k,
            ms: vec![
                d(&format!("L{fa_layer} q_proj")).m,
                d(&format!("L{fa_layer} k_proj")).m,
                d(&format!("L{fa_layer} v_proj")).m,
            ],
            kind: 2,
            w_names: vec![
                format!("L{fa_layer} q_proj"),
                format!("L{fa_layer} k_proj"),
                format!("L{fa_layer} v_proj"),
            ],
        },
        Arm {
            label: format!("L{fa_layer} o_proj (residual)"),
            entry: "gpu.gemm_mq4g256v2_residual_wmma".into(),
            w_bytes: d(&format!("L{fa_layer} o_proj")).w_bytes,
            k: d(&format!("L{fa_layer} o_proj")).k,
            ms: vec![d(&format!("L{fa_layer} o_proj")).m],
            kind: 0,
            w_names: vec![format!("L{fa_layer} o_proj")],
        },
        Arm {
            label: format!("L{fa_layer} gate+up (fused)"),
            entry: "gpu.gemm_gate_up_hfq4g256_mq4v2".into(),
            w_bytes: d(&format!("L{fa_layer} gate_proj")).w_bytes
                + d(&format!("L{fa_layer} up_proj")).w_bytes,
            k: d(&format!("L{fa_layer} gate_proj")).k,
            ms: vec![
                d(&format!("L{fa_layer} gate_proj")).m,
                d(&format!("L{fa_layer} up_proj")).m,
            ],
            kind: 1,
            w_names: vec![
                format!("L{fa_layer} gate_proj"),
                format!("L{fa_layer} up_proj"),
            ],
        },
        Arm {
            label: format!("L{fa_layer} down_proj (residual)"),
            entry: "gpu.gemm_mq4g256v2_residual_wmma".into(),
            w_bytes: d(&format!("L{fa_layer} down_proj")).w_bytes,
            k: d(&format!("L{fa_layer} down_proj")).k,
            ms: vec![d(&format!("L{fa_layer} down_proj")).m],
            kind: 0,
            w_names: vec![format!("L{fa_layer} down_proj")],
        },
    ];

    let mut gpu = Gpu::init_with_device(0).expect("Gpu init");
    emit(&format!("\narch: {}", gpu.arch));

    // Upload real weights once; X per (arm, N) once.
    struct Live {
        ws: Vec<rdna_compute::GpuTensor>,
        xs: Vec<rdna_compute::GpuTensor>, // indexed by NS position
    }
    let mut live: Vec<Live> = Vec::new();
    for (ai, arm) in arms.iter().enumerate() {
        let mut ws = Vec::new();
        for wn in &arm.w_names {
            let dd = d(wn);
            ws.push(
                gpu.upload_raw(&dd.payload, &[dd.m, dd.k])
                    .unwrap_or_else(|e| panic!("upload {}: {e:?}", wn)),
            );
        }
        let mut xs = Vec::new();
        for (ni, n) in NS.iter().enumerate() {
            let xv = random_x(n * arm.k, 0x1234 + (ai as u64) * 7919 + (ni as u64) * 104729);
            xs.push(gpu.upload_f32(&xv, &[*n, arm.k]).expect("upload x"));
        }
        live.push(Live { ws, xs });
    }

    // Profiled symbol per arm (one launch at N=16).
    emit("\nkernel symbols (one profiled launch per arm, N=16):");
    let n16 = NS.iter().position(|&n| n == 16).unwrap();
    let mut syms: Vec<String> = Vec::new();
    for (ai, arm) in arms.iter().enumerate() {
        let yg: Vec<rdna_compute::GpuTensor> = arm
            .ms
            .iter()
            .map(|&m| gpu.zeros(&[16, m], DType::F32).expect("zeros y"))
            .collect();
        let ws = &live[ai].ws;
        let x = &live[ai].xs[n16];
        let sym = match arm.kind {
            0 => profile_symbol(&mut gpu, &mut |g: &mut Gpu| {
                g.gemm_mq4g256v2_residual_wmma(&ws[0], x, &yg[0], arm.ms[0], arm.k, 16)
                    .unwrap()
            }),
            1 => profile_symbol(&mut gpu, &mut |g: &mut Gpu| {
                g.gemm_gate_up_hfq4g256_mq4v2(
                    &ws[0], &ws[1], x, &yg[0], &yg[1], arm.ms[0], arm.ms[1], arm.k, 16,
                )
                .unwrap()
            }),
            2 => profile_symbol(&mut gpu, &mut |g: &mut Gpu| {
                g.gemm_qkv_hfq4g256_mq4v2(
                    &ws[0], &ws[1], &ws[2], x, &yg[0], &yg[1], &yg[2],
                    arm.ms[0], arm.ms[1], arm.ms[2], arm.k, 16,
                )
                .unwrap()
            }),
            _ => profile_symbol(&mut gpu, &mut |g: &mut Gpu| {
                g.gemm_qkvza_hfq4g256_mq4v2(
                    &ws[0], &ws[1], &ws[2], &ws[3], x, &yg[0], &yg[1], &yg[2], &yg[3],
                    arm.ms[0], arm.ms[1], arm.ms[2], arm.ms[3], arm.k, 16,
                )
                .unwrap()
            }),
        };
        emit(&format!("  {:>28}  {}  ->  {sym}", arm.label, arm.entry));
        syms.push(sym);
    }
    for (ai, sym) in syms.iter().enumerate() {
        arms[ai].entry = format!("{} [{}]", arms[ai].entry, sym);
    }

    // Warmups (per arm, N=16 X reused — values don't matter for timing).
    for (ai, arm) in arms.iter().enumerate() {
        let yg: Vec<rdna_compute::GpuTensor> = arm
            .ms
            .iter()
            .map(|&m| gpu.zeros(&[16, m], DType::F32).expect("zeros y"))
            .collect();
        for _ in 0..WARMUP {
            let ws = &live[ai].ws;
            let x = &live[ai].xs[n16];
            match arm.kind {
                0 => gpu
                    .gemm_mq4g256v2_residual_wmma(&ws[0], x, &yg[0], arm.ms[0], arm.k, 16)
                    .unwrap(),
                1 => gpu
                    .gemm_gate_up_hfq4g256_mq4v2(
                        &ws[0], &ws[1], x, &yg[0], &yg[1], arm.ms[0], arm.ms[1], arm.k, 16,
                    )
                    .unwrap(),
                2 => gpu
                    .gemm_qkv_hfq4g256_mq4v2(
                        &ws[0], &ws[1], &ws[2], x, &yg[0], &yg[1], &yg[2],
                        arm.ms[0], arm.ms[1], arm.ms[2], arm.k, 16,
                    )
                    .unwrap(),
                _ => gpu
                    .gemm_qkvza_hfq4g256_mq4v2(
                        &ws[0], &ws[1], &ws[2], &ws[3], x, &yg[0], &yg[1], &yg[2], &yg[3],
                        arm.ms[0], arm.ms[1], arm.ms[2], arm.ms[3], arm.k, 16,
                    )
                    .unwrap(),
            }
        }
    }
    sync(&gpu);

    // ---- timed rows: SAMPLES interleaved, arm loop inside -----------------
    // bytes = weights + staged fp16 X + F32 Y traffic (RMW x2).
    emit(&format!(
        "\nN=rows: warmup={WARMUP}, launches/sample={LAUNCHES}, samples={SAMPLES} interleaved arm-by-arm"
    ));
    emit(&format!(
        "{:>28} {:>3} {:>10} {:>10} {:>12} {:>9} {:>7} {:>10}",
        "arm", "N", "min_us", "med_us", "bytes", "GB/s", "%roof", "floor_us"
    ));
    // medians[arm][ni]
    let mut medians: Vec<Vec<f64>> = vec![vec![0.0; NS.len()]; arms.len()];
    let mut floors: Vec<Vec<f64>> = vec![vec![0.0; NS.len()]; arms.len()];
    // (arm, n-idx, sample, per-launch us, bytes, floor us)
    let mut samples: Vec<(usize, usize, usize, f64, usize, f64)> = Vec::new();
    for s in 0..SAMPLES {
        for (ai, arm) in arms.iter().enumerate() {
            for (ni, &n) in NS.iter().enumerate() {
                let m_out: usize = arm.ms.iter().sum();
                let bytes = arm.w_bytes + n * arm.k * 2 + n * m_out * 4 * 2;
                let floor_us = bytes as f64 / (ROOFLINE_GBS * 1e9) * 1e6;
                let yg: Vec<rdna_compute::GpuTensor> = arm
                    .ms
                    .iter()
                    .map(|&m| gpu.zeros(&[n, m], DType::F32).expect("zeros y"))
                    .collect();
                let ws = &live[ai].ws;
                let x = &live[ai].xs[ni];
                let us = match arm.kind {
                    0 => time_batch(&mut gpu, &mut |g: &mut Gpu| {
                        g.gemm_mq4g256v2_residual_wmma(&ws[0], x, &yg[0], arm.ms[0], arm.k, n)
                            .unwrap()
                    }),
                    1 => time_batch(&mut gpu, &mut |g: &mut Gpu| {
                        g.gemm_gate_up_hfq4g256_mq4v2(
                            &ws[0], &ws[1], x, &yg[0], &yg[1], arm.ms[0], arm.ms[1], arm.k, n,
                        )
                        .unwrap()
                    }),
                    2 => time_batch(&mut gpu, &mut |g: &mut Gpu| {
                        g.gemm_qkv_hfq4g256_mq4v2(
                            &ws[0], &ws[1], &ws[2], x, &yg[0], &yg[1], &yg[2],
                            arm.ms[0], arm.ms[1], arm.ms[2], arm.k, n,
                        )
                        .unwrap()
                    }),
                    _ => time_batch(&mut gpu, &mut |g: &mut Gpu| {
                        g.gemm_qkvza_hfq4g256_mq4v2(
                            &ws[0], &ws[1], &ws[2], &ws[3], x, &yg[0], &yg[1], &yg[2], &yg[3],
                            arm.ms[0], arm.ms[1], arm.ms[2], arm.ms[3], arm.k, n,
                        )
                        .unwrap()
                    }),
                };
                // stash per-sample; print after all samples collected.
                samples.push((ai, ni, s, us, bytes, floor_us));
            }
        }
    }
    // Aggregate + print.
    for (ai, arm) in arms.iter().enumerate() {
        for (ni, &n) in NS.iter().enumerate() {
            let mut us: Vec<f64> = samples
                .iter()
                .filter(|&&(a, i, _, _, _, _)| a == ai && i == ni)
                .map(|&(_, _, _, u, _, _)| u)
                .collect();
            us.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let min = us[0];
            let med = median(us.clone());
            let bytes = samples
                .iter()
                .find(|&&(a, i, _, _, _, _)| a == ai && i == ni)
                .unwrap()
                .4;
            let floor_us = samples
                .iter()
                .find(|&&(a, i, _, _, _, _)| a == ai && i == ni)
                .unwrap()
                .5;
            medians[ai][ni] = med;
            floors[ai][ni] = floor_us;
            emit(&format!(
                "{:>28} {:>3} {:>10.1} {:>10.1} {:>12} {:>9.1} {:>6.1}% {:>10.1}",
                arm.label,
                n,
                min,
                med,
                bytes,
                gbps(bytes, med),
                gbps(bytes, med) / ROOFLINE_GBS * 100.0,
                floor_us
            ));
        }
    }

    // ---- layer sums (median) + 64-layer extrapolation at N=16 --------------
    let ni16 = 2; // NS = [1, 8, 16]
    let la_arms = 0..4;
    let fa_arms = 4..8;
    let sum = |range: std::ops::Range<usize>| -> (f64, f64) {
        let med: f64 = range.clone().map(|a| medians[a][ni16]).sum();
        let fl: f64 = range.map(|a| floors[a][ni16]).sum();
        (med, fl)
    };
    let (la_med, la_fl) = sum(la_arms);
    let (fa_med, fa_fl) = sum(fa_arms);
    emit(&format!(
        "\nLAYER SUM N=16 (median): LA L0: today {la_med:.1} us, roofline {la_fl:.1} us, {:.2}x over roofline",
        la_med / la_fl
    ));
    emit(&format!(
        "LAYER SUM N=16 (median): FA L{fa_layer}: today {fa_med:.1} us, roofline {fa_fl:.1} us, {:.2}x over roofline",
        fa_med / fa_fl
    ));
    let today_ms = (n_la as f64 * la_med + n_fa as f64 * fa_med) / 1000.0;
    let roof_ms = (n_la as f64 * la_fl + n_fa as f64 * fa_fl) / 1000.0;
    emit(&format!(
        "64-LAYER EXTRAPOLATION ({n_la} LA + {n_fa} FA) N=16: today {today_ms:.2} ms, roofline {roof_ms:.2} ms, ceiling speedup {:.2}x",
        today_ms / roof_ms
    ));

    emit("\nentry points (= fused-family batched run-arm callees, batch_size=Some(N); no DispatchCtx needed):");
    for arm in &arms {
        emit(&format!("  {:>28}  {}", arm.label, arm.entry));
    }
    emit("dispatch tier at N=16 (by policy, gemm.rs mqv2_prefill_batch_tile/mqv2_mw_waves):");
    emit("  all BT/policy arms require batch >= 96 (residual/gateup/qkv/qkvza BT4/6/8/12) or");
    emit("  MW waves >= 384; N=16 matches none, so every projection fires the BASE BT1");
    emit("  WMMA kernel (gemm_mq4g256v2_residual_wmma base, gemm_gate_up/qkv/qkvza_mq4g256v2_wmma");
    emit("  base). Symbols above assert this — no _bt4/_bt6/_bt8/_bt12/_mw suffix expected.");

    // ---- save ---------------------------------------------------------------
    let home = std::env::var("HOME").expect("HOME");
    let dir = format!("{home}/dflash-m0");
    std::fs::create_dir_all(&dir).expect("mkdir dflash-m0");
    let fpath = format!("{dir}/verify-shapes.txt");
    let mut f = File::create(&fpath).expect("create verify-shapes.txt");
    f.write_all(out.as_bytes()).expect("write output");
    println!("saved {fpath}");
}
