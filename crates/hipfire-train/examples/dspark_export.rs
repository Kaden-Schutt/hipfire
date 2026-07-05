// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! dspark_export (T5b): convert a NATIVE-trained DSpark drafter checkpoint
//! (`DSCK`, written by `hipfire_train::dspark_train::save_dspark_ckpt`) into a
//! runtime-loadable `<target>-<quant>.dspark.hfq` sidecar.
//!
//! This is the train→runtime bridge. It is the sibling of
//! `hipfire-quantize/src/bin/dspark_convert.rs` (which produces the SAME sidecar
//! FROM a HuggingFace `Qwen3DSparkModel` safetensors export) — this example
//! sources the trained weights from the `DSCK` checkpoint instead, and pulls the
//! FROZEN shared `embed_tokens` + `lm_head` from the dense target `.hfq`.
//!
//! It reuses dspark_convert's EXACT sidecar contract verbatim (the runtime loader
//! `hipfire_arch_llama::dspark_body::load_qwen3_dspark` is strict):
//!   - `arch_id = 22` (`hipfire_arch_api::ARCH_ID_DSPARK_DRAFT`).
//!   - the `"dspark"` metadata JSON keys read by
//!     `DsparkConfig::from_metadata_json`.
//!   - the flat sidecar tensor names.
//!   - the quant recipe: body attn/mlp matmuls → Q8F16 (quant_type 3), everything
//!     else (norms, embed, main_proj/main_norm, markov + confidence heads,
//!     lm_head) → F16 (quant_type 1). `--all-f16` forces the body matmuls to F16.
//!
//! ## Why a hipfire-train example (not a hipfire-quantize bin)
//! The `DSCK` reader + drafter geometry live in `hipfire-train`, and the frozen
//! target `embed_tokens`/`lm_head` must be pulled from the target `.hfq` — which
//! `hipfire-train` can parse on the HOST via `hfq_patch::parse_hfq` +
//! `hipfire_runtime::quant::dequant_q8f16`. `hipfire-quantize` does not depend on
//! `hipfire-runtime`, so it cannot read a runtime `.hfq`. Everything here is
//! pure-host (no GPU), matching dspark_convert.
//!
//! Usage:
//!   cargo run --release -p hipfire-train --example dspark_export -- \
//!     --ckpt drafter.dsck --target Qwen3-8B.oq4.hfq \
//!     --block 7 --target-layers 1,7,13,19,25 --markov-rank 256 \
//!     --out Qwen3-8B-oq4.dspark.hfq

#![allow(clippy::manual_div_ceil, clippy::needless_range_loop)]

use hipfire_primitives::conv::{f16_to_f32, f32_slice_to_f16_bytes, f32_to_f16};
use hipfire_train::hfq_patch::parse_hfq;
use std::io::Write;
use std::path::Path;

/// arch_id for the DSpark drafter sidecar. Mirrors
/// `hipfire_arch_api::ARCH_ID_DSPARK_DRAFT` (22); hardcoded so this example does
/// not pull the capability-layer crate as a new dep.
const ARCH_ID_DSPARK_DRAFT: u32 = 22;

// ─── HFQ writer (schema copied verbatim from dspark_convert.rs) ──────────────
//
// Byte-identical to `hipfire-quantize/src/bin/dspark_convert.rs::write_hfq`, so
// the runtime loader accepts this sidecar unchanged. The converter family
// (dflash_convert / dspark_convert) already mirrors this writer between bins;
// this example follows the same convention rather than diverging the format.

const HFQ_MAGIC: &[u8; 4] = b"HFQM";
const HFQ_VERSION: u32 = 1;

#[repr(u8)]
#[derive(Clone, Copy)]
enum QuantType {
    F16 = 1,
    /// Group-32 int8 + F16 scale. Runtime `sidecar_weight` → `DType::Q8_0`.
    Q8F16 = 3,
}

struct HfqTensor {
    name: String,
    quant_type: QuantType,
    shape: Vec<u32>,
    group_size: u32,
    data: Vec<u8>,
}

fn write_hfq(
    path: &Path,
    arch: u32,
    metadata_json: &str,
    tensors: &[HfqTensor],
) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    let metadata_bytes = metadata_json.as_bytes();

    let header_size = 32u64;
    let metadata_offset = header_size;
    let metadata_size = metadata_bytes.len() as u64;

    let index_offset = metadata_offset + metadata_size;
    let mut index_bytes = Vec::new();
    index_bytes.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for t in tensors {
        let name_bytes = t.name.as_bytes();
        index_bytes.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        index_bytes.extend_from_slice(name_bytes);
        index_bytes.push(t.quant_type as u8);
        index_bytes.push(t.shape.len() as u8);
        for &d in &t.shape {
            index_bytes.extend_from_slice(&d.to_le_bytes());
        }
        index_bytes.extend_from_slice(&t.group_size.to_le_bytes());
        index_bytes.extend_from_slice(&(t.data.len() as u64).to_le_bytes());
    }

    let data_start_unaligned = index_offset + index_bytes.len() as u64;
    let data_offset = (data_start_unaligned + 4095) & !4095;

    f.write_all(HFQ_MAGIC)?;
    f.write_all(&HFQ_VERSION.to_le_bytes())?;
    f.write_all(&arch.to_le_bytes())?;
    f.write_all(&(tensors.len() as u32).to_le_bytes())?;
    f.write_all(&metadata_offset.to_le_bytes())?;
    f.write_all(&data_offset.to_le_bytes())?;

    f.write_all(metadata_bytes)?;
    f.write_all(&index_bytes)?;

    let pad_size = (data_offset - data_start_unaligned) as usize;
    f.write_all(&vec![0u8; pad_size])?;

    for t in tensors {
        f.write_all(&t.data)?;
    }
    Ok(())
}

/// Group-of-32 symmetric int8 with an F16 per-group scale (34 bytes/group).
/// quant_type 3 → the runtime maps it to `DType::Q8_0` in `sidecar_weight`.
/// Lifted verbatim from dspark_convert.rs (== the quantizer's `quantize_q8f16`).
fn quantize_q8f16(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 32;
    let block_bytes = 34;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];

        let max_abs = group.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = max_abs / 127.0;
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());

        for i in 0..32 {
            let val = if start + i < end { group[i] } else { 0.0 };
            let q = (val * inv_scale).round().max(-128.0).min(127.0) as i8;
            output[out_off + 2 + i] = q as u8;
        }
    }
    output
}

/// True for the drafter-body 2D matmul weights that get Q8F16: attention
/// q/k/v/o projections and the MLP gate/up/down projections. Everything else
/// stays F16. Operates on the SIDECAR name (post-mapping). Mirrors
/// dspark_convert.rs::is_dspark_matmul_weight.
fn is_dspark_matmul_weight(name: &str) -> bool {
    let is_attn = name.contains("self_attn.")
        && (name.ends_with("q_proj.weight")
            || name.ends_with("k_proj.weight")
            || name.ends_with("v_proj.weight")
            || name.ends_with("o_proj.weight"));
    let is_mlp = name.contains("mlp.")
        && (name.ends_with("gate_proj.weight")
            || name.ends_with("up_proj.weight")
            || name.ends_with("down_proj.weight"));
    is_attn || is_mlp
}

/// Encode one already-f32 tensor into an `HfqTensor` under the shared recipe:
/// body matmul → Q8F16 (unless `all_f16`), everything else → F16.
fn encode_tensor(name: String, shape: Vec<u32>, f32_data: &[f32], all_f16: bool) -> HfqTensor {
    let n_elements = f32_data.len();
    if !all_f16 && is_dspark_matmul_weight(&name) && n_elements >= 32 {
        HfqTensor {
            name,
            quant_type: QuantType::Q8F16,
            shape,
            group_size: 32,
            data: quantize_q8f16(f32_data),
        }
    } else {
        HfqTensor {
            name,
            quant_type: QuantType::F16,
            shape,
            group_size: 0,
            data: f32_slice_to_f16_bytes(f32_data),
        }
    }
}

// ─── DSCK checkpoint reader (host, no GPU) ──────────────────────────────────
//
// Format from `hipfire_train::dspark_train::save_dspark_ckpt`:
//   "DSCK" | u32 version | u32 epoch | u32 np |
//   np × { u32 len, len × f32-le }   (weights, in DsparkFullWeights::params order)
//   i32 t | np × wvec (AdamW m) | np × wvec (AdamW v)
// We only need the weight vectors (the optimizer moments are ignored here).

fn read_u32(b: &[u8], p: &mut usize) -> u32 {
    let v = u32::from_le_bytes(b[*p..*p + 4].try_into().unwrap());
    *p += 4;
    v
}

fn read_dsck_weights(path: &Path) -> Result<(u32, Vec<Vec<f32>>), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() < 16 || &bytes[0..4] != b"DSCK" {
        return Err(format!("{}: bad DSCK magic", path.display()));
    }
    let mut p = 4usize;
    let _version = read_u32(&bytes, &mut p);
    let epoch = read_u32(&bytes, &mut p);
    let np = read_u32(&bytes, &mut p) as usize;

    let mut weights = Vec::with_capacity(np);
    for i in 0..np {
        if p + 4 > bytes.len() {
            return Err(format!("DSCK truncated at param {i} length"));
        }
        let len = read_u32(&bytes, &mut p) as usize;
        let need = len * 4;
        if p + need > bytes.len() {
            return Err(format!("DSCK truncated at param {i} data ({len} f32)"));
        }
        let v: Vec<f32> = bytes[p..p + need]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        p += need;
        weights.push(v);
    }
    Ok((epoch, weights))
}

// ─── Target .hfq reader (host, no GPU): frozen embed_tokens + lm_head ────────

/// Decode one HFQM tensor payload to f32 on the host. Handles the dense-target
/// quant types the trainer's own `load_llama_from_hfq` accepts (F16/F32/BF16/
/// Q8F16); anything else is an explicit error.
fn decode_hfq_payload(quant_type: u8, data: &[u8], n: usize) -> Result<Vec<f32>, String> {
    match quant_type {
        1 => Ok(data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        2 => Ok(data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        16 => Ok(data
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect()),
        3 => Ok(hipfire_runtime::quant::dequant_q8f16(data, n)),
        q => Err(format!(
            "target .hfq: unsupported quant_type {q} for a frozen embed/lm_head \
             (expected F16/F32/BF16/Q8F16)"
        )),
    }
}

/// A frozen tensor pulled from the target: its `[vocab, dim]` shape + host f32.
struct Frozen {
    shape: Vec<u32>,
    data: Vec<f32>,
}

/// Read `embed_tokens` + `lm_head` from the dense target `.hfq`. Names follow the
/// dense-model convention (`model.embed_tokens.weight`, `lm_head.weight`). If the
/// target ties embeddings (no `lm_head.weight`), embed_tokens is reused for both.
fn read_frozen_target(path: &Path) -> Result<(Frozen, Frozen), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let (entries, _meta) = parse_hfq(&bytes)?;

    let fetch = |name: &str| -> Option<Frozen> {
        let e = entries.iter().find(|e| e.name == name)?;
        let n: usize = e.shape.iter().map(|&s| s as usize).product();
        let raw = &bytes[e.data_offset..e.data_offset + e.data_size];
        let data = decode_hfq_payload(e.quant_type, raw, n)
            .unwrap_or_else(|err| panic!("target {name}: {err}"));
        Some(Frozen {
            shape: e.shape.clone(),
            data,
        })
    };

    let embed = fetch("model.embed_tokens.weight")
        .ok_or_else(|| "target .hfq: model.embed_tokens.weight missing".to_string())?;
    let lm_head = match fetch("lm_head.weight") {
        Some(t) => t,
        None => {
            eprintln!("  target: lm_head.weight absent (tied) — reusing embed_tokens as lm_head");
            Frozen {
                shape: embed.shape.clone(),
                data: embed.data.clone(),
            }
        }
    };
    Ok((embed, lm_head))
}

// ─── CLI ────────────────────────────────────────────────────────────────────

struct Args {
    ckpt: String,
    target: String,
    out: String,
    block: usize,
    target_layers: Vec<u64>,
    markov_rank: usize,
    noise_token_id: u32,
    norm_eps: f64,
    all_f16: bool,
}

fn parse_args() -> Args {
    let a: Vec<String> = std::env::args().collect();
    let (mut ckpt, mut target, mut out) = (None, None, None);
    let (mut block, mut markov_rank) = (7usize, 256usize);
    let mut target_layers: Vec<u64> = vec![1, 7, 13, 19, 25];
    let mut noise_token_id = 151669u32;
    let mut norm_eps = 1e-6f64;
    let mut all_f16 = false;
    let mut i = 1;
    while i < a.len() {
        match a[i].as_str() {
            "--ckpt" => {
                ckpt = Some(a[i + 1].clone());
                i += 2;
            }
            "--target" => {
                target = Some(a[i + 1].clone());
                i += 2;
            }
            "--out" | "-o" => {
                out = Some(a[i + 1].clone());
                i += 2;
            }
            "--block" => {
                block = a[i + 1].parse().expect("--block usize");
                i += 2;
            }
            "--target-layers" => {
                target_layers = a[i + 1]
                    .split(',')
                    .map(|s| s.trim().parse().expect("--target-layers CSV of u64"))
                    .collect();
                i += 2;
            }
            "--markov-rank" => {
                markov_rank = a[i + 1].parse().expect("--markov-rank usize");
                i += 2;
            }
            "--noise-token-id" => {
                noise_token_id = a[i + 1].parse().expect("--noise-token-id u32");
                i += 2;
            }
            "--norm-eps" => {
                norm_eps = a[i + 1].parse().expect("--norm-eps f64");
                i += 2;
            }
            "--all-f16" => {
                all_f16 = true;
                i += 1;
            }
            "-h" | "--help" => {
                eprintln!(
                    "Usage: dspark_export --ckpt <DSCK> --target <hfq> --out <out>.dspark.hfq \\\n\
                     \x20 [--block 7] [--target-layers 1,7,13,19,25] [--markov-rank 256] \\\n\
                     \x20 [--noise-token-id 151669] [--norm-eps 1e-6] [--all-f16]"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    Args {
        ckpt: ckpt.expect("--ckpt required"),
        target: target.expect("--target required"),
        out: out.expect("--out required"),
        block,
        target_layers,
        markov_rank,
        noise_token_id,
        norm_eps,
        all_f16,
    }
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();
    let n_targets = args.target_layers.len();
    let rank = args.markov_rank;

    eprintln!("dspark_export (T5b)");
    eprintln!("  ckpt  : {}", args.ckpt);
    eprintln!("  target: {}", args.target);
    eprintln!("  out   : {}", args.out);
    eprintln!(
        "  dspark: block={} target_layers={:?} markov_rank={} noise_token_id={} norm_eps={:e}",
        args.block, args.target_layers, rank, args.noise_token_id, args.norm_eps
    );
    eprintln!(
        "  dtype : {}",
        if args.all_f16 {
            "F16 (all tensors)"
        } else {
            "Q8F16 (body matmuls), F16 (norms/globals/embed/lm_head)"
        }
    );

    // ── 1. Frozen embed_tokens + lm_head from the dense target ──────────────
    let (embed, lm_head) =
        read_frozen_target(Path::new(&args.target)).unwrap_or_else(|e| panic!("read target: {e}"));
    if embed.shape.len() != 2 {
        panic!("target embed_tokens shape {:?} not 2D", embed.shape);
    }
    let vocab = embed.shape[0] as usize;
    let h = embed.shape[1] as usize; // hidden dim, shared drafter/target
    eprintln!("  target geometry: vocab={vocab} dim(h)={h}");

    // ── 2. Trained drafter weights from the DSCK checkpoint ─────────────────
    let (epoch, w) =
        read_dsck_weights(Path::new(&args.ckpt)).unwrap_or_else(|e| panic!("read ckpt: {e}"));
    eprintln!("  ckpt: epoch={epoch} params={}", w.len());

    // DsparkFullWeights::params() layout (fixed order):
    //   [0] fc(main_proj), [1] hidden_norm(main_norm),
    //   n_layers × [wq,wk,wv,wo,wgate,wup,wdown,input_ln,post_ln,q_norm,k_norm],
    //   [.] out_norm(norm),
    //   [.] markov_w1, markov_w2, confidence_proj, confidence_bias.
    // → count = 2 + n_layers*11 + 1 + 4.
    const PER_LAYER: usize = 11;
    if w.len() < 7 || (w.len() - 7) % PER_LAYER != 0 {
        panic!(
            "DSCK param count {} inconsistent with 2 + n_layers*11 + 1 + 4",
            w.len()
        );
    }
    let n_layers = (w.len() - 7) / PER_LAYER;

    // Sanity: fc = [h, n_targets*h]; hidden_norm = [h].
    assert_eq!(w[0].len(), h * n_targets * h, "fc len != h*n_targets*h");
    assert_eq!(w[1].len(), h, "hidden_norm len != h");

    // Derive the drafter attention/MLP geometry from layer-0 vector lengths.
    let l0 = 2usize; // base of layer 0
    let head_dim = w[l0 + 9].len(); // q_norm = [head_dim]
    let q_dim = w[l0].len() / h; // wq = [q_dim, h]
    let kv_dim = w[l0 + 1].len() / h; // wk = [kv_dim, h]
    let inter = w[l0 + 4].len() / h; // wgate = [inter, h]
    eprintln!(
        "  drafter geometry: n_layers={n_layers} head_dim={head_dim} \
         q_dim={q_dim} kv_dim={kv_dim} inter={inter}"
    );

    let shp = |dims: &[usize]| -> Vec<u32> { dims.iter().map(|&d| d as u32).collect() };

    let mut tensors: Vec<HfqTensor> = Vec::new();
    let mut push = |name: &str, shape: Vec<u32>, data: &[f32]| {
        let expect: usize = shape.iter().map(|&s| s as usize).product();
        assert_eq!(
            data.len(),
            expect,
            "{name}: {} elems != shape {:?}",
            data.len(),
            shape
        );
        tensors.push(encode_tensor(name.to_string(), shape, data, args.all_f16));
    };

    // Body globals (F16).
    push("main_proj.weight", shp(&[h, n_targets * h]), &w[0]);
    push("main_norm.weight", shp(&[h]), &w[1]);

    // Per-layer body. Matmuls → Q8F16, norms → F16.
    for li in 0..n_layers {
        let b = 2 + li * PER_LAYER;
        let p = format!("layers.{li}");
        push(
            &format!("{p}.self_attn.q_proj.weight"),
            shp(&[q_dim, h]),
            &w[b],
        );
        push(
            &format!("{p}.self_attn.k_proj.weight"),
            shp(&[kv_dim, h]),
            &w[b + 1],
        );
        push(
            &format!("{p}.self_attn.v_proj.weight"),
            shp(&[kv_dim, h]),
            &w[b + 2],
        );
        push(
            &format!("{p}.self_attn.o_proj.weight"),
            shp(&[h, q_dim]),
            &w[b + 3],
        );
        push(
            &format!("{p}.mlp.gate_proj.weight"),
            shp(&[inter, h]),
            &w[b + 4],
        );
        push(
            &format!("{p}.mlp.up_proj.weight"),
            shp(&[inter, h]),
            &w[b + 5],
        );
        push(
            &format!("{p}.mlp.down_proj.weight"),
            shp(&[h, inter]),
            &w[b + 6],
        );
        push(&format!("{p}.input_layernorm.weight"), shp(&[h]), &w[b + 7]);
        push(
            &format!("{p}.post_attention_layernorm.weight"),
            shp(&[h]),
            &w[b + 8],
        );
        push(
            &format!("{p}.self_attn.q_norm.weight"),
            shp(&[head_dim]),
            &w[b + 9],
        );
        push(
            &format!("{p}.self_attn.k_norm.weight"),
            shp(&[head_dim]),
            &w[b + 10],
        );
    }

    // Final norm + heads.
    let tail = 2 + n_layers * PER_LAYER;
    push("norm.weight", shp(&[h]), &w[tail]);
    push(
        "markov_head.markov_w1.weight",
        shp(&[vocab, rank]),
        &w[tail + 1],
    );
    push(
        "markov_head.markov_w2.weight",
        shp(&[vocab, rank]),
        &w[tail + 2],
    );
    push(
        "confidence_head.proj.weight",
        shp(&[1, h + rank]),
        &w[tail + 3],
    );
    push("confidence_head.proj.bias", shp(&[1]), &w[tail + 4]);

    // Frozen shared embed + lm_head from the target (F16).
    push("embed_tokens.weight", shp(&[vocab, h]), &embed.data);
    push("lm_head.weight", shp(&[vocab, h]), &lm_head.data);

    // ── 3. Metadata JSON — keys read by DsparkConfig::from_metadata_json ────
    // Identical schema to dspark_convert.rs.
    let metadata = serde_json::json!({
        "architecture": "qwen3",
        "config": {
            "dspark_block_size": args.block,
            "dspark_target_layer_ids": args.target_layers,
            "dspark_markov_rank": rank,
            "dspark_noise_token_id": args.noise_token_id,
            "dspark_enable_confidence": true,
            "dspark_confidence_uses_normed": true,
            "norm_eps": args.norm_eps,
        },
    });
    let metadata_json = serde_json::to_string(&metadata).unwrap();

    // ── 4. Write the sidecar ────────────────────────────────────────────────
    let out_path = Path::new(&args.out);
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("mkdir -p output parent");
        }
    }
    let q8 = tensors
        .iter()
        .filter(|t| matches!(t.quant_type, QuantType::Q8F16))
        .count();
    write_hfq(out_path, ARCH_ID_DSPARK_DRAFT, &metadata_json, &tensors)
        .unwrap_or_else(|e| panic!("write_hfq: {e}"));

    let file_size = std::fs::metadata(out_path).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "  wrote: {} ({} tensors, {q8} Q8F16, {:.1} MB, arch_id={ARCH_ID_DSPARK_DRAFT})",
        out_path.display(),
        tensors.len(),
        file_size as f64 / 1e6
    );
}
