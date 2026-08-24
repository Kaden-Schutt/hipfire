// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.


#![allow(dead_code, unused_imports, unused_variables, non_snake_case, clippy::all)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs::File;
use std::io::Write;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use clap::Parser;
use hipfire_quantize::float16::{bf16_to_f32, f16_to_f32, f32_to_f16};
use hipfire_quantize::safetensors_file::{SafetensorsFile, TensorMeta};
use hipfire_quantize::hessian_io;
use crate::e8;
use crate::e8_gptq;
use crate::gguf_input;
use crate::pipeline_gguf::GgufFormat;
use crate::reap_overlay;

// ─── HFQ File Format ────────────────────────────────────────────────────────

pub(crate) const HFQ_MAGIC: &[u8; 4] = b"HFQM";
pub(crate) const HFQ_VERSION: u32 = 1;

impl QuantType {
    /// Reconstruct a `QuantType` from its serialized HFQ byte.
    ///
    /// Needed to copy tensors through an HFQ->HFQ rewrite byte-for-byte
    /// (`build_deepseek4_dspark_e8soa_sidecar`) without knowing each tensor's
    /// tier statically. Generated from the `#[repr(u8)]` discriminants; keep in
    /// sync when adding a variant.
    pub(crate) fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Q4F16G64),
            1 => Some(Self::F16),
            2 => Some(Self::F32),
            3 => Some(Self::Q8F16),
            4 => Some(Self::Q4K),
            5 => Some(Self::Q8HFQ),
            6 => Some(Self::HFQ4G256),
            7 => Some(Self::HFQ4G128),
            8 => Some(Self::HFQ6G256),
            9 => Some(Self::HFQ2G256),
            10 => Some(Self::HFQ2G128),
            11 => Some(Self::HFQ3G256),
            12 => Some(Self::HFQ3G128),
            13 => Some(Self::MQ4G256),
            14 => Some(Self::MQ8G256),
            15 => Some(Self::MQ6G256),
            16 => Some(Self::BF16),
            17 => Some(Self::MQ3G256),
            18 => Some(Self::MQ2G256),
            19 => Some(Self::MQ2G256Lloyd),
            20 => Some(Self::MQ3G256Lloyd),
            21 => Some(Self::HFP4G32),
            24 => Some(Self::MFP4G32),
            22 => Some(Self::TidI32),
            28 => Some(Self::PARO4G128),
            29 => Some(Self::PARO4G128T),
            30 => Some(Self::MQ4G256Lloyd),
            31 => Some(Self::MQ5G256),
            32 => Some(Self::MFP4G32Lloyd),
            33 => Some(Self::MFP4G32P),
            34 => Some(Self::MFP4G32E8),
            35 => Some(Self::MFP4G32E8SOA),
            36 => Some(Self::MFP3G32E8),
            37 => Some(Self::MFP2G32E8),
            38 => Some(Self::MQ2G256GL),
            39 => Some(Self::MQ3G256GL),
            _ => None,
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QuantType {
    Q4F16G64 = 0,
    F16 = 1,
    F32 = 2,
    Q8F16 = 3,
    Q4K = 4,
    Q8HFQ = 5,
    HFQ4G256 = 6,
    HFQ4G128 = 7,
    HFQ6G256 = 8,
    HFQ2G256 = 9,
    HFQ2G128 = 10,
    HFQ3G256 = 11,
    HFQ3G128 = 12,
    MQ4G256 = 13,      // MagnumQuant: FWHT-rotated HFQ4-G256
    MQ8G256 = 14,      // MagnumQuant: FWHT-rotated symmetric INT8, dp4a target
    MQ6G256 = 15,      // MagnumQuant: FWHT-rotated HFQ6-G256 (6-bit, 200 B/group)
    BF16 = 16,         // Original BF16 weights (zero precision loss for vision)
    MQ3G256 = 17,      // MagnumQuant: FWHT-rotated HFQ3-G256 (3-bit, 104 B/group)
    MQ2G256 = 18,      // MagnumQuant: FWHT-rotated HFQ2-G256 (2-bit, 72 B/group)
    MQ2G256Lloyd = 19, // MagnumQuant 2-bit + per-block Lloyd-Max 4-entry fp16 codebook (72 B/group)
    MQ3G256Lloyd = 20, // MagnumQuant 3-bit + per-block Lloyd-Max 8-entry fp16 codebook (112 B/group)
    // HFP4 family — RDNA-optimal FP4 (E2M1 elements + UE8M0 block scale + FP16 row scale).
    // See docs/quant-formats/hfp4.md for byte layout, dequant, rotation modes.
    // Per-row header is 16 B; per-block payload is (1 + g/2) bytes (UE8M0 + nibbles).
    HFP4G32 = 21, // E2M1 + UE8M0 g32 + FP16 row scale — canonical (FP8-WMMA-K aligned)
    // MFP4G32 = HFP4G32 + offline FWHT rotation (256-element FWHT applied to weights at quant time;
    // runtime applies the same FWHT to x via mq_rotate_x). format_flags bit 0 + bits 2-3 = 0b0101
    // signals "rotation present, offline FWHT" for future interop/detection.
    MFP4G32 = 24, // v1.5 — HFP4G32 + offline FWHT (drop-in MQ4 replacement)
    /// I64→U32 downcast of DeepSeek V4 hash-routing `tid2eid` lookup tables.
    /// Shape `[vocab, num_experts_per_tok]`. Stored as raw u32 LE; the
    /// loader reads `bytes.chunks_exact(4)`. ID 22 was reserved for the
    /// HFP4G16 NV-aligned ablation (never built) — we re-use the slot
    /// for tid2eid storage to stay byte-compatible with antirezQ8.hfq.
    TidI32 = 22,
    // Reserved IDs — DO NOT REUSE for unrelated formats. Documented in docs/quant-formats/hfp4.md.
    // HFP4G16     = 22, // v1.5 — NV-aligned FP16-WMMA-K alignment ablation (re-used by TidI32)
    // HFP4G64     = 23, // v1.5 — RDNA1/2 sweet-spot ablation
    // HFP4G32MX   = 25, // v2  — strict OCP MXFP4 interop alias (no row scale, UE8M0 only)
    // HFP4G16NV   = 26, // v2  — strict NVFP4 interop alias (E4M3 scale + FP32 tensor)
    // HFP8E4M3G32 = 27, // v2  — HFP8 E4M3 family
    PARO4G128 = 28,  // ParoQuant native AWQ W4 + pairwise activation rotation metadata
    PARO4G128T = 29, // ParoQuant engine-tiled qweight [M/8, K] for coalesced GEMV reads
    // MFP4G32R    = 29, // v3  — HFP4G32 + online block-diag-128 rotation (AMD recipe)
    // HFP8E5M2G32 = 30, // v2  — HFP8 E5M2 family
    MQ4G256Lloyd = 30, // MagnumQuant 4-bit + per-block Lloyd-Max 16-entry fp16 codebook (160 B/group)
    // Renumbered from 21 → 30 in mq4-lloyd merge to avoid HFP4G32=21 collision.
    // Models quantized pre-renumber MUST be re-quantized.
    MQ5G256 = 31,      // MagnumQuant: FWHT-rotated 5-bit (168 B/group, 5.25 bpw).
    MFP4G32Lloyd = 32, // mfp4 (E2M1+UE8M0 g32+FP16 row scale+offline FWHT) with the fixed
    // E2M1 grid replaced by ONE per-tensor 16-entry fp16 Lloyd codebook
    // prepended (32 B) before row 0. Rows byte-identical to MFP4G32 (qt 24).
    // 8B affine header (f32 scale + f32 min) + 160B payload
    // (5 bits × 256, cross-byte: 8 codes per 5 bytes). NOTE: 16=BF16.
    MFP4G32P = 33, // mfp4+P: mfp4 (E2M1+FP16 row scale+offline FWHT) with the per-32-block
    // UE8M0 scale promoted to E4M3 (FP8, non-power-of-2). Byte layout
    // BYTE-IDENTICAL to MFP4G32 (qt 24): 16-B hdr + n_blocks×17 B, NO prefix.
    // Only the per-block scale byte's meaning differs (E4M3 vs UE8M0).
    MFP4G32E8 = 34, // mfp4-E8: mfp4+P container (E4M3 block scale, NO prefix, same row_bytes)
    // with the 32 E2M1 nibbles replaced by 4x32-bit E8-lattice codewords
    // (8 weights/codeword, QUANT_STEP=0.88). 4.25 bpw, FWHT rotation.
    // MQ*-GL ("global Lloyd"): N-bit codes against ONE tensor-global codebook
    // plus a per-block fp16 scale, in structure-of-arrays layout:
    //     [0 .. M*gpr*P)                  packed N-bit indices, P B/group
    //     [M*gpr*P .. +M*gpr*2)           fp16 per-block scales
    // vs the per-block Lloyd formats (qt 19/20), which interleave a fitted
    // 2^N-entry fp16 codebook into every group.
    //
    // Rationale (measured 2026-08-04, docs/investigations/2026-08-04-a3b-lowbit-quality.md):
    // post-FWHT blocks are Gaussian by CLT, so the optimal LEVEL SHAPE is the
    // same in every block — a per-block fit re-derives it ~4000x per tensor and
    // differs only by scale. Fitting a global codebook on 28.3M real a3b expert
    // weights reproduces the textbook Lloyd-Max Gaussian levels to 3 decimals.
    // Cost on real weights: +2.35% NRMSE / +1.16% end-to-end KLD, for -0.1875 bpw
    // (MQ2) — and the group base becomes naturally aligned (64 B vs 72 B stride).
    MQ2G256GL = 38, // 2-bit + global codebook: 64 B idx/group + 2 B scale = 2.0625 bpw
    MQ3G256GL = 39, // 3-bit + global codebook: 96 B idx/group + 2 B scale = 3.0625 bpw
    MFP4G32E8SOA = 35, // mfp4-E8 SoA: same E8 data as qt=34 but in structure-of-arrays layout.
    // [16B hdr] + [n_blocks B E4M3 scales, pad 16B] + [n_blocks*16B codewords].
    MFP3G32E8 = 36, // mfp3-E8: MFP4G32E8 frame, 3-bit lattice (center 3), 13 B/blk, 3.25 bpw.
    // Drop-in cold tier for MQ3G256Lloyd (tag 3 → tag 5).
    MFP2G32E8 = 37, // mfp2-E8: MFP4G32E8 frame, 2-bit lattice (center 1), 9 B/blk, 2.25 bpw.
                    // Drop-in cold tier for MQ2G256Lloyd (tag 1 → tag 6).
}

/// Per-tensor precision level assigned by the K-map pre-pass.
/// Determines whether a tensor gets the base format, a 6-bit promotion,
/// Q8, or F16. See docs/superpowers/specs/2026-05-08-mixed-quant-kmap-design.md.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum QuantLevel {
    /// Store as F16 (norms, biases, 1D tensors).
    F16,
    /// Store as Q8_F16 (embeddings, lm_head, MoE routers).
    Q8,
    /// Promote to 6-bit variant of the base format (edge layers, MoE expert FFN).
    Promote6,
    /// Override the default for a specific tensor class (today: lm_head)
    /// to a CLI-specified format. Currently unused on this branch (no emission
    /// site); kept so origin/master's lm_head-format override match arms
    /// compile after the merge. Re-wire to `--lm-head-format` when the
    /// configurable-kmap-pair refactor lands here.
    #[allow(dead_code)]
    Override(GgufFormat),
    /// Use the base format as-is.
    Base,
}

/// Default kmap promote target for a given base format. Preserves the
/// pre-`--kmap-promote` behavior byte-for-byte: MQ-family bases promote to
/// MQ6, HFQ-family to HFQ6, FP4-family is a no-op (no FP6 sibling).
pub(crate) fn default_promote_target(base: GgufFormat) -> GgufFormat {
    match base {
        GgufFormat::Mq2
        | GgufFormat::Mq3
        | GgufFormat::Mq4
        | GgufFormat::Mq5
        | GgufFormat::Mq6
        | GgufFormat::Mq2Lloyd
        | GgufFormat::Mq3Lloyd
        | GgufFormat::Mq4Lloyd => GgufFormat::Mq6,
        GgufFormat::Hfq4 | GgufFormat::Hfq6 => GgufFormat::Hfq6,
        GgufFormat::Hfp4 => GgufFormat::Hfp4,
        GgufFormat::Mfp4 => GgufFormat::Mfp4,
        GgufFormat::Mfp4Lloyd => GgufFormat::Mfp4Lloyd,
        GgufFormat::Mfp4P => GgufFormat::Mfp4P,
        GgufFormat::Mfp4E8 => GgufFormat::Mfp4E8,
        GgufFormat::Mfp4E8Soa => GgufFormat::Mfp4E8Soa,
        GgufFormat::Mfp3E8 => GgufFormat::Mfp3E8,
        GgufFormat::Mfp2E8 => GgufFormat::Mfp2E8,
    }
}

/// Allowlist for explicit `--kmap-promote` overrides. Runtime mixed-format
/// dispatch (post-#257) is validated only within same-rotation-family,
/// upward-in-bit-width pairings. Cross-family (MQ↔HFQ, MQ↔HFP) and
/// downward-in-bits promotions are rejected at parse time.
pub(crate) fn is_promote_pair_supported(base: GgufFormat, promote: GgufFormat) -> bool {
    use crate::pipeline_gguf::GgufFormat::*;
    if base == promote {
        return true; // no-op promotion is always safe
    }
    match (base, promote) {
        // Lloyd-to-Lloyd only — Lloyd variants use different codebooks +
        // different runtime kernel families from standard MQ. Lloyd→non-Lloyd
        // mixed-format dispatch has no runtime support today; the plan's
        // "Future expansion" section targets the MQ2-Lloyd + MQ3-Lloyd pair
        // specifically. Tightened per combined-review finding G2.
        (Mq2Lloyd, Mq3Lloyd) => true,
        (Mq2Lloyd | Mq3Lloyd, _) => false,
        (_, Mq2Lloyd | Mq3Lloyd) => false,

        // MQ-family upward bit-width (non-Lloyd)
        (Mq2, Mq3 | Mq4 | Mq5 | Mq6) => true,
        (Mq3, Mq4 | Mq5 | Mq6) => true,
        (Mq4, Mq5 | Mq6) => true,
        (Mq5, Mq6) => true,

        // HFQ-family upward bit-width
        (Hfq4, Hfq6) => true,

        // Everything else: explicitly not in the supported matrix.
        // Cross-family (MQ↔HFQ↔FP4) rejected — runtime mixed-format dispatch
        // (post-#257) is only same-rotation-family-safe.
        _ => false,
    }
}

/// Extract layer index from a tensor name.
/// Handles both safetensors (`layers.{N}.`) and GGUF (`blk.{N}.`) patterns.
/// Uses unanchored search to handle any prefix (model.layers, model.language_model.layers, etc.).
pub(crate) fn parse_layer_idx(name: &str) -> Option<usize> {
    // Vision towers contain `vision_tower.layers.N` — don't treat that as a
    // text layer index (would pick up edge-layer Promote6 for vision). Return
    // None early for vision prefixes additively (no behaviour change for text).
    if name.starts_with("model.vision_tower.")
        || name.starts_with("vision_tower.")
        || name.starts_with("model.vision_adapter.")
        || name.starts_with("model.vision_projection.")
        || name.starts_with("model.visual.")
        || name.starts_with("visual.")
    {
        return None;
    }
    // Try safetensors pattern: "layers.{N}."
    if let Some(pos) = name.find("layers.") {
        let after = &name[pos + 7..]; // skip "layers."
        if let Some(dot) = after.find('.') {
            if let Ok(idx) = after[..dot].parse::<usize>() {
                return Some(idx);
            }
        }
    }
    // Try GGUF pattern: "blk.{N}."
    if let Some(pos) = name.find("blk.") {
        let after = &name[pos + 4..]; // skip "blk."
        if let Some(dot) = after.find('.') {
            if let Ok(idx) = after[..dot].parse::<usize>() {
                return Some(idx);
            }
        }
    }
    None
}

/// Stride for alternating-mode promotion: edge layers always promoted,
/// plus every Nth middle layer. 3 was chosen empirically — promotes ~40%
/// of middle layers, matching llama.cpp Q4_K_M's budget-allocation pattern.
/// On MoE 3.6-35B-A3B: stride=3 gives PPL 8K=19.96 at 21.8 GB vs full
/// K-map PPL 8K=20.07 at 27.7 GB.
pub(crate) const ALTERNATING_STRIDE: usize = 3;

/// llama.cpp-style alternating promotion: edge layers always promoted,
/// middle layers promoted every `stride` layers.
pub(crate) fn is_positional_promote(idx: usize, n_layers: usize, stride: usize) -> bool {
    if n_layers == 0 || stride == 0 {
        return false;
    }
    if idx < 2 || idx >= n_layers.saturating_sub(2) {
        return true;
    }
    (idx - 2) % stride == 0
}

/// Resolve the quantization level for a tensor based on its name, the model's
/// layer count, whether the model is MoE, and the K-map mode.
///
/// `kmap_mode`: 0 = full (all candidates promoted), 1 = alternating
/// (experts + ffn_down every 3rd middle layer, edge layers always),
/// 2 = typed (ffn_down + attn_v everywhere).
///
/// Note: In the safetensors path, norms/biases are filtered by `should_quantize()`
/// before this function is called. Rules 1-2 exist for the GGUF path and completeness.
pub(crate) fn kmap_resolve(name: &str, n_layers: usize, is_moe: bool) -> QuantLevel {
    kmap_resolve_mode(name, n_layers, is_moe, 0)
}

pub(crate) fn kmap_resolve_mode(name: &str, n_layers: usize, is_moe: bool, kmap_mode: u8) -> QuantLevel {
    // Vision tensors (809 on Glimmer) stay F16 and must not be mis-classified
    // as text. This also prevents `vision_tower.layers.N` from being parsed
    // as a text layer index for edge-layer Promote6. Additive: text tensors
    // never match these prefixes, so no existing arch changes.
    if name.starts_with("model.vision_tower.")
        || name.starts_with("vision_tower.")
        || name.starts_with("model.vision_adapter.")
        || name.starts_with("model.vision_projection.")
        || name.starts_with("model.visual.")
        || name.starts_with("visual.")
    {
        return QuantLevel::F16;
    }
    // Rule 1: norms, biases, 1D (GGUF path mainly)
    if name.contains("norm") || name.contains("bias") {
        return QuantLevel::F16;
    }

    // Rule 2: embeddings, lm_head, output projection
    if name.contains("embed_tokens")
        || name.contains("token_embd")
        || name.contains("lm_head")
        || name.ends_with("output.weight")
    {
        return QuantLevel::Q8;
    }

    // Rule 3: MoE routers
    if is_moe
        && (name.ends_with("mlp.gate.weight")
            || name.contains("shared_expert_gate")
            || name.ends_with("router.proj.weight"))
    {
        return QuantLevel::Q8;
    }

    // Rule 4: MoE expert FFN weights
    if is_moe && name.contains("mlp.experts.") {
        if kmap_mode == 1 {
            // Alternating: promote expert groups only in positional layers
            if let Some(idx) = parse_layer_idx(name) {
                if is_positional_promote(idx, n_layers, ALTERNATING_STRIDE) {
                    return QuantLevel::Promote6;
                }
                return QuantLevel::Base;
            }
        }
        return QuantLevel::Promote6;
    }

    // Mode 2 (typed): promote ffn_down and attn_v in all layers.
    // UNCHANGED semantics — every model that already ships with `--kmap-mode
    // typed` must keep producing byte-identical output. Gemma 4's variant of
    // this rule lives in mode 3 below rather than mutating this one.
    if kmap_mode == 2 {
        let is_down = name.contains("down_proj") || name.contains("ffn_down");
        let is_v = name.contains("v_proj") || name.contains("attn_v");
        if is_down || is_v {
            return QuantLevel::Promote6;
        }
        if n_layers > 0 {
            if let Some(idx) = parse_layer_idx(name) {
                if idx < 2 || idx >= n_layers.saturating_sub(2) {
                    return QuantLevel::Promote6;
                }
            }
        }
        return QuantLevel::Base;
    }

    // Mode 3 (typed-gemma4): mode 2, except edge layers promote FFN + v_proj
    // only and leave attn q/k/o at Base — dense attn promotion regresses PPL
    // +3.1% on 27B (see ppl_kmap_20260508.md).
    //
    // This is a SEPARATE mode rather than a tweak to mode 2 so that no model
    // already quantized with `--kmap-mode typed` changes bytes. Selected
    // automatically for gemma4 (arch_id 13); reachable explicitly as
    // `--kmap-mode typed-gemma4`.
    if kmap_mode == 3 {
        let is_down = name.contains("down_proj") || name.contains("ffn_down");
        let is_v = name.contains("v_proj") || name.contains("attn_v");
        if is_down || is_v {
            return QuantLevel::Promote6;
        }
        if n_layers > 0 {
            if let Some(idx) = parse_layer_idx(name) {
                if idx < 2 || idx >= n_layers.saturating_sub(2) {
                    let is_attn_qko = name.contains("q_proj")
                        || name.contains("attn_q")
                        || name.contains("k_proj")
                        || name.contains("attn_k")
                        || name.contains("o_proj")
                        || name.contains("attn_o");
                    if !is_attn_qko {
                        return QuantLevel::Promote6;
                    }
                }
            }
        }
        return QuantLevel::Base;
    }

    // Mode 1 (alternating): ffn_down in edge + every 3rd middle layer.
    // Edge-layer rule mirrors mode 0 below: attn+FFN for MoE (full promotion
    // gives -19.8% PPL on 3.6-35B-A3B), FFN only for dense (attn promotion
    // regresses PPL +3.1% on 27B). Bench: asym4 KV, ctx=8192, wikitext-2-test.
    // See ppl_kmap_20260508.md.
    if kmap_mode == 1 {
        let is_down = name.contains("down_proj") || name.contains("ffn_down");
        if n_layers > 0 {
            if let Some(idx) = parse_layer_idx(name) {
                if is_down && is_positional_promote(idx, n_layers, ALTERNATING_STRIDE) {
                    return QuantLevel::Promote6;
                }
                // Edge layers: attn+FFN for MoE, FFN only for dense.
                if idx < 2 || idx >= n_layers.saturating_sub(2) {
                    if is_moe {
                        return QuantLevel::Promote6;
                    }
                    let is_ffn = name.contains("mlp.") || name.contains("ffn");
                    if is_ffn {
                        return QuantLevel::Promote6;
                    }
                }
            }
        }
        return QuantLevel::Base;
    }

    // Rule 5 (full mode 0): edge layers (first 2 + last 2).
    // Dense models: FFN only — attn promotion regresses PPL (+3.1% on 27B).
    // MoE models: attn+FFN — full promotion gives -19.8% PPL on 3.6-35B-A3B.
    // Bench: asym4 KV, ctx=8192, wikitext-2-test. See ppl_kmap_20260508.md.
    if n_layers > 0 {
        if let Some(idx) = parse_layer_idx(name) {
            if idx < 2 || idx >= n_layers.saturating_sub(2) {
                if is_moe {
                    // MoE: promote all tensors in edge layers (attn + FFN)
                    return QuantLevel::Promote6;
                }
                // Dense: promote FFN only — attn stays at Base
                let is_ffn = name.contains("mlp.") || name.contains("ffn");
                if is_ffn {
                    return QuantLevel::Promote6;
                }
            }
        }
    }

    // Rule 6: everything else
    QuantLevel::Base
}

#[derive(Debug)]
pub(crate) struct HfqTensor {
    pub(crate) name: String,
    pub(crate) quant_type: QuantType,
    pub(crate) shape: Vec<u32>,
    pub(crate) group_size: u32,
    pub(crate) data: Vec<u8>,
    /// When data is spilled to disk, this holds the byte count.
    /// `data` is empty and the bytes live in the spill file.
    pub(crate) spilled_len: u64,
}

/// Streaming tensor spill file. When the quantizer accumulates more than
/// `SPILL_THRESHOLD` bytes of tensor data in memory, it flushes completed
/// tensors to this file. At write_hfq time, spilled data is copied from
/// the spill file instead of from memory, keeping peak RSS bounded.
pub(crate) struct TensorSpill {
    file: std::io::BufWriter<File>,
    path: PathBuf,
    offset: u64,
}

impl TensorSpill {
    pub(crate) fn new(dir: &Path) -> std::io::Result<Self> {
        // PID-unique so concurrent quantize runs in the same output dir don't
        // share a spill path (a sibling run's Drop would otherwise delete this
        // run's spill file → write_hfq NotFound panic).
        let path = dir.join(format!(".hipfire_quant_spill.{}.tmp", std::process::id()));
        let file = std::io::BufWriter::with_capacity(4 * 1024 * 1024, File::create(&path)?);
        Ok(Self {
            file,
            path,
            offset: 0,
        })
    }

    /// Write tensor data to the spill file. Returns the byte count written.
    pub(crate) fn spill(&mut self, data: &[u8]) -> std::io::Result<u64> {
        use std::io::Write;
        self.file.write_all(data)?;
        self.offset += data.len() as u64;
        Ok(data.len() as u64)
    }

    pub(crate) fn flush(&mut self) -> std::io::Result<()> {
        use std::io::Write;
        self.file.flush()
    }

    pub(crate) fn cleanup(self) {
        // Explicit cleanup — Drop impl handles the actual removal.
        drop(self);
    }
}

impl Drop for TensorSpill {
    fn drop(&mut self) {
        // Ensure the temp file is removed even on panic.
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Spill tensors whose data is in memory to the spill file, freeing RAM.
/// Called after each layer's expert batch to keep peak RSS bounded.
pub(crate) fn maybe_spill(tensors: &mut [HfqTensor], spill: &mut TensorSpill, threshold: usize) {
    let in_mem: usize = tensors
        .iter()
        .filter(|t| t.spilled_len == 0)
        .map(|t| t.data.len())
        .sum();
    if in_mem < threshold {
        return;
    }
    for t in tensors.iter_mut() {
        if t.spilled_len == 0 && !t.data.is_empty() {
            let len = spill.spill(&t.data).unwrap_or(0);
            t.spilled_len = len;
            t.data = Vec::new(); // free the memory
        }
    }
    let _ = spill.flush();
}

/// The arch-correct config field naming the routed-expert count, as the arch's
/// loader config parser reads it from the HFQ metadata `config` object:
///   * deepseek4 → `n_routed_experts`
///   * qwen3.5-moe / lfm2moe → `num_experts`
///   * minimax → `num_local_experts`
pub(crate) fn reap_expert_count_field(arch: reap_overlay::ReapArch) -> &'static str {
    match arch {
        reap_overlay::ReapArch::Deepseek4 => "n_routed_experts",
        reap_overlay::ReapArch::Qwen35 => "num_experts",
        reap_overlay::ReapArch::Lfm2Moe => "num_experts",
        reap_overlay::ReapArch::Minimax => "num_local_experts",
    }
}

/// Patch the HFQ metadata envelope's `config` so the routed-expert count reads
/// `kept` (the pruned/compact count) for `arch`. The arch loaders parse the
/// inner `config` object (qwen3.5 additionally descends into `config.text_config`
/// when present), so patch the field WHEREVER it currently exists under `config`:
/// at `config[field]` and, if present, at `config.text_config[field]`. Erroring
/// (rather than silently no-op'ing) when the field is absent prevents shipping a
/// baked model whose metadata still claims the original expert count.
pub(crate) fn patch_expert_count_metadata(
    metadata_json: &str,
    arch: reap_overlay::ReapArch,
    kept: usize,
) -> Result<String, String> {
    let field = reap_expert_count_field(arch);
    let mut v: serde_json::Value =
        serde_json::from_str(metadata_json).map_err(|e| format!("metadata not valid JSON: {e}"))?;
    let config = v
        .get_mut("config")
        .ok_or_else(|| "metadata missing `config` object".to_string())?;
    let mut patched = false;
    if config.get(field).is_some() {
        config[field] = serde_json::json!(kept);
        patched = true;
    }
    if let Some(tc) = config.get_mut("text_config") {
        if tc.get(field).is_some() {
            tc[field] = serde_json::json!(kept);
            patched = true;
        }
    }
    if !patched {
        return Err(format!(
            "expert-count field '{field}' not found under config (or config.text_config) for {arch:?}"
        ));
    }
    serde_json::to_string(&v).map_err(|e| format!("re-serialize metadata: {e}"))
}

pub(crate) fn write_hfq(
    path: &Path,
    arch: u32,
    metadata_json: &str,
    tensors: &[HfqTensor],
    spill: Option<&mut TensorSpill>,
) -> std::io::Result<()> {
    let mut f = File::create(path)?;

    let metadata_bytes = metadata_json.as_bytes();

    // Calculate offsets
    let header_size = 32u64;
    let metadata_offset = header_size;
    let metadata_size = metadata_bytes.len() as u64;

    // Tensor index follows metadata
    let index_offset = metadata_offset + metadata_size;
    let mut index_bytes = Vec::new();
    // Write tensor count
    index_bytes.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for t in tensors {
        // name length + name
        let name_bytes = t.name.as_bytes();
        index_bytes.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        index_bytes.extend_from_slice(name_bytes);
        // quant type
        index_bytes.push(t.quant_type as u8);
        // n_dims + shape
        index_bytes.push(t.shape.len() as u8);
        for &d in &t.shape {
            index_bytes.extend_from_slice(&d.to_le_bytes());
        }
        // group size
        index_bytes.extend_from_slice(&t.group_size.to_le_bytes());
        // data size (offset computed at read time from cumulative sizes)
        let data_len = if t.spilled_len > 0 {
            t.spilled_len
        } else {
            t.data.len() as u64
        };
        index_bytes.extend_from_slice(&data_len.to_le_bytes());
    }

    // Data starts after index, aligned to 4096
    let data_start_unaligned = index_offset + index_bytes.len() as u64;
    let data_offset = (data_start_unaligned + 4095) & !4095;

    // Write header (32 bytes)
    f.write_all(HFQ_MAGIC)?;
    f.write_all(&HFQ_VERSION.to_le_bytes())?;
    f.write_all(&arch.to_le_bytes())?;
    f.write_all(&(tensors.len() as u32).to_le_bytes())?;
    f.write_all(&metadata_offset.to_le_bytes())?;
    f.write_all(&data_offset.to_le_bytes())?;

    // Write metadata
    f.write_all(metadata_bytes)?;

    // Write tensor index
    f.write_all(&index_bytes)?;

    // Pad to data alignment
    let pad_size = (data_offset - data_start_unaligned) as usize;
    f.write_all(&vec![0u8; pad_size])?;

    // Write tensor data — from spill file or from memory
    if let Some(spill) = spill {
        let _ = spill.flush();
        let mut spill_reader = std::io::BufReader::new(File::open(&spill.path)?);
        let mut buf = vec![0u8; 4 * 1024 * 1024]; // 4 MB copy buffer
        for t in tensors {
            if t.spilled_len > 0 {
                // Copy from spill file
                let mut remaining = t.spilled_len as usize;
                while remaining > 0 {
                    let chunk = remaining.min(buf.len());
                    use std::io::Read;
                    spill_reader.read_exact(&mut buf[..chunk])?;
                    f.write_all(&buf[..chunk])?;
                    remaining -= chunk;
                }
            } else {
                f.write_all(&t.data)?;
            }
        }
    } else {
        for t in tensors {
            f.write_all(&t.data)?;
        }
    }

    Ok(())
}
