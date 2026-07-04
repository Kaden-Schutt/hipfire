// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Shared batched-prefill **composition seam** for dense and hybrid arches.
//!
//! # Why this module exists
//!
//! Kernel *dispatch* is already centralized in `hipfire_dispatch` (the
//! gemv/gemm/fused_qkv/moe/attention families, re-exported via
//! [`crate::dispatch`]). What was NOT shared is the layer-body *composition*:
//! the glue that sequences `embed → rmsnorm → qkv → attn → o → ffn` over a
//! `[seq, hidden]` batch. Every arch re-implemented it:
//!   - `crate::llama::forward_prefill_chunk` (dense LLaMA / qwen2),
//!   - `hipfire_arch_zaya::gpu::gpu_forward_serve` (hand-rolled `zaya_*` kernels),
//!   - `hipfire_arch_lfm2moe::forward::prefill_batch`,
//!   - the qwen35 hybrid monolith.
//!
//! That duplication is the seam this module owns. The contract is two-tier:
//!
//! 1. **Generic batched primitives + predicates (here).** Arch-agnostic helpers
//!    over a `[seq, hidden]` batch and the eligibility predicate that decides
//!    whether the batched path is even available for a model's quant dtypes on
//!    the running arch. The per-(dtype × arch) coverage decision lives in ONE
//!    place ([`crate::dispatch::is_batchable_la`]) instead of being re-derived
//!    per arch.
//!
//! 2. **Per-arch mixer composition (in the arch crate).** Each arch builds the
//!    `[seq, hidden]` batch and runs a layer loop; full-attention layers reuse
//!    the shared attention path, while a mixer kernel that is genuinely
//!    arch-specific (zaya CCA conv + delayed-value, lfm2 LIV short-conv, qwen35
//!    DeltaNet) stays in the arch crate and *composes* the shared FFN/attn
//!    primitives.
//!
//! The object-safe serving boundary above this (`SimpleAr` / `ServingBackend` /
//! `run_simple_ar` in [`crate::arch`]) is unchanged — this module fills in what
//! sits *below* it.

use crate::kv::KvCache;
use crate::llama::LlamaWeights;
use hipfire_rdna::DType;

pub use crate::dispatch::is_batchable_la;

/// Minimum prompt length below which the batched prefill path is not worth its
/// fixed setup cost; shorter prompts run the per-token decode loop. Shared by
/// every arch's prefill-eligibility check so the threshold is defined once.
pub const MIN_BATCH: usize = 4;

/// KV-cache quantization regime, as an axis value for prefill-batchability
/// reasoning. `Asym` collapses the rotated asym{2,3,4} K modes (they share the
/// same batched flash-masked kernel family). This is the GPU-free axis enum the
/// model-support generator iterates over; the live `KvCache` maps onto it via
/// [`kv_cache_prefill_mode`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvPrefillMode {
    /// Unquantized fp32 KV — no batched flash-masked prefill kernel.
    Fp32,
    /// Q8_0 KV.
    Q8,
    /// Rotated asymmetric K (givens2/givens3/planar4) + Q8 V.
    Asym,
    /// Recurrent / SSM arch with no KV cache at all.
    NoKv,
}

/// Whether a KV mode has a batched flash-masked prefill kernel. Only Q8_0 and
/// the rotated asym K modes do; fp32 and no-kv fall back to per-token decode.
/// The single mechanical predicate behind both the runtime eligibility check
/// and the model-support matrix's kv axis.
pub fn kv_mode_prefill_batchable(m: KvPrefillMode) -> bool {
    matches!(m, KvPrefillMode::Q8 | KvPrefillMode::Asym)
}

/// Map a live `KvCache`'s quant flags onto the [`KvPrefillMode`] axis value.
pub fn kv_cache_prefill_mode(kv: &KvCache) -> KvPrefillMode {
    if kv.quant_q8 {
        KvPrefillMode::Q8
    } else if kv.quant_asym2 || kv.quant_asym3 || kv.quant_asym4 {
        KvPrefillMode::Asym
    } else {
        KvPrefillMode::Fp32
    }
}

/// Whether the KV-cache quantization mode has a batched flash-masked prefill
/// kernel. Centralizes the `kv_cache.quant_q8 || quant_asym2 || quant_asym3 ||
/// quant_asym4` test that was copied into both `forward_prefill_batch` and
/// `forward_prefill_batch_chunk_captured`; now expressed via the shared
/// [`KvPrefillMode`] axis so the matrix generator and the runtime agree.
pub fn kv_quant_batchable(kv: &KvCache) -> bool {
    kv_mode_prefill_batchable(kv_cache_prefill_mode(kv))
}

/// Whether every linear weight in a dense LLaMA-family model has a
/// batched-prefill GEMM kernel on `arch`. Delegates to the canonical
/// per-(dtype × arch) predicate [`is_batchable_la`] for each of the seven
/// linear weights per layer (wq/wk/wv/wo + w_gate/w_up/w_down).
///
/// When this returns `false`, at least one (dtype, arch) combination lacks a
/// batched kernel (e.g. MQ3 / FP4 on a non-WMMA arch) and the caller must use
/// the per-token fallback. This is the single source of the llama "partial"
/// prefill rating: on a WMMA arch (gfx11/gfx12) every shipped quant batches, so
/// the fallback is dead code; on gfx9 CDNA some quants still route per-token.
pub fn llama_weights_batchable(weights: &LlamaWeights, arch: &str) -> bool {
    weights.layers.iter().all(|l| {
        is_batchable_la(l.wq.gpu_dtype, arch)
            && is_batchable_la(l.wk.gpu_dtype, arch)
            && is_batchable_la(l.wv.gpu_dtype, arch)
            && is_batchable_la(l.wo.gpu_dtype, arch)
            && is_batchable_la(l.w_gate.gpu_dtype, arch)
            && is_batchable_la(l.w_up.gpu_dtype, arch)
            && is_batchable_la(l.w_down.gpu_dtype, arch)
    })
}

/// Whether a dense LLaMA-family prefill of `n` tokens can take the batched path:
/// the batched feature is enabled, the prompt is long enough ([`MIN_BATCH`]),
/// the KV mode has a batched kernel, and every linear weight is batchable on
/// `arch`. The single predicate behind both `forward_prefill_batch`'s
/// fall-back-or-batch branch and `forward_prefill_batch_chunk_captured`'s
/// capture-eligibility assertion.
pub fn llama_prefill_batchable(
    weights: &LlamaWeights,
    kv: &KvCache,
    arch: &str,
    n: usize,
    batched_enabled: bool,
) -> bool {
    batched_enabled
        && n >= MIN_BATCH
        && kv_quant_batchable(kv)
        && llama_weights_batchable(weights, arch)
}

/// The set of quant dtypes that always have a batched-prefill GEMM kernel on
/// every arch, for callers that want to reason about coverage without an arch
/// string (e.g. admission docs). WMMA-gated dtypes (MQ3 / FP4) are intentionally
/// excluded — use [`is_batchable_la`] with the concrete arch for those.
pub fn dtype_always_batchable(dt: DType) -> bool {
    matches!(
        dt,
        DType::MQ4G256 | DType::HFQ4G256 | DType::MQ6G256 | DType::HFQ6G256 | DType::Q8_0
    )
}

/// Map a `model-support.toml` quant token to the representative weight [`DType`]
/// whose batched-prefill GEMM availability stands in for that quant.
///
/// `None` means the quant has **no mechanical batched-prefill predicate** and its
/// prefill availability is governed by a quality `[[gate]]` instead:
///   - `bf16` is `None` here but is handled as always-batchable (plain GEMM on
///     every arch) by [`quant_prefill_batchable`];
///   - the OQ W4A4 / W8A8 activation-quant formats (`oq4`, `oq4+`, `oq8`,
///     `oq8+`) route through parity-gated activation paths, not the weight-only
///     `is_batchable_la` GEMM kernels.
fn quant_repr_dtype(quant: &str) -> Option<DType> {
    Some(match quant {
        "q8" => DType::Q8_0,
        "mq4" => DType::MQ4G256,
        "mq6" => DType::MQ6G256,
        "mq3" => DType::MQ3G256,
        _ => return None,
    })
}

/// Whether a quant token's batched-prefill weight GEMM exists on GPU arch `gfx`
/// (a concrete `gfx*` id, e.g. `"gfx1151"`), GPU-free — the gen-time entry point
/// for deriving the model-support prefill matrix from the runtime predicates
/// without the caller depending on `hipfire_rdna::DType`.
///
/// Returns `None` for quant tokens whose prefill availability is a quality-gate
/// decision rather than a kernel predicate (the OQ activation-quant formats); the
/// generator should consult `[[gate]]` for those. `bf16` is `Some(true)`
/// everywhere (unquantized weights use the plain batched GEMM, present on every
/// arch).
pub fn quant_prefill_batchable(quant: &str, gfx: &str) -> Option<bool> {
    if quant == "bf16" {
        return Some(true);
    }
    quant_repr_dtype(quant).map(|dt| is_batchable_la(dt, gfx))
}

/// Whether the dflash / DDTree speculative-decode path is available on GPU arch
/// `gfx` (a concrete `gfx*` id), GPU-free. dflash requires WMMA matrix units;
/// [`hipfire_rdna::arch_caps::gfx_has_wmma`] is the single source of truth (the
/// same predicate behind the runtime's `arch_caps.has_wmma()` gate), so the
/// model-support matrix's dflash gfx axis can't drift from the runtime.
///
/// This is the per-(gfx) mechanical half of dflash availability; the per-family
/// half (whether the arch has a draft head / spec path at all) stays in the TOML
/// `[[arch]].dflash` intent.
pub fn dflash_gfx_supported(gfx: &str) -> bool {
    hipfire_rdna::arch_caps::gfx_has_wmma(gfx)
}
