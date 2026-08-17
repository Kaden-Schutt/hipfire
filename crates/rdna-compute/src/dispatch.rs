// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! High-level GPU dispatch interface.
//! Manages compiled kernels, provides typed tensor operations.

use crate::compiler::KernelCompiler;
use crate::feature_flags::FeatureFlags;
use crate::kernels;
use hip_bridge::{
    DeviceBuffer, HipError, HipMemAllocationProp, HipResult, HipRuntime, Rocblas, VmmArena,
    HIP_MEM_ALLOCATION_GRANULARITY_RECOMMENDED,
};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, OnceLock};

/// Per-group byte size of the MQ3-Lloyd quantization layout.
///
/// 16 B fp16 codebook (8 entries) + 96 B 3-bit packed indices = 112 B.
/// Compare to HFQ3 / uniform MQ3's 104 B/group (8 B affine header).
///
/// Every Lloyd-MQ3 dispatch arm references this constant; **never use a
/// literal 112 in dispatch.rs** — keeping the named constant lets a
/// future review grep `\* 1(04|12)` and find any Lloyd-related hits as
/// stride-mismatch bugs (followup discipline from
/// docs/plans/mq-lloyd-batched-prefill-followup.md).
pub const LLOYD_MQ3_GROUP_BYTES: usize = 112;

/// Per-group byte size of the MQ4-Lloyd quantization layout.
///
/// 32 B fp16 codebook (16 entries) + 128 B 4-bit nibble-pair indices = 160 B.
/// Compare to HFQ4 / uniform MQ4's 136 B/group (8 B affine header).
///
/// Every Lloyd-MQ4 dispatch arm references this constant; **never use a
/// literal 160 in dispatch.rs** — keeping the named constant lets a
/// future review grep `\* 1(36|60)` and find any Lloyd-related hits as
/// stride-mismatch bugs (followup discipline from
/// docs/plans/mq-lloyd-batched-prefill-followup.md).
pub const LLOYD_MQ4_GROUP_BYTES: usize = 160;

// ── MQ*-G256-GL ("global Lloyd") format constants ────────────────────────────
//
// GL = one codebook shared by the WHOLE tensor + a per-block fp16 scale, laid
// out as TWO SoA regions rather than the interleaved per-group header the
// MQ2/MQ3-Lloyd formats use:
//
//   MQ2G256GL (qt 38): [0 .. M*gpr*64)   2-bit indices, 64 B/group
//                      [M*gpr*64 .. +M*gpr*2)  fp16 per-block scales  → 2.0625 bpw
//   MQ3G256GL (qt 39): [0 .. M*gpr*96)   3-bit indices, 96 B/group
//                      [M*gpr*96 .. +M*gpr*2)  fp16 per-block scales  → 3.0625 bpw
//
// with gpr = K/256. There is no inline per-group header — any loader or size
// estimator that assumes "one contiguous blob per row with a header" is wrong
// for these two dtypes.

/// Per-group INDEX bytes for MQ2-G256-GL (2 bits × 256 weights). The fp16
/// per-block scale lives in a SEPARATE trailing region, NOT in the group.
pub const GL_MQ2_GROUP_IDX_BYTES: usize = 64;

/// Per-group INDEX bytes for MQ3-G256-GL (3 bits × 256 weights). Same SoA
/// split as MQ2-GL — the scale is in the trailing region, not inline.
pub const GL_MQ3_GROUP_IDX_BYTES: usize = 96;

/// Per-group INDEX bytes for MQ4-G256-GL (4 bits × 256 weights). Same SoA split
/// as its 2- and 3-bit siblings: `[M*gpr*128 B indices][M*gpr*2 B fp16 scales]`
/// → 130 B/group → **4.0625 bpw**, i.e. 6 B/group CHEAPER than MQ4G256's 136 B
/// (4.25 bpw) while using Gaussian-optimal levels instead of a uniform affine
/// grid. Nibble packing matches MQ4G256 (`lo | hi << 4`, weight 2i in the low
/// nibble) so a decode path can share the unpack.
pub const GL_MQ4_GROUP_IDX_BYTES: usize = 128;

/// Bytes per group in the trailing fp16 scale region (both GL dtypes).
pub const GL_GROUP_SCALE_BYTES: usize = 2;

/// **MUST STAY BIT-IDENTICAL TO `hipfire-quantize::main::GL_CB2`.**
///
/// The MQ2-GL GEMV kernels take the codebook as four SCALAR FLOAT KERNEL ARGS
/// (`cb0..cb3`) — it is *not* stored in the `.hfq` file — so the runtime must
/// reproduce the exact levels the encoder quantized against. Encoder side:
/// `crates/hipfire-quantize/src/main.rs` (`GL_CB2`, consumed by
/// `gl_encode_block` from `quantize_mq2g256gl`).
///
/// A silent drift between the two arrays is a **silent accuracy failure**, not a
/// crash: every weight decodes to a plausible-but-wrong level and the model
/// degrades without any error. The coupling is machine-checked by
/// `gl_codebooks_match_runtime` in `hipfire-quantize/src/main.rs`; if you change
/// one array you MUST change the other and that test will tell you if you didn't.
///
/// Values are the textbook Lloyd–Max reconstruction levels for a unit Gaussian
/// (2-bit, MSE 0.1175), reproduced to 3 decimals by fitting on 28.3 M real a3b
/// post-FWHT expert weights (2026-08-04).
pub const GL_CB2: [f32; 4] = [-1.5104, -0.4528, 0.4528, 1.5104];

/// **MUST STAY BIT-IDENTICAL TO `hipfire-quantize::main::GL_CB3`.**
///
/// 3-bit sibling of [`GL_CB2`]; passed to the MQ3-GL GEMV kernels as eight
/// scalar float kernel args (`cb0..cb7`, ascending). Same drift hazard, same
/// machine check (`gl_codebooks_match_runtime`). Textbook Lloyd–Max 3-bit
/// Gaussian levels (MSE 0.03454).
pub const GL_CB3: [f32; 8] = [
    -2.1520, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1520,
];

/// **MUST STAY BIT-IDENTICAL TO `hipfire-quantize::main::GL_CB4`.**
///
/// 4-bit sibling of [`GL_CB2`] / [`GL_CB3`]; a decode kernel takes it as sixteen
/// scalar float kernel args (`cb0..cb15`, ascending). Same drift hazard, same
/// machine check (`gl_codebooks_match_runtime`).
///
/// Lloyd–Max reconstruction levels for a unit Gaussian at 4 bit, MSE 0.009501.
/// Derived by running Lloyd iterations to convergence against the exact
/// Gaussian moments; the same solver reproduces [`GL_CB2`] and [`GL_CB3`] to
/// four decimals and their documented MSEs, which is what validates it.
///
/// NOTE: no kernel consumes this yet — MQ4-GL is currently encode-only. The
/// constant lives here so the encoder/runtime pair is drift-guarded from the
/// start rather than after a kernel lands and silently disagrees.
pub const GL_CB4: [f32; 16] = [
    -2.7326, -2.0690, -1.6180, -1.2562, -0.9423, -0.6568, -0.3880, -0.1284,
    0.1284, 0.3880, 0.6568, 0.9423, 1.2562, 1.6180, 2.0690, 2.7326,
];

/// Current layer index, set by the qwen35 forward_prefill_chunk at the
/// start of each layer iteration. Used by `hfq3_mmq_layer_gate_pass` to
/// support per-layer MMQ-on/off experiments (see issue #302 — KLD
/// attribution sweep). Default 0; no semantic meaning outside an
/// instrumented sweep.
pub static MMQ_CURRENT_LAYER: AtomicUsize = AtomicUsize::new(0);

fn pm4_dynamic_grid_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        hipfire_config::process_value("HIPFIRE_REPLAY_PM4_DYNAMIC_GRID")
            .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "yes" | "on"))
    })
}

/// Minimum batch size at which the FP8 WMMA prefill path is enabled.
/// Below this, the FP16 WMMA path wins on gfx1201 (measured 0.71-0.94×
/// at N ≤ 512, 0.82-1.26× only at N ≥ 2048 with high DPM variance —
/// see project_fp8_wmma_hfp4g32_2026_05_10.md). Decode (batch_size=1)
/// must never hit FP8 WMMA. Threshold tuned conservatively; A/B against
/// FP16 WMMA on the production prefill bench can lower it later.
pub(crate) const FP8_WMMA_MIN_BATCH: usize = 1024;

// AR-forward hipGraph policy (2026-05-15, after `<think>\n!!!!!` attractor
// debug on Qwen3.5-27B mq4 gfx1100):
//
//   - `ar_forward_kernel_dirty`: true on init / after kernel module change.
//     Forces direct dispatch on the very first call so any inline JIT or
//     lazy hipMalloc happens outside a captured region.
//   - `ar_forward_replay_enabled`: true only after the caller has signalled
//     `end_decode_turn()` AND a capture exists AND kernels are not dirty.
//     Until then, every forward call captures a fresh graph and launches it
//     (correct output per call; cheaper than full direct on amortization).
//
// Why caller-driven commit instead of auto-enable: empirically, captured
// graphs on this codebase + ROCm 7.2.2 sometimes snapshot stale kernarg
// state mid-decode, producing a token-0 attractor on every replay. Gating
// replay until a FULL decode turn completes via the captured-launch path
// gives the captured graph the longest possible runway to be invalidated
// by JIT recompilation; if a turn finishes coherently with capture+launch,
// the same graph is more likely to replay coherently on the next turn.

/// Minimum output dimension M at which the FP8-dot4 decode GEMV path
/// is enabled. Below this, the fallback wins or ties on gfx1201
/// (measured 0.92-1.03× on wo M=2048 K=2048 vs 1.17-1.21× on FFN
/// shapes M ≥ 4096 — see mq_rotate_x_dual_fp8 bench, 2026-05-11).
/// This is the empirical embodiment of "Option α" mixed-precision
/// routing — choose the kernel that wins for the actual shape rather
/// than uniformly applying FP8 everywhere.
pub(crate) const FP8_GEMV_MIN_M: usize = 4096;

/// Tensor stored on the GPU. Tracks shape and element type.
pub struct GpuTensor {
    pub buf: DeviceBuffer,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl GpuTensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn byte_size(&self) -> usize {
        self.numel() * self.dtype.size()
    }

    /// A `GpuTensor` whose buffer is a null pointer of size 0, for CPU-only unit
    /// tests in **dependent crates** that read only tensor metadata (shape/dtype/op)
    /// and never touch the device.
    ///
    /// CONTRACT: the returned tensor must NEVER be passed to a HIP call — its buffer
    /// is null and dereferencing it on the GPU is undefined behavior. It exists only
    /// so cross-crate tests can borrow a `&GpuTensor` for metadata-only logic.
    ///
    /// Not `#[cfg(test)]`-gated on purpose: `#[cfg(test)]` here would only be active
    /// when `rdna-compute`'s own tests build, making this invisible to dependent
    /// crates' tests (e.g. `hipfire-dispatch`). `#[doc(hidden)]` keeps it out of the
    /// public API surface while remaining reachable cross-crate, matching the
    /// `FeatureFlags::for_test` precedent.
    #[doc(hidden)]
    pub fn null_for_test() -> Self {
        GpuTensor {
            buf: unsafe {
                hip_bridge::DeviceBuffer::from_raw(std::ptr::null_mut::<std::ffi::c_void>(), 0)
            },
            shape: vec![0],
            dtype: crate::DType::F32,
        }
    }

    /// Create a non-owning sub-view at a byte offset. For F32 tensors,
    /// `offset_elems` is the number of f32 elements to skip.
    /// The returned tensor is a view — do NOT free it.
    pub fn sub_offset(&self, offset_elems: usize, len_elems: usize) -> GpuTensor {
        let element_size = self.dtype.size();
        let byte_off = offset_elems
            .checked_mul(element_size)
            .expect("tensor sub-view offset overflow");
        let byte_len = len_elems
            .checked_mul(element_size)
            .expect("tensor sub-view length overflow");
        let byte_end = byte_off
            .checked_add(byte_len)
            .expect("tensor sub-view range overflow");
        assert!(
            byte_end <= self.buf.size(),
            "tensor sub-view exceeds accessible buffer prefix"
        );
        let ptr = unsafe { (self.buf.as_ptr() as *mut u8).add(byte_off) as *mut std::ffi::c_void };
        GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, byte_len) },
            shape: vec![len_elems],
            dtype: self.dtype,
        }
    }

    /// Full-buffer non-owning alias (the whole-tensor form of `sub_offset`).
    /// The returned tensor shares the source's device pointer; it is a VIEW —
    /// do NOT pass it to `free_tensor`. `DeviceBuffer::from_raw` has no
    /// Drop-time free, so the alias and source coexist safely until the OWNER
    /// is freed exactly once.
    pub fn shallow_clone(&self) -> GpuTensor {
        GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(self.buf.as_ptr(), self.buf.size()) },
            shape: self.shape.clone(),
            dtype: self.dtype,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,         // 2 bytes; native bf16 reference (KLD oracle). Widen→f32 = high-16-bit shift.
    Q4K,          // 144 bytes per 256 elements
    Q6K,          // 210 bytes per 256 elements
    Q8_0,         // 34 bytes per 32 elements
    Q4F16G64,     // 36 bytes per 64 elements (RDNA-native FP16 dequant)
    Q4F16G32,     // 20 bytes per 32 elements (RDNA-native FP16 dequant)
    Q8HFQ,        // split-metadata: scales contiguous then values contiguous, 128B-aligned rows
    HFQ4G256,     // 136 bytes per 256 elements (flat 4-bit, f32 scale+zero, 18 VGPRs)
    HFQ4G128,     // 72 bytes per 128 elements (flat 4-bit, f32 scale+zero, 14 VGPRs)
    HFQ3G256,     // 104 bytes per 256 elements (flat 3-bit, f32 scale+zero)
    HFQ3G128,     // 56 bytes per 128 elements (flat 3-bit, f32 scale+zero)
    MQ4G256,      // MagnumQuant: FWHT-rotated HFQ4-G256 (136 bytes/group, same as HFQ4G256)
    MQ4G128,      // MagnumQuant: FWHT-128-rotated INT4 (72 bytes/group, same layout as HFQ4G128)
    MQ8G256,      // MagnumQuant: FWHT-rotated symmetric INT8, dp4a target (258 bytes/group)
    MQ6G256,      // MagnumQuant: FWHT-rotated HFQ6-G256 (200 bytes/group, same as HFQ6G256)
    MQ5G256,      // MagnumQuant: FWHT-rotated 5-bit (168 bytes/group, 5.25 bpw)
    MQ3G256,      // MagnumQuant: FWHT-rotated HFQ3-G256 (104 bytes/group, same as HFQ3G256)
    MQ2G256,      // MagnumQuant: FWHT-rotated HFQ2-G256 (72 bytes/group, same as HFQ2G256)
    MQ2G256Lloyd, // MagnumQuant 2-bit + Lloyd-Max 4-entry fp16 codebook (72 bytes/group)
    MQ3G256Lloyd, // MagnumQuant 3-bit + Lloyd-Max 8-entry fp16 codebook (112 bytes/group)
    MQ4G256Lloyd, // MagnumQuant 4-bit + Lloyd-Max 16-entry fp16 codebook (160 bytes/group)
    MQ2G256GL,    // MagnumQuant 2-bit + TENSOR-GLOBAL 4-entry codebook (GL_CB2), SoA:
    // [M*gpr*64 B indices][M*gpr*2 B fp16 per-block scales] = 2.0625 bpw. NOT
    // interleaved — no per-group header. Codebook is a compile-time constant
    // passed as scalar kernel args, not stored in the file. MoE-routed-expert
    // only (indexed gate_up + atomic self-combining down); there is no dense /
    // plain / prerotated single-weight GEMV kernel for this dtype.
    MQ3G256GL, // MagnumQuant 3-bit + TENSOR-GLOBAL 8-entry codebook (GL_CB3), SoA:
    // [M*gpr*96 B indices][M*gpr*2 B fp16 per-block scales] = 3.0625 bpw. Same
    // MoE-only scope + scalar-arg codebook as MQ2G256GL.
    HFP4G32, // HFP4: E2M1 element + UE8M0 g32 block scale + FP16 row scale.
    // Per-row header 16 B; per-block payload 17 B (UE8M0 + 16 packed nibbles).
    // See docs/quant-formats/hfp4.md.
    MFP4G32, // MFP4: HFP4G32 + offline FWHT (drop-in MQ4 replacement). Same byte layout
    // as HFP4G32; format_flags bit 0 + bits 2-3 = 01 stamps the rotation kind.
    // Runtime applies the matching FWHT to x via mq_rotate_x; the kernel itself
    // is shared with HFP4G32.
    MFP4G32Lloyd, // mfp4 + per-tensor 16-entry fp16 Lloyd codebook. Same per-row byte
    // layout as MFP4G32, plus a 32-B codebook prefix before row 0. format_flags=0x05
    // (same FWHT rotation as MFP4G32); recon uses codebook[nibble] not E2M1_LUT.
    MFP4G32P, // mfp4+P — mfp4 (E2M1 + FP16 row scale + offline FWHT) with the per-32-block
    // UE8M0 scale promoted to E4M3 (FP8, non-power-of-2). Byte layout BYTE-IDENTICAL to
    // MFP4G32 (NO prefix); only the per-block scale byte's decode differs (E4M3 vs UE8M0).
    MFP4G32E8, // mfp4-E8: mfp4+P container (E4M3 block scale, NO prefix, same row_bytes)
    // with the per-32-block 16 E2M1 nibbles replaced by 4x32-bit E8-lattice codewords
    // (8 weights/codeword, QUANT_STEP=0.88). 4.25 bpw, byte-IDENTICAL footprint to MFP4G32P.
    MFP4G32E8SOA, // mfp4-E8 SoA: same E8 data as MFP4G32E8 permuted for coalesced reads.
    // Per-row: [16B hdr: row_scale_a:f16@0, n_blocks:u16@4, flag:0x06@6]
    //   + [n_blocks B: E4M3 scales, pad to 16B boundary]
    //   + [n_blocks*16 B: 4xu32 E8 codewords/block, 16B-aligned].
    // Pure byte-permutation of MFP4G32E8 => dequant result IDENTICAL.
    MFP3G32E8, // mfp3-E8: MFP4G32E8 frame, 3-bit lattice (center 4), 13 B/blk, 104 B/grp, 3.25 bpw. Drop-in for MQ3G256Lloyd.
    MFP2G32E8, // mfp2-E8: MFP4G32E8 frame, 2-bit lattice (center 2),  9 B/blk,  72 B/grp, 2.25 bpw. Drop-in for MQ2G256Lloyd.
    HFQ2G256,  // 72 bytes per 256 elements (flat 2-bit, f32 scale+zero, ~19 VGPRs)
    HFQ2G128,  // 40 bytes per 128 elements (flat 2-bit, f32 scale+zero)
    HFQ6G256,  // 200 bytes per 256 elements (6-bit, f32 scale+zero)
    ParoQ4G128, // ParoQuant: AWQ-packed INT4 G128 repacked to HFQ4G128 layout at load.
    // Weights are standard HFQ4G128 (72 bytes/group); the ParoQuant distinction
    // is that weight_gemv applies Givens rotation to activations before GEMV.
    // Rotation metadata (pairs, theta, channel_scales) lives on WeightTensor::paro.
    Raw, // raw bytes, no element interpretation
}

impl DType {
    pub fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::Q4K
            | DType::Q6K
            | DType::Q8_0
            | DType::Q4F16G64
            | DType::Q4F16G32
            | DType::Q8HFQ
            | DType::HFQ4G256
            | DType::HFQ4G128
            | DType::HFQ3G256
            | DType::HFQ3G128
            | DType::HFQ2G256
            | DType::HFQ2G128
            | DType::HFQ6G256
            | DType::MQ4G256
            | DType::MQ4G128
            | DType::MQ6G256
            | DType::MQ5G256
            | DType::MQ8G256
            | DType::MQ3G256
            | DType::MQ2G256
            | DType::MQ2G256Lloyd
            | DType::MQ3G256Lloyd
            | DType::MQ4G256Lloyd
            | DType::MQ2G256GL
            | DType::MQ3G256GL
            | DType::HFP4G32
            | DType::MFP4G32
            | DType::MFP4G32Lloyd
            | DType::MFP4G32P
            | DType::MFP4G32E8
            | DType::MFP4G32E8SOA
            | DType::MFP3G32E8
            | DType::MFP2G32E8
            | DType::ParoQ4G128
            | DType::Raw => 1, // byte-level
        }
    }

    /// Whether a `WeightTensor` of this dtype should have the
    /// `<weight>.awq_scale.weight` F16 sidecar attached at load time.
    ///
    /// Centralizes the gate that previously lived inline at every
    /// loader call site (qwen35.rs `load_weight_tensor`, etc.). The
    /// motivation is the May 2026 regression where `qwen35.rs:907`
    /// gated on `matches!(wt.gpu_dtype, DType::MQ4G256)` and silently
    /// dropped AWQ sidecars for `MQ3G256`-quantized Qwen3.5 weights,
    /// producing fluent-but-nonsensical token soup for ~5 hours
    /// before the missing arm was traced. Adding a new AWQ-eligible
    /// dtype is now a one-line edit here instead of two scattered
    /// edits per loader.
    ///
    /// Current allow-list = the empirical truth of which dtypes ship
    /// AWQ sidecars from the quantizer AND have an AWQ-aware forward
    /// path (`rotate_x_mq_for` etc., wired through `awq_scale.is_some()`).
    ///
    /// **Forward-path-ready candidates not currently in the allow-list**
    /// (forward kernels exist but no `.hfq` file in tree ships sidecars
    /// for them — widen only after the quantizer side is verified to
    /// emit sidecars and at least one coherence-gate row exercises the
    /// combination):
    /// - `MQ6G256`
    /// - `MQ2G256`, `MQ2G256Lloyd`
    /// - `MQ3G256Lloyd`
    /// - `MFP4G32` (forward path has explicit `awq_scale.is_some()`
    ///   branching at llama.rs:609 but the quantizer comment says
    ///   "AWQ is gated to MQ4G256 today" — confirm before widening)
    ///
    /// `MQ8G256` is explicitly **not** a candidate: it uses its own
    /// INT8-quantized scratch path (`gemv_mq8g256_with_rotate`,
    /// `rotate_quantize_x_mq8`) and does not flow through
    /// `rotate_x_mq_for`, so there is no AWQ-aware kernel to dispatch
    /// to.
    ///
    /// **lm_head / embed_tokens callers:** as of the lm_head-AWQ
    /// runtime PR, this helper IS safe for the `output` weight in
    /// `qwen35.rs::load_weights` / `load_weights_vl`. Both dispatch
    /// paths that consume `weights.output` now route through
    /// AWQ-aware rotations when a sidecar is attached:
    /// - Decode: `weight_gemv` → `rotate_x_mq_for` (llama.rs)
    /// - Spec-decode verify: `speculative.rs::rotate_x_mq_batched_for`
    ///
    /// Pre-runtime-fix, attaching a sidecar on lm_head would have
    /// produced `(W·s)·x ≠ W·x` via the spec-verify path's plain
    /// `rotate_x_mq_batched` and driven the KLD 0.67 → 13.5
    /// corruption documented at `docs/plans/awq_fix_claude.md`. The
    /// quantizer-side `awq_eligible` whitelist
    /// (`hipfire-quantize/src/main.rs:3849`) still gates which
    /// tensors actually receive `W' = W·s` pre-multiplication at
    /// quant time — this helper governs only whether the loader
    /// attaches an already-emitted sidecar.
    pub fn supports_awq_sidecar(self) -> bool {
        // MQ3G256Lloyd / MQ2G256Lloyd added 2026-05-28: they are "forward-path-ready"
        // (flow through rotate_x_mq_for, which applies x/=awq_scale when a sidecar is
        // attached) — see the doc block above. Enables AWQ×Lloyd composition once the
        // quantizer emits sidecars for the Lloyd arms.
        matches!(
            self,
            DType::MQ4G256
                | DType::MQ3G256
                | DType::MQ2G256
                | DType::MQ3G256Lloyd
                | DType::MQ2G256Lloyd
        )
    }

    /// Per-row byte stride for split-metadata layouts. Q8HFQ packs `n_groups*2`
    /// scale bytes + `k` value bytes per row, padded to a 128-byte boundary; all
    /// other dtypes encode their layout internally and ignore this value (returns 0).
    /// Single source of truth for the formula previously inlined at hfq.rs:748.
    pub fn row_stride(self, k: usize) -> usize {
        match self {
            DType::Q8HFQ => {
                let raw_row = (k / 32) * 2 + k;
                (raw_row + 127) & !127
            }
            _ => 0,
        }
    }

    /// Whether this format's GEMV kernel requires K%256==0 (HFP4 family: the
    /// gemv_hfp4g32 kernel + FWHT both need it). Refuse at load, not first
    /// dispatch. Centralizes the guard inlined at the weight-decode qt 21/24 arms.
    ///
    /// MQ2/MQ3-G256-GL are included because their SoA region split is derived
    /// from `gpr = K/256` on BOTH sides (encoder and kernel). A K that is not a
    /// multiple of 256 truncates `gpr`, which silently shifts the scale-region
    /// base — garbage weights with no error. Fail at load instead.
    pub fn requires_k_mod_256(self) -> bool {
        matches!(
            self,
            DType::HFP4G32 | DType::MFP4G32 | DType::MQ2G256GL | DType::MQ3G256GL
        )
    }
}

/// Activation-capture hook for the Tier 1 hipfire-native calibration path.
///
/// Foundation scaffold (2026-05-19) — the field on `Gpu` is set by
/// `collect_imatrix` / `collect_hessian` (see
/// `crates/hipfire-runtime/src/bin/`) and called from each linear-layer
/// dispatch site to feed activations into an on-GPU reduction
/// (per-channel `Σ act²` for imatrix, K×K outer-product for the GPTQ
/// Hessian).
///
/// `tensor_name` is the canonical hipfire tensor identifier (the same
/// string the .hfq loader uses, e.g. `model.layers.0.self_attn.q_proj`)
/// so the reduction kernel can key its on-GPU accumulator dictionary
/// by name without ambiguity across MoE expert indices.
///
/// `input_ptr` / `numel` / `dtype` describe the activation tensor in
/// HBM at the moment of the linear-layer dispatch. The capture
/// implementation is responsible for launching its own reduction
/// kernel on the same stream as the producing GEMM (so ordering is
/// preserved without an extra `hipDeviceSynchronize`). The hook MUST
/// NOT free or reallocate the input tensor.
///
/// `Send + Sync` lets the same handler be shared across multi-GPU
/// dispatch threads (one `Gpu` instance per device, all pointing at
/// the same Arc'd handler that funnels into a per-tensor accumulator).
pub trait ActivationCapture: Send + Sync {
    /// Called by linear-layer dispatch arms when calibration is active.
    ///
    /// `gpu`         — the dispatcher, so the collector can run its on-GPU
    ///                 reduction kernels (`calib_sumsq_reduce_f32` /
    ///                 `calib_hessian_outer_f32`). Safe to take `&mut Gpu`:
    ///                 the dispatch site clones the collector `Arc` before
    ///                 calling, so `gpu.active_capture` is not aliased here.
    /// `tensor_name` — canonical .hfq / GGUF tensor name (resolved from the
    ///                 weight buffer pointer via `gpu.capture_names`).
    /// `input`       — the input-activation buffer (borrowed; do NOT retain past
    ///                 the call). NOTE its `.shape` may be a shared scratch sized
    ///                 to `max(dim, hidden)`, so it is NOT a reliable source of
    ///                 `k`/`n` — use the passed `n`/`k` instead.
    /// `n`           — number of activation rows (tokens / batch) this call.
    /// `k`           — the linear's input dim (the meaningful width of each row).
    /// Interior mutability (`&self`) lets the collector accumulate without an
    /// exclusive borrow.
    fn capture(&self, gpu: &mut Gpu, tensor_name: &str, input: &GpuTensor, n: usize, k: usize);
}

/// Per-weight MMQ screening state (issue #87).
pub struct MmqScreenState {
    pub cache: HashMap<usize, bool>,
    pub enabled: bool,
    pub threshold: f32,
}

/// High-level GPU context. Owns the HIP runtime, compiler, and loaded kernels.
pub struct Gpu {
    pub hip: HipRuntime,
    pub arch: String,
    pub flags: Arc<FeatureFlags>,
    pub arch_caps: crate::arch_caps::ArchCaps,
    pub device_id: i32,
    pub(crate) compiler: KernelCompiler,
    pub(crate) modules: HashMap<String, hip_bridge::Module>,
    pub(crate) functions: HashMap<String, hip_bridge::Function>,
    pub(crate) pool: crate::pool::GpuPool,
    /// VMM owners keyed by their base virtual address. VMM-backed tensors use
    /// non-owning DeviceBuffer views and must bypass GpuPool/hipFree teardown.
    vmm_arenas: HashMap<usize, VmmArena>,
    /// Arenas whose cleanup failed before they could enter the owner map.
    /// They have no tensor owner and are retried during explicit/Gpu teardown.
    orphan_vmm_arenas: Vec<VmmArena>,
    /// When set, all kernel launches go to this stream instead of null stream.
    pub active_stream: Option<hip_bridge::Stream>,
    /// Scratch buffers for FWHT rotation, FP16/FP8 activation conversion, etc.
    pub scratch: crate::scratch::ScratchState,
    /// Model-scoped Redline warmup recorder and fail-closed backend gate.
    pub replay: crate::replay::ReplayController,

    // ── MMQ per-weight screening (#87) — extracted to MmqScreenState ──────
    pub mmq_screen: MmqScreenState,

    // ── hipGraph capture state (extracted to graph.rs) ─────────────────────
    pub graphs: crate::graph::GraphState,

    // ── rocBLAS (CDNA3 MFMA-accelerated GEMM) ─────────────────────────────
    /// Optional rocBLAS handle. `None` on non-CDNA3 archs or when
    /// librocblas.so fails to load. Engine code should always gate on
    /// `.is_some()` and fall back to the hand-rolled HFQ4 kernels otherwise.
    pub rocblas: Option<Rocblas>,

    /// FP16 shadow cache for HFQ4-G256 weights. Populated lazily on first
    /// batched prefill through the rocBLAS path: we dequantize the MQ4
    /// weight into an FP16 buffer once, then reuse for every subsequent
    /// prefill call. Key is the MQ4 device pointer (usize for Hash); value
    /// owns the GPU-side FP16 tensor. Memory is not freed until the Gpu
    /// itself drops (weights are assumed immutable for a model's lifetime).
    ///
    /// Only populated on CDNA3 when rocBLAS loaded — 4× VRAM blow-up vs MQ4
    /// so consumer cards stay on the wave32/64 hand-rolled GEMV path.
    fp16_shadow_cache: HashMap<usize, GpuTensor>,

    /// Calibration activation capture (Tier-1 collector). When `Some`, the
    /// instrumented linear dispatch arms resolve their weight buffer pointer
    /// to a tensor name via `capture_names` and invoke `capture()` with the
    /// input activation. `None` (the default) ⇒ the check is a single
    /// `is_none()` and forwards are byte-identical. The collector is held by
    /// `Arc` so the dispatch site can clone it (breaking the borrow on `self`)
    /// before calling `capture(self, …)`.
    pub active_capture: Option<Arc<dyn ActivationCapture>>,
    /// Weight-buffer-pointer → canonical tensor name, populated when calibration
    /// is armed. Lets capture fire from ANY forward path (hand or lowered,
    /// fused or not) keyed by the weight the gemv received.
    pub capture_names: HashMap<usize, String>,

    /// Native GPTQ-on-E8 Hessian collection. `None` in production (zero
    /// overhead -- a single `is_some()` branch in the MoE CPU-top-K fallback
    /// per routed expert). The `collect_e8_hessian_native` example sets this to
    /// `Some(default)` before running the calibration forward; every routed
    /// expert then accumulates its per-256-block XX^T over the RAW pre-rotation
    /// input (gate_up: post-rmsnorm x; down: silu(g)*u), keyed by the full
    /// safetensors tensor name == `hipfire-quantize::main::hessian_key`. Drained
    /// to per-(tensor,expert) `.hblk` files after the pass. See
    /// `hipfire-dispatch::pipeline::run_moe_decode_cpu_fallback` for the hook.
    pub hessian_capture: Option<HessianCapture>,
}
/// Per-256-block XX^T accumulator for ONE weight tensor (one expert), keyed
/// inside [`HessianCapture`] by the full safetensors name. Byte-for-byte the
/// same accumulation + `.hblk` layout as
/// `hipfire-quantize::bin::collect_e8_hessian::BlockHessian` (duplicated here
/// because that lives in a `bin` crate the GPU layer cannot import). The
/// quantizer's `load_hessian_blocks` reads exactly this format.
#[derive(Debug)]
pub struct BlockHessianAcc {
    pub k: usize,
    pub n_blocks: usize,
    /// `n_blocks` blocks of `256*256` f64 accumulators (row-major per block).
    pub blocks: Vec<Vec<f64>>,
    pub n_rows: u64,
}

impl BlockHessianAcc {
    pub fn new(k: usize) -> Self {
        assert!(k % 256 == 0, "Hessian K={k} must be divisible by 256");
        let n_blocks = k / 256;
        BlockHessianAcc {
            k,
            n_blocks,
            blocks: (0..n_blocks).map(|_| vec![0.0f64; 256 * 256]).collect(),
            n_rows: 0,
        }
    }

    /// Accumulate one pre-rotation activation row `x[0..K]` into the per-256-block
    /// diagonal XX^T (block b += x_b x_b^T over its 256 channels).
    pub fn accumulate_row(&mut self, x: &[f32]) {
        debug_assert_eq!(x.len(), self.k);
        for b in 0..self.n_blocks {
            let xb = &x[b * 256..b * 256 + 256];
            let acc = &mut self.blocks[b];
            for i in 0..256 {
                let xi = xb[i] as f64;
                if xi == 0.0 {
                    continue;
                }
                let row = &mut acc[i * 256..i * 256 + 256];
                for j in 0..256 {
                    row[j] += xi * xb[j] as f64;
                }
            }
        }
        self.n_rows += 1;
    }

    /// Serialize to the `.hblk` format consumed by
    /// `hipfire-quantize::main::load_hessian_blocks`:
    /// `[u32 magic=0x45384831][u32 n_blocks=K/256][u32 K][f32 ... n_blocks*256*256]`.
    pub fn write_hblk(&self, dir: &std::path::Path, tensor_name: &str) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        // MUST byte-match hessian_key(): replace('/','_').replace('\\','_').replace("..","_").
        let key = tensor_name.replace(['/', '\\'], "_").replace("..", "_");
        let path = dir.join(format!("{key}.hblk"));
        let mut buf = Vec::with_capacity(12 + self.n_blocks * 256 * 256 * 4);
        buf.extend_from_slice(&0x45_38_48_31u32.to_le_bytes()); // "E8H1"
        buf.extend_from_slice(&(self.n_blocks as u32).to_le_bytes());
        buf.extend_from_slice(&(self.k as u32).to_le_bytes());
        for b in 0..self.n_blocks {
            for &v in &self.blocks[b] {
                buf.extend_from_slice(&(v as f32).to_le_bytes());
            }
        }
        std::fs::write(&path, &buf)
    }

    /// Mean of the per-block diagonal (sanity: should be > 0 for a frequently
    /// routed expert; 0 == never accumulated).
    pub fn mean_diag(&self) -> f64 {
        let mut s = 0.0f64;
        let mut n = 0u64;
        for b in 0..self.n_blocks {
            for i in 0..256 {
                s += self.blocks[b][i * 256 + i];
                n += 1;
            }
        }
        if n == 0 {
            0.0
        } else {
            s / n as f64
        }
    }
}

/// Host-side accumulator for native GPTQ-on-E8 Hessian collection. Lives on
/// [`Gpu`] so the MoE CPU-top-K fallback chokepoint can reach it without
/// threading a parameter through every forward path. Pure host data.
///
/// Keyed by the FULL safetensors tensor name (== the quantizer's
/// `hessian_key` input), e.g.
/// `model.language_model.layers.7.mlp.experts.42.gate_up_proj.weight`.
#[derive(Debug, Default)]
pub struct HessianCapture {
    /// `tensor_name -> per-256-block XX^T`. Lazily sized to the tensor's K on
    /// first sighting of that (tensor,expert).
    pub entries: HashMap<String, BlockHessianAcc>,
    /// Total calibration tokens processed (collector increments once/step).
    pub n_tokens: u64,
}

impl HessianCapture {
    /// Accumulate one RAW pre-rotation activation row for `name`. `x` is the
    /// linear's pre-rotation input (gate_up: post-rmsnorm x; down: silu(g)*u);
    /// only the first `k` channels are used. Creates the entry on first sight.
    pub fn accumulate(&mut self, name: &str, x: &[f32], k: usize) {
        let acc = self
            .entries
            .entry(name.to_string())
            .or_insert_with(|| BlockHessianAcc::new(k));
        if x.len() >= k {
            acc.accumulate_row(&x[..k]);
        } else {
            let mut row = vec![0.0f32; k];
            row[..x.len()].copy_from_slice(x);
            acc.accumulate_row(&row);
        }
    }

    /// Accumulate ALL of one token's routed (tensor,expert) activation rows in
    /// PARALLEL across rayon worker threads. `items` is the token's list of
    /// `(name, x, k)` work-units (gate_up + down for each of the top-k experts).
    ///
    /// WHY THIS IS THE HOT-PATH OPTIMIZATION: the per-token capture cost is
    /// dominated by the `accumulate_row` rank-1 XX^T updates over a HUGE, COLD
    /// working set (A3B: ~16k distinct (tensor,expert) accumulators, ~30 GB of
    /// f64), which is memory-system-bound (cold-line latency + first-touch page
    /// faults), single-threaded, GPU idle, 20 host cores free. Each work-unit in
    /// `items` targets a DISTINCT (tensor,expert) accumulator (distinct experts
    /// per token, distinct gate_up/down tensors), so the accumulators are
    /// pairwise DISJOINT and can be updated concurrently — spreading the cold
    /// memory traffic + faults across cores.
    ///
    /// BIT-IDENTICAL TO SERIAL: each accumulator receives EXACTLY the same
    /// `accumulate_row(x)` call it would in the per-expert serial loop, computed
    /// by exactly ONE thread; only the (already order-independent, because the
    /// targets are disjoint) cross-accumulator iteration is parallelized. The
    /// internal float sum of every H[i,j] entry is byte-for-byte the serial
    /// result. See `accumulate_token_bit_identical_to_serial` parity test.
    pub fn accumulate_token(&mut self, items: &[(String, &[f32], usize)]) {
        use rayon::prelude::*;
        // 1) Ensure every entry exists (serial; allocation must not race the
        //    HashMap). Entries are keyed by full name; first sight sizes to k.
        for (name, _x, k) in items {
            self.entries
                .entry(name.clone())
                .or_insert_with(|| BlockHessianAcc::new(*k));
        }
        // 2) Gather a raw *mut to each target accumulator. The targets are
        //    pairwise disjoint iff the names within this token are distinct —
        //    which routing guarantees (distinct expert ids; distinct tensors).
        //    A SyncPtr wrapper lets rayon move the (provably disjoint) pointers
        //    across threads; no two closures touch the same accumulator.
        #[derive(Clone, Copy)]
        struct SyncPtr(*mut BlockHessianAcc);
        // SAFETY: each pointer addresses a distinct HashMap value (distinct keys,
        // asserted below); the map is not mutated for the duration of the
        // parallel section, so the pointees are stable and non-overlapping.
        unsafe impl Send for SyncPtr {}
        unsafe impl Sync for SyncPtr {}

        // Debug guard: catch any accidental duplicate key (would alias &mut).
        debug_assert!(
            {
                let mut ns: Vec<&str> = items.iter().map(|(n, _, _)| n.as_str()).collect();
                ns.sort_unstable();
                let before = ns.len();
                ns.dedup();
                ns.len() == before
            },
            "accumulate_token: duplicate (tensor,expert) name within one token              would alias &mut — parallel accumulate requires distinct keys"
        );

        let ptrs: Vec<(SyncPtr, &[f32], usize)> = items
            .iter()
            .map(|(name, x, k)| {
                let acc: &mut BlockHessianAcc =
                    self.entries.get_mut(name).expect("entry ensured above");
                (SyncPtr(acc as *mut BlockHessianAcc), *x, *k)
            })
            .collect();

        // 3) Parallel accumulate, one work-unit per rayon task. Each task calls
        //    the SAME `accumulate_row` as serial on its own disjoint accumulator.
        ptrs.par_iter().for_each(|&(p, x, k)| {
            // SAFETY: disjoint targets (distinct keys); no aliasing across tasks.
            let acc: &mut BlockHessianAcc = unsafe { &mut *p.0 };
            if x.len() >= k {
                acc.accumulate_row(&x[..k]);
            } else {
                let mut row = vec![0.0f32; k];
                row[..x.len()].copy_from_slice(x);
                acc.accumulate_row(&row);
            }
        });
    }
}

/// Generate `n` FWHT sign values (+1.0 / -1.0) from a simple LCG seeded with `seed`.
/// Deterministic and portable; used by both host-side codec (weight encoding) and
/// device-side init (`ensure_mq_signs` / `ensure_mq_signs_128`).
pub fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
            if (state >> 16) & 1 == 1 {
                1.0f32
            } else {
                -1.0f32
            }
        })
        .collect()
}

impl Gpu {
    /// Diagnostic: trace multi-slot session continuation matching.
    pub fn slot_trace(&self) -> bool {
        // bind_thread: skip — pure flag read, touches no device state.
        self.flags.slot_trace
    }

    /// Whether the multi-slot decode step should be hipGraph-captured.
    pub fn slots_decode_graph(&self) -> bool {
        // bind_thread: skip — pure flag read, touches no device state.
        self.flags.slots_decode_graph
    }

    /// Install a real stream if launches are still going to the null stream.
    ///
    /// HIP refuses to capture the legacy default stream, so anything that
    /// wants `begin_stream_capture` must call this first. Idempotent, and the
    /// stream stays installed afterwards: every later launch simply goes to it
    /// instead of the null stream.
    pub fn ensure_capture_stream(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        if self.active_stream.is_none() {
            self.active_stream = Some(self.hip.stream_create()?);
        }
        Ok(())
    }

    /// Begin capturing this `Gpu`'s stream into a graph.
    ///
    /// Mode 1 is `hipStreamCaptureModeThreadLocal`: only this thread is
    /// restricted, rather than mode 0's process-wide restriction. Everything
    /// this crate launches during capture must already be warm -- a kernel
    /// compile mid-capture is exactly the kind of call the mode forbids.
    pub fn begin_stream_capture(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        let stream = self.active_stream.as_ref().ok_or_else(|| {
            hip_bridge::HipError::new(0, "begin_stream_capture: no active stream")
        })?;
        self.hip.stream_begin_capture(stream, 1)
    }

    /// Close the capture started by `begin_stream_capture`.
    pub fn end_stream_capture(&mut self) -> HipResult<hip_bridge::Graph> {
        self.bind_thread()?;
        let stream = self
            .active_stream
            .as_ref()
            .ok_or_else(|| hip_bridge::HipError::new(0, "end_stream_capture: no active stream"))?;
        self.hip.stream_end_capture(stream)
    }

    /// Launch a previously instantiated graph on this `Gpu`'s stream.
    pub fn launch_graph(&mut self, exec: &hip_bridge::GraphExec) -> HipResult<()> {
        self.bind_thread()?;
        let stream = self
            .active_stream
            .as_ref()
            .ok_or_else(|| hip_bridge::HipError::new(0, "launch_graph: no active stream"))?;
        self.hip.graph_launch(exec, stream)
    }

    /// Returns the active stream ref for kernel launches (None = null stream).
    pub(crate) fn stream_ref(&self) -> Option<&hip_bridge::Stream> {
        self.active_stream.as_ref()
    }

    /// Bind this `Gpu`'s device on the calling thread. Delegates to
    /// `crate::graph::bind_thread`.
    #[inline]
    pub fn bind_thread(&self) -> HipResult<()> {
        crate::graph::bind_thread(&self.hip, self.device_id)
    }

    /// `bind_thread` for `&mut self -> ()` and `Drop` contexts. Delegates to
    /// `crate::graph::bind_thread_or_warn`.
    #[inline]
    pub fn bind_thread_or_warn(&self) {
        crate::graph::bind_thread_or_warn(&self.hip, self.device_id)
    }

    /// Drive the GPU to full DPM perf level before a perf-sensitive measurement.
    ///
    /// gfx1100 (and other RDNA cards) return to a low-power DPM state when
    /// GPU utilization drops. A fresh process, or a process that just did
    /// light CPU-side setup, will find the GPU partially idling. Kernels run
    /// at reduced sclk/mclk until enough sustained load convinces the driver
    /// to ramp up. That ramp-up is slow and variable (~1-10 s observed), and
    /// its variance produces cycle-time swings like 52 ms vs 358 ms on the
    /// same bench. See `docs/methodology/perf-benchmarking.md`.
    ///
    /// This runs a tight memset + small-gemm loop for `secs` seconds to pin
    /// the GPU at high DPM before the caller's timer starts. Memset stresses
    /// mclk; the existing JITed `gemv_hfq4g256` kernel (available on any
    /// caller that has compiled a DFlash/Qwen3.5 model) stresses sclk.
    pub fn dpm_warmup(&mut self, secs: f32) -> HipResult<()> {
        self.bind_thread()?;
        // 256 MB scratch — large enough to defeat L2 and tax the memory
        // controller. GDDR6 on the 7900 XTX is 24 GB so 256 MB is trivial.
        const SCRATCH_BYTES: usize = 256 * 1024 * 1024;
        let scratch = self.hip.malloc(SCRATCH_BYTES)?;
        eprint!("warming caches...");
        let t0 = std::time::Instant::now();
        let mut n: u64 = 0;
        while t0.elapsed().as_secs_f32() < secs {
            // Rotate the fill byte so the driver/card can't short-circuit
            // repeated identical writes via any dedup or cache-match path.
            self.hip
                .memset(&scratch, (n & 0xFF) as i32, SCRATCH_BYTES)?;
            self.hip.device_synchronize()?;
            n = n.wrapping_add(1);
        }
        let elapsed = t0.elapsed().as_secs_f32();
        eprintln!(" took {elapsed:.2}s");
        // Free the 256 MB scratch — DeviceBuffer has no Drop, so scope exit
        // would otherwise leak it for the lifetime of the process.
        let _ = self.hip.free(scratch);
        Ok(())
    }

    pub fn init() -> HipResult<Self> {
        Self::init_with_device(0)
    }

    pub fn init_with_device(id: i32) -> HipResult<Self> {
        let hip = HipRuntime::load()?;
        let count = hip.device_count()?;
        if count == 0 {
            return Err(hip_bridge::HipError::new(0, "no GPU devices found"));
        }
        if id < 0 || id >= count {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("device id {id} out of range (count={count})"),
            ));
        }
        if let Ok(mode) = hipfire_config::developer_var("HIPFIRE_HIP_WAIT") {
            let mode_lc = mode.to_ascii_lowercase();
            let flags = match mode_lc.as_str() {
                "auto" => Some(0x00),
                "spin" => Some(0x01),
                "yield" => Some(0x02),
                "block" | "blocking" | "blocking_sync" => Some(0x04),
                "" => None,
                other => {
                    eprintln!(
                        "WARNING: unknown HIPFIRE_HIP_WAIT={other:?}; expected auto|spin|yield|blocking"
                    );
                    None
                }
            };
            if let Some(flags) = flags {
                hip.set_device_flags(flags)?;
                eprintln!("[hipfire] HIP wait mode: {mode_lc}");
            }
        }
        // set_device must precede try_init_rocblas — rocBLAS captures the
        // currently-bound device into its handle.
        hip.set_device(id)?;

        // HIPFIRE_TARGET_ARCH overrides the detected GPU arch for kernel
        // compilation. Used to test cross-arch family targets like
        // `gfx10-1-generic` (covers Navi 10/12/14) without per-arch JIT
        // cache fragmentation. Empty / unset preserves prior behavior.
        let detected_arch = hip.get_arch(id).unwrap_or_else(|_| "gfx1010".to_string());
        let arch = hipfire_config::developer_var("HIPFIRE_TARGET_ARCH")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or(detected_arch);
        let (_, vram_total) = hip.get_vram_info().unwrap_or((0, 0));

        // Check HIP runtime version matches GPU arch requirements
        let (hip_major, hip_minor) = hip.runtime_version().unwrap_or((0, 0));
        let (min_major, min_minor) = match arch.as_str() {
            "gfx1200" | "gfx1201" => (6, 4),             // RDNA4 needs ROCm 6.4+
            "gfx1150" | "gfx1151" | "gfx1152" => (7, 2), // RDNA3.5 (Strix) needs ROCm 7.2+
            "gfx1100" | "gfx1101" | "gfx1102" => (5, 5), // RDNA3 needs ROCm 5.5+
            _ => (5, 0),
        };
        if hip_major > 0
            && (hip_major < min_major || (hip_major == min_major && hip_minor < min_minor))
        {
            eprintln!(
                "WARNING: HIP runtime {}.{} may not support {}. Minimum: {}.{}",
                hip_major, hip_minor, arch, min_major, min_minor
            );
            eprintln!("  Update your HIP runtime or kernels may fail to load.");
        }
        eprintln!(
            "GPU dev {}: {} ({:.1} GB VRAM, HIP {}.{})",
            id,
            arch,
            vram_total as f64 / 1e9,
            hip_major,
            hip_minor
        );

        let flags = Arc::new(FeatureFlags::from_active_config(&arch));
        let arch_caps = crate::arch_caps::ArchCaps::new(&arch, flags.clone());

        let compiler = KernelCompiler::new(&arch, flags.hipcc_extra_flags.clone())?;

        crate::graph::LAST_BOUND_DEVICE.with(|c| c.set(id));

        let mmq_screen = flags.mmq_screen;
        let mmq_screen_threshold = flags.mmq_screen_threshold;

        Ok(Self {
            hip,
            arch,
            flags,
            arch_caps,
            device_id: id,
            compiler,
            modules: HashMap::new(),
            functions: HashMap::new(),
            pool: crate::pool::GpuPool::new(),
            vmm_arenas: HashMap::new(),
            orphan_vmm_arenas: Vec::new(),
            active_stream: None,
            scratch: crate::scratch::ScratchState {
                mq_signs1: None,
                mq_signs2: None,
                mq_signs1_128: None,
                mq_signs2_128: None,
                mq_x_rot: None,
                mq_x_rot_fp8: None,
                mq_x_rot_fp8_bytes: 0,
                mq_x_q8: None,
                mq_x_scales: None,
                mq_rmsnorm_wavegrid_scratch: None,
                gemv_residual_tmp: None,
                paro_x_scratch: None,
                paro_fused_scratch: None,
                fp16_x_scratch: None,
                fp16_x_scratch_bytes: 0,
                fp16_x_source_ptr: std::ptr::null_mut(),
                fp8_x_scratch: None,
                fp8_x_scratch_bytes: 0,
                fp8_x_source_ptr: std::ptr::null_mut(),
                q8_1_mmq_x_scratch: None,
                q8_1_mmq_x_scratch_bytes: 0,
                ksplit_det_partials: None,
                ksplit_det_partials_bytes: 0,
                sample_partials: None,
                sample_partials_bytes: 0,
            },
            replay: crate::replay::ReplayController::from_config(),
            mmq_screen: MmqScreenState {
                cache: HashMap::new(),
                enabled: mmq_screen,
                threshold: mmq_screen_threshold,
            },
            graphs: crate::graph::GraphState {
                capture_mode: false,
                capture_blobs: Vec::new(),
                graph_exec: None,
                captured_graph: None,
                ar_forward_blobs: Vec::new(),
                ar_forward_kernel_dirty: true,
                ar_forward_replay_enabled: false,
                ar_graph_eligible: true,
                verify: crate::graph::PerBGraphCache {
                    cache: std::collections::HashMap::new(),
                    warmed_up: std::collections::HashSet::new(),
                    capturing: None,
                    lmhead_argmax: std::collections::HashSet::new(),
                },
                replay: crate::graph::PerBGraphCache {
                    cache: std::collections::HashMap::new(),
                    warmed_up: std::collections::HashSet::new(),
                    capturing: None,
                    lmhead_argmax: std::collections::HashSet::new(),
                },
            },
            rocblas: None,
            fp16_shadow_cache: HashMap::new(),
            active_capture: None,
            capture_names: HashMap::new(),
            hessian_capture: None,
        })
        .map(|mut gpu| {
            if gpu.flags.force_blob_path {
                eprintln!("[diag] HIPFIRE_BLOB_FORCE=1: all kernel launches will use the blob path (kernelParams bypassed). Diagnostic only.");
            }
            // Auto-init rocBLAS on CDNA3 so the batched-prefill MFMA path is
            // available out of the box. No-op on consumer arches.
            gpu.try_init_rocblas();
            gpu
        })
    }

    /// Try to load rocBLAS. Safe no-op on non-CDNA3 archs (we don't use
    /// rocBLAS on RDNA — the hand-rolled kernels outperform it there).
    ///
    /// On success, sets `self.rocblas = Some(_)`; prefill dispatch paths can
    /// then route through MFMA-backed GEMM. On failure (library missing,
    /// symbol missing, handle init fail), logs once and leaves `None`.
    /// Callers always fall back to the non-rocBLAS path.
    pub fn try_init_rocblas(&mut self) {
        self.bind_thread_or_warn();
        if self.rocblas.is_some() {
            return;
        }
        let cdna3 = self.arch_caps.is_cdna3();
        let all_archs = self.flags.rocblas_all_archs;
        if !cdna3 && !all_archs {
            return;
        }
        match Rocblas::load() {
            Ok(rb) => {
                // Bind to the active stream if present; otherwise rocBLAS uses
                // the default (null) stream, which still works — just bigger
                // host-side sync cost.
                if let Some(stream) = self.active_stream.as_ref() {
                    if let Err(e) = rb.set_stream(stream) {
                        eprintln!(
                            "[rocblas] failed to bind active stream ({e}); using default stream"
                        );
                    }
                }
                eprintln!("[rocblas] loaded for {}", self.arch);
                self.rocblas = Some(rb);
            }
            Err(e) => {
                eprintln!(
                    "[rocblas] not available ({}); falling back to hand-rolled GEMMs",
                    e
                );
            }
        }
    }

    /// Dequantize an HFQ4-G256 weight [M × K] into an FP16 buffer [M × K]
    /// row-major. The FP16 buffer must be pre-allocated to M*K*2 bytes.
    ///
    /// Used as a one-shot model-load step on CDNA3 when the downstream
    /// prefill GEMM path is rocBLAS/hipBLASLt. Cost scales as O(MK) — for
    /// a 35B-A3B target at load time, ~10 GB dequantized; MI300X handles
    /// this in well under a second (the math is trivial, the launch is
    /// BW-bound at HBM3 write speed).
    pub fn dequantize_hfq4g256_to_f16(
        &mut self,
        w_mq4: &DeviceBuffer,
        w_fp16: &DeviceBuffer,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            k % 256 == 0,
            "hfq4g256 dequant: K must be multiple of 256 (got {k})"
        );
        self.ensure_kernel(
            "hfq4g256_dequantize_to_f16",
            kernels::HFQ4G256_DEQUANTIZE_TO_F16_SRC,
            "hfq4g256_dequantize_to_f16",
        )?;
        let func = &self.functions["hfq4g256_dequantize_to_f16"];
        let mut w_in = w_mq4.as_ptr();
        let mut w_out = w_fp16.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut w_in as *mut _ as *mut c_void,
            &mut w_out as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
        ];
        let groups = (k / 256) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, groups, 1],
                [128, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Dequantize MFP4G32-Lloyd matrix [M x K] to FP16 [M x K] row-major.
    /// Input `w_mq4` = full tensor bytes (32-B codebook prefix + M rows).
    /// Output `w_fp16` = FP16 row-major (M x K), in the ROTATED domain.
    /// Grid: [M, K/256]. Block: [32]. Mirrors dequantize_hfq4g256_to_f16 shape.
    pub fn dequantize_mfp4g32_lloyd_to_f16(
        &mut self,
        w_mq4: &DeviceBuffer,
        w_fp16: &DeviceBuffer,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            k % 256 == 0,
            "mfp4g32_lloyd dequant: K must be multiple of 256 (got {k})"
        );
        self.ensure_kernel(
            "dequantize_mfp4g32_lloyd_to_f16",
            kernels::DEQUANTIZE_MFP4G32_LLOYD_TO_F16_SRC,
            "dequantize_mfp4g32_lloyd_to_f16",
        )?;
        let func = &self.functions["dequantize_mfp4g32_lloyd_to_f16"];
        let mut w_in = w_mq4.as_ptr();
        let mut w_out = w_fp16.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut w_in as *mut _ as *mut c_void,
            &mut w_out as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
        ];
        let groups = (k / 256) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, groups, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Dequantize an mfp4+P matrix [M x K] to FP16 [M x K] row-major.
    /// Input `w_mq4` = full tensor bytes (M rows, NO prefix; byte-identical to mfp4).
    /// Output `w_fp16` = FP16 row-major (M x K), in the ROTATED domain.
    /// Decodes the per-block scale byte as E4M3 (FP8). Grid: [M, K/256]. Block: [32].
    pub fn dequantize_mfp4g32_p_to_f16(
        &mut self,
        w_mq4: &DeviceBuffer,
        w_fp16: &DeviceBuffer,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            k % 256 == 0,
            "mfp4g32_p dequant: K must be multiple of 256 (got {k})"
        );
        self.ensure_kernel(
            "dequantize_mfp4g32_p_to_f16",
            kernels::DEQUANTIZE_MFP4G32_P_TO_F16_SRC,
            "dequantize_mfp4g32_p_to_f16",
        )?;
        let func = &self.functions["dequantize_mfp4g32_p_to_f16"];
        let mut w_in = w_mq4.as_ptr();
        let mut w_out = w_fp16.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut w_in as *mut _ as *mut c_void,
            &mut w_out as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
        ];
        let groups = (k / 256) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, groups, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Dequantize an mfp4-E8 matrix [M x K] to FP16 [M x K] row-major.
    /// Input `w_mq4` = full tensor bytes (M rows, NO prefix; byte-identical footprint to mfp4+P).
    /// Output `w_fp16` = FP16 row-major (M x K), in the ROTATED domain.
    /// Decodes the per-block scale byte as E4M3 (FP8) * QUANT_STEP, then E8 coords.
    /// Grid: [M, K/256]. Block: [32].
    pub fn dequantize_mfp4g32_e8_to_f16(
        &mut self,
        w_mq4: &DeviceBuffer,
        w_fp16: &DeviceBuffer,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            k % 256 == 0,
            "mfp4g32_e8 dequant: K must be multiple of 256 (got {k})"
        );
        self.ensure_kernel(
            "dequantize_mfp4g32_e8_to_f16",
            kernels::DEQUANTIZE_MFP4G32_E8_TO_F16_SRC,
            "dequantize_mfp4g32_e8_to_f16",
        )?;
        let func = &self.functions["dequantize_mfp4g32_e8_to_f16"];
        let mut w_in = w_mq4.as_ptr();
        let mut w_out = w_fp16.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut w_in as *mut _ as *mut c_void,
            &mut w_out as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
        ];
        let groups = (k / 256) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, groups, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Expand qt=35 MFP4G32E8SOA rows into row-major FP16 on gfx942.
    ///
    /// Unlike [`Self::dequantize_mfp4g32_e8_to_f16`], this reads the SoA wire
    /// layout used by the frozen DeepSeek4 MQ2R dense tier. The output remains
    /// in the FWHT-rotated domain, matching the rotated activation supplied by
    /// the normal MFP4E8 projection path.
    pub fn dequantize_mfp4g32_e8_soa_to_f16_gfx942(
        &mut self,
        packed: &DeviceBuffer,
        expanded: &DeviceBuffer,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            self.arch_caps.is_gfx942(),
            "qt35 SoA staging requires gfx942"
        );
        assert!(
            k % 256 == 0,
            "qt35 SoA staging requires K%256==0, got K={k}"
        );
        const KERNEL: &str = "dequantize_mfp4g32_e8_soa_to_f16_gfx942";
        self.ensure_kernel(
            KERNEL,
            kernels::DEQUANTIZE_MFP4G32_E8_SOA_TO_F16_GFX942_SRC,
            KERNEL,
        )?;
        let packed_ptr = packed.as_ptr();
        let expanded_ptr = expanded.as_ptr();
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &packed_ptr as *const _ as *mut c_void,
            &expanded_ptr as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            KERNEL,
            [m as u32, (k / 32).div_ceil(16) as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(packed_ptr);
                blob.push_ptr(expanded_ptr);
                blob.push_i32(m_i32);
                blob.push_i32(k_i32);
                blob
            },
        )
    }

    /// Dense parent-checkpoint FP8 weight decode on gfx942.
    ///
    /// Expands `F8_E4M3 [M,K]` with `F8_E8M0 [ceil(M/128), ceil(K/128)]`
    /// block scales into row-major BF16 `[M,K]`. Model-lifetime staging —
    /// not a hot-path decode.
    ///
    /// Grid `[M,1,1]`, block `[256,1,1]` (one workgroup per output row).
    pub fn dequant_fp8_e4m3_ue8m0_blk128_to_bf16_gfx942(
        &mut self,
        w: &DeviceBuffer,
        s: &DeviceBuffer,
        out: &DeviceBuffer,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx942() {
            return Err(HipError::new(
                0,
                "dequant_fp8_e4m3_ue8m0_blk128_to_bf16_gfx942 requires gfx942",
            ));
        }
        if m == 0 || k == 0 {
            return Err(HipError::new(
                0,
                &format!(
                    "dequant_fp8_e4m3_ue8m0_blk128_to_bf16_gfx942: M and K must be positive (got M={m} K={k})"
                ),
            ));
        }
        let s_rows = m.div_ceil(128);
        let s_cols = k.div_ceil(128);
        let w_bytes = m
            .checked_mul(k)
            .ok_or_else(|| HipError::new(0, "dequant_fp8: M*K overflow"))?;
        let s_bytes = s_rows
            .checked_mul(s_cols)
            .ok_or_else(|| HipError::new(0, "dequant_fp8: scale shape overflow"))?;
        let out_bytes = w_bytes
            .checked_mul(2)
            .ok_or_else(|| HipError::new(0, "dequant_fp8: BF16 out size overflow"))?;
        if w.size() < w_bytes {
            return Err(HipError::new(
                0,
                &format!(
                    "dequant_fp8_e4m3_ue8m0_blk128_to_bf16_gfx942: w buffer too small (have {} need {w_bytes})",
                    w.size()
                ),
            ));
        }
        if s.size() < s_bytes {
            return Err(HipError::new(
                0,
                &format!(
                    "dequant_fp8_e4m3_ue8m0_blk128_to_bf16_gfx942: s buffer too small (have {} need {s_bytes})",
                    s.size()
                ),
            ));
        }
        if out.size() < out_bytes {
            return Err(HipError::new(
                0,
                &format!(
                    "dequant_fp8_e4m3_ue8m0_blk128_to_bf16_gfx942: out buffer too small (have {} need {out_bytes})",
                    out.size()
                ),
            ));
        }
        const KERNEL: &str = "dequant_fp8_e4m3_ue8m0_blk128_to_bf16_gfx942";
        self.ensure_kernel(
            KERNEL,
            kernels::DEQUANT_FP8_E4M3_UE8M0_BLK128_TO_BF16_GFX942_SRC,
            KERNEL,
        )?;
        let w_ptr = w.as_ptr();
        let s_ptr = s.as_ptr();
        let out_ptr = out.as_ptr();
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &w_ptr as *const _ as *mut c_void,
            &s_ptr as *const _ as *mut c_void,
            &out_ptr as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            KERNEL,
            [m as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(w_ptr);
                blob.push_ptr(s_ptr);
                blob.push_ptr(out_ptr);
                blob.push_i32(m_i32);
                blob.push_i32(k_i32);
                blob
            },
        )
    }

    /// Routed-expert parent-checkpoint FP4 weight decode on gfx942.
    ///
    /// Expands packed E2M1 `I8 [M, K/2]` with per-row UE8M0 scales
    /// `F8_E8M0 [M, K/32]` into row-major BF16 `[M,K]`. Hot path — one
    /// thread owns one 32-wide K-group.
    ///
    /// Grid `[M, ceil((K/32)/256), 1]`, block `[256,1,1]`.
    /// Requires `K % 32 == 0` (group size) and therefore even `K`.
    pub fn dequant_fp4_e2m1_ue8m0_g32_to_bf16_gfx942(
        &mut self,
        w: &DeviceBuffer,
        s: &DeviceBuffer,
        out: &DeviceBuffer,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx942() {
            return Err(HipError::new(
                0,
                "dequant_fp4_e2m1_ue8m0_g32_to_bf16_gfx942 requires gfx942",
            ));
        }
        if m == 0 || k == 0 {
            return Err(HipError::new(
                0,
                &format!(
                    "dequant_fp4_e2m1_ue8m0_g32_to_bf16_gfx942: M and K must be positive (got M={m} K={k})"
                ),
            ));
        }
        if k % 32 != 0 {
            return Err(HipError::new(
                0,
                &format!(
                    "dequant_fp4_e2m1_ue8m0_g32_to_bf16_gfx942: K must be a multiple of 32 (got K={k})"
                ),
            ));
        }
        let n_groups = k / 32;
        let w_bytes = m
            .checked_mul(k / 2)
            .ok_or_else(|| HipError::new(0, "dequant_fp4: packed W size overflow"))?;
        let s_bytes = m
            .checked_mul(n_groups)
            .ok_or_else(|| HipError::new(0, "dequant_fp4: scale size overflow"))?;
        let out_bytes = m
            .checked_mul(k)
            .and_then(|e| e.checked_mul(2))
            .ok_or_else(|| HipError::new(0, "dequant_fp4: BF16 out size overflow"))?;
        if w.size() < w_bytes {
            return Err(HipError::new(
                0,
                &format!(
                    "dequant_fp4_e2m1_ue8m0_g32_to_bf16_gfx942: w buffer too small (have {} need {w_bytes})",
                    w.size()
                ),
            ));
        }
        if s.size() < s_bytes {
            return Err(HipError::new(
                0,
                &format!(
                    "dequant_fp4_e2m1_ue8m0_g32_to_bf16_gfx942: s buffer too small (have {} need {s_bytes})",
                    s.size()
                ),
            ));
        }
        if out.size() < out_bytes {
            return Err(HipError::new(
                0,
                &format!(
                    "dequant_fp4_e2m1_ue8m0_g32_to_bf16_gfx942: out buffer too small (have {} need {out_bytes})",
                    out.size()
                ),
            ));
        }
        const KERNEL: &str = "dequant_fp4_e2m1_ue8m0_g32_to_bf16_gfx942";
        self.ensure_kernel(
            KERNEL,
            kernels::DEQUANT_FP4_E2M1_UE8M0_G32_TO_BF16_GFX942_SRC,
            KERNEL,
        )?;
        let w_ptr = w.as_ptr();
        let s_ptr = s.as_ptr();
        let out_ptr = out.as_ptr();
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &w_ptr as *const _ as *mut c_void,
            &s_ptr as *const _ as *mut c_void,
            &out_ptr as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
        ];
        let groups_y = n_groups.div_ceil(256) as u32;
        self.launch_maybe_blob(
            KERNEL,
            [m as u32, groups_y, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(w_ptr);
                blob.push_ptr(s_ptr);
                blob.push_ptr(out_ptr);
                blob.push_i32(m_i32);
                blob.push_i32(k_i32);
                blob
            },
        )
    }

    /// Parent-checkpoint FP8 activation quant simulation (fused quant+dequant).
    ///
    /// In-place over a row-major BF16 buffer shaped `[rows, last_dim]`.
    /// Groups of `block` contiguous elements along the last dim use a single
    /// UE8M0 power-of-two scale (`fast_round_scale`) and round through OCP
    /// E4M3 RNE before writing BF16(`e4m3_to_f32(q) * s`) back.
    ///
    /// `block` must be 64 (KV/compressor non-RoPE sites) or 128 (linear
    /// boundaries). `last_dim % block == 0` is required. Fail-closed on
    /// anything else — no MQ2R / Raw fallback.
    ///
    /// Launch: grid `[rows, last_dim/block, 1]`, block `[64,1,1]` (one wave
    /// per group).
    pub fn act_quant_fp8_ue8m0_inplace_gfx942(
        &mut self,
        x: &DeviceBuffer,
        rows: usize,
        last_dim: usize,
        block: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx942() {
            return Err(HipError::new(
                0,
                "act_quant_fp8_ue8m0_inplace_gfx942 requires gfx942",
            ));
        }
        if block != 64 && block != 128 {
            return Err(HipError::new(
                0,
                &format!(
                    "act_quant_fp8_ue8m0_inplace_gfx942: block must be 64 or 128 (got {block})"
                ),
            ));
        }
        if rows == 0 || last_dim == 0 {
            return Err(HipError::new(
                0,
                &format!(
                    "act_quant_fp8_ue8m0_inplace_gfx942: rows and last_dim must be positive (got rows={rows} last_dim={last_dim})"
                ),
            ));
        }
        if last_dim % block != 0 {
            return Err(HipError::new(
                0,
                &format!(
                    "act_quant_fp8_ue8m0_inplace_gfx942: last_dim must be a multiple of block (got last_dim={last_dim} block={block})"
                ),
            ));
        }
        let need_bytes = rows
            .checked_mul(last_dim)
            .and_then(|e| e.checked_mul(2))
            .ok_or_else(|| HipError::new(0, "act_quant_fp8_ue8m0_inplace_gfx942: size overflow"))?;
        if x.size() < need_bytes {
            return Err(HipError::new(
                0,
                &format!(
                    "act_quant_fp8_ue8m0_inplace_gfx942: buffer too small (have {} need {need_bytes} for {rows}×{last_dim} BF16)",
                    x.size()
                ),
            ));
        }
        const KERNEL: &str = "act_quant_fp8_ue8m0_inplace_gfx942";
        self.ensure_kernel(
            KERNEL,
            kernels::ACT_QUANT_FP8_UE8M0_INPLACE_GFX942_SRC,
            KERNEL,
        )?;
        let x_ptr = x.as_ptr();
        let rows_i32 = rows as i32;
        let last_dim_i32 = last_dim as i32;
        let block_i32 = block as i32;
        let mut params: Vec<*mut c_void> = vec![
            &x_ptr as *const _ as *mut c_void,
            &rows_i32 as *const _ as *mut c_void,
            &last_dim_i32 as *const _ as *mut c_void,
            &block_i32 as *const _ as *mut c_void,
        ];
        let n_groups = last_dim / block;
        self.launch_maybe_blob(
            KERNEL,
            [rows as u32, n_groups as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(x_ptr);
                blob.push_i32(rows_i32);
                blob.push_i32(last_dim_i32);
                blob.push_i32(block_i32);
                blob
            },
        )
    }

    /// Parent-checkpoint FP4 activation quant simulation (fused quant+dequant).
    ///
    /// In-place over a row-major BF16 buffer shaped `[rows, last_dim]`.
    /// Groups of 32 along the last dim; UE8M0 power-of-two scale via
    /// `fast_round_scale(amax, 1/6)`, then E2M1 RNE onto
    /// `{0,.5,1,1.5,2,3,4,6}` (sign preserved) and BF16 write-back.
    ///
    /// `last_dim % 32 == 0` is required. Fail-closed otherwise.
    ///
    /// Launch: grid `[rows, last_dim/32, 1]`, block `[32,1,1]` (one group
    /// per 32-lane cohort).
    pub fn act_quant_fp4_ue8m0_g32_inplace_gfx942(
        &mut self,
        x: &DeviceBuffer,
        rows: usize,
        last_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx942() {
            return Err(HipError::new(
                0,
                "act_quant_fp4_ue8m0_g32_inplace_gfx942 requires gfx942",
            ));
        }
        if rows == 0 || last_dim == 0 {
            return Err(HipError::new(
                0,
                &format!(
                    "act_quant_fp4_ue8m0_g32_inplace_gfx942: rows and last_dim must be positive (got rows={rows} last_dim={last_dim})"
                ),
            ));
        }
        if last_dim % 32 != 0 {
            return Err(HipError::new(
                0,
                &format!(
                    "act_quant_fp4_ue8m0_g32_inplace_gfx942: last_dim must be a multiple of 32 (got last_dim={last_dim})"
                ),
            ));
        }
        let need_bytes = rows
            .checked_mul(last_dim)
            .and_then(|e| e.checked_mul(2))
            .ok_or_else(|| {
                HipError::new(0, "act_quant_fp4_ue8m0_g32_inplace_gfx942: size overflow")
            })?;
        if x.size() < need_bytes {
            return Err(HipError::new(
                0,
                &format!(
                    "act_quant_fp4_ue8m0_g32_inplace_gfx942: buffer too small (have {} need {need_bytes} for {rows}×{last_dim} BF16)",
                    x.size()
                ),
            ));
        }
        const KERNEL: &str = "act_quant_fp4_ue8m0_g32_inplace_gfx942";
        self.ensure_kernel(
            KERNEL,
            kernels::ACT_QUANT_FP4_UE8M0_G32_INPLACE_GFX942_SRC,
            KERNEL,
        )?;
        let x_ptr = x.as_ptr();
        let rows_i32 = rows as i32;
        let last_dim_i32 = last_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &x_ptr as *const _ as *mut c_void,
            &rows_i32 as *const _ as *mut c_void,
            &last_dim_i32 as *const _ as *mut c_void,
        ];
        let n_groups = last_dim / 32;
        self.launch_maybe_blob(
            KERNEL,
            [rows as u32, n_groups as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(x_ptr);
                blob.push_i32(rows_i32);
                blob.push_i32(last_dim_i32);
                blob
            },
        )
    }

    /// D→D copy with offsets that picks async (on the active stream) when
    /// a stream is set and sync otherwise. Captured graphs require async on
    /// the captured stream — sync `hipMemcpy` errors with "would make the
    /// legacy stream depend on a capturing blocking stream" under capture
    /// mode Global. Use this helper whenever the copy might live inside
    /// a captured region.
    /// `HIPFIRE_DTOD_DUMP=1` prints `file:line` and size per D→D copy.
    ///
    /// Mirrors `HIPFIRE_MEMSET_DUMP`. Added because `__amd_rocclr_copyBuffer`
    /// costs 0.378 ms/step (19.8 calls) in the ds4 gfx1151 AR decode — a 64 KB
    /// copy taking 19.11 us against 0.33 us of actual traffic, 57x off — and
    /// there are a dozen `memcpy_dtod_auto` call sites in the ds4 forward with
    /// no way to tell which ones fire. Grep the dump by source location.
    #[track_caller]
    pub fn memcpy_dtod_at_auto(
        &self,
        dst: &hip_bridge::DeviceBuffer,
        dst_offset: usize,
        src: &hip_bridge::DeviceBuffer,
        src_offset: usize,
        size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        static DUMP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let dump = *DUMP.get_or_init(|| {
            hipfire_config::developer_var("HIPFIRE_DTOD_DUMP")
                .ok()
                .as_deref()
                == Some("1")
        });
        if dump {
            let loc = std::panic::Location::caller();
            eprintln!("dtod bytes={} at {}:{}", size, loc.file(), loc.line());
        }
        if let Some(stream) = self.active_stream.as_ref() {
            self.hip
                .memcpy_dtod_async_at(dst, dst_offset, src, src_offset, size, stream)
        } else {
            self.hip
                .memcpy_dtod_at(dst, dst_offset, src, src_offset, size)
        }
    }

    /// D→D copy with offsets that is always host-asynchronous: uses the
    /// active stream when set, otherwise the legacy/default stream via
    /// `memcpy_dtod_async_default_at`. Prefer this on non-capture hot
    /// paths where `memcpy_dtod_at_auto` would fall back to a host-sync
    /// copy. Capture paths still need an explicit active stream (same as
    /// `memcpy_dtod_at_auto`).
    /// `HIPFIRE_DTOD_DUMP=1` prints `file:line` and size per D→D copy.
    #[track_caller]
    pub fn memcpy_dtod_at_ordered_async(
        &self,
        dst: &hip_bridge::DeviceBuffer,
        dst_offset: usize,
        src: &hip_bridge::DeviceBuffer,
        src_offset: usize,
        size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        static DUMP: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            hipfire_config::developer_var("HIPFIRE_DTOD_DUMP")
                .ok()
                .as_deref()
                == Some("1")
        });
        let dump = *DUMP;
        if dump {
            let loc = std::panic::Location::caller();
            eprintln!("dtod bytes={} at {}:{}", size, loc.file(), loc.line());
        }
        if let Some(stream) = self.active_stream.as_ref() {
            self.hip
                .memcpy_dtod_async_at(dst, dst_offset, src, src_offset, size, stream)
        } else {
            self.hip
                .memcpy_dtod_async_default_at(dst, dst_offset, src, src_offset, size)
        }
    }

    /// D→D copy (whole buffer) that picks async on the active stream when set.
    ///
    /// `#[track_caller]` so `HIPFIRE_DTOD_DUMP` attributes the copy to the real
    /// call site rather than to this forwarder.
    #[track_caller]
    pub fn memcpy_dtod_auto(
        &self,
        dst: &hip_bridge::DeviceBuffer,
        src: &hip_bridge::DeviceBuffer,
        size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.memcpy_dtod_at_auto(dst, 0, src, 0, size)
    }

    /// H→D copy that picks async on the active stream when capturing.
    ///
    /// During hipGraph capture (`capture_mode == true`), operations on the
    /// legacy/null stream are forbidden because they would create a blocking
    /// dependency with the capturing stream. This method routes to
    /// `memcpy_htod_async` on the active (capturing) stream when in capture
    /// mode, falling back to sync `memcpy_htod` otherwise.
    pub fn memcpy_htod_auto(&self, dst: &hip_bridge::DeviceBuffer, src: &[u8]) -> HipResult<()> {
        self.bind_thread()?;
        if self.graphs.capture_mode {
            let stream = self
                .active_stream
                .as_ref()
                .expect("capture mode requires an active stream");
            self.hip.memcpy_htod_async(dst, src, stream)
        } else {
            self.hip.memcpy_htod(dst, src)
        }
    }

    /// Helper: launch a kernel using the blob path during graph capture,
    /// or the normal kernelParams path otherwise. The `blob_builder` closure
    /// constructs the KernargBlob; it's only called when capturing.
    pub(crate) fn launch_maybe_blob(
        &mut self,
        func_name: &str,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        params: &mut [*mut std::ffi::c_void],
        blob_builder: impl FnOnce() -> hip_bridge::KernargBlob,
    ) -> HipResult<()> {
        self.launch_maybe_blob_bound(
            func_name,
            grid,
            block,
            shared_mem,
            params,
            None,
            blob_builder,
        )
    }

    /// Record a position-derived PM4 workgroup binding while retaining the
    /// recorded maximum grid for HIP, hipGraph, AQL, and capture validation.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_maybe_blob_position_grid(
        &mut self,
        func_name: &str,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        params: &mut [*mut std::ffi::c_void],
        axis: u8,
        addend: u32,
        divisor: u32,
        blob_builder: impl FnOnce() -> hip_bridge::KernargBlob,
    ) -> HipResult<()> {
        let grid_binding = pm4_dynamic_grid_enabled().then_some(
            crate::replay::ReplayGridBinding::PositionCeilDiv {
                axis,
                addend,
                divisor,
            },
        );
        self.launch_maybe_blob_bound(
            func_name,
            grid,
            block,
            shared_mem,
            params,
            grid_binding,
            blob_builder,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_maybe_blob_bound(
        &mut self,
        func_name: &str,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        params: &mut [*mut std::ffi::c_void],
        grid_binding: Option<crate::replay::ReplayGridBinding>,
        blob_builder: impl FnOnce() -> hip_bridge::KernargBlob,
    ) -> HipResult<()> {
        let record = self.replay.is_recording();
        if record || self.graphs.capture_mode || self.flags.force_blob_path {
            let mut blob = blob_builder();
            blob.pad_to(16);
            if record {
                let artifact = self
                    .compiler
                    .compiled_kernels()
                    .get(func_name)
                    .or_else(|| match func_name {
                        "mq_rotate_x" => self.compiler.compiled_kernels().get("gemv_mq4g256"),
                        "deinterleave_f32_batched" => {
                            self.compiler.compiled_kernels().get("deinterleave_batched")
                        }
                        name if name.starts_with("gemv_hfq4g256_residual_sigmoid_scaled_gpu") => {
                            self.compiler
                                .compiled_kernels()
                                .get("gemv_hfq4g256_residual_scaled")
                        }
                        "gemv_hfq4g256_moe_gate_up_k8_indexed" => self
                            .compiler
                            .compiled_kernels()
                            .get("gemv_hfq4g256_moe_gate_up_indexed"),
                        name if name.starts_with("gemv_hfq4g256_multirow_r") => self
                            .compiler
                            .compiled_kernels()
                            .get("gemv_hfq4g256_multirow_default")
                            .or_else(|| {
                                self.compiler
                                    .compiled_kernels()
                                    .get("gemv_hfq4g256_multirow_rdna3")
                            }),
                        name if name.starts_with("gemv_hfq4g256_residual_multirow_r") => self
                            .compiler
                            .compiled_kernels()
                            .get("gemv_hfq4g256_residual_multirow_default")
                            .or_else(|| {
                                self.compiler
                                    .compiled_kernels()
                                    .get("gemv_hfq4g256_residual_multirow_rdna3")
                            }),
                        _ => None,
                    })
                    .or_else(|| {
                        func_name
                            .strip_suffix("_f32")
                            .and_then(|name| self.compiler.compiled_kernels().get(name))
                    })
                    .cloned();
                self.replay.record_hip_launch_typed_bound(
                    &self.hip,
                    func_name,
                    artifact,
                    grid,
                    block,
                    shared_mem,
                    blob.as_bytes(),
                    grid_binding,
                );
            }
            let func = &self.functions[func_name];
            if self.graphs.capture_mode {
                self.graphs.capture_blobs.push(blob.into_vec());
                let buf = self.graphs.capture_blobs.last_mut().unwrap();
                // SAFETY: every caller's builder encodes the same argument
                // values supplied by `params`; graph-owned storage stays live
                // through graph instantiation and replay.
                unsafe {
                    self.hip.launch_kernel_blob(
                        func,
                        grid,
                        block,
                        shared_mem,
                        self.active_stream.as_ref(),
                        buf.as_mut_slice(),
                    )
                }
            } else {
                let mut bytes = blob.into_vec();
                // SAFETY: HIP consumes the contiguous argument bytes during
                // this one-shot launch; `bytes` remains live across the call.
                unsafe {
                    self.hip.launch_kernel_blob(
                        func,
                        grid,
                        block,
                        shared_mem,
                        self.active_stream.as_ref(),
                        bytes.as_mut_slice(),
                    )
                }
            }
        } else {
            let func = &self.functions[func_name];
            // SAFETY: forwarded from the typed launch wrapper that assembled
            // `params` for this kernel signature.
            unsafe {
                self.hip.launch_kernel(
                    func,
                    grid,
                    block,
                    shared_mem,
                    self.active_stream.as_ref(),
                    params,
                )
            }
        }
    }

    /// Diagnostic oracle for Redline prefix localization: relaunch the exact
    /// captured HIP blob sequence without re-entering model dispatch logic.
    /// Inputs/state must be restored by the caller first.
    pub fn replay_recorded_hip_prefix(&self, count: usize) -> HipResult<()> {
        self.bind_thread()?;
        if count > self.replay.recorded_launches().len() {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "captured HIP prefix {count} exceeds {} launches",
                    self.replay.recorded_launches().len()
                ),
            ));
        }
        for launch in self.replay.recorded_launches().iter().take(count) {
            let func = self.functions.get(&launch.kernel).ok_or_else(|| {
                hip_bridge::HipError::new(
                    0,
                    &format!("captured HIP function {:?} is not loaded", launch.kernel),
                )
            })?;
            let mut kernarg = launch.kernarg.clone();
            // SAFETY: the bytes were captured from this exact loaded function
            // and all pointees remain owned by this Gpu/model instance.
            unsafe {
                self.hip.launch_kernel_blob(
                    func,
                    launch.grid,
                    launch.block,
                    launch.shared_mem,
                    self.active_stream.as_ref(),
                    &mut kernarg,
                )?;
            }
        }
        Ok(())
    }

    /// Compile and load a kernel if missing. Public variant of `ensure_kernel`
    /// for callers that need to JIT a kernel by name from outside the crate
    /// (primarily the hipGraph capture/replay path).
    pub fn ensure_kernel_public(
        &mut self,
        module_name: &str,
        source: &str,
        func_name: &str,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(module_name, source, func_name)
    }

    /// Launch a pre-loaded kernel by name using the `extra`-mode kernarg
    /// blob path. This is the only launch path that survives hipGraph
    /// capture on gfx1100 / ROCm 6.x — the traditional `kernelParams`
    /// (`void**`) path records stack pointers that dangle by the time the
    /// captured graph is replayed.
    ///
    /// Caller is responsible for:
    ///  - keeping `kernargs` alive across the life of any graph that
    ///    captured this launch (HIP records the blob pointer, not the data);
    ///  - building `kernargs` with the layout matching the kernel signature
    ///    (use `hip_bridge::KernargBlob` for correct alignment).
    pub fn launch_kernel_blob(
        &self,
        func_name: &str,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        kernargs: &mut [u8],
    ) -> HipResult<()> {
        self.bind_thread()?;
        let func = self.functions.get(func_name).ok_or_else(|| {
            hip_bridge::HipError::new(
                0,
                &format!("launch_kernel_blob: function '{func_name}' not loaded"),
            )
        })?;
        unsafe {
            self.hip
                .launch_kernel_blob(func, grid, block, shared_mem, self.stream_ref(), kernargs)
        }
    }

    /// Compile and load a kernel, caching the result.
    pub(crate) fn ensure_kernel(
        &mut self,
        module_name: &str,
        source: &str,
        func_name: &str,
    ) -> HipResult<()> {
        crate::scratch::compile_and_load_kernel(
            &mut self.compiler,
            &self.hip,
            &mut self.modules,
            &mut self.functions,
            module_name,
            source,
            func_name,
        )
    }

    /// Ensure the FP16 X scratch contains the conversion of `x`. Skips the
    /// convert kernel if `x.buf.as_ptr()` matches the last converted source.
    /// Returns the FP16 device pointer.
    pub(crate) fn ensure_fp16_x(
        &mut self,
        x: &GpuTensor,
        n_elems: usize,
    ) -> HipResult<*mut c_void> {
        self.scratch.ensure_fp16_x(
            &self.hip,
            &mut self.compiler,
            &mut self.modules,
            &mut self.functions,
            self.active_stream.as_ref(),
            &mut self.graphs.capture_blobs,
            self.graphs.capture_mode,
            self.flags.force_blob_path,
            x,
            n_elems,
        )
    }

    /// Ensure the deterministic-ksplit partials scratch is at least `n_bytes`.
    pub(crate) fn ensure_ksplit_det_partials(&mut self, n_bytes: usize) -> HipResult<*mut c_void> {
        self.scratch.ensure_ksplit_det_partials(&self.hip, n_bytes)
    }

    /// Convert F32 to F16 without caching. Used when the same x tensor
    /// pointer is reused with different contents across layers, where
    /// pointer-keyed caching would read stale FP16.
    pub(crate) fn convert_fp16_x_uncached(
        &mut self,
        x: &GpuTensor,
        n_elems: usize,
    ) -> HipResult<*mut c_void> {
        self.scratch.convert_fp16_x_uncached(
            &self.hip,
            &mut self.compiler,
            &mut self.modules,
            &mut self.functions,
            self.active_stream.as_ref(),
            &mut self.graphs.capture_blobs,
            self.graphs.capture_mode,
            self.flags.force_blob_path,
            x,
            n_elems,
        )
    }

    /// Ensure the FP8 (E4M3) X scratch contains the conversion of `x`
    /// (an F32 GpuTensor). Returns the FP8 device pointer. gfx12 only —
    /// uses cvt_pk_fp8_f32. Caches by `x.buf.as_ptr()` like its FP16
    /// sibling so back-to-back same-X GEMM dispatches skip reconversion.
    pub(crate) fn ensure_fp8_x(&mut self, x: &GpuTensor, n_elems: usize) -> HipResult<*mut c_void> {
        self.scratch.ensure_fp8_x(
            &self.hip,
            &mut self.compiler,
            &mut self.modules,
            &mut self.functions,
            self.active_stream.as_ref(),
            &mut self.graphs.capture_blobs,
            self.graphs.capture_mode,
            self.flags.force_blob_path,
            x,
            n_elems,
        )
    }

    /// Ensure prefill activations are quantized into a llama.cpp-style
    /// `block_q8_1_mmq` layout. The scratch is ordered by [K/128 block, batch]
    /// so a 128-column batch tile is contiguous for each K tile.
    pub fn ensure_q8_1_mmq_x(
        &mut self,
        x: &GpuTensor,
        batch_size: usize,
        k: usize,
    ) -> HipResult<*mut c_void> {
        // bind_thread: skip — delegated to scratch.rs
        self.scratch.ensure_q8_1_mmq_x(
            &self.hip,
            &mut self.compiler,
            &mut self.modules,
            &mut self.functions,
            self.active_stream.as_ref(),
            &mut self.graphs.capture_blobs,
            self.graphs.capture_mode,
            self.flags.force_blob_path,
            self.device_id,
            x,
            batch_size,
            k,
        )
    }

    /// Screen a weight matrix for MMQ safety (#87). Runs a small synthetic
    /// comparison (batch=16): f16 WMMA vs MMQ on random activations. If any
    /// output row's max abs error exceeds `mmq_screen_threshold`, the weight
    /// is marked unsafe. Result is cached by device pointer.
    ///
    /// Returns `true` if MMQ is safe for this weight, `false` if it should
    /// fall back to WMMA.
    pub fn mmq_screen_weight(&mut self, a_raw: &GpuTensor, m: usize, k: usize) -> bool {
        self.bind_thread_or_warn();
        let key = a_raw.buf.as_ptr() as usize;
        if let Some(&safe) = self.mmq_screen.cache.get(&key) {
            return safe;
        }

        let screen_batch = 16usize;
        let threshold = self.mmq_screen.threshold;

        // Generate synthetic activations on CPU
        let mut state = 0xDEAD_BEEF_CAFE_BABEu64;
        let x_data: Vec<f32> = (0..screen_batch * k)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let t = (state >> 33) as f32 / (u32::MAX as f32);
                t * 4.0 - 2.0
            })
            .collect();

        let result = (|| -> HipResult<bool> {
            let x_gpu = self.upload_f32(&x_data, &[screen_batch * k])?;
            let y_wmma = self.zeros(&[screen_batch * m], DType::F32)?;
            let y_mmq = self.zeros(&[screen_batch * m], DType::F32)?;

            let saved_capture = self.graphs.capture_mode;
            self.graphs.capture_mode = true;

            // Reference path: use FP16 wave64 on gfx906, WMMA otherwise
            if self.arch_caps.is_gfx906() {
                self.gemm_hfq4g256_residual_fp16_wave64(
                    a_raw,
                    &x_gpu,
                    &y_wmma,
                    m,
                    k,
                    screen_batch,
                )?;
            } else {
                self.gemm_hfq4g256_residual_wmma(a_raw, &x_gpu, &y_wmma, m, k, screen_batch)?;
            }

            // MMQ path
            let xq = self.ensure_q8_1_mmq_x(&x_gpu, screen_batch, k)?;
            if self.arch_caps.is_gfx906() {
                self.gemm_hfq4g256_residual_mmq_gfx906(a_raw, &x_gpu, &y_mmq, m, k, screen_batch)?;
            } else {
                self.gemm_hfq4g256_mmq_set_prequant(a_raw, xq, &y_mmq, m, k, screen_batch)?;
            }

            self.graphs.capture_mode = saved_capture;
            self.hip.device_synchronize()?;

            let ref_out = self.download_f32(&y_wmma)?;
            let mmq_out = self.download_f32(&y_mmq)?;

            self.free_tensor(x_gpu).ok();
            self.free_tensor(y_wmma).ok();
            self.free_tensor(y_mmq).ok();

            // Per-row max error check
            let mut worst_err = 0f32;
            for r in 0..m {
                let mut row_max = 0f32;
                for b in 0..screen_batch {
                    let idx = b * m + r;
                    let err = (ref_out[idx] - mmq_out[idx]).abs();
                    if err > row_max {
                        row_max = err;
                    }
                }
                if row_max > worst_err {
                    worst_err = row_max;
                }
            }

            let safe = worst_err <= threshold;
            Ok(safe)
        })();

        let safe = result.unwrap_or_else(|e| {
            eprintln!("  MMQ screen: error during screening ({e}), assuming unsafe");
            false
        });
        self.mmq_screen.cache.insert(key, safe);
        safe
    }

    /// Ensure an FP16 shadow of `w_mq4` (HFQ4-G256 format, [M × K]) exists in
    /// `fp16_shadow_cache`. First call allocates M*K*2 bytes on device and
    /// runs the dequantize kernel; subsequent calls return the cached pointer.
    ///
    /// Cache is keyed on the MQ4 device pointer — this assumes weights are
    /// immutable after model load (standard in this engine). If the same
    /// pointer is ever reused for a different M or K, cache would return
    /// stale data: we don't try to detect that (weights don't reshape).
    ///
    /// Returns `None` if rocBLAS is not loaded (caller should fall back to
    /// the hand-rolled GEMV path). Memory is freed when the Gpu drops.
    pub(crate) fn ensure_fp16_shadow(
        &mut self,
        w_mq4: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<Option<*mut c_void>> {
        if self.rocblas.is_none() {
            return Ok(None);
        }
        let key = w_mq4.buf.as_ptr() as usize;
        if let Some(shadow) = self.fp16_shadow_cache.get(&key) {
            return Ok(Some(shadow.buf.as_ptr()));
        }
        // Allocate + dequantize. Use alloc_tensor so the shadow follows the
        // same GpuTensor hygiene (tracked in pool if applicable).
        let fp16 = self.alloc_tensor(&[m * k], DType::F16)?;
        self.dequantize_hfq4g256_to_f16(&w_mq4.buf, &fp16.buf, m, k)?;
        let ptr = fp16.buf.as_ptr();
        self.fp16_shadow_cache.insert(key, fp16);
        Ok(Some(ptr))
    }

    /// Ensure a model-lifetime FP16 shadow of a qt=35 MFP4G32E8SOA matrix.
    /// This entry is deliberately separate from the HFQ4 helper so adding the
    /// DeepSeek4 CDNA path cannot change any existing Qwen/HFQ4 dequant route.
    pub(crate) fn ensure_mfp4e8_soa_fp16_shadow_gfx942(
        &mut self,
        weight: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<Option<*mut c_void>> {
        if self.arch != "gfx942" || self.rocblas.is_none() {
            return Ok(None);
        }
        debug_assert_eq!(weight.dtype, DType::MFP4G32E8SOA);
        let key = weight.buf.as_ptr() as usize;
        if let Some(shadow) = self.fp16_shadow_cache.get(&key) {
            return Ok(Some(shadow.buf.as_ptr()));
        }
        let fp16 = self.alloc_tensor(&[m * k], DType::F16)?;
        self.dequantize_mfp4g32_e8_soa_to_f16_gfx942(&weight.buf, &fp16.buf, m, k)?;
        let ptr = fp16.buf.as_ptr();
        self.fp16_shadow_cache.insert(key, fp16);
        Ok(Some(ptr))
    }

    /// Whether the arch is eligible for the rocBLAS/MFMA batched-prefill
    /// path. Default: CDNA3 only (MI300-series, gfx94x). Override with
    /// `HIPFIRE_ROCBLAS_ALL_ARCHS=1` for local testing on RDNA3+ — rocBLAS
    /// runs fine there (uses WMMA backends on RDNA3, not MFMA) so this is
    /// a useful smoke-path in the absence of an MI300.
    pub(crate) fn rocblas_arch_eligible(&self) -> bool {
        static CACHE: OnceLock<bool> = OnceLock::new();
        let all_archs = *CACHE.get_or_init(|| self.flags.rocblas_all_archs);
        if all_archs {
            return self.rocblas.is_some();
        }
        self.arch_caps.is_cdna3()
    }

    /// Configurable batch threshold for MFMA dispatch. Below this we stay on
    /// the hand-rolled GEMV — rocBLAS launch overhead eats the compute win
    /// at tiny batches. Overridable via `HIPFIRE_ROCBLAS_MIN_BATCH` env var.
    ///
    /// Kill-switch: `HIPFIRE_ROCBLAS_OFF=1` forces the threshold to usize::MAX,
    /// which disables the rocBLAS path entirely for A/B benchmarking against
    /// the hand-rolled GEMV baseline.
    pub(crate) fn rocblas_min_batch(&self) -> usize {
        static CACHE: OnceLock<usize> = OnceLock::new();
        *CACHE.get_or_init(|| {
            if self.flags.rocblas_off {
                return usize::MAX;
            }
            self.flags.rocblas_min_batch.unwrap_or(4)
        })
    }

    /// Batched-attention tile size, from `HIPFIRE_ATTN_TILE_SIZE`.
    ///
    /// Falls back to 128 when unset, zero, or not a multiple of 32. This is the
    /// single resolver for the whole crate — three hand-mirrored copies of this
    /// logic previously caused silent device-memory corruption when one of them
    /// drifted (see docs/plans/2026-08-07-batched-attention-slots.md).
    ///
    /// Directional safety note: RAISING the tile is always safe. LOWERING it
    /// increases `max_tiles` and therefore the `partials` bytes each query row
    /// needs, which can exceed buffers sized elsewhere against the 128 default.
    pub fn attn_tile_size(&self) -> usize {
        // bind_thread: skip — pure flag read, touches no device state.
        static CACHE: OnceLock<usize> = OnceLock::new();
        *CACHE.get_or_init(|| {
            self.flags
                .attn_tile_size
                .filter(|&t| t > 0 && t % 32 == 0)
                .unwrap_or(128)
        })
    }

    /// Multi-slot attention flash-vs-scalar crossover override, in tokens.
    /// `None` leaves the per-arch default in place.
    pub fn slots_attn_crossover(&self) -> Option<usize> {
        // bind_thread: skip — pure flag read, touches no device state.
        self.flags.slots_attn_crossover
    }

    /// Pre-compile a batch of kernels in parallel (hipcc), then load modules + functions.
    /// Each entry is (module_name, source, func_name). Turbo kernels should have
    /// TURBO_COMMON_H already prepended in their source.
    pub fn precompile_kernels(&mut self, specs: &[(&str, &str, &str)]) -> HipResult<()> {
        self.bind_thread()?;
        // Collect (name, source) pairs for the compiler batch, skipping already-loaded
        let batch: Vec<(&str, &str)> = specs
            .iter()
            .filter(|(_, _, func)| !self.functions.contains_key(*func))
            .map(|(module, source, _)| (*module, *source))
            .collect();

        if batch.is_empty() {
            return Ok(());
        }

        // Parallel hipcc compilation
        self.compiler.compile_batch(&batch)?;

        // Now load modules + extract functions (must be sequential — GPU API calls)
        for &(module_name, source, func_name) in specs {
            if self.functions.contains_key(func_name) {
                continue;
            }
            let obj_path = self.compiler.compile(module_name, source)?;
            let obj_path_str = obj_path.to_str().unwrap().to_string();
            if !self.modules.contains_key(module_name) {
                let module = crate::scratch::module_load_or_recompile(
                    &self.hip,
                    &mut self.compiler,
                    module_name,
                    source,
                    &obj_path_str,
                )?;
                self.modules.insert(module_name.to_string(), module);
            }
            let module = &self.modules[module_name];
            let func = self.hip.module_get_function(module, func_name)?;
            self.functions.insert(func_name.to_string(), func);
        }
        Ok(())
    }

    // ── Tensor allocation ───────────────────────────────────────

    pub fn ensure_gemv_residual_tmp(&mut self, min_elems: usize) -> HipResult<&GpuTensor> {
        self.scratch
            .ensure_gemv_residual_tmp(&self.hip, self.device_id, min_elems)
    }

    pub fn alloc_tensor(&mut self, shape: &[usize], dtype: DType) -> HipResult<GpuTensor> {
        self.bind_thread()?;
        let numel: usize = shape.iter().product();
        let byte_size = numel * dtype.size();
        let buf = self.pool.alloc(&self.hip, byte_size)?;
        Ok(GpuTensor {
            buf,
            shape: shape.to_vec(),
            dtype,
        })
    }

    /// Reserve a dense virtual tensor and optionally map its initial prefix.
    /// The returned tensor follows the normal kernel ABI, but its ownership is
    /// registered separately so `free_tensor` releases VMM handles instead of
    /// routing the address through `hipFree`.
    ///
    /// # Safety
    ///
    /// The returned tensor describes its full reserved shape. The caller must
    /// keep every operation within the prefix reported by `vmm_mapped_bytes`,
    /// growing the mapping before any wider access.
    pub unsafe fn alloc_vmm_tensor(
        &mut self,
        shape: &[usize],
        dtype: DType,
        initial_mapped_bytes: usize,
        access_devices: &[i32],
    ) -> HipResult<GpuTensor> {
        self.bind_thread()?;
        let numel = shape
            .iter()
            .try_fold(1usize, |product, &dimension| product.checked_mul(dimension))
            .ok_or_else(|| HipError::new(0, "VMM tensor element count overflowed"))?;
        let byte_size = numel
            .checked_mul(dtype.size())
            .ok_or_else(|| HipError::new(0, "VMM tensor byte size overflowed"))?;
        let mut arena = VmmArena::reserve(&self.hip, self.device_id, byte_size)?;
        if initial_mapped_bytes > 0 {
            if let Err(err) = arena.map_next(&self.hip, initial_mapped_bytes, access_devices) {
                return Err(self.retain_failed_vmm_arena(arena, err));
            }
        }
        let buf = match arena.owner_buffer(byte_size) {
            Ok(buf) => buf,
            Err(err) => {
                return Err(self.retain_failed_vmm_arena(arena, err));
            }
        };
        let key = buf.as_ptr() as usize;
        if self.vmm_arenas.contains_key(&key) {
            let duplicate =
                HipError::new(0, &format!("duplicate VMM tensor base address 0x{key:x}"));
            return Err(self.retain_failed_vmm_arena(arena, duplicate));
        }
        self.vmm_arenas.insert(key, arena);
        Ok(GpuTensor {
            buf,
            shape: shape.to_vec(),
            dtype,
        })
    }

    fn retain_failed_vmm_arena(
        &mut self,
        mut arena: VmmArena,
        operation_error: HipError,
    ) -> HipError {
        let cleanup_error = arena.release(&self.hip).err();
        if !arena.is_released() {
            self.orphan_vmm_arenas.push(arena);
        }
        match cleanup_error {
            Some(cleanup) => HipError::new(
                0,
                &format!(
                    "{operation_error}; cleanup also failed: {cleanup}; arena retained for retry"
                ),
            ),
            None => operation_error,
        }
    }

    /// Grow the mapped prefix of a VMM-backed tensor by page-aligned bytes.
    pub fn grow_vmm_tensor(
        &mut self,
        tensor: &mut GpuTensor,
        additional_bytes: usize,
        access_devices: &[i32],
    ) -> HipResult<usize> {
        self.bind_thread()?;
        let key = tensor.buf.as_ptr() as usize;
        let logical_bytes = tensor.byte_size();
        let arena = self.vmm_arenas.get_mut(&key).ok_or_else(|| {
            HipError::new(
                0,
                &format!("tensor at 0x{key:x} is not a registered VMM owner"),
            )
        })?;
        arena.map_next(&self.hip, additional_bytes, access_devices)?;
        let mapped_bytes = arena.mapped_bytes();
        tensor.buf = unsafe { arena.owner_buffer(logical_bytes)? };
        Ok(mapped_bytes)
    }

    pub fn vmm_mapped_bytes(&self, tensor: &GpuTensor) -> Option<usize> {
        self.vmm_arenas
            .get(&(tensor.buf.as_ptr() as usize))
            .map(VmmArena::mapped_bytes)
    }

    /// Return the physical allocation granularity for a registered VMM tensor.
    pub fn vmm_granularity(&self, tensor: &GpuTensor) -> Option<usize> {
        self.vmm_arenas
            .get(&(tensor.buf.as_ptr() as usize))
            .map(VmmArena::granularity)
    }

    /// Return the driver's recommended physical mapping granularity without
    /// reserving an address range. Model-owned VMM planners use this for a
    /// dry-run admission check before mapping any cache pages.
    pub fn vmm_recommended_granularity(&self) -> HipResult<usize> {
        let prop = HipMemAllocationProp::device_pinned(self.device_id);
        self.hip
            .mem_get_allocation_granularity(&prop, HIP_MEM_ALLOCATION_GRANULARITY_RECOMMENDED)
    }

    pub fn vmm_allocation_count(&self) -> usize {
        self.vmm_arenas.len() + self.orphan_vmm_arenas.len()
    }

    /// Retry teardown for VMM arenas retained after an earlier cleanup error.
    /// Returns the number of arenas still pending cleanup.
    pub fn retry_vmm_cleanup(&mut self) -> HipResult<usize> {
        self.bind_thread()?;
        self.release_pending_vmm()
    }

    /// Retry pending VMM arenas and require a fully idle owner table.
    ///
    /// Success means `vmm_allocation_count() == 0`. Any remaining registration
    /// is an error so unload/load cannot claim a clean handoff.
    pub fn ensure_vmm_cleaned(&mut self) -> HipResult<()> {
        let live = self.vmm_arenas.len();
        if live != 0 {
            return Err(HipError::new(
                0,
                &format!(
                    "refusing VMM cleanup while {live} live VMM tensor owner(s) remain; unload the active model first"
                ),
            ));
        }
        match self.retry_vmm_cleanup() {
            Ok(0) => Ok(()),
            Ok(n) => Err(HipError::new(
                0,
                &format!("VMM teardown incomplete: {n} arena(s) still pending"),
            )),
            Err(err) => {
                let n = self.orphan_vmm_arenas.len();
                if n == 0 {
                    Err(err)
                } else {
                    Err(HipError::new(
                        err.code,
                        &format!("{err}; {n} arena(s) still pending"),
                    ))
                }
            }
        }
    }

    fn release_pending_vmm(&mut self) -> HipResult<usize> {
        let mut first_error = None;
        let mut orphan_index = 0;
        while orphan_index < self.orphan_vmm_arenas.len() {
            let (result, released) = {
                let arena = &mut self.orphan_vmm_arenas[orphan_index];
                let result = arena.release(&self.hip);
                (result, arena.is_released())
            };
            if released {
                let released_arena = self.orphan_vmm_arenas.swap_remove(orphan_index);
                debug_assert!(released_arena.is_released());
            } else {
                orphan_index += 1;
            }
            if let Err(err) = result {
                if first_error.is_none() {
                    first_error = Some(err);
                }
            }
        }
        match first_error {
            Some(err) => Err(err),
            None => Ok(self.orphan_vmm_arenas.len()),
        }
    }

    fn release_registered_vmm(&mut self) -> HipResult<usize> {
        let keys: Vec<usize> = self.vmm_arenas.keys().copied().collect();
        let mut first_error = None;
        for key in keys {
            let (result, released) = {
                let arena = self.vmm_arenas.get_mut(&key).unwrap();
                let result = arena.release(&self.hip);
                (result, arena.is_released())
            };
            if released {
                self.vmm_arenas.remove(&key);
            }
            if let Err(err) = result {
                if first_error.is_none() {
                    first_error = Some(err);
                }
            }
        }
        if let Err(err) = self.release_pending_vmm() {
            if first_error.is_none() {
                first_error = Some(err);
            }
        }
        match first_error {
            Some(err) => Err(err),
            None => Ok(self.vmm_allocation_count()),
        }
    }

    pub fn upload_f32(&mut self, data: &[f32], shape: &[usize]) -> HipResult<GpuTensor> {
        self.bind_thread()?;
        let tensor = self.alloc_tensor(shape, DType::F32)?;
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        self.hip.memcpy_htod(&tensor.buf, bytes)?;
        Ok(tensor)
    }

    /// Allocate an F32 tensor filled with a constant `value` (host-side fill +
    /// sync htod). Used for `-inf`-initialised buffers where a byte-memset
    /// can't express the bit pattern (e.g. the compressor `score_state`, which
    /// the reference inits to `float("-inf")` so unfilled pool slots get zero
    /// softmax weight).
    pub fn full_f32(&mut self, shape: &[usize], value: f32) -> HipResult<GpuTensor> {
        self.bind_thread()?;
        let tensor = self.alloc_tensor(shape, DType::F32)?;
        let data = vec![value; tensor.numel()];
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        self.hip.memcpy_htod(&tensor.buf, bytes)?;
        Ok(tensor)
    }

    /// In-place constant fill of an existing F32 tensor (sync htod).
    pub fn fill_f32(&mut self, tensor: &GpuTensor, value: f32) -> HipResult<()> {
        self.bind_thread()?;
        let data = vec![value; tensor.numel()];
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        self.hip.memcpy_htod(&tensor.buf, bytes)?;
        Ok(())
    }

    pub fn download_f32(&self, tensor: &GpuTensor) -> HipResult<Vec<f32>> {
        self.bind_thread()?;
        let numel = tensor.numel();
        let mut data = vec![0.0f32; numel];
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, numel * 4) };
        self.hip.memcpy_dtoh(bytes, &tensor.buf)?;
        Ok(data)
    }

    pub fn zeros(&mut self, shape: &[usize], dtype: DType) -> HipResult<GpuTensor> {
        self.bind_thread()?;
        let tensor = self.alloc_tensor(shape, dtype)?;
        match self.active_stream.as_ref() {
            Some(stream) => self
                .hip
                .memset_async(&tensor.buf, 0, tensor.byte_size(), stream)?,
            None => self.hip.memset(&tensor.buf, 0, tensor.byte_size())?,
        }
        Ok(tensor)
    }

    /// Upload raw bytes to GPU (for quantized weights).
    pub fn upload_raw(&self, data: &[u8], shape: &[usize]) -> HipResult<GpuTensor> {
        self.bind_thread()?;
        let buf = self.hip.malloc(data.len())?;
        self.hip.memcpy_htod(&buf, data)?;
        Ok(GpuTensor {
            buf,
            shape: shape.to_vec(),
            dtype: DType::Raw,
        })
    }

    /// Free a tensor. Contiguous buffers return to the pool. VMM owners run
    /// arena release once: success removes the registration; failure **retains**
    /// the arena for [`Self::retry_vmm_cleanup`] / [`Self::ensure_vmm_cleaned`]
    /// and returns the error (never a false clean free).
    pub fn free_tensor(&mut self, tensor: GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        let key = tensor.buf.as_ptr() as usize;
        if self.vmm_arenas.contains_key(&key) {
            if !tensor.buf.is_vmm_owner() {
                return Err(HipError::new(
                    0,
                    &format!("refusing to free borrowed VMM view at 0x{key:x}"),
                ));
            }
            let mut arena = self.vmm_arenas.remove(&key).unwrap();
            let result = arena.release(&self.hip);
            if arena.is_released() {
                result
            } else {
                // The tensor owner has been consumed, so keep its still-live
                // arena only in the pending table. Cleanup retries must never
                // touch entries in `vmm_arenas`, which are active owners.
                self.orphan_vmm_arenas.push(arena);
                match result {
                    Ok(()) => Err(HipError::new(
                        0,
                        &format!(
                            "VMM tensor at 0x{key:x} reported success but arena is still live"
                        ),
                    )),
                    Err(err) => Err(HipError::new(
                        err.code,
                        &format!("{err}; arena retained for retry"),
                    )),
                }
            }
        } else if tensor.buf.is_borrowed() || tensor.buf.is_vmm_owner() {
            Err(HipError::new(
                0,
                &format!("refusing to pool non-owning tensor at 0x{key:x}"),
            ))
        } else {
            self.pool.free(tensor.buf);
            Ok(())
        }
    }
    /// Calibration capture hook for an instrumented linear: if a collector is
    /// armed and `weight`'s buffer pointer is a known calibration target, invoke
    /// `capture(name, input)`. Zero-cost (`is_none()` + return) when no collector
    /// is armed, so non-calibration forwards are byte-identical. The collector
    /// `Arc` is cloned before the call so `self` is not aliased by `active_capture`.
    #[inline]
    pub fn maybe_capture_activation(
        &mut self,
        weight: &GpuTensor,
        input: &GpuTensor,
        n: usize,
        k: usize,
    ) {
        if self.active_capture.is_none() {
            return;
        }
        let ptr = weight.buf.as_ptr() as usize;
        let name = match self.capture_names.get(&ptr) {
            Some(nm) => nm.clone(),
            None => return,
        };
        if let Some(cap) = self.active_capture.clone() {
            cap.capture(self, &name, input, n, k);
        }
    }

    /// Calibration: `acc[c] += Σ_n x[n,c]²` (per-column sum-of-squares, the
    /// imatrix / diag(H) signal). `x` is [N, K] F32; `acc` is [K] F32, ADDED into
    /// (caller zeroes once, then accumulates across the calibration corpus).
    pub fn calib_sumsq_reduce_f32(
        &mut self,
        x: &GpuTensor,
        acc: &GpuTensor,
        n: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "calib_reduce",
            crate::kernels::CALIB_REDUCE_SRC,
            "calib_sumsq_reduce_f32",
        )?;
        let x_ptr = x.buf.as_ptr();
        let acc_ptr = acc.buf.as_ptr();
        let n_i = n as i32;
        let k_i = k as i32;
        let block = 256u32;
        let grid = ((k as u32) + block - 1) / block;
        let mut params: Vec<*mut c_void> = vec![
            &x_ptr as *const _ as *mut c_void,
            &acc_ptr as *const _ as *mut c_void,
            &n_i as *const _ as *mut c_void,
            &k_i as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "calib_sumsq_reduce_f32",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(x_ptr);
                blob.push_ptr(acc_ptr);
                blob.push_i32(n_i);
                blob.push_i32(k_i);
                blob
            },
        )
    }

    /// Calibration: `H[i,j] += Σ_n x[n,i]·x[n,j]` (the K×K GPTQ Hessian, tiled
    /// GEMM accumulate). `x` is [N, K] F32; `H` is [K, K] F32 row-major, ADDED
    /// into (caller zeroes once, then accumulates across the calibration corpus).
    pub fn calib_hessian_outer_f32(
        &mut self,
        x: &GpuTensor,
        h: &GpuTensor,
        n: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "calib_reduce",
            crate::kernels::CALIB_REDUCE_SRC,
            "calib_hessian_outer_f32",
        )?;
        let x_ptr = x.buf.as_ptr();
        let h_ptr = h.buf.as_ptr();
        let n_i = n as i32;
        let k_i = k as i32;
        let tile = 16u32;
        let grid_x = ((k as u32) + tile - 1) / tile;
        let grid_y = ((k as u32) + tile - 1) / tile;
        let mut params: Vec<*mut c_void> = vec![
            &x_ptr as *const _ as *mut c_void,
            &h_ptr as *const _ as *mut c_void,
            &n_i as *const _ as *mut c_void,
            &k_i as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "calib_hessian_outer_f32",
            [grid_x, grid_y, 1],
            [tile, tile, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(x_ptr);
                blob.push_ptr(h_ptr);
                blob.push_i32(n_i);
                blob.push_i32(k_i);
                blob
            },
        )
    }

    /// `w[n] += (1/k)·Σ_c d[n,c]²` — per-row (per-token) mean-square of an
    /// output-grad block `d[n,k]`. Used by GuidedQuant calibration to turn a
    /// linear's output adjoint `∂ℓ/∂z` into a per-token Fisher weight. `w` is
    /// caller-zeroed `[n]`.
    pub fn calib_row_meansq_f32(
        &mut self,
        d: &GpuTensor,
        w: &GpuTensor,
        n: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "calib_reduce",
            crate::kernels::CALIB_REDUCE_SRC,
            "calib_row_meansq_f32",
        )?;
        let d_ptr = d.buf.as_ptr();
        let w_ptr = w.buf.as_ptr();
        let n_i = n as i32;
        let k_i = k as i32;
        let block = 64u32;
        let grid = ((n as u32) + block - 1) / block;
        let mut params: Vec<*mut c_void> = vec![
            &d_ptr as *const _ as *mut c_void,
            &w_ptr as *const _ as *mut c_void,
            &n_i as *const _ as *mut c_void,
            &k_i as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "calib_row_meansq_f32",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(d_ptr);
                blob.push_ptr(w_ptr);
                blob.push_i32(n_i);
                blob.push_i32(k_i);
                blob
            },
        )
    }

    /// `acc[c] += Σ_n w[n]·x[n,c]²` — the per-row-weighted column sum-of-squares,
    /// i.e. the diagonal of the weighted Hessian. `w≡1` reduces to
    /// [`Self::calib_sumsq_reduce_f32`].
    pub fn calib_sumsq_weighted_f32(
        &mut self,
        x: &GpuTensor,
        w: &GpuTensor,
        acc: &GpuTensor,
        n: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "calib_reduce",
            crate::kernels::CALIB_REDUCE_SRC,
            "calib_sumsq_weighted_f32",
        )?;
        let x_ptr = x.buf.as_ptr();
        let w_ptr = w.buf.as_ptr();
        let a_ptr = acc.buf.as_ptr();
        let n_i = n as i32;
        let k_i = k as i32;
        let block = 64u32;
        let grid = ((k as u32) + block - 1) / block;
        let mut params: Vec<*mut c_void> = vec![
            &x_ptr as *const _ as *mut c_void,
            &w_ptr as *const _ as *mut c_void,
            &a_ptr as *const _ as *mut c_void,
            &n_i as *const _ as *mut c_void,
            &k_i as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "calib_sumsq_weighted_f32",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(x_ptr);
                blob.push_ptr(w_ptr);
                blob.push_ptr(a_ptr);
                blob.push_i32(n_i);
                blob.push_i32(k_i);
                blob
            },
        )
    }

    /// `H[i,j] += Σ_n w[n]·x[n,i]·x[n,j]` — the per-row-weighted (GuidedQuant /
    /// empirical-Fisher) Hessian. `w≡1` reduces to [`Self::calib_hessian_outer_f32`].
    pub fn calib_hessian_outer_weighted_f32(
        &mut self,
        x: &GpuTensor,
        w: &GpuTensor,
        h: &GpuTensor,
        n: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "calib_reduce",
            crate::kernels::CALIB_REDUCE_SRC,
            "calib_hessian_outer_weighted_f32",
        )?;
        let x_ptr = x.buf.as_ptr();
        let w_ptr = w.buf.as_ptr();
        let h_ptr = h.buf.as_ptr();
        let n_i = n as i32;
        let k_i = k as i32;
        let tile = 16u32;
        let grid_x = ((k as u32) + tile - 1) / tile;
        let grid_y = ((k as u32) + tile - 1) / tile;
        let mut params: Vec<*mut c_void> = vec![
            &x_ptr as *const _ as *mut c_void,
            &w_ptr as *const _ as *mut c_void,
            &h_ptr as *const _ as *mut c_void,
            &n_i as *const _ as *mut c_void,
            &k_i as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "calib_hessian_outer_weighted_f32",
            [grid_x, grid_y, 1],
            [tile, tile, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(x_ptr);
                blob.push_ptr(w_ptr);
                blob.push_ptr(h_ptr);
                blob.push_i32(n_i);
                blob.push_i32(k_i);
                blob
            },
        )
    }


    /// Free a newly allocated ordinary contiguous tensor immediately via HIP
    /// `free`, without returning the buffer to the reusable pool.
    ///
    /// Intended for allocation-failure cleanup of tensors that were just
    /// created (e.g. via [`Self::zeros`]) and must not race a pending async
    /// memset on the active stream. Rejects VMM owners and borrowed buffers
    /// with the same policy as [`Self::free_tensor`].
    pub fn release_tensor_immediate(&mut self, tensor: GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        let key = tensor.buf.as_ptr() as usize;
        if self.vmm_arenas.contains_key(&key) {
            if !tensor.buf.is_vmm_owner() {
                return Err(HipError::new(
                    0,
                    &format!("refusing to free borrowed VMM view at 0x{key:x}"),
                ));
            }
            return Err(HipError::new(
                0,
                &format!("refusing immediate free of VMM owner at 0x{key:x}"),
            ));
        }
        if tensor.buf.is_borrowed() || tensor.buf.is_vmm_owner() {
            return Err(HipError::new(
                0,
                &format!("refusing to free non-owning tensor at 0x{key:x}"),
            ));
        }
        // zeros() may have queued an async memset on the active stream; free
        // must not race that write.
        if let Some(stream) = self.active_stream.as_ref() {
            self.hip.stream_synchronize(stream)?;
        }
        self.hip.free(tensor.buf)
    }

    /// Drain the GPU memory pool. Actually calls hipFree on all pooled buffers.
    /// Call after model unload to return VRAM to the system.
    pub fn drain_pool(&mut self) {
        self.bind_thread_or_warn();
        self.pool.drain(&self.hip);
    }

    /// Invalidate every weight-pointer-keyed cache on the Gpu. Must be called
    /// any time a loaded model's weights are about to be freed; otherwise the
    /// next model load can allocate buffers at addresses that previously held
    /// different weights and the cache will incorrectly hit on stale entries.
    /// Affected caches:
    ///   * mmq_screen_cache: per-weight (safe, unsafe) screening verdicts (#87).
    ///   * fp16_shadow_cache: lazily-built FP16 dequant of HFQ4 weights for
    ///     the rocBLAS prefill path (CDNA3-only). Owns GpuTensors, so the
    ///     entries are released back to the pool here.
    pub fn invalidate_weight_caches(&mut self) {
        self.bind_thread_or_warn();
        self.mmq_screen.cache.clear();
        let shadows: Vec<GpuTensor> = self.fp16_shadow_cache.drain().map(|(_, t)| t).collect();
        for t in shadows {
            let _ = self.free_tensor(t);
        }
    }

    /// Invalidate the pointer-keyed F16 conversion cache. Must be called
    /// between layers in batched prefill when the same activation buffer
    /// (e.g. `pb_tmp`) is reused with different contents each layer —
    /// the cache sees the same GPU pointer and skips the F32→F16
    /// conversion, silently serving stale F16 data from the previous
    /// layer.
    pub fn invalidate_fp16_cache(&mut self) {
        self.scratch.fp16_x_source_ptr = std::ptr::null_mut();
    }

    /// Tear down all captured hipGraphs + their kernarg blobs. Captured
    /// graphs hold device pointers into the model's KV cache, scratch, and
    /// draft weights baked into kernarg memory by hipStreamEndCapture. Once
    /// any of those tensors are freed and the pool re-uses their buffers
    /// for the next model, replaying the captured graph would execute against
    /// either dangling or wrong-content pointers. The warmup sets would also
    /// wrongly skip the per-B / per-n_steps JIT step on the new model. Must
    /// be called from `unload_model` before the underlying tensors are
    /// returned to the pool.
    ///
    /// Affected state:
    ///   * graph_exec / captured_graph: single-slot AR forward graph.
    ///   * verify_graph_cache + verify_warmed_up + verify_capturing_b:
    ///     DFlash per-B verify-forward graphs.
    ///   * replay_graph_cache + replay_warmed_up + replay_capturing_n:
    ///     DFlash per-n_steps tape-replay graphs.
    pub fn invalidate_graph_state(&mut self) {
        self.bind_thread_or_warn();
        self.graphs.graph_destroy(&self.hip, self.device_id);
        self.graphs
            .verify_graph_destroy_all(&self.hip, self.device_id);
        self.graphs
            .replay_graph_destroy_all(&self.hip, self.device_id);
    }

    /// Typed F32 device-buffer copy. Unlike hipMemcpy D2D, this launch is
    /// visible to the retained-replay recorder and carries an explicit ABI.
    pub fn copy_f32_buffer(&mut self, dst: &GpuTensor, src: &GpuTensor, n: usize) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "copy_f32_buffer",
            crate::kernels::COPY_F32_BUFFER_SRC,
            "copy_f32_buffer",
        )?;
        let dp = dst.buf.as_ptr();
        let sp = src.buf.as_ptr();
        let mut n_i32 =
            i32::try_from(n).map_err(|_| HipError::new(0, "copy_f32_buffer length exceeds i32"))?;
        let mut params: Vec<*mut c_void> = vec![
            &dp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &mut n_i32 as *mut _ as *mut c_void,
        ];
        let blob_builder = || {
            let mut blob = hip_bridge::KernargBlob::new();
            blob.push_ptr(dp);
            blob.push_ptr(sp);
            blob.push_i32(n_i32);
            blob
        };
        let grid = (n as u32).div_ceil(256) * 256;
        self.launch_maybe_blob(
            "copy_f32_buffer",
            [grid, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            blob_builder,
        )
    }

    /// Copy `rows` contiguous source rows into one slot of a strided
    /// destination row. Used by DSpark hidden capture to replace B separate
    /// D2D memcpy nodes with one typed dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_f32_strided_slot_buffer(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        rows: usize,
        row_elems: usize,
        dst_row_stride: usize,
        dst_slot_offset: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "copy_f32_strided_slot_buffer",
            crate::kernels::COPY_F32_BUFFER_SRC,
            "copy_f32_strided_slot_buffer",
        )?;
        let dp = dst.buf.as_ptr();
        let sp = src.buf.as_ptr();
        let mut rows_i32 = i32::try_from(rows)
            .map_err(|_| HipError::new(0, "copy_f32_strided_slot_buffer rows exceed i32"))?;
        let mut row_elems_i32 = i32::try_from(row_elems)
            .map_err(|_| HipError::new(0, "copy_f32_strided_slot_buffer row length exceeds i32"))?;
        let mut stride_i32 = i32::try_from(dst_row_stride)
            .map_err(|_| HipError::new(0, "copy_f32_strided_slot_buffer stride exceeds i32"))?;
        let mut slot_i32 = i32::try_from(dst_slot_offset)
            .map_err(|_| HipError::new(0, "copy_f32_strided_slot_buffer slot exceeds i32"))?;
        let mut params: Vec<*mut c_void> = vec![
            &dp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &mut rows_i32 as *mut _ as *mut c_void,
            &mut row_elems_i32 as *mut _ as *mut c_void,
            &mut stride_i32 as *mut _ as *mut c_void,
            &mut slot_i32 as *mut _ as *mut c_void,
        ];
        let blob_builder = || {
            let mut blob = hip_bridge::KernargBlob::new();
            blob.push_ptr(dp);
            blob.push_ptr(sp);
            blob.push_i32(rows_i32);
            blob.push_i32(row_elems_i32);
            blob.push_i32(stride_i32);
            blob.push_i32(slot_i32);
            blob
        };
        let total = rows
            .checked_mul(row_elems)
            .ok_or_else(|| HipError::new(0, "copy_f32_strided_slot_buffer size overflow"))?;
        let grid = (total as u32).div_ceil(256) * 256;
        self.launch_maybe_blob(
            "copy_f32_strided_slot_buffer",
            [grid, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            blob_builder,
        )
    }

    /// Drop captured graph state and retained Redline replay after a live KV
    /// layout switch so the next forward cannot replay stale K/V modes, base
    /// pointers, or kernarg blobs baked under the prior tier.
    pub fn invalidate_for_kv_mode_switch(&mut self) {
        // bind_thread: skip — invalidate_graph_state binds; replay.poison is CPU.
        self.invalidate_graph_state();
        // Retained AQL/PM4 tapes capture KV base pointers and tier-dependent
        // kernargs. Poison so no old tape can run against the new logical layout.
        if self.replay.is_enabled() || self.replay.state() != crate::replay::ReplayState::Hip {
            self.replay
                .poison("KV mode switch invalidated retained Redline replay state");
        }
    }

    /// Invalidate captured execution after a model-owned cache capacity
    /// bucket grows. The allocation change is intentional and recoverable, so
    /// retained replay is re-armed rather than placed in sticky fallback.
    pub fn invalidate_for_layout_growth(&mut self) {
        self.invalidate_graph_state();
        self.replay.rearm_after_layout_growth();
    }

    // ── Kernel operations ───────────────────────────────────────

    /// y = A * x (matrix-vector multiply, A is [M, K], x is [K], y is [M])

    /// y = A_q4k * x (quantized matrix-vector multiply, A stored as Q4_K on GPU)
    /// a_raw: raw Q4_K bytes on GPU, x: F32 input, y: F32 output
    /// m: number of output rows, k: number of input columns (must be multiple of 256)

    /// HFQ4-G128 GEMV: flat 4-bit with 128-weight groups.
    /// K must be multiple of 128.

    /// ParoQuant Givens rotation: apply learned pairwise rotations + channel
    /// scaling to activation vector x in-place. Called before GEMV on
    /// ParoQ4G128 weights.
    ///
    /// x: [seq_len, hidden_dim] F16 (modified in place)
    /// pairs: [krot, hidden_dim] I16
    /// theta: [krot, hidden_dim/2] F16
    /// channel_scales: [hidden_dim] F16

    /// Out-of-place Givens rotation. Reads `x_in`, writes rotated
    /// activations to `x_out`. Replaces the
    /// `copy_d2d + givens_rotate` pair used by `rotate_x_paro_for` —
    /// one graph node + one inter-node dependency removed.
    #[allow(clippy::too_many_arguments)]

    /// Fused silu(gate)*up + per-channel scale + krot rounds of Givens
    /// rotation. Single-launch replacement for the
    /// `silu_mul_f32 + givens_rotate` pair used by the ParoQuant routed
    /// gate→down hop. Same shared-memory + grid contract as
    /// `givens_rotate`, plus two additional input pointers (gate, up)
    /// and a separate output pointer.
    #[allow(clippy::too_many_arguments)]

    /// Ensure the ParoQuant activation scratch buffer is allocated (F32, sized for dim).

    /// Device-to-device copy.
    ///
    /// Routes through `memcpy_dtod_auto` so it picks `memcpy_dtod_async` on
    /// the active (capturing) stream when one is set, falling back to the sync
    /// legacy-stream path otherwise. The raw `hip.memcpy_dtod` call would
    /// deadlock hipGraph capture with "operation would make the legacy stream
    /// depend on a capturing blocking stream" (matches the H2D fix in 7790ac6a).
    ///
    /// Callers must pass `n_bytes` explicitly to state intent — the prior
    /// implicit `min(src.size(), dst.size())` silently truncated mismatched
    /// copies, which was a footgun.
    pub fn copy_d2d(&self, src: &GpuTensor, dst: &GpuTensor, n_bytes: usize) -> HipResult<()> {
        // bind_thread: skip — delegates to memcpy_dtod_auto which binds
        debug_assert!(
            n_bytes <= src.buf.size(),
            "copy_d2d: n_bytes ({n_bytes}) exceeds src.buf.size ({})",
            src.buf.size()
        );
        debug_assert!(
            n_bytes <= dst.buf.size(),
            "copy_d2d: n_bytes ({n_bytes}) exceeds dst.buf.size ({})",
            dst.buf.size()
        );
        self.memcpy_dtod_auto(&dst.buf, &src.buf, n_bytes)
    }

    /// PARO4-G128 GEMV: ParoQuant pair-rotated activation + W4 weights.
    /// K must be multiple of 128 and M must be a multiple of the AWQ pack size
    /// (8). Each block computes one packed output column (8 output rows).

    /// Residual PARO4-G128 GEMV: y += A(x) where x is pair-rotated per
    /// ParoQuant metadata. One block computes one AWQ packed output column.

    /// PARO4-G128 fused SwiGLU down projection: y += W * (silu(gate) * up).
    /// Saves the standalone `silu_mul_f32` launch and ffn_hidden global write/read.

    /// PARO4-G128T direct GEMV for tiny-M projections. This keeps the Paro
    /// rotation inside the GEMV block instead of materializing x_rot globally.

    /// Residual PARO4-G128T direct GEMV for tiny-M projections.

    /// PARO4-G128 activation pre-rotation. This materializes the ParoQuant
    /// channel-scale + pair-rotation transform once per projection so the
    /// packed GEMV does not repeat it for every 8-output pack.

    /// PARO4-G128 fused SwiGLU activation + Paro pre-rotation. This is the
    /// useful fused shape for down projection: `x_rot = rotate(silu(gate)*up)`.

    /// PARO4-G128T activation pre-rotation. Same math as PARO4-G128, but
    /// theta is stored as precomputed f16 sin/cos pairs in the payload.

    /// PARO4-G128T fused SwiGLU activation + Paro pre-rotation.

    /// PARO4-G128 GEMV over an already materialized Paro-rotated activation.

    /// Residual PARO4-G128 GEMV over an already materialized Paro-rotated
    /// activation.

    /// PARO4-G128T GEMV over an already materialized Paro-rotated activation.
    /// The payload stores qweight as [M/8, K], making the inner-loop reads
    /// contiguous for the GEMV access pattern.

    /// Residual PARO4-G128T GEMV over an already materialized Paro-rotated
    /// activation.

    /// PARO4-G128T prerotated GEMV with four output lanes per block. This
    /// duplicates qweight reads relative to the 8-lane pack but lowers
    /// accumulator/register pressure for empirical Atlas testing.

    /// Residual PARO4-G128T pack4 prerotated GEMV.

    /// PARO4-G128T prerotated GEMV with two output lanes per block. This is
    /// an Atlas probe for whether lower accumulator pressure beats duplicate
    /// qweight traffic on the residual/down hot path.

    /// Residual PARO4-G128T pack2 prerotated GEMV.

    /// PARO4-G128T prerotated GEMV with one output lane per block.

    /// Residual PARO4-G128T pack1 prerotated GEMV.

    /// PARO4-G128 rotate-once wrapper used for env-gated runtime probes.

    /// PARO4-G128 rotate-once residual wrapper used for env-gated runtime probes.

    /// PARO4-G128 fused SwiGLU rotate-once down projection.

    /// PARO4-G128T rotate-once wrapper for engine-tiled qweight payloads.

    /// PARO4-G128T rotate-once residual wrapper for engine-tiled qweight payloads.

    /// PARO4-G128T fused SwiGLU rotate-once down projection.

    /// PARO4-G128T fused gate/up decode path. Gate and up have distinct
    /// Paro rotations, so this still rotates both, but batches the two
    /// rotations and the two pack4 GEMVs into two launches instead of four.

    /// PARO4-G128T fused LA projection path. The four Paro projections have
    /// distinct rotations, so this batches four rotates and four pack4 GEMVs
    /// into two launches.
    #[allow(clippy::too_many_arguments)]
    // ═══════════════════════════════════════════════════════════════════════════
    // Batch precompilation — compile all kernels a model needs in parallel
    // ═══════════════════════════════════════════════════════════════════════════

    /// Pre-compile all kernels needed for Qwen3.5 inference with a given
    /// weight quantization and KV cache type. Runs hipcc in parallel.
    #[cfg(feature = "deltanet")]
    pub fn precompile_qwen35(
        &mut self,
        weight_quant: &str,
        kv_type: &str,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // asym kernels #include "turbo_common.h" + "givens_common.h"; the
        // runtime dispatch path (see ensure_givens4_kernel) prepends the
        // header bodies and strips the #includes. We mirror that exactly so
        // the hash matches and the runtime re-uses our cached .hsaco.
        //
        // asym3 (SP1) additionally #includes "kv_slot_desc.h" — mirror
        // ensure_givens4_kernel's conditional handling (only sources that
        // actually contain the directive get it stripped/prepended) so the
        // other assemble_asym callers (asym2/asym4/fwht2/fwht3/fwht4), which
        // don't include it, keep producing byte-identical source to what the
        // runtime compiles for them, and only asym3's precompiled hash grows
        // to include the header.
        let assemble_asym = |body: &str| -> String {
            let needs_kv_slot_desc = body.contains("#include \"kv_slot_desc.h\"");
            let stripped = body
                .replace("#include \"turbo_common.h\"", "")
                .replace("#include \"givens_common.h\"", "")
                .replace("#include \"kv_slot_desc.h\"", "");
            if needs_kv_slot_desc {
                format!(
                    "{}\n{}\n{}\n{}",
                    kernels::TURBO_COMMON_H,
                    kernels::GIVENS_COMMON_SRC,
                    kernels::KV_SLOT_DESC_H,
                    stripped
                )
            } else {
                format!(
                    "{}\n{}\n{}",
                    kernels::TURBO_COMMON_H,
                    kernels::GIVENS_COMMON_SRC,
                    stripped
                )
            }
        };

        // Common kernels for all Qwen3.5 models (DeltaNet + FullAttn shared ops)
        let mut specs: Vec<(&str, String)> = vec![
            ("rmsnorm", kernels::RMSNORM_SRC.to_string()),
            ("add_inplace", kernels::ADD_INPLACE_SRC.to_string()),
            ("mul", kernels::MUL_SRC.to_string()),
            ("silu_mul", kernels::SILU_MUL_SRC.to_string()),
            ("sigmoid", kernels::SIGMOID_SRC.to_string()),
            ("alpha_gate", kernels::ALPHA_GATE_SRC.to_string()),
            ("conv1d_silu", kernels::CONV1D_SILU_SRC.to_string()),
            ("l2_norm", kernels::L2_NORM_SRC.to_string()),
            (
                "fused_qk_l2_norm_scale",
                kernels::FUSED_QK_L2_NORM_SCALE_SRC.to_string(),
            ),
            (
                "fused_sigmoid_alpha_gate",
                kernels::FUSED_SIGMOID_ALPHA_GATE_SRC.to_string(),
            ),
            (
                "conv1d_silu_split",
                kernels::CONV1D_SILU_SPLIT_SRC.to_string(),
            ),
            (
                "conv1d_silu_split_tree",
                kernels::CONV1D_SILU_SPLIT_TREE_SRC.to_string(),
            ),
            (
                "gated_delta_net_q8_tree",
                kernels::GATED_DELTA_NET_Q8_TREE_SRC.to_string(),
            ),
            ("sigmoid_mul", kernels::SIGMOID_MUL_SRC.to_string()),
            ("topk_logits", kernels::TOPK_LOGITS_SRC.to_string()),
            ("scale_f32", kernels::SCALE_F32_SRC.to_string()),
            ("gated_norm", kernels::GATED_NORM_SRC.to_string()),
            (
                "rope_partial_interleaved",
                kernels::ROPE_PARTIAL_INTERLEAVED_SRC.to_string(),
            ),
            // FullAttn: Q+gate deinterleave split
            ("deinterleave", kernels::DEINTERLEAVE_SRC.to_string()),
            // DeltaNet: Q/K repeat-interleave for asymmetric MQA (replaces 64+ memcpy_dtod calls per layer on 4B/9B)
            (
                "repeat_interleave_qk",
                kernels::REPEAT_INTERLEAVE_QK_SRC.to_string(),
            ),
        ];

        // Weight-format-specific GEMV
        match weight_quant {
            "hfq6" => {
                specs.push(("gemv_hfq6g256", kernels::GEMV_HFQ6G256_SRC.to_string()));
            }
            "paro4" => {
                specs.push(("gemv_paro4g128", kernels::GEMV_PARO4G128_SRC.to_string()));
            }
            "mq6" => {
                // MQ6 = FWHT-rotated HFQ6-G256. Needs both the MQ6 GEMV and the
                // raw HFQ6 GEMV (used by a few residual paths), plus the new
                // batched MQ6 GEMM for gemma4 Promote6 tensors (v_proj/down_proj).
                specs.push(("gemv_mq6g256", kernels::GEMV_MQ6G256_SRC.to_string()));
                specs.push(("gemv_hfq6g256", kernels::GEMV_HFQ6G256_SRC.to_string()));
                specs.push(("gemm_mq6g256", kernels::GEMM_MQ6G256_SRC.to_string()));
            }
            "hfq4" => {
                let (src, module) =
                    kernels::gemv_hfq4g256_for_arch(&self.arch_caps, self.flags.rdna2_variant);
                specs.push((module, src.to_string()));
                specs.push((
                    "gemv_hfq4g256_wide",
                    kernels::GEMV_HFQ4G256_WIDE_SRC.to_string(),
                ));
                // Multi-projection fused kernels (LA 4-way, FA 3-way, FFN
                // gate+up). Cross-arch — same 4-accumulator inner loop as
                // gemv_hfq4g256.hip; precompile on every arch that uses
                // the HFQ4 weight path.
                if self.arch_caps.is_rdna3_dgpu() && self.flags.rdna3_hfq4_qkvza_hoist_x32 {
                    specs.push((
                        "fused_qkvza_hfq4g256_hoist_x32_gfx1100",
                        kernels::FUSED_QKVZA_HFQ4G256_HOIST_X32_GFX1100_SRC.to_string(),
                    ));
                } else {
                    specs.push((
                        "fused_qkvza_hfq4g256",
                        kernels::FUSED_QKVZA_HFQ4G256_SRC.to_string(),
                    ));
                }
                if self.arch_caps.is_rdna3_dgpu() && self.flags.rdna3_hfq4_qkvza_2wave {
                    specs.push((
                        "fused_qkvza_hfq4g256_2wave",
                        kernels::FUSED_QKVZA_HFQ4G256_2WAVE_GFX1100_SRC.to_string(),
                    ));
                }
                if self.arch_caps.is_gfx1100() && self.flags.rdna3_hfq4_qkvza_k2048 {
                    specs.push((
                        "fused_qkvza_hfq4g256_k2048_gfx1100",
                        kernels::FUSED_QKVZA_HFQ4G256_K2048_GFX1100_SRC.to_string(),
                    ));
                }
                specs.push((
                    "fused_qkv_hfq4g256",
                    kernels::FUSED_QKV_HFQ4G256_SRC.to_string(),
                ));
                specs.push((
                    "fused_gate_up_hfq4g256",
                    kernels::FUSED_GATE_UP_HFQ4G256_SRC.to_string(),
                ));
                // gfx906/gfx908/gfx94x wave64-native variants — cut
                // wavefront pressure in half on the hottest kernels. Wave32
                // block=[32,1,1] kernels otherwise waste the upper 32 lanes
                // of every wave slot on these wave64-native arches.
                if self.arch_caps.is_wave64_native()
                    || (self.arch_caps.is_rdna3_dgpu() && self.flags.rdna3_hfq4_qkv_wave64)
                {
                    // Single-token QKV paths. RDNA3 may opt into Radiowave's
                    // explicit wave64 packing experiment; native-wave64
                    // targets retain their established selection.
                    specs.push((
                        "fused_qkvza_hfq4g256_wave64",
                        kernels::FUSED_QKVZA_HFQ4G256_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "fused_qkv_hfq4g256_wave64",
                        kernels::FUSED_QKV_HFQ4G256_WAVE64_SRC.to_string(),
                    ));
                }
                if self.arch_caps.is_wave64_native() {
                    // Remaining single-token and batched native-wave64 paths.
                    specs.push((
                        "fused_gate_up_hfq4g256_wave64",
                        kernels::FUSED_GATE_UP_HFQ4G256_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "gemv_hfq4g256_moe_gate_up_indexed_wave64",
                        kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "gemv_hfq4g256_moe_down_indexed_wave64",
                        kernels::GEMV_HFQ4G256_MOE_DOWN_INDEXED_WAVE64_SRC.to_string(),
                    ));
                    // Batched (DFlash verify path — hottest).
                    specs.push((
                        "gemm_qkvza_hfq4g256_wave64",
                        kernels::GEMM_QKVZA_HFQ4G256_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "gemm_qkv_hfq4g256_wave64",
                        kernels::GEMM_QKV_HFQ4G256_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "gemm_hfq4g256_wave64",
                        kernels::GEMM_HFQ4G256_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "gemm_hfq4g256_residual_wave64",
                        kernels::GEMM_HFQ4G256_RESIDUAL_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "gemv_hfq4g256_moe_gate_up_indexed_batched_wave64",
                        kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_BATCHED_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "gemv_hfq4g256_moe_down_indexed_batched_wave64",
                        kernels::GEMV_HFQ4G256_MOE_DOWN_INDEXED_BATCHED_WAVE64_SRC.to_string(),
                    ));
                }
                // gfx1100 multi-row GEMV is opt-in via HIPFIRE_GEMV_ROWS={2,4,8}.
                // Empirically slower than the single-row kernel on gfx1100 at all
                // tested matrix sizes (see commit log / multi-row kernel header),
                // so we only precompile when the env var explicitly requests it.
                if self.arch_caps.is_rdna3_dgpu() && self.flags.gemv_rows.unwrap_or(1) > 1 {
                    specs.push((
                        "gemv_hfq4g256_multirow_rdna3",
                        kernels::GEMV_HFQ4G256_MULTIROW_GFX1100_SRC.to_string(),
                    ));
                    specs.push((
                        "gemv_hfq4g256_residual_multirow_rdna3",
                        kernels::GEMV_HFQ4G256_RESIDUAL_MULTIROW_GFX1100_SRC.to_string(),
                    ));
                }
            }
            "mq4" => {
                // MQ4 = FWHT-rotated HFQ4-G256 — default format for current registry.
                // Shares the HFQ4 fused kernels (same blob, different dispatch key)
                // plus MQ-specific rotation kernels.
                let (src, module) =
                    kernels::gemv_hfq4g256_for_arch(&self.arch_caps, self.flags.rdna2_variant);
                specs.push((module, src.to_string()));
                specs.push(("gemv_mq4g256", kernels::GEMV_MQ4G256_SRC.to_string()));
                if self.arch_caps.is_rdna3_dgpu() && self.flags.rdna3_hfq4_qkvza_hoist_x32 {
                    specs.push((
                        "fused_qkvza_hfq4g256_hoist_x32_gfx1100",
                        kernels::FUSED_QKVZA_HFQ4G256_HOIST_X32_GFX1100_SRC.to_string(),
                    ));
                } else {
                    specs.push((
                        "fused_qkvza_hfq4g256",
                        kernels::FUSED_QKVZA_HFQ4G256_SRC.to_string(),
                    ));
                }
                if self.arch_caps.is_rdna3_dgpu() && self.flags.rdna3_hfq4_qkvza_2wave {
                    specs.push((
                        "fused_qkvza_hfq4g256_2wave",
                        kernels::FUSED_QKVZA_HFQ4G256_2WAVE_GFX1100_SRC.to_string(),
                    ));
                }
                if self.arch_caps.is_gfx1100() && self.flags.rdna3_hfq4_qkvza_k2048 {
                    specs.push((
                        "fused_qkvza_hfq4g256_k2048_gfx1100",
                        kernels::FUSED_QKVZA_HFQ4G256_K2048_GFX1100_SRC.to_string(),
                    ));
                }
                specs.push((
                    "fused_qkv_hfq4g256",
                    kernels::FUSED_QKV_HFQ4G256_SRC.to_string(),
                ));
                specs.push((
                    "fused_gate_up_hfq4g256",
                    kernels::FUSED_GATE_UP_HFQ4G256_SRC.to_string(),
                ));
                specs.push((
                    "fused_rmsnorm_mq_rotate",
                    kernels::FUSED_RMSNORM_MQ_ROTATE_SRC.to_string(),
                ));
                if self.arch_caps.is_gfx1100() && self.flags.rdna3_rmsnorm_wavegrid {
                    specs.push((
                        "fused_rmsnorm_mq_rotate_wavegrid",
                        kernels::FUSED_RMSNORM_MQ_ROTATE_WAVEGRID_GFX1100_SRC.to_string(),
                    ));
                }
                if self.arch_caps.is_gfx1100() && self.flags.rdna3_rmsnorm_split {
                    specs.push((
                        "rmsnorm_reduce_gfx1100",
                        kernels::RMSNORM_REDUCE_GFX1100_SRC.to_string(),
                    ));
                    specs.push((
                        "rotate_with_rms_gfx1100",
                        kernels::ROTATE_WITH_RMS_GFX1100_SRC.to_string(),
                    ));
                }
                if self.arch_caps.is_gfx1100() && self.flags.rdna3_rmsnorm_vecsum {
                    specs.push((
                        "fused_rmsnorm_mq_rotate_vecsum",
                        kernels::FUSED_RMSNORM_MQ_ROTATE_VECSUM_GFX1100_SRC.to_string(),
                    ));
                }
                specs.push((
                    "fused_silu_mul_mq_rotate",
                    kernels::FUSED_SILU_MUL_MQ_ROTATE_SRC.to_string(),
                ));
                // gfx906/gfx908/gfx94x wave64 variants — see hfq4 branch for rationale.
                if self.arch_caps.is_wave64_native()
                    || (self.arch_caps.is_rdna3_dgpu() && self.flags.rdna3_hfq4_qkv_wave64)
                {
                    // Single-token QKV paths. RDNA3 may opt into Radiowave's
                    // explicit wave64 packing experiment; native-wave64
                    // targets retain their established selection.
                    specs.push((
                        "fused_qkvza_hfq4g256_wave64",
                        kernels::FUSED_QKVZA_HFQ4G256_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "fused_qkv_hfq4g256_wave64",
                        kernels::FUSED_QKV_HFQ4G256_WAVE64_SRC.to_string(),
                    ));
                }
                if self.arch_caps.is_wave64_native() {
                    // Remaining single-token and batched native-wave64 paths.
                    specs.push((
                        "fused_gate_up_hfq4g256_wave64",
                        kernels::FUSED_GATE_UP_HFQ4G256_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "gemv_hfq4g256_moe_gate_up_indexed_wave64",
                        kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "gemv_hfq4g256_moe_down_indexed_wave64",
                        kernels::GEMV_HFQ4G256_MOE_DOWN_INDEXED_WAVE64_SRC.to_string(),
                    ));
                    // Batched (DFlash verify path — hottest).
                    specs.push((
                        "gemm_qkvza_hfq4g256_wave64",
                        kernels::GEMM_QKVZA_HFQ4G256_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "gemm_qkv_hfq4g256_wave64",
                        kernels::GEMM_QKV_HFQ4G256_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "gemm_hfq4g256_wave64",
                        kernels::GEMM_HFQ4G256_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "gemm_hfq4g256_residual_wave64",
                        kernels::GEMM_HFQ4G256_RESIDUAL_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "gemv_hfq4g256_moe_gate_up_indexed_batched_wave64",
                        kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_BATCHED_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "gemv_hfq4g256_moe_down_indexed_batched_wave64",
                        kernels::GEMV_HFQ4G256_MOE_DOWN_INDEXED_BATCHED_WAVE64_SRC.to_string(),
                    ));
                }
            }
            "q8" => {
                specs.push(("gemv_q8_0", kernels::GEMV_Q8_0_SRC.to_string()));
            }
            _ => {}
        }

        // Embedding kernels — Q8_0 is most common, also cover HFQ4G256/G128 variants
        specs.push(("embedding_q8", kernels::EMBEDDING_Q8_SRC.to_string()));
        specs.push((
            "embedding_hfq4g256",
            kernels::EMBEDDING_HFQ4G256_SRC.to_string(),
        ));
        specs.push((
            "embedding_hfq4g128",
            kernels::EMBEDDING_HFQ4G128_SRC.to_string(),
        ));
        specs.push((
            "embedding_hfq4g128_batched",
            kernels::EMBEDDING_HFQ4G128_BATCHED_SRC.to_string(),
        ));
        specs.push((
            "embedding_hfq4g256_batched",
            kernels::EMBEDDING_HFQ4G256_BATCHED_SRC.to_string(),
        ));
        specs.push((
            "embedding_q8_batched",
            kernels::EMBEDDING_Q8_BATCHED_SRC.to_string(),
        ));

        // DeltaNet kernels
        specs.push((
            "gated_delta_net_q8",
            kernels::GATED_DELTA_NET_Q8_SRC.to_string(),
        ));

        // KV cache kernels. asym3 is the current default — always ships flash.
        // q8 is the compat path with its own flash tile+reduce for long context.
        match kv_type {
            "asym4" => {
                specs.push((
                    "kv_cache_write_asym_k_givens4",
                    assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_GIVENS4_SRC),
                ));
                specs.push((
                    "kv_cache_write_asym_k_givens4_batched",
                    assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_GIVENS4_BATCHED_SRC),
                ));
                specs.push((
                    "attention_flash_asym4_tile",
                    assemble_asym(kernels::ATTENTION_FLASH_ASYM4_TILE_SRC),
                ));
                specs.push((
                    "attention_flash_asym4_tile_batched",
                    assemble_asym(kernels::ATTENTION_FLASH_ASYM4_TILE_BATCHED_SRC),
                ));
                specs.push((
                    "attention_flash_asym_reduce_batched",
                    kernels::ATTENTION_FLASH_ASYM_REDUCE_BATCHED_SRC.to_string(),
                ));
            }
            "fwht4" => {
                // Same byte layout as asym4 — just different K-rotation primitive.
                specs.push((
                    "kv_cache_write_asym_k_fwht4",
                    assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_FWHT4_SRC),
                ));
                specs.push((
                    "kv_cache_write_asym_k_fwht4_batched",
                    assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_FWHT4_BATCHED_SRC),
                ));
                specs.push((
                    "attention_flash_fwht4_tile",
                    assemble_asym(kernels::ATTENTION_FLASH_FWHT4_TILE_SRC),
                ));
                specs.push((
                    "attention_flash_fwht4_tile_batched",
                    assemble_asym(kernels::ATTENTION_FLASH_FWHT4_TILE_BATCHED_SRC),
                ));
                specs.push((
                    "attention_flash_asym_reduce_batched",
                    kernels::ATTENTION_FLASH_ASYM_REDUCE_BATCHED_SRC.to_string(),
                ));
            }
            "fwht3" => {
                // Same byte layout as asym3 (single-pass 256-element), FWHT rotation.
                specs.push((
                    "kv_cache_write_asym_k_fwht3",
                    assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_FWHT3_SRC),
                ));
                specs.push((
                    "kv_cache_write_asym_k_fwht3_batched",
                    assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_FWHT3_BATCHED_SRC),
                ));
                specs.push((
                    "attention_flash_fwht3_tile",
                    assemble_asym(kernels::ATTENTION_FLASH_FWHT3_TILE_SRC),
                ));
                specs.push((
                    "attention_flash_fwht3_tile_batched",
                    assemble_asym(kernels::ATTENTION_FLASH_FWHT3_TILE_BATCHED_SRC),
                ));
                specs.push((
                    "attention_flash_asym_reduce_batched",
                    kernels::ATTENTION_FLASH_ASYM_REDUCE_BATCHED_SRC.to_string(),
                ));
            }
            "fwht2" => {
                // Same byte layout as asym2, FWHT rotation. 2-pass over 128.
                specs.push((
                    "kv_cache_write_asym_k_fwht2",
                    assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_FWHT2_SRC),
                ));
                specs.push((
                    "kv_cache_write_asym_k_fwht2_batched",
                    assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_FWHT2_BATCHED_SRC),
                ));
                specs.push((
                    "attention_flash_fwht2_tile",
                    assemble_asym(kernels::ATTENTION_FLASH_FWHT2_TILE_SRC),
                ));
                specs.push((
                    "attention_flash_fwht2_tile_batched",
                    assemble_asym(kernels::ATTENTION_FLASH_FWHT2_TILE_BATCHED_SRC),
                ));
                specs.push((
                    "attention_flash_asym_reduce_batched",
                    kernels::ATTENTION_FLASH_ASYM_REDUCE_BATCHED_SRC.to_string(),
                ));
            }
            "asym3" => {
                specs.push((
                    "kv_cache_write_asym_k_givens3",
                    assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_GIVENS3_SRC),
                ));
                specs.push((
                    "kv_cache_write_asym_k_givens3_batched",
                    assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_GIVENS3_BATCHED_SRC),
                ));
                specs.push((
                    "attention_flash_asym3_tile",
                    assemble_asym(kernels::ATTENTION_FLASH_ASYM3_TILE_SRC),
                ));
                specs.push((
                    "attention_flash_asym3_tile_batched",
                    assemble_asym(kernels::ATTENTION_FLASH_ASYM3_TILE_BATCHED_SRC),
                ));
                specs.push((
                    "attention_flash_asym_reduce_batched",
                    kernels::ATTENTION_FLASH_ASYM_REDUCE_BATCHED_SRC.to_string(),
                ));
            }
            "asym2" => {
                specs.push((
                    "kv_cache_write_asym_k_givens2",
                    assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_GIVENS2_SRC),
                ));
                specs.push((
                    "kv_cache_write_asym_k_givens2_batched",
                    assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_GIVENS2_BATCHED_SRC),
                ));
                specs.push((
                    "attention_flash_asym2_tile",
                    assemble_asym(kernels::ATTENTION_FLASH_ASYM2_TILE_SRC),
                ));
                specs.push((
                    "attention_flash_asym2_tile_batched",
                    assemble_asym(kernels::ATTENTION_FLASH_ASYM2_TILE_BATCHED_SRC),
                ));
                specs.push((
                    "attention_flash_asym_reduce_batched",
                    kernels::ATTENTION_FLASH_ASYM_REDUCE_BATCHED_SRC.to_string(),
                ));
            }
            "q8" | _ => {
                specs.push((
                    "kv_cache_write_q8_0",
                    kernels::KV_CACHE_WRITE_Q8_0_SRC.to_string(),
                ));
                specs.push((
                    "attention_q8_0_kv",
                    kernels::ATTENTION_Q8_0_KV_SRC.to_string(),
                ));
                specs.push(("attention_q8_0_kv_batched", {
                    // Same header-stripping treatment as
                    // attention_q8_0_kv_batched_masked_slots: the kernel
                    // #includes kv_slot_desc.h, but this precompile path
                    // (like the runtime hipcc compile) has no -I to
                    // kernels/src, so the directive must be stripped and
                    // the header body prepended instead.
                    let stripped = kernels::ATTENTION_Q8_0_KV_BATCHED_SRC
                        .replace("#include \"kv_slot_desc.h\"", "");
                    format!("{}\n{}", kernels::KV_SLOT_DESC_H, stripped)
                }));
                specs.push(("attention_q8_0_kv_independent_masked_windowed", {
                    // Shares ATTENTION_Q8_0_KV_BATCHED_SRC with the entry
                    // above, so it needs the identical strip-and-prepend:
                    // that source #includes kv_slot_desc.h and this
                    // precompile path has no -I to kernels/src.
                    let stripped = kernels::ATTENTION_Q8_0_KV_BATCHED_SRC
                        .replace("#include \"kv_slot_desc.h\"", "");
                    format!("{}\n{}", kernels::KV_SLOT_DESC_H, stripped)
                }));
                specs.push(("attention_q8_0_flash_prefill", {
                    // Same header-stripping treatment as
                    // attention_q8_0_kv_batched above: the kernel now
                    // #includes kv_slot_desc.h (Task 6), but this
                    // precompile path (like the runtime hipcc compile in
                    // attention_q8_0_flash_prefill_slots) has no -I to
                    // kernels/src, so the directive must be stripped and
                    // the header body prepended instead.
                    let stripped = kernels::ATTENTION_Q8_0_FLASH_PREFILL_SRC
                        .replace("#include \"kv_slot_desc.h\"", "");
                    format!("{}\n{}", kernels::KV_SLOT_DESC_H, stripped)
                }));
                specs.push(("kv_cache_write_q8_0_batched", {
                    // Same header-stripping treatment as
                    // attention_q8_0_kv_batched above: the kernel now
                    // #includes kv_slot_desc.h (Task 3), but this
                    // precompile path (like the runtime hipcc compile in
                    // kv_cache_write_q8_0_batched_slots) has no -I to
                    // kernels/src, so the directive must be stripped and
                    // the header body prepended instead.
                    let stripped = kernels::KV_CACHE_WRITE_Q8_0_BATCHED_SRC
                        .replace("#include \"kv_slot_desc.h\"", "");
                    format!("{}\n{}", kernels::KV_SLOT_DESC_H, stripped)
                }));
                specs.push((
                    "attention_flash_q8_0_tile",
                    kernels::ATTENTION_FLASH_Q8_0_TILE_SRC.to_string(),
                ));
                specs.push((
                    "attention_flash_q8_0_reduce",
                    kernels::ATTENTION_FLASH_Q8_0_REDUCE_SRC.to_string(),
                ));
            }
        }

        // Exact parallel-sampler modules used by sample_top_p_pf. Sources are
        // the same string rewrites as runtime (see sampling.rs helpers) so the
        // compile-cache hash hits on first token instead of hipcc JIT.
        for (name, src) in crate::sampling::sample_top_p_parallel_precompile_specs() {
            specs.push((name, src));
        }

        // Convert to (&str, &str) for the batch API
        let batch: Vec<(&str, &str)> = specs
            .iter()
            .map(|(name, src)| (*name, src.as_str()))
            .collect();
        self.compiler.compile_batch(&batch)?;

        // Now load all modules + functions sequentially (GPU API)
        for (name, src) in &specs {
            // Map module name → function name(s). Most modules expose exactly one
            // function; multirow modules expose three (r2/r4/r8).
            let func_names: Vec<&str> = match *name {
                "rmsnorm" => vec!["rmsnorm_f32"],
                "add_inplace" => vec!["add_inplace_f32"],
                "mul" => vec!["mul_f32"],
                "silu_mul" => vec!["silu_mul_f32"],
                "sigmoid" => vec!["sigmoid_f32"],
                "alpha_gate" => vec!["alpha_gate_f32"],
                "conv1d_silu" => vec!["conv1d_silu_f32"],
                "l2_norm" => vec!["l2_norm_f32"],
                "fused_qk_l2_norm_scale" => vec!["fused_qk_l2_norm_scale_f32"],
                "fused_sigmoid_alpha_gate" => vec!["fused_sigmoid_alpha_gate_f32"],
                "conv1d_silu_split" => vec!["conv1d_silu_split_f32"],
                "conv1d_silu_split_tree" => vec!["conv1d_silu_split_tree_f32"],
                "gated_delta_net_q8_tree" => vec!["gated_delta_net_q8_tree"],
                "sigmoid_mul" => vec!["sigmoid_mul_f32"],
                "topk_logits" => vec!["topk_logits_f32"],
                "scale_f32" => vec!["scale_f32"],
                "gated_norm" => vec!["gated_norm_f32"],
                "rope_partial_interleaved" => vec!["rope_partial_interleaved_f32"],
                "deinterleave" => vec!["deinterleave_f32"],
                "repeat_interleave_qk" => vec!["repeat_interleave_qk_f32"],
                "gated_delta_net_q8" => vec!["gated_delta_net_q8"],
                // MQ4 GEMV module exports both the main GEMV and the standalone
                // x rotation kernel used by the prerotated dispatch path.
                "gemv_mq4g256" => vec!["gemv_mq4g256", "mq_rotate_x"],
                // Arch-variant HFQ4 GEMV modules all expose the same symbol.
                n if n.starts_with("gemv_hfq4g256_rdna") => vec!["gemv_hfq4g256"],
                n if n.starts_with("gemv_hfq4g256_gfx") => vec!["gemv_hfq4g256"],
                "fused_qkvza_hfq4g256_k2048_gfx1100" => {
                    vec!["fused_qkvza_hfq4g256_k2048"]
                }
                // Multi-row RDNA3 modules expose three entry points per .hsaco
                "gemv_hfq4g256_multirow_rdna3" => vec![
                    "gemv_hfq4g256_multirow_r2",
                    "gemv_hfq4g256_multirow_r4",
                    "gemv_hfq4g256_multirow_r8",
                ],
                "gemv_hfq4g256_residual_multirow_rdna3" => vec![
                    "gemv_hfq4g256_residual_multirow_r2",
                    "gemv_hfq4g256_residual_multirow_r4",
                    "gemv_hfq4g256_residual_multirow_r8",
                ],
                "gemv_hfq4g256_moe_gate_up_indexed_wave64" => {
                    vec!["gemv_hfq4g256_moe_gate_up_k8_indexed_wave64"]
                }
                "gemv_hfq4g256_moe_down_indexed_wave64" => {
                    vec!["gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_wave64"]
                }
                "gemv_hfq4g256_moe_gate_up_indexed_batched_wave64" => {
                    vec!["gemv_hfq4g256_moe_gate_up_k8_indexed_batched_wave64"]
                }
                "gemv_hfq4g256_moe_down_indexed_batched_wave64" => {
                    vec!["gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched_wave64"]
                }
                "sample_top_p_parallel" => vec![
                    "sample_apply_repeat_penalty",
                    "sample_topk_partial",
                    "sample_topk_finalize",
                ],
                "sample_top_p_parallel_w64" => vec![
                    "sample_apply_repeat_penalty_w64",
                    "sample_topk_partial_w64",
                    "sample_topk_finalize_w64",
                ],
                "sample_top_p_parallel_fast21" => vec![
                    "sample_apply_repeat_penalty_fast21",
                    "sample_topk_partial_fast21",
                    "sample_topk_finalize_fast21",
                ],
                "sample_top_p_parallel_fast65" => vec![
                    "sample_apply_repeat_penalty_fast65",
                    "sample_topk_partial_fast65",
                    "sample_topk_finalize_fast65",
                ],
                other => vec![other],
            };
            // Compile and ensure the module is loaded once.
            let obj_path = self.compiler.compile(name, src)?;
            let obj_path_str = obj_path.to_str().unwrap().to_string();
            if !self.modules.contains_key(*name) {
                let module = crate::scratch::module_load_or_recompile(
                    &self.hip,
                    &mut self.compiler,
                    name,
                    src,
                    &obj_path_str,
                )?;
                self.modules.insert(name.to_string(), module);
            }
            let module = &self.modules[*name];
            for func_name in &func_names {
                if self.functions.contains_key(*func_name) {
                    continue;
                }
                let func = self.hip.module_get_function(module, func_name)?;
                self.functions.insert(func_name.to_string(), func);
            }
        }

        Ok(())
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Kernel profiler
    // ═══════════════════════════════════════════════════════════════════════════

    /// Profile all compiled kernels: hardware caps + ISA metadata + occupancy.
    pub fn profile(
        &self,
    ) -> (
        crate::profiler::GpuCapability,
        Vec<crate::profiler::KernelProfile>,
    ) {
        self.bind_thread_or_warn();
        let vram = self.hip.get_vram_info().map(|(_, t)| t as u64).unwrap_or(0);
        let cu_hint = self
            .hip
            .get_device_attribute(
                crate::profiler::HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                0,
            )
            .ok()
            .filter(|&v| v > 0)
            .map(|v| crate::profiler::hip_mp_count_to_cu_count(&self.arch, v as u32))
            .filter(|&v| (4..=256).contains(&v));
        crate::profiler::profile_kernels_with_hint(
            &self.arch,
            vram,
            self.compiler.compiled_kernels(),
            cu_hint,
        )
    }

    /// Bulk F32 → BF16 conversion with round-to-nearest-even.
    ///
    /// Elementwise `dst[i] = (bf16)src[i]` for `i < nelems`, matching the
    /// host reference `round_to_bf16` (`crates/hipfire-arch-deepseek4/src/parent/codec.rs:92-110`).
    /// Used to stage BF16 activations for `gemm_bf16_mfma_gfx942` under
    /// `HIPFIRE_CALIB_BF16=1` so calibration GEMMs land on the CDNA3 MFMA
    /// kernel instead of the scalar F32 kernel.
    pub fn convert_f32_to_bf16(
        &mut self,
        src: &GpuTensor,
        dst: &GpuTensor,
        nelems: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if nelems == 0 {
            return Ok(());
        }
        let src_need = nelems.checked_mul(4).ok_or_else(|| {
            HipError::new(0, "convert_f32_to_bf16: nelems*4 overflow")
        })?;
        let dst_need = nelems.checked_mul(2).ok_or_else(|| {
            HipError::new(0, "convert_f32_to_bf16: nelems*2 overflow")
        })?;
        if src.buf.size() < src_need {
            return Err(HipError::new(
                0,
                &format!(
                    "convert_f32_to_bf16: src buffer too small (have {} need {src_need} for nelems={nelems})",
                    src.buf.size()
                ),
            ));
        }
        if dst.buf.size() < dst_need {
            return Err(HipError::new(
                0,
                &format!(
                    "convert_f32_to_bf16: dst buffer too small (have {} need {dst_need} for nelems={nelems})",
                    dst.buf.size()
                ),
            ));
        }
        if nelems > i32::MAX as usize {
            return Err(HipError::new(
                0,
                &format!(
                    "convert_f32_to_bf16: nelems {nelems} exceeds i32::MAX"
                ),
            ));
        }
        const KERNEL: &str = "convert_f32_to_bf16";
        const SRC: &str = include_str!("../../../kernels/src/convert_f32_to_bf16.hip");
        self.ensure_kernel(KERNEL, SRC, KERNEL)?;
        let src_ptr = src.buf.as_ptr();
        let dst_ptr = dst.buf.as_ptr();
        let nelems_i32 = nelems as i32;
        let mut params: Vec<*mut c_void> = vec![
            &src_ptr as *const _ as *mut c_void,
            &dst_ptr as *const _ as *mut c_void,
            &nelems_i32 as *const _ as *mut c_void,
        ];
        let grid = nelems.div_ceil(256) as u32;
        let bytes = src_need + dst_need;
        let timer = crate::profile::begin_timer(&self.hip, "convert", KERNEL, bytes);
        let result = self.launch_maybe_blob(
            KERNEL,
            [grid, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(src_ptr);
                blob.push_ptr(dst_ptr);
                blob.push_i32(nelems_i32);
                blob
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}

impl Drop for Gpu {
    /// Defensive: bind owning device before any future per-field `Drop`
    /// impls call `hipFree` etc. Uses `bind_thread_or_warn` to avoid
    /// panic-in-Drop from `bind_thread`'s `debug_assert!`.
    fn drop(&mut self) {
        if std::thread::panicking() && self.vmm_allocation_count() == 0 {
            return;
        }
        self.bind_thread_or_warn();
        if let Err(err) = self.release_registered_vmm() {
            eprintln!("[rdna-compute] failed to release VMM arena during Gpu drop: {err}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::gen_fwht_signs;
    use super::DType;
    use super::HessianCapture;

    #[test]
    fn q8hfq_row_stride_matches_legacy_formula() {
        // Hard-coded expected stride (128B-aligned) — independent oracle, NOT the
        // formula re-run. raw_row = (k/32)*2 + k, then padded up to a 128B boundary.
        assert_eq!(DType::Q8HFQ.row_stride(4096), 4352); // raw 4352, already aligned
        assert_eq!(DType::Q8HFQ.row_stride(11008), 11776); // raw 11696 → padded
        assert_eq!(DType::Q8HFQ.row_stride(5120), 5504); // raw 5440 → padded
        assert_eq!(DType::Q8HFQ.row_stride(14336), 15232); // raw 15232, already aligned
        assert_eq!(DType::Q8HFQ.row_stride(256), 384); // raw 272 → padded
        assert_eq!(DType::Q8HFQ.row_stride(96), 128); // raw 102 → padded
    }

    #[test]
    fn non_q8hfq_row_stride_is_zero() {
        for dt in [
            DType::HFQ4G256,
            DType::MQ4G256,
            DType::Q8_0,
            DType::F16,
            DType::HFP4G32,
            DType::MFP4G32,
        ] {
            assert_eq!(dt.row_stride(4096), 0, "{dt:?} must have stride 0");
        }
    }

    #[test]
    fn only_hfp4_family_requires_k_mod_256() {
        assert!(DType::HFP4G32.requires_k_mod_256());
        assert!(DType::MFP4G32.requires_k_mod_256());
        for dt in [
            DType::HFQ4G256,
            DType::MQ4G256,
            DType::Q8HFQ,
            DType::Q8_0,
            DType::F16,
            DType::F32,
        ] {
            assert!(!dt.requires_k_mod_256(), "{dt:?} must NOT require k%256");
        }
    }

    /// Deterministic pseudo-random rows (no RNG crate): mix in exact zeros,
    /// signed-zero, negatives, and large/small magnitudes to exercise the
    /// zero-skip branch and fp rounding.
    fn fixed_row(k: usize, seed: u64) -> Vec<f32> {
        let mut state: u64 = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as u32) as f64 / (u32::MAX as f64)
        };
        (0..k)
            .map(|c| {
                let r = next();
                if r < 0.20 {
                    0.0f32
                } else if r < 0.25 {
                    -0.0f32
                } else {
                    let mag = (r - 0.5) * 4.0;
                    let scale = if c % 7 == 0 { 1.0e3 } else { 1.0 };
                    (mag * scale) as f32
                }
            })
            .collect()
    }

    /// LOAD-BEARING CORRECTNESS GATE for the rayon-parallel `accumulate_token`.
    /// The parallel per-token accumulate (across disjoint (tensor,expert)
    /// accumulators) MUST be BIT-IDENTICAL to the serial per-item `accumulate`
    /// reference — a wrong parallel sum would silently produce wrong Hessians
    /// and reconfound the GPTQ-on-E8 experiment. We compare every f64 entry by
    /// raw bits (`to_bits()`), not an epsilon: each accumulator is updated by
    /// exactly one thread with the SAME ops as serial, so exact equality is the
    /// correct, strongest assertion. Multiple tokens are accumulated to verify
    /// the parallel path is also correct across repeated touches of an entry.
    #[test]
    fn accumulate_token_bit_identical_to_serial() {
        // A3B-shaped: gate_up K=2048 (8 blocks), down K=512 (2 blocks); 8 experts
        // (distinct ids) -> 16 distinct (tensor,expert) keys per token.
        let n_experts = 8usize;
        let n_tokens = 5usize;
        let k_gate = 2048usize;
        let k_down = 512usize;

        let mut par = HessianCapture::default();
        let mut ser = HessianCapture::default();

        for t in 0..n_tokens {
            // Build this token's distinct work-units.
            let mut names: Vec<String> = Vec::new();
            let mut xs: Vec<Vec<f32>> = Vec::new();
            let mut ks: Vec<usize> = Vec::new();
            for e in 0..n_experts {
                let l = 7usize;
                names.push(format!(
                    "model.language_model.layers.{l}.mlp.experts.{e}.gate_up_proj.weight"
                ));
                xs.push(fixed_row(k_gate, (t * 131 + e) as u64));
                ks.push(k_gate);
                names.push(format!(
                    "model.language_model.layers.{l}.mlp.experts.{e}.down_proj.weight"
                ));
                xs.push(fixed_row(k_down, (t * 977 + e + 1) as u64));
                ks.push(k_down);
            }
            // Serial reference: per-item accumulate in fixed order.
            for i in 0..names.len() {
                ser.accumulate(&names[i], &xs[i], ks[i]);
            }
            // Parallel path: one batched call.
            let items: Vec<(String, &[f32], usize)> = (0..names.len())
                .map(|i| (names[i].clone(), xs[i].as_slice(), ks[i]))
                .collect();
            par.accumulate_token(&items);
        }

        // Compare every accumulator bit-for-bit.
        assert_eq!(par.entries.len(), ser.entries.len());
        for (name, pacc) in &par.entries {
            let sacc = ser.entries.get(name).expect("name in serial map");
            assert_eq!(pacc.n_blocks, sacc.n_blocks, "{name} n_blocks");
            assert_eq!(pacc.n_rows, sacc.n_rows, "{name} n_rows");
            for b in 0..pacc.n_blocks {
                let pb = &pacc.blocks[b];
                let sb = &sacc.blocks[b];
                for idx in 0..pb.len() {
                    assert_eq!(
                        pb[idx].to_bits(),
                        sb[idx].to_bits(),
                        "{name} block={b} idx={idx}: parallel {} != serial {} (NOT bit-identical)",
                        pb[idx],
                        sb[idx]
                    );
                }
            }
        }
    }

    #[test]
    fn mq_signs_128_deterministic() {
        let s1 = gen_fwht_signs(43, 128);
        let s2 = gen_fwht_signs(1043, 128);
        assert_eq!(s1.len(), 128);
        assert_eq!(s2.len(), 128);
        for x in &s1 {
            assert!(*x == 1.0 || *x == -1.0, "signs1 contains {x}");
        }
        for x in &s2 {
            assert!(*x == 1.0 || *x == -1.0, "signs2 contains {x}");
        }
        // Reproducibility
        assert_eq!(gen_fwht_signs(43, 128), s1);
        assert_eq!(gen_fwht_signs(1043, 128), s2);
        // Distinct from G256 seeds
        assert_ne!(
            gen_fwht_signs(42, 128),
            s1,
            "seed 43 should differ from seed 42"
        );
    }

    fn try_gpu() -> Option<super::Gpu> {
        super::Gpu::init().ok()
    }

    #[test]
    fn ensure_vmm_cleaned_never_releases_a_live_owner() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        hip_bridge::clear_vmm_faults();
        let tensor = match unsafe { gpu.alloc_vmm_tensor(&[4096], super::DType::Raw, 0, &[]) } {
            Ok(tensor) => tensor,
            Err(_) => {
                eprintln!("skip: VMM unavailable");
                return;
            }
        };
        assert_eq!(gpu.vmm_allocation_count(), 1);

        let err = gpu
            .ensure_vmm_cleaned()
            .expect_err("load cleanup must refuse a still-owned VMM arena");
        assert!(err.to_string().contains("live VMM"), "{err}");
        assert_eq!(gpu.vmm_allocation_count(), 1);
        assert_eq!(gpu.vmm_mapped_bytes(&tensor), Some(0));

        gpu.free_tensor(tensor).expect("free live owner");
        assert_eq!(gpu.vmm_allocation_count(), 0);
    }

    #[test]
    fn free_tensor_unmap_failure_retains_owner_for_retry() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        hip_bridge::clear_vmm_faults();
        let chunk = {
            // Discover granularity via a throwaway arena path.
            let probe = unsafe { gpu.alloc_vmm_tensor(&[4096], super::DType::Raw, 0, &[]) };
            match probe {
                Ok(t) => {
                    let g = gpu.vmm_granularity(&t).unwrap_or(2 * 1024 * 1024);
                    let _ = gpu.free_tensor(t);
                    g
                }
                Err(_) => {
                    eprintln!("skip: VMM unavailable");
                    return;
                }
            }
        };
        let access = [gpu.device_id];
        let tensor = unsafe {
            gpu.alloc_vmm_tensor(&[chunk], super::DType::Raw, chunk, &access)
                .expect("alloc vmm")
        };
        assert_eq!(gpu.vmm_allocation_count(), 1);
        hip_bridge::inject_vmm_fault(hip_bridge::VmmFaultKind::Unmap, 1);
        let err = gpu
            .free_tensor(tensor)
            .expect_err("unmap fault must surface");
        assert!(
            err.to_string().contains("retained") || err.to_string().contains("injected"),
            "{err}"
        );
        assert_eq!(
            gpu.vmm_allocation_count(),
            1,
            "failed free must keep the arena registered"
        );
        // No double-free: retry with faults cleared must release exactly once.
        hip_bridge::clear_vmm_faults();
        gpu.ensure_vmm_cleaned().expect("retry cleanup");
        assert_eq!(gpu.vmm_allocation_count(), 0);
        // Idle ensure is a no-op success (no double free).
        gpu.ensure_vmm_cleaned().expect("second ensure on idle");
    }

    #[test]
    fn free_tensor_release_failure_retains_owner_for_retry() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        hip_bridge::clear_vmm_faults();
        let access = [gpu.device_id];
        let chunk = 2 * 1024 * 1024;
        let tensor =
            match unsafe { gpu.alloc_vmm_tensor(&[chunk], super::DType::Raw, chunk, &access) } {
                Ok(t) => t,
                Err(_) => {
                    eprintln!("skip: VMM unavailable");
                    return;
                }
            };
        assert_eq!(gpu.vmm_allocation_count(), 1);
        hip_bridge::inject_vmm_fault(hip_bridge::VmmFaultKind::Release, 1);
        let err = gpu
            .free_tensor(tensor)
            .expect_err("release fault must surface");
        assert!(
            err.to_string().contains("retained") || err.to_string().contains("injected"),
            "{err}"
        );
        assert_eq!(gpu.vmm_allocation_count(), 1);
        hip_bridge::clear_vmm_faults();
        assert_eq!(gpu.retry_vmm_cleanup().expect("retry"), 0);
        assert_eq!(gpu.vmm_allocation_count(), 0);
    }

    #[test]
    fn access_reset_failure_does_not_publish_live_owner() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        hip_bridge::clear_vmm_faults();
        let access = [gpu.device_id];
        let chunk = 2 * 1024 * 1024;
        let before = gpu.vmm_allocation_count();
        hip_bridge::inject_vmm_fault(hip_bridge::VmmFaultKind::AccessReset, 1);
        let err = match unsafe { gpu.alloc_vmm_tensor(&[chunk], super::DType::Raw, chunk, &access) }
        {
            Ok(tensor) => {
                let _ = gpu.free_tensor(tensor);
                panic!("access-reset fault must fail initial map");
            }
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("injected") || err.to_string().contains("access"),
            "{err}"
        );
        hip_bridge::clear_vmm_faults();
        // Successful cleanup leaves no pending owner; if cleanup itself failed
        // the arena stays registered and ensure_vmm_cleaned must still drain it.
        if gpu.vmm_allocation_count() > before {
            gpu.ensure_vmm_cleaned()
                .expect("drain retained after access fault");
        }
        assert_eq!(gpu.vmm_allocation_count(), before);
        // Subsequent alloc works (no poisoned process state).
        let ok = unsafe { gpu.alloc_vmm_tensor(&[chunk], super::DType::Raw, chunk, &access) }
            .expect("alloc after access fault");
        let _ = gpu.free_tensor(ok).expect("free");
        assert_eq!(gpu.vmm_allocation_count(), before);
    }

    #[test]
    fn ensure_vmm_cleaned_refuses_while_pending() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        hip_bridge::clear_vmm_faults();
        let access = [gpu.device_id];
        let chunk = 2 * 1024 * 1024;
        let tensor =
            match unsafe { gpu.alloc_vmm_tensor(&[chunk], super::DType::Raw, chunk, &access) } {
                Ok(t) => t,
                Err(_) => {
                    eprintln!("skip: VMM unavailable");
                    return;
                }
            };
        hip_bridge::inject_vmm_fault(hip_bridge::VmmFaultKind::Unmap, 2); // free + ensure attempt
        let _ = gpu.free_tensor(tensor).expect_err("fault");
        let err = gpu
            .ensure_vmm_cleaned()
            .expect_err("must refuse while pending");
        assert!(
            err.to_string().contains("pending") || err.to_string().contains("injected"),
            "{err}"
        );
        assert!(gpu.vmm_allocation_count() >= 1);
        hip_bridge::clear_vmm_faults();
        gpu.ensure_vmm_cleaned()
            .expect("clears after faults drained");
        assert_eq!(gpu.vmm_allocation_count(), 0);
    }
}
