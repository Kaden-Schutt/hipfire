// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Leaf GPU value types shared across the workspace.
//!
//! [`GpuTensor`] (a device buffer + shape + [`DType`]) and the [`DType`]
//! element-type enum. Extracted from `rdna-compute` (review 3.11b) so crates
//! that only need to *name* a device tensor or element type depend on this
//! ~200-line leaf (over `hip-bridge`) instead of the whole 63k-LOC compute
//! crate. `rdna-compute` re-exports both, so `rdna_compute::{GpuTensor, DType}`
//! keeps resolving unchanged.

use hip_bridge::DeviceBuffer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    Q4K,       // 144 bytes per 256 elements
    Q6K,       // 210 bytes per 256 elements
    Q8_0,      // 34 bytes per 32 elements
    Q4F16G64,  // 36 bytes per 64 elements (RDNA-native FP16 dequant)
    Q4F16G32,  // 20 bytes per 32 elements (RDNA-native FP16 dequant)
    Q8HFQ,     // split-metadata: scales contiguous then values contiguous, 128B-aligned rows
    HFQ4G256,  // 136 bytes per 256 elements (flat 4-bit, f32 scale+zero, 18 VGPRs)
    HFQ4G128,  // 72 bytes per 128 elements (flat 4-bit, f32 scale+zero, 14 VGPRs)
    HFQ3G256,  // 104 bytes per 256 elements (flat 3-bit, f32 scale+zero)
    HFQ3G128,  // 56 bytes per 128 elements (flat 3-bit, f32 scale+zero)
    MQ4G256,   // MagnumQuant: FWHT-rotated HFQ4-G256 (136 bytes/group, same as HFQ4G256)
    MQ4G128,   // MagnumQuant: FWHT-128-rotated INT4 (72 bytes/group, same layout as HFQ4G128)
    MQ8G256,   // MagnumQuant: FWHT-rotated symmetric INT8, dp4a target (258 bytes/group)
    MQ6G256,   // MagnumQuant: FWHT-rotated HFQ6-G256 (200 bytes/group, same as HFQ6G256)
    MQ3G256,   // MagnumQuant: FWHT-rotated HFQ3-G256 (104 bytes/group, same as HFQ3G256)
    Qtip3G256, // QTIP-3: FWHT-rotated trellis-coded 3-bit (100 bytes/group: f32 scale + 96 B
    // packed symbols). Decoded by gemv_qtip3g256 (computed 1MAD codebook, zero LDS); runtime
    // FWHT-rotates x like MQ3/MQ4. See kernels/src/gemv_qtip3g256.hip / qtip.rs.
    Qtip4G256, // QTIP-4: FWHT-rotated trellis-coded 4-bit (132 bytes/group: f32 scale + 128 B
    // nibble-packed symbols). Same 1MAD codebook/12-bit trellis as Qtip3G256, decoded by
    // gemv_qtip4g256. See kernels/src/gemv_qtip4g256.hip / qtip::pack_qtip4_group.
    MQ2G256,      // MagnumQuant: FWHT-rotated HFQ2-G256 (72 bytes/group, same as HFQ2G256)
    MQ2G256Lloyd, // MagnumQuant 2-bit + Lloyd-Max 4-entry fp16 codebook (72 bytes/group)
    MQ3G256Lloyd, // MagnumQuant 3-bit + Lloyd-Max 8-entry fp16 codebook (112 bytes/group)
    MQ4G256Lloyd, // MagnumQuant 4-bit + Lloyd-Max 16-entry fp16 codebook (160 bytes/group)
    HFP4G32,      // HFP4: E2M1 element + UE8M0 g32 block scale + FP16 row scale.
    // Per-row header 16 B; per-block payload 17 B (UE8M0 + 16 packed nibbles).
    // See docs/quant-formats/hfp4.md.
    MFP4G32, // MFP4: HFP4G32 + offline FWHT (drop-in MQ4 replacement). Same byte layout
    // as HFP4G32; format_flags bit 0 + bits 2-3 = 01 stamps the rotation kind.
    // Runtime applies the matching FWHT to x via mq_rotate_x; the kernel itself
    // is shared with HFP4G32.
    HFQ2G256,   // 72 bytes per 256 elements (flat 2-bit, f32 scale+zero, ~19 VGPRs)
    HFQ2G128,   // 40 bytes per 128 elements (flat 2-bit, f32 scale+zero)
    HFQ6G256,   // 200 bytes per 256 elements (6-bit, f32 scale+zero)
    ParoQ4G128, // ParoQuant: AWQ-packed INT4 G128 repacked to HFQ4G128 layout at load.
    // Weights are standard HFQ4G128 (72 bytes/group); the ParoQuant distinction
    // is that weight_gemv applies Givens rotation to activations before GEMV.
    // Rotation metadata (pairs, theta, channel_scales) lives on WeightTensor::paro.
    Oq4G256, // Opus Quant W4A4: symmetric signed-INT4, FWHT-rotated, per-group f32 scale.
    Oq8G256, // Opus Quant W8A8: symmetric signed-INT8, FWHT-rotated, per-group f32 scale (iu8 WMMA).
    // On-disk storage is [f16 scale][128 nibbles]/256-group (130 B/group, codec
    // `quantize_oq4g256`). The loader repacks to the kernel layout: packed nibbles
    // [M,K/2] followed by per-group f32 scales [M,K/256] in one buffer. The forward
    // quantizes activations to int4 at runtime (`quantize_act_oq4`) and dispatches
    // `gemm_oq4_grouped_wmma` — the only W4A4 (int4-activation) path in the engine.
    W8A8Ref, // Reference kernel layer W8A8: per-channel symmetric int8 weights followed by
    // per-channel f32 scales in one buffer ([M*K int8 | M f32]). A8 activations are
    // quantized per-token at runtime; iu8 WMMA + dequant by w_scale·x_scale. Boring
    // reference (no grouping/rotation) — produced by quantize-on-load (HIPFIRE_W8A8=1).
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
            | DType::MQ8G256
            | DType::MQ3G256
            | DType::Qtip3G256
            | DType::Qtip4G256
            | DType::MQ2G256
            | DType::MQ2G256Lloyd
            | DType::MQ3G256Lloyd
            | DType::MQ4G256Lloyd
            | DType::HFP4G32
            | DType::MFP4G32
            | DType::ParoQ4G128
            | DType::Oq4G256
            | DType::W8A8Ref
            | DType::Oq8G256
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
                // Opus Quant W4A4: SmoothQuant migrates a per-channel scale into the
                // weight offline (W·s); the forward divides x by the awq_scale
                // sidecar before FWHT+int4-quantize, completing (W·s)·(x/s) = W·x.
                | DType::Oq4G256
                // OQ+ / Opus Plus W4A8 loads as Oq8G256 (int4 weights upcast to
                // int8); same AWQ contract — the forward divides x by the sidecar
                // before FWHT+int8-quantize. (Real W8A8 Oq8 has no sidecar → no-op.)
                | DType::Oq8G256
        )
    }
}

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
    /// `FeatureFlags::from_env_for_test` precedent.
    #[doc(hidden)]
    pub fn null_for_test() -> Self {
        GpuTensor {
            buf: unsafe {
                hip_bridge::DeviceBuffer::from_raw(std::ptr::null_mut::<std::ffi::c_void>(), 0)
            },
            shape: vec![0],
            dtype: DType::F32,
        }
    }

    /// Create a non-owning sub-view at a byte offset. For F32 tensors,
    /// `offset_elems` is the number of f32 elements to skip.
    /// The returned tensor is a view — do NOT free it.
    pub fn sub_offset(&self, offset_elems: usize, len_elems: usize) -> GpuTensor {
        let byte_off = offset_elems * self.dtype.size();
        let ptr = unsafe { (self.buf.as_ptr() as *mut u8).add(byte_off) as *mut std::ffi::c_void };
        GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, len_elems * self.dtype.size()) },
            shape: vec![len_elems],
            dtype: self.dtype,
        }
    }
}
