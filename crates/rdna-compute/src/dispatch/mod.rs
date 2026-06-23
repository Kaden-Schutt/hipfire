// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! High-level GPU dispatch interface.
//! Manages compiled kernels, provides typed tensor operations.

use crate::compiler::KernelCompiler;
use crate::feature_flags::FeatureFlags;
use crate::kernels;
use hip_bridge::{DeviceBuffer, HipResult, HipRuntime, Rocblas};
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicUsize};
use std::sync::{Arc, OnceLock};

// Op-family submodules split out of this file (dispatch-refactor Phase 1). Each
// is a child `impl Gpu` block; as a descendant of `dispatch` it reaches Gpu's
// module-private fields without any visibility change.
mod activation;
mod attention;
mod conv1d;
mod deepseek4;
mod embedding;
mod fused;
mod gated;
mod gemm_base;
mod gemm_gate;
mod gemm_hfq;
mod gemm_misc;
mod gemm_qkv;
mod gemv;
mod kv;
mod misc;
mod moe;
mod norm;
mod overlays;
mod quant;
mod rocblas;
mod rope;
mod sampling;

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

thread_local! {
    /// Per-thread cache for `Gpu::bind_thread`. Sentinel `-1` forces the
    /// first call to issue `hipSetDevice` even when the target id is 0.
    static LAST_BOUND_DEVICE: Cell<i32> = const { Cell::new(-1) };
}

/// Current layer index, set by the qwen35 forward_prefill_chunk at the
/// start of each layer iteration. Used by `hfq3_mmq_layer_gate_pass` to
/// support per-layer MMQ-on/off experiments (see issue #302 — KLD
/// attribution sweep). Default 0; no semantic meaning outside an
/// instrumented sweep.
pub static MMQ_CURRENT_LAYER: AtomicUsize = AtomicUsize::new(0);

/// Per-launch entropy for Q8 GatedDeltaNet stochastic rounding.
#[allow(dead_code)]
static GDN_REQUANT_FRAME: AtomicU32 = AtomicU32::new(0);

/// Minimum batch size at which the FP8 WMMA prefill path is enabled.
/// Below this, the FP16 WMMA path wins on gfx1201 (measured 0.71-0.94×
/// at N ≤ 512, 0.82-1.26× only at N ≥ 2048 with high DPM variance —
/// see project_fp8_wmma_hfp4g32_2026_05_10.md). Decode (batch_size=1)
/// must never hit FP8 WMMA. Threshold tuned conservatively; A/B against
/// FP16 WMMA on the production prefill bench can lower it later.
const FP8_WMMA_MIN_BATCH: usize = 1024;

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
const FP8_GEMV_MIN_M: usize = 4096;

/// Tensor stored on the GPU. Tracks shape and element type.
pub struct GpuTensor {
    pub buf: DeviceBuffer,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

macro_rules! moe_scalar_indexed_wrappers {
    ($gate_fn:ident, $down_fn:ident, $gate_kernel:literal, $down_kernel:literal, $stride:expr) => {
        #[allow(clippy::too_many_arguments)]
        pub fn $gate_fn(
            &mut self,
            expert_ptrs: &GpuTensor,
            topk_indices: &GpuTensor,
            x: &GpuTensor,
            y_gate: &GpuTensor,
            y_up: &GpuTensor,
            m: usize,
            k: usize,
            k_top: usize,
            batch_size: usize,
        ) -> HipResult<()> {
            self.bind_thread()?;
            self.gemv_moe_scalar_gate_up_indexed_batched(
                $gate_kernel,
                $stride,
                expert_ptrs,
                topk_indices,
                x,
                y_gate,
                y_up,
                m,
                k,
                k_top,
                batch_size,
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn $down_fn(
            &mut self,
            expert_ptrs: &GpuTensor,
            topk_indices: &GpuTensor,
            rot_batch: &GpuTensor,
            expert_outputs: &GpuTensor,
            m: usize,
            k: usize,
            k_top: usize,
            batch_size: usize,
        ) -> HipResult<()> {
            self.bind_thread()?;
            self.gemv_moe_scalar_down_indexed_batched_expanded(
                $down_kernel,
                $stride,
                expert_ptrs,
                topk_indices,
                rot_batch,
                expert_outputs,
                m,
                k,
                k_top,
                batch_size,
            )
        }
    };
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
            dtype: crate::DType::F32,
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

/// Activation-capture hook for the hipfire-native calibration path.
///
/// The field on `Gpu` is set by the calibration collector
/// (`hipfire_runtime::calibration::CalibCollector`, driven by the
/// `collect_artifacts` example / `hipfire collect-artifacts` / daemon `Collect`
/// op) and called from each linear-layer dispatch site to feed activations into
/// an on-GPU reduction (per-channel `Σ act²` for imatrix, K×K outer-product for the GPTQ
/// Hessian).
///
/// Phase 1 ships only the trait, the `Gpu::capture_handler` field, and
/// a default of `None` — so existing call sites are unaffected. Phase 2
/// threads the `if let Some(h) = &gpu.capture_handler { h.capture(...) }`
/// call into every fused/unfused GEMM dispatch arm.
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

/// High-level GPU context. Owns the HIP runtime, compiler, and loaded kernels.
pub struct Gpu {
    pub hip: HipRuntime,
    pub arch: String,
    pub flags: Arc<FeatureFlags>,
    pub arch_caps: crate::arch_caps::ArchCaps,
    pub device_id: i32,
    /// HIP-reported integrated GPU flag. hipfire treats this as the loader's
    /// UMA signal for deciding whether the slab path should be default-auto.
    pub integrated: bool,
    compiler: KernelCompiler,
    modules: HashMap<String, hip_bridge::Module>,
    functions: HashMap<String, hip_bridge::Function>,
    pool: crate::pool::GpuPool,
    /// Calibration activation capture (Tier-1 collector). When `Some`, the
    /// instrumented linear dispatch arms (`gemv_f16_xf32`, `fused_qkvza_f16_xf32`,
    /// `fused_gate_up_f16_xf32`) resolve their weight buffer pointer to a tensor
    /// name via `capture_names` and invoke `capture()` with the input activation.
    /// `None` (the default) ⇒ the check is a single `is_none()` and forwards are
    /// byte-identical. The collector is held by `Arc` so the dispatch site can
    /// clone it (breaking the borrow on `self`) before calling `capture(self, …)`.
    pub active_capture: Option<Arc<dyn ActivationCapture>>,
    /// Weight-buffer-pointer → canonical tensor name, populated by the loader
    /// when calibration is armed. Lets capture fire from ANY forward path
    /// (hand or lowered, fused or not) keyed by the weight the gemv received.
    pub capture_names: HashMap<usize, String>,
    /// When set, all kernel launches go to this stream instead of null stream.
    pub active_stream: Option<hip_bridge::Stream>,
    /// Task #93 Phase A (2026-04-24): optional secondary streams for
    /// inter-cycle pipelining. `draft_stream` is where a speculatively-
    /// launched draft N+1 runs concurrently with verify N on
    /// `verify_stream`. Left as None until a pipeline-aware caller opts
    /// in via `init_pipeline_streams()`. Currently unused by any caller
    /// — Phase A is a non-behavioral scaffold.
    pub draft_stream: Option<hip_bridge::Stream>,
    pub verify_stream: Option<hip_bridge::Stream>,
    /// MagnumQuant FWHT signs (256 floats each) + rotation scratch buffer.
    pub mq_signs1: Option<GpuTensor>,
    pub mq_signs2: Option<GpuTensor>,
    /// MagnumQuant FWHT signs for G128 (128 floats each, seeds 43 and 1043).
    pub mq_signs1_128: Option<GpuTensor>,
    pub mq_signs2_128: Option<GpuTensor>,
    pub mq_x_rot: Option<GpuTensor>, // scratch for rotated x, sized to max K
    // Opus Quant W4A4 persistent decode scratch (B=1). Hoisted out of the
    // per-projection dispatch so the forward issues ZERO hipMalloc/hipFree inside
    // the (future) hipGraph-captured region — per-call alloc would trip
    // "hipMalloc not permitted under stream capture". Stream-ordered reuse across
    // sequential projections is safe (one stream, in-order). Sized to max K/M.
    pub oq4_xq: Option<GpuTensor>, // packed int4 activation, K/2 bytes
    pub oq4_xs: Option<GpuTensor>, // per-group f32 activation scales, K/256
    pub oq4_xr: Option<GpuTensor>, // rotated f32 activation (Raw paths), K
    pub oq4_ytmp: Option<GpuTensor>, // f32 residual GEMM scratch, M
    // Batched-prefill counterparts: int4-quantized activation for N tokens at
    // once (W4A4 oq4 batched WMMA path). Sized lazily to hold N*K/2 packed
    // nibbles / N*K/256 f32 scales / M*N f32 residual scratch; grown (never
    // shrunk) on demand by `ensure_oq4_scratch_batched`. GpuTensor has no
    // pool-return Drop, so growth leaks the old buffer — bounded because the
    // capacity only ratchets up to the largest prefill chunk seen.
    pub oq4_xq_batch: Option<GpuTensor>, // packed int4 activation, N*K/2 bytes
    pub oq4_xs_batch: Option<GpuTensor>, // per-group f32 activation scales, N*K/256
    pub oq4_ytmp_batch: Option<GpuTensor>, // f32 residual GEMM scratch, M*N
    pub paro_x_scratch: Option<GpuTensor>, // ParoQuant: scratch for rotated activation copy
    pub paro_fused_scratch: Option<Vec<GpuTensor>>, // ParoQuant fused paths: multiple rotation scratch buffers
    pub mq_x_q8: Option<hip_bridge::DeviceBuffer>,  // INT8 quantized rotated x for dp4a
    pub mq_x_scales: Option<hip_bridge::DeviceBuffer>, // per-group f32 scales for x quantization
    /// FP16 scratch buffer for prefill X conversion. Sized to max(batch_size × K) × 2 bytes.
    fp16_x_scratch: Option<hip_bridge::DeviceBuffer>,
    fp16_x_scratch_bytes: usize,
    /// Pointer to the last FP32 source that was converted to fp16_x_scratch.
    /// If the next GEMM uses the same X, skip the conversion.
    pub fp16_x_source_ptr: *mut c_void,
    /// BF16 scratch buffer for WMMA paths that consume raw BF16 operands.
    /// Sized to max(batch_size × K) × 2 bytes and cached by source pointer
    /// like `fp16_x_scratch`.
    bf16_x_scratch: Option<hip_bridge::DeviceBuffer>,
    bf16_x_scratch_bytes: usize,
    bf16_x_source_ptr: *mut c_void,
    /// Displaced FP16/BF16 activation-staging buffers that may still be
    /// referenced by captured graph nodes.
    ///
    /// The shared activation staging scratch is grow-only during normal
    /// dispatch. Graph capture changes the lifetime contract: kernel nodes bake
    /// the staging pointer used at capture time, so replacing the scratch while
    /// any graph is alive would free a pointer the graph may replay. Keep old
    /// staging allocations here until graph state is destroyed.
    capture_staging_scratch: Vec<hip_bridge::DeviceBuffer>,
    /// FP8 (E4M3) scratch buffer for the gfx12 FP8-WMMA prefill path.
    /// Sized to max(batch_size × K) × 1 byte. Cached by src_ptr like
    /// `fp16_x_scratch`.
    fp8_x_scratch: Option<hip_bridge::DeviceBuffer>,
    fp8_x_scratch_bytes: usize,
    fp8_x_source_ptr: *mut c_void,
    /// FP8 (E4M3) sibling of `mq_x_rot`. Filled by
    /// `mq_rotate_x_dual_fp8` so the FP8 decode GEMV can read FP8
    /// activations without a separate pack launch. Lifetime is tied
    /// to mq_x_rot; reallocated together.
    pub mq_x_rot_fp8: Option<hip_bridge::DeviceBuffer>,
    pub mq_x_rot_fp8_bytes: usize,
    /// Q8_1/MMQ scratch for prefill activations. Layout matches llama.cpp's
    /// `block_q8_1_mmq`, ordered by [K/128 block, batch column].
    q8_1_mmq_x_scratch: Option<hip_bridge::DeviceBuffer>,
    q8_1_mmq_x_scratch_bytes: usize,

    // ── MMQ per-weight screening (#87) ──────────────────────────────────
    // When enabled, each weight matrix is screened on first MMQ use: a
    // small synthetic comparison (batch=16, WMMA vs MMQ) checks per-row
    // max abs error. Weights exceeding the threshold fall back to WMMA.
    //
    // Disabled by default on all arches as of 2026-05-18; opt-in for
    // defensive screening when adding new quant formats. Configurable via:
    //   - config.json: `mmq_screen` (bool), `mmq_screen_threshold` (float)
    //   - per-model config overlay
    //   - daemon load params: `mmq_screen`, `mmq_screen_threshold`
    //   - env override: `HIPFIRE_MMQ_SCREEN=1` to enable,
    //     `HIPFIRE_MMQ_SCREEN_THRESHOLD=0.05` to tune
    mmq_screen_cache: HashMap<usize, bool>,
    /// Whether MMQ per-weight screening is enabled. Default: false on all arches.
    pub mmq_screen: bool,
    /// Max per-row abs error threshold for screening. Weights with any row
    /// exceeding this fall back to WMMA.
    /// Per-arch default (set in `Gpu::init`): 0.50 on gfx906, 0.10 elsewhere.
    /// Override via env: `HIPFIRE_MMQ_SCREEN_THRESHOLD`.
    pub mmq_screen_threshold: f32,

    // ── hipGraph capture state ────────────────────────────────────────────
    /// When true, dispatch methods use the blob launch path (graph-capture-safe).
    /// Kernarg blobs are stored in `capture_blobs` and must stay alive until the
    /// captured graph is destroyed.
    pub capture_mode: bool,
    /// Diagnostic: when true, `launch_maybe_blob` takes the blob path even when
    /// `capture_mode=false`. Isolates "blob-vs-kernelParams path" bugs without
    /// the rest of the graph-capture machinery (stream capture, staging, etc).
    /// Heap-stored kernarg blobs for the current capture session. The blob
    /// pointers are baked into the graph at capture time — do NOT clear this
    /// vec until after `graph_exec_destroy`.
    pub capture_blobs: Vec<Vec<u8>>,
    /// The captured graph exec, ready for replay.
    pub graph_exec: Option<hip_bridge::GraphExec>,
    /// The raw captured graph (kept alive for potential update operations).
    captured_graph: Option<hip_bridge::Graph>,
    /// When the captured graph belongs to a verify-forward, this is the batch
    /// size it was captured for. `None` means no verify graph captured (the
    /// graph slot may hold the AR forward graph instead, or be unused).
    /// Used to invalidate + re-capture when the DFlash budget changes mid-run.
    ///
    /// DEPRECATED for verify: the verify path now uses `verify_graph_cache`
    /// keyed by B, keeping separate graphs live for each B value PLD may
    /// oscillate through. This field stays for any legacy single-slot usage.
    pub graph_verify_n: Option<usize>,
    /// Counter of verify forward calls seen since the last graph invalidate.
    /// We run the first call direct (no capture) to let kernel JIT and any
    /// lazy scratch allocations settle — then capture on the second call.
    /// Capturing the first call itself hits "hipMalloc not permitted during
    /// stream capture" the first time a kernel is JITted inside capture.
    ///
    /// DEPRECATED for verify: replaced by `verify_warmed_up` (per-B set).
    pub graph_verify_warmup: u32,

    /// AR `forward_scratch` (single-token decode) capture warmup flag.
    /// First call with `HIPFIRE_GRAPH=1` runs direct so kernel JIT and lazy
    /// AR-forward hipGraph capture/replay state. See block comment near
    /// AR_FORWARD_WARMUP_CALLS in this file for policy.
    /// True on init / after any kernel-module change. The next AR forward
    /// runs direct (no capture) so inline JIT / lazy hipMalloc don't trip
    /// `hipMalloc not permitted under stream capture`.
    pub ar_forward_kernel_dirty: bool,
    /// True after `end_decode_turn()` commits a capture (kernels clean,
    /// graph_exec exists). Replay path enabled for next decode turn until
    /// reset by kernel reload.
    pub ar_forward_replay_enabled: bool,

    /// Per-B cache of captured verify-forward graphs. Each entry owns its
    /// graph + exec + the kernarg blobs that graph captured pointers into.
    /// Blobs must stay alive for the life of the graph — they're baked into
    /// the graph nodes by hipStreamEndCapture.
    ///
    /// Keyed by `b` (draft block size). DFlash's PLD intermittently shortens
    /// b from 16 → 8 on short self-match spines; caching graphs per-B avoids
    /// graph_destroy + re-capture every oscillation, which was wiping out the
    /// hipGraph replay gain entirely.
    pub verify_graph_cache:
        HashMap<usize, (hip_bridge::Graph, hip_bridge::GraphExec, Vec<Vec<u8>>)>,
    /// Subset of `verify_graph_cache` whose captured region also includes the
    /// DFlash verify lm_head + argmax tail. Forward-only and extended verify
    /// graphs share the same per-B cache, so callers need this side metadata
    /// before deciding whether to enqueue lm_head outside the graph.
    pub verify_graph_lmhead_argmax: HashSet<usize>,
    /// Set of B values that have completed the once-per-B JIT/scratch warmup.
    /// Capture can safely begin only after warmup — see graph_verify_warmup doc.
    pub verify_warmed_up: HashSet<usize>,
    /// B being captured right now (between begin_verify_graph_capture and
    /// end_verify_graph_capture). None outside that window.
    verify_capturing_b: Option<usize>,

    /// Per-n_steps cache of captured tape-replay graphs (DeltaNetTape::replay_gdn).
    /// Keyed by n_steps = accept_len + 1 (per-cycle accepted count). On 27B
    /// HumanEval, replay scales linearly with accept — e.g. accept=10 runs
    /// 48 LA layers × 4 kernels = ~192 launches. Graphing collapses those
    /// into one replay. Same shape as verify_graph_cache: graph + exec + blobs.
    pub replay_graph_cache:
        HashMap<usize, (hip_bridge::Graph, hip_bridge::GraphExec, Vec<Vec<u8>>)>,
    /// n_steps values that have completed their once-per-n_steps JIT/scratch warmup.
    pub replay_warmed_up: HashSet<usize>,
    /// n_steps being captured right now. None outside the capture window.
    replay_capturing_n: Option<usize>,

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

    /// Activation-capture hook for the hipfire-native calibration path.
    /// `None` by default — set by the calibration collector
    /// (`CalibCollector`, via `collect_artifacts` / `hipfire collect-artifacts`
    /// / the daemon `Collect` op) when calibration is active, and threaded into
    /// each linear-layer dispatch arm. See `ActivationCapture` trait doc above.
    ///
    /// `Arc<dyn>` so the same handler can be shared across multi-GPU
    /// dispatch threads (one `Gpu` per device, all routing into a single
    /// per-tensor accumulator).
    pub capture_handler: Option<Arc<dyn ActivationCapture>>,
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
    /// Returns the active stream ref for kernel launches (None = null stream).
    fn stream_ref(&self) -> Option<&hip_bridge::Stream> {
        self.active_stream.as_ref()
    }

    /// Bind this `Gpu`'s device on the calling thread. Cached via thread_local
    /// — only issues `hipSetDevice` when the cached id changes.
    #[inline]
    pub fn bind_thread(&self) -> HipResult<()> {
        if LAST_BOUND_DEVICE.with(|c| c.get()) != self.device_id {
            self.hip.set_device(self.device_id)?;
            LAST_BOUND_DEVICE.with(|c| c.set(self.device_id));
        }
        debug_assert_eq!(
            self.hip.current_device()?,
            self.device_id,
            "bind_thread invariant: current device must match self.device_id",
        );
        Ok(())
    }

    /// Debug-only access to the Q8 GatedDeltaNet stochastic requantization
    /// frame. Production paths should consume the frame monotonically; rollback
    /// diagnostics use this to prove whether two otherwise identical replay
    /// paths diverge only because they launched with different stochastic
    /// rounding seeds.
    #[cfg(feature = "deltanet")]
    pub fn debug_gdn_requant_frame(&self) -> u32 {
        // bind_thread: skip — pure atomic state query
        GDN_REQUANT_FRAME.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// See [`Gpu::debug_gdn_requant_frame`].
    #[cfg(feature = "deltanet")]
    pub fn debug_set_gdn_requant_frame(&self, frame: u32) {
        // bind_thread: skip — pure atomic state update
        GDN_REQUANT_FRAME.store(frame, std::sync::atomic::Ordering::Relaxed);
    }

    /// `bind_thread` for `&mut self -> ()` and `Drop` contexts. Logs to
    /// stderr on hipSetDevice failure instead of swallowing it silently;
    /// no debug_assert (would risk panic-in-Drop on top of an unwinding
    /// panic).
    #[inline]
    pub fn bind_thread_or_warn(&self) {
        if LAST_BOUND_DEVICE.with(|c| c.get()) != self.device_id {
            match self.hip.set_device(self.device_id) {
                Ok(()) => LAST_BOUND_DEVICE.with(|c| c.set(self.device_id)),
                Err(e) => eprintln!(
                    "WARN: bind_thread_or_warn(dev {}) failed: {} — \
                     subsequent ops run on the currently-bound device",
                    self.device_id, e,
                ),
            }
        }
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
        eprintln!("[dpm-warmup] running memset loop for {secs:.1}s to pin GPU at high DPM...");
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
        eprintln!(
            "[dpm-warmup] {n} memsets in {elapsed:.2}s ({:.2} ms/iter, {:.1} GiB/s effective)",
            1000.0 * elapsed / n as f32,
            (n as f64 * SCRATCH_BYTES as f64) / (1024.0 * 1024.0 * 1024.0) / elapsed as f64
        );
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
        if let Ok(mode) = std::env::var("HIPFIRE_HIP_WAIT") {
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
        let arch = std::env::var("HIPFIRE_TARGET_ARCH")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or(detected_arch);
        let integrated = hip.is_integrated_device(id).unwrap_or(false);
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

        let flags = Arc::new(FeatureFlags::from_env(&arch));
        let arch_caps = crate::arch_caps::ArchCaps::new(&arch, flags.clone());
        if flags.force_generic {
            // Operator-visible confirmation the reference-floor test cell is
            // active (HIPFIRE_FORCE_GENERIC=1): every is_gfxNNNN() getter is
            // masked, so per-arch whole-method dispatch overlays are skipped.
            eprintln!(
                "[force-generic] HIPFIRE_FORCE_GENERIC=1 on {arch}: per-arch dispatch \
                 overlays DISABLED (reference-floor cell; derived caps WMMA/MMQ/wave intact)"
            );
        }

        let compiler = KernelCompiler::new(&arch, flags.hipcc_extra_flags.clone())?;

        LAST_BOUND_DEVICE.with(|c| c.set(id));

        let mmq_screen = flags.mmq_screen;
        let mmq_screen_threshold = flags.mmq_screen_threshold;

        Ok(Self {
            hip,
            arch,
            flags,
            arch_caps,
            device_id: id,
            integrated,
            compiler,
            modules: HashMap::new(),
            functions: HashMap::new(),
            pool: crate::pool::GpuPool::new(),
            active_capture: None,
            capture_names: HashMap::new(),
            active_stream: None,
            draft_stream: None,
            verify_stream: None,
            mq_signs1: None,
            mq_signs2: None,
            mq_signs1_128: None,
            mq_signs2_128: None,
            mq_x_rot: None,
            oq4_xq: None,
            oq4_xs: None,
            oq4_xr: None,
            oq4_ytmp: None,
            oq4_xq_batch: None,
            oq4_xs_batch: None,
            oq4_ytmp_batch: None,
            paro_x_scratch: None,
            paro_fused_scratch: None,
            mq_x_q8: None,
            mq_x_scales: None,
            fp16_x_scratch: None,
            fp16_x_scratch_bytes: 0,
            fp16_x_source_ptr: std::ptr::null_mut(),
            bf16_x_scratch: None,
            bf16_x_scratch_bytes: 0,
            bf16_x_source_ptr: std::ptr::null_mut(),
            capture_staging_scratch: Vec::new(),
            fp8_x_scratch: None,
            fp8_x_scratch_bytes: 0,
            fp8_x_source_ptr: std::ptr::null_mut(),
            mq_x_rot_fp8: None,
            mq_x_rot_fp8_bytes: 0,
            q8_1_mmq_x_scratch: None,
            q8_1_mmq_x_scratch_bytes: 0,
            mmq_screen_cache: HashMap::new(),
            mmq_screen,
            mmq_screen_threshold,
            capture_mode: false,
            capture_blobs: Vec::new(),
            graph_exec: None,
            captured_graph: None,
            graph_verify_n: None,
            graph_verify_warmup: 0,
            ar_forward_kernel_dirty: true,
            ar_forward_replay_enabled: false,
            verify_graph_cache: HashMap::new(),
            verify_graph_lmhead_argmax: HashSet::new(),
            verify_warmed_up: HashSet::new(),
            verify_capturing_b: None,
            replay_graph_cache: HashMap::new(),
            replay_warmed_up: HashSet::new(),
            replay_capturing_n: None,
            rocblas: None,
            fp16_shadow_cache: HashMap::new(),
            capture_handler: None,
        }).map(|mut gpu| {
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
                    let raw = stream as *const _ as *mut c_void;
                    let _ = rb.set_stream(raw);
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

    // ── hipGraph capture/replay ───────────────────────────────────────────

    /// Begin capturing all kernel launches on the active stream into a graph.
    /// While capturing, dispatch methods that support it will use the blob
    /// launch path so that kernarg pointers survive until graph replay.
    pub fn begin_graph_capture(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        self.capture_blobs.clear();
        self.capture_mode = true;
        let stream = self
            .active_stream
            .as_ref()
            .expect("graph capture requires an explicit stream (not null stream)");
        self.hip.stream_begin_capture(stream, 0) // 0 = hipStreamCaptureModeGlobal
    }

    /// End capture, instantiate the graph for replay.
    pub fn end_graph_capture(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        self.capture_mode = false;
        let stream = self.active_stream.as_ref().unwrap();
        let graph = self.hip.stream_end_capture(stream)?;
        let exec = self.hip.graph_instantiate(&graph)?;
        self.captured_graph = Some(graph);
        self.graph_exec = Some(exec);
        Ok(())
    }

    /// Replay the captured graph.
    pub fn graph_launch(&self) -> HipResult<()> {
        self.bind_thread()?;
        let exec = self
            .graph_exec
            .as_ref()
            .expect("no captured graph to replay");
        let stream = self.active_stream.as_ref().unwrap();
        self.hip.graph_launch(exec, stream)
    }

    fn graph_state_live(&self) -> bool {
        self.capture_mode
            || self.graph_exec.is_some()
            || self.captured_graph.is_some()
            || !self.verify_graph_cache.is_empty()
            || !self.replay_graph_cache.is_empty()
    }

    fn retain_displaced_staging_scratch(&mut self, scratch: Option<hip_bridge::DeviceBuffer>) {
        if let Some(scratch) = scratch {
            if self.graph_state_live() {
                self.capture_staging_scratch.push(scratch);
            }
        }
    }

    fn maybe_release_staging_scratch_keepalive(&mut self) {
        if !self.graph_state_live() {
            self.capture_staging_scratch.clear();
        }
    }

    /// Caller signals end of a decode turn (EOS or max_tokens reached). If a
    /// captured graph exists and kernels are clean, replay is enabled for the
    /// next decode turn. Per the AR-forward hipGraph policy: "at least one
    /// captured full turn must run before replay can be enabled."
    /// No-op if no capture exists (e.g., turn ran fully direct because kernels
    /// were dirty or graph was disabled by the caller).
    pub fn end_decode_turn(&mut self) {
        // bind_thread: skip - pure state (flips replay-enable bool, no GPU calls)
        if !self.ar_forward_kernel_dirty && self.graph_exec.is_some() {
            self.ar_forward_replay_enabled = true;
        }
    }

    /// Drop the currently captured graph (if any) without touching kernel /
    /// replay state. Used by the capture+launch hot-path to free the previous
    /// per-call capture before recording a fresh one — bare `graph_destroy()`
    /// would also mark kernels dirty + disable replay, which is wrong here.
    pub fn drop_captured_graph(&mut self) {
        self.bind_thread_or_warn();
        if let Some(exec) = self.graph_exec.take() {
            let _ = self.hip.graph_exec_destroy(exec);
        }
        if let Some(graph) = self.captured_graph.take() {
            let _ = self.hip.graph_destroy(graph);
        }
        self.capture_blobs.clear();
        self.maybe_release_staging_scratch_keepalive();
    }

    /// Caller signals a kernel-module change (model load, dtype switch, etc).
    /// Forces the next AR forward call to dispatch direct (no capture) so any
    /// inline JIT / lazy hipMalloc happens outside a captured region. Replay
    /// stays disabled until a fresh full turn completes via `end_decode_turn`.
    pub fn mark_kernels_dirty(&mut self) {
        // bind_thread: skip - pure state (flips dirty/replay bools, no GPU calls)
        self.ar_forward_kernel_dirty = true;
        self.ar_forward_replay_enabled = false;
    }

    /// Destroy the captured graph and free all retained kernarg blobs.
    pub fn graph_destroy(&mut self) {
        self.bind_thread_or_warn();
        if let Some(exec) = self.graph_exec.take() {
            let _ = self.hip.graph_exec_destroy(exec);
        }
        if let Some(graph) = self.captured_graph.take() {
            let _ = self.hip.graph_destroy(graph);
        }
        self.capture_blobs.clear();
        self.graph_verify_n = None;
        self.graph_verify_warmup = 0;
        self.maybe_release_staging_scratch_keepalive();
        // Without this, model swap leaves replay enabled, so forward_scratch
        // jumps straight to graph_launch on the new model's stale captured
        // graph (whose kernel pointers reference the OLD model's weights).
        // Mark kernels dirty so the next call goes direct and skips capture
        // until JIT/scratch settles for the new model.
        self.ar_forward_kernel_dirty = true;
        self.ar_forward_replay_enabled = false;
    }

    // ── Per-B verify-forward graph cache ─────────────────────────────────
    //
    // DFlash's PLD intermittently changes b (e.g. 16 → 8 on short self-match
    // spines). With the old single-slot graph API, every b transition triggered
    // `graph_destroy` + warmup + re-capture, wiping out the hipGraph replay
    // gain. These methods cache one graph per distinct b value so oscillation
    // becomes free.

    pub fn verify_has_graph(&self, b: usize) -> bool {
        // bind_thread: skip — pure state query
        self.verify_graph_cache.contains_key(&b)
    }

    pub fn verify_graph_has_lmhead_argmax(&self, b: usize) -> bool {
        // bind_thread: skip — pure state query
        self.verify_graph_lmhead_argmax.contains(&b)
    }

    pub fn verify_mark_graph_lmhead_argmax(&mut self, b: usize) {
        // bind_thread: skip — pure state update
        self.verify_graph_lmhead_argmax.insert(b);
    }

    pub fn verify_needs_warmup(&self, b: usize) -> bool {
        // bind_thread: skip — pure state query
        !self.verify_warmed_up.contains(&b)
    }

    pub fn verify_mark_warmup_done(&mut self, b: usize) {
        // bind_thread: skip — pure state query
        self.verify_warmed_up.insert(b);
    }

    /// Begin capturing a verify-forward graph for batch size `b`. Subsequent
    /// launch_maybe_blob calls will push their kernargs into `capture_blobs`,
    /// which is drained into the per-B cache entry on end_verify_graph_capture.
    pub fn begin_verify_graph_capture(&mut self, b: usize) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert!(
            self.verify_capturing_b.is_none(),
            "begin_verify_graph_capture: already capturing for b={:?}",
            self.verify_capturing_b
        );
        debug_assert!(
            !self.capture_mode,
            "begin_verify_graph_capture: capture_mode already set"
        );
        self.capture_blobs.clear();
        self.verify_capturing_b = Some(b);
        self.capture_mode = true;
        let stream = self
            .active_stream
            .as_ref()
            .expect("verify graph capture requires an explicit stream");
        self.hip.stream_begin_capture(stream, 0) // hipStreamCaptureModeGlobal
    }

    /// End capture, instantiate, stash into the per-B cache (taking ownership
    /// of the current capture_blobs).
    pub fn end_verify_graph_capture(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        let b = self
            .verify_capturing_b
            .take()
            .expect("end_verify_graph_capture without matching begin");
        self.capture_mode = false;
        let stream = self.active_stream.as_ref().unwrap();
        let graph = self.hip.stream_end_capture(stream)?;
        let exec = self.hip.graph_instantiate(&graph)?;
        let blobs = std::mem::take(&mut self.capture_blobs);
        self.verify_graph_cache.insert(b, (graph, exec, blobs));
        Ok(())
    }

    /// Replay the cached verify graph for batch size `b`.
    pub fn verify_graph_launch(&self, b: usize) -> HipResult<()> {
        self.bind_thread()?;
        let entry = self
            .verify_graph_cache
            .get(&b)
            .unwrap_or_else(|| panic!("no captured verify graph for b={}", b));
        let stream = self.active_stream.as_ref().unwrap();
        self.hip.graph_launch(&entry.1, stream)
    }

    /// How many captured verify graphs are in the cache (for debug logs).
    pub fn verify_graph_count(&self) -> usize {
        // bind_thread: skip — pure state query
        self.verify_graph_cache.len()
    }

    /// Destroy all cached verify graphs and their blobs.
    pub fn verify_graph_destroy_all(&mut self) {
        self.bind_thread_or_warn();
        for (_, (graph, exec, _blobs)) in self.verify_graph_cache.drain() {
            let _ = self.hip.graph_exec_destroy(exec);
            let _ = self.hip.graph_destroy(graph);
        }
        self.verify_graph_lmhead_argmax.clear();
        self.verify_warmed_up.clear();
        self.verify_capturing_b = None;
        self.maybe_release_staging_scratch_keepalive();
    }

    // ── Replay-graph cache (tape replay after verify) ────────────────────
    // Same pattern as verify graph, keyed by n_steps instead of B. Captured
    // once per distinct accept_len + 1 seen in a run; reused across cycles.
    // On 27B HumanEval where n_steps hovers around 8-11, this caches 3-4
    // graphs. Per-cycle savings target: 1-3 ms of launch overhead over
    // ~192 kernel dispatches per replay.

    pub fn replay_has_graph(&self, n_steps: usize) -> bool {
        // bind_thread: skip — pure state query
        self.replay_graph_cache.contains_key(&n_steps)
    }

    pub fn replay_needs_warmup(&self, n_steps: usize) -> bool {
        // bind_thread: skip — pure state query
        !self.replay_warmed_up.contains(&n_steps)
    }

    pub fn replay_mark_warmup_done(&mut self, n_steps: usize) {
        // bind_thread: skip — pure state query
        self.replay_warmed_up.insert(n_steps);
    }

    pub fn begin_replay_graph_capture(&mut self, n_steps: usize) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert!(
            self.replay_capturing_n.is_none(),
            "begin_replay_graph_capture: already capturing for n_steps={:?}",
            self.replay_capturing_n
        );
        debug_assert!(
            !self.capture_mode,
            "begin_replay_graph_capture: capture_mode already set"
        );
        self.capture_blobs.clear();
        self.replay_capturing_n = Some(n_steps);
        self.capture_mode = true;
        let stream = self
            .active_stream
            .as_ref()
            .expect("replay graph capture requires an explicit stream");
        self.hip.stream_begin_capture(stream, 0)
    }

    pub fn end_replay_graph_capture(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        let n_steps = self
            .replay_capturing_n
            .take()
            .expect("end_replay_graph_capture without matching begin");
        self.capture_mode = false;
        let stream = self.active_stream.as_ref().unwrap();
        let graph = self.hip.stream_end_capture(stream)?;
        let exec = self.hip.graph_instantiate(&graph)?;
        let blobs = std::mem::take(&mut self.capture_blobs);
        self.replay_graph_cache
            .insert(n_steps, (graph, exec, blobs));
        Ok(())
    }

    pub fn replay_graph_launch(&self, n_steps: usize) -> HipResult<()> {
        self.bind_thread()?;
        let entry = self
            .replay_graph_cache
            .get(&n_steps)
            .unwrap_or_else(|| panic!("no captured replay graph for n_steps={}", n_steps));
        let stream = self.active_stream.as_ref().unwrap();
        self.hip.graph_launch(&entry.1, stream)
    }

    pub fn replay_graph_count(&self) -> usize {
        // bind_thread: skip — pure state query
        self.replay_graph_cache.len()
    }

    pub fn replay_graph_destroy_all(&mut self) {
        self.bind_thread_or_warn();
        for (_, (graph, exec, _blobs)) in self.replay_graph_cache.drain() {
            let _ = self.hip.graph_exec_destroy(exec);
            let _ = self.hip.graph_destroy(graph);
        }
        self.replay_warmed_up.clear();
        self.replay_capturing_n = None;
        self.maybe_release_staging_scratch_keepalive();
    }

    /// D→D copy with offsets that picks async (on the active stream) when
    /// a stream is set and sync otherwise. Captured graphs require async on
    /// the captured stream — sync `hipMemcpy` errors with "would make the
    /// legacy stream depend on a capturing blocking stream" under capture
    /// mode Global. Use this helper whenever the copy might live inside
    /// a captured region.
    pub fn memcpy_dtod_at_auto(
        &self,
        dst: &hip_bridge::DeviceBuffer,
        dst_offset: usize,
        src: &hip_bridge::DeviceBuffer,
        src_offset: usize,
        size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if let Some(stream) = self.active_stream.as_ref() {
            self.hip
                .memcpy_dtod_async_at(dst, dst_offset, src, src_offset, size, stream)
        } else {
            self.hip
                .memcpy_dtod_at(dst, dst_offset, src, src_offset, size)
        }
    }

    /// D→D copy (whole buffer) that picks async on the active stream when set.
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
        if self.capture_mode {
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
    fn launch_maybe_blob(
        &mut self,
        func_name: &str,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        params: &mut Vec<*mut std::ffi::c_void>,
        blob_builder: impl FnOnce() -> hip_bridge::KernargBlob,
    ) -> HipResult<()> {
        if self.capture_mode || self.flags.force_blob_path {
            let mut blob = blob_builder();
            // Pad tail to 16-byte alignment — some kernel struct layouts that
            // HIP's loader expects have an implicit final pad to the struct's
            // alignment. gfx1100 typically doesn't care, but under graph
            // capture on ROCm 7.x the loader is stricter and unpadded tails
            // have been observed to cause silent argument corruption.
            blob.pad_to(16);
            self.capture_blobs.push(blob.into_vec());
            // Re-borrow fields separately to avoid conflicting borrows on self
            let buf = self.capture_blobs.last_mut().unwrap();
            let func = &self.functions[func_name];
            let stream = self
                .active_stream
                .as_ref()
                .map(|s| s as &hip_bridge::Stream);
            unsafe {
                self.hip.launch_kernel_blob(
                    func,
                    grid,
                    block,
                    shared_mem,
                    stream,
                    buf.as_mut_slice(),
                )
            }
        } else {
            let func = &self.functions[func_name];
            let stream = self
                .active_stream
                .as_ref()
                .map(|s| s as &hip_bridge::Stream);
            unsafe {
                self.hip
                    .launch_kernel(func, grid, block, shared_mem, stream, params)
            }
        }
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

    /// Synthetic gfx1151 IU4/IU8 WMMA probe. This is diagnostic-only and is
    /// intentionally not routed into any model path: it validates instruction
    /// availability and accumulator layout before Q4 activation-scratch MMQ
    /// work begins.
    pub fn bench_iu_wmma_gfx1151(
        &mut self,
        output: &GpuTensor,
        blocks: usize,
        iters: usize,
        use_iu4: bool,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let func_name = if use_iu4 {
            "bench_iu4_wmma_gfx1151"
        } else {
            "bench_iu8_wmma_gfx1151"
        };
        self.ensure_kernel(
            "bench_iu4_wmma_gfx1151",
            kernels::BENCH_IU4_WMMA_GFX1151_SRC,
            func_name,
        )?;

        let out_ptr = output.buf.as_ptr();
        let iters_i32 = iters as i32;
        let mut params: Vec<*mut c_void> = vec![
            &out_ptr as *const _ as *mut c_void,
            &iters_i32 as *const _ as *mut c_void,
        ];

        self.launch_maybe_blob(
            func_name,
            [blocks as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(out_ptr);
                b.push_i32(iters_i32);
                b
            },
        )
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
    fn ensure_kernel(&mut self, module_name: &str, source: &str, func_name: &str) -> HipResult<()> {
        if self.functions.contains_key(func_name) {
            return Ok(());
        }

        let obj_path = self.compiler.compile(module_name, source)?;
        let obj_path_str = obj_path.to_str().unwrap().to_string();

        if !self.modules.contains_key(module_name) {
            let module = self.hip.module_load(&obj_path_str)?;
            self.modules.insert(module_name.to_string(), module);
        }

        let module = &self.modules[module_name];
        let func = self.hip.module_get_function(module, func_name)?;
        self.functions.insert(func_name.to_string(), func);
        Ok(())
    }

    /// Ensure the FP16 X scratch contains the current conversion of `x`.
    ///
    /// Decode and prefill scratch tensors reuse stable device pointers while
    /// their contents change after nearly every layer. Pointer-keyed staging is
    /// therefore unsafe for correctness unless every writer invalidates this
    /// cache. Refresh on every call.
    /// Returns the FP16 device pointer.
    fn ensure_fp16_x(&mut self, x: &GpuTensor, n_elems: usize) -> HipResult<*mut c_void> {
        self.ensure_kernel(
            "convert_f32_to_f16",
            kernels::GEMM_HFQ4G256_RESIDUAL_FP16_SRC,
            "convert_f32_to_f16",
        )?;

        let src_ptr = x.buf.as_ptr();
        let needed = n_elems * 2;

        // Grow scratch if needed (never shrinks)
        if self.fp16_x_scratch_bytes < needed {
            let displaced = self.fp16_x_scratch.take();
            self.retain_displaced_staging_scratch(displaced);
            self.fp16_x_scratch = Some(self.hip.malloc(needed)?);
            self.fp16_x_scratch_bytes = needed;
            self.fp16_x_source_ptr = std::ptr::null_mut(); // force reconversion after realloc
        }

        let must_convert = true;
        if must_convert {
            let in_ptr = src_ptr;
            let out_ptr = self.fp16_x_scratch.as_ref().unwrap().as_ptr();
            let n_val = n_elems as i32;
            let mut in_ptr_m = in_ptr;
            let mut out_ptr_m = out_ptr;
            let mut n_val_m = n_val;
            let mut conv_params: Vec<*mut c_void> = vec![
                &mut in_ptr_m as *mut _ as *mut c_void,
                &mut out_ptr_m as *mut _ as *mut c_void,
                &mut n_val_m as *mut _ as *mut c_void,
            ];
            let grid = ((n_elems + 255) / 256) as u32;
            self.launch_maybe_blob(
                "convert_f32_to_f16",
                [grid, 1, 1],
                [256, 1, 1],
                0,
                &mut conv_params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(in_ptr);
                    b.push_ptr(out_ptr);
                    b.push_i32(n_val);
                    b
                },
            )?;
            self.fp16_x_source_ptr = src_ptr;
        }

        Ok(self.fp16_x_scratch.as_ref().unwrap().as_ptr())
    }

    /// Ensure the BF16 X scratch contains the current round-to-nearest-even
    /// conversion of `x`. See `ensure_fp16_x` for why this refreshes even when
    /// the source pointer matches the previous call.
    fn ensure_bf16_x(&mut self, x: &GpuTensor, n_elems: usize) -> HipResult<*mut c_void> {
        self.ensure_kernel(
            "convert_f32_to_bf16",
            kernels::CONVERT_F32_TO_BF16_SRC,
            "convert_f32_to_bf16",
        )?;

        let src_ptr = x.buf.as_ptr();
        let needed = n_elems * 2;

        if self.bf16_x_scratch_bytes < needed {
            let displaced = self.bf16_x_scratch.take();
            self.retain_displaced_staging_scratch(displaced);
            self.bf16_x_scratch = Some(self.hip.malloc(needed)?);
            self.bf16_x_scratch_bytes = needed;
            self.bf16_x_source_ptr = std::ptr::null_mut();
        }

        let must_convert = true;
        if must_convert {
            let in_ptr = src_ptr;
            let out_ptr = self.bf16_x_scratch.as_ref().unwrap().as_ptr();
            let n_val = n_elems as i32;
            let mut in_ptr_m = in_ptr;
            let mut out_ptr_m = out_ptr;
            let mut n_val_m = n_val;
            let mut conv_params: Vec<*mut c_void> = vec![
                &mut in_ptr_m as *mut _ as *mut c_void,
                &mut out_ptr_m as *mut _ as *mut c_void,
                &mut n_val_m as *mut _ as *mut c_void,
            ];
            let grid = ((n_elems + 255) / 256) as u32;
            self.launch_maybe_blob(
                "convert_f32_to_bf16",
                [grid, 1, 1],
                [256, 1, 1],
                0,
                &mut conv_params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(in_ptr);
                    b.push_ptr(out_ptr);
                    b.push_i32(n_val);
                    b
                },
            )?;
            self.bf16_x_source_ptr = src_ptr;
        }

        Ok(self.bf16_x_scratch.as_ref().unwrap().as_ptr())
    }

    /// Convert F32 X to the shared FP16 scratch on every call. Use this when
    /// the source pointer is stable but the contents change between launches.
    fn convert_fp16_x_uncached(&mut self, x: &GpuTensor, n_elems: usize) -> HipResult<*mut c_void> {
        self.ensure_kernel(
            "convert_f32_to_f16",
            kernels::GEMM_HFQ4G256_RESIDUAL_FP16_SRC,
            "convert_f32_to_f16",
        )?;

        let needed = n_elems * 2;
        if self.fp16_x_scratch_bytes < needed {
            let displaced = self.fp16_x_scratch.take();
            self.retain_displaced_staging_scratch(displaced);
            self.fp16_x_scratch = Some(self.hip.malloc(needed)?);
            self.fp16_x_scratch_bytes = needed;
            self.fp16_x_source_ptr = std::ptr::null_mut();
        }

        let in_ptr = x.buf.as_ptr();
        let out_ptr = self.fp16_x_scratch.as_ref().unwrap().as_ptr();
        let n_val = n_elems as i32;
        let mut in_ptr_m = in_ptr;
        let mut out_ptr_m = out_ptr;
        let mut n_val_m = n_val;
        let mut conv_params: Vec<*mut c_void> = vec![
            &mut in_ptr_m as *mut _ as *mut c_void,
            &mut out_ptr_m as *mut _ as *mut c_void,
            &mut n_val_m as *mut _ as *mut c_void,
        ];
        let grid = ((n_elems + 255) / 256) as u32;
        self.launch_maybe_blob(
            "convert_f32_to_f16",
            [grid, 1, 1],
            [256, 1, 1],
            0,
            &mut conv_params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(in_ptr);
                b.push_ptr(out_ptr);
                b.push_i32(n_val);
                b
            },
        )?;
        self.fp16_x_source_ptr = in_ptr;

        Ok(out_ptr)
    }

    /// Ensure the FP8 (E4M3) X scratch contains the conversion of `x`
    /// (an F32 GpuTensor). Returns the FP8 device pointer. gfx12 only —
    /// uses cvt_pk_fp8_f32. Caches by `x.buf.as_ptr()` like its FP16
    /// sibling so back-to-back same-X GEMM dispatches skip reconversion.
    fn ensure_fp8_x(&mut self, x: &GpuTensor, n_elems: usize) -> HipResult<*mut c_void> {
        self.ensure_kernel(
            "pack_f32_to_fp8_gfx12",
            kernels::PACK_F32_TO_FP8_GFX12_SRC,
            "pack_f32_to_fp8_gfx12",
        )?;

        let src_ptr = x.buf.as_ptr();
        let needed = n_elems; // 1 byte per element

        if self.fp8_x_scratch_bytes < needed {
            self.fp8_x_scratch = Some(self.hip.malloc(needed)?);
            self.fp8_x_scratch_bytes = needed;
            self.fp8_x_source_ptr = std::ptr::null_mut();
        }

        let must_convert = self.capture_mode || self.fp8_x_source_ptr != src_ptr;
        if must_convert {
            let in_ptr = src_ptr;
            let out_ptr = self.fp8_x_scratch.as_ref().unwrap().as_ptr();
            let n_val = n_elems as i32;
            let mut in_ptr_m = in_ptr;
            let mut out_ptr_m = out_ptr;
            let mut n_val_m = n_val;
            let mut conv_params: Vec<*mut c_void> = vec![
                &mut in_ptr_m as *mut _ as *mut c_void,
                &mut out_ptr_m as *mut _ as *mut c_void,
                &mut n_val_m as *mut _ as *mut c_void,
            ];
            // 16 elements per thread, 256 threads per block = 4096 elements/block.
            let grid = ((n_elems + 4095) / 4096) as u32;
            self.launch_maybe_blob(
                "pack_f32_to_fp8_gfx12",
                [grid, 1, 1],
                [256, 1, 1],
                0,
                &mut conv_params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(in_ptr);
                    b.push_ptr(out_ptr);
                    b.push_i32(n_val);
                    b
                },
            )?;
            self.fp8_x_source_ptr = src_ptr;
        }

        Ok(self.fp8_x_scratch.as_ref().unwrap().as_ptr())
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
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfq4g256_residual_mmq",
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_SRC,
            "quantize_q8_1_mmq_ds4",
        )?;

        let blocks_k = (k + 127) / 128;
        let block_q8_1_mmq_bytes = 144usize;
        let needed = blocks_k * batch_size * block_q8_1_mmq_bytes;
        if self.q8_1_mmq_x_scratch_bytes < needed {
            self.q8_1_mmq_x_scratch = Some(self.hip.malloc(needed)?);
            self.q8_1_mmq_x_scratch_bytes = needed;
        }

        let src_ptr = x.buf.as_ptr();
        // Unlike the FP16 helper, the same scratch pointer is reused for many
        // different hidden states during prefill. Pointer equality is therefore
        // not a safe freshness test. Higher-level fused MMQ callers quantize
        // once and reuse the returned pointer across sibling projections.
        let must_convert = true;
        if must_convert {
            let out_ptr = self.q8_1_mmq_x_scratch.as_ref().unwrap().as_ptr();
            let mut xp = src_ptr;
            let mut yp = out_ptr;
            let mut k_val = k as i32;
            let mut n_val = batch_size as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut xp as *mut _ as *mut c_void,
                &mut yp as *mut _ as *mut c_void,
                &mut k_val as *mut _ as *mut c_void,
                &mut n_val as *mut _ as *mut c_void,
            ];
            let grid_x = ((k + 1023) / 1024) as u32;
            let grid_y = batch_size as u32;
            self.launch_maybe_blob(
                "quantize_q8_1_mmq_ds4",
                [grid_x, grid_y, 1],
                [256, 1, 1],
                0,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(src_ptr);
                    b.push_ptr(out_ptr);
                    b.push_i32(k_val);
                    b.push_i32(n_val);
                    b
                },
            )?;
        }

        Ok(self.q8_1_mmq_x_scratch.as_ref().unwrap().as_ptr())
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
        if let Some(&safe) = self.mmq_screen_cache.get(&key) {
            return safe;
        }

        let screen_batch = 16usize;
        let threshold = self.mmq_screen_threshold;

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

            let saved_capture = self.capture_mode;
            self.capture_mode = true;

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

            self.capture_mode = saved_capture;
            self.hip.device_synchronize()?;

            let ref_out = self.download_f32(&y_wmma)?;
            let mmq_out = self.download_f32(&y_mmq)?;

            self.free_tensor(x_gpu).ok();
            self.free_tensor(y_wmma).ok();
            self.free_tensor(y_mmq).ok();

            // Per-row max error check
            let mut worst_row = 0usize;
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
                    worst_row = r;
                }
            }

            let safe = worst_err <= threshold;
            if !safe {
                eprintln!(
                    "  MMQ screen: UNSAFE weight ptr={key:#x} m={m} k={k} \
                     worst_row={worst_row} max_err={worst_err:.4} > threshold={threshold:.4} — falling back to WMMA"
                );
            }
            Ok(safe)
        })();

        let safe = result.unwrap_or_else(|e| {
            eprintln!("  MMQ screen: error during screening ({e}), assuming unsafe");
            false
        });
        self.mmq_screen_cache.insert(key, safe);
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
    fn ensure_fp16_shadow(
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
                let module = self.hip.module_load(&obj_path_str)?;
                self.modules.insert(module_name.to_string(), module);
            }
            let module = &self.modules[module_name];
            let func = self.hip.module_get_function(module, func_name)?;
            self.functions.insert(func_name.to_string(), func);
        }
        Ok(())
    }

    // ── Tensor allocation ───────────────────────────────────────

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

    /// Download `n_bytes` of raw device memory to host bytes. Dtype-agnostic;
    /// used by the generic-kernel-library tests to read back BF16/F16/I32
    /// outputs without a dedicated typed helper.
    pub fn download_raw(&self, tensor: &GpuTensor, n_bytes: usize) -> HipResult<Vec<u8>> {
        self.bind_thread()?;
        let mut data = vec![0u8; n_bytes];
        self.hip.memcpy_dtoh(&mut data, &tensor.buf)?;
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

    pub fn free_tensor(&mut self, tensor: GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        self.pool.free(tensor.buf);
        Ok(())
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
        self.mmq_screen_cache.clear();
        let shadows: Vec<GpuTensor> = self.fp16_shadow_cache.drain().map(|(_, t)| t).collect();
        for t in shadows {
            let _ = self.free_tensor(t);
        }
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
        self.graph_destroy();
        self.verify_graph_destroy_all();
        self.replay_graph_destroy_all();
    }

    /// Drop captured graph state after a live KV layout switch so the next
    /// forward captures the current K/V modes and kernarg blobs.
    pub fn invalidate_for_kv_mode_switch(&mut self) {
        // bind_thread: skip — delegates to invalidate_graph_state(), which binds.
        self.invalidate_graph_state();
    }

    // ── Kernel operations ───────────────────────────────────────

    /// Ensure the ParoQuant activation scratch buffer is allocated (F32, sized for dim).
    pub fn ensure_paro_scratch(&mut self, dim: usize) -> HipResult<()> {
        self.bind_thread()?;
        if let Some(ref s) = self.paro_x_scratch {
            if s.buf.size() >= dim * 4 {
                return Ok(());
            }
        }
        let buf = self.hip.malloc(dim * 4)?; // F32
        self.paro_x_scratch = Some(GpuTensor {
            buf,
            shape: vec![dim],
            dtype: DType::F32,
        });
        Ok(())
    }

    /// Ensure the ParoQuant fused rotation scratch buffers are allocated.
    ///
    /// Fused QKVZA needs four independent rotated activation buffers; gate+up
    /// uses the first explicit buffer and aliases `mq_x_rot` internally for the
    /// second rotation.
    pub fn ensure_paro_fused_scratch(&mut self, dim: usize) -> HipResult<()> {
        self.bind_thread()?;
        if let Some(ref scratch) = self.paro_fused_scratch {
            if scratch.len() >= 4 && scratch.iter().all(|s| s.buf.size() >= dim * 4) {
                return Ok(());
            }
        }
        let mut scratch = Vec::with_capacity(4);
        for _ in 0..4 {
            scratch.push(self.alloc_tensor(&[dim], DType::F32)?);
        }
        self.paro_fused_scratch = Some(scratch);
        Ok(())
    }

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

    /// Lazily initialize MagnumQuant FWHT sign tables (256 floats each, seeds 42 and 1042).
    pub fn ensure_mq_signs(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        if self.mq_signs1.is_some() {
            return Ok(());
        }
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let s1b: Vec<u8> = s1.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let s2b: Vec<u8> = s2.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let s1t = self.alloc_tensor(&[256], DType::F32)?;
        let s2t = self.alloc_tensor(&[256], DType::F32)?;
        self.hip.memcpy_htod(&s1t.buf, &s1b)?;
        self.hip.memcpy_htod(&s2t.buf, &s2b)?;
        // Allocate scratch buffers — 32K elements covers K up to 32768
        let x_rot = self.alloc_tensor(&[32768], DType::F32)?;
        let x_q8 = self.hip.malloc(32768)?; // INT8 buffer for dp4a
        let x_scales = self.hip.malloc(128 * 4)?; // up to 128 groups × f32
        self.mq_signs1 = Some(s1t);
        self.mq_signs2 = Some(s2t);
        self.mq_x_rot = Some(x_rot);
        self.mq_x_q8 = Some(x_q8);
        self.mq_x_scales = Some(x_scales);
        Ok(())
    }

    /// Lazily allocate the Opus Quant W4A4 persistent decode scratch (B=1).
    /// Sized to 32768-element max (K/M ≤ 32768) — mirrors `mq_x_rot`. Idempotent.
    /// Callers alias these (`oq4_xq`/`oq4_xs`/`oq4_xr`/`oq4_ytmp`) so the per-token
    /// forward does NO hipMalloc/hipFree → hipGraph-capture-clean.
    pub fn ensure_oq4_scratch(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        if self.oq4_xq.is_some() {
            return Ok(());
        }
        self.oq4_xq = Some(self.alloc_tensor(&[16384], DType::Raw)?); // K/2 ≤ 16384
        self.oq4_xs = Some(self.alloc_tensor(&[128], DType::F32)?); // K/256 ≤ 128
        self.oq4_xr = Some(self.alloc_tensor(&[32768], DType::F32)?); // K ≤ 32768
        self.oq4_ytmp = Some(self.alloc_tensor(&[32768], DType::F32)?); // M ≤ 32768
        Ok(())
    }

    /// Ensure the batched-prefill int4-activation scratch holds `n` tokens of a
    /// K-wide activation: `oq4_xq_batch` ≥ n*k/2 bytes (packed nibbles),
    /// `oq4_xs_batch` ≥ n*k/256 f32 scales, `oq4_ytmp_batch` ≥ m_max*n f32
    /// residual scratch. Capacity ratchets up (never shrinks) so repeated chunks
    /// of the same or smaller size reuse one allocation. `m_max` is the largest
    /// output-row count fed to a residual GEMM in this layer family (pass the
    /// model hidden_dim — every wo/w_down output ≤ dim for the dense path).
    pub fn ensure_oq4_scratch_batched(
        &mut self,
        n: usize,
        k: usize,
        m_max: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let need_xq = n * (k / 2);
        let need_xs = n * (k / 256);
        let need_y = n * m_max;
        let grow = |cur: &Option<GpuTensor>, need: usize| -> bool {
            cur.as_ref().map(|t| t.numel() < need).unwrap_or(true)
        };
        if grow(&self.oq4_xq_batch, need_xq) {
            self.oq4_xq_batch = Some(self.alloc_tensor(&[need_xq], DType::Raw)?);
        }
        if grow(&self.oq4_xs_batch, need_xs) {
            self.oq4_xs_batch = Some(self.alloc_tensor(&[need_xs], DType::F32)?);
        }
        if grow(&self.oq4_ytmp_batch, need_y) {
            self.oq4_ytmp_batch = Some(self.alloc_tensor(&[need_y], DType::F32)?);
        }
        Ok(())
    }

    /// Lazily initialize MagnumQuant FWHT sign tables for G128 (128 floats each, seeds 43 and 1043).
    /// Also allocates the shared `mq_x_rot` scratch if not already present — the G256 path
    /// (`ensure_mq_signs`) normally owns that allocation, but the G128 path must be
    /// self-sufficient so models that carry only MQ4G128 weights still get the scratch buffer.
    pub fn ensure_mq_signs_128(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        if self.mq_signs1_128.is_some() && self.mq_x_rot.is_some() {
            return Ok(());
        }
        if self.mq_signs1_128.is_none() {
            let signs1 = gen_fwht_signs(43, 128);
            let signs2 = gen_fwht_signs(1043, 128);
            let s1b: Vec<u8> = signs1.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let s2b: Vec<u8> = signs2.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let s1t = self.alloc_tensor(&[128], DType::F32)?;
            let s2t = self.alloc_tensor(&[128], DType::F32)?;
            self.hip.memcpy_htod(&s1t.buf, &s1b)?;
            self.hip.memcpy_htod(&s2t.buf, &s2b)?;
            self.mq_signs1_128 = Some(s1t);
            self.mq_signs2_128 = Some(s2t);
        }
        // Allocate shared rotation scratch if ensure_mq_signs (G256 path) has not run yet.
        // 32K elements covers K up to 32768, matching ensure_mq_signs's allocation.
        if self.mq_x_rot.is_none() {
            let x_rot = self.alloc_tensor(&[32768], DType::F32)?;
            self.mq_x_rot = Some(x_rot);
        }
        Ok(())
    }

    /// Invalidate any `ensure_*_x` caches whose source pointer matches
    /// `dst_ptr`. Must be called by any kernel that overwrites data at
    /// `dst_ptr` since the caches key on raw pointer equality and have
    /// no way to detect data changes otherwise. The `mq_x_rot` scratch
    /// buffer used by the MagnumQuant rotation wrappers is the canonical
    /// case — its pointer is stable across all gemv calls but its data
    /// changes per rotation; without this invalidation, the FP8/FP16
    /// activation scratch returns stale data on every call after the
    /// first within a forward pass (silent correctness bug — coherence
    /// detectors miss it because output stays vaguely on-topic).
    fn invalidate_x_caches_for(&mut self, dst_ptr: *mut c_void) {
        if self.fp16_x_source_ptr == dst_ptr {
            self.fp16_x_source_ptr = std::ptr::null_mut();
        }
        if self.bf16_x_source_ptr == dst_ptr {
            self.bf16_x_source_ptr = std::ptr::null_mut();
        }
        if self.fp8_x_source_ptr == dst_ptr {
            self.fp8_x_source_ptr = std::ptr::null_mut();
        }
    }

    // HFQ2 GEMV dispatch already exists at line ~521 from the HFQ family

    fn launch_hfq3_mmq_tile(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        mmq_x: usize,
        kernel_name: &'static str,
        src: &'static str,
    ) -> HipResult<()> {
        self.launch_hfq3_mmq_tile_with_y(
            a_raw,
            x,
            y,
            m,
            k,
            batch_size,
            mmq_x,
            128,
            kernel_name,
            src,
        )
    }

    /// MMQ_Y-parameterized variant of `launch_hfq3_mmq_tile`. The body.cuh
    /// lets wrappers override the row-tile size for occupancy probes while
    /// preserving the same x-quantization path.
    #[allow(clippy::too_many_arguments)]
    fn launch_hfq3_mmq_tile_with_y(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        mmq_x: usize,
        mmq_y: usize,
        kernel_name: &'static str,
        src: &'static str,
    ) -> HipResult<()> {
        // Inline the body .cuh — same pattern as the gfx906 MMQ family.
        let inlined = src.replace(
            "#include \"gemm_hfq3g256_residual_mmq_body.cuh\"",
            kernels::GEMM_HFQ3G256_RESIDUAL_MMQ_BODY_CUH,
        );
        self.ensure_kernel(kernel_name, &inlined, kernel_name)?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        let func = &self.functions[kernel_name];

        let mut ap = a_raw.buf.as_ptr();
        let mut xq = xq_ptr;
        let mut yp = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        // LDS layout — must match the body.cuh constants:
        //   x_qs: mmq_y × X_STRIDE(40) ints + x_dm: mmq_y × float2
        //   tile_y: mmq_x × Y_STRIDE(36) ints
        const X_STRIDE: usize = 40;
        const Y_STRIDE: usize = 36;
        let shared_mem = (mmq_y * X_STRIDE * 4 + mmq_y * 8 + mmq_x * Y_STRIDE * 4) as u32;

        let row_tiles = (m + mmq_y - 1) / mmq_y;
        let col_tiles = (batch_size + mmq_x - 1) / mmq_x;

        let bytes = crate::profile::gemm_hfq3g256_bytes(m, k, batch_size)
            + batch_size * k
            + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [row_tiles as u32, col_tiles as u32, 1],
                [32, 4, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    // ── HFQ3 qkv MMQ family — 3-way fused (Q + K + V) ────────────────────
    //
    // Auto-selector picks tile size by batch_size, falling back to dot2 at
    // small N. Same gate boundaries as the residual family from the
    // bench_hfq3_mmq_sweep microbench.

    #[allow(clippy::too_many_arguments)]
    fn launch_qkv_hfq3_mmq_tile(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
        mmq_x: usize,
        kernel_name: &'static str,
        src: &'static str,
    ) -> HipResult<()> {
        let inlined = src.replace(
            "#include \"gemm_qkv_hfq3g256_mmq_body.cuh\"",
            kernels::GEMM_QKV_HFQ3G256_MMQ_BODY_CUH,
        );
        self.ensure_kernel(kernel_name, &inlined, kernel_name)?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        let func = &self.functions[kernel_name];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xq = xq_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        const MMQ_Y: usize = 128;
        const X_STRIDE: usize = 40;
        const Y_STRIDE: usize = 36;
        let shared_mem = (MMQ_Y * X_STRIDE * 4 + MMQ_Y * 8 + mmq_x * Y_STRIDE * 4) as u32;

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + MMQ_Y - 1) / MMQ_Y;
        let col_tiles = (batch_size + mmq_x - 1) / mmq_x;

        let bytes = crate::profile::gemm_hfq3g256_bytes(q_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(k_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(v_m, k, batch_size)
            + batch_size * k
            + batch_size * total_m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [row_tiles as u32, col_tiles as u32, 1],
                [32, 4, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    // ── HFQ3 gate_up MMQ family — 2-way fused ─────────────────────────────

    #[allow(clippy::too_many_arguments)]
    fn launch_gate_up_hfq3_mmq_tile(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
        mmq_x: usize,
        kernel_name: &'static str,
        src: &'static str,
    ) -> HipResult<()> {
        self.launch_gate_up_hfq3_mmq_tile_with_y(
            a_gate,
            a_up,
            x,
            y_gate,
            y_up,
            gate_m,
            up_m,
            k,
            batch_size,
            mmq_x,
            128,
            kernel_name,
            src,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_gate_up_hfq3_mmq_tile_with_y(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
        mmq_x: usize,
        mmq_y: usize,
        kernel_name: &'static str,
        src: &'static str,
    ) -> HipResult<()> {
        let inlined = src.replace(
            "#include \"gemm_gate_up_hfq3g256_mmq_body.cuh\"",
            kernels::GEMM_GATE_UP_HFQ3G256_MMQ_BODY_CUH,
        );
        self.ensure_kernel(kernel_name, &inlined, kernel_name)?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        let func = &self.functions[kernel_name];

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xq = xq_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut gate_m_val = gate_m as i32;
        let mut up_m_val = up_m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut gate_m_val as *mut _ as *mut c_void,
            &mut up_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        const X_STRIDE: usize = 40;
        const Y_STRIDE: usize = 36;
        let shared_mem = (mmq_y * X_STRIDE * 4 + mmq_y * 8 + mmq_x * Y_STRIDE * 4) as u32;

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + mmq_y - 1) / mmq_y;
        let col_tiles = (batch_size + mmq_x - 1) / mmq_x;

        let bytes = crate::profile::gemm_hfq3g256_bytes(gate_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(up_m, k, batch_size)
            + batch_size * k
            + batch_size * total_m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [row_tiles as u32, col_tiles as u32, 1],
                [32, 4, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    // ── HFQ3 qkvza MMQ family — 4-way fused LinearAttention preamble ─────

    #[allow(clippy::too_many_arguments)]
    fn launch_qkvza_hfq3_mmq_tile(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
        mmq_x: usize,
        kernel_name: &'static str,
        src: &'static str,
    ) -> HipResult<()> {
        let inlined = src.replace(
            "#include \"gemm_qkvza_hfq3g256_mmq_body.cuh\"",
            kernels::GEMM_QKVZA_HFQ3G256_MMQ_BODY_CUH,
        );
        self.ensure_kernel(kernel_name, &inlined, kernel_name)?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        let func = &self.functions[kernel_name];

        let mut aqkv = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xq = xq_ptr;
        let mut yqkv = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut qkv_m_val = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut beta_m_val = beta_m as i32;
        let mut alpha_m_val = alpha_m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aqkv as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &mut yqkv as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut qkv_m_val as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut beta_m_val as *mut _ as *mut c_void,
            &mut alpha_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        const MMQ_Y: usize = 128;
        const X_STRIDE: usize = 40;
        const Y_STRIDE: usize = 36;
        let shared_mem = (MMQ_Y * X_STRIDE * 4 + MMQ_Y * 8 + mmq_x * Y_STRIDE * 4) as u32;

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + MMQ_Y - 1) / MMQ_Y;
        let col_tiles = (batch_size + mmq_x - 1) / mmq_x;

        let bytes = crate::profile::gemm_hfq3g256_bytes(qkv_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(z_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(beta_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(alpha_m, k, batch_size)
            + batch_size * k
            + batch_size * total_m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [row_tiles as u32, col_tiles as u32, 1],
                [32, 4, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    // ── HFQ4 RDNA2 tiled MMQ families (issue #299) ─────────────────────
    fn launch_hfq4_mmq_tile_with_y(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        mmq_x: usize,
        mmq_y: usize,
        kernel_name: &'static str,
        src: &'static str,
    ) -> HipResult<()> {
        let inlined = src.replace(
            "#include \"gemm_hfq4g256_residual_mmq_body.cuh\"",
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_BODY_CUH,
        );
        self.ensure_kernel(kernel_name, &inlined, kernel_name)?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        let func = &self.functions[kernel_name];

        let mut ap = a_raw.buf.as_ptr();
        let mut xq = xq_ptr;
        let mut yp = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        const X_STRIDE: usize = 40;
        const Y_STRIDE: usize = 36;
        let shared_mem = (mmq_y * X_STRIDE * 4 + mmq_y * 8 + mmq_x * Y_STRIDE * 4) as u32;
        let row_tiles = (m + mmq_y - 1) / mmq_y;
        let col_tiles = (batch_size + mmq_x - 1) / mmq_x;

        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [row_tiles as u32, col_tiles as u32, 1],
                [32, 4, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    // ── HFQ4 qkv MMQ family (3-way fused Q+K+V) — issue #299 Phase 2 ────
    #[allow(clippy::too_many_arguments)]
    fn launch_qkv_hfq4_mmq_tile(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
        mmq_x: usize,
        kernel_name: &'static str,
        src: &'static str,
    ) -> HipResult<()> {
        let inlined = src.replace(
            "#include \"gemm_qkv_hfq4g256_mmq_body.cuh\"",
            kernels::GEMM_QKV_HFQ4G256_MMQ_BODY_CUH,
        );
        self.ensure_kernel(kernel_name, &inlined, kernel_name)?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        let func = &self.functions[kernel_name];

        let mut a_q_p = a_q.buf.as_ptr();
        let mut a_k_p = a_k.buf.as_ptr();
        let mut a_v_p = a_v.buf.as_ptr();
        let mut xq = xq_ptr;
        let mut y_q_p = y_q.buf.as_ptr();
        let mut y_k_p = y_k.buf.as_ptr();
        let mut y_v_p = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_q_p as *mut _ as *mut c_void,
            &mut a_k_p as *mut _ as *mut c_void,
            &mut a_v_p as *mut _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &mut y_q_p as *mut _ as *mut c_void,
            &mut y_k_p as *mut _ as *mut c_void,
            &mut y_v_p as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        const MMQ_Y: usize = 128;
        const X_STRIDE: usize = 40;
        const Y_STRIDE: usize = 36;
        let shared_mem = (MMQ_Y * X_STRIDE * 4 + MMQ_Y * 8 + mmq_x * Y_STRIDE * 4) as u32;
        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + MMQ_Y - 1) / MMQ_Y;
        let col_tiles = (batch_size + mmq_x - 1) / mmq_x;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
            + crate::profile::gemv_hfq4g256_bytes(k_m, k)
            + crate::profile::gemv_hfq4g256_bytes(v_m, k)
            + batch_size * k
            + batch_size * (q_m + k_m + v_m) * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [row_tiles as u32, col_tiles as u32, 1],
                [32, 4, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    // ── HFQ4 gate_up MMQ family (2-way fused gate+up) — issue #299 Phase 3 ──
    #[allow(clippy::too_many_arguments)]
    fn launch_gate_up_hfq4_mmq_tile(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
        mmq_x: usize,
        kernel_name: &'static str,
        src: &'static str,
    ) -> HipResult<()> {
        let inlined = src.replace(
            "#include \"gemm_gate_up_hfq4g256_mmq_body.cuh\"",
            kernels::GEMM_GATE_UP_HFQ4G256_MMQ_BODY_CUH,
        );
        self.ensure_kernel(kernel_name, &inlined, kernel_name)?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        let func = &self.functions[kernel_name];

        let mut a_gate_p = a_gate.buf.as_ptr();
        let mut a_up_p = a_up.buf.as_ptr();
        let mut xq = xq_ptr;
        let mut y_gate_p = y_gate.buf.as_ptr();
        let mut y_up_p = y_up.buf.as_ptr();
        let mut gate_m_val = gate_m as i32;
        let mut up_m_val = up_m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_gate_p as *mut _ as *mut c_void,
            &mut a_up_p as *mut _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &mut y_gate_p as *mut _ as *mut c_void,
            &mut y_up_p as *mut _ as *mut c_void,
            &mut gate_m_val as *mut _ as *mut c_void,
            &mut up_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        const MMQ_Y: usize = 128;
        const X_STRIDE: usize = 40;
        const Y_STRIDE: usize = 36;
        let shared_mem = (MMQ_Y * X_STRIDE * 4 + MMQ_Y * 8 + mmq_x * Y_STRIDE * 4) as u32;
        let total_m = gate_m + up_m;
        let row_tiles = (total_m + MMQ_Y - 1) / MMQ_Y;
        let col_tiles = (batch_size + mmq_x - 1) / mmq_x;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
            + crate::profile::gemv_hfq4g256_bytes(up_m, k)
            + batch_size * k
            + batch_size * (gate_m + up_m) * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [row_tiles as u32, col_tiles as u32, 1],
                [32, 4, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    // ── HFQ4 qkvza MMQ family (4-way fused LA preamble) — issue #299 Phase 4 ──
    #[allow(clippy::too_many_arguments)]
    fn launch_qkvza_hfq4_mmq_tile(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
        mmq_x: usize,
        kernel_name: &'static str,
        src: &'static str,
    ) -> HipResult<()> {
        let inlined = src.replace(
            "#include \"gemm_qkvza_hfq4g256_mmq_body.cuh\"",
            kernels::GEMM_QKVZA_HFQ4G256_MMQ_BODY_CUH,
        );
        self.ensure_kernel(kernel_name, &inlined, kernel_name)?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        let func = &self.functions[kernel_name];

        let mut a_qkv_p = a_qkv.buf.as_ptr();
        let mut a_z_p = a_z.buf.as_ptr();
        let mut a_beta_p = a_beta.buf.as_ptr();
        let mut a_alpha_p = a_alpha.buf.as_ptr();
        let mut xq = xq_ptr;
        let mut y_qkv_p = y_qkv.buf.as_ptr();
        let mut y_z_p = y_z.buf.as_ptr();
        let mut y_beta_p = y_beta.buf.as_ptr();
        let mut y_alpha_p = y_alpha.buf.as_ptr();
        let mut qkv_m_val = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut beta_m_val = beta_m as i32;
        let mut alpha_m_val = alpha_m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_qkv_p as *mut _ as *mut c_void,
            &mut a_z_p as *mut _ as *mut c_void,
            &mut a_beta_p as *mut _ as *mut c_void,
            &mut a_alpha_p as *mut _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &mut y_qkv_p as *mut _ as *mut c_void,
            &mut y_z_p as *mut _ as *mut c_void,
            &mut y_beta_p as *mut _ as *mut c_void,
            &mut y_alpha_p as *mut _ as *mut c_void,
            &mut qkv_m_val as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut beta_m_val as *mut _ as *mut c_void,
            &mut alpha_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        const MMQ_Y: usize = 128;
        const X_STRIDE: usize = 40;
        const Y_STRIDE: usize = 36;
        let shared_mem = (MMQ_Y * X_STRIDE * 4 + MMQ_Y * 8 + mmq_x * Y_STRIDE * 4) as u32;
        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + MMQ_Y - 1) / MMQ_Y;
        let col_tiles = (batch_size + mmq_x - 1) / mmq_x;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
            + crate::profile::gemv_hfq4g256_bytes(z_m, k)
            + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
            + crate::profile::gemv_hfq4g256_bytes(alpha_m, k)
            + batch_size * k
            + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [row_tiles as u32, col_tiles as u32, 1],
                [32, 4, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    moe_scalar_indexed_wrappers!(
        gemv_hfq2g256_moe_gate_up_k8_indexed_batched,
        gemv_hfq2g256_moe_down_k8_indexed_batched_expanded,
        "gemv_hfq2g256_moe_gate_up_k8_indexed_batched",
        "gemv_hfq2g256_moe_down_k8_indexed_batched_expanded",
        72
    );
    moe_scalar_indexed_wrappers!(
        gemv_hfq8g256_moe_gate_up_k8_indexed_batched,
        gemv_hfq8g256_moe_down_k8_indexed_batched_expanded,
        "gemv_hfq8g256_moe_gate_up_k8_indexed_batched",
        "gemv_hfq8g256_moe_down_k8_indexed_batched_expanded",
        258
    );
    moe_scalar_indexed_wrappers!(
        gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_batched,
        gemv_mq2g256_lloyd_moe_down_k8_indexed_batched_expanded,
        "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_batched",
        "gemv_mq2g256_lloyd_moe_down_k8_indexed_batched_expanded",
        72
    );
    moe_scalar_indexed_wrappers!(
        gemv_mq3g256_lloyd_moe_gate_up_k8_indexed_batched,
        gemv_mq3g256_lloyd_moe_down_k8_indexed_batched_expanded,
        "gemv_mq3g256_lloyd_moe_gate_up_k8_indexed_batched",
        "gemv_mq3g256_lloyd_moe_down_k8_indexed_batched_expanded",
        112
    );

    fn hfq4g256_mmq_gfx1151_enabled(&self, m: usize, k: usize, batch_size: usize) -> bool {
        static MODE: OnceLock<Option<bool>> = OnceLock::new();
        if !self.arch.starts_with("gfx1151")
            || batch_size < 16
            || batch_size % 16 != 0
            || m % 16 != 0
            || k % 256 != 0
        {
            return false;
        }

        match *MODE.get_or_init(
            || match std::env::var("HIPFIRE_HFQ4G256_MMQ_GFX1151").as_deref() {
                Ok("0") => Some(false),
                Ok("1") => Some(true),
                _ => None,
            },
        ) {
            Some(force) => force,
            None => k == 2048,
        }
    }

    fn gfx1151_fused_hfq4_2row_enabled(&self) -> bool {
        static MODE: OnceLock<Option<bool>> = OnceLock::new();
        if !self.arch.starts_with("gfx1151") {
            return false;
        }
        match *MODE.get_or_init(|| {
            match std::env::var("HIPFIRE_FUSED_HFQ4_2ROW_GFX1151").as_deref() {
                Ok("0") => Some(false),
                Ok("1") => Some(true),
                _ => None,
            }
        }) {
            Some(force) => force,
            None => false,
        }
    }

    fn gfx1151_moe_indexed_2row_enabled(&self) -> bool {
        static MODE: OnceLock<Option<bool>> = OnceLock::new();
        if !self.arch.starts_with("gfx1151") {
            return false;
        }
        match *MODE.get_or_init(|| {
            // Opt-in ("1") gfx1151 two-row indexed MoE HFQ4 decode probe for gate/up and expanded down; default off after flat A3B measurements.
            match std::env::var("HIPFIRE_MOE_INDEXED_2ROW_GFX1151").as_deref() {
                Ok("0") => Some(false),
                Ok("1") => Some(true),
                _ => None,
            }
        }) {
            Some(force) => force,
            None => false,
        }
    }

    fn q8_4w_mode(env: &str) -> Option<bool> {
        match std::env::var(env).ok().as_deref() {
            Some("0" | "off" | "false" | "no") => Some(false),
            Some("1" | "on" | "true" | "yes") => Some(true),
            _ => None,
        }
    }

    fn gfx1151_q8_4w_enabled(mode: Option<bool>, auto: bool) -> bool {
        mode.unwrap_or(auto)
    }

    // ========================================================================
    // HFQ6-G256 GEMM variants (residual, fused)
    // ========================================================================

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
        // bind_thread: skip — launches no kernel itself; the collector's capture()
        // calls calib_*_reduce_f32, which bind_thread on their own.
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
            kernels::CALIB_REDUCE_SRC,
            "calib_sumsq_reduce_f32",
        )?;
        let x_ptr = x.buf.as_ptr();
        let acc_ptr = acc.buf.as_ptr();
        let n_i = n as i32;
        let k_i = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &x_ptr as *const _ as *mut c_void,
            &acc_ptr as *const _ as *mut c_void,
            &n_i as *const _ as *mut c_void,
            &k_i as *const _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((k as u32) + block - 1) / block;
        self.launch_maybe_blob(
            "calib_sumsq_reduce_f32",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut bb = hip_bridge::KernargBlob::new();
                bb.push_ptr(x_ptr);
                bb.push_ptr(acc_ptr);
                bb.push_i32(n_i);
                bb.push_i32(k_i);
                bb
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
            kernels::CALIB_REDUCE_SRC,
            "calib_hessian_outer_f32",
        )?;
        let x_ptr = x.buf.as_ptr();
        let h_ptr = h.buf.as_ptr();
        let n_i = n as i32;
        let k_i = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &x_ptr as *const _ as *mut c_void,
            &h_ptr as *const _ as *mut c_void,
            &n_i as *const _ as *mut c_void,
            &k_i as *const _ as *mut c_void,
        ];
        let tile = 16u32;
        let grid_x = ((k as u32) + tile - 1) / tile;
        let grid_y = ((k as u32) + tile - 1) / tile;
        self.launch_maybe_blob(
            "calib_hessian_outer_f32",
            [grid_x, grid_y, 1],
            [tile, tile, 1],
            0,
            &mut params,
            || {
                let mut bb = hip_bridge::KernargBlob::new();
                bb.push_ptr(x_ptr);
                bb.push_ptr(h_ptr);
                bb.push_i32(n_i);
                bb.push_i32(k_i);
                bb
            },
        )
    }

    /// Compile a givens4 kernel — prepends turbo_common + givens_common headers.
    fn ensure_givens4_kernel(
        &mut self,
        name: &str,
        body_src: &str,
        func_name: &str,
    ) -> HipResult<()> {
        if self.functions.contains_key(func_name) {
            return Ok(());
        }
        let stripped = body_src
            .replace("#include \"turbo_common.h\"", "")
            .replace("#include \"givens_common.h\"", "");
        let full_src = format!(
            "{}\n{}\n{}",
            kernels::TURBO_COMMON_H,
            kernels::GIVENS_COMMON_SRC,
            stripped
        );
        let obj_path = self.compiler.compile(name, &full_src)?;
        let obj_path_str = obj_path.to_str().unwrap().to_string();
        if !self.modules.contains_key(name) {
            let module = self.hip.module_load(&obj_path_str)?;
            self.modules.insert(name.to_string(), module);
        }
        let module = &self.modules[name];
        let func = self.hip.module_get_function(module, func_name)?;
        self.functions.insert(func_name.to_string(), func);
        Ok(())
    }

    /// Shared helper: launch a batched K-only rotated write kernel.
    fn launch_asym_k_batched(
        &mut self,
        kernel_key: &str,
        src_const: &'static str,
        func_name: &'static str,
        k_dst: &GpuTensor,
        k_src: &GpuTensor,
        positions: &GpuTensor,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.ensure_givens4_kernel(kernel_key, src_const, func_name)?;
        let mut kdp = k_dst.buf.as_ptr();
        let mut ksp = k_src.buf.as_ptr();
        let mut pp = positions.buf.as_ptr();
        let mut ctp = cos_theta.buf.as_ptr();
        let mut stp = sin_theta.buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut bs = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut kdp as *mut _ as *mut c_void,
            &mut ksp as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut ctp as *mut _ as *mut c_void,
            &mut stp as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        let shared_mem = ((head_dim + 32) * 4) as u32;
        self.launch_maybe_blob(
            func_name,
            [n_kv_heads as u32, batch_size as u32, 1],
            [32, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(kdp);
                b.push_ptr(ksp);
                b.push_ptr(pp);
                b.push_ptr(ctp);
                b.push_ptr(stp);
                b.push_i32(nkv);
                b.push_i32(hd);
                b.push_i32(bs);
                b
            },
        )
    }

    /// Shared helper: launch a batched asym flash tile + the shared asym reduce.
    ///
    /// `tree_bias` / `block_start` / `block_cols` activate DDTree tree-attention
    /// mode (bias added to in-block qk scores; seq_len extends to full cache
    /// including the tree block). When `tree_bias` is None and `block_cols` is
    /// 0, behavior is byte-identical to the legacy causal path.
    #[allow(clippy::too_many_arguments)]
    fn launch_asym_flash_batched(
        &mut self,
        tile_key: &'static str,
        tile_src: &'static str,
        tile_func_name: &'static str,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        partials: &GpuTensor,
        tree_bias: Option<&GpuTensor>,
        block_start: usize,
        block_cols: usize,
    ) -> HipResult<()> {
        const TILE_SIZE: usize = 128;
        let max_tiles = (max_ctx_len + TILE_SIZE - 1) / TILE_SIZE;
        let stride = 2 + head_dim;
        let per_pos_bytes = n_heads * max_tiles * stride * 4;
        let partials_capacity = partials.numel() * 4;
        let sub_batch = if per_pos_bytes > 0 {
            (partials_capacity / per_pos_bytes).max(1).min(batch_size)
        } else {
            batch_size
        };

        self.ensure_givens4_kernel(tile_key, tile_src, tile_func_name)?;
        self.ensure_kernel(
            "attention_flash_asym_reduce_batched",
            kernels::ATTENTION_FLASH_ASYM_REDUCE_BATCHED_SRC,
            "attention_flash_asym_reduce_batched",
        )?;

        let q_dim = n_heads * head_dim;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut offset = 0usize;
        while offset < batch_size {
            let chunk = (batch_size - offset).min(sub_batch);
            {
                let q_ptr =
                    unsafe { (q.buf.as_ptr() as *mut u8).add(offset * q_dim * 4) as *mut c_void };
                let k_ptr = k_cache.buf.as_ptr();
                let v_ptr = v_cache.buf.as_ptr();
                let p_ptr = partials.buf.as_ptr();
                let pos_ptr = positions.buf.as_ptr();
                let ct_ptr = cos_theta.buf.as_ptr();
                let st_ptr = sin_theta.buf.as_ptr();
                let bias_ptr: *mut std::ffi::c_void = match tree_bias {
                    Some(t) => t.buf.as_ptr(),
                    None => std::ptr::null_mut(),
                };
                let nh = n_heads as i32;
                let nkv = n_kv_heads as i32;
                let hd = head_dim as i32;
                let ms = max_seq as i32;
                let sc = scale;
                let ts = TILE_SIZE as i32;
                let mt = max_tiles as i32;
                let bo = offset as i32;
                let bs = block_start as i32;
                let bc = block_cols as i32;
                let mut params: Vec<*mut c_void> = vec![
                    &q_ptr as *const _ as *mut c_void,
                    &k_ptr as *const _ as *mut c_void,
                    &v_ptr as *const _ as *mut c_void,
                    &p_ptr as *const _ as *mut c_void,
                    &pos_ptr as *const _ as *mut c_void,
                    &ct_ptr as *const _ as *mut c_void,
                    &st_ptr as *const _ as *mut c_void,
                    &bias_ptr as *const _ as *mut c_void,
                    &nh as *const _ as *mut c_void,
                    &nkv as *const _ as *mut c_void,
                    &hd as *const _ as *mut c_void,
                    &ms as *const _ as *mut c_void,
                    &sc as *const _ as *mut c_void,
                    &ts as *const _ as *mut c_void,
                    &mt as *const _ as *mut c_void,
                    &bo as *const _ as *mut c_void,
                    &bs as *const _ as *mut c_void,
                    &bc as *const _ as *mut c_void,
                ];
                self.launch_maybe_blob(
                    tile_func_name,
                    [n_heads as u32, max_tiles as u32, chunk as u32],
                    [32, 1, 1],
                    (TILE_SIZE * 4) as u32,
                    &mut params,
                    || {
                        let mut b = hip_bridge::KernargBlob::new();
                        b.push_ptr(q_ptr);
                        b.push_ptr(k_ptr);
                        b.push_ptr(v_ptr);
                        b.push_ptr(p_ptr);
                        b.push_ptr(pos_ptr);
                        b.push_ptr(ct_ptr);
                        b.push_ptr(st_ptr);
                        b.push_ptr(bias_ptr);
                        b.push_i32(nh);
                        b.push_i32(nkv);
                        b.push_i32(hd);
                        b.push_i32(ms);
                        b.push_f32(sc);
                        b.push_i32(ts);
                        b.push_i32(mt);
                        b.push_i32(bo);
                        b.push_i32(bs);
                        b.push_i32(bc);
                        b
                    },
                )?;
            }
            {
                let p_ptr = partials.buf.as_ptr();
                let o_ptr =
                    unsafe { (out.buf.as_ptr() as *mut u8).add(offset * q_dim * 4) as *mut c_void };
                let pos_ptr = positions.buf.as_ptr();
                let nh = n_heads as i32;
                let hd = head_dim as i32;
                let ts = TILE_SIZE as i32;
                let mt = max_tiles as i32;
                let bo = offset as i32;
                let bs = block_start as i32;
                let bc = block_cols as i32;
                let mut params: Vec<*mut c_void> = vec![
                    &p_ptr as *const _ as *mut c_void,
                    &o_ptr as *const _ as *mut c_void,
                    &pos_ptr as *const _ as *mut c_void,
                    &nh as *const _ as *mut c_void,
                    &hd as *const _ as *mut c_void,
                    &ts as *const _ as *mut c_void,
                    &mt as *const _ as *mut c_void,
                    &bo as *const _ as *mut c_void,
                    &bs as *const _ as *mut c_void,
                    &bc as *const _ as *mut c_void,
                ];
                self.launch_maybe_blob(
                    "attention_flash_asym_reduce_batched",
                    [n_heads as u32, chunk as u32, 1],
                    [32, 1, 1],
                    0,
                    &mut params,
                    || {
                        let mut b = hip_bridge::KernargBlob::new();
                        b.push_ptr(p_ptr);
                        b.push_ptr(o_ptr);
                        b.push_ptr(pos_ptr);
                        b.push_i32(nh);
                        b.push_i32(hd);
                        b.push_i32(ts);
                        b.push_i32(mt);
                        b.push_i32(bo);
                        b.push_i32(bs);
                        b.push_i32(bc);
                        b
                    },
                )?;
            }
            offset += chunk;
        }
        Ok(())
    }

    // ── DeltaNet ops (feature-gated) ─────────────────────────────────────

    /// In-place F32 → bf16 → F32 round-trip on `x`. Used by the
    /// dots.ocr vision encoder for HF-bf16-precision emulation
    /// (see `kernels/src/bf16_round_trip.hip`).
    pub fn bf16_round_trip_f32(&mut self, x: &GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "bf16_round_trip",
            kernels::BF16_ROUND_TRIP_SRC,
            "bf16_round_trip_f32",
        )?;
        let xp = x.buf.as_ptr();
        let n = x.numel() as i32;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &n as *const _ as *mut c_void,
        ];
        let block_size = 256u32;
        let grid = (((n as u32) + block_size - 1) / block_size).max(1);
        let bytes = crate::profile::elementwise_bytes(n as usize);
        let timer =
            crate::profile::begin_timer(&self.hip, "bf16_round_trip", "bf16_round_trip_f32", bytes);
        let result = self.launch_maybe_blob(
            "bf16_round_trip_f32",
            [grid, 1, 1],
            [block_size, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp);
                b.push_i32(n);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Exact per-row/per-2048-token chunk top-256 + logsumexp stats for
    /// KLD reference generation. The caller merges the emitted chunk
    /// candidates across chunks/tiles on the host.
    pub fn kld_tile_topk_lse_f32(
        &mut self,
        logits: &GpuTensor,
        top_vals: &GpuTensor,
        top_idx: &GpuTensor,
        chunk_max: &GpuTensor,
        chunk_sum: &GpuTensor,
        batch_size: usize,
        vocab_tile: usize,
        global_start: usize,
        n_chunks: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "kld_tile_topk_lse",
            kernels::KLD_TILE_TOPK_LSE_SRC,
            "kld_tile_topk_lse_f32",
        )?;
        let func = &self.functions["kld_tile_topk_lse_f32"];
        let lp = logits.buf.as_ptr();
        let vp = top_vals.buf.as_ptr();
        let ip = top_idx.buf.as_ptr();
        let mp = chunk_max.buf.as_ptr();
        let sp = chunk_sum.buf.as_ptr();
        let mut bi = batch_size as i32;
        let mut vi = vocab_tile as i32;
        let mut gi = global_start as i32;
        let mut ci = n_chunks as i32;
        let mut params: Vec<*mut c_void> = vec![
            &lp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &mp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut vi as *mut _ as *mut c_void,
            &mut gi as *mut _ as *mut c_void,
            &mut ci as *mut _ as *mut c_void,
        ];
        let bytes = batch_size * vocab_tile * 4
            + batch_size * n_chunks * 256 * 8
            + batch_size * n_chunks * 2 * 4;
        let timer = crate::profile::begin_timer(&self.hip, "kld", "kld_tile_topk_lse_f32", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [batch_size as u32, n_chunks as u32, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Validate the fixed tile shape used by the current Gated DeltaNet HIP kernels.
    #[cfg(feature = "deltanet")]
    fn ensure_gdn_hd128(head_dim: usize) -> HipResult<()> {
        if head_dim == 128 {
            Ok(())
        } else {
            Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "Gated DeltaNet kernels are currently specialized for head_dim=128; got {head_dim}. \
                     Add a matching HD-specialized kernel or generalize the kernel before dispatching this shape."
                ),
            ))
        }
    }

    #[cfg(feature = "deltanet")]
    fn gdn_q8_reg_gfx1151_enabled(&self) -> bool {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        let enabled = *ENABLED.get_or_init(|| {
            // HIPFIRE_GDN_Q8_REG_GFX1151=1 enables the gfx1151 register-state
            // GDN Q8 probe. Default off after A3B pp256 prefill regressed
            // 11.8ms -> 168.5ms total GDN time.
            std::env::var("HIPFIRE_GDN_Q8_REG_GFX1151").as_deref() == Ok("1")
        });
        enabled && self.arch.starts_with("gfx1151")
    }

    #[cfg(feature = "deltanet")]
    #[allow(clippy::too_many_arguments)]
    fn conv1d_silu_split_f32_n_gfx1151(
        &mut self,
        q_out: &GpuTensor,
        k_out: &GpuTensor,
        v_out: &GpuTensor,
        input: &GpuTensor,
        weight: &GpuTensor,
        state: &GpuTensor,
        k_dim: usize,
        v_dim: usize,
        n_tokens: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "conv1d_silu_split_gfx1151",
            kernels::CONV1D_SILU_SPLIT_GFX1151_SRC,
            "conv1d_silu_split_f32_gfx1151",
        )?;
        self.ensure_kernel(
            "conv1d_silu_split_state_gfx1151",
            kernels::CONV1D_SILU_SPLIT_GFX1151_SRC,
            "conv1d_silu_split_update_state_gfx1151",
        )?;

        let qp = q_out.buf.as_ptr();
        let kp = k_out.buf.as_ptr();
        let vp = v_out.buf.as_ptr();
        let ip = input.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let sp = state.buf.as_ptr();
        let kd = k_dim as i32;
        let vd = v_dim as i32;
        let nt = n_tokens as i32;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &kd as *const _ as *mut c_void,
            &vd as *const _ as *mut c_void,
            &nt as *const _ as *mut c_void,
        ];
        let n_channels = 2 * k_dim + v_dim;
        let block = 256u32;
        let grid = ((n_channels as u32) + block - 1) / block;
        let bytes = crate::profile::conv1d_silu_bytes(n_channels) * n_tokens;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "deltanet",
            "conv1d_silu_split_f32_n_gfx1151",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "conv1d_silu_split_f32_gfx1151",
            [grid, n_tokens as u32, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_ptr(vp);
                b.push_ptr(ip);
                b.push_ptr(wp);
                b.push_ptr(sp);
                b.push_i32(kd);
                b.push_i32(vd);
                b.push_i32(nt);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result?;

        let nc = n_channels as i32;
        let mut update_params: Vec<*mut c_void> = vec![
            &ip as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &nc as *const _ as *mut c_void,
            &nt as *const _ as *mut c_void,
        ];
        let update_bytes = n_channels * 4 * 4;
        let update_timer = crate::profile::begin_timer(
            &self.hip,
            "deltanet",
            "conv1d_silu_split_state_gfx1151",
            update_bytes,
        );
        let update_result = self.launch_maybe_blob(
            "conv1d_silu_split_update_state_gfx1151",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut update_params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ip);
                b.push_ptr(sp);
                b.push_i32(nc);
                b.push_i32(nt);
                b
            },
        );
        if let Some(t) = update_timer {
            t.finish(&self.hip);
        }
        update_result
    }

    // ═══ Vision encoder dispatch (GEMM, LayerNorm, GELU, bias-add) ═══

    /// Block until all prior GPU work on this device completes.
    pub fn device_synchronize(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        self.hip.device_synchronize()
    }

    /// Read-and-clear the HIP last-error flag (hipGetLastError). Returns the code.
    pub fn clear_last_error(&mut self) -> u32 {
        // bind_thread: skip — pure error-flag query, no kernel dispatch / device work.
        self.hip.last_error()
    }

    /// Apply a causal mask to attention scores `[seq_q*seq_k]` (j>i → −1e30).
    pub fn causal_mask_train(
        &mut self,
        scores: &GpuTensor,
        seq_q: usize,
        seq_k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "causal_mask_train",
            kernels::CAUSAL_MASK_TRAIN_SRC,
            "causal_mask_train",
        )?;
        let func = &self.functions["causal_mask_train"];
        let mut sp = scores.buf.as_ptr();
        let mut sq = seq_q as i32;
        let mut sk = seq_k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut sp as *mut _ as *mut c_void,
            &mut sq as *mut _ as *mut c_void,
            &mut sk as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [seq_q as u32, 1, 1],
                [64, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Strided 2D copy (fp32): `dst[dst_off+r*dst_stride+c] (+=|=) src[...]`.
    /// `accumulate` selects scatter-add. Element units throughout.
    #[allow(clippy::too_many_arguments)]
    pub fn strided_copy_2d(
        &mut self,
        src: &GpuTensor,
        src_off: usize,
        src_stride: usize,
        dst: &GpuTensor,
        dst_off: usize,
        dst_stride: usize,
        rows: usize,
        cols: usize,
        accumulate: bool,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "strided_copy_2d",
            kernels::STRIDED_COPY_2D_SRC,
            "strided_copy_2d",
        )?;
        let func = &self.functions["strided_copy_2d"];
        let mut sp = src.buf.as_ptr();
        let mut so = src_off as i32;
        let mut ss = src_stride as i32;
        let mut dp = dst.buf.as_ptr();
        let mut do_ = dst_off as i32;
        let mut ds = dst_stride as i32;
        let mut rr = rows as i32;
        let mut cc = cols as i32;
        let mut acc = accumulate as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut sp as *mut _ as *mut c_void,
            &mut so as *mut _ as *mut c_void,
            &mut ss as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut do_ as *mut _ as *mut c_void,
            &mut ds as *mut _ as *mut c_void,
            &mut rr as *mut _ as *mut c_void,
            &mut cc as *mut _ as *mut c_void,
            &mut acc as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [rows as u32, 1, 1],
                [64, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// AdamW step (fp32). `p`,`g`,`m`,`v`: `[n]` (m,v persisted across steps).
    /// `bc1`/`bc2` are host-computed bias corrections (1−β^t).
    #[allow(clippy::too_many_arguments)]
    pub fn adamw_step(
        &mut self,
        p: &GpuTensor,
        g: &GpuTensor,
        m: &GpuTensor,
        v: &GpuTensor,
        n: usize,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
        bc1: f32,
        bc2: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("adamw_step", kernels::ADAMW_TRAIN_SRC, "adamw_step")?;
        let func = &self.functions["adamw_step"];
        let mut pp = p.buf.as_ptr();
        let mut gp = g.buf.as_ptr();
        let mut mp = m.buf.as_ptr();
        let mut vp = v.buf.as_ptr();
        let mut ni = n as i32;
        let mut lrf = lr;
        let mut b1 = beta1;
        let mut b2 = beta2;
        let mut epsf = eps;
        let mut wdf = wd;
        let mut bc1f = bc1;
        let mut bc2f = bc2;
        let mut params: Vec<*mut c_void> = vec![
            &mut pp as *mut _ as *mut c_void,
            &mut gp as *mut _ as *mut c_void,
            &mut mp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut lrf as *mut _ as *mut c_void,
            &mut b1 as *mut _ as *mut c_void,
            &mut b2 as *mut _ as *mut c_void,
            &mut epsf as *mut _ as *mut c_void,
            &mut wdf as *mut _ as *mut c_void,
            &mut bc1f as *mut _ as *mut c_void,
            &mut bc2f as *mut _ as *mut c_void,
        ];
        let grid = ((n as u32) + 255) / 256;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// KL distillation loss fwd+bwd (fp32). `student`,`d_logits`: `[rows*v]`;
    /// `teacher_p`: `[rows*v]` (probabilities); `loss`: `[rows]`. `d_logits` is
    /// the sum-reduction gradient (q − p_t).
    pub fn distill_kl_train(
        &mut self,
        student: &GpuTensor,
        teacher_p: &GpuTensor,
        loss: &GpuTensor,
        d_logits: &GpuTensor,
        rows: usize,
        v: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "distill_kl_train",
            kernels::DISTILL_TRAIN_SRC,
            "distill_kl_train",
        )?;
        let func = &self.functions["distill_kl_train"];
        let mut sp = student.buf.as_ptr();
        let mut tp = teacher_p.buf.as_ptr();
        let mut lp = loss.buf.as_ptr();
        let mut dp = d_logits.buf.as_ptr();
        let mut rowsi = rows as i32;
        let mut vi = v as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut sp as *mut _ as *mut c_void,
            &mut tp as *mut _ as *mut c_void,
            &mut lp as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut rowsi as *mut _ as *mut c_void,
            &mut vi as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [rows as u32, 1, 1],
                [64, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Bias-add: x[batch, n] += bias[n] (in-place, broadcast over batch dim)
    pub fn bias_add_f32(
        &mut self,
        x: &GpuTensor,
        bias: &GpuTensor,
        batch: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("bias_add_f32", kernels::BIAS_ADD_SRC, "bias_add_f32")?;
        let xp = x.buf.as_ptr();
        let bp = bias.buf.as_ptr();
        let ni = n as i32;
        let total = (batch * n) as i32;
        let ti = total;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &ni as *const _ as *mut c_void,
            &ti as *const _ as *mut c_void,
        ];
        let blocks = ((total as usize + 255) / 256) as u32;
        self.launch_maybe_blob(
            "bias_add_f32",
            [blocks, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(bp);
                b.push_i32(ni);
                b.push_i32(ti);
                b
            },
        )
    }

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
        _head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // asym kernels #include "turbo_common.h" + "givens_common.h"; the
        // runtime dispatch path (see ensure_givens4_kernel) prepends the
        // header bodies and strips the #includes. We mirror that exactly so
        // the hash matches and the runtime re-uses our cached .hsaco.
        let assemble_asym = |body: &str| -> String {
            let stripped = body
                .replace("#include \"turbo_common.h\"", "")
                .replace("#include \"givens_common.h\"", "");
            format!(
                "{}\n{}\n{}",
                kernels::TURBO_COMMON_H,
                kernels::GIVENS_COMMON_SRC,
                stripped
            )
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
                // raw HFQ6 GEMV (used by a few residual paths).
                specs.push(("gemv_mq6g256", kernels::GEMV_MQ6G256_SRC.to_string()));
                specs.push(("gemv_hfq6g256", kernels::GEMV_HFQ6G256_SRC.to_string()));
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
                specs.push((
                    "fused_qkvza_hfq4g256",
                    kernels::FUSED_QKVZA_HFQ4G256_SRC.to_string(),
                ));
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
                if self.arch_caps.is_wave64_native() {
                    // Single-token (draft / single-layer paths).
                    specs.push((
                        "fused_qkvza_hfq4g256_wave64",
                        kernels::FUSED_QKVZA_HFQ4G256_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "fused_qkv_hfq4g256_wave64",
                        kernels::FUSED_QKV_HFQ4G256_WAVE64_SRC.to_string(),
                    ));
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
                // RDNA3 multi-row GEMV is opt-in via HIPFIRE_GEMV_ROWS={2,4,8}.
                // Precompile whenever the effective row selector asks for it.
                if self.arch_caps.is_rdna3() && self.arch_caps.gemv_rows_default() > 1 {
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
                specs.push((
                    "fused_qkvza_hfq4g256",
                    kernels::FUSED_QKVZA_HFQ4G256_SRC.to_string(),
                ));
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
                specs.push((
                    "fused_silu_mul_mq_rotate",
                    kernels::FUSED_SILU_MUL_MQ_ROTATE_SRC.to_string(),
                ));
                // gfx906/gfx908/gfx94x wave64 variants — see hfq4 branch for rationale.
                if self.arch_caps.is_wave64_native() {
                    // Single-token (draft / single-layer paths).
                    specs.push((
                        "fused_qkvza_hfq4g256_wave64",
                        kernels::FUSED_QKVZA_HFQ4G256_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "fused_qkv_hfq4g256_wave64",
                        kernels::FUSED_QKV_HFQ4G256_WAVE64_SRC.to_string(),
                    ));
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
                if self.arch_caps.is_rdna3() && self.arch_caps.gemv_rows_default() > 1 {
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
                specs.push((
                    "attention_q8_0_kv_batched",
                    kernels::ATTENTION_Q8_0_KV_BATCHED_SRC.to_string(),
                ));
                specs.push((
                    "kv_cache_write_q8_0_batched",
                    kernels::KV_CACHE_WRITE_Q8_0_BATCHED_SRC.to_string(),
                ));
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
                other => vec![other],
            };
            // Compile and ensure the module is loaded once.
            let obj_path = self.compiler.compile(name, src)?;
            let obj_path_str = obj_path.to_str().unwrap().to_string();
            if !self.modules.contains_key(*name) {
                let module = self.hip.module_load(&obj_path_str)?;
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
    // PFlash scoring
    // ═══════════════════════════════════════════════════════════════════════════

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

    // ── Stubs for dispatch-unification features not yet implemented ───────
}

impl Drop for Gpu {
    /// Defensive: bind owning device before any future per-field `Drop`
    /// impls call `hipFree` etc. Uses `bind_thread_or_warn` to avoid
    /// panic-in-Drop from `bind_thread`'s `debug_assert!`.
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }
        self.bind_thread_or_warn();
    }
}

// ─── DeepSeek V4 Flash (arch_id=7) — dispatch ─────────────────────────────────
// DeepSeek V4-required Gpu methods lifted from the DeepSeek V4 development branch. Each method
// dispatches one of the DeepSeek V4 kernels registered in kernels.rs. Doc comments and
// implementation preserved verbatim. See `crates/hipfire-arch-deepseek4/` for
// the caller surface.
impl Gpu {
    /// Generic kernel library: WMMA GEMM, F16 inputs → F16 output.
    /// `a_f16` [M,K], `x_f16` [B,K], `y_f16` [B,M], all raw F16 (u16) payloads.
    /// gfx1103/RDNA3 wave32, zero LDS. F32 accumulation, F16 round on store.
    /// Requires `k % 16 == 0` and wave32 WMMA.
    /// Generic kernel library GEMV launcher: one wave32 per output row, params
    /// `(W, x, y, M, K)`, grid `[M]`, block `[32]`. Shared by all six
    /// `gemv_*` generic-library kernels (gfx1103, zero LDS).
    fn launch_gemv_generic(
        &mut self,
        name: &'static str,
        src: &'static str,
        w: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(name, src, name)?;
        let wp = w.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yp = y.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
        ];
        let func = &self.functions[name];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// HIP-graphs-safe overlap-shift gated by commit slot: when
    /// `commit_slot_buf[0] >= 0`, copies `state[ratio*proj_dim..2*ratio*proj_dim]`
    /// down to `state[0..ratio*proj_dim]`. Otherwise no-op.
    #[allow(dead_code)]
    pub fn state_overlap_shift_f32_buf(
        &mut self,
        state: &GpuTensor,
        commit_slot_buf: &GpuTensor,
        ratio: i32,
        proj_dim: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "state_overlap_shift_f32_buf",
            kernels::STATE_OVERLAP_SHIFT_F32_BUF_SRC,
            "state_overlap_shift_f32_buf",
        )?;
        let stp = state.buf.as_ptr();
        let cp = commit_slot_buf.buf.as_ptr();
        let mut rv = ratio;
        let mut pd = proj_dim;
        let mut params: Vec<*mut c_void> = vec![
            &stp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &mut rv as *mut _ as *mut c_void,
            &mut pd as *mut _ as *mut c_void,
        ];
        let total = (ratio * proj_dim) as u32;
        let block = 256u32;
        let grid = (total + block - 1) / block;
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(stp);
            b.push_ptr(cp);
            b.push_i32(rv);
            b.push_i32(pd);
            b
        };
        self.launch_maybe_blob(
            "state_overlap_shift_f32_buf",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut params,
            blob_builder,
        )
    }

    /// HIP-graphs-safe ring write: `state[ring_slot_buf[0]*proj_dim..]
    /// = src[0..proj_dim]`. -1 sentinel → no-op.
    #[allow(dead_code)]
    pub fn state_ring_write_f32_buf(
        &mut self,
        src: &GpuTensor,
        state: &GpuTensor,
        ring_slot_buf: &GpuTensor,
        proj_dim: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "state_ring_write_f32_buf",
            kernels::STATE_RING_WRITE_F32_BUF_SRC,
            "state_ring_write_f32_buf",
        )?;
        let sp = src.buf.as_ptr();
        let stp = state.buf.as_ptr();
        let rp = ring_slot_buf.buf.as_ptr();
        let mut pd = proj_dim;
        let mut params: Vec<*mut c_void> = vec![
            &sp as *const _ as *mut c_void,
            &stp as *const _ as *mut c_void,
            &rp as *const _ as *mut c_void,
            &mut pd as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((proj_dim as u32) + block - 1) / block;
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(sp);
            b.push_ptr(stp);
            b.push_ptr(rp);
            b.push_i32(pd);
            b
        };
        self.launch_maybe_blob(
            "state_ring_write_f32_buf",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut params,
            blob_builder,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::gen_fwht_signs;

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
}
