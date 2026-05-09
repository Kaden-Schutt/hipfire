//! High-level GPU dispatch interface.
//! Manages compiled kernels, provides typed tensor operations.

use crate::compiler::KernelCompiler;
use crate::kernels;
use hip_bridge::{DeviceBuffer, HipResult, HipRuntime, Rocblas};
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::OnceLock;

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

thread_local! {
    /// Per-thread cache for `Gpu::bind_thread`. Sentinel `-1` forces the
    /// first call to issue `hipSetDevice` even when the target id is 0.
    static LAST_BOUND_DEVICE: Cell<i32> = const { Cell::new(-1) };
}

/// gfx1100 multi-row GEMV tile selector.
/// HIPFIRE_GEMV_ROWS ∈ {1, 2, 4, 8}. Default 1 = single-row kernel (legacy).
/// Cached in a OnceLock — the env var is read exactly once per process.
/// Returns the runtime HIPFIRE_GEMV_ROWS override if set, otherwise None.
/// Valid values: 1, 2, 4, 8. Anything else is clamped to 1.
fn gemv_rows_override() -> Option<u32> {
    static CACHE: OnceLock<Option<u32>> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("HIPFIRE_GEMV_ROWS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .map(|r| match r { 1 | 2 | 4 | 8 => r, _ => 1 })
    })
}

/// gfx906 dp4a-port toggle for memory-bound fused GEMVs.
///
/// `fused_gate_up_hfq4g256_dp4a` pre-quantizes x to Q8_1 and uses
/// v_dot4_i32_i8 for the inner-loop multiply. Per the per-kernel PMC
/// pass at 2026-05-05, this kernel was memory-bound (3.86 % MemUnit
/// stall, 41 % VALUBusy) — dp4a's 75 % x-traffic reduction lands on
/// the actual bottleneck.
///
/// Measured on MI50 / qwen3.5-9b.mq4 AR decode: +7.1 % tok/s
/// (54.6 → 58.5 median, 3-run; BW 270 → 290 GiB/s) on top of the
/// prefetch win on gemv_residual. Coherence gate clean.
///
/// Default-on for gfx906 only. Override with HIPFIRE_GEMV_DP4A={0,1}.
/// fused_qkv / fused_qkvza ports for HFQ4 (PR #167) and HFQ6 (PR #187)
/// have shipped; this lever now toggles every fused dp4a path together.
pub fn gemv_dp4a_enabled(arch: &str) -> bool {
    static CACHE: OnceLock<Option<bool>> = OnceLock::new();
    let override_ = *CACHE.get_or_init(|| {
        std::env::var("HIPFIRE_GEMV_DP4A").ok().and_then(|v| match v.as_str() {
            "1" | "true" | "TRUE" | "on" | "ON" => Some(true),
            "0" | "false" | "FALSE" | "off" | "OFF" => Some(false),
            _ => None,
        })
    });
    override_.unwrap_or(arch == "gfx906")
}

/// Weight-prefetch variant of the wave64 residual-GEMV.
///
/// The prefetch kernel does software-pipelined across-quad weight loads —
/// quad q+1's 12 dwords are issued before quad q's compute chain runs, so
/// L2 fills overlap with the FMA chain instead of stalling the load unit.
///
/// Measured on gfx906 (MI50) AR decode of qwen3.5-9b.mq4: +4.8% tok/s
/// (51.9 → 54.4 median, 3-run; BW 256.7 → 269.1 GiB/s). PMC L2CacheHit
/// pass showed ~40 % L2 hit on the non-prefetched kernel — this lever
/// shifts a fraction of those misses into the L2-hit regime while
/// compute is in flight. Coherence gate clean (b37068c).
///
/// **Default-on for gfx906 only.** Other wave64-native archs
/// (gfx908/MI100, gfx940-942/MI300x) take the original kernel until
/// measured. Override with HIPFIRE_GEMV_PREFETCH={0,1}.
///
/// See docs/perf-checkpoints/2026-05-05-gfx906-decode-investigation.md.
fn gemv_prefetch_enabled(arch: &str) -> bool {
    static CACHE: OnceLock<Option<bool>> = OnceLock::new();
    let override_ = *CACHE.get_or_init(|| {
        std::env::var("HIPFIRE_GEMV_PREFETCH").ok().and_then(|v| match v.as_str() {
            "1" | "true" | "TRUE" | "on" | "ON" => Some(true),
            "0" | "false" | "FALSE" | "off" | "OFF" => Some(false),
            _ => None,
        })
    });
    override_.unwrap_or(arch == "gfx906")
}

/// Per-arch default R for the multi-row HFQ4 GEMV kernel family.
///
/// - RDNA3 (gfx1100/1101/1102): R=1. Measured negative on 7900 XTX —
///   single-row is already near-BW-saturated (577 GiB/s on 9B ≈ 60% of
///   the 960 GiB/s peak) and multi-row under-subscribes the wave scheduler.
/// - RDNA2 (gfx1030/1031): R=1. These have their own arch-optimized narrow
///   kernels via gemv_hfq4g256_for_arch; the multi-row path is bypassed.
/// - Default (gfx1010 baseline, gfx1013 Cyan Skillfish / BC-250, others):
///   R=2. Measured +2.75% on BC-250 Qwen3.5 0.8B MQ4 in the session 1
///   perf work — the x-hoist amortization across 2 rows pays for the
///   minor occupancy drop from 20 → 18 waves/SIMD.
fn gemv_rows_default(arch: &str) -> u32 {
    match arch {
        "gfx1100" | "gfx1101" | "gfx1102" => 1,
        "gfx1030" | "gfx1031" => 1,
        // Vega 20 / GCN5 (gfx906), CDNA1 (MI100, gfx908), and CDNA3
        // (MI300X): wave64 native.
        // `gemv_hfq4g256_wide` uses block=[64,1,1] = exactly one wave —
        // zero lane waste. The 32-thread multirow variants run on half a
        // wave, so the wide kernel is the natural fit. Return rows=1 to
        // trigger use_wide.
        "gfx906" | "gfx908" | "gfx940" | "gfx941" | "gfx942" => 1,
        _ => 2,
    }
}

/// Whether this GPU architecture supports the `v_dot2_f32_f16` instruction
/// (dot10-insts feature in LLVM). This is required for the FP16 "dot2" GEMM fast path.
///
/// Notably:
/// - gfx1010 (Navi 10 / RX 5700 XT) lacks this instruction despite being RDNA1.
/// - gfx1011 (Navi 12) and gfx1012 (Navi 14) have it, also despite being RDNA1.
/// - gfx1013 (Van Gogh / BC-250 APU) lacks it despite being RDNA2-ish.
/// - gfx1030+ (standard RDNA2) and gfx1100+ (RDNA3/4) have it.
fn has_dot2_f32_f16(arch: &str) -> bool {
    matches!(arch,
        "gfx1011" | "gfx1012"
        | "gfx1030" | "gfx1031" | "gfx1032"
        | "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103"
        | "gfx1150" | "gfx1151" | "gfx1152"
        | "gfx1200" | "gfx1201")
}

/// Whether this arch has WMMA kernels that compile + run on the user's
/// ROCm toolchain right now.
///
/// gfx11 (RDNA3, Navi 31/32/33) ships with the
/// `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` builtin and has been
/// the WMMA workhorse since 0.1.4. gfx12 (RDNA4, Navi 48/RX 9070
/// series) has WMMA in hardware too, but the existing kernels use the
/// gfx11 builtin which AMD clang 22.x in ROCm 7.x does NOT pattern-
/// match on gfx12 — it errors with `Cannot select: intrinsic
/// %llvm.amdgcn.wmma.f32.16x16x16.f16` at codegen time. The gfx12 sister
/// path uses `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` and is
/// gated by `has_wmma_f16_gfx12` below.
fn has_wmma_f16(arch: &str) -> bool {
    arch.starts_with("gfx11")
}

/// gfx12 (RDNA4) WMMA fp16. Kernels live at
/// `kernels/src/gemm_*_wmma.gfx12.hip` and are validated by the
/// `test_wmma_*_gfx12` channel-test examples (issue #54, PR #56).
fn has_wmma_f16_gfx12(arch: &str) -> bool {
    arch.starts_with("gfx12")
}

/// Gates the wave64 FP16 hybrid prefill path. gfx906 (Vega 20, MI50) is the
/// only arch with measured data: +90% prefill on Qwen 3.5 9B (74 → 141 tk/s).
/// gfx908 (CDNA1, MI100) shares __hfma2 + wave64 and would code-correctly
/// run the same kernels, but we have no perf data and MFMA likely wants a
/// different optimum. MI100 owners can opt in for A/B testing via
/// `HIPFIRE_GCN5_WAVE64_HYBRID=1`. CDNA3 (gfx94x) uses rocBLAS MFMA.
fn is_gcn5_wave64(arch: &str) -> bool {
    if arch == "gfx906" {
        return true;
    }
    if arch == "gfx908"
        && std::env::var("HIPFIRE_GCN5_WAVE64_HYBRID")
            .map_or(false, |v| v == "1")
    {
        return true;
    }
    false
}

/// Wave64-native arches: Vega 20 / GCN5 (gfx906), CDNA1 (gfx908, MI100),
/// and CDNA3 (gfx94x, MI300X). On these, wave32 kernels (block=[32,1,1])
/// waste the upper 32 lanes of every wave slot. The `*_wave64.hip` kernel
/// variants pack two rows per block (one per warp) with block=[64,1,1] and
/// halve the grid count. Adding gfx90a (CDNA2, MI200) here is a one-line
/// change once it has been bring-up validated.
fn has_wave64_native(arch: &str) -> bool {
    matches!(arch, "gfx906" | "gfx908" | "gfx940" | "gfx941" | "gfx942")
}

/// Architectures that have an integer-MMQ prefill path:
/// - RDNA3/3.5 (gfx1100..gfx1152): i8 WMMA via `__builtin_amdgcn_wmma_i32_16x16x16_iu8`
/// - gfx906 (Vega 20, MI50/MI60): dp4a via `__builtin_amdgcn_sdot4`
///
/// The two dispatch through different Rust routines because the launch
/// shape and LDS budget differ — see `gemm_hfq4g256_residual_mmq` (RDNA3,
/// 32×8 block, 128×128 tile, WMMA) vs the gfx906 redesign
/// (`gemm_hfq4g256_residual_mmq_gfx906_x{N}` for N ∈ {8..64}, 64×4 block,
/// 128×mmq_x tile, dp4a, per-mmq_x X_STRIDE; see
/// `kernels/src/gemm_hfq4g256_residual_mmq_gfx906_body.cuh`).
fn has_mmq_dp4a_or_wmma(arch: &str) -> bool {
    matches!(arch,
        "gfx906"
        | "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103"
        | "gfx1150" | "gfx1151" | "gfx1152")
}

/// Decide whether an integer-MMQ prefill path should be used for a given
/// GEMM call. Combines the arch gate, the env override, and an empirical
/// batch-size threshold.
///
/// **RDNA3 i8-WMMA MMQ:** uses a 128×128 batch tile (vs the fp16 WMMA path's
/// 16×16), so it amortizes its high per-launch fixed cost only when
/// batch_size is large enough to fill multiple tiles. Empirical sweep on
/// Qwen 3.5 9B (gfx1100, ROCm 7.2, residual at m=4096) across pp ∈ {32..512}:
///   pp32-pp192: MMQ regresses 23-69% (per-launch overhead dominates).
///   pp224:      within noise (-8%).
///   pp256+:     MMQ wins at multiples of 128 (+12% to +29%).
/// Default RDNA3 threshold is 256.
///
/// **gfx906 dp4a MMQ:** uses runtime-dispatched mmq_x ∈ {8,16,24,32,40,48,56,64}
/// per the post-redesign kernel (plans/gfx906_mmq_redesign.md, commit
/// c022682). Default-on at batch_size ≥ 16 — pp128 hits 462 tok/s on
/// Qwen 9B mq4 (3.28× over FP16 wave64); below pp16 the Q8_1 quantize +
/// per-output launch overhead dominates so FP16 wave64 wins.
///
/// `HIPFIRE_MMQ` env override:
///   `0` / `off`            — force MMQ off (debug / regression bisect)
///   `1` / `on`             — force MMQ on at every batch (legacy behavior)
///   `auto` / unset / other — auto-route by batch_size threshold (default)
fn should_use_mmq(arch: &str, batch_size: usize) -> bool {
    if !has_mmq_dp4a_or_wmma(arch) {
        return false;
    }
    match std::env::var("HIPFIRE_MMQ").ok().as_deref() {
        Some("0") | Some("off") => false,
        Some("1") | Some("on") => true,
        _ => {
            // Per-arch default min_batch:
            //   gfx906: 8 — empirically validated for both prefill (pp512
            //     within noise of min_batch=16) and DFlash 27B verify
            //     (B ∈ [12, 14] previously fell to FP16 wave64; lifting
            //     them to MMQ gives +64.8% tok/s on humaneval-0 prompt,
            //     +39% on lru_cache prose, 3-run deterministic). Earlier
            //     min_batch=16 was set on the prefill `gemm_hfq4g256`
            //     non-residual sweep; the *residual* batched GEMM
            //     (used by DFlash verify) crosses below that and wins
            //     down to B=8. AR decode at B=1 stays unchanged
            //     (well below cutover). PMC pass at 2026-05-06:
            //     `gemm_hfq4g256_residual_fp16_wave64` was 23.5% of
            //     DFlash time at min_batch=16 — root cause of the gap.
            //   other archs: 256 — RDNA3+ has WMMA which is genuinely faster
            //     than MMQ at small batches; flip only when MMQ amortization
            //     dominates.
            let arch_min_batch: usize = if arch == "gfx906" { 8 } else { 256 };
            let min_batch = std::env::var("HIPFIRE_MMQ_MIN_BATCH")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(arch_min_batch);
            batch_size >= min_batch
        }
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
    Q4K,  // 144 bytes per 256 elements
    Q6K,  // 210 bytes per 256 elements
    Q8_0,      // 34 bytes per 32 elements
    Q4F16G64,  // 36 bytes per 64 elements (RDNA-native FP16 dequant)
    Q4F16G32,  // 20 bytes per 32 elements (RDNA-native FP16 dequant)
    Q8HFQ,     // split-metadata: scales contiguous then values contiguous, 128B-aligned rows
    HFQ4G256,  // 136 bytes per 256 elements (flat 4-bit, f32 scale+zero, 18 VGPRs)
    HFQ4G128,  // 72 bytes per 128 elements (flat 4-bit, f32 scale+zero, 14 VGPRs)
    HFQ3G256,  // 104 bytes per 256 elements (flat 3-bit, f32 scale+zero)
    HFQ3G128,  // 56 bytes per 128 elements (flat 3-bit, f32 scale+zero)
    MQ4G256,   // MagnumQuant: FWHT-rotated HFQ4-G256 (136 bytes/group, same as HFQ4G256)
    MQ8G256,   // MagnumQuant: FWHT-rotated symmetric INT8, dp4a target (258 bytes/group)
    MQ6G256,   // MagnumQuant: FWHT-rotated HFQ6-G256 (200 bytes/group, same as HFQ6G256)
    MQ3G256,   // MagnumQuant: FWHT-rotated HFQ3-G256 (104 bytes/group, same as HFQ3G256)
    MQ2G256,   // MagnumQuant: FWHT-rotated HFQ2-G256 (72 bytes/group, same as HFQ2G256)
    MQ2G256Lloyd, // MagnumQuant 2-bit + Lloyd-Max 4-entry fp16 codebook (72 bytes/group)
    MQ3G256Lloyd, // MagnumQuant 3-bit + Lloyd-Max 8-entry fp16 codebook (112 bytes/group)
    HFP4G32,   // HFP4: E2M1 element + UE8M0 g32 block scale + FP16 row scale.
               // Per-row header 16 B; per-block payload 17 B (UE8M0 + 16 packed nibbles).
               // See docs/quant-formats/hfp4.md.
    HFQ2G256,  // 72 bytes per 256 elements (flat 2-bit, f32 scale+zero, ~19 VGPRs)
    HFQ2G128,  // 40 bytes per 128 elements (flat 2-bit, f32 scale+zero)
    HFQ6G256,  // 200 bytes per 256 elements (6-bit, f32 scale+zero)
    Raw,       // raw bytes, no element interpretation
}

impl DType {
    pub fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::Q4K | DType::Q6K | DType::Q8_0 | DType::Q4F16G64 | DType::Q4F16G32 | DType::Q8HFQ | DType::HFQ4G256 | DType::HFQ4G128 | DType::HFQ3G256 | DType::HFQ3G128 | DType::HFQ2G256 | DType::HFQ2G128 | DType::HFQ6G256 | DType::MQ4G256 | DType::MQ6G256 | DType::MQ8G256 | DType::MQ3G256 | DType::MQ2G256 | DType::MQ2G256Lloyd | DType::MQ3G256Lloyd | DType::HFP4G32 | DType::Raw => 1, // byte-level
        }
    }
}

/// High-level GPU context. Owns the HIP runtime, compiler, and loaded kernels.
pub struct Gpu {
    pub hip: HipRuntime,
    pub arch: String,
    pub device_id: i32,
    compiler: KernelCompiler,
    modules: HashMap<String, hip_bridge::Module>,
    functions: HashMap<String, hip_bridge::Function>,
    pool: crate::pool::GpuPool,
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
    pub mq_x_rot: Option<GpuTensor>,  // scratch for rotated x, sized to max K
    pub mq_x_q8: Option<hip_bridge::DeviceBuffer>,   // INT8 quantized rotated x for dp4a
    pub mq_x_scales: Option<hip_bridge::DeviceBuffer>, // per-group f32 scales for x quantization
    /// FP16 scratch buffer for prefill X conversion. Sized to max(batch_size × K) × 2 bytes.
    fp16_x_scratch: Option<hip_bridge::DeviceBuffer>,
    fp16_x_scratch_bytes: usize,
    /// Pointer to the last FP32 source that was converted to fp16_x_scratch.
    /// If the next GEMM uses the same X, skip the conversion.
    pub fp16_x_source_ptr: *mut c_void,
    /// Q8_1/MMQ scratch for prefill activations. Layout matches llama.cpp's
    /// `block_q8_1_mmq`, ordered by [K/128 block, batch column].
    q8_1_mmq_x_scratch: Option<hip_bridge::DeviceBuffer>,
    q8_1_mmq_x_scratch_bytes: usize,

    // ── MMQ per-weight screening (#87) ──────────────────────────────────
    // When enabled, each weight matrix is screened on first MMQ use: a
    // small synthetic comparison (batch=16, WMMA vs MMQ) checks per-row
    // max abs error. Weights exceeding the threshold fall back to WMMA.
    //
    // Enabled by default on RDNA3/3.5. Configurable via:
    //   - config.json: `mmq_screen` (bool), `mmq_screen_threshold` (float)
    //   - per-model config overlay
    //   - daemon load params: `mmq_screen`, `mmq_screen_threshold`
    //   - env override: `HIPFIRE_MMQ_SCREEN=0` to disable,
    //     `HIPFIRE_MMQ_SCREEN_THRESHOLD=0.05` to tune
    mmq_screen_cache: HashMap<usize, bool>,
    /// Whether MMQ per-weight screening is enabled.
    /// Per-arch default (set in `Gpu::init`): true on gfx906, false elsewhere.
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
    /// Set via `HIPFIRE_BLOB_FORCE=1` at init. Blobs accumulate unbounded in
    /// `capture_blobs` while set — only intended for short diagnostic runs.
    pub force_blob_path: bool,
    /// Diagnostic: when true, gfx906 MMQ residual quantizes X to Q8_1 then
    /// returns FP16 wave64 instead of running dp4a — isolates the cost of
    /// the activation pre-quantize pass. Read once via `HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY=1`
    /// at init so the per-call hot path doesn't hit `env::var`'s global lock.
    pub mmq_diag_quantize_only: bool,
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
    /// scratch allocations (MQ signs/x_rot/x_q8, FP16 shadow, kernel modules)
    /// happen outside any captured region. Capturing the first call hits
    /// `hipMalloc not permitted under stream capture`. Set after the first
    /// direct run; the next call captures the graph for replay. Mirrors
    /// `verify_warmed_up` but uses a scalar flag (no per-B keying).
    pub ar_forward_warmed_up: bool,

    /// Per-B cache of captured verify-forward graphs. Each entry owns its
    /// graph + exec + the kernarg blobs that graph captured pointers into.
    /// Blobs must stay alive for the life of the graph — they're baked into
    /// the graph nodes by hipStreamEndCapture.
    ///
    /// Keyed by `b` (draft block size). DFlash's PLD intermittently shortens
    /// b from 16 → 8 on short self-match spines; caching graphs per-B avoids
    /// graph_destroy + re-capture every oscillation, which was wiping out the
    /// hipGraph replay gain entirely.
    pub verify_graph_cache: HashMap<usize, (hip_bridge::Graph, hip_bridge::GraphExec, Vec<Vec<u8>>)>,
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
    pub replay_graph_cache: HashMap<usize, (hip_bridge::Graph, hip_bridge::GraphExec, Vec<Vec<u8>>)>,
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
            self.hip.memset(&scratch, (n & 0xFF) as i32, SCRATCH_BYTES)?;
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
        let (_, vram_total) = hip.get_vram_info().unwrap_or((0, 0));

        // Check HIP runtime version matches GPU arch requirements
        let (hip_major, hip_minor) = hip.runtime_version().unwrap_or((0, 0));
        let (min_major, min_minor) = match arch.as_str() {
            "gfx1200" | "gfx1201" => (6, 4), // RDNA4 needs ROCm 6.4+
            "gfx1150" | "gfx1151" | "gfx1152" => (7, 2), // RDNA3.5 (Strix) needs ROCm 7.2+
            "gfx1100" | "gfx1101" | "gfx1102" => (5, 5), // RDNA3 needs ROCm 5.5+
            _ => (5, 0),
        };
        if hip_major > 0 && (hip_major < min_major || (hip_major == min_major && hip_minor < min_minor)) {
            eprintln!("WARNING: HIP runtime {}.{} may not support {}. Minimum: {}.{}", hip_major, hip_minor, arch, min_major, min_minor);
            eprintln!("  Update your HIP runtime or kernels may fail to load.");
        }
        eprintln!("GPU dev {}: {} ({:.1} GB VRAM, HIP {}.{})", id, arch, vram_total as f64 / 1e9, hip_major, hip_minor);

        let compiler = KernelCompiler::new(&arch)?;

        LAST_BOUND_DEVICE.with(|c| c.set(id));

        // Per-arch defaults for MMQ screening. See the mmq_screen and
        // mmq_screen_threshold fields below for rationale.
        let mmq_screen_default: bool = arch == "gfx906";
        let mmq_screen_threshold_default: f32 = if arch == "gfx906" { 0.50 } else { 0.10 };

        Ok(Self {
            hip,
            arch,
            device_id: id,
            compiler,
            modules: HashMap::new(),
            functions: HashMap::new(),
            pool: crate::pool::GpuPool::new(),
            active_stream: None,
            draft_stream: None,
            verify_stream: None,
            mq_signs1: None,
            mq_signs2: None,
            mq_x_rot: None,
            mq_x_q8: None,
            mq_x_scales: None,
            fp16_x_scratch: None,
            fp16_x_scratch_bytes: 0,
            fp16_x_source_ptr: std::ptr::null_mut(),
            q8_1_mmq_x_scratch: None,
            q8_1_mmq_x_scratch_bytes: 0,
            mmq_screen_cache: HashMap::new(),
            // Per-arch default for MMQ per-weight screening:
            //   gfx906: on (paired with the 0.50 threshold default below).
            //     Acts as a regression safety net; expected to reject 0
            //     weights at 0.50 threshold but catches future distribution
            //     issues. Cached per weight pointer, so cost is amortized.
            //   other archs: off — preserves prior behavior; flip only after
            //     similar validation.
            mmq_screen: std::env::var("HIPFIRE_MMQ_SCREEN").ok()
                .map(|v| v == "1")
                .unwrap_or(mmq_screen_default),
            // Default screening threshold: 0.10 absolute error per row,
            // measured against synthetic uniform [-2, 2] activations.
            // The 0.10 default was set when the gfx906 dp4a kernel was
            // buggy (commit 8081822); the post-redesign kernel (commit
            // c022682) is structurally cleaner and the same prompts pass
            // coherence at 0.50. Bumping the gfx906 default to 0.50
            // recovers the 30/72 weights that get rejected per Qwen 9B
            // load (mostly row 3994 of m=4096 matrices, a known
            // degenerate quant group). pp128 lifts 355 → 462 tok/s
            // (1.30×) at threshold=0.50 with no coherence regression
            // across all 4 mq4 rows of the gate. Other archs keep the
            // conservative 0.10 default until similar validation.
            mmq_screen_threshold: std::env::var("HIPFIRE_MMQ_SCREEN_THRESHOLD")
                .ok().and_then(|s| s.parse().ok())
                .unwrap_or(mmq_screen_threshold_default),
            capture_mode: false,
            force_blob_path: std::env::var("HIPFIRE_BLOB_FORCE").ok().as_deref() == Some("1"),
            mmq_diag_quantize_only: std::env::var("HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY").ok().as_deref() == Some("1"),
            capture_blobs: Vec::new(),
            graph_exec: None,
            captured_graph: None,
            graph_verify_n: None,
            graph_verify_warmup: 0,
            ar_forward_warmed_up: false,
            verify_graph_cache: HashMap::new(),
            verify_warmed_up: HashSet::new(),
            verify_capturing_b: None,
            replay_graph_cache: HashMap::new(),
            replay_warmed_up: HashSet::new(),
            replay_capturing_n: None,
            rocblas: None,
            fp16_shadow_cache: HashMap::new(),
        }).map(|mut gpu| {
            if gpu.force_blob_path {
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
        if self.rocblas.is_some() { return; }
        let cdna3 = matches!(self.arch.as_str(), "gfx940" | "gfx941" | "gfx942");
        let all_archs = std::env::var("HIPFIRE_ROCBLAS_ALL_ARCHS").ok().as_deref() == Some("1");
        if !cdna3 && !all_archs { return; }
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
                eprintln!("[rocblas] not available ({}); falling back to hand-rolled GEMMs", e);
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
        m: usize, k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(k % 256 == 0, "hfq4g256 dequant: K must be multiple of 256 (got {k})");
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
            self.hip.launch_kernel(func, [m as u32, groups, 1], [128, 1, 1], 0,
                self.stream_ref(), &mut params)
        }
    }

    /// CDNA3-only: prefill GEMM used by `gemm_hfq4g256` rocBLAS path.
    ///
    /// Computes Y_rowmajor[N × M] = X_rowmajor[N × K] · W_transposed, where
    /// the weight is stored row-major [M × K] but the operation needs W^T.
    /// This matches the engine's convention (weight dotted with each row of X
    /// produces one output column per batch row).
    ///
    /// rocBLAS is column-major. A row-major [M × K] matrix is byte-identical
    /// to a column-major [K × M] matrix. So the call is:
    ///   col-major C[M × N] = op_A(W) · X_col[K × N]
    /// with op_A = T (transpose the col-major [K × M] view of W to get [M × K]).
    /// X_row[N × K] viewed col-major is [K × N] with ld=K. Y_row[N × M] viewed
    /// col-major is [M × N] with ld=M — so pointer+ld match C directly.
    pub fn rocblas_gemm_hfq4_prefill(
        &self,
        w_fp16: &DeviceBuffer, // row-major [M × K]
        x_fp16: &DeviceBuffer, // row-major [N × K]
        y_fp32: &DeviceBuffer, // row-major [N × M]
        m: usize, n: usize, k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.rocblas_gemm_hfq4_generic(w_fp16, x_fp16, y_fp32, m, n, k, 1.0, 0.0)
    }

    /// Same op as `rocblas_gemm_hfq4_prefill` but with Y += alpha·(X·W^T) +
    /// beta·Y. Covers the residual-GEMM pattern (w_down on LA path, wo on
    /// attention path) where the existing hand-rolled kernels fuse the add.
    pub fn rocblas_gemm_hfq4_prefill_residual(
        &self,
        w_fp16: &DeviceBuffer,
        x_fp16: &DeviceBuffer,
        y_fp32: &DeviceBuffer,
        m: usize, n: usize, k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.rocblas_gemm_hfq4_generic(w_fp16, x_fp16, y_fp32, m, n, k, 1.0, 1.0)
    }

    fn rocblas_gemm_hfq4_generic(
        &self,
        w_fp16: &DeviceBuffer,
        x_fp16: &DeviceBuffer,
        y_fp32: &DeviceBuffer,
        m: usize, n: usize, k: usize,
        alpha: f32, beta: f32,
    ) -> HipResult<()> {
        use hip_bridge::{RocblasDatatype, RocblasOperation};
        let rb = self.rocblas.as_ref()
            .expect("rocblas_gemm_hfq4: rocBLAS not initialized");
        unsafe {
            rb.gemm_ex(
                RocblasOperation::Transpose, RocblasOperation::None,
                m as i32, n as i32, k as i32,
                &alpha as *const f32 as *const c_void,
                w_fp16.as_ptr(), RocblasDatatype::F16, k as i32,
                x_fp16.as_ptr(), RocblasDatatype::F16, k as i32,
                &beta as *const f32 as *const c_void,
                y_fp32.as_ptr(), RocblasDatatype::F32, m as i32,
                y_fp32.as_ptr(), RocblasDatatype::F32, m as i32,
                RocblasDatatype::F32,
            ).map_err(|e| hip_bridge::HipError::new(e.status, &format!("rocblas_gemm: {}", e.context)))
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
        let stream = self.active_stream.as_ref()
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
        let exec = self.graph_exec.as_ref().expect("no captured graph to replay");
        let stream = self.active_stream.as_ref().unwrap();
        self.hip.graph_launch(exec, stream)
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
        // Without this, model swap leaves the flag stuck on `true`. The
        // forward path in qwen35::forward_scratch then jumps straight from
        // graph_exec.is_none() into capture mode on the new model's first
        // AR forward call, before kernel JIT / scratch allocations have
        // happened against the new tensors. That trips
        // "hipMalloc not permitted under stream capture" the same way
        // verify_warmed_up does for the DFlash verify path.
        self.ar_forward_warmed_up = false;
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
        debug_assert!(self.verify_capturing_b.is_none(),
            "begin_verify_graph_capture: already capturing for b={:?}",
            self.verify_capturing_b);
        debug_assert!(!self.capture_mode,
            "begin_verify_graph_capture: capture_mode already set");
        self.capture_blobs.clear();
        self.verify_capturing_b = Some(b);
        self.capture_mode = true;
        let stream = self.active_stream.as_ref()
            .expect("verify graph capture requires an explicit stream");
        self.hip.stream_begin_capture(stream, 0) // hipStreamCaptureModeGlobal
    }

    /// End capture, instantiate, stash into the per-B cache (taking ownership
    /// of the current capture_blobs).
    pub fn end_verify_graph_capture(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        let b = self.verify_capturing_b.take()
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
        let entry = self.verify_graph_cache.get(&b)
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
        self.verify_warmed_up.clear();
        self.verify_capturing_b = None;
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
        debug_assert!(self.replay_capturing_n.is_none(),
            "begin_replay_graph_capture: already capturing for n_steps={:?}",
            self.replay_capturing_n);
        debug_assert!(!self.capture_mode,
            "begin_replay_graph_capture: capture_mode already set");
        self.capture_blobs.clear();
        self.replay_capturing_n = Some(n_steps);
        self.capture_mode = true;
        let stream = self.active_stream.as_ref()
            .expect("replay graph capture requires an explicit stream");
        self.hip.stream_begin_capture(stream, 0)
    }

    pub fn end_replay_graph_capture(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        let n_steps = self.replay_capturing_n.take()
            .expect("end_replay_graph_capture without matching begin");
        self.capture_mode = false;
        let stream = self.active_stream.as_ref().unwrap();
        let graph = self.hip.stream_end_capture(stream)?;
        let exec = self.hip.graph_instantiate(&graph)?;
        let blobs = std::mem::take(&mut self.capture_blobs);
        self.replay_graph_cache.insert(n_steps, (graph, exec, blobs));
        Ok(())
    }

    pub fn replay_graph_launch(&self, n_steps: usize) -> HipResult<()> {
        self.bind_thread()?;
        let entry = self.replay_graph_cache.get(&n_steps)
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
            self.hip.memcpy_dtod_async_at(dst, dst_offset, src, src_offset, size, stream)
        } else {
            self.hip.memcpy_dtod_at(dst, dst_offset, src, src_offset, size)
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
        if self.capture_mode || self.force_blob_path {
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
            let stream = self.active_stream.as_ref().map(|s| s as &hip_bridge::Stream);
            unsafe {
                self.hip.launch_kernel_blob(func, grid, block, shared_mem, stream, buf.as_mut_slice())
            }
        } else {
            let func = &self.functions[func_name];
            let stream = self.active_stream.as_ref().map(|s| s as &hip_bridge::Stream);
            unsafe {
                self.hip.launch_kernel(func, grid, block, shared_mem, stream, params)
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
            hip_bridge::HipError::new(0, &format!("launch_kernel_blob: function '{func_name}' not loaded"))
        })?;
        unsafe {
            self.hip.launch_kernel_blob(func, grid, block, shared_mem, self.stream_ref(), kernargs)
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

    /// Ensure the FP16 X scratch contains the conversion of `x`. Skips the
    /// convert kernel if `x.buf.as_ptr()` matches the last converted source.
    /// Returns the FP16 device pointer.
    fn ensure_fp16_x(&mut self, x: &GpuTensor, n_elems: usize) -> HipResult<*mut c_void> {
        self.ensure_kernel("convert_f32_to_f16", kernels::GEMM_HFQ4G256_RESIDUAL_FP16_SRC, "convert_f32_to_f16")?;

        let src_ptr = x.buf.as_ptr();
        let needed = n_elems * 2;

        // Grow scratch if needed (never shrinks)
        if self.fp16_x_scratch_bytes < needed {
            self.fp16_x_scratch = Some(self.hip.malloc(needed)?);
            self.fp16_x_scratch_bytes = needed;
            self.fp16_x_source_ptr = std::ptr::null_mut(); // force reconversion after realloc
        }

        // Under graph capture, convert EVERY call: the src/dst pointers are
        // stable (PrefillBatchScratch + persistent FP16 scratch), but the
        // DATA at src changes every replay, so the captured convert-node
        // needs to re-run. The pointer-equality cache would wrongly skip
        // the node and read stale FP16 on replay. During normal dispatch
        // the skip is still correct.
        let must_convert = self.capture_mode || self.fp16_x_source_ptr != src_ptr;
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
                "convert_f32_to_f16", [grid, 1, 1], [256, 1, 1], 0, &mut conv_params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(in_ptr); b.push_ptr(out_ptr); b.push_i32(n_val);
                    b
                },
            )?;
            self.fp16_x_source_ptr = src_ptr;
        }

        Ok(self.fp16_x_scratch.as_ref().unwrap().as_ptr())
    }

    /// Ensure prefill activations are quantized into a llama.cpp-style
    /// `block_q8_1_mmq` layout. The scratch is ordered by [K/128 block, batch]
    /// so a 128-column batch tile is contiguous for each K tile.
    pub fn ensure_q8_1_mmq_x(&mut self, x: &GpuTensor, batch_size: usize, k: usize) -> HipResult<*mut c_void> {
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
        let x_data: Vec<f32> = (0..screen_batch * k).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let t = (state >> 33) as f32 / (u32::MAX as f32);
            t * 4.0 - 2.0
        }).collect();

        let result = (|| -> HipResult<bool> {
            let x_gpu = self.upload_f32(&x_data, &[screen_batch * k])?;
            let y_wmma = self.zeros(&[screen_batch * m], DType::F32)?;
            let y_mmq = self.zeros(&[screen_batch * m], DType::F32)?;

            let saved_capture = self.capture_mode;
            self.capture_mode = true;

            // Reference path: use FP16 wave64 on gfx906, WMMA otherwise
            if self.arch == "gfx906" {
                self.gemm_hfq4g256_residual_fp16_wave64(a_raw, &x_gpu, &y_wmma, m, k, screen_batch)?;
            } else {
                self.gemm_hfq4g256_residual_wmma(a_raw, &x_gpu, &y_wmma, m, k, screen_batch)?;
            }

            // MMQ path
            let xq = self.ensure_q8_1_mmq_x(&x_gpu, screen_batch, k)?;
            if self.arch == "gfx906" {
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
                    if err > row_max { row_max = err; }
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
        m: usize, k: usize,
    ) -> HipResult<Option<*mut c_void>> {
        if self.rocblas.is_none() { return Ok(None); }
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

    /// Whether the arch is eligible for the rocBLAS/MFMA batched-prefill
    /// path. Default: CDNA3 only (MI300-series, gfx94x). Override with
    /// `HIPFIRE_ROCBLAS_ALL_ARCHS=1` for local testing on RDNA3+ — rocBLAS
    /// runs fine there (uses WMMA backends on RDNA3, not MFMA) so this is
    /// a useful smoke-path in the absence of an MI300.
    fn rocblas_arch_eligible(&self) -> bool {
        static CACHE: OnceLock<bool> = OnceLock::new();
        let all_archs = *CACHE.get_or_init(|| {
            std::env::var("HIPFIRE_ROCBLAS_ALL_ARCHS").ok().as_deref() == Some("1")
        });
        if all_archs { return self.rocblas.is_some(); }
        matches!(self.arch.as_str(), "gfx940" | "gfx941" | "gfx942")
    }

    /// Configurable batch threshold for MFMA dispatch. Below this we stay on
    /// the hand-rolled GEMV — rocBLAS launch overhead eats the compute win
    /// at tiny batches. Overridable via `HIPFIRE_ROCBLAS_MIN_BATCH` env var.
    ///
    /// Kill-switch: `HIPFIRE_ROCBLAS_OFF=1` forces the threshold to usize::MAX,
    /// which disables the rocBLAS path entirely for A/B benchmarking against
    /// the hand-rolled GEMV baseline.
    fn rocblas_min_batch(&self) -> usize {
        static CACHE: OnceLock<usize> = OnceLock::new();
        *CACHE.get_or_init(|| {
            if std::env::var("HIPFIRE_ROCBLAS_OFF").ok().as_deref() == Some("1") {
                return usize::MAX;
            }
            std::env::var("HIPFIRE_ROCBLAS_MIN_BATCH")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(4)
        })
    }

    /// Pre-compile a batch of kernels in parallel (hipcc), then load modules + functions.
    /// Each entry is (module_name, source, func_name). Turbo kernels should have
    /// TURBO_COMMON_H already prepended in their source.
    pub fn precompile_kernels(&mut self, specs: &[(&str, &str, &str)]) -> HipResult<()> {
        self.bind_thread()?;
        // Collect (name, source) pairs for the compiler batch, skipping already-loaded
        let batch: Vec<(&str, &str)> = specs.iter()
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
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        self.hip.memcpy_htod(&tensor.buf, bytes)?;
        Ok(tensor)
    }

    pub fn download_f32(&self, tensor: &GpuTensor) -> HipResult<Vec<f32>> {
        self.bind_thread()?;
        let numel = tensor.numel();
        let mut data = vec![0.0f32; numel];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, numel * 4)
        };
        self.hip.memcpy_dtoh(bytes, &tensor.buf)?;
        Ok(data)
    }

    pub fn zeros(&mut self, shape: &[usize], dtype: DType) -> HipResult<GpuTensor> {
        self.bind_thread()?;
        let tensor = self.alloc_tensor(shape, dtype)?;
        match self.active_stream.as_ref() {
            Some(stream) => self.hip.memset_async(&tensor.buf, 0, tensor.byte_size(), stream)?,
            None => self.hip.memset(&tensor.buf, 0, tensor.byte_size())?,
        }
        Ok(tensor)
    }

    /// GPU-side embedding lookup: copy row `token_id` from embedding table to output.
    /// Avoids downloading the entire embedding table to CPU.
    pub fn embedding_lookup(
        &self,
        table: &GpuTensor,  // [vocab_size * dim] F32
        output: &GpuTensor, // [dim] F32
        token_id: u32,
        dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let byte_offset = (token_id as usize) * dim * 4;
        let byte_size = dim * 4;
        self.hip.memcpy_dtod_offset(&output.buf, &table.buf, byte_offset, byte_size)
    }

    /// Q4_LUT GEMV: 4-bit with LDS codebook lookup. 48 bytes per 32 elements.
    pub fn gemv_q4lut(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_q4lut", kernels::GEMV_Q4LUT_SRC, "gemv_q4lut")?;
        let func = &self.functions["gemv_q4lut"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        // LDS: 8 codebooks × 16 entries × 2 bytes = 256 bytes
        let shared_mem = 256u32;
        unsafe {
            self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], shared_mem, None, &mut params)
        }
    }

    /// Wave-cooperative Q4 GEMV (Q4_F16_G32 format, 0.625 B/w). Shuffle-based nibble distribution.
    pub fn gemv_q4wave(
        &mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_q4wave", kernels::GEMV_Q4WAVE_SRC, "gemv_q4wave")?;
        let func = &self.functions["gemv_q4wave"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void, &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void, &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, None, &mut params) }
    }

    /// Q4-as-Q8 GEMV: 4-bit precision stored in Q8_0 format (1.0625 B/w). Gets Q8 occupancy.
    pub fn gemv_q4as8(
        &mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_q4as8", kernels::GEMV_Q4AS8_SRC, "gemv_q4as8")?;
        let func = &self.functions["gemv_q4as8"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void, &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void, &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, None, &mut params) }
    }

    /// Q8_0 embedding lookup: dequantize one row on GPU, output F32.
    pub fn embedding_lookup_q8(
        &mut self,
        table: &GpuTensor,
        output: &GpuTensor,
        token_id: u32,
        dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("embedding_q8", kernels::EMBEDDING_Q8_SRC, "embedding_q8")?;
        let func = &self.functions["embedding_q8"];

        let mut tp = table.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut tid = token_id as i32;
        let mut d = dim as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut tp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut tid as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
        ];

        unsafe {
            self.hip.launch_kernel(func, [1, 1, 1], [256, 1, 1], 0, None, &mut params)
        }
    }

    /// Q4_K embedding lookup: dequantize one row on GPU, output F32.
    /// table is raw Q4_K bytes on GPU, output is [dim] F32.
    pub fn embedding_lookup_q4k(
        &mut self,
        table: &GpuTensor,
        output: &GpuTensor,
        token_id: u32,
        dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("embedding_q4k", kernels::EMBEDDING_Q4K_SRC, "embedding_q4k")?;
        let func = &self.functions["embedding_q4k"];

        let mut tp = table.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut tid = token_id as i32;
        let mut d = dim as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut tp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut tid as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
        ];

        unsafe {
            self.hip.launch_kernel(func, [1, 1, 1], [256, 1, 1], 0, None, &mut params)
        }
    }

    /// HFQ4-G256 embedding lookup: dequantize one row on GPU, output F32.
    pub fn embedding_lookup_hfq4g256(
        &mut self,
        table: &GpuTensor,
        output: &GpuTensor,
        token_id: u32,
        dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("embedding_hfq4g256", kernels::EMBEDDING_HFQ4G256_SRC, "embedding_hfq4g256")?;
        let func = &self.functions["embedding_hfq4g256"];

        let mut tp = table.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut tid = token_id as i32;
        let mut d = dim as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut tp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut tid as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
        ];

        let bytes = crate::profile::embedding_hfq4g256_bytes(dim);
        let timer = crate::profile::begin_timer(&self.hip, "embedding", "embedding_lookup_hfq4g256", bytes);
        let result = unsafe {
            self.hip.launch_kernel(func, [1, 1, 1], [256, 1, 1], 0, self.stream_ref(), &mut params)
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched Q8_0 embedding lookup. Same hipGraph-captureable pattern as
    /// the HFQ4G256 variant. `output` shape: `[n × dim]` row-major.
    pub fn embedding_lookup_q8_batched(
        &mut self,
        table: &GpuTensor,
        output: &GpuTensor,
        token_ids: &GpuTensor,
        n: usize,
        dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "embedding_q8_batched",
            kernels::EMBEDDING_Q8_BATCHED_SRC,
            "embedding_q8_batched",
        )?;

        let mut tp = table.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut tidp = token_ids.buf.as_ptr();
        let mut d = dim as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut tp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut tidp as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
        ];

        self.launch_maybe_blob(
            "embedding_q8_batched",
            [n as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(tp); b.push_ptr(op); b.push_ptr(tidp); b.push_i32(d);
                b
            },
        )
    }

    /// Batched HFQ4-G256 embedding lookup. Dequantizes N rows in a single
    /// launch, reading token ids from a device buffer. hipGraph-capture-safe:
    /// callers update `token_ids` between replays and replay the same graph.
    ///
    /// `output` shape: `[n × dim]` row-major. `token_ids` shape: `[n]` i32.
    pub fn embedding_lookup_hfq4g256_batched(
        &mut self,
        table: &GpuTensor,
        output: &GpuTensor,
        token_ids: &GpuTensor,
        n: usize,
        dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "embedding_hfq4g256_batched",
            kernels::EMBEDDING_HFQ4G256_BATCHED_SRC,
            "embedding_hfq4g256_batched",
        )?;

        let mut tp = table.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut tidp = token_ids.buf.as_ptr();
        let mut d = dim as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut tp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut tidp as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
        ];

        self.launch_maybe_blob(
            "embedding_hfq4g256_batched",
            [n as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(tp); b.push_ptr(op); b.push_ptr(tidp); b.push_i32(d);
                b
            },
        )
    }

    /// HFQ4-G128 embedding lookup: dequantize one row on GPU, output F32.
    pub fn embedding_lookup_hfq4g128(
        &mut self,
        table: &GpuTensor,
        output: &GpuTensor,
        token_id: u32,
        dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("embedding_hfq4g128", kernels::EMBEDDING_HFQ4G128_SRC, "embedding_hfq4g128")?;
        let func = &self.functions["embedding_hfq4g128"];

        let mut tp = table.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut tid = token_id as i32;
        let mut d = dim as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut tp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut tid as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
        ];

        unsafe {
            self.hip.launch_kernel(func, [1, 1, 1], [256, 1, 1], 0, self.stream_ref(), &mut params)
        }
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

    // ── Kernel operations ───────────────────────────────────────

    /// y = A * x (matrix-vector multiply, A is [M, K], x is [K], y is [M])
    pub fn gemv_f32(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv", kernels::GEMV_SRC, "gemv_f32")?;
        let func = &self.functions["gemv_f32"];

        let m = a.shape[0] as i32;
        let k = a.shape[1] as i32;
        let alpha = 1.0f32;
        let beta = 0.0f32;

        let mut a_ptr = a.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m;
        let mut k_val = k;
        let mut alpha_val = alpha;
        let mut beta_val = beta;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut alpha_val as *mut _ as *mut c_void,
            &mut beta_val as *mut _ as *mut c_void,
        ];

        // One block per row, 256 threads per block with shared memory reduction
        let block_size = 256u32.min(k as u32);
        let shared_mem = block_size * 4; // one float per thread
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// y = A_q4k * x (quantized matrix-vector multiply, A stored as Q4_K on GPU)
    /// a_raw: raw Q4_K bytes on GPU, x: F32 input, y: F32 output
    /// m: number of output rows, k: number of input columns (must be multiple of 256)
    pub fn gemv_q4k(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_q4k", kernels::GEMV_Q4K_SRC, "gemv_q4k")?;
        let func = &self.functions["gemv_q4k"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        let block_size = 32u32; // single warp — no shared memory needed
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [block_size, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// HFQ4-G128 GEMV: flat 4-bit with 128-weight groups.
    /// K must be multiple of 128.
    pub fn gemv_hfq4g128(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_hfq4g128", kernels::GEMV_HFQ4G128_SRC, "gemv_hfq4g128")?;
        let func = &self.functions["gemv_hfq4g128"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

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

    /// Batched HFQ4-G128 GEMM. Same tiled approach as G256.
    pub fn gemm_hfq4g128(
        &mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor,
        m: usize, k: usize, batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemm_hfq4g128", kernels::GEMM_HFQ4G128_SRC, "gemm_hfq4g128")?;
        let func = &self.functions["gemm_hfq4g128"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];
        let batch_tiles = ((batch_size + 7) / 8) as u32;
        unsafe {
            self.hip.launch_kernel(func, [m as u32, batch_tiles, 1], [32, 1, 1], 0, self.stream_ref(), &mut params)
        }
    }

    /// HFQ2-G256 GEMV. K must be multiple of 256.
    pub fn gemv_hfq2g256(&mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_hfq2g256", kernels::GEMV_HFQ2G256_SRC, "gemv_hfq2g256")?;
        let func = &self.functions["gemv_hfq2g256"];
        let mut a_ptr = a_raw.buf.as_ptr(); let mut x_ptr = x.buf.as_ptr(); let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32; let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void, &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void, &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// MQ2-Lloyd GEMV (2-bit + per-block 4-entry fp16 codebook). K must be a
    /// multiple of 256. Same launch shape as gemv_hfq2g256 — header is the
    /// only layout difference.
    pub fn gemv_mq2g256_lloyd(&mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_mq2g256_lloyd", kernels::GEMV_MQ2G256_LLOYD_SRC, "gemv_mq2g256_lloyd")?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "gemv_mq2g256_lloyd", [m as u32, 1, 1], [32, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr); b.push_ptr(x_ptr); b.push_ptr(y_ptr);
                b.push_i32(m_val); b.push_i32(k_val);
                b
            },
        )
    }

    /// MQ2-Lloyd GEMV with engine-side x rotation (matches `gemv_mq2g256_with_rotate`).
    pub fn gemv_mq2g256_lloyd_with_rotate(
        &mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor,
        x_rot: &GpuTensor, m: usize, k: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to rotate_x_mq + gemv_mq2g256_lloyd, both of which bind.
        self.rotate_x_mq(x, x_rot, k)?;
        self.gemv_mq2g256_lloyd(a_raw, x_rot, y, m, k)
    }

    /// MQ3-Lloyd GEMV (3-bit + per-block 8-entry fp16 codebook). K must be a
    /// multiple of 256. gfx1100/1101/1102 use the K4-unrolled + LDS-codebook
    /// variant; other archs fall back to the baseline switch-dispatch path.
    pub fn gemv_mq3g256_lloyd(&mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemv_mq3g256_lloyd_for_arch(&self.arch);
        self.ensure_kernel(module, src, "gemv_mq3g256_lloyd")?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::gemv_mq3g256_lloyd_bytes(m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", "gemv_mq3g256_lloyd", bytes);
        let result = self.launch_maybe_blob(
            "gemv_mq3g256_lloyd", [m as u32, 1, 1], [32, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr); b.push_ptr(x_ptr); b.push_ptr(y_ptr);
                b.push_i32(m_val); b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// MQ3-Lloyd GEMV with engine-side x rotation.
    pub fn gemv_mq3g256_lloyd_with_rotate(
        &mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor,
        x_rot: &GpuTensor, m: usize, k: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to rotate_x_mq + gemv_mq3g256_lloyd, both of which bind.
        self.rotate_x_mq(x, x_rot, k)?;
        self.gemv_mq3g256_lloyd(a_raw, x_rot, y, m, k)
    }

    /// MQ3-Lloyd GEMV with fused residual add: y[row] += A[row] · x. Used by
    /// `weight_gemv_residual` MQ3-Lloyd arm to eliminate the alloc + gemv +
    /// add_inplace_f32 + free fallback chain (saves ~4.4% of decode time on
    /// 9B Lloyd-MQ3, gfx1100, per the 2026-05-06 decode profile).
    pub fn gemv_mq3g256_lloyd_residual(&mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemv_mq3g256_lloyd_residual_for_arch(&self.arch);
        self.ensure_kernel(module, src, "gemv_mq3g256_lloyd_residual")?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::gemv_mq3g256_lloyd_bytes(m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", "gemv_mq3g256_lloyd_residual", bytes);
        let result = self.launch_maybe_blob(
            "gemv_mq3g256_lloyd_residual", [m as u32, 1, 1], [32, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr); b.push_ptr(x_ptr); b.push_ptr(y_ptr);
                b.push_i32(m_val); b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// MQ3-Lloyd residual GEMV with engine-side x rotation.
    pub fn gemv_mq3g256_lloyd_residual_with_rotate(
        &mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor,
        x_rot: &GpuTensor, m: usize, k: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to rotate_x_mq + gemv_mq3g256_lloyd_residual.
        self.rotate_x_mq(x, x_rot, k)?;
        self.gemv_mq3g256_lloyd_residual(a_raw, x_rot, y, m, k)
    }

    /// MQ3-Lloyd WMMA residual GEMM (Phase 5 / issue #116, Phase B1).
    /// Mirrors `gemm_hfq3g256_residual_wmma` shape + grid; group stride is 112 B
    /// (16 B fp16 codebook + 96 B 3-bit indices) instead of HFQ3's 104. K must
    /// be a multiple of 256. gfx11/gfx12 wave32 WMMA; other archs fall through
    /// to the baseline kernel (which itself currently requires WMMA — caller
    /// must check arch before dispatching).
    /// Caller is responsible for pre-rotating X (FWHT) for the MQ3-Lloyd dtype;
    /// this dispatch mirrors `gemm_hfq3g256_residual_wmma` and does not rotate.
    /// fp16-LDS staging — fp16 won the Phase A bench by 7.15% (devlog
    /// 2026-05-07).
    pub fn gemm_mq3g256_lloyd_residual_wmma(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemm_mq3g256_lloyd_residual_wmma_for_arch(&self.arch);
        self.ensure_kernel(module, src, "gemm_mq3g256_lloyd_residual_wmma")?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        let row_tiles = (m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let weight_bytes = m * (k / 256) * LLOYD_MQ3_GROUP_BYTES;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip, "gemm", "gemm_mq3g256_lloyd_residual_wmma", bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_mq3g256_lloyd_residual_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr); b.push_ptr(x_ptr); b.push_ptr(y_ptr);
                b.push_i32(m_val); b.push_i32(k_val); b.push_i32(bs_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// MQ3-Lloyd WMMA fused QKVZA GEMM (LA preamble: qkv + z + beta + alpha).
    /// 4-way fused — one launch covers all four projections of the LA layer.
    /// Caller pre-rotates X (FWHT) for MQ3-Lloyd dtype.
    pub fn gemm_qkvza_mq3g256_lloyd_wmma(
        &mut self,
        a_qkv: &GpuTensor, a_z: &GpuTensor, a_beta: &GpuTensor, a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor, y_z: &GpuTensor, y_beta: &GpuTensor, y_alpha: &GpuTensor,
        qkv_m: usize, z_m: usize, beta_m: usize, alpha_m: usize,
        k: usize, n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemm_qkvza_mq3g256_lloyd_wmma_for_arch(&self.arch);
        self.ensure_kernel(module, src, "gemm_qkvza_mq3g256_lloyd_wmma")?;
        let x_f16_ptr = self.ensure_fp16_x(x, n * k)?;

        let mut a_qkv_p = a_qkv.buf.as_ptr();
        let mut a_z_p = a_z.buf.as_ptr();
        let mut a_beta_p = a_beta.buf.as_ptr();
        let mut a_alpha_p = a_alpha.buf.as_ptr();
        let mut x_p = x_f16_ptr;
        let mut y_qkv_p = y_qkv.buf.as_ptr();
        let mut y_z_p = y_z.buf.as_ptr();
        let mut y_beta_p = y_beta.buf.as_ptr();
        let mut y_alpha_p = y_alpha.buf.as_ptr();
        let mut qkv_m_v = qkv_m as i32;
        let mut z_m_v = z_m as i32;
        let mut beta_m_v = beta_m as i32;
        let mut alpha_m_v = alpha_m as i32;
        let mut k_v = k as i32;
        let mut n_v = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_qkv_p as *mut _ as *mut c_void,
            &mut a_z_p as *mut _ as *mut c_void,
            &mut a_beta_p as *mut _ as *mut c_void,
            &mut a_alpha_p as *mut _ as *mut c_void,
            &mut x_p as *mut _ as *mut c_void,
            &mut y_qkv_p as *mut _ as *mut c_void,
            &mut y_z_p as *mut _ as *mut c_void,
            &mut y_beta_p as *mut _ as *mut c_void,
            &mut y_alpha_p as *mut _ as *mut c_void,
            &mut qkv_m_v as *mut _ as *mut c_void,
            &mut z_m_v as *mut _ as *mut c_void,
            &mut beta_m_v as *mut _ as *mut c_void,
            &mut alpha_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (n + 15) / 16;
        let weight_bytes = total_m * (k / 256) * LLOYD_MQ3_GROUP_BYTES;
        let bytes = weight_bytes + n * k * 2 + n * total_m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip, "gemm", "gemm_qkvza_mq3g256_lloyd_wmma", bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_qkvza_mq3g256_lloyd_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_qkv_p); b.push_ptr(a_z_p); b.push_ptr(a_beta_p); b.push_ptr(a_alpha_p);
                b.push_ptr(x_p);
                b.push_ptr(y_qkv_p); b.push_ptr(y_z_p); b.push_ptr(y_beta_p); b.push_ptr(y_alpha_p);
                b.push_i32(qkv_m_v); b.push_i32(z_m_v); b.push_i32(beta_m_v); b.push_i32(alpha_m_v);
                b.push_i32(k_v); b.push_i32(n_v);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// MQ3-Lloyd WMMA fused QKV GEMM (FA preamble: q + k + v).
    pub fn gemm_qkv_mq3g256_lloyd_wmma(
        &mut self,
        a_q: &GpuTensor, a_k: &GpuTensor, a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor, y_k: &GpuTensor, y_v: &GpuTensor,
        q_m: usize, k_m: usize, v_m: usize,
        k: usize, n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemm_qkv_mq3g256_lloyd_wmma_for_arch(&self.arch);
        self.ensure_kernel(module, src, "gemm_qkv_mq3g256_lloyd_wmma")?;
        let x_f16_ptr = self.ensure_fp16_x(x, n * k)?;

        let mut a_q_p = a_q.buf.as_ptr();
        let mut a_k_p = a_k.buf.as_ptr();
        let mut a_v_p = a_v.buf.as_ptr();
        let mut x_p = x_f16_ptr;
        let mut y_q_p = y_q.buf.as_ptr();
        let mut y_k_p = y_k.buf.as_ptr();
        let mut y_v_p = y_v.buf.as_ptr();
        let mut q_m_v = q_m as i32;
        let mut k_m_v = k_m as i32;
        let mut v_m_v = v_m as i32;
        let mut k_v = k as i32;
        let mut n_v = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_q_p as *mut _ as *mut c_void,
            &mut a_k_p as *mut _ as *mut c_void,
            &mut a_v_p as *mut _ as *mut c_void,
            &mut x_p as *mut _ as *mut c_void,
            &mut y_q_p as *mut _ as *mut c_void,
            &mut y_k_p as *mut _ as *mut c_void,
            &mut y_v_p as *mut _ as *mut c_void,
            &mut q_m_v as *mut _ as *mut c_void,
            &mut k_m_v as *mut _ as *mut c_void,
            &mut v_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (n + 15) / 16;
        let weight_bytes = total_m * (k / 256) * LLOYD_MQ3_GROUP_BYTES;
        let bytes = weight_bytes + n * k * 2 + n * total_m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip, "gemm", "gemm_qkv_mq3g256_lloyd_wmma", bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_qkv_mq3g256_lloyd_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_q_p); b.push_ptr(a_k_p); b.push_ptr(a_v_p);
                b.push_ptr(x_p);
                b.push_ptr(y_q_p); b.push_ptr(y_k_p); b.push_ptr(y_v_p);
                b.push_i32(q_m_v); b.push_i32(k_m_v); b.push_i32(v_m_v);
                b.push_i32(k_v); b.push_i32(n_v);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// MQ3-Lloyd WMMA fused gate+up GEMM (FFN preamble).
    pub fn gemm_gate_up_mq3g256_lloyd_wmma(
        &mut self,
        a_gate: &GpuTensor, a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor, y_up: &GpuTensor,
        gate_m: usize, up_m: usize,
        k: usize, n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemm_gate_up_mq3g256_lloyd_wmma_for_arch(&self.arch);
        self.ensure_kernel(module, src, "gemm_gate_up_mq3g256_lloyd_wmma")?;
        let x_f16_ptr = self.ensure_fp16_x(x, n * k)?;

        let mut a_gate_p = a_gate.buf.as_ptr();
        let mut a_up_p = a_up.buf.as_ptr();
        let mut x_p = x_f16_ptr;
        let mut y_gate_p = y_gate.buf.as_ptr();
        let mut y_up_p = y_up.buf.as_ptr();
        let mut gate_m_v = gate_m as i32;
        let mut up_m_v = up_m as i32;
        let mut k_v = k as i32;
        let mut n_v = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_gate_p as *mut _ as *mut c_void,
            &mut a_up_p as *mut _ as *mut c_void,
            &mut x_p as *mut _ as *mut c_void,
            &mut y_gate_p as *mut _ as *mut c_void,
            &mut y_up_p as *mut _ as *mut c_void,
            &mut gate_m_v as *mut _ as *mut c_void,
            &mut up_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (n + 15) / 16;
        let weight_bytes = total_m * (k / 256) * LLOYD_MQ3_GROUP_BYTES;
        let bytes = weight_bytes + n * k * 2 + n * total_m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip, "gemm", "gemm_gate_up_mq3g256_lloyd_wmma", bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_gate_up_mq3g256_lloyd_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_gate_p); b.push_ptr(a_up_p);
                b.push_ptr(x_p);
                b.push_ptr(y_gate_p); b.push_ptr(y_up_p);
                b.push_i32(gate_m_v); b.push_i32(up_m_v);
                b.push_i32(k_v); b.push_i32(n_v);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Fused Gate+Up MQ3-Lloyd: two GEMVs in one launch. Mirrors
    /// `fused_gate_up_hfq4g256` for the Lloyd-MQ3 dtype. Caller is
    /// responsible for pre-rotating x (FWHT) before invoking; the kernel
    /// itself only does the GEMV. Both `a_gate` and `a_up` must be MQ3-Lloyd
    /// matrices with the same K and codebook layout.
    pub fn fused_gate_up_mq3g256_lloyd(
        &mut self,
        a_gate: &GpuTensor, a_up: &GpuTensor, x: &GpuTensor,
        y_gate: &GpuTensor, y_up: &GpuTensor,
        gate_m: usize, up_m: usize, k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::fused_gate_up_mq3g256_lloyd_for_arch(&self.arch);
        self.ensure_kernel(module, src, "fused_gate_up_mq3g256_lloyd")?;
        let ag = a_gate.buf.as_ptr();
        let au = a_up.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yg = y_gate.buf.as_ptr();
        let yu = y_up.buf.as_ptr();
        let gm = gate_m as i32;
        let um = up_m as i32;
        let kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ag as *const _ as *mut c_void, &au as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void, &yg as *const _ as *mut c_void,
            &yu as *const _ as *mut c_void, &gm as *const _ as *mut c_void,
            &um as *const _ as *mut c_void, &kv as *const _ as *mut c_void,
        ];
        let total = (gate_m + up_m) as u32;
        // Bandwidth: A_gate + A_up read, x read once, y_gate + y_up written.
        let bytes = crate::profile::gemv_mq3g256_lloyd_bytes(gate_m, k)
            + crate::profile::gemv_mq3g256_lloyd_bytes(up_m, k)
            - k * 4;  // x is shared, don't double-count
        let timer = crate::profile::begin_timer(&self.hip, "fused", "fused_gate_up_mq3g256_lloyd", bytes);
        let result = self.launch_maybe_blob(
            "fused_gate_up_mq3g256_lloyd", [total, 1, 1], [32, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ag); b.push_ptr(au); b.push_ptr(xp);
                b.push_ptr(yg); b.push_ptr(yu);
                b.push_i32(gm); b.push_i32(um); b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Fused QKVZA MQ3-Lloyd: 4 LA-preamble GEMVs in one launch. Used by
    /// qwen35.rs DeltaNet decode when wqkv + wz + w_beta + w_alpha are
    /// all MQ3G256Lloyd. Mirrors `fused_qkvza_hfq4g256` — same routing
    /// (grid = qkv_m + z_m + beta_m + alpha_m, block picks A by gid),
    /// Lloyd K4+LDS body on gfx1100. Caller is responsible for
    /// pre-rotating x (FWHT); the kernel only does the GEMVs.
    pub fn fused_qkvza_mq3g256_lloyd(
        &mut self,
        a_qkv: &GpuTensor, a_z: &GpuTensor, a_beta: &GpuTensor, a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor, y_z: &GpuTensor, y_beta: &GpuTensor, y_alpha: &GpuTensor,
        qkv_m: usize, z_m: usize, beta_m: usize, alpha_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::fused_qkvza_mq3g256_lloyd_for_arch(&self.arch);
        self.ensure_kernel(module, src, "fused_qkvza_mq3g256_lloyd")?;
        let aq = a_qkv.buf.as_ptr();
        let az = a_z.buf.as_ptr();
        let ab = a_beta.buf.as_ptr();
        let aa = a_alpha.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yq = y_qkv.buf.as_ptr();
        let yz = y_z.buf.as_ptr();
        let yb = y_beta.buf.as_ptr();
        let ya = y_alpha.buf.as_ptr();
        let q_m_i = qkv_m as i32;
        let z_m_i = z_m as i32;
        let b_m_i = beta_m as i32;
        let a_m_i = alpha_m as i32;
        let k_i = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &aq as *const _ as *mut c_void, &az as *const _ as *mut c_void,
            &ab as *const _ as *mut c_void, &aa as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yq as *const _ as *mut c_void, &yz as *const _ as *mut c_void,
            &yb as *const _ as *mut c_void, &ya as *const _ as *mut c_void,
            &q_m_i as *const _ as *mut c_void, &z_m_i as *const _ as *mut c_void,
            &b_m_i as *const _ as *mut c_void, &a_m_i as *const _ as *mut c_void,
            &k_i as *const _ as *mut c_void,
        ];
        let total = (qkv_m + z_m + beta_m + alpha_m) as u32;
        // Bandwidth: 4 weight matrices read once each, x shared (read once).
        let bytes = crate::profile::gemv_mq3g256_lloyd_bytes(qkv_m, k)
            + crate::profile::gemv_mq3g256_lloyd_bytes(z_m, k)
            + crate::profile::gemv_mq3g256_lloyd_bytes(beta_m, k)
            + crate::profile::gemv_mq3g256_lloyd_bytes(alpha_m, k)
            - 3 * (k * 4); // x is shared, don't quadruple-count
        let timer = crate::profile::begin_timer(&self.hip, "fused", "fused_qkvza_mq3g256_lloyd", bytes);
        let result = self.launch_maybe_blob(
            "fused_qkvza_mq3g256_lloyd", [total, 1, 1], [32, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq); b.push_ptr(az); b.push_ptr(ab); b.push_ptr(aa);
                b.push_ptr(xp);
                b.push_ptr(yq); b.push_ptr(yz); b.push_ptr(yb); b.push_ptr(ya);
                b.push_i32(q_m_i); b.push_i32(z_m_i); b.push_i32(b_m_i); b.push_i32(a_m_i);
                b.push_i32(k_i);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Fused QKV MQ3-Lloyd: 3 FA-preamble GEMVs in one launch. Used by
    /// qwen35.rs FullAttention decode when wq + wk + wv are all
    /// MQ3G256Lloyd. Sibling of `fused_qkvza_mq3g256_lloyd` for the
    /// 3-projection FA case (vs LA's 4-projection QKVZA). Caller is
    /// responsible for pre-rotating x; the kernel only does the GEMVs.
    pub fn fused_qkv_mq3g256_lloyd(
        &mut self,
        a_q: &GpuTensor, a_k: &GpuTensor, a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor, y_k: &GpuTensor, y_v: &GpuTensor,
        q_m: usize, k_m: usize, v_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::fused_qkv_mq3g256_lloyd_for_arch(&self.arch);
        self.ensure_kernel(module, src, "fused_qkv_mq3g256_lloyd")?;
        let aq = a_q.buf.as_ptr();
        let ak = a_k.buf.as_ptr();
        let av = a_v.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yq = y_q.buf.as_ptr();
        let yk = y_k.buf.as_ptr();
        let yv = y_v.buf.as_ptr();
        let q_m_i = q_m as i32;
        let k_m_i = k_m as i32;
        let v_m_i = v_m as i32;
        let k_i = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &aq as *const _ as *mut c_void, &ak as *const _ as *mut c_void,
            &av as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yq as *const _ as *mut c_void, &yk as *const _ as *mut c_void,
            &yv as *const _ as *mut c_void,
            &q_m_i as *const _ as *mut c_void, &k_m_i as *const _ as *mut c_void,
            &v_m_i as *const _ as *mut c_void,
            &k_i as *const _ as *mut c_void,
        ];
        let total = (q_m + k_m + v_m) as u32;
        // Bandwidth: 3 weight matrices read once each, x shared (read once).
        let bytes = crate::profile::gemv_mq3g256_lloyd_bytes(q_m, k)
            + crate::profile::gemv_mq3g256_lloyd_bytes(k_m, k)
            + crate::profile::gemv_mq3g256_lloyd_bytes(v_m, k)
            - 2 * (k * 4); // x is shared, don't triple-count
        let timer = crate::profile::begin_timer(&self.hip, "fused", "fused_qkv_mq3g256_lloyd", bytes);
        let result = self.launch_maybe_blob(
            "fused_qkv_mq3g256_lloyd", [total, 1, 1], [32, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq); b.push_ptr(ak); b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq); b.push_ptr(yk); b.push_ptr(yv);
                b.push_i32(q_m_i); b.push_i32(k_m_i); b.push_i32(v_m_i);
                b.push_i32(k_i);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Lazily initialize MagnumQuant FWHT sign tables (256 floats each, seeds 42 and 1042).
    pub fn ensure_mq_signs(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        if self.mq_signs1.is_some() { return Ok(()); }
        fn gen_signs(seed: u32) -> Vec<f32> {
            let mut state = seed;
            (0..256).map(|_| {
                state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
                if (state >> 16) & 1 == 1 { 1.0f32 } else { -1.0f32 }
            }).collect()
        }
        let s1 = gen_signs(42);
        let s2 = gen_signs(1042);
        let s1b: Vec<u8> = s1.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let s2b: Vec<u8> = s2.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let s1t = self.alloc_tensor(&[256], DType::F32)?;
        let s2t = self.alloc_tensor(&[256], DType::F32)?;
        self.hip.memcpy_htod(&s1t.buf, &s1b)?;
        self.hip.memcpy_htod(&s2t.buf, &s2b)?;
        // Allocate scratch buffers — 32K elements covers K up to 32768
        let x_rot = self.alloc_tensor(&[32768], DType::F32)?;
        let x_q8 = self.hip.malloc(32768)?;  // INT8 buffer for dp4a
        let x_scales = self.hip.malloc(128 * 4)?; // up to 128 groups × f32
        self.mq_signs1 = Some(s1t);
        self.mq_signs2 = Some(s2t);
        self.mq_x_rot = Some(x_rot);
        self.mq_x_q8 = Some(x_q8);
        self.mq_x_scales = Some(x_scales);
        Ok(())
    }

    /// MagnumQuant GEMV: FWHT-rotated HFQ4-G256. Rotates x per group via ds_swizzle,
    /// then standard 4-bit dot product. signs1/signs2 are the FWHT sign tables (256 floats each).
    pub fn gemv_mq4g256(
        &mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor,
        signs1: &GpuTensor, signs2: &GpuTensor,
        m: usize, k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_mq4g256", kernels::GEMV_MQ4G256_SRC, "gemv_mq4g256")?;
        let func = &self.functions["gemv_mq4g256"];
        let mut a_ptr = a_raw.buf.as_ptr(); let mut x_ptr = x.buf.as_ptr(); let mut y_ptr = y.buf.as_ptr();
        let mut s1_ptr = signs1.buf.as_ptr(); let mut s2_ptr = signs2.buf.as_ptr();
        let mut m_val = m as i32; let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void, &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut s1_ptr as *mut _ as *mut c_void, &mut s2_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void, &mut k_val as *mut _ as *mut c_void,
        ];
        // LDS for rotated x: 256 floats = 1024 bytes
        unsafe { self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 1024, self.stream_ref(), &mut params) }
    }

    /// HFP4-G32 GEMV — RDNA-optimal FP4 (E2M1 + UE8M0 g32 + FP16 row scale).
    ///
    /// v1 correctness anchor: no WMMA, no FP8, no rotation. K must be a multiple of 256
    /// (the kernel's 4-accumulator + tail-by-g%4 outer loop assumes the 256-element
    /// "iter window" stride; v2 will lift this to k%32==0). See `kernels/src/gemv_hfp4g32.hip`
    /// and `docs/quant-formats/hfp4.md`.
    pub fn gemv_hfp4g32(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        assert!(k % 256 == 0, "gemv_hfp4g32 requires K%256==0 in v1, got K={}", k);
        self.bind_thread()?;
        let (src, module) = kernels::gemv_hfp4g32_for_arch(&self.arch);
        self.ensure_kernel(module, src, "gemv_hfp4g32")?;
        let func = &self.functions["gemv_hfp4g32"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        // LDS: 16-entry FP16 LUT = 32 bytes.
        unsafe { self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 32, self.stream_ref(), &mut params) }
    }

    /// Fused RMSNorm + MagnumQuant FWHT rotation. Replaces the
    /// `rmsnorm_f32` + `rotate_x_mq` sequence with a single kernel launch.
    /// Reads unnormalized `x` + rmsnorm `weight`, computes rmsnorm in LDS,
    /// applies the same per-256-element FWHT as `mq_rotate_x`, and writes
    /// the rotated normalized vector into `x_rot`.
    ///
    /// Preconditions:
    /// - `k` is a multiple of 256 (enforced by callers via `config.dim`)
    /// - `k` ≤ 16384 (LDS ceiling; 16K floats = 64KB minus reduce buffer)
    pub fn fused_rmsnorm_rotate_mq(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "fused_rmsnorm_mq_rotate",
            kernels::FUSED_RMSNORM_MQ_ROTATE_SRC,
            "fused_rmsnorm_mq_rotate",
        )?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();

        let xp = x.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let s1 = s1_ptr;
        let s2 = s2_ptr;
        let kv = k as i32;
        let eps_v = eps;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
            &eps_v as *const _ as *mut c_void,
        ];

        let block_size = 256u32;
        // Dynamic LDS: K floats for x_shared + 256 floats for reduce buffer.
        let shared_mem = ((k + 256) * 4) as u32;

        // Bandwidth: read x (K*4) + weight (K*4) + signs (2*256*4) + write x_rot (K*4)
        let bytes = k * 4 * 3 + 2 * 256 * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fused",
            "fused_rmsnorm_mq_rotate",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "fused_rmsnorm_mq_rotate", [1, 1, 1], [block_size, 1, 1], shared_mem, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp); b.push_ptr(wp);
                b.push_ptr(s1); b.push_ptr(s2); b.push_ptr(xrp);
                b.push_i32(kv); b.push_f32(eps_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Batched `fused_rmsnorm_rotate_mq`. Grid.x is the batch dim — processes
    /// N tokens' [N × K] x into [N × K] x_rot in a single launch. Byte-exact
    /// against calling `fused_rmsnorm_rotate_mq` N times on separate x/x_rot
    /// buffers. Weight/signs are shared across the batch.
    pub fn fused_rmsnorm_rotate_mq_batched(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        eps: f32,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "fused_rmsnorm_mq_rotate",
            kernels::FUSED_RMSNORM_MQ_ROTATE_SRC,
            "fused_rmsnorm_mq_rotate",
        )?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();

        let mut xp = x.buf.as_ptr();
        let mut wp = weight.buf.as_ptr();
        let mut xrp = x_rot.buf.as_ptr();
        let mut s1 = s1_ptr;
        let mut s2 = s2_ptr;
        let mut kv = k as i32;
        let mut eps_v = eps;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut s1 as *mut _ as *mut c_void,
            &mut s2 as *mut _ as *mut c_void,
            &mut xrp as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
            &mut eps_v as *mut _ as *mut c_void,
        ];
        let block_size = 256u32;
        let shared_mem = ((k + 256) * 4) as u32;
        let bytes = (k * 4 * 3 + 2 * 256 * 4) * batch_size;
        let timer = crate::profile::begin_timer(&self.hip, "fused", "fused_rmsnorm_mq_rotate_batched", bytes);
        let result = self.launch_maybe_blob(
            "fused_rmsnorm_mq_rotate",
            [batch_size as u32, 1, 1],
            [block_size, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp); b.push_ptr(wp);
                b.push_ptr(s1); b.push_ptr(s2);
                b.push_ptr(xrp);
                b.push_i32(kv); b.push_f32(eps_v);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Fused SwiGLU + FWHT rotation. Reads gate/up, computes
    /// silu(gate[k])*up[k] on the fly, applies FWHT rotation, writes x_rot.
    /// Used as the w_down input stage for MQ4 — replaces the pair
    /// silu_mul_f32 + mq_rotate_x with one launch.
    pub fn fused_silu_mul_rotate_mq(
        &mut self,
        gate: &GpuTensor,
        up: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "fused_silu_mul_mq_rotate",
            kernels::FUSED_SILU_MUL_MQ_ROTATE_SRC,
            "fused_silu_mul_mq_rotate",
        )?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;
        let gp = gate.buf.as_ptr();
        let up_p = up.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let s1 = s1_ptr;
        let s2 = s2_ptr;
        let kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &gp as *const _ as *mut c_void,
            &up_p as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];
        // Bandwidth: read gate + up, 2x256 signs, write x_rot.
        let bytes = k * 4 * 3 + 2 * 256 * 4;
        let timer = crate::profile::begin_timer(&self.hip, "fused", "fused_silu_mul_mq_rotate", bytes);
        let result = self.launch_maybe_blob(
            "fused_silu_mul_mq_rotate", [n_groups, 1, 1], [32, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(gp); b.push_ptr(up_p);
                b.push_ptr(s1); b.push_ptr(s2); b.push_ptr(xrp);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched `fused_silu_mul_rotate_mq`. Grid.y is the batch dim — processes
    /// N tokens' [N × K] gate/up/x_rot in a single launch.
    pub fn fused_silu_mul_rotate_mq_batched(
        &mut self,
        gate: &GpuTensor,
        up: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "fused_silu_mul_mq_rotate",
            kernels::FUSED_SILU_MUL_MQ_ROTATE_SRC,
            "fused_silu_mul_mq_rotate",
        )?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;
        let mut gp = gate.buf.as_ptr();
        let mut up_p = up.buf.as_ptr();
        let mut xrp = x_rot.buf.as_ptr();
        let mut s1 = s1_ptr;
        let mut s2 = s2_ptr;
        let mut kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut gp as *mut _ as *mut c_void,
            &mut up_p as *mut _ as *mut c_void,
            &mut s1 as *mut _ as *mut c_void,
            &mut s2 as *mut _ as *mut c_void,
            &mut xrp as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
        ];
        let bytes = (k * 4 * 3 + 2 * 256 * 4) * batch_size;
        let timer = crate::profile::begin_timer(&self.hip, "fused", "fused_silu_mul_mq_rotate_batched", bytes);
        let result = self.launch_maybe_blob(
            "fused_silu_mul_mq_rotate",
            [n_groups, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(gp); b.push_ptr(up_p);
                b.push_ptr(s1); b.push_ptr(s2); b.push_ptr(xrp);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Standalone FWHT rotation for MagnumQuant (MQ4). Writes K floats into x_rot.
    /// Exposed so callers can batch one rotation across multiple GEMVs that share x
    /// (e.g., Q/K/V projections all consume the same post-RMSNorm x).
    pub fn rotate_x_mq(&mut self, x: &GpuTensor, x_rot: &GpuTensor, k: usize) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        // `mq_rotate_x` lives inside the `gemv_mq4g256` module — precompile
        // writes the .hsaco/.hash sidecar under that module name, so the
        // runtime cache key here MUST match or we silently JIT on first use.
        self.ensure_kernel("gemv_mq4g256", kernels::GEMV_MQ4G256_SRC, "mq_rotate_x")?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;
        let xp = x.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void, &xrp as *const _ as *mut c_void,
            &s1_ptr as *const _ as *mut c_void, &s2_ptr as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::mq_rotate_bytes(k);
        let timer = crate::profile::begin_timer(&self.hip, "fwht", "mq_rotate_x", bytes);
        let result = self.launch_maybe_blob(
            "mq_rotate_x",
            [n_groups, 1, 1], [32, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp); b.push_ptr(xrp);
                b.push_ptr(s1_ptr); b.push_ptr(s2_ptr);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched `rotate_x_mq`. Grid.y is the batch dim.
    pub fn rotate_x_mq_batched(
        &mut self,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        // Same cache-key contract as `rotate_x_mq` — see comment there.
        self.ensure_kernel("gemv_mq4g256", kernels::GEMV_MQ4G256_SRC, "mq_rotate_x")?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;
        let mut xp = x.buf.as_ptr();
        let mut xrp = x_rot.buf.as_ptr();
        let mut s1 = s1_ptr;
        let mut s2 = s2_ptr;
        let mut kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut xrp as *mut _ as *mut c_void,
            &mut s1 as *mut _ as *mut c_void,
            &mut s2 as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
        ];
        let bytes = crate::profile::mq_rotate_bytes(k) * batch_size;
        let timer = crate::profile::begin_timer(&self.hip, "fwht", "mq_rotate_x_batched", bytes);
        let result = self.launch_maybe_blob(
            "mq_rotate_x",
            [n_groups, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp); b.push_ptr(xrp);
                b.push_ptr(s1); b.push_ptr(s2);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// MagnumQuant MQ4: rotate x once, then GEMV against rotated x.
    /// MQ4 weights are stored in HFQ4-G256 format with FWHT pre-applied, so the GEMV
    /// inner loop is identical to standard HFQ4 — we reuse the arch-tuned HFQ4 kernel.
    pub fn gemv_mq4g256_with_rotate(
        &mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor,
        x_rot: &GpuTensor, m: usize, k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.rotate_x_mq(x, x_rot, k)?;
        // MQ4 = FWHT-rotated HFQ4-G256. dot(rot(W), rot(x)) = dot(W, x).
        // Route through the arch-specific HFQ4 kernel (4x unroll on gfx1100, etc).
        self.gemv_hfq4g256(a_raw, x_rot, y, m, k)
    }

    /// MagnumQuant MQ4 with pre-rotated x. Skips the rotation step entirely —
    /// caller must have called `rotate_x_mq` into `x_rot` first.
    pub fn gemv_mq4g256_prerotated(
        &mut self, a_raw: &GpuTensor, x_rot: &GpuTensor, y: &GpuTensor, m: usize, k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.gemv_hfq4g256(a_raw, x_rot, y, m, k)
    }

    /// MagnumQuant MQ3: rotate x once, then HFQ3-G256 GEMV against rotated x.
    /// MQ3 weights are stored in HFQ3-G256 format (104 B/group) with FWHT pre-applied,
    /// so the GEMV inner loop is identical to standard HFQ3.
    pub fn gemv_mq3g256_with_rotate(
        &mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor,
        x_rot: &GpuTensor, m: usize, k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.rotate_x_mq(x, x_rot, k)?;
        self.gemv_hfq3g256(a_raw, x_rot, y, m, k)
    }

    /// MagnumQuant MQ3 with pre-rotated x.
    pub fn gemv_mq3g256_prerotated(
        &mut self, a_raw: &GpuTensor, x_rot: &GpuTensor, y: &GpuTensor, m: usize, k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.gemv_hfq3g256(a_raw, x_rot, y, m, k)
    }

    /// MagnumQuant MQ2: rotate x once, then HFQ2-G256 GEMV against rotated x.
    /// MQ2 weights are stored in HFQ2-G256 format (72 B/group) with FWHT pre-applied.
    pub fn gemv_mq2g256_with_rotate(
        &mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor,
        x_rot: &GpuTensor, m: usize, k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.rotate_x_mq(x, x_rot, k)?;
        self.gemv_hfq2g256(a_raw, x_rot, y, m, k)
    }

    /// MagnumQuant MQ2 with pre-rotated x.
    pub fn gemv_mq2g256_prerotated(
        &mut self, a_raw: &GpuTensor, x_rot: &GpuTensor, y: &GpuTensor, m: usize, k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.gemv_hfq2g256(a_raw, x_rot, y, m, k)
    }

    /// MagnumQuant MQ6: rotate x via FWHT, then HFQ6 GEMV.
    pub fn gemv_mq6g256_with_rotate(
        &mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor,
        x_rot: &GpuTensor, m: usize, k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.rotate_x_mq(x, x_rot, k)?;
        self.gemv_hfq6g256(a_raw, x_rot, y, m, k)
    }

    /// MagnumQuant MQ6 with pre-rotated x.
    pub fn gemv_mq6g256_prerotated(
        &mut self, a_raw: &GpuTensor, x_rot: &GpuTensor, y: &GpuTensor, m: usize, k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.gemv_hfq6g256(a_raw, x_rot, y, m, k)
    }

    /// Standalone MQ8 rotate + INT8 quantize of x into internal `mq_x_q8`/`mq_x_scales`.
    /// After this, `gemv_mq8g256_prerotated` can be called multiple times with the same x.
    pub fn rotate_quantize_x_mq8(&mut self, x: &GpuTensor, k: usize) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel("mq8_rotate_quantize_x", kernels::GEMV_MQ8G256_SRC, "mq8_rotate_quantize_x")?;

        let xq_ptr = self.mq_x_q8.as_ref().unwrap().as_ptr();
        let xs_ptr = self.mq_x_scales.as_ref().unwrap().as_ptr();
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;

        let rot_func = &self.functions["mq8_rotate_quantize_x"];
        let mut xp = x.buf.as_ptr();
        let mut xq = xq_ptr; let mut xs = xs_ptr;
        let mut s1 = s1_ptr; let mut s2 = s2_ptr;
        let mut kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void, &mut xq as *mut _ as *mut c_void,
            &mut xs as *mut _ as *mut c_void,
            &mut s1 as *mut _ as *mut c_void, &mut s2 as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(rot_func, [n_groups, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// MQ8 dp4a GEMV using pre-rotated+quantized x. Caller must have called
    /// `rotate_quantize_x_mq8(x, k)` first — results use the internal `mq_x_q8`/`mq_x_scales`.
    pub fn gemv_mq8g256_prerotated(
        &mut self, a_raw: &GpuTensor, y: &GpuTensor, m: usize, k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_mq8g256", kernels::GEMV_MQ8G256_SRC, "gemv_mq8g256")?;

        let xq_ptr = self.mq_x_q8.as_ref().unwrap().as_ptr();
        let xs_ptr = self.mq_x_scales.as_ref().unwrap().as_ptr();

        let func = &self.functions["gemv_mq8g256"];
        let mut ap = a_raw.buf.as_ptr();
        let mut xq = xq_ptr; let mut xs = xs_ptr;
        let mut yp = y.buf.as_ptr();
        let mut mv = m as i32; let mut kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void, &mut xq as *mut _ as *mut c_void,
            &mut xs as *mut _ as *mut c_void, &mut yp as *mut _ as *mut c_void,
            &mut mv as *mut _ as *mut c_void, &mut kv as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// MagnumQuant MQ8: FWHT rotate + INT8 quantize x, then dp4a GEMV.
    pub fn gemv_mq8g256_with_rotate(
        &mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.rotate_quantize_x_mq8(x, k)?;
        self.gemv_mq8g256_prerotated(a_raw, y, m, k)
    }

    /// HFQ3-G256 GEMV. K must be multiple of 256.
    /// Per-arch dispatch: gfx1100/1101/1102 uses the K4-unrolled
    /// 4-accumulator variant. The default kernel was re-ported to match
    /// the same ordering so non-RDNA3 archs (gfx1010, gfx1030, gfx12,
    /// gfx9xx) produce byte-exact results against the RDNA3 baseline.
    /// Uses `launch_maybe_blob` for HIPFIRE_GRAPH=1 capture safety.
    pub fn gemv_hfq3g256(&mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemv_hfq3g256_for_arch(&self.arch);
        self.ensure_kernel(module, src, "gemv_hfq3g256")?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "gemv_hfq3g256", [m as u32, 1, 1], [32, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr); b.push_ptr(x_ptr); b.push_ptr(y_ptr);
                b.push_i32(m_val); b.push_i32(k_val);
                b
            },
        )
    }

    /// HFQ3-G256 GEMV with fused residual add: y[row] += A[row] dot x.
    /// Used by `weight_gemv_residual` MQ3 arm to eliminate the
    /// alloc+gemv+add+free fallback chain (saves ~3 launches per residual).
    /// gfx1100 selects the K4-unrolled chip-specific variant (commit 0003103,
    /// 9B MQ3 decode 114 to 141 tok/s); other archs use the K4-ported default
    /// (re-port in 9fdba4d keeps non-RDNA3 archs byte-exact with the prior
    /// gemv + add_inplace path). Uses launch_maybe_blob for HIPFIRE_GRAPH=1
    /// capture safety.
    pub fn gemv_hfq3g256_residual(
        &mut self,
        a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor,
        m: usize, k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemv_hfq3g256_residual_for_arch(&self.arch);
        self.ensure_kernel(module, src, "gemv_hfq3g256_residual")?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "gemv_hfq3g256_residual", [m as u32, 1, 1], [32, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr); b.push_ptr(x_ptr); b.push_ptr(y_ptr);
                b.push_i32(m_val); b.push_i32(k_val);
                b
            },
        )
    }

    /// MagnumQuant MQ3-G256 GEMV with fused residual add. The pre-rotation
    /// happens in a separate kernel via fused_silu_mul_mq_rotate or
    /// rotate_x_for_mq; this function just dispatches the underlying
    /// hfq3g256_residual against the already-rotated x.
    pub fn gemv_mq3g256_residual_prerotated(
        &mut self, a_raw: &GpuTensor, x_rot: &GpuTensor, y: &GpuTensor, m: usize, k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.gemv_hfq3g256_residual(a_raw, x_rot, y, m, k)
    }

    /// HFQ3-G128 GEMV. K must be multiple of 128. Finer granularity than G256.
    pub fn gemv_hfq3g128(&mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_hfq3g128", kernels::GEMV_HFQ3G128_SRC, "gemv_hfq3g128")?;
        let func = &self.functions["gemv_hfq3g128"];
        let mut a_ptr = a_raw.buf.as_ptr(); let mut x_ptr = x.buf.as_ptr(); let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32; let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void, &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void, &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// HFQ2-G128 GEMV. K must be multiple of 128. Finer granularity than G256.
    pub fn gemv_hfq2g128(&mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_hfq2g128", kernels::GEMV_HFQ2G128_SRC, "gemv_hfq2g128")?;
        let func = &self.functions["gemv_hfq2g128"];
        let mut ap = a_raw.buf.as_ptr(); let mut xp = x.buf.as_ptr(); let mut yp = y.buf.as_ptr();
        let mut mv = m as i32; let mut kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void, &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void, &mut mv as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// HFQ6-G256 GEMV with fused residual add: y[row] += A[row] . x.
    /// Same shape as gemv_hfq6g256; only the final write differs (+= vs =).
    /// Used for wo and w_down in HFQ6 / MQ6 forward paths so the
    /// add_inplace_f32 follow-up launch can be elided.
    pub fn gemv_hfq6g256_residual(&mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize) -> HipResult<()> {
        self.bind_thread()?;
        let mut a_ptr = a_raw.buf.as_ptr(); let mut x_ptr = x.buf.as_ptr(); let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32; let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void, &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void, &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        // Wave64-native fast path (gfx906/908/94x): 2 rows per block, halves
        // grid.x. Mirrors the HFQ4 sibling at line ~5378. Plan §3.1.1 item 2
        // (gfx906-mq6-mq8-port.md v3.2.1 + v3.2.2). Byte-exact with the
        // wave32 base since each warp's 32-lane reduction stays in-warp.
        // ILP-prefetch variant gates on gemv_prefetch_enabled(arch) — default
        // on for gfx906 (Phase A.1b, mirror of HFQ4 +4.8% lever from `3ef127d`).
        if has_wave64_native(&self.arch) {
            let (kname, ksrc): (&str, &str) = if gemv_prefetch_enabled(&self.arch) {
                (
                    "gemv_hfq6g256_residual_wave64_prefetch",
                    kernels::GEMV_HFQ6G256_RESIDUAL_WAVE64_PREFETCH_SRC,
                )
            } else {
                (
                    "gemv_hfq6g256_residual_wave64",
                    kernels::GEMV_HFQ6G256_RESIDUAL_WAVE64_SRC,
                )
            };
            self.ensure_kernel(kname, ksrc, kname)?;
            let func = &self.functions[kname];
            let grid = ((m as u32) + 1) / 2;
            return unsafe { self.hip.launch_kernel(func, [grid, 1, 1], [64, 1, 1], 0, self.stream_ref(), &mut params) };
        }

        self.ensure_kernel("gemv_hfq6g256_residual", kernels::GEMV_HFQ6G256_RESIDUAL_SRC, "gemv_hfq6g256_residual")?;
        let func = &self.functions["gemv_hfq6g256_residual"];
        unsafe { self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// HFQ6-G256 GEMV. K must be multiple of 256.
    pub fn gemv_hfq6g256(&mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_hfq6g256", kernels::GEMV_HFQ6G256_SRC, "gemv_hfq6g256")?;
        let func = &self.functions["gemv_hfq6g256"];
        let mut a_ptr = a_raw.buf.as_ptr(); let mut x_ptr = x.buf.as_ptr(); let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32; let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void, &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void, &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// HFQ8-G256 GEMV. K must be multiple of 256.
    pub fn gemv_hfq8g256(&mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_hfq8g256", kernels::GEMV_HFQ8G256_SRC, "gemv_hfq8g256")?;
        let func = &self.functions["gemv_hfq8g256"];
        let mut a_ptr = a_raw.buf.as_ptr(); let mut x_ptr = x.buf.as_ptr(); let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32; let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void, &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void, &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// HFQ4-G512 GEMV. K must be multiple of 512.
    pub fn gemv_hfq4g512(&mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_hfq4g512", kernels::GEMV_HFQ4G512_SRC, "gemv_hfq4g512")?;
        let func = &self.functions["gemv_hfq4g512"];
        let mut a_ptr = a_raw.buf.as_ptr(); let mut x_ptr = x.buf.as_ptr(); let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32; let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void, &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void, &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// HFQ4-G1024 GEMV. K must be multiple of 1024.
    pub fn gemv_hfq4g1024(&mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_hfq4g1024", kernels::GEMV_HFQ4G1024_SRC, "gemv_hfq4g1024")?;
        let func = &self.functions["gemv_hfq4g1024"];
        let mut a_ptr = a_raw.buf.as_ptr(); let mut x_ptr = x.buf.as_ptr(); let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32; let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void, &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void, &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// HFQ4-G256 GEMV: flat 4-bit with 256-weight groups. K must be multiple of 256.
    pub fn gemv_hfq4g256(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (hfq4g256_src, hfq4g256_module) = kernels::gemv_hfq4g256_for_arch(&self.arch);
        self.ensure_kernel(hfq4g256_module, hfq4g256_src, "gemv_hfq4g256")?;

        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(a_ptr); b.push_ptr(x_ptr); b.push_ptr(y_ptr);
            b.push_i32(m_val); b.push_i32(k_val);
            b
        };

        // Multi-row GEMV: one warp computes R output rows, sharing x register
        // state across rows. Per-arch default picks R=1 on RDNA3 (negative)
        // and RDNA2 (has its own arch-specific narrow path), R=2 on the
        // default gfx1010-baseline path (gfx1010, gfx1013 Cyan Skillfish,
        // etc.). Override any arch with HIPFIRE_GEMV_ROWS ∈ {1, 2, 4, 8}.
        //
        // See gemv_rows_default() for the measurement data that motivates
        // the per-arch defaults.
        let rdna3 = matches!(self.arch.as_str(), "gfx1100" | "gfx1101" | "gfx1102");
        let rows = gemv_rows_override().unwrap_or_else(|| gemv_rows_default(self.arch.as_str()));
        let use_multirow = rows > 1;

        // RDNA2 (gfx1030/1031): always use the arch-optimized narrow kernel.
        // Other non-RDNA3 archs: use wide kernel (2 rows/block) for large M.
        let use_wide = !use_multirow
            && m >= 64
            && !matches!(self.arch.as_str(), "gfx1030" | "gfx1031" | "gfx1100" | "gfx1101" | "gfx1102");

        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", "gemv_hfq4g256", bytes);
        let result = if use_multirow {
            let (func_name, grid_div) = match rows {
                2 => ("gemv_hfq4g256_multirow_r2", 2u32),
                4 => ("gemv_hfq4g256_multirow_r4", 4u32),
                8 => ("gemv_hfq4g256_multirow_r8", 8u32),
                _ => unreachable!(),
            };
            let (mr_name, mr_src) = if rdna3 {
                ("gemv_hfq4g256_multirow_rdna3", kernels::GEMV_HFQ4G256_MULTIROW_GFX1100_SRC)
            } else {
                ("gemv_hfq4g256_multirow_default", kernels::GEMV_HFQ4G256_MULTIROW_SRC)
            };
            self.ensure_kernel(mr_name, mr_src, func_name)?;
            let grid = ((m as u32) + grid_div - 1) / grid_div;
            self.launch_maybe_blob(
                func_name,
                [grid, 1, 1], [32, 1, 1], 0, &mut params,
                blob_builder,
            )
        } else if use_wide {
            self.ensure_kernel("gemv_hfq4g256_wide", kernels::GEMV_HFQ4G256_WIDE_SRC, "gemv_hfq4g256_wide")?;
            let grid = ((m + 1) / 2) as u32;
            self.launch_maybe_blob(
                "gemv_hfq4g256_wide",
                [grid, 1, 1], [64, 1, 1], 0, &mut params,
                blob_builder,
            )
        } else {
            self.launch_maybe_blob(
                "gemv_hfq4g256",
                [m as u32, 1, 1], [32, 1, 1], 0, &mut params,
                blob_builder,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// dp4a-port of fused_qkv_hfq4g256 for gfx906. Pre-quantizes x to
    /// Q8_1 via the shared MMQ scratch, then runs the dp4a-based GEMV.
    /// Math is identical modulo Q8_1 quant noise. Targets gfx906's
    /// memory-bound regime per the per-kernel PMC pass at 2026-05-05.
    pub fn fused_qkv_hfq4g256_dp4a(
        &mut self,
        a_q: &GpuTensor, a_k: &GpuTensor, a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor, y_k: &GpuTensor, y_v: &GpuTensor,
        q_m: usize, k_m: usize, v_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, 1, k)?;

        self.ensure_kernel(
            "fused_qkv_hfq4g256_wave64_dp4a",
            kernels::FUSED_QKV_HFQ4G256_WAVE64_DP4A_SRC,
            "fused_qkv_hfq4g256_wave64_dp4a",
        )?;

        let aq = a_q.buf.as_ptr();
        let ak = a_k.buf.as_ptr();
        let av = a_v.buf.as_ptr();
        let yq = y_q.buf.as_ptr();
        let yk = y_k.buf.as_ptr();
        let yv = y_v.buf.as_ptr();
        let q_m_val = q_m as i32;
        let k_m_val = k_m as i32;
        let v_m_val = v_m as i32;
        let k_val = k as i32;
        let total = (q_m + k_m + v_m) as u32;
        let mut xq = xq_ptr;

        let mut params: Vec<*mut c_void> = vec![
            &aq as *const _ as *mut c_void,
            &ak as *const _ as *mut c_void,
            &av as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &yq as *const _ as *mut c_void,
            &yk as *const _ as *mut c_void,
            &yv as *const _ as *mut c_void,
            &q_m_val as *const _ as *mut c_void,
            &k_m_val as *const _ as *mut c_void,
            &v_m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(k_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(v_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "fused", "fused_qkv_hfq4g256_dp4a", bytes);
        let result = self.launch_maybe_blob(
            "fused_qkv_hfq4g256_wave64_dp4a",
            [(total + 1) / 2, 1, 1], [64, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq); b.push_ptr(ak); b.push_ptr(av); b.push_ptr(xq);
                b.push_ptr(yq); b.push_ptr(yk); b.push_ptr(yv);
                b.push_i32(q_m_val); b.push_i32(k_m_val);
                b.push_i32(v_m_val); b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    // HFQ2 GEMV dispatch already exists at line ~521 from the HFQ family

    /// gfx906 dp4a-port — see fused_gate_up_hfq6g256_wave64_dp4a.hip for
    /// the math derivation. Plan §3.1.1 item 3 / v3.2.2 §5.1 item 1c.
    pub fn fused_qkv_hfq6g256_dp4a(
        &mut self,
        a_q: &GpuTensor, a_k: &GpuTensor, a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor, y_k: &GpuTensor, y_v: &GpuTensor,
        q_m: usize, k_m: usize, v_m: usize,
        k: usize,
    ) -> HipResult<()> {
        let xq_ptr = self.ensure_q8_1_mmq_x(x, 1, k)?;

        self.ensure_kernel(
            "fused_qkv_hfq6g256_wave64_dp4a",
            kernels::FUSED_QKV_HFQ6G256_WAVE64_DP4A_SRC,
            "fused_qkv_hfq6g256_wave64_dp4a",
        )?;

        let aq = a_q.buf.as_ptr();
        let ak = a_k.buf.as_ptr();
        let av = a_v.buf.as_ptr();
        let yq = y_q.buf.as_ptr();
        let yk = y_k.buf.as_ptr();
        let yv = y_v.buf.as_ptr();
        let q_m_val = q_m as i32;
        let k_m_val = k_m as i32;
        let v_m_val = v_m as i32;
        let k_val = k as i32;
        let total = (q_m + k_m + v_m) as u32;
        let mut xq = xq_ptr;

        let mut params: Vec<*mut c_void> = vec![
            &aq as *const _ as *mut c_void,
            &ak as *const _ as *mut c_void,
            &av as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &yq as *const _ as *mut c_void,
            &yk as *const _ as *mut c_void,
            &yv as *const _ as *mut c_void,
            &q_m_val as *const _ as *mut c_void,
            &k_m_val as *const _ as *mut c_void,
            &v_m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        self.launch_maybe_blob(
            "fused_qkv_hfq6g256_wave64_dp4a",
            [(total + 1) / 2, 1, 1], [64, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq); b.push_ptr(ak); b.push_ptr(av); b.push_ptr(xq);
                b.push_ptr(yq); b.push_ptr(yk); b.push_ptr(yv);
                b.push_i32(q_m_val); b.push_i32(k_m_val);
                b.push_i32(v_m_val); b.push_i32(k_val);
                b
            },
        )
    }

    /// gfx906 dp4a-port — 4-output deltanet QKV preamble.
    pub fn fused_qkvza_hfq6g256_dp4a(
        &mut self,
        a_qkv: &GpuTensor, a_z: &GpuTensor, a_beta: &GpuTensor, a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor, y_z: &GpuTensor, y_beta: &GpuTensor, y_alpha: &GpuTensor,
        qkv_m: usize, z_m: usize, beta_m: usize, alpha_m: usize,
        k: usize,
    ) -> HipResult<()> {
        let xq_ptr = self.ensure_q8_1_mmq_x(x, 1, k)?;

        self.ensure_kernel(
            "fused_qkvza_hfq6g256_wave64_dp4a",
            kernels::FUSED_QKVZA_HFQ6G256_WAVE64_DP4A_SRC,
            "fused_qkvza_hfq6g256_wave64_dp4a",
        )?;

        let aqkv = a_qkv.buf.as_ptr();
        let az = a_z.buf.as_ptr();
        let ab = a_beta.buf.as_ptr();
        let aa = a_alpha.buf.as_ptr();
        let yqkv = y_qkv.buf.as_ptr();
        let yz = y_z.buf.as_ptr();
        let yb = y_beta.buf.as_ptr();
        let ya = y_alpha.buf.as_ptr();
        let qkv_m_val = qkv_m as i32;
        let z_m_val = z_m as i32;
        let beta_m_val = beta_m as i32;
        let alpha_m_val = alpha_m as i32;
        let k_val = k as i32;
        let total = (qkv_m + z_m + beta_m + alpha_m) as u32;
        let mut xq = xq_ptr;

        let mut params: Vec<*mut c_void> = vec![
            &aqkv as *const _ as *mut c_void,
            &az as *const _ as *mut c_void,
            &ab as *const _ as *mut c_void,
            &aa as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &yqkv as *const _ as *mut c_void,
            &yz as *const _ as *mut c_void,
            &yb as *const _ as *mut c_void,
            &ya as *const _ as *mut c_void,
            &qkv_m_val as *const _ as *mut c_void,
            &z_m_val as *const _ as *mut c_void,
            &beta_m_val as *const _ as *mut c_void,
            &alpha_m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        self.launch_maybe_blob(
            "fused_qkvza_hfq6g256_wave64_dp4a",
            [(total + 1) / 2, 1, 1], [64, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aqkv); b.push_ptr(az); b.push_ptr(ab); b.push_ptr(aa);
                b.push_ptr(xq);
                b.push_ptr(yqkv); b.push_ptr(yz); b.push_ptr(yb); b.push_ptr(ya);
                b.push_i32(qkv_m_val); b.push_i32(z_m_val);
                b.push_i32(beta_m_val); b.push_i32(alpha_m_val);
                b.push_i32(k_val);
                b
            },
        )
    }

    /// gfx906 dp4a-port — 2-output FFN gate+up projection.
    pub fn fused_gate_up_hfq6g256_dp4a(
        &mut self,
        a_gate: &GpuTensor, a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor, y_up: &GpuTensor,
        gate_m: usize, up_m: usize,
        k: usize,
    ) -> HipResult<()> {
        let xq_ptr = self.ensure_q8_1_mmq_x(x, 1, k)?;

        self.ensure_kernel(
            "fused_gate_up_hfq6g256_wave64_dp4a",
            kernels::FUSED_GATE_UP_HFQ6G256_WAVE64_DP4A_SRC,
            "fused_gate_up_hfq6g256_wave64_dp4a",
        )?;

        let agate = a_gate.buf.as_ptr();
        let aup = a_up.buf.as_ptr();
        let ygate = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let gate_m_val = gate_m as i32;
        let up_m_val = up_m as i32;
        let k_val = k as i32;
        let total = (gate_m + up_m) as u32;
        let mut xq = xq_ptr;

        let mut params: Vec<*mut c_void> = vec![
            &agate as *const _ as *mut c_void,
            &aup as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &ygate as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &gate_m_val as *const _ as *mut c_void,
            &up_m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        self.launch_maybe_blob(
            "fused_gate_up_hfq6g256_wave64_dp4a",
            [(total + 1) / 2, 1, 1], [64, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(agate); b.push_ptr(aup); b.push_ptr(xq);
                b.push_ptr(ygate); b.push_ptr(yup);
                b.push_i32(gate_m_val); b.push_i32(up_m_val);
                b.push_i32(k_val);
                b
            },
        )
    }

    /// 3-way fused HFQ4-G256 projection — cross-arch.
    ///
    /// Performs y_q=A_q·x, y_k=A_k·x, y_v=A_v·x in a single kernel launch
    /// for the Qwen3.5 FullAttention layer preamble. Same rationale and
    /// tail-handling guarantees as `fused_qkvza_hfq4g256`.
    pub fn fused_qkv_hfq4g256(
        &mut self,
        a_q: &GpuTensor, a_k: &GpuTensor, a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor, y_k: &GpuTensor, y_v: &GpuTensor,
        q_m: usize, k_m: usize, v_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if gemv_dp4a_enabled(&self.arch) {
            return self.fused_qkv_hfq4g256_dp4a(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k,
            );
        }

        let cdna_wave64 = has_wave64_native(&self.arch);
        let (func_name, block, grid_x) = if cdna_wave64 {
            self.ensure_kernel(
                "fused_qkv_hfq4g256_wave64",
                kernels::FUSED_QKV_HFQ4G256_WAVE64_SRC,
                "fused_qkv_hfq4g256_wave64",
            )?;
            let total = (q_m + k_m + v_m) as u32;
            ("fused_qkv_hfq4g256_wave64", [64u32, 1, 1], (total + 1) / 2)
        } else {
            self.ensure_kernel(
                "fused_qkv_hfq4g256",
                kernels::FUSED_QKV_HFQ4G256_SRC,
                "fused_qkv_hfq4g256",
            )?;
            ("fused_qkv_hfq4g256", [32u32, 1, 1], (q_m + k_m + v_m) as u32)
        };

        let aq = a_q.buf.as_ptr();
        let ak = a_k.buf.as_ptr();
        let av = a_v.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yq = y_q.buf.as_ptr();
        let yk = y_k.buf.as_ptr();
        let yv = y_v.buf.as_ptr();
        let q_m_val = q_m as i32;
        let k_m_val = k_m as i32;
        let v_m_val = v_m as i32;
        let k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &aq as *const _ as *mut c_void,
            &ak as *const _ as *mut c_void,
            &av as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yq as *const _ as *mut c_void,
            &yk as *const _ as *mut c_void,
            &yv as *const _ as *mut c_void,
            &q_m_val as *const _ as *mut c_void,
            &k_m_val as *const _ as *mut c_void,
            &v_m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(k_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(v_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "fused", "fused_qkv_hfq4g256", bytes);
        let result = self.launch_maybe_blob(
            func_name, [grid_x, 1, 1], block, 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq); b.push_ptr(ak); b.push_ptr(av); b.push_ptr(xp);
                b.push_ptr(yq); b.push_ptr(yk); b.push_ptr(yv);
                b.push_i32(q_m_val); b.push_i32(k_m_val);
                b.push_i32(v_m_val); b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// 4-way fused HFQ4-G256 projection — cross-arch.
    ///
    /// Performs y_qkv=A_qkv·x, y_z=A_z·x, y_beta=A_beta·x, y_alpha=A_alpha·x
    /// in a single kernel launch, where all four matrices share the same
    /// input `x` and the same K. Used by the Qwen3.5 DeltaNet LA layer
    /// preamble to collapse four launches (one per projection) into one.
    /// Bit-exact with four sequential `gemv_hfq4g256` calls.
    ///
    /// Works on every RDNA generation (gfx1010 / gfx1013 / gfx1030 /
    /// gfx1100+) because the inner loop and the standalone gemv_hfq4g256
    /// inner loop were unified onto the same 4-accumulator structure
    /// after commit 5302926.
    pub fn fused_qkvza_hfq4g256(
        &mut self,
        a_qkv: &GpuTensor, a_z: &GpuTensor, a_beta: &GpuTensor, a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor, y_z: &GpuTensor, y_beta: &GpuTensor, y_alpha: &GpuTensor,
        qkv_m: usize, z_m: usize, beta_m: usize, alpha_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if gemv_dp4a_enabled(&self.arch) {
            return self.fused_qkvza_hfq4g256_dp4a(
                a_qkv, a_z, a_beta, a_alpha, x,
                y_qkv, y_z, y_beta, y_alpha,
                qkv_m, z_m, beta_m, alpha_m, k,
            );
        }
        // gfx906/gfx908/gfx94x wave64-native path:
        // 2 rows per block, halves grid count vs wave32 kernel which wastes half
        // the wave slot. This kernel uses no MFMA, just FMA + shfl_down within
        // wave64, so it is safe for Vega 20 as well as CDNA.
        let cdna_wave64 = has_wave64_native(&self.arch);
        let (func_name, block, grid_x) = if cdna_wave64 {
            self.ensure_kernel(
                "fused_qkvza_hfq4g256_wave64",
                kernels::FUSED_QKVZA_HFQ4G256_WAVE64_SRC,
                "fused_qkvza_hfq4g256_wave64",
            )?;
            let total = (qkv_m + z_m + beta_m + alpha_m) as u32;
            ("fused_qkvza_hfq4g256_wave64", [64u32, 1, 1], (total + 1) / 2)
        } else {
            self.ensure_kernel(
                "fused_qkvza_hfq4g256",
                kernels::FUSED_QKVZA_HFQ4G256_SRC,
                "fused_qkvza_hfq4g256",
            )?;
            ("fused_qkvza_hfq4g256", [32u32, 1, 1], (qkv_m + z_m + beta_m + alpha_m) as u32)
        };
        let aq = a_qkv.buf.as_ptr();
        let az = a_z.buf.as_ptr();
        let ab = a_beta.buf.as_ptr();
        let aa = a_alpha.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yq = y_qkv.buf.as_ptr();
        let yz = y_z.buf.as_ptr();
        let yb = y_beta.buf.as_ptr();
        let ya = y_alpha.buf.as_ptr();
        let q_m_i = qkv_m as i32;
        let z_m_i = z_m as i32;
        let b_m_i = beta_m as i32;
        let a_m_i = alpha_m as i32;
        let k_i = k as i32;

        let grid = [grid_x, 1, 1];

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(z_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(alpha_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "fused", "fused_qkvza_hfq4g256", bytes);

        let mut params: Vec<*mut c_void> = vec![
            &aq as *const _ as *mut c_void, &az as *const _ as *mut c_void,
            &ab as *const _ as *mut c_void, &aa as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void, &yq as *const _ as *mut c_void,
            &yz as *const _ as *mut c_void, &yb as *const _ as *mut c_void,
            &ya as *const _ as *mut c_void,
            &q_m_i as *const _ as *mut c_void, &z_m_i as *const _ as *mut c_void,
            &b_m_i as *const _ as *mut c_void, &a_m_i as *const _ as *mut c_void,
            &k_i as *const _ as *mut c_void,
        ];
        let result = self.launch_maybe_blob(
            func_name, grid, block, 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq); b.push_ptr(az); b.push_ptr(ab); b.push_ptr(aa);
                b.push_ptr(xp); b.push_ptr(yq); b.push_ptr(yz); b.push_ptr(yb); b.push_ptr(ya);
                b.push_i32(q_m_i); b.push_i32(z_m_i); b.push_i32(b_m_i); b.push_i32(a_m_i);
                b.push_i32(k_i);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// dp4a-port of fused_qkvza_hfq4g256 for gfx906. Pre-quantizes x to
    /// Q8_1 via the shared MMQ scratch, then runs the dp4a-based GEMV.
    /// Math is identical modulo Q8_1 quant noise. Targets gfx906's
    /// memory-bound regime per the per-kernel PMC pass at 2026-05-05.
    pub fn fused_qkvza_hfq4g256_dp4a(
        &mut self,
        a_qkv: &GpuTensor, a_z: &GpuTensor, a_beta: &GpuTensor, a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor, y_z: &GpuTensor, y_beta: &GpuTensor, y_alpha: &GpuTensor,
        qkv_m: usize, z_m: usize, beta_m: usize, alpha_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, 1, k)?;

        self.ensure_kernel(
            "fused_qkvza_hfq4g256_wave64_dp4a",
            kernels::FUSED_QKVZA_HFQ4G256_WAVE64_DP4A_SRC,
            "fused_qkvza_hfq4g256_wave64_dp4a",
        )?;

        let aq = a_qkv.buf.as_ptr();
        let az = a_z.buf.as_ptr();
        let ab = a_beta.buf.as_ptr();
        let aa = a_alpha.buf.as_ptr();
        let yq = y_qkv.buf.as_ptr();
        let yz = y_z.buf.as_ptr();
        let yb = y_beta.buf.as_ptr();
        let ya = y_alpha.buf.as_ptr();
        let q_m_i = qkv_m as i32;
        let z_m_i = z_m as i32;
        let b_m_i = beta_m as i32;
        let a_m_i = alpha_m as i32;
        let k_i = k as i32;
        let total = (qkv_m + z_m + beta_m + alpha_m) as u32;
        let mut xq = xq_ptr;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(z_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(alpha_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "fused", "fused_qkvza_hfq4g256_dp4a", bytes);

        let mut params: Vec<*mut c_void> = vec![
            &aq as *const _ as *mut c_void, &az as *const _ as *mut c_void,
            &ab as *const _ as *mut c_void, &aa as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &yq as *const _ as *mut c_void, &yz as *const _ as *mut c_void,
            &yb as *const _ as *mut c_void, &ya as *const _ as *mut c_void,
            &q_m_i as *const _ as *mut c_void, &z_m_i as *const _ as *mut c_void,
            &b_m_i as *const _ as *mut c_void, &a_m_i as *const _ as *mut c_void,
            &k_i as *const _ as *mut c_void,
        ];
        let result = self.launch_maybe_blob(
            "fused_qkvza_hfq4g256_wave64_dp4a",
            [(total + 1) / 2, 1, 1], [64, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq); b.push_ptr(az); b.push_ptr(ab); b.push_ptr(aa);
                b.push_ptr(xq);
                b.push_ptr(yq); b.push_ptr(yz); b.push_ptr(yb); b.push_ptr(ya);
                b.push_i32(q_m_i); b.push_i32(z_m_i); b.push_i32(b_m_i); b.push_i32(a_m_i);
                b.push_i32(k_i);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched 4-way fused HFQ4-G256 GEMM for the LA preamble.
    ///
    /// Processes N tokens × four projections (wqkv + wz + w_beta + w_alpha)
    /// in one launch. Bitwise-identical output to calling `fused_qkvza_hfq4g256`
    /// N times on the same x[b] — 4-accumulator interleave + pairwise combine
    /// are preserved per batch element.
    ///
    /// `x`: [N × K] row-major activation batch.
    /// `y_*`: [N × *_m] row-major outputs (overwrite semantics).
    pub fn gemm_qkvza_hfq4g256(
        &mut self,
        a_qkv: &GpuTensor, a_z: &GpuTensor, a_beta: &GpuTensor, a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor, y_z: &GpuTensor, y_beta: &GpuTensor, y_alpha: &GpuTensor,
        qkv_m: usize, z_m: usize, beta_m: usize, alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // CDNA3 MFMA path — 4 back-to-back rocBLAS calls. The last two
        // matrices (beta, alpha) are tiny (n_v_heads = 128 on A3B) so we
        // could skip them and stay on the GEMV path, but dispatching all
        // four via rocBLAS keeps the codepath uniform. Amortizes well.
        if self.rocblas_arch_eligible()
            && batch_size >= self.rocblas_min_batch()
            && self.rocblas.is_some()
            && !self.capture_mode
        {
            let shadow_qkv = self.ensure_fp16_shadow(a_qkv, qkv_m, k)?;
            let shadow_z = self.ensure_fp16_shadow(a_z, z_m, k)?;
            let shadow_beta = self.ensure_fp16_shadow(a_beta, beta_m, k)?;
            let shadow_alpha = self.ensure_fp16_shadow(a_alpha, alpha_m, k)?;
            if let (Some(pq), Some(pz), Some(pb), Some(pa)) =
                (shadow_qkv, shadow_z, shadow_beta, shadow_alpha) {
                let x_fp16 = self.ensure_fp16_x(x, batch_size * k)?;
                let xb = unsafe { DeviceBuffer::from_raw(x_fp16, (batch_size * k) * 2) };
                let wq = unsafe { DeviceBuffer::from_raw(pq, (qkv_m * k) * 2) };
                let wz_b = unsafe { DeviceBuffer::from_raw(pz, (z_m * k) * 2) };
                let wb = unsafe { DeviceBuffer::from_raw(pb, (beta_m * k) * 2) };
                let wa = unsafe { DeviceBuffer::from_raw(pa, (alpha_m * k) * 2) };
                let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq4g256_rocblas",
                    crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
                    + crate::profile::gemv_hfq4g256_bytes(z_m, k)
                    + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
                    + crate::profile::gemv_hfq4g256_bytes(alpha_m, k));
                let r1 = self.rocblas_gemm_hfq4_prefill(&wq, &xb, &y_qkv.buf, qkv_m, batch_size, k);
                let r2 = if r1.is_ok() { self.rocblas_gemm_hfq4_prefill(&wz_b, &xb, &y_z.buf, z_m, batch_size, k) } else { Ok(()) };
                let r3 = if r2.is_ok() { self.rocblas_gemm_hfq4_prefill(&wb, &xb, &y_beta.buf, beta_m, batch_size, k) } else { Ok(()) };
                let r4 = if r3.is_ok() { self.rocblas_gemm_hfq4_prefill(&wa, &xb, &y_alpha.buf, alpha_m, batch_size, k) } else { Ok(()) };
                std::mem::forget(xb); std::mem::forget(wq); std::mem::forget(wz_b);
                std::mem::forget(wb); std::mem::forget(wa);
                if let Some(t) = timer { t.finish(&self.hip); }
                return r1.and(r2).and(r3).and(r4);
            }
        }
        // Fast paths for prefill (batch_size > 1). Disable with HIPFIRE_FP16=0.
        if batch_size > 1 && !std::env::var("HIPFIRE_FP16").map_or(false, |v| v == "0") {
            // Wave64 FP16 hybrid — best of both worlds for gfx906 (MI50).
            if is_gcn5_wave64(&self.arch) {
                // gfx906 dp4a MMQ split: qkv + z route through the new MMQ
                // kernel (large-M outputs); beta + alpha keep the fused
                // wave64 kernel because their M (=linear_num_value_heads,
                // typically 32) is far below MMQ_Y=128 — bounds-checked
                // MMQ would waste ~75% of each row-tile.
                //
                // The fused wave64 kernel accepts qkv_m=0, z_m=0 to handle
                // the beta+alpha tail alone (its row-routing logic skips
                // the qkv/z branches when those Ms are zero). See
                // kernels/src/gemm_qkvza_hfq4g256_fp16_wave64.hip:54-61.
                //
                // Routes through MMQ at batch_size ≥ 16 (per
                // should_use_mmq's gfx906 default). Falls through to the
                // fused wave64 if any of qkv/z screening rejects (matches
                // gate_up's behavior in gemm_gate_up_hfq4g256).
                if self.arch == "gfx906" && should_use_mmq(&self.arch, batch_size) {
                    let qz_safe = if self.mmq_screen {
                        self.mmq_screen_weight(a_qkv, qkv_m, k)
                            && self.mmq_screen_weight(a_z, z_m, k)
                    } else { true };
                    if qz_safe {
                        let xq = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
                        let r1 = self.gemm_hfq4g256_mmq_set_gfx906(a_qkv, xq, y_qkv, qkv_m, k, batch_size);
                        let r2 = if r1.is_ok() {
                            self.gemm_hfq4g256_mmq_set_gfx906(a_z, xq, y_z, z_m, k, batch_size)
                        } else { Ok(()) };
                        // Tail: beta+alpha through the fused wave64 with
                        // qkv_m=0, z_m=0. a_qkv/a_z pointers are passed but
                        // unread because no thread satisfies gid<qkv_m or
                        // gid<qkv_m+z_m when both are zero.
                        let r3 = if r2.is_ok() {
                            self.gemm_qkvza_hfq4g256_fp16_wave64(
                                a_qkv, a_z, a_beta, a_alpha, x,
                                y_qkv, y_z, y_beta, y_alpha,
                                0, 0, beta_m, alpha_m, k, batch_size,
                            )
                        } else { Ok(()) };
                        return r1.and(r2).and(r3);
                    }
                    // else: qkv or z screening rejected — fall through
                    // to fused wave64 (handles all 4 outputs together).
                }
                return self.gemm_qkvza_hfq4g256_fp16_wave64(a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m, alpha_m, k, batch_size);
            }
            if should_use_mmq(&self.arch, batch_size) {
                let use_mmq = if self.mmq_screen {
                    self.mmq_screen_weight(a_qkv, qkv_m, k)
                        && self.mmq_screen_weight(a_z, z_m, k)
                        && self.mmq_screen_weight(a_beta, beta_m, k)
                        && self.mmq_screen_weight(a_alpha, alpha_m, k)
                } else { true };
                if use_mmq {
                    let xq = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
                    let r1 = self.gemm_hfq4g256_mmq_set_prequant(a_qkv, xq, y_qkv, qkv_m, k, batch_size);
                    let r2 = if r1.is_ok() { self.gemm_hfq4g256_mmq_set_prequant(a_z, xq, y_z, z_m, k, batch_size) } else { Ok(()) };
                    let r3 = if r2.is_ok() { self.gemm_hfq4g256_mmq_set_prequant(a_beta, xq, y_beta, beta_m, k, batch_size) } else { Ok(()) };
                    let r4 = if r3.is_ok() { self.gemm_hfq4g256_mmq_set_prequant(a_alpha, xq, y_alpha, alpha_m, k, batch_size) } else { Ok(()) };
                    return r1.and(r2).and(r3).and(r4);
                }
            }
            if has_wmma_f16_gfx12(&self.arch) {
                return self.gemm_qkvza_hfq4g256_wmma_gfx12(a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m, alpha_m, k, batch_size);
            }
            if has_wmma_f16(&self.arch) {
                return self.gemm_qkvza_hfq4g256_wmma(a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m, alpha_m, k, batch_size);
            }
            // v_dot2_f32_f16 on archs that have it (gfx1011/1012/1030-1032).
            // Excludes gfx1010 (Navi 10, 5700 XT) and gfx1013 (Van Gogh/BC-250 APU).
            if has_dot2_f32_f16(&self.arch) {
                return self.gemm_qkvza_hfq4g256_dot2(a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m, alpha_m, k, batch_size);
            }
            // FP16 packed (v_pk_fma_f16) for gfx1010/1013 — 2× scalar FP32.
            return self.gemm_qkvza_hfq4g256_fp16(a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m, alpha_m, k, batch_size);
        }
        let cdna_wave64 = has_wave64_native(&self.arch);
        let (func_name, block, grid_div): (&str, [u32; 3], u32) = if cdna_wave64 {
            self.ensure_kernel(
                "gemm_qkvza_hfq4g256_wave64",
                kernels::GEMM_QKVZA_HFQ4G256_WAVE64_SRC,
                "gemm_qkvza_hfq4g256_wave64",
            )?;
            ("gemm_qkvza_hfq4g256_wave64", [64, 1, 1], 2)
        } else {
            self.ensure_kernel(
                "gemm_qkvza_hfq4g256",
                kernels::GEMM_QKVZA_HFQ4G256_SRC,
                "gemm_qkvza_hfq4g256",
            )?;
            ("gemm_qkvza_hfq4g256", [32, 1, 1], 1)
        };
        let func = &self.functions[func_name];

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = { const BATCH_TILE: usize = 8; (batch_size + BATCH_TILE - 1) / BATCH_TILE };
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;
        let grid_x = (total_m + grid_div - 1) / grid_div;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(z_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(alpha_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq4g256", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, batch_tiles as u32, 1],
                block,
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// FP16-packed batched 4-way fused HFQ4-G256 GEMM (qkv + z + beta + alpha).
    /// RDNA1/2 fast path — v_pk_fma_f16 inner loop, 2× scalar FP32 throughput.
    /// Requires FP16-converted X (provided via ensure_fp16_x).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq4g256_fp16(
        &mut self,
        a_qkv: &GpuTensor, a_z: &GpuTensor, a_beta: &GpuTensor, a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor, y_z: &GpuTensor, y_beta: &GpuTensor, y_alpha: &GpuTensor,
        qkv_m: usize, z_m: usize, beta_m: usize, alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq4g256_fp16",
            kernels::GEMM_QKVZA_HFQ4G256_FP16_SRC,
            "gemm_qkvza_hfq4g256_fp16",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkvza_hfq4g256_fp16"];

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = { const BATCH_TILE: usize = 8; (batch_size + BATCH_TILE - 1) / BATCH_TILE };
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(z_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(alpha_m, k)
                  + batch_size * k * 2  // FP16 X
                  + batch_size * (qkv_m + z_m + beta_m + alpha_m) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq4g256_fp16", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Wave64 FP16 hybrid batched 4-way fused HFQ4-G256 GEMM (qkv + z + beta + alpha).
    /// Combines wave64 block structure (2 rows/block, full lane utilization) with
    /// FP16 packed arithmetic (__hfma2). Target: gfx906 (MI50) prefill optimization.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq4g256_fp16_wave64(
        &mut self,
        a_qkv: &GpuTensor, a_z: &GpuTensor, a_beta: &GpuTensor, a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor, y_z: &GpuTensor, y_beta: &GpuTensor, y_alpha: &GpuTensor,
        qkv_m: usize, z_m: usize, beta_m: usize, alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq4g256_fp16_wave64",
            kernels::GEMM_QKVZA_HFQ4G256_FP16_WAVE64_SRC,
            "gemm_qkvza_hfq4g256_fp16_wave64",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkvza_hfq4g256_fp16_wave64"];

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = { const BATCH_TILE: usize = 8; (batch_size + BATCH_TILE - 1) / BATCH_TILE };
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;
        let grid_x = (total_m + 1) / 2;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(z_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(alpha_m, k)
                  + batch_size * k * 2  // FP16 X
                  + batch_size * (qkv_m + z_m + beta_m + alpha_m) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq4g256_fp16_wave64", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, batch_tiles as u32, 1],
                [64, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// v_dot2_f32_f16-accelerated batched 4-way fused HFQ4-G256 GEMM (qkv + z + beta + alpha).
    /// RDNA2 (gfx1011/1012/1030-1032) fast path using `amd_mixed_dot`.
    /// One instruction per half2 dot with FP32 accumulation — 1.2-1.5× over FP16 packed.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq4g256_dot2(
        &mut self,
        a_qkv: &GpuTensor, a_z: &GpuTensor, a_beta: &GpuTensor, a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor, y_z: &GpuTensor, y_beta: &GpuTensor, y_alpha: &GpuTensor,
        qkv_m: usize, z_m: usize, beta_m: usize, alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq4g256_dot2",
            kernels::GEMM_QKVZA_HFQ4G256_DOT2_SRC,
            "gemm_qkvza_hfq4g256_dot2",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkvza_hfq4g256_dot2"];

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = { const BATCH_TILE: usize = 8; (batch_size + BATCH_TILE - 1) / BATCH_TILE };
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(z_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(alpha_m, k)
                  + batch_size * k * 2
                  + batch_size * (qkv_m + z_m + beta_m + alpha_m) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq4g256_dot2", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func, [total_m, batch_tiles as u32, 1], [32, 1, 1], 0, self.stream_ref(), &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched 3-way fused HFQ4-G256 GEMM for the FA preamble.
    ///
    /// Processes N tokens × three projections (wq + wk + wv) in one launch.
    /// Bitwise-identical to calling `fused_qkv_hfq4g256` N times on the same
    /// x[b] — 4-accumulator interleave + pairwise combine preserved per
    /// batch element.
    pub fn gemm_qkv_hfq4g256(
        &mut self,
        a_q: &GpuTensor, a_k: &GpuTensor, a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor, y_k: &GpuTensor, y_v: &GpuTensor,
        q_m: usize, k_m: usize, v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // CDNA3 MFMA path — 3 back-to-back rocBLAS calls for Q, K, V.
        if self.rocblas_arch_eligible()
            && batch_size >= self.rocblas_min_batch()
            && self.rocblas.is_some()
            && !self.capture_mode
        {
            let sq = self.ensure_fp16_shadow(a_q, q_m, k)?;
            let sk = self.ensure_fp16_shadow(a_k, k_m, k)?;
            let sv = self.ensure_fp16_shadow(a_v, v_m, k)?;
            if let (Some(pq), Some(pk), Some(pv)) = (sq, sk, sv) {
                let x_fp16 = self.ensure_fp16_x(x, batch_size * k)?;
                let xb = unsafe { DeviceBuffer::from_raw(x_fp16, (batch_size * k) * 2) };
                let wq = unsafe { DeviceBuffer::from_raw(pq, (q_m * k) * 2) };
                let wk = unsafe { DeviceBuffer::from_raw(pk, (k_m * k) * 2) };
                let wv = unsafe { DeviceBuffer::from_raw(pv, (v_m * k) * 2) };
                let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq4g256_rocblas",
                    crate::profile::gemv_hfq4g256_bytes(q_m, k)
                    + crate::profile::gemv_hfq4g256_bytes(k_m, k)
                    + crate::profile::gemv_hfq4g256_bytes(v_m, k));
                let r1 = self.rocblas_gemm_hfq4_prefill(&wq, &xb, &y_q.buf, q_m, batch_size, k);
                let r2 = if r1.is_ok() { self.rocblas_gemm_hfq4_prefill(&wk, &xb, &y_k.buf, k_m, batch_size, k) } else { Ok(()) };
                let r3 = if r2.is_ok() { self.rocblas_gemm_hfq4_prefill(&wv, &xb, &y_v.buf, v_m, batch_size, k) } else { Ok(()) };
                std::mem::forget(xb); std::mem::forget(wq); std::mem::forget(wk); std::mem::forget(wv);
                if let Some(t) = timer { t.finish(&self.hip); }
                return r1.and(r2).and(r3);
            }
        }
        // Fast paths for prefill (batch_size > 1). Disable with HIPFIRE_FP16=0.
        if batch_size > 1 && !std::env::var("HIPFIRE_FP16").map_or(false, |v| v == "0") {
            // Wave64 FP16 hybrid — best of both worlds for gfx906 (MI50).
            if is_gcn5_wave64(&self.arch) {
                // gfx906 dp4a MMQ: route q+k+v through the new MMQ kernel.
                // Unlike qkvza, all three qkv outputs have M well above
                // MMQ_Y=128 (Qwen 9B full-attn: q_m=4096, k_m=v_m=1024),
                // so no tail kernel is needed — straight 3× MMQ-set.
                //
                // Routes through MMQ at batch_size ≥ 16 (per
                // should_use_mmq's gfx906 default). Falls through to the
                // fused wave64 if any of q/k/v screening rejects.
                if self.arch == "gfx906" && should_use_mmq(&self.arch, batch_size) {
                    let qkv_safe = if self.mmq_screen {
                        self.mmq_screen_weight(a_q, q_m, k)
                            && self.mmq_screen_weight(a_k, k_m, k)
                            && self.mmq_screen_weight(a_v, v_m, k)
                    } else { true };
                    if qkv_safe {
                        let xq = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
                        let r1 = self.gemm_hfq4g256_mmq_set_gfx906(a_q, xq, y_q, q_m, k, batch_size);
                        let r2 = if r1.is_ok() {
                            self.gemm_hfq4g256_mmq_set_gfx906(a_k, xq, y_k, k_m, k, batch_size)
                        } else { Ok(()) };
                        let r3 = if r2.is_ok() {
                            self.gemm_hfq4g256_mmq_set_gfx906(a_v, xq, y_v, v_m, k, batch_size)
                        } else { Ok(()) };
                        return r1.and(r2).and(r3);
                    }
                    // else: fall through to fused wave64
                }
                return self.gemm_qkv_hfq4g256_fp16_wave64(a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size);
            }
            if should_use_mmq(&self.arch, batch_size) {
                let use_mmq = if self.mmq_screen {
                    self.mmq_screen_weight(a_q, q_m, k)
                        && self.mmq_screen_weight(a_k, k_m, k)
                        && self.mmq_screen_weight(a_v, v_m, k)
                } else { true };
                if use_mmq {
                    let xq = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
                    let r1 = self.gemm_hfq4g256_mmq_set_prequant(a_q, xq, y_q, q_m, k, batch_size);
                    let r2 = if r1.is_ok() { self.gemm_hfq4g256_mmq_set_prequant(a_k, xq, y_k, k_m, k, batch_size) } else { Ok(()) };
                    let r3 = if r2.is_ok() { self.gemm_hfq4g256_mmq_set_prequant(a_v, xq, y_v, v_m, k, batch_size) } else { Ok(()) };
                    return r1.and(r2).and(r3);
                }
            }
            if has_wmma_f16_gfx12(&self.arch) {
                return self.gemm_qkv_hfq4g256_wmma_gfx12(a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size);
            }
            if has_wmma_f16(&self.arch) {
                return self.gemm_qkv_hfq4g256_wmma(a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size);
            }
            // v_dot2_f32_f16 on archs that have it (gfx1011/1012/1030-1032).
            // Excludes gfx1010 (Navi 10, 5700 XT) and gfx1013 (Van Gogh/BC-250 APU).
            if has_dot2_f32_f16(&self.arch) {
                return self.gemm_qkv_hfq4g256_dot2(a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size);
            }
            // FP16 packed (v_pk_fma_f16) for gfx1010/1013 — 2× scalar FP32.
            return self.gemm_qkv_hfq4g256_fp16(a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size);
        }
        let cdna_wave64 = has_wave64_native(&self.arch);
        let (func_name, block, grid_div): (&str, [u32; 3], u32) = if cdna_wave64 {
            self.ensure_kernel(
                "gemm_qkv_hfq4g256_wave64",
                kernels::GEMM_QKV_HFQ4G256_WAVE64_SRC,
                "gemm_qkv_hfq4g256_wave64",
            )?;
            ("gemm_qkv_hfq4g256_wave64", [64, 1, 1], 2)
        } else {
            self.ensure_kernel(
                "gemm_qkv_hfq4g256",
                kernels::GEMM_QKV_HFQ4G256_SRC,
                "gemm_qkv_hfq4g256",
            )?;
            ("gemm_qkv_hfq4g256", [32, 1, 1], 1)
        };
        let func = &self.functions[func_name];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = { const BATCH_TILE: usize = 8; (batch_size + BATCH_TILE - 1) / BATCH_TILE };
        let total_m = (q_m + k_m + v_m) as u32;
        let grid_x = (total_m + grid_div - 1) / grid_div;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(k_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(v_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq4g256", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, batch_tiles as u32, 1],
                block,
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// FP16-packed batched 3-way fused HFQ4-G256 GEMM (Q + K + V).
    /// RDNA1/2 fast path — v_pk_fma_f16 inner loop, 2× scalar FP32 throughput.
    /// Requires FP16-converted X (provided via ensure_fp16_x).
    pub fn gemm_qkv_hfq4g256_fp16(
        &mut self,
        a_q: &GpuTensor, a_k: &GpuTensor, a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor, y_k: &GpuTensor, y_v: &GpuTensor,
        q_m: usize, k_m: usize, v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq4g256_fp16",
            kernels::GEMM_QKV_HFQ4G256_FP16_SRC,
            "gemm_qkv_hfq4g256_fp16",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkv_hfq4g256_fp16"];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = { const BATCH_TILE: usize = 8; (batch_size + BATCH_TILE - 1) / BATCH_TILE };
        let total_m = (q_m + k_m + v_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(k_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(v_m, k)
                  + batch_size * k * 2  // FP16 X
                  + batch_size * (q_m + k_m + v_m) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq4g256_fp16", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Wave64 FP16 hybrid batched 3-way fused HFQ4-G256 GEMM (Q + K + V).
    /// Combines wave64 block structure (2 rows/block, full lane utilization) with
    /// FP16 packed arithmetic (__hfma2). Target: gfx906 (MI50) prefill optimization.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkv_hfq4g256_fp16_wave64(
        &mut self,
        a_q: &GpuTensor, a_k: &GpuTensor, a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor, y_k: &GpuTensor, y_v: &GpuTensor,
        q_m: usize, k_m: usize, v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq4g256_fp16_wave64",
            kernels::GEMM_QKV_HFQ4G256_FP16_WAVE64_SRC,
            "gemm_qkv_hfq4g256_fp16_wave64",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkv_hfq4g256_fp16_wave64"];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = { const BATCH_TILE: usize = 8; (batch_size + BATCH_TILE - 1) / BATCH_TILE };
        let total_m = (q_m + k_m + v_m) as u32;
        let grid_x = (total_m + 1) / 2;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(k_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(v_m, k)
                  + batch_size * k * 2  // FP16 X
                  + batch_size * (q_m + k_m + v_m) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq4g256_fp16_wave64", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, batch_tiles as u32, 1],
                [64, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// v_dot2_f32_f16-accelerated batched 3-way fused HFQ4-G256 GEMM (Q + K + V).
    /// RDNA2 (gfx1011/1012/1030-1032) fast path using `amd_mixed_dot`.
    /// One instruction per half2 dot with FP32 accumulation — 1.2-1.5× over FP16 packed.
    pub fn gemm_qkv_hfq4g256_dot2(
        &mut self,
        a_q: &GpuTensor, a_k: &GpuTensor, a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor, y_k: &GpuTensor, y_v: &GpuTensor,
        q_m: usize, k_m: usize, v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq4g256_dot2",
            kernels::GEMM_QKV_HFQ4G256_DOT2_SRC,
            "gemm_qkv_hfq4g256_dot2",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkv_hfq4g256_dot2"];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = { const BATCH_TILE: usize = 8; (batch_size + BATCH_TILE - 1) / BATCH_TILE };
        let total_m = (q_m + k_m + v_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(k_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(v_m, k)
                  + batch_size * k * 2
                  + batch_size * (q_m + k_m + v_m) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq4g256_dot2", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func, [total_m, batch_tiles as u32, 1], [32, 1, 1], 0, self.stream_ref(), &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched 2-way fused HFQ4-G256 GEMM for the FFN preamble (gate + up).
    ///
    /// Processes N tokens × both projections (w_gate + w_up) in one launch.
    /// Bitwise-identical to calling `fused_gate_up_hfq4g256` N times on the
    /// same x[b] — 4-accumulator interleave + pairwise combine preserved
    /// per batch element.
    pub fn gemm_gate_up_hfq4g256(
        &mut self,
        a_gate: &GpuTensor, a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor, y_up: &GpuTensor,
        gate_m: usize, up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // CDNA3 MFMA path (task #130): two back-to-back rocBLAS calls against
        // the gate/up FP16 shadows. rocBLAS launch overhead is small compared
        // to the GEMM work at prefill batches, so fusing into a single
        // concatenated matrix isn't worth the extra kernel code tonight.
        let cdna3 = matches!(self.arch.as_str(), "gfx940" | "gfx941" | "gfx942");
        if cdna3
            && batch_size >= self.rocblas_min_batch()
            && self.rocblas.is_some()
            && !self.capture_mode
        {
            if let Ok(Some(w_gate_ptr)) = self.ensure_fp16_shadow(a_gate, gate_m, k) {
                if let Ok(Some(w_up_ptr)) = self.ensure_fp16_shadow(a_up, up_m, k) {
                    let x_fp16 = self.ensure_fp16_x(x, batch_size * k)?;
                    let xb = unsafe { DeviceBuffer::from_raw(x_fp16, (batch_size * k) * 2) };
                    let wgate = unsafe { DeviceBuffer::from_raw(w_gate_ptr, (gate_m * k) * 2) };
                    let wup = unsafe { DeviceBuffer::from_raw(w_up_ptr, (up_m * k) * 2) };
                    let gate_bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k);
                    let up_bytes = crate::profile::gemv_hfq4g256_bytes(up_m, k);
                    let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq4g256_rocblas", gate_bytes + up_bytes);
                    let r1 = self.rocblas_gemm_hfq4_prefill(&wgate, &xb, &y_gate.buf, gate_m, batch_size, k);
                    let r2 = if r1.is_ok() {
                        self.rocblas_gemm_hfq4_prefill(&wup, &xb, &y_up.buf, up_m, batch_size, k)
                    } else { Ok(()) };
                    std::mem::forget(xb);
                    std::mem::forget(wgate);
                    std::mem::forget(wup);
                    if let Some(t) = timer { t.finish(&self.hip); }
                    return r1.and(r2);
                }
            }
        }
        // Fast paths for prefill (batch_size > 1). Disable with HIPFIRE_FP16=0.
        if batch_size > 1 && !std::env::var("HIPFIRE_FP16").map_or(false, |v| v == "0") {
            // gfx906 dp4a MMQ — default-on at batch_size ≥ 16 (per
            // should_use_mmq's gfx906 default). Quantize X once, screen
            // both weights, dispatch MMQ for each in set mode (add=0).
            // Falls through to fused FP16 wave64 if either screening
            // rejects. See docs/plans/gfx906-mmq-prd.md for context.
            if self.arch == "gfx906" && should_use_mmq(&self.arch, batch_size) {
                let use_mmq = if self.mmq_screen {
                    self.mmq_screen_weight(a_gate, gate_m, k)
                        && self.mmq_screen_weight(a_up, up_m, k)
                } else { true };
                if use_mmq {
                    let xq = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
                    let r1 = self.gemm_hfq4g256_mmq_set_gfx906(a_gate, xq, y_gate, gate_m, k, batch_size);
                    let r2 = if r1.is_ok() {
                        self.gemm_hfq4g256_mmq_set_gfx906(a_up, xq, y_up, up_m, k, batch_size)
                    } else { Ok(()) };
                    return r1.and(r2);
                }
                // else: screening rejected at least one weight — fall through to wave64.
            }
            // Wave64 FP16 hybrid — best of both worlds for gfx906 (MI50).
            if is_gcn5_wave64(&self.arch) {
                return self.gemm_gate_up_hfq4g256_fp16_wave64(a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size);
            }
            if should_use_mmq(&self.arch, batch_size) {
                let use_mmq = if self.mmq_screen {
                    self.mmq_screen_weight(a_gate, gate_m, k)
                        && self.mmq_screen_weight(a_up, up_m, k)
                } else { true };
                if use_mmq {
                    let xq = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
                    let r1 = self.gemm_hfq4g256_mmq_set_prequant(a_gate, xq, y_gate, gate_m, k, batch_size);
                    let r2 = if r1.is_ok() { self.gemm_hfq4g256_mmq_set_prequant(a_up, xq, y_up, up_m, k, batch_size) } else { Ok(()) };
                    return r1.and(r2);
                }
            }
            // WMMA on gfx12 (RDNA4)
            if has_wmma_f16_gfx12(&self.arch) {
                return self.gemm_gate_up_hfq4g256_wmma_gfx12(a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size);
            }
            // WMMA on gfx11 (RDNA3)
            if has_wmma_f16(&self.arch) {
                return self.gemm_gate_up_hfq4g256_wmma(a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size);
            }
            // v_dot2_f32_f16 on archs that have it (gfx1011/1012/1030-1032).
            // Excludes gfx1010 (Navi 10, 5700 XT) and gfx1013 (Van Gogh/BC-250 APU).
            if has_dot2_f32_f16(&self.arch) {
                return self.gemm_gate_up_hfq4g256_dot2(a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size);
            }
            // FP16 packed (v_pk_fma_f16) for gfx1010/1013 — 2× scalar FP32.
            return self.gemm_gate_up_hfq4g256_fp16(a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size);
        }
        self.ensure_kernel("gemm_gate_up_hfq4g256", kernels::GEMM_GATE_UP_HFQ4G256_SRC, "gemm_gate_up_hfq4g256")?;
        let func = &self.functions["gemm_gate_up_hfq4g256"];

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m = gate_m as i32;
        let mut u_m = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m as *mut _ as *mut c_void,
            &mut u_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = { const BATCH_TILE: usize = 8; (batch_size + BATCH_TILE - 1) / BATCH_TILE };
        let total_m = (gate_m + up_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(up_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq4g256", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// v_dot2_f32_f16-accelerated batched 2-way fused HFQ4-G256 GEMM (gate + up).
    /// RDNA2 (gfx1011/1012/1030-1032) fast path using `__ockl_fdot2`.
    /// One instruction per half2 dot with FP32 accumulation — 1.2-1.5× over FP16 packed.
    pub fn gemm_gate_up_hfq4g256_dot2(
        &mut self,
        a_gate: &GpuTensor, a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor, y_up: &GpuTensor,
        gate_m: usize, up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_gate_up_hfq4g256_dot2",
            kernels::GEMM_GATE_UP_HFQ4G256_DOT2_SRC,
            "gemm_gate_up_hfq4g256_dot2",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_gate_up_hfq4g256_dot2"];

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m = gate_m as i32;
        let mut u_m = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m as *mut _ as *mut c_void,
            &mut u_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = { const BATCH_TILE: usize = 8; (batch_size + BATCH_TILE - 1) / BATCH_TILE };
        let total_m = (gate_m + up_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(up_m, k)
                  + batch_size * k * 2
                  + batch_size * (gate_m + up_m) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq4g256_dot2", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func, [total_m, batch_tiles as u32, 1], [32, 1, 1], 0, self.stream_ref(), &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// FP16-packed batched 2-way fused HFQ4-G256 GEMM (gate + up).
    /// RDNA1/2 fast path — v_pk_fma_f16 inner loop, 2× scalar FP32 throughput.
    /// Requires FP16-converted X (provided via ensure_fp16_x).
    pub fn gemm_gate_up_hfq4g256_fp16(
        &mut self,
        a_gate: &GpuTensor, a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor, y_up: &GpuTensor,
        gate_m: usize, up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_gate_up_hfq4g256_fp16",
            kernels::GEMM_GATE_UP_HFQ4G256_FP16_SRC,
            "gemm_gate_up_hfq4g256_fp16",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_gate_up_hfq4g256_fp16"];

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m = gate_m as i32;
        let mut u_m = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m as *mut _ as *mut c_void,
            &mut u_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = { const BATCH_TILE: usize = 8; (batch_size + BATCH_TILE - 1) / BATCH_TILE };
        let total_m = (gate_m + up_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(up_m, k)
                  + batch_size * k * 2  // FP16 X
                  + batch_size * (gate_m + up_m) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq4g256_fp16", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// GCN5 wave64 FP16 hybrid batched 2-way fused HFQ4-G256 GEMM (gate + up).
    /// block=[64,1,1] with 2 rows/block via warp_id. Halves grid.x vs wave32.
    /// Default-on for gfx906; gfx908 opts in via HIPFIRE_GCN5_WAVE64_HYBRID=1.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_gate_up_hfq4g256_fp16_wave64(
        &mut self,
        a_gate: &GpuTensor, a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor, y_up: &GpuTensor,
        gate_m: usize, up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_gate_up_hfq4g256_fp16_wave64",
            kernels::GEMM_GATE_UP_HFQ4G256_FP16_WAVE64_SRC,
            "gemm_gate_up_hfq4g256_fp16_wave64",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_gate_up_hfq4g256_fp16_wave64"];

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m = gate_m as i32;
        let mut u_m = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m as *mut _ as *mut c_void,
            &mut u_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = { const BATCH_TILE: usize = 8; (batch_size + BATCH_TILE - 1) / BATCH_TILE };
        let total_m = (gate_m + up_m) as u32;
        let grid_x = (total_m + 1) / 2;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(up_m, k)
                  + batch_size * k * 2  // FP16 X
                  + batch_size * (gate_m + up_m) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq4g256_fp16_wave64", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, batch_tiles as u32, 1],
                [64, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// WMMA-accelerated batched 5-way fused HFQ4-G256 GEMM (qkv + z + beta + alpha).
    /// gfx1100+ only. 16x16 output tiles via wave32 WMMA.
    pub fn gemm_qkvza_hfq4g256_wmma(
        &mut self,
        a_qkv: &GpuTensor, a_z: &GpuTensor, a_beta: &GpuTensor, a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor, y_z: &GpuTensor, y_beta: &GpuTensor, y_alpha: &GpuTensor,
        qkv_m: usize, z_m: usize, beta_m: usize, alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemm_qkvza_hfq4g256_wmma", kernels::GEMM_QKVZA_HFQ4G256_WMMA_SRC, "gemm_qkvza_hfq4g256_wmma")?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(z_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(alpha_m, k)
                  + batch_size * k * 2
                  + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq4g256_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkvza_hfq4g256_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq); b.push_ptr(az); b.push_ptr(ab); b.push_ptr(aa);
                b.push_ptr(xp);
                b.push_ptr(yq); b.push_ptr(yz); b.push_ptr(yb); b.push_ptr(ya);
                b.push_i32(q_m); b.push_i32(z_m_val); b.push_i32(b_m); b.push_i32(a_m);
                b.push_i32(k_val); b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// HFQ3-G256 sister of `gemm_qkvza_hfq4g256_wmma`. Same WMMA shape +
    /// lane decomposition; only the inner K-tile unpack differs (3-bit
    /// cross-byte vs 4-bit nibble) and the per-group byte stride is 104
    /// instead of 136. Used for MQ3 prefill via dispatch wrappers that
    /// pre-rotate `x` (see `gemm_qkvza_mq3g256_wmma` below). gfx11 K2
    /// unroll variant — gfx12 K4 to follow once K2 is validated.
    pub fn gemm_qkvza_hfq3g256_wmma(
        &mut self,
        a_qkv: &GpuTensor, a_z: &GpuTensor, a_beta: &GpuTensor, a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor, y_z: &GpuTensor, y_beta: &GpuTensor, y_alpha: &GpuTensor,
        qkv_m: usize, z_m: usize, beta_m: usize, alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if has_wmma_f16_gfx12(&self.arch) {
            return self.gemm_qkvza_hfq3g256_wmma_gfx12(
                a_qkv, a_z, a_beta, a_alpha, x,
                y_qkv, y_z, y_beta, y_alpha,
                qkv_m, z_m, beta_m, alpha_m, k, batch_size);
        }
        self.ensure_kernel("gemm_qkvza_hfq3g256_wmma", kernels::GEMM_QKVZA_HFQ3G256_WMMA_SRC, "gemm_qkvza_hfq3g256_wmma")?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        // HFQ3 storage = 104 B/group → ~3.06 bits/weight (vs HFQ4's 4.25).
        let weight_bytes = (qkv_m + z_m + beta_m + alpha_m) * (k / 256) * 104;
        let bytes = weight_bytes
                  + batch_size * k * 2
                  + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq3g256_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkvza_hfq3g256_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq); b.push_ptr(az); b.push_ptr(ab); b.push_ptr(aa);
                b.push_ptr(xp);
                b.push_ptr(yq); b.push_ptr(yz); b.push_ptr(yb); b.push_ptr(ya);
                b.push_i32(q_m); b.push_i32(z_m_val); b.push_i32(b_m); b.push_i32(a_m);
                b.push_i32(k_val); b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// MQ3 wrapper: rotates `x` via `mq_rotate_x` (FWHT with shared sign
    /// vectors) into the caller-provided `x_rot` scratch, then invokes
    /// `gemm_qkvza_hfq3g256_wmma`. Mirror of `gemm_qkvza_mq4g256_wmma`.
    /// Caller is responsible for `x_rot` being [batch × K] f32 scratch.
    pub fn gemm_qkvza_mq3g256_wmma(
        &mut self,
        a_qkv: &GpuTensor, a_z: &GpuTensor, a_beta: &GpuTensor, a_alpha: &GpuTensor,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        y_qkv: &GpuTensor, y_z: &GpuTensor, y_beta: &GpuTensor, y_alpha: &GpuTensor,
        qkv_m: usize, z_m: usize, beta_m: usize, alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Rotate batched x. mq_rotate_x_batched applies FWHT per-row.
        for b in 0..batch_size {
            let x_row = x.sub_offset(b * k, k);
            let x_rot_row = x_rot.sub_offset(b * k, k);
            self.rotate_x_mq(&x_row, &x_rot_row, k)?;
        }
        // Invalidate the fp16-conversion cache: `x_rot`'s pointer is stable
        // across consecutive MQ3 wrapper calls (same scratch buffer reused
        // per layer), but the underlying data was just rewritten by the
        // rotate loop above. Without this, `ensure_fp16_x` would see the
        // matching `fp16_x_source_ptr` and skip the f32→fp16 conversion,
        // and the kernel would read stale fp16 values from the previous
        // layer's rotation.
        self.fp16_x_source_ptr = std::ptr::null_mut();
        self.gemm_qkvza_hfq3g256_wmma(
            a_qkv, a_z, a_beta, a_alpha, x_rot,
            y_qkv, y_z, y_beta, y_alpha,
            qkv_m, z_m, beta_m, alpha_m, k, batch_size,
        )
    }

    /// gfx12 (RDNA4) sister of `gemm_qkvza_hfq3g256_wmma`. K4-unrolled
    /// half8_t lane-split per `gemm_qkvza_hfq4g256_wmma_gfx12`. Wired via
    /// the `gemm_qkvza_hfq3g256_wmma` arch dispatch — direct callers can
    /// also use this if they know they're on gfx12.
    pub fn gemm_qkvza_hfq3g256_wmma_gfx12(
        &mut self,
        a_qkv: &GpuTensor, a_z: &GpuTensor, a_beta: &GpuTensor, a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor, y_z: &GpuTensor, y_beta: &GpuTensor, y_alpha: &GpuTensor,
        qkv_m: usize, z_m: usize, beta_m: usize, alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq3g256_wmma_gfx12",
            kernels::GEMM_QKVZA_HFQ3G256_WMMA_GFX12_SRC,
            "gemm_qkvza_hfq3g256_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let weight_bytes = total_m * (k / 256) * 104;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq3g256_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkvza_hfq3g256_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq); b.push_ptr(az); b.push_ptr(ab); b.push_ptr(aa);
                b.push_ptr(xp);
                b.push_ptr(yq); b.push_ptr(yz); b.push_ptr(yb); b.push_ptr(ya);
                b.push_i32(q_m); b.push_i32(z_m_val); b.push_i32(b_m); b.push_i32(a_m);
                b.push_i32(k_val); b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// gfx12 (RDNA4) sister of `gemm_qkvza_hfq4g256_wmma`. Same gfx12
    /// recipe as the other scaffolds (validated on R9700) extended to
    /// 4-output qkv/z/beta/alpha routing. Not yet wired into the public
    /// dispatch tree — exposed only for the channel-test harness.
    pub fn gemm_qkvza_hfq4g256_wmma_gfx12(
        &mut self,
        a_qkv: &GpuTensor, a_z: &GpuTensor, a_beta: &GpuTensor, a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor, y_z: &GpuTensor, y_beta: &GpuTensor, y_alpha: &GpuTensor,
        qkv_m: usize, z_m: usize, beta_m: usize, alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq4g256_wmma_gfx12",
            kernels::GEMM_QKVZA_HFQ4G256_WMMA_GFX12_SRC,
            "gemm_qkvza_hfq4g256_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(z_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(alpha_m, k)
                  + batch_size * k * 2
                  + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq4g256_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkvza_hfq4g256_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq); b.push_ptr(az); b.push_ptr(ab); b.push_ptr(aa);
                b.push_ptr(xp);
                b.push_ptr(yq); b.push_ptr(yz); b.push_ptr(yb); b.push_ptr(ya);
                b.push_i32(q_m); b.push_i32(z_m_val); b.push_i32(b_m); b.push_i32(a_m);
                b.push_i32(k_val); b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// WMMA-accelerated batched 3-way fused HFQ4-G256 GEMM (Q + K + V).
    /// gfx1100+ only. 16x16 output tiles via wave32 WMMA.
    pub fn gemm_qkv_hfq4g256_wmma(
        &mut self,
        a_q: &GpuTensor, a_k: &GpuTensor, a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor, y_k: &GpuTensor, y_v: &GpuTensor,
        q_m: usize, k_m: usize, v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemm_qkv_hfq4g256_wmma", kernels::GEMM_QKV_HFQ4G256_WMMA_SRC, "gemm_qkv_hfq4g256_wmma")?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(k_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(v_m, k)
                  + batch_size * k * 2
                  + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq4g256_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_hfq4g256_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq); b.push_ptr(ak); b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq); b.push_ptr(yk); b.push_ptr(yv);
                b.push_i32(q_m_val); b.push_i32(k_m_val); b.push_i32(v_m_val);
                b.push_i32(k_val); b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// HFQ3-G256 sister of `gemm_qkv_hfq4g256_wmma`. Same WMMA shape +
    /// lane decomposition; only the inner K-tile unpack differs (3-bit
    /// cross-byte vs 4-bit nibble) and the per-group byte stride is 104
    /// instead of 136. Used for MQ3 prefill via dispatch sites in
    /// qwen35.rs FullAttention branch (X is pre-rotated upstream).
    pub fn gemm_qkv_hfq3g256_wmma(
        &mut self,
        a_q: &GpuTensor, a_k: &GpuTensor, a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor, y_k: &GpuTensor, y_v: &GpuTensor,
        q_m: usize, k_m: usize, v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if has_wmma_f16_gfx12(&self.arch) {
            return self.gemm_qkv_hfq3g256_wmma_gfx12(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size);
        }
        self.ensure_kernel("gemm_qkv_hfq3g256_wmma", kernels::GEMM_QKV_HFQ3G256_WMMA_SRC, "gemm_qkv_hfq3g256_wmma")?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let weight_bytes = (q_m + k_m + v_m) * (k / 256) * 104;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq3g256_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_hfq3g256_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq); b.push_ptr(ak); b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq); b.push_ptr(yk); b.push_ptr(yv);
                b.push_i32(q_m_val); b.push_i32(k_m_val); b.push_i32(v_m_val);
                b.push_i32(k_val); b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// gfx12 (RDNA4) sister of `gemm_qkv_hfq4g256_wmma`. Identical signature
    /// and grid/block; only the kernel-side intrinsic + operand vector size
    /// differs. NOT yet wired into the public dispatch tree — exposed only
    /// for the channel-test (`test_wmma_qkv_gfx12`) that validates the
    /// gfx12 C-output mapping hypothesis on real RDNA4 silicon. See issue
    /// #54 and `.skills/hipfire-arch-port/wmma-matrix.md`.
    pub fn gemm_qkv_hfq4g256_wmma_gfx12(
        &mut self,
        a_q: &GpuTensor, a_k: &GpuTensor, a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor, y_k: &GpuTensor, y_v: &GpuTensor,
        q_m: usize, k_m: usize, v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq4g256_wmma_gfx12",
            kernels::GEMM_QKV_HFQ4G256_WMMA_GFX12_SRC,
            "gemm_qkv_hfq4g256_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(k_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(v_m, k)
                  + batch_size * k * 2
                  + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq4g256_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_hfq4g256_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq); b.push_ptr(ak); b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq); b.push_ptr(yk); b.push_ptr(yv);
                b.push_i32(q_m_val); b.push_i32(k_m_val); b.push_i32(v_m_val);
                b.push_i32(k_val); b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// WMMA-accelerated batched 2-way fused HFQ4-G256 GEMM (gate + up).
    /// gfx1100+ only. 16x16 output tiles via wave32 WMMA.
    pub fn gemm_gate_up_hfq4g256_wmma(
        &mut self,
        a_gate: &GpuTensor, a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor, y_up: &GpuTensor,
        gate_m: usize, up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // HIPFIRE_GATE_UP_VARIANT=ldsx routes to the LDS-staged X variant
        // (Gate 1 microbench, opt-in only, default off). See
        // docs/perf-checkpoints/2026-05-01-gate-up-lds-x-share-plan.md.
        let variant_override = std::env::var("HIPFIRE_GATE_UP_VARIANT").ok();
        let (kernel_name, kernel_src) = match variant_override.as_deref() {
            Some("ldsx") => ("gemm_gate_up_hfq4g256_wmma_ldsx",
                             kernels::GEMM_GATE_UP_HFQ4G256_WMMA_LDSX_SRC),
            _            => ("gemm_gate_up_hfq4g256_wmma",
                             kernels::GEMM_GATE_UP_HFQ4G256_WMMA_SRC),
        };
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m = gate_m as i32;
        let mut u_m = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m as *mut _ as *mut c_void,
            &mut u_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(up_m, k)
                  + batch_size * k * 2
                  + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ag); b.push_ptr(au);
                b.push_ptr(xp);
                b.push_ptr(yg); b.push_ptr(yu);
                b.push_i32(g_m); b.push_i32(u_m); b.push_i32(k_val); b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// HFQ3-G256 sister of `gemm_gate_up_hfq4g256_wmma`. Same WMMA shape
    /// + lane decomposition; only the inner K-tile unpack differs (3-bit
    /// cross-byte vs 4-bit nibble) and the per-group byte stride is 104
    /// instead of 136. Used for MQ3 prefill via `gemm_gate_up_mq3g256_wmma`.
    /// gfx12 (RDNA4) sister of `gemm_qkv_hfq3g256_wmma`.
    pub fn gemm_qkv_hfq3g256_wmma_gfx12(
        &mut self,
        a_q: &GpuTensor, a_k: &GpuTensor, a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor, y_k: &GpuTensor, y_v: &GpuTensor,
        q_m: usize, k_m: usize, v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq3g256_wmma_gfx12",
            kernels::GEMM_QKV_HFQ3G256_WMMA_GFX12_SRC,
            "gemm_qkv_hfq3g256_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let weight_bytes = total_m * (k / 256) * 104;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq3g256_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_hfq3g256_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq); b.push_ptr(ak); b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq); b.push_ptr(yk); b.push_ptr(yv);
                b.push_i32(q_m_val); b.push_i32(k_m_val); b.push_i32(v_m_val);
                b.push_i32(k_val); b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    pub fn gemm_gate_up_hfq3g256_wmma(
        &mut self,
        a_gate: &GpuTensor, a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor, y_up: &GpuTensor,
        gate_m: usize, up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if has_wmma_f16_gfx12(&self.arch) {
            return self.gemm_gate_up_hfq3g256_wmma_gfx12(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size);
        }
        self.ensure_kernel("gemm_gate_up_hfq3g256_wmma", kernels::GEMM_GATE_UP_HFQ3G256_WMMA_SRC, "gemm_gate_up_hfq3g256_wmma")?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m = gate_m as i32;
        let mut u_m = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m as *mut _ as *mut c_void,
            &mut u_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let weight_bytes = (gate_m + up_m) * (k / 256) * 104;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq3g256_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_gate_up_hfq3g256_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ag); b.push_ptr(au);
                b.push_ptr(xp);
                b.push_ptr(yg); b.push_ptr(yu);
                b.push_i32(g_m); b.push_i32(u_m); b.push_i32(k_val); b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// gfx12 (RDNA4) sister of `gemm_gate_up_hfq3g256_wmma`.
    pub fn gemm_gate_up_hfq3g256_wmma_gfx12(
        &mut self,
        a_gate: &GpuTensor, a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor, y_up: &GpuTensor,
        gate_m: usize, up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_gate_up_hfq3g256_wmma_gfx12",
            kernels::GEMM_GATE_UP_HFQ3G256_WMMA_GFX12_SRC,
            "gemm_gate_up_hfq3g256_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m = gate_m as i32;
        let mut u_m = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m as *mut _ as *mut c_void,
            &mut u_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let weight_bytes = total_m * (k / 256) * 104;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq3g256_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_gate_up_hfq3g256_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ag); b.push_ptr(au);
                b.push_ptr(xp);
                b.push_ptr(yg); b.push_ptr(yu);
                b.push_i32(g_m); b.push_i32(u_m); b.push_i32(k_val); b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// MQ3 wrapper for `gemm_gate_up_hfq3g256_wmma`: pre-rotates X then
    /// dispatches the HFQ3 kernel. See `gemm_qkvza_mq3g256_wmma` for
    /// the cache-invalidation rationale.
    pub fn gemm_gate_up_mq3g256_wmma(
        &mut self,
        a_gate: &GpuTensor, a_up: &GpuTensor,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        y_gate: &GpuTensor, y_up: &GpuTensor,
        gate_m: usize, up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        for b in 0..batch_size {
            let x_row = x.sub_offset(b * k, k);
            let x_rot_row = x_rot.sub_offset(b * k, k);
            self.rotate_x_mq(&x_row, &x_rot_row, k)?;
        }
        self.fp16_x_source_ptr = std::ptr::null_mut();
        self.gemm_gate_up_hfq3g256_wmma(
            a_gate, a_up, x_rot, y_gate, y_up, gate_m, up_m, k, batch_size,
        )
    }

    /// gfx12 (RDNA4) sister of `gemm_gate_up_hfq4g256_wmma`. Same recipe
    /// as the QKV gfx12 scaffold (validated on R9700). Not yet wired into
    /// the public dispatch tree — exposed only for the channel-test
    /// harness. See issue #54.
    pub fn gemm_gate_up_hfq4g256_wmma_gfx12(
        &mut self,
        a_gate: &GpuTensor, a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor, y_up: &GpuTensor,
        gate_m: usize, up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_gate_up_hfq4g256_wmma_gfx12",
            kernels::GEMM_GATE_UP_HFQ4G256_WMMA_GFX12_SRC,
            "gemm_gate_up_hfq4g256_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m = gate_m as i32;
        let mut u_m = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m as *mut _ as *mut c_void,
            &mut u_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(up_m, k)
                  + batch_size * k * 2
                  + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq4g256_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_gate_up_hfq4g256_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ag); b.push_ptr(au);
                b.push_ptr(xp);
                b.push_ptr(yg); b.push_ptr(yu);
                b.push_i32(g_m); b.push_i32(u_m); b.push_i32(k_val); b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// gfx12 (RDNA4) sister of `gemm_hfq4g256_residual_wmma` (specifically
    /// the `_k2` variant — the gfx11 dispatch default for M >= 8192, with
    /// the validated C-output mapping).
    ///
    /// Closes the residual-GEMM gap on 9B prefill: before this kernel,
    /// gfx12 fell through to the dot2 fp16 fallback for the residual call
    /// site (attn-out + ffn-down), which accounted for ~42% of 9B prefill
    /// time on R9700. The other six gfx12 WMMA kernels shipped in PR #62.
    ///
    /// Same recipe as the qkv / qkvza / gate_up gfx12 ports: `_w32_gfx12`
    /// builtin, half8_t operands, K-split via `tid >> 4`, contiguous
    /// C-row mapping (`acc[j] = C[8*(tid>>4) + j][tid & 15]`). Validated
    /// on R9700 by the `test_wmma_residual_gfx12` channel-test against
    /// the dot2 reference path.
    pub fn gemm_hfq4g256_residual_wmma_gfx12(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfq4g256_residual_wmma_gfx12",
            kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_GFX12_SRC,
            "gemm_hfq4g256_residual_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        let row_tiles = (m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k)
            + batch_size * k * 2
            + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq4g256_residual_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_hfq4g256_residual_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr); b.push_ptr(x_ptr); b.push_ptr(y_ptr);
                b.push_i32(m_val); b.push_i32(k_val); b.push_i32(bs_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// HFQ4-G256 GEMV with fused residual add: y[row] += A[row] · x.
    /// Same math as `gemv_hfq4g256` but the final write accumulates into `y`
    /// instead of overwriting. Used for wo / w_down projections where the
    /// following step would have been `x += gemv_out` via add_inplace_f32.
    pub fn gemv_hfq4g256_residual(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemv_hfq4g256_residual_for_arch(&self.arch);
        self.ensure_kernel(module, src, "gemv_hfq4g256_residual")?;

        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        // CDNA3 wave64 fast path: 2 rows per block, halves grid.x. The base
        // kernel runs at half throughput on a wave64-native arch because
        // half the wave masks out per `__shfl_down`. Byte-exact with base.
        let cdna3 = has_wave64_native(&self.arch);

        // RDNA3 multi-row override path. Same selector as the non-residual
        // variant but there's currently no gfx1010-default multi-row residual
        // kernel, so non-RDNA3 archs still take the single-row residual path
        // regardless of HIPFIRE_GEMV_ROWS. (TODO: port the multi-row residual
        // kernel to the default path if/when the non-residual multi-row wins
        // scale to justify residual too.)
        let rdna3 = matches!(self.arch.as_str(), "gfx1100" | "gfx1101" | "gfx1102");
        let rows = if rdna3 { gemv_rows_override().unwrap_or(1) } else { 1 };
        let use_multirow = rdna3 && rows > 1;

        // Bandwidth: weight + x + y_read (for residual) + y_write.
        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemv", "gemv_hfq4g256_residual", bytes);
        let result = if cdna3 {
            let (kname, ksrc): (&str, &str) = if gemv_prefetch_enabled(&self.arch) {
                (
                    "gemv_hfq4g256_residual_wave64_prefetch",
                    kernels::GEMV_HFQ4G256_RESIDUAL_WAVE64_PREFETCH_SRC,
                )
            } else {
                (
                    "gemv_hfq4g256_residual_wave64",
                    kernels::GEMV_HFQ4G256_RESIDUAL_WAVE64_SRC,
                )
            };
            self.ensure_kernel(kname, ksrc, kname)?;
            let grid = ((m as u32) + 1) / 2;
            self.launch_maybe_blob(
                kname,
                [grid, 1, 1], [64, 1, 1], 0, &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(a_ptr); b.push_ptr(x_ptr); b.push_ptr(y_ptr);
                    b.push_i32(m_val); b.push_i32(k_val);
                    b
                },
            )
        } else if use_multirow {
            let (func_name, grid_div) = match rows {
                2 => ("gemv_hfq4g256_residual_multirow_r2", 2u32),
                4 => ("gemv_hfq4g256_residual_multirow_r4", 4u32),
                8 => ("gemv_hfq4g256_residual_multirow_r8", 8u32),
                _ => unreachable!(),
            };
            self.ensure_kernel(
                "gemv_hfq4g256_residual_multirow_rdna3",
                kernels::GEMV_HFQ4G256_RESIDUAL_MULTIROW_GFX1100_SRC,
                func_name,
            )?;
            let grid = ((m as u32) + grid_div - 1) / grid_div;
            self.launch_maybe_blob(
                func_name,
                [grid, 1, 1], [32, 1, 1], 0, &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(a_ptr); b.push_ptr(x_ptr); b.push_ptr(y_ptr);
                    b.push_i32(m_val); b.push_i32(k_val);
                    b
                },
            )
        } else {
            self.launch_maybe_blob(
                "gemv_hfq4g256_residual", [m as u32, 1, 1], [32, 1, 1], 0, &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(a_ptr); b.push_ptr(x_ptr); b.push_ptr(y_ptr);
                    b.push_i32(m_val); b.push_i32(k_val);
                    b
                },
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// HFQ4-G256 GEMV with fused SCALED residual add, CPU-scalar variant:
    ///   y[row] += scale * (A[row] · x)
    /// where `scale` is host-supplied by kernarg. Replaces the three-kernel
    /// tail of the MoE routed-expert epilogue (gemv → scale → add_inplace)
    /// with a single launch. Bit-exact with gemv_hfq4g256_residual followed
    /// by scaled_add_inplace_cpu_scalar when the inputs are identical —
    /// same accumulator layout, same pairwise combine.
    pub fn gemv_hfq4g256_residual_scaled_cpu(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        scale: f32,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq4g256_residual_scaled",
            kernels::GEMV_HFQ4G256_RESIDUAL_SCALED_SRC,
            "gemv_hfq4g256_residual_scaled_cpu",
        )?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let s_val = scale;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &s_val as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip, "gemv", "gemv_hfq4g256_residual_scaled_cpu", bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq4g256_residual_scaled_cpu", [m as u32, 1, 1], [32, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr); b.push_ptr(x_ptr); b.push_ptr(y_ptr);
                b.push_i32(m_val); b.push_i32(k_val);
                b.push_f32(s_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// HFQ4-G256 GEMV with fused SCALED residual add, GPU-scalar variant:
    ///   y[row] += c_buf[0] * (A[row] · x)
    /// Reads the scale from a 1-element device buffer. Used by the MoE
    /// shared-expert epilogue where `c_buf` holds sigmoid(gate · x) computed
    /// entirely on-device, avoiding a D2H sync.
    pub fn gemv_hfq4g256_residual_scaled_gpu(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        c_buf: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq4g256_residual_scaled",
            kernels::GEMV_HFQ4G256_RESIDUAL_SCALED_SRC,
            "gemv_hfq4g256_residual_scaled_gpu",
        )?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let c_ptr = c_buf.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &c_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip, "gemv", "gemv_hfq4g256_residual_scaled_gpu", bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq4g256_residual_scaled_gpu", [m as u32, 1, 1], [32, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr); b.push_ptr(x_ptr); b.push_ptr(y_ptr); b.push_ptr(c_ptr);
                b.push_i32(m_val); b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Same as `gemv_hfq4g256_residual_scaled_gpu` but applies sigmoid to
    /// `c_buf[0]` before scaling — lets the caller skip a separate
    /// `sigmoid_f32` launch on the 1-elem shared-expert gate scalar.
    /// Used by the A3B MoE FFN shared-expert down path.
    pub fn gemv_hfq4g256_residual_sigmoid_scaled_gpu(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        c_buf: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq4g256_residual_scaled",
            kernels::GEMV_HFQ4G256_RESIDUAL_SCALED_SRC,
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
        )?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let c_ptr = c_buf.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &c_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip, "gemv", "gemv_hfq4g256_residual_sigmoid_scaled_gpu", bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu", [m as u32, 1, 1], [32, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr); b.push_ptr(x_ptr); b.push_ptr(y_ptr); b.push_ptr(c_ptr);
                b.push_i32(m_val); b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// N-batched variant of `gemv_hfq4g256_residual_sigmoid_scaled_gpu`.
    /// `x_batch` is [N × K], `y_batch` is [N × M], `c_batch` is [N]. Each
    /// (row, token) block runs the HFQ4G256 GEMV body on its token's x
    /// row and atomicAdd's `sigmoid(c_batch[token]) * acc` into
    /// `y_batch[token × M + row]`. Used by the batched MoE FFN shared-
    /// expert down projection to eliminate N per-token launches.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched(
        &mut self,
        a_raw: &GpuTensor,
        x_batch: &GpuTensor,
        y_batch: &GpuTensor,
        c_batch: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq4g256_residual_scaled",
            kernels::GEMV_HFQ4G256_RESIDUAL_SCALED_SRC,
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched",
        )?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x_batch.buf.as_ptr();
        let y_ptr = y_batch.buf.as_ptr();
        let c_ptr = c_batch.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &c_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = batch_size * (crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip, "gemv", "gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched", bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched",
            [m as u32, batch_size as u32, 1], [32, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr); b.push_ptr(x_ptr); b.push_ptr(y_ptr); b.push_ptr(c_ptr);
                b.push_i32(m_val); b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// MoE fused gate_up GEMV: runs 8 top-K experts' HFQ4-G256 GEMV in a
    /// single launch. Caller passes the 8 selected experts' weight
    /// tensors (in top-K order); the kernel's grid.y picks which expert
    /// each block uses. Outputs are SPLIT into `y_gate` (first mi rows of
    /// each expert) and `y_up` (second mi rows), both `[k_top × mi]`
    /// row-major, so the next-stage batched silu_mul_rotate can consume
    /// them as plain [batch × K] buffers without extra strided reads.
    ///
    /// Bit-exact with running `gemv_hfq4g256` 8 times (same accumulator
    /// layout and pairwise final combine). `k_top` is currently hardcoded
    /// to 8 to match A3B; a generic path can follow alongside Phase 2b.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq4g256_moe_gate_up_k8(
        &mut self,
        w0: &GpuTensor, w1: &GpuTensor, w2: &GpuTensor, w3: &GpuTensor,
        w4: &GpuTensor, w5: &GpuTensor, w6: &GpuTensor, w7: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,   // [k_top × mi] — first half
        y_up:   &GpuTensor,   // [k_top × mi] — second half
        m: usize, k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq4g256_moe_gate_up",
            kernels::GEMV_HFQ4G256_MOE_GATE_UP_SRC,
            "gemv_hfq4g256_moe_gate_up_k8",
        )?;
        let w0p = w0.buf.as_ptr(); let w1p = w1.buf.as_ptr();
        let w2p = w2.buf.as_ptr(); let w3p = w3.buf.as_ptr();
        let w4p = w4.buf.as_ptr(); let w5p = w5.buf.as_ptr();
        let w6p = w6.buf.as_ptr(); let w7p = w7.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &w0p as *const _ as *mut c_void, &w1p as *const _ as *mut c_void,
            &w2p as *const _ as *mut c_void, &w3p as *const _ as *mut c_void,
            &w4p as *const _ as *mut c_void, &w5p as *const _ as *mut c_void,
            &w6p as *const _ as *mut c_void, &w7p as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &ygp as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        // Bandwidth: 8× weight, x read 8× (cached in practice), 8×m writes.
        let bytes = 8 * (crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip, "gemv", "gemv_hfq4g256_moe_gate_up_k8", bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq4g256_moe_gate_up_k8", [m as u32, 8, 1], [32, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(w0p); b.push_ptr(w1p); b.push_ptr(w2p); b.push_ptr(w3p);
                b.push_ptr(w4p); b.push_ptr(w5p); b.push_ptr(w6p); b.push_ptr(w7p);
                b.push_ptr(xp); b.push_ptr(ygp); b.push_ptr(yup);
                b.push_i32(m_val); b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// MoE fused down GEMV with scaled residual: accumulates 8 top-K
    /// experts' weighted contributions into `x_residual` in a single
    /// kernel launch. Grid.y selects the expert; each block atomicAdds
    /// `s_rank * (W_rank[row] · rot_batch[rank, :])` into `x_residual[row]`.
    /// Replaces 8 separate `gemv_hfq4g256_residual_scaled_cpu` calls.
    ///
    /// Atomic-add summation order is non-deterministic, so bit-exactness
    /// across runs isn't guaranteed (vs the sequential per-expert path).
    /// For A3B the MoE contribution is added on top of a non-trivial base,
    /// so the ordering-dependent FP noise is tiny in practice and the
    /// smoke-test decode still matches the Phase 2c step 2 output.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq4g256_moe_down_residual_scaled_k8(
        &mut self,
        w0: &GpuTensor, w1: &GpuTensor, w2: &GpuTensor, w3: &GpuTensor,
        w4: &GpuTensor, w5: &GpuTensor, w6: &GpuTensor, w7: &GpuTensor,
        rot_batch: &GpuTensor,
        x_residual: &GpuTensor,
        scales: [f32; 8],
        m: usize, k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq4g256_moe_down",
            kernels::GEMV_HFQ4G256_MOE_DOWN_SRC,
            "gemv_hfq4g256_moe_down_residual_scaled_k8",
        )?;
        let w0p = w0.buf.as_ptr(); let w1p = w1.buf.as_ptr();
        let w2p = w2.buf.as_ptr(); let w3p = w3.buf.as_ptr();
        let w4p = w4.buf.as_ptr(); let w5p = w5.buf.as_ptr();
        let w6p = w6.buf.as_ptr(); let w7p = w7.buf.as_ptr();
        let rbp = rot_batch.buf.as_ptr();
        let xrp = x_residual.buf.as_ptr();
        let [s0, s1, s2, s3, s4, s5, s6, s7] = scales;
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &w0p as *const _ as *mut c_void, &w1p as *const _ as *mut c_void,
            &w2p as *const _ as *mut c_void, &w3p as *const _ as *mut c_void,
            &w4p as *const _ as *mut c_void, &w5p as *const _ as *mut c_void,
            &w6p as *const _ as *mut c_void, &w7p as *const _ as *mut c_void,
            &rbp as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &s0 as *const _ as *mut c_void, &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void, &s3 as *const _ as *mut c_void,
            &s4 as *const _ as *mut c_void, &s5 as *const _ as *mut c_void,
            &s6 as *const _ as *mut c_void, &s7 as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = 8 * (crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip, "gemv", "gemv_hfq4g256_moe_down_residual_scaled_k8", bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq4g256_moe_down_residual_scaled_k8",
            [m as u32, 8, 1], [32, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(w0p); b.push_ptr(w1p); b.push_ptr(w2p); b.push_ptr(w3p);
                b.push_ptr(w4p); b.push_ptr(w5p); b.push_ptr(w6p); b.push_ptr(w7p);
                b.push_ptr(rbp); b.push_ptr(xrp);
                b.push_f32(s0); b.push_f32(s1); b.push_f32(s2); b.push_f32(s3);
                b.push_f32(s4); b.push_f32(s5); b.push_f32(s6); b.push_f32(s7);
                b.push_i32(m_val); b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// MoE router GPU softmax + top-K + (optional) renormalize. One
    /// workgroup, no D2H sync. Writes [k_top] i32 indices and [k_top]
    /// f32 weights to device buffers. Hardcoded k_top=8 to match A3B.
    pub fn moe_softmax_topk_renorm_k8(
        &mut self,
        logits: &GpuTensor,
        topk_idx: &GpuTensor,    // i32 [k_top]
        topk_w:   &GpuTensor,    // f32 [k_top]
        n_exp: usize,
        norm_topk: bool,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "moe_softmax_topk_k8",
            kernels::MOE_SOFTMAX_TOPK_K8_SRC,
            "moe_softmax_topk_renorm_k8",
        )?;
        let lp = logits.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let wp = topk_w.buf.as_ptr();
        let n  = n_exp as i32;
        let nr = if norm_topk { 1i32 } else { 0i32 };
        let mut params: Vec<*mut c_void> = vec![
            &lp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &n  as *const _ as *mut c_void,
            &nr as *const _ as *mut c_void,
        ];
        let bytes = n_exp * 4 + 8 * 8;
        let timer = crate::profile::begin_timer(
            &self.hip, "elementwise", "moe_softmax_topk_renorm_k8", bytes,
        );
        let result = self.launch_maybe_blob(
            "moe_softmax_topk_renorm_k8", [1, 1, 1], [256, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(lp); b.push_ptr(ip); b.push_ptr(wp);
                b.push_i32(n); b.push_i32(nr);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// MoE top-K + renorm given pre-softmaxed probs. Companion to the
    /// regular `softmax_f32`. The dispatch site runs `softmax_f32` first,
    /// then this kernel — same softmax math everywhere, no 1-ULP
    /// divergence between the routing path and a CPU reference.
    pub fn moe_topk_renorm_k8(
        &mut self,
        probs: &GpuTensor,        // [n_exp] f32, pre-softmaxed
        topk_idx: &GpuTensor,     // i32 [k_top]
        topk_w:   &GpuTensor,     // f32 [k_top]
        n_exp: usize,
        norm_topk: bool,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "moe_topk_renorm_k8",
            kernels::MOE_TOPK_RENORM_K8_SRC,
            "moe_topk_renorm_k8",
        )?;
        let lp = probs.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let wp = topk_w.buf.as_ptr();
        let n  = n_exp as i32;
        let nr = if norm_topk { 1i32 } else { 0i32 };
        let mut params: Vec<*mut c_void> = vec![
            &lp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &n  as *const _ as *mut c_void,
            &nr as *const _ as *mut c_void,
        ];
        let bytes = n_exp * 4 + 8 * 8;
        let timer = crate::profile::begin_timer(
            &self.hip, "elementwise", "moe_topk_renorm_k8", bytes,
        );
        let result = self.launch_maybe_blob(
            "moe_topk_renorm_k8", [1, 1, 1], [256, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(lp); b.push_ptr(ip); b.push_ptr(wp);
                b.push_i32(n); b.push_i32(nr);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Index-aware MoE gate_up GEMV. Reads expert_ids from a device-side
    /// topk_indices buffer and weight bases from expert_ptrs[expert_id].
    /// hipGraph-capture-safe replacement for the kernarg-pointer variant.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq4g256_moe_gate_up_k8_indexed(
        &mut self,
        expert_ptrs: &GpuTensor,   // [n_exp] of u64 device pointers
        topk_indices: &GpuTensor,  // [k_top] i32
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up:   &GpuTensor,
        m: usize, k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let cdna_wave64 = has_wave64_native(&self.arch);
        let (func_name, block, grid_x) = if cdna_wave64 {
            self.ensure_kernel(
                "gemv_hfq4g256_moe_gate_up_indexed_wave64",
                kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_WAVE64_SRC,
                "gemv_hfq4g256_moe_gate_up_k8_indexed_wave64",
            )?;
            ("gemv_hfq4g256_moe_gate_up_k8_indexed_wave64", [64u32, 1, 1], ((m as u32) + 1) / 2)
        } else {
            self.ensure_kernel(
                "gemv_hfq4g256_moe_gate_up_indexed",
                kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_SRC,
                "gemv_hfq4g256_moe_gate_up_k8_indexed",
            )?;
            ("gemv_hfq4g256_moe_gate_up_k8_indexed", [32u32, 1, 1], m as u32)
        };
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &ygp as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = 8 * (crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip, "gemv", "gemv_hfq4g256_moe_gate_up_k8_indexed", bytes,
        );
        let result = self.launch_maybe_blob(
            func_name,
            [grid_x, 8, 1], block, 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp); b.push_ptr(ip); b.push_ptr(xp);
                b.push_ptr(ygp); b.push_ptr(yup);
                b.push_i32(m_val); b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Index-aware MoE down GEMV with scaled residual. Same pattern as
    /// the indexed gate_up; also reads scales from a device topk_weights
    /// buffer and atomicAdds the contribution into x_residual.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq4g256_moe_down_residual_scaled_k8_indexed(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        topk_weights: &GpuTensor,
        rot_batch: &GpuTensor,
        x_residual: &GpuTensor,
        m: usize, k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let cdna_wave64 = has_wave64_native(&self.arch);
        let (func_name, block, grid_x) = if cdna_wave64 {
            self.ensure_kernel(
                "gemv_hfq4g256_moe_down_indexed_wave64",
                kernels::GEMV_HFQ4G256_MOE_DOWN_INDEXED_WAVE64_SRC,
                "gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_wave64",
            )?;
            ("gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_wave64", [64u32, 1, 1], ((m as u32) + 1) / 2)
        } else {
            self.ensure_kernel(
                "gemv_hfq4g256_moe_down_indexed",
                kernels::GEMV_HFQ4G256_MOE_DOWN_INDEXED_SRC,
                "gemv_hfq4g256_moe_down_residual_scaled_k8_indexed",
            )?;
            ("gemv_hfq4g256_moe_down_residual_scaled_k8_indexed", [32u32, 1, 1], m as u32)
        };
        let pp  = expert_ptrs.buf.as_ptr();
        let ip  = topk_indices.buf.as_ptr();
        let wp  = topk_weights.buf.as_ptr();
        let rbp = rot_batch.buf.as_ptr();
        let xrp = x_residual.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp  as *const _ as *mut c_void,
            &ip  as *const _ as *mut c_void,
            &wp  as *const _ as *mut c_void,
            &rbp as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = 8 * (crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip, "gemv", "gemv_hfq4g256_moe_down_residual_scaled_k8_indexed", bytes,
        );
        let result = self.launch_maybe_blob(
            func_name,
            [grid_x, 8, 1], block, 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp); b.push_ptr(ip); b.push_ptr(wp);
                b.push_ptr(rbp); b.push_ptr(xrp);
                b.push_i32(m_val); b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// N-batched MoE softmax + top-K + renorm. Grid = (N, 1, 1); one
    /// workgroup per token. `logits` is [N × n_exp], `topk_idx` is
    /// [N × K_TOP] i32, `topk_w` is [N × K_TOP] f32.
    pub fn moe_softmax_topk_renorm_k8_batched(
        &mut self,
        logits: &GpuTensor,
        topk_idx: &GpuTensor,
        topk_w:   &GpuTensor,
        n_exp: usize,
        norm_topk: bool,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "moe_softmax_topk_k8_batched",
            kernels::MOE_SOFTMAX_TOPK_K8_BATCHED_SRC,
            "moe_softmax_topk_renorm_k8_batched",
        )?;
        let lp = logits.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let wp = topk_w.buf.as_ptr();
        let n  = n_exp as i32;
        let nr = if norm_topk { 1i32 } else { 0i32 };
        let mut params: Vec<*mut c_void> = vec![
            &lp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &n  as *const _ as *mut c_void,
            &nr as *const _ as *mut c_void,
        ];
        let bytes = (n_exp * 4 + 8 * 8) * batch_size;
        let timer = crate::profile::begin_timer(
            &self.hip, "elementwise", "moe_softmax_topk_renorm_k8_batched", bytes,
        );
        let result = self.launch_maybe_blob(
            "moe_softmax_topk_renorm_k8_batched",
            [batch_size as u32, 1, 1], [256, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(lp); b.push_ptr(ip); b.push_ptr(wp);
                b.push_i32(n); b.push_i32(nr);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched companion of `moe_topk_renorm_k8` for the prefill path.
    /// Takes pre-softmaxed probs of shape `[batch_size × n_exp]` and writes
    /// `[batch_size × K_TOP]` indices and weights. Caller must run a batched
    /// softmax (`gpu.softmax_f32` on a [batch_size × n_exp] tensor) before
    /// calling this kernel.
    pub fn moe_topk_renorm_k8_batched(
        &mut self,
        probs: &GpuTensor,
        topk_idx: &GpuTensor,
        topk_w:   &GpuTensor,
        n_exp: usize,
        norm_topk: bool,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "moe_topk_renorm_k8_batched",
            kernels::MOE_TOPK_RENORM_K8_BATCHED_SRC,
            "moe_topk_renorm_k8_batched",
        )?;
        let lp = probs.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let wp = topk_w.buf.as_ptr();
        let n  = n_exp as i32;
        let nr = if norm_topk { 1i32 } else { 0i32 };
        let mut params: Vec<*mut c_void> = vec![
            &lp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &n  as *const _ as *mut c_void,
            &nr as *const _ as *mut c_void,
        ];
        let bytes = (n_exp * 4 + 8 * 8) * batch_size;
        let timer = crate::profile::begin_timer(
            &self.hip, "elementwise", "moe_topk_renorm_k8_batched", bytes,
        );
        let result = self.launch_maybe_blob(
            "moe_topk_renorm_k8_batched",
            [batch_size as u32, 1, 1], [256, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(lp); b.push_ptr(ip); b.push_ptr(wp);
                b.push_i32(n); b.push_i32(nr);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// N-batched indexed MoE gate_up. Grid = (M, K_TOP, N). `x` is
    /// [N × K], `topk_indices` is [N × K_TOP] i32, `y_gate` and `y_up`
    /// are [N × K_TOP × MI] where MI = M / 2.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up:   &GpuTensor,
        m: usize, k: usize, k_top: usize, batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let cdna_wave64 = has_wave64_native(&self.arch);
        let (func_name, block, grid_div): (&str, [u32; 3], u32) = if cdna_wave64 {
            self.ensure_kernel(
                "gemv_hfq4g256_moe_gate_up_indexed_batched_wave64",
                kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_BATCHED_WAVE64_SRC,
                "gemv_hfq4g256_moe_gate_up_k8_indexed_batched_wave64",
            )?;
            ("gemv_hfq4g256_moe_gate_up_k8_indexed_batched_wave64", [64, 1, 1], 2)
        } else {
            self.ensure_kernel(
                "gemv_hfq4g256_moe_gate_up_indexed_batched",
                kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_BATCHED_SRC,
                "gemv_hfq4g256_moe_gate_up_k8_indexed_batched",
            )?;
            ("gemv_hfq4g256_moe_gate_up_k8_indexed_batched", [32, 1, 1], 1)
        };
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let kt_val = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &ygp as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
        ];
        let bytes = batch_size * k_top * (crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip, "gemv", "gemv_hfq4g256_moe_gate_up_k8_indexed_batched", bytes,
        );
        let grid_x = (m as u32 + grid_div - 1) / grid_div;
        let result = self.launch_maybe_blob(
            func_name,
            [grid_x, k_top as u32, batch_size as u32], block, 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp); b.push_ptr(ip); b.push_ptr(xp);
                b.push_ptr(ygp); b.push_ptr(yup);
                b.push_i32(m_val); b.push_i32(k_val); b.push_i32(kt_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// N-batched indexed MoE down + scaled residual. Grid = (M, K_TOP, N).
    /// `rot_batch` is [N × K_TOP × K], `x_residual` is [N × M]; the kernel
    /// atomicAdd's per-token slices. `topk_indices` / `topk_weights` are
    /// [N × K_TOP].
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        topk_weights: &GpuTensor,
        rot_batch: &GpuTensor,
        x_residual: &GpuTensor,
        m: usize, k: usize, k_top: usize, batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let cdna_wave64 = has_wave64_native(&self.arch);
        let (func_name, block, grid_div): (&str, [u32; 3], u32) = if cdna_wave64 {
            self.ensure_kernel(
                "gemv_hfq4g256_moe_down_indexed_batched_wave64",
                kernels::GEMV_HFQ4G256_MOE_DOWN_INDEXED_BATCHED_WAVE64_SRC,
                "gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched_wave64",
            )?;
            ("gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched_wave64", [64, 1, 1], 2)
        } else {
            self.ensure_kernel(
                "gemv_hfq4g256_moe_down_indexed_batched",
                kernels::GEMV_HFQ4G256_MOE_DOWN_INDEXED_BATCHED_SRC,
                "gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched",
            )?;
            ("gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched", [32, 1, 1], 1)
        };
        let pp  = expert_ptrs.buf.as_ptr();
        let ip  = topk_indices.buf.as_ptr();
        let wp  = topk_weights.buf.as_ptr();
        let rbp = rot_batch.buf.as_ptr();
        let xrp = x_residual.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let kt_val = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp  as *const _ as *mut c_void,
            &ip  as *const _ as *mut c_void,
            &wp  as *const _ as *mut c_void,
            &rbp as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
        ];
        let bytes = batch_size * k_top * (crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip, "gemv", "gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched", bytes,
        );
        let grid_x = (m as u32 + grid_div - 1) / grid_div;
        let result = self.launch_maybe_blob(
            func_name,
            [grid_x, k_top as u32, batch_size as u32], block, 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp); b.push_ptr(ip); b.push_ptr(wp);
                b.push_ptr(rbp); b.push_ptr(xrp);
                b.push_i32(m_val); b.push_i32(k_val); b.push_i32(kt_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched HFQ4-G256 GEMM with fused residual add:
    ///   for b in 0..batch_size: y[b][row] += A[row] · x[b]
    ///
    /// Bitwise-identical output to calling `gemv_hfq4g256_residual` N times
    /// (preserves the 4-accumulator interleave and pairwise final combine),
    /// so safe to use in the quality-gated forward path. Each block handles
    /// one row × up to BATCH_TILE batch elements, amortizing the weight
    /// fetch across the batch loop.
    ///
    /// `x`: [batch_size × K] row-major, `y`: [batch_size × M] row-major.
    /// `y` must already hold the residual summand to accumulate into.
    pub fn gemm_hfq4g256_residual(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // CDNA3 MFMA path — Y += X·W^T via rocBLAS with beta=1.
        if self.rocblas_arch_eligible()
            && batch_size >= self.rocblas_min_batch()
            && self.rocblas.is_some()
            && !self.capture_mode
        {
            if let Ok(Some(shadow_ptr)) = self.ensure_fp16_shadow(a_raw, m, k) {
                let x_fp16 = self.ensure_fp16_x(x, batch_size * k)?;
                let w_buf = unsafe { DeviceBuffer::from_raw(shadow_ptr, (m * k) * 2) };
                let x_buf = unsafe { DeviceBuffer::from_raw(x_fp16, (batch_size * k) * 2) };
                let bytes = crate::profile::gemv_hfq4g256_bytes(m, k)
                    + batch_size * k * 4 + batch_size * m * 4 * 2;
                let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq4g256_residual_rocblas", bytes);
                let result = self.rocblas_gemm_hfq4_prefill_residual(
                    &w_buf, &x_buf, &y.buf, m, batch_size, k,
                );
                std::mem::forget(w_buf);
                std::mem::forget(x_buf);
                if let Some(t) = timer { t.finish(&self.hip); }
                return result;
            }
        }

        // Fast paths for prefill (batch_size > 1). Disable with HIPFIRE_FP16=0.
        if batch_size > 1 && !std::env::var("HIPFIRE_FP16").map_or(false, |v| v == "0") {
            // gfx906 dp4a MMQ residual path — default-on at batch ≥ 16.
            if self.arch == "gfx906" && should_use_mmq(&self.arch, batch_size) {
                let use_mmq = if self.mmq_screen {
                    self.mmq_screen_weight(a_raw, m, k)
                } else {
                    true
                };
                if use_mmq {
                    return self.gemm_hfq4g256_residual_mmq_gfx906(a_raw, x, y, m, k, batch_size);
                }
            }

            // Wave64 FP16 hybrid — best of both worlds for gfx906 (MI50).
            if is_gcn5_wave64(&self.arch) {
                return self.gemm_hfq4g256_residual_fp16_wave64(a_raw, x, y, m, k, batch_size);
            }

            // Opt-in MMQ path (RDNA3/3.5, HIPFIRE_MMQ=1 or HIPFIRE_WO_MMQ=1).
            if std::env::var("HIPFIRE_WO_MMQ").ok().as_deref() == Some("1")
                || should_use_mmq(&self.arch, batch_size)
            {
                let use_mmq = if self.mmq_screen {
                    self.mmq_screen_weight(a_raw, m, k)
                } else {
                    true
                };
                if use_mmq {
                    return self.gemm_hfq4g256_residual_mmq(a_raw, x, y, m, k, batch_size);
                }
            }

            // WMMA on gfx12 (RDNA4): K2-unroll port
            if has_wmma_f16_gfx12(&self.arch) {
                return self.gemm_hfq4g256_residual_wmma_gfx12(a_raw, x, y, m, k, batch_size);
            }

            // WMMA on gfx11+ (RDNA3): 16×16 tiled, ~8-10× over scalar
            if self.arch.starts_with("gfx11") {
                return self.gemm_hfq4g256_residual_wmma(a_raw, x, y, m, k, batch_size);
            }

            // FP16 packed on all other RDNA: ~15% prefill improvement
            return self.gemm_hfq4g256_residual_fp16(a_raw, x, y, m, k, batch_size);
        }

        let cdna_wave64 = has_wave64_native(&self.arch);
        let (func_name, block, grid_div): (&str, [u32; 3], u32) = if cdna_wave64 {
            self.ensure_kernel(
                "gemm_hfq4g256_residual_wave64",
                kernels::GEMM_HFQ4G256_RESIDUAL_WAVE64_SRC,
                "gemm_hfq4g256_residual_wave64",
            )?;
            ("gemm_hfq4g256_residual_wave64", [64, 1, 1], 2)
        } else {
            self.ensure_kernel(
                "gemm_hfq4g256_residual",
                kernels::GEMM_HFQ4G256_RESIDUAL_SRC,
                "gemm_hfq4g256_residual",
            )?;
            ("gemm_hfq4g256_residual", [32, 1, 1], 1)
        };
        let func = &self.functions[func_name];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = { const BATCH_TILE: usize = 8; (batch_size + BATCH_TILE - 1) / BATCH_TILE };
        let grid_x = (m as u32 + grid_div - 1) / grid_div;

        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k)
            + batch_size * k * 4
            + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq4g256_residual", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, batch_tiles as u32, 1],
                block,
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// FP16-input batched HFQ4-G256 GEMM with residual add.
    /// Converts X from FP32 to FP16 (halving X bandwidth), then runs the
    /// FP16-packed GEMM kernel. The conversion is a one-shot pass amortized
    /// across M rows.
    pub fn gemm_hfq4g256_residual_fp16(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,  // FP32 [batch_size × K]
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemm_hfq4g256_residual_fp16", kernels::GEMM_HFQ4G256_RESIDUAL_FP16_SRC, "gemm_hfq4g256_residual_fp16")?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        // FP16 GEMM
        let func = &self.functions["gemm_hfq4g256_residual_fp16"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = { const BATCH_TILE: usize = 8; (batch_size + BATCH_TILE - 1) / BATCH_TILE };

        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k)
            + batch_size * k * 2  // FP16 X (half bandwidth!)
            + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq4g256_residual_fp16", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, batch_tiles as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Wave64 FP16 hybrid batched HFQ4-G256 GEMM with fused residual add.
    /// Combines wave64 block structure (2 rows/block, full lane utilization) with
    /// FP16 packed arithmetic (__hfma2). Target: gfx906 (MI50) prefill optimization.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_hfq4g256_residual_fp16_wave64(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,  // FP32 [batch_size × K]
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemm_hfq4g256_residual_fp16_wave64", kernels::GEMM_HFQ4G256_RESIDUAL_FP16_WAVE64_SRC, "gemm_hfq4g256_residual_fp16_wave64")?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let func = &self.functions["gemm_hfq4g256_residual_fp16_wave64"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = { const BATCH_TILE: usize = 8; (batch_size + BATCH_TILE - 1) / BATCH_TILE };
        let grid_x = (m as u32 + 1) / 2;

        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k)
            + batch_size * k * 2  // FP16 X (half bandwidth!)
            + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq4g256_residual_fp16_wave64", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, batch_tiles as u32, 1],
                [64, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Experimental llama.cpp-style MMQ residual GEMM for HFQ4-G256.
    /// Opt-in only via `HIPFIRE_WO_MMQ=1` while the tiled path is validated.
    pub fn gemm_hfq4g256_residual_mmq(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let x_q8_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        let kernel_name = if m % 128 == 0 && batch_size % 128 == 0 {
            "gemm_hfq4g256_residual_mmq_full_add"
        } else {
            "gemm_hfq4g256_residual_mmq"
        };
        self.ensure_kernel(
            "gemm_hfq4g256_residual_mmq",
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_SRC,
            kernel_name,
        )?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut xq_ptr = x_q8_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;
        let mut add_val = 1i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut add_val as *mut _ as *mut c_void,
        ];

        const MMQ_X: usize = 128;
        const MMQ_Y: usize = 128;
        const MMQ_TILE_Y_K: usize = 36;
        const MMQ_TILE_X_K: usize = 76;
        let row_tiles = (m + MMQ_Y - 1) / MMQ_Y;
        let batch_tiles = (batch_size + MMQ_X - 1) / MMQ_X;
        let shared_mem = ((MMQ_X * MMQ_TILE_Y_K + MMQ_Y * MMQ_TILE_X_K) * std::mem::size_of::<i32>()) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k)
            + batch_size * k
            + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq4g256_residual_mmq", bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 8, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(xq_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b.push_i32(add_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// gfx906 dp4a MMQ residual GEMM. Wave-native topology (block 64×2,
    /// tile 128×64) per llama.cpp-gfx906 reference. Distinct from the
    /// RDNA3 i8-WMMA variant above — different block dim, different
    /// LDS layout, different kernel symbols.
    ///
    /// Phase 1 implementation; opt-in via `HIPFIRE_MMQ=1` while correctness
    /// is being validated. See plans/gfx906_mmq_plan.md and
    /// plans/p1.2_dp4a_mmq_design.md.
    pub fn gemm_hfq4g256_residual_mmq_gfx906(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Quantize activations to Q8_1.
        let x_q8_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;

        // Diagnostic: HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY=1 isolates the cost of
        // the Q8_1 activation pre-quantize by running the FP16 wave64 path
        // *after* paying the quantize cost. The flag is read once at init
        // (see `Gpu::new`) so this check is a single bool load, not a
        // per-call env::var lookup.
        if self.mmq_diag_quantize_only {
            let _ = x_q8_ptr;
            return self.gemm_hfq4g256_residual_fp16_wave64(a_raw, x, y, m, k, batch_size);
        }

        // Greedy mmq_x selection matching stock.
        let mmq_x = if batch_size <= 8 { 8 }
            else if batch_size <= 16 { 16 }
            else if batch_size <= 24 { 24 }
            else if batch_size <= 32 { 32 }
            else if batch_size <= 40 { 40 }
            else if batch_size <= 48 { 48 }
            else if batch_size <= 56 { 56 }
            else { 64 };

        // Pick variant name and source.
        let is_full = m % 128 == 0 && batch_size % mmq_x == 0;
        let base_name = "gemm_hfq4g256_residual_mmq_gfx906";
        let kernel_name = if is_full {
            format!("{}_full_add_x{}", base_name, mmq_x)
        } else {
            format!("{}_x{}", base_name, mmq_x)
        };

        let wrapper_src = match mmq_x {
            8  => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X8_SRC,
            16 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X16_SRC,
            24 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X24_SRC,
            32 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X32_SRC,
            40 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X40_SRC,
            48 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X48_SRC,
            56 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X56_SRC,
            64 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X64_SRC,
            _ => unreachable!(),
        };
        // Inline the body .cuh: the runtime hipcc compiles from cache_dir,
        // which doesn't have kernels/src on its -I path. Strip the
        // `#include "..._body.cuh"` line and prepend the body content.
        let inlined = wrapper_src.replace(
            "#include \"gemm_hfq4g256_residual_mmq_gfx906_body.cuh\"",
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_BODY_CUH,
        );

        self.ensure_kernel(&format!("{}_x{}", base_name, mmq_x), &inlined, &kernel_name)?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut xq_ptr = x_q8_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;
        let mut add_val = 1i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut add_val as *mut _ as *mut c_void,
        ];

        // Option C streaming topology — KEEP IN SYNC WITH body.cuh:
        //   x_qs   : MMQ_Y * x_stride ints  (per-mmq_x: 40 if mmq_x≥64 else 33)
        //   x_dm   : MMQ_Y float2
        //   tile_y : mmq_x * Y_STRIDE ints
        const MMQ_Y: usize = 128;
        let x_stride: usize = if mmq_x >= 64 { 40 } else { 33 };
        const Y_STRIDE: usize = 36;
        const X_DM_HALF2: usize = 128;
        let row_tiles = (m + MMQ_Y - 1) / MMQ_Y;
        let batch_tiles = (batch_size + mmq_x - 1) / mmq_x;

        let shared_mem = (
            (MMQ_Y * x_stride * 4)
            + (X_DM_HALF2 * 8)
            + (mmq_x * Y_STRIDE * 4)
        ) as u32;
        // 2 WGs/CU on gfx906 needs ≤32 KiB/WG (64 KiB cap).
        debug_assert!(shared_mem as usize <= 32 * 1024,
            "gfx906 MMQ LDS budget exceeded: {} B > 32 KiB", shared_mem);

        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k)
            + batch_size * k
            + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", base_name, bytes);
        let result = self.launch_maybe_blob(
            &kernel_name,
            [row_tiles as u32, batch_tiles as u32, 1],
            [64, 4, 1], // nwarps=4
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr); b.push_ptr(xq_ptr); b.push_ptr(y_ptr);
                b.push_i32(m_val); b.push_i32(k_val); b.push_i32(n_val);
                b.push_i32(add_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Set-mode (add=0) variant of the gfx906 MMQ kernel.
    pub fn gemm_hfq4g256_mmq_set_gfx906(
        &mut self,
        a_raw: &GpuTensor,
        x_q8_ptr: *mut c_void,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let mmq_x = if batch_size <= 8 { 8 }
            else if batch_size <= 16 { 16 }
            else if batch_size <= 24 { 24 }
            else if batch_size <= 32 { 32 }
            else if batch_size <= 40 { 40 }
            else if batch_size <= 48 { 48 }
            else if batch_size <= 56 { 56 }
            else { 64 };

        let is_full = m % 128 == 0 && batch_size % mmq_x == 0;
        let base_name = "gemm_hfq4g256_residual_mmq_gfx906";
        let kernel_name = if is_full {
            format!("{}_full_set_x{}", base_name, mmq_x)
        } else {
            format!("{}_x{}", base_name, mmq_x)
        };

        let wrapper_src = match mmq_x {
            8  => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X8_SRC,
            16 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X16_SRC,
            24 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X24_SRC,
            32 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X32_SRC,
            40 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X40_SRC,
            48 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X48_SRC,
            56 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X56_SRC,
            64 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X64_SRC,
            _ => unreachable!(),
        };
        let inlined = wrapper_src.replace(
            "#include \"gemm_hfq4g256_residual_mmq_gfx906_body.cuh\"",
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_BODY_CUH,
        );

        self.ensure_kernel(&format!("{}_x{}", base_name, mmq_x), &inlined, &kernel_name)?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut xq_ptr = x_q8_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;
        let mut add_val = 0i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut add_val as *mut _ as *mut c_void,
        ];

        // Option C streaming topology — KEEP IN SYNC WITH body.cuh
        // (same layout invariant as residual variant above).
        const MMQ_Y: usize = 128;
        let x_stride: usize = if mmq_x >= 64 { 40 } else { 33 };
        const Y_STRIDE: usize = 36;
        const X_DM_HALF2: usize = 128;
        let row_tiles = (m + MMQ_Y - 1) / MMQ_Y;
        let batch_tiles = (batch_size + mmq_x - 1) / mmq_x;

        let shared_mem = (
            (MMQ_Y * x_stride * 4)
            + (X_DM_HALF2 * 8)
            + (mmq_x * Y_STRIDE * 4)
        ) as u32;
        debug_assert!(shared_mem as usize <= 32 * 1024,
            "gfx906 MMQ LDS budget exceeded: {} B > 32 KiB", shared_mem);

        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k)
            + batch_size * m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq4g256_mmq_set_gfx906", bytes);
        let result = self.launch_maybe_blob(
            &kernel_name,
            [row_tiles as u32, batch_tiles as u32, 1],
            [64, 4, 1], // nwarps=4
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr); b.push_ptr(xq_ptr); b.push_ptr(y_ptr);
                b.push_i32(m_val); b.push_i32(k_val); b.push_i32(n_val);
                b.push_i32(add_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    pub fn gemm_hfq4g256_mmq_set(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let x_q8_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        let kernel_name = if m % 128 == 0 && batch_size % 128 == 0 {
            "gemm_hfq4g256_residual_mmq_full_set"
        } else {
            "gemm_hfq4g256_residual_mmq"
        };
        self.ensure_kernel(
            "gemm_hfq4g256_residual_mmq",
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_SRC,
            kernel_name,
        )?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut xq_ptr = x_q8_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;
        let mut add_val = 0i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut add_val as *mut _ as *mut c_void,
        ];

        const MMQ_X: usize = 128;
        const MMQ_Y: usize = 128;
        const MMQ_TILE_Y_K: usize = 36;
        const MMQ_TILE_X_K: usize = 76;
        let row_tiles = (m + MMQ_Y - 1) / MMQ_Y;
        let batch_tiles = (batch_size + MMQ_X - 1) / MMQ_X;
        let shared_mem = ((MMQ_X * MMQ_TILE_Y_K + MMQ_Y * MMQ_TILE_X_K) * std::mem::size_of::<i32>()) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k)
            + batch_size * k
            + batch_size * m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq4g256_mmq_set", bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 8, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(xq_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b.push_i32(add_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    pub fn gemm_hfq4g256_mmq_set_prequant(
        &mut self,
        a_raw: &GpuTensor,
        x_q8_ptr: *mut c_void,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch == "gfx906" {
            // gfx906 has its own dispatcher (`gemm_hfq4g256_residual_mmq_gfx906`)
            // that handles its own quantize internally, called directly from
            // mmq_screen_weight on gfx906. _set_prequant is RDNA3-only.
            return Err(hip_bridge::HipError::new(0,
                "gemm_hfq4g256_mmq_set_prequant is not supported on gfx906; \
                 callers should route to gemm_hfq4g256_residual_mmq_gfx906 directly"));
        }
        let kernel_name = if m % 128 == 0 && batch_size % 128 == 0 {
            "gemm_hfq4g256_residual_mmq_full_set"
        } else {
            "gemm_hfq4g256_residual_mmq"
        };
        self.ensure_kernel(
            "gemm_hfq4g256_residual_mmq",
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_SRC,
            kernel_name,
        )?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut xq_ptr = x_q8_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;
        let mut add_val = 0i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut add_val as *mut _ as *mut c_void,
        ];

        const MMQ_X: usize = 128;
        const MMQ_Y: usize = 128;
        const MMQ_TILE_Y_K: usize = 36;
        const MMQ_TILE_X_K: usize = 76;
        let row_tiles = (m + MMQ_Y - 1) / MMQ_Y;
        let batch_tiles = (batch_size + MMQ_X - 1) / MMQ_X;
        let shared_mem = ((MMQ_X * MMQ_TILE_Y_K + MMQ_Y * MMQ_TILE_X_K) * std::mem::size_of::<i32>()) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k)
            + batch_size * m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq4g256_mmq_set", bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 8, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(xq_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b.push_i32(add_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// WMMA-accelerated batched HFQ4-G256 GEMM with residual add.
    /// gfx1100+ only. 16×16 output tiles via wave32 WMMA.
    /// Converts X to FP16, then uses __builtin_amdgcn_wmma_f32_16x16x16_f16_w32.
    pub fn gemm_hfq4g256_residual_wmma(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Compile both kernels (convert + WMMA GEMM share the FP16 convert)
        // Kernel variant selection
        // MW16 path: dequant weights to FP16 per-call, then run no-dequant WMMA
        if std::env::var("HIPFIRE_MW16").map_or(false, |v| v == "1") {
            return self.gemm_mw16_residual_wmma_via_dequant(a_raw, x, y, m, k, batch_size);
        }
        // Shape-aware default: ksplit only pays for itself when the un-split
        // grid is CU-starved (target wo_residual at M=5120 → 320 blocks,
        // ~3.3/CU on gfx1100 — ksplit 4×'s it to 13/CU). For draft-FFN shapes
        // (M=17408, K=5120, B=16) the un-split grid is already 1088 blocks
        // (~11/CU) and the atomicAdd reduce is pure overhead. k2 removes the
        // split + atomics and runs deterministically.
        //
        // Threshold picked at M=8192: covers M∈{5120,6144} (target wo) on the
        // ksplit side and M∈{17408} (draft gate/up/down) on the k2 side. lm_head
        // (M=vocab) is always way above threshold → k2.
        //
        // HIPFIRE_WO_WMMA_VARIANT=ksplit|k2|k2x32|k4|wmma|wmma2 overrides the
        // auto selection (applies to every call, both target and draft).
        //   ksplit — K-split + atomicAdd (non-deterministic accum order)
        //   k2     — 2× K-tile pipeline (byte-exact accum order)
        //   k2x32  — 32-row block with shared X fragment per K-tile. Slower
        //            than k2 on gfx1100, but faster on gfx1151 Strix Halo for
        //            small-M residual projections at prefill-sized batches.
        //            DFlash verify/lm_head runs at B<=16 and large-M draft
        //            FFN/lm_head also prefer k2.
        //   k4     — 4× K-tile pipeline. Fixed 2026-05-01 (commit pending):
        //            output mapping was swapped relative to K2's canonical
        //            wave32 WMMA C-mapping. Channel-test passes at K∈{256,512,4096}
        //            × batch∈{1,2,4,16}. At m<8192 (9B residual at m=4096) K4
        //            ties K2 within FP drift but loses to ksplit by ~33%
        //            per-call at small batch (CU-starved grid: 3.3 vs 13
        //            blocks/CU); auto-dispatch correctly stays on ksplit. K4
        //            vs K2 at m≥8192 not yet benched on available models. See
        //            plans/k4_plan.md.
        //   wmma   — base WMMA         (output-mapping bug — debug only)
        //   wmma2  — 2-wave block, 32 rows × 16 batch (output-mapping bug — debug only)
        let is_gfx115x = matches!(self.arch.as_str(), "gfx1150" | "gfx1151" | "gfx1152");
        // ksplit's atomicAdd reduction across K_SPLITS partials is fp-non-
        // associative — order varies with warp scheduling, so output bytes
        // drift between processes and between cold/hot runs. The drift is
        // sub-argmax-margin per call but cascades on long greedy decode
        // (>50 tokens). HIPFIRE_DETERMINISTIC=1 forces k2 (single-block
        // K reduction) at the cost of ~33% perf on small-batch / small-M.
        // Required when chasing multi-GPU parity: pp=1 vs pp=2 outputs
        // can't be compared byte-for-byte when the underlying single-GPU
        // path itself is non-deterministic.
        // Cached — getenv on every decode token would re-parse 6× per layer
        // × N layers per step. Read once at first dispatch.
        static FORCE_DET: OnceLock<bool> = OnceLock::new();
        let force_det = *FORCE_DET.get_or_init(|| {
            std::env::var("HIPFIRE_DETERMINISTIC").ok().as_deref() == Some("1")
        });
        let auto_variant = if force_det {
            "k2"
        } else if is_gfx115x && batch_size <= 16 {
            "k2"
        } else if is_gfx115x && m < 8192 {
            "k2x32"
        } else if m >= 8192 {
            "k2"
        } else {
            "ksplit"
        };
        let variant_override = std::env::var("HIPFIRE_WO_WMMA_VARIANT").ok();
        let variant = variant_override.as_deref().unwrap_or(auto_variant);
        let (kernel_name, kernel_src, block_size, row_step, k_splits) = match variant {
            "k2"     => ("gemm_hfq4g256_residual_wmma_k2",
                         kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_K2_SRC, 32u32, 16usize, 1u32),
            "k2x32"  => ("gemm_hfq4g256_residual_wmma_k2x32",
                         kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_K2X32_SRC, 32u32, 32usize, 1u32),
            "k4"     => ("gemm_hfq4g256_residual_wmma_k4",
                         kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_K4_SRC, 32u32, 16usize, 1u32),
            "wmma"   => ("gemm_hfq4g256_residual_wmma",
                         kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_SRC, 32u32, 16usize, 1u32),
            "wmma2"  => ("gemm_hfq4g256_residual_wmma2",
                         kernels::GEMM_HFQ4G256_RESIDUAL_WMMA2_SRC, 64u32, 32usize, 1u32),
            _        => ("gemm_hfq4g256_residual_wmma_ksplit",
                         kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_KSPLIT_SRC, 32u32, 16usize, 4u32),
        };
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        let row_tiles = (m + row_step - 1) / row_step;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k)
            + batch_size * k * 2
            + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        // HIPFIRE_GEMM_DUMP=1: per-call shape+wall-clock dump of this kernel.
        // Synchronously times only the ksplit kernel launch (not memset / convert).
        // Measures actual GPU execution time via device_synchronize pre+post —
        // costs latency vs async pipelining but gives shape-accurate µs.
        static DUMP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let dump = *DUMP.get_or_init(|| {
            std::env::var("HIPFIRE_GEMM_DUMP").ok().as_deref() == Some("1")
        });
        if dump { self.hip.device_synchronize()?; }
        let dump_start = if dump { Some(std::time::Instant::now()) } else { None };
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles as u32, batch_tiles as u32, k_splits],
            [block_size, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr); b.push_ptr(x_ptr); b.push_ptr(y_ptr);
                b.push_i32(m_val); b.push_i32(k_val); b.push_i32(bs_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        if let Some(t) = dump_start {
            self.hip.device_synchronize()?;
            let us = t.elapsed().as_micros();
            let gbs = (bytes as f64) / (us.max(1) as f64) / 1000.0; // MB/ms == GB/s
            eprintln!("[gemm-dump] {} M={} K={} B={} bytes={}KB us={} GB/s={:.1}",
                kernel_name, m, k, batch_size, bytes / 1024, us, gbs);
        }
        result
    }

    /// HFQ3-G256 sister of `gemm_hfq4g256_residual_wmma` (basic WMMA
    /// variant). Same WMMA shape + lane decomposition; only the inner
    /// K-tile unpack differs (3-bit cross-byte vs 4-bit nibble) and the
    /// per-group byte stride is 104 instead of 136. Y += acc[j] (fused
    /// residual add — caller must initialize Y with the residual stream
    /// before launching).
    pub fn gemm_hfq3g256_residual_wmma(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if has_wmma_f16_gfx12(&self.arch) {
            return self.gemm_hfq3g256_residual_wmma_gfx12(a_raw, x, y, m, k, batch_size);
        }
        self.ensure_kernel("gemm_hfq3g256_residual_wmma", kernels::GEMM_HFQ3G256_RESIDUAL_WMMA_SRC, "gemm_hfq3g256_residual_wmma")?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        let row_tiles = (m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let weight_bytes = m * (k / 256) * 104;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq3g256_residual_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_hfq3g256_residual_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr); b.push_ptr(x_ptr); b.push_ptr(y_ptr);
                b.push_i32(m_val); b.push_i32(k_val); b.push_i32(bs_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// MQ3 wrapper for `gemm_hfq3g256_residual_wmma`: pre-rotates X then
    /// dispatches the HFQ3 kernel. See `gemm_qkvza_mq3g256_wmma` for
    /// the cache-invalidation rationale.
    /// gfx12 (RDNA4) sister of `gemm_hfq3g256_residual_wmma`.
    pub fn gemm_hfq3g256_residual_wmma_gfx12(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfq3g256_residual_wmma_gfx12",
            kernels::GEMM_HFQ3G256_RESIDUAL_WMMA_GFX12_SRC,
            "gemm_hfq3g256_residual_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        let row_tiles = (m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let weight_bytes = m * (k / 256) * 104;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq3g256_residual_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_hfq3g256_residual_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr); b.push_ptr(x_ptr); b.push_ptr(y_ptr);
                b.push_i32(m_val); b.push_i32(k_val); b.push_i32(bs_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    pub fn gemm_mq3g256_residual_wmma(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        for b in 0..batch_size {
            let x_row = x.sub_offset(b * k, k);
            let x_rot_row = x_rot.sub_offset(b * k, k);
            self.rotate_x_mq(&x_row, &x_rot_row, k)?;
        }
        self.fp16_x_source_ptr = std::ptr::null_mut();
        self.gemm_hfq3g256_residual_wmma(a_raw, x_rot, y, m, k, batch_size)
    }

    /// MW16: dequant 4-bit weights to FP16, then run the no-dequant WMMA kernel.
    /// Per-call dequant (wasteful) — for benchmarking only. Production would
    /// dequant at model load time.
    fn gemm_mw16_residual_wmma_via_dequant(
        &mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor,
        m: usize, k: usize, batch_size: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("dequant_hfq4g256_to_f16", kernels::DEQUANT_HFQ4G256_TO_F16_SRC, "dequant_hfq4g256_to_f16")?;
        self.ensure_kernel("gemm_mw16_residual_wmma", kernels::GEMM_MW16_RESIDUAL_WMMA_SRC, "gemm_mw16_residual_wmma")?;
        let x_f16 = self.ensure_fp16_x(x, batch_size * k)?;

        // Dequant weights to FP16 scratch
        let w_elems = m * k;
        let w_f16 = self.hip.malloc(w_elems * 2)?;
        {
            let f = &self.functions["dequant_hfq4g256_to_f16"];
            let groups = k / 256;
            let mut ap = a_raw.buf.as_ptr(); let mut wp = w_f16.as_ptr();
            let mut mv = m as i32; let mut kv = k as i32;
            let mut p: Vec<*mut c_void> = vec![
                &mut ap as *mut _ as *mut c_void, &mut wp as *mut _ as *mut c_void,
                &mut mv as *mut _ as *mut c_void, &mut kv as *mut _ as *mut c_void,
            ];
            unsafe { self.hip.launch_kernel(f, [m as u32, groups as u32, 1], [32,1,1], 0, self.stream_ref(), &mut p)?; }
        }

        // MW16 WMMA GEMM
        let f = &self.functions["gemm_mw16_residual_wmma"];
        let mut wp = w_f16.as_ptr(); let mut xp = x_f16;
        let mut yp = y.buf.as_ptr();
        let mut mv = m as i32; let mut kv = k as i32; let mut nv = batch_size as i32;
        let mut p: Vec<*mut c_void> = vec![
            &mut wp as *mut _ as *mut c_void, &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mv as *mut _ as *mut c_void, &mut kv as *mut _ as *mut c_void,
            &mut nv as *mut _ as *mut c_void,
        ];
        let rows = (m + 15) / 16;
        let batches = (batch_size + 15) / 16;
        let bytes = m * k * 2 + batch_size * k * 2 + batch_size * m * 8;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_mw16_residual_wmma", bytes);
        let result = unsafe {
            self.hip.launch_kernel(f, [rows as u32, batches as u32, 1], [32,1,1], 0, self.stream_ref(), &mut p)
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        drop(w_f16);
        result
    }

    /// Batched HFQ4-G256 GEMM: y[b][row] = A[row] · x[b] for all batch elements.
    /// x: [batch_size × K], y: [batch_size × M], both row-major.
    ///
    /// This is the portable scalar kernel — stays byte-exact with the AR
    /// greedy prefill's numerical baseline. For the DFlash lm_head fast
    /// path (batched, tolerates small FP16 drift for 8-10× speedup), use
    /// `gemm_hfq4g256_batched_lmhead` instead.
    pub fn gemm_hfq4g256(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // gfx906 dp4a opt-in for the LM-head batched GEMM. PMC at 2026-05-06
        // showed gemm_hfq4g256_wave64 was 17 % of DFlash 27B steady-state
        // decode time on the FP wave64 path. The dp4a port pre-quantizes x
        // to Q8_1 (shared scratch with the prefill MMQ + the gate_up/qkv/qkvza
        // GEMV ports) and runs v_dot4_i32_i8.
        //
        // Only fires on gfx906 (other wave64-native archs have rocBLAS or
        // larger MFMA paths that beat dp4a at large batches). Skip in
        // capture mode (matches the rocBLAS branch's caveat — Q8_1
        // quantize launch must be reachable from the captured graph or
        // pre-baked).
        if gemv_dp4a_enabled(&self.arch) && !self.capture_mode {
            return self.gemm_hfq4g256_dp4a(a_raw, x, y, m, k, batch_size);
        }

        // CDNA3 MFMA path (task #130): when rocBLAS is loaded and batch is
        // big enough for the launch overhead to amortize, route through the
        // dequantize-once FP16 shadow + rocBLAS GEMM. Expected 20-100× over
        // the wave64 GEMV on prefill-heavy workloads (sidecar cal, DFlash
        // target verify). Falls back to wave64 GEMV on: single-token decode
        // (batch<4), capture mode (rocBLAS launches don't graph-capture
        // cleanly; revisit if hipGraph becomes critical for CDNA3 prefill),
        // or if the fp16 shadow alloc fails under VRAM pressure.
        if self.rocblas_arch_eligible()
            && batch_size >= self.rocblas_min_batch()
            && self.rocblas.is_some()
            && !self.capture_mode
        {
            if let Ok(Some(shadow_ptr)) = self.ensure_fp16_shadow(a_raw, m, k) {
                // Convert X to FP16 via the existing ensure_fp16_x helper.
                let x_fp16 = self.ensure_fp16_x(x, batch_size * k)?;
                // Wrap the raw device pointers as non-owning DeviceBuffers so
                // the rocBLAS helper's signature works. The underlying memory
                // is owned by the fp16 shadow cache / fp16_x_scratch / caller's
                // y GpuTensor — all live beyond this call.
                let w_buf = unsafe { DeviceBuffer::from_raw(shadow_ptr, (m * k) * 2) };
                let x_buf = unsafe { DeviceBuffer::from_raw(x_fp16, (batch_size * k) * 2) };
                let bytes = crate::profile::gemm_hfq4g256_bytes(m, k, batch_size);
                let timer = crate::profile::begin_timer(&self.hip, "gemv", "gemm_hfq4g256_rocblas", bytes);
                let result = self.rocblas_gemm_hfq4_prefill(
                    &w_buf, &x_buf, &y.buf,
                    m, batch_size, k,
                );
                // Suppress the non-owning DeviceBuffer drop; HipError::Drop on
                // hip_free would clobber memory we don't own.
                std::mem::forget(w_buf);
                std::mem::forget(x_buf);
                if let Some(t) = timer { t.finish(&self.hip); }
                return result;
            }
            // Shadow allocation failed — fall through to the GEMV path.
        }

        let cdna_wave64 = has_wave64_native(&self.arch);
        let (func_name, block, grid_div): (&str, [u32; 3], u32) = if cdna_wave64 {
            self.ensure_kernel(
                "gemm_hfq4g256_wave64",
                kernels::GEMM_HFQ4G256_WAVE64_SRC,
                "gemm_hfq4g256_wave64",
            )?;
            ("gemm_hfq4g256_wave64", [64, 1, 1], 2)
        } else {
            self.ensure_kernel(
                "gemm_hfq4g256",
                kernels::GEMM_HFQ4G256_SRC,
                "gemm_hfq4g256",
            )?;
            ("gemm_hfq4g256", [32, 1, 1], 1)
        };

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = { const BATCH_TILE: usize = 8; ((batch_size + BATCH_TILE - 1) / BATCH_TILE) as u32 };
        let grid_x = (m as u32 + grid_div - 1) / grid_div;
        let bytes = crate::profile::gemm_hfq4g256_bytes(m, k, batch_size);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", "gemm_hfq4g256", bytes);
        let result = self.launch_maybe_blob(
            func_name,
            [grid_x, batch_tiles, 1],
            block,
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr); b.push_ptr(x_ptr); b.push_ptr(y_ptr);
                b.push_i32(m_val); b.push_i32(k_val); b.push_i32(bs_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// dp4a-port of gemm_hfq4g256 for gfx906. Pre-quantizes x to Q8_1 via
    /// the shared MMQ x-scratch (kblock-major: `[K/128, batch_size]`),
    /// then dispatches the wave64 dp4a GEMM. Math is identical modulo
    /// Q8_1 quant noise.
    ///
    /// Targets the LM-head batched GEMM hot path on DFlash 27B (PMC at
    /// 2026-05-06 showed 17 % of decode time was here on the FP path).
    /// Same Q8_1 layout as the prefill MMQ kernel + the four PR-158
    /// fused GEMVs, so `ensure_q8_1_mmq_x` reuses the existing scratch.
    pub fn gemm_hfq4g256_dp4a(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Quantize x → Xq[K/128 * batch_size] block_q8_1_mmq via the
        // shared scratch. Stride layout: kblock-major (matches
        // quantize_q8_1_mmq_ds4 at gemm_hfq4g256_residual_mmq.hip:80).
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;

        self.ensure_kernel(
            "gemm_hfq4g256_wave64_dp4a",
            kernels::GEMM_HFQ4G256_WAVE64_DP4A_SRC,
            "gemm_hfq4g256_wave64_dp4a",
        )?;

        let a_ptr = a_raw.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let bs_val = batch_size as i32;
        let mut xq = xq_ptr;
        let grid_x = (m as u32 + 1) / 2;
        const BATCH_TILE: usize = 8;
        let grid_y = ((batch_size + BATCH_TILE - 1) / BATCH_TILE) as u32;

        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &bs_val as *const _ as *mut c_void,
        ];

        let bytes = crate::profile::gemm_hfq4g256_bytes(m, k, batch_size);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", "gemm_hfq4g256_dp4a", bytes);
        let result = self.launch_maybe_blob(
            "gemm_hfq4g256_wave64_dp4a",
            [grid_x, grid_y, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr); b.push_ptr(xq); b.push_ptr(y_ptr);
                b.push_i32(m_val); b.push_i32(k_val); b.push_i32(bs_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// FP16-weight lm_head fast path for DFlash drafts that ship F16 (not
    /// quantized) weights. Routes through `gemm_mw16_residual_wmma` with the
    /// usual memset-then-atomicAdd residual pattern.
    ///
    /// Shape requirements: K must be a multiple of 32 (mw16 processes 32 K
    /// elements per WMMA iteration). All 27B/9B draft shapes satisfy this
    /// (hidden=5120, intermediate=17408, q_dim=4096, kv_dim=1024, fc-K=25600).
    ///
    /// Non-gfx11 falls through to `gemm_f32_batched` — same semantics but
    /// weight is read as F16 bytes, so the caller must have uploaded it that
    /// way. (Currently only gfx11 is expected to hit this path; other archs
    /// should use MQ4/HFQ4 drafts.)
    pub fn gemm_f16_batched_lmhead(
        &mut self,
        w_f16: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch.starts_with("gfx11") {
            // No mw16 WMMA on non-RDNA3 — fall back to the scalar F32 GEMM.
            // This is slow but correct; non-gfx11 isn't the intended target.
            return self.gemm_f32_batched(x, w_f16, y, batch_size, k, m);
        }
        self.ensure_kernel(
            "gemm_mw16_residual_wmma",
            kernels::GEMM_MW16_RESIDUAL_WMMA_SRC,
            "gemm_mw16_residual_wmma",
        )?;
        // Pre-zero Y (residual WMMA does y += acc) and force FP16-X reconversion
        // (the draft reuses the same scratch pointer every cycle with new data).
        self.fp16_x_source_ptr = std::ptr::null_mut();
        match self.active_stream.as_ref() {
            Some(stream) => self.hip.memset_async(&y.buf, 0, batch_size * m * 4, stream)?,
            None => self.hip.memset(&y.buf, 0, batch_size * m * 4)?,
        }
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let func = &self.functions["gemm_mw16_residual_wmma"];
        let mut wp = w_f16.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yp = y.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut ni = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut wp as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        let rows = ((m + 15) / 16) as u32;
        let batches = ((batch_size + 15) / 16) as u32;
        // Bytes: FP16 weight + FP16 x + FP32 y (read+write).
        let bytes = m * k * 2 + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_mw16_residual_wmma", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func, [rows, batches, 1], [32, 1, 1], 0, self.stream_ref(), &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// WMMA lm_head fast path for DFlash. Computes y = A @ x at batch>1 via
    /// the residual-WMMA kernel on pre-zeroed y — 8-10× faster than the
    /// scalar `gemm_hfq4g256` on 9B lm_head (batch=16, vocab=248K, k=2560).
    ///
    /// NOT numerically identical to `gemm_hfq4g256`. Uses FP16 tensor cores
    /// with the accumulators in FP32 the residual kernel ships. On the
    /// DFlash target-verify + draft-lm_head hot path this is a win (~13 ms
    /// saved per cycle), and the small FP16 drift doesn't meaningfully
    /// affect greedy acceptance. Do NOT use for AR greedy prefill — it will
    /// break byte-exact quality-gate reproducibility.
    ///
    /// Fallbacks: non-gfx11 or HIPFIRE_FP16=0 or HIPFIRE_LM_HEAD_WMMA=0 →
    /// routes to plain `gemm_hfq4g256`.
    ///
    /// Subtle: the residual-WMMA kernel goes through `ensure_fp16_x`, which
    /// caches the FP32→FP16 conversion keyed on source pointer. DFlash
    /// callers reuse the SAME hidden buffer pointer every cycle (draft
    /// scratch sub-offset, verify's persistent final_hidden) but with NEW
    /// data — so the cache entry is silently stale. Stomp the cache pointer
    /// before the dispatch to force reconversion.
    pub fn gemm_hfq4g256_batched_lmhead(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let wmma_eligible = batch_size > 1
            && self.arch.starts_with("gfx11")
            && !std::env::var("HIPFIRE_FP16").map_or(false, |v| v == "0")
            && !std::env::var("HIPFIRE_LM_HEAD_WMMA").map_or(false, |v| v == "0");
        if wmma_eligible {
            self.fp16_x_source_ptr = std::ptr::null_mut();
            match self.active_stream.as_ref() {
                Some(stream) => self.hip.memset_async(&y.buf, 0, batch_size * m * 4, stream)?,
                None => self.hip.memset(&y.buf, 0, batch_size * m * 4)?,
            }
            return self.gemm_hfq4g256_residual_wmma(a_raw, x, y, m, k, batch_size);
        }
        self.gemm_hfq4g256(a_raw, x, y, m, k, batch_size)
    }

    /// HFQ6-G256 sister of `gemm_hfq4g256_batched_lmhead`. Phase A.4
    /// (plan v3.2.3 §5.1 item 4). On gfx906 uses the dp4a residual GEMM
    /// (Phase A.2) with a zero-init of Y, mirroring the HFQ4 WMMA pattern
    /// at line 8019-8022. Lets the residual `+=` collapse to `=` semantics
    /// without needing a separate non-residual kernel.
    ///
    /// Caller is responsible for FWHT-rotating x first when the weights
    /// are MQ6 (FWHT-rotated at quant time).
    pub fn gemm_hfq6g256_batched_lmhead(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // gfx906: dp4a residual + zero-init Y for `=` semantics.
        // Skip in capture mode (the residual kernel calls ensure_q8_1_mmq_x
        // which launches an internal quantize kernel — matches HFQ4 sibling).
        if batch_size > 1 && gemv_dp4a_enabled(&self.arch) && !self.capture_mode {
            match self.active_stream.as_ref() {
                Some(stream) => self.hip.memset_async(&y.buf, 0, batch_size * m * 4, stream)?,
                None => self.hip.memset(&y.buf, 0, batch_size * m * 4)?,
            }
            return self.gemm_hfq6g256_residual_wave64_dp4a(a_raw, x, y, m, k, batch_size);
        }
        // gfx11+: WMMA residual + zero-init.
        let wmma_eligible = batch_size > 1
            && self.arch.starts_with("gfx11")
            && !std::env::var("HIPFIRE_FP16").map_or(false, |v| v == "0")
            && !std::env::var("HIPFIRE_LM_HEAD_WMMA").map_or(false, |v| v == "0");
        if wmma_eligible {
            self.fp16_x_source_ptr = std::ptr::null_mut();
            match self.active_stream.as_ref() {
                Some(stream) => self.hip.memset_async(&y.buf, 0, batch_size * m * 4, stream)?,
                None => self.hip.memset(&y.buf, 0, batch_size * m * 4)?,
            }
            return self.gemm_hfq6g256_residual_wmma(a_raw, x, y, m, k, batch_size);
        }
        // Fallback: use the residual dispatcher with zero-init Y. This
        // routes to fp16-packed or scalar depending on arch.
        match self.active_stream.as_ref() {
            Some(stream) => self.hip.memset_async(&y.buf, 0, batch_size * m * 4, stream)?,
            None => self.hip.memset(&y.buf, 0, batch_size * m * 4)?,
        }
        self.gemm_hfq6g256_residual(a_raw, x, y, m, k, batch_size)
    }

    /// HFQ3-G256 sister of `gemm_hfq4g256_batched_lmhead`. Same FP16-X cache
    /// stomp + zero-init of Y, then `gemm_hfq3g256_residual_wmma` to compute
    /// y[b][row] = A[row] · x[b]. Used by `dflash::gemm_dispatch` for MQ3
    /// drafts so DFlash works with MQ3-quantized draft weights.
    ///
    /// Caller is responsible for FWHT-rotating x first when the weights are
    /// MQ3 (FWHT-rotated at quant time) — `dflash::gemm_dispatch` handles
    /// that via `rotate_x_mq_batched`. This wrapper is dtype-agnostic in
    /// the same sense as `gemm_hfq4g256_batched_lmhead`.
    pub fn gemm_hfq3g256_batched_lmhead(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // WMMA eligibility: any arch with an MQ3 WMMA family ported. Today
        // that's gfx11 (RDNA3, _w32 builtin) and gfx12 (RDNA4, _w32_gfx12
        // builtin) — `gemm_hfq3g256_residual_wmma` dispatches internally to
        // the correct variant per arch. Other archs (gfx10/906/94x) fall
        // through to the per-row GEMV path.
        let wmma_eligible = batch_size > 1
            && (has_wmma_f16(&self.arch) || has_wmma_f16_gfx12(&self.arch))
            && !std::env::var("HIPFIRE_FP16").map_or(false, |v| v == "0")
            && !std::env::var("HIPFIRE_LM_HEAD_WMMA").map_or(false, |v| v == "0");
        if wmma_eligible {
            self.fp16_x_source_ptr = std::ptr::null_mut();
            match self.active_stream.as_ref() {
                Some(stream) => self.hip.memset_async(&y.buf, 0, batch_size * m * 4, stream)?,
                None => self.hip.memset(&y.buf, 0, batch_size * m * 4)?,
            }
            return self.gemm_hfq3g256_residual_wmma(a_raw, x, y, m, k, batch_size);
        }
        // Non-WMMA fallback: per-batch GEMV. Slow but functional. DFlash on
        // non-gfx11/gfx12 archs is already gated upstream by the daemon's
        // DFlash refusal guard (lm_head whitelist requires gfx11 or gfx12
        // for MQ3) — this fallback is reachable only via direct callers
        // that bypass the daemon (e.g., bench harnesses, channel tests).
        for b in 0..batch_size {
            let x_row = x.sub_offset(b * k, k);
            let y_row = y.sub_offset(b * m, m);
            self.gemv_hfq3g256(a_raw, &x_row, &y_row, m, k)?;
        }
        Ok(())
    }

    // ========================================================================
    // HFQ6-G256 GEMM variants (residual, fused)
    // ========================================================================

    /// gfx906 wave64+dp4a batched residual GEMM for HFQ6/MQ6.
    /// Phase A.2 (plan v3.2.3 §5.1 item 2). Pre-quantizes x to Q8_1 and
    /// dispatches the dp4a kernel; output is residual `+=` semantics.
    ///
    /// Math identity: same as the fused-GEMV dp4a kernels (plan §2.2
    /// Option A — HFQ6 unsigned weights, no zp shift correction).
    pub fn gemm_hfq6g256_residual_wave64_dp4a(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;

        self.ensure_kernel(
            "gemm_hfq6g256_residual_wave64_dp4a",
            kernels::GEMM_HFQ6G256_RESIDUAL_WAVE64_DP4A_SRC,
            "gemm_hfq6g256_residual_wave64_dp4a",
        )?;

        let a_ptr = a_raw.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let bs_val = batch_size as i32;
        let mut xq = xq_ptr;

        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &bs_val as *const _ as *mut c_void,
        ];

        const BATCH_TILE: usize = 8;
        let batch_tiles = (batch_size + BATCH_TILE - 1) / BATCH_TILE;
        let grid_x = ((m as u32) + 1) / 2;

        self.launch_maybe_blob(
            "gemm_hfq6g256_residual_wave64_dp4a",
            [grid_x, batch_tiles as u32, 1],
            [64, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr); b.push_ptr(xq); b.push_ptr(y_ptr);
                b.push_i32(m_val); b.push_i32(k_val); b.push_i32(bs_val);
                b
            },
        )
    }

    /// Batched HFQ6-G256 GEMM with fused residual add:
    ///   for b in 0..batch_size: y[b][row] += A[row] · x[b]
    ///
    /// Auto-selects: gfx11 -> WMMA, gfx906 -> dp4a (Phase A.2),
    /// else -> FP16 packed, fallback -> FP32 scalar.
    pub fn gemm_hfq6g256_residual(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Fast paths for prefill (batch_size > 1). Disable with HIPFIRE_FP16=0.
        if batch_size > 1 && !std::env::var("HIPFIRE_FP16").map_or(false, |v| v == "0") {
            // WMMA on gfx11+ (RDNA3): 16x16 tiled
            if self.arch.starts_with("gfx11") {
                return self.gemm_hfq6g256_residual_wmma(a_raw, x, y, m, k, batch_size);
            }
            // gfx906: dp4a + wave64 batched residual (Phase A.2, plan v3.2.3
            // §5.1 item 2). Pre-quantize x to Q8_1 and dispatch the dp4a
            // kernel. Mirror of the HFQ4 sibling pattern at gemm_hfq4g256_wave64_dp4a.
            // Skip in capture mode: ensure_q8_1_mmq_x launches an internal
            // quantize kernel that the captured graph may not record (matches
            // gemm_hfq4g256_dp4a's `&& !self.capture_mode` guard at line ~7889).
            if gemv_dp4a_enabled(&self.arch) && !self.capture_mode {
                return self.gemm_hfq6g256_residual_wave64_dp4a(a_raw, x, y, m, k, batch_size);
            }
            // FP16 packed on all other RDNA
            return self.gemm_hfq6g256_residual_fp16(a_raw, x, y, m, k, batch_size);
        }
        self.ensure_kernel("gemm_hfq6g256_residual", kernels::GEMM_HFQ6G256_RESIDUAL_SRC, "gemm_hfq6g256_residual")?;
        let func = &self.functions["gemm_hfq6g256_residual"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = { const BATCH_TILE: usize = 8; (batch_size + BATCH_TILE - 1) / BATCH_TILE };

        // Bandwidth: weight (HFQ6: 200 bytes/group vs HFQ4: 136), per-batch x read, per-batch y RMW.
        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k)  // placeholder until hfq6 profiling added
            + batch_size * k * 4
            + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq6g256_residual", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, batch_tiles as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// FP16-input batched HFQ6-G256 GEMM with residual add.
    /// Converts X from FP32 to FP16 (halving X bandwidth), then runs the
    /// FP16-packed GEMM kernel.
    pub fn gemm_hfq6g256_residual_fp16(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,  // FP32 [batch_size x K]
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemm_hfq6g256_residual_fp16", kernels::GEMM_HFQ6G256_RESIDUAL_FP16_SRC, "gemm_hfq6g256_residual_fp16")?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        // FP16 GEMM
        let func = &self.functions["gemm_hfq6g256_residual_fp16"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = { const BATCH_TILE: usize = 8; (batch_size + BATCH_TILE - 1) / BATCH_TILE };

        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k)
            + batch_size * k * 2  // FP16 X (half bandwidth)
            + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq6g256_residual_fp16", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, batch_tiles as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// WMMA-accelerated batched HFQ6-G256 GEMM with residual add.
    /// gfx1100+ only. 16x16 output tiles via wave32 WMMA.
    pub fn gemm_hfq6g256_residual_wmma(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (kernel_name, kernel_src, block_size, row_step) =
            ("gemm_hfq6g256_residual_wmma_k2", kernels::GEMM_HFQ6G256_RESIDUAL_WMMA_K2_SRC, 32u32, 16usize);
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        // WMMA GEMM
        let func = &self.functions[kernel_name];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        let row_tiles = (m + row_step - 1) / row_step;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k)
            + batch_size * k * 2
            + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [row_tiles as u32, batch_tiles as u32, 1],
                [block_size, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched 4-way fused HFQ6-G256 GEMM (qkv + z + beta + alpha).
    /// Auto-selects: gfx11 -> WMMA, else -> scalar.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq6g256(
        &mut self,
        a_qkv: &GpuTensor, a_z: &GpuTensor, a_beta: &GpuTensor, a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor, y_z: &GpuTensor, y_beta: &GpuTensor, y_alpha: &GpuTensor,
        qkv_m: usize, z_m: usize, beta_m: usize, alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Fast paths for prefill (batch_size > 1). Disable with HIPFIRE_FP16=0.
        if batch_size > 1 && !std::env::var("HIPFIRE_FP16").map_or(false, |v| v == "0") {
            if has_wmma_f16_gfx12(&self.arch) {
                return self.gemm_qkvza_hfq6g256_wmma_gfx12(a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m, alpha_m, k, batch_size);
            }
            if has_wmma_f16(&self.arch) {
                return self.gemm_qkvza_hfq6g256_wmma(a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m, alpha_m, k, batch_size);
            }
            // gfx906: wave64+dp4a batched fused (Phase A.3, plan v3.2.3 §5.1
            // item 3). Pre-quantize x to Q8_1 and dispatch the dp4a kernel.
            // Skip in capture mode (Q8_1 quantize launch must be reachable
            // from captured graph or pre-baked) — matches HFQ4 sibling pattern.
            if gemv_dp4a_enabled(&self.arch) && !self.capture_mode {
                return self.gemm_qkvza_hfq6g256_wave64_dp4a(a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m, alpha_m, k, batch_size);
            }
            // v_dot2_f32_f16 on archs that have it (gfx1011/1012/1030-1032).
            // Excludes gfx1010 (Navi 10, 5700 XT) and gfx1013 (Van Gogh/BC-250 APU).
            if has_dot2_f32_f16(&self.arch) {
                return self.gemm_qkvza_hfq6g256_dot2(a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m, alpha_m, k, batch_size);
            }
            // FP16 packed (v_pk_fma_f16) for gfx1010/1013 — 2× scalar FP32.
            return self.gemm_qkvza_hfq6g256_fp16(a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m, alpha_m, k, batch_size);
        }
        self.ensure_kernel("gemm_qkvza_hfq6g256", kernels::GEMM_QKVZA_HFQ6G256_SRC, "gemm_qkvza_hfq6g256")?;
        let func = &self.functions["gemm_qkvza_hfq6g256"];

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = { const BATCH_TILE: usize = 8; (batch_size + BATCH_TILE - 1) / BATCH_TILE };
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(z_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(alpha_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq6g256", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// FP16-packed batched 4-way fused HFQ6-G256 GEMM (qkv + z + beta + alpha).
    /// RDNA1/2 fast path — v_pk_fma_f16 inner loop, 2× scalar FP32 throughput.
    /// Requires FP16-converted X (provided via ensure_fp16_x).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq6g256_fp16(
        &mut self,
        a_qkv: &GpuTensor, a_z: &GpuTensor, a_beta: &GpuTensor, a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor, y_z: &GpuTensor, y_beta: &GpuTensor, y_alpha: &GpuTensor,
        qkv_m: usize, z_m: usize, beta_m: usize, alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq6g256_fp16",
            kernels::GEMM_QKVZA_HFQ6G256_FP16_SRC,
            "gemm_qkvza_hfq6g256_fp16",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkvza_hfq6g256_fp16"];

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = { const BATCH_TILE: usize = 8; (batch_size + BATCH_TILE - 1) / BATCH_TILE };
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(z_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(alpha_m, k)
                  + batch_size * k * 2  // FP16 X
                  + batch_size * (qkv_m + z_m + beta_m + alpha_m) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq6g256_fp16", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// v_dot2_f32_f16-accelerated batched 4-way fused HFQ6-G256 GEMM (qkv + z + beta + alpha).
    /// RDNA2 (gfx1011/1012/1030-1032) fast path using `amd_mixed_dot`.
    /// One instruction per half2 dot with FP32 accumulation — 1.2-1.5× over FP16 packed.
    #[allow(clippy::too_many_arguments)]
    /// gfx906 wave64+dp4a batched 4-way fused QKVZA GEMM. Phase A.3
    /// (plan v3.2.3 §5.1 item 3). Uses Q8_1 activation pre-quantize
    /// (shared with A.1c GEMV-shape dp4a kernels) and HFQ6 6-bit
    /// unsigned weight unpack.
    pub fn gemm_qkvza_hfq6g256_wave64_dp4a(
        &mut self,
        a_qkv: &GpuTensor, a_z: &GpuTensor, a_beta: &GpuTensor, a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor, y_z: &GpuTensor, y_beta: &GpuTensor, y_alpha: &GpuTensor,
        qkv_m: usize, z_m: usize, beta_m: usize, alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;

        self.ensure_kernel(
            "gemm_qkvza_hfq6g256_wave64_dp4a",
            kernels::GEMM_QKVZA_HFQ6G256_WAVE64_DP4A_SRC,
            "gemm_qkvza_hfq6g256_wave64_dp4a",
        )?;

        let aq = a_qkv.buf.as_ptr();
        let az = a_z.buf.as_ptr();
        let ab = a_beta.buf.as_ptr();
        let aa = a_alpha.buf.as_ptr();
        let yq = y_qkv.buf.as_ptr();
        let yz = y_z.buf.as_ptr();
        let yb = y_beta.buf.as_ptr();
        let ya = y_alpha.buf.as_ptr();
        let qkv_m_val = qkv_m as i32;
        let z_m_val = z_m as i32;
        let beta_m_val = beta_m as i32;
        let alpha_m_val = alpha_m as i32;
        let k_val = k as i32;
        let bs_val = batch_size as i32;
        let mut xq = xq_ptr;

        let mut params: Vec<*mut c_void> = vec![
            &aq as *const _ as *mut c_void,
            &az as *const _ as *mut c_void,
            &ab as *const _ as *mut c_void,
            &aa as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &yq as *const _ as *mut c_void,
            &yz as *const _ as *mut c_void,
            &yb as *const _ as *mut c_void,
            &ya as *const _ as *mut c_void,
            &qkv_m_val as *const _ as *mut c_void,
            &z_m_val as *const _ as *mut c_void,
            &beta_m_val as *const _ as *mut c_void,
            &alpha_m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &bs_val as *const _ as *mut c_void,
        ];

        const BATCH_TILE: usize = 8;
        let batch_tiles = (batch_size + BATCH_TILE - 1) / BATCH_TILE;
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;
        let grid_x = (total_m + 1) / 2;

        self.launch_maybe_blob(
            "gemm_qkvza_hfq6g256_wave64_dp4a",
            [grid_x, batch_tiles as u32, 1],
            [64, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq); b.push_ptr(az); b.push_ptr(ab); b.push_ptr(aa);
                b.push_ptr(xq);
                b.push_ptr(yq); b.push_ptr(yz); b.push_ptr(yb); b.push_ptr(ya);
                b.push_i32(qkv_m_val); b.push_i32(z_m_val);
                b.push_i32(beta_m_val); b.push_i32(alpha_m_val);
                b.push_i32(k_val); b.push_i32(bs_val);
                b
            },
        )
    }

    pub fn gemm_qkvza_hfq6g256_dot2(
        &mut self,
        a_qkv: &GpuTensor, a_z: &GpuTensor, a_beta: &GpuTensor, a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor, y_z: &GpuTensor, y_beta: &GpuTensor, y_alpha: &GpuTensor,
        qkv_m: usize, z_m: usize, beta_m: usize, alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq6g256_dot2",
            kernels::GEMM_QKVZA_HFQ6G256_DOT2_SRC,
            "gemm_qkvza_hfq6g256_dot2",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkvza_hfq6g256_dot2"];

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = { const BATCH_TILE: usize = 8; (batch_size + BATCH_TILE - 1) / BATCH_TILE };
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;

        unsafe {
            self.hip.launch_kernel(
                func, [total_m, batch_tiles as u32, 1], [32, 1, 1], 0, self.stream_ref(), &mut params,
            )
        }
    }

    /// WMMA-accelerated batched 4-way fused HFQ6-G256 GEMM (qkv + z + beta + alpha).
    /// gfx1100+ only. 16x16 output tiles via wave32 WMMA.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq6g256_wmma(
        &mut self,
        a_qkv: &GpuTensor, a_z: &GpuTensor, a_beta: &GpuTensor, a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor, y_z: &GpuTensor, y_beta: &GpuTensor, y_alpha: &GpuTensor,
        qkv_m: usize, z_m: usize, beta_m: usize, alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemm_qkvza_hfq6g256_wmma", kernels::GEMM_QKVZA_HFQ6G256_WMMA_SRC, "gemm_qkvza_hfq6g256_wmma")?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(z_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(alpha_m, k)
                  + batch_size * k * 2
                  + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq6g256_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkvza_hfq6g256_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq); b.push_ptr(az); b.push_ptr(ab); b.push_ptr(aa);
                b.push_ptr(xp);
                b.push_ptr(yq); b.push_ptr(yz); b.push_ptr(yb); b.push_ptr(ya);
                b.push_i32(q_m); b.push_i32(z_m_val); b.push_i32(b_m); b.push_i32(a_m);
                b.push_i32(k_val); b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// gfx12 (RDNA4) sister of `gemm_qkvza_hfq6g256_wmma`. Pure scaffold
    /// composition (hfq6 dequant + 4-output qkvza routing, both validated
    /// on R9700). Not yet wired into the public dispatch tree — exposed
    /// only for the channel-test harness. See issue #54.
    pub fn gemm_qkvza_hfq6g256_wmma_gfx12(
        &mut self,
        a_qkv: &GpuTensor, a_z: &GpuTensor, a_beta: &GpuTensor, a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor, y_z: &GpuTensor, y_beta: &GpuTensor, y_alpha: &GpuTensor,
        qkv_m: usize, z_m: usize, beta_m: usize, alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq6g256_wmma_gfx12",
            kernels::GEMM_QKVZA_HFQ6G256_WMMA_GFX12_SRC,
            "gemm_qkvza_hfq6g256_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(z_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(alpha_m, k)
                  + batch_size * k * 2
                  + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq6g256_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkvza_hfq6g256_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq); b.push_ptr(az); b.push_ptr(ab); b.push_ptr(aa);
                b.push_ptr(xp);
                b.push_ptr(yq); b.push_ptr(yz); b.push_ptr(yb); b.push_ptr(ya);
                b.push_i32(q_m); b.push_i32(z_m_val); b.push_i32(b_m); b.push_i32(a_m);
                b.push_i32(k_val); b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched 3-way fused HFQ6-G256 GEMM for the FA preamble (Q + K + V).
    /// Auto-selects: gfx11 -> WMMA, else -> scalar.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkv_hfq6g256(
        &mut self,
        a_q: &GpuTensor, a_k: &GpuTensor, a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor, y_k: &GpuTensor, y_v: &GpuTensor,
        q_m: usize, k_m: usize, v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Fast paths for prefill (batch_size > 1). Disable with HIPFIRE_FP16=0.
        if batch_size > 1 && !std::env::var("HIPFIRE_FP16").map_or(false, |v| v == "0") {
            if has_wmma_f16_gfx12(&self.arch) {
                return self.gemm_qkv_hfq6g256_wmma_gfx12(a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size);
            }
            if has_wmma_f16(&self.arch) {
                return self.gemm_qkv_hfq6g256_wmma(a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size);
            }
            // gfx906: wave64+dp4a batched fused (Phase A.3).
            // Skip in capture mode (Q8_1 quantize) — matches HFQ4 sibling.
            if gemv_dp4a_enabled(&self.arch) && !self.capture_mode {
                return self.gemm_qkv_hfq6g256_wave64_dp4a(a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size);
            }
            // v_dot2_f32_f16 on archs that have it (gfx1011/1012/1030-1032).
            // Excludes gfx1010 (Navi 10, 5700 XT) and gfx1013 (Van Gogh/BC-250 APU).
            if has_dot2_f32_f16(&self.arch) {
                return self.gemm_qkv_hfq6g256_dot2(a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size);
            }
            // FP16 packed (v_pk_fma_f16) for gfx1010/1013 — 2× scalar FP32.
            return self.gemm_qkv_hfq6g256_fp16(a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size);
        }
        self.ensure_kernel("gemm_qkv_hfq6g256", kernels::GEMM_QKV_HFQ6G256_SRC, "gemm_qkv_hfq6g256")?;
        let func = &self.functions["gemm_qkv_hfq6g256"];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = { const BATCH_TILE: usize = 8; (batch_size + BATCH_TILE - 1) / BATCH_TILE };
        let total_m = (q_m + k_m + v_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(k_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(v_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq6g256", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// FP16-packed batched 3-way fused HFQ6-G256 GEMM (Q + K + V).
    /// RDNA1/2 fast path — v_pk_fma_f16 inner loop, 2× scalar FP32 throughput.
    /// Requires FP16-converted X (provided via ensure_fp16_x).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkv_hfq6g256_fp16(
        &mut self,
        a_q: &GpuTensor, a_k: &GpuTensor, a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor, y_k: &GpuTensor, y_v: &GpuTensor,
        q_m: usize, k_m: usize, v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq6g256_fp16",
            kernels::GEMM_QKV_HFQ6G256_FP16_SRC,
            "gemm_qkv_hfq6g256_fp16",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkv_hfq6g256_fp16"];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = { const BATCH_TILE: usize = 8; (batch_size + BATCH_TILE - 1) / BATCH_TILE };
        let total_m = (q_m + k_m + v_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(k_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(v_m, k)
                  + batch_size * k * 2  // FP16 X
                  + batch_size * (q_m + k_m + v_m) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq6g256_fp16", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// v_dot2_f32_f16-accelerated batched 3-way fused HFQ6-G256 GEMM (Q + K + V).
    /// RDNA2 (gfx1011/1012/1030-1032) fast path using `amd_mixed_dot`.
    /// One instruction per half2 dot with FP32 accumulation — 1.2-1.5× over FP16 packed.
    #[allow(clippy::too_many_arguments)]
    /// gfx906 wave64+dp4a batched 3-way fused QKV GEMM. Phase A.3
    /// (plan v3.2.3 §5.1 item 3). Sibling of qkvza_wave64_dp4a.
    pub fn gemm_qkv_hfq6g256_wave64_dp4a(
        &mut self,
        a_q: &GpuTensor, a_k: &GpuTensor, a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor, y_k: &GpuTensor, y_v: &GpuTensor,
        q_m: usize, k_m: usize, v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;

        self.ensure_kernel(
            "gemm_qkv_hfq6g256_wave64_dp4a",
            kernels::GEMM_QKV_HFQ6G256_WAVE64_DP4A_SRC,
            "gemm_qkv_hfq6g256_wave64_dp4a",
        )?;

        let aq = a_q.buf.as_ptr();
        let ak = a_k.buf.as_ptr();
        let av = a_v.buf.as_ptr();
        let yq = y_q.buf.as_ptr();
        let yk = y_k.buf.as_ptr();
        let yv = y_v.buf.as_ptr();
        let q_m_val = q_m as i32;
        let k_m_val = k_m as i32;
        let v_m_val = v_m as i32;
        let k_val = k as i32;
        let bs_val = batch_size as i32;
        let mut xq = xq_ptr;

        let mut params: Vec<*mut c_void> = vec![
            &aq as *const _ as *mut c_void,
            &ak as *const _ as *mut c_void,
            &av as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &yq as *const _ as *mut c_void,
            &yk as *const _ as *mut c_void,
            &yv as *const _ as *mut c_void,
            &q_m_val as *const _ as *mut c_void,
            &k_m_val as *const _ as *mut c_void,
            &v_m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &bs_val as *const _ as *mut c_void,
        ];

        const BATCH_TILE: usize = 8;
        let batch_tiles = (batch_size + BATCH_TILE - 1) / BATCH_TILE;
        let total_m = (q_m + k_m + v_m) as u32;
        let grid_x = (total_m + 1) / 2;

        self.launch_maybe_blob(
            "gemm_qkv_hfq6g256_wave64_dp4a",
            [grid_x, batch_tiles as u32, 1],
            [64, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq); b.push_ptr(ak); b.push_ptr(av);
                b.push_ptr(xq);
                b.push_ptr(yq); b.push_ptr(yk); b.push_ptr(yv);
                b.push_i32(q_m_val); b.push_i32(k_m_val); b.push_i32(v_m_val);
                b.push_i32(k_val); b.push_i32(bs_val);
                b
            },
        )
    }

    pub fn gemm_qkv_hfq6g256_dot2(
        &mut self,
        a_q: &GpuTensor, a_k: &GpuTensor, a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor, y_k: &GpuTensor, y_v: &GpuTensor,
        q_m: usize, k_m: usize, v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq6g256_dot2",
            kernels::GEMM_QKV_HFQ6G256_DOT2_SRC,
            "gemm_qkv_hfq6g256_dot2",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkv_hfq6g256_dot2"];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = { const BATCH_TILE: usize = 8; (batch_size + BATCH_TILE - 1) / BATCH_TILE };
        let total_m = (q_m + k_m + v_m) as u32;

        unsafe {
            self.hip.launch_kernel(
                func, [total_m, batch_tiles as u32, 1], [32, 1, 1], 0, self.stream_ref(), &mut params,
            )
        }
    }

    /// WMMA-accelerated batched 3-way fused HFQ6-G256 GEMM (Q + K + V).
    /// gfx1100+ only. 16x16 output tiles via wave32 WMMA.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkv_hfq6g256_wmma(
        &mut self,
        a_q: &GpuTensor, a_k: &GpuTensor, a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor, y_k: &GpuTensor, y_v: &GpuTensor,
        q_m: usize, k_m: usize, v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemm_qkv_hfq6g256_wmma", kernels::GEMM_QKV_HFQ6G256_WMMA_SRC, "gemm_qkv_hfq6g256_wmma")?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(k_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(v_m, k)
                  + batch_size * k * 2
                  + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq6g256_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_hfq6g256_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq); b.push_ptr(ak); b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq); b.push_ptr(yk); b.push_ptr(yv);
                b.push_i32(q_m_val); b.push_i32(k_m_val); b.push_i32(v_m_val);
                b.push_i32(k_val); b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// gfx12 (RDNA4) sister of `gemm_qkv_hfq6g256_wmma`. Same gfx12 recipe
    /// as the hfq4 QKV scaffold (validated on R9700) with the hfq6 dequant
    /// inner loop carried over. Not yet wired into the public dispatch
    /// tree — exposed only for the channel-test harness. See issue #54.
    pub fn gemm_qkv_hfq6g256_wmma_gfx12(
        &mut self,
        a_q: &GpuTensor, a_k: &GpuTensor, a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor, y_k: &GpuTensor, y_v: &GpuTensor,
        q_m: usize, k_m: usize, v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq6g256_wmma_gfx12",
            kernels::GEMM_QKV_HFQ6G256_WMMA_GFX12_SRC,
            "gemm_qkv_hfq6g256_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(k_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(v_m, k)
                  + batch_size * k * 2
                  + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq6g256_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_hfq6g256_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq); b.push_ptr(ak); b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq); b.push_ptr(yk); b.push_ptr(yv);
                b.push_i32(q_m_val); b.push_i32(k_m_val); b.push_i32(v_m_val);
                b.push_i32(k_val); b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched 2-way fused HFQ6-G256 GEMM for the FFN preamble (gate + up).
    /// Auto-selects: gfx11 -> WMMA, else -> scalar.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_gate_up_hfq6g256(
        &mut self,
        a_gate: &GpuTensor, a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor, y_up: &GpuTensor,
        gate_m: usize, up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Fast paths for prefill (batch_size > 1). Disable with HIPFIRE_FP16=0.
        if batch_size > 1 && !std::env::var("HIPFIRE_FP16").map_or(false, |v| v == "0") {
            if has_wmma_f16_gfx12(&self.arch) {
                return self.gemm_gate_up_hfq6g256_wmma_gfx12(a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size);
            }
            if has_wmma_f16(&self.arch) {
                return self.gemm_gate_up_hfq6g256_wmma(a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size);
            }
            // gfx906: wave64+dp4a batched fused (Phase A.3).
            // Skip in capture mode (Q8_1 quantize) — matches HFQ4 sibling.
            if gemv_dp4a_enabled(&self.arch) && !self.capture_mode {
                return self.gemm_gate_up_hfq6g256_wave64_dp4a(a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size);
            }
            // v_dot2_f32_f16 on archs that have it (gfx1011/1012/1030-1032).
            // Excludes gfx1010 (Navi 10, 5700 XT) and gfx1013 (Van Gogh/BC-250 APU).
            if has_dot2_f32_f16(&self.arch) {
                return self.gemm_gate_up_hfq6g256_dot2(a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size);
            }
            // FP16 packed (v_pk_fma_f16) for gfx1010/1013 — 2× scalar FP32.
            return self.gemm_gate_up_hfq6g256_fp16(a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size);
        }
        self.ensure_kernel("gemm_gate_up_hfq6g256", kernels::GEMM_GATE_UP_HFQ6G256_SRC, "gemm_gate_up_hfq6g256")?;
        let func = &self.functions["gemm_gate_up_hfq6g256"];

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m = gate_m as i32;
        let mut u_m = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m as *mut _ as *mut c_void,
            &mut u_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = { const BATCH_TILE: usize = 8; (batch_size + BATCH_TILE - 1) / BATCH_TILE };
        let total_m = (gate_m + up_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(up_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq6g256", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// FP16-packed batched 2-way fused HFQ6-G256 GEMM (gate + up).
    /// RDNA1/2 fast path — v_pk_fma_f16 inner loop, 2× scalar FP32 throughput.
    /// Requires FP16-converted X (provided via ensure_fp16_x).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_gate_up_hfq6g256_fp16(
        &mut self,
        a_gate: &GpuTensor, a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor, y_up: &GpuTensor,
        gate_m: usize, up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_gate_up_hfq6g256_fp16",
            kernels::GEMM_GATE_UP_HFQ6G256_FP16_SRC,
            "gemm_gate_up_hfq6g256_fp16",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_gate_up_hfq6g256_fp16"];

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m = gate_m as i32;
        let mut u_m = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m as *mut _ as *mut c_void,
            &mut u_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = { const BATCH_TILE: usize = 8; (batch_size + BATCH_TILE - 1) / BATCH_TILE };
        let total_m = (gate_m + up_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(up_m, k)
                  + batch_size * k * 2  // FP16 X
                  + batch_size * (gate_m + up_m) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq6g256_fp16", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// v_dot2_f32_f16-accelerated batched 2-way fused HFQ6-G256 GEMM (gate + up).
    /// RDNA2 (gfx1011/1012/1030-1032) fast path using `amd_mixed_dot`.
    /// One instruction per half2 dot with FP32 accumulation — 1.2-1.5× over FP16 packed.
    #[allow(clippy::too_many_arguments)]
    /// gfx906 wave64+dp4a batched 2-way fused gate+up GEMM. Phase A.3.
    pub fn gemm_gate_up_hfq6g256_wave64_dp4a(
        &mut self,
        a_gate: &GpuTensor, a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor, y_up: &GpuTensor,
        gate_m: usize, up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;

        self.ensure_kernel(
            "gemm_gate_up_hfq6g256_wave64_dp4a",
            kernels::GEMM_GATE_UP_HFQ6G256_WAVE64_DP4A_SRC,
            "gemm_gate_up_hfq6g256_wave64_dp4a",
        )?;

        let agate = a_gate.buf.as_ptr();
        let aup = a_up.buf.as_ptr();
        let ygate = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let gate_m_val = gate_m as i32;
        let up_m_val = up_m as i32;
        let k_val = k as i32;
        let bs_val = batch_size as i32;
        let mut xq = xq_ptr;

        let mut params: Vec<*mut c_void> = vec![
            &agate as *const _ as *mut c_void,
            &aup as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &ygate as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &gate_m_val as *const _ as *mut c_void,
            &up_m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &bs_val as *const _ as *mut c_void,
        ];

        const BATCH_TILE: usize = 8;
        let batch_tiles = (batch_size + BATCH_TILE - 1) / BATCH_TILE;
        let total_m = (gate_m + up_m) as u32;
        let grid_x = (total_m + 1) / 2;

        self.launch_maybe_blob(
            "gemm_gate_up_hfq6g256_wave64_dp4a",
            [grid_x, batch_tiles as u32, 1],
            [64, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(agate); b.push_ptr(aup); b.push_ptr(xq);
                b.push_ptr(ygate); b.push_ptr(yup);
                b.push_i32(gate_m_val); b.push_i32(up_m_val);
                b.push_i32(k_val); b.push_i32(bs_val);
                b
            },
        )
    }

    pub fn gemm_gate_up_hfq6g256_dot2(
        &mut self,
        a_gate: &GpuTensor, a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor, y_up: &GpuTensor,
        gate_m: usize, up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_gate_up_hfq6g256_dot2",
            kernels::GEMM_GATE_UP_HFQ6G256_DOT2_SRC,
            "gemm_gate_up_hfq6g256_dot2",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_gate_up_hfq6g256_dot2"];

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m = gate_m as i32;
        let mut u_m = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m as *mut _ as *mut c_void,
            &mut u_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = { const BATCH_TILE: usize = 8; (batch_size + BATCH_TILE - 1) / BATCH_TILE };
        let total_m = (gate_m + up_m) as u32;

        unsafe {
            self.hip.launch_kernel(
                func, [total_m, batch_tiles as u32, 1], [32, 1, 1], 0, self.stream_ref(), &mut params,
            )
        }
    }

    /// WMMA-accelerated batched 2-way fused HFQ6-G256 GEMM (gate + up).
    /// gfx1100+ only. 16x16 output tiles via wave32 WMMA.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_gate_up_hfq6g256_wmma(
        &mut self,
        a_gate: &GpuTensor, a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor, y_up: &GpuTensor,
        gate_m: usize, up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemm_gate_up_hfq6g256_wmma", kernels::GEMM_GATE_UP_HFQ6G256_WMMA_SRC, "gemm_gate_up_hfq6g256_wmma")?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m = gate_m as i32;
        let mut u_m = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m as *mut _ as *mut c_void,
            &mut u_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(up_m, k)
                  + batch_size * k * 2
                  + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq6g256_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_gate_up_hfq6g256_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ag); b.push_ptr(au);
                b.push_ptr(xp);
                b.push_ptr(yg); b.push_ptr(yu);
                b.push_i32(g_m); b.push_i32(u_m);
                b.push_i32(k_val); b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// gfx12 (RDNA4) sister of `gemm_gate_up_hfq6g256_wmma`. Same gfx12
    /// recipe as the other scaffolds (validated on R9700). Not yet wired
    /// into the public dispatch tree — exposed only for the channel-test
    /// harness. See issue #54.
    pub fn gemm_gate_up_hfq6g256_wmma_gfx12(
        &mut self,
        a_gate: &GpuTensor, a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor, y_up: &GpuTensor,
        gate_m: usize, up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_gate_up_hfq6g256_wmma_gfx12",
            kernels::GEMM_GATE_UP_HFQ6G256_WMMA_GFX12_SRC,
            "gemm_gate_up_hfq6g256_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m = gate_m as i32;
        let mut u_m = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m as *mut _ as *mut c_void,
            &mut u_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(up_m, k)
                  + batch_size * k * 2
                  + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq6g256_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_gate_up_hfq6g256_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ag); b.push_ptr(au);
                b.push_ptr(xp);
                b.push_ptr(yg); b.push_ptr(yu);
                b.push_i32(g_m); b.push_i32(u_m);
                b.push_i32(k_val); b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Compute max softmax probability on GPU. Downloads 4 bytes instead of vocab×4.
    pub fn max_prob(
        &mut self, logits: &GpuTensor, result: &GpuTensor, vocab_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("max_prob", kernels::MAX_PROB_SRC, "max_prob")?;
        let func = &self.functions["max_prob"];
        let mut lp = logits.buf.as_ptr();
        let mut rp = result.buf.as_ptr();
        let mut vs = vocab_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut lp as *mut _ as *mut c_void, &mut rp as *mut _ as *mut c_void,
            &mut vs as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let shared = (block * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [1, 1, 1], [block, 1, 1], shared, self.stream_ref(), &mut params) }
    }

    /// Fused QKV: three Q4_K GEMVs in one launch (saves 2 kernel launches per layer).
    /// q = Wq * x, k = Wk * x, v = Wv * x — all read the same input x.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_qkv_q4k(
        &mut self,
        wq: &GpuTensor, wk: &GpuTensor, wv: &GpuTensor,
        x: &GpuTensor,
        yq: &GpuTensor, yk: &GpuTensor, yv: &GpuTensor,
        q_m: usize, k_m: usize, v_m: usize, k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("fused_qkv_q4k", kernels::FUSED_QKV_Q4K_SRC, "fused_qkv_q4k")?;
        let func = &self.functions["fused_qkv_q4k"];

        let mut aq = wq.buf.as_ptr();
        let mut ak = wk.buf.as_ptr();
        let mut av = wv.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yqp = yq.buf.as_ptr();
        let mut ykp = yk.buf.as_ptr();
        let mut yvp = yv.buf.as_ptr();
        let mut qm = q_m as i32;
        let mut km = k_m as i32;
        let mut vm = v_m as i32;
        let mut kk = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yqp as *mut _ as *mut c_void,
            &mut ykp as *mut _ as *mut c_void,
            &mut yvp as *mut _ as *mut c_void,
            &mut qm as *mut _ as *mut c_void,
            &mut km as *mut _ as *mut c_void,
            &mut vm as *mut _ as *mut c_void,
            &mut kk as *mut _ as *mut c_void,
        ];

        let grid = (q_m + k_m + v_m) as u32;
        unsafe {
            self.hip.launch_kernel(func, [grid, 1, 1], [32, 1, 1], 0, None, &mut params)
        }
    }

    /// Fused Gate+Up: two Q4_K GEMVs in one launch (saves 1 kernel launch per layer).
    #[allow(clippy::too_many_arguments)]
    pub fn fused_gate_up_q4k(
        &mut self,
        w_gate: &GpuTensor, w_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor, y_up: &GpuTensor,
        gate_m: usize, up_m: usize, k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("fused_gate_up_q4k", kernels::FUSED_GATE_UP_Q4K_SRC, "fused_gate_up_q4k")?;
        let func = &self.functions["fused_gate_up_q4k"];

        let mut ag = w_gate.buf.as_ptr();
        let mut au = w_up.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut ygp = y_gate.buf.as_ptr();
        let mut yup = y_up.buf.as_ptr();
        let mut gm = gate_m as i32;
        let mut um = up_m as i32;
        let mut kk = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut ygp as *mut _ as *mut c_void,
            &mut yup as *mut _ as *mut c_void,
            &mut gm as *mut _ as *mut c_void,
            &mut um as *mut _ as *mut c_void,
            &mut kk as *mut _ as *mut c_void,
        ];

        let grid = (gate_m + up_m) as u32;
        unsafe {
            self.hip.launch_kernel(func, [grid, 1, 1], [32, 1, 1], 0, None, &mut params)
        }
    }

    /// y = A_q8_0 * x (quantized GEMV for Q8_0)
    pub fn gemv_q8_0(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(a_ptr); b.push_ptr(x_ptr); b.push_ptr(y_ptr);
            b.push_i32(m_val); b.push_i32(k_val);
            b
        };

        // Adaptive dispatch: wide kernel for small K (more threads per row),
        // narrow kernel for large K (more blocks, better occupancy).
        if k <= 1536 {
            self.ensure_kernel("gemv_q8_0_wide", kernels::GEMV_Q8_0_WIDE_SRC, "gemv_q8_0_wide")?;
            let block_size = 64u32; // 2 warps, each processes one row
            let grid = ((m + 1) / 2) as u32; // ceil(M/2)
            return self.launch_maybe_blob(
                "gemv_q8_0_wide",
                [grid, 1, 1], [block_size, 1, 1], 0, &mut params,
                blob_builder,
            );
        }

        self.ensure_kernel("gemv_q8_0", kernels::GEMV_Q8_0_SRC, "gemv_q8_0")?;
        let block_size = 32u32;
        self.launch_maybe_blob(
            "gemv_q8_0",
            [m as u32, 1, 1], [block_size, 1, 1], 0, &mut params,
            blob_builder,
        )
    }

    /// Y[batch, M] = X[batch, K] @ A_q8[M, K]^T — batched Q8_0 GEMM.
    /// One block per output row (32 threads, one wave). Each thread holds
    /// MAX_BATCH=16 per-batch accumulators and broadcasts each weight load.
    /// Drops the (batch_size − 1)× weight re-reads of the GEMV-loop path
    /// without splitting launches.
    pub fn gemm_q8_0_batched(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            batch_size <= 16,
            "gemm_q8_0_batched: batch_size {batch_size} exceeds kernel MAX_BATCH=16"
        );
        self.ensure_kernel(
            "gemm_q8_0_batched",
            kernels::GEMM_Q8_0_BATCHED_SRC,
            "gemm_q8_0_batched",
        )?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        self.launch_maybe_blob(
            "gemm_q8_0_batched",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr); b.push_ptr(x_ptr); b.push_ptr(y_ptr);
                b.push_i32(m_val); b.push_i32(k_val); b.push_i32(bs_val);
                b
            },
        )
    }

    /// y = A_q8hfq * x (split-metadata Q8 GEMV, row_stride = padded row bytes)
    pub fn gemv_q8hfq(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        row_stride: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut rs_val = row_stride as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut rs_val as *mut _ as *mut c_void,
        ];

        if k <= 1536 {
            self.ensure_kernel("gemv_q8hfq_wide", kernels::GEMV_Q8HFQ_WIDE_SRC, "gemv_q8hfq_wide")?;
            let func = &self.functions["gemv_q8hfq_wide"];
            let block_size = 64u32;
            let grid = ((m + 1) / 2) as u32;
            return unsafe {
                self.hip.launch_kernel(func, [grid, 1, 1], [block_size, 1, 1], 0, None, &mut params)
            };
        }

        self.ensure_kernel("gemv_q8hfq", kernels::GEMV_Q8HFQ_SRC, "gemv_q8hfq")?;
        let func = &self.functions["gemv_q8hfq"];
        unsafe {
            self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, None, &mut params)
        }
    }

    /// y = A_q6k * x (quantized GEMV for Q6_K)
    pub fn gemv_q6k(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_q6k", kernels::GEMV_Q6K_SRC, "gemv_q6k")?;
        let func = &self.functions["gemv_q6k"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        let block_size = 256u32;
        let shared_mem = block_size * 4;
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// y = A_q4f16 * x (RDNA-native Q4_F16 GEMV, group size 64)
    /// a_raw: raw Q4_F16_G64 bytes on GPU, x: F32 input, y: F32 output
    /// Block: 36 bytes per 64 elements. K must be multiple of 64.
    /// Uses 128 threads (4 warps) with shared memory reduction for increased MLP.
    pub fn gemv_q4f16_g64(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_q4f16_g64", kernels::GEMV_Q4F16_G64_SRC, "gemv_q4f16_g64")?;
        let func = &self.functions["gemv_q4f16_g64"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        let block_size = 32u32; // single warp — no shared memory
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [block_size, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// y = A_q4f16 * x (256-thread wide variant for occupancy testing)
    /// Element-strided access pattern matching F32 GEMV. Shared memory reduction.
    pub fn gemv_q4f16_g64_wide(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_q4f16_g64_wide", kernels::GEMV_Q4F16_G64_WIDE_SRC, "gemv_q4f16_g64_wide")?;
        let func = &self.functions["gemv_q4f16_g64_wide"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        let block_size = 256u32;
        let shared_mem = block_size * 4; // one float per thread
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// y = A_q4f16 * x (RDNA-native Q4_F16 GEMV, group size 32)
    /// Block: 20 bytes per 32 elements. K must be multiple of 32.
    pub fn gemv_q4f16_g32(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_q4f16_g32", kernels::GEMV_Q4F16_G32_SRC, "gemv_q4f16_g32")?;
        let func = &self.functions["gemv_q4f16_g32"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        let block_size = 32u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [block_size, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// GPU-side batched argmax: writes one i32 index per row into `result`
    /// (shape `[batch_size]`). Avoids downloading `batch_size × n` floats
    /// to the host — only `batch_size × 4` bytes land on PCIe.
    pub fn argmax_f32_batched(
        &mut self,
        data: &GpuTensor,
        result: &GpuTensor,
        n: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "argmax_f32_batched",
            kernels::ARGMAX_BATCHED_SRC,
            "argmax_f32_batched",
        )?;

        let mut dp = data.buf.as_ptr();
        let mut rp = result.buf.as_ptr();
        let mut nn = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut dp as *mut _ as *mut c_void,
            &mut rp as *mut _ as *mut c_void,
            &mut nn as *mut _ as *mut c_void,
        ];

        let block_size = 256u32;
        let shared = block_size * 8; // f32 + i32 per thread
        self.launch_maybe_blob(
            "argmax_f32_batched",
            [batch_size as u32, 1, 1],
            [block_size, 1, 1],
            shared,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(dp); b.push_ptr(rp); b.push_i32(nn);
                b
            },
        )
    }

    /// GPU-side argmax: returns index of max value. Avoids downloading full logits.
    pub fn argmax_f32(&mut self, data: &GpuTensor, n: usize) -> HipResult<u32> {
        self.bind_thread()?;
        self.ensure_kernel("argmax_f32", kernels::ARGMAX_SRC, "argmax_f32")?;
        let func = &self.functions["argmax_f32"];

        let result_buf = self.hip.malloc(4)?; // single int
        self.hip.memset(&result_buf, 0, 4)?;

        let mut dp = data.buf.as_ptr();
        let mut rp = result_buf.as_ptr();
        let mut nn = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut dp as *mut _ as *mut c_void,
            &mut rp as *mut _ as *mut c_void,
            &mut nn as *mut _ as *mut c_void,
        ];

        let block_size = 256u32;
        let shared = block_size * 8; // float + int per thread
        unsafe {
            self.hip.launch_kernel(func, [1, 1, 1], [block_size, 1, 1], shared, None, &mut params)?;
        }

        let mut result = [0i32];
        let result_bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, 4)
        };
        self.hip.memcpy_dtoh(result_bytes, &result_buf)?;
        self.hip.free(result_buf)?;
        Ok(result[0] as u32)
    }

    /// out = rmsnorm(x, weight, eps)
    pub fn rmsnorm_f32(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        out: &GpuTensor,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("rmsnorm", kernels::RMSNORM_SRC, "rmsnorm_f32")?;

        let batch = if x.shape.len() > 1 { x.shape[0] } else { 1 };
        let n = x.shape.last().copied().unwrap() as i32;

        let x_ptr = x.buf.as_ptr();
        let w_ptr = weight.buf.as_ptr();
        let out_ptr = out.buf.as_ptr();
        let n_val = n;
        let eps_val = eps;

        let mut params: Vec<*mut c_void> = vec![
            &x_ptr as *const _ as *mut c_void,
            &w_ptr as *const _ as *mut c_void,
            &out_ptr as *const _ as *mut c_void,
            &n_val as *const _ as *mut c_void,
            &eps_val as *const _ as *mut c_void,
        ];

        let block_size = 256u32.min(n as u32);
        let shared_mem = block_size * 4; // float per thread

        let bytes = crate::profile::rmsnorm_bytes(batch * n as usize);
        let timer = crate::profile::begin_timer(&self.hip, "rmsnorm", "rmsnorm_f32", bytes);
        let result = self.launch_maybe_blob(
            "rmsnorm_f32", [batch as u32, 1, 1], [block_size, 1, 1], shared_mem, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(x_ptr); b.push_ptr(w_ptr); b.push_ptr(out_ptr);
                b.push_i32(n_val); b.push_f32(eps_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched RMSNorm: normalize `batch` vectors of length `n` independently.
    /// x and out can be the same buffer (in-place). Weight is [n], applied per vector.
    /// TriAttention sidecar calibration: accumulate band statistics for one
    /// chunk's Q tensor (batched across all tokens in the chunk).
    ///
    /// q_batch: [n_tokens, n_heads, head_dim] f32 pre-RoPE Q (already on GPU).
    /// accs_sum_re/im/abs: [n_layers * n_heads * n_bands] f64 accumulators.
    /// accs_count: [n_layers * n_heads * n_bands] u64 sample counters.
    /// All accs_* buffers persist across calls; the kernel ADDS into them.
    ///
    /// Grid = [n_heads, n_bands, 1]. Block = [64, 1, 1]. Zero cross-block
    /// contention since each (layer, head, band) is written by exactly one
    /// block at a time (called sequentially per layer per chunk).
    pub fn triattn_accumulate(
        &mut self,
        q_batch: &DeviceBuffer,
        accs_sum_re: &DeviceBuffer,
        accs_sum_im: &DeviceBuffer,
        accs_sum_abs: &DeviceBuffer,
        accs_count: &DeviceBuffer,
        n_tokens: usize, n_heads: usize, head_dim: usize,
        layer_idx: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "triattn_accumulate",
            kernels::TRIATTN_ACCUMULATE_SRC,
            "triattn_accumulate_f32",
        )?;

        let n_bands = head_dim / 2;

        let mut q_ptr = q_batch.as_ptr();
        let mut sre_ptr = accs_sum_re.as_ptr();
        let mut sim_ptr = accs_sum_im.as_ptr();
        let mut sab_ptr = accs_sum_abs.as_ptr();
        let mut cnt_ptr = accs_count.as_ptr();
        let mut nt = n_tokens as i32;
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut li = layer_idx as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut sre_ptr as *mut _ as *mut c_void,
            &mut sim_ptr as *mut _ as *mut c_void,
            &mut sab_ptr as *mut _ as *mut c_void,
            &mut cnt_ptr as *mut _ as *mut c_void,
            &mut nt as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut li as *mut _ as *mut c_void,
        ];

        self.launch_maybe_blob(
            "triattn_accumulate_f32",
            [n_heads as u32, n_bands as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(q_ptr); b.push_ptr(sre_ptr); b.push_ptr(sim_ptr);
                b.push_ptr(sab_ptr); b.push_ptr(cnt_ptr);
                b.push_i32(nt); b.push_i32(nh); b.push_i32(hd); b.push_i32(li);
                b
            },
        )
    }

    pub fn rmsnorm_batched(
        &mut self,
        x: &GpuTensor, weight: &GpuTensor, out: &GpuTensor,
        batch: usize, n: usize, eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("rmsnorm", kernels::RMSNORM_SRC, "rmsnorm_f32")?;

        let mut x_ptr = x.buf.as_ptr();
        let mut w_ptr = weight.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut n_val = n as i32;
        let mut eps_val = eps;

        let mut params: Vec<*mut c_void> = vec![
            &mut x_ptr as *mut _ as *mut c_void,
            &mut w_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut eps_val as *mut _ as *mut c_void,
        ];

        let block_size = 256u32.min(n as u32);
        let shared_mem = block_size * 4;
        let bytes = crate::profile::rmsnorm_bytes(batch * n);
        let timer = crate::profile::begin_timer(&self.hip, "rmsnorm", "rmsnorm_batched", bytes);
        let result = self.launch_maybe_blob(
            "rmsnorm_f32",
            [batch as u32, 1, 1],
            [block_size, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(x_ptr); b.push_ptr(w_ptr); b.push_ptr(out_ptr);
                b.push_i32(n_val); b.push_f32(eps_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// c = a + b (element-wise)
    pub fn add_f32(&mut self, a: &GpuTensor, b: &GpuTensor, c: &GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("add", kernels::ADD_SRC, "add_f32")?;
        let func = &self.functions["add_f32"];

        let n = a.numel() as i32;
        let mut a_ptr = a.buf.as_ptr();
        let mut b_ptr = b.buf.as_ptr();
        let mut c_ptr = c.buf.as_ptr();
        let mut n_val = n;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut b_ptr as *mut _ as *mut c_void,
            &mut c_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        unsafe { self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, None, &mut params) }
    }

    /// a += b (in-place element-wise add)
    pub fn add_inplace_f32(&mut self, a: &GpuTensor, b: &GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("add_inplace", kernels::ADD_INPLACE_SRC, "add_inplace_f32")?;

        let n = a.numel() as i32;
        let a_ptr = a.buf.as_ptr();
        let b_ptr = b.buf.as_ptr();
        let n_val = n;

        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &b_ptr as *const _ as *mut c_void,
            &n_val as *const _ as *mut c_void,
        ];

        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = crate::profile::elementwise_bytes(n as usize);
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", "add_inplace_f32", bytes);
        let result = self.launch_maybe_blob(
            "add_inplace_f32",
            [grid, 1, 1], [block, 1, 1], 0, &mut params,
            || {
                let mut bb = hip_bridge::KernargBlob::new();
                bb.push_ptr(a_ptr); bb.push_ptr(b_ptr); bb.push_i32(n_val);
                bb
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// c = a * b (element-wise)
    pub fn mul_f32(&mut self, a: &GpuTensor, b: &GpuTensor, c: &GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("mul", kernels::MUL_SRC, "mul_f32")?;
        let func = &self.functions["mul_f32"];

        let n = a.numel() as i32;
        let mut a_ptr = a.buf.as_ptr();
        let mut b_ptr = b.buf.as_ptr();
        let mut c_ptr = c.buf.as_ptr();
        let mut n_val = n;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut b_ptr as *mut _ as *mut c_void,
            &mut c_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = crate::profile::elementwise_bytes(n as usize);
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", "mul_f32", bytes);
        let result = unsafe { self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, None, &mut params) };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// out = silu(x)
    pub fn silu_f32(&mut self, x: &GpuTensor, out: &GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("silu", kernels::SILU_SRC, "silu_f32")?;
        let func = &self.functions["silu_f32"];

        let n = x.numel() as i32;
        let mut x_ptr = x.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut n_val = n;

        let mut params: Vec<*mut c_void> = vec![
            &mut x_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        unsafe { self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, None, &mut params) }
    }

    /// out = silu(gate) * up — fused to avoid intermediate buffer
    pub fn silu_mul_f32(&mut self, gate: &GpuTensor, up: &GpuTensor, out: &GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("silu_mul", kernels::SILU_MUL_SRC, "silu_mul_f32")?;

        let n = gate.numel() as i32;
        let mut gate_ptr = gate.buf.as_ptr();
        let mut up_ptr = up.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut n_val = n;

        let mut params: Vec<*mut c_void> = vec![
            &mut gate_ptr as *mut _ as *mut c_void,
            &mut up_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = crate::profile::elementwise_bytes(n as usize);
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", "silu_mul_f32", bytes);
        let result = self.launch_maybe_blob(
            "silu_mul_f32",
            [grid, 1, 1], [block, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(gate_ptr); b.push_ptr(up_ptr); b.push_ptr(out_ptr);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// In-place softmax over last dimension
    pub fn softmax_f32(&mut self, x: &GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("softmax", kernels::SOFTMAX_SRC, "softmax_f32")?;

        let rows = if x.shape.len() > 1 { x.shape[0] } else { 1 };
        let n = x.shape.last().copied().unwrap() as i32;

        let x_ptr = x.buf.as_ptr();
        let n_val = n;

        let mut params: Vec<*mut c_void> = vec![
            &x_ptr as *const _ as *mut c_void,
            &n_val as *const _ as *mut c_void,
        ];

        let block = 256u32.min(n as u32);
        let shared_mem = block * 4;

        // Graph-safe launch via launch_maybe_blob. Path B inserts this
        // call into the MoE forward path which gets captured under the
        // verify/HIPFIRE_GRAPH path; raw self.hip.launch_kernel would
        // capture stack-borne kernarg pointers that go dangling on replay.
        self.launch_maybe_blob(
            "softmax_f32",
            [rows as u32, 1, 1],
            [block, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(x_ptr);
                b.push_i32(n_val);
                b
            },
        )
    }

    /// GPU-side RoPE (rotary positional embedding) applied in-place to Q and K.
    /// pos_buf: GPU buffer containing a single i32 position value.
    pub fn rope_f32(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        pos_buf: &DeviceBuffer,
        n_heads_q: usize,
        n_heads_k: usize,
        head_dim: usize,
        freq_base: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("rope", kernels::ROPE_SRC, "rope_f32")?;
        let func = &self.functions["rope_f32"];

        let mut q_ptr = q.buf.as_ptr();
        let mut k_ptr = k.buf.as_ptr();
        let mut pos_ptr = pos_buf.as_ptr();
        let mut nhq = n_heads_q as i32;
        let mut nhk = n_heads_k as i32;
        let mut hd = head_dim as i32;
        let mut fb = freq_base;

        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut k_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void,
            &mut nhq as *mut _ as *mut c_void,
            &mut nhk as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut fb as *mut _ as *mut c_void,
        ];

        let half = (head_dim / 2) as u32;
        let block = 256u32.min(half);
        let grid = (half + block - 1) / block;

        unsafe {
            self.hip.launch_kernel(
                func,
                [grid, 1, 1],
                [block, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Batched RoPE: apply to [batch_size] positions in one launch.
    /// q: [batch_size × q_dim], k: [batch_size × kv_dim].
    /// positions: GPU buffer of [batch_size] i32 position indices.
    pub fn rope_batched_f32(
        &mut self, q: &GpuTensor, k: &GpuTensor, positions: &GpuTensor,
        n_heads_q: usize, n_heads_k: usize, head_dim: usize, freq_base: f32, batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("rope_batched", kernels::ROPE_BATCHED_SRC, "rope_batched_f32")?;
        let func = &self.functions["rope_batched_f32"];
        let mut q_ptr = q.buf.as_ptr();
        let mut k_ptr = k.buf.as_ptr();
        let mut pos_ptr = positions.buf.as_ptr();
        let mut nhq = n_heads_q as i32;
        let mut nhk = n_heads_k as i32;
        let mut hd = head_dim as i32;
        let mut fb = freq_base;
        let mut bs = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut k_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void,
            &mut nhq as *mut _ as *mut c_void,
            &mut nhk as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut fb as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        let half = (head_dim / 2) as u32;
        let block = 256u32.min(half);
        let grid_x = (half + block - 1) / block;
        unsafe {
            self.hip.launch_kernel(func, [grid_x, batch_size as u32, 1], [block, 1, 1], 0, self.stream_ref(), &mut params)
        }
    }

    /// GPU-side GQA attention.
    /// pos_buf: GPU buffer with single i32 position. Kernel computes seq_len = pos_buf[0] + 1.
    /// seq_len_hint: host-side seq_len for shared memory sizing (= pos + 1).
    pub fn attention_f32(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("attention", kernels::ATTENTION_SRC, "attention_f32")?;
        let func = &self.functions["attention_f32"];

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_ptr = q.buf.as_ptr();
        let mut k_ptr = k_cache.buf.as_ptr();
        let mut v_ptr = v_cache.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut pos_ptr = pos_buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut sc = scale;

        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];

        // When a stream is active (graph capture mode), use max_seq for shared mem
        // so the captured graph works for all sequence lengths.
        let effective_seq = if self.active_stream.is_some() { max_seq } else { seq_len_hint };
        let block_size = (effective_seq.max(head_dim) as u32).next_power_of_two().min(256);
        let shared_mem = ((effective_seq + block_size as usize) * 4) as u32;

        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Flash-decoding attention: split KV scan for long sequences.
    /// Automatically chooses single-block or multi-block based on seq_len.
    pub fn attention_flash(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        partials: &GpuTensor,
        seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        // Choose chunk size: aim for 4-16 chunks
        let chunk_size = if seq_len <= 128 { seq_len } else { 128 };
        let n_chunks = (seq_len + chunk_size - 1) / chunk_size;

        // Phase 1: compute partial attention per chunk
        self.ensure_kernel("attention_flash_partial", kernels::ATTENTION_FLASH_SRC, "attention_flash_partial")?;
        let func1 = &self.functions["attention_flash_partial"];

        let mut q_ptr = q.buf.as_ptr();
        let mut k_ptr = k_cache.buf.as_ptr();
        let mut v_ptr = v_cache.buf.as_ptr();
        let mut p_ptr = partials.buf.as_ptr();
        let mut sl = seq_len as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut sc = scale;
        let mut cs = chunk_size as i32;

        let mut params1: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void,
            &mut p_ptr as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut cs as *mut _ as *mut c_void,
        ];

        let block_size = 128u32.min(chunk_size as u32).next_power_of_two();
        let shared_mem = ((chunk_size + block_size as usize) * 4) as u32;

        unsafe {
            self.hip.launch_kernel(
                func1,
                [n_heads as u32, n_chunks as u32, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params1,
            )?;
        }

        // Phase 2: reduce partials
        self.ensure_kernel("attention_flash_reduce", kernels::ATTENTION_FLASH_SRC, "attention_flash_reduce")?;
        let func2 = &self.functions["attention_flash_reduce"];

        let mut p_ptr2 = partials.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut nh2 = n_heads as i32;
        let mut nc = n_chunks as i32;
        let mut hd2 = head_dim as i32;

        let mut params2: Vec<*mut c_void> = vec![
            &mut p_ptr2 as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut nh2 as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
            &mut hd2 as *mut _ as *mut c_void,
        ];

        let reduce_block = head_dim.min(256) as u32;
        unsafe {
            self.hip.launch_kernel(
                func2,
                [n_heads as u32, 1, 1],
                [reduce_block, 1, 1],
                0,
                self.stream_ref(),
                &mut params2,
            )
        }
    }

    /// Fused Gate+Up HFQ4-G256: two GEMVs in one launch.
    pub fn fused_gate_up_hfq4g256(
        &mut self,
        a_gate: &GpuTensor, a_up: &GpuTensor, x: &GpuTensor,
        y_gate: &GpuTensor, y_up: &GpuTensor,
        gate_m: usize, up_m: usize, k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // gfx906 dp4a opt-in: pre-quantize x to Q8_1 and use the
        // v_dot4_i32_i8 path. PMC at 2026-05-05 showed this kernel
        // was memory-bound; dp4a's 75% x-traffic reduction lands on
        // the actual bottleneck.
        if gemv_dp4a_enabled(&self.arch) {
            return self.fused_gate_up_hfq4g256_dp4a(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k,
            );
        }

        let cdna_wave64 = has_wave64_native(&self.arch);
        let (func_name, block, grid_x) = if cdna_wave64 {
            self.ensure_kernel(
                "fused_gate_up_hfq4g256_wave64",
                kernels::FUSED_GATE_UP_HFQ4G256_WAVE64_SRC,
                "fused_gate_up_hfq4g256_wave64",
            )?;
            let total = (gate_m + up_m) as u32;
            ("fused_gate_up_hfq4g256_wave64", [64u32, 1, 1], (total + 1) / 2)
        } else {
            self.ensure_kernel(
                "fused_gate_up_hfq4g256",
                kernels::FUSED_GATE_UP_HFQ4G256_SRC,
                "fused_gate_up_hfq4g256",
            )?;
            ("fused_gate_up_hfq4g256", [32u32, 1, 1], (gate_m + up_m) as u32)
        };
        let ag = a_gate.buf.as_ptr();
        let au = a_up.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yg = y_gate.buf.as_ptr();
        let yu = y_up.buf.as_ptr();
        let gm = gate_m as i32;
        let um = up_m as i32;
        let kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ag as *const _ as *mut c_void, &au as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void, &yg as *const _ as *mut c_void,
            &yu as *const _ as *mut c_void, &gm as *const _ as *mut c_void,
            &um as *const _ as *mut c_void, &kv as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            func_name, [grid_x, 1, 1], block, 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ag); b.push_ptr(au); b.push_ptr(xp);
                b.push_ptr(yg); b.push_ptr(yu);
                b.push_i32(gm); b.push_i32(um); b.push_i32(kv);
                b
            },
        )
    }

    /// dp4a-port of fused_gate_up_hfq4g256 for gfx906. Pre-quantizes
    /// `x` to Q8_1 (block_q8_1_mmq, 144 B per 128-K block) using the
    /// shared MMQ x-scratch buffer, then runs the dp4a-based GEMV. Math
    /// is identical modulo Q8_1 quant noise (~1 % per-element relative).
    /// Targeted at gfx906 where the FP wave64 fused_gate_up sat at
    /// 41 % VALUBusy + 3.86 % MemUnitStalled — memory-bound, so dp4a's
    /// 75 % x-traffic reduction lands on the actual bottleneck.
    pub fn fused_gate_up_hfq4g256_dp4a(
        &mut self,
        a_gate: &GpuTensor, a_up: &GpuTensor, x: &GpuTensor,
        y_gate: &GpuTensor, y_up: &GpuTensor,
        gate_m: usize, up_m: usize, k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Quantize x → Xq[K/128] block_q8_1_mmq via the existing shared
        // scratch path. Batch=1 for GEMV.
        let xq_ptr = self.ensure_q8_1_mmq_x(x, 1, k)?;

        self.ensure_kernel(
            "fused_gate_up_hfq4g256_wave64_dp4a",
            kernels::FUSED_GATE_UP_HFQ4G256_WAVE64_DP4A_SRC,
            "fused_gate_up_hfq4g256_wave64_dp4a",
        )?;

        let ag = a_gate.buf.as_ptr();
        let au = a_up.buf.as_ptr();
        let yg = y_gate.buf.as_ptr();
        let yu = y_up.buf.as_ptr();
        let gm = gate_m as i32;
        let um = up_m as i32;
        let kv = k as i32;
        let total = (gate_m + up_m) as u32;
        let mut xq = xq_ptr;
        let mut params: Vec<*mut c_void> = vec![
            &ag as *const _ as *mut c_void,
            &au as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &yg as *const _ as *mut c_void,
            &yu as *const _ as *mut c_void,
            &gm as *const _ as *mut c_void,
            &um as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "fused_gate_up_hfq4g256_wave64_dp4a",
            [(total + 1) / 2, 1, 1], [64, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ag); b.push_ptr(au); b.push_ptr(xq);
                b.push_ptr(yg); b.push_ptr(yu);
                b.push_i32(gm); b.push_i32(um); b.push_i32(kv);
                b
            },
        )
    }

    /// Write KV to HFQ4 co-located block (72 bytes per head: scale+zero+nibbles).
    pub fn kv_cache_write_hfq4(
        &mut self, dst: &GpuTensor, src: &GpuTensor, pos_buf: &DeviceBuffer,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("kv_cache_write_hfq4", kernels::KV_CACHE_WRITE_HFQ4_SRC, "kv_cache_write_hfq4")?;
        let func = &self.functions["kv_cache_write_hfq4"];
        let mut d = dst.buf.as_ptr(); let mut s = src.buf.as_ptr();
        let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32; let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut d as *mut _ as *mut c_void, &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// Attention with HFQ4 KV blocks (72 bytes per head, co-located).
    pub fn attention_hfq4_kv(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer, seq_len_hint: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("attention_hfq4_kv", kernels::ATTENTION_HFQ4_KV_SRC, "attention_hfq4_kv")?;
        let func = &self.functions["attention_hfq4_kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr(); let mut kp = k_cache.buf.as_ptr();
        let mut vp = v_cache.buf.as_ptr(); let mut op = out.buf.as_ptr();
        let mut pp = pos_buf.as_ptr();
        let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32; let mut ms = max_seq as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void, &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void, &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32).next_power_of_two().min(256);
        // scores[seq_len] + ws[block_size] + q_shared[head_dim]
        let shared_mem = ((seq_len_hint + block_size as usize + head_dim) * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// INT8 co-located with f16 scale (matches Q8_0 precision, one block per head).
    pub fn kv_cache_write_int8c_f16(
        &mut self, dst: &GpuTensor, src: &GpuTensor, pos_buf: &DeviceBuffer,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("kv_cache_write_int8c_f16", kernels::KV_CACHE_WRITE_INT8C_F16_SRC, "kv_cache_write_int8c_f16")?;
        let func = &self.functions["kv_cache_write_int8c_f16"];
        let mut d = dst.buf.as_ptr(); let mut s = src.buf.as_ptr(); let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32; let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut d as *mut _ as *mut c_void, &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    pub fn attention_int8c_f16_kv(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer, seq_len_hint: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("attention_int8c_f16_kv", kernels::ATTENTION_INT8C_F16_KV_SRC, "attention_int8c_f16_kv")?;
        let func = &self.functions["attention_int8c_f16_kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr(); let mut kp = k_cache.buf.as_ptr();
        let mut vp = v_cache.buf.as_ptr(); let mut op = out.buf.as_ptr(); let mut pp = pos_buf.as_ptr();
        let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32; let mut ms = max_seq as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void, &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void, &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32).next_power_of_two().min(256);
        let shared_mem = ((seq_len_hint + block_size as usize) * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// Write KV to INT8 co-located block (f32 scale + int8 data, symmetric).
    pub fn kv_cache_write_int8c(
        &mut self, dst: &GpuTensor, src: &GpuTensor, pos_buf: &DeviceBuffer,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("kv_cache_write_int8c", kernels::KV_CACHE_WRITE_INT8C_SRC, "kv_cache_write_int8c")?;
        let func = &self.functions["kv_cache_write_int8c"];
        let mut d = dst.buf.as_ptr(); let mut s = src.buf.as_ptr();
        let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32; let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut d as *mut _ as *mut c_void, &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// Attention with INT8 co-located KV blocks.
    pub fn attention_int8c_kv(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer, seq_len_hint: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("attention_int8c_kv", kernels::ATTENTION_INT8C_KV_SRC, "attention_int8c_kv")?;
        let func = &self.functions["attention_int8c_kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr(); let mut kp = k_cache.buf.as_ptr();
        let mut vp = v_cache.buf.as_ptr(); let mut op = out.buf.as_ptr();
        let mut pp = pos_buf.as_ptr();
        let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32; let mut ms = max_seq as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void, &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void, &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32).next_power_of_two().min(256);
        let shared_mem = ((seq_len_hint + block_size as usize + head_dim) * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// Write KV to HFQ8 cache (FP32 scale+zero, contiguous uint8).
    pub fn kv_cache_write_hfq8(
        &mut self, dst_data: &GpuTensor, dst_scales: &GpuTensor, src: &GpuTensor,
        pos_buf: &DeviceBuffer, n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("kv_cache_write_hfq8", kernels::KV_CACHE_WRITE_HFQ8_SRC, "kv_cache_write_hfq8")?;
        let func = &self.functions["kv_cache_write_hfq8"];
        let mut dd = dst_data.buf.as_ptr(); let mut ds = dst_scales.buf.as_ptr();
        let mut s = src.buf.as_ptr(); let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32; let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut dd as *mut _ as *mut c_void, &mut ds as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void, &mut p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// Attention with HFQ8 KV cache.
    pub fn attention_hfq8_kv(
        &mut self, q: &GpuTensor,
        k_data: &GpuTensor, k_scales: &GpuTensor,
        v_data: &GpuTensor, v_scales: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer, seq_len_hint: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("attention_hfq8_kv", kernels::ATTENTION_HFQ8_KV_SRC, "attention_hfq8_kv")?;
        let func = &self.functions["attention_hfq8_kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr();
        let mut kd = k_data.buf.as_ptr(); let mut ks = k_scales.buf.as_ptr();
        let mut vd = v_data.buf.as_ptr(); let mut vs = v_scales.buf.as_ptr();
        let mut op = out.buf.as_ptr(); let mut pp = pos_buf.as_ptr();
        let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32; let mut ms = max_seq as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kd as *mut _ as *mut c_void, &mut ks as *mut _ as *mut c_void,
            &mut vd as *mut _ as *mut c_void, &mut vs as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void, &mut pp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32).next_power_of_two().min(256);
        let shared_mem = ((seq_len_hint + block_size as usize) * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// Write KV to INT8 cache (separate scale array).
    pub fn kv_cache_write_int8(
        &mut self, dst_vals: &GpuTensor, dst_scales: &GpuTensor, src: &GpuTensor,
        pos_buf: &DeviceBuffer, n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("kv_cache_write_int8", kernels::KV_CACHE_WRITE_INT8_SRC, "kv_cache_write_int8")?;
        let func = &self.functions["kv_cache_write_int8"];
        let mut dv = dst_vals.buf.as_ptr(); let mut ds = dst_scales.buf.as_ptr();
        let mut s = src.buf.as_ptr(); let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32; let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut dv as *mut _ as *mut c_void, &mut ds as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void, &mut p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// Attention with INT8 KV (separate scale array).
    pub fn attention_int8_kv(
        &mut self, q: &GpuTensor,
        k_vals: &GpuTensor, k_scales: &GpuTensor,
        v_vals: &GpuTensor, v_scales: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer, seq_len_hint: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("attention_int8_kv", kernels::ATTENTION_INT8_KV_SRC, "attention_int8_kv")?;
        let func = &self.functions["attention_int8_kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_ptr = q.buf.as_ptr();
        let mut kv_ptr = k_vals.buf.as_ptr(); let mut ks_ptr = k_scales.buf.as_ptr();
        let mut vv_ptr = v_vals.buf.as_ptr(); let mut vs_ptr = v_scales.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr(); let mut pos_ptr = pos_buf.as_ptr();
        let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32; let mut ms = max_seq as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut kv_ptr as *mut _ as *mut c_void, &mut ks_ptr as *mut _ as *mut c_void,
            &mut vv_ptr as *mut _ as *mut c_void, &mut vs_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void, &mut pos_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32).next_power_of_two().min(256);
        let shared_mem = ((seq_len_hint + block_size as usize) * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// Batched causal attention: all query positions in one launch.
    /// Q: [seq_len × n_heads × head_dim], K/V: [seq_len × n_kv_heads × head_dim].
    pub fn attention_causal_batched(
        &mut self, q: &GpuTensor, k: &GpuTensor, v: &GpuTensor, out: &GpuTensor,
        seq_len: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("attention_causal_batched", kernels::ATTENTION_CAUSAL_BATCHED_SRC, "attention_causal_batched")?;
        let func = &self.functions["attention_causal_batched"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr(); let mut kp = k.buf.as_ptr();
        let mut vp = v.buf.as_ptr(); let mut op = out.buf.as_ptr();
        let mut sl = seq_len as i32; let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32; let mut hd = head_dim as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void, &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        // Block size: enough threads to cover head_dim and seq_len
        let block_size = 128u32.min((seq_len.max(head_dim) as u32).next_power_of_two());
        // Shared: scores[seq_len] + workspace[block_size]
        let shared_mem = ((seq_len + block_size as usize) * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, seq_len as u32, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// Batched Q8_0 KV cache write: quantize multiple positions in one launch.
    pub fn kv_cache_write_q8_0_batched(
        &mut self, dst: &GpuTensor, src: &GpuTensor, positions: &GpuTensor,
        n_kv_heads: usize, head_dim: usize, batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("kv_cache_write_q8_0_batched", kernels::KV_CACHE_WRITE_Q8_0_BATCHED_SRC, "kv_cache_write_q8_0_batched")?;
        let mut d = dst.buf.as_ptr(); let mut s = src.buf.as_ptr();
        let mut p = positions.buf.as_ptr();
        let mut nkv = n_kv_heads as i32; let mut hd = head_dim as i32; let mut bs = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut d as *mut _ as *mut c_void, &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut bs as *mut _ as *mut c_void,
        ];
        let total_blocks = (n_kv_heads * head_dim / 32) as u32;
        self.launch_maybe_blob(
            "kv_cache_write_q8_0_batched",
            [total_blocks, batch_size as u32, 1], [32, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(d); b.push_ptr(s); b.push_ptr(p);
                b.push_i32(nkv); b.push_i32(hd); b.push_i32(bs);
                b
            },
        )
    }

    /// Write KV vector to Q8_0 quantized cache (same format as GGML Q8_0).
    pub fn kv_cache_write_q8_0(
        &mut self, dst: &GpuTensor, src: &GpuTensor, pos_buf: &DeviceBuffer,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("kv_cache_write_q8_0", kernels::KV_CACHE_WRITE_Q8_0_SRC, "kv_cache_write_q8_0")?;
        let d = dst.buf.as_ptr();
        let s = src.buf.as_ptr();
        let p = pos_buf.as_ptr();
        let nkv = n_kv_heads as i32;
        let hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &d as *const _ as *mut c_void, &s as *const _ as *mut c_void,
            &p as *const _ as *mut c_void, &nkv as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
        ];
        let total_blocks = (n_kv_heads * head_dim / 32) as u32;
        let bytes = crate::profile::kv_cache_write_q8_0_bytes(n_kv_heads, head_dim);
        let timer = crate::profile::begin_timer(&self.hip, "kv_write", "kv_cache_write_q8_0", bytes);
        let result = self.launch_maybe_blob(
            "kv_cache_write_q8_0",
            [total_blocks, 1, 1], [32, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(d); b.push_ptr(s); b.push_ptr(p);
                b.push_i32(nkv); b.push_i32(hd);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched causal attention with Q8_0 quantized KV cache. Processes N
    /// queries in one launch; each query b has its own causal window read
    /// from positions[b] (i.e. attend to 0..positions[b]+1). Q and out are
    /// [batch_size × n_heads × head_dim] row-major; K/V caches are the same
    /// layout as `attention_q8_0_kv` and must already contain the prefix
    /// through positions[batch_size-1].
    ///
    /// Byte-exact with N single-token calls at batch_size=1, positions[0]=pos.
    ///
    /// `max_ctx_len` is the maximum seq_len = max(positions[b]) + 1 across
    /// the batch; used to size the shared memory allocation for scores[].
    pub fn attention_q8_0_kv_batched(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.attention_q8_0_kv_batched_masked(
            q, k_cache, v_cache, out, positions,
            n_heads, n_kv_heads, head_dim, max_seq, max_ctx_len, batch_size,
            None, 0, 0,
        )
    }

    /// Tree-mask variant of `attention_q8_0_kv_batched`. When `tree_bias` is
    /// `Some`, the kernel ignores the causal cutoff and iterates over
    /// `[0, block_start + block_cols)`, applying an additive bias from
    /// `tree_bias[b × block_cols + (t - block_start)]` for in-block keys.
    /// Caller passes `-inf` on non-ancestor slots and `0.0` on ancestors
    /// (see `hipfire_runtime::ddtree::linearize_tree`).
    ///
    /// When `tree_bias` is `None`, `block_start` / `block_cols` are ignored
    /// and behavior is byte-identical to the legacy causal path.
    ///
    /// Shared memory: the tree-mode `seq_len` is always `block_start +
    /// block_cols`. Caller must pass `max_ctx_len` ≥ that value so the
    /// scores[] LDS slice is sized correctly.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_q8_0_kv_batched_masked(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        tree_bias: Option<&GpuTensor>,
        block_start: usize,
        block_cols: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "attention_q8_0_kv_batched",
            kernels::ATTENTION_Q8_0_KV_BATCHED_SRC,
            "attention_q8_0_kv_batched",
        )?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_ptr = q.buf.as_ptr();
        let mut k_ptr = k_cache.buf.as_ptr();
        let mut v_ptr = v_cache.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut pos_ptr = positions.buf.as_ptr();
        // tree_bias = null when None; the kernel branches on bias != nullptr.
        let mut bias_ptr: *mut std::ffi::c_void = match tree_bias {
            Some(t) => t.buf.as_ptr(),
            None => std::ptr::null_mut(),
        };
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut sc = scale;
        let mut bs = block_start as i32;
        let mut bc = block_cols as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void,
            &mut bias_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut bc as *mut _ as *mut c_void,
        ];
        let block_size = (max_ctx_len.max(head_dim) as u32).next_power_of_two().min(256);
        // Shared memory must accommodate the LARGEST batch row's seq_len for
        // scores[], plus nthreads workspace and head_dim q_shared.
        let shared_mem = ((max_ctx_len + block_size as usize + head_dim) * 4) as u32;
        let bytes = crate::profile::attention_q8_0_kv_bytes(n_heads, n_kv_heads, head_dim, max_ctx_len) * batch_size;
        let timer = crate::profile::begin_timer(&self.hip, "attention", "attention_q8_0_kv_batched", bytes);
        let bias_raw = bias_ptr; // alias for move into closure
        let result = self.launch_maybe_blob(
            "attention_q8_0_kv_batched",
            [n_heads as u32, batch_size as u32, 1],
            [block_size, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(q_ptr); b.push_ptr(k_ptr); b.push_ptr(v_ptr);
                b.push_ptr(out_ptr); b.push_ptr(pos_ptr); b.push_ptr(bias_raw);
                b.push_i32(nh); b.push_i32(nkv); b.push_i32(hd); b.push_i32(ms);
                b.push_f32(sc); b.push_i32(bs); b.push_i32(bc);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Flash attention with Q8_0 KV cache — tile + reduce two-kernel path.
    /// Tiles seq_len into chunks of `tile_size`, launches [n_heads, n_tiles]
    /// blocks for the tile kernel, then [n_heads] blocks for the reduce.
    /// Requires a pre-allocated `partials` buffer of size
    /// n_heads * max_tiles * (2 + head_dim) floats.
    pub fn attention_flash_q8_0(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer, seq_len_hint: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const TILE_SIZE: usize = 128;
        // Graph-safe: use max_tiles so the grid is position-independent.
        // The tile kernel exits early for tiles beyond actual seq_len.
        let max_tiles = (max_seq + TILE_SIZE - 1) / TILE_SIZE;
        // For profiling / non-graph code paths, the actual tile count:
        let actual_tiles = (seq_len_hint + TILE_SIZE - 1) / TILE_SIZE;
        let launch_tiles = if self.capture_mode { max_tiles } else { actual_tiles };

        // ── Tile kernel ──
        self.ensure_kernel(
            "attention_flash_q8_0_tile",
            kernels::ATTENTION_FLASH_Q8_0_TILE_SRC,
            "attention_flash_q8_0_tile",
        )?;
        {
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let q_ptr = q.buf.as_ptr();
            let k_ptr = k_cache.buf.as_ptr();
            let v_ptr = v_cache.buf.as_ptr();
            let p_ptr = partials.buf.as_ptr();
            let pos_ptr = pos_buf.as_ptr();
            let nh = n_heads as i32; let nkv = n_kv_heads as i32;
            let hd = head_dim as i32; let ms = max_seq as i32;
            let sc = scale; let ts = TILE_SIZE as i32;
            let grid = [n_heads as u32, launch_tiles as u32, 1];
            let shared = ((TILE_SIZE + head_dim) * 4) as u32;
            let mut params: Vec<*mut c_void> = vec![
                &q_ptr as *const _ as *mut c_void, &k_ptr as *const _ as *mut c_void,
                &v_ptr as *const _ as *mut c_void, &p_ptr as *const _ as *mut c_void,
                &pos_ptr as *const _ as *mut c_void, &nh as *const _ as *mut c_void,
                &nkv as *const _ as *mut c_void, &hd as *const _ as *mut c_void,
                &ms as *const _ as *mut c_void, &sc as *const _ as *mut c_void,
                &ts as *const _ as *mut c_void,
            ];
            self.launch_maybe_blob(
                "attention_flash_q8_0_tile", grid, [32, 1, 1], shared, &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(q_ptr); b.push_ptr(k_ptr); b.push_ptr(v_ptr);
                    b.push_ptr(p_ptr); b.push_ptr(pos_ptr);
                    b.push_i32(nh); b.push_i32(nkv); b.push_i32(hd); b.push_i32(ms);
                    b.push_f32(sc); b.push_i32(ts);
                    b
                },
            )?;
        }

        // ── Reduce kernel (reads seq_len from pos_buf, computes n_tiles) ──
        self.ensure_kernel(
            "attention_flash_q8_0_reduce",
            kernels::ATTENTION_FLASH_Q8_0_REDUCE_SRC,
            "attention_flash_q8_0_reduce",
        )?;
        {
            let p_ptr = partials.buf.as_ptr();
            let o_ptr = out.buf.as_ptr();
            let nh = n_heads as i32;
            let hd = head_dim as i32;
            let pos_ptr = pos_buf.as_ptr();
            let ts = TILE_SIZE as i32;
            let mt = max_tiles as i32;
            let mut params: Vec<*mut c_void> = vec![
                &p_ptr as *const _ as *mut c_void, &o_ptr as *const _ as *mut c_void,
                &nh as *const _ as *mut c_void, &hd as *const _ as *mut c_void,
                &pos_ptr as *const _ as *mut c_void, &ts as *const _ as *mut c_void,
                &mt as *const _ as *mut c_void,
            ];
            self.launch_maybe_blob(
                "attention_flash_q8_0_reduce", [n_heads as u32, 1, 1], [32, 1, 1], 0, &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(p_ptr); b.push_ptr(o_ptr);
                    b.push_i32(nh); b.push_i32(hd);
                    b.push_ptr(pos_ptr); b.push_i32(ts); b.push_i32(mt);
                    b
                },
            )?;
        }
        Ok(())
    }

    /// Compile a givens4 kernel — prepends turbo_common + givens_common headers.
    fn ensure_givens4_kernel(&mut self, name: &str, body_src: &str, func_name: &str) -> HipResult<()> {
        if self.functions.contains_key(func_name) {
            return Ok(());
        }
        let stripped = body_src
            .replace("#include \"turbo_common.h\"", "")
            .replace("#include \"givens_common.h\"", "");
        let full_src = format!("{}\n{}\n{}", kernels::TURBO_COMMON_H, kernels::GIVENS_COMMON_SRC, stripped);
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


    /// Fused K+V write for asym4: K at givens4 (rotated 4-bit), V at Q8_0 (normal space).
    /// Launches two kernels — K-only givens4 writer + standard Q8_0 writer.
    pub fn kv_cache_write_asym4_fused(
        &mut self, k_dst: &GpuTensor, v_dst: &GpuTensor,
        k_src: &GpuTensor, v_src: &GpuTensor, pos_buf: &DeviceBuffer,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // K: rotated 4-bit
        self.ensure_givens4_kernel(
            "kv_cache_write_asym_k_givens4",
            kernels::KV_CACHE_WRITE_ASYM_K_GIVENS4_SRC,
            "kv_cache_write_asym_k_givens4",
        )?;
        {
            let func = &self.functions["kv_cache_write_asym_k_givens4"];
            let mut kdp = k_dst.buf.as_ptr();
            let mut ksp = k_src.buf.as_ptr();
            let mut pp = pos_buf.as_ptr();
            let mut ctp = cos_theta.buf.as_ptr();
            let mut stp = sin_theta.buf.as_ptr();
            let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut kdp as *mut _ as *mut c_void,
                &mut ksp as *mut _ as *mut c_void,
                &mut pp as *mut _ as *mut c_void,
                &mut ctp as *mut _ as *mut c_void,
                &mut stp as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
            ];
            let shared_mem = ((head_dim + 32) * 4) as u32;
            unsafe {
                self.hip.launch_kernel(
                    func, [n_kv_heads as u32, 1, 1], [32, 1, 1], shared_mem,
                    self.stream_ref(), &mut params,
                )?;
            }
        }
        // V: standard Q8_0
        self.kv_cache_write_q8_0(v_dst, v_src, pos_buf, n_kv_heads, head_dim)
    }

    /// Fused K+V write for asym3: K at 3-bit rotated (RotorQuant "planar3"), V at Q8_0.
    /// Best-quality rotated K per RotorQuant paper. Head geometry: 32 threads × 8
    /// values = 256 dims single-pass. 100 bytes/head for hd=256.
    pub fn kv_cache_write_asym3_fused(
        &mut self, k_dst: &GpuTensor, v_dst: &GpuTensor,
        k_src: &GpuTensor, v_src: &GpuTensor, pos_buf: &DeviceBuffer,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "kv_cache_write_asym_k_givens3",
            kernels::KV_CACHE_WRITE_ASYM_K_GIVENS3_SRC,
            "kv_cache_write_asym_k_givens3",
        )?;
        {
            let func = &self.functions["kv_cache_write_asym_k_givens3"];
            let mut kdp = k_dst.buf.as_ptr();
            let mut ksp = k_src.buf.as_ptr();
            let mut pp = pos_buf.as_ptr();
            let mut ctp = cos_theta.buf.as_ptr();
            let mut stp = sin_theta.buf.as_ptr();
            let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut kdp as *mut _ as *mut c_void,
                &mut ksp as *mut _ as *mut c_void,
                &mut pp as *mut _ as *mut c_void,
                &mut ctp as *mut _ as *mut c_void,
                &mut stp as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
            ];
            let shared_mem = ((head_dim + 32) * 4) as u32;
            unsafe {
                self.hip.launch_kernel(
                    func, [n_kv_heads as u32, 1, 1], [32, 1, 1], shared_mem,
                    self.stream_ref(), &mut params,
                )?;
            }
        }
        self.kv_cache_write_q8_0(v_dst, v_src, pos_buf, n_kv_heads, head_dim)
    }

    /// Shared helper: launch a batched K-only rotated write kernel.
    fn launch_asym_k_batched(
        &mut self, kernel_key: &str, src_const: &'static str, func_name: &'static str,
        k_dst: &GpuTensor, k_src: &GpuTensor, positions: &GpuTensor,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        n_kv_heads: usize, head_dim: usize, batch_size: usize,
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
                b.push_ptr(kdp); b.push_ptr(ksp); b.push_ptr(pp);
                b.push_ptr(ctp); b.push_ptr(stp);
                b.push_i32(nkv); b.push_i32(hd); b.push_i32(bs);
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
        tile_key: &'static str, tile_src: &'static str, tile_func_name: &'static str,
        q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, positions: &GpuTensor,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        n_heads: usize, n_kv_heads: usize, head_dim: usize,
        max_seq: usize, max_ctx_len: usize, batch_size: usize,
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
                let q_ptr = unsafe {
                    (q.buf.as_ptr() as *mut u8).add(offset * q_dim * 4) as *mut c_void
                };
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
                let nh = n_heads as i32; let nkv = n_kv_heads as i32;
                let hd = head_dim as i32; let ms = max_seq as i32;
                let sc = scale; let ts = TILE_SIZE as i32;
                let mt = max_tiles as i32; let bo = offset as i32;
                let bs = block_start as i32; let bc = block_cols as i32;
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
                        b.push_ptr(q_ptr); b.push_ptr(k_ptr); b.push_ptr(v_ptr);
                        b.push_ptr(p_ptr); b.push_ptr(pos_ptr);
                        b.push_ptr(ct_ptr); b.push_ptr(st_ptr); b.push_ptr(bias_ptr);
                        b.push_i32(nh); b.push_i32(nkv); b.push_i32(hd); b.push_i32(ms);
                        b.push_f32(sc); b.push_i32(ts); b.push_i32(mt); b.push_i32(bo);
                        b.push_i32(bs); b.push_i32(bc);
                        b
                    },
                )?;
            }
            {
                let p_ptr = partials.buf.as_ptr();
                let o_ptr = unsafe {
                    (out.buf.as_ptr() as *mut u8).add(offset * q_dim * 4) as *mut c_void
                };
                let pos_ptr = positions.buf.as_ptr();
                let nh = n_heads as i32; let hd = head_dim as i32;
                let ts = TILE_SIZE as i32; let mt = max_tiles as i32;
                let bo = offset as i32;
                let bs = block_start as i32; let bc = block_cols as i32;
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
                        b.push_ptr(p_ptr); b.push_ptr(o_ptr); b.push_ptr(pos_ptr);
                        b.push_i32(nh); b.push_i32(hd); b.push_i32(ts); b.push_i32(mt);
                        b.push_i32(bo); b.push_i32(bs); b.push_i32(bc);
                        b
                    },
                )?;
            }
            offset += chunk;
        }
        Ok(())
    }

    /// Batched K+V write for asym4 (K 4-bit rotated + V Q8_0).
    pub fn kv_cache_write_asym4_batched(
        &mut self,
        k_dst: &GpuTensor, v_dst: &GpuTensor,
        k_src: &GpuTensor, v_src: &GpuTensor, positions: &GpuTensor,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        n_kv_heads: usize, head_dim: usize, batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_k_batched(
            "kv_cache_write_asym_k_givens4_batched",
            kernels::KV_CACHE_WRITE_ASYM_K_GIVENS4_BATCHED_SRC,
            "kv_cache_write_asym_k_givens4_batched",
            k_dst, k_src, positions, cos_theta, sin_theta,
            n_kv_heads, head_dim, batch_size,
        )?;
        self.kv_cache_write_q8_0_batched(v_dst, v_src, positions, n_kv_heads, head_dim, batch_size)
    }

    /// Batched K+V write for asym2 (K 2-bit rotated + V Q8_0).
    pub fn kv_cache_write_asym2_batched(
        &mut self,
        k_dst: &GpuTensor, v_dst: &GpuTensor,
        k_src: &GpuTensor, v_src: &GpuTensor, positions: &GpuTensor,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        n_kv_heads: usize, head_dim: usize, batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_k_batched(
            "kv_cache_write_asym_k_givens2_batched",
            kernels::KV_CACHE_WRITE_ASYM_K_GIVENS2_BATCHED_SRC,
            "kv_cache_write_asym_k_givens2_batched",
            k_dst, k_src, positions, cos_theta, sin_theta,
            n_kv_heads, head_dim, batch_size,
        )?;
        self.kv_cache_write_q8_0_batched(v_dst, v_src, positions, n_kv_heads, head_dim, batch_size)
    }

    /// Batched flash attention for asym4 (K 4-bit rotated + V Q8_0).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_asym4_batched(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, positions: &GpuTensor,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        n_heads: usize, n_kv_heads: usize, head_dim: usize,
        max_seq: usize, max_ctx_len: usize, batch_size: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.attention_flash_asym4_batched_masked(
            q, k_cache, v_cache, out, positions, cos_theta, sin_theta,
            n_heads, n_kv_heads, head_dim, max_seq, max_ctx_len, batch_size, partials,
            None, 0, 0,
        )
    }

    /// Tree-mask variant of `attention_flash_asym4_batched`. See
    /// `attention_q8_0_kv_batched_masked` and `ddtree::linearize_tree` for the
    /// bias layout. Passes `tree_bias` / `block_start` / `block_cols` into the
    /// tile + reduce kernels.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_asym4_batched_masked(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, positions: &GpuTensor,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        n_heads: usize, n_kv_heads: usize, head_dim: usize,
        max_seq: usize, max_ctx_len: usize, batch_size: usize,
        partials: &GpuTensor,
        tree_bias: Option<&GpuTensor>,
        block_start: usize,
        block_cols: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_flash_batched(
            "attention_flash_asym4_tile_batched",
            kernels::ATTENTION_FLASH_ASYM4_TILE_BATCHED_SRC,
            "attention_flash_asym4_tile_batched",
            q, k_cache, v_cache, out, positions, cos_theta, sin_theta,
            n_heads, n_kv_heads, head_dim, max_seq, max_ctx_len, batch_size, partials,
            tree_bias, block_start, block_cols,
        )
    }

    /// Batched flash attention for asym2 (K 2-bit rotated + V Q8_0).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_asym2_batched(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, positions: &GpuTensor,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        n_heads: usize, n_kv_heads: usize, head_dim: usize,
        max_seq: usize, max_ctx_len: usize, batch_size: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_flash_batched(
            "attention_flash_asym2_tile_batched",
            kernels::ATTENTION_FLASH_ASYM2_TILE_BATCHED_SRC,
            "attention_flash_asym2_tile_batched",
            q, k_cache, v_cache, out, positions, cos_theta, sin_theta,
            n_heads, n_kv_heads, head_dim, max_seq, max_ctx_len, batch_size, partials,
            None, 0, 0,
        )
    }

    /// Batched K+V write for asym3 — processes N positions in one launch.
    /// K-only givens3 write (batched) + Q8_0 V write (batched).
    pub fn kv_cache_write_asym3_batched(
        &mut self,
        k_dst: &GpuTensor, v_dst: &GpuTensor,
        k_src: &GpuTensor, v_src: &GpuTensor, positions: &GpuTensor,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        n_kv_heads: usize, head_dim: usize, batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // K: batched 3-bit rotated write.
        self.ensure_givens4_kernel(
            "kv_cache_write_asym_k_givens3_batched",
            kernels::KV_CACHE_WRITE_ASYM_K_GIVENS3_BATCHED_SRC,
            "kv_cache_write_asym_k_givens3_batched",
        )?;
        {
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
                "kv_cache_write_asym_k_givens3_batched",
                [n_kv_heads as u32, batch_size as u32, 1],
                [32, 1, 1],
                shared_mem,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(kdp); b.push_ptr(ksp); b.push_ptr(pp);
                    b.push_ptr(ctp); b.push_ptr(stp);
                    b.push_i32(nkv); b.push_i32(hd); b.push_i32(bs);
                    b
                },
            )?;
        }
        // V: batched Q8_0 write.
        self.kv_cache_write_q8_0_batched(v_dst, v_src, positions, n_kv_heads, head_dim, batch_size)
    }

    /// Batched flash attention for asym3 KV.
    /// Grid: [n_heads, max_tiles, sub_batch] tile + [n_heads, sub_batch] reduce,
    /// chunked by partials buffer capacity.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_asym3_batched(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, positions: &GpuTensor,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        n_heads: usize, n_kv_heads: usize, head_dim: usize,
        max_seq: usize, max_ctx_len: usize, batch_size: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.attention_flash_asym3_batched_masked(
            q, k_cache, v_cache, out, positions, cos_theta, sin_theta,
            n_heads, n_kv_heads, head_dim, max_seq, max_ctx_len, batch_size, partials,
            None, 0, 0,
        )
    }

    /// Tree-mask variant of `attention_flash_asym3_batched`. asym3 is the
    /// default live KV path on 9B MQ4 — this is the primary target for
    /// DDTree batched verify on the hybrid arch.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_asym3_batched_masked(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, positions: &GpuTensor,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        n_heads: usize, n_kv_heads: usize, head_dim: usize,
        max_seq: usize, max_ctx_len: usize, batch_size: usize,
        partials: &GpuTensor,
        tree_bias: Option<&GpuTensor>,
        block_start: usize,
        block_cols: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_flash_batched(
            "attention_flash_asym3_tile_batched",
            kernels::ATTENTION_FLASH_ASYM3_TILE_BATCHED_SRC,
            "attention_flash_asym3_tile_batched",
            q, k_cache, v_cache, out, positions, cos_theta, sin_theta,
            n_heads, n_kv_heads, head_dim, max_seq, max_ctx_len, batch_size, partials,
            tree_bias, block_start, block_cols,
        )
    }

    /// Flash attention for asym3 KV (K at 3-bit rotated, V at Q8_0).
    /// Reuses Q8_0 flash reduce (output in normal space — V was un-rotated).
    pub fn attention_flash_asym3(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        seq_len_hint: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const TILE_SIZE: usize = 128;
        let max_tiles = (max_seq + TILE_SIZE - 1) / TILE_SIZE;
        let actual_tiles = (seq_len_hint + TILE_SIZE - 1) / TILE_SIZE;
        let launch_tiles = if self.capture_mode { max_tiles } else { actual_tiles };

        self.ensure_givens4_kernel(
            "attention_flash_asym3_tile",
            kernels::ATTENTION_FLASH_ASYM3_TILE_SRC,
            "attention_flash_asym3_tile",
        )?;
        {
            let func = &self.functions["attention_flash_asym3_tile"];
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let mut q_ptr = q.buf.as_ptr();
            let mut k_ptr = k_cache.buf.as_ptr();
            let mut v_ptr = v_cache.buf.as_ptr();
            let mut p_ptr = partials.buf.as_ptr();
            let mut pos_ptr = pos_buf.as_ptr();
            let mut ct_ptr = cos_theta.buf.as_ptr();
            let mut st_ptr = sin_theta.buf.as_ptr();
            let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32; let mut ms = max_seq as i32;
            let mut sc = scale; let mut ts = TILE_SIZE as i32;
            let mut mt = max_tiles as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut q_ptr as *mut _ as *mut c_void,
                &mut k_ptr as *mut _ as *mut c_void,
                &mut v_ptr as *mut _ as *mut c_void,
                &mut p_ptr as *mut _ as *mut c_void,
                &mut pos_ptr as *mut _ as *mut c_void,
                &mut ct_ptr as *mut _ as *mut c_void,
                &mut st_ptr as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut ms as *mut _ as *mut c_void,
                &mut sc as *mut _ as *mut c_void,
                &mut ts as *mut _ as *mut c_void,
                &mut mt as *mut _ as *mut c_void,
            ];
            unsafe {
                self.hip.launch_kernel(
                    func,
                    [n_heads as u32, launch_tiles as u32, 1],
                    [32, 1, 1],
                    (TILE_SIZE * 4) as u32,
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }

        self.ensure_kernel(
            "attention_flash_q8_0_reduce",
            kernels::ATTENTION_FLASH_Q8_0_REDUCE_SRC,
            "attention_flash_q8_0_reduce",
        )?;
        {
            let func = &self.functions["attention_flash_q8_0_reduce"];
            let mut p_ptr = partials.buf.as_ptr();
            let mut o_ptr = out.buf.as_ptr();
            let mut nh = n_heads as i32;
            let mut hd = head_dim as i32;
            let mut pos_ptr = pos_buf.as_ptr();
            let mut ts = TILE_SIZE as i32;
            let mut mt = max_tiles as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut p_ptr as *mut _ as *mut c_void,
                &mut o_ptr as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut pos_ptr as *mut _ as *mut c_void,
                &mut ts as *mut _ as *mut c_void,
                &mut mt as *mut _ as *mut c_void,
            ];
            unsafe {
                self.hip.launch_kernel(
                    func, [n_heads as u32, 1, 1], [32, 1, 1], 0,
                    self.stream_ref(), &mut params,
                )?;
            }
        }
        Ok(())
    }

    /// Fused K+V write for asym2: K at givens2 (rotated 2-bit), V at Q8_0 (normal space).
    pub fn kv_cache_write_asym2_fused(
        &mut self, k_dst: &GpuTensor, v_dst: &GpuTensor,
        k_src: &GpuTensor, v_src: &GpuTensor, pos_buf: &DeviceBuffer,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "kv_cache_write_asym_k_givens2",
            kernels::KV_CACHE_WRITE_ASYM_K_GIVENS2_SRC,
            "kv_cache_write_asym_k_givens2",
        )?;
        {
            let func = &self.functions["kv_cache_write_asym_k_givens2"];
            let mut kdp = k_dst.buf.as_ptr();
            let mut ksp = k_src.buf.as_ptr();
            let mut pp = pos_buf.as_ptr();
            let mut ctp = cos_theta.buf.as_ptr();
            let mut stp = sin_theta.buf.as_ptr();
            let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut kdp as *mut _ as *mut c_void,
                &mut ksp as *mut _ as *mut c_void,
                &mut pp as *mut _ as *mut c_void,
                &mut ctp as *mut _ as *mut c_void,
                &mut stp as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
            ];
            let shared_mem = ((head_dim + 32) * 4) as u32;
            unsafe {
                self.hip.launch_kernel(
                    func, [n_kv_heads as u32, 1, 1], [32, 1, 1], shared_mem,
                    self.stream_ref(), &mut params,
                )?;
            }
        }
        self.kv_cache_write_q8_0(v_dst, v_src, pos_buf, n_kv_heads, head_dim)
    }

    /// Flash attention for asym4 KV (K at rotated 4-bit, V at Q8_0 normal space).
    /// Reuses the Q8_0 flash reduce since V was un-rotated — no inverse rotation needed.
    pub fn attention_flash_asym4(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        seq_len_hint: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const TILE_SIZE: usize = 128;
        let max_tiles = (max_seq + TILE_SIZE - 1) / TILE_SIZE;
        let actual_tiles = (seq_len_hint + TILE_SIZE - 1) / TILE_SIZE;
        let launch_tiles = if self.capture_mode { max_tiles } else { actual_tiles };

        // Tile kernel
        self.ensure_givens4_kernel(
            "attention_flash_asym4_tile",
            kernels::ATTENTION_FLASH_ASYM4_TILE_SRC,
            "attention_flash_asym4_tile",
        )?;
        {
            let func = &self.functions["attention_flash_asym4_tile"];
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let mut q_ptr = q.buf.as_ptr();
            let mut k_ptr = k_cache.buf.as_ptr();
            let mut v_ptr = v_cache.buf.as_ptr();
            let mut p_ptr = partials.buf.as_ptr();
            let mut pos_ptr = pos_buf.as_ptr();
            let mut ct_ptr = cos_theta.buf.as_ptr();
            let mut st_ptr = sin_theta.buf.as_ptr();
            let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32; let mut ms = max_seq as i32;
            let mut sc = scale; let mut ts = TILE_SIZE as i32;
            let mut mt = max_tiles as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut q_ptr as *mut _ as *mut c_void,
                &mut k_ptr as *mut _ as *mut c_void,
                &mut v_ptr as *mut _ as *mut c_void,
                &mut p_ptr as *mut _ as *mut c_void,
                &mut pos_ptr as *mut _ as *mut c_void,
                &mut ct_ptr as *mut _ as *mut c_void,
                &mut st_ptr as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut ms as *mut _ as *mut c_void,
                &mut sc as *mut _ as *mut c_void,
                &mut ts as *mut _ as *mut c_void,
                &mut mt as *mut _ as *mut c_void,
            ];
            unsafe {
                self.hip.launch_kernel(
                    func,
                    [n_heads as u32, launch_tiles as u32, 1],
                    [32, 1, 1],
                    (TILE_SIZE * 4) as u32, // scores[tile_size]
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }

        // Reuse Q8_0 flash reduce (output already in normal space).
        self.ensure_kernel(
            "attention_flash_q8_0_reduce",
            kernels::ATTENTION_FLASH_Q8_0_REDUCE_SRC,
            "attention_flash_q8_0_reduce",
        )?;
        {
            let func = &self.functions["attention_flash_q8_0_reduce"];
            let mut p_ptr = partials.buf.as_ptr();
            let mut o_ptr = out.buf.as_ptr();
            let mut nh = n_heads as i32;
            let mut hd = head_dim as i32;
            let mut pos_ptr = pos_buf.as_ptr();
            let mut ts = TILE_SIZE as i32;
            let mut mt = max_tiles as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut p_ptr as *mut _ as *mut c_void,
                &mut o_ptr as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut pos_ptr as *mut _ as *mut c_void,
                &mut ts as *mut _ as *mut c_void,
                &mut mt as *mut _ as *mut c_void,
            ];
            unsafe {
                self.hip.launch_kernel(
                    func, [n_heads as u32, 1, 1], [32, 1, 1], 0,
                    self.stream_ref(), &mut params,
                )?;
            }
        }
        Ok(())
    }

    /// Flash attention for asym2 KV (K at rotated 2-bit, V at Q8_0 normal space).
    pub fn attention_flash_asym2(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        seq_len_hint: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const TILE_SIZE: usize = 128;
        let max_tiles = (max_seq + TILE_SIZE - 1) / TILE_SIZE;
        let actual_tiles = (seq_len_hint + TILE_SIZE - 1) / TILE_SIZE;
        let launch_tiles = if self.capture_mode { max_tiles } else { actual_tiles };

        self.ensure_givens4_kernel(
            "attention_flash_asym2_tile",
            kernels::ATTENTION_FLASH_ASYM2_TILE_SRC,
            "attention_flash_asym2_tile",
        )?;
        {
            let func = &self.functions["attention_flash_asym2_tile"];
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let mut q_ptr = q.buf.as_ptr();
            let mut k_ptr = k_cache.buf.as_ptr();
            let mut v_ptr = v_cache.buf.as_ptr();
            let mut p_ptr = partials.buf.as_ptr();
            let mut pos_ptr = pos_buf.as_ptr();
            let mut ct_ptr = cos_theta.buf.as_ptr();
            let mut st_ptr = sin_theta.buf.as_ptr();
            let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32; let mut ms = max_seq as i32;
            let mut sc = scale; let mut ts = TILE_SIZE as i32;
            let mut mt = max_tiles as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut q_ptr as *mut _ as *mut c_void,
                &mut k_ptr as *mut _ as *mut c_void,
                &mut v_ptr as *mut _ as *mut c_void,
                &mut p_ptr as *mut _ as *mut c_void,
                &mut pos_ptr as *mut _ as *mut c_void,
                &mut ct_ptr as *mut _ as *mut c_void,
                &mut st_ptr as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut ms as *mut _ as *mut c_void,
                &mut sc as *mut _ as *mut c_void,
                &mut ts as *mut _ as *mut c_void,
                &mut mt as *mut _ as *mut c_void,
            ];
            unsafe {
                self.hip.launch_kernel(
                    func,
                    [n_heads as u32, launch_tiles as u32, 1],
                    [32, 1, 1],
                    (TILE_SIZE * 4) as u32,
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }

        self.ensure_kernel(
            "attention_flash_q8_0_reduce",
            kernels::ATTENTION_FLASH_Q8_0_REDUCE_SRC,
            "attention_flash_q8_0_reduce",
        )?;
        {
            let func = &self.functions["attention_flash_q8_0_reduce"];
            let mut p_ptr = partials.buf.as_ptr();
            let mut o_ptr = out.buf.as_ptr();
            let mut nh = n_heads as i32;
            let mut hd = head_dim as i32;
            let mut pos_ptr = pos_buf.as_ptr();
            let mut ts = TILE_SIZE as i32;
            let mut mt = max_tiles as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut p_ptr as *mut _ as *mut c_void,
                &mut o_ptr as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut pos_ptr as *mut _ as *mut c_void,
                &mut ts as *mut _ as *mut c_void,
                &mut mt as *mut _ as *mut c_void,
            ];
            unsafe {
                self.hip.launch_kernel(
                    func, [n_heads as u32, 1, 1], [32, 1, 1], 0,
                    self.stream_ref(), &mut params,
                )?;
            }
        }
        Ok(())
    }

    /// Attention with Q8_0 quantized KV cache.
    pub fn attention_q8_0_kv(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer, seq_len_hint: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("attention_q8_0_kv", kernels::ATTENTION_Q8_0_KV_SRC, "attention_q8_0_kv")?;
        let func = &self.functions["attention_q8_0_kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_ptr = q.buf.as_ptr(); let mut k_ptr = k_cache.buf.as_ptr();
        let mut v_ptr = v_cache.buf.as_ptr(); let mut out_ptr = out.buf.as_ptr();
        let mut pos_ptr = pos_buf.as_ptr();
        let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32; let mut ms = max_seq as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void, &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void, &mut out_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void, &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32).next_power_of_two().min(256);
        // Extra shared mem for Q head vector preloaded into shared memory
        let shared_mem = ((seq_len_hint + block_size as usize + head_dim) * 4) as u32;
        let bytes = crate::profile::attention_q8_0_kv_bytes(n_heads, n_kv_heads, head_dim, seq_len_hint);
        let timer = crate::profile::begin_timer(&self.hip, "attention", "attention_q8_0_kv", bytes);
        let result = unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Phase-instrumented variant of attention_q8_0_kv. Identical to the
    /// baseline kernel but additionally writes per-head cycle counts for
    /// each internal phase into `cycle_counts` (layout: [n_heads * 3],
    /// per-head order = phase1(QK^T), phase2(softmax), phase3(V-weighted)).
    ///
    /// Uses __builtin_amdgcn_s_memrealtime() which returns a wall-clock
    /// counter. On gfx1100 the tick rate is approximately 1e8 Hz (10 ns
    /// per tick); confirm empirically by comparing against the kernel's
    /// total elapsed time from event timing.
    ///
    /// Use only for diagnostic profiling — the memrealtime reads serialize
    /// execution and inflate total time slightly.
    pub fn attention_q8_0_kv_timed(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer, seq_len_hint: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
        cycle_counts: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("attention_q8_0_kv_timed", kernels::ATTENTION_Q8_0_KV_TIMED_SRC, "attention_q8_0_kv_timed")?;
        let func = &self.functions["attention_q8_0_kv_timed"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_ptr = q.buf.as_ptr(); let mut k_ptr = k_cache.buf.as_ptr();
        let mut v_ptr = v_cache.buf.as_ptr(); let mut out_ptr = out.buf.as_ptr();
        let mut pos_ptr = pos_buf.as_ptr();
        let mut cc_ptr = cycle_counts.buf.as_ptr();
        let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32; let mut ms = max_seq as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void, &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void, &mut out_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut cc_ptr as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32).next_power_of_two().min(256);
        let shared_mem = ((seq_len_hint + block_size as usize + head_dim) * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// TriAttention importance scoring over a Q8 post-RoPE K cache.
    /// Produces one score per cached position per query head; caller picks
    /// top-B for eviction (see arXiv:2604.04921 §4).
    ///
    /// `centers`: `[n_heads × n_bands × 3]` float32 packed as
    /// `(Re(E[q_f]), Im(E[q_f]), E[||q_f||])`. `scores`: `[n_heads × seq_len]`
    /// float32 output. One block per (pos, head); 32 threads reduce across
    /// the head's frequency bands.
    pub fn triattn_score_q8(
        &mut self,
        k_cache: &GpuTensor,
        centers: &GpuTensor,
        scores: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_rot: usize,
        rope_theta: f32,
        p_q: f32,
        seq_len: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "triattn_score_q8",
            kernels::TRIATTN_SCORE_Q8_SRC,
            "triattn_score_q8",
        )?;
        let func = &self.functions["triattn_score_q8"];
        let mut k_ptr = k_cache.buf.as_ptr();
        let mut c_ptr = centers.buf.as_ptr();
        let mut s_ptr = scores.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut nr = n_rot as i32;
        let mut th = rope_theta;
        let mut pq = p_q;
        let mut sl = seq_len as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut k_ptr as *mut _ as *mut c_void,
            &mut c_ptr as *mut _ as *mut c_void,
            &mut s_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut th as *mut _ as *mut c_void,
            &mut pq as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [seq_len as u32, n_heads as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// TriAttention importance scoring over an asym2 post-RoPE K cache.
    /// Same shape as `triattn_score_asym3` but reads the 2-bit packed
    /// layout (4 indices per byte) and the TURBO_C2_256 codebook.
    pub fn triattn_score_asym2(
        &mut self,
        k_cache: &GpuTensor,
        centers: &GpuTensor,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        scores: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_rot: usize,
        rope_theta: f32,
        p_q: f32,
        seq_len: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "triattn_score_asym2",
            kernels::TRIATTN_SCORE_ASYM2_SRC,
            "triattn_score_asym2",
        )?;
        let func = &self.functions["triattn_score_asym2"];
        let mut k_ptr = k_cache.buf.as_ptr();
        let mut c_ptr = centers.buf.as_ptr();
        let mut ct_ptr = cos_theta.buf.as_ptr();
        let mut st_ptr = sin_theta.buf.as_ptr();
        let mut s_ptr = scores.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut nr = n_rot as i32;
        let mut th = rope_theta;
        let mut pq = p_q;
        let mut sl = seq_len as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut k_ptr as *mut _ as *mut c_void,
            &mut c_ptr as *mut _ as *mut c_void,
            &mut ct_ptr as *mut _ as *mut c_void,
            &mut st_ptr as *mut _ as *mut c_void,
            &mut s_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut th as *mut _ as *mut c_void,
            &mut pq as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [seq_len as u32, n_heads as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// TriAttention importance scoring over an asym4 post-RoPE K cache.
    /// Same shape as `triattn_score_asym3` but reads the 4-bit nibble
    /// layout and the TURBO_C4 codebook.
    pub fn triattn_score_asym4(
        &mut self,
        k_cache: &GpuTensor,
        centers: &GpuTensor,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        scores: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_rot: usize,
        rope_theta: f32,
        p_q: f32,
        seq_len: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "triattn_score_asym4",
            kernels::TRIATTN_SCORE_ASYM4_SRC,
            "triattn_score_asym4",
        )?;
        let func = &self.functions["triattn_score_asym4"];
        let mut k_ptr = k_cache.buf.as_ptr();
        let mut c_ptr = centers.buf.as_ptr();
        let mut ct_ptr = cos_theta.buf.as_ptr();
        let mut st_ptr = sin_theta.buf.as_ptr();
        let mut s_ptr = scores.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut nr = n_rot as i32;
        let mut th = rope_theta;
        let mut pq = p_q;
        let mut sl = seq_len as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut k_ptr as *mut _ as *mut c_void,
            &mut c_ptr as *mut _ as *mut c_void,
            &mut ct_ptr as *mut _ as *mut c_void,
            &mut st_ptr as *mut _ as *mut c_void,
            &mut s_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut th as *mut _ as *mut c_void,
            &mut pq as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [seq_len as u32, n_heads as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// TriAttention importance scoring over an asym3 post-RoPE K cache.
    /// Same contract as `triattn_score_q8` but reads asym3's Givens-rotated
    /// 3-bit layout and applies the inverse Givens rotation on the fly to
    /// recover post-RoPE K per band.
    pub fn triattn_score_asym3(
        &mut self,
        k_cache: &GpuTensor,
        centers: &GpuTensor,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        scores: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_rot: usize,
        rope_theta: f32,
        p_q: f32,
        seq_len: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "triattn_score_asym3",
            kernels::TRIATTN_SCORE_ASYM3_SRC,
            "triattn_score_asym3",
        )?;
        let func = &self.functions["triattn_score_asym3"];
        let mut k_ptr = k_cache.buf.as_ptr();
        let mut c_ptr = centers.buf.as_ptr();
        let mut ct_ptr = cos_theta.buf.as_ptr();
        let mut st_ptr = sin_theta.buf.as_ptr();
        let mut s_ptr = scores.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut nr = n_rot as i32;
        let mut th = rope_theta;
        let mut pq = p_q;
        let mut sl = seq_len as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut k_ptr as *mut _ as *mut c_void,
            &mut c_ptr as *mut _ as *mut c_void,
            &mut ct_ptr as *mut _ as *mut c_void,
            &mut st_ptr as *mut _ as *mut c_void,
            &mut s_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut th as *mut _ as *mut c_void,
            &mut pq as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [seq_len as u32, n_heads as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Compact a KV cache row by row: writes `dst[new_pos] = src[retain[new_pos]]`
    /// for `new_pos` in `[0, budget)`. Works for any byte-addressable cache
    /// layout — pass the layout's bytes-per-position.
    ///
    /// `retain_indices` must live on the device. Caller allocates `dst` with
    /// at least `budget × bytes_per_pos` bytes of capacity.
    pub fn kv_compact_gather(
        &mut self,
        src: &GpuTensor,
        dst: &GpuTensor,
        retain_indices: &GpuTensor,
        bytes_per_pos: usize,
        budget: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "kv_compact_gather",
            kernels::KV_COMPACT_GATHER_SRC,
            "kv_compact_gather",
        )?;
        let func = &self.functions["kv_compact_gather"];
        let mut sp = src.buf.as_ptr();
        let mut dp = dst.buf.as_ptr();
        let mut rp = retain_indices.buf.as_ptr();
        let mut bpp = bytes_per_pos as i32;
        let mut b = budget as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut sp as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut rp as *mut _ as *mut c_void,
            &mut bpp as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
        ];
        // Choose thread count to saturate per-row bandwidth: ~1 thread per
        // 16-byte chunk, capped at 256 threads per block.
        let threads = ((bytes_per_pos / 16) as u32).clamp(32, 256);
        unsafe {
            self.hip.launch_kernel(
                func,
                [budget as u32, 1, 1],
                [threads, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// CASK m-folding merge for Q8_0 KV cache (arXiv:2604.10900).
    ///
    /// Computes `budget` output rows from `budget × m` source rows via
    /// weighted average + per-block requantization. Core (singleton)
    /// slots are handled uniformly by the caller: set `src_indices[s×m]`
    /// to the core source position and `src_weights[s×m] = 1.0`, rest = 0.
    ///
    /// All tensors live on the device. Caller allocates `dst` with at
    /// least `budget × n_kv × n_blocks × 34` bytes.
    pub fn kv_fold_q8(
        &mut self,
        src: &GpuTensor,
        dst: &GpuTensor,
        src_indices: &GpuTensor,     // [budget × m] i32
        src_weights: &GpuTensor,     // [budget × m] f32
        n_kv: usize,
        n_blocks: usize,
        m: usize,
        budget: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "kv_fold_q8",
            kernels::KV_FOLD_Q8_SRC,
            "kv_fold_q8",
        )?;
        let func = &self.functions["kv_fold_q8"];
        let mut sp = src.buf.as_ptr();
        let mut dp = dst.buf.as_ptr();
        let mut ip = src_indices.buf.as_ptr();
        let mut wp = src_weights.buf.as_ptr();
        let mut nkv = n_kv as i32;
        let mut nb = n_blocks as i32;
        let mut mi = m as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut sp as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut ip as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut nb as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [budget as u32, n_kv as u32, n_blocks as u32],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// CASK m-folding merge for asym3 K (givens3).
    ///
    /// Same calling convention as `kv_fold_q8` but takes `head_dim` (whole head)
    /// since asym3 doesn't block-wise split. One thread block per
    /// (slot, kv_head), 32 threads.
    pub fn kv_fold_asym3(
        &mut self,
        src: &GpuTensor, dst: &GpuTensor,
        src_indices: &GpuTensor, src_weights: &GpuTensor,
        n_kv: usize, head_dim: usize, m: usize, budget: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel("kv_fold_asym3", kernels::KV_FOLD_ASYM3_SRC, "kv_fold_asym3")?;
        let func = &self.functions["kv_fold_asym3"];
        let mut sp = src.buf.as_ptr();
        let mut dp = dst.buf.as_ptr();
        let mut ip = src_indices.buf.as_ptr();
        let mut wp = src_weights.buf.as_ptr();
        let mut nkv = n_kv as i32;
        let mut hd = head_dim as i32;
        let mut mi = m as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut sp as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut ip as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func, [budget as u32, n_kv as u32, 1],
                [32, 1, 1], 0, self.stream_ref(), &mut params,
            )
        }
    }

    /// CASK m-folding merge for asym4 K (givens4).
    pub fn kv_fold_asym4(
        &mut self,
        src: &GpuTensor, dst: &GpuTensor,
        src_indices: &GpuTensor, src_weights: &GpuTensor,
        n_kv: usize, head_dim: usize, m: usize, budget: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel("kv_fold_asym4", kernels::KV_FOLD_ASYM4_SRC, "kv_fold_asym4")?;
        let func = &self.functions["kv_fold_asym4"];
        let mut sp = src.buf.as_ptr();
        let mut dp = dst.buf.as_ptr();
        let mut ip = src_indices.buf.as_ptr();
        let mut wp = src_weights.buf.as_ptr();
        let mut nkv = n_kv as i32;
        let mut hd = head_dim as i32;
        let mut mi = m as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut sp as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut ip as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func, [budget as u32, n_kv as u32, 1],
                [32, 1, 1], 0, self.stream_ref(), &mut params,
            )
        }
    }

    /// CASK m-folding merge for asym2 K (givens2).
    pub fn kv_fold_asym2(
        &mut self,
        src: &GpuTensor, dst: &GpuTensor,
        src_indices: &GpuTensor, src_weights: &GpuTensor,
        n_kv: usize, head_dim: usize, m: usize, budget: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel("kv_fold_asym2", kernels::KV_FOLD_ASYM2_SRC, "kv_fold_asym2")?;
        let func = &self.functions["kv_fold_asym2"];
        let mut sp = src.buf.as_ptr();
        let mut dp = dst.buf.as_ptr();
        let mut ip = src_indices.buf.as_ptr();
        let mut wp = src_weights.buf.as_ptr();
        let mut nkv = n_kv as i32;
        let mut hd = head_dim as i32;
        let mut mi = m as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut sp as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut ip as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func, [budget as u32, n_kv as u32, 1],
                [32, 1, 1], 0, self.stream_ref(), &mut params,
            )
        }
    }

    /// Write KV vector to Q8 (int8 symmetric) quantized cache.
    pub fn kv_cache_write_q8(
        &mut self, dst: &GpuTensor, src: &GpuTensor, pos_buf: &DeviceBuffer,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("kv_cache_write_q8", kernels::KV_CACHE_WRITE_Q8_SRC, "kv_cache_write_q8")?;
        let func = &self.functions["kv_cache_write_q8"];
        let mut d = dst.buf.as_ptr();
        let mut s = src.buf.as_ptr();
        let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut d as *mut _ as *mut c_void, &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let block = 64u32.min(head_dim as u32);
        let shared = (block * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [block, 1, 1], shared, self.stream_ref(), &mut params) }
    }

    /// Attention with Q8 quantized KV cache.
    pub fn attention_q8kv(
        &mut self, q: &GpuTensor, k_cache_q8: &GpuTensor, v_cache_q8: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer, seq_len_hint: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("attention_q8kv", kernels::ATTENTION_Q8KV_SRC, "attention_q8kv")?;
        let func = &self.functions["attention_q8kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_ptr = q.buf.as_ptr(); let mut k_ptr = k_cache_q8.buf.as_ptr();
        let mut v_ptr = v_cache_q8.buf.as_ptr(); let mut out_ptr = out.buf.as_ptr();
        let mut pos_ptr = pos_buf.as_ptr();
        let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32; let mut ms = max_seq as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void, &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void, &mut out_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void, &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32).next_power_of_two().min(256);
        let shared_mem = ((seq_len_hint + block_size as usize) * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// Write KV vector to quantized HFQ4 cache.
    pub fn kv_cache_write_q4(
        &mut self, dst: &GpuTensor, src: &GpuTensor, pos_buf: &DeviceBuffer,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("kv_cache_write_q4", kernels::KV_CACHE_WRITE_Q4_SRC, "kv_cache_write_q4")?;
        let func = &self.functions["kv_cache_write_q4"];
        let mut d = dst.buf.as_ptr();
        let mut s = src.buf.as_ptr();
        let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut d as *mut _ as *mut c_void, &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let block = 64u32.min(head_dim as u32);
        let shared = (block * 2 * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [block, 1, 1], shared, self.stream_ref(), &mut params) }
    }

    /// Attention with quantized HFQ4 KV cache — dequantizes K/V on the fly.
    pub fn attention_q4kv(
        &mut self, q: &GpuTensor, k_cache_q4: &GpuTensor, v_cache_q4: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer, seq_len_hint: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("attention_q4kv", kernels::ATTENTION_Q4KV_SRC, "attention_q4kv")?;
        let func = &self.functions["attention_q4kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_ptr = q.buf.as_ptr();
        let mut k_ptr = k_cache_q4.buf.as_ptr();
        let mut v_ptr = v_cache_q4.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut pos_ptr = pos_buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void, &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void, &mut out_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void, &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32).next_power_of_two().min(256);
        let shared_mem = ((seq_len_hint + block_size as usize) * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// GPU-side KV cache write. Copies kv_dim floats from src to dst[pos_buf[0] * kv_dim].
    pub fn kv_cache_write(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        kv_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("kv_cache_write", kernels::KV_CACHE_WRITE_SRC, "kv_cache_write")?;
        let func = &self.functions["kv_cache_write"];

        let mut dst_ptr = dst.buf.as_ptr();
        let mut src_ptr = src.buf.as_ptr();
        let mut pos_ptr = pos_buf.as_ptr();
        let mut kd = kv_dim as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut dst_ptr as *mut _ as *mut c_void,
            &mut src_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void,
            &mut kd as *mut _ as *mut c_void,
        ];

        let block = 256u32;
        let grid = (kv_dim as u32 + block - 1) / block;

        unsafe {
            self.hip.launch_kernel(
                func,
                [grid, 1, 1],
                [block, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// GPU-side top-K + top-P sampling. Returns (token_id, new_rng_state).
    /// Eliminates 600KB logits download per token.
    pub fn sample_top_p(
        &mut self,
        logits: &GpuTensor,
        result_buf: &GpuTensor,
        repeat_buf: &GpuTensor,
        vocab_size: usize,
        temperature: f32,
        top_p: f32,
        rng_state: u32,
        repeat_window: usize,
        repeat_penalty: f32,
    ) -> HipResult<(u32, u32)> {
        self.bind_thread()?;
        self.ensure_kernel("sample_top_p", kernels::SAMPLE_TOP_P_SRC, "sample_top_p")?;
        let func = &self.functions["sample_top_p"];

        let mut logits_ptr = logits.buf.as_ptr();
        let mut result_ptr = result_buf.buf.as_ptr();
        let mut repeat_ptr = repeat_buf.buf.as_ptr();
        let mut vs = vocab_size as i32;
        let mut temp = temperature;
        let mut tp = top_p;
        let mut rng = rng_state;
        let mut rw = repeat_window as i32;
        let mut rp = repeat_penalty;

        let mut params: Vec<*mut std::ffi::c_void> = vec![
            &mut logits_ptr as *mut _ as *mut std::ffi::c_void,
            &mut result_ptr as *mut _ as *mut std::ffi::c_void,
            &mut repeat_ptr as *mut _ as *mut std::ffi::c_void,
            &mut vs as *mut _ as *mut std::ffi::c_void,
            &mut temp as *mut _ as *mut std::ffi::c_void,
            &mut tp as *mut _ as *mut std::ffi::c_void,
            &mut rng as *mut _ as *mut std::ffi::c_void,
            &mut rw as *mut _ as *mut std::ffi::c_void,
            &mut rp as *mut _ as *mut std::ffi::c_void,
        ];

        let block_size = 256u32;
        // topk_val[nthreads*20] + topk_idx[nthreads*20] = 256*20*4 + 256*20*4 = 40960 bytes
        let shared_mem = 256u32 * 20 * 4 * 2;

        unsafe {
            self.hip.launch_kernel(
                func,
                [1, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )?;
        }

        let mut out = [0u8; 8];
        self.hip.memcpy_dtoh(&mut out, &result_buf.buf)?;
        let token_id = u32::from_ne_bytes([out[0], out[1], out[2], out[3]]);
        let new_rng = u32::from_ne_bytes([out[4], out[5], out[6], out[7]]);
        Ok((token_id, new_rng))
    }

    /// Launch sampling kernel only (no readback). For use during graph capture.
    pub fn sample_top_p_launch(
        &mut self,
        logits: &GpuTensor,
        result_buf: &GpuTensor,
        repeat_buf: &GpuTensor,
        vocab_size: usize,
        temperature: f32,
        top_p: f32,
        rng_state: u32,
        repeat_window: usize,
        repeat_penalty: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("sample_top_p", kernels::SAMPLE_TOP_P_SRC, "sample_top_p")?;
        let func = &self.functions["sample_top_p"];

        let mut logits_ptr = logits.buf.as_ptr();
        let mut result_ptr = result_buf.buf.as_ptr();
        let mut repeat_ptr = repeat_buf.buf.as_ptr();
        let mut vs = vocab_size as i32;
        let mut temp = temperature;
        let mut tp = top_p;
        let mut rng = rng_state;
        let mut rw = repeat_window as i32;
        let mut rp = repeat_penalty;

        let mut params: Vec<*mut std::ffi::c_void> = vec![
            &mut logits_ptr as *mut _ as *mut std::ffi::c_void,
            &mut result_ptr as *mut _ as *mut std::ffi::c_void,
            &mut repeat_ptr as *mut _ as *mut std::ffi::c_void,
            &mut vs as *mut _ as *mut std::ffi::c_void,
            &mut temp as *mut _ as *mut std::ffi::c_void,
            &mut tp as *mut _ as *mut std::ffi::c_void,
            &mut rng as *mut _ as *mut std::ffi::c_void,
            &mut rw as *mut _ as *mut std::ffi::c_void,
            &mut rp as *mut _ as *mut std::ffi::c_void,
        ];

        let block_size = 256u32;
        // topk_val[nthreads*20] + topk_idx[nthreads*20] = 256*20*4 + 256*20*4 = 40960 bytes
        let shared_mem = 256u32 * 20 * 4 * 2;

        unsafe {
            self.hip.launch_kernel(
                func,
                [1, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    // ── DeltaNet ops (feature-gated) ─────────────────────────────────────

    /// Partial interleaved RoPE for Qwen3.5 full attention layers.
    #[cfg(feature = "deltanet")]
    /// Single-token RoPE. `pos_buf` is a device buffer holding one i32 position
    /// value (graph-capture-safe: the pointer is stable, content updated before replay).
    pub fn rope_partial_interleaved_f32(
        &mut self, q: &GpuTensor, k: &GpuTensor, pos_buf: &hip_bridge::DeviceBuffer,
        n_heads_q: usize, n_heads_k: usize, head_dim: usize, n_rot: usize, freq_base: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("rope_partial_interleaved", kernels::ROPE_PARTIAL_INTERLEAVED_SRC, "rope_partial_interleaved_f32")?;
        let qp = q.buf.as_ptr(); let kp = k.buf.as_ptr();
        let pp = pos_buf.as_ptr();
        let nhq = n_heads_q as i32; let nhk = n_heads_k as i32;
        let hd = head_dim as i32; let nr = n_rot as i32; let fb = freq_base;
        let n_pairs = (n_rot / 2) as u32;
        let block = 32u32.min(n_pairs);
        let grid = [(n_pairs + block - 1) / block, 1, 1];
        let bytes = crate::profile::rope_bytes(n_heads_q, n_heads_k, head_dim);
        let timer = crate::profile::begin_timer(&self.hip, "rope", "rope_partial_interleaved_f32", bytes);
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void, &kp as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void, &nhq as *const _ as *mut c_void,
            &nhk as *const _ as *mut c_void, &hd as *const _ as *mut c_void,
            &nr as *const _ as *mut c_void, &fb as *const _ as *mut c_void,
        ];
        let result = self.launch_maybe_blob(
            "rope_partial_interleaved_f32", grid, [block, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp); b.push_ptr(kp); b.push_ptr(pp);
                b.push_i32(nhq); b.push_i32(nhk); b.push_i32(hd); b.push_i32(nr);
                b.push_f32(fb);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched partial-interleaved RoPE. Each batch row reads its absolute
    /// position from positions[b] and rotates the first n_rot dims of every
    /// Q and K head. Q/K are [batch_size × n_heads × head_dim] row-major.
    /// Byte-exact with rope_partial_interleaved_f32 at batch_size=1.
    #[cfg(feature = "deltanet")]
    pub fn rope_partial_interleaved_f32_batched(
        &mut self,
        q: &GpuTensor, k: &GpuTensor, positions: &GpuTensor,
        n_heads_q: usize, n_heads_k: usize, head_dim: usize, n_rot: usize,
        freq_base: f32, batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("rope_partial_interleaved_batched",
            kernels::ROPE_PARTIAL_INTERLEAVED_BATCHED_SRC,
            "rope_partial_interleaved_batched_f32")?;
        let mut qp = q.buf.as_ptr();
        let mut kp = k.buf.as_ptr();
        let mut pp = positions.buf.as_ptr();
        let mut nhq = n_heads_q as i32;
        let mut nhk = n_heads_k as i32;
        let mut hd = head_dim as i32;
        let mut nr = n_rot as i32;
        let mut fb = freq_base;
        let mut bs = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut nhq as *mut _ as *mut c_void,
            &mut nhk as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut fb as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        let n_pairs = (n_rot / 2) as u32;
        let block = 32u32.min(n_pairs);
        let grid_x = (n_pairs + block - 1) / block;
        let bytes = crate::profile::rope_bytes(n_heads_q, n_heads_k, head_dim) * batch_size;
        let timer = crate::profile::begin_timer(&self.hip, "rope", "rope_partial_interleaved_batched_f32", bytes);
        let result = self.launch_maybe_blob(
            "rope_partial_interleaved_batched_f32",
            [grid_x, batch_size as u32, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp); b.push_ptr(kp); b.push_ptr(pp);
                b.push_i32(nhq); b.push_i32(nhk); b.push_i32(hd); b.push_i32(nr);
                b.push_f32(fb); b.push_i32(bs);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Sigmoid activation, in-place.
    #[cfg(feature = "deltanet")]
    /// Repeat-interleave Q and K key heads up to value heads count.
    /// Replaces the per-head memcpy loop in DeltaNet for ratio>1 configs:
    /// `dst[(kh*ratio+r)*hd + d] = src[kh*hd + d]`. Does Q and K together
    /// in one launch. For Qwen3.5 9B (24 layers × 64 D2D each), this saves
    /// ~1500 hipMemcpy calls per forward.
    pub fn repeat_interleave_qk_f32(
        &mut self,
        q_src: &GpuTensor,
        k_src: &GpuTensor,
        q_dst: &GpuTensor,
        k_dst: &GpuTensor,
        n_key_heads: usize,
        ratio: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("repeat_interleave_qk", kernels::REPEAT_INTERLEAVE_QK_SRC, "repeat_interleave_qk_f32")?;
        let qsp = q_src.buf.as_ptr();
        let ksp = k_src.buf.as_ptr();
        let qdp = q_dst.buf.as_ptr();
        let kdp = k_dst.buf.as_ptr();
        let nkh = n_key_heads as i32;
        let r = ratio as i32;
        let hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &qsp as *const _ as *mut c_void,
            &ksp as *const _ as *mut c_void,
            &qdp as *const _ as *mut c_void,
            &kdp as *const _ as *mut c_void,
            &nkh as *const _ as *mut c_void,
            &r as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
        ];
        let total = (n_key_heads * ratio * head_dim) as u32;
        let block = 256u32;
        let grid = (total + block - 1) / block;
        let bytes = (n_key_heads * head_dim * 4) * 2 // Q/K reads
                  + (n_key_heads * ratio * head_dim * 4) * 2; // Q/K writes
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", "repeat_interleave_qk_f32", bytes);
        let result = self.launch_maybe_blob(
            "repeat_interleave_qk_f32", [grid, 1, 1], [block, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qsp); b.push_ptr(ksp);
                b.push_ptr(qdp); b.push_ptr(kdp);
                b.push_i32(nkh); b.push_i32(r); b.push_i32(hd);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched repeat-interleave: repeat key heads across N batch elements in one launch.
    /// q_src/k_src: [N × n_key_heads × head_dim], q_dst/k_dst: [N × n_key_heads × ratio × head_dim].
    pub fn repeat_interleave_qk_f32_batched(
        &mut self,
        q_src: &GpuTensor, k_src: &GpuTensor,
        q_dst: &GpuTensor, k_dst: &GpuTensor,
        n_key_heads: usize, ratio: usize, head_dim: usize, n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("repeat_interleave_qk_batched", kernels::REPEAT_INTERLEAVE_QK_BATCHED_SRC, "repeat_interleave_qk_f32_batched")?;
        let mut qsp = q_src.buf.as_ptr();
        let mut ksp = k_src.buf.as_ptr();
        let mut qdp = q_dst.buf.as_ptr();
        let mut kdp = k_dst.buf.as_ptr();
        let mut nkh = n_key_heads as i32;
        let mut r = ratio as i32;
        let mut hd = head_dim as i32;
        let mut nn = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qsp as *mut _ as *mut c_void,
            &mut ksp as *mut _ as *mut c_void,
            &mut qdp as *mut _ as *mut c_void,
            &mut kdp as *mut _ as *mut c_void,
            &mut nkh as *mut _ as *mut c_void,
            &mut r as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nn as *mut _ as *mut c_void,
        ];
        let total = (n_key_heads * ratio * head_dim) as u32;
        let block = 256u32;
        let grid_x = (total + block - 1) / block;
        let bytes = n * ((n_key_heads * head_dim * 4) * 2
                       + (n_key_heads * ratio * head_dim * 4) * 2);
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", "repeat_interleave_qk_f32_batched", bytes);
        let result = self.launch_maybe_blob(
            "repeat_interleave_qk_f32_batched",
            [grid_x, n as u32, 1], [block, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qsp); b.push_ptr(ksp);
                b.push_ptr(qdp); b.push_ptr(kdp);
                b.push_i32(nkh); b.push_i32(r); b.push_i32(hd); b.push_i32(nn);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Deinterleave: split [A_h0(hd), B_h0(hd), A_h1(hd), B_h1(hd), ...] into A and B.
    /// Replaces per-head memcpy loop (n_heads × 2 ioctls → 1 dispatch).
    pub fn deinterleave_f32(&mut self, interleaved: &GpuTensor, out_a: &GpuTensor, out_b: &GpuTensor,
                            n_heads: usize, head_dim: usize) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("deinterleave", kernels::DEINTERLEAVE_SRC, "deinterleave_f32")?;
        let inp = interleaved.buf.as_ptr();
        let ap = out_a.buf.as_ptr();
        let bp = out_b.buf.as_ptr();
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &inp as *const _ as *mut c_void,
            &ap as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
        ];
        let total = (n_heads * head_dim) as u32;
        let block = 256u32;
        let grid = (total + block - 1) / block;
        let bytes = n_heads * head_dim * 4 * 3; // read interleaved, write both outputs
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", "deinterleave_f32", bytes);
        let result = self.launch_maybe_blob(
            "deinterleave_f32",
            [grid, 1, 1], [block, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(inp); b.push_ptr(ap); b.push_ptr(bp);
                b.push_i32(nh); b.push_i32(hd);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched deinterleave: split [N × n_heads × head_dim × 2] interleaved
    /// Q+Gate into separate [N × n_heads × head_dim] Q and Gate tensors.
    /// Replaces the per-token gather/deinterleave/scatter loop in the FA
    /// batched prefill path.
    pub fn deinterleave_f32_batched(&mut self, interleaved: &GpuTensor, out_q: &GpuTensor, out_gate: &GpuTensor,
                                    n_heads: usize, head_dim: usize, n: usize) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("deinterleave_batched", kernels::DEINTERLEAVE_BATCHED_SRC, "deinterleave_f32_batched")?;
        let mut inp = interleaved.buf.as_ptr();
        let mut qp = out_q.buf.as_ptr();
        let mut gp = out_gate.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut nn = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut inp as *mut _ as *mut c_void,
            &mut qp as *mut _ as *mut c_void,
            &mut gp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nn as *mut _ as *mut c_void,
        ];
        let total = (n_heads * head_dim) as u32;
        let block = 256u32;
        let grid_x = (total + block - 1) / block;
        let bytes = n * n_heads * head_dim * 4 * 3;
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", "deinterleave_f32_batched", bytes);
        let result = self.launch_maybe_blob(
            "deinterleave_f32_batched",
            [grid_x, n as u32, 1], [block, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(inp); b.push_ptr(qp); b.push_ptr(gp);
                b.push_i32(nh); b.push_i32(hd); b.push_i32(nn);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    #[cfg(feature = "deltanet")]
    pub fn sigmoid_f32(&mut self, x: &GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("sigmoid", kernels::SIGMOID_SRC, "sigmoid_f32")?;
        let func = &self.functions["sigmoid_f32"];
        let mut xp = x.buf.as_ptr();
        let mut n = x.numel() as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void, &mut n as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = crate::profile::elementwise1_bytes(n as usize);
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", "sigmoid_f32", bytes);
        let result = unsafe { self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, self.stream_ref(), &mut params) };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Softplus activation, in-place.
    #[cfg(feature = "deltanet")]
    pub fn softplus_f32(&mut self, x: &GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("softplus", kernels::SOFTPLUS_SRC, "softplus_f32")?;
        let func = &self.functions["softplus_f32"];
        let mut xp = x.buf.as_ptr();
        let mut n = x.numel() as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void, &mut n as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        unsafe { self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// L2 normalization per head, in-place. One warp per head.
    #[cfg(feature = "deltanet")]
    pub fn l2_norm_f32(&mut self, x: &GpuTensor, n_heads: usize, head_dim: usize, eps: f32) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("l2_norm", kernels::L2_NORM_SRC, "l2_norm_f32")?;
        let func = &self.functions["l2_norm_f32"];
        let mut xp = x.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut ep = eps;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut ep as *mut _ as *mut c_void,
        ];
        let bytes = crate::profile::elementwise1_bytes(n_heads * head_dim);
        let timer = crate::profile::begin_timer(&self.hip, "rmsnorm", "l2_norm_f32", bytes);
        let result = unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Fused `out *= sigmoid(gate)`. Replaces the sigmoid_f32+mul_f32 pair
    /// in the FA attention epilogue (one launch per full-attention layer).
    pub fn sigmoid_mul_f32(
        &mut self,
        out: &GpuTensor,
        gate: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("sigmoid_mul", kernels::SIGMOID_MUL_SRC, "sigmoid_mul_f32")?;
        let mut op = out.buf.as_ptr();
        let mut gp = gate.buf.as_ptr();
        let mut n = out.numel() as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut op as *mut _ as *mut c_void,
            &mut gp as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = crate::profile::elementwise1_bytes(n as usize) * 3;
        let timer = crate::profile::begin_timer(&self.hip, "fused", "sigmoid_mul_f32", bytes);
        let result = self.launch_maybe_blob(
            "sigmoid_mul_f32",
            [grid, 1, 1], [block, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(op); b.push_ptr(gp); b.push_i32(n);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Top-K=1024 extraction over a logits vector. Populates an 8 KB
    /// buffer with [1024 × u32 indices | 1024 × f32 values]. One
    /// device→host copy pulls the whole thing. The host then runs its
    /// existing top-20 min-tracking loop over the 1024 candidates.
    ///
    /// Previous version used 1 wave of 32 threads and measured at ~1.4 ms
    /// because the compiler couldn't pipeline loads through the branchy
    /// min-tracking path. Current version uses 256 threads (8 waves) on
    /// a single workgroup — roughly 10× faster.
    pub fn topk_logits_f32(
        &mut self,
        logits: &GpuTensor,
        topk_buf: &GpuTensor,   // DType::F32 shape [2048] = 8192 bytes
        vocab_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("topk_logits", kernels::TOPK_LOGITS_SRC, "topk_logits_f32")?;
        let func = &self.functions["topk_logits_f32"];
        let mut lp = logits.buf.as_ptr();
        let mut bp = topk_buf.buf.as_ptr();
        let mut vs = vocab_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut lp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut vs as *mut _ as *mut c_void,
        ];
        let bytes = vocab_size * 4 + 8192;
        let timer = crate::profile::begin_timer(&self.hip, "sampling", "topk_logits_f32", bytes);
        let result = unsafe {
            self.hip.launch_kernel(func, [1, 1, 1], [256, 1, 1], 0, self.stream_ref(), &mut params)
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Per-row top-K + log-sum-exp over `[B × vocab]` f32 logits.
    /// Writes `top_idx[B × K]` and `top_logp[B × K]` where `top_logp[r,k] =
    /// logit[r, top_idx[r,k]] - log_z[r]` with `log_z` = row-wise
    /// log-sum-exp. Replaces 20 ms of CPU sort + log_z per DDTree cycle.
    ///
    /// Constraints: K ≤ 8 (kernel-enforced). For larger K, extend MAX_K in
    /// the kernel source and the per-thread arrays.
    pub fn topk_logsumexp_batched_f32(
        &mut self,
        logits: &GpuTensor,   // [B × vocab] f32
        top_idx: &GpuTensor,  // [B × K] i32 (we use f32 tensor for storage — caller reinterprets)
        top_logp: &GpuTensor, // [B × K] f32
        vocab: usize,
        k: usize,
        b: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(k >= 1 && k <= 8, "topk_logsumexp_batched: K={} must be in [1,8]", k);
        self.ensure_kernel(
            "topk_logsumexp_batched",
            kernels::TOPK_LOGSUMEXP_BATCHED_SRC,
            "topk_logsumexp_batched_f32",
        )?;
        let func = &self.functions["topk_logsumexp_batched_f32"];
        let mut lp = logits.buf.as_ptr();
        let mut ti = top_idx.buf.as_ptr();
        let mut tl = top_logp.buf.as_ptr();
        let mut vs = vocab as i32;
        let mut kk = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut lp as *mut _ as *mut c_void,
            &mut ti as *mut _ as *mut c_void,
            &mut tl as *mut _ as *mut c_void,
            &mut vs as *mut _ as *mut c_void,
            &mut kk as *mut _ as *mut c_void,
        ];
        // LDS: (nth_warps=8 floats) + (nth × MAX_K × 2 floats). At nth=256,
        // MAX_K=8: 32 + 4096 = 4128 floats = 16,512 bytes. Fits in 64 KB LDS.
        const MAX_K: u32 = 8;
        let nth: u32 = 256;
        let lds = ((32 + nth * MAX_K * 2) * 4) as u32;
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [b as u32, 1, 1],
                [nth, 1, 1],
                lds,
                self.stream_ref(),
                &mut params,
            )
        };
        result
    }

    /// Fused sigmoid(dn_beta) + alpha_gate(dn_alpha). Both ops are element-wise
    /// scalar transforms applied to independent buffers of size n_v_heads in the
    /// DeltaNet preamble. Saves one launch per linear-attention layer.
    #[cfg(feature = "deltanet")]
    pub fn fused_sigmoid_alpha_gate_f32(
        &mut self,
        beta: &GpuTensor,
        alpha: &GpuTensor,
        dt_bias: &GpuTensor,
        a_log: &GpuTensor,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "fused_sigmoid_alpha_gate",
            kernels::FUSED_SIGMOID_ALPHA_GATE_SRC,
            "fused_sigmoid_alpha_gate_f32",
        )?;
        let bp = beta.buf.as_ptr();
        let ap = alpha.buf.as_ptr();
        let dp = dt_bias.buf.as_ptr();
        let lp = a_log.buf.as_ptr();
        let nn = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &bp as *const _ as *mut c_void,
            &ap as *const _ as *mut c_void,
            &dp as *const _ as *mut c_void,
            &lp as *const _ as *mut c_void,
            &nn as *const _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = n * 4 * 4;
        let timer = crate::profile::begin_timer(&self.hip, "fused", "fused_sigmoid_alpha_gate_f32", bytes);
        let result = self.launch_maybe_blob(
            "fused_sigmoid_alpha_gate_f32", [grid, 1, 1], [block, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(bp); b.push_ptr(ap); b.push_ptr(dp); b.push_ptr(lp);
                b.push_i32(nn);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched `fused_sigmoid_alpha_gate_f32`. Grid.y is the batch dim.
    #[cfg(feature = "deltanet")]
    pub fn fused_sigmoid_alpha_gate_f32_batched(
        &mut self,
        beta: &GpuTensor,
        alpha: &GpuTensor,
        dt_bias: &GpuTensor,
        a_log: &GpuTensor,
        n: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "fused_sigmoid_alpha_gate",
            kernels::FUSED_SIGMOID_ALPHA_GATE_SRC,
            "fused_sigmoid_alpha_gate_f32",
        )?;
        let mut bp = beta.buf.as_ptr();
        let mut ap = alpha.buf.as_ptr();
        let mut dp = dt_bias.buf.as_ptr();
        let mut lp = a_log.buf.as_ptr();
        let mut nn = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut bp as *mut _ as *mut c_void,
            &mut ap as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut lp as *mut _ as *mut c_void,
            &mut nn as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = n * 4 * 4 * batch_size;
        let timer = crate::profile::begin_timer(&self.hip, "fused", "fused_sigmoid_alpha_gate_f32_batched", bytes);
        let result = self.launch_maybe_blob(
            "fused_sigmoid_alpha_gate_f32",
            [grid, batch_size as u32, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(bp); b.push_ptr(ap); b.push_ptr(dp); b.push_ptr(lp);
                b.push_i32(nn);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Fused L2-norm(Q) + L2-norm(K) + scale(Q). Replaces three back-to-back
    /// launches in DeltaNet's attention path with one — ~2 launches saved per
    /// linear-attention layer, so on Qwen3.5 (18-32 LA layers) we shave ~36-64
    /// launches per forward.
    #[cfg(feature = "deltanet")]
    pub fn fused_qk_l2_norm_scale_f32(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        n_heads: usize,
        head_dim: usize,
        q_scale: f32,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "fused_qk_l2_norm_scale",
            kernels::FUSED_QK_L2_NORM_SCALE_SRC,
            "fused_qk_l2_norm_scale_f32",
        )?;
        let qp = q.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let qs = q_scale;
        let ep = eps;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &qs as *const _ as *mut c_void,
            &ep as *const _ as *mut c_void,
        ];
        // Covers both Q and K reads/writes.
        let bytes = crate::profile::elementwise1_bytes(n_heads * head_dim) * 2;
        let timer = crate::profile::begin_timer(&self.hip, "fused", "fused_qk_l2_norm_scale_f32", bytes);
        let result = self.launch_maybe_blob(
            "fused_qk_l2_norm_scale_f32", [n_heads as u32, 1, 1], [32, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp); b.push_ptr(kp);
                b.push_i32(nh); b.push_i32(hd);
                b.push_f32(qs); b.push_f32(ep);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched `fused_qk_l2_norm_scale_f32`. Grid.y is the batch dim.
    #[cfg(feature = "deltanet")]
    pub fn fused_qk_l2_norm_scale_f32_batched(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        n_heads: usize,
        head_dim: usize,
        q_scale: f32,
        eps: f32,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "fused_qk_l2_norm_scale",
            kernels::FUSED_QK_L2_NORM_SCALE_SRC,
            "fused_qk_l2_norm_scale_f32",
        )?;
        let mut qp = q.buf.as_ptr();
        let mut kp = k.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut qs = q_scale;
        let mut ep = eps;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut qs as *mut _ as *mut c_void,
            &mut ep as *mut _ as *mut c_void,
        ];
        let bytes = crate::profile::elementwise1_bytes(n_heads * head_dim) * 2 * batch_size;
        let timer = crate::profile::begin_timer(&self.hip, "fused", "fused_qk_l2_norm_scale_f32_batched", bytes);
        let result = self.launch_maybe_blob(
            "fused_qk_l2_norm_scale_f32",
            [n_heads as u32, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp); b.push_ptr(kp);
                b.push_i32(nh); b.push_i32(hd);
                b.push_f32(qs); b.push_f32(ep);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// 1D causal conv (kernel_size=4) for decode. Updates ring buffer state.
    #[cfg(feature = "deltanet")]
    pub fn conv1d_decode_f32(
        &mut self, output: &GpuTensor, input: &GpuTensor, weight: &GpuTensor,
        state: &GpuTensor, n_channels: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("conv1d_decode", kernels::CONV1D_DECODE_SRC, "conv1d_decode_f32")?;
        let func = &self.functions["conv1d_decode_f32"];
        let mut op = output.buf.as_ptr();
        let mut ip = input.buf.as_ptr();
        let mut wp = weight.buf.as_ptr();
        let mut sp = state.buf.as_ptr();
        let mut nc = n_channels as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut op as *mut _ as *mut c_void, &mut ip as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void, &mut sp as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n_channels as u32) + block - 1) / block;
        unsafe { self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// Gated output norm: rmsnorm(x) * silu(z). Fused kernel.
    #[cfg(feature = "deltanet")]
    pub fn gated_norm_f32(
        &mut self, x: &GpuTensor, z: &GpuTensor, weight: &GpuTensor,
        out: &GpuTensor, n_heads: usize, head_dim: usize, eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gated_norm", kernels::GATED_NORM_SRC, "gated_norm_f32")?;
        let xp = x.buf.as_ptr();
        let zp = z.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let op = out.buf.as_ptr();
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let ep = eps;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void, &zp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void, &op as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void, &hd as *const _ as *mut c_void,
            &ep as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::gated_norm_bytes(n_heads * head_dim);
        let timer = crate::profile::begin_timer(&self.hip, "rmsnorm", "gated_norm_f32", bytes);
        let result = self.launch_maybe_blob(
            "gated_norm_f32", [n_heads as u32, 1, 1], [32, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp); b.push_ptr(zp);
                b.push_ptr(wp); b.push_ptr(op);
                b.push_i32(nh); b.push_i32(hd); b.push_f32(ep);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched `gated_norm_f32`. Grid.y is the batch dim.
    #[cfg(feature = "deltanet")]
    pub fn gated_norm_f32_batched(
        &mut self,
        x: &GpuTensor, z: &GpuTensor, weight: &GpuTensor, out: &GpuTensor,
        n_heads: usize, head_dim: usize, eps: f32,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gated_norm", kernels::GATED_NORM_SRC, "gated_norm_f32")?;
        let mut xp = x.buf.as_ptr();
        let mut zp = z.buf.as_ptr();
        let mut wp = weight.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut ep = eps;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void, &mut zp as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut ep as *mut _ as *mut c_void,
        ];
        let bytes = crate::profile::gated_norm_bytes(n_heads * head_dim) * batch_size;
        let timer = crate::profile::begin_timer(&self.hip, "rmsnorm", "gated_norm_f32_batched", bytes);
        let result = self.launch_maybe_blob(
            "gated_norm_f32",
            [n_heads as u32, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp); b.push_ptr(zp);
                b.push_ptr(wp); b.push_ptr(op);
                b.push_i32(nh); b.push_i32(hd); b.push_f32(ep);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Gated Delta Net recurrence. S matrix in LDS. Processes all tokens sequentially.
    #[cfg(feature = "deltanet")]
    pub fn gated_delta_net_f32(
        &mut self, q: &GpuTensor, k: &GpuTensor, v: &GpuTensor,
        gate: &GpuTensor, beta: &GpuTensor,
        state: &GpuTensor, output: &GpuTensor,
        n_tokens: usize, n_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gated_delta_net", kernels::GATED_DELTA_NET_SRC, "gated_delta_net_f32")?;
        let func = &self.functions["gated_delta_net_f32"];
        let mut qp = q.buf.as_ptr();
        let mut kp = k.buf.as_ptr();
        let mut vp = v.buf.as_ptr();
        let mut gp = gate.buf.as_ptr();
        let mut bp = beta.buf.as_ptr();
        let mut sp = state.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut nt = n_tokens as i32;
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void, &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void, &mut gp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void, &mut sp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void, &mut nt as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
        ];
        // 32 threads, tiled S in LDS (4KB per tile). Grid: [n_heads, 128/8=16].
        let n_tiles = (128 / 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, n_tiles, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// GDN recurrence with Q8-quantized S state — tiled LDS + warp-shuffle.
    #[cfg(feature = "deltanet")]
    pub fn gated_delta_net_q8(
        &mut self, q: &GpuTensor, k: &GpuTensor, v: &GpuTensor,
        gate: &GpuTensor, beta: &GpuTensor,
        s_q8: &GpuTensor, s_scales: &GpuTensor, output: &GpuTensor,
        n_tokens: usize, n_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gated_delta_net_q8", kernels::GATED_DELTA_NET_Q8_SRC, "gated_delta_net_q8")?;
        let qp = q.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let vp = v.buf.as_ptr();
        let gp = gate.buf.as_ptr();
        let bp = beta.buf.as_ptr();
        let sp = s_q8.buf.as_ptr();
        let scp = s_scales.buf.as_ptr();
        let op = output.buf.as_ptr();
        let nt = n_tokens as i32;
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void, &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void, &gp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void, &sp as *const _ as *mut c_void,
            &scp as *const _ as *mut c_void, &op as *const _ as *mut c_void,
            &nt as *const _ as *mut c_void, &nh as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
        ];
        let n_tiles = (128 / 4) as u32;
        let bytes = crate::profile::gated_delta_net_q8_bytes(n_tokens, n_heads, head_dim);
        let timer = crate::profile::begin_timer(&self.hip, "deltanet", "gated_delta_net_q8", bytes);
        let result = self.launch_maybe_blob(
            "gated_delta_net_q8", [n_heads as u32, n_tiles, 1], [32, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp); b.push_ptr(kp); b.push_ptr(vp);
                b.push_ptr(gp); b.push_ptr(bp); b.push_ptr(sp);
                b.push_ptr(scp); b.push_ptr(op);
                b.push_i32(nt); b.push_i32(nh); b.push_i32(hd);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched sequential `gated_delta_net_q8` for prefill.
    ///
    /// Launches the single-token kernel N times with offset pointers into
    /// [N × stride]-laid-out Q/K/V/gate/beta/output buffers. This preserves
    /// bit-exact semantics with N × `gated_delta_net_q8(n_tokens=1)` calls
    /// (i.e., dequant→update→requant per token, with stochastic rounding
    /// applied each step) — critical for byte-exact quality gate compliance.
    ///
    /// Why not just call the kernel once with `n_tokens=N`? The existing
    /// kernel dequants S_q8 once at start, runs N updates in FP32 inside
    /// LDS, and requants once at end. That collapses N rounding steps into
    /// one, producing numerically different output from sequential calls —
    /// diverges from the decode-path baseline.
    ///
    /// Q/K/V/output are [N × n_heads × head_dim] row-major.
    /// gate/beta are [N × n_heads] row-major.
    /// S_q8 / s_scales are the shared state (advanced N steps).
    #[cfg(feature = "deltanet")]
    pub fn gated_delta_net_q8_batch_seq(
        &mut self,
        q_batch: &GpuTensor,
        k_batch: &GpuTensor,
        v_batch: &GpuTensor,
        gate_batch: &GpuTensor,
        beta_batch: &GpuTensor,
        s_q8: &GpuTensor,
        s_scales: &GpuTensor,
        output_batch: &GpuTensor,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gated_delta_net_q8", kernels::GATED_DELTA_NET_Q8_SRC, "gated_delta_net_q8")?;

        let n_tiles = (128 / 4) as u32;

        let mut qp = q_batch.buf.as_ptr();
        let mut kp = k_batch.buf.as_ptr();
        let mut vp = v_batch.buf.as_ptr();
        let mut gp = gate_batch.buf.as_ptr();
        let mut bp = beta_batch.buf.as_ptr();
        let mut sp = s_q8.buf.as_ptr();
        let mut scp = s_scales.buf.as_ptr();
        let mut op = output_batch.buf.as_ptr();
        let mut nt = n_tokens as i32;
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut gp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut scp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut nt as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];

        let bytes = crate::profile::gated_delta_net_q8_bytes(n_tokens, n_heads, head_dim);
        let timer = crate::profile::begin_timer(&self.hip, "deltanet", "gated_delta_net_q8_batch_seq", bytes);
        // Single launch — the kernel loops over n_tokens internally,
        // keeping state in F32 LDS across all tokens. Q8 quantization
        // happens once at the end instead of per-token, reducing noise
        // accumulation. Not byte-exact with N×1 decode calls but
        // strictly higher quality.
        let result = self.launch_maybe_blob(
            "gated_delta_net_q8",
            [n_heads as u32, n_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp); b.push_ptr(kp); b.push_ptr(vp);
                b.push_ptr(gp); b.push_ptr(bp);
                b.push_ptr(sp); b.push_ptr(scp); b.push_ptr(op);
                b.push_i32(nt); b.push_i32(nh); b.push_i32(hd);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Tree-aware variant of `gated_delta_net_q8_batch_seq`. Per-token
    /// S-tile persist-write so sibling tokens read the parent's post-update
    /// state via `s_tape_q8[parent_indices[t]]`. `parent_indices[t] < 0`
    /// means "read pre-block initial state from `s_q8_init`".
    ///
    /// Does NOT advance persistent `s_q8_init` / `s_scales_init` (those
    /// are the pre-block snapshot, read-only). Caller runs linear replay
    /// on the accepted spine post-acceptance to commit the trajectory.
    ///
    /// Tape layout (caller responsibility):
    /// - `s_tape_q8`:     `[n_tokens × n_heads × HD × HD]` i8 (scratch)
    /// - `s_tape_scales`: `[n_tokens × n_heads × HD]` f32 (scratch)
    /// - `parent_indices`: `[n_tokens]` i32 (host materialized by
    ///   `ddtree::linearize_tree`; spine topology is [-1, 0, 1, 2, ...])
    #[cfg(feature = "deltanet")]
    pub fn gated_delta_net_q8_tree_batch_seq(
        &mut self,
        q_batch: &GpuTensor,
        k_batch: &GpuTensor,
        v_batch: &GpuTensor,
        gate_batch: &GpuTensor,
        beta_batch: &GpuTensor,
        s_q8_init: &GpuTensor,
        s_scales_init: &GpuTensor,
        s_tape_q8: &GpuTensor,
        s_tape_scales: &GpuTensor,
        parent_indices: &GpuTensor,
        output_batch: &GpuTensor,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gated_delta_net_q8_tree",
            kernels::GATED_DELTA_NET_Q8_TREE_SRC,
            "gated_delta_net_q8_tree",
        )?;

        let n_tiles = (128 / 4) as u32;

        let mut qp = q_batch.buf.as_ptr();
        let mut kp = k_batch.buf.as_ptr();
        let mut vp = v_batch.buf.as_ptr();
        let mut gp = gate_batch.buf.as_ptr();
        let mut bp = beta_batch.buf.as_ptr();
        let mut sip = s_q8_init.buf.as_ptr();
        let mut scip = s_scales_init.buf.as_ptr();
        let mut stp = s_tape_q8.buf.as_ptr();
        let mut stsp = s_tape_scales.buf.as_ptr();
        let mut pp = parent_indices.buf.as_ptr();
        let mut op = output_batch.buf.as_ptr();
        let mut nt = n_tokens as i32;
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp   as *mut _ as *mut c_void,
            &mut kp   as *mut _ as *mut c_void,
            &mut vp   as *mut _ as *mut c_void,
            &mut gp   as *mut _ as *mut c_void,
            &mut bp   as *mut _ as *mut c_void,
            &mut sip  as *mut _ as *mut c_void,
            &mut scip as *mut _ as *mut c_void,
            &mut stp  as *mut _ as *mut c_void,
            &mut stsp as *mut _ as *mut c_void,
            &mut pp   as *mut _ as *mut c_void,
            &mut op   as *mut _ as *mut c_void,
            &mut nt   as *mut _ as *mut c_void,
            &mut nh   as *mut _ as *mut c_void,
            &mut hd   as *mut _ as *mut c_void,
        ];

        let bytes = crate::profile::gated_delta_net_q8_bytes(n_tokens, n_heads, head_dim);
        let timer = crate::profile::begin_timer(
            &self.hip, "deltanet", "gated_delta_net_q8_tree_batch_seq", bytes,
        );
        let result = self.launch_maybe_blob(
            "gated_delta_net_q8_tree",
            [n_heads as u32, n_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp); b.push_ptr(kp); b.push_ptr(vp);
                b.push_ptr(gp); b.push_ptr(bp);
                b.push_ptr(sip); b.push_ptr(scip);
                b.push_ptr(stp); b.push_ptr(stsp);
                b.push_ptr(pp); b.push_ptr(op);
                b.push_i32(nt); b.push_i32(nh); b.push_i32(hd);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// GDN recurrence with Q4-quantized S state.
    #[cfg(feature = "deltanet")]
    pub fn gated_delta_net_q4(
        &mut self, q: &GpuTensor, k: &GpuTensor, v: &GpuTensor,
        gate: &GpuTensor, beta: &GpuTensor,
        s_q4: &GpuTensor, s_scales: &GpuTensor, output: &GpuTensor,
        n_tokens: usize, n_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gated_delta_net_q4", kernels::GATED_DELTA_NET_Q4_SRC, "gated_delta_net_q4")?;
        let func = &self.functions["gated_delta_net_q4"];
        let mut qp = q.buf.as_ptr();
        let mut kp = k.buf.as_ptr();
        let mut vp = v.buf.as_ptr();
        let mut gp = gate.buf.as_ptr();
        let mut bp = beta.buf.as_ptr();
        let mut sp = s_q4.buf.as_ptr();
        let mut scp = s_scales.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut nt = n_tokens as i32;
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void, &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void, &mut gp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void, &mut sp as *mut _ as *mut c_void,
            &mut scp as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void,
            &mut nt as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [128, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// Alpha gate compute: alpha[i] = softplus(alpha[i] + dt_bias[i]) * (-exp(a_log[i])).
    /// Replaces 85µs CPU roundtrip with ~3µs GPU kernel.
    #[cfg(feature = "deltanet")]
    pub fn alpha_gate_f32(
        &mut self, alpha: &GpuTensor, dt_bias: &GpuTensor, a_log: &GpuTensor, n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("alpha_gate", kernels::ALPHA_GATE_SRC, "alpha_gate_f32")?;
        let func = &self.functions["alpha_gate_f32"];
        let mut ap = alpha.buf.as_ptr();
        let mut dp = dt_bias.buf.as_ptr();
        let mut lp = a_log.buf.as_ptr();
        let mut nv = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void, &mut dp as *mut _ as *mut c_void,
            &mut lp as *mut _ as *mut c_void, &mut nv as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = n * 4 * 4;
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", "alpha_gate_f32", bytes);
        let result = unsafe { self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, self.stream_ref(), &mut params) };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Scale vector by constant: x[i] *= scale. Replaces 48µs CPU roundtrip.
    #[cfg(feature = "deltanet")]
    pub fn scale_f32(&mut self, x: &GpuTensor, scale: f32) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("scale_f32", kernels::SCALE_F32_SRC, "scale_f32")?;
        let func = &self.functions["scale_f32"];
        let n = x.numel();
        let mut xp = x.buf.as_ptr();
        let mut nv = n as i32;
        let mut sv = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void, &mut nv as *mut _ as *mut c_void,
            &mut sv as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = crate::profile::elementwise1_bytes(n);
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", "scale_f32", bytes);
        let result = unsafe { self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, self.stream_ref(), &mut params) };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Fused `y[i] += c * x[i]` with a CPU-supplied scalar. Merges the
    /// (scale_f32 + add_inplace_f32) pair used by the MoE routed-expert
    /// epilogue — one kernel launch instead of two.
    pub fn scaled_add_inplace_cpu_scalar_f32(
        &mut self, y: &GpuTensor, x: &GpuTensor, c: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "scaled_add_inplace",
            kernels::SCALED_ADD_INPLACE_SRC,
            "scaled_add_inplace_cpu_scalar_f32",
        )?;
        let func = &self.functions["scaled_add_inplace_cpu_scalar_f32"];
        let n = y.numel();
        let mut yp = y.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut cv = c;
        let mut nv = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut yp as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut cv as *mut _ as *mut c_void,
            &mut nv as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = crate::profile::elementwise1_bytes(n);
        let timer = crate::profile::begin_timer(
            &self.hip, "elementwise", "scaled_add_inplace_cpu_scalar_f32", bytes,
        );
        let result = unsafe {
            self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, self.stream_ref(), &mut params)
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Fused `y[i] += c_buf[0] * x[i]` where `c_buf` is a 1-element GPU
    /// tensor. Used by the MoE shared-expert epilogue: the scalar gate
    /// is `sigmoid(W_shared_gate · x)` computed entirely on-device, so
    /// passing the result by device pointer saves the D2H sync that a
    /// plain `scale_f32(c_host)` would require.
    pub fn scaled_add_inplace_gpu_scalar_f32(
        &mut self, y: &GpuTensor, x: &GpuTensor, c_buf: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "scaled_add_inplace",
            kernels::SCALED_ADD_INPLACE_SRC,
            "scaled_add_inplace_gpu_scalar_f32",
        )?;
        let func = &self.functions["scaled_add_inplace_gpu_scalar_f32"];
        let n = y.numel();
        let mut yp = y.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut cp = c_buf.buf.as_ptr();
        let mut nv = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut yp as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut cp as *mut _ as *mut c_void,
            &mut nv as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = crate::profile::elementwise1_bytes(n);
        let timer = crate::profile::begin_timer(
            &self.hip, "elementwise", "scaled_add_inplace_gpu_scalar_f32", bytes,
        );
        let result = unsafe {
            self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, self.stream_ref(), &mut params)
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Fused conv1d (kernel_size=4) + SiLU decode.
    #[cfg(feature = "deltanet")]
    pub fn conv1d_silu_f32(
        &mut self, output: &GpuTensor, input: &GpuTensor, weight: &GpuTensor,
        state: &GpuTensor, n_channels: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("conv1d_silu", kernels::CONV1D_SILU_SRC, "conv1d_silu_f32")?;
        let func = &self.functions["conv1d_silu_f32"];
        let mut op = output.buf.as_ptr();
        let mut ip = input.buf.as_ptr();
        let mut wp = weight.buf.as_ptr();
        let mut sp = state.buf.as_ptr();
        let mut nc = n_channels as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut op as *mut _ as *mut c_void, &mut ip as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void, &mut sp as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n_channels as u32) + block - 1) / block;
        let bytes = crate::profile::conv1d_silu_bytes(n_channels);
        let timer = crate::profile::begin_timer(&self.hip, "deltanet", "conv1d_silu_f32", bytes);
        let result = unsafe { self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, self.stream_ref(), &mut params) };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Fused conv1d+SiLU that writes directly to Q/K/V buffers, replacing
    /// the conv1d_silu_f32 + three DtoD split copies in the DeltaNet path.
    /// Channel layout: [Q (k_dim) | K (k_dim) | V (v_dim)] — matches the
    /// wqkv projection output layout.
    #[cfg(feature = "deltanet")]
    pub fn conv1d_silu_split_f32(
        &mut self,
        q_out: &GpuTensor,
        k_out: &GpuTensor,
        v_out: &GpuTensor,
        input: &GpuTensor,
        weight: &GpuTensor,
        state: &GpuTensor,
        k_dim: usize,
        v_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.conv1d_silu_split_f32_n(q_out, k_out, v_out, input, weight, state, k_dim, v_dim, 1)
    }

    /// Batched conv1d + silu + Q/K/V split. Processes `n_tokens` tokens in
    /// order through the conv, advancing the ring-buffer state N times
    /// (identical state trajectory to calling the single-token variant N
    /// times). `input` / `q_out` / `k_out` / `v_out` are all [N × stride]
    /// row-major.
    #[cfg(feature = "deltanet")]
    pub fn conv1d_silu_split_f32_n(
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
            "conv1d_silu_split",
            kernels::CONV1D_SILU_SPLIT_SRC,
            "conv1d_silu_split_f32",
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
        let timer = crate::profile::begin_timer(&self.hip, "deltanet", "conv1d_silu_split_f32_n", bytes);
        let result = self.launch_maybe_blob(
            "conv1d_silu_split_f32", [grid, 1, 1], [block, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp); b.push_ptr(kp); b.push_ptr(vp);
                b.push_ptr(ip); b.push_ptr(wp); b.push_ptr(sp);
                b.push_i32(kd); b.push_i32(vd); b.push_i32(nt);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Tree-aware variant of `conv1d_silu_split_f32_n`. `parent_indices[t]`
    /// is the linear slot index of token t's parent within the block, or
    /// a negative sentinel for pre-block ancestors: -1 selects conv_state[0]
    /// (most recent pre-block), -2 → state[1], -3 → state[2].
    ///
    /// Does NOT update conv_state — caller runs linear conv1d on the
    /// accepted spine post-acceptance to advance state.
    ///
    /// Port of SGLang's `HAS_EAGLE_TREE_CUSTOM_ATTN_MASK` branch in
    /// `causal_conv1d_update`. parent_indices supersedes retrieve_next_token
    /// / retrieve_next_sibling / retrieve_parent_token (the tree is already
    /// materialized host-side by `ddtree::linearize_tree`).
    #[cfg(feature = "deltanet")]
    pub fn conv1d_silu_split_tree_f32_n(
        &mut self,
        q_out: &GpuTensor,
        k_out: &GpuTensor,
        v_out: &GpuTensor,
        input: &GpuTensor,
        weight: &GpuTensor,
        state: &GpuTensor,
        parent_indices: &GpuTensor,
        k_dim: usize,
        v_dim: usize,
        n_tokens: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "conv1d_silu_split_tree",
            kernels::CONV1D_SILU_SPLIT_TREE_SRC,
            "conv1d_silu_split_tree_f32",
        )?;
        let qp = q_out.buf.as_ptr();
        let kp = k_out.buf.as_ptr();
        let vp = v_out.buf.as_ptr();
        let ip = input.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let sp = state.buf.as_ptr();
        let pp = parent_indices.buf.as_ptr();
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
            &pp as *const _ as *mut c_void,
            &kd as *const _ as *mut c_void,
            &vd as *const _ as *mut c_void,
            &nt as *const _ as *mut c_void,
        ];
        let n_channels = 2 * k_dim + v_dim;
        let block = 256u32;
        let grid = ((n_channels as u32) + block - 1) / block;
        let bytes = crate::profile::conv1d_silu_bytes(n_channels) * n_tokens;
        let timer = crate::profile::begin_timer(&self.hip, "deltanet", "conv1d_silu_split_tree_f32_n", bytes);
        let result = self.launch_maybe_blob(
            "conv1d_silu_split_tree_f32", [grid, 1, 1], [block, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp); b.push_ptr(kp); b.push_ptr(vp);
                b.push_ptr(ip); b.push_ptr(wp); b.push_ptr(sp);
                b.push_ptr(pp);
                b.push_i32(kd); b.push_i32(vd); b.push_i32(nt);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Compute cross-entropy loss for a single token on GPU.
    /// Returns -log(softmax(logits)[target]). Downloads 4 bytes instead of 600KB.
    pub fn cross_entropy_loss(
        &mut self, logits: &GpuTensor, target_buf: &DeviceBuffer, loss_buf: &GpuTensor,
        vocab_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("cross_entropy_loss", kernels::CROSS_ENTROPY_LOSS_SRC, "cross_entropy_loss")?;
        let func = &self.functions["cross_entropy_loss"];
        let mut lp = logits.buf.as_ptr();
        let mut tp = target_buf.as_ptr();
        let mut op = loss_buf.buf.as_ptr();
        let mut vs = vocab_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut lp as *mut _ as *mut c_void, &mut tp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void, &mut vs as *mut _ as *mut c_void,
        ];
        let block_size = 256u32;
        let shared_mem = (block_size * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [1, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    // ═══ Vision encoder dispatch (GEMM, LayerNorm, GELU, bias-add) ═══

    /// Batched GEMV (GEMM) for F16 weights: Y[M,N] = W_f16[M,K] @ X_f32[N,K]^T
    pub fn gemm_f16(
        &mut self, w: &GpuTensor, x: &GpuTensor, y: &GpuTensor,
        m: usize, k: usize, n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemm_f16", kernels::GEMM_F16_SRC, "gemm_f16")?;
        let func = &self.functions["gemm_f16"];
        let mut wp = w.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut wp as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [m as u32, n as u32, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// WMMA-accelerated batched GEMM for F16 weights × F32 activations (gfx1100+).
    /// Y[M,N] = W_f16[M,K] @ X_f32[N,K]^T.  Tiled 16×16 WMMA matrix multiply.
    /// Grid=[ceil(M/16), ceil(N/16)], Block=[32].  Replaces naive gemm_f16 for vision encoder.
    pub fn gemm_f16_wmma(
        &mut self, w: &GpuTensor, x: &GpuTensor, y: &GpuTensor,
        m: usize, k: usize, n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemm_f16_wmma", kernels::GEMM_F16_WMMA_SRC, "gemm_f16_wmma")?;
        let func = &self.functions["gemm_f16_wmma"];
        let mut wp = w.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut wp as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        let grid_m = ((m + 15) / 16) as u32;
        let grid_n = ((n + 15) / 16) as u32;
        unsafe { self.hip.launch_kernel(func, [grid_m, grid_n, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// Tiled F16 GEMM — 4-way ILP unrolled, no shared memory (high occupancy).
    /// Grid=[M, N], Block=[32], LDS=0.
    pub fn gemm_f16_tiled(
        &mut self, w: &GpuTensor, x: &GpuTensor, y: &GpuTensor,
        m: usize, k: usize, n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemm_f16_tiled", kernels::GEMM_F16_TILED_SRC, "gemm_f16_tiled")?;
        let func = &self.functions["gemm_f16_tiled"];
        let mut wp = w.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut wp as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        // Same grid as naive: [M, N], block [32], no LDS
        unsafe { self.hip.launch_kernel(func, [m as u32, n as u32, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// Fused GEMM + bias: Y[N,M] = X[N,K] @ W_f16[M,K]^T + bias[M].
    /// Replaces gemm_f16 + transpose_f32 + bias_add_f32 (3 ops → 1).
    /// Grid=[N, 1], Block=[256].
    pub fn gemm_f16_bias(
        &mut self, w: &GpuTensor, x: &GpuTensor, bias: &GpuTensor, y: &GpuTensor,
        m: usize, k: usize, n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemm_f16_bias", kernels::GEMM_F16_BIAS_SRC, "gemm_f16_bias")?;
        let func = &self.functions["gemm_f16_bias"];
        let mut wp = w.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut bp = bias.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut wp as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        // One block per row of X, 256 threads, no LDS
        unsafe { self.hip.launch_kernel(func, [n as u32, 1, 1], [256, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// Batched GEMM for F32: Y[M,N] = A[M,K] @ B[N,K]^T
    pub fn gemm_f32_batched(
        &mut self, a: &GpuTensor, b: &GpuTensor, y: &GpuTensor,
        m: usize, k: usize, n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemm_f32_batched", kernels::GEMM_F32_SRC, "gemm_f32_batched")?;
        let func = &self.functions["gemm_f32_batched"];
        let mut ap = a.buf.as_ptr();
        let mut bp = b.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [m as u32, n as u32, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// LayerNorm with bias (batched): out = gamma * (x - mean) / sqrt(var + eps) + beta
    pub fn layernorm_batched(
        &mut self, x: &GpuTensor, gamma: &GpuTensor, beta: &GpuTensor,
        out: &GpuTensor, batch: usize, n: usize, eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("layernorm_f32", kernels::LAYERNORM_SRC, "layernorm_f32")?;
        let func = &self.functions["layernorm_f32"];
        let mut xp = x.buf.as_ptr();
        let mut gp = gamma.buf.as_ptr();
        let mut bp = beta.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut ni = n as i32;
        let mut ep = eps;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut gp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut ep as *mut _ as *mut c_void,
        ];
        let block_size = std::cmp::min(256, n) as u32;
        // Round up to power of 2 for reduction
        let block_size = block_size.next_power_of_two();
        let shared_mem = block_size * 4;
        unsafe { self.hip.launch_kernel(func, [batch as u32, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// GELU tanh approximation (in-place capable if x == out)
    pub fn gelu_tanh_f32(&mut self, x: &GpuTensor, out: &GpuTensor, n: usize) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gelu_tanh_f32", kernels::GELU_TANH_SRC, "gelu_tanh_f32")?;
        let func = &self.functions["gelu_tanh_f32"];
        let mut xp = x.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        let blocks = ((n + 255) / 256) as u32;
        unsafe { self.hip.launch_kernel(func, [blocks, 1, 1], [256, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// Bias-add: x[batch, n] += bias[n] (in-place, broadcast over batch dim)
    pub fn bias_add_f32(&mut self, x: &GpuTensor, bias: &GpuTensor, batch: usize, n: usize) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("bias_add_f32", kernels::BIAS_ADD_SRC, "bias_add_f32")?;
        let func = &self.functions["bias_add_f32"];
        let mut xp = x.buf.as_ptr();
        let mut bp = bias.buf.as_ptr();
        let mut ni = n as i32;
        let total = (batch * n) as i32;
        let mut ti = total;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut ti as *mut _ as *mut c_void,
        ];
        let blocks = ((total as usize + 255) / 256) as u32;
        unsafe { self.hip.launch_kernel(func, [blocks, 1, 1], [256, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// Transpose [rows, cols] → [cols, rows]
    pub fn transpose_f32(
        &mut self, src: &GpuTensor, dst: &GpuTensor, rows: usize, cols: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("transpose_f32", kernels::TRANSPOSE_SRC, "transpose_f32")?;
        let func = &self.functions["transpose_f32"];
        let mut sp = src.buf.as_ptr();
        let mut dp = dst.buf.as_ptr();
        let mut ri = rows as i32;
        let mut ci = cols as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut sp as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut ri as *mut _ as *mut c_void,
            &mut ci as *mut _ as *mut c_void,
        ];
        let total = rows * cols;
        let blocks = ((total + 255) / 256) as u32;
        unsafe { self.hip.launch_kernel(func, [blocks, 1, 1], [256, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// Fused ViT self-attention: reads QKV [N, 3*hidden], writes out [N, hidden].
    pub fn vit_attention_f32(
        &mut self, qkv: &GpuTensor, out: &GpuTensor,
        n: usize, hidden: usize, num_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("vit_attention_f32", kernels::VIT_ATTENTION_SRC, "vit_attention_f32")?;
        let func = &self.functions["vit_attention_f32"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = qkv.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut ni = n as i32;
        let mut hi = hidden as i32;
        let mut nh = num_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut hi as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = std::cmp::min(256, std::cmp::max(n, head_dim)) as u32;
        let block_size = block_size.next_power_of_two();
        // Shared memory: scores[N] + workspace[block_size]
        let shared_mem = ((n + block_size as usize) * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [num_heads as u32, n as u32, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// Optimized vision attention with tiled K/V loading and 4 queries per block.
    /// ~3-5x faster than vit_attention_f32 via shared memory reuse.
    /// Grid=[num_heads, ceil(N/4)], Block=[256].
    pub fn vit_attention_opt(
        &mut self, qkv: &GpuTensor, out: &GpuTensor,
        n: usize, hidden: usize, num_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("vit_attention_opt", kernels::VIT_ATTENTION_OPT_SRC, "vit_attention_opt")?;
        let func = &self.functions["vit_attention_opt"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = qkv.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut ni = n as i32;
        let mut hi = hidden as i32;
        let mut nh = num_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut hi as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let qpb = 2u32;
        let grid_y = ((n as u32 + qpb - 1) / qpb) as u32;
        // LDS: K_TILE * head_dim * 4 + N * 4 + 256 * 4
        let k_tile = 64u32;
        let shared_mem = (k_tile * head_dim as u32 * 4) + (n as u32 * 4) + (256 * 4);
        unsafe { self.hip.launch_kernel(func, [num_heads as u32, grid_y, 1], [256, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// DFlash draft cross-attention: `B` queries attend to `L` keys/values
    /// with NO causal mask (bidirectional). Supports GQA; `n_heads` must be
    /// a multiple of `n_kv_heads`. See `kernels/src/attention_dflash.hip`
    /// for the full contract.
    ///
    /// Layouts:
    ///   q : [B * n_heads    * head_dim]
    ///   k : [L * n_kv_heads * head_dim]
    ///   v : [L * n_kv_heads * head_dim]
    ///   out: [B * n_heads    * head_dim]
    pub fn attention_dflash_f32(
        &mut self,
        q: &GpuTensor, k: &GpuTensor, v: &GpuTensor, out: &GpuTensor,
        b: usize, l: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("attention_dflash_f32", kernels::ATTENTION_DFLASH_SRC, "attention_dflash_f32")?;
        let func = &self.functions["attention_dflash_f32"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr();
        let mut kp = k.buf.as_ptr();
        let mut vp = v.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut bi = b as i32;
        let mut li = l as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut li as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = std::cmp::min(256, std::cmp::max(l, head_dim)) as u32;
        let block_size = block_size.next_power_of_two();
        // Shared: scores[L] + workspace[block_size]
        let shared_mem = ((l + block_size as usize) * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, b as u32, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch precompilation — compile all kernels a model needs in parallel
    // ═══════════════════════════════════════════════════════════════════════════

    /// Pre-compile all kernels needed for Qwen3.5 inference with a given
    /// weight quantization and KV cache type. Runs hipcc in parallel.
    #[cfg(feature = "deltanet")]
    pub fn precompile_qwen35(&mut self, weight_quant: &str, kv_type: &str, head_dim: usize) -> HipResult<()> {
        self.bind_thread()?;
        // asym kernels #include "turbo_common.h" + "givens_common.h"; the
        // runtime dispatch path (see ensure_givens4_kernel) prepends the
        // header bodies and strips the #includes. We mirror that exactly so
        // the hash matches and the runtime re-uses our cached .hsaco.
        let assemble_asym = |body: &str| -> String {
            let stripped = body
                .replace("#include \"turbo_common.h\"", "")
                .replace("#include \"givens_common.h\"", "");
            format!("{}\n{}\n{}", kernels::TURBO_COMMON_H, kernels::GIVENS_COMMON_SRC, stripped)
        };

        // Common kernels for all Qwen3.5 models (DeltaNet + FullAttn shared ops)
        let mut specs: Vec<(&str, String)> = vec![
            ("rmsnorm",                  kernels::RMSNORM_SRC.to_string()),
            ("add_inplace",              kernels::ADD_INPLACE_SRC.to_string()),
            ("mul",                      kernels::MUL_SRC.to_string()),
            ("silu_mul",                 kernels::SILU_MUL_SRC.to_string()),
            ("sigmoid",                  kernels::SIGMOID_SRC.to_string()),
            ("alpha_gate",               kernels::ALPHA_GATE_SRC.to_string()),
            ("conv1d_silu",              kernels::CONV1D_SILU_SRC.to_string()),
            ("l2_norm",                  kernels::L2_NORM_SRC.to_string()),
            ("fused_qk_l2_norm_scale",   kernels::FUSED_QK_L2_NORM_SCALE_SRC.to_string()),
            ("fused_sigmoid_alpha_gate", kernels::FUSED_SIGMOID_ALPHA_GATE_SRC.to_string()),
            ("conv1d_silu_split",        kernels::CONV1D_SILU_SPLIT_SRC.to_string()),
            ("conv1d_silu_split_tree",   kernels::CONV1D_SILU_SPLIT_TREE_SRC.to_string()),
            ("gated_delta_net_q8_tree",  kernels::GATED_DELTA_NET_Q8_TREE_SRC.to_string()),
            ("sigmoid_mul",              kernels::SIGMOID_MUL_SRC.to_string()),
            ("topk_logits",              kernels::TOPK_LOGITS_SRC.to_string()),
            ("scale_f32",                kernels::SCALE_F32_SRC.to_string()),
            ("gated_norm",               kernels::GATED_NORM_SRC.to_string()),
            ("rope_partial_interleaved", kernels::ROPE_PARTIAL_INTERLEAVED_SRC.to_string()),
            // FullAttn: Q+gate deinterleave split
            ("deinterleave",             kernels::DEINTERLEAVE_SRC.to_string()),
            // DeltaNet: Q/K repeat-interleave for asymmetric MQA (replaces 64+ memcpy_dtod calls per layer on 4B/9B)
            ("repeat_interleave_qk",     kernels::REPEAT_INTERLEAVE_QK_SRC.to_string()),
        ];

        // Weight-format-specific GEMV
        match weight_quant {
            "hfq6" => {
                specs.push(("gemv_hfq6g256", kernels::GEMV_HFQ6G256_SRC.to_string()));
            }
            "mq6" => {
                // MQ6 = FWHT-rotated HFQ6-G256. Needs both the MQ6 GEMV and the
                // raw HFQ6 GEMV (used by a few residual paths).
                specs.push(("gemv_mq6g256", kernels::GEMV_MQ6G256_SRC.to_string()));
                specs.push(("gemv_hfq6g256", kernels::GEMV_HFQ6G256_SRC.to_string()));
            }
            "hfq4" => {
                let (src, module) = kernels::gemv_hfq4g256_for_arch(&self.arch);
                specs.push((module, src.to_string()));
                specs.push(("gemv_hfq4g256_wide", kernels::GEMV_HFQ4G256_WIDE_SRC.to_string()));
                // Multi-projection fused kernels (LA 4-way, FA 3-way, FFN
                // gate+up). Cross-arch — same 4-accumulator inner loop as
                // gemv_hfq4g256.hip; precompile on every arch that uses
                // the HFQ4 weight path.
                specs.push(("fused_qkvza_hfq4g256",
                            kernels::FUSED_QKVZA_HFQ4G256_SRC.to_string()));
                specs.push(("fused_qkv_hfq4g256",
                            kernels::FUSED_QKV_HFQ4G256_SRC.to_string()));
                specs.push(("fused_gate_up_hfq4g256",
                            kernels::FUSED_GATE_UP_HFQ4G256_SRC.to_string()));
                // gfx906/gfx908/gfx94x wave64-native variants — cut
                // wavefront pressure in half on the hottest kernels. Wave32
                // block=[32,1,1] kernels otherwise waste the upper 32 lanes
                // of every wave slot on these wave64-native arches.
                if has_wave64_native(&self.arch) {
                    // Single-token (draft / single-layer paths).
                    specs.push(("fused_qkvza_hfq4g256_wave64",
                                kernels::FUSED_QKVZA_HFQ4G256_WAVE64_SRC.to_string()));
                    specs.push(("fused_qkv_hfq4g256_wave64",
                                kernels::FUSED_QKV_HFQ4G256_WAVE64_SRC.to_string()));
                    specs.push(("fused_gate_up_hfq4g256_wave64",
                                kernels::FUSED_GATE_UP_HFQ4G256_WAVE64_SRC.to_string()));
                    specs.push(("gemv_hfq4g256_moe_gate_up_indexed_wave64",
                                kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_WAVE64_SRC.to_string()));
                    specs.push(("gemv_hfq4g256_moe_down_indexed_wave64",
                                kernels::GEMV_HFQ4G256_MOE_DOWN_INDEXED_WAVE64_SRC.to_string()));
                    // Batched (DFlash verify path — hottest).
                    specs.push(("gemm_qkvza_hfq4g256_wave64",
                                kernels::GEMM_QKVZA_HFQ4G256_WAVE64_SRC.to_string()));
                    specs.push(("gemm_qkv_hfq4g256_wave64",
                                kernels::GEMM_QKV_HFQ4G256_WAVE64_SRC.to_string()));
                    specs.push(("gemm_hfq4g256_wave64",
                                kernels::GEMM_HFQ4G256_WAVE64_SRC.to_string()));
                    specs.push(("gemm_hfq4g256_residual_wave64",
                                kernels::GEMM_HFQ4G256_RESIDUAL_WAVE64_SRC.to_string()));
                    specs.push(("gemv_hfq4g256_moe_gate_up_indexed_batched_wave64",
                                kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_BATCHED_WAVE64_SRC.to_string()));
                    specs.push(("gemv_hfq4g256_moe_down_indexed_batched_wave64",
                                kernels::GEMV_HFQ4G256_MOE_DOWN_INDEXED_BATCHED_WAVE64_SRC.to_string()));
                }
                // gfx1100 multi-row GEMV is opt-in via HIPFIRE_GEMV_ROWS={2,4,8}.
                // Empirically slower than the single-row kernel on gfx1100 at all
                // tested matrix sizes (see commit log / multi-row kernel header),
                // so we only precompile when the env var explicitly requests it.
                if matches!(self.arch.as_str(), "gfx1100" | "gfx1101" | "gfx1102")
                    && gemv_rows_override().unwrap_or(1) > 1
                {
                    specs.push(("gemv_hfq4g256_multirow_rdna3",
                                kernels::GEMV_HFQ4G256_MULTIROW_GFX1100_SRC.to_string()));
                    specs.push(("gemv_hfq4g256_residual_multirow_rdna3",
                                kernels::GEMV_HFQ4G256_RESIDUAL_MULTIROW_GFX1100_SRC.to_string()));
                }
            }
            "mq4" => {
                // MQ4 = FWHT-rotated HFQ4-G256 — default format for current registry.
                // Shares the HFQ4 fused kernels (same blob, different dispatch key)
                // plus MQ-specific rotation kernels.
                let (src, module) = kernels::gemv_hfq4g256_for_arch(&self.arch);
                specs.push((module, src.to_string()));
                specs.push(("gemv_mq4g256", kernels::GEMV_MQ4G256_SRC.to_string()));
                specs.push(("fused_qkvza_hfq4g256",
                            kernels::FUSED_QKVZA_HFQ4G256_SRC.to_string()));
                specs.push(("fused_qkv_hfq4g256",
                            kernels::FUSED_QKV_HFQ4G256_SRC.to_string()));
                specs.push(("fused_gate_up_hfq4g256",
                            kernels::FUSED_GATE_UP_HFQ4G256_SRC.to_string()));
                specs.push(("fused_rmsnorm_mq_rotate",
                            kernels::FUSED_RMSNORM_MQ_ROTATE_SRC.to_string()));
                specs.push(("fused_silu_mul_mq_rotate",
                            kernels::FUSED_SILU_MUL_MQ_ROTATE_SRC.to_string()));
                // gfx906/gfx908/gfx94x wave64 variants — see hfq4 branch for rationale.
                if has_wave64_native(&self.arch) {
                    // Single-token (draft / single-layer paths).
                    specs.push(("fused_qkvza_hfq4g256_wave64",
                                kernels::FUSED_QKVZA_HFQ4G256_WAVE64_SRC.to_string()));
                    specs.push(("fused_qkv_hfq4g256_wave64",
                                kernels::FUSED_QKV_HFQ4G256_WAVE64_SRC.to_string()));
                    specs.push(("fused_gate_up_hfq4g256_wave64",
                                kernels::FUSED_GATE_UP_HFQ4G256_WAVE64_SRC.to_string()));
                    specs.push(("gemv_hfq4g256_moe_gate_up_indexed_wave64",
                                kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_WAVE64_SRC.to_string()));
                    specs.push(("gemv_hfq4g256_moe_down_indexed_wave64",
                                kernels::GEMV_HFQ4G256_MOE_DOWN_INDEXED_WAVE64_SRC.to_string()));
                    // Batched (DFlash verify path — hottest).
                    specs.push(("gemm_qkvza_hfq4g256_wave64",
                                kernels::GEMM_QKVZA_HFQ4G256_WAVE64_SRC.to_string()));
                    specs.push(("gemm_qkv_hfq4g256_wave64",
                                kernels::GEMM_QKV_HFQ4G256_WAVE64_SRC.to_string()));
                    specs.push(("gemm_hfq4g256_wave64",
                                kernels::GEMM_HFQ4G256_WAVE64_SRC.to_string()));
                    specs.push(("gemm_hfq4g256_residual_wave64",
                                kernels::GEMM_HFQ4G256_RESIDUAL_WAVE64_SRC.to_string()));
                    specs.push(("gemv_hfq4g256_moe_gate_up_indexed_batched_wave64",
                                kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_BATCHED_WAVE64_SRC.to_string()));
                    specs.push(("gemv_hfq4g256_moe_down_indexed_batched_wave64",
                                kernels::GEMV_HFQ4G256_MOE_DOWN_INDEXED_BATCHED_WAVE64_SRC.to_string()));
                }
            }
            "q8" => {
                specs.push(("gemv_q8_0", kernels::GEMV_Q8_0_SRC.to_string()));
            }
            _ => {}
        }

        // Embedding kernels — Q8_0 is most common, also cover HFQ4G256/G128 variants
        specs.push(("embedding_q8", kernels::EMBEDDING_Q8_SRC.to_string()));
        specs.push(("embedding_hfq4g256", kernels::EMBEDDING_HFQ4G256_SRC.to_string()));
        specs.push(("embedding_hfq4g128", kernels::EMBEDDING_HFQ4G128_SRC.to_string()));
        specs.push(("embedding_hfq4g256_batched", kernels::EMBEDDING_HFQ4G256_BATCHED_SRC.to_string()));
        specs.push(("embedding_q8_batched", kernels::EMBEDDING_Q8_BATCHED_SRC.to_string()));

        // DeltaNet kernels
        specs.push(("gated_delta_net_q8", kernels::GATED_DELTA_NET_Q8_SRC.to_string()));

        // KV cache kernels. asym3 is the current default — always ships flash.
        // q8 is the compat path with its own flash tile+reduce for long context.
        match kv_type {
            "asym4" => {
                specs.push(("kv_cache_write_asym_k_givens4",
                            assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_GIVENS4_SRC)));
                specs.push(("kv_cache_write_asym_k_givens4_batched",
                            assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_GIVENS4_BATCHED_SRC)));
                specs.push(("attention_flash_asym4_tile",
                            assemble_asym(kernels::ATTENTION_FLASH_ASYM4_TILE_SRC)));
                specs.push(("attention_flash_asym4_tile_batched",
                            assemble_asym(kernels::ATTENTION_FLASH_ASYM4_TILE_BATCHED_SRC)));
                specs.push(("attention_flash_asym_reduce_batched",
                            kernels::ATTENTION_FLASH_ASYM_REDUCE_BATCHED_SRC.to_string()));
            }
            "asym3" => {
                specs.push(("kv_cache_write_asym_k_givens3",
                            assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_GIVENS3_SRC)));
                specs.push(("kv_cache_write_asym_k_givens3_batched",
                            assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_GIVENS3_BATCHED_SRC)));
                specs.push(("attention_flash_asym3_tile",
                            assemble_asym(kernels::ATTENTION_FLASH_ASYM3_TILE_SRC)));
                specs.push(("attention_flash_asym3_tile_batched",
                            assemble_asym(kernels::ATTENTION_FLASH_ASYM3_TILE_BATCHED_SRC)));
                specs.push(("attention_flash_asym_reduce_batched",
                            kernels::ATTENTION_FLASH_ASYM_REDUCE_BATCHED_SRC.to_string()));
            }
            "asym2" => {
                specs.push(("kv_cache_write_asym_k_givens2",
                            assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_GIVENS2_SRC)));
                specs.push(("kv_cache_write_asym_k_givens2_batched",
                            assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_GIVENS2_BATCHED_SRC)));
                specs.push(("attention_flash_asym2_tile",
                            assemble_asym(kernels::ATTENTION_FLASH_ASYM2_TILE_SRC)));
                specs.push(("attention_flash_asym2_tile_batched",
                            assemble_asym(kernels::ATTENTION_FLASH_ASYM2_TILE_BATCHED_SRC)));
                specs.push(("attention_flash_asym_reduce_batched",
                            kernels::ATTENTION_FLASH_ASYM_REDUCE_BATCHED_SRC.to_string()));
            }
            "q8" | _ => {
                specs.push(("kv_cache_write_q8_0", kernels::KV_CACHE_WRITE_Q8_0_SRC.to_string()));
                specs.push(("attention_q8_0_kv",   kernels::ATTENTION_Q8_0_KV_SRC.to_string()));
                specs.push(("attention_q8_0_kv_batched",
                            kernels::ATTENTION_Q8_0_KV_BATCHED_SRC.to_string()));
                specs.push(("kv_cache_write_q8_0_batched",
                            kernels::KV_CACHE_WRITE_Q8_0_BATCHED_SRC.to_string()));
                specs.push(("attention_flash_q8_0_tile",
                            kernels::ATTENTION_FLASH_Q8_0_TILE_SRC.to_string()));
                specs.push(("attention_flash_q8_0_reduce",
                            kernels::ATTENTION_FLASH_Q8_0_REDUCE_SRC.to_string()));
            }
        }

        // Convert to (&str, &str) for the batch API
        let batch: Vec<(&str, &str)> = specs.iter()
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
                "topk_logits"  => vec!["topk_logits_f32"],
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
                "gemv_hfq4g256_moe_gate_up_indexed_wave64" => vec![
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_wave64",
                ],
                "gemv_hfq4g256_moe_down_indexed_wave64" => vec![
                    "gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_wave64",
                ],
                "gemv_hfq4g256_moe_gate_up_indexed_batched_wave64" => vec![
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_batched_wave64",
                ],
                "gemv_hfq4g256_moe_down_indexed_batched_wave64" => vec![
                    "gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched_wave64",
                ],
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

    /// Per-block scoring on Q8_0 K cache. Reads `k_cache` (one layer's
    /// K-cache backing memory; the buffer must be the Q8_0-formatted slab
    /// produced by `KvCache::new_gpu_q8`) for the first `n_pos` positions,
    /// computes per-block mean K and cosine similarity vs the K at
    /// `last_pos`, and writes `n_blocks` f32 scores into `scores_out`.
    ///
    /// One workgroup per output block, 256 threads per workgroup. Each
    /// thread strides through `kv_dim` doing inline f16-scale + i8-value
    /// dequant; a 256-thread shared-memory reduction folds the partial
    /// (dot, ||block||^2, ||last||^2) fragments into one cosine score.
    ///
    /// Phase 2.1 of #93. Replaces the CPU-side dequant + mean-pool +
    /// cosine in `pflash::compute_scores_batched`.
    #[allow(clippy::too_many_arguments)]
    pub fn pflash_score_q8_kv(
        &mut self,
        k_cache: &GpuTensor,
        scores_out: &GpuTensor,
        n_pos: usize,
        n_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        n_blocks: usize,
        last_pos: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(head_dim % 32 == 0, "head_dim must be a multiple of 32 for Q8 KV cache");
        assert!(n_blocks > 0 && block_size > 0 && n_pos > 0);
        assert!(last_pos < n_pos, "last_pos {last_pos} >= n_pos {n_pos}");
        self.ensure_kernel(
            "pflash_score_q8_kv",
            kernels::PFLASH_SCORE_Q8_KV_SRC,
            "pflash_score_q8_kv_blocks",
        )?;
        let func = &self.functions["pflash_score_q8_kv_blocks"];

        let k_ptr = k_cache.buf.as_ptr();
        let s_ptr = scores_out.buf.as_ptr();
        let mut np = n_pos as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut bs = block_size as i32;
        let mut nb = n_blocks as i32;
        let mut lp = last_pos as i32;

        let mut params: Vec<*mut c_void> = vec![
            &k_ptr as *const _ as *mut c_void,
            &s_ptr as *const _ as *mut c_void,
            &mut np as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut nb as *mut _ as *mut c_void,
            &mut lp as *mut _ as *mut c_void,
        ];

        unsafe {
            self.hip.launch_kernel(
                func,
                [n_blocks as u32, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Kernel profiler
    // ═══════════════════════════════════════════════════════════════════════════

    /// Profile all compiled kernels: hardware caps + ISA metadata + occupancy.
    pub fn profile(&self) -> (crate::profiler::GpuCapability, Vec<crate::profiler::KernelProfile>) {
        self.bind_thread_or_warn();
        let vram = self.hip.get_vram_info().map(|(_, t)| t as u64).unwrap_or(0);
        let cu_hint = self.hip
            .get_device_attribute(crate::profiler::HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0)
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
