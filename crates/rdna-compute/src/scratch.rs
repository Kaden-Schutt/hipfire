// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Scratch buffer state extracted from the Gpu god object.
//! Owns all per-GPU scratch allocations for FWHT rotation, FP16/FP8/INT8
//! activation conversion, and ParoQuant activation copies.

use crate::kernels;
use crate::{DType, GpuTensor};
use hip_bridge::{
    DeviceBuffer, Function, HipResult, HipRuntime, KernargBlob, Module, Stream,
    HIP_ERROR_INVALID_IMAGE,
};
use std::collections::HashMap;
use std::ffi::c_void;

// ── ScratchState ─────────────────────────────────────────────────────────

pub struct ScratchState {
    pub mq_signs1: Option<GpuTensor>,
    pub mq_signs2: Option<GpuTensor>,
    pub mq_signs1_128: Option<GpuTensor>,
    pub mq_signs2_128: Option<GpuTensor>,
    pub mq_x_rot: Option<GpuTensor>,
    pub mq_x_rot_fp8: Option<DeviceBuffer>,
    pub mq_x_rot_fp8_bytes: usize,
    pub mq_x_q8: Option<DeviceBuffer>,
    pub mq_x_scales: Option<DeviceBuffer>,
    /// Persistent gfx1100 K=2048 RMSNorm+MQ state shared by the split and
    /// wavegrid experiments. The wavegrid layout is eight f32 partials, one
    /// f32 RMS value, and three u32 epoch counters, padded to 64 bytes; the
    /// split path uses its first f32 as the RMS handoff.
    pub mq_rmsnorm_wavegrid_scratch: Option<DeviceBuffer>,
    /// Dedicated F32 temporary for the unfused GEMV-residual alias fallback.
    /// Lazily allocated and grown on demand; no other scratch path uses it.
    pub gemv_residual_tmp: Option<GpuTensor>,
    pub paro_x_scratch: Option<GpuTensor>,
    /// Rotation scratch buffers for PARO fused-kernel dispatch. 4 × [k] F32
    /// buffers, lazily allocated and grown on demand. Used by
    /// `fused_qkvza_paro4g128t` (4 explicit) and `fused_gate_up_paro4g128t`
    /// (1 explicit + `mq_x_rot` internal). `ensure_paro_fused_scratch`
    /// allocates/grows; `DeviceBuffer::alias()` builds per-call descriptors.
    pub paro_fused_scratch: Option<Vec<GpuTensor>>,
    pub fp16_x_scratch: Option<DeviceBuffer>,
    pub fp16_x_scratch_bytes: usize,
    pub fp16_x_source_ptr: *mut c_void,
    pub fp8_x_scratch: Option<DeviceBuffer>,
    pub fp8_x_scratch_bytes: usize,
    pub fp8_x_source_ptr: *mut c_void,
    pub q8_1_mmq_x_scratch: Option<DeviceBuffer>,
    pub q8_1_mmq_x_scratch_bytes: usize,
    /// Partials buffer for the deterministic K-split GEMM (ksplit_det):
    /// [K_SPLITS][batch_size][M] fp32, grows-never-shrinks.
    pub ksplit_det_partials: Option<DeviceBuffer>,
    pub ksplit_det_partials_bytes: usize,
    /// Partials buffer for the multi-workgroup parallel sampler
    /// (`sample_topk_partial` → `sample_topk_finalize`). Holds
    /// `[n_blocks*TOP_K]` f32 values followed by `[n_blocks*TOP_K]` i32 indices
    /// in one allocation; grows-never-shrinks.
    pub sample_partials: Option<DeviceBuffer>,
    pub sample_partials_bytes: usize,
}

/// Restore the pre-fix FP16 activation-cache semantics:
/// `convert_fp16_x_uncached` leaves `fp16_x_source_ptr` set, and
/// `gemm_q8_0_residual_wmma` takes the pointer-keyed `ensure_fp16_x` path.
///
/// A DIAGNOSTIC, not a supported mode. It re-enables a read of the shared FP16
/// scratch that can hold a different tensor's conversion — the defect that made
/// Escha-W2 batched prefill produce finite, fluent, ~1e-1-wrong output from
/// layer 1 onward.
///
/// It exists so the blast radius of that fix is a MEASUREMENT rather than an
/// argument: run any model's prefill with and without it and diff the logits
/// byte-for-byte. Bit-identical means that model never hit the stale read, so
/// the fix did not change it. Doing this with one binary and two runs is
/// strictly better evidence than two binaries, which can differ for reasons
/// nobody enumerated.
pub fn fp16_x_legacy_cache() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("HIPFIRE_FP16_X_LEGACY_CACHE").as_deref() == Ok("1"))
}

// ── Shared kernel dispatch helpers ──────────────────────────────────────

/// Compile and load a kernel, caching the result in `modules`/`functions`.
pub(crate) fn compile_and_load_kernel(
    compiler: &mut crate::compiler::KernelCompiler,
    hip: &HipRuntime,
    modules: &mut HashMap<String, Module>,
    functions: &mut HashMap<String, Function>,
    module_name: &str,
    source: &str,
    func_name: &str,
) -> HipResult<()> {
    if functions.contains_key(func_name) {
        return Ok(());
    }
    let obj_path = compiler.compile(module_name, source)?;
    let obj_path_str = obj_path.to_str().unwrap().to_string();
    // Alias the launched function name to this arch's compiled artifact so the
    // retained-PM4 capture can resolve func_name -> owning .hsaco even when the
    // arch-selected module name differs (e.g. gemv_hfq4g256_residual launched vs
    // module gemv_hfq4g256_residual_rdna3 on RDNA3). Additive; no-op when equal.
    if func_name != module_name {
        compiler.register_func_artifact(func_name, std::path::PathBuf::from(&obj_path_str));
    }
    if !modules.contains_key(module_name) {
        let module = module_load_or_recompile(hip, compiler, module_name, source, &obj_path_str)?;
        modules.insert(module_name.to_string(), module);
    }
    let module = &modules[module_name];
    let func = hip.module_get_function(module, func_name).map_err(|error| {
        let context = format!(
            "hipModuleGetFunction failed for symbol {func_name:?} in module {module_name:?}: {error}"
        );
        hip_bridge::HipError::new(error.code, &context)
    })?;
    functions.insert(func_name.to_string(), func);
    Ok(())
}

/// Load a compiled module, self-healing a stale/invalid cached image. If
/// `hipModuleLoad` rejects the `.hsaco` as an invalid device image
/// (`HIP_ERROR_INVALID_IMAGE`) — e.g. a cross-build blob left in a shared
/// `.hipfire_kernels` cache — evict it, recompile from source, and retry once.
/// Any other error propagates unchanged. (Fix for the bench/run "device kernel
/// image is invalid" crash when two daemon builds share a cwd kernel cache.)
pub(crate) fn module_load_or_recompile(
    hip: &HipRuntime,
    compiler: &mut crate::compiler::KernelCompiler,
    module_name: &str,
    source: &str,
    obj_path: &str,
) -> HipResult<Module> {
    match hip.module_load(obj_path) {
        Ok(m) => Ok(m),
        Err(e) if e.code == HIP_ERROR_INVALID_IMAGE => {
            eprintln!(
                "  {module_name}: cached kernel image invalid (HIP {}); recompiling from source",
                e.code
            );
            let fresh = compiler.recompile(module_name, source)?;
            hip.module_load(fresh.to_str().unwrap())
        }
        Err(e) => Err(e),
    }
}

/// Launch a kernel, routing through the blob path when graph capture, replay
/// recording, or force_blob is active. Shared between `Gpu::launch_maybe_blob`
/// and `ScratchState` methods so the branching logic stays in one place.
///
/// Invariant: for any body, `capture_blobs.len()` after a HipGraph capture
/// equals `replay.recorded_launches().len()` after a ReplayController capture.
/// A divergence means a helper bypassed the replay recorder.
pub(crate) fn launch_maybe_blob(
    hip: &HipRuntime,
    compiler: Option<&crate::compiler::KernelCompiler>,
    functions: &HashMap<String, Function>,
    stream: Option<&Stream>,
    capture_blobs: &mut Vec<Vec<u8>>,
    capture_mode: bool,
    force_blob_path: bool,
    mut replay: Option<&mut crate::replay::ReplayController>,
    func_name: &str,
    grid: [u32; 3],
    block: [u32; 3],
    shared_mem: u32,
    params: &mut [*mut c_void],
    blob_builder: impl FnOnce() -> KernargBlob,
) -> HipResult<()> {
    let record = replay.as_ref().map_or(false, |r| r.is_recording());
    if record || capture_mode || force_blob_path {
        let mut blob = blob_builder();
        blob.pad_to(16);
        if record {
            // Single decision point for how a launch is recorded: same
            // artifact lookup shape as `Gpu::launch_maybe_blob_bound`.
            let artifact = compiler
                .as_ref()
                .and_then(|c| {
                    c
                .compiled_kernels()
                .get(func_name)
                .or_else(|| match func_name {
                    "mq_rotate_x" => c.compiled_kernels().get("gemv_mq4g256"),
                    "deinterleave_f32_batched" => {
                        c.compiled_kernels().get("deinterleave_batched")
                    }
                    name if name.starts_with("gemv_hfq4g256_residual_sigmoid_scaled_gpu") => {
                        c
                            .compiled_kernels()
                            .get("gemv_hfq4g256_residual_scaled")
                    }
                    "gemv_hfq4g256_moe_gate_up_k8_indexed" => c
                        .compiled_kernels()
                        .get("gemv_hfq4g256_moe_gate_up_indexed"),
                    name if name.starts_with("gemv_hfq4g256_multirow_r") => c
                        .compiled_kernels()
                        .get("gemv_hfq4g256_multirow_default")
                        .or_else(|| {
                            c
                                .compiled_kernels()
                                .get("gemv_hfq4g256_multirow_rdna3")
                        }),
                    name if name.starts_with("gemv_hfq4g256_residual_multirow_r") => c
                        .compiled_kernels()
                        .get("gemv_hfq4g256_residual_multirow_default")
                        .or_else(|| {
                            c
                                .compiled_kernels()
                                .get("gemv_hfq4g256_residual_multirow_rdna3")
                        }),
                    _ => None,
                })
                .or_else(|| {
                    func_name
                        .strip_suffix("_f32")
                        .and_then(|name| c.compiled_kernels().get(name))
                })
                        .cloned()
                });
            replay.as_mut().unwrap().record_hip_launch_typed_bound(
                hip,
                func_name,
                artifact,
                grid,
                block,
                shared_mem,
                blob.as_bytes(),
                None,
            );
        }
        if capture_mode {
            capture_blobs.push(blob.into_vec());
            let buf = capture_blobs.last_mut().unwrap();
            let func = &functions[func_name];
            unsafe { hip.launch_kernel_blob(func, grid, block, shared_mem, stream, buf.as_mut_slice()) }
        } else {
            let mut bytes = blob.into_vec();
            let func = &functions[func_name];
            unsafe { hip.launch_kernel_blob(func, grid, block, shared_mem, stream, bytes.as_mut_slice()) }
        }
    } else {
        let func = &functions[func_name];
        unsafe { hip.launch_kernel(func, grid, block, shared_mem, stream, params) }
    }
}

/// Predicate for the FP16/FP8 scratch fast path. The convert kernel must run
/// iff either recorder is active (`is_recording || capture_mode`) or the
/// cached source pointer differs. This is the same predicate
/// `Gpu::launch_maybe_blob_bound` uses for deciding whether to record, so
/// the skip and the record stay coupled: if the kernel does not run it is
/// not recorded, and if a recorder is active the kernel always runs.
#[inline]
pub(crate) fn scratch_must_convert(
    capture_mode: bool,
    is_recording: bool,
    cached_ptr: *mut c_void,
    src_ptr: *mut c_void,
) -> bool {
    is_recording || capture_mode || cached_ptr != src_ptr
}

/// Unified predicate for routing through the blob path. Mirrors
/// `Gpu::launch_maybe_blob_bound`'s `record || capture_mode || force_blob_path`.
#[inline]
pub(crate) fn use_blob_path(is_recording: bool, capture_mode: bool, force_blob_path: bool) -> bool {
    is_recording || capture_mode || force_blob_path
}

// ── FWHT sign table generation (deterministic LCG) ──────────────────────

fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
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

// ── ScratchState helpers ────────────────────────────────────────────────

/// Grow a `Option<DeviceBuffer>` scratch slot to at least `needed` bytes,
/// RELEASING the previous allocation.
///
/// `DeviceBuffer` has no `Drop` impl, so the natural-looking
///
/// ```ignore
/// if self.foo_bytes < needed {
///     self.foo = Some(hip.malloc(needed)?);
///     self.foo_bytes = needed;
/// }
/// ```
///
/// silently leaks the old allocation on every growth step, for every
/// architecture. These scratches are keyed on batch/context, so a serving
/// session that sees a sequence of increasing shapes leaks one buffer per
/// distinct larger shape.
///
/// Measured on gfx1100 (25.8 GB) before this fix, issuing requests with
/// growing prompts and sampling VRAM between them: repeating the SAME shape
/// cost +0.0 MB (the buffer is correctly reused), while each larger shape
/// added its FULL size rather than the increment --
/// +199.6, +411.1, +754.9, +1459.9, +2906.8 MB across five shapes, 5.7 GB
/// retained. That is what pushed a multi-turn DFlash session into
/// `hipMalloc: out of memory` on turn 3.
///
/// The old buffer may still be referenced by kernels already enqueued on the
/// stream, and `HipRuntime::free` documents that the caller must ensure the
/// buffer is idle, so synchronise before releasing. Growth is monotonic and
/// bounded by the largest shape ever seen, so this sync is rare and its cost
/// is irrelevant next to the allocation it replaces.
///
/// Frees BEFORE allocating: the whole point is to run when memory is tight,
/// and holding both at once is what we are trying to avoid. On allocation
/// failure the slot is left empty with a zero byte count, so the `?` in the
/// caller returns before any `unwrap`, and a later call retries cleanly.
fn grow_scratch_buffer(
    hip: &HipRuntime,
    slot: &mut Option<DeviceBuffer>,
    have_bytes: &mut usize,
    needed: usize,
) -> HipResult<()> {
    if *have_bytes >= needed && slot.is_some() {
        return Ok(());
    }
    if let Some(old) = slot.take() {
        if crate::graph::any_graph_captured() {
            // A captured hipGraph embeds the pointers live at capture time, so
            // releasing this buffer would make every later replay read freed
            // memory. Measured: freeing here breaks qwen35 outright, every turn
            // empty with `spec_step: HipError(700) ... reset_recurrent`. Retain
            // it (the pre-existing behaviour) and let the process reclaim it.
            std::mem::forget(old);
        } else {
            hip.device_synchronize()?;
            let _ = hip.free(old);
        }
    }
    *have_bytes = 0;
    let fresh = hip.malloc(needed)?;
    *slot = Some(fresh);
    *have_bytes = needed;
    Ok(())
}

impl ScratchState {
    /// Ensure the ksplit_det partials scratch is at least `n_bytes`, growing
    /// (never shrinking). Returns the device pointer. No init needed: every
    /// valid output cell is written exactly once per K-split before finalize.
    pub fn ensure_ksplit_det_partials(
        &mut self,
        hip: &HipRuntime,
        n_bytes: usize,
    ) -> HipResult<*mut c_void> {
        grow_scratch_buffer(
            hip,
            &mut self.ksplit_det_partials,
            &mut self.ksplit_det_partials_bytes,
            n_bytes,
        )?;
        Ok(self.ksplit_det_partials.as_ref().unwrap().as_ptr())
    }

    /// Ensure the parallel-sampler partials scratch is at least `n_bytes`,
    /// growing (never shrinking). Returns the device base pointer. No init
    /// needed: every valid cell is written by `sample_topk_partial` before
    /// `sample_topk_finalize` reads it.
    pub fn ensure_sample_partials(
        &mut self,
        hip: &HipRuntime,
        n_bytes: usize,
    ) -> HipResult<*mut c_void> {
        grow_scratch_buffer(
            hip,
            &mut self.sample_partials,
            &mut self.sample_partials_bytes,
            n_bytes,
        )?;
        Ok(self.sample_partials.as_ref().unwrap().as_ptr())
    }

    /// Ensure the dedicated GEMV-residual temporary can hold at least
    /// `min_elems` F32 values, growing on demand and never shrinking.
    pub fn ensure_gemv_residual_tmp(
        &mut self,
        hip: &HipRuntime,
        device_id: i32,
        min_elems: usize,
    ) -> HipResult<&GpuTensor> {
        crate::graph::bind_thread(hip, device_id)?;
        let needed_bytes = min_elems * 4;
        let needs_grow = self
            .gemv_residual_tmp
            .as_ref()
            .map_or(true, |tmp| tmp.buf.size() < needed_bytes);
        if needs_grow {
            self.gemv_residual_tmp = Some(GpuTensor {
                buf: hip.malloc(needed_bytes)?,
                shape: vec![min_elems],
                dtype: DType::F32,
            });
        }
        Ok(self.gemv_residual_tmp.as_ref().unwrap())
    }

    /// Lazily initialize MagnumQuant FWHT sign tables (256 floats each, seeds 42 and 1042).
    pub fn ensure_mq_signs(
        &mut self,
        hip: &HipRuntime,
        pool: &mut crate::pool::GpuPool,
        device_id: i32,
    ) -> HipResult<()> {
        crate::graph::bind_thread(hip, device_id)?;
        if self.mq_signs1.is_some() {
            return Ok(());
        }
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let s1b: Vec<u8> = s1.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let s2b: Vec<u8> = s2.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let s1t = alloc_tensor_on(hip, pool, device_id, &[256], DType::F32)?;
        let s2t = alloc_tensor_on(hip, pool, device_id, &[256], DType::F32)?;
        hip.memcpy_htod(&s1t.buf, &s1b)?;
        hip.memcpy_htod(&s2t.buf, &s2b)?;
        // Allocate scratch buffers — 32K elements covers K up to 32768
        let x_rot = alloc_tensor_on(hip, pool, device_id, &[32768], DType::F32)?;
        let x_q8 = hip.malloc(32768)?; // INT8 buffer for dp4a
        let x_scales = hip.malloc(128 * 4)?; // up to 128 groups × f32
        self.mq_signs1 = Some(s1t);
        self.mq_signs2 = Some(s2t);
        self.mq_x_rot = Some(x_rot);
        self.mq_x_q8 = Some(x_q8);
        self.mq_x_scales = Some(x_scales);
        Ok(())
    }

    /// Lazily allocate and zero the persistent gfx1100 RMSNorm state.
    pub fn ensure_mq_rmsnorm_wavegrid_scratch(
        &mut self,
        hip: &HipRuntime,
        device_id: i32,
    ) -> HipResult<()> {
        crate::graph::bind_thread(hip, device_id)?;
        if self.mq_rmsnorm_wavegrid_scratch.is_none() {
            let scratch = hip.malloc(64)?;
            hip.memset(&scratch, 0, 64)?;
            self.mq_rmsnorm_wavegrid_scratch = Some(scratch);
        }
        Ok(())
    }

    /// Lazily initialize MagnumQuant FWHT sign tables for G128 (128 floats each,
    /// seeds 43 and 1043). Also allocates the shared `mq_x_rot` scratch if not
    /// already present — the G256 path (`ensure_mq_signs`) normally owns that
    /// allocation, but the G128 path must be self-sufficient so models that carry
    /// only MQ4G128 weights still get the scratch buffer.
    pub fn ensure_mq_signs_128(
        &mut self,
        hip: &HipRuntime,
        pool: &mut crate::pool::GpuPool,
        device_id: i32,
    ) -> HipResult<()> {
        crate::graph::bind_thread(hip, device_id)?;
        if self.mq_signs1_128.is_some() && self.mq_x_rot.is_some() {
            return Ok(());
        }
        if self.mq_signs1_128.is_none() {
            let signs1 = gen_fwht_signs(43, 128);
            let signs2 = gen_fwht_signs(1043, 128);
            let s1b: Vec<u8> = signs1.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let s2b: Vec<u8> = signs2.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let s1t = alloc_tensor_on(hip, pool, device_id, &[128], DType::F32)?;
            let s2t = alloc_tensor_on(hip, pool, device_id, &[128], DType::F32)?;
            hip.memcpy_htod(&s1t.buf, &s1b)?;
            hip.memcpy_htod(&s2t.buf, &s2b)?;
            self.mq_signs1_128 = Some(s1t);
            self.mq_signs2_128 = Some(s2t);
        }
        // Allocate shared rotation scratch if ensure_mq_signs (G256 path) has not run yet.
        if self.mq_x_rot.is_none() {
            let x_rot = alloc_tensor_on(hip, pool, device_id, &[32768], DType::F32)?;
            self.mq_x_rot = Some(x_rot);
        }
        Ok(())
    }

    /// Ensure the ParoQuant activation scratch buffer is allocated (F32, sized for dim).
    pub fn ensure_paro_scratch(
        &mut self,
        hip: &HipRuntime,
        device_id: i32,
        dim: usize,
    ) -> HipResult<()> {
        crate::graph::bind_thread(hip, device_id)?;
        if let Some(ref s) = self.paro_x_scratch {
            if s.buf.size() >= dim * 4 {
                return Ok(());
            }
        }
        let buf = hip.malloc(dim * 4)?; // F32
        self.paro_x_scratch = Some(GpuTensor {
            buf,
            shape: vec![dim],
            dtype: DType::F32,
        });
        Ok(())
    }

    /// Ensure 4 rotation scratch buffers for Paro fused-kernel dispatch.
    /// Each buffer is sized `[k]` F32. On first call, allocates all 4;
    /// on subsequent calls, grows any buffer whose size is < k.
    /// Separate from `paro_x_scratch` (single activation buffer) because
    /// the fused kernels rotate weights internally and need multiple
    /// independent rotation output buffers.
    pub fn ensure_paro_fused_scratch(
        &mut self,
        hip: &HipRuntime,
        device_id: i32,
        k: usize,
    ) -> HipResult<()> {
        crate::graph::bind_thread(hip, device_id)?;
        let needed_bytes = k * 4; // F32
        match &mut self.paro_fused_scratch {
            Some(bufs) => {
                // Grow any buffer that's too small (never shrinks).
                for buf in bufs.iter_mut() {
                    if buf.buf.size() < needed_bytes {
                        *buf = GpuTensor {
                            buf: hip.malloc(needed_bytes)?,
                            shape: vec![k],
                            dtype: DType::F32,
                        };
                    }
                }
            }
            None => {
                let mut vec = Vec::with_capacity(4);
                for _ in 0..4 {
                    vec.push(GpuTensor {
                        buf: hip.malloc(needed_bytes)?,
                        shape: vec![k],
                        dtype: DType::F32,
                    });
                }
                self.paro_fused_scratch = Some(vec);
            }
        }
        Ok(())
    }

    /// Ensure the FP16 X scratch contains the conversion of `x`. Skips the
    /// convert kernel if `x.buf.as_ptr()` matches the last converted source.
    /// When either recorder is active (`capture_mode` or `replay.is_recording()`)
    /// the kernel always runs so the tape stays complete; the skip only applies
    /// to the live non-recording path. Returns the FP16 device pointer.
    ///
    /// # The pointer key is only sound for back-to-back same-`x` dispatches
    ///
    /// It identifies the SOURCE BUFFER, not its CONTENTS. A caller that hands
    /// this a stable scratch allocation whose contents are rewritten between
    /// calls — a per-layer activation buffer, which is most of them — gets the
    /// first layer's conversion for every subsequent layer, silently. Two
    /// call sites have already been bitten (the MTP lm_head, τ 1.85 → 1.01;
    /// Escha-W2's `wo`) and both were fixed by switching to
    /// [`Self::convert_fp16_x_uncached`].
    ///
    /// So: use this ONLY when the same `x` is consumed by several dispatches
    /// with nothing writing to it in between (the Q/K/V case it was built
    /// for). If `x` is a per-layer or per-step buffer, use
    /// `convert_fp16_x_uncached`; one extra elementwise kernel is far cheaper
    /// than the GEMM it feeds, and is always correct. Writers of a cached
    /// buffer may instead call [`Self::invalidate_x_caches_for`].
    pub fn ensure_fp16_x(
        &mut self,
        hip: &HipRuntime,
        compiler: &mut crate::compiler::KernelCompiler,
        modules: &mut HashMap<String, Module>,
        functions: &mut HashMap<String, Function>,
        stream: Option<&Stream>,
        capture_blobs: &mut Vec<Vec<u8>>,
        capture_mode: bool,
        force_blob_path: bool,
        replay: &mut crate::replay::ReplayController,
        x: &GpuTensor,
        n_elems: usize,
    ) -> HipResult<*mut c_void> {
        compile_and_load_kernel(
            compiler,
            hip,
            modules,
            functions,
            "convert_f32_to_f16",
            kernels::GEMM_HFQ4G256_RESIDUAL_FP16_SRC,
            "convert_f32_to_f16",
        )?;

        let src_ptr = x.buf.as_ptr();
        let needed = n_elems * 2;

        // Grow scratch if needed (never shrinks), releasing the old buffer.
        if self.fp16_x_scratch_bytes < needed {
            grow_scratch_buffer(
                hip,
                &mut self.fp16_x_scratch,
                &mut self.fp16_x_scratch_bytes,
                needed,
            )?;
            self.fp16_x_source_ptr = std::ptr::null_mut(); // force reconversion after realloc
        }

        let must_convert = scratch_must_convert(capture_mode, replay.is_recording(), self.fp16_x_source_ptr, src_ptr);
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
            launch_maybe_blob(
                hip,
                Some(&*compiler),
                functions,
                stream,
                capture_blobs,
                capture_mode,
                force_blob_path,
                Some(replay),
                "convert_f32_to_f16",
                [grid, 1, 1],
                [256, 1, 1],
                0,
                &mut conv_params,
                || {
                    let mut b = KernargBlob::new();
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

    /// Convert F32 to F16 without caching. Used when the same x tensor
    /// pointer is reused with different contents across layers (e.g.
    /// DeepSeek V4 prefill reuses the same x_in pointer with new contents
    /// every layer), where pointer-keyed caching would read stale FP16.
    /// Always launches; both recorders observe the same launch via
    /// `launch_maybe_blob`'s unified `record || capture_mode || force_blob` gate.
    ///
    /// # It also INVALIDATES the cache, and must
    ///
    /// This writes the SHARED `fp16_x_scratch` — the same buffer
    /// [`Self::ensure_fp16_x`] hands out — so after it runs, any cached
    /// `fp16_x_source_ptr` marker describes a buffer that no longer holds that
    /// pointer's data. Leaving the marker set makes the very next
    /// `ensure_fp16_x` for that pointer a CACHE HIT onto this call's
    /// conversion of a completely different tensor.
    ///
    /// That is not hypothetical. On Escha-W2 batched prefill it fired every
    /// layer: `wo` (`gemm_q8_0_residual_wmma`, cached) converted
    /// `dn_normed_batch` in layer 0; `wqkv` (`gemm_q8_0_wmma`, uncached)
    /// overwrote the scratch with `x_rot_batch` in layer 1; layer 1's `wo`
    /// then hit the stale marker and multiplied its weights by the LA
    /// rmsnorm output (magnitude ~7.5e-1) instead of the gated-norm output
    /// (~1.9e-3) — a ~400x amplification, finite and fluent, that showed up
    /// only as a moved argmax. The DeepSeek-V4 gfx942 call site had already
    /// discovered this and nulls the marker by hand after calling; doing it
    /// here makes that hand-repair unnecessary and closes the same hole for
    /// every other caller.
    pub fn convert_fp16_x_uncached(
        &mut self,
        hip: &HipRuntime,
        compiler: &mut crate::compiler::KernelCompiler,
        modules: &mut HashMap<String, Module>,
        functions: &mut HashMap<String, Function>,
        stream: Option<&Stream>,
        capture_blobs: &mut Vec<Vec<u8>>,
        capture_mode: bool,
        force_blob_path: bool,
        replay: &mut crate::replay::ReplayController,
        x: &GpuTensor,
        n_elems: usize,
    ) -> HipResult<*mut c_void> {
        compile_and_load_kernel(
            compiler,
            hip,
            modules,
            functions,
            "convert_f32_to_f16",
            kernels::GEMM_HFQ4G256_RESIDUAL_FP16_SRC,
            "convert_f32_to_f16",
        )?;

        let needed = n_elems * 2;
        if self.fp16_x_scratch_bytes < needed {
            grow_scratch_buffer(
                hip,
                &mut self.fp16_x_scratch,
                &mut self.fp16_x_scratch_bytes,
                needed,
            )?;
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
        launch_maybe_blob(
                hip,
                Some(&*compiler),
                functions,
                stream,
                capture_blobs,
                capture_mode,
                force_blob_path,
                Some(replay),
            "convert_f32_to_f16",
            [grid, 1, 1],
            [256, 1, 1],
            0,
            &mut conv_params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(in_ptr);
                b.push_ptr(out_ptr);
                b.push_i32(n_val);
                b
            },
        )?;
        // The shared scratch no longer holds whatever `fp16_x_source_ptr`
        // says it holds. See this function's doc comment — dropping the
        // marker is part of the contract, not an optimisation.
        if !fp16_x_legacy_cache() {
            self.fp16_x_source_ptr = std::ptr::null_mut();
        }
        Ok(self.fp16_x_scratch.as_ref().unwrap().as_ptr())
    }

    /// Ensure the FP8 (E4M3) X scratch contains the conversion of `x`
    /// (an F32 GpuTensor). Returns the FP8 device pointer. gfx12 only —
    /// uses cvt_pk_fp8_f32. Caches by `x.buf.as_ptr()` like its FP16
    /// sibling so back-to-back same-X GEMM dispatches skip reconversion.
    /// The cache is bypassed when either recorder is active, matching
    /// `ensure_fp16_x` and `scratch_must_convert`.
    pub fn ensure_fp8_x(
        &mut self,
        hip: &HipRuntime,
        compiler: &mut crate::compiler::KernelCompiler,
        modules: &mut HashMap<String, Module>,
        functions: &mut HashMap<String, Function>,
        stream: Option<&Stream>,
        capture_blobs: &mut Vec<Vec<u8>>,
        capture_mode: bool,
        force_blob_path: bool,
        replay: &mut crate::replay::ReplayController,
        x: &GpuTensor,
        n_elems: usize,
    ) -> HipResult<*mut c_void> {
        compile_and_load_kernel(
            compiler,
            hip,
            modules,
            functions,
            "pack_f32_to_fp8_gfx12",
            kernels::PACK_F32_TO_FP8_GFX12_SRC,
            "pack_f32_to_fp8_gfx12",
        )?;

        let src_ptr = x.buf.as_ptr();
        let needed = n_elems; // 1 byte per element

        if self.fp8_x_scratch_bytes < needed {
            grow_scratch_buffer(
                hip,
                &mut self.fp8_x_scratch,
                &mut self.fp8_x_scratch_bytes,
                needed,
            )?;
            self.fp8_x_source_ptr = std::ptr::null_mut();
        }

        let must_convert = scratch_must_convert(capture_mode, replay.is_recording(), self.fp8_x_source_ptr, src_ptr);
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
            let grid = ((n_elems + 4095) / 4096) as u32;
            launch_maybe_blob(
                hip,
                Some(&*compiler),
                functions,
                stream,
                capture_blobs,
                capture_mode,
                force_blob_path,
                Some(replay),
                "pack_f32_to_fp8_gfx12",
                [grid, 1, 1],
                [256, 1, 1],
                0,
                &mut conv_params,
                || {
                    let mut b = KernargBlob::new();
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
    /// so a 128-column batch tile is contiguous for each K tile. Always
    /// launches; both recorders observe it via the unified blob gate.
    pub fn ensure_q8_1_mmq_x(
        &mut self,
        hip: &HipRuntime,
        compiler: &mut crate::compiler::KernelCompiler,
        modules: &mut HashMap<String, Module>,
        functions: &mut HashMap<String, Function>,
        stream: Option<&Stream>,
        capture_blobs: &mut Vec<Vec<u8>>,
        capture_mode: bool,
        force_blob_path: bool,
        replay: &mut crate::replay::ReplayController,
        device_id: i32,
        x: &GpuTensor,
        batch_size: usize,
        k: usize,
    ) -> HipResult<*mut c_void> {
        crate::graph::bind_thread(hip, device_id)?;
        compile_and_load_kernel(
            compiler,
            hip,
            modules,
            functions,
            "gemm_hfq4g256_residual_mmq",
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_SRC,
            "quantize_q8_1_mmq_ds4",
        )?;

        let blocks_k = (k + 127) / 128;
        let block_q8_1_mmq_bytes = 144usize;
        let needed = blocks_k * batch_size * block_q8_1_mmq_bytes;
        grow_scratch_buffer(
            hip,
            &mut self.q8_1_mmq_x_scratch,
            &mut self.q8_1_mmq_x_scratch_bytes,
            needed,
        )?;

        let src_ptr = x.buf.as_ptr();
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
            launch_maybe_blob(
                hip,
                Some(&*compiler),
                functions,
                stream,
                capture_blobs,
                capture_mode,
                force_blob_path,
                Some(replay),
                "quantize_q8_1_mmq_ds4",
                [grid_x, grid_y, 1],
                [256, 1, 1],
                0,
                &mut params,
                || {
                    let mut b = KernargBlob::new();
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

    /// Invalidate the FP16/FP8 activation scratch caches. Must be called
    /// whenever the scratch buffer used by MagnumQuant rotation is
    /// written — the scratch pointer is stable but the DATA changes per
    /// rotation; without this invalidation, FP8/FP16 activation scratch
    /// returns stale data on every call after the first within a forward
    /// pass (silent correctness bug).
    pub fn invalidate_x_caches_for(&mut self, dst_ptr: *mut c_void) {
        if self.fp16_x_source_ptr == dst_ptr {
            self.fp16_x_source_ptr = std::ptr::null_mut();
        }
        if self.fp8_x_source_ptr == dst_ptr {
            self.fp8_x_source_ptr = std::ptr::null_mut();
        }
    }

    // ── Rotation methods ────────────────────────────────────────────────

    /// Standalone FWHT rotation for MagnumQuant (MQ4). Writes K floats into x_rot.
    /// Exposed so callers can batch one rotation across multiple GEMVs that share x
    /// (e.g., Q/K/V projections all consume the same post-RMSNorm x).
    ///
    /// NOTE: caller must have ensured the kernel (`mq_rotate_x` in module
    /// `gemv_mq4g256`) before calling this method.
    pub fn rotate_x_mq(
        &mut self,
        hip: &HipRuntime,
        compiler: &crate::compiler::KernelCompiler,
        functions: &HashMap<String, Function>,
        stream: Option<&Stream>,
        capture_blobs: &mut Vec<Vec<u8>>,
        capture_mode: bool,
        force_blob_path: bool,
        replay: &mut crate::replay::ReplayController,
        pool: &mut crate::pool::GpuPool,
        device_id: i32,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
    ) -> HipResult<()> {
        crate::graph::bind_thread(hip, device_id)?;
        self.ensure_mq_signs(hip, pool, device_id)?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;
        let xp = x.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &s1_ptr as *const _ as *mut c_void,
            &s2_ptr as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::mq_rotate_bytes(k);
        let timer = crate::profile::begin_timer(hip, "fwht", "mq_rotate_x", bytes);
        let result = launch_maybe_blob(
                hip,
                Some(compiler),
                functions,
                stream,
                capture_blobs,
                capture_mode,
                force_blob_path,
                Some(replay),
            "mq_rotate_x",
            [n_groups, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(xrp);
                b.push_ptr(s1_ptr);
                b.push_ptr(s2_ptr);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }

    /// Batched `rotate_x_mq`. Grid.y is the batch dim.
    pub fn rotate_x_mq_batched(
        &mut self,
        hip: &HipRuntime,
        compiler: &crate::compiler::KernelCompiler,
        functions: &HashMap<String, Function>,
        stream: Option<&Stream>,
        capture_blobs: &mut Vec<Vec<u8>>,
        capture_mode: bool,
        force_blob_path: bool,
        replay: &mut crate::replay::ReplayController,
        pool: &mut crate::pool::GpuPool,
        device_id: i32,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        crate::graph::bind_thread(hip, device_id)?;
        self.ensure_mq_signs(hip, pool, device_id)?;
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
        let timer = crate::profile::begin_timer(hip, "fwht", "mq_rotate_x_batched", bytes);
        let result = launch_maybe_blob(
                hip,
                Some(compiler),
                functions,
                stream,
                capture_blobs,
                capture_mode,
                force_blob_path,
                Some(replay),
            "mq_rotate_x",
            [n_groups * batch_size as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(xrp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }

    /// FWHT-128 standalone rotation for MQ4G128 activations.
    pub fn rotate_x_mq_128(
        &mut self,
        hip: &HipRuntime,
        compiler: &crate::compiler::KernelCompiler,
        functions: &HashMap<String, Function>,
        stream: Option<&Stream>,
        capture_blobs: &mut Vec<Vec<u8>>,
        capture_mode: bool,
        force_blob_path: bool,
        replay: &mut crate::replay::ReplayController,
        pool: &mut crate::pool::GpuPool,
        device_id: i32,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
    ) -> HipResult<()> {
        crate::graph::bind_thread(hip, device_id)?;
        self.ensure_mq_signs_128(hip, pool, device_id)?;
        let s1_ptr = self.mq_signs1_128.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2_128.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 128) as u32;
        let xp = x.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &s1_ptr as *const _ as *mut c_void,
            &s2_ptr as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::mq_rotate_bytes(k);
        let timer = crate::profile::begin_timer(hip, "fwht", "mq_rotate_x_128", bytes);
        let result = launch_maybe_blob(
                hip,
                Some(compiler),
                functions,
                stream,
                capture_blobs,
                capture_mode,
                force_blob_path,
                Some(replay),
            "mq_rotate_x_128",
            [n_groups, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(xrp);
                b.push_ptr(s1_ptr);
                b.push_ptr(s2_ptr);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }

    pub fn rotate_x_mq_awq(
        &mut self,
        hip: &HipRuntime,
        compiler: &crate::compiler::KernelCompiler,
        functions: &HashMap<String, Function>,
        stream: Option<&Stream>,
        capture_blobs: &mut Vec<Vec<u8>>,
        capture_mode: bool,
        force_blob_path: bool,
        replay: &mut crate::replay::ReplayController,
        pool: &mut crate::pool::GpuPool,
        device_id: i32,
        x: &GpuTensor,
        awq_scale: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
    ) -> HipResult<()> {
        crate::graph::bind_thread(hip, device_id)?;
        self.ensure_mq_signs(hip, pool, device_id)?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;
        let xp = x.buf.as_ptr();
        let awp = awq_scale.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &awp as *const _ as *mut c_void,
            &s1_ptr as *const _ as *mut c_void,
            &s2_ptr as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];
        let bytes = k * 4 * 3 + 2 * 256 * 4;
        let timer = crate::profile::begin_timer(hip, "fwht", "rotate_x_mq_awq", bytes);
        let result = launch_maybe_blob(
                hip,
                Some(compiler),
                functions,
                stream,
                capture_blobs,
                capture_mode,
                force_blob_path,
                Some(replay),
            "rotate_x_mq_awq",
            [n_groups, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(xrp);
                b.push_ptr(awp);
                b.push_ptr(s1_ptr);
                b.push_ptr(s2_ptr);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }

    /// Phase A Stage A — F2 batched AWQ variant of `rotate_x_mq`.
    /// Grid.y is the batch dim — processes [N × K] x/x_rot.
    pub fn rotate_x_mq_awq_batched(
        &mut self,
        hip: &HipRuntime,
        compiler: &crate::compiler::KernelCompiler,
        functions: &HashMap<String, Function>,
        stream: Option<&Stream>,
        capture_blobs: &mut Vec<Vec<u8>>,
        capture_mode: bool,
        force_blob_path: bool,
        replay: &mut crate::replay::ReplayController,
        pool: &mut crate::pool::GpuPool,
        device_id: i32,
        x: &GpuTensor,
        awq_scale: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        crate::graph::bind_thread(hip, device_id)?;
        self.ensure_mq_signs(hip, pool, device_id)?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;
        let mut xp = x.buf.as_ptr();
        let mut awp = awq_scale.buf.as_ptr();
        let mut xrp = x_rot.buf.as_ptr();
        let mut s1 = s1_ptr;
        let mut s2 = s2_ptr;
        let mut kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut xrp as *mut _ as *mut c_void,
            &mut awp as *mut _ as *mut c_void,
            &mut s1 as *mut _ as *mut c_void,
            &mut s2 as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
        ];
        let bytes = (k * 4 * 3 + 2 * 256 * 4) * batch_size;
        let timer = crate::profile::begin_timer(hip, "fwht", "rotate_x_mq_awq_batched", bytes);
        let result = launch_maybe_blob(
                hip,
                Some(compiler),
                functions,
                stream,
                capture_blobs,
                capture_mode,
                force_blob_path,
                Some(replay),
            "rotate_x_mq_awq",
            [n_groups, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(xrp);
                b.push_ptr(awp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }

    /// Fused FWHT rotation + FP8 pack for the decode FP8 path.
    /// Writes both F32 (into `x_rot`) and FP8 (into `mq_x_rot_fp8`
    /// sibling scratch) in one kernel launch. Returns the FP8 buffer's
    /// device pointer for the caller to feed directly to the FP8 GEMV.
    /// gfx12-only — uses cvt_pk_fp8_f32.
    pub fn rotate_x_mq_dual_fp8(
        &mut self,
        hip: &HipRuntime,
        functions: &mut HashMap<String, Function>,
        stream: Option<&Stream>,
        capture_blobs: &mut Vec<Vec<u8>>,
        capture_mode: bool,
        force_blob_path: bool,
        compiler: &mut crate::compiler::KernelCompiler,
        modules: &mut HashMap<String, Module>,
        replay: &mut crate::replay::ReplayController,
        pool: &mut crate::pool::GpuPool,
        device_id: i32,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
    ) -> HipResult<*mut c_void> {
        self.ensure_mq_signs(hip, pool, device_id)?;
        compile_and_load_kernel(
            compiler,
            hip,
            modules,
            functions,
            "mq_rotate_x_dual_fp8_gfx12",
            kernels::MQ_ROTATE_X_DUAL_FP8_GFX12_SRC,
            "mq_rotate_x_dual_fp8_gfx12",
        )?;
        // Lazily allocate the FP8 sibling scratch sized to match k bytes.
        grow_scratch_buffer(hip, &mut self.mq_x_rot_fp8, &mut self.mq_x_rot_fp8_bytes, k)?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let xp = x.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let xfp = self.mq_x_rot_fp8.as_ref().unwrap().as_ptr();
        let n_groups = (k / 256) as u32;
        let kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &xfp as *const _ as *mut c_void,
            &s1_ptr as *const _ as *mut c_void,
            &s2_ptr as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::mq_rotate_bytes(k) + k;
        let timer = crate::profile::begin_timer(hip, "fwht", "mq_rotate_x_dual_fp8", bytes);
        let result = launch_maybe_blob(
                hip,
                Some(compiler),
                functions,
                stream,
                capture_blobs,
                capture_mode,
                force_blob_path,
                Some(replay),
            "mq_rotate_x_dual_fp8_gfx12",
            [n_groups, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(xrp);
                b.push_ptr(xfp);
                b.push_ptr(s1_ptr);
                b.push_ptr(s2_ptr);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(hip);
        }
        self.invalidate_x_caches_for(xrp);
        result?;
        Ok(xfp)
    }

    /// Standalone MQ8 rotate + INT8 quantize of x into internal `mq_x_q8`/`mq_x_scales`.
    /// After this, `gemv_mq8g256_prerotated` can be called multiple times with the same x.
    pub fn rotate_quantize_x_mq8(
        &mut self,
        hip: &HipRuntime,
        compiler: &crate::compiler::KernelCompiler,
        functions: &HashMap<String, Function>,
        stream: Option<&Stream>,
        capture_blobs: &mut Vec<Vec<u8>>,
        capture_mode: bool,
        force_blob_path: bool,
        replay: &mut crate::replay::ReplayController,
        pool: &mut crate::pool::GpuPool,
        device_id: i32,
        x: &GpuTensor,
        k: usize,
    ) -> HipResult<()> {
        crate::graph::bind_thread(hip, device_id)?;
        self.ensure_mq_signs(hip, pool, device_id)?;

        let xq_ptr = self.mq_x_q8.as_ref().unwrap().as_ptr();
        let xs_ptr = self.mq_x_scales.as_ref().unwrap().as_ptr();
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;

        let xp = x.buf.as_ptr();
        let xq = xq_ptr;
        let xs = xs_ptr;
        let s1 = s1_ptr;
        let s2 = s2_ptr;
        let kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &xq as *const _ as *mut c_void,
            &xs as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::mq_rotate_bytes(k) + (k / 256) * 4 + k;
        let timer = crate::profile::begin_timer(hip, "fwht", "mq8_rotate_quantize_x", bytes);
        let result = launch_maybe_blob(
            hip,
            Some(compiler),
            functions,
            stream,
            capture_blobs,
            capture_mode,
            force_blob_path,
            Some(replay),
            "mq8_rotate_quantize_x",
            [n_groups, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(xq);
                b.push_ptr(xs);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(hip);
        }
        // Invalidate caches for the internal q8 buffers' output? The output
        // is internal scratch, not x_rot, so no fp16/fp8 cache invalidation needed.
        // But keep symmetric: if any future caller aliases, this remains safe.
        result
    }
}

// ── Internal helpers ────────────────────────────────────────────────────

fn alloc_tensor_on(
    hip: &HipRuntime,
    pool: &mut crate::pool::GpuPool,
    device_id: i32,
    shape: &[usize],
    dtype: DType,
) -> HipResult<GpuTensor> {
    crate::graph::bind_thread(hip, device_id)?;
    let numel: usize = shape.iter().product();
    let byte_size = numel * dtype.size();
    let buf = pool.alloc(hip, byte_size)?;
    Ok(GpuTensor {
        buf,
        shape: shape.to_vec(),
        dtype,
    })
}
