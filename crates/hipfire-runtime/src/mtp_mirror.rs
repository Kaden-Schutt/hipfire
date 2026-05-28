// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! Cross-device weight mirroring for hetero MTP.
//!
//! When the MTP head lives on a sibling GPU (e.g. gfx1031) while the trunk
//! lives on the primary (e.g. gfx906), the MTP chain still reads from a
//! small set of trunk weight tensors per cycle — specifically
//! `trunk.token_embd` (embedding lookup, every chain step) and, in
//! `use_full_vocab` mode, `trunk.output` (lm_head GEMV).
//!
//! Trunk weights cannot peer-copy per step (too much data, GB/cycle). They
//! are mirrored ONCE at session init via `peer_clone_tensor` and live on
//! both devices for the lifetime of the session.
//!
//! See `docs/plans/mtp_multi_gpu_split_audit.md` for the per-call lane
//! classification this module enables.

use hip_bridge::HipResult;
use rdna_compute::{DType, Gpu, GpuTensor};

/// Allocate a same-shape, same-dtype tensor on `dst_gpu` and peer-copy the
/// bytes from `src_gpu`'s tensor. Both gpus must already have bidirectional
/// peer access enabled (caller's responsibility — typically via
/// `hip.enable_peer_access` on both sides).
///
/// Synchronous: the peer copy completes before this function returns. Use
/// at session init only; not in any hot path.
///
/// Caller owns the returned tensor and must free it via `dst_gpu.free_tensor`.
/// The source tensor is unchanged.
pub fn peer_clone_tensor(
    src_gpu: &Gpu,
    dst_gpu: &mut Gpu,
    src: &GpuTensor,
) -> HipResult<GpuTensor> {
    let dtype = src.dtype;
    let shape = src.shape.clone();
    let byte_size = src.byte_size();

    dst_gpu.bind_thread()?;
    let dst = dst_gpu.alloc_tensor(&shape, dtype)?;

    // Synchronous peer copy. memcpy_peer blocks the host until the copy
    // lands — fine for init-time, eliminates the need for an event +
    // wait_event handshake here.
    src_gpu.hip.memcpy_peer(
        &dst.buf, dst_gpu.device_id,
        &src.buf,  src_gpu.device_id,
        byte_size,
    )?;

    debug_assert_eq!(dst.dtype, dtype);
    debug_assert_eq!(dst.shape, shape);
    debug_assert_eq!(dst.byte_size(), byte_size);

    Ok(dst)
}

/// Convenience: clone an `[m, k]` tensor of arbitrary dtype. Asserts the
/// source shape matches `[m, k]` so callers fail fast on the mirror boundary.
pub fn peer_clone_2d(
    src_gpu: &Gpu,
    dst_gpu: &mut Gpu,
    src: &GpuTensor,
    m: usize,
    k: usize,
) -> HipResult<GpuTensor> {
    assert!(
        (src.shape.len() == 2 && src.shape[0] == m && src.shape[1] == k)
            || (src.shape.len() == 1 && src.shape[0] == m * k),
        "peer_clone_2d: expected source shape [{m}, {k}] or [{}], got {:?}",
        m * k, src.shape,
    );
    peer_clone_tensor(src_gpu, dst_gpu, src)
}

/// Hetero-MTP view of the trunk weights that the MTP head's per-step
/// forward path actually reads. Holds drafter-resident clones of the
/// fields needed; the original `Qwen35Weights` stays on the target gpu
/// for trunk verify.
///
/// Construct via [`MirroredTrunkWeights::for_compressed_mtp`] (sidecar
/// path: only `token_embd` is mirrored) or
/// [`MirroredTrunkWeights::for_full_vocab_mtp`] (bundled path: both
/// `token_embd` and `output` mirrored).
pub struct MirroredTrunkWeights {
    /// Drafter-gpu copy of `Qwen35Weights.token_embd`.
    pub token_embd: GpuTensor,
    /// Optional drafter-gpu copy of `Qwen35Weights.output` (lm_head).
    /// Only allocated for the `use_full_vocab` chain path. `None` when
    /// the head ships a compressed sidecar (lm_head_draft) — in that
    /// case the chain reads from `head.weights.lm_head_draft` instead
    /// of trunk lm_head.
    pub output: Option<GpuTensor>,
}

impl MirroredTrunkWeights {
    /// Free the drafter-side clones. Pass the SAME `drafter_gpu` instance
    /// that allocated them. Idempotent on the `output` Option.
    pub fn free_gpu(self, drafter_gpu: &mut Gpu) {
        let _ = drafter_gpu.free_tensor(self.token_embd);
        if let Some(out) = self.output {
            let _ = drafter_gpu.free_tensor(out);
        }
    }

    /// Size (in bytes) of all mirrored tensors on the drafter side. Useful
    /// for VRAM accounting / pre-flight checks.
    pub fn drafter_bytes(&self) -> usize {
        self.token_embd.byte_size()
            + self.output.as_ref().map(|t| t.byte_size()).unwrap_or(0)
    }

    /// `Some(DType)` of `token_embd`. Tiny helper to make accounting
    /// printing easier without exposing the GpuTensor internals.
    pub fn token_embd_dtype(&self) -> DType {
        self.token_embd.dtype
    }
}
