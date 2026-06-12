// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! One-time tensor mirroring for hetero / PP MTP.

use hip_bridge::HipResult;
use rdna_compute::{Gpu, GpuTensor};

/// Allocate a same-shape, same-dtype tensor on `gpu` and copy `src` into it.
pub fn clone_tensor_same(gpu: &mut Gpu, src: &GpuTensor) -> HipResult<GpuTensor> {
    let dst = gpu.alloc_tensor(&src.shape, src.dtype)?;
    gpu.hip
        .memcpy_dtod_at(&dst.buf, 0, &src.buf, 0, src.byte_size())?;
    Ok(dst)
}

/// Allocate a same-shape, same-dtype tensor on `dst_gpu` and peer-copy `src`.
pub fn clone_tensor_peer(
    src_gpu: &Gpu,
    dst_gpu: &mut Gpu,
    src: &GpuTensor,
) -> HipResult<GpuTensor> {
    debug_assert_ne!(
        src_gpu.device_id, dst_gpu.device_id,
        "clone_tensor_peer: same device; use clone_tensor_same",
    );
    dst_gpu.bind_thread()?;
    let dst = dst_gpu.alloc_tensor(&src.shape, src.dtype)?;
    src_gpu.hip.memcpy_peer(
        &dst.buf,
        dst_gpu.device_id,
        &src.buf,
        src_gpu.device_id,
        src.byte_size(),
    )?;
    Ok(dst)
}
