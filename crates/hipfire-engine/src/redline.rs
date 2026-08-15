// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Generic Redline bench helpers — architecture-neutral GPU snapshot primitives.
//!
//! Arch-specific snapshot builders (`redline_qwen_snapshot`, etc.) remain in
//! `daemon.rs` because they branch on `Qwen35Bundle` / `Deepseek4Bundle` etc.
//! Only the generic helpers that operate on raw buffers are moved here.
//!
//! Relocated verbatim from `crates/hipfire-daemon/src/main.rs` (wave 3).

pub fn redline_capture_json(
    gpu: &rdna_compute::Gpu,
    summary: rdna_compute::replay::ReplayCaptureSummary,
    detail: bool,
) -> serde_json::Value {
    let mut value = serde_json::json!({
        "launches": summary.launch_count,
        "unique_kernels": summary.unique_kernel_count,
        "sequence_hash": format!("{:016x}", summary.sequence_hash),
    });
    if detail {
        value["sequence"] = serde_json::Value::Array(
            gpu.replay
                .recorded_launches()
                .iter()
                .map(|launch| {
                    serde_json::json!({
                        "kernel": launch.kernel.as_str(),
                        "artifact": launch.artifact.as_ref().map(|path| path.display().to_string()),
                        "grid": launch.grid,
                        "block": launch.block,
                        "shared_mem": launch.shared_mem,
                        "kernarg_bytes": launch.kernarg.len(),
                        "kernarg_hex": launch.kernarg.iter().map(|byte| format!("{byte:02x}")).collect::<String>(),
                        "kernarg_hash": format!("{:016x}", {
                            let mut hash = 0xcbf29ce484222325_u64;
                            for byte in &launch.kernarg {
                                hash ^= u64::from(*byte);
                                hash = hash.wrapping_mul(0x100000001b3);
                            }
                            hash
                        }),
                    })
                })
                .collect(),
        );
    }
    value
}

pub fn redline_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

pub fn redline_append_buffer(
    gpu: &rdna_compute::Gpu,
    output: &mut Vec<u8>,
    buffer: &hip_bridge::DeviceBuffer,
) -> Result<(), String> {
    let start = output.len();
    output.resize(start + buffer.size(), 0);
    gpu.hip
        .memcpy_dtoh(&mut output[start..], buffer)
        .map_err(|error| error.to_string())
}

pub fn redline_append_tensor(
    gpu: &rdna_compute::Gpu,
    output: &mut Vec<u8>,
    tensor: &Option<rdna_compute::GpuTensor>,
) -> Result<(), String> {
    if let Some(tensor) = tensor {
        redline_append_buffer(gpu, output, &tensor.buf)?;
    }
    Ok(())
}

pub fn redline_append_tensor_region(
    gpu: &rdna_compute::Gpu,
    output: &mut Vec<u8>,
    regions: &mut Vec<RedlineRegionHash>,
    name: String,
    tensor: &Option<rdna_compute::GpuTensor>,
) -> Result<(), String> {
    let Some(tensor) = tensor else {
        return Ok(());
    };
    let start = output.len();
    redline_append_buffer(gpu, output, &tensor.buf)?;
    let bytes = output.len() - start;
    regions.push(RedlineRegionHash {
        name,
        bytes,
        hash: redline_hash(&output[start..]),
    });
    Ok(())
}

#[derive(PartialEq, Debug)]
pub struct RedlineRegionHash {
    pub name: String,
    pub bytes: usize,
    pub hash: u64,
}

pub fn redline_append_tensor_slice(
    gpu: &rdna_compute::Gpu,
    output: &mut Vec<u8>,
    tensor: &rdna_compute::GpuTensor,
    offset: usize,
    len: usize,
) -> Result<(), String> {
    if offset.saturating_add(len) > tensor.numel() {
        return Err(format!(
            "redline tensor slice {}+{} exceeds {}",
            offset,
            len,
            tensor.numel()
        ));
    }
    let view = tensor.sub_offset(offset, len);
    redline_append_buffer(gpu, output, &view.buf)
}
