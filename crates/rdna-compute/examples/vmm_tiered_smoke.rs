// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Exactness and bandwidth smoke for one VMM address with device and host backing.
//!
//! Run on the target GPU with:
//! `HIPFIRE_VMM_SMOKE_DEVICE=0 cargo run -p rdna-compute --features deltanet --example vmm_tiered_smoke`

use hip_bridge::VmmMemoryTier;
use rdna_compute::{DType, Gpu, GpuTensor};

const DEFAULT_TIER_BYTES: usize = 64 << 20;
const DEFAULT_REPEATS: usize = 16;

fn env_usize(name: &str, fallback: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(fallback)
}

fn scale_gb_s(
    gpu: &mut Gpu,
    tensor: &GpuTensor,
    repeats: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    for _ in 0..4 {
        gpu.scale_f32(tensor, 1.0)?;
    }
    gpu.hip.device_synchronize()?;
    let start = gpu.hip.event_create()?;
    let stop = gpu.hip.event_create()?;
    gpu.hip.event_record(&start, None)?;
    for _ in 0..repeats {
        gpu.scale_f32(tensor, 1.0)?;
    }
    gpu.hip.event_record(&stop, None)?;
    gpu.hip.event_synchronize(&stop)?;
    let elapsed_ms = gpu.hip.event_elapsed_ms(&start, &stop)? as f64;
    gpu.hip.event_destroy(start)?;
    gpu.hip.event_destroy(stop)?;
    let traffic_bytes = tensor.byte_size() as f64 * 2.0 * repeats as f64;
    Ok(traffic_bytes / (elapsed_ms / 1_000.0) / 1.0e9)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = env_usize("HIPFIRE_VMM_SMOKE_DEVICE", 0) as i32;
    let requested_tier_bytes = env_usize("HIPFIRE_VMM_TIER_BYTES", DEFAULT_TIER_BYTES);
    let repeats = env_usize("HIPFIRE_VMM_REPEATS", DEFAULT_REPEATS);
    let mut gpu = Gpu::init_with_device(device)?;
    let (free_before, _) = gpu.hip.get_vram_info()?;
    let granularity = gpu.vmm_tiered_recommended_granularity()?;
    let tier_bytes = requested_tier_bytes.div_ceil(granularity) * granularity;
    assert_eq!(tier_bytes % std::mem::size_of::<f32>(), 0);
    let total_bytes = tier_bytes * 2;
    let access = [device];

    let mut mixed = unsafe {
        gpu.alloc_tiered_vmm_tensor(&[total_bytes / 4], DType::F32, tier_bytes, &access)?
    };
    let (free_after_device_prefix, _) = gpu.hip.get_vram_info()?;
    gpu.grow_vmm_tensor_in_tier(&mut mixed, tier_bytes, &access, VmmMemoryTier::Host)?;
    let (free_after_host_tail, _) = gpu.hip.get_vram_info()?;
    assert_eq!(
        gpu.vmm_tier_mapped_bytes(&mixed),
        Some((tier_bytes, tier_bytes))
    );

    let input: Vec<f32> = (0..mixed.numel())
        .map(|i| ((i % 251) as f32 - 125.0) * 0.25)
        .collect();
    let input_bytes =
        unsafe { std::slice::from_raw_parts(input.as_ptr().cast::<u8>(), total_bytes) };
    gpu.hip.memcpy_htod(&mixed.buf, input_bytes)?;
    gpu.scale_f32(&mixed, 2.0)?;
    gpu.hip.device_synchronize()?;
    let mut output = vec![0.0f32; mixed.numel()];
    let output_bytes =
        unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr().cast::<u8>(), total_bytes) };
    gpu.hip.memcpy_dtoh(output_bytes, &mixed.buf)?;
    for (index, (&got, &expected)) in output.iter().zip(&input).enumerate() {
        assert_eq!(
            got.to_bits(),
            (expected * 2.0).to_bits(),
            "mismatch at {index}"
        );
    }

    let device_only = unsafe {
        gpu.alloc_tiered_vmm_tensor(&[total_bytes / 4], DType::F32, total_bytes, &access)?
    };
    let (free_after_device_control, _) = gpu.hip.get_vram_info()?;
    gpu.hip.memcpy_htod(&device_only.buf, input_bytes)?;
    let device_gb_s = scale_gb_s(&mut gpu, &device_only, repeats)?;
    let mixed_gb_s = scale_gb_s(&mut gpu, &mixed, repeats)?;

    println!(
        "vmm_tiered_smoke: PASS device={} granularity={} device_bytes={} host_bytes={} repeats={} device_gb_s={:.3} mixed_gb_s={:.3} mixed_over_device={:.3} vram_delta_device_prefix={} vram_delta_host_tail={} vram_delta_device_control={}",
        device,
        granularity,
        tier_bytes,
        tier_bytes,
        repeats,
        device_gb_s,
        mixed_gb_s,
        mixed_gb_s / device_gb_s,
        free_before.saturating_sub(free_after_device_prefix),
        free_after_device_prefix.saturating_sub(free_after_host_tail),
        free_after_host_tail.saturating_sub(free_after_device_control),
    );
    gpu.free_tensor(device_only)?;
    gpu.free_tensor(mixed)?;
    gpu.ensure_vmm_cleaned()?;
    Ok(())
}
