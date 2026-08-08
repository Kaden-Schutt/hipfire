// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Prove whether gfx1201 HMM can migrate one stable allocation between host
//! memory and VRAM, and measure the first-touch and resident GPU paths.

use hip_bridge::{
    HIP_CPU_DEVICE_ID, HIP_MEM_ADVISE_SET_ACCESSED_BY, HIP_MEM_ADVISE_SET_PREFERRED_LOCATION,
};
use rdna_compute::{DType, Gpu, GpuTensor};

const DEFAULT_BYTES: usize = 1 << 30;
const DEFAULT_REPEATS: usize = 8;

fn env_usize(name: &str, fallback: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(fallback)
}

fn repeated_scale_gb_s(
    gpu: &mut Gpu,
    tensor: &GpuTensor,
    repeats: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
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
    Ok((tensor.byte_size() * 2 * repeats) as f64 / (elapsed_ms / 1_000.0) / 1.0e9)
}

fn first_touch_ms(
    gpu: &mut Gpu,
    tensor: &GpuTensor,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    let start = gpu.hip.event_create()?;
    let stop = gpu.hip.event_create()?;
    let wall_start = std::time::Instant::now();
    gpu.hip.event_record(&start, None)?;
    gpu.scale_f32(tensor, 1.0)?;
    gpu.hip.event_record(&stop, None)?;
    gpu.hip.event_synchronize(&stop)?;
    let wall_ms = wall_start.elapsed().as_secs_f64() * 1_000.0;
    let event_ms = gpu.hip.event_elapsed_ms(&start, &stop)? as f64;
    gpu.hip.event_destroy(start)?;
    gpu.hip.event_destroy(stop)?;
    Ok((wall_ms, event_ms))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = env_usize("HIPFIRE_MANAGED_SMOKE_DEVICE", 0) as i32;
    let requested_bytes = env_usize("HIPFIRE_MANAGED_SMOKE_BYTES", DEFAULT_BYTES);
    let repeats = env_usize("HIPFIRE_MANAGED_SMOKE_REPEATS", DEFAULT_REPEATS);
    let bytes = requested_bytes.div_ceil(4096) * 4096;
    assert_eq!(bytes % std::mem::size_of::<f32>(), 0);

    let mut gpu = Gpu::init_with_device(device)?;
    let (free_before, _) = gpu.hip.get_vram_info()?;
    let managed_buf = gpu.hip.malloc_managed(bytes)?;
    let attrs = gpu.hip.pointer_get_attributes(&managed_buf)?;
    let (free_after_alloc, _) = gpu.hip.get_vram_info()?;
    gpu.hip.mem_advise(
        &managed_buf,
        0,
        bytes,
        HIP_MEM_ADVISE_SET_ACCESSED_BY,
        device,
    )?;
    gpu.hip.mem_advise(
        &managed_buf,
        0,
        bytes,
        HIP_MEM_ADVISE_SET_PREFERRED_LOCATION,
        HIP_CPU_DEVICE_ID,
    )?;
    gpu.hip
        .mem_prefetch_async(&managed_buf, 0, bytes, HIP_CPU_DEVICE_ID, None)?;
    gpu.hip.device_synchronize()?;
    let (free_after_cpu_prefetch, _) = gpu.hip.get_vram_info()?;

    let host_values =
        unsafe { std::slice::from_raw_parts_mut(managed_buf.as_ptr().cast::<f32>(), bytes / 4) };
    for (index, value) in host_values.iter_mut().enumerate() {
        *value = ((index % 251) as f32 - 125.0) * 0.25;
    }

    gpu.hip.mem_advise(
        &managed_buf,
        0,
        bytes,
        HIP_MEM_ADVISE_SET_PREFERRED_LOCATION,
        device,
    )?;
    let gpu_prefetch_start = std::time::Instant::now();
    gpu.hip
        .mem_prefetch_async(&managed_buf, 0, bytes, device, None)?;
    gpu.hip.device_synchronize()?;
    let gpu_prefetch_ms = gpu_prefetch_start.elapsed().as_secs_f64() * 1_000.0;
    let (free_after_gpu_prefetch, _) = gpu.hip.get_vram_info()?;

    let tensor = GpuTensor {
        buf: managed_buf,
        shape: vec![bytes / 4],
        dtype: DType::F32,
    };
    let resident_gb_s = repeated_scale_gb_s(&mut gpu, &tensor, repeats)?;

    gpu.hip.mem_advise(
        &tensor.buf,
        0,
        bytes,
        HIP_MEM_ADVISE_SET_PREFERRED_LOCATION,
        HIP_CPU_DEVICE_ID,
    )?;
    let cpu_prefetch_start = std::time::Instant::now();
    gpu.hip
        .mem_prefetch_async(&tensor.buf, 0, bytes, HIP_CPU_DEVICE_ID, None)?;
    gpu.hip.device_synchronize()?;
    let cpu_prefetch_ms = cpu_prefetch_start.elapsed().as_secs_f64() * 1_000.0;
    let (free_after_evict, _) = gpu.hip.get_vram_info()?;
    let (first_wall_ms, first_event_ms) = first_touch_ms(&mut gpu, &tensor)?;
    let post_touch_gb_s = repeated_scale_gb_s(&mut gpu, &tensor, repeats)?;
    let (free_after_first_touch, _) = gpu.hip.get_vram_info()?;

    println!(
        "managed_paging_smoke: PASS device={} bytes={} attrs_mem_type={} attrs_managed={} alloc_vram_delta={} cpu_prefetch_vram_delta={} gpu_prefetch_vram_delta={} evicted_vram={} first_touch_vram={} gpu_prefetch_ms={:.3} cpu_prefetch_ms={:.3} first_touch_wall_ms={:.3} first_touch_event_ms={:.3} resident_gb_s={:.3} post_touch_gb_s={:.3}",
        device,
        bytes,
        attrs.mem_type,
        attrs.is_managed,
        free_before.saturating_sub(free_after_alloc),
        free_before.saturating_sub(free_after_cpu_prefetch),
        free_after_cpu_prefetch.saturating_sub(free_after_gpu_prefetch),
        free_before.saturating_sub(free_after_evict),
        free_before.saturating_sub(free_after_first_touch),
        gpu_prefetch_ms,
        cpu_prefetch_ms,
        first_wall_ms,
        first_event_ms,
        resident_gb_s,
        post_touch_gb_s,
    );

    gpu.hip.free(tensor.buf)?;
    Ok(())
}
