// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! Raw-bit oracle for the fixed-geometry gfx1100 harmonic local expert tail.

use hip_bridge::HipRuntime;
use rdna_compute::{DType, Gpu, GpuTensor};

const HIDDEN: usize = 4096;
const INTERMEDIATE: usize = 2048;
const TOP_K: usize = 6;
const SWIGLU_LIMIT: f32 = 10.0;

fn as_bytes<T>(values: &[T]) -> &[u8] {
    // SAFETY: every initialized value is readable as bytes and the returned
    // slice cannot outlive the borrowed input.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn as_bytes_mut<T>(values: &mut [T]) -> &mut [u8] {
    // SAFETY: u8 has alignment one and aliases only for the duration of the
    // synchronous HIP copy.
    unsafe {
        std::slice::from_raw_parts_mut(
            values.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(values),
        )
    }
}

fn resolve_device(selector: &str) -> Result<(i32, String, String, String), String> {
    let hip = HipRuntime::load().map_err(|error| format!("HIP discovery: {error:?}"))?;
    let count = hip
        .device_count()
        .map_err(|error| format!("HIP device count: {error:?}"))?;
    let mut matches = Vec::new();
    for device_id in 0..count {
        let arch = hip
            .get_arch(device_id)
            .map_err(|error| format!("HIP device {device_id} arch: {error:?}"))?;
        let name = hip
            .device_name(device_id)
            .map_err(|error| format!("HIP device {device_id} name: {error:?}"))?;
        let pci = hip
            .device_pci_bus_id(device_id)
            .map_err(|error| format!("HIP device {device_id} PCI identity: {error:?}"))?;
        let selected = selector
            .strip_prefix("arch:")
            .is_some_and(|expected| arch.eq_ignore_ascii_case(expected))
            || selector.strip_prefix("name:").is_some_and(|needle| {
                name.to_ascii_lowercase()
                    .contains(&needle.to_ascii_lowercase())
            })
            || selector
                .strip_prefix("pci:")
                .is_some_and(|expected| pci.eq_ignore_ascii_case(expected));
        if selected {
            matches.push((device_id, pci, arch, name));
        }
    }
    let [(device_id, pci, arch, name)] = matches.as_slice() else {
        return Err(format!(
            "selector {selector:?} matched {} visible devices; use a unique selector",
            matches.len()
        ));
    };
    let pinned = hip
        .device_by_pci_bus_id(pci)
        .map_err(|error| format!("HIP pin {pci}: {error:?}"))?;
    if pinned != *device_id {
        return Err(format!(
            "selector {selector:?} changed ordinal during PCI pin: {device_id} -> {pinned}"
        ));
    }
    Ok((*device_id, pci.clone(), arch.clone(), name.clone()))
}

fn synthetic_mq2_lloyd(rows: usize, k: usize) -> Vec<u8> {
    let groups = k / 256;
    let row_bytes = groups * 72;
    let mut bytes = vec![0_u8; rows * row_bytes];
    // Finite non-negative fp16 codebook: 0, 0.25, 0.5, 1.0.
    let codebook = [0x0000_u16, 0x3400, 0x3800, 0x3c00];
    for row in 0..rows {
        for group in 0..groups {
            let base = row * row_bytes + group * 72;
            for (index, bits) in codebook.into_iter().enumerate() {
                bytes[base + 2 * index..base + 2 * index + 2]
                    .copy_from_slice(&bits.to_le_bytes());
            }
            for byte in 0..64 {
                bytes[base + 8 + byte] = (row
                    .wrapping_mul(29)
                    .wrapping_add(group * 17)
                    .wrapping_add(byte * 73)) as u8;
            }
        }
    }
    bytes
}

fn clear(gpu: &Gpu, tensors: &[&GpuTensor], bits: u32) -> Result<(), String> {
    for tensor in tensors {
        let words = tensor.buf.size() / std::mem::size_of::<u32>();
        let sentinel = vec![bits; words];
        gpu.hip
            .memcpy_htod(&tensor.buf, as_bytes(&sentinel))
            .map_err(|error| format!("clear oracle tensor: {error}"))?;
    }
    Ok(())
}

fn download_bits(gpu: &Gpu, tensor: &GpuTensor) -> Result<Vec<u32>, String> {
    let mut values = vec![0_u32; tensor.buf.size() / std::mem::size_of::<u32>()];
    gpu.hip
        .memcpy_dtoh(as_bytes_mut(&mut values), &tensor.buf)
        .map_err(|error| format!("download oracle tensor: {error}"))?;
    Ok(values)
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut device = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--device" => device = Some(args.next().ok_or("--device needs a selector")?),
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    let selector = device.ok_or("--device is required")?;
    let (device_id, pci, arch, name) = resolve_device(&selector)?;
    if !arch.eq_ignore_ascii_case("gfx1100") {
        return Err(format!("local-tail oracle requires gfx1100, got {arch}"));
    }
    let mut gpu = Gpu::init_with_device(device_id)
        .map_err(|error| format!("initialize gfx1100 at {pci}: {error}"))?;

    let gate_up_weight = gpu
        .upload_raw(
            &synthetic_mq2_lloyd(2 * INTERMEDIATE, HIDDEN),
            &[2 * INTERMEDIATE, HIDDEN],
        )
        .map_err(|error| format!("upload gate/up weight: {error}"))?;
    let down_weight = gpu
        .upload_raw(
            &synthetic_mq2_lloyd(HIDDEN, INTERMEDIATE),
            &[HIDDEN, INTERMEDIATE],
        )
        .map_err(|error| format!("upload down weight: {error}"))?;
    let gate_up_pointer = [gate_up_weight.buf.as_ptr() as usize as u64];
    let down_pointer = [down_weight.buf.as_ptr() as usize as u64];
    let gate_up_ptrs = gpu
        .upload_raw(as_bytes(&gate_up_pointer), &[1])
        .map_err(|error| format!("upload gate/up pointer table: {error}"))?;
    let down_ptrs = gpu
        .upload_raw(as_bytes(&down_pointer), &[1])
        .map_err(|error| format!("upload down pointer table: {error}"))?;
    let local_ids = gpu
        .upload_raw(as_bytes(&[0_u32; TOP_K]), &[TOP_K])
        .map_err(|error| format!("upload local IDs: {error}"))?;
    let local_count = gpu
        .alloc_tensor(&[1], DType::F32)
        .map_err(|error| format!("allocate local count: {error}"))?;
    let x_values: Vec<f32> = (0..HIDDEN)
        .map(|index| ((index % 53) as f32 - 26.0) * 0.001_953_125)
        .collect();
    let x_rot = gpu
        .upload_f32(&x_values, &[HIDDEN])
        .map_err(|error| format!("upload x_rot: {error}"))?;

    let mut baseline = Vec::new();
    let mut candidate = Vec::new();
    for shape in [
        [TOP_K, INTERMEDIATE],
        [TOP_K, INTERMEDIATE],
        [TOP_K, INTERMEDIATE],
        [TOP_K, HIDDEN],
    ] {
        baseline.push(
            gpu.alloc_tensor(&shape, DType::F32)
                .map_err(|error| format!("allocate baseline scratch: {error}"))?,
        );
        candidate.push(
            gpu.alloc_tensor(&shape, DType::F32)
                .map_err(|error| format!("allocate candidate scratch: {error}"))?,
        );
    }

    const SENTINEL: u32 = 0x7f12_3456;
    let mut comparisons = 0_u64;
    for count in 0..=TOP_K {
        clear(&gpu, &baseline.iter().collect::<Vec<_>>(), SENTINEL)?;
        clear(&gpu, &candidate.iter().collect::<Vec<_>>(), SENTINEL)?;
        gpu.hip
            .memcpy_htod(&local_count.buf, as_bytes(&[count as u32]))
            .map_err(|error| format!("upload local count {count}: {error}"))?;

        if count != 0 {
            {
                let mut device = gpu
                    .try_gfx1100()
                    .ok_or_else(|| "baseline exact-gfx1100 proof disappeared".to_owned())?;
                device
                    .mq2_lloyd_moe_gate_up_k4_lds(
                        &gate_up_ptrs,
                        &local_ids,
                        &x_rot,
                        &baseline[0],
                        &baseline[1],
                        2 * INTERMEDIATE,
                        HIDDEN,
                        count,
                    )
                    .map_err(|error| format!("baseline gate/up count {count}: {error}"))?;
            }
            gpu.deepseek4_silu_mul_clamp_f32_batched(
                &baseline[0],
                &baseline[1],
                &baseline[0],
                INTERMEDIATE,
                count,
                SWIGLU_LIMIT,
            )
            .map_err(|error| format!("baseline activation count {count}: {error}"))?;
            gpu.rotate_x_mq_batched(
                &baseline[0],
                &baseline[2],
                INTERMEDIATE,
                count,
            )
            .map_err(|error| format!("baseline rotate count {count}: {error}"))?;
            {
                let mut device = gpu
                    .try_gfx1100()
                    .ok_or_else(|| "baseline down exact-gfx1100 proof disappeared".to_owned())?;
                device
                    .mq2_lloyd_moe_down_expanded_lds_candidate(
                        &down_ptrs,
                        &local_ids,
                        &baseline[2],
                        &baseline[3],
                        HIDDEN,
                        INTERMEDIATE,
                        count,
                        1,
                    )
                    .map_err(|error| format!("baseline down count {count}: {error}"))?;
            }
        }

        {
            let mut device = gpu
                .try_gfx1100()
                .ok_or_else(|| "candidate exact-gfx1100 proof disappeared".to_owned())?;
            device
                .harmonic_mq2_lloyd_local_tail(
                    &gate_up_ptrs,
                    &down_ptrs,
                    &local_ids,
                    &local_count,
                    &x_rot,
                    &candidate[0],
                    &candidate[1],
                    &candidate[2],
                    &candidate[3],
                    HIDDEN,
                    INTERMEDIATE,
                    SWIGLU_LIMIT,
                )
                .map_err(|error| format!("candidate tail count {count}: {error}"))?;
        }
        gpu.hip
            .device_synchronize()
            .map_err(|error| format!("synchronize count {count}: {error}"))?;
        for index in 0..baseline.len() {
            let expected = download_bits(&gpu, &baseline[index])?;
            let actual = download_bits(&gpu, &candidate[index])?;
            if actual != expected {
                let mismatch = actual
                    .iter()
                    .zip(&expected)
                    .position(|(got, expected)| got != expected);
                return Err(format!(
                    "local-tail mismatch count={count} tensor={index} word={mismatch:?}"
                ));
            }
            comparisons += actual.len() as u64;
        }
    }

    println!(
        "harmonic local tail exact: selector={} pci={} arch={} name={:?} counts=0..={} raw_bit_comparisons={}",
        selector, pci, arch, name, TOP_K, comparisons
    );
    for tensor in baseline
        .into_iter()
        .chain(candidate)
        .chain([
            gate_up_weight,
            down_weight,
            gate_up_ptrs,
            down_ptrs,
            local_ids,
            local_count,
            x_rot,
        ])
    {
        gpu.free_tensor(tensor)
            .map_err(|error| format!("free local-tail tensor: {error}"))?;
    }
    Ok(())
}
