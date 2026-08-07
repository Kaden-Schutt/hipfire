// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! Raw-bit oracle for the exact-gfx1100 harmonic route partition kernel.

use std::path::PathBuf;

use hip_bridge::HipRuntime;
use hipfire_arch_deepseek4::harmonic::{
    HarmonicExpertResidencyPlan, HARMONIC_EXPERT_COUNT, HARMONIC_LAYER_COUNT, HARMONIC_TOP_K,
};
use rdna_compute::{DType, Gpu};

const ROUTES_PER_LAYER: usize = 256;

fn as_bytes(values: &[u32]) -> &[u8] {
    // SAFETY: every u32 bit pattern is valid and the returned slice cannot
    // outlive the borrowed input.
    unsafe {
        std::slice::from_raw_parts(
            values.as_ptr().cast::<u8>(),
            std::mem::size_of_val(values),
        )
    }
}

fn as_bytes_mut(values: &mut [u32]) -> &mut [u8] {
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

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut device = None;
    let mut hotset = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--device" => device = Some(args.next().ok_or("--device needs a selector")?),
            "--hotset-plan" => {
                hotset = Some(PathBuf::from(args.next().ok_or("--hotset-plan needs a path")?))
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    let selector = device.ok_or("--device is required")?;
    let manifest_path = hotset.ok_or("--hotset-plan is required")?;
    let manifest = std::fs::read_to_string(&manifest_path)
        .map_err(|error| format!("read {}: {error}", manifest_path.display()))?;
    let plan = HarmonicExpertResidencyPlan::from_manifest(&manifest)
        .map_err(|error| format!("parse {}: {error}", manifest_path.display()))?;
    let (device_id, pci, arch, name) = resolve_device(&selector)?;
    if !arch.eq_ignore_ascii_case("gfx1100") {
        return Err(format!("partition oracle requires gfx1100, got {arch}"));
    }
    let mut gpu = Gpu::init_with_device(device_id)
        .map_err(|error| format!("initialize gfx1100 at {pci}: {error}"))?;

    let compact = plan.compact_index_table();
    let compact_gpu = gpu
        .alloc_tensor(
            &[
                HARMONIC_LAYER_COUNT as usize,
                HARMONIC_EXPERT_COUNT as usize,
            ],
            DType::F32,
        )
        .map_err(|error| format!("allocate compact map: {error}"))?;
    gpu.hip
        .memcpy_htod(&compact_gpu.buf, as_bytes(&compact))
        .map_err(|error| format!("upload compact map: {error}"))?;
    let ids_gpu = gpu
        .alloc_tensor(&[HARMONIC_TOP_K], DType::F32)
        .map_err(|error| format!("allocate route IDs: {error}"))?;
    let local_gpu = gpu
        .alloc_tensor(&[HARMONIC_TOP_K], DType::F32)
        .map_err(|error| format!("allocate local IDs: {error}"))?;
    let sources_gpu = gpu
        .alloc_tensor(&[HARMONIC_TOP_K], DType::F32)
        .map_err(|error| format!("allocate slot sources: {error}"))?;
    let count_gpu = gpu
        .alloc_tensor(&[1], DType::F32)
        .map_err(|error| format!("allocate local count: {error}"))?;

    let mut comparisons = 0_u64;
    for layer in 0..HARMONIC_LAYER_COUNT as usize {
        for route_index in 0..ROUTES_PER_LAYER {
            let ids = std::array::from_fn(|slot| {
                ((route_index * 73 + slot * 41 + layer * 17) % HARMONIC_EXPERT_COUNT as usize)
                    as u32
            });
            gpu.hip
                .memcpy_htod(&ids_gpu.buf, as_bytes(&ids))
                .map_err(|error| format!("upload route l{layer} r{route_index}: {error}"))?;
            {
                let mut gfx1100 = gpu
                    .try_gfx1100()
                    .ok_or_else(|| "exact gfx1100 proof disappeared".to_owned())?;
                gfx1100
                    .harmonic_partition_route(
                        &ids_gpu,
                        &compact_gpu,
                        &local_gpu,
                        &sources_gpu,
                        &count_gpu,
                        layer,
                        HARMONIC_EXPERT_COUNT as usize,
                        HARMONIC_TOP_K,
                    )
                    .map_err(|error| format!("partition l{layer} r{route_index}: {error}"))?;
            }
            let mut local = [0_u32; HARMONIC_TOP_K];
            let mut sources = [0_u32; HARMONIC_TOP_K];
            let mut count = [0_u32; 1];
            gpu.hip
                .memcpy_dtoh(as_bytes_mut(&mut local), &local_gpu.buf)
                .map_err(|error| format!("download local IDs: {error}"))?;
            gpu.hip
                .memcpy_dtoh(as_bytes_mut(&mut sources), &sources_gpu.buf)
                .map_err(|error| format!("download slot sources: {error}"))?;
            gpu.hip
                .memcpy_dtoh(as_bytes_mut(&mut count), &count_gpu.buf)
                .map_err(|error| format!("download local count: {error}"))?;

            let mut expected = plan.split_result_layout(layer, ids).pack_route(ids);
            for index in 0..expected.local_count as usize {
                expected.local_expert_ids[index] = plan
                    .compact_expert_index(layer, expected.local_expert_ids[index])
                    .expect("local expert must have a compact index");
            }
            if count[0] != u32::from(expected.local_count)
                || local[..count[0] as usize]
                    != expected.local_expert_ids[..expected.local_count as usize]
                || sources != expected.slot_sources
            {
                return Err(format!(
                    "partition mismatch l{layer} r{route_index}: ids={ids:?} count={count:?}/{:?} local={local:?}/{:?} sources={sources:?}/{:?}",
                    expected.local_count,
                    expected.local_expert_ids,
                    expected.slot_sources
                ));
            }
            comparisons += 1 + count[0] as u64 + HARMONIC_TOP_K as u64;
        }
    }

    println!(
        "harmonic partition exact: selector={} pci={} arch={} name={:?} routes={} raw_bit_comparisons={}",
        selector,
        pci,
        arch,
        name,
        HARMONIC_LAYER_COUNT as usize * ROUTES_PER_LAYER,
        comparisons
    );
    for tensor in [compact_gpu, ids_gpu, local_gpu, sources_gpu, count_gpu] {
        gpu.free_tensor(tensor)
            .map_err(|error| format!("free oracle tensor: {error}"))?;
    }
    Ok(())
}
