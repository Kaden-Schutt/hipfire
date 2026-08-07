// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact-device micro-screen for the DS4 MQ2-Lloyd decode projections that a
//! harmonic gfx1100 owner would execute.
//!
//! Device selection is stable across host enumeration changes: a portable
//! selector resolves through live HIP/ROCr discovery to a PCI BDF, and `Gpu`
//! is then constructed from the BDF plus exact architecture proof. No ordinal
//! or host-specific BDF is compiled into this tool.

use std::ffi::c_void;
use std::time::Instant;

use hip_bridge::{HipResult, HipRuntime};
use rdna_compute::{DType, Gpu, GpuTensor};
use redline_rocr::{GpuSelector, Runtime};

const EXPERTS: usize = 256;
const TOP_K: usize = 6;
const GATE_M: usize = 4096;
const GATE_K: usize = 4096;
const DOWN_M: usize = 4096;
const DOWN_K: usize = 2048;
const WARMUP: usize = 8;
const TRIALS: usize = 40;

#[derive(Debug)]
struct Args {
    device: String,
    histogram: Option<Vec<u64>>,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut device = "arch:gfx1100".to_owned();
        let mut histogram = None;
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--device" => {
                    device = args.next().ok_or("--device requires a selector")?;
                }
                "--hot-count-histogram" => {
                    let raw = args
                        .next()
                        .ok_or("--hot-count-histogram requires 0:N,1:N,...")?;
                    histogram = Some(parse_histogram(&raw)?);
                }
                _ => {
                    return Err(format!(
                        "unknown argument {flag:?}; use --device arch:gfx1100|pci:BDF|name:TEXT|rocr:N [--hot-count-histogram 0:N,...]"
                    ));
                }
            }
        }
        Ok(Self { device, histogram })
    }
}

fn parse_histogram(raw: &str) -> Result<Vec<u64>, String> {
    let mut histogram = vec![0_u64; TOP_K + 1];
    for item in raw.split(',') {
        let (hot, count) = item
            .split_once(':')
            .ok_or_else(|| format!("bad histogram item {item:?}"))?;
        let hot = hot
            .parse::<usize>()
            .map_err(|error| format!("bad hot count {hot:?}: {error}"))?;
        let count = count
            .parse::<u64>()
            .map_err(|error| format!("bad occurrence count {count:?}: {error}"))?;
        if hot > TOP_K {
            return Err(format!("hot count {hot} exceeds DS4 top-k {TOP_K}"));
        }
        histogram[hot] = count;
    }
    if histogram.iter().sum::<u64>() == 0 {
        return Err("hot-count histogram is empty".to_owned());
    }
    Ok(histogram)
}

fn resolve_device(selector: &str) -> Result<(String, String, String), String> {
    let hip = HipRuntime::load().map_err(|error| format!("HIP discovery: {error:?}"))?;
    let (pci, source_name) = if let Some(expected) = selector.strip_prefix("arch:") {
        let count = hip
            .device_count()
            .map_err(|error| format!("HIP device count: {error:?}"))?;
        let mut matches = Vec::new();
        for device in 0..count {
            let arch = hip
                .get_arch(device)
                .map_err(|error| format!("HIP device {device} arch: {error:?}"))?;
            if arch.eq_ignore_ascii_case(expected) {
                matches.push(device);
            }
        }
        let [device] = matches.as_slice() else {
            return Err(format!(
                "selector {selector:?} matched {} HIP devices; exact architecture selectors must be unique",
                matches.len()
            ));
        };
        (
            hip.device_pci_bus_id(*device)
                .map_err(|error| format!("HIP PCI identity: {error:?}"))?,
            format!("hip-arch:{expected}"),
        )
    } else if let Some(pci) = selector.strip_prefix("pci:") {
        (pci.to_ascii_lowercase(), "explicit-pci".to_owned())
    } else if let Some(needle) = selector.strip_prefix("name:") {
        let count = hip
            .device_count()
            .map_err(|error| format!("HIP device count: {error:?}"))?;
        let needle_lower = needle.to_ascii_lowercase();
        let mut matches = Vec::new();
        for device in 0..count {
            let name = hip
                .device_name(device)
                .map_err(|error| format!("HIP device {device} name: {error:?}"))?;
            if name.to_ascii_lowercase().contains(&needle_lower) {
                matches.push((device, name));
            }
        }
        let [(device, name)] = matches.as_slice() else {
            return Err(format!(
                "selector {selector:?} matched {} HIP marketing names; name selectors must be unique",
                matches.len()
            ));
        };
        (
            hip.device_pci_bus_id(*device)
                .map_err(|error| format!("HIP PCI identity: {error:?}"))?,
            name.clone(),
        )
    } else if let Some(ordinal) = selector.strip_prefix("rocr:") {
        let ordinal = ordinal
            .parse::<usize>()
            .map_err(|error| format!("bad ROCr ordinal {ordinal:?}: {error}"))?;
        let runtime = Runtime::initialize(
            redline_rocr::load_symbols().map_err(|error| format!("ROCr load: {error}"))?,
        )
        .map_err(|error| format!("ROCr initialize: {error}"))?;
        let device = runtime
            .select_gpu(GpuSelector::Ordinal(ordinal))
            .map_err(|error| format!("ROCr ordinal {ordinal}: {error}"))?;
        (device.pci_bus_id().to_string(), device.name().to_owned())
    } else {
        return Err(format!(
            "device selector {selector:?} must start with arch:, pci:, name:, or rocr:"
        ));
    };
    let ordinal = hip
        .device_by_pci_bus_id(&pci)
        .map_err(|error| format!("HIP resolve PCI {pci}: {error:?}"))?;
    let roundtrip = hip
        .device_pci_bus_id(ordinal)
        .map_err(|error| format!("HIP PCI round trip: {error:?}"))?;
    if !roundtrip.eq_ignore_ascii_case(&pci) {
        return Err(format!("PCI round trip changed {pci} to {roundtrip}"));
    }
    let arch = hip
        .get_arch(ordinal)
        .map_err(|error| format!("HIP resolved arch: {error:?}"))?;
    Ok((roundtrip, arch, source_name))
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = (((bits >> 23) & 0xff) as i32) - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    if exponent <= 0 {
        return sign;
    }
    if exponent >= 31 {
        return sign | 0x7c00;
    }
    sign | ((exponent as u16) << 10) | ((mantissa >> 13) as u16)
}

fn mq2_weights(k: usize, rows: usize, experts: usize, seed: u64) -> (Vec<u8>, usize) {
    assert!(k.is_multiple_of(256));
    let expert_bytes = rows * (k / 256) * 72;
    let mut bytes = Vec::with_capacity(expert_bytes * experts);
    let mut rng = seed;
    for _ in 0..experts {
        for _ in 0..rows * (k / 256) {
            for value in [-3.0_f32, -1.0, 1.0, 3.0] {
                bytes.extend_from_slice(&f32_to_f16_bits(value).to_le_bytes());
            }
            for _ in 0..64 {
                let mut packed = 0_u8;
                for lane in 0..4 {
                    rng = rng
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    packed |= (((rng >> 48) & 3) as u8) << (lane * 2);
                }
                bytes.push(packed);
            }
        }
    }
    (bytes, expert_bytes)
}

fn tensor_alias(pointer: *mut c_void, bytes: usize, shape: Vec<usize>, dtype: DType) -> GpuTensor {
    GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(pointer, bytes) },
        shape,
        dtype,
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_down(
    gpu: &mut Gpu,
    candidate: bool,
    expert_ptrs: &GpuTensor,
    topk_indices: &GpuTensor,
    x: &GpuTensor,
    output: &GpuTensor,
    k_top: usize,
) -> HipResult<()> {
    if candidate {
        gpu.try_gfx1100()
            .expect("exact gfx1100 proof vanished")
            .mq2_lloyd_moe_down_expanded_lds_candidate(
                expert_ptrs,
                topk_indices,
                x,
                output,
                DOWN_M,
                DOWN_K,
                k_top,
                1,
            )
    } else {
        gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_expanded_k4(
            expert_ptrs,
            topk_indices,
            x,
            output,
            DOWN_M,
            DOWN_K,
            k_top,
            1,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn time_down(
    gpu: &mut Gpu,
    candidate: bool,
    expert_ptrs: &GpuTensor,
    topk_indices: &GpuTensor,
    x: &GpuTensor,
    output: &GpuTensor,
    k_top: usize,
) -> f64 {
    for _ in 0..WARMUP {
        dispatch_down(gpu, candidate, expert_ptrs, topk_indices, x, output, k_top).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let started = Instant::now();
    for _ in 0..TRIALS {
        dispatch_down(gpu, candidate, expert_ptrs, topk_indices, x, output, k_top).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    started.elapsed().as_secs_f64() * 1.0e6 / TRIALS as f64
}

#[allow(clippy::too_many_arguments)]
fn time_gate(
    gpu: &mut Gpu,
    expert_ptrs: &GpuTensor,
    topk_indices: &GpuTensor,
    x: &GpuTensor,
    gate: &GpuTensor,
    up: &GpuTensor,
    k_top: usize,
) -> f64 {
    for _ in 0..WARMUP {
        gpu.try_gfx1100()
            .expect("exact gfx1100 proof vanished")
            .mq2_lloyd_moe_gate_up_k4_lds(
                expert_ptrs,
                topk_indices,
                x,
                gate,
                up,
                GATE_M,
                GATE_K,
                k_top,
            )
            .unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let started = Instant::now();
    for _ in 0..TRIALS {
        gpu.try_gfx1100()
            .expect("exact gfx1100 proof vanished")
            .mq2_lloyd_moe_gate_up_k4_lds(
                expert_ptrs,
                topk_indices,
                x,
                gate,
                up,
                GATE_M,
                GATE_K,
                k_top,
            )
            .unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    started.elapsed().as_secs_f64() * 1.0e6 / TRIALS as f64
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    (values[0] + values[1]) * 0.5
}

fn main() -> Result<(), String> {
    let args = Args::parse()?;
    let (pci, arch, discovered_name) = resolve_device(&args.device)?;
    if arch != "gfx1100" {
        return Err(format!(
            "DS4 harmonic MQ2 micro requires exact gfx1100; selector {} resolved {arch} at {pci}",
            args.device
        ));
    }
    println!(
        "device_selector={} discovered_name={} resolved_pci={} resolved_arch={}",
        args.device, discovered_name, pci, arch
    );
    let mut gpu = Gpu::init_with_pci_bus_id(&pci, "gfx1100")
        .map_err(|error| format!("exact gfx1100 init: {error:?}"))?;
    if gpu.try_gfx1100().is_none() {
        return Err("exact gfx1100 proof unavailable after initialization".to_owned());
    }

    let (gate_weights, gate_expert_bytes) =
        mq2_weights(GATE_K, GATE_M, TOP_K, 0x7100_6a7e_cafe_f00d);
    let (down_weights, down_expert_bytes) =
        mq2_weights(DOWN_K, DOWN_M, TOP_K, 0x7100_d04e_cafe_f00d);
    let gate_weights_gpu = gpu
        .hip
        .malloc(gate_weights.len())
        .map_err(|e| format!("gate weights alloc: {e:?}"))?;
    let down_weights_gpu = gpu
        .hip
        .malloc(down_weights.len())
        .map_err(|e| format!("down weights alloc: {e:?}"))?;
    gpu.hip
        .memcpy_htod(&gate_weights_gpu, &gate_weights)
        .map_err(|e| format!("gate weights upload: {e:?}"))?;
    gpu.hip
        .memcpy_htod(&down_weights_gpu, &down_weights)
        .map_err(|e| format!("down weights upload: {e:?}"))?;

    let gate_base = gate_weights_gpu.as_ptr() as usize;
    let down_base = down_weights_gpu.as_ptr() as usize;
    let gate_ptr_bytes = (0..EXPERTS)
        .flat_map(|expert| {
            (gate_base + (expert % TOP_K) * gate_expert_bytes)
                .to_le_bytes()
                .into_iter()
        })
        .collect::<Vec<_>>();
    let down_ptr_bytes = (0..EXPERTS)
        .flat_map(|expert| {
            (down_base + (expert % TOP_K) * down_expert_bytes)
                .to_le_bytes()
                .into_iter()
        })
        .collect::<Vec<_>>();
    let gate_ptrs_gpu = gpu
        .hip
        .malloc(gate_ptr_bytes.len())
        .map_err(|e| format!("gate ptrs alloc: {e:?}"))?;
    let down_ptrs_gpu = gpu
        .hip
        .malloc(down_ptr_bytes.len())
        .map_err(|e| format!("down ptrs alloc: {e:?}"))?;
    gpu.hip
        .memcpy_htod(&gate_ptrs_gpu, &gate_ptr_bytes)
        .map_err(|e| format!("gate ptrs upload: {e:?}"))?;
    gpu.hip
        .memcpy_htod(&down_ptrs_gpu, &down_ptr_bytes)
        .map_err(|e| format!("down ptrs upload: {e:?}"))?;

    let index_bytes = (0..TOP_K)
        .flat_map(|expert| (expert as i32).to_le_bytes())
        .collect::<Vec<_>>();
    let indices_gpu = gpu
        .hip
        .malloc(index_bytes.len())
        .map_err(|e| format!("indices alloc: {e:?}"))?;
    gpu.hip
        .memcpy_htod(&indices_gpu, &index_bytes)
        .map_err(|e| format!("indices upload: {e:?}"))?;

    let gate_x_bytes = (0..GATE_K)
        .flat_map(|index| (((index * 17) % 31) as f32 / 15.0 - 1.0).to_le_bytes())
        .collect::<Vec<_>>();
    let down_x_bytes = (0..TOP_K * DOWN_K)
        .flat_map(|index| (((index * 19) % 37) as f32 / 18.0 - 1.0).to_le_bytes())
        .collect::<Vec<_>>();
    let gate_x_gpu = gpu
        .hip
        .malloc(gate_x_bytes.len())
        .map_err(|e| format!("gate x alloc: {e:?}"))?;
    let down_x_gpu = gpu
        .hip
        .malloc(down_x_bytes.len())
        .map_err(|e| format!("down x alloc: {e:?}"))?;
    gpu.hip
        .memcpy_htod(&gate_x_gpu, &gate_x_bytes)
        .map_err(|e| format!("gate x upload: {e:?}"))?;
    gpu.hip
        .memcpy_htod(&down_x_gpu, &down_x_bytes)
        .map_err(|e| format!("down x upload: {e:?}"))?;

    let gate_output_bytes = TOP_K * (GATE_M / 2) * 4;
    let down_output_bytes = TOP_K * DOWN_M * 4;
    let gate_gpu = gpu
        .hip
        .malloc(gate_output_bytes)
        .map_err(|e| format!("gate output alloc: {e:?}"))?;
    let up_gpu = gpu
        .hip
        .malloc(gate_output_bytes)
        .map_err(|e| format!("up output alloc: {e:?}"))?;
    let down_base_gpu = gpu
        .hip
        .malloc(down_output_bytes)
        .map_err(|e| format!("down baseline alloc: {e:?}"))?;
    let down_candidate_gpu = gpu
        .hip
        .malloc(down_output_bytes)
        .map_err(|e| format!("down candidate alloc: {e:?}"))?;

    let gate_ptrs = tensor_alias(
        gate_ptrs_gpu.as_ptr(),
        gate_ptr_bytes.len(),
        vec![EXPERTS],
        DType::Raw,
    );
    let down_ptrs = tensor_alias(
        down_ptrs_gpu.as_ptr(),
        down_ptr_bytes.len(),
        vec![EXPERTS],
        DType::Raw,
    );
    let indices = tensor_alias(
        indices_gpu.as_ptr(),
        index_bytes.len(),
        vec![TOP_K],
        DType::Raw,
    );
    let gate_x = tensor_alias(
        gate_x_gpu.as_ptr(),
        gate_x_bytes.len(),
        vec![GATE_K],
        DType::F32,
    );
    let down_x = tensor_alias(
        down_x_gpu.as_ptr(),
        down_x_bytes.len(),
        vec![TOP_K, DOWN_K],
        DType::F32,
    );
    let gate = tensor_alias(
        gate_gpu.as_ptr(),
        gate_output_bytes,
        vec![TOP_K, GATE_M / 2],
        DType::F32,
    );
    let up = tensor_alias(
        up_gpu.as_ptr(),
        gate_output_bytes,
        vec![TOP_K, GATE_M / 2],
        DType::F32,
    );
    let down_base = tensor_alias(
        down_base_gpu.as_ptr(),
        down_output_bytes,
        vec![TOP_K, DOWN_M],
        DType::F32,
    );
    let down_candidate = tensor_alias(
        down_candidate_gpu.as_ptr(),
        down_output_bytes,
        vec![TOP_K, DOWN_M],
        DType::F32,
    );

    let mut gate_us = [0.0_f64; TOP_K + 1];
    let mut down_base_us = [0.0_f64; TOP_K + 1];
    let mut down_candidate_us = [0.0_f64; TOP_K + 1];
    for k_top in 1..=TOP_K {
        dispatch_down(
            &mut gpu, false, &down_ptrs, &indices, &down_x, &down_base, k_top,
        )
        .map_err(|e| format!("down baseline dispatch: {e:?}"))?;
        dispatch_down(
            &mut gpu,
            true,
            &down_ptrs,
            &indices,
            &down_x,
            &down_candidate,
            k_top,
        )
        .map_err(|e| format!("down candidate dispatch: {e:?}"))?;
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("correctness sync: {e:?}"))?;
        let bytes = k_top * DOWN_M * 4;
        let mut baseline = vec![0_u8; bytes];
        let mut candidate = vec![0_u8; bytes];
        gpu.hip
            .memcpy_dtoh(&mut baseline, &down_base_gpu)
            .map_err(|e| format!("baseline download: {e:?}"))?;
        gpu.hip
            .memcpy_dtoh(&mut candidate, &down_candidate_gpu)
            .map_err(|e| format!("candidate download: {e:?}"))?;
        let mismatches = baseline
            .chunks_exact(4)
            .zip(candidate.chunks_exact(4))
            .filter(|(left, right)| left != right)
            .count();
        if mismatches != 0 {
            return Err(format!("k_top={k_top} raw-bit mismatches={mismatches}"));
        }

        gate_us[k_top] = time_gate(&mut gpu, &gate_ptrs, &indices, &gate_x, &gate, &up, k_top);
        let mut base_rows = Vec::with_capacity(2);
        let mut candidate_rows = Vec::with_capacity(2);
        for candidate_arm in [false, true, true, false] {
            let us = time_down(
                &mut gpu,
                candidate_arm,
                &down_ptrs,
                &indices,
                &down_x,
                if candidate_arm {
                    &down_candidate
                } else {
                    &down_base
                },
                k_top,
            );
            if candidate_arm {
                candidate_rows.push(us);
            } else {
                base_rows.push(us);
            }
        }
        down_base_us[k_top] = median(&mut base_rows);
        down_candidate_us[k_top] = median(&mut candidate_rows);
        println!(
            "shape k_top={k_top} gate_us={:.3} down_baseline_us={:.3} down_candidate_us={:.3} down_speedup={:.6} bit_mismatch=0",
            gate_us[k_top],
            down_base_us[k_top],
            down_candidate_us[k_top],
            down_base_us[k_top] / down_candidate_us[k_top]
        );
    }

    if let Some(histogram) = args.histogram {
        let records = histogram.iter().sum::<u64>() as f64;
        let weighted = |times: &[f64; TOP_K + 1]| {
            histogram
                .iter()
                .zip(times)
                .map(|(count, time)| *count as f64 * time)
                .sum::<f64>()
                / records
        };
        let weighted_gate = weighted(&gate_us);
        let weighted_down_base = weighted(&down_base_us);
        let weighted_down_candidate = weighted(&down_candidate_us);
        println!(
            "weighted records={} gate_us={weighted_gate:.3} down_baseline_us={weighted_down_base:.3} down_candidate_us={weighted_down_candidate:.3} down_speedup={:.6} projection_speedup={:.6}",
            records as u64,
            weighted_down_base / weighted_down_candidate,
            (weighted_gate + weighted_down_base) / (weighted_gate + weighted_down_candidate)
        );
    }

    // These tensors alias the owning DeviceBuffers above.
    for tensor in [
        gate_ptrs,
        down_ptrs,
        indices,
        gate_x,
        down_x,
        gate,
        up,
        down_base,
        down_candidate,
    ] {
        std::mem::forget(tensor);
    }
    Ok(())
}
