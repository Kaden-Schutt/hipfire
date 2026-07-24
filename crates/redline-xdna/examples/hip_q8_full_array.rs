// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Verifies the batch-256, K=N=2048 native-Q8 XDNA2 full-array artifact.
//!
//! Usage:
//! `HIPFIRE_XDNA_DANGEROUS_INTEROP=1 hip_q8_full_array \
//!  <main.pdi> <instructions.bin>`

use hsa_bridge::HsaRuntime;
use redline_xdna::{
    resolve_device_path, ArtifactBundle, ArtifactFile, ArtifactManifest, Binding, BindingAccess,
    BindingLayout, Device, FirmwareCompatibility, IoLayout, ProjectionArithmetic, ProjectionShape,
    SUPPORTED_ABI_VERSION, SUPPORTED_MANIFEST_VERSION,
};
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::time::Duration;

const M: usize = 256;
const K: usize = 2048;
const N: usize = 2048;
const Q8_BLOCK_ELEMENTS: usize = 32;
const Q8_BLOCK_BYTES: usize = 34;
const PACKED_ROW_BYTES: usize = K / Q8_BLOCK_ELEMENTS * Q8_BLOCK_BYTES;
const ACTIVATION_BYTES: usize = M * K * 2;
const WEIGHT_BYTES: usize = N * PACKED_ROW_BYTES;
const OUTPUT_BYTES: usize = M * N * 4;
const OPERATIONS: u64 = 2 * M as u64 * K as u64 * N as u64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("HIPFIRE_XDNA_DANGEROUS_INTEROP").as_deref()
        != Some(std::ffi::OsStr::new("1"))
    {
        return Err(
            "refusing live cross-driver dispatch; set HIPFIRE_XDNA_DANGEROUS_INTEROP=1".into(),
        );
    }
    let mut arguments = std::env::args_os().skip(1);
    let pdi_path = PathBuf::from(arguments.next().ok_or("missing PDI path")?);
    let instructions_path = PathBuf::from(arguments.next().ok_or("missing instruction path")?);
    let pdi = std::fs::read(&pdi_path)?;
    let instructions = std::fs::read(&instructions_path)?;
    if instructions.len() % 4 != 0 {
        return Err("instruction stream length is not a multiple of four".into());
    }

    let fixture = Fixture::new();
    let hip = hip_bridge::HipRuntime::load()?;
    let hip_device = select_gfx1151(&hip)?;
    hip.set_device(hip_device)?;
    let activation = hip.malloc(ACTIVATION_BYTES)?;
    let weight = hip.malloc(WEIGHT_BYTES)?;
    let output = hip.malloc(OUTPUT_BYTES)?;
    hip.memcpy_htod(&activation, &fixture.activation)?;
    hip.memcpy_htod(&weight, &fixture.weight)?;
    hip.memset(&output, 0xa5, output.size())?;
    hip.device_synchronize()?;

    let hsa = HsaRuntime::load()?;
    let activation_export =
        unsafe { hsa.export_dmabuf(activation.as_ptr().cast_const(), activation.size())? };
    let weight_export = unsafe { hsa.export_dmabuf(weight.as_ptr().cast_const(), weight.size())? };
    let output_export = unsafe { hsa.export_dmabuf(output.as_ptr().cast_const(), output.size())? };

    let configured_device = std::env::var_os("HIPFIRE_XDNA_DEVICE").map(PathBuf::from);
    let device_path = resolve_device_path(configured_device.as_deref())?;
    let device = Device::open(&device_path)?;
    let firmware = device.metadata().firmware;
    let bundle = ArtifactBundle {
        manifest_path: "<live-q8-full-array-probe>".into(),
        manifest: ArtifactManifest {
            manifest_version: SUPPORTED_MANIFEST_VERSION,
            abi_version: SUPPORTED_ABI_VERSION,
            artifact_id: "diagnostic-q8-full-array-m256-k2048-n2048".into(),
            device: "gfx1151".into(),
            firmware: FirmwareCompatibility {
                minimum: firmware,
                maximum: firmware,
            },
            arithmetic: ProjectionArithmetic::Q8W8A16FullArrayDiagnostic,
            layout: IoLayout {
                activation: "bf16".into(),
                weight: "q8_0".into(),
                accumulator: "f32".into(),
                output: "f32".into(),
                q8_block_elements: Q8_BLOCK_ELEMENTS as u32,
                q8_block_bytes: Q8_BLOCK_BYTES as u32,
            },
            shapes: vec![ProjectionShape {
                k: K as u32,
                n: N as u32,
                max_batch: M as u32,
                masked_batch_tail: false,
                masked_output_tail: false,
            }],
            bindings: vec![
                BindingLayout {
                    name: "activation".into(),
                    access: BindingAccess::Read,
                    minimum_bytes: activation.size() as u64,
                    alignment: 64,
                },
                BindingLayout {
                    name: "weight".into(),
                    access: BindingAccess::Read,
                    minimum_bytes: weight.size() as u64,
                    alignment: 64,
                },
                BindingLayout {
                    name: "output".into(),
                    access: BindingAccess::Write,
                    minimum_bytes: output.size() as u64,
                    alignment: 64,
                },
            ],
            instruction_count: (instructions.len() / 4) as u32,
            pdi: ArtifactFile {
                path: pdi_path,
                sha256: String::new(),
            },
            instructions: ArtifactFile {
                path: instructions_path,
                sha256: String::new(),
            },
        },
        pdi,
        instructions,
    };

    let context = device.create_context(2048)?;
    let program = context.load_program(&bundle)?;
    let ring = context.command_ring(4)?;
    let activation_bo = device.import_dmabuf(
        activation_export.as_raw_fd(),
        activation_export.offset(),
        activation.size(),
    )?;
    let weight_bo = device.import_dmabuf(
        weight_export.as_raw_fd(),
        weight_export.offset(),
        weight.size(),
    )?;
    let output_bo = device.import_dmabuf(
        output_export.as_raw_fd(),
        output_export.offset(),
        output.size(),
    )?;
    for bo in [&activation_bo, &weight_bo, &output_bo] {
        bo.sync_to_device(0, bo.len())?;
    }
    let bindings = [
        Binding::whole(&activation_bo).with_access(BindingAccess::Read),
        Binding::whole(&weight_bo).with_access(BindingAccess::Read),
        Binding::whole(&output_bo).with_access(BindingAccess::Write),
    ];

    let iterations = bounded_env_usize("HIPFIRE_XDNA_ITERATIONS", 5, 1, 1_000)?;
    let mut timings_ns = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let timing = ring
            .submit(&program, &bindings)?
            .wait(Duration::from_secs(1))?;
        timings_ns.push(timing.elapsed.as_nanos());
    }

    let mut mapped_bytes = vec![0_u8; OUTPUT_BYTES];
    output_bo.read_mapped(0, &mut mapped_bytes)?;
    let mapped = decode_f32(&mapped_bytes);
    let mapped_metrics = verify("mapped dma-buf", &mapped, &fixture.expected)?;

    let mut hip_bytes = vec![0_u8; OUTPUT_BYTES];
    hip.memcpy_dtoh(&mut hip_bytes, &output)?;
    let hip_output = decode_f32(&hip_bytes);
    let hip_metrics = verify("HIP-visible", &hip_output, &fixture.expected)?;

    timings_ns.sort_unstable();
    let median_ns = timings_ns[timings_ns.len() / 2];
    let p99_index = ((timings_ns.len() * 99).div_ceil(100)).saturating_sub(1);
    let p99_ns = timings_ns[p99_index];
    let median_us = median_ns as f64 / 1_000.0;
    let p99_us = p99_ns as f64 / 1_000.0;
    let effective_tops = OPERATIONS as f64 / (median_ns as f64) / 1_000.0;
    let gate_pass = median_us <= 340.0 && p99_us <= 400.0;
    println!(
        "PASS hip_device={} device={} \
         arithmetic=q8_w8a16_full_array_diagnostic m={} k={} n={} \
         iterations={} weight_imports=1 commands_per_projection=1 \
         median_us={:.3} p99_us={:.3} effective_tops={:.3} \
         max_abs={:.6} cosine={:.9} nrmse={:.9} complete_panel_gate={}",
        hip_device,
        device_path.display(),
        M,
        K,
        N,
        iterations,
        median_us,
        p99_us,
        effective_tops,
        mapped_metrics.max_abs.max(hip_metrics.max_abs),
        mapped_metrics.cosine.min(hip_metrics.cosine),
        mapped_metrics.nrmse.max(hip_metrics.nrmse),
        gate_pass,
    );

    drop((activation_bo, weight_bo, output_bo));
    drop((activation_export, weight_export, output_export));
    hip.free(activation)?;
    hip.free(weight)?;
    hip.free(output)?;
    Ok(())
}

struct Fixture {
    activation: Vec<u8>,
    weight: Vec<u8>,
    expected: Vec<f32>,
}

impl Fixture {
    fn new() -> Self {
        let row_factors = [0.5_f32, 1.0, -1.0, 2.0];
        let reduction_factors = [-2.0_f32, -1.0, 0.0, 1.0, 2.0];
        let mut activation = Vec::with_capacity(ACTIVATION_BYTES);
        for row in 0..M {
            let row_factor = row_factors[row % row_factors.len()];
            for reduction in 0..K {
                let value = row_factor * reduction_factors[reduction % reduction_factors.len()];
                activation.extend_from_slice(&f32_to_bf16(value).to_le_bytes());
            }
        }

        let scale_bits = [0x3000_u16, 0xb400, 0x3800, 0x2c00];
        let mut weight = vec![0_u8; WEIGHT_BYTES];
        let mut base_dot = vec![0_f32; N];
        for output in 0..N {
            for block in 0..K / Q8_BLOCK_ELEMENTS {
                let bits = scale_bits[(output + block) % scale_bits.len()];
                let scale = fp16_to_f32(bits);
                let offset = output * PACKED_ROW_BYTES + block * Q8_BLOCK_BYTES;
                weight[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
                for lane in 0..Q8_BLOCK_ELEMENTS {
                    let reduction = block * Q8_BLOCK_ELEMENTS + lane;
                    let quantized = (((output * 17 + block * 7 + lane * 3) % 15) as i16 - 7) as i8;
                    weight[offset + 2 + lane] = quantized as u8;
                    base_dot[output] += reduction_factors[reduction % reduction_factors.len()]
                        * scale
                        * f32::from(quantized);
                }
            }
        }

        let mut expected = vec![0_f32; M * N];
        for row in 0..M {
            let row_factor = row_factors[row % row_factors.len()];
            for output in 0..N {
                expected[row * N + output] = row_factor * base_dot[output];
            }
        }
        Self {
            activation,
            weight,
            expected,
        }
    }
}

#[derive(Clone, Copy)]
struct Metrics {
    max_abs: f64,
    cosine: f64,
    nrmse: f64,
}

fn verify(
    route: &str,
    actual: &[f32],
    expected: &[f32],
) -> Result<Metrics, Box<dyn std::error::Error>> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{route} output length {} != expected {}",
            actual.len(),
            expected.len()
        )
        .into());
    }
    let mut max_abs = 0_f64;
    let mut squared_error = 0_f64;
    let mut squared_reference = 0_f64;
    let mut squared_actual = 0_f64;
    let mut dot = 0_f64;
    let mut mismatches = Vec::new();
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if !actual.is_finite() {
            return Err(format!("{route} produced non-finite output at {index}: {actual}").into());
        }
        let error = f64::from(actual) - f64::from(expected);
        max_abs = max_abs.max(error.abs());
        squared_error += error * error;
        squared_reference += f64::from(expected) * f64::from(expected);
        squared_actual += f64::from(actual) * f64::from(actual);
        dot += f64::from(actual) * f64::from(expected);
        let tolerance = 0.05 + f64::from(expected).abs() * 0.000_01;
        if error.abs() > tolerance && mismatches.len() < 8 {
            mismatches.push((index, actual, expected, error));
        }
    }
    let cosine = dot / (squared_actual.sqrt() * squared_reference.sqrt());
    let nrmse = (squared_error / squared_reference).sqrt();
    if !mismatches.is_empty() || cosine < 0.9995 || nrmse > 0.005 {
        return Err(format!(
            "{route} full-array parity failed: mismatches={mismatches:?} \
             cosine={cosine} nrmse={nrmse}"
        )
        .into());
    }
    Ok(Metrics {
        max_abs,
        cosine,
        nrmse,
    })
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
        .collect()
}

fn fp16_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let mut exponent = u32::from((bits >> 10) & 0x1f);
    let mut fraction = u32::from(bits & 0x03ff);
    let f32_bits = if exponent == 0 {
        if fraction == 0 {
            sign
        } else {
            let mut shift = 0;
            while fraction & 0x0400 == 0 {
                fraction <<= 1;
                shift += 1;
            }
            fraction &= 0x03ff;
            sign | ((113 - shift) << 23) | (fraction << 13)
        }
    } else if exponent == 0x1f {
        sign | 0x7f80_0000 | (fraction << 13)
    } else {
        exponent += 112;
        sign | (exponent << 23) | (fraction << 13)
    };
    f32::from_bits(f32_bits)
}

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    bits.wrapping_add(rounding_bias).wrapping_shr(16) as u16
}

fn bounded_env_usize(
    name: &str,
    default: usize,
    minimum: usize,
    maximum: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    let value = match std::env::var(name) {
        Ok(value) => value.parse()?,
        Err(std::env::VarError::NotPresent) => default,
        Err(error) => return Err(error.into()),
    };
    if !(minimum..=maximum).contains(&value) {
        return Err(format!("{name} must be in {minimum}..={maximum}, got {value}").into());
    }
    Ok(value)
}

fn select_gfx1151(hip: &hip_bridge::HipRuntime) -> Result<i32, Box<dyn std::error::Error>> {
    if let Some(value) = std::env::var_os("HIPFIRE_XDNA_HIP_DEVICE") {
        let device: i32 = value
            .to_str()
            .ok_or("HIPFIRE_XDNA_HIP_DEVICE is not valid UTF-8")?
            .parse()?;
        let arch = hip.get_arch(device)?;
        if arch != "gfx1151" {
            return Err(format!(
                "refusing HIP device {device}: architecture is {arch}, expected gfx1151"
            )
            .into());
        }
        return Ok(device);
    }

    let mut matches = Vec::new();
    for device in 0..hip.device_count()? {
        if hip.get_arch(device)? == "gfx1151" {
            matches.push(device);
        }
    }
    match matches.as_slice() {
        [device] => Ok(*device),
        [] => Err("no gfx1151 HIP device is visible".into()),
        _ => Err(format!(
            "multiple gfx1151 HIP devices are visible ({matches:?}); set HIPFIRE_XDNA_HIP_DEVICE"
        )
        .into()),
    }
}
