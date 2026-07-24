// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! End-to-end verifier for the retained Q8_0 microtile diagnostic artifact.
//!
//! Usage:
//! `HIPFIRE_XDNA_DANGEROUS_INTEROP=1 hip_q8_persistent_microtile \
//!  <2|8|16> <main.pdi> <instructions.bin>`

use hsa_bridge::HsaRuntime;
use redline_xdna::{
    resolve_device_path, ArtifactBundle, ArtifactFile, ArtifactManifest, Binding, BindingAccess,
    BindingLayout, Device, FirmwareCompatibility, IoLayout, ProjectionArithmetic, ProjectionShape,
    SUPPORTED_ABI_VERSION, SUPPORTED_MANIFEST_VERSION,
};
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::time::Duration;

const ROWS_PER_CHUNK: usize = 8;
const K: usize = 64;
const OUTPUTS: usize = 16;
const Q8_BLOCK_ELEMENTS: usize = 32;
const Q8_BLOCK_BYTES: usize = 34;
const A_TILE_ROWS: usize = 4;
const A_TILE_K: usize = 8;
const C_TILE_OUTPUTS: usize = 8;
const PACKED_BYTES: usize = OUTPUTS * (K / Q8_BLOCK_ELEMENTS) * Q8_BLOCK_BYTES;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("HIPFIRE_XDNA_DANGEROUS_INTEROP").as_deref()
        != Some(std::ffi::OsStr::new("1"))
    {
        return Err(
            "refusing live cross-driver dispatch; set HIPFIRE_XDNA_DANGEROUS_INTEROP=1".into(),
        );
    }
    let mut arguments = std::env::args_os().skip(1);
    let chunks: usize = arguments
        .next()
        .ok_or("missing chunk count")?
        .to_str()
        .ok_or("chunk count is not valid UTF-8")?
        .parse()?;
    if ![2, 8, 16].contains(&chunks) {
        return Err(format!("chunk count must be 2, 8, or 16, got {chunks}").into());
    }
    let pdi_path = PathBuf::from(arguments.next().ok_or("missing PDI path")?);
    let instructions_path = PathBuf::from(arguments.next().ok_or("missing instruction path")?);
    let pdi = std::fs::read(&pdi_path)?;
    let instructions = std::fs::read(&instructions_path)?;

    let fixture = Fixture::new(chunks);
    let hip = hip_bridge::HipRuntime::load()?;
    let hip_device = select_gfx1151(&hip)?;
    hip.set_device(hip_device)?;
    let activation = hip.malloc(fixture.activation_bytes.len())?;
    let packed = hip.malloc(fixture.packed_bytes.len())?;
    let output = hip.malloc(fixture.expected_tiled.len() * 4)?;
    hip.memcpy_htod(&activation, &fixture.activation_bytes)?;
    hip.memcpy_htod(&packed, &fixture.packed_bytes)?;
    hip.memset(&output, 0xa5, output.size())?;
    hip.device_synchronize()?;

    let hsa = HsaRuntime::load()?;
    let activation_export =
        unsafe { hsa.export_dmabuf(activation.as_ptr().cast_const(), activation.size())? };
    let packed_export = unsafe { hsa.export_dmabuf(packed.as_ptr().cast_const(), packed.size())? };
    let output_export = unsafe { hsa.export_dmabuf(output.as_ptr().cast_const(), output.size())? };

    let configured_device = std::env::var_os("HIPFIRE_XDNA_DEVICE").map(PathBuf::from);
    let device_path = resolve_device_path(configured_device.as_deref())?;
    let device = Device::open(&device_path)?;
    let firmware = device.metadata().firmware;
    let bundle = ArtifactBundle {
        manifest_path: "<live-q8-persistent-microtile-probe>".into(),
        manifest: ArtifactManifest {
            manifest_version: SUPPORTED_MANIFEST_VERSION,
            abi_version: SUPPORTED_ABI_VERSION,
            artifact_id: format!("diagnostic-q8-persistent-microtile-c{chunks}"),
            device: "gfx1151".into(),
            firmware: FirmwareCompatibility {
                minimum: firmware,
                maximum: firmware,
            },
            arithmetic: ProjectionArithmetic::Q8W8A16MicrotileDiagnostic,
            layout: IoLayout {
                activation: "bf16_aie_tile_4x8".into(),
                weight: "q8_0".into(),
                accumulator: "f32".into(),
                output: "f32_aie_tile_4x8".into(),
                q8_block_elements: Q8_BLOCK_ELEMENTS as u32,
                q8_block_bytes: Q8_BLOCK_BYTES as u32,
            },
            shapes: vec![ProjectionShape {
                k: K as u32,
                n: OUTPUTS as u32,
                max_batch: (chunks * ROWS_PER_CHUNK) as u32,
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
                    name: "packed_q8".into(),
                    access: BindingAccess::Read,
                    minimum_bytes: packed.size() as u64,
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
    let packed_bo = device.import_dmabuf(
        packed_export.as_raw_fd(),
        packed_export.offset(),
        packed.size(),
    )?;
    let output_bo = device.import_dmabuf(
        output_export.as_raw_fd(),
        output_export.offset(),
        output.size(),
    )?;
    for bo in [&activation_bo, &packed_bo, &output_bo] {
        bo.sync_to_device(0, bo.len())?;
    }
    let bindings = [
        Binding::whole(&activation_bo).with_access(BindingAccess::Read),
        Binding::whole(&packed_bo).with_access(BindingAccess::Read),
        Binding::whole(&output_bo).with_access(BindingAccess::Write),
    ];
    let iterations = bounded_env_usize("HIPFIRE_XDNA_ITERATIONS", 30, 1, 1_000)?;
    let mut timings_ns = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let timing = ring
            .submit(&program, &bindings)?
            .wait(Duration::from_secs(1))?;
        timings_ns.push(timing.elapsed.as_nanos());
    }

    let mut mapped_bytes = vec![0_u8; output.size()];
    output_bo.read_mapped(0, &mut mapped_bytes)?;
    let mapped = decode_f32(&mapped_bytes);
    let mapped_metrics = verify("mapped dma-buf", &mapped, &fixture.expected_tiled)?;

    let mut hip_bytes = vec![0_u8; output.size()];
    hip.memcpy_dtoh(&mut hip_bytes, &output)?;
    let hip_output = decode_f32(&hip_bytes);
    let hip_metrics = verify("HIP-visible", &hip_output, &fixture.expected_tiled)?;

    timings_ns.sort_unstable();
    let median_ns = timings_ns[timings_ns.len() / 2];
    let p99_index = ((timings_ns.len() * 99).div_ceil(100)).saturating_sub(1);
    let p99_ns = timings_ns[p99_index];
    println!(
        "PASS hip_device={} device={} arithmetic=q8_w8a16_microtile_diagnostic \
         chunks={} rows={} k={} outputs={} iterations={} b_imports=1 \
         b_decodes_per_submission=1 median_us={:.3} p99_us={:.3} \
         amortized_median_us_per_chunk={:.3} max_abs={:.6} cosine={:.9} nrmse={:.9}",
        hip_device,
        device_path.display(),
        chunks,
        chunks * ROWS_PER_CHUNK,
        K,
        OUTPUTS,
        iterations,
        median_ns as f64 / 1_000.0,
        p99_ns as f64 / 1_000.0,
        median_ns as f64 / 1_000.0 / chunks as f64,
        mapped_metrics.max_abs.max(hip_metrics.max_abs),
        mapped_metrics.cosine.min(hip_metrics.cosine),
        mapped_metrics.nrmse.max(hip_metrics.nrmse),
    );

    drop((activation_bo, packed_bo, output_bo));
    drop((activation_export, packed_export, output_export));
    hip.free(activation)?;
    hip.free(packed)?;
    hip.free(output)?;
    Ok(())
}

struct Fixture {
    activation_bytes: Vec<u8>,
    packed_bytes: Vec<u8>,
    expected_tiled: Vec<f32>,
}

impl Fixture {
    fn new(chunks: usize) -> Self {
        let rows = chunks * ROWS_PER_CHUNK;
        let mut activation_row_major = vec![0_f32; rows * K];
        for row in 0..rows {
            for reduction in 0..K {
                activation_row_major[row * K + reduction] =
                    ((row * 11 + reduction * 5) % 7) as f32 - 3.0;
            }
        }

        let mut activation_bytes = Vec::with_capacity(rows * K * 2);
        for chunk in 0..chunks {
            for row_tile in 0..ROWS_PER_CHUNK / A_TILE_ROWS {
                for reduction_tile in 0..K / A_TILE_K {
                    for row_lane in 0..A_TILE_ROWS {
                        for reduction_lane in 0..A_TILE_K {
                            let row = chunk * ROWS_PER_CHUNK + row_tile * A_TILE_ROWS + row_lane;
                            let reduction = reduction_tile * A_TILE_K + reduction_lane;
                            activation_bytes.extend_from_slice(
                                &f32_to_bf16(activation_row_major[row * K + reduction])
                                    .to_le_bytes(),
                            );
                        }
                    }
                }
            }
        }

        let scale_bits = [0x3000_u16, 0xb400, 0x3800, 0x2c00];
        let mut packed_bytes = vec![0_u8; PACKED_BYTES];
        let mut weight_row_major = vec![0_f32; OUTPUTS * K];
        for output in 0..OUTPUTS {
            for block in 0..K / Q8_BLOCK_ELEMENTS {
                let bits = scale_bits[(output + block) % scale_bits.len()];
                let scale = fp16_to_f32(bits);
                let offset = (output * (K / Q8_BLOCK_ELEMENTS) + block) * Q8_BLOCK_BYTES;
                packed_bytes[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
                for lane in 0..Q8_BLOCK_ELEMENTS {
                    let quantized = (((output * 17 + block * 7 + lane * 3) % 15) as i16 - 7) as i8;
                    packed_bytes[offset + 2 + lane] = quantized as u8;
                    weight_row_major[output * K + block * Q8_BLOCK_ELEMENTS + lane] =
                        scale * f32::from(quantized);
                }
            }
        }

        let mut expected_row_major = vec![0_f32; rows * OUTPUTS];
        for row in 0..rows {
            for output in 0..OUTPUTS {
                let mut sum = 0_f32;
                for reduction in 0..K {
                    sum += activation_row_major[row * K + reduction]
                        * weight_row_major[output * K + reduction];
                }
                expected_row_major[row * OUTPUTS + output] = sum;
            }
        }

        let mut expected_tiled = Vec::with_capacity(rows * OUTPUTS);
        for chunk in 0..chunks {
            for row_tile in 0..ROWS_PER_CHUNK / A_TILE_ROWS {
                for output_tile in 0..OUTPUTS / C_TILE_OUTPUTS {
                    for row_lane in 0..A_TILE_ROWS {
                        for output_lane in 0..C_TILE_OUTPUTS {
                            let row = chunk * ROWS_PER_CHUNK + row_tile * A_TILE_ROWS + row_lane;
                            let output = output_tile * C_TILE_OUTPUTS + output_lane;
                            expected_tiled.push(expected_row_major[row * OUTPUTS + output]);
                        }
                    }
                }
            }
        }

        Self {
            activation_bytes,
            packed_bytes,
            expected_tiled,
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
    let mut dot = 0_f64;
    let mut squared_actual = 0_f64;
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
        if error.abs() > 0.001 && mismatches.len() < 8 {
            mismatches.push((index, actual, expected));
        }
    }
    if !mismatches.is_empty() {
        return Err(format!("{route} W8A16 mismatches: {mismatches:?}").into());
    }
    let cosine = dot / (squared_actual.sqrt() * squared_reference.sqrt());
    let nrmse = (squared_error / squared_reference).sqrt();
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
