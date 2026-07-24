// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Verifies the XDNA2 Q8_0 decoder against GPU-owned Hipfire-layout buffers.
//!
//! This is a diagnostic proof, not a production projection route. It exercises
//! HIP allocation -> HSA dma-buf export -> amdxdna import and reads the
//! XDNA-written BF16 output through both the mapped dma-buf and HIP.
//!
//! Usage:
//! `HIPFIRE_XDNA_DANGEROUS_INTEROP=1 hip_q8_decode <main.pdi> <instructions.bin>`

use hsa_bridge::HsaRuntime;
use redline_xdna::{
    resolve_device_path, ArtifactBundle, ArtifactFile, ArtifactManifest, Binding, BindingAccess,
    BindingLayout, Device, FirmwareCompatibility, IoLayout, ProjectionArithmetic, ProjectionShape,
    SUPPORTED_ABI_VERSION, SUPPORTED_MANIFEST_VERSION,
};
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::time::Duration;

const K: usize = 2048;
const BLOCK_ELEMENTS: usize = 32;
const BLOCK_BYTES: usize = 34;
const BLOCKS: usize = K / BLOCK_ELEMENTS;
const PACKED_BYTES: usize = BLOCKS * BLOCK_BYTES;
const OUTPUT_BYTES: usize = K * 2;
const SCRATCH_BYTES: usize = 64;

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

    let (packed_host, expected_bf16) = q8_fixture();
    let hip = hip_bridge::HipRuntime::load()?;
    let hip_device = select_gfx1151(&hip)?;
    hip.set_device(hip_device)?;
    let packed = hip.malloc(PACKED_BYTES)?;
    let output = hip.malloc(OUTPUT_BYTES)?;
    let scratch = hip.malloc(SCRATCH_BYTES)?;
    hip.memcpy_htod(&packed, &packed_host)?;
    hip.memset(&output, 0xa5, output.size())?;
    hip.memset(&scratch, 0, scratch.size())?;
    hip.device_synchronize()?;

    let hsa = HsaRuntime::load()?;
    let packed_export = unsafe { hsa.export_dmabuf(packed.as_ptr().cast_const(), packed.size())? };
    let output_export = unsafe { hsa.export_dmabuf(output.as_ptr().cast_const(), output.size())? };
    let scratch_export =
        unsafe { hsa.export_dmabuf(scratch.as_ptr().cast_const(), scratch.size())? };

    let configured_device = std::env::var_os("HIPFIRE_XDNA_DEVICE").map(PathBuf::from);
    let device_path = resolve_device_path(configured_device.as_deref())?;
    let device = Device::open(&device_path)?;
    let firmware = device.metadata().firmware;
    let bundle = ArtifactBundle {
        manifest_path: "<live-q8-decoder-probe>".into(),
        manifest: ArtifactManifest {
            manifest_version: SUPPORTED_MANIFEST_VERSION,
            abi_version: SUPPORTED_ABI_VERSION,
            artifact_id: "diagnostic-q8-decode-bf16-k2048".into(),
            device: "gfx1151".into(),
            firmware: FirmwareCompatibility {
                minimum: firmware,
                maximum: firmware,
            },
            arithmetic: ProjectionArithmetic::Q8DecodeBf16Diagnostic,
            layout: IoLayout {
                activation: "none".into(),
                weight: "q8_0".into(),
                accumulator: "none".into(),
                output: "bf16".into(),
                q8_block_elements: BLOCK_ELEMENTS as u32,
                q8_block_bytes: BLOCK_BYTES as u32,
            },
            shapes: vec![ProjectionShape {
                k: K as u32,
                n: 1,
                max_batch: 1,
                masked_batch_tail: false,
                masked_output_tail: false,
            }],
            bindings: vec![
                BindingLayout {
                    name: "packed_q8".into(),
                    access: BindingAccess::Read,
                    minimum_bytes: packed.size() as u64,
                    alignment: 64,
                },
                BindingLayout {
                    name: "decoded_bf16".into(),
                    access: BindingAccess::Write,
                    minimum_bytes: output.size() as u64,
                    alignment: 64,
                },
                BindingLayout {
                    name: "scratch".into(),
                    access: BindingAccess::ReadWrite,
                    minimum_bytes: scratch.size() as u64,
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
    let scratch_bo = device.import_dmabuf(
        scratch_export.as_raw_fd(),
        scratch_export.offset(),
        scratch.size(),
    )?;
    for bo in [&packed_bo, &output_bo, &scratch_bo] {
        bo.sync_to_device(0, bo.len())?;
    }
    let bindings = [
        Binding::whole(&packed_bo).with_access(BindingAccess::Read),
        Binding::whole(&output_bo).with_access(BindingAccess::Write),
        Binding::whole(&scratch_bo),
    ];
    let iterations = bounded_env_usize("HIPFIRE_XDNA_ITERATIONS", 10, 1, 1_000)?;
    let mut timings_ns = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let timing = ring
            .submit(&program, &bindings)?
            .wait(Duration::from_secs(1))?;
        timings_ns.push(timing.elapsed.as_nanos());
    }

    let mut mapped_bytes = vec![0_u8; OUTPUT_BYTES];
    output_bo.read_mapped(0, &mut mapped_bytes)?;
    let mapped_bf16 = decode_u16(&mapped_bytes);
    verify("mapped dma-buf", &mapped_bf16, &expected_bf16)?;

    let mut hip_bytes = vec![0_u8; OUTPUT_BYTES];
    hip.memcpy_dtoh(&mut hip_bytes, &output)?;
    let hip_bf16 = decode_u16(&hip_bytes);
    verify("HIP-visible", &hip_bf16, &expected_bf16)?;

    timings_ns.sort_unstable();
    let median_ns = timings_ns[timings_ns.len() / 2];
    let p99_index = ((timings_ns.len() * 99).div_ceil(100)).saturating_sub(1);
    let p99_ns = timings_ns[p99_index];
    println!(
        "PASS hip_device={} device={} arithmetic=q8_decode_bf16_diagnostic \
         k={} blocks={} iterations={} median_us={:.3} p99_us={:.3}",
        hip_device,
        device_path.display(),
        K,
        BLOCKS,
        iterations,
        median_ns as f64 / 1_000.0,
        p99_ns as f64 / 1_000.0,
    );

    drop((packed_bo, output_bo, scratch_bo));
    drop((packed_export, output_export, scratch_export));
    hip.free(packed)?;
    hip.free(output)?;
    hip.free(scratch)?;
    Ok(())
}

fn q8_fixture() -> (Vec<u8>, Vec<u16>) {
    let scales = [0x3c00_u16, 0xb800, 0x3000, 0x4000];
    let mut packed = vec![0_u8; PACKED_BYTES];
    let mut expected = vec![0_u16; K];
    for block in 0..BLOCKS {
        let scale_bits = scales[block % scales.len()];
        let scale = fp16_to_f32(scale_bits);
        let offset = block * BLOCK_BYTES;
        packed[offset..offset + 2].copy_from_slice(&scale_bits.to_le_bytes());
        for lane in 0..BLOCK_ELEMENTS {
            let quantized = (((block * 37 + lane * 13) % 255) as i16 - 127) as i8;
            packed[offset + 2 + lane] = quantized as u8;
            expected[block * BLOCK_ELEMENTS + lane] = f32_to_bf16(scale * f32::from(quantized));
        }
    }
    (packed, expected)
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

fn decode_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect()
}

fn verify(route: &str, actual: &[u16], expected: &[u16]) -> Result<(), Box<dyn std::error::Error>> {
    let mismatches: Vec<_> = actual
        .iter()
        .zip(expected)
        .enumerate()
        .filter(|(_, (actual, expected))| actual != expected)
        .take(8)
        .map(|(index, (actual, expected))| (index, *actual, *expected))
        .collect();
    if mismatches.is_empty() {
        Ok(())
    } else {
        Err(format!("{route} Q8 decode mismatches: {mismatches:?}").into())
    }
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
