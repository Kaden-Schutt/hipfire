// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! End-to-end interop proof using an existing BF16 mlir-aie GEMM artifact.
//!
//! Usage:
//! `HIPFIRE_XDNA_DANGEROUS_INTEROP=1 hip_bf16_gemm <main.pdi> <instructions.bin>`

use hsa_bridge::HsaRuntime;
use redline_xdna::{
    resolve_device_path, ArtifactBundle, ArtifactFile, ArtifactManifest, Binding, BindingAccess,
    BindingLayout, Device, FirmwareCompatibility, IoLayout, ProjectionArithmetic, ProjectionShape,
    SUPPORTED_ABI_VERSION, SUPPORTED_MANIFEST_VERSION,
};
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::time::Duration;

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

    const M: usize = 512;
    const K: usize = 2048;
    const N: usize = 2048;
    let hip = hip_bridge::HipRuntime::load()?;
    let hip_device = select_gfx1151(&hip)?;
    hip.set_device(hip_device)?;
    let a = hip.malloc(M * K * 2)?;
    let b = hip.malloc(K * N * 2)?;
    let c = hip.malloc(M * N * 4)?;
    let temporary = hip.malloc(4096)?;
    let trace = hip.malloc(4096)?;
    let one_bf16 = 0x3f80_u16.to_le_bytes();
    let a_host: Vec<u8> = one_bf16.into_iter().cycle().take(a.size()).collect();
    let b_host: Vec<u8> = one_bf16.into_iter().cycle().take(b.size()).collect();
    hip.memcpy_htod(&a, &a_host)?;
    hip.memcpy_htod(&b, &b_host)?;
    hip.memset(&c, 0, c.size())?;
    hip.memset(&temporary, 0, temporary.size())?;
    hip.memset(&trace, 0, trace.size())?;
    hip.device_synchronize()?;

    let hsa = HsaRuntime::load()?;
    let a_export = unsafe { hsa.export_dmabuf(a.as_ptr().cast_const(), a.size())? };
    let b_export = unsafe { hsa.export_dmabuf(b.as_ptr().cast_const(), b.size())? };
    let c_export = unsafe { hsa.export_dmabuf(c.as_ptr().cast_const(), c.size())? };
    let temporary_export =
        unsafe { hsa.export_dmabuf(temporary.as_ptr().cast_const(), temporary.size())? };
    let trace_export = unsafe { hsa.export_dmabuf(trace.as_ptr().cast_const(), trace.size())? };

    let configured_device = std::env::var_os("HIPFIRE_XDNA_DEVICE").map(PathBuf::from);
    let device_path = resolve_device_path(configured_device.as_deref())?;
    let device = Device::open(&device_path)?;
    let firmware = device.metadata().firmware;
    let bundle = ArtifactBundle {
        manifest_path: "<live-probe>".into(),
        manifest: ArtifactManifest {
            manifest_version: SUPPORTED_MANIFEST_VERSION,
            abi_version: SUPPORTED_ABI_VERSION,
            artifact_id: "live-bf16-512x2048x2048".into(),
            device: "gfx1151".into(),
            firmware: FirmwareCompatibility {
                minimum: firmware,
                maximum: firmware,
            },
            arithmetic: ProjectionArithmetic::Bf16Bf16F32Diagnostic,
            layout: IoLayout {
                activation: "bf16".into(),
                weight: "bf16".into(),
                accumulator: "f32".into(),
                output: "f32".into(),
                q8_block_elements: 0,
                q8_block_bytes: 0,
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
                    minimum_bytes: a.size() as u64,
                    alignment: 64,
                },
                BindingLayout {
                    name: "weight".into(),
                    access: BindingAccess::Read,
                    minimum_bytes: b.size() as u64,
                    alignment: 64,
                },
                BindingLayout {
                    name: "output".into(),
                    access: BindingAccess::Write,
                    minimum_bytes: c.size() as u64,
                    alignment: 64,
                },
                BindingLayout {
                    name: "temporary".into(),
                    access: BindingAccess::ReadWrite,
                    minimum_bytes: temporary.size() as u64,
                    alignment: 64,
                },
                BindingLayout {
                    name: "trace".into(),
                    access: BindingAccess::ReadWrite,
                    minimum_bytes: trace.size() as u64,
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
    let a_bo = device.import_dmabuf(a_export.as_raw_fd(), a_export.offset(), a.size())?;
    let b_bo = device.import_dmabuf(b_export.as_raw_fd(), b_export.offset(), b.size())?;
    let c_bo = device.import_dmabuf(c_export.as_raw_fd(), c_export.offset(), c.size())?;
    let temporary_bo = device.import_dmabuf(
        temporary_export.as_raw_fd(),
        temporary_export.offset(),
        temporary.size(),
    )?;
    let trace_bo = device.import_dmabuf(
        trace_export.as_raw_fd(),
        trace_export.offset(),
        trace.size(),
    )?;
    for bo in [&a_bo, &b_bo, &c_bo, &temporary_bo, &trace_bo] {
        bo.sync_to_device(0, bo.len())?;
    }
    let bindings = [
        Binding::whole(&a_bo).with_access(BindingAccess::Read),
        Binding::whole(&b_bo).with_access(BindingAccess::Read),
        Binding::whole(&c_bo).with_access(BindingAccess::Write),
        Binding::whole(&temporary_bo),
        Binding::whole(&trace_bo),
    ];
    let iterations = bounded_env_usize("HIPFIRE_XDNA_ITERATIONS", 1, 1, 1_000)?;
    let warmup = usize::from(iterations > 1);
    let mut timings_ns = Vec::with_capacity(iterations);
    for iteration in 0..warmup + iterations {
        let timing = ring
            .submit(&program, &bindings)?
            .wait(Duration::from_secs(2))?;
        if iteration >= warmup {
            timings_ns.push(timing.elapsed.as_nanos());
        }
    }
    timings_ns.sort_unstable();
    let median_ns = timings_ns[timings_ns.len() / 2];
    let p99_index = ((timings_ns.len() * 99).div_ceil(100)).saturating_sub(1);
    let p99_ns = timings_ns[p99_index];
    let mut mapped_result = vec![0_u8; 16 * 4];
    c_bo.read_mapped(0, &mut mapped_result)?;
    let mapped_values: Vec<f32> = mapped_result
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    let mut hip_result = vec![0_u8; mapped_result.len()];
    hip.memcpy_dtoh(&mut hip_result, &c)?;
    let hip_values: Vec<f32> = hip_result
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    if mapped_values
        .iter()
        .chain(&hip_values)
        .any(|value| (*value - K as f32).abs() > 0.01)
    {
        return Err(format!(
            "unexpected first output values: mapped={mapped_values:?} hip={hip_values:?}"
        )
        .into());
    }
    println!(
        "PASS hip_device={} iterations={} median_us={:.3} p99_us={:.3} mapped={:?} hip={:?}",
        hip_device,
        iterations,
        median_ns as f64 / 1_000.0,
        p99_ns as f64 / 1_000.0,
        &mapped_values[..4],
        &hip_values[..4]
    );

    drop((a_bo, b_bo, c_bo, temporary_bo, trace_bo));
    drop((a_export, b_export, c_export, temporary_export, trace_export));
    hip.free(a)?;
    hip.free(b)?;
    hip.free(c)?;
    hip.free(temporary)?;
    hip.free(trace)?;
    Ok(())
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
