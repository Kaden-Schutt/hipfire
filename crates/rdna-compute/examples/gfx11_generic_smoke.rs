// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Compile one Code Object V6 `gfx11-generic` image, load the exact same
//! bytes on two gfx11 devices, and verify integer output bit-for-bit.
//!
//! This is a portability gate, not a performance benchmark. Hot kernels may
//! still use exact `gfx1100` / `gfx1151` code objects after sharing source and
//! ABI through the generic family implementation.
//!
//! ```text
//! HIP_VISIBLE_DEVICES=0,1 cargo run --release -p rdna-compute \
//!   --example gfx11_generic_smoke -- \
//!   --expect-arch0 gfx1100 --expect-arch1 gfx1151
//! ```

use hip_bridge::{DeviceBuffer, Function, HipRuntime};
use rdna_compute::KernelCompiler;
use std::ffi::c_void;

const N: usize = 4096;
const SALTS: &[u32] = &[0, 1, 0x1357_9bdf, 0x8000_0001, 0xffff_ffff];

const SOURCE: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void gfx11_generic_rawbits(
    const unsigned int* input,
    unsigned int* output,
    unsigned int salt,
    unsigned int n
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned int x = input[i] ^ salt;
    output[i] = (x << 7 | x >> 25) + 0x9e3779b9u + i * 0x45d9f3bu;
}
"#;

#[derive(Debug, Default)]
struct Config {
    expect_arch0: Option<String>,
    expect_arch1: Option<String>,
}

impl Config {
    fn parse() -> Result<Self, String> {
        let mut cfg = Self::default();
        let args = std::env::args().skip(1).collect::<Vec<_>>();
        let mut i = 0;
        while i < args.len() {
            let flag = &args[i];
            let value = |i: &mut usize| -> Result<&str, String> {
                *i += 1;
                args.get(*i)
                    .map(String::as_str)
                    .ok_or_else(|| format!("{flag} requires a value"))
            };
            match flag.as_str() {
                "--expect-arch0" => cfg.expect_arch0 = Some(value(&mut i)?.to_owned()),
                "--expect-arch1" => cfg.expect_arch1 = Some(value(&mut i)?.to_owned()),
                "-h" | "--help" => {
                    println!(
                        "gfx11_generic_smoke [--expect-arch0 ARCH] [--expect-arch1 ARCH]"
                    );
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument {flag:?}; use --help")),
            }
            i += 1;
        }
        Ok(cfg)
    }
}

fn u32_bytes(values: &[u32]) -> &[u8] {
    // SAFETY: every possible bit pattern is valid for `u8`, and the byte
    // extent exactly covers the source slice.
    unsafe {
        std::slice::from_raw_parts(
            values.as_ptr().cast::<u8>(),
            std::mem::size_of_val(values),
        )
    }
}

fn u32_bytes_mut(values: &mut [u32]) -> &mut [u8] {
    // SAFETY: same representation argument as `u32_bytes`; the mutable borrow
    // of `values` is held for the returned slice's lifetime.
    unsafe {
        std::slice::from_raw_parts_mut(
            values.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(values),
        )
    }
}

fn expected(input: &[u32], salt: u32) -> Vec<u32> {
    input
        .iter()
        .enumerate()
        .map(|(i, value)| {
            (value ^ salt)
                .rotate_left(7)
                .wrapping_add(0x9e37_79b9)
                .wrapping_add((i as u32).wrapping_mul(0x045d_9f3b))
        })
        .collect()
}

fn run_device(
    hip: &HipRuntime,
    device: i32,
    arch: &str,
    image: &[u8],
    input: &[u32],
) -> Result<(), Box<dyn std::error::Error>> {
    hip.set_device(device)?;
    let module = hip.module_load_data(image)?;
    let function = hip.module_get_function(&module, "gfx11_generic_rawbits")?;
    let bytes = std::mem::size_of_val(input);
    let d_input = hip.malloc(bytes)?;
    let d_output = hip.malloc(bytes)?;
    hip.memcpy_htod(&d_input, u32_bytes(input))?;

    for &salt in SALTS {
        launch(hip, &function, &d_input, &d_output, salt, input.len())?;
        let mut output = vec![0_u32; input.len()];
        hip.memcpy_dtoh(u32_bytes_mut(&mut output), &d_output)?;
        assert_eq!(
            output,
            expected(input, salt),
            "generic code object produced wrong bits on device {device} ({arch}), salt={salt:#010x}"
        );
    }

    hip.free(d_input)?;
    hip.free(d_output)?;
    println!(
        "device={device} arch={arch} salts={} elements={} raw_bits=PASS",
        SALTS.len(),
        input.len()
    );
    Ok(())
}

fn launch(
    hip: &HipRuntime,
    function: &Function,
    input: &DeviceBuffer,
    output: &DeviceBuffer,
    salt: u32,
    n: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut input_ptr = input.as_ptr();
    let mut output_ptr = output.as_ptr();
    let mut salt_arg = salt;
    let mut n_arg = u32::try_from(n)?;
    let mut params = [
        (&mut input_ptr as *mut *mut c_void).cast::<c_void>(),
        (&mut output_ptr as *mut *mut c_void).cast::<c_void>(),
        (&mut salt_arg as *mut u32).cast::<c_void>(),
        (&mut n_arg as *mut u32).cast::<c_void>(),
    ];
    let block = 256_u32;
    let grid = n_arg.div_ceil(block);
    // SAFETY: argument order and widths exactly match `gfx11_generic_rawbits`;
    // both device buffers cover `n_arg * sizeof(u32)` bytes.
    unsafe {
        hip.launch_kernel(
            function,
            [grid, 1, 1],
            [block, 1, 1],
            0,
            None,
            &mut params,
        )?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = Config::parse().map_err(|error| format!("gfx11_generic_smoke: {error}"))?;
    let hip = HipRuntime::load()?;
    let count = hip.device_count()?;
    if count < 2 {
        return Err(format!("two visible gfx11 devices required, got {count}").into());
    }
    let arch0 = hip.get_arch(0)?;
    let arch1 = hip.get_arch(1)?;
    for (device, arch, expected_arch) in [
        (0, arch0.as_str(), cfg.expect_arch0.as_deref()),
        (1, arch1.as_str(), cfg.expect_arch1.as_deref()),
    ] {
        if !arch.starts_with("gfx11") {
            return Err(format!("device {device} is {arch}, not a gfx11 target").into());
        }
        if let Some(expected_arch) = expected_arch {
            if arch != expected_arch {
                return Err(
                    format!("device {device} is {arch}, expected {expected_arch}").into(),
                );
            }
        }
    }

    let mut compiler =
        KernelCompiler::new("gfx11-generic", "-mcode-object-version=6".to_owned())?;
    let artifact = compiler.compile("gfx11_generic_rawbits", SOURCE)?.to_owned();
    let image = std::fs::read(&artifact)?;
    println!(
        "artifact={} target=gfx11-generic code_object=6 bytes={} arch0={} arch1={}",
        artifact.display(),
        image.len(),
        arch0,
        arch1
    );

    let input = (0..N)
        .map(|i| (i as u32).wrapping_mul(0x27d4_eb2d) ^ 0xa5a5_5a5a)
        .collect::<Vec<_>>();
    run_device(&hip, 0, &arch0, &image, &input)?;
    run_device(&hip, 1, &arch1, &image, &input)?;
    println!("gfx11_generic_smoke: PASS");
    Ok(())
}
