// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Live interop probe: HIP allocation -> ROCr dma-buf export -> amdxdna import.
//!
//! This deliberately performs no NPU dispatch and is safe to run without an
//! artifact bundle. It proves the ownership direction used by production.

use hsa_bridge::HsaRuntime;
use redline_xdna::{resolve_device_path, Device};
use std::os::fd::{AsRawFd, RawFd};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let hip = hip_bridge::HipRuntime::load()?;
    hip.set_device(0)?;
    let allocation = hip.malloc(4096)?;
    hip.memset(&allocation, 0x5a, allocation.size())?;
    hip.device_synchronize()?;

    let hsa = HsaRuntime::load()?;
    let export = unsafe { hsa.export_dmabuf(allocation.as_ptr().cast_const(), allocation.size())? };
    let configured_device = std::env::var_os("HIPFIRE_XDNA_DEVICE").map(PathBuf::from);
    let device_path = resolve_device_path(configured_device.as_deref())?;
    let device = Device::open(&device_path)?;
    let imported = device.import_dmabuf(
        export.as_raw_fd() as RawFd,
        export.offset(),
        allocation.size(),
    )?;
    println!(
        "PASS arch=gfx1151 firmware={} tiles={} export_offset={} xdna_address=0x{:x} bytes={}",
        device.metadata().firmware,
        device.metadata().tiles,
        export.offset(),
        imported.address(),
        imported.len()
    );

    drop(imported);
    drop(export);
    hip.free(allocation)?;
    Ok(())
}
