// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Host hardware profile collection.
//!
//! Builds a `HostProfile` from KFD topology (`/sys/class/kfd`), sysfs DRM
//! attributes, and an optional `libdrm_amdgpu` dlopen probe (CU count, VRAM
//! size, memory class/width/clock). All probes degrade gracefully to `None` on
//! non-AMD or headless hosts.
//!
//! Relocated from `hipfire-eval` into this HIP-independent leaf crate so the
//! inference hot path (`hipfire-runtime`) can collect a host profile without
//! pulling the eval harness (and its tokio-process / daemon-adapter closure)
//! into its dependency graph. The pure host-profile math (`hardware_bucket`,
//! `host_profile_hash`, `compute_peak_bandwidth_gbps`, `classify_hardware_kind`)
//! stays in `hipfire-evidence`, which owns the `HostProfile` type; this module
//! only owns the sysfs/libdrm probing and the arch detection.

use hipfire_evidence::{
    classify_hardware_kind, compute_peak_bandwidth_gbps, hardware_bucket, host_profile_hash,
    EvalStatus, HostProfile, SourcedField,
};
use std::collections::BTreeMap;
use std::ffi::{c_void, CString};
use std::fs;
use std::path::{Path, PathBuf};

/// Operator-supplied overrides for host-profile fields that can't be probed
/// reliably (or that a bench wants to pin). An unset field falls back to the
/// libdrm probe / sysfs value.
#[derive(Debug, Clone, Default)]
pub struct HostProfileOverrides {
    pub memory_class: Option<String>,
    pub memory_width_bits: Option<u32>,
    pub memory_bandwidth_gbps: Option<f64>,
}

/// Collect a full [`HostProfile`] with the default (empty) overrides and the
/// arch auto-detected from KFD topology.
pub fn collect_default_host_profile() -> HostProfile {
    collect_host_profile(detect_arch(), HostProfileOverrides::default())
}

pub fn collect_host_profile(arch: Option<String>, overrides: HostProfileOverrides) -> HostProfile {
    let topology = read_primary_kfd_properties();
    let drm = topology
        .as_ref()
        .and_then(|props| props.get("drm_render_minor"))
        .and_then(|minor| minor.parse::<u32>().ok())
        .map(|minor| PathBuf::from(format!("/sys/class/drm/renderD{minor}/device")))
        .filter(|path| path.exists())
        .or_else(primary_amd_drm_device);

    let vendor_id = drm
        .as_ref()
        .and_then(|path| read_sysfs_trimmed(&path.join("vendor")));
    let device_id = drm
        .as_ref()
        .and_then(|path| read_sysfs_trimmed(&path.join("device")));
    let render_node = topology
        .as_ref()
        .and_then(|props| props.get("drm_render_minor"))
        .map(|minor| format!("/dev/dri/renderD{minor}"));
    let libdrm_probe = render_node
        .as_deref()
        .and_then(probe_amdgpu_dev_info_libdrm)
        .or_else(|| probe_amdgpu_dev_info_libdrm("/dev/dri/renderD128"));
    let gfx = arch.or_else(|| {
        topology
            .as_ref()
            .and_then(|props| props.get("gfx_target_version"))
            .and_then(|raw| raw.parse::<u32>().ok())
            .map(gfx_target_version_to_arch)
    });
    let simd_count = topology
        .as_ref()
        .and_then(|props| props.get("simd_count"))
        .and_then(|raw| raw.parse::<u32>().ok());
    let simd_per_cu = topology
        .as_ref()
        .and_then(|props| props.get("simd_per_cu"))
        .and_then(|raw| raw.parse::<u32>().ok())
        .filter(|value| *value > 0);
    let cu_count = libdrm_probe
        .as_ref()
        .and_then(|probe| probe.cu_count)
        .or_else(|| match (simd_count, simd_per_cu) {
            (Some(simd_count), Some(simd_per_cu)) => Some(simd_count / simd_per_cu),
            _ => None,
        });
    let vram_bytes = drm
        .as_ref()
        .and_then(|path| read_sysfs_trimmed(&path.join("mem_info_vram_total")))
        .and_then(|raw| raw.parse::<u64>().ok())
        .or_else(|| {
            topology
                .as_ref()
                .and_then(|props| props.get("local_mem_size"))
                .and_then(|raw| raw.parse::<u64>().ok())
        })
        .or_else(|| libdrm_probe.as_ref().and_then(|probe| probe.vram_bytes));
    let gtt_bytes = drm
        .as_ref()
        .and_then(|path| read_sysfs_trimmed(&path.join("mem_info_gtt_total")))
        .and_then(|raw| raw.parse::<u64>().ok());
    let memory_clock_mhz = libdrm_probe
        .as_ref()
        .and_then(|probe| probe.max_memory_clock_mhz)
        .or_else(|| {
            drm.as_ref()
                .and_then(|path| fs::read_to_string(path.join("pp_dpm_mclk")).ok())
                .and_then(|raw| parse_pp_dpm_mclk_max_mhz(&raw))
        });
    let system_memory_bytes = linux_mem_total_bytes();
    let hardware_kind = classify_hardware_kind(vram_bytes, gtt_bytes);

    let mut memory_class = SourcedField::unknown();
    if let Some(value) = overrides.memory_class.clone() {
        memory_class = SourcedField::override_value(value);
    } else if let Some(value) = libdrm_probe
        .as_ref()
        .and_then(|probe| probe.memory_class.clone())
    {
        memory_class = SourcedField::libdrm_value(value);
    }
    let mut memory_width_bits = SourcedField::unknown();
    if let Some(value) = overrides.memory_width_bits {
        memory_width_bits = SourcedField::override_value(value);
    } else if let Some(value) = libdrm_probe
        .as_ref()
        .and_then(|probe| probe.memory_width_bits)
    {
        memory_width_bits = SourcedField::libdrm_value(value);
    }
    let memory_clock_mhz = memory_clock_mhz
        .map(|value| {
            if libdrm_probe
                .as_ref()
                .and_then(|probe| probe.max_memory_clock_mhz)
                == Some(value)
            {
                SourcedField::libdrm_value(value)
            } else {
                SourcedField::sysfs_value(value)
            }
        })
        .unwrap_or_else(SourcedField::unknown);
    let peak_bandwidth_gbps = if let Some(value) = overrides.memory_bandwidth_gbps {
        SourcedField::override_value(value)
    } else if let (Some(clock), Some(width), Some(class)) = (
        memory_clock_mhz.value,
        memory_width_bits.value,
        memory_class.value.as_deref(),
    ) {
        compute_peak_bandwidth_gbps(clock, width, class)
            .map(SourcedField::computed_value)
            .unwrap_or_else(SourcedField::unknown)
    } else {
        SourcedField::unknown()
    };

    let probe_status = if topology.is_some() || drm.is_some() {
        EvalStatus::Pass
    } else {
        EvalStatus::Skip
    };
    let reason = if probe_status == EvalStatus::Skip {
        Some("no AMD KFD/DRM device metadata found".to_string())
    } else {
        None
    };
    let hardware_bucket = hardware_bucket(
        &hardware_kind,
        gfx.as_deref(),
        device_id.as_deref(),
        cu_count,
        vram_bytes,
        memory_class.value.as_deref(),
        memory_width_bits.value,
        peak_bandwidth_gbps.value,
    );
    let mut profile = HostProfile {
        schema: 1,
        source: if libdrm_probe.is_some() {
            "libdrm_amdgpu+kfd-sysfs".to_string()
        } else {
            "linux-kfd-drm-sysfs".to_string()
        },
        probe_status,
        reason,
        hardware_kind,
        hardware_bucket,
        host_profile_hash: String::new(),
        gpu_model: drm
            .as_ref()
            .and_then(|path| read_sysfs_trimmed(&path.join("product_name"))),
        gfx,
        vendor_id,
        device_id: libdrm_probe
            .as_ref()
            .map(|probe| format!("0x{:04x}", probe.asic_id))
            .or(device_id),
        render_node,
        cu_count,
        vram_bytes,
        gtt_bytes,
        system_memory_bytes,
        memory_class,
        memory_width_bits,
        memory_clock_mhz,
        peak_bandwidth_gbps,
    };
    profile.host_profile_hash = host_profile_hash(&profile);
    profile
}

/// Detect the primary GPU arch string (e.g. `gfx1103`) from KFD topology.
pub fn detect_arch() -> Option<String> {
    for node in ["1", "0"] {
        let path = format!("/sys/class/kfd/kfd/topology/nodes/{node}/properties");
        let raw = match fs::read_to_string(path) {
            Ok(raw) => raw,
            Err(_) => continue,
        };
        for line in raw.lines() {
            if let Some(v) = line.strip_prefix("gfx_target_version") {
                if let Ok(ver) = v.trim().parse() {
                    return Some(gfx_target_version_to_arch(ver));
                }
            }
        }
    }
    None
}

fn gfx_target_version_to_arch(ver: u32) -> String {
    match ver {
        100100 => "gfx1010".to_string(),
        100300 | 100302 => "gfx1030".to_string(),
        110000 | 110001 => "gfx1100".to_string(),
        110501 => "gfx1151".to_string(),
        120000 => "gfx1200".to_string(),
        120001 => "gfx1201".to_string(),
        _ => {
            let major = ver / 10000;
            let minor = (ver % 10000) / 100;
            let step = ver % 100;
            format!("gfx{major}{minor}{step}")
        }
    }
}

#[derive(Debug, Clone)]
struct LibDrmAmdgpuProbe {
    asic_id: u32,
    cu_count: Option<u32>,
    vram_bytes: Option<u64>,
    memory_class: Option<String>,
    memory_width_bits: Option<u32>,
    max_memory_clock_mhz: Option<f64>,
}

#[repr(C)]
#[derive(Default)]
struct AmdgpuGpuInfo {
    asic_id: u32,
    chip_rev: u32,
    chip_external_rev: u32,
    family_id: u32,
    ids_flags: u64,
    max_engine_clk: u64,
    max_memory_clk: u64,
    num_shader_engines: u32,
    num_shader_arrays_per_engine: u32,
    avail_quad_shader_pipes: u32,
    max_quad_shader_pipes: u32,
    cache_entries_per_quad_pipe: u32,
    num_hw_gfx_contexts: u32,
    rb_pipes: u32,
    enabled_rb_pipes_mask: u32,
    gpu_counter_freq: u32,
    backend_disable: [u32; 4],
    mc_arb_ramcfg: u32,
    gb_addr_cfg: u32,
    gb_tile_mode: [u32; 32],
    gb_macro_tile_mode: [u32; 16],
    pa_sc_raster_cfg: [u32; 4],
    pa_sc_raster_cfg1: [u32; 4],
    cu_active_number: u32,
    cu_ao_mask: u32,
    cu_bitmap: [[u32; 4]; 4],
    vram_type: u32,
    vram_bit_width: u32,
    ce_ram_size: u32,
    vce_harvest_config: u32,
    pci_rev_id: u32,
}

#[repr(C)]
#[derive(Default)]
struct AmdgpuHeapInfo {
    total_heap_size: u64,
    usable_heap_size: u64,
    heap_usage: u64,
    max_allocation: u64,
}

type AmdgpuDeviceHandle = *mut c_void;
type AmdgpuDeviceInitialize =
    unsafe extern "C" fn(i32, *mut u32, *mut u32, *mut AmdgpuDeviceHandle) -> i32;
type AmdgpuDeviceDeinitialize = unsafe extern "C" fn(AmdgpuDeviceHandle) -> i32;
type AmdgpuQueryGpuInfo = unsafe extern "C" fn(AmdgpuDeviceHandle, *mut AmdgpuGpuInfo) -> i32;
type AmdgpuQueryHeapInfo =
    unsafe extern "C" fn(AmdgpuDeviceHandle, u32, u32, *mut AmdgpuHeapInfo) -> i32;

const AMDGPU_GEM_DOMAIN_VRAM: u32 = 0x4;

fn probe_amdgpu_dev_info_libdrm(render_node: &str) -> Option<LibDrmAmdgpuProbe> {
    unsafe {
        let lib = dlopen_first(&["libdrm_amdgpu.so.1", "libdrm_amdgpu.so"])?;
        let device_initialize: AmdgpuDeviceInitialize =
            std::mem::transmute(dlsym_required(lib, "amdgpu_device_initialize")?);
        let device_deinitialize: AmdgpuDeviceDeinitialize =
            std::mem::transmute(dlsym_required(lib, "amdgpu_device_deinitialize")?);
        let query_gpu_info: AmdgpuQueryGpuInfo =
            std::mem::transmute(dlsym_required(lib, "amdgpu_query_gpu_info")?);
        let query_heap_info: AmdgpuQueryHeapInfo =
            std::mem::transmute(dlsym_required(lib, "amdgpu_query_heap_info")?);

        let path = CString::new(render_node).ok()?;
        let fd = libc::open(path.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC);
        if fd < 0 {
            libc::dlclose(lib);
            return None;
        }
        let mut major = 0u32;
        let mut minor = 0u32;
        let mut handle: AmdgpuDeviceHandle = std::ptr::null_mut();
        if device_initialize(fd, &mut major, &mut minor, &mut handle) != 0 || handle.is_null() {
            libc::close(fd);
            libc::dlclose(lib);
            return None;
        }

        let mut gpu_info = AmdgpuGpuInfo::default();
        let gpu_ok = query_gpu_info(handle, &mut gpu_info) == 0;
        let mut heap_info = AmdgpuHeapInfo::default();
        let heap_ok = query_heap_info(handle, AMDGPU_GEM_DOMAIN_VRAM, 0, &mut heap_info) == 0;
        let _ = device_deinitialize(handle);
        libc::close(fd);
        libc::dlclose(lib);
        if !gpu_ok {
            return None;
        }
        Some(LibDrmAmdgpuProbe {
            asic_id: gpu_info.asic_id,
            cu_count: nonzero_u32(gpu_info.cu_active_number),
            vram_bytes: heap_ok.then_some(heap_info.total_heap_size),
            memory_class: amdgpu_vram_type_name(gpu_info.vram_type).map(str::to_string),
            memory_width_bits: nonzero_u32(gpu_info.vram_bit_width),
            max_memory_clock_mhz: nonzero_u64(gpu_info.max_memory_clk)
                .map(|khz| khz as f64 / 1000.0),
        })
    }
}

unsafe fn dlopen_first(names: &[&str]) -> Option<*mut c_void> {
    for name in names {
        let c_name = CString::new(*name).ok()?;
        let lib = libc::dlopen(c_name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
        if !lib.is_null() {
            return Some(lib);
        }
    }
    None
}

unsafe fn dlsym_required(lib: *mut c_void, name: &str) -> Option<*mut c_void> {
    let c_name = CString::new(name).ok()?;
    let symbol = libc::dlsym(lib, c_name.as_ptr());
    if symbol.is_null() {
        None
    } else {
        Some(symbol)
    }
}

fn nonzero_u32(value: u32) -> Option<u32> {
    (value != 0).then_some(value)
}

fn nonzero_u64(value: u64) -> Option<u64> {
    (value != 0).then_some(value)
}

fn amdgpu_vram_type_name(raw: u32) -> Option<&'static str> {
    match raw {
        2 => Some("ddr2"),
        5 => Some("gddr5"),
        6 => Some("hbm"),
        7 => Some("ddr3"),
        8 => Some("ddr4"),
        9 => Some("gddr6"),
        10 => Some("ddr5"),
        11 => Some("lpddr4"),
        12 => Some("lpddr5"),
        13 => Some("hbm3e"),
        14 => Some("hbm4"),
        _ => None,
    }
}

fn read_primary_kfd_properties() -> Option<BTreeMap<String, String>> {
    for node in ["1", "0"] {
        let path = format!("/sys/class/kfd/kfd/topology/nodes/{node}/properties");
        if let Ok(raw) = fs::read_to_string(path) {
            let props = parse_kfd_properties(&raw);
            if props.get("gfx_target_version").is_some() {
                return Some(props);
            }
        }
    }
    None
}

fn parse_kfd_properties(raw: &str) -> BTreeMap<String, String> {
    raw.lines()
        .filter_map(|line| {
            let (key, value) = line.split_once(' ')?;
            Some((key.trim().to_string(), value.trim().to_string()))
        })
        .collect()
}

fn primary_amd_drm_device() -> Option<PathBuf> {
    let entries = fs::read_dir("/sys/class/drm").ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("renderD") {
            continue;
        }
        let device = entry.path().join("device");
        if read_sysfs_trimmed(&device.join("vendor")).as_deref() == Some("0x1002") {
            return Some(device);
        }
    }
    None
}

fn read_sysfs_trimmed(path: &Path) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .map(|raw| raw.trim().to_string())
        .filter(|raw| !raw.is_empty())
}

pub fn parse_pp_dpm_mclk_max_mhz(raw: &str) -> Option<f64> {
    raw.lines()
        .filter_map(|line| {
            let after_colon = line.split_once(':').map(|(_, rest)| rest).unwrap_or(line);
            let token = after_colon
                .split_whitespace()
                .find(|part| part.to_ascii_lowercase().contains("mhz"))?;
            let digits = token
                .trim_end_matches('*')
                .trim_end_matches("Mhz")
                .trim_end_matches("MHz")
                .trim_end_matches("mhz");
            digits.parse::<f64>().ok()
        })
        .filter(|value| value.is_finite())
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
}

fn linux_mem_total_bytes() -> Option<u64> {
    let raw = fs::read_to_string("/proc/meminfo").ok()?;
    for line in raw.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kib = rest.split_whitespace().next()?.parse::<u64>().ok()?;
            return Some(kib * 1024);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_pp_dpm_mclk_picks_max_mhz() {
        let raw = "0: 400Mhz\n1: 800Mhz *\n2: 937Mhz\n";
        assert_eq!(parse_pp_dpm_mclk_max_mhz(raw), Some(937.0));
    }

    #[test]
    fn gfx_target_version_maps_known_and_unknown() {
        assert_eq!(gfx_target_version_to_arch(110501), "gfx1151");
        assert_eq!(gfx_target_version_to_arch(100300), "gfx1030");
        // Unknown version falls back to the decomposed major/minor/step string.
        assert_eq!(gfx_target_version_to_arch(110003), "gfx1103");
        assert_eq!(gfx_target_version_to_arch(110300), "gfx1130");
    }

    #[test]
    fn collect_host_profile_is_deterministic_and_self_consistent() {
        // Environment-agnostic: same inputs must yield the same profile (no
        // clock/randomness baked in), and the stored hash must equal a fresh
        // hash of the profile. Holds whether or not this host has an AMD GPU.
        let a = collect_host_profile(None, HostProfileOverrides::default());
        let b = collect_host_profile(None, HostProfileOverrides::default());
        assert_eq!(a.host_profile_hash, b.host_profile_hash);
        assert_eq!(a.host_profile_hash, host_profile_hash(&a));
        assert_eq!(a.schema, 1);
    }
}
