// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! AMD XDNA (Ryzen AI NPU) device layer.
//!
//! This crate is the device/runtime boundary that `hipfire-npu` (pure admission
//! policy) deliberately does **not** own. It talks to the in-tree `amdxdna`
//! kernel driver via the `DRM_IOCTL_AMDXDNA_GET_INFO` ioctl on
//! `/dev/accel/accelN` and decodes live NPU telemetry:
//!
//! - [`XdnaDevice::sensors`] — total power (mW) + per-column utilization (%),
//!   sourced from the `amd_pmf` driver (`amd_pmf_get_npu_data`).
//! - [`XdnaDevice::resource_info`] — max/current TOPS, max/current task counts,
//!   max H-clock.
//! - [`XdnaDevice::clocks`] — live MP-NPU and H clock frequencies (MHz).
//!
//! Scope today is read-only telemetry. xclbin/instr load and AIE command
//! dispatch are future modules in this same crate (mirroring how `hipfire-rocm`
//! is the ROCm device layer beneath the GPU policy crates).
//!
//! The ioctl path is Linux-only; on other targets every constructor returns
//! [`XdnaError::Unsupported`] so the crate still builds everywhere.

use std::fmt;

/// Default search set for the NPU accel node. The amdxdna NPU enumerates as a
/// DRM accel device; on a single-NPU box it is `accel0`.
const DEFAULT_ACCEL_NODES: &[&str] = &["/dev/accel/accel0", "/dev/accel/accel1"];

/// Errors from opening or querying the XDNA device.
#[derive(Debug)]
pub enum XdnaError {
    /// Built for a non-Linux target; the amdxdna ioctl ABI is unavailable.
    Unsupported,
    /// No `/dev/accel/accelN` node could be opened.
    NotFound,
    /// Opening the device node failed.
    Open(std::io::Error),
    /// The `GET_INFO` ioctl failed (e.g. `-EOPNOTSUPP` when `amd_pmf` is absent).
    Ioctl(std::io::Error),
    /// The kernel returned fewer bytes than one record.
    ShortResponse,
}

impl fmt::Display for XdnaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            XdnaError::Unsupported => write!(f, "XDNA ioctl ABI is Linux-only"),
            XdnaError::NotFound => write!(f, "no /dev/accel/accelN NPU device found"),
            XdnaError::Open(e) => write!(f, "open NPU device: {e}"),
            XdnaError::Ioctl(e) => write!(f, "GET_INFO ioctl: {e}"),
            XdnaError::ShortResponse => write!(f, "kernel returned a short telemetry buffer"),
        }
    }
}

impl std::error::Error for XdnaError {}

/// Live NPU sensor snapshot (from `DRM_AMDXDNA_QUERY_SENSORS`).
#[derive(Debug, Clone, Default)]
pub struct NpuSensors {
    /// Total NPU power in milliwatts, if the power sensor was present.
    pub power_mw: Option<u32>,
    /// Per-column utilization percentage `[0, 100]`, one entry per active column.
    pub column_utilization_pct: Vec<u32>,
}

impl NpuSensors {
    /// Mean utilization across reported columns (`0.0` if none).
    pub fn mean_utilization_pct(&self) -> f32 {
        if self.column_utilization_pct.is_empty() {
            return 0.0;
        }
        let sum: u32 = self.column_utilization_pct.iter().copied().sum();
        sum as f32 / self.column_utilization_pct.len() as f32
    }
}

/// NPU resource limits/usage (from `DRM_AMDXDNA_QUERY_RESOURCE_INFO`).
#[derive(Debug, Clone, Copy, Default)]
pub struct NpuResourceInfo {
    /// Max H-clock (MHz).
    pub npu_clk_max: u64,
    /// Max TOPS the device can deliver.
    pub npu_tops_max: u64,
    /// Max concurrent tasks (hardware-context limit).
    pub npu_task_max: u64,
    /// Current TOPS (scales with the active DPM level).
    pub npu_tops_curr: u64,
    /// Current number of active tasks (hardware contexts).
    pub npu_task_curr: u64,
}

/// Live NPU clocks in MHz (from `DRM_AMDXDNA_QUERY_CLOCK_METADATA`).
#[derive(Debug, Clone, Copy, Default)]
pub struct NpuClocks {
    /// MP-NPU clock (MHz).
    pub mp_npu_mhz: u32,
    /// H clock (MHz).
    pub h_mhz: u32,
}

// ── amdxdna uapi ABI (include/uapi/drm/amdxdna_accel.h) ──────────────────────
// enum amdxdna_drm_get_param
const PARAM_CLOCK_METADATA: u32 = 3;
const PARAM_SENSORS: u32 = 4;
const PARAM_RESOURCE_INFO: u32 = 12;

// enum amdxdna_sensor_type
const SENSOR_TYPE_POWER: u8 = 0;
const SENSOR_TYPE_COLUMN_UTILIZATION: u8 = 1;

// Strix Halo has 8 columns + 1 power sensor; allow generous headroom.
const MAX_SENSORS: usize = 16;

#[cfg(target_os = "linux")]
mod imp {
    use super::*;
    use std::os::fd::{IntoRawFd, RawFd};

    #[repr(C)]
    struct GetInfo {
        param: u32,
        buffer_size: u32, // in/out
        buffer: u64,      // userspace pointer
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct SensorRaw {
        label: [u8; 64],
        input: u32,
        max: u32,
        average: u32,
        highest: u32,
        status: [u8; 64],
        units: [u8; 16],
        unitm: i8,
        kind: u8,
        pad: [u8; 6],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct ResourceInfoRaw {
        npu_clk_max: u64,
        npu_tops_max: u64,
        npu_task_max: u64,
        npu_tops_curr: u64,
        npu_task_curr: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct ClockRaw {
        name: [u8; 16],
        freq_mhz: u32,
        pad: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct ClockMetadataRaw {
        mp_npu_clock: ClockRaw,
        h_clock: ClockRaw,
    }

    // ABI guards: any drift vs the kernel header is a compile error.
    const _: () = assert!(core::mem::size_of::<GetInfo>() == 16);
    const _: () = assert!(core::mem::size_of::<SensorRaw>() == 168);
    const _: () = assert!(core::mem::size_of::<ResourceInfoRaw>() == 40);
    const _: () = assert!(core::mem::size_of::<ClockMetadataRaw>() == 48);

    // DRM_IOCTL_AMDXDNA_GET_INFO = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDXDNA_GET_INFO,
    //                                       struct amdxdna_drm_get_info)
    const fn ioc(dir: u64, typ: u64, nr: u64, size: u64) -> u64 {
        (dir << 30) | (size << 16) | (typ << 8) | nr
    }
    const DRM_COMMAND_BASE: u64 = 0x40;
    const DRM_AMDXDNA_GET_INFO: u64 = 7;
    const IOC_READ_WRITE: u64 = 3; // _IOC_READ | _IOC_WRITE
    const DRM_TYPE: u64 = b'd' as u64;
    const GET_INFO_REQUEST: u64 = ioc(
        IOC_READ_WRITE,
        DRM_TYPE,
        DRM_COMMAND_BASE + DRM_AMDXDNA_GET_INFO,
        core::mem::size_of::<GetInfo>() as u64,
    );

    /// An open handle to the XDNA NPU accel device.
    pub struct XdnaDevice {
        fd: RawFd,
        path: String,
    }

    impl XdnaDevice {
        /// Open the first available NPU accel node from [`DEFAULT_ACCEL_NODES`].
        pub fn open_default() -> Result<Self, XdnaError> {
            let mut last = XdnaError::NotFound;
            for node in DEFAULT_ACCEL_NODES {
                match Self::open_path(node) {
                    Ok(dev) => return Ok(dev),
                    Err(e) => last = e,
                }
            }
            Err(last)
        }

        /// Open a specific accel node path.
        pub fn open_path(path: &str) -> Result<Self, XdnaError> {
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(path)
                .map_err(XdnaError::Open)?;
            Ok(XdnaDevice {
                fd: file.into_raw_fd(),
                path: path.to_string(),
            })
        }

        /// The device node path this handle was opened from.
        pub fn path(&self) -> &str {
            &self.path
        }

        /// SAFETY: `buf` must point to `param`'s record type with `cap` bytes.
        /// Returns the number of bytes the kernel reports written.
        fn get_info(&self, param: u32, buf: *mut u8, cap: u32) -> Result<u32, XdnaError> {
            let mut req = GetInfo {
                param,
                buffer_size: cap,
                buffer: buf as u64,
            };
            // SAFETY: req is a valid GetInfo; buffer points at `cap` writable bytes.
            let rc = unsafe {
                libc::ioctl(
                    self.fd,
                    GET_INFO_REQUEST as libc::c_ulong,
                    &mut req as *mut GetInfo as *mut libc::c_void,
                )
            };
            if rc != 0 {
                return Err(XdnaError::Ioctl(std::io::Error::last_os_error()));
            }
            Ok(req.buffer_size)
        }

        /// Query total power + per-column utilization.
        pub fn sensors(&self) -> Result<NpuSensors, XdnaError> {
            let mut raw = [SensorRaw {
                label: [0; 64],
                input: 0,
                max: 0,
                average: 0,
                highest: 0,
                status: [0; 64],
                units: [0; 16],
                unitm: 0,
                kind: 0,
                pad: [0; 6],
            }; MAX_SENSORS];
            let cap = (MAX_SENSORS * core::mem::size_of::<SensorRaw>()) as u32;
            let written = self.get_info(PARAM_SENSORS, raw.as_mut_ptr() as *mut u8, cap)?;
            let count = (written as usize) / core::mem::size_of::<SensorRaw>();

            let mut out = NpuSensors::default();
            for s in raw.iter().take(count) {
                match s.kind {
                    SENSOR_TYPE_POWER => out.power_mw = Some(s.input),
                    SENSOR_TYPE_COLUMN_UTILIZATION => out.column_utilization_pct.push(s.input),
                    _ => {}
                }
            }
            Ok(out)
        }

        /// Query resource limits/usage (TOPS, task counts, max clock).
        pub fn resource_info(&self) -> Result<NpuResourceInfo, XdnaError> {
            let mut raw = ResourceInfoRaw::default();
            let cap = core::mem::size_of::<ResourceInfoRaw>() as u32;
            let written = self.get_info(PARAM_RESOURCE_INFO, &mut raw as *mut _ as *mut u8, cap)?;
            if (written as usize) < core::mem::size_of::<ResourceInfoRaw>() {
                return Err(XdnaError::ShortResponse);
            }
            Ok(NpuResourceInfo {
                npu_clk_max: raw.npu_clk_max,
                npu_tops_max: raw.npu_tops_max,
                npu_task_max: raw.npu_task_max,
                npu_tops_curr: raw.npu_tops_curr,
                npu_task_curr: raw.npu_task_curr,
            })
        }

        /// Query live MP-NPU and H clock frequencies.
        pub fn clocks(&self) -> Result<NpuClocks, XdnaError> {
            let mut raw = ClockMetadataRaw {
                mp_npu_clock: ClockRaw {
                    name: [0; 16],
                    freq_mhz: 0,
                    pad: 0,
                },
                h_clock: ClockRaw {
                    name: [0; 16],
                    freq_mhz: 0,
                    pad: 0,
                },
            };
            let cap = core::mem::size_of::<ClockMetadataRaw>() as u32;
            let written =
                self.get_info(PARAM_CLOCK_METADATA, &mut raw as *mut _ as *mut u8, cap)?;
            if (written as usize) < core::mem::size_of::<ClockMetadataRaw>() {
                return Err(XdnaError::ShortResponse);
            }
            Ok(NpuClocks {
                mp_npu_mhz: raw.mp_npu_clock.freq_mhz,
                h_mhz: raw.h_clock.freq_mhz,
            })
        }
    }

    impl Drop for XdnaDevice {
        fn drop(&mut self) {
            // SAFETY: fd is owned by this handle and not closed elsewhere.
            unsafe {
                libc::close(self.fd);
            }
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    use super::*;

    /// Stub handle for non-Linux targets; all constructors fail.
    pub struct XdnaDevice {
        _priv: (),
    }

    impl XdnaDevice {
        pub fn open_default() -> Result<Self, XdnaError> {
            Err(XdnaError::Unsupported)
        }
        pub fn open_path(_path: &str) -> Result<Self, XdnaError> {
            Err(XdnaError::Unsupported)
        }
        pub fn path(&self) -> &str {
            ""
        }
        pub fn sensors(&self) -> Result<NpuSensors, XdnaError> {
            Err(XdnaError::Unsupported)
        }
        pub fn resource_info(&self) -> Result<NpuResourceInfo, XdnaError> {
            Err(XdnaError::Unsupported)
        }
        pub fn clocks(&self) -> Result<NpuClocks, XdnaError> {
            Err(XdnaError::Unsupported)
        }
    }
}

pub use imp::XdnaDevice;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_utilization_handles_empty() {
        let s = NpuSensors::default();
        assert_eq!(s.mean_utilization_pct(), 0.0);
    }

    #[test]
    fn mean_utilization_averages() {
        let s = NpuSensors {
            power_mw: Some(1200),
            column_utilization_pct: vec![0, 50, 100, 50],
        };
        assert_eq!(s.mean_utilization_pct(), 50.0);
    }

    #[test]
    fn open_default_is_graceful_when_absent() {
        // Must never panic; on hardware without the node this is NotFound/Open,
        // on non-Linux it is Unsupported. Either way it is an Err or Ok, not a panic.
        let _ = XdnaDevice::open_default();
    }
}
