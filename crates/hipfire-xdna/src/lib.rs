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
    /// A DEV BO's device address fell outside the backing heap mapping.
    DevBoOutsideHeap,
}

impl fmt::Display for XdnaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            XdnaError::Unsupported => write!(f, "XDNA ioctl ABI is Linux-only"),
            XdnaError::NotFound => write!(f, "no /dev/accel/accelN NPU device found"),
            XdnaError::Open(e) => write!(f, "open NPU device: {e}"),
            XdnaError::Ioctl(e) => write!(f, "amdxdna ioctl: {e}"),
            XdnaError::ShortResponse => write!(f, "kernel returned a short telemetry buffer"),
            XdnaError::DevBoOutsideHeap => write!(f, "DEV BO device address outside heap mapping"),
        }
    }
}

impl std::error::Error for XdnaError {}

/// Live NPU sensor snapshot (from `DRM_AMDXDNA_QUERY_SENSORS`).
#[derive(Debug, Clone, Default)]
pub struct NpuSensors {
    /// Total NPU power in milliwatts, if the power sensor was present.
    pub power_mw: Option<u32>,
    /// NPU temperature in degrees C, if the temperature sensor was present.
    pub temp_c: Option<u32>,
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
const SENSOR_TYPE_TEMPERATURE: u8 = 2;

// Strix Halo has 8 columns + 1 power sensor; allow generous headroom.
const MAX_SENSORS: usize = 16;

// W1: amdxdna command-submission ABI (structs + ioctl numbers), foundation for
// the W4A8 kernel wire-in. See docs/npu/wire-in-amdxdna-command-submission.md.
#[cfg(target_os = "linux")]
pub mod submit;

// W3a: AXLF (xclbin2) container parser — enumerate sections / extract the AIE
// partition + PDI. Pure byte parsing, target-independent.
pub mod xclbin;

#[cfg(target_os = "linux")]
mod imp {
    use super::*; // brings the crate-root `submit` module into scope
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

    /// Fixed userspace VA for the device heap mapping — must be a moderate,
    /// 2 MiB-aligned address inside the NPU's IOMMU-addressable window (the
    /// kernel's default placement is too high and the firmware rejects it).
    const DEV_HEAP_VA: usize = 0x7000_0000_0000;

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
                    SENSOR_TYPE_TEMPERATURE => out.temp_c = Some(s.input),
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

        // ── W3c: hardware contexts ────────────────────────────────────────
        // A hwctx reserves `num_tiles / row_count` AIE columns (no program runs
        // until CONFIG_HWCTX loads a PDI + EXEC_CMD). `num_tiles` = num_col *
        // core row_count (aie2p Strix Halo: 4 rows, so 8 cols => 32 tiles).

        /// Create a hardware context reserving `num_tiles` AIE tiles. Returns
        /// `(handle, syncobj_handle)`. QoS is passed by pointer as the driver
        /// requires; zeros are accepted.
        pub fn create_hwctx(
            &self,
            num_tiles: u32,
            mem_size: u32,
            max_opc: u32,
            qos: &submit::QosInfo,
        ) -> Result<(u32, u32), XdnaError> {
            let mut c = submit::CreateHwctx {
                qos_p: qos as *const submit::QosInfo as u64,
                num_tiles,
                mem_size,
                max_opc,
                ..Default::default()
            };
            self.submit_ioctl(
                submit::CREATE_HWCTX_REQUEST,
                &mut c as *mut _ as *mut libc::c_void,
            )?;
            Ok((c.handle, c.syncobj_handle))
        }

        /// Destroy a hardware context created by [`Self::create_hwctx`].
        pub fn destroy_hwctx(&self, handle: u32) -> Result<(), XdnaError> {
            let mut d = submit::DestroyHwctx { handle, pad: 0 };
            self.submit_ioctl(
                submit::DESTROY_HWCTX_REQUEST,
                &mut d as *mut _ as *mut libc::c_void,
            )
        }

        /// Configure the hwctx's CU (loads the compiled tile program): CONFIG_HWCTX
        /// with `DRM_AMDXDNA_HWCTX_CONFIG_CU` and one `cu_config{ cu_bo, cu_func=0 }`.
        /// `cu_bo` is a BO holding the PDI (from the xclbin AIE_PARTITION).
        pub fn config_hwctx_cu(&self, hwctx: u32, cu_bo: u32) -> Result<(), XdnaError> {
            // struct hwctx_param_config_cu { u16 num_cus; u16 pad[3];
            //   cu_config { u32 cu_bo; u8 cu_func; u8 pad[3]; } } = 16 bytes.
            let mut cfg = [0u8; 16];
            cfg[0..2].copy_from_slice(&1u16.to_le_bytes()); // num_cus = 1
            cfg[8..12].copy_from_slice(&cu_bo.to_le_bytes()); // cu_config[0].cu_bo
            cfg[12] = 0; // cu_func
            let mut c = submit::ConfigHwctx {
                handle: hwctx,
                param_type: submit::DRM_AMDXDNA_HWCTX_CONFIG_CU,
                param_val: cfg.as_ptr() as u64,
                param_val_size: cfg.len() as u32,
                pad: 0,
            };
            self.submit_ioctl(
                submit::CONFIG_HWCTX_REQUEST,
                &mut c as *mut _ as *mut libc::c_void,
            )
        }

        /// Submit one ERT command BO to a hwctx (`AMDXDNA_CMD_SUBMIT_EXEC_BUF`).
        /// `arg_bos` are the data + instruction BO handles the command references
        /// (for residency/pin); the PDI/cu_bo is configured separately and is not
        /// listed here. Returns the submission sequence number to wait on.
        pub fn exec_cmd(&self, hwctx: u32, cmd_bo: u32, arg_bos: &[u32]) -> Result<u64, XdnaError> {
            let mut e = submit::ExecCmd {
                hwctx,
                cmd_type: submit::AMDXDNA_CMD_SUBMIT_EXEC_BUF,
                // For cmd_count==1 this field is the BO handle itself, not a pointer.
                cmd_handles: cmd_bo as u64,
                args: arg_bos.as_ptr() as u64,
                cmd_count: 1,
                arg_count: arg_bos.len() as u32,
                ..Default::default()
            };
            self.submit_ioctl(
                submit::EXEC_CMD_REQUEST,
                &mut e as *mut _ as *mut libc::c_void,
            )?;
            Ok(e.seq)
        }

        /// Block until a submitted command's timeline `point` signals on the hwctx
        /// `syncobj` (from [`Self::create_hwctx`]). Mirrors XRT: `WAIT_FOR_SUBMIT`,
        /// `timeout_nsec = i64::MAX`.
        pub fn syncobj_wait(&self, syncobj: u32, point: u64) -> Result<(), XdnaError> {
            let handles = [syncobj];
            let points = [point];
            let mut w = submit::SyncobjTimelineWait {
                handles: handles.as_ptr() as u64,
                points: points.as_ptr() as u64,
                timeout_nsec: i64::MAX,
                count_handles: 1,
                flags: submit::SYNCOBJ_WAIT_FOR_SUBMIT,
                ..Default::default()
            };
            self.submit_ioctl(
                submit::SYNCOBJ_TIMELINE_WAIT_REQUEST,
                &mut w as *mut _ as *mut libc::c_void,
            )
        }

        /// Allocate a DEV buffer object (for the PDI / instruction stream, which
        /// live in device memory) and fill it with `data`. DEV BOs are carved out
        /// of `heap` by the driver and are *not* directly mmap-able (GET_BO_INFO
        /// returns `map_offset = INVALID`); userspace fills them by writing into the
        /// heap's own mapping at the BO's offset (`bo.xdna_addr - heap.xdna_addr`),
        /// then SYNC_BO flushes those heap pages. Returns `(handle, xdna_addr)` —
        /// the device address goes in the command packet.
        pub fn alloc_dev_bo(
            &self,
            heap: &mut DeviceBuffer,
            data: &[u8],
        ) -> Result<(u32, u64), XdnaError> {
            let mut cb = submit::CreateBo {
                size: data.len() as u64,
                bo_type: submit::AMDXDNA_BO_DEV,
                ..Default::default()
            };
            self.submit_ioctl(
                submit::CREATE_BO_REQUEST,
                &mut cb as *mut _ as *mut libc::c_void,
            )?;
            let handle = cb.handle;
            let mut info = submit::GetBoInfo {
                handle,
                ..Default::default()
            };
            self.submit_ioctl(
                submit::GET_BO_INFO_REQUEST,
                &mut info as *mut _ as *mut libc::c_void,
            )?;
            // Write through the heap mapping at the BO's offset within the heap.
            let off = info
                .xdna_addr
                .checked_sub(heap.xdna_addr())
                .and_then(|o| usize::try_from(o).ok())
                .filter(|&o| o + data.len() <= heap.len())
                .ok_or(XdnaError::DevBoOutsideHeap)?;
            heap.as_mut_slice()[off..off + data.len()].copy_from_slice(data);
            self.sync_bo(handle, submit::SYNC_DIRECT_TO_DEVICE, data.len())?;
            Ok((handle, info.xdna_addr))
        }

        // ── W2: buffer objects (command-submission path) ──────────────────
        // See docs/npu/wire-in-amdxdna-command-submission.md.

        /// Allocate a buffer object of `size` bytes and mmap it into this process.
        /// `bo_type` is one of `submit::AMDXDNA_BO_*` (e.g. `AMDXDNA_BO_SHMEM`).
        pub fn alloc_buffer(&self, size: usize, bo_type: u32) -> Result<DeviceBuffer, XdnaError> {
            let mut cb = submit::CreateBo {
                size: size as u64,
                bo_type,
                ..Default::default()
            };
            self.submit_ioctl(
                submit::CREATE_BO_REQUEST,
                &mut cb as *mut _ as *mut libc::c_void,
            )?;
            let handle = cb.handle;

            let mut info = submit::GetBoInfo {
                handle,
                ..Default::default()
            };
            self.submit_ioctl(
                submit::GET_BO_INFO_REQUEST,
                &mut info as *mut _ as *mut libc::c_void,
            )?;

            // SAFETY: map_offset is the driver's fake mmap offset for this BO; the
            // fd is our open device; PROT/flags match a shared host mapping.
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    // MAP_LOCKED pins the pages so the firmware can map the buffer
                    // (a DEV_HEAP without it fails aie2_hwctx_init's host-buf map).
                    libc::MAP_SHARED | libc::MAP_LOCKED,
                    self.fd,
                    info.map_offset as libc::off_t,
                )
            };
            if ptr == libc::MAP_FAILED {
                return Err(XdnaError::Ioctl(std::io::Error::last_os_error()));
            }
            Ok(DeviceBuffer {
                handle,
                ptr: ptr as *mut u8,
                len: size,
                xdna_addr: info.xdna_addr,
            })
        }

        /// Allocate + map the device heap the way XRT does: CREATE_BO(DEV_HEAP),
        /// then mmap at the fixed DEV_HEAP offset 0x1_0000_0000 with MAP_LOCKED so
        /// the firmware host-buffer map in aie2_hwctx_init succeeds. Returns the
        /// mapped DeviceBuffer (keep it alive for the hwctx's lifetime).
        pub fn alloc_dev_heap(&self, size: usize) -> Result<DeviceBuffer, XdnaError> {
            let mut cb = submit::CreateBo {
                size: size as u64,
                bo_type: submit::AMDXDNA_BO_DEV_HEAP,
                ..Default::default()
            };
            self.submit_ioctl(
                submit::CREATE_BO_REQUEST,
                &mut cb as *mut _ as *mut libc::c_void,
            )?;
            let handle = cb.handle;
            let mut info = submit::GetBoInfo {
                handle,
                ..Default::default()
            };
            self.submit_ioctl(
                submit::GET_BO_INFO_REQUEST,
                &mut info as *mut _ as *mut libc::c_void,
            )?;
            // The DEV_HEAP must be mmap'd MAP_FIXED at a fixed VA inside the NPU's
            // addressable window (GET_BO_INFO's map_offset is the fixed 0x1_0000_0000
            // DEV_HEAP offset). Without MAP_FIXED the kernel places the heap too high
            // (~0x7f..) and `aie2_hwctx_init`'s firmware host-buffer map is rejected;
            // any moderate 2 MiB-aligned VA (~0x70..-0x7b..) is accepted — XRT does the
            // same. Confirmed against the driver (dev_addr = AIE2_DEVM_BASE, 64-bit DMA).
            let fixed_va = DEV_HEAP_VA as *mut libc::c_void;
            let ptr = unsafe {
                libc::mmap(
                    fixed_va,
                    size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED | libc::MAP_LOCKED | libc::MAP_FIXED,
                    self.fd,
                    info.map_offset as libc::off_t,
                )
            };
            if ptr == libc::MAP_FAILED {
                return Err(XdnaError::Ioctl(std::io::Error::last_os_error()));
            }
            Ok(DeviceBuffer {
                handle,
                ptr: ptr as *mut u8,
                len: size,
                xdna_addr: info.xdna_addr,
            })
        }

        /// Create a buffer object WITHOUT mmap-ing it (for DEV_HEAP / device BOs
        /// that userspace must not map — the firmware maps their physical pages).
        /// Returns `(handle, xdna_addr)`.
        pub fn create_bo(&self, size: usize, bo_type: u32) -> Result<(u32, u64), XdnaError> {
            let mut cb = submit::CreateBo {
                size: size as u64,
                bo_type,
                ..Default::default()
            };
            self.submit_ioctl(
                submit::CREATE_BO_REQUEST,
                &mut cb as *mut _ as *mut libc::c_void,
            )?;
            let handle = cb.handle;
            let mut info = submit::GetBoInfo {
                handle,
                ..Default::default()
            };
            self.submit_ioctl(
                submit::GET_BO_INFO_REQUEST,
                &mut info as *mut _ as *mut libc::c_void,
            )?;
            Ok((handle, info.xdna_addr))
        }

        /// Sync a BO's cache to/from the device (`submit::SYNC_DIRECT_*`).
        pub fn sync_bo(&self, handle: u32, direction: u32, size: usize) -> Result<(), XdnaError> {
            let mut s = submit::SyncBo {
                handle,
                direction,
                offset: 0,
                size: size as u64,
            };
            self.submit_ioctl(
                submit::SYNC_BO_REQUEST,
                &mut s as *mut _ as *mut libc::c_void,
            )
        }

        /// Raw ioctl helper for the submission path: Ok(()) on rc==0 else OS error.
        fn submit_ioctl(&self, request: u64, arg: *mut libc::c_void) -> Result<(), XdnaError> {
            // SAFETY: request matches arg's struct type; arg is a valid writable ptr.
            let rc = unsafe { libc::ioctl(self.fd, request as libc::c_ulong, arg) };
            if rc != 0 {
                return Err(XdnaError::Ioctl(std::io::Error::last_os_error()));
            }
            Ok(())
        }
    }

    /// An amdxdna buffer object created via `CREATE_BO` and mmap'd into this
    /// process. `xdna_addr` is its device virtual address (used in command args).
    /// The BO handle is released when the owning device fd closes.
    pub struct DeviceBuffer {
        handle: u32,
        ptr: *mut u8,
        len: usize,
        xdna_addr: u64,
    }

    impl DeviceBuffer {
        /// The BO handle (for EXEC_CMD arg lists / CONFIG_HWCTX).
        pub fn handle(&self) -> u32 {
            self.handle
        }
        /// Device virtual address of this BO.
        pub fn xdna_addr(&self) -> u64 {
            self.xdna_addr
        }
        /// Host virtual address of the mapping. For SHMEM / CMD BOs this is the
        /// device-accessible address to place in a command regmap (the NPU reaches
        /// host memory at the same VA via PASID); DEV BOs use [`Self::xdna_addr`].
        pub fn host_addr(&self) -> u64 {
            self.ptr as u64
        }
        /// Length in bytes of the mapped region.
        pub fn len(&self) -> usize {
            self.len
        }
        /// Whether the mapped region is empty.
        pub fn is_empty(&self) -> bool {
            self.len == 0
        }
        /// Mutable view of the mapped bytes.
        pub fn as_mut_slice(&mut self) -> &mut [u8] {
            // SAFETY: ptr/len come from a successful mmap of this BO.
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
        /// Read-only view of the mapped bytes.
        pub fn as_slice(&self) -> &[u8] {
            // SAFETY: ptr/len come from a successful mmap of this BO.
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }

    impl Drop for DeviceBuffer {
        fn drop(&mut self) {
            // SAFETY: ptr/len from a successful mmap; unmapped exactly once.
            unsafe {
                libc::munmap(self.ptr as *mut libc::c_void, self.len);
            }
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

#[cfg(target_os = "linux")]
pub use imp::DeviceBuffer;
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
            temp_c: None,
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
