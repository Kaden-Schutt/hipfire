//! W1 — amdxdna command-submission ABI layer (foundation for the W4A8 kernel
//! wire-in; see `docs/npu/wire-in-amdxdna-command-submission.md`).
//!
//! Pure ABI: the `#[repr(C)]` submission structs, the DRM ioctl request numbers,
//! and `size_of` asserts pinned against `/usr/include/drm/amdxdna_accel.h`. The
//! actual submission flow (BO alloc/mmap/sync, xclbin CONFIG_HWCTX, ERT command
//! packet, EXEC_CMD + syncobj wait) is W2–W5 and consumes these types. Kept
//! separate so the tedious, error-prone ABI is landed and asserted on its own.
//!
//! Linux-only: these are amdxdna DRM ioctls on `/dev/accel/accel0`.
#![allow(dead_code)] // W1 is the ABI foundation; W2+ consume these.

/// Linux `_IOC` request encoder (matches the `ioc` in lib.rs).
const fn ioc(dir: u64, typ: u64, nr: u64, size: u64) -> u64 {
    (dir << 30) | (size << 16) | (typ << 8) | nr
}
const DRM_COMMAND_BASE: u64 = 0x40;
const IOC_READ_WRITE: u64 = 3; // _IOC_READ | _IOC_WRITE
const DRM_TYPE: u64 = b'd' as u64;

// enum amdxdna_drm_ioctl_id — the command-submission subset (GET_INFO=7 lives in
// lib.rs). All are DRM_IOWR(DRM_COMMAND_BASE + id, struct).
const DRM_AMDXDNA_CREATE_HWCTX: u64 = 0;
const DRM_AMDXDNA_DESTROY_HWCTX: u64 = 1;
const DRM_AMDXDNA_CONFIG_HWCTX: u64 = 2;
const DRM_AMDXDNA_CREATE_BO: u64 = 3;
const DRM_AMDXDNA_GET_BO_INFO: u64 = 4;
const DRM_AMDXDNA_SYNC_BO: u64 = 5;
const DRM_AMDXDNA_EXEC_CMD: u64 = 6;

macro_rules! iowr {
    ($id:expr, $ty:ty) => {
        ioc(
            IOC_READ_WRITE,
            DRM_TYPE,
            DRM_COMMAND_BASE + $id,
            core::mem::size_of::<$ty>() as u64,
        )
    };
}

pub const CREATE_HWCTX_REQUEST: u64 = iowr!(DRM_AMDXDNA_CREATE_HWCTX, CreateHwctx);
pub const DESTROY_HWCTX_REQUEST: u64 = iowr!(DRM_AMDXDNA_DESTROY_HWCTX, DestroyHwctx);
pub const CONFIG_HWCTX_REQUEST: u64 = iowr!(DRM_AMDXDNA_CONFIG_HWCTX, ConfigHwctx);
pub const CREATE_BO_REQUEST: u64 = iowr!(DRM_AMDXDNA_CREATE_BO, CreateBo);
pub const GET_BO_INFO_REQUEST: u64 = iowr!(DRM_AMDXDNA_GET_BO_INFO, GetBoInfo);
pub const SYNC_BO_REQUEST: u64 = iowr!(DRM_AMDXDNA_SYNC_BO, SyncBo);
pub const EXEC_CMD_REQUEST: u64 = iowr!(DRM_AMDXDNA_EXEC_CMD, ExecCmd);

// enum amdxdna_bo_type
pub const AMDXDNA_BO_INVALID: u32 = 0;
pub const AMDXDNA_BO_SHMEM: u32 = 1;
pub const AMDXDNA_BO_DEV_HEAP: u32 = 2;
pub const AMDXDNA_BO_DEV: u32 = 3;
pub const AMDXDNA_BO_CMD: u32 = 4;

// amdxdna_drm_sync_bo direction
pub const SYNC_DIRECT_TO_DEVICE: u32 = 0;
pub const SYNC_DIRECT_FROM_DEVICE: u32 = 1;

// enum amdxdna_cmd_type
pub const AMDXDNA_CMD_SUBMIT_EXEC_BUF: u32 = 0;
pub const AMDXDNA_CMD_SUBMIT_DEPENDENCY: u32 = 1;
pub const AMDXDNA_CMD_SUBMIT_SIGNAL: u32 = 2;

// enum amdxdna_drm_config_hwctx_param
pub const DRM_AMDXDNA_HWCTX_CONFIG_CU: u32 = 0;

/// struct amdxdna_qos_info — pointed to by `CreateHwctx::qos_p`.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct QosInfo {
    pub gops: u32,
    pub fps: u32,
    pub dma_bandwidth: u32,
    pub latency: u32,
    pub frame_exec_time: u32,
    pub priority: u32,
}

/// struct amdxdna_drm_create_hwctx
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct CreateHwctx {
    pub ext: u64,
    pub ext_flags: u64,
    pub qos_p: u64,
    pub umq_bo: u32,
    pub log_buf_bo: u32,
    pub max_opc: u32,
    pub num_tiles: u32,
    pub mem_size: u32,
    pub umq_doorbell: u32,
    pub handle: u32,         // out
    pub syncobj_handle: u32, // out
}

/// struct amdxdna_drm_destroy_hwctx
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct DestroyHwctx {
    pub handle: u32,
    pub pad: u32,
}

/// struct amdxdna_cu_config
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct CuConfig {
    pub cu_bo: u32,
    pub cu_func: u8,
    pub pad: [u8; 3],
}

/// struct amdxdna_drm_config_hwctx
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct ConfigHwctx {
    pub handle: u32,
    pub param_type: u32,
    pub param_val: u64, // pointer to param struct (e.g. hwctx_param_config_cu)
    pub param_val_size: u32,
    pub pad: u32,
}

/// struct amdxdna_drm_create_bo
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct CreateBo {
    pub flags: u64,
    pub vaddr: u64,
    pub size: u64,
    pub bo_type: u32, // `type` in C
    pub handle: u32,  // out
}

/// struct amdxdna_drm_get_bo_info
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct GetBoInfo {
    pub ext: u64,
    pub ext_flags: u64,
    pub handle: u32,
    pub pad: u32,
    pub map_offset: u64, // out — mmap() offset
    pub vaddr: u64,      // out
    pub xdna_addr: u64,  // out — device VA
}

/// struct amdxdna_drm_sync_bo
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct SyncBo {
    pub handle: u32,
    pub direction: u32,
    pub offset: u64,
    pub size: u64,
}

/// struct amdxdna_drm_exec_cmd
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct ExecCmd {
    pub ext: u64,
    pub ext_flags: u64,
    pub hwctx: u32,
    pub cmd_type: u32, // `type` in C
    pub cmd_handles: u64,
    pub args: u64,
    pub cmd_count: u32,
    pub arg_count: u32,
    pub seq: u64, // out
}

// ABI guards: any drift vs the kernel header is a compile error.
const _: () = assert!(core::mem::size_of::<QosInfo>() == 24);
const _: () = assert!(core::mem::size_of::<CreateHwctx>() == 56);
const _: () = assert!(core::mem::size_of::<DestroyHwctx>() == 8);
const _: () = assert!(core::mem::size_of::<CuConfig>() == 8);
const _: () = assert!(core::mem::size_of::<ConfigHwctx>() == 24);
const _: () = assert!(core::mem::size_of::<CreateBo>() == 32);
const _: () = assert!(core::mem::size_of::<GetBoInfo>() == 48);
const _: () = assert!(core::mem::size_of::<SyncBo>() == 24);
const _: () = assert!(core::mem::size_of::<ExecCmd>() == 56);
