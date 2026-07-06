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

/// `DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT` — a *core* DRM ioctl (no `DRM_COMMAND_BASE`
/// offset), used to block until a submitted command's timeline point signals.
pub const SYNCOBJ_TIMELINE_WAIT_REQUEST: u64 = ioc(
    IOC_READ_WRITE,
    DRM_TYPE,
    0xCA,
    core::mem::size_of::<SyncobjTimelineWait>() as u64,
);

/// `DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT` — wait for the point to be submitted
/// rather than erroring if it hasn't been yet (what XRT passes for a fresh submit).
pub const SYNCOBJ_WAIT_FOR_SUBMIT: u32 = 0x2;

/// `DRM_IOCTL_GEM_CLOSE` — a *core* DRM ioctl (`_IOW('d', 0x09, drm_gem_close)`)
/// that releases a BO handle without closing the device fd.
pub const GEM_CLOSE_REQUEST: u64 = ioc(
    1, // _IOC_WRITE
    DRM_TYPE,
    0x09,
    core::mem::size_of::<GemClose>() as u64,
);

/// struct drm_gem_close (core DRM).
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct GemClose {
    pub handle: u32,
    pub pad: u32,
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

/// struct amdxdna_drm_va_tbl (header only; `va_entries[]` is empty for the dma-buf path).
/// Passed via `CreateBo.vaddr` with `bo_type = AMDXDNA_BO_SHMEM` to import an external
/// dma-buf: a valid `dmabuf_fd` (>= 0) selects the driver's prime-import path, and
/// `num_entries` must be 0. (The driver distinguishes this from a plain SHMEM alloc by
/// `vaddr != 0`.)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VaTbl {
    pub dmabuf_fd: i32,
    pub num_entries: u32,
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

/// Size of the ERT command BO / DPU packet (page-aligned by the driver; XRT uses
/// a 144-byte packet inside a 4 KiB BO).
pub const DPU_CMD_PACKET_LEN: usize = 144;

/// Assemble the ERT "start DPU" command packet issued via `EXEC_CMD`. Layout is
/// captured byte-exact from XRT's amdxdna path (`docs/npu/wire-in-amdxdna-command-submission.md`):
///
/// ```text
///   @0x00 u32  header 0x30010001 (state=NEW, count=16, opcode=START_CU, type=3)
///   @0x04 u32  cu_mask = 0x1  (CU 0)
///   @0x08 u64  opcode = 3     (DPU)
///   @0x10 u64  instruction-buffer device address
///   @0x18 u32  instruction-buffer size in BYTES
///   @0x1c u64* per-arg device-accessible addresses, in kernel-signature order
/// ```
///
/// `arg_addrs` are host VAs for SHMEM BOs (`DeviceBuffer::host_addr`) and
/// `xdna_addr` for DEV BOs. Handles up to 5 args (fits the fixed 15-word regmap).
pub fn dpu_cmd_packet(
    instr_addr: u64,
    instr_size: usize,
    arg_addrs: &[u64],
) -> [u8; DPU_CMD_PACKET_LEN] {
    assert!(arg_addrs.len() <= 5, "DPU regmap holds at most 5 args");
    let mut b = [0u8; DPU_CMD_PACKET_LEN];
    b[0x00..0x04].copy_from_slice(&0x3001_0001u32.to_le_bytes());
    b[0x04..0x08].copy_from_slice(&1u32.to_le_bytes());
    b[0x08..0x10].copy_from_slice(&3u64.to_le_bytes());
    b[0x10..0x18].copy_from_slice(&instr_addr.to_le_bytes());
    b[0x18..0x1c].copy_from_slice(&(instr_size as u32).to_le_bytes());
    let mut off = 0x1c;
    for &a in arg_addrs {
        b[off..off + 8].copy_from_slice(&a.to_le_bytes());
        off += 8;
    }
    b
}

/// struct drm_syncobj_timeline_wait (core DRM, `include/uapi/drm/drm.h`).
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct SyncobjTimelineWait {
    pub handles: u64,       // ptr to array of u32 syncobj handles
    pub points: u64,        // ptr to array of u64 timeline points
    pub timeout_nsec: i64,  // absolute or relative timeout
    pub count_handles: u32, // number of handles/points
    pub flags: u32,
    pub first_signaled: u32, // out
    pub pad: u32,
}

// ABI guards: any drift vs the kernel header is a compile error.
const _: () = assert!(core::mem::size_of::<SyncobjTimelineWait>() == 40);

#[cfg(test)]
mod tests {
    use super::*;

    /// Locks the DPU packet layout against the bytes captured from XRT's amdxdna
    /// path (r2a: instr@0x04028000, ninstr=420, 3 args).
    #[test]
    fn dpu_packet_matches_capture() {
        let p = dpu_cmd_packet(0x0402_8000, 420, &[0x1111_2222, 0x3333_4444, 0x5555_6666]);
        assert_eq!(&p[0x00..0x04], &0x3001_0001u32.to_le_bytes()); // ERT header
        assert_eq!(&p[0x04..0x08], &1u32.to_le_bytes()); // cu_mask
        assert_eq!(&p[0x08..0x10], &3u64.to_le_bytes()); // opcode = DPU
        assert_eq!(&p[0x10..0x18], &0x0402_8000u64.to_le_bytes()); // instr addr
        assert_eq!(&p[0x18..0x1c], &420u32.to_le_bytes()); // ninstr (bytes)
        assert_eq!(&p[0x1c..0x24], &0x1111_2222u64.to_le_bytes()); // arg0
        assert_eq!(&p[0x24..0x2c], &0x3333_4444u64.to_le_bytes()); // arg1
        assert_eq!(&p[0x2c..0x34], &0x5555_6666u64.to_le_bytes()); // arg2
        assert_eq!(&p[0x34..], &[0u8; DPU_CMD_PACKET_LEN - 0x34]); // zero tail
    }

    #[test]
    #[should_panic]
    fn dpu_packet_rejects_too_many_args() {
        dpu_cmd_packet(0, 0, &[0; 6]);
    }
}
const _: () = assert!(core::mem::size_of::<QosInfo>() == 24);
const _: () = assert!(core::mem::size_of::<CreateHwctx>() == 56);
const _: () = assert!(core::mem::size_of::<DestroyHwctx>() == 8);
const _: () = assert!(core::mem::size_of::<CuConfig>() == 8);
const _: () = assert!(core::mem::size_of::<ConfigHwctx>() == 24);
const _: () = assert!(core::mem::size_of::<CreateBo>() == 32);
const _: () = assert!(core::mem::size_of::<GetBoInfo>() == 48);
const _: () = assert!(core::mem::size_of::<SyncBo>() == 24);
const _: () = assert!(core::mem::size_of::<ExecCmd>() == 56);
