// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use crate::{ArtifactBundle, BindingAccess, BindingLayout, FirmwareVersion, Result, XdnaError};
use libc::{c_void, timespec};
use std::fs::{self, File, OpenOptions};
use std::mem::size_of;
use std::os::fd::{AsRawFd, FromRawFd, RawFd};
use std::os::unix::fs::FileTypeExt;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::{Duration, Instant};

const DRM_COMMAND_BASE: u32 = 0x40;
const CREATE_HWCTX: u32 = 0;
const DESTROY_HWCTX: u32 = 1;
const CONFIG_HWCTX: u32 = 2;
const CREATE_BO: u32 = 3;
const GET_BO_INFO: u32 = 4;
const SYNC_BO: u32 = 5;
const EXEC_CMD: u32 = 6;
const GET_INFO: u32 = 7;
const QUERY_AIE_METADATA: u32 = 1;
const QUERY_FIRMWARE_VERSION: u32 = 8;

const BO_SHARE: u32 = 1;
const BO_DEV_HEAP: u32 = 2;
const BO_DEV: u32 = 3;
const BO_CMD: u32 = 4;
const CONFIG_CU: u32 = 0;
const EXEC_BUF: u32 = 0;

const ERT_NEW: u32 = 1;
const ERT_COMPLETED: u32 = 4;
const ERT_START_CU: u32 = 0;
const ERT_CU: u32 = 3;

const DRM_IOCTL_GEM_CLOSE: libc::c_ulong = drm_iow::<DrmGemClose>(0x09);
const DRM_IOCTL_PRIME_FD_TO_HANDLE: libc::c_ulong = drm_iowr_nr::<DrmPrimeHandle>(0x2e);
const DRM_IOCTL_SYNCOBJ_DESTROY: libc::c_ulong = drm_iowr_nr::<DrmSyncobjDestroy>(0xc0);
const DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT: libc::c_ulong = drm_iowr_nr::<SyncobjTimelineWait>(0xca);
const SYNCOBJ_WAIT_FOR_SUBMIT: u32 = 1 << 1;
const DMA_BUF_IOCTL_SYNC: libc::c_ulong = ioctl_code_for::<DmaBufSync>(1, 0x62, 0);
const DMA_BUF_SYNC_READ: u64 = 1 << 0;
const DMA_BUF_SYNC_START: u64 = 0;
const DMA_BUF_SYNC_END: u64 = 1 << 2;

const HEAP_SIZE: usize = 64 << 20;
const HEAP_ALIGNMENT: usize = 64 << 20;
const COMMAND_BUFFER_SIZE: usize = 0x1000;
const DEVICE_ROOT: &str = "/dev/accel";
const SYSFS_ACCEL_ROOT: &str = "/sys/class/accel";

/// Resolves the one real amdxdna character device without manufacturing an
/// alias. An explicit path is still validated against sysfs so callers cannot
/// accidentally pass the `accel0 -> accel1` compatibility symlink that breaks
/// ROCr's basename-to-sysfs pairing.
pub fn resolve_device_path(explicit: Option<&Path>) -> Result<PathBuf> {
    let device_root = Path::new(DEVICE_ROOT);
    let sysfs_root = Path::new(SYSFS_ACCEL_ROOT);
    if let Some(path) = explicit {
        validate_amdxdna_node(path, device_root, sysfs_root)?;
        return Ok(path.to_path_buf());
    }

    let entries = fs::read_dir(device_root).map_err(|source| XdnaError::Io {
        operation: "enumerate amdxdna devices",
        source,
    })?;
    let mut candidates = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if validate_amdxdna_node(&path, device_root, sysfs_root).is_ok() {
            candidates.push(path);
        }
    }
    select_unique_device(candidates)
}

fn validate_amdxdna_node(path: &Path, device_root: &Path, sysfs_root: &Path) -> Result<()> {
    let reject = |message: String| XdnaError::UnsafeDevicePath {
        path: path.to_path_buf(),
        message,
    };
    if path.parent() != Some(device_root) {
        return Err(reject(format!(
            "device must be a direct child of {}",
            device_root.display()
        )));
    }
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| reject("device basename is not valid UTF-8".into()))?;
    let suffix = name
        .strip_prefix("accel")
        .ok_or_else(|| reject("device basename must match accelN".into()))?;
    if suffix.is_empty() || !suffix.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(reject("device basename must match accelN".into()));
    }

    let metadata = fs::symlink_metadata(path).map_err(|source| XdnaError::Io {
        operation: "inspect amdxdna device",
        source,
    })?;
    if metadata.file_type().is_symlink() {
        return Err(reject(
            "symlinks are forbidden because ROCr pairs accel basenames with sysfs".into(),
        ));
    }
    if !metadata.file_type().is_char_device() {
        return Err(reject("device is not a character device".into()));
    }

    let driver =
        fs::canonicalize(sysfs_root.join(name).join("device/driver")).map_err(|source| {
            XdnaError::Io {
                operation: "resolve amdxdna sysfs driver",
                source,
            }
        })?;
    if driver.file_name().and_then(|value| value.to_str()) != Some("amdxdna") {
        return Err(reject(format!(
            "sysfs driver is {}, expected amdxdna",
            driver.display()
        )));
    }
    Ok(())
}

fn select_unique_device(mut candidates: Vec<PathBuf>) -> Result<PathBuf> {
    candidates.sort();
    match candidates.as_slice() {
        [path] => Ok(path.clone()),
        [] => Err(XdnaError::DeviceDiscovery(
            "no real amdxdna accelN character device is available".into(),
        )),
        paths => Err(XdnaError::DeviceDiscovery(format!(
            "multiple amdxdna devices are available ({}); set HIPFIRE_XDNA_DEVICE to one real node",
            paths
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ))),
    }
}

const fn ioctl_code_for<T>(direction: u64, ioctl_type: u64, nr: u32) -> libc::c_ulong {
    ((direction << 30) | ((size_of::<T>() as u64) << 16) | (ioctl_type << 8) | nr as u64)
        as libc::c_ulong
}

const fn ioctl_code<T>(direction: u64, nr: u32) -> libc::c_ulong {
    ioctl_code_for::<T>(direction, 0x64, nr)
}

const fn drm_iow<T>(nr: u32) -> libc::c_ulong {
    ioctl_code::<T>(1, nr)
}

const fn drm_iowr_nr<T>(nr: u32) -> libc::c_ulong {
    ioctl_code::<T>(3, nr)
}

const fn drm_iowr<T>(command: u32) -> libc::c_ulong {
    drm_iowr_nr::<T>(DRM_COMMAND_BASE + command)
}

#[repr(C)]
#[derive(Default)]
struct QosInfo {
    gops: u32,
    fps: u32,
    dma_bandwidth: u32,
    latency: u32,
    frame_exec_time: u32,
    priority: u32,
}

#[repr(C)]
#[derive(Default)]
struct CreateHwctx {
    ext: u64,
    ext_flags: u64,
    qos_p: u64,
    umq_bo: u32,
    log_buf_bo: u32,
    max_opc: u32,
    num_tiles: u32,
    mem_size: u32,
    umq_doorbell: u32,
    handle: u32,
    syncobj_handle: u32,
}

#[repr(C)]
#[derive(Default)]
struct DestroyHwctx {
    handle: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Default)]
struct ConfigHwctx {
    handle: u32,
    param_type: u32,
    param_val: u64,
    param_val_size: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Default)]
struct CreateBo {
    flags: u64,
    vaddr: u64,
    size: u64,
    kind: u32,
    handle: u32,
}

#[repr(C)]
#[derive(Default)]
struct GetBoInfo {
    ext: u64,
    ext_flags: u64,
    handle: u32,
    pad: u32,
    map_offset: u64,
    vaddr: u64,
    xdna_addr: u64,
}

#[repr(C)]
#[derive(Default)]
struct SyncBo {
    handle: u32,
    direction: u32,
    offset: u64,
    size: u64,
}

#[repr(C)]
#[derive(Default)]
struct ExecCmd {
    ext: u64,
    ext_flags: u64,
    hwctx: u32,
    kind: u32,
    cmd_handles: u64,
    args: u64,
    cmd_count: u32,
    arg_count: u32,
    seq: u64,
}

#[repr(C)]
#[derive(Default)]
struct GetInfo {
    param: u32,
    buffer_size: u32,
    buffer: u64,
}

#[repr(C)]
#[derive(Default)]
struct SyncobjTimelineWait {
    handles: u64,
    points: u64,
    timeout_nsec: i64,
    count_handles: u32,
    flags: u32,
    first_signaled: u32,
    pad: u32,
    deadline_nsec: u64,
}

#[repr(C)]
#[derive(Default)]
struct DrmGemClose {
    handle: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Default)]
struct DrmSyncobjDestroy {
    handle: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Default)]
struct DrmPrimeHandle {
    handle: u32,
    flags: u32,
    fd: i32,
}

#[repr(C)]
struct DmaBufSync {
    flags: u64,
}

fn ioctl<T>(
    fd: RawFd,
    request: libc::c_ulong,
    value: &mut T,
    operation: &'static str,
) -> Result<()> {
    let status = unsafe { libc::ioctl(fd, request, value as *mut T as *mut c_void) };
    if status < 0 {
        Err(XdnaError::ioctl(operation))
    } else {
        Ok(())
    }
}

fn gem_close(fd: RawFd, handle: u32) {
    if handle == 0 {
        return;
    }
    let mut close = DrmGemClose { handle, pad: 0 };
    unsafe {
        libc::ioctl(fd, DRM_IOCTL_GEM_CLOSE, &mut close);
    }
}

struct DeviceIo {
    file: File,
}

impl DeviceIo {
    fn fd(&self) -> RawFd {
        self.file.as_raw_fd()
    }
}

struct Mapping {
    base: NonNull<u8>,
    length: usize,
    data: NonNull<u8>,
    data_length: usize,
}

unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}

impl Mapping {
    fn map_bo(fd: RawFd, offset: u64, length: usize) -> Result<Self> {
        Self::map_bo_with_flags(fd, offset, length, libc::MAP_SHARED | libc::MAP_LOCKED)
    }

    fn map_imported_bo(fd: RawFd, offset: u64, length: usize) -> Result<Self> {
        // Imported model weights remain GPU-owned and can be much larger than
        // RLIMIT_MEMLOCK. A plain shared VMA is sufficient to establish the
        // PASID address; no pages are copied or prefaulted.
        Self::map_bo_with_flags(fd, offset, length, libc::MAP_SHARED)
    }

    fn map_bo_with_flags(
        fd: RawFd,
        offset: u64,
        length: usize,
        flags: libc::c_int,
    ) -> Result<Self> {
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length,
                libc::PROT_READ | libc::PROT_WRITE,
                flags,
                fd,
                offset as libc::off_t,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(XdnaError::io("mmap amdxdna BO"));
        }
        let data = NonNull::new(ptr.cast::<u8>())
            .ok_or_else(|| XdnaError::InvalidResponse("mmap returned null".into()))?;
        Ok(Self {
            base: data,
            length,
            data,
            data_length: length,
        })
    }

    fn map_aligned_heap(fd: RawFd, offset: u64) -> Result<Self> {
        let reserve_length = HEAP_SIZE + HEAP_ALIGNMENT;
        let reserve = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                reserve_length,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if reserve == libc::MAP_FAILED {
            return Err(XdnaError::io("reserve aligned amdxdna heap VA"));
        }
        let reserve_ptr = NonNull::new(reserve.cast::<u8>())
            .ok_or_else(|| XdnaError::InvalidResponse("heap reservation returned null".into()))?;
        let aligned_address = (reserve as usize + HEAP_ALIGNMENT - 1) & !(HEAP_ALIGNMENT - 1);
        let mapped = unsafe {
            libc::mmap(
                aligned_address as *mut c_void,
                HEAP_SIZE,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_LOCKED | libc::MAP_FIXED,
                fd,
                offset as libc::off_t,
            )
        };
        if mapped == libc::MAP_FAILED {
            unsafe {
                libc::munmap(reserve, reserve_length);
            }
            return Err(XdnaError::io("mmap aligned amdxdna heap"));
        }
        let data = NonNull::new(mapped.cast::<u8>())
            .ok_or_else(|| XdnaError::InvalidResponse("heap mmap returned null".into()))?;
        Ok(Self {
            base: reserve_ptr,
            length: reserve_length,
            data,
            data_length: HEAP_SIZE,
        })
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.base.as_ptr().cast(), self.length);
        }
    }
}

struct DeviceState {
    io: Arc<DeviceIo>,
    heap_handle: u32,
    heap_address: u64,
    heap: Mapping,
    imported_handles: Mutex<std::collections::HashMap<u32, Weak<BoResource>>>,
}

impl Drop for DeviceState {
    fn drop(&mut self) {
        gem_close(self.io.fd(), self.heap_handle);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceMetadata {
    pub columns: u16,
    pub core_rows: u16,
    pub tiles: u32,
    pub firmware: FirmwareVersion,
}

#[derive(Clone)]
pub struct Device {
    state: Arc<DeviceState>,
    metadata: DeviceMetadata,
}

impl Device {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        validate_amdxdna_node(path, Path::new(DEVICE_ROOT), Path::new(SYSFS_ACCEL_ROOT))?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|source| XdnaError::Io {
                operation: "open amdxdna device",
                source,
            })?;
        let io = Arc::new(DeviceIo { file });

        let mut metadata_bytes = [0_u8; 64];
        let mut metadata_query = GetInfo {
            param: QUERY_AIE_METADATA,
            buffer_size: metadata_bytes.len() as u32,
            buffer: metadata_bytes.as_mut_ptr() as u64,
        };
        ioctl(
            io.fd(),
            drm_iowr::<GetInfo>(GET_INFO),
            &mut metadata_query,
            "query AIE metadata",
        )?;
        let columns = u16::from_le_bytes([metadata_bytes[4], metadata_bytes[5]]);
        let core_rows = u16::from_le_bytes([metadata_bytes[16], metadata_bytes[17]]);
        let tiles = u32::from(columns) * u32::from(core_rows);
        if tiles == 0 {
            return Err(XdnaError::InvalidResponse(format!(
                "AIE metadata reported columns={columns}, core_rows={core_rows}"
            )));
        }

        let mut firmware_words = [0_u32; 4];
        let mut firmware_query = GetInfo {
            param: QUERY_FIRMWARE_VERSION,
            buffer_size: size_of::<[u32; 4]>() as u32,
            buffer: firmware_words.as_mut_ptr() as u64,
        };
        ioctl(
            io.fd(),
            drm_iowr::<GetInfo>(GET_INFO),
            &mut firmware_query,
            "query firmware version",
        )?;
        let firmware = FirmwareVersion {
            major: firmware_words[0],
            minor: firmware_words[1],
            patch: firmware_words[2],
            build: firmware_words[3],
        };

        let mut heap = CreateBo {
            size: HEAP_SIZE as u64,
            kind: BO_DEV_HEAP,
            ..Default::default()
        };
        ioctl(
            io.fd(),
            drm_iowr::<CreateBo>(CREATE_BO),
            &mut heap,
            "create device heap",
        )?;
        let mut heap_info = GetBoInfo {
            handle: heap.handle,
            ..Default::default()
        };
        if let Err(error) = ioctl(
            io.fd(),
            drm_iowr::<GetBoInfo>(GET_BO_INFO),
            &mut heap_info,
            "query device heap",
        ) {
            gem_close(io.fd(), heap.handle);
            return Err(error);
        }
        let mapping = match Mapping::map_aligned_heap(io.fd(), heap_info.map_offset) {
            Ok(mapping) => mapping,
            Err(error) => {
                gem_close(io.fd(), heap.handle);
                return Err(error);
            }
        };
        Ok(Self {
            state: Arc::new(DeviceState {
                io,
                heap_handle: heap.handle,
                heap_address: heap_info.xdna_addr,
                heap: mapping,
                imported_handles: Mutex::new(std::collections::HashMap::new()),
            }),
            metadata: DeviceMetadata {
                columns,
                core_rows,
                tiles,
                firmware,
            },
        })
    }

    pub fn metadata(&self) -> DeviceMetadata {
        self.metadata
    }

    pub fn create_context(&self, max_operations_per_cycle: u32) -> Result<HardwareContext> {
        let qos = QosInfo::default();
        let mut create = CreateHwctx {
            qos_p: (&qos as *const QosInfo) as u64,
            max_opc: max_operations_per_cycle,
            num_tiles: self.metadata.tiles,
            ..Default::default()
        };
        ioctl(
            self.state.io.fd(),
            drm_iowr::<CreateHwctx>(CREATE_HWCTX),
            &mut create,
            "create hardware context",
        )?;
        Ok(HardwareContext {
            inner: Arc::new(ContextState {
                device: self.clone(),
                handle: create.handle,
                syncobj: create.syncobj_handle,
            }),
        })
    }

    pub fn import_dmabuf(&self, fd: RawFd, offset: u64, length: usize) -> Result<Bo> {
        if fd < 0 || length == 0 {
            return Err(XdnaError::InvalidResponse(
                "dma-buf import requires a valid fd and nonzero length".into(),
            ));
        }
        let mut import = DrmPrimeHandle {
            fd,
            ..Default::default()
        };
        ioctl(
            self.state.io.fd(),
            DRM_IOCTL_PRIME_FD_TO_HANDLE,
            &mut import,
            "import dma-buf",
        )?;
        let info = match get_bo_info(&self.state.io, import.handle) {
            Ok(info) => info,
            Err(error) => {
                gem_close(self.state.io.fd(), import.handle);
                return Err(error);
            }
        };
        let exported_length = unsafe { libc::lseek(fd, 0, libc::SEEK_END) };
        if exported_length < 0 {
            gem_close(self.state.io.fd(), import.handle);
            return Err(XdnaError::io("query dma-buf length"));
        }
        let exported_length = exported_length as usize;
        let offset_usize = usize::try_from(offset).map_err(|_| {
            gem_close(self.state.io.fd(), import.handle);
            XdnaError::InvalidResponse("dma-buf offset does not fit usize".into())
        })?;
        if offset_usize
            .checked_add(length)
            .is_none_or(|end| end > exported_length)
        {
            gem_close(self.state.io.fd(), import.handle);
            return Err(XdnaError::BufferRange {
                offset,
                length,
                bo_length: exported_length,
            });
        }
        let resource = {
            let mut resources = self
                .state
                .imported_handles
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(resource) = resources.get(&import.handle).and_then(Weak::upgrade) {
                resource
            } else {
                let dma_buf = match duplicate_fd(fd) {
                    Ok(dma_buf) => dma_buf,
                    Err(error) => {
                        gem_close(self.state.io.fd(), import.handle);
                        return Err(error);
                    }
                };
                let mapping = match Mapping::map_imported_bo(
                    self.state.io.fd(),
                    info.map_offset,
                    exported_length,
                ) {
                    Ok(mapping) => mapping,
                    Err(error) => {
                        gem_close(self.state.io.fd(), import.handle);
                        return Err(error);
                    }
                };
                let resource = Arc::new(BoResource {
                    io: self.state.io.clone(),
                    handle: import.handle,
                    mapping: Some(mapping),
                    dma_buf: Some(dma_buf),
                });
                resources.insert(import.handle, Arc::downgrade(&resource));
                resource
            }
        };
        let base = resource
            .mapping
            .as_ref()
            .expect("imported dma-buf resource retains its mapping")
            .data
            .as_ptr();
        let address = unsafe { base.add(offset_usize) } as u64;
        Ok(Bo {
            resource,
            address,
            length,
            kind: BoKind::Imported,
            backing_offset: offset,
        })
    }

    fn create_dev_bo(&self, bytes: &[u8]) -> Result<Bo> {
        if bytes.is_empty() {
            return Err(XdnaError::InvalidResponse(
                "device BO payload must not be empty".into(),
            ));
        }
        let mut create = CreateBo {
            size: bytes.len() as u64,
            kind: BO_DEV,
            ..Default::default()
        };
        ioctl(
            self.state.io.fd(),
            drm_iowr::<CreateBo>(CREATE_BO),
            &mut create,
            "create device BO",
        )?;
        let info = match get_bo_info(&self.state.io, create.handle) {
            Ok(info) => info,
            Err(error) => {
                gem_close(self.state.io.fd(), create.handle);
                return Err(error);
            }
        };
        let heap_offset = info
            .xdna_addr
            .checked_sub(self.state.heap_address)
            .ok_or_else(|| {
                gem_close(self.state.io.fd(), create.handle);
                XdnaError::InvalidResponse("device BO is below heap base".into())
            })? as usize;
        if heap_offset
            .checked_add(bytes.len())
            .is_none_or(|end| end > self.state.heap.data_length)
        {
            gem_close(self.state.io.fd(), create.handle);
            return Err(XdnaError::InvalidResponse(format!(
                "device BO range {heap_offset}+{} exceeds heap",
                bytes.len()
            )));
        }
        unsafe {
            let destination = self.state.heap.data.as_ptr().add(heap_offset);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), destination, bytes.len());
            flush_range(destination, bytes.len());
        }
        Ok(Bo {
            resource: Arc::new(BoResource {
                io: self.state.io.clone(),
                handle: create.handle,
                mapping: None,
                dma_buf: None,
            }),
            address: info.xdna_addr,
            length: bytes.len(),
            kind: BoKind::Device,
            backing_offset: 0,
        })
    }

    fn create_mapped_bo(&self, length: usize, kind: BoKind) -> Result<Bo> {
        let raw_kind = match kind {
            BoKind::Shared => BO_SHARE,
            BoKind::Command => BO_CMD,
            _ => {
                return Err(XdnaError::InvalidResponse(
                    "only shared and command BOs may be mapped".into(),
                ));
            }
        };
        let mut create = CreateBo {
            size: length as u64,
            kind: raw_kind,
            ..Default::default()
        };
        ioctl(
            self.state.io.fd(),
            drm_iowr::<CreateBo>(CREATE_BO),
            &mut create,
            "create mapped BO",
        )?;
        let initial_info = match get_bo_info(&self.state.io, create.handle) {
            Ok(info) => info,
            Err(error) => {
                gem_close(self.state.io.fd(), create.handle);
                return Err(error);
            }
        };
        let mapping = match Mapping::map_bo(self.state.io.fd(), initial_info.map_offset, length) {
            Ok(mapping) => mapping,
            Err(error) => {
                gem_close(self.state.io.fd(), create.handle);
                return Err(error);
            }
        };
        let address = mapping.data.as_ptr() as u64;
        Ok(Bo {
            resource: Arc::new(BoResource {
                io: self.state.io.clone(),
                handle: create.handle,
                mapping: Some(mapping),
                dma_buf: None,
            }),
            address,
            length,
            kind,
            backing_offset: 0,
        })
    }
}

fn get_bo_info(io: &DeviceIo, handle: u32) -> Result<GetBoInfo> {
    let mut info = GetBoInfo {
        handle,
        ..Default::default()
    };
    ioctl(
        io.fd(),
        drm_iowr::<GetBoInfo>(GET_BO_INFO),
        &mut info,
        "query BO",
    )?;
    Ok(info)
}

struct ContextState {
    device: Device,
    handle: u32,
    syncobj: u32,
}

impl Drop for ContextState {
    fn drop(&mut self) {
        let fd = self.device.state.io.fd();
        let mut destroy_context = DestroyHwctx {
            handle: self.handle,
            pad: 0,
        };
        unsafe {
            libc::ioctl(
                fd,
                drm_iowr::<DestroyHwctx>(DESTROY_HWCTX),
                &mut destroy_context,
            );
        }
        let mut destroy_syncobj = DrmSyncobjDestroy {
            handle: self.syncobj,
            pad: 0,
        };
        unsafe {
            libc::ioctl(fd, DRM_IOCTL_SYNCOBJ_DESTROY, &mut destroy_syncobj);
        }
    }
}

#[derive(Clone)]
pub struct HardwareContext {
    inner: Arc<ContextState>,
}

impl HardwareContext {
    pub fn load_program(&self, bundle: &ArtifactBundle) -> Result<Program> {
        let pdi = self.inner.device.create_dev_bo(&bundle.pdi)?;
        let instructions = self.inner.device.create_dev_bo(&bundle.instructions)?;
        let mut configuration = [0_u8; 16];
        configuration[0..2].copy_from_slice(&1_u16.to_le_bytes());
        configuration[8..12].copy_from_slice(&pdi.handle().to_le_bytes());
        let mut config = ConfigHwctx {
            handle: self.inner.handle,
            param_type: CONFIG_CU,
            param_val: configuration.as_ptr() as u64,
            param_val_size: configuration.len() as u32,
            ..Default::default()
        };
        ioctl(
            self.inner.device.state.io.fd(),
            drm_iowr::<ConfigHwctx>(CONFIG_HWCTX),
            &mut config,
            "configure compute unit",
        )?;
        Ok(Program {
            context: self.clone(),
            pdi,
            instructions,
            instruction_count: bundle.manifest.instruction_count,
            artifact_id: bundle.manifest.artifact_id.clone(),
            arithmetic: bundle.manifest.arithmetic,
            bindings: bundle.manifest.bindings.clone(),
            shapes: bundle.manifest.shapes.clone(),
        })
    }

    pub fn command_ring(&self, slots: usize) -> Result<CommandRing> {
        if slots == 0 {
            return Err(XdnaError::InvalidResponse(
                "command ring must have at least one slot".into(),
            ));
        }
        let mut buffers = Vec::with_capacity(slots);
        for _ in 0..slots {
            buffers.push(Mutex::new(
                self.inner
                    .device
                    .create_mapped_bo(COMMAND_BUFFER_SIZE, BoKind::Command)?,
            ));
        }
        Ok(CommandRing {
            context: self.clone(),
            buffers,
            next: AtomicUsize::new(0),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoKind {
    Imported,
    Shared,
    Device,
    Command,
}

struct BoResource {
    io: Arc<DeviceIo>,
    handle: u32,
    mapping: Option<Mapping>,
    dma_buf: Option<File>,
}

impl Drop for BoResource {
    fn drop(&mut self) {
        self.mapping.take();
        gem_close(self.io.fd(), self.handle);
    }
}

pub struct Bo {
    resource: Arc<BoResource>,
    address: u64,
    length: usize,
    kind: BoKind,
    backing_offset: u64,
}

unsafe impl Send for Bo {}
unsafe impl Sync for Bo {}

impl Bo {
    pub fn address(&self) -> u64 {
        self.address
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn kind(&self) -> BoKind {
        self.kind
    }

    /// Makes a BO range resident and visible before an NPU submission.
    ///
    /// There is intentionally no symmetric `FROM_DEVICE` API. On the
    /// currently supported amdxdna driver that direction enters the
    /// hardware-context debug-BO path and can fault the kernel for ordinary
    /// buffers. Output handoff must use submission completion plus dma-buf CPU
    /// access or a separately certified explicit GPU fence.
    pub fn sync_to_device(&self, offset: u64, length: usize) -> Result<()> {
        validate_buffer_range(self.length, offset, length)?;
        let mut sync = SyncBo {
            handle: self.handle(),
            direction: 0,
            offset: self
                .backing_offset
                .checked_add(offset)
                .ok_or_else(|| XdnaError::InvalidResponse("sync offset overflow".into()))?,
            size: length as u64,
        };
        ioctl(
            self.resource.io.fd(),
            drm_iowr::<SyncBo>(SYNC_BO),
            &mut sync,
            "synchronize BO",
        )
    }

    /// Reads a CPU-mapped BO after synchronizing dma-buf ownership for CPU access.
    ///
    /// This proves device output without relying on a HIP cache transition.
    /// It is not a substitute for the explicit fence/cache handoff required
    /// before a subsequent HIP kernel consumes XDNA-written bytes.
    pub fn read_mapped(&self, offset: u64, destination: &mut [u8]) -> Result<()> {
        let pointer = self.mapped_range(offset, destination.len())?;
        let dma_buf = self.resource.dma_buf.as_ref();
        if let Some(dma_buf) = dma_buf {
            dma_buf_sync(dma_buf.as_raw_fd(), DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ)?;
        }
        unsafe {
            flush_range(pointer.as_ptr(), destination.len());
            std::ptr::copy_nonoverlapping(
                pointer.as_ptr(),
                destination.as_mut_ptr(),
                destination.len(),
            );
        }
        if let Some(dma_buf) = dma_buf {
            dma_buf_sync(dma_buf.as_raw_fd(), DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ)?;
        }
        Ok(())
    }

    /// Invalidates CPU cache lines for a mapped range after device execution.
    ///
    /// Callers that need the bytes should prefer [`Bo::read_mapped`], which
    /// copies them while the dma-buf CPU-access interval is open. Completing
    /// this interval also gives the exporting driver an opportunity to perform
    /// its cache/domain transition before later GPU work is submitted.
    pub fn invalidate_cpu_cache(&self, offset: u64, length: usize) -> Result<()> {
        let pointer = self.mapped_range(offset, length)?;
        let dma_buf = self.resource.dma_buf.as_ref();
        if let Some(dma_buf) = dma_buf {
            dma_buf_sync(dma_buf.as_raw_fd(), DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ)?;
        }
        unsafe {
            flush_range(pointer.as_ptr(), length);
        }
        if let Some(dma_buf) = dma_buf {
            dma_buf_sync(dma_buf.as_raw_fd(), DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ)?;
        }
        Ok(())
    }

    fn mapped_range(&self, offset: u64, length: usize) -> Result<NonNull<u8>> {
        let relative = validate_buffer_range(self.length, offset, length)?;
        let backing_offset =
            usize::try_from(self.backing_offset).map_err(|_| XdnaError::BufferRange {
                offset,
                length,
                bo_length: self.length,
            })?;
        let absolute = backing_offset
            .checked_add(relative)
            .ok_or_else(|| XdnaError::InvalidResponse("mapped BO offset overflow".into()))?;
        let mapping = self
            .resource
            .mapping
            .as_ref()
            .ok_or_else(|| XdnaError::InvalidResponse("BO is not CPU mapped".into()))?;
        if absolute
            .checked_add(length)
            .is_none_or(|end| end > mapping.data_length)
        {
            return Err(XdnaError::BufferRange {
                offset,
                length,
                bo_length: self.length,
            });
        }
        NonNull::new(unsafe { mapping.data.as_ptr().add(absolute) })
            .ok_or_else(|| XdnaError::InvalidResponse("mapped BO range returned null".into()))
    }

    fn mapped_ptr(&self) -> Result<NonNull<u8>> {
        self.resource
            .mapping
            .as_ref()
            .map(|mapping| mapping.data)
            .ok_or_else(|| XdnaError::InvalidResponse("BO is not CPU mapped".into()))
    }

    fn handle(&self) -> u32 {
        self.resource.handle
    }
}

fn validate_buffer_range(bo_length: usize, offset: u64, length: usize) -> Result<usize> {
    let offset_usize = usize::try_from(offset).map_err(|_| XdnaError::BufferRange {
        offset,
        length,
        bo_length,
    })?;
    if offset_usize
        .checked_add(length)
        .is_none_or(|end| end > bo_length)
    {
        return Err(XdnaError::BufferRange {
            offset,
            length,
            bo_length,
        });
    }
    Ok(offset_usize)
}

fn duplicate_fd(fd: RawFd) -> Result<File> {
    let duplicate = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 0) };
    if duplicate < 0 {
        return Err(XdnaError::io("duplicate dma-buf fd"));
    }
    // SAFETY: F_DUPFD_CLOEXEC returned a new descriptor owned by this process.
    Ok(unsafe { File::from_raw_fd(duplicate) })
}

fn dma_buf_sync(fd: RawFd, flags: u64) -> Result<()> {
    let mut sync = DmaBufSync { flags };
    ioctl(
        fd,
        DMA_BUF_IOCTL_SYNC,
        &mut sync,
        "synchronize dma-buf CPU access",
    )
}

pub struct Program {
    context: HardwareContext,
    #[allow(dead_code)]
    pdi: Bo,
    instructions: Bo,
    instruction_count: u32,
    artifact_id: String,
    arithmetic: crate::ProjectionArithmetic,
    bindings: Vec<BindingLayout>,
    shapes: Vec<crate::ProjectionShape>,
}

impl Program {
    pub fn artifact_id(&self) -> &str {
        &self.artifact_id
    }

    pub fn arithmetic(&self) -> crate::ProjectionArithmetic {
        self.arithmetic
    }

    pub fn instruction_count(&self) -> u32 {
        self.instruction_count
    }

    pub fn supports_shape(&self, k: u32, n: u32, batch: u32) -> bool {
        batch > 0
            && self.shapes.iter().any(|shape| {
                shape.k == k
                    && (shape.n == n || (shape.masked_output_tail && n < shape.n))
                    && (shape.max_batch == batch
                        || (shape.masked_batch_tail && batch < shape.max_batch))
            })
    }
}

#[derive(Clone, Copy)]
pub struct Binding<'a> {
    pub bo: &'a Bo,
    pub offset: u64,
    pub length: usize,
    pub access: BindingAccess,
}

impl<'a> Binding<'a> {
    pub fn whole(bo: &'a Bo) -> Self {
        Self {
            bo,
            offset: 0,
            length: bo.len(),
            access: BindingAccess::ReadWrite,
        }
    }

    pub fn with_access(mut self, access: BindingAccess) -> Self {
        self.access = access;
        self
    }

    fn address(self) -> Result<u64> {
        let offset = usize::try_from(self.offset).map_err(|_| XdnaError::BufferRange {
            offset: self.offset,
            length: self.length,
            bo_length: self.bo.len(),
        })?;
        if offset
            .checked_add(self.length)
            .is_none_or(|end| end > self.bo.len())
        {
            return Err(XdnaError::BufferRange {
                offset: self.offset,
                length: self.length,
                bo_length: self.bo.len(),
            });
        }
        self.bo
            .address
            .checked_add(self.offset)
            .ok_or_else(|| XdnaError::InvalidResponse("binding address overflow".into()))
    }
}

pub struct CommandRing {
    context: HardwareContext,
    buffers: Vec<Mutex<Bo>>,
    next: AtomicUsize,
}

impl CommandRing {
    pub fn len(&self) -> usize {
        self.buffers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }

    pub fn submit<'a>(
        &'a self,
        program: &'a Program,
        bindings: &[Binding<'_>],
    ) -> Result<SubmissionTicket<'a>> {
        if !Arc::ptr_eq(&self.context.inner, &program.context.inner) {
            return Err(XdnaError::InvalidResponse(
                "program and command ring use different hardware contexts".into(),
            ));
        }
        if bindings.len() > 5 {
            return Err(XdnaError::TooManyBindings(bindings.len()));
        }
        validate_binding_abi(&program.bindings, bindings)?;
        let index = self.next.fetch_add(1, Ordering::Relaxed) % self.buffers.len();
        let command = self.buffers[index]
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        write_ert_packet(
            &command,
            &program.instructions,
            program.instruction_count,
            bindings,
        )?;

        let mut handles = Vec::with_capacity(1 + bindings.len());
        handles.push(program.instructions.handle());
        for handle in bindings.iter().map(|binding| binding.bo.handle()) {
            if !handles.contains(&handle) {
                handles.push(handle);
            }
        }
        let mut execute = ExecCmd {
            hwctx: self.context.inner.handle,
            kind: EXEC_BUF,
            cmd_handles: command.handle() as u64,
            args: handles.as_ptr() as u64,
            cmd_count: 1,
            arg_count: handles.len() as u32,
            ..Default::default()
        };
        let started = Instant::now();
        ioctl(
            self.context.inner.device.state.io.fd(),
            drm_iowr::<ExecCmd>(EXEC_CMD),
            &mut execute,
            "submit command",
        )?;
        Ok(SubmissionTicket {
            context: &self.context,
            command: Some(command),
            sequence: execute.seq,
            started,
            finished: false,
        })
    }
}

fn validate_binding_abi(
    expected_bindings: &[BindingLayout],
    bindings: &[Binding<'_>],
) -> Result<()> {
    if bindings.len() != expected_bindings.len() {
        return Err(XdnaError::BindingCount {
            expected: expected_bindings.len(),
            actual: bindings.len(),
        });
    }
    for (index, (binding, expected)) in bindings.iter().zip(expected_bindings).enumerate() {
        if binding.access != expected.access {
            return Err(XdnaError::BindingAccess {
                index,
                name: expected.name.clone(),
                expected: expected.access,
                actual: binding.access,
            });
        }
        if (binding.length as u128) < u128::from(expected.minimum_bytes) {
            return Err(XdnaError::BindingTooSmall {
                index,
                name: expected.name.clone(),
                minimum: expected.minimum_bytes,
                actual: binding.length,
            });
        }
        let address = binding.address()?;
        if address % u64::from(expected.alignment) != 0 {
            return Err(XdnaError::BindingAlignment {
                index,
                name: expected.name.clone(),
                address,
                alignment: expected.alignment,
            });
        }
    }
    Ok(())
}

fn write_ert_packet(
    command: &Bo,
    instructions: &Bo,
    instruction_count: u32,
    bindings: &[Binding<'_>],
) -> Result<()> {
    let pointer = command.mapped_ptr()?.as_ptr().cast::<u32>();
    let mut register_map = [0_u32; 15];
    register_map[0] = 3;
    register_map[2] = instructions.address as u32;
    register_map[3] = (instructions.address >> 32) as u32;
    register_map[4] = instruction_count;
    for (index, binding) in bindings.iter().copied().enumerate() {
        let address = binding.address()?;
        register_map[5 + index * 2] = address as u32;
        register_map[6 + index * 2] = (address >> 32) as u32;
    }
    let count = 1 + register_map.len() as u32;
    let header = ert_header(ERT_NEW, 0, count, ERT_START_CU, ERT_CU);
    unsafe {
        pointer.write(header);
        pointer.add(1).write(1);
        for (index, word) in register_map.iter().enumerate() {
            pointer.add(2 + index).write(*word);
        }
        flush_range(pointer.cast(), (2 + register_map.len()) * size_of::<u32>());
    }
    Ok(())
}

const fn ert_header(state: u32, extra: u32, count: u32, opcode: u32, kind: u32) -> u32 {
    (state & 0xf)
        | ((extra & 0x3) << 10)
        | ((count & 0x7ff) << 12)
        | ((opcode & 0x1f) << 23)
        | ((kind & 0xf) << 28)
}

#[derive(Clone, Copy, Debug)]
pub struct SubmissionTiming {
    pub elapsed: Duration,
    pub sequence: u64,
}

#[must_use = "an XDNA submission ticket must be waited; abandoning it quarantines its ring slot"]
pub struct SubmissionTicket<'a> {
    context: &'a HardwareContext,
    command: Option<MutexGuard<'a, Bo>>,
    sequence: u64,
    started: Instant,
    finished: bool,
}

impl SubmissionTicket<'_> {
    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn wait(mut self, timeout: Duration) -> Result<SubmissionTiming> {
        self.finish(timeout)
    }

    fn finish(&mut self, timeout: Duration) -> Result<SubmissionTiming> {
        wait_syncobj(
            self.context.inner.device.state.io.fd(),
            self.context.inner.syncobj,
            self.sequence,
            timeout,
        )?;
        let command = self
            .command
            .as_ref()
            .expect("submission command guard is present until completion");
        let pointer = command.mapped_ptr()?.as_ptr();
        unsafe {
            flush_range(pointer, size_of::<u32>());
        }
        let state = unsafe { pointer.cast::<u32>().read_volatile() } & 0xf;
        if state != ERT_COMPLETED {
            return Err(XdnaError::TerminalState { state });
        }
        self.finished = true;
        self.command.take();
        Ok(SubmissionTiming {
            elapsed: self.started.elapsed(),
            sequence: self.sequence,
        })
    }
}

impl Drop for SubmissionTicket<'_> {
    fn drop(&mut self) {
        if !self.finished {
            // Never issue a surprise wait or teardown ioctl from Drop after a
            // timeout/error. Keep this ring slot permanently locked; the
            // controller quarantines the entire ReadyState and leaves cleanup
            // to process exit, where the kernel closes the descriptors.
            if let Some(command) = self.command.take() {
                std::mem::forget(command);
            }
        }
    }
}

fn wait_syncobj(fd: RawFd, syncobj: u32, sequence: u64, timeout: Duration) -> Result<()> {
    let handles = [syncobj];
    let points = [sequence];
    let timeout_ms = timeout.as_millis().min(u128::from(u32::MAX)) as u32;
    let mut now = timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut now) } < 0 {
        return Err(XdnaError::io("clock_gettime"));
    }
    let now_ns = i128::from(now.tv_sec) * 1_000_000_000 + i128::from(now.tv_nsec);
    let deadline = now_ns
        .checked_add(timeout.as_nanos() as i128)
        .and_then(|value| i64::try_from(value).ok())
        .ok_or_else(|| XdnaError::InvalidResponse("timeline deadline overflow".into()))?;
    let mut wait = SyncobjTimelineWait {
        handles: handles.as_ptr() as u64,
        points: points.as_ptr() as u64,
        timeout_nsec: deadline,
        count_handles: 1,
        flags: SYNCOBJ_WAIT_FOR_SUBMIT,
        ..Default::default()
    };
    match ioctl(
        fd,
        DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT,
        &mut wait,
        "wait for command",
    ) {
        Ok(()) => Ok(()),
        Err(XdnaError::Ioctl { source, .. })
            if source.raw_os_error() == Some(libc::ETIME)
                || source.raw_os_error() == Some(libc::ETIMEDOUT) =>
        {
            Err(XdnaError::Timeout { timeout_ms })
        }
        Err(error) => Err(error),
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn flush_range(pointer: *const u8, length: usize) {
    use std::arch::x86_64::{_mm_clflush, _mm_mfence};
    _mm_mfence();
    let mut offset = 0;
    while offset < length {
        _mm_clflush(pointer.add(offset));
        offset += 64;
    }
    _mm_mfence();
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn flush_range(_pointer: *const u8, _length: usize) {
    std::sync::atomic::fence(Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uapi_layouts_match_kernel_headers() {
        assert_eq!(size_of::<QosInfo>(), 24);
        assert_eq!(size_of::<CreateHwctx>(), 56);
        assert_eq!(size_of::<DestroyHwctx>(), 8);
        assert_eq!(size_of::<ConfigHwctx>(), 24);
        assert_eq!(size_of::<CreateBo>(), 32);
        assert_eq!(size_of::<GetBoInfo>(), 48);
        assert_eq!(size_of::<SyncBo>(), 24);
        assert_eq!(size_of::<ExecCmd>(), 56);
        assert_eq!(size_of::<GetInfo>(), 16);
        assert_eq!(size_of::<SyncobjTimelineWait>(), 48);
        assert_eq!(size_of::<DrmPrimeHandle>(), 12);
    }

    #[test]
    fn command_ring_index_wraps() {
        let next = AtomicUsize::new(usize::MAX);
        let first = next.fetch_add(1, Ordering::Relaxed) % 4;
        let second = next.fetch_add(1, Ordering::Relaxed) % 4;
        assert_eq!(first, 3);
        assert_eq!(second, 0);
    }

    #[test]
    fn binding_count_is_checked_before_submission() {
        let expected = [BindingLayout {
            name: "activation".into(),
            access: BindingAccess::Read,
            minimum_bytes: 4096,
            alignment: 64,
        }];
        assert!(matches!(
            validate_binding_abi(&expected, &[]),
            Err(XdnaError::BindingCount {
                expected: 1,
                actual: 0
            })
        ));
    }

    #[test]
    fn ert_header_matches_start_cu_layout() {
        assert_eq!(
            ert_header(ERT_NEW, 0, 16, ERT_START_CU, ERT_CU),
            0x3001_0001
        );
    }

    #[test]
    fn buffer_range_validation_accepts_edges_and_rejects_overflow() {
        assert_eq!(validate_buffer_range(4096, 0, 4096).unwrap(), 0);
        assert_eq!(validate_buffer_range(4096, 4096, 0).unwrap(), 4096);
        assert!(matches!(
            validate_buffer_range(4096, 4095, 2),
            Err(XdnaError::BufferRange { .. })
        ));
        assert!(matches!(
            validate_buffer_range(4096, u64::MAX, 1),
            Err(XdnaError::BufferRange { .. })
        ));
    }

    #[test]
    fn device_selection_requires_exactly_one_real_candidate() {
        let only = PathBuf::from("/dev/accel/accel7");
        assert_eq!(select_unique_device(vec![only.clone()]).unwrap(), only);
        assert!(matches!(
            select_unique_device(Vec::new()),
            Err(XdnaError::DeviceDiscovery(message)) if message.contains("no real")
        ));
        assert!(matches!(
            select_unique_device(vec![
                PathBuf::from("/dev/accel/accel1"),
                PathBuf::from("/dev/accel/accel0"),
            ]),
            Err(XdnaError::DeviceDiscovery(message))
                if message.contains("multiple") && message.contains("HIPFIRE_XDNA_DEVICE")
        ));
    }

    #[test]
    fn device_validation_rejects_compatibility_symlinks() {
        let directory = tempfile::tempdir().unwrap();
        let device_root = directory.path().join("dev/accel");
        let sysfs_root = directory.path().join("sys/class/accel");
        fs::create_dir_all(&device_root).unwrap();
        fs::create_dir_all(&sysfs_root).unwrap();
        let target = directory.path().join("real-device");
        fs::write(&target, []).unwrap();
        let alias = device_root.join("accel0");
        std::os::unix::fs::symlink(&target, &alias).unwrap();

        assert!(matches!(
            validate_amdxdna_node(&alias, &device_root, &sysfs_root),
            Err(XdnaError::UnsafeDevicePath { message, .. })
                if message.contains("symlinks are forbidden")
        ));
    }
}
