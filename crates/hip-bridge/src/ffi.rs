//! FFI bindings to libamdhip64.so via dlopen.
//! No link-time dependency — runtime loads the shared library.

use crate::error::{HipError, HipResult};
use crate::{DeviceBuffer, MemcpyKind};
use libloading::{Library, Symbol};
use std::ffi::{c_char, c_int, c_uint, c_void, CString};
use std::ptr;

/// Per-thread accumulators for time spent inside HIP FFI calls. Used by
/// Phase 3a host-vs-GPU diagnostics to attribute the forward pass wall
/// clock to specific HIP runtime calls.
pub mod launch_counters {
    use std::cell::Cell;

    macro_rules! counter {
        ($mod_name:ident) => {
            pub mod $mod_name {
                use std::cell::Cell;
                thread_local! {
                    pub(super) static TIME_NS: Cell<u64> = const { Cell::new(0) };
                    pub(super) static COUNT: Cell<u64> = const { Cell::new(0) };
                    pub(super) static BYTES: Cell<u64> = const { Cell::new(0) };
                }
                #[inline]
                pub fn record(ns: u64) {
                    TIME_NS.with(|c| c.set(c.get() + ns));
                    COUNT.with(|c| c.set(c.get() + 1));
                }
                #[inline]
                pub fn record_bytes(ns: u64, bytes: u64) {
                    TIME_NS.with(|c| c.set(c.get() + ns));
                    COUNT.with(|c| c.set(c.get() + 1));
                    BYTES.with(|c| c.set(c.get() + bytes));
                }
                pub fn time_ns() -> u64 { TIME_NS.with(|c| c.get()) }
                pub fn count() -> u64 { COUNT.with(|c| c.get()) }
                pub fn bytes() -> u64 { BYTES.with(|c| c.get()) }
                pub fn reset() {
                    TIME_NS.with(|c| c.set(0));
                    COUNT.with(|c| c.set(0));
                    BYTES.with(|c| c.set(0));
                }
            }
        };
    }

    // Existing counter — kept for back-compat with profile_host_vs_gpu.
    thread_local! {
        static TIME_NS: Cell<u64> = const { Cell::new(0) };
        static COUNT: Cell<u64> = const { Cell::new(0) };
    }

    #[inline]
    pub(super) fn record(ns: u64) {
        TIME_NS.with(|c| c.set(c.get() + ns));
        COUNT.with(|c| c.set(c.get() + 1));
        launch_kernel::record(ns);
    }

    pub fn reset() {
        TIME_NS.with(|c| c.set(0));
        COUNT.with(|c| c.set(0));
        launch_kernel::reset();
        memcpy_dtod::reset();
        memcpy_htod::reset();
        memcpy_dtoh::reset();
        memset::reset();
        ensure_kernel_lookup::reset();
        stream_sync::reset();
        event_sync::reset();
        device_sync::reset();
        graph_launch::reset();
    }

    pub fn time_ns() -> u64 { TIME_NS.with(|c| c.get()) }
    pub fn count() -> u64 { COUNT.with(|c| c.get()) }

    // Per-API counters
    counter!(launch_kernel);
    counter!(memcpy_dtod);
    counter!(memcpy_htod);
    counter!(memcpy_dtoh);
    counter!(memset);
    counter!(ensure_kernel_lookup);
    counter!(stream_sync);
    counter!(event_sync);
    counter!(device_sync);
    counter!(graph_launch);
}

// Opaque HIP handles (pointers to internal structs)
type HipStream = *mut c_void;
type HipModule = *mut c_void;
type HipFunction = *mut c_void;
type HipEvent = *mut c_void;
type HipGraph = *mut c_void;
type HipGraphExec = *mut c_void;

const HIP_SUCCESS: u32 = 0;

/// `hipPointerAttribute_t` per ROCm 6.4.3 layout. ROCm 5.x not supported.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HipPointerAttribute {
    pub mem_type: u32,
    pub device: c_int,
    pub device_pointer: *mut c_void,
    pub host_pointer: *mut c_void,
    pub is_managed: c_int,
    pub allocation_flags: c_uint,
}

impl Default for HipPointerAttribute {
    fn default() -> Self {
        Self {
            mem_type: 0,
            device: -1,
            device_pointer: ptr::null_mut(),
            host_pointer: ptr::null_mut(),
            is_managed: 0,
            allocation_flags: 0,
        }
    }
}

/// Loaded HIP runtime — holds the dlopen'd library and resolved function pointers.
pub struct HipRuntime {
    _lib: Library,

    // Version
    fn_runtime_get_version: unsafe extern "C" fn(*mut c_int) -> u32,

    // Device management
    fn_get_device_count: unsafe extern "C" fn(*mut c_int) -> u32,
    fn_set_device: unsafe extern "C" fn(c_int) -> u32,
    fn_get_device: unsafe extern "C" fn(*mut c_int) -> u32,

    // Multi-device / peer access
    fn_device_can_access_peer: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> u32,
    fn_device_enable_peer_access: unsafe extern "C" fn(c_int, c_uint) -> u32,
    fn_memcpy_peer:
        unsafe extern "C" fn(*mut c_void, c_int, *const c_void, c_int, usize) -> u32,
    fn_memcpy_peer_async:
        unsafe extern "C" fn(*mut c_void, c_int, *const c_void, c_int, usize, HipStream) -> u32,
    fn_pointer_get_attributes:
        unsafe extern "C" fn(*mut HipPointerAttribute, *const c_void) -> u32,

    // Memory
    fn_malloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> u32,
    fn_free: unsafe extern "C" fn(*mut c_void) -> u32,
    fn_memcpy: unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_uint) -> u32,
    fn_memcpy_async:
        unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_uint, HipStream) -> u32,
    fn_memset: unsafe extern "C" fn(*mut c_void, c_int, usize) -> u32,
    fn_memset_async: unsafe extern "C" fn(*mut c_void, c_int, usize, HipStream) -> u32,

    // Streams
    fn_stream_create: unsafe extern "C" fn(*mut HipStream) -> u32,
    fn_stream_synchronize: unsafe extern "C" fn(HipStream) -> u32,
    fn_stream_destroy: unsafe extern "C" fn(HipStream) -> u32,

    // Modules & kernels
    fn_module_load: unsafe extern "C" fn(*mut HipModule, *const c_char) -> u32,
    fn_module_load_data: unsafe extern "C" fn(*mut HipModule, *const c_void) -> u32,
    fn_module_get_function:
        unsafe extern "C" fn(*mut HipFunction, HipModule, *const c_char) -> u32,
    fn_module_launch_kernel: unsafe extern "C" fn(
        HipFunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        HipStream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> u32,

    // Events
    fn_event_create: unsafe extern "C" fn(*mut HipEvent) -> u32,
    fn_event_record: unsafe extern "C" fn(HipEvent, HipStream) -> u32,
    fn_event_synchronize: unsafe extern "C" fn(HipEvent) -> u32,
    fn_event_elapsed_time: unsafe extern "C" fn(*mut f32, HipEvent, HipEvent) -> u32,
    fn_event_destroy: unsafe extern "C" fn(HipEvent) -> u32,
    fn_stream_wait_event: unsafe extern "C" fn(HipStream, HipEvent, c_uint) -> u32,

    // Error
    fn_get_error_string: unsafe extern "C" fn(u32) -> *const i8,
    fn_get_last_error: unsafe extern "C" fn() -> u32,

    // Graph capture & replay
    fn_stream_begin_capture:
        unsafe extern "C" fn(HipStream, c_uint) -> u32,
    fn_stream_end_capture:
        unsafe extern "C" fn(HipStream, *mut HipGraph) -> u32,
    fn_graph_instantiate:
        unsafe extern "C" fn(*mut HipGraphExec, HipGraph, *mut HipGraph, *mut c_void, usize) -> u32,
    fn_graph_launch:
        unsafe extern "C" fn(HipGraphExec, HipStream) -> u32,
    fn_graph_exec_destroy: unsafe extern "C" fn(HipGraphExec) -> u32,
    fn_graph_destroy: unsafe extern "C" fn(HipGraph) -> u32,
    // Stream memory ops (HIP 7.2+)
    fn_stream_write_value32:
        unsafe extern "C" fn(HipStream, *mut c_void, u32, c_uint) -> u32,
    fn_device_synchronize: unsafe extern "C" fn() -> u32,
    fn_get_device_properties: unsafe extern "C" fn(*mut u8, c_int) -> u32,
    fn_get_device_attribute: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> u32,
    fn_mem_get_info: unsafe extern "C" fn(*mut usize, *mut usize) -> u32,
}

// HipRuntime is Send+Sync — the underlying HIP runtime is thread-safe for API calls.
unsafe impl Send for HipRuntime {}
unsafe impl Sync for HipRuntime {}

macro_rules! load_fn {
    ($lib:expr, $name:expr, $ty:ty) => {{
        let sym: Symbol<'_, $ty> = $lib
            .get($name.as_bytes())
            .map_err(|e| HipError::new(0, &format!("failed to load symbol {}: {e}", $name)))?;
        *sym.into_raw()
    }};
}

impl HipRuntime {
    /// Load the HIP runtime via dlopen.
    /// Searches standard paths: /opt/rocm/lib, system library path.
    pub fn load() -> HipResult<Self> {
        #[cfg(target_os = "windows")]
        let lib = unsafe {
            let userprofile = std::env::var("USERPROFILE").unwrap_or_default();
            let hip_path = std::env::var("HIP_PATH").unwrap_or_default();
            let p1 = format!(r"{userprofile}\.hipfire\runtime\amdhip64.dll");
            let p2 = format!(r"{hip_path}\bin\amdhip64.dll");
            // Try unversioned first, then versioned names (HIP SDK 7.x installs amdhip64_7.dll)
            Library::new(&p1)
                .or_else(|_| Library::new(&p2))
                .or_else(|_| Library::new("amdhip64.dll"))
                .or_else(|_| Library::new("amdhip64_7.dll"))
                .or_else(|_| Library::new("amdhip64_6.dll"))
                .or_else(|_| {
                    // Try versioned names in explicit paths (runtime dir + HIP_PATH)
                    let rt = format!(r"{userprofile}\.hipfire\runtime");
                    let hp = format!(r"{hip_path}\bin");
                    Library::new(&format!(r"{rt}\amdhip64_7.dll"))
                        .or_else(|_| Library::new(&format!(r"{rt}\amdhip64_6.dll")))
                        .or_else(|_| Library::new(&format!(r"{hp}\amdhip64_7.dll")))
                        .or_else(|_| Library::new(&format!(r"{hp}\amdhip64_6.dll")))
                })
                .map_err(|e| {
                    HipError::new(
                        0,
                        &format!(
                            "failed to load amdhip64.dll: {e}. \
                             Searched: {p1}, {p2}, amdhip64.dll, amdhip64_7.dll, amdhip64_6.dll (PATH). \
                             Is ROCm/HIP installed?"
                        ),
                    )
                })?
        };

        #[cfg(not(target_os = "windows"))]
        let lib = unsafe {
            // Try unversioned first (canonical with rocm-hip-devel symlink),
            // then versioned SONAMEs. Fedora's `rocm-hip` package ships only
            // `libamdhip64.so.6` — the unversioned `.so` symlink is in the
            // `-devel` package which most users don't have. Reported in #64.
            Library::new("libamdhip64.so")
                .or_else(|_| Library::new("libamdhip64.so.7"))
                .or_else(|_| Library::new("libamdhip64.so.6"))
                .or_else(|_| Library::new("libamdhip64.so.5"))
                .map_err(|e| {
                    HipError::new(
                        0,
                        &format!(
                            "failed to dlopen libamdhip64.so: {e}. \
                             Tried: libamdhip64.so, libamdhip64.so.7, .so.6, .so.5. \
                             Is ROCm installed?"
                        ),
                    )
                })?
        };

        unsafe {
            Ok(Self {
                fn_runtime_get_version: load_fn!(lib, "hipRuntimeGetVersion", unsafe extern "C" fn(*mut c_int) -> u32),
                fn_get_device_count: load_fn!(lib, "hipGetDeviceCount", unsafe extern "C" fn(*mut c_int) -> u32),
                fn_set_device: load_fn!(lib, "hipSetDevice", unsafe extern "C" fn(c_int) -> u32),
                fn_get_device: load_fn!(lib, "hipGetDevice", unsafe extern "C" fn(*mut c_int) -> u32),
                fn_device_can_access_peer: load_fn!(lib, "hipDeviceCanAccessPeer",
                    unsafe extern "C" fn(*mut c_int, c_int, c_int) -> u32),
                fn_device_enable_peer_access: load_fn!(lib, "hipDeviceEnablePeerAccess",
                    unsafe extern "C" fn(c_int, c_uint) -> u32),
                fn_memcpy_peer: load_fn!(lib, "hipMemcpyPeer",
                    unsafe extern "C" fn(*mut c_void, c_int, *const c_void, c_int, usize) -> u32),
                fn_memcpy_peer_async: load_fn!(lib, "hipMemcpyPeerAsync",
                    unsafe extern "C" fn(*mut c_void, c_int, *const c_void, c_int, usize, HipStream) -> u32),
                fn_pointer_get_attributes: load_fn!(lib, "hipPointerGetAttributes",
                    unsafe extern "C" fn(*mut HipPointerAttribute, *const c_void) -> u32),
                fn_malloc: load_fn!(lib, "hipMalloc", unsafe extern "C" fn(*mut *mut c_void, usize) -> u32),
                fn_free: load_fn!(lib, "hipFree", unsafe extern "C" fn(*mut c_void) -> u32),
                fn_memcpy: load_fn!(lib, "hipMemcpy", unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_uint) -> u32),
                fn_memcpy_async: load_fn!(lib, "hipMemcpyAsync", unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_uint, HipStream) -> u32),
                fn_memset: load_fn!(lib, "hipMemset", unsafe extern "C" fn(*mut c_void, c_int, usize) -> u32),
                fn_memset_async: load_fn!(lib, "hipMemsetAsync", unsafe extern "C" fn(*mut c_void, c_int, usize, HipStream) -> u32),
                fn_stream_create: load_fn!(lib, "hipStreamCreate", unsafe extern "C" fn(*mut HipStream) -> u32),
                fn_stream_synchronize: load_fn!(lib, "hipStreamSynchronize", unsafe extern "C" fn(HipStream) -> u32),
                fn_stream_destroy: load_fn!(lib, "hipStreamDestroy", unsafe extern "C" fn(HipStream) -> u32),
                fn_module_load: load_fn!(lib, "hipModuleLoad", unsafe extern "C" fn(*mut HipModule, *const c_char) -> u32),
                fn_module_load_data: load_fn!(lib, "hipModuleLoadData", unsafe extern "C" fn(*mut HipModule, *const c_void) -> u32),
                fn_module_get_function: load_fn!(lib, "hipModuleGetFunction", unsafe extern "C" fn(*mut HipFunction, HipModule, *const c_char) -> u32),
                fn_module_launch_kernel: load_fn!(lib, "hipModuleLaunchKernel", unsafe extern "C" fn(HipFunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, HipStream, *mut *mut c_void, *mut *mut c_void) -> u32),
                fn_event_create: load_fn!(lib, "hipEventCreate", unsafe extern "C" fn(*mut HipEvent) -> u32),
                fn_event_record: load_fn!(lib, "hipEventRecord", unsafe extern "C" fn(HipEvent, HipStream) -> u32),
                fn_event_synchronize: load_fn!(lib, "hipEventSynchronize", unsafe extern "C" fn(HipEvent) -> u32),
                fn_event_elapsed_time: load_fn!(lib, "hipEventElapsedTime", unsafe extern "C" fn(*mut f32, HipEvent, HipEvent) -> u32),
                fn_event_destroy: load_fn!(lib, "hipEventDestroy", unsafe extern "C" fn(HipEvent) -> u32),
                fn_stream_wait_event: load_fn!(lib, "hipStreamWaitEvent", unsafe extern "C" fn(HipStream, HipEvent, c_uint) -> u32),
                fn_get_error_string: load_fn!(lib, "hipGetErrorString", unsafe extern "C" fn(u32) -> *const i8),
                fn_get_last_error: load_fn!(lib, "hipGetLastError", unsafe extern "C" fn() -> u32),
                fn_stream_begin_capture: load_fn!(lib, "hipStreamBeginCapture", unsafe extern "C" fn(HipStream, c_uint) -> u32),
                fn_stream_end_capture: load_fn!(lib, "hipStreamEndCapture", unsafe extern "C" fn(HipStream, *mut HipGraph) -> u32),
                fn_graph_instantiate: load_fn!(lib, "hipGraphInstantiate", unsafe extern "C" fn(*mut HipGraphExec, HipGraph, *mut HipGraph, *mut c_void, usize) -> u32),
                fn_graph_launch: load_fn!(lib, "hipGraphLaunch", unsafe extern "C" fn(HipGraphExec, HipStream) -> u32),
                fn_graph_exec_destroy: load_fn!(lib, "hipGraphExecDestroy", unsafe extern "C" fn(HipGraphExec) -> u32),
                fn_graph_destroy: load_fn!(lib, "hipGraphDestroy", unsafe extern "C" fn(HipGraph) -> u32),
                fn_stream_write_value32: load_fn!(lib, "hipStreamWriteValue32",
                    unsafe extern "C" fn(HipStream, *mut c_void, u32, c_uint) -> u32),
                fn_device_synchronize: load_fn!(lib, "hipDeviceSynchronize", unsafe extern "C" fn() -> u32),
                fn_get_device_properties: load_fn!(lib, "hipGetDeviceProperties", unsafe extern "C" fn(*mut u8, c_int) -> u32),
                fn_get_device_attribute: load_fn!(lib, "hipDeviceGetAttribute", unsafe extern "C" fn(*mut c_int, c_int, c_int) -> u32),
                fn_mem_get_info: load_fn!(lib, "hipMemGetInfo", unsafe extern "C" fn(*mut usize, *mut usize) -> u32),
                _lib: lib,
            })
        }
    }

    fn check(&self, code: u32, context: &str) -> HipResult<()> {
        if code == HIP_SUCCESS {
            Ok(())
        } else {
            Err(HipError::from_code(
                code,
                context,
                Some(&self.fn_get_error_string),
            ))
        }
    }

    // ── Version ────────────────────────────────────────────────

    /// Get HIP runtime version as (major, minor). E.g. ROCm 6.3 → (6, 3).
    pub fn runtime_version(&self) -> HipResult<(i32, i32)> {
        let mut version: c_int = 0;
        let code = unsafe { (self.fn_runtime_get_version)(&mut version) };
        self.check(code, "hipRuntimeGetVersion")?;
        // HIP version encoding: major * 10000000 + minor * 100000 + patch
        let major = version / 10_000_000;
        let minor = (version % 10_000_000) / 100_000;
        Ok((major, minor))
    }

    // ── Device management ───────────────────────────────────────

    pub fn device_count(&self) -> HipResult<i32> {
        let mut count: c_int = 0;
        let code = unsafe { (self.fn_get_device_count)(&mut count) };
        self.check(code, "hipGetDeviceCount")?;
        Ok(count)
    }

    pub fn set_device(&self, id: i32) -> HipResult<()> {
        let code = unsafe { (self.fn_set_device)(id) };
        self.check(code, "hipSetDevice")
    }

    pub fn current_device(&self) -> HipResult<i32> {
        let mut id: c_int = 0;
        let code = unsafe { (self.fn_get_device)(&mut id) };
        self.check(code, "hipGetDevice")?;
        Ok(id)
    }

    // ── Multi-device / peer access ──────────────────────────────

    pub fn can_access_peer(&self, device: i32, peer_device: i32) -> HipResult<bool> {
        let mut can: c_int = 0;
        let code = unsafe { (self.fn_device_can_access_peer)(&mut can, device, peer_device) };
        self.check(code, "hipDeviceCanAccessPeer")?;
        Ok(can != 0)
    }

    /// Idempotent: hipErrorPeerAccessAlreadyEnabled (704) → Ok. Caller must
    /// have bound the source device first (`set_device`).
    pub fn enable_peer_access(&self, peer_device: i32) -> HipResult<()> {
        let code = unsafe { (self.fn_device_enable_peer_access)(peer_device, 0) };
        if code == HIP_SUCCESS || code == crate::HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED {
            Ok(())
        } else {
            Err(HipError::from_code(
                code,
                "hipDeviceEnablePeerAccess",
                Some(&self.fn_get_error_string),
            ))
        }
    }

    pub fn memcpy_peer(
        &self,
        dst: &DeviceBuffer,
        dst_device: i32,
        src: &DeviceBuffer,
        src_device: i32,
        size: usize,
    ) -> HipResult<()> {
        assert!(size <= dst.size(), "size ({size}) exceeds dst ({})", dst.size());
        assert!(size <= src.size(), "size ({size}) exceeds src ({})", src.size());
        let code = unsafe {
            (self.fn_memcpy_peer)(
                dst.as_ptr(),
                dst_device,
                src.as_ptr() as *const c_void,
                src_device,
                size,
            )
        };
        self.check(code, "hipMemcpyPeer")
    }

    /// Stream must belong to one of the two devices involved.
    pub fn memcpy_peer_async(
        &self,
        dst: &DeviceBuffer,
        dst_device: i32,
        src: &DeviceBuffer,
        src_device: i32,
        size: usize,
        stream: &Stream,
    ) -> HipResult<()> {
        assert!(size <= dst.size(), "size ({size}) exceeds dst ({})", dst.size());
        assert!(size <= src.size(), "size ({size}) exceeds src ({})", src.size());
        let code = unsafe {
            (self.fn_memcpy_peer_async)(
                dst.as_ptr(),
                dst_device,
                src.as_ptr() as *const c_void,
                src_device,
                size,
                stream.0,
            )
        };
        self.check(code, "hipMemcpyPeerAsync")
    }

    pub fn pointer_get_attributes(&self, buf: &DeviceBuffer) -> HipResult<HipPointerAttribute> {
        let mut attr = HipPointerAttribute::default();
        let code = unsafe { (self.fn_pointer_get_attributes)(&mut attr, buf.as_ptr()) };
        self.check(code, "hipPointerGetAttributes")?;
        Ok(attr)
    }

    // ── Memory management ───────────────────────────────────────

    pub fn malloc(&self, size: usize) -> HipResult<DeviceBuffer> {
        let mut ptr: *mut c_void = ptr::null_mut();
        let code = unsafe { (self.fn_malloc)(&mut ptr, size) };
        self.check(code, "hipMalloc")?;
        Ok(DeviceBuffer { ptr, size })
    }

    /// # Safety
    /// Caller must ensure the buffer is not in use by any pending GPU operations.
    pub fn free(&self, buf: DeviceBuffer) -> HipResult<()> {
        let code = unsafe { (self.fn_free)(buf.ptr) };
        std::mem::forget(buf); // prevent double-free
        self.check(code, "hipFree")
    }

    /// Copy host data into GPU buffer at a byte offset.
    pub fn memcpy_htod_offset(
        &self,
        dst: &DeviceBuffer,
        offset: usize,
        src: &[u8],
    ) -> HipResult<()> {
        assert!(
            offset + src.len() <= dst.size,
            "offset ({}) + source ({}) exceeds device buffer ({})",
            offset,
            src.len(),
            dst.size
        );
        let dst_ptr = unsafe { (dst.ptr as *mut u8).add(offset) as *mut c_void };
        let code = unsafe {
            (self.fn_memcpy)(
                dst_ptr,
                src.as_ptr() as *const c_void,
                src.len(),
                MemcpyKind::HostToDevice as c_uint,
            )
        };
        self.check(code, "hipMemcpy H2D offset")
    }

    /// Copy bytes between GPU buffers with offsets on both sides.
    pub fn memcpy_dtod_at(
        &self,
        dst: &DeviceBuffer,
        dst_offset: usize,
        src: &DeviceBuffer,
        src_offset: usize,
        size: usize,
    ) -> HipResult<()> {
        assert!(dst_offset + size <= dst.size);
        assert!(src_offset + size <= src.size);
        let dst_ptr = unsafe { (dst.ptr as *mut u8).add(dst_offset) as *mut c_void };
        let src_ptr = unsafe { (src.ptr as *const u8).add(src_offset) as *const c_void };
        let t = std::time::Instant::now();
        let code = unsafe {
            (self.fn_memcpy)(dst_ptr, src_ptr, size, MemcpyKind::DeviceToDevice as c_uint)
        };
        crate::ffi::launch_counters::memcpy_dtod::record(t.elapsed().as_nanos() as u64);
        self.check(code, "hipMemcpy D2D at offset")
    }

    /// Copy bytes from one GPU buffer at an offset to another GPU buffer.
    pub fn memcpy_dtod_offset(
        &self,
        dst: &DeviceBuffer,
        src: &DeviceBuffer,
        src_offset: usize,
        size: usize,
    ) -> HipResult<()> {
        assert!(size <= dst.size, "size ({size}) exceeds dst ({})", dst.size);
        assert!(src_offset + size <= src.size, "src_offset+size exceeds src");
        let src_ptr = unsafe { (src.ptr as *const u8).add(src_offset) as *const c_void };
        let t = std::time::Instant::now();
        let code = unsafe {
            (self.fn_memcpy)(
                dst.ptr,
                src_ptr,
                size,
                MemcpyKind::DeviceToDevice as c_uint,
            )
        };
        crate::ffi::launch_counters::memcpy_dtod::record(t.elapsed().as_nanos() as u64);
        self.check(code, "hipMemcpy D2D offset")
    }

    pub fn memcpy_htod(&self, dst: &DeviceBuffer, src: &[u8]) -> HipResult<()> {
        assert!(
            src.len() <= dst.size,
            "source ({}) exceeds device buffer ({})",
            src.len(),
            dst.size
        );
        let t = std::time::Instant::now();
        let code = unsafe {
            (self.fn_memcpy)(
                dst.ptr,
                src.as_ptr() as *const c_void,
                src.len(),
                MemcpyKind::HostToDevice as c_uint,
            )
        };
        crate::ffi::launch_counters::memcpy_htod::record(t.elapsed().as_nanos() as u64);
        self.check(code, "hipMemcpy H2D")
    }

    #[track_caller]
    pub fn memcpy_dtoh(&self, dst: &mut [u8], src: &DeviceBuffer) -> HipResult<()> {
        assert!(
            dst.len() <= src.size,
            "destination ({}) exceeds device buffer ({})",
            dst.len(),
            src.size
        );
        let loc = std::panic::Location::caller();
        let t = std::time::Instant::now();
        let code = unsafe {
            (self.fn_memcpy)(
                dst.as_mut_ptr() as *mut c_void,
                src.ptr as *const c_void,
                dst.len(),
                MemcpyKind::DeviceToHost as c_uint,
            )
        };
        let elapsed = t.elapsed().as_nanos() as u64;
        crate::ffi::launch_counters::memcpy_dtoh::record_bytes(elapsed, dst.len() as u64);
        static DUMP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let dump = *DUMP.get_or_init(|| {
            std::env::var("HIPFIRE_DTOH_DUMP").ok().as_deref() == Some("1")
        });
        if dump {
            eprintln!("dtoh bytes={} us={} at {}:{}", dst.len(), elapsed / 1000, loc.file(), loc.line());
        }
        self.check(code, "hipMemcpy D2H")
    }

    /// Copy bytes from a GPU buffer at a given source offset to host.
    /// `dst.len()` bytes are copied starting from `src.ptr + src_offset`.
    #[track_caller]
    pub fn memcpy_dtoh_at(
        &self,
        dst: &mut [u8],
        src: &DeviceBuffer,
        src_offset: usize,
    ) -> HipResult<()> {
        assert!(
            src_offset + dst.len() <= src.size,
            "src_offset ({}) + dst len ({}) exceeds device buffer ({})",
            src_offset, dst.len(), src.size
        );
        let src_ptr = unsafe { (src.ptr as *const u8).add(src_offset) as *const c_void };
        let loc = std::panic::Location::caller();
        let t = std::time::Instant::now();
        let code = unsafe {
            (self.fn_memcpy)(
                dst.as_mut_ptr() as *mut c_void,
                src_ptr,
                dst.len(),
                MemcpyKind::DeviceToHost as c_uint,
            )
        };
        let elapsed = t.elapsed().as_nanos() as u64;
        crate::ffi::launch_counters::memcpy_dtoh::record_bytes(elapsed, dst.len() as u64);
        static DUMP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let dump = *DUMP.get_or_init(|| {
            std::env::var("HIPFIRE_DTOH_DUMP").ok().as_deref() == Some("1")
        });
        if dump {
            eprintln!("dtoh_at bytes={} us={} at {}:{}", dst.len(), elapsed / 1000, loc.file(), loc.line());
        }
        self.check(code, "hipMemcpy D2H at offset")
    }

    pub fn memcpy_dtod(
        &self,
        dst: &DeviceBuffer,
        src: &DeviceBuffer,
        size: usize,
    ) -> HipResult<()> {
        assert!(size <= dst.size && size <= src.size);
        let t = std::time::Instant::now();
        let code = unsafe {
            (self.fn_memcpy)(
                dst.ptr,
                src.ptr as *const c_void,
                size,
                MemcpyKind::DeviceToDevice as c_uint,
            )
        };
        crate::ffi::launch_counters::memcpy_dtod::record(t.elapsed().as_nanos() as u64);
        self.check(code, "hipMemcpy D2D")
    }

    #[track_caller]
    pub fn memset(&self, buf: &DeviceBuffer, value: i32, size: usize) -> HipResult<()> {
        assert!(size <= buf.size);
        let loc = std::panic::Location::caller();
        let t = std::time::Instant::now();
        let code = unsafe { (self.fn_memset)(buf.ptr, value, size) };
        let elapsed = t.elapsed().as_nanos() as u64;
        crate::ffi::launch_counters::memset::record_bytes(elapsed, size as u64);
        static DUMP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let dump = *DUMP.get_or_init(|| {
            std::env::var("HIPFIRE_MEMSET_DUMP").ok().as_deref() == Some("1")
        });
        if dump {
            eprintln!("memset bytes={} us={} at {}:{}", size, elapsed / 1000, loc.file(), loc.line());
        }
        self.check(code, "hipMemset")
    }

    /// Async memset on a specific stream — does NOT block the host.
    /// Caller must ensure stream-ordering downstream work syncs correctly.
    #[track_caller]
    pub fn memset_async(
        &self,
        buf: &DeviceBuffer,
        value: i32,
        size: usize,
        stream: &Stream,
    ) -> HipResult<()> {
        assert!(size <= buf.size);
        let loc = std::panic::Location::caller();
        let t = std::time::Instant::now();
        let code = unsafe { (self.fn_memset_async)(buf.ptr, value, size, stream.0) };
        let elapsed = t.elapsed().as_nanos() as u64;
        crate::ffi::launch_counters::memset::record_bytes(elapsed, size as u64);
        static DUMP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let dump = *DUMP.get_or_init(|| {
            std::env::var("HIPFIRE_MEMSET_DUMP").ok().as_deref() == Some("1")
        });
        if dump {
            eprintln!("memset_async bytes={} us={} at {}:{}", size, elapsed / 1000, loc.file(), loc.line());
        }
        self.check(code, "hipMemsetAsync")
    }

    // ── Streams ─────────────────────────────────────────────────

    pub fn stream_create(&self) -> HipResult<Stream> {
        let mut stream: HipStream = ptr::null_mut();
        let code = unsafe { (self.fn_stream_create)(&mut stream) };
        self.check(code, "hipStreamCreate")?;
        Ok(Stream(stream))
    }

    pub fn stream_synchronize(&self, stream: &Stream) -> HipResult<()> {
        let t = std::time::Instant::now();
        let code = unsafe { (self.fn_stream_synchronize)(stream.0) };
        crate::ffi::launch_counters::stream_sync::record(t.elapsed().as_nanos() as u64);
        self.check(code, "hipStreamSynchronize")
    }

    pub fn stream_destroy(&self, stream: Stream) -> HipResult<()> {
        let code = unsafe { (self.fn_stream_destroy)(stream.0) };
        std::mem::forget(stream);
        self.check(code, "hipStreamDestroy")
    }

    // ── Modules & Kernels ───────────────────────────────────────

    pub fn module_load(&self, path: &str) -> HipResult<Module> {
        let c_path =
            CString::new(path).map_err(|_| HipError::new(1, "invalid path for module_load"))?;
        let mut module: HipModule = ptr::null_mut();
        let code = unsafe { (self.fn_module_load)(&mut module, c_path.as_ptr()) };
        self.check(code, "hipModuleLoad")?;
        Ok(Module(module))
    }

    pub fn module_load_data(&self, image: &[u8]) -> HipResult<Module> {
        let mut module: HipModule = ptr::null_mut();
        let code =
            unsafe { (self.fn_module_load_data)(&mut module, image.as_ptr() as *const c_void) };
        self.check(code, "hipModuleLoadData")?;
        Ok(Module(module))
    }

    pub fn module_get_function(&self, module: &Module, name: &str) -> HipResult<Function> {
        let c_name = CString::new(name)
            .map_err(|_| HipError::new(1, "invalid kernel name for module_get_function"))?;
        let mut func: HipFunction = ptr::null_mut();
        let code =
            unsafe { (self.fn_module_get_function)(&mut func, module.0, c_name.as_ptr()) };
        self.check(code, "hipModuleGetFunction")?;
        Ok(Function(func))
    }

    /// Launch a kernel on the GPU.
    ///
    /// # Safety
    /// `params` must contain valid pointers to kernel arguments matching the kernel signature.
    pub unsafe fn launch_kernel(
        &self,
        func: &Function,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: Option<&Stream>,
        params: &mut [*mut c_void],
    ) -> HipResult<()> {
        let stream_raw = stream.map_or(ptr::null_mut(), |s| s.0);
        let t = std::time::Instant::now();
        let code = (self.fn_module_launch_kernel)(
            func.0,
            grid[0],
            grid[1],
            grid[2],
            block[0],
            block[1],
            block[2],
            shared_mem,
            stream_raw,
            params.as_mut_ptr(),
            ptr::null_mut(),
        );
        crate::ffi::launch_counters::record(t.elapsed().as_nanos() as u64);
        self.check(code, "hipModuleLaunchKernel")
    }

    /// Launch a kernel using the `extra` path, passing a contiguous kernarg
    /// byte buffer instead of the traditional `void**` pointer-per-arg array.
    ///
    /// This path is REQUIRED for graph capture on gfx1100 / ROCm 6.3: when a
    /// launch is captured into a stream graph via `hipStreamBeginCapture`, the
    /// kernelParams path (`*mut *mut c_void`) only captures pointers, not the
    /// pointed-to values. By the time the graph is replayed, the stack frame
    /// that held those values is gone and the kernel reads garbage. The
    /// `extra` path, on the other hand, hands HIP a single blob pointer +
    /// size, and HIP copies the blob contents into the kernel node at capture
    /// time.
    ///
    /// The caller owns the `kernarg_blob` slice and is responsible for keeping
    /// it alive for the lifetime of any graph that captured this launch. For
    /// one-shot launches (no capture) the blob may be stack-local.
    ///
    /// Layout contract: `kernarg_blob` must be the kernel's full kernarg
    /// struct, laid out with natural alignment per field (matching the way
    /// hipcc emits the kernel's argument ABI). Total blob size is passed
    /// alongside the pointer via HIP_LAUNCH_PARAM_BUFFER_SIZE.
    ///
    /// # Safety
    /// `kernarg_blob` must have layout + size matching the kernel signature,
    /// and all contained pointers must be valid GPU addresses.
    pub unsafe fn launch_kernel_blob(
        &self,
        func: &Function,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: Option<&Stream>,
        kernarg_blob: &mut [u8],
    ) -> HipResult<()> {
        // HIP `extra` mode sentinel constants (from hip_runtime.h):
        //   HIP_LAUNCH_PARAM_BUFFER_POINTER = 0x01
        //   HIP_LAUNCH_PARAM_BUFFER_SIZE    = 0x02
        //   HIP_LAUNCH_PARAM_END            = 0x03
        // The `extra` array alternates sentinel, value pointer, ..., END.
        let mut blob_size: usize = kernarg_blob.len();
        let blob_ptr: *mut c_void = kernarg_blob.as_mut_ptr() as *mut c_void;
        let size_ptr: *mut c_void = (&mut blob_size as *mut usize) as *mut c_void;
        let mut extra: [*mut c_void; 5] = [
            0x01 as *mut c_void,       // HIP_LAUNCH_PARAM_BUFFER_POINTER
            blob_ptr,                  // → persistent kernarg blob
            0x02 as *mut c_void,       // HIP_LAUNCH_PARAM_BUFFER_SIZE
            size_ptr,                  // → &blob_size (must live across the call)
            0x03 as *mut c_void,       // HIP_LAUNCH_PARAM_END
        ];

        let stream_raw = stream.map_or(ptr::null_mut(), |s| s.0);
        let t = std::time::Instant::now();
        let code = (self.fn_module_launch_kernel)(
            func.0,
            grid[0],
            grid[1],
            grid[2],
            block[0],
            block[1],
            block[2],
            shared_mem,
            stream_raw,
            ptr::null_mut(),           // kernelParams = null (we use extra)
            extra.as_mut_ptr(),
        );
        crate::ffi::launch_counters::record(t.elapsed().as_nanos() as u64);
        self.check(code, "hipModuleLaunchKernel(extra blob)")
    }

    // ── Events ──────────────────────────────────────────────────

    pub fn event_create(&self) -> HipResult<Event> {
        let mut event: HipEvent = ptr::null_mut();
        let code = unsafe { (self.fn_event_create)(&mut event) };
        self.check(code, "hipEventCreate")?;
        Ok(Event(event))
    }

    pub fn event_record(&self, event: &Event, stream: Option<&Stream>) -> HipResult<()> {
        let stream_raw = stream.map_or(ptr::null_mut(), |s| s.0);
        let code = unsafe { (self.fn_event_record)(event.0, stream_raw) };
        self.check(code, "hipEventRecord")
    }

    pub fn event_synchronize(&self, event: &Event) -> HipResult<()> {
        let t = std::time::Instant::now();
        let code = unsafe { (self.fn_event_synchronize)(event.0) };
        crate::ffi::launch_counters::event_sync::record(t.elapsed().as_nanos() as u64);
        self.check(code, "hipEventSynchronize")
    }

    pub fn event_elapsed_ms(&self, start: &Event, stop: &Event) -> HipResult<f32> {
        let mut ms: f32 = 0.0;
        let code = unsafe { (self.fn_event_elapsed_time)(&mut ms, start.0, stop.0) };
        self.check(code, "hipEventElapsedTime")?;
        Ok(ms)
    }

    pub fn event_destroy(&self, event: Event) -> HipResult<()> {
        let code = unsafe { (self.fn_event_destroy)(event.0) };
        std::mem::forget(event);
        self.check(code, "hipEventDestroy")
    }

    pub fn stream_wait_event(&self, stream: &Stream, event: &Event) -> HipResult<()> {
        let code = unsafe { (self.fn_stream_wait_event)(stream.0, event.0, 0) };
        self.check(code, "hipStreamWaitEvent")
    }

    // ── Error query ─────────────────────────────────────────────

    pub fn last_error(&self) -> u32 {
        unsafe { (self.fn_get_last_error)() }
    }

    // ── Async memory ops ────────────────────────────────────────

    pub fn memcpy_htod_async(
        &self,
        dst: &DeviceBuffer,
        src: &[u8],
        stream: &Stream,
    ) -> HipResult<()> {
        assert!(src.len() <= dst.size);
        let code = unsafe {
            (self.fn_memcpy_async)(
                dst.ptr,
                src.as_ptr() as *const c_void,
                src.len(),
                MemcpyKind::HostToDevice as c_uint,
                stream.0,
            )
        };
        self.check(code, "hipMemcpyAsync H2D")
    }

    pub fn memcpy_dtoh_async(
        &self,
        dst: &mut [u8],
        src: &DeviceBuffer,
        stream: &Stream,
    ) -> HipResult<()> {
        assert!(dst.len() <= src.size);
        let code = unsafe {
            (self.fn_memcpy_async)(
                dst.as_mut_ptr() as *mut c_void,
                src.ptr as *const c_void,
                dst.len(),
                MemcpyKind::DeviceToHost as c_uint,
                stream.0,
            )
        };
        self.check(code, "hipMemcpyAsync D2H")
    }

    /// Async D→D copy with optional offsets on both sides. Ordered on
    /// `stream` and capturable by hipStreamBeginCapture — use this in
    /// place of sync `memcpy_dtod_at` wherever the copy needs to live
    /// inside a hipGraph.
    pub fn memcpy_dtod_async_at(
        &self,
        dst: &DeviceBuffer,
        dst_offset: usize,
        src: &DeviceBuffer,
        src_offset: usize,
        size: usize,
        stream: &Stream,
    ) -> HipResult<()> {
        assert!(dst_offset + size <= dst.size);
        assert!(src_offset + size <= src.size);
        let dst_ptr = unsafe { (dst.ptr as *mut u8).add(dst_offset) as *mut c_void };
        let src_ptr = unsafe { (src.ptr as *const u8).add(src_offset) as *const c_void };
        let code = unsafe {
            (self.fn_memcpy_async)(
                dst_ptr, src_ptr, size,
                MemcpyKind::DeviceToDevice as c_uint,
                stream.0,
            )
        };
        self.check(code, "hipMemcpyAsync D2D offset")
    }

    // ── Graph capture & replay ──────────────────────────────────

    /// Begin capturing all operations on `stream` into a graph.
    /// mode=0 is hipStreamCaptureModeGlobal.
    pub fn stream_begin_capture(&self, stream: &Stream, mode: u32) -> HipResult<()> {
        let code = unsafe { (self.fn_stream_begin_capture)(stream.0, mode as c_uint) };
        self.check(code, "hipStreamBeginCapture")
    }

    /// End capture on `stream`, returning the captured graph.
    pub fn stream_end_capture(&self, stream: &Stream) -> HipResult<Graph> {
        let mut graph: HipGraph = ptr::null_mut();
        let code = unsafe { (self.fn_stream_end_capture)(stream.0, &mut graph) };
        self.check(code, "hipStreamEndCapture")?;
        Ok(Graph(graph))
    }

    /// Instantiate an executable graph from a captured graph.
    pub fn graph_instantiate(&self, graph: &Graph) -> HipResult<GraphExec> {
        let mut exec: HipGraphExec = ptr::null_mut();
        let code = unsafe {
            (self.fn_graph_instantiate)(&mut exec, graph.0, ptr::null_mut(), ptr::null_mut(), 0)
        };
        self.check(code, "hipGraphInstantiate")?;
        Ok(GraphExec(exec))
    }

    /// Launch an executable graph on `stream`.
    pub fn graph_launch(&self, exec: &GraphExec, stream: &Stream) -> HipResult<()> {
        let t = std::time::Instant::now();
        let code = unsafe { (self.fn_graph_launch)(exec.0, stream.0) };
        crate::ffi::launch_counters::graph_launch::record(t.elapsed().as_nanos() as u64);
        self.check(code, "hipGraphLaunch")
    }

    pub fn graph_exec_destroy(&self, exec: GraphExec) -> HipResult<()> {
        let code = unsafe { (self.fn_graph_exec_destroy)(exec.0) };
        std::mem::forget(exec);
        self.check(code, "hipGraphExecDestroy")
    }

    pub fn graph_destroy(&self, graph: Graph) -> HipResult<()> {
        let code = unsafe { (self.fn_graph_destroy)(graph.0) };
        std::mem::forget(graph);
        self.check(code, "hipGraphDestroy")
    }

    /// Write a 32-bit value to a device address on the stream.
    /// The write is ordered with respect to other operations on the stream.
    /// Graph-safe: can be used before hipGraphLaunch to update device state
    /// that captured kernels will read (e.g., position buffers).
    pub fn stream_write_value32(&self, stream: &Stream, ptr: &DeviceBuffer, value: u32, flags: u32) -> HipResult<()> {
        let code = unsafe { (self.fn_stream_write_value32)(stream.0, ptr.as_ptr(), value, flags) };
        self.check(code, "hipStreamWriteValue32")
    }

    pub fn device_synchronize(&self) -> HipResult<()> {
        let t = std::time::Instant::now();
        let code = unsafe { (self.fn_device_synchronize)() };
        crate::ffi::launch_counters::device_sync::record(t.elapsed().as_nanos() as u64);
        self.check(code, "hipDeviceSynchronize")
    }

    /// Get GPU architecture string (e.g., "gfx1010", "gfx1030", "gfx1100").
    /// Allocates a large buffer for hipDeviceProp_t, reads gcnArchName from offset 0.
    pub fn get_arch(&self, device_id: i32) -> HipResult<String> {
        let mut buf = vec![0u8; 1024]; // hipDeviceProp_t varies by ROCm version, 1024 is safe
        let code = unsafe { (self.fn_get_device_properties)(buf.as_mut_ptr(), device_id as c_int) };
        self.check(code, "hipGetDeviceProperties")?;
        // gcnArchName is a null-terminated C string at the start of the struct
        // Actually it's at a fixed offset. On ROCm 5/6, gcnArchName is at offset 500+.
        // Safer: search for "gfx" in the buffer.
        let s = String::from_utf8_lossy(&buf);
        if let Some(pos) = s.find("gfx") {
            let arch_str = &s[pos..];
            let end = arch_str.find(|c: char| c == '\0' || c == ':' || c == ' ').unwrap_or(arch_str.len());
            Ok(arch_str[..end].to_string())
        } else {
            // Fallback: read as null-terminated string from known offsets
            // gcnArchName is typically at offset 0 in older ROCm or at a named field
            let cstr = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr() as *const c_char) };
            let name = cstr.to_string_lossy().to_string();
            if name.starts_with("gfx") {
                let end = name.find(':').unwrap_or(name.len());
                Ok(name[..end].to_string())
            } else {
                Ok("unknown".to_string())
            }
        }
    }

    /// Query a HIP device attribute by enum ID. See `hipDeviceAttribute_t` in
    /// `hip_runtime_api.h` for valid IDs. Used by the profiler to read CU count
    /// when sysfs/KFD is unavailable (Windows, restricted containers).
    pub fn get_device_attribute(&self, attr_id: i32, device_id: i32) -> HipResult<i32> {
        let mut value: c_int = 0;
        let code = unsafe { (self.fn_get_device_attribute)(&mut value, attr_id as c_int, device_id as c_int) };
        self.check(code, "hipDeviceGetAttribute")?;
        Ok(value as i32)
    }

    /// Get VRAM info: (free_bytes, total_bytes).
    pub fn get_vram_info(&self) -> HipResult<(usize, usize)> {
        let mut free: usize = 0;
        let mut total: usize = 0;
        let code = unsafe { (self.fn_mem_get_info)(&mut free, &mut total) };
        self.check(code, "hipMemGetInfo")?;
        Ok((free, total))
    }
}

// ── Handle wrappers ─────────────────────────────────────────────

/// GPU stream handle.
pub struct Stream(HipStream);
unsafe impl Send for Stream {}

/// Loaded GPU module (compiled kernels).
pub struct Module(HipModule);
unsafe impl Send for Module {}

/// Handle to a specific kernel function within a module.
pub struct Function(HipFunction);
unsafe impl Send for Function {}

/// GPU event for timing.
pub struct Event(HipEvent);
unsafe impl Send for Event {}

/// Captured GPU operation graph.
pub struct Graph(HipGraph);
unsafe impl Send for Graph {}

/// Executable (instantiated) graph ready for replay.
pub struct GraphExec(HipGraphExec);
unsafe impl Send for GraphExec {}
