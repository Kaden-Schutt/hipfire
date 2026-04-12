//! hip-bridge: Safe Rust FFI to AMD HIP runtime via dlopen.
//! Modeled after rustane's ane-bridge — no link-time dependency on libamdhip64.

mod ffi;
mod error;
mod kernarg;

pub use error::{HipError, HipResult};
pub use ffi::{Event, Function, Graph, GraphExec, HipRuntime, Module, Stream};
pub use ffi::launch_counters;
pub use kernarg::KernargBlob;

/// Re-export memory copy direction for callers.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    Default = 4,
}

/// Opaque GPU buffer handle. Tracks pointer + size for safety.
pub struct DeviceBuffer {
    ptr: *mut std::ffi::c_void,
    size: usize,
}

impl DeviceBuffer {
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Create a non-owning DeviceBuffer from a raw pointer and size.
    /// The caller must ensure the pointer is valid GPU memory.
    /// The resulting buffer must NOT be freed (it doesn't own the memory).
    pub unsafe fn from_raw(ptr: *mut std::ffi::c_void, size: usize) -> DeviceBuffer {
        DeviceBuffer { ptr, size }
    }

    /// Create a non-owning alias to the same GPU memory.
    /// The alias must not outlive the original buffer.
    /// Used for reshaping tensors without reallocating.
    /// # Safety
    /// Caller must ensure the alias doesn't outlive the original.
    pub unsafe fn alias(&self) -> DeviceBuffer {
        DeviceBuffer { ptr: self.ptr, size: self.size }
    }
}

// DeviceBuffer is Send — GPU pointers can be sent between threads.
// They are NOT Sync — concurrent access requires stream synchronization.
unsafe impl Send for DeviceBuffer {}
