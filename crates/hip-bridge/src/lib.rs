// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 alpineq
// hipfire — see LICENSE and NOTICE in the project root.

//! hip-bridge: Safe Rust FFI to AMD HIP runtime via dlopen.
//! Modeled after rustane's ane-bridge — no link-time dependency on libamdhip64.

mod error;
mod ffi;
mod kernarg;
mod rccl;
mod rocblas;
mod rocsolver;
mod vmm;

pub use error::{
    HipError, HipResult, HIP_ERROR_INVALID_IMAGE, HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED,
    HIP_ERROR_PEER_ACCESS_NOT_ENABLED, HIP_ERROR_PEER_ACCESS_UNSUPPORTED,
};
pub use ffi::launch_counters;
pub use ffi::{
    Event, Function, Graph, GraphExec, HipMemAccessDesc, HipMemAllocationProp,
    HipMemGenericAllocationHandle, HipMemLocation, HipPointerAttribute, HipRuntime, Module, Stream,
    HIP_EVENT_DISABLE_TIMING, HIP_EVENT_RELEASE_TO_SYSTEM, HIP_MEM_ALLOCATION_GRANULARITY_MINIMUM,
    HIP_MEM_ALLOCATION_GRANULARITY_RECOMMENDED,
};
pub use kernarg::KernargBlob;
pub use rccl::{RcclComms, RcclDataType, RcclError, RcclRedOp, RcclResult, NCCL_SUCCESS};
pub use rocblas::{Rocblas, RocblasDatatype, RocblasError, RocblasOperation, RocblasResult};
pub use rocsolver::{
    Rocsolver, RocsolverError, RocsolverResult, RocblasDiagonal, RocblasFill, ROCSOLVER_STATUS_SUCCESS,
};
pub use vmm::{clear_vmm_faults, inject_vmm_fault, VmmArena, VmmFaultKind};

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

/// Mirrors `hipMemoryType`. FFI stores raw `u32`; use `from_raw` to convert.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryType {
    Unregistered = 0,
    Host = 1,
    Device = 2,
    Managed = 3,
    Array = 10,
    Unified = 11,
}

impl MemoryType {
    pub fn from_raw(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Unregistered),
            1 => Some(Self::Host),
            2 => Some(Self::Device),
            3 => Some(Self::Managed),
            10 => Some(Self::Array),
            11 => Some(Self::Unified),
            _ => None,
        }
    }
}

/// Opaque GPU buffer handle. Tracks pointer + size for safety.
pub struct DeviceBuffer {
    ptr: *mut std::ffi::c_void,
    size: usize,
    ownership: DeviceBufferOwnership,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeviceBufferOwnership {
    HipMalloc,
    Vmm,
    Borrowed,
}

impl DeviceBuffer {
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn is_hip_allocation(&self) -> bool {
        self.ownership == DeviceBufferOwnership::HipMalloc
    }

    pub fn is_vmm_owner(&self) -> bool {
        self.ownership == DeviceBufferOwnership::Vmm
    }

    pub fn is_borrowed(&self) -> bool {
        self.ownership == DeviceBufferOwnership::Borrowed
    }

    /// Create a non-owning DeviceBuffer from a raw pointer and size.
    /// The caller must ensure the pointer is valid GPU memory.
    /// The resulting buffer must NOT be freed (it doesn't own the memory).
    ///
    /// # Safety
    ///
    /// `ptr` must point to at least `size` bytes of valid GPU-accessible
    /// memory for the lifetime of the returned non-owning wrapper.
    pub unsafe fn from_raw(ptr: *mut std::ffi::c_void, size: usize) -> DeviceBuffer {
        DeviceBuffer {
            ptr,
            size,
            ownership: DeviceBufferOwnership::Borrowed,
        }
    }

    /// Create the unique owner descriptor for a VMM arena base address.
    ///
    /// # Safety
    ///
    /// The caller must register exactly one such descriptor with the VMM owner
    /// that will unmap and release it. Aliases must use `from_raw` or `alias`.
    pub unsafe fn from_vmm_owner(ptr: *mut std::ffi::c_void, size: usize) -> DeviceBuffer {
        DeviceBuffer {
            ptr,
            size,
            ownership: DeviceBufferOwnership::Vmm,
        }
    }

    /// Create a non-owning alias to the same GPU memory.
    /// The alias must not outlive the original buffer.
    /// Used for reshaping tensors without reallocating.
    /// # Safety
    /// Caller must ensure the alias doesn't outlive the original.
    pub unsafe fn alias(&self) -> DeviceBuffer {
        DeviceBuffer {
            ptr: self.ptr,
            size: self.size,
            ownership: DeviceBufferOwnership::Borrowed,
        }
    }
}

// DeviceBuffer is Send — GPU pointers can be sent between threads.
// They are NOT Sync — concurrent access requires stream synchronization.
unsafe impl Send for DeviceBuffer {}

#[cfg(test)]
mod device_buffer_tests {
    use super::*;

    #[test]
    fn raw_and_alias_buffers_are_borrowed() {
        let raw = unsafe { DeviceBuffer::from_raw(std::ptr::dangling_mut(), 4096) };
        assert!(raw.is_borrowed());
        assert!(!raw.is_hip_allocation());
        assert!(!raw.is_vmm_owner());

        let alias = unsafe { raw.alias() };
        assert!(alias.is_borrowed());
    }

    #[test]
    fn vmm_owner_marker_is_distinct_from_views() {
        let owner = unsafe { DeviceBuffer::from_vmm_owner(std::ptr::dangling_mut(), 4096) };
        assert!(owner.is_vmm_owner());
        assert!(!owner.is_borrowed());
        let view = unsafe { owner.alias() };
        assert!(view.is_borrowed());
        assert!(!view.is_vmm_owner());
    }
}
