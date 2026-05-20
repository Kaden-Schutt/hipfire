// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! KFD (Kernel Fusion Driver) interface for user-mode AQL compute queues.
//!
//! This bypasses the amdgpu_cs_submit ioctl path entirely.
//! Dispatch is done by writing a 64-byte AQL packet to a memory-mapped ring
//! buffer and writing a doorbell register. No syscall per dispatch.

use crate::device::Device;
use crate::{RedlineError, Result};
use std::sync::atomic::{AtomicU64, Ordering};

// KFD ioctl numbers (base 'K')
const AMDKFD_IOCTL_BASE: u8 = b'K';

// Queue types
const KFD_IOC_QUEUE_TYPE_COMPUTE_AQL: u32 = 0x2;

// Memory allocation flags
#[allow(dead_code)]
const KFD_IOC_ALLOC_MEM_FLAGS_GTT: u32 = 1 << 1;
const KFD_IOC_ALLOC_MEM_FLAGS_USERPTR: u32 = 1 << 2;
const KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE: u32 = 1 << 31;
const KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE: u32 = 1 << 30;

// Ring buffer size (64KB = 1024 AQL packets)
const RING_SIZE: u32 = 64 * 1024;
const EOP_SIZE: u64 = 4096;

/// KFD ioctl argument structs (must match kernel headers exactly)
#[repr(C)]
struct KfdGetVersionArgs {
    major_version: u32,
    minor_version: u32,
}

#[repr(C)]
#[derive(Clone)]
struct KfdProcessDeviceApertures {
    lds_base: u64,
    lds_limit: u64,
    scratch_base: u64,
    scratch_limit: u64,
    gpuvm_base: u64,
    gpuvm_limit: u64,
    gpu_id: u32,
    pad: u32,
}

#[repr(C)]
struct KfdGetProcessAperturesNewArgs {
    kfd_process_device_apertures_ptr: u64,
    num_of_nodes: u32,
    pad: u32,
}

#[repr(C)]
struct KfdAcquireVmArgs {
    drm_fd: u32,
    gpu_id: u32,
}

#[repr(C)]
struct KfdAllocMemoryArgs {
    va_addr: u64,
    size: u64,
    handle: u64,
    mmap_offset: u64,
    gpu_id: u32,
    flags: u32,
}

#[repr(C)]
struct KfdMapMemoryArgs {
    handle: u64,
    device_ids_array_ptr: u64,
    n_devices: u32,
    n_success: u32,
}

#[repr(C)]
struct KfdCreateQueueArgs {
    ring_base_address: u64,
    write_pointer_address: u64,
    read_pointer_address: u64,
    doorbell_offset: u64,
    ring_size: u32,
    gpu_id: u32,
    queue_type: u32,
    queue_percentage: u32,
    queue_priority: u32,
    queue_id: u32,
    eop_buffer_address: u64,
    eop_buffer_size: u64,
    ctx_save_restore_address: u64,
    ctx_save_restore_size: u32,
    ctl_stack_size: u32,
}

#[repr(C)]
struct KfdDestroyQueueArgs {
    queue_id: u32,
    pad: u32,
}

// ioctl number computation: _IOWR('K', nr, type) = direction(3) | size | type | nr
fn kfd_iowr<T>(nr: u8) -> libc::c_ulong {
    let size = std::mem::size_of::<T>() as libc::c_ulong;
    (3 << 30) | (size << 16) | ((AMDKFD_IOCTL_BASE as libc::c_ulong) << 8) | (nr as libc::c_ulong)
}
fn kfd_iow<T>(nr: u8) -> libc::c_ulong {
    let size = std::mem::size_of::<T>() as libc::c_ulong;
    (1 << 30) | (size << 16) | ((AMDKFD_IOCTL_BASE as libc::c_ulong) << 8) | (nr as libc::c_ulong)
}
fn kfd_ior<T>(nr: u8) -> libc::c_ulong {
    let size = std::mem::size_of::<T>() as libc::c_ulong;
    (2 << 30) | (size << 16) | ((AMDKFD_IOCTL_BASE as libc::c_ulong) << 8) | (nr as libc::c_ulong)
}

/// AQL dispatch packet (64 bytes, hardware-parsed).
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct AqlPacket {
    pub header: u16,         // [0:1]
    pub setup: u16,          // [2:3]
    pub workgroup_size_x: u16,
    pub workgroup_size_y: u16,
    pub workgroup_size_z: u16,
    pub _reserved0: u16,
    pub grid_size_x: u32,
    pub grid_size_y: u32,
    pub grid_size_z: u32,
    pub private_segment_size: u32,
    pub group_segment_size: u32,
    pub kernel_object: u64,  // GPU VA of kernel DESCRIPTOR
    pub kernarg_address: u64,
    pub _reserved1: u64,
    pub completion_signal: u64,
}

/// Result of a userptr KFD allocation.
struct KfdUserAlloc {
    handle: u64,
    gpu_va: u64,
    cpu_ptr: *mut u8,
}

/// A user-mode AQL compute queue via /dev/kfd.
pub struct AqlQueue {
    kfd_fd: i32,
    gpu_id: u32,
    queue_id: u32,
    ring_base: *mut u8,          // mmap'd ring buffer
    ring_size: u32,
    write_ptr: *mut AtomicU64,   // mmap'd write pointer (kernel manages)
    read_ptr: *mut AtomicU64,    // mmap'd read pointer
    doorbell: *mut u32,          // mmap'd doorbell register
    ring_handle: u64,            // KFD allocation handle for ring
    eop_handle: u64,             // KFD allocation handle for EOP
    signal_buf: *mut u64,        // mmap'd signal buffer for completion
    signal_handle: u64,
    signal_va: u64,
}

impl AqlQueue {
    /// Create a new AQL compute queue on the same GPU as the Device.
    pub fn new(dev: &Device) -> Result<Self> {
        // Open /dev/kfd
        let kfd_fd = unsafe {
            libc::open(b"/dev/kfd\0".as_ptr() as *const i8, libc::O_RDWR | libc::O_CLOEXEC)
        };
        if kfd_fd < 0 {
            return Err(RedlineError { code: kfd_fd, message: "failed to open /dev/kfd".into() });
        }

        // Get KFD version
        let mut ver = KfdGetVersionArgs { major_version: 0, minor_version: 0 };
        let ret = unsafe { libc::ioctl(kfd_fd, kfd_ior::<KfdGetVersionArgs>(0x01), &mut ver) };
        if ret != 0 {
            unsafe { libc::close(kfd_fd); }
            return Err(RedlineError { code: ret, message: "KFD get_version failed".into() });
        }
        eprintln!("[redline/kfd] KFD version {}.{}", ver.major_version, ver.minor_version);

        // Get process apertures to discover gpu_id
        let mut apertures = vec![KfdProcessDeviceApertures {
            lds_base: 0, lds_limit: 0, scratch_base: 0, scratch_limit: 0,
            gpuvm_base: 0, gpuvm_limit: 0, gpu_id: 0, pad: 0,
        }; 8];
        let mut get_apt = KfdGetProcessAperturesNewArgs {
            kfd_process_device_apertures_ptr: apertures.as_mut_ptr() as u64,
            num_of_nodes: apertures.len() as u32,
            pad: 0,
        };
        let ret = unsafe { libc::ioctl(kfd_fd, kfd_iowr::<KfdGetProcessAperturesNewArgs>(0x14), &mut get_apt) };
        if ret != 0 {
            unsafe { libc::close(kfd_fd); }
            return Err(RedlineError { code: ret, message: "KFD get_process_apertures_new failed".into() });
        }

        // Find GPU node (non-zero gpu_id)
        let gpu_id = apertures[..get_apt.num_of_nodes as usize]
            .iter().find(|a| a.gpu_id != 0)
            .map(|a| a.gpu_id)
            .ok_or(RedlineError { code: -1, message: "no GPU found in KFD topology".into() })?;
        eprintln!("[redline/kfd] gpu_id={}", gpu_id);

        // Acquire VM — bridge KFD and DRM address spaces
        let mut acq = KfdAcquireVmArgs { drm_fd: dev.fd as u32, gpu_id };
        let ret = unsafe { libc::ioctl(kfd_fd, kfd_iow::<KfdAcquireVmArgs>(0x15), &mut acq) };
        if ret != 0 {
            unsafe { libc::close(kfd_fd); }
            return Err(RedlineError { code: ret, message: format!("KFD acquire_vm failed: {}", std::io::Error::last_os_error()) });
        }
        eprintln!("[redline/kfd] VM acquired (drm_fd={})", dev.fd);

        // Allocate ring buffer, EOP, and signal via KFD userptr.
        // mmap system memory, then register with KFD for GPU access.
        let ring_alloc = Self::kfd_alloc_userptr(kfd_fd, gpu_id, RING_SIZE as u64)?;
        Self::kfd_map(kfd_fd, ring_alloc.handle, gpu_id)?;
        let ring_base = ring_alloc.cpu_ptr;

        let eop_alloc = Self::kfd_alloc_userptr(kfd_fd, gpu_id, EOP_SIZE)?;
        Self::kfd_map(kfd_fd, eop_alloc.handle, gpu_id)?;

        let sig_alloc = Self::kfd_alloc_userptr(kfd_fd, gpu_id, 4096)?;
        Self::kfd_map(kfd_fd, sig_alloc.handle, gpu_id)?;
        let signal_buf = sig_alloc.cpu_ptr as *mut u64;

        // CWSR (Context Wave Save/Restore) buffer — required when cwsr_enable=1
        let cwsr_size: u64 = 2 * 1024 * 1024 + 512 * 1024; // 2.5 MB
        let ctl_stack_size: u32 = 12 * 1024; // 12 KB control stack within CWSR
        let cwsr_alloc = Self::kfd_alloc_userptr(kfd_fd, gpu_id, cwsr_size)?;
        Self::kfd_map(kfd_fd, cwsr_alloc.handle, gpu_id)?;

        // Write/read pointers — must be pre-allocated (kernel expects caller to provide)
        let wptr_alloc = Self::kfd_alloc_userptr(kfd_fd, gpu_id, 4096)?;
        Self::kfd_map(kfd_fd, wptr_alloc.handle, gpu_id)?;
        let rptr_alloc = Self::kfd_alloc_userptr(kfd_fd, gpu_id, 4096)?;
        Self::kfd_map(kfd_fd, rptr_alloc.handle, gpu_id)?;

        eprintln!("[redline/kfd] ring va=0x{:x} eop va=0x{:x} cwsr va=0x{:x}",
            ring_alloc.gpu_va, eop_alloc.gpu_va, cwsr_alloc.gpu_va);

        // Create AQL queue
        let mut cq = KfdCreateQueueArgs {
            ring_base_address: ring_alloc.gpu_va,
            write_pointer_address: wptr_alloc.gpu_va,
            read_pointer_address: rptr_alloc.gpu_va,
            doorbell_offset: 0,
            ring_size: RING_SIZE,
            gpu_id,
            queue_type: KFD_IOC_QUEUE_TYPE_COMPUTE_AQL,
            queue_percentage: 100,
            queue_priority: 15,
            queue_id: 0,
            eop_buffer_address: eop_alloc.gpu_va,
            eop_buffer_size: EOP_SIZE,
            ctx_save_restore_address: cwsr_alloc.gpu_va,
            ctx_save_restore_size: cwsr_size as u32,
            ctl_stack_size,
        };
        let ret = unsafe { libc::ioctl(kfd_fd, kfd_iowr::<KfdCreateQueueArgs>(0x02), &mut cq) };
        if ret != 0 {
            unsafe { libc::close(kfd_fd); }
            return Err(RedlineError { code: ret, message: format!("KFD create_queue failed: {}", std::io::Error::last_os_error()) });
        }
        eprintln!("[redline/kfd] AQL queue created: id={}, doorbell_offset=0x{:x}",
            cq.queue_id, cq.doorbell_offset);

        // Write/read pointers are already CPU-mapped (userptr)
        let write_ptr = wptr_alloc.cpu_ptr as *mut AtomicU64;
        let read_ptr = rptr_alloc.cpu_ptr as *mut AtomicU64;

        // mmap doorbell page
        let doorbell_page = unsafe {
            libc::mmap(std::ptr::null_mut(), 8192, libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED, kfd_fd, cq.doorbell_offset as i64)
        };
        if doorbell_page == libc::MAP_FAILED {
            unsafe { libc::close(kfd_fd); }
            return Err(RedlineError { code: -1, message: format!("mmap doorbell failed: {}", std::io::Error::last_os_error()) });
        }

        eprintln!("[redline/kfd] AQL queue ready — user-mode dispatch enabled");

        Ok(Self {
            kfd_fd,
            gpu_id,
            queue_id: cq.queue_id,
            ring_base: ring_base as *mut u8,
            ring_size: RING_SIZE,
            write_ptr,
            read_ptr,
            doorbell: doorbell_page as *mut u32,
            ring_handle: ring_alloc.handle,
            eop_handle: eop_alloc.handle,
            signal_buf,
            signal_handle: sig_alloc.handle,
            signal_va: sig_alloc.gpu_va,
        })
    }

    /// Dispatch a kernel via AQL — NO SYSCALL.
    /// `kd_va`: GPU VA of the kernel descriptor (NOT code entry).
    /// `kernarg_va`: GPU VA of kernarg buffer.
    pub fn dispatch(
        &self,
        kd_va: u64,
        grid: [u32; 3],
        block: [u32; 3],
        kernarg_va: u64,
        lds_bytes: u32,
    ) {
        let write_idx = unsafe { &*self.write_ptr };
        let idx = write_idx.load(Ordering::Relaxed);
        let ring_mask = (self.ring_size as u64 / 64) - 1;
        let pkt_offset = ((idx & ring_mask) * 64) as usize;
        let pkt_ptr = unsafe { self.ring_base.add(pkt_offset) as *mut AqlPacket };

        let ndims = if grid[2] > 1 { 3u16 } else if grid[1] > 1 { 2 } else { 1 };

        // Write payload first (everything except header)
        let pkt = AqlPacket {
            header: 0, // written last with atomic store
            setup: ndims,
            workgroup_size_x: block[0] as u16,
            workgroup_size_y: block[1] as u16,
            workgroup_size_z: block[2] as u16,
            _reserved0: 0,
            grid_size_x: grid[0] * block[0],
            grid_size_y: grid[1] * block[1],
            grid_size_z: grid[2] * block[2],
            private_segment_size: 0,
            group_segment_size: lds_bytes,
            kernel_object: kd_va,
            kernarg_address: kernarg_va,
            _reserved1: 0,
            completion_signal: 0,
        };

        unsafe {
            // Write all fields except header
            let src = &pkt as *const AqlPacket as *const u8;
            let dst = pkt_ptr as *mut u8;
            std::ptr::copy_nonoverlapping(src.add(2), dst.add(2), 62);
            // Memory barrier before header write
            std::sync::atomic::fence(Ordering::Release);
            // Write header atomically — makes packet visible to hardware
            // header: type=2 (dispatch), barrier=1, acquire=2 (agent), release=2 (agent)
            let header: u16 = (2 << 0) | (1 << 8) | (2 << 9) | (2 << 11);
            (dst as *mut u16).write_volatile(header);
        }

        // Increment write pointer and ring doorbell
        write_idx.store(idx + 1, Ordering::Release);
        unsafe {
            self.doorbell.write_volatile(idx as u32 + 1);
        }
    }

    /// Dispatch and wait for completion using a signal.
    pub fn dispatch_and_wait(
        &self,
        kd_va: u64,
        grid: [u32; 3],
        block: [u32; 3],
        kernarg_va: u64,
        lds_bytes: u32,
    ) {
        // Reset signal
        unsafe { self.signal_buf.write_volatile(1); }

        let write_idx = unsafe { &*self.write_ptr };
        let idx = write_idx.load(Ordering::Relaxed);
        let ring_mask = (self.ring_size as u64 / 64) - 1;
        let pkt_offset = ((idx & ring_mask) * 64) as usize;
        let pkt_ptr = unsafe { self.ring_base.add(pkt_offset) };

        let ndims = if grid[2] > 1 { 3u16 } else if grid[1] > 1 { 2 } else { 1 };

        unsafe {
            let dst = pkt_ptr as *mut u8;
            // setup
            (dst.add(2) as *mut u16).write(ndims);
            // workgroup sizes
            (dst.add(4) as *mut u16).write(block[0] as u16);
            (dst.add(6) as *mut u16).write(block[1] as u16);
            (dst.add(8) as *mut u16).write(block[2] as u16);
            (dst.add(10) as *mut u16).write(0);
            // grid sizes (total work items)
            (dst.add(12) as *mut u32).write(grid[0] * block[0]);
            (dst.add(16) as *mut u32).write(grid[1] * block[1]);
            (dst.add(20) as *mut u32).write(grid[2] * block[2]);
            // private + group segment
            (dst.add(24) as *mut u32).write(0);
            (dst.add(28) as *mut u32).write(lds_bytes);
            // kernel object (descriptor VA)
            (dst.add(32) as *mut u64).write(kd_va);
            // kernarg
            (dst.add(40) as *mut u64).write(kernarg_va);
            // reserved
            (dst.add(48) as *mut u64).write(0);
            // completion signal — GPU decrements this to 0
            (dst.add(56) as *mut u64).write(self.signal_va);

            // Fence + header write
            std::sync::atomic::fence(Ordering::Release);
            let header: u16 = (2 << 0) | (1 << 8) | (2 << 9) | (2 << 11);
            (dst as *mut u16).write_volatile(header);
        }

        // Ring doorbell
        unsafe {
            let wi = &*self.write_ptr;
            wi.store(idx + 1, Ordering::Release);
            self.doorbell.write_volatile(idx as u32 + 1);
        }

        // Spin-wait for completion (signal decremented from 1 to 0)
        let timeout = std::time::Instant::now();
        loop {
            let val = unsafe { self.signal_buf.read_volatile() };
            if val == 0 { break; }
            if timeout.elapsed().as_secs() > 10 {
                eprintln!("[redline/kfd] TIMEOUT waiting for AQL dispatch (signal={})", val);
                break;
            }
            std::hint::spin_loop();
        }
    }

    /// Allocate system memory and register with KFD for GPU access.
    fn kfd_alloc_userptr(kfd_fd: i32, gpu_id: u32, size: u64) -> Result<KfdUserAlloc> {
        // mmap anonymous system memory
        let cpu_ptr = unsafe {
            libc::mmap(std::ptr::null_mut(), size as usize, libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANONYMOUS | libc::MAP_PRIVATE, -1, 0)
        };
        if cpu_ptr == libc::MAP_FAILED {
            return Err(RedlineError { code: -1, message: "mmap anon failed".into() });
        }
        // Zero it
        unsafe { std::ptr::write_bytes(cpu_ptr as *mut u8, 0, size as usize); }

        // Register with KFD as userptr
        let mut args = KfdAllocMemoryArgs {
            va_addr: cpu_ptr as u64,
            size,
            handle: 0,
            mmap_offset: cpu_ptr as u64, // for userptr, mmap_offset = cpu address
            gpu_id,
            flags: KFD_IOC_ALLOC_MEM_FLAGS_USERPTR | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
                 | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE,
        };
        let ret = unsafe { libc::ioctl(kfd_fd, kfd_iowr::<KfdAllocMemoryArgs>(0x16), &mut args) };
        if ret != 0 {
            unsafe { libc::munmap(cpu_ptr, size as usize); }
            return Err(RedlineError { code: ret,
                message: format!("KFD alloc_userptr({} bytes) failed: {}", size, std::io::Error::last_os_error()) });
        }
        let gpu_va = args.va_addr;
        Ok(KfdUserAlloc { handle: args.handle, gpu_va, cpu_ptr: cpu_ptr as *mut u8 })
    }

    /// KFD memory allocation helper (GTT/VRAM). Returns (handle, gpu_va, mmap_offset).
    #[allow(dead_code)]
    fn kfd_alloc(kfd_fd: i32, gpu_id: u32, size: u64, flags: u32) -> Result<(u64, u64, u64)> {
        let mut args = KfdAllocMemoryArgs {
            va_addr: 0,
            size,
            handle: 0,
            mmap_offset: 0,
            gpu_id,
            flags,
        };
        let ret = unsafe { libc::ioctl(kfd_fd, kfd_iowr::<KfdAllocMemoryArgs>(0x16), &mut args) };
        if ret != 0 {
            return Err(RedlineError { code: ret,
                message: format!("KFD alloc_memory({} bytes, flags=0x{:x}) failed: {}",
                    size, flags, std::io::Error::last_os_error()) });
        }
        Ok((args.handle, args.va_addr, args.mmap_offset))
    }

    /// Map KFD memory to GPU.
    fn kfd_map(kfd_fd: i32, handle: u64, gpu_id: u32) -> Result<()> {
        let mut gpu_ids = [gpu_id];
        let mut args = KfdMapMemoryArgs {
            handle,
            device_ids_array_ptr: gpu_ids.as_mut_ptr() as u64,
            n_devices: 1,
            n_success: 0,
        };
        let ret = unsafe { libc::ioctl(kfd_fd, kfd_iowr::<KfdMapMemoryArgs>(0x18), &mut args) };
        if ret != 0 {
            return Err(RedlineError { code: ret,
                message: format!("KFD map_memory failed: {}", std::io::Error::last_os_error()) });
        }
        Ok(())
    }

    pub fn destroy(&self) {
        let mut dq = KfdDestroyQueueArgs { queue_id: self.queue_id, pad: 0 };
        unsafe {
            libc::ioctl(self.kfd_fd, kfd_iowr::<KfdDestroyQueueArgs>(0x03), &mut dq);
            libc::close(self.kfd_fd);
        }
    }
}
