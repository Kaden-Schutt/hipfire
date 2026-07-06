//! Shared GTT dma-buf: allocate a CPU-accessible GTT buffer on the GPU's amdgpu render node
//! and PRIME-export it, so the NPU (`XdnaDevice::import_dmabuf` / `NpuGemmMp::attach_*_dmabuf`)
//! and the CPU address the *same physical pages* on this UMA APU — the zero-copy
//! GPU⇄NPU⇄CPU handoff. Hoisted out of the raw-ioctl probes (the export belongs in the GPU
//! crate). See `docs/npu/concurrent-prefill-split-design.md`.
//!
//! NOTE: this is a raw-DRM GTT allocation, distinct from a HIP tensor. A GPU *compute* kernel
//! that operates on it needs `hipHostRegister` on `as_mut_slice()` (a follow-up); today it is
//! the CPU/NPU-shared staging buffer the heterogeneous handoff needs.

use crate::Gpu;
use hip_bridge::{DeviceBuffer, HipError, HipResult, ImportedBuffer};
use hipfire_gpu_types::{DType, GpuTensor};
use std::os::fd::RawFd;

const fn iowr(nr: u32, size: u32) -> libc::c_ulong {
    ((3u32 << 30) | (size << 16) | (0x64u32 << 8) | nr) as libc::c_ulong
}
const fn iow(nr: u32, size: u32) -> libc::c_ulong {
    ((1u32 << 30) | (size << 16) | (0x64u32 << 8) | nr) as libc::c_ulong
}
const GEM_CREATE: libc::c_ulong = iowr(0x40, 32); // amdgpu union drm_amdgpu_gem_create
const GEM_MMAP: libc::c_ulong = iowr(0x41, 8); // amdgpu union drm_amdgpu_gem_mmap
const PRIME_HANDLE_TO_FD: libc::c_ulong = iowr(0x2d, 12); // core drm_prime_handle
const GEM_CLOSE: libc::c_ulong = iow(0x09, 8); // core drm_gem_close { handle, pad }
const DOMAIN_GTT: u64 = 0x2;
const CPU_ACCESS_REQUIRED: u64 = 1;

/// A CPU-accessible GTT buffer PRIME-exported as a dma-buf. The CPU reads/writes it via
/// [`Self::as_mut_slice`]; the NPU imports [`Self::dmabuf_fd`]; all address the same pages.
/// Drops munmap + close the export fd + free the GEM + close the render node.
pub struct SharedGttBuffer {
    render_fd: RawFd,
    gem_handle: u32,
    ptr: *mut u8,
    len: usize,
    dmabuf_fd: RawFd,
}

// The pages are shared physical memory; the handle is just an owned staging buffer.
unsafe impl Send for SharedGttBuffer {}

impl SharedGttBuffer {
    /// CPU view of the shared pages.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    /// Mutable CPU view of the shared pages.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
    /// The PRIME-exported dma-buf fd, to import on the NPU. Borrowed — the buffer keeps
    /// ownership; the importer `dma_buf_get`s its own ref, so it stays valid after.
    pub fn dmabuf_fd(&self) -> RawFd {
        self.dmabuf_fd
    }
    /// Byte length.
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for SharedGttBuffer {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                libc::munmap(self.ptr as *mut libc::c_void, self.len);
            }
            if self.dmabuf_fd >= 0 {
                libc::close(self.dmabuf_fd);
            }
            let mut close = [self.gem_handle, 0u32];
            libc::ioctl(
                self.render_fd,
                GEM_CLOSE,
                close.as_mut_ptr() as *mut libc::c_void,
            );
            libc::close(self.render_fd);
        }
    }
}

#[repr(C)]
#[derive(Default)]
struct GemCreate {
    bo_size: u64,
    alignment: u64,
    domains: u64,
    domain_flags: u64,
}
#[repr(C)]
struct PrimeHandle {
    handle: u32,
    flags: u32,
    fd: i32,
}

impl Gpu {
    /// Allocate a CPU-accessible GTT buffer on this GPU's amdgpu render node (`renderD{128+
    /// device_id}`) and PRIME-export it as a dma-buf. The NPU imports the fd
    /// ([`SharedGttBuffer::dmabuf_fd`] → `NpuGemmMp::attach_input_dmabuf`/`attach_output_dmabuf`)
    /// and the CPU uses [`SharedGttBuffer::as_mut_slice`] — all three engines share the same
    /// physical pages (UMA zero-copy), removing host-copy handoffs in a heterogeneous pipeline.
    pub fn alloc_shared_gtt(&self, bytes: usize) -> HipResult<SharedGttBuffer> {
        let node = format!("/dev/dri/renderD{}\0", 128 + self.device_id.max(0));
        let render_fd = unsafe { libc::open(node.as_ptr() as *const libc::c_char, libc::O_RDWR) };
        if render_fd < 0 {
            return Err(HipError::new(
                0,
                &format!("alloc_shared_gtt: open {node} failed"),
            ));
        }
        let fail = |fd: RawFd, stage: &str| -> HipError {
            let e = std::io::Error::last_os_error();
            unsafe { libc::close(fd) };
            HipError::new(0, &format!("alloc_shared_gtt: {stage}: {e}"))
        };

        let mut gc = GemCreate {
            bo_size: bytes as u64,
            alignment: 4096,
            domains: DOMAIN_GTT,
            domain_flags: CPU_ACCESS_REQUIRED,
        };
        if unsafe {
            libc::ioctl(
                render_fd,
                GEM_CREATE,
                &mut gc as *mut _ as *mut libc::c_void,
            )
        } != 0
        {
            return Err(fail(render_fd, "GEM_CREATE(GTT)"));
        }
        let gem_handle = unsafe { *(&gc as *const _ as *const u32) }; // out.handle @ offset 0

        let mut mm: u64 = gem_handle as u64;
        if unsafe { libc::ioctl(render_fd, GEM_MMAP, &mut mm as *mut _ as *mut libc::c_void) } != 0
        {
            return Err(fail(render_fd, "GEM_MMAP"));
        }
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                render_fd,
                mm as libc::off_t,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(fail(render_fd, "mmap"));
        }

        let mut ph = PrimeHandle {
            handle: gem_handle,
            flags: (libc::O_RDWR | libc::O_CLOEXEC) as u32,
            fd: -1,
        };
        if unsafe {
            libc::ioctl(
                render_fd,
                PRIME_HANDLE_TO_FD,
                &mut ph as *mut _ as *mut libc::c_void,
            )
        } != 0
        {
            unsafe { libc::munmap(ptr, bytes) };
            return Err(fail(render_fd, "PRIME_HANDLE_TO_FD"));
        }

        Ok(SharedGttBuffer {
            render_fd,
            gem_handle,
            ptr: ptr as *mut u8,
            len: bytes,
            dmabuf_fd: ph.fd,
        })
    }

    /// Import an external dma-buf (e.g. [`SharedGttBuffer::dmabuf_fd`], or one exported by the
    /// NPU) as a NATIVE GPU tensor — a GPU compute kernel operates on the shared pages
    /// directly (zero-copy), via HIP external-memory import (`hipImportExternalMemory`).
    /// `bytes` is the exported buffer's byte size; `shape`/`dtype` are metadata. Verified on
    /// ROCm-7.14/gfx1151. This is the GPU-side mirror of the NPU's `import_dmabuf` — the two
    /// complete the three-engine (GPU/NPU/CPU) native-import triangle over one dma-buf.
    pub fn import_dmabuf(
        &self,
        fd: RawFd,
        bytes: usize,
        shape: &[usize],
        dtype: DType,
    ) -> HipResult<ImportedTensor> {
        self.bind_thread()?;
        let buf = self.hip.import_dmabuf(fd, bytes)?;
        Ok(ImportedTensor {
            buf,
            shape: shape.to_vec(),
            dtype,
        })
    }
}

/// A GPU tensor backed by an imported dma-buf (via HIP external-memory). Owns the imported
/// external-memory object; drop frees the mapping + destroys it. A kernel runs on it through
/// [`Self::view`].
pub struct ImportedTensor {
    buf: ImportedBuffer,
    shape: Vec<usize>,
    dtype: DType,
}

impl ImportedTensor {
    /// A non-owning [`GpuTensor`] view for kernel dispatch. Valid only while `self` is alive;
    /// do NOT `free_tensor` it — the memory is owned by the imported dma-buf.
    pub fn view(&self) -> GpuTensor {
        GpuTensor {
            buf: unsafe { DeviceBuffer::from_raw(self.buf.as_ptr(), self.buf.size()) },
            shape: self.shape.clone(),
            dtype: self.dtype,
        }
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn byte_size(&self) -> usize {
        self.buf.size()
    }
}
