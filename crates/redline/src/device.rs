//! GPU device initialization and info queries via libdrm_amdgpu.

use crate::drm::*;
use crate::{RedlineError, Result};
#[allow(unused_imports)]
use std::ffi::c_void;

/// An open GPU device.
pub struct Device {
    pub(crate) drm: DrmLib,
    pub(crate) handle: AmdgpuDeviceHandle,
    pub fd: i32,
    pub info: GpuInfo,
}

/// Parsed GPU information.
#[derive(Debug)]
pub struct GpuInfo {
    pub asic_id: u32,
    pub family_id: u32,
    pub chip_rev: u32,
    pub chip_external_rev: u32,
    pub num_cu: u32,
    pub num_shader_engines: u32,
    pub vram_total_bytes: u64,
    pub vram_usable_bytes: u64,
    pub vram_used_bytes: u64,
    pub gfx_arch: String,
}

impl Device {
    /// Open the GPU at /dev/dri/renderD128 (or specified path).
    pub fn open(render_node: Option<&str>) -> Result<Self> {
        let path = render_node.unwrap_or("/dev/dri/renderD128");
        let fd = unsafe {
            libc::open(
                std::ffi::CString::new(path).unwrap().as_ptr(),
                libc::O_RDWR | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            return Err(RedlineError {
                code: fd,
                message: format!("failed to open {path}. Check permissions (user must be in 'render' or 'video' group)"),
            });
        }

        let drm = DrmLib::load()?;

        let mut handle: AmdgpuDeviceHandle = std::ptr::null_mut();
        let mut major = 0u32;
        let mut minor = 0u32;
        let ret = unsafe { (drm.device_initialize)(fd, &mut major, &mut minor, &mut handle) };
        if ret != 0 {
            unsafe { libc::close(fd); }
            return Err(RedlineError {
                code: ret,
                message: format!("amdgpu_device_initialize failed: {ret}"),
            });
        }

        eprintln!("[redline] DRM version {major}.{minor}");

        // Query GPU info
        let mut gpu_info = AmdgpuGpuInfo::default();
        let ret = unsafe { (drm.query_gpu_info)(handle, &mut gpu_info) };
        if ret != 0 {
            return Err(RedlineError { code: ret, message: format!("query_gpu_info failed: {ret}") });
        }

        // Query VRAM heap
        let mut heap = HeapInfo::default();
        let ret = unsafe { (drm.query_heap_info)(handle, AMDGPU_GEM_DOMAIN_VRAM, 0, &mut heap) };
        if ret != 0 {
            return Err(RedlineError { code: ret, message: format!("query_heap_info failed: {ret}") });
        }

        // Map family_id + asic_id to gfx arch string
        // family 143 (AMDGPU_FAMILY_NV) covers both RDNA1 and RDNA2
        let gfx_arch = match gpu_info.family_id {
            141 => {
                // AMDGPU_FAMILY_AI covers Vega 10/20. Vega 20 devices report
                // gfx906 through ROCm and use the same wave64 execution model
                // as the hipfire runtime port.
                match gpu_info.chip_external_rev {
                    0x3c | 0x3d | 0x3e | 0x3f => "gfx906",
                    _ => "gfx900",
                }
            },
            142 => "gfx902",  // AMDGPU_FAMILY_RV (Raven Ridge)
            143 => {
                // AMDGPU_FAMILY_NV: distinguish by asic_id
                // Navi10=0x731x, Navi12=0x736x, Navi14=0x734x (RDNA1)
                // Navi21=0x73Ax, Navi22=0x73Cx, Navi23=0x73Ex (RDNA2)
                match (gpu_info.asic_id >> 4) & 0xF {
                    1 => "gfx1010",       // Navi 10 (RX 5600/5700)
                    6 => "gfx1011",       // Navi 12
                    3 | 4 => "gfx1012",   // Navi 14 (RX 5300/5500)
                    0xA | 0xB => "gfx1030", // Navi 21 (RX 6800/6900)
                    0xC | 0xD => "gfx1031", // Navi 22 (RX 6700)
                    0xE | 0xF => "gfx1032", // Navi 23 (RX 6600)
                    _ => "gfx10xx",
                }
            },
            145 | 146 | 147 => "gfx1100", // RDNA3
            148 | 149 => "gfx1200",        // RDNA4
            _ => "unknown",
        };

        let info = GpuInfo {
            asic_id: gpu_info.asic_id,
            family_id: gpu_info.family_id,
            chip_rev: gpu_info.chip_rev,
            chip_external_rev: gpu_info.chip_external_rev,
            num_cu: gpu_info.cu_active_number,
            num_shader_engines: gpu_info.num_shader_engines,
            vram_total_bytes: heap.total_heap_size,
            vram_usable_bytes: heap.usable_heap_size,
            vram_used_bytes: heap.heap_usage,
            gfx_arch: gfx_arch.to_string(),
        };

        eprintln!("[redline] GPU: {} (asic 0x{:x}) — {} CUs, {} SEs, {:.1} GB VRAM",
            info.gfx_arch, info.asic_id, info.num_cu, info.num_shader_engines,
            info.vram_total_bytes as f64 / 1e9);

        Ok(Self { drm, handle, fd, info })
    }

    /// Allocate VRAM buffer object with GPU virtual address mapping.
    pub fn alloc_vram(&self, size: u64) -> Result<GpuBuffer> {
        let aligned_size = (size + 4095) & !4095; // page-align

        // 1. Allocate backing storage (BO)
        let req = AmdgpuBoAllocRequest {
            alloc_size: aligned_size,
            phys_alignment: 4096,
            preferred_heap: AMDGPU_GEM_DOMAIN_VRAM,
            flags: AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED,
        };
        let mut bo_handle: AmdgpuBoHandle = std::ptr::null_mut();
        let ret = unsafe { (self.drm.bo_alloc)(self.handle, &req, &mut bo_handle) };
        if ret != 0 {
            return Err(RedlineError { code: ret, message: format!("bo_alloc({aligned_size} bytes) failed: {ret}") });
        }

        // 2. Allocate GPU virtual address range
        let mut gpu_addr: u64 = 0;
        let mut va_handle: AmdgpuVaHandle = std::ptr::null_mut();
        let ret = unsafe {
            (self.drm.va_range_alloc)(self.handle, 0, aligned_size, 4096, 0, &mut gpu_addr, &mut va_handle, 0)
        };
        if ret != 0 {
            unsafe { (self.drm.bo_free)(bo_handle); }
            return Err(RedlineError { code: ret, message: format!("va_range_alloc failed: {ret}") });
        }

        // 3. Map BO to virtual address
        let ret = unsafe {
            (self.drm.bo_va_op)(bo_handle, 0, aligned_size, gpu_addr, 0, AMDGPU_VA_OP_MAP)
        };
        if ret != 0 {
            unsafe {
                (self.drm.va_range_free)(va_handle);
                (self.drm.bo_free)(bo_handle);
            }
            return Err(RedlineError { code: ret, message: format!("bo_va_op MAP failed: {ret}") });
        }

        Ok(GpuBuffer {
            handle: bo_handle,
            va_handle,
            gpu_addr,
            size: aligned_size,
        })
    }

    /// Upload data from CPU to GPU buffer.
    pub fn upload(&self, buf: &GpuBuffer, data: &[u8]) -> Result<()> {
        assert!(data.len() as u64 <= buf.size);
        let mut cpu_ptr: *mut c_void = std::ptr::null_mut();
        let ret = unsafe { (self.drm.bo_cpu_map)(buf.handle, &mut cpu_ptr) };
        if ret != 0 {
            return Err(RedlineError { code: ret, message: format!("bo_cpu_map failed: {ret}") });
        }
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), cpu_ptr as *mut u8, data.len());
            (self.drm.bo_cpu_unmap)(buf.handle);
        }
        Ok(())
    }

    /// Download data from GPU buffer to CPU.
    pub fn download(&self, buf: &GpuBuffer, data: &mut [u8]) -> Result<()> {
        assert!(data.len() as u64 <= buf.size);
        let mut cpu_ptr: *mut c_void = std::ptr::null_mut();
        let ret = unsafe { (self.drm.bo_cpu_map)(buf.handle, &mut cpu_ptr) };
        if ret != 0 {
            return Err(RedlineError { code: ret, message: format!("bo_cpu_map failed: {ret}") });
        }
        unsafe {
            std::ptr::copy_nonoverlapping(cpu_ptr as *const u8, data.as_mut_ptr(), data.len());
            (self.drm.bo_cpu_unmap)(buf.handle);
        }
        Ok(())
    }

    /// Free a GPU buffer (unmap VA + free BO).
    pub fn free_buffer(&self, buf: GpuBuffer) -> Result<()> {
        unsafe {
            (self.drm.bo_va_op)(buf.handle, 0, buf.size, buf.gpu_addr, 0, AMDGPU_VA_OP_UNMAP);
            (self.drm.va_range_free)(buf.va_handle);
            let ret = (self.drm.bo_free)(buf.handle);
            if ret != 0 {
                return Err(RedlineError { code: ret, message: format!("bo_free failed: {ret}") });
            }
        }
        Ok(())
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            (self.drm.device_deinitialize)(self.handle);
            libc::close(self.fd);
        }
    }
}

/// A GPU buffer object (VRAM allocation + VA mapping).
pub struct GpuBuffer {
    pub(crate) handle: AmdgpuBoHandle,
    pub(crate) va_handle: AmdgpuVaHandle,
    pub gpu_addr: u64,
    pub size: u64,
}
