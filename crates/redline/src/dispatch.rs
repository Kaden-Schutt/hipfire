// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! High-level compute dispatch — builds PM4 internally, handles kernarg layout.
//!
//! Users load a module, get kernels, and dispatch without touching PM4 packets.

use crate::device::{Device, GpuBuffer};
use crate::hsaco::{HsacoModule, KernelMeta};
use crate::queue::ComputeQueue;
use crate::{RedlineError, Result};

/// A loaded GPU module — ELF uploaded to VRAM with parsed kernels.
pub struct LoadedModule {
    pub kernels: Vec<Kernel>,
    pub code_buf: GpuBuffer,
}

/// A compute kernel ready for dispatch.
pub struct Kernel {
    pub name: String,
    pub code_va: u64,
    pub pgm_rsrc1: u32,
    pub pgm_rsrc2: u32,
    pub kernarg_size: u64,
    pub group_segment_size: u32,
    /// Total user SGPRs (private seg buf + dispatch ptr + kernarg ptr + ...)
    user_sgpr_count: u32,
    /// Index within user SGPRs where kernarg pointer goes (None if no kernarg)
    kernarg_sgpr_idx: Option<u32>,
}

/// A command buffer that accumulates PM4 dispatch packets.
pub struct CommandBuffer {
    pub(crate) dwords: Vec<u32>,
}

// PM4 helpers
fn pkt3(opcode: u32, body_count: u32) -> u32 {
    (3u32 << 30) | ((body_count - 1) << 16) | (opcode << 8) | (1 << 1) // SHADER_TYPE=1 (compute)
}

const SET_SH_REG: u32 = 0x76;
const DISPATCH_DIRECT: u32 = 0x15;
const RELEASE_MEM: u32 = 0x49;
const ACQUIRE_MEM: u32 = 0x58;

impl Device {
    /// Load a .hsaco module: parse ELF, upload code to VRAM, return ready-to-dispatch kernels.
    pub fn load_module(&self, hsaco_bytes: &[u8]) -> Result<LoadedModule> {
        let module = HsacoModule::from_bytes(hsaco_bytes.to_vec())?;
        let code_buf = self.alloc_vram(module.elf.len() as u64)?;
        self.upload(&code_buf, &module.elf)?;

        let kernels: Vec<Kernel> = module.kernels.iter().map(|km| {
            let kd_off = km.kd_offset as usize;
            let kcp = if kd_off + 58 <= module.elf.len() {
                u16::from_le_bytes([module.elf[kd_off + 56], module.elf[kd_off + 57]])
            } else {
                0
            };
            Kernel::from_meta(km, code_buf.gpu_addr, kcp)
        }).collect();

        Ok(LoadedModule { kernels, code_buf })
    }

    /// Load a .hsaco file from disk.
    pub fn load_module_file(&self, path: &str) -> Result<LoadedModule> {
        let data = std::fs::read(path)
            .map_err(|e| RedlineError { code: -1, message: format!("read {path}: {e}") })?;
        self.load_module(&data)
    }
}

impl Kernel {
    fn from_meta(km: &KernelMeta, code_buf_base: u64, kcp: u16) -> Self {
        let code_va = code_buf_base + km.code_offset;

        // Decode kernel_code_properties to determine user SGPR layout
        let mut count = 0u32;
        let mut kernarg_idx = None;

        if kcp & (1 << 0) != 0 { count += 4; } // private segment buffer
        if kcp & (1 << 1) != 0 { count += 2; } // dispatch ptr
        if kcp & (1 << 2) != 0 { count += 2; } // queue ptr
        if kcp & (1 << 3) != 0 {
            kernarg_idx = Some(count);
            count += 2; // kernarg segment ptr
        }
        if kcp & (1 << 4) != 0 { count += 2; } // dispatch id
        if kcp & (1 << 5) != 0 { count += 2; } // flat scratch init
        if kcp & (1 << 6) != 0 { count += 1; } // private segment size

        Kernel {
            name: km.name.clone(),
            code_va,
            pgm_rsrc1: km.pgm_rsrc1,
            pgm_rsrc2: km.pgm_rsrc2,
            kernarg_size: km.kernarg_size,
            group_segment_size: km.group_segment_size,
            user_sgpr_count: count,
            kernarg_sgpr_idx: kernarg_idx,
        }
    }

    /// Find a kernel by name in a loaded module.
    pub fn find<'a>(module: &'a LoadedModule, name: &str) -> Option<&'a Kernel> {
        module.kernels.iter().find(|k| k.name == name)
    }
}

impl CommandBuffer {
    pub fn new() -> Self {
        Self { dwords: Vec::with_capacity(512) }
    }

    /// Append a single dispatch to this command buffer.
    /// `kernarg_va`: GPU virtual address of the kernarg buffer for this dispatch.
    pub fn dispatch(&mut self, k: &Kernel, grid: [u32; 3], block: [u32; 3], kernarg_va: u64) {
        let d = &mut self.dwords;

        // COMPUTE_PGM_LO/HI
        d.push(pkt3(SET_SH_REG, 3));
        d.push(0x020C);
        d.push((k.code_va >> 8) as u32);
        d.push((k.code_va >> 40) as u32);

        // COMPUTE_PGM_RSRC1/RSRC2
        d.push(pkt3(SET_SH_REG, 3));
        d.push(0x0212);
        d.push(k.pgm_rsrc1);
        d.push(k.pgm_rsrc2);

        // COMPUTE_PGM_RSRC3 (GFX10 required)
        d.push(pkt3(SET_SH_REG, 2));
        d.push(0x0228);
        d.push(0);

        // COMPUTE_TMPRING_SIZE = 0 (no scratch)
        d.push(pkt3(SET_SH_REG, 2));
        d.push(0x0218);
        d.push(0);

        // COMPUTE_NUM_THREAD_X/Y/Z
        d.push(pkt3(SET_SH_REG, 4));
        d.push(0x0207);
        d.push(block[0]);
        d.push(block[1]);
        d.push(block[2]);

        // COMPUTE_RESOURCE_LIMITS = 0
        d.push(pkt3(SET_SH_REG, 2));
        d.push(0x0215);
        d.push(0);

        // USER_DATA — fill with zeros, place kernarg pointer at the right index
        if k.user_sgpr_count > 0 {
            d.push(pkt3(SET_SH_REG, 1 + k.user_sgpr_count));
            d.push(0x0240); // COMPUTE_USER_DATA_0
            for i in 0..k.user_sgpr_count {
                if Some(i) == k.kernarg_sgpr_idx {
                    d.push(kernarg_va as u32);
                } else if Some(i) == k.kernarg_sgpr_idx.map(|x| x + 1) {
                    d.push((kernarg_va >> 32) as u32);
                } else {
                    d.push(0);
                }
            }
        }

        // DISPATCH_DIRECT
        // CS_EN=1 | CS_W32_EN=1 (HIP on RDNA always wave32)
        let di = (1u32 << 0) | (1 << 15);
        d.push(pkt3(DISPATCH_DIRECT, 4));
        d.push(grid[0]);
        d.push(grid[1]);
        d.push(grid[2]);
        d.push(di);
    }

    /// Append a dispatch with explicit dynamic LDS (shared memory) size.
    /// `lds_bytes` is the dynamic shared memory in bytes (added to kernel's static LDS).
    pub fn dispatch_with_lds(&mut self, k: &Kernel, grid: [u32; 3], block: [u32; 3],
                              kernarg_va: u64, lds_bytes: u32) {
        let d = &mut self.dwords;

        // COMPUTE_PGM_LO/HI
        d.push(pkt3(SET_SH_REG, 3));
        d.push(0x020C);
        d.push((k.code_va >> 8) as u32);
        d.push((k.code_va >> 40) as u32);

        // COMPUTE_PGM_RSRC1
        d.push(pkt3(SET_SH_REG, 2));
        d.push(0x0212);
        d.push(k.pgm_rsrc1);

        // COMPUTE_PGM_RSRC2 with LDS_SIZE override
        // LDS_SIZE field is bits [20:14] — number of 512-byte blocks
        // Total LDS = kernel static + dynamic lds_bytes
        let total_lds = k.group_segment_size + lds_bytes;
        let lds_blocks = (total_lds + 511) / 512; // round up to 512-byte blocks
        let rsrc2_base = k.pgm_rsrc2 & !(0x7F << 14); // clear existing LDS_SIZE
        let rsrc2 = rsrc2_base | ((lds_blocks & 0x7F) << 14);

        d.push(pkt3(SET_SH_REG, 2));
        d.push(0x0213); // COMPUTE_PGM_RSRC2 offset (0x0212 + 1)
        d.push(rsrc2);

        // PGM_RSRC3
        d.push(pkt3(SET_SH_REG, 2));
        d.push(0x0228);
        d.push(0);

        // TMPRING_SIZE
        d.push(pkt3(SET_SH_REG, 2));
        d.push(0x0218);
        d.push(0);

        // NUM_THREAD
        d.push(pkt3(SET_SH_REG, 4));
        d.push(0x0207);
        d.push(block[0]);
        d.push(block[1]);
        d.push(block[2]);

        // RESOURCE_LIMITS
        d.push(pkt3(SET_SH_REG, 2));
        d.push(0x0215);
        d.push(0);

        // USER_DATA
        if k.user_sgpr_count > 0 {
            d.push(pkt3(SET_SH_REG, 1 + k.user_sgpr_count));
            d.push(0x0240);
            for i in 0..k.user_sgpr_count {
                if Some(i) == k.kernarg_sgpr_idx {
                    d.push(kernarg_va as u32);
                } else if Some(i) == k.kernarg_sgpr_idx.map(|x| x + 1) {
                    d.push((kernarg_va >> 32) as u32);
                } else {
                    d.push(0);
                }
            }
        }

        // DISPATCH_DIRECT
        let di = (1u32 << 0) | (1 << 15); // CS_EN | CS_W32_EN
        d.push(pkt3(DISPATCH_DIRECT, 4));
        d.push(grid[0]);
        d.push(grid[1]);
        d.push(grid[2]);
        d.push(di);
    }

    /// Insert a compute barrier: RELEASE_MEM (fence write + L2 flush) + WAIT_REG_MEM (poll).
    /// `fence_va`: GPU VA of a dword-aligned fence location.
    /// `fence_value`: value RELEASE_MEM writes; WAIT_REG_MEM polls for it.
    pub fn barrier(&mut self, fence_va: u64, fence_value: u32) {
        let d = &mut self.dwords;

        // RELEASE_MEM: wait for prior dispatches + flush caches + write fence value.
        // Encoding verified in C (test_release_mem.c + test_wrm.c).
        // CRITICAL: header uses PACKET3() WITHOUT SHADER_TYPE bit.
        d.push(0xC006_4900);  // PACKET3(RELEASE_MEM, 6), NO shader_type
        d.push(0x0660_3514);  // event + GCR flags (from nvd.h, matches kernel driver)
        d.push(0x2000_0000);  // DATA_SEL(1) = 32-bit write
        d.push(fence_va as u32);
        d.push((fence_va >> 32) as u32);
        d.push(fence_value);
        d.push(0);
        d.push(0);

        // WAIT_REG_MEM: poll fence_va until value == fence_value.
        d.push(0xC005_3C00);  // PACKET3(WAIT_REG_MEM, 5), NO shader_type
        d.push(0x0000_0013);  // MEM_SPACE=1(memory) | FUNCTION=3(equal)
        d.push(fence_va as u32);
        d.push((fence_va >> 32) as u32);
        d.push(fence_value);
        d.push(0xFFFF_FFFF);  // mask
        d.push(4);             // poll interval
    }

    /// Number of PM4 dwords in this command buffer.
    pub fn len_dwords(&self) -> u32 {
        self.dwords.len() as u32
    }

    /// Serialize to bytes for upload.
    pub fn as_bytes(&self) -> Vec<u8> {
        self.dwords.iter().flat_map(|d| d.to_le_bytes()).collect()
    }
}

/// Convenience wrapper: dispatch queue with persistent IB + kernarg buffers.
pub struct DispatchQueue {
    pub queue: ComputeQueue,
    ib_buf: GpuBuffer,
    ka_buf: GpuBuffer,
}

const IB_SIZE: u64 = 64 * 1024; // 64KB IB buffer (plenty for hundreds of dispatches)
const KA_SIZE: u64 = 64 * 1024; // 64KB kernarg buffer

impl DispatchQueue {
    pub fn new(dev: &Device) -> Result<Self> {
        let queue = ComputeQueue::new(dev)?;
        let ib_buf = dev.alloc_vram(IB_SIZE)?;
        let ka_buf = dev.alloc_vram(KA_SIZE)?;
        Ok(Self { queue, ib_buf, ka_buf })
    }

    /// Single dispatch: upload args, build PM4, submit, wait.
    ///
    /// `args` should contain only the explicit kernel arguments.
    /// Hidden arguments (block counts, group sizes) are populated automatically.
    pub fn dispatch(
        &self,
        dev: &Device,
        kernel: &Kernel,
        grid: [u32; 3],
        block: [u32; 3],
        args: &[u8],
        extra_bos: &[&GpuBuffer],
    ) -> Result<()> {
        // Build kernarg: explicit args + hidden args
        let ka_size = std::cmp::max(kernel.kernarg_size as usize, args.len());
        let mut ka_data = vec![0u8; std::cmp::max(ka_size, 256)];
        ka_data[..args.len()].copy_from_slice(args);

        // Populate hidden args if kernarg_size > explicit args
        // Layout (code object V5): block_count_x/y/z (u32×3), group_size_x/y/z (u16×3),
        // remainder_x/y/z (u16×3), then global_offset_x/y/z (u64×3), grid_dims (u16)
        let hidden_off = (args.len() + 7) & !7; // 8-byte aligned after explicit args
        if ka_size > hidden_off {
            let mut w = |off: usize, val: &[u8]| {
                if off + val.len() <= ka_data.len() {
                    ka_data[off..off + val.len()].copy_from_slice(val);
                }
            };
            // block_count_x/y/z (u32 each)
            w(hidden_off, &grid[0].to_le_bytes());
            w(hidden_off + 4, &grid[1].to_le_bytes());
            w(hidden_off + 8, &grid[2].to_le_bytes());
            // group_size_x/y/z (u16 each)
            w(hidden_off + 12, &(block[0] as u16).to_le_bytes());
            w(hidden_off + 14, &(block[1] as u16).to_le_bytes());
            w(hidden_off + 16, &(block[2] as u16).to_le_bytes());
            // remainder = 0 for uniform work groups (already zeroed)
            // grid_dims
            let ndims = if grid[2] > 1 { 3u16 } else if grid[1] > 1 { 2 } else { 1 };
            w(hidden_off + 64, &ndims.to_le_bytes());
        }
        dev.upload(&self.ka_buf, &ka_data)?;

        // Build PM4
        let mut cb = CommandBuffer::new();
        cb.dispatch(kernel, grid, block, self.ka_buf.gpu_addr);

        // Upload IB
        let ib_bytes = cb.as_bytes();
        dev.upload(&self.ib_buf, &ib_bytes)?;

        // Collect BO list
        let mut bos: Vec<&GpuBuffer> = vec![&self.ib_buf, &self.ka_buf];
        bos.extend_from_slice(extra_bos);

        self.queue.submit_and_wait(dev, &self.ib_buf, cb.len_dwords(), &bos)
    }

    /// Submit a pre-built command buffer. Caller manages kernarg separately.
    pub fn submit(
        &self,
        dev: &Device,
        cb: &CommandBuffer,
        bos: &[&GpuBuffer],
    ) -> Result<()> {
        let ib_bytes = cb.as_bytes();
        if ib_bytes.len() as u64 > IB_SIZE {
            return Err(RedlineError { code: -1, message: "command buffer exceeds IB size".into() });
        }
        dev.upload(&self.ib_buf, &ib_bytes)?;

        let mut all_bos: Vec<&GpuBuffer> = vec![&self.ib_buf];
        all_bos.extend_from_slice(bos);

        self.queue.submit_and_wait(dev, &self.ib_buf, cb.len_dwords(), &all_bos)
    }

    /// Get a reference to the persistent kernarg buffer.
    pub fn kernarg_buf(&self) -> &GpuBuffer {
        &self.ka_buf
    }

    pub fn destroy(self, dev: &Device) {
        self.queue.destroy(dev);
    }
}

/// Optimized dispatch path: persistent CPU-mapped IB and kernarg buffers.
/// Eliminates per-dispatch heap allocations, memcpy overhead, and Vec creation.
/// The ioctl overhead (~50µs) remains but everything around it is minimized.
pub struct FastDispatch {
    pub queue: ComputeQueue,
    ib_buf: GpuBuffer,
    ka_buf: GpuBuffer,
    ib_ptr: *mut u8,    // persistent CPU mapping of IB
    ka_ptr: *mut u8,    // persistent CPU mapping of kernarg
    bo_list_handle: crate::drm::AmdgpuBoListHandle, // persistent BO list
}

unsafe impl Send for FastDispatch {}

impl FastDispatch {
    /// Create a fast dispatch context. `extra_bos` are buffers referenced by dispatches
    /// (kernel code, data buffers) — included in a persistent BO list.
    pub fn new(dev: &Device, extra_bos: &[&GpuBuffer]) -> Result<Self> {
        let queue = ComputeQueue::new(dev)?;
        let ib_buf = dev.alloc_vram(IB_SIZE)?;
        let ka_buf = dev.alloc_vram(KA_SIZE)?;

        // Persistent CPU mappings
        let mut ib_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let ret = unsafe { (dev.drm.bo_cpu_map)(ib_buf.handle, &mut ib_ptr) };
        if ret != 0 {
            return Err(RedlineError { code: ret, message: "map IB failed".into() });
        }
        let mut ka_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let ret = unsafe { (dev.drm.bo_cpu_map)(ka_buf.handle, &mut ka_ptr) };
        if ret != 0 {
            return Err(RedlineError { code: ret, message: "map KA failed".into() });
        }

        // Persistent BO list including IB + KA + all extra buffers
        let mut bo_handles: Vec<crate::drm::AmdgpuBoHandle> = vec![ib_buf.handle, ka_buf.handle];
        bo_handles.extend(extra_bos.iter().map(|b| b.handle));
        let prios = vec![0u8; bo_handles.len()];
        let mut bo_list: crate::drm::AmdgpuBoListHandle = std::ptr::null_mut();
        let ret = unsafe {
            (dev.drm.bo_list_create)(dev.handle, bo_handles.len() as u32,
                bo_handles.as_ptr(), prios.as_ptr(), &mut bo_list)
        };
        if ret != 0 {
            return Err(RedlineError { code: ret, message: "bo_list_create failed".into() });
        }

        Ok(Self {
            queue, ib_buf, ka_buf,
            ib_ptr: ib_ptr as *mut u8,
            ka_ptr: ka_ptr as *mut u8,
            bo_list_handle: bo_list,
        })
    }

    /// Fast dispatch: write PM4 + kernarg to persistent mappings, submit via ioctl.
    /// No heap allocations in the hot path. Only the ioctl syscall remains.
    pub fn dispatch(
        &self,
        dev: &Device,
        kernel: &Kernel,
        grid: [u32; 3],
        block: [u32; 3],
        args: &[u8],
    ) -> Result<()> {
        // Write kernarg directly to persistent mapping
        let ka_size = std::cmp::max(kernel.kernarg_size as usize, args.len());
        unsafe {
            std::ptr::copy_nonoverlapping(args.as_ptr(), self.ka_ptr, args.len());
            // Zero remaining
            if ka_size > args.len() {
                std::ptr::write_bytes(self.ka_ptr.add(args.len()), 0, ka_size - args.len());
            }
        }
        // Hidden args
        let hidden_off = (args.len() + 7) & !7;
        if ka_size > hidden_off {
            unsafe {
                let p = self.ka_ptr;
                (p.add(hidden_off) as *mut u32).write(grid[0]);
                (p.add(hidden_off + 4) as *mut u32).write(grid[1]);
                (p.add(hidden_off + 8) as *mut u32).write(grid[2]);
                (p.add(hidden_off + 12) as *mut u16).write(block[0] as u16);
                (p.add(hidden_off + 14) as *mut u16).write(block[1] as u16);
                (p.add(hidden_off + 16) as *mut u16).write(block[2] as u16);
            }
        }

        // Build PM4 directly into persistent IB mapping
        let mut cb = CommandBuffer::new();
        cb.dispatch(kernel, grid, block, self.ka_buf.gpu_addr);
        let dwords = &cb.dwords;
        unsafe {
            std::ptr::copy_nonoverlapping(
                dwords.as_ptr() as *const u8,
                self.ib_ptr,
                dwords.len() * 4,
            );
        }

        // Submit with persistent BO list — only the ioctl remains
        self.queue.submit_with_bo_list(dev, &self.ib_buf, cb.len_dwords(), self.bo_list_handle)
    }

    /// Get a reference to the persistent kernarg buffer.
    pub fn ka_buf_ref(&self) -> &GpuBuffer {
        &self.ka_buf
    }

    /// Submit a pre-built command buffer using the persistent BO list.
    pub fn submit_cmdbuf(&self, dev: &Device, cb: &CommandBuffer) -> Result<()> {
        let ib_bytes = cb.as_bytes();
        unsafe {
            std::ptr::copy_nonoverlapping(ib_bytes.as_ptr(), self.ib_ptr, ib_bytes.len());
        }
        self.queue.submit_with_bo_list(dev, &self.ib_buf, cb.len_dwords(), self.bo_list_handle)
    }

    pub fn destroy(self, dev: &Device) {
        unsafe {
            (dev.drm.bo_cpu_unmap)(self.ib_buf.handle);
            (dev.drm.bo_cpu_unmap)(self.ka_buf.handle);
            (dev.drm.bo_list_destroy)(self.bo_list_handle);
        }
        self.queue.destroy(dev);
    }
}

/// Build a kernarg byte buffer from typed arguments.
/// Each arg is written at the correct alignment.
pub struct KernargBuilder {
    data: Vec<u8>,
}

impl KernargBuilder {
    pub fn new(capacity: usize) -> Self {
        Self { data: vec![0u8; capacity] }
    }

    pub fn write_u32(&mut self, offset: usize, val: u32) -> &mut Self {
        self.data[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
        self
    }

    pub fn write_u64(&mut self, offset: usize, val: u64) -> &mut Self {
        self.data[offset..offset + 8].copy_from_slice(&val.to_le_bytes());
        self
    }

    pub fn write_f32(&mut self, offset: usize, val: f32) -> &mut Self {
        self.data[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
        self
    }

    pub fn write_ptr(&mut self, offset: usize, gpu_addr: u64) -> &mut Self {
        self.write_u64(offset, gpu_addr)
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }
}
