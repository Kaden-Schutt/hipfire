// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Compute queue — submit PM4 command buffers to the GPU.

use crate::device::{Device, GpuBuffer};
use crate::drm::*;
pub use crate::drm::AmdgpuBoListHandle;
use crate::{RedlineError, Result};

pub const AMDGPU_HW_IP_COMPUTE: u32 = 1;

/// A compute context for GPU command submission.
pub struct ComputeQueue {
    ctx: AmdgpuContext,
}

impl ComputeQueue {
    pub fn new(dev: &Device) -> Result<Self> {
        let mut ctx: AmdgpuContext = std::ptr::null_mut();
        let ret = unsafe { (dev.drm.cs_ctx_create2)(dev.handle, 0, &mut ctx) };
        if ret != 0 {
            return Err(RedlineError { code: ret, message: format!("cs_ctx_create2 failed: {ret}") });
        }
        eprintln!("[redline] Compute context created");
        Ok(Self { ctx })
    }

    /// Submit an indirect buffer (PM4 commands) and wait for completion.
    /// `ib_buf`: GPU buffer containing PM4 dwords
    /// `ib_size_dwords`: number of PM4 dwords to execute
    /// `bo_refs`: all GPU buffers the commands reference (for BO list)
    pub fn submit_and_wait(
        &self,
        dev: &Device,
        ib_buf: &GpuBuffer,
        ib_size_dwords: u32,
        bo_refs: &[&GpuBuffer],
    ) -> Result<()> {
        // Create BO list so the kernel knows which buffers are in use
        let bo_handles: Vec<AmdgpuBoHandle> = bo_refs.iter().map(|b| b.handle).collect();
        let prios: Vec<u8> = vec![0; bo_handles.len()];
        let mut bo_list: AmdgpuBoListHandle = std::ptr::null_mut();
        let ret = unsafe {
            (dev.drm.bo_list_create)(
                dev.handle,
                bo_handles.len() as u32,
                bo_handles.as_ptr(),
                prios.as_ptr(),
                &mut bo_list,
            )
        };
        if ret != 0 {
            return Err(RedlineError { code: ret, message: format!("bo_list_create failed: {ret}") });
        }

        // Build IB info
        let mut ib = CsIbInfo {
            flags: 0,
            ib_mc_address: ib_buf.gpu_addr,
            size: ib_size_dwords,
            _pad: 0,
        };

        // Build submission request — MUST match amdgpu_cs_request layout exactly
        let mut request = CsRequest {
            flags: 0,
            ip_type: AMDGPU_HW_IP_COMPUTE,
            ip_instance: 0,
            ring: 0,
            _pad0: 0,
            resources: bo_list,
            number_of_dependencies: 0,
            _pad1: 0,
            dependencies: std::ptr::null(),
            number_of_ibs: 1,
            _pad2: 0,
            ibs: &mut ib,
            seq_no: 0,
            fence_info: CsFenceInfo::default(),
        };

        // Submit
        let ret = unsafe { (dev.drm.cs_submit)(self.ctx, 0, &mut request, 1) };
        // Destroy BO list regardless of submit result
        unsafe { (dev.drm.bo_list_destroy)(bo_list); }
        if ret != 0 {
            return Err(RedlineError { code: ret, message: format!("cs_submit failed: {ret}") });
        }

        // Wait for completion
        let mut fence = CsFence {
            context: self.ctx,
            ip_type: AMDGPU_HW_IP_COMPUTE,
            ip_instance: 0,
            ring: 0,
            fence: request.seq_no,
        };
        let mut expired = 0u32;
        let timeout_ns = 10_000_000_000u64; // 10 seconds
        let ret = unsafe { (dev.drm.cs_query_fence_status)(&mut fence, timeout_ns, 0, &mut expired) };
        if ret != 0 {
            return Err(RedlineError { code: ret, message: format!("fence wait failed: {ret}") });
        }
        if expired == 0 {
            return Err(RedlineError { code: -1, message: "GPU timeout (10s)".into() });
        }

        Ok(())
    }

    /// Submit with a pre-created BO list (avoids bo_list_create/destroy per dispatch).
    pub fn submit_with_bo_list(
        &self,
        dev: &Device,
        ib_buf: &GpuBuffer,
        ib_size_dwords: u32,
        bo_list: AmdgpuBoListHandle,
    ) -> Result<()> {
        let mut ib = CsIbInfo {
            flags: 0,
            ib_mc_address: ib_buf.gpu_addr,
            size: ib_size_dwords,
            _pad: 0,
        };
        let mut request = CsRequest {
            flags: 0,
            ip_type: AMDGPU_HW_IP_COMPUTE,
            ip_instance: 0,
            ring: 0,
            _pad0: 0,
            resources: bo_list,
            number_of_dependencies: 0,
            _pad1: 0,
            dependencies: std::ptr::null(),
            number_of_ibs: 1,
            _pad2: 0,
            ibs: &mut ib,
            seq_no: 0,
            fence_info: CsFenceInfo::default(),
        };

        let ret = unsafe { (dev.drm.cs_submit)(self.ctx, 0, &mut request, 1) };
        if ret != 0 {
            return Err(RedlineError { code: ret, message: format!("cs_submit failed: {ret}") });
        }

        let mut fence = CsFence {
            context: self.ctx,
            ip_type: AMDGPU_HW_IP_COMPUTE,
            ip_instance: 0,
            ring: 0,
            fence: request.seq_no,
        };
        let mut expired = 0u32;
        let ret = unsafe { (dev.drm.cs_query_fence_status)(&mut fence, 10_000_000_000, 0, &mut expired) };
        if ret != 0 {
            return Err(RedlineError { code: ret, message: format!("fence wait failed: {ret}") });
        }
        if expired == 0 {
            return Err(RedlineError { code: -1, message: "GPU timeout (10s)".into() });
        }
        Ok(())
    }

    pub fn destroy(self, dev: &Device) {
        unsafe { (dev.drm.cs_ctx_free)(self.ctx); }
    }
}
