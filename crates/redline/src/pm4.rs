// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! PM4 command buffer builder for GFX10 (RDNA1) compute dispatch.
//!
//! PM4 (Packet Manager 4) is AMD's GPU command protocol. We build a
//! sequence of PM4 packets that configure and launch a compute shader.
//!
//! Packet sequence for compute dispatch:
//! 1. PKT3_SET_SH_REG: COMPUTE_PGM_LO/HI — shader program address
//! 2. PKT3_SET_SH_REG: COMPUTE_PGM_RSRC1/2 — register allocation
//! 3. PKT3_SET_SH_REG: COMPUTE_NUM_THREAD_X/Y/Z — workgroup size
//! 4. PKT3_SET_SH_REG: COMPUTE_USER_DATA_0+ — kernel arguments pointer
//! 5. PKT3_DISPATCH_DIRECT — grid dimensions + launch
//!
//! Register offsets from AMD's GFX10 spec (COMPUTE_* SH registers):
//! Base for SET_SH_REG = 0x2C00 (SH register space)
//! COMPUTE_PGM_LO     = 0x2E0C (offset 0x20C from base)
//! COMPUTE_PGM_HI     = 0x2E10
//! COMPUTE_PGM_RSRC1  = 0x2E14 (but we set these via the kernel descriptor)
//! COMPUTE_PGM_RSRC2  = 0x2E18
//! COMPUTE_NUM_THREAD_X = 0x2E04
//! COMPUTE_NUM_THREAD_Y = 0x2E08
//! COMPUTE_NUM_THREAD_Z = 0x2E0C — wait, that overlaps PGM_LO...
//!
//! Actually for GFX10, the AQL (Architected Queuing Language) dispatch
//! is simpler: the hardware reads the kernel descriptor directly.
//! We don't need to manually set SH registers — we submit an AQL
//! dispatch packet to an HSA-style compute queue.

/// PM4 packet opcodes
pub const PKT3_SET_SH_REG: u32 = 0x76;
pub const PKT3_DISPATCH_DIRECT: u32 = 0x15;
pub const PKT3_ACQUIRE_MEM: u32 = 0x58;
pub const PKT3_RELEASE_MEM: u32 = 0x49;

/// AQL dispatch packet (64 bytes) — the modern dispatch mechanism.
/// Hardware reads this directly from the queue ring buffer.
/// No PM4 needed for basic dispatch on GFX10+ with AQL queues.
#[repr(C, align(64))]
#[derive(Clone)]
pub struct AqlDispatchPacket {
    /// [0:1] Header: packet type (2=dispatch) + barrier bit + acquire/release fence
    pub header: u16,
    /// [2:3] Setup: number of dimensions (1, 2, or 3)
    pub setup: u16,
    /// [4:5] Workgroup size X
    pub workgroup_size_x: u16,
    /// [6:7] Workgroup size Y
    pub workgroup_size_y: u16,
    /// [8:9] Workgroup size Z
    pub workgroup_size_z: u16,
    /// [10:11] Reserved
    pub _reserved0: u16,
    /// [12:15] Grid size X (total threads, not groups)
    pub grid_size_x: u32,
    /// [16:19] Grid size Y
    pub grid_size_y: u32,
    /// [20:23] Grid size Z
    pub grid_size_z: u32,
    /// [24:27] Private segment size per work-item
    pub private_segment_size: u32,
    /// [28:31] Group segment size (LDS) in bytes
    pub group_segment_size: u32,
    /// [32:39] Kernel object address (GPU VA of kernel descriptor)
    pub kernel_object: u64,
    /// [40:47] Kernarg address (GPU VA of kernel arguments buffer)
    pub kernarg_address: u64,
    /// [48:55] Reserved
    pub _reserved1: u64,
    /// [56:63] Completion signal (GPU VA, 0 = no signal)
    pub completion_signal: u64,
}

impl AqlDispatchPacket {
    /// Build a dispatch packet for a kernel.
    pub fn new(
        kernel_descriptor_addr: u64,
        kernarg_addr: u64,
        grid: [u32; 3],
        block: [u32; 3],
        lds_bytes: u32,
        private_bytes: u32,
    ) -> Self {
        // Header: packet_type=2 (dispatch), barrier=1, scacquire_fence=2, screlease_fence=2
        let header: u16 = (2 << 0)      // HSA_PACKET_TYPE_KERNEL_DISPATCH
                        | (1 << 8)       // barrier bit
                        | (2 << 9)       // acquire fence scope (agent)
                        | (2 << 11);     // release fence scope (agent)

        let ndims = if grid[2] > 1 { 3 } else if grid[1] > 1 { 2 } else { 1 };

        Self {
            header,
            setup: ndims,
            workgroup_size_x: block[0] as u16,
            workgroup_size_y: block[1] as u16,
            workgroup_size_z: block[2] as u16,
            _reserved0: 0,
            grid_size_x: grid[0] * block[0],
            grid_size_y: grid[1] * block[1],
            grid_size_z: grid[2] * block[2],
            private_segment_size: private_bytes,
            group_segment_size: lds_bytes,
            kernel_object: kernel_descriptor_addr,
            kernarg_address: kernarg_addr,
            _reserved1: 0,
            completion_signal: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self as *const _ as *const u8, 64)
        }
    }
}

/// PM4 command buffer builder (for non-AQL submission paths).
pub struct Pm4Builder {
    pub dwords: Vec<u32>,
}

impl Pm4Builder {
    pub fn new() -> Self {
        Self { dwords: Vec::with_capacity(256) }
    }

    /// Emit PKT3 header
    fn pkt3(&mut self, opcode: u32, count: u32) {
        self.dwords.push((3 << 30) | (opcode << 8) | (count - 1));
    }

    /// SET_SH_REG: write to shader register space
    pub fn set_sh_reg(&mut self, reg_offset: u32, value: u32) {
        self.pkt3(PKT3_SET_SH_REG, 2);
        self.dwords.push(reg_offset);
        self.dwords.push(value);
    }

    /// DISPATCH_DIRECT: launch compute workgroups
    pub fn dispatch_direct(&mut self, groups_x: u32, groups_y: u32, groups_z: u32) {
        self.pkt3(PKT3_DISPATCH_DIRECT, 4);
        self.dwords.push(groups_x);
        self.dwords.push(groups_y);
        self.dwords.push(groups_z);
        self.dwords.push(1); // DISPATCH_INITIATOR: compute shader enable
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self.dwords.as_ptr() as *const u8, self.dwords.len() * 4)
        }
    }
}
