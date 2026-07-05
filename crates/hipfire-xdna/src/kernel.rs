//! W5 — reusable NPU kernel dispatch: the `run_smoke` flow behind a Rust API the
//! runtime can call. An [`NpuKernel`] is one compiled mlir-aie kernel (one shape);
//! it opens the device, allocates the heap, creates a hwctx, loads the tile
//! program (PDI) and the instruction stream once, and then dispatches repeatedly.
//!
//! xclbins are built offline by the mlir-aie toolchain (Python is not in the
//! inference hot path); the runtime loads the cached bytes and dispatches through
//! this HIP-direct-adjacent amdxdna path.
//!
//! Linux-only (amdxdna DRM ioctls).
#![cfg(target_os = "linux")]

use crate::submit::{self, QosInfo, AMDXDNA_BO_CMD, AMDXDNA_BO_SHMEM};
use crate::xclbin::Axlf;
use crate::{DeviceBuffer, XdnaDevice, XdnaError};

/// A single compiled NPU kernel with its hwctx and loaded program. Bind argument
/// buffers with [`Self::alloc_arg`], fill inputs, then [`Self::dispatch`].
pub struct NpuKernel {
    dev: XdnaDevice,
    // Backing heap for the PDI + instruction DEV BOs; must outlive the hwctx.
    _heap: DeviceBuffer,
    hwctx: u32,
    syncobj: u32,
    instr_bo: u32,
    instr_addr: u64,
    instr_size: usize,
}

impl NpuKernel {
    /// Heap size backing the PDI + instruction streams. The AIE2 dev-mem window is
    /// 64 MiB; PDIs/instructions are a few KiB, so one 64 MiB heap is ample.
    const HEAP_BYTES: usize = 64 * 1024 * 1024;

    /// Load a compiled kernel: `xclbin` bytes (for the PDI) and its `insts`
    /// instruction stream. Sets up the hwctx and loads the program on hardware.
    pub fn load(xclbin: &[u8], insts: &[u8]) -> Result<Self, XdnaError> {
        let dev = XdnaDevice::open_default()?;
        let mut heap = dev.alloc_dev_heap(Self::HEAP_BYTES)?;

        let axlf = Axlf::parse(xclbin)?;
        let part = axlf.aie_partition().ok_or(XdnaError::NoAiePartition)?;
        let num_tiles = part.column_width as u32 * 4; // aie2p: 4 core rows/column

        let (hwctx, syncobj) = dev.create_hwctx(num_tiles, 0, 0x800, &QosInfo::default())?;
        let (pdi_bo, _) = dev.alloc_dev_bo(&mut heap, part.pdi)?;
        if let Err(e) = dev.config_hwctx_cu(hwctx, pdi_bo) {
            let _ = dev.destroy_hwctx(hwctx);
            return Err(e);
        }
        let (instr_bo, instr_addr) = match dev.alloc_dev_bo(&mut heap, insts) {
            Ok(v) => v,
            Err(e) => {
                let _ = dev.destroy_hwctx(hwctx);
                return Err(e);
            }
        };

        Ok(Self {
            dev,
            _heap: heap,
            hwctx,
            syncobj,
            instr_bo,
            instr_addr,
            instr_size: insts.len(),
        })
    }

    /// Allocate a SHMEM argument buffer (host-visible, NPU-accessible via PASID).
    /// The caller fills inputs and reads outputs directly through its slices.
    pub fn alloc_arg(&self, size: usize) -> Result<DeviceBuffer, XdnaError> {
        self.dev.alloc_buffer(size, AMDXDNA_BO_SHMEM)
    }

    /// Run the kernel over `args` in kernel-signature order (e.g. A, W, C). Flushes
    /// the argument buffers to the device, submits the command, and blocks until it
    /// completes; on return the output buffers are readable directly (SHMEM is
    /// coherent once the timeline signals).
    pub fn dispatch(&self, args: &[&DeviceBuffer]) -> Result<(), XdnaError> {
        for a in args {
            self.dev
                .sync_bo(a.handle(), submit::SYNC_DIRECT_TO_DEVICE, a.len())?;
        }

        let addrs: Vec<u64> = args.iter().map(|b| b.host_addr()).collect();
        let packet = submit::dpu_cmd_packet(self.instr_addr, self.instr_size, &addrs);
        let mut cmd = self.dev.alloc_buffer(4096, AMDXDNA_BO_CMD)?;
        cmd.as_mut_slice()[..packet.len()].copy_from_slice(&packet);

        let mut handles: Vec<u32> = args.iter().map(|b| b.handle()).collect();
        handles.push(self.instr_bo); // instruction BO is an EXEC arg (residency)

        let seq = self.dev.exec_cmd(self.hwctx, cmd.handle(), &handles)?;
        self.dev.syncobj_wait(self.syncobj, seq)
    }
}

impl Drop for NpuKernel {
    fn drop(&mut self) {
        let _ = self.dev.destroy_hwctx(self.hwctx);
    }
}
