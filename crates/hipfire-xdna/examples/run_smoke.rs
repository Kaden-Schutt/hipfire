//! W4 milestone: run a compiled mlir-aie kernel end-to-end on the NPU through the
//! hipfire amdxdna path — the first hipfire-driven NPU dispatch. Parses the xclbin
//! for its PDI, loads it (CONFIG_CU), stages the instruction stream + A/W/C data
//! BOs, builds the ERT DPU command, submits it (EXEC_CMD), waits on the hwctx
//! timeline, and reads back C.
//!
//! Targets the R2a single-core W4A8 GEMM (NACC=8, INNER=64, N_BTILES=4):
//!   A = 512 i8, W = 512 B packed int4, C = 512 i32 (2048 B).
//!
//! Run: `cargo run -p hipfire-xdna --example run_smoke -- <npu-cache-dir>`
//! where the dir holds `final.xclbin` + `insts.bin`.

fn main() {
    #[cfg(target_os = "linux")]
    {
        use hipfire_xdna::submit::{self, QosInfo, AMDXDNA_BO_CMD, AMDXDNA_BO_SHMEM};
        use hipfire_xdna::xclbin::Axlf;
        use hipfire_xdna::XdnaDevice;

        let dir = std::env::args().nth(1).unwrap_or_else(|| {
            eprintln!("usage: run_smoke <npu-cache-dir with final.xclbin + insts.bin>");
            std::process::exit(2);
        });
        let xclbin = std::fs::read(format!("{dir}/final.xclbin")).expect("read final.xclbin");
        let insts = std::fs::read(format!("{dir}/insts.bin")).expect("read insts.bin");
        let axlf = Axlf::parse(&xclbin).expect("parse xclbin");
        let part = axlf.aie_partition().expect("AIE_PARTITION");
        println!(
            "xclbin: {} columns, PDI {} B, insts {} B",
            part.column_width,
            part.pdi.len(),
            insts.len()
        );

        let dev = XdnaDevice::open_default().expect("open accel");
        let mut heap = dev.alloc_dev_heap(64 * 1024 * 1024).expect("dev_heap");

        // 1. hwctx + load the tile program (PDI).
        let num_tiles = part.column_width as u32 * 4;
        let (hwctx, syncobj) = dev
            .create_hwctx(num_tiles, 0, 0x800, &QosInfo::default())
            .expect("create_hwctx");
        let (pdi_bo, _) = dev.alloc_dev_bo(&mut heap, part.pdi).expect("pdi bo");
        dev.config_hwctx_cu(hwctx, pdi_bo).expect("config_cu");
        println!("hwctx={hwctx} PDI loaded");

        // 2. instruction stream (DEV BO, addressed by xdna_addr).
        let (instr_bo, instr_addr) = dev.alloc_dev_bo(&mut heap, &insts).expect("instr bo");

        // 3. data BOs (SHMEM, addressed by host VA). R2a: A/W/C.
        let mut a_bo = dev.alloc_buffer(512, AMDXDNA_BO_SHMEM).expect("A");
        let mut w_bo = dev.alloc_buffer(512, AMDXDNA_BO_SHMEM).expect("W");
        let mut c_bo = dev.alloc_buffer(2048, AMDXDNA_BO_SHMEM).expect("C");
        a_bo.as_mut_slice().fill(1); // int8 activations = 1
        w_bo.as_mut_slice().fill(0x11); // two int4 = 1 per byte
        c_bo.as_mut_slice().fill(0);
        for b in [&a_bo, &w_bo, &c_bo] {
            dev.sync_bo(b.handle(), submit::SYNC_DIRECT_TO_DEVICE, b.len())
                .expect("sync in");
        }

        // 4. ERT DPU command BO.
        let packet = submit::dpu_cmd_packet(
            instr_addr,
            insts.len(),
            &[a_bo.host_addr(), w_bo.host_addr(), c_bo.host_addr()],
        );
        let mut cmd_bo = dev.alloc_buffer(4096, AMDXDNA_BO_CMD).expect("cmd bo");
        cmd_bo.as_mut_slice()[..packet.len()].copy_from_slice(&packet);

        // 5. submit + wait.
        let arg_handles = [a_bo.handle(), w_bo.handle(), c_bo.handle(), instr_bo];
        let seq = dev
            .exec_cmd(hwctx, cmd_bo.handle(), &arg_handles)
            .expect("exec_cmd");
        println!("EXEC_CMD submitted seq={seq}, waiting…");
        dev.syncobj_wait(syncobj, seq).expect("syncobj_wait");

        // 6. read C back. SHMEM BOs are coherent host memory (pinned + PASID), so
        // once the timeline signals the NPU's writes are visible directly — the
        // FROM_DEVICE sync direction is reserved for debug BOs and EINVALs here.
        let c: &[i32] = unsafe {
            std::slice::from_raw_parts(c_bo.as_slice().as_ptr() as *const i32, c_bo.len() / 4)
        };
        let nonzero = c.iter().filter(|&&v| v != 0).count();
        println!(
            "C[0..8] = {:?}  ({} of {} lanes nonzero)",
            &c[..8.min(c.len())],
            nonzero,
            c.len()
        );
        if nonzero == 0 {
            eprintln!("C is all-zero — kernel did not execute");
            let _ = dev.destroy_hwctx(hwctx);
            std::process::exit(4);
        }
        println!("NPU dispatch OK — first hipfire-driven kernel execution");
        let _ = dev.destroy_hwctx(hwctx);
    }
    #[cfg(not(target_os = "linux"))]
    eprintln!("amdxdna is Linux-only");
}
