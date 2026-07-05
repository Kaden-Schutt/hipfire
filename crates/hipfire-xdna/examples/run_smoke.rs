//! W3d milestone: parse an mlir-aie xclbin, create a hwctx, and load its PDI via
//! CONFIG_HWCTX(CONFIG_CU) on real hardware. Next step (W4) adds the instruction
//! BO + ERT command + EXEC_CMD.
//!
//! Run: `cargo run -p hipfire-xdna --example run_smoke -- <path/to/final.xclbin>`

fn main() {
    #[cfg(target_os = "linux")]
    {
        use hipfire_xdna::submit::QosInfo;
        use hipfire_xdna::xclbin::Axlf;
        use hipfire_xdna::XdnaDevice;

        let path = std::env::args().nth(1).unwrap_or_else(|| {
            eprintln!("usage: run_smoke <final.xclbin>");
            std::process::exit(2);
        });
        let bytes = std::fs::read(&path).expect("read xclbin");
        let axlf = Axlf::parse(&bytes).expect("parse xclbin");
        let part = axlf.aie_partition().expect("AIE_PARTITION");
        println!(
            "xclbin: {} columns, PDI {} bytes",
            part.column_width,
            part.pdi.len()
        );

        let dev = XdnaDevice::open_default().expect("open accel");
        // dev_heap first (DEV BOs come from it; create_hwctx needs it).
        let mut heap = dev.alloc_dev_heap(64 * 1024 * 1024).expect("dev_heap");

        // num_tiles = columns * core row_count (aie2p = 4).
        let num_tiles = part.column_width as u32 * 4;
        let (hwctx, syncobj) = dev
            .create_hwctx(num_tiles, 0, 0x800, &QosInfo::default())
            .expect("create_hwctx");
        println!("hwctx={hwctx} syncobj={syncobj} (num_tiles={num_tiles})");

        // Load the PDI into a DEV BO and configure the CU.
        let (pdi_bo, pdi_addr) = dev.alloc_dev_bo(&mut heap, part.pdi).expect("pdi dev_bo");
        println!("pdi_bo={pdi_bo} dev_addr={pdi_addr:#x}");
        match dev.config_hwctx_cu(hwctx, pdi_bo) {
            Ok(()) => println!("CONFIG_HWCTX(CONFIG_CU) ok — PDI loaded"),
            Err(e) => {
                eprintln!("config_hwctx_cu: {e}");
                let _ = dev.destroy_hwctx(hwctx);
                std::process::exit(3);
            }
        }

        let _ = dev.destroy_hwctx(hwctx);
        println!("done");
    }
    #[cfg(not(target_os = "linux"))]
    eprintln!("amdxdna is Linux-only");
}
