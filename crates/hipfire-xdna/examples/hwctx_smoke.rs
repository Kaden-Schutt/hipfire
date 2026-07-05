//! W3c smoke test: create + destroy an amdxdna hardware context on the NPU.
//! A hwctx reserves AIE columns but runs no program (no PDI/EXEC), so this is a
//! safe on-hardware check of the CREATE_HWCTX / DESTROY_HWCTX path. Probes a few
//! `num_tiles` values to discover the core row_count (num_col = num_tiles/row_count).
//!
//! Run: `cargo run -p hipfire-xdna --example hwctx_smoke`

fn main() {
    #[cfg(target_os = "linux")]
    {
        use hipfire_xdna::submit::{QosInfo, AMDXDNA_BO_DEV_HEAP};
        use hipfire_xdna::XdnaDevice;

        let dev = match XdnaDevice::open_default() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("open accel: {e}");
                std::process::exit(1);
            }
        };
        println!("opened {}", dev.path());

        // aie2_hwctx_init requires the client to have a device heap first, else
        // CREATE_HWCTX returns -ENOENT ("dev heap object not exist").
        const HEAP: usize = 64 * 1024 * 1024;
        // DEV_HEAP: mmap it AND fault in the pages (fill), else the firmware
        // host-buffer map fails ("Map host buffer failed" on lazy MAP_SHARED).
        let _ = AMDXDNA_BO_DEV_HEAP;
        let _heap = match dev.alloc_dev_heap(HEAP) {
            Ok(b) => {
                println!("dev_heap: {} MB handle={}", HEAP >> 20, b.handle());
                b
            }
            Err(e) => {
                eprintln!("alloc dev_heap: {e}");
                std::process::exit(2);
            }
        };

        // Mirror the captured pyxrt order: two SHMEM BOs, then CREATE_HWCTX with
        // exactly the args XRT uses (num_tiles=32=8col*4row, mem_size=0,
        // max_opc=0x800, all-zero QoS).
        use hipfire_xdna::submit::AMDXDNA_BO_SHMEM;
        let _b2 = dev.alloc_buffer(256 * 1024, AMDXDNA_BO_SHMEM);
        let _b3 = dev.alloc_buffer(256, AMDXDNA_BO_SHMEM);
        // pyxrt issues a GET_INFO query right before CREATE_HWCTX; replicate it in
        // case it lazily initializes the resource solver / AIE metadata.
        let _ = dev.resource_info();
        let _ = dev.clocks();
        let qos = QosInfo::default();
        match dev.create_hwctx(32, 0, 0x800, &qos) {
            Ok((handle, syncobj)) => {
                println!("CREATE_HWCTX ok: handle={handle} syncobj={syncobj}");
                if let Err(e) = dev.destroy_hwctx(handle) {
                    eprintln!("destroy_hwctx: {e}");
                }
                std::process::exit(0);
            }
            Err(e) => {
                eprintln!("CREATE_HWCTX(32,0,0x800,zeros): {e}");
                std::process::exit(5);
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        eprintln!("amdxdna is Linux-only");
        std::process::exit(1);
    }
}
