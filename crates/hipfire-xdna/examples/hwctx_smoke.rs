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
        let _heap = match dev.alloc_buffer(HEAP, AMDXDNA_BO_DEV_HEAP) {
            Ok(b) => {
                println!("dev_heap: {} MB BO handle={}", HEAP >> 20, b.handle());
                b
            }
            Err(e) => {
                eprintln!("alloc dev_heap: {e}");
                std::process::exit(2);
            }
        };

        // The resource solver rejects an all-zero QoS; give it realistic caps.
        let qos = QosInfo {
            gops: 1000,
            fps: 60,
            dma_bandwidth: 0,
            latency: 0,
            frame_exec_time: 0,
            priority: 0x180,
        };
        let mut created_any = false;
        // aie2p Strix Halo: 4 core rows/col, 8 cols. Probe 1..8 columns worth.
        for &num_tiles in &[4u32, 8, 16, 20, 32] {
            match dev.create_hwctx(num_tiles, 0, 0x800, &qos) {
                Ok((handle, syncobj)) => {
                    created_any = true;
                    println!("  num_tiles={num_tiles:>2} -> hwctx handle={handle} syncobj={syncobj}  (created OK)");
                    if let Err(e) = dev.destroy_hwctx(handle) {
                        eprintln!("    destroy_hwctx({handle}): {e}");
                    }
                }
                Err(e) => println!("  num_tiles={num_tiles:>2} -> {e}"),
            }
        }
        std::process::exit(if created_any { 0 } else { 5 });
    }
    #[cfg(not(target_os = "linux"))]
    {
        eprintln!("amdxdna is Linux-only");
        std::process::exit(1);
    }
}
