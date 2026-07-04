//! W2 smoke test: allocate an amdxdna buffer object on the NPU accel device,
//! mmap it, write a pattern, read it back, and sync to device. Exercises the
//! CREATE_BO / GET_BO_INFO / mmap / SYNC_BO path on real hardware.
//!
//! Run: `cargo run -p hipfire-xdna --example bo_smoke`

fn main() {
    #[cfg(target_os = "linux")]
    {
        use hipfire_xdna::submit::{AMDXDNA_BO_SHMEM, SYNC_DIRECT_TO_DEVICE};
        use hipfire_xdna::XdnaDevice;

        let dev = match XdnaDevice::open_default() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("open accel device: {e} (is the amdxdna NPU present?)");
                std::process::exit(1);
            }
        };
        println!("opened {}", dev.path());

        const N: usize = 4096;
        let mut buf = match dev.alloc_buffer(N, AMDXDNA_BO_SHMEM) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("alloc_buffer({N}, SHMEM): {e}");
                std::process::exit(2);
            }
        };
        println!(
            "BO handle={} xdna_addr={:#x}",
            buf.handle(),
            buf.xdna_addr()
        );

        // write a pattern through the mmap and read it back
        for (i, b) in buf.as_mut_slice().iter_mut().enumerate() {
            *b = (i * 7 + 1) as u8;
        }
        if let Err(e) = dev.sync_bo(buf.handle(), SYNC_DIRECT_TO_DEVICE, N) {
            eprintln!("sync_bo TO_DEVICE: {e}");
            std::process::exit(3);
        }
        let s = buf.as_slice();
        let ok = s.iter().enumerate().all(|(i, &b)| b == (i * 7 + 1) as u8);
        println!(
            "mmap read-back {} (first 8 bytes: {:?})",
            if ok { "OK" } else { "MISMATCH" },
            &s[..8]
        );
        std::process::exit(if ok { 0 } else { 4 });
    }
    #[cfg(not(target_os = "linux"))]
    {
        eprintln!("amdxdna is Linux-only");
        std::process::exit(1);
    }
}
