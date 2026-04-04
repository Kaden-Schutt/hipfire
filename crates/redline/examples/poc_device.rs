//! Redline PoC: open GPU, query info, alloc VRAM, upload/download data.
//! This validates the basic KMD path works WITHOUT HIP.
//!
//! Usage: cargo run -p redline --example poc_device

fn main() {
    eprintln!("=== redline PoC: device + memory ===\n");

    // Step 1: Open GPU
    let dev = match redline::device::Device::open(None) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("FAILED: {e}");
            std::process::exit(1);
        }
    };

    eprintln!("\nGPU info:");
    eprintln!("  arch:       {}", dev.info.gfx_arch);
    eprintln!("  CUs:        {}", dev.info.num_cu);
    eprintln!("  VGPRs:      {}/SIMD", dev.info.num_vgprs);
    eprintln!("  wave size:  {}", dev.info.wave_front_size);
    eprintln!("  VRAM total: {:.1} GB", dev.info.vram_total_bytes as f64 / 1e9);
    eprintln!("  VRAM used:  {:.1} GB", dev.info.vram_used_bytes as f64 / 1e9);
    eprintln!("  VRAM free:  {:.1} GB", (dev.info.vram_total_bytes - dev.info.vram_used_bytes) as f64 / 1e9);

    // Step 2: Alloc VRAM
    let size = 4096u64; // 4KB test buffer
    eprintln!("\nAllocating {} bytes VRAM...", size);
    let buf = match dev.alloc_vram(size) {
        Ok(b) => {
            eprintln!("  OK — gpu_addr: 0x{:016x}", b.gpu_addr);
            b
        }
        Err(e) => {
            eprintln!("  FAILED: {e}");
            std::process::exit(1);
        }
    };

    // Step 3: Upload test data
    let test_data: Vec<u8> = (0..size as usize).map(|i| (i & 0xFF) as u8).collect();
    eprintln!("Uploading {} bytes...", test_data.len());
    if let Err(e) = dev.upload(&buf, &test_data) {
        eprintln!("  FAILED: {e}");
        std::process::exit(1);
    }
    eprintln!("  OK");

    // Step 4: Download and verify
    let mut readback = vec![0u8; size as usize];
    eprintln!("Downloading {} bytes...", readback.len());
    if let Err(e) = dev.download(&buf, &mut readback) {
        eprintln!("  FAILED: {e}");
        std::process::exit(1);
    }

    if readback == test_data {
        eprintln!("  OK — data matches!");
    } else {
        let mismatches = readback.iter().zip(test_data.iter())
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .count();
        eprintln!("  MISMATCH — {} bytes differ", mismatches);
        std::process::exit(1);
    }

    // Step 5: Free
    eprintln!("Freeing buffer...");
    if let Err(e) = dev.free_buffer(buf) {
        eprintln!("  FAILED: {e}");
        std::process::exit(1);
    }
    eprintln!("  OK");

    eprintln!("\n=== redline PoC PASSED ===");
    eprintln!("Device open, VRAM alloc, upload, download, free — all working.");
    eprintln!("Next: compute queue + kernel dispatch (PM4 command buffers).");
}
