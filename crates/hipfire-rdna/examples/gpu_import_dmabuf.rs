//! Validate Gpu::import_dmabuf: allocate a shared GTT dma-buf, import it back as a NATIVE
//! GPU tensor via HIP external-memory, run a GPU op (hipMemset) on the imported device
//! pointer, and confirm the write lands on the shared pages (read via the CPU mapping). This
//! proves a GPU compute kernel can operate on an imported dma-buf directly — the GPU-side of
//! the three-engine native-import triangle (GPU ↔ NPU ↔ CPU over one buffer).
//!
//! Run: hipfire lock acquire; cargo run -p hipfire-rdna --example gpu_import_dmabuf; hipfire lock release

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_gpu_types::DType;
    use hipfire_rdna::Gpu;

    const SZ: usize = 1 << 16; // 64 KiB
    let mut gpu = Gpu::init().expect("Gpu::init");

    // GPU allocates a shared GTT dma-buf, then imports it back as a native GPU tensor.
    let mut shared = gpu.alloc_shared_gtt(SZ)?;
    // Seed via the CPU mapping so a stale value can't masquerade as success.
    for b in shared.as_mut_slice().iter_mut() {
        *b = 0x00;
    }
    let imp = gpu.import_dmabuf(shared.dmabuf_fd(), SZ, &[SZ], DType::Raw)?;

    // GPU op on the imported native device pointer.
    let t = imp.view();
    gpu.hip.memset(&t.buf, 0xAB, SZ)?;
    gpu.device_synchronize()?;

    // The CPU mapping of the SAME dma-buf must now read the GPU's write.
    let cpu = shared.as_slice();
    let ok = cpu.iter().all(|&b| b == 0xAB);
    let first = cpu[0];
    println!(
        "GPU memset(0xAB) on imported dma-buf tensor; CPU reads byte0=0x{first:02x}, all-0xAB={ok}"
    );
    if !ok {
        return Err("import ok but GPU write not visible on shared pages".into());
    }
    println!("Gpu::import_dmabuf CORRECT — GPU computed on an imported dma-buf, coherent with CPU (native, no hipHostRegister)");
    Ok(())
}
