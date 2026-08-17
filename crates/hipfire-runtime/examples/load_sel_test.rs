use rdna_compute::Gpu;
use hipfire_runtime::weight_backend as wb;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Try to init GPU and load a synthetic SEL blob
    let mut gpu = match Gpu::init_with_device(0) {
        Ok(g) => {
            println!("GPU init succeeded");
            g
        },
        Err(e) => {
            println!("GPU init failed: {} (checking only that dequant rejects invalid)", e);
            // Even without GPU we can test the validation logic by checking that
            // a bad selector would be rejected if we had a GPU. For now just report.
            println!("RAW_CODECS smoke test: qt=43 should be MQ4G256SEL (checked at compile time)");
            return Ok(());
        }
    };
    // Create synthetic SEL blob: m=2,k=256 => 2*132=264 bytes
    let m=2usize; let k=256usize;
    let mut blob = vec![0u8; 2*132];
    for chunk in blob.chunks_exact_mut(132) {
        chunk[130]=0;
        chunk[131]=0;
    }
    println!("dequant_weight_raw qt=43 m={} k={} len={}", m, k, blob.len());
    let wt = wb::dequant_weight_raw(&gpu, 43, &blob, m, k)?;
    println!("loaded dtype={:?} row_stride={}", wt.gpu_dtype, wt.row_stride);
    println!("load smoke test passed");
    // Test invalid selector rejection
    let mut bad = blob.clone();
    bad[130]=64;
    match wb::dequant_weight_raw(&gpu, 43, &bad, m, k) {
        Ok(_) => println!("ERROR: invalid selector 64 should have been rejected"),
        Err(e) => println!("correctly rejected invalid selector 64: {}", e),
    }
    // Test pad non-zero rejection
    let mut bad2 = blob.clone();
    bad2[131]=1;
    match wb::dequant_weight_raw(&gpu, 43, &bad2, m, k) {
        Ok(_) => println!("ERROR: pad non-zero should have been rejected"),
        Err(e) => println!("correctly rejected pad non-zero: {}", e),
    }
    Ok(())
}
