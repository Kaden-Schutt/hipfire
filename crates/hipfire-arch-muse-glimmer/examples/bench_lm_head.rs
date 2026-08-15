use hipfire_runtime::hfq::HfqFile;
use hipfire_arch_muse_glimmer::config::GlimmerConfig;
use hipfire_arch_muse_glimmer::glimmer::GlimmerWeights;
use rdna_compute::{DType, Gpu};
use std::time::Instant;
fn main() {
    let model = std::env::args().nth(1).unwrap();
    let mut gpu = Gpu::init().unwrap();
    let hfq = HfqFile::open(std::path::Path::new(&model)).unwrap();
    let cfg = GlimmerConfig::from_hfq(&hfq).unwrap();
    let weights = GlimmerWeights::load(&hfq, &cfg, &mut gpu).unwrap();
    let dim = cfg.dim;
    let vocab = cfg.vocab_size;
    for &batch in &[15usize, 16] {
        let hidden = gpu.alloc_tensor(&[batch*dim], DType::F32).unwrap();
        // fill hidden with some data
        let mut host = vec![0.1f32; batch*dim];
        for i in 0..host.len() { host[i] = (i as f32 % 10.0) * 0.01; }
        let bytes = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len()*4) };
        gpu.hip.memcpy_htod(&hidden.buf, bytes).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let logits = gpu.alloc_tensor(&[batch*vocab], DType::F32).unwrap();
        let t0 = Instant::now();
        gpu.gemm_q8_0_batched_chunked(&weights.lm_head.buf, &hidden, &logits, vocab, dim, batch).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let dt = t0.elapsed();
        println!("batch {} gemm_q8_0_batched_chunked: {:.2}ms", batch, dt.as_secs_f64()*1000.0);
        let t0 = Instant::now();
        gpu.gemm_q8_0_wmma(&weights.lm_head.buf, &hidden, &logits, vocab, dim, batch).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let dt = t0.elapsed();
        println!("batch {} gemm_q8_0_wmma: {:.2}ms", batch, dt.as_secs_f64()*1000.0);
        gpu.free_tensor(hidden).ok();
        gpu.free_tensor(logits).ok();
    }
}
