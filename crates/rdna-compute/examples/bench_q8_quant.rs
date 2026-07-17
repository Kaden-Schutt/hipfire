use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    println!("arch={}", gpu.arch);
    let cases = [
        ("gate_up B=256", 256usize, 2048usize),
        ("down B=256*8", 2048, 512),
        ("gate_up B=1024", 1024, 2048),
        ("down B=1024*8", 8192, 512),
    ];
    for (label, rows, k) in cases {
        let n = rows * k;
        let x: Vec<f32> = (0..n).map(|i| ((i % 11) as f32 - 5.0) / 5.0).collect();
        let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        let x_gpu = gpu.hip.malloc(bytes.len()).unwrap();
        gpu.hip.memcpy_htod(&x_gpu, &bytes).unwrap();
        let xt = GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(x_gpu.as_ptr(), bytes.len()) },
            shape: vec![rows, k],
            dtype: DType::F32,
        };
        for _ in 0..3 {
            let _ = gpu.ensure_q8_1_mmq_x(&xt, rows, k).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let t0 = Instant::now();
        let trials = 20;
        for _ in 0..trials {
            let _ = gpu.ensure_q8_1_mmq_x(&xt, rows, k).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let us = t0.elapsed().as_secs_f64() / trials as f64 * 1e6;
        println!("{label:>16} rows={rows:<5} k={k:<5}  {us:>8.1} µs/call");
        std::mem::forget(xt);
    }
}
