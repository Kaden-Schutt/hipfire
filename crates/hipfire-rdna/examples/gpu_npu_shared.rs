//! GPU-allocated shared dma-buf → NPU GEMM, zero-copy round-trip using the HOISTED API:
//! `Gpu::alloc_shared_gtt` (this crate) allocates the A and C buffers, the NPU imports them
//! via `NpuGemmMp::attach_*_dmabuf` (dev-dep on hipfire-xdna), and the CPU fills A / reads C
//! through the same mappings — all three engines on the same physical pages (UMA). Replaces
//! the raw amdgpu ioctls the hipfire-xdna probes carried.
//!
//! Build the NPU xclbin: R6_KERNEL_SRC=<r6>/r6_gemm_ts.cc R6_GEN=r6_gen_mp.py R6_OUT_TAG=r6mp <r6>/r6_cache.sh MT 4 KCHUNK COLS NB
//! Run: hipfire lock acquire; cargo run -p hipfire-rdna --example gpu_npu_shared -- <mp-xclbin-dir>; hipfire lock release

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_rdna::Gpu;
    use hipfire_xdna::NpuGemmMp;
    const MN: usize = 16;
    const NT: usize = 4;

    let dir = std::env::args()
        .nth(1)
        .expect("usage: gpu_npu_shared <mp-xclbin-dir>");
    let gpu = Gpu::init().expect("Gpu::init");
    let mut g = NpuGemmMp::load_cached(&dir).expect("load_cached");
    let (k, n, rows_per) = (g.k(), g.n(), g.rows_per_dispatch());
    let (abytes, cbytes) = (g.a_buf_bytes(), g.c_buf_bytes());
    let nb = n / (NT * MN);
    let cols: usize = dir
        .rsplit("_c")
        .next()
        .and_then(|s| s.split(['_', 'x']).next())
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let block_rows = rows_per / cols;
    let cw_i32 = block_rows * NT * MN;

    // GPU allocates the shared A and C dma-bufs (the hoisted API).
    let mut a_shared = gpu.alloc_shared_gtt(abytes)?;
    let c_shared = gpu.alloc_shared_gtt(cbytes)?;
    g.attach_input_dmabuf(a_shared.dmabuf_fd(), abytes)?;
    g.attach_output_dmabuf(c_shared.dmabuf_fd(), cbytes)?;

    // Weights + one M-block of A written into the shared input buffer (CPU producer proxy).
    let rnd = |i: usize| -> i8 {
        let s = (i as u32)
            .wrapping_mul(2654435761)
            .wrapping_add(0x9e37_79b9);
        (((s >> 13) & 0xf) as i32 - 8) as i8
    };
    let wv: Vec<i8> = (0..k * n).map(|i| rnd(7_777_777 + i)).collect();
    g.load_weights(&g.prepack_weights(k, n, &wv));
    let m = rows_per;
    let av: Vec<i8> = (0..m * k).map(rnd).collect();
    for (dst, &v) in a_shared.as_mut_slice().iter_mut().zip(av.iter()) {
        *dst = v as u8;
    }

    // Fully zero-copy dispatch, then read C from the GPU-allocated shared buffer.
    g.run_shared(k, n)?;
    let c_gpu: &[i32] = unsafe {
        std::slice::from_raw_parts(c_shared.as_slice().as_ptr() as *const i32, cbytes / 4)
    };
    let mut mism = 0usize;
    for &row in &[0usize, m / 2, m - 1] {
        let (ci, lr) = (row / block_rows, row % block_rows);
        for nn in 0..n {
            let (j, col) = (nn / (NT * MN), nn % (NT * MN));
            let got = c_gpu[(ci * nb + j) * cw_i32 + lr * (NT * MN) + col];
            let acc: i32 = (0..k)
                .map(|kk| av[row * k + kk] as i32 * wv[kk * n + nn] as i32)
                .sum();
            if got != acc {
                mism += 1;
            }
        }
    }
    println!("M={m} K={k} N={n} (COLS={cols}); GPU-allocated shared A+C; {mism} mismatches vs CPU");
    if mism != 0 {
        return Err("gpu_npu_shared WRONG".into());
    }
    println!(
        "GPU↔NPU SHARED-DMABUF CORRECT — Gpu::alloc_shared_gtt fed the NPU GEMM, zero host copies"
    );
    Ok(())
}
