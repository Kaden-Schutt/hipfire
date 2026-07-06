//! Fully zero-copy NpuGemmMp: BOTH A (input) and C (output) live in GPU-shared dma-bufs, so
//! a dispatch does NO host copies at all. A producer (here the CPU, standing in for a GPU
//! op on this UMA APU) writes one M-block of A into the input dma-buf; the NPU reads it and
//! writes C into the output dma-buf; a consumer reads C from the GPU mapping. Proves the NPU
//! DMA engine both READS and WRITES imported amdgpu buffers correctly.
//!
//! Build: R6_KERNEL_SRC=<r6>/r6_gemm_ts.cc R6_GEN=r6_gen_mp.py R6_OUT_TAG=r6mp <r6>/r6_cache.sh MT 4 KCHUNK COLS NB
//! Run:   cargo run -p hipfire-xdna --example npu_gemm_mp_zerocopy_io -- <mp-xclbin-dir>

#[cfg(target_os = "linux")]
const fn iowr(nr: u32, size: u32) -> libc::c_ulong {
    ((3u32 << 30) | (size << 16) | (0x64u32 << 8) | nr) as libc::c_ulong
}

fn main() {
    #[cfg(target_os = "linux")]
    {
        use hipfire_xdna::NpuGemmMp;
        const MN: usize = 16;
        const NT: usize = 4;

        // amdgpu: GEM_CREATE(GTT, CPU-accessible) -> mmap -> PRIME export. Returns (host ptr,
        // dma-buf fd).
        const GEM_CREATE: libc::c_ulong = iowr(0x40, 32);
        const GEM_MMAP: libc::c_ulong = iowr(0x41, 8);
        const PRIME_HANDLE_TO_FD: libc::c_ulong = iowr(0x2d, 12);
        #[repr(C)]
        #[derive(Default)]
        struct GemCreate {
            bo_size: u64,
            alignment: u64,
            domains: u64,
            domain_flags: u64,
        }
        #[repr(C)]
        struct PrimeHandle {
            handle: u32,
            flags: u32,
            fd: i32,
        }
        let gtt_export = |gpu_fd: i32, bytes: usize| -> (*mut libc::c_void, i32) {
            let mut gc = GemCreate {
                bo_size: bytes as u64,
                alignment: 4096,
                domains: 0x2,
                domain_flags: 1,
            };
            assert_eq!(
                unsafe { libc::ioctl(gpu_fd, GEM_CREATE, &mut gc as *mut _ as *mut libc::c_void) },
                0,
                "GEM_CREATE"
            );
            let handle = unsafe { *(&gc as *const _ as *const u32) };
            let mut mm: u64 = handle as u64;
            assert_eq!(
                unsafe { libc::ioctl(gpu_fd, GEM_MMAP, &mut mm as *mut _ as *mut libc::c_void) },
                0,
                "GEM_MMAP"
            );
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    bytes,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    gpu_fd,
                    mm as libc::off_t,
                )
            };
            assert_ne!(ptr, libc::MAP_FAILED, "mmap GTT");
            let mut ph = PrimeHandle {
                handle,
                flags: (libc::O_RDWR | libc::O_CLOEXEC) as u32,
                fd: -1,
            };
            assert_eq!(
                unsafe {
                    libc::ioctl(
                        gpu_fd,
                        PRIME_HANDLE_TO_FD,
                        &mut ph as *mut _ as *mut libc::c_void,
                    )
                },
                0,
                "PRIME"
            );
            (ptr, ph.fd)
        };

        let dir = std::env::args()
            .nth(1)
            .expect("usage: npu_gemm_mp_zerocopy_io <mp-xclbin-dir>");
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
        let block_rows = rows_per / cols; // MT*MR
        let cw_i32 = block_rows * NT * MN;

        let gpu_fd = unsafe { libc::open(c"/dev/dri/renderD128".as_ptr(), libc::O_RDWR) };
        assert!(gpu_fd >= 0, "open renderD128");
        let (a_ptr, a_fd) = gtt_export(gpu_fd, abytes);
        let (c_ptr, c_fd) = gtt_export(gpu_fd, cbytes);
        g.attach_input_dmabuf(a_fd, abytes).expect("attach_input");
        g.attach_output_dmabuf(c_fd, cbytes).expect("attach_output");
        unsafe {
            libc::close(a_fd);
            libc::close(c_fd);
        }

        // Weights.
        let rnd = |i: usize| -> i8 {
            let s = (i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(0x9e37_79b9);
            (((s >> 13) & 0xf) as i32 - 8) as i8
        };
        let wv: Vec<i8> = (0..k * n).map(|i| rnd(7_777_777 + i)).collect();
        g.load_weights(&g.prepack_weights(k, n, &wv));

        // Producer writes A row-major [rows_per][K] into the INPUT GTT buffer (a_buf layout
        // == row-major A, no reshuffle). Done via the GPU mapping to model a GPU producer.
        let m = rows_per;
        let av: Vec<i8> = (0..m * k).map(rnd).collect();
        let a_gpu = unsafe { std::slice::from_raw_parts_mut(a_ptr as *mut i8, abytes) };
        a_gpu.copy_from_slice(&av);

        // Fully zero-copy dispatch: no host copies.
        g.run_shared(k, n).expect("run_shared");

        // Consumer reads C from the OUTPUT GTT buffer, de-blocks, compares to CPU.
        let c_gpu: &[i32] = unsafe { std::slice::from_raw_parts(c_ptr as *const i32, cbytes / 4) };
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
        println!("M-block M={m} K={k} N={n} (COLS={cols}); A+C in GPU dma-bufs; {mism} mismatches vs CPU");
        if mism != 0 {
            eprintln!("FULL ZERO-COPY WRONG");
            std::process::exit(4);
        }
        println!("FULL ZERO-COPY CORRECT — NPU read A and wrote C via GPU-shared dma-bufs, zero host copies");
    }
    #[cfg(not(target_os = "linux"))]
    eprintln!("amdxdna is Linux-only");
}
