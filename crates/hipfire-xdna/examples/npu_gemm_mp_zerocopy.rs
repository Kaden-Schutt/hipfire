//! Zero-copy C: NpuGemmMp writes its GEMM output directly into a GPU-shared dma-buf via a
//! real kernel dispatch (no SHMEM host copy), and the GPU-side mapping reads it back. This
//! proves the NPU DMA engine — not just the CPU — can target an imported amdgpu GTT buffer,
//! which removes the ~37%-of-e2e C-readback host copy and is the handoff a heterogeneous
//! NPU‖GPU pipeline needs.
//!
//! Flow: amdgpu GEM_CREATE(GTT, c_buf_bytes) → PRIME export → NpuGemmMp::attach_output_dmabuf
//! → run_into_shared (NPU kernel writes C) → read the GPU mapping, de-block, compare to CPU.
//!
//! Build: R6_KERNEL_SRC=<r6>/r6_gemm_ts.cc R6_GEN=r6_gen_mp.py R6_OUT_TAG=r6mp <r6>/r6_cache.sh MT 4 KCHUNK COLS NB
//! Run:   cargo run -p hipfire-xdna --example npu_gemm_mp_zerocopy -- <mp-xclbin-dir>

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

        let dir = std::env::args()
            .nth(1)
            .expect("usage: npu_gemm_mp_zerocopy <mp-xclbin-dir>");
        let mut g = NpuGemmMp::load_cached(&dir).expect("load_cached");
        let (k, n, rows_per) = (g.k(), g.n(), g.rows_per_dispatch());
        let cbytes = g.c_buf_bytes();
        // Block layout: C is COLS·NB blocks of (block_rows)×(NT·MN) i32, block (ci,j) at
        // (ci*nb+j)*cw_i32; core ci owns global rows [ci*block_rows,+), slab j cols
        // [j*NT*MN,+). rows_per = COLS*block_rows and n = NB*NT*MN, so:
        let nb = n / (NT * MN);
        assert_eq!(cbytes / 4, rows_per * nb * NT * MN, "shape self-check");
        // COLS comes from the cache dir name (`..._c{COLS}_nb{NB}`); the rest is derived.
        let cols: usize = dir
            .rsplit("_c")
            .next()
            .and_then(|s| s.split(['_', 'x']).next())
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);
        let block_rows = rows_per / cols; // = MT*MR
        let cw_i32 = block_rows * NT * MN; // i32 per block

        // amdgpu GTT buffer sized to C, CPU-accessible, exported as a dma-buf.
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
        let gpu_fd = unsafe { libc::open(c"/dev/dri/renderD128".as_ptr(), libc::O_RDWR) };
        assert!(gpu_fd >= 0, "open renderD128");
        let mut gc = GemCreate {
            bo_size: cbytes as u64,
            alignment: 4096,
            domains: 0x2,
            domain_flags: 1,
        };
        assert_eq!(
            unsafe { libc::ioctl(gpu_fd, GEM_CREATE, &mut gc as *mut _ as *mut libc::c_void) },
            0,
            "GEM_CREATE"
        );
        let gpu_handle = unsafe { *(&gc as *const _ as *const u32) };
        let mut mm: u64 = gpu_handle as u64;
        assert_eq!(
            unsafe { libc::ioctl(gpu_fd, GEM_MMAP, &mut mm as *mut _ as *mut libc::c_void) },
            0,
            "GEM_MMAP"
        );
        let gpu_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                cbytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                gpu_fd,
                mm as libc::off_t,
            )
        };
        assert_ne!(gpu_ptr, libc::MAP_FAILED, "mmap GTT");
        let mut ph = PrimeHandle {
            handle: gpu_handle,
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
            "PRIME export"
        );

        g.attach_output_dmabuf(ph.fd, cbytes)
            .expect("attach_output_dmabuf");
        unsafe { libc::close(ph.fd) };

        // Weights + one M-block of A.
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

        // NPU kernel writes C straight into the GPU's GTT pages — no host copy.
        g.run_into_shared(k, n, &av).expect("run_into_shared");

        // Read the GPU-side mapping, de-block, compare to CPU. If this matches, the NPU DMA
        // wrote correct C into the GPU-shared buffer (zero-copy handoff works).
        let gpu_c: &[i32] =
            unsafe { std::slice::from_raw_parts(gpu_ptr as *const i32, cbytes / 4) };
        let mut mism = 0usize;
        for &mm_row in &[0usize, m / 2, m - 1] {
            let ci = mm_row / block_rows;
            let lr = mm_row % block_rows;
            for nn in 0..n {
                let j = nn / (NT * MN);
                let col = nn % (NT * MN);
                let got = gpu_c[(ci * nb + j) * cw_i32 + lr * (NT * MN) + col];
                let acc: i32 = (0..k)
                    .map(|kk| av[mm_row * k + kk] as i32 * wv[kk * n + nn] as i32)
                    .sum();
                if got != acc {
                    mism += 1;
                }
            }
        }
        println!(
            "M-block M={m} K={k} N={n} (COLS={cols}); GPU-side reads {mism} mismatches vs CPU"
        );
        if mism != 0 {
            eprintln!("ZERO-COPY C WRONG (NPU did not write correct C into the shared dma-buf)");
            std::process::exit(4);
        }
        println!(
            "ZERO-COPY C CORRECT — NPU kernel wrote C into the GPU-shared dma-buf, no host copy"
        );
    }
    #[cfg(not(target_os = "linux"))]
    eprintln!("amdxdna is Linux-only");
}
