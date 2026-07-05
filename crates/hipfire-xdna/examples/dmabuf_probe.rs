//! Probe whether the RAW amdxdna path (no XRT) can import an amdgpu GTT dma-buf and
//! share memory zero-copy with the GPU — the NPU⇄GPU data primitive the heterogeneous
//! prefill / spec-draft pipeline needs. Prior interop was proven only via XRT on halo
//! (aie2p); this checks the raw path on THIS box (gfx1103 / XDNA1).
//!
//! Flow: amdgpu (renderD128) GEM_CREATE a GTT BO → mmap → write a marker → PRIME export
//! → amdxdna (accel0) imports the dma-buf as a SHARE BO → mmap → read back (does the NPU
//! side see the GPU's marker?) → write a second marker on the NPU side → read on the GPU
//! side (bidirectional coherence?).
//!
//! Run: cargo run -p hipfire-xdna --example dmabuf_probe

#[cfg(target_os = "linux")]
const fn iowr(nr: u32, size: u32) -> libc::c_ulong {
    // _IOWR('d', nr, size): dir=READ|WRITE=3, type='d'=0x64.
    ((3u32 << 30) | (size << 16) | (0x64u32 << 8) | nr) as libc::c_ulong
}

fn main() {
    #[cfg(target_os = "linux")]
    {
        use hipfire_xdna::XdnaDevice;
        use std::io::Error;

        const GEM_CREATE: libc::c_ulong = iowr(0x40, 32);
        const GEM_MMAP: libc::c_ulong = iowr(0x41, 8);
        const PRIME_HANDLE_TO_FD: libc::c_ulong = iowr(0x2d, 12);
        const DOMAIN_GTT: u64 = 0x2;
        const CPU_ACCESS_REQUIRED: u64 = 1;
        const SZ: usize = 4096;

        // amdgpu union drm_amdgpu_gem_create (in is 32B; out.handle overlays offset 0).
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

        let die = |stage: &str| -> ! {
            eprintln!("FAIL at {stage}: {}", Error::last_os_error());
            std::process::exit(1);
        };

        // 1) open amdgpu render node
        let gpu_fd = unsafe { libc::open(c"/dev/dri/renderD128".as_ptr(), libc::O_RDWR) };
        if gpu_fd < 0 {
            die("open renderD128");
        }

        // 2) GEM_CREATE a GTT BO, CPU-accessible
        let mut gc = GemCreate {
            bo_size: SZ as u64,
            alignment: 4096,
            domains: DOMAIN_GTT,
            domain_flags: CPU_ACCESS_REQUIRED,
        };
        if unsafe { libc::ioctl(gpu_fd, GEM_CREATE, &mut gc as *mut _ as *mut libc::c_void) } != 0 {
            die("amdgpu GEM_CREATE(GTT)");
        }
        let gpu_handle = unsafe { *(&gc as *const _ as *const u32) }; // out.handle at offset 0
        println!("[1] amdgpu GTT BO created: handle={gpu_handle}");

        // 3) GEM_MMAP → offset, then mmap it
        let mut mm: u64 = gpu_handle as u64; // in.handle at offset 0
        if unsafe { libc::ioctl(gpu_fd, GEM_MMAP, &mut mm as *mut _ as *mut libc::c_void) } != 0 {
            die("amdgpu GEM_MMAP");
        }
        let gpu_off = mm; // out.addr_ptr
        let gpu_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                SZ,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                gpu_fd,
                gpu_off as libc::off_t,
            )
        };
        if gpu_ptr == libc::MAP_FAILED {
            die("mmap amdgpu BO");
        }
        let gpu_i32 = gpu_ptr as *mut i32;
        const MARK_GPU: i32 = 0x11223344;
        unsafe { *gpu_i32 = MARK_GPU };
        println!("[2] GPU wrote marker 0x{MARK_GPU:08x} into the GTT BO");

        // 4) export dma-buf fd
        let mut ph = PrimeHandle {
            handle: gpu_handle,
            flags: (libc::O_RDWR | libc::O_CLOEXEC) as u32,
            fd: -1,
        };
        if unsafe {
            libc::ioctl(
                gpu_fd,
                PRIME_HANDLE_TO_FD,
                &mut ph as *mut _ as *mut libc::c_void,
            )
        } != 0
        {
            die("PRIME_HANDLE_TO_FD");
        }
        let dmabuf_fd = ph.fd;
        println!("[3] exported dma-buf fd={dmabuf_fd}");

        // 5) import into the NPU via the RAW amdxdna path
        let dev = XdnaDevice::open_default().unwrap_or_else(|e| {
            eprintln!("FAIL open accel0: {e:?}");
            std::process::exit(1);
        });
        let mut npu_bo = match dev.import_dmabuf(dmabuf_fd, SZ, true) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("FAIL amdxdna import_dmabuf: {e:?}");
                eprintln!(
                    "  Go/no-go: the amdgpu half (GTT alloc + dma-buf export) works on the raw"
                );
                eprintln!(
                    "  path. EINVAL here means the INSTALLED amdxdna lacks the import UAPI: the"
                );
                eprintln!(
                    "  mainline 6.17 header has no amdxdna_drm_va_tbl / AMDXDNA_BO_SHARE. The"
                );
                eprintln!(
                    "  out-of-tree ~/xdna-driver implements it (create_ubuf_object -> dma_buf_get"
                );
                eprintln!("  -> prime_import); build+load that module to enable zero-copy import.");
                std::process::exit(2);
            }
        };
        unsafe { libc::close(dmabuf_fd) };
        println!("[4] amdxdna imported the dma-buf as a SHARE BO (import OK)");

        // 6) NPU side reads back → does it see the GPU's marker? (zero-copy share)
        let seen = unsafe { *(npu_bo.as_slice().as_ptr() as *const i32) };
        let share_read = seen == MARK_GPU;
        println!(
            "[5] NPU reads 0x{seen:08x} (GPU wrote 0x{MARK_GPU:08x}) -> shared-read {}",
            if share_read { "PASS" } else { "FAIL" }
        );

        // 7) NPU writes a marker → does the GPU side see it? (bidirectional coherence)
        const MARK_NPU: i32 = 0x55667788;
        unsafe { *(npu_bo.as_mut_slice().as_mut_ptr() as *mut i32) = MARK_NPU };
        let gpu_sees = unsafe { *gpu_i32 };
        let share_write = gpu_sees == MARK_NPU;
        println!(
            "[6] GPU reads 0x{gpu_sees:08x} (NPU wrote 0x{MARK_NPU:08x}) -> shared-write {}",
            if share_write { "PASS" } else { "FAIL" }
        );

        println!(
            "\nRESULT: raw-amdxdna dma-buf import={}, zero-copy share r/w={}/{}",
            "OK",
            if share_read { "yes" } else { "no" },
            if share_write { "yes" } else { "no" }
        );
    }
    #[cfg(not(target_os = "linux"))]
    eprintln!("amdxdna is Linux-only");
}
