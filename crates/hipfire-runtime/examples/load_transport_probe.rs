//! Probe HFQ file-to-GPU load transport bandwidth.
//!
//! Usage:
//!   cargo run --release --features deltanet --example load_transport_probe -- \
//!     MODEL.hfq --transport pread
//!   cargo run --release --features deltanet --example load_transport_probe -- \
//!     MODEL.hfq --transport pinned
//!   cargo run --release --features deltanet --example load_transport_probe -- \
//!     MODEL.hfq --transport pinned --mode slab --bank-size-mib 256
//!   cargo run --release --features deltanet --example load_transport_probe -- \
//!     MODEL.hfq --transport direct --mode slab --bank-size-mib 256
//!   cargo run --release --features deltanet --example load_transport_probe -- \
//!     MODEL.hfq --mode slab --bank-size-mib 256 --profile-io --read-mode direct
//!   cargo run --release --features deltanet --example load_transport_probe -- \
//!     MODEL.hfq --mode slab --bank-size-mib 256 --prealloc-slabs --read-mode direct

use std::path::Path;
use std::time::Instant;

use hip_bridge::{DeviceBuffer, Stream};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::weight_pager::{
    DirectH2DTransport, PinnedH2DTransport, PreadH2DTransport, Transport,
};
use rdna_compute::{DType, Gpu, GpuTensor};

#[cfg(not(unix))]
use std::os::unix::fs::FileExt;
#[cfg(unix)]
use std::os::unix::fs::{FileExt, OpenOptionsExt};

const DIRECT_ALIGN: usize = 4096;

struct Bank {
    offset: usize,
    len: usize,
    tensor_indices: Vec<usize>,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .expect("usage: load_transport_probe MODEL.hfq [--transport pread|pinned] [--limit N]");
    let mut transport_name = String::from("pread");
    let mut mode = String::from("tensor");
    let mut bank_size_mib = usize::MAX / (1024 * 1024);
    let mut limit: Option<usize> = None;
    let mut drop_cache = false;
    let mut profile_io = false;
    let mut read_mode = String::from("cached");
    let mut read_only = false;
    let mut prealloc_slabs = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--transport" => {
                transport_name = args.next().expect("--transport needs pread or pinned");
            }
            "--mode" => {
                mode = args.next().expect("--mode needs tensor or slab");
            }
            "--bank-size-mib" => {
                bank_size_mib = args
                    .next()
                    .expect("--bank-size-mib needs N")
                    .parse()
                    .expect("bad --bank-size-mib");
            }
            "--limit" => {
                limit = Some(
                    args.next()
                        .expect("--limit needs N")
                        .parse()
                        .expect("bad --limit"),
                );
            }
            "--drop-cache" => {
                drop_cache = true;
            }
            "--profile-io" => {
                profile_io = true;
            }
            "--read-mode" => {
                read_mode = args.next().expect("--read-mode needs cached or direct");
            }
            "--read-only" => {
                read_only = true;
                profile_io = true;
            }
            "--prealloc-slabs" => {
                prealloc_slabs = true;
            }
            other => panic!("unknown argument: {other}"),
        }
    }

    if drop_cache {
        drop_file_cache(Path::new(&model));
    }
    let hfq = HfqFile::open(Path::new(&model)).expect("open HFQ");
    let mut gpu = Gpu::init().expect("init GPU");
    let mut transport: Box<dyn Transport> = match transport_name.as_str() {
        "pread" => {
            Box::new(PreadH2DTransport::open(Path::new(&model)).expect("open pread transport"))
        }
        "pinned" | "pinned-h2d" => {
            Box::new(PinnedH2DTransport::open(Path::new(&model)).expect("open pinned transport"))
        }
        "direct" | "direct-h2d" | "odirect" => {
            Box::new(DirectH2DTransport::open(Path::new(&model)).expect("open direct transport"))
        }
        other => panic!("unsupported transport: {other}"),
    };

    match mode.as_str() {
        "tensor" => run_tensor_mode(&hfq, transport.as_mut(), &mut gpu, limit, &transport_name),
        "slab" => run_slab_mode(
            &hfq,
            transport.as_mut(),
            &mut gpu,
            limit,
            &transport_name,
            bank_size_mib.saturating_mul(1024 * 1024),
            profile_io,
            &read_mode,
            read_only,
            prealloc_slabs,
            Path::new(&model),
        ),
        other => panic!("unsupported mode: {other}"),
    }
}

fn drop_file_cache(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let file = std::fs::File::open(path).expect("open for cache drop");
        unsafe {
            libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
        }
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
}

fn run_tensor_mode(
    hfq: &HfqFile,
    transport: &mut dyn Transport,
    gpu: &mut Gpu,
    limit: Option<usize>,
    transport_name: &str,
) {
    let mut total_bytes = 0usize;
    let mut n = 0usize;
    let t0 = Instant::now();
    for info in hfq.tensors() {
        if limit.is_some_and(|max| n >= max) {
            break;
        }
        let (tensor, handle) = transport
            .fetch(info.data_offset, info.data_size, gpu)
            .unwrap_or_else(|e| panic!("fetch {} failed: {e}", info.name));
        transport.wait(&[handle]).expect("wait");
        total_bytes += info.data_size;
        n += 1;
        gpu.free_tensor(tensor).expect("free tensor");
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let gib = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    eprintln!(
        "mode=tensor transport={} tensors={} bytes={} elapsed={:.3}s throughput={:.2} GiB/s",
        transport_name,
        n,
        total_bytes,
        elapsed,
        gib / elapsed
    );
}

fn run_slab_mode(
    hfq: &HfqFile,
    transport: &mut dyn Transport,
    gpu: &mut Gpu,
    limit: Option<usize>,
    transport_name: &str,
    bank_size: usize,
    profile_io: bool,
    read_mode: &str,
    read_only: bool,
    prealloc_slabs: bool,
    model_path: &Path,
) {
    let tensors = hfq.tensors();
    let max_tensors = limit.unwrap_or(tensors.len()).min(tensors.len());
    let banks = build_banks(hfq, max_tensors, bank_size.max(1));
    if prealloc_slabs {
        return run_slab_prealloc_mode(
            hfq,
            gpu,
            &banks,
            transport_name,
            read_mode,
            read_only,
            model_path,
        );
    }
    if profile_io {
        return run_slab_profile_mode(
            hfq,
            gpu,
            &banks,
            transport_name,
            read_mode,
            read_only,
            model_path,
        );
    }

    let mut copied_bytes = 0usize;
    let mut logical_bytes = 0usize;
    let mut alias_count = 0usize;
    let t0 = Instant::now();
    for bank in &banks {
        let (slab, handle) = transport
            .fetch(bank.offset, bank.len, gpu)
            .unwrap_or_else(|e| {
                panic!(
                    "fetch bank offset={} len={} failed: {e}",
                    bank.offset, bank.len
                )
            });
        transport.wait(&[handle]).expect("wait");
        copied_bytes += bank.len;

        // Simulate the runtime view construction the real slab loader would
        // do. These aliases are non-owning and must not be freed separately.
        for &idx in &bank.tensor_indices {
            let info = &tensors[idx];
            let rel = info.data_offset - bank.offset;
            let alias = unsafe { alias_raw(&slab, rel, info.data_size) };
            logical_bytes += alias.buf.size();
            alias_count += 1;
            std::mem::forget(alias);
        }
        gpu.free_tensor(slab).expect("free slab");
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let copied_gib = copied_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let logical_gib = logical_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    eprintln!(
        "mode=slab transport={} banks={} aliases={} copied_bytes={} logical_bytes={} elapsed={:.3}s copied_throughput={:.2} GiB/s logical_throughput={:.2} GiB/s",
        transport_name,
        banks.len(),
        alias_count,
        copied_bytes,
        logical_bytes,
        elapsed,
        copied_gib / elapsed,
        logical_gib / elapsed
    );
}

#[derive(Default)]
struct SlabProfileStats {
    disk_read_bytes: usize,
    copied_bytes: usize,
    logical_bytes: usize,
    alias_count: usize,
    read_s: f64,
    alloc_s: f64,
    copy_s: f64,
    alias_s: f64,
}

fn run_slab_profile_mode(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    banks: &[Bank],
    transport_name: &str,
    read_mode: &str,
    read_only: bool,
    model_path: &Path,
) {
    let tensors = hfq.tensors();
    let file = open_probe_file(model_path, read_mode);
    let file_len = file.metadata().expect("model metadata").len() as usize;
    let max_bank_len = banks.iter().map(|b| b.len).max().unwrap_or(0);
    let max_direct_len = banks
        .iter()
        .map(|b| aligned_read_window(b, file_len).1)
        .max()
        .unwrap_or(0);
    let mut stats = SlabProfileStats::default();
    let mut heap = Vec::<u8>::new();
    let mut direct = AlignedBuf::new(max_direct_len.max(DIRECT_ALIGN), DIRECT_ALIGN);
    let mut pinned = if read_mode == "cached" && matches!(transport_name, "pinned" | "pinned-h2d") {
        Some(gpu.hip.host_malloc(max_bank_len, 0).expect("hipHostMalloc"))
    } else {
        None
    };

    if pinned.is_none() && read_mode == "cached" {
        heap.resize(max_bank_len, 0);
    }

    let total_t0 = Instant::now();
    for bank in banks {
        let (read_len, src, used_pinned): (usize, &[u8], bool) = if read_mode == "direct" {
            let (aligned_start, aligned_len, rel) = aligned_read_window(bank, file_len);
            let t_read = Instant::now();
            let got = read_direct_allow_eof(
                &file,
                direct.as_mut_slice(aligned_len),
                aligned_start as u64,
            )
            .unwrap_or_else(|e| {
                panic!(
                    "O_DIRECT read failed offset={} len={} aligned_offset={} aligned_len={}: {e}",
                    bank.offset, bank.len, aligned_start, aligned_len
                )
            });
            if got < rel + bank.len {
                panic!(
                    "short O_DIRECT read offset={} len={} got={} need={}",
                    bank.offset,
                    bank.len,
                    got,
                    rel + bank.len
                );
            }
            stats.read_s += t_read.elapsed().as_secs_f64();
            (got, &direct.as_slice(got)[rel..rel + bank.len], false)
        } else if let Some(host) = pinned.as_mut() {
            let dst = &mut unsafe { host.as_mut_slice() }[..bank.len];
            let t_read = Instant::now();
            file.read_exact_at(dst, bank.offset as u64)
                .unwrap_or_else(|e| {
                    panic!(
                        "read bank offset={} len={} failed: {e}",
                        bank.offset, bank.len
                    )
                });
            stats.read_s += t_read.elapsed().as_secs_f64();
            (bank.len, &host.as_slice()[..bank.len], true)
        } else {
            let dst = &mut heap[..bank.len];
            let t_read = Instant::now();
            file.read_exact_at(dst, bank.offset as u64)
                .unwrap_or_else(|e| {
                    panic!(
                        "read bank offset={} len={} failed: {e}",
                        bank.offset, bank.len
                    )
                });
            stats.read_s += t_read.elapsed().as_secs_f64();
            (bank.len, &heap[..bank.len], false)
        };
        stats.disk_read_bytes += read_len;
        stats.copied_bytes += bank.len;

        if read_only {
            continue;
        }

        let t_alloc = Instant::now();
        let buf = gpu.hip.malloc(bank.len).expect("hipMalloc slab");
        stats.alloc_s += t_alloc.elapsed().as_secs_f64();

        let t_copy = Instant::now();
        if used_pinned {
            let stream = Stream::null();
            gpu.hip
                .memcpy_htod_async(&buf, src, &stream)
                .expect("hipMemcpyAsync H2D");
            gpu.hip.stream_synchronize(&stream).expect("stream sync");
        } else {
            gpu.hip.memcpy_htod(&buf, src).expect("hipMemcpy H2D");
        }
        stats.copy_s += t_copy.elapsed().as_secs_f64();

        let slab = GpuTensor {
            buf,
            shape: vec![bank.len],
            dtype: DType::Raw,
        };
        let t_alias = Instant::now();
        for &idx in &bank.tensor_indices {
            let info = &tensors[idx];
            let rel = info.data_offset - bank.offset;
            let alias = unsafe { alias_raw(&slab, rel, info.data_size) };
            stats.logical_bytes += alias.buf.size();
            stats.alias_count += 1;
            std::mem::forget(alias);
        }
        stats.alias_s += t_alias.elapsed().as_secs_f64();
        gpu.free_tensor(slab).expect("free slab");
    }
    let total_s = total_t0.elapsed().as_secs_f64();
    let disk_gib = stats.disk_read_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let copied_gib = stats.copied_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let logical_gib = stats.logical_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    eprintln!(
        "mode=slab-profile transport={} read_mode={} read_only={} banks={} aliases={} disk_read_bytes={} copied_bytes={} logical_bytes={} total={:.3}s read={:.3}s alloc={:.3}s copy={:.3}s alias={:.6}s disk_read_throughput={:.2} GiB/s logical_total_throughput={:.2} GiB/s copy_throughput={:.2} GiB/s",
        transport_name,
        read_mode,
        read_only,
        banks.len(),
        stats.alias_count,
        stats.disk_read_bytes,
        stats.copied_bytes,
        stats.logical_bytes,
        total_s,
        stats.read_s,
        stats.alloc_s,
        stats.copy_s,
        stats.alias_s,
        disk_gib / stats.read_s.max(f64::MIN_POSITIVE),
        logical_gib / total_s.max(f64::MIN_POSITIVE),
        copied_gib / stats.copy_s.max(f64::MIN_POSITIVE)
    );
}

fn run_slab_prealloc_mode(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    banks: &[Bank],
    transport_name: &str,
    read_mode: &str,
    read_only: bool,
    model_path: &Path,
) {
    let effective_read_mode =
        if matches!(transport_name, "direct" | "direct-h2d" | "odirect") && read_mode == "cached" {
            "direct"
        } else {
            read_mode
        };
    let tensors = hfq.tensors();
    let file = open_probe_file(model_path, effective_read_mode);
    let file_len = file.metadata().expect("model metadata").len() as usize;
    let max_bank_len = banks.iter().map(|b| b.len).max().unwrap_or(0);
    let max_direct_len = banks
        .iter()
        .map(|b| aligned_read_window(b, file_len).1)
        .max()
        .unwrap_or(0);
    let mut stats = SlabProfileStats::default();
    let mut heap = Vec::<u8>::new();
    let mut direct = AlignedBuf::new(max_direct_len.max(DIRECT_ALIGN), DIRECT_ALIGN);
    let mut pinned =
        if effective_read_mode == "cached" && matches!(transport_name, "pinned" | "pinned-h2d") {
            Some(gpu.hip.host_malloc(max_bank_len, 0).expect("hipHostMalloc"))
        } else {
            None
        };

    if pinned.is_none() && effective_read_mode == "cached" {
        heap.resize(max_bank_len, 0);
    }

    let total_t0 = Instant::now();
    let t_alloc = Instant::now();
    let mut slabs = Vec::with_capacity(banks.len());
    if !read_only {
        for bank in banks {
            let buf = gpu
                .hip
                .malloc(bank.len)
                .expect("hipMalloc preallocated slab");
            slabs.push(GpuTensor {
                buf,
                shape: vec![bank.len],
                dtype: DType::Raw,
            });
        }
    }
    stats.alloc_s = t_alloc.elapsed().as_secs_f64();

    let load_t0 = Instant::now();
    for (bank_idx, bank) in banks.iter().enumerate() {
        let (read_len, src, used_pinned): (usize, &[u8], bool) = if effective_read_mode == "direct"
        {
            let (aligned_start, aligned_len, rel) = aligned_read_window(bank, file_len);
            let t_read = Instant::now();
            let got = read_direct_allow_eof(
                &file,
                direct.as_mut_slice(aligned_len),
                aligned_start as u64,
            )
            .unwrap_or_else(|e| {
                panic!(
                    "O_DIRECT read failed offset={} len={} aligned_offset={} aligned_len={}: {e}",
                    bank.offset, bank.len, aligned_start, aligned_len
                )
            });
            if got < rel + bank.len {
                panic!(
                    "short O_DIRECT read offset={} len={} got={} need={}",
                    bank.offset,
                    bank.len,
                    got,
                    rel + bank.len
                );
            }
            stats.read_s += t_read.elapsed().as_secs_f64();
            (got, &direct.as_slice(got)[rel..rel + bank.len], false)
        } else if let Some(host) = pinned.as_mut() {
            let dst = &mut unsafe { host.as_mut_slice() }[..bank.len];
            let t_read = Instant::now();
            file.read_exact_at(dst, bank.offset as u64)
                .unwrap_or_else(|e| {
                    panic!(
                        "read bank offset={} len={} failed: {e}",
                        bank.offset, bank.len
                    )
                });
            stats.read_s += t_read.elapsed().as_secs_f64();
            (bank.len, &host.as_slice()[..bank.len], true)
        } else {
            let dst = &mut heap[..bank.len];
            let t_read = Instant::now();
            file.read_exact_at(dst, bank.offset as u64)
                .unwrap_or_else(|e| {
                    panic!(
                        "read bank offset={} len={} failed: {e}",
                        bank.offset, bank.len
                    )
                });
            stats.read_s += t_read.elapsed().as_secs_f64();
            (bank.len, &heap[..bank.len], false)
        };
        stats.disk_read_bytes += read_len;
        stats.copied_bytes += bank.len;

        if read_only {
            continue;
        }

        let slab = &slabs[bank_idx];
        let t_copy = Instant::now();
        if used_pinned {
            let stream = Stream::null();
            gpu.hip
                .memcpy_htod_async(&slab.buf, src, &stream)
                .expect("hipMemcpyAsync H2D");
            gpu.hip.stream_synchronize(&stream).expect("stream sync");
        } else {
            gpu.hip.memcpy_htod(&slab.buf, src).expect("hipMemcpy H2D");
        }
        stats.copy_s += t_copy.elapsed().as_secs_f64();

        let t_alias = Instant::now();
        for &idx in &bank.tensor_indices {
            let info = &tensors[idx];
            let rel = info.data_offset - bank.offset;
            let alias = unsafe { alias_raw(slab, rel, info.data_size) };
            stats.logical_bytes += alias.buf.size();
            stats.alias_count += 1;
            std::mem::forget(alias);
        }
        stats.alias_s += t_alias.elapsed().as_secs_f64();
    }
    let load_s = load_t0.elapsed().as_secs_f64();
    let total_s = total_t0.elapsed().as_secs_f64();

    for slab in slabs {
        gpu.free_tensor(slab).expect("free preallocated slab");
    }

    let disk_gib = stats.disk_read_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let copied_gib = stats.copied_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let logical_gib = stats.logical_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    eprintln!(
        "mode=slab-prealloc transport={} read_mode={} read_only={} banks={} aliases={} disk_read_bytes={} copied_bytes={} logical_bytes={} total={:.3}s prealloc={:.3}s load={:.3}s read={:.3}s copy={:.3}s alias={:.6}s disk_read_throughput={:.2} GiB/s logical_load_throughput={:.2} GiB/s logical_total_throughput={:.2} GiB/s copy_throughput={:.2} GiB/s",
        transport_name,
        effective_read_mode,
        read_only,
        banks.len(),
        stats.alias_count,
        stats.disk_read_bytes,
        stats.copied_bytes,
        stats.logical_bytes,
        total_s,
        stats.alloc_s,
        load_s,
        stats.read_s,
        stats.copy_s,
        stats.alias_s,
        disk_gib / stats.read_s.max(f64::MIN_POSITIVE),
        logical_gib / load_s.max(f64::MIN_POSITIVE),
        logical_gib / total_s.max(f64::MIN_POSITIVE),
        copied_gib / stats.copy_s.max(f64::MIN_POSITIVE)
    );
}

fn open_probe_file(path: &Path, read_mode: &str) -> std::fs::File {
    match read_mode {
        "cached" => std::fs::File::open(path).expect("open model"),
        "direct" => {
            #[cfg(unix)]
            {
                std::fs::OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_DIRECT)
                    .open(path)
                    .expect("open model with O_DIRECT")
            }
            #[cfg(not(unix))]
            {
                panic!("--read-mode direct requires unix");
            }
        }
        other => panic!("unsupported --read-mode: {other}"),
    }
}

fn aligned_read_window(bank: &Bank, file_len: usize) -> (usize, usize, usize) {
    let start = align_down(bank.offset, DIRECT_ALIGN);
    let end = (bank.offset + bank.len).min(file_len);
    let aligned_end = align_up(end, DIRECT_ALIGN).min(file_len);
    (start, aligned_end - start, bank.offset - start)
}

fn align_down(v: usize, align: usize) -> usize {
    v & !(align - 1)
}

fn align_up(v: usize, align: usize) -> usize {
    (v + align - 1) & !(align - 1)
}

fn read_direct_allow_eof(
    file: &std::fs::File,
    dst: &mut [u8],
    offset: u64,
) -> std::io::Result<usize> {
    let mut done = 0usize;
    while done < dst.len() {
        let remaining = dst.len() - done;
        let n = file.read_at(&mut dst[done..], offset + done as u64)?;
        if n == 0 {
            break;
        }
        done += n;
        if n < remaining {
            break;
        }
    }
    Ok(done)
}

struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
}

impl AlignedBuf {
    fn new(len: usize, align: usize) -> Self {
        let mut ptr = std::ptr::null_mut();
        let rc = unsafe { libc::posix_memalign(&mut ptr, align, len.max(1)) };
        if rc != 0 {
            panic!("posix_memalign failed rc={rc}");
        }
        Self {
            ptr: ptr.cast(),
            len,
        }
    }

    fn as_mut_slice(&mut self, len: usize) -> &mut [u8] {
        assert!(len <= self.len);
        unsafe { std::slice::from_raw_parts_mut(self.ptr, len) }
    }

    fn as_slice(&self, len: usize) -> &[u8] {
        assert!(len <= self.len);
        unsafe { std::slice::from_raw_parts(self.ptr, len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        unsafe { libc::free(self.ptr.cast()) };
    }
}

fn build_banks(hfq: &HfqFile, max_tensors: usize, bank_size: usize) -> Vec<Bank> {
    let tensors = hfq.tensors();
    let mut banks = Vec::new();
    let mut cur: Option<Bank> = None;

    for (idx, info) in tensors.iter().take(max_tensors).enumerate() {
        let start = info.data_offset;
        let end = info.data_offset + info.data_size;
        match cur.as_mut() {
            Some(bank) => {
                let next_len = end - bank.offset;
                if next_len <= bank_size || bank.tensor_indices.is_empty() {
                    bank.len = next_len;
                    bank.tensor_indices.push(idx);
                } else {
                    banks.push(cur.take().unwrap());
                    cur = Some(Bank {
                        offset: start,
                        len: info.data_size,
                        tensor_indices: vec![idx],
                    });
                }
            }
            None => {
                cur = Some(Bank {
                    offset: start,
                    len: info.data_size,
                    tensor_indices: vec![idx],
                });
            }
        }
    }
    if let Some(bank) = cur {
        banks.push(bank);
    }
    banks
}

unsafe fn alias_raw(slab: &GpuTensor, byte_offset: usize, len: usize) -> GpuTensor {
    let ptr = (slab.buf.as_ptr() as *mut u8).add(byte_offset) as *mut std::ffi::c_void;
    GpuTensor {
        buf: DeviceBuffer::from_raw(ptr, len),
        shape: vec![len],
        dtype: DType::Raw,
    }
}
