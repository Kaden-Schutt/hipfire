// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Screen bounded parallel HFQ reads without changing the production loader.
//!
//! The benchmark compares the current loader-shaped path (one reader, reusable
//! scratch buffer, then copy into the final packed allocation) with direct
//! reads into final buffers using one or more reader lanes. Results are consumed
//! in canonical HFQ order, so `--gpu-upload` also tests overlap between parallel
//! storage reads and the existing synchronous HIP upload path without changing
//! allocation order.
//!
//! This is a screening tool, not model throughput or promotion evidence.
//!
//! Examples:
//! ```text
//! cargo run --release -p hipfire-runtime --example hfq_load_pipeline_bench -- \
//!   --model /models/lfm2.5-350m.mq4 --lanes 1,2,4 --repeat 2
//!
//! cargo run --release -p hipfire-runtime --example hfq_load_pipeline_bench -- \
//!   --model /models/deepseek-v4-flash-0731.mq2r \
//!   --ds4-expert-layers 0:2 --lanes 1,2,4 --max-bytes 8GiB
//!
//! cargo run --release -p hipfire-runtime --example hfq_load_pipeline_bench -- \
//!   --model /models/lfm2.5-350m.mq4 --lanes 4 --gpu-upload --device 0
//! ```

use hipfire_runtime::hfq::{HfqFile, HfqTensorInfo};
use rdna_compute::{Gpu, GpuTensor};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const REPORT_SCHEMA: u32 = 1;
const DEFAULT_LANES: &[usize] = &[1, 2, 4];

type AnyError = Box<dyn Error>;
type Result<T> = std::result::Result<T, AnyError>;

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum ReadStrategy {
    ScratchPack,
    DirectPack,
}

impl ReadStrategy {
    fn label(self) -> &'static str {
        match self {
            Self::ScratchPack => "scratch-pack",
            Self::DirectPack => "direct-pack",
        }
    }
}

#[derive(Debug)]
struct Args {
    model: PathBuf,
    lanes: Vec<usize>,
    repeat: usize,
    prefix: Option<String>,
    ds4_expert_layers: Option<(usize, usize)>,
    max_bytes: Option<usize>,
    keep_cache: bool,
    gpu_upload: bool,
    device: i32,
    json_out: Option<PathBuf>,
}

#[derive(Clone, Debug)]
struct Segment {
    source_offset: usize,
    len: usize,
    destination_offset: usize,
}

#[derive(Clone, Debug)]
struct ReadJob {
    index: usize,
    name: String,
    output_len: usize,
    segments: Vec<Segment>,
}

#[derive(Debug)]
struct WorkerResult {
    job_index: usize,
    name: String,
    data: Vec<u8>,
    data_hash: u64,
    pread: Duration,
    pack_copy: Duration,
    fadvise: Duration,
    checksum: Duration,
}

#[derive(Debug, Serialize)]
struct RunReport {
    strategy: ReadStrategy,
    lanes: usize,
    repetition: usize,
    jobs: usize,
    segments: usize,
    selected_bytes: u64,
    cache_drop_ms: f64,
    pipeline_ms: f64,
    throughput_gb_s: f64,
    throughput_gib_s: f64,
    summed_pread_ms: f64,
    summed_pack_copy_ms: f64,
    summed_fadvise_ms: f64,
    summed_checksum_ms: f64,
    upload_ms: Option<f64>,
    cleanup_ms: Option<f64>,
    peak_in_flight_bytes: u64,
    out_of_order_completions: usize,
    canonical_consumption: bool,
    checksum: String,
    checksum_matches_baseline: bool,
}

#[derive(Debug, Serialize)]
struct BenchReport {
    schema_version: u32,
    model: PathBuf,
    model_file_bytes: u64,
    arch_id: u32,
    selection: String,
    selected_jobs: usize,
    selected_segments: usize,
    selected_bytes: u64,
    max_bytes: Option<u64>,
    drop_cache: bool,
    gpu_upload: bool,
    device: Option<i32>,
    plan_hash: String,
    runs: Vec<RunReport>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ExpertPart {
    W1,
    W2,
    W3,
}

fn usage() {
    println!(
        "hfq_load_pipeline_bench --model PATH [OPTIONS]\n\
         \n\
         Options:\n\
           --lanes 1,2,4                 Reader-lane candidates (default: 1,2,4)\n\
           --repeat N                    Repetitions per case (default: 1)\n\
           --prefix TEXT                 Select tensor names containing TEXT\n\
           --ds4-expert-layers A:B       Pack DS4 expert layers A through B-1\n\
           --max-bytes SIZE              Stop at a whole-job boundary (e.g. 8GiB)\n\
           --keep-cache                  Do not issue POSIX_FADV_DONTNEED\n\
           --gpu-upload                  Upload in canonical order while readers run\n\
           --device N                    HIP device for --gpu-upload (default: 0)\n\
           --json-out PATH               Write the complete report as JSON\n\
           -h, --help                    Show this help\n\
         \n\
         The scratch-pack/1-lane baseline is always run first. Every candidate\n\
         must produce the same full-data checksum. This tool never mutates the HFQ."
    );
}

fn next_value<I>(args: &mut I, flag: &str) -> Result<String>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| format!("{flag} requires a value").into())
}

fn parse_args() -> Result<Option<Args>> {
    let mut raw = std::env::args().skip(1);
    let mut model = None;
    let mut lanes = DEFAULT_LANES.to_vec();
    let mut repeat = 1usize;
    let mut prefix = None;
    let mut ds4_expert_layers = None;
    let mut max_bytes = None;
    let mut keep_cache = false;
    let mut gpu_upload = false;
    let mut device = 0i32;
    let mut json_out = None;

    while let Some(arg) = raw.next() {
        match arg.as_str() {
            "-h" | "--help" => return Ok(None),
            "--model" => model = Some(PathBuf::from(next_value(&mut raw, "--model")?)),
            "--lanes" => {
                lanes = next_value(&mut raw, "--lanes")?
                    .split(',')
                    .map(|value| {
                        value
                            .parse::<usize>()
                            .map_err(|error| format!("invalid lane count {value:?}: {error}"))
                    })
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                if lanes.is_empty() || lanes.iter().any(|&lane| lane == 0) {
                    return Err("--lanes requires positive comma-separated integers".into());
                }
                lanes.sort_unstable();
                lanes.dedup();
            }
            "--repeat" => {
                repeat = next_value(&mut raw, "--repeat")?.parse()?;
                if repeat == 0 {
                    return Err("--repeat must be positive".into());
                }
            }
            "--prefix" => prefix = Some(next_value(&mut raw, "--prefix")?),
            "--ds4-expert-layers" => {
                ds4_expert_layers =
                    Some(parse_range(&next_value(&mut raw, "--ds4-expert-layers")?)?);
            }
            "--max-bytes" => {
                max_bytes = Some(parse_size(&next_value(&mut raw, "--max-bytes")?)?);
            }
            "--keep-cache" => keep_cache = true,
            "--gpu-upload" => gpu_upload = true,
            "--device" => device = next_value(&mut raw, "--device")?.parse()?,
            "--json-out" => json_out = Some(PathBuf::from(next_value(&mut raw, "--json-out")?)),
            _ => return Err(format!("unknown argument {arg:?}; use --help").into()),
        }
    }

    if prefix.is_some() && ds4_expert_layers.is_some() {
        return Err("--prefix and --ds4-expert-layers are mutually exclusive".into());
    }

    Ok(Some(Args {
        model: model.ok_or("--model PATH is required")?,
        lanes,
        repeat,
        prefix,
        ds4_expert_layers,
        max_bytes,
        keep_cache,
        gpu_upload,
        device,
        json_out,
    }))
}

fn parse_range(raw: &str) -> Result<(usize, usize)> {
    let (start, end) = raw.split_once(':').ok_or("layer range must be START:END")?;
    let start = start.parse::<usize>()?;
    let end = end.parse::<usize>()?;
    if start >= end {
        return Err("layer range must satisfy START < END".into());
    }
    Ok((start, end))
}

fn parse_size(raw: &str) -> Result<usize> {
    let split = raw
        .find(|character: char| !character.is_ascii_digit() && character != '.')
        .unwrap_or(raw.len());
    let number = raw[..split].parse::<f64>()?;
    let suffix = raw[split..].trim().to_ascii_lowercase();
    let multiplier = match suffix.as_str() {
        "" | "b" => 1f64,
        "k" | "kb" => 1_000f64,
        "m" | "mb" => 1_000_000f64,
        "g" | "gb" => 1_000_000_000f64,
        "t" | "tb" => 1_000_000_000_000f64,
        "kib" => 1_024f64,
        "mib" => 1_048_576f64,
        "gib" => 1_073_741_824f64,
        "tib" => 1_099_511_627_776f64,
        _ => return Err(format!("unsupported size suffix {suffix:?}").into()),
    };
    let bytes = number * multiplier;
    if !bytes.is_finite() || bytes <= 0.0 || bytes > usize::MAX as f64 {
        return Err(format!("invalid byte size {raw:?}").into());
    }
    Ok(bytes as usize)
}

fn generic_jobs(tensors: &[HfqTensorInfo], prefix: Option<&str>) -> Vec<ReadJob> {
    tensors
        .iter()
        .filter(|tensor| prefix.is_none_or(|needle| tensor.name.contains(needle)))
        .enumerate()
        .map(|(index, tensor)| ReadJob {
            index,
            name: tensor.name.clone(),
            output_len: tensor.data_size,
            segments: vec![Segment {
                source_offset: tensor.data_offset,
                len: tensor.data_size,
                destination_offset: 0,
            }],
        })
        .collect()
}

fn parse_expert_tensor(name: &str) -> Option<(usize, usize, ExpertPart)> {
    let marker = ".ffn.experts.";
    let marker_offset = name.find(marker)?;
    let layer = name[..marker_offset].rsplit('.').next()?.parse().ok()?;
    let mut suffix = name[marker_offset + marker.len()..].split('.');
    let expert = suffix.next()?.parse().ok()?;
    let part = match suffix.next()? {
        "w1" => ExpertPart::W1,
        "w2" => ExpertPart::W2,
        "w3" => ExpertPart::W3,
        _ => return None,
    };
    (suffix.next()? == "weight" && suffix.next().is_none()).then_some((layer, expert, part))
}

fn ds4_expert_jobs(tensors: &[HfqTensorInfo], start: usize, end: usize) -> Result<Vec<ReadJob>> {
    let mut table = BTreeMap::new();
    let mut experts_by_layer: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for tensor in tensors {
        let Some((layer, expert, part)) = parse_expert_tensor(&tensor.name) else {
            continue;
        };
        if (start..end).contains(&layer) {
            table.insert((layer, expert, part), tensor.clone());
            experts_by_layer.entry(layer).or_default().insert(expert);
        }
    }

    let mut jobs = Vec::new();
    for layer in start..end {
        let experts = experts_by_layer
            .get(&layer)
            .ok_or_else(|| format!("no routed-expert tensors found for DS4 layer {layer}"))?;
        for &expert in experts {
            for part in [ExpertPart::W1, ExpertPart::W2, ExpertPart::W3] {
                if !table.contains_key(&(layer, expert, part)) {
                    return Err(
                        format!("DS4 layer {layer} expert {expert} is missing {part:?}").into(),
                    );
                }
            }
        }

        let mut w2_segments = Vec::with_capacity(experts.len());
        let mut w2_len = 0usize;
        for &expert in experts {
            let tensor = &table[&(layer, expert, ExpertPart::W2)];
            w2_segments.push(Segment {
                source_offset: tensor.data_offset,
                len: tensor.data_size,
                destination_offset: w2_len,
            });
            w2_len += tensor.data_size;
        }
        jobs.push(ReadJob {
            index: jobs.len(),
            name: format!("layers.{layer}.ffn.experts.w2.packed"),
            output_len: w2_len,
            segments: w2_segments,
        });

        let mut gate_up_segments = Vec::with_capacity(experts.len() * 2);
        let mut gate_up_len = 0usize;
        for &expert in experts {
            for part in [ExpertPart::W1, ExpertPart::W3] {
                let tensor = &table[&(layer, expert, part)];
                gate_up_segments.push(Segment {
                    source_offset: tensor.data_offset,
                    len: tensor.data_size,
                    destination_offset: gate_up_len,
                });
                gate_up_len += tensor.data_size;
            }
        }
        jobs.push(ReadJob {
            index: jobs.len(),
            name: format!("layers.{layer}.ffn.experts.gate_up.packed"),
            output_len: gate_up_len,
            segments: gate_up_segments,
        });
    }
    Ok(jobs)
}

fn cap_jobs(mut jobs: Vec<ReadJob>, max_bytes: Option<usize>) -> Vec<ReadJob> {
    let Some(max_bytes) = max_bytes else {
        return jobs;
    };
    let mut retained = 0usize;
    let mut total = 0usize;
    for job in &jobs {
        if retained > 0 && total.saturating_add(job.output_len) > max_bytes {
            break;
        }
        total = total.saturating_add(job.output_len);
        retained += 1;
    }
    jobs.truncate(retained);
    for (index, job) in jobs.iter_mut().enumerate() {
        job.index = index;
    }
    jobs
}

#[cfg(unix)]
fn read_exact_at(file: &File, mut buffer: &mut [u8], mut offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    while !buffer.is_empty() {
        let read = file.read_at(buffer, offset)?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short HFQ pread",
            ));
        }
        offset += read as u64;
        buffer = &mut buffer[read..];
    }
    Ok(())
}

#[cfg(not(unix))]
fn read_exact_at(file: &File, buffer: &mut [u8], offset: u64) -> io::Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = file.try_clone()?;
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(buffer)
}

#[cfg(unix)]
fn advise(file: &File, offset: usize, len: usize, advice: i32) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    let status = unsafe {
        libc::posix_fadvise(
            file.as_raw_fd(),
            offset as libc::off_t,
            len as libc::off_t,
            advice,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(io::Error::from_raw_os_error(status))
    }
}

#[cfg(not(unix))]
fn advise(_file: &File, _offset: usize, _len: usize, _advice: i32) -> io::Result<()> {
    Ok(())
}

fn drop_selected_pages(path: &Path, jobs: &[ReadJob]) -> Result<Duration> {
    let started = Instant::now();
    let file = File::open(path)?;
    for segment in jobs.iter().flat_map(|job| &job.segments) {
        advise(
            &file,
            segment.source_offset,
            segment.len,
            libc::POSIX_FADV_DONTNEED,
        )?;
    }
    Ok(started.elapsed())
}

fn hash_bytes(data: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut chunks = data.chunks_exact(8);
    for chunk in &mut chunks {
        let value = u64::from_le_bytes(chunk.try_into().expect("eight-byte chunk"));
        hash ^= value;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    for &byte in chunks.remainder() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn combine_hash(hash: u64, value: u64) -> u64 {
    (hash ^ value)
        .rotate_left(17)
        .wrapping_mul(0x9e37_79b1_85eb_ca87)
}

fn plan_hash(jobs: &[ReadJob]) -> u64 {
    let mut hash = 0x6a09_e667_f3bc_c909u64;
    for job in jobs {
        hash = combine_hash(hash, job.index as u64);
        hash = combine_hash(hash, job.output_len as u64);
        hash = combine_hash(hash, hash_bytes(job.name.as_bytes()));
        for segment in &job.segments {
            hash = combine_hash(hash, segment.source_offset as u64);
            hash = combine_hash(hash, segment.len as u64);
            hash = combine_hash(hash, segment.destination_offset as u64);
        }
    }
    hash
}

fn worker(
    path: PathBuf,
    strategy: ReadStrategy,
    drop_cache: bool,
    receiver: Arc<Mutex<mpsc::Receiver<ReadJob>>>,
    sender: mpsc::SyncSender<std::result::Result<WorkerResult, String>>,
) {
    let result = (|| -> std::result::Result<(), String> {
        let file =
            File::open(&path).map_err(|error| format!("open {}: {error}", path.display()))?;
        advise(&file, 0, 0, libc::POSIX_FADV_SEQUENTIAL)
            .map_err(|error| format!("fadvise sequential: {error}"))?;
        let mut scratch = Vec::new();

        loop {
            let job = {
                let guard = receiver
                    .lock()
                    .map_err(|_| "reader command queue poisoned".to_string())?;
                match guard.recv() {
                    Ok(job) => job,
                    Err(_) => break,
                }
            };
            let mut data = vec![0u8; job.output_len];
            let mut pread = Duration::ZERO;
            let mut pack_copy = Duration::ZERO;
            let mut fadvise = Duration::ZERO;

            for segment in &job.segments {
                let destination =
                    segment.destination_offset..segment.destination_offset + segment.len;
                match strategy {
                    ReadStrategy::DirectPack => {
                        let started = Instant::now();
                        read_exact_at(&file, &mut data[destination], segment.source_offset as u64)
                            .map_err(|error| format!("pread {}: {error}", job.name))?;
                        pread += started.elapsed();
                    }
                    ReadStrategy::ScratchPack if job.segments.len() > 1 => {
                        scratch.resize(segment.len, 0);
                        let started = Instant::now();
                        read_exact_at(&file, &mut scratch, segment.source_offset as u64)
                            .map_err(|error| format!("pread {}: {error}", job.name))?;
                        pread += started.elapsed();
                        let started = Instant::now();
                        data[destination].copy_from_slice(&scratch);
                        pack_copy += started.elapsed();
                    }
                    ReadStrategy::ScratchPack => {
                        let started = Instant::now();
                        read_exact_at(&file, &mut data[destination], segment.source_offset as u64)
                            .map_err(|error| format!("pread {}: {error}", job.name))?;
                        pread += started.elapsed();
                    }
                }
                if drop_cache {
                    let started = Instant::now();
                    advise(
                        &file,
                        segment.source_offset,
                        segment.len,
                        libc::POSIX_FADV_DONTNEED,
                    )
                    .map_err(|error| format!("fadvise {}: {error}", job.name))?;
                    fadvise += started.elapsed();
                }
            }
            let started = Instant::now();
            let data_hash = hash_bytes(&data);
            let checksum = started.elapsed();
            sender
                .send(Ok(WorkerResult {
                    job_index: job.index,
                    name: job.name,
                    data,
                    data_hash,
                    pread,
                    pack_copy,
                    fadvise,
                    checksum,
                }))
                .map_err(|_| "result consumer disconnected".to_string())?;
        }
        Ok(())
    })();
    if let Err(error) = result {
        let _ = sender.send(Err(error));
    }
}

fn run_case(
    args: &Args,
    jobs: &[ReadJob],
    strategy: ReadStrategy,
    lanes: usize,
    repetition: usize,
    baseline_checksum: Option<u64>,
    mut gpu: Option<&mut Gpu>,
) -> Result<(RunReport, u64)> {
    let cache_drop = if args.keep_cache {
        Duration::ZERO
    } else {
        drop_selected_pages(&args.model, jobs)?
    };
    let (job_sender, job_receiver) = mpsc::sync_channel::<ReadJob>(lanes);
    let (result_sender, result_receiver) =
        mpsc::sync_channel::<std::result::Result<WorkerResult, String>>(lanes);
    let job_receiver = Arc::new(Mutex::new(job_receiver));
    let drop_cache = !args.keep_cache;
    let mut handles = Vec::with_capacity(lanes);
    for _ in 0..lanes {
        let path = args.model.clone();
        let receiver = Arc::clone(&job_receiver);
        let sender = result_sender.clone();
        handles.push(thread::spawn(move || {
            worker(path, strategy, drop_cache, receiver, sender)
        }));
    }
    drop(result_sender);

    let selected_bytes = jobs.iter().map(|job| job.output_len).sum::<usize>();
    let mut next_submit = 0usize;
    let mut next_consume = 0usize;
    let mut in_flight_bytes = 0usize;
    let mut peak_in_flight_bytes = 0usize;
    while next_submit < jobs.len() && next_submit < lanes {
        let job = jobs[next_submit].clone();
        in_flight_bytes += job.output_len;
        peak_in_flight_bytes = peak_in_flight_bytes.max(in_flight_bytes);
        job_sender.send(job)?;
        next_submit += 1;
    }

    let pipeline_started = Instant::now();
    let mut pending = BTreeMap::new();
    let mut out_of_order_completions = 0usize;
    let mut summed_pread = Duration::ZERO;
    let mut summed_pack_copy = Duration::ZERO;
    let mut summed_fadvise = Duration::ZERO;
    let mut summed_checksum = Duration::ZERO;
    let mut upload = Duration::ZERO;
    let mut final_hash = 0x510e_527f_ade6_82d1u64;
    let mut gpu_tensors: Vec<GpuTensor> = Vec::new();

    while next_consume < jobs.len() {
        let result = result_receiver
            .recv()
            .map_err(|_| "all reader lanes exited before completing the plan")??;
        if result.job_index != next_consume {
            out_of_order_completions += 1;
        }
        pending.insert(result.job_index, result);

        while let Some(result) = pending.remove(&next_consume) {
            if result.name != jobs[next_consume].name {
                return Err(format!(
                    "canonical order mismatch at {}: {:?} != {:?}",
                    next_consume, result.name, jobs[next_consume].name
                )
                .into());
            }
            summed_pread += result.pread;
            summed_pack_copy += result.pack_copy;
            summed_fadvise += result.fadvise;
            summed_checksum += result.checksum;
            final_hash = combine_hash(final_hash, next_consume as u64);
            final_hash = combine_hash(final_hash, result.data.len() as u64);
            final_hash = combine_hash(final_hash, result.data_hash);

            if let Some(gpu) = gpu.as_deref_mut() {
                let started = Instant::now();
                let tensor = gpu.upload_raw(&result.data, &[result.data.len()])?;
                upload += started.elapsed();
                gpu_tensors.push(tensor);
            }
            in_flight_bytes = in_flight_bytes.saturating_sub(result.data.len());
            next_consume += 1;

            if next_submit < jobs.len() {
                let job = jobs[next_submit].clone();
                in_flight_bytes += job.output_len;
                peak_in_flight_bytes = peak_in_flight_bytes.max(in_flight_bytes);
                job_sender.send(job)?;
                next_submit += 1;
            }
        }
    }
    let pipeline = pipeline_started.elapsed();
    drop(job_sender);
    for handle in handles {
        handle.join().map_err(|_| "reader lane panicked")?;
    }

    let cleanup = if let Some(gpu) = gpu.as_deref_mut() {
        let started = Instant::now();
        for tensor in gpu_tensors {
            gpu.free_tensor(tensor)?;
        }
        gpu.drain_pool();
        Some(started.elapsed())
    } else {
        None
    };

    let seconds = pipeline.as_secs_f64();
    let checksum_matches_baseline = baseline_checksum.is_none_or(|expected| expected == final_hash);
    let report = RunReport {
        strategy,
        lanes,
        repetition,
        jobs: jobs.len(),
        segments: jobs.iter().map(|job| job.segments.len()).sum(),
        selected_bytes: selected_bytes as u64,
        cache_drop_ms: duration_ms(cache_drop),
        pipeline_ms: duration_ms(pipeline),
        throughput_gb_s: selected_bytes as f64 / 1_000_000_000.0 / seconds,
        throughput_gib_s: selected_bytes as f64 / 1_073_741_824.0 / seconds,
        summed_pread_ms: duration_ms(summed_pread),
        summed_pack_copy_ms: duration_ms(summed_pack_copy),
        summed_fadvise_ms: duration_ms(summed_fadvise),
        summed_checksum_ms: duration_ms(summed_checksum),
        upload_ms: args.gpu_upload.then(|| duration_ms(upload)),
        cleanup_ms: cleanup.map(duration_ms),
        peak_in_flight_bytes: peak_in_flight_bytes as u64,
        out_of_order_completions,
        canonical_consumption: true,
        checksum: format!("{final_hash:016x}"),
        checksum_matches_baseline,
    };
    Ok((report, final_hash))
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn print_run(report: &RunReport) {
    let upload = report
        .upload_ms
        .map(|value| format!("{value:10.1}"))
        .unwrap_or_else(|| "         -".to_string());
    println!(
        "{:<12} {:>5} {:>4} {:>11.1} {:>9.3} {:>9.3} {:>10.1} {:>10} {}",
        report.strategy.label(),
        report.lanes,
        report.repetition,
        report.pipeline_ms,
        report.throughput_gb_s,
        report.throughput_gib_s,
        report.summed_pack_copy_ms,
        upload,
        if report.checksum_matches_baseline {
            "exact"
        } else {
            "MISMATCH"
        }
    );
}

fn main() -> Result<()> {
    let Some(args) = parse_args()? else {
        usage();
        return Ok(());
    };
    let metadata = std::fs::metadata(&args.model)?;
    let hfq = HfqFile::open(&args.model)?;
    if hfq.has_overlay() {
        return Err(
            "an HFQ overlay is active; unset HIPFIRE_REAP_PLAN so direct file offsets are unambiguous"
                .into(),
        );
    }
    let arch_id = hfq.arch_id;
    let tensors = hfq.tensors().to_vec();
    drop(hfq); // The benchmark requires the model mmap to be gone before DONTNEED.

    let (selection, jobs) = if let Some((start, end)) = args.ds4_expert_layers {
        (
            format!("ds4_expert_layers_{start}:{end}"),
            ds4_expert_jobs(&tensors, start, end)?,
        )
    } else {
        (
            args.prefix
                .as_ref()
                .map(|prefix| format!("tensor_name_contains_{prefix}"))
                .unwrap_or_else(|| "all_tensors".to_string()),
            generic_jobs(&tensors, args.prefix.as_deref()),
        )
    };
    let jobs = cap_jobs(jobs, args.max_bytes);
    if jobs.is_empty() {
        return Err("selection produced no read jobs".into());
    }
    let selected_bytes = jobs.iter().map(|job| job.output_len).sum::<usize>();
    let selected_segments = jobs.iter().map(|job| job.segments.len()).sum::<usize>();
    let plan_hash = plan_hash(&jobs);

    println!("HFQ parallel-load screening microbench");
    println!("model:      {}", args.model.display());
    println!("arch_id:    {arch_id}");
    println!("selection:  {selection}");
    println!(
        "plan:       {} jobs, {} segments, {:.3} GB, hash {:016x}",
        jobs.len(),
        selected_segments,
        selected_bytes as f64 / 1_000_000_000.0,
        plan_hash
    );
    println!(
        "cache:      {}",
        if args.keep_cache {
            "retained"
        } else {
            "DONTNEED before and after each read"
        }
    );
    println!(
        "upload:     {}",
        if args.gpu_upload {
            format!("canonical HIP device {}", args.device)
        } else {
            "disabled".to_string()
        }
    );
    println!();
    println!(
        "{:<12} {:>5} {:>4} {:>11} {:>9} {:>9} {:>10} {:>10} verdict",
        "strategy", "lanes", "rep", "wall_ms", "GB/s", "GiB/s", "copy_ms", "upload_ms"
    );

    let mut gpu = if args.gpu_upload {
        Some(Gpu::init_with_device(args.device)?)
    } else {
        None
    };
    let mut runs = Vec::new();
    let mut baseline_checksum = None;
    for repetition in 1..=args.repeat {
        let (report, checksum) = run_case(
            &args,
            &jobs,
            ReadStrategy::ScratchPack,
            1,
            repetition,
            baseline_checksum,
            gpu.as_mut(),
        )?;
        baseline_checksum.get_or_insert(checksum);
        if !report.checksum_matches_baseline {
            return Err("scratch-pack baseline changed across repetitions".into());
        }
        print_run(&report);
        runs.push(report);
    }

    for &lanes in &args.lanes {
        for repetition in 1..=args.repeat {
            let (report, _) = run_case(
                &args,
                &jobs,
                ReadStrategy::DirectPack,
                lanes,
                repetition,
                baseline_checksum,
                gpu.as_mut(),
            )?;
            print_run(&report);
            if !report.checksum_matches_baseline {
                return Err(format!(
                    "direct-pack/{lanes} checksum differs from scratch-pack baseline"
                )
                .into());
            }
            runs.push(report);
        }
    }

    let report = BenchReport {
        schema_version: REPORT_SCHEMA,
        model: args.model.clone(),
        model_file_bytes: metadata.len(),
        arch_id,
        selection,
        selected_jobs: jobs.len(),
        selected_segments,
        selected_bytes: selected_bytes as u64,
        max_bytes: args.max_bytes.map(|bytes| bytes as u64),
        drop_cache: !args.keep_cache,
        gpu_upload: args.gpu_upload,
        device: args.gpu_upload.then_some(args.device),
        plan_hash: format!("{plan_hash:016x}"),
        runs,
    };
    if let Some(path) = &args.json_out {
        let json = serde_json::to_vec_pretty(&report)?;
        std::fs::write(path, json)?;
        println!("\nJSON: {}", path.display());
    }
    println!(
        "\nAll cases produced checksum {}. Screening only; no runtime path changed.",
        report.runs[0].checksum
    );
    Ok(())
}
