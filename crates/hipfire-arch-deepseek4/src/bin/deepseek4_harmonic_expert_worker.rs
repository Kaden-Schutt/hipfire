// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! Fault-contained gfx1151 routed-expert worker for DeepSeek V4 harmonic AR.
//!
//! This process owns exactly one HIP context, one ROCr physical-device
//! identity, the routed MQ2R weights, and local expert scratch. The parent
//! supervisor owns deadlines and may terminate this process without asking the
//! dense GPU to wait for it. No peer GPU is opened here.

#[cfg(not(unix))]
compile_error!("deepseek4 harmonic workers currently require Unix-domain sockets");

use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use hipfire_arch_deepseek4::{
    harmonic_monotonic_tick, DeepseekV4, DeepseekV4ArtifactReceipt,
    DeepseekV4HarmonicExpertService, DeepseekV4VerifiedArtifact, HarmonicExpertMappedPoll,
    HarmonicExpertWorkerCommand, HarmonicExpertWorkerEvent, HarmonicGpuMapping, HarmonicOwner,
    HarmonicSharedRing,
};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use redline_rocr::{GpuSelector, PciBusId, Runtime};

const EXPECTED_ARCH: &str = "gfx1151";

#[derive(Debug)]
struct Args {
    model: PathBuf,
    model_sha256: String,
    model_len: u64,
    model_mtime_secs: u64,
    model_mtime_nanos: u32,
    pci_bus_id: String,
    ring: PathBuf,
    control_socket: PathBuf,
    allocation_generation: u64,
    first_epoch: u64,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut model = None;
        let mut model_sha256 = None;
        let mut model_len = None;
        let mut model_mtime_secs = None;
        let mut model_mtime_nanos = None;
        let mut pci_bus_id = None;
        let mut ring = None;
        let mut control_socket = None;
        let mut allocation_generation = None;
        let mut first_epoch = None;
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            let mut value = || {
                args.next()
                    .ok_or_else(|| format!("missing value after {flag}"))
            };
            match flag.as_str() {
                "--model" => model = Some(PathBuf::from(value()?)),
                "--model-sha256" => model_sha256 = Some(value()?),
                "--model-len" => {
                    let raw = value()?;
                    model_len = Some(
                        raw.parse::<u64>()
                            .map_err(|error| format!("invalid --model-len {raw:?}: {error}"))?,
                    );
                }
                "--model-mtime-secs" => {
                    let raw = value()?;
                    model_mtime_secs =
                        Some(raw.parse::<u64>().map_err(|error| {
                            format!("invalid --model-mtime-secs {raw:?}: {error}")
                        })?);
                }
                "--model-mtime-nanos" => {
                    let raw = value()?;
                    model_mtime_nanos = Some(raw.parse::<u32>().map_err(|error| {
                        format!("invalid --model-mtime-nanos {raw:?}: {error}")
                    })?);
                }
                "--pci-bdf" => pci_bus_id = Some(value()?),
                "--ring" => ring = Some(PathBuf::from(value()?)),
                "--control-socket" => control_socket = Some(PathBuf::from(value()?)),
                "--generation" => {
                    let raw = value()?;
                    allocation_generation = Some(raw.parse::<u64>().map_err(|error| {
                        format!("invalid nonzero --generation {raw:?}: {error}")
                    })?);
                }
                "--first-epoch" => {
                    let raw = value()?;
                    first_epoch = Some(raw.parse::<u64>().map_err(|error| {
                        format!("invalid nonzero --first-epoch {raw:?}: {error}")
                    })?);
                }
                "--help" | "-h" => {
                    return Err(
                        "usage: deepseek4-harmonic-expert-worker --model PATH --model-sha256 HEX --model-len N --model-mtime-secs N --model-mtime-nanos N --pci-bdf 0000:BB:DD.F --ring PATH --control-socket PATH --generation N --first-epoch N"
                            .to_owned(),
                    );
                }
                _ => return Err(format!("unknown argument {flag:?}")),
            }
        }
        let allocation_generation =
            allocation_generation.ok_or_else(|| "missing --generation".to_owned())?;
        if allocation_generation == 0 {
            return Err("--generation must be nonzero".to_owned());
        }
        let first_epoch = first_epoch.ok_or_else(|| "missing --first-epoch".to_owned())?;
        if first_epoch == 0 {
            return Err("--first-epoch must be nonzero".to_owned());
        }
        Ok(Self {
            model: model.ok_or_else(|| "missing --model".to_owned())?,
            model_sha256: model_sha256.ok_or_else(|| "missing --model-sha256".to_owned())?,
            model_len: model_len.ok_or_else(|| "missing --model-len".to_owned())?,
            model_mtime_secs: model_mtime_secs
                .ok_or_else(|| "missing --model-mtime-secs".to_owned())?,
            model_mtime_nanos: model_mtime_nanos
                .ok_or_else(|| "missing --model-mtime-nanos".to_owned())?,
            pci_bus_id: pci_bus_id.ok_or_else(|| "missing --pci-bdf".to_owned())?,
            ring: ring.ok_or_else(|| "missing --ring".to_owned())?,
            control_socket: control_socket.ok_or_else(|| "missing --control-socket".to_owned())?,
            allocation_generation,
            first_epoch,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WorkerLoopExit {
    ControlClosed,
    ShutdownRequested,
}

fn send_event(stream: &mut UnixStream, event: &HarmonicExpertWorkerEvent) -> Result<(), String> {
    serde_json::to_writer(&mut *stream, &event)
        .map_err(|error| format!("serialize worker event: {error}"))?;
    stream
        .write_all(b"\n")
        .and_then(|_| stream.flush())
        .map_err(|error| format!("write worker event: {error}"))
}

fn open_ring(path: &Path) -> Result<HarmonicSharedRing, String> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|error| format!("open harmonic ring {}: {error}", path.display()))?;
    HarmonicSharedRing::open(&file)
        .map_err(|error| format!("map harmonic ring {}: {error}", path.display()))
}

fn run_loaded_worker(
    args: &Args,
    control: &mut UnixStream,
    ring: &mut HarmonicSharedRing,
    mut gpu: Gpu,
    mut hfq: HfqFile,
    model_sha256: &str,
    rocr_agent_name: &str,
) -> Result<(), String> {
    let mut cfg = DeepseekV4::config_from_hfq(&hfq)?;
    cfg.load_dspark = false;
    send_event(
        control,
        &HarmonicExpertWorkerEvent::Phase {
            phase: "load_routed".to_owned(),
        },
    )?;
    let mut weights = Some(DeepseekV4::load_weights_harmonic_experts_gfx1151(
        &mut hfq, &cfg, &mut gpu,
    )?);
    let mut service = None;
    let mut mapping = None;
    let result = (|| -> Result<WorkerLoopExit, String> {
        let receipt = weights.as_ref().unwrap().audit_local_owner(&mut gpu)?;
        if !receipt.pci_bus_id.eq_ignore_ascii_case(&args.pci_bus_id) {
            return Err(format!(
                "routed ownership BDF {} does not match requested {}",
                receipt.pci_bus_id, args.pci_bus_id
            ));
        }
        service = Some(DeepseekV4HarmonicExpertService::new(
            &mut gpu,
            &cfg,
            args.allocation_generation,
        )?);
        mapping = Some(
            HarmonicGpuMapping::register(ring, &gpu.hip)
                .map_err(|error| format!("register harmonic expert payloads: {error}"))?,
        );
        send_event(
            control,
            &HarmonicExpertWorkerEvent::Ready {
                model_sha256: model_sha256.to_owned(),
                architecture: receipt.architecture.clone(),
                pci_bus_id: receipt.pci_bus_id.clone(),
                hip_device_ordinal: receipt.hip_device_ordinal,
                rocr_agent_name: rocr_agent_name.to_owned(),
                allocation_generation: args.allocation_generation,
                routed_tensor_count: receipt.tensor_count,
                routed_bytes: receipt.bytes,
            },
        )?;

        hip_bridge::launch_counters::reset();
        let exit = worker_loop(
            control,
            ring,
            args.allocation_generation,
            args.first_epoch,
            &mut gpu,
            &cfg,
            weights.as_ref().unwrap(),
            service.as_mut().unwrap(),
            mapping.as_ref().unwrap(),
        )?;
        #[cfg(feature = "harmonic-stage-profile")]
        let stage_timing = service.as_ref().unwrap().stage_timing();
        #[allow(unused_mut)]
        let mut report = serde_json::json!({
            "harmonic_expert_owner_hip_calls": {
                "launch_count": hip_bridge::launch_counters::launch_kernel::count(),
                "launch_time_ns": hip_bridge::launch_counters::launch_kernel::time_ns(),
                "stream_sync_count": hip_bridge::launch_counters::stream_sync::count(),
                "stream_sync_time_ns": hip_bridge::launch_counters::stream_sync::time_ns(),
                "event_sync_count": hip_bridge::launch_counters::event_sync::count(),
                "event_sync_time_ns": hip_bridge::launch_counters::event_sync::time_ns(),
                "dtoh_count": hip_bridge::launch_counters::memcpy_dtoh::count(),
                "dtoh_time_ns": hip_bridge::launch_counters::memcpy_dtoh::time_ns(),
            },
        });
        #[cfg(feature = "harmonic-stage-profile")]
        {
            report["harmonic_expert_stage_device"] = serde_json::to_value(stage_timing)
                .map_err(|error| format!("serialize harmonic expert stage timing: {error}"))?;
        }
        eprintln!("{report}");
        Ok(exit)
    })();
    if let Some(service) = service.take() {
        service.release_quiesced(&mut gpu);
    }
    let mapping_cleanup = if let Some(mut mapping) = mapping.take() {
        mapping
            .unregister(&gpu.hip)
            .map_err(|error| format!("unregister harmonic expert payloads: {error}"))
    } else {
        Ok(())
    };
    if let Some(weights) = weights.take() {
        weights.free_gpu(&mut gpu);
    }
    gpu.invalidate_weight_caches();
    gpu.invalidate_graph_state();
    gpu.drain_pool();
    let result = match (result, mapping_cleanup) {
        (Ok(exit), Ok(())) => Ok(exit),
        (Err(run_error), Ok(())) => Err(run_error),
        (Ok(_), Err(cleanup_error)) => Err(cleanup_error),
        (Err(run_error), Err(cleanup_error)) => Err(format!("{run_error}; {cleanup_error}")),
    }?;
    match result {
        WorkerLoopExit::ControlClosed => Ok(()),
        WorkerLoopExit::ShutdownRequested => {
            send_event(control, &HarmonicExpertWorkerEvent::Shutdown)
        }
    }
}

fn worker_loop(
    control: &mut UnixStream,
    ring: &HarmonicSharedRing,
    allocation_generation: u64,
    first_epoch: u64,
    gpu: &mut Gpu,
    cfg: &hipfire_arch_deepseek4::DeepseekV4Config,
    weights: &hipfire_arch_deepseek4::DeepseekV4RoutedWeights,
    service: &mut DeepseekV4HarmonicExpertService,
    mapping: &HarmonicGpuMapping,
) -> Result<WorkerLoopExit, String> {
    let reader_stream = control
        .try_clone()
        .map_err(|error| format!("clone control socket: {error}"))?;
    reader_stream
        .set_nonblocking(true)
        .map_err(|error| format!("set control socket nonblocking: {error}"))?;
    let mut reader = BufReader::new(reader_stream);
    let mut epoch = first_epoch;
    let mut idle_since = Instant::now();
    loop {
        match ring
            .expert_poll_gpu_mailbox(epoch, allocation_generation)
            .map_err(|error| format!("poll epoch {epoch}: {error}"))?
        {
            HarmonicExpertMappedPoll::Work(packet) => {
                let now_tick = harmonic_monotonic_tick()
                    .map_err(|error| format!("clock epoch {epoch}: {error}"))?;
                let activation_payload = mapping.activation_buffer(epoch);
                let result_payload = mapping.result_buffer(epoch);
                let completion = service
                    .execute_mapped_packet(
                        gpu,
                        cfg,
                        weights,
                        &packet,
                        &activation_payload,
                        &result_payload,
                        now_tick,
                    )
                    .map_err(|error| format!("execute epoch {epoch}: {error}"))?;
                ring.expert_complete_gpu_mailbox(epoch, allocation_generation, completion)
                    .map_err(|error| format!("complete epoch {epoch}: {error}"))?;
                epoch = epoch
                    .checked_add(1)
                    .ok_or_else(|| "harmonic expert epoch exhausted".to_owned())?;
                idle_since = Instant::now();
            }
            HarmonicExpertMappedPoll::Terminal(_) => {
                ring.expert_acknowledge_terminal(epoch, allocation_generation)
                    .map_err(|error| format!("acknowledge epoch {epoch}: {error}"))?;
                epoch = epoch
                    .checked_add(1)
                    .ok_or_else(|| "harmonic expert epoch exhausted".to_owned())?;
                idle_since = Instant::now();
            }
            HarmonicExpertMappedPoll::Pending => {
                let idle = idle_since.elapsed();
                if idle < Duration::from_millis(2) {
                    std::hint::spin_loop();
                    continue;
                }
                if let Some(exit) = poll_control(&mut reader)? {
                    return Ok(exit);
                }
                if idle < Duration::from_millis(10) {
                    std::thread::yield_now();
                } else {
                    std::thread::sleep(Duration::from_micros(50));
                }
            }
        }
    }
}

fn poll_control(reader: &mut BufReader<UnixStream>) -> Result<Option<WorkerLoopExit>, String> {
    let mut line = String::new();
    let bytes = match reader.read_line(&mut line) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => return Ok(None),
        Err(error) => return Err(format!("read worker control command: {error}")),
    };
    if bytes == 0 {
        return Ok(Some(WorkerLoopExit::ControlClosed));
    }
    let command: HarmonicExpertWorkerCommand = serde_json::from_str(line.trim())
        .map_err(|error| format!("decode worker control command: {error}"))?;
    match command {
        HarmonicExpertWorkerCommand::Shutdown {} => Ok(Some(WorkerLoopExit::ShutdownRequested)),
    }
}

fn run(args: &Args, control: &mut UnixStream) -> Result<(), String> {
    let requested_bdf = args
        .pci_bus_id
        .parse::<PciBusId>()
        .map_err(|error| format!("parse --pci-bdf: {error}"))?;
    if !requested_bdf
        .to_string()
        .eq_ignore_ascii_case(&args.pci_bus_id)
    {
        return Err(format!(
            "--pci-bdf must be canonical; got {}, expected {}",
            args.pci_bus_id, requested_bdf
        ));
    }
    send_event(
        control,
        &HarmonicExpertWorkerEvent::Phase {
            phase: "verify_model".to_owned(),
        },
    )?;
    let artifact = DeepseekV4VerifiedArtifact::accept_parent_receipt(&DeepseekV4ArtifactReceipt {
        canonical_path: args.model.clone(),
        len: args.model_len,
        modified_unix_secs: args.model_mtime_secs,
        modified_subsec_nanos: args.model_mtime_nanos,
        sha256: args.model_sha256.clone(),
    })?;

    let mut ring = open_ring(&args.ring)?;
    let contract = ring.contract();
    if contract.destination_allocation_generation != args.allocation_generation {
        return Err(format!(
            "ring expert generation {} does not match worker generation {}",
            contract.destination_allocation_generation, args.allocation_generation
        ));
    }

    send_event(
        control,
        &HarmonicExpertWorkerEvent::Phase {
            phase: "bind_device".to_owned(),
        },
    )?;
    let symbols = redline_rocr::load_symbols()
        .map_err(|error| format!("load independent Redline ROCr symbols: {error}"))?;
    let rocr = Runtime::initialize(symbols)
        .map_err(|error| format!("initialize independent Redline ROCr runtime: {error}"))?;
    let rocr_device = rocr
        .select_gpu(GpuSelector::PciBusId(requested_bdf))
        .map_err(|error| format!("select Redline ROCr device {requested_bdf}: {error}"))?;
    HarmonicOwner::ExpertGfx1151
        .validate_arch(rocr_device.name())
        .map_err(|error| format!("Redline ROCr exact architecture: {error}"))?;

    let gpu = Gpu::init_with_pci_bus_id(&args.pci_bus_id, EXPECTED_ARCH)
        .map_err(|error| format!("initialize HIP device {}: {error}", args.pci_bus_id))?;
    let hip_bdf = gpu
        .pci_bus_id()
        .map_err(|error| format!("read HIP PCI identity: {error}"))?;
    if !hip_bdf.eq_ignore_ascii_case(&rocr_device.pci_bus_id().to_string()) {
        return Err(format!(
            "HIP/ROCr physical-device mismatch: HIP {hip_bdf}, ROCr {}",
            rocr_device.pci_bus_id()
        ));
    }

    let hfq = HfqFile::open(artifact.path()).map_err(|error| {
        format!(
            "open verified harmonic artifact {}: {error:?}",
            artifact.path().display()
        )
    })?;
    run_loaded_worker(
        args,
        control,
        &mut ring,
        gpu,
        hfq,
        &artifact.sha256,
        rocr_device.name(),
    )
}

fn main() {
    let args = match Args::parse() {
        Ok(args) => args,
        Err(error) => {
            eprintln!("deepseek4 harmonic expert worker: {error}");
            std::process::exit(2);
        }
    };
    let mut control = match UnixStream::connect(&args.control_socket) {
        Ok(stream) => stream,
        Err(error) => {
            eprintln!(
                "deepseek4 harmonic expert worker: connect {}: {error}",
                args.control_socket.display()
            );
            std::process::exit(1);
        }
    };
    if let Err(error) = run(&args, &mut control) {
        let _ = send_event(
            &mut control,
            &HarmonicExpertWorkerEvent::Fatal {
                error: error.clone(),
            },
        );
        eprintln!("deepseek4 harmonic expert worker: {error}");
        std::process::exit(1);
    }
}
