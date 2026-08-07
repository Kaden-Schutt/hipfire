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

use hipfire_arch_deepseek4::{
    DeepseekV4, DeepseekV4HarmonicExpertService, DeepseekV4VerifiedArtifact, HarmonicOwner,
    HarmonicSharedRing,
};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use redline_rocr::{GpuSelector, PciBusId, Runtime};
use serde::{Deserialize, Serialize};

const EXPECTED_ARCH: &str = "gfx1151";

#[derive(Debug)]
struct Args {
    model: PathBuf,
    pci_bus_id: String,
    ring: PathBuf,
    control_socket: PathBuf,
    allocation_generation: u64,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut model = None;
        let mut pci_bus_id = None;
        let mut ring = None;
        let mut control_socket = None;
        let mut allocation_generation = None;
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            let mut value = || {
                args.next()
                    .ok_or_else(|| format!("missing value after {flag}"))
            };
            match flag.as_str() {
                "--model" => model = Some(PathBuf::from(value()?)),
                "--pci-bdf" => pci_bus_id = Some(value()?),
                "--ring" => ring = Some(PathBuf::from(value()?)),
                "--control-socket" => control_socket = Some(PathBuf::from(value()?)),
                "--generation" => {
                    let raw = value()?;
                    allocation_generation = Some(raw.parse::<u64>().map_err(|error| {
                        format!("invalid nonzero --generation {raw:?}: {error}")
                    })?);
                }
                "--help" | "-h" => {
                    return Err(
                        "usage: deepseek4-harmonic-expert-worker --model PATH --pci-bdf 0000:BB:DD.F --ring PATH --control-socket PATH --generation N"
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
        Ok(Self {
            model: model.ok_or_else(|| "missing --model".to_owned())?,
            pci_bus_id: pci_bus_id.ok_or_else(|| "missing --pci-bdf".to_owned())?,
            ring: ring.ok_or_else(|| "missing --ring".to_owned())?,
            control_socket: control_socket.ok_or_else(|| "missing --control-socket".to_owned())?,
            allocation_generation,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum WorkerCommand {
    Execute { epoch: u64, now_tick: u64 },
    AcknowledgeTerminal { epoch: u64 },
    Shutdown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WorkerLoopExit {
    ControlClosed,
    ShutdownRequested,
}

#[derive(Debug, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum WorkerEvent<'a> {
    Phase {
        phase: &'a str,
    },
    Ready {
        model_sha256: &'a str,
        architecture: &'a str,
        pci_bus_id: &'a str,
        hip_device_ordinal: i32,
        rocr_agent_name: &'a str,
        allocation_generation: u64,
        routed_tensor_count: usize,
        routed_bytes: usize,
    },
    Completed {
        epoch: u64,
        result_fingerprint: u64,
    },
    Acknowledged {
        epoch: u64,
    },
    Shutdown,
    Fatal {
        error: &'a str,
    },
}

fn send_event(stream: &mut UnixStream, event: WorkerEvent<'_>) -> Result<(), String> {
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
    ring: &HarmonicSharedRing,
    mut gpu: Gpu,
    mut hfq: HfqFile,
    model_sha256: &str,
    rocr_agent_name: &str,
) -> Result<(), String> {
    let mut cfg = DeepseekV4::config_from_hfq(&hfq)?;
    cfg.load_dspark = false;
    send_event(
        control,
        WorkerEvent::Phase {
            phase: "load_routed",
        },
    )?;
    let mut weights = Some(DeepseekV4::load_weights_harmonic_experts_gfx1151(
        &mut hfq, &cfg, &mut gpu,
    )?);
    let mut service = None;
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
        send_event(
            control,
            WorkerEvent::Ready {
                model_sha256,
                architecture: &receipt.architecture,
                pci_bus_id: &receipt.pci_bus_id,
                hip_device_ordinal: receipt.hip_device_ordinal,
                rocr_agent_name,
                allocation_generation: args.allocation_generation,
                routed_tensor_count: receipt.tensor_count,
                routed_bytes: receipt.bytes,
            },
        )?;

        command_loop(
            control,
            ring,
            args.allocation_generation,
            &mut gpu,
            &cfg,
            weights.as_ref().unwrap(),
            service.as_mut().unwrap(),
        )
    })();
    if let Some(service) = service.take() {
        service.release_quiesced(&mut gpu);
    }
    if let Some(weights) = weights.take() {
        weights.free_gpu(&mut gpu);
    }
    gpu.invalidate_weight_caches();
    gpu.invalidate_graph_state();
    gpu.drain_pool();
    match result? {
        WorkerLoopExit::ControlClosed => Ok(()),
        WorkerLoopExit::ShutdownRequested => send_event(control, WorkerEvent::Shutdown),
    }
}

fn command_loop(
    control: &mut UnixStream,
    ring: &HarmonicSharedRing,
    allocation_generation: u64,
    gpu: &mut Gpu,
    cfg: &hipfire_arch_deepseek4::DeepseekV4Config,
    weights: &hipfire_arch_deepseek4::DeepseekV4RoutedWeights,
    service: &mut DeepseekV4HarmonicExpertService,
) -> Result<WorkerLoopExit, String> {
    let reader_stream = control
        .try_clone()
        .map_err(|error| format!("clone control socket: {error}"))?;
    let mut reader = BufReader::new(reader_stream);
    loop {
        let mut line = String::new();
        let bytes = reader
            .read_line(&mut line)
            .map_err(|error| format!("read worker command: {error}"))?;
        if bytes == 0 {
            return Ok(WorkerLoopExit::ControlClosed);
        }
        let command: WorkerCommand = serde_json::from_str(line.trim())
            .map_err(|error| format!("decode worker command: {error}"))?;
        match command {
            WorkerCommand::Execute { epoch, now_tick } => {
                let work = ring
                    .expert_begin(epoch, allocation_generation, now_tick)
                    .map_err(|error| format!("begin epoch {epoch}: {error}"))?;
                let completion = service
                    .execute_work_item(gpu, cfg, weights, &work, now_tick)
                    .map_err(|error| format!("execute epoch {epoch}: {error}"))?;
                ring.expert_complete(
                    epoch,
                    allocation_generation,
                    completion,
                    service.result_payload(),
                )
                .map_err(|error| format!("complete epoch {epoch}: {error}"))?;
                send_event(
                    control,
                    WorkerEvent::Completed {
                        epoch,
                        result_fingerprint: completion.result_fingerprint,
                    },
                )?;
            }
            WorkerCommand::AcknowledgeTerminal { epoch } => {
                ring.expert_acknowledge_terminal(epoch, allocation_generation)
                    .map_err(|error| format!("acknowledge epoch {epoch}: {error}"))?;
                send_event(control, WorkerEvent::Acknowledged { epoch })?;
            }
            WorkerCommand::Shutdown => return Ok(WorkerLoopExit::ShutdownRequested),
        }
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
        WorkerEvent::Phase {
            phase: "verify_model",
        },
    )?;
    let artifact = DeepseekV4VerifiedArtifact::verify(&args.model)?;

    let ring = open_ring(&args.ring)?;
    let contract = ring.contract();
    if contract.destination_allocation_generation != args.allocation_generation {
        return Err(format!(
            "ring expert generation {} does not match worker generation {}",
            contract.destination_allocation_generation, args.allocation_generation
        ));
    }

    send_event(
        control,
        WorkerEvent::Phase {
            phase: "bind_device",
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
        &ring,
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
        let _ = send_event(&mut control, WorkerEvent::Fatal { error: &error });
        eprintln!("deepseek4 harmonic expert worker: {error}");
        std::process::exit(1);
    }
}
