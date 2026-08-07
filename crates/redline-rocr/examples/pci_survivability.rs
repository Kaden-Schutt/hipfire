// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! Fail-closed ROCr queue survivability oracle for one explicit PCI device.
//!
//! The blocked worker publishes a barrier whose process-local dependency never
//! completes. The parent SIGKILLs that worker, then requires a fresh process to
//! open the same physical GPU and complete a bounded barrier dispatch. This is
//! deliberately not a product benchmark and never creates a peer mapping.

#[cfg(not(unix))]
compile_error!("pci_survivability requires Unix process semantics");

use std::env;
use std::error::Error;
use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::os::unix::process::ExitStatusExt;
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use redline_rocr::packet::{BarrierAndPacket, PacketImage};
use redline_rocr::{
    CompletionSignal, GpuDevice, GpuSelector, KernargPool, PciBusId, QueueSet, Runtime,
    load_symbols,
};

const AMD_VENDOR_ID: u16 = 0x1002;
const CHILD_TIMEOUT: Duration = Duration::from_secs(10);
const DISPATCH_TIMEOUT: Duration = Duration::from_secs(2);
const HELD_ALLOCATION_BYTES: usize = 64 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Orchestrate,
    Healthy,
    Blocked,
}

impl Mode {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "orchestrate" => Ok(Self::Orchestrate),
            "healthy" => Ok(Self::Healthy),
            "blocked" => Ok(Self::Blocked),
            _ => Err(invalid(format!("unknown --mode {value:?}"))),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Orchestrate => "orchestrate",
            Self::Healthy => "healthy",
            Self::Blocked => "blocked",
        }
    }
}

#[derive(Clone, Debug)]
struct Config {
    mode: Mode,
    pci_bus_id: PciBusId,
    expected_name: String,
    expected_device_id: u16,
    cycles: usize,
}

fn invalid(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
}

fn parse_hex_u16(value: &str, flag: &str) -> Result<u16, Box<dyn Error>> {
    let digits = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
        .unwrap_or(value);
    u16::from_str_radix(digits, 16)
        .map_err(|_| invalid(format!("{flag} expects a 16-bit hexadecimal value")))
}

fn next_value(
    arguments: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    arguments
        .next()
        .ok_or_else(|| invalid(format!("{flag} requires a value")))
}

fn parse_args() -> Result<Config, Box<dyn Error>> {
    let mut mode = Mode::Orchestrate;
    let mut pci_bus_id = None;
    let mut expected_name = None;
    let mut expected_device_id = None;
    let mut cycles = 3_usize;
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--mode" => mode = Mode::parse(&next_value(&mut arguments, &argument)?)?,
            "--pci-bus-id" => {
                pci_bus_id = Some(next_value(&mut arguments, &argument)?.parse::<PciBusId>()?);
            }
            "--expected-name" => expected_name = Some(next_value(&mut arguments, &argument)?),
            "--expected-device-id" => {
                let raw = next_value(&mut arguments, &argument)?;
                expected_device_id = Some(parse_hex_u16(&raw, "--expected-device-id")?);
            }
            "--cycles" => {
                cycles = next_value(&mut arguments, &argument)?.parse::<usize>()?;
                if !(1..=32).contains(&cycles) {
                    return Err(invalid("--cycles must be in 1..=32"));
                }
            }
            "--help" | "-h" => {
                println!(
                    "{}",
                    concat!(
                        "usage: pci_survivability --pci-bus-id dddd:bb:dd.f ",
                        "--expected-name gfxNNNN --expected-device-id hhhh [--cycles 1..32]"
                    )
                );
                std::process::exit(0);
            }
            _ => return Err(invalid(format!("unknown argument {argument:?}"))),
        }
    }
    Ok(Config {
        mode,
        pci_bus_id: pci_bus_id.ok_or_else(|| invalid("--pci-bus-id is required"))?,
        expected_name: expected_name.ok_or_else(|| invalid("--expected-name is required"))?,
        expected_device_id: expected_device_id
            .ok_or_else(|| invalid("--expected-device-id is required"))?,
        cycles,
    })
}

fn read_sysfs_hex(path: PathBuf) -> Result<u16, Box<dyn Error>> {
    let raw = fs::read_to_string(&path)?;
    parse_hex_u16(raw.trim(), &path.display().to_string())
}

fn preflight_physical_identity(config: &Config) -> Result<(), Box<dyn Error>> {
    let root = PathBuf::from("/sys/bus/pci/devices").join(config.pci_bus_id.to_string());
    let vendor = read_sysfs_hex(root.join("vendor"))?;
    let device = read_sysfs_hex(root.join("device"))?;
    if vendor != AMD_VENDOR_ID || device != config.expected_device_id {
        return Err(invalid(format!(
            "PCI {} identity mismatch: got vendor/device={vendor:04x}/{device:04x}, expected={AMD_VENDOR_ID:04x}/{:04x}",
            config.pci_bus_id, config.expected_device_id
        )));
    }
    let driver = fs::read_link(root.join("driver"))?;
    if driver.file_name().and_then(|name| name.to_str()) != Some("amdgpu") {
        return Err(invalid(format!(
            "PCI {} is not bound to amdgpu: {}",
            config.pci_bus_id,
            driver.display()
        )));
    }
    Ok(())
}

fn open_exact_device(config: &Config) -> Result<(Runtime, GpuDevice), Box<dyn Error>> {
    preflight_physical_identity(config)?;
    let runtime = Runtime::initialize(load_symbols()?)?;
    let device = runtime.select_gpu(GpuSelector::PciBusId(config.pci_bus_id))?;
    if !device.name().eq_ignore_ascii_case(&config.expected_name) {
        return Err(invalid(format!(
            "ROCr agent {} at {} does not match expected architecture {}",
            device.name(),
            device.pci_bus_id(),
            config.expected_name
        )));
    }
    Ok((runtime, device))
}

fn make_queue(device: &GpuDevice) -> Result<QueueSet, Box<dyn Error>> {
    let queue_size = *device.queue_size_range().start();
    Ok(QueueSet::create(device, 1, queue_size)?)
}

fn run_healthy(config: &Config) -> Result<(), Box<dyn Error>> {
    let (runtime, device) = open_exact_device(config)?;
    let pool = KernargPool::discover(&device)?;
    let allocation = pool.allocate_executable_bytes(HELD_ALLOCATION_BYTES)?;
    let completion = CompletionSignal::new(&device)?;
    let barrier = BarrierAndPacket::new(&[], completion.raw())?;
    let mut queues = make_queue(&device)?;
    queues.prepare_batches(&[vec![PacketImage::barrier(&barrier)]])?;
    queues.ring_prepared()?;
    queues.wait_signal(&completion, DISPATCH_TIMEOUT)?;
    queues.inactivate_all()?;
    println!(
        "PASS pci={} name={} queue={} allocation_bytes={}",
        device.pci_bus_id(),
        device.name(),
        queues.queue_ids().next().unwrap_or(0),
        HELD_ALLOCATION_BYTES
    );
    drop(queues);
    drop(completion);
    drop(allocation);
    drop(pool);
    drop(device);
    drop(runtime);
    Ok(())
}

fn run_blocked(config: &Config) -> Result<(), Box<dyn Error>> {
    let (_runtime, device) = open_exact_device(config)?;
    let pool = KernargPool::discover(&device)?;
    let _allocation = pool.allocate_executable_bytes(HELD_ALLOCATION_BYTES)?;
    let dependency = CompletionSignal::new(&device)?;
    let completion = CompletionSignal::new(&device)?;
    let barrier = BarrierAndPacket::new(&[dependency.raw()], completion.raw())?;
    let mut queues = make_queue(&device)?;
    queues.prepare_batches(&[vec![PacketImage::barrier(&barrier)]])?;
    queues.ring_prepared()?;
    println!(
        "READY pci={} name={} queue={} pid={}",
        device.pci_bus_id(),
        device.name(),
        queues.queue_ids().next().unwrap_or(0),
        std::process::id()
    );
    io::stdout().flush()?;
    loop {
        thread::park_timeout(Duration::from_secs(1));
    }
}

fn configure_child(command: &mut Command, config: &Config, mode: Mode) {
    command
        .arg("--mode")
        .arg(mode.as_str())
        .arg("--pci-bus-id")
        .arg(config.pci_bus_id.to_string())
        .arg("--expected-name")
        .arg(&config.expected_name)
        .arg("--expected-device-id")
        .arg(format!("{:04x}", config.expected_device_id))
        .arg("--cycles")
        .arg("1")
        .env_remove("ROCR_VISIBLE_DEVICES")
        .env_remove("HIP_VISIBLE_DEVICES")
        .env_remove("CUDA_VISIBLE_DEVICES")
        .env_remove("GPU_DEVICE_ORDINAL");
}

fn wait_bounded(child: &mut Child, timeout: Duration) -> Result<ExitStatus, Box<dyn Error>> {
    let started = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(status);
        }
        if started.elapsed() >= timeout {
            child.kill()?;
            let _ = child.wait();
            return Err(Box::new(io::Error::new(
                io::ErrorKind::TimedOut,
                format!("child {} exceeded {timeout:?}", child.id()),
            )));
        }
        thread::sleep(Duration::from_millis(10));
    }
}

fn run_orchestrator(config: &Config) -> Result<(), Box<dyn Error>> {
    preflight_physical_identity(config)?;
    let executable = env::current_exe()?;
    for cycle in 0..config.cycles {
        let mut blocked_command = Command::new(&executable);
        configure_child(&mut blocked_command, config, Mode::Blocked);
        blocked_command
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        let mut blocked = blocked_command.spawn()?;
        let stdout = blocked
            .stdout
            .take()
            .ok_or_else(|| invalid("blocked child stdout was not piped"))?;
        let (sender, receiver) = mpsc::sync_channel(1);
        thread::spawn(move || {
            let mut line = String::new();
            let result = BufReader::new(stdout).read_line(&mut line).map(|_| line);
            let _ = sender.send(result);
        });
        let ready = match receiver.recv_timeout(CHILD_TIMEOUT) {
            Ok(result) => result?,
            Err(error) => {
                let _ = blocked.kill();
                let _ = blocked.wait();
                return Err(Box::new(error));
            }
        };
        let expected_pci = format!("pci={}", config.pci_bus_id);
        let expected_name = format!("name={}", config.expected_name);
        if !ready.starts_with("READY ")
            || !ready.contains(&expected_pci)
            || !ready.contains(&expected_name)
        {
            let _ = blocked.kill();
            let _ = blocked.wait();
            return Err(invalid(format!(
                "invalid blocked-worker receipt: {ready:?}"
            )));
        }
        blocked.kill()?;
        let killed = wait_bounded(&mut blocked, Duration::from_secs(5))?;
        if killed.signal() != Some(9) {
            return Err(invalid(format!(
                "blocked worker exited without SIGKILL: {killed}"
            )));
        }

        let mut healthy_command = Command::new(&executable);
        configure_child(&mut healthy_command, config, Mode::Healthy);
        healthy_command
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
        let mut healthy = healthy_command.spawn()?;
        let recovered = wait_bounded(&mut healthy, CHILD_TIMEOUT)?;
        if !recovered.success() {
            return Err(invalid(format!(
                "fresh recovery worker failed in cycle {cycle}: {recovered}"
            )));
        }
    }
    println!(
        "{{\"status\":\"pass\",\"pci_bus_id\":\"{}\",\"expected_name\":\"{}\",\"cycles\":{},\"blocked_packet\":\"process_local_barrier\",\"peer_access\":false,\"product_gpu_queues\":0}}",
        config.pci_bus_id, config.expected_name, config.cycles
    );
    Ok(())
}

fn main() {
    let result = parse_args().and_then(|config| match config.mode {
        Mode::Orchestrate => run_orchestrator(&config),
        Mode::Healthy => run_healthy(&config),
        Mode::Blocked => run_blocked(&config),
    });
    if let Err(error) = result {
        eprintln!("pci_survivability: {error}");
        std::process::exit(1);
    }
}
