// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! Cross-process, cross-GPU ROCr IPC-signal oracle.
//!
//! The consumer queues one live one-way dependency before the producer GPU
//! decrements the shared IPC signal. A finite host supervisor inactivates the
//! consumer queue and kills the producer on timeout. No peer memory mapping or
//! reciprocal GPU wait is created.

#[cfg(not(unix))]
compile_error!("ipc_signal_cross_gpu requires Unix process semantics");

use std::env;
use std::error::Error;
use std::io::{self, BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use redline_rocr::packet::{BarrierAndPacket, PacketImage};
use redline_rocr::{
    CompletionSignal, GpuDevice, GpuSelector, IpcSignalHandle, PciBusId, QueueSet, Runtime,
    load_symbols,
};

const STEP_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Orchestrate,
    Producer,
}

fn invalid(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
}

fn next_value(
    arguments: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    arguments
        .next()
        .ok_or_else(|| invalid(format!("{flag} requires a value")))
}

#[derive(Debug)]
struct Config {
    mode: Mode,
    consumer_name: String,
    producer_name: String,
    producer_pci: Option<PciBusId>,
    handle: Option<IpcSignalHandle>,
    cycles: usize,
}

fn parse_handle(value: &str) -> Result<IpcSignalHandle, Box<dyn Error>> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(invalid("--handle requires exactly 64 hexadecimal digits"));
    }
    let mut words = [0_u32; 8];
    for (index, word) in words.iter_mut().enumerate() {
        let offset = index * 8;
        *word = u32::from_str_radix(&value[offset..offset + 8], 16)?;
    }
    Ok(IpcSignalHandle { words })
}

fn format_handle(handle: IpcSignalHandle) -> String {
    handle
        .words
        .iter()
        .map(|word| format!("{word:08x}"))
        .collect::<Vec<_>>()
        .join("")
}

fn parse_args() -> Result<Config, Box<dyn Error>> {
    let mut mode = Mode::Orchestrate;
    let mut consumer_name = "gfx1100".to_owned();
    let mut producer_name = "gfx1151".to_owned();
    let mut producer_pci = None;
    let mut handle = None;
    let mut cycles = 32_usize;
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--mode" => {
                mode = match next_value(&mut arguments, &argument)?.as_str() {
                    "orchestrate" => Mode::Orchestrate,
                    "producer" => Mode::Producer,
                    value => return Err(invalid(format!("unknown --mode {value:?}"))),
                }
            }
            "--consumer-name" => consumer_name = next_value(&mut arguments, &argument)?,
            "--producer-name" => producer_name = next_value(&mut arguments, &argument)?,
            "--producer-pci" => {
                producer_pci = Some(next_value(&mut arguments, &argument)?.parse()?)
            }
            "--handle" => handle = Some(parse_handle(&next_value(&mut arguments, &argument)?)?),
            "--cycles" => {
                cycles = next_value(&mut arguments, &argument)?.parse()?;
                if !(1..=4096).contains(&cycles) {
                    return Err(invalid("--cycles must be in 1..=4096"));
                }
            }
            "--help" | "-h" => {
                println!(
                    "usage: ipc_signal_cross_gpu [--consumer-name gfx1100] [--producer-name gfx1151] [--cycles N]"
                );
                std::process::exit(0);
            }
            value => return Err(invalid(format!("unknown argument {value:?}"))),
        }
    }
    Ok(Config {
        mode,
        consumer_name,
        producer_name,
        producer_pci,
        handle,
        cycles,
    })
}

fn exact_device(runtime: &Runtime, name: &str) -> Result<GpuDevice, Box<dyn Error>> {
    let mut matching = runtime
        .gpu_devices()?
        .into_iter()
        .filter(|device| device.name().eq_ignore_ascii_case(name));
    let device = matching
        .next()
        .ok_or_else(|| invalid(format!("no exact ROCr device named {name}")))?;
    if matching.next().is_some() {
        return Err(invalid(format!(
            "more than one exact ROCr device named {name}; resolve a physical identity first"
        )));
    }
    Ok(device)
}

fn make_queue(device: &GpuDevice) -> Result<QueueSet, Box<dyn Error>> {
    Ok(QueueSet::create(
        device,
        1,
        *device.queue_size_range().start(),
    )?)
}

fn producer(config: &Config) -> Result<(), Box<dyn Error>> {
    let runtime = Runtime::initialize(load_symbols()?)?;
    let pci = config
        .producer_pci
        .ok_or_else(|| invalid("producer requires --producer-pci"))?;
    let device = runtime.select_gpu(GpuSelector::PciBusId(pci))?;
    if !device.name().eq_ignore_ascii_case(&config.producer_name) {
        return Err(invalid(format!(
            "producer PCI {pci} resolved to {}, expected {}",
            device.name(),
            config.producer_name
        )));
    }
    let shared = CompletionSignal::attach_ipc(
        &device,
        &config
            .handle
            .ok_or_else(|| invalid("producer requires --handle"))?,
    )?;
    let packet = BarrierAndPacket::new(&[], shared.raw())?;
    let mut queues = make_queue(&device)?;
    println!("READY pci={pci} arch={}", device.name());
    io::stdout().flush()?;
    for line in io::stdin().lock().lines() {
        let line = line?;
        if line == "STOP" {
            queues.inactivate_all()?;
            println!("STOPPED");
            io::stdout().flush()?;
            return Ok(());
        }
        let cycle = line
            .strip_prefix("GO ")
            .ok_or_else(|| invalid(format!("unexpected producer command {line:?}")))?;
        queues.prepare_batches(&[vec![PacketImage::barrier(&packet)]])?;
        queues.ring_prepared()?;
        queues.wait_signal(&shared, STEP_TIMEOUT)?;
        println!("DONE {cycle}");
        io::stdout().flush()?;
    }
    Err(invalid("producer control pipe closed before STOP"))
}

fn read_lines(child: &mut Child) -> Result<mpsc::Receiver<String>, Box<dyn Error>> {
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| invalid("producer stdout was not piped"))?;
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else {
                break;
            };
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    Ok(rx)
}

fn abort(
    queues: &mut QueueSet,
    child: &mut Child,
    message: impl Into<String>,
) -> Result<(), Box<dyn Error>> {
    let message = message.into();
    let queue_result = queues.inactivate_all();
    let _ = child.kill();
    let _ = child.wait();
    if let Err(error) = queue_result {
        return Err(invalid(format!(
            "{message}; consumer queue inactivation also failed: {error}"
        )));
    }
    Err(invalid(message))
}

fn orchestrate(config: &Config) -> Result<(), Box<dyn Error>> {
    let runtime = Runtime::initialize(load_symbols()?)?;
    let consumer = exact_device(&runtime, &config.consumer_name)?;
    let producer = exact_device(&runtime, &config.producer_name)?;
    if consumer.pci_bus_id() == producer.pci_bus_id() {
        return Err(invalid("producer and consumer resolved to the same PCI device"));
    }
    let mut shared = CompletionSignal::new_ipc(&consumer)?;
    let handle = shared.export_ipc()?;
    let mut child = Command::new(env::current_exe()?)
        .arg("--mode")
        .arg("producer")
        .arg("--consumer-name")
        .arg(&config.consumer_name)
        .arg("--producer-name")
        .arg(&config.producer_name)
        .arg("--producer-pci")
        .arg(producer.pci_bus_id().to_string())
        .arg("--handle")
        .arg(format_handle(handle))
        .arg("--cycles")
        .arg(config.cycles.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let lines = read_lines(&mut child)?;
    let ready = lines
        .recv_timeout(STEP_TIMEOUT)
        .map_err(|_| invalid("producer did not attach the IPC signal before timeout"))?;
    if !ready.starts_with("READY ") {
        let _ = child.kill();
        let _ = child.wait();
        return Err(invalid(format!("unexpected producer startup {ready:?}")));
    }
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| invalid("producer stdin was not piped"))?;
    let mut terminal = CompletionSignal::new(&consumer)?;
    let dependency = BarrierAndPacket::new(&[shared.raw()], terminal.raw())?;
    let mut queues = make_queue(&consumer)?;
    let mut latencies = Vec::with_capacity(config.cycles);
    for cycle in 0..config.cycles {
        shared.reset();
        terminal.reset();
        queues.prepare_batches(&[vec![PacketImage::barrier(&dependency)]])?;
        queues.ring_prepared()?;
        let started = Instant::now();
        writeln!(stdin, "GO {cycle}")?;
        stdin.flush()?;
        if let Err(error) = queues.wait_signal(&terminal, STEP_TIMEOUT) {
            return abort(
                &mut queues,
                &mut child,
                format!("consumer dependency failed at cycle {cycle}: {error}"),
            );
        }
        let done = match lines.recv_timeout(STEP_TIMEOUT) {
            Ok(done) => done,
            Err(_) => {
                return abort(
                    &mut queues,
                    &mut child,
                    format!("producer completion receipt timed out at cycle {cycle}"),
                );
            }
        };
        if done != format!("DONE {cycle}") {
            return abort(
                &mut queues,
                &mut child,
                format!("unexpected producer receipt {done:?}"),
            );
        }
        latencies.push(started.elapsed().as_secs_f64() * 1_000_000.0);
    }
    writeln!(stdin, "STOP")?;
    stdin.flush()?;
    let stopped = lines
        .recv_timeout(STEP_TIMEOUT)
        .map_err(|_| invalid("producer did not acknowledge STOP"))?;
    if stopped != "STOPPED" {
        return abort(
            &mut queues,
            &mut child,
            format!("unexpected producer shutdown {stopped:?}"),
        );
    }
    let status = child.wait()?;
    if !status.success() {
        return Err(invalid(format!("producer exited with {status}")));
    }
    queues.inactivate_all()?;
    latencies.sort_by(f64::total_cmp);
    let median = latencies[latencies.len() / 2];
    let p95 = latencies[(latencies.len() * 95 / 100).min(latencies.len() - 1)];
    println!(
        "PASS consumer={}@{} producer={}@{} cycles={} median_us={median:.3} p95_us={p95:.3} max_us={:.3}",
        consumer.name(),
        consumer.pci_bus_id(),
        producer.name(),
        producer.pci_bus_id(),
        config.cycles,
        latencies[latencies.len() - 1]
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args()?;
    match config.mode {
        Mode::Orchestrate => orchestrate(&config),
        Mode::Producer => producer(&config),
    }
}
