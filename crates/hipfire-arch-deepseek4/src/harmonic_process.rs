// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! Host-side lifecycle authority for one harmonic gfx1151 expert worker.
//!
//! The supervisor owns an exact child PID and a private Unix-domain control
//! socket. The socket is deliberately a cold control plane: startup, shutdown,
//! and fatal diagnostics only. Steady-state expert work travels through the
//! shared ring and never pays a JSON or process-control round trip. A failed,
//! malformed, or timed-out worker is killed and reaped before its shared-ring
//! owner is isolated. No command here can wait on, reset, or address either
//! GPU.

#[cfg(not(unix))]
compile_error!("deepseek4 harmonic worker supervision currently requires Unix");

use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::harmonic::HarmonicOwner;
use crate::harmonic_ipc::HarmonicSharedRing;
use crate::heterogeneous::MQ2R_0731_SHA256;

const CONTROL_LINE_LIMIT: usize = 16 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum HarmonicExpertWorkerCommand {
    Shutdown {},
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "event", rename_all = "snake_case", deny_unknown_fields)]
pub enum HarmonicExpertWorkerEvent {
    Phase {
        phase: String,
    },
    Ready {
        model_sha256: String,
        architecture: String,
        pci_bus_id: String,
        hip_device_ordinal: i32,
        rocr_agent_name: String,
        allocation_generation: u64,
        routed_tensor_count: usize,
        routed_bytes: usize,
    },
    Shutdown,
    Fatal {
        error: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HarmonicExpertWorkerReady {
    pub model_sha256: String,
    pub architecture: String,
    pub pci_bus_id: String,
    pub hip_device_ordinal: i32,
    pub rocr_agent_name: String,
    pub allocation_generation: u64,
    pub routed_tensor_count: usize,
    pub routed_bytes: usize,
    pub startup_phases: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct HarmonicExpertWorkerSpec {
    pub executable: PathBuf,
    pub model: PathBuf,
    pub pci_bus_id: String,
    pub ring: PathBuf,
    pub control_socket: PathBuf,
    pub allocation_generation: u64,
    /// First ring epoch consumed by this worker generation. Work after this
    /// point is discovered directly from the ring, not commanded over the
    /// control socket.
    pub first_epoch: u64,
    pub startup_timeout: Duration,
    pub control_timeout: Duration,
    pub exit_timeout: Duration,
}

impl HarmonicExpertWorkerSpec {
    fn validate(&self) -> Result<(), String> {
        if !self.executable.is_file() {
            return Err(format!(
                "harmonic expert executable is not a file: {}",
                self.executable.display()
            ));
        }
        if !self.model.is_file() {
            return Err(format!(
                "harmonic expert model is not a file: {}",
                self.model.display()
            ));
        }
        if !self.ring.is_file() {
            return Err(format!(
                "harmonic expert ring is not a file: {}",
                self.ring.display()
            ));
        }
        if self.control_socket.exists() {
            return Err(format!(
                "harmonic expert control socket already exists: {}",
                self.control_socket.display()
            ));
        }
        let canonical = self
            .pci_bus_id
            .parse::<redline_rocr::PciBusId>()
            .map_err(|error| format!("harmonic expert PCI BDF: {error}"))?
            .to_string();
        if !canonical.eq_ignore_ascii_case(&self.pci_bus_id) {
            return Err(format!(
                "harmonic expert PCI BDF must be canonical: got {}, expected {canonical}",
                self.pci_bus_id
            ));
        }
        if self.allocation_generation == 0 {
            return Err("harmonic expert allocation generation must be nonzero".to_owned());
        }
        if self.first_epoch == 0 {
            return Err("harmonic expert first epoch must be nonzero".to_owned());
        }
        if self.startup_timeout.is_zero()
            || self.control_timeout.is_zero()
            || self.exit_timeout.is_zero()
        {
            return Err("harmonic expert process timeouts must be nonzero".to_owned());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HarmonicWorkerIsolationReceipt {
    pub pid: u32,
    pub exit_status: String,
    pub isolated_slots: usize,
    pub cause: String,
}

pub struct HarmonicExpertWorkerProcess {
    child: Option<Child>,
    listener: Option<UnixListener>,
    reader: BufReader<UnixStream>,
    writer: UnixStream,
    isolation_ring: Option<HarmonicSharedRing>,
    control_socket: PathBuf,
    allocation_generation: u64,
    control_timeout: Duration,
    exit_timeout: Duration,
    ready: HarmonicExpertWorkerReady,
}

impl HarmonicExpertWorkerProcess {
    pub fn spawn(spec: HarmonicExpertWorkerSpec) -> Result<Self, String> {
        spec.validate()?;
        let ring_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&spec.ring)
            .map_err(|error| format!("open harmonic isolation ring: {error}"))?;
        let isolation_ring = HarmonicSharedRing::open(&ring_file)
            .map_err(|error| format!("map harmonic isolation ring: {error}"))?;
        let ring_generation = isolation_ring.contract().destination_allocation_generation;
        if ring_generation != spec.allocation_generation {
            return Err(format!(
                "harmonic ring expert generation {ring_generation} does not match process generation {}",
                spec.allocation_generation
            ));
        }

        let listener = UnixListener::bind(&spec.control_socket).map_err(|error| {
            format!(
                "bind harmonic control socket {}: {error}",
                spec.control_socket.display()
            )
        })?;
        if let Err(error) = listener.set_nonblocking(true) {
            drop(listener);
            let _ = fs::remove_file(&spec.control_socket);
            return Err(format!("set harmonic listener nonblocking: {error}"));
        }
        let child = Command::new(&spec.executable)
            .arg("--model")
            .arg(&spec.model)
            .arg("--pci-bdf")
            .arg(&spec.pci_bus_id)
            .arg("--ring")
            .arg(&spec.ring)
            .arg("--control-socket")
            .arg(&spec.control_socket)
            .arg("--generation")
            .arg(spec.allocation_generation.to_string())
            .arg("--first-epoch")
            .arg(spec.first_epoch.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| {
                let _ = fs::remove_file(&spec.control_socket);
                format!("spawn harmonic expert worker: {error}")
            })?;

        let deadline = Instant::now() + spec.startup_timeout;
        let mut pending_child = Some(child);
        let accepted = loop {
            match listener.accept() {
                Ok((stream, _)) => break Ok(stream),
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    match pending_child.as_mut().unwrap().try_wait() {
                        Ok(Some(status)) => {
                            break Err(format!(
                                "harmonic expert worker exited before connect: {status}"
                            ));
                        }
                        Ok(None) => {}
                        Err(error) => {
                            break Err(format!("poll harmonic expert startup: {error}"));
                        }
                    }
                    if Instant::now() >= deadline {
                        break Err("harmonic expert worker connect timed out".to_owned());
                    }
                    thread::sleep(Duration::from_millis(1));
                }
                Err(error) => break Err(format!("accept harmonic expert worker: {error}")),
            }
        };
        let stream = match accepted {
            Ok(stream) => stream,
            Err(cause) => {
                return Err(cleanup_failed_spawn(
                    pending_child.take().unwrap(),
                    listener,
                    isolation_ring,
                    &spec.control_socket,
                    spec.allocation_generation,
                    spec.exit_timeout,
                    cause,
                ));
            }
        };
        if let Err(error) = stream.set_write_timeout(Some(spec.startup_timeout)) {
            drop(stream);
            return Err(cleanup_failed_spawn(
                pending_child.take().unwrap(),
                listener,
                isolation_ring,
                &spec.control_socket,
                spec.allocation_generation,
                spec.exit_timeout,
                format!("set harmonic startup write timeout: {error}"),
            ));
        }
        if let Err(error) = stream.set_read_timeout(Some(spec.startup_timeout)) {
            drop(stream);
            return Err(cleanup_failed_spawn(
                pending_child.take().unwrap(),
                listener,
                isolation_ring,
                &spec.control_socket,
                spec.allocation_generation,
                spec.exit_timeout,
                format!("set harmonic startup read timeout: {error}"),
            ));
        }
        let writer = match stream.try_clone() {
            Ok(writer) => writer,
            Err(error) => {
                drop(stream);
                return Err(cleanup_failed_spawn(
                    pending_child.take().unwrap(),
                    listener,
                    isolation_ring,
                    &spec.control_socket,
                    spec.allocation_generation,
                    spec.exit_timeout,
                    format!("clone harmonic control stream: {error}"),
                ));
            }
        };
        let mut process = Self {
            child: pending_child,
            listener: Some(listener),
            reader: BufReader::new(stream),
            writer,
            isolation_ring: Some(isolation_ring),
            control_socket: spec.control_socket,
            allocation_generation: spec.allocation_generation,
            control_timeout: spec.control_timeout,
            exit_timeout: spec.exit_timeout,
            ready: HarmonicExpertWorkerReady {
                model_sha256: String::new(),
                architecture: String::new(),
                pci_bus_id: String::new(),
                hip_device_ordinal: -1,
                rocr_agent_name: String::new(),
                allocation_generation: 0,
                routed_tensor_count: 0,
                routed_bytes: 0,
                startup_phases: Vec::new(),
            },
        };
        if let Err(cause) = process.wait_ready(deadline, &spec.pci_bus_id) {
            let isolation = process.terminate_and_isolate(cause.clone());
            return Err(match isolation {
                Ok(receipt) => format!(
                    "{cause}; exact child pid={} exit={} isolated_slots={}",
                    receipt.pid, receipt.exit_status, receipt.isolated_slots
                ),
                Err(error) => format!("{cause}; worker isolation failed: {error}"),
            });
        }
        if let Err(error) = process
            .reader
            .get_ref()
            .set_read_timeout(Some(process.control_timeout))
        {
            let cause = format!("set harmonic control read timeout: {error}");
            let isolation = process.terminate_and_isolate(cause.clone());
            return Err(format_isolation_failure(cause, isolation));
        }
        if let Err(error) = process
            .writer
            .set_write_timeout(Some(process.control_timeout))
        {
            let cause = format!("set harmonic control write timeout: {error}");
            let isolation = process.terminate_and_isolate(cause.clone());
            return Err(format_isolation_failure(cause, isolation));
        }
        Ok(process)
    }

    pub fn ready(&self) -> &HarmonicExpertWorkerReady {
        &self.ready
    }

    pub fn pid(&self) -> Option<u32> {
        self.child.as_ref().map(Child::id)
    }

    pub fn shutdown_and_isolate(mut self) -> Result<HarmonicWorkerIsolationReceipt, String> {
        self.reader
            .get_ref()
            .set_read_timeout(Some(self.exit_timeout))
            .map_err(|error| format!("set harmonic shutdown read timeout: {error}"))?;
        self.writer
            .set_write_timeout(Some(self.exit_timeout))
            .map_err(|error| format!("set harmonic shutdown write timeout: {error}"))?;
        let graceful = self
            .send(&HarmonicExpertWorkerCommand::Shutdown {})
            .and_then(|_| match self.read_event()? {
                HarmonicExpertWorkerEvent::Shutdown => Ok(()),
                HarmonicExpertWorkerEvent::Fatal { error } => Err(format!(
                    "harmonic expert worker fatal during shutdown: {error}"
                )),
                event => Err(format!(
                    "harmonic expert worker unexpected shutdown response: {event:?}"
                )),
            });
        let cause = graceful
            .err()
            .unwrap_or_else(|| "graceful harmonic expert shutdown".to_owned());
        self.terminate_and_isolate(cause)
    }

    pub fn terminate_and_isolate(
        &mut self,
        cause: impl Into<String>,
    ) -> Result<HarmonicWorkerIsolationReceipt, String> {
        let cause = cause.into();
        let child = self
            .child
            .as_mut()
            .ok_or_else(|| "harmonic expert child already reaped".to_owned())?;
        let pid = child.id();
        let status = terminate_child(child, self.exit_timeout)?;
        // Keep ownership until exit is confirmed. If termination failed, Drop
        // still owns the exact PID and can finish reaping it asynchronously.
        self.child.take();
        let isolated_slots = self
            .isolation_ring
            .as_ref()
            .ok_or_else(|| "harmonic expert isolation ring missing".to_owned())?
            .isolate_owner(HarmonicOwner::ExpertGfx1151, self.allocation_generation)
            .map_err(|error| format!("isolate confirmed harmonic expert exit: {error}"))?;
        self.cleanup_socket();
        Ok(HarmonicWorkerIsolationReceipt {
            pid,
            exit_status: status.to_string(),
            isolated_slots,
            cause,
        })
    }

    fn wait_ready(&mut self, deadline: Instant, expected_pci: &str) -> Result<(), String> {
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err("harmonic expert ready receipt timed out".to_owned());
            }
            self.reader
                .get_ref()
                .set_read_timeout(Some(remaining))
                .map_err(|error| format!("set harmonic ready timeout: {error}"))?;
            match self.read_event()? {
                HarmonicExpertWorkerEvent::Phase { phase } => {
                    self.ready.startup_phases.push(phase);
                }
                HarmonicExpertWorkerEvent::Ready {
                    model_sha256,
                    architecture,
                    pci_bus_id,
                    hip_device_ordinal,
                    rocr_agent_name,
                    allocation_generation,
                    routed_tensor_count,
                    routed_bytes,
                } => {
                    if model_sha256 != MQ2R_0731_SHA256
                        || !architecture.eq_ignore_ascii_case("gfx1151")
                        || !rocr_agent_name.eq_ignore_ascii_case("gfx1151")
                        || !pci_bus_id.eq_ignore_ascii_case(expected_pci)
                        || allocation_generation != self.allocation_generation
                        || routed_tensor_count == 0
                        || routed_bytes == 0
                    {
                        return Err(format!(
                            "harmonic expert invalid ready receipt: sha={model_sha256} arch={architecture} pci={pci_bus_id} rocr={rocr_agent_name} generation={allocation_generation} tensors={routed_tensor_count} bytes={routed_bytes}"
                        ));
                    }
                    self.ready.model_sha256 = model_sha256;
                    self.ready.architecture = architecture;
                    self.ready.pci_bus_id = pci_bus_id;
                    self.ready.hip_device_ordinal = hip_device_ordinal;
                    self.ready.rocr_agent_name = rocr_agent_name;
                    self.ready.allocation_generation = allocation_generation;
                    self.ready.routed_tensor_count = routed_tensor_count;
                    self.ready.routed_bytes = routed_bytes;
                    return Ok(());
                }
                HarmonicExpertWorkerEvent::Fatal { error } => {
                    return Err(format!("harmonic expert worker startup fatal: {error}"));
                }
                event => {
                    return Err(format!(
                        "harmonic expert worker unexpected startup event: {event:?}"
                    ));
                }
            }
        }
    }

    fn send(&mut self, command: &HarmonicExpertWorkerCommand) -> Result<(), String> {
        serde_json::to_writer(&mut self.writer, command)
            .map_err(|error| format!("serialize harmonic worker command: {error}"))?;
        self.writer
            .write_all(b"\n")
            .and_then(|_| self.writer.flush())
            .map_err(|error| format!("write harmonic worker command: {error}"))
    }

    fn read_event(&mut self) -> Result<HarmonicExpertWorkerEvent, String> {
        let mut line = String::new();
        let bytes = self
            .reader
            .read_line(&mut line)
            .map_err(|error| format!("read harmonic worker event: {error}"))?;
        if bytes == 0 {
            return Err("harmonic expert worker closed control socket".to_owned());
        }
        if bytes > CONTROL_LINE_LIMIT {
            return Err(format!(
                "harmonic expert worker event exceeded {CONTROL_LINE_LIMIT} bytes"
            ));
        }
        serde_json::from_str(line.trim())
            .map_err(|error| format!("decode harmonic worker event: {error}"))
    }

    fn cleanup_socket(&mut self) {
        self.listener.take();
        let _ = fs::remove_file(&self.control_socket);
    }
}

fn format_isolation_failure(
    cause: String,
    isolation: Result<HarmonicWorkerIsolationReceipt, String>,
) -> String {
    match isolation {
        Ok(receipt) => format!(
            "{cause}; exact child pid={} exit={} isolated_slots={}",
            receipt.pid, receipt.exit_status, receipt.isolated_slots
        ),
        Err(error) => format!("{cause}; worker isolation failed: {error}"),
    }
}

impl Drop for HarmonicExpertWorkerProcess {
    fn drop(&mut self) {
        let Some(mut child) = self.child.take() else {
            self.cleanup_socket();
            return;
        };
        let _ = child.kill();
        let pid = child.id();
        let generation = self.allocation_generation;
        let ring = self.isolation_ring.take();
        self.cleanup_socket();
        let spawn = thread::Builder::new()
            .name(format!("ds4-harmonic-reap-{pid}"))
            .spawn(move || match child.wait() {
                Ok(_) => {
                    if let Some(ring) = ring {
                        let _ = ring.isolate_owner(HarmonicOwner::ExpertGfx1151, generation);
                    }
                }
                Err(error) => eprintln!(
                    "deepseek4 harmonic supervisor: failed to confirm dropped child {pid}: {error}"
                ),
            });
        if let Err(error) = spawn {
            eprintln!(
                "deepseek4 harmonic supervisor: failed to start nonblocking reaper for {pid}: {error}"
            );
        }
    }
}

fn cleanup_failed_spawn(
    mut child: Child,
    listener: UnixListener,
    ring: HarmonicSharedRing,
    control_socket: &Path,
    allocation_generation: u64,
    exit_timeout: Duration,
    cause: String,
) -> String {
    let pid = child.id();
    drop(listener);
    let detail = match terminate_child(&mut child, exit_timeout) {
        Ok(status) => match ring.isolate_owner(HarmonicOwner::ExpertGfx1151, allocation_generation)
        {
            Ok(slots) => format!("exit={status}; isolated_slots={slots}"),
            Err(error) => format!("exit={status}; isolation_failed={error}"),
        },
        Err(error) => format!("exit_unconfirmed={error}; isolation_withheld"),
    };
    let _ = fs::remove_file(control_socket);
    format!("{cause}; exact child pid={pid}; {detail}")
}

fn terminate_child(child: &mut Child, timeout: Duration) -> Result<ExitStatus, String> {
    if let Some(status) = child
        .try_wait()
        .map_err(|error| format!("poll harmonic expert child: {error}"))?
    {
        return Ok(status);
    }
    if let Err(kill_error) = child.kill() {
        if let Some(status) = child
            .try_wait()
            .map_err(|error| format!("poll harmonic expert child after kill error: {error}"))?
        {
            return Ok(status);
        }
        return Err(format!(
            "kill exact harmonic expert child {}: {kill_error}",
            child.id()
        ));
    }
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|error| format!("reap harmonic expert child: {error}"))?
        {
            return Ok(status);
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "exact harmonic expert child {} did not exit within {timeout:?}",
                child.id()
            ));
        }
        thread::sleep(Duration::from_millis(1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_protocol_round_trips_and_rejects_unknown_fields() {
        let command = HarmonicExpertWorkerCommand::Shutdown {};
        let encoded = serde_json::to_string(&command).unwrap();
        assert_eq!(
            serde_json::from_str::<HarmonicExpertWorkerCommand>(&encoded).unwrap(),
            command
        );
        assert!(serde_json::from_str::<HarmonicExpertWorkerCommand>(
            r#"{"op":"shutdown","device":0}"#
        )
        .is_err());
    }

    #[test]
    fn control_protocol_cannot_command_steady_state_work() {
        let source = include_str!("harmonic_process.rs");
        for forbidden in [
            ["Exec", "ute {"].concat(),
            ["Compl", "eted {"].concat(),
            ["Acknowledge", "Terminal"].concat(),
            ["execute_or_", "isolate"].concat(),
        ] {
            assert!(!source.contains(&forbidden), "found {forbidden}");
        }
        let worker = include_str!("bin/deepseek4_harmonic_expert_worker.rs");
        assert!(worker.contains("expert_poll"));
    }

    #[test]
    fn supervisor_source_has_no_broad_process_or_gpu_recovery() {
        let source = include_str!("harmonic_process.rs");
        for forbidden in [
            ["p", "kill"].concat(),
            ["device", "_reset"].concat(),
            ["enable_peer", "_access"].concat(),
            ["memcpy_", "peer"].concat(),
            ["stream_wait_", "value32"].concat(),
            ["stream_write_", "value32"].concat(),
        ] {
            assert!(!source.contains(&forbidden), "found {forbidden}");
        }
        assert!(source.contains("child.kill()"));
        assert!(source.contains("isolate_owner"));
    }
}
