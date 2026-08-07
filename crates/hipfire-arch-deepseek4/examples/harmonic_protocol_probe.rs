// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-only process-isolation probe for the DS4 harmonic transport.
//!
//! This executable never initializes HIP. The controller owns no worker GPU
//! state; each future device owner is represented by a distinct long-lived
//! process mapping the persistent packet ring. All RPC reads are bounded.

#[cfg(not(unix))]
fn main() {
    eprintln!("harmonic_protocol_probe requires Unix process and socket semantics");
    std::process::exit(2);
}

#[cfg(unix)]
mod unix {
    use std::fs::{self, OpenOptions};
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::path::{Path, PathBuf};
    use std::process::{Child, Command, ExitStatus, Stdio};
    use std::thread;
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    use hipfire_arch_deepseek4::{
        harmonic_payload_fingerprint, HarmonicCompletion, HarmonicContract, HarmonicOwner,
        HarmonicSharedRing, HarmonicSlotState, HARMONIC_ACTIVATION_EXTENT, HARMONIC_RESULT_EXTENT,
    };
    use serde::{Deserialize, Serialize};

    const RPC_TIMEOUT: Duration = Duration::from_secs(2);
    const EXIT_TIMEOUT: Duration = Duration::from_millis(500);

    type ProbeResult<T> = Result<T, String>;

    #[derive(Clone, Copy, Debug, Deserialize, Serialize)]
    #[serde(rename_all = "snake_case")]
    enum Role {
        Dense,
        Expert,
    }

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(tag = "op", rename_all = "snake_case")]
    enum Request {
        Ping,
        Publish {
            epoch: u64,
            now: u64,
            deadline: u64,
            malformed_owner: bool,
        },
        PublishPartial {
            epoch: u64,
            now: u64,
            deadline: u64,
            payload_words: usize,
        },
        Begin {
            epoch: u64,
            now: u64,
        },
        Process {
            epoch: u64,
            now: u64,
        },
        Resolve {
            epoch: u64,
        },
        Cancel {
            epoch: u64,
        },
        Acknowledge {
            epoch: u64,
        },
        Shutdown,
    }

    #[derive(Debug, Deserialize, Serialize)]
    struct Response {
        ok: bool,
        role: Role,
        state: Option<String>,
        fingerprint: Option<u64>,
        error: Option<String>,
    }

    impl Response {
        fn ok(role: Role) -> Self {
            Self {
                ok: true,
                role,
                state: None,
                fingerprint: None,
                error: None,
            }
        }

        fn error(role: Role, error: impl ToString) -> Self {
            Self {
                ok: false,
                role,
                state: None,
                fingerprint: None,
                error: Some(error.to_string()),
            }
        }
    }

    struct ProbeWorkspace {
        root: PathBuf,
        ring: PathBuf,
    }

    impl ProbeWorkspace {
        fn create() -> ProbeResult<Self> {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|error| error.to_string())?
                .as_nanos();
            // Keep Unix-domain socket names below SUN_LEN even from a deeply
            // nested worktree. Workers inherit the controller's current dir.
            let root = PathBuf::from("target").join(format!("hp-{}-{nonce}", std::process::id()));
            fs::create_dir_all(&root).map_err(|error| error.to_string())?;
            Ok(Self {
                ring: root.join("ring.bin"),
                root,
            })
        }

        fn socket(&self, role: Role, generation: u64) -> PathBuf {
            self.root.join(format!("{role:?}-{generation}.sock"))
        }
    }

    impl Drop for ProbeWorkspace {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    struct Worker {
        role: Role,
        child: Child,
        reader: BufReader<UnixStream>,
        writer: UnixStream,
        reaped: bool,
    }

    impl Worker {
        fn spawn(
            executable: &Path,
            workspace: &ProbeWorkspace,
            role: Role,
            generation: u64,
        ) -> ProbeResult<Self> {
            let socket = workspace.socket(role, generation);
            let _ = fs::remove_file(&socket);
            let listener = UnixListener::bind(&socket).map_err(|error| error.to_string())?;
            listener
                .set_nonblocking(true)
                .map_err(|error| error.to_string())?;
            let mut child = Command::new(executable)
                .arg("--worker")
                .arg(match role {
                    Role::Dense => "dense",
                    Role::Expert => "expert",
                })
                .arg(&workspace.ring)
                .arg(&socket)
                .arg(generation.to_string())
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::inherit())
                .spawn()
                .map_err(|error| error.to_string())?;

            let deadline = Instant::now() + RPC_TIMEOUT;
            let stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        if let Some(status) = child.try_wait().map_err(|error| error.to_string())? {
                            return Err(format!("{role:?} worker exited during connect: {status}"));
                        }
                        if Instant::now() >= deadline {
                            let _ = child.kill();
                            return Err(format!("{role:?} worker connect timed out"));
                        }
                        thread::sleep(Duration::from_millis(1));
                    }
                    Err(error) => return Err(error.to_string()),
                }
            };
            stream
                .set_read_timeout(Some(RPC_TIMEOUT))
                .map_err(|error| error.to_string())?;
            stream
                .set_write_timeout(Some(RPC_TIMEOUT))
                .map_err(|error| error.to_string())?;
            let writer = stream.try_clone().map_err(|error| error.to_string())?;
            let mut worker = Self {
                role,
                child,
                reader: BufReader::new(stream),
                writer,
                reaped: false,
            };
            worker.require_ok(Request::Ping)?;
            Ok(worker)
        }

        fn rpc(&mut self, request: Request) -> ProbeResult<Response> {
            serde_json::to_writer(&mut self.writer, &request).map_err(|error| error.to_string())?;
            self.writer
                .write_all(b"\n")
                .and_then(|_| self.writer.flush())
                .map_err(|error| error.to_string())?;
            let mut line = String::new();
            let bytes = self
                .reader
                .read_line(&mut line)
                .map_err(|error| format!("{:?} worker RPC read: {error}", self.role))?;
            if bytes == 0 {
                return Err(format!("{:?} worker closed RPC socket", self.role));
            }
            serde_json::from_str(line.trim()).map_err(|error| error.to_string())
        }

        fn require_ok(&mut self, request: Request) -> ProbeResult<Response> {
            let response = self.rpc(request)?;
            if !response.ok {
                return Err(response
                    .error
                    .unwrap_or_else(|| format!("{:?} worker rejected request", self.role)));
            }
            Ok(response)
        }

        fn kill_confirmed(&mut self) -> ProbeResult<ExitStatus> {
            self.child.kill().map_err(|error| error.to_string())?;
            let status = wait_bounded(&mut self.child, EXIT_TIMEOUT)?;
            self.reaped = true;
            Ok(status)
        }

        fn shutdown(&mut self) -> ProbeResult<ExitStatus> {
            if self.reaped {
                return Err(format!("{:?} worker already reaped", self.role));
            }
            let _ = self.rpc(Request::Shutdown);
            let status = wait_bounded(&mut self.child, EXIT_TIMEOUT)?;
            self.reaped = true;
            Ok(status)
        }
    }

    impl Drop for Worker {
        fn drop(&mut self) {
            if self.reaped {
                return;
            }
            let _ = self.child.kill();
            let _ = wait_bounded(&mut self.child, EXIT_TIMEOUT);
            self.reaped = true;
        }
    }

    fn wait_bounded(child: &mut Child, timeout: Duration) -> ProbeResult<ExitStatus> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(status) = child.try_wait().map_err(|error| error.to_string())? {
                return Ok(status);
            }
            if Instant::now() >= deadline {
                return Err(format!(
                    "worker PID {} did not exit within {} ms",
                    child.id(),
                    timeout.as_millis()
                ));
            }
            thread::sleep(Duration::from_millis(1));
        }
    }

    fn payload(len: usize, seed: u64) -> Vec<u8> {
        (0..len)
            .map(|index| {
                seed.wrapping_mul(131)
                    .wrapping_add(index as u64)
                    .rotate_left(11) as u8
            })
            .collect()
    }

    fn activation(epoch: u64) -> Vec<u8> {
        payload(HARMONIC_ACTIVATION_EXTENT as usize, epoch)
    }

    fn result(epoch: u64) -> Vec<u8> {
        payload(
            HARMONIC_RESULT_EXTENT as usize,
            epoch ^ 0xa5a5_5a5a_1234_5678,
        )
    }

    fn worker_main(args: &[String]) -> ProbeResult<()> {
        if args.len() != 6 {
            return Err("worker expects role, ring, socket, and generation".to_owned());
        }
        let role = match args[2].as_str() {
            "dense" => Role::Dense,
            "expert" => Role::Expert,
            other => return Err(format!("unknown worker role {other:?}")),
        };
        let ring_path = PathBuf::from(&args[3]);
        let socket_path = PathBuf::from(&args[4]);
        let generation = args[5].parse::<u64>().map_err(|error| error.to_string())?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&ring_path)
            .map_err(|error| error.to_string())?;
        let ring = HarmonicSharedRing::open(&file).map_err(|error| error.to_string())?;
        let stream = UnixStream::connect(&socket_path).map_err(|error| error.to_string())?;
        stream
            .set_read_timeout(Some(RPC_TIMEOUT))
            .map_err(|error| error.to_string())?;
        stream
            .set_write_timeout(Some(RPC_TIMEOUT))
            .map_err(|error| error.to_string())?;
        let mut writer = stream.try_clone().map_err(|error| error.to_string())?;
        let mut reader = BufReader::new(stream);
        loop {
            let mut line = String::new();
            let bytes = reader
                .read_line(&mut line)
                .map_err(|error| error.to_string())?;
            if bytes == 0 {
                return Ok(());
            }
            let request: Request =
                serde_json::from_str(line.trim()).map_err(|error| error.to_string())?;
            let shutdown = matches!(request, Request::Shutdown);
            let response = match handle_request(role, generation, &ring, request) {
                Ok(response) => response,
                Err(error) => Response::error(role, error),
            };
            serde_json::to_writer(&mut writer, &response).map_err(|error| error.to_string())?;
            writer
                .write_all(b"\n")
                .and_then(|_| writer.flush())
                .map_err(|error| error.to_string())?;
            if shutdown {
                return Ok(());
            }
        }
    }

    fn handle_request(
        role: Role,
        generation: u64,
        ring: &HarmonicSharedRing,
        request: Request,
    ) -> ProbeResult<Response> {
        match request {
            Request::Ping | Request::Shutdown => Ok(Response::ok(role)),
            Request::Publish {
                epoch,
                now,
                deadline,
                malformed_owner,
            } => {
                if !matches!(role, Role::Dense) {
                    return Err("publish sent to non-dense worker".to_owned());
                }
                let activation = activation(epoch);
                let mut packet = ring.contract().packet(
                    epoch,
                    epoch as u16 % 43,
                    std::array::from_fn(|index| ((epoch + index as u64) % 256) as u32),
                    [0.5f32, 0.4, 0.3, 0.2, 0.1, 0.05].map(f32::to_bits),
                    deadline,
                    harmonic_payload_fingerprint(&activation),
                );
                if malformed_owner {
                    packet.destination_owner = HarmonicOwner::DenseGfx1100;
                }
                ring.publish(packet, generation, now, &activation)
                    .map_err(|error| error.to_string())?;
                Ok(Response::ok(role))
            }
            Request::PublishPartial {
                epoch,
                now,
                deadline,
                payload_words,
            } => {
                if !matches!(role, Role::Dense) {
                    return Err("partial publish sent to non-dense worker".to_owned());
                }
                let activation = activation(epoch);
                let packet = ring.contract().packet(
                    epoch,
                    epoch as u16 % 43,
                    std::array::from_fn(|index| ((epoch + index as u64) % 256) as u32),
                    [0.5f32, 0.4, 0.3, 0.2, 0.1, 0.05].map(f32::to_bits),
                    deadline,
                    harmonic_payload_fingerprint(&activation),
                );
                ring.fault_inject_partial_publish_for_probe(
                    packet,
                    generation,
                    now,
                    &activation,
                    payload_words,
                )
                .map_err(|error| error.to_string())?;
                Ok(Response::ok(role))
            }
            Request::Begin { epoch, now } => {
                if !matches!(role, Role::Expert) {
                    return Err("begin sent to non-expert worker".to_owned());
                }
                let work = ring
                    .expert_begin(epoch, generation, now)
                    .map_err(|error| error.to_string())?;
                if work.activation_payload != activation(epoch) {
                    return Err("activation bytes differ".to_owned());
                }
                Ok(Response::ok(role))
            }
            Request::Process { epoch, now } => {
                if !matches!(role, Role::Expert) {
                    return Err("process sent to non-expert worker".to_owned());
                }
                let work = ring
                    .expert_begin(epoch, generation, now)
                    .map_err(|error| error.to_string())?;
                if work.activation_payload != activation(epoch) {
                    return Err("activation bytes differ".to_owned());
                }
                let result = result(epoch);
                let fingerprint = harmonic_payload_fingerprint(&result);
                ring.expert_complete(
                    epoch,
                    generation,
                    HarmonicCompletion {
                        result_extent: HARMONIC_RESULT_EXTENT,
                        result_fingerprint: fingerprint,
                    },
                    &result,
                )
                .map_err(|error| error.to_string())?;
                let mut response = Response::ok(role);
                response.fingerprint = Some(fingerprint);
                Ok(response)
            }
            Request::Resolve { epoch } => {
                if !matches!(role, Role::Dense) {
                    return Err("resolve sent to non-dense worker".to_owned());
                }
                let resolved = ring
                    .source_resolve(epoch, generation)
                    .map_err(|error| error.to_string())?;
                if resolved.state == HarmonicSlotState::Completed
                    && resolved.result_payload.as_deref() != Some(result(epoch).as_slice())
                {
                    return Err("result bytes differ".to_owned());
                }
                let mut response = Response::ok(role);
                response.state = Some(format!("{:?}", resolved.state));
                response.fingerprint = resolved
                    .completion
                    .map(|completion| completion.result_fingerprint);
                Ok(response)
            }
            Request::Cancel { epoch } => {
                if !matches!(role, Role::Dense) {
                    return Err("cancel sent to non-dense worker".to_owned());
                }
                ring.source_cancel(epoch, generation)
                    .map_err(|error| error.to_string())?;
                Ok(Response::ok(role))
            }
            Request::Acknowledge { epoch } => {
                if !matches!(role, Role::Expert) {
                    return Err("acknowledge sent to non-expert worker".to_owned());
                }
                ring.expert_acknowledge_terminal(epoch, generation)
                    .map_err(|error| error.to_string())?;
                Ok(Response::ok(role))
            }
        }
    }

    fn controller_main() -> ProbeResult<()> {
        let workspace = ProbeWorkspace::create()?;
        let file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&workspace.ring)
            .map_err(|error| error.to_string())?;
        let ring = HarmonicSharedRing::create(&file, HarmonicContract::frozen(1, 1))
            .map_err(|error| error.to_string())?;
        let executable = std::env::current_exe().map_err(|error| error.to_string())?;
        let mut dense_generation = 1;
        let mut expert_generation = 1;
        let mut dense = Worker::spawn(&executable, &workspace, Role::Dense, dense_generation)?;
        let mut expert = Worker::spawn(&executable, &workspace, Role::Expert, expert_generation)?;

        let chains = 10_000_u64;
        let started = Instant::now();
        for epoch in 1..=chains {
            let now = epoch * 4;
            dense.require_ok(Request::Publish {
                epoch,
                now,
                deadline: now + 3,
                malformed_owner: false,
            })?;
            let processed = expert.require_ok(Request::Process {
                epoch,
                now: now + 1,
            })?;
            let resolved = dense.require_ok(Request::Resolve { epoch })?;
            let expected = harmonic_payload_fingerprint(&result(epoch));
            if processed.fingerprint != Some(expected) || resolved.fingerprint != Some(expected) {
                return Err(format!("fingerprint mismatch at epoch {epoch}"));
            }
            ring.recycle(epoch).map_err(|error| error.to_string())?;
        }
        let chain_elapsed = started.elapsed();

        // Malformed owner fails before occupying a physical slot.
        let malformed = dense.rpc(Request::Publish {
            epoch: chains + 1,
            now: 0,
            deadline: 10,
            malformed_owner: true,
        })?;
        if malformed.ok {
            return Err("malformed owner was admitted".to_owned());
        }

        // Mid-service cancellation after destination acquisition requires
        // explicit destination quiescence.
        let cancel_epoch = chains + 1;
        dense.require_ok(Request::Publish {
            epoch: cancel_epoch,
            now: 0,
            deadline: 10,
            malformed_owner: false,
        })?;
        expert.require_ok(Request::Begin {
            epoch: cancel_epoch,
            now: 1,
        })?;
        dense.require_ok(Request::Cancel {
            epoch: cancel_epoch,
        })?;
        if ring.recycle(cancel_epoch).is_ok() {
            return Err("cancelled slot recycled before expert acknowledgement".to_owned());
        }
        expert.require_ok(Request::Acknowledge {
            epoch: cancel_epoch,
        })?;
        ring.recycle(cancel_epoch)
            .map_err(|error| error.to_string())?;

        // Timeout similarly requires both source observation and destination
        // acknowledgement; no worker waits on the peer.
        let timeout_epoch = chains + 2;
        dense.require_ok(Request::Publish {
            epoch: timeout_epoch,
            now: 0,
            deadline: 10,
            malformed_owner: false,
        })?;
        expert.require_ok(Request::Begin {
            epoch: timeout_epoch,
            now: 1,
        })?;
        if !ring
            .expire(timeout_epoch, 10)
            .map_err(|error| error.to_string())?
        {
            return Err("timeout did not transition".to_owned());
        }
        dense.require_ok(Request::Resolve {
            epoch: timeout_epoch,
        })?;
        expert.require_ok(Request::Acknowledge {
            epoch: timeout_epoch,
        })?;
        ring.recycle(timeout_epoch)
            .map_err(|error| error.to_string())?;

        // Kill the dense producer after it has reserved a slot and written
        // half the activation, but before its release-publication. The expert
        // must remain responsive and must never acquire the partial payload.
        let partial_publish_epoch = chains + 3;
        dense.require_ok(Request::PublishPartial {
            epoch: partial_publish_epoch,
            now: 0,
            deadline: 10,
            payload_words: (HARMONIC_ACTIVATION_EXTENT as usize / 8) / 2,
        })?;
        if ring
            .state(partial_publish_epoch)
            .map_err(|error| error.to_string())?
            != hipfire_arch_deepseek4::HarmonicWireState::Publishing
        {
            return Err("partial publication did not remain Publishing".to_owned());
        }
        dense.kill_confirmed()?;
        ring.isolate_owner(HarmonicOwner::DenseGfx1100, dense_generation)
            .map_err(|error| error.to_string())?;
        expert.require_ok(Request::Ping)?;
        let partial_begin = expert.rpc(Request::Begin {
            epoch: partial_publish_epoch,
            now: 1,
        })?;
        if partial_begin.ok {
            return Err("expert acquired a partial publication".to_owned());
        }
        ring.recycle(partial_publish_epoch)
            .map_err(|error| error.to_string())?;
        ring.advance_generation(
            HarmonicOwner::DenseGfx1100,
            dense_generation,
            dense_generation + 1,
        )
        .map_err(|error| error.to_string())?;
        dense_generation += 1;
        dense = Worker::spawn(&executable, &workspace, Role::Dense, dense_generation)?;

        // Expert producer loss: reap first, then mark isolation. The dense
        // worker remains responsive and resolves the failed epoch.
        let expert_loss_epoch = chains + 4;
        dense.require_ok(Request::Publish {
            epoch: expert_loss_epoch,
            now: 0,
            deadline: 10,
            malformed_owner: false,
        })?;
        expert.require_ok(Request::Begin {
            epoch: expert_loss_epoch,
            now: 1,
        })?;
        expert.kill_confirmed()?;
        ring.isolate_owner(HarmonicOwner::ExpertGfx1151, expert_generation)
            .map_err(|error| error.to_string())?;
        dense.require_ok(Request::Ping)?;
        let failed = dense.require_ok(Request::Resolve {
            epoch: expert_loss_epoch,
        })?;
        if failed.state.as_deref() != Some("Failed(ExpertGfx1151)") {
            return Err(format!("wrong expert-loss state: {:?}", failed.state));
        }
        ring.recycle(expert_loss_epoch)
            .map_err(|error| error.to_string())?;
        ring.advance_generation(
            HarmonicOwner::ExpertGfx1151,
            expert_generation,
            expert_generation + 1,
        )
        .map_err(|error| error.to_string())?;
        expert_generation += 1;
        expert = Worker::spawn(&executable, &workspace, Role::Expert, expert_generation)?;

        // Dense producer loss: the surviving expert acknowledges terminal
        // quiescence before the slot is reclaimed and the dense generation is
        // replaced.
        let dense_loss_epoch = chains + 5;
        dense.require_ok(Request::Publish {
            epoch: dense_loss_epoch,
            now: 0,
            deadline: 10,
            malformed_owner: false,
        })?;
        expert.require_ok(Request::Begin {
            epoch: dense_loss_epoch,
            now: 1,
        })?;
        dense.kill_confirmed()?;
        ring.isolate_owner(HarmonicOwner::DenseGfx1100, dense_generation)
            .map_err(|error| error.to_string())?;
        expert.require_ok(Request::Ping)?;
        expert.require_ok(Request::Acknowledge {
            epoch: dense_loss_epoch,
        })?;
        ring.recycle(dense_loss_epoch)
            .map_err(|error| error.to_string())?;
        ring.advance_generation(
            HarmonicOwner::DenseGfx1100,
            dense_generation,
            dense_generation + 1,
        )
        .map_err(|error| error.to_string())?;
        dense_generation += 1;
        dense = Worker::spawn(&executable, &workspace, Role::Dense, dense_generation)?;

        // One exact chain after both replacements proves the new generations
        // can use the same physical ring without accepting stale ownership.
        let recovery_epoch = chains + 6;
        dense.require_ok(Request::Publish {
            epoch: recovery_epoch,
            now: 0,
            deadline: 10,
            malformed_owner: false,
        })?;
        expert.require_ok(Request::Process {
            epoch: recovery_epoch,
            now: 1,
        })?;
        dense.require_ok(Request::Resolve {
            epoch: recovery_epoch,
        })?;
        ring.recycle(recovery_epoch)
            .map_err(|error| error.to_string())?;

        let dense_status = dense.shutdown()?;
        let expert_status = expert.shutdown()?;
        if !dense_status.success() || !expert_status.success() {
            return Err(format!(
                "worker shutdown failed: dense={dense_status}, expert={expert_status}"
            ));
        }

        let rpc_count = chains * 3;
        println!(
            "{{\"status\":\"pass\",\"gpu_touched\":false,\"chains\":{chains},\"payload_bytes_per_chain\":{},\"rpc_count\":{rpc_count},\"elapsed_ms\":{},\"mean_rpc_us\":{:.3},\"faults\":[\"malformed_owner\",\"mid_service_cancel\",\"timeout\",\"dense_exit_during_publish\",\"expert_exit\",\"dense_exit_during_service\"],\"dense_generation\":{dense_generation},\"expert_generation\":{expert_generation}}}",
            u64::from(HARMONIC_ACTIVATION_EXTENT) + u64::from(HARMONIC_RESULT_EXTENT),
            chain_elapsed.as_millis(),
            chain_elapsed.as_secs_f64() * 1_000_000.0 / rpc_count as f64,
        );
        Ok(())
    }

    pub fn run() -> ProbeResult<()> {
        let args = std::env::args().collect::<Vec<_>>();
        if args.get(1).is_some_and(|arg| arg == "--worker") {
            worker_main(&args)
        } else {
            controller_main()
        }
    }
}

#[cfg(unix)]
fn main() {
    if let Err(error) = unix::run() {
        eprintln!("harmonic_protocol_probe: {error}");
        std::process::exit(1);
    }
}
