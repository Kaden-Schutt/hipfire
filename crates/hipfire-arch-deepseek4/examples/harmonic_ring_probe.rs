// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! CPU-only zero-control-RPC probe for the harmonic steady-state ring.
//!
//! The controller publishes and resolves 10,000 full-size typed packets while
//! an independent persistent process discovers epochs directly from the mmap.
//! There is no socket, JSON, pipe, signal, or process-control operation per
//! chain. This probes protocol shape and host overhead only; it never opens a
//! HIP or ROCr runtime.

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("harmonic_ring_probe requires Linux process and monotonic-clock semantics");
    std::process::exit(2);
}

#[cfg(target_os = "linux")]
mod linux {
    use std::fs::{self, OpenOptions};
    use std::path::{Path, PathBuf};
    use std::process::{Child, Command, ExitStatus, Stdio};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    use hipfire_arch_deepseek4::{
        harmonic_monotonic_tick, HarmonicCompletion, HarmonicContract, HarmonicExpertPoll,
        HarmonicSharedRing, HarmonicWireState, HARMONIC_ACTIVATION_EXTENT, HARMONIC_RESULT_EXTENT,
    };

    const CHAINS: u64 = 10_000;
    const SOURCE_GENERATION: u64 = 7;
    const EXPERT_GENERATION: u64 = 11;
    const CHAIN_TIMEOUT: Duration = Duration::from_secs(2);
    const EXIT_TIMEOUT: Duration = Duration::from_millis(500);
    static NONCE: AtomicU64 = AtomicU64::new(0);

    type ProbeResult<T> = Result<T, String>;

    struct ProbeRing {
        path: PathBuf,
        ring: HarmonicSharedRing,
    }

    impl ProbeRing {
        fn create() -> ProbeResult<Self> {
            let nonce = NONCE.fetch_add(1, Ordering::Relaxed);
            let root = PathBuf::from("target").join(format!(
                "harmonic-ring-probe-{}-{nonce}",
                std::process::id()
            ));
            fs::create_dir_all(&root).map_err(|error| error.to_string())?;
            let path = root.join("ring.bin");
            let file = OpenOptions::new()
                .create_new(true)
                .read(true)
                .write(true)
                .open(&path)
                .map_err(|error| error.to_string())?;
            let ring = HarmonicSharedRing::create_data_plane(
                &file,
                HarmonicContract::frozen(SOURCE_GENERATION, EXPERT_GENERATION),
            )
            .map_err(|error| error.to_string())?;
            Ok(Self { path, ring })
        }
    }

    impl Drop for ProbeRing {
        fn drop(&mut self) {
            if let Some(root) = self.path.parent() {
                let _ = fs::remove_dir_all(root);
            }
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

    fn wait_bounded(child: &mut Child, timeout: Duration) -> ProbeResult<ExitStatus> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(status) = child.try_wait().map_err(|error| error.to_string())? {
                return Ok(status);
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!(
                    "exact ring worker {} did not exit within {timeout:?}",
                    child.id()
                ));
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn worker(path: &Path) -> ProbeResult<()> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|error| error.to_string())?;
        let ring = HarmonicSharedRing::open(&file).map_err(|error| error.to_string())?;
        let result_bytes = result(1);
        let mut epoch = 1_u64;
        while epoch <= CHAINS {
            match ring
                .expert_poll(epoch, EXPERT_GENERATION)
                .map_err(|error| error.to_string())?
            {
                HarmonicExpertPoll::Pending => std::hint::spin_loop(),
                HarmonicExpertPoll::Terminal(_) => {
                    ring.expert_acknowledge_terminal(epoch, EXPERT_GENERATION)
                        .map_err(|error| error.to_string())?;
                    epoch += 1;
                }
                HarmonicExpertPoll::Work(_) => {
                    ring.expert_complete(
                        epoch,
                        EXPERT_GENERATION,
                        HarmonicCompletion {
                            result_extent: HARMONIC_RESULT_EXTENT,
                            result_fingerprint: 0,
                        },
                        &result_bytes,
                    )
                    .map_err(|error| error.to_string())?;
                    epoch += 1;
                }
            }
        }
        Ok(())
    }

    fn controller() -> ProbeResult<()> {
        let probe = ProbeRing::create()?;
        let executable = std::env::current_exe().map_err(|error| error.to_string())?;
        let mut child = Command::new(executable)
            .arg("--worker")
            .arg(&probe.path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| error.to_string())?;

        let activation = activation(1);
        let expected_result = result(1);
        let started = Instant::now();
        let run = (|| -> ProbeResult<()> {
            for epoch in 1..=CHAINS {
                let now = harmonic_monotonic_tick().map_err(|error| error.to_string())?;
                let packet = probe.ring.contract().packet(
                    epoch,
                    (epoch % 43) as u16,
                    [0, 1, 2, 3, 4, 5],
                    [0.5f32, 0.4, 0.3, 0.2, 0.1, 0.05].map(f32::to_bits),
                    now.saturating_add(CHAIN_TIMEOUT.as_nanos() as u64),
                    0,
                );
                probe
                    .ring
                    .publish(packet, SOURCE_GENERATION, now, &activation)
                    .map_err(|error| error.to_string())?;

                let deadline = Instant::now() + CHAIN_TIMEOUT;
                loop {
                    match probe.ring.state(epoch).map_err(|error| error.to_string())? {
                        state if state.is_terminal() => break,
                        HarmonicWireState::Publishing
                        | HarmonicWireState::Published
                        | HarmonicWireState::Running
                        | HarmonicWireState::Completing => {}
                        HarmonicWireState::Vacant => {
                            return Err(format!("epoch {epoch} became vacant before resolution"));
                        }
                        _ => unreachable!(),
                    }
                    if Instant::now() >= deadline {
                        return Err(format!("epoch {epoch} completion timed out"));
                    }
                    std::hint::spin_loop();
                }
                let resolved = probe
                    .ring
                    .source_resolve(epoch, SOURCE_GENERATION)
                    .map_err(|error| error.to_string())?;
                if resolved.result_payload.as_deref() != Some(expected_result.as_slice()) {
                    return Err(format!("epoch {epoch} result mismatch"));
                }
                probe
                    .ring
                    .recycle(epoch)
                    .map_err(|error| error.to_string())?;
            }
            Ok(())
        })();

        if let Err(error) = run {
            let _ = child.kill();
            let _ = child.wait();
            return Err(error);
        }
        let status = wait_bounded(&mut child, EXIT_TIMEOUT)?;
        if !status.success() {
            return Err(format!("ring worker exited with {status}"));
        }
        let elapsed = started.elapsed();
        println!(
            "{{\"status\":\"pass\",\"gpu_touched\":false,\"chains\":{CHAINS},\"control_messages\":0,\"payload_bytes_per_chain\":{},\"elapsed_ms\":{},\"mean_chain_us\":{:.3}}}",
            HARMONIC_ACTIVATION_EXTENT as u64 + HARMONIC_RESULT_EXTENT as u64,
            elapsed.as_millis(),
            elapsed.as_secs_f64() * 1_000_000.0 / CHAINS as f64,
        );
        Ok(())
    }

    pub fn run() -> ProbeResult<()> {
        let args = std::env::args().collect::<Vec<_>>();
        if args.get(1).map(String::as_str) == Some("--worker") {
            let path = args
                .get(2)
                .ok_or_else(|| "--worker requires a ring path".to_owned())?;
            return worker(Path::new(path));
        }
        controller()
    }
}

#[cfg(target_os = "linux")]
fn main() {
    if let Err(error) = linux::run() {
        eprintln!("harmonic_ring_probe: {error}");
        std::process::exit(1);
    }
}
