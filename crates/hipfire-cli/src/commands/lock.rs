// SPDX-License-Identifier: Apache-2.0
// hipfire — native GPU resource lock (the legacy scripts/gpu-lock.sh shell adapter has been removed).
//
//! `hipfire lock {acquire,release,status}` (alias: `gpu-lock`) — a flock(2)-backed
//! GPU resource mutex for multi-agent coordination, owned by the engine instead
//! of a shell script. Locks `hipfire_lock::gpu_resource_lock_path()` (=
//! `resource_lock_path("hip-gpu-0")`) — the SAME inode the daemon's default GPU
//! resource lease uses, so non-daemon GPU binaries and the daemon coordinate.
//!
//! Mechanism (flock + lock-holder helper):
//! - `acquire <label>` opens the lockfile, takes a blocking `LOCK_EX` (polling
//!   with a `busy` message + optional timeout), writes holder metadata, then
//!   spawns a detached `setsid` holder (`lock hold`, hidden) that INHERITS
//!   the already-locked fd. The acquiring process then exits; its fd copy
//!   closes, but the holder's inherited copy keeps the lock on the same open
//!   file description. So `acquire` returns immediately with the lock held.
//! - The holder watches `--watch-pid` (default: the calling shell). When that
//!   pid dies for ANY reason — or `release` SIGTERMs the holder — the holder
//!   exits, the kernel drops the flock, and the GPU is free. Stale locks are
//!   structurally impossible (kernel-backed release), exactly like the shell
//!   version, while still supporting standalone acquire/release.
//! - `release` reads the holder pid from the lockfile and SIGTERMs it.
//! - `status` takes a non-blocking probe lock: success ⇒ free, EWOULDBLOCK ⇒
//!   busy (prints the holder metadata line).
//!
//! Preferred form — `run <label> -- <cmd>`: acquire, spawn `<cmd>` as a child
//! while holding the fd, and release on exit. The lock lives exactly as long as
//! this process — no detached holder, no watched pid, no polling: if the wrapper
//! (or its whole tree) is killed for any reason, the kernel drops the flock.
//! `acquire`/`release`/`hold` remain for the split-across-processes pattern but
//! `run` is the scoped, footgun-free way to hold the GPU for a unit of work.
//!
//! NB: the lockfile is never unlinked — unlinking a flock'd file lets the next
//! acquirer lock a different inode and yields two simultaneous holders.

use std::os::fd::{FromRawFd, OwnedFd};
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicI32, Ordering};

use clap::{Args, Subcommand};

#[derive(Debug, Args)]
pub struct LockArgs {
    #[command(subcommand)]
    action: LockAction,
}

#[derive(Debug, Subcommand)]
enum LockAction {
    /// Acquire the GPU lock (blocks until free). A detached holder keeps it
    /// until `release` or the calling shell exits.
    Acquire {
        /// Human label recorded in the lockfile (who/what holds it).
        label: String,
        /// Pid whose death auto-releases the lock (default: the calling shell).
        #[arg(long)]
        watch_pid: Option<i32>,
        /// Hard cap in seconds to wait for a busy lock; 0 = wait forever.
        #[arg(long, default_value_t = default_timeout())]
        timeout_secs: u64,
        /// Cadence of "busy" messages while waiting, in seconds.
        #[arg(long, default_value_t = default_poll())]
        poll_secs: u64,
    },
    /// Release the GPU lock (SIGTERM the holder recorded in the lockfile).
    Release,
    /// Print lock status: "gpu is free" or "gpu BUSY: <holder>".
    Status,
    /// Acquire the lock, run `command` under it, release on exit — the scoped
    /// form. The lock lives exactly as long as this process (killing it drops
    /// the flock via the kernel); no detached holder or watched pid. Exit code
    /// is the command's; 2 on acquire timeout. Usage: `lock run <label> -- cmd…`.
    Run {
        /// Human label recorded in the lockfile (who/what holds it).
        label: String,
        /// Hard cap in seconds to wait for a busy lock; 0 = wait forever.
        #[arg(long, default_value_t = default_timeout())]
        timeout_secs: u64,
        /// Cadence of "busy" messages while waiting, in seconds.
        #[arg(long, default_value_t = default_poll())]
        poll_secs: u64,
        /// The command (and args) to run under the lock — everything after `--`.
        #[arg(last = true, required = true)]
        command: Vec<String>,
    },
    /// INTERNAL: the detached lock holder spawned by `acquire`. Not for direct use.
    #[command(hide = true)]
    Hold {
        /// Inherited, already-flock'd fd to hold open.
        #[arg(long)]
        lock_fd: i32,
        /// Pid to watch; exit (release the lock) when it dies.
        #[arg(long)]
        watch_pid: i32,
        /// Liveness poll cadence, seconds.
        #[arg(long)]
        poll_secs: u64,
    },
}

fn default_poll() -> u64 {
    std::env::var("GPU_POLL_INTERVAL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5)
}

fn default_timeout() -> u64 {
    std::env::var("GPU_LOCK_TIMEOUT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1800)
}

fn lockfile_path() -> PathBuf {
    // The canonical path/env-var contract lives in `hipfire-lock` so the daemon,
    // gpu-lock.sh, and any future participant agree on one file + env var.
    hipfire_lock::gpu_resource_lock_path()
}

fn pid_alive(pid: i32) -> bool {
    if pid <= 1 {
        return false;
    }
    // kill(pid, 0): 0 ⇒ alive; EPERM ⇒ alive (not permitted); ESRCH ⇒ gone.
    unsafe { libc::kill(pid, 0) == 0 || *libc::__errno_location() == libc::EPERM }
}

pub fn run(args: LockArgs) -> anyhow::Result<()> {
    match args.action {
        LockAction::Acquire {
            label,
            watch_pid,
            timeout_secs,
            poll_secs,
        } => acquire(&label, watch_pid, timeout_secs, poll_secs.max(1)),
        LockAction::Release => release(),
        LockAction::Status => {
            println!("{}", status_line());
            Ok(())
        }
        LockAction::Run {
            label,
            timeout_secs,
            poll_secs,
            command,
        } => {
            let code = run_scoped(&label, timeout_secs, poll_secs.max(1), &command)?;
            std::process::exit(code);
        }
        LockAction::Hold {
            lock_fd,
            watch_pid,
            poll_secs,
        } => hold(lock_fd, watch_pid, poll_secs.max(1)),
    }
}

fn acquire(
    label: &str,
    watch_pid: Option<i32>,
    timeout_secs: u64,
    poll_secs: u64,
) -> anyhow::Result<()> {
    // The pid whose death releases the lock: the caller's shell by default.
    let watch_pid = watch_pid.unwrap_or_else(|| unsafe { libc::getppid() });
    let path = lockfile_path();

    // Block (poll) until we hold LOCK_EX, surfacing the holder + a hard cap, via
    // the shared `hipfire-lock` flock primitive.
    let mut guard = hipfire_lock::FlockGuard::open(&path)?;
    let timeout = (timeout_secs > 0).then(|| std::time::Duration::from_secs(timeout_secs));
    let mut waited = 0u64;
    let acquired = guard.lock_blocking(
        std::time::Duration::from_secs(poll_secs),
        timeout,
        |holder| {
            waited += poll_secs;
            let who = if holder.is_empty() { "unknown" } else { holder };
            eprintln!("[gpu-lock] busy: {who} — waited {waited}s, still waiting…");
        },
    )?;
    if !acquired {
        // Exit code 2 on timeout — the historical gpu-lock.sh contract that
        // gates/waiters distinguish from other failures.
        eprintln!(
            "[gpu-lock] TIMEOUT after {timeout_secs}s; holder still alive: {}",
            guard.holder().unwrap_or_else(|| "unknown".into())
        );
        std::process::exit(2);
    }

    // We hold it. Clear CLOEXEC so the holder inherits this fd across exec, then
    // spawn the detached holder; its inherited copy keeps the lock after we exit
    // (and our `guard` drops, closing our copy).
    let fd = guard.raw_fd();
    clear_cloexec(fd)?;
    let exe = std::env::current_exe()?;
    let mut cmd = Command::new(exe);
    cmd.args([
        "lock",
        "hold",
        "--lock-fd",
        &fd.to_string(),
        "--watch-pid",
        &watch_pid.to_string(),
        "--poll-secs",
        &poll_secs.to_string(),
    ])
    .stdin(Stdio::null())
    .stdout(Stdio::null())
    .stderr(Stdio::null());
    // New session: detach from the shell's process group so a Ctrl-C during the
    // locked work doesn't kill the holder and release early. SAFETY: setsid is
    // async-signal-safe and touches no shared state in the forked child.
    unsafe {
        cmd.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let holder = cmd.spawn()?;

    // Record metadata (truncate + write under the held flock — `flock` is on the
    // open fd, so rewriting the contents doesn't drop it).
    let meta = format!(
        "{label} pid={watch_pid} host={} acquired_epoch={} holder={}",
        hostname(),
        now_iso(),
        holder.id()
    );
    guard.write_holder(&meta)?;
    eprintln!("[gpu-lock] acquired by {label}");
    // Returning drops `guard` (closes our fd copy); the holder's inherited copy
    // keeps the lock.
    Ok(())
}

fn release() -> anyhow::Result<()> {
    let path = lockfile_path();
    let Some(holder_pid) = read_holder_pid(&path) else {
        eprintln!("[gpu-lock] no lock held");
        return Ok(());
    };
    if pid_alive(holder_pid) {
        unsafe { libc::kill(holder_pid, libc::SIGTERM) };
    }
    eprintln!("[gpu-lock] released");
    Ok(())
}

/// Process-group id of the child spawned by `lock run`, for the signal
/// forwarder to target. 0 = no child yet.
static CHILD_PGID: AtomicI32 = AtomicI32::new(0);

/// Signal handler installed by `lock run`: forward the received signal to the
/// child's whole process group, then let `wait` reap it and propagate the code.
/// Async-signal-safe: only an atomic load + `kill(2)`.
extern "C" fn forward_signal(sig: i32) {
    let pgid = CHILD_PGID.load(Ordering::SeqCst);
    if pgid > 0 {
        unsafe {
            libc::kill(-pgid, sig);
        }
    }
}

fn install_signal_forwarder() -> anyhow::Result<()> {
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = forward_signal as extern "C" fn(i32) as libc::sighandler_t;
        libc::sigemptyset(&mut sa.sa_mask);
        sa.sa_flags = libc::SA_RESTART;
        for sig in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP] {
            if libc::sigaction(sig, &sa, std::ptr::null_mut()) == -1 {
                anyhow::bail!("sigaction({sig}): {}", std::io::Error::last_os_error());
            }
        }
    }
    Ok(())
}

/// Map a child's exit status to a process exit code: its own code, or the
/// shell convention `128 + signal` when it was terminated by a signal.
fn exit_code_from(status: std::process::ExitStatus) -> i32 {
    use std::os::unix::process::ExitStatusExt;
    if let Some(code) = status.code() {
        code
    } else if let Some(sig) = status.signal() {
        128 + sig
    } else {
        1
    }
}

fn run_scoped(
    label: &str,
    timeout_secs: u64,
    poll_secs: u64,
    command: &[String],
) -> anyhow::Result<i32> {
    run_scoped_at(&lockfile_path(), label, timeout_secs, poll_secs, command, true)
}

/// Core of `lock run`, parameterized on the lockfile path and whether to install
/// the process-wide signal forwarder (tests pass `false` to avoid mutating the
/// harness's signal disposition). Returns the exit code to surface.
fn run_scoped_at(
    path: &std::path::Path,
    label: &str,
    timeout_secs: u64,
    poll_secs: u64,
    command: &[String],
    install_signals: bool,
) -> anyhow::Result<i32> {
    let (program, args) = command
        .split_first()
        .ok_or_else(|| anyhow::anyhow!("lock run requires a command after `--`"))?;

    // Block (poll) until we hold LOCK_EX via the shared `hipfire-lock` primitive.
    let mut guard = hipfire_lock::FlockGuard::open(path)?;
    let timeout = (timeout_secs > 0).then(|| std::time::Duration::from_secs(timeout_secs));
    let mut waited = 0u64;
    let acquired = guard.lock_blocking(
        std::time::Duration::from_secs(poll_secs),
        timeout,
        |holder| {
            waited += poll_secs;
            let who = if holder.is_empty() { "unknown" } else { holder };
            eprintln!("[gpu-lock] busy: {who} — waited {waited}s, still waiting…");
        },
    )?;
    if !acquired {
        eprintln!(
            "[gpu-lock] TIMEOUT after {timeout_secs}s; holder still alive: {}",
            guard.holder().unwrap_or_else(|| "unknown".into())
        );
        // Exit code 2 on timeout — the historical gpu-lock.sh contract.
        return Ok(2);
    }

    // We hold it. FlockGuard's fd is O_CLOEXEC, so the child does NOT inherit
    // it: THIS process is the sole owner, and when it dies for any reason the
    // kernel drops the lock — no detached holder, no watch-pid. Put the child in
    // its own process group so we are the sole recipient of terminal/kill
    // signals and can forward them to the whole workload tree.
    let mut cmd = Command::new(program);
    cmd.args(args).process_group(0);
    let mut child = cmd
        .spawn()
        .map_err(|e| anyhow::anyhow!("failed to spawn {program:?}: {e}"))?;
    // `process_group(0)` made the child a group leader, so pgid == child pid.
    CHILD_PGID.store(child.id() as i32, Ordering::SeqCst);

    // Record holder metadata: OUR pid is the holder (`release`/`kill` find it via
    // `holder=`). Written under the held flock — rewriting contents can't drop it.
    let meta = format!(
        "{label} host={} acquired_epoch={} holder={} mode=run",
        hostname(),
        now_iso(),
        std::process::id()
    );
    let _ = guard.write_holder(&meta);

    if install_signals {
        install_signal_forwarder()?;
    }
    eprintln!("[gpu-lock] acquired by {label} (run)");

    let status = child.wait()?;
    CHILD_PGID.store(0, Ordering::SeqCst);
    // Returning drops `guard` (closes our fd) → the kernel releases the flock.
    Ok(exit_code_from(status))
}

fn hold(lock_fd: i32, watch_pid: i32, poll_secs: u64) -> anyhow::Result<()> {
    // Take ownership of the inherited, already-locked fd so it stays open (and
    // is closed — releasing the flock — when this process exits for any reason,
    // including SIGTERM from `release`).
    let _held: OwnedFd = unsafe { OwnedFd::from_raw_fd(lock_fd) };
    loop {
        std::thread::sleep(std::time::Duration::from_secs(poll_secs));
        if !pid_alive(watch_pid) {
            return Ok(());
        }
    }
}

/// Non-blocking probe via the shared primitive: free if we can take the lock.
fn status_line() -> String {
    match hipfire_lock::probe(lockfile_path()) {
        Ok(hipfire_lock::LockState::Free) => "gpu is free".to_string(),
        Ok(hipfire_lock::LockState::Busy(holder)) => {
            let who = if holder.is_empty() {
                "unknown"
            } else {
                &holder
            };
            format!("gpu BUSY: {who}")
        }
        // Probe I/O error → report free (best-effort, matches the prior
        // open-failure fallback).
        Err(_) => "gpu is free".to_string(),
    }
}

fn read_holder(path: &std::path::Path) -> Option<String> {
    let s = std::fs::read_to_string(path).ok()?;
    let line = s.lines().next()?.trim().to_string();
    if line.is_empty() {
        None
    } else {
        Some(line)
    }
}

fn read_holder_pid(path: &std::path::Path) -> Option<i32> {
    let line = read_holder(path)?;
    line.split_whitespace()
        .find_map(|tok| tok.strip_prefix("holder="))
        .and_then(|v| v.parse().ok())
}

fn clear_cloexec(fd: i32) -> anyhow::Result<()> {
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
    if flags == -1 {
        anyhow::bail!("fcntl F_GETFD: {}", std::io::Error::last_os_error());
    }
    if unsafe { libc::fcntl(fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC) } == -1 {
        anyhow::bail!("fcntl F_SETFD: {}", std::io::Error::last_os_error());
    }
    Ok(())
}

fn hostname() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .or_else(|| {
            std::fs::read_to_string("/proc/sys/kernel/hostname")
                .ok()
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn now_iso() -> String {
    // Epoch seconds — enough provenance without pulling in a time crate. Field
    // is rendered as `acquired_epoch=<secs>`.
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_lock(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-lock-run-{name}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("gpu.lock")
    }

    #[test]
    fn run_scoped_propagates_child_exit_code_and_records_holder() {
        let path = temp_lock("code");
        // A successful command returns its own 0.
        let code = run_scoped_at(&path, "t-ok", 5, 1, &["true".into()], false).unwrap();
        assert_eq!(code, 0);
        // The holder line the run wrapper wrote names our pid + mode=run, and is
        // parseable by the same `holder=` reader `release`/`status` use.
        let holder = read_holder(&path).unwrap();
        assert!(holder.contains("mode=run"), "holder: {holder}");
        assert_eq!(read_holder_pid(&path), Some(std::process::id() as i32));

        // A failing command's non-zero code is surfaced. (Lock is free again —
        // the previous guard dropped at return, so this re-acquires cleanly.)
        let code = run_scoped_at(&path, "t-fail", 5, 1, &["false".into()], false).unwrap();
        assert_eq!(code, 1);

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn run_scoped_times_out_with_code_2_when_lock_is_held() {
        let path = temp_lock("busy");
        // Hold the lock out-of-band; the run wrapper must not steal it.
        let mut held = hipfire_lock::FlockGuard::open(&path).unwrap();
        assert!(held.try_lock().unwrap());

        let code = run_scoped_at(&path, "waiter", 1, 1, &["true".into()], false).unwrap();
        assert_eq!(code, 2, "a held lock must make `run` time out, not run the child");

        drop(held);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn run_scoped_reports_signal_death_as_128_plus_signal() {
        let path = temp_lock("signal");
        // Child SIGKILLs itself → 128 + 9 = 137.
        let code = run_scoped_at(
            &path,
            "t-sig",
            5,
            1,
            &["sh".into(), "-c".into(), "kill -9 $$".into()],
            false,
        )
        .unwrap();
        assert_eq!(code, 137);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn empty_command_is_rejected() {
        let path = temp_lock("empty");
        assert!(run_scoped_at(&path, "t", 5, 1, &[], false).is_err());
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }
}
