// SPDX-License-Identifier: Apache-2.0
// hipfire — native GPU mutex (the legacy scripts/gpu-lock.sh shell adapter has been removed).
//
//! `hipfire gpu-lock {acquire,release,status}` — a flock(2)-backed GPU mutex for
//! multi-agent coordination, owned by the engine instead of a shell script.
//!
//! Mechanism (flock + lock-holder helper):
//! - `acquire <label>` opens the lockfile, takes a blocking `LOCK_EX` (polling
//!   with a `busy` message + optional timeout), writes holder metadata, then
//!   spawns a detached `setsid` holder (`gpu-lock hold`, hidden) that INHERITS
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
//! NB: the lockfile is never unlinked — unlinking a flock'd file lets the next
//! acquirer lock a different inode and yields two simultaneous holders.

use std::os::fd::{FromRawFd, OwnedFd};
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use clap::{Args, Subcommand};

#[derive(Debug, Args)]
pub struct GpuLockArgs {
    #[command(subcommand)]
    action: GpuLockAction,
}

#[derive(Debug, Subcommand)]
enum GpuLockAction {
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
    hipfire_lock::gpu_lock_path()
}

fn pid_alive(pid: i32) -> bool {
    if pid <= 1 {
        return false;
    }
    // kill(pid, 0): 0 ⇒ alive; EPERM ⇒ alive (not permitted); ESRCH ⇒ gone.
    unsafe { libc::kill(pid, 0) == 0 || *libc::__errno_location() == libc::EPERM }
}

pub fn run(args: GpuLockArgs) -> anyhow::Result<()> {
    match args.action {
        GpuLockAction::Acquire {
            label,
            watch_pid,
            timeout_secs,
            poll_secs,
        } => acquire(&label, watch_pid, timeout_secs, poll_secs.max(1)),
        GpuLockAction::Release => release(),
        GpuLockAction::Status => {
            println!("{}", status_line());
            Ok(())
        }
        GpuLockAction::Hold {
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
        "gpu-lock",
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
