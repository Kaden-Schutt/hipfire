// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Reusable `flock(2)` file-lock primitive.
//!
//! The single home for hipfire's `flock`-based mutexes — previously hand-rolled
//! separately in the daemon singleton lock (`~/.hipfire/daemon.pid`) and the GPU
//! mutex (`hipfire gpu-lock` / `scripts/gpu-lock.sh`, on `/tmp/hipfire-gpu.lock`).
//! Both reduce to: open a lockfile, take `LOCK_EX` (non-blocking or blocking with
//! a poll/timeout), optionally write a human-readable holder line, and rely on
//! the kernel to drop the lock when the holding fd closes (incl. SIGKILL) — so
//! stale lockfile *contents* are harmless; the open fd is what holds the lock.
//!
//! `flock` is a kernel primitive, so a [`FlockGuard`] interoperates with **any**
//! process holding `flock` on the same inode regardless of language — which is
//! why `scripts/gpu-lock.sh` (bash) and a Python `fcntl.flock` wrapper share the
//! same mutex as the Rust callers without any FFI: same path + same syscall.
//!
//! Unix only. On non-unix the guard opens the file but performs no kernel
//! locking (matching the daemon's prior best-effort Windows behavior).

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// A held (or holdable) `flock` on a lockfile. Dropping it closes the fd, which
/// releases the kernel lock — keep it alive for as long as the lock should hold.
#[derive(Debug)]
pub struct FlockGuard {
    file: File,
    path: PathBuf,
    locked: bool,
}

/// Result of a non-blocking [`probe`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LockState {
    /// Nobody holds the lock right now.
    Free,
    /// Held; carries the holder line written by [`FlockGuard::write_holder`]
    /// (empty string if the holder wrote none).
    Busy(String),
}

impl FlockGuard {
    /// Open (creating if needed, mode `0600`, parent dirs created) the lockfile
    /// at `path` **without** taking the lock yet. Use [`try_lock`](Self::try_lock)
    /// or [`lock_blocking`](Self::lock_blocking) next.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let mut opts = OpenOptions::new();
        opts.read(true).write(true).create(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let file = opts.open(&path)?;
        Ok(Self {
            file,
            path,
            locked: false,
        })
    }

    /// Try to take `LOCK_EX | LOCK_NB`. `Ok(true)` = acquired (the guard now
    /// holds it), `Ok(false)` = currently held by someone else, `Err` = I/O.
    pub fn try_lock(&mut self) -> io::Result<bool> {
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let rc = unsafe { libc::flock(self.file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
            if rc == 0 {
                self.locked = true;
                return Ok(true);
            }
            let err = io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::EWOULDBLOCK) {
                return Ok(false);
            }
            Err(err)
        }
        #[cfg(not(unix))]
        {
            // No kernel locking off-unix; opening succeeded, treat as acquired.
            self.locked = true;
            Ok(true)
        }
    }

    /// Block until the lock is acquired, polling every `interval` and invoking
    /// `on_busy(holder)` once per poll while it stays held. Returns `Ok(true)`
    /// when acquired, `Ok(false)` if `timeout` elapses first (`None` = wait
    /// forever).
    pub fn lock_blocking(
        &mut self,
        interval: Duration,
        timeout: Option<Duration>,
        mut on_busy: impl FnMut(&str),
    ) -> io::Result<bool> {
        let start = Instant::now();
        loop {
            if self.try_lock()? {
                return Ok(true);
            }
            if let Some(t) = timeout {
                if start.elapsed() >= t {
                    return Ok(false);
                }
            }
            on_busy(&read_holder_line(&self.path).unwrap_or_default());
            std::thread::sleep(interval);
        }
    }

    /// Truncate the lockfile and write a single holder line (e.g. a pid or a
    /// `pid host label` string). Preserves the held lock — `flock` is on the
    /// open description, not the contents.
    pub fn write_holder(&mut self, line: &str) -> io::Result<()> {
        self.file.set_len(0)?;
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(line.as_bytes())?;
        if !line.ends_with('\n') {
            self.file.write_all(b"\n")?;
        }
        self.file.flush()
    }

    /// The holder line currently in the lockfile, if any (trimmed).
    pub fn holder(&self) -> Option<String> {
        read_holder_line(&self.path)
    }

    /// Raw fd of the held lockfile — for callers that pass it to a detached
    /// holder process that inherits the open description (the `gpu-lock`
    /// acquire-returns-but-holder-keeps-it design).
    #[cfg(unix)]
    pub fn raw_fd(&self) -> std::os::unix::io::RawFd {
        use std::os::unix::io::AsRawFd;
        self.file.as_raw_fd()
    }

    /// Whether this guard currently holds the lock.
    pub fn is_locked(&self) -> bool {
        self.locked
    }

    /// The lockfile path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Non-blocking check of whether `path` is currently flock'd, **without**
/// holding it (takes `LOCK_EX|LOCK_NB`, immediately unlocks on success). On a
/// busy lock, returns the holder line.
pub fn probe(path: impl AsRef<Path>) -> io::Result<LockState> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(LockState::Free);
    }
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let fd = file.as_raw_fd();
        let rc = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
        if rc == 0 {
            unsafe { libc::flock(fd, libc::LOCK_UN) };
            Ok(LockState::Free)
        } else {
            let err = io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::EWOULDBLOCK) {
                Ok(LockState::Busy(read_holder_line(path).unwrap_or_default()))
            } else {
                Err(err)
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(LockState::Free)
    }
}

fn read_holder_line(path: &Path) -> Option<String> {
    let mut s = String::new();
    File::open(path).ok()?.read_to_string(&mut s).ok()?;
    let t = s.trim();
    if t.is_empty() {
        None
    } else {
        Some(t.to_string())
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    fn tmp_lockfile(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "hipfire-lock-test-{}-{}.lock",
            std::process::id(),
            name
        ))
    }

    #[test]
    fn second_guard_sees_busy_then_free_after_drop() {
        let p = tmp_lockfile("excl");
        let _ = std::fs::remove_file(&p);

        let mut a = FlockGuard::open(&p).unwrap();
        assert!(a.try_lock().unwrap(), "first acquire should succeed");
        assert!(a.is_locked());

        // A second guard on the same path can't take it while `a` holds it.
        let mut b = FlockGuard::open(&p).unwrap();
        assert!(!b.try_lock().unwrap(), "second acquire should see busy");

        // probe() agrees it's busy.
        assert!(matches!(probe(&p).unwrap(), LockState::Busy(_)));

        drop(a); // releases the kernel lock
        assert!(b.try_lock().unwrap(), "acquire should succeed after drop");
        assert!(matches!(probe(&p).unwrap(), LockState::Free) || b.is_locked());

        drop(b);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn holder_line_roundtrips_under_lock() {
        let p = tmp_lockfile("holder");
        let _ = std::fs::remove_file(&p);

        let mut a = FlockGuard::open(&p).unwrap();
        assert!(a.try_lock().unwrap());
        a.write_holder("4242 k9lin speed-gate").unwrap();
        assert_eq!(a.holder().as_deref(), Some("4242 k9lin speed-gate"));

        // Writing the holder doesn't drop the lock.
        let mut b = FlockGuard::open(&p).unwrap();
        assert!(!b.try_lock().unwrap());
        if let LockState::Busy(h) = probe(&p).unwrap() {
            assert_eq!(h, "4242 k9lin speed-gate");
        } else {
            panic!("expected busy with holder line");
        }

        drop(a);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn lock_blocking_times_out_when_held() {
        let p = tmp_lockfile("timeout");
        let _ = std::fs::remove_file(&p);

        let mut a = FlockGuard::open(&p).unwrap();
        assert!(a.try_lock().unwrap());

        let mut b = FlockGuard::open(&p).unwrap();
        let mut busy_calls = 0;
        let got = b
            .lock_blocking(
                Duration::from_millis(10),
                Some(Duration::from_millis(40)),
                |_| {
                    busy_calls += 1;
                },
            )
            .unwrap();
        assert!(!got, "should time out while held");
        assert!(busy_calls >= 1, "on_busy should fire at least once");

        drop(a);
        let _ = std::fs::remove_file(&p);
    }
}
