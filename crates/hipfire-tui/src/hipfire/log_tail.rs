// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Background tailer for `~/.hipfire/serve.log` — the daemon's stdout+stderr.
//!
//! A dedicated thread re-reads the tail of the file (bounded to the last 64 KiB,
//! so a huge log never blocks) only while the Logs tab is focused, exposing the
//! last N lines via a snapshot the UI thread clones each frame. Mirrors the
//! DashboardWorker pattern (Arc<Mutex<snapshot>> + atomic flags + wake channel +
//! Drop-shutdown). Honest states: Missing / Empty / Error are surfaced, never
//! faked.

use std::{
    fs::File,
    io::{ErrorKind, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Sender},
        Arc, Mutex,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

const MAX_LINES: usize = 400;
const TAIL_WINDOW: u64 = 64 * 1024;
const ACTIVE_POLL: Duration = Duration::from_millis(1000);
const IDLE_POLL: Duration = Duration::from_millis(4000);

/// Why the log isn't showing content (or that it is).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum LogStatus {
    #[default]
    Pending,
    Ok,
    Missing,
    Empty,
    Error(String),
}

/// The latest tail snapshot, mirrored to the UI thread.
#[derive(Clone, Debug, Default)]
pub struct LogSnapshot {
    pub lines: Vec<String>,
    pub status: LogStatus,
}

pub struct LogTailer {
    snapshot: Arc<Mutex<LogSnapshot>>,
    active: Arc<AtomicBool>,
    shutdown: Arc<AtomicBool>,
    wake: Sender<()>,
    handle: Option<JoinHandle<()>>,
}

impl LogTailer {
    pub fn spawn(path: PathBuf) -> Self {
        let snapshot = Arc::new(Mutex::new(LogSnapshot::default()));
        let active = Arc::new(AtomicBool::new(false));
        let shutdown = Arc::new(AtomicBool::new(false));
        let (wake, wake_rx) = mpsc::channel::<()>();
        let handle = {
            let snapshot = Arc::clone(&snapshot);
            let active = Arc::clone(&active);
            let shutdown = Arc::clone(&shutdown);
            thread::spawn(move || worker_loop(path, snapshot, active, shutdown, wake_rx))
        };
        Self {
            snapshot,
            active,
            shutdown,
            wake,
            handle: Some(handle),
        }
    }

    /// Mark the Logs tab focused/unfocused; waking the worker on activation so
    /// the tail is fresh immediately.
    pub fn set_active(&self, active: bool) {
        let was = self.active.swap(active, Ordering::SeqCst);
        if active && !was {
            let _ = self.wake.send(());
        }
    }

    /// Cheap clone of the latest tail snapshot for the UI thread.
    pub fn snapshot(&self) -> LogSnapshot {
        self.snapshot.lock().map(|g| g.clone()).unwrap_or_default()
    }
}

impl Drop for LogTailer {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        let _ = self.wake.send(());
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn worker_loop(
    path: PathBuf,
    snapshot: Arc<Mutex<LogSnapshot>>,
    active: Arc<AtomicBool>,
    shutdown: Arc<AtomicBool>,
    wake_rx: mpsc::Receiver<()>,
) {
    loop {
        if shutdown.load(Ordering::SeqCst) {
            return;
        }
        let is_active = active.load(Ordering::SeqCst);
        if is_active {
            let snap = read_tail(&path, MAX_LINES);
            if let Ok(mut g) = snapshot.lock() {
                *g = snap;
            }
        }
        let wait = if is_active { ACTIVE_POLL } else { IDLE_POLL };
        let _ = wake_rx.recv_timeout(wait);
    }
}

/// Read the last `max_lines` lines of `path`, bounded to the final `TAIL_WINDOW`
/// bytes so a large log never causes a big read. Pure (no shared state) so it is
/// directly unit-testable.
pub fn read_tail(path: &Path, max_lines: usize) -> LogSnapshot {
    let mut f = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == ErrorKind::NotFound => {
            return LogSnapshot {
                lines: Vec::new(),
                status: LogStatus::Missing,
            }
        }
        Err(e) => {
            return LogSnapshot {
                lines: Vec::new(),
                status: LogStatus::Error(e.to_string()),
            }
        }
    };
    let len = match f.metadata() {
        Ok(m) => m.len(),
        Err(e) => {
            return LogSnapshot {
                lines: Vec::new(),
                status: LogStatus::Error(e.to_string()),
            }
        }
    };
    if len == 0 {
        return LogSnapshot {
            lines: Vec::new(),
            status: LogStatus::Empty,
        };
    }
    let from = len.saturating_sub(TAIL_WINDOW);
    if let Err(e) = f.seek(SeekFrom::Start(from)) {
        return LogSnapshot {
            lines: Vec::new(),
            status: LogStatus::Error(e.to_string()),
        };
    }
    let mut bytes = Vec::new();
    if let Err(e) = f.read_to_end(&mut bytes) {
        return LogSnapshot {
            lines: Vec::new(),
            status: LogStatus::Error(e.to_string()),
        };
    }
    if bytes.is_empty() {
        // The file shrank (truncated/rotated) between the stat and the read, so
        // the seek landed past EOF. Report Empty rather than a dishonest "Ok with
        // no lines"; the next poll (<=1s) recovers the real content.
        return LogSnapshot {
            lines: Vec::new(),
            status: LogStatus::Empty,
        };
    }
    // Normalize line endings so a bare '\r' (some progress/spinner writers) does
    // not collapse many updates into one long line. (str::lines handles \r\n.)
    let text = String::from_utf8_lossy(&bytes)
        .replace("\r\n", "\n")
        .replace('\r', "\n");
    let mut lines: Vec<&str> = text.lines().collect();
    // Started mid-file: the first line is a partial fragment — drop it, UNLESS it
    // is the only line (a single line longer than the window), where keeping it
    // beats showing an empty log exactly when the user wants it most.
    if from > 0 && lines.len() > 1 {
        lines.remove(0);
    }
    let start = lines.len().saturating_sub(max_lines);
    let tail: Vec<String> = lines[start..].iter().map(|s| s.to_string()).collect();
    LogSnapshot {
        lines: tail,
        status: LogStatus::Ok,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("hipfire-logtail-{}-{name}", std::process::id()))
    }

    #[test]
    fn missing_file_is_honest() {
        let p = tmp("nope.log");
        let _ = std::fs::remove_file(&p);
        let s = read_tail(&p, 100);
        assert_eq!(s.status, LogStatus::Missing);
        assert!(s.lines.is_empty());
    }

    #[test]
    fn empty_file_is_empty_status() {
        let p = tmp("empty.log");
        File::create(&p).unwrap();
        let s = read_tail(&p, 100);
        assert_eq!(s.status, LogStatus::Empty);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn tails_last_n_lines() {
        let p = tmp("tail.log");
        let mut f = File::create(&p).unwrap();
        for i in 0..50 {
            writeln!(f, "line {i}").unwrap();
        }
        let s = read_tail(&p, 5);
        assert_eq!(s.status, LogStatus::Ok);
        assert_eq!(s.lines.len(), 5);
        assert_eq!(s.lines.last().unwrap(), "line 49");
        assert_eq!(s.lines.first().unwrap(), "line 45");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn mid_file_drops_partial_first_line_keeps_whole_lines() {
        // Write > TAIL_WINDOW so read_tail starts mid-file (from > 0).
        let p = tmp("midfile.log");
        let mut f = File::create(&p).unwrap();
        for i in 0..6000 {
            writeln!(
                f,
                "log line number {i} with padding text to lengthen the line"
            )
            .unwrap();
        }
        let s = read_tail(&p, 10);
        assert_eq!(s.status, LogStatus::Ok);
        assert_eq!(s.lines.len(), 10);
        assert!(s.lines.last().unwrap().contains("number 5999"));
        // The first returned line is a complete line, not a truncated fragment.
        assert!(s.lines.first().unwrap().starts_with("log line number"));
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn single_line_longer_than_window_is_kept_not_dropped() {
        let p = tmp("longline.log");
        let mut f = File::create(&p).unwrap();
        let big = "X".repeat(TAIL_WINDOW as usize + 1000); // > window, no '\n' in window
        write!(f, "{big}").unwrap();
        let s = read_tail(&p, 10);
        assert_eq!(s.status, LogStatus::Ok);
        assert_eq!(
            s.lines.len(),
            1,
            "the sole long line is kept, not dropped to empty"
        );
        assert!(!s.lines[0].is_empty());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn last_line_without_trailing_newline_is_retained() {
        let p = tmp("noeol.log");
        let mut f = File::create(&p).unwrap();
        writeln!(f, "first").unwrap();
        write!(f, "second-no-newline").unwrap();
        let s = read_tail(&p, 10);
        assert_eq!(s.lines.last().unwrap(), "second-no-newline");
        let _ = std::fs::remove_file(&p);
    }
}
