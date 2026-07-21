// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Background runners for Models-tab actions: pull (with live progress) and rm.
//!
//! Both invoke the native `hipfire` binary on a dedicated thread so the UI
//! thread never blocks. `pull` streams the CLI's stderr progress line (which is
//! carriage-return-updated, so we split on '\r' AND '\n') and reports a parsed
//! percentage; `remove` runs `rm <tag> --yes` (the TUI does its own y/n confirm
//! first) and reports a single outcome. Events arrive on mpsc channels the App
//! drains each frame.

use std::{
    io::Read,
    process::Stdio,
    sync::mpsc::{self, Receiver, Sender},
    thread,
};

use crate::hipfire::native_cli_command;

/// Streamed events from a `pull` run.
#[derive(Clone, Debug, PartialEq)]
pub enum PullEvent {
    /// A progress update: parsed percent (None if not yet sampled) + the raw
    /// CLI progress line (bar / rate / ETA / bytes) for display.
    Progress {
        percent: Option<f64>,
        line: String,
    },
    Done,
    Failed(String),
}

/// Outcome of a one-shot model action (currently `remove`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RmOutcome {
    Ok(String),
    Failed(String),
}

/// Spawn native `hipfire pull <tag>`, streaming progress on the returned
/// receiver and finishing with Done or Failed.
pub fn pull(tag: String) -> Receiver<PullEvent> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || pull_inner(tag, tx));
    rx
}

/// Spawn native `hipfire rm <tag> --yes`; the single outcome arrives on the
/// returned receiver.
pub fn remove(tag: String) -> Receiver<RmOutcome> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(remove_inner(&tag));
    });
    rx
}

/// Extract the percentage from a CLI progress line like
/// `[████▏      ]  45.2%   8.1 MB/s   ETA 1m12s   123/272 MB` — find the '%' and
/// read the trailing number before it (robust to the bar's internal spaces).
pub fn parse_percent(line: &str) -> Option<f64> {
    let pct = line.find('%')?;
    let num: String = line[..pct]
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    num.parse::<f64>().ok()
}

fn pull_inner(tag: String, tx: Sender<PullEvent>) {
    let mut cmd = match native_cli_command() {
        Some(c) => c,
        None => {
            let _ = tx.send(PullEvent::Failed(
                "native hipfire binary not found (set HIPFIRE_CLI_BIN or install hipfire)".into(),
            ));
            return;
        }
    };
    let mut child = match cmd
        .arg("pull")
        .arg(&tag)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            let _ = tx.send(PullEvent::Failed(format!("spawn: {e}")));
            return;
        }
    };

    // Drain stdout on its own thread so a full pipe can't deadlock the child.
    let stdout_handle = child.stdout.take().map(|mut so| {
        thread::spawn(move || {
            let mut s = String::new();
            let _ = so.read_to_string(&mut s);
            s
        })
    });

    // Read stderr byte-by-byte, flushing a "line" on each '\r' or '\n' (the CLI
    // overwrites one line with '\r' and only emits '\n' at completion). Retain
    // the most-recent non-empty line so a failure reports the real cause — the
    // CLI writes its errors to stderr (the same stream as the progress).
    let mut last_line = String::new();
    if let Some(mut stderr) = child.stderr.take() {
        let mut buf: Vec<u8> = Vec::new();
        let mut byte = [0u8; 1];
        loop {
            match stderr.read(&mut byte) {
                Ok(0) => break, // EOF
                Ok(_) => {
                    if byte[0] == b'\r' || byte[0] == b'\n' {
                        if let Some(line) = flush_progress(&mut buf, &tx) {
                            last_line = line;
                        }
                    } else {
                        buf.push(byte[0]);
                    }
                }
                Err(_) => break,
            }
        }
        if let Some(line) = flush_progress(&mut buf, &tx) {
            last_line = line;
        }
    }

    let status = child.wait();
    let stdout = stdout_handle
        .map(|h| h.join().unwrap_or_default())
        .unwrap_or_default();
    match status {
        Ok(s) if s.success() => {
            let _ = tx.send(PullEvent::Done);
        }
        Ok(_) => {
            // Prefer the last stderr line (where the CLI writes errors); fall
            // back to stdout, then a generic message.
            let msg = if !last_line.is_empty() {
                last_line
            } else {
                stdout
                    .lines()
                    .rev()
                    .find(|l| !l.trim().is_empty())
                    .unwrap_or("pull failed")
                    .trim()
                    .to_string()
            };
            let _ = tx.send(PullEvent::Failed(msg));
        }
        Err(e) => {
            let _ = tx.send(PullEvent::Failed(format!("wait: {e}")));
        }
    }
}

/// Flush the accumulated stderr line: send it as a Progress event and return it
/// (so the caller can keep the latest line for error reporting). None if empty.
fn flush_progress(buf: &mut Vec<u8>, tx: &Sender<PullEvent>) -> Option<String> {
    if buf.is_empty() {
        return None;
    }
    let line = String::from_utf8_lossy(buf).trim().to_string();
    buf.clear();
    if line.is_empty() {
        return None;
    }
    let percent = parse_percent(&line);
    let _ = tx.send(PullEvent::Progress {
        percent,
        line: line.clone(),
    });
    Some(line)
}

fn remove_inner(tag: &str) -> RmOutcome {
    let mut cmd = match native_cli_command() {
        Some(c) => c,
        None => {
            return RmOutcome::Failed(
                "native hipfire binary not found (set HIPFIRE_CLI_BIN or install hipfire)".into(),
            )
        }
    };
    let output = cmd.arg("rm").arg(tag).arg("--yes").output();
    match output {
        Ok(o) if o.status.success() => RmOutcome::Ok(format!("removed {tag}")),
        Ok(o) => {
            let err = String::from_utf8_lossy(&o.stderr);
            let tail = err
                .lines()
                .rev()
                .find(|l| !l.trim().is_empty())
                .unwrap_or("rm failed")
                .trim();
            RmOutcome::Failed(format!("rm {tag}: {tail}"))
        }
        Err(e) => RmOutcome::Failed(format!("rm {tag}: spawn failed: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_percent_handles_real_line() {
        assert_eq!(
            parse_percent("[████▏      ]  45.2%   8.1 MB/s   ETA 1m12s   123/272 MB"),
            Some(45.2)
        );
        assert_eq!(
            parse_percent("[          ]   0.0%   —   123/272 MB"),
            Some(0.0)
        );
        assert_eq!(parse_percent("[██████████] 100.0% done"), Some(100.0));
        // Unknown-total downloads render "?%" — must yield None (drives the
        // PullEvent::Progress { percent: None } path).
        assert_eq!(parse_percent("[          ]   ?%   —   ?/? MB"), None);
        assert_eq!(parse_percent("no percent here"), None);
    }

    #[test]
    fn flush_progress_parses_sends_and_returns_line() {
        let (tx, rx) = mpsc::channel();
        let mut buf = b"[bar] 42.0% 8 MB/s".to_vec();
        let line = flush_progress(&mut buf, &tx);
        assert_eq!(line.as_deref(), Some("[bar] 42.0% 8 MB/s"));
        assert!(buf.is_empty(), "buffer is consumed");
        match rx.try_recv() {
            Ok(PullEvent::Progress { percent, line }) => {
                assert_eq!(percent, Some(42.0));
                assert!(line.contains("42.0%"));
            }
            other => panic!("expected a Progress event, got {other:?}"),
        }
        // An empty buffer flushes nothing and emits no event.
        let mut empty = Vec::new();
        assert_eq!(flush_progress(&mut empty, &tx), None);
        assert!(rx.try_recv().is_err());
    }
}
