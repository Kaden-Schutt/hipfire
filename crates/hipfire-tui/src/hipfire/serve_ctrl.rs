// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Background runner for serve lifecycle commands (start / stop / restart).
//!
//! Each command invokes native `hipfire <args>` on a dedicated thread,
//! so the UI thread never blocks on a spawn or on the CLI's `/health` readiness
//! poll. The single outcome arrives on an mpsc channel that the App drains each
//! frame — the same pattern as chat streaming. The live serve up/down state is
//! reflected separately by the DashboardWorker `/health` probe, so this runner
//! only reports the command's own success/failure.

use std::{
    sync::mpsc::{self, Receiver},
    thread,
};

use crate::hipfire::native_cli_command;

/// Result of a serve lifecycle command, surfaced to the user as a toast.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ServeOutcome {
    Ok(String),
    Failed(String),
}

/// A serve lifecycle action. `args()` maps it to the native CLI argv.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServeAction {
    Start,
    Stop,
    Restart,
}

impl ServeAction {
    pub fn label(self) -> &'static str {
        match self {
            ServeAction::Start => "start",
            ServeAction::Stop => "stop",
            ServeAction::Restart => "restart",
        }
    }

    /// Argv for the native CLI. Start/restart detach (`-d`) so the daemon outlives
    /// the command; stop is synchronous.
    pub fn args(self) -> Vec<String> {
        match self {
            ServeAction::Start => vec!["serve".into(), "-d".into()],
            ServeAction::Stop => vec!["stop".into()],
            ServeAction::Restart => vec!["restart".into(), "-d".into()],
        }
    }
}

/// Spawn native `hipfire <action.args()>` on a background thread. The single
/// outcome is sent on the returned receiver when the command exits. Dropping the
/// receiver (e.g. on quit) silently discards the result; the spawned daemon is
/// unaffected.
pub fn run(action: ServeAction) -> Receiver<ServeOutcome> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(run_inner(action));
    });
    rx
}

fn run_inner(action: ServeAction) -> ServeOutcome {
    let mut cmd = match native_cli_command() {
        Some(c) => c,
        None => {
            return ServeOutcome::Failed(
                "native hipfire binary not found (set HIPFIRE_CLI_BIN or install hipfire)".into(),
            )
        }
    };
    let output = cmd.args(action.args()).output();
    match output {
        Ok(o) if o.status.success() => ServeOutcome::Ok(format!("serve {}: done", action.label())),
        Ok(o) => {
            let err = String::from_utf8_lossy(&o.stderr);
            let tail = err
                .lines()
                .rev()
                .find(|l| !l.trim().is_empty())
                .unwrap_or("command failed")
                .trim();
            ServeOutcome::Failed(format!("serve {}: {tail}", action.label()))
        }
        Err(e) => ServeOutcome::Failed(format!("serve {}: spawn failed: {e}", action.label())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_argv_is_stable() {
        assert_eq!(ServeAction::Start.args(), vec!["serve", "-d"]);
        assert_eq!(ServeAction::Stop.args(), vec!["stop"]);
        assert_eq!(ServeAction::Restart.args(), vec!["restart", "-d"]);
        assert_eq!(ServeAction::Start.label(), "start");
    }
}
