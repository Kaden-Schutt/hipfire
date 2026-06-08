// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use std::sync::OnceLock;

static LOGGING_INIT: OnceLock<()> = OnceLock::new();

/// Initialize structured logging for runtime binaries.
///
/// The daemon reserves stdout for JSONL IPC, so this writes to stderr for now.
/// File sinks should be added here later so daemon and server business logic
/// does not learn about production log destinations.
pub fn init_stderr_logging(component: &'static str) {
    LOGGING_INIT.get_or_init(|| {
        let filter = tracing_subscriber::EnvFilter::try_from_env("HIPFIRE_LOG")
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(true)
            .with_writer(std::io::stderr)
            .init();

        tracing::info!(component, sink = "stderr", "logging initialized");
    });
}
