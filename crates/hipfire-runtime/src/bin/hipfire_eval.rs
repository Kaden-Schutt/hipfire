// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

fn main() {
    if let Err(e) = hipfire_runtime::eval_harness::run_from_env() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
