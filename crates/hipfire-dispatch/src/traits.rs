// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

pub trait KernelFamily: Send + Sync {
    fn name(&self) -> &'static str;
}
