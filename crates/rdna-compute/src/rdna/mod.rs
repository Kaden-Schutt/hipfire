// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! RDNA-owned exact-device compute backends.
//!
//! Operations in this module require an exact architecture proof. Generic
//! model code cannot select these kernels through a broad RDNA capability or
//! an environment variable.

pub mod gfx1201;
