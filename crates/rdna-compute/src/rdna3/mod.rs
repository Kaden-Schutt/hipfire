// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! RDNA3-owned compute backends.
//!
//! Exact-device proof objects keep product code from selecting an
//! architecture-specific candidate through a broad gfx11 feature predicate.

pub mod gfx1100;
