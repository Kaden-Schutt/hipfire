// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CDNA-owned compute backends.
//!
//! Code in this module must be admitted by an exact architecture proof. It
//! must not select an RDNA implementation as a fallback; callers that cannot
//! construct the proof object stay on their model backend's portable path.

pub mod gfx942;
