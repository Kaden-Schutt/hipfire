// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The generic autoregressive generate path.
//!
//! STUB — filled by the D3 tail.
//!
//! This is the fallback every model without a specialised route takes, and the
//! last daemon code that manipulates architecture types directly. It is also
//! the reason the daemon still names `qwen35::forward_scratch`,
//! `qwen35::prepare_scratch_inputs`, `llama::forward_scratch` and friends.
//!
//! Note for whoever fills this: the acceptance test is NOT the
//! `hipfire_arch_` grep. An earlier attempt drove that to zero by re-exporting
//! the architecture crates through another module, which changed nothing. The
//! measure that matters is whether the daemon still names an architecture type
//! at all.
