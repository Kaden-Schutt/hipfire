// SPDX-License-Identifier: MIT
// Copyright (c) 2026 alpineq
// hipfire — see LICENSE and NOTICE in the project root.

//! `multi_gpu` moved to the leaf crate `hipfire-hardware` (Phase 0 of the
//! device-mesh work) so `hipfire-dispatch` can depend on the collective /
//! `Gpus` layer without a dispatch→runtime cycle. Re-exported here so every
//! existing `hipfire_runtime::multi_gpu::…` path keeps working unchanged.
//!
//! The implementation — `Gpus`, `BoundaryEvent`, the peer-reduce scratch
//! lease (`PeerReduceScratchLease`, `acquire_peer_reduce_scratch`,
//! `release_peer_reduce_scratch`, `peer_reduce_scratch_bytes_per_rank`,
//! `peer_reduce_scratch_total_bytes`), the rooted / leased peer reduces,
//! the reusable rank barriers / handoffs, the gfx1201 TP-graph signal tape,
//! and the TP band-start helper — lives in `hipfire-hardware`.
pub use hipfire_hardware::*;
