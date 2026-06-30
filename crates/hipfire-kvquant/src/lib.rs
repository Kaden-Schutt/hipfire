// SPDX-License-Identifier: MIT OR Apache-2.0
// hipfire — KVarN codec + deferred KV-compaction, extracted to a leaf lib so both
// the quantizer bin and the engine read path can use them (Phase 2b crate move).
// `conv` (f16↔f32) and `fwht` (per-256 signed FWHT) moved to the shared
// `hipfire-primitives` leaf — they were never KV-specific. Import them from
// there.
pub mod kv_compact;
pub mod kvarn;
