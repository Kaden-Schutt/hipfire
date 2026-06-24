// SPDX-License-Identifier: MIT OR Apache-2.0
// hipfire — KVarN codec + deferred KV-compaction, extracted to a leaf lib so both
// the quantizer bin and the engine read path can use them (Phase 2b crate move).
pub mod conv;
pub mod fwht;
pub mod kv_compact;
pub mod kvarn;
