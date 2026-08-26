// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Mtp host timing contract.
//!
//! Moved out of `hipfire-daemon`'s `main.rs`. Compiled into a bin crate these
//! never appeared as their own test target; as integration tests they are
//! reported individually.

#![allow(unused_imports, dead_code, clippy::all)]

use hipfire_engine::emit::*;
use hipfire_engine::scheduler::*;
use hipfire_engine::terminal::*;
use hipfire_generate::ar::*;
use hipfire_generate::batch::*;
use hipfire_generate::common::*;

    use hipfire_generate::qwen::{attach_mtp_window_timings, mtp_window_timing_kind, mtp_window_timing_record};

    #[test]
    fn route_kind_covers_ngram_mtp_and_ar() {
        // Ngram hit wins regardless of retirement latch.
        assert_eq!(hipfire_generate::qwen::mtp_window_timing_kind(true, true, false), "ngram");
        assert_eq!(hipfire_generate::qwen::mtp_window_timing_kind(true, true, true), "ngram");
        assert_eq!(hipfire_generate::qwen::mtp_window_timing_kind(true, false, false), "ngram");
        // Miss after retirement → AR (trunk-only k=0).
        assert_eq!(hipfire_generate::qwen::mtp_window_timing_kind(false, true, true), "ar");
        // Miss before retirement / ngram off → native MTP.
        assert_eq!(hipfire_generate::qwen::mtp_window_timing_kind(false, true, false), "mtp");
        assert_eq!(hipfire_generate::qwen::mtp_window_timing_kind(false, false, false), "mtp");
        assert_eq!(hipfire_generate::qwen::mtp_window_timing_kind(false, false, true), "mtp");
    }

    #[test]
    fn timing_record_preserves_exact_wire_fields() {
        let rec = hipfire_generate::qwen::mtp_window_timing_record("ngram", 11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12);
        let obj = rec.as_object().expect("object");
        let expected = [
            "kind",
            "wall_us",
            "draft_lookup_us",
            "launch_us",
            "h2d_us",
            "d2h_us",
            "d2d_us",
            "memset_us",
            "stream_sync_us",
            "event_sync_us",
            "device_sync_us",
            "graph_launch_us",
        ];
        assert_eq!(obj.len(), expected.len());
        for key in expected {
            assert!(obj.contains_key(key), "missing wire field {key}");
        }
        assert_eq!(rec["kind"], "ngram");
        assert_eq!(rec["wall_us"], 11);
        assert_eq!(rec["draft_lookup_us"], 2);
        assert_eq!(rec["launch_us"], 3);
        assert_eq!(rec["h2d_us"], 4);
        assert_eq!(rec["d2h_us"], 5);
        assert_eq!(rec["d2d_us"], 6);
        assert_eq!(rec["memset_us"], 7);
        assert_eq!(rec["stream_sync_us"], 8);
        assert_eq!(rec["event_sync_us"], 9);
        assert_eq!(rec["device_sync_us"], 10);
        assert_eq!(rec["graph_launch_us"], 12);
        // All eleven numeric fields are nonnegative integers on the wire.
        for key in [
            "wall_us",
            "draft_lookup_us",
            "launch_us",
            "h2d_us",
            "d2h_us",
            "d2d_us",
            "memset_us",
            "stream_sync_us",
            "event_sync_us",
            "device_sync_us",
            "graph_launch_us",
        ] {
            assert!(rec[key].as_u64().is_some(), "{key} must be u64");
        }
    }

    #[test]
    fn attach_omits_field_when_disabled_preserves_order_when_enabled() {
        let r0 = hipfire_generate::qwen::mtp_window_timing_record("mtp", 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let r1 = hipfire_generate::qwen::mtp_window_timing_record("ngram", 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let r2 = hipfire_generate::qwen::mtp_window_timing_record("ar", 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let ordered = vec![r0.clone(), r1.clone(), r2.clone()];

        let mut disabled = serde_json::json!({"tokens": 1});
        hipfire_generate::qwen::attach_mtp_window_timings(&mut disabled, false, ordered.clone());
        assert!(
            disabled.get("mtp_window_timings").is_none(),
            "disabled must omit the field entirely"
        );

        let mut enabled = serde_json::json!({"tokens": 1});
        hipfire_generate::qwen::attach_mtp_window_timings(&mut enabled, true, ordered);
        let arr = enabled["mtp_window_timings"]
            .as_array()
            .expect("enabled attaches array");
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0]["kind"], "mtp");
        assert_eq!(arr[1]["kind"], "ngram");
        assert_eq!(arr[2]["kind"], "ar");
        assert_eq!(arr[0]["wall_us"], 1);
        assert_eq!(arr[1]["wall_us"], 2);
        assert_eq!(arr[2]["wall_us"], 3);
    }
