// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Vl adaptive admission tests.
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

    use hipfire_generate::vision::vl_no_eviction_kv_cap;

    #[test]
    fn adaptive_admits_against_max_seq_not_start_tier_physical() {
        // physical_cap may equal max_seq at load, but the important case is
        // that adaptive never silently shrinks admission to start-tier cap.
        let physical_cap = 8192;
        let max_seq = 32768;
        assert_eq!(
            vl_no_eviction_kv_cap(physical_cap, max_seq, true),
            max_seq,
            "adaptive VL must admit against floor-tier max_seq"
        );
        assert_eq!(
            vl_no_eviction_kv_cap(physical_cap, max_seq, false),
            physical_cap,
            "non-adaptive VL keeps physical_cap contract"
        );
    }

    #[test]
    fn equal_caps_identical_either_mode() {
        assert_eq!(vl_no_eviction_kv_cap(4096, 4096, false), 4096);
        assert_eq!(vl_no_eviction_kv_cap(4096, 4096, true), 4096);
    }
