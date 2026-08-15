// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Serve fault inject tests.
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
#[cfg(feature = "serve-fault-inject")]
use hipfire_generate::ar::{arm_fault_after_prefill, take_fault_after_prefill};


    #[cfg(feature = "serve-fault-inject")]
    #[test]
    fn fault_inject_routes_qwen35_only() {
        assert_eq!(
            hipfire_runtime::reset_core::fault_inject_eligible_routes("qwen35"),
            &["qwen_ar", "qwen_dflash"][..]
        );
        assert!(hipfire_runtime::reset_core::fault_inject_eligible_routes("deepseek4").is_empty());
        assert!(hipfire_runtime::reset_core::fault_inject_eligible_routes("llama").is_empty());
    }

    #[cfg(feature = "serve-fault-inject")]
    #[test]
    fn one_shot_arm_take_clears() {
        arm_fault_after_prefill(true);
        assert!(take_fault_after_prefill());
        assert!(!take_fault_after_prefill());
        arm_fault_after_prefill(false);
        assert!(!take_fault_after_prefill());
    }

    #[cfg(feature = "serve-fault-inject")]
    #[test]
    fn retry_eligible_only_qwen35() {
        assert!(model_retry_reset_eligible(5));
        assert!(model_retry_reset_eligible(6));
        assert!(!model_retry_reset_eligible(9)); // deepseek4
        assert!(!model_retry_reset_eligible(0)); // llama
    }
