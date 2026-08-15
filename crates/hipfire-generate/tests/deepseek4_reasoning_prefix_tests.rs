// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Deepseek4 reasoning prefix tests.
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
use hipfire_runtime::prompt_frame::ThinkMode;

    

    #[test]
    fn parent_effort_prefixes_are_distinct_and_low_is_empty() {
        assert_eq!(hipfire_generate::common::deepseek4_reasoning_prefix(ThinkMode::NonThink), "");
        assert_eq!(hipfire_generate::common::deepseek4_reasoning_prefix(ThinkMode::Low), "");
        assert_eq!(
            hipfire_generate::common::deepseek4_reasoning_prefix(ThinkMode::High),
            hipfire_generate::common::DEEPSEEK4_REASONING_HIGH_PREFIX
        );
        assert_eq!(
            hipfire_generate::common::deepseek4_reasoning_prefix(ThinkMode::Max),
            hipfire_generate::common::DEEPSEEK4_REASONING_MAX_PREFIX
        );
        assert_ne!(
            hipfire_generate::common::DEEPSEEK4_REASONING_HIGH_PREFIX,
            hipfire_generate::common::DEEPSEEK4_REASONING_MAX_PREFIX
        );
        assert!(hipfire_generate::common::DEEPSEEK4_REASONING_HIGH_PREFIX.ends_with("\n\n"));
        assert!(hipfire_generate::common::DEEPSEEK4_REASONING_MAX_PREFIX.ends_with("\n\n"));
    }
