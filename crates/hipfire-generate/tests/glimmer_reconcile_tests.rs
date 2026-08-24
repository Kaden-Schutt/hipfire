// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Glimmer reconcile tests.
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

/// The model card exposes exactly four reasoning strengths — low / medium / high / xhigh —
/// as a system-prompt directive. All four must be reachable, and an EXPLICIT
/// `reasoning_effort` must beat whatever token cap happens to be set.
/// Strength is raw semantic effort; `max_think_tokens` is orthogonal force-close.
#[test]
fn reasoning_strength_covers_all_four_card_levels() {
    // Explicit semantic effort drives strength directly; cap does not influence.
    assert_eq!(
        hipfire_generate::dense::glimmer_reasoning_strength(Some("high"), 1),
        "high"
    );
    assert_eq!(
        hipfire_generate::dense::glimmer_reasoning_strength(Some("xhigh"), 1),
        "xhigh"
    );
    assert_eq!(
        hipfire_generate::dense::glimmer_reasoning_strength(Some("max"), 1),
        "xhigh"
    );
    assert_eq!(
        hipfire_generate::dense::glimmer_reasoning_strength(Some("low"), 512),
        "low"
    );
    assert_eq!(
        hipfire_generate::dense::glimmer_reasoning_strength(Some("medium"), 2048),
        "medium"
    );
    // medium remains medium when uncapped – cap is force-close only, not strength.
    assert_eq!(
        hipfire_generate::dense::glimmer_reasoning_strength(Some("medium"), 0),
        "medium"
    );
    assert_eq!(
        hipfire_generate::dense::glimmer_reasoning_strength(Some("medium"), 8192),
        "medium"
    );
    assert_eq!(
        hipfire_generate::dense::glimmer_reasoning_strength(Some("high"), 8192),
        "high"
    );
    assert_eq!(
        hipfire_generate::dense::glimmer_reasoning_strength(Some("low"), 0),
        "low"
    );
    // No effort (auto) => template default high regardless of cap
    assert_eq!(
        hipfire_generate::dense::glimmer_reasoning_strength(None, 1),
        "high"
    );
    assert_eq!(
        hipfire_generate::dense::glimmer_reasoning_strength(None, 0),
        "high"
    );
    assert_eq!(
        hipfire_generate::dense::glimmer_reasoning_strength(None, 512),
        "high"
    );
    // Glimmer has no non-thinking mode; strength is always one of the four
    let produced: std::collections::BTreeSet<&str> = [
        hipfire_generate::dense::glimmer_reasoning_strength(Some("low"), 0),
        hipfire_generate::dense::glimmer_reasoning_strength(Some("medium"), 0),
        hipfire_generate::dense::glimmer_reasoning_strength(Some("high"), 0),
        hipfire_generate::dense::glimmer_reasoning_strength(Some("xhigh"), 0),
    ]
    .into_iter()
    .collect();
    assert_eq!(
        produced,
        ["high", "low", "medium", "xhigh"].into_iter().collect()
    );
}
#[test]
fn mirror_action_aligned() {
    assert_eq!(
        hipfire_generate::dense::glimmer_mirror_action(5, 5),
        hipfire_generate::dense::GlimmerMirrorAction::Aligned
    );
    assert_eq!(
        hipfire_generate::dense::glimmer_mirror_action(0, 0),
        hipfire_generate::dense::GlimmerMirrorAction::Aligned
    );
}
#[test]
fn mirror_action_truncate() {
    assert_eq!(
        hipfire_generate::dense::glimmer_mirror_action(5, 3),
        hipfire_generate::dense::GlimmerMirrorAction::TruncateMirror(3)
    );
    assert_eq!(
        hipfire_generate::dense::glimmer_mirror_action(10, 0),
        hipfire_generate::dense::GlimmerMirrorAction::TruncateMirror(0)
    );
}
#[test]
fn mirror_action_rollback() {
    assert_eq!(
        hipfire_generate::dense::glimmer_mirror_action(3, 5),
        hipfire_generate::dense::GlimmerMirrorAction::RollbackCursor(3)
    );
    assert_eq!(
        hipfire_generate::dense::glimmer_mirror_action(0, 5),
        hipfire_generate::dense::GlimmerMirrorAction::RollbackCursor(0)
    );
}
// glimmer_hidden_keep_len removed with device-capture session API cutover.
#[test]
fn glimmer_turn_key_ordinal_salts() {
    let fp = hipfire_generate::common::asst_turn_fingerprint("Done.", &[]);
    let k0 = hipfire_generate::dense::glimmer_turn_key(fp, 0);
    let k1 = hipfire_generate::dense::glimmer_turn_key(fp, 1);
    let k2 = hipfire_generate::dense::glimmer_turn_key(fp, 2);
    assert_ne!(
        k0, k1,
        "identical content at different ordinals must have different keys"
    );
    assert_ne!(k1, k2);
    assert_ne!(k0, k2);
    assert_eq!(k0, hipfire_generate::dense::glimmer_turn_key(fp, 0));
    assert_eq!(k1, hipfire_generate::dense::glimmer_turn_key(fp, 1));
}
