// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.
//! Gemma 4 vision tower — placeholder.
//!
//! The vision tower (27-layer ViT, `hidden_size=1152`, `head_dim=72`,
//! `patch_size=16`, 280 soft tokens per image) is out of scope for
//! Ships 1–5. This file exists so the crate compiles and the `Architecture`
//! trait can reference the module.
//!
//! Wire into the VL dispatch path in a follow-up (Phase N+1).

// Stub types — expand when vision tower is implemented.
pub struct Gemma4VisionConfig;
pub struct Gemma4VisionWeights;
pub struct Gemma4VisionScratch;
