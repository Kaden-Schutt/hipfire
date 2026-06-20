// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Request-shape types parsed from the JSONL protocol.
//!
//! Extracted verbatim from the former daemon `main.rs` monolith (no behavior
//! change) so the generate paths (now in this crate) can reference them without
//! reaching back into the bin.

/// Thinking mode requested for a generation, parsed from the JSONL protocol's
/// OpenAI-compatible `reasoning_effort` or project-custom `thinking_mode` field.
#[derive(Copy, Clone, Debug)]
pub enum ThinkMode {
    /// Non-thinking. Frame: `<｜Assistant｜></think>{response}`.
    /// Model skips reasoning, replies directly. HF default for chat.
    NonThink,
    /// Thinking-high. Frame: `<｜Assistant｜><think>{reasoning}</think>{response}`.
    /// Model produces a `<think>` block before responding.
    High,
    /// Thinking-max. Same frame as `High`, plus prepended
    /// "Reasoning Effort: Absolute maximum..." system instruction.
    /// HF recommends context ≥ 384K for this mode.
    Max,
}

impl ThinkMode {
    /// Map a JSONL field value (OpenAI-compatible `reasoning_effort` or
    /// project-custom `thinking_mode`) to a mode.
    /// Accepted: "none|off|chat|minimal" → NonThink;
    ///           "low|medium|high|thinking" → High;
    ///           "max" → Max. Anything else → NonThink (safe default).
    pub fn from_str(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "max" => Self::Max,
            "high" | "thinking" | "low" | "medium" => Self::High,
            _ => Self::NonThink,
        }
    }
}
