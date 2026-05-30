// SPDX-License-Identifier: MIT OR Apache-2.0
// Model-specific kernel extensions.
// These live outside the main dispatch families because only one model
// uses them. If a second model needs the same pattern, promote to families/.

pub mod deepseek4;
pub mod qwen35;
