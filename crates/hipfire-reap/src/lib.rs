//! Model-agnostic REAP: selective expert pruning + (SP2+) selective re-quant
//! overlay for MoE models. See docs/superpowers/specs/2026-06-11-generic-moe-reap-design.md

pub mod gather;
pub mod hook;
pub mod load;
pub mod plan;
pub mod source;

pub use hook::ReapArchHook;
pub use source::ExpertPlan;
