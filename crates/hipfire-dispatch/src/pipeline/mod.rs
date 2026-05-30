// SPDX-License-Identifier: MIT OR Apache-2.0
use crate::types::*;

/// A resolved pipeline: the sequence of ops a kernel performs.
/// Used by the best-fit dispatcher to match model intent against available kernels.
pub struct Pipeline {
    pub ops: &'static [PipelineOp],
}

impl Pipeline {
    pub fn new(ops: &'static [PipelineOp]) -> Self {
        Self { ops }
    }

    /// Check if this pipeline can satisfy the requested ops.
    /// A pipeline satisfies if its ops are a prefix of the requested ops,
    /// or if the kernel is fused (single kernel does more than the minimum).
    pub fn can_satisfy(&self, requested: &[PipelineOp]) -> bool {
        if self.ops.len() > requested.len() {
            return false;
        }
        self.ops.iter().zip(requested.iter()).all(|(a, b)| a == b)
    }
}
