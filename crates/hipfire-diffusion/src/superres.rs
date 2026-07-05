// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Native pixel-space super-resolution (RealESRGAN / RRDBNet) for the MrFlow
//! staged-sampling pipeline: a low-resolution generate is decoded, upscaled in
//! pixel space by this model, re-encoded, and refined. This module holds the
//! RRDBNet building blocks. Weight import lives in the offline
//! `hipfire-diffusion-coexist` crate; this is the runtime forward path.
//!
//! Scaffolding until the full net + MrFlow wiring land; individual blocks are
//! validated CPU-vs-GPU before assembly.
#![allow(dead_code)]

use super::*;

/// RealESRGAN / RRDBNet residual negative slope.
pub(crate) const RRDB_LEAKY_SLOPE: f32 = 0.2;
/// RealESRGAN / RRDBNet residual scaling (applied before the residual add).
pub(crate) const RRDB_RESIDUAL_SCALE: f32 = 0.2;

fn leaky_relu_cpu(input: &CpuTensor) -> CpuTensor {
    tensor_map(input, |value| {
        if value >= 0.0 {
            value
        } else {
            RRDB_LEAKY_SLOPE * value
        }
    })
}

/// One Residual Dense Block (RDB): five 3x3 convs with growing dense
/// concatenation of every prior output, LeakyReLU(0.2) after convs 1-4, and a
/// `x + 0.2 * x5` residual. Matches basicsr's `ResidualDenseBlock`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SuperResResidualDenseBlock {
    pub conv1: Conv2dLayer,
    pub conv2: Conv2dLayer,
    pub conv3: Conv2dLayer,
    pub conv4: Conv2dLayer,
    pub conv5: Conv2dLayer,
}

impl SuperResResidualDenseBlock {
    /// Load from an HFQ under `prefix` (e.g. `body.0.rdb1`); each conv is
    /// `{prefix}.conv{k}.{weight,bias}`, all 3x3 with padding 1.
    pub fn from_hfq(hfq: &HfqFile, prefix: &str) -> DiffusionResult<Self> {
        let conv = |k: usize| -> DiffusionResult<Conv2dLayer> {
            Conv2dLayer::from_hfq(
                hfq,
                &format!("{prefix}.conv{k}.weight"),
                Some(&format!("{prefix}.conv{k}.bias")),
                1,
            )
        };
        Ok(Self {
            conv1: conv(1)?,
            conv2: conv(2)?,
            conv3: conv(3)?,
            conv4: conv(4)?,
            conv5: conv(5)?,
        })
    }

    /// CPU reference forward.
    pub fn forward(&self, x: &CpuTensor) -> DiffusionResult<CpuTensor> {
        let x1 = leaky_relu_cpu(&self.conv1.forward(x)?);
        let cat1 = concat_channels_nchw(x, &x1)?;
        let x2 = leaky_relu_cpu(&self.conv2.forward(&cat1)?);
        let cat2 = concat_channels_nchw(&cat1, &x2)?;
        let x3 = leaky_relu_cpu(&self.conv3.forward(&cat2)?);
        let cat3 = concat_channels_nchw(&cat2, &x3)?;
        let x4 = leaky_relu_cpu(&self.conv4.forward(&cat3)?);
        let cat4 = concat_channels_nchw(&cat3, &x4)?;
        // conv5 has no activation.
        let x5 = self.conv5.forward(&cat4)?;
        if x5.data.len() != x.data.len() {
            return Err(DiffusionError::InvalidMetadata(format!(
                "RDB residual shape {:?} != input shape {:?}",
                x5.shape, x.shape
            )));
        }
        let data = x
            .data
            .iter()
            .zip(&x5.data)
            .map(|(xv, x5v)| xv + RRDB_RESIDUAL_SCALE * x5v)
            .collect();
        Ok(CpuTensor {
            shape: x.shape.clone(),
            data,
        })
    }

    /// Device-resident forward. Consumes nothing (caller owns `input`); frees
    /// every intermediate as it goes.
    pub(crate) fn forward_resident(
        &self,
        input: &hipfire_rdna::GpuTensor,
        gpu: &mut hipfire_rdna::Gpu,
        cache: &mut RocmWeightCache,
    ) -> DiffusionResult<hipfire_rdna::GpuTensor> {
        let x1 = {
            let conv = self.conv1.forward_resident(input, gpu, cache)?;
            let act = leaky_relu_resident(gpu, &conv, RRDB_LEAKY_SLOPE)?;
            free_resident(gpu, conv)?;
            act
        };
        let cat1 = concat_channels_nchw_resident(gpu, input, &x1)?;
        free_resident(gpu, x1)?;

        let x2 = {
            let conv = self.conv2.forward_resident(&cat1, gpu, cache)?;
            let act = leaky_relu_resident(gpu, &conv, RRDB_LEAKY_SLOPE)?;
            free_resident(gpu, conv)?;
            act
        };
        let cat2 = concat_channels_nchw_resident(gpu, &cat1, &x2)?;
        free_resident(gpu, cat1)?;
        free_resident(gpu, x2)?;

        let x3 = {
            let conv = self.conv3.forward_resident(&cat2, gpu, cache)?;
            let act = leaky_relu_resident(gpu, &conv, RRDB_LEAKY_SLOPE)?;
            free_resident(gpu, conv)?;
            act
        };
        let cat3 = concat_channels_nchw_resident(gpu, &cat2, &x3)?;
        free_resident(gpu, cat2)?;
        free_resident(gpu, x3)?;

        let x4 = {
            let conv = self.conv4.forward_resident(&cat3, gpu, cache)?;
            let act = leaky_relu_resident(gpu, &conv, RRDB_LEAKY_SLOPE)?;
            free_resident(gpu, conv)?;
            act
        };
        let cat4 = concat_channels_nchw_resident(gpu, &cat3, &x4)?;
        free_resident(gpu, cat3)?;
        free_resident(gpu, x4)?;

        // conv5 has no activation; residual add x + 0.2 * x5.
        let x5 = self.conv5.forward_resident(&cat4, gpu, cache)?;
        free_resident(gpu, cat4)?;
        let out = scaled_add_resident(gpu, input, &x5, RRDB_RESIDUAL_SCALE)?;
        free_resident(gpu, x5)?;
        Ok(out)
    }
}

/// Residual-in-Residual Dense Block (RRDB): three RDBs in series with a
/// `x + 0.2 * out` residual. Matches basicsr's `RRDB`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SuperResRrdb {
    pub rdb1: SuperResResidualDenseBlock,
    pub rdb2: SuperResResidualDenseBlock,
    pub rdb3: SuperResResidualDenseBlock,
}

impl SuperResRrdb {
    /// Load from an HFQ under `prefix` (e.g. `body.0`); the three RDBs are
    /// `{prefix}.rdb1`, `{prefix}.rdb2`, `{prefix}.rdb3`.
    pub fn from_hfq(hfq: &HfqFile, prefix: &str) -> DiffusionResult<Self> {
        Ok(Self {
            rdb1: SuperResResidualDenseBlock::from_hfq(hfq, &format!("{prefix}.rdb1"))?,
            rdb2: SuperResResidualDenseBlock::from_hfq(hfq, &format!("{prefix}.rdb2"))?,
            rdb3: SuperResResidualDenseBlock::from_hfq(hfq, &format!("{prefix}.rdb3"))?,
        })
    }

    /// CPU reference forward.
    pub fn forward(&self, x: &CpuTensor) -> DiffusionResult<CpuTensor> {
        let out = self.rdb1.forward(x)?;
        let out = self.rdb2.forward(&out)?;
        let out = self.rdb3.forward(&out)?;
        if out.data.len() != x.data.len() {
            return Err(DiffusionError::InvalidMetadata(format!(
                "RRDB residual shape {:?} != input shape {:?}",
                out.shape, x.shape
            )));
        }
        let data = x
            .data
            .iter()
            .zip(&out.data)
            .map(|(xv, ov)| xv + RRDB_RESIDUAL_SCALE * ov)
            .collect();
        Ok(CpuTensor {
            shape: x.shape.clone(),
            data,
        })
    }

    /// Device-resident forward. Caller owns `input`; intermediates are freed.
    pub(crate) fn forward_resident(
        &self,
        input: &hipfire_rdna::GpuTensor,
        gpu: &mut hipfire_rdna::Gpu,
        cache: &mut RocmWeightCache,
    ) -> DiffusionResult<hipfire_rdna::GpuTensor> {
        let o1 = self.rdb1.forward_resident(input, gpu, cache)?;
        let o2 = self.rdb2.forward_resident(&o1, gpu, cache)?;
        free_resident(gpu, o1)?;
        let o3 = self.rdb3.forward_resident(&o2, gpu, cache)?;
        free_resident(gpu, o2)?;
        let out = scaled_add_resident(gpu, input, &o3, RRDB_RESIDUAL_SCALE)?;
        free_resident(gpu, o3)?;
        Ok(out)
    }
}
