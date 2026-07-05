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

fn tensor_add_cpu(a: &CpuTensor, b: &CpuTensor) -> DiffusionResult<CpuTensor> {
    if a.data.len() != b.data.len() {
        return Err(DiffusionError::InvalidMetadata(format!(
            "add shape mismatch {:?} vs {:?}",
            a.shape, b.shape
        )));
    }
    Ok(CpuTensor {
        shape: a.shape.clone(),
        data: a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect(),
    })
}

fn conv3x3_from_hfq(hfq: &HfqFile, name: &str) -> DiffusionResult<Conv2dLayer> {
    Conv2dLayer::from_hfq(
        hfq,
        &format!("{name}.weight"),
        Some(&format!("{name}.bias")),
        1,
    )
}

/// RealESRGAN RRDBNet: pixel-unshuffle input stage (for scale 1/2), a
/// conv_first, `num_block` RRDBs with a conv_body and a plain global residual,
/// two nearest-upsample + conv + LeakyReLU stages, and a conv_hr / conv_last
/// head. The two fixed upsample stages give x4; scale 2 pre-unshuffles by 2 and
/// scale 1 by 4, so the net output ratio is `4 / input_unshuffle_scale`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SuperResRrdbNet {
    pub scale: usize,
    pub conv_first: Conv2dLayer,
    pub body: Vec<SuperResRrdb>,
    pub conv_body: Conv2dLayer,
    pub conv_up1: Conv2dLayer,
    pub conv_up2: Conv2dLayer,
    pub conv_hr: Conv2dLayer,
    pub conv_last: Conv2dLayer,
}

impl SuperResRrdbNet {
    /// Pre-body space-to-depth factor: scale 2 -> 2, scale 1 -> 4, otherwise 1
    /// (no unshuffle). Matches basicsr's RRDBNet.
    fn input_unshuffle_scale(&self) -> usize {
        match self.scale {
            2 => 2,
            1 => 4,
            _ => 1,
        }
    }

    /// Open a RealESRGAN RRDBNet `.hfq` produced by
    /// `hipfire_diffusion_coexist::import_realesrgan_to_hfq`, reading `scale` and
    /// `num_block` from the package metadata.
    pub(crate) fn open_hfq(path: &std::path::Path) -> DiffusionResult<Self> {
        let hfq =
            HfqFile::open_index_only(path).map_err(|err| DiffusionError::Io(err.to_string()))?;
        let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).map_err(|err| {
            DiffusionError::InvalidMetadata(format!("super-res metadata parse failed: {err}"))
        })?;
        let scale = meta.get("scale").and_then(serde_json::Value::as_u64).ok_or_else(|| {
            DiffusionError::InvalidMetadata("super-res metadata missing scale".to_string())
        })? as usize;
        let num_block =
            meta.get("num_block").and_then(serde_json::Value::as_u64).ok_or_else(|| {
                DiffusionError::InvalidMetadata("super-res metadata missing num_block".to_string())
            })? as usize;
        Self::from_hfq(&hfq, num_block, scale)
    }

    pub fn from_hfq(hfq: &HfqFile, num_block: usize, scale: usize) -> DiffusionResult<Self> {
        let body = (0..num_block)
            .map(|i| SuperResRrdb::from_hfq(hfq, &format!("body.{i}")))
            .collect::<DiffusionResult<Vec<_>>>()?;
        Ok(Self {
            scale,
            conv_first: conv3x3_from_hfq(hfq, "conv_first")?,
            body,
            conv_body: conv3x3_from_hfq(hfq, "conv_body")?,
            conv_up1: conv3x3_from_hfq(hfq, "conv_up1")?,
            conv_up2: conv3x3_from_hfq(hfq, "conv_up2")?,
            conv_hr: conv3x3_from_hfq(hfq, "conv_hr")?,
            conv_last: conv3x3_from_hfq(hfq, "conv_last")?,
        })
    }

    /// CPU reference forward. `input` is NCHW RGB in model range.
    pub fn forward(&self, input: &CpuTensor) -> DiffusionResult<CpuTensor> {
        let unshuffle = self.input_unshuffle_scale();
        let feat = if unshuffle > 1 {
            pixel_unshuffle_nchw(input, unshuffle)?
        } else {
            input.clone()
        };
        let feat = self.conv_first.forward(&feat)?;
        let mut body = feat.clone();
        for rrdb in &self.body {
            body = rrdb.forward(&body)?;
        }
        let body = self.conv_body.forward(&body)?;
        // Global residual is a plain add (no 0.2 scaling here).
        let mut feat = tensor_add_cpu(&feat, &body)?;
        feat = leaky_relu_cpu(&self.conv_up1.forward(&upsample_nearest2d_nchw(&feat, 2)?)?);
        feat = leaky_relu_cpu(&self.conv_up2.forward(&upsample_nearest2d_nchw(&feat, 2)?)?);
        let hr = leaky_relu_cpu(&self.conv_hr.forward(&feat)?);
        self.conv_last.forward(&hr)
    }

    /// Device-resident forward.
    pub(crate) fn forward_resident(
        &self,
        input: &hipfire_rdna::GpuTensor,
        gpu: &mut hipfire_rdna::Gpu,
        cache: &mut RocmWeightCache,
    ) -> DiffusionResult<hipfire_rdna::GpuTensor> {
        let unshuffle = self.input_unshuffle_scale();
        let unshuffled = if unshuffle > 1 {
            Some(pixel_unshuffle_nchw_resident(gpu, input, unshuffle)?)
        } else {
            None
        };
        let feat = self.conv_first.forward_resident(
            unshuffled.as_ref().unwrap_or(input),
            gpu,
            cache,
        )?;
        if let Some(unshuffled) = unshuffled {
            free_resident(gpu, unshuffled)?;
        }

        // The body chain produces a fresh tensor; `forward_resident` never frees
        // its input, so running the first block (or conv_body) on `feat`
        // directly keeps `feat` alive for the global residual add.
        let body_conv = if self.body.is_empty() {
            self.conv_body.forward_resident(&feat, gpu, cache)?
        } else {
            let mut body = self.body[0].forward_resident(&feat, gpu, cache)?;
            for rrdb in self.body.iter().skip(1) {
                let next = rrdb.forward_resident(&body, gpu, cache)?;
                free_resident(gpu, body)?;
                body = next;
            }
            let conv = self.conv_body.forward_resident(&body, gpu, cache)?;
            free_resident(gpu, body)?;
            conv
        };
        let mut feat_res = tensor_add_resident(gpu, &feat, &body_conv)?;
        free_resident(gpu, feat)?;
        free_resident(gpu, body_conv)?;

        for conv in [&self.conv_up1, &self.conv_up2] {
            let up = upsample_nearest2d_nchw_resident(gpu, &feat_res, 2)?;
            free_resident(gpu, feat_res)?;
            let conv_out = conv.forward_resident(&up, gpu, cache)?;
            free_resident(gpu, up)?;
            feat_res = leaky_relu_resident(gpu, &conv_out, RRDB_LEAKY_SLOPE)?;
            free_resident(gpu, conv_out)?;
        }

        let hr_conv = self.conv_hr.forward_resident(&feat_res, gpu, cache)?;
        free_resident(gpu, feat_res)?;
        let hr = leaky_relu_resident(gpu, &hr_conv, RRDB_LEAKY_SLOPE)?;
        free_resident(gpu, hr_conv)?;
        let out = self.conv_last.forward_resident(&hr, gpu, cache)?;
        free_resident(gpu, hr)?;
        Ok(out)
    }
}
