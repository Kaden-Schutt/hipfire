//! Arch-generic whole-model weight orchestration (Tier-2). Sequences
//! embed → final-norm → output → per-device layer loop over a `WeightSource`,
//! whose impls own the format-specific reads (HFQ vs ParoQuant) and bake their
//! own config. Per-arch crates wrap the returned `LoadedWeights<L>` into their
//! own weights struct. Complements `weight_backend::WeightBackend` (Tier-3,
//! per-tensor dequant), which `WeightSource::read_layer` calls internally.

use crate::llama::{EmbeddingFormat, LayerWeights, WeightTensor};
use crate::multi_gpu::Gpus;
use hip_bridge::HipResult;
use rdna_compute::{Gpu, GpuTensor};

/// Where each piece of the model lands across a device slice. `single` = the
/// n==1 degenerate case (everything on device 0). Moved verbatim from
/// `hipfire-arch-qwen35::qwen35::Layout` — arch-agnostic (depends only on `Gpus`).
pub struct Layout {
    output_device: usize,
    layer_to_device: Vec<usize>,
}
impl Layout {
    pub fn single(n_layers: usize) -> Self {
        Self {
            output_device: 0,
            layer_to_device: vec![0; n_layers],
        }
    }
    pub fn from_gpus(g: &Gpus, n_layers: usize) -> Self {
        Self {
            output_device: g.output_device,
            layer_to_device: (0..n_layers).map(|i| g.device_for_layer(i)).collect(),
        }
    }
    pub fn device_for_layer(&self, i: usize) -> usize {
        self.layer_to_device[i]
    }
    pub fn output_device(&self) -> usize {
        self.output_device
    }
}

/// Neutral result of the orchestrator. Each arch assembles its own weights
/// struct from this (qwen35 adds `pager`; llama drops `lm_head_aliases_embd`).
pub struct LoadedWeights<L> {
    pub token_embd: GpuTensor,
    pub embd_format: EmbeddingFormat,
    pub output_norm: GpuTensor,
    pub output: WeightTensor,
    pub layers: Vec<L>,
    /// True iff the tied lm_head aliases the embedding buffer (qwen35 single-GPU);
    /// llama always returns `false` (it reuploads).
    pub lm_head_aliases_embd: bool,
}

pub trait LayerGpuFree {
    fn free_gpu(self, gpu: &mut Gpu);
}

impl LayerGpuFree for LayerWeights {
    fn free_gpu(self, gpu: &mut Gpu) {
        let LayerWeights {
            attn_norm,
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            ffn_norm,
            w_gate,
            w_up,
            w_down,
        } = self;
        let _ = gpu.free_tensor(attn_norm);
        wq.free_all(gpu);
        wk.free_all(gpu);
        wv.free_all(gpu);
        wo.free_all(gpu);
        if let Some(tensor) = q_norm {
            let _ = gpu.free_tensor(tensor);
        }
        if let Some(tensor) = k_norm {
            let _ = gpu.free_tensor(tensor);
        }
        let _ = gpu.free_tensor(ffn_norm);
        w_gate.free_all(gpu);
        w_up.free_all(gpu);
        w_down.free_all(gpu);
    }
}

/// Whole-model weight source — the one place HFQ vs PaRo differs. Config is held
/// by the impl (not passed per-call) so the orchestrator stays config-agnostic.
/// `read_layer` reuses Tier-3 `load_layer<B: WeightBackend>` internally.
pub trait WeightSource {
    type Layer;
    fn n_layers(&self) -> usize;
    /// Pre-load hook. HFQ drops the mmap when n==1; PaRo rejects n>1; llama no-op.
    fn prepare(&mut self, n_devices: usize) -> HipResult<()>;
    fn read_embed(&mut self, gpu: &mut Gpu) -> HipResult<(GpuTensor, EmbeddingFormat)>;
    fn read_final_norm(&mut self, gpu: &mut Gpu) -> HipResult<GpuTensor>;
    /// `can_alias` is true iff embed and output share a device (n==1); the impl
    /// decides whether to use it (qwen35 aliases; llama ignores it and reuploads).
    fn read_output(
        &mut self,
        gpu: &mut Gpu,
        embd: &GpuTensor,
        embd_fmt: EmbeddingFormat,
        can_alias: bool,
    ) -> HipResult<(WeightTensor, bool)>;
    fn read_layer(&mut self, gpu: &mut Gpu, layer_idx: usize) -> HipResult<Self::Layer>;
}

/// Drive a `WeightSource` across a device slice. Single shared copy of the
/// embed → norm → output → per-device layer loop.
pub fn load_weights_transactional<S: WeightSource>(
    source: &mut S,
    devices: &mut [Gpu],
    layout: &Layout,
) -> HipResult<LoadedWeights<S::Layer>>
where
    S::Layer: LayerGpuFree,
{
    let mut staged = LoadedWeightsStaging {
        token_embd: None,
        output_norm: None,
        output: None,
        layers: Vec::with_capacity(source.n_layers()),
        lm_head_aliases_embd: false,
    };
    let result = (|| -> HipResult<(EmbeddingFormat, bool)> {
        source.prepare(devices.len())?;
        let out_dev = layout.output_device();
        let can_alias = devices.len() == 1;
        let (token_embd, embd_format) = source.read_embed(&mut devices[0])?;
        staged.token_embd = Some(token_embd);
        #[cfg(feature = "dflash-fault-inject")]
        crate::dflash_generic::generic_dflash_allocation_boundary(
            crate::dflash_generic::GenericDflashConstructionStage::TargetWeightsAllocation(0),
        )
        .map_err(|e| hip_bridge::HipError::new(0, &e))?;
        let output_norm = source.read_final_norm(&mut devices[out_dev])?;
        staged.output_norm = Some(output_norm);
        #[cfg(feature = "dflash-fault-inject")]
        crate::dflash_generic::generic_dflash_allocation_boundary(
            crate::dflash_generic::GenericDflashConstructionStage::TargetWeightsAllocation(1),
        )
        .map_err(|e| hip_bridge::HipError::new(0, &e))?;
        let (output, lm_head_aliases_embd) = source.read_output(
            &mut devices[out_dev],
            staged.token_embd.as_ref().expect("staged token embedding"),
            embd_format,
            can_alias,
        )?;
        staged.lm_head_aliases_embd = lm_head_aliases_embd;
        staged.output = Some(output);
        #[cfg(feature = "dflash-fault-inject")]
        crate::dflash_generic::generic_dflash_allocation_boundary(
            crate::dflash_generic::GenericDflashConstructionStage::TargetWeightsAllocation(2),
        )
        .map_err(|e| hip_bridge::HipError::new(0, &e))?;
        for i in 0..source.n_layers() {
            let d = layout.device_for_layer(i);
            staged.layers.push(source.read_layer(&mut devices[d], i)?);
        }
        Ok((embd_format, lm_head_aliases_embd))
    })();
    match result {
        Ok((embd_format, lm_head_aliases_embd)) => Ok(LoadedWeights {
            token_embd: staged.token_embd.take().expect("staged token embedding"),
            embd_format,
            output_norm: staged.output_norm.take().expect("staged output norm"),
            output: staged.output.take().expect("staged output"),
            layers: staged.layers,
            lm_head_aliases_embd,
        }),
        Err(error) => {
            staged.free_gpu(devices, layout);
            Err(error)
        }
    }
}

struct LoadedWeightsStaging<L> {
    token_embd: Option<GpuTensor>,
    output_norm: Option<GpuTensor>,
    output: Option<WeightTensor>,
    layers: Vec<L>,
    lm_head_aliases_embd: bool,
}

impl<L: LayerGpuFree> LoadedWeightsStaging<L> {
    fn free_gpu(mut self, devices: &mut [Gpu], layout: &Layout) {
        for (i, layer) in self.layers.drain(..).enumerate().rev() {
            layer.free_gpu(&mut devices[layout.device_for_layer(i)]);
        }
        if let Some(output) = self.output.take() {
            if !self.lm_head_aliases_embd {
                output.free_all(&mut devices[layout.output_device()]);
            }
        }
        if let Some(output_norm) = self.output_norm.take() {
            let _ = devices[layout.output_device()].free_tensor(output_norm);
        }
        if let Some(token_embd) = self.token_embd.take() {
            let _ = devices[0].free_tensor(token_embd);
        }
    }
}

/// Legacy orchestration entry point retained for non-LLaMA arch loaders whose
/// layer type does not carry a runtime-side GPU teardown implementation.
pub fn load_weights<S: WeightSource>(
    source: &mut S,
    devices: &mut [Gpu],
    layout: &Layout,
) -> HipResult<LoadedWeights<S::Layer>> {
    source.prepare(devices.len())?;
    let out_dev = layout.output_device();
    let can_alias = devices.len() == 1;
    let (token_embd, embd_format) = source.read_embed(&mut devices[0])?;
    let output_norm = source.read_final_norm(&mut devices[out_dev])?;
    let (output, lm_head_aliases_embd) =
        source.read_output(&mut devices[out_dev], &token_embd, embd_format, can_alias)?;
    let mut layers = Vec::with_capacity(source.n_layers());
    for i in 0..source.n_layers() {
        let d = layout.device_for_layer(i);
        layers.push(source.read_layer(&mut devices[d], i)?);
    }
    Ok(LoadedWeights {
        token_embd,
        embd_format,
        output_norm,
        output,
        layers,
        lm_head_aliases_embd,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_layout_all_on_device_0() {
        let l = Layout::single(5);
        assert_eq!(l.output_device(), 0);
        for i in 0..5 {
            assert_eq!(l.device_for_layer(i), 0);
        }
    }
}
