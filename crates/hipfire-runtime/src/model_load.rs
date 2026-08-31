//! Arch-generic whole-model weight orchestration (Tier-2). Sequences
//! embed → final-norm → output → per-device layer loop over a `WeightSource`,
//! whose impls own the format-specific reads (HFQ vs ParoQuant) and bake their
//! own config. Per-arch crates wrap the returned `LoadedWeights<L>` into their
//! own weights struct. Complements `weight_backend::WeightBackend` (Tier-3,
//! per-tensor dequant), which `WeightSource::read_layer` calls internally.

use crate::llama::{EmbeddingFormat, WeightTensor};
use hipfire_hardware::{DeviceMesh, DimKind};
use hip_bridge::HipResult;
use hipfire_hardware::Gpus;
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

    /// Build the canonical stage/rank-0 view from an admitted mesh. The
    /// manifest planner owns the full stage grid; this legacy loader view
    /// selects rank zero for each layer so existing orchestrators continue to
    /// have one deterministic device index until their typed mesh path lands.
    pub fn from_mesh(mesh: &DeviceMesh, n_layers: usize) -> Self {
        let mut output_coord = mesh.coord_of(0);
        if let Some(index) = mesh.axes().iter().position(|axis| axis.kind == DimKind::Pp) {
            output_coord[index] = mesh.size_of(DimKind::Pp).saturating_sub(1);
        }
        let layer_to_device = (0..n_layers)
            .map(|layer| {
                let mut coord = mesh.coord_of(0);
                if let Some(index) = mesh.axes().iter().position(|axis| axis.kind == DimKind::Pp) {
                    coord[index] = mesh.stage_for_layer(layer, n_layers);
                }
                mesh.device_of(&coord)
            })
            .collect();
        Self {
            output_device: mesh.device_of(&output_coord),
            layer_to_device,
        }
    }

    /// Validate the pure layout before any source preparation or GPU upload.
    pub fn validate(&self, n_devices: usize, n_layers: usize) -> Result<(), String> {
        if self.output_device >= n_devices {
            return Err(format!(
                "layout output device {} outside device count {}",
                self.output_device, n_devices
            ));
        }
        if self.layer_to_device.len() != n_layers {
            return Err(format!(
                "layout has {} layer assignments, expected {n_layers}",
                self.layer_to_device.len()
            ));
        }
        if let Some((layer, &device)) = self
            .layer_to_device
            .iter()
            .enumerate()
            .find(|(_, &device)| device >= n_devices)
        {
            return Err(format!(
                "layout layer {layer} device {device} outside device count {n_devices}"
            ));
        }
        Ok(())
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
pub fn load_weights<S: WeightSource>(
    source: &mut S,
    devices: &mut [Gpu],
    layout: &Layout,
) -> HipResult<LoadedWeights<S::Layer>> {
    if devices.is_empty() {
        return Err(hip_bridge::HipError::new(
            0,
            "load_weights: at least one device is required",
        ));
    }
    layout
        .validate(devices.len(), source.n_layers())
        .map_err(|reason| hip_bridge::HipError::new(0, &reason))?;
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

    #[test]
    fn mesh_layout_selects_stage_rank_zero_without_io() {
        let mesh = DeviceMesh::rect(&[(DimKind::Pp, 2), (DimKind::Tp, 2)]);
        let layout = Layout::from_mesh(&mesh, 4);
        assert_eq!(layout.output_device(), 2);
        assert_eq!(
            (0..4)
                .map(|layer| layout.device_for_layer(layer))
                .collect::<Vec<_>>(),
            vec![0, 0, 2, 2]
        );
        assert!(layout.validate(mesh.n_devices(), 4).is_ok());
    }

    #[test]
    fn invalid_layout_is_rejected_before_source_work() {
        let layout = Layout::single(2);
        assert!(layout.validate(0, 2).is_err());
        assert!(layout.validate(1, 3).is_err());
    }
}
