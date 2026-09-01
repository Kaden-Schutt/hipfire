//! Arch-generic whole-model weight orchestration (Tier-2). Sequences
//! embed → final-norm → output → per-device layer loop over a `WeightSource`,
//! whose impls own the format-specific reads (HFQ vs ParoQuant) and bake their
//! own config. Per-arch crates wrap the returned `LoadedWeights<L>` into their
//! own weights struct. Complements `weight_backend::WeightBackend` (Tier-3,
//! per-tensor dequant), which `WeightSource::read_layer` calls internally.

use crate::llama::{EmbeddingFormat, WeightTensor};
use hip_bridge::HipResult;
use hipfire_hardware::{DeviceMesh, DimKind, Gpus, MeshError};
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
    pub fn from_mesh(mesh: &DeviceMesh, n_layers: usize) -> Result<Self, MeshError> {
        let mut output_coord = mesh.coord_of(0)?;
        if let Some(index) = mesh.axes().iter().position(|axis| axis.kind == DimKind::Pp) {
            output_coord[index] = mesh.size_of(DimKind::Pp).saturating_sub(1);
        }
        let layer_to_device = (0..n_layers)
            .map(|layer| {
                let mut coord = mesh.coord_of(0)?;
                if let Some(index) = mesh.axes().iter().position(|axis| axis.kind == DimKind::Pp) {
                    coord[index] = mesh.stage_for_layer(layer, n_layers);
                }
                mesh.device_of(&coord)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            output_device: mesh.device_of(&output_coord)?,
            layer_to_device,
        })
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
/// struct from this (qwen35 adds `pager`).
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
    /// `can_alias` is true iff embed and output share a device (n==1); the
    /// implementation decides whether to use it (single-device LLaMA and
    /// qwen35 alias tied embeddings; multi-device routes re-materialize).
    fn read_output(
        &mut self,
        gpu: &mut Gpu,
        embd: &GpuTensor,
        embd_fmt: EmbeddingFormat,
        can_alias: bool,
    ) -> HipResult<(WeightTensor, bool)>;
    fn read_layer(&mut self, gpu: &mut Gpu, layer_idx: usize) -> HipResult<Self::Layer>;
    /// Release one successfully loaded layer during whole-model rollback.
    ///
    /// Layer ownership is architecture-specific (Qwen3.5 carries MoE
    /// pointer tables and shared Paro sidecars), so the source supplies the
    /// exact teardown instead of relying on a generic `Drop`.
    fn free_layer(&mut self, gpu: &mut Gpu, layer: Self::Layer);
}

/// Resource-neutral operations used by the staged load transaction.
///
/// Keeping the transaction separate from HIP resource types gives the CPU
/// contract tests a deterministic allocator/source seam. The production
/// adapter below is the only implementation that knows how to free a
/// `GpuTensor` or `WeightTensor`; the ordering and ownership rules are shared
/// by both paths.
trait StagedLoadOps {
    type Layer;
    type Embedding;
    type Norm;
    type Output;
    type Error;

    fn prepare(&mut self, n_devices: usize) -> Result<(), Self::Error>;
    fn n_layers(&self) -> usize;
    fn read_embed(&mut self) -> Result<(Self::Embedding, EmbeddingFormat), Self::Error>;
    fn read_final_norm(&mut self) -> Result<Self::Norm, Self::Error>;
    fn read_output(
        &mut self,
        embedding: &Self::Embedding,
        format: EmbeddingFormat,
        can_alias: bool,
    ) -> Result<(Self::Output, bool), Self::Error>;
    fn read_layer(&mut self, layer_idx: usize) -> Result<Self::Layer, Self::Error>;
    fn free_layer(&mut self, layer_idx: usize, layer: Self::Layer);
    fn free_output(&mut self, output: Self::Output, aliases_embedding: bool);
    fn free_final_norm(&mut self, norm: Self::Norm);
    fn free_embed(&mut self, embedding: Self::Embedding);
}

struct StagedWeights<E, N, O, L> {
    token_embd: E,
    embd_format: EmbeddingFormat,
    output_norm: N,
    output: O,
    layers: Vec<L>,
    lm_head_aliases_embd: bool,
}

/// Execute the common embed → norm → output → layer transaction.
///
/// Every successful publication is retained until the transaction commits.
/// Any error drains completed layers in reverse publication order, then output,
/// final norm, and embedding. This is deliberately generic so a CPU test source
/// can observe the exact order without initializing HIP.
fn run_staged_load<O: StagedLoadOps>(
    ops: &mut O,
    n_devices: usize,
    can_alias: bool,
) -> Result<StagedWeights<O::Embedding, O::Norm, O::Output, O::Layer>, O::Error> {
    ops.prepare(n_devices)?;
    let mut staged_embedding = None;
    let mut staged_norm = None;
    let mut staged_output = None;
    let mut staged_layers = Vec::with_capacity(ops.n_layers());

    let result = (|| {
        let (embedding, format) = ops.read_embed()?;
        staged_embedding = Some((embedding, format));

        let norm = ops.read_final_norm()?;
        staged_norm = Some(norm);

        let (output, aliases_embedding) = ops.read_output(
            &staged_embedding
                .as_ref()
                .expect("embedding staged before output")
                .0,
            staged_embedding
                .as_ref()
                .expect("embedding staged before output")
                .1,
            can_alias,
        )?;
        staged_output = Some((output, aliases_embedding));

        for layer_idx in 0..ops.n_layers() {
            staged_layers.push(ops.read_layer(layer_idx)?);
        }

        let (token_embd, embd_format) = staged_embedding.take().expect("embedding staged");
        let output_norm = staged_norm.take().expect("output norm staged");
        let (output, lm_head_aliases_embd) = staged_output.take().expect("output staged");
        Ok(StagedWeights {
            token_embd,
            embd_format,
            output_norm,
            output,
            layers: std::mem::take(&mut staged_layers),
            lm_head_aliases_embd,
        })
    })();

    if result.is_err() {
        for (layer_idx, layer) in staged_layers.drain(..).enumerate().rev() {
            ops.free_layer(layer_idx, layer);
        }
        if let Some((output, aliases_embedding)) = staged_output.take() {
            ops.free_output(output, aliases_embedding);
        }
        if let Some(norm) = staged_norm.take() {
            ops.free_final_norm(norm);
        }
        if let Some((embedding, _)) = staged_embedding.take() {
            ops.free_embed(embedding);
        }
    }
    result
}

struct GpuStagedLoadOps<'a, S> {
    source: &'a mut S,
    devices: &'a mut [Gpu],
    layout: &'a Layout,
}

impl<S: WeightSource> StagedLoadOps for GpuStagedLoadOps<'_, S> {
    type Layer = S::Layer;
    type Embedding = GpuTensor;
    type Norm = GpuTensor;
    type Output = WeightTensor;
    type Error = hip_bridge::HipError;

    fn prepare(&mut self, n_devices: usize) -> Result<(), Self::Error> {
        self.source.prepare(n_devices)
    }

    fn n_layers(&self) -> usize {
        self.source.n_layers()
    }

    fn read_embed(&mut self) -> Result<(Self::Embedding, EmbeddingFormat), Self::Error> {
        self.source.read_embed(&mut self.devices[0])
    }

    fn read_final_norm(&mut self) -> Result<Self::Norm, Self::Error> {
        let device = self.layout.output_device();
        self.source.read_final_norm(&mut self.devices[device])
    }

    fn read_output(
        &mut self,
        embedding: &Self::Embedding,
        format: EmbeddingFormat,
        can_alias: bool,
    ) -> Result<(Self::Output, bool), Self::Error> {
        let device = self.layout.output_device();
        self.source
            .read_output(&mut self.devices[device], embedding, format, can_alias)
    }

    fn read_layer(&mut self, layer_idx: usize) -> Result<Self::Layer, Self::Error> {
        let device = self.layout.device_for_layer(layer_idx);
        self.source.read_layer(&mut self.devices[device], layer_idx)
    }

    fn free_layer(&mut self, layer_idx: usize, layer: Self::Layer) {
        let device = self.layout.device_for_layer(layer_idx);
        self.source.free_layer(&mut self.devices[device], layer);
    }

    fn free_output(&mut self, output: Self::Output, aliases_embedding: bool) {
        let device = self.layout.output_device();
        if aliases_embedding {
            output.free_metadata_only(&mut self.devices[device]);
        } else {
            output.free_all(&mut self.devices[device]);
        }
    }

    fn free_final_norm(&mut self, norm: Self::Norm) {
        let device = self.layout.output_device();
        let _ = self.devices[device].free_tensor(norm);
    }

    fn free_embed(&mut self, embedding: Self::Embedding) {
        let _ = self.devices[0].free_tensor(embedding);
    }
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
    let n_devices = devices.len();
    let mut ops = GpuStagedLoadOps {
        source,
        devices,
        layout,
    };
    let staged = run_staged_load(&mut ops, n_devices, n_devices == 1)?;
    Ok(LoadedWeights {
        token_embd: staged.token_embd,
        embd_format: staged.embd_format,
        output_norm: staged.output_norm,
        output: staged.output,
        layers: staged.layers,
        lm_head_aliases_embd: staged.lm_head_aliases_embd,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

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
        let mesh = DeviceMesh::rect(&[(DimKind::Pp, 2), (DimKind::Tp, 2)])
            .expect("small test mesh construction cannot overflow");
        let layout =
            Layout::from_mesh(&mesh, 4).expect("valid mesh must produce a deterministic layout");
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

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum FailAt {
        Prepare,
        Embed,
        FinalNorm,
        Output,
        Layer(usize),
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Allocation {
        id: usize,
        kind: &'static str,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct OutputAllocation {
        primary: Option<Allocation>,
        metadata: Allocation,
    }

    #[derive(Debug, Default)]
    struct TestAllocator {
        next_id: usize,
        live: BTreeSet<usize>,
        frees: Vec<String>,
    }

    impl TestAllocator {
        fn alloc(&mut self, kind: &'static str) -> Allocation {
            let id = self.next_id;
            self.next_id += 1;
            assert!(self.live.insert(id), "test allocator id reused: {id}");
            Allocation { id, kind }
        }

        fn free(&mut self, allocation: Allocation) {
            self.free_named(allocation, None);
        }

        fn free_named(&mut self, allocation: Allocation, label: Option<String>) {
            assert!(
                self.live.remove(&allocation.id),
                "double free of {}#{}",
                allocation.kind,
                allocation.id
            );
            self.frees
                .push(label.unwrap_or_else(|| format!("free {}", allocation.kind)));
        }
    }

    /// CPU-only WeightSource seam: every resource is a tracked token, so a
    /// failure test can prove ownership transfer and exact reverse cleanup
    /// without constructing `Gpu` or relying on a global `Drop` implementation.
    struct TestWeightSource {
        allocator: TestAllocator,
        n_layers: usize,
        layer_devices: Vec<usize>,
        fail_at: Option<FailAt>,
        alias_output: bool,
    }

    impl TestWeightSource {
        fn new(n_layers: usize, fail_at: Option<FailAt>) -> Self {
            Self {
                allocator: TestAllocator::default(),
                n_layers,
                layer_devices: (0..n_layers).map(|i| i % 2).collect(),
                fail_at,
                alias_output: false,
            }
        }

        fn fail(&self, at: FailAt) -> Result<(), String> {
            if self.fail_at == Some(at) {
                Err(format!("injected {at:?} failure"))
            } else {
                Ok(())
            }
        }

        fn assert_clean(&self) {
            assert!(
                self.allocator.live.is_empty(),
                "staged resources leaked: {:?}",
                self.allocator.live
            );
        }
    }

    impl StagedLoadOps for TestWeightSource {
        type Layer = Allocation;
        type Embedding = Allocation;
        type Norm = Allocation;
        type Output = OutputAllocation;
        type Error = String;

        fn prepare(&mut self, _n_devices: usize) -> Result<(), Self::Error> {
            self.fail(FailAt::Prepare)
        }

        fn n_layers(&self) -> usize {
            self.n_layers
        }

        fn read_embed(&mut self) -> Result<(Self::Embedding, EmbeddingFormat), Self::Error> {
            self.fail(FailAt::Embed)?;
            Ok((self.allocator.alloc("embedding"), EmbeddingFormat::F32))
        }

        fn read_final_norm(&mut self) -> Result<Self::Norm, Self::Error> {
            self.fail(FailAt::FinalNorm)?;
            Ok(self.allocator.alloc("final-norm"))
        }

        fn read_output(
            &mut self,
            _embedding: &Self::Embedding,
            _format: EmbeddingFormat,
            can_alias: bool,
        ) -> Result<(Self::Output, bool), Self::Error> {
            self.fail(FailAt::Output)?;
            let aliases_embedding = can_alias && self.alias_output;
            let primary = (!aliases_embedding).then(|| self.allocator.alloc("output"));
            let metadata = self.allocator.alloc("output-metadata");
            Ok((OutputAllocation { primary, metadata }, aliases_embedding))
        }

        fn read_layer(&mut self, layer_idx: usize) -> Result<Self::Layer, Self::Error> {
            self.fail(FailAt::Layer(layer_idx))?;
            Ok(self.allocator.alloc("layer"))
        }

        fn free_layer(&mut self, layer_idx: usize, layer: Self::Layer) {
            self.allocator.free_named(
                layer,
                Some(format!("free layer@{}", self.layer_devices[layer_idx])),
            );
        }

        fn free_output(&mut self, mut output: Self::Output, aliases_embedding: bool) {
            self.allocator.free(output.metadata);
            if !aliases_embedding {
                self.allocator
                    .free(output.primary.take().expect("owned output primary"));
            } else {
                assert!(
                    output.primary.is_none(),
                    "alias output must not own a second primary buffer"
                );
            }
        }

        fn free_final_norm(&mut self, norm: Self::Norm) {
            self.allocator.free(norm);
        }

        fn free_embed(&mut self, embedding: Self::Embedding) {
            self.allocator.free(embedding);
        }
    }

    #[test]
    fn cpu_staged_load_sweep_rolls_back_every_failure_stage() {
        let cases = [
            (FailAt::Prepare, &[][..]),
            (FailAt::Embed, &[][..]),
            (FailAt::FinalNorm, &["free embedding"][..]),
            (FailAt::Output, &["free final-norm", "free embedding"][..]),
            (
                FailAt::Layer(0),
                &[
                    "free output-metadata",
                    "free output",
                    "free final-norm",
                    "free embedding",
                ][..],
            ),
            (
                FailAt::Layer(2),
                &[
                    "free layer@1",
                    "free layer@0",
                    "free output-metadata",
                    "free output",
                    "free final-norm",
                    "free embedding",
                ][..],
            ),
        ];

        for (failure, expected_frees) in cases {
            let mut source = TestWeightSource::new(4, Some(failure));
            let result = run_staged_load(&mut source, 2, false);
            assert!(result.is_err(), "{failure:?} must fail");
            assert_eq!(
                source.allocator.frees, expected_frees,
                "{failure:?} cleanup order changed"
            );
            source.assert_clean();
        }
    }

    #[test]
    fn cpu_staged_load_success_preserves_alias_and_frees_each_owner_once() {
        let mut source = TestWeightSource::new(2, None);
        source.alias_output = true;
        let loaded = run_staged_load(&mut source, 1, true).expect("staged load");
        assert!(loaded.lm_head_aliases_embd);
        assert_eq!(source.allocator.live.len(), 5);

        let StagedWeights {
            token_embd,
            output_norm,
            output,
            layers,
            lm_head_aliases_embd,
            ..
        } = loaded;
        for (layer_idx, layer) in layers.into_iter().enumerate().rev() {
            source.free_layer(layer_idx, layer);
        }
        source.free_output(output, lm_head_aliases_embd);
        source.free_final_norm(output_norm);
        source.free_embed(token_embd);

        assert_eq!(
            source.allocator.frees,
            vec![
                "free layer@1",
                "free layer@0",
                "free output-metadata",
                "free final-norm",
                "free embedding",
            ]
        );
        source.assert_clean();
    }
}
