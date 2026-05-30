// ── Init-time kernel matrix validation ──
//
// Every kernel combo is checked at Dispatch::build() time.
// No more "segfault on first decode because gfx1100 doesn't have MQ3 WMMA."

use std::collections::BTreeSet;

pub struct DispatchBuilder {
    required: Vec<KernelRequirement>,
}

struct KernelRequirement {
    family: &'static str,
    variant: &'static str,
    dtypes: BTreeSet<(&'static str, &'static str)>, // (arch, quant) pairs
}

/// What a concrete device + model needs.
pub struct ModelKernelProfile {
    pub device: GpuInfo,
    // Each family knows its own variant matrix:
    pub fused_qkv: FusedQkvConfig,
    pub attention: AttentionConfig,
    pub gemv: GemvConfig,
    pub rotation: RotationConfig,
    pub model_specific: Vec<(&'static str, Box<dyn Fn(&GpuInfo) -> bool>)>,
}

impl DispatchBuilder {
    pub fn from_profile(profile: &ModelKernelProfile) -> Result<Self, DispatchError> {
        let mut required = Vec::new();
        
        // Check every required variant against the device
        for (arch_quant, needs) in profile.fused_qkv.required_variants() {
            if !profile.device.supports(arch_quant.0, arch_quant.1) {
                return Err(DispatchError::UnsupportedVariant {
                    family: "fused_qkv",
                    variant: needs,
                    arch: arch_quant.0,
                    quant: arch_quant.1,
                });
            }
        }
        // ... same for attention, gemv, rotation, model_specific
        
        Ok(Self { required })
    }

    pub fn build(self, gpu: &Gpu, weights: &WeightMap) -> Result<Dispatch, DispatchError> {
        let rotation = self.select_kernel::<dyn RotationKernel>("rotation", gpu, weights)?;
        let fused_qkv = self.select_kernel::<dyn FusedQkvKernel>("fused_qkv", gpu, weights)?;
        let attn = self.select_kernel::<dyn AttentionKernel>("attention", gpu, weights)?;
        let fused_gate_up = self.select_kernel::<dyn FusedGateUpKernel>("fused_gate_up", gpu, weights)?;
        let gemv = self.select_kernel::<dyn GemvKernel>("gemv", gpu, weights)?;
        let gemm = self.select_kernel::<dyn GemmKernel>("gemm", gpu, weights)?;
        let model_specific = self.select_model_specific(gpu, weights)?;

        Ok(Dispatch {
            rotation,
            fused_qkv,
            attn,
            fused_gate_up,
            gemv,
            gemm,
            model_specific,
            graph: Mutex::new(None),
        })
    }

    fn select_kernel<T: ?Sized + Send + Sync>(
        &self, family: &str, gpu: &Gpu, weights: &WeightMap,
    ) -> Result<Arc<dyn T>, DispatchError> {
        // Query the kernel registry (a lazy-static BTreeMap from
        // (family, arch, quant) -> Arc<dyn Any>).
        //
        // Registry is populated at compile time by a macro in
        // hipfire-kernels that auto-generates entries from .hip files.
        //
        // If not found → Error::UnsupportedVariant with the
        // full (family, arch, quant) tuple so the error message is
        // immediately actionable.
        todo!()
    }
}

// ── Weight-safe view ──
// No model code ever matches on DType. The dispatch layer
// erases the concrete quant format into WeightView, which carries
// an opaque kernel tag + offset table.
pub struct WeightView {
    pub data: GpuPtr,
    pub kernel_tag: KernelTag, // selects which concrete kernel variant
    pub rows: u32,
    pub cols: u32,
}
