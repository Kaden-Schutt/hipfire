# Unified qwen35 Weight Loading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the three qwen35 weight loaders (`load_weights`, `load_weights_multi`, `load_weights_paroquant`) behind one `WeightBackend` trait in `hipfire-runtime`, deleting the triplicated layer-walk and the `paro_load_moe_ffn`/`load_moe_ffn` duplication.

**Architecture:** A `WeightBackend` trait in `hipfire-runtime` exposes single-tensor primitives (`load_wt`/`load_norm`/`load_f32`/`raw_bytes`/`after_layer`/`kind`/`moe_expert_layout`/`quant_config`) with two impls (`HfqBackend`, `ParoBackend`). qwen35 keeps a single generic layer-walk + MoE assembly over `&dyn WeightBackend`, with three small `kind()`/`moe_expert_layout()`-gated forks (routed-expert packing, embed/output, multi-GPU-HFQ-only). Public `load_weights`/`load_weights_multi` become thin HFQ wrappers so no examples change.

**Tech Stack:** Rust, ROCm/HIP, the hipfire `rdna-compute` (`Gpu`/`GpuTensor`) + `hipfire-runtime` (`WeightTensor`/`HfqFile`/`ModelSource`) + `hipfire-arch-qwen35` crates.

**Spec:** `docs/plans/2026-06-11-qwen35-unified-weight-loading-design.md`

**Crate dep direction (load-bearing):** `rdna-compute` → `hipfire-runtime` → `hipfire-arch-qwen35`. The trait lives in runtime and must never reference a qwen35 type.

**Verification reality:** Loaders need a GPU + model file, so fine-grained unit TDD is not feasible. The regression oracle is a **characterization capture** (Task 0): greedy logits/token-ids for one HFQ model and the A3B PARO model, captured BEFORE any change and asserted identical AFTER. Plus `./scripts/coherence-gate.sh`. GPU work goes under the repo GPU lock (cargo hooks handle it automatically).

---

## File structure

- **Create** `crates/hipfire-runtime/src/weight_backend.rs` — trait, `MoeExpertLayout`, `BackendKind`, `HfqBackend`, `ParoBackend`, and the relocated `pub` format helpers.
- **Modify** `crates/hipfire-runtime/src/lib.rs` — `pub mod weight_backend;`.
- **Modify** `crates/hipfire-arch-qwen35/src/qwen35.rs` — relocate helpers (Task 2/3), add generic `load_weights_generic`/`load_layer`/`load_moe_ffn`/embed-output dispatch (Task 4-7), rewrite `load_weights`/`load_weights_multi` as wrappers (Task 7), delete `load_weights_paroquant`/`load_layer_into`/`paro_load_moe_ffn` (Task 8).
- **Modify** `crates/hipfire-runtime/examples/daemon.rs` — Paro path → `load_weights_generic(&ParoBackend…)` (Task 8).
- **Create** `crates/hipfire-runtime/examples/weight_capture.rs` — characterization oracle (Task 0).

---

## Task 0: Characterization capture (regression oracle)

**Files:**
- Create: `crates/hipfire-runtime/examples/weight_capture.rs`
- Baseline output: `.scratch/loadcap-hfq-before.txt`, `.scratch/loadcap-paro-before.txt`

- [ ] **Step 1: Write the capture example**

A minimal binary that loads weights via the CURRENT entry points and prints a stable digest. Reuse the existing logit-dump path — model loads, runs one forward step, prints the top-8 logit ids+values for a fixed prompt. (Modeled on `examples/dump_logits_qwen35.rs`.)

```rust
// crates/hipfire-runtime/examples/weight_capture.rs
//! Characterization oracle for the WeightBackend refactor. Loads a model and
//! prints a deterministic digest (top-8 logits over a fixed 1-token prompt).
//! Identical output before/after the refactor == loaders byte-equivalent.
use std::path::Path;
fn main() {
    let path = std::env::args().nth(1).expect("usage: weight_capture <model>");
    let prompt = "The"; // fixed, deterministic
    // Delegate to the same load+forward the daemon uses for this model kind.
    // (HFQ -> qwen35::load_weights; safetensors dir -> qwen35::load_weights_paroquant)
    hipfire_runtime::eval_common::capture_digest(Path::new(&path), prompt);
}
```

If `eval_common::capture_digest` does not exist, instead copy the load+single-forward+top-k-print body from `examples/dump_logits_qwen35.rs` directly into this example, branching on `path.is_dir()` to pick `load_weights` vs `load_weights_paroquant`. Do NOT add new logic to the engine — this is a read-only probe.

- [ ] **Step 2: Build it**

Run: `cargo build -p hipfire-runtime --example weight_capture`
Expected: compiles.

- [ ] **Step 3: Capture HFQ baseline**

Run (under GPU lock — a plain `cargo run` triggers the hook):
`cargo run -q -p hipfire-runtime --example weight_capture -- <some .hfq model> > .scratch/loadcap-hfq-before.txt`
Expected: a file with 8 `id=… logit=…` lines.

- [ ] **Step 4: Capture A3B PARO baseline**

Run: `cargo run -q -p hipfire-runtime --example weight_capture -- <shisa-ai A3B PARO dir> > .scratch/loadcap-paro-before.txt`
Expected: 8 lines. (Set `HIPFIRE_GRAPH=0` if the A3B path still requires it — see memory.)

- [ ] **Step 5: Commit the oracle (not the .scratch outputs)**

```bash
git add crates/hipfire-runtime/examples/weight_capture.rs
git commit -m "test(qwen35): weight-load characterization oracle for backend refactor"
```

---

## Task 1: `weight_backend.rs` scaffold (trait + enums + export)

**Files:**
- Create: `crates/hipfire-runtime/src/weight_backend.rs`
- Modify: `crates/hipfire-runtime/src/lib.rs`

- [ ] **Step 1: Create the module with trait + enums**

```rust
// crates/hipfire-runtime/src/weight_backend.rs
//! Unified weight-loading backend. HfqBackend / ParoBackend implement
//! single-tensor loading; qwen35 (and later other arches) drive a single
//! generic layer-walk over `&dyn WeightBackend`. The trait references ONLY
//! runtime/rdna-compute types — never an arch-crate type.
use std::borrow::Cow;
use rdna_compute::dispatch::{Gpu, GpuTensor};
use crate::llama::WeightTensor;
use crate::model_source::QuantConfig;
use hip_bridge::HipResult;

/// Routed-expert on-disk packing (the MoE inner-loop fork).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MoeExpertLayout { Fused, ParoRepack }

/// Coarse backend identity for the embed/output forks.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BackendKind { Hfq, Paro }

pub trait WeightBackend {
    /// Single matmul weight. `leaf` is a BARE logical name, no `.weight` suffix.
    /// Both impls prepend `model.language_model.{p}.` and own the suffix:
    /// HFQ `.weight`; Paro `.qweight`(+`.qzeros`/`.scales`) else `.weight` (FP16).
    fn load_wt(&self, gpu: &mut Gpu, p: &str, leaf: &str, m: usize, k: usize) -> HipResult<WeightTensor>;

    /// RMSNorm weight WITH the GemmaRMSNorm `+= 1.0` bake. `p` MAY be "" for the
    /// top-level `norm.weight`; impl joins p/leaf skipping empty p.
    fn load_norm(&self, gpu: &mut Gpu, p: &str, leaf: &str, shape: &[usize]) -> HipResult<GpuTensor>;

    /// Raw f32 tensor, NO bake. HFQ decodes arbitrary quant types; Paro F16/F32.
    fn load_f32(&self, gpu: &mut Gpu, p: &str, leaf: &str, n: usize) -> HipResult<GpuTensor>;

    fn moe_expert_layout(&self) -> MoeExpertLayout;
    fn kind(&self) -> BackendKind;

    /// Raw bytes for Paro MoE CPU repack. HFQ unimplemented (Fused never calls).
    fn raw_bytes(&self, name: &str) -> Option<Cow<'_, [u8]>>;

    /// Post-layer page hook. HFQ drops the layer's pages; Paro no-op.
    fn after_layer(&self, _p: &str) {}

    fn quant_config(&self) -> Option<&QuantConfig> { None }
}
```

- [ ] **Step 2: Export it**

Modify `crates/hipfire-runtime/src/lib.rs` — add after `pub mod weight_pager;` (line ~38):

```rust
pub mod weight_backend;
```

- [ ] **Step 3: Build**

Run: `cargo build -p hipfire-runtime`
Expected: compiles (unused-trait warning is fine — impls land next). If `rdna_compute::dispatch::Gpu`/`GpuTensor` or `hip_bridge::HipResult` paths differ, fix the `use` to match the paths already used at the top of `crates/hipfire-arch-qwen35/src/qwen35.rs`.

- [ ] **Step 4: Commit**

```bash
git add crates/hipfire-runtime/src/weight_backend.rs crates/hipfire-runtime/src/lib.rs
git commit -m "feat(runtime): WeightBackend trait scaffold"
```

---

## Task 2: Relocate HFQ format helpers into runtime (kept working via old path)

**Files:**
- Modify: `crates/hipfire-runtime/src/weight_backend.rs`
- Modify: `crates/hipfire-arch-qwen35/src/qwen35.rs`

**Move (verbatim, body unchanged) these fns from `qwen35.rs` into `weight_backend.rs`, changing visibility to `pub`:**
- `load_weight_tensor_raw` (`qwen35.rs:842`)
- `load_norm_weight` (`qwen35.rs:817`)
- `load_any_as_f32` (`qwen35.rs:1395`)
- `load_raw_f32` (`qwen35.rs:1795` — it just calls `load_any_as_f32`)
- `load_awq_scale_for` (`qwen35.rs:951`)
- `load_weight_tensor` (`qwen35.rs:988`)

These take only `&HfqFile`/`&Gpu`/`&mut Gpu` and return `WeightTensor`/`GpuTensor`/`Option<GpuTensor>` — all available in runtime. They reference `f16_to_f32` (already `crate::llama::f16_to_f32`) and `DType` (use the same path qwen35 imports).

- [ ] **Step 1: Cut the six fns from qwen35.rs, paste into weight_backend.rs as `pub fn`**

Add the needed `use` lines to `weight_backend.rs` (e.g. `use crate::hfq::HfqFile; use crate::llama::f16_to_f32; use rdna_compute::dispatch::DType;`). Do not alter the fn bodies.

- [ ] **Step 2: Repoint qwen35's remaining old loaders at the moved fns**

In `qwen35.rs`, add `use hipfire_runtime::weight_backend::{load_weight_tensor, load_norm_weight, load_any_as_f32, load_raw_f32, load_awq_scale_for, load_weight_tensor_raw};` and delete the now-moved local definitions. All existing call sites (`load_weights`, `load_layer_into`, `load_moe_ffn`, `load_token_embd_into`, `load_output_into`) keep compiling against the imported names.

- [ ] **Step 3: Build the workspace**

Run: `cargo build`
Expected: compiles. Fix any `pub`/visibility or import-path errors. No behavior change — the old loaders now call the relocated fns.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(qwen35): relocate HFQ tensor helpers into runtime::weight_backend"
```

---

## Task 3: Relocate Paro format helpers + implement both backends

**Files:**
- Modify: `crates/hipfire-runtime/src/weight_backend.rs`
- Modify: `crates/hipfire-arch-qwen35/src/qwen35.rs`

**Move (verbatim) into `weight_backend.rs` as `pub fn`:** `paro_load_wt` (`2029`), `paro_load_norm` (`2046`), `paro_load_f32` (`2053`), `load_fp16_weight_from_source` (`1180`), `load_paroquant_weight` (`1111`). Repoint qwen35 via `use` (same pattern as Task 2). `paro_repack_moe_projection`/`alias_paro_rotation`/`paro_load_moe_shared_sidecars` STAY in qwen35 (MoE fork).

- [ ] **Step 1: Move the five Paro helpers; repoint qwen35; build**

Run: `cargo build`
Expected: compiles, old Paro path unchanged.

- [ ] **Step 2: Implement `HfqBackend`**

Append to `weight_backend.rs`:

```rust
pub struct HfqBackend<'a> { pub hfq: &'a crate::hfq::HfqFile }
impl<'a> HfqBackend<'a> { pub fn new(hfq: &'a crate::hfq::HfqFile) -> Self { Self { hfq } } }

impl<'a> WeightBackend for HfqBackend<'a> {
    fn load_wt(&self, gpu: &mut Gpu, p: &str, leaf: &str, m: usize, k: usize) -> HipResult<WeightTensor> {
        let name = if p.is_empty() { format!("{leaf}.weight") } else { format!("{p}.{leaf}.weight") };
        load_weight_tensor(self.hfq, gpu, &name, m, k)
    }
    fn load_norm(&self, gpu: &mut Gpu, p: &str, leaf: &str, shape: &[usize]) -> HipResult<GpuTensor> {
        let name = if p.is_empty() { format!("{leaf}.weight") } else { format!("{p}.{leaf}.weight") };
        load_norm_weight(self.hfq, gpu, &name, shape)
    }
    fn load_f32(&self, gpu: &mut Gpu, p: &str, leaf: &str, n: usize) -> HipResult<GpuTensor> {
        let name = if p.is_empty() { leaf.to_string() } else { format!("{p}.{leaf}") };
        load_any_as_f32(self.hfq, gpu, &name, n)
    }
    fn moe_expert_layout(&self) -> MoeExpertLayout { MoeExpertLayout::Fused }
    fn kind(&self) -> BackendKind { BackendKind::Hfq }
    fn raw_bytes(&self, _name: &str) -> Option<Cow<'_, [u8]>> { unimplemented!("Fused layout never repacks") }
    fn after_layer(&self, p: &str) {
        if let Some((start, end)) = self.hfq.layer_data_range(p) {
            self.hfq.drop_pages_range(start, end - start);
        }
    }
}
```

Note: `load_f32` passes a name WITHOUT `.weight` because `load_any_as_f32` today is called with raw tensor names like `{p}.linear_attn.A_log` and `{p}.linear_attn.norm.weight` — i.e. the `.weight`/no-`.weight` is already part of the `leaf` the caller passes. **The generic walk (Task 4) must pass f32 leaves exactly as the current call sites spell them** (e.g. `"linear_attn.A_log"`, `"linear_attn.norm.weight"`, `"linear_attn.conv1d.weight"`). Verify each against `qwen35.rs:1940-1948`.

- [ ] **Step 3: Implement `ParoBackend`**

```rust
pub struct ParoBackend<'a> {
    pub source: &'a dyn crate::model_source::ModelSource,
    gs: u32, kr: u8,
}
impl<'a> ParoBackend<'a> {
    pub fn new(source: &'a dyn crate::model_source::ModelSource) -> Self {
        let qc = source.quant_config().expect("ParoBackend requires quant_config");
        Self { source, gs: qc.group_size, kr: qc.krot }
    }
}
impl<'a> WeightBackend for ParoBackend<'a> {
    fn load_wt(&self, gpu: &mut Gpu, p: &str, leaf: &str, m: usize, k: usize) -> HipResult<WeightTensor> {
        let prefix = if p.is_empty() { leaf.to_string() } else { format!("{p}.{leaf}") };
        paro_load_wt(self.source, gpu, &prefix, m, k, self.gs, self.kr)
    }
    fn load_norm(&self, gpu: &mut Gpu, p: &str, leaf: &str, shape: &[usize]) -> HipResult<GpuTensor> {
        let name = if p.is_empty() { format!("{leaf}.weight") } else { format!("{p}.{leaf}.weight") };
        paro_load_norm(self.source, gpu, &name, shape)
    }
    fn load_f32(&self, gpu: &mut Gpu, p: &str, leaf: &str, n: usize) -> HipResult<GpuTensor> {
        let name = if p.is_empty() { leaf.to_string() } else { format!("{p}.{leaf}") };
        paro_load_f32(self.source, gpu, &name, n)
    }
    fn moe_expert_layout(&self) -> MoeExpertLayout { MoeExpertLayout::ParoRepack }
    fn kind(&self) -> BackendKind { BackendKind::Paro }
    fn raw_bytes(&self, name: &str) -> Option<Cow<'_, [u8]>> {
        self.source.tensor_data(name).map(|(_, d)| Cow::Borrowed(d))
    }
    fn quant_config(&self) -> Option<&QuantConfig> { self.source.quant_config() }
    // after_layer: default no-op (safetensors has no page-drop)
}
```

Cross-check the `paro_load_norm` name form against `qwen35.rs:2046` — confirm it internally prepends `model.language_model.` and expects the `.weight` suffix in the passed name; adjust the `format!` here to match exactly.

- [ ] **Step 4: Build**

Run: `cargo build`
Expected: compiles. Both backends exist but are not yet consumed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(runtime): HfqBackend + ParoBackend impls over relocated helpers"
```

---

## Task 4: Generic `load_layer<B>` (non-MoE arms)

**Files:**
- Modify: `crates/hipfire-arch-qwen35/src/qwen35.rs`

- [ ] **Step 1: Add the generic layer walk**

Add (near the old `load_layer_into`). Use the dense (`is_moe == false`) arms from `load_layer_into` (`qwen35.rs:2337-2377`) as the reference, but call backend methods with BARE leaves. MoE arms call `load_moe_ffn_generic` (Task 5).

```rust
use hipfire_runtime::weight_backend::WeightBackend;

fn load_layer<B: WeightBackend + ?Sized>(
    b: &B, gpu: &mut Gpu, config: &Qwen35Config, layer_idx: usize, p: &str,
) -> HipResult<LayerWeights> {
    let is_moe = config.num_experts > 0;
    Ok(match (config.layer_types[layer_idx], is_moe) {
        (LayerType::LinearAttention, false) => {
            let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
                        + config.linear_num_value_heads * config.linear_value_head_dim;
            let d_inner = config.linear_num_value_heads * config.linear_value_head_dim;
            LayerWeights::DeltaNet(DeltaNetLayerWeights {
                attn_norm: b.load_norm(gpu, p, "input_layernorm", &[config.dim])?,
                wqkv: b.load_wt(gpu, p, "linear_attn.in_proj_qkv", qkv_dim, config.dim)?,
                wz: b.load_wt(gpu, p, "linear_attn.in_proj_z", d_inner, config.dim)?,
                w_alpha: b.load_wt(gpu, p, "linear_attn.in_proj_a", config.linear_num_value_heads, config.dim)?,
                w_beta:  b.load_wt(gpu, p, "linear_attn.in_proj_b", config.linear_num_value_heads, config.dim)?,
                a_log: b.load_f32(gpu, p, "linear_attn.A_log", config.linear_num_value_heads)?,
                dt_bias: b.load_f32(gpu, p, "linear_attn.dt_bias", config.linear_num_value_heads)?,
                conv_weight: b.load_f32(gpu, p, "linear_attn.conv1d.weight", qkv_dim * config.conv_kernel_dim)?,
                norm_weight: b.load_f32(gpu, p, "linear_attn.norm.weight", config.linear_value_head_dim)?,
                wo: b.load_wt(gpu, p, "linear_attn.out_proj", config.dim, d_inner)?,
                ffn_norm: b.load_norm(gpu, p, "post_attention_layernorm", &[config.dim])?,
                w_gate: b.load_wt(gpu, p, "mlp.gate_proj", config.hidden_dim, config.dim)?,
                w_up: b.load_wt(gpu, p, "mlp.up_proj", config.hidden_dim, config.dim)?,
                w_down: b.load_wt(gpu, p, "mlp.down_proj", config.dim, config.hidden_dim)?,
            })
        }
        (LayerType::FullAttention, false) => {
            let q_out_dim = config.n_heads * config.head_dim * 2;
            let kv_dim = config.n_kv_heads * config.head_dim;
            LayerWeights::FullAttn(FullAttnLayerWeights {
                attn_norm: b.load_norm(gpu, p, "input_layernorm", &[config.dim])?,
                wq: b.load_wt(gpu, p, "self_attn.q_proj", q_out_dim, config.dim)?,
                wk: b.load_wt(gpu, p, "self_attn.k_proj", kv_dim, config.dim)?,
                wv: b.load_wt(gpu, p, "self_attn.v_proj", kv_dim, config.dim)?,
                wo: b.load_wt(gpu, p, "self_attn.o_proj", config.dim, config.n_heads * config.head_dim)?,
                q_norm: b.load_norm(gpu, p, "self_attn.q_norm", &[config.head_dim])?,
                k_norm: b.load_norm(gpu, p, "self_attn.k_norm", &[config.head_dim])?,
                ffn_norm: b.load_norm(gpu, p, "post_attention_layernorm", &[config.dim])?,
                w_gate: b.load_wt(gpu, p, "mlp.gate_proj", config.hidden_dim, config.dim)?,
                w_up: b.load_wt(gpu, p, "mlp.up_proj", config.hidden_dim, config.dim)?,
                w_down: b.load_wt(gpu, p, "mlp.down_proj", config.dim, config.hidden_dim)?,
            })
        }
        (LayerType::LinearAttention, true) => {
            let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
                        + config.linear_num_value_heads * config.linear_value_head_dim;
            let d_inner = config.linear_num_value_heads * config.linear_value_head_dim;
            LayerWeights::DeltaNetMoe(DeltaNetMoeLayerWeights {
                attn_norm: b.load_norm(gpu, p, "input_layernorm", &[config.dim])?,
                wqkv: b.load_wt(gpu, p, "linear_attn.in_proj_qkv", qkv_dim, config.dim)?,
                wz: b.load_wt(gpu, p, "linear_attn.in_proj_z", d_inner, config.dim)?,
                w_alpha: b.load_wt(gpu, p, "linear_attn.in_proj_a", config.linear_num_value_heads, config.dim)?,
                w_beta:  b.load_wt(gpu, p, "linear_attn.in_proj_b", config.linear_num_value_heads, config.dim)?,
                a_log: b.load_f32(gpu, p, "linear_attn.A_log", config.linear_num_value_heads)?,
                dt_bias: b.load_f32(gpu, p, "linear_attn.dt_bias", config.linear_num_value_heads)?,
                conv_weight: b.load_f32(gpu, p, "linear_attn.conv1d.weight", qkv_dim * config.conv_kernel_dim)?,
                norm_weight: b.load_f32(gpu, p, "linear_attn.norm.weight", config.linear_value_head_dim)?,
                wo: b.load_wt(gpu, p, "linear_attn.out_proj", config.dim, d_inner)?,
                ffn_norm: b.load_norm(gpu, p, "post_attention_layernorm", &[config.dim])?,
                ffn: load_moe_ffn_generic(b, gpu, p, config, layer_idx as u16)?,
            })
        }
        (LayerType::FullAttention, true) => {
            let q_out_dim = config.n_heads * config.head_dim * 2;
            let kv_dim = config.n_kv_heads * config.head_dim;
            LayerWeights::FullAttnMoe(FullAttnMoeLayerWeights {
                attn_norm: b.load_norm(gpu, p, "input_layernorm", &[config.dim])?,
                wq: b.load_wt(gpu, p, "self_attn.q_proj", q_out_dim, config.dim)?,
                wk: b.load_wt(gpu, p, "self_attn.k_proj", kv_dim, config.dim)?,
                wv: b.load_wt(gpu, p, "self_attn.v_proj", kv_dim, config.dim)?,
                wo: b.load_wt(gpu, p, "self_attn.o_proj", config.dim, config.n_heads * config.head_dim)?,
                q_norm: b.load_norm(gpu, p, "self_attn.q_norm", &[config.head_dim])?,
                k_norm: b.load_norm(gpu, p, "self_attn.k_norm", &[config.head_dim])?,
                ffn_norm: b.load_norm(gpu, p, "post_attention_layernorm", &[config.dim])?,
                ffn: load_moe_ffn_generic(b, gpu, p, config, layer_idx as u16)?,
            })
        }
    })
}
```

- [ ] **Step 2: Build (expect one missing-fn error)**

Run: `cargo build -p hipfire-arch-qwen35`
Expected: FAIL — `load_moe_ffn_generic` not defined (added in Task 5). Confirms the dense arms type-check. If any dense field/name mismatches the current structs, fix to match `qwen35.rs:2337-2377` exactly.

- [ ] **Step 3: Commit (compiles after Task 5; commit together)** — proceed to Task 5 before committing.

---

## Task 5: Generic `load_moe_ffn_generic` (2-arm expert fork)

**Files:**
- Modify: `crates/hipfire-arch-qwen35/src/qwen35.rs`

- [ ] **Step 1: Add the generic MoE loader**

Shared scaffold from `load_moe_ffn` (`qwen35.rs:2422`); the routed-expert loop forks on `b.moe_expert_layout()`. Router/shared/gate use `load_wt` uniformly (identical-by-fallback for Paro). The `Fused` arm body = current `load_moe_ffn` expert loop (`2451-2460`); the `ParoRepack` arm body = current `paro_load_moe_ffn` expert loop (`1312-1367`). The pointer-table + assembly tail is shared.

```rust
fn load_moe_ffn_generic<B: WeightBackend + ?Sized>(
    b: &B, gpu: &mut Gpu, p: &str, config: &Qwen35Config, layer_idx: u16,
) -> HipResult<MoeFfnWeights> {
    let n_exp = config.num_experts;
    let mi = config.moe_intermediate_size;
    let smi = config.shared_expert_intermediate_size;
    let dim = config.dim;

    // Router + shared-expert-gate: load_wt (HFQ quant-aware; Paro FP16 fallback).
    let router = b.load_wt(gpu, p, "mlp.gate", n_exp, dim)?;
    let shared_expert_gate = b.load_wt(gpu, p, "mlp.shared_expert_gate", 1, dim)?;
    let shared_expert = SharedExpertWeights {
        gate: b.load_wt(gpu, p, "mlp.shared_expert.gate_proj", smi, dim)?,
        up:   b.load_wt(gpu, p, "mlp.shared_expert.up_proj",   smi, dim)?,
        down: b.load_wt(gpu, p, "mlp.shared_expert.down_proj", dim, smi)?,
    };

    let (experts, paro_shared) = match b.moe_expert_layout() {
        MoeExpertLayout::Fused => {
            let mut experts = Vec::with_capacity(n_exp);
            for x in 0..n_exp {
                let gate_up = b.load_wt(gpu, p, &format!("mlp.experts.{x}.gate_up_proj"), 2 * mi, dim)?;
                let down    = b.load_wt(gpu, p, &format!("mlp.experts.{x}.down_proj"), dim, mi)?;
                experts.push(ExpertWeights { gate_up, down });
            }
            (experts, None)
        }
        MoeExpertLayout::ParoRepack => {
            // Body relocated from paro_load_moe_ffn (qwen35.rs:1312-1367):
            // paro_load_moe_shared_sidecars + per-expert paro_repack_moe_projection
            // (fetch raw via b.raw_bytes(...)), concat, upload_raw, alias_paro_rotation.
            paro_load_moe_experts(b, gpu, p, config)?
        }
    };

    // Shared tail: device pointer tables + assembly (verbatim from load_moe_ffn:2467-2489).
    let mut gu_ptrs: Vec<u64> = Vec::with_capacity(n_exp);
    let mut dn_ptrs: Vec<u64> = Vec::with_capacity(n_exp);
    for e in &experts { gu_ptrs.push(e.gate_up.buf.buf.as_ptr() as u64); dn_ptrs.push(e.down.buf.buf.as_ptr() as u64); }
    let gu_bytes: Vec<u8> = gu_ptrs.iter().flat_map(|q| q.to_ne_bytes()).collect();
    let dn_bytes: Vec<u8> = dn_ptrs.iter().flat_map(|q| q.to_ne_bytes()).collect();
    let expert_gate_up_ptrs = gpu.alloc_tensor(&[2 * n_exp], DType::F32)?;
    let expert_down_ptrs    = gpu.alloc_tensor(&[2 * n_exp], DType::F32)?;
    gpu.hip.memcpy_htod(&expert_gate_up_ptrs.buf, &gu_bytes)?;
    gpu.hip.memcpy_htod(&expert_down_ptrs.buf,    &dn_bytes)?;

    Ok(MoeFfnWeights {
        router, experts, shared_expert, shared_expert_gate,
        expert_gate_up_ptrs, expert_down_ptrs, layer_idx,
        expert_shape: None, paro_shared,
    })
}
```

- [ ] **Step 2: Add `paro_load_moe_experts` helper (extract from `paro_load_moe_ffn`)**

Move the routed-expert loop body of `paro_load_moe_ffn` (`qwen35.rs:1312-1367`) into a `fn paro_load_moe_experts<B: WeightBackend + ?Sized>(b, gpu, p, config) -> HipResult<(Vec<ExpertWeights>, Option<MoeParoSidecars>)>`. Replace its `source.tensor_data(...)` reads inside `paro_repack_moe_projection` calls with `b.raw_bytes(...)`. Keep `paro_load_moe_shared_sidecars` + `alias_paro_rotation` calls. Returns `(experts, Some(shared))`.

Note: `paro_load_moe_shared_sidecars` currently takes `source: &dyn ModelSource`. Give it `b.raw_bytes`/a `&dyn ModelSource` accessor — simplest is to add `fn source_for_repack(&self) -> Option<&dyn ModelSource>` to `ParoBackend` only, OR pass `b.raw_bytes` closures. Prefer keeping `paro_load_moe_shared_sidecars` taking the raw-bytes accessor.

- [ ] **Step 3: Build**

Run: `cargo build -p hipfire-arch-qwen35`
Expected: compiles. `load_layer`/`load_moe_ffn_generic` exist but unused (warnings OK).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(qwen35): generic load_layer + load_moe_ffn over WeightBackend"
```

---

## Task 6: Embed/output `kind()`-selected dispatch

**Files:**
- Modify: `crates/hipfire-arch-qwen35/src/qwen35.rs`

- [ ] **Step 1: Add generic embed/output dispatchers**

These wrap the EXISTING bodies. Keep `load_token_embd_into`/`load_output_into` (HFQ, `2227`/`2258`) and the Paro inline embed/output (`2070-2099`) — extract the Paro inline into `load_token_embd_paro`/`load_output_paro` fns. Then:

```rust
fn load_token_embd_generic<B: WeightBackend + ?Sized>(
    b: &B, gpu: &mut Gpu, config: &Qwen35Config,
) -> HipResult<(GpuTensor, EmbeddingFormat)> {
    match b.kind() {
        BackendKind::Hfq  => load_token_embd_into(hfq_of(b), config, gpu),
        BackendKind::Paro => load_token_embd_paro(source_of(b), gpu, config),
    }
}
// analogous load_output_generic -> (GpuTensor /*norm*/, WeightTensor /*lm_head*/)
```

`hfq_of`/`source_of` need the concrete backing store. Cleanest: add `fn as_hfq(&self) -> Option<&HfqFile>` and `fn as_source(&self) -> Option<&dyn ModelSource>` to the `WeightBackend` trait (default `None`), implemented on the respective backend. The two embed/output forks call the matching accessor. Document these as the embed/output-only escape hatch.

- [ ] **Step 2: Add `as_hfq`/`as_source` to the trait + impls**

In `weight_backend.rs` trait: `fn as_hfq(&self) -> Option<&crate::hfq::HfqFile> { None }` and `fn as_source(&self) -> Option<&dyn crate::model_source::ModelSource> { None }`. `HfqBackend` returns `Some(self.hfq)` / `None`; `ParoBackend` the reverse.

- [ ] **Step 3: Build**

Run: `cargo build`
Expected: compiles.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(qwen35): kind()-selected embed/output dispatch"
```

---

## Task 7: `load_weights_generic` + rewrite public entry points as wrappers

**Files:**
- Modify: `crates/hipfire-arch-qwen35/src/qwen35.rs`

- [ ] **Step 1: Add the single-GPU generic driver**

```rust
pub fn load_weights_generic<B: WeightBackend + ?Sized>(
    b: &B, config: &Qwen35Config, gpu: &mut Gpu,
) -> HipResult<Qwen35Weights> {
    let (token_embd, embd_format) = load_token_embd_generic(b, gpu, config)?;
    let (output_norm, output) = load_output_generic(b, gpu, config)?;
    let mut layers = Vec::with_capacity(config.n_layers);
    for i in 0..config.n_layers {
        let p = format!("layers.{i}");
        layers.push(load_layer(b, gpu, config, i, &p)?);
        b.after_layer(&p);
    }
    Ok(Qwen35Weights { token_embd, embd_format, output_norm, output, layers, pager: None })
}
```

Cross-check the embed/output ORDER and the `drop_mmap` placement against current `load_weights` (`qwen35.rs:1808-1865`): `drop_mmap()` happens in the wrapper (Step 2), embeddings before norm/output before the layer loop.

- [ ] **Step 2: Rewrite `load_weights` and `load_weights_multi` as thin wrappers**

```rust
pub fn load_weights(hfq: &mut HfqFile, config: &Qwen35Config, gpu: &mut Gpu) -> HipResult<Qwen35Weights> {
    hfq.drop_mmap();
    let b = hipfire_runtime::weight_backend::HfqBackend::new(hfq);
    load_weights_generic(&b, config, gpu)
}
```

For `load_weights_multi` (`qwen35.rs:2195`): keep its signature `(&HfqFile, &Qwen35Config, &mut Gpus)`. Its per-layer loop builds a fresh `HfqBackend::new(hfq)` borrow is not possible across devices simultaneously — instead, inside the loop construct `HfqBackend::new(hfq)` once before the loop and call `load_layer(&b, &mut gpus.devices[dev], config, i, &p)?` then `b.after_layer(&p)`. Preserve the existing `device_for_layer`, the per-layer logging, and drop the now-redundant inline `drop_pages_range` (after_layer handles it). embed/output via `load_token_embd_generic`/`load_output_generic` on `&b` targeting the right device (match current device placement at `2200-2203`).

- [ ] **Step 3: Build the workspace**

Run: `cargo build`
Expected: compiles. `load_weights`/`load_weights_multi` now route through the generic driver; `load_layer_into` is now unused (warning) — deleted in Task 8.

- [ ] **Step 4: Coherence + characterization (HFQ path now unified)**

Run: `cargo run -q -p hipfire-runtime --example weight_capture -- <same .hfq model> > .scratch/loadcap-hfq-after.txt`
Run: `diff .scratch/loadcap-hfq-before.txt .scratch/loadcap-hfq-after.txt`
Expected: **no diff**. If different, a name/bake/order regression exists — bisect against the dense-arm names and the f32/norm split before proceeding.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(qwen35): load_weights/_multi as thin HfqBackend wrappers over generic driver"
```

---

## Task 8: Swap the daemon Paro path + delete dead loaders

**Files:**
- Modify: `crates/hipfire-runtime/examples/daemon.rs`
- Modify: `crates/hipfire-arch-qwen35/src/qwen35.rs`

- [ ] **Step 1: Repoint the daemon safetensors path**

In `daemon.rs` (~1969), replace:
```rust
let weights = qwen35::load_weights_paroquant(&source, &config, gpu)
```
with:
```rust
let backend = hipfire_runtime::weight_backend::ParoBackend::new(&*source);
let weights = qwen35::load_weights_generic(&backend, &config, gpu)
```
Keep the `.map_err(...)` tail.

- [ ] **Step 2: Delete the now-dead fns from qwen35.rs**

Delete `load_weights_paroquant` (`2061`), `load_layer_into` (`2328`), `paro_load_moe_ffn` (`1274`). Their bodies are now covered by `load_weights_generic` + `load_layer` + `load_moe_ffn_generic` + `paro_load_moe_experts`.

- [ ] **Step 3: Build + warning check**

Run: `cargo build 2>&1 | grep -E "warning: (function|method) .* never used" || echo CLEAN`
Expected: `CLEAN` (or only pre-existing unrelated warnings). Fix any newly-orphaned helper (e.g. if `paro_load_moe_shared_sidecars` is now only called from `paro_load_moe_experts`, that's fine — it IS called).

- [ ] **Step 4: Characterization — PARO path now unified**

Run: `cargo run -q -p hipfire-runtime --example weight_capture -- <A3B PARO dir> > .scratch/loadcap-paro-after.txt`
Run: `diff .scratch/loadcap-paro-before.txt .scratch/loadcap-paro-after.txt`
Expected: **no diff**.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(qwen35): delete load_weights_paroquant/_layer_into/paro_load_moe_ffn; daemon uses ParoBackend"
```

---

## Task 9: Full verification gate

**Files:** none (verification only)

- [ ] **Step 1: Coherence gate (mandatory — touches the loader)**

Run: `./scripts/coherence-gate.sh`
Expected: PASS for both an HFQ model and the A3B PARO model — fluent, on-topic, no loops. (The pre-commit hook also runs this when loader files are staged.)

- [ ] **Step 2: Page-drop behavior check (after_layer — not caught by the diff)**

Run the HFQ daemon load on a model and confirm `after_layer` fires per layer:
`HIPFIRE_MEMSET_DUMP=0 cargo run -q -p hipfire-runtime --example weight_capture -- <.hfq model>` and watch RSS does not grow unbounded across layers (per-layer `drop_pages_range` still active). If unsure, add a temporary `eprintln!` in `HfqBackend::after_layer` and confirm N-layers invocations, then remove it.

- [ ] **Step 3: Multi-GPU parity (if a 2-GPU box is available)**

Run: `cargo test -p hipfire-arch-qwen35 --test pp_parity` (or `examples/pp_parity.rs`). Expected: PASS / single-vs-multi logits match. If no multi-GPU hardware, note it as untested and rely on the unchanged `load_weights_multi` signature + shared `load_layer`.

- [ ] **Step 4: Commit (no-op if clean) / record results**

```bash
git commit --allow-empty -m "test(qwen35): coherence + characterization green post-unification"
```

---

## Task 10: Deferred-items markers + reminder

**Files:**
- Modify: `crates/hipfire-runtime/src/weight_backend.rs`
- Modify: `crates/hipfire-arch-qwen35/src/qwen35.rs`

- [ ] **Step 1: Add `@todo` comments at the deferred code paths**

Place these exact markers:

1. At the top of the `match b.moe_expert_layout()` in `load_moe_ffn_generic` and the `match b.kind()` in `load_token_embd_generic`/`load_output_generic`:
```rust
// @todo(unified-loading): deferred fork — kept in qwen35 rather than the
// WeightBackend trait (returns qwen35 types). See
// docs/plans/2026-06-11-qwen35-unified-weight-loading-design.md "Residual forks".
```

2. In `load_weights_multi`, above the loop:
```rust
// @todo(unified-loading): multi-GPU is HFQ-only. ParoBackend has no page-drop /
// band-routing equivalent; safetensors multi-GPU is deferred. Asserts kind()==Hfq.
```
Add a runtime guard: `debug_assert_eq!(b.kind(), BackendKind::Hfq, "multi-GPU is HFQ-only");`

3. In `weight_backend.rs`, doc-comment on `as_hfq`/`as_source`:
```rust
// @todo(unified-loading): embed/output escape hatch. Remove if/when embed/output
// loses its tied-detection + AWQ-probe divergence and moves onto the trait.
```

4. On the llama path `hfq.rs:load_weights_paroquant_llama` (~1050), a one-line pointer:
```rust
// @todo(unified-loading): llama family not yet migrated to WeightBackend. See plan.
```

- [ ] **Step 2: Append a Deferred Items reminder to the design doc**

Append to `docs/plans/2026-06-11-qwen35-unified-weight-loading-design.md`:

```markdown
## Deferred items (post-merge reminder)

This cut intentionally left these for follow-up — grep `@todo(unified-loading)`:
1. **Llama family** — `hfq.rs:load_weights_paroquant_llama` still has its own
   loader; migrate it onto `WeightBackend` (the trait is arch-agnostic).
2. **Multi-GPU for safetensors/Paro** — `load_weights_multi` asserts `kind()==Hfq`;
   needs a `ModelSource` page-drop/band-routing story before Paro multi-GPU.
3. **Embed/output on the trait** — currently `kind()`-selected free fns via
   `as_hfq`/`as_source`; fold onto the trait if the tied-detection + AWQ-probe
   divergence is unified.
4. **MoE-expert packing on the trait** — the `Fused`/`ParoRepack` arm stays in
   qwen35 (produces qwen35 types); revisit if a third layout appears.
```

- [ ] **Step 3: Build + commit**

Run: `cargo build`
Expected: compiles (the `debug_assert_eq!` needs `BackendKind` in scope in qwen35 — add the import).

```bash
git add -A
git commit -m "docs(qwen35): @todo markers + deferred-items reminder for unified loading"
```

---

## Deferred items reminder (plan-level)

**This is the final task on purpose.** After Task 10, the four deferred paths (llama migration, Paro multi-GPU, embed/output-on-trait, MoE-packing-on-trait) are each marked in-code with `@todo(unified-loading)` and listed in the design doc's "Deferred items" section. Before declaring the work done, `grep -rn "@todo(unified-loading)"` and confirm all four markers are present and point at the design doc.
