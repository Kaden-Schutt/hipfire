# qwen35 Unified Native-Multi-GPU Loader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Canonical save location on approval:** copy this file to `docs/superpowers/plans/2026-06-12-qwen35-unified-loader.md` (written here under plan-mode constraints).

**Goal:** Replace the three `load_weights*` entry points in `qwen35.rs` with **one** weight loader that transparently handles HFQ or PaRo sources and treats single-GPU as the `n==1` case of native multi-GPU — keeping greedy-token output byte-identical.

**Architecture:** A single `assemble_weights` driver operates over a device slice (`&mut [Gpu]`) + a `Layout` (output_device + per-layer routing); single-GPU is just `len()==1`. The HFQ-vs-PaRo difference is isolated behind a new whole-model `WeightSource` trait (`HfqSource`/`ParoSource`) that takes the target `&mut Gpu` per call and reuses the existing `WeightBackend` + `load_layer<B>` for per-layer reads. Two source impls + one driver replace three assembly paths.

**Tech Stack:** Rust, `hipfire-arch-qwen35`, `hipfire-runtime::{weight_backend, multi_gpu, paro, model_source}`.

---

## Context

`unified-loading-review-todos.md` #4: `qwen35.rs` (~12,739 lines) has three near-parallel weight loaders (`load_weights` single-GPU HFQ, `load_weights_paroquant` safetensors, `load_weights_multi` multi-GPU HFQ) plus inline embed/output and a duplicated AWQ-sidecar block. #1–#3 already unified the per-tensor read (`WeightBackend`) and per-layer schema (`load_layer<B>`); the carrier registry (Tier 1) already table-dispatches model selection. The **whole-model assembly** (Tier 2) is the un-unified rung.

Design decisions (confirmed with user):
- **Native multi-GPU; single-GPU = `n==1`.** The loader works over a device slice + layout, not a single `&mut Gpu`. (`Gpus` *owns* `devices: Vec<Gpu>` at `multi_gpu.rs:65`, so the unifying abstraction is a device **slice**, which both a lone `&mut Gpu` (`slice::from_mut`) and `&mut gpus.devices` provide.)
- **Transparent HFQ/PaRo** via one `WeightSource` trait, two impls — not three loaders.
- **HFQ-only multi-GPU**; PaRo stays single-device (its `prepare(n>1)` errors). Matches the existing `qwen35.rs:1563` @todo (band-routing is HFQ-mmap-specific; `drop_pages_range` has no `ModelSource` equivalent).
- **Byte-identical to today**: tied-embedding alias gated to `len()==1`; each source reproduces today's reads. The greedy-token guard must show zero diff.
- **Public API collapse is the LAST task**: Tasks 1–4 keep the three public wrappers (byte-identical, low-churn); Task 5 collapses to one public `load_weights` and migrates callers.

```
TIER 1  carrier registry          carriers.rs                         unchanged
TIER 2  load_weights (unified)     qwen35.rs assemble_weights(&mut[Gpu], Layout, WeightSource)   ← THIS PLAN
TIER 2.5 WeightSource             HfqSource / ParoSource (whole-model read, device-per-call)     ← new
TIER 3  load_layer<B>+WeightBackend  layer_driver.rs / weight_backend.rs   reused inside read_layer
```

**Worktree:** `/home/bjoern/hipfire/.worktrees/feature-paro-transparent-loading` (branch `feature/paro-transparent-loading`). Target: `crates/hipfire-arch-qwen35/src/qwen35.rs`.

---

## File Structure

- **`crates/hipfire-arch-qwen35/src/qwen35.rs`** (modify) — add `Layout`, `WeightSource` trait, `HfqSource`, `ParoSource`, `assemble_weights`, `attach_lm_head_awq_sidecar`; rewrite the three `load_weights*` as wrappers (Tasks 2–3), then collapse to one public entry (Task 5). Delete `load_output_into`; fold `load_token_embd_into` into `HfqSource`; keep `load_layer_into` (reused by `HfqSource::read_layer`).
- **`docs/superpowers/specs/2026-06-11-carrier-registry-unified-design.md`** (modify) — Tier-2 reconciliation note (Task 4).
- **Callers** (modify, Task 5 only) — ~40 sites migrate to the single entry; pattern described once below.

### Reused existing code (do NOT reimplement)
- `crate::layer_driver::load_layer<B>` (`layer_driver.rs:19`); `HfqBackend`/`ParoBackend` (`weight_backend.rs:980`/`:1034`).
- In-file: `load_layer_into`, `load_norm_weight`, `load_weight_tensor_raw`, `qwen35_tensor_data{,_vec}`.
- `weight_backend`: `load_embedding`, `dequant_weight_raw`, `embedding_format_dtype`, `load_awq_scale_for` (imported, lines 19–22).
- `paro`: `paro_text_prefix`, `paro_load_norm` (imported, 23–25). `Gpus::{output_device field, device_for_layer}` (`multi_gpu.rs:74,275`).

### Verification model (TDD-for-refactor)
The "test" is a **characterization guard**: capture greedy (temp 0) token output BEFORE changes, assert byte-identical after the core task (3) and the collapse (5). Plus `cargo build`/`clippy` per step and `scripts/coherence-gate.sh` (mandatory) before each commit touching qwen35.rs. GPU access via `scripts/gpu-lock.sh` (k9lin/gfx1100). Multi-GPU routing (`n>1`) cannot be functionally validated locally (needs hiptrx) — `n==1` is the local guard; the `n>1` code path is exercised by `cargo test -p hipfire-arch-qwen35 --test pp_parity` build + review.

---

## Task 1: Capture the byte-identical baseline (the guard)

**Files:** Create scratch dir `/home/bjoern/hipfire-refactor-baseline/` (NOT /tmp, per user rule).

- [ ] **Step 1: Pick one HFQ + one PaRo model**

Run:
```bash
ls -1 ~/.hipfire/models/
```
Expected: at least one `*.hfq` and one PaRo dir (e.g. `shisa-*-PARO-packed`). Record as `$HFQ_MODEL` / `$PARO_MODEL`.

- [ ] **Step 2: Build greedy dumpers on the unmodified tree**

Run:
```bash
cd /home/bjoern/hipfire/.worktrees/feature-paro-transparent-loading
source scripts/gpu-lock.sh && gpu_acquire "loader-baseline"
cargo build --release -p hipfire-runtime --example greedy_dump --example eval_hipfire
```
Expected: compiles clean.

- [ ] **Step 3: Capture HFQ + PaRo baselines**

Run:
```bash
mkdir -p /home/bjoern/hipfire-refactor-baseline
./target/release/examples/greedy_dump --model "$HFQ_MODEL" \
  --prompt-file benchmarks/prompts/lru_cache_pep8_strict.txt --max-tokens 64 \
  > /home/bjoern/hipfire-refactor-baseline/hfq.before.txt 2>/dev/null
./target/release/examples/eval_hipfire --model "$PARO_MODEL" --greedy --max-tokens 64 \
  > /home/bjoern/hipfire-refactor-baseline/paro.before.txt 2>/dev/null
gpu_release
md5sum /home/bjoern/hipfire-refactor-baseline/*.before.txt
```
Expected: two deterministic dumps + md5s. These are the guard. (If a flag differs, `--help`; requirement = deterministic per-token dump.)

---

## Task 2: Extract the duplicated AWQ-sidecar block

Byte-identical at `qwen35.rs:1406-1427` (`load_weights`) and `1672-1693` (`load_output_into`). Extract in isolation to de-risk Task 3.

**Files:** Modify `crates/hipfire-arch-qwen35/src/qwen35.rs`.

- [ ] **Step 1: Add the helper** (above `pub fn load_weights`, ~1316)

```rust
/// Attach the lm_head / tied-embed AWQ sidecar when the output dtype supports it.
/// Byte-identical no-op on current files. MUST be called AFTER `output.gpu_dtype`
/// is set (the gate reads it). See docs/plans/awq_fix_claude.md.
fn attach_lm_head_awq_sidecar(hfq: &HfqFile, gpu: &Gpu, output: &mut WeightTensor, k: usize) {
    if output.gpu_dtype.supports_awq_sidecar() {
        output.awq_scale = load_awq_scale_for(hfq, gpu, "lm_head.weight", k)
            .or_else(|| load_awq_scale_for(hfq, gpu, "model.language_model.lm_head.weight", k))
            .or_else(|| load_awq_scale_for(hfq, gpu, "model.language_model.embed_tokens.weight", k));
        eprintln!(
            "  lm_head AWQ sidecar: {}",
            if output.awq_scale.is_some() { "attached" } else { "absent (no-op)" }
        );
    }
}
```

- [ ] **Step 2: Replace both blocks with `attach_lm_head_awq_sidecar(hfq, gpu, &mut output, config.dim);`**

In `load_weights` (1406–1427) and `load_output_into` (1672–1693), replace the `if output.gpu_dtype.supports_awq_sidecar() { … }` block with the one-line call.

- [ ] **Step 3: Build + clippy + byte-identical + commit**

Run:
```bash
cd /home/bjoern/hipfire/.worktrees/feature-paro-transparent-loading
cargo build -p hipfire-arch-qwen35 && cargo clippy -p hipfire-arch-qwen35
source scripts/gpu-lock.sh && gpu_acquire "loader-awq"
cargo build --release -p hipfire-runtime --example greedy_dump
./target/release/examples/greedy_dump --model "$HFQ_MODEL" \
  --prompt-file benchmarks/prompts/lru_cache_pep8_strict.txt --max-tokens 64 \
  > /home/bjoern/hipfire-refactor-baseline/hfq.awq.txt 2>/dev/null
gpu_release
diff /home/bjoern/hipfire-refactor-baseline/hfq.before.txt /home/bjoern/hipfire-refactor-baseline/hfq.awq.txt
```
Expected: no diff. Then:
```bash
cargo fmt -p hipfire-arch-qwen35
git add crates/hipfire-arch-qwen35/src/qwen35.rs
git commit -m "refactor(loading): extract attach_lm_head_awq_sidecar helper"
```

---

## Task 3: Add `Layout` + `WeightSource` + driver; rewrite the three publics as wrappers

Core change. One commit, ordered steps. `HfqSource` holds `&HfqFile` (shared) this task; `drop_mmap` stays in the single wrapper (Task 5 tightens this).

**Files:** Modify `crates/hipfire-arch-qwen35/src/qwen35.rs`.

- [ ] **Step 1: Add `Layout`** (below the AWQ helper)

```rust
/// Where each piece of the model lands across a device slice. `single` = the
/// n==1 degenerate case (everything on device 0).
struct Layout {
    output_device: usize,
    layer_to_device: Vec<usize>,
}
impl Layout {
    fn single(n_layers: usize) -> Self {
        Self { output_device: 0, layer_to_device: vec![0; n_layers] }
    }
    fn from_gpus(g: &Gpus, n_layers: usize) -> Self {
        Self {
            output_device: g.output_device,
            layer_to_device: (0..n_layers).map(|i| g.device_for_layer(i)).collect(),
        }
    }
    fn device_for_layer(&self, i: usize) -> usize { self.layer_to_device[i] }
}
```

- [ ] **Step 2: Add the `WeightSource` trait + `assemble_weights` driver**

```rust
/// Whole-model weight source — the one place HFQ vs PaRo differs. Each method
/// uploads to the caller-chosen `gpu` (native multi-GPU: the driver picks the
/// device). `read_layer` reuses Tier-3 `load_layer<B>` internally.
trait WeightSource {
    /// Pre-load hook. HFQ drops the mmap when n==1; PaRo rejects n>1.
    fn prepare(&mut self, n_devices: usize) -> HipResult<()>;
    fn read_embed(&mut self, gpu: &mut Gpu, c: &Qwen35Config) -> HipResult<(GpuTensor, EmbeddingFormat)>;
    fn read_final_norm(&mut self, gpu: &mut Gpu, c: &Qwen35Config) -> HipResult<GpuTensor>;
    /// `can_alias` is true iff embed and output share a device (n==1); then the
    /// tied lm_head aliases the embedding buffer instead of re-uploading.
    fn read_output(
        &mut self, gpu: &mut Gpu, c: &Qwen35Config,
        embd: &GpuTensor, embd_fmt: EmbeddingFormat, can_alias: bool,
    ) -> HipResult<(WeightTensor, bool)>;
    fn read_layer(&mut self, gpu: &mut Gpu, c: &Qwen35Config, layer_idx: usize) -> HipResult<LayerWeights>;
}

fn assemble_weights(
    source: &mut dyn WeightSource,
    devices: &mut [Gpu],
    layout: &Layout,
    config: &Qwen35Config,
) -> HipResult<Qwen35Weights> {
    source.prepare(devices.len())?;
    let out_dev = layout.output_device;
    let can_alias = devices.len() == 1; // byte-identical: alias only on the single-device path
    let (token_embd, embd_format) = source.read_embed(&mut devices[0], config)?;
    let output_norm = source.read_final_norm(&mut devices[out_dev], config)?;
    let (output, lm_head_aliases_embd) =
        source.read_output(&mut devices[out_dev], config, &token_embd, embd_format, can_alias)?;
    let mut layers = Vec::with_capacity(config.n_layers);
    for i in 0..config.n_layers {
        let d = layout.device_for_layer(i);
        layers.push(source.read_layer(&mut devices[d], config, i)?);
    }
    Ok(Qwen35Weights {
        token_embd, embd_format, output_norm, output, layers,
        pager: None, lm_head_aliases_embd,
    })
}
```

- [ ] **Step 3: Add `HfqSource`** (reproduces today's HFQ single+multi exactly; embed/output read via pread `_vec`, which yields bytes identical to today's reads — verified by the guard)

```rust
struct HfqSource<'a> {
    hfq: &'a HfqFile,
}
impl<'a> HfqSource<'a> {
    fn new(hfq: &'a HfqFile) -> Self { Self { hfq } }
}
impl WeightSource for HfqSource<'_> {
    fn prepare(&mut self, _n_devices: usize) -> HipResult<()> {
        Ok(()) // drop_mmap handled by the single wrapper this task (see Task 5)
    }

    fn read_embed(&mut self, gpu: &mut Gpu, c: &Qwen35Config) -> HipResult<(GpuTensor, EmbeddingFormat)> {
        eprintln!("  loading token_embd...");
        if c.is_vl_text {
            eprintln!("  qwen3.5-vl text wrapper: mrope_interleaved={} mrope_section={:?}",
                c.mrope_interleaved, c.mrope_section);
        }
        let (embd_meta, embd_data) = qwen35_tensor_data_vec(self.hfq, "embed_tokens.weight")
            .expect("embed_tokens not found");
        let out = load_embedding(gpu, embd_meta.quant_type, &embd_data, c.vocab_size, c.dim)?;
        drop(embd_data);
        Ok(out)
    }

    fn read_final_norm(&mut self, gpu: &mut Gpu, c: &Qwen35Config) -> HipResult<GpuTensor> {
        eprintln!("  loading output_norm...");
        load_norm_weight(self.hfq, gpu, "norm.weight", &[c.dim])
    }

    fn read_output(
        &mut self, gpu: &mut Gpu, c: &Qwen35Config,
        embd: &GpuTensor, embd_fmt: EmbeddingFormat, can_alias: bool,
    ) -> HipResult<(WeightTensor, bool)> {
        let lm_head_info = qwen35_tensor_data_vec(self.hfq, "lm_head.weight");
        let lm_head_is_tied = lm_head_info.is_none();
        let mut output = if let Some((lm_info, lm_data)) = lm_head_info {
            eprintln!("  loading output (separate lm_head, qt={})...", lm_info.quant_type);
            load_weight_tensor_raw(gpu, lm_info.quant_type, &lm_data, c.vocab_size, c.dim)?
        } else if can_alias {
            eprintln!("  loading output (tied embeddings, aliased)...");
            WeightTensor {
                buf: embd.shallow_clone(),
                gpu_dtype: embedding_format_dtype(embd_fmt),
                m: c.vocab_size, k: c.dim, row_stride: 0, paro: None, awq_scale: None,
            }
        } else {
            let (embd_meta, embd_data) = qwen35_tensor_data_vec(self.hfq, "embed_tokens.weight")
                .expect("embed_tokens not found");
            eprintln!("  loading output (tied embeddings, reupload qt={})...", embd_meta.quant_type);
            dequant_weight_raw(gpu, embd_meta.quant_type, &embd_data, c.vocab_size, c.dim)?
        };
        attach_lm_head_awq_sidecar(self.hfq, gpu, &mut output, c.dim);
        Ok((output, lm_head_is_tied && can_alias))
    }

    fn read_layer(&mut self, gpu: &mut Gpu, c: &Qwen35Config, layer_idx: usize) -> HipResult<LayerWeights> {
        let is_moe = c.num_experts > 0;
        eprintln!("  loading layer {layer_idx}/{} ({:?}{})...",
            c.n_layers, c.layer_types[layer_idx], if is_moe { " + MoE" } else { "" });
        let p = format!("layers.{layer_idx}");
        let page = self.hfq.layer_data_range(&p);
        let lw = load_layer_into(self.hfq, c, layer_idx, &p, gpu)?;
        if let Some((start, end)) = page { self.hfq.drop_pages_range(start, end - start); }
        Ok(lw)
    }
}
```

Note: `lm_head_aliases_embd` is `tied && can_alias` — true only for n==1 tied (today's single); n>1 tied reuploads → `false` (today's multi). Byte-identical.

- [ ] **Step 4: Add `ParoSource`** (F32; never aliases; `prepare` rejects n>1)

```rust
struct ParoSource<'a> {
    source: &'a dyn ModelSource,
    mp: &'static str,
}
impl<'a> ParoSource<'a> {
    fn new(source: &'a dyn ModelSource) -> HipResult<Self> {
        source.quant_config()
            .ok_or_else(|| HipError::new(0, "ParoQuant model must have quantization_config"))?;
        let mp = paro_text_prefix(source)?;
        Ok(Self { source, mp })
    }
    fn read_f16_as_f32(&self, name: &str) -> HipResult<Vec<f32>> {
        let (_, data) = self.source.tensor_data(name)
            .ok_or_else(|| HipError::new(0, &format!("PARO tensor not found: {name}")))?;
        Ok(data.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect())
    }
}
impl WeightSource for ParoSource<'_> {
    fn prepare(&mut self, n_devices: usize) -> HipResult<()> {
        if n_devices > 1 {
            return Err(HipError::new(0, "ParoQuant multi-GPU loading is not supported (HFQ-only)"));
        }
        Ok(())
    }

    fn read_embed(&mut self, gpu: &mut Gpu, c: &Qwen35Config) -> HipResult<(GpuTensor, EmbeddingFormat)> {
        eprintln!("  loading token_embd (ParoQuant)...");
        let f32_embd = self.read_f16_as_f32(&format!("{}.embed_tokens.weight", self.mp))?;
        let token_embd = gpu.upload_f32(&f32_embd, &[c.vocab_size, c.dim])?;
        Ok((token_embd, EmbeddingFormat::F32))
    }

    fn read_final_norm(&mut self, gpu: &mut Gpu, c: &Qwen35Config) -> HipResult<GpuTensor> {
        eprintln!("  loading output_norm...");
        paro_load_norm(self.source, gpu, "norm.weight", &[c.dim], 1.0)
    }

    fn read_output(
        &mut self, gpu: &mut Gpu, c: &Qwen35Config,
        _embd: &GpuTensor, _embd_fmt: EmbeddingFormat, _can_alias: bool,
    ) -> HipResult<(WeightTensor, bool)> {
        let embd_name = format!("{}.embed_tokens.weight", self.mp);
        let (src_name, tied) = if self.source.tensor_data("lm_head.weight").is_some() {
            (String::from("lm_head.weight"), false)
        } else {
            (embd_name, true)
        };
        eprintln!("  loading output ({})...", if tied { "tied embeddings" } else { "separate lm_head" });
        let f = self.read_f16_as_f32(&src_name)?;
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(f.as_ptr() as *const u8, f.len() * 4) };
        let buf = gpu.upload_raw(bytes, &[c.vocab_size, c.dim])?;
        let output = WeightTensor {
            buf, gpu_dtype: DType::F32, m: c.vocab_size, k: c.dim,
            row_stride: 0, paro: None, awq_scale: None,
        };
        Ok((output, false))
    }

    fn read_layer(&mut self, gpu: &mut Gpu, c: &Qwen35Config, layer_idx: usize) -> HipResult<LayerWeights> {
        eprintln!("  loading layer {layer_idx}/{} ({:?}, ParoQuant)...",
            c.n_layers, c.layer_types[layer_idx]);
        let mut b = ParoBackend { source: self.source, gpu, mp: self.mp, layer: layer_idx, norm_bias: 1.0 };
        let moe = |bk: &mut ParoBackend, cfg: &Qwen35Config, li: usize| {
            crate::paro_moe::paro_load_moe_ffn(bk.source, bk.gpu, &format!("layers.{li}"), cfg, li as u16)
        };
        crate::layer_driver::load_layer(&mut b, c, layer_idx, moe)
    }
}
```

Note: the PaRo `read_output` no longer reconstructs `embd_name` to reuse the embed tensor across calls — it re-resolves the source name, byte-identical to today's `load_weights_paroquant`. The original `on dev` / per-layer logs are reproduced per impl.

- [ ] **Step 5: Rewrite the three publics as wrappers**

Replace `load_weights` body (1316–1462):
```rust
pub fn load_weights(hfq: &mut HfqFile, config: &Qwen35Config, gpu: &mut Gpu) -> HipResult<Qwen35Weights> {
    #[cfg(unix)]
    hfq.drop_mmap(); // single-GPU only; embed/output read via pread, so safe
    let mut source = HfqSource::new(hfq);
    let layout = Layout::single(config.n_layers);
    assemble_weights(&mut source, std::slice::from_mut(gpu), &layout, config)
}
```

Replace `load_weights_paroquant` body (1465–1561):
```rust
pub fn load_weights_paroquant(source: &dyn ModelSource, config: &Qwen35Config, gpu: &mut Gpu) -> HipResult<Qwen35Weights> {
    let mut src = ParoSource::new(source)?;
    let layout = Layout::single(config.n_layers);
    assemble_weights(&mut src, std::slice::from_mut(gpu), &layout, config)
}
```

Replace `load_weights_multi` body (1573–1615), **keeping its doc comment block** (1563–1572):
```rust
pub fn load_weights_multi(hfq: &HfqFile, config: &Qwen35Config, gpus: &mut Gpus) -> HipResult<Qwen35Weights> {
    let layout = Layout::from_gpus(gpus, config.n_layers);
    let mut source = HfqSource::new(hfq);
    assemble_weights(&mut source, &mut gpus.devices, &layout, config)
}
```
(`Layout::from_gpus` borrows `gpus` immutably and returns an owned `Layout` before `&mut gpus.devices` is taken — no borrow conflict.)

- [ ] **Step 6: Delete dead helpers**

Delete `load_output_into` (1633–1695) and `load_token_embd_into` (1617–1631) — both subsumed by `HfqSource`. **Keep** `load_layer_into` (used by `HfqSource::read_layer`).

- [ ] **Step 7: Build, clippy, daemon, multi build**

Run:
```bash
cd /home/bjoern/hipfire/.worktrees/feature-paro-transparent-loading
cargo build -p hipfire-arch-qwen35 --examples
cargo clippy -p hipfire-arch-qwen35
cargo build --example daemon -p hipfire-runtime
cargo test -p hipfire-arch-qwen35 --test pp_parity --no-run   # builds the n>1 path
```
Expected: clean. Borrow note: in `assemble_weights`, `&mut devices[i]` are sequential reborrows; `&token_embd` is an owned local (no slice borrow). Reusing `_n_devices`/`_embd` underscores silence clippy.

- [ ] **Step 8: Byte-identical guard (HFQ + PaRo)**

Run:
```bash
source scripts/gpu-lock.sh && gpu_acquire "loader-core"
cargo build --release -p hipfire-runtime --example greedy_dump --example eval_hipfire
./target/release/examples/greedy_dump --model "$HFQ_MODEL" \
  --prompt-file benchmarks/prompts/lru_cache_pep8_strict.txt --max-tokens 64 \
  > /home/bjoern/hipfire-refactor-baseline/hfq.after.txt 2>/dev/null
./target/release/examples/eval_hipfire --model "$PARO_MODEL" --greedy --max-tokens 64 \
  > /home/bjoern/hipfire-refactor-baseline/paro.after.txt 2>/dev/null
gpu_release
diff /home/bjoern/hipfire-refactor-baseline/hfq.before.txt  /home/bjoern/hipfire-refactor-baseline/hfq.after.txt
diff /home/bjoern/hipfire-refactor-baseline/paro.before.txt /home/bjoern/hipfire-refactor-baseline/paro.after.txt
```
Expected: **both empty**. Non-empty → STOP, bisect against the invariants below.

- [ ] **Step 9: Coherence gate + commit**

Run:
```bash
source scripts/gpu-lock.sh && gpu_acquire "loader-gate"
./scripts/coherence-gate.sh
gpu_release
cargo fmt -p hipfire-arch-qwen35
git add crates/hipfire-arch-qwen35/src/qwen35.rs
git commit -m "refactor(loading): unify qwen35 loader over device-slice + WeightSource"
```

---

## Task 4: Reconcile the design doc

**Files:** Modify `docs/superpowers/specs/2026-06-11-carrier-registry-unified-design.md`.

- [ ] **Step 1: Add a Tier-2 note** (under `## Out of scope` or a new subsection)

```markdown
### Tier 2: unified loader (`WeightSource` + device slice) — added 2026-06-12

The three `load_weights*` entry points were unified behind one `assemble_weights`
driver over a `&mut [Gpu]` device slice + `Layout`, with HFQ/PaRo isolated behind a
`WeightSource` trait (`HfqSource`/`ParoSource`). Single-GPU is the `len()==1` case;
multi-GPU is HFQ-only (PaRo `prepare(n>1)` errors). Tied-embedding alias is gated to
`len()==1` to stay byte-identical. The public API was collapsed to a single
`load_weights` entry (was 3). Carrier registry (Tier 1) unchanged in behavior.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-06-11-carrier-registry-unified-design.md
git commit -m "docs(spec): note Tier-2 unified loader in carrier-registry design"
```

---

## Task 5 (LAST): Collapse the public API to one entry + migrate callers

Now that one driver exists, expose **one** public entry and delete the three wrappers. This churns ~40 call sites — done last, on its own commit(s), so the byte-identical core (Task 3) is already landed and gated.

**Files:** Modify `crates/hipfire-arch-qwen35/src/qwen35.rs` + ~40 caller files (carriers, `arch.rs` shim, examples, tests).

- [ ] **Step 1: Make the driver public + handle drop_mmap inside HFQ source**

Change `HfqSource` to hold `&'a mut HfqFile` and move `drop_mmap` into `prepare`:
```rust
struct HfqSource<'a> { hfq: &'a mut HfqFile }
impl WeightSource for HfqSource<'_> {
    fn prepare(&mut self, n_devices: usize) -> HipResult<()> {
        #[cfg(unix)]
        if n_devices == 1 { self.hfq.drop_mmap(); }
        Ok(())
    }
    // read_* now borrow `self.hfq` immutably as before (auto-reborrow from &mut)
    // …unchanged bodies…
}
```
Rename `assemble_weights` → `pub fn load_weights` with signature:
```rust
pub fn load_weights(
    source: &mut dyn WeightSource,
    devices: &mut [Gpu],
    layout: &Layout,
    config: &Qwen35Config,
) -> HipResult<Qwen35Weights>
```
Make `Layout`, `WeightSource`, `HfqSource`, `ParoSource` `pub`. Add `pub use` as needed for callers. Delete the old `load_weights`/`load_weights_paroquant`/`load_weights_multi` wrappers.

- [ ] **Step 2: Migrate callers (mechanical pattern)**

The ~40 sites fall into three shapes. Apply per shape:

- Single-GPU HFQ — `qwen35::load_weights(&mut hfq, &cfg, &mut gpu)` becomes
  ```rust
  qwen35::load_weights(&mut HfqSource::new(&mut hfq), std::slice::from_mut(&mut gpu),
      &Layout::single(cfg.n_layers), &cfg)
  ```
  (ensure the `hfq` binding is `let mut`). Representative: `crates/hipfire-arch-qwen35/src/arch.rs:74`, `crates/hipfire-arch-qwen35/src/carrier.rs:25`, `crates/hipfire-runtime/examples/infer_qwen35.rs:137`, `crates/hipfire-arch-qwen35/src/speculative.rs:563`, `crates/hipfire-arch-qwen35/src/pflash.rs:561`.
- PaRo — `qwen35::load_weights_paroquant(&source, &cfg, &mut gpu)` becomes
  ```rust
  qwen35::load_weights(&mut ParoSource::new(&source)?, std::slice::from_mut(&mut gpu),
      &Layout::single(cfg.n_layers), &cfg)
  ```
  Representative: `crates/hipfire-loader/src/carriers.rs:171`, `crates/hipfire-runtime/examples/eval_hipfire.rs:187`.
- Multi-GPU HFQ — `qwen35::load_weights_multi(&hfq, &cfg, &mut gpus)` becomes
  ```rust
  let layout = Layout::from_gpus(&gpus, cfg.n_layers);
  qwen35::load_weights(&mut HfqSource::new(&mut hfq), &mut gpus.devices, &layout, &cfg)
  ```
  (make `hfq` `let mut`). Representative: `crates/hipfire-loader/src/carriers.rs:83`, `crates/hipfire-arch-qwen35/tests/pp_parity.rs:108`, `crates/hipfire-runtime/examples/pp2_vram_probe.rs:66`.

Find every site:
```bash
grep -rn 'load_weights\b\|load_weights_paroquant\|load_weights_multi' crates --include=*.rs \
  | grep -v 'fn load_weights' | grep 'qwen35'
```

- [ ] **Step 3: Build everything + clippy**

```bash
cargo build --workspace --examples --tests
cargo clippy -p hipfire-arch-qwen35
cargo build --example daemon -p hipfire-runtime
```
Expected: clean across the workspace (this is the churn-catching step).

- [ ] **Step 4: Byte-identical guard again + coherence gate**

Repeat Task 3 Step 8 (HFQ + PaRo diffs empty) and Step 9's `coherence-gate.sh`. Then:
```bash
cargo fmt -p hipfire-arch-qwen35
git add -A
git commit -m "refactor(loading): collapse qwen35 to one public load_weights entry"
```

> If the ~40-site migration is too large for one commit, split Step 2 by crate (arch-qwen35 internal sites → carriers/loader → runtime examples → tests), building between each. Each split still ends green.

---

## Behavior-preservation invariants (recheck if any guard diff appears)

1. Tied-embedding **alias** only when `can_alias == (devices.len()==1)`: n==1 aliases (`shallow_clone` + `embedding_format_dtype`), n>1 reuploads (`dequant_weight_raw`). `lm_head_aliases_embd = tied && can_alias`.
2. Separate lm_head always via `load_weight_tensor_raw` (F16-mode aware); never collapsed onto `dequant_weight_raw`.
3. `drop_mmap()` only on the single-GPU HFQ path (Task 3: single wrapper; Task 5: `HfqSource::prepare` when `n==1`). Embed/output read via pread `_vec`, so safe post-drop and identical bytes for n>1.
4. Per-layer page-dropping (`layer_data_range`+`drop_pages_range`) for HFQ in `read_layer`; PaRo has none.
5. AWQ sidecar on both HFQ output branches, after `gpu_dtype` is set; never on PaRo.
6. PaRo `prepare(n>1)` errors (HFQ-only multi-GPU); PaRo `quant_config()` precondition runs in `ParoSource::new`.
7. Cosmetic stderr deltas allowed (alias/reupload log wording, `on dev` suffix); weights unchanged.

## Out of scope

- PaRo multi-GPU (kept single-device by `prepare`). PP>1 and VL functional validation (need hiptrx) — todo #6.
- The other ~12k lines of `qwen35.rs` (MoE loading, tensor-name tables, forward pass) — untouched.

## Self-Review

- **Spec coverage:** three loaders → one `assemble_weights` + 2 `WeightSource` impls (Task 3); native multi-GPU → device-slice + `Layout` (Task 3 Step 2); transparent HFQ/PaRo → `WeightSource` (Steps 3–4); AWQ dedup → Task 2; public collapse → Task 5; doc → Task 4. ✔
- **Placeholder scan:** all code steps complete; env-resolved values are `$HFQ_MODEL`/`$PARO_MODEL` (Task 1) and exact line ranges from the current file. Caller migration uses pattern + representative paths per skill guidance. ✔
- **Type consistency:** `WeightSource` methods (`prepare`/`read_embed`/`read_final_norm`/`read_output`/`read_layer`) match both impls and the driver calls; `Layout::{single,from_gpus,device_for_layer,output_device}` consistent; `Qwen35Weights` fields match; `load_weights(source, devices, layout, config)` signature consistent between Task 5 definition and all three migration patterns. ✔
- **Adversarial-review carry-over:** Blocker 1 (mmap-after-drop) → invariant 3 + pread embed in `HfqSource`; Blocker 2 (two-helper split) → invariant 2 + `read_output` branches.
