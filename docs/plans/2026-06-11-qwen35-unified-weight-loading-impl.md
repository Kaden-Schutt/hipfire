# Unified qwen35 Weight Loading — Reconciliation & Remaining Loose Ends

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

> **⚠️ RECONCILED 2026-06-11.** The original from-scratch version of this plan was
> written against a STALE snapshot (a detached HEAD in the *master* worktree where
> none of the unification existed). On the actual feature branch
> (`feature/paro-transparent-loading`, tip `b187e792`) the `WeightBackend`
> unification is **already implemented — and went further than the plan**. This
> document is rewritten to reflect reality: most of the original 10 tasks are DONE;
> what remains is small dedup + verification. The companion design doc
> (`2026-06-11-qwen35-unified-weight-loading-design.md`) describes the *intended*
> shape; the **authoritative architecture record is**
> `2026-06-11-carrier-registry-unified-design.md` (the approach the branch actually
> took, which supersedes the WeightBackend-only design).

**Goal:** Finish the last loose ends of qwen35 weight-loading unification on top of
the already-shipped `WeightBackend` + `layer_driver` + carrier-registry work.

**Tech Stack:** Rust; `rdna-compute` (`Gpu`/`GpuTensor`) → `hipfire-runtime`
(`WeightBackend`/`HfqBackend`/`ParoBackend`/`HfqFile`/`ModelSource`) → `hipfire-loader`
(`Carrier` registry) → `hipfire-arch-qwen35` (`load_layer`, `load_moe_ffn`).

---

## Already implemented on the branch (DO NOT re-do)

Verified against the feature worktree on 2026-06-11:

| Plan item | Reality on `b187e792` |
|---|---|
| `WeightBackend` trait + `HfqBackend`/`ParoBackend` | **Done** — `crates/hipfire-runtime/src/weight_backend.rs:968+`. Stateful (`set_layer`), relative-name `proj`/`norm`/`raw_f32`, a `candidates` name-resolver fn-ptr, `norm_bias` (1.0 qwen3.5/gemma · 0.0 qwen2/llama), and a Paro **augmentor chain** (`try_augmentors`/`DEFAULT_AUGMENTORS`). More general than the design — covers qwen2/llama too. |
| Single generic layer-walk | **Done** — `crates/hipfire-arch-qwen35/src/layer_driver.rs::load_layer<B: WeightBackend>`. All three loaders funnel through it with a `load_moe` closure for the MoE fork. |
| `paro_load_moe_ffn`/`load_moe_ffn` duplication | **Done** — `paro_load_moe_ffn` moved to `crates/hipfire-arch-qwen35/src/paro_moe.rs`; HFQ `load_moe_ffn` stays in `qwen35.rs:1734`. The MoE fork is the caller-supplied `load_moe` closure (intended residual). |
| Embed/output, `+1.0` bake, page-drop (`drop_mmap`/`drop_pages_range`) | **Done** — `load_embedding` + `norm_bias` + per-layer `layer_data_range`/`drop_pages_range` in `load_weights` (`qwen35.rs:1340`, `:1440-1456`). |
| Daemon/safetensors routing, multi-GPU, VL, llama | **Done & beyond** — `hipfire-loader` `Carrier` registry: `enum ModelSource{Hfq,Dir}` + `from_path` (`loader_api.rs`), `Qwen35Carrier`/`LlamaCarrier` (`carriers.rs`). pp>1 folded into `Qwen35Carrier` (commit `48c3acc6`); side-doors removed. |

The original premise of this whole effort — *"`paro_load_moe_ffn` still referenced, check dead code/loose ends"* — was a stale-tree artifact. On this branch that function isn't even in `qwen35.rs` anymore.

---

## Remaining loose ends (the actual work)

Three small items. Each is dedup or verification — none is new architecture.

### Task 1: Dedupe `load_weights`' inline per-layer body via `load_layer_into`

**Problem:** `load_weights` (`qwen35.rs:1442-1452`) builds an `HfqBackend` + a `moe`
closure + calls `load_layer` **inline**, which is byte-identical to `load_layer_into`
(`qwen35.rs:1710-1733`) except `load_weights` wraps it in page-drop tracking. The
`HfqBackend` literal and the `moe` closure are thus written twice (`:1442`/`:1448`
vs `:1718`/`:1724`).

**Files:**
- Modify: `crates/hipfire-arch-qwen35/src/qwen35.rs`

- [ ] **Step 1: Make `load_layer_into` callable for single-GPU**

It already takes `(hfq, config, layer_idx, p, gpu)`. Confirm its body (`:1710-1733`)
is exactly the `HfqBackend{…, candidates: qwen35_tensor_name_candidates, read_proj:
load_weight_tensor, …}` + `moe` closure + `load_layer` that `load_weights` inlines.
No change needed if identical.

- [ ] **Step 2: Replace the inline block in `load_weights` with a call**

In `load_weights` (`qwen35.rs:1442-1452`), replace:

```rust
        let mut b = HfqBackend {
            hfq, gpu, norm_bias: 1.0,
            candidates: qwen35_tensor_name_candidates,
            read_proj: load_weight_tensor,
            layer: i,
        };
        let moe = |bk: &mut HfqBackend, cfg: &Qwen35Config, li: usize| {
            load_moe_ffn(bk.hfq, bk.gpu, &format!("layers.{li}"), cfg, li as u16)
        };
        layers.push(crate::layer_driver::load_layer(&mut b, config, i, moe)?);
```

with:

```rust
        layers.push(load_layer_into(hfq, config, i, &p, gpu)?);
```

Keep the surrounding `layer_page_start`/`drop_pages_range` lines unchanged. This
removes one `HfqBackend` literal and one `moe` closure.

- [ ] **Step 3: Build**

Run: `cargo build -p hipfire-arch-qwen35`
Expected: compiles. `load_layer_into` was previously only used by
`load_weights_multi`; it now has two callers. The `is_moe` local in `load_weights`
used only for the log line stays.

- [ ] **Step 4: Byte-identical loader check (HFQ)**

Run (in the feature worktree, under the GPU lock per CLAUDE.md):
`cargo run -q -p hipfire-runtime --example dump_logits_qwen35 -- <an .hfq qwen3.5 model>`
before and after, on a fixed prompt; diff the top-k logit lines. Expected: **no diff**
(this is a pure call-site dedup, output must be identical).

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-arch-qwen35/src/qwen35.rs
git commit -m "refactor(qwen35): load_weights reuses load_layer_into; drop duplicated HfqBackend+moe closure"
```

### Task 2: Coherence gate (mandatory — loader path touched)

**Files:** none (verification).

- [ ] **Step 1: Run the coherence gate**

Per CLAUDE.md, any loader/forward change must pass it. The pre-commit hook runs it
when loader files are staged; run it explicitly too:

Run: `./scripts/coherence-gate.sh`
Expected: PASS for both an HFQ model and the A3B PARO model — fluent, on-topic, no
verbatim loop. If the daemon path changed, also confirm the safetensors/PARO route
(`Qwen35Carrier` → `load_weights_paroquant`) still loads and generates coherently.

- [ ] **Step 2: Record + commit the report**

```bash
git add -A && git commit -m "test(qwen35): coherence gate green after load_weights dedup" --allow-empty
```

### Task 3: Mark the genuinely-deferred paths

Two paths remain intentionally un-unified; mark them so future sessions don't
re-investigate them as "loose ends."

**Files:**
- Modify: `crates/hipfire-arch-qwen35/src/qwen35.rs`
- Modify: `crates/hipfire-runtime/src/hfq.rs`

- [ ] **Step 1: `@todo` on multi-GPU = HFQ-only**

Above `load_weights_multi` (`qwen35.rs:1582`):

```rust
// @todo(unified-loading): multi-GPU is HFQ-only — no ParoBackend/safetensors
// multi-GPU path (drop_pages_range has no ModelSource equivalent, band-routing
// is HFQ-mmap-specific). See 2026-06-11-carrier-registry-unified-design.md.
```

- [ ] **Step 2: `@todo` on the separate llama PARO loader**

Above `load_weights_paroquant_llama` (`hfq.rs`, ~line 1050):

```rust
// @todo(unified-loading): llama PARO still has its own loader; qwen35 already
// runs through layer_driver::load_layer + WeightBackend. Migrate llama onto the
// same generic walk (the trait is arch-agnostic via norm_bias + candidates).
```

- [ ] **Step 3: Build + commit**

Run: `cargo build`
Expected: compiles (comment-only changes).

```bash
git add -A
git commit -m "docs(loading): @todo markers for deferred multi-GPU-Paro + llama PARO unification"
```

---

## Deferred items reminder (plan-level — final item)

After Task 3, `grep -rn "@todo(unified-loading)"` should list exactly the two
deferred paths below. Confirm both are present before declaring done:

1. **Multi-GPU for safetensors/Paro** — `load_weights_multi` is HFQ-only; a
   `ModelSource` page-drop/band-routing story is needed before Paro multi-GPU.
2. **Llama-family PARO** — `hfq.rs:load_weights_paroquant_llama` still bypasses
   `layer_driver::load_layer`; migrate it onto the shared `WeightBackend` walk.

Optional / not scheduled (low value, note only): the three `load_weights*` entry
points still exist separately (single-HFQ, single-Paro, multi-HFQ). They legitimately
differ in backend construction + embed/output handling and already share
`load_layer`; collapsing them into one `load_weights_generic<B>(embed_fn, output_fn,
moe_fn)` is possible but buys little — defer unless a 4th entry point appears.
