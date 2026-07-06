# Declarative Weight Loading — Schema + Thunk Registry + Validation Contract

**Status:** design / blessed concept, pre-implementation
**Date:** 2026-06-20
**Branch context:** follows `feature/transparent-loading-all-models` (PR #455 lineage)
**Related:** `docs/design/2026-06-13-greenfield-engine-architecture.md`,
`crates/hipfire-runtime/src/model_load.rs` (`WeightSource`),
`crates/hipfire-runtime/src/model_source.rs` (`ModelSource`),
`crates/hipfire-loader/src/carriers.rs` (`Carrier` registry).

---

## 0. Problem statement

We want **one** unified weight-loading path instead of per-architecture ×
per-format (HFQ vs safetensors) duplicated code. Today each arch crate
hand-codes its layer schema *twice*: once in the HFQ `load_weights` path and
once in the safetensors `load_weights_from_source` path. Example
(`hipfire-arch-qwen2/src/qwen2.rs:657`):

```rust
let p = format!("model.layers.{i}");
let wq = load_weight_tensor_from_source(source, gpu,
    &format!("{p}.self_attn.q_proj.weight"), q_dim, cfg.hidden_size)?;
let wq_bias = load_bias_f32_from_source(source, gpu,
    &format!("{p}.self_attn.q_proj.bias"), q_dim)?;
// ... 9 more tensors, then assemble Qwen2LayerWeights { wq, wq_bias, ... }
```

The tensor **names**, their **shapes-as-fn-of-config**, and which **struct
slot** they land in are all encoded as imperative Rust — duplicated across
formats and across the ~8 arch crates.

### What is already unified (do not re-solve)

- **Container** — `ModelSource` (`model_source.rs:36`) abstracts HFQ file vs
  safetensors dir down to `tensor_data(name) -> (&TensorInfo, &[u8])`,
  `metadata_json()`, `arch_id()`, `quant_config()`.
- **Orchestration** — `WeightSource` trait + generic `model_load::load_weights()`
  (`model_load.rs:78`) already drives the `embed → final_norm → output →
  per-device layer loop`.
- **Dispatch** — `Carrier` trait + `REGISTRY` (`hipfire-loader/src/lib.rs`)
  is the single arch-dispatch point: `load_model` probes `arch_id` and calls
  the matching carrier's `load()`.

### The actual residue

1. The **layer schema** is code, not data (names × shapes × slots).
2. ~10% of loading is genuine **compute** that pure data can't express:
   dequant, FWHT rotation (MQ formats), transpose, fuse-QKV (3→1 buffer),
   split-QKV (1→3), tied-lm_head aliasing, ParoQuant sidecar application.
   These are the reason a naive "just a TOML manifest" approach stalls.
3. There is **no GPU-free validation** — a mis-mapped tensor (wrong slot,
   wrong shape, missing→zero-defaulted) produces a *coherent-but-wrong*
   model that passes smoke tests and ships garbage. This is the documented
   worst-case failure mode in this repo.

---

## 1. Design in one sentence

> A **declarative `const` schema** (tensor names, shapes, slots, transform
> tags) is **validated GPU-free** against a real `ModelSource` header into a
> frozen **`LoadPlan`**, which is **executed by a small registry of named
> transform thunks** that own the *only* imperative load-time code; each arch
> destructures the resulting `RoleMap` into its existing concrete,
> static-dispatch forward struct exactly once.

Three layers, each independently shippable:

```
  ┌─────────────────────────────────────────────────────────────┐
  │ Layer A: SCHEMA (data)      const QWEN2: Schema = &[ ... ]   │
  │   block fragments → LineItems(role,name_tmpl,shape,transform)│
  └───────────────────────────────┬─────────────────────────────┘
                                   │ validate(schema, header)  ← NO GPU
                                   ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ Layer C: VALIDATION         LoadPlan { Vec<ResolvedSlot> }   │
  │   bijection · shape-algebra · disjointness → hard refuse     │
  └───────────────────────────────┬─────────────────────────────┘
                                   │ execute(plan, registry, gpu)
                                   ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ Layer B: THUNK REGISTRY     HashMap<TransformId, dyn Thunk>  │
  │   Dequant·Fwht·Transpose·FuseQKV·SplitQKV·AliasOf·Paro       │
  │   → RoleMap (HashMap<RoleKey, WeightTensor>)                 │
  └───────────────────────────────┬─────────────────────────────┘
                                   │ impl From<RoleMap> for Qwen2Weights
                                   ▼
              concrete static-dispatch forward struct (unchanged)
```

**Key invariant:** the schema touches only the **load path**. Forward stays
exactly as concrete-typed and static-dispatch as it is now (respecting the
deliberate hot-path constraint documented in `runtime/src/arch.rs:29-46`).

---

## 2. Layer A — Schema as `const` data

### 2.1 Why Rust `const`, not `.ron`/`.toml`

The `Role` enum must be a **closed compile-time set** so each arch's
`impl From<RoleMap> for XWeights` can *exhaustively* destructure. Shape
evaluation wants real typed config access, not a stringly-typed expression
parser. So the schema is `&'static` Rust data, not an external file. (A
`.ron` export is still useful as a *diffable artifact* — see §4.3 — but the
source of truth is the `const`.)

### 2.2 Core types

```rust
// crates/hipfire-loader/src/schema.rs  (new module)

/// Closed set of logical tensor roles. Exhaustively matched by each arch's
/// `From<RoleMap>`. Layer-scoped roles carry the layer index in RoleKey.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Role {
    TokenEmbd, OutputNorm, Output,          // model-level
    AttnNorm, Wq, Wk, Wv, Wo,               // attention
    WqBias, WkBias, WvBias,                 // optional biases
    FfnNorm, FfnGate, FfnUp, FfnDown,       // dense FFN
    Router, ExpertGate, ExpertUp, ExpertDown, // MoE (carry expert idx)
    // ... DeltaNet / vision roles added as fragments land
}

/// Identifies a concrete slot: a role plus optional layer / expert index.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RoleKey { pub role: Role, pub layer: Option<u32>, pub expert: Option<u32> }

/// Shape as an expression over named config dimensions. Evaluated once per
/// slot against the typed &Config. Reuses today's `q_dim = n_heads*head_dim`
/// arithmetic instead of re-deriving it.
#[derive(Clone, Copy, Debug)]
pub enum ShapeExpr {
    Hidden,
    Vocab,
    Inter,                       // intermediate_size
    QDim,                        // n_heads * head_dim
    KvDim,                       // n_kv_heads * head_dim
    Mat(Dim, Dim),               // [out, in]
    LikeSibling(Role),           // shape resolved from another slot (fixpoint)
    Custom(fn(&dyn ConfigDims) -> [usize; 2]), // escape hatch, last resort
}

#[derive(Clone, Copy, Debug)]
pub enum Presence { Required, Optional }

/// Named imperative transform applied during execution. Params inline so the
/// schema is fully self-describing and the registry stays a pure dispatch
/// table (no hidden cfg re-reads — kills the "special/general mixture" smell).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TransformId {
    Identity,                    // upload as-is
    Dequant,                     // dispatch on TensorInfo.quant_type
    Fwht { dim: usize },         // MQ rotation (wraps fused_rmsnorm_rotate_for_mq)
    Transpose,
    FuseQkv,                     // 3 inputs → 1 buffer
    SplitQkv { slot: u8 },       // 1 input → this slot of 3
    AliasOf(Role),               // view of an already-built buffer (tied lm_head)
    ApplyParoSidecar,            // wraps load_paroquant_weight_from_source
}

/// One declarative tensor line-item.
pub struct LineItem {
    pub role: Role,
    /// `{i}` = layer index, `{e}` = expert index. Format-neutral; ModelSource
    /// hides the container. Optional alias list for cross-format name drift.
    pub name_template: &'static str,
    pub aliases: &'static [&'static str],
    pub shape: ShapeExpr,
    pub presence: Presence,
    pub transform: TransformId,
}

/// A reusable group of line-items (AttnBlock, MoEBlock, DeltaNetBlock, ...).
pub struct Fragment { pub items: &'static [LineItem] }

/// A full arch schema. `layer_fragments` is indexed by a per-layer-type
/// classifier so hybrid models (DeltaNet+attn) select a different fragment
/// list per layer without code forks.
pub struct Schema {
    pub model_items: &'static [LineItem],               // embed/norm/output
    pub layer_fragments: fn(layer: u32, cfg: &dyn ConfigDims) -> &'static [Fragment],
}
```

### 2.3 How variations collapse to data

| Variation | Becomes |
|---|---|
| Fused vs split QKV | one `FuseQkv`/`SplitQkv` line-item vs three plain ones |
| Optional bias (qwen2 has, llama doesn't) | `Presence::Optional` |
| MoE 256 experts | `{e}` expansion of `ExpertGate/Up/Down` line-items |
| Hybrid DeltaNet+attn | `layer_fragments(i, cfg)` returns different fragments |
| Tied lm_head | `Output` line-item with `transform: AliasOf(TokenEmbd)` |
| ParoQuant sidecar | `transform: ApplyParoSidecar` + sidecar siblings as inputs |
| Vision tower | a `VisionFragment` appended to the model/layer schema |
| Mixed precision per layer | `Dequant` reads `quant_type` from the file header |

---

## 3. Layer B — Thunk registry (the imperative residue)

### 3.1 The trait

```rust
// crates/hipfire-loader/src/thunk.rs  (new module)

pub trait Thunk: Send + Sync {
    /// `inputs` are already-built upstream buffers (DAG in-edges). `src`
    /// gives header+bytes for leaf reads. `ctx` carries arch/layer/role for
    /// error messages and debugging.
    fn apply(
        &self,
        inputs: &[&GpuTensor],
        src: &ModelSource,
        file_name: &str,
        ctx: &BuildContext,
        gpu: &mut Gpu,
    ) -> Result<WeightTensor, String>;
}

pub struct BuildContext { pub arch: &'static str, pub layer: Option<u32>, pub role: Role }
```

### 3.2 The thunks already exist — they just need extracting

Most thunk bodies are **lifted verbatim** from code that exists inline today:

| Thunk | Lift from | Status |
|---|---|---|
| `Dequant` | `weight_backend::dequant_weight_raw` (`weight_backend.rs:441`) | ✅ real |
| `RmsBake { bias }` | `weight_backend::dequant_norm` (`weight_backend.rs:988`) | ✅ real — see ⚠️1 |
| `ApplyParoSidecar` | `hfq::load_paroquant_weight_from_source` (`hfq.rs:1006`) | ⚠️ 6-in/4-out, not 1-in — see ⚠️2 |
| `Identity` | `gpu.upload_raw` / `upload_f32` | ✅ real |
| `AliasOf` | the tied-lm_head re-alias branch in `qwen2::load_lm_head` | ⚠️ tri-state, not 2 — see ⚠️3 |
| ~~`Fwht`~~ | ~~`fused_rmsnorm_rotate_for_mq`~~ | ❌ **WRONG — no load-time referent** |
| ~~`FuseQkv`/`SplitQkv`~~ | ~~inline fuse/split in qwen35/llama~~ | ❌ **WRONG — don't exist at load** |

> **Factual corrections (from the 2026-06-20 review — see §10):**
> - ⚠️0 **`Fwht` is a design-time error.** `fused_rmsnorm_rotate_for_mq`
>   (`llama.rs:935`) rotates *activations* in the **forward** pass (all 9
>   call sites are forward code); hipfire stores weights already-rotated, so
>   there is **no load-time FWHT weight transform** to wrap. Drop `Fwht`.
> - ⚠️0 **`FuseQkv`/`SplitQkv` don't exist at load time.** Q/K/V load as three
>   separate `WeightTensor` slots; "fused QKV" is a *forward kernel*
>   (`fused_qkv_hfq4g256`) reading the three buffers. The only real load-time
>   concats are MoE 2-way (`w1‖w3`, `fc1‖fc3`) — needs a general
>   `ConcatRows { n }`, not the 3-way `FuseQkv`.
> - ⚠️1 the `(1+weight)` RMSNorm bake is a **mandatory per-arch transform on
>   every norm slot** (`norm_bias` threaded into `dequant_norm`), resolving
>   Open Question §9.1: yes, a dedicated `RmsBake` thunk.
> - ⚠️2 ParoQuant reads **6 sibling tensors** and emits a `WeightTensor` plus
>   **3 sidecar GPU buffers**; the coverage validator must claim all 6 or
>   they read as "unclaimed."
> - ⚠️3 lm_head resolution is **tri-state** (load `lm_head.weight` / alias
>   embed / re-upload tied), gated on conditional presence + an embd-dtype
>   alias-compatibility check — not the 2-state `AliasOf`.

The registry is a `HashMap<TransformId, Box<dyn Thunk>>` built once at
startup. No new algorithms — pure refactor of placement. **But the review
found the cross-tensor MoE operations (device-pointer tables, expert-blob
concat, EP sharding) fit neither a `Thunk` nor a dumb `From` — see §10.**

### 3.3 Execution as a DAG

Loading is a DAG: nodes = `RoleKey` buffers, edges = transform inputs.
`FuseQkv` has 3 in-edges; `SplitQkv` emits 3 roles from 1 source; `AliasOf`
is a view edge. **Topological order over the DAG *is* the load order** —
this deletes "loading order matters" as a concern (it becomes a graph
property, not arch knowledge). Toposort over ~10 roles/layer is microseconds
against multi-GB uploads, so the indirection is free.

Aliasing is modelled with **owned-vs-view** node tags so the exhaustive-free
teardown drops each *physical* buffer exactly once (a view must never be
freed as an owner → no double-free).

---

## 4. Layer C — GPU-free validation contract

### 4.1 The pure function

```rust
// crates/hipfire-loader/src/validate.rs  (new module)

/// Pure, no GPU. Operates on header metadata only:
///   HfqFile::tensors() -> &[HfqTensorInfo]   (name, shape, quant_type)
///   ModelSource::tensor_names() / tensor_info()
pub fn validate(schema: &Schema, cfg: &dyn ConfigDims, src: &ModelSource)
    -> Result<LoadPlan, Vec<Violation>>;

pub struct LoadPlan { pub slots: Vec<ResolvedSlot>, pub content_hash: u64 }
pub struct ResolvedSlot {
    pub key: RoleKey, pub file_name: String,
    pub shape: [usize; 2], pub dtype: DType, pub transform: TransformId,
}
pub enum Violation {
    UnfilledRequiredSlot(RoleKey, /*expanded name*/ String),
    UnclaimedTensor(String),
    ShapeMismatch { key: RoleKey, expected: [usize;2], actual: Vec<usize> },
    AmbiguousMatch { tensor: String, slots: Vec<RoleKey> },
    UnsolvableShape(RoleKey),
}
```

### 4.2 Three accumulating passes (collect all violations, don't short-circuit)

1. **Bijection / coverage.** Expand every slot across `0..n_layers`
   (×experts for MoE). Require: every `Required` slot filled by exactly one
   file tensor, and every file tensor claimed by exactly one slot. Refuse
   with **both** `unfilled_slots` and `unclaimed_tensors` fully enumerated.
   *No zero-defaulting, ever.* This replaces the
   `unwrap_or_else(|| panic!("tensor not found"))` sites and the silent
   `unwrap_or` config defaults that produce coherent-but-wrong.
2. **Shape algebra.** `hidden`, `n_heads`, `head_dim`, `kv_heads`,
   `experts`, `vocab` are typed dimension variables bound from config. Solve
   each slot's `ShapeExpr` and assert-equal against the file header shape. A
   GQA `kv_dim` vs `hidden` swap or an unflagged transpose rejects the whole
   schema **before tensor one is touched**.
3. **Disjointness.** No two expanded slot names collide; no file tensor is
   matched by two slot globs (MoE expert globs included). Proves the
   name→slot mapping is a *function*, independent of any actual file.

### 4.3 LoadPlan is the single source of truth

Validation and execution share the schema; `validate` plans once, the
uploader executes many. The uploader iterates `LoadPlan.slots` instead of
ad-hoc `tensor_data(name)`, so an upload can never reach a tensor the
contract didn't prove. Serialize the `LoadPlan` to a stable, diffable
manifest (`name → role → shape → dtype`) for golden regression tests and a
`hipfire verify-model` subcommand.

---

## 5. How it plugs into the existing `Carrier` seam

No new dispatch mechanism. Each `Carrier::load()` body
(`hipfire-loader/src/carriers.rs`) changes from "call the arch's hand-coded
`load_weights`" to:

```rust
fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
    let cfg = qwen2::config_from_source(&src)?;
    let plan = validate(&qwen2::SCHEMA, &cfg, &src).map_err(fmt_violations)?;
    let role_map = execute(&plan, &REGISTRY, &mut ctx.gpus[0])?;
    let weights = Qwen2Weights::from(role_map);            // exhaustive destructure
    // ... new_state, assemble LoadedModel (unchanged)
}
```

The arch crate keeps: `Config`, `SCHEMA` const, `impl From<RoleMap> for
XWeights`, and forward. It *loses*: both `load_weights` name-walks.

---

## 6. The two load-bearing risks (and mitigations)

1. **The `From<RoleMap>` impl re-importing per-arch knowledge.** If
   arch-specific *numerical* fixups (Qwen3.5's `(1+weight)` RMSNorm bake,
   DeltaNet-vs-FA split, MoE sidecar rotation) leak into the `From` impl or
   into per-LineItem closures, the interpreter grows a hook per idiosyncrasy
   and becomes as complex as the code it replaced — a shallow module hiding
   nothing.
   **Rule:** the schema captures the 90% mechanical *name × shape × slot*
   mapping; genuine numerical fixups go into the **thunk registry** (Layer
   B), nowhere else. The `From` impl must be a dumb 1:1 destructure. If a
   `From` impl needs an `if`, that logic belongs in a thunk.
2. **Debuggability of the DAG.** A flat `load_weights` becomes "breakpoint in
   `Dequant::apply` fires 200× with no context."
   **Mitigation:** thread `BuildContext{arch, layer, role}` into every
   `apply` and error string (`qwen35 L17 wk: Dequant(q4k) shape mismatch`),
   and add `--dump-load-dag` to print the resolved graph before upload.

**Excluded as traps:** shape-inference-from-byte-count (transpose &
fused-QKV packing unrecoverable); one superset "genome" schema (couples all
arches → change amplification); pure reflective discovery as the *authority*
(the silent-mismap vector — fine only as a validation input);
`EnumMap<Role,Tensor>` *replacing* the forward struct (fights static
dispatch — use `RoleMap` transiently at load, destructure to concrete).

---

## 7. Step-by-step implementation plan

Bottom-up; every step is independently shippable and gated. **Each step must
keep `./scripts/coherence-gate.sh` green** and prove byte-identical GPU
uploads vs the pre-change path on a real checkpoint.

### Step 0 — scaffolding (no behavior change)
- New modules in `hipfire-loader`: `schema.rs`, `thunk.rs`, `validate.rs`.
- Define `Role`, `RoleKey`, `ShapeExpr`, `TransformId`, `LineItem`,
  `Fragment`, `Schema`, `Thunk`, `BuildContext`, `LoadPlan`, `Violation`,
  `RoleMap = HashMap<RoleKey, WeightTensor>`.
- **Verify:** `cargo build` + `cargo test` (types only, no wiring yet).

### Step 1 — extract one thunk, prove the seam
- Lift tied-lm_head into a 2-variant registry (`Dequant` + `AliasOf`).
- Rewrite `qwen2::load_lm_head` to go through the registry. No other change.
- **Verify:** coherence-gate green on a tied-embedding qwen2 model; the
  alias path still aliases (no re-upload regression).

### Step 2 — schema-ize qwen2's layer (simplest real arch)
- `const QWEN2_LAYER: Fragment` (11 line-items) + `const QWEN2: Schema`.
- `interpret_layer` reproducing today's `load_layer_from_source` reads.
- `impl From<RoleMap> for Qwen2LayerWeights` / `Qwen2Weights`.
- Rewrite both `qwen2::load_weights` (HFQ) and `load_weights_from_source`
  (safetensors) to call the **one** schema path.
- **Verify:** load a real qwen2 HFQ both old and new ways; diff tensor
  metadata (shape/dtype/offset) for byte-identical uploads. Coherence-gate.

### Step 3 — bolt on validation (GPU-free)
- Implement `validate` against `HfqFile::tensors()` / `ModelSource` headers.
- Unit test: real qwen2 metadata (no `Gpu`) → `Ok(LoadPlan)` matching a
  checked-in golden manifest. Second test: delete one slot name from a
  synthetic header → bijection refusal naming exactly that slot.
- Wire `Qwen2Carrier::load` to validate→execute.
- **Verify:** `cargo test` golden + refusal tests; coherence-gate.

### Step 4 — generalize thunks for the hard cases
- Add `Fwht`, `Transpose`, `FuseQkv`, `SplitQkv`, `ApplyParoSidecar` (lifted
  verbatim from qwen35/llama inline code). Build the DAG + toposort +
  fixpoint sizing pass for `LikeSibling`.
- Port `llama` (dense, no bias, optional fused QKV) as the second arch.
- **Verify:** byte-diff + coherence-gate on a llama and a ParoQuant model.

### Step 5 — port the remaining arches one at a time
- Order by difficulty: `llama` → `minimax` → `lfm2moe` → `qwen35` (hybrid +
  MoE, hardest) → `dots-ocr` / `qwen35-vl` (vision fragments) → `deepseek4`.
- Each port deletes that arch's two name-walks; each is its own commit
  (`refactor(loader): schema-ize <arch>`), each byte-diff + coherence-gated.

### Step 6 — harvest the downstream wins (optional, high value)
- Drive `unload_model` teardown from the same `RoleMap` → "add a tensor
  without a free arm" becomes a compile error.
- Feed the DFlash lm_head quant-support gate from line-item metadata (kills
  name drift between gate and loader).
- `hipfire verify-model <path>` + load `--dry-run`; golden-manifest
  regression per supported model; schema fuzzing (rename/drop/transpose must
  always yield a `Violation`).

---

## 8. Success criteria

- [ ] Each arch crate has **one** schema-driven load path, zero per-format
      name-walk duplication.
- [ ] `validate` runs GPU-free in CI and **hard-refuses** any missing /
      extra / mis-shaped tensor (no coherent-but-wrong).
- [ ] Forward pass untouched — still concrete static-dispatch (byte-diff
      proves identical uploads; coherence-gate green every step).
- [ ] Adding a near-neighbour arch = "write a `Schema` const + `From` impl",
      no new loader Rust.
- [ ] `LoadPlan` manifest is golden-tested per supported checkpoint.

## 9. Open questions for implementation

1. Does `qwen35`'s `(1+weight)` RMSNorm bake fit cleanly as a `Fwht`-adjacent
   thunk, or does it need a dedicated `RmsBake` thunk? (Resolve at Step 5.)
2. MoE expert globbing: does disjointness validation stay tractable at 256
   experts × 40 layers, or do we special-case expert ranges? (Step 4.)
3. Should the per-layer-type classifier live in `Schema` or in `Config`?
   (Leaning `Schema.layer_fragments(i, cfg)` to keep Config a pure data bag.)
4. Provocation worth deciding early: if a port is "schema const + golden
   file, zero loader Rust", do the `hipfire-arch-*` crates eventually
   collapse into one `hipfire-arch` crate holding a directory of schemas?

---

## 10. Review findings (2026-06-20) — SCOPE CUT

A 4-agent review (architecture / codebase-reality / shippability / skeptic)
stress-tested §1–§9 against the real arch crates. The verdict was strongly
convergent: **build only the validation contract (Layer C); drop the schema
(Layer A) and the thunk-registry + DAG (Layer B).** This section records why
and supersedes the §7 plan.

### 10.1 What the review confirmed

- **Layer C (GPU-free validation) is the deep, high-value module.** A pure
  `validate(...) -> Result<LoadPlan, Vec<Violation>>` over header metadata
  (`HfqFile::tensors()` :487, `ModelSource::tensor_info()`) retires the
  documented worst case (coherent-but-wrong) in CI with no GPU. Keep it.
- The `ModelSource` header surface genuinely supports validation for both
  HFQ and safetensors (caveat: safetensors sets `quant_type = 0xFF`, real
  method in `quant_config()` — validation must consult both).
- The per-layer-type classifier (`layer_fragments(i, cfg)` reading
  `cfg.layer_types`) is sound *if* a schema is ever built.

### 10.2 Why Layers A + B were cut

1. **Two thunks are factual errors** (`Fwht`, `FuseQkv`/`SplitQkv` — see the
   §3.2 corrections). They reference forward-pass code, not load code.
2. **The cross-tensor MoE ops leak through every seam.** MoE device-pointer
   tables (`qwen35.rs:1642`, deepseek4 `arch.rs:235`, lfm2moe), expert-blob
   concat + EP sharding + a `mem::forget`-leaked dummy buffer (deepseek4),
   and conditional AWQ-sidecar mutate-in-place fit neither a `Thunk`
   (1-output) nor a "dumb `From`". The design predicted this leakage in §6
   Risk 1 and did not close it.
3. **The DAG solves a non-problem.** `model_load::load_weights` (:78) is a
   flat linear sequence with exactly one real cross-buffer edge (tied lm_head
   aliases embed). Toposort over a one-edge graph is speculative generality.
4. **The premise is weak.** The name-walk a schema replaces is ~25 lines/arch
   of stable code (~200 total — CLAUDE.md calls that cheap). Every shipped
   loader bug in git log (`68c38e0d`, `1e4d0bb6`, …) is a **missing BF16
   match arm** in dtype-conversion plumbing (~110 lines/arch, 4× bigger) —
   **zero** are the wrong-slot/wrong-shape class A+B armor against.
5. **`ShapeExpr::Custom(fn)`** would become where all real complexity hides,
   degrading the shape-algebra guarantee to "ran your closure, no
   disagreement."

### 10.3 Shippability corrections (apply regardless)

- The "byte-identical GPU upload" gate is **impossible as written**
  (`download_f32` is F32-only; quant bytes can't be read back). The real gate
  is the **metadata manifest diff** `(name, shape, dtype, offset, m, k,
  row_stride)` — GPU-free, deterministic.
- **coherence-gate is insufficient** for a loader refactor: it gates
  *fluency*, not *fidelity* — a swapped `wk`/`wv` passes it. Any loader change
  needs the manifest diff + a logits/KLD equivalence probe vs the pre-change
  binary on a fixed greedy prompt.
- Use **shadow mode** (run old + new, assert manifest equality behind a flag,
  old stays authoritative) instead of in-place rewrites that destroy the
  diff baseline.

### 10.4 Revised plan (supersedes §7)

PR #455 (transparent loading) **merged into upstream/master at `68b7f839`
(2026-06-19)**, so this is now a clean follow-up off updated master — no
longer stacked on an unlanded refactor.

#### 10.4.0 Status update — what the follow-up branch already landed (2026-06-20)

The branch `feature/transparent-loading-all-models` (rebased onto merged
master) already advanced two of the items this section names, so the "now"
worklist below is smaller than first written:

- **KV `from_mode` unification (roadmap followup #3) — DONE.** `KvCache::from_mode`
  dispatcher + 6 per-site policies with disagreeing-defaults preserved
  (`kv_mode.rs`, truth-table tests green). Behavior-preserving, not
  behavior-changing — verified arm-for-arm against the old `match` ladders.
- **dtype-plumbing fold (item 2 below) — ~60% done.** The shared helpers
  (`source_bytes_to_f32_vec`, `bf16_bytes_to_f16`, `bf16_to_f32` in
  `safetensors_source.rs`) landed (`c5fdb3ff`); **deepseek4 and minimax
  already route through them**. Only **qwen2 + dots-ocr** still carry private
  `source_bytes_to_f16_stream` copies — the item-2 list below is stale.
- **lfm2moe BF16 decode — FIXED as a prerequisite to wiring its Dir arm.**
  `effective_quant_type` collapsed `BF16→F16`, so the qt-keyed decode widened
  BF16 with `f16_to_f32` → silently-wrong values. Now threads the source
  dtype and routes fp conversions through the shared helpers.
- **Additional-arch safetensors reach (followup #1) — wired AND validated with
  real downloaded checkpoints** (`derive_arch_id` now maps `dots_ocr→8`,
  `deepseek_v4→9`, `lfm2_moe|lfm2→11`):
  - **dots-ocr Dir — WORKS, byte-identical to the q8-HFQ reference** on
    `rednote-hilab/dots.ocr`. Fixing it surfaced two real bugs the dead
    from_source path hid: (1) the vision tower loaded norms/biases as exact
    `bf16→f32` instead of the validated `f16`-narrowed recipe — in a tower so
    sensitive a ~5e-4 per-weight delta compounds ~1.36×/block over 42 blocks
    to garbage; (2) `daemon.rs` hardcoded `embedding_lookup_q8`, misreading the
    F32 Dir embedding as Q8 blocks. Both fixed; dense raw-HF safetensors now
    loads correctly.
  - **lfm2moe / deepseek4 Dir (MoE) — REFUSE raw-HF cleanly.** The indexed-MoE
    GEMV forward has no float-expert path (it needs FWHT-rotated MQ4G256/HFQ4
    experts); raw bf16 experts would be misread as 4-bit blocks → silent
    garbage. lfm2moe now hard-errors on raw-float experts pointing at
    quantize-first. deepseek4's Dir arm stays flagged unvalidated (no
    checkpoint).
  - **General finding:** raw-HF safetensors loading works for *dense* arches
    but not *MoE* arches whose forward requires pre-quantized+rotated experts;
    supporting raw-HF MoE means load-time quantization (out of scope, separate
    feature). This is the real boundary the validation exposed.

#### Remaining now-worklist

1. **`validate_coverage` pass (the keeper — NOT yet built).** After each arch's
   *existing* loader, a GPU-free assertion: every config-derived tensor was
   claimed, every shape matches config algebra, and the silent `unwrap_or`
   defaults + `unwrap_or_else(|| panic!(...))` sites (`qwen2.rs:453,558,571`,
   and per-arch equivalents — **~110 such sites total**, deepseek4 ≈95,
   qwen35 ≈55, qwen2 13) become **enumerated hard errors**. No schema, no
   thunks, no DAG, no new type zoo. ~1/5 the cost, captures the one real win.
   **This is the active next module.**
2. **Finish the dtype-conversion fold** — only **qwen2 + dots-ocr** remain
   (fold their private `source_bytes_to_f16_stream` copies, with the extra
   `n_elements` assert, onto the shared helper). deepseek4/minimax/lfm2moe
   already routed. This kills the duplication that *actually caused bugs*
   (the BF16-arm class).
3. **Schema (Layer A) becomes a future gate**, not a now-build. Revisit only
   if a future event proves the name-walk is a real sink — e.g. porting 3+
   near-identical dense arches in a row where the mechanical walk dominates
   (the §9.4 trigger). Layers B (thunk-DAG) stays cut.

#### 10.4.1 Post-review cleanup decisions (2026-06-20)

Three cleanup conclusions came out of the PR review. Resolved:

1. **Embedding-format dispatch dedup — DONE.** The per-token `match embd_format`
   ladder had grown to 5 copies (4 in `llama.rs`, 1 the new dots-ocr text path
   in `daemon.rs`). Extracted `llama::embedding_lookup_dispatch(gpu, format,
   table, output, token, dim)` (free fn beside `EmbeddingFormat`); all 5 sites
   call it. Behavior-identical (same arms, same kernels) — a newly-added
   embedding format is now wired in exactly one place.

2. **MoE float-expert guard — kept per-arch, NOT unified.** Tempting to collapse
   lfm2moe's + deepseek4's "refuse raw-float experts" guards into one shared
   `is_raw_float_dtype(&str)` predicate. Rejected: they are *not* duplicated
   logic. lfm2moe checks `effective_quant_type` (which already absorbs the dtype
   string AND a byte-heuristic fallback for unknown dtypes); deepseek4 checks the
   raw safetensors dtype string. A shared `matches!` predicate would NARROW the
   validated lfm2moe guard — dropping its unknown-dtype fallback — to save one
   line: a false unification (Ousterhout: special/general mixture). Each arch
   reads its source through that arch's own validated lens; that seam is correct.
   Follow-up only if deepseek4 Dir is ever validated: hoist `effective_quant_type`
   into `safetensors_source` as the *richer* shared classifier and have both
   consume it (widen coverage, never narrow it).

3. **qwen2 text-norm BF16 widening — kept exact, NOT narrowed through F16.**
   qwen2's `source_bytes_to_f32_vec` widens BF16 norms directly to F32; dots-ocr
   narrows BF16→F16→F32 for its vision tower. The divergence is intentional and
   correct: the dots F16 narrowing exists to byte-match the HFQ reference for a
   hyper-sensitive tower (a fidelity-to-reference constraint), not because F16 is
   more correct. For a raw Dir load there is no HFQ container forcing F16 storage,
   so exact BF16→F32 is *strictly more faithful* to the checkpoint and the text
   decoder tolerates the difference. Narrowing qwen2 to match dots would REDUCE
   precision on an unvalidated path with no parity test to justify it. Revisit
   only if exact Dir↔HFQ byte-parity becomes a hard requirement.
