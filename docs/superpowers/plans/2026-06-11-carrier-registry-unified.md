# Unified Carrier Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every arch dispatch in `hipfire-loader::load_model` (the `arch_id` ladders + the `pp>1`/`is_dir` side-doors) with a single `&[&dyn Carrier]` registry where each carrier owns all load paths for its arch (HFQ, safetensors-dir, pp>1, VL).

**Architecture:** Make `Carrier` object-safe (`load → Result<LoadedModel, String>`, no associated type), move all impls top-of-DAG into `hipfire-loader/src/carriers.rs` (arch crates can't return `LoadedModel` — cycle), and dispatch via `REGISTRY.iter().find(|c| c.probe(&src))`. `probe` is source-aware because HFQ and Dir use different arch-id namespaces. A `LoadedModel::skeleton` constructor kills the repeated 40-field struct literal.

**Tech Stack:** Rust workspace (cargo), AMD ROCm/HIP. No daemon-path unit harness — verification is `cargo build --example daemon -p hipfire-runtime` + `./scripts/coherence-gate.sh` + `./scripts/coherence-gate-dflash.sh` + functional smoke.

**Spec:** `docs/superpowers/specs/2026-06-11-carrier-registry-unified-design.md` (read it; this plan implements its §1–4 + migration order).

**Worktree:** `/home/bjoern/hipfire/.worktrees/feature-paro-transparent-loading` (branch `feature/paro-transparent-loading`).

---

## Conventions for this plan

- **"Move verbatim X→Y"** means cut the existing block at the cited `lib.rs` line range and paste it unchanged into the new location, fixing only `use`/visibility. Line numbers are from HEAD `ffbe3bcb`; re-grep if they drift.
- After **every** task: `cargo build --example daemon -p hipfire-runtime` must finish with warnings-only. A non-exhaustive match or dropped field is a compile error — that is the safety net.
- Coherence gates write reports to `/home/bjoern/hipfire/.scratch/` (absolute path; the worktree-relative `.scratch` does not exist). Gates manage their own GPU lock.
- Commit after each task. The pre-commit coherence hook is NOT active in this worktree (`core.hooksPath = .git/hooks`), so gates are run manually per the steps.

## File structure

**New:**
- (none — `carriers.rs` and `loader_api.rs` already exist)

**Modified:**
- `crates/hipfire-loader/src/lib.rs` — `LoadedModel::skeleton`, registry, `load_model` rewrite, fold-in of `load_model_pp`/`load_model_safetensors`/VL, dead-code removal.
- `crates/hipfire-loader/src/carriers.rs` — the 7 carrier structs + `Carrier` impls (currently just re-exports).
- `crates/hipfire-runtime/src/loader_api.rs` — drop the associated-`Bundle` `Carrier` trait; extend `ModelSource` (`Dir` carries `SafetensorsSource`, `from_path`, `arch_id()` for Dir, `describe()`).
- `crates/hipfire-arch-{qwen2,qwen35,llama}/src/carrier.rs` — drop `Carrier` impl + carrier struct; keep `Bundle` + rename `load` → free `pub fn load_bundle`.
- `crates/hipfire-arch-{qwen2,qwen35,llama}/src/lib.rs` — fix re-exports.

---

## Task 1: `LoadedModel::skeleton` + `skeleton_pp` (no behavior change)

**Files:**
- Modify: `crates/hipfire-loader/src/lib.rs` (add `impl LoadedModel`, refactor the 8 construction sites)

- [ ] **Step 1: Add the two constructors**

Add near the `LoadedModel` definition (after line ~249) in `crates/hipfire-loader/src/lib.rs`:

```rust
impl LoadedModel {
    /// Shared-field skeleton: arch state None, pp = 1, all non-core arch slots
    /// None, collections empty, mtp defaults, asst cache from env. Callers set
    /// only the fields they own via struct-update (`..LoadedModel::skeleton(..)`).
    pub fn skeleton(
        arch_id: u32,
        tokenizer: hipfire_runtime::tokenizer::Tokenizer,
        max_seq: usize,
        physical_cap: usize,
        model_path: String,
        chat_template: Option<String>,
    ) -> Self {
        LoadedModel {
            arch_id, pp: 1, ep: None,
            pp_gpus: None, pp_scratch_set: None, pp_dn_la_to_device: None,
            state: None, kv_cache: None, dn_state: None, qwen2_state: None,
            deepseek4_config: None, deepseek4_weights: None, deepseek4_state: None,
            deepseek4_pbs: None, deepseek4_eos_tok: 0,
            lfm2moe_config: None, lfm2moe_weights: None, lfm2moe_state: None, lfm2moe_eos_tok: 0,
            minimax_config: None, minimax_weights: None, minimax_state: None, minimax_eos_tok: 0,
            mtp_mode: "auto".to_string(), mtp_k: 3, mtp_weights_present: false,
            dots_ocr_config: None, dots_ocr_weights: None,
            vision_config: None, vision_weights: None,
            tokenizer: Some(tokenizer),
            seq_pos: 0, max_seq, physical_cap,
            eviction: None, kv_adaptive: None,
            conversation_tokens: Vec::new(),
            asst_turn_cache: AsstTurnCache::new_from_env(),
            prefill_checkpoints: Vec::new(), dflash_checkpoints: Vec::new(),
            decoded_vocab: None,
            model_path,
            dflash: None,
            chat_template,
        }
    }

    /// pp>1 skeleton — sets all four load-bearing multi-GPU fields together so
    /// they cannot be set piecemeal (a dropped `pp_scratch_set` is a silent
    /// VRAM leak; `pp_gpus`/`pp_dn_la_to_device` are `.expect()`ed in unload).
    pub fn skeleton_pp(
        arch_id: u32,
        tokenizer: hipfire_runtime::tokenizer::Tokenizer,
        max_seq: usize,
        physical_cap: usize,
        model_path: String,
        chat_template: Option<String>,
        pp: usize,
        pp_gpus: Gpus,
        pp_scratch_set: Qwen35ScratchSet,
        pp_dn_la_to_device: Vec<u8>,
    ) -> Self {
        LoadedModel {
            pp,
            pp_gpus: Some(pp_gpus),
            pp_scratch_set: Some(pp_scratch_set),
            pp_dn_la_to_device: Some(pp_dn_la_to_device),
            ..LoadedModel::skeleton(arch_id, tokenizer, max_seq, physical_cap, model_path, chat_template)
        }
    }
}
```

- [ ] **Step 2: Refactor the 8 construction sites to struct-update form**

Replace each full `LoadedModel { … }` literal with the skeleton form, keeping only the owned fields. The sites (re-grep `Ok(LoadedModel {` and `return Ok(LoadedModel {` and `= LoadedModel {`):
- qwen2 core (~688): `state: Some(ModelState::Qwen2(..)), ..skeleton(arch_id, tokenizer, max_seq, physical_cap, path.into(), chat_template)`
- llama core (~719): `state: Some(ModelState::Llama(..)), ..skeleton(...)`
- `finish_qwen35_load` (~502): `state, eviction, dflash, vision_config, vision_weights, ..skeleton(arch_id, tokenizer, ctx.max_seq, physical_cap, ctx.path.into(), chat_template)`
- dots_ocr (~758): `state: None, qwen2_state: Some(state), dots_ocr_config: Some(config), dots_ocr_weights: Some(weights), ..skeleton(...)`
- deepseek4 (~780): `deepseek4_* : Some(..), deepseek4_eos_tok, ..skeleton(...)`
- lfm2moe (~802): `lfm2moe_*`, `lfm2moe_eos_tok`, `..skeleton(...)`
- minimax (~825): `minimax_*`, `minimax_eos_tok`, `..skeleton(...)`
- safetensors llama (~882): `state: Some(ModelState::Llama(..)), ..skeleton(...)`
- safetensors qwen35 (~929): `state, ..skeleton(...)`
- `load_model_pp` (~1010): replace with `..skeleton_pp(arch_id, tokenizer, effective_max_seq, effective_max_seq, path.into(), chat_template, pp, gpus, scratch_set, la_to_device)` plus `state`.

Note: where a site set `physical_cap: max_seq`, pass `max_seq` for both args.

- [ ] **Step 3: Build**

Run: `cargo build --example daemon -p hipfire-runtime`
Expected: warnings-only. (A field-count mismatch is a compile error here.)

- [ ] **Step 4: Coherence gate (no behavior change expected)**

Run: `HIPFIRE_COHERENCE_OUT=/home/bjoern/hipfire/.scratch/cg-t1.md ./scripts/coherence-gate.sh`
Expected: "no hard errors". Eyeball the report — every model still fluent.

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-loader/src/lib.rs
git commit -m "refactor(loader): LoadedModel::skeleton/skeleton_pp — kill the 40-field literal"
```

---

## Task 2: Object-safe `Carrier` trait + core registry (HFQ only)

This is the core of the change. After it, the `if 7 / 5|6 / else` ladder is gone; `pp>1` and `is_dir` stay as pre-branches **for now** (Tasks 5–6 remove them).

**Files:**
- Modify: `crates/hipfire-runtime/src/loader_api.rs` (delete old trait)
- Modify: `crates/hipfire-arch-{qwen2,qwen35,llama}/src/carrier.rs` + `lib.rs` (drop impl/struct, expose `load_bundle`)
- Modify: `crates/hipfire-loader/src/carriers.rs` (define carriers)
- Modify: `crates/hipfire-loader/src/lib.rs` (trait, registry, dispatch)

- [ ] **Step 1: Delete the associated-`Bundle` trait from runtime**

In `crates/hipfire-runtime/src/loader_api.rs`, delete the entire `pub trait Carrier { type Bundle; … }` block (lines ~42-50). Keep `ModelSource`, `LoadCtx`, `CaskConfig`.

- [ ] **Step 2: Convert each arch carrier impl to a free `load_bundle` fn**

In `crates/hipfire-arch-qwen2/src/carrier.rs`: delete `pub struct Qwen2Carrier;` and `impl Carrier for Qwen2Carrier { … }`; delete the `use hipfire_runtime::loader_api::{Carrier, …}` (keep `ModelSource, LoadCtx`). Replace with:

```rust
use hipfire_runtime::loader_api::{ModelSource, LoadCtx};

/// Build the Qwen2 GPU bundle from an HFQ source. Refusals owned here.
pub fn load_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<Qwen2Bundle, String> {
    // (body identical to the old Qwen2Carrier::load — the ModelSource destructure,
    //  draft/cask refusals, config_from_hfq, load_weights, Qwen2State::new)
```

Keep the existing body verbatim (it already takes `src, ctx` and returns `Qwen2Bundle`). In `crates/hipfire-arch-qwen2/src/lib.rs` change `pub use carrier::{Qwen2Bundle, Qwen2Carrier};` → `pub use carrier::{Qwen2Bundle, load_bundle as load_qwen2_bundle};`.

Repeat for qwen35 (`crates/hipfire-arch-qwen35/src/carrier.rs`: the big body at lines 25-227 becomes `pub fn load_bundle(...) -> Result<Qwen35Bundle, String>`; lib.rs export `load_bundle as load_qwen35_bundle`) and llama (`crates/hipfire-arch-llama/src/carrier.rs` → `load_bundle as load_llama_bundle`).

- [ ] **Step 3: Define the object-safe trait in the loader**

In `crates/hipfire-loader/src/lib.rs`, near the registry section, add:

```rust
use hipfire_runtime::loader_api::{ModelSource, LoadCtx};

/// One arch's complete load contract. Object-safe → usable as `&dyn Carrier`.
pub trait Carrier {
    fn name(&self) -> &'static str;
    fn probe(&self, src: &ModelSource) -> bool;
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String>;
}
```

- [ ] **Step 4: Define the 7 carriers in `carriers.rs`**

Replace the re-export body of `crates/hipfire-loader/src/carriers.rs` with carrier structs. Each `load` resolves tokenizer/chat_template from the source, builds via the arch `load_bundle` (or the existing non-core `load_*`), and wraps with `skeleton`. Core carriers (HFQ-only arms in this task; Dir arms added in Task 5):

```rust
use hipfire_runtime::loader_api::{ModelSource, LoadCtx};
use crate::{Carrier, LoadedModel, ModelState, /* load_* fns, finish_qwen35_load */};

pub struct Qwen2Carrier;
impl Carrier for Qwen2Carrier {
    fn name(&self) -> &'static str { "qwen2" }
    fn probe(&self, src: &ModelSource) -> bool {
        matches!(src, ModelSource::Hfq(h) if h.arch_id == 7)
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.pp > 1 { return Err("qwen2: pipeline-parallel (pp>1) unsupported".into()); }
        let ModelSource::Hfq(hfq) = &src else { return Err("qwen2: directory source unsupported".into()); };
        let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
            .map_err(|e| format!("tokenizer not found: {e}"))?;
        let chat_template = crate::resolve_chat_template(hfq, ctx.path);
        let arch_id = hfq.arch_id;
        let bundle = hipfire_arch_qwen2::load_qwen2_bundle(src, ctx)?;
        Ok(LoadedModel {
            state: Some(ModelState::Qwen2(bundle)),
            ..LoadedModel::skeleton(arch_id, tokenizer, ctx.max_seq, ctx.max_seq, ctx.path.to_string(), chat_template)
        })
    }
}
```

`Qwen35Carrier::probe` = `matches!(src, ModelSource::Hfq(h) if h.arch_id == 5 || h.arch_id == 6)`; its `load` (HFQ arm) calls `hipfire_arch_qwen35::load_qwen35_bundle(src, ctx)?` then the existing `finish_qwen35_load` wrapping (eviction/dflash/vision) — move `finish_qwen35_load`'s body into this carrier or call it. `LlamaCarrier::probe` = `matches!(src, ModelSource::Hfq(h) if h.arch_id < 5)`; `load` mirrors qwen2 with `ModelState::Llama`. Add `if ctx.pp > 1 { return Err("<arch>: pp>1 unsupported") }` to qwen2 and llama (qwen35 keeps pp via Task 5/Task 6 — for now its HFQ pp>1 still routes through the `load_model` pre-branch, so guard qwen35 with the same Err in this task too and remove it in Task 6).

Non-core carriers (define all four):

```rust
pub struct Deepseek4Carrier;
impl Carrier for Deepseek4Carrier {
    fn name(&self) -> &'static str { "deepseek4" }
    fn probe(&self, src: &ModelSource) -> bool { matches!(src, ModelSource::Hfq(h) if h.arch_id == 9) }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.pp > 1 { return Err("deepseek4: pp>1 unsupported via registry".into()); }
        let ModelSource::Hfq(hfq) = src else { return Err("deepseek4: directory source unsupported".into()); };
        let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
            .map_err(|e| format!("tokenizer not found: {e}"))?;
        crate::load_deepseek4(hfq, tokenizer, ctx.gpu, ctx.max_seq, ctx.path)
    }
}
```

Repeat the same shape for `DotsOcrCarrier` (arch 8 → `load_dots_ocr`), `MinimaxCarrier` (arch 10 → `load_minimax`), `Lfm2MoeCarrier` (arch 11 → `load_lfm2moe`). The existing `load_*` fns already parse their own tokenizer — if so, drop the tokenizer line here and pass through (check each signature; `load_dots_ocr(hfq, tokenizer, …)` takes it, so parse it here).

- [ ] **Step 5: Build the registry + rewrite the HFQ dispatch in `load_model`**

In `crates/hipfire-loader/src/lib.rs`, add after the carrier `use`:

```rust
use crate::carriers::*;
const REGISTRY: &[&dyn Carrier] = &[
    &Qwen2Carrier, &Qwen35Carrier, &LlamaCarrier,
    &DotsOcrCarrier, &Deepseek4Carrier, &MinimaxCarrier, &Lfm2MoeCarrier,
];
```

In `load_model`, **keep** the `pp > 1` (line ~544) and `is_dir` (line ~558) pre-branches and the DFlash lm_head/MQ gate (lines ~567-633) for now. Delete the non-core `match hfq.arch_id { 8|9|10|11 }` ladder (lines ~648-654) AND the `if arch_id == 7 / else if 5|6 / else` block (lines ~685-744). Replace both with:

```rust
let src = ModelSource::Hfq(hfq);
let carrier = REGISTRY.iter().find(|c| c.probe(&src))
    .ok_or_else(|| format!("no carrier for arch_id={:?}", src.arch_id()))?;
carrier.load(src, &mut ctx)
```

(`ctx` is the existing `LoadCtx` built at ~679. **VL detection moves into `Qwen35Carrier::load` now**, not later — the trait signature `load(&self, src, ctx)` has no channel to inject externally-detected vision, and the carrier already owns `hfq` + `ctx.gpu`. Cut the VL detection block `lib.rs:659-676` into the qwen35 HFQ arm and set `vision_config`/`vision_weights` via struct-update. Delete it from `load_model`. This keeps VL models working from this task onward; Task 6 then only folds pp>1.)

- [ ] **Step 6: Build**

Run: `cargo build --example daemon -p hipfire-runtime`
Expected: warnings-only. Fix visibility (`pub(crate)` on `load_*`, `finish_qwen35_load`, `resolve_chat_template`).

- [ ] **Step 7: Coherence gates (dispatch changed)**

```bash
HIPFIRE_COHERENCE_OUT=/home/bjoern/hipfire/.scratch/cg-t2.md ./scripts/coherence-gate.sh
HIPFIRE_COHERENCE_OUT=/home/bjoern/hipfire/.scratch/cgd-t2.md ./scripts/coherence-gate-dflash.sh
```
Expected: both "no hard errors". Eyeball both reports (qwen35 fluent; dflash no tier breaches).

- [ ] **Step 8: Commit**

```bash
git add crates/hipfire-loader/src crates/hipfire-runtime/src/loader_api.rs crates/hipfire-arch-*/src
git commit -m "feat(loader): object-safe Carrier + &[&dyn Carrier] registry; HFQ dispatch unified"
```

---

## Task 3: Verify non-core + behavior-change smoke

**Files:** none (validation task).

- [ ] **Step 1: Confirm unknown-arch behavior change**

The llama HFQ probe is now `< 5`, so an unknown HFQ `arch_id` (≥12) returns `no carrier for arch_id=Some(12)` instead of entering the llama loader. There is no arch≥12 model to test; confirm by reading the dispatch — the `find().ok_or(...)` path is the only fallthrough. No code change.

- [ ] **Step 2: Functional smoke per available arch**

Qwen3.5 (arch 5/6) is covered by the coherence gate in Task 2. Note in the PR that **qwen2 (arch 7) and llama (arch <5) have compile-coverage only** (no local models — see spec "Verification"). If a qwen2/llama HFQ model is sourced, run:
`./target/release/examples/daemon` load + a short generate, confirm fluent + that a draft request errors via the pre-dispatch gate.

- [ ] **Step 3: No commit** (validation only).

---

## Task 4: `ModelSource::from_path` + `Dir` carries `SafetensorsSource`

Prereq for folding safetensors. No dispatch change yet — just enrich `ModelSource`.

**Files:**
- Modify: `crates/hipfire-runtime/src/loader_api.rs`

- [ ] **Step 1: Change `Dir` to carry the source + add `arch_id()`/`from_path`/`describe()`**

```rust
use crate::safetensors_source::SafetensorsSource;

pub enum ModelSource {
    Hfq(HfqFile),
    Dir(SafetensorsSource),
}

impl ModelSource {
    pub fn from_path(path: &str) -> Result<Self, String> {
        if std::path::Path::new(path).is_dir() {
            Ok(ModelSource::Dir(SafetensorsSource::open(path).map_err(|e| format!("{e:?}"))?))
        } else {
            Ok(ModelSource::Hfq(HfqFile::open(std::path::Path::new(path)).map_err(|e| format!("{e}"))?))
        }
    }
    pub fn arch_id(&self) -> Option<u32> {
        match self {
            ModelSource::Hfq(h) => Some(h.arch_id),
            ModelSource::Dir(s) => Some(s.arch_id),   // derive_arch_id result, already on the source
        }
    }
    pub fn describe(&self) -> String {
        match self {
            ModelSource::Hfq(h) => format!("HFQ arch_id={}", h.arch_id),
            ModelSource::Dir(s) => format!("safetensors-dir arch_id={}", s.arch_id),
        }
    }
}
```

Confirm `SafetensorsSource` has a public `arch_id` field (set from `derive_arch_id` at `safetensors_source.rs:40`) and `open(path)`; if `arch_id` is private, add a `pub fn arch_id(&self) -> u32` accessor.

- [ ] **Step 2: Build the whole workspace** (the enum shape changed; callers of `ModelSource::Dir(PathBuf)` break)

Run: `cargo build --example daemon -p hipfire-runtime 2>&1 | rg -n "error\[" || echo OK`
Fix any `ModelSource::Dir(...)` constructions to the new shape. (Today the only `Dir` construction is nominal — grep `ModelSource::Dir`.)

- [ ] **Step 3: Commit**

```bash
git add crates/hipfire-runtime/src/loader_api.rs
git commit -m "feat(loader): ModelSource::from_path + Dir carries SafetensorsSource (arch_id for dirs)"
```

---

## Task 5: Fold safetensors into qwen35/llama carriers (HIGHEST RISK)

**Files:**
- Modify: `crates/hipfire-loader/src/carriers.rs` (add Dir arms + Dir probe)
- Modify: `crates/hipfire-loader/src/lib.rs` (delete `is_dir` pre-branch + `load_model_safetensors` entry; switch to `ModelSource::from_path`)

- [ ] **Step 1: Add Dir matching to the two probes**

```rust
// Qwen35Carrier::probe
matches!(src, ModelSource::Hfq(h) if matches!(h.arch_id, 5|6))
    || matches!(src, ModelSource::Dir(s) if matches!(s.arch_id, 5|6))
// LlamaCarrier::probe
matches!(src, ModelSource::Hfq(h) if h.arch_id < 5)
    || matches!(src, ModelSource::Dir(s) if matches!(s.arch_id, 0|1))
```

- [ ] **Step 2: Add the `Dir` arm to each carrier's `load`**

In `Qwen35Carrier::load`, branch on the source. The `Dir` arm holds the body currently in `load_model_safetensors` for `arch_id == 5 || 6` (the block from `lib.rs:885` onward) — moved verbatim, sourcing tokenizer from `tokenizer.json` and template from `source.chat_template()` (as that fn does at `lib.rs:847,855`), wrapped via `skeleton`. Reject `Dir × pp>1`:

```rust
fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
    match src {
        ModelSource::Hfq(_) => { /* existing HFQ arm from Task 2 */ }
        ModelSource::Dir(source) => {
            if ctx.pp > 1 { return Err("qwen35: safetensors + pp>1 unsupported".into()); }
            // verbatim from load_model_safetensors qwen35 branch (lib.rs:885..end),
            // returning via ..skeleton(arch_id, tokenizer, ctx.max_seq, ctx.max_seq, ctx.path.into(), chat_template)
        }
    }
}
```

`LlamaCarrier::load` Dir arm = `load_model_safetensors` `arch_id == 0 || 1` block (`lib.rs:857-883`), same pattern.

- [ ] **Step 3: Rewrite `load_model`'s entry to `from_path`; delete the `is_dir` pre-branch**

In `load_model`, replace `let mut hfq = HfqFile::open(...)` + the `if is_dir` short-circuit (lines ~558-562) with:

```rust
let src = ModelSource::from_path(path)?;
```

The DFlash gate (now operating on `src`) must be wrapped `if let ModelSource::Hfq(hfq) = &src { /* lines 567-633, reading hfq */ }` — a Dir source skips it (preserves today's behavior, where the gate sat after the is_dir short-circuit). Then dispatch via the registry on `src`. Delete `load_model_safetensors` (now dead) — or keep its helpers if the carriers call into them.

- [ ] **Step 4: Build**

Run: `cargo build --example daemon -p hipfire-runtime`
Expected: warnings-only.

- [ ] **Step 5: Safetensors smoke — VERIFY IT ROUTES TO QWEN35, NOT LLAMA**

```bash
./target/release/examples/daemon &  # or via coherence harness
# load ~/.hipfire/models/shisa-Qwen3.6-35B-A3B-PARO-packed, generate a short completion
```
Expected: loads as qwen35 (MoE/A3B), fluent output. If it loads as llama or errors with an arch mismatch, the namespace bridge (probe Dir 5|6) is wrong — fix before continuing. Capture the daemon log line showing the arch.

- [ ] **Step 6: Coherence gate**

Run: `HIPFIRE_COHERENCE_OUT=/home/bjoern/hipfire/.scratch/cg-t5.md ./scripts/coherence-gate.sh`
Expected: "no hard errors".

- [ ] **Step 7: Commit**

```bash
git add crates/hipfire-loader/src
git commit -m "feat(loader): fold safetensors-dir into qwen35/llama carriers via ModelSource::Dir"
```

---

## Task 6: Fold pp>1 into `Qwen35Carrier`; delete pp pre-branch

(VL was already folded into `Qwen35Carrier` in Task 2 — nothing VL-related remains in `load_model`.)

**Files:**
- Modify: `crates/hipfire-loader/src/carriers.rs`, `crates/hipfire-loader/src/lib.rs`

- [ ] **Step 1: Add the pp>1 arm to the qwen35 HFQ branch; use `skeleton_pp`**

Move `load_model_pp`'s body (`lib.rs:934-1010`) into `Qwen35Carrier::load` under `if ctx.pp > 1` (HFQ only — reject `Dir × pp>1` already done in Task 5). Reproduce the deliberate drop: do NOT read `ctx.draft_path`/`ctx.cask`/`ctx.kv_adaptive_override` in this arm. Construct the result with `..LoadedModel::skeleton_pp(arch_id, tokenizer, effective_max_seq, effective_max_seq, ctx.path.into(), chat_template, ctx.pp, gpus, scratch_set, la_to_device)` plus `state`. Remove the per-arch `if ctx.pp>1 { Err }` guard from `Qwen35Carrier` added in Task 2 (qwen35 now supports it).

- [ ] **Step 2: Delete the `pp > 1` pre-branch in `load_model`; add the fail-fast assert**

Remove lines ~544-547 (`if pp > 1 { … return load_model_pp(...) }`). `pp` now flows via `ctx.pp` into the carrier. After dispatch, before returning, add:

```rust
debug_assert!(!(result.pp > 1) || result.pp_gpus.is_some(),
    "pp>1 LoadedModel missing pp_gpus");
```

Delete `load_model_pp` (now dead). Confirm non-qwen35 carriers still `Err` on `ctx.pp > 1` (Task 2 guards).

- [ ] **Step 3: Build**

Run: `cargo build --example daemon -p hipfire-runtime`
Expected: warnings-only.

- [ ] **Step 4: Coherence gate (single-GPU) + note pp validation gap**

Run: `HIPFIRE_COHERENCE_OUT=/home/bjoern/hipfire/.scratch/cg-t6.md ./scripts/coherence-gate.sh`
Expected: "no hard errors" (single-GPU qwen35 path unaffected).
⚠️ **pp>1 functional validation requires hiptrx (4×R9700)** — mark required-before-merge. On hiptrx: `HIP_VISIBLE_DEVICES=0,1,2,3` load a qwen35 model with `pp=4`, confirm fluent generate + clean unload (no VRAM leak: `rocm-smi` before/after).
⚠️ **VL** has no local model — compile-coverage only unless one is sourced.

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-loader/src
git commit -m "feat(loader): fold pp>1 into Qwen35Carrier; remove pp/safetensors side-doors"
```

---

## Task 7: Dead-code sweep + final gate

**Files:**
- Modify: `crates/hipfire-loader/src/lib.rs`, `crates/hipfire-runtime/examples/daemon.rs`

- [ ] **Step 1: Delete orphaned functions**

Remove from `crates/hipfire-loader/src/lib.rs` (duplicated into the qwen35 carrier in Step B / superseded): `parse_kv_adaptive`, `screen_weights_qwen35`, `warn_tiny_model_state`, plus `hfq_parameter_count` if now unused. Re-grep to confirm zero call sites before deleting each.

- [ ] **Step 2: Remove unused imports in daemon**

In `crates/hipfire-runtime/examples/daemon.rs` delete `use hip_bridge::HipResult;` (line ~25) and `use hipfire_arch_llama::Llama;` (line ~30) if the build reports them unused.

- [ ] **Step 3: Build — expect zero `never used` warnings from the loader**

Run: `cargo build --example daemon -p hipfire-runtime 2>&1 | rg "never used|unused import" || echo CLEAN`
Expected: CLEAN (or only pre-existing daemon warnings unrelated to this work).

- [ ] **Step 4: Full gate battery + perf sanity**

```bash
HIPFIRE_COHERENCE_OUT=/home/bjoern/hipfire/.scratch/cg-final.md ./scripts/coherence-gate.sh
HIPFIRE_COHERENCE_OUT=/home/bjoern/hipfire/.scratch/cgd-final.md ./scripts/coherence-gate-dflash.sh
scripts/probe_commits.sh HEAD~1 HEAD   # expect Δ≈0% — dispatch is once-per-call
```
Expected: gates "no hard errors"; perf Δ within ±5% (investigate per CLAUDE.md if not).

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-loader/src/lib.rs crates/hipfire-runtime/examples/daemon.rs
git commit -m "cleanup(loader): remove orphaned fns + unused daemon imports post-registry"
```

---

## Required-before-merge checklist (from spec)

- [ ] `cargo build --example daemon -p hipfire-runtime` clean.
- [ ] `coherence-gate.sh` green + eyeball (qwen35 5/6).
- [ ] `coherence-gate-dflash.sh` green + three-tier eyeball.
- [ ] Safetensors smoke routes to qwen35 (Task 5 Step 5) — captured log.
- [ ] Perf Δ≈0% fresh-process (Task 7 Step 4).
- [ ] **pp>1 functional smoke on hiptrx** (4×R9700) — code lands here, validation is remote, not skippable.
- [ ] qwen2 (arch 7) + llama (arch <5) functional smoke — needs sourced models (compile-coverage only otherwise; note in PR).
- [ ] VL smoke — needs a model (compile-coverage only otherwise; note in PR).
