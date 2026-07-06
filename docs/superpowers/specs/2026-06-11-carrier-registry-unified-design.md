# Carrier Registry — Unified Loader Design

**Status:** design (supersedes the Step-A/B "Deferred (separate plans)" section of
`2026-06-11-carrier-registry-loader.md`).
**Date:** 2026-06-11
**Branch:** `feature/paro-transparent-loading`
**Prereq:** Step A + B already landed (`6b566654`, `b50f5ff4`).

## Problem

Step A/B introduced `ModelSource`, `LoadCtx`, per-arch `Bundle`s, and `enum ModelState`,
but the registry itself was never built. Dispatch in `hipfire-loader::load_model` is still:

- a `match hfq.arch_id { 8|9|10|11 => … }` ladder (non-core arches), then
- an `if arch_id == 7 / else if 5|6 / else` ladder (core arches), plus
- two side-doors: `if pp > 1 → load_model_pp`, `if is_dir() → load_model_safetensors`.

`probe()` / `name()` on the `Carrier` trait are dead code (zero call sites). The associated
`type Bundle` added in Step B makes `Carrier` non-object-safe, so a homogeneous
`&[&dyn Carrier]` registry is impossible — the two steps silently contradict each other.

Net: "adding an arch = one registry line" is false. You still edit `load_model`, in up to
four places. The side-doors mean format and parallelism are *not* unified under the
abstraction `ModelSource::{Hfq, Dir}` was created to provide.

## Goal

One dispatch mechanism for **every** load path. `load_model` becomes:

```rust
let src = ModelSource::from_path(path)?;   // NEW constructor (see §3) — resolves arch up front
let carrier = REGISTRY.iter().find(|c| c.probe(&src))
    .ok_or_else(|| format!("no carrier for {}", src.describe()))?;
carrier.load(src, &mut ctx)
```

No arch ladders, no `pp`/`is_dir` pre-branches. Adding an arch (or adding pp/safetensors
support to an existing arch) is editing exactly one carrier + one `REGISTRY` line.

**`probe` is source-aware, not `arch_id()`-only.** HFQ and safetensors-dir use *different*
arch-id integer namespaces (verified — see §3, the B1 finding). `probe(&self, src: &ModelSource)`
already receives the whole source, so each carrier matches per-format and bridges the two
namespaces itself. It must NOT be reduced to a single `src.arch_id()`.

### Format coverage is per-role, not global (goal-2 precision)

This refactor unifies *dispatch*, and the loader now accepts every format the registry
knows. It does **not** widen *kernel* coverage. A model is runnable iff **every tensor's
format lands in the kernel family for the role that tensor plays** — and the three families
have very different coverage, so "all formats unlocked" is true only at the loader layer.
Runnable = `(loader accepts) ∩ (a kernel exists for this tensor's role)`:

| Role | Kernel family | Supported quant types | Source of truth |
|------|---------------|-----------------------|-----------------|
| `token_embd`, **tied** `lm_head` | embedding lookup | F32 (2), Q8_0 (3), Q4K (4), HFQ4G256 (6), HFQ4G128 (7) | `hfq.rs:805` classify, `llama.rs:641` `EmbeddingFormat`, `embedding_lookup_*` |
| proj / attn / ffn weights, **untied** `lm_head`, per-token prefill | decode GEMV | ~everything (F32, F16, Q8, Q4K/Q6K, HFQ2-6, HFP4, MFP4, MQ2-8, Lloyd); only BF16 (vision) outside | `hipfire-dispatch/src/families/gemv.rs:383-409` |
| DFlash draft target | WMMA batched prefill | Q8_0 (3), HFQ4G256 (6), MQ4G256 (13) all archs; MQ3G256 (17) gfx11 + dense-qwen35 only | gate at `lib.rs:528-588` |

Implications: (a) an MQ4/MQ8/HFQ6/MFP4 **embedding table** (or tied lm_head in those
formats) cannot run — no lookup kernel. The packager never emits this (it demotes
embeddings to Q8/HFQ4, *"Q4 too lossy for embedding tables"* `main.rs:4063`), but an
externally-produced `.hfq` can; both the qwen35 `embed_classify` guard (715c9063) and the
`hfq.rs:805` classifier now fail loud instead of silently reinterpreting bytes as F16. (b)
DFlash is the narrowest path — its gate already errors clearly for unsupported targets. So
goal-2 reads precisely: **the loader is format-agnostic; runnable coverage is kernel-bounded
per role**, not a single global format set.

## Design

### 1. Object-safe `Carrier`, impls top-of-DAG

`LoadedModel` lives in `hipfire-loader` and depends on every arch crate. A carrier that
*returns* `LoadedModel` therefore **cannot** live in an arch crate (dependency cycle). So:

- Define the trait in `hipfire-loader` (not runtime), object-safe (no associated type):

  ```rust
  pub trait Carrier {
      fn name(&self) -> &'static str;
      fn probe(&self, src: &ModelSource) -> bool;
      fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String>;
  }
  ```

- Carrier **structs + impls live in `hipfire-loader/src/carriers.rs`**.
- Remove the associated-`Bundle` `Carrier` trait from `hipfire-runtime::loader_api`
  (keep `ModelSource`, `LoadCtx`, `CaskConfig` there).
- Remove the `Carrier` impls from the arch crates. **Keep** each arch's `Bundle` struct and
  expose a typed builder it already effectively has:
  `hipfire_arch_qwen35::load_bundle(src, ctx) -> Result<Qwen35Bundle, String>`
  (= Step B's current `Qwen35Carrier::load` body, minus the LoadedModel-level
  eviction/dflash/vision wrapping). Step B's real value — arch crates owning their
  KV/adaptive/DeltaNet/refusal logic — is preserved; only the *impl* moves up.

### 2. Registry & dispatch

```rust
const REGISTRY: &[&dyn Carrier] = &[
    &Qwen2Carrier, &Qwen35Carrier, &LlamaCarrier,
    &DotsOcrCarrier, &Deepseek4Carrier, &MinimaxCarrier, &Lfm2MoeCarrier,
];
```

`probe()` matches **per source variant** because HFQ and Dir arch-ids are different
namespaces (§3). Mapping:

| carrier | probe(Hfq) | probe(Dir) | source/mode support |
|---|---|---|---|
| Qwen2 | `== 7` | — (no Dir path) | Hfq + pp1 only |
| Qwen35 | `5 \| 6` | `5 \| 6` | Hfq+pp1, Hfq+ppN, Dir+pp1, VL |
| Llama | `< 5` | `0 \| 1` | Hfq+pp1, Dir+pp1 |
| DotsOcr | `== 8` | — | Hfq+pp1 |
| Deepseek4 | `== 9` | — | Hfq+pp1 (+ EP path as today) |
| Minimax | `== 10` | — | Hfq+pp1 (+ EP) |
| Lfm2Moe | `== 11` | — | Hfq+pp1 |

The Dir column reproduces today's `load_model_safetensors` routing exactly: `derive_arch_id`
0/1 → llama path, 5/6 → qwen35 path (`lib.rs:857,885`). No qwen2 Dir path exists today and
none is added. Non-core carriers' `load()` simply call the existing `load_deepseek4()`/etc.

**Behavior change (HFQ only):** llama's HFQ probe is `arch_id < 5`, not the current "else"
catch-all. An unknown HFQ `arch_id` (≥ 12) now returns a clean `no carrier` error instead of
silently entering the llama loader and failing deep in `load_weights`. Intentional.

### 3. Unify the two axes (no side-doors)

**⚠️ B1 — the two arch-id namespaces are NOT reconciled (verified).** `derive_arch_id`
(`safetensors_source.rs:179-223`) emits `{llama/mistral→0, qwen2/qwen3→1, qwen3.5/3.6→5/6}`
— a *different* integer space from HFQ (`{qwen2→7, llama→<5, qwen35→5/6, dots→8, ds4→9,
minimax→10, lfm2→11}`). We do **not** merge them (that would be a risky data-model migration
touching every model on disk). Instead `probe` matches per-variant (§2 table). qwen35's 5/6
coincide across both spaces; llama bridges HFQ `<5` and Dir `0|1`; qwen2 has no Dir form.

**Source format → `ModelSource`.** Add `ModelSource::from_path(path) -> Result<Self, String>`:
for a file, open the HFQ header (as `load_model` does today at `lib.rs:562`); for a dir,
construct the existing `SafetensorsSource` (which already runs `derive_arch_id` at
`safetensors_source.rs:40`). `ModelSource::Dir` must therefore carry the `SafetensorsSource`
(or at least the derived arch_id + handle), not a bare `PathBuf` — today `Dir.arch_id()`
returns `None` (`loader_api.rs:23-25`), which would match nothing. `arch_id()` returns the
HFQ id for `Hfq` and the derived id for `Dir`; carriers' `probe` knows which space it's in.

**Tokenizer + chat_template are owned by the carrier, not pre-parsed (S1).** They come from
different places per format — HFQ from `hfq.metadata_json` (`lib.rs:563`), Dir from
`tokenizer.json` + `source.chat_template()` (`lib.rs:847,855`) — and `LoadCtx` carries
neither. Since the carrier consumes the source, the carrier resolves both and feeds them to
`LoadedModel::skeleton`. `load_model` does not pre-parse them (avoids the move-out-of-borrow
where `hfq` is both read for the tokenizer and moved into the carrier).

**The DFlash lm_head/MQ gate STAYS pre-dispatch (S2).** The block at `lib.rs:567-633` runs
before any arch dispatch and today guards *every* HFQ arch when `draft_path.is_some()`.
Folding it into `Qwen35Carrier` would stop it firing for non-core arches loaded with a draft
(behavior regression). It remains an arch-agnostic pre-check in `load_model` (it is a
DFlash/spec concern, explicitly out of the carrier-ownership scope of this design). One
mechanical change: the block currently binds `hfq` from `HfqFile::open`; under the unified
flow it must be re-expressed as `if let ModelSource::Hfq(hfq) = &src { … }` so it runs only
for HFQ sources (a Dir source never reaches it today — it sits after the `is_dir`
short-circuit — so the match arm preserves behavior exactly).

**Parallelism → `LoadCtx.pp`** (already present). pp>1 is a *mode*, not a new arch. Each
carrier's `load()` branches on `ctx.pp`. **The pp>1 arm must reproduce today's deliberate
drop of `draft_path`/`cask`/`kv_adaptive_override`** (`lib.rs:545`: `let _ = (...)`); these
are single-GPU concerns and `load_model_pp` ignores them. The pp arm must NOT route through
the single-GPU `finish_qwen35_load` eviction/dflash logic.

**Every non-qwen35 carrier MUST explicitly reject `ctx.pp > 1` (NEW — round-2 finding).**
Today pp>1 short-circuits at the top of `load_model` and `load_model_pp` errors for any arch
≠ 5/6 (`lib.rs:950`). Once dispatch is by *source*, a pp>1 request for e.g. a llama model
routes to `LlamaCarrier.load` — which would silently build a single-GPU model unless it
guards `if ctx.pp > 1 { return Err(...) }`. The "pp1 only" column in §2 is a **mandated
guard**, not a comment. Likewise `Qwen35Carrier` must reject the newly-reachable
`Dir × pp>1` combo (its pp path uses HFQ-only `load_weights_multi`).

Each carrier therefore owns **all (format × pp) combinations its arch supports** and returns
a clean error for the rest (same shape as existing refusals):

- `load_model_safetensors`' qwen35/llama bodies move into those carriers' `Dir` arm.
- `load_model_pp`'s body moves into `Qwen35Carrier`'s `pp>1` arm.
- VL detection (currently flat-field logic in `load_model` ~659-676) moves into
  `Qwen35Carrier`.

`load_model_pp` / `load_model_safetensors` stop being entry points. The sparse capability
matrix is expressed as `load()` errors, never as missing dispatch.

### 4. `LoadedModel::skeleton` — kill the 40-field literal

8 load sites today repeat a ~40-field struct literal (mostly `None`). Add:

```rust
impl LoadedModel {
    /// Shared-field skeleton: arch state None, pp 1, all non-core arch slots
    /// None, collections empty, mtp defaults, asst cache from env.
    pub fn skeleton(arch_id: u32, tokenizer: Tokenizer, max_seq: usize,
                    physical_cap: usize, model_path: String,
                    chat_template: Option<String>) -> Self { … }
}
```

Carriers set only owned fields via struct-update:

```rust
Ok(LoadedModel {
    state: Some(ModelState::Qwen35(bundle)),
    eviction, dflash, vision_config, vision_weights,
    ..LoadedModel::skeleton(arch_id, tokenizer, max_seq, physical_cap, path.into(), chat_template)
})
```

`#[derive(Default)]` is unavailable (tokenizer required; `AsstTurnCache`/`ModelState` have no
default), so `skeleton` is an explicit constructor of the genuinely-required shared fields.
Removes a latent-bug class (forgotten `None` field silently wrong).

**S4 — load-bearing override fields.** `skeleton` defaults `pp = 1` and all multi-GPU/VL
slots to `None`. The pp>1 arm MUST struct-update **all four** of `pp`, `pp_gpus`,
`pp_scratch_set`, `pp_dn_la_to_device`. Failure modes differ per field (verified against
`unload_model`): `pp_gpus` (`lib.rs:1269`) and `pp_dn_la_to_device` (`:1275`) are `.expect()`
→ **panic** if missing; `pp_scratch_set` (`:1270`) is `if let Some` → **silent VRAM leak** if
missing. So a `pp>1 ⇒ pp_gpus.is_some()` debug-assert catches the panic-fields but NOT a
dropped `pp_scratch_set` — the only real guard is a single construction helper. Prefer a
`LoadedModel::skeleton_pp(...)` (or a `Qwen35Bundle`→pp builder) that sets all four together,
so they can't be set piecemeal. The VL arm must likewise set `vision_config`/`vision_weights`
(freed at `lib.rs:1336`). These are the few fields where `skeleton` does *not* remove risk.

## Out of scope

- **DFlash 8-restore-site collapse** (review Finding 2) — daemon-side, functionally correct
  today (dflash coherence gate passes), tracked separately.
- No change to `generate*`, `ModelState`, or the daemon beyond imports.

## Migration order (each stage: compiles + gated + commit)

1. **`skeleton` cleanup** — add constructor, refactor the 8 existing sites. No behavior change.
   `cargo build --example daemon` + `coherence-gate.sh`.
2. **Object-safe trait + core registry** — trait in loader, core carrier structs/impls into
   `carriers.rs` returning `LoadedModel`, drop runtime trait + arch-crate impls (keep
   `Bundle` + `load_bundle`), replace the `if 7/5|6/else` ladder with `find(probe).load()`.
   Build + `coherence-gate.sh` + `coherence-gate-dflash.sh`.
3. **Fold non-core** — wrap dots_ocr/deepseek4/minimax/lfm2moe as carriers, extend `REGISTRY`,
   delete the `match hfq.arch_id` ladder. Build + coherence.
4. **Fold safetensors (highest-risk stage)** — add `ModelSource::from_path` + make `Dir`
   carry `SafetensorsSource` so `arch_id()`/`probe` work on the Dir namespace (B1/B2); give
   qwen35/llama carriers a `Dir` arm reproducing today's routing (5/6→qwen35, 0/1→llama);
   carriers own tokenizer/template resolution (S1); delete `is_dir` pre-branch. Build +
   **safetensors smoke** on `~/.hipfire/models/shisa-Qwen3.6-35B-A3B-PARO-packed` — and
   verify it routes to qwen35, not llama. Own commit.
5. **Fold pp>1** — move `load_model_pp` into `Qwen35Carrier` `pp>1` arm, reproducing the
   draft/cask/kv_adaptive drop (S3) and setting all four pp fields + the
   `pp>1 ⇒ pp_gpus.is_some()` debug-assert (S4); delete `pp>1` pre-branch. Build + coherence.
   ⚠️ pp>1 functional validation requires hiptrx (4×R9700) — required-before-merge, not skippable.
6. **Fold VL** — move vision detection into `Qwen35Carrier`. Build + VL smoke if a model is
   present.
7. **Dead-code sweep** — delete orphaned `parse_kv_adaptive` / `screen_weights_qwen35` /
   `warn_tiny_model_state` and the unused `daemon.rs` imports (`hip_bridge::HipResult`,
   `hipfire_arch_llama::Llama`). Final full gate.

## Verification / required-before-merge

- `cargo build --example daemon -p hipfire-runtime` clean.
- `coherence-gate.sh` green (qwen35 arch 5/6 covered) + human eyeball.
- `coherence-gate-dflash.sh` green (spec path) + three-tier eyeball.
- Fresh-process perf A/B (`scripts/probe_commits.sh`) Δ≈0% — dispatch is once-per-call.
- **Functional smoke gaps to close (cannot run on k9lin):**
  - qwen2 (arch 7) + llama (arch <5): no local models — source one each.
  - pp>1: hiptrx only.
  - VL: no local model.

### Tier 2: unified loader (WeightSource + device slice) — added 2026-06-12

The three `load_weights*` entry points were unified behind one `assemble_weights`
driver over a `&mut [Gpu]` device slice + `Layout`, with HFQ/PaRo isolated behind a
`WeightSource` trait (`HfqSource`/`ParoSource`). Single-GPU is the `len()==1` case;
multi-GPU is HFQ-only (PaRo `prepare(n>1)` errors). Tied-embedding alias is gated to
`len()==1` to stay byte-identical. The public API was collapsed to a single
`load_weights` entry (was 3). Carrier registry (Tier 1) unchanged in behavior.
