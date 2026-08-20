<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Kaden Schutt
hipfire — see LICENSE and NOTICE in the project root.
-->

# Roundhouse — training and alignment system (design)

Consolidate teacher/capture/score *orchestration*, alignment or training loops,
and promotion *decision packaging* into one planned workflow owner, without
absorbing packing codecs, quality wire formats, corpus/eval primitives, serving,
bench, Redline certification, or admissions.

| Field | Value |
|---|---|
| Date | 2026-08-20 |
| Status | **planned** — intent only. No Roundhouse crate, binary, or API exists. This document authorizes no code moves, renames, or admissions. |
| Lifecycle (INDEX) | Design-time architecture draft under [`docs/design/`](README.md). Cite only as intent / proposal. Not shipped, not branch-implemented, not measured product authority. |
| Name | **Roundhouse** (subsystem label). Code and contracts use descriptive technical names (see § Naming). |
| Scope | One planned crate initially. Existing functional crates keep their names. |
| Product admission | **None implied.** Roundhouse may *invoke* serve / perf / Redline / registry tooling and may package evidence; it must not self-certify serve correctness, performance floors, or [`admissions.yml`](../admissions.yml) rows. |

Truth labels follow [`docs/INDEX.md`](../INDEX.md). Where a claim is inference
rather than tree fact it is marked `[INFERENCE]`.

---

## 1 · Problem

The teacher → calibrate → capture → pack → score → promote loop is real and
repeated, but ownership is scattered across examples, scripts, one research
parent crate, and the quantizer. A future maintainer cannot place new work
without reverse-engineering paths.

Concrete symptoms visible in-tree today:

1. **Teacher / KLD reference producers are arch-forked examples**, not one
   contract:
   - `crates/hipfire-runtime/examples/build_kld_ref.rs` (llama-perplexity FIFO → HFKLDR)
   - `crates/saddle-lab/examples/build_kld_ref_native.rs` (hipfire F32 oracle → HFKLDR)
   - `crates/hipfire-runtime/examples/build_kld_ref_native_gemma4.rs`
   - `crates/hipfire-runtime/examples/build_kld_ref_native_glimmer.rs`
   - Candidate scoring: `crates/hipfire-runtime/examples/eval_hipfire.rs`
   - KV-mode quality gate: `crates/saddle-lab/examples/kv_mode_kld.rs`

2. **Calibration and Hessian capture live in multiple homes with overlapping
   contracts:**
   - `crates/hipfire-runtime/src/calibration.rs` — on-GPU `CalibCollector`,
     `.calib.hfq` streaming, daemon `CalibratableBackend` seam
   - `crates/hipfire-runtime/examples/calib_sweep.rs` — dense HFIM+HFHS shim
   - `crates/hipfire-quantize/src/bin/collect_e8_hessian.rs` — E8H1 `.hblk`
     from activation dumps
   - `crates/hip-bridge/examples/collect_e8_hessian_rocblas.rs` — rocBLAS path
   - `crates/hipfire-ds4-parent/src/hessian.rs` — parent online accumulator
     (same E8H1 wire) behind a backend marked **not** a calibration reference
   - `scripts/collect_hessian.py`, `scripts/*calib*`, `scripts/verify_calib_artifacts.py`

3. **Alignment / draft training is script-local**, not a scheduled candidate
   loop: `scripts/mtp_train/*`, `scripts/distill/*`, `scripts/dflash_train_poc.py`.
   Path C trainer work is already closed as a failed experiment in the lean-up
   map; Roundhouse must not resurrect it as product authority.

4. **Packing, serve validation, and admission already have owners**, but
   calibration scripts and research crates often blur “I produced a candidate”
   with “this may ship.” That blur is the failure mode Roundhouse is meant to
   end.

5. **`hipfire-ds4-parent` is a known-incorrect end-to-end forward** (PPL ~59.5
   vs torch teacher ~4.7 on the locked 1024-token fixture) yet still holds
   reusable plog / Hessian / manifest machinery. Without an explicit quarantine
   rule, salvage and teacher authority get conflated again.

---

## 2 · Boundary (one sentence)

**Roundhouse owns Waybill workflow state, GPU/process scheduling of the teacher →
corpus/calibration → capture → alignment/training → pack-invoke → score →
promotion-*decision* loop, external-backend invocation, autograd workers, and
candidate decision packaging; it reuses `saddle-quant` for corpus/eval/format/
statistics contracts, calls `hipfire-quantize` for deterministic packing,
execution crates for Hipfire model forwards, and independent serve / bench /
Redline / registry-admission tools for validation — and never self-certifies
those results or re-owns saddle-quant primitives.**

---

## 3 · Proposed shape (single crate)

**Planned crate name:** `roundhouse` (path `crates/roundhouse/`).
**Not created yet.** One crate initially; no sibling crates in v1.

Rust owns **waybill manifests, scheduling, decision packaging, and shared wire
contract consumption**. Authoritative **external teacher backends** (including
Python/PyTorch oracles such as the DS4 `reference_oracle`) are first-class
invoked authorities for teacher logits; they are not “dev-only workers.” Python
may also exist as **developer-only autograd workers** under `workers/` for
experimental draft-head / distill steps. In all cases Python is never the
authority for waybill schemas, promotion decisions, or hipfire quality wire
formats — those remain Rust contracts owned by Roundhouse (waybill/schedule) or
`saddle-quant` (formats/eval/corpus schemas).

### 3.1 Internal modules

Era-themed module labels are stable *directory* names. Public types, CLI flags,
and manifest keys use the technical names in the right-hand column. Modules
named `corpus` / `teacher` / `capture` / `score` are **orchestration adapters**
over `saddle-quant` and execution backends — they do **not** duplicate or
re-own corpus schemas, `ArtifactId` / `Estimator` / `WindowSpec`, KLD
reference/sequence formats, HFQ / Hessian / imatrix formats, teacher/candidate
evaluation primitives, or statistical reduction. Those stay in `saddle-quant`;
wire-format documentation stays with `saddle-quant` and is **linked** from
Roundhouse.

| Module label | Technical responsibility | Owns (orchestration) | Does not own (canonical elsewhere) |
|---|---|---|---|
| `waybill` | Run / candidate **manifests and provenance** | Schema for identity (model hash, corpus id, teacher id, capture id, recipe, git SHA, binary digest); join keys for later validation artifacts; fail-closed required fields | Admissions rows; registry model entries; serve config; quality wire bytes |
| `corpus` | **Corpus schedule adapter** | Selecting/fetching/verifying corpus artifacts for a waybill; invoking `saddle-quant` corpus builders; recording digests on the waybill | Corpus schemas/building contracts; long-term dataset hosting; Redline golden fixtures — **`saddle-quant`** |
| `teacher` | **Teacher backend invocation** | Registering and invoking a *declared* teacher backend (Hipfire F32 oracle, llama-perplexity, **or external PyTorch/other authority**); attaching produced HFKLDR / plog / PPL baselines to a waybill; recording teacher self-consistency floors | HFKLDR/plog wire schemas; `ArtifactId` / estimator contracts; implementing arch forwards; claiming a quarantined parent forward is teacher — formats/eval in **`saddle-quant`**, forwards in **arch crates** |
| `capture` | **Capture job orchestration** | Arming capture seams on execution backends; scheduling acts / imatrix / Fisher / Hessian jobs; requiring producing-weight identity on the waybill | E8H1 / HFIM+HFHS / `.calib.hfq` / acts-dump format contracts; quant math; packing HFQM — formats in **`saddle-quant`**, pack in **`hipfire-quantize`** |
| `score` | **Score job orchestration** | Scheduling candidate eval vs a pinned teacher artifact; attaching score cards to a waybill | KLD / PPL / top-1 / bucket metric primitives, HFKSEQ reducers, statistical reduction — **`saddle-quant`**; perf floors; Redline retained-PM4 proof; admissions |
| `schedule` | **Loop orchestration** | Ordering teacher → corpus → capture → (optional train) → pack-invoke → score → decision package; retries; artifact layout; bounded search recipes (e.g. fixed-byte mixed-precision candidate sets) | Daemon request scheduling; GPU kernel launch policy; pack codecs; quality optimizers claimed as product |
| `workers/` (optional tree) | **Developer-only autograd workers** | Python (or other) *training* steps called as subprocesses with explicit I/O contracts | Promotion authority; production serve path; **authoritative external teachers** (those register under `teacher`, not `workers/`) |

### 3.2 Language and backend split

| Concern | Owner | Rationale |
|---|---|---|
| Waybill manifests, IDs, scheduling, decision packaging, CLI | Rust (`roundhouse`) | Deterministic, reviewable workflow authority |
| Corpus schemas/building, `ArtifactId` / `Estimator` / `WindowSpec`, KLD ref/seq + HFQ/Hessian/imatrix formats, teacher/candidate eval primitives, stats | Rust (`saddle-quant`) | Canonical quality/tooling layer; Roundhouse consumes, does not fork |
| Deterministic weight packing / codecs | `hipfire-quantize` (invoked) | Already the packing authority |
| Hipfire model forward during capture / native teacher / score | Existing execution stack (daemon/engine/loader/runtime + arch + saddle-core) | **“No second engine”** = do not reimplement shipped Hipfire execution |
| Authoritative external teacher backends (e.g. DS4 PyTorch oracle) | External process (often Python/PyTorch), registered by `teacher` | Teacher *authority* may leave Rust; Roundhouse still owns manifests, scheduling, decisions, and shared wire *consumption* |
| Autograd / experimental draft-head training | Python worker under `workers/` (optional) | Existing `scripts/mtp_train` / distill patterns; stays outside default product graph and is not a teacher authority |

### 3.3 Relationship to existing layers (planned)

```
                    ┌─────────────────────────────────────┐
                    │ roundhouse (planned)                │
                    │ waybill · schedule · workers/       │
                    │ corpus/teacher/capture/score        │
                    │   = adapters over saddle-quant +    │
                    │     execution / external backends   │
                    └──────────────┬──────────────────────┘
           invokes                 │                invokes
    ┌──────────────┐   ┌───────────┴──────────┐   ┌──────────────────────┐
    │ saddle-quant │   │ execution stack      │   │ independent truth    │
    │ (corpus /    │   │ hipfire-daemon       │   │ (not Roundhouse):    │
    │  eval /      │   │ hipfire-engine       │   │ · Atlas = measure    │
    │  formats /   │   │ hipfire-loader       │   │   observation only   │
    │  stats)      │   │ hipfire-runtime      │   │ · registry = metadata│
    │ hipfire-     │   │ saddle-core          │   │ · VALIDATION = route │
    │ quantize     │   │ hipfire-arch-*       │   │   selector           │
    │ (pack only)  │   │ (+ external teacher  │   │ · Redline = retained │
    └──────────────┘   │  backends when       │   │   replay certify     │
                       │  declared)           │   │ · admissions.yml =   │
                       └──────────────────────┘   │   independent auth.  │
                                                   │ · serve/bench harness│
                                                   └──────────────────────┘
```

Roundhouse **reads** validation outputs into a waybill decision package. It does
**not** become the owner of VALIDATION routes, Redline policy, Atlas schema,
registry cards, or `admissions.yml`. Atlas remains measurement observation only;
registry remains metadata; VALIDATION selects human/CI routes; Redline certifies
retained replay; `admissions.yml` is an independent authority Roundhouse cannot
mint by packaging evidence.

---

## 4 · Current-infrastructure migration table

What moves *into* Roundhouse ownership when implementation is authorized.
Until then these paths remain the live owners. “Migrate” means clean cutover of
*orchestration and contracts*, not a silent second copy.

| Current location | Role today | Planned Roundhouse home | Notes |
|---|---|---|---|
| `hipfire-runtime/examples/build_kld_ref.rs` | llama.cpp KLD base → HFKLDR | `teacher` (+ thin CLI) | Keep llama pin as one teacher backend, not the only one |
| `saddle-lab/examples/build_kld_ref_native.rs` | hipfire F32 oracle → HFKLDR (qwen35) | `teacher` | Prefer native teacher when engine-port confound must cancel |
| `hipfire-runtime/examples/build_kld_ref_native_gemma4.rs` | gemma4 native HFKLDR | `teacher` (arch adapter) | Arch-specific forward stays in arch crate |
| `hipfire-runtime/examples/build_kld_ref_native_glimmer.rs` | glimmer native HFKLDR | `teacher` (arch adapter) | Same |
| `hipfire-runtime/examples/eval_hipfire.rs` | HFKSEQ candidate KLD vs HFKLDR | `score` | Scoring modes stay documented; reducer scripts may remain shared |
| `saddle-lab/examples/kv_mode_kld.rs` | asym3 vs q8 KV full-softmax KLD | `score` (specialized card) | Still not a serve admission |
| `hipfire-runtime/examples/calib_sweep.rs` | dense calib shim over `calibration::collect*` | `schedule` + `capture` driver | Core collector may stay in runtime initially (see keep table) |
| `hipfire-quantize/src/bin/collect_e8_hessian.rs` | acts dump → E8H1 `.hblk` | `capture` | Wire format stays shared with quantize loader |
| `hip-bridge/examples/collect_e8_hessian_rocblas.rs` | rocBLAS E8H1 builder | `capture` backend | Model-agnostic math; provenance from caller |
| `hipfire-ds4-parent/src/hessian.rs` (generic E8H1 / accum API) | online parent Hessian | **salvage into `capture`** | Only generic accum + write path — see § 6 |
| `hipfire-ds4-parent/src/plog.rs` | plog read/write/compare | **salvage into `teacher` / `score`** | Format helpers only |
| `hipfire-ds4-parent/src/manifest.rs` | parent capture manifest schema | **salvage patterns into `waybill`** | Do not import DS4-only fields as global required set without generalization |
| `scripts/fetch_calibration_corpus.sh` | corpus fetch | `corpus` | |
| `scripts/verify_calib_artifacts.py` | artifact checks | `corpus` / `waybill` verify | |
| `scripts/collect_hessian.py` | Hessian helper | `capture` or delete after cutover | |
| `scripts/calibrate_multigpu.sh`, `mi300x_calib_rest.sh`, `package_mi300x_calib.sh`, `mq4_masked_calib.py` | campaign wrappers | `schedule` recipes or stay as thin callers | Campaign glue is not product admission |
| `scripts/mtp_train/*` | MTP head training experiments | `workers/` (dev-only) | Not Path C resurrection; explicit research worker |
| `scripts/distill/*` | distill parallel helpers | `workers/` (dev-only) | |
| `scripts/dflash_train_poc.py` | DFlash train POC | `workers/` or archive | Lean-up map closed Path C as failed experiment — no product promise |
| `scripts/reap/kld_compare.*`, `run_ppl_kld.sh` | PPL/KLD compare helpers | `score` adapters or keep under Reap callers | Reap plan ownership stays in `hipfire-reap` |
| `benchmarks/quality-baselines/harness/{kldref_format.py,kld_diff.py,kld_reduce.py}` | Legacy Python KLD parse/reduce | Call or retire behind `saddle-quant` | Canonical Rust formats live in `crates/saddle-quant/src/format/{kldref,kldseq}.rs`; do not fork them |
| `hipfire-arch-deepseek4/reference_oracle/` (torch teacher) | DS4 teacher PPL/plog | External **teacher backend** registered by `teacher` | Remains the DS4 calibration teacher authority |
| `hipfire-arch-deepseek4/scripts/plog_fine_scan.py` | position scan | `score` tooling | |

---

## 5 · Keep-in-place table

These stay where they are. Roundhouse **calls** them; it does not absorb them.

| Owner | What stays | Why |
|---|---|---|
| `hipfire-quantize` | Deterministic packing, GPTQ/E8/MQ/HFQ codecs, imatrix/AWQ apply, HFQM write, `--hessian-dir` consume, REAP overlay bake hooks | Packing and on-disk quant math authority ([`QUANTIZE.md`](../QUANTIZE.md), [`QUANTIZATION.md`](../QUANTIZATION.md)) |
| `hipfire-daemon` | Product process/message dispatch and lifetime | Roundhouse does not enter the serving process |
| `hipfire-engine` | Serve scheduler, terminal control, prompt, emit | Training workflow is not request scheduling |
| `hipfire-loader` | Composition root, carrier registry, `load_model`, `LoadedModel` | Roundhouse invokes supported execution; it does not own model composition |
| `hipfire-runtime` | Shared model/tokenizer/spec/calibration infrastructure and `Architecture` trait | Capture may invoke existing seams; it does not relocate execution infrastructure |
| `saddle-core` | Grammar, KV cache, capabilities, sampling policy | Serving substrate remains below runtime |
| `saddle-lab` | One-off research harnesses not yet migrated | Lab remains available for experiments |
| `saddle-quant` | Corpus construction, artifact identities, estimator/window contracts, canonical HFQ/Hessian/imatrix/KLD formats, evaluation primitives, statistics | Roundhouse orchestrates these APIs; it does not duplicate them |
| `hipfire-arch-*` | Architecture forwards, weight load, arch-specific calibration taps | “Abstract the model, never the kernel” |
| `hipfire-ds4-parent` (crate retained) | DS4 parent checkpoint load/forward research backend | Research crate; **not** teacher authority (§ 6) |
| `hipfire-reap` | Expert importance gather, `ExpertPlan`, load-time overlay | Selective MoE prune/re-quant owner |
| `hipfire-atlas` + `scripts/kernel_atlas.py` | Perf observation corpus | Atlas is not a gate and not a quant teacher |
| Redline crates + [`REDLINE.md`](../REDLINE.md) | Retained-PM4 certification ladder | Independent validation |
| `hipfire-registry` + `registry/*` | Model cards, VRAM, sampling defaults | Registry ≠ admission |
| [`docs/admissions.yml`](../admissions.yml) | Machine admission registry schema v2 | Only earned rows; Roundhouse cannot create authority by writing one |
| [`docs/VALIDATION.md`](../VALIDATION.md) | Human route selector | Sole validation route map |
| Serve / bench harnesses (`tools/serve_harness`, `tools/benchlocal`, bench examples) | Latency/throughput measurement | Perf truth stays outside Roundhouse |
| `rdna-compute` / `hip-bridge` / `radiowave` / kernels | Compute substrate | Out of scope |

---

## 6 · DS4-parent quarantine and salvage rule

### 6.1 Quarantine (hard)

`crates/hipfire-ds4-parent` is **not** a calibration or teacher reference.

Recorded on the crate banner and in
[`docs/investigations/2026-08-02-ds4-parent-not-calibration-ref.md`](../investigations/2026-08-02-ds4-parent-not-calibration-ref.md):

| system | PPL @1024 | mean KLD vs ref_fp8 |
|---|---:|---:|
| torch teacher (fp8) | **4.693** | 0 |
| torch teacher (exact) | 4.624 | 0.040 |
| **Rust parent forward** | **59.507** | **2.718** |

Rules:

1. **Do not** elevate the Rust parent forward to Roundhouse `teacher` authority.
2. **Do not** use any parent-forward-derived activation, imatrix, Fisher, or
   Hessian artifact for calibration, GPTQ, or alignment. A future exception
   requires a separate source and governing-document reclassification backed
   by an independent teacher; a downstream candidate win alone is insufficient.
3. **Do not** rename or promote historical captures that lack provenance
   manifests (see rejected MQ2R-driven “native” capture in
   [`docs/investigations/2026-08-01-ds4-parent-hessian-handoff.md`](../investigations/2026-08-01-ds4-parent-hessian-handoff.md)).
4. Component-gate green (load, HC, MoE smoke, residual norms, head path) **must
   not** be read as end-to-end teacher fidelity.

**DS4 teacher authority** remains the independent torch harness under
`hipfire-arch-deepseek4/reference_oracle/` (`ref_ppl_e2e.py` → `ref_fp8_*.plog`),
registered as a Roundhouse teacher backend when implemented.

### 6.2 Salvage (allowed)

Generic, model-agnostic pieces may be lifted into Roundhouse **without**
bringing the parent forward:

| Salvage | Destination | Condition |
|---|---|---|
| E8H1 header/layout constants and block-accum write path | `capture` | Byte-compatible with `hipfire-quantize` loader; no parent backend type in the public API |
| plog container read/write/compare | `teacher` / `score` | No dependency on `Ds4ParentBackend` |
| Manifest field patterns (producer, corpus, capture tensor list, hashes) | `waybill` | Generalize; drop DS4-only required keys from the base schema |
| Acts dump `[u32 rows][u32 K][f32…]` contract | `capture` | Already shared with quantize/bin tools |

Salvage is a **copy or move of generic code** under Roundhouse ownership, not a
re-export of `hipfire-ds4-parent` as the training stack. The research crate may
remain for DS4 parent load experiments; its status stays `research` per glossary
vocabulary.

---

## 7 · End-to-end data flow (planned)

```text
[source weights / HF / GGUF / parent ckpt]
            │
            ▼
     ┌─────────────┐
     │   corpus    │  slice + tokenize policy + digest
     └──────┬──────┘
            ▼
     ┌─────────────┐
     │   teacher   │  declared backend → HFKLDR / plog / PPL floor
     └──────┬──────┘
            ▼
     ┌─────────────┐
     │   capture   │  acts / imatrix / Fisher / Hessian (E8H1, HFIM…)
     └──────┬──────┘     provenance: producing weight identity required
            │
            ├──────────────────────────────┐
            ▼                              ▼
   ┌─────────────────┐            ┌─────────────────┐
   │ workers/ (opt.) │            │ hipfire-quantize│
   │ align / train   │            │ pack HFQM       │
   └────────┬────────┘            └────────┬────────┘
            │                              │
            └──────────────┬───────────────┘
                           ▼
                    ┌─────────────┐
                    │    score    │  KLD/PPL/top-1 vs pinned teacher
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │   waybill   │  candidate decision package
                    └──────┬──────┘
                           │
           may invoke (read-only w.r.t. authority)
                           ▼
        serve harness · bench · Atlas · Redline · VALIDATION routes
                           │
                           ▼
              human / CI records admissions.yml  ← NOT Roundhouse
```

**Invariant:** every artifact edge carries a waybill id. Capture without
producing-weight identity is rejected. Score without pinned teacher artifact is
rejected. Pack is a subprocess/library call into `hipfire-quantize`, not a
reimplementation.

**Decision package** (planned contents): waybill id, score card, pack output
hash, optional paths to external validation artifacts. Status values are
explicitly non-admission, e.g. `quality_scored`, `pack_complete`,
`external_validation_attached`, `rejected_provenance` — never `admitted`.

---

## 8 · Staged clean-cutover migration order

Implementation is **not** authorized by this document. When a future change
authorizes work, cut over in this order so no second source of truth lingers.

| Stage | Work | Exit gate |
|---|---|---|
| **S0** | Land the crate with complete, versioned `Waybill` and `CorpusIdentity` contracts for one existing calibration campaign | Round-trip tests pass; missing producer/model/corpus/teacher identities fail closed; no empty scaffold |
| **S1** | `teacher`: unify HFKLDR producers behind one CLI/API; arch adapters call arch crates | `build_kld_ref*` examples become wrappers or move; one format test vector |
| **S2** | `capture`: E8H1 + acts contracts; fold `collect_e8_hessian` CLI; optional salvage from ds4-parent hessian/plog **without** parent forward | Quantize still reads the same `.hblk`; provenance required on write |
| **S3** | `score`: fold `eval_hipfire` + reducers; attach score cards to waybills | Quality-baselines harness still parses outputs |
| **S4** | `schedule`: one documented loop recipe (teacher → capture → pack-invoke → score) for one arch | Replaces ad-hoc shell for that recipe only |
| **S5** | `workers/`: optional MTP/distill entrypoints as dev-only | Default build does not require Python training |
| **S6** | Delete or archive superseded examples/scripts after callers update | One owner remains per migrated contract |

Rules for every stage:

- **Clean cutover:** migrate callers; no permanent shim dual-path.
- **Do not rename** `hipfire-quantize`, runtime, arch, Reap, Atlas, Redline, or
  registry crates.
- **Do not** move GPU calib reduce kernels out of runtime/compute in S0–S5;
  schedule/capture may call them in place.
- **No admissions language** in Roundhouse CLIs.

---

## 9 · Naming and legibility

| Layer | Rule |
|---|---|
| Subsystem | **Roundhouse** — human/docs label for the training/alignment loop owner |
| Crate | `roundhouse` |
| Modules | `waybill`, `corpus`, `teacher`, `capture`, `score`, `schedule`, `workers` |
| Types / CLI / JSON keys | Descriptive technical English: `TeacherBackend`, `CaptureProvenance`, `ScoreCard`, `PromotionDecision` (decision ≠ admission), `HessianBlockFile`, `KldReferenceArtifact` |
| Wire formats | Keep existing magic/names where shared (`HFKLDR`, `HFKSEQ`, `E8H1`, `.calib.hfq`); document in one place under Roundhouse + link from QUANTIZE |
| Era metaphors | Allowed only as module directory aliases explained in this doc; never as the sole name of a public contract |
| Status vocabulary | INDEX truth states for docs; glossary `production` / `research` / `legacy` for crates — do not invent “Roundhouse-admitted” |

---

## 10 · Non-goals

- **Not** a product admission system and not a writer of [`admissions.yml`](../admissions.yml).
- **Not** a replacement for [`VALIDATION.md`](../VALIDATION.md), Redline
  certification, or Atlas measurement ownership.
- **Not** a second quantizer or HFQM packer (`hipfire-quantize` remains).
- **Not** a second inference engine or arch-crate merger.
- **Not** resurrection of Path C / failed DFlash trainer as roadmap product
  (lean-up map E1 closed).
- **Not** elevating `hipfire-ds4-parent` forward to teacher authority.
- **Not** absorbing Reap planning, registry model cards, or serve HTTP API.
- **Not** implying schedule, staffing, or ship dates; status is **planned** only.
- **Not** multi-crate split in v1 (no `roundhouse-teacher` / `roundhouse-score`
  crates until a later design revisits).
- **Not** requiring Python in the default user install path.

---

## 11 · Acceptance criteria (for this document and future implementation)

### 11.1 Document acceptance (this change)

A future maintainer can answer, without reverse-engineering scripts:

| New work | Belongs in |
|---|---|
| Teacher / reference job scheduling and backend invocation | Roundhouse `teacher` (planned); evaluation contracts remain in `saddle-quant`; arch and external teacher forwards stay with their backends |
| Calibration corpus selection, fetch, and campaign provenance | Roundhouse `corpus` over canonical `saddle-quant` corpus contracts |
| Activation / Fisher / Hessian capture orchestration + provenance | Roundhouse `capture`; formats remain in `saddle-quant`, execution taps in runtime/arch owners |
| Deterministic weight packing / codec | **`hipfire-quantize`** (keep) |
| Draft-head or distill autograd experiment | Roundhouse `workers/` (dev-only) or explicit research archive — not product serve |
| Candidate score-job scheduling and decision attachment | Roundhouse `score` over **`saddle-quant`** evaluation/statistics |
| Perf observation / kernel ranking | **Atlas** (keep) |
| Retained-PM4 / route proof | **Redline** (keep) |
| Model registry metadata | **hipfire-registry** (keep) |
| Product admission row | **`docs/admissions.yml` via human/CI process** (keep) — never Roundhouse self-certify |
| Product process/message lifetime | **`hipfire-daemon`** |
| Serve scheduling/prompt/emit | **`hipfire-engine`** |
| Model composition/loading | **`hipfire-loader`** |
| Shared execution infrastructure / architecture forward | **`hipfire-runtime`**, **`saddle-core`**, and **`hipfire-arch-*`** |

### 11.2 Future implementation acceptance (not claimed done)

When Roundhouse is implemented, it is acceptable only if:

1. Status can move from **planned** only by an INDEX/owner update backed by
   code on a named ref — not by this file’s age.
2. One waybill schema is the join key from teacher artifact → capture → pack
   output → score card.
3. DS4 calibration docs and CLIs point at the torch teacher backend, not the
   Rust parent forward.
4. Invocations of serve/bench/Redline/Atlas are clearly “attach external
   evidence,” with no API that returns `admitted`.
5. Superseded examples/scripts are removed or are one-line wrappers after
   cutover stages complete.

---

## 12 · Explicit non-claims

- No Roundhouse source tree exists at the time of writing.
- No performance, quality, or admission number is introduced here.
- Existing measured KLD/PPL figures cited above are **historical investigation
  evidence** for the quarantine rule, not Roundhouse product baselines.
- Saddle lean-up and Roundhouse are complementary: saddle-quant remains the
  broader quant/tooling layer discussion; Roundhouse is the narrower planned
  owner of the **training/alignment decision loop**.

---

## References (in-tree)

| Doc / path | Use |
|---|---|
| [`docs/INDEX.md`](../INDEX.md) | Truth states; ownership; admissions fail-closed |
| [`docs/design/README.md`](README.md) | Design directory lifecycle (planned intent) |
| [`docs/QUANTIZE.md`](../QUANTIZE.md) / [`QUANTIZATION.md`](../QUANTIZATION.md) | Packing authority |
| [`docs/VALIDATION.md`](../VALIDATION.md) / [`REDLINE.md`](../REDLINE.md) | Validation and certification owners |
| [`docs/methodology/kernel-atlas.md`](../methodology/kernel-atlas.md) | Atlas non-gate measurement |
| [`docs/governance/2026-08-15-saddle-design-grounding.md`](../governance/2026-08-15-saddle-design-grounding.md) | Layering context; parent eviction history |
| [`docs/governance/2026-08-15-hipfire-leanup-map.md`](../governance/2026-08-15-hipfire-leanup-map.md) | Path C closed; examples triage |
| [`docs/investigations/2026-08-02-ds4-parent-not-calibration-ref.md`](../investigations/2026-08-02-ds4-parent-not-calibration-ref.md) | Parent quarantine numbers |
| [`docs/investigations/2026-08-01-ds4-parent-hessian-handoff.md`](../investigations/2026-08-01-ds4-parent-hessian-handoff.md) | Provenance failure; salvageable tooling |
| [`docs/GLOSSARY.md`](../GLOSSARY.md) | Atlas, Reap, MagnumQuant status vocabulary |
| `crates/hipfire-ds4-parent/src/lib.rs` | In-code NOT A CALIBRATION REFERENCE banner |
