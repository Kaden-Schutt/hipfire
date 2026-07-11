# The RDNA Kernel Oracle — unified design

**Status:** design (unification spec — no new implementation yet; cross-model review incorporated 2026-07-02)
**Date:** 2026-07-02
**Branch (rebase target):** `feat/mtp-config-default` — **NOTE: this worktree has
diverged from that target** (as of 2026-07-02: 11 commits on `feat/mtp-config-default`
are not in HEAD, 41 on HEAD are not in it; merge-base ≠ tip). Rebasing/merging onto
`feat/mtp-config-default` is a **required pre-implementation step**, not the current
state — do not treat this worktree as the integration base until that reconciliation
lands (see §h and Open Question 4).
**Worktree:** `perfmaxx-gfx1201-pkf16-gemv`
**Supersedes / absorbs:**
- `docs/serve-harness-gate-spec.md` (serve-harness + behavior-gate, DRAFT, uncommitted in `wf_2054dc39-912-4`)
- `docs/research/2026-07-01-gfx1201-native-kernel-perf-instrument-design.md` (#488 perf-instrument, Phase A landed no-GPU-green on this branch)

**Does NOT absorb:** the `hipfire-atlas` crate. We reuse its *data* (the `.hip`
kernel corpus, the kernel-fire sets, the `AtlasRow` schema as a read-only source)
and drop its *framing*. Atlas failed as a passive pre-tagged corpus that nothing
ever queried; the Oracle is not another one of those.

---

## (a) Mission

The RDNA Kernel Oracle is a **fleet-wide, daemon-measured, committed-in-repo corpus**
of, for every RDNA generation we own:

> **{ which kernel-modules fire } × { how each one is BOUND, measured } × { its
> in-model contribution, measured through the real serve path }**

— that LLMs and the agent-panel **reason over at query time** to answer three
questions no other tool answers for consumer AMD hardware:

1. **Where is the time actually going** in a real forward pass (not a microbench)?
2. **What binds each hot kernel** — bandwidth, VALU-issue, latency, occupancy, or
   launch-overhead — from **measured PMC counters**, not a byte model or a guess?
3. **Did this diff move a kernel it wasn't supposed to** (arch-bleed), and **did it
   cost tok/s** anywhere on the fleet?

AMD ships no such tool for RDNA. Omniperf/rocprof-compute is CDNA/MI-only; there is
no roofline or occupancy analyzer for gfx10xx–gfx12xx. The Oracle fills that gap.
Call it **omniperf-for-RDNA**, scoped to what our engine actually runs.

**Scope honesty (load-bearing, see §g).** The *measured* bound-class (the PMC
ratio domain) is validated on **gfx1201 only** today. "Fleet-wide" in v1 means the
**fired-kernel-set + wall-time-% domains** run on every RDNA gen we own; the **PMC
ratio domain is gfx1201-complete and per-arch-gated** — each other arch promotes to
full measured bound-class only after its own perfmon-counter map and gating lever are
validated (Open Question 3). The mission is fleet-wide; v1 does not claim the ratio
domain is fleet-wide yet.

The Oracle is **tooling** — offline, dev, and CI. It never touches the serving hot
path. It is Rust-native, and where it needs live counters it drives `rocprofv3`
out-of-process (never in-daemon).

---

## (b) Problem — why this, why now

**Atlas was passive and never used.** The `hipfire-atlas` crate + `kernel_atlas.py`
built a JSONL corpus of kernel rows. Nobody queried it, because it pre-tagged
kernels with static categories and stored isolated microbench numbers divorced from
any real forward pass. A corpus you don't reason over is dead weight. The Oracle's
first job is to not repeat that: **store raw measured data, tag nothing, and pair
every isolated number with an in-model number.**

**Measurement-hygiene churn cost real time.** The June MTP-perf week burned days on
ten confounds — temp-0.3 defaults, greedy overstatement, single-run noise, q8-OOM,
`max_tokens ≤ think_cap` empties, `mtp_mode=off` contamination, a wrong-generation
model (3.5 vs 3.6). Per the methodology-lessons note, a committed provenance record
would have caught 9 of the 10 at the source. The Oracle makes that record the price
of admission for any perf number.

**Arch-bleed is real and silent.** A change to a shared dispatch path or a
multi-arch kernel can move a kernel on gfx1100 while you were only testing gfx1201.
The `reference_kernel_module_cache_collision` bug (gfx12 MMQ HFQ4G256 loaded under
the gfx12 module name → empty-stub kernel → ~100% NRMSE) is the sharp end of this.
Today we catch it by luck and eyeball.

**Three teams built the same instinct three times.** This spec exists because three
parallel efforts are each a facet of one tool:

1. **The serve-harness + gate** (`serve_harness.py`, `serve-harness-gate-spec.md`):
   *measure real serve-path behavior + provenance, then gate on it.* Owns the
   **in-model data source** and the **policy layer**.
2. **The empirical kernel-trace / coverage infra** (`rocprof-wrap.sh`,
   `kernel_atlas.py`, `coverage-audit.py`, `docs/methodology/rocprof-coverage.md`):
   *which kernels really fire, and is the internal profiler blind to any of them?*
   The raw material for the **corpus**, currently scoped only to timing-coverage
   auditing, not to a comparative cross-arch corpus.
3. **The #488 perf-instrument** (`chip_profile.rs`, `roofline.rs`,
   `isa_histogram.rs`, `kernel_ledger.rs`, `profile_rocprof.rs`): *chip-tied
   roofline / occupancy / ISA / bound-class + a coherence-stamped regression ledger.*
   The **measurement engine**.

They share one instinct — *measure the chip honestly, store it, reason over it* —
and duplicate the pieces around it. Unifying them is the point of this document.

**The `profile_standard` unlock (2026-07-02) makes real bound-class measurement newly
possible.** On RDNA3/4 the `auto`/`high` DPM governor gates the perfmon clock, so
`rocprofv3 --pmc` read **exactly zero** for ALU-busy / mem-busy / occupancy —
which is why every prior bound-class verdict on gfx1201 was *inferred* from a byte
model and a failed VALU-vs-mem split. Setting
`power_dpm_force_performance_level=profile_standard` (STABLE_STD) ungates the
counters: validated tonight at **100% non-zero counters across 49 kernels** on
gfx1201. This upgrades the corpus from *"which kernels fire + wall-time"* to
*"which kernels fire + **measured** bound-class."* It is the technical enabler that
makes the Oracle worth building now rather than as more inferred scaffolding.

> **`profile_standard` caveat (load-bearing):** it pins clocks *low*, so it is valid
> for **counter RATIOS only** (ALU-busy%, mem-busy%, occ%). Absolute tok/s and
> wall-time-% stay on `high`/`auto` runs. The corpus keeps the two provenance
> domains separate and never mixes a `profile_standard` ratio with a `high`-clock
> latency.

---

## (c) Design principles (load-bearing — the spec honors each)

**1. DATA, NOT TAGS.** The corpus stores raw *measured* data: which kernel-modules
fired, their PMC counters and raw ratios, wall-time, ISA histogram, and in-model %.
Semantic and behavioral tags ("this is the bottleneck," "dp4a would help," "this is
arch-bleed") are **never pre-inferred or baked in.** LLMs and the agent-panel reason
over the raw data at query time. Atlas died as a passive pre-tagged corpus; we do not
build another.

*(Reconciliation of "measured bound-class" — resolved per the cross-model review, see
Open Question 2.) `bound_class` is a **query-time derivation**, so it is **NOT a
committed corpus field** — not in the isolated block, not in the in-model block. The
canonical corpus commits only the raw inputs the derivation runs over: PMC ratios
(ALU-busy% / mem-busy% / occupancy%), counters, the ISA histogram, register/LDS/scratch
static facts, timing, and provenance. The reasoning layer (§ layer 4) computes the
bound-class from those raw numbers on every query, via the roofline heuristic + the
five-class lens (§e). It may optionally memoize the result in a **non-committed**
query/render cache stamped with a `derived_by` version — but that cache is a
convenience, never the store, and it never enters the diffable corpus. Storing a
`bound_class` string in the corpus — even "recomputable, derived-not-authored" — was
the review's load-bearing refutation: a stored label is one editorial edit from a
frozen tag, and Atlas's exact failure mode. We do not store it.)*

**2. ISOLATED ≠ IN-MODEL.** A kernel can microbench as near-optimal yet be 0.1% of a
real forward pass; another can look fine isolated but be the hidden in-model
bottleneck. Every corpus row therefore carries **both**: isolated microbench fields
*and* in-model contribution measured through the **real daemon serve path**, never a
microbench standing in for the model. Tonight's A3B example is the canonical proof
(see §e / §b) — `attention_flash_q8_0_tile` reads 60% mem-busy in-window but 1%
occupancy, so at chip scale it is latency/launch-floor-gated, not BW-saturated; no
isolated microbench would have said that.

**3. FUZZ SHAPES.** Grow the corpus by fuzzing kernel shapes and configs (the fleet
battery) until optimal algorithms surface from the data. Until they do, we
hand-roll kernels — the corpus tells us *which* hand-roll to attempt and *whether it
moved the bound-class the way we predicted.*

**4. REUSE THE KERNEL DATA, DROP THE TOOL.** We reuse the existing `.hip` kernel
corpus and the kernel-fire sets **as data** (via a read-only `AtlasRow` mapping).
We do **not** absorb the `hipfire-atlas` crate or its passive-corpus framing. Reuse
the data; drop the tool that nothing used.

**5. MISSION-FIT.** Tooling / offline / dev only — **never** the serving hot path.
Rust-native; `rocprofv3` driven out-of-process via a separate step (never dlopen'd
into the daemon, to dodge the documented rocprofv3 SIGABRT/hang). AMD ships no RDNA
roofline/occupancy tool (Omniperf is CDNA/MI-only), so this fills a real gap.

---

## (d) Architecture — the layered unification

Five layers, bottom-up. Each maps to exactly one of the three prior threads (or is
the new seam that joins them). The rule: **lower layers only ever produce data;
only the top layer interprets it.**

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. POLICY LAYER  — the gate                          [from: serve-gate]    │
│    reads the DB → PR-readiness. AR −3% block, spec −8% block,              │
│    coherence hard-fail, perfdrain flag, arch-bleed required-set.          │
│    Applies the regression LABELS (RegressionHard/Soft/Gain/               │
│    RejectedGainNoCoherence) to 2a's raw facts. The ONLY layer pass/fail.  │
├─────────────────────────────────────────────────────────────────────────┤
│ 4. REASONING LAYER  — LLM / agent-panel        [new seam over the corpus] │
│    QUERY-TIME interpretation: bound-class reading, arch-bleed verdict,    │
│    optimization advice, required-arch set. Tags are produced HERE,         │
│    ephemerally, never written back as corpus tags.                        │
├─────────────────────────────────────────────────────────────────────────┤
│ 3. CORPUS DB  — committed in-repo JSONL             [new; unifies all 3]  │
│    per (behavior_key × domain × kernel): fired kernel-set; per-kernel RAW  │
│    ALU/mem/occ ratios + wall-time-% + in-model-contribution (NO stored     │
│    bound-class label — derived at layer 4); isolated ISA/registers/static; │
│    golden medians; perfdrain series; arch-coverage ledger. Append-only     │
│    runs (unique run_id) + curated baselines/ledger, joined by behavior_key.│
├──────────────────────────────────┬──────────────────────────────────────┤
│ 2a. MEASUREMENT ENGINE           │ 2b. IN-MODEL DATA SOURCE              │
│     (the #488 instrument)        │     (serve_harness)          [serve]  │
│     chip profiles · roofline ·   │     real daemon /v1/chat path ·       │
│     ISA histogram · ledger diff  │     rocprofv3 --kernel-trace + --pmc  │
│       (raw facts, not labels) ·  │                                       │
│     PMC-via-profile_standard ·   │     under profile_standard ·          │
│     microbenches (bw, latency)   │     provenance run-record             │
│           [#488]                 │           [serve-harness]             │
├──────────────────────────────────┴──────────────────────────────────────┤
│ 1. SUBSTRATE  — the fleet + the hardware honesty guards                   │
│    gpu-lock (flock) · profile_standard PMC protocol · arch→device by     │
│    gfxNNNN · ChipProfile.verify_live honesty guard · committed .hsaco/    │
│    chip-profile fixtures                                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2a. Measurement engine — the #488 instrument (LANDED, no-GPU-green)

Already built and green on this branch (88 lib tests). Six reusable components,
all NO-GPU except the two microbenches:

- **`ChipProfile`** (`crates/rdna-compute/src/chip_profile.rs`) — per-arch committed
  record fusing static caps (VGPR/SIMD, waves/SIMD, LDS/CU, wavefront) with measured
  fields (`peak_bw_gbps`, `cache_bw_gbps`, `effective_cache_mib`, `mem_latency_ns`)
  plus intrinsics + provenance. `load_committed` / `for_unprofiled` (withholds
  unmeasured fields as `None`, never a fabricated zero) / `verify_live` (fails loud
  on any static mismatch against a live `GpuCapability` — the honesty guard against
  a stale row after a driver/SKU swap) / `detect_bw_tiers` (separates the DRAM
  plateau from a faster cache-resident tier — the fix for the gfx1151 APU mislabel).
  All 5 RDNA gens committed at `crates/rdna-compute/tests/chip-profiles/*.json`
  (gfx1010/1030/1100/1151/1201), each `mem_latency_ns` **measured**, never estimated.
- **`roofline.rs` + `isa_histogram.rs`** — 3 independent `[0,1]` bound scores
  (Bandwidth / ValuIssue / Latency-by-Little's-Law), tightest wins, plus
  `second_wall_margin` and `trust_score`. ISA histogram shells out to the offline
  LLVM toolchain (`clang-offload-bundler`/`llvm-objdump`/`llvm-readelf`) against a
  `.hsaco`. Documented as a **diagnostic heuristic, not a certified predictor** — the
  static roofline *proposes*; the measured PMC (2b) *disposes*.
- **`kernel_ledger.rs`** — `LedgerKey = (arch, kernel, shape_bucket, quant, workload,
  phase)` full-tuple match; `diff()` emits **raw facts only** (per the layer rule:
  2a produces data, layer 5 interprets it): the static-field equality booleans
  (vgpr/sgpr/lds/scratch/isa_fingerprint changed vs unchanged), the dynamic-field
  signed %Δ against baseline, and whether a `coherence_stamp` is present. It does
  **not** emit the pass/fail *labels* (`RegressionHard` / `RegressionSoft` / `Gain` /
  `RejectedGainNoCoherence`) — those are the policy layer's job (§ layer 5), which
  applies the CLAUDE.md coherence rule to these raw facts. (Prior draft had `diff()`
  minting the labels in 2a; the review correctly flagged that as leaking policy into a
  data-only layer.) `module_collision_scan` is the static detector for the
  module-name-collision bug and likewise reports the raw collision fact, not a verdict.
- **`profile_rocprof.rs`** — rocprof CSV → coverage blindspots + achieved-BW +
  bound-class, all NO-GPU (parses a CSV a prior GPU step produced). Arch-keyed
  `trust_score` (0.6 on gfx12 historically, because `--pmc` read zero — **now
  upgradeable to a real counter-backed read via `profile_standard`; see below**).
- **Microbenches** (GPU-required): `pointer_chase_latency.rs` (true DRAM round-trip
  → `mem_latency_ns`) and `peak_bw_probe.rs` (D2D copy sweep → `detect_bw_tiers`).

**New work here (§b thread-3 → thread-2 join): the measured-PMC upgrade.**
`profile_rocprof.rs` currently *infers* a bound-class from the byte-model achieved-BW +
static roofline. With `profile_standard` unlocking `--pmc`, the engine emits a
**measured** ALU-busy% / mem-busy% / occupancy% read per kernel. The corpus commits
those **raw ratios**; the bound-class derived from them lives in the query-time
reasoning layer (§ Principle 1, layer 4) — the measured taxonomy (§e) is a *lens the
reasoning layer reads through*, not a field `profile_rocprof.rs` writes into the
corpus. The static roofline is likewise a **query-time proposer** (retained as the
NO-GPU fallback, `trust_score` 0.5) whose label is never committed; what 2a persists
is the ISA histogram it runs over.

### 2b. In-model data source — serve_harness (Phase A WIP)

The measure-and-record half. **Dumb, model-agnostic, no pass/fail, no baseline
comparison** — it only measures the **real daemon serve path**
(`/v1/chat/completions`), never bare `infer_*`. Phase A exists at 327 lines; modes
and record schema do not yet match the spec (known TODO, not a review defect).

- **Sampling presets:** `prod` (registry `recommended_settings` verbatim, **hard
  error** if the tag has none — the exact bug that produced the temp-0.3 27b run),
  `coding`, `nothink` (degeneration probe), `greedy` (determinism/parity only).
- **`--show-config` preflight:** resolves and prints every sampling value + its
  **source** (`registry`/`recipe`/`explicit`/`default`), the concrete think-cap, and
  WARNS when `max_tokens ≤ think_cap` — all without running. Implemented, working.
- **Modes:** `battery` / `multiturn` / `parity` (FP32 + `HIPFIRE_DETERMINISTIC=1`) /
  `agentic`. Genre battery `genre-v1` (target 8 prompts, md5'd; changing the set
  bumps `genre-v2` and invalidates baselines keyed on `batch_md5`).
- **Run record** (`schema_version 1`) — **two keys, not one** (the review's second
  load-bearing fix; the old single `run_hash` conflated logical config identity with
  physical run identity and could collide across commits):
  - **`behavior_key = md5(batch_md5 + sampling_values + kv + mtp + thinking +
    max_tokens + model_tag + model_file_md5)`** — the *logical experiment* identity.
    Two runs with the same `behavior_key` are "the same experiment" and are what the
    corpus staples together across measurement domains. It is a **join key, never a
    uniqueness key.**
  - **`run_id = md5(behavior_key + git_sha + daemon_md5 + arch + host + rocm_version +
    measurement_domain + started_at_nonce)`** — the *physical run* identity, unique
    per invocation. It cannot collide across commits (git_sha + daemon_md5), across
    boxes/archs (host + arch), or across profiler modes (measurement_domain). This is
    the corpus's row-uniqueness key.
  - Full provenance (all folded into `run_id`, all recorded raw on the run row):
    timestamp, host, `gpu.marketing` + `gpu.arch`, engine git_sha / daemon_md5 / rocm
    version, model path/md5/tag, config, sampling (values + source), mode, batch,
    per-turn array. Per-turn: `finish_reason` (**the runaway discriminator** —
    `length` where content was expected), prefill/decode tok/s, tau, ttft, tiered
    attractor, RUNAWAY/EMPTY/ATTRACTOR flags, answer preview.

- **Three measurement domains, modeled explicitly** (the review's fix for the unsafe
  cross-domain staple — a `high`-clock wall-time number must never be joined to a
  `profile_standard` PMC ratio via a shared identity). Each domain is its own `run_id`
  under a shared `behavior_key`; each carries a `measurement_domain` discriminator so a
  consumer can never mistake one for another:
  - **`serve_behavior`** — `high`/`auto` clocks. tok/s, τ, ttft, finish_reason,
    attractor. The behavior half; no profiler attached.
  - **`kernel_trace_walltime`** — `high`/`auto` clocks under `rocprofv3
    --kernel-trace`. The fired-kernel-set + per-kernel wall-time-% (the Amdahl
    contribution). Absolute-timing domain; **valid for wall-time, never for ratios.**
  - **`pmc_profile_standard`** — clocks pinned by `profile_standard` under `rocprofv3
    --pmc`. Per-kernel ALU-busy% / mem-busy% / occupancy% **ratios only** (clocks are
    low, so absolute timing from this domain is discarded).

**New work here: the harness must additionally run the `kernel_trace_walltime` and
`pmc_profile_standard` domain passes** (separate, gpu-locked invocations of the *same
battery*) to emit the fired-kernel-set + per-kernel counters. Today the harness only
runs `serve_behavior`. The three passes are **stapled by `behavior_key`, never by a
single `run_hash`** — the corpus joins the fired-set, the wall-time-%, and the ratios
for a kernel by `(behavior_key, kernel_module, shape_bucket)`, keeping each domain's
`run_id` + clock/profiler provenance intact so a low-clock ratio is structurally
incapable of being read as a high-clock latency.

### 3. Corpus DB — committed in-repo JSONL (new; the unification core)

Committed, versioned, structured, **in-repo** (travels with the code, diffs in
review). Append-only JSONL for raw runs + curated `baselines.json` / `ledger.json` /
`perfdrain.json`, with a thin `oracle-db query` CLI for agent-traceability. The
codebase-memory MCP **MAY index** it for convenience but is explicitly **NOT the
store** (it is blind to `examples/`, phase-2 accelerator only — per
`reference_codebase_memory_mcp_hipfire_probe`).

Four tables (see §e for the row schema that unifies the corpus grain):

- **`runs`** — every harness run record, raw, append-only, **keyed by `run_id`**
  (unique per physical run × domain), each carrying its `behavior_key` and
  `measurement_domain` so all three domain-passes of one experiment group under one
  `behavior_key`.
- **`kernels`** — the corpus proper: one row per `(behavior_key, measurement_domain,
  kernel_module, shape_bucket)`, carrying the domain's raw measured block + the
  domain-independent isolated (static/microbench) block, and the source `run_id` for
  provenance. The reasoning layer joins a kernel's `kernel_trace_walltime` row to its
  `pmc_profile_standard` row by `(behavior_key, kernel_module, shape_bucket)`. **This
  table is new and is the Oracle's reason to exist.**
- **`baselines` (golden)** — blessed per `(arch, model_tag, config_key)` where
  `config_key = (kv, mtp, thinking, sampling_preset, mode, batch_md5)`; stores
  `golden_behavior_keys[]` + the constituent `golden_run_ids[]`, median perf,
  coherence summary, `blessed_at`/`blessed_by`.
- **`ledger`** — `(arch, behavior_key)` coverage state, four states: `verified` /
  `unverified` (honest gap — path exists, no passing run) / `unimplemented` (arch
  doesn't support it) / `cant-validate-here` (panel-blessed, physically unrunnable,
  e.g. ds4 EP needing `--ep 4`). A generic-kernel PR writes `unverified`; a passing
  gate run flips it to `verified`.
- **`perfdrain`** — time-series of `(arch, model_tag, config_key) → median` per
  merge, for cumulative "death by 1000 cuts" drift (rolling-N > −5% unjustified →
  flag).

### 4. Reasoning layer — LLM / agent-panel (query-time only)

The only place tags are born, and they die when the query ends. Reads the corpus and
emits, per §III.2/§III.7 of the gate spec: the **required-arch set** (starts
`{gfx1100, gfx1151, gfx1201}`, narrowed only by ≥2-agent agreement + recorded
reason), the **arch-bleed verdict**, the **safety verdict** for opt-in
auto-validation, and — the Oracle's headline capability — **bound-class
interpretation and optimization advice reasoned from the measured ratios.** ≥2
independent agents (codex 5.5 + sonnet 5); human maintainer can override.

### 5. Policy layer — the gate (reads the DB, decides PR-readiness)

The only layer that says pass/fail. Perf guards (§below), coherence hard-fails, the
required-arch set, perfdrain flags. Never re-measures — it only reads what 2a/2b
wrote to the DB.

---

## (e) Data model — the corpus row

The schema is the spec's center of gravity: it is where **DATA-NOT-TAGS** and
**ISOLATED ≠ IN-MODEL** become concrete. One row = one kernel-module × **one
measurement domain** in one real serve experiment. Every committed field is a
**raw measured number or a raw static fact** — counters, ratios, timing, ISA
histogram, register/LDS facts, provenance. **There is no stored bound-class label
and no free-text semantic tag field** (the review's load-bearing fix; the prior draft
committed `roofline_static.bound_class` and `in_model.bound_class_measured`, which is
the exact DATA-NOT-TAGS violation). Bound-class is derived at query time by layer 4
(see the taxonomy below) and, if memoized at all, lives only in a non-committed cache.

Because the three measurement domains (§2b) must never be stapled inside one row,
each row carries a `measurement_domain` and its own `run_id`; the two GPU-measured
domains for one kernel are **joined at query time** by `(behavior_key, kernel_module,
shape_bucket)`. Below are the two sibling rows for one kernel — a
`pmc_profile_standard` (ratios) row shown in full, then its compact
`kernel_trace_walltime` sibling — sharing a `behavior_key`, never merged:

```jsonc
// table: kernels — append-only
// one row per (behavior_key × measurement_domain × kernel_module × shape_bucket)
{
  "schema_version": 1,
  "record": "hipfire.kernel_oracle.kernels.v1",

  // ---- identity (join key + physical-run provenance + domain) ----
  "behavior_key": "…",            // JOIN KEY (logical experiment) — staples the domains
  "run_id": "…",                  // unique physical run (git_sha+daemon_md5+arch+host+domain+nonce)
  "measurement_domain": "pmc_profile_standard",   // this row is RATIOS ONLY (clocks pinned low)
  "arch": "gfx1201",
  "kernel_module": "gemv_hfq4g256_moe_gate_up_k8_indexed",  // ELF .kd symbol, the real fired module
  "shape_bucket": "decode.b1.d2048",
  "quant": "hfq4g256",
  "workload": "moe.gate_up",
  "phase": "decode",              // decode | prefill | attention | glue
  "model_tag": "qwen3.6-35b-a3b.mq4r",
  "config_key": { "kv": "q8", "mtp": "off", "thinking": "med",
                  "sampling_preset": "greedy", "mode": "battery", "batch_md5": "…" },

  // ===== ISOLATED block (static / microbench — domain-independent, denormalized) =====
  // RAW facts only. The roofline is NOT stored here — layer 4 recomputes it at query
  // time from `isa_hist` + `chip_profile_ref` (its only inputs).
  "isolated": {
    "isa_fingerprint": 14872…,          // u64 hash of ISA-histogram shape
    "vgpr": 84, "sgpr": 48, "lds_bytes": 0, "scratch_bytes": 0, "spill": false,
    "occupancy_waves_theoretical": 8,
    "isa_hist": { "v_dot4": 0, "f32_fma": 90, "v_bfe": 121, "global_load_b128": 16,
                  "s_delay_alu": 34, "vmem_valu_ratio": 0.31 },
    "microbench_achieved_bw_gbps": null,   // WITHHELD until a real isolated bench exists
    "chip_profile_ref": "gfx1201"          // → ChipProfile row (peak_bw, mem_latency, …)
  },

  // ===== MEASURED block — RATIO domain (profile_standard, clocks pinned, ratios only) =====
  // NO bound-class field. Raw ALU/mem/occupancy ratios; layer 4 derives the class.
  "measured": {
    "alu_busy_pct":  { "mean": 45.6, "med": 43.2 },
    "mem_busy_pct":  { "mean": 56.3, "med": 47.9 },
    "occupancy_pct": { "mean": 54.6, "med": 46.4 },
    "calls_per_decode": 96
    // wall_time_* fields are ABSENT in this domain (clocks pinned → timing invalid)
  },

  // ---- reproducer + coherence (never a verdict, only provenance) ----
  "reproducer": { "cmd": "serve_harness --mode battery --preset greedy --domain pmc …",
                  "fixture_hsaco": "tests/kernel-fixtures/gfx1201/gemv_…hsaco",
                  "prompt_md5": "…" },
  "coherence_stamp": { "gate": "coherence-gate.sh", "passed": true, "at": "2026-07-02T…" }
}

// ── sibling row: same kernel, WALL-TIME domain (high clocks) — joined by behavior_key ──
{
  "schema_version": 1, "record": "hipfire.kernel_oracle.kernels.v1",
  "behavior_key": "…",            // SAME behavior_key → these two rows describe one kernel
  "run_id": "…",                  // DIFFERENT run_id (different domain, different pass)
  "measurement_domain": "kernel_trace_walltime",  // this row is WALL-TIME ONLY (high clocks)
  "arch": "gfx1201", "kernel_module": "gemv_hfq4g256_moe_gate_up_k8_indexed",
  "shape_bucket": "decode.b1.d2048",
  "isolated": { "…": "same static block, denormalized" },
  "measured": {
    "wall_time_pct_of_decode": 11.9,     // in-model contribution — THE Amdahl number
    "wall_time_ms_total": 42.3,
    "calls_per_decode": 96
    // alu/mem/occ ratios are ABSENT in this domain (high clocks → PMC read zero)
  },
  "reproducer": { "…": "…" }, "coherence_stamp": { "…": "…" }
}
```

**Why the raw numbers, never a resolved label.** The ISA histogram *proposed*
issue-bound (121 scalar unpacks `v_bfe`, 90 scalar `f32_fma` — looks VALU-issue). The
measured ratios showed what the chip *actually did* under a real forward pass: ALU
45.6% / mem 56.3% / occ 54.6% — mem highest, then occupancy, then ALU (`mem 56.3 >
occ 54.6 > alu 45.6`), **none saturated, all with headroom.** The corpus stores that
disagreement as raw numbers on two joinable rows and lets layer 4 resolve it freshly,
instead of freezing a `ValuIssue`-vs-`something` verdict into a field. It was exactly
this disagreement that **retired the inferred "gate_up is pure-DRAM-bound → dp4a will
help" story** and re-explained the empirical dp4a −1.52% daemon loss (a kernel already
carrying real ALU load, not idling on DRAM, should regress when you fold in more ALU).
No isolated microbench and no byte model produced that correction — the measured
in-model ratios did. (Note the prior draft's sample derivation was doubly wrong: it
labeled this `mixed-mem-leaning`, which is not one of the five taxonomy classes, and
its `argmax(mem 56.3 > alu 45.6 > occ 54.6)` mis-ordered occupancy below ALU. Deleting
the stored derivation deletes that whole class of transcription bug.)

**No baked tags.** There is no `"bottleneck": true`, no `"recommend": "dp4a"`, no
`"arch_bleed": …`, and no `"bound_class": …`. Those are all query-time outputs of
layer 4, computed from `wall_time_pct_of_decode` × the raw ALU/mem/occ ratios × the
ISA histogram, and they are never written back. The corpus stores what the chip did;
the reasoning layer says what it means, freshly, each time it is asked.

### The measured bound-class taxonomy (the layer-4 lens, NOT a stored field)

The five classes below are the **lens the reasoning layer (layer 4) reads the raw
ratios through** at query time. They are **not** a stored corpus field — the corpus
holds only `alu_busy_pct` / `mem_busy_pct` / `occupancy_pct` (+ wall-time-%), and the
class is derived on every query:

| class | measured signature (over the raw ratios) | tonight's A3B `mq4r` decode example |
|---|---|---|
| **bandwidth** | mem-busy high, ALU low, occupancy high | *(none clean this run — the inferred "BW-bound" verdict was retired)* |
| **VALU-issue** | ALU-busy high, mem moderate, occupancy high | — |
| **latency** | mem-busy high *in-window* but occupancy very low → chip idle waiting | `attention_flash_q8_0_tile`: mem 60.4% / occ 1.06% — starved, latency/launch-floor-gated |
| **occupancy / utilization** | ALU ≈ mem, both moderate, occupancy moderate (room left) | `gemv_..._moe_gate_up`: ALU 45.6 / mem 56.3 / occ 54.6 — mem > occ > alu, **none saturated → occupancy/utilization-bound, NOT bandwidth-bound** |
| **launch-overhead** | ALU ≈ 0, occupancy ≈ 0, small transient mem | `fused_rmsnorm_mq_rotate`: ALU 0.11% / occ 0.15% — textbook glue launch-overhead → fusion lever |

This table *is* the "isolated ≠ in-model, measure don't infer" lesson
institutionalized — but as a **reasoning lens over stored raw data**, not a stored
label: the corpus commits the ratios that let layer 4 land `gate_up` as
**occupancy/utilization-bound** (retiring the byte-model "bandwidth-bound" conclusion)
and `attention` as **occupancy-starved/latency-gated** (refining, not just confirming,
the "mem-bound" read). Full ground-truth in `docs/gfx1201-native-surface.md`
§"MEASURED bound-class".

---

## (f) Fleet substrate

Required-arch set = the existing validation triple `{gfx1100, gfx1151, gfx1201}`,
mirroring `feedback_gpu_validation_policy`. Per-box coordination via the existing
`scripts/gpu-lock.sh` (`flock`, kernel auto-releases on holder death — never `rm`
the lockfile).

| box | GPU(s) | Oracle role | sensitivity |
|---|---|---|---|
| **hipx** | gfx1151 (96GB Strix Halo, dev 1) **+ gfx1010 (5700XT, dev 0)** | **every-RDNA-gen fuzzer** — disposable, runs the widest battery incl. RDNA1 corpus | disposable |
| **hiptrx** | 4× gfx1201 (R9700, RDNA4) | gfx1201 measured bound-class + EP (`--ep 4`) `cant-validate-here` cells | blocking-validate |
| **k9lin** | gfx1100 (7900 XTX, RDNA3) | **SENSITIVE / OUT of the fuzz loop** — zero-validation host, gfx1100 corpus comes from elsewhere or read-only | sensitive → out |

**Arch → device selection is by the `gfxNNNN` line the bench prints, NOT by index**
(`HIP_VISIBLE_DEVICES` uses ROCR enumeration, not `rocm-smi` — per
`feedback_rocr_hip_visible_enumeration`; pin `HIP_VISIBLE_DEVICES=1` for gfx1151 WMMA
on hipx).

**`profile_standard` PMC protocol** (per-box, for the ratio-domain pass only):

```
# ratio pass ONLY — pins clocks low, ungates the perfmon counters
echo profile_standard | sudo tee /sys/class/drm/card<N>/device/power_dpm_force_performance_level
#   … run serve_harness under `rocprofv3 --pmc` …
echo auto           | sudo tee /sys/class/drm/card<N>/device/power_dpm_force_performance_level
```

The wall-time / tok/s pass runs separately on `auto`/`high`. `ChipProfile.verify_live`
runs at the top of every fleet session as the honesty guard: a committed row that no
longer matches the live chip (driver / firmware / SKU swap) fails loud rather than
silently poisoning the corpus.

---

## (g) Scope & phasing

Deliberately staged so **v1 stands alone** — the corpus + instrument is useful before
any gate exists.

### v1 — the corpus + instrument on the fleet (the standalone Oracle)
*Measure → store → query, across the fleet. No gate yet.* **v1 is gfx1201-complete
plus fired-set/wall-time on the other archs** — not a fleet-wide ratio-domain claim
(the review's scope fix; the PMC counter map is validated only on gfx1201, so v1 does
not pretend the measured bound-class exists everywhere on day one).

- Wire `serve_harness` to run the two GPU-measured domain passes (`kernel_trace_walltime`
  under `rocprofv3 --kernel-trace` on `high`/`auto`; `pmc_profile_standard` under
  `rocprofv3 --pmc` on `profile_standard`) and emit fired-kernel-set + per-kernel
  counters, keyed by `behavior_key` / `run_id` (§2b).
- Land the `kernels` corpus table + row schema (§e) and the `oracle-db query` CLI.
- **Per-arch domain scope (explicit, not aspirational):**
  - **gfx1201** — **full**: fired-set + wall-time-% **+ measured PMC bound-class
    ratios** (the counter map is validated here). This is the reference arch.
  - **gfx1100 / gfx1151 / gfx1010 / gfx1030** — **fired-set + wall-time-% only** in
    v1. The `pmc_profile_standard` domain stays **off** for these archs until each
    one's perfmon-counter map + gating lever is validated (Open Question 3); until
    then their `pmc_profile_standard` rows are simply not emitted (withheld, never a
    fabricated zero — mirroring `ChipProfile::for_unprofiled`). Each promotes to full
    the moment its counter-map validation lands — a per-arch step, not a config flip.
- Fleet bootstrap: run the battery over everything that fits each GPU
  (all models × `prod`+`nothink` × {AR,MTP} × {battery,multiturn}) to seed the
  fired-set + wall-time corpus fleet-wide, and the ratio corpus on gfx1201. **No
  baselines ⇒ no regression detection — this is the foundation, not garnish.**
- **Deliverable value on its own:** "for arch X model Y config Z, here is every
  kernel that fired, ranked by in-model wall-time-% (fleet-wide), with its measured
  bound-class ratios where the arch's counter map is validated (gfx1201 today)" — the
  omniperf-for-RDNA report, queryable by an agent. This alone already drove real
  decisions on this branch (the +4.24% `moe_down` R=2 multirow win shipped opt-in;
  the dp4a spike NO-GO'd by measurement) — proving the measurement layer works in
  anger, not just as scaffolding.

### v2 — the gate / CI policy layer
*Reads the v1 corpus → PR-readiness.*

- Golden baselines + ledger from the v1 bootstrap.
- Gate wrapper: **arch-bleed** (empirical, see §below), **perf guards** (table below),
  ledger transitions, PR-ready verdict.
- CI hook + agent-panel + opt-in codex auto-validation (safety-gated, GPU-time-capped,
  watchdog'd, `flock`-scoped to the gate run — worst case is one killed daemon).

**Perf guards (v2):**

| path | guard | action |
|---|---|---|
| AR decode AND prefill | median Δ ≤ **−3%** vs golden (warmed, median-of-N, byte-identical prompt — same `behavior_key`, `serve_behavior` domain) | **auto-rerun to confirm**; confirmed → hard block until golden restored |
| spec-decode (MTP/DFlash) | median Δ ≤ **−8%** | block — but the *primary* spec guard is coherence/attractor, not tok/s (τ variance ±~8%) |
| perfdrain | cumulative golden creep > **−5%** over rolling N merges, unjustified | flag for review |
| coherence (hard fail) | RUNAWAY (`finish_reason=length` where content expected) / EMPTY / ATTRACTOR / recall-floor | block |
| parity | byte-identical across affected archs under FP32 + `HIPFIRE_DETERMINISTIC=1` | block |

(−3% AR threshold rationale: CLAUDE.md pins single-run AR noise at ±1–3%, so a −3%
warmed median-of-N is real signal; the auto-rerun makes a false positive cheap.)

### v3 — autonomy / advisory (research-y — honestly labeled)

The corpus feeds an advisory loop: fuzz shapes, surface which kernels are
occupancy/issue-bound with headroom, propose the hand-roll. **This is research, not a
promise.** We do **NOT** commit to an auto-optimizer that writes kernels. The honest
v3 claim is: *"the corpus ranks where the wins are and predicts the bound-class flip;
a human (or a coherence-gated agent) does the hand-roll and the corpus confirms the
flip."*

### Honest about what is hard

- **Per-gen PMC counter maps.** The `profile_standard` unlock is validated on
  gfx1201. Each RDNA gen has a different perfmon counter set and a different
  perfmon-gating story; gfx1010 (RDNA1, no Infinity Cache) and gfx1151 (unified-memory
  APU) will each need their own gating validation and counter-name map. This is real
  per-arch work, not a config flip.
- **`profile_standard` teardown fragility.** Tonight's run saw GPU-teardown blocked
  within 8s under the reduced-clock state and per-dispatch PMC serialization slowing
  the profiled run vs. uninstrumented. The ratio pass is *slow and separate*; treat it
  as an offline batch job, not an interactive probe.
- **Empirical arch-bleed corpus is new.** It does not exist yet in any thread (see
  §below). The static dispatch-table map is the fallback if the empirical corpus
  proves too noisy across shape-fuzzing.
- **The advisory layer (v3) is speculative.** Do not scope v1/v2 as if v3 is
  guaranteed.

---

## Arch-bleed — static today, empirical is the new work (reconciling the threads)

**Flagged by the gate-design thread and load-bearing:** the "empirical kernel-trace
arch-bleed corpus" the mission implies **does not exist in either prior spec.** The
serve-gate spec's arch-bleed (§III.3) is **static** — grounded in three engine
artifacts:

1. the dispatch table (`hipfire-dispatch`: kernel = f(quant × arch × flags)),
2. arch-feature predicates (`rdna-compute`),
3. the per-arch kernel-module naming convention (a shared module name across archs =
   dead code — `reference_kernel_module_cache_collision`).

Rule: a change to an arch-gated path affects only that arch; a change to a shared path
affects all archs routing through it. This needs a machine-readable
`changed_path → affected_archs` map — **Open Question 1**, unresolved.

The Oracle **answers OQ1 empirically.** The `kernels` corpus already records, per
`(arch, model, config)`, the **exact set of kernel-modules that fired** (from
`rocprofv3 --kernel-trace`). Diffing the fired-module-set across arch cells for the
same model×config, before vs. after a diff, is a **measured** arch-bleed signal that
grounds or replaces the static derivation: if a diff to a "gfx1201-only" path changes
the fired-module-set on gfx1100, that is arch-bleed, measured, not inferred. The
static dispatch-table map remains the cheap NO-GPU first pass; the empirical
fired-set diff is the ground-truth confirmer. Note this is **genuinely new** —
existing infra (`coverage-audit.py`) checks *timing-coverage completeness* on a single
run ("did we forget a timer wrapper"), not a *comparative cross-arch fired-set corpus*.

---

## (h) Relationship to prior work

- **Supersedes / absorbs `docs/serve-harness-gate-spec.md`.** Its two-layer
  harness-vs-gate split, sampling presets, `--show-config` preflight, run-record,
  four DB tables, perf guards, agent-panel, and fleet-bootstrap phase all survive as
  layers 2b / 3 / 4 / 5 here — **except** the run-record's single `run_hash`, which
  this spec splits into `behavior_key` (join) + `run_id` (unique per physical run ×
  domain) per the cross-model review (§2b). The Phase-A/B/C/D/E phasing folds into
  v1/v2. The serve-gate spec's static arch-bleed becomes the NO-GPU first pass under
  the new empirical corpus.
- **Supersedes / absorbs `docs/research/2026-07-01-…perf-instrument-design.md` (#488).**
  Its six components become layer 2a verbatim; `kernel_ledger` becomes the corpus's
  static block + the diff engine — but the diff engine now emits **raw deltas /
  equality facts**, with the `RegressionHard/Soft/Gain/RejectedGainNoCoherence`
  *labels* applied in the policy layer (§ layer 5), not in 2a (the review's layering
  fix). `profile_rocprof` gains the `profile_standard` measured-PMC upgrade (raw
  ratios only; the bound-class it used to infer moves to query-time layer 4). The
  native-required manifest (`native_manifest.rs`, `Dp4aDecodeGemvCoverage`) stays a
  sibling Phase-A step, not part of the Oracle's measurement core.
- **Does NOT absorb `hipfire-atlas`.** We map `AtlasRow` **read-only** into the corpus
  (reuse the `.hip` kernel data + kernel-fire-set grammar), and we drop the crate's
  passive-corpus framing and the `kernel_atlas.py` tool as the store. Principle 4:
  reuse the data, drop the tool that nothing used.
- **Rebase target is `feat/mtp-config-default`** (the branch carrying the landed #488
  Phase A and the uncommitted serve-harness), not master — **but this worktree has
  diverged from it** (as of 2026-07-02: 11 commits on `feat/mtp-config-default` absent
  from HEAD, 41 on HEAD absent from it; merge-base ≠ tip, confirming the review's last
  point). The rebase/merge onto `feat/mtp-config-default` is a **prerequisite of any
  Oracle implementation work**, not a done deal — see Open Question 4.

---

## (i) Open questions

1. **`changed_path → affected_archs` map — three options now on the table.**
   Dispatch-table-derived at build time (cheap, NO-GPU, static) vs. explicit
   annotations (accurate, maintenance burden) vs. **the new empirical fired-module-set
   diff from the corpus** (ground-truth, GPU-cost, shape-fuzz-noise risk). The Oracle
   makes the third option possible; the open question is whether it *replaces* or only
   *ground-truths* the static map.

2. **~~How does `bound_class` stay DATA-NOT-TAGS in practice?~~ — RESOLVED by the
   cross-model review (2026-07-02) in favor of option (b).** The prior draft proposed
   option (c) (store the label + a `derivation` string + a `derived_by` version). The
   review refuted that: a stored label is a frozen tag no matter how it is annotated,
   and the sample derivation was itself internally inconsistent — proof that a stored
   derivation invites exactly the transcription rot it claims to prevent. **Decision:
   store only the raw ratios/counters/ISA-histogram/timing/provenance; derive the
   bound-class on every query in layer 4; optionally memoize in a *non-committed*
   render/query cache stamped with `derived_by`.** No `bound_class` field enters the
   canonical corpus (see §c Principle 1 and §e). This is now a design decision, not an
   open question — left here only as a resolved-record so a future reader sees why the
   schema has no label field.

3. **Per-gen PMC counter maps + perfmon-gating validation.** `profile_standard` is
   validated on gfx1201 only — which is why **v1's ratio domain is gfx1201-only** and
   the other archs ship fired-set + wall-time until validated (§g). What is the minimum
   viable counter-name map per gen (gfx1010/1030/1100/1151), and is `profile_standard`
   the right gating lever on the APU (gfx1151) and RDNA1 (gfx1010), or do those need a
   different perf-level? This gates each arch's promotion to the full ratio domain.

4. **Rebase/merge onto `feat/mtp-config-default` before treating this worktree as the
   integration base.** The worktree has diverged from the target (11 behind / 41
   ahead; merge-base ≠ tip — the review's final point, verified). Open sub-question:
   rebase HEAD onto `feat/mtp-config-default`, or merge `feat/mtp-config-default` into
   HEAD, or land the Oracle work as a fresh branch cut from `feat/mtp-config-default`?
   This must be settled before implementation, independent of the design.

*(Carried from the serve-gate spec, still open: DB substrate choice —
in-repo-JSONL vs sqlite vs MCP-primary; `agentic` mode sampling; golden re-bless
policy vs. perfdrain false-flag; panel tie-break when codex 5.5 and sonnet 5
disagree on required-archs/safety.)*
