# Oracle Genesis — Bill of Debt (Phase 0), and the CI/Validation Roadmap

**Status:** design spec (Phase 0 detailed; Phases 1–2 sketched)
**Date:** 2026-07-02
**Branch:** `feat/rdna-kernel-oracle` (stacked on `feat/mtp-config-default`)
**Supersedes framing in:** `docs/superpowers/specs/2026-07-02-kernel-oracle-design.md` (the
gfx1201-centric "CI target" framing is retired here — see §1).

---

## 1. Mission & framing

hipfire's mission is absmax performance on **all** RDNA, a lineage AMD has treated as
disposable. The Oracle's job is to make that measurable and enforceable.

**The CI/Oracle target is `hipx`-as-a-box — all five RDNA arches
(gfx1010 / gfx1030 / gfx1100 / gfx1151 / gfx1201) as equal, first-class citizens.**
gfx1201 was the target of the #488 *kernel campaign* only; it holds no special status here.
Every arch is measured, every arch is validated, every arch gets a seat.

Two invariants govern everything downstream:

- **Parity.** The mission is to drive all arches *toward even* — no arch left as a
  second-class fallback. The bill of debt (§4d) makes "how far from even" a number.
- **No arch may clobber another.** This is the anti-clobber principle promoted from
  files/PRs up to **arches themselves**: a change that advances one arch by regressing
  another is a *cross-arch clobber*, and CI's job (Phase 2) is to make it visible and block it.

### The three-phase pipeline — CI is the culmination, not the start

The anti-clobber CI is only *authoritative* once it is backed by real per-arch data. So it
comes last. Each phase empirically feeds the next:

| Phase | Name | What it produces | Status |
|---|---|---|---|
| **0** | **Genesis Oracle / Bill of Debt** | Hardened instrument, measuring through the real serve path on hipx across all 5 arches, prints an honest empirical per-arch bill of debt. | **This spec** |
| **1** | **Arch-gated Autoresearch** | The bill's WITHHELD + highest-in-model-debt rows become a work queue; arch-gated agentic swarms auto-research/build/optimize/certify optimal kernels per arch, validated by the serve_harness ∩ oracle *generative validation oracle*. | Sketch (§6); own spec later |
| **2** | **CI / Stamp-of-Approval / Anti-clobber** | With authoritative per-model × per-arch native knowledge in hand, the anti-clobber CI (provenance stamp, tiered agentic review, cross-arch no-clobber gate) has real ground to gate against. | Sketch (§7); absorbs `docs/serve-harness-gate-spec.md` |

---

## 2. Phase 0 north star & principles

**North star:** an accurate, honest, empirical **per-arch bill of debt** across
gfx1010/1030/1100/1151/1201 — measured through the *real serve path* on hipx, for **every
model that fits each arch's VRAM.**

Principles (non-negotiable, inherited and sharpened):

- **Live measurement is the northstar.** Every number comes from a fresh run on the target
  arch, never a stale-corpus lookup. The corpus is the committed *sediment* of measurements
  — a regression baseline and a targeting aid, never a substitute for measuring.
- **Withhold, never fake.** An uncorroborated cell emits `None`/WITHHELD and contributes
  nothing to a verdict, rather than shipping an analytic guess dressed as a measurement.
- **Withheld is leverage, not cosmetic.** A WITHHELD cell is *unknown debt, flagged as a
  target* — it is where we cannot yet see, which is exactly where to point the next phase.
  The withheld set is a first-class part of the work queue.
- **Per-arch symmetric.** Promotion is per-cell, per-arch. A cell clearing on gfx1100 does
  **not** promote gfx1201 — no arch-invariance transfer, ever.
- **serve_harness ∩ oracle closes the loop.** Dynamic measurement routes through the real
  serve path, so *the kernels measured are the kernels the user hits.* This kills the
  isolated ≠ in-model gap (a microbench-efficient kernel that is 0.1% in-model, or a
  hidden in-model bottleneck).
- **Reference Atlas, never trust it.** Atlas kernel-fire-sets may seed the static targeting
  map as a weak prior; the empirical kernel-trace overrides it; regression baselines come
  only from our own measured rows.

---

## 3. Why instrument-hardening comes first

A cold-start adversarial review (7-agent workflow, reading the real grafted instrument,
cross-verified) established that today the instrument's **static-shape data is trustworthy**
but its **dynamic bound-class / latency / in-model verdicts are not** — on any arch. An
inaccurate or gfx1201-only tool cannot print a trustworthy all-arch bill of debt. So Phase 0
*starts* by hardening the tool until each cell can be honestly promoted. §5 embeds the full
cold-start trust model (cross-check matrix, risk register, promotion gate).

---

## 4. Phase 0 design

### 4a. Instrument accuracy hardening (the audit fix-list)

Confirmed by the adversarial audit (§5.3). The two HIGH items are the two hardware constants
the latency roofline rests on; they push the latency bound-class in *opposite* directions, so
**any latency-bound verdict is untrustworthy in magnitude and sign until both land.**

- **HIGH — `mem_latency_ns` is cache-deflated on every Infinity-Cache part.** The fixed
  128 MiB pointer-chase buffer fits inside the IC (gfx1030 fully; gfx1100/1201 cleared by
  only 1.3×/1.9×). Committed latencies track cache *size*, not memory tech. **Fix:** size the
  buffer per-arch to ≥ 4×(L2+IC) from `arch_spec` (≥512 MiB on RDNA2); add a cache-residency
  guard; re-measure gfx1030/1100/1201; stop shipping cache-deflated rows as clean.
  *[[#490](https://github.com/Kaden-Schutt/hipfire/issues/490)]*
- **HIGH — `latency_score` drops `simds_per_cu` (=2 on all RDNA) → a 2× error** biased toward
  Latency. **Fix:** add `simds_per_cu` to `ChipProfile`, multiply it into `available_bytes`,
  add a numeric `latency_score` regression test. *[[#491](https://github.com/Kaden-Schutt/hipfire/issues/491)]*
- **MED — round-up:** derive `bound_class` at query time (don't persist the tag; make `diff`
  compare it); propagate `AchievedBw.trust_score` into `Roofline` (reserve 1.0 for a live PMC
  read, not "BW-axis has a value"); fix cache-tier working-set accounting (report cache size
  in working-set bytes; commit the raw curve; assert working-set ≤ L2+IC); resolve sysfs
  reads by HIP ordinal so `verify_live` cross-checks the *pinned* card on a zoo box; stamp DPM
  state (`power_dpm_force_performance_level`, sclk/mclk) on every `AchievedBw`/`LedgerRow` and
  refuse/flag timings captured under `profile_standard`; sum all VMEM load widths in the
  latency short-circuit (not just `global_load_b128`); word-boundary coverage matching; read
  `.vgpr_count` from the msgpack note (cross-check the bit-decode, fail loud on disagreement).

### 4b. Close the in-model loop (serve_harness ∩ oracle)

The audit found the in-model *importance* axis — the half that answers "is this kernel a real
lever" — parsed then thrown away. Fix by routing measurement through the real serve path:

- **Measure through the daemon serve path**, so the measured kernels *are* the served kernels.
- **Surface in-model wall-time-%** with its true denominator (stop dropping rocprof's
  `Percentage`; stop dropping unmatched rows so the 100% denominator reconstructs).
- **Wire the blindspot detector** (`compute_coverage`/`stop_with_rocprof`) into the dynamic
  entry point — it currently has zero callers outside its tests.
- **Record + enforce in-model provenance** (`Reproducer.cmd` / `prompt_md5`): a bill-of-debt
  row must *prove* it came from real serving, not a standalone microbench.

### 4c. Per-arch promotion / census

- **Per-arch PMC census.** Enumerate, on *each* arch, exactly which counters read non-zero
  under `profile_standard` — specifically whether the *derived* metrics the roofline needs
  (`FetchSize`, `WriteSize`, `VALUBusy`, `MemUnitBusy`, occupancy) revive, or only *raw
  accumulators* (`SQ_BUSY_CYCLES`, `Wavefronts`). gfx1100 (RDNA3) likely works; gfx1010/1030/
  1151 unknown; gfx1201 in doubt. Each arch's dynamic domains promote or stay WITHHELD on its
  **own** evidence.
- **Per-cell, per-arch promotion gate** (§5.5): a cell is trusted-for-verdict iff ≥2
  mutually-independent ROCm-grounded sources agree within tolerance on that arch **and**
  intended in-model behavior is verified **and** no unmitigated HIGH risk touches it.
  Otherwise WITHHELD. Never faked; no arch-invariance transfer.

### 4d. The bill of debt output

For each arch, an itemized ledger of what it "owes" to reach absmax, one row per
**(arch × fitting-model × kernel × domain)**:

- **measured** → %-of-roofline, in-model wall-time-%, recoverable time (the debt magnitude);
- **WITHHELD** → unknown, flagged = a leverage / Phase-1 autoresearch target (*not* zero);
- **structural** → no native kernel on this arch, generic fallback (debt = fallback penalty).

Aggregations:
- **Ranked by in-model lever size** — biggest recoverable *real* time first, *not* microbench
  efficiency (`pct_of_peak` is an efficiency axis, orthogonal to importance).
- **Per-arch total debt** + a **cross-arch unevenness score** (the spread across arches =
  distance from parity).
- **No-arch-clobber invariant** (used by Phase 2): a change's per-arch debt *delta* must be
  ≤ 0 for every arch, else it is flagged — "improved gfx1201 +5%, regressed gfx1010 −8%" is a
  cross-arch clobber.

Committed as JSONL (raw fields only; `bound_class` and debt magnitudes derived at query, never
stored as frozen tags — data-not-tags). This committed ledger is the genesis baseline Phase 1
launches from.

### 4e. Execution on hipx

One self-hosted hipx runner. Matrix over all 5 arches; each pinned via `HIP_VISIBLE_DEVICES`
+ a per-card `HIPFIRE_GPU_LOCKFILE`. hipx drives ~one GPU at load at a time (power/thermal),
so the matrix executes **serially under the locks** — logical fan-out, physically serialized.
Two DPM passes per cell: kernel-trace wall-time + tok/s on `high`/`auto`; PMC ratios on
`profile_standard` (ratios only — it pins clocks low, so tok/s never comes from that pass).

---

## 5. Cold-start defense-in-depth (the Phase 0 trust model)

> This section is the grounded output of the 7-agent adversarial cross-validation workflow
> (`wf_8437a146`), reframed to the all-arch target: the promotion table below is **five equal
> columns**, not "gfx1201 + others."

### 5.1 Framing — who validates the validator

The RDNA instrument stands in for a tool that does not exist on our fleet
(`rocprof-compute`/omniperf is CDNA/Instinct-only and refuses to run on any RDNA arch). That
absence is its reason to exist — and the reason it cannot be its own northstar. Every headline
it emits (achieved-BW, %-of-peak, bound-class, occupancy, latency) is an analytic byte-model or
a static ISA heuristic, not a hardware counter read. Before a number may issue or gate a PR
verdict, two independent questions must be answered *per arch*: (a) **external correctness** —
does it agree with an *independent* ROCm ground-truth source, on the *target* arch, within a
stated tolerance? (b) **intended in-model behavior** — does the tool actually surface what the
design claims (in-model wall-time-%, isolated-vs-in-model separation, blindspot detection)?
A number that passes neither is a *hypothesis*, not a verdict.

### 5.2 Cross-check matrix — independent ROCm ground truth

Only cross-checks that survived the adversarial independence audit (different method **and**
on-target **and** measuring the claimed quantity) appear here.

| Our output | Independent ROCm cross-check | Tolerance | Independence |
|---|---|---|---|
| `peak_bw_gbps` (DRAM-plateau) | `rocm-bandwidth-test` DtoD unidirectional, warmed at DPM `high` | ±15% (SDMA copy engine vs float4 STREAM compute) | Independent, on-target (separate binary + engine, same GDDR6 bus). Both must exceed the IC to reach DRAM — a shared, *detectable* residency assumption. |
| Per-kernel `VGPR/SGPR/LDS/scratch` + spill | `llvm-readelf --notes` (`amdhsa.kernels` msgpack); `llvm-objdump ; Num*` as a third read | Exact (VGPR granule-rounded to 8) | Independent decoders of a shared code object (our `pgm_rsrc1` bit-extraction vs the compiler's own note). |
| Static arch constants (`vgprs_per_simd`, `max_waves_per_simd`, `lds_bytes_per_cu`, `wavefront_size`) | `rocminfo` HSA/KFD topology | Exact | Independent for the compile-time literals `verify_live` cannot self-check. **Carve-out:** `cu_count`/`vram` are live sysfs reads sharing the amdgpu/KFD driver with rocminfo — redundant-with-self, not independent. |

**Corroboration gaps** (proposed check exists but was refuted as non-independent — each needs
a second, genuinely independent method before its domain can promote):

- **(A) Same counter / same source** — timing vs `--hip-trace` (same ROCr dispatch
  timestamps); coverage vs a second `--kernel-trace` (rocprof is its own ground truth); ISA
  counts vs `llvm-objdump | grep` (shared disassembly); `theoretical_peak` vs
  `rocm-smi --showmemclk` (shared sysfs + same 16× multiplier). Catches parse bugs only.
- **(B) Off-target + arch-invariance** — `achieved_gbps`/`pct_of_peak`/`bound_class`/
  `occupancy` vs PMC `FetchSize`/`WriteSize`/`VALUBusy`/`MemUnitBusy`/`Wavefronts`, which read
  0.0 on gfx1201 and only run on gfx1100. Transferring gfx1100 → gfx1201 assumes RDNA3=RDNA4
  for exactly the ALU-vs-mem split in question — invalid. *Needs:* on-target PMC (the §4c
  census) or an end-to-end tok/s A/B that does not depend on the gated counter.
- **(C) Quantity / method mismatch** — `cache_bw`/`effective_cache_mib` vs `TCC_HIT/MISS`
  (hit-rate is not GB/s); `mem_latency_ns` vs re-timing the same chain (cannot prove DRAM
  residency). **Cache BW/size and DRAM latency are structurally un-cross-checkable in
  first-party ROCm** — they must stay WITHHELD as magnitudes (confirmable only as "this row was
  cache-resident"), never a CI verdict.

**Pivotal per-arch open question — does `profile_standard` revive the *derived* metrics?** The
survey saw 169k non-zero samples on gfx1201 under `profile_standard`, but the independence
refutation holds those may be *raw accumulators* while `FetchSize/WriteSize/VALUBusy/
MemUnitBusy` still read 0.0. **Cold-start action (per arch): a per-counter census.** If the
derived metrics come alive, archetype (B) becomes on-target-independent and that arch's dynamic
domains graduate; if only accumulators do, dynamic stays WITHHELD for that arch. (Caveats:
`profile_standard` pins clocks LOW — ratios only, keep tok/s on `high`/`auto`; ≤2 counters/pass
over ≤2 HW blocks; drive the daemon directly and SIGKILL the child.)

### 5.3 Adversarial audit risk register (verify-confirmed)

Twelve of thirteen audited risks survived a source-grounded verify pass (one — a `peak_bw`
cache-reuse inflation — was refuted: `detect_bw_tiers` reports the 512 MiB swept size a 64 MB
cache cannot inflate). Most severe first:

| Sev | Risk | Domain | Mitigation |
|---|---|---|---|
| **HIGH** | `mem_latency_ns` cache-deflated (128 MiB buffer ≤ IC; latencies track cache size) | latency roofline | per-arch buffer ≥4×(L2+IC); residency guard; re-measure; §4a |
| **HIGH** | `latency_score` drops `simds_per_cu` → 2× error, biased to Latency | bound_class (latency) | add `simds_per_cu`; numeric regression test; §4a |
| MED | committed `bound_class` persisted as a bare tag; `diff` ignores it; `from_fixture` defaults ValuIssue | bound_class corpus | derive at query / tag with the trust it was derived under; make `diff` compare it |
| MED | `roofline.trust_score` hardcodes 1.0 when `achieved_bw.is_some()`, though it's analytic (0.6/0.8) | trust semantics | propagate `AchievedBw.trust_score` (min); reserve 1.0 for live PMC |
| MED | cache-tier physically inconsistent (`effective_cache_mib=64` vs a 128 MiB working set) | cache_bw / size | report working-set bytes; commit raw curve; assert ws ≤ L2+IC |
| MED | sysfs readers ignore `HIP_VISIBLE_DEVICES` → wrong card on the zoo box | cu_count/vram/peak | resolve KFD/DRM node from the HIP ordinal; source static fields from rocminfo on the pinned device |
| MED | `achieved_bw` carries no DPM provenance → a `profile_standard` CSV is clock-inflated | achieved_bw/pct_of_peak | stamp DPM state; flag/refuse `profile_standard` timings |
| MED | `latency_score` short-circuits on `global_load_b128==0`; misses other VMEM load widths | bound_class (latency) | sum all `buffer_/flat_/global_/scratch_ × b32/b64/b96/b128` by byte width |
| LOW | substring matching inflates `coverage_pct` (short alias matches many mangled names) | coverage/blindspots | word/anchor-boundary matching; min alias length |
| LOW | `decode_vgprs` hardcodes granularity ×8 instead of reading `.vgpr_count` | per-kernel VGPR | read the msgpack note; cross-check + fail loud |
| LOW | `achieved_gbps` divides ideal weight-only bytes by measured time, labeled "achieved BW" | achieved_bw | relabel `ideal_bytes_per_time`; document as a lower bound; cross-check vs `FetchSize` where obtainable |

### 5.4 Intended-behavior review

**Verified:** roofline *denominators* (`peak_bw` DRAM-plateau, `mem_latency`) are genuine
isolated chip microbenches, WITHHELD when unmeasured; per-kernel throughput is honestly labeled
analytic-byte-model ÷ measured-time (never a live counter); `bound_class` is derived per query.
**Gaps (the in-model importance axis is not surfaced):** rocprof `Percentage` is parsed into
`RocprofKernel.percent` then thrown away (`AchievedBw` has no percent field, `run_dynamic` never
emits it, unmatched rows dropped) — only absolute `total_us` survives, so a 0.1%-of-decode and a
40%-of-decode kernel look comparable; the blindspot detector has zero callers outside tests;
in-model provenance is neither recorded nor enforced (a microbench CSV flows through the identical
path). These are §4b.

### 5.5 Per-arch promotion gate

> A cell (arch × measurement-domain) is **PROMOTED** to trusted-for-CI-verdict iff **all
> three** hold: (a) ≥2 mutually-independent ROCm-grounded sources agree within tolerance *on
> that arch*; (b) intended in-model behavior is verified for that domain; (c) no unmitigated
> HIGH audit risk touches it. Otherwise **WITHHELD** — emits `None`, contributes nothing.
> **Withheld, never faked.** Per-cell, per-arch, incremental; no arch-invariance transfer.

| Measurement domain | Posture (same rule, all 5 arches) | Blocking condition |
|---|---|---|
| Static arch constants (VGPR-file, max_waves, LDS, wavefront) | **PROMOTED** | Independent (rocminfo) + fixed literal + `verify_live`; no HIGH risk |
| `cu_count` / `vram` | PROMOTED single-GPU; **WITHHELD on zoo box** | Risk 6 (sysfs ignores HIP ordinal) until node resolves from the ordinal |
| Per-kernel VGPR/SGPR/LDS/scratch | **PROMOTED** | Two independent decoders; land msgpack read to retire Risk 10 |
| `peak_bw_gbps` | **BLOCKED-ON-WIRING** | 2nd source (`rocm-bandwidth-test`) on-target but not yet ingested; promote on ±15% agreement |
| `cache_bw` / `effective_cache_mib` | **WITHHELD (no source)** | No first-party ROCm tool (archetype C) + Risk 5; assert residency only |
| `mem_latency_ns` | **WITHHELD** (clean only on gfx1010, no IC) | No independent ROCm latency tool + two HIGH risks; re-measure hypothesis, never a verdict |
| `achieved_bw` / `pct_of_peak` (dynamic) | **WITHHELD** until per-arch census + wiring | archetype B — gated on the §4c census + DPM provenance |
| `bound_class` verdict (dynamic) | **WITHHELD** until census + latency fixes | rests on ISA + a broken latency axis until §4a lands |
| in-model wall-time-% / coverage / blindspots | **WITHHELD (behavior)** on every arch | criterion (b) fails everywhere until §4b lands — the foundational cold-start deliverable |

**Net cold-start posture:** only the static-shape domains promote today; `peak_bw` promotes the
moment `rocm-bandwidth-test` is wired; every dynamic roofline output is WITHHELD on every arch
pending the per-arch census, the latency fixes, and the in-model-importance wiring. The gate is a
**standing precondition**, re-run on any firmware/ROCm/SKU/kernel-shape change; any cell can
**demote** if a cross-check drifts. The tool graduates from "untrusted instrument" to "trusted
gatekeeper" **cell by cell, arch by arch** — never by asserting what the fleet's tools cannot
corroborate.

---

## 6. Phase 1 sketch — Arch-gated Autoresearch (design later)

Downstream of the bill of debt; its own spec. Key properties fixed now:

- **Input = the bill of debt's work queue:** WITHHELD cells (unknown debt — make it visible) and
  highest-in-model-lever measured cells (biggest recoverable real time).
- **Arch-gated agentic swarms** auto-research → build → optimize → certify optimal kernel designs
  per arch (a kernel win on gfx1100 does not assume gfx1201; each arch's swarm is gated to its own
  ISA/roofline).
- **Validated by the serve_harness ∩ oracle generative validation oracle** — the *opposite* of the
  historical pass/fail coherence/perf gates: a full-surface real-serving oracle. Any model that
  fits the arch's VRAM is fair game; every candidate is exercised on the real serve path with full
  instrumentation: coherence, perf, TTFT, multiturn, spec-decode, long-context, prefix-caching. The
  kernels measured are the kernels the user hits — the loop closes empirically.
- Runs on hipx.

## 7. Phase 2 sketch — CI / Stamp-of-Approval / Anti-clobber (culmination)

After autoresearch certifies per arch, we hold authoritative per-model × per-arch native
knowledge. Now the anti-clobber CI has real ground to gate against. Absorbs the
`serve-harness-gate` design. Components (from the earlier design dialogue):

- **Provenance stamp:** serve_harness emits `{arch_id, timestamp, content_hash}`; a PR carries it,
  or hipx CI auto-generates it by benching the change (targeted to what changed) and returning an
  agent-panel verdict as a PR comment.
- **Tiered agentic review:** cheap diff panel (Codex + Claude) on every PR for arch-bleed /
  state-bleed / silent-no-op / perf-clobber patterns; GPU behavior review on hipx gated to hotspot
  PRs.
- **Clobber gates:** cross-PR file-collision (advisory merge-order), semantic/behavioral clobber
  (hard fail), perf clobber (AR −3% hard fail), local agent-race discipline (worktree isolation +
  return-diffs).
- **Cross-arch no-clobber invariant:** a PR's per-arch bill-of-debt delta must not worsen any arch.
- **Targeting:** static dispatch/registry map bootstrap → empirical fire-sets as the corpus
  accumulates; measurement always live.

---

## 8. Open questions

- **Q0.1 — Per-arch census outcomes.** Which arches' derived PMC metrics revive under
  `profile_standard`? Gates which dynamic domains can *ever* promote per arch. (§4c/§5.2.)
- **Q0.2 — Bill-of-debt "recoverable" estimator.** For a measured cell, how is *recoverable* time
  computed — roofline headroom, or a more conservative bound? Must not itself become an un-validated
  analytic guess.
- **Q0.3 — VRAM-fit enumeration.** How is "every model that fits arch X's VRAM" enumerated and
  bounded (registry-driven; skip OOM; long-ctx KV budget)?
- **Q0.4 — Unevenness metric.** Definition of the cross-arch parity score (variance of normalized
  debt? worst-arch gap?) — decides what "toward even" optimizes.
- **Q0.5 — Corpus substrate.** JSONL (git-diffable) vs sqlite; how the bill of debt is queried by
  humans and by Phase-1 swarms.

## 9. Sequencing & deliverables (Phase 0)

1. **0a — accuracy hardening.** Land the two HIGH fixes (file both as issues first), then the MED
   round-up. Unblocks trustworthy measurement; no latency-dependent domain promotes until the two
   HIGH land.
2. **0c — per-arch census.** Wire `rocm-bandwidth-test` ingestion (promotes `peak_bw`) and run the
   per-arch `profile_standard` counter census (decides dynamic-domain promotability per arch).
3. **0b — close the in-model loop.** Route measurement through serve, surface in-model wall-time-%,
   wire the blindspot detector, enforce provenance.
4. **0d — the bill of debt.** Emit + commit the per-arch ledger with the unevenness score and the
   no-arch-clobber delta; withheld/high-debt rows become the Phase-1 queue.
5. **0e — hipx execution.** Drive the all-5-arch matrix under per-card locks.

**Definition of done (Phase 0):** a committed, honest, per-arch bill of debt across all 5 RDNA
arches, every cell either PROMOTED (independently corroborated) or WITHHELD-with-reason, measured
through the real serve path on hipx — the empirical genesis baseline for autoresearch.
