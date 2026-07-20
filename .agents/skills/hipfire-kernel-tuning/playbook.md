# Kernel-tuning playbook

Actionable loop for a single lever on a known-hot kernel. Each step must
clear before the next. Skipping steps is how silent corruption and fake
wins reach master.

Mutable thresholds, noise bands, and route tables live in owners linked
below — do not paste them here.

## 0. Preconditions

- Target **arch + quant + phase** (prefill / decode_ar / decode_dflash) named.
- Hardware available for every arch you intend to **enable** in public
  dispatch. No hardware → land kernel + channel test only; leave dispatch
  flip for a measured follow-up (`cross-arch.md`).
- Worktree identity recorded (commit or dirty `diff_md5`) before any timed run.
- Read [`docs/VALIDATION.md`](../../../docs/VALIDATION.md) claim→route map
  for the surface you will touch.

## 1. Profile first

Do not optimize from intuition.

```bash
# Internal timers (wired through dispatched kernels)
HIPFIRE_PROFILE=1 HIPFIRE_DPM_WARMUP_SECS=10 \
  hipfire bench <model> --runs 5 2>&1 | tee bench.profile.log

# Optional: phase-aware Atlas row + ISA + dispatch provenance
# See docs/methodology/kernel-atlas.md and hipfire-kernel-atlas skill.
python3 scripts/kernel_atlas.py collect-ar \
  --model <path> --workload <id> --quant <q> \
  --prefill <N> --gen <N> \
  --profile-prefill --profile-decode \
  --isa-dir .hipfire_kernels --isa-filter '<hot_kernel_stem>' \
  --dispatch-provenance \
  --output .codeinsight+research/kernel-atlas/runs/tune-$(date -u +%Y%m%dT%H%M%SZ).jsonl
```

Read the profile for **class**, not just rank:

| Signal | Likely class | Lever family (see `levers.md`) |
|---|---|---|
| High µs, low GB/s | Latency / occupancy / launch | Occupancy, multi-row, fuse launches, barrier-free |
| High GB/s near peak | Bandwidth-saturated | Algorithm change or less traffic; micro-ISA rarely wins |
| Low %-of-cycle | Not the bottleneck | Stop; re-profile end-to-end |
| High launch count | Launch overhead | Fused projections, multi-row, batch across forward |
| Matrix engine idle on prefill GEMM | Wrong path or shape | WMMA/MFMA route (capability-gated) |
| Spills / high VGPR in ISA | Register pressure | Smaller tile, less multi-row R, simpler live ranges |

Reconcile internal timers with **rocprof** when wall time and profile
disagree. Use the canonical route in
[`docs/methodology/rocprof-coverage.md`](../../../docs/methodology/rocprof-coverage.md):
`scripts/rocprof-wrap.sh` + `scripts/coverage-audit.py`. Do not blame
"launch overhead" until the trace shows it. Atlas ISA notes are for
codegen/resource analysis only — they cannot expose missing kernel time.

## 2. Root-cause (source + ISA)

Before picking a lever:

1. **Source** — open the `.hip` Atlas/dispatch attributes to the hot symbol.
   Confirm the runtime path actually loads that file (chip > family > base
   tags in `scripts/compile-kernels.sh`; see `cross-arch.md`).
2. **ISA** — VGPR/SGPR, LDS, private/scratch, spills, wave size, matrix-op
   mix. Prefer Atlas ISA manifests or `llvm-readobj` / `llvm-objdump` on the
   HSACO under `.hipfire_kernels/`.
3. **Dispatch** — capability gate vs perf sub-variant. Capability predicates
   (`has_wmma_w32`, `has_wmma_w32_gfx12`, `is_rdna4`, …) answer "is the ISA
   legal?". Perf variants (`plain` / `ldscoop` / `nosync` / `k4` / multi-row R)
   need a **measured arch allowlist** and a conservative default
   ([`perf-arch-discipline.md`](../../../docs/methodology/perf-arch-discipline.md)).
4. **Adjacent archs** — list every chip that shares the file tag or
   predicate you will touch. Those are your regression surface.

Common root causes: wave-size mismatch (CDNA wave64 vs RDNA wave32), VGPR
spills, LDS bank conflicts, uncoalesced global access, multi-wave
`__syncthreads` serialization under low BW%, wrong builtin family on gfx12
vs gfx11, perf variant inherited via a widened capability predicate.

## 3. One lever

Open [`levers.md`](levers.md). Choose **exactly one** lever that matches the
diagnosed class. Write the hypothesis in one sentence:

> On `<arch>` / `<kernel>` / `<phase+shape>`, `<lever>` should improve
> `<metric>` because `<mechanism>`; risk is `<spill|correctness|adjacent>`.

If the lever already has a **documented rejection** in `levers.md` § negative
results or `case-studies.md`, do not re-run it without a *new* mechanism.

## 4. Implement with arch boundaries

- Prefer chip tag (`*.gfx1201.hip`) or family tag (`*.gfx12.hip`) over editing
  the portable base when the win is arch-local.
- Register source via existing `include_str!` / kernel tables in
  `crates/rdna-compute` (or the owning arch crate for crate-local kernels).
- Dispatch: fast correct path first, portable baseline last; **no unreachable
  branches** when narrowing predicates (`cross-arch.md`).
- Perf sub-variant: explicit arch allowlist + portable default — never
  `if is_rdna3() { tuned }` as a perf key.

Compile **concrete gfx atoms** before timing. `compile-kernels.sh` passes
each argument to `hipcc --offload-arch`; family tags such as `gfx12` are
**not** valid compile targets. A `*.gfx12.hip` (or other family-tag) edit
must be compiled for **every affected concrete chip**:

```bash
# Example: family-tag edit covering RDNA4 chips
./scripts/compile-kernels.sh gfx1200 gfx1201
# Chip-local override
./scripts/compile-kernels.sh gfx1201
```

## 4b. Candidate ISA inspect (post-compile, pre-measure)

After the candidate builds, inspect its ISA **before** correctness timing:

1. Confirm the intended symbol / instruction mix (WMMA/MFMA/dot2/scalar,
   wave size, barrier presence) matches the lever hypothesis.
2. Compare VGPR/SGPR, LDS, private/scratch, spills, and wave size against
   the §2 baseline ISA for the same symbol.
3. Prefer Atlas ISA manifests or `llvm-readobj` / `llvm-objdump` on the
   candidate HSACO under `.hipfire_kernels/`.

If spills rose, matrix ops are missing, or the wrong wave size landed,
stop and fix the lever — do not measure a miscompiled candidate.

## 5. Correctness route (claim-scoped)

Select routes from [`docs/VALIDATION.md`](../../../docs/VALIDATION.md).
Minimum patterns for kernel work:

| Change | Typical minimum |
|---|---|
| New/changed `.hip` numeric behavior | `test_kernels` (channel vs CPU/reference) on the target arch, **then** the applicable model/path-level manual route on that arch from [`docs/VALIDATION.md`](../../../docs/VALIDATION.md); add a dedicated element-wise check for new WMMA/MFMA mappings. **Blocked** if no model-level route exists for the surface |
| Forward / fusion / KV **state or logits** | Path-specific parity oracle for that arch — **blocked** if none exists; serve harness is **not** parity |
| User-facing serve behavior only | `scripts/serve_harness.py` (or LFM harness for thinking frames) **after** parity if numbers can break |
| Dispatch `bind_thread` surface | `scripts/verify-bind-thread.sh` |
| Perf-only microkernel with identical math | Channel/parity still required when lane mapping or reduction order can change |

**Do not** treat retired `scripts/coherence-gate-*.sh` as merge or promotion
acceptance. **Do not** invent a one-script universal gate.

For WMMA/MFMA: element-wise reference comparison with row-mod histogram
diagnostics (see `case-studies.md` silent-corruption lesson). Speed floors
and English-shaped output miss systematic C-mapping bugs.

## 6. Fresh-process measurement

Follow [`docs/methodology/perf-benchmarking.md`](../../../docs/methodology/perf-benchmarking.md)
end-to-end. Skill-level checklist:

1. Warm DPM / JIT (`HIPFIRE_DPM_WARMUP_SECS` or throwaway run).
2. **Delete** the bench/daemon binary before rebuild so `ensure_build` cannot
   measure a stale artifact.
3. Record **binary md5**, **prompt md5** (byte-identical fixture files), model
   identity, arch, flags, and phase.
4. Cross-commit helper limits: `scripts/probe_commits.sh <baseline> <candidate>`
   is a **Qwen3.5 in-process** one-sample-per-ref probe — not a general-model
   product-path harness and not protocol-complete alone. Prefer repeated refs
   (e.g. A/B/B/A) or another surface-matched fresh-process harness that keeps
   **raw samples**. Do **not** treat `scripts/gates.sh --perf` as
   protocol-complete evidence: it runs one baseline and one HEAD sample and
   also pulls unrelated Redline/serve arms.
5. Use daemon-authoritative prefill/decode rates for product-shaped claims;
   do not cite eval-loop wall tok/s as kernel speed.
6. ABBA or equivalent when the delta is near the protocol noise band; keep
   the raw-sample report/artifact for the `measured` truth state.
7. Optional: `scripts/speed-gate.sh` against
   `tests/speed-baselines/<arch>.txt` when that floor policy applies to the
   touched paths — update baselines in the **same** commit as a deliberate
   trade-off, never as a silent chore.

A delta that does not survive fresh-process is **not** a win.

## 7. Adjacent-arch regression surface

Before merge intent:

- Every arch that can load the edited source tag or hit the edited predicate
  must either stay on an unchanged path or be measured.
- No local hardware for an enablement → **do not** flip public dispatch;
  expose methods + tests only.
- Capability-legal on many chips ≠ perf-safe on many chips
  (`perf-arch-discipline.md`).

## 8. Decide: promote, reject, or park

| Outcome | Action |
|---|---|
| Correct + real e2e win on target; adjacent OK | Ship with commit template below; keep claim **measured** unless a separate admission process applies |
| Correct + microkernel win, flat e2e | Ship only if it unblocks a known next fuse; otherwise reject as non-goal |
| Correct + no win / regression | **Reject and log** (next section); revert or leave opt-in dead code only with explicit env gate |
| Incorrect | Fix or revert; do not keep "fast wrong" behind defaults |

This skill does **not** admit product defaults. Empty
`docs/admissions.yml` means fail closed on inferred admissions.

## 9. Rejection logging

1. **Hypothesis** and lever id from `levers.md`.
2. **Identity**: arch, model md5, prompt md5, binary md5, commit, flags,
   **`bench_date`** (UTC measurement date).
3. **Numbers**: baseline vs candidate metric + protocol (fresh-process, N runs,
   raw samples).
4. **Raw report/artifact path** (and digest where retained) for the timed run.
5. **Mechanism guess** (spills, BW already saturated, barrier was free, etc.).
6. **Disposition**: rejected / opt-in-only / revisit-if.

Where to put it:

- Commit message (required for reverts and null-result experiments).
- Atlas task `ledger.jsonl` / `result.json` when using Kernel Atlas eval.
- One-line pointer in `levers.md` negative section if the failure generalizes.

Do not delete failed variant sources without a log — silent deletion causes
re-discovery.

## Commit message template (wins and rejects)

```
perf(<arch>): <one-line outcome> — <metric> <before> → <after> | REJECT <reason>

Baseline: <sha>  Candidate: <sha>
Arch/quant/phase: <...>
Bench: <exact command>
Bench date (UTC): <YYYY-MM-DD or ISO>
Binary md5: <...>  Prompt md5: <...>  Model: <id/md5>
Raw report/artifact: <path>  (digest: <...> if retained)
Protocol: fresh-process N=<n> warmup=<...> repeated refs or surface harness (see methodology/perf-benchmarking.md)
Correctness: <VALIDATION routes run — test_kernels + model-level route>
Hypothesis: <...>
Adjacent archs: <unchanged | measured | dispatch not flipped>
Candidate ISA: <VGPR/SGPR/LDS/spills/wave vs baseline>
```

Match the shape used by historical recovery/revert commits (e.g. `4105035`,
`34eb024`) so `git log` remains searchable.

## Pitfalls

- **One-shell +8%.** Measure again under the methodology protocol.
- **Compiled on my GPU.** That is one of many tags; compile the matrix you touch.
- **Test passed but dispatch never hit the new path.** Confirm symbol/profile
  or temporary log on the candidate branch.
- **Widened `is_rdna3()` to ship a Strix tuning.** That is the
  `perf-arch-discipline` failure mode — allowlist atoms/classes instead.
- **Serve harness green as numeric proof.** Rejected by `VALIDATION.md`.
- **Coherence-gate script as current acceptance.** Retired; historical only.
- **eval_hipfire tok/s as kernel speed.** Scoring loop dominated; use daemon
  or bench protocol metrics.
- **Bypass speed-floor hooks to "iterate faster".** You lose the bisect
  signal the floor exists to provide.

## Done criteria

- [ ] Single lever; hypothesis written.
- [ ] Profile + source/ISA evidence for bottleneck class (rocprof coverage
      audit when wall/profile disagree).
- [ ] Post-compile candidate ISA inspected vs baseline.
- [ ] VALIDATION routes for the claim class executed (or explicitly blocked),
      including model-level route after `test_kernels` when required.
- [ ] Fresh-process protocol numbers with identity hashes, **`bench_date`**,
      and raw-sample/report artifact path.
- [ ] Adjacent-arch story recorded.
- [ ] Win shipped **or** rejection logged — no silent dead ends.
