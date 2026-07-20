# Perf-arch discipline

Keep **ISA / correctness gating** separate from **performance-variant
selection**. Violating that separation caused a measured ~14% DFlash
decode regression on gfx1100 when an LPDDR-tuned `ldscoop` sub-variant
was inherited via `is_rdna3()` (fixed in `24e4baa9`; attributed with
rocprofv3). Do not repeat the pattern.

| Concern | Owner |
|---|---|
| How to time a claim | [`perf-benchmarking.md`](perf-benchmarking.md) |
| Daemon suite producer | [`bench-suite.md`](bench-suite.md) |
| Validation route selection | [`docs/VALIDATION.md`](../VALIDATION.md) |
| Numbers / tables | [`docs/BENCHMARKS.md`](../BENCHMARKS.md), `docs/perf-checkpoints/` |
| Kernel inventory / Atlas | [`kernel-atlas.md`](kernel-atlas.md) |

## Rule

**Capability predicates answer “is this ISA legal?”**
Examples: `is_rdna3()`, `has_wmma_w32()`, `is_rdna3p5()`, family
`arch.starts_with("gfx12")` when choosing a **different kernel** required
by instruction set.

**They never choose among perf sub-variants of the same kernel**
(`plain` vs `ldscoop` vs `ldscoop_nosync`, MMQ on/off as a speed pick,
i8 grouped MoE, ksplit tables, and similar).

**Perf-variant selection is a positive allowlist** keyed by a specific
measured arch (or a narrowly named arch class), with a **conservative
portable default** for everything else. New silicon gets the portable
default until someone measures it and records evidence. There is no
“inherit the best-tuned cousin by capability.”

Polarity:

- **Wrong:** `if is_rdna3() { fast_variant } else { plain }`
- **Right:** explicit arch/class rows for measured winners; `else` →
  portable default (or an explicitly labeled unbenched placeholder that
  is not treated as approved).

ISA forks that are **different kernels** (e.g. gfx11 residual WMMA vs
gfx12 residual WMMA) remain capability-correct. Document them; do not
confuse them with same-kernel sub-variant picks.

## Motivating failure (summary)

`gemm_gate_up_hfq4g256_wmma` sub-variants: `plain`, `ldscoop`,
`ldscoop_nosync`. Best on gfx1151 (Strix Halo, LPDDR5X) was selected with
`is_rdna3()`, which also matches gfx1100 dGPU (GDDR6 + Infinity Cache).
Prior measurement had already shown `ldscoop` slower on gfx1100; the
predicate expansion reintroduced it. rocprofv3: large per-launch tax on
hundreds of DFlash launches ≈ the wall regression; variant also unstable
at large prefill batches on that dGPU.

Correct shape: gfx115x → measured winner; RDNA3 dGPU → `plain`; other
rows explicit or portable — never silent inheritance.

## Selection model

Two levels:

```text
Level 1  ArchPredicate / ISA     → which kernel family may run
Level 2  Perf allowlist          → (kernel_id, arch_atom|class) → variant_id
```

Suggested class names when grouping atoms (implement in
`arch_caps` / dispatch when wiring; names are documentation until code
owns them):

| Class | Atoms (illustrative) | Memory note |
|---|---|---|
| Rdna3Dgpu | gfx1100/1101/1102 | GDDR6 + Infinity Cache |
| Rdna3Apu | gfx1103 | APU, no IC |
| Rdna3p5 | gfx1150/1151/1152 | LPDDR5X, large L3, no IC |
| Rdna4 | gfx1200/1201 | GDDR6 + Infinity Cache / large on-die cache |
| Cdna3 | gfx940/941/942 | HBM |
| Rdna2 / Rdna1 / Gcn5 | gfx103x / gfx101x / gfx906 | portable-first |
| Unknown | everything else | portable default only |

`Unknown` and portable `*` rows must be **correct**, not optimal.

### Allowlist row fields

| Field | Meaning |
|---|---|
| `kernel_id` | Dispatch entry, e.g. `gemm_gate_up_hfq4g256_wmma` |
| `arch_class` / `arch_atom` | Class and/or exact gfx string measured |
| `variant_id` | `plain`, `ldscoop`, `mmq`, `i8`, `ksplit_*`, … |
| `measured` | `true` only with evidence; `false` = acknowledged debt |
| `source_commit` | Short git id that landed the measurement or selection |
| `bench_date` | UTC date of the supporting run |
| `notes` | Delta, fixture, caveats |

Machine-readable variant ledger: **planned / blocked** (no checked-in path).
Treat allowlist claims as **unimplemented debt** until a ledger lands —
do not invent rows from memory. Human summary may live in campaign
checkpoints in the meantime.

### Evidence hash pin (same as bench protocol)

A `measured: true` row that justifies shipping a non-default variant
needs:

- `model_md5`, `prompt_md5`, `binary_md5` (md5 only)
- `source_commit`, `bench_date`
- metric identity (decode vs prefill length, flags string including
  `HIPFIRE_DPM_WARMUP_SECS` / graph / KV)

Rows missing hashes are **advisory** and must not alone justify flipping
a default. Produce numbers via
[`perf-benchmarking.md`](perf-benchmarking.md) +
[`bench-suite.md`](bench-suite.md) (daemon path for production decode;
in-process only when the claim is about that path).

## Working checklist (dispatch change)

1. Name `kernel_id` and every arch atom that can hit the new branch.
2. Classify each gate: `// correctness:` (ISA) vs `// perf:` (variant).
3. For perf gates: portable default first; add allowlist rows only for
   measured atoms/classes.
4. Do not widen a capability predicate to “carry” a perf win to cousins.
5. Time with full identity; matched fresh-process samples required; use
   declared ABBA/interleave **ordering** over those samples when ordering or
   thermal bias matters (ABBA is never a substitute for fresh-process);
   raw samples; no point-estimate promotion
   ([`perf-benchmarking.md`](perf-benchmarking.md)).
6. Attribute with rocprof when the claim is kernel-level
   ([`rocprof-coverage.md`](rocprof-coverage.md)).
7. Run the **VALIDATION route** for the change class (numeric channel,
   path oracle, serve semantics, etc.). Do **not** treat retired batteries
   as acceptance.
8. Land code + evidence references together; leave `measured: false`
   placeholders only with notes and a non-approved default.

## Anti-patterns

| Pattern | Disposition |
|---|---|
| Capability predicate selects ldscoop/MMQ/i8/ksplit | Reject |
| Unmeasured arch inherits neighbor’s winner | Reject |
| Point estimate or best run flips default | Reject |
| Stale baseline (old binary/model/prompt md5) as proof | Reject — re-measure |
| Green unrelated harness as variant proof | Reject |
| Retired batteries as current acceptance | Reject — historical only; use VALIDATION |
| Universal “one gate for all dispatch” | Reject — [`VALIDATION.md`](../VALIDATION.md) |
| Measurement alone writes `admissions.yml` | Reject — fail closed |

## Historical inventory (audit aid)

Instances that motivated the rule (line numbers drift — search symbols
in `gemm` / dispatch). Re-audit against current source before citing as
live bugs:

| Area | Notes |
|---|---|
| `gemm_gate_up_hfq4g256_wmma` default arm | gfx115x vs RDNA3 dGPU vs else; else must not silently optimize RDNA4 |
| HFQ4g128 MMQ | Atom-specific env gate is polarity-OK; still needs evidence if default-on |
| Grouped MoE i8 on gfx11 | Measured on one atom; do not extrapolate to APUs without data |
| Residual WMMA gfx11 vs gfx12 | ISA split — correctness-adjacent |
| HFQ6 residual ksplit via `is_rdna3p5()` | Prefer explicit class + ledger over bare capability |

## Stale baseline

Comparing today’s binary to a weeks-old remembered tok/s hides mid-size
regressions. Durable claims carry `bench_date` + `binary_md5` + model /
prompt identity. If digests do not match the fixture under test,
re-run. Optional commit probes:
`scripts/probe_commits.sh` (in-process lower bound) or daemon suite
A/B under the perf protocol — choose the path that matches the claim.

## Explicit non-goals

- This file does not define a universal CI gate, scoped coherence env
  contract, or promised GitHub workflow. Hooks and workflows exist only
  where the tree implements them; verify in source.
- This file does not store speed tables.
- Executable agent skills live under [`.agents/skills/`](../../.agents/skills/)
  only — sole skill root; do not invent alternate skill trees.
