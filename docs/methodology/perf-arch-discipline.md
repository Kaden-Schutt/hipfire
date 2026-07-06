# Perf-Arch Discipline: Capability Predicates vs. Perf Selection

**Mandate date:** 2026-06-12 (Kaden)
**Scope:** All kernel-variant selection in `crates/rdna-compute/src/gemm.rs` and any
future dispatch-side code that selects between performance sub-variants.

This document defines the rules, infrastructure, and enforcement machinery for keeping
**ISA-correctness gating** and **performance-variant selection** cleanly separated.
Violating this separation caused a ~14% DFlash decode regression (fixed in 24e4baa9)
that took rocprofv3 to attribute. Don't repeat it.

---

## 1. The Rule

**Capability predicates gate correctness. They NEVER select a perf variant.**

A capability predicate (`is_rdna3()`, `has_wmma_w32()`, `is_rdna3p5()`,
`arch.starts_with("gfx12")`) answers the question: *"does this arch support this ISA
feature?"* It is the right guard for choosing between a WMMA path and a scalar path,
or between a gfx12-specific intrinsic and a portable one. It is the wrong key for
choosing between `plain` and `ldscoop` variants of the same WMMA kernel.

**Perf-variant selection keys on a specific measured arch with a conservative portable
default.** New archs get the portable default until someone benches them and records the
result in the perf ledger. There is no "inherit the best-tuned variant by capability
inheritance" mechanism because that is the exact failure mode this mandate closes.

**Polarity of enforcement is inverted from the current code.** The old code used a
positive capability gate to select an optimized variant: "if `is_rdna3`, use ldscoop"
— meaning any arch that satisfied `is_rdna3` inherited ldscoop regardless of whether
it had been measured. The new polarity is: **default is conservative; optimizations
are positive allowlist entries.** A gfx1151 tuning cannot silently immigrate to
gfx1100 because gfx1100 has its own explicit entry or falls through to the portable
default.

---

## 2. The Motivating Failure (24e4baa9)

### What happened

`gemm_gate_up_hfq4g256_wmma` has three sub-variants: `plain`, `ldscoop`, and
`ldscoop_nosync`. The `ldscoop` variant was measured as best on gfx1151 (Strix Halo,
LPDDR5X, 32 MB L3). It was selected via:

```rust
// BEFORE 24e4baa9 — bug
if self.arch_caps.is_rdna3() {
    ldscoop
} else {
    plain
}
```

`is_rdna3()` returns true for all gfx1100/1101/1102/1103 AND gfx1150/1151/1152 archs.
When gfx1151 was added to the fleet, the `ldscoop` measurement was recorded for
gfx1151 and the predicate was widened to cover it. gfx1100 (RDNA3 dGPU, GDDR6 +
Infinity-Cache — a completely different memory hierarchy from gfx1151's LPDDR5X) fell
into `ldscoop` by predicate inheritance.

Commit 303d69e9 had already measured and documented that `ldscoop` is SLOWER than
`plain` on gfx1100. That measurement was overridden by the capability predicate silently
expanding to cover a new arch.

### What rocprofv3 showed

```
ldscoop vs plain on gfx1100 (DFlash, 448 launches):
  ldscoop:  +48% per launch (+49.6 ms total)
```

+49.6 ms over 448 DFlash launches = ~97% of a measured ~14% DFlash decode regression.
Additionally, `ldscoop` crashes at large prefill batch sizes on gfx1100 — the variant
is not just slower, it is less stable.

### The fix (24e4baa9)

```rust
// AFTER 24e4baa9 — correct
let def = if self.arch.starts_with("gfx1151") || self.arch.starts_with("gfx1150") {
    ldscoop_nosync   // measured best on Strix Halo (LPDDR5X)
} else if self.arch_caps.is_rdna3_dgpu() {
    plain            // measured best on gfx1100 (GDDR6 + Infinity-Cache)
} else {
    ldscoop          // gfx1201 RDNA4: unbenched, pending hiptrx measurement
};
```

The `else` arm is now an explicit "unbenched, this is a placeholder" rather than
"inherited by capability." The code is still not fully compliant with this mandate
(the else arm still implicitly selects a variant for gfx1201 without a ledger entry),
but it documents the debt explicitly. The mandate formalizes the expectation.

---

## 3. The Bug Class Inventory

Five live instances of the pattern (as of 2026-06-12); use this as a checklist when
auditing new dispatch code.

### Instance 1 — gate_up hfq4g256 wmma default arm (partially fixed)

`crates/rdna-compute/src/gemm.rs` ~line 9204

The `else` arm selects `ldscoop` for all non-gfx115x, non-RDNA3-dGPU archs. This
includes gfx1201 (RDNA4, GDDR6X + 64 MB L3) and gfx1103 (RDNA3 APU, LPDDR5, no
Infinity-Cache). Neither is benched. **Action:** add gfx1201 and gfx1103 to the
perf ledger; until then the `else` is acknowledged debt, not approved default.

### Instance 2 — hfq4g128 MMQ gate

`crates/rdna-compute/src/gemm.rs` ~line 124

```rust
let use_mmq = self.arch.starts_with("gfx1151")
    && std::env::var("HIPFIRE_HFQ4G128_MMQ").as_deref() != Ok("0")
```

Atom-specific (gfx1151 only, not predicate-inherited). Compliant with the polarity rule.
Not in the perf ledger. **Action:** add a ledger entry citing the measurement commit.

### Instance 3 — grouped MoE GEMM i8 gfx11 dGPU gate

`crates/rdna-compute/src/gemm.rs` ~line 10165

```rust
let use_i8_gfx11_dgpu = (self.arch.starts_with("gfx1100")
    || self.arch.starts_with("gfx1101")
    || self.arch.starts_with("gfx1102")
    || self.arch.starts_with("gfx1103"))
    && self.flags.moe_grouped_i8.unwrap_or(true);
```

Measured on gfx1100 (+2.8%), extrapolated to gfx1101/1102/1103. gfx1103 is an APU
with different cache geometry. **Action:** ledger entry noting extrapolation; add
gfx1103 opt-out or explicit bench.

### Instance 4 — residual WMMA gfx11 vs gfx12 split

`crates/rdna-compute/src/gemm.rs` ~line 14879

```rust
return if arch.starts_with("gfx12") {
    self.gemm_hfq4g256_residual_wmma_gfx12(...)
} else {
    self.gemm_hfq4g256_residual_wmma(...)
};
```

Correctness-adjacent (ISA intrinsic differences per arch family). This is a valid use
of a capability predicate — the two paths are not perf variants of the same kernel,
they are different kernels for different ISA. **Status:** compliant; document in ledger
for auditability.

### Instance 5 — hfq6g256 residual ksplit selection

`crates/rdna-compute/src/gemm.rs` ~line 13442

```rust
let is_gfx115x = self.arch_caps.is_rdna3p5();
// downstream: picks "k2", "k2x32", "k4", "ksplit_det" by is_gfx115x + batch_size
```

Inline measured-best table using a capability predicate as the arch discriminant.
Compliant in spirit (gfx115x is specific enough that there are no ambiguous arch
class members today), but not in the ledger. **Action:** add ledger entry.

---

## 4. Kernel-Variant Selection Table Design

The current dispatch infra (`crates/hipfire-dispatch/src/tables/`, `types.rs`) encodes
ISA-availability predicates in `ArchPredicate`. The perf-variant layer is a separate
second level that sits on top of it.

### Two-level model

```
Level 1: ArchPredicate gate  — correctness / ISA availability
          KernelKey → ArchPredicate     (already in tables/*.rs)

Level 2: PerfVariant allowlist — performance-variant selection
          (KernelKey, ArchClass) → PerfVariantId
```

### ArchClass enum

Derived from `ArchCaps` molecules, not from raw arch strings. Location:
`crates/rdna-compute/src/arch_caps.rs` (new method `arch_class()`).

```rust
pub enum ArchClass {
    Rdna3Dgpu,   // gfx1100/1101/1102 — GDDR6 + Infinity-Cache
    Rdna3p5,     // gfx1150/1151/1152 — LPDDR5X + 32 MB L3, no Infinity-Cache
    Rdna3Apu,    // gfx1103 — LPDDR5 APU, small cache, no Infinity-Cache
    Rdna4,       // gfx1200/1201 — GDDR6X + 64 MB L3
    Cdna3,       // gfx940/941/942 — HBM, rocBLAS preferred
    Rdna2,       // gfx1030-1032
    Rdna1,       // gfx1010
    Gcn5,        // gfx906
    Unknown,     // portable-only fallback for any arch not enumerated
}
```

`Unknown` is the catch-all. Variants selected via the `Unknown` row are the portable
defaults — the conservative fallback. They must be correct and functional across all
archs; they do not need to be optimal.

### Variant selection table schema

Flat table with these columns:

```
kernel_id | arch_class | variant_id | measured | source_commit | bench_date | notes
```

- `kernel_id` — identifies the dispatch entry point, e.g. `gemm_gate_up_hfq4g256_wmma`
- `arch_class` — one of the `ArchClass` values above, or `*` for the portable default
- `variant_id` — opaque string identifying the sub-variant, e.g. `plain`, `ldscoop`,
  `ldscoop_nosync`, `mmq`, `i8`, `ksplit_det`
- `measured` — boolean; `no` entries are acknowledged debt, not approved selections
- `source_commit` — the git commit that established this measurement (or `pending`)
- `bench_date` — YYYY-MM-DD of the bench run
- `notes` — one-line context: what was measured, what the delta was

### Current known-good entries (as of 2026-06-12)

```
kernel_id                          arch_class  variant_id      measured  source      bench_date   notes
─────────────────────────────────  ──────────  ──────────────  ────────  ──────────  ───────────  ────────────────────────────────────────────────────
gemm_gate_up_hfq4g256_wmma         Rdna3Dgpu   plain           yes       24e4baa9    2026-06-12   plain beats ldscoop +48%/launch on gfx1100 (GDDR6+IC)
gemm_gate_up_hfq4g256_wmma         Rdna3p5     ldscoop_nosync  yes       e3232034    2026-06-xx   nosync best on Strix Halo LPDDR5X
gemm_gate_up_hfq4g256_wmma         Rdna4       ldscoop         no        pending     pending      else-arm placeholder; hiptrx bench required
gemm_gate_up_hfq4g256_wmma         *           plain           yes       303d69e9    2026-xx-xx   portable baseline
gemm_hfq4g128_mmq                  Rdna3p5     mmq             yes       (inline)    unknown      gfx1151 only; needs ledger source citation
gemm_grouped_moe_hfq4_wmma_i8      Rdna3Dgpu   i8              yes       (inline)    unknown      +2.8% on gfx1100; gfx1101/1102 extrapolated
gemm_grouped_moe_hfq4_wmma_i8      Rdna3Apu    i8              no        pending     pending      gfx1103 cache differs; bench or opt-out required
gemm_grouped_moe_hfq4_wmma_i8      *           noi8            yes       (portable)  baseline     portable fallback, always correct
gemm_hfq6g256_residual_wmma        Rdna3p5     ksplit_det       yes       (inline)    unknown      needs ledger source citation
gemm_hfq6g256_residual_wmma        *           k2              yes       (portable)  baseline     portable fallback
```

This table lives at `docs/methodology/perf-variant-ledger.json` (machine-readable) and
is summarized here for human reference. See section 5 for the full schema.

---

## 5. Hashed Per-Arch Perf Ledger

The variant selection table above is the *allowlist*. The perf ledger is the *evidence
store* — the actual bench numbers that justify each allowlist entry. They are one system:
the bench suite is the ledger's write path, the allowlist is the ledger's read path.

### Ledger file location

`docs/methodology/perf-variant-ledger.json`

One JSON object per arch/kernel/variant triple. Schema:

```json
{
  "schema_version": 1,
  "entries": [
    {
      "kernel_id": "gemm_gate_up_hfq4g256_wmma",
      "arch_class": "Rdna3Dgpu",
      "arch_atom": "gfx1100",
      "variant_id": "plain",
      "delta_vs_portable_pct": 0.0,
      "delta_vs_alt_pct": -32.0,
      "alt_id": "ldscoop",
      "tok_s": 241.3,
      "model_id": "qwen3.6-27b.mq4",
      "model_md5": "...",
      "prompt_file": "benchmarks/prompts/lru_cache_pep8_strict.txt",
      "prompt_md5": "...",
      "binary_md5": "...",
      "source_commit": "24e4baa9",
      "bench_date": "2026-06-12",
      "bench_host": "k9lin",
      "hipfire_flags": "HIPFIRE_DPM_WARMUP_SECS=10",
      "notes": "ldscoop +48%/launch, +49.6ms/448 DFlash launches = ~97% of 14% regression"
    }
  ]
}
```

### Hash pinning requirements

Every ledger entry MUST record:
- `model_md5` — md5 of the `.mq4`/`.mq6`/etc. weight file used
- `prompt_md5` — md5 of the prompt bytes (one whitespace char = 17% τ swing; see
  `perf-benchmarking.md` prompt-structure section)
- `binary_md5` — md5 of the bench binary that produced the number. For the canonical
  daemon-driven path (`cli/bench_sweep.ts`) this is the daemon binary
  (`target/release/examples/daemon`); the suite emits the same `binary_md5`
- `source_commit` — full 8-char git hash

An entry without all four hashes is advisory only and MUST NOT be cited as justification
for a variant allowlist selection.

### How to add a new entry

The ledger's write path is the unified bench suite (`cli/bench_sweep.ts`), NOT the
in-process `bench_qwen35_mq4` microbench. `bench_qwen35_mq4` is retired from the bench
path because it misses the daemon's AR hipGraph and reads ~20% low on MoE decode — see
`bench-suite.md` §"Migration table". Drive the daemon suite, then transcribe its
hashed-JSON line into a ledger entry:

```bash
# 1. Build the daemon and pin the warmup/graph env (suite handles DPM + JIT warmup
#    internally via its two-phase protocol — see bench-suite.md §"Continuous warmup")
cargo build --release -p hipfire-runtime --example daemon --features deltanet
export HIPFIRE_DAEMON_BIN=./target/release/examples/daemon
export HIPFIRE_AR_GRAPH=1
export HIPFIRE_DPM_WARMUP_SECS=10

# 2. Run the suite (it warms, then medians the measured pass; emits ONE JSON line)
bun cli/bench_sweep.ts ~/.hipfire/models/qwen3.6-27b.mq4 \
  9216 256,1024,4096 128 "$(cat benchmarks/prompts/lru_cache_pep8_strict.txt)"

# 3. Map the suite's hashed-JSON fields → ledger fields (identical hash basis;
#    see bench-suite.md §"Hashed-JSON output schema" for the field crosswalk):
#      suite model_md5      → ledger model_md5
#      suite prompt_md5     → ledger prompt_md5
#      suite binary_md5     → ledger binary_md5   (md5 of the daemon binary)
#      suite hipfire_ar_graph + KV/flags → ledger hipfire_flags
#      suite decode_tok_s / pp[...] → ledger tok_s (record which metric in notes)
#      suite timestamp_utc  → ledger bench_date (date portion)
# If the suite has not yet been augmented to emit the hash fields, record them by hand:
md5sum ~/.hipfire/models/qwen3.6-27b.mq4
md5sum benchmarks/prompts/lru_cache_pep8_strict.txt
md5sum "$HIPFIRE_DAEMON_BIN"

# 4. Append the entry to perf-variant-ledger.json
# 5. Update the allowlist entry in dispatch code (or add one)
# 6. Commit both changes atomically
```

All four hashes in the ledger are **md5** (model/prompt/binary + the 8-char
`source_commit`). The suite emits the same md5 basis — there is exactly one hash
convention across the producer and the consumer; do not introduce a second
(e.g. sha256) algorithm on either side.

---

## 6. Scoped Coherence Gate

**The coherence gate runs are scoped to the set of variant-table cells touched by a
diff.** Running the full gate for a bench comment update wastes 10+ minutes and
discourages discipline. Running nothing for a dispatch change is the failure mode.

### Scope inference (pre-commit hook)

The pre-commit hook at `.githooks/pre-commit` already gates on file patterns. Extend it
to additionally:

1. Parse `git diff --staged` for changes to files matching `gemm.rs`, `dispatch*.rs`,
   `tables/*.rs`, `arch_caps.rs`
2. Extract the set of `kernel_id` strings touched (grep for the function name)
3. Emit `HIPFIRE_COHERENCE_KERNELS=<comma-list>` into the child environment
4. The coherence gate reads this env var and runs only the relevant model/prompt pairs

Until the scoped gate is implemented, the full `./scripts/coherence-gate.sh` is
mandatory for any dispatch-side change. The scope inference reduces this to a few
seconds for the common case of touching one kernel family.

### Which cells require coherence gating

Any change that modifies a `variant_id` mapping in the allowlist (or the code that
implements that mapping in `gemm.rs`) requires coherence gating on ALL models/archs
that use that kernel. A change to a `Rdna4`-only entry still requires a gfx1201 pass
because the new variant might introduce an attractor or token-loop on that arch.

---

## 7. Watchdog GitHub Action

A CI action validates the ledger without needing a GPU. It runs on every PR that
touches dispatch code or the ledger.

### What it validates (no GPU required)

1. **Schema validation** — `perf-variant-ledger.json` parses and all required fields
   are present
2. **Allowlist cross-reference** — every `(kernel_id, arch_class)` entry in the
   allowlist has a corresponding ledger entry with `measured: true`, or is explicitly
   flagged `measured: false` (acknowledged debt)
3. **No new `measured: false` entries without a `notes` explanation** — a PR that adds
   a placeholder must document why the bench is deferred
4. **Regression detection** — if the PR modifies an existing ledger entry for a
   `arch_class` that already has a `measured: true` entry, the action flags it for
   human review. The gfx1100 row is specifically named in the action config so that any
   PR touching the `Rdna3Dgpu` entry for a kernel that has a confirmed gfx1100 number
   generates a mandatory reviewer annotation

### What it CANNOT validate (GPU required)

- Whether the actual bench numbers are correct
- Whether the new variant causes a coherence regression
- Whether the binary_md5 in the ledger matches the current build

These require a GPU and are gated by the pre-commit hook + manual perf probe protocol
described in `perf-benchmarking.md`.

### Action location

`.github/workflows/perf-arch-discipline.yml`

Runs on: `pull_request` touching `crates/rdna-compute/src/gemm.rs`,
`crates/hipfire-dispatch/src/tables/**`, `docs/methodology/perf-variant-ledger.json`

---

## 8. Soft Enforcement: Pre-Commit Hook

The existing hook at `.githooks/pre-commit` (activated by `git config core.hooksPath
.githooks`) is extended with a perf-arch-discipline check:

```bash
# In .githooks/pre-commit, after the existing coherence-gate section:

# Perf-arch discipline check
if git diff --staged --name-only | grep -qE '(gemm\.rs|dispatch.*\.rs|tables/)'; then
  echo "[pre-commit] Dispatch-side change detected — checking perf-arch discipline"

  # Reject direct is_rdna3/has_wmma_w32 usage as a perf-variant selector
  if git diff --staged -- crates/rdna-compute/src/gemm.rs \
     | grep '^+' \
     | grep -E 'is_rdna3\(\)|has_wmma_w32\(\)' \
     | grep -v '// correctness:' ; then
    echo "[pre-commit] FAIL: capability predicate used in new dispatch code."
    echo "  Capability predicates (is_rdna3, has_wmma_w32) may only gate ISA"
    echo "  correctness. Perf variants must key on arch_class enum + ledger."
    echo "  See docs/methodology/perf-arch-discipline.md section 1."
    exit 1
  fi

  # Warn if a new arch.starts_with() perf branch lacks a ledger entry
  if git diff --staged -- crates/rdna-compute/src/gemm.rs \
     | grep '^+' \
     | grep -E 'arch\.starts_with\("gfx' ; then
    echo "[pre-commit] WARNING: new arch.starts_with() in gemm.rs."
    echo "  Ensure a perf-variant-ledger.json entry exists for this arch."
    echo "  If this is a correctness (ISA) gate, add a '// correctness:' comment."
  fi
fi
```

The hook emits a hard failure only for the clearest violation (capability predicate in
new dispatch code without a correctness annotation). New arch.starts_with() gates emit
a warning that can be overridden by adding a `// correctness:` comment to document that
the gate is ISA-driven, not perf-driven.

---

## 9. Build Order

Implement in this order to avoid building on unvalidated infrastructure:

### Step 1 — Variant table (no GPU required)

Add `ArchClass` enum to `crates/rdna-compute/src/arch_caps.rs`. Create
`docs/methodology/perf-variant-ledger.json` with the currently-known entries from
section 4 (marking unbenched entries as `measured: false`). Add the allowlist read
path to `gemm.rs` for the five instances catalogued in section 3. This is a pure
refactor: no behavior change, no new bench numbers required.

Validation: `cargo check` clean + `cargo test` green + `git diff` shows the five
instances converted from inline predicates to allowlist lookups.

### Step 2 — Ledger write path (GPU required for new entries)

For each `measured: false` entry in the ledger, run the bench protocol (section 5)
on the relevant arch and record the result. Start with gfx1201 (Rdna4) since it has
the highest probability of having a different optimal variant from gfx1100.

Priority order: gfx1201 > gfx1103 > gfx1101/1102 (the latter two are architecturally
close enough to gfx1100 that the gfx1100 measurement is probably directionally correct,
but they still need entries to close the acknowledged-debt flag).

### Step 3 — Scoped coherence gate (software only)

Extend `.githooks/pre-commit` with the scope inference logic (section 6). Test by
staging a change to a single kernel family and confirming that only the relevant
coherence subset runs.

### Step 4 — Watchdog GH action (software only)

Implement `.github/workflows/perf-arch-discipline.yml` against the ledger schema.
Confirm that a PR adding a `measured: false` entry without `notes` generates a CI
failure, and that a PR modifying the `Rdna3Dgpu` row generates a reviewer annotation.

---

## 10. The Stale Baseline Failure (Motivating the Ledger)

The pflash baseline on k9lin was 6+ weeks stale when the DFlash regression was caught.
The consequence: the regressed number (post-24e4baa9 regression) was being compared
against a baseline from a different binary, a different kernel cache state, and
potentially a different DPM configuration. The delta was visible only because the
regression was large (~14%); a 5% regression against a 6-week-old baseline would have
been invisible.

The perf ledger closes this by making the baseline an explicit artifact with a
`bench_date` and `binary_md5`. If the `binary_md5` in the ledger does not match the
current build, the bench suite emits a staleness warning. If `bench_date` is more than
30 days ago, the watchdog action flags the entry as potentially stale in PR review.

**Rule:** any regression claim MUST be supported by a fresh probe run (`scripts/probe_commits.sh`)
against the current binary, not a recalled number from a prior session. The ledger
provides the reference; the probe provides the comparison. Together they form a
reproducible record that can be audited months later.

---

## 11. Perf-Arch Discipline Agent Skill

A future `docs/skills/perf-arch-discipline.md` skill will provide a step-by-step
checklist for an agent adding a new kernel variant:

1. Identify the `kernel_id` and the archs it will be dispatched on
2. Look up the current allowlist entry for each arch in `perf-variant-ledger.json`
3. Add a `measured: false` placeholder entry for any arch that lacks a measurement
4. Implement the variant with a `// correctness:` or `// perf:` annotation on each
   ISA/perf gate
5. Run the bench protocol on each target arch; record results in the ledger
6. Update the allowlist to `measured: true` for benched entries
7. Run the scoped coherence gate for the affected kernel family
8. Commit allowlist + ledger + code atomically

Until that skill file is written, this document is the authoritative reference. Agents
dispatched on dispatch-side tasks should be pointed here before writing any new
`arch.starts_with()` or `arch_caps.is_*()` calls in perf-variant selection logic.

---

*Cross-reference: `perf-benchmarking.md` (warmup protocol, noise band, cross-process verification),
`arch-port-validation.md` (per-arch validation gates), `kernel-atlas.md` (kernel inventory).*
