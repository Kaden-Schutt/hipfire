# Cross-arch portability

The rule that makes hipfire run on RDNA1 → RDNA4 + APU + CDNA3
without per-arch forks is: **dispatch is layered, kernels are
arch-tagged, and the speed-gate enforces no regression on the
baseline**. Get this right and your fast path adds a win on the
target arch without breaking anyone else.

## The dispatch tree

`crates/rdna-compute/src/dispatch.rs` is the kernel-selection hot
path. Every GEMM / GEMV / norm / fused op routes through here:

```rust
pub fn gemm_qkv_hfq4g256(&self, ...) -> HipResult<()> {
    if has_wmma_f16(&self.arch) {
        return self.gemm_qkv_hfq4g256_wmma(...);   // gfx11 (validated)
    }
    if has_dot2_f32_f16(&self.arch) {
        return self.gemm_qkv_hfq4g256_dot2(...);   // gfx10/11/12 fallback
    }
    self.gemm_qkv_hfq4g256_baseline(...)             // scalar, last resort
}
```

Three principles:

### 1. Fast paths first; baseline last

Each branch is a feature predicate (`has_wmma_f16`,
`has_dot2_f32_f16`) defined at the top of `dispatch.rs`. Predicate
helpers exist so the same arch-feature check doesn't get duplicated
across 6 call sites. Add a new fast path ABOVE the existing
fallbacks; the chain falls through naturally to the slowest path
that's still correct.

### 2. No unreachable branches

When you add a more-specific check that absorbs an arch the broader
check used to handle, **narrow the broader check in the same diff**.
The skill `.skills/hipfire-arch-port/` enforces this — it's how the
gfx12 dispatch (PR #56) didn't introduce a dead `|| starts_with("gfx12")`
clause in the gfx11 branch.

Predicate helpers like `has_dot2_f32_f16` that legitimately cover a
broad family (RDNA1.5 / RDNA2 / RDNA3 / RDNA4) typically don't need
narrowing — the helper intentionally covers the family, and downstream
sites rely on that coverage. Edit the helper definition only if it's
genuinely wrong for the new arch.

### 3. Speed-gate the baseline arch every dispatch.rs change

The pre-commit hook fires `./scripts/speed-gate.sh --fast` whenever a
staged file matches the hotspot regex. This catches inlining /
register-allocator regressions from "should-be-no-op" refactors.

Real example from this session: commit `a048544` looked like a pure
refactor (replace 6 inline `||` checks with one predicate function);
the gate flagged a "50% prefill regression" that turned out to be
a stale-binary measurement artifact. The lesson is *not* "avoid
predicate helpers" — it's "trust the gate, run it from a clean build".
See `playbook.md` step 6.

## File-level arch tags

Per-arch kernel variants follow the dot-separated convention:

```
kernels/src/<base>.hip                  # default for all archs
kernels/src/<base>.gfx1100.hip          # chip-specific override
kernels/src/<base>.gfx12.hip            # family-wide override (gfx1200 + gfx1201)
kernels/src/<base>.gfx1030.v4.hip       # versioned variant (multiple ABIs for one chip)
kernels/src/<base>.wave64.hip           # wave-size variant (CDNA / RDNA wave64 mode)
```

Resolution priority in `scripts/compile-kernels.sh`:

1. `<base>.<chip>.hip` (e.g. `.gfx1201.hip`) — chip-specific wins.
2. `<base>.<family>.hip` (e.g. `.gfx12.hip`) — family wide if no chip-specific.
3. `<base>.hip` — default.

Family detection uses `${arch:0:5}` — works for RDNA1-4 (gfx10/11/12)
and CDNA3 (gfx94X → gfx94). The shipped scaffold for gfx12 (PR #56)
uses the family tag because the same kernel is correct for both
gfx1200 and gfx1201; if you're tuning specifically for a chip
(prefetch parameters, occupancy, etc.) use the chip tag.

## Adding a new fast path

The minimum-viable workflow:

1. **Author the kernel** with the arch-tag naming convention.
   `kernels/src/<existing>.<chip>.hip` for chip-specific or
   `<existing>.<family>.hip` for a family.
2. **Register the source** in
   `crates/rdna-compute/src/kernels.rs` via `include_str!`.
3. **Add a method to `Gpu`** that wires the kernel: parameters,
   grid/block dims, kernarg blob.
4. **Add the dispatch branch** in `dispatch.rs` above the existing
   fallback. Follow the no-unreachable-branches rule (step 2 of
   `playbook.md`).
5. **Run the three gates** per `playbook.md` step 5.

## What "won't break other arches" actually means

Concretely: for every arch in `tests/speed-baselines/`:

- If your fast path applies (the predicate matches), the speed-gate
  must measure ≥ baseline × (1 - tolerance) on that arch.
- If your fast path doesn't apply (predicate misses), nothing
  about the dispatch tree visible to that arch should have
  changed — the speed-gate should be a no-op delta.

The pre-commit hook only runs the speed-gate against the local
arch (typically gfx1100). For changes that touch dispatch.rs in
ways that affect multiple branches, you OR a contributor with the
target hardware should run the gate on each affected arch before
merging. This is what issue #57 is for on the gfx12 path — Robin
or another R9700-equipped contributor will measure and flip the
public dispatch when the perf delta is verified.

## When you can't validate every arch

Reality: the maintainer has gfx1100 + gfx1010 + gfx1030 (V620) +
gfx1013 (BC-250 APU) + remote MI300X access. Other arches require
contributor hardware:

- gfx1031 / gfx1032 — expected to work via gfx1030 family path
- gfx1100 / gfx1101 / gfx1102 — same family, gfx1100 numbers
  generalize within ~5%
- gfx1150 / gfx1151 — Strix Halo (issue #50), no local hardware
- gfx1200 / gfx1201 — RDNA4 (issue #57), validated on R9700 by
  PR #56 contributor

If your change is a fast path on an arch you DON'T have hardware
for: **do not enable it in the public dispatch**. Land the kernel
+ channel-test as PR #56 did (methods exposed on `Gpu`, not routed
through the public path), and open an issue for the dispatch flip
that asks for a perf measurement on real hardware. That's how Robin
contributed gfx12 — code now, dispatch flip after numbers.

## Cross-arch matrix (current targets)

From `cli/index.ts::archDefaults` + the speed-baselines tree
(`tests/speed-baselines/`). The "Speed-baseline" column is literal:
yes = a `<arch>.txt` file exists on disk and the speed-gate
compares against it; no = the gate refuses to run for that arch
unless `HIPFIRE_BASELINE_ARCH` overrides AND a baseline file is
authored.

| Arch | Wave | Matrix engine | KV default | Speed-baseline |
|---|---|---|---|---|
| gfx1010 (RX 5700 XT) | 32 | none | asym2 | no |
| gfx1013 (BC-250 APU) | 32 | none | asym2 | yes (`gfx1013.txt`) |
| gfx1030 (V620 / RX 6800 XT) | 32 | dot2 | asym3 | yes (`gfx1030.txt`) |
| gfx1031 (RX 6700 XT) | 32 | dot2 | asym3 | no (gfx1030-class) |
| gfx1032 (RX 6600 XT) | 32 | dot2 | asym2 | no |
| gfx1100 (7900 XTX) | 32 | WMMA | asym3 | yes (`gfx1100.txt`) |
| gfx1101 (7900 XT) | 32 | WMMA | asym3 | no (gfx1100-class) |
| gfx1102 (7800 XT) | 32 | WMMA | asym3 | no (gfx1100-class) |
| gfx1150 (Strix Halo APU) | 32 | WMMA | asym2 | no (issue #50 reproducer pending) |
| gfx1200 (Radeon AI Pro R9700) | 32 | WMMA-gfx12 | asym3 | no |
| gfx1201 (RX 9070 XT) | 32 | WMMA-gfx12 | asym3 | no (issue #57 — needs first measurement) |
| gfx94x (MI300X) | 64 | MFMA | asym3 (default) | no (remote rentals only) |

A row reading "no (gfx1100-class)" means the chip is in the same
arch family with the same matrix engine and is expected to inherit
the parent's perf shape within ~5%; if you have one, contributing a
`gfx1101.txt` (etc.) is welcome — see [`hipfire-tester`](../hipfire-tester/)
for the bench-submission flow.

If you add a new arch, update `archDefaults`, ship a speed-baseline
file, and add a row here. Adding the row without the file is the
mistake this section was just rewritten to fix — Codex stop-time
review caught it on the initial draft, 2026-04-27.
