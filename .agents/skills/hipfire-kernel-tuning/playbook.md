# Kernel-tuning playbook

The 6-step workflow that's produced real perf wins in this repo. Each
step has a gate that has to clear before you advance — skipping a
step is how regressions ship.

## 1. Measure first, hypothesize second

Don't optimize from intuition. Pull the actual profile.

```bash
HIPFIRE_PROFILE=1 hipfire bench qwen3.5:9b --runs 5 2>&1 | tee bench.log
```

The `crate::profile::begin_timer` instrumentation is wired through
every dispatched kernel. The output gives per-kernel µs / GB-s /
%-of-cycle so you can identify the bottleneck.

Specific patterns to look for:

- **High µs but low GB/s** — latency-bound, not bandwidth-bound. Adding
  more compute won't help. Look at occupancy, kernel launch overhead,
  or memory-access pattern.
- **High GB/s near peak** — bandwidth-bound, already saturated. Don't
  spend time here unless you can change the algorithm to need less BW
  (e.g. fused kernels that share weight reads).
- **Low GB/s + low %-of-cycle** — kernel is fine, look elsewhere.
- **High launch count** — kernel-launch overhead dominates. Fuse with
  a neighboring op, or batch invocations across a forward pass.

Anti-pattern: "this kernel feels slow on my machine" — without a
profile, you're going to optimize the wrong thing or spend hours
on a gain that's inside the noise band.

## 2. Root-cause the bottleneck

Once you have a number, figure out WHY before picking a lever:

- **VGPR pressure** — check `--save-temps` output for spills. RDNA3
  budget is 256 VGPRs/wave at 100% occupancy; spilling drops you to
  the next occupancy step.
- **LDS bank conflicts** — happen when concurrent threads in a wave
  hit the same LDS bank. RDNA has 32 banks at 32-bit each; stride
  your LDS layout to avoid stride-of-32 access patterns.
- **Coalescing** — adjacent threads should hit adjacent global-memory
  addresses, otherwise you eat 2-4× the BW per fetch. Check the
  inner loop's address arithmetic.
- **Wave-size mismatch** — a wave32 kernel running on a wave64 arch
  (or vice versa) silently does the wrong thing computationally OR
  drops half the lanes. CDNA3 (gfx94x) is wave64 native; RDNA is
  wave32. See `case-studies.md` §1.
- **Builtin not available on target** — gfx12 doesn't have the gfx11
  WMMA builtin; rocm 7's clang errors at codegen rather than fall
  back. See `cross-arch.md`.

Tools that help: `--save-temps`, `rocprof` (per-kernel
hardware counters), the per-kernel timer dumps from
`HIPFIRE_PROFILE=1`.

## 3. Pick a lever from `levers.md`

Read [`levers.md`](levers.md). Pick ONE lever that addresses the
diagnosed bottleneck. Don't bundle three speculative changes into one
commit — that breaks the bisect path if any one of them is the
fake-win.

Common matchings:

| Bottleneck | Lever |
|---|---|
| Low occupancy from VGPR pressure | Tighter `__launch_bounds__`, smaller K-tile, or simpler inner loop |
| Wave-size mismatch on CDNA | Wave64 port (`*.wave64.hip` variant) |
| Kernel-launch overhead at small M | Fused projections (qkv → 1 launch) or multi-row variant |
| Per-token weight rereads at decode | Multi-row GEMV (process N output rows per warp) |
| BW-bound on long prefill | WMMA / MFMA (matrix engine throughput beats raw FMA) |
| L2 misses on hot decode weights | `s_prefetch_data` software prefetch (gfx12 only — see PR #56's gemv_hfq4g256.gfx1201.hip) |
| Kernel slow only on one arch | Per-chip override `<name>.gfx1100.hip` or family `<name>.gfx12.hip` |
| Low BW utilization (<10%) + high wave-count from multi-wave WG | Barrier-free syncthreads elimination (levers.md §10) — remove LDS staging, each warp loads from global independently |

## 4. Implement + compile-check across the arch matrix

Author your kernel. Then before benching:

```bash
./scripts/compile-kernels.sh gfx1010 gfx1030 gfx1100 gfx1200 gfx1201
```

This catches the most common cross-arch failure mode: a builtin or
intrinsic that exists on your target arch but not on others. The
script's family-tag handling (`.gfx12.hip` covers gfx1200 + gfx1201)
keeps the fix scoped — see `cross-arch.md`.

If the kernel is per-chip-specific (e.g. `s_prefetch_data` on
gfx1201 only), name it `<base>.gfx1201.hip` so other archs keep
using the family-default `<base>.hip`. The compile script will
respect the override.

## 5. Validate against the three gates

This is non-negotiable. Skipping any of these is how silent
corruption (commit `b7ac66a`, 6 weeks) and fake wins (commit
`0532579`, −13% disguised as +2%) shipped to master.

### Gate A — channel-test (correctness)

```bash
./target/release/examples/test_kernels      # all-kernel synthetic battery
```

For a new fast-path variant on a new arch, ALSO write a dedicated
channel-test example that compares your kernel's output element-
by-element against a validated reference (typically the dot2 or
scalar fallback) on synthetic data. PR #56 is the worked example —
six tests, one per kernel, with row-mod-16 histogram diagnostics
that catch C-mapping row swaps.

### Gate B — coherence-gate (output sanity)

```bash
./scripts/coherence-gate.sh           # AR runtime smoke, when relevant
./scripts/coherence-gate-dflash.sh    # canonical correctness gate
```

Hard fails on panics, zero tokens, timeouts, or attractor-loop
fingerprints. Soft warns on output diffs that need human eyeball.

### Gate C — speed-gate (no regression on baseline arch)

```bash
# CRITICAL: rm the bench exe BEFORE running the gate to bypass the
# stale-binary trap. The gate's `ensure_build` is a no-op when the
# binary already exists, so a "stash and re-run" flow can measure
# the same code twice. See docs/methodology/perf-benchmarking.md.
rm -f target/release/examples/bench_qwen35_mq4
./scripts/speed-gate.sh --fast
```

Tolerance is ±5% from `tests/speed-baselines/<arch>.txt`. If your
change legitimately trades a small regression on the baseline arch
for a much bigger win on another arch, update the baseline in the
SAME commit:

```bash
./scripts/speed-gate.sh --update-baselines
git add tests/speed-baselines/
```

So reviewers see the trade-off explicitly. Don't sneak baseline
updates into a separate "chore" commit.

## 6. Cross-process verify the win

Within-session noise on gfx1100 is ±10–15%. A measurement taken in
the same shell session as your code edits is inside that band — even
if the same code measured "+5% over baseline" three times in a row.
Real wins survive a fresh process.

```bash
./scripts/probe_commits.sh <baseline-sha> <candidate-sha>
```

This rebuilds from clean checkout per commit, runs the bench in a
fresh process, and reports a multi-run median. A delta that survives
this is real; one that doesn't probably isn't.

If you don't have `probe_commits.sh` plumbed for your kernel, do it
manually:

```bash
git checkout <baseline-sha>
cargo clean -p rdna-compute
rm -f target/release/examples/bench_qwen35_mq4
cargo build --release --features deltanet -p hipfire-runtime --example bench_qwen35_mq4
./scripts/speed-gate.sh --fast > before.log

git checkout <candidate-sha>
cargo clean -p rdna-compute
rm -f target/release/examples/bench_qwen35_mq4
cargo build --release --features deltanet -p hipfire-runtime --example bench_qwen35_mq4
./scripts/speed-gate.sh --fast > after.log

diff before.log after.log
```

The `cargo clean -p rdna-compute` + `rm -f .../bench_qwen35_mq4`
combo is the load-bearing part. Skipping either gives you a stale
binary measuring code from a prior run.

## Commit message template

For perf wins or perf reverts, the commit message must include:

- The before/after numbers with binary md5 + prompt md5.
- The hypothesis for why the win works (or why the candidate didn't).
- The bench commands so the next contributor can reproduce.
- For reverts, the bisect commit hash that established the regression.

Example shape (from commit `4105035`):

```
perf(cdna3): full wave64 port of all hot HFQ4 kernels — MI300X decode 48.6 → 96 tok/s

Bisect baseline: <baseline-sha> (record the exact MI300X 9B AR baseline
commit and decode tok/s from your run).
Candidate: this commit (decode 96.0 tok/s on MI300X 9B AR, 2× win).
Bench: HIPFIRE_BASELINE_ARCH=gfx942 ./scripts/speed-gate.sh
Binary md5: <hash>
Prompt md5: <hash>

Hypothesis: ... [explain why wave64 lifts on this hardware]
```

This template is the format the project's perf-recovery commits
(`9a2c667`) and reverts (`34eb024`) follow. Match it.

## Common pitfalls

- **"My change wins +5% in this terminal."** Almost certainly noise.
  See step 6.
- **"It compiled fine on my GPU."** That's one of 6+ supported arches.
  Run step 4.
- **"I wrote a test, it passed."** A test that exercises the
  modified path is necessary but not sufficient — the corollary case
  is the fix touches the dispatch tree, the test still goes through
  the OLD path, and the modified path silently breaks. Verify the
  test actually exercises your code (eyeball the daemon log for the
  expected dispatch print, or temporarily `eprintln!`).
- **"--no-verify just to skip the gate while I iterate."** The gate
  is fast on `--fast` mode. Iterating without it means landing
  regressions you'll spend longer un-bisecting later.
