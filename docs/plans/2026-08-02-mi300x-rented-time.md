<!-- SPDX-License-Identifier: Apache-2.0 -->
# What to run on rented MI300X time

Date: 2026-08-02 · Status: living · Context: the MI300X is metered

## The selection rule

The MI300X VF is rented by the minute. Two properties make it different from
the rest of the fleet, and **work that does not need one of them should not run
here**:

1. **192 GiB HBM** — 8x a 24 GB RDNA card. Enables the 150.756 GiB original DS4
   checkpoint, long context, and MoE at scale.
2. **CDNA3 matrix cores** — native FP8 MFMA, wave64. A different architecture
   from hipfire's RDNA production target, and the one datacenter deployments
   run on.

Anything targeting gfx11/gfx12 belongs on a local 9070 XT. That includes the
Q8 prefill specialisation, the DDTree gfx1100 regression, MQ3 work, and most of
Redline. Those are important; they are just not worth rented CDNA minutes.

## Ranked

### 1. gfx942 FP8: close the CDNA gap in Radiowave — IN PROGRESS

Radiowave landed gfx11 OCP FP8 lowering on 2026-07-30 (`1d6cfd08a`) with a
bench harness, a correctness methodology, and results for two RDNA parts. There
is **no CDNA row**:

| arch | mode | logical_wmma_M/s | vs FP16 control |
|---|---|---|---|
| gfx1100 | ocp-e4m3 | 355.6 | 1.03x (software decode → FP16 WMMA) |
| gfx1201 | ocp-e4m3 | 1182.2 | **1.96x** (native FP8 WMMA) |

Native FP8 nearly doubles throughput on gfx12. gfx942's MFMA FP8 should match
or beat that and has never been measured.

The blocker is real, not an oversight. `recipes_fp8.rs:16` excludes gfx942
because Radiowave targets **OCP** FP8 while CDNA3 speaks **FNUZ** — different
exponent bias, different special-value encoding — and tests at 630/650/679/682
assert the exclusion.

**DS4 makes it concrete rather than academic:** its checkpoint is
`torch.float8_e4m3fn`, i.e. OCP E4M3FN, sitting on hardware whose FP8 matrix
arithmetic is FNUZ. The unanswered product question is whether hipfire can
serve OCP FP8 checkpoints on CDNA3 losslessly, cheaply, or not at all.

Bounded by the existing harness. Even a well-evidenced "no" is worth the time,
because it closes the question permanently.

### 2. DS4 gfx942 serving performance

CDNA support is real and active — 562 lines in `rdna-compute/src/cdna/gfx942.rs`,
dedicated `*.gfx942.hip` kernels, and **76 gfx942 arch gates** in DS4's
`forward.rs`. So there is a live serving path here whose performance nobody has
characterised against the RDNA numbers the project quotes.

Natural targets: the A3B MoE DFlash line (`AGENTS.md` pins fixtures and a best
observed 151.00 tok/s at tau 2.711), and MQ2R decode throughput. Needs the
model resident, so it wants the memory.

### 3. Long context above 8K

MQ2R was measured to 8192 with no collapse (9.254 / 10.810 / 10.846 / 10.273 at
route_scale 2.0). 16K and 32K need this card and nothing else in the fleet can
host them. Cheap to run, and it either confirms the plateau holds or finds where
it breaks.

### 4. The DS4 parent 12.7x gap — deprioritised

`crate::parent` scores PPL 59.507 against the torch teacher's 4.693 and is
marked NOT A CALIBRATION REFERENCE. Residuals match the teacher to sub-1%,
every measured stage sits at its quantization floor, and the head path is clean,
so the defect is somewhere in layers 3-41. Nothing downstream waits on it: the
torch harness is the teacher now. Resume only if a second implementation becomes
worth having on its own merits.

### Not here

- **Lloyd shrinkage** (`docs/plans/2026-08-02-lloyd-shrinkage-gain.md`) —
  CPU-only.
- **Gates 7-9 / parent-calibrated GPTQ** — gated on a value test, and the bar
  moved to beating PPL 9.254 after the `route_scale` fix.

## Operational notes, learned the hard way

- **One agent per remote checkout.** `/root/hipfire-work/ds4-parent-gate` is
  pinned at an old commit and receives files by copy, so it silently lags. Two
  concurrent agents disagreed about a constant for an hour because one rebuilt
  over the other. Author locally, sync over, never edit only on the box.
- **`pgrep -f` matches the polling script's own command line.** Three separate
  pollers hung today, one for 40 minutes against a process that had finished in
  33 seconds. Use `pgrep -af "[d]s4_..."` or poll for output artifacts.
- **Do not wrap gate binaries in `set -e`.** `ds4_parent_forward_gate` exits
  nonzero when its gate fails, which is not the same as the run failing; it
  silently killed a capture chain.
- **sha256 of an 82 GB artifact costs ~5 minutes** of single-threaded CPU and
  looks identical to a hang from outside. `ds4_quant_plog --trust-sha256` skips
  the re-hash when a campaign already verified the same path and digest.
- **Never touch `/root/hipfire-work/ds4-gfx942-port`** — someone else's
  uncommitted work.
