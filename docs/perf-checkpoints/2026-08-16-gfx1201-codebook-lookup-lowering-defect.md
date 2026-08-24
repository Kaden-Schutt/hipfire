# gfx1201 — codebook-lookup lowering defect in the MQ*-Lloyd / MQ*-GL kernel family

**Date:** 2026-08-16
**Lifecycle:** `historical` (as every file in this ledger, including the newest)
**Disposition:** Measured microbenchmark evidence. **Not** an admission decision, **not**
a current baseline, **not** an end-to-end model claim.
**Commits:** `e7b0fa56c` (Lloyd fix), `518e09857` (qt=40 wiring), `814d8fd95` (MQ4-GL kernels + harness)

---

## What this record is

A single-shape, single-arch A/B that identifies a **code-generation defect** in the
per-group-codebook GEMV kernels, and the measured effect of removing it. It also
records the first numbers for MQ4-G256-GL (qt=40), a weight format added the same
day.

## What this record is NOT

- **Not an end-to-end decode number.** Every figure below is an isolated GEMV
  microbenchmark on synthetic weights. No model was loaded. qt=40 is not
  dispatch-resolvable at these commits, so no model-level MQ4-GL number exists.
- **Not transferable off this fixture.** One shape, one arch, one hipcc. The
  defect is a property of *this* compiler lowering *this* source pattern on
  gfx1201; gfx1100/gfx1151/gfx942 are unmeasured here and must not inherit it.
- **Not a format verdict.** See "Open, not concluded" below.

---

## Fixture

| | |
|---|---|
| GPU | Navi 48, gfx1201, wave32 |
| Toolchain | hipcc `/opt/rocm/core/bin/hipcc`, HIP 7.14.60850, AMD clang 23.0.0git (`46fcb339fb61` +PATCHED) |
| Compiler boundary | radiowave, `scheduler_profile: default`, wave32 |
| Shape | `m=4096, k=5120` (`gpr = 20`) |
| Weights | deterministic seeded pseudo-Gaussian, FWHT-rotated; **synthetic, not a real quantized checkpoint** |
| Batch | 1 (GEMV), `x` pre-rotated |
| Protocol | warm median, n=30 after 5 warmup, in-process |
| Harness | `crates/hipfire-runtime/examples/mq4gl_parity.rs` |

**Method caveat, load-bearing:** gfx1201 shows cold/warm bimodality on this
harness. The MQ4-GL single-row arm measured ~208–214 µs cold and ~135 µs warm on
identical bytes. All figures below are warm. A single cold sample would have
misattributed a ~55% error to the kernel.

**Lloyd blob caveat:** the harness's MQ4-Lloyd arm is packed by
`pack_mq4_lloyd_for_ab`, which reuses the GL arm's codes and scales so the two
differ only in layout and decode path. It is a bandwidth/decode A/B, not a
quantization-quality comparison, and not a real Lloyd-quantized tensor.

---

## The defect

A 15-deep `?:` chain selecting a 16-entry codebook entry does **not** lower to
conditional moves or a jump table on gfx1201. hipcc lowers it to divergent
control flow with exec-mask save/restore. With 32 lanes each holding an
independent 4-bit code, the select diverges up to 16 ways and serializes.

The source comment in `gemv_mq4g256_lloyd.hip` asserted the opposite —
*"Compiler folds to either a sequence of conditional moves or a small jump table
on RDNA3; either way the lookup stays register-resident"* — and that assertion is
false on this target. The comment, not the code, was the bug's hiding place.

Emitted-object evidence (`llvm-objdump` over the unbundled `hipv4-amdgcn-amd-amdhsa--gfx1201`
object, plus the radiowave `.radiowave.json` inspection block):

| counter | `gemv_mq4g256_lloyd` before | after | `gemv_mq4g256gl` before | after |
|---|---|---|---|---|
| `s_cbranch_execz` | 114 | 4 | 115 | 0 |
| `v_cmpx_lt_i32` | 105 | 0 | 105 | 0 |
| `ds_load_b32` | 0 | 24 | 0 | 9 |
| `static_instructions` | 1115 | 298 | 1091 | 250 |
| `wait_instructions` | 152 | 73 | 130 | 73 |
| `vgpr_count` | 38 | 36 | 25 | 22 |
| `vgpr_spill` / `sgpr_spill` | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

For scale: the pre-fix GL object carried 115 `s_cbranch_execz` and 105
`v_cmpx_lt_i32` against **8** `v_fmac_f32`. The kernel was spending its
instruction budget on control flow, not arithmetic.

**Fix:** stage the codebook into `__shared__ float`, index `cbt[q]`.
Reconstruction semantics unchanged (`w = cb[q]` for Lloyd,
`w = scale * GL_CB4[q]` for GL).

**Barrier note.** The block is `[32,1,1]` = one wave32. Wave lockstep does *not*
provide LDS write → cross-lane-read visibility; a `ds_store` needs a sync before
a cross-lane `ds_load`. `__syncthreads()` is the portable form, and the emitted
object contains **zero `s_barrier`** — hipcc drops the redundant cross-wave
rendezvous for a single-wave block while keeping 23 `s_wait_dscnt`, which is the
instruction actually providing visibility. The sync is load-bearing in source
even though it compiles to waits. A no-barrier variant was measured and was
**not** faster; it must not ship.

---

## Measured

### Effect of the fix

| kernel | B/group | weight bytes | before | after | speedup |
|---|---|---|---|---|---|
| `gemv_mq4g256_lloyd` | 160 | 13.11 MB | ~133 µs | ~48 µs | **2.7×** |
| `gemv_mq4g256gl` | 130 | 10.65 MB | ~135.6 µs | ~48.4 µs | **2.8×** |

Observed spreads across runs: Lloyd before 131.9–134.1; Lloyd after 46.6–52.7;
GL before 135.0–135.8 warm; GL after 46.5–56.2.

### MQ4-GL memory-level parallelism (post-fix)

| arm | B/group | weight bytes | median | GB/s | VGPR | clauses | max VMEM |
|---|---|---|---|---|---|---|---|
| GL single-row | 130 | 10.65 MB | 48.4 µs | 220 | 22 | 2 | 3 |
| GL multirow r2 | 130 | 10.65 MB | 37.6 µs | 283 | 94 | 8 | 8 |
| **GL multirow r4** | 130 | 10.65 MB | **37.4 µs** | **285** | 120 | 14 | 11 |
| GL multirow r8 | 130 | 10.65 MB | 39.9 µs | 267 | 213 | 16 | 32 |
| MQ4-Lloyd (fixed) | 160 | 13.11 MB | 46.6 µs | 282 | 36 | 4 | 2 |
| uniform HFQ4G256 multirow | 136 | 11.14 MB | **36.3 µs** | **307** | 93 (r4) | 16 | 16 |

Spills are 0/0 for every kernel in this table.

### Reading

1. **The branch defect dominated everything else.** Before the fix, both codebook
   kernels sat at 12–15% of this part's achievable bandwidth, and the 18.75%
   weight-byte difference between GL (130 B) and Lloyd (160 B) was invisible —
   the kernels were not memory-bound, so bytes did not matter.
2. **Two hypotheses were tested and falsified** before the real cause was found:
   ALU throughput, and a narrow dependent scale load. Hoisting the entire row's
   scale stream out of the K loop — which does remove one
   `global_load_d16_b16` per group — changed the median by under 1%. Only the
   emitted instruction histogram located the cause.
3. **MQ4-GL remains 2.9% behind uniform on wall time while reading 4.4% fewer
   bytes**, i.e. a 7.2% bandwidth-efficiency deficit. Cause is register pressure,
   not spilling: GL needs 120 VGPRs at R=4 where HFQ4G256 needs 93, and GL's 213
   at R=8 collapses occupancy, which is why R=8 *regresses* rather than improving.
4. **The comparison is not maturity-matched.** HFQ4G256 is the most-tuned weight
   format in this tree (its 32-thread workgroup + `__launch_bounds__(32,16)` shape
   is catalogued in `PRIOR-ART.md`). MQ4-GL's kernels were hours old at
   measurement time. A format conclusion drawn from this table would be a
   kernel-maturity conclusion wearing a format label.

---

## Correctness evidence

- `gemv_mq4g256gl`: F64-referenced parity at `(m,k)` ∈ {(1,256), (3,512),
  (17,2048), (4096,5120)}; max normalized error 1.468094e-7 at the largest shape.
- Multirow: 18 parity checks across R ∈ {2,4,8} including the non-aligned tails
  `m=17 k=256` and `m=4095 k=5120`. Multirow max normalized error 1.488924e-7 vs
  single-row 1.468094e-7 — FMA reassociation across rows, not a semantic change.
- Codebook-order probe: all 16 codes against one-hot `x`, each returning
  `scale * GL_CB4[q]`. This is the only check that catches a `cb`-argument
  permutation, because every wrong permutation still returns a plausible number.
- Edge cases: all-codes-0, all-codes-15, and a group whose fp16 scale is exactly
  zero (must contribute exactly zero).
- `gemv_mq4g256_lloyd`: `test_gemv_mq4g256_lloyd_tail` passes K ∈ {1024…12288},
  max_abs ≤ 2.3e-6 vs CPU reference.

## Codebook provenance

`GL_CB4` was verified against an independent from-scratch Lloyd–Max solve for a
unit Gaussian (standard normal pdf/cdf, no project code): max level deviation
4.8e-5, MSE 0.009501 against the documented 0.009501. The same solver reproduces
`GL_CB2` (0.117482 vs 0.1175) and `GL_CB3` (0.034548 vs 0.03454). The
encoder/runtime copies are machine-checked equal by
`gl_codebooks_match_runtime`, because a drifted codebook decodes every weight to
a plausible-but-wrong level with no error raised.

---

## Open, not concluded

- **Whether MQ4-GL's 4.0625 bpw is a net win is undetermined by this record.** At
  HFQ4G256's 307 GB/s, GL's 10.65 MB would take 34.7 µs — 4.4% faster than
  uniform, matching its byte advantage exactly. Whether that is reachable is open.
- **11 sibling kernels carry the same `?:` pattern** and are unmeasured here:
  `gemv_mq{2,3}g256_lloyd{,_residual}`, `gemv_mq4g256_lloyd_residual`, and
  `fused_{gate_up,qkv,qkvza}_mq{3,4}g256_lloyd`. MQ2's chain is only 4-way and may
  not exhibit the pathology; it must be measured, not assumed. The `.gfx1100` and
  K4 variants already use LDS tables.
- **Non-gfx1201 arches are unmeasured.** `gemv_hfq4g256_multirow.hip` already
  documents that R=1 is near-peak on gfx1100, so the multirow conclusion in
  particular is arch-local.
- **No end-to-end decode measurement exists** for any arm in this record.
