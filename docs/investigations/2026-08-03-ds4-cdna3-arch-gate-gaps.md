# DeepSeek-V4 on CDNA3 (gfx942): the missing-arch-gate defect class

*2026-08-03 · branch `ds4-cdna-test-fail` · MI300X VF, gfx942, HIP 7.0 / ROCm core-7.14*

## Summary

Bringing DeepSeek-V4 up on MI300X surfaced **five** instances of one defect
class: a wave32-WMMA kernel reachable on CDNA3 with no architecture predicate.
Because `__builtin_amdgcn_wmma_*_w32` requires `gfx11-insts,wavefrontsize32`,
these do not degrade — they fail to **compile**, so the failure lands at JIT in
the middle of a forward pass rather than at model load.

Three are fixed. Two are open, and one of them blocks all batched/long-context
prefill on gfx942.

| # | kernel | status | commit |
|---|---|---|---|
| 1 | `gemm_f16_x_f16_wmma` (F16 compressor path) | FIXED — new wave64 MFMA port | `398c3d176` |
| 2 | `gemm_hfq4g256_wmma` | FIXED — gated on `has_wmma()`, falls to `gemm_hfq4g256` | `692ee9ab6` |
| 3 | `gemm_q8_0_batched` / `wo_per_group_batched_q8_0_1w` | OPEN — no CDNA3 kernel exists; instrumented | `23de0d081` |
| 4 | `gemm_mq2g256_lloyd_moe_grouped_wmma_k2` | OPEN — blocks batched prefill | — |
| 5 | grouped-MoE MFMA path is dead code | OPEN — faults when enabled | — |

`crates/rdna-compute/src/arch_caps.rs:124-126` defines `has_wmma()` as
`is_rdna3 || is_rdna4`, so it is correctly false on CDNA3. The bug is never the
predicate — it is call sites that do not consult one.

## The 3x trunk gap (root-caused)

Same harness, same prompt (md5 `70dd00052d9ff000`), same binary, back to back,
`dspark_bench --AR`, 64 tokens:

```
MQ2R      31.78 tok/s   (repeat 31.77, 31.80)
MQ2-Lloyd 10.62 tok/s   (repeat 10.63)      -> 2.99x
```

Independently corroborated by the DSpark block controller's own fitted cost
model (`crates/hipfire-runtime/src/dspark_block_controller.rs:160-163`):
`t_ar` = 35.5 ms on MQ2R vs 100.1 ms on MQ2-Lloyd.

Per-kernel attribution via `profile_prefill_deepseek4` after adding the missing
timers (`--prefill 13 --warmup 4 --gen 8`):

```
MQ2-Lloyd                                       calls   total_us   GiB/s     %
  gemm_q8_0_batched                               344   174827.3    24.6  68.5
  ...lloyd_moe_gate_up_indexed_batched_k4          43    26992.0   549.9  10.6
  gemv_mq2g256_lloyd_moe_down_expanded_k4          43    14423.0   516.3   5.7
  wo_per_group_batched_q8_0_1w                     43    12782.3   117.0   5.0
  gemm_f16_x_f16_mfma_gfx942                      166    11964.9    77.9   4.7
  TOTAL 255219.9 us

MQ2R: neither Q8 kernel present.  TOTAL 57905.8 us
```

**The two Q8 kernels are 73.5% of MQ2-Lloyd kernel time.**

The decisive datum is intra-process and therefore cannot be attributed to
bandwidth, clocks, or the artifact: in the *same capture on the same GPU*, the
routed-expert GEMVs sustain **549.9 GiB/s** while `gemm_q8_0_batched` manages
**24.6 GiB/s** — a 22x spread between two kernels in one profile.

Cause: MQ2-Lloyd ships hot dense projections as qt=3 **Q8_0**; MQ2R ships the
same tensors as qt=35 **MFP4G32E8SOA** (enforced by the loader at
`crates/hipfire-arch-deepseek4/src/arch.rs:708-710`). Routed experts are qt=19
MQ2-Lloyd in **both** artifacts and take identical kernels — MoE is *not* the
differentiator. There is **no Q8 MFMA kernel anywhere in the tree**, so on CDNA3
every Q8_0 dense weight lands on a scalar kernel launched with block `[32,1,1]`
— a 32-thread workgroup on native wave64, half of every wave idle — plus
byte-wide loads on 34-byte strides and a `float sums[64]` accumulator
predicated 64 ways per weight byte although AR decode only uses `b=0`
(`kernels/src/gemm_q8_0_batched.hip:22-29,41-60`).

### Fix shape (not yet built)

`gemm_q8_0_batched.gfx942.hip`: native wave64, `n==1` specialization (no
`MAX_BATCH` array or predication), coalesced dword loads instead of byte loads,
several output rows per wave. MFMA is not required — at B=1 this is
memory-bound; the win is lane utilization and load width. Same treatment for
`wo_per_group_batched_q8_0`. Iterate against a standalone kernel microbench at
the real shapes, not 3-minute model loads.

## Open gap 4/5: batched MoE prefill is broken on gfx942

Any prefill with `pp_batch` large enough to take the grouped MoE path dies:

```
ffn_batched l0 dispatch: hipcc compilation failed for
gemm_mq2g256_lloyd_moe_grouped_wmma_k2 ... needs gfx11-insts,wavefrontsize32
```

B=1 AR decode never reaches it, which is why `dspark_bench` runs fine. This
blocks the long-context work outright — no prefill, no context ladder.

A CDNA3 replacement already exists and is fully wired:
`kernels/src/gemm_mq2g256_lloyd_moe_grouped_mfma.gfx942.hip`, binding
`gemm_mq2g256_lloyd_moe_grouped_mfma_gfx942`, dispatch arm
`GroupedLloydVariant::MfmaGfx942` (`crates/hipfire-dispatch/src/pipeline/mod.rs:1518`),
and top priority in `select_grouped_lloyd_variant` (`:1462`).

**It is dead code.** Both call sites (`:1764` gate_up, `:1832` down) pass a
literal `false` for the `mfma_gfx942` argument, so the variant can never be
chosen.

Passing `gpu.arch.starts_with("gfx942")` instead clears the compile failure and
then faults:

```
htod n_active_topk_arr: HipError(700): illegal memory access
```

HIP 700 on a memcpy is a sticky deferred fault from an earlier async kernel, so
the MFMA kernel itself is faulting. Grid geometry is **not** the cause — both
variants use `row_tiles=(m+15)/16`, `slot_tiles=(m_total+15)/16` and both call
`ensure_fp16_x`, so the tile contract matches. Leading suspect: the kernel reads
`expert_tile_ids[blockIdx.y]` and guards only `expert_id < 0`, then dereferences
`expert_weight_ptrs[expert_id]` with no upper bound
(`gemm_mq2g256_lloyd_moe_grouped_mfma.gfx942.hip:34-35,46-47`); a stale tile id
>= n_experts yields a garbage pointer. **Unconfirmed** — needs a semantic diff
against the WMMA kernel plus on-hardware iteration.

The enablement was deliberately **not** landed: a clean compile-time refusal is
safer than an illegal memory access, which had already corrupted the rocBLAS
handle by teardown.

## Scope note: what this is and is not worth

These fixes are **CDNA3-only**. gfx1151 already has purpose-built fast paths for
every kernel above — `gemm_q8_0_mmq_4w.gfx1151.hip`,
`wo_per_group_batched_q8_0_wmma_4w` (gfx1151-gated at
`crates/rdna-compute/src/gemv.rs:10886`), and 12 further Q8 WMMA kernels. None
of this work advances the Strix Halo long-context goal directly.

MI300X is also a poor performance proxy for Strix Halo: ~5.3 TB/s HBM3 against
~256 GB/s LPDDR5X unified [spec, unmeasured here] inverts most bandwidth-bound
conclusions. Its unique value is capacity and turnaround — long-context
coherence probes, memory-model validation at 1M, and reference-output
generation — none of which is reachable until gap 4 is closed.

## Also settled

- **MQ2R DSpark works — the restamp was never the problem.** *(Supersedes an
  earlier conclusion in this document, and two records in the tree.)* A sidecar
  stamped via `scripts/reap/hfq_metadata_stamp.rs` initially accepted **0 of
  87** drafts (tau 1.016), matching the historical gfx1151 result (0 of 89).
  That was blamed on the artifact. It was a **missing `match` arm**:
  `dspark_core::gemv_auto_batched_wmma` had arms for F32/Q8_0/F16 and a
  catch-all `_` that decodes everything else as HFQ4-G256, while MQ2R's trunk
  head — which the draft lm_head is bound from — is qt=35 MFP4G32E8SOA. E8-SoA
  bytes decoded as HFQ4-G256 produce garbage draft logits, and
  `accept_greedy_prefix` (`crates/hipfire-runtime/src/spec.rs:149-178`) is exact
  u32 equality, so every draft lost. Fixed in `c420159e2`:

  ```
                    before          after
    accept          0.000 (0/87)    0.582 (32/55)
    tau             1.016           2.065
    DSpark tok/s    19.13           34.70
    AR control      31.78           31.84      -> 1.09x
  ```

  0.582 is the **highest acceptance measured in any configuration**, above
  MQ2-Lloyd's 0.515 on the same sidecar bytes — the expected ordering, since
  MQ2R is the better-calibrated target. Decoded output is coherent and matches
  AR but for one near-tie word.

  Two records in the tree are therefore **misattributed** and should be
  revisited: `registry/deepseek4-mq2r-gfx1151-v2.json` marks the artifact class
  `rejected_mq2lloyd_payload_diagnostic_only`, and
  `docs/specs/2026-07-23-deepseek4-mq2r-e8-recipe.md` records a rejected restamp
  at 7.92 tok/s vs 30.21 AR. The artifact is fine — its 2376 payload tensors are
  recipe-agnostic Q8/F16 (`scripts/quantize-dspark.sh`, `--format
  deepseek4-q8-mtp`). Because the defect is recipe-driven and arch-independent,
  the July gfx1151 measurement is very likely the same bug, not a bad sidecar.
  Whether a *properly P3-calibrated* sidecar beats this one is now an open
  question worth asking again, on a fair basis.
- **The 0731 `-mtp` sidecars are the 3-stage DSpark module**, not MTP: stages
  `mtp.{0,1,2}`, 2376 tensors (791/789/796), no `hnorm`/`e_proj`/`h_proj`. A
  genuine MTP addon is one stage, 797 tensors, with those projections present
  (`crates/hipfire-arch-deepseek4/src/arch.rs:1638-1642`). Both 0731 sidecars
  are byte-identical, sha256 `c123b976…b248`. The quantizer should emit
  `-dspark.<ext>` directly.
- **DSpark on CDNA3 works**: 16.25 tok/s vs 10.63 AR on the MQ2-Lloyd trunk
  (1.53x), tau 2.207, accept 0.515, coherent output. But that is 1.53x on a
  3x-handicapped trunk — *half* plain AR on MQ2R. Speculation numbers measured
  on MQ2-Lloyd are not comparable to MQ2R.
- **A 4-wave K-pipelined rewrite of `gemm_f16_x_f16_mfma.gfx942.hip`** was
  bit-exact (token-identical, 3 fresh processes) but measured 16.17-16.19 vs
  16.24-16.25 tok/s — no gain, reverted. It targeted rank 5 at 4.7% of kernel
  time; the profile above is what should have been gathered first.
- **DSpark is not bit-identical to AR at greedy, and that is very likely
  expected.** On the MQ2-Lloyd trunk at temp 0.00 the two streams agree for 29
  tokens, diverge at exactly one position, then **re-synchronize**:

  ```
  AR      … 34105, 16754,                       22467, 53330, 294, 2900, 14, 1277 …
  DSpark  … 34105,   344, 1949, 850, 5379, 362, 22467, 53330, 294, 2900, 14, 1277 …
  ```

  `AR[30:36] == DSpark[34:40]` exactly. Both paths saw an **identical context**
  through index 28, so the differing argmax at 29 means the verify forward's
  logits differ from the B=1 AR forward's — a reduction-order effect of batch
  shape (AR decode is batch=1 GEMV; verify is batch=n GEMM). Two facts argue
  against an acceptance bug: accept=0.515 shows verification is actively
  rejecting rather than rubber-stamping, and a broken verifier would drift into
  different content instead of rejoining on an identical 6-token run.
  The MFMA port is **exonerated** — bypassing it entirely
  (`HIPFIRE_DEEPSEEK4_COMP_F16_WMMA=0`) yields byte-identical tokens, and the
  pipelined kernel variant reproduced the sequence exactly.
  Still **circumstantial**: forcing the DSpark block to 1 would settle it
  directly, but `HIPFIRE_DEEPSEEK4_SPEC_K` only applies to the MTP branch
  (`examples/dspark_bench.rs:208`) — DSpark takes `block` from the sidecar, so
  that test needs a code change.

## Rig fidelity: MI300X is NOT a faithful proxy for gfx1151 DSpark tuning

MI300X is used as a fast prototyping rig for gfx1151 (hours per iteration instead
of days on a Strix Halo). For that to work, conclusions must transfer. On the Q8
path they currently do **not**, in two independent ways.

Measured with `crates/rdna-compute/examples/bench_q8_0_batched.rs` (`prod` arm =
`gemm_q8_0_batched_chunked`, the entry point production calls), M=4096 K=4096,
identical binary blob `aecb1d7d…` on both boxes:

| B | 1 | 2 | 4 | 5 | 6 | 8 | 13 |
|---|---|---|---|---|---|---|---|
| gfx942 (scalar) | 1589 | 1230 | 934 | **198** | 167 | 150 | **86** |
| gfx1151 (f16 WMMA) | 202 | 368 | 352 | 352 | 349 | 358 | **364** |

GiB/s over Q8 weight bytes. Both are cache-warm (17 MiB working set), so the
ABSOLUTE numbers are not HBM-representative — the **shape** is the point.

**1. Cost curves are opposite.** gfx942 degrades 18x from B=1 to B=13; gfx1151 is
flat (0.045-0.048 ms) from B=2 to B=13, i.e. deeper drafting is nearly free on the
target. The routing that causes this: `gemm.rs:19753` returns `gemm_q8_0_wmma`
whenever `has_wmma() && k % 32 == 0`, and K%32==0 is guaranteed by the Q8_0 group
size — so gfx1151 NEVER executes the scalar kernel, and gfx942 (has_wmma false)
always does.

Consequence: the DSpark block controller fits
`t_window(n) = t_ar + (n-1)*dt` and argmaxes `tau(N)/(t_ar + N*dt)`
(`crates/hipfire-runtime/src/dspark_block_controller.rs:160-163,210-234`). The
`ratio = dt/t_ar = 0.356` measured on MQ2R/MI300X is a rig artifact: the Q8 draft
contribution to `dt` is ~0 on gfx1151 out to B=13. **Block-size tuning done on
MI300X will systematically UNDER-shoot the gfx1151 optimum.** MI300X says "keep
blocks small"; the target says "draft as deep as acceptance allows."

**2. Numerics already diverge, independent of any kernel work.**
`gemm_q8_0_wmma` does not preserve the scalar reduction order or precision, and
the tree's own parity test accepts that: `test_gemm_q8_0_wmma_parity.rs:23-25`
passes at `mean_rel < 2e-3` and `max_rel < 3.5e-2`, explicitly not bitwise. The
harness reproduces it directly — `exact = NO`, max_absdiff 1.9e-2 to 2.7e-2 on
gfx1151, versus bit-identical on gfx942 where both arms are the same kernel.
Since a flipped argmax at a near-tie is an exact-token-id rejection in the accept
path, **tau and accept measured on MI300X are rig-local**. The MQ2R accept=0.582
figure should not be quoted as a gfx1151 expectation.

### What this means for using the rig

- **Transfers:** correctness and structural fixes (the missing E8 arm in
  `c420159e2` is recipe-driven and arch-independent — it should fix gfx1151's
  historical 0-of-89 too), missing-arch-gate defects, instrumentation.
- **Does NOT transfer:** DSpark block size, confidence-truncation threshold,
  adaptive-B behaviour, and absolute tau/accept. These must be measured on
  gfx1151 itself, or the rig needs Q8 cost AND numerics parity first.
- A gfx942 Q8 kernel would fix the cost half but not the numerics half, and an
  attempt at one measured neutral (see below), so parity is not cheap.

### A rewrite that measured neutral, and was not landed

An exact-B-template wave64 port (b1..b6, killing the `MAX_BATCH=64` predicated
accumulator array, one shared `.cuh` body serving both Q8 kernels) was written,
built and measured. It was **bit-exact on every cell** and **0.87-1.01x** — no
gain. The hypothesis it was built on (VGPR pressure from the 64-accumulator
array) is refuted: VGPR allocation is compile-time fixed while `batch_size` is a
runtime argument, so the compiler could never have specialised it away, and
removing it changed nothing. A follow-up probe also refuted X-re-read traffic: at
B=6, M=1024 takes 0.089 ms vs M=4096 at 0.094 ms — near-identical wall time for
4x the output rows and 4x the weight bytes, so above B=5 the cost is essentially
M-independent. **The B=5 cliff on gfx942 remains unexplained** and is the open
question for anyone resuming this.

## The RDNA matrix — and why the CDNA3 Q8 thread is closed

`hipx` carries four RDNA generations, which makes it the real bench for "fastest
engine on RDNA":

| HIP dev | card | arch | VRAM |
|---|---|---|---|
| 0 | RX 7900 XTX | gfx1100 (RDNA3) | 25.8 GB |
| 1 | Radeon 8060S | gfx1151 (RDNA3.5, Strix Halo) | 103.1 GB |
| 2 | RX 5700 XT | gfx1010 (RDNA1) | 8.6 GB |
| 3 | RX 6950 XT | gfx1030 (RDNA2) | 17.2 GB |

(Note HIP device order != `rocm-smi` GPU[n] order.)

Q8_0 `prod` path (`gemm_q8_0_batched_chunked`), M=4096 K=4096, 17 MiB weights,
**ms per call** (lower better), same binary blob `aecb1d7d…`:

| arch | path | B=1 | B=2 | B=4 | B=5 | B=6 | B=8 | B=13 |
|---|---|---|---|---|---|---|---|---|
| gfx1151 | WMMA | 0.045 | 0.045 | 0.045 | 0.045 | 0.045 | 0.045 | 0.045 |
| gfx1100 | WMMA | 0.089 | 0.086 | 0.076 | 0.054 | 0.054 | 0.053 | 0.052 |
| gfx1030 | scalar | 0.204 | 0.259 | 0.358 | 0.409 | 0.461 | 0.599 | 0.889 |
| gfx1010 | scalar | 0.476 | 0.589 | 0.866 | 0.975 | 1.071 | 1.417 | 3.731 |
| gfx942 | scalar | 0.010 | 0.013 | 0.018 | 0.084 | 0.099 | 0.111 | 0.194 |

### 1. The gfx942 B=5 cliff is CDNA-specific — do not chase it

gfx1030 runs the SAME scalar kernel on wave32 and degrades smoothly (2.0x from
B=1 to B=5, against 5x the work — sublinear). gfx942 steps 8.4x over the same
range. The cliff is an artifact of a wave32-shaped kernel (block [32,1,1],
32-lane shfl tree) executing on wave64 hardware, not an algorithmic property.
Since CDNA3 is not a product target, this closes the Q8 CDNA3 kernel effort. The
neutral rewrite was correctly not landed.

### 2. On WMMA parts, deeper speculation is free or profitable

gfx1151 is perfectly flat B=1..13. gfx1100 gets FASTER with batch — 0.052 ms at
B=13 vs 0.089 ms at B=1 — because a 16-wide WMMA tile is 1/16 utilised at B=1.
So on RDNA3/3.5 the Q8 draft cost contributes ~0 (or negative) marginal `dt`,
and **DSpark's optimal block is bounded by ACCEPTANCE, not by cost**. Any
controller fitted on a scalar part, or on MI300X, will select blocks far too
small. This is the actionable RDNA perf lever.

Corollary: AR decode (B=1) is the WORST case on gfx1100 — single-token decode
wastes 15/16 of every WMMA tile.

### 3. gfx1100 is a valid numerics proxy for gfx1151

Both WMMA parts report IDENTICAL deviation from the scalar reference at every
batch (max_absdiff 1.932e-2, 1.937e-2, 2.419e-2 x3, 2.695e-2 x2). RDNA3 and
RDNA3.5 agree with each other exactly; only RDNA-vs-scalar differs. So short-
context correctness work can iterate on the 7900 XTX, and the 96 GiB Strix Halo
is needed only for genuinely long context.

### 4. Role of MI300X, corrected

MI300X is an ORACLE and data-generation vehicle, not a performance proxy — its
cost curve is the inverse of the targets'. Use it for PyTorch reference
generation (`crates/hipfire-arch-deepseek4/reference_oracle/`, teacher artifacts
at `/mnt/scratch/quantization/deepseek-v4-flash-0731-teacher/`), long-context
ground truth, and calibration compute. Do NOT use it to tune block size,
thresholds, or to quote tau/accept.

## CONFIRMED ON TARGET: MQ2R DSpark ACCEPTS on gfx1151 (correctness, not a perf claim)

The E8 draft-head fix (`c420159e2`) was predicted to transfer, because it is
recipe-driven and architecture-independent. **It does** — as a correctness
result. The throughput figure below is harness-local and must NOT be quoted as a
production speedup; see the correction at the end of this section.

gfx1151 (Radeon 8060S / Strix Halo, HIP device 1), 0731 MQ2R trunk
(sha256 `cbf2bbcf…8cce`, verified on-box) plus a metadata-stamped
`-dspark.mq2r` sidecar, certified `dspark_bench`, prompt md5
`70dd00052d9ff000`, 64 tokens greedy:

```
DSpark  tok/s=14.42  tau=2.286  accept=0.538 (35/65)
AR      tok/s= 8.39
```

**The load-bearing number is `accept=0.538`**, against the historical gfx1151
measurement for this artifact class of **0 of 89 accepted, tau 1.016**
(`2026-07-24-dspark-mq2r-p3`). Acceptance is a property of the model and the
draft path, not of the harness, so this result stands on its own.

Therefore `registry/deepseek4-mq2r-gfx1151-v2.json`'s
`rejected_mq2lloyd_payload_diagnostic_only` and the recipe spec's rejected
restamp (7.92 vs 30.21 AR) are **wrong about the cause** on the target hardware:
the sidecar was fine, the missing `MFP4G32E8SOA` match arm in
`dspark_core::gemv_auto_batched_wmma` was the defect. Both records should be
updated. Whether a properly P3-calibrated sidecar beats a restamped one is now
an open question that can finally be asked on a fair basis.

### CORRECTION: the 1.72x is harness-local, not a production number

`dspark_bench` is a research loop, not the production decode path. On the SAME
GPU with the SAME model file and digest, today's product bench via the daemon
measured **24.78 tok/s** at ctx 2048
(`hipx:/home/kaden/ds4-gfx1151-evidence/2026-08-03-ds4-0731-scoregrid-headpair/product-bench.json`,
`us_per_token=40356`, `kv_mode=q8`, `daemon_sha256=b82085af…`), versus
`dspark_bench`'s AR at **8.39 tok/s**. A ~3x harness gap.

Ruled out as causes: KV mode (forcing `HIPFIRE_KV_MODE=q8` changed nothing —
8.39 both ways; at 77 tokens of context KV traffic is negligible) and JIT
(`HIPFIRE_DEEPSEEK4_WARMUP=48` changed nothing). The remaining difference is the
binary and the path: the daemon carries graph capture, retained replay and the
optimised decode loop.

Two consequences:
1. **Do not quote 1.72x.** DSpark has NOT been measured on the production daemon
   path on gfx1151. Against a 3x-faster AR baseline the ratio will compress, as
   it did on MI300X (1.09x against a fast MQ2R trunk vs 1.53x against a slow
   MQ2-Lloyd one). The next measurement must be daemon-side.
2. **`dspark_bench` AR is not a valid perf baseline on gfx1151** and should not
   be used for one. It remains fine for acceptance/tau, which is what it was
   used for here.

Also observed: `.hipfire_kernels/` in that checkout holds a **gfx1100** cache
(261 kernels), so gfx1151 JITs cold, and the startup banner prints the
**gfx942** lever block (`deepseek4 gfx942 A2 levers: … F2 indexer_topk_bounded=OFF
(gfx942-v1 default ON …)`) even when the loader has selected `gfx1151 route v2`.
That banner is misleading on RDNA and cost real debugging time here.

### The rig-fidelity prediction held quantitatively

| | MI300X (gfx942) | gfx1151 |
|---|---|---|
| controller `ratio = dt/t_ar` | 0.356 | **0.119** |
| `t_ar` | 36.0 ms | 121.3 ms |
| `dt` | 12.82 ms | 14.46 ms |
| DSpark vs AR | 1.09x | **1.72x** |
| accept | 0.582 | 0.538 |

Marginal draft cost is 3x cheaper on the target, exactly as the flat WMMA
B-scaling curve predicted — so MI300X UNDERSTATED the win, in the predicted
direction. Acceptance differs (0.582 vs 0.538), consistent with the WMMA-vs-
scalar numerics divergence: acceptance measured on MI300X is rig-local.

### Immediate next lever

`ratio = 0.119` means the block controller's objective
`tau(N)/(t_ar + N*dt)` is very flat in N on gfx1151 — deeper blocks are nearly
free. The sidecar ships `block=5`. Since the Q8 draft path is flat in B out to
at least 13 on this part, the optimal block is bounded by ACCEPTANCE, not cost,
and block=5 is likely leaving throughput on the table. Tune it on gfx1151 —
never on MI300X.

### hipx staging notes

- `/home/kaden/hipfire-ds4-twostage` is NOT a git repo; deploy by rsync and
  verify with `git hash-object` / `sha256sum`. Its `dspark_core.rs` diverges from
  this branch (gfx942-only edits are absent), so patch the E8 arm by ANCHOR, not
  by line number or wholesale copy.
- `hipfire-arch-deepseek4` there has no `deltanet` feature; build
  `-p hipfire-arch-deepseek4 --example dspark_bench` with no `--features`.
- Artifacts staged at `/home/kaden/.cache/hipfire-surgery/`.

## Reproduction

```bash
# per-kernel attribution (needs the timers from 23de0d081)
cargo build --release --features deltanet -p hipfire-runtime \
  --example profile_prefill_deepseek4
./target/release/examples/profile_prefill_deepseek4 <trunk> \
  --prefill 13 --warmup 4 --gen 8

# end-to-end AR A/B
env HIPFIRE_DEEPSEEK4_MODEL=<trunk> HIPFIRE_DEEPSEEK4_MAX=64 \
    HIPFIRE_DEEPSEEK4_AR=1 ./target/release/examples/dspark_bench
```

Trunks: `/mnt/scratch/quantization/deepseek-v4-flash-0731-{mq2lloyd,mq2r-p3}/artifacts/`.

**Do not use `rocprofv3` for this.** `rocprofv3 --kernel-trace --stats` on this
fixture deadlocked at startup — 34 minutes at 0% CPU, 0% GPU, 152 MB RSS, no
output files, model never loaded. The in-tree profiler produced the full
attribution in 3m19s.
