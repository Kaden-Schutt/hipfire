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

- **MQ2R DSpark by metadata restamp is dead.** A sidecar stamped via
  `scripts/reap/hfq_metadata_stamp.rs` loads and passes
  `validate_mq2r_dspark_sidecar`, then accepts **0 of 87** drafts (tau 1.016),
  reproducing the historical gfx1151 result (0 of 89) on a second architecture
  and a newer checkpoint. `registry/deepseek4-mq2r-gfx1151-v2.json` already
  records the class as `rejected_mq2lloyd_payload_diagnostic_only`; that
  rejection holds on CDNA3. A usable MQ2R DSpark needs a real P3-calibrated
  build. The diagnostic artifact was deleted rather than left as a footgun.
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
