<!-- SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Kaden Schutt; hipfire — see LICENSE and NOTICE in the project root. -->

# MQ4G256V2 (qt=44) kernel-family audit — 2026-09-03

_Lifecycle: planned intent. Read-only audit of master `8cd15a62b`. Findings cite `path:line` at that commit._

**Scope.** All 34 HIP translation units that decode the qt=44 header (6 scalar decode,
21 dense WMMA prefill across gfx11/gfx12 incl. BT and MW-LDS, 6 MoE, 1 RDNA3 MMQ; ~8.4k
lines) plus the Rust encoder (`hipfire-quantize`), dispatch (`hipfire-dispatch`),
launchers (`rdna-compute`), replay contracts, and the parity examples. Five read-only
slices (scalar decode, WMMA gfx11, WMMA gfx12 + MMQ, MoE + reachability, Rust side), and
an independent pass by the auditor over every half-select site in the family.

**Oracle.** `docs/quant-formats/mq4-v2.md` §§2–4: 136 B/group, fp16 `(scale, zero)` for
half 0 at `[0..4)` and half 1 at `[4..8)`, nibbles at `[8..136)`; the half-select
predicate must be derived from the kernel's own nibble addressing, never assumed from
lane id — a wrong predicate "compiles, runs, and silently applies the wrong scale to half
of every tensor."

## Verdict

**The format is sound end to end.** No wrong half-select, no scale/zero or half0/half1
swap, no division by scale (degenerate `scale=0` reproduces `zero` exactly everywhere),
no last-group header over-read, no kernel key without a launch arm, WMMA fragment
k-order matches the dequant order on both gfx11 (full-K per lane, C rows `2j+(tid>>4)`)
and gfx12 (`k_grp` K-split, C rows `8*k_grp+j`), MMQ applies scale per 128-K half (not
a per-256 V1-ism), and Redline replay/graph capture pin the fixed base kernels rather
than the adaptive BT policy, as the spec claims.

Half-select sites, verified against their own addressing (auditor's pass, all 34 files):

| predicate | nibble address | files |
|---|---|---|
| `tid < 16` | `gp + 8 + tid*4` (`boff`) | 6 scalar decode, 4 MoE GEMV (`hoff` form) |
| `kt < 8` / `k_off >= 128` / `kt >= 8` | `gp + 8 + kt*8 [+ k_grp*4]` | 20 WMMA main/BT bodies, 2 MoE grouped WMMA |
| `segment < 4` | `gp + 8 + segment*16` (kt0 = 2·segment) | 3 MW-LDS stagers |
| `quarter_in_group < 2` | `gp + 8 + quarter*32 + {0,8,16,24} + k_grp*4` | 2 gfx12 ldsstage bodies |
| `ksc` slot ownership | dm slots 0..3 ← K 0..127, 4..7 ← K 128..255 | MMQ |

MoE is **reachable and correct** on dedicated V2 kernels (loader → `MoeResolution.routed_indexable_mq4v2`
at `crates/hipfire-dispatch/src/families/moe.rs:244` → `pipeline/mod.rs:1244-1262`,
`:1460-1472`, `:1591-1607`, `:3056`, `:3291+`), with a parity harness that uses disjoint
halves and a negative control (`crates/rdna-compute/examples/mq4v2_moe_parity.rs`).

## Broken

1. **`hipfire_runtime::llama::is_batchable_la` is not in lockstep with `qwen35::is_batchable_la`** — high, verified.
   `crates/hipfire-runtime/src/llama.rs:1878-1892` admits `MQ4G256V2` (and qt 47–50) for
   WMMA prefill only on `gfx1200|gfx1201`; `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:1549-1572`
   admits them on gfx11 + gfx12 via `mqv2_gfx11_wmma_enabled_from_env` (kill-switch
   `HIPFIRE_MQV2_GFX11_WMMA=0`). Both doc-comments (`llama.rs:1841-1844`, `prefill.rs:1441-1442`)
   claim the two "match exactly". Effect: plain Llama / Qwen3 dense models carrying qt=44
   prefill per-token on gfx1100/1151 while Qwen3.5/3.8 take WMMA. Fix is a one-line
   admit plus the kill-switch, or a shared function so the lockstep is structural.

## Missing

1. **`mq4v2_gemm_parity` cannot discriminate a wrong half-select** — med/high, verified.
   `crates/hipfire-runtime/examples/mq4v2_gemm_parity.rs:51-63` builds Gaussian
   weights (σ≈0.011) whose two halves have near-identical `(scale, zero)`, so a wrong
   predicate lands inside 4-bit quantization noise — the header comment (`:19-22`)
   promises a "systematic blow-up" the fixture cannot produce. `mq4v2_residual_parity.rs:15-21`
   has the right construction (half 0 in `[-1,1]`, half 1 in `[96,160]`). The batch-size
   sweep design in `gemm_parity` is correct and worth keeping; add a disjoint-halves arm.
2. **Production V2 kernels with no discriminating parity example:** `gemv_mq4g256v2_multirow`
   (R=2/4/8, the default-R path on gfx1151/1201), `gemv_mq4g256v2_residual_sigmoid_scaled_k512`
   (MoE shared-down), the MW-LDS bodies (`*_gfx1100_mw_lds`, `gemm_mqv2_wmma_gfx11_mw_lds`),
   the MMQ path (`gemm_mq4g256v2_residual_mmq` — production on gfx1100/1151 at
   `batch ≥ 128 && batch % 128 == 0`, `crates/rdna-compute/src/gemm.rs:17860-17909`,
   call sites `:26326`, `:26815`, `:27450`, `:28035`), and the gfx1100 decode specials
   (`fused_qkv_mq4g256v2_k2048_x_buffer_gfx1100`, `fused_qkvza_mq4g256v2_k2048_hoist_x32_gfx1100`).
   BT is covered by `test_mq4v2_*_bt_gfx{1100,1151,1201}`.
3. **No test that `mqv2_prefill_batch_tile` (`gemm.rs:124`) only selects tiles the arch's
   launcher accepts.** Policy and launchers agree today (gfx1100 BT4/12 QKV, BT6/12 gate,
   BT4/6/8 residual; gfx1151 BT12 gate, BT4 else; gfx1201 BT8 QKV); drift is untested.
4. **`KernelKey::GemvMq4G256V2SwiGLUResidual` is an alias** (`families/gemv.rs:609` →
   `gemv_hfq4g256_residual_mq4v2`), not a fused SwiGLU+residual kernel; the caller
   pre-fuses. Registry admits a key that has no distinct symbol.
5. **No degenerate-scale (`scale bits = 0`, nonzero nibbles) fixture** through any kernel.
   Arithmetic is `q*sc+zp` everywhere so it is correct by construction; a fixture would
   make that a tested contract.

## Would-change

1. **Spec §9 (`docs/quant-formats/mq4-v2.md`) is stale in three places:** says MoE is
   "out of scope / fail-closed" (it is production-wired, decode + prefill, gfx11 + gfx12);
   says the XBATCH single-row path was "not ported" (`gemv_mq4g256v2.hip:295-361` has it,
   correct); §4 narrative says main WMMA bodies step `kt += 4` (gfx11 bodies step by 2,
   residual by 1 — still correct, each body selects with `kt < 8`).
2. **Stale in-file status comments:** `gemm_mq4g256v2_residual_mmq.hip:10` says
   "Experimental" for a production fast path; `gemm_qkv_mq4g256v2_wmma.gfx12.hip:15-24,169-175`
   still calls its C-map a "HYPOTHESIS"/scaffold while the sibling gate_up file
   (`:30-35`) documents the same map as R9700-validated.
3. **FWHT sign generation is duplicated, and the seeds are convention:** identical LCG
   (`*1103515245 + 12345`) in `hipfire-quantize` and `rdna-compute`; seeds `42`/`1042`
   hardcoded at `crates/rdna-compute/src/scratch.rs:408-409` (`ensure_mq_signs`) and in
   the quantizer/loader paths. Correct today; a drift in either copy is silent logit
   corruption. One shared function and a named constant.
4. **Four dedicated MQ4 gfx11 BT files** (`gemm_{qkvza,qkv,gate_up,residual}_mq4g256v2_wmma_gfx11_bt.hip`)
   duplicate the shared `gemm_mqv2_wmma_gfx11_bt.hip` body, which instantiates
   BITS ∈ {2,3,5,6} but not 4. Likewise two MW-LDS implementations (`*_gfx1100_mw_lds`
   with explicit u32 nibbles vs `gemm_mqv2_wmma_gfx11_mw_lds.hip` which also emits MQ4
   `_gfx11_mw*` symbols). A future half-select or C-map fix has to land in two places.
5. **MoE GEMV header address is lane-dependent** (`*(gp + hoff)`, `hoff = (tid<16)?0:4`)
   in the four `gemv_mq4g256v2_moe_*` TUs, vs the dual-scalar-load + select form the
   dense kernels use. Correct; the spec §4 notes this form turns two scalar loads into a
   vector load. Perf contract, not math.
6. **BT policy keys on exact arch strings** `gfx1100|gfx1151|gfx1201`; gfx1101/1102/1150/1200
   get base WMMA with no adaptive BT. Fine if intentional; not documented as such.
7. **`gemm_mq4g256v2` "plain" GEMM is memset-Y + residual WMMA** (`gemm.rs:29030-29055`);
   the name and registry entry imply a kernel that does not exist.
8. `gemv_mq4g256v2_residual.hip:92-105` macros are still named `HIPFIRE_RESIDUAL_LOAD_SC`/`_ZP`
   but load the two packed half-header dwords; the v1 name invites a future "fix" that
   reinterprets them as f32. `fused_qkvza_mq4g256v2.hip` carries 16+ textual copies of
   the header decode across `PAIR_BUFFER`/`HYBRID_BUFFER`/default branches; all agree today.
9. gfx11/gfx12 WMMA headers decode via `(_Float16)__half2float(__ushort_as_half(...))`
   (fp16→f32→fp16) where a bitcast would do; BT N-tail gathers `X` from batch 0 and
   discards it instead of zero-filling. Both perf nits.

## Confirmed by design (not findings)

- gfx1030 with qt=44: per-token decode fallback, no WMMA (`is_batchable_la` false,
  `gemm.rs:29055-29061` errors without `has_wmma()`); the spec §6 R=1 regression is not
  a live path. The "gfx1030 default-R decision" in §9 can be closed as moot.
- No wave64 V2 decode path exists in any of the 34 files; every launch is wave32
  (`__launch_bounds__(32, …)`, or 32·NW). Spec §9's "wave64 unverified" is therefore
  "wave64 unsupported", which is what `dtype_arch_predicate → HasWave32` enforces.
- No gfx12 MMQ for qt=44 (HFQ4 has one); gfx12 stays on fp16 WMMA. Intentional per
  the RDNA3-only guard (`residual_mmq.hip:92-124` stubs otherwise).
- Encoder (`crates/hipfire-quantize/src/quant_fwht.rs:200-253`): fp16 round-trip before
  quantizing, `degenerate = hi==lo || step==0 || st==0` → all `q=0`, so an fp16
  underflow of the step cannot produce `inf` codes.

## Not read

- The v1 (`*_hfq4g256*`) sisters were not line-diffed; the port was audited against the
  spec, not against its source.
- Mixed-tier MoE TUs (`gemv_mixed_moe_*`, tags 7–18) beyond a spot-check of the
  `_MQ4V2G` macro; Muse/DS4 arch crates' MoE call sites end to end.
- HIP parameter lists were not compared to the Rust kernarg blobs line by line
  (representative orders recorded in the Rust slice; a mismatch would show in parity).
- Fixture contents of `test_mq4v2_*_bt_gfx*` (names and coverage only).
- Nothing was built or run; this is a source audit. Parity examples exist for every
  route above except those listed under Missing 2 — running them on gfx1100/1151/1201
  is the hardware confirmation this audit does not provide.

## Recommendation

One code fix (Broken 1 — shared `is_batchable_la` or a matching admit), one test PR
(disjoint-halves arm in `mq4v2_gemm_parity`; parity for multirow / MW-LDS / MMQ /
sigmoid-scaled k512; a `mqv2_prefill_batch_tile` ↔ registry table test), and a docs pass
(spec §9, the two in-file status comments). None of it blocks the format; qt=44 can stay
the production default.
