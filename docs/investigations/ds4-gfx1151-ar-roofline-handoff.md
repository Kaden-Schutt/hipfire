# ds4 gfx1151 AR decode — roofline handoff

**Scope: AR decode only.** Speculative decode (MTP, DSpark, G4 batched path) is
explicitly OUT of scope per operator direction. Do not optimize the batched
forward.

Measured 2026-07-27 on hipx / gfx1151 (Radeon 8060S, 103.1 GB, ROCR index 1,
rocm-smi index 3, `/sys/class/drm/card1`). Model: MQ2R P3
`/home/kaden/.cache/hipfire-surgery/deepseek-v4-flash.mq2r` (82,191,362,222 B).

## READ THIS FIRST

`.codeinsight+research/ds4-gfx1151-campaign/ledger.jsonl` (149 entries, active)
is the authoritative record of what has been tried on this route. This brief was
originally written without consulting it and contained a lever the ledger had
already rejected the same day. Check it before proposing anything.

See also
[`docs/perf-checkpoints/2026-07-27-ds4-gfx1151-ar-roofline-amendment-1.md`](../perf-checkpoints/2026-07-27-ds4-gfx1151-ar-roofline-amendment-1.md).

## Established facts — do not re-derive

**Product baseline is 28.079 tok/s** (settled screening, `auto`/retained-PM4
arm, spread 0.149%) per the campaign ledger. The figures below are the
IN-PROCESS `--ar-ref` reference — a micro, not a product claim. GPU-us savings
have measured a ~3% transfer ratio to shipping throughput on this route; do not
convert ms to tok/s without the settled product instrument.

```
AR decode wall      35.08 ms/token (28.50 tok/s)  in-process, pos 2048
  GPU-busy          33.673 ms      (2344 launches/step)
  non-kernel gap     1.407 ms      (4.0% of the token)
bytes/token          5.48 GB
achieved BW          162.7 GB/s busy / 156 GB/s wall
DRAM ceiling         206-210 GB/s  (measured two independent ways)
dispatch floor       1.77 us       FLAT from 1 wave to 1024 waves
```

Byte accounting from the HFQ tensor table (`dump_hfq_dtypes`), closes to 0.09%
of file size:

```
qt=19 MQ2G256Lloyd  33,024 tensors (256 exp x 43 L x 3)  @2.25bpw = 77.90 GB
qt=35 MFP4G32E8SOA     554                               @4.25bpw =  3.58 GB
qt=3  Q8_0               1  129280x4096 embed+lm_head TIED        =  0.56 GB
qt=1  F16              641                                        =  0.07 GB

per token = all-but-experts + 6/256 of experts
          = 0.07 + 1.826 (experts) + 3.582 (dense incl. lm_head) = 5.48 GB
```

The `qt=3` vocab tensor is the EMBEDDING, used as a lookup (~4 KB/token), NOT
streamed. There is no lm_head GEMV in the decode trace; lm_head is one of the
554 `qt=35` E8 tensors. Do not add 0.56 GB to the per-token figure.

## Per-family efficiency — the two big GEMV families are DONE

```
family                    ms      GB    GB/s   vs 207 ceiling   recoverable
dense E8 u4           14.403   2.810   195.1       94%             0.83 ms
dense E8 grouped(wo_a) 4.367   0.772   176.8       85%             0.64
MoE gate_up            5.643   1.218   215.8      104%             0
MoE down               2.875   0.608   211.5      102%             0
small kernels x26      6.559   0.070    10.7        —              3.54
non-kernel gap         1.407                        —              1.41
                                                                  ──────
                                                                   6.41 ms
```

Max kernel-side recovery is 6.41 ms of GPU time. **Do NOT convert that to
tok/s.** The ledger measured 495 GPU us of bit-exact saving producing +0.041%
product throughput — a ~3% transfer ratio — so ms-to-tok/s arithmetic on this
route is Exploratory at best. The earlier form of this section claimed
"6.41 ms -> 34.9 tok/s = 191 GB/s, and reaching 200 GB/s requires launching
FEWER kernels": the first half is unwarranted extrapolation and the second half
is falsified (see lever 1).

## The biggest lever is BYTES, not kernels

Dense is 65% of bytes/token at 4.25 bpw while experts run at 2.25 bpw.

```
                dense/token  total/token   bytes removed vs today
MFP4 E8 (now)     3.58 GB      5.48 GB      —
MFP3 dense        2.74         4.64        -15.3%
MQ2 dense         1.90         3.79        -30.8%
```

Stated as BYTES, deliberately. The tok/s these imply (35.1 and 42.9 at today's
bandwidth) are Exploratory arithmetic, not forecasts — but unlike a scheduling
change, a byte reduction alters the work itself rather than its ordering, which
is the one category the ledger's ~3% transfer finding does not obviously
apply to. That is the argument for measuring it, not a claim about the result.

Artifact already built and unbenchmarked:
`/home/kaden/ds4-mfp3-p1-gptq-v1/standalone/deepseek-v4-flash-mfp3p1.mq2r`
(81.61 GB, 0.58 GB lighter — consistent with most of the dense tier converted).

**Start here.** Bench it against the MQ2R P3 baseline for both tok/s AND
quality; the quality question is the whole reason it was parked. A dense-tier
bpw reduction beats every kernel optimization on this list combined and needs
no kernel work.

## Kernel levers, ranked

1. ~~**Fusion of the small-kernel bucket — 3.02 ms, the only path past 191 GB/s.**~~
   **REJECTED 2026-07-27 by the campaign ledger — do not pursue as written.**
   Ledger entry `2026-07-27-gate-e-swa-compressor-product-rejection` removed 231
   dispatches (2320 -> 2089): full-prefix micro +1.477% (495 GPU us saved),
   settled product +0.041%. The micro magnitude agrees with the 1.77 us floor
   below (231 x 1.77 = 409 us), so the floor is right and the INFERENCE was
   wrong: "launch-count reduction is not the shipping bottleneck for these copy
   kernels once the retained one-IB path is resident". A fusion that also removes
   retained-path waits/acquires is a different, still-open candidate.
   Original sizing retained below for reference only.
   1704 launches/step across 26 kernels, 3.85 us average, floor 1.77 us. 92% of
   the mass runs below ONE occupancy fill. A family runs exactly 1 or 8 waves on
   100% of dispatches: `fused_rmsnorm_mq_rotate_plain_nox` (86/step),
   `rope_tail_yarn_interleaved_wide_f32` (86), `rope_tail_interleaved_f32` (21,
   ONE wave), `deepseek4_moe_topk_bias_aware_f32` (40),
   `rope_tail_yarn_interleaved_at_slot_buf_f32` (62), `rmsnorm_f32_at_slot_buf`
   (62), `deepseek4_fused_silu_mul_clamp_mq_rotate` (43), `sqrt_softplus_f32`
   (43.9), `indexer_top_k_buf_parallel` (21). Halving launch count saves ~1.5 ms.

2. **`deepseek4_attn_swa_topk_scoregrid_f32_buf` — 0.937 ms, 0.86 ms headroom.**
   22.86 us/call, 64 blocks x 512t = 0.8 fills. NOT primarily occupancy-bound:
   ~2.6M MACs at 114 GFLOP/s = 0.8% of peak, latency-bound on the strided gather
   `k_col[d * stride + col]`, worst wait/load ratio in the tape (281/41). Fixing
   it means splitting the POSITION axis across blocks, which needs a cross-block
   softmax (flash-attention online rescaling). Multi-hour rewrite, numerics
   change. Study harness exists: `bench_scoregrid_shape.rs`.

   **Latent bug, fix regardless of perf:** the kernel does `const int h =
   blockIdx.x; if (h >= n_heads) return;` but `attention.rs` launches
   `[head_dim as u32, 1, 1]`. Works only because DS4 has n_heads == head_dim ==
   64. Any model where they differ silently drops heads.

3. **dense E8 grouped (wo_a) — 0.64 ms.** 176.8 GB/s, the furthest-from-ceiling
   of the big families. 43 calls at 101 us. Worth a look before the harder items.

4. **`hc_compute_control_vec4_finalize` — 0.43 ms remaining.** Already improved
   0.942 -> 0.764 ms by widening 256 -> 1024 threads (commit d1fe42409,
   `HIPFIRE_HC_CTRL_T1024=1`, default OFF, +0.65% AR). Still 88.5 GB/s. Further
   gain needs splitting x_dim across blocks (24 -> 96) plus a combine pass.

5. **`rmsnorm_f32` — ~0.30 ms.** 8 waves 65% of dispatches. Single-workgroup
   widening, cheapest remaining item.

6. **non-kernel gap — 1.41 ms.** Nine lever classes already falsified against
   this in earlier campaign work. Low expected value.

## DEAD LEVERS — measured, do not retry

- **Clock / power settings.** `platform_profile` does not exist on hipx. Three
  arms (auto / high / manual-forced) measured IDENTICAL DRAM bandwidth to within
  0.5% (206/207/206 GB/s). Under load `auto` already reaches sclk 2897 of 2900
  and mclk 1000 of 1000; the driver REJECTS manual sclk and mclk writes. The
  Lucebox-advertised settings are a no-op here.
- **wave64.** Strictly dominated. Its one real mechanism was 2x memory-level
  parallelism from a single wave; re-gridding to 1024 waves costs the SAME
  1.77 us and yields up to 1024x. Also every `__shfl_down` reduction in the
  tree hardcodes `offset = 16` for wave32 and would silently produce wrong
  results at wave64.
- **rocBLAS / hipBLAS.** Structurally unusable: they consume dense fp16/int8,
  our weights are MQ2G256Lloyd (2.25 bpw codebook) and MFP4G32E8SOA (E8
  lattice). Dequant to fp16 costs ~1.8x MORE bytes on the biggest tier. Fusing
  dequant into the GEMV is incompatible with the BLAS interface, not an
  optimization on top of it.
- **rocWMMA.** Header wrapper over `__builtin_amdgcn_wmma_*` already called
  directly. Maintenance only, zero perf.
- **rocPRIM / hipCUB.** ~0.18 ms addressable (`indexer_top_k_buf_parallel`
  0.027, `moe_topk_bias_aware` 0.153). Not primitive-quality-bound.
- **`mq_rotate_x` re-grid.** 0.855 ms, but `grid=[K/256]` with one wave per
  256-element FWHT group IS the call's entire parallelism. Butterflies are
  register-local + `ds_swizzle`, designed around wave32. Fusion target only.
- **`copyBuffer` d2d elimination.** AR decode issues only 4.6 `memcpy_dtod_auto`
  per step (64 KB). The other ~15 copyBuffer dispatches/step come from another
  path (H2D staging is the likely source — not yet instrumented). The two 41.9 MB
  `hc_mix -> copy back` sites (forward.rs:7308, 9486) are PREFILL/batched only:
  0.6% of prefill, 0.13% of a B=6 window. Use `HIPFIRE_DTOD_DUMP=1`.
- **`hc_control_rsqrt_once`.** `!mq2r && ...` — the cross-block rsqrt spin-wait
  in `hc_compute_control` is DEAD on the MQ2R route. Do not diagnose it as a
  bottleneck.

## Measurement protocol — what actually works

- **Differenced rocprof arms.** Profile arm A and arm B with an IDENTICAL
  `--prefix` prefill so the prefill's kernel time cancels in the per-kernel
  diff. This predicted a kernel win to 0.2% (predicted -10.38 ms, measured
  -10.36). `deepseek4_prefill_bench` supports `--tokens 0` (window mode),
  `--prefix P`, `--ar-ref N`, `--e8-batched`.
- **Weight grids across ALL dispatches, never the first.** `sqrt_softplus_f32`
  shows 5120 waves on dispatch 1 and 8 waves on 99% of the rest. First-sample
  grids moved the "starved" mass from 68% to 92% when corrected.
- **AR reference in-process.** `--ar-ref N` measures decode in the same process
  and thermal state as whatever else is being measured. Reproduces to 0.08%
  (35.57 / 35.60 across runs).
- Env flags cached in an `AtomicUsize` (not `OnceLock`) can be A/B'd in one
  process against one loaded 80 GB trunk — see
  `set_e8_batched_gemv_max_batch`.

## Traps that cost time this session

- `launch_kernel_blob` takes the grid in WORKGROUPS, not work-items. Passing
  work-items produced a nonsensical 237 us "empty kernel".
- `rsync` + `setsid nohup script.sh` silently no-ops if the script is not
  `chmod +x` — the log lands 0 bytes and nothing runs.
- `pkill -f <name>` over ssh can self-kill the ssh command; a polling loop whose
  own command line contains the pattern will match itself and never exit.
- A fast cargo "Finished" is NOT proof of a no-op build — mold relinks a large
  example in under a second. Check binary mtime against source mtime.
- The JIT kernel cache is keyed by MODULE NAME ONLY. A kernel variant loaded
  under an existing module name is silently dead code. Give variants a distinct
  module AND symbol name.
- rocm-smi device order != `/sys/class/drm/cardN` order != ROCR order. gfx1151
  is rocm-smi 3, drm card1, ROCR 1. Resolve by VRAM size (103.1 GB) or the
  `gfxNNNN` line, never a fixed index.
