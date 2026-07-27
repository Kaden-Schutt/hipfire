# ds4 gfx1151 AR decode — roofline characterization and two kernel screens

**Workload:** DeepSeek V4 Flash, MQ2R P3
(`/home/kaden/.cache/hipfire-surgery/deepseek-v4-flash.mq2r`, 82,191,362,222 B)

**Hardware:** Radeon 8060S / Ryzen AI Max+ 395 (`gfx1151`), 103.1 GB unified,
hipx. ROCR index 1, rocm-smi index 3, `/sys/class/drm/card1`, pci `bf:00.0`.

**Runtime:** ROCm 7.14 (HIP), rocprofv3 1.3.2

**Mode:** ordinary autoregressive decode, top-k 6, pos 2048. No MTP, no
speculative decoding, no clock pinning.

**Clock policy:** automatic (and see the Rejected/null section — nothing else
is reachable on this part)

**Fixture:** `deepseek4_prefill_bench --prefix 2048 --ar-ref N --tokens 0
--batch 1 --reps 1 --warmup 3`, AR reference measured in-process.

**Date:** 2026-07-27

**Reproduction:**
[`scripts/analysis/ds4-gfx1151-roofline/`](../../scripts/analysis/ds4-gfx1151-roofline/)

---

## Baseline — Measured

```
AR decode wall      35.31 ms/token (28.31 tok/s)   in-process, pos 2048
  GPU-busy          33.788 ms      (2344 launches/step)
  non-kernel gap     1.407-1.812 ms (4.0-5.1% of the token)
bytes/token          5.48 GB
achieved BW          162.7 GB/s busy · 156 GB/s wall
```

AR reference reproduces to 0.08% across runs (35.57 / 35.60 / 35.31 / 35.26 on
separate processes). The banked product number for this route remains
27.70 tok/s under retained PM4; this checkpoint's 28.3–28.5 figures are the
`deepseek4_prefill_bench` in-process AR reference and are **not** a product
claim.

### Byte accounting — Measured

From the HFQ tensor table (`dump_hfq_dtypes`); closes to 0.09% of file size.

| qt | dtype | tensors | elems | bpw | GB (file) |
| --- | --- | ---: | ---: | ---: | ---: |
| 19 | MQ2G256Lloyd | 33,024 (256 exp × 43 L × 3) | 277.0e9 | 2.25 | 77.90 |
| 35 | MFP4G32E8SOA | 554 | 6.742e9 | 4.25 | 3.58 |
| 3 | Q8_0 | 1 (129280×4096, embed+lm_head TIED) | 529.5e6 | ~8.5 | 0.56 |
| 1 | F16 | 641 | 35.8e6 | 16 | 0.07 |

**per token = all-but-experts + 6/256 of experts = 0.07 + 1.826 + 3.582 =
5.48 GB.**

The `qt=3` vocab tensor is the **embedding**, used as a lookup (~4 KB/token),
not streamed — there is no lm_head GEMV anywhere in the 30-kernel decode trace,
and lm_head is one of the 554 `qt=35` E8 tensors. An earlier working figure of
6.04 GB/token double-counted it; an earlier 4.68 GB/token figure is also
superseded. **Historical: do not cite 4.68 or 6.04.**

Independent confirmation: top-k 6→4 removes 10.1% of bytes and predicted +11.2%;
measured +12% (range +9.5% to +14.3%, two runs per arm, `deepseek4_chat`
harness). Also proves the MoE kernels track `cfg.num_experts_per_tok` at
runtime — the `k8` in `gemv_mq2g256_lloyd_moe_gate_up_k8_indexed` is a legacy
symbol name, not a bound (`k_top` is `blockIdx.y`).

### Per-family efficiency — Measured

DRAM ceiling **206–210 GB/s**, measured two independent ways (synthetic
DRAM-resident WAVESWEEP, and the dense E8 GEMV at high wave count).

| family | ms/step | GB | GB/s | vs ceiling | recoverable |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense E8 u4 | 14.403 | 2.810 | 195.1 | 94% | 0.83 |
| dense E8 grouped (wo_a) | 4.367 | 0.772 | 176.8 | 85% | 0.64 |
| MoE gate_up | 5.643 | 1.218 | 215.8 | 104% | 0 |
| MoE down | 2.875 | 0.608 | 211.5 | 102% | 0 |
| small kernels ×26 | 6.559 | 0.070 | 10.7 | — | 3.54 |
| non-kernel gap | 1.407 | — | — | — | 1.41 |

**The two big GEMV families are at 94–104% of ceiling and are done.**

**Max kernel-side recovery = 6.41 ms → 28.67 ms/token = 34.9 tok/s = 191 GB/s.
200 GB/s requires 27.40 ms and is unreachable by kernel work**, because the
dispatch floor (1704 small-kernel launches × 1.77 µs) is 3.02 ms of that budget.
Reaching 200 GB/s requires launching fewer kernels, not faster ones.

### Dispatch floor — Measured

`bench_dispatch_floor` (committed, `crates/rdna-compute/examples/`), null kernel:

| waves | fill | pipelined µs |
| ---: | ---: | ---: |
| 1 | 0.001 | 1.774 |
| 8 | 0.006 | 1.767 |
| 192 | 0.150 | 1.775 |
| 1024 | 0.800 | 1.891 |
| 5120 | 4.000 | 2.642 |
| 20480 | 16.0 | 5.298 |

**Flat at ~1.77 µs from 1 wave to 1024 waves.** A real read-modify-write kernel
runs only 0.19 µs above the empty one at those sizes. 92% of the small-kernel
bucket (6.044 of 6.559 ms) runs below one occupancy fill, weighted across every
dispatch.

---

## Screen 1: `HIPFIRE_HC_CTRL_T1024` — Measured, promotion NOT claimed

`hc_compute_control_vec4_finalize` was the least efficient kernel in the step:
786 KB at 71.8 GB/s (35% of ceiling), because `grid=[n_ctrl=24] × block=[256]`
is 192 waves = 0.15 fills. Widened to 1024 threads.

Interleaved pairs, same process class, fresh process per arm:

| arm | run 1 | run 2 |
| --- | ---: | ---: |
| `T1024=0` | 35.36 | 35.26 ms/token |
| `T1024=1` | 35.06 | 35.10 ms/token |

**−0.23 ms, +0.65%.** Consistent ordering across both pairs; within-arm spread
0.10 and 0.04 ms.

rocprof attribution on a separate run confirms the mechanism rather than a
coincidence:

```
hc_compute_control  0.942 → 0.764 ms/step   (10.95 → 8.88 µs/call)
decode GPU-busy    33.788 → 33.673 ms/step
```

Only 19% of the 0.789 ms theoretical headroom; at 8.88 µs the kernel still runs
88.5 GB/s (43% of ceiling). 24 blocks cannot fill the machine however wide each
is; the rest needs an `x_dim` split (24 → 96 blocks) plus a combine pass.

**NOT bit-exact** — the LDS partial tree widens 8 → 32 and the serial combine at
`tid==0` runs 32 iterations instead of 8. Default **OFF**. A default flip needs
`serve_harness.py` sign-off, not a ULP gate. Commit `d1fe42409`.

## Screen 2: batched E8 GEMVs — Measured, out of current scope

Two kernels landed for the speculative-verify window
(`b8efa47a7`, `7f954ad98`), both bit-exact at B=1 against the decode kernels
they replace (4096/4096 and 8192/8192, max_ulp 0). They collapse
`window(B)` from `107.7 + 12.0·B` to `22.9 + 18.3·B` ms.

**Retained here only as evidence; speculative decode was subsequently placed out
of scope by the operator.** Neither kernel is on the AR decode path and neither
changes AR output. Default off behind `HIPFIRE_DEEPSEEK4_E8_BATCHED_GEMV`.

---

## Rejected / null — do not retry

| Lever | Result |
| --- | --- |
| **Clock / power settings** | `platform_profile` does not exist on hipx. Three arms (auto / high / manual-forced) measured **identical** DRAM bandwidth within 0.5% (206 / 207 / 206 GB/s). Under load `auto` already reaches sclk 2897 of 2900 and mclk 1000 of 1000; the driver **rejects** manual sclk and mclk writes. The Lucebox-advertised settings are a no-op on this part. |
| **wave64** | Strictly dominated. Its one real mechanism here was 2× memory-level parallelism from a single wave; re-gridding to 1024 waves costs the **same 1.77 µs** and yields up to 1024×. Supersedes the inherited gfx1201 "wave64 neutral" note with a measured reason. Also: every `__shfl_down` in the tree hardcodes `offset = 16` for wave32. |
| **rocBLAS / hipBLAS** | Structurally unusable — they consume dense fp16/int8; our weights are MQ2G256Lloyd (2.25 bpw codebook) and MFP4G32E8SOA (E8 lattice). Dequant to fp16 costs ~1.8× **more** bytes on the biggest tier. Fusing dequant into the GEMV is incompatible with the BLAS interface, not an optimization on top of it. |
| **rocWMMA** | Header wrapper over `__builtin_amdgcn_wmma_*` already called directly. Maintenance only, zero perf. |
| **rocPRIM / hipCUB** | ~0.18 ms addressable (`indexer_top_k_buf_parallel` 0.027, `moe_topk_bias_aware` 0.153). The bucket is not primitive-quality-bound. |
| **`mq_rotate_x` re-grid** | 0.855 ms/step, but `grid=[K/256]` with one wave per 256-element FWHT group **is** the call's entire parallelism. Butterflies are register-local + `ds_swizzle`, designed around wave32. Fusion target only. |
| **`copyBuffer` d2d elimination** | AR decode issues only 4.6 `memcpy_dtod_auto`/step (64 KB). The other ~15 copyBuffer dispatches/step come from a path that is **not** `memcpy_dtod_auto` (H2D staging suspected, not instrumented). The two 41.9 MB `hc_mix → copy back` sites (`forward.rs:7308`, `:9486`) are prefill/batched only: 0.6% of prefill. Projected 0.343 ms, actual ~0.02 ms. |

---

## Defect found, not yet fixed

`deepseek4_attn_swa_topk_scoregrid_f32_buf` does `const int h = blockIdx.x;
if (h >= n_heads) return;` but `attention.rs` launches `[head_dim as u32, 1, 1]`.
Correct only because DS4 has `n_heads == head_dim == 64`. Any model where they
differ silently drops heads or wastes blocks.

---

## Raw evidence

Workstation-local on hipx (limitation, not omission — these are ~36 MB each):

```
/home/kaden/g4prof/arA_kernel_trace.csv   md5 be53755b158b9e42152b0291a0b270dc
/home/kaden/g4prof/arB_kernel_trace.csv   md5 3dc911058314e1e8a7badf4cc938c95a
/home/kaden/hcprof/t1024_kernel_trace.csv md5 5bb208c50b7ac939f11110feb3e57f97
/home/kaden/clk_ab.log  /home/kaden/floor2.log  /home/kaden/hcctrl.log
/home/kaden/dtod2_err.txt
```

Committed instruments: `bench_dispatch_floor.rs` (`9b6d2ac48`),
`HIPFIRE_DTOD_DUMP` (`f6254f9ff`), `deepseek4_prefill_bench` gains `--tokens 0`
window mode / `--prefix` / `--ar-ref` / `--e8-batched` (`b8efa47a7`).

## Standing conclusion

The kernels are largely finished; **the remaining AR lever is bytes.** The dense
tier is 65% of bytes/token at 4.25 bpw while experts run at 2.25 bpw. At today's
162.7 GB/s, an MFP3 dense tier projects to 4.64 GB/token = 28.5 ms = 35.1 tok/s
with **no kernel work**. Artifact exists and is unbenchmarked:
`/home/kaden/ds4-mfp3-p1-gptq-v1/standalone/deepseek-v4-flash-mfp3p1.mq2r`
(81.61 GB). That projection is **Exploratory** — it is arithmetic from the byte
accounting above, not a measurement, and the quality question is unanswered.
