# gfx1201 retained-PM4: our reference configuration and what it proves

Publishing our two reference machines in full — configuration, method, and
results — so anyone seeing a different number can diff against a concrete
baseline rather than guess at ours. Every row below is measured on hardware we
own, not inferred.

## Reference machines

| | hiptrx | k9lin |
|---|---|---|
| GPU | AMD Radeon AI PRO R9700 (gfx1201) | AMD Radeon RX 9070 XT (gfx1201) |
| VBIOS | `00158738` | `023.008.000.068.000001` |
| Power cap | 300 W | 304 W |
| Rated GFX max clock | 2350 MHz | 2400 MHz |
| PCIe | x16 @ 32.0 GT/s (Gen5) | x16 @ 32.0 GT/s (Gen5) |
| RAS: UMC / parity | ENABLED / ENABLED | n/a (consumer part) |
| RAS: GFX / SDMA | DISABLED / DISABLED | n/a |
| CPU | Threadripper 9970X (family 26, Zen 5), 64 threads | Ryzen 9 3900X (family 23, Zen 2), 24 threads |
| Kernel | 7.0.0-27-generic | 6.17.0-35-generic |
| amdgpu driver | **in-tree** (`kernel/drivers/gpu/drm/amd/amdgpu`) | **in-tree** (same path) |
| amdgpu version string | N/A (in-tree reports none) | N/A (in-tree reports none) |
| ROCm | 7.14.0 | 7.14.0 |
| HIP | 7.14.60850-0000000 | 7.14.60850-0000000 |
| Code generator | AMD clang 23.0.0git `46fcb339fb61…` | AMD clang 23.0.0git `46fcb339fb61…` |
| AMD-SMI | 26.5.0+2b22ab01 | 26.5.0+2b22ab01 |

All PM4 policy variables **unset** on both, i.e. committed defaults:
`HIPFIRE_REPLAY_PM4_{QUEUES,STATEFUL,WAIT_POLICY,ACQUIRE_POLICY,GCR_TRIM,NATIVE_PHASES,DYNAMIC_GRID}`.
`ROCM_PATH` / `HIP_PATH` / `HIPFIRE_KERNEL_CACHE` also unset.

## Results

Benchmark contract identical everywhere: `--context 128 --iterations 128
--warmups 10 --warmup-iterations 32 --runs 10 --transport pm4 --kv-mode q8
--max-seq 2048`.

### qwen3.6-35b-a3b.mq4r — `4685c140c4…`

| | HIP | PM4 | ratio |
|---|---:|---:|---:|
| hiptrx | 174.265 | 201.057 | **1.154** |
| clean-room rebuild (hiptrx) | 171.890 | 201.059 | **1.170** |

Route: 733 dispatches / 23 kernels / `3318ffca3daf2338`, identity `[733,1,2,20511]`.

### qwen3.5-0.8b.mq4 — `aedfe31be6…` (registry artifact)

Three repetitions per arm.

| arm | HIP median | PM4 median | ratio |
|---|---:|---:|---:|
| hiptrx, native kernels | 473.283 | **562.964** | 1.186 |
| k9lin, native kernels | 476.597 | **547.325** | 1.147 |
| k9lin, running hiptrx's exact code objects | 476.184 | **548.111** | 1.150 |

Route on both boards: 338 dispatches / 20 kernels / `f3fb9f6309726bb9`,
identity `[338,1,2,9662]`. Within-arm spread ≤0.43%; arms do not overlap.

## Proven

| # | Claim | Evidence |
|---|---|---|
| 1 | The golden baseline reproduces from source | Pristine worktree, cold build, kernels JIT'd from source (mtimes inside the run window, 20.4 s first warm-up vs 2.2 s warm). PM4 201.059, ratio 1.170. |
| 2 | Cached code objects were never stale | Freshly compiled manifest is byte-identical to the cached one: `c111fc7b17e40017…`, 46 files. |
| 3 | Two gfx1201 boards emit the identical tape | 338/20/`f3fb9f63`, `[338,1,2,9662]` on both; `retained_rows=10`, `errors=[]`. |
| 4 | The two boards compile identical machine code | Unbundled `.text` sha256 `7d829812216e4953…`, 896 bytes, on both. The 48 differing container bytes are metadata outside `.text`. |
| 5 | Code generation does not explain the cross-board delta | Running hiptrx's exact bundles on k9lin moves it **+0.14%** (547.325 → 548.111), still 2.6% below hiptrx. Pin verified by hashing the blobs the engine actually loaded. |
| 6 | The retained path is board-sensitive in a way HIP is not | k9lin is 0.70% **faster** on HIP and 2.78% **slower** on PM4. Non-overlapping arms, 3 reps each. |
| 7 | Device selection is not a factor | Reproducing taniguchi's `ROCR_VISIBLE_DEVICES=1 / HIP_VISIBLE_DEVICES=0` on hiptrx passes at 1.158×. |
| 8 | ECC is not costing us throughput | hiptrx runs UMC ECC + parity ENABLED and is the **faster** of our two boards. |
| 9 | PM4 policy, model bytes and wait-policy derivation are identical to the failing report | All 7 policy vars match; model SHA matches; the `PM4 wait audit` line is byte-identical to taniguchi's (732/732 boundaries, 60/130 split, same two `resource_only` pairs at 30 and 40). |

## Not matched, and therefore still open

| Variable | hiptrx | k9lin | taniguchi | HUSRCF |
|---|---|---|---|---|
| VBIOS | `00158738` | consumer | **`00158742`** | unknown |
| amdgpu driver | **in-tree**, no DKMS | **in-tree**, no DKMS | **`6.19.14.31400000`** = packaged DKMS | unknown |
| Kernel | 7.0.0-27 | 6.17.0-35 | unknown | unknown |
| CPU / platform | Zen 5 TR, 64t | **Zen 2 AM4, 24t** | Zen 4 AM5, 16t | unknown |
| perf level / achieved clock | unknown at load | unknown at load | unknown | unknown |
| RAS / ECC posture | UMC+parity on | n/a | unknown | unknown |

### The driver difference is categorical, not a version delta

Both our boxes load amdgpu from
`/lib/modules/<ver>/kernel/drivers/gpu/drm/amd/amdgpu/amdgpu.ko.zst` with an
empty `dkms status` — the in-tree kernel driver, which reports no version
string, hence `amd-smi ... VERSION: N/A`. A reported version like
`6.19.14.31400000` comes from the packaged out-of-tree ROCm DKMS driver.

That is a different driver, not a newer one. amdgpu owns HSA queue
management, command-processor programming, and doorbell/IB submission — the
machinery the retained path depends on and that ordinary HIP dispatch exercises
differently. On the evidence above (identical tape, identical `.text`, pinning
excluded), driver provenance is a stronger candidate than VBIOS for a
path-specific divergence.

We have no in-house DKMS box, so we cannot test this ourselves.

Two caveats on our own control, stated plainly:

- **Our cross-board test confounds GPU with host platform.** hiptrx is Zen 5
  Threadripper; k9lin is a Zen 2 Ryzen 3900X on AM4. The 2.78% PM4 delta cannot
  be attributed to the GPU alone. Notably k9lin is *faster* on HIP, which a
  simply-slower host does not explain.
- **We never observed a ratio inversion.** Both our boards keep PM4 > HIP on
  every run. Nothing we measured reproduces PM4 losing to HIP.

## The two reported failures are not the same failure

| | HIP vs ours | PM4 vs ours | ratio | reading |
|---|---:|---:|---:|---|
| taniguchi | **+1.1%** | **−29.1%** | 0.809 | machine healthy, retained path broken |
| HUSRCF (with #543) | **−16.1%** | −18.4% | 1.122 | both arms down together, retained path working |

HUSRCF's box is uniformly slow with an intact ratio. `perf_level=high` pins the
R9700 at 2350 MHz where auto reaches 2838–3260 MHz — a −17.2% clock deficit
against their measured −16.1% / −18.4%. That fits a clock-constrained board,
not a defect.

taniguchi's box matches our HIP to 1.1% and loses 29% only on PM4. That is
path-specific and is the one unexplained result.

This matters for #543: it was diagnosed and validated on HUSRCF's machine,
whose deficit is most consistent with clocks. The `COMPUTE_PGM_RSRC3` offset
correction is a real bug fix and should land on its own merit, but we have no
evidence it addresses taniguchi's inversion — and measured on hiptrx it costs
**−1.05%** (199.613 → 197.512 median, 4 alternating runs), leaving 0.26% margin
above the 197.0 floor.

## To match our configuration

```bash
# unset every PM4 policy override; committed defaults are the certified route
unset HIPFIRE_REPLAY_PM4_QUEUES HIPFIRE_REPLAY_PM4_STATEFUL \
      HIPFIRE_REPLAY_PM4_WAIT_POLICY HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY \
      HIPFIRE_REPLAY_PM4_GCR_TRIM HIPFIRE_REPLAY_PM4_NATIVE_PHASES \
      HIPFIRE_REPLAY_PM4_DYNAMIC_GRID

# clocks at AUTO — perf_level=high UNDER-clocks the R9700
rocm-smi --setperflevel auto

# then report these alongside any number
amd-smi static --vbios --ras --limit
rocm-smi --showperflevel
amd-smi metric --clock          # sampled DURING the PM4 arm, not at idle
uname -r
```
