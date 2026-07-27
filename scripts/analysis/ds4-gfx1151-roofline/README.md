# ds4 gfx1151 AR roofline — analysis scripts (2026-07-27)

Reproduction path for
[`docs/perf-checkpoints/2026-07-27-ds4-gfx1151-ar-roofline.md`](../../../docs/perf-checkpoints/2026-07-27-ds4-gfx1151-ar-roofline.md).

These were authored ad hoc during the campaign and originally lived in an
ephemeral agent scratch directory. They are committed here because the
checkpoint cites their output; without them the numbers are unreproducible.
They are campaign artifacts, not maintained tooling — they hardcode hipx paths
and the MQ2R P3 fixture.

Run from the repo root on hipx with `HIP_VISIBLE_DEVICES=1` (gfx1151, ROCR
index 1 — resolve by the `gfxNNNN` line or 103.1 GB VRAM, never a fixed index).
All `.sh` here acquire the GPU lock themselves.

| Script | Produces |
| --- | --- |
| `floor.sh` | per-dispatch cost floor (`bench_dispatch_floor`) — 1.77 us, flat 1→1024 waves |
| `g4_window2.sh` | window(B) sweep, both arms, AR reference in-process |
| `hcctrl.sh` | `HIPFIRE_HC_CTRL_T1024` A/B, interleaved pairs |
| `hcprof.sh` | rocprof of the T1024 arm + per-kernel table |
| `clk_ab.sh` | perf-level auto/high/manual arms with observed sclk/fclk/mclk |
| `dtod2.sh` | `HIPFIRE_DTOD_DUMP` split into prefill vs per-decode-step |
| `gpubusy2.py` | GPU-busy vs non-kernel gap, separating prefill by call scaling |
| `griddist.py` | grid distribution across ALL dispatches (not the first) |
| `smallk.py` | small-kernel occupancy/launch table |
| `diffprof.py` | differenced two-arm rocprof per-kernel diff |

**`griddist.py` matters more than it looks.** Taking each kernel's *first*
dispatch grid understates starvation badly — `sqrt_softplus_f32` reports 5120
waves on dispatch 1 and 8 waves on 99% of the rest. Weighting every dispatch
moved the sub-one-fill mass from 68% to 92%.

**`diffprof.py` is the technique worth keeping.** Profile two arms with an
identical `--prefix` prefill so the prefill's kernel time cancels in the
per-kernel diff. It predicted a kernel win to 0.2% (predicted −10.38 ms,
measured −10.36).

Raw rocprof traces stay on hipx (workstation-local, ~36 MB each); digests are
recorded in the checkpoint.
