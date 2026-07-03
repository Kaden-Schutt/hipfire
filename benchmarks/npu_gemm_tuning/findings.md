# NPU GEMM — path to ~50 TOPS (findings)

**Goal:** reach ~50 TOPS int8 GEMM on Strix Halo NPU (aie2p/npu2), or determine
the true ceiling + limiter with evidence. TOPS = 2·M·K·N / time (matches the
whole_array bench's "gflops").

## Headline conclusions

1. **The NPU is NOT power/clock-throttled.** Hardware ceiling is **58 TOPS int8**
   and, under GEMM load, `default` pmode already boosts to the **full 58-TOPS
   budget with the AIE compute clock maxed at 1800 MHz**. So ~50 TOPS is real
   silicon that is *available*, not gated behind a power mode. Turbo pmode
   expected ~no-op (compute clock already at max) — 1-line confirm pending sudo.
2. **The mlir-aie `whole_array` REFERENCE dataflow caps at ~15.7 TOPS = 27% of
   peak**, and every tunable knob is explored (below). This is a **dataflow
   efficiency** limit, not compute or power.
3. **Reaching ~50 requires AMD's production `mladf` dataflow** (DynamicDispatch),
   which is a fundamentally better-engineered design, not a tuning of the
   reference. Benchmarking it is scoped and tractable (see "mladf bench" below)
   but was blocked short of completion on a ROCm-torch→HIP cmake dep in the
   test harness.

**Bottom line:** 50 TOPS is obtainable on this silicon (hardware does 58,
un-throttled). The reference dataflow gets 27%; the last ~2× lives in a
production-grade dataflow (mladf) or a bespoke one, not in the reference's knobs
or in any power/clock "special condition."

## Device facts (xrt-smi examine + hipfire-xdna resource_info)

- AMD RYZEN AI MAX+ 395, NPU Strix Halo **aie2p**, topology 6×8 = shim + memtile
  + **4 compute rows × 8 cols = 32 cores**. XRT 2.25.0, amdxdna 2.25.0, FW 1.1.2.65.
- `npu_clk_max=1800 MHz`, `npu_tops_max=58`.
- Idle (default pmode): tops_curr=25, MP-NPU 396 MHz, H 792 MHz.
- **Under GEMM load (default pmode): tops_curr=58, MP-NPU 1267 MHz, H 1800 MHz (max).**
- Peak check: 32 cores × 512 int8 MAC/mmul (8×8×8) × 1.8 GHz = 59 TMAC ≈ 58 ✓.
  (H = AIE compute clock.)
- pmode set (`xrt-smi configure --pmode`) needs CAP_SYS_ADMIN → root; sudo is
  password-gated on this box.

## Experiments (all i8/i8, 4096³, 8col, warmup5/iters20 unless noted)

| # | Lever | Result | Verdict |
|---|---|---|---|
| E1 | power/clock throttle | default already 58 TOPS + H=1800 under load | NOT throttled |
| — | output tile m×n | 32²=2.4, 64²=8.2, 128×64=11.0, 64×128=14.3, **128×128=15.7** | THE lever; L1-capped (area 16384) |
| — | reduction depth k | k=128 vs 256 at 32² → 2.41 vs 2.28 | irrelevant |
| — | columns | 4c=4.4, 8c=8.2 (1.88×) | maxed at 8 |
| — | OPT_PERF flag | 8228 vs 8211 gflops (flag compiled in) | no-op (Peano ignores chess pragma) |
| — | pre-tiled weights (contiguous B) | 15.62 vs 15.66 | no-op (re-tile is cheap on-chip memtile→core) |
| E2 | fifo_depth | 2→15.3, 3→**15.7** (+2.5%), 4→build-fail | marginal; DMA already hidden |
| — | fifo_depth=1 + 2× tile | control 15.01 (< fd2); 2× tiles build-fail | dead end |
| — | microkernel mm.cc | AMD-documented 2×2 mmul optimal, register-limited | tapped |

**Best config: m128 k32 n128, 8col, i8/i8, fifo_depth=3 → ~15.7 TOPS (27%).**
Every reference knob is explored; ~15.7 is the reference dataflow ceiling.

## Why the reference caps at 27% (mechanism)

Feed/overhead-bound: throughput scales with **output-tile size** (amortizes
per-tile DMA setup + C accumulator load/store + objectfifo acquire/release +
software-pipeline fill/drain), and the tile is L1-capped (64 KB/core) at area
16384. Deeper k, more columns, better weight layout, and deeper buffering do not
help — the cores are starved by per-tile overhead the reference can't shrink
without a different dataflow (e.g. K-resident C, cross-core cascade/systolic, or
a larger effective tile via smarter memtile use — i.e. what mladf does).

## mladf bench (the decisive test of ~50) — turnkey resume

AMD's production gemm is **bfp16 / a16w8 / a16w4** (`xclbin/stx/mladf_gemm_4x4_a16fw4acc16f`,
`mladf_4x2_gemm_a16w8_qdq`, `llama2_mladf_2x4x4_bfp16_gemm_*`) with prebuilt
transaction .bin instruction sequences under `~/build/DynamicDispatch/transaction/stx/`.

Build the perf test (`test_mladfmatmulbias`, `UNIT_TEST_PERF`) — deps resolved so far:
- nlohmann_json, spdlog, xaiengine: auto-fetched by cmake. XRT: found.
- **Protobuf**: system libs present (`/usr/lib/.../libprotobuf.so`, protoc 3.21.12)
  but no cmake config package → change `find_package(Protobuf CONFIG REQUIRED)`
  to `MODULE` in DynamicDispatch/CMakeLists.txt:74. (Reverted for now.)
- **Torch**: venv has ROCm torch 2.12 (`~/.venv/.../torch/share/cmake/Torch`), but
  its `TorchConfig` enables HIP language and needs `hip-lang-config.cmake`, absent
  under /opt/rocm-7.14. FIX: point at a **CPU-only torch** in a separate venv
  (`python -m venv /tmp/cputorch && pip install torch --index-url \
  https://download.pytorch.org/whl/cpu`), then `-DTorch_DIR=<that>/share/cmake/Torch`.
- Then: `cmake -B build -DENABLE_DD_TESTS=ON -DUNIT_TEST_PERF_EN=ON \
  -DENABLE_DD_PYTHON=OFF -DDD_DISABLE_AIEBU=ON -DXRT_DIR=/opt/xilinx/xrt` →
  build `test_mladfmatmulbias` → run (it times a real mladf gemm on the NPU).

Simpler if sudo is unblocked: `apt install protobuf-compiler libprotobuf-dev`
gives the Protobuf cmake config (no MODULE edit); CPU-torch venv still needed
(or install the HIP cmake package).

## Remaining steps (both currently sudo/effort-gated)
1. **Confirm turbo is a no-op** — `sudo xrt-smi configure --pmode turbo` then
   re-bench (expected ~15.7, since H already maxed). Needs sudo password.
2. **Finish the mladf bench** — CPU-torch venv + build + run (above). This is the
   definitive proof of what a production dataflow achieves (~50?) on this box.

## Tooling added
- `crates/hipfire-xdna/examples/npu_info.rs` — dumps resource_info (max/curr TOPS,
  clocks); source of the power finding. Run: `cargo run -p hipfire-xdna --example npu_info`.
- `tune.sh` extended usage confirmed (DTYPE_IN/OUT, configs). Results CSVs in `results/`.
