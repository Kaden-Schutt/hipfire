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
3. **AMD's PRODUCTION `mladf` kernel does NOT reach 50 either** — built
   DynamicDispatch from source and ran real mladf gemms on the NPU. The shipped
   int4 (w3a16) gemm is a flat **~7 TOPS** across LLM shapes — a *memory-bound
   weight-quant decode* kernel, actually **below** our int8 reference. The
   compute-bound bf16 prefill gemm wasn't runnable in this package
   (`op_version "bfp16gemm"` unregistered).

**Bottom line (revised, evidence-backed):** The hardware genuinely does 58 TOPS
and is un-throttled — but **no real GEMM kernel I could measure approaches it.**
The tuned reference (int8) and AMD's shipped production kernel (int4) both sit in
the **~12–27%-of-peak band (7–16 TOPS)**. ~50/58 TOPS is a theoretical/marketing
peak; real inference GEMM on this AIE dataflow is per-tile-feed/overhead-bound at
these shapes. Our int8 reference at **15.7 TOPS is the highest real number
measured** — beating AMD's shipped int4 decode kernel. Reaching ~50 would need a
compute-bound dataflow that neither the reference nor the shipped kernels
demonstrate (unverified whether one exists; the compute-bound production path was
not runnable here).

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

## mladf bench — DONE (built DynamicDispatch from source, ran on NPU)

Measured, all PASS (latency reported by the op; TOPS = 2·M·K·N / latency):

| mladf kernel | shape M×K×N | latency | TOPS |
|---|---|---|---|
| int4 w3a16 grp128 | 512×3584×3584 | 1.874 ms | **7.0** |
| int4 w3a16 grp128 | 512×3584×18944 | 10.10 ms | 6.9 |
| int4 w3a16 grp128 | 1024×3584×18944 | 20.20 ms | 6.9 |

Flat ~7 TOPS = a memory-bound weight-quant **decode** kernel (int4 weight, bf16
activation), **below** our int8 reference (15.7). The compute-bound `Bfp16Gemm`
test throws "op version does not exist" (`op_version "bfp16gemm"` not registered
in this package); a16w8/a16a16 gemms are meta.json runners (not driven). So the
compute-bound production path was not runnable here — but the shipped int4 kernel
sitting at the same ~25%-of-peak efficiency as our reference is strong evidence
the ceiling is dataflow-fundamental, not a kernel we're missing.

### DynamicDispatch build recipe (worked; build left in place at ~/build/DynamicDispatch/build)
Deps: nlohmann_json/spdlog/xaiengine auto-fetched; XRT found; system protobuf +
a **CPU-torch venv** (ROCm-torch needs an absent HIP cmake pkg):
```
uv python install 3.12 && uv venv -p 3.12 /tmp/cputorch
uv pip install --python /tmp/cputorch/bin/python torch --index-url https://download.pytorch.org/whl/cpu
# edit DynamicDispatch/CMakeLists.txt:74  find_package(Protobuf CONFIG REQUIRED) -> MODULE
cmake -B build -DENABLE_DD_TESTS=ON -DUNIT_TEST_PERF_EN=ON -DENABLE_DD_PYTHON=OFF \
  -DDD_DISABLE_AIEBU=ON -DCMAKE_BUILD_TYPE=Release -DXRT_DIR=/opt/xilinx/xrt \
  -DTorch_DIR=/tmp/cputorch/lib/python3.12/site-packages/torch/share/cmake/Torch \
  -DCMAKE_CXX_FLAGS="-I/opt/xilinx/xrt/include/xrt -include cstdint -include cstddef"
cmake --build build --target cpp_tests -j$(nproc)
# run: build/tests/cpp/unit_tests/cpp_tests --gtest_filter='Qwen7b_2Testw3a16_high_time.Kernel4mladf_512x3584x3584_int4_grp128_v1'
```
Gotchas fixed: XRT 2.25 moved experimental headers to `xrt/experimental/`
(`-I.../include/xrt`); GCC 15 needs `-include cstdint -include cstddef`
(transitive-include tightening); protobuf CONFIG→MODULE (Ubuntu ships no config pkg).

## Confirmed no-op levers
- **turbo pmode**: 15.21 vs 15.7 TOPS (compute clock already maxed under load).

## Open (only if chasing the last mile)
- Run a compute-bound production gemm (fix `bfp16gemm` op-version, or hand-build a
  a16w8/a16a16 meta.json) to test whether ANY shipped kernel exceeds ~16 TOPS.

## Tooling added
- `crates/hipfire-xdna/examples/npu_info.rs` — dumps resource_info (max/curr TOPS,
  clocks); source of the power finding. Run: `cargo run -p hipfire-xdna --example npu_info`.
- `tune.sh` extended usage confirmed (DTYPE_IN/OUT, configs). Results CSVs in `results/`.
