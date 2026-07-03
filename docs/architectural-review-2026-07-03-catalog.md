# hipfire Architectural Review — Findings Catalog (2026-07-03)

Companion to `architectural-review-2026-07-03.md`. Every finding was produced by a subsystem reviewer and independently re-checked by an adversarial verifier that re-read the cited code. Verdicts: **confirmed** (facts re-checked), **adjusted** (core claim real; location/numbers/severity corrected as noted), **refuted** (dropped, listed at the end).


## rdna-compute — GEMM/GEMV dispatch family

*Subsystem key: `rdna-dispatch-gemm` — 8 finding(s)*

**Subsystem assessment:** The GEMM/GEMV dispatch family (gemv.rs 6121 LOC, gemm_qkv.rs 5890, gemm_hfq.rs 2916, gemm_gate.rs 2603, fused.rs 4308, plus gemm_base/gemm_misc) is a single god-object: every file is one `impl Gpu` block, contributing to ~803 pub fns across dispatch/. Each dispatch method is a near-verbatim template (bind_thread -> ensure_kernel -> build kernarg list -> launch_maybe_blob -> timer), and critically every one of ~207 launch sites maintains TWO parallel argument lists (a `Vec<*mut c_void>` and a duplicate `KernargBlob` closure) with no compile-time or test check that they agree. Kernel-selection cascades (which quant/tile/arch kernel to run) are the highest-risk logic yet live inline, coupled to GPU state, and are entirely untested: the whole family has zero `#[test]`. Leaf arch predicates in arch_caps.rs are pure and well-tested (19 tests), so the composite routing is the extractable, testable gap. Main risks are silent kernarg drift, routing-cascade drift (including one dead duplicate selector), and truncating `as i32` casts on unvalidated dims.

### [High] duplication — `crates/rdna-compute/src/dispatch/gemm_hfq.rs:1891-1935 (representative of ~207 sites)`

**Observation:** Every kernel launch maintains two hand-written, order-sensitive copies of the same argument list: a `params: Vec<*mut c_void>` built with `&mut x as *mut _ as *mut c_void` casts, and a second `KernargBlob` closure that re-pushes the identical args via push_ptr/push_i32. The two must match each other AND the HIP kernel signature exactly, but nothing enforces this — a reorder or a forgotten arg in one list produces silent kernarg corruption, not a compile error. There are ~207 launch_maybe_blob call sites (gemv 81, fused 47, gemm_qkv 24, gemm_misc 20, gemm_hfq 14, gemm_gate 13, gemm_base 8) and gemv.rs alone has 476 push_* lines.

**Recommendation:** Introduce one kernarg abstraction that emits both representations from a single declaration, e.g. a `kernargs![a_ptr, x_ptr, y_ptr, m_val:i32, k_val:i32]` macro that expands to both the `Vec<*mut c_void>` and the `KernargBlob`, or fold the blob-building into `launch_maybe_blob` by having it take a `&[KernArg]` slice and derive the pointer vec internally. This removes the dual-maintenance hazard entirely and shrinks each dispatch fn by ~20 lines.

**Evidence:** 207 launch_maybe_blob sites; 207 matching KernargBlob::new() closures; 2454 `as *mut c_void` casts across the 7 files; example params-vec at gemm_hfq.rs:1891-1898 duplicated by push_* at 1927-1934.

*Verification: confirmed*

### [High] missing-tests — `crates/rdna-compute/src/dispatch/{gemv,gemm_qkv,gemm_hfq,gemm_gate,fused,gemm_base,gemm_misc}.rs (0 tests); gemm_qkv.rs:844-914`

**Observation:** The kernel-selection cascades — the logic most likely to harbor a correctness bug (e.g. 'gfx1103 + batch 32 + M=4096 + hfq4 → which kernel?') — have zero test coverage. All 7 scope files contain 0 `#[test]`; the entire dispatch/ tree has exactly one test (mod.rs:5131, covering gen_fwht_signs, unrelated to routing). The atomic predicates in arch_caps.rs (is_gfx906, should_use_mmq, has_wmma_w32, etc.) ARE pure and have 19 tests, but the composite decision tree that combines them is untested and unpurified: the selector at gemm_qkv.rs:844-914 calls self.bind_thread_or_warn() (845) and self.mmq_screen_weight() (856), which uploads synthetic activations and runs kernels (mod.rs:1952-1991), so it cannot run without a GPU.

**Recommendation:** Extract the routing cascade into a pure free function per family, e.g. `fn choose_qkvza_kernel(caps: &ArchCaps, flags: &Flags, shapes: QkvShapes, batch: usize, screen: ScreenOutcome) -> QkvzaKernel`, where GPU-dependent inputs (the mmq_screen_weight result, rocblas availability, capture_mode) are passed in as plain values/enums. The dispatch fn then does the GPU work, calls the pure selector, and matches on the returned enum. This makes the full arch×shape×format decision table unit-testable on CI with no GPU and directly satisfies the RDNA2/3/4 portability constraint by letting each arch's routing be asserted in a table test.

**Evidence:** grep -c '#[test]' = 0 for all of gemv/gemm_qkv/gemm_hfq/gemm_gate/fused/gemm_base/gemm_misc; only test in dispatch/ is mod.rs:5131; arch_caps.rs has 19 #[test]; selector gemm_qkv.rs:845 calls bind_thread_or_warn, :856 calls mmq_screen_weight (GPU upload at mod.rs:1975).

*Verification: confirmed*

### [Medium] coupling — `crates/rdna-compute/src/dispatch/gemm_qkv.rs:832-915 vs 916-1263`

**Observation:** `gemm_qkvza_hfq4g256_route_label` (832-915) is a `pub fn` that reproduces the exact routing cascade (rocblas eligibility, is_gcn5_wave64, should_use_mmq, has_hfq4_mmq, has_wmma_w32, dot2, fp16 fallback) of the real dispatch fn `gemm_qkvza_hfq4g256` (916-1263), returning the chosen kernel name string. It has ZERO callers anywhere in the crate, so it is dead code that must be manually kept in sync with a 348-line sibling — a guaranteed source of routing drift where the label reports one kernel while dispatch runs another.

**Recommendation:** This is the symptom that the pure-selector refactor (previous finding) cures. Make both the label path and the dispatch path call the same pure `choose_qkvza_kernel(...) -> enum`, then derive the label from the enum via a single `impl Display`/`as_str`. Until then, if the label fn is genuinely unused, delete it; if it is intended for telemetry, wire it to the real cascade so one edit updates both.

**Evidence:** gemm_qkv.rs:832 defines the fn; crate-wide grep for `_route_label` outside its own definition returns no callers; cascade branches at 846-914 mirror dispatch branches at 939-1180.

*Verification: adjusted — Verified: route_label (832-915) is pub with ZERO callers crate-wide (only its own definition), and it reproduces the real dispatch cascade (rocblas@939/846, is_gcn5_wave64@998/854, is_gfx906+should_use_mmq@1022/855, should_use_mmq@1092/876, has_hfq4_mmq@1135/889, wmma@1158/902, dot2@1172/905). Duplication and drift risk are real. Downgrading to Medium: with zero callers nothing consumes the label output, so there is no ACTIVE correctness risk (the 'label says X, dispatch runs Y' scenario needs a consumer that does not exist) — it is dead duplicated code, a clear antipattern worth scheduled removal/refactor, not High.*

### [Medium] monolith — `crates/rdna-compute/src/dispatch/gemm_qkv.rs:916-1263 (and siblings)`

**Observation:** `gemm_qkvza_hfq4g256` is a single 348-line function that mixes four responsibilities: rocBLAS fp16-shadow prefill path (939-992), MMQ screening + gfx906 wave64 split routing (996-1180), WMMA/dot2/fp16 fallback selection, and the actual kernarg build + launch. It uses raw `DeviceBuffer::from_raw` + `std::mem::forget` lifetime juggling (952-986) inline. Peers are similarly oversized: gemm_qkv_hfq4g256 is 252 lines (1961), gemm_qkvza_hfq6g256 142 lines (4670). These are hard to review and impossible to test in pieces.

**Recommendation:** Split each along its natural seams: a pure selector (see above), a `rocblas_qkvza_prefill(...)` helper owning the shadow/forget dance, and thin per-kernel launch fns that only build kernargs. The rocBLAS interop and fp16-shadow lifetime handling in particular should be one audited helper rather than copy-inlined into every projection variant.

**Evidence:** awk span: gemm_qkvza_hfq4g256 runs 916→1263 (348 lines); next fn starts at 1264; DeviceBuffer::from_raw + mem::forget at gemm_qkv.rs:952-986.

*Verification: confirmed*

### [Medium] abstraction — `crates/rdna-compute/src/dispatch/gemv.rs:12 (impl Gpu, 124 pub fns) — family-wide god object`

**Observation:** The entire GEMM/GEMV family is one inherent `impl Gpu` split across 7 files (each opens `impl Gpu {`), contributing to ~803 pub fns on the Gpu struct across dispatch/. gemv.rs alone exposes 124 near-identical methods (gemv_q4lut, gemv_q4wave, gemv_q4as8, gemv_q4k, gemv_hfq4g128, ... 40+ paro variants) that differ only in kernel-name string, kernarg tuple, and grid/block. This is a classic data-in-code smell: what varies is data (kernel name, module const, arg schema, grid formula), but it is encoded as hundreds of hand-written functions.

**Recommendation:** Model each kernel as a descriptor — `struct GemvKernel { name, src, block: [u32;3], grid: fn(m,k)->[u32;3], args: &[ArgKind] }` — and drive dispatch from a table plus one generic `launch(desc, tensors)`. Variants that only toggle residual/prerotate/pack become fields, not functions. This collapses the bulk of gemv.rs and makes adding a new quant format a one-row change rather than a new copy-pasted method, while keeping arch portability in the (tested) selector.

**Evidence:** 7 files each contain `impl Gpu {`; `grep -c '^  *pub fn'` sums to 803 across dispatch/; gemv.rs has 124 pub fns; functions gemv_q4lut/q4wave/q4as8 (gemv.rs:14/54/83) are byte-for-byte identical except the kernel name.

*Verification: confirmed*

### [Medium] duplication — `crates/rdna-compute/src/dispatch/gemv.rs:1324-1333, 1347-1356, 1371-1380`

**Observation:** The env-var runtime dispatch block reading HIPFIRE_PARO_PACK1/2/4 and branching to the pack1/pack2/pack4 kernel is copy-pasted verbatim in three wrapper functions (gemv_paro4g128t_with_prerotate, _residual_with_prerotate, _swiglu_residual_with_prerotate). Each re-does `std::env::var_os(...).is_some()` three times per call on the hot decode path, and any change to pack selection must be made in three places.

**Recommendation:** Resolve pack mode once into an enum — `enum ParoPack { P1, P2, P4, Default }` read from env at startup (cached in flags like the existing OnceLock patterns) — and have each wrapper `match pack_mode` to pick the prerotated variant. Removes the triple duplication and the repeated per-token getenv.

**Evidence:** `grep -c HIPFIRE_PARO_PACK1 gemv.rs` = 3; identical 3-line is_some() cascades at gemv.rs:1324-1332, 1347-1355, 1371-1379.

*Verification: confirmed*

### [Low] coupling — `crates/rdna-compute/src/dispatch/ (family-wide) — e.g. gemm_hfq.rs:1887-1889, gemv.rs:29-30`

**Observation:** There are 966 truncating `as i32` casts on usize dimensions (m/k/batch) across the 7 files, with zero `try_into`, `checked_*`, or bounds guards (grep returned none). Dims are passed to kernels as i32; a K or M ≥ 2^31, or a `batch_size * k` product exceeding i32, would silently wrap to a negative/small value and produce wrong-shape launches rather than an error. In practice LLM layer dims are bounded well under 2^31, so this is latent rather than active, but there is no defensive check anywhere on parsed/caller-supplied shapes.

**Recommendation:** Add a single `fn dim_i32(x: usize) -> HipResult<i32>` (or debug_assert) used at the top of the kernarg builder, returning an error for out-of-range dims. Cheap, centralizes the invariant, and pairs naturally with the descriptor-table refactor so it is enforced in one place instead of 966.

**Evidence:** `grep -h ' as i32'` totals 966 across the 7 files; `grep 'try_into|checked_|i32::MAX'` over gemv/gemm_qkv/gemm_hfq returns 0 matches.

*Verification: confirmed*

### [Low] utils-sprawl — `Remaining lower-severity items (folded)`

**Observation:** Additional hygiene items verified but not individually High/Medium: (a) mod.rs is itself 5153 LOC mixing the Gpu struct definition (532), GpuTensor (124), the launch_maybe_blob abstraction (1504), mmq_screen_weight (1952), and misc helpers — the crate's launch primitive and its god-struct share one oversized module. (b) The per-variant config `match` tables (gemm_hfq.rs:1837-1880, gemm_gate.rs:1567-1607) are the good pattern — small, local, data-shaped — and should be the template the rest of the family migrates toward. (c) The `auto_variant` heuristic (gemm_hfq.rs:1824-1834: force_det / is_gfx115x / batch / m thresholds) is pure arch+shape logic that is trivially unit-testable today but sits inline and untested. (d) profiling boilerplate (`begin_timer`/`if let Some(t) = timer { t.finish }`) is repeated at essentially every launch site and could be a scope guard.

**Recommendation:** Split mod.rs so the launch primitive and Gpu struct live in focused modules. Adopt the match-table + pure-selector pattern uniformly. Wrap the profiling timer in a Drop guard to delete the repeated finish() epilogue. These are schedule-when-convenient cleanups that compound with the pure-selector and kernarg-macro refactors above.

**Evidence:** mod.rs = 5153 LOC with struct Gpu at :532 and launch_maybe_blob at :1504; variant match tables at gemm_hfq.rs:1837 and gemm_gate.rs:1567; auto_variant heuristic at gemm_hfq.rs:1824-1834; timer epilogue `if let Some(t) = timer` recurs at every launch.

*Verification: confirmed*


## rdna-compute — attention, dispatch core, kernels

*Subsystem key: `rdna-attn-core` — 11 finding(s)*

**Subsystem assessment:** This subsystem is the HIP/ROCm-direct kernel dispatch core of rdna-compute: a single god struct `Gpu` (dispatch/mod.rs) carries GPU state and ~800 dispatch methods spread across 24 `impl Gpu` files, plus a 258KB embedded-kernel-source blob (kernels.rs), the JIT compiler, memory pool, arch capability tables, and three "profil*" modules. The generic-compute framing is sound in places (pool.rs, arch_caps.rs, and the arch-selection logic in kernels.rs are clean and well-tested), but the crate has absorbed model-family-specific dispatch (deepseek4.rs, zaya_cca.rs, mamba2.rs) even though dedicated arch crates exist for those models, and the `Gpu` struct has become a ~50-field, ~9-responsibility monolith. Main risks: the god object blocks parallel work and makes state lifetimes hard to reason about; attention.rs contains large blocks of byte-identical copy-paste (asym/fwht flash variants) that must be hand-kept in sync; and essentially all dispatch logic plus the pure byte/occupancy/ELF-parsing helpers ship with zero unit tests. The dispatch hot path itself is HIP-correct and portability-conscious (arch atoms), so most issues are maintainability rather than acute correctness, with the ELF parser and truncating as-casts as the notable latent-panic exceptions.

### [High] monolith — `crates/rdna-compute/src/dispatch/mod.rs:532-769`

**Observation:** `Gpu` is a god object: ~50 fields spanning at least 9 distinct responsibilities (HIP runtime + arch/caps/flags; JIT compiler + module/function caches; memory pool + free mailbox; calibration capture hooks; MQ/OQ/PARO quant scratch (~20 fields); fp16/bf16/fp8 activation-conversion scratch with src-ptr caches; MMQ per-weight screening cache; three separate hipGraph capture subsystems — AR-forward, verify_graph_cache, replay_graph_cache — plus blobs/exec; rocBLAS + fp16 shadow cache). Its behavior is grafted on via `impl Gpu` blocks in 24 files totaling ~800 methods (mod.rs 134, gemv.rs 125, attention.rs 76, gemm_qkv.rs 67, deepseek4.rs 65, fused.rs 52, ...). Every unrelated feature widens the same struct and shares its `&mut self`, serializing all work through one borrow.

**Recommendation:** Decompose `Gpu` into cohesive owned sub-structs behind narrow methods: e.g. `KernelCache { compiler, modules, functions }`, `ActivationStaging { fp16_x_*, bf16_x_*, fp8_x_*, q8_1_mmq_* }`, `GraphCapture { ar_*, verify_graph_cache, replay_graph_cache, blobs, exec }`, `QuantScratch { mq_*, oq4_*, paro_* }`. Group dispatch methods into traits (e.g. `trait AttentionDispatch`, `trait GemmDispatch`) implemented for `Gpu` so families are discoverable and independently reviewable. This does not change the HIP-direct hot path; it partitions state ownership so capture/scratch lifetimes are local instead of 50 flat pub fields.

**Evidence:** struct Gpu has ~50 fields (mod.rs:532-769); 3 distinct graph caches (verify_graph_cache mod.rs:716, replay_graph_cache mod.rs:735, ar_forward_* mod.rs:701-705); ~800 `impl Gpu` methods across 24 dispatch/*.rs files (per-file fn counts verified via grep).

*Verification: confirmed*

### [High] crate-boundary — `crates/rdna-compute/src/dispatch/deepseek4.rs:12-3273`

**Observation:** Model-family-specific dispatch lives inside the generic `rdna-compute` ("Compute kernel dispatch for RDNA GPUs") crate. deepseek4.rs adds 65 DeepSeek-V4-specific `impl Gpu` methods — hyper-connection Sinkhorn (`hc_sinkhorn_4x4`), hash routing (`hash_router_normalize_f32`), NSA indexer/compressor (`indexer_top_k`, `compressor_compress_aligned_batched_f32`), `deepseek4_attn_swa_*`, `deepseek4_gemv_mq2g256_lloyd_moe_*`. zaya_cca.rs and mamba2.rs do the same for ZAYA1 and nemotron_h Mamba-2. Dedicated arch crates already exist (hipfire-arch-deepseek4, hipfire-arch-zaya, hipfire-arch-nemotron), and zaya_cca.rs's own header says it is 'Faithful to crates/hipfire-arch-zaya/src/cpu.rs' — i.e. the model logic is mirrored across two crates. This inverts the intended boundary: the generic compute crate must be edited/recompiled for per-model changes and its god `Gpu` accumulates model-named methods.

**Recommendation:** Keep rdna-compute holding only generic primitives (gemv/gemm/attention/norm/rope/moe families parameterized by dtype and quant token). Move model-specific orchestration (hyper-connections, hash routing, NSA index/compress, SWA-topk gather) into the matching hipfire-arch-* crate, expressed in terms of the generic dispatch traits from finding #1. Where a genuinely novel kernel is needed, expose it as a generic dispatch method (e.g. `sinkhorn_4x4`, `topk_gather`) without the `deepseek4_`/`zaya_` prefix so it is reusable, and let the arch crate sequence it. This preserves the HIP-direct path while restoring the generic-vs-model separation.

**Evidence:** deepseek4.rs:15-3273 = 65 model-specific `impl Gpu` fns; zaya_cca.rs:5-8 header references hipfire-arch-zaya/src/cpu.rs; `ls crates/` shows hipfire-arch-deepseek4, hipfire-arch-zaya, hipfire-arch-nemotron; rdna-compute/Cargo.toml has no arch-crate dependency (boundary points the wrong way).

*Verification: confirmed*

### [High] duplication — `crates/rdna-compute/src/dispatch/attention.rs:2469-2696`

**Observation:** The non-batched flash-attention variants are near-verbatim copy-paste. `attention_flash_asym4` (2583-2696) is byte-identical to `attention_flash_asym3` (2469-2580) except the tile kernel name/SRC string ('asym3'→'asym4'); both hand-roll the same two-phase launch (givens tile kernel + shared `attention_flash_q8_0_reduce`). The code even documents it: line 2697-2698 'Same launch geometry + Q8_0 reduce as asym4 — only the tile kernel differs.' The same body is duplicated for asym2/asym3/asym4/fwht2/fwht3/fwht4 — 11 hand-rolled `ensure_givens4_kernel` launch bodies (~113 lines each). Notably the BATCHED siblings already funnel through one helper `launch_asym_flash_batched` (8+ call sites), so the refactor template exists but was never applied to the non-batched path. Divergence risk: a fix to the reduce/geometry must be applied 11× by hand.

**Recommendation:** Introduce a `launch_asym_flash(&mut self, tile_module: &str, tile_src: &str, ...)` helper mirroring the existing `launch_asym_flash_batched`, and reduce each of the 6-11 public variants to a one-line delegation passing only the kernel-name/SRC pair (exactly as the batched wrappers already do). This deletes ~600+ lines and makes 'only the tile kernel differs' true in code rather than in a comment.

**Evidence:** asym3 body attention.rs:2469-2580 vs asym4 body 2583-2696 differ only in kernel string; comment 2697-2698; 11 `ensure_givens4_kernel` call sites vs the already-shared `launch_asym_flash_batched` (defined once, 8+ delegations at 1523/1572/1974/2065/2108/2152/2242/2330).

*Verification: confirmed*

### [Medium] monolith — `crates/rdna-compute/src/kernels.rs:1-4751`

**Observation:** kernels.rs is a 4751-line / 258KB single module mixing three unrelated things: 736 `pub const *_SRC: &str` HIP-source string constants, ~60 `*_for_arch(caps) -> (&str, &str)` arch-selection functions, and a 26-test `dispatch_tests` module. Everything is flat and `pub`, so the compile unit and the public surface are enormous, grep is the only navigation, and adding a kernel means editing the same file as the selection logic and tests.

**Recommendation:** Split by concern: move the embedded source blobs into per-family submodules (e.g. `kernels/gemm_mq4_lloyd.rs`, `kernels/attention_flash.rs`) or, better, load them from the existing kernels/src/*.hip files via include_str! so the HIP source is authored as HIP, not as Rust string literals. Keep the `*_for_arch` selectors in a separate `kernel_select.rs` next to their tests. This shrinks the god module without touching dispatch behavior.

**Evidence:** wc -l kernels.rs = 4751; grep counts 736 `pub const *_SRC`, 748 total `pub const`, 60 fns; `#[cfg(test)] mod dispatch_tests` at kernels.rs:4139.

*Verification: adjusted — Monolith core confirmed: 4751 lines, 737 pub const *_SRC (749 total pub const), dispatch_tests at 4140 (obs says 4139, off-by-one), mixes source blobs + selectors + tests. But the observation's '~60 *_for_arch selection functions' is wrong: there are only 33 _for_arch fns — the '60' is the TOTAL fn count (33 selectors + 26 tests + 1). kernels/src/*.hip exist (572 files) so include_str! recommendation is viable. Severity Medium stands; number needs correcting to 33.*

### [Medium] missing-tests — `crates/rdna-compute/src/profile.rs:144-373`

**Observation:** The subsystem's pure, GPU-independent logic is almost entirely untested. profile.rs has 15+ analytical byte-count formulas (hfq4g256_weight_bytes, gemv_oq4g256_moe_bytes, attention_q8_0_kv_bytes, etc.) used for bandwidth attribution correctness, with 0 unit tests. profiler.rs occupancy math (decode_vgprs, occupancy_pct, ridge_point_flop_per_byte) and compiler.rs pure helpers (per_kernel_flags tag parsing, cache_valid, hsaco_is_elf) and pool.rs bucket_key are all 0-test. Across the 20+ dispatch files (~800 methods) there is 1 test total. While kernel launches need a GPU, the embedded pure logic — tile/grid math (`(max_seq + TILE-1)/TILE`), chunking, capture-mode branching, byte formulas — is trivially unit-testable and currently unguarded against silent regressions.

**Recommendation:** Add table-driven unit tests for the byte formulas (assert exact expected bytes for known m/k/group shapes — these encode format layout constants like '136 B/group' that silently drift), for pool bucket_key (power-of-2 rounding + MIN clamp), for compiler per_kernel_flags (tag present/absent/multiple), and for occupancy math. Extract the pure tile/grid arithmetic out of the launch methods into free functions so they can be tested without a GPU; the arch_caps.rs (19 tests) and kernels.rs selector tests (26) are the model to follow.

**Evidence:** Per-file `#[test]` counts: profile.rs 0, profiler.rs 0, compiler.rs 0, pool.rs 0, and 0 across all dispatch/*.rs except mod.rs (1); crate total 52 tests concentrated in arch_caps.rs (19) + kernels.rs (26) + profile_rocprof.rs (4).

*Verification: confirmed*

### [Medium] missing-tests — `crates/rdna-compute/src/profiler.rs:289-556`

**Observation:** profile_hsaco parses untrusted-shaped ELF/hsaco binaries using raw indexing helpers `u16_le/u32_le/u64_le` (539-556) that do `d[o+7]` with no bounds check. profile_hsaco guards some offsets (`if base + 40 > elf.len()`) but not all derived reads — e.g. line 332 `u64_le(elf, shoff + shstrndx*shentsize + 24)` and the segment/kd offset reads can index past a truncated or malformed file, panicking the caller. This binary parser has 0 unit tests, so no fixture exercises a short/garbage .hsaco.

**Recommendation:** Make the little-endian readers fallible (`fn u32_le(d,o) -> Option<u32>` returning None when `o+4 > d.len()`, or use `d.get(o..o+4)`), thread the Option through profile_hsaco (it already returns Option), and add tests feeding truncated buffers, a non-ELF blob, and a valid minimal descriptor. This removes a panic path on file input and locks the occupancy decode against regressions.

**Evidence:** profiler.rs:539-556 index d[o+1..o+7] with no bounds guard; profile_hsaco (289-425) computes offsets like shoff+shstrndx*shentsize+24 (332) and kd_off reads (383-386) not fully length-checked; 0 `#[test]` in profiler.rs.

*Verification: confirmed*

### [Medium] module-structure — `crates/rdna-compute/src/profiler.rs:29-112`

**Observation:** The hypothesized 'profiler triplication' is not code triplication — profile.rs (runtime hipEvent timing + byte formulas), profiler.rs (static hsaco ISA/occupancy analysis), and profile_rocprof.rs (rocprofv3 CSV cross-check) are three genuinely distinct concerns. The real issues are (a) discoverability: three top-level `pub mod`s share the `profil*` prefix, so callers cannot tell which does what; and (b) a real data duplication — profiler.rs::arch_spec (41-112) encodes per-arch hardware knowledge (generation string, simds_per_cu, waves, vgprs, lds, caches, bus width) that overlaps the domain of arch_caps.rs, which independently maps the same gfx ids to generations/wave behavior. Two arch tables can drift (e.g. a new gfx1300 must be added in both).

**Recommendation:** Rename for intent (e.g. `bandwidth` / `occupancy` / `rocprof_coverage`, or nest them under one `profiling` module) so the three roles are legible. Have profiler.rs derive its generation/wave facts from arch_caps.rs (or a shared `arch_table`) instead of a private `arch_spec` match, leaving only the perf-constant fields (clocks, BW multipliers) local. This makes arch_caps.rs the single source of truth for 'which gfx is which generation'.

**Evidence:** Three separate pub modules profile/profiler/profile_rocprof (lib.rs:13-16); profiler.rs::arch_spec match on gfx ids (41-112) parallels arch_caps.rs ArchCaps gfx atoms + generation grouping (arch_caps.rs:96-135).

*Verification: confirmed*

### [Medium] duplication — `crates/rdna-compute/src/kernels.rs:257-838`

**Observation:** ~60 `*_for_arch(caps: &ArchCaps) -> (&'static str, &'static str)` selectors repeat an identical shape: `match caps.arch() { gfx12 => (SRC_GFX12, name_rdna4), gfx11/1151 => (SRC, name_rdna3), _ => panic!(...) }`. The bodies differ only in the SRC constant, the module-name string, and the panic text. This is ~60 copies of one dispatch template. (Mitigating: these are pure and well covered by dispatch_tests, and the panic arms are guarded upstream by is_batchable_la.) A portability note: many arms hard-panic for RDNA2 (gfx103x) and some for RDNA4 (gfx120x), so extending is_batchable_la without updating each selector is a latent panic.

**Recommendation:** Replace the hand-written selectors with a small declarative table or a macro, e.g. `arch_kernel!{ gemm_qkv_mq4g256_lloyd_wmma, gfx12 => (SRC_GFX12,"..rdna4"), gfx1151 => (..), gfx11 => (..) }` that expands to the match + a uniform panic message referencing the family name. One macro definition + N one-line invocations removes the copy-paste and guarantees consistent unsupported-arch messaging.

**Evidence:** kernels.rs:257-363 shows 4 consecutive selectors identical except SRC/name/panic; the same shape repeats for mq3 (588-838) and hfq (2177-2257); dispatch_tests (4139+) covers them.

*Verification: adjusted — Read confirms the identical selector template (match caps.arch(){ gfx12 => SRC_GFX12/rdna4; gfx11/1151 => SRC/rdna3; _ => panic! }) at 257-363, repeating for mq3 (588-838) and hfq (2177-2257); the RDNA2/RDNA4 hard-panic portability note is real. But the '~60 selectors' figure is inflated — there are 33 _for_arch selectors total (~27 in the 257-838 range). Macro/table recommendation is sound; severity Medium unchanged, count needs correcting.*

### [Low] utils-sprawl — `crates/rdna-compute/src/dispatch/misc.rs:1-1146`

**Observation:** misc.rs is a self-declared grab-bag: its module doc lists 'residual-quant, standalone Paro, Givens rotation, deinterleave, cross-entropy, cast, attn-bias, transpose, scatter, scale, l2-norm, qkv-split' — 21 unrelated `impl Gpu` ops bundled because they don't fit another family. gemm_misc.rs (31 fns) is a similar catch-all. This makes the file a magnet for any op without an obvious home and obscures which ops are hot-path vs training-only (cross_entropy_train, transpose_f32).

**Recommendation:** Redistribute into intent-named modules: rotations (givens/paro) into rope.rs or a rotation.rs, cross_entropy_* into a training/loss module (they are training-only per the deltanet/train context), casts (cast_f32_to_f16/bf16) into a small `convert.rs`, and scatter/gather/transpose into a `layout.rs`. Reserve misc.rs only for genuine one-offs, and revisit whenever it grows past a handful.

**Evidence:** misc.rs module doc (line 5) enumerates 12 unrelated op categories; 21 pub fns from givens_rotate (21) to cast_f32_to_f16 (1146) with no shared theme.

*Verification: confirmed*

### [Low] abstraction — `crates/rdna-compute/src/dispatch/deepseek4.rs:20`

**Observation:** Truncating `as` casts are pervasive and unchecked: attention.rs has 454 and deepseek4.rs 122 `as i32/u32/usize/i8/u8` conversions, including element counts fed to kernels as i32 (`x.numel() as i32` at deepseek4.rs:20, 1668, 2828). For the 122B-class targets this crate serves, most such dims are per-token/per-head and safely small, but numel-as-i32 on a full activation or a large K silently wraps negative past 2^31 with no debug assert, producing a corrupt grid rather than an error.

**Recommendation:** Add a single checked helper (e.g. `fn i32_dim(n: usize) -> HipResult<i32>` returning an error above i32::MAX, or `debug_assert!(n <= i32::MAX as usize)`) and route numel/K/M conversions through it at the dispatch boundary. This is cheap, keeps the kernel ABI (i32 params) unchanged, and turns a silent wrap into a diagnosable failure.

**Evidence:** grep: attention.rs 454 `as {i32,u32,usize,i8,u8}` casts, deepseek4.rs 122; `x.numel() as i32` at deepseek4.rs:20,1668,2828 with no bounds check.

*Verification: confirmed*

### [Low] monolith — `crates/rdna-compute/src/dispatch/mod.rs:876-1054`

**Observation:** `Gpu::init_with_device` is a ~178-line constructor that queries the device, builds arch_caps/flags, and initializes the full ~50-field struct (including all the quant-scratch, graph-cache, and screening fields to None/default) in one block. It is a direct consequence of the god-object in finding #1: the struct's breadth forces a wide, hard-to-review init.

**Recommendation:** Once the state sub-structs from finding #1 exist, give each a `Default`/`new()` and let `Gpu::init_with_device` compose them (`Gpu { cache: KernelCache::new(..)?, graph: GraphCapture::default(), scratch: QuantScratch::default(), .. }`). This shrinks the constructor to the genuinely device-dependent queries and makes the default-None fields self-documenting.

**Evidence:** init_with_device spans mod.rs:876-1054 (next method try_init_rocblas at 1055); it initializes the ~50 fields declared at mod.rs:532-769.

*Verification: confirmed*


## rdna-compute remainder + hip-bridge + hsa-bridge (FFI floor)

*Subsystem key: `rdna-rest-bridges` — 6 finding(s)*

**Subsystem assessment:** This subsystem is the FFI floor of hipfire: hip-bridge (dlopen wrapper over libamdhip64), hsa-bridge (experimental AQL-direct dispatch), the RCCL/rocBLAS wrappers, and the thin rdna-compute glue (pool, generic_warn, sampling dispatch). The wrappers are mostly well-shaped — raw function pointers are held in a Send/Sync *Lib struct, calls go through `check()`-style status translation, and most callers see `Result` rather than raw status codes. However FFI hygiene is uneven: one device-properties call writes a 1472-byte C struct into a 1024-byte buffer (heap overflow on a load-bearing arch-detection path), RAII/Drop coverage is inconsistent across handle types, and hsa-bridge ships ~2988 LOC with zero tests despite containing byte-layout-critical pure logic that is trivially host-testable. The two bridge crates also duplicate four structurally-identical error types, the symbol-loading macro, and the dlopen fallback chain, none of which is factored into a shared low-level FFI-loader abstraction. Main risks: the buffer overflow (correctness/memory safety) and the untested AQL packet-builder (silent dispatch corruption).

### [Medium] missing-tests — `crates/hsa-bridge/src (0 files with #[cfg(test)])`

**Observation:** hsa-bridge (~2988 LOC) has zero unit tests, yet it contains byte-layout-critical pure logic that runs with no GPU and would silently corrupt dispatch if wrong: `dispatch_packet_header()` bit-packing (lib.rs:663-669), `build_dispatch_packet` ndims derivation and `grid = grid*block` math (lib.rs:690-707), the Clang-offload-bundle ELF-magic unwrap in `from_code_object` (lib.rs:486-496), and `packet_slot`'s `index & (size-1)` ring-index wrap (lib.rs:323-327). All are host-testable with plain byte assertions. A wrong header word or setup field does not fail loudly — it mis-dispatches, which is exactly the kind of defect a test catches cheaply.

**Recommendation:** Add a `#[cfg(test)]` module exercising: `dispatch_packet_header()` equals the documented HIP header constant; `build_dispatch_packet` on a stack `HsaKernelDispatchPacket` yields expected setup/grid for 1D/2D/3D grids; the offload-bundle unwrap finds ELF after the bundle magic and errors when absent; and the power-of-2 slot masking. None require ROCm, so they run in no-gpu-ci. `const _: assert!(size_of==64)` already exists — extend that discipline to behavior.

**Evidence:** grep '#[cfg(test)]' crates/hsa-bridge/src -> 0 files; pure fns at lib.rs:663 (dispatch_packet_header), lib.rs:682 (build_dispatch_packet), lib.rs:486 (ELF unwrap)

*Verification: adjusted — Facts confirmed: grep shows 0 #[test]/#[cfg(test)] in hsa-bridge, and all cited pure fns exist — dispatch_packet_header (lib.rs:664), build_dispatch_packet ndims+grid*block (lib.rs:690-707), from_code_object ELF-magic search (lib.rs:486-496), packet_slot index&(size-1) (lib.rs:323-327). The const _: assert!(size_of::<HsaKernelDispatchPacket>()==64) exists at ffi.rs:66. Note the ~2988 LOC is the whole crate including examples; untested src is 1224 LOC. Severity downgraded: missing tests on currently-working byte-critical code is a latent-risk/maintainability gap (Medium, scheduled refactor), not a High-severity actively-occurring correctness risk; sibling missing-tests findings here are rated Low.*

### [Medium] duplication — `crates/hip-bridge/src/{error.rs:29,rccl.rs:29,rocblas.rs:23,ffi.rs:235}; crates/hsa-bridge/src/{error.rs:15,ffi.rs:238}`

**Observation:** The two bridges independently reimplement the same FFI scaffolding four+ times. Four structurally-identical error types exist — HipError{code:u32,message:String}, HsaError{code,message}, RcclError{status:u32,context:String}, RocblasError{status,context} — each with the same Display/`std::error::Error` boilerplate and a hand-rolled status->Result translator. The `load_fn!`/`load_sym!` symbol-resolution macro is copy-pasted (hip ffi.rs:235 vs hsa ffi.rs:238, differing only in error type), and the dlopen SONAME fallback chain is open-coded four times (hip ffi.rs:290-316, hsa ffi.rs:250-262, rccl.rs:103-119, rocblas.rs:109-126).

**Recommendation:** Introduce a small internal `ffi-loader` crate (or a shared module) providing: a generic `FfiError<Code>` (or a `BridgeError` with code+context), a `load_fn!` macro, and a `dlopen_first(&[&str]) -> Result<Library>` helper that takes the candidate SONAME list. hip-bridge and hsa-bridge then depend on it; each keeps only its status constants and typed enums. This removes ~150 lines of parallel boilerplate and makes the loader fallback policy uniform across libamdhip64/libhsa/librccl/librocblas.

**Evidence:** 4 error structs with identical shape (grep confirms); load_fn! duplicated at hip ffi.rs:235 and hsa ffi.rs:238; dlopen chains at ffi.rs:290, ffi.rs:250, rccl.rs:103, rocblas.rs:109

*Verification: confirmed*

### [Medium] abstraction — `crates/hip-bridge/src/ffi.rs:1446-1531; crates/hip-bridge/src/lib.rs:73-110; crates/hsa-bridge/src/lib.rs:426-468`

**Observation:** RAII coverage of FFI handles is inconsistent, making resource leaks easy. In hip-bridge only HostBuffer, Rocblas, and RcclComms implement Drop; Stream, Module, Function, Event, Graph, GraphExec, and DeviceBuffer have NONE — they must be destroyed via by-value `stream_destroy`/`event_destroy`/`graph_destroy`/`free` that the caller must remember to call, or the HIP object leaks. In hsa-bridge the reverse gap exists: HsaRuntime/Queue/Signal/Executable Drop correctly, but `HsaMemoryPool::allocate` (lib.rs:438) returns a bare `*mut u8` with no owning wrapper and no size tracking, so the unsafe pointer and its manual `free` leak straight up to callers — weaker than hip's DeviceBuffer which at least carries ptr+size.

**Recommendation:** Decide one ownership model and apply it uniformly. Simplest: give each handle a Drop that calls the destroy fn — which requires each wrapper to hold an `Arc<HipRuntime>` (hsa-bridge already does this for its handles, so the pattern is proven). At minimum, wrap `HsaMemoryPool::allocate` in a `HsaBuffer{ptr,size,pool:Arc<..>}` with a Drop mirroring hip's DeviceBuffer, so hsa callers get the same ptr+size safety envelope instead of raw pointers.

**Evidence:** Drop present only for HostBuffer(ffi.rs:1503)/Rocblas(rocblas.rs:287)/RcclComms(rccl.rs:364); Stream/Module/Function/Event/Graph/GraphExec/DeviceBuffer have none; HsaMemoryPool (lib.rs:426) has no Drop and allocate returns *mut u8

*Verification: confirmed*

### [Medium] duplication — `crates/hip-bridge/src/ffi.rs:793-1051`

**Observation:** The memcpy/memset family is ~10 near-identical methods (memcpy_htod, _htod_offset, _dtoh, _dtoh_at, _dtod, _dtod_at, _dtod_offset, memset, memset_async, async variants). Each repeats the same shape: a size `assert!`, offset pointer arithmetic via `(ptr as *mut u8).add(off)`, `Instant::now()` + `launch_counters::X::record`, and — six separate times — an inline `static DUMP: OnceLock<bool>` reading a distinct `HIPFIRE_*_DUMP` env var and eprintln'ing bytes+caller location (ffi.rs:838, 909, 953, 987, 1007, 1038). This is a large copy-paste surface where a bounds-check or direction constant can drift between siblings unnoticed.

**Recommendation:** Factor the timing+dump+record wrapper into one helper, e.g. `fn timed_copy(&self, counter, dump_env, bytes, loc, f: impl FnOnce()->u32) -> HipResult<()>`, and a single private `raw_memcpy(dst_ptr, src_ptr, size, kind)` that all public variants call after computing offsets. The public methods shrink to argument validation + offset math. This collapses the six OnceLock/env blocks into one and removes the risk of an inconsistent assert across variants.

**Evidence:** grep 'OnceLock<bool>' ffi.rs -> 6 copies; near-identical bodies span ffi.rs:793-1051; each memcpy_* repeats assert+add+Instant+record

*Verification: confirmed*

### [Low] missing-tests — `crates/hsa-bridge/src/lib.rs:158-178, 323-327, 690-707`

**Observation:** Several documented invariants at the hsa dispatch boundary are unchecked and untested. `create_queue` doc says "size must be a power of 2" (lib.rs:158) but never validates it, and `packet_slot` relies on that with `index & (size-1)` (lib.rs:326) — a non-pow2 size silently indexes the wrong or out-of-bounds slot. `build_dispatch_packet` truncates workgroup dims with `block[i] as u16` (lib.rs:701-703): a block dim >= 65536 silently wraps rather than erroring, and `grid[i].saturating_mul(block[i])` silently clamps. RcclComms::all_reduce also does an unchecked `self.comms[rank]` (rccl.rs:294) that panics on an out-of-range rank.

**Recommendation:** Add a `debug_assert!(size.is_power_of_two())` in create_queue (or return an error), and in build_dispatch_packet guard block dims against u16::MAX with a typed error instead of a silent `as` truncation. Bounds-check `rank` in all_reduce and return RcclError. These are all cheaply unit-testable host-side and belong in the same new hsa test module recommended above.

**Evidence:** create_queue pow2 doc-only at lib.rs:158; packet_slot mask at lib.rs:326; `block[0] as u16` at lib.rs:701; unchecked `self.comms[rank]` at rccl.rs:294

*Verification: confirmed*

### [Low] missing-tests — `crates/rdna-compute/src/pool.rs:40-47; crates/hip-bridge/src/lib.rs:58-69`

**Observation:** Two small pure-logic units on hot/allocation paths have no tests. `GpuPool::bucket_key` (pool.rs:40) does power-of-2 rounding with a 256 B floor and is the key that governs free-list reuse correctness, and the alloc/free-list eviction logic (pool.rs:59-90) is pure aside from the `hip.malloc/free` calls. `MemoryType::from_raw` (hip-bridge/src/lib.rs:58) is a hand-written u32->enum map that must stay in sync with `hipMemoryType`. None are covered, though bucket_key and from_raw are trivially testable without a GPU.

**Recommendation:** Add unit tests for bucket_key (256 floor, exact pow2, next_power_of_two boundaries) and MemoryType::from_raw (each mapped value plus the None arm). For the free-list, consider threading allocation through a tiny trait so the reuse/eviction decision (`buf.size() >= size` pop loop) can be tested with a fake allocator; this is the logic that prevented the 27B draft-fits regression described in the pool.rs comment and deserves a guard test.

**Evidence:** pool.rs has 0 #[test]; bucket_key at pool.rs:40; MemoryType::from_raw at hip-bridge/src/lib.rs:59 with no test

*Verification: confirmed*


## hipfire-runtime — hot-path core (hfq/kv/llama/arch)

*Subsystem key: `runtime-core` — 6 finding(s)*

**Subsystem assessment:** runtime-core is the arch-agnostic inference substrate: HFQ file I/O + weight loading (hfq.rs), the LLaMA-family forward + shared transformer primitives (llama.rs), the KV cache (kv.rs + kv_adaptive.rs + kv_hier.rs), the Architecture/SimpleAr/ServingBackend trait contract (arch.rs), and the sampler. Cohesion is mixed. The trait layer (arch.rs), sampler.rs, tool_call.rs, sequence_state.rs and kv_adaptive.rs are well-factored and tested, and the eos_filter/loop_guard modules are thin compat re-exports over hipfire-generate (which carries the real tests). The kv* family is genuine layering, not sprawl — kv_adaptive.rs is GPU-free byte math with 8 tests and kv_hier.rs is a flag-gated two-tier feature; the real sprawl lives inside kv.rs itself. The dominant risks are: kv.rs's boolean-flag mode representation with 49 combinatorial constructors and a 33x-duplicated struct literal; a 4.3k-line generated dead-code env-doc table compiled into the hot-path crate; and llama.rs being a 3.2k-line de-facto shared-transformer core that is mislabeled and only partially extracted (transformer.rs holds predicates, not the forward). llama.rs is the real implementation; hipfire-arch-llama is a thin facade that re-exports it.

### [High] duplication — `crates/hipfire-runtime/src/kv.rs:31-96 (struct), 106-2482 (impl KvCache)`

**Observation:** KvCache encodes its quantization mode as 9 parallel booleans (quantized + 8 pub quant_* fields: quant_q8/int8/hfq4/asym4/asym3/asym2/fwht/kvarn, kv.rs:43-76) rather than a single mode enum. This has metastasized into 49 `new_gpu*` constructors (one per codec x {plain,capped,filtered} x {single,multi-GPU}), each ending in a near-identical ~25-field `Self { .. }` literal — that literal appears 33 times. The single-GPU and _multi pairs are verbatim clones except for the K/V allocation call: e.g. new_gpu_q8_capped (L294) and new_gpu_q8_capped_multi (L1819) differ only in the `gpu.zeros` loop vs `alloc_kv_per_layer_multi` and the assert message; the whole flag block is copy-pasted. The 2596-LOC file has 0 tests.

**Recommendation:** Replace the 9 booleans with one `enum KvQuantMode { Fp32, Q8, Int8, Hfq4, Asym{bits}, Fwht{bits}, Kvarn }` stored in a single field, and collapse the 49 constructors into a handful parametrized by (mode, cap, filter-mask, allocator). Factor the shared tail into `fn from_alloc(mode, dims, alloc: impl Fn(&mut _, usize)->HipResult<GpuTensor>) -> Self` so the single/multi split is just which closure is passed, eliminating the 33 duplicated literals. A mode enum also makes illegal flag combinations (e.g. quant_q8 && quant_asym3) unrepresentable and lets the prefill-mode mapping in transformer.rs match on one value.

**Evidence:** 8 `pub quant_` fields (grep count) + `quantized`; 49 `pub fn new_gpu` in kv.rs; the `quant_q8:` struct-literal line appears 33x; new_gpu_q8_capped:294 vs new_gpu_q8_capped_multi:1819 identical field block; 0 `#[cfg(test)]` in kv.rs

*Verification: confirmed*

### [Medium] crate-boundary — `crates/hipfire-runtime/src/env_docs.rs:1-4259 (527 EnvVarDoc entries, not 1057)`

**Observation:** env_docs.rs is a 4259-line, machine-generated `EnvVarDoc` const table (1057 entries) compiled into the runtime hot-path crate. Its header declares `#![allow(dead_code)]` and 'Generated automatically ... Do not hand-edit'; the entries index env usages across the whole workspace (e.g. rdna-compute/examples). It is produced and consumed only by the CLI tooling command crates/hipfire-cli/src/commands/gen_env_docs.rs (which writes it via fs::write at :129) — no runtime code path references it. This is offline documentation tooling bloating compile time of the crate that AGENTS.md says must stay lean and HIP-direct.

**Recommendation:** Move the generated registry out of hipfire-runtime into the tooling crate that owns it (hipfire-cli's gen-env-docs, or a dedicated `hipfire-env-docs` data crate), or emit it as a build artifact / include_str! data file rather than 1057 compiled `pub const` items. Nothing in the inference path needs the `EnvVarDoc` symbols, so the runtime crate should not pay to type-check and codegen them. This mirrors the repo invariant that offline/tooling concerns live outside the inference binaries.

**Evidence:** env_docs.rs first line `#![allow(dead_code)]`; 1057 `EnvVarDoc {` entries; only cross-crate reference is hipfire-cli/src/commands/gen_env_docs.rs which fs::write's it (:129); lib.rs:29 `pub mod env_docs;`

*Verification: adjusted — Core claim fully holds: 4259-LOC file with `#![allow(dead_code)]` and 'Generated automatically... Do not hand-edit' banner, entries source workspace-wide env usages (e.g. rdna-compute/examples), only cross-crate consumer is hipfire-cli/gen_env_docs.rs which fs::write's it at :129; no runtime code path references EnvVarDoc/ENV_* symbols; lib.rs:29 exports it. But the entry count is wrong: `pub const ENV_` = 527 and `= EnvVarDoc {` = 527, not the claimed 1057 (roughly 2x overstated). Severity Medium and recommendation (move to tooling crate) stay valid.*

### [Medium] monolith — `crates/hipfire-runtime/src/llama.rs:1-3153`

**Observation:** llama.rs is named after one arch but is the de-facto shared transformer core for the whole runtime. It carries LlamaConfig/LlamaWeights/ForwardScratch plus the full forward family (forward, forward_scratch_layers at L1843 is ~300 LOC by itself, forward_prefill_batch, forward_early_exit, forward_logits_gpu) AND the cross-arch primitives (KvCache re-exports, dequantizers, GEMV helpers) that hipfire-arch-qwen35's pflash branch reaches into directly. The sibling hipfire-arch-llama crate is a thin facade: its lib.rs is `pub use hipfire_runtime::llama;` plus the trait impl. transformer.rs — created as the intended extraction home — holds only 8 batchability predicate fns, so the shared/forward split never happened and transformer.rs is itself misnamed.

**Recommendation:** Answer to 'which is real': the runtime llama.rs is the implementation; the arch crate is a facade. Finish the documented split: move the genuinely arch-neutral primitives (KvCache glue, dequantizers, GEMV/rmsnorm helpers, ForwardScratch) into transformer.rs (or a `transformer_core` module) and physically relocate the LLaMA-only forward fns into hipfire-arch-llama, leaving hipfire-runtime with only shared infra. Until then, at minimum rename transformer.rs to reflect that it currently holds prefill-batchability predicates, not transformer composition, so its 195 LOC don't imply the extraction is done.

**Evidence:** llama.rs 3153 LOC; forward_scratch_layers spans L1843-~2144 (~300 lines); hipfire-arch-llama/src/lib.rs:56 `pub use hipfire_runtime::llama;`; transformer.rs has only 8 `pub fn` (all batchability predicates)

*Verification: confirmed*

### [Medium] monolith — `crates/hipfire-runtime/src/hfq.rs (2526 LOC)`

**Observation:** hfq.rs bundles five distinct responsibilities: HFQM package reading (HfqPackage::open L191), HFQM package writing (write_hfqm_package_from_files L232, _streaming L337, _mem L423), single-file HfqFile mmap reading (L502), model-config parsing (config_from_hfq L1167, config_from_safetensors_llama L2035), and weight loading including AWQ repacking (load_weight_tensor L1331, repack_awq_to_hfq4g128 L2112, load_weights_hfq L1732, load_weights_paroquant_llama L2328). The write/repack paths are packaging/import-conversion concerns living beside the hot-path readers in one 2.5k-line file. hfq_modules.rs already cleanly holds the HFQM v2 module-table data model, showing the split is feasible.

**Recommendation:** Separate the file into a read-side module (HfqPackage/HfqFile/parse_hfqm_index — the mmap hot path), a write-side module (write_hfqm_package_* + repack_awq_to_hfq4g128 — packaging/conversion), and a config/weight-load module (config_from_*, load_weights_*). The AWQ repack and package-writer logic in particular is format-conversion that AGENTS.md steers toward dedicated tooling rather than the lean runtime reader; at minimum give it its own module so the mmap reader isn't recompiled with the writer.

**Evidence:** hfq.rs top-level: HfqPackage::open:191, write_hfqm_package_from_files:232/_streaming:337/_mem:423, HfqFile::open:503, config_from_hfq:1167, load_weights_hfq:1732, repack_awq_to_hfq4g128:2112, load_weights_paroquant_llama:2328

*Verification: confirmed*

### [Medium] missing-tests — `crates/hipfire-runtime/src/hfq.rs:58-188 (json_blob_end, parse_hfqm_index)`

**Observation:** The on-disk HFQM index parser is the most panic-sensitive pure code in the subsystem: parse_hfqm_index reads little-endian lengths/offsets straight from an untrusted mmap and guards each read with a distinct bounds check (invalid offsets L96, missing count L111, name-length truncation L130, name/shape truncation L138, shape/data_size truncation L150, data range past EOF L165), and json_blob_end is a hand-rolled brace-depth JSON scanner. All of this is GPU-free and trivially unit-testable, yet the only test (writes_and_reads_sidecar_hfqm_package L1974) exercises the happy-path round trip; none of the Err/truncation branches or malformed-JSON cases are covered. A regression that turns one of those `return Err` checks into an out-of-bounds slice would panic the loader on a corrupt file with no test to catch it.

**Recommendation:** Add table-driven tests that feed deliberately corrupt byte buffers (short header, metadata_offset > data_offset, name_len running past data_offset, data_size overflowing file size, unterminated JSON) and assert each returns Err rather than panicking, plus json_blob_end edge cases (escaped quotes, nested braces, unterminated). These need no GPU and lock in the bounds contract of the loader.

**Evidence:** parse_hfqm_index has 6 Err bounds checks (hfq.rs:96,111,130,138,150,165); json_blob_end:58 hand-rolled scanner; only hfq.rs test is happy-path round-trip at :1974-2028

*Verification: confirmed*

### [Low] test-structure — `crates/hipfire-runtime/src/arch.rs:352, 594-790`

**Observation:** Three smaller hygiene items. (1) Two distinct enums named StopReason coexist in the crate: crate::arch::StopReason (arch.rs:352, {Eos,MaxTokens,StopSequence}) and crate::loop_guard::StopReason (re-exported from hipfire_generate, {NgramRepeat{ngram,count},...}); same name, same crate, unrelated meaning — a readability trap on import. (2) The shared decode loop's pure stop-classification logic — the single-token-attractor guard (arch.rs:714) and the finish_reason mapping (arch.rs:762-766) — is inlined inside decode_loop_with_timing, a GPU-driving function with 0 tests in arch.rs, so that pure policy can't be unit-tested without a GPU backend. (3) transformer.rs is named for transformer composition but only holds prefill-batchability predicates.

**Recommendation:** Rename one StopReason (e.g. arch's to ServeStopReason or loop_guard's to LoopStopReason) to remove the collision. Extract the stop-classification (`fn classify_stop(next, eos, committed, stop_sequences, generated, max_tokens) -> Option<StopReason>` and `fn finish_reason(stop, generated, max) -> &str`) as pure free fns so they get unit tests independent of the GPU loop. Rename transformer.rs to match its actual batchability-predicate content until the llama.rs extraction lands.

**Evidence:** arch.rs:352 `pub enum StopReason`; hipfire-generate loop_guard.rs:25 `pub enum StopReason { NgramRepeat {...} }`; attractor guard arch.rs:714, finish_reason arch.rs:762-766; arch.rs test-marker count = 0

*Verification: confirmed*


## hipfire-runtime — loaders, spec-decode plumbing, examples tree

*Subsystem key: `runtime-rest` — 9 finding(s)*

**Subsystem assessment:** This slice of hipfire-runtime holds the loader/dispatch and spec-decode plumbing (weights.rs, weight_pager.rs, multi_gpu.rs, tp_shard.rs, dflash.rs, ddtree.rs, cask.rs, triattn.rs) plus a large amount of non-hot-path tooling (host_profile.rs, calibration.rs, kld_eval.rs, speed_bench.rs) and format loaders (safetensors_source.rs, quant.rs). Core structure is mostly idiomatic: gguf.rs/tokenizer.rs/model_source.rs are thin re-export shims onto hipfire-model (no duplication there), and weights.rs uses tidy route enums. The dominant health risk is not the library modules but the examples/ tree: 148 files / 42,292 LOC functioning as a de-facto QA/bench/eval harness that hipfire-eval drives by shelling out to example binaries, which contradicts the AGENTS rule that admission evidence lives in hipfire-eval batteries. Secondary risks are cross-crate GGML dequant duplication, benchmark/eval tooling compiled into the inference library, and unguarded panics plus zero tests in the safetensors and quant load paths.

### [High] test-structure — `crates/hipfire-runtime/examples/ (148 files) + crates/hipfire-eval/src/executor_examples.rs:146,545,752,965`

**Observation:** examples/ is a de-facto manual test/QA/bench suite, not a set of usage samples. Of 148 files (42,292 LOC), 31 are prefixed test_, 16 bench_, 11 profile_, 6 dump_, and there are QA-pair clones (test_inference.rs+test_inferenceQA.rs, test_kernels.rs+test_kernelsQA.rs, test_gemv_q4k.rs+test_gemv_q4kQA.rs, test_q4f16.rs+test_q4f16QA.rs, test_q8kv.rs+test_q8kvQA.rs, test_qwen35_load.rs+test_qwen35_loadQA.rs). hipfire-eval treats these as admission evidence by shelling out: executor_examples.rs (4310 LOC) resolves example binaries and skips with messages like 'perplexity example binary not found; build with cargo build --release -p hipfire-runtime --example perplexity'. This inverts the AGENTS rule that model/runtime admission evidence belongs in hipfire-eval batteries; the harness depends on out-of-band cargo builds of throwaway bins.

**Recommendation:** Promote the genuinely load-bearing examples (perplexity, dflash_spec_demo, bench_qwen35_speed, coherence_probe) into library functions inside hipfire-runtime or hipfire-eval and invoke them in-process from eval executors instead of spawning `cargo build --example` binaries. Convert the test_*/QA pairs into #[test] integration tests under tests/ or hipfire-eval suites, deleting the clones. Keep a small number of true examples as documentation only.

**Evidence:** 148 example files, 42,292 LOC total; 31 test_*, 16 bench_* by prefix; executor_examples.rs is 4310 LOC of shim; skip strings reference `cargo build --example` at lines 545/752/965

*Verification: confirmed*

### [High] duplication — `crates/hipfire-runtime/src/quant.rs:82,278,46,14 vs crates/hipfire-quantize/src/gguf_input.rs:333,439`

**Observation:** The GGML dequant codecs are copy-pasted across a crate boundary. runtime/quant.rs::dequantize_q4_k (L82) and dequantize_q6_k (L278) are byte-for-byte identical to hipfire-quantize/gguf_input.rs::dequant_q4_k (L333) and dequant_q6_k (L439): same block_bytes=144, same 12-byte scale/min unpack loops, same 4-group/2-subblock dequant loop. dequantize_q8_0/q4_0 mirror the same GGML layouts. Notably f16_to_f32 was already de-duplicated into hipfire-primitives (quant.rs:74 re-exports it) but the block dequant was left duplicated, so the two copies can silently drift on a format fix.

**Recommendation:** Hoist the GGML block dequant (q4_0/q8_0/q4_k/q6_k) into a shared leaf crate (hipfire-primitives, alongside the existing f16 conversions, or hipfire-quant-format) and have both runtime/quant.rs and hipfire-quantize/gguf_input.rs call it. This keeps the byte-contract authoritative in one place and matches the pattern already used for f16_to_f32.

**Evidence:** grep found fn dequant_q4_k/q6_k in gguf_input.rs and fn dequantize_q4_k/q6_k in quant.rs; side-by-side read shows identical block_bytes=144, identical scales/mins bit-unpack, identical group loop

*Verification: confirmed*

### [Medium] missing-tests — `crates/hipfire-runtime/src/safetensors_source.rs:71-72,96-97`

**Observation:** The safetensors loader panics on malformed on-disk input and has zero unit tests. L71 `u64::from_le_bytes(mmap[0..8].try_into().unwrap())` panics if a .safetensors file is smaller than 8 bytes; L72 `&mmap[8..8 + header_len]` is an unchecked slice that panics if a corrupt header_len exceeds the mapping; L96-97 `a[0]`/`a[1]` on data_offsets panic if the array has fewer than two elements. This is legitimate load-at-inference code (not conversion, so it correctly stays in runtime), but a truncated or hostile checkpoint aborts the process instead of returning the io::Error the rest of open() already produces.

**Recommendation:** Replace the unwrap and unchecked slices with length-validated reads returning io::Error(InvalidData): check mmap.len() >= 8 before reading header_len, verify 8 + header_len <= mmap.len(), and use .get(0)/.get(1) on data_offsets. Add a #[cfg(test)] module with truncated-header and short-file fixtures (no GPU needed) to lock the error path.

**Evidence:** safetensors_source.rs:71 `mmap[0..8].try_into().unwrap()`; L72 slice `&mmap[8..8 + header_len]`; L96-97 `a[0].as_u64()`/`a[1].as_u64()`; file has 0 `#[test]`

*Verification: confirmed*

### [Medium] missing-tests — `crates/hipfire-runtime/src/quant.rs:144-270 (convert_q4k_to_q4f16_g64 / _g32), whole file`

**Observation:** quant.rs is 384 lines of pure, deterministic, GPU-free bit manipulation (dequant plus the Q4_K->Q4_F16 nibble-repack transcodes) with zero unit tests. convert_q4k_to_q4f16_g32 (L192-270) does intricate nibble reshuffling with comments describing exact byte layouts and truncating `.min(15)` packing (L178-181) — exactly the kind of golden-vector logic that regresses silently. Everything here is trivially testable on the host, yet has no coverage. (Separately, convert_q4k_to_q4f16_* is a format transcode; because it runs at load time to feed a runtime dtype it is defensible in runtime rather than hipfire-coexistence, but it should be flagged if a pure offline path ever needs it.)

**Recommendation:** Add a #[cfg(test)] module with round-trip golden vectors: build a known Q4_K block, dequantize, transcode to G32/G64, and assert the reconstructed values against the direct dequant within tolerance. This costs nothing at runtime and pins the byte layout the GPU kernels rely on.

**Evidence:** quant.rs is 384 LOC with 0 `#[test]`; convert_q4k_to_q4f16_g32 spans L192-270 with hand-written nibble packing; truncating pack at L178-181

*Verification: confirmed*

### [Medium] crate-boundary — `crates/hipfire-runtime/src/host_profile.rs:1-1455; lib.rs:35 (pub mod host_profile)`

**Observation:** host_profile.rs is a 1455-LOC standalone benchmarking/profiling tool compiled into the inference runtime library. It bundles GPU/storage/CPU-memcpy bandwidth sweeps (gpu_records L496, storage_records L365, cpu_memcpy_record L339), CLI arg parsing (parse_args_from L133, usage L204, take_value/parse_usize L1259-1281), and timestamp formatting (utc_now/utc_stamp_compact/unix_secs L1300-1330). Per AGENTS, model/runtime evidence and benches belong in hipfire-eval / benchmarks, not the hot-path lib. Its presence (and its own bin src/bin/hipfire_host_profile.rs) inflates the runtime crate's compile surface and mixes measurement tooling into the inference API.

**Recommendation:** Move host_profile.rs into a dedicated tooling crate or into hipfire-eval/benchmarks as a bandwidth battery, exposing only a thin capability-report struct to runtime if consumers need it. Keep the arg-parsing/time-formatting helpers local to that tool rather than in the shared runtime module tree.

**Evidence:** host_profile.rs 1455 LOC exporting run_profile/parse_args_from/usage; separate binary src/bin/hipfire_host_profile.rs; declared pub in runtime lib.rs

*Verification: confirmed*

### [Medium] utils-sprawl — `crates/hipfire-runtime/src/lib.rs:16-56 (kld_eval, speed_bench, host_profile, calibration modules)`

**Observation:** The runtime library re-exports ~30 pub modules that mix the inference hot path (kv, sampler, weights, kv_hier) with evaluation/benchmark/calibration tooling that is not hot-path: kld_eval.rs (KLD self-scoring, kld_self_score L138), speed_bench.rs (LatencyStats, arg parsing, kv-cache bench setup), host_profile.rs (bandwidth benches), and calibration.rs (offline Hessian/activation capture). This turns hipfire-runtime into a grab-bag crate whose public surface conflates 'run the model' with 'measure/calibrate the model', contradicting the AGENTS separation between the lean inference path and evidence tooling.

**Recommendation:** Group the measurement/eval modules (kld_eval, speed_bench, host_profile) behind a cargo feature or extract them into a hipfire-runtime-tools / hipfire-eval home so the default runtime build and its API stay focused on inference. Keep calibration as runtime only if it must hook a live forward pass; otherwise move it toward the quantize tooling side.

**Evidence:** lib.rs declares pub mod kld_eval, speed_bench, host_profile, calibration alongside kv/sampler/weights; kld_eval.rs exposes kld_self_score/kld_build_ref/kld_score; speed_bench.rs exposes SpeedBenchArgs/LatencyStats

*Verification: confirmed*

### [Medium] monolith — `crates/hipfire-runtime/examples/dflash_spec_demo.rs:1-2636`

**Observation:** dflash_spec_demo.rs is a single 2636-LOC example file that hipfire-eval anchors as its Dflash battery (executor_examples.rs::run_dflash_spec_demo_anchor, resolve_dflash_spec_demo_bin at L146). A 2.6k-line example is not a demo; it is an unmodularized speculative-decode driver whose correctness the eval harness depends on, yet it lives outside the library where it cannot be unit-tested or reused and can only be exercised by building the example binary.

**Recommendation:** Extract the reusable spec-decode driver logic into hipfire-runtime (or a hipfire-eval executor) as testable functions, leaving at most a thin example that calls into it. This lets the Dflash battery invoke library code in-process rather than depending on a giant standalone binary, and enables unit tests for the pure orchestration logic.

**Evidence:** wc -l dflash_spec_demo.rs = 2636; graphify shows hipfire-eval/executor_examples.rs::run_dflash_spec_demo_anchor and resolve_dflash_spec_demo_bin gating the Dflash battery

*Verification: confirmed*

### [Low] abstraction — `crates/hipfire-runtime/src/weights.rs:425-1355 (repeated gpu.mq_x_rot alias pattern); weight_gemv L384-706`

**Observation:** weight_gemv is a ~320-line match-dispatch function (L384-706) and the two-line idiom `buf: unsafe { gpu.mq_x_rot.as_ref().unwrap().buf.alias() }, shape: vec![gpu.mq_x_rot.as_ref().unwrap().buf.size() / 4]` is copy-pasted ~15 times across the file to build the rotated-activation scratch tensor. Each copy re-does the unwrap and the `/4` size math, so a change to the scratch layout must be edited in a dozen places and any one can drift. The surrounding DenseGemvRoute/DensePrerotatedGemvRoute enums are otherwise a good dispatch abstraction.

**Recommendation:** Add a small helper like `fn mq_x_rot_tensor(gpu: &Gpu) -> GpuTensor` (or a method on Gpu) that performs the alias + size/4 once, and call it from the route arms. This removes the repeated unwrap and centralizes the scratch-buffer contract.

**Evidence:** grep shows the identical `gpu.mq_x_rot.as_ref().unwrap().buf.alias()` / `.buf.size() / 4` pair at ~15 sites (L425,509,1119-1355); weight_gemv spans L384-706

*Verification: confirmed*

### [Low] missing-tests — `Assorted: calibration.rs:49,505,731 (2 tests / 867 LOC); dflash.rs:52-1163 (0 tests); ep.rs, mtp_mirror.rs, speed_bench.rs, kld_eval.rs, config.rs (0 tests)`

**Observation:** Several modules carrying pure, host-testable logic have little or no coverage. calibration.rs (867 LOC, 23 non-test unwrap/expect) tests only 2 things while compact_hessian_bytes (L49), combine_calib_parts (L505), and write_hessian_bf16_tril_diag_f32 (L731) — all pure byte/file math — are untested. dflash.rs (1609 LOC) has 0 tests despite DflashConfig parsing (L52-135) and gemm_dispatch shape logic being GPU-free. config.rs (RuntimeConfig::from_env env parsing), ep.rs, mtp_mirror.rs, speed_bench.rs, and kld_eval.rs (kld_mean/kld_p99 statistics) also have 0 tests. By contrast ddtree.rs (23), tp_shard.rs (18), triattn.rs (9), and cask.rs (7) are well covered, showing the crate can test pure logic — these gaps are unforced.

**Recommendation:** Add focused unit tests for the pure helpers: compact_hessian_bytes / combine_calib_parts round-trips, DflashConfig env parsing, RuntimeConfig::from_env truthy/falsey handling, and kld_mean/kld_p99 on known vectors. None require a GPU. Prioritize config.rs and calibration.rs since they parse external input.

**Evidence:** grep #[test] counts: calibration.rs 2, dflash.rs 0, config.rs 0, ep.rs 0, mtp_mirror.rs 0, speed_bench.rs 0, kld_eval.rs 0; calibration.rs has 23 non-test unwrap/expect before mod tests at L835

*Verification: confirmed*


## hipfire-arch-qwen35 (+ qwen35-vl)

*Subsystem key: `arch-qwen35` — 8 finding(s)*

**Subsystem assessment:** crates/hipfire-arch-qwen35 owns the Qwen3.5 hybrid (DeltaNet + FullAttention, dense + MoE) forward pass, weight loading, KV/DeltaNet state layout, and the qwen35-specific speculative/MTP glue. The crate is functionally rich and, notably, its pure plan/contract-builder logic (batch-shape validation, pointer-table planning, dispatch-admissibility) is well factored and well unit-tested in qwen35.rs's own mod tests. The dominant health risk is scale: qwen35.rs is a single 32,648-LOC flat file with 70 pub fns, and it plus speculative.rs contain several 1,500-5,600-LOC god functions where per-layer GPU dispatch, policy env reads, and pure index/mask math are all interleaved. Secondary risks are a duplicated single-vs-multi-GPU decode layer loop (fix-in-two-places hazard), a boolean/option parameter explosion behind 29 too_many_arguments allows, ~60 scattered HIPFIRE_* env toggles, and pure contract tests for the sibling hipfire-generate crate parked inside the hipfire-daemon binary. None of these are correctness bugs today; they are maintainability and testability structural debt at large scale.

### [High] monolith — `crates/hipfire-arch-qwen35/src/qwen35.rs:1-32648`

**Observation:** qwen35.rs is a single flat 32,648-LOC file exposing 70 `pub fn`, 53 structs, 10 enums, and 18 impl blocks, with only two inline submodules (`mod q35_op` at L27229, `mod tests` at L30476). It mixes at least eight distinct responsibilities in one compilation unit: env/mode parsing (L44-284), config extraction (L318-687), weight/state structs (L826-1055, L1330-1657), the entire weight-loader family (L1658-7200), MoE decode (L7204-8720), prefill-session batch contracts + pointer tables (L8822-12816), the prefill-chunk and decode layer loops (L12817-30434), plus an op-lowering DSL (L27229-27932).

**Recommendation:** Split into a `src/qwen35/` module tree along the region boundaries: `config.rs` (Qwen35Config + config_from_hfq/safetensors), `weights.rs` + `state.rs` (weight/DeltaNet structs), `load/` (load_weights*, load_*_into, slab/rq/paro/moe loaders), `moe.rs` (moe_ffn_decode*), `prefill_session/` (the pure contract/plan/pointer-table family, which is already independently tested), `prefill.rs` (forward_prefill_chunk + batch wrappers), `decode.rs` (forward_scratch* layer loops), `lowered.rs` (q35_op), and `multi.rs` (EP/multi-GPU). Re-export the trait-facing surface from a slim `qwen35/mod.rs`. This is mechanical (no logic change) and unblocks every finding below.

**Evidence:** wc -l = 32648; grep counts: 70 `^pub fn`, 53 `^(pub )?struct`, 18 `^impl`, 10 enums; only `mod q35_op` (L27229) and `mod tests` (L30476) as internal namespaces

*Verification: confirmed*

### [High] monolith — `fa_bridge_valid_slots is L2332-2354 (~22 LOC, not 3062); the real god functions are forward_prefill_chunk (L16941, 5571), forward_scratch_layers (L23403, 3006), spec_step_dflash (L7020, 3278)`

**Observation:** Multiple functions span thousands of lines between top-level fn defs: forward_prefill_chunk ~5,577 LOC (L16941-22518), forward_scratch_layers ~3,011 LOC (L23403-26414), forward_scratch_layers_multi ~1,730 LOC (L28326), prefill_moe_ffn_body_batched ~1,592 LOC (L15253); in speculative.rs, spec_step_dflash ~3,293 LOC (L7020-10313) and fa_bridge_valid_slots ~3,062 LOC. These bodies interleave per-layer kernel dispatch, env-flag reads, and pure index/mask/shape math, so the non-GPU logic embedded inside them cannot be unit-tested without extraction.

**Recommendation:** Extract the pure inner logic (mask/tree-bias construction, band/round index math, per-dtype branch selection) into small free functions returning plain values, mirroring the already-extracted-and-tested helpers (moe_decode_dispatch_flags_for_dtypes, is_batchable_la, prefill_batch_pbs_eligible). Decompose each layer loop into a per-layer step fn plus attention/FFN/MoE sub-steps so the 5k-line bodies become orchestrators over testable units. Prioritize forward_prefill_chunk and spec_step_dflash.

**Evidence:** awk span between consecutive `^(pub )?fn` defs: forward_prefill_chunk=5577, forward_scratch_layers=3011, forward_scratch_layers_multi=1730, prefill_moe_ffn_body_batched=1592; speculative.rs spec_step_dflash=3293

*Verification: adjusted — Brace-matched bodies confirm the three named functions plus others are genuinely huge (forward_prefill_chunk 5571, forward_scratch_layers 3006, forward_scratch_layers_multi 1724, prefill_moe_ffn_body_batched 1552, spec_step_dflash 3278 lines). But the observation's fa_bridge_valid_slots ~3,062 LOC is false: it starts at L2332 and closes at L2354 (22 lines, returns Vec<bool>). The awk span-between-fns evidence method overcounted by counting three intervening impl blocks (GdnTape, GdnTapeShards, DeltaNetTape). Core High-severity claim (multiple thousand-line functions blocking unit testing) stands.*

### [High] duplication — `crates/hipfire-arch-qwen35/src/qwen35.rs::forward_scratch_layers (L23403) vs forward_scratch_layers_multi (L28326)`

**Observation:** The single-GPU decode layer loop (forward_scratch_layers, 3,011 LOC) and the multi-GPU one (forward_scratch_layers_multi, 1,730 LOC) are parallel copies over the same kernel vocabulary (rmsnorm_batched, gemv_paro4g128t_prerotated, fused_qkvza_paro4g128t, kv_cache_write_q8_0, rope/repeat_interleave, etc.). The multi copy already carries hazards that exist only in it — the forward_scratch_multi header (L30063-30081) documents that ct!()/st!() in the multi loop silently fall back to a wrong-device givens tensor for asym-KV. This is a classic fix-in-two-places structure: any kernel-order or state-quant correctness fix must be applied to both loops or they diverge.

**Recommendation:** Note that the prefill path already solved this: forward_prefill_batch_multi (L30141) reuses the single forward_prefill_chunk band-restricted via PrefillBandCtx. Apply the same pattern to decode — introduce a DecodeBandCtx and drive both single- and multi-GPU decode through one banded layer-loop body, so per-device replicas (givens, pos_buf) are handled in one place instead of duplicated.

**Evidence:** Both regions share the identical kernel-call set (grep uniq -c of `gpu\.[a-z_]+\(`); forward_prefill_batch_multi docstring (L30141+) explicitly states prefill was unified via forward_prefill_chunk + PrefillBandCtx while decode was not

*Verification: confirmed*

### [Medium] abstraction — `crates/hipfire-arch-qwen35/src/qwen35.rs:13660-14147 (forward_prefill_batch family)`

**Observation:** forward_prefill_batch is a 4-level wrapper chain: forward_prefill_batch (L13660) -> forward_prefill_batch_with_pbs (L13729) -> forward_prefill_batch_with_pbs_opts (L13781, the real body), with forward_prefill_batch_force_q8_gdn_per_token (L13694) as a fifth entry, each forwarding 12-16 positional args and adding trailing bool/Option flags (needs_last_token_logits, force_q8_gdn_per_token, mask_override, max_layer). The file carries 29 `#[allow(clippy::too_many_arguments)]` to suppress the resulting lint. Trailing-bool call sites (e.g. `true, true` at L13723, `true, false` at L13777) are position-error-prone.

**Recommendation:** Introduce a `PrefillBatchOpts` struct (pbs, mask_override, max_layer, needs_last_token_logits, force_q8_gdn_per_token) with a Default and builder-style setters, and collapse the wrapper chain to one entry taking `(gpu, weights, config, tokens, start_pos, kv, dn_state, scratch, io: PrefillBatchIo, opts: PrefillBatchOpts)`. This removes the too_many_arguments allows, kills the boolean-blindness at call sites, and makes the variant surface (31 forward* fns total) legible.

**Evidence:** grep -c too_many_arguments qwen35.rs = 29; 31 forward* entry points; wrapper bodies at L13674-13691, L13741-13780 forward all args unchanged before appending literal bool flags

*Verification: confirmed*

### [Medium] utils-sprawl — `crates/hipfire-arch-qwen35/src/qwen35.rs (24 *_from_env/*_enabled fns; ~60 HIPFIRE_* vars)`

**Observation:** Dispatch and policy decisions are driven by ~60 distinct HIPFIRE_* environment variables read via 84 `env::var` call sites and 24 dedicated `_from_env()/_enabled()` functions (e.g. kld_direct_f16kv_attention_enabled L12863, q8_fa_attention_row_loop_enabled L12928, moe_grouped_gemm_path2_enabled_from_env L14906, forward_lowered_enabled L27830). These toggles are read ad hoc deep inside the hot path rather than resolved once. Because many gate arch-specific kernel paths (gfx1151/gfx11/gfx12), the portability policy (RDNA2/3/4) is smeared across the file instead of centralized. Parsing itself is defensive (`.parse().ok().unwrap_or(default)`), so this is a maintainability/portability concern, not a panic risk.

**Recommendation:** Resolve all env toggles once into a `Qwen35RuntimePolicy` struct at model-load time (with the arch string available), thread it through forward calls, and keep env parsing in one `policy.rs` module with a single table mapping var name -> field + default. This makes the RDNA2/3/4 gating auditable in one place and lets the policy resolver be unit-tested exhaustively (a few such tests already exist, e.g. moe_prefill_paro_i8_env_policy).

**Evidence:** grep -oE '"HIPFIRE_[A-Z0-9_]+"' unique ~60 names; 84 `env::var` occurrences; 24 fns matching `^fn .*_(enabled|from_env)`

*Verification: confirmed*

### [Medium] test-structure — `crates/hipfire-daemon/src/main.rs:330 (mod generate_batch_prefill_tests) vs crates/hipfire-generate/src/lib.rs:1226,1298`

**Observation:** The assignment's named pure builders build_qwen35_fused_dense_prefill_batch_contract and validate_qwen35_fused_grouped_moe_prefill_batch_preflight are pure `Result<_, String>` functions but live in hipfire-generate (lib.rs L1298, L1226), not in arch-qwen35. Their tests are parked in a 63-test `generate_batch_prefill_tests` module inside the hipfire-daemon *binary* (main.rs L330+), which reaches across into hipfire-generate, hipfire-state, and qwen35. hipfire-generate has its own `mod tests` (L1560, 27 tests), so the split is arbitrary: pure-lib contract tests sit in a binary crate where `cargo test -p hipfire-generate` will never run them.

**Recommendation:** Move the contract/preflight tests from hipfire-daemon/src/main.rs into hipfire-generate's own test module (or a tests/ integration file in that crate) next to the code under test, keeping only daemon-wiring tests in the binary. For contrast/positive note: the arch-local equivalents (validate_dense_prefill_session_batch_*, build_dense_prefill_session_batch_*, pointer-table plans, L10356-11254) are correctly unit-tested in qwen35.rs's own mod tests (dense_session_prefill_* cases) — that is the pattern to replicate.

**Evidence:** grep: contract fns defined in hipfire-generate/src/lib.rs:1226,1298; daemon main.rs mod at L331 with 63 `#[test]` importing them via `use hipfire_generate::{...}` (L334-343); hipfire-generate own mod tests at L1560

*Verification: confirmed*

### [Low] duplication — `bf16_to_f32 dup is in crates/hipfire-arch-lfm2moe/src/lfm2moe.rs:33 (and rdna-compute examples), not hipfire-runtime/src/weight_pager.rs; align_up_usize (L12829) has an extra debug_assert so not byte-identical`

**Observation:** align_up (L4511) and align_up_usize (L12829) have byte-identical bodies `(x + align - 1) & !(align - 1)`; the file also carries three near-identical `*_slice_as_bytes` helpers (f32 L7703, u64 L9831, i32 L9835) plus scattered dtype/byte converters (bf16_to_f32 L2071, bf16_bytes_to_f32 L2307, bf16_bytes_to_f16_bytes L2313, gib L4249, load_throughput_gibs L4253). These are grab-bag primitives duplicated within the file and, in a couple cases, across crates (bf16_to_f32 also defined in hipfire-runtime/src/weight_pager.rs).

**Recommendation:** Collapse align_up/align_up_usize to one function, replace the three slice_as_bytes helpers with a single generic `fn as_byte_slice<T: bytemuck::Pod>(&[T]) -> &[u8]` (or a small local generic if bytemuck is undesired), and hoist the bf16/dtype/throughput converters into hipfire-primitives so every arch crate shares one implementation.

**Evidence:** sed of L4511 and L12829 shows identical body; grep found f32/u64/i32 `_slice_as_bytes` at L7703/9831/9835; bf16_to_f32 defined in 2 crates

*Verification: adjusted — align_up (L4511) and align_up_usize (L12829) both exist and share the arithmetic `(x + align - 1) & !(align - 1)`, but they are NOT byte-identical: align_up_usize adds a debug_assert!(align.is_power_of_two()) line and uses x vs v. The three *_slice_as_bytes helpers (f32 L7703, u64 L9831, i32 L9835) are confirmed. However bf16_to_f32 is NOT defined in hipfire-runtime/src/weight_pager.rs (not anywhere in hipfire-runtime); the actual cross-crate duplicate is in hipfire-arch-lfm2moe/src/lfm2moe.rs L33 plus rdna-compute/nemotron examples. Core dup/hygiene claim holds; two specific facts corrected. Low severity appropriate.*

### [Low] missing-tests — `crates/hipfire-arch-qwen35/src/mtp_head.rs (2683 LOC, 0 tests); mtp_spec.rs::spec_step_mtp_compressed_serial (L2155, ~1134 LOC)`

**Observation:** Remaining smaller structural items folded here: (1) mtp_head.rs is 2,683 LOC with 12 pub fns and zero `#[test]`, despite containing extractable pure logic (embed_lookup_into L2181, offset/shape math in load_mtp_head_at_offset L827). (2) mtp_spec.rs has a 1,134-LOC god function spec_step_mtp_compressed_serial (L2155) and 6 too_many_arguments allows. (3) speculative.rs carries 5 too_many_arguments allows and the two 3k-LOC functions already noted. (4) The build.rs for this crate is 21.5 KB of NPU-kernel build scripting, another sizable untested surface. Each is a scheduled-cleanup item rather than an active risk.

**Recommendation:** When splitting the crate (finding 1), give mtp_head/mtp_spec the same treatment: extract pure offset/shape/mask helpers into free functions and add unit tests (mtp_spec already models this with softmax/threshold tests). Track the too_many_arguments allows as debt to be retired via option structs.

**Evidence:** wc/grep: mtp_head.rs 2683 lines, 0 `#[test]`, 12 pub fn; mtp_spec.rs spec_step_mtp_compressed_serial span 1134; too_many_arguments counts mtp_spec=6, mtp_head=6, speculative=5; build.rs 21522 bytes

*Verification: confirmed*


## Arch crates A: deepseek4, lfm2moe, minimax, zaya, nemotron

*Subsystem key: `arch-family-a` — 11 finding(s)*

**Subsystem assessment:** This subsystem holds five model-arch crates (deepseek4, lfm2moe, minimax, zaya, nemotron) that each translate a config + weights into a GPU forward pass. The dominant health problem is deepseek4: a 9202-LOC forward.rs and a 3536-LOC arch-specific dispatch module wedged inside the generic rdna-compute crate, both driven by a 54-field lazily-allocated God-state struct read through ~180 as_ref().unwrap() calls. A recurring structural pattern across the family is full duplication of the per-layer math between decode (single-token) and prefill (batched) paths, and copy-paste of per-arch scaffolding (kld.rs adapters, superop constructors, lowered-forward toggles) that a shared trait/macro would collapse. Every arch also carries two live forward implementations (hand-written execute path plus the newer lowered superop path) behind HIPFIRE_FORWARD_LOWERED, doubling maintenance surface. The bright spot is nemotron: 37 files is not fragmentation but proper modularization (focused mlp/attn/moe/ssd/block modules plus example binaries) with co-located unit tests on pure config/shape logic — the pattern the heavier crates should move toward. Main risks: two hand-synced copies of numerically-sensitive kernels, order-dependent state that panics rather than type-errors, and an arch leaking its private concepts into the shared GPU dispatch layer.

### [High] monolith — `crates/hipfire-arch-deepseek4/src/forward.rs:1-9202`

**Observation:** forward.rs is a single 9202-LOC file with 85 functions spanning unrelated responsibilities: single-token decode (decode_step_body ~L1824), batched prefill (forward_prefill_batch* + PrefillBatchScratch L5429-6017), MTP speculative heads (mtp_* L2552-3384), expert-parallel forward (forward_ep L2314), MoE routing math (moe_route, bias_aware_topk_weights), RoPE/YaRN math (rope_yarn_corr_dim, apply_tail_rope), and the MLA compressor/indexer. No submodule boundaries exist; everything shares file scope. This is the single largest maintainability liability in the family.

**Recommendation:** Split forward.rs into a module directory: decode.rs, prefill.rs, mtp.rs, ep.rs, attention.rs (compressor/indexer/SWA), moe.rs (routing + ffn), and rope.rs. Pure functions like rope_yarn_corr_dim, moe_route, bias_aware_topk_weights, and gather_normalized_weights should live in a small rope/routing module where they can be tested in isolation. The batched vs decode split (see separate finding) becomes far clearer once decode.rs and prefill.rs are distinct.

**Evidence:** wc -l = 9202; grep counts 85 fn definitions; distinct concerns at L1824 decode_step_body, L2314 forward_ep, L2552 mtp_forward, L5429 PrefillBatchScratch, L8670 forward_prefill_batch.

*Verification: confirmed*

### [High] crate-boundary — `crates/rdna-compute/src/dispatch/deepseek4.rs:1-3536`

**Observation:** The generic GPU-dispatch crate rdna-compute contains a 3536-LOC module of 65 pub methods on `impl Gpu` named for DeepSeek-V4-private concepts: compressor_* (MLA), hc_* (hyper-connection 4-stream), indexer_* (sparse top-K), hash_router_*, deepseek4_attn_swa_topk_*. It is declared unconditionally (dispatch/mod.rs:24, not feature-gated), has 0 tests, and grep confirms it is called only from the deepseek4 arch crate. This bolts one architecture's vocabulary onto the shared Gpu type every other arch also uses, inverting the intended dependency direction (arch depends on compute, not compute on arch).

**Recommendation:** Move these methods out of rdna-compute. Expose only generic primitives (gather, softmax-pool, top-k, scaled-gemv) on Gpu and implement the deepseek4-specific sequencing in the arch crate as free functions taking `&mut Gpu`, or behind an extension trait defined in the arch crate (`trait Deepseek4Dispatch { ... } impl Deepseek4Dispatch for Gpu`). At minimum feature-gate the module so a build without the deepseek4 arch does not compile 3536 LOC of dead dispatch. The same applies to dispatch/zaya_cca.rs.

**Evidence:** grep -c pub fn = 65; wc -l = 3536; dispatch/mod.rs:24 `mod deepseek4;` unconditional; callers grep resolves only to hipfire-arch-deepseek4/src/{deepseek4,forward}.rs; 0 `#[test]`.

*Verification: confirmed*

### [High] duplication — `crates/hipfire-arch-deepseek4/src/forward.rs::q_lora(4980)/q_lora_batched(8527)`

**Observation:** The decode path and prefill path re-implement the same per-layer algorithms as parallel functions that differ only in single-tensor-from-state vs batched-tensor-from-PrefillBatchScratch and the batched kernel suffix. q_lora (L4980) and q_lora_batched (L8527) are the identical 6-step sequence (rmsnorm[+FWHT] -> gemv wq_a -> q_norm -> rotate -> gemv wq_b -> per-head rmsnorm) with the same `if wq_a_needs_fwht`/`if wq_b_needs_fwht` control flow. The same twinning exists for mhc_pre/mhc_pre_batched, kv_joint/kv_joint_batched, hc_attn_mix/hc_attn_mix_batched, hc_ffn_mix/hc_ffn_mix_batched, apply_tail_rope/apply_tail_rope_batched — roughly 2600 LOC of batched code mirroring ~2000 LOC of decode code. Any numeric fix must be applied twice; divergence is a silent correctness bug.

**Recommendation:** Introduce a tensor-op abstraction that unifies batch_size=1 with batch_size=N: a small trait (e.g. `trait NormGemvOps { fn rmsnorm(...); fn gemv_auto(...); fn rotate_mq(...); }`) implemented once for the single path and once for the batched path, with the step sequence written once against the trait. Alternatively make decode a batch=1 call into the batched code (the kernels already accept a batch arg). Either collapses each twin pair to one body.

**Evidence:** q_lora L4980-5110 and q_lora_batched L8527-8651 are the same 6 numbered steps; sibling pairs at L4615/L8331 (mhc_pre), L4916/L8465 (kv_joint), L4742/L6018 (hc_attn_mix), L3908/L8293 (hc_ffn_mix), L4853/L8422 (apply_tail_rope).

*Verification: confirmed*

### [Medium] coupling — `crates/hipfire-arch-deepseek4/src/deepseek4.rs::DeepseekV4State(663-924)`

**Observation:** DeepseekV4State is a 54-field struct of almost entirely Option<GpuTensor> scratch/cache slots, allocated lazily on first use. forward.rs contains 78 `state.<field>.is_none()` alloc guards and 181 `.as_ref().unwrap()` reads. The invariant 'slot X was allocated before it is read' is enforced only by call ordering, not the type system: any decode/prefill/MTP/EP path that reads a slot a sibling path was responsible for allocating panics at runtime instead of failing to compile. This God-state also couples every forward function to the same mutable object.

**Recommendation:** Group the slots into cohesive sub-structs (AttnState, MoeScratch, MtpState, PrefillScratch) that are constructed fully-initialized in a single `new(gpu, cfg)` returning Result, eliminating the Option+lazy-alloc+unwrap triad. Where laziness is genuinely needed, wrap allocation in a `get_or_alloc(&mut self, gpu) -> &GpuTensor` accessor so the alloc-then-use contract is centralized and the 181 unwraps disappear.

**Evidence:** State struct L663-924 = 54 fields; forward.rs grep: 78 `state.*.is_none()`, 181 `.as_ref().unwrap()`; concrete cluster at q_lora L5041-5047 (hc_x_in/tmp/tmp_plain/q_lat/q all `.as_ref().unwrap()`).

*Verification: confirmed*

### [Medium] duplication — `crates/hipfire-arch-{deepseek4,minimax,lfm2moe}/src/kld.rs`

**Observation:** Each arch's kld.rs is a near-identical copy: a forward_chunk_scored-style teacher-forcing function plus `struct XxxKldForward<'a> { weights, config }` and `impl ChunkScoredForward for XxxKldForward`. Diff of deepseek4 vs minimax kld.rs shows the only differences are the type names (DeepseekV4* vs MiniMax*) and doc comments; the adapter structure is identical. This pattern is copy-pasted across at least three crates (deepseek4/minimax/lfm2moe all 70-71 LOC).

**Recommendation:** Factor the adapter into a generic blanket: define an arch trait supplying `new_state`, `feed_token_logits`, and the config/weights refs, then `impl<A: ArchKld> ChunkScoredForward for KldForward<A>` once in the shared kld crate. Each arch provides a tiny trait impl instead of a full copy of the adapter. A declarative macro is an acceptable lighter-weight alternative if the trait bounds get awkward.

**Evidence:** diff deepseek4/kld.rs minimax/kld.rs = only type-name + doc lines differ; wc -l 71/70/71 for the three crates; each defines `struct *KldForward` + `impl ChunkScoredForward`.

*Verification: confirmed*

### [Medium] duplication — `crates/hipfire-arch-*/src/forward.rs (hand path vs *_lowered)`

**Observation:** Every arch maintains TWO live forward implementations: the hand-written execute loop (decode_step_body, 536 LOC in minimax at L190-725) and the newer superop lowered loop (decode_step_body_lowered), selected at runtime by HIPFIRE_FORWARD_LOWERED (default ON per the deepseek4/minimax comments). The hand path is now the fallback but is still fully present and must be kept behaviorally byte-identical to the lowered path. This is a transitional duplication that doubles the maintenance surface of the hottest code in each crate.

**Recommendation:** Now that the lowered path is default-on and oracle-validated, schedule removal of the hand execute loops (or reduce them to a thin, explicitly-deprecated debug fallback compiled only under a cfg feature). Track per-arch cutover so the dead hand path does not silently rot out of sync with the lowered path it is supposed to mirror.

**Evidence:** deepseek4 decode_step_body L1824 + decode_step_body_lowered L2259; minimax L190 (536 LOC) + L1190; lfm2 decode_step_layers_and_head L1137 + _lowered L1964; toggles default to `!= Some("0")`.

*Verification: confirmed*

### [Medium] duplication — `crates/hipfire-arch-zaya/src/gpu.rs::gpu_forward_serve(665)/gpu_forward_calib(953)/gpu_decode(1204)/gpu_forward_prefill(1453)`

**Observation:** zaya packs four ~250-340 LOC forward entry points into one 1793-LOC gpu.rs, and they share the same per-layer skeleton: gpu_forward_serve and gpu_forward_calib have nearly identical op counts (21 vs 20 gemv_seq/rmsnorm/residual calls) and differ mainly in capture/calibration hooks; gpu_decode is the single-token subset; gpu_forward_prefill is the batched superset. The layer loop is written four times.

**Recommendation:** Extract the shared per-layer body into one function parameterized by a small hook enum/closure (serve = no capture, calib = capture activations, decode = batch 1, prefill = batch N). The four public entry points then become thin wrappers that set up state and call the shared loop, cutting ~700 LOC and removing the four-way sync burden.

**Evidence:** gpu.rs L665/953/1204/1453; measured op-call counts 21/20/11/28 over the four ranges; file is 1793 LOC.

*Verification: confirmed*

### [Medium] missing-tests — `crates/hipfire-arch-minimax/src/forward.rs (4 tests total in crate)`

**Observation:** minimax has only 4 #[test] in the whole crate and exactly one in forward.rs (minimax_program_is_attend_then_moe, L1769) despite a 1773-LOC forward.rs and 1199-LOC minimax.rs. Pure host-side logic that could be unit-tested without a GPU — MoE top-k selection, config field derivation, and shape/stride math — is exercised only indirectly through GPU end-to-end paths. By contrast deepseek4 (51 tests) and nemotron (18 tests) unit-test their pure routing/config functions.

**Recommendation:** Mirror deepseek4's approach: extract minimax's routing top-k and any position/shape math into free functions and add table-driven unit tests (including zero-sum / k>=n edge cases, as deepseek4's bias_aware_topk tests already do). Add a config-parse test that pins the released MiniMax-M2 config.json schema, matching deepseek4::parses_real_deepseek4_config_json.

**Evidence:** crate-wide `#[test]` count = 4; forward.rs has one test at L1769; minimax.rs config parse has a test but routing math (minimax_moe_block L795) has none; deepseek4 has 51 tests incl. L9105-9188 topk/gather suite.

*Verification: confirmed*

### [Low] duplication — `crates/hipfire-arch-{deepseek4,minimax}/src/forward.rs (ds4_superop / mm_superop / *_forward_lowered_enabled)`

**Observation:** The superop scaffolding helpers are byte-identical across crates. ds4_superop (deepseek4 L2226) and mm_superop (minimax L1155) are the same constructor building a SuperOp with an empty OpBinding. *_forward_lowered_enabled() is the same OnceLock read of HIPFIRE_FORWARD_LOWERED in deepseek4 (L2250), minimax (L1181), and lfm2moe (L1955). These belong in the shared superop module, not copied per arch.

**Recommendation:** Add `SuperOp::bare(kind)` (and lfm2's `SuperOp::with_weight(kind, code)`) plus a `superop::forward_lowered_enabled()` helper to hipfire-dispatch/pipeline/superop.rs, and have each arch call them. Removes a small but pure copy-paste and centralizes the env-toggle semantics.

**Evidence:** ds4_superop L2226-2236 == mm_superop L1155-1165 verbatim; three identical `*_forward_lowered_enabled` OnceLock bodies at deepseek4 L2250, minimax L1181, lfm2 L1955.

*Verification: confirmed*

### [Low] missing-tests — `crates/rdna-compute/src/dispatch/{deepseek4.rs,zaya_cca.rs}`

**Observation:** The two arch-specific dispatch modules (3536 + ~490 LOC) have zero unit tests. They contain host-side pure logic that is GPU-independent and testable: the nibble_expand_int4_to_int8 packing (L382), oq4/oq8 repack helpers in zaya gpu.rs (oq4_pack_arch_combined L63, oq4_to_oq8_combined L101), and various index/length computations. These are exactly the truncation/packing paths where an off-by-one silently corrupts weights.

**Recommendation:** Extract the byte-level pack/expand and length-math helpers into pure functions (several already are) and add unit tests over known-answer vectors, including boundary sizes (odd K, single-group, max nibble). This is achievable without a GPU and guards the most corruption-prone code.

**Evidence:** 0 `#[test]` in dispatch/deepseek4.rs and dispatch/zaya_cca.rs; pure helpers nibble_expand_int4_to_int8 L382, zaya gpu.rs oq4_pack_arch_combined L63 / oq4_to_oq8_combined L101 / oq8_combined L133.

*Verification: adjusted — Core claim holds narrowly: dispatch/deepseek4.rs (3536 LOC) and dispatch/zaya_cca.rs (561 LOC, not ~490) both have 0 tests, and nibble_expand_int4_to_int8 L382 is genuinely untested. But the finding's headline 'most corruption-prone' examples -- the zaya oq repack helpers oq4_pack_arch_combined L63, oq4_to_oq8_combined L101, oq8_combined L133 -- are ALREADY unit-tested in zaya gpu.rs (oq8_combined_layout L1715 and oq4_pack_and_oq8_expand_layout L1732 exercise all three with known-answer 256-elem vectors). Those helpers also live in the arch crate, not the cited dispatch modules. So the test gap is real but materially overstated and the recommendation is partly redundant; severity stays Low.*

### [Low] abstraction — `crates/hipfire-arch-deepseek4/src/forward.rs (as-cast + unwrap density)`

**Observation:** deepseek4/forward.rs performs 340 truncating as-casts and 198 unwrap() calls with no centralized bounds checking. Position/size values are cast usize/u32 -> i32 in many places (e.g. `batch_size as i32` L1083/L1155/L1290, `(absolute_event_pos / ratio * ratio) as i32` L1183-1188, `position as i32` L1692/L1795). These are individually low-risk at realistic model sizes but are a diffuse hazard: a large batch or context silently wraps to a negative kernarg with no debug assertion.

**Recommendation:** Route position/size narrowing through a single checked helper (e.g. `fn to_i32(x: usize) -> i32 { i32::try_from(x).expect("kernarg overflow") }` or a debug_assert-guarded cast) so an overflow is a loud panic at the conversion site rather than a corrupt kernel launch. Apply the same for the batched-prefill size casts.

**Evidence:** grep counts: 340 as-casts, 198 `.unwrap()` in forward.rs; sample truncating casts at L1083, L1155, L1183-1188, L1290, L1692, L1795.

*Verification: confirmed*


## Arch crates B: qwen2, gemma3, gemma3-vl, llama, dots-ocr, toy

*Subsystem key: `arch-family-b` — 6 finding(s)*

**Subsystem assessment:** Six arch crates: qwen2, gemma3, gemma3-vl, llama, dots-ocr, toy. The architectural layering is broadly healthy — the VL crates (dots-ocr, gemma3-vl) correctly *depend on and reuse* their base-arch crates rather than forking them (dots-ocr uses Qwen2Config/Weights/State and delegates the text forward to hipfire_arch_qwen2::qwen2::forward_step; gemma3-vl uses Gemma3Config/load_weights_prefixed and only adds a distinct SigLIP encoder), so the "VL duplicates base arch" risk does not materialize. The arch.rs trait-adapter files are thin, consistent, and idiomatic across the family. The two real structural problems are (1) leftover half-finished pipeline migrations that left dead/redundant forward code — qwen2's hand forward path double-computes QKV and gate/up, and arch-llama ships a 247-line dead reimplementation of the runtime forward — and (2) an acknowledged-but-unresolved cross-arch copy-paste of weight/norm/bias loaders (the promised hipfire_runtime::transformer::* module does not exist). qwen2.rs is a 2115-line monolith spanning six responsibilities. Testable pure logic (config parsing) is well covered; forward paths are GPU-bound and only structurally testable.

### [High] duplication — `crates/hipfire-arch-qwen2/src/qwen2.rs:1010-1234 (forward_step_after_x)`

**Observation:** The non-lowered hand forward path computes QKV and FFN gate/up TWICE. Block A (lines 1013-1041) runs rmsnorm_f32 -> tmp then fused_qkv_hfq4g256 / three weight_gemv into state.q/k/v; with no read of q/k/v in between, Block B (execute_steps at 1049-1080) immediately re-runs RmsnormAutomatic and three Prerotated Gemvs into the SAME state.q/k/v, so Block A is 100% dead compute. The FFN mirrors this in reverse: execute_steps writes gate/up (1189-1215), then weight_gemv overwrites gate/up (1218-1219). This path is the HIPFIRE_FORWARD_LOWERED=0 escape hatch that the code (comment at 1519-1522) uses as the md5 A/B *parity reference* for validating the default lowered path — so the reference itself does 1.5-2x redundant work, and for a rotation (MQ) dtype Block D's weight_gemv on un-rotated tmp could diverge from Block B's Prerotated(x_rot), undermining the parity claim.

**Recommendation:** Delete Block A (lines 1013-1041) and the FFN weight_gemv pair (1218-1219); keep only the execute_steps sequences so the hand path matches the lowered path op-for-op. Compare against gemma3's forward_after_x, which does each projection exactly once. Longer term, retire the hand path entirely now that the lowered super-op path is default-on, leaving one forward implementation.

**Evidence:** qwen2.rs:1014 rmsnorm_f32->tmp and 1023/1038-1040 write q/k/v; qwen2.rs:1049-1080 execute_steps RmsnormAutomatic + 3x Gemv overwrite the same state.q/k/v with no intervening read; qwen2.rs:1189-1215 vs 1218-1219 duplicate gate/up; contrast gemma3 forward.rs:262-265 (single weight_gemv per projection).

*Verification: confirmed*

### [Medium] duplication — `crates/hipfire-arch-llama/src/arch.rs:143-390 (Llama::forward_scratch_layers)`

**Observation:** This 247-line pub method is a full reimplementation of the LLaMA decode+sample forward (rmsnorm/QKV/rope/attention/o_proj/FFN/lm_head/sample_top_p) that duplicates hipfire_runtime::llama::forward_scratch_layers (llama.rs:1861). A repo-wide grep finds zero callers: LlamaBackend's SimpleAr impl instead calls the runtime functions llama::forward_prefill_batch / forward_scratch_embed / forward_scratch_compute (arch.rs:428-455). It is orphaned migration scaffolding (see the stale 'new-dispatch' / 'ModelDispatch (to be created)' comment block at 108-135). This is ~half the crate's hand-written code and will silently drift from the live runtime forward.

**Recommendation:** Delete Llama::forward_scratch_layers and the stale migration comment block (108-135). The crate's real job is the thin Architecture/SimpleAr/ServingBackend adapter plus the `pub use hipfire_runtime::llama` re-export, all of which delegate to the runtime; the dead method adds only drift risk. If a dispatch-family variant is genuinely wanted, gate it behind the feature the comments describe and add a caller, otherwise remove it.

**Evidence:** arch.rs:143 `pub fn forward_scratch_layers`; repo-wide grep for forward_scratch_layers shows only the definition + comments in arch-llama, with the live callers pointing at runtime/src/llama.rs:1861; LlamaBackend::decode_step (arch.rs:445-455) calls llama::forward_scratch_embed/compute, not this method.

*Verification: adjusted — Location and orphaned-duplicate facts confirmed: forward_scratch_layers (143-390, ~247 lines) is a full execute_steps reimplementation of decode+sample; repo-wide grep shows only comments (117/122/131) and the definition, never a call. LlamaBackend delegates to llama::forward_prefill_batch (428) and forward_scratch_embed/compute (445-455). Secondary cite llama.rs:1861 is actually 1843 (trivial). Severity adjusted High->Medium: it is unreachable dead code, so no ACTIVE correctness risk to shipped paths; it is a clear antipattern/drift risk worth scheduled removal (Medium), not a High that blocks maintainability.*

### [Medium] monolith — `crates/hipfire-arch-qwen2/src/qwen2.rs:1-2116`

**Observation:** qwen2.rs is a 2115-line single file carrying six distinct responsibilities: config parsing (66-192), weight structs + per-tensor loaders (201-675), per-decode GPU state alloc/free (710-836), single-token forward + prefill (873-1508), and the #397 super-op lowering machinery (q2_op module, Qwen2Bindings, LayerProgram, lowered driver at 1529-1913). By contrast the sibling gemma3 crate splits the same concerns into config.rs / weights.rs / forward.rs / calibration.rs. The size and the mixed hand-path + lowered-path forward (see the double-compute finding) make the hot path hard to review.

**Recommendation:** Adopt the gemma3 module layout for qwen2: split into config.rs, weights.rs (structs + loaders), state.rs, forward.rs (decode + prefill), and lowering.rs (super-op program + bindings), re-exported from qwen2.rs or lib.rs. This isolates the forward hot path from loader boilerplate and makes the lowered-vs-hand paths reviewable side by side.

**Evidence:** wc -l qwen2.rs = 2116; single file spans Qwen2Config (66), Qwen2Weights::load (234), Qwen2State (710), forward_step (873), forward_prefill_batch_embeds (1284), mod q2_op + Qwen2Bindings + forward_step_after_x_lowered (1529-1913); gemma3 splits equivalent scope across config.rs/weights.rs/forward.rs.

*Verification: confirmed*

### [Medium] abstraction — `crates/hipfire-arch-qwen2/src/qwen2.rs:524-675; crates/hipfire-arch-dots-ocr/src/dots_ocr.rs:526-590`

**Observation:** load_norm_weight_raw and load_bias_f32 are byte-for-byte identical between qwen2 and dots-ocr except for the 'qwen2:'/'dots-ocr:' panic prefixes (same quant_type match on 1/2/16, same f16/bf16 decode, same assert_eq! on element count). qwen35 carries a third copy (qwen35.rs:2369), and load_weight_tensor / load_f16_or_dequant / smart_resize are likewise forked across qwen2, qwen35, dots-ocr and qwen35-vl. Every one of these is tagged 'TODO(transformer-extraction)' pointing at a hipfire_runtime::transformer::* module that does not exist, so the debt is documented but unpaid and grows with each new arch crate.

**Recommendation:** Create the promised hipfire_runtime::transformer (or a hipfire-arch-common crate) hosting the shared loaders as generics/params: e.g. load_norm_weight(hfq, gpu, name, n, add_one: bool, prefix) covering the qwen2 (raw) vs qwen35 (+1.0) delta, plus load_bias_f32, load_weight_tensor keyed by the full quant_type matrix, and smart_resize. Each arch crate then calls the shared helper instead of re-pasting the quant_type match. This is a natural home given all callers already depend on hipfire-runtime.

**Evidence:** qwen2.rs:524 load_norm_weight_raw / :573 load_bias_f32 / :603 load_weight_tensor vs dots_ocr.rs:526 / :568 / :608 (identical bodies, differing panic prefixes); qwen35.rs:2369 load_norm_weight_raw; smart_resize duplicated at dots-ocr image.rs:104 and qwen35-vl image.rs:26; all carry 'TODO(transformer-extraction)' markers (qwen2.rs:22-28, 510-523, 597-602).

*Verification: adjusted — Duplication confirmed: diff of load_norm_weight_raw (qwen2:524 vs dots-ocr:526) and load_bias_f32 (qwen2:573 vs dots-ocr:568) differs only in qwen2:/dots-ocr: panic prefixes (plus one extra comment block), same quant_type match and assert_eq!. Third copy at qwen35:2369; smart_resize forked (dots-ocr image.rs:104, qwen35-vl image.rs:26); all carry TODO(transformer-extraction). BUT the finding's key fact is wrong: hipfire_runtime::transformer DOES exist (transformer.rs, 195 LOC, lib.rs:53) — it is a batched-prefill composition seam; the TODO SUB-targets (transformer::norm/vision_weights/vision_linear/lm_head) don't exist and loaders aren't extracted. Severity Medium stands; recommendation should EXTEND the existing transformer module, not 'create' it.*

### [Low] coupling — `crates/hipfire-arch-qwen2/src/arch.rs:161-195; crates/hipfire-arch-llama/src/arch.rs:405-504`

**Observation:** The SimpleAr + ServingBackend impls for Qwen2Backend and LlamaBackend are near-identical boilerplate: new(), logits()/vocab_size(), caps()=ArchCaps::default(), eos_token(), serve()=run_simple_ar(...), reset_session() (rewind KV cursor), and unload() (free weights/scratch/kv). Only the arch-specific prefill/decode forward calls differ. Gemma3Backend repeats the shape again. This is low-severity because object-safety limits how much can be shared, but the serve/caps/unload trio is mechanically copyable and drift-prone.

**Recommendation:** Factor the invariant parts into a small blanket helper or a default-method-carrying supertrait: e.g. a DenseArBackend trait providing serve()/caps()/reset_session()/unload() in terms of associated free-fns, leaving each backend to implement only prefill/decode/logits. Keeps the hot path static while removing ~40 lines of copy-paste per arch.

**Evidence:** qwen2 arch.rs:174-194 serve/reset_session/unload vs llama arch.rs:482-503 serve/reset_session/unload — structurally identical bodies differing only in config field names and forward-fn calls.

*Verification: confirmed*

### [Low] module-structure — `crates/hipfire-arch-dots-ocr/src/dots_ocr.rs:20-39; crates/hipfire-arch-qwen2/src/qwen2.rs:1299-1313`

**Observation:** Grab-bag of stale/minor hygiene issues folded together. (1) dots_ocr.rs's module docstring (lines 20-31) still says 'Bring-up status (rev 0)': 'DotsOcrWeights::load vision-side currently a stub that returns an empty struct' and 'vision_forward — stub returning an error', but the vision side is fully implemented — DotsVisionWeights are loaded (dots_ocr.rs:500-513) and vision_forward is real (dots_ocr.rs:952). Misleading on a 1743-line file. (2) forward_prefill_batch_embeds has a copy-pasted duplicate assert_eq! (qwen2.rs:1299-1304 and again 1308-1313) with the local `let gemv/ctx` wedged between them. (3) hipfire-arch-llama has zero unit tests (0 #[test] markers) despite exposing a Backend and a bring-up adapter; its dead forward method is also never exercised. (4) gemma3's forward still uses direct weight_gemv while qwen2/llama were migrated to execute_steps/super-ops — an un-tracked modernization inconsistency across the family.

**Recommendation:** Refresh the dots_ocr.rs module doc to reflect the shipped vision tower; delete the duplicate assert at qwen2.rs:1308-1313; add at least a config/arch-id smoke test to arch-llama (mirroring qwen2 arch.rs:197-206); and either migrate gemma3's forward to execute_steps or note in AGENTS/docs why it stays on the direct path.

**Evidence:** dots_ocr.rs:27-31 'stub' text vs implemented vision_forward at :952 and DotsVisionWeights load at :500-513; qwen2.rs:1299-1304 and 1308-1313 are identical assert_eq! blocks; grep '#[test]' in crates/hipfire-arch-llama/src = 0; gemma3 forward.rs:263-265/353-356 use weight_gemv where qwen2/llama use execute_steps.

*Verification: confirmed*


## Cross-architecture duplication sweep (all 13 arch crates)

*Subsystem key: `cross-arch-dup` — 6 finding(s)*

**Subsystem assessment:** The 13 hipfire-arch-* crates share two structural seams: the static `Architecture` bring-up trait (config/weights/state) in hipfire-runtime/src/arch.rs L85 and the dyn-dispatched `ServingBackend`/`SimpleAr` runtime traits (same file, L333/L420). The bring-up triple is cleanly delegated, but the serving impls and — more seriously — the per-arch config structs, HFQ-metadata parsers, and low-level weight/quant helpers are copy-pasted with only cosmetic drift. The codebase itself documents this: 16 `TODO(transformer-extraction)` markers and a dozen "Mirrors …"/"Replicated from …" comments point at a shared home (`hipfire_runtime::transformer`) that was never populated for these helpers. The `Architecture` trait is also never used polymorphically (no `T: Architecture`/`dyn Architecture`), so it costs boilerplate without buying dispatch. Main risk: correctness drift — a fix to one copy of `sext4`/`dequant_hfq4`/config parsing silently diverges from 3+ others, and there are near-zero unit tests over these pure, GPU-free functions.

### [Medium] duplication — `crates/hipfire-arch-{llama,qwen2,gemma3,zaya,lfm2moe,nemotron,gemma3-vl}/src/arch.rs`

**Observation:** Seven arch crates each hand-write a full `impl ServingBackend` + `impl SimpleAr` whose non-forward bodies are near-identical: `caps()` returns `ArchCaps::default()` in 7/7, `serve()` is `let eos = <field>; run_simple_ar(gpu, self, tok, eos, ctx)` in 6/7 (gemma3 only differs by resolving `<end_of_turn>`), and `reset_session`/`unload`/`eos_token`/`vocab_size`/`logits` are one-line accessors. Three files even repeat the identical comment `// Single-session bring-up: rewind the KV cursor`. These 7 arch.rs files total 1790 lines, a large fraction of which is this boilerplate, not arch-specific logic.

**Recommendation:** Collapse the shared ServingBackend surface. Give `ServingBackend` default method bodies driven off a single required accessor, e.g. `fn as_simple_ar(&mut self) -> &mut dyn SimpleAr` plus `fn eos_token(&self) -> u32`, with a default `serve()` that calls `run_simple_ar`. Arches with a bespoke loop (VL splice, qwen35 DFlash) still override `serve`. Alternatively expose a `declare_dense_ar_backend!(Type { arch_id, eos: field })` macro. Either removes ~30 boilerplate lines per crate and centralizes the KV-reset/unload contract.

**Evidence:** grep found 7 `impl ServingBackend for` and 7 `impl SimpleAr for`; 6 identical `run_simple_ar(gpu, self, tok, eos, ctx)` call sites; 7 `ArchCaps::default()`; 3 identical 'Single-session bring-up' comments (llama:493, qwen2:185, gemma3:206).

*Verification: adjusted — Core duplication is real: 7 ServingBackend + 7 SimpleAr impls, 6 identical run_simple_ar(gpu,self,tok,eos,ctx) call sites, and LOC total is exactly 1790 as claimed. But two evidence details are off: the 'Single-session bring-up' comment is byte-identical only in qwen2:185/gemma3:206 — llama:493 reads 'rewind the KV write cursor.' (a variant), so 2 not 3; and gemma3-vl caps() is ArchCaps{vision:true,..default()}, not bare default. Severity overstated: this is ~210 boilerplate lines (~12%, and the two big files are forward-logic dominated), a one-line-delegation maintainability tax with no active correctness risk — Medium (scheduled refactor), not High. Recommendation (default trait bodies / macro) is pure-Rust and invariant-compatible.*

### [Medium] duplication — `crates/hipfire-arch-{qwen2/src/qwen2.rs:67,gemma3/src/config.rs:18,minimax/src/minimax.rs:25,lfm2moe/src/config.rs:30}`

**Observation:** At least 8 arch crates redeclare the same core transformer config fields (hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, head_dim, intermediate_size, vocab_size, rope_theta, rms_norm_eps, eos_token_id) and re-implement the same HFQ parser: `serde_json::from_str(metadata_json)` → `.get("config")` → `from_value::<RawConfig>` → `head_dim.unwrap_or(hidden_size/num_attention_heads)`. MiniMaxConfig::from_hfq (minimax.rs:97-127) and Lfm2MoeConfig::from_hfq (config.rs:142-149) are structurally identical down to the error strings. `q_dim()`/`kv_dim()` are byte-identical in minimax.rs:130-136 and lfm2moe/config.rs:229-235.

**Recommendation:** Introduce a shared `TransformerBase` config struct + `parse_config_wrapper(&HfqFile) -> Result<serde_json::Value>` helper (natural home: `hipfire_runtime::config` or `hipfire-model`). Arches compose it: `struct MiniMaxConfig { base: TransformerBase, num_local_experts: usize, ... }`, and inherit `q_dim`/`kv_dim` as `TransformerBase` methods. This keeps genuinely arch-specific fields (layer_types, MoE topology, conv_kernel_size) local while removing ~10 duplicated fields and the wrapper-parse boilerplate from every crate.

**Evidence:** grep confirmed identical field blocks in 4 config files; `from_hfq`/`config_from_hfq` defined 22 times across arch crates; `q_dim`/`kv_dim` byte-identical in two crates; 6 files repeat the `metadata_json missing 'config' wrapper` error string.

*Verification: adjusted — Verified: 4 config structs share the identical core field block in the same order; q_dim/kv_dim are byte-identical at minimax.rs:130-136 and lfm2moe/config.rs:229-235; the from_hfq wrapper-parse + error strings match. Evidence numbers are imprecise: the exact 'missing `config` wrapper' string is in only 3 files (minimax/deepseek4/lfm2moe), though the broader .get("config") wrapper-parse appears in 9 files (understated, not overstated); from_hfq/config_from_hfq count is 21, not 22. Severity: config/parser duplication is a scheduled-refactor antipattern with latent drift risk, not active correctness risk or blocked maintainability — Medium. Shared-config-struct recommendation is invariant-compatible.*

### [Medium] duplication — `crates/hipfire-arch-{gemma3/src/weights.rs:542,minimax/src/minimax.rs:213,lfm2moe/src/lfm2moe.rs:158,zaya/src/gpu.rs:42} and dots-ocr/src/dots_ocr.rs:761 vs qwen35-vl/src/qwen35_vl.rs:209`

**Observation:** Low-level dequant/read helpers are copy-pasted with the code itself flagging it. `fn sext4(nib)->i8` (4-bit sign extend) appears byte-identical in 4 crates. `fn dequant_hfq4` is byte-identical between dots-ocr:761 and qwen35-vl:209 (only the param name `n_elements` vs `n` differs). `fn read_tensor` is duplicated in minimax:143 and lfm2moe:26. The comments are explicit: minimax.rs:140 'Replicated from the qwen35 loader', minimax.rs:203 'Mirrors lfm2moe.rs (which mirrors qwen35)', and dots-ocr carries three 'Mirrors hipfire-arch-qwen35-vl::…' plus `TODO(transformer-extraction)` markers.

**Recommendation:** Extract the format-level primitives to a shared crate: `sext4`, `dequant_hfq4`, `read_tensor`, and the OQ4/OQ8 repack belong in `hipfire-runtime::quant` (or `hipfire-primitives`, which already hosts `f16_to_f32`); the `dequant_hfq4`/`linear_f16` vision pair belongs in the `hipfire_runtime::transformer::vision_linear` the TODOs already name. These are pure, arch-neutral byte math — ideal shared-crate candidates and trivially unit-testable.

**Evidence:** grep: `fn sext4` in 4 files (identical bodies verified); `fn dequant_hfq4` identical in 2 files; `fn read_tensor` in 2 files; 16 `TODO(transformer-extraction)` markers; explicit 'Replicated from'/'Mirrors qwen35' comments at minimax.rs:140,203.

*Verification: confirmed*

### [Medium] abstraction — `crates/hipfire-runtime/src/arch.rs:85-210 (trait Architecture)`

**Observation:** The `Architecture` trait is never consumed polymorphically: there is no `T: Architecture` bound, no `impl Architecture` argument, and no `dyn Architecture` anywhere in the workspace. It is only invoked via fully-qualified static syntax `<Concrete as Architecture>::config_from_hfq/load_weights/new_state` at compile-time-known call sites (examples/*.rs, hipfire-serving-core/src/load.rs:583). So the trait provides zero dispatch; its entire runtime payload is the 4 default override hooks (loop_guard/sampler/prompt_frame/eos_filter). Meanwhile it forces every arch to write ~15 lines of one-line delegation (config_from_hfq/load_weights/new_state each just forward to the crate's real fn).

**Recommendation:** Keep the trait thin and honest rather than growing it (correctly, forward stays off it). Either (a) treat it as a documented naming convention and eliminate the delegation boilerplate by generating the triple with a macro over the crate's real fns, or (b) since the only real dispatch trait is `ServingBackend`, add a `fn load(hfq, gpu) -> Result<Box<dyn ServingBackend>>` factory so bring-up flows straight into the dyn boundary the daemon actually holds. Do not add forward to the trait — the module docs' static-dispatch rationale is sound.

**Evidence:** grep for `T: Architecture`/`dyn Architecture`/`impl Architecture` (excluding `impl … for`) returned zero hits; all usages are `<Type as Architecture>::` in examples and serving-core/src/load.rs:583-584.

*Verification: confirmed*

### [Low] duplication — `crates/hipfire-arch-{minimax/src/arch.rs:27,deepseek4/src/arch.rs:532,gemma3/src/arch.rs:30}`

**Observation:** arch_id() hardcodes integer literals in several crates (minimax returns `10`, deepseek4 `9`, gemma3 `12`) even though hipfire-model/src/lib.rs:150-159 already centralizes the canonical constants (ARCH_ID_MINIMAX_M2=10, ARCH_ID_DEEPSEEK4_FLASH=9, ARCH_ID_GEMMA3_TEXT=12). Other crates do it right — nemotron uses `ARCH_ID_NEMOTRON_H`, zaya `ARCH_ID_ZAYA`, lfm2moe `ARCH_ID` — so the source of truth is duplicated inconsistently, inviting an ID collision when the next arch is added.

**Recommendation:** Reference the `hipfire_model::ARCH_ID_*` constants from every `arch_id()` and every hardcoded dispatch comparison. This is a mechanical change that makes hipfire-model the single registry and lets a reviewer see all IDs in one place.

**Evidence:** hipfire-model/src/lib.rs:150-159 defines the constants; minimax/deepseek4/gemma3 arch.rs return bare literals while nemotron/zaya/lfm2moe reference the constants (grep of arch_id bodies).

*Verification: confirmed*

### [Low] duplication — `crates/hipfire-arch-{qwen2:115,gemma3:88}/src/arch.rs (Backend structs + tests)`

**Observation:** Two further small copy-paste families round out the boilerplate story. (1) `Qwen2Backend`/`Gemma3Backend` (and effectively `Lfm2Backend`/`ZayaModel`) each define a `{config, weights, state}` bundle plus a trivial `pub fn new(config, weights, state) -> Self`. (2) `*_arch_id_and_name` unit tests are duplicated with identical shape in qwen2, dots-ocr, gemma3, and minimax — the only arches with any arch.rs test at all, meaning the shared serving/config logic (the code most worth testing) has effectively no direct unit coverage.

**Recommendation:** Fold the bundle into a generic `SimpleArBackend<A: Architecture>` holding `(A::Config, A::Weights, A::State)` once the Architecture-triple factory (finding 4) exists, deleting the per-arch structs+new(). Replace the four near-identical `arch_id_and_name` tests with one table-driven test, and add unit tests for the newly-shared config parser and `dequant_hfq4`/`sext4` primitives (all pure, no GPU needed) so consolidation is regression-guarded.

**Evidence:** grep: 2 explicit `pub fn new(config: …Config, weights: …Weights, state: …State)` constructors; 4 crates carry `*_arch_id_and_name` tests; only qwen2/dots-ocr/gemma3/minimax have any `#[cfg(test)]` block in arch.rs.

*Verification: adjusted — Confirmed: Qwen2Backend/Gemma3Backend {config,weights,state} bundles with trivial pub fn new at qwen2:115 and gemma3:88 (exactly 2 such constructors), and 4 near-identical *_arch_id_and_name tests (qwen2/dots-ocr/gemma3/minimax). But the observation's key claim is false: these are NOT 'the only arches with any arch.rs test.' arch.rs has #[cfg(test)] blocks in 7 crates; deepseek4 (deepseek4_arch_id_is_nine) and toy (toy_arch_id_is_reserved) add arch_id-style tests (so 6 crates, not 4), and gemma3-vl has 4 substantive splice/serving-logic tests (splice_expands_marker_in_place, etc.) — directly refuting 'the shared serving/config logic has effectively no direct unit coverage.' Duplication core (Low) stands; the test-coverage numbers and 'no coverage' characterization are the corrected parts.*


## hipfire-diffusion

*Subsystem key: `diffusion` — 7 finding(s)*

**Subsystem assessment:** crates/hipfire-diffusion implements a from-scratch CPU-reference + HIP-boundary diffusion pipeline (SD/SDXL UNet, VAE, CLIP, and a Krea2/QwenImage MMDiT transformer) plus HFQ metadata and diffusers/single-file import. The subsystem is functionally rich and, encouragingly, the past channels_last/stride loader bug is now pinned by two pure-CPU regression tests (tests.rs:6200 pytorch_contiguous_detection_matches_torch_semantics, tests.rs:6218 channels_last_storage_reorders_to_contiguous_oihw), so that specific correctness risk is closed. The main health problems are structural, not behavioral: lib.rs is a 10.3k-line grab-bag mixing metadata, CPU ops, the CLIP encoder, a 1.9k-line pipeline god-impl, and 1.9k lines of untrusted-format import (pickle VM + zip reader) that per AGENTS.md belongs in hipfire-coexistence. Pervasive mechanical duplication (a 213-copy error-map closure and ~445 hand-written CPU/GPU dispatch wrappers) and a single 13.5k-line test module make the crate hard to navigate and evolve. Kernel dispatch correctly reuses rdna-compute's Gpu abstraction rather than reinventing it; the duplication is in the per-op boilerplate wrapping those calls, not in the dispatch layer itself. None of the findings are active correctness bugs, but several are High on the maintainability-at-scale axis.

### [High] monolith — `crates/hipfire-diffusion/src/lib.rs:1-10349`

**Observation:** lib.rs is 10,349 LOC carrying at least six distinct responsibilities: HFQ metadata/config structs (L48-1137), primitive CPU tensor ops (linear/conv2d_nchw/group_norm_nchw/layer_norm/attention, L6612-7075), the full CLIP text encoder (ClipTextEncoder L6100-6600), a DiffusionPipeline god-impl of 27 methods spanning ~1,890 lines (L2788-4678), JSON/shape helper grab-bags (json_* L7264-7335, shape2/3/4 L7043-7073), and ~1,899 lines of import/conversion (L8444-10343). Sibling concerns (vae.rs, unet.rs, transformer.rs, scheduler.rs, layers.rs, tokenizer.rs) are already split out, so the split convention exists but was not applied to the largest concerns.

**Recommendation:** Extract along the seams already implied by the code: `clip.rs` (ClipTextEncoder + ClipEncoderLayer), `cpu_ops.rs` (the primitive nchw/linear/attention reference ops that layers.rs builds on), `metadata.rs` (the Diffusion*Metadata/Config structs + json_* helpers), and `pipeline.rs` (the DiffusionPipeline impl). Keep lib.rs as a thin re-export/facade like it already does for `mod vae; pub use vae::*;`. This is mechanical (move + `pub(crate)`), turns a 10k file into ~2k-line modules, and makes the CPU-reference vs orchestration boundary explicit.

**Evidence:** wc -l lib.rs = 10349; DiffusionPipeline impl L2788-4678 contains 27 fns; ClipTextEncoder L6100, primitive ops linear L6612 / conv2d_nchw L6857 / group_norm_nchw L6948; already-split modules declared at L5622-5680 (layers/unet/vae/hip_kernels/gpu_ops).

*Verification: confirmed*

### [High] crate-boundary — `crates/hipfire-diffusion/src/lib.rs:8444-10343`

**Observation:** ~1,899 lines of import/format-conversion tooling live in the same crate as the inference DiffusionPipeline: import_diffusers_to_hfq (L8452), single-file checkpoint import (L8680), safetensors state-dict parsing (L9649-9788), a hand-rolled PyTorch pickle interpreter (PickleValue enum + parse_pytorch_pickle_tensor_index L9939-10063 + reduce/rebuild helpers) and a from-scratch zip reader (MiniZipArchive L10228-10325). AGENTS.md states import/export/format-conversion/interop tooling belongs in hipfire-coexistence 'not folded into the daemon, server, or runtime hot path', and hipfire-coexistence already exists. This also embeds a security-sensitive parser of untrusted pickle/zip input into the crate that links the inference path (e.g. MiniZipArchive::open allocates vec![0u8; central_size as usize] and vec![0u8; uncompressed_size as usize] straight from attacker-controlled header fields, L10251/L10306).

**Recommendation:** Move the entire import block (diffusers/single-file mappers, safetensors + pytorch state-dict parsers, PickleValue VM, MiniZipArchive) into hipfire-coexistence (or a dedicated hipfire-diffusion-import tooling crate) and have hipfire-cli's diffusion import command call it there. hipfire-diffusion should retain only the runtime HFQ-consuming path. This satisfies the AGENTS.md coexistence invariant and keeps the pickle/zip attack surface out of the inference library. While relocating, bound the untrusted allocations (cap or checked reserve on central_size/uncompressed_size).

**Evidence:** import span L8444-10343 = 1899 lines; hand-rolled pickle VM at L9920 (enum PickleValue) / L9939 (parse_pytorch_pickle_tensor_index); MiniZipArchive at L10228; only external caller is hipfire-cli/src/commands/diffusion.rs:421; `ls crates/` shows hipfire-coexistence exists but does not reference hipfire_diffusion.

*Verification: confirmed*

### [Medium] duplication — `crates/hipfire-diffusion/src/gpu_ops.rs (197x), crates/hipfire-diffusion/src/lib.rs (16x)`

**Observation:** The exact closure `.map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))` appears 213 times (197 in gpu_ops.rs alone, 16 in lib.rs). Combined with the surrounding per-op boilerplate (40 KernargBlob::new, 45 bind_thread, 43 upload_f32, 24 device_synchronize across gpu_ops.rs's 61 functions), each *_hip_on_gpu function is a near-identical bind/upload/malloc/push-kernargs/launch/sync/copy/free sequence differing only in kernarg layout and grid math. This noise makes real per-op logic hard to see and makes it easy to omit a bounds check or a free in a new op.

**Recommendation:** Add a small extension trait to collapse the error mapping, e.g. `trait BackendResultExt<T> { fn backend(self) -> DiffusionResult<T>; }` implemented for `Result<T, E: Display>`, turning 213 closures into `.backend()?`. Then factor the launch skeleton into one helper that takes input tensors, an output-bytes computation, and a `FnMut(&mut KernargBlob)` kernarg builder + grid function, so each op body shrinks to its unique shape/kernarg logic. rdna-compute::Gpu is already the right dispatch abstraction — this is purely about the diffusion-side wrapper boilerplate.

**Evidence:** grep -c 'BackendUnavailable(error.to_string())' gpu_ops.rs = 197, lib.rs = 16; gpu_ops.rs has 61 fns with KernargBlob::new x40, bind_thread x45, upload_f32 x43; representative full skeleton at gpu_ops.rs:29-94 (rgb_tensor_to_u8_hip_on_gpu).

*Verification: confirmed*

### [Medium] coupling — `crates/hipfire-diffusion/src/lib.rs:2101-2520 (and ~445 sites crate-wide)`

**Observation:** Every CPU op has a hand-written `_with_runtime_context` twin whose body is the identical 5-line CPU-or-GPU switch: `let Some(_device_id) = runtime_context.rocm_device_id() else { return CPU_OP(...) }; runtime_context.with_rocm_gpu(|gpu| GPU_OP(gpu, ...))` (see scale_tensor L2101, silu L2169, tensor_add L2252, conv2d L2291, etc.). The `_with_runtime_context` token occurs 445 times across the crate (lib 105, transformer 121, vae 50, unet 44, layers 63) and the `runtime_context: &mut DiffusionGenerationRuntimeContext` god-parameter is threaded through 121 signatures. The bound `_device_id` is even unused. This is high coupling (one mutable context passed everywhere) plus mechanical duplication of the dispatch decision, and it is exactly the surface where CPU/GPU op-parity drift can hide.

**Recommendation:** Replace the per-op twins with a single dispatch helper or macro, e.g. `dispatch!(ctx, |gpu, cache| gpu_op(...), || cpu_op(...))` that centralizes the rocm_device_id check and with_rocm_gpu(_weighted) call, so adding an op no longer means copying the switch. Alternatively define a `DiffusionOps` trait with `Cpu` and `RocmHybrid` implementors so the backend is chosen once at the pipeline boundary rather than re-decided at every leaf op; this also gives a natural seam for CPU/GPU parity tests. Keeps the runtime-choice (not cargo-feature) design intact.

**Evidence:** identical switch bodies at lib.rs:2101-2119, 2169-2179, 2252-2263, 2291-2300; DiffusionGenerationRuntimeContext L415-484 with with_rocm_gpu/with_rocm_gpu_weighted; grep counts: _with_runtime_context = 445 occurrences, `runtime_context: &mut DiffusionGenerationRuntimeContext` param in 121 sites.

*Verification: confirmed*

### [Medium] test-structure — `crates/hipfire-diffusion/src/tests.rs:1-13491`

**Observation:** The entire test suite is one 13,491-line `#[cfg(test)] mod tests` (included at lib.rs:10348) with 217 #[test]s. Roughly 55 are GPU tests that use a runtime detect-and-skip pattern (55 `ROCm GPU unavailable` eprintln skips, 36 rocm_hybrid usages) rather than #[ignore] or a cfg gate, so they silently pass as skips in no-GPU CI; the remaining ~162 are pure-CPU logic tests (metadata parse, config validation, shape/overflow math, resize, stride/contiguous). transformer.rs already keeps its own inline `#[cfg(test)]` unit tests (repeat_kv_heads_* at L3484+), so the monolith is inconsistent with the crate's own convention. A single 13.5k file is slow to compile per-edit and hides which behaviors are actually covered.

**Recommendation:** Split by test type: co-locate the pure-CPU unit tests as `#[cfg(test)] mod tests` blocks in the module that owns the code (metadata tests with metadata, stride/pickle tests with the loader once it moves to coexistence, conv/attention parity math with cpu_ops), following transformer.rs. Promote the GPU parity/generation tests to hipfire-eval batteries or a `tests/` integration dir per AGENTS.md ('model/runtime admission evidence belongs in hipfire-eval'), and mark GPU-required cases `#[ignore]` so no-GPU CI reports them as ignored, not passed. The two stride regression tests (L6200, L6218) are pure and should move next to the reorder logic.

**Evidence:** wc -l tests.rs = 13491; single `#[cfg(test)] mod tests;` at lib.rs:10348; grep -c '#[test]' = 217; 'ROCm GPU unavailable' x55; rocm_hybrid x36; #[ignore] x5; contrasting inline unit tests at transformer.rs:3484.

*Verification: confirmed*

### [Medium] abstraction — `crates/hipfire-diffusion/src/lib.rs:3888-3928, 4026-4066`

**Observation:** The public generation API is a combinatorial method explosion: generate_batch, generate_batch_with_progress, generate_batch_with_runtime_options, generate_batch_with_progress_and_runtime_options, and private generate_batch_inner — then the identical fivefold set again for img2img (generate_img2img_batch* L4026-4066). That is 8 public entry points plus 2 inner delegators encoding the cartesian product of two optional axes (progress callback, runtime options). Adding a third optional axis would double the surface again.

**Recommendation:** Collapse to one entry point per operation taking an options struct with sensible defaults, e.g. `pub fn generate_batch(&self, req: &DiffusionBatchRequest, opts: DiffusionGenerateOptions)` where `DiffusionGenerateOptions { progress: Option<&mut dyn FnMut(DiffusionProgress)>, runtime: DiffusionGenerationRuntimeOptions }` (or a small builder). The four current variants become defaulted-field call sites, and img2img shares the same options type. This removes 6+ of the 10 methods and makes the optional axes composable instead of multiplicative.

**Evidence:** generate_batch family at lib.rs:3888/3895/3907/3915/3924 (inner); img2img family at lib.rs:4026/4037/4049/4057/4066; DiffusionGenerationRuntimeOptions and DiffusionProgress already exist as the two axes (L192, L669).

*Verification: confirmed*

### [Low] utils-sprawl — `crates/hipfire-diffusion/src/hip_kernels.rs:5, lib.rs:10194-10201, gpu_ops.rs:63`

**Observation:** Assorted hygiene items folded together: (1) hip_kernels.rs's module doc claims the kernel constants are 'Compiled only under the `rocm` feature', but the crate has no [features] table and no cfg(feature="rocm") anywhere in src/, and gpu_ops.rs:4 explicitly documents the opposite ('GPU code always compiles ... not a cargo feature') — directly contradictory docs. (2) read_bytes uses unchecked `*pos + len > data.len()` (lib.rs:10195) on untrusted pickle input while the rest of the loader consistently uses checked_add (e.g. L9860, L9693, L9752). (3) gpu_ops.rs kernarg builders truncate with `total_pixels as i32` / `height as i32` (L63-65) with no guard; fine for current image sizes but a silent-wrap footgun for large latents. Individually minor.

**Recommendation:** (1) Fix the hip_kernels.rs doc to match reality (always-compiled, runtime-selected) or introduce a real feature and gate consistently; do not leave contradictory module docs. (2) Make read_bytes use `pos.checked_add(len)` to match the loader's own convention. (3) Use `u32::try_from(total_pixels)` (or i32) with an InvalidMetadata error instead of `as` in the kernarg builders. All three are small, isolated edits.

**Evidence:** hip_kernels.rs:5 'under the `rocm` feature' vs gpu_ops.rs:4 'not a cargo feature'; Cargo.toml has no [features] table and grep 'cfg(feature' src/ = 0 hits; read_bytes unchecked add at lib.rs:10195 vs checked_add at lib.rs:9860; `as i32` kernarg casts at gpu_ops.rs:63-65.

*Verification: confirmed*


## Quantization: hipfire-quantize, kvquant, quant-format, kld, lora-hfq

*Subsystem key: `quant` — 10 finding(s)*

**Subsystem assessment:** The quant subsystem centers on the offline hipfire-quantize crate: a lib (codecs.rs, gptq.rs, ldlq.rs, qtip.rs, hessian_io.rs, gguf_input.rs, fixture.rs) plus a 13,875-LOC binary whose main.rs is dominated by a single ~5,300-line main() doing CLI parsing, format-recipe resolution, and pipeline orchestration inline. A real decomposition effort is underway (codecs.rs is a "pure codecs" module with a byte-stable golden battery, and QuantType byte identity was homed in the leaf hipfire-quant-format crate), but it is half-finished: quant math is still split between main.rs and codecs.rs, the "pure" codecs read a process-global toggle, and on-disk block geometry is re-hardcoded across at least three crates plus runtime examples rather than owned by the format crate. Calibration machinery (gptq 24 tests, qtip 11, hessian_io 7) is reasonably tested, but the pure codecs have zero direct tests and the GGUF binary parser has none. The main risks are (1) format-change fragility from duplicated block layout and copy-pasted decoders, and (2) an unmaintainable god-function. Numeric edge handling inside the codecs is actually fairly defensive (range/counts/1e-12 guards), so correctness risk is concentrated in the duplication rather than the math.

### [High] monolith — `crates/hipfire-quantize/src/main.rs:5444-10743 (fn main)`

**Observation:** main() is a single ~5,300-line function. It handles --help/--emit-fixture/--ldlq-probe early exits, rayon pool setup, hand-rolled flag parsing (arg_value/args.iter().position), a large format-flag normalization maze (mq_plus, oq4_recipe, oq8_recipe, is_legacy_opus_plus at L5528-5560), calibration sidecar loading, MoE K-map promotion, and the per-tensor quantize pipeline, all inline. None of it is reachable by a unit test; all 81 tests exercise helper fns instead.

**Recommendation:** Extract three testable stages: a CliArgs struct parsed from argv (Clap or a hand-rolled parse_args -> Result), a pure QuantPlan/recipe resolver that turns flags+sidecars into a resolved config (the L5528-5560 string logic is pure and belongs in a unit-tested function), and a thin pipeline driver. main() should shrink to arg-parse, plan, run. This makes the recipe maze (mq+/oq4+/oq4++/legacy opplus) directly testable.

**Evidence:** fn main at main.rs:5444; brace-matched close at line 10743 (~5,299 lines); format-recipe branching at L5528-5560; 81 #[test] all in mod tests, zero cover main()

*Verification: confirmed*

### [High] crate-boundary — `crates/hipfire-quant-format/src/lib.rs (193 LOC) vs codecs.rs / arch loaders`

**Observation:** hipfire-quant-format owns only the 1-byte QuantType identity and code<->from_code, not the on-disk block geometry. That geometry is re-hardcoded everywhere: MQ4 block=136 appears in 10 sites (codecs.rs:345,398,917,1053; gptq.rs:590; bin/dflash_convert.rs:180; bin/mtp_extract.rs:127; bin/draft_to_mq4.rs:24; runtime examples), and OQ4 block=130 appears in codecs.rs (651,685,834), ldlq.rs (287,646,702), AND per-arch runtime loaders arch-qwen35/qwen35.rs:2465,2873 and arch-minimax/minimax.rs:230. A single header-size change to one format touches ~10 files across 3 crates with no shared constant to catch a miss.

**Recommendation:** Promote block geometry into hipfire-quant-format: a const table or `impl QuantType { const fn block_bytes(self)->usize; const fn group_size(self)->usize }`, plus a canonical reference dequant. Writer (codecs), calibration (ldlq/gptq), and every arch loader then derive the layout from one authority, exactly as the crate already did for the byte id (its own doc notes readers used to 'drift'). This closes the same drift risk for block layout.

**Evidence:** quant-format lib.rs only defines QuantType+code/from_code; grep '136' -> 10 hardcoded sites; grep '130' block_bytes -> codecs/ldlq + arch-qwen35 + arch-minimax

*Verification: confirmed*

### [Medium] duplication — `crates/hipfire-quantize/src/codecs.rs:915 vs crates/hipfire-runtime/examples/quant_quality_mse.rs:207 (only dequant_mq4g256 is an existing pub API; dequant_mq6g256 has no library home; runtime lacks a dependency edge on hipfire-quantize)`

**Observation:** dequant_mq4g256 is byte-for-byte copy-pasted: both use block=136, identical scale/min extraction, identical nibble unpack, and the same signs-swapped inverse FWHT. dequant_mq6g256 (quant_quality_mse.rs:236), dequant_hfp4g32_row (:289), and dequant_mq4g256 in rq_real_gemv_check.rs:208 are further verbatim copies. codecs.rs already exports these decoders as pub, so the examples are duplicating an available library API.

**Recommendation:** Delete the example-local decoders and call hipfire_quantize::codecs::{dequant_mq4g256,dequant_mq6g256,dequant_hfp4g32_row}. Copy-pasted decoders silently diverge from the encoder when the layout changes (see the block=136/130 spread), defeating the parity these examples exist to check.

**Evidence:** codecs.rs:915-942 and quant_quality_mse.rs:207-234 are identical modulo one comment; both hardcode block=136 and cpu_fwht_256(&mut,signs2,signs1)

*Verification: adjusted — The mq4 duplication is real and byte-identical (codecs.rs:915-940 vs quant_quality_mse.rs:207-234, one comment differs); mq6 (:236), hfp4 (:289), and rq_real_gemv_check.rs:208 copies exist. But the claim 'codecs.rs already exports these decoders as pub' is only true for dequant_mq4g256; dequant_mq6g256 does not exist anywhere in hipfire-quantize, and pub dequant_hfp4g32_row (codecs.rs:1299) has a different signature. Also hipfire-runtime has no dependency on hipfire-quantize, so fixing this requires adding a dev-dependency plus exporting a new mq6 decoder — Medium severity stands, recommendation needs those corrections.*

### [Medium] coupling — `crates/hipfire-quantize/src/codecs.rs:341,948,1006,1606,1663 + lib.rs:42-52`

**Observation:** codecs.rs advertises 'no I/O, globals, or arch awareness' (module doc, L5-8), but five MQ codecs branch on crate::mq_clipsearch_enabled(), a lib-level OnceLock<bool> (lib.rs:42) armed from the CLI via set_mq_clipsearch(true) at main.rs:5557. The codecs' byte output therefore depends on hidden process state; the golden battery only passes because the global defaults to false in the test process.

**Recommendation:** Make the option explicit: pass `clipsearch: bool` (or a small `MqOpts { clipsearch, awq_alpha }`) into quantize_mq*g256 and delete the OnceLock. This restores true purity, makes both variants directly unit-testable without mutating a global, and removes an ordering hazard (first set() wins) between the CLI and any library consumer such as hipfire-diffusion.

**Evidence:** codecs.rs:341 `if crate::mq_clipsearch_enabled()`; lib.rs:42 `static MQ_CLIPSEARCH: OnceLock<bool>`; main.rs:5557 set_mq_clipsearch(true)

*Verification: confirmed*

### [Medium] module-structure — `crates/hipfire-quantize/src/main.rs:1682,1826,2015,478,542,601,3269`

**Observation:** The codecs.rs decomposition is incomplete: the mq2-lloyd family is split across two files. Plain quantize_mq2g256_lloyd lives in codecs.rs:1999, but the weighted (main.rs:1682), GPTQ (main.rs:1826), and decode (dequantize_mq2g256_lloyd_to_f32 main.rs:2015) variants, plus the FP8 decoders (478/542/601) and quantize_hfq_source_tensor (3269), remain in the binary. codecs.rs's own doc admits 'They will move here in a later batch.'

**Recommendation:** Finish moving the pure Lloyd/weighted/GPTQ codec bodies and FP8 decoders into codecs.rs (or a codecs::lloyd submodule) so all weight<->bytes math lives in the lib behind the golden battery, and main.rs keeps only orchestration. Splitting one codec family across a lib module and a 14k-line binary makes it easy to edit one half and forget the other.

**Evidence:** grep '^fn (quantize|dequant)' main.rs -> 1682/1826/2015/3269 + fp8 478/542/601; codecs.rs module doc line 6-11 'move here in a later batch'

*Verification: confirmed*

### [Medium] missing-tests — `crates/hipfire-quantize/src/codecs.rs (0 #[test]); affine_clipsearch:454, symmetric_clipsearch:483, quantize_mq2g256_lloyd:1999`

**Observation:** codecs.rs has zero direct tests; all 81 tests are in main.rs and are mostly golden byte-stability over fixed pseudo-random input plus a couple of round-trips. The pure, GPU-free branches most likely to misbehave are untested at the unit level: clip-search scale flooring (scale.max(1e-12) at :464/:488), the Lloyd degenerate guard (if range > 0.0 at :2040), the empty-centroid guard (if counts[k] > 0 at :2093), and affine_clipsearch on an all-equal group (half==0). No test feeds an empty, single-element, all-equal, or NaN/Inf group directly to these functions.

**Recommendation:** Add a #[cfg(test)] mod in codecs.rs with targeted edge tests: empty slice, one element, all-equal (range==0), all-zero, and NaN/Inf inputs for affine_clipsearch/symmetric_clipsearch and the lloyd encoders, asserting finite outputs and stable byte length. These run without a GPU and lock the exact guard branches the golden battery skips.

**Evidence:** grep -c '#[test]' codecs.rs = 0; guards at codecs.rs:464,488,2040,2093; tests concentrated in main.rs mod tests (golden GOLDENS table at main.rs:13712)

*Verification: confirmed*

### [Medium] missing-tests — `crates/hipfire-quantize/src/gguf_input.rs (522 LOC, 0 #[test])`

**Observation:** gguf_input.rs parses untrusted GGUF binaries with no tests. It reads tensor_count/kv_count via read_u64 as usize (L176-177) then indexes tensor payloads with unchecked slices — dequant_q6_k reads data[off+208]/data[off+209] (L452), dequant_q4_k/q5_k similarly — and panics on an unknown dtype (panic! at L517). A truncated or malformed GGUF panics rather than returning an error.

**Recommendation:** Add round-trip and malformed-input tests (short buffer, bad magic, unknown dtype, zero-dim tensor), convert the L517 panic to a Result error, and bounds-check block offsets before indexing (chunks_exact or explicit len checks). This is offline tooling so it is not a hot-path safety issue, but a 522-line binary parser with zero coverage is fragile and easy to regress.

**Evidence:** wc -l gguf_input.rs = 522; grep -c '#[test]' = 0; unchecked index dequant_q6_k data[off+208] at :452; panic! at :517; read_u64 as usize at :176-177,201,242

*Verification: confirmed*

### [Medium] duplication — `crates/hipfire-quantize/src/gguf_input.rs:286-490 vs crates/hipfire-runtime/src/quant.rs:14-360`

**Observation:** The GGML block decoders exist twice: gguf_input.rs defines dequant_q4_0/q8_0/q4_k/q5_k/q6_k, and hipfire-runtime/src/quant.rs defines dequantize_q4_0/q8_0/q4_k/q6_k with the same block constants and layout math. Two independent implementations of the same external codec can drift, and the runtime copy sits in the inference crate while AGENTS.md routes import/interop formats to offline tooling. runtime/quant.rs also has 0 tests.

**Recommendation:** Pick one home for GGML decode (the offline side, or a small shared leaf crate) and have both consumers depend on it; if the runtime copies are only used by the offline q4k->q4f16 conversion path (convert_q4k_to_q4f16_g64 at quant.rs:144) they may be dead in the hot path and removable. Either way, one tested implementation of the external format.

**Evidence:** gguf_input.rs dequant_q8_0:312/dequant_q4_k:333/dequant_q6_k:439 vs runtime quant.rs dequantize_q8_0:46/dequantize_q4_k:82/dequantize_q6_k:278; runtime quant.rs #[test] count = 0

*Verification: confirmed*

### [Low] duplication — `crates/hipfire-arch-nemotron/src/loader.rs:151, crates/hipfire-arch-zaya/src/gpu.rs:313, arch-qwen35-vl/src/qwen35_vl.rs:209, arch-dots-ocr/src/dots_ocr.rs:761 (dtype_for_quant_type gap is OQ/OqPlus/PARO/BF16/F32 at quant.rs:382; HFP4G32 is mapped at quant.rs:377)`

**Observation:** Per-arch loaders re-implement quant-byte dispatch and small decoders instead of a shared path. nemotron and zaya each define their own dequant_qt(qt, bytes), and dequant_hfq4 is re-defined in qwen35-vl (:209) and dots-ocr (:761). The centralized runtime dtype_for_quant_type (quant.rs:360) uses QuantType::from_code correctly but returns None for the OQ/HFP4 family, which is what forces the per-arch crates to hand-roll decode.

**Recommendation:** Extend the centralized mapping/decoder to cover the OQ/OqPlus/HFP4 variants (or expose a shared cpu-dequant helper keyed by QuantType) so arch crates dispatch through one function rather than copying dequant_hfq4/dequant_qt. Fewer copies to update when a layout or a new qt id lands.

**Evidence:** dequant_qt duplicated at nemotron/loader.rs:151 and zaya/gpu.rs:313; dequant_hfq4 duplicated at qwen35_vl.rs:209 and dots_ocr.rs:761; dtype_for_quant_type returns None for OQ/HFP4 (quant.rs:378-382)

*Verification: adjusted — All four duplication sites verified at the exact lines (dequant_qt x2, dequant_hfq4 x2, dots_ocr even carries a TODO to extract). But the evidence 'dtype_for_quant_type returns None for OQ/HFP4' is half-wrong: HFP4G32 IS mapped at quant.rs:377 (conditional on k%256==0); only OQ4/OQ8/OqPlus/PARO/BF16/F32 fall through to None at :382, and the duplicated dequant_hfq4 copies decode HFQ4 (qt 6/7), which is already mapped — so the causal story only holds for the OQ/plain-precision family. Core duplication claim stands at Low.*

### [Low] utils-sprawl — `crates/hipfire-quantize/src/bin/{dflash_convert.rs,draft_to_mq4.rs,mq4_merge_mtp.rs,mtp_extract.rs} + examples/`

**Observation:** The quantize crate bundles four extra binaries and five examples that each re-hardcode codec internals (block_bytes=136 in dflash_convert.rs:180, mtp_extract.rs:127, draft_to_mq4.rs:24) and call cpu_fwht_256 directly rather than going through codecs.rs. dflash_convert (687 LOC) and mtp_extract (1298 LOC) are format-extraction/conversion tools, which are closer to coexistence/offline-tooling concerns than to the core quantizer.

**Recommendation:** Have these bins consume codecs.rs helpers instead of re-deriving block layout, and consider relocating the conversion/extraction bins (dflash_convert, mtp_extract) toward the coexistence/tooling boundary so the quantize crate stays focused on weight quantization. This shrinks the surface that a format change must chase.

**Evidence:** src/bin has 4 bins; block_bytes=136 hardcoded in dflash_convert.rs:180 / mtp_extract.rs:127 / draft_to_mq4.rs:24; mtp_extract.rs = 1298 LOC, dflash_convert.rs = 687 LOC

*Verification: confirmed*


## Serving: serving-core, server, daemon, protocol, adapter, operator

*Subsystem key: `serving` — 12 finding(s)*

**Subsystem assessment:** The serving subsystem spans the daemon (JSONL engine), the axum HTTP server (OpenAI + SD API), the serving-core library (load/generate/session/model), and the typed protocol/adapter/admin-types crates. It is functionally rich but structurally strained: the load and generate paths are giant per-arch dispatch functions, LoadedModel is a 57-field Option-soup god-struct, and session behavior is expressed as ~13 duplicated qwen35_/lfm2_ free-function pairs mutating that struct (the source of the model<->session module cycle). The biggest risks are maintainability-at-scale (main() is ~3900 lines; sdapi.rs is one 8249-line file) and a protocol-drift correctness risk: the wire protocol is typed once in daemon-protocol but the daemon server hand-parses serde_json::Value and never depends on that crate. A ServingBackend/SessionServingBackend abstraction already exists to collapse the per-arch duplication but is only half-adopted. Test coverage is inverted — the hot path (generate/session/model) has zero in-file tests while their tests live inline in the daemon bin.

### [High] monolith — `crates/hipfire-daemon/src/main.rs:2497-6434`

**Observation:** main() is a single ~3937-line function. Its body is one for-loop over stdin lines wrapping a stringly-typed `match msg_type` (~2650-4737+) with 12 arms, several of which are enormous inline handlers: the "load" arm is ~712 lines and the "generate" arm is ~655 lines. No handler is extracted into its own function, and the crate is bin-only (no lib.rs) so none of this logic is reachable or testable except through the process entrypoint.

**Recommendation:** Split into hipfire-daemon (thin bin) + a library module tree. Each message type should become `fn handle_load(state, msg) -> DaemonResponse` etc., dispatched from a small `match`. This makes handlers unit-testable and lets the load/generate arms be reviewed independently. Combine with the typed-protocol adoption (below) so arms match on enum variants instead of string literals.

**Evidence:** fn main at L2497, file ends L6434 (one function); grep of top-level match arms shows "load" at rel-L23 and "generate" at rel-L735 within the 2640-4760 window; zero `fn`/`async fn` defined inside main()

*Verification: confirmed*

### [High] crate-boundary — `crates/hipfire-daemon-protocol/src/lib.rs:206-256 vs crates/hipfire-daemon/src/main.rs:88-152`

**Observation:** The wire protocol is defined twice. daemon-protocol has fully typed `DaemonRequest`/`DaemonResponse` enums (Load/Unload/Generate/Reset/...) with serde derive, and the adapter/eval/steer-harness use them. But hipfire-daemon does NOT depend on hipfire-daemon-protocol at all; the daemon server hand-parses each message with json_u64/json_string/json_opt_bool helpers, matches on string literals ("load", "generate", ...), and writes responses via raw `writeln!(stdout, r#"{...}"#)`. Client half is typed; server half is stringly-typed, so a field rename or variant change on one side is invisible to the compiler on the other.

**Recommendation:** Make the daemon depend on hipfire-daemon-protocol and deserialize into `DaemonRequest`, serialize `DaemonResponse`. The match then becomes exhaustive over enum variants (compiler-checked), and request/response shapes have exactly one definition shared by both ends. This directly de-risks protocol drift between the daemon and its adapter/clients.

**Evidence:** `grep hipfire-daemon-protocol crates/*/Cargo.toml` lists adapter/eval/steer-harness but not hipfire-daemon; `grep DaemonRequest crates/hipfire-daemon/src/main.rs` returns nothing; adapter serializes DaemonRequest (adapter/lib.rs:126,163) while daemon parses Value (main.rs:92-117)

*Verification: confirmed*

### [High] test-structure — `crates/hipfire-daemon/src/main.rs:331-2495`

**Observation:** The `generate_batch_prefill_tests` module (63 test fns, ~2164 lines) lives inline in the daemon bin crate but exercises functions owned by OTHER crates: hipfire-generate (validate_generate_batch_prefill, build_qwen35_fused_dense_prefill_batch_contract, plan_generate_batch_prefill_qwen35, ...), hipfire-state (parse_*/*_json), and hipfire-model. Meanwhile the serving hot path itself — generate.rs, generate_arch.rs, session.rs, model.rs — has zero in-file tests. Coverage is inverted: the tests are far from the code they test, and the code most worth testing has none.

**Recommendation:** Relocate each test to the crate that owns the function under test (hipfire-generate / hipfire-state / hipfire-model tests modules). Then add unit tests to serving-core for the GPU-free logic that currently has none: arch_id dispatch selection, ThinkMode::from_str, prompt-framing selection, stop-sequence handling, and session bookkeeping math (logical position, prefilled-suffix length). None of those require a GPU.

**Evidence:** test module spans L331-2495 with 63 `#[test]`/`#[tokio::test]`; its `use` list imports validators from hipfire_generate/hipfire_state/hipfire_model; per-file test counts: generate.rs=0, generate_arch.rs=0, session.rs=0, model.rs=0

*Verification: confirmed*

### [High] coupling — `crates/hipfire-serving-core/src/model.rs:242-448 (+ session.rs cycle)`

**Observation:** LoadedModel is a god-struct: 57 pub fields, 41 of them Option<>, covering ~10 arch families as parallel Option slots (the comment itself calls it 'Option-soup'). It drives the model.rs<->session.rs module cycle: model.rs imports four session types (SessionRegistry, Qwen35/Lfm2RequestSessionState, SessionCursor) at L40-43, while session.rs imports LoadedModel/ResidentSession at L48 and 24 of its function signatures take `&mut LoadedModel`. The cycle compiles (intra-crate modules), but it reflects a data/behavior split where LoadedModel owns the session types yet all session behavior is external free functions mutating it.

**Recommendation:** Collapse the Option-soup behind the already-drafted ServingBackend seam so LoadedModel holds one `Box<dyn ServingBackend>` (or an arch enum) instead of ~40 arch-specific Options. Move the session-op free functions into methods on the session backend (or a SessionServingBackend impl) so behavior lives with data; this breaks the module cycle by inverting the dependency (session types own their operations, model.rs just holds a trait object).

**Evidence:** model.rs L242-448: 57 `pub` fields / 41 `Option<`; `use crate::session::{...}` L40-43; session.rs `use crate::model::{LoadedModel, ResidentSession}` L48; 24 signatures with `&mut LoadedModel`

*Verification: confirmed*

### [High] duplication — `crates/hipfire-serving-core/src/session.rs (qwen35_*/lfm2_* pairs)`

**Observation:** Session operations are hand-duplicated per arch as ~13 qwen35_*/lfm2_* function pairs: allocate_session_state, activate_session, save_active_session, reset_active_session, fork_session_state, checkpoint_session_state, release_sessions, validate_prefix_hash, active_logical_position, session_resident, request_session_count, session_page_descriptors, state_page_descriptors. The daemon then selects between them with `if is_qwen35 {} else if is_lfm2 {}` ladders. A `SessionServingBackend` trait was explicitly drafted (hipfire-runtime/src/arch.rs, doc references docs/plans/2026-06-29-session-serving-backend.md) to unify exactly these, but it is not yet the dispatch surface.

**Recommendation:** Finish the SessionServingBackend migration: move the session-op set onto the trait and have both qwen35 and lfm2 backends implement it, so the daemon dispatches one `&mut dyn SessionServingBackend`. This removes the 13 duplicated pairs, the arch ladders, and (per the finding above) the model<->session cycle in one refactor.

**Evidence:** session.rs: 13 op names appear as both qwen35_ and lfm2_ variants (uniq -c of `pub fn (qwen35|lfm2)_*` shows 2-3 each); 13 `pub fn qwen35_` + 16 `pub fn lfm2_`; hipfire-runtime/src/arch.rs defines `trait SessionServingBackend` whose doc says it replaces 'the duplicated per-arch qwen35_/lfm2_ free functions over &mut LoadedModel'

*Verification: confirmed*

### [Medium] abstraction — `crates/hipfire-serving-core/src/generate.rs:1848-3678`

**Observation:** generate() is a ~1830-line function whose top is an `if m.arch_id == N { generate_N(...) }` ladder routing to ~9 bespoke per-arch free functions in generate_arch.rs (generate_llama/zaya/nemotron/qwen2/gemma3/gemma3_vl_text/deepseek4/minimax/lfm2moe). This bypasses the object-safe `ServingBackend::serve()` trait that already exists in hipfire-runtime and is already implemented by LlamaBackend/Qwen2Backend/Gemma3Backend/Lfm2Backend — so the abstraction is defined but the dispatch still hard-codes each arch.

**Recommendation:** Route dispatch through the trait: `m.backend.serve(gpu, tok, &mut ctx)` instead of the arch_id ladder, keeping only genuinely-special paths (dflash/mtp/pp) as explicit branches. The per-arch generate_* wrappers that merely frame the prompt and call `.serve()` can then be deleted.

**Evidence:** generate.rs L1848 fn generate; dispatch ladder at rel-L44..309 (arch_id 0/1,16,14/15,7,12,13,9,10,11 each calling generate_*); ServingBackend trait at hipfire-runtime/src/arch.rs:420 with 4 arch impls found via grep

*Verification: adjusted — fn generate at L1848 actually runs to EOF L3678 (1830 lines, exactly matching the observation's "~1830-line function"); the cited end L2360 is wrong (only 512 lines). arch_id ladder confirmed at L1891-2156 routing to generate_llama/zaya/nemotron/qwen2/gemma3/gemma3_vl_text/deepseek4/minimax/lfm2moe. ServingBackend trait at arch.rs:420; 7 impls exist (finding said 4 — an undercount). Core claim real, location end-line corrected.*

### [Medium] duplication — `crates/hipfire-serving-core/src/generate_arch.rs (per-arch fns) + generate.rs`

**Observation:** Prompt-framing logic is copy-pasted across the per-arch generate functions: the block that builds a JinjaChatFrame, synthesizes a `Vec<prompt_frame::Message>` from system+user (or uses messages_history), calls render_messages with tools, and falls back to ChatFrame::Plain on error appears ~6-7 times in generate_arch.rs and ~8 more times in generate.rs. Each copy is ~40-60 lines and identical up to the backend variable name.

**Recommendation:** Extract a single `build_prompt_tokens(m, prompt, system_prompt, tools, messages_history, think_mode) -> Vec<u32>` helper (or a method on a PromptFramer) and call it from every generate_* path. This removes hundreds of duplicated lines and centralizes the jinja-vs-Plain decision that currently drifts per arch.

**Evidence:** generate_arch.rs: `synthesized: Vec<prompt_frame::Message>` block ×6, try_jinja ×12, JinjaChatFrame ×7, 'Plain fallback' ×3; generate.rs: try_jinja ×8; visible near-identical block at generate_arch.rs:1550-1600 (generate_llama)

*Verification: confirmed*

### [Medium] abstraction — `crates/hipfire-serving-core/src/generate_arch.rs:1101,1172,1345,1518,1709 (signatures)`

**Observation:** Every generate_* function threads sampling parameters as loose positional scalars — temp: f32, top_p: f32, max_tokens: usize, repeat_penalty: f32, repeat_window: usize (12-16 positional args per function). A typed GenerationSamplingPolicy struct (temperature/top_p/repeat_penalty/max_tokens) already exists in hipfire-generate and is used by the typed protocol, but the serving generate paths reference it zero times. Adjacent same-typed floats (temp, top_p, repeat_penalty) are easy to transpose at a call site with no compiler protection.

**Recommendation:** Pass a single `GenerationSamplingPolicy` (or a serving-local SamplingParams) struct instead of 5+ positional scalars. This shrinks the signatures, kills a whole class of arg-transposition bugs, and lets the daemon build the policy once from the parsed request and forward it unchanged.

**Evidence:** generate_qwen2/nemotron/zaya/llama signatures each list temp/top_p/max_tokens/repeat_penalty/repeat_window separately; GenerationSamplingPolicy defined at hipfire-generate/src/lib.rs:27; `grep GenerationSamplingPolicy` in generate_arch.rs and generate.rs = 0

*Verification: confirmed*

### [Medium] monolith — `crates/hipfire-server/src/routes/sdapi.rs (8249 LOC)`

**Observation:** sdapi.rs is a single flat 8249-line route file mixing several unrelated responsibilities: HTTP handlers (post_txt2img/img2img/interrogate/extras), Automatic1111 infotext parsing (~15 sdapi_infotext_* helpers), manual PNG byte surgery (png_crc32, find_png_iend_offset, insert/extract_png_text_chunk), inpaint mask/crop/composite math, image-grid assembly, plus ~90 sdapi_* helpers and 97 inline tests (~4000 lines). Production code and tests are each roughly 4000 lines in one module.

**Recommendation:** Split into an sdapi/ submodule tree: sdapi/routes.rs (handlers), sdapi/infotext.rs (parse/format), sdapi/png.rs (chunk/CRC byte surgery — pure and highly testable), sdapi/inpaint.rs (mask/crop/composite), sdapi/mapping.rs (SdGenerationRequest -> DiffusionBatchRequest). The PNG and infotext helpers are GPU-free pure logic and belong in dedicated, independently-testable modules.

**Evidence:** wc -l = 8249; tests begin at L4282 (#[cfg(test)]); 97 `#[test]`/`#[tokio::test]`; 91 `fn sdapi*` helpers; PNG helpers at L1536/1558/1603/1639, infotext parse at L1221-1366, inpaint composite at L2110-2569

*Verification: confirmed*

### [Medium] monolith — `crates/hipfire-serving-core/src/load.rs:322-2114`

**Observation:** load_model() is a single ~1792-line function that dispatches and constructs LoadedModel for all ~10 arch families inline, mirroring the generate() monolith. The surrounding file also holds load_model_safetensors (~446 lines), load_model_pp (~288), and unload_model (~190), so a large fraction of the crate's load logic sits in two oversized functions.

**Recommendation:** Factor per-arch construction out of load_model into `fn load_qwen35(...) -> LoadedModel`, `fn load_llama(...)`, etc., leaving load_model as a thin arch selector. This pairs naturally with the ServingBackend collapse (each arch's loader can return the boxed backend), and makes each arch's load path reviewable and testable in isolation.

**Evidence:** load.rs: fn load_model L322, next top-level fn load_model_safetensors L2114 (so load_model ~1792 lines); load_model_pp L2560, unload_model L2916; only 2 `#[test]` in the whole 3401-line file

*Verification: confirmed*

### [Low] crate-boundary — `crates/hipfire-serving-core/src/load.rs:2114 (called from :360)`

**Observation:** load_model_safetensors reads external HuggingFace format (safetensors + config.json + tokenizer.json) directly inside the serving-core load path, invoked from load_model at L360 on the daemon's inference load hot path. AGENTS.md scopes external-format import (explicitly naming safetensors/GGUF import) to the hipfire-coexistence tooling crate rather than the daemon/runtime. Direct-serving of safetensors is a gray area (a read-only model source, not a conversion tool), but it sits on the exact boundary the invariant draws.

**Recommendation:** Confirm the intended policy: if safetensors is meant to be a first-class runtime source, document that carve-out; if it is import tooling, move the safetensors ingestion (SafetensorsSource + config/tokenizer parsing) into hipfire-coexistence and have the daemon consume only converted .hfq. Either way, keep the serving path free of ad hoc external-format parsing that could accrete GGUF/other importers.

**Evidence:** load.rs L2114 load_model_safetensors uses hipfire_runtime::safetensors_source::SafetensorsSource + reads tokenizer.json/config.json; called at L360 from load_model; AGENTS.md: 'safetensors/GGUF import ... belongs in the hipfire-coexistence binary'

*Verification: confirmed*

### [Low] missing-tests — `crates/hipfire-serving-core/src/{request.rs,model.rs,output_filter.rs,generate.rs,generate_arch.rs,session.rs}`

**Observation:** Grab-bag of smaller hygiene items verified during review: (1) ThinkMode::from_str (request.rs) parses an untrusted request field with a silent NonThink fallback and has no test, despite being pure and trivially testable; (2) request.rs, model.rs, output_filter.rs, and the entire generate/session hot path carry zero in-file tests; (3) the per-arch generate_* functions all share the same ~16-arg shape and per-arch prompt-framing, so the abstraction gaps in findings 6-8 compound the testability gap (a trait-driven dispatch would be mockable, the current free-function ladder is not).

**Recommendation:** Add focused unit tests for the pure request-parsing surface (ThinkMode::from_str truth table, sampling-param clamping, stop-sequence matching) as low-cost regression guards, and prioritize the ServingBackend/SessionServingBackend adoption which is the enabling refactor for making the generation path testable without a GPU.

**Evidence:** request.rs ThinkMode::from_str has no `#[test]`; per-file test counts request.rs/model.rs/output_filter.rs/generate.rs/generate_arch.rs/session.rs all 0

*Verification: adjusted — ThinkMode::from_str at request.rs:33 confirmed with silent `_ => Self::NonThink` fallback (L37) and no test. request.rs/model.rs/generate.rs/generate_arch.rs/session.rs all 0 tests confirmed. However the evidence's claim that output_filter.rs has 0 tests is FALSE: it has 2 tests (#[cfg(test)] at L199, #[test] at L204 and L213). Core thrust holds; output_filter.rs should be dropped from the zero-test list. Low stays.*


## Eval & dispatch: hipfire-eval, evidence, hipfire-dispatch(+tests), steer-harness

*Subsystem key: `eval-dispatch` — 8 finding(s)*

**Subsystem assessment:** This subsystem spans the evidence-driven eval runner (hipfire-eval), shared provenance helpers (hipfire-evidence), the model-facing kernel-family resolver (hipfire-dispatch) plus its integration-test crate (hipfire-dispatch-tests), and a thin daemon steer adapter (hipfire-steer-harness). Overall discipline is good: production code in hipfire-eval and hipfire-evidence has zero .unwrap() (all 391/35 unwraps live in test regions), test counts are high (eval 104, evidence 38, dispatch 93 inline + a dedicated integration crate), and hipfire-dispatch has a well-documented family contract in families/mod.rs. The layering is clean and one-directional: hipfire-dispatch depends on rdna-compute, whose private dispatch module is the low-level Gpu/GpuTensor/kernel-launch layer, while hipfire-dispatch is the higher-level family/arch/quant resolver — this is deliberate layering, not migration leftover or an accidental collision. The main risks are structural: a few oversized single files (lib.rs at 8117 LOC, evidence lib.rs at 3089 LOC), duplication in the examples executor, tests concentrated away from the units they cover, and a "dispatch" name reused across three layers.

### [High] monolith — `crates/hipfire-eval/src/lib.rs:1-8117`

**Observation:** lib.rs is 8117 LOC and mixes at least five responsibilities in one file: the core data model (EvalConfig struct at L411 plus the domain enums), stdout reporting (13 stdout_/render_/format_stdout functions), host/arch/git detection (detect_arch L1967, gfx_target_version_to_arch L1985, rocm_version L2002, git_dirty L1941), path/binary resolution utilities (sanitize_path_component, temp_path, repo_root, resolve_repo_path, and 7 resolve_*_bin helpers), and the run_eval orchestrator (L635) — despite config.rs, datasets.rs, driver.rs, and evidence.rs already existing as sibling modules. The production span is only ~2030 lines; the remaining ~6086 lines (L2031-8117) are a single inline mod tests. The mismatch (heavy modularization elsewhere, grab-bag here) makes lib.rs the hardest file in the subsystem to navigate.

**Recommendation:** Peel the distinct concerns into sibling modules matching the existing pattern: a reporting module for the stdout_*/render_/write_summary functions, a sysdetect module for arch/rocm/git detection, and a paths module for sanitize/temp/repo/resolve_*_bin helpers. Move the EvalConfig struct+enums next to their parser in config.rs so the data model and its CLI parsing live together. This shrinks lib.rs toward a thin orchestrator + re-export facade.

**Evidence:** wc -l = 8117; EvalConfig struct at L411, run_eval at L635; 13 stdout/render reporting fns (grep); detect_arch/rocm_version/git_dirty/gfx_target_version_to_arch all in lib.rs; mod tests spans L2031-8117 (~6086 lines).

*Verification: confirmed*

### [Medium] duplication — `crates/hipfire-eval/src/executor_examples.rs::run_examples_gpqa_item / run_examples_humaneval_item / run_examples_lm_eval_micro_item / run_examples_builtin_barrage_item`

**Observation:** The row-construction helper row_for_model (defined in result.rs:124 with 12 positional parameters) is called 72 times in executor_examples.rs alone. Each run_examples_*_item function (each 200+ lines) repeats near-identical early-return boilerplate for the same failure cases — model-not-a-local-path skip, run-example-binary-not-found skip, create-dir fail, write-prompt fail — differing only in the SuiteId and prompt_format literal. The 12-positional-argument call site is itself a long-parameter-list smell, and battery/suite/config/ctx/model/prompt are constant across every early return inside a single function.

**Recommendation:** Introduce a RowSpec/RowBuilder struct capturing the invariant context (battery, suite, case_id, config, ctx, model, prompt) once per *_item function, so early returns collapse to spec.skip(reason) / spec.fail(err). Factor the shared prelude (resolve bin, create prompt dir, write prompt file) into one helper returning Result<PreparedRun, EvalResult> so the four item builders differ only in dataset-specific metric assembly and scoring.

**Evidence:** grep -c 'row_for_model(' executor_examples.rs = 72; row_for_model signature has 12 params (result.rs:124-137); 'run example binary not found' skip boilerplate repeated 6x; run_examples_humaneval_item L3230-3431 and run_examples_gpqa_item L3431-3663 share the identical skip/fail early-return structure.

*Verification: confirmed*

### [Medium] module-structure — `crates/hipfire-evidence/src/lib.rs:1-3089`

**Observation:** hipfire-evidence is an entire crate implemented as one 3089-line lib.rs presenting a flat API of 64 pub fn, 23 pub struct, 19 pub const, and 51 private fn, with no submodules. Distinct concerns are interleaved: the STANDARD_EVIDENCE_ARTIFACT_SPECS table (L23+), the metric-classification helpers (18 has_*_metric / *_metrics pairs, L685-820), artifact/provenance builders, and hashing. The 18 metric-classification pairs form a repetitive public surface. Production code here is clean (0 unwraps; the metric helpers correctly delegate to has_any_metric/select_metrics), so the issue is organizational reach, not logic quality.

**Recommendation:** Split into modules under a lib.rs facade: specs (artifact spec tables/consts), metrics (classification + copy_* helpers), artifacts (admission/comparison/host-profile artifact builders), and provenance (run provenance + hashing). Consider collapsing the 18 has_X_metric/X_metrics pairs into a single MetricClass enum with .matches()/.select() methods driven by the existing const key arrays, eliminating two near-identical public fns per class.

**Evidence:** wc -l = 3089; item-kind counts: 64 pub fn / 51 fn / 23 pub struct / 19 pub const / 1 pub enum; metric pairs at L685-820 (has_launch_count_metric/launch_count_metrics, has_moe_router_metric/moe_router_metrics, ...); 0 production unwraps (all 35 are in mod tests starting L1910).

*Verification: confirmed*

### [Medium] test-structure — `crates/hipfire-eval/src/lib.rs:2031-8117 and crates/hipfire-eval/src/executor_examples.rs`

**Observation:** All 98 of lib.rs's tests sit in one inline mod tests (L2031-8117); across the 16-file crate only lib.rs and executor_daemon.rs contain any #[test] (executor_daemon has 6). The other 14 files — including datasets.rs, driver.rs, evidence.rs, and the 4310-line executor_examples.rs (the largest run-logic file) — have zero colocated tests. Pure, GPU-free functions that parse subprocess stdout are consequently untested: parse_labeled_f64 (L702), parse_collected_hessian_count (L644), runtime_script_output_is_environment_skip (L1770), and truncate_for_metric (L1787) have no test references at all. These are exactly the pure logic that is unit-testable without a GPU.

**Recommendation:** Move tests into #[cfg(test)] blocks colocated with the module under test (datasets tests in datasets.rs, driver tests in driver.rs, examples tests in executor_examples.rs) so per-module coverage is visible and lib.rs shrinks. Add direct unit tests for the pure parsers listed — table-driven cases over representative stdout strings, including empty/garbled/UTF-8 inputs — since they gate metric extraction from every examples-backed battery.

**Evidence:** grep -c '#[test]' per file: lib.rs=98, executor_daemon.rs=6, all others=0; parse_labeled_f64/parse_collected_hessian_count/runtime_script_output_is_environment_skip/truncate_for_metric each referenced only at their def+call sites in executor_examples.rs, no test region references.

*Verification: confirmed*

### [Low] crate-boundary — `crates/hipfire-dispatch/src/lib.rs:1-23 vs crates/rdna-compute/src/lib.rs:9 (mod dispatch) vs crates/hipfire-runtime/src/lib.rs:28 (pub mod dispatch)`

**Observation:** Three distinct layers are all named 'dispatch'. rdna_compute::dispatch is a private module (mod dispatch) documented as 'High-level GPU dispatch interface' that owns Gpu, GpuTensor, DType, and raw kernel launch (launch_maybe_blob, ensure_kernel, alloc_tensor). hipfire-dispatch is a separate crate documented as 'unified kernel dispatch abstraction' that resolves kernels by family x arch x quant. hipfire_runtime also exposes pub mod dispatch (used as hipfire_runtime::dispatch::is_batchable_la). The dependency direction is clean and one-way (hipfire-dispatch depends on rdna-compute via Cargo path), so this is deliberate layering — NOT migration leftover and NOT an accidental namespace collision — but reusing the exact token 'dispatch' for the low-level Gpu launcher, the mid-level family resolver, and a runtime re-export forces every reader to disambiguate by crate path.

**Recommendation:** Keep the layering (it is correct) but disambiguate the names: rdna_compute::dispatch is really the GPU-handle / kernel-launch layer (consider gpu or launch), while hipfire-dispatch is the family resolver. At minimum, cross-link the two module docs so each states 'this is the low-level launcher, see hipfire-dispatch for the family resolver' and vice versa, so the one-directional relationship is discoverable without tracing Cargo.toml.

**Evidence:** rdna-compute/src/lib.rs L9 'mod dispatch;' (private), L19-22 pub use dispatch::{...Gpu, GpuTensor, DType}; dispatch/mod.rs L5 '//! High-level GPU dispatch interface'; hipfire-dispatch/src/lib.rs L4 'unified kernel dispatch abstraction'; hipfire-dispatch/Cargo.toml depends on rdna-compute; hipfire-runtime/src/lib.rs:28 'pub mod dispatch;'.

*Verification: adjusted — All facts exact: rdna-compute/lib.rs L9 'mod dispatch;' (private) with pub use of Gpu/GpuTensor/DType L19-22; dispatch/mod.rs L5 '//! High-level GPU dispatch interface'; hipfire-dispatch/lib.rs L4 doc + Cargo dep on rdna-compute L8; hipfire-runtime/Cargo.toml:70 dep on hipfire-dispatch; hipfire-runtime/lib.rs:28 'pub mod dispatch'; is_batchable_la in runtime/src/dispatch.rs. But the finding itself concedes the layering is deliberate and correct (clean one-way deps, not a collision), and the minimum recommendation is cross-linking module docs — that is naming/discoverability hygiene, not a 'clear antipattern worth scheduled refactor', so Low fits better than Medium.*

### [Low] test-structure — `crates/hipfire-dispatch-tests/src/lib.rs and crates/hipfire-dispatch-tests/Cargo.toml`

**Observation:** The separate hipfire-dispatch-tests crate is justified, not redundant: it depends on both hipfire-runtime and hipfire-dispatch (with feature test-utils), and since hipfire-runtime already depends on hipfire-dispatch (hipfire-runtime/Cargo.toml:70), integration tests that need both cannot live inside hipfire-dispatch without creating a dependency cycle — a third crate is the correct way to exercise the family x arch x quant matrix end-to-end (e.g. hipfire_runtime::dispatch::is_batchable_la). One minor quirk: the family test modules live in src/*.rs gated on #[cfg(test)] rather than in the tests/ directory (only tests/golden.rs is a true integration test), done so modules can share helpers like make_caps/ALL_ARCHS. hipfire-dispatch also keeps its own inline tests.rs (80 tests) and coverage_tests.rs (13 tests).

**Recommendation:** Keep the crate — the cycle-break rationale is sound. Optionally note the reason in the crate doc so the src/-with-cfg(test) layout is not mistaken for a mispackaged library. Consider whether hipfire-dispatch/src/coverage_tests.rs and tests.rs could be consolidated to reduce the number of parallel dispatch test locations (inline tests.rs, inline coverage_tests.rs, external crate).

**Evidence:** hipfire-dispatch-tests/Cargo.toml deps: rdna-compute + hipfire-runtime + hipfire-dispatch(test-utils); hipfire-runtime/Cargo.toml:70 depends on hipfire-dispatch (confirms cycle if tests were internal); src/lib.rs declares mod arch_caps/deepseek4/dtype/llama/qwen2/qwen35 all under #[cfg(test)]; deepseek4.rs uses hipfire_runtime::dispatch::is_batchable_la.

*Verification: confirmed*

### [Low] abstraction — `crates/hipfire-dispatch/src/families/rotation.rs:53,89-113,139-192`

**Observation:** RotationFamily::run derives has_awq = params.awq_scale.is_some() (L53), then branches on match (has_awq, batched) and calls params.awq_scale.unwrap() in the true arms (L100, L112, L139, L165, L192). It is safe by construction today, but it launders an Option through a boolean and re-asserts the invariant with unwrap on the hot dispatch path, so a future refactor that changes how has_awq is computed would turn these into inference-time panics. It is also inconsistent with the Givens arm immediately above (L58-69), which extracts every optional parameter via .ok_or_else(HipError::new(...))? and returns typed errors.

**Recommendation:** Bind the value directly instead of re-unwrapping: match on params.awq_scale (Some(scale) vs None) or use if let Some(scale) = params.awq_scale to carry the scale into the AWQ arms, mirroring the ok_or_else style used for Givens. This removes the unwrap and makes the awq-present invariant type-enforced.

**Evidence:** rotation.rs L53 'let has_awq = params.awq_scale.is_some();'; unwraps at L100/112/139/165/192 all inside has_awq==true arms; Givens arm L58-69 uses ok_or_else typed errors; file has no #[cfg(test)] so this is production code.

*Verification: confirmed*

### [Low] abstraction — `crates/hipfire-dispatch/src/traits.rs:5-7`

**Observation:** The KernelFamily trait declares only fn name(&self) -> &'static str, yet families/mod.rs documents a rich per-family contract (each family owns a KernelRegistry, exposes resolve() and typed run*() entry points, validates non-empty tables at construction). The shared behavior that actually defines a 'family' is enforced entirely by documentation and per-struct convention, not by the trait, so the trait provides almost no abstraction and could be inlined as a plain name() method.

**Recommendation:** Either lift the common surface into the trait (e.g. an associated Registry accessor and a validate()/resolve hook) so the family contract is compiler-checked, or drop the near-empty trait and document the convention. Given that each family's run*() signature legitimately differs, a small trait covering name()+registry validation is the pragmatic middle ground.

**Evidence:** traits.rs L5-7 defines KernelFamily with a single method name(); families/mod.rs L10-38 describes the multi-point family contract (KernelKey arms, <family>_table::populate, registry.validate(), typed resolve()+run*()) none of which is expressed in the trait.

*Verification: confirmed*


## Core libraries: model, generate, prompt, state, scheduler, detect, primitives, …

*Subsystem key: `core-libs` — 5 finding(s)*

**Subsystem assessment:** The core-libs subsystem is broadly healthy and clearly has already been through dedup/hygiene passes: the tokenizer "duplication" the assignment flagged is resolved (crates/hipfire-runtime/src/tokenizer.rs is an 11-line `pub use` re-export, and the whitespace-normalization fns in hipfire-model/tokenizer.rs are thin shims delegating to hipfire-prompt), the numeric primitives are centralized in a zero-dep hipfire-primitives leaf used by 11 crates, and the micro-crates are justified boundaries (hipfire-build-info isolates the vergen-gitcl build-dep; hipfire-hash/primitives are zero-dep leaves). Test coverage is a strength: hipfire-detect is exemplary (one module per detector, ~64 tests across 13 files), and state/model/prompt/generate/scheduler each carry 20-48 tests. The main risks are coupling, not correctness: architecture-specific knowledge (qwen35, mamba/lfm2/nemotron/minimax) has leaked into nominally generic control-plane crates — most severely hipfire-generate (218 qwen35 references) and hipfire-scheduler (fragile stringified-arch_id/substring dispatch that bypasses the canonical arch-family table). Secondary concerns are module structure: hipfire-model is a 3332-LOC multi-responsibility grab-bag (including misplaced HIP device inventory), and hipfire-state is a 2592-LOC flat lib mixing wire-JSON, parsers, and the allocator. The HFQ binary parser is defensively bounds-checked and safe.

### [High] coupling — `crates/hipfire-generate/src/lib.rs::(compute_qwen35_prefix_hash L407, plan_generate_batch_prefill_qwen35 L1017, Qwen35FusedDensePrefillSessionSpec L1167, validate_qwen35_fused_grouped_moe_prefill_batch_preflight L1226)`

**Observation:** hipfire-generate is presented as the generic generation/batch-protocol crate (it owns GenerateTextRequest, sampling policy, OpenAI response builders) but its 2522-LOC lib.rs is saturated with one architecture's batch protocol. grep counts 218 references to qwen35/qwen3.5 and 43 qwen35-named fn/type declarations (prefix-hash, semantic-boundary checkpoints, fused-dense/grouped-MoE prefill contracts, decode-batch step results), versus only 4-6 mentions each of lfm2/nemotron/minimax. The batch prefill/decode contract is effectively qwen35-only. As other families gain batch protocols they will either bloat this crate further or fork their own paths, and the crate name misrepresents its contents.

**Recommendation:** As an immediate, mechanical step, move the qwen35 symbols into a `generate::qwen35` submodule so the generic contract (envelopes, validation, JSON builders) is visibly separated. Longer term, define a per-arch batch-protocol trait (e.g. `trait BatchPrefillProtocol { fn prefix_hash(..); fn plan(..); fn validate(..); }`) that the arch crates implement, leaving hipfire-generate to own only arch-agnostic request/response and sampling types. This unblocks adding new families without editing the shared crate.

**Evidence:** grep -ci qwen35 = 218; grep -cE '(fn|struct|enum) .*[Qq]wen35' = 43; lfm2/nemotron/minimax each 4-6; submodules sampler.rs/loop_guard.rs/eos_filter.rs exist but lib.rs is 2522 LOC and does not depend on hipfire-arch-qwen35 (Cargo.toml).

*Verification: confirmed*

### [High] coupling — `crates/hipfire-scheduler/src/lib.rs:575-617 (worker_key_is_qwen35, worker_key_is_state_arena_conservative), 656-671 (inferred_decode_state_kinds)`

**Observation:** The scheduler classifies model families for batching decisions with fragile string matching: worker_key_is_qwen35 does `matches!(worker_key.arch_id.as_str(), "5" | "6")` and worker_key_is_state_arena_conservative matches `"10" | "11" | "14"`, plus `contains("qwen35"/"minimax"/"lfm2"/"nemotron"/"mamba")` over arch_id, artifact_path and feature_flags. inferred_decode_state_kinds lowercases worker_key_id and substring-matches to synthesize state kinds. This bypasses the canonical `model_arch_family(arch_id: u32)` / `is_qwen35_family_arch_id` table in hipfire-model (lib.rs:202,230). The root cause is an impedance mismatch: ModelWorkerKey.arch_id is a String (lib.rs:100) produced by stringifying a u32 (ModelWorkerId::from_runtime_parts, lib.rs:114). These classifications gate mamba/recurrent singleton-batching (decode_sessions_compatible_for_batch requires_singleton), so a silent misclassification when arch ids or filenames change can batch recurrent-state sessions incorrectly.

**Recommendation:** Carry the numeric arch id (or a resolved ModelArchFamily enum) on ModelWorkerKey instead of a stringified id, and have the scheduler call the existing hipfire-model classifiers (model_arch_family / is_qwen35_family_arch_id) rather than magic-number string matches. Replace substring family sniffing with explicit capability fields on the worker key (e.g. requires_token_ordered_recurrent, supports_multi_session_state_batch) populated by the arch crate at load time, so batching policy reads declared capabilities, not filename heuristics.

**Evidence:** matches!(arch_id.as_str(), "5"|"6") at L576; matches!(..., "10"|"11"|"14") at L582; ModelWorkerKey.arch_id: String at lib.rs:100; canonical model_arch_family(u32) at hipfire-model/lib.rs:202; scheduler imports from hipfire_model but not the arch-family fns (L8-12).

*Verification: confirmed*

### [Medium] monolith — `crates/hipfire-model/src/lib.rs (AcceleratorInventory L286-338, LlmModelRegistry L1249, parse_canonical_model_artifact_name L1741, discover_dflash_draft_for_model L1890, read_hfq_metadata L998)`

**Observation:** hipfire-model is a 3332-LOC single lib.rs (only gguf/tokenizer/model_support_generated are submodules) whose stated purpose is "Model artifact identity and source contracts" but which actually spans at least seven distinct responsibilities: HFQ container binary parsing, model-worker identity/keys, HIP accelerator/device inventory, arch-family mapping, canonical artifact-name parsing (the AGENTS.md naming spec), the model registry + parameter counting, and model/dflash-draft file discovery. AcceleratorInventory/AcceleratorDeviceInfo (HIP device probing, memory bytes, ordinals) is clearly misplaced in a model-metadata crate and overlaps conceptually with hipfire-sysinfo/gpu.rs.

**Recommendation:** Split lib.rs into cohesive modules (hfq, identity, naming, registry, discovery, arch) — this is a low-risk mechanical extraction since the concerns barely share state. Relocate AcceleratorInventory/AcceleratorDeviceInfo to hipfire-sysinfo (or a small device-info crate) and re-export if needed, so scheduler and others depend on hardware types from a hardware crate rather than the model crate. hipfire-detect's one-file-per-concern layout is the target shape.

**Evidence:** wc -l lib.rs = 3332; only 3 `pub mod` (gguf, tokenizer, model_support_generated); ~40 pub fns + ~30 structs across the concerns above; crate description in Cargo.toml is model-identity only; AcceleratorDeviceInfo carries kind/ordinal/total_memory_bytes/integrated/runtime (L308-320).

*Verification: confirmed*

### [Medium] module-structure — `crates/hipfire-state/src/lib.rs (GenericSequenceStateArena impl L985-1101 [struct decl at L26]; ~27 *_json builders scattered file-wide, only 12 within L649-985; ~20 parse_* fns; arch label fns L380-410)`

**Observation:** hipfire-state is a 2592-LOC flat lib.rs (~1380 non-test LOC; test module starts L1380) that mixes three different kinds of code with no module boundaries: the actual allocator/eviction logic (GenericSequenceStateArena, select_lru_sequence_state_eviction_candidates), ~30 hand-rolled `*_json` wire-envelope builders and ~10 `parse_*` request parsers, and arch-specific label helpers (qwen35_kv_deltanet_state_kind_labels, minimax/lfm2/nemotron_h_state_kind_labels) that re-encode arch knowledge belonging to the arch crates. Coverage is good (48 tests), so this is structure rather than correctness, but the single file obscures where the real allocator logic lives.

**Recommendation:** Split into submodules: `arena` (allocator/eviction, the part with real logic worth focused tests), `wire` (the *_json builders + parse_* request parsers), and `handle` (SequenceStateHandle parsing). Move the per-family state-kind label functions to the respective hipfire-arch-* crates (or drive them from a capability descriptor) so state stops enumerating families by name. Follow the hipfire-detect pattern of one concern per file.

**Evidence:** wc -l = 2592; first test marker L1380; arena impl L985-1101; arch label fns qwen35_kv_deltanet/minimax/lfm2/nemotron_h at L380-410; ~30 pub fns ending in _json between L649-985.

*Verification: adjusted — Core claim (2592 LOC flat file mixing arena/eviction logic, JSON wire builders, and per-family label fns) is real: impl GenericSequenceStateArena at L985-1101, arch label fns L380-410, 48 tests all verified. But the '~30 *_json builders L649-985' number/range is off — there are 27 _json fns total across the whole file and only 12 within L649-985, and parse_* fns count 20 (not ~10). Numbers need correction; Medium severity unchanged.*

### [Low] utils-sprawl — `crates/hipfire-config/src/lib.rs (38 fn default_*), plus low-test FFI/subprocess crates: hipfire-rocm (3 tests), hipfire-lock (3), hipfire-coherence (4)`

**Observation:** Remaining minor items, none structural. (1) hipfire-config/lib.rs has 38 flat `default_*` serde helper functions before the config struct — idiomatic serde but voluminous; grouping related defaults or using a defaults module would aid navigation. (2) hipfire-rocm (283 LOC, 3 tests), hipfire-lock (326 LOC, 3 tests) and hipfire-coherence (638 LOC, 4 tests) have low unit-test counts, but their logic is dominated by FFI/syscall/subprocess driving with limited pure surface, so this is acceptable rather than a gap. (3) hipfire-prompt (2378 LOC) and hipfire-scheduler (1850 LOC) are large single files but are mostly cohesive and test-heavy (prompt's non-test code is only ~878 LOC), so they are lower priority than model/state for splitting. Verified positives worth preserving: micro-crates are justified (hipfire-build-info's 15-LOC lib exists solely to isolate the vergen-gitcl build-dependency; hipfire-hash/primitives are zero-dep leaves; primitives is used by 11 crates), the tokenizer duplication is already resolved via a re-export shim plus delegating wrappers, and the HFQ binary parser (read_hfq_inventory L1030-1121) bounds-checks every variable-length slice so its try_into().unwrap()s cannot panic on malformed model files.

**Recommendation:** No urgent action. Optionally group the config default_* helpers into a submodule and add a couple of pure-logic unit tests for hipfire-coherence's coherence_output_from_stats. Otherwise treat these crates as healthy and leave the boundaries as-is.

**Evidence:** grep -c 'fn default_' config/lib.rs = 38; test counts: rocm 3, lock 3, coherence 4; build-info/lib.rs = 15 LOC with vergen-gitcl build-dependency; runtime/tokenizer.rs = 11-line `pub use hipfire_model::tokenizer::*`; HFQ parser bounds checks at L1069/1082/1087/1097.

*Verification: confirmed*


## Tooling: cli, tui, train, steer, atlas, coexistence, npu, xdna, hneurons, redline

*Subsystem key: `tooling` — 8 finding(s)*

**Subsystem assessment:** The tooling subsystem spans a healthy clap-based CLI, a ratatui TUI, a large fp32 training crate, steering/coexistence LoRA tools, three small well-factored accelerator crates (npu/xdna/hneurons), and the experimental redline direct-KMD driver. Overall structure is idiomatic, but two problems stand out: the documented coexistence boundary (all import/export/conversion tooling in hipfire-coexistence) is violated by a full diffusers-to-hfq converter living inside the runtime hipfire-diffusion crate, and the 11.3k-LOC hipfire-train crate has effectively no unit tests over its pure config/schedule/statistics logic, some of which panics on adversarial input. Secondary risks: a 2.5k-LOC untested unsafe redline crate that no crate consumes and whose own design docs recommend abandoning, a 1.7k-LOC CLI diffusion monolith, and process-global mutable steering state read on the inference hot path. The small accelerator crates (npu/xdna/hneurons) are the healthiest part of the subsystem: clear SRP, module docs, and tests.

### [High] crate-boundary — `crates/hipfire-diffusion/src/lib.rs:8452-8680 (import_diffusers_to_hfq, import_single_file_checkpoint_to_hfq); invoked from crates/hipfire-cli/src/commands/diffusion.rs:421`

**Observation:** AGENTS.md mandates that ALL import/export/format-conversion/interop tooling live in hipfire-coexistence, not in runtime/inference crates. A complete offline Diffusers-and-single-file-safetensors to .hfq converter (import_diffusers_to_hfq reads model_index.json, walks components, imports tokenizer/text_encoder/unet weights) lives inside hipfire-diffusion, the runtime inference crate. This is offline format conversion, not the permitted load-at-inference path: it writes a new .hfq artifact. hipfire-coexistence is only 209 LOC and handles just LoRA export/merge/convert, so the diffusion importers leaked into the wrong crate.

**Recommendation:** Move import_diffusers_to_hfq, import_single_file_checkpoint_to_hfq, and their helpers (push_import_file_entry, add_component, read_json wrappers) into hipfire-coexistence (e.g. a `diffusion` subcommand group), leaving hipfire-diffusion with only the open_hfq/inspect load path used at inference. The CLI `hipfire diffusion import` arm should either call into hipfire-coexistence or be relocated there, matching how lora export/merge/convert are already isolated. This shrinks the already-8000+-LOC diffusion lib and restores the documented lean inference boundary.

**Evidence:** import_diffusers_to_hfq at lib.rs:8452 returns DiffusionModelSummary and writes an .hfq; hipfire-coexistence/src/main.rs is 209 LOC covering only lora {export,merge,convert}; CLI diffusion.rs:421 dispatches DiffusionCommand::Import into the diffusion-crate importer.

*Verification: confirmed*

### [High] missing-tests — `crates/hipfire-train (11314 LOC, 71 files, 1 total #[test] in src/oqplus_quant.rs); crates/hipfire-train/src/train_loop.rs:50-91; crates/hipfire-train/src/config.rs:48-106`

**Observation:** The largest tooling crate has exactly one #[test] in the whole src tree; all pure, GPU-free logic is unverified by cargo test. The 41 gradcheck files under examples/ are manual binaries, not test targets. Pure functions carry latent panics: rank() sorts with a[i].partial_cmp(&a[j]).unwrap() (train_loop.rs:52) which panics on any NaN score; softmax_t() divides every element by z with no guard for z==0 or tau==0 (train_loop.rs:90); and LlamaConfig::from_json_value computes head_dim = hidden_size / num_attention_heads (config.rs:65) which panics on a config with num_attention_heads=0. None of the config parser, correlation stats, or optimizer bias-correction math is exercised by a unit test.

**Recommendation:** Add a #[cfg(test)] module covering LlamaConfig::from_json_value (valid llama, rejected model_type, missing fields via uget, and the num_attention_heads=0 / head_dim-absent edge), and pearson/spearman/rank/softmax_t (known-value vectors, NaN input, constant input, tau=0). Harden the panics: use total_cmp for rank ordering, guard softmax_t against z==0 and non-positive tau, and validate num_attention_heads>0 in the config parser. The gradcheck examples are valuable but should be complemented by fast deterministic unit tests so regressions in config/schedule/stats math are caught in CI without a GPU.

**Evidence:** grep of #[test] across crates/hipfire-train/src returns only oqplus_quant.rs; train_loop.rs:52 uses partial_cmp().unwrap(); train_loop.rs:90 `v / z` with no zero-guard; config.rs:65 `hidden_size / num_attention_heads`.

*Verification: confirmed*

### [Medium] module-structure — `crates/redline/src/ (2584 LOC src, 5660 LOC total, 0 test files); crates/redline/src/lib.rs:28-31; Cargo.toml:59`

**Observation:** redline is a 2.5k-LOC (src) unsafe direct-KMD GPU driver (device/dispatch/drm/hsaco/kfd/pm4/queue, raw ioctls) with zero tests and zero external consumers: it is listed as a workspace member (Cargo.toml:59) but no other crate depends on it. Its own lib.rs architecture doc advertises modules memory.rs and sync.rs that do not exist in src/, so the top-level documentation is already stale. PHASE2_RESULT.md concludes with a recommendation to not port to HSA and to possibly skip redline entirely. It is compiled experimental scaffolding accruing maintenance and portability surface with no coverage.

**Recommendation:** Either (a) gate redline out of the default workspace build (move it behind an explicit feature or an experimental workspace exclude) until a consumer exists, or (b) if it is kept, fix the lib.rs module table to match the actual files and add smoke/unit tests for the pure parsing pieces that don't need hardware (hsaco ELF parsing, pm4 packet construction). At minimum reconcile the doc drift so the crate does not claim modules it lacks. Given the AGENTS.md portability constraint, an untested raw-ioctl path across RDNA2/3/4 is a liability while unmaintained.

**Evidence:** redline src has no #[test]/#[cfg(test)]; grep for `redline` in crates/*/Cargo.toml matches only redline's own manifest plus the root workspace list; lib.rs:29-31 references memory.rs/sync.rs absent from src/; PHASE2_RESULT.md 'Recommendation' section says don't port / skip redline.

*Verification: confirmed*

### [Medium] monolith — `crates/hipfire-cli/src/commands/diffusion.rs:1-1724 (run() at :418)`

**Observation:** A single 1724-LOC command module implements eight subcommands (Import, Inspect, Preflight, Txt2Img, Img2Img, Smoke, Quantize, Calibrate), all their clap arg structs (DiffusionCalibrateArgs, DiffusionQuantizeArgs, DiffusionImportArgs, DiffusionTxt2ImgArgs, DiffusionImg2ImgArgs, DiffusionSmokeArgs at :117-418), plus high-res resolution math (highres_first_pass_dimensions, highres_target_dimensions, scaled_highres_dimension, aspect_scaled_dimension) and prompt/batch expansion helpers. It is by far the largest CLI command file (next is gen_env_docs at 800). The dimension helpers are pure and testable but sit in the same grab-bag file.

**Recommendation:** Split into a diffusion command submodule: e.g. commands/diffusion/{args.rs, txt2img.rs, img2img.rs, smoke.rs, quantize.rs, calibrate.rs, highres.rs} with a small mod.rs run() dispatcher. The pure highres_*/aspect_scaled_dimension helpers should move to a highres.rs with #[test]s over the zero/aspect-preserving branches (the match at :770-818 has an unreachable! that a test would pin). This mirrors the already-clean per-command layout the rest of hipfire-cli uses.

**Evidence:** diffusion.rs is 1724 LOC; run() at :418 matches 8 DiffusionCommand variants; arg structs span :117-418; highres dimension fns at :761-843 are pure but co-located.

*Verification: confirmed*

### [Medium] coupling — `crates/hipfire-steer/src/lib.rs:150-193 (enum Session, session(), set_session, mark_session_changed, begin_capture); read on hot path via crates/hipfire-arch-gemma3/src/forward.rs`

**Observation:** Steering state is a process-global mutable singleton: session() returns a &'static RwLock<Session> and maybe_steer_block()/maybe_steer_block_batched() consult it during the transformer forward pass, which is invoked from an architecture crate's forward.rs. This couples the inference hot path to hidden global state, makes it impossible to run two models or two steering configurations concurrently in one process, and forces tests to coordinate a shared global (setup/teardown ordering). The capture accumulators and loaded-adapter stack all hang off this one static.

**Recommendation:** Thread a steering context/handle through the forward call (or store it on the model/session struct the arch crate already owns) instead of a global RwLock, so steering becomes an explicit per-model dependency. If a global must remain for the injection-hook ergonomics, document the single-session invariant and keep the pure transforms (apply_direction, derive_directions, dot, normalize) free of the global so they stay independently testable — those already have good coverage (27 tests) and should not regress when the state is de-globalized.

**Evidence:** lib.rs:165 `fn session() -> &'static RwLock<Session>`; :175 set_session; :417 maybe_steer_block reads it; grep shows hipfire_steer::/maybe_steer_block used in crates/hipfire-arch-gemma3/src/forward.rs.

*Verification: confirmed*

### [Low] duplication — `crates/hipfire-atlas/src/parse.rs:32,62-82,98,108,122-126`

**Observation:** Regexes are compiled with Regex::new(...).expect('valid regex') inside the parse functions, so every call to parse_bench_summary/parse_dflash_summary/parse_profile_sections recompiles constant patterns. This is repeated across at least five sites and is a well-known Rust antipattern (regex compilation is expensive relative to matching). The patterns are compile-time constants, so the expect() cannot fail in practice but re-pays compilation on each invocation.

**Recommendation:** Hoist each pattern into a std::sync::LazyLock<Regex> (or once_cell::Lazy) at module scope and reference it in the functions. This removes the per-call compile cost and the repeated expect() boilerplate, and centralizes the pattern definitions. The existing parse tests (parse_bench_summary/parse_dflash_summary at the bottom of the file) will continue to pass unchanged.

**Evidence:** Regex::new(...).expect('valid regex') appears at parse.rs:32, :82, :98, :108, :126, each inside a pub fn body rather than a static.

*Verification: confirmed*

### [Low] utils-sprawl — `crates/hipfire-cli/src/commands/{gen_env_docs.rs:800, gen_model_support.rs:627, gen_config_schema.rs:235, gen_docs.rs:130}`

**Observation:** About 1800 LOC of hidden documentation-generation commands (source-tree env-var scanner, model-support-matrix renderer, config-schema emitter, clap-to-markdown/man renderer) are embedded in the shipping hipfire inference CLI binary. These are build/maintenance tooling that pull in extra deps (clap-markdown, clap_mangen) and inflate the inference binary's command surface. gen_env_docs.rs alone (800 LOC) statically scans the tracked source tree.

**Recommendation:** Consider relocating these to a dedicated xtask-style tooling binary or a docs-tooling crate so the primary CLI stays lean, matching the AGENTS.md intent of keeping non-inference tooling out of the shipped inference path. If they must stay in-tree with the CLI, at least gate them behind a non-default `docs-tooling` feature so release builds do not carry the clap-markdown/clap_mangen dependencies.

**Evidence:** gen_env_docs.rs is 800 LOC and its module doc says it scans the tracked source tree; Cargo.toml pulls clap-markdown and clap_mangen used only by these hidden commands.

*Verification: confirmed*

### [Low] test-structure — `crates/hipfire-tui/src/ui.rs:1004-1132 (visible_rows:1125, scroll_start:1129); tests only in crates/hipfire-tui/src/hipfire/registry.rs`

**Observation:** ui.rs is a 1132-LOC single render module covering eight distinct screens (home, chat, models, runtime, logs, settings, training, system) via 22 draw_* functions, and contains pure, GPU-free scroll/layout helpers (visible_rows, scroll_start, training_detail_lines) that are untested — the only test file in the whole TUI crate is registry.rs (3 tests). The scroll math (scroll_start computing a viewport offset from selected/height/chrome) is exactly the kind of off-by-one-prone pure logic that unit tests should pin.

**Recommendation:** Extract the pure viewport helpers (visible_rows, scroll_start) into a small module and add #[test]s for the boundary cases (selected at 0, selected past the last visible row, chrome larger than height). Optionally split ui.rs into per-screen submodules (ui/home.rs, ui/models.rs, ...) so the render monolith becomes navigable; each draw_* is already self-contained, so the split is mechanical.

**Evidence:** ui.rs is 1132 LOC with 22 fn definitions; visible_rows at :1125 and scroll_start at :1129 are pure; grep shows the only #[test] in hipfire-tui/src is registry.rs.

*Verification: confirmed*


## Workspace structure & crate layering

*Subsystem key: `workspace-structure` — 10 finding(s)*

**Subsystem assessment:** The hipfire workspace is a 65-member Cargo workspace (66 crate dirs; 3 wasm UI crates correctly excluded) implementing a HIP/ROCm-direct LLM+diffusion inference stack. The overall layering is broadly sound — clean leaf crates (hip-bridge, hipfire-lock, hipfire-hash), a compute layer (rdna-compute, hipfire-dispatch), a runtime hot path, sibling arch-* crates, serving, and binaries — and the arch crates avoid a cycle with runtime via the dev-dependency trick. Health risks are concentrated in three areas: (1) a hot-path layering inversion where hipfire-runtime pulls the entire hipfire-eval evidence harness (and transitively the tokio-process daemon adapter) for a single host-profile function; (2) god-node fanout because GpuTensor lives inside the 63.5k-LOC rdna-compute monolith, forcing 20 crates to depend on the whole compute crate; and (3) zero dependency centralization — no [workspace.dependencies], so 40+ crates repeat and diverge third-party version/feature pins. Within-crate monoliths (qwen35.rs at 32.6k lines, diffusion lib.rs at 10.3k lines, rdna-compute dispatch/) are the main maintainability drags. Manifest hygiene is weak: a duplicated member entry and an orphan crate indicate the members list is hand-maintained without lint.

### [High] crate-boundary — `crates/hipfire-runtime/Cargo.toml:61 + crates/rdna-compute/src/dispatch/mod.rs:124`

**Observation:** Two dependency-direction problems distort the layering. (a) hipfire-runtime, the inference hot path, has a non-dev dependency on hipfire-eval (a ~21k-LOC evidence/battery harness) used for exactly ONE symbol: host_profile.rs:12 `use hipfire_eval::collect_default_host_profile` (defined hipfire-eval/src/lib.rs:631). hipfire-eval in turn depends on hipfire-daemon-adapter (tokio process-spawning, out-of-process daemon client) and hipfire-daemon-protocol, so the hot path drags the whole eval+daemon-client+tokio-process closure into its build — directly against the AGENTS.md 'inference path stays lean' invariant. (b) GpuTensor, referenced by 130 files across the tree, is defined inside the 63.5k-LOC rdna-compute monolith (dispatch/mod.rs:124), so all 20 crates that need the GPU tensor handle must depend on the entire compute crate (kernels, compiler, profiler, rocblas). HipError/HipResult are already correctly isolated in the hip-bridge leaf (error.rs:14,29), which is the pattern the rest of the primitives should follow.

**Recommendation:** Extract shared leaf types so hot-path and high-layer crates stop reaching across the stack. Move HostProfile/collect_host_profile into a leaf (hipfire-sysinfo already exists and fits) and have both runtime and eval depend on it; delete the runtime->eval edge. Extract GpuTensor plus the core buffer/handle types into a thin rdna-compute-core (or reuse the mis-named hipfire-primitives) leaf that hip-bridge-only depends on. Target layered diagram (arrows = depends-on, downward):

BINARIES  cli / daemon / server / coexistence / tui
   |
SERVING   serving-core --- diffusion
   |
ARCH      arch-* (qwen35, llama, gemma3, ...)   siblings; only VL->base laterals
   |
RUNTIME   hipfire-runtime   (hot path; MUST NOT depend on eval/daemon-adapter)
   |
MODEL/IO  model / state / generate / prompt / quantize
   |
COMPUTE   rdna-compute-core (GpuTensor,pool,ctx) - rdna-compute-{gemm,attn,moe} - hipfire-dispatch
   |
LEAF      hip-bridge(HipError) / hipfire-sysinfo(HostProfile) / hipfire-numeric(conv,fwht) / lock / hash

SIDE (offline tooling, never a runtime dep):  eval / daemon-adapter / coherence

This severs runtime->eval, collapses the GpuTensor fanout to a leaf, and quarantines the eval/daemon cluster as offline tooling.

**Evidence:** hipfire-runtime/Cargo.toml:61 `hipfire-eval = { path = ... }`; only user is host_profile.rs:12 (single symbol collect_default_host_profile, eval/lib.rs:631); GpuTensor rdna-compute/src/dispatch/mod.rs:124 referenced by 130 files; 20 crates depend on rdna-compute.

*Verification: confirmed*

### [High] crate-boundary — `Cargo.toml:79-82 no [workspace.dependencies]; divergence at hipfire-diffusion/Cargo.toml:11 vs hipfire-media/Cargo.toml:11, and hipfire-prompt/Cargo.toml:12 vs hipfire-runtime/Cargo.toml:95`

**Observation:** The root manifest declares [workspace.package] but has no [workspace.dependencies] block (grep count 0). Every third-party version and feature set is therefore repeated per crate and drifts silently: serde is declared in 33 crates, serde_json in 42, image in 7, tokio in 5, minijinja in 2. Features already diverge in ways that only cargo's implicit feature-unification papers over: image is png-only in hipfire-diffusion (Cargo.toml:361) but png+jpeg elsewhere; minijinja is loop_controls+json in hipfire-prompt (line 501) but loop_controls+json+preserve_order in hipfire-runtime (line 573); tokio has five distinct feature sets (['full'], ['rt-multi-thread'], ['io-util','macros','process','rt'], etc.). At 65 crates this is a maintainability hazard — a version bump or a feature audit must touch dozens of files, and correctness (e.g. serde_json preserve_order, which is load-bearing for chat-template key order) depends on unification rather than intent.

**Recommendation:** Add [workspace.dependencies] to the root Cargo.toml pinning every shared third-party crate once (serde, serde_json, tokio, image, regex, rayon, base64, twox-hash, tracing, memmap2, reqwest, uuid, anyhow, minijinja*), then convert crate manifests to `serde.workspace = true` / `serde = { workspace = true, features = [...] }`. Where a feature is truly hot-path-critical (serde_json preserve_order), declare it in the workspace default so it cannot be accidentally dropped. This makes version/feature intent explicit and reduces a bump to a one-line change.

**Evidence:** Cargo.toml has [workspace.package] (79-82) but no [workspace.dependencies]; serde in 33 Cargo.tomls, serde_json in 42, image in 7 with png-only in diffusion/Cargo.toml:361 vs png+jpeg elsewhere; minijinja feature divergence prompt:501 vs runtime:573.

*Verification: adjusted — No [workspace.dependencies] block confirmed; feature drift is real (serde=33, serde_json=43 [not 42], image=7, tokio=5, minijinja=2). But the observation's line citations are fabricated: diffusion image png-only is Cargo.toml:11 (file is 22 lines, not 361); prompt minijinja is line 12 (not 501); runtime minijinja+preserve_order is line 95 (not 573). Core claim and primary location (79-82) hold; supporting numbers need correction.*

### [Medium] monolith — `crates/hipfire-arch-qwen35/src/qwen35.rs (32,648 lines)`

**Observation:** hipfire-arch-qwen35 is the second-largest crate (58k src LOC) and 32,648 of those lines live in a single file, qwen35.rs. A file this size is effectively unnavigable, is a merge-conflict magnet, and defeats module-level unit testing — pure logic (tensor-shape inference, module-table parsing like qwen35_tensor_data/hfq_plain_tensor_as_f32, MoE expert routing, KLD eval glue at line ~5051) is interleaved with GPU dispatch and cannot be tested in isolation.

**Recommendation:** Split qwen35.rs along its natural responsibilities into a module tree within the crate: config/weight-layout parsing, hfq tensor accessors, attention/rope forward, MoE expert dispatch, and eval/KLD hooks. Extract the pure CPU-side helpers (shape math, module-table parsing, dtype conversion) into their own modules with #[cfg(test)] unit tests that need no GPU. This is a within-crate refactor first; if build times warrant, the config/tokenizer/layout portions can later become a hipfire-arch-qwen35-model sub-crate separate from the GPU forward pass.

**Evidence:** wc -l crates/hipfire-arch-qwen35/src/qwen35.rs = 32648; contains mixed concerns e.g. qwen35_tensor_data (L1695), hfq_plain_tensor_as_f32 (L2322), KldEvalOutcome (L5051).

*Verification: confirmed*

### [Medium] monolith — `crates/rdna-compute/src (63.5k LOC), dispatch/ subdir`

**Observation:** rdna-compute is the largest crate (63.5k src LOC, ~92.9k with tests) and every consumer takes the whole thing. The dispatch/ subdirectory already presents clean split seams: gemv.rs (6,121), gemm_qkv.rs (5,890), attention.rs (5,563), mod.rs (5,153), fused.rs (4,308), deepseek4.rs (3,536), moe.rs, rope.rs, kv.rs. These are largely independent kernel-dispatch wrapper families that share only the core context/pool/GpuTensor types.

**Recommendation:** Peel rdna-compute into a small rdna-compute-core (KernelCompiler, pool, arch_caps, feature_flags, GpuTensor, HipResult re-export) plus feature-grouped sub-crates rdna-compute-gemm, rdna-compute-attention, rdna-compute-moe that depend on core. Arch crates then pull only the families they use, shrinking their build closure and letting the pure host-side portions (arch_caps gating, feature_flags, tiling math) be unit-tested without a GPU. Keep the split aligned to the existing dispatch/*.rs file boundaries to minimize churn.

**Evidence:** find rdna-compute/src: gemv.rs 6121, gemm_qkv.rs 5890, attention.rs 5563, dispatch/mod.rs 5153, fused.rs 4308, deepseek4.rs 3536; 20 crates depend on the whole crate.

*Verification: confirmed*

### [Medium] monolith — `crates/hipfire-diffusion/src/lib.rs:2750-2774 (10,349-line lib.rs)`

**Observation:** hipfire-diffusion's lib.rs is a 10,349-line monolith. The crate's public error surface — DiffusionError (line 2750) and DiffusionResult (line 2774), the high-fanout result type flagged for this subsystem — is buried ~2,750 lines into the file alongside pipeline, scheduler, VAE, and DiT code. This concentrates unrelated responsibilities (error taxonomy, flow-matching schedulers, tensor pipeline, image I/O) in one compilation unit and makes the error type hard to locate or evolve.

**Recommendation:** Break lib.rs into a conventional module tree: error.rs (DiffusionError/DiffusionResult as the crate's leaf error surface), pipeline.rs, scheduler.rs, vae.rs, dit.rs, and image_io.rs. Lift the error enum to the top of the module hierarchy so downstream (server sdapi, cli) import a stable hipfire_diffusion::DiffusionError path. While splitting, confirm no model-format import/pickle/safetensors decode lives in this runtime crate — per AGENTS.md that belongs in hipfire-coexistence, not the diffusion inference path.

**Evidence:** wc -l hipfire-diffusion/src/lib.rs = 10349; DiffusionError defined at L2750, DiffusionResult at L2774 inside that single file.

*Verification: confirmed*

### [Low] module-structure — `crates/hipfire-dispatch/ vs crates/rdna-compute/src/dispatch/`

**Observation:** There is a name collision at the busiest layer boundary: the crate hipfire-dispatch (families/, model_ext/, pipeline/, tables/, traits.rs — model-family routing) and the module rdna_compute::dispatch (activation/attention/gemm/moe/rope — raw kernel-launch wrappers) both present as 'dispatch'. hipfire-dispatch depends on rdna-compute, so every arch crate imports both rdna_compute::dispatch::… and hipfire_dispatch::… in the same file, forcing readers to constantly disambiguate two unrelated 'dispatch' concepts at different abstraction levels.

**Recommendation:** Rename one side to reflect its layer. Cheapest and clearest: rename the low-level module rdna_compute::dispatch -> rdna_compute::launch (or ::kernel), since it is kernel launch/dispatch, freeing 'dispatch' for the higher-level hipfire-dispatch router. Alternatively rename the crate to hipfire-op-router. Do this as a mechanical rename during the rdna-compute split above so it lands once.

**Evidence:** rdna-compute/src/dispatch/ contains attention.rs, gemm_*.rs, moe.rs, rope.rs (kernel wrappers); hipfire-dispatch/src has families/, model_ext/, pipeline/, tables/, traits.rs (routing); arch crates depend on both (e.g. hipfire-arch-llama/Cargo.toml deps hipfire-dispatch + rdna-compute).

*Verification: adjusted — The directory/name overlap is real (crate hipfire-dispatch has families/model_ext/pipeline/tables/traits.rs; rdna-compute/src/dispatch/ has attention/gemm_*/moe/rope). But the load-bearing claim is false: ZERO occurrences of 'rdna_compute::dispatch::' anywhere in the tree — consumers use root re-exports (e.g. arch.rs:25 'use rdna_compute::{Gpu, GpuTensor}'), so readers never disambiguate two ::dispatch:: imports as claimed. Only the weak dir-name overlap survives; Low severity still fits.*

### [Low] crate-boundary — `Cargo.toml:26 and Cargo.toml:49`

**Observation:** crates/hipfire-primitives is listed twice in the workspace members array (lines 26 and 49). Cargo tolerates and de-duplicates this, so it builds, but a duplicated member entry is a concrete manifest defect that signals the 65-entry members list is hand-maintained with no lint guarding it — the same gap that produced the orphan crate below.

**Recommendation:** Delete one of the two hipfire-primitives entries. Consider adding a tiny CI check (or the `cargo-workspace-hack`/a sort-and-dedup lint) that asserts members is sorted and unique, so member-list drift is caught mechanically rather than in review.

**Evidence:** Root Cargo.toml members: 'crates/hipfire-primitives' appears at line 26 and again at line 49.

*Verification: confirmed*

### [Low] crate-boundary — `Cargo.toml:67 (crates/hipfire-hneurons)`

**Observation:** hipfire-hneurons (329 LOC, lib-only) is a workspace member (line 67) but has zero consumers: no other crate references it in any Cargo.toml and there is no `hipfire_hneurons` use anywhere in crates/*/src. It is dead weight in the dependency graph that still gets compiled by `cargo check --workspace` and included in workspace-wide operations.

**Recommendation:** Either wire hipfire-hneurons into its intended consumer (if it is WIP, note that in the crate README/AGENTS and track it), or move it out of [workspace.members] into a staging location until it has a consumer. Same treatment for any other member with no reverse dependency.

**Evidence:** grep for hipfire-hneurons / hipfire_hneurons across all crates outside its own dir returns nothing; it is a member at Cargo.toml:67 with no [[bin]].

*Verification: confirmed*

### [Low] module-structure — `crates/hipfire-arch-nemotron/Cargo.toml:9-16, crates/hipfire-arch-zaya/Cargo.toml:9-11`

**Observation:** The arch-* crates do not share a consistent dependency contract. Most (llama, qwen2, minimax, gemma3, toy) depend on only {hipfire-runtime, hipfire-dispatch, hip-bridge, rdna-compute} and reach shared types through runtime re-exports, but nemotron and zaya additionally depend directly on hipfire-mixer, hipfire-model, hipfire-primitives (nemotron also hipfire-kld). Separately, two arch crates depend on sibling arch crates (hipfire-arch-dots-ocr -> hipfire-arch-qwen2, hipfire-arch-gemma3-vl -> hipfire-arch-gemma3); those laterals are reasonable VL/OCR-on-base compositions, but combined with the direct-dep inconsistency it means arch crates are not clean interchangeable siblings.

**Recommendation:** Define one arch-crate dependency contract and apply it uniformly: either all arch crates depend directly on the shared leaves they use, or all reach them via a single documented hipfire-runtime (or a new hipfire-arch-prelude) re-export. Keep the VL/OCR->base-arch laterals but document them as the one allowed lateral pattern so the sibling model stays predictable.

**Evidence:** nemotron Cargo.toml:105-112 and zaya:166-171 add hipfire-mixer/model/primitives(/kld) that peers omit; dots-ocr->qwen2 and gemma3-vl->gemma3 are the only arch->arch edges.

*Verification: adjusted — Core claim confirmed: nemotron adds hipfire-mixer/model/primitives/kld and zaya adds hipfire-mixer/model/primitives while peer llama has none; dots-ocr->qwen2 and gemma3-vl->gemma3 laterals confirmed. But the line citations are impossible: nemotron/Cargo.toml is only 18 lines (deps at 9,10,11,16) and zaya/Cargo.toml is 17 lines (deps at 9,10,11), not 105-112/166-171. Numbers need correction; Low severity fits.*

### [Low] utils-sprawl — `crates/hipfire-primitives/src/lib.rs:18-19 (+ misc workspace hygiene)`

**Observation:** Folded minor items. (1) hipfire-primitives is mis-named: it exports only `conv` and `fwht` numeric kernels (lib.rs:18-19), not shared primitive types — the name invites future dumping and shadows the natural home for extracted GpuTensor/error types. (2) Several very thin leaves exist (hipfire-build-info 15 LOC, hipfire-hash 93, hipfire-media 340, hipfire-quant-format 193); build-info and hash are justified rebuild-isolation/shared leaves and should stay, but media (used by only gemma3-vl + serving-core) and quant-format warrant a periodic 'does this boundary still earn its keep' review. (3) The UI exclusion (Cargo.toml:69-77) is correct and well-justified — admin-ui/chat-ui/web-ui target wasm32 via leptos CSR + web-sys/wasm-bindgen, confirmed in their manifests, so keeping them out of [workspace.members] to avoid native `cargo check --workspace` breakage is the right call.

**Recommendation:** Rename hipfire-primitives -> hipfire-numeric (or -kernels) to reflect its conv/fwht content and free 'primitives' for a genuine shared-types leaf if the GpuTensor/HostProfile extractions land there. Leave the wasm UI exclusion as-is; the rationale holds. Treat the thin single-consumer leaves as a low-priority consolidation candidate, not an urgent one.

**Evidence:** hipfire-primitives/src/lib.rs exports only `pub mod conv` (L18) and `pub mod fwht` (L19); UI crates excluded with wasm rationale at Cargo.toml:69-77; hipfire-media referenced by only gemma3-vl and serving-core Cargo.tomls.

*Verification: confirmed*


## Workspace-wide unit-test audit

*Subsystem key: `test-audit` — 9 finding(s)*

**Subsystem assessment:** Workspace-wide unit-test coverage is highly uneven. The genuinely-critical pure logic that IS testable-without-a-GPU splits into three tiers: (a) well-covered — the token sampler (crates/hipfire-runtime/src/sampler.rs, 17 tests incl. NaN edge cases), the tokenizer (crates/hipfire-model/src/tokenizer.rs, 58 tests), the .hfq artifact-name grammar parser (crates/hipfire-model/src/lib.rs::parse_canonical_model_artifact_name, guarded + tested), tool_call.rs (14), and eos_filter.rs (12); (b) severely under-tested critical hot-path math — the quant codec integer pack/unpack pairs (codecs.rs, 40 fns, 0 tests), KV-cache index arithmetic (kv.rs, 2596 LOC, 0 tests), and the HFQ container binary parsers (hfq.rs, 1 test); (c) whole crates near zero — hipfire-serving-core (20,142 LOC / 11 tests, the worst LOC-per-test of any large runtime crate), redline (5,660 LOC / 0), and hipfire-train (11,314 LOC / 1). The main risks are silent-correctness regressions in codec and cache-index math (bad weights/garbage tokens, not crashes) and untested untrusted-binary parsing of .hfq files. Recommended placement: pure integer/parse/index math as in-crate #[cfg(test)] unit tests; HFQ write→read round-trips as crate-level integration tests (no GPU); model coherence/quality as hipfire-eval batteries behind coherence-gate-dflash.sh; GPU dispatch correctness stays in hipfire-eval, not unit tests. There are ~7,600 unwrap() calls workspace-wide (307 in hipfire-server alone) but the sampled parsers bounds-check before their try_into().unwrap(), so the acute gap is missing tests around that logic rather than raw unwrap density.

### [High] missing-tests — `crates/hipfire-quantize/src/codecs.rs:1-2243`

**Observation:** codecs.rs is 2243 LOC of quant codec integer math (40 pub fns: quantize_hfq2g128, quantize_oq4g256, quantize_oq8g256, quantize_mq3g256_lloyd, symmetric_clipsearch, etc.) with ZERO in-file #[test] functions. Matching decode functions live in the same module (dequant_oq4g256 L684, dequant_oq8g256 L886, dequant_mq4g256 L915, dequant_hfp4g32_row L1299), so encode/decode round-trips are trivial to assert, yet none exist. The only codec-adjacent test in the crate (main.rs::codec_hashes L13037) checks dtype labels and residual reader/writer flags, not the pack/unpack numeric pairs. Risk is silent correctness: a wrong scale formula, off-by-one group stride, or mismatched nibble order produces garbage weights that surface only as model-quality regression, never a crash. The 128 float-to-int as-casts are individually clamp/mask-guarded at pack time (e.g. L170 q.min(3)), so this is a coverage gap, not UB.

**Recommendation:** Add a #[cfg(test)] mod with per-codec round-trip and known-vector tests; sibling qtip.rs already models this (qtip3_pack_roundtrip_matches_decode). 8-bit codecs should be near-lossless. Ready-to-paste (signs must match between encode/decode; encode passes (signs1,signs2), decode swaps internally):

#[cfg(test)]
mod codec_tests {
    use super::*;
    #[test]
    fn oq8g256_roundtrip_is_near_lossless() {
        let signs = vec![1.0f32; 256]; // identity FWHT sign table
        let data: Vec<f32> = (0..512).map(|i| ((i as f32) * 0.013).sin()).collect();
        let packed = quantize_oq8g256(&data, &signs, &signs);
        assert_eq!(packed.len(), data.len().div_ceil(256) * 258);
        let deq = dequant_oq8g256(&packed, data.len(), &signs, &signs);
        let mse: f32 = data.iter().zip(&deq).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / data.len() as f32;
        assert!(mse < 1e-4, "oq8 mse too high: {mse}");
    }
}

**Evidence:** wc -l codecs.rs=2243; grep -c '#[test]' codecs.rs=0; 40 pub fns; dequant pairs at L684/L886/L915/L1299; encode L730 cpu_fwht_256(_,signs1,signs2), decode L901 swaps to (signs2,signs1)

*Verification: confirmed*

### [High] missing-tests — `crates/hipfire-runtime/src/kv.rs:1-2596`

**Observation:** kv.rs (KV-cache) is 2596 LOC with ZERO #[test] functions despite being dense with hot-path index/size arithmetic: kv_dim = n_kv_heads * head_dim, cache_size = max_seq_len * kv_dim, packed-element math like (max_seq_len*kv_dim+3)/4 and (physical_cap*n_kv_heads*k_bph+3)/4, plus block-table offsets (data_offset = cumulative - base). It also exposes pure helpers that need no GPU: is_boundary(kv_ordinal) L100, kvarn_k_record_bytes(head_dim) L1024, and gen_givens_angles(seed, n_blocks) L661. An off-by-one in any of the ceil-div packing or offset formulas corrupts cache reads silently. The file's many assert!(head_dim % 32 == 0)/capacity asserts are runtime guards, not tests.

**Recommendation:** Extract the size/offset formulas into small free functions (fn kv_cache_size(n_kv_heads, head_dim, max_seq)->usize, fn packed_u8_elems(elems)->usize) and unit-test them with hand-computed expected values plus boundary cases (max_seq_len=0, non-multiple head_dim). Directly test the existing pure helpers: assert kvarn_k_record_bytes and is_boundary at group boundaries, and gen_givens_angles determinism for a fixed seed (same seed => identical vectors, correct length n_blocks). These are pure and belong in an in-crate #[cfg(test)] mod, runnable under ./tests/no-gpu-ci.sh.

**Evidence:** wc -l kv.rs=2596; grep -c '#[test]'=0; index math L113-114, L562, L615, L710, L781; pure helpers is_boundary L100, gen_givens_angles L661, kvarn_k_record_bytes L1024

*Verification: confirmed*

### [Medium] test-structure — `crates/hipfire-serving-core/src/generate.rs, session.rs, generate_arch.rs, qwen35_prefill.rs, qwen35_decode.rs`

**Observation:** hipfire-serving-core is 20,142 LOC with only 11 #[test] functions (~1,831 LOC/test — the worst ratio of any large runtime crate; for comparison hipfire-server is 153, hipfire-runtime 519). All 11 tests live in generate_vl.rs (7), output_filter.rs (2), load.rs (2). The core orchestration files carry zero tests: generate.rs (3678 LOC), generate_arch.rs (2988), session.rs (2435), qwen35_prefill.rs (1933), qwen35_decode.rs (1350). Much of this is GPU-coupled, but the crate holds token-budget/stop-condition/session-state bookkeeping and epoch allocation (session.rs::next_qwen35_state_allocation_epoch L99) that are pure and currently unverified.

**Recommendation:** Separate GPU-driving code from decision logic: pull stop-sequence detection, max-token/budget accounting, and session token-window trimming into pure functions and cover them with in-crate unit tests (edge cases: empty prompt, stop string spanning two decode steps, budget exactly hit). Whole-generation-loop correctness (does a model still produce coherent text) belongs in hipfire-eval batteries gated by coherence-gate-dflash.sh, not unit tests. This gives the crate a testable seam without a GPU in CI.

**Evidence:** crate 20142 LOC / 11 tests; generate.rs 3678 / 0, generate_arch.rs 2988 / 0, session.rs 2435 / 0, qwen35_prefill.rs 1933 / 0, qwen35_decode.rs 1350 / 0; tests only in generate_vl.rs(7)/output_filter.rs(2)/load.rs(2)

*Verification: adjusted — Facts exact: 11 tests total (generate_vl 7, output_filter 2, load 2); generate.rs 3678/0, generate_arch 2988/0, session 2435/0, qwen35_prefill 1933/0, qwen35_decode 1350/0; next_qwen35_state_allocation_epoch L99; ratios verified (server 153, runtime 519 LOC/test whole-crate, serving-core ~1831). But this is a test-ratio/structure finding on a crate the finding itself calls 'much GPU-coupled'; the pure testable surface is a modest subset needing extraction first. That fits Medium (scheduled refactor), not High (active correctness/maintainability blocker).*

### [Medium] missing-tests — `crates/hipfire-runtime/src/hfq.rs:89 and :1010`

**Observation:** hfq.rs (2526 LOC) parses untrusted on-disk .hfq container binaries and has only 1 #[test]. Two near-identical hand-rolled binary index parsers exist — parse_hfqm_index (L89) and parse_hfqm_meta_index (L1010) — duplicating the same json_blob_end + tensor-count + per-entry name_len/n_dims/shape/data_size loop. They are defensively coded (bounds-check before each mmap[pos..pos+N].try_into().unwrap(), e.g. L130/L138/L150), but there is no test for round-trip (write_hfqm_package_from_files L232 -> HfqPackage::open L191) and no malformed-input test proving the bounds checks reject truncated/oversized-offset files instead of panicking. Duplication also means a fix to one parser can silently skip the other.

**Recommendation:** Add a crate-level integration test (no GPU) that writes a small package with write_hfqm_package_from_files, reopens it with HfqPackage::open, and asserts entries()/metadata_json/blob_data round-trip. Add malformed-input unit tests feeding parse_hfqm_index truncated buffers and offsets where metadata_offset > data_offset > len, asserting Err (not panic) for each guard. Separately, deduplicate the two parsers into one shared routine parameterized over the entry type to remove the divergence risk. Place these in crates/hipfire-runtime/tests/.

**Evidence:** wc -l hfq.rs=2526, grep -c '#[test]'=1; duplicate parsers at L89 and L1010 with identical loop shape; guarded try_into().unwrap() at L117/L136/L158/L163/L200-204; writer at L232

*Verification: confirmed*

### [Medium] missing-tests — `crates/redline/src/pm4.rs and crates/redline/src (whole crate)`

**Observation:** The redline crate (5,660 LOC, direct-KMD GPU engine: pm4/kfd/drm/queue/hsaco) has ZERO tests. pm4.rs builds PM4 command packets by pure dword bit-packing — Pm4Builder::pkt3 emits (3<<30)|(opcode<<8)|(count-1) at L138, set_sh_reg L142, dispatch_direct L149 — which is fully unit-testable without hardware by asserting the emitted dwords slice. The count-1 subtraction underflows if a caller ever passes count==0 (debug panic; release wrap to 0xFFFFFFFF -> malformed header). Per the machine's LDS/dispatch hazard notes, a malformed PM4 dispatch can wedge the GPU into a sticky reset, so silent header corruption here is unusually costly.

**Recommendation:** Assert the exact dword layout of the packet builders and guard pkt3 against count==0 (debug_assert!(count > 0)). Ready-to-paste:

#[cfg(test)]
mod pm4_tests {
    use super::*;
    #[test]
    fn set_sh_reg_emits_pkt3_header_and_payload() {
        let mut b = Pm4Builder::new();
        b.set_sh_reg(0x2C, 0xDEAD_BEEF);
        assert_eq!(b.dwords.len(), 3);
        assert_eq!(b.dwords[0], (3 << 30) | (PKT3_SET_SH_REG << 8) | 1);
        assert_eq!(b.dwords[1], 0x2C);
        assert_eq!(b.dwords[2], 0xDEAD_BEEF);
    }
    #[test]
    fn dispatch_direct_sets_compute_enable_initiator() {
        let mut b = Pm4Builder::new();
        b.dispatch_direct(4, 2, 1);
        assert_eq!(b.dwords[0], (3 << 30) | (PKT3_DISPATCH_DIRECT << 8) | 3);
        assert_eq!(&b.dwords[1..], &[4, 2, 1, 1]);
    }
}

**Evidence:** crate 5660 LOC, 0 #[test]; pm4.rs pkt3 L138 '(3<<30)|(opcode<<8)|(count-1)', set_sh_reg L142, dispatch_direct L149; AGENTS.local.md notes malformed dispatch -> MES hang -> sticky 719 reset

*Verification: confirmed*

### [Medium] test-structure — `crates/hipfire-diffusion/src/tests.rs vs src/lib.rs, src/gpu_ops.rs, src/tokenizer.rs`

**Observation:** hipfire-diffusion reports 226 #[test], but 217 are crammed into a single src/tests.rs, while the code they cover is not co-located: lib.rs is a 10,349-LOC monolith with 0 inline tests, gpu_ops.rs (3481 LOC) has 0, and the CLIP/BPE tokenizer (src/tokenizer.rs, 177 LOC) has 0. Only vae.rs (4), transformer.rs (3), scheduler.rs (2) carry inline tests. A 10k-line lib.rs with all its verification in a separate file is hard to keep honest as functions move, and the pure tokenizer (byte-pair merges, special-token handling) — exactly the edge-case-prone logic — is entirely unverified.

**Recommendation:** Co-locate unit tests with the units under test via #[cfg(test)] mod at the bottom of each module, especially tokenizer.rs (round-trip encode/decode, unknown-byte fallback, empty string, unicode) and the pure numeric helpers currently buried in lib.rs. Keep src/tests.rs for cross-module/pipeline scenarios only. This also nudges lib.rs toward being split, since inline tests force smaller, nameable units.

**Evidence:** diffusion 226 tests: tests.rs 217, vae.rs 4, transformer.rs 3, scheduler.rs 2; lib.rs 10349 LOC / 0 inline, gpu_ops.rs 3481 / 0, tokenizer.rs 177 / 0

*Verification: confirmed*

### [Medium] missing-tests — `crates/hipfire-train/src (71 files, 11,314 LOC) — only oqplus_quant.rs tested`

**Observation:** hipfire-train has 1 #[test] across 71 files / 11,314 LOC (the single tested file is oqplus_quant.rs). Training is offline tooling (not the inference hot path, so lower correctness urgency), but OQ+/OQ++ quantization math it produces — clip-search scaling and LDLQ/Hessian error-feedback — directly determines the numeric quality of shipped .hfq artifacts. That pure linear-algebra/integer logic is a strong unit-test candidate and is almost entirely uncovered.

**Recommendation:** Prioritize deterministic unit tests on the pure quant/error-feedback kernels in this crate: seeded-input round-trips, monotonicity of clip-search objective, and equivalence of the error-feedback accumulation against a slow reference implementation on small matrices. Leave end-to-end training-run validation to hipfire-eval/tinyquant batteries. Since this is offline tooling, in-crate #[cfg(test)] units under ./tests/no-gpu-ci.sh are the right home.

**Evidence:** grep -rln '#[test]' crates/hipfire-train/src => only oqplus_quant.rs; crate 11314 LOC / 71 .rs files / 1 test

*Verification: confirmed*

### [Low] missing-tests — `crates/rdna-compute/src/dispatch/{gemv.rs,gemm_qkv.rs,attention.rs,fused.rs}`

**Observation:** rdna-compute is 92,859 LOC with 52 #[test] (~1,786 LOC/test). The bulk is legitimately GPU-bound and cannot execute in CI, but the largest dispatch files are entirely test-free: gemv.rs (6121 LOC / 0), gemm_qkv.rs (5890 / 0), attention.rs (5563 / 0), fused.rs (4308 / 0). These files compute launch geometry (grid/block dims, tile counts, workgroup rounding, shared-memory sizing) as pure integer arithmetic that is portability-sensitive across RDNA2/3/4 and could regress silently on a new arch overlay.

**Recommendation:** Extract the pure launch-parameter math (tiles = ceil_div(n, tile), workgroup/wave rounding, LDS byte budgeting) into free functions and unit-test them with per-arch expected values, so an overlay change for gfx12/gfx1151 that breaks grid sizing fails in CI rather than on hardware. Keep actual kernel numeric correctness in hipfire-eval + coherence-gate-dflash.sh. Do not attempt to unit-test the dispatch calls themselves.

**Evidence:** crate 92859 LOC / 52 tests; gemv.rs 6121/0, gemm_qkv.rs 5890/0, attention.rs 5563/0, fused.rs 4308/0 (grep -c '#[test]' each = 0)

*Verification: confirmed*

### [Low] missing-tests — `workspace roundup (multiple crates)`

**Observation:** Remaining zero/near-zero test modules holding testable pure logic, folded here: hsa-bridge (2988 LOC / 0 tests — HSA queue/signal plumbing, some pure packet math); crates/hipfire-arch-qwen35/src/mtp_head.rs (2683 LOC / 0 tests, contains pure sanity_check_2d_shape L1112); crates/hipfire-arch-llama/src (564 LOC / 0); crates/hipfire-coexistence/src (209 LOC / 0 — the import/export tooling crate, where format-conversion parsing edge cases matter); crates/hipfire-runtime/src/env_docs.rs (4259 LOC / 0, mostly static docs — low value). Also note ~7,600 unwrap() workspace-wide (307 in hipfire-server, 183 in hipfire-runtime); the parsers sampled here bounds-check before unwrap, so this is a latent rather than confirmed panic surface, but it argues for targeted #[should_panic]/Result tests on any unwrap that consumes external/HTTP/file input.

**Recommendation:** Triage by input-source: prioritize tests for modules that parse external bytes (hipfire-coexistence conversion parsers, hsa-bridge packet builders) over static/doc modules (env_docs.rs). Add malformed-input unit tests around the highest-fanout unwrap sites in hipfire-server request handling. Do not chase raw unwrap count; convert unwrap-on-external-input to Result at the boundaries you add tests for. Keep GPU/hardware modules out of unit scope and validate via hipfire-eval.

**Evidence:** hsa-bridge 2988/0, mtp_head.rs 2683/0 (sanity_check_2d_shape L1112), arch-llama 564/0, coexistence 209/0, env_docs.rs 4259/0; unwrap counts: server 307, runtime 183, daemon 52

*Verification: confirmed*


## Utility sprawl & API hygiene

*Subsystem key: `utils-sprawl` — 11 finding(s)*

**Subsystem assessment:** The workspace has no files literally named utils.rs/helpers.rs/common.rs, but the equivalent sprawl is concentrated in a handful of giant crate-root lib.rs files that flatten many unrelated responsibilities into one namespace. The two worst offenders are hipfire-diffusion/src/lib.rs (10,349 LOC of non-test code, tests split out) and hipfire-eval/src/lib.rs (8,117 LOC, ~6,085 of which are inline tests). The diffusion lib.rs is both a monolith and a crate-boundary violation: ~1,900 lines of offline diffusers/safetensors/pickle import and a MiniZip reader live in the runtime crate that hipfire-server links, directly contradicting the AGENTS.md rule that all import/conversion tooling belongs in hipfire-coexistence. Secondary grab-bags: hipfire-model (3,332 LOC spanning 5-6 concerns including misplaced accelerator-inventory hardware types), rdna-compute dispatch/misc.rs (a 21-method "misc" Gpu impl in an otherwise cleanly partitioned dispatch tree), and OpenAI wire-format code scattered across hipfire-prompt + hipfire-generate + hipfire-server with no shared adapter. Health is maintainability-risk rather than correctness-risk, but the diffusion monolith and coexistence boundary breach are structural and worth prioritizing; much of the buried logic (CPU tensor math, artifact-name parsing, resolve_*_bin) is pure and trivially unit-testable if extracted.

### [High] utils-sprawl — `crates/hipfire-diffusion/src/lib.rs:1-10349`

**Observation:** The crate root is a 10,349-line non-test monolith (tests already split into a separate 13,491-line tests.rs) that aggregates at least nine unrelated responsibility clusters and flattens them into one namespace via five `pub use ...::*` wildcards (scheduler L1259, layers L5623, unet L5626, vae L5629, tokenizer L7076). Distinct clusters: HFQ metadata types+parsing (L48-146, parse_diffusion_metadata L7111), the DiffusionPipeline orchestrator (impl L2788-4729, 29 methods, ~1,940 lines), a full CLIP text-encoder transformer (ClipTextEncoder L6100/6132, ClipEncoderLayer L6416), pure CPU tensor math (L6799-7075), PNG/base64 encode (L5714), image resize (L7963-8444), and a MiniZip+diffusers importer (L8444-10349). The crate already has topical submodules (transformer.rs, gpu_ops.rs, vae.rs, unet.rs, scheduler.rs, layers.rs), so the lib.rs remainder is pure leftover sprawl.

**Recommendation:** Reduce lib.rs to type/trait declarations and a small facade, and move each cluster into its own module: `metadata.rs` (HFQ parse/inspect), `pipeline.rs` (DiffusionPipeline), `clip.rs` (text encoder), `cpu_ops.rs` (conv/norm/silu/resize), `png.rs`. Keep the wildcard re-exports only for genuinely public leaf types; prefer `pub use scheduler::{SchedulerConfig, ...}` over glob so the crate's public surface is intentional rather than incidental.

**Evidence:** wc -l lib.rs = 10349; DiffusionPipeline impl L2788-4729 has 29 fns; 5 `pub use x::*` at L1259/5623/5626/5629/7076; tests.rs is a separate 13491-line file

*Verification: confirmed*

### [High] crate-boundary — `crates/hipfire-diffusion/src/lib.rs:8444-10349`

**Observation:** Roughly 1,900 lines of offline format-conversion tooling live inside the runtime crate: import_diffusers_to_hfq (L8452), import_single_file_checkpoint_to_hfq (L8680), parse_safetensors_state_dict / parse_sharded_safetensors_state_dict (L9649), a hand-rolled MiniZipArchive pickle/zip reader (L10233), and model_index.json/config.json readers. This crate is a dependency of hipfire-server (crates/hipfire-server/Cargo.toml), so all of this import/conversion code is compiled into the inference server binary. AGENTS.md is explicit: import/export, format conversion, and interop tooling belong in hipfire-coexistence, and coexistence tooling must be kept out of the inference binaries. hipfire-coexistence already exists and self-documents as 'offline import/export/conversion/interop tooling' (crates/hipfire-coexistence/src/main.rs:2), so this is a direct boundary breach, not a gray area.

**Recommendation:** Move the entire importer block (import_diffusers_to_hfq, import_single_file_checkpoint_to_hfq, safetensors/sharded parsers, MiniZipArchive, JSON config readers, DiffusersImportOptions) into hipfire-coexistence (or a dedicated hipfire-diffusion-import tooling crate) and expose it via the coexistence CLI. hipfire-diffusion should consume only already-produced .hfq artifacts. The CLI call site (crates/hipfire-cli/src/commands/diffusion.rs:421) should route through the tooling crate, not the runtime crate, so hipfire-server stops linking pickle/zip parsing.

**Evidence:** import_diffusers_to_hfq L8452, MiniZipArchive::open L10233; hipfire-server/Cargo.toml depends on hipfire-diffusion; coexistence main.rs L2 doc-comment claims this exact responsibility; called from cli/commands/diffusion.rs:421

*Verification: confirmed*

### [Medium] utils-sprawl — `crates/rdna-compute/src/dispatch/misc.rs:12-1191`

**Observation:** misc.rs is a 1,191-line file containing a single `impl Gpu` block (L12) with 21 methods spanning at least five unrelated concerns: rotations (givens_rotate L21, paro4g128_rotate L171, paro4g128_swiglu_rotate L242, paro4g128t_* L318/388), gather/scatter (rq_gather_f32 L464, rq_scatter_add_f32 L506, scatter_session_last_logits_f32 L547), layout shuffles (qkv_split_interleaved_f32 L602, deinterleave_f32 L671/728, transpose_f32 L950), dtype casts (cast_f32_to_bf16 L986, cast_f32_to_f16 L1146), attention padding (attn_split_pad_f16kv L1028, attn_unpad L1100), and losses (cross_entropy_loss L864, cross_entropy_train L905). This is a literal 'misc' grab-bag even though the sibling dispatch files (norm.rs, activation.rs, gemm_base.rs, embedding.rs, quant.rs, conv1d.rs, mamba2.rs) already establish a clean by-concern partition of the same Gpu type.

**Recommendation:** Dissolve misc.rs into topical partitioned-impl files following the existing convention: rotation.rs (givens/paro4g128*), gather_scatter.rs (rq_gather/scatter, scatter_session_last_logits), layout.rs (deinterleave/qkv_split/transpose), cast.rs (cast_f32_to_bf16/f16), attn_pad.rs (attn_split_pad/attn_unpad), and loss.rs (cross_entropy_*). Rust lets `impl Gpu` be split across files freely, so no API change is needed. Also rename gemm_misc.rs, the second 'misc' file, to a concern-specific name.

**Evidence:** misc.rs = 1191 lines, single `impl Gpu` at L12 with 21 fns; sibling files embedding.rs/gated.rs/norm.rs/quant.rs each hold `impl Gpu` partitions

*Verification: confirmed*

### [Medium] utils-sprawl — `crates/hipfire-model/src/lib.rs:37-640`

**Observation:** The model crate root mixes five to six unrelated responsibility clusters in 3,332 lines: HFQ container metadata (TensorInfo L37, HfqMetadata L77), model-worker identity/protocol (ModelWorkerKey L97, parse_model_worker_id L127, same_model_worker_key L281), architecture-family detection (model_arch_family L202, is_qwen35_dense_arch_id L222), hardware/accelerator inventory (AcceleratorInventory L286, AcceleratorDeviceInfo L308, accelerator_inventory_json L373), the ModelSource abstraction (trait L405, detect_model_artifact_format L450, open_model_source_with L473), tokenizer metadata (tokenizer_signature L497, hfq_chat_template L547), and artifact-name string parsing (normalize_tag_stem L566, is_role_sidecar_name L571, quant_preference_rank L581, model_display_name L618). The accelerator-inventory hardware types are especially out of place given hipfire-sysinfo already owns GPU telemetry (crates/hipfire-sysinfo/src/gpu.rs, gpu_metrics.rs).

**Recommendation:** Split into modules by concern: hfq.rs (container metadata), worker.rs (ModelWorkerKey/identity), arch.rs (family detection), source.rs (ModelSource/format detection), artifact_name.rs (the AGENTS.md canonical-name parsers: normalize_tag_stem/is_role_sidecar_name/quant_preference_rank/model_display_name). Relocate AcceleratorInventory/AcceleratorDeviceInfo to hipfire-sysinfo (or a shared inventory crate) so hardware descriptors live with the hardware-probing code rather than in the model crate.

**Evidence:** wc -l = 3332; AcceleratorInventory L286 vs sysinfo/gpu.rs read_gpu_telemetry L22; artifact-name parsers L566-618; ModelWorkerKey L97 alongside HfqMetadata L77 and ModelSource L405

*Verification: confirmed*

### [Medium] duplication — `crates/hipfire-eval/src/lib.rs:1148-1255`

**Observation:** Seven near-identical `resolve_*_bin` helpers (resolve_dflash_spec_demo_bin L1148, resolve_bench_qwen35_speed_bin L1163, resolve_pflash_niah_bench_bin L1180, resolve_run_example_bin L1195, resolve_collect_artifacts_bin L1210, resolve_perplexity_bin L1225, resolve_host_profile_bin L1240) are copy-paste clones. Each is the same ~15-line body: check an env var, canonicalize to a PathBuf, then probe target/release/examples and target/debug/examples for a fixed binary name. The only per-function variance is the env-var string and the example name; bodies were verified byte-identical in structure across at least four of them.

**Recommendation:** Collapse into one parametrized helper, e.g. `fn resolve_example_bin(env_var: &str, example: &str) -> Option<PathBuf>` that does the env probe then `newest_existing_path` over the release/debug candidates. Replace the seven functions with call sites like `resolve_example_bin("HIPFIRE_BENCH_QWEN35_SPEED_BIN", "bench_qwen35_speed")`. This removes ~90 lines and makes the release/debug search order impossible to get inconsistent (some current copies use `.find()` while others use `newest_existing_path`).

**Evidence:** grep -c 'fn resolve_.*_bin' lib.rs = 7 at L1148-1255; bodies at L1163-1210 differ only in the env-var literal and example filename

*Verification: confirmed*

### [Medium] abstraction — `crates/hipfire-prompt/src/lib.rs:849-936, crates/hipfire-generate/src/lib.rs:284-349`

**Observation:** OpenAI chat wire-format adaptation is split across three crates with no shared owner. Request-side conversion lives in hipfire-prompt (openai_chat_role_to_prompt_role L849, openai_chat_content_to_text L859, openai_chat_message_to_prompt_message L895, openai_chat_messages_to_prompt_messages L907, openai_chat_last_user_prompt L917); response-side serialization lives in hipfire-generate (openai_chat_completion_response_json L284, openai_chat_completion_token_chunk_json L310, openai_chat_completion_done_chunk_json L327, openai_finish_reason L344); and hipfire-server/src/routes/chat.rs consumes both. The OpenAI JSON schema is thus a de-facto cross-cutting protocol whose knowledge is scattered, making schema drift (finish_reason values, chunk shape) easy to introduce on only one side.

**Recommendation:** Introduce a single OpenAI-adapter module or small crate (e.g. hipfire-openai-compat, or a `openai` module under serving-core) that owns both request parsing and response/stream serialization and the shared DTOs. hipfire-prompt should depend on it for input mapping and hipfire-generate for output mapping, so the wire schema has exactly one source of truth. This also aligns with the AGENTS.md preference for keeping protocol/interop concerns out of the lean generation/prompt hot-path types.

**Evidence:** openai_chat_* input fns at prompt/lib.rs L849-936; openai_chat_completion_* output fns at generate/lib.rs L284-349; grep shows the same family also consumed in hipfire-server/src/routes/chat.rs

*Verification: confirmed*

### [Medium] coupling — `crates/hipfire-generate/src/lib.rs:407-582`

**Observation:** The generic hipfire-generate crate hardcodes Qwen3.5-specific session, checkpoint, and prefix-hash logic directly at the crate root: compute_qwen35_prefix_hash (L407), qwen35_checkpoint_session_id (L515), qwen35_boundary_checkpoint_session_id (L523), qwen35_prefill_checkpoint_session_id (L534), qwen35_prefill_checkpoint_boundary_kind (L550), Qwen35SemanticBoundaryCheckpoint (L487), Qwen35DecodeTokenOutcome (L575), Qwen35DecodeBatchStepResult (L582). A crate whose job is arch-agnostic generation should not name one model family in its public API; this couples the generic decode/prefill path to Qwen3.5 and invites parallel per-arch copies as other architectures (LFM2, Minimax, Nemotron) grow the same needs.

**Recommendation:** Hoist the Qwen35 session/checkpoint/prefix-hash behavior behind an arch-parameterized trait (e.g. `trait PrefillCheckpointPolicy { fn session_id(...); fn boundary_kind(...); fn prefix_hash(...); }`) implemented per-arch, or move the Qwen35 concretes into hipfire-arch-qwen35. hipfire-generate then takes a `&dyn PrefillCheckpointPolicy`, keeping the generic path free of family names and giving other arch crates a clear extension point.

**Evidence:** compute_qwen35_prefix_hash L407, qwen35_prefill_checkpoint_session_id L534, Qwen35DecodeBatchStepResult L582 all at the root of a crate named hipfire-generate

*Verification: confirmed*

### [Medium] utils-sprawl — `crates/hipfire-diffusion/src/lib.rs:6799-7075`

**Observation:** Pure CPU tensor-math primitives are buried in the 10k-line diffusion monolith: silu (L6799), conv2d_nchw (L6857), conv2d_nchw_with_stride (L6866), group_norm_nchw (L6948), upsample_nearest2d_nchw (L7008). These are GPU-free, deterministic numeric functions that are trivially unit-testable in isolation, yet they sit next to pipeline orchestration and importer code. Meanwhile hipfire-cpu (a dedicated 894-line CPU-ops crate) is the natural home for exactly this. The mislocation both worsens the sprawl and hides easily-testable logic inside a file that otherwise requires heavy setup to exercise.

**Recommendation:** Move these primitives into hipfire-cpu (or a diffusion `cpu_ops` submodule) and give each a focused unit test with a small hand-computed golden (e.g. a 1x1x3x3 conv2d, a 2-group group_norm) so the reference CPU path is independently verified. This directly improves testability of pure logic without a GPU and shrinks the crate-root surface.

**Evidence:** conv2d_nchw L6857, group_norm_nchw L6948, upsample_nearest2d_nchw L7008 are `pub fn`s over CpuTensor; hipfire-cpu/src/lib.rs is a 894-line CPU-ops crate that already exists as their home

*Verification: confirmed*

### [Medium] utils-sprawl — `crates/hipfire-prompt/src/lib.rs:53-1253`

**Observation:** The prompt crate root (2,378 lines) is a grab-bag mixing five loosely-related concerns: a text-normalization cluster (normalize_prompt_text L53 plus needs_newline_collapse/needs_nbsp_replace/collapse_newline_runs/replace_nbsp_with_space/strip_trailing_line_ws L80-181), chat-template resolution and framing (resolve_chat_template L206, ChatFrame L384, ChatScaffold L483, JinjaChatFrame L612), a generic JSON utility (canonical_json L787, write_canonical_json L793), content hashing (assistant_turn_fingerprint L833), the OpenAI input adapter (L849-936, see separate finding), and HF/Jinja templating (hf_tojson L985, build_cached_history_jinja L1253). canonical_json in particular is a general-purpose serializer that has nothing model-prompt-specific about it.

**Recommendation:** Split into modules: text_normalize.rs (the needs_*/collapse/strip helpers), template.rs (resolve/frames/scaffold/jinja), and move canonical_json + assistant_turn_fingerprint to a shared serialization/hashing home (hipfire-hash already exists for hashing). Keep lib.rs as the re-export facade. This clarifies which functions are prompt-domain versus generic utilities.

**Evidence:** canonical_json L787 and write_canonical_json L793 are generic JSON; text-munging cluster L53-181; assistant_turn_fingerprint L833; a dedicated hipfire-hash crate exists for the hashing piece

*Verification: confirmed*

### [Low] test-structure — `crates/hipfire-eval/src/lib.rs:2031-8117`

**Observation:** hipfire-eval/src/lib.rs is 8,117 lines but ~6,085 of them are a single inline `#[cfg(test)] mod tests` (starts L2031/2032, runs to EOF). Coverage is genuinely strong here (dozens of CLI-parsing and executor-planning tests), so this is a structural-hygiene note, not a coverage gap: a 6k-line inline test module makes the source file appear far larger than its ~2,030 lines of production code and slows navigation/incremental compile.

**Recommendation:** Move the test module out of lib.rs into `#[path]` submodule files or, where the tests only exercise the public API (parse_args_from, run planning), into `tests/` integration files. Group by theme (cli_parsing, executor_planning, dataset_resolution) mirroring the production modules they cover.

**Evidence:** #[cfg(test)] at L2031, mod tests at L2032, file EOF at L8117 => ~6085 test lines; production code (config enums, run_eval, EvalContext, resolve_*_bin) ends around L2030

*Verification: confirmed*

### [Low] utils-sprawl — `crates/hipfire-eval/src/config.rs; crates/hipfire-diffusion/src/lib.rs; crates/hipfire-state/src/lib.rs; crates/rdna-compute/src/dispatch/gemm_misc.rs`

**Observation:** Remaining lower-severity sprawl/hygiene items verified but not warranting separate findings: (1) eval config has a split-brain layout -- the config *types* (EvalConfig, EvalTier, BatteryId, SuiteId enums) are defined in lib.rs L81-630 while their *parsing and defaults* (parse_args_from L16, usage L351, default_batteries L479, default_suites L513) live in config.rs, so one logical concern spans two files by an unclear rule. (2) diffusion re-exports four whole submodules with glob `pub use` (scheduler/layers/unet/vae/tokenizer at L1259/5623/5626/5629/7076), making the public surface incidental. (3) hipfire-state accumulates per-arch hardcoded label lists (qwen35_kv_deltanet_state_kind_labels L380, minimax_state_kind_labels L387, lfm2_state_kind_labels L395, nemotron_h_state_kind_labels L403) that would be better as a data-driven registry keyed by arch. (4) rdna-compute has a second 'misc' file, gemm_misc.rs, alongside dispatch/misc.rs.

**Recommendation:** Consolidate eval config types and their parsers into one config module; replace diffusion glob re-exports with explicit named re-exports; drive the state arch-label lists from a table/registry instead of one function per arch; rename gemm_misc.rs to a concern-specific name. All are scheduled-cleanup hygiene, not correctness risks.

**Evidence:** eval config.rs parse_args_from L16 / usage L351 / default_batteries L479 vs enums in lib.rs L81-630; diffusion `pub use x::*` at L1259/5623/5626/5629/7076; state per-arch label fns L380-403; gemm_misc.rs present in dispatch/

*Verification: confirmed*


## Refuted findings (dropped)

- `rdna-rest-bridges` / `crates/hip-bridge/src/ffi.rs:1385-1411` — Location and the 1472/1024 numbers are correct, but the core heap-overflow claim is disproven. get_arch loads the bare string "hipGetDeviceProperties" via dlsym (ffi.rs:540), which resolves the ELF versioned-default symbol hipGetDeviceProperties@@hip_4.2 — a 5-byte thunk to an internal 4.2-ABI implementation, NOT hipGetDevicePropertiesR0600. I empirically dlopen'd libamdhip64.so.7 on this box, put a 0xAB canary in an 8KB buffer, and called the symbol: it wrote exactly 792 bytes (last modified index 791), matching sizeof(hipDeviceProp_tR0000)=792 from hip_deprecated.h, and placed gcnArchName ('gfx1103') at offset 396 — both inside the 1024 buffer with 232 bytes headroom. The 1472-byte R0600 layout is only reached via the explicit R0600 symbol or the header macro, neither of which this code uses. The comment '1024 is safe' is empirically correct; High severity for a nonexistent overflow is not defensible.


---

# Follow-up round (round 2)

Four scopes chosen by the completeness critic after round 1. Same reviewer→verifier method.


## hipfire-daemon-adapter (client IPC crate)

*Subsystem key: `daemon-adapter` — 5 finding(s)*

**Subsystem assessment:** The adapter (crates/hipfire-daemon-adapter/src/lib.rs, 1387 LOC) is a well-typed async IPC client that correctly consumes hipfire-daemon-protocol's DaemonRequest/DaemonResponse enums (import at L17-21) for both serialize and deserialize. The critic's headline suspicion — that the adapter is a THIRD hand-maintained copy of the wire surface — is REFUTED: the adapter re-declares nothing. The real duplication lives on the daemon SERVER side: hipfire-daemon/src/main.rs never references the typed protocol enums at all (zero DaemonRequest/DaemonResponse mentions), instead hand-parsing serde_json::Value and dispatching ~34 "type"-string match arms, and hand-building every response with serde_json::json!. So there are two hand-kept representations of one wire protocol (typed enum vs Value-parsing), agreeing only by convention — a real drift/compatibility risk. The critic's raw counts are both off: DaemonEngine has ~26 pub + 4 private = ~30 async methods (not ~72), and the crate has 11 test fns (not 4) — but a genuine coverage gap remains (all steer_*/lora_* families untested). The steer_*/lora_* ack-wrappers and the payload-recv-loop methods are indeed near-identical and collapsible. The lock-resolution code (L665-1008) is COMPLIANT with the hipfire-lock invariant: it uses hipfire_lock::FlockGuard/try_lock/lock_blocking/probe on per-resource .lock files with holder lines written under the held flock — no sentinel files, no pidfile-liveness, no create_dir mutex, no alternate lockfile.

### [High] duplication — `crates/hipfire-daemon/src/main.rs:2630-2665 vs crates/hipfire-daemon-adapter/src/lib.rs:17-21 vs crates/hipfire-daemon-protocol/src/lib.rs:205-255`

**Observation:** The daemon request/response wire surface is defined once as typed enums in hipfire-daemon-protocol (39 request+response variants) and consumed by the adapter via serde. But the daemon SERVER never uses those enums: main.rs deserializes into a bare serde_json::Value, dispatches on msg.get("type").as_str() across ~34 hand-written string match arms, and hand-builds every response with serde_json::json!({"type": ...}). This is a second, drift-prone hand-maintained copy of the same protocol: adding a variant to the protocol enum updates the adapter automatically but silently leaves the daemon parser stale. The hand-rolled error path also formats JSON via a raw string (r#"{{\"type\":\"error\",\"message\":\"invalid JSON: {}\"}}"#, e) which emits malformed JSON when the serde error text contains a quote or newline.

**Recommendation:** Make hipfire-daemon-protocol the single source of truth for BOTH sides: have the daemon deserialize each line into DaemonRequest (serde tag=\"type\" already exists) and match on the typed enum, and construct DaemonResponse values that it serializes, instead of Value probing and json! literals. This deletes the ~34-arm string dispatch and the raw-string error emitter, and makes protocol additions a compile error on the server until handled. Correct the review record: the adapter is NOT the third copy; it is the correct typed consumer, and the daemon is the surface to migrate.

**Evidence:** main.rs:2630 `let msg: serde_json::Value = serde_json::from_str(&line)`; main.rs:2643 `msg.get("type").and_then(|v| v.as_str())`; grep of main.rs for `DaemonResponse::` returns 0 and for `use hipfire_daemon_protocol`/`DaemonRequest`/`DaemonResponse` returns nothing; adapter lib.rs:17-21 imports the typed enums.

*Verification: confirmed*

### [Medium] abstraction — `crates/hipfire-daemon-adapter/src/lib.rs:197-289, 341-454`

**Observation:** Two boilerplate wrapper families dominate DaemonEngine. Family A (send + drain to a fixed OK ack): steer_begin_capture (L313), steer_capture (L327), steer_begin_apply (L359), steer_clear (L365) route through expect_steer_ok (L371-380); lora_load (L385), lora_set_scale (L401), lora_unload (L415), lora_clear (L424) route through expect_lora_ok (L445-454). expect_steer_ok and expect_lora_ok are byte-identical except for the SteerOk vs LoraOk variant. Family B (send + loop matching one payload variant, Error->bail 'daemon <op> error', Unknown->{}, other->warn 'unexpected response during <op>') repeats verbatim across unload (L199), reset (L217), ping (L232), inventory (L246), model_registry (L261), collect (L279), steer_finish_capture (L343), lora_list (L432) — 8 methods differing only in the matched variant and extracted payload.

**Recommendation:** Collapse into a single generic drain helper, e.g. `async fn recv_until<T>(&mut self, op: &str, mut f: impl FnMut(DaemonResponse) -> ControlFlow<anyhow::Result<T>>) -> anyhow::Result<T>` that centralizes the Error/Unknown/other arms, or a small declarative macro that generates each wrapper from (method name, request variant, response variant, payload extractor). This removes ~16 near-duplicate method bodies plus the two identical expect_*_ok helpers and makes the Error/warn text uniform.

**Evidence:** expect_steer_ok (L371-380) and expect_lora_ok (L445-454) differ only in DaemonResponse::SteerOk vs ::LoraOk; the 8 Family-B methods share the identical `loop { match self.recv().await? { <one variant> => return Ok(..), Error(e) => bail!, Unknown => {}, other => warn! } }` skeleton.

*Verification: confirmed*

### [Medium] missing-tests — `crates/hipfire-daemon-adapter/src/lib.rs:311-454, 1010-1387`

**Observation:** The MockTransport harness makes every method cheaply testable, yet ~18 pub async methods have zero coverage. Tested: load, inventory, generate_collected, reset, abort/force_answer, generate_streaming_events, generate_streaming_events_controlled. Untested: unload, ping, model_registry, collect, kld_eval, and the ENTIRE steer_* family (5 methods) and lora_* family (6 methods) — precisely the near-identical wrappers most prone to a copy-paste variant mismatch (e.g. matching the wrong OK ack). The critic's counts are inaccurate: DaemonEngine has ~26 pub + 4 private async methods (~30), not ~72, and the crate has 11 test fns (7 tokio + 4 sync), not 4 — but the coverage hole is real.

**Recommendation:** After collapsing the wrappers (prior finding), add one table-driven test that drives each request variant through MockTransport and asserts the emitted wire line and the OK/payload/Error handling — this covers all steer_*/lora_* families at once. At minimum add direct tests for lora_list, collect, kld_eval (on_chunk streaming), and steer_finish_capture, whose payload extraction is non-trivial.

**Evidence:** grep 'pub async fn' = 26 and 'async fn' = 38 in lib.rs; grep '#[test]/#[tokio::test]' = 11; test bodies (L1069-1386) exercise only load/inventory/generate*/reset/abort/force_answer; no test references lora_ or steer_.

*Verification: confirmed*

### [Medium] module-structure — `crates/hipfire-daemon-adapter/src/lib.rs:95-596, 603-663, 665-1008`

**Observation:** A crate named '-adapter' (client IPC) also hosts daemon SERVER startup logic in the same 1387-line file: acquire_resource_lease_or_exit (L924) and fatal_startup_error (L906) are invoked from the daemon binary itself (main.rs:2576), together with HIP/NPU/CPU-core lock-id resolution (L732-818), hostname probing (L820), and holder-line formatting. This mixes three responsibilities — (1) the async DaemonEngine client + transport, (2) daemon-binary discovery, (3) daemon-side resource-lease acquisition and process-fatal startup — that have different consumers (server admin route consumes resource_lock_report at admin.rs:207; the daemon binary consumes the lease/exit path; CLI/eval/server consume find_daemon_bin and DaemonEngine).

**Recommendation:** Split by consumer: keep the DaemonEngine client + transport + find_daemon_bin in the adapter crate, and move the daemon-startup resource-lease/fatal-error/id-resolution surface into a hipfire-daemon (or hipfire-lock-adjacent) module the server binary owns. This stops a client-side crate from carrying process-exit and server-startup semantics and shrinks the god-file.

**Evidence:** acquire_resource_lease_or_exit defined at lib.rs:924 and called at hipfire-daemon/src/main.rs:2576; fatal_startup_error (lib.rs:906) calls std::process::exit(1); DaemonEngine client at lib.rs:95; find_daemon_bin at lib.rs:603 — all in one file.

*Verification: confirmed*

### [Low] invariant-violation — `crates/hipfire-daemon-adapter/src/lib.rs:665-1008`

**Observation:** Compliance check requested by the critic: the resource-lock code is fully compliant with the single-lock-primitive invariant. try_acquire_resource_lock (L880) and acquire_resource_lease_or_exit (L924) use hipfire_lock::FlockGuard::open + try_lock/lock_blocking; resource_lock_report (L849) uses hipfire_lock::probe / gpu_resource_lock_path; the lease root is hipfire_lock::resource_lock_root (L942). The per-resource '<resource>.lock' files ARE the flock targets (not sentinel/liveness files), and holder metadata is written under the held flock via guard.write_holder (L889, L975) for status display only. No pidfile-liveness check, create_dir mutex, or alternate lockfile is present; the drop-releases-flock behavior is asserted by the existing test (L1366-1386).

**Recommendation:** No change required — record as verified-compliant. If the module is later split (prior finding), keep the FlockGuard usage intact and do not introduce a fallback lock path.

**Evidence:** lib.rs:885 `hipfire_lock::FlockGuard::open`; lib.rs:961 `guard.lock_blocking`; lib.rs:854/869 `hipfire_lock::probe`; test at L1379-1382 confirms flock is released on guard drop with no manual cleanup.

*Verification: confirmed*


## Accelerator/fallback crates: npu, xdna, hneurons, cpu, vision-cache + lock-discipline audit

*Subsystem key: `npu-xdna-cpu` — 7 finding(s)*

**Subsystem assessment:** These five accelerator/fallback crates are small, well-documented, and mostly clean. hipfire-xdna (telemetry-only ioctl device layer), hipfire-hneurons (CETT + L1 probe), and hipfire-vision-cache (content-addressed LRU) each have a single clear responsibility and good tests. hipfire-npu is a pure admission-policy layer that correctly delegates device access to hipfire-xdna. The weakest crate is hipfire-cpu: despite the name/description ("Deterministic CPU oracle backends"), ~95% of its 894 LOC is a Qwen3.5-specific backend-selection/module-evidence policy DSL, and its one actual CPU compute path (the "oracle") is only smoke-tested. Two clarifications on the assignment's premises: (1) CpuTensor is NOT in hipfire-cpu — it is defined in hipfire-diffusion/src/lib.rs:2455; hipfire-cpu's only overlap with hipfire-primitives is three 1:1 conv re-export wrappers. (2) npu/xdna do not duplicate rdna-compute's dispatch surface — there is no NPU kernel dispatch yet (xdna is read-only telemetry; dispatch is documented as "future modules"); the only decision-logic duplication is between hipfire-cpu and hipfire-npu. Lock discipline is otherwise sound across the workspace: daemon singleton (~/.hipfire/daemon.pid), resource leases (hipfire-daemon-adapter), and gpu-lock all go through hipfire_lock::FlockGuard/probe. The only one-lock-primitive violations are two dev scripts that treat /tmp/hipfire-gpu.lock as an unlinkable sentinel rather than a stable-inode flock target — both explicitly forbidden by crates/hipfire-lock/AGENTS.md.

### [Medium] module-structure — `crates/hipfire-cpu/src/lib.rs:9-551`

**Observation:** The crate is named 'hipfire-cpu' and described as 'Deterministic CPU oracle backends', but ~540 of its 894 lines are a Qwen3.5-specific backend-selection and module-evidence policy DSL (DenseFfnBackend, BackendSelection, ModuleInvocation/ModuleOutput enums, and their JSON serializers). The only genuine CPU compute is swiglu_down_bf16_cpu (L594) and diff_stats (L624). The policy types are hardcoded to one model family (module_id format 'qwen35.layers.{}.mlp.swiglu_down', module_kind strings 'qwen35_dense_ffn_swiglu_down'/'qwen35_attention_wo_residual'), so this is not a reusable numeric leaf — it is model-admission policy wearing a 'cpu' name.

**Recommendation:** Split the crate: keep the numeric oracle (swiglu_down_bf16_cpu, diff_stats, decode_w_down_shadow) in a small CPU/oracle crate, and move the Qwen3.5 backend-selection/evidence contract types into a model- or evidence-scoped crate where the qwen35 coupling belongs. At minimum, rename to reflect that it is a backend-policy crate, not a CPU-compute leaf.

**Evidence:** L281-283 dense_ffn_module_id -> 'qwen35.layers.{layer_idx}.mlp.swiglu_down'; L304 module_kind 'qwen35_dense_ffn_swiglu_down'; only two real compute fns at L594 (swiglu_down_bf16_cpu) and L624 (diff_stats).

*Verification: confirmed*

### [Medium] duplication — `crates/hipfire-npu/src/lib.rs:231-247`

**Observation:** xdna_swiglu_admission hand-rolls its own preference+availability -> (DenseFfnBackend, fallback_reason) state machine, parallel to hipfire-cpu's dense_ffn_backend_decision (crates/hipfire-cpu/src/lib.rs:323). Both encode 'if opt-in and available -> NpuXdna, else GpuProduction + a reason string', but with divergent fallback vocabularies (npu uses NPU_ARTIFACTS_MISSING_FALLBACK; cpu uses "npu_backend_unavailable"). This is the only dispatch-decision duplication in the subsystem — npu/xdna do NOT duplicate rdna-compute (no NPU kernel dispatch exists yet; xdna is telemetry-only), but the two backend-selection functions can drift independently.

**Recommendation:** Have xdna_swiglu_admission reuse dense_ffn_backend_decision (passing artifacts_available as the availability signal) or hoist a single backend-decision function into one crate so the selected-backend + fallback-reason vocabulary has one source of truth.

**Evidence:** hipfire-npu L238-247 computes (selected_backend, fallback_reason) inline; hipfire-cpu L323-336 dense_ffn_backend_decision computes the same tuple with a different reason string.

*Verification: confirmed*

### [Medium] missing-tests — `crates/hipfire-xdna/src/lib.rs:251-320`

**Observation:** The ioctl decode paths (sensors, resource_info, clocks) parse raw repr(C) kernel buffers by manual field access and byte offsets, yet there are no tests exercising the parse logic — the tests only cover mean_utilization_pct arithmetic (L371-383) and that open_default doesn't panic (L386). The ABI structs are guarded only by compile-time size asserts (L170-173); the runtime decode (e.g. counting SensorRaw records by written/size_of, splitting POWER vs COLUMN_UTILIZATION by kind) is the real risk surface and is untestable-by-hardware in CI but trivially testable with a crafted byte buffer.

**Recommendation:** Factor the buffer->NpuSensors/NpuResourceInfo/NpuClocks decode into pure functions taking &[u8] and add unit tests with synthetic little-endian buffers (including short-buffer -> ShortResponse and mixed sensor kinds), so ABI/offset regressions are caught without an NPU.

**Evidence:** sensors() L266 `count = written / size_of::<SensorRaw>()` and L270-274 kind dispatch have no test; tests module L366-391 contains only mean_utilization and open_default_is_graceful_when_absent.

*Verification: confirmed*

### [Medium] invariant-violation — `tests/pp-gate.sh:503`

**Observation:** The gate decides whether to acquire the GPU lock via `[ ! -f /tmp/hipfire-gpu.lock ]` — treating lockfile presence as 'lock is currently held'. This is sentinel-file semantics, not flock semantics: a flock lockfile is created on first acquire and is meant to persist with a stable inode, so once anything has ever taken the GPU lock on the box the file exists permanently, making this test skip acquisition even when the lock is free. It also interacts badly with serve-restart.sh deleting the same file. The one-lock-primitive invariant means held/free state must come from flock, not file existence.

**Recommendation:** Replace the `-f` existence probe with `hipfire lock status` / `gpu-lock status` (backed by hipfire_lock::probe -> LockState) to detect whether a parent already holds the lock, instead of inferring it from the lockfile's presence.

**Evidence:** tests/pp-gate.sh:503 guards acquisition with `[ ! -f /tmp/hipfire-gpu.lock ]`; hipfire_lock exposes probe()/LockState (crates/hipfire-lock/src/lib.rs) which resource_lock_report already uses for the same question.

*Verification: adjusted — Line confirmed: acquisition guarded by `[ ! -f /tmp/hipfire-gpu.lock ]` (L503). The antipattern (inferring flock held/free from file existence instead of the flock primitive) and the recommendation (use `hipfire lock status`/`gpu-lock status` backed by probe()->LockState, which is exactly what resource_lock_report in hipfire-daemon-adapter uses) are correct and invariant-aligned, so the finding is directionally right. But the described MECHANISM is wrong: gpu-lock acquire writes ~/.hipfire/locks/hip-gpu-0.lock, never /tmp/hipfire-gpu.lock (see finding above), so that file is essentially never created by hipfire. Consequently the `-f` test is almost always false and the gate ALWAYS attempts acquisition -- the opposite of the finding's 'file exists permanently -> skips acquisition even when free'. The real bug is that the parent-lock-detection guard (comment L499-502, meant to avoid deadlocking on a parent's already-held lock) is effectively DEAD: it probes a stale path, so when run nested under a holder of the real lock it will still try to acquire and block/timeout (exit 2). The claimed bad interaction with serve-restart.sh deleting the same file is moot since neither touches the real lock. Kept at Medium because the guard being broken plus the sentinel-file-vs-flock invariant issue is a genuine functional/maintainability concern; the fix (probe the canonical path via hipfire lock status) also repairs the stale-path bug.*

### [Low] duplication — `crates/hipfire-cpu/src/lib.rs:553-563`

**Observation:** The assignment's premise that 'CpuTensor in hipfire-cpu overlaps hipfire-primitives' is inaccurate: CpuTensor is defined in hipfire-diffusion (crates/hipfire-diffusion/src/lib.rs:2455), not here. hipfire-cpu's actual overlap with the numeric leaf hipfire-primitives is three thin 1:1 re-export wrappers (f32_to_bf16_bits_rne, bf16_bits_to_f32, round_f32_to_bf16) that just forward to hipfire_primitives::conv::*. These add an indirection layer and a second import surface for the same functions.

**Recommendation:** Drop the wrappers and have callers use hipfire_primitives::conv directly, or if a stable re-export facade is wanted, `pub use hipfire_primitives::conv::{...}` instead of hand-written forwarders. Separately, the real numeric-leaf duplication to track is hipfire-diffusion defining its own CpuTensor rather than sharing a workspace numeric tensor type.

**Evidence:** L554 `hipfire_primitives::conv::f32_to_bf16_bits(x)`, L558 `bf16_bits_to_f32(bits)`, L562 `round_f32_to_bf16(x)` are 1-line pass-throughs; CpuTensor lives at crates/hipfire-diffusion/src/lib.rs:2455 per graphify node.

*Verification: confirmed*

### [Low] missing-tests — `crates/hipfire-cpu/src/lib.rs:883-893`

**Observation:** swiglu_down_bf16_cpu is the crate's stated reason to exist (the 'deterministic CPU oracle'), but its only test tiny_swiglu_down_bf16_cpu asserts output length == 2 and that all values are finite — it never checks a golden numeric value. An oracle whose output is not pinned to a reference cannot detect a silent arithmetic regression, which defeats its purpose as the drift baseline for GPU/NPU backends.

**Recommendation:** Add a test that pins swiglu_down_bf16_cpu output to hand-computed bf16-rounded reference values for a tiny known input, so the oracle itself is protected against regressions.

**Evidence:** L890-892 assert only `out.len() == 2` and `out.iter().all(|v| v.is_finite())`; no expected-value comparison.

*Verification: confirmed*

### [Low] invariant-violation — `scripts/serve-restart.sh:15`

**Observation:** `rm -f ~/.hipfire/daemon.pid ~/.hipfire/serve.pid /tmp/hipfire-gpu.lock` unlinks the GPU flock lockfile as part of restart cleanup. crates/hipfire-lock/AGENTS.md explicitly states 'Keep the lockfile inode stable. Do not unlink a flocked lockfile as a release mechanism.' flock releases automatically on process death (the script already kill -9's all hipfire processes on L12-13), so the rm is both unnecessary and a contract violation: unlinking changes the inode, so a concurrent GPU consumer holding flock on the old inode and a new acquirer creating a fresh file at the same path both believe they hold the lock — silently breaking mutual exclusion.

**Recommendation:** Remove /tmp/hipfire-gpu.lock (and the pid files) from the rm line; rely on flock auto-release after the process kill, or call `hipfire lock release` / `gpu-lock release` if an explicit release is wanted.

**Evidence:** scripts/serve-restart.sh:15 rm's /tmp/hipfire-gpu.lock; crates/hipfire-lock/AGENTS.md 'Keep the lockfile inode stable. Do not unlink a flocked lockfile as a release mechanism.'

*Verification: adjusted — Line confirmed: `rm -f ~/.hipfire/daemon.pid ~/.hipfire/serve.pid /tmp/hipfire-gpu.lock`. But the core claim is REFUTED: /tmp/hipfire-gpu.lock is NOT the current GPU flock lockfile. The `gpu-lock` CLI uses hipfire_lock::gpu_resource_lock_path() = resource_lock_path('hip-gpu-0') = ~/.hipfire/locks/hip-gpu-0.lock (hipfire-cli/src/commands/lock.rs:92-95, 135; hipfire-lock/src/lib.rs:201-203). lib.rs:199 explicitly says gpu_lock_path()/HIPFIRE_GPU_LOCKFILE was REPLACED. Grep confirms NO Rust code opens/flocks /tmp/hipfire-gpu.lock -- it survives only in stale docs (hipfire-lock/AGENTS.md:9, hipfire-daemon/AGENTS.md:11, lib.rs:9) and these two scripts. Therefore the claimed mutual-exclusion break (concurrent flock holder on old inode vs new acquirer) cannot occur, and the 'do not unlink a flocked lockfile' invariant is not actually violated because nothing flocks that path. Residual valid issue: it is a dead/legacy-path reference that should be removed (AGENTS.md: update stale references / remove legacy fallback). Downgraded to Low hygiene. Note the stale hipfire-lock/AGENTS.md itself still lists /tmp/hipfire-gpu.lock and is out of sync with the code.*


## hipfire-kvquant (KVarN codec)

*Subsystem key: `kvquant` — 5 finding(s)*

**Subsystem assessment:** hipfire-kvquant is a well-formed leaf crate (deps: only hipfire-primitives) holding the clean-room KVarN codec (kvarn.rs: Sinkhorn variance-normalize + per-channel affine 4-bit + on-device record pack/unpack) and the deferred cold-tier compaction pass (kv_compact.rs). Layering holds and is healthier than the main-review baseline: it does NOT re-declare kv.rs's 9-boolean quant-mode soup (those booleans stay in hipfire-runtime; kvquant is selected by exactly one of them, quant_kvarn), and it correctly delegates f16<->f32 to hipfire-primitives::conv instead of copying codecs.rs's third f16 encoder — so the two most obvious duplication risks are actually avoided. Consumers are hipfire-quantize (offline bin) and hipfire-runtime/kv_hier.rs (runtime cold-tier read), matching the Cargo.toml stated intent; no coexistence-rule violation (KVarN is a live KV format, not import/export). The real issues are narrower: (1) a production-reachable sub-nibble packing path (bits in {1,2,3} via HIPFIRE_KV_COLD_BITS) with zero byte-exact round-trip test — only bits=4 is covered; (2) pack_kvarn_tile_bits silently truncates codes to the bit width with no fit-check, and qmax/bits are decoupled parameters; (3) the on-device record byte-layout is re-declared independently in kvarn.rs plus four HIP kernels with no shared source, and has already partially diverged; (4) FWHT seed/width and GROUP constants are duplicated; (5) the cold pass is hard-locked to head_dim==256 though the tile codec is dimension-agnostic. Test coverage of the codec math (cos-sim reconstruction, byte-exact bits=4 round-trip, Sinkhorn imbalance drop, two-tier attention) is otherwise strong and host-only.

### [Medium] input-validation — `/home/sadara/hipfire/crates/hipfire-kvquant/src/kvarn.rs:279 (pack_kvarn_tile_bits inner loop)`

**Observation:** pack_kvarn_tile_bits does `let code = qt.q[i] & mask;` with mask=(1<<bits)-1 and no assertion that the code actually fits in `bits`. The code width (qmax, set in quantize_tile_qmax) and the storage width (bits, set at pack time) are independent parameters passed through separate calls. If a caller quantizes at qmax=15 (4-bit codes) but packs at bits=2, every code 4..15 is silently truncated to its low 2 bits and reconstructs as garbage, with no panic, warning, or test to catch it. Callers happen to keep them consistent today (kv_hier derives both from HIPFIRE_KV_COLD_BITS), but the codec API invites silent corruption.

**Recommendation:** Add `debug_assert!(qt.q[i] <= mask)` (or a checked precondition asserting max code <= 2^bits-1 once before the loop), or couple the two by storing the effective qmax/bit-width in QuantTile so pack cannot be called with a mismatched width.

**Evidence:** kvarn.rs:277 mask=(1u8<<bits)-1; kvarn.rs:279 `let code = qt.q[i] & mask;` (silent truncation, no assert). quantize_tile_qmax (kvarn.rs:187) takes qmax independently; pack_kvarn_tile_bits (kvarn.rs:272) takes bits independently; QuantTile (kvarn.rs:165) stores neither, so nothing links them.

*Verification: confirmed*

### [Medium] duplication — `/home/sadara/hipfire/crates/hipfire-kvquant/src/kvarn.rs:237-246 vs /home/sadara/hipfire/kernels/src/kvarn_dequant_tile.hip:40-45, kvarn_build_kcache.hip:41-57, attention_flash_kvarn_tile_batched.hip:26-28`

**Observation:** The on-device KVarN record layout — nibble/sub-nibble code block, then fp16 scale_abs[r], zp_abs[r], s_col[c] in that fixed order, with offsets off_scale=qbytes, off_zp=+r*2, off_scol=+r*2 — is re-declared by hand in the Rust host codec and independently in at least four HIP kernels, with no single shared definition. This host/device split is inherent, but the metadata block order and offset arithmetic are copy-pasted 5x, so a layout change requires editing 5 files with zero compile-time linkage. The mirror has already partially diverged: kvarn_build_kcache.hip:57 hard-codes 4-bit extraction `(i&1)==0 ? (byte&0xf) : (byte>>4)` while the Rust codec and kvarn_dequant_tile.hip are generic over `bits`. It is correct today only because the hot-window path is always 4-bit and the cold path routes through the generic kvarn_dequant_tile.

**Recommendation:** Emit or generate the byte offsets from one source (a small shared const/struct the kernels also consume, or a codegen header), and add a host-vs-device parity test that packs at each supported `bits` and dequants through every kernel that reads the record, so a layout or bit-width change fails loudly instead of silently mis-decoding one kernel.

**Evidence:** kvarn.rs:241-245 record layout comment matches kvarn_dequant_tile.hip:40-42 (off_scale=qbytes, off_zp=+r_dim*2, off_scol=+r_dim*2) and kvarn_build_kcache.hip:41-43. kvarn_build_kcache.hip:57 hard-codes 4-bit nibble extract; kvarn_dequant_tile.hip:27 takes a runtime `bits` arg. kv_hier.rs:505/514 cold read passes seg.bits into kvarn_dequant_tile (generic path).

*Verification: confirmed*

### [Low] missing-tests — `/home/sadara/hipfire/crates/hipfire-kvquant/src/kvarn.rs:272 (pack_kvarn_tile_bits) and :304 (unpack_kvarn_tile_bits); test at :413`

**Observation:** The bit-parametric pack/unpack + record-size index math (cpb=8/bits, mask, div_ceil byte offsets) is production-reachable — kv_hier.rs reads HIPFIRE_KV_COLD_BITS, clamps to 1..4, and passes it straight into pack_kvarn_tile_bits/kvarn_record_bytes_bits — yet the only byte-exact round-trip test (kvarn_tile_record_roundtrips) goes through pack_kvarn_tile/unpack_kvarn_tile, which are hard-wired to bits=4. No test in the workspace exercises bits in {1,2,3}. The fiddly sub-nibble index math (multiple codes per byte at 1/2/3 bits, odd-dim div_ceil) is exactly the host-testable path most likely to have off-by-one packing bugs, and it is uncovered.

**Recommendation:** Add a parameterized byte-exact round-trip test over bits in {1,2,3,4} and an odd c_dim: assert kvarn_record_bytes_bits equals rec.len(), pack->unpack recovers q exactly, and codes are masked-consistent. Cheap, pure-CPU, and closes the gap the runtime already depends on.

**Evidence:** kvarn.rs:264-265 pack_kvarn_tile forwards bits=4; test kvarn.rs:417 calls pack_kvarn_tile (bits=4 only). kv_hier.rs:174-179 clamps HIPFIRE_KV_COLD_BITS to 1..4; kv_hier.rs:340-349 passes bits into pack_kvarn_tile_bits/kvarn_record_bytes_bits. Workspace grep for pack_kvarn_tile_bits/unpack_kvarn_tile_bits inside test/assert context returns nothing.

*Verification: adjusted — Facts all verified: the only byte-exact round-trip test kvarn_tile_record_roundtrips (kvarn.rs:413) goes through pack_kvarn_tile/unpack_kvarn_tile (bits=4, lines 417/419). No workspace test exercises pack_kvarn_tile_bits/unpack_kvarn_tile_bits at bits in {1,2,3} (grep for those symbols in test/assert context is empty; the only non-4-bit caller is the example parity_kv_hier.rs:112/118, not a byte-exact assert test). Production-reachable confirmed: kv_hier.rs:174-179 clamps HIPFIRE_KV_COLD_BITS to (1,4) and kv_hier.rs:341/348-349 feed bits into kvarn_record_bytes_bits/pack_kvarn_tile_bits. Downgrading Medium->Low: (1) missing-tests is hygiene per the rubric, not a code antipattern; (2) the recommended test is largely tautological — pack allocates via kvarn_record_bytes_bits so rec.len() assert is trivially true, and pack/unpack compute cpb+mask identically so pack->unpack recovers q by construction (a symmetric bug survives a round-trip). The genuinely load-bearing coverage gap is host<->device parity, which is exactly finding #3. So the finding is real but its stated payoff is overstated.*

### [Low] duplication — `/home/sadara/hipfire/crates/hipfire-kvquant/src/kv_compact.rs:120 (encode) vs :177 (dequant_head); /home/sadara/hipfire/crates/hipfire-runtime/src/kv.rs:1018 vs /home/sadara/hipfire/crates/rdna-compute/src/dispatch/kv.rs:1669`

**Observation:** compact_cold_kv builds its FWHT sign tables with the magic seed pair gen_fwht_signs(42,256)/gen_fwht_signs(1042,256), and dequant_head repeats the identical seeds/width to invert the rotation. The two must agree exactly or every cold round-trip silently corrupts, yet the constants live as untied literals in two functions of the same file. Separately, the KVarN block width is a named const KVARN_GROUP=128 in kv.rs but is re-hard-coded as `const GROUP=128` in the rdna-compute dispatch.

**Recommendation:** Hoist the FWHT seed pair and width to a single `const (KVARN_FWHT_SEED_A, KVARN_FWHT_SEED_B, KVARN_FWHT_DIM)` in kvquant used by both encode and decode, and have the dispatch import KVARN_GROUP instead of re-declaring 128.

**Evidence:** kv_compact.rs:120 gen_fwht_signs(42,256)/gen_fwht_signs(1042,256); kv_compact.rs:177 same pair in dequant_head. kv.rs:1018 `pub const KVARN_GROUP: usize = 128`; rdna-compute/src/dispatch/kv.rs:1669 `const GROUP: usize = 128`.

*Verification: confirmed*

### [Low] coupling — `/home/sadara/hipfire/crates/hipfire-kvquant/src/kv_compact.rs:65`

**Observation:** compact_cold_kv hard-asserts head_dim==256 ("KVarN v1 FWHT is 256-wide"), permanently coupling the cold-tier compaction to a 256-wide head even though the underlying tile codec (quantize_tile/dequantize_tile) is fully dimension-agnostic and the rest of the KVarN runtime path explicitly supports head_dim==128 or 256 (kv.rs guards). This blocks the cold tier for any 128-dim-head model with no codec reason — the limitation is purely the fixed-width FWHT built inside this function.

**Recommendation:** Parameterize the FWHT width (already a natural fit given gen_fwht_signs takes a width) or relax the assert to head_dim in {128,256} to match the hot path, so the cold tier is not narrower than the format it feeds.

**Evidence:** kv_compact.rs:65 assert_eq!(head_dim, 256, "KVarN v1 FWHT is 256-wide"). kv.rs guards accept head_dim==128||256 (e.g. kv.rs:703,846,1061). quantize_tile/dequantize_tile (kvarn.rs:178,225) take r_dim/c_dim generically with no 256 requirement.

*Verification: confirmed*


## SD API input-validation review (hipfire-server/routes/sdapi.rs)

*Subsystem key: `sdapi-validation` — 5 finding(s)*

**Subsystem assessment:** crates/hipfire-server/src/routes/sdapi.rs is a ~8250-LOC Automatic1111-compatible HTTP surface. Several things are done well: numeric fields are typed as Option<u32>/Option<i64> and deserialized by serde, so malformed numbers yield a 422 rather than a panic (no unwrap()/parse().unwrap() on user-supplied scalars in the request path — the many unwrap() calls after L4300 are all in #[cfg(test)] code); errors funnel through DiffusionError -> diffusion_error_response (L2986) mapping to proper 400/409/500 status codes instead of leaking panics; overflow is guarded with checked_mul in decode_sd_init_images (L2944), sdapi_checked_shape_elements (L1895), and u32::try_from guards; axum 0.8's Json extractor applies the implicit 2 MB DefaultBodyLimit, capping raw body size (so base64 payload flooding is bounded by default). The gaps: request geometry (width/height/steps/batch_size/n_iter) is passed to allocation/generation with zero upper bound; a user-supplied output directory in override_settings is used verbatim for filesystem writes; model names are not confined to the models directory; and the entire /sdapi/* surface is unauthenticated (only /admin/* is gated). Default bind is 127.0.0.1 which mitigates exposure, but host is configurable to 0.0.0.0 and CORS can be set to allow any origin. The idiomatic fix is a TryFrom/validate step converting SdGenerationRequest into a bounds-checked internal request type that rejects absurd geometry with 400, plus canonicalize+prefix confinement for any user path.

### [High] input-validation — `crates/hipfire-server/src/routes/sdapi.rs:1711-1765 (sd_request_to_diffusion_batch_request); helpers batch_size_for_body:2885, sd_request_n_iter:2889; alloc at crates/hipfire-diffusion/src/lib.rs:1145`

**Observation:** width/height/steps/batch_size/n_iter are accepted with only unwrap_or defaults and .max(1) — no upper bound anywhere. width=body.width.unwrap_or(512) and height=...unwrap_or(512) (L1735/1740), steps=body.steps.unwrap_or(20) (L1763), batch_size=...max(1) (L1716), n_iter=...max(1) (L2889) flow straight into the latent allocation Vec::with_capacity(batch*channels*height*width) in the diffusion crate (L1145) and into a Vec::with_capacity(n_iter as usize) output buffer (sdapi.rs:480). A tiny unauthenticated JSON body like {"width":100000,"height":100000} (well under the 2 MB body limit) drives a multi-hundred-GB allocation; huge steps/n_iter tie up the GPU indefinitely. Classic unbounded width*height memory-DoS plus compute-DoS.

**Recommendation:** Add an explicit validation pass before any allocation: reject width/height above a portability-safe cap (e.g. <=4096 and multiple-of-8), steps above a ceiling (e.g. <=200), and batch_size*n_iter above a small bound, returning DiffusionError::InvalidRequest (400). Idiomatically, implement TryFrom<SdGenerationRequest> for a validated ValidatedSdRequest newtype so the bounds are enforced at the extractor boundary rather than scattered through the handlers.

**Evidence:** sdapi.rs:1735/1740/1763 apply unwrap_or with no max; sd_request_n_iter/batch_size_for_body only .max(1) (2885-2891); latent buffer Vec::with_capacity(batch*channels*height*width) at hipfire-diffusion/src/lib.rs:1145; grep for .min(/.clamp(/MAX_WIDTH over sdapi.rs finds only pixel-value and denoising_strength clamps, none on geometry

*Verification: confirmed*

### [High] input-validation — `crates/hipfire-server/src/routes/sdapi.rs:1651-1685 (sdapi_output_dir) and 1484-1526 (save_sdapi_images_with_kind)`

**Observation:** sdapi_output_dir reads a fully client-controlled path from override_settings (outdir_txt2img_samples / outdir_img2img_samples / outdir_samples / outdir_*_grids) and turns it into a directory with PathBuf::from(str) (L1684) with no sanitization — absolute paths and .. are accepted. save_sdapi_images_with_kind then does fs::create_dir_all(&output_dir) (L1490) and writes attacker-supplied PNG bytes via output_dir.join(...)/File::create (L1513-1518). An unauthenticated client that sets save_images:true and override_settings.outdir_samples to any writable location (e.g. a home dir, a systemd/cron drop path) causes the daemon to create that directory and write files there. Content is constrained to PNG-magic bytes and a .png filename, but the destination directory is arbitrary.

**Recommendation:** Do not honor client-supplied output directories from override_settings on a network surface, or confine them: canonicalize the requested path and require it to be a prefix of a configured, server-owned output root; reject absolute paths and any component containing '..'. Treat outdir_* as admin-only config, not per-request input.

**Evidence:** sdapi_output_dir maps override_settings.get(mode_key/kind_key/"outdir_samples").as_str() -> PathBuf::from with unwrap_or fallback (1673-1684), no traversal check; fs::create_dir_all(&output_dir) at 1490; File::create(&path) writing decoded bytes at 1513-1518

*Verification: confirmed*

### [Medium] input-validation — `crates/hipfire-server/src/routes/sdapi.rs:392-399 (resolve_diffusion_hfq_candidate) -> crates/hipfire-server/src/model/discovery.rs:16 (find_model) -> crates/hipfire-model/src/lib.rs:1125-1140 (find_model_in)`

**Observation:** The user-supplied model field (via sd_requested_model / override_settings.sd_model_checkpoint) reaches find_model_in, which first does PathBuf::from(arg).exists() (accepting absolute paths) and then models_dir.join(arg) (accepting ../ traversal). Resolution is not confined to the models directory, so a client can point the daemon at arbitrary filesystem paths. Practical impact is bounded because the sdapi caller then requires inspect_hfq(&path).is_ok() (L397), so only parseable .hfq files load — but the traversal itself (open/parse any path on disk as a model) is unvalidated and is a robustness/least-surprise gap on a network surface.

**Recommendation:** Confine model lookup for network requests: after resolution, canonicalize and assert the result is within models_dir (or an allowlist of roots); reject arg values that are absolute or contain path separators / '..' before joining. Keep the permissive direct-path behavior for CLI-only callers, not the HTTP path.

**Evidence:** find_model_in: `let direct = PathBuf::from(arg); if direct.exists() { return Some(direct); }` and `models_dir.join(arg)` with no traversal guard (lib.rs:1126-1134); reached from sdapi resolve_diffusion_hfq_candidate calling find_model(candidate) at sdapi.rs:396

*Verification: confirmed*

### [Medium] input-validation — `crates/hipfire-server/src/lib.rs:79-82 (admin_gate route_layer on admin_data) and 127-230 (sdapi routes, no auth layer) and 231-240 (global CORS + touch_last_request layers)`

**Observation:** Auth middleware (auth::admin_gate) is attached only to the admin_data sub-router (route_layer at L84-87). The /sdapi/* routes (L127+) receive only the global touch_last_request and CORS layers — there is no authentication or rate limiting on any SD endpoint, including the expensive GPU-bound txt2img/img2img and the state-mutating post_options / reload-checkpoint / unload-checkpoint. This is consistent with Automatic1111's own unauthenticated convention and the default bind is 127.0.0.1 (hipfire-config default_host, lib.rs:24-25), which limits exposure, but host is configurable to 0.0.0.0 and cors_allowed_origins can be set to "*", at which point any network client can trigger the unbounded-geometry DoS (finding 1) and arbitrary-write (finding 2) with no credential.

**Recommendation:** Decide explicitly whether /sdapi/* should be reachable off-host. If it can bind non-localhost, gate the generation and options-mutating endpoints behind the existing bearer/session auth (reuse admin_gate or a dedicated api gate) and/or add per-client rate limiting via a tower layer. At minimum, document that binding host!=127.0.0.1 exposes unauthenticated GPU endpoints.

**Evidence:** admin_gate applied via route_layer only on admin_data (lib.rs:84-87); sdapi routes added with no route_layer (lib.rs:127-238); only touch_last_request + optional CORS layered globally (lib.rs:244-250); default_host() returns "127.0.0.1" (hipfire-config/src/lib.rs:24-25)

*Verification: adjusted — Substantive claim CONFIRMED but the cited line anchors are materially off. The admin_gate route_layer is actually at lib.rs:79-82 (`.route_layer(middleware::from_fn_with_state(state.clone(), auth::admin_gate))`) attached to the admin_data sub-router (48-82); lines 84-87 are the start of the main `let router = Router::new()` and its first routes, NOT the route_layer. The sdapi routes are at 127-230 (not through 238) and carry no route_layer. The global layers are at 231-240 (CORS at 231-234, touch_last_request at 236-239) -- lines 244-250 are the `serve`/`serve_loaded` fns, not layers. All substance holds: admin_gate gates only admin_data; every /sdapi/* route (incl. txt2img/img2img and state-mutating post_options/reload-checkpoint/unload-checkpoint) has only touch_last_request + optional CORS; no auth, no rate limiting. default_host() returns "127.0.0.1" (hipfire-config:24-25) and default_cors_allowed_origins() returns Vec::new() (30-32), both user-configurable. Medium is appropriate. Recommendation (gate/rate-limit if binding non-localhost, or document exposure) violates no invariant.*

### [Medium] input-validation — `crates/hipfire-server/src/routes/sdapi.rs:1528-1534 (decode_base64_image_payload), 2969-2980 (decode_sd_init_image), 2944-2951 (decode_sd_init_images)`

**Observation:** Base64 image inputs (init_images, mask, extras image, interrogate image) are decoded with STANDARD.decode() with no explicit pre-decode size limit; the only cap is axum's implicit 2 MB DefaultBodyLimit on the Json extractor. Two robustness concerns: (1) the protection is implicit — if the body limit is ever raised (a realistic change for an SD API that needs large init images), decode and image::load_from_memory become unbounded with no local guard; and (2) decode_sd_init_images computes bytes_per_image with checked_mul (good) but then does Vec::with_capacity(bytes_per_image * decoded.len()) with an unchecked multiply (L2950) — currently masked by the 2 MB body cap, but a latent overflow/huge-alloc if that cap changes.

**Recommendation:** Set an explicit DefaultBodyLimit for the sdapi router sized to the intended max image payload, and add a local guard in the decoders: cap decoded byte length and cap decoded image pixel dimensions before allocating, and use checked_mul for bytes_per_image * decoded.len(). This keeps the invariant robust independent of the global body-limit setting.

**Evidence:** decode_base64_image_payload/decode_sd_init_image call general_purpose::STANDARD.decode(payload) with no length check (1533, 2974); Vec::with_capacity(bytes_per_image * decoded.len()) unchecked multiply at 2950; no DefaultBodyLimit override in hipfire-server (grep for DefaultBodyLimit/RequestBodyLimit returns nothing), axum = 0.8 whose Json default limit is 2 MB

*Verification: confirmed*
