# Scope: SoA-coalesced E8 indexed-MoE decode (gfx1100/RDNA3 dGPU)

**Goal.** Bank the measured SoA-coalescing win into the *production* A3B MoE decode
path. The standalone E8 GEMV bench (commit `b74c21be`) shows SoA layout is **+38–73%**
over AoS on the FFN shapes (gfx1100, cache-resident: AoS 432 → SoA 595 GB/s @ M=32768;
gate_up 335 → 586). The MoE *decode* currently uses the **AoS** indexed-batched kernels
(`gemv_mfp4g32_e8_moe_{gate_up,down}_k8_indexed_batched*`), so it leaves that on the
table. Wiring SoA into the indexed path should move E8 decode from ~102 → **~125–130
tok/s** (toward uniform-mq4's 150), at MQ6-class quality, on gfx1100.

**Why SoA wins (root cause, established).** The E8 AoS layout packs each g32 block as
17 contiguous bytes (1 E4M3 scale + 16 nibble bytes); a wave reading 4-byte codewords
strided by 17/136 B drags partial cache lines it mostly discards → cache-line over-fetch.
SoA (scales contiguous, codewords contiguous 16B-aligned) coalesces, so the wave pulls
lines it fully uses. Confirmed memory-bound (not occupancy: 16≈10 waves on real decode;
not compute: strip = +1% full decode / +25% cache-resident; LUT decode no help). Full
L3 roofline (~3500 GB/s) needs a structural rewrite (multi-row/WMMA) — OUT OF SCOPE here;
this scope only banks the coalescing lever.

## Two pieces

### 1. SoA weight layout for routed E8 experts — transpose-on-load (no re-quant)
The transform is pure byte-reordering, same total size (no VRAM change):
- AoS row: `[16B hdr][n_blocks × (1B scale + 16B codewords)]`
- SoA row: `[16B hdr (flag=0x06)][n_blocks scales, pad16][n_blocks × 16B codewords]`
- Reference impl: `aos_to_soa_row` / `aos_to_soa_full` in
  `crates/rdna-compute/examples/bench_e8_soa_correctness.rs`.

**Path: transpose at model load**, not at quant time — works with existing `.hfq`/`.mq4`
E8 files, no re-quant. When loading routed E8 experts (find the upload site in
`crates/hipfire-arch-qwen35/src/qwen35.rs` `load_moe_ffn` / wherever expert
`WeightTensor`s are uploaded to VRAM), run AoS→SoA on the host bytes before upload, gated
by a flag (`HIPFIRE_E8_SOA_EXPERTS`, default off until validated). One-time CPU cost,
~tens of ms for A3B. The expert pointer table then points at SoA blocks.
- Alternative (later): quantizer emits SoA directly (a layout byte in the header). Not
  needed for v1; transpose-on-load is the cheaper proof.

### 2. SoA indexed-MoE GEMV kernels (net-new)
Port the standalone SoA-2w body (`kernels/src/gemv_mfp4g32_e8_soa.hip`) into the
indexed-batched MoE structure of the existing AoS kernels:
- `gemv_mfp4g32_e8_soa_moe_gate_up_k8_indexed_batched` (mirror
  `gemv_mfp4g32_e8_moe_gate_up_k8_indexed_batched.gfx1151.hip`): per-slot expert pointer
  via `topk_indices` ptr-table, SoA row addressing (scale_base / cw_base + scale_padded),
  2-way unroll (max occupancy — do NOT 4/8-way, the bench proved it cuts occupancy).
- `gemv_mfp4g32_e8_soa_moe_down_k8_indexed_batched_expanded` (mirror the `_down_..._expanded`
  AoS kernel): SoA addressing + the same expanded-buffer + `moe_down_combine_k8_batched`
  contract.
- Keep the per-row math byte-identical to the AoS kernel (only the address layout differs).

## Dispatch wiring
- A layout flag on the routed-E8 path (e.g. `MoeDtypes`/`MoeParams` carries `e8_soa: bool`,
  set when experts were transposed at load). `MoeResolution::resolve_arch`
  (`crates/hipfire-dispatch/src/families/moe.rs`) routes the `use_gpu_topk` E8 path to the
  SoA kernels when `e8_soa && has_wmma_w32()` (gfx11 dGPU; gfx1151 keeps its own).
- The decode rotate/dispatch in `crates/hipfire-dispatch/src/pipeline/mod.rs` selects the
  SoA gate_up / down launchers (new `Gpu` methods, mirror the AoS ones).

## Validation (gate before default-on)
1. **Kernel numeric**: SoA-indexed output == AoS-indexed within FP-reduction noise (extend
   a parity harness; the standalone bench already shows SoA≡AoS bit-close).
2. **Coherence** (mandatory): bare "capital of France"→Paris + a code prompt, on the real
   A3B E8 model (`q36a3b.mfp4e8-gptq-v2.hfq`), graph off AND on.
3. **Decode A/B**: `HIPFIRE_E8_SOA_EXPERTS=0` (AoS baseline, ~102) vs `=1` (SoA), warm
   median, byte-identical prompt. Target ≥ +8% to default on (the project's ship bar);
   below that, keep flag default-off + document (like the gfx11 twin).

## Risks / unknowns
- **Indexed context dilutes the win**: the standalone GEMV is cache-resident + has no
  expert-pointer indirection; the real MoE decode chases 8 expert pointers/token and the
  experts may be VRAM-cold (520 MB/token >> 96 MB L3 if routing isn't hot). Coalescing
  helps MORE when cold (less wasted VRAM BW), but the per-expert pointer latency is new.
  Measure on the real model, don't assume the 38% transfers 1:1.
- **The `_down_expanded` + combine path** is more intricate than gate_up — do gate_up first
  (the dominant cost), prove the win, then down.
- **gfx1151 untouched**: SoA-indexed is a new gfx11-dGPU path; the existing AoS indexed +
  the gfx1151 SoA standalone stay as-is.

## Increments
1. **gate_up only** — DONE (commit 637341f5). SoA-indexed gate_up kernel + transpose-on-
   load + dispatch (decode `_indexed` AND prefill `_indexed_batched`). Validated COHERENT
   on gfx1100 (kernel + transpose correct). **RESULT: A3B decode WASH — 102.0 (SoA) vs
   102.0 (AoS) tok/s.** The bench's +38-73% was on LARGE DENSE shapes (M=11008); A3B's
   per-expert gate_up is small (M=2*moe_intermediate ~1536), in the bench WASH regime
   (cf. qkv-kv M=512). SoA coalescing helps big dense GEMVs, not small per-expert MoE
   GEMVs — the indexed-context dilution risk, realized. Kept opt-in (default OFF).
   Gotchas: download+reupload transpose OOM'd -> in-place overwrite; decode uses the
   non-batched `_indexed` launcher (not `_indexed_batched`) -> wire both.
2. **down** — DO NOT PROCEED for A3B: same small per-expert shape, same wash expected.
   Only revisit if a model has large per-expert FFN (mi >> 4096) where SoA pays.
3. **Ship** — N/A (no win to ship). The kernel/transpose/dispatch remain as correct,
   reusable infra behind HIPFIRE_E8_SOA_EXPERTS for any future large-expert MoE.

**VERDICT: SoA coalescing is a dense-GEMV win that does NOT transfer to small-per-expert
MoE decode. The E8 102-vs-mq4-150 gap is not closeable via expert-GEMV layout on A3B —
it's the structural ceiling + g32 format (see [[project_gfx11_e8_port_2026_06_13]]).**

Owner notes: bench tree (`bench_e8_soa_correctness`, 5 variants × shapes + strip/LUT) is
the reusable measurement rig. See [[project_gfx11_e8_port_2026_06_13]] for the full
occupancy→compute→coalescing triangulation and the dGPU-vs-iGPU regime model.
