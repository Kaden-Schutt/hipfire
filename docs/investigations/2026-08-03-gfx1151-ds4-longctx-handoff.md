# gfx1151 DS4 long-context handoff

**Scope: gfx1151 (Strix Halo) only.** MI300X data appears here as *reference
only* — assume you have no access to it and cannot reproduce those numbers.

**Branch:** `ds4-cdna-test-fail` @ `a098bd274`, pushed to `origin`
(`warpfront/hipfire`). Everything below is committed; nothing is local-only.

---

## 1. The one thing to know first

**On gfx1151, every top-K improvement is currently opt-in. The default path is
still the O(N²) kernel.** This is the main hazard: you can run a long-context
benchmark, see terrible numbers, and conclude the engine is slow when you
simply did not turn anything on.

| lever | env | gfx1151 default | what it selects |
|---|---|---|---|
| — | `HIPFIRE_DEEPSEEK4_MAX_COMPRESS_POS` | **2048** | compressed-cache rows. **2048 rows = 8192 tokens of coverage at ratio 4.** Long context REQUIRES raising this. |
| F2-equiv | `HIPFIRE_DEEPSEEK4_INDEXER_TOPK_BOUNDED` | **OFF** | `=1` selects the bounded tile-merge kernel, O(N log²K) |
| G1 | `HIPFIRE_DEEPSEEK4_GFX1151_INDEXER_TOPK_TWOSTAGE` | **OFF** | `=1` selects the two-stage merge tree |
| — | `HIPFIRE_DEEPSEEK4_INDEXER_TOPK_TWOSTAGE_MIN` | 1024 | min `MAX_COMPRESS_POS` for two-stage to engage |
| — | `HIPFIRE_DEEPSEEK4_INDEXER_TOPK_SERIAL` | OFF | `=1` forces the portable serial kernel. Hard-caps at N=65536 (dynamic LDS = N bytes vs a 64 KiB limit). Avoid. |
| — | `HIPFIRE_DEEPSEEK4_INDEXER_TOPK_UNROLLED` | OFF | third gfx1151 variant, unmeasured |
| — | `HIPFIRE_DEEPSEEK4_GRAPH` | ON | hipGraph capture |

Dispatch precedence in `indexer_forward` (`crates/hipfire-arch-deepseek4/src/forward.rs`):
two-stage if `G1 && max_compress_pos() >= two_stage_min()`, else
`gpu.indexer_top_k_buf(...)`, which picks among the gfx1151 variants by the env
flags above, defaulting to the O(N²) one.

**A sane long-context invocation therefore looks like:**

```bash
export HIPFIRE_DEEPSEEK4_MAX_COMPRESS_POS=8192      # 32768 tokens of coverage
export HIPFIRE_DEEPSEEK4_GFX1151_INDEXER_TOPK_TWOSTAGE=1
```

Whether these should be defaults on gfx1151 is an open decision — G1 has passed
its exactness channel but has no model-level A/B on this part. That A/B is your
job (§4).

---

## 2. Access and build

Box `hipx`. **Four GPUs, and HIP's ordering does NOT match `rocm-smi`'s:**

```
HIP_VISIBLE_DEVICES=0 -> gfx1100 (7900 XTX, 25.8 GB)
HIP_VISIBLE_DEVICES=1 -> gfx1151 (Strix Halo 8060S, 103.1 GB)   <-- THIS ONE
HIP_VISIBLE_DEVICES=2 -> gfx1010
HIP_VISIBLE_DEVICES=3 -> gfx1030
```

`rocm-smi` lists the Strix Halo as GPU[3]. It is HIP device **1**. Always set
`HIP_VISIBLE_DEVICES=1` and confirm the probe/daemon prints `gfx1151`.

VRAM is a 96 GiB carve-out (103,079,215,104 B) of the 128 GB unified pool, plus
16 GiB GTT; the host only sees ~30 GiB of system RAM as a result.

**Use git, not rsync.** There are git remotes for the boxes already:
`hipx -> ssh://hipx/home/kaden/hipfire`, `mi300 -> ssh://mi300/root/hipfire`.
The previous session synced with `rsync` and hit a real bug because of it — see
§6. Pull from `origin` on the box.

```bash
ssh hipx
export PATH=$PATH:$HOME/.cargo/bin:/opt/rocm/bin
cd <checkout> && git fetch origin && git checkout ds4-cdna-test-fail && git pull
cargo build --release -p rdna-compute --example test_indexer_top_k_buf
cargo build --release --features deltanet -p hipfire-runtime --example daemon   # deltanet REQUIRED
cargo build --release --bin hipfire
cargo build --release -p hipfire-arch-deepseek4 --example ds4_longctx_probe
```

`--features deltanet` is required for the daemon example; without it the build
succeeds but produces no binary.

---

## 3. Models on / for this box

| artifact | bytes | sha256 | notes |
|---|---|---|---|
| `/home/kaden/.cache/hipfire-surgery/deepseek-v4-flash.mq2r` | 82,191,362,222 | `392325b5…4fc0e511` | **PREVIEW** generation. All existing gfx1151 evidence used this. |
| `/mnt/nas/kaden/models/hipfire-deepseek-v4-flash-0731/deepseek-v4-flash-0731.mq2r` | 82,191,359,851 | `cbf2bbcf…a9318cce` | **0731**, the current checkpoint. VERIFIED on NAS. |
| `…/deepseek-v4-flash-0731-mtp.mq2r` | 5,996,333,198 | `c123b976…52f1b248` | MTP sidecar, VERIFIED |
| `…/deepseek-v4-flash-0731-mtp.mq2lloyd` | 5,996,333,198 | `c123b976…52f1b248` | byte-identical to the mq2r sidecar |

**Preview and 0731 `.mq2r` differ by 2,371 bytes and are indistinguishable by
`ls -lh`. Always pass `--expected-model-sha256`.** Full inventory:
`/mnt/nas/kaden/models/DS4-ARTIFACT-MANIFEST.md`.

NAS is NFS-mounted on hipx at `/mnt/nas/kaden`. Loading 82 GB over NFS is slow;
copy to local disk first (560 GB free on `/`).

---

## 4. The open experiment

**Question:** does the two-stage top-K (G1) help *more* under Redline PM4
retained replay than without it?

**Why it matters:** the two-stage kernel is **launch-bound, not work-bound** —
7.3–13.5 µs per launch, near-flat in N (gfx1151: 0.0146 ms at N=512 with 2
launches, 0.1245 ms at N=262144 with 11 — 512× the data for 8.5× the time). It
adds 231 launches per token across 21 ratio-4 layers. Existing gfx1151 evidence
(`/home/kaden/ds4-gfx1151-evidence/2026-07-27-g8-single-ib-reorder/promotion-correctness-no-dspark/shadow15.json`)
shows PM4 cutting host dispatch cost from **2.016 µs to 0.04 µs per dispatch**.
So PM4 should make G1 look *better*. Every G1 number so far is graphs-off and
no-replay, i.e. conservative. Do not conclude anything about G1 from a
graphs-off benchmark.

**Tool:** `python3 -m tools.redline bench` (module under `tools/`, NOT a script
in `scripts/`). `tools/redline/product_bench.py`, argparse at line 1785.
`--transport pm4` is the Redline arm; `aql` is the comparison.
Do **not** use `python3 -m tools.redline lower` — that is a delegation shim to
`radiowave` / `redline_daemon_harness.py` and produces no throughput number.

**Script, already staged:** `/home/kaden/g1_pm4_ab.sh` on hipx. It runs four
cells: {pm4, aql} × {G1 off, G1 on}.

**Known blocker, not yet fixed.** The last run failed with:

```
PM4 preflight failed before benchmark warmup:
bench_decode context+iterations exceeds loaded physical_cap=8192
```

The script uses `--context 8192 --max-seq 8192 --iterations 128`, and the
preflight requires `context + iterations <= physical_cap`. **Fix: raise
`--max-seq` to 16384** (or lower `--context`). This is a `max_seq` sizing issue,
not a capability limit — the memory curve says this box can hold far more (§5).

Two other gotchas already hit: `--runs` must be ≥ 5, and `--skip-coherence` is
needed unless you want the serve_harness smoke (keep it on if you can — decoded
text is the only real coherence evidence).

Also note `--context 2048` is useless for this question: at ctx 2048 the live
N is 512, which takes the finalize identity fast path, so you would be measuring
the tree's launch overhead against no work. Use ctx ≥ 8192.

---

## 5. What is already established on gfx1151 (do not re-derive)

**Isolated top-K kernel, ms per decode step per layer, measured on this box:**

| N | ctx | default O(N²) | bounded (F2-equiv) | two-stage (G1) |
|---|---|---|---|---|
| 512 | 2,048 | 0.0023 | 0.0024 | 0.0146 |
| 1024 | 4,096 | 0.1598 | 0.0255 | 0.0198 |
| 2048 | 8,192 | 0.5593 | 0.0660 | 0.0250 |
| 8192 | 32,768 | 8.9869 | 0.2150 | 0.0407 |
| 32768 | 131,072 | 143.414 | 0.8101 | 0.0479 |
| 262144 | 1,048,576 | 10,494.75 | 6.4678 | 0.1245 |

21 ratio-4 layers, so per-token top-K cost is 21× these. At 1M context that is
**220 s → 136 ms → 2.6 ms**.

**Exactness:** `cargo run --release -p rdna-compute --example test_indexer_top_k_buf`
(model-free, no env, seconds). Currently `OVERALL: PASS` on gfx1151, 13 cases ×
4 arms. It was `FAIL` before `de6f88c02` — both gfx1151 kernels lacked an
eligibility filter and the NaN case left two output ranks **unwritten**.

**Memory:** `peak_bytes = 83,532,360,363 + 94,695.619 × compressed_cap`
(R²=0.999999912, fitted on MI300X but it is a model-shape property). Against
this box's 103,079,215,104 B that allows **cap 206,417 → ~825K tokens** with the
current F32 compressed cache, or ~1.28M with fwht4 KV. Memory is not the
constraint below ~800K; compute is.

**Existing PM4 baseline (preview weights, ctx 2048):** 28.078 tok/s, PM4 vs HIP
speedup 1.0973–1.1497 across 30 product-bench reports, median 1.1183. Corpus:
`/home/kaden/ds4-gfx1151-evidence/` (18,540 files, 3,315 JSON). **All of it is
ctx 2048 only** — nothing in that corpus touches the long-context region.
At ctx 2048 the top-K work is 0.14% of the step and on the identity path, so
none of the top-K levers change that 28.078 number. Do not expect them to.

**Projection for this box** (top-K measured, residual slope inferred — the
residual is the uncertainty, and one long-context run collapses it):

| ctx | default | +bounded | +two-stage |
|---|---|---|---|
| 8,192 | 20–21 | 25–27 | 26–27 |
| 32,768 | 4.2–4.4 | 19–24 | **20–26** |
| 131,072 | 0.3 | 9–16 | 11–22 |
| 1,048,576 | ~0 | 1.7–4.0 | 2.1–8.4 |

---

## 6. Pitfalls that have already cost time

1. **`rsync -a` + cargo.** `-a` preserves mtimes; cargo fingerprints on mtime.
   A synced file can land older than the existing build artifact, so cargo
   declares the crate fresh and **silently links a stale object**. This produced
   a build error contradicting the source on disk. Use git. If you must rsync,
   use `--no-times` or `touch` the files after.
2. **HIP device ordering ≠ `rocm-smi` ordering.** See §2.
3. **`tools/` vs `scripts/`.** The Redline bench is `python3 -m tools.redline bench`.
   `scripts/redline_daemon_harness.py` is the manual shadow-capture correctness
   route, a different thing.
4. **`--features deltanet`** for the daemon example, else no binary.
5. **Digest every model.** 2,371 bytes separate two 82 GB generations.

---

## 7. Reference only — MI300X (gfx942), not reproducible here

Recorded so you can sanity-check magnitudes, not to reproduce. The gfx942
equivalents of these levers (F1/F2/F3) are **default ON** there; the gfx1151
ones are not.

- Model-level A/B, 0731 weights, 18/18 cells: cap 2048 34.481→36.327,
  cap 8192 25.941→33.367, cap 32768 **13.043→25.961** tok/s (F2→F3).
- Cumulative against the pre-F2 default: 1.5× / 8.5× / **94.7×**.
- Redline PM4 does **not** run on gfx942 at all: `Pm4Architecture::from_device`
  (`crates/rdna-compute/src/replay.rs:61`) accepts gfx10/11/12 and errors
  otherwise. There is no GFX9/CDNA PM4 register map. So the PM4 question in §4
  is answerable *only* on gfx1151.

---

## 8. After §4, the next bottleneck is already identified

With top-K at ~2.6 ms/token at 1M context, decode becomes dominated by the O(N)
indexer `relu_score`. Its launch shape is `grid=[N,1,1] block=[h,1,1]`
(`crates/rdna-compute/src/attention.rs:9546`) — that is 262,144 blocks of 64
threads at 1M context. On MI300X it measured ~150× off the memory-bandwidth
bound, so it is launch/occupancy-shaped, not bandwidth-shaped. That is the next
kernel to look at, not top-K.
