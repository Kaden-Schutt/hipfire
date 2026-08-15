# Qwen3.8 27B — Qwen3.6 27B parity and reasoning serve proof

**Lifecycle:** `historical`. This is fixture-bound measured evidence from the
Qwen3.8 reasoning-effort bring-up. It is **not** a current default, an automatic
baseline, a product claim, or an admission decision.

**Disposition:** Qwen3.8-27B met the predeclared ±3% matrix-equivalence band
against Qwen3.6-27B on gfx1100 and gfx1201. Its sampled `reasoning_effort=xhigh`
battery and eight-turn coding session completed coherently through the native
serve path. The matrix result is one fresh process per model/architecture with
three samples per row; the maintainer accepted that as sufficient for this
support proof, but it is not a three-fresh-process promotion result.

**Source:** `glimmer-opt` @ `aca070c1b72a23c612aa70b2fcf11ba53ab700e2`
(reasoning propagation landed in `3c8e6141c544b25f9c98f6d7fd8c825b18654ab9`).

---

## Fixture identity

| field | value |
|---|---|
| Qwen3.6 model | `/home/kaden/.hipfire/models/qwen3.6-27b.mq4`; 14,979,312,640 bytes; md5 `9a6acdc49bcaa6a7b52ac161444cb769` |
| Qwen3.8 model | `/home/kaden/.hipfire/models/qwen3.8-27b.mq4`; 14,980,361,216 bytes; md5 `129909ad0fed21dcf72b5b9225e85604` |
| loaded shape | `qwen3_5`, dim 5120, 64 layers, vocab 248320 for both |
| native CLI md5 | `0d81b067576dc042f955fcd431a0548c` |
| daemon md5 | `143d0084b0d5b3c4980e4ccbe622c65b` (identical on both hosts) |
| `serve_harness.py` md5 | `594dedfda9843a2c4f2a1d3d19677159` |
| coding session fixture | `/home/kaden/mv/session_coding.json`; md5 `c0d470288bde3f1e54e4bba04da8f8a2` (byte-identical staged copy on hiptrx) |
| ROCm | HIP 7.14.60850; AMD clang 23, git `46fcb...+PATCHED` on both hosts |
| gfx1201 | hiptrx GPU 0/1, 32,624 MiB reported VRAM |
| gfx1100 | hipx GPU 0, 24,560 MiB reported VRAM |

Graphs, MTP, DFlash, n-gram speculation, and every other speculative path were
off. The KV mode was Q8. Matrix runs used contiguous KV. Full-context serve
used contiguous KV on gfx1201 and VMM KV on gfx1100; see the allocation finding
below.

## Direct `hipfire bench --matrix` protocol

Each arm was a fresh daemon process using the native CLI directly:

```bash
hipfire bench --matrix \
  --pp 128,512,2048,8192 \
  --ctx 128,512,2048,8192 \
  --tg 128 --runs 3 --warmups 10 \
  --kv-mode q8 --kv-backend contiguous \
  --spec off --json <model>
```

The values below are each row's median of three in-process samples. Positive
delta means Qwen3.8 was faster.

### gfx1201

| phase | prompt/context tokens | Qwen3.6 tok/s | Qwen3.8 tok/s | delta |
|---|---:|---:|---:|---:|
| prefill | 128 | 769.5 | 772.4 | +0.377% |
| prefill | 512 | 754.7 | 759.0 | +0.570% |
| prefill | 2048 | 736.9 | 737.2 | +0.041% |
| prefill | 8192 | 660.7 | 662.7 | +0.303% |
| decode | 128 | 36.1733 | 36.1818 | +0.024% |
| decode | 512 | 36.1305 | 36.1458 | +0.042% |
| decode | 2048 | 35.9816 | 35.9945 | +0.036% |
| decode | 8192 | 34.7124 | 34.7216 | +0.027% |

### gfx1100

| phase | prompt/context tokens | Qwen3.6 tok/s | Qwen3.8 tok/s | delta |
|---|---:|---:|---:|---:|
| prefill | 128 | 724.8 | 722.7 | -0.290% |
| prefill | 512 | 887.6 | 883.3 | -0.484% |
| prefill | 2048 | 854.9 | 852.2 | -0.316% |
| prefill | 8192 | 748.1 | 745.0 | -0.414% |
| decode | 128 | 46.0451 | 45.9409 | -0.226% |
| decode | 512 | 45.8918 | 45.8399 | -0.113% |
| decode | 2048 | 45.3728 | 45.1657 | -0.456% |
| decode | 8192 | 42.7993 | 42.8074 | +0.019% |

All 16 comparisons were within ±0.57%, far inside the predeclared ±3%
equivalence band. The evidence is intentionally limited to the first completed
fresh arm per model/architecture after the maintainer accepted it; partially
started second arms are excluded.

## Sampled reasoning serve protocol

The native `scripts/serve_harness.py` route was used directly; no demo or
hand-written generation harness was used. The final runs used the models'
full native context and full allowed output instead of imposing a small
reasoning cap:

```bash
python3 scripts/serve_harness.py \
  --model <model> \
  --kv q8 --kv-backend <contiguous-or-vmm> \
  --mtp off --dflash off --speculation off \
  --thinking-effort xhigh \
  --max-seq 262144 --max-tokens 81920 \
  --sampling recipe:general --seed 7 \
  --mode <battery-or-session> [--session session_coding.json]
```

`recipe:general` resolved to temperature 1.0, top-p 0.95, top-k 20,
min-p 0.0, and presence penalty 0.0. The thinking budget was uncapped. The
large limits were load/request allowances: prompts were not padded to 262,144
tokens and generations self-terminated before 81,920 tokens.

### Five-prompt battery

| host/arch | KV backend | model | turns stopped | empty | runaway | detector flags | avg prefill tok/s | avg decode tok/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| hiptrx/gfx1201 | contiguous | Qwen3.6 | 5/5 | 0 | 0 | 3 | 482.8 | 36.1 |
| hiptrx/gfx1201 | contiguous | Qwen3.8 | 5/5 | 0 | 0 | 1 | 482.2 | 36.2 |
| hipx/gfx1100 | VMM | Qwen3.6 | 5/5 | 0 | 0 | 3 | 402.2 | 45.8 |
| hipx/gfx1100 | VMM | Qwen3.8 | 5/5 | 0 | 0 | 0 | 400.8 | 45.9 |

The decoded visible answers were read, not inferred from counters. Both models
returned correct merge functions, the 210-mile arithmetic answer, accurate
three-sentence season explanations, coherent four-sentence lighthouse stories,
and five-item maintainability lists. The detector flags were repetition in
Qwen3.6's long hidden reasoning (and one Qwen3.8 gfx1201 prose reasoning trace),
not visible-answer attractors; the visible text was coherent and obeyed the
requested shapes.

### Eight-turn `session_coding.json`

The session exercises architecture recall, a streaming BLAKE3 function, memory
analysis, capped rayon parallelism, FastCDC comparison, a topic switch, and two
long-range retrieval checks.

| host/arch | model | turns stopped | empty/runaway/attractor | final ctx | retrieval | avg prefill tok/s | avg decode tok/s | harness exit |
|---|---|---:|---:|---:|---|---:|---:|---:|
| hiptrx/gfx1201 GPU 1 | Qwen3.8 | 8/8 | 0/0/0 | 24,715 | 6/6 expected terms | 254.3 | 33.4 | 0 |
| hiptrx/gfx1201 GPU 0 | Qwen3.6 | 8/8 | 0/0/0 | 18,653 | 5/6 expected terms | 225.5 | 33.9 | 1 |
| hipx/gfx1100 | Qwen3.6 | 8/8 | 0/0/0 | 19,563 | 4/6 expected terms | 216.0 | 40.3 | 1 |

Qwen3.8 recalled `hash_file`, returned a streaming BLAKE3
`std::io::Result<[u8; 32]>` implementation, and included all of `dedupe`,
`hash`, `Result`, `rayon`, and `chunk` where the fixture required them. Its
session had no empty output, runaway, attractor, `<|...|>` leak, or retrieval
miss.

Both Qwen3.6 runs remained coherent and completed every turn, but the strict
lexical gate rejected them because their retrieval answers omitted the exact
word `dedupe` while still discussing the requested hash/deduplication function.
That comparator miss is recorded rather than hidden or rerun away.

## Findings exposed by full limits

1. An earlier discarded `max_tokens=640` trial was too small for uncapped
   `xhigh` reasoning. Qwen3.6 exhausted the allowance inside `<think>` on four
   of five battery prompts, and the daemon correctly rejected each with
   `open think span at end of generation`. None of those rows is used above.
2. Loading `max_seq=262144` with contiguous Q8 KV on the 24 GB gfx1100 failed
   at `hipMalloc` with 960 MiB free of 24,560 MiB. The final gfx1100 runs kept
   the full context and used the supported VMM KV backend rather than reducing
   coverage. gfx1201 fit the full contiguous allocation.
3. Qwen3.8 had no MTP sidecar (`MTP head loaded lines=0`); Qwen3.6's sidecar was
   present but MTP and speculation were explicitly off. The comparison is
   ordinary sampled AR serve.

## Evidence locations

- hiptrx matrix: `/home/kaden/qwen38-proof/results/matrix-gfx1201/`
- hipx matrix: `/home/kaden/qwen38-proof/results/matrix-gfx1100/`
- hiptrx serve: `/home/kaden/qwen38-proof/results/serve-sampled-gfx1201/`
- hipx serve: `/home/kaden/qwen38-proof/results/serve-sampled-gfx1100/`

The directories retain row JSON, daemon logs, CLI stdout/stderr, request md5s,
prompt md5s, and daemon-binary md5s for the measurements summarized here.
