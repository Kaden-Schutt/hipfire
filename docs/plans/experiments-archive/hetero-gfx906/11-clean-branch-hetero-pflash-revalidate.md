# Hetero PFlash re-validated on clean master branch (2026-05-25)

After PR #315 (HFQ4 MMQ family) and PR #323 (hetero PP=2 prereqs) merged
to master, rebuilt this work on a clean branch. **No code to port** —
bind_thread, `Gpu::init_with_device`, `multi_gpu.rs` are all in master.
Only `--target-device`/`--drafter-device` bench flags replayed (d9b08cb8).

- branch `feat/hetero-pflash-decode-v2`, 1 commit ahead of master `117c9c78`
- target: gfx906 (MI50), drafter: gfx1031 (RX 6700 XT), ROCm runtime 1.15
- target = `qwen3.5-9b.mq4-awq-gptq-f2-lmhead-a100.hfq`, drafter = `qwen3.5-0.8b.mq4`
- fixture niah_16k (10881 tok), keep_ratio 0.30, asym3 KV, maxgen 16

## Result: PFlash compress on drafter, AR prefill+decode on main

| config | compress | prefill | **TTFT** | × |
|---|---|---|---|---|
| no-PFlash, gfx906 | — | 24.15s | **24.15s** | 1.0 |
| PFlash solo, gfx906 | 3.34s | 5.19s | **8.52s** | 2.8 |
| PFlash hetero, drafter gfx1031 | 1.35s | 5.18s | **6.53s** | **3.7** |

Win is the compress: gfx1031 runs the 0.8B drafter forward 2.5× faster
than gfx906 (1.35 vs 3.34s). Prefill (3265 kept tok @ 630 tok/s) and
decode (~51 tok/s AR) stay on gfx906. Reproduces d9b08cb8's 6.4s.

Solo/hetero pick slightly different kept-span sets (13 vs 12 ranges,
diff compressed_md5) — separate Gpu handle, expected; needle region
kept either way. NIAH substring miss = maxgen=16 cap, not correctness.

This is prefill-compress, not spec-decode. Decode = plain AR. Verified
host-side handoff (Vec<u32> kept IDs), drafter handle dropped pre-KV.
