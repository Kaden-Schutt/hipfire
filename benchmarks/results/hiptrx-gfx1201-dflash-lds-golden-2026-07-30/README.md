# 27B DFlash golden run — LDS-staged weight GEMMs, gfx1201

Winning config: **`HIPFIRE_HFQ4G256_LDSSTAGE=1`, `HIPFIRE_Q8_PREFILL_M4=0`, `--ctx 2048`** → **277.97 tok/s** decode, **+9.06%** over the same binary with LDS staging off.

| arm | decode tok/s (3 fresh reps) | median | prefill | τ | accept | tokens |
|---|---|---:|---:|---:|---:|---:|
| LDS off | 255.53 / 254.88 / 254.84 | 254.88 | 453.5 | 11.3846 | 0.7590 | 162 |
| **LDS on** | **278.24 / 277.97 / 276.92** | **277.97** | 466.6 | 11.3846 | 0.7590 | 162 |
| ratio | | **1.0906×** | 1.029× | | | |

τ, accept, and token count are identical to four decimals across every rep, so the arms did the same work and the delta is throughput. Spreads 0.27% / 0.47%.

Reproduces a prior +9.1% recorded on an older tree to **four digits** (1.0906 vs 1.0904), and the LDS-off arm lands 0.02% from the published golden baseline of 254.83 tok/s (`v0.2.0+3730b58`).

## Identity

| item | value |
|---|---|
| host / GPU | hiptrx · AMD Radeon AI PRO R9700 · gfx1201 (RDNA4) · `HIP_VISIBLE_DEVICES=3` |
| tree | `bcd1ef0082e7f7f7a1eff9252da5d76e17d042b0`, 0 dirty |
| binary sha256 | `a107cf8640abf1d7cb1b3aeec26f33a07a01b00abdab53956c29b6a7b83c1ab5` (13 569 168 B) |
| trunk sha256 | `86a5f80fd29d545abb1093dead242725ced6d68b8607c6d566d897b1a82442dc` (14 984 158 208 B) |
| draft md5 | `204c4c4ceab30cb9ebc118fa9d59a446` (919 401 472 B) |
| prompt md5 | `253c7ac50857fe6d0e10fb0d2c5e35c0` — `benchmarks/prompts/merge_sort_thinking_off.txt` |
| HIP / ROCm | 7.14.60850 · `/opt/rocm/core-7.14` |

One binary in both arms; LDS and M4 are env flags, never rebuilds. Fixtures identified by digest, not filename.

## Reproduce

```bash
export PATH=/opt/rocm/core-7.14/bin:$PATH
cargo build --release --locked --features deltanet \
  -p hipfire-runtime --example dflash_spec_demo

export HIP_VISIBLE_DEVICES=3
export HIPFIRE_Q8_PREFILL_M4=0        # M4 costs ~4% on the spec path
export HIPFIRE_HFQ4G256_LDSSTAGE=1    # unset for the baseline arm

./target/release/examples/dflash_spec_demo \
  --target ~/.hipfire/models/qwen3.6-27b-awq.mq4 \
  --draft  ~/.hipfire/models/qwen36-27b-dflash-mq4.hf4 \
  --prompt-file benchmarks/prompts/merge_sort_thinking_off.txt \
  --max 256 --temp 0.0 --no-chatml --kv-mode q8 --ctx 2048
```

Discard one warm-up run: the first LDS-on process JIT-compiles the two staged
kernels. Adaptive B on purpose — do **not** add `--block-size N --no-adaptive-b`,
that is a different regime (~2.3× apart) and not comparable to this line.

Routing proof, since the flag is silent when it doesn't take: an LDS-on run must
leave both objects in `$CWD/.hipfire_kernels/gfx1201/` (not under `HOME`) —
`gemm_gate_up_hfq4g256_wmma_gfx12_ldsstage.hsaco` (17 448 B) and
`gemm_hfq4g256_residual_wmma_gfx12_ldsstage.hsaco` (15 400 B).

## Caveats

- **Not bit-exact.** LDS staging reorders f32 K-accumulation (8 waves × disjoint
  64-K slices). τ/accept/output matched here, but that is one prompt at temp 0.0,
  not a quality result. The flag stays opt-in for that reason.
- **Not a serve number.** This is `dflash_spec_demo`, an isolated kernel-path
  instrument. The user-facing serve delta for this flag measured **+3.9%**.
- **`--ctx` is load-bearing.** Same 162 tokens at ctx 4096 gives 273.55 (LDS on)
  and 251.97 (LDS off) — the allocation itself costs ~1.2–1.6%, because under
  hipGraph capture `max_ctx_len` is baked as `kv_cache.physical_cap` to size the
  attention LDS for the worst case.

Raw logs: `ldson_r{1,2,3}.log`, `ldsoff_r{1,2,3}.log` (unedited stdout+stderr).
