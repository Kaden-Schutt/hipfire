//! G7: the FUSED routed GEMV — the one that decodes escha's trellis code
//! inside the matmul instead of reading an expanded copy — must produce
//! **bit-identical** f32 output to
//!
//!   1. the same GEMV reading the **F16 expert store**
//!      (`escha_decode_tiles` -> `escha_bare_to_f16`, i.e. the exactly-decoded
//!      weights with nothing re-quantised), and
//!   2. a CPU replica built from **`escha_ref`**, the frozen oracle, summed in
//!      the kernel's own order.
//!
//! Run:
//!   cargo run --release -p rdna-compute --example test_escha_native_gemv_gpu_vs_cpu
//!
//! # Why the comparison is against F16 and NOT against the Q8_0 store
//!
//! The Q8_0 expert store is a LOSSY re-quantisation of the decoded weight
//! (that loss is the dominant term in the G4 block gate: 2.633e-4 max against
//! the F32 arm's 1.828e-4). The fused kernel uses the decoded fp16 value
//! itself. Asserting equality against Q8_0 would therefore be asserting
//! something false; the arm that holds the same values the fused kernel
//! decodes is the F16 store, and that is what arm 1 compares against.
//!
//! # Why arm 2 exists as well
//!
//! Arm 1 alone would be satisfied by two GPU paths that are wrong in the same
//! way — they share `escha_decode_tiles`' idea of where a weight lives. Arm 2
//! closes that: it takes the weights from `escha_ref::reconstruct`, the CPU
//! oracle that has been frozen since commit 11 and that every bit-exact claim
//! in this port rests on, and it re-does the summation in the kernel's exact
//! order (lane-strided partial sums, the accumulator count the variant uses,
//! the `__shfl_down` ladder). So arm 2 checks the DECODE against the oracle
//! and the ACCUMULATION against the transcription contract at once.
//!
//! Arm 2 also reports WHICH f32 contraction the compiler chose (fused
//! multiply-add or separate multiply and add) rather than assuming one: the
//! two differ in the last bit and the gate is an exact-equality gate, so
//! guessing would make it flaky rather than wrong.
//!
//! # Coverage
//!
//! Both shipped A3B projections, i.e. both trellis orders AND both accumulator
//! variants, on the SHIPPED golden code (not synthetic):
//!
//! | projection | ic (K) | oc (M) | trellis K | variant       |
//! |---|---|---|---|---|
//! | `gate_up`  | 2048 | 1024 | 2 | narrow (K > 1536) |
//! | `down`     |  512 | 2048 | 3 | wide   (K <= 1536) |
//!
//! `n_exp = 4` experts, each a distinct rotation of the golden code, and eight
//! slots whose ids repeat and are out of order — so a kernel that ignored
//! `expert_ptrs[topk_indices[krank]]`, or that mixed slots up, fails here
//! rather than passing on an accidentally-uniform fixture.
use hipfire_quantize::escha_ref;
use rdna_compute::{DType, Gpu, GpuTensor};

/// xorshift64* — the same inline PRNG the other GPU parity examples in this
/// crate use (no `rand` dependency).
struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }
    /// Activations in roughly the range the H128 input transform emits.
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

/// How the kernel folds one product into an accumulator. HIP compiles with
/// `-ffp-contract=fast` by default, so `acc += w * x` becomes a single-rounding
/// `v_fmac_f32`; but that is a toolchain default, not a guarantee, and the two
/// forms differ in the last bit. The gate determines which one the build
/// actually produced instead of assuming — see `main`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Fold {
    Fma,
    MulAdd,
}
impl Fold {
    #[inline(always)]
    fn apply(self, acc: f32, w: f32, x: f32) -> f32 {
        match self {
            Fold::Fma => w.mul_add(x, acc),
            Fold::MulAdd => acc + w * x,
        }
    }
}

/// The `__shfl_down(sum, 16/8/4/2/1)` ladder both kernel variants end with,
/// evaluated for lane 0 — the only lane whose result is stored.
///
/// Every value lane 0 consumes comes from an in-range lane, so the ladder is
/// exactly a balanced binary tree over the 32 partial sums and the
/// out-of-range-returns-self behaviour of `__shfl_down` never enters.
fn warp_reduce(mut lanes: [f32; 32]) -> f32 {
    let mut width = 16;
    while width > 0 {
        for t in 0..width {
            lanes[t] += lanes[t + width];
        }
        width >>= 1;
    }
    lanes[0]
}

/// CPU replica of `escha_gemv_native_*_moe_k8_indexed_batched` for one output
/// row: the NARROW form (one accumulator, `K > 1536`).
///
/// `w_row[i]` is the weight for contraction index `i`, already fp16-decoded.
fn cpu_row_narrow(w_row: &dyn Fn(usize) -> f32, x: &[f32], k: usize, fold: Fold) -> f32 {
    let blocks = k / 32;
    let mut lanes = [0.0f32; 32];
    for (t, lane) in lanes.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for bi in 0..blocks {
            let i = bi * 32 + t;
            sum = fold.apply(sum, w_row(i), x[i]);
        }
        *lane = sum;
    }
    warp_reduce(lanes)
}

/// CPU replica of the WIDE form (`K <= 1536`): four interleaved accumulators
/// folded `(acc0 + acc1) + (acc2 + acc3)`, tail blocks landing in `acc[t]`.
fn cpu_row_wide(w_row: &dyn Fn(usize) -> f32, x: &[f32], k: usize, fold: Fold) -> f32 {
    let blocks = k / 32;
    let quads = blocks >> 2;
    let tail = blocks & 3;
    let mut lanes = [0.0f32; 32];
    for (t, lane) in lanes.iter_mut().enumerate() {
        let mut acc = [0.0f32; 4];
        for q in 0..quads {
            let bi = q << 2;
            for (s, a) in acc.iter_mut().enumerate() {
                let i = (bi + s) * 32 + t;
                *a = fold.apply(*a, w_row(i), x[i]);
            }
        }
        for s in 0..tail {
            let bi = (quads << 2) + s;
            let i = bi * 32 + t;
            let contrib = w_row(i) * x[i];
            // Verbatim from the kernel, `t == 3` included in its unreachability.
            if s < 3 {
                acc[s] += contrib;
            }
        }
        *lane = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    }
    warp_reduce(lanes)
}

/// The `[n_exp]` u64 weight-base table the indexed GEMVs index with
/// `expert_ptrs[topk_indices[krank]]`, packed into an F32 tensor (2 f32 per
/// pointer) exactly as the model loader packs it.
fn ptr_table(gpu: &Gpu, owner: &GpuTensor, n_exp: usize, stride_bytes: usize) -> GpuTensor {
    let bytes: Vec<u8> = (0..n_exp)
        .map(|e| owner.buf.as_ptr() as u64 + (e * stride_bytes) as u64)
        .flat_map(|p| p.to_ne_bytes())
        .collect();
    gpu.upload_raw(&bytes, &[2 * n_exp]).expect("ptr table")
}

struct Case {
    name: &'static str,
    fixture: &'static str,
    ic: usize,
    oc: usize,
    trellis_k: usize,
}

fn main() {
    let cases = [
        Case {
            name: "gate_up",
            fixture: "packed_gu_e0_k2.i16",
            ic: 2048,
            oc: 1024,
            trellis_k: 2,
        },
        Case {
            name: "down",
            fixture: "packed_down_e0_k3.i16",
            ic: 512,
            oc: 2048,
            trellis_k: 3,
        },
    ];

    let n_exp = 4usize;
    let slots = 8usize;
    // Repeating, out-of-order ids: an implementation that dropped the indirection
    // (or that used `krank` where it meant `ids[krank]`) cannot pass.
    let ids: [i32; 8] = [3, 0, 2, 1, 0, 3, 1, 2];

    let mut gpu = Gpu::init().expect("gpu");
    let mut failures = 0usize;

    for case in &cases {
        let path = format!(
            "{}/../hipfire-quantize/tests/data/escha/{}",
            env!("CARGO_MANIFEST_DIR"),
            case.fixture
        );
        let raw = std::fs::read(&path).expect("run fetch-goldens.sh first");
        let base: Vec<i16> = raw
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        let words_per_expert = (case.ic / 16) * (case.oc / 16) * 16 * case.trellis_k;
        assert_eq!(
            base.len(),
            words_per_expert,
            "{}: fixture is {} i16, expected {words_per_expert}",
            case.name,
            base.len()
        );

        // Four DIFFERENT experts out of one fixture: rotating the code stream
        // keeps it a valid trellis code (every tile is still 16*K words) while
        // making no two experts share a single tile.
        let experts: Vec<Vec<i16>> = (0..n_exp)
            .map(|e| {
                let shift = (e * 16 * case.trellis_k * 37) % words_per_expert;
                let mut v = base.clone();
                v.rotate_left(shift);
                v
            })
            .collect();

        // ── the two device-side weight stores ────────────────────────────────
        // NATIVE: the code bytes, verbatim, one contiguous buffer with the
        // per-expert slot stride the packed loader uses.
        let code_bytes: Vec<u8> = experts
            .iter()
            .flat_map(|e| e.iter().flat_map(|v| v.to_le_bytes()))
            .collect();
        let d_code = gpu
            .upload_raw(&code_bytes, &[code_bytes.len()])
            .expect("upload code");

        // F16: the production store path — `escha_decode_tiles` then
        // `escha_bare_to_f16` — into the same kind of packed buffer.
        let d_f16 = gpu
            .alloc_tensor(&[n_exp * case.ic * case.oc], DType::F16)
            .expect("alloc f16 store");
        let bare = gpu
            .alloc_tensor(&[case.ic * case.oc], DType::F16)
            .expect("alloc bare");
        let stage = gpu
            .alloc_tensor(&[words_per_expert], DType::F16)
            .expect("alloc stage");
        for (e, code) in experts.iter().enumerate() {
            let bytes: Vec<u8> = code.iter().flat_map(|v| v.to_le_bytes()).collect();
            gpu.hip.memcpy_htod(&stage.buf, &bytes).expect("stage code");
            gpu.escha_decode_tiles(
                &stage,
                &bare,
                case.ic as u32,
                case.oc as u32,
                case.trellis_k as u32,
            )
            .expect("decode");
            let slot = d_f16.sub_offset(e * case.ic * case.oc, case.ic * case.oc);
            gpu.escha_bare_to_f16(&bare, &slot, case.ic, case.oc)
                .expect("bare->f16");
        }

        // ── pointer tables (the same [n_exp] u64 packing the loader builds) ──
        let code_ptrs = ptr_table(&gpu, &d_code, n_exp, words_per_expert * 2);
        let f16_ptrs = ptr_table(&gpu, &d_f16, n_exp, case.ic * case.oc * 2);

        let id_bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
        let d_ids = gpu.upload_raw(&id_bytes, &[slots]).expect("ids");

        // ── activations: one distinct vector per slot ────────────────────────
        let mut rng = Rng(0x5EED_0000_0000_0001 ^ (case.ic as u64));
        let x: Vec<f32> = (0..slots * case.ic).map(|_| rng.next_f32()).collect();
        let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        let d_x = gpu
            .upload_raw(&x_bytes, &[slots * case.ic])
            .expect("upload x");
        let d_y_native = gpu
            .alloc_tensor(&[slots * case.oc], DType::F32)
            .expect("y native");
        let d_y_f16 = gpu
            .alloc_tensor(&[slots * case.oc], DType::F32)
            .expect("y f16");

        gpu.escha_gemv_native_moe_k8_indexed_batched(
            &code_ptrs,
            &d_ids,
            &d_x,
            &d_y_native,
            case.oc,
            case.ic,
            slots,
            case.trellis_k as u32,
        )
        .expect("native gemv");
        gpu.escha_gemv_f16_moe_k8_indexed_batched(
            &f16_ptrs, &d_ids, &d_x, &d_y_f16, case.oc, case.ic, slots,
        )
        .expect("f16 gemv");

        let y_native = gpu.download_f32(&d_y_native).expect("dl native");
        let y_f16 = gpu.download_f32(&d_y_f16).expect("dl f16");

        // ── arm 1: fused native vs the F16 expert store, bit for bit ─────────
        let diff_f16 = y_native
            .iter()
            .zip(&y_f16)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let worst = y_native
            .iter()
            .zip(&y_f16)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!(
            "{}: fused-native vs F16 store — {diff_f16} differing floats of {} (max |delta| {worst:e})",
            case.name,
            y_native.len()
        );
        if diff_f16 != 0 {
            failures += 1;
        }

        // ── arm 2: fused native vs escha_ref, summed in the kernel's order ───
        // `reconstruct` is IN-major `[ic, oc]`; the GEMV's output row `o`
        // contracts over `i`, so the weight is `bare[i * oc + o]`.
        let decoded: Vec<Vec<u16>> = experts
            .iter()
            .map(|c| escha_ref::reconstruct(c, case.ic, case.oc, case.trellis_k))
            .collect();
        let wide = case.ic <= 1536;

        // Which f32 contraction did this build produce? Decide it from ONE
        // element rather than assuming, then hold the whole gate to it.
        let probe_row = |fold: Fold, slot: usize, o: usize| -> f32 {
            let w = &decoded[ids[slot] as usize];
            let w_row = |i: usize| f16_to_f32(w[i * case.oc + o]);
            let xs = &x[slot * case.ic..(slot + 1) * case.ic];
            if wide {
                cpu_row_wide(&w_row, xs, case.ic, fold)
            } else {
                cpu_row_narrow(&w_row, xs, case.ic, fold)
            }
        };
        let fold = if probe_row(Fold::Fma, 0, 0).to_bits() == y_native[0].to_bits() {
            Fold::Fma
        } else {
            Fold::MulAdd
        };
        println!("{}: CPU replica folding as {fold:?}", case.name);

        let mut diff_ref = 0usize;
        let mut worst_ref = 0.0f32;
        for slot in 0..slots {
            let w = &decoded[ids[slot] as usize];
            let xs = &x[slot * case.ic..(slot + 1) * case.ic];
            for o in 0..case.oc {
                let w_row = |i: usize| f16_to_f32(w[i * case.oc + o]);
                let want = if wide {
                    cpu_row_wide(&w_row, xs, case.ic, fold)
                } else {
                    cpu_row_narrow(&w_row, xs, case.ic, fold)
                };
                let got = y_native[slot * case.oc + o];
                if want.to_bits() != got.to_bits() {
                    diff_ref += 1;
                    worst_ref = worst_ref.max((want - got).abs());
                }
            }
        }
        println!(
            "{}: fused-native vs escha_ref (kernel order) — {diff_ref} differing floats of {} \
             (max |delta| {worst_ref:e})",
            case.name,
            y_native.len()
        );
        if diff_ref != 0 {
            failures += 1;
        }

        // A gate that compared two all-zero buffers would report 0 differences
        // and prove nothing. Assert the output is actually a GEMV result.
        let nonzero = y_native.iter().filter(|v| **v != 0.0).count();
        assert!(
            nonzero > y_native.len() / 2,
            "{}: only {nonzero} of {} outputs are non-zero — the fixture or the launch is \
             degenerate and the equality above is vacuous",
            case.name,
            y_native.len()
        );

        for t in [
            d_code, d_f16, bare, stage, code_ptrs, f16_ptrs, d_ids, d_x, d_y_native, d_y_f16,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }

    assert_eq!(failures, 0, "G7: the fused native GEMV is not bit-exact");
    println!("G7 PASS");
}
