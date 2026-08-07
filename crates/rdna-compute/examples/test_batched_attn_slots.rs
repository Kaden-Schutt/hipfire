// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Task 7 — the correctness gate for the whole multi-slot batched-attention
// sub-project (SP1). Tasks 4-6 ported four `_slots` kernel entry points
// (LDS decode, Q8 tile, asym3 tile, Q8 flash prefill) to a shared
// `KvSlotDesc` addressing scheme, verified ONLY against the legacy
// null-descriptor path. This harness is the first thing to exercise the
// multi-slot path itself.
//
// Three layers, each run for both KV modes (Q8_0 and asym3) wherever the
// kernel family supports both:
//   1. Golden equivalence  — per-slot legacy reference vs. one batched
//      multi-slot launch, tolerance-based (tiling reorders accumulation).
//   2. Cross-slot isolation — every OTHER slot's KV poisoned with NaN;
//      the target slot's output must stay finite and unchanged. The
//      sharpest instrument here: a wrong k_base/v_base or stride pulls a
//      NaN straight through softmax, unlike a merely "plausible but wrong"
//      value that only a tolerance check would catch.
//   3. Adversarial shapes — ragged tiles, mixed M (0/1/3/8), wildly unequal
//      per-slot context, slot counts 1..8, sub-tile-size seq_len, non-BR/BC
//      multiples, GQA 6:1 and 8:1, and forced partials sub-batching (the
//      class of bug 566ce7d1 fixed: correct at small batch, wrong once
//      global-row bookkeeping is required across sub-batch chunks).
//
// Kernel families covered:
//   - attention_flash_q8_0_batched_masked_slots   (Task 5, Q8 tile, no LDS cap)
//   - attention_flash_asym3_batched_masked_slots  (Task 6, asym3 tile)
//   - attention_q8_0_kv_batched_masked_slots      (Task 4, LDS decode/verify;
//     context capped well under the ~16k LDS ceiling this kernel has always
//     had, multi-slot or not)
//   - attention_q8_0_flash_prefill_slots          (Task 6, BR/BC-tiled
//     prefill; the one kernel where a single tile can span several query
//     rows of one slot, so ragged-tile-at-slot-boundary is its own hazard)
//
// Negative control: set NEGATIVE_CONTROL=1 to corrupt every slot's
// descriptor to slot 0's before the multi-slot launch (simulating "reverted
// a descriptor to slot 0"). Expected to fail loudly and immediately. See
// task-7-report.md for the captured failure.
//
// The corruption is applied to the DEVICE-side descriptor table only (what
// the candidate multi-slot kernel reads) — never to the host-side `descs`
// that `run_*_reference` uses to slice the very same arenas for its
// per-slot legacy-kernel reference. An earlier version of this control
// corrupted both, which meant the reference read the identical wrong slab
// as the candidate and the numeric comparison could never observe a fault
// (see maybe_corrupt's doc comment). `main()` also restricts which shapes
// run under NEGATIVE_CONTROL=1 to those with a uniform seq_len across
// slots, so the device-side `positions[row]+1 <= desc.seq_len` guard added
// in Tasks 5/6 does not trip and abort the process before the numeric
// comparison gets a chance to run.
//
// Positive poison control: the isolation layer (layer 2 above) asserts a
// NaN-poisoned NEIGHBOUR does not corrupt the target slot's output. That is
// silent if the poison mechanism itself were inert. `test_poison_is_live`
// poisons the TARGET slot itself and asserts its output DOES go
// non-finite, proving the poison is live rather than assumed.

use rdna_compute::kv_slots::{build_arena, build_asym3_k_arena, build_tiles, KvSlotDesc};
use rdna_compute::{DType, Gpu, GpuTensor};

// ─────────────────────────── shared helpers ────────────────────────────

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

/// Pack KvSlotDesc records byte-identically to kernels/src/kv_slot_desc.h.
fn pack_descs(descs: &[KvSlotDesc]) -> Vec<u8> {
    let mut out = Vec::with_capacity(descs.len() * 24);
    for d in descs {
        out.extend_from_slice(&d.k_base.to_ne_bytes());
        out.extend_from_slice(&d.v_base.to_ne_bytes());
        out.extend_from_slice(&d.seq_len.to_ne_bytes());
        out.extend_from_slice(&d.cap.to_ne_bytes());
    }
    out
}

/// build_arena is called once for K and once for V (different strides);
/// each call independently numbers its own base offsets starting from 0
/// (since build_arena sets `k_base == v_base == base` within one call).
/// Merge the two so the final descriptor's k_base points into the K arena
/// and v_base into the V arena.
fn merge_descs(k_descs: &[KvSlotDesc], v_descs: &[KvSlotDesc]) -> Vec<KvSlotDesc> {
    k_descs
        .iter()
        .zip(v_descs)
        .map(|(k, v)| KvSlotDesc {
            k_base: k.k_base,
            v_base: v.k_base,
            seq_len: k.seq_len,
            cap: k.cap,
        })
        .collect()
}

/// Replicates hipfire_runtime::llama::KvCache::gen_givens_angles locally —
/// rdna-compute (where this example lives) does not depend on
/// hipfire-runtime, and the specific angle values don't matter for an
/// addressing-correctness test: candidate and reference are always given
/// the SAME cos/sin tensor, so any deterministic, non-degenerate angle set
/// exercises the same question.
fn gen_givens_angles(seed: u32, n_blocks: usize) -> (Vec<f32>, Vec<f32>) {
    let mut state = seed;
    let mut cos_vals = Vec::with_capacity(n_blocks);
    let mut sin_vals = Vec::with_capacity(n_blocks);
    for _ in 0..n_blocks {
        state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fff_ffff;
        let angle = (state as f64 / 0x7fff_ffff as f64) * std::f64::consts::TAU;
        cos_vals.push(angle.cos() as f32);
        sin_vals.push(angle.sin() as f32);
    }
    (cos_vals, sin_vals)
}

/// Task 5 fixer's near-miss guard, generalized: a synthetic generator that
/// computes e.g. `pos*7 % 7` is always zero and would let a broken port
/// pass by symmetry. Assert the generator actually varies before trusting
/// any comparison against it.
fn assert_varying(data: &[u8], label: &str) {
    let all_same = data.windows(2).all(|w| w[0] == w[1]);
    assert!(!all_same, "{label}: generator produced constant bytes — not a real test");
}

fn assert_close(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    // Guard against a blind spot in the tolerance loop below: `(g - w).abs()
    // / tol` and the `err > worst` comparison that follows are silently
    // vacuous when both `g` and `w` are NaN (IEEE: any comparison against
    // NaN is false), so a candidate and reference that independently
    // compute the same garbage from the same bytes would both "pass" with
    // worst=0.000x. Caught empirically while building this harness: a raw
    // pseudo-random byte generator produced a non-finite `cnorm` for
    // asym3's K read, in BOTH the candidate and legacy-reference paths,
    // and this loop alone reported a perfect match. Explicit finiteness
    // check first, so a shared-NaN input is a hard failure, not a silent
    // 0.000x.
    if let Some(i) = got.iter().position(|v| !v.is_finite()) {
        panic!("{label}: candidate[{i}]={} is non-finite (want[{i}]={})", got[i], want[i]);
    }
    if let Some(i) = want.iter().position(|v| !v.is_finite()) {
        panic!("{label}: reference[{i}]={} is non-finite (got[{i}]={})", want[i], got[i]);
    }
    // Non-degeneracy guard: both `out` buffers start life as `gpu.zeros`, and
    // this harness's own comments (see build_tiles' doc comment on empty
    // tiles) document a kernel mode that can leave rows unwritten. Two
    // all-zero arrays pass the tolerance loop below at a perfect 0.000x —
    // indistinguishable from genuine agreement. Require the reference to
    // contain at least one nonzero element before trusting the comparison.
    if !want.is_empty() {
        assert!(
            want.iter().any(|v| v.abs() > 0.0),
            "{label}: reference array is all-zero ({} elements) — a kernel that wrote nothing \
             would pass this comparison by accident; refusing to treat an all-zero pair as agreement",
            want.len()
        );
    }
    let mut worst = 0.0f32;
    let mut worst_i = 0usize;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = 1e-3 * w.abs().max(1.0);
        let err = (g - w).abs() / tol;
        if err > worst {
            worst = err;
            worst_i = i;
        }
    }
    assert!(
        worst <= 1.0,
        "{label}: worst element {worst_i} at {worst:.2}x tolerance (got {}, want {})",
        got[worst_i],
        want[worst_i]
    );
    println!("  {label}: OK (worst {worst:.3}x tolerance)");
}

/// If NEGATIVE_CONTROL=1, every slot's descriptor is overwritten with
/// slot 0's — simulating "a descriptor reverted to slot 0" / "two slots
/// pointed at the same base offset". Proves the harness can actually fail.
///
/// CALLER CONTRACT: apply this ONLY to the descriptor table that gets
/// uploaded to the device for the CANDIDATE multi-slot launch, never to the
/// host-side `Vec<KvSlotDesc>` that `run_*_reference` uses to slice the
/// arenas for its per-slot legacy-kernel reference. If both arms see the
/// corrupted table, the reference reads the identical wrong slab as the
/// candidate and the comparison can never observe the fault — this was
/// exactly the bug in an earlier version of this control (task-7 review,
/// Fix 1): golden fell through to the numeric comparison, both sides
/// silently agreed on the same wrong answer, and NEGATIVE_CONTROL=1 could
/// only ever be observed to fail via the unrelated device-side
/// `seq_len <= desc.seq_len` guard on shapes with non-uniform seq_lens —
/// never via the tolerance comparison itself.
fn maybe_corrupt(mut descs: Vec<KvSlotDesc>) -> Vec<KvSlotDesc> {
    if std::env::var("NEGATIVE_CONTROL").as_deref() == Ok("1") {
        let d0 = descs[0];
        for d in descs.iter_mut() {
            *d = d0;
        }
    }
    descs
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum KvMode {
    Q8,
    Asym3,
}

impl KvMode {
    fn k_per_pos(self, n_kv_heads: usize, head_dim: usize) -> usize {
        match self {
            KvMode::Q8 => n_kv_heads * (head_dim / 32) * 34,
            KvMode::Asym3 => n_kv_heads * (4 + (head_dim * 3) / 8),
        }
    }
    // V is Q8_0 in BOTH modes (asym3 only rotates K).
    fn v_per_pos(self, n_kv_heads: usize, head_dim: usize) -> usize {
        n_kv_heads * (head_dim / 32) * 34
    }
}

// ────────────────────────────── Shape ───────────────────────────────────

#[derive(Clone)]
struct Shape {
    n_slots: usize,
    seq_lens: Vec<usize>,
    m_per_slot: Vec<usize>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl Shape {
    fn label(&self) -> String {
        format!(
            "slots={} seq_lens={:?} m={:?} nh={} nkv={} hd={}",
            self.n_slots, self.seq_lens, self.m_per_slot, self.n_heads, self.n_kv_heads, self.head_dim
        )
    }

    /// Per-slot query-row positions: a contiguous TAIL of `m` positions
    /// ending at `seq_len - 1`. This is deliberate, not incidental: it makes
    /// `positions[last_row] + 1 == desc.seq_len` exactly (the invariant's
    /// tight boundary) while every earlier row in an M>1 slot has a
    /// strictly SMALLER causal bound — exercising the "row 0 must not see
    /// row 2's key" trap the brief calls out, on every M>1 slot in every
    /// shape, for free.
    fn positions(&self) -> Vec<Vec<i32>> {
        self.seq_lens
            .iter()
            .zip(&self.m_per_slot)
            .map(|(&sl, &m)| {
                assert!(m <= sl, "shape {}: m={m} > seq_len={sl}", self.label());
                ((sl - m) as i32..sl as i32).collect()
            })
            .collect()
    }
}

fn shapes() -> Vec<Shape> {
    let mut v = Vec::new();
    // GQA 8:1 — qwen3.6-35b-a3b full-attention layers.
    // GQA 6:1 — qwen3.6-27b full-attention layers.
    for &(nh, nkv) in &[(16usize, 2usize), (24, 4)] {
        // Wildly unequal context in one batch.
        v.push(Shape {
            n_slots: 4,
            seq_lens: vec![1, 512, 8192, 100_000],
            m_per_slot: vec![1, 1, 1, 1],
            n_heads: nh,
            n_kv_heads: nkv,
            head_dim: 256,
        });
        // Mixed M: a zero-query slot, a decode, a small verify, a big verify.
        v.push(Shape {
            n_slots: 4,
            seq_lens: vec![4096, 4096, 4096, 4096],
            m_per_slot: vec![0, 1, 3, 8],
            n_heads: nh,
            n_kv_heads: nkv,
            head_dim: 256,
        });
        // Mixed prefill + decode — the batch shape SP1 exists for.
        v.push(Shape {
            n_slots: 4,
            seq_lens: vec![32_768, 1024, 512, 512],
            m_per_slot: vec![8, 256, 1, 1],
            n_heads: nh,
            n_kv_heads: nkv,
            head_dim: 256,
        });
        // seq_len below TILE_SIZE, and non-multiples of BR/BC.
        v.push(Shape {
            n_slots: 3,
            seq_lens: vec![7, 129, 131],
            m_per_slot: vec![1, 5, 1],
            n_heads: nh,
            n_kv_heads: nkv,
            head_dim: 256,
        });
        // Slot-count sweep 1..=8 at a fixed modest context.
        for n in 1..=8usize {
            v.push(Shape {
                n_slots: n,
                seq_lens: vec![2048; n],
                m_per_slot: vec![1; n],
                n_heads: nh,
                n_kv_heads: nkv,
                head_dim: 256,
            });
        }
    }
    v
}

/// Reduced-context subset for the LDS decode kernel (Task 4), which — multi-
/// slot or not — has always had an LDS-capacity ceiling around 15-16k tokens
/// (see attention_flash_q8_0_tile_batched.hip's header comment): shared_mem
/// scales with max_ctx_len and a 64KB budget caps it well under 16384.
fn shapes_lds() -> Vec<Shape> {
    let mut v = Vec::new();
    for &(nh, nkv) in &[(16usize, 2usize), (24, 4)] {
        v.push(Shape {
            n_slots: 4,
            seq_lens: vec![32, 500, 2000, 8000],
            m_per_slot: vec![1, 1, 3, 1],
            n_heads: nh,
            n_kv_heads: nkv,
            head_dim: 256,
        });
        // Mixed M including a zero-query slot (Fix 5, task-7 review):
        // shapes_lds() previously never exercised M=0, though the brief
        // requires M mixed across 0/1/3/8. A zero-query slot is a distinct
        // hazard for the LDS decode path specifically — it creates a gap in
        // descriptor indexing (slot 0 contributes no rows to the flat
        // batch) that a wrong row->slot mapping could paper over on shapes
        // where every slot has >=1 row.
        v.push(Shape {
            n_slots: 4,
            seq_lens: vec![32, 500, 2000, 8000],
            m_per_slot: vec![0, 1, 3, 8],
            n_heads: nh,
            n_kv_heads: nkv,
            head_dim: 256,
        });
        v.push(Shape {
            n_slots: 3,
            seq_lens: vec![100, 100, 100],
            m_per_slot: vec![1, 5, 8],
            n_heads: nh,
            n_kv_heads: nkv,
            head_dim: 256,
        });
        for n in [1usize, 2, 4, 8] {
            v.push(Shape {
                n_slots: n,
                seq_lens: vec![1500; n],
                m_per_slot: vec![1; n],
                n_heads: nh,
                n_kv_heads: nkv,
                head_dim: 256,
            });
        }
    }
    v
}

/// Shapes for the BR/BC-tiled Q8 flash-prefill kernel (Task 6), which is
/// Q8-only (no asym3 prefill kernel exists). Includes the mid-batch ragged-
/// tile-at-slot-boundary case Task 6's report flagged as the one class of
/// bug this kernel family, uniquely among the four, can have.
fn shapes_prefill() -> Vec<(Shape, usize, usize)> {
    let mut v = Vec::new();
    for &(nh, nkv) in &[(16usize, 2usize), (24, 4)] {
        // Ragged tiles ending mid-batch at a slot boundary (BR=4 vs M=5/10/3).
        v.push((
            Shape {
                n_slots: 3,
                seq_lens: vec![40, 50, 20],
                m_per_slot: vec![5, 10, 3],
                n_heads: nh,
                n_kv_heads: nkv,
                head_dim: 256,
            },
            4,
            8,
        ));
        // Mixed M incl. a zero-row slot and a big verify.
        v.push((
            Shape {
                n_slots: 4,
                seq_lens: vec![4096, 4096, 4096, 4096],
                m_per_slot: vec![0, 1, 3, 8],
                n_heads: nh,
                n_kv_heads: nkv,
                head_dim: 256,
            },
            8,
            32,
        ));
        // Mixed prefill + decode.
        v.push((
            Shape {
                n_slots: 4,
                seq_lens: vec![32_768, 1024, 512, 512],
                m_per_slot: vec![8, 256, 1, 1],
                n_heads: nh,
                n_kv_heads: nkv,
                head_dim: 256,
            },
            16,
            64,
        ));
        // seq_len below tile size, non-multiples of BR/BC.
        v.push((
            Shape {
                n_slots: 3,
                seq_lens: vec![7, 129, 131],
                m_per_slot: vec![1, 5, 1],
                n_heads: nh,
                n_kv_heads: nkv,
                head_dim: 256,
            },
            4,
            8,
        ));
        for n in 1..=8usize {
            v.push((
                Shape {
                    n_slots: n,
                    seq_lens: vec![2048; n],
                    m_per_slot: vec![1; n],
                    n_heads: nh,
                    n_kv_heads: nkv,
                    head_dim: 256,
                },
                8,
                32,
            ));
        }
    }
    v
}

// ───────────────────────── general tile-kernel path ─────────────────────
// (attention_flash_q8_0_batched_masked_slots / attention_flash_asym3_batched_masked_slots)

struct GeneralBatch {
    k_arena: GpuTensor,
    v_arena: GpuTensor,
    descs: Vec<KvSlotDesc>,
    descs_dev: GpuTensor,
    q: GpuTensor,
    positions: GpuTensor,
    row_slot: GpuTensor,
    batch_size: usize,
    max_ctx_len: usize,
}

impl GeneralBatch {
    /// `DeviceBuffer` (hip-bridge) has no `Drop` impl — every owned
    /// `GpuTensor` this harness allocates must be explicitly returned via
    /// `gpu.free_tensor` (which pools it for reuse) or live device memory
    /// grows unbounded across the full shape sweep. Hit in practice: the
    /// first full run OOM'd (`hipMalloc: out of memory`) partway through
    /// the second GQA config, after 301 clean "OK" checks had already
    /// accumulated their allocations. Not a kernel defect — a harness
    /// resource-management gap.
    fn free(self, gpu: &mut Gpu) {
        gpu.free_tensor(self.k_arena).expect("free k_arena");
        gpu.free_tensor(self.v_arena).expect("free v_arena");
        gpu.free_tensor(self.descs_dev).expect("free descs_dev");
        gpu.free_tensor(self.q).expect("free q");
        gpu.free_tensor(self.positions).expect("free positions");
        gpu.free_tensor(self.row_slot).expect("free row_slot");
    }
}

fn build_k_arena_for_mode(shape: &Shape, mode: KvMode, poison_except: Option<usize>) -> (Vec<u8>, Vec<KvSlotDesc>) {
    match mode {
        // Generic block filler is safe for Q8_0: the "scale" it poisons is
        // always exactly the f16 field the kernel reads.
        KvMode::Q8 => build_arena(&shape.seq_lens, mode.k_per_pos(shape.n_kv_heads, shape.head_dim), poison_except),
        // asym3's K `cnorm` is a 4-byte float read at a non-34-byte-aligned
        // offset — see build_asym3_k_arena's doc comment for why the
        // generic filler is unsafe here (empirically found while building
        // this harness, not a hypothetical).
        KvMode::Asym3 => build_asym3_k_arena(&shape.seq_lens, shape.n_kv_heads, shape.head_dim, poison_except),
    }
}

fn build_general_batch(gpu: &mut Gpu, shape: &Shape, mode: KvMode, poison_except: Option<usize>) -> GeneralBatch {
    build_general_batch_ex(gpu, shape, mode, poison_except, true)
}

/// `apply_negative_control=false` bypasses `maybe_corrupt` regardless of the
/// NEGATIVE_CONTROL env var. Needed by `test_poison_is_live`: that check
/// builds a batch via `poison_except` for an UNRELATED reason (proving the
/// NaN-poison byte generator is live), and if NEGATIVE_CONTROL=1 happened to
/// be set in the environment, the global descriptor corruption would
/// override every slot's descriptor to slot 0's BEFORE the kernel launch —
/// silently redirecting the "poisoned" slot's read to a clean neighbour and
/// making the positive control fail for a reason that has nothing to do
/// with whether the poison bytes themselves are live. The two controls test
/// independent hazards (descriptor corruption vs. arena-content poisoning)
/// and must not be coupled through the same env var.
fn build_general_batch_ex(
    gpu: &mut Gpu,
    shape: &Shape,
    mode: KvMode,
    poison_except: Option<usize>,
    apply_negative_control: bool,
) -> GeneralBatch {
    let v_per_pos = mode.v_per_pos(shape.n_kv_heads, shape.head_dim);
    let (k_bytes, k_descs) = build_k_arena_for_mode(shape, mode, poison_except);
    let (v_bytes, v_descs) = build_arena(&shape.seq_lens, v_per_pos, poison_except);
    assert_varying(&k_bytes, "K arena");
    assert_varying(&v_bytes, "V arena");
    // NEGATIVE_CONTROL corrupts only the device-bound copy — `descs` (used
    // by run_general_reference to slice these same arenas) stays correct.
    // See maybe_corrupt's doc comment.
    let descs = merge_descs(&k_descs, &v_descs);
    let descs_for_device = if apply_negative_control { maybe_corrupt(descs.clone()) } else { descs.clone() };
    let descs_dev = gpu.upload_raw(&pack_descs(&descs_for_device), &[shape.n_slots]).expect("descs upload");
    let k_arena = gpu.upload_raw(&k_bytes, &[k_bytes.len()]).expect("k arena upload");
    let v_arena = gpu.upload_raw(&v_bytes, &[v_bytes.len()]).expect("v arena upload");

    let q_dim = shape.n_heads * shape.head_dim;
    let positions_per_slot = shape.positions();
    let mut q_data = Vec::new();
    let mut positions_flat = Vec::new();
    let mut row_slot = Vec::new();
    for s in 0..shape.n_slots {
        for (r, &p) in positions_per_slot[s].iter().enumerate() {
            for _ in 0..q_dim {
                let i = q_data.len();
                q_data.push((((i * 37 + s * 53 + r * 11) % 101) as f32 - 50.0) * 0.01);
            }
            positions_flat.push(p);
            row_slot.push(s as i32);
        }
    }
    let batch_size = positions_flat.len();
    let q = gpu.upload_f32(&q_data, &[batch_size.max(1) * q_dim]).expect("q upload");
    let positions = gpu.upload_raw(&i32_bytes(&positions_flat), &[batch_size.max(1)]).expect("positions upload");
    let row_slot_dev = gpu.upload_raw(&i32_bytes(&row_slot), &[batch_size.max(1)]).expect("row_slot upload");
    let max_ctx_len = *shape.seq_lens.iter().max().unwrap();

    GeneralBatch {
        k_arena,
        v_arena,
        descs,
        descs_dev,
        q,
        positions,
        row_slot: row_slot_dev,
        batch_size,
        max_ctx_len,
    }
}

#[allow(clippy::too_many_arguments)]
fn run_general_candidate(
    gpu: &mut Gpu,
    shape: &Shape,
    mode: KvMode,
    batch: &GeneralBatch,
    force_subbatch: bool,
    cos_theta: &GpuTensor,
    sin_theta: &GpuTensor,
) -> Vec<f32> {
    let q_dim = shape.n_heads * shape.head_dim;
    if batch.batch_size == 0 {
        return Vec::new();
    }
    const TILE_SIZE: usize = 128;
    let max_tiles = batch.max_ctx_len.div_ceil(TILE_SIZE);
    let rows_for_partials = if force_subbatch {
        batch.batch_size.div_ceil(3).max(1)
    } else {
        batch.batch_size
    };
    let partials = gpu
        .zeros(&[rows_for_partials * shape.n_heads * max_tiles * (2 + shape.head_dim)], DType::F32)
        .expect("partials");
    let out = gpu.zeros(&[batch.batch_size * q_dim], DType::F32).expect("out");

    match mode {
        KvMode::Q8 => {
            gpu.attention_flash_q8_0_batched_masked_slots(
                &batch.q,
                &batch.k_arena,
                &batch.v_arena,
                &out,
                &batch.positions,
                shape.n_heads,
                shape.n_kv_heads,
                shape.head_dim,
                batch.max_ctx_len,
                batch.max_ctx_len,
                batch.batch_size,
                &partials,
                None,
                0,
                0,
                Some(&batch.descs_dev),
                Some(&batch.row_slot),
            )
            .expect("q8 tile candidate");
        }
        KvMode::Asym3 => {
            gpu.attention_flash_asym3_batched_masked_slots(
                &batch.q,
                &batch.k_arena,
                &batch.v_arena,
                &out,
                &batch.positions,
                cos_theta,
                sin_theta,
                shape.n_heads,
                shape.n_kv_heads,
                shape.head_dim,
                batch.max_ctx_len,
                batch.max_ctx_len,
                batch.batch_size,
                &partials,
                None,
                0,
                0,
                Some(&batch.descs_dev),
                Some(&batch.row_slot),
            )
            .expect("asym3 tile candidate");
        }
    }
    gpu.hip.device_synchronize().expect("sync");
    let result = gpu.download_f32(&out).expect("download");
    gpu.free_tensor(partials).expect("free partials");
    gpu.free_tensor(out).expect("free out");
    result
}

fn run_general_reference(gpu: &mut Gpu, shape: &Shape, mode: KvMode, batch: &GeneralBatch, cos_theta: &GpuTensor, sin_theta: &GpuTensor) -> Vec<f32> {
    let q_dim = shape.n_heads * shape.head_dim;
    let k_per_pos = mode.k_per_pos(shape.n_kv_heads, shape.head_dim);
    let v_per_pos = mode.v_per_pos(shape.n_kv_heads, shape.head_dim);
    let positions_per_slot = shape.positions();
    let mut out = vec![0f32; batch.batch_size * q_dim];
    let mut row0 = 0usize;
    for s in 0..shape.n_slots {
        let m = shape.m_per_slot[s];
        if m == 0 {
            continue;
        }
        let desc = batch.descs[s];
        let cap = desc.cap as usize;
        let sl = shape.seq_lens[s];
        let k_view = batch.k_arena.sub_offset(desc.k_base as usize, cap * k_per_pos);
        let v_view = batch.v_arena.sub_offset(desc.v_base as usize, cap * v_per_pos);
        let q_slice = batch.q.sub_offset(row0 * q_dim, m * q_dim);
        let pos_dev = gpu.upload_raw(&i32_bytes(&positions_per_slot[s]), &[m]).expect("pos_ref upload");
        let max_tiles = sl.div_ceil(128);
        let partials_ref = gpu.zeros(&[m * shape.n_heads * max_tiles * (2 + shape.head_dim)], DType::F32).expect("partials_ref");
        let out_ref = gpu.zeros(&[m * q_dim], DType::F32).expect("out_ref");
        match mode {
            KvMode::Q8 => {
                gpu.attention_flash_q8_0_batched_masked(
                    &q_slice, &k_view, &v_view, &out_ref, &pos_dev, shape.n_heads, shape.n_kv_heads, shape.head_dim, sl, sl, m, &partials_ref,
                    None, 0, 0,
                )
                .expect("q8 legacy reference");
            }
            KvMode::Asym3 => {
                gpu.attention_flash_asym3_batched_masked(
                    &q_slice, &k_view, &v_view, &out_ref, &pos_dev, cos_theta, sin_theta, shape.n_heads, shape.n_kv_heads, shape.head_dim, sl,
                    sl, m, &partials_ref, None, 0, 0,
                )
                .expect("asym3 legacy reference");
            }
        }
        gpu.hip.device_synchronize().expect("sync");
        let ref_out = gpu.download_f32(&out_ref).expect("download ref");
        out[row0 * q_dim..(row0 + m) * q_dim].copy_from_slice(&ref_out);
        row0 += m;
        // k_view/v_view/q_slice are Borrowed sub_offset views into
        // batch.{k_arena,v_arena,q} — freeing those would double-free the
        // arena. Only the per-iteration owned scratch goes back to the pool,
        // so this loop's live footprint is O(1) slot, not O(n_slots).
        gpu.free_tensor(pos_dev).expect("free pos_dev");
        gpu.free_tensor(partials_ref).expect("free partials_ref");
        gpu.free_tensor(out_ref).expect("free out_ref");
    }
    out
}

fn slot_row_range(shape: &Shape, target: usize) -> (usize, usize) {
    let row0: usize = shape.m_per_slot[..target].iter().sum();
    (row0, shape.m_per_slot[target])
}

fn slot_output<'a>(flat: &'a [f32], shape: &Shape, target: usize, q_dim: usize) -> &'a [f32] {
    let (row0, m) = slot_row_range(shape, target);
    &flat[row0 * q_dim..(row0 + m) * q_dim]
}

fn test_general_golden(gpu: &mut Gpu, shape: &Shape, mode: KvMode, cos_theta: &GpuTensor, sin_theta: &GpuTensor, force_subbatch: bool) {
    let batch = build_general_batch(gpu, shape, mode, None);
    if batch.batch_size == 0 {
        batch.free(gpu);
        return;
    }
    let reference = run_general_reference(gpu, shape, mode, &batch, cos_theta, sin_theta);
    let candidate = run_general_candidate(gpu, shape, mode, &batch, force_subbatch, cos_theta, sin_theta);
    batch.free(gpu);
    assert_close(
        &format!("golden [{:?} subbatch={}] {}", mode, force_subbatch, shape.label()),
        &candidate,
        &reference,
    );
}

fn test_general_isolation(gpu: &mut Gpu, shape: &Shape, mode: KvMode, cos_theta: &GpuTensor, sin_theta: &GpuTensor, force_subbatch: bool) {
    let q_dim = shape.n_heads * shape.head_dim;
    let clean_batch = build_general_batch(gpu, shape, mode, None);
    if clean_batch.batch_size == 0 {
        clean_batch.free(gpu);
        return;
    }
    let clean = run_general_candidate(gpu, shape, mode, &clean_batch, force_subbatch, cos_theta, sin_theta);
    // Free clean_batch's arenas/q/positions/row_slot now — every remaining
    // use in this function reads only the downloaded `clean: Vec<f32>`, not
    // the GPU tensors. Each loop iteration below builds and frees its own
    // poisoned_batch, so the live footprint stays O(1) slot across the
    // n_slots loop, not O(n_slots).
    clean_batch.free(gpu);
    for target in 0..shape.n_slots {
        if shape.m_per_slot[target] == 0 {
            continue;
        }
        let poisoned_batch = build_general_batch(gpu, shape, mode, Some(target));
        let poisoned = run_general_candidate(gpu, shape, mode, &poisoned_batch, force_subbatch, cos_theta, sin_theta);
        poisoned_batch.free(gpu);
        let a = slot_output(&clean, shape, target, q_dim);
        let b = slot_output(&poisoned, shape, target, q_dim);
        if std::env::var("DEBUG_ISO").as_deref() == Ok("1") {
            let m = shape.m_per_slot[target];
            for r in 0..m {
                let row_a = &a[r * q_dim..(r + 1) * q_dim];
                let row_b = &b[r * q_dim..(r + 1) * q_dim];
                let bad: Vec<(usize, f32, f32)> = row_a
                    .iter()
                    .zip(row_b)
                    .enumerate()
                    .filter(|(_, (_, bb))| !bb.is_finite())
                    .map(|(i, (aa, bb))| (i, *aa, *bb))
                    .take(5)
                    .collect();
                println!(
                    "    DEBUG row {r}: {} non-finite of {q_dim}; first few: {:?}",
                    row_b.iter().filter(|v| !v.is_finite()).count(),
                    bad
                );
            }
        }
        assert!(
            b.iter().all(|v| v.is_finite()),
            "isolation [{:?} subbatch={}] {}: slot {target} NaN leaked in from a neighbouring slot",
            mode,
            force_subbatch,
            shape.label()
        );
        assert_close(
            &format!("isolation [{:?} subbatch={}] slot={target} {}", mode, force_subbatch, shape.label()),
            b,
            a,
        );
    }
}

/// Positive control for the poison mechanism itself (Fix 2, task-7 review).
/// The isolation checks above (`test_*_isolation`) all assert a NEGATIVE
/// result: a NaN-poisoned NEIGHBOUR must not corrupt the target slot's
/// output. If `poison_except` were inert — a no-op, or wired to the wrong
/// slot — every one of those 542 checks would compare a run against a
/// byte-identical run and pass at a perfect 0.000x, indistinguishable from
/// the current (genuinely working) output. This flips the polarity: poison
/// the TARGET slot itself (by naming a *different* slot as the
/// `poison_except` survivor) and assert the target's own output DOES
/// become non-finite. That is only true if the NaN bytes this harness
/// writes (Q8_0's f16 scale field `0x7E00`, or asym3's `cnorm = NaN`) are
/// actually reaching the device read the kernel performs for that slot.
fn test_poison_is_live(gpu: &mut Gpu, mode: KvMode, cos_theta: &GpuTensor, sin_theta: &GpuTensor) {
    let shape = Shape {
        n_slots: 3,
        seq_lens: vec![2048, 2048, 2048],
        m_per_slot: vec![1, 1, 1],
        n_heads: 16,
        n_kv_heads: 2,
        head_dim: 256,
    };
    let target = 1usize;
    let bystander = 0usize; // poison_except's survivor — every OTHER slot,
                             // including `target`, gets NaN'd.
    // apply_negative_control=false: this check is orthogonal to
    // NEGATIVE_CONTROL and must not be silently defeated by it if that env
    // var happens to be set for an unrelated run — see build_general_batch_ex's
    // doc comment.
    let batch = build_general_batch_ex(gpu, &shape, mode, Some(bystander), false);
    let out = run_general_candidate(gpu, &shape, mode, &batch, false, cos_theta, sin_theta);
    batch.free(gpu);
    let q_dim = shape.n_heads * shape.head_dim;
    let victim = slot_output(&out, &shape, target, q_dim);
    let n_bad = victim.iter().filter(|v| !v.is_finite()).count();
    assert!(
        n_bad > 0,
        "positive poison control [{mode:?}]: target slot {target}'s OWN KV was poisoned with \
         NaN (bystander slot {bystander} left clean), but its output stayed fully finite \
         ({} elements, 0 non-finite) — the poison mechanism is not reaching this kernel's \
         device reads, which means the isolation layer's passing checks are meaningless",
        victim.len()
    );
    println!(
        "  positive poison control [{mode:?}]: OK (target slot {target}'s poisoned output has \
         {n_bad}/{} non-finite elements)",
        victim.len()
    );
}

// ─────────────────────────── LDS decode kernel path ──────────────────────
// (attention_q8_0_kv_batched_masked_slots, Q8-only)

struct LdsBatch {
    k_arena: GpuTensor,
    v_arena: GpuTensor,
    descs: Vec<KvSlotDesc>,
    descs_dev: GpuTensor,
    q: GpuTensor,
    positions: GpuTensor,
    row_slot: GpuTensor,
    batch_size: usize,
    max_ctx_len: usize,
}

fn build_lds_batch(gpu: &mut Gpu, shape: &Shape, poison_except: Option<usize>) -> LdsBatch {
    let per_pos = KvMode::Q8.k_per_pos(shape.n_kv_heads, shape.head_dim);
    let (k_bytes, k_descs) = build_arena(&shape.seq_lens, per_pos, poison_except);
    let (v_bytes, v_descs) = build_arena(&shape.seq_lens, per_pos, poison_except);
    assert_varying(&k_bytes, "LDS K arena");
    assert_varying(&v_bytes, "LDS V arena");
    // See build_general_batch: corrupt only the device-bound copy.
    let descs = merge_descs(&k_descs, &v_descs);
    let descs_for_device = maybe_corrupt(descs.clone());
    let descs_dev = gpu.upload_raw(&pack_descs(&descs_for_device), &[shape.n_slots]).expect("descs upload");
    let k_arena = gpu.upload_raw(&k_bytes, &[k_bytes.len()]).expect("k arena upload");
    let v_arena = gpu.upload_raw(&v_bytes, &[v_bytes.len()]).expect("v arena upload");

    let q_dim = shape.n_heads * shape.head_dim;
    let positions_per_slot = shape.positions();
    let mut q_data = Vec::new();
    let mut positions_flat = Vec::new();
    let mut row_slot = Vec::new();
    for s in 0..shape.n_slots {
        for (r, &p) in positions_per_slot[s].iter().enumerate() {
            for _ in 0..q_dim {
                let i = q_data.len();
                q_data.push((((i * 43 + s * 59 + r * 13) % 101) as f32 - 50.0) * 0.01);
            }
            positions_flat.push(p);
            row_slot.push(s as i32);
        }
    }
    let batch_size = positions_flat.len();
    let q = gpu.upload_f32(&q_data, &[batch_size.max(1) * q_dim]).expect("q upload");
    let positions = gpu.upload_raw(&i32_bytes(&positions_flat), &[batch_size.max(1)]).expect("positions upload");
    let row_slot_dev = gpu.upload_raw(&i32_bytes(&row_slot), &[batch_size.max(1)]).expect("row_slot upload");
    let max_ctx_len = *shape.seq_lens.iter().max().unwrap();
    LdsBatch { k_arena, v_arena, descs, descs_dev, q, positions, row_slot: row_slot_dev, batch_size, max_ctx_len }
}

impl LdsBatch {
    fn free(self, gpu: &mut Gpu) {
        gpu.free_tensor(self.k_arena).expect("free k_arena");
        gpu.free_tensor(self.v_arena).expect("free v_arena");
        gpu.free_tensor(self.descs_dev).expect("free descs_dev");
        gpu.free_tensor(self.q).expect("free q");
        gpu.free_tensor(self.positions).expect("free positions");
        gpu.free_tensor(self.row_slot).expect("free row_slot");
    }
}

fn run_lds_candidate(gpu: &mut Gpu, shape: &Shape, batch: &LdsBatch) -> Vec<f32> {
    let q_dim = shape.n_heads * shape.head_dim;
    if batch.batch_size == 0 {
        return Vec::new();
    }
    let out = gpu.zeros(&[batch.batch_size * q_dim], DType::F32).expect("out");
    gpu.attention_q8_0_kv_batched_masked_slots(
        &batch.q,
        &batch.k_arena,
        &batch.v_arena,
        &out,
        &batch.positions,
        shape.n_heads,
        shape.n_kv_heads,
        shape.head_dim,
        batch.max_ctx_len,
        batch.max_ctx_len,
        batch.batch_size,
        None,
        0,
        0,
        Some(&batch.descs_dev),
        Some(&batch.row_slot),
    )
    .expect("lds candidate");
    gpu.hip.device_synchronize().expect("sync");
    let result = gpu.download_f32(&out).expect("download");
    gpu.free_tensor(out).expect("free out");
    result
}

fn run_lds_reference(gpu: &mut Gpu, shape: &Shape, batch: &LdsBatch) -> Vec<f32> {
    let q_dim = shape.n_heads * shape.head_dim;
    let per_pos = KvMode::Q8.k_per_pos(shape.n_kv_heads, shape.head_dim);
    let positions_per_slot = shape.positions();
    let mut out = vec![0f32; batch.batch_size * q_dim];
    let mut row0 = 0usize;
    for s in 0..shape.n_slots {
        let m = shape.m_per_slot[s];
        if m == 0 {
            continue;
        }
        let desc = batch.descs[s];
        let cap = desc.cap as usize;
        let sl = shape.seq_lens[s];
        let k_view = batch.k_arena.sub_offset(desc.k_base as usize, cap * per_pos);
        let v_view = batch.v_arena.sub_offset(desc.v_base as usize, cap * per_pos);
        let q_slice = batch.q.sub_offset(row0 * q_dim, m * q_dim);
        let pos_dev = gpu.upload_raw(&i32_bytes(&positions_per_slot[s]), &[m]).expect("pos_ref upload");
        let out_ref = gpu.zeros(&[m * q_dim], DType::F32).expect("out_ref");
        gpu.attention_q8_0_kv_batched_masked(
            &q_slice, &k_view, &v_view, &out_ref, &pos_dev, shape.n_heads, shape.n_kv_heads, shape.head_dim, sl, sl, m, None, 0, 0,
        )
        .expect("lds legacy reference");
        gpu.hip.device_synchronize().expect("sync");
        let ref_out = gpu.download_f32(&out_ref).expect("download ref");
        out[row0 * q_dim..(row0 + m) * q_dim].copy_from_slice(&ref_out);
        row0 += m;
        gpu.free_tensor(pos_dev).expect("free pos_dev");
        gpu.free_tensor(out_ref).expect("free out_ref");
    }
    out
}

fn test_lds_golden(gpu: &mut Gpu, shape: &Shape) {
    let batch = build_lds_batch(gpu, shape, None);
    if batch.batch_size == 0 {
        batch.free(gpu);
        return;
    }
    let reference = run_lds_reference(gpu, shape, &batch);
    let candidate = run_lds_candidate(gpu, shape, &batch);
    batch.free(gpu);
    assert_close(&format!("LDS golden {}", shape.label()), &candidate, &reference);
}

fn test_lds_isolation(gpu: &mut Gpu, shape: &Shape) {
    let q_dim = shape.n_heads * shape.head_dim;
    let clean_batch = build_lds_batch(gpu, shape, None);
    if clean_batch.batch_size == 0 {
        clean_batch.free(gpu);
        return;
    }
    let clean = run_lds_candidate(gpu, shape, &clean_batch);
    clean_batch.free(gpu);
    for target in 0..shape.n_slots {
        if shape.m_per_slot[target] == 0 {
            continue;
        }
        let poisoned_batch = build_lds_batch(gpu, shape, Some(target));
        let poisoned = run_lds_candidate(gpu, shape, &poisoned_batch);
        poisoned_batch.free(gpu);
        let a = slot_output(&clean, shape, target, q_dim);
        let b = slot_output(&poisoned, shape, target, q_dim);
        assert!(
            b.iter().all(|v| v.is_finite()),
            "LDS isolation {}: slot {target} NaN leaked in from a neighbouring slot",
            shape.label()
        );
        assert_close(&format!("LDS isolation slot={target} {}", shape.label()), b, a);
    }
}

// ───────────────────────── Q8 flash-prefill kernel path ──────────────────
// (attention_q8_0_flash_prefill_slots, Q8-only)

struct PrefillBatch {
    k_arena: GpuTensor,
    v_arena: GpuTensor,
    descs: Vec<KvSlotDesc>,
    descs_dev: GpuTensor,
    q: GpuTensor,
    positions: GpuTensor,
    tile_slot: GpuTensor,
    tile_row0: GpuTensor,
    tile_qbase: GpuTensor,
    n_tiles: usize,
    batch_size: usize,
}

fn build_prefill_batch(gpu: &mut Gpu, shape: &Shape, br: usize, poison_except: Option<usize>) -> PrefillBatch {
    let per_pos = KvMode::Q8.k_per_pos(shape.n_kv_heads, shape.head_dim); // K == V stride for prefill
    let (k_bytes, k_descs) = build_arena(&shape.seq_lens, per_pos, poison_except);
    let (v_bytes, v_descs) = build_arena(&shape.seq_lens, per_pos, poison_except);
    assert_varying(&k_bytes, "prefill K arena");
    assert_varying(&v_bytes, "prefill V arena");
    // See build_general_batch: corrupt only the device-bound copy.
    let descs = merge_descs(&k_descs, &v_descs);
    let descs_for_device = maybe_corrupt(descs.clone());
    let descs_dev = gpu.upload_raw(&pack_descs(&descs_for_device), &[shape.n_slots]).expect("descs upload");
    let k_arena = gpu.upload_raw(&k_bytes, &[k_bytes.len()]).expect("k arena upload");
    let v_arena = gpu.upload_raw(&v_bytes, &[v_bytes.len()]).expect("v arena upload");

    let q_dim = shape.n_heads * shape.head_dim;
    let positions_per_slot = shape.positions();
    let mut q_data = Vec::new();
    let mut positions_flat = Vec::new();
    for s in 0..shape.n_slots {
        for (r, &p) in positions_per_slot[s].iter().enumerate() {
            for _ in 0..q_dim {
                let i = q_data.len();
                q_data.push((((i * 41 + s * 67 + r * 19) % 101) as f32 - 50.0) * 0.01);
            }
            positions_flat.push(p);
        }
    }
    let batch_size = positions_flat.len();
    let q = gpu.upload_f32(&q_data, &[batch_size.max(1) * q_dim]).expect("q upload");
    let positions = gpu.upload_raw(&i32_bytes(&positions_flat), &[batch_size.max(1)]).expect("positions upload");

    let (tile_slot, tile_row0, tile_qbase) = build_tiles(&shape.m_per_slot, br);
    let n_tiles = tile_slot.len();
    let tile_slot_dev = gpu.upload_raw(&i32_bytes(&tile_slot), &[n_tiles.max(1)]).expect("tile_slot upload");
    let tile_row0_dev = gpu.upload_raw(&i32_bytes(&tile_row0), &[n_tiles.max(1)]).expect("tile_row0 upload");
    let tile_qbase_dev = gpu.upload_raw(&i32_bytes(&tile_qbase), &[n_tiles.max(1)]).expect("tile_qbase upload");

    PrefillBatch {
        k_arena,
        v_arena,
        descs,
        descs_dev,
        q,
        positions,
        tile_slot: tile_slot_dev,
        tile_row0: tile_row0_dev,
        tile_qbase: tile_qbase_dev,
        n_tiles,
        batch_size,
    }
}

impl PrefillBatch {
    fn free(self, gpu: &mut Gpu) {
        gpu.free_tensor(self.k_arena).expect("free k_arena");
        gpu.free_tensor(self.v_arena).expect("free v_arena");
        gpu.free_tensor(self.descs_dev).expect("free descs_dev");
        gpu.free_tensor(self.q).expect("free q");
        gpu.free_tensor(self.positions).expect("free positions");
        gpu.free_tensor(self.tile_slot).expect("free tile_slot");
        gpu.free_tensor(self.tile_row0).expect("free tile_row0");
        gpu.free_tensor(self.tile_qbase).expect("free tile_qbase");
    }
}

fn run_prefill_candidate(gpu: &mut Gpu, shape: &Shape, batch: &PrefillBatch, br: usize, bc: usize) -> Vec<f32> {
    let q_dim = shape.n_heads * shape.head_dim;
    if batch.batch_size == 0 {
        return Vec::new();
    }
    let out = gpu.zeros(&[batch.batch_size * q_dim], DType::F32).expect("out");
    gpu.attention_q8_0_flash_prefill_slots(
        &batch.q,
        &batch.k_arena,
        &batch.v_arena,
        &out,
        &batch.positions,
        shape.n_heads,
        shape.n_kv_heads,
        shape.head_dim,
        64, // max_ctx_len: unused by the kernel body, kept for ABI parity
        batch.batch_size,
        br,
        bc,
        Some(&batch.descs_dev),
        Some(&batch.tile_slot),
        Some(&batch.tile_row0),
        Some(&batch.tile_qbase),
    )
    .expect("prefill candidate");
    let _ = batch.n_tiles;
    gpu.hip.device_synchronize().expect("sync");
    let result = gpu.download_f32(&out).expect("download");
    gpu.free_tensor(out).expect("free out");
    result
}

fn run_prefill_reference(gpu: &mut Gpu, shape: &Shape, batch: &PrefillBatch, br: usize, bc: usize) -> Vec<f32> {
    let q_dim = shape.n_heads * shape.head_dim;
    let per_pos = KvMode::Q8.k_per_pos(shape.n_kv_heads, shape.head_dim);
    let positions_per_slot = shape.positions();
    let mut out = vec![0f32; batch.batch_size * q_dim];
    let mut row0 = 0usize;
    for s in 0..shape.n_slots {
        let m = shape.m_per_slot[s];
        if m == 0 {
            continue;
        }
        let desc = batch.descs[s];
        let cap = desc.cap as usize;
        let k_view = batch.k_arena.sub_offset(desc.k_base as usize, cap * per_pos);
        let v_view = batch.v_arena.sub_offset(desc.v_base as usize, cap * per_pos);
        let q_slice = batch.q.sub_offset(row0 * q_dim, m * q_dim);
        let pos_dev = gpu.upload_raw(&i32_bytes(&positions_per_slot[s]), &[m]).expect("pos_ref upload");
        let out_ref = gpu.zeros(&[m * q_dim], DType::F32).expect("out_ref");
        gpu.attention_q8_0_flash_prefill(
            &q_slice, &k_view, &v_view, &out_ref, &pos_dev, shape.n_heads, shape.n_kv_heads, shape.head_dim, cap, m, br, bc,
        )
        .expect("prefill legacy reference");
        gpu.hip.device_synchronize().expect("sync");
        let ref_out = gpu.download_f32(&out_ref).expect("download ref");
        out[row0 * q_dim..(row0 + m) * q_dim].copy_from_slice(&ref_out);
        row0 += m;
        gpu.free_tensor(pos_dev).expect("free pos_dev");
        gpu.free_tensor(out_ref).expect("free out_ref");
    }
    out
}

fn test_prefill_golden(gpu: &mut Gpu, shape: &Shape, br: usize, bc: usize) {
    let batch = build_prefill_batch(gpu, shape, br, None);
    if batch.batch_size == 0 {
        batch.free(gpu);
        return;
    }
    let reference = run_prefill_reference(gpu, shape, &batch, br, bc);
    let candidate = run_prefill_candidate(gpu, shape, &batch, br, bc);
    batch.free(gpu);
    assert_close(&format!("prefill golden [br={br} bc={bc}] {}", shape.label()), &candidate, &reference);
}

fn test_prefill_isolation(gpu: &mut Gpu, shape: &Shape, br: usize, bc: usize) {
    let q_dim = shape.n_heads * shape.head_dim;
    let clean_batch = build_prefill_batch(gpu, shape, br, None);
    if clean_batch.batch_size == 0 {
        clean_batch.free(gpu);
        return;
    }
    let clean = run_prefill_candidate(gpu, shape, &clean_batch, br, bc);
    clean_batch.free(gpu);
    for target in 0..shape.n_slots {
        if shape.m_per_slot[target] == 0 {
            continue;
        }
        let poisoned_batch = build_prefill_batch(gpu, shape, br, Some(target));
        let poisoned = run_prefill_candidate(gpu, shape, &poisoned_batch, br, bc);
        poisoned_batch.free(gpu);
        let a = slot_output(&clean, shape, target, q_dim);
        let b = slot_output(&poisoned, shape, target, q_dim);
        assert!(
            b.iter().all(|v| v.is_finite()),
            "prefill isolation [br={br} bc={bc}] {}: slot {target} NaN leaked in from a neighbouring slot",
            shape.label()
        );
        assert_close(
            &format!("prefill isolation [br={br} bc={bc}] slot={target} {}", shape.label()),
            b,
            a,
        );
    }
}

// ──────────────────────────────── main ────────────────────────────────

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

/// True when the NEGATIVE_CONTROL=1 env var is set.
fn negative_control_active() -> bool {
    std::env::var("NEGATIVE_CONTROL").as_deref() == Ok("1")
}

/// Shapes with every slot at the same seq_len. Under NEGATIVE_CONTROL=1
/// (every descriptor forced to slot 0's), a shape with UNEQUAL seq_lens
/// trips the device-side `positions[row]+1 <= desc.seq_len` guard (Tasks
/// 5/6) as soon as a later slot's real, larger position is checked against
/// slot 0's smaller seq_len/cap — aborting the process before the numeric
/// `assert_close` comparison ever runs. Restricting to uniform-seq_len
/// shapes keeps that guard quiet (every desc — corrupted or not — has the
/// same seq_len/cap) so the corrupted addressing shows up ONLY as a wrong
/// numeric result, which is what Fix 1 (task-7 review) needs to prove
/// assert_close is sensitive to a real addressing fault.
fn uniform_seq_len(shape: &Shape) -> bool {
    shape.seq_lens.iter().all(|&sl| sl == shape.seq_lens[0])
}

#[cfg(feature = "deltanet")]
fn main() {
    let mut gpu = Gpu::init().expect("gpu init");

    if negative_control_active() {
        println!(
            "=== NEGATIVE_CONTROL=1: every descriptor forced to slot 0, candidate arm only \
             — restricted to uniform-seq_len shapes so the numeric comparison (not the device \
             seq_len<=cap guard) is what has to catch it — expecting a hard failure ==="
        );
    }

    // Step 4: assert the asym3 arm is really asym3 (not silently downgraded
    // to q8 — see spec §4.4).
    let hd = 256usize;
    let nkv = 4usize; // 27B GQA config, arbitrary for this byte-budget check
    let asym3_bytes_per_pos = KvMode::Asym3.k_per_pos(nkv, hd);
    let q8_bytes_per_pos = KvMode::Q8.k_per_pos(nkv, hd);
    assert!(
        asym3_bytes_per_pos < q8_bytes_per_pos,
        "asym3 arena is not smaller than Q8 — the asym3 path is not active \
         (see spec §4.4: QWEN35_PARO_POLICY silently downgrades asym3 to q8)"
    );
    println!("asym3 K bytes/pos={asym3_bytes_per_pos} < Q8 K bytes/pos={q8_bytes_per_pos}: OK (asym3 path confirmed active)\n");

    let (cos_vals, sin_vals) = gen_givens_angles(42, hd / 2);
    let cos_theta = gpu.upload_f32(&cos_vals, &[cos_vals.len()]).expect("cos upload");
    let sin_theta = gpu.upload_f32(&sin_vals, &[sin_vals.len()]).expect("sin upload");

    println!("### Positive poison control (Fix 2, task-7 review) ###");
    test_poison_is_live(&mut gpu, KvMode::Q8, &cos_theta, &sin_theta);
    test_poison_is_live(&mut gpu, KvMode::Asym3, &cos_theta, &sin_theta);
    println!();

    let mut n_ok = 0usize;
    let mut n_total = 0usize;

    let smoke = std::env::var("SMOKE").as_deref() == Ok("1");
    let smoke_shape = || Shape {
        n_slots: 3,
        seq_lens: vec![40, 129, 300],
        m_per_slot: vec![1, 5, 3],
        n_heads: 16,
        n_kv_heads: 2,
        head_dim: 256,
    };
    println!("### General tile-kernel sweep (Task 5 Q8 tile + Task 6 asym3 tile) ###");
    let general_shapes: Vec<Shape> = if smoke {
        vec![smoke_shape()]
    } else if negative_control_active() {
        shapes().into_iter().filter(uniform_seq_len).collect()
    } else {
        shapes()
    };
    for shape in general_shapes {
        for &mode in &[KvMode::Q8, KvMode::Asym3] {
            for &force_subbatch in &[false, true] {
                println!("-- {} mode={mode:?} subbatch={force_subbatch}", shape.label());
                n_total += 2;
                test_general_golden(&mut gpu, &shape, mode, &cos_theta, &sin_theta, force_subbatch);
                n_ok += 1;
                test_general_isolation(&mut gpu, &shape, mode, &cos_theta, &sin_theta, force_subbatch);
                n_ok += 1;
            }
        }
    }

    println!("\n### LDS decode kernel sweep (Task 4, Q8-only, capped context) ###");
    let lds_shapes: Vec<Shape> = if smoke {
        vec![smoke_shape()]
    } else if negative_control_active() {
        shapes_lds().into_iter().filter(uniform_seq_len).collect()
    } else {
        shapes_lds()
    };
    for shape in lds_shapes {
        println!("-- {}", shape.label());
        n_total += 2;
        test_lds_golden(&mut gpu, &shape);
        n_ok += 1;
        test_lds_isolation(&mut gpu, &shape);
        n_ok += 1;
    }

    println!("\n### Q8 flash-prefill kernel sweep (Task 6, BR/BC-tiled) ###");
    // Fresh `Gpu` per (shape, br, bc) rather than reusing the shared `gpu`
    // from the general/LDS sweeps above.
    //
    // Found empirically while running this sweep (see
    // docs/perf-checkpoints/2026-08-07-flash-prefill-brbc-cache-defect.md
    // for the full writeup and Task 8 warning): `attention_q8_0_flash_prefill_slots`
    // (crates/rdna-compute/src/attention.rs) compiles a BR/BC-templated
    // kernel per `(br, bc)` pair (module name
    // "attention_q8_0_flash_prefill_br{br}_bc{bc}", body `#define BR
    // {br}` / `#define BC {bc}`), but its recompile guard checks
    // `self.functions.contains_key("attention_q8_0_flash_prefill")` — a
    // key that does NOT vary with br/bc. The first (br, bc) pair used by a
    // `Gpu` instance "wins": every later call with a DIFFERENT (br, bc)
    // silently reuses that first compiled kernel binary, while grid/LDS/dpt
    // sizing on the host is computed from the new, ineffective (br, bc).
    // Net effect: a launch believes it covers all query rows but the
    // running kernel's real (stale) BR partitions the grid differently, so
    // rows beyond `grid_x_requested * BR_stale` are silently left
    // unwritten (zero) — a correctness bug, not a multi-slot addressing
    // bug, and NOT what this harness is chartered to test. shapes_prefill()
    // deliberately varies (br, bc) across shapes (realistic — production
    // chunk-prefill sizing varies BR/BC too), so a single shared `Gpu` here
    // would produce false "golden mismatch" failures purely from this
    // confound starting at the second distinct (br, bc) pair. A fresh `Gpu`
    // per (br, bc) keeps each prefill kernel variant's compile-and-launch
    // pairing consistent, isolating what this harness actually tests
    // (multi-slot descriptor addressing) from this separate, pre-existing
    // kernel-cache defect (pre-existing on origin/beta; written up in
    // docs/perf-checkpoints/2026-08-07-flash-prefill-brbc-cache-defect.md,
    // not fixed here — out of Task 7's scope, and the shared compile-cache
    // code path is used by many other kernels. Task 8 sweeps br/bc/TILE_SIZE
    // and MUST read that writeup before trusting any measurement from it).
    // Drain the shared `gpu`'s pool first: `free_tensor` above only recycles
    // buffers into `GpuPool`'s free-lists (real `hipFree` is deferred to
    // `drain_pool`/process exit — see kv_slots.rs / GeneralBatch::free's doc
    // comment), and this run is about to spin up several more `Gpu`
    // instances that need real headroom, not just pool-internal reuse.
    gpu.drain_pool();
    let prefill_shapes: Vec<(Shape, usize, usize)> = if smoke {
        vec![(smoke_shape(), 4, 8)]
    } else if negative_control_active() {
        shapes_prefill().into_iter().filter(|(s, _, _)| uniform_seq_len(s)).collect()
    } else {
        shapes_prefill()
    };
    for (shape, br, bc) in prefill_shapes {
        println!("-- {} br={br} bc={bc}", shape.label());
        let mut pgpu = Gpu::init().expect("gpu init (prefill, fresh per br/bc)");
        n_total += 2;
        test_prefill_golden(&mut pgpu, &shape, br, bc);
        n_ok += 1;
        test_prefill_isolation(&mut pgpu, &shape, br, bc);
        n_ok += 1;
        // Actually release this instance's device memory before the next
        // fresh Gpu::init() — a bare Drop only releases VMM arenas (see
        // Gpu's Drop impl), not pooled buffers, so 24 fresh instances in a
        // row without this would leak real device memory just as surely as
        // the pre-fix version of this harness did.
        pgpu.drain_pool();
    }

    println!("\n{n_ok}/{n_total} test groups passed.");
    println!("ALL SHAPES PASS");
}
