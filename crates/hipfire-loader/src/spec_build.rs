// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Speculative-decode build/glue that lives at the top of the DAG, where both
//! `LoadedModel` and the arch crates are in scope.
//!
//! Contents: the [`Qwen35SlotGuard`] RAII target borrow, and the generic
//! [`build_speculator`] registry that dispatches on draft kind: a loaded DFlash
//! draft → `hipfire_arch_qwen35::dflash_spec::build_dflash_speculator` (the
//! `DflashSpeculator` + `DflashState` now live in the qwen35 crate), else
//! (opt-in) the model-free n-gram drafter (`ChainSpeculator<NgramDrafter>` from
//! `spec_ngram`). The registry is what lets the loader pick a drafter at load
//! time without the daemon learning which ran.

use hipfire_arch_qwen35::dflash_spec::{build_dflash_speculator, DflashState};
use hipfire_arch_qwen35::mtp_head::Qwen35MtpHead;
use hipfire_arch_qwen35::speculative::ModelSlot;
use hipfire_arch_qwen35::Qwen35Bundle;
use hipfire_runtime::spec::{SpecTarget, SpecTargetGuard, Speculator};
use hipfire_runtime::spec_ngram::{ChainSpeculator, NgramDrafter};
use std::any::Any;
use std::path::Path;

/// RAII scope that moves the live `Qwen35Bundle` out of `m.state`, lends it to
/// the spec-decode loop as a [`ModelSlot`], and — on `Drop`, via *every* exit
/// path including `?`, early return, and panic-unwind — restores it into
/// `m.state`.
///
/// This is the single chokepoint that replaces the eight hand-written
/// `m.state = Some(Box::new(..))` reconstructions in the daemon's
/// DFlash loop, structurally eliminating the "forgot to restore on early
/// return" cross-request state-bleed class (#462): there is no longer a code
/// path on which the bundle fails to return to `m.state`.
///
/// The `HfqFile` (an mmap handle that `ModelSlot` carries but the spec kernels
/// never read) is opened **lazily**, on the first [`model_slot`](Self::model_slot)
/// call. Two payoffs: (1) an autoregressive caller that only needs the bundle fields
/// never pays the mmap, and (2) an open failure leaves the bundle parked for
/// `Drop` to restore — so a reopen error can surface as `Err` without ever
/// leaving `m.state == None`.
pub struct Qwen35SlotGuard<'m> {
    state_back: &'m mut Option<Box<dyn hipfire_runtime::arch_model::ArchModel>>,
    model_path: String,
    // `Option` only so `Drop` can move the contents out; it is `Some` for the
    // guard's entire observable lifetime.
    parked: Option<Parked>,
}

// Both variants hold the same ~5.6 KB of live model state by value — that is
// the point: the guard *moves* the bundle, it does not copy it. The two differ
// by only the lazily-opened `HfqFile` handle + name + slot_config (~240 B), so
// boxing to flatten the delta would mean boxing BOTH variants (two ~5.8 KB
// heap alloc/free per generation) for no real saving on a short-lived
// stack-local guard. Keep it inline.
#[allow(clippy::large_enum_variant)]
enum Parked {
    /// The bundle as taken — fields untouched, no `HfqFile` opened yet.
    Bundle(Qwen35Bundle),
    /// The bundle assembled into a `ModelSlot` (HfqFile opened) for the spec
    /// helpers. `Drop` rebuilds the bundle from these fields.
    Slot(ModelSlot),
}

impl<'m> Qwen35SlotGuard<'m> {
    /// Take the `Qwen35Bundle` out of `state`. Returns `Err` (leaving `state`
    /// untouched) if the model is not a loaded Qwen3.5 bundle — note the
    /// `matches!` guard *before* `take()` so a non-Qwen35 model is never moved
    /// out and dropped.
    pub fn take(
        state: &'m mut Option<Box<dyn hipfire_runtime::arch_model::ArchModel>>,
        model_path: &str,
    ) -> Result<Self, String> {
        if !state
            .as_ref()
            .is_some_and(|s| (s.as_ref() as &dyn Any).is::<Qwen35Bundle>())
        {
            return Err("Qwen35SlotGuard: model state is not a loaded Qwen3.5 bundle".into());
        }
        let Some(state_box) = state.take() else {
            unreachable!("guarded by the matches! above")
        };
        let bundle = *(state_box as Box<dyn Any>)
            .downcast::<Qwen35Bundle>()
            .unwrap();
        Ok(Self {
            state_back: state,
            model_path: model_path.to_string(),
            parked: Some(Parked::Bundle(bundle)),
        })
    }

    /// Borrow the target as a concrete [`ModelSlot`], opening the `HfqFile` on
    /// first use. On reopen failure the bundle stays parked (so `Drop` still
    /// restores it) and the error is returned. The arch-erased
    /// [`SpecTargetGuard::slot`] impl upcasts this to `&mut dyn SpecTarget`.
    pub fn model_slot(&mut self) -> Result<&mut ModelSlot, String> {
        if let Some(Parked::Bundle(_)) = self.parked {
            let Some(Parked::Bundle(bundle)) = self.parked.take() else {
                unreachable!("guarded by the if-let above")
            };
            // The bundle→slot field transform + `HfqFile` reopen is qwen35 field
            // knowledge, owned by the arch crate. It is fallible-WITHOUT-loss: on
            // reopen failure the bundle comes back in the `Err`, so we re-park it
            // for `Drop` to restore — never leaving `m.state == None` (#462 guard).
            match ModelSlot::from_bundle(bundle, Path::new(&self.model_path)) {
                Ok(slot) => self.parked = Some(Parked::Slot(slot)),
                Err((bundle, msg)) => {
                    self.parked = Some(Parked::Bundle(bundle));
                    return Err(msg);
                }
            }
        }
        match self.parked.as_mut() {
            Some(Parked::Slot(slot)) => Ok(slot),
            _ => unreachable!("model_slot() leaves `parked` as Slot on success"),
        }
    }
}

impl Drop for Qwen35SlotGuard<'_> {
    fn drop(&mut self) {
        let bundle = match self.parked.take() {
            Some(Parked::Bundle(b)) => b,
            // slot.hfq (mmap), slot.name, slot.slot_config drop inside
            // `into_bundle`; the five live pieces go back into the bundle.
            Some(Parked::Slot(slot)) => slot.into_bundle(),
            None => return, // only reachable if `Drop` ran twice — it cannot.
        };
        *self.state_back = Some(Box::new(bundle));
    }
}

impl SpecTargetGuard for Qwen35SlotGuard<'_> {
    fn slot(&mut self) -> Result<&mut dyn SpecTarget, String> {
        Ok(self.model_slot()? as &mut dyn SpecTarget)
    }
}

// The spec-decode target borrow is now dispatched per-arch by each carrier's
// `Carrier::spec_target_guard` (qwen35 → the move-out + lazy-`HfqFile`-reopen
// [`Qwen35SlotGuard`] above; the pure-attention arms → the generic
// `hipfire_runtime::spec::InPlaceGuard<B>`). The daemon resolves the carrier via
// `hipfire_loader::carrier_for(arch_id)` and never arch-matches itself.

/// Pick the speculative-decode drafter for a freshly-loaded model. This is the
/// single load-time registry the daemon's `generate_dflash` routes through —
/// it never learns which arm was chosen.
///
/// Dispatch:
/// 1. A loaded DFlash draft (`dflash = Some`) → [`DflashSpeculator`].
/// 2. Else, when `speculation.ngram = "on"` and the arch has a `SpecTarget` impl
///    (qwen35 5/6, llama 0/1), the model-free `ChainSpeculator<NgramDrafter>` —
///    spec-decode with no draft model. Opt-in until validated.
/// 3. Otherwise `None` (AR-only).
///
/// **Note (arch_id 0/1 — llama/qwen3):** `LlamaCarrier::load` handles the
/// DSpark sidecar arm BEFORE calling this function.  The precedence for llama
/// is: **DSpark > DFlash > n-gram**.  If a `-dspark` sidecar was loaded the
/// carrier returns early with the `DsparkDrafter` speculator and never calls
/// `build_speculator`.
///
/// The n-gram arm is arch-typeless: it builds its target-side verify scratch
/// lazily on first `prefill` via `SpecTarget::new_spec_scratch`, so this fn needs
/// only `arch_id`, the resolved policy, and the target's `ctx_capacity`. `arch_id`
/// gates which arches the model-free arm is enabled for (qwen35 5/6 today; llama
/// added with its `SpecTarget` impl).
pub fn build_speculator(
    arch_id: u32,
    dflash: Option<DflashState>,
    mtp: Option<Qwen35MtpHead>,
    eviction_is_none: bool,
    ctx_capacity: usize,
    spec: hipfire_runtime::loader_api::SpecLoadCfg,
) -> Option<Box<dyn Speculator>> {
    if let Some(df) = dflash {
        return Some(build_dflash_speculator(df, eviction_is_none));
    }
    // qwen35 MTP head (arch 5/6). MTP wins over n-gram when present; the load
    // site is authoritative for whether a head was loaded (no `&mut Gpu` here
    // to free it). Re-assert the eviction guard here (the MTP head KV is NOT
    // FlashCASK-compacted, so building under eviction would desync head/trunk
    // positions).
    if let Some(head) = mtp {
        if eviction_is_none && matches!(arch_id, 5 | 6) {
            let max_n = spec
                .mtp_k
                .unwrap_or(hipfire_runtime::config::DEFAULT_MTP_K)
                .clamp(1, 10);
            eprintln!("  qwen35 MTP speculator enabled (compressed-serial, K={max_n})");
            return Some(hipfire_arch_qwen35::build_qwen35_mtp_speculator(
                head,
                max_n,
                ctx_capacity,
            ));
        }
        // Declined (eviction/arch mismatch): the load site is authoritative and
        // must NOT load a head we'd decline (no `&mut Gpu` here to free it), so
        // this is unreachable in practice — drop it rather than leak silently.
        eprintln!("  qwen35 MTP head present but arm declined (eviction/arch) — ignored");
        let _ = head;
    }
    // The CLI resolves per-model and global TOML into this per-load policy.
    // Direct protocol clients inherit the daemon's typed process policy before
    // the carrier is invoked, so the loader never consults ambient env.
    let ngram_enabled = spec.ngram_draft.unwrap_or(false);
    // Spec-capable arches with a `SpecTarget` impl: dense LLaMA family
    // (0 = LLaMA/Mistral, 1 = plain Qwen3), qwen35 DeltaNet (5/6), Qwen2
    // (7 = VibeThinker, own `Qwen2State` KV), dots-ocr (8, VL decode-phase),
    // minimax (10), lfm2moe (11, conv-state rollback), cohere2moe (12).
    if ngram_enabled && matches!(arch_id, 0 | 1 | 5 | 6 | 7 | 8 | 10 | 11 | 12) {
        // Default K=12: the batched (weight-BW-bound) verify makes wider draft
        // windows nearly free, and an n-gram K-sweep (vibethinker-3b, 2026-06-23)
        // showed acceptance saturates at K≈12 (tau ~0.38) — K=12 peaks decode
        // tok/s, K=16 ties, K≥24 regresses (wasted verify on drafts past the
        // acceptance plateau). Values are already resolved through the TOML
        // ladder. K keeps its `.max(2)` floor.
        let block_size = spec.ngram_k.unwrap_or(12usize).max(2);
        let min_count = spec.ngram_min_count.unwrap_or(2u32);
        eprintln!(
            "  n-gram speculator enabled (model-free, K={}, min_count={})",
            block_size, min_count
        );
        // `samples` = the target arch implements `SpecTarget::verify_block_sampled`
        // (qwen35 5/6 today). It drives the n-gram speculator's `requires_greedy()`
        // so a temp>0 request only takes the sampled n-gram path where the target
        // can actually sample; elsewhere it routes to AR.
        let samples = matches!(arch_id, 5 | 6);
        return Some(Box::new(ChainSpeculator::new(
            NgramDrafter::new(min_count, block_size),
            block_size,
            ctx_capacity,
            samples,
        )));
    }
    None
}
