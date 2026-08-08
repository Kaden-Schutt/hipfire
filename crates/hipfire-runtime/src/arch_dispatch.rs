// crates/hipfire-runtime/src/arch_dispatch.rs
//! The AR-decode arch abstraction — the decode-loop analog of the load-time
//! `Carrier` and the TP/PP `DenseServed`. Introduced by the god-struct-collapse
//! refactor (docs/superpowers/specs/2026-07-09-daemon-god-struct-archdispatch-design.md).
//! Inc 0 defines only the arch-invariant + reset surface; AR-phase hooks land in Inc 1.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingDefaults {
    pub temp: f32,
    pub top_p: f32,
    pub repeat_penalty: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct ArchFeatures {
    pub supports_think: bool,
    pub supports_stop_seq: bool,
    pub supports_grammar: bool,
    pub supports_vision: bool,
}

/// Grammar constraint for structured decoding (tool-call / DSML).
///
/// The trait is expressed in terms of primitive/std types so the runtime
/// library can name it without taking a dependency on any arch crate (which
/// would introduce a cycle: arch crates already depend on hipfire-runtime).
/// Arch-specific `Matcher` types implement this via a newtype in daemon.rs
/// (the example crate, which has both sides as dev-deps and is therefore
/// free of the cycle).
///
/// Method signatures mirror the real `hipfire_arch_qwen35::grammar::Matcher`
/// exactly:
///   - `token_mask(&self, vocab: &[String], out: &mut [bool])` — populate a
///     per-token boolean allow-mask over the decoded vocab slice.
///   - `advance(&mut self, text: &str)` — commit a decoded token's text,
///     advancing grammar state.
///   - `is_free(&self) -> bool` — true when no constraint is active at the
///     current position (the caller may skip the mask scan).
pub trait GrammarMatcher {
    fn token_mask(&self, vocab: &[String], out: &mut [bool]);
    fn advance(&mut self, text: &str);
    fn is_free(&self) -> bool;
    /// True once the matcher has observed a structural attractor (a decode
    /// loop the grammar state machine can detect). Mirrors the concrete
    /// `Matcher::attractor_detected`. Default `false` for arches whose matcher
    /// doesn't track attractors — the driver's warn at the main-loop advance
    /// site (daemon.rs:9751) then never fires, which is a benign no-op.
    fn attractor_detected(&self) -> bool {
        false
    }
}

/// Device context threaded through the AR-phase forward hooks so ONE driver
/// (`ar_generate`) serves both single-GPU and multi-GPU (deepseek4 EP) arches.
///
/// `Single` carries the daemon-owned bare `Gpu` that the single-GPU arches run
/// on. `Mesh` is a UNIT marker — it carries no device: an EP model's `Gpus`
/// lives INSIDE `LoadedModel` (`m.ep.gpus`), i.e. inside the same `&mut m` the
/// dispatch borrows as `&mut self`, so a `Mesh(&mut Gpus)` variant would alias
/// that borrow (two overlapping `&mut m`). The mesh dispatch therefore reaches
/// its `Gpus` through `&mut self` and treats `Mesh` purely as "I am the mesh
/// arm". Single-GPU dispatches match `Single(gpu)` and run byte-unchanged.
/// See docs/superpowers/specs/2026-07-10-axis-b-modelparallel-collapse-design.md.
pub enum ForwardCtx<'a> {
    Single(&'a mut rdna_compute::Gpu),
    Mesh,
}

impl ForwardCtx<'_> {
    /// Re-borrow the context for a single hook call. `ar_generate` holds one
    /// `ForwardCtx` for the whole decode and hands each forward hook a fresh
    /// short-lived reborrow (the `Single` device can't be moved out per call).
    pub fn reborrow(&mut self) -> ForwardCtx<'_> {
        match self {
            ForwardCtx::Single(g) => ForwardCtx::Single(&mut **g),
            ForwardCtx::Mesh => ForwardCtx::Mesh,
        }
    }
}

/// One trait object per loaded model. Owns arch-specific decode behavior so the
/// daemon's dispatch stops branching on `arch_id`.
pub trait ArchDispatch {
    fn arch_id(&self) -> u32;
    fn eos_token(&self) -> u32;
    /// True when `tok` should terminate AR generation for this arch. The stop set
    /// is arch-specific: qwen35 = primary eos OR the tokenizer's terminator set;
    /// qwen2 = the full `eos_token_ids` config set. Default = primary eos only.
    /// `ar_generate` consults this plus the driver-generic im_end / stop-seq checks
    /// — so the per-arch eos/terminator convention lives here, not hardcoded in the
    /// generic driver.
    fn is_eos(&self, tok: u32) -> bool {
        tok == self.eos_token()
    }
    fn sampling_defaults(&self) -> SamplingDefaults;
    fn features(&self) -> ArchFeatures;
    /// Bridge to the existing spec-decode verify seam; None for AR-only arches.
    fn as_spec_target(&mut self) -> Option<&mut dyn crate::spec::SpecTarget> {
        None
    }

    // ── AR-phase forward hooks (Inc 1, Task 1.4a) ─────────────────────────

    /// Run the prefill forward pass for `chunk` tokens starting at `seq_pos`.
    /// Equivalent to the arch-specific `forward_prefill_batch` call in each
    /// generate arm. Default returns Err so un-ported arches fail loudly if
    /// `ar_generate` is accidentally routed through them.
    #[allow(dead_code)]
    fn prefill_forward(
        &mut self,
        ctx: ForwardCtx<'_>,
        chunk: &[u32],
        seq_pos: usize,
    ) -> Result<(), String> {
        let _ = (ctx, chunk, seq_pos);
        Err("prefill_forward not implemented for this arch".into())
    }

    /// Run a single decode-step forward pass for `token` at `seq_pos`.
    /// Equivalent to the arch-specific `forward_scratch` call in each generate
    /// arm. Default returns Err so un-ported arches fail loudly.
    #[allow(dead_code)]
    fn decode_step_forward(
        &mut self,
        ctx: ForwardCtx<'_>,
        token: u32,
        seq_pos: usize,
    ) -> Result<(), String> {
        let _ = (ctx, token, seq_pos);
        Err("decode_step_forward not implemented for this arch".into())
    }

    /// Build a grammar matcher for the given tool schemas, or return None for
    /// arches without grammar support.
    ///
    /// `tool_schemas` is a slice of `(name, required_fields)` pairs — the
    /// primitive representation used to cross the runtime→arch crate boundary
    /// without naming arch-specific types here. The qwen35 impl converts each
    /// pair to a `hipfire_arch_qwen35::grammar::ToolSchema` before constructing
    /// the `Matcher` newtype.
    #[allow(dead_code)]
    fn init_grammar(
        &self,
        tool_schemas: &[(String, Vec<String>)],
    ) -> Option<Box<dyn GrammarMatcher>> {
        let _ = tool_schemas;
        None
    }

    // ── AR-phase tangle hooks (Inc 1, Task 1.4b) ──────────────────────────
    // The "arch-neutral tangle" (eviction / adaptive-KV / checkpoint / abort)
    // is neutral in LOGIC but operates on the arch bundle's `kv_cache` /
    // `dn_state`, which a generic driver cannot name (they live behind
    // `ModelState::<Arch>`). So each is a hook: the impl field-splits `m`
    // internally (`m.eviction`/`m.kv_adaptive`/`m.prefill_checkpoints` are
    // disjoint from `m.state`), keeping the generic driver arch-blind.

    /// Apply KV eviction after a forward advanced `seq_pos`. Returns
    /// `Ok(Some(new_physical))` when an eviction compacted the cache (the
    /// driver then adopts it as the new physical write slot), `Ok(None)` when
    /// no eviction is configured or none fired. Mirrors the arm's
    /// `m.eviction.maybe_evict(gpu, kv, seq_pos)` sites. Default: no eviction.
    #[allow(dead_code)]
    fn maybe_evict(
        &mut self,
        ctx: ForwardCtx<'_>,
        seq_pos: usize,
    ) -> Result<Option<usize>, String> {
        let _ = (ctx, seq_pos);
        Ok(None)
    }

    /// Downshift adaptive-KV precision if `seq_pos` crossed a capacity
    /// threshold. Logs applied steps to stderr internally (matching the arm),
    /// returns nothing. Mirrors `m.kv_adaptive.maybe_downshift(gpu, kv,
    /// seq_pos)`. Default: no adaptive-KV.
    #[allow(dead_code)]
    fn maybe_adaptive_downshift(&mut self, ctx: ForwardCtx<'_>, seq_pos: usize) {
        let _ = (ctx, seq_pos);
    }

    /// Snapshot the recurrent state at `seq_pos` for later checkpoint-resume.
    /// The driver gates this on its own `ckpt_resume_enabled()` before calling.
    /// Mirrors `speculative::take_dn_checkpoint(&mut m.prefill_checkpoints, dn,
    /// gpu, seq_pos, ckpt_interval(), ckpt_max())`. Default: no-op.
    #[allow(dead_code)]
    fn take_prefill_checkpoint(&mut self, ctx: ForwardCtx<'_>, seq_pos: usize) {
        let _ = (ctx, seq_pos);
    }

    /// Sample one token from the arch's `scratch.logits`. When `grammar_mask`
    /// is `Some`, take the CPU path (download logits, apply the mask, then
    /// `sample_cpu`); otherwise the GPU fast path (`sampler::sample`). The
    /// driver builds the mask via the `GrammarMatcher` and passes it here so
    /// the arch bundle's `scratch` never crosses the trait boundary. Mirrors
    /// the three sample sites (daemon.rs:9119, 9581, 9729). Default fails
    /// loudly — a real arch must sample.
    ///
    /// Receiver is `&mut self` (not `&self`): the mesh (EP) impl reaches its
    /// `Gpus` through `&mut self` (the `Mesh` ctx carries no device). The
    /// single-GPU impls don't mutate self, so this is a behavior-preserving
    /// widening for them.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    fn sample(
        &mut self,
        ctx: ForwardCtx<'_>,
        cfg: &crate::sampler::SamplerConfig,
        vocab_size: usize,
        ngram_scope: &[u32],
        grammar_mask: Option<&[bool]>,
        rng_state: &mut u32,
    ) -> Result<u32, String> {
        let _ = (ctx, cfg, vocab_size, ngram_scope, grammar_mask, rng_state);
        Err("sample not implemented for this arch".into())
    }

    // ── AR-phase driver accessors (Inc 1, Task 1.4b-iii) ──────────────────
    // The generic ar_generate driver keeps loop state (seq_pos, streamed
    // tokens, counters) as locals but still needs a handful of the model's
    // driver-generic fields, which live on `LoadedModel` (a hipfire-loader
    // type the runtime lib cannot name). These accessors expose them via
    // runtime-nameable types so the driver stays arch-blind.

    /// The model's tokenizer. Re-fetched per use in the driver (short shared
    /// borrow) so it never aliases a `&mut self` hook call.
    #[allow(dead_code)]
    fn tokenizer(&self) -> &crate::tokenizer::Tokenizer {
        unimplemented!("tokenizer accessor not implemented for this arch")
    }

    /// The model's cumulative sequence position (physical KV write slot). The
    /// driver seeds its `seq_pos` local from this and writes the final value
    /// back via `set_seq_pos`.
    #[allow(dead_code)]
    fn seq_pos(&self) -> usize {
        0
    }

    /// Write the driver's final `seq_pos` back to the model (the old arm
    /// mutated `m.seq_pos` directly; the driver mirrors it here at finalize /
    /// on abort).
    #[allow(dead_code)]
    fn set_seq_pos(&mut self, seq_pos: usize) {
        let _ = seq_pos;
    }

    /// The running cross-turn conversation-token buffer (pushed per committed
    /// token, extended after prefill, cleared on abort).
    #[allow(dead_code)]
    fn conversation_tokens_mut(&mut self) -> &mut Vec<u32> {
        unimplemented!("conversation_tokens_mut not implemented for this arch")
    }

    /// Vocab size (from the arch config).
    #[allow(dead_code)]
    fn vocab_size(&self) -> usize {
        0
    }

    /// Byte size of the GPU repeat-penalty scratch buffer (used to bound the
    /// effective repeat window). Mirrors `scratch.repeat_buf.buf.size()`.
    #[allow(dead_code)]
    fn repeat_buf_cap_bytes(&self) -> usize {
        0
    }

    /// Output-stream filter config for this arch's decode loop. Default = the
    /// empty pass-through (`EosFilterConfig::default()`), correct for arches
    /// whose eos / special tokens decode to EMPTY text (e.g. ChatML `<|im_end|>`,
    /// which never reaches the visible stream). Arches whose structural eos
    /// marker decodes to a LITERAL that would otherwise leak into visible output
    /// override this to strip it — `ar_generate` commits then emits the eos token
    /// before the `is_eos` break, so the marker must be filtered here (e.g.
    /// MiniMax `[e~[`). Off-arch behavior is unchanged (default is byte-identical
    /// to the prior `EosFilterConfig::default()`).
    #[allow(dead_code)]
    fn eos_filter_config(&self) -> crate::eos_filter::EosFilterConfig {
        crate::eos_filter::EosFilterConfig::default()
    }

    /// The per-request output parser. Default builds a `DefaultStreamParser` from the
    /// driver-supplied config (reproduces the inline emit / stop-seq / think-cap /
    /// n-gram path). Arches with structured output (cohere2moe markers) override.
    #[allow(dead_code)]
    fn stream_parser(
        &self,
        cfg: crate::stream_parser::DefaultStreamParserConfig,
    ) -> Box<dyn crate::stream_parser::StreamParser> {
        Box::new(crate::stream_parser::DefaultStreamParser::new(cfg))
    }

    /// The arch's batched-prefill chunk boundary (qwen35 `PREFILL_MAX_BATCH`).
    #[allow(dead_code)]
    fn prefill_max_batch(&self) -> usize {
        256
    }

    /// Build (once, cached) + return the decoded-vocab table used for grammar
    /// token masks. Mirrors the `m.decoded_vocab` lazy cache.
    #[allow(dead_code)]
    fn ensure_decoded_vocab(&mut self) -> std::sync::Arc<Vec<String>> {
        unimplemented!("ensure_decoded_vocab not implemented for this arch")
    }

    /// True when KV eviction is configured.
    #[allow(dead_code)]
    fn has_eviction(&self) -> bool {
        false
    }

    /// Physical KV capacity (upper bound for the budget-alert headroom check).
    #[allow(dead_code)]
    fn physical_cap(&self) -> usize {
        usize::MAX
    }

    /// The eviction prefill-chunk window (`budget + beta`) when eviction is
    /// configured; `None` selects the plain `prefill_max_batch` chunk path.
    #[allow(dead_code)]
    fn eviction_window(&self) -> Option<usize> {
        None
    }

    /// Store the model's verbatim emitted token sequence under its turn
    /// fingerprint (the asst-turn prompt cache).
    #[allow(dead_code)]
    fn insert_asst_turn(&mut self, fp: u64, seq: Vec<u32>) {
        let _ = (fp, seq);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arch_features_default_is_all_false() {
        let f = ArchFeatures::default();
        assert!(
            !f.supports_think && !f.supports_stop_seq && !f.supports_grammar && !f.supports_vision
        );
    }

    #[test]
    fn sampling_defaults_are_plain_data() {
        let d = SamplingDefaults {
            temp: 0.3,
            top_p: 0.8,
            repeat_penalty: 1.0,
        };
        assert_eq!(
            d,
            SamplingDefaults {
                temp: 0.3,
                top_p: 0.8,
                repeat_penalty: 1.0
            }
        );
    }
}
