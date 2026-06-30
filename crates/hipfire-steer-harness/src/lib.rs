// SPDX-License-Identifier: Apache-2.0
//! Daemon-backed adapter for the hipfire-steer driver.
//!
//! The generic driver, scoring, and the [`ModelHarness`] trait live in
//! `hipfire-steer`. This crate drives a real model through a **`hipfire-daemon`
//! subprocess** instead of a reimplemented in-process forward: the daemon already
//! does chat templating, BOS/special-token encoding, prefill, the decode loop,
//! and EOS correctly (`hipfire chat` is coherent on the exact prompts an
//! in-process harness garbled), and the steer hook (`maybe_steer_block`) is
//! compiled into the daemon's arch forward — so the capture/apply session lives
//! in the daemon process. We just expose control over it through the steer
//! protocol ops and route the [`ModelHarness`] trait to them.
//!
//! See `docs/plans/2026-06-30-steer-daemon-pivot.md`.

use std::path::{Path, PathBuf};

use hip_bridge::{HipError, HipResult};
use hipfire_daemon_adapter::DaemonEngine;
use hipfire_daemon_protocol::{KldEvalMode, KldEvalRequest, SteerApplyRequest};
use hipfire_generate::{GenerateTextRequest, GenerationSamplingPolicy};
use hipfire_model::ModelLoadParams;
use hipfire_steer::driver::{ModelHarness, Prompt};
use hipfire_steer::{CaptureMeans, SteerMode, SteerSpec};

/// Map a daemon/client error into the trait's [`HipResult`] error channel. The
/// code is cosmetic (0); the message carries the real context.
fn herr(ctx: &str, e: anyhow::Error) -> HipError {
    HipError::new(0, &format!("{ctx}: {e}"))
}

/// MedGemma 1.5 (and other gemma3 reasoning variants) wrap a `<unused94> …
/// <unused95>` thinking block before the answer; the daemon emits it verbatim
/// (gemma3 ignores `thinking_mode`). Refusal scoring must see the *answer*, not
/// the reasoning trace — so drop everything up to and including the close marker.
/// Non-thinking responses (no marker) pass through unchanged. This is the
/// daemon-path equivalent of the in-process harness's token-level `skip_thinking`.
fn strip_thinking(text: &str) -> &str {
    const THINK_END: &str = "<unused95>";
    match text.rfind(THINK_END) {
        Some(idx) => text[idx + THINK_END.len()..].trim_start(),
        None => text,
    }
}

/// Upper bound on the KLD context window for the capability-damage guard. The
/// daemon clamps this down to the corpus length (`n_ctx.min(tokens.len())`), so a
/// short good-eval set forms one chunk covering the whole corpus instead of
/// flooring to zero chunks — the corpus does NOT need to reach `KLD_N_CTX`. This
/// only caps chunk size for a long good-eval; the `total_scored == 0` guard in
/// `kld_score` then catches only a genuinely empty corpus.
const KLD_N_CTX: usize = 512;

/// A [`ModelHarness`] backed by a `hipfire-daemon` subprocess. Holds the tokio
/// runtime the async [`DaemonEngine`] needs and blocks on each op so the sync
/// driver can drive it. `num_layers`/`hidden` come from the load response and
/// size the capture/derive; KLD scratch (corpus + reference) lives under `tmp`.
pub struct DaemonHarness {
    rt: tokio::runtime::Runtime,
    engine: DaemonEngine,
    num_layers: usize,
    hidden: usize,
    system: String,
    max_new_tokens: usize,
    tmp: PathBuf,
}

impl DaemonHarness {
    /// Spawn a daemon, load `hfq`, and build the harness. `system` is folded into
    /// every captured/generated turn (gemma3 has no system role). `tmp` holds the
    /// transient KLD corpus + reference files.
    pub fn connect(
        daemon_bin: &Path,
        hfq: &Path,
        max_seq: usize,
        max_new_tokens: usize,
        system: String,
        tmp: PathBuf,
    ) -> Result<Self, String> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| format!("steer harness: build tokio runtime: {e}"))?;
        let model = hfq
            .to_str()
            .ok_or_else(|| format!("steer harness: non-UTF8 hfq path {hfq:?}"))?
            .to_string();
        let params = ModelLoadParams {
            max_seq: max_seq.min(u32::MAX as usize) as u32,
            ..Default::default()
        };
        let (engine, loaded) = rt
            .block_on(async {
                let mut engine = DaemonEngine::spawn(daemon_bin).await?;
                let loaded = engine.load(&model, params).await?;
                anyhow::Ok((engine, loaded))
            })
            .map_err(|e| format!("steer harness: spawn+load daemon: {e}"))?;

        let num_layers = loaded
            .layers
            .ok_or_else(|| "steer harness: load response missing 'layers'".to_string())?
            as usize;
        let hidden = loaded
            .dim
            .ok_or_else(|| "steer harness: load response missing 'dim'".to_string())?
            as usize;

        std::fs::create_dir_all(&tmp)
            .map_err(|e| format!("steer harness: create temp dir {tmp:?}: {e}"))?;

        Ok(Self {
            rt,
            engine,
            num_layers,
            hidden,
            system,
            max_new_tokens,
            tmp,
        })
    }

    /// Write the prompts' user text (one turn per line) to the KLD corpus file.
    fn write_kld_corpus(&self, prompts: &[Prompt]) -> HipResult<PathBuf> {
        let path = self.tmp.join("good_eval_corpus.txt");
        let body = prompts
            .iter()
            .map(|p| p.user.replace('\n', " "))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&path, body)
            .map_err(|e| HipError::new(0, &format!("steer harness: write kld corpus: {e}")))?;
        Ok(path)
    }

    fn ref_path(&self) -> PathBuf {
        self.tmp.join("base.kldref")
    }

    // ── LoRA adapter stack control (proxies to the daemon `lora_*` ops) ──────

    /// Load a `.lora` adapter container onto the live model; `scale` overrides the
    /// adapter's default intensity if given.
    pub fn lora_load(&mut self, path: &Path, scale: Option<f32>) -> HipResult<()> {
        let p = path.display().to_string();
        self.rt
            .block_on(self.engine.lora_load(p, scale))
            .map_err(|e| herr("lora_load", e))
    }

    /// Dial a loaded adapter's live intensity.
    pub fn lora_set_scale(&mut self, id: &str, scale: f32) -> HipResult<()> {
        self.rt
            .block_on(self.engine.lora_set_scale(id.to_string(), scale))
            .map_err(|e| herr("lora_set_scale", e))
    }

    /// Drop the whole adapter stack.
    pub fn lora_clear(&mut self) -> HipResult<()> {
        self.rt
            .block_on(self.engine.lora_clear())
            .map_err(|e| herr("lora_clear", e))
    }

    /// `(id, scale)` for each loaded adapter.
    pub fn lora_list(&mut self) -> HipResult<Vec<(String, f32)>> {
        self.rt
            .block_on(self.engine.lora_list())
            .map_err(|e| herr("lora_list", e))
    }
}

impl ModelHarness for DaemonHarness {
    fn num_layers(&self) -> usize {
        self.num_layers
    }
    fn hidden(&self) -> usize {
        self.hidden
    }

    fn begin_capture(&mut self) -> HipResult<()> {
        let (n, h) = (self.num_layers, self.hidden);
        self.rt
            .block_on(self.engine.steer_begin_capture(n, h))
            .map_err(|e| herr("steer_begin_capture", e))
    }

    fn capture(&mut self, prompts: &[Prompt]) -> HipResult<()> {
        for p in prompts {
            let (sys, user) = (self.system.clone(), p.user.clone());
            self.rt
                .block_on(self.engine.steer_capture(sys, user))
                .map_err(|e| herr("steer_capture", e))?;
        }
        Ok(())
    }

    fn finish_capture(&mut self) -> HipResult<CaptureMeans> {
        let means = self
            .rt
            .block_on(self.engine.steer_finish_capture())
            .map_err(|e| herr("steer_finish_capture", e))?;
        Ok(CaptureMeans(means))
    }

    fn begin_apply(&mut self, spec: &SteerSpec) -> HipResult<()> {
        let req = SteerApplyRequest {
            directions: spec.directions.clone(),
            mode: match spec.mode {
                SteerMode::Steer => "steer".to_string(),
                SteerMode::Ablate => "ablate".to_string(),
            },
            strength: spec.strength,
            layer_start: spec.layer_range.start,
            layer_end: spec.layer_range.end,
        };
        self.rt
            .block_on(self.engine.steer_begin_apply(req))
            .map_err(|e| herr("steer_begin_apply", e))
    }

    fn clear(&mut self) -> HipResult<()> {
        self.rt
            .block_on(self.engine.steer_clear())
            .map_err(|e| herr("steer_clear", e))
    }

    fn generate(&mut self, prompts: &[Prompt]) -> HipResult<Vec<String>> {
        let worker = self.engine.worker_key_id.clone();
        let mut out = Vec::with_capacity(prompts.len());
        for (i, p) in prompts.iter().enumerate() {
            // Clear KV between requests so positions don't accumulate across the
            // batch (the daemon generate path is stateful and does not self-reset;
            // the eval executor resets the same way). `reset` leaves the steer
            // session untouched, so an active apply persists across the batch.
            self.rt
                .block_on(self.engine.reset())
                .map_err(|e| herr("generate.reset", e))?;
            let req = GenerateTextRequest::from_prompt(
                format!("steer-gen-{i}"),
                p.user.clone(),
                GenerationSamplingPolicy::greedy(self.max_new_tokens.min(u32::MAX as usize) as u32),
            )
            .with_worker_key_id(worker.clone())
            .with_system(Some(self.system.clone()));
            let (text, _done) = self
                .rt
                .block_on(self.engine.generate(req))
                .map_err(|e| herr("generate", e))?;
            out.push(strip_thinking(&text).to_string());
        }
        Ok(out)
    }

    fn kld_build_ref(&mut self, prompts: &[Prompt]) -> HipResult<()> {
        let corpus = self.write_kld_corpus(prompts)?;
        let req = KldEvalRequest {
            mode: KldEvalMode::BuildRef,
            corpus: Some(corpus.display().to_string()),
            ref_path: Some(self.ref_path().display().to_string()),
            output: None,
            max_chunks: None,
            n_ctx: Some(KLD_N_CTX),
            config: None,
            capture_hidden_layers: false,
            dump_logits: false,
        };
        let resp = self
            .rt
            .block_on(self.engine.kld_eval(req, |_| {}))
            .map_err(|e| herr("kld_build_ref", e))?;
        // The daemon clamps the window to the corpus, so a short good-eval still
        // forms one chunk; zero chunks now means the corpus tokenized to nothing
        // (empty good-eval). Fail loudly rather than silently disabling the guard.
        if resp.n_chunk == 0 {
            return Err(HipError::new(
                0,
                "kld_build_ref: good-eval corpus is empty (0 chunks) — provide good-eval prompts",
            ));
        }
        Ok(())
    }

    fn kld_score(&mut self, _prompts: &[Prompt]) -> HipResult<f32> {
        // Score the steered model against the base reference. The token stream
        // comes from the reference, so no corpus is resent.
        let req = KldEvalRequest {
            mode: KldEvalMode::Score,
            corpus: None,
            ref_path: Some(self.ref_path().display().to_string()),
            output: None,
            max_chunks: None,
            n_ctx: Some(KLD_N_CTX),
            config: None,
            capture_hidden_layers: false,
            dump_logits: false,
        };
        let resp = self
            .rt
            .block_on(self.engine.kld_eval(req, |_| {}))
            .map_err(|e| herr("kld_score", e))?;
        // `mean_kld` defaults to 0.0 when nothing was scored — that is the silent
        // failure that made over-ablation look damage-free. Treat 0 scored as an
        // error so a real 0.0 can only mean "measured, no divergence".
        if resp.total_scored == 0 {
            return Err(HipError::new(
                0,
                "kld_score: scored 0 positions (empty/too-short reference) — KLD guard is blind",
            ));
        }
        Ok(resp.mean_kld.unwrap_or(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::strip_thinking;

    #[test]
    fn strip_thinking_drops_reasoning_keeps_answer() {
        let t = "<unused94> thought planning... <unused95>You should consult a doctor.";
        assert_eq!(strip_thinking(t), "You should consult a doctor.");
    }

    #[test]
    fn strip_thinking_passes_through_plain_answer() {
        let t = "The MRI shows edema in the left temporal lobe.";
        assert_eq!(strip_thinking(t), t);
    }
}
