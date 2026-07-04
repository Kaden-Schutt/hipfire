// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Daemon JSONL protocol contracts.

use serde::{Deserialize, Serialize};

use hipfire_generate::{DoneEvent, ErrorEvent, GenerateTextRequest, TokenEvent, ToolCallsEvent};
use hipfire_kld::KldConfig;
use hipfire_model::{
    AcceleratorInventory, LlmModelRegistry, ModelLoadRequest, ModelLoadedResponse,
};

#[derive(Debug, Deserialize, Serialize)]
pub struct RequestControl {
    pub id: String,
}

/// Calibrate the resident model in place (no reload): run the Tier-1
/// calibration collector over `corpus` and write a unified `.calib.hfq` to
/// `output`. The data plane stays daemon-internal — only this request and the
/// resulting path cross the JSONL boundary. Requires a resident bf16 model
/// (capture fires at the bf16/f16 gemm chokepoints).
#[derive(Debug, Deserialize, Serialize)]
pub struct CollectRequest {
    /// Path to a UTF-8 corpus file (tokenized with the resident tokenizer).
    pub corpus: String,
    /// Path to write the `.calib.hfq` artifact.
    pub output: String,
    /// Max calibration tokens (default 512 if absent).
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Also capture the lm-head top-K KLDREF reference.
    #[serde(default)]
    pub kldref: bool,
}

/// Result of a [`DaemonRequest::Collect`] op.
#[derive(Debug, Deserialize, Serialize)]
pub struct CollectResponse {
    pub output: String,
    pub n_hessian: usize,
    pub n_calib_tokens: usize,
    pub max_consistency: f32,
}

/// What a [`KldEvalRequest`] does against the resident model — without reload.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum KldEvalMode {
    /// Build a KLD reference from the resident model over the corpus → `.kldref`.
    BuildRef,
    /// Score the resident model against an existing reference → `.kldseq`.
    Score,
    /// Build a reference AND score the same resident model against it in one
    /// session — the self-consistency check (must be ≈0). The reference build
    /// and the scoring share the identical resident forward + config, so the
    /// historical two-binary drift is impossible by construction.
    SelfScore,
}

/// Daemon-resident KLD evaluation: reference build and/or candidate scoring
/// against the loaded model, with no reload. The unified scoring core lives in
/// `hipfire-kld`; this request carries the same [`KldConfig`] the math uses, so
/// the wire config and the scoring config cannot diverge.
#[derive(Debug, Deserialize, Serialize)]
pub struct KldEvalRequest {
    pub mode: KldEvalMode,
    /// Corpus/slice file (build_ref / self_score). For `Score` the tokens come
    /// from the reference, so this may be omitted.
    #[serde(default)]
    pub corpus: Option<String>,
    /// Reference path — output for `build_ref`, input for `score`. `self_score`
    /// keeps it in memory unless a path is given (to also persist it).
    #[serde(default)]
    pub ref_path: Option<String>,
    /// Output `.kldseq` (score / self_score).
    #[serde(default)]
    pub output: Option<String>,
    #[serde(default)]
    pub max_chunks: Option<usize>,
    #[serde(default)]
    pub n_ctx: Option<usize>,
    /// Scoring/determinism contract. Absent → the canonical [`KldConfig`] default.
    #[serde(default)]
    pub config: Option<KldConfig>,
    /// Instrument: capture per-layer hidden states for diagnosis.
    #[serde(default)]
    pub capture_hidden_layers: bool,
    /// Instrument: dump per-position logits.
    #[serde(default)]
    pub dump_logits: bool,
}

/// Final result of a [`DaemonRequest::KldEval`] op.
#[derive(Debug, Deserialize, Serialize)]
pub struct KldEvalResponse {
    pub mode: KldEvalMode,
    pub n_chunk: usize,
    pub total_scored: usize,
    /// Slice-mean KLD (score / self_score); `None` for `build_ref`.
    pub mean_kld: Option<f32>,
    pub p99_kld: Option<f32>,
    pub mean_nll: Option<f32>,
    pub ppl: Option<f32>,
    /// Written reference path (`build_ref` / `self_score` if persisted).
    pub ref_output: Option<String>,
    /// Written `.kldseq` path (score / self_score).
    pub seq_output: Option<String>,
    /// `RefMeta` compat findings (score mode): non-empty means the candidate's
    /// (code, config, arch, tokenizer) differed from the reference's.
    #[serde(default)]
    pub compat_findings: Vec<String>,
}

/// Per-chunk streaming progress for a [`DaemonRequest::KldEval`] op.
#[derive(Debug, Deserialize, Serialize)]
pub struct KldChunkEvent {
    pub chunk: usize,
    pub n_chunk: usize,
    pub scored: usize,
    pub mean_kld: f32,
}

/// Begin a steering CAPTURE session: the in-forward `maybe_steer_block` hook
/// accumulates per-block residuals until `SteerFinishCapture`. `num_layers` /
/// `hidden` size the per-block means.
#[derive(Debug, Deserialize, Serialize)]
pub struct SteerBeginCaptureRequest {
    pub num_layers: usize,
    pub hidden: usize,
}

/// Prefill one chat turn (system+user) through the hooked forward — recording the
/// last-prompt-token residual per block — and commit it into the capture means.
/// Prefill-only: no decode (a decoded token's forward would overwrite the
/// captured residual).
#[derive(Debug, Deserialize, Serialize)]
pub struct SteerCaptureRequest {
    #[serde(default)]
    pub system: String,
    pub user: String,
}

/// Begin a steering APPLY session: the in-forward hook steers/ablates each block
/// in `[layer_start, layer_end)` using the per-block `directions`.
#[derive(Debug, Deserialize, Serialize)]
pub struct SteerApplyRequest {
    /// Per-block unit direction (`num_layers × hidden`, row-major).
    pub directions: Vec<Vec<f32>>,
    /// `"steer"` (additive `x += s·v`) or `"ablate"` (projective `x -= s·(v·x)v`).
    pub mode: String,
    pub strength: f32,
    pub layer_start: usize,
    pub layer_end: usize,
}

/// Per-block capture means returned by `SteerFinishCapture`.
#[derive(Debug, Deserialize, Serialize)]
pub struct SteerMeansResponse {
    pub means: Vec<Vec<f32>>,
}

/// Load per-layer `down_proj` column norms (`‖W_down[:,j]‖`) for H-Neurons CETT
/// capture from a host-side binary at `path`
/// (`[u32 n_layers][u32 intermediate][f32 × n_layers*intermediate]`, LE).
#[derive(Debug, Deserialize, Serialize)]
pub struct CettLoadColnormsRequest {
    pub path: String,
}

/// Prefill one framed `(system, user)` turn plus its `response`, and return the
/// per-layer mean-over-response-tokens CETT feature. Requires prior
/// `CettLoadColnorms` and a resident llama backend.
#[derive(Debug, Deserialize, Serialize)]
pub struct CettCaptureRequest {
    #[serde(default)]
    pub system: String,
    pub user: String,
    #[serde(default)]
    pub response: String,
    /// Optional factual-answer-token span WITHIN the response (token offset +
    /// length, from the dataset's tokenized_response/answer_tokens). When set,
    /// CETT is captured over just that span (paper's answer-token features);
    /// otherwise over the whole response.
    #[serde(default)]
    pub answer_offset: Option<usize>,
    #[serde(default)]
    pub answer_len: Option<usize>,
}

/// Set a process-global H-Neuron intervention gain on the resident dense model.
/// `indices` are FLAT feature indices (`layer * intermediate + neuron`) — the
/// positive-weight probe set; each is scaled by `gain` in the FFN forward
/// (prefill + decode). `gain == 1.0` or an empty `indices` clears the session
/// (the identity control point of the dose-response sweep).
#[derive(Debug, Deserialize, Serialize)]
pub struct HneuronInterveneRequest {
    #[serde(default)]
    pub indices: Vec<u32>,
    pub gain: f32,
}

/// Load a `.lora` adapter container (path on the daemon host) onto the APPLY
/// stack. `scale` overrides the adapter's baked-in default intensity if present;
/// `id` renames the adapter on load (so the same container can be stacked under
/// distinct names, e.g. two intensities of one abliteration).
#[derive(Debug, Deserialize, Serialize)]
pub struct LoraLoadRequest {
    pub path: String,
    #[serde(default)]
    pub scale: Option<f32>,
    #[serde(default)]
    pub id: Option<String>,
}

/// Adjust a loaded adapter's live `scale` (intensity).
#[derive(Debug, Deserialize, Serialize)]
pub struct LoraSetScaleRequest {
    pub id: String,
    pub scale: f32,
}

/// Remove a loaded adapter by id.
#[derive(Debug, Deserialize, Serialize)]
pub struct LoraUnloadRequest {
    pub id: String,
}

/// One loaded adapter in a [`LoraListResponse`].
#[derive(Debug, Deserialize, Serialize)]
pub struct LoraAdapterInfo {
    pub id: String,
    pub scale: f32,
}

/// Result of a `LoraList` op: the currently loaded adapter stack.
#[derive(Debug, Deserialize, Serialize)]
pub struct LoraListResponse {
    pub adapters: Vec<LoraAdapterInfo>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DaemonRequest {
    Load(ModelLoadRequest),
    Unload,
    Reset,
    Ping,
    Inventory,
    ModelRegistry,
    Generate(GenerateTextRequest),
    Abort(RequestControl),
    ForceAnswer(RequestControl),
    Collect(CollectRequest),
    KldEval(KldEvalRequest),
    SteerBeginCapture(SteerBeginCaptureRequest),
    SteerCapture(SteerCaptureRequest),
    SteerFinishCapture,
    SteerBeginApply(SteerApplyRequest),
    SteerClear,
    LoraLoad(LoraLoadRequest),
    LoraSetScale(LoraSetScaleRequest),
    LoraUnload(LoraUnloadRequest),
    LoraClear,
    LoraList,
    CettLoadColnorms(CettLoadColnormsRequest),
    CettCapture(CettCaptureRequest),
    HneuronIntervene(HneuronInterveneRequest),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DaemonResponse {
    Loaded(ModelLoadedResponse),
    Unloaded,
    Reset,
    Pong,
    Inventory(AcceleratorInventory),
    ModelRegistry {
        registry: LlmModelRegistry,
    },
    Token(TokenEvent),
    ToolCalls(ToolCallsEvent),
    Done(DoneEvent),
    Error(ErrorEvent),
    Collected(CollectResponse),
    KldChunk(KldChunkEvent),
    KldEvaled(KldEvalResponse),
    SteerCaptured(SteerMeansResponse),
    SteerOk,
    LoraListed(LoraListResponse),
    LoraOk,
    CettOk {
        n_layers: usize,
        intermediate: usize,
    },
    CettFeature {
        feature: Vec<Vec<f32>>,
        #[serde(default)]
        count: usize,
    },
    HneuronOk {
        n_intervened: usize,
        gain: f32,
    },
    #[serde(other)]
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_generate::{GenerateTextRequest, GenerationSamplingPolicy};
    use hipfire_prompt::{Message, Role};
    use serde_json::json;

    #[test]
    fn generate_request_serializes_structured_messages_without_nested_prompt() {
        let req = DaemonRequest::Generate(GenerateTextRequest {
            id: "req-1".to_string(),
            prompt: "last user text".to_string(),
            messages: Some(vec![
                Message {
                    role: Role::System,
                    content: "be brief".to_string(),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                },
                Message {
                    role: Role::User,
                    content: "last user text".to_string(),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                },
            ]),
            sampling: GenerationSamplingPolicy {
                temperature: 0.7,
                max_tokens: 32,
                top_p: None,
                repeat_penalty: None,
            },
            worker_key_id: Some("worker-a".to_string()),
            tools: None,
            system: None,
            stop: None,
            image_base64: None,
            thinking: None,
            thinking_mode: None,
            reasoning_effort: None,
            assistant_prefix: None,
            max_think_tokens: None,
            presence_penalty: None,
            frequency_penalty: None,
            request_id: None,
            evidence_dir: None,
        });

        let value = serde_json::to_value(req).unwrap();
        assert_eq!(value["type"], "generate");
        assert_eq!(value["prompt"], "last user text");
        assert_eq!(value["messages"][0]["role"], "system");
        assert_eq!(value["messages"][1]["content"], "last user text");
        assert!(value.get("tools").is_none());
        assert!(value.get("request_id").is_none());
    }

    #[test]
    fn request_control_messages_match_bun_wire_shape() {
        let abort = serde_json::to_value(DaemonRequest::Abort(RequestControl {
            id: "chatcmpl-1".to_string(),
        }))
        .unwrap();
        assert_eq!(abort, json!({"type": "abort", "id": "chatcmpl-1"}));

        let force = serde_json::to_value(DaemonRequest::ForceAnswer(RequestControl {
            id: "chatcmpl-1".to_string(),
        }))
        .unwrap();
        assert_eq!(force, json!({"type": "force_answer", "id": "chatcmpl-1"}));

        let parsed: DaemonRequest = serde_json::from_value(json!({
            "type": "abort",
            "id": "chatcmpl-1",
            "ignored": true
        }))
        .unwrap();
        let DaemonRequest::Abort(control) = parsed else {
            panic!("expected abort request");
        };
        assert_eq!(control.id, "chatcmpl-1");
    }

    #[test]
    fn generate_request_deserializes_jsonl_wire_shape() {
        let req: DaemonRequest = serde_json::from_value(json!({
            "type": "generate",
            "id": "req-1",
            "prompt": "last user text",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "last user text"}
            ],
            "temperature": 0.7,
            "max_tokens": 32,
            "top_p": 0.8,
            "worker_key_id": "worker-a",
            "ignored_legacy_field": true
        }))
        .unwrap();

        let DaemonRequest::Generate(req) = req else {
            panic!("expected generate request");
        };
        assert_eq!(req.id, "req-1");
        assert_eq!(req.prompt, "last user text");
        assert_eq!(req.messages.as_ref().unwrap()[0].role, Role::System);
        assert_eq!(req.sampling.top_p, Some(0.8));
        assert_eq!(req.worker_key_id.as_deref(), Some("worker-a"));
    }

    #[test]
    fn kld_eval_request_round_trips_with_config() {
        let req: DaemonRequest = serde_json::from_value(json!({
            "type": "kld_eval",
            "mode": "self_score",
            "corpus": "slice.txt",
            "max_chunks": 4,
            "config": { "scoring_mode": "prefill", "top_k": 256, "kv_mode": "fp32",
                        "normalize_prompt": false, "graph": false, "fp32_gqa4_attn": true,
                        "direct_f16kv_attn": false, "reuse_pbs": true, "prefill_max_batch": null },
            "capture_hidden_layers": true,
            "ignored_legacy": 1
        }))
        .unwrap();
        let DaemonRequest::KldEval(req) = req else {
            panic!("expected kld_eval request");
        };
        assert_eq!(req.mode, KldEvalMode::SelfScore);
        assert_eq!(req.max_chunks, Some(4));
        assert!(req.capture_hidden_layers && !req.dump_logits);
        assert!(req.config.as_ref().unwrap().fp32_gqa4_attn);

        // Minimal form: only mode required, config defaults applied downstream.
        let min: DaemonRequest =
            serde_json::from_value(json!({"type": "kld_eval", "mode": "build_ref"})).unwrap();
        let DaemonRequest::KldEval(min) = min else {
            panic!("expected kld_eval");
        };
        assert_eq!(min.mode, KldEvalMode::BuildRef);
        assert!(min.config.is_none() && min.corpus.is_none());
    }

    #[test]
    fn reset_request_and_response_preserve_existing_wire_shape() {
        let value = serde_json::to_value(DaemonRequest::Reset).unwrap();
        assert_eq!(value, json!({"type": "reset"}));

        let response: DaemonResponse =
            serde_json::from_value(json!({"type": "reset", "seq_pos": 0})).unwrap();
        assert!(matches!(response, DaemonResponse::Reset));
    }

    #[test]
    fn inventory_request_and_response_use_shared_model_contract() {
        let value = serde_json::to_value(DaemonRequest::Inventory).unwrap();
        assert_eq!(value, json!({"type": "inventory"}));

        let response: DaemonResponse = serde_json::from_value(json!({
            "type": "inventory",
            "source": "daemon",
            "devices": [{
                "kind": "hip",
                "device_id": "0",
                "ordinal": 0,
                "arch": "gfx1201",
                "name": null,
                "total_memory_bytes": 24000000000u64,
                "integrated": false,
                "runtime": "HIP 6.4",
                "available": true,
                "selected": true,
                "reason": null
            }, {
                "kind": "npu",
                "device_id": "xdna:0",
                "ordinal": 0,
                "arch": "xdna",
                "name": "XDNA NPU",
                "total_memory_bytes": null,
                "integrated": null,
                "runtime": "xdna1_ffi",
                "available": true,
                "selected": false,
                "reason": null
            }]
        }))
        .unwrap();

        let DaemonResponse::Inventory(inventory) = response else {
            panic!("expected inventory response");
        };
        assert_eq!(inventory.source, "daemon");
        assert_eq!(inventory.devices.len(), 2);
        assert_eq!(inventory.devices[0].device_id, "0");
        assert_eq!(inventory.devices[0].arch.as_deref(), Some("gfx1201"));
        assert_eq!(inventory.devices[1].kind, "npu");
        assert_eq!(inventory.devices[1].device_id, "xdna:0");
        assert_eq!(inventory.devices[1].arch.as_deref(), Some("xdna"));
    }

    #[test]
    fn model_registry_request_and_response_use_nested_registry_payload() {
        let value = serde_json::to_value(DaemonRequest::ModelRegistry).unwrap();
        assert_eq!(value, json!({"type": "model_registry"}));

        let response: DaemonResponse = serde_json::from_value(json!({
            "type": "model_registry",
            "registry": {
                "models_dir": "/home/user/.hipfire/models",
                "triattn_dir": "/home/user/.hipfire/triattn",
                "drafts_dir": "/home/user/.hipfire/drafts",
                "templates_dir": "/home/user/.hipfire/templates",
                "models": [],
                "triattn": [],
                "drafts": [],
                "chat_templates": []
            }
        }))
        .unwrap();

        let DaemonResponse::ModelRegistry { registry } = response else {
            panic!("expected model registry response");
        };
        assert_eq!(registry.model_count(), 0);
    }

    #[test]
    fn load_request_deserializes_jsonl_wire_shape() {
        let req: DaemonRequest = serde_json::from_value(json!({
            "type": "load",
            "model": "model.hfq",
            "params": {
                "max_seq": 4096,
                "physical_cap": 2048,
                "kv_cache": "fp16",
                "dflash_mode": "off",
                "draft": "draft.hfq",
                "cask_sidecar": "sidecar.triattn.hfq",
                "mtp_mode": "auto",
                "mtp_k": 3,
                "ignored": true
            },
            "request_id": "load-1",
            "ignored_legacy_field": true
        }))
        .unwrap();

        let DaemonRequest::Load(req) = req else {
            panic!("expected load request");
        };
        assert_eq!(req.model, "model.hfq");
        assert_eq!(req.params.max_seq, 4096);
        assert_eq!(req.params.physical_cap, Some(2048));
        assert_eq!(req.params.kv_cache.as_deref(), Some("fp16"));
        assert_eq!(req.params.dflash_mode.as_deref(), Some("off"));
        assert_eq!(req.params.draft.as_deref(), Some("draft.hfq"));
        assert_eq!(
            req.params.cask_sidecar.as_deref(),
            Some("sidecar.triattn.hfq")
        );
        assert_eq!(req.params.mtp_mode.as_deref(), Some("auto"));
        assert_eq!(req.params.mtp_k, Some(3));
        assert_eq!(req.request_id.as_deref(), Some("load-1"));
    }

    #[test]
    fn done_response_preserves_unknown_metrics_in_extra() {
        let done: DaemonResponse = serde_json::from_value(json!({
            "type": "done",
            "id": "req-1",
            "tokens": 5,
            "tok_s": 10.5,
            "finish_reason": "stop",
            "backend_path": "hip_hipfire_rdna"
        }))
        .unwrap();

        let DaemonResponse::Done(done) = done else {
            panic!("expected done response");
        };
        assert_eq!(done.id, "req-1");
        assert_eq!(done.extra["backend_path"], "hip_hipfire_rdna");
    }

    #[test]
    fn tool_calls_response_deserializes_daemon_shape() {
        let response: DaemonResponse = serde_json::from_value(json!({
            "type": "tool_calls",
            "id": "req-1",
            "calls": [{"name": "lookup", "arguments": {"q": "hipfire"}}]
        }))
        .unwrap();

        let DaemonResponse::ToolCalls(tool_calls) = response else {
            panic!("expected tool_calls response");
        };
        assert_eq!(tool_calls.id, "req-1");
        assert_eq!(tool_calls.calls[0].name, "lookup");
        assert_eq!(tool_calls.calls[0].arguments, json!({"q": "hipfire"}));
    }
}
