// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Static explainer text for Settings keys (5e). A plain, GPU-independent data
//! table: selecting a key in the Settings tab shows what it is, what it does,
//! its tradeoff, the default, when to use it, and any load-bearing interaction
//! (e.g. thinking↔dflash, kv_cache:auto = inherit). Authored as a const table so
//! it is testable without a GPU or a daemon and can be cross-checked against the
//! config defaults.

/// One key's help. Every field is a short, honest sentence (no marketing).
#[derive(Clone, Copy, Debug)]
pub struct KnobInfo {
    pub key: &'static str,
    /// Human-friendly title.
    pub title: &'static str,
    /// What it is (one line).
    pub summary: &'static str,
    /// What it does + the tradeoff (pros/cons).
    pub effect: &'static str,
    /// The default value, as a string (cross-checked against config defaults).
    pub default: &'static str,
    /// When to use / leave it.
    pub when: &'static str,
    /// A load-bearing interaction or caveat, if any.
    pub note: Option<&'static str>,
    /// For enum / bool keys: each selectable value and what it does. Empty for
    /// numeric / free-string keys. Rendered in the explainer so users see what
    /// every option means without leaving the Settings tab.
    pub options: &'static [(&'static str, &'static str)],
}

/// Help for `key`, if a curated entry exists.
pub fn knob_info(key: &str) -> Option<&'static KnobInfo> {
    KNOBS.iter().find(|k| k.key == key)
}

/// Curated, honest help for the Settings keys. Order is not significant (lookup
/// is by key). Defaults here mirror `config::defaults()` — see the cross-check
/// test below so the two can't silently drift.
pub const KNOBS: &[KnobInfo] = &[
    KnobInfo {
        key: "default_model",
        title: "Default model",
        summary: "The model serve pre-warms and chat uses when no model is named.",
        effect: "Picks which weights load. No quality/speed tradeoff by itself — it just selects the model.",
        default: "qwen3.5:9b",
        when: "Set it to the model you use most. Change it from the Models tab.",
        note: None,
        options: &[],
    },
    KnobInfo {
        key: "max_seq",
        title: "Context (KV cache capacity)",
        summary: "Maximum tokens the KV cache is allocated to hold at load.",
        effect: "Larger = longer context, but more VRAM reserved up front. Smaller = less VRAM, shorter ceiling.",
        default: "32768",
        when: "Raise for long documents/agents; lower to fit a tight VRAM budget.",
        note: Some("Allocated at load — changing it takes effect on the next serve/restart."),
        options: &[],
    },
    KnobInfo {
        key: "kv_cache",
        title: "KV cache precision",
        summary: "Precision/memory tradeoff for the attention key/value cache.",
        effect: "Lower precision saves VRAM and can speed decode, at some quality risk. Higher keeps fidelity.",
        default: "auto",
        when: "Leave on auto unless you need to fit a larger context or are A/B-testing precision.",
        note: Some("auto is the inherit sentinel — it resolves to the model's registry default_kv_mode if set, else q8 (the universal default; near-reference and DFlash-safe). A model can ship a compressed default via the registry; it's no longer guessed from the GPU arch."),
        options: &[
            ("auto", "Inherit: registry default_kv_mode, else q8."),
            ("q8", "Near-reference, ~2x vs fp16. The default. DFlash-safe."),
            ("fwht4", "4-bit FWHT-rotated K + Q8 V. DFlash-safe."),
            ("fwht3", "3-bit FWHT-rotated, ~5.5x. Best compressed-yet-DFlash-safe."),
            ("fwht2", "2-bit FWHT. Smallest; for genuinely tight VRAM."),
            ("asym4", "4-bit Givens basis. Legacy; degrades DFlash acceptance."),
            ("asym3", "3-bit Givens. Legacy; avoid with DFlash."),
            ("asym2", "2-bit Givens. Legacy; smallest + lowest quality."),
            ("turbo / turbo2-4", "Legacy aliases for the asym tiers."),
        ],
    },
    KnobInfo {
        key: "kv_adaptive",
        title: "Adaptive KV downshift",
        summary: "Dynamically drops KV precision as context grows, instead of a fixed mode.",
        effect: "Saves VRAM on long sessions by spending bits where they matter. Off = fixed precision. Requires an FWHT K mode (kv_cache fwht*).",
        default: "off",
        when: "Leave off unless you run long contexts on a tight VRAM budget and want graceful downshift.",
        note: Some("Requires an FWHT K mode to take effect. The `advanced:k=<fwht4|fwht3|fwht2>,v=<lloyd4|lloyd3|lloyd2>` combos set explicit K/V floor tiers — all 9 are cyclable here (and via the CLI)."),
        options: &[
            ("off", "No adaptive downshift — fixed KV precision."),
            ("conservative", "Gentle floor: K=fwht4, V=lloyd4 — downshifts least."),
            ("balanced", "Middle floor: K=fwht3, V=lloyd3."),
            ("aggressive", "Deep floor: K=fwht2, V=lloyd2 — saves the most VRAM."),
            ("advanced:k=…,v=…", "Pick the K (fwht4/3/2) and V (lloyd4/3/2) floor tiers explicitly — 9 combos."),
        ],
    },
    KnobInfo {
        key: "dflash_mode",
        title: "Spec decode (DFlash)",
        summary: "Speculative decoding: a small drafter proposes tokens the target verifies in batches.",
        effect: "Big decode speedup on predictable output (code, tool-calls, structured: ~4-5x). On free-form prose the acceptance rate is ~1, so it is a slight loss. Correctness is unchanged (the target verifies every token).",
        default: "off",
        when: "Turn on for code/agentic workloads with a drafter attached; leave off for open-ended chat.",
        note: Some("A thinking BUDGET routes DFlash to plain decode: the dispatch gate bails when max_think_tokens > 0 and the think block isn't closed (the default thinking-on path sets a budget). Turn thinking off — or set its budget to 0 — and use greedy to actually get spec-decode."),
        options: &[
            ("off", "No speculative decoding (plain autoregressive)."),
            ("on", "Use the drafter whenever one is attached + the gate allows."),
            ("auto", "Engage spec-decode when eligible (greedy, thinking-off, arch)."),
        ],
    },
    KnobInfo {
        key: "thinking",
        title: "Thinking (reasoning block)",
        summary: "Whether reasoning models emit a hidden <think> block before the answer.",
        effect: "On = better reasoning on hard prompts, but more tokens/latency and it suppresses spec-decode. Off = faster, leaner, spec-decode eligible.",
        default: "on",
        when: "On for hard reasoning/math; off for throughput, code, or to enable DFlash.",
        note: Some("Interacts with dflash_mode: an active thinking budget (max_think_tokens > 0) makes DFlash fall back to plain decode. Turn thinking off to use spec-decode."),
        options: &[
            ("on", "Emit a hidden think block — stronger reasoning, more tokens, no spec-decode."),
            ("off", "No think block — faster, leaner, spec-decode eligible."),
        ],
    },
    KnobInfo {
        key: "reasoning_effort",
        title: "Reasoning effort",
        summary: "How hard the model is told to think, in its own vocabulary.",
        effect: "Shapes the reasoning up front by setting the model's native effort level, rather than truncating it after the fact. On Qwen3.8 this injects the card's effort instruction; medium is deliberately the unsteered baseline (no instruction at all).",
        default: "auto",
        when: "Prefer this over thinking_budget on any model that has it — lowering effort actually shortens the reasoning, where lowering the budget just cuts it off.",
        note: Some("The offered levels come from the MODEL, probed from its chat template at load, so they differ per model: Qwen3.8 has low/medium/xhigh and rejects high/max outright. A level the model lacks is projected UP to its next real one (high -> xhigh), never down. When a model has no native effort, this falls back to the thinking_budget ladder."),
        options: &[
            ("auto", "Leave the template's own default in force (Qwen3.8: xhigh)."),
            ("none", "No thinking — closed think block; see the thinking knob."),
            ("low", "Brief, focused reasoning; straight to the conclusion."),
            ("medium", "The model's unsteered baseline — no effort instruction."),
            ("high", "Projected up to the model's next real rung if it has no 'high'."),
            ("xhigh", "Thorough analysis: validate assumptions, weigh alternatives."),
            ("max", "Saturates at the model's top rung."),
        ],
    },
    KnobInfo {
        key: "thinking_budget",
        title: "Reasoning budget",
        summary: "Named per-turn cap on tokens the model may spend inside <think>.",
        effect: "Higher budgets allow deeper reasoning at the cost of latency; lower budgets force the model to commit to an answer sooner. Picks a preset instead of a raw token count.",
        default: "med",
        when: "med for everyday use; low for snappy chat; high/xhigh/max for hard reasoning; uncapped only if latency is no concern.",
        note: Some("A CAP, not a reasoning dial — it truncates, it does not shape. On a model with a native reasoning.effort (Qwen3.8, DeepSeek4) set the effort instead: a budget below what the chosen effort spends force-closes <think> mid-thought, so you pay the deep reasoning and lose the conclusion. Effort-native models therefore leave this uncapped by default. Drives max_think_tokens unless a raw max_think_tokens override is set (the advanced knob wins). Only applies when thinking is on; thinking=off caps reasoning to a single token regardless."),
        options: &[
            ("low", "512 tokens — quick answers, minimal reasoning."),
            ("med", "2048 tokens — the default; balanced reasoning vs latency."),
            ("high", "8192 tokens — deeper reasoning for harder prompts."),
            ("xhigh", "24576 tokens — very deep reasoning; watch latency."),
            ("max", "32768 tokens — the ceiling; longest bounded reasoning."),
            ("uncapped", "0 = no limit — model reasons until it stops (watch latency)."),
        ],
    },
    KnobInfo {
        key: "prefill_compression",
        title: "Prefill compression (pflash)",
        summary: "Compresses a long prompt's KV during prefill using a drafter's importance scoring.",
        effect: "Cuts prefill time and KV footprint on long prompts. Costs a little quality on what it drops, and does nothing without a drafter.",
        default: "off",
        when: "auto = compress only past prefill_threshold tokens; always = every prefill; off = never.",
        note: Some("Needs a prefill_drafter (.hfq path). With no drafter set, compression is disabled — the TUI warns when you turn it on."),
        options: &[
            ("off", "Never compress the prefill KV."),
            ("auto", "Compress only when the prompt exceeds prefill_threshold."),
            ("always", "Compress every eligible prefill (skips the threshold gate). Still bypassed without a drafter and on the DFlash/tool-call paths."),
        ],
    },
    KnobInfo {
        key: "prefill_drafter",
        title: "Prefill drafter",
        summary: "Path to the small drafter model (.hfq) that scores prompt importance for pflash.",
        effect: "Required for prefill_compression to engage. Empty disables compression entirely.",
        default: "",
        when: "Set to a drafter .hfq when using prefill compression; leave empty otherwise.",
        note: Some("An empty drafter disables prefill_compression regardless of its setting (the TUI warns when you enable compression without one)."),
        options: &[],
    },
    KnobInfo {
        key: "prefill_threshold",
        title: "Prefill threshold",
        summary: "Token count above which auto-mode prefill compression kicks in.",
        effect: "Higher = only very long prompts compress; lower = compress sooner. Only relevant when prefill_compression=auto.",
        default: "32768",
        when: "Lower it if you want shorter prompts compressed too; raise to restrict to very long contexts.",
        note: None,
        options: &[],
    },
    KnobInfo {
        key: "mtp_mode",
        title: "Multi-token prediction (MTP)",
        summary: "Uses a model's MTP head to predict several tokens per step where supported.",
        effect: "Can raise decode throughput on MTP-capable models with no correctness cost; a no-op on models without an MTP head.",
        default: "auto",
        when: "Leave on auto; force on/off only when A/B-testing throughput.",
        note: None,
        options: &[
            ("off", "No multi-token prediction."),
            ("on", "Force the MTP head where the model has one."),
            ("auto", "Use MTP when supported, else single-token."),
        ],
    },
    KnobInfo {
        key: "mtp_k",
        title: "MTP depth (K)",
        summary: "How many tokens the MTP head proposes per step.",
        effect: "Higher K can accept more per step (faster) but risks lower acceptance on hard text. Only used when MTP is active.",
        default: "3",
        when: "Tune only when measuring MTP throughput; the default is a safe middle.",
        note: None,
        options: &[],
    },
    KnobInfo {
        key: "flash_mode",
        title: "Flash attention mode",
        summary: "Controls the flash-attention path for the Q8 KV cache.",
        effect: "auto picks per arch; always/never force it. Mostly a performance knob; the asym KV modes are flash-only.",
        default: "auto",
        when: "Leave on auto unless debugging an attention path.",
        note: None,
        options: &[
            ("auto", "Engine decides per arch — flash on where it helps (already on by default on gfx11/gfx12)."),
            ("always", "Force flash attention on."),
            ("never", "Disable flash attention (asym modes are flash-only)."),
        ],
    },
    KnobInfo {
        key: "mmq_screen",
        title: "MMQ screening",
        summary: "Mixed-precision matmul screening for quantized GEMMs.",
        effect: "Detects Q8_1 outlier rows and falls back to WMMA. Arch-gated to gfx906 + RDNA3/3.5 (gfx1100-1103/1150-1152) and only fires when MMQ is active.",
        default: "auto",
        when: "Leave on auto unless profiling kernels.",
        note: None,
        options: &[
            ("off", "No outlier sweep — max speed, risks tool-call corruption (#87)."),
            ("on", "Force the per-weight outlier sweep."),
            ("auto", "Screen on the supported arches (gfx906 + RDNA3/3.5) when MMQ is active."),
        ],
    },
    KnobInfo {
        key: "cask",
        title: "CASK (TriAttention eviction)",
        summary: "KV-cache eviction policy that keeps a working set instead of the full context.",
        effect: "Frees KV VRAM on long contexts. Switches between m-fold and drop eviction; only actually evicts when a TriAttention sidecar is attached.",
        default: "false",
        when: "Enable only with a published sidecar and after validating output quality.",
        note: Some("m-fold eviction + DFlash can produce a repetition attractor; validate together. Eviction needs a sidecar path, not just this toggle."),
        options: &[
            ("true", "m-fold eviction policy (needs a sidecar to actually evict)."),
            ("false", "drop-eviction policy — NOT off (eviction is gated by the sidecar, not this flag; clear cask_sidecar to disable)."),
        ],
    },
    KnobInfo {
        key: "dflash_adaptive_b",
        title: "DFlash adaptive batch",
        summary: "Lets DFlash adapt its draft batch size to recent acceptance.",
        effect: "Can improve spec-decode throughput by sizing drafts to how well they're accepted. No correctness effect.",
        default: "true",
        when: "Leave on; only relevant when dflash_mode is active.",
        note: None,
        options: &[
            ("true", "Size the draft batch to recent acceptance."),
            ("false", "Fixed draft batch size."),
        ],
    },
    KnobInfo {
        key: "prompt_normalize",
        title: "Prompt normalization",
        summary: "Collapses 3+ consecutive newlines to 2 before tokenizing.",
        effect: "Stabilizes tokenization so whitespace differences don't swing performance; no meaning change for normal text.",
        default: "true",
        when: "Leave on. Turn off only when raw repeated newlines are semantically load-bearing.",
        note: None,
        options: &[
            ("true", "Collapse 3+ newlines to 2 before tokenizing."),
            ("false", "Leave whitespace as-is (when newlines are load-bearing)."),
        ],
    },
    KnobInfo {
        key: "default_chatml",
        title: "Default ChatML",
        summary: "Whether to apply a ChatML scaffold when a model has no embedded chat template.",
        effect: "Improves chat formatting for templateless models; harmless when a model carries its own template.",
        default: "true",
        when: "Leave on; turn off only for raw/base models where a ChatML scaffold hurts.",
        note: None,
        options: &[
            ("true", "Use a ChatML scaffold as the fallback when a model has no embedded template."),
            ("false", "No fallback scaffold — a templateless model gets none (models with their own template are unaffected either way)."),
        ],
    },
    KnobInfo {
        key: "chat_template",
        title: "Chat template",
        summary: "Path to a custom Jinja chat template (.j2/.jinja). Empty = model default.",
        effect: "Overrides how messages are rendered into the prompt. Wrong templates degrade output.",
        default: "",
        when: "Set only when you need a specific template; leave empty to use the model's own.",
        note: Some("Must point to a file that exists; an invalid path is rejected on save."),
        options: &[],
    },
    KnobInfo {
        key: "temperature",
        title: "Temperature",
        summary: "Sampling randomness for generation.",
        effect: "Lower = more deterministic/repeatable; higher = more varied/creative but less reliable.",
        default: "0.3",
        when: "Low (<=0.3) for code/factual; higher (0.7-1.0) for brainstorming.",
        note: Some("Greedy (~0) is what spec-decode runs at; sampling adds cost."),
        options: &[],
    },
    KnobInfo {
        key: "top_p",
        title: "Top-p (nucleus)",
        summary: "Caps sampling to the smallest set of tokens whose probability sums to p.",
        effect: "Lower = safer, more focused; higher = more diverse. Interacts with temperature.",
        default: "0.8",
        when: "Leave near default; lower it to tighten outputs.",
        note: None,
        options: &[],
    },
    KnobInfo {
        key: "host",
        title: "Serve endpoint",
        summary: "The bind address (and port) of the OpenAI-compatible serve API.",
        effect: "0.0.0.0 listens on all interfaces; 127.0.0.1 is local-only. Chat and API clients connect here.",
        default: "0.0.0.0",
        when: "Use 127.0.0.1 to keep serve private to this machine; 0.0.0.0 to expose it on the network.",
        note: Some("Paired with port (default 11435). Changes take effect on the next serve/restart."),
        options: &[],
    },
    KnobInfo {
        key: "max_tokens",
        title: "Max tokens",
        summary: "Upper bound on tokens generated per response.",
        effect: "Caps response length and runaway generation. No quality effect within the cap.",
        default: "4096",
        when: "Raise for long outputs; lower to bound latency/cost.",
        note: None,
        options: &[],
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hipfire::writer::EDITABLE_FIELDS;

    #[test]
    fn lookup_returns_curated_entry() {
        let dflash = knob_info("dflash_mode").expect("dflash entry");
        assert_eq!(dflash.title, "Spec decode (DFlash)");
        assert!(knob_info("does_not_exist").is_none());
    }

    #[test]
    fn kv_adaptive_has_explainer_with_options() {
        let k = knob_info("kv_adaptive").expect("kv_adaptive explainer");
        // 4 presets + the advanced-family entry.
        assert_eq!(k.options.len(), 5);
        assert!(k.options.iter().any(|(v, _)| *v == "aggressive"));
        assert!(k.options.iter().any(|(v, _)| v.starts_with("advanced")));
        assert!(
            k.note.unwrap().to_lowercase().contains("fwht"),
            "note explains the FWHT-K requirement"
        );
    }

    #[test]
    fn enum_and_bool_knobs_carry_per_option_help() {
        // Every editable enum/bool key's explainer lists what each value does, so
        // the option semantics show in the in-tab pane (no tab switching).
        use crate::hipfire::writer::FieldKind;
        for f in EDITABLE_FIELDS {
            if matches!(f.kind, FieldKind::Enum(_) | FieldKind::Bool) {
                let info = knob_info(f.key).unwrap_or_else(|| panic!("no explainer for {}", f.key));
                assert!(
                    !info.options.is_empty(),
                    "enum/bool key {} has no per-option help",
                    f.key
                );
            }
        }
    }

    #[test]
    fn every_editable_field_has_help() {
        // 5e: selecting any editable key must show an explainer.
        for f in EDITABLE_FIELDS {
            assert!(
                knob_info(f.key).is_some(),
                "no explainer for editable key {}",
                f.key
            );
        }
    }

    #[test]
    fn defaults_match_config_defaults() {
        // STRICT: every knob's `default` must exist in config::defaults() AND
        // match it — no silent skip. config::defaults() is the TUI's mirror of bun
        // CONFIG_DEFAULTS (verified key-by-key), so this also pins knob defaults to
        // the real defaults. A knob without a config default fails loudly so it
        // can't drift unnoticed (this is exactly the gap that hid default_chatml).
        let cfg = crate::hipfire::config::ConfigState::defaults_for_test();
        for k in KNOBS {
            let expected = cfg.get(k.key).unwrap_or_else(|| {
                panic!(
                    "knob {} has no config default to cross-check — add it to config::defaults()",
                    k.key
                )
            });
            assert_eq!(
                &k.default, expected,
                "knob default for {} drifted from config defaults ({} vs {})",
                k.key, k.default, expected
            );
        }
    }

    #[test]
    fn load_bearing_interactions_are_documented() {
        // The two interactions this phase exists to surface.
        assert!(
            knob_info("dflash_mode")
                .unwrap()
                .note
                .unwrap()
                .contains("thinking"),
            "dflash note must mention the thinking interaction"
        );
        assert!(
            knob_info("thinking")
                .unwrap()
                .note
                .unwrap()
                .to_lowercase()
                .contains("dflash")
                || knob_info("thinking")
                    .unwrap()
                    .note
                    .unwrap()
                    .contains("spec-decode"),
            "thinking note must mention the dflash interaction"
        );
        assert!(
            knob_info("kv_cache")
                .unwrap()
                .note
                .unwrap()
                .contains("inherit"),
            "kv_cache note must explain auto = inherit"
        );
    }
}
