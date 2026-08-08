// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#![cfg(feature = "test-utils")]

use hipfire_arch_qwen35::qwen35::{DeltaNetState, Qwen35Config, Qwen35Scratch, Qwen35Weights};
use hipfire_arch_qwen35::test_utils::{
    invalidate_legacy_replay, legacy_delta_net_sequence, raw_delta_net_enabled,
    with_raw_delta_net_scope,
};
use hipfire_runtime::llama::KvCache;
use quote::ToTokens;
use rdna_compute::Gpu;
use syn::visit::Visit;
use syn::{Expr, ExprCall, ExprMatch, ExprMethodCall, ImplItemFn, ItemFn};

const QWEN35_SOURCE: &str = include_str!("../src/qwen35.rs");
const SPECULATIVE_SOURCE: &str = include_str!("../src/speculative.rs");
const MTP_COMPOSE_SOURCE: &str = include_str!("../src/mtp_compose.rs");
const MTP_SPEC_SOURCE: &str = include_str!("../src/mtp_spec.rs");
const MTP_PROBE_SOURCE: &str = include_str!("../src/mtp_probe.rs");
const MTP_SPECULATOR_SOURCE: &str = include_str!("../src/mtp_speculator.rs");
const SPEC_IMPL_SOURCE: &str = include_str!("../src/spec_impl.rs");
const DISPATCH_OPS_SOURCE: &str = include_str!("../../hipfire-dispatch/src/ops/delta_net.rs");
const TEST_UTILS_SOURCE: &str = include_str!("../src/test_utils.rs");

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct InventoryEntry {
    source: &'static str,
    function: String,
    class: &'static str,
    operation: String,
}

fn is_gpu_receiver(expr: &Expr) -> bool {
    match expr {
        Expr::Path(path) => path.path.segments.last().is_some_and(|s| s.ident == "gpu"),
        Expr::Field(field) => is_gpu_receiver(&field.base),
        Expr::Paren(paren) => is_gpu_receiver(&paren.expr),
        _ => false,
    }
}

fn classify(
    source: &'static str,
    function: &str,
    operation: &str,
    arms: &[String],
) -> &'static str {
    let in_moe_arm = arms.iter().any(|p| p.contains("Moe"));
    if source.starts_with("dispatch/") {
        return "dispatch-ops";
    }
    if source == "speculative.rs" {
        return "speculative-replay";
    }
    if source != "qwen35.rs" {
        return "UNCLASSIFIED";
    }
    if function.contains("multi") {
        return if in_moe_arm { "pp-moe" } else { "pp-dense" };
    }
    if function.contains("prefill") {
        return if operation.contains("_tree") {
            if in_moe_arm {
                "prefill-tree-moe"
            } else {
                "prefill-tree-dense"
            }
        } else if in_moe_arm {
            "prefill-moe"
        } else {
            "prefill-dense"
        };
    }
    if function.contains("scratch") {
        return if in_moe_arm {
            "decode-moe"
        } else {
            "decode-dense"
        };
    }
    "UNCLASSIFIED"
}

struct RawCallVisitor {
    source: &'static str,
    function: Vec<String>,
    arms: Vec<String>,
    calls: Vec<InventoryEntry>,
}

impl RawCallVisitor {
    fn new(source: &'static str) -> Self {
        Self {
            source,
            function: Vec::new(),
            arms: Vec::new(),
            calls: Vec::new(),
        }
    }
}

impl<'ast> Visit<'ast> for RawCallVisitor {
    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        self.function.push(node.sig.ident.to_string());
        syn::visit::visit_block(self, &node.block);
        self.function.pop();
    }

    fn visit_impl_item_fn(&mut self, node: &'ast ImplItemFn) {
        self.function.push(node.sig.ident.to_string());
        syn::visit::visit_block(self, &node.block);
        self.function.pop();
    }

    fn visit_expr_match(&mut self, node: &'ast ExprMatch) {
        syn::visit::visit_expr(self, &node.expr);
        for arm in &node.arms {
            self.arms.push(arm.pat.to_token_stream().to_string());
            syn::visit::visit_expr(self, &arm.body);
            self.arms.pop();
        }
    }

    fn visit_expr_method_call(&mut self, node: &'ast ExprMethodCall) {
        let operation = node.method.to_string();
        let is_raw = [
            "fused_sigmoid_alpha_gate",
            "conv1d_silu_split",
            "fused_qk_l2_norm_scale",
            "repeat_interleave_qk",
            "gated_delta_net",
            "gated_norm",
        ]
        .iter()
        .any(|prefix| operation.starts_with(prefix));
        if is_raw && is_gpu_receiver(&node.receiver) {
            let function = self
                .function
                .last()
                .cloned()
                .unwrap_or_else(|| "<module>".into());
            let class = classify(self.source, &function, &operation, &self.arms);
            self.calls.push(InventoryEntry {
                source: self.source,
                function,
                class,
                operation,
            });
        }
        syn::visit::visit_expr_method_call(self, node);
    }
}

fn discover_raw_calls(source: &'static str, text: &str) -> Vec<InventoryEntry> {
    let file = syn::parse_file(text).expect("source fixture must parse as Rust");
    let mut visitor = RawCallVisitor::new(source);
    visitor.visit_file(&file);
    visitor.calls.sort();
    visitor.calls
}

fn discover_inventory(source: &'static str, text: &str) -> Vec<InventoryEntry> {
    let mut records = discover_raw_calls(source, text);
    if source.starts_with("mtp_") {
        records.extend(routes(source, text));
        records.sort();
    }
    records
}

fn all_sources() -> [(&'static str, &'static str); 8] {
    [
        ("qwen35.rs", QWEN35_SOURCE),
        ("speculative.rs", SPECULATIVE_SOURCE),
        ("mtp_compose.rs", MTP_COMPOSE_SOURCE),
        ("mtp_spec.rs", MTP_SPEC_SOURCE),
        ("mtp_probe.rs", MTP_PROBE_SOURCE),
        ("mtp_speculator.rs", MTP_SPECULATOR_SOURCE),
        ("spec_impl.rs", SPEC_IMPL_SOURCE),
        ("dispatch/ops/delta_net.rs", DISPATCH_OPS_SOURCE),
    ]
}

fn expected_entry(
    source: &'static str,
    function: &str,
    class: &'static str,
    operation: &str,
) -> InventoryEntry {
    InventoryEntry {
        source,
        function: function.into(),
        class,
        operation: operation.into(),
    }
}

fn add_expected(
    records: &mut Vec<InventoryEntry>,
    source: &'static str,
    function: &str,
    class: &'static str,
    operation: &str,
    count: usize,
) {
    records.extend((0..count).map(|_| expected_entry(source, function, class, operation)));
}

fn expected_inventory() -> Vec<InventoryEntry> {
    let mut records = Vec::new();

    for (function, operation) in [
        (
            "run_delta_net_batch_with_intent",
            "gated_delta_net_f32_batch_seq",
        ),
        (
            "run_delta_net_batch_with_intent",
            "gated_delta_net_f32_chunked",
        ),
        (
            "run_delta_net_batch_with_intent",
            "gated_delta_net_q8_batch_seq",
        ),
        ("run_delta_net_batch_with_intent", "gated_delta_net_q4"),
        ("run_delta_net_step", "gated_delta_net_f32"),
        ("run_delta_net_step", "gated_delta_net_q8"),
        ("run_delta_net_step", "gated_delta_net_q4"),
        ("run_delta_net_tree", "gated_delta_net_f32_tree_batch_seq"),
        ("run_delta_net_tree", "gated_delta_net_q8_tree_batch_seq"),
    ] {
        add_expected(
            &mut records,
            "dispatch/ops/delta_net.rs",
            function,
            "dispatch-ops",
            operation,
            1,
        );
    }

    for (source, function, class, operation, count) in [
        (
            "mtp_compose.rs",
            "spec_step_dflash_mtp",
            "mtp-prefill",
            "qwen35::forward_prefill_batch",
            1,
        ),
        (
            "mtp_compose.rs",
            "spec_step_dflash_mtp",
            "mtp-replay",
            "tape.replay_gdn",
            1,
        ),
        (
            "mtp_compose.rs",
            "spec_step_dflash_mtp",
            "mtp-verify",
            "speculative::verify_dflash_block",
            1,
        ),
        (
            "mtp_compose.rs",
            "spec_step_dflash_mtp_tree",
            "mtp-prefill",
            "qwen35::forward_prefill_batch",
            2,
        ),
        (
            "mtp_compose.rs",
            "spec_step_dflash_mtp_tree",
            "mtp-replay",
            "tape.replay_gdn",
            1,
        ),
        (
            "mtp_compose.rs",
            "spec_step_dflash_mtp_tree",
            "mtp-verify",
            "speculative::verify_dflash_block_tree",
            1,
        ),
        (
            "mtp_spec.rs",
            "spec_step_mtp",
            "mtp-decode",
            "qwen35::forward_scratch",
            1,
        ),
        (
            "mtp_spec.rs",
            "spec_step_mtp",
            "mtp-prefill",
            "qwen35::forward_prefill_batch",
            1,
        ),
        (
            "mtp_spec.rs",
            "spec_step_mtp",
            "mtp-replay",
            "state.trunk_gdn_tape.replay_gdn",
            1,
        ),
        (
            "mtp_spec.rs",
            "spec_step_mtp_compressed",
            "mtp-decode",
            "qwen35::forward_scratch",
            1,
        ),
        (
            "mtp_spec.rs",
            "spec_step_mtp_compressed",
            "mtp-prefill",
            "qwen35::forward_prefill_batch",
            1,
        ),
        (
            "mtp_spec.rs",
            "spec_step_mtp_compressed_serial",
            "mtp-decode",
            "qwen35::forward_scratch",
            1,
        ),
        (
            "mtp_spec.rs",
            "spec_step_mtp_compressed_serial",
            "mtp-prefill",
            "qwen35::forward_prefill_batch",
            1,
        ),
        (
            "mtp_spec.rs",
            "spec_step_mtp_compressed_serial",
            "mtp-replay",
            "state.trunk_gdn_tape.replay_gdn",
            1,
        ),
        (
            "mtp_probe.rs",
            "mtp_probe_step",
            "mtp-prefill",
            "qwen35::forward_prefill_batch_with_pbs",
            1,
        ),
    ] {
        add_expected(&mut records, source, function, class, operation, count);
    }
    records.sort();
    records
}

fn assert_inventory_exact(source: &'static str, text: &str, expected: &[InventoryEntry]) {
    assert_eq!(
        discover_raw_calls(source, text),
        expected,
        "raw DeltaNet operation inventory mismatch in {source}"
    );
}

fn assert_inventory_matches(discovered: &[InventoryEntry], expected: &[InventoryEntry]) {
    assert_eq!(
        discovered, expected,
        "classified Qwen35 inventory multiset mismatch"
    );
}

struct RouteVisitor {
    source: &'static str,
    function: Vec<String>,
    routes: Vec<InventoryEntry>,
}

fn normalized_tokens<T: ToTokens>(value: &T) -> String {
    value.to_token_stream().to_string().replace(' ', "")
}

impl RouteVisitor {
    fn new(source: &'static str) -> Self {
        Self {
            source,
            function: Vec::new(),
            routes: Vec::new(),
        }
    }
}

impl<'ast> Visit<'ast> for RouteVisitor {
    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        self.function.push(node.sig.ident.to_string());
        syn::visit::visit_block(self, &node.block);
        self.function.pop();
    }

    fn visit_impl_item_fn(&mut self, node: &'ast ImplItemFn) {
        self.function.push(node.sig.ident.to_string());
        syn::visit::visit_block(self, &node.block);
        self.function.pop();
    }

    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        let operation = normalized_tokens(&node.func);
        if matches!(
            operation.as_str(),
            "speculative::verify_dflash_block"
                | "speculative::verify_dflash_block_tree"
                | "qwen35::forward_prefill_batch"
                | "qwen35::forward_prefill_batch_with_pbs"
                | "qwen35::forward_scratch"
        ) {
            let class = if operation.contains("verify") {
                "mtp-verify"
            } else if operation.contains("prefill") {
                "mtp-prefill"
            } else {
                "mtp-decode"
            };
            self.routes.push(InventoryEntry {
                source: self.source,
                function: self
                    .function
                    .last()
                    .cloned()
                    .unwrap_or_else(|| "<module>".into()),
                class,
                operation,
            });
        }
        syn::visit::visit_expr_call(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast ExprMethodCall) {
        if node.method == "replay_gdn" {
            self.routes.push(InventoryEntry {
                source: self.source,
                function: self
                    .function
                    .last()
                    .cloned()
                    .unwrap_or_else(|| "<module>".into()),
                class: "mtp-replay",
                operation: format!("{}.{}", normalized_tokens(&node.receiver), node.method),
            });
        }
        syn::visit::visit_expr_method_call(self, node);
    }
}

fn routes(source: &'static str, text: &str) -> Vec<InventoryEntry> {
    let file = syn::parse_file(text).expect("MTP source fixture must parse as Rust");
    let mut visitor = RouteVisitor::new(source);
    visitor.visit_file(&file);
    visitor.routes.sort();
    visitor.routes
}

#[test]
fn legacy_sequence_accepts_caller_owned_scratch_and_state() {
    let _ = legacy_delta_net_sequence
        as fn(
            &mut Gpu,
            &Qwen35Weights,
            &Qwen35Config,
            u32,
            usize,
            &mut KvCache,
            &mut DeltaNetState,
            &Qwen35Scratch,
        ) -> hip_bridge::HipResult<()>;
}

#[test]
fn legacy_replay_invalidation_is_behavioral_and_unit_testable() {
    let mut graphs = rdna_compute::graph::GraphState {
        capture_mode: false,
        capture_blobs: Vec::new(),
        graph_exec: None,
        captured_graph: None,
        ar_forward_blobs: vec![vec![1, 2, 3]],
        ar_forward_kernel_dirty: false,
        ar_forward_replay_enabled: true,
        ar_graph_eligible: true,
        verify: rdna_compute::graph::PerBGraphCache {
            cache: std::collections::HashMap::new(),
            warmed_up: std::collections::HashSet::new(),
            capturing: None,
            lmhead_argmax: std::collections::HashSet::new(),
        },
        replay: rdna_compute::graph::PerBGraphCache {
            cache: std::collections::HashMap::new(),
            warmed_up: std::collections::HashSet::new(),
            capturing: None,
            lmhead_argmax: std::collections::HashSet::new(),
        },
    };
    invalidate_legacy_replay(&mut graphs);
    assert!(!graphs.ar_graph_eligible);
    assert!(graphs.ar_forward_kernel_dirty);
    assert!(!graphs.ar_forward_replay_enabled);
    assert_eq!(graphs.ar_forward_blobs, vec![vec![1, 2, 3]]);
}

#[test]
fn raw_oracle_is_structurally_independent_of_step_lowering() {
    assert!(TEST_UTILS_SOURCE.contains("raw_delta_net_decode_body"));
    assert!(TEST_UTILS_SOURCE.contains("raw_delta_net_batch_body"));
    assert!(TEST_UTILS_SOURCE.contains("gated_delta_net_f32"));
    assert!(TEST_UTILS_SOURCE.contains("gated_delta_net_q8"));
    assert!(TEST_UTILS_SOURCE.contains("gated_delta_net_q4"));
    assert!(TEST_UTILS_SOURCE.contains("DeltaNetBatchIntent::NormalPrefill"));
    assert!(TEST_UTILS_SOURCE.contains("gated_delta_net_f32_chunked"));
    assert!(TEST_UTILS_SOURCE.contains("raw_delta_net_replay_body"));
    assert!(TEST_UTILS_SOURCE.contains("gated_delta_net_f32_batch_seq"));
    assert!(!TEST_UTILS_SOURCE.contains("build_delta_net_"));
    assert!(!TEST_UTILS_SOURCE.contains("execute_steps"));
    assert!(!TEST_UTILS_SOURCE.contains("Step::"));
}

#[test]
fn qwen35_raw_gate_and_batch_paths_are_step_seams() {
    assert!(QWEN35_SOURCE.contains("raw_delta_net_gate_prep"));
    assert!(QWEN35_SOURCE.contains("raw_delta_net_batch_body"));
    assert!(QWEN35_SOURCE.contains("DeltaNetBatchIntent::NormalPrefill"));
    assert!(QWEN35_SOURCE.contains("Step::RmsnormBatched"));
    assert!(QWEN35_SOURCE.contains("Step::FusedQkvzaBatched"));
    assert!(QWEN35_SOURCE.contains("Step::GemmResidualBatched"));
    assert!(QWEN35_SOURCE.contains("Step::GivensRotateBatched"));
}

#[test]
fn raw_scope_is_behaviorally_callable_for_n1_and_batched_replay_harnesses() {
    assert!(!raw_delta_net_enabled());
    for n in [1usize, 4usize] {
        let observed = with_raw_delta_net_scope(|| {
            assert!(raw_delta_net_enabled());
            // The same scope is used by production outer calls for decode
            // (N=1), prefill/tree (N>1), and speculative replay.
            with_raw_delta_net_scope(|| {
                assert!(raw_delta_net_enabled());
                n
            })
        });
        assert_eq!(observed, n);
        assert!(!raw_delta_net_enabled());
    }
}

#[test]
fn raw_scope_restores_after_panic() {
    assert!(!raw_delta_net_enabled());
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_raw_delta_net_scope(|| {
            assert!(raw_delta_net_enabled());
            panic!("scope restoration probe");
        });
    }));
    assert!(result.is_err());
    assert!(!raw_delta_net_enabled());
}

#[test]
fn classified_delta_net_raw_call_inventory_is_exhaustive() {
    let discovered: Vec<_> = all_sources()
        .iter()
        .flat_map(|&(source, text)| discover_inventory(source, text))
        .collect();
    let mut discovered = discovered;
    discovered.sort();
    assert_inventory_matches(&discovered, &expected_inventory());
}

#[test]
fn mtp_replay_and_prefill_routing_is_covered() {
    let actual: Vec<_> = [
        ("mtp_compose.rs", MTP_COMPOSE_SOURCE),
        ("mtp_spec.rs", MTP_SPEC_SOURCE),
        ("mtp_probe.rs", MTP_PROBE_SOURCE),
    ]
    .into_iter()
    .flat_map(|(source, text)| routes(source, text))
    .collect();
    let mut actual = actual;
    actual.sort();
    let expected: Vec<_> = expected_inventory()
        .into_iter()
        .filter(|entry| entry.source.starts_with("mtp_"))
        .collect();
    assert_inventory_matches(&actual, &expected);
}

#[test]
fn inventory_rejects_removed_expected_record() {
    let expected = expected_inventory();
    let mut actual = expected.clone();
    actual.pop();
    let rejected = std::panic::catch_unwind(|| assert_inventory_matches(&actual, &expected));
    assert!(rejected.is_err(), "removed expected record was accepted");
}

#[test]
fn inventory_rejects_added_record() {
    let expected = expected_inventory();
    let mut actual = expected.clone();
    actual.push(expected_entry(
        "mtp_spec.rs",
        "spec_step_mtp",
        "mtp-verify",
        "speculative::verify_dflash_block",
    ));
    actual.sort();
    let rejected = std::panic::catch_unwind(|| assert_inventory_matches(&actual, &expected));
    assert!(rejected.is_err(), "added inventory record was accepted");
}

#[test]
fn inventory_rejects_misclassified_record() {
    let expected = expected_inventory();
    let mut actual = expected.clone();
    actual[0].class = "misclassified";
    let rejected = std::panic::catch_unwind(|| assert_inventory_matches(&actual, &expected));
    assert!(
        rejected.is_err(),
        "misclassified inventory record was accepted"
    );
}

#[test]
fn unexpected_raw_call_fixture_is_rejected() {
    let fixture = "fn fixture(gpu: &mut Gpu) { gpu.gated_norm_f32(); }";
    let rejected = std::panic::catch_unwind(|| assert_inventory_exact("fixture", fixture, &[]));
    assert!(
        rejected.is_err(),
        "unexpected raw call fixture was accepted"
    );
}
