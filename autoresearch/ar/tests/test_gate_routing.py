# Copyright (c) Kaden Schutt
from autoresearch.ar.gate.routing import classify_pr


def test_kernel_change_is_high_risk():
    assert classify_pr(["kernels/src/gemv_hfq4g256_moe_down.hip"]) == "high-risk"


def test_dispatch_and_forward_are_high_risk():
    assert classify_pr(["crates/rdna-compute/src/dispatch.rs"]) == "high-risk"
    assert classify_pr(["crates/hipfire-arch-qwen35/src/forward.rs"]) == "high-risk"
    assert classify_pr(["crates/hipfire-quantize/src/hfq.rs"]) == "high-risk"


def test_high_risk_wins_even_mixed_with_docs():
    assert classify_pr(["docs/x.md", "kernels/src/a.hip"]) == "high-risk"


def test_docs_only_is_trivial():
    assert classify_pr(["docs/specs/x.md", ".github/workflows/ci.yml", "README.md"]) == "trivial"


def test_empty_diff_is_trivial():
    assert classify_pr([]) == "trivial"


def test_small_nonkernel_rust_is_low():
    assert classify_pr(["crates/hipfire-loader/src/lib.rs"], lines_changed=12) == "low"


def test_large_nonkernel_rust_is_moderate():
    assert classify_pr(["crates/hipfire-runtime/src/daemon_util.rs"], lines_changed=300) == "moderate"


def test_unknown_size_nonkernel_defaults_moderate():
    # conservative: if we can't size it, assume moderate (more coverage)
    assert classify_pr(["cli/index.ts"]) == "moderate"
