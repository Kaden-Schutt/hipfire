# Copyright (c) Kaden Schutt
"""No-GPU unit tests for the certify target guards (Task 2.4).

resolve.py — symbol→file resolution + DEAD_FILE/NO_OP (the overnight-26-wins-inflation guard: a
kernel BOD names a __global__ SYMBOL whose source often lives in a differently-named base file; a swap
into a file that is NOT compiled-in via include_str! is a no-op recompile whose A/B "win" is a
measurement phantom).

cross_arch.py — preprocessor-invariance: a gfx1201 edit must not change any OTHER arch's device TU.
The hipcc dependency is behind an injectable seam so the diff logic is no-GPU-testable.
"""
import pytest

from autoresearch.ar.certify.resolve import resolve_kernel_file, DeadFile, NoOp
from autoresearch.ar.certify import cross_arch as cx


# ---- resolve ----

def test_resolve_uncompiled_symbol_raises():
    # a kernel symbol with no include_str!'d .hip => DeadFile (the plan's canonical failing case)
    with pytest.raises(DeadFile):
        resolve_kernel_file("definitely_not_a_kernel_xyz", ".")


def test_resolve_compiled_in_file():
    # attention_causal_batched.hip is include_str!'d and defines the symbol -> resolves to itself
    assert resolve_kernel_file("attention_causal_batched", ".") == \
        "kernels/src/attention_causal_batched.hip"


def test_resolve_no_op_when_variant_identical(tmp_path):
    # a variant byte-identical to the baseline version of the resolved file => NoOp
    resolved = "kernels/src/attention_causal_batched.hip"
    with open(resolved, "rb") as f:
        original = f.read()
    variant = tmp_path / "v.hip"
    variant.write_bytes(original)

    def fake_base_bytes(repo, ref_path):
        return original                       # baseline == variant bytes

    with pytest.raises(NoOp):
        resolve_kernel_file("attention_causal_batched", ".", variant=str(variant),
                            base_sha="HEAD", _base_reader=fake_base_bytes)


def test_resolve_real_edit_is_not_no_op(tmp_path):
    variant = tmp_path / "v.hip"
    variant.write_text("__global__ void attention_causal_batched(){ /* changed */ }\n")

    def fake_base_bytes(repo, ref_path):
        return b"__global__ void attention_causal_batched(){ /* original */ }\n"

    assert resolve_kernel_file("attention_causal_batched", ".", variant=str(variant),
                               base_sha="HEAD", _base_reader=fake_base_bytes) == \
        "kernels/src/attention_causal_batched.hip"


# ---- cross_arch ----

def test_cross_arch_skips_arch_suffixed_file():
    # foo.gfx12.hip is already isolated by naming — nothing to check
    assert cx.check_cross_arch("kernels/src/attention_dflash_wmma.gfx12.hip", "gfx1201",
                               ["gfx1100", "gfx1151"], ".") == []


def test_cross_arch_detects_changed_arch():
    # inject a fake preprocessor: gfx1100's TU changes, gfx1151's does not -> only gfx1100 flagged
    def fake_pp(arch, kernel_file, variant, repo, base_sha):
        if arch == "gfx1100":
            return "VARIANT_TU" if variant else "BASE_TU"
        return "SAME_TU"                       # unchanged across base/variant

    changed = cx.check_cross_arch("kernels/src/gemv.hip", "gfx1201", ["gfx1100", "gfx1151"], ".",
                                  base_sha="HEAD", preprocess=fake_pp)
    assert changed == ["gfx1100"]


def test_cross_arch_clean_edit_returns_empty():
    def fake_pp(arch, kernel_file, variant, repo, base_sha):
        return "SAME_TU_FOR_ALL"

    assert cx.check_cross_arch("kernels/src/gemv.hip", "gfx1201", ["gfx1100", "gfx1151"], ".",
                               base_sha="HEAD", preprocess=fake_pp) == []


def test_cross_arch_unattributable_baseline_skipped():
    # if the baseline won't preprocess for an arch (empty TU), we can't attribute -> skip it
    def fake_pp(arch, kernel_file, variant, repo, base_sha):
        return "" if not variant else "SOMETHING"   # empty baseline TU

    assert cx.check_cross_arch("kernels/src/gemv.hip", "gfx1201", ["gfx1100"], ".",
                               base_sha="HEAD", preprocess=fake_pp) == []


def test_cross_arch_missing_hipcc_skips(monkeypatch):
    # no hipcc on the box -> SKIP (return []), never a false CROSS_ARCH
    monkeypatch.setattr(cx.shutil, "which", lambda name: None)
    assert cx.check_cross_arch("kernels/src/gemv.hip", "gfx1201", ["gfx1100"], ".") == []
