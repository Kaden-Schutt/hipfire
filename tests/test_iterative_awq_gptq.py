#!/usr/bin/env python3
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from mq4_masked_calib import (
    compute_awq_scale_dict_with_raw,
    compute_awq_scales,
    compute_awq_scales_from_hessian,
    load_raw_sumsq_dict,
    quantize_candidate,
    run_iterative_awq_gptq_with_stats_sequence,
    safe_key,
    write_awq_sidecar_hfq,
)


TENSOR_NAME = "model.language_model.layers.0.self_attn.q_proj.weight"


def write_hfq(path: Path, payload: bytes) -> None:
    metadata = json.dumps(
        {"architecture": "qwen3", "config": {"model_type": "qwen3", "num_hidden_layers": 1}},
        separators=(",", ":"),
    ).encode("utf-8")
    raw_name = TENSOR_NAME.encode("utf-8")
    index = bytearray()
    index += struct.pack("<I", 1)
    index += struct.pack("<H", len(raw_name))
    index += raw_name
    index += struct.pack("<B", 13)  # MQ4G256
    index += struct.pack("<B", 2)
    index += struct.pack("<I", 1)
    index += struct.pack("<I", 256)
    index += struct.pack("<I", 256)
    index += struct.pack("<Q", len(payload))
    data_offset = 32 + len(metadata) + len(index)
    path.write_bytes(
        b"HFQM"
        + struct.pack("<I", 1)
        + struct.pack("<I", 5)
        + struct.pack("<I", 1)
        + struct.pack("<Q", 32)
        + struct.pack("<Q", data_offset)
        + metadata
        + index
        + payload
    )


def write_safetensors(path: Path, values: np.ndarray) -> None:
    payload = np.asarray(values, dtype="<f4").reshape(-1).tobytes()
    header = json.dumps(
        {
            TENSOR_NAME: {
                "dtype": "F32",
                "shape": [1, 256],
                "data_offsets": [0, len(payload)],
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


def write_mask(path: Path, base: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "hipfire.astrea.mq4_masked.mask.v0",
                "base": str(base),
                "target": str(base),
                "tensors": [
                    {
                        "hfq_name": TENSOR_NAME,
                        "module_name": "model.layers.0.self_attn.q_proj",
                        "base_quant_type": "MQ4G256",
                        "target_quant_type": "MQ4G256",
                        "base_data_size": 136,
                        "target_data_size": 136,
                        "shape": [1, 256],
                        "numel": 256,
                        "packable_flat_mq4": True,
                        "kind": "linear_2d",
                    }
                ],
            },
            indent=2,
        )
        + "\n"
    )


def write_stats(round_dir: Path, hessian: np.ndarray) -> tuple[Path, Path]:
    round_dir.mkdir(parents=True, exist_ok=True)
    stats_npz = round_dir / "imatrix.npz"
    stats_json = round_dir / "stats.json"
    np.savez_compressed(stats_npz, **{safe_key(TENSOR_NAME): hessian.astype(np.float64)})
    stats_json.write_text(json.dumps({"counts": {TENSOR_NAME: 17}}, indent=2) + "\n")
    return stats_npz, stats_json


def make_hessian(seed: int, *, tilt: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(288, 256)).astype(np.float32)
    x *= np.linspace(1.0, 1.0 + tilt, 256, dtype=np.float32)
    h = (x.T @ x).astype(np.float64)
    h += np.eye(256, dtype=np.float64) * 1.0e-2
    return h.reshape(1, 256, 256)


def make_fixture(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    values = rng.normal(size=(1, 256)).astype(np.float32)
    write_safetensors(source_dir / "model-00001-of-00001.safetensors", values)
    base = tmp_path / "base.hfq"
    write_hfq(base, b"\0" * 136)
    mask = tmp_path / "mask.json"
    write_mask(mask, base)
    return SimpleNamespace(source_dir=source_dir, base=base, mask=mask, values=values)


def quantize_one_shot(tmp_path: Path, fixture, hessian: np.ndarray) -> Path:
    stats_npz, stats_json = write_stats(tmp_path / "one-shot-stats", hessian)
    scales = {TENSOR_NAME: compute_awq_scales_from_hessian(hessian, alpha=0.5)}
    sidecar_base = tmp_path / "one-shot-sidecar-base.hfq"
    write_awq_sidecar_hfq(fixture.base, sidecar_base, scales)
    output = tmp_path / "one-shot.hfq"
    quantize_candidate(
        SimpleNamespace(
            base=str(sidecar_base),
            source_dir=str(fixture.source_dir),
            mask=str(fixture.mask),
            stats_npz=str(stats_npz),
            stats_json=str(stats_json),
            output=str(output),
            out=str(tmp_path / "one-shot.json"),
            clip_ratio=1.0,
            ls_iters=5,
            method="gptq",
            gpu=None,
            awq_aware_hessian=str(sidecar_base),
            gptq_damp=0.01,
            gptq_refit_iters=2,
            skip_unsupported=False,
            max_tensors=None,
            tensor_filter=None,
            exclude_tensor_filter=None,
            tensor_name=None,
            tensor_list=None,
            sort_by="mask-order",
            progress_every=0,
        )
    )
    return output


def test_iterate_round_zero_is_byte_identical_to_one_shot_awq_gptq(tmp_path):
    fixture = make_fixture(tmp_path)
    h0 = make_hessian(1, tilt=0.2)

    one_shot = quantize_one_shot(tmp_path, fixture, h0)
    result = run_iterative_awq_gptq_with_stats_sequence(
        hf_model=str(fixture.source_dir),
        mask_path=str(fixture.mask),
        base_output_dir=str(tmp_path / "iter"),
        stats_sequence=[{TENSOR_NAME: h0}],
        awq_alpha=0.5,
        damping=0.5,
        epsilon=0.01,
        max_rounds=1,
    )

    assert result["rounds"][-1]["round"] == 0
    assert (tmp_path / "iter" / "round_0" / "model.hfq").read_bytes() == one_shot.read_bytes()


def test_damping_zero_keeps_scales_identical_to_round_zero(tmp_path):
    fixture = make_fixture(tmp_path)
    h0 = make_hessian(2, tilt=0.1)
    h1 = make_hessian(3, tilt=1.5)
    h2 = make_hessian(4, tilt=2.0)

    run_iterative_awq_gptq_with_stats_sequence(
        hf_model=str(fixture.source_dir),
        mask_path=str(fixture.mask),
        base_output_dir=str(tmp_path / "iter"),
        stats_sequence=[{TENSOR_NAME: h0}, {TENSOR_NAME: h1}, {TENSOR_NAME: h2}],
        awq_alpha=0.5,
        damping=0.0,
        epsilon=0.0,
        max_rounds=3,
    )

    round0 = np.load(tmp_path / "iter" / "round_0" / "awq_scales.npz")
    round2 = np.load(tmp_path / "iter" / "round_2" / "awq_scales.npz")
    key = f"damped__{safe_key(TENSOR_NAME)}"
    np.testing.assert_array_equal(round2[key], round0[key])


def test_synthetic_damped_fixed_point_deltas_shrink(tmp_path):
    fixture = make_fixture(tmp_path)
    h0 = make_hessian(5, tilt=0.1)
    target = make_hessian(6, tilt=2.5)

    result = run_iterative_awq_gptq_with_stats_sequence(
        hf_model=str(fixture.source_dir),
        mask_path=str(fixture.mask),
        base_output_dir=str(tmp_path / "iter"),
        stats_sequence=[
            {TENSOR_NAME: h0},
            {TENSOR_NAME: target},
            {TENSOR_NAME: target},
            {TENSOR_NAME: target},
        ],
        awq_alpha=0.5,
        damping=0.5,
        epsilon=0.0,
        max_rounds=4,
    )

    deltas = [entry["scale_delta"] for entry in result["rounds"]]
    assert deltas[3] < deltas[2]


def test_iterative_awq_gptq_is_reproducible_for_same_inputs(tmp_path):
    h0 = make_hessian(7, tilt=0.4)
    fixture_a = make_fixture(tmp_path / "a")
    fixture_b = make_fixture(tmp_path / "b")

    run_iterative_awq_gptq_with_stats_sequence(
        hf_model=str(fixture_a.source_dir),
        mask_path=str(fixture_a.mask),
        base_output_dir=str(tmp_path / "a" / "iter"),
        stats_sequence=[{TENSOR_NAME: h0}],
        awq_alpha=0.5,
        damping=0.5,
        epsilon=0.01,
        max_rounds=1,
    )
    run_iterative_awq_gptq_with_stats_sequence(
        hf_model=str(fixture_b.source_dir),
        mask_path=str(fixture_b.mask),
        base_output_dir=str(tmp_path / "b" / "iter"),
        stats_sequence=[{TENSOR_NAME: h0}],
        awq_alpha=0.5,
        damping=0.5,
        epsilon=0.01,
        max_rounds=1,
    )

    assert (tmp_path / "a" / "iter" / "round_0" / "model.hfq").read_bytes() == (
        tmp_path / "b" / "iter" / "round_0" / "model.hfq"
    ).read_bytes()


def test_raw_sumsq_override_loads_and_falls_back_on_missing(tmp_path):
    selected = [
        {"hfq_name": "model.layers.0.self_attn.q_proj.weight"},
        {"hfq_name": "model.layers.0.self_attn.k_proj.weight"},
        {"hfq_name": "model.layers.0.absent.weight"},
    ]
    npz = tmp_path / "raw.npz"
    np.savez_compressed(
        npz,
        **{
            safe_key("model.layers.0.self_attn.q_proj.weight"): np.array([4.0, 1.0, 1.0, 1.0], dtype=np.float32),
            safe_key("model.layers.0.self_attn.k_proj.weight"): np.array([1.0, 1.0, 1.0, 4.0], dtype=np.float32),
        },
    )
    mapping, missing = load_raw_sumsq_dict(str(npz), selected)
    assert set(mapping) == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
    }
    assert missing == ["model.layers.0.absent.weight"]
    assert mapping["model.layers.0.self_attn.q_proj.weight"].dtype == np.float64


def test_raw_sumsq_override_changes_scales_vs_hessian_diag(tmp_path):
    """The override path must produce different scales than the rotated-
    Hessian-diagonal fallback for the SAME tensors. This is the structural
    fix for the run-006 plateau diagnosed in
    docs/investigations/2026-05-18-awq-gptq-sub-0.10-kld/results.md.
    """
    name = "model.layers.0.self_attn.q_proj.weight"
    K = 256  # one group
    # Skewed activation distribution: one channel dominates → AWQ should
    # produce a high pre-scale for that channel.
    raw_sumsq = np.ones(K, dtype=np.float64)
    raw_sumsq[0] = 10_000.0
    # Build a [G=1, K, K] Hessian whose diagonal is the *un-rotated* raw_sumsq
    # but whose off-diagonal mass is large enough that diag(R H R.T) ≠ raw_sumsq.
    rng = np.random.default_rng(0)
    M = rng.standard_normal((K, K)).astype(np.float64)
    H = M @ M.T + np.diag(raw_sumsq)
    hessians = {name: H[np.newaxis, :, :]}
    raw_map = {name: raw_sumsq}

    s_hess = compute_awq_scales_from_hessian(H[np.newaxis, :, :], 0.5)
    s_raw = compute_awq_scales(raw_sumsq, 0.5)
    # Sanity: distinct
    assert np.abs(s_hess - s_raw).max() > 0.1

    out = compute_awq_scale_dict_with_raw(hessians, raw_map, alpha=0.5)
    np.testing.assert_allclose(out[name], s_raw)
    # Missing-from-raw-map case falls through to Hessian-diag
    out2 = compute_awq_scale_dict_with_raw(hessians, {}, alpha=0.5)
    np.testing.assert_allclose(out2[name], s_hess)
