#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""`arch_id_for` must recognise the Ornith 1.5 family, in BOTH spellings.

The daily registry workflow is fail-closed: an unknown tag family makes
`arch_id_for` return None, which aborts the ENTIRE run and writes nothing —
every other model's entry included, not just the new one. So a missing family
is not a cosmetic omission, it is an outage of the published registry.

Ornith 1.5 is a Qwen3.5-family VL finetune: the 35B-A3B is qwen3_5_moe
(arch 6), the 9B is dense qwen3_5 (arch 5). Keyed on "a3b" exactly like the
qwen3.5 family it derives from.

The canonical tag is the hyphenated `ornith-1.5:35b-a3b`. The artifacts were
briefly published as `ornith1.5-*` and that spelling is kept as an alias, so
both families are mapped. `arch_id_for` splits on the FIRST colon, which means
the hyphen lands inside the family segment — `ornith-1.5:35b-a3b` yields the
family `"ornith-1.5"`, not `"ornith"`. That is the exact trap this file exists
to catch: a rename that updates the tag without updating the family tuple
passes review and takes the whole registry down at 06:23 UTC.
"""
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location(
    "registry_gen", Path(__file__).parent / "registry_gen.py"
)
rg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rg)

CANONICAL = "ornith-1.5:35b-a3b"


def _curated() -> dict:
    return json.loads((REPO_ROOT / "registry" / "models.json").read_text())


def test_canonical_hyphenated_tag_is_arch6():
    # The spelling that actually ships. Family is "ornith-1.5".
    entry = {"file": "ornith-1.5-35b-a3b.mq4"}
    assert rg.arch_id_for(CANONICAL, entry) == 6


def test_legacy_unhyphenated_tag_is_arch6():
    # Kept mapped so reverting to, or re-adding, the old spelling cannot
    # fail-close the daily run.
    entry = {"file": "ornith1.5-35b-a3b.mq4"}
    assert rg.arch_id_for("ornith1.5:35b-a3b", entry) == 6


def test_dense_is_arch5_in_both_spellings():
    # The 9B is dense qwen3_5. Not shipped by this PR, but the mapping must not
    # silently hand it arch 6 if someone adds it later.
    assert rg.arch_id_for("ornith-1.5:9b", {"file": "ornith-1.5-9b.mq4"}) == 5
    assert rg.arch_id_for("ornith1.5:9b", {"file": "ornith1.5-9b.mq4"}) == 5


def test_curated_entry_uses_the_canonical_tag_and_hyphenated_files():
    # Pins the rename itself: repo id, tag, trunk and MTP sidecar all hyphenated,
    # and the byte-size claim that SIZE_TOLERANCE checks against the live HF repo.
    #
    # The repo id is pinned to the CURRENT canonical name. HF redirects the old
    # id, so both resolve today -- but only one of them survives the redirect
    # being dropped, and a curated `repo` that 404s fails the run closed.
    models = _curated()["models"]
    assert CANONICAL in models, "curated entry must use the hyphenated tag"
    assert "ornith1.5:35b-a3b" not in models, "old tag must be an alias, not a model"
    entry = models[CANONICAL]
    assert entry["repo"] == "hipfire-models/ornith-1.5-35b-a3b"
    assert entry["file"] == "ornith-1.5-35b-a3b.mq4"
    assert entry["mtp"]["file"] == "ornith-1.5-35b-a3b.mtp"
    assert entry["size_gb"] == 19.02
    expected_sampling = {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
    }
    assert entry["recommended_settings"] == expected_sampling
    assert entry["sampling_profiles"]["general"] == expected_sampling
    assert entry["sampling_profiles"]["coding"] == expected_sampling
    assert entry["sampling_profiles"]["instruct"] == expected_sampling
    assert "zero MQ4G256V1" in entry["desc"]


def test_legacy_spellings_alias_to_the_canonical_tag():
    # The back-compat surface. Anyone who typed the old tag keeps working.
    aliases = _curated()["aliases"]
    for legacy in ("ornith", "ornith-1.5", "ornith1.5", "ornith1.5:35b-a3b"):
        assert aliases.get(legacy) == CANONICAL, f"{legacy} must alias to {CANONICAL}"


def test_fast_aliases_point_at_the_mq4r_speed_sku():
    # `:fast` is the speed SKU, as for qwen3.8 and muse-glimmer. The bare tag
    # stays on the plain MQ4 trunk; the MQ4V2 router-fix artifact (PR #664,
    # ~206 vs ~140 tok/s AR on gfx1201) is reached through -mq4r or :fast.
    curated = _curated()
    for alias in ("ornith-1.5:fast", "ornith-1.5:35b-a3b-fast"):
        assert curated["aliases"].get(alias) == "ornith-1.5:35b-a3b-mq4r", alias
    assert curated["models"]["ornith-1.5:35b-a3b-mq4r"]["file"] == "ornith-1.5-35b-a3b.mq4r"


def test_every_curated_alias_target_has_an_arch_mapping():
    # An alias may only point at a tag that itself maps, otherwise the alias is
    # a live grenade: it resolves for users but the target aborts the run.
    curated = _curated()
    models = curated["models"]
    for alias, target in curated["aliases"].items():
        assert target in models, f"alias {alias!r} points at missing tag {target!r}"
        assert rg.arch_id_for(target, models[target]) is not None, (
            f"alias {alias!r} -> {target!r}, which has no arch_id mapping"
        )


def test_unknown_family_still_fails_closed():
    # The fail-closed contract itself is worth pinning: if this ever starts
    # returning a default instead of None, an unmapped model would ship with a
    # wrong arch_id rather than stopping the run.
    assert rg.arch_id_for("notamodel:1b", {"file": "x.mq4"}) is None


def test_mq4_extension_is_a_known_quant():
    assert rg.quant_for("ornith-1.5-35b-a3b.mq4") == "mq4"
