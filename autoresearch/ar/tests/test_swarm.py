# Copyright (c) Kaden Schutt
"""No-GPU unit tests for ar.swarm — the config-driven parallel worker launcher.

Ports ``harness/swarm_explore.sh`` but replaces its ``sed``-prompt-munge with the
config ``workers`` list: each worker's ``{card, dev, model, effort}`` flows
STRAIGHT from the TOML into a per-worker launch plan
(``{card, dev, model, effort, worktree, anchor, lockfile}``). The headline
contract (plan Task 5.1) is per-worker heterogeneity WITHOUT any prompt rewrite:
three distinct models/efforts, distinct worktree anchors, per-dev GPU lockfiles.
Every GPU/git/spawn seam is injected so this runs no-GPU.
"""
from autoresearch.ar.config import Bounds, LoopConfig, WorkerCfg
from autoresearch.ar.swarm import launch, plan_workers


def _cfg3():
    """The Sol/Terra/Luna eval config: 3 heterogeneous workers on cards 1/2/3."""
    return LoopConfig(
        "gfx1201",
        "loop/gfx1201",
        "m",
        "q8",
        128,
        "md5",
        3.0,
        5,
        "codex",
        [
            WorkerCfg(1, 1, "gpt-5.6-luna", "max"),
            WorkerCfg(2, 2, "gpt-5.6-terra", "xhigh"),
            WorkerCfg(3, 3, "gpt-5.6-sol", "medium"),
        ],
        Bounds(400, 43200),
    )


# ── plan Task 5.1 Step 1 (the mandated contract) ──────────────────────────────
def test_per_worker_heterogeneity_no_sed():
    cfg = LoopConfig(
        "gfx1201",
        "loop/gfx1201",
        "m",
        "q8",
        128,
        "md5",
        3.0,
        5,
        "codex",
        [
            WorkerCfg(1, 1, "gpt-5.6-luna", "max"),
            WorkerCfg(2, 2, "gpt-5.6-terra", "xhigh"),
            WorkerCfg(3, 3, "gpt-5.6-sol", "medium"),
        ],
        Bounds(400, 43200),
    )
    plans = plan_workers(cfg, "/repo")
    assert [p["model"] for p in plans] == ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"]
    assert [p["effort"] for p in plans] == ["max", "xhigh", "medium"]
    assert plans[0]["lockfile"].endswith("gfx1201-dev1.lock")
    assert plans[2]["worktree"].endswith("sw_card3")


# ── the plan dict is exactly the {card,dev,model,effort,worktree,anchor,lockfile}
#    contract — nothing derived from a prompt sed ──────────────────────────────
def test_plan_dict_has_the_seven_config_derived_keys():
    plans = plan_workers(_cfg3(), "/repo")
    assert len(plans) == 3
    for w, p in zip(_cfg3().workers, plans):
        assert set(p) == {"card", "dev", "model", "effort", "worktree", "anchor", "lockfile"}
        assert p["card"] == w.card and p["dev"] == w.dev
        assert p["model"] == w.model and p["effort"] == w.effort


# ── worktree is .aw/sw_card<card>; anchor is loop/<arch>_w<i> (0-based index);
#    lockfile is per-DEV (/tmp/hipfire-gpu-<arch>-dev<dev>.lock) ────────────────
def test_worktree_anchor_lockfile_derivation():
    plans = plan_workers(_cfg3(), "/repo")
    assert [p["worktree"] for p in plans] == [
        "/repo/.aw/sw_card1",
        "/repo/.aw/sw_card2",
        "/repo/.aw/sw_card3",
    ]
    assert [p["anchor"] for p in plans] == [
        "loop/gfx1201_w0",
        "loop/gfx1201_w1",
        "loop/gfx1201_w2",
    ]
    assert [p["lockfile"] for p in plans] == [
        "/tmp/hipfire-gpu-gfx1201-dev1.lock",
        "/tmp/hipfire-gpu-gfx1201-dev2.lock",
        "/tmp/hipfire-gpu-gfx1201-dev3.lock",
    ]


# ── two workers pinned to the SAME dev share a GPU lockfile (their certifies
#    serialize) but keep DISTINCT worktree anchors (independent kernel gen) ─────
def test_same_dev_shares_lock_distinct_anchor():
    cfg = LoopConfig(
        "gfx1201",
        "loop/gfx1201",
        "m",
        "q8",
        128,
        "md5",
        3.0,
        5,
        "codex",
        [WorkerCfg(1, 0, "a", "max"), WorkerCfg(2, 0, "b", "medium")],
        Bounds(400, 43200),
    )
    plans = plan_workers(cfg, "/repo")
    assert plans[0]["lockfile"] == plans[1]["lockfile"]  # dev0 → serialized certifies
    assert plans[0]["anchor"] != plans[1]["anchor"]  # distinct worktree anchors
    assert plans[0]["worktree"] != plans[1]["worktree"]


# ── launch iterates the plans and returns one pid per worker via the injected
#    spawn seam — the heterogeneous worker cfgs flow through untouched ─────────
def test_launch_returns_pid_per_worker_via_injected_spawn():
    cfg = _cfg3()
    seen = []

    def spawn(cfg_, worker, plan, safety_cap, repo):
        seen.append((worker.model, worker.effort, plan["anchor"], plan["lockfile"], safety_cap, repo))
        return 1000 + len(seen)

    pids = launch(
        cfg,
        "/repo",
        spawn=spawn,
        prepare=lambda *a, **k: None,
        require_worktree=False,
        safety_cap=7,
    )
    assert pids == [1001, 1002, 1003]
    assert [s[0] for s in seen] == ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"]
    assert [s[1] for s in seen] == ["max", "xhigh", "medium"]
    assert [s[2] for s in seen] == ["loop/gfx1201_w0", "loop/gfx1201_w1", "loop/gfx1201_w2"]
    assert all(s[4] == 7 and s[5] == "/repo" for s in seen)  # safety_cap + repo forwarded


# ── require_worktree=True SKIPs a worker whose worktree is absent (bash parity:
#    "w$i SKIP: no worktree") — no spawn, no pid ────────────────────────────────
def test_launch_skips_missing_worktree_when_required(tmp_path):
    repo = str(tmp_path)
    cfg = _cfg3()
    spawned = []

    def spawn(cfg_, worker, plan, safety_cap, repo_):
        spawned.append(worker.card)
        return 42

    pids = launch(cfg, repo, spawn=spawn, prepare=lambda *a, **k: None, require_worktree=True)
    assert pids == []  # no .aw/sw_card*/kernels/src under a fresh tmp repo
    assert spawned == []


# ── prepare seam fires once per launched worker (the anchor-branch ensure) ─────
def test_launch_calls_prepare_per_worker():
    cfg = _cfg3()
    prepared = []
    pids = launch(
        cfg,
        "/repo",
        spawn=lambda *a, **k: 1,
        prepare=lambda c, w, p, r: prepared.append((w.card, p["anchor"])),
        require_worktree=False,
    )
    assert len(pids) == 3
    assert prepared == [(1, "loop/gfx1201_w0"), (2, "loop/gfx1201_w1"), (3, "loop/gfx1201_w2")]
