# Copyright (c) Kaden Schutt
"""ar.gate.device — resolve an arch (gfxNNNN) to its HIP device index on this box.

A box can host several GPUs (hipx = gfx1100 + gfx1151; hiptrx = 4× gfx1201), and the
gate must pin the CORRECT one — never a fixed index (ROCR reorders across
reboot/hotplug; ``HIP_VISIBLE_DEVICES`` follows ROCR enumeration, not rocm-smi). We
parse ``rocminfo`` (which IS the ROCR order) for GPU agents and return the index of
the first agent whose ``Name: gfxNNNN`` matches the requested arch.

Injectable ``rocminfo_text`` so the parsing is unit-testable without a GPU.
"""
from __future__ import annotations

import subprocess


def _rocminfo() -> str:
    try:
        proc = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=15)
        return proc.stdout if proc.returncode == 0 else ""
    except Exception:
        return ""


def gpu_gfx_order(rocminfo_text: str) -> list[str]:
    """The gfx name of each GPU agent, in ROCR enumeration order (= HIP device order).

    rocminfo prints one ``Agent N`` block per agent; GPU agents carry both a
    ``Device Type: GPU`` line and a ``Name: gfxNNNN`` line. CPU agents (Device Type:
    CPU, Name: the host cpu) are skipped so the returned list index == HIP dev index.
    """
    order: list[str] = []
    cur_name = None
    cur_is_gpu = False
    for raw in rocminfo_text.splitlines():
        line = raw.strip()
        if line.startswith("Agent "):
            cur_name, cur_is_gpu = None, False
            continue
        if line.startswith("Name:"):
            val = line.split(":", 1)[1].strip()
            # The GPU agent's Name is the gfx target (gfx1100 / gfx1201 / …); the
            # Marketing Name line is separate and not matched here.
            if val.startswith("gfx"):
                cur_name = val
        elif line.startswith("Device Type:"):
            cur_is_gpu = line.split(":", 1)[1].strip().upper() == "GPU"
            if cur_is_gpu and cur_name:
                order.append(cur_name)
                cur_name = None
    return order


def resolve_device(arch, *, rocminfo_text=None, default=0, strict=False) -> int:
    """HIP device index for ``arch`` on this box (``default`` if rocminfo is unusable
    or the arch isn't found — a single-GPU box then correctly uses 0).

    ``strict=True`` is the merge-gate contract: an unavailable/hung ``rocminfo`` or
    a missing arch raises instead of silently guessing device 0. Explicit local
    callers retain the legacy fallback unless they opt into strict discovery.
    """
    text = _rocminfo() if rocminfo_text is None else rocminfo_text
    order = gpu_gfx_order(text)
    for i, name in enumerate(order):
        if name == arch:
            return i
    if strict:
        if not order:
            raise RuntimeError("rocminfo unavailable, timed out, or reported no GPU agents")
        raise RuntimeError(f"required arch {arch} absent from rocminfo GPU order {order}")
    return default
