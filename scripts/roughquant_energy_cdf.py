#!/usr/bin/env python3
"""RoughQuant energy-concentration analysis (no GPU, no forward pass).

Answers "where do we start losing energy?" directly from the activation Hessian
(C = XᵀX, diag = E[x²]) + the bf16 weights, with NO noisy PPL forward pass.

Two views, both as cumulative-energy CDFs over the residual stream:
  - RAW per-channel energy: product ‖W[:,c]‖²·E[x_c²] and activation E[x_c²].
    This is the FOLDABLE basis (channel protection / permutation operate here).
  - EIGENBASIS energy: eigenvalues of each weight's input Hessian. This is the
    PCA-rotated basis (per-weight, dense, NOT foldable for free).

Historical note: the original aggregate CDF read as "raw energy is spread",
but phase2g showed that was an aggregation artifact. Per-input outlier
channels are strong and largely shared across layers, so foldable channel
protection is real. Keep this script as a CDF/eigenbasis inspection helper, not
as the final verdict; see docs/roughquant/README.md and phase2g/phase2h.

Usage: python3 scripts/roughquant_energy_cdf.py [hessian.bin] [hf_model_dir]
"""
import struct, sys, glob, json
import numpy as np

HESS = sys.argv[1] if len(sys.argv) > 1 else \
    f"{__import__('os').path.expanduser('~')}/.hipfire/hessians/qwen3.5-0.8b.hessian.bin"
SNAP = sys.argv[2] if len(sys.argv) > 2 else \
    glob.glob("/srv/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/*/")[0]
DMODEL = 1024


def load_hessian(path, want_full=None):
    """Return {name: diag(np.float64)} and (if name in want_full) {name: full C}."""
    f = open(path, "rb")
    f.read(4); struct.unpack("<I", f.read(4)); nt, = struct.unpack("<Q", f.read(8)); f.read(8)
    diag, full = {}, {}
    for _ in range(nt):
        nl, = struct.unpack("<I", f.read(4)); name = f.read(nl).decode()
        struct.unpack("<I", f.read(4)); k, = struct.unpack("<I", f.read(4)); dt, = struct.unpack("<I", f.read(4))
        sz = 4 if dt == 1 else 8
        a = np.frombuffer(f.read(k * k * sz), dtype=np.float32 if dt == 1 else np.float64).reshape(k, k).astype(np.float64)
        diag[name] = np.ascontiguousarray(a.diagonal())
        if want_full and name in want_full:
            full[name] = 0.5 * (a + a.T)
    return diag, full


def st_loader(snap):
    st = glob.glob(snap.rstrip("/") + "/*.safetensors")[0]
    g = open(st, "rb"); hlen, = struct.unpack("<Q", g.read(8)); hdr = json.loads(g.read(hlen)); base = 8 + hlen
    def load(name):
        m = hdr[name]; s, e = m["data_offsets"]; g.seek(base + s); raw = g.read(e - s); dt = m["dtype"]
        if dt == "BF16":
            u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
            return (u << 16).view(np.float32).reshape(m["shape"])
        if dt == "F32":
            return np.frombuffer(raw, dtype=np.float32).reshape(m["shape"])
        if dt == "F16":
            return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(m["shape"])
        raise ValueError(dt)
    return hdr, load


def cdf_report(label, energy, fracs=(0.01, 0.02, 0.05, 0.10, 0.20, 0.50)):
    idx = np.argsort(energy)[::-1]; cum = np.cumsum(energy[idx]); tot = cum[-1]; n = len(energy)
    print(f"\n=== {label} (sorted desc) ===")
    for fr in fracs:
        i = max(1, int(fr * n)); print(f"  top {fr*100:4.0f}% ({i:5d}) hold {100*cum[i-1]/tot:5.1f}%")
    for thr in (0.90, 0.99):
        i = int(np.searchsorted(cum, thr * tot)) + 1; print(f"  {thr*100:.0f}% energy in top {i} ({100*i/n:.1f}%)")


def main():
    diag, _ = load_hessian(HESS)
    hdr, load = st_loader(SNAP)
    # RAW residual-channel energy (k==DMODEL readers)
    resid_prod = np.zeros(DMODEL); resid_act = np.zeros(DMODEL); nr = 0
    for hname, d in diag.items():
        if d.shape[0] != DMODEL:
            continue
        w = hname + ".weight"
        if w not in hdr:
            continue
        W = load(w)
        if W.shape[1] != DMODEL:
            continue
        cn2 = (W.astype(np.float64) ** 2).sum(0)
        resid_prod += cn2 * d; resid_act += d; nr += 1
    print(f"residual readers aggregated: {nr}")
    cdf_report("RAW residual PRODUCT ‖W‖²·E[x²]", resid_prod)
    cdf_report("RAW residual ACTIVATION E[x²]", resid_act)
    # EIGENBASIS concentration for a few representative inputs
    reps = [n for n in (
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.mlp.gate_proj",
        "model.language_model.layers.0.linear_attn.out_proj",
        "model.language_model.layers.0.mlp.down_proj",
    ) if n in diag]
    _, full = load_hessian(HESS, want_full=set(reps))
    for name in reps:
        ev = np.clip(np.linalg.eigvalsh(full[name])[::-1], 0, None)
        cdf_report(f"EIGENBASIS {name.split('layers.')[1]}", ev)


if __name__ == "__main__":
    main()
