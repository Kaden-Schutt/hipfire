#!/usr/bin/env python3
"""
Compute exact VRAM footprint for a hipfire .hfq/.mq4 model file.

Reads the HFQ binary header/tensor index, sums per-tensor GPU memory,
adds known overheads (KV cache, scratch, DeltaNet state), and reports
whether the model fits on a given GPU VRAM budget.

Usage:
    python3 scripts/vram_calc.py path/to/model.mq4 [--vram 16] [--kv-seq 512] [--kv-mode q8]
"""

import struct
import json
import sys
import os
import math
from dataclasses import dataclass
from typing import List, Optional

# ── Quant type constants (from hfq.rs) ──────────────────────────────
QT_F16 = 1
QT_F32 = 2
QT_MQ4 = 13   # MagnumQuant 4-bit FWHT-rotated (G256)
QT_MQ8 = 14   # MagnumQuant 8-bit FWHT-rotated
QT_HFP4 = 21  # HFP4G32
QT_MFP4 = 24  # MFP4G32 (HFP4 + FWHT)
QT_Q8F16 = 3  # Q8_0 block format

# Quant types that stay quantized on GPU (no decompression)
QUANTIZED_ON_GPU = {QT_MQ4, QT_MQ8, QT_HFP4, QT_MFP4, QT_Q8F16}
# Quant types loaded as F32 on GPU (dequantized from F16)
DEQUANT_TO_F32 = {QT_F16}
# Quant types loaded as-is F32
F32_AS_F32 = {QT_F32}


@dataclass
class TensorInfo:
    name: str
    quant_type: int
    shape: List[int]
    data_size: int  # compressed size on disk


def parse_hfq(path: str):
    """Parse HFQ file header and tensor index. Returns (metadata_json, tensors)."""
    with open(path, 'rb') as f:
        header = f.read(32)
        magic = header[0:4]
        assert magic == b'HFQM', f"Not an HFQ file (magic={magic})"
        version = struct.unpack('<I', header[4:8])[0]
        arch_id = struct.unpack('<I', header[8:12])[0]
        n_tensors = struct.unpack('<I', header[12:16])[0]
        meta_off = struct.unpack('<Q', header[16:24])[0]
        data_off = struct.unpack('<Q', header[24:32])[0]

        # Read metadata blob (JSON)
        meta_size = data_off - meta_off
        f.seek(meta_off)
        meta_raw = f.read(meta_size)

        # Find JSON end by brace matching
        depth = 0
        in_str = False
        escaped = False
        json_end = 0
        for i, b in enumerate(meta_raw):
            if escaped:
                escaped = False
                continue
            if b == 0x5c and in_str:  # backslash
                escaped = True
                continue
            if b == 0x22:  # double quote
                in_str = not in_str
                continue
            if not in_str:
                if b == 0x7b:  # {
                    depth += 1
                elif b == 0x7d:  # }
                    depth -= 1
                    if depth == 0:
                        json_end = i + 1
                        break

        metadata_raw = meta_raw[:json_end].decode('utf-8', errors='replace')
        metadata = json.loads(metadata_raw)

        # Parse tensor index
        pos = json_end
        idx_n = struct.unpack('<I', meta_raw[pos:pos+4])[0]
        assert idx_n == n_tensors
        pos += 4

        tensors = []
        for _ in range(n_tensors):
            name_len = struct.unpack('<H', meta_raw[pos:pos+2])[0]
            pos += 2
            name = meta_raw[pos:pos+name_len].decode('utf-8', errors='replace')
            pos += name_len
            qt = meta_raw[pos]
            pos += 1
            n_dims = meta_raw[pos]
            pos += 1
            shape = []
            for _ in range(n_dims):
                shape.append(struct.unpack('<I', meta_raw[pos:pos+4])[0])
                pos += 4
            gs = struct.unpack('<I', meta_raw[pos:pos+4])[0]
            pos += 4
            ds = struct.unpack('<Q', meta_raw[pos:pos+8])[0]
            pos += 8
            tensors.append(TensorInfo(name=name, quant_type=qt, shape=shape, data_size=ds))

        return metadata, tensors, arch_id


def tensor_gpu_bytes(t: TensorInfo) -> int:
    """Compute the GPU VRAM footprint for a single tensor."""
    total_elems = 1
    for d in t.shape:
        total_elems *= d

    if t.quant_type == QT_F16:
        # F16 is dequantized to F32 on GPU → 4 bytes per element
        return total_elems * 4
    elif t.quant_type == QT_F32:
        # F32 stays F32 → 4 bytes per element
        return total_elems * 4
    elif t.quant_type in (QT_Q8F16,):
        # Q8F16 uploads raw without dequantization (for Q8 embeddings etc.)
        return t.data_size
    elif t.quant_type in QUANTIZED_ON_GPU:
        # Stays in quantized format on GPU → data_size is the footprint
        return t.data_size
    else:
        # Unknown format, assume data_size is correct
        return t.data_size


def kv_cache_bytes(n_layers: int, n_kv_heads: int, head_dim: int,
                   seq: int, mode: str) -> int:
    """Compute KV cache VRAM for a given mode and sequence length."""
    if mode == 'q8':
        # Q8_0: 34 bytes per 32 elements
        blocks_per_head = head_dim // 32
        bytes_per_head = blocks_per_head * 34
        bytes_per_layer = n_kv_heads * bytes_per_head * 2 * seq  # K + V
        return bytes_per_layer * n_layers
    elif mode == 'asym3':
        # K: 4 + (head_dim * 3) / 8 bytes per head; V: Q8_0
        k_bph = 4 + (head_dim * 3) // 8
        v_blocks = head_dim // 32
        v_bph = v_blocks * 34
        bytes_per_layer = n_kv_heads * (k_bph + v_bph) * seq
        return bytes_per_layer * n_layers
    elif mode == 'asym4':
        k_bph = 4 + head_dim // 2
        v_blocks = head_dim // 32
        v_bph = v_blocks * 34
        bytes_per_layer = n_kv_heads * (k_bph + v_bph) * seq
        return bytes_per_layer * n_layers
    elif mode == 'hfq4':
        # 8 + head_dim/2 bytes per head
        bph = 8 + head_dim // 2
        bytes_per_layer = n_kv_heads * bph * 2 * seq
        return bytes_per_layer * n_layers
    elif mode == 'fp32':
        # F32 KV: 4 bytes per element
        bytes_per_layer = n_kv_heads * head_dim * 2 * 4 * seq
        return bytes_per_layer * n_layers
    else:
        raise ValueError(f"Unknown KV mode: {mode}")


def scratch_bytes(config: dict, kv_max_seq: int) -> int:
    """Estimate Qwen35Scratch VRAM footprint.

    Based on the actual Qwen35Scratch::new_with_kv_max implementation (qwen35.rs:2823-2899):
    - Persistent decode buffers (all fixed size based on model dims)
    - flash_partials: the only component that scales with kv_max_seq
      Formula: batch_mult * n_heads * max_tiles * (2 + head_dim) * 4
      where max_tiles = ceil(kv_max_seq / 128), batch_mult = 16 (default)
    - prefill_batch is None by default (HIPFIRE_PREFILL_REUSE_PBS opt-in)
    """
    dim = config.get('dim', config.get('hidden_size', 5120))
    hidden_dim = config.get('hidden_dim', config.get('intermediate_size', 13824))
    n_heads = config.get('n_heads', config.get('num_attention_heads', 28))
    n_kv_heads = config.get('n_kv_heads', config.get('num_key_value_heads', 4))
    head_dim = config.get('head_dim', 128)
    vocab_size = config.get('vocab_size', 152064)

    # For DeltaNet (Qwen35 hybrid): linear_num_key_heads and linear_num_value_heads
    # Typical: num_key_heads ≈ 32, num_value_heads ≈ n_kv_heads or n_heads
    k_dim = 32 * head_dim  # typical linear key dim
    v_dim = n_heads * head_dim  # typical linear value dim
    qkv_dim = k_dim * 2 + v_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    # Persistent decode buffers (all F32 = 4 bytes/elem)
    total = 0

    # x, tmp: [dim] F32
    total += 2 * dim * 4
    # pos_buf: 4 bytes
    total += 4

    # DeltaNet temps
    total += qkv_dim * 4  # dn_qkv
    total += v_dim * 4    # dn_z
    total += 32 * 4       # dn_alpha (~linear_num_value_heads)
    total += 32 * 4       # dn_beta
    total += qkv_dim * 4  # dn_conv_out
    total += 4 * v_dim * 4  # dn_q, dn_k, dn_v
    total += 2 * k_dim * 4  # dn_q_raw, dn_k_raw
    total += v_dim * 4    # dn_attn_out
    total += v_dim * 4    # dn_normed

    # FullAttn temps
    total += q_dim * 2 * 4  # fa_q_full
    total += q_dim * 4      # fa_q
    total += q_dim * 4      # fa_gate
    total += kv_dim * 4     # fa_k
    total += kv_dim * 4     # fa_v
    total += q_dim * 4      # fa_attn_out

    # Shared FFN
    total += dim * 4        # o
    total += hidden_dim * 4 # gate_ffn
    total += hidden_dim * 4 # up
    total += hidden_dim * 4 # ffn_hidden
    total += dim * 4        # ffn_out

    # Sampling
    total += vocab_size * 4 # logits
    total += 2 * 4          # sample_buf
    total += 128 * 4        # repeat_buf (128 default, caller-specified)

    # MagnumQuant rotation scratch
    total += max(dim, hidden_dim) * 4  # x_rot

    # Flash attention partials — the ONLY component that scales with kv_max_seq
    tile_size = 128
    max_tiles = (kv_max_seq + tile_size - 1) // tile_size
    batch_mult = 16  # default HIPFIRE_FLASH_PARTIALS_BATCH
    total += batch_mult * n_heads * max_tiles * (2 + head_dim) * 4

    return total


def deltanet_bytes(config: dict) -> int:
    """Estimate DeltaNet state VRAM.

    DeltaNet state includes per-layer s_matrices, s_scales, conv_states.
    For 27B with 28 layers, hidden_dim=5120:
    - s_matrices: n_layers * hidden_dim * state_dim (f32)
    - conv_states: n_layers * hidden_dim * conv_width
    - Scales typically small
    """
    n_layers = config.get('n_layers', config.get('num_hidden_layers', 28))
    hidden_dim = config.get('hidden_dim', config.get('intermediate_size', 13824))
    dim = config.get('dim', config.get('hidden_size', 5120))

    # Rough estimate based on empirical sizes in codebase
    # s_matrices: [n_layers × dim × 4] for state vectors
    s_size = n_layers * dim * 4  # f32

    # conv_states: [n_layers × conv_width × dim × 4]
    conv_width = 4  # typical
    conv_size = n_layers * conv_width * dim * 4

    # s_scales: small per-layer scale vectors
    scale_size = n_layers * 256 * 4  # conservative

    return s_size + conv_size + scale_size


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compute VRAM footprint for HFQ model')
    parser.add_argument('model_path', help='Path to .hfq/.mq4 model file')
    parser.add_argument('--vram', type=float, default=16.0, help='GPU VRAM in GB (default: 16)')
    parser.add_argument('--kv-seq', type=int, default=512, help='KV sequence length for calibration (default: 512)')
    parser.add_argument('--kv-mode', default='q8', choices=['q8', 'asym3', 'asym4', 'fp32'],
                        help='KV cache mode (default: q8)')
    parser.add_argument('--full-seq', type=int, default=0,
                        help='Full KV sequence length without sidecar (default: from config)')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: file not found: {args.model_path}")
        sys.exit(1)

    # Parse
    metadata, tensors, arch_id = parse_hfq(args.model_path)

    # Extract model config (handles multiple nesting patterns)
    config = {}

    def extract_config(src):
        """Recursively extract configuration values from nested dicts."""
        if isinstance(src, dict):
            for key, val in src.items():
                if isinstance(val, dict):
                    # Some models nest config under 'text_config' or 'language_model'
                    extract_config(val)
                elif isinstance(val, (int, float, str)) and key not in config:
                    config[key] = val

    # Start with top-level keys from metadata
    extract_config(metadata)

    # Huggingface convention: config lives in metadata['config']
    if 'config' in metadata and isinstance(metadata['config'], dict):
        extract_config(metadata['config'])

    # Map huggingface keys to our names
    key_map = {
        'num_hidden_layers': 'n_layers',
        'num_attention_heads': 'n_heads',
        'num_key_value_heads': 'n_kv_heads',
        'intermediate_size': 'hidden_dim',
        'hidden_size': 'dim',
        'head_dim': 'head_dim',
        'vocab_size': 'vocab_size',
        'rope_theta': 'rope_theta',
        'max_position_embeddings': 'max_seq_len',
    }
    for hf_key, our_key in key_map.items():
        if our_key not in config and hf_key in config:
            config[our_key] = config[hf_key]

    # Set defaults for missing keys (Qwen3.6-27B typical)
    config.setdefault('n_layers', 28)
    config.setdefault('n_heads', 28)
    config.setdefault('n_kv_heads', 4)
    config.setdefault('head_dim', 128)
    config.setdefault('dim', 5120)
    config.setdefault('hidden_dim', 13824)
    config.setdefault('vocab_size', 152064)
    config.setdefault('max_seq_len', 4096)

    n_layers = config['n_layers']
    n_heads = config['n_heads']
    n_kv_heads = config['n_kv_heads']
    head_dim = config['head_dim']
    hidden_dim = config['hidden_dim']
    dim = config['dim']
    vocab_size = config['vocab_size']

    full_seq = args.full_seq if args.full_seq > 0 else config.get('max_seq_len', 4096)

    # ── Compute VRAM footprint ─────────────────────────────────────

    # 1. Weights
    total_weight_bytes = 0
    weight_details = []
    for t in tensors:
        gpu_bytes = tensor_gpu_bytes(t)
        total_weight_bytes += gpu_bytes

        # Classify tensor
        if 'embed' in t.name.lower() or 'tok_embeddings' in t.name:
            kind = 'embed'
        elif 'output' in t.name.lower() or 'lm_head' in t.name or 'head' in t.name.lower():
            kind = 'lm_head'
        else:
            kind = 'layer'

        gpu_mb = gpu_bytes / (1024 * 1024)
        disk_mb = t.data_size / (1024 * 1024)
        ratio = gpu_bytes / t.data_size if t.data_size > 0 else 1
        weight_details.append((t.name, t.quant_type, kind, disk_mb, gpu_mb, ratio))

    # 2. KV cache
    calib_kv = kv_cache_bytes(n_layers, n_kv_heads, head_dim, args.kv_seq, args.kv_mode)
    full_kv = kv_cache_bytes(n_layers, n_kv_heads, head_dim, full_seq, 'asym3')

    # 3. Scratch at kv_max_seq (calibration uses default 8192)
    calib_scratch = scratch_bytes(config, 8192)

    # 4. DeltaNet state
    dn = deltanet_bytes(config)

    # 5. ROCm/driver overhead (conservative)
    driver_overhead = 256 * 1024 * 1024  # 256 MB

    # ── Summary ─────────────────────────────────────────────────────

    vram_bytes = int(args.vram * 1024**3)

    calib_total = total_weight_bytes + calib_kv + calib_scratch + dn + driver_overhead
    full_total = total_weight_bytes + full_kv + calib_scratch + dn + driver_overhead

    print(f"\n{'='*72}")
    print(f"  VRAM Calculator — {os.path.basename(args.model_path)}")
    print(f"{'='*72}")
    print(f"  Architecture: {config.get('arch', 'unknown')} (arch_id={arch_id})")
    print(f"  Layers: {n_layers}  Heads: {n_heads}  KV Heads: {n_kv_heads}")
    print(f"  Head dim: {head_dim}  Hidden dim: {hidden_dim}  Model dim: {dim}")
    print(f"  Vocab: {vocab_size}")
    print(f"  GPU VRAM: {args.vram:.1f} GB")
    print()

    # Weight breakdown by category
    embed_bytes = sum(b for _, _, k, _, b, _ in weight_details if k == 'embed')
    lmhead_bytes = sum(b for _, _, k, _, b, _ in weight_details if k == 'lm_head')
    layer_bytes = sum(b for _, _, k, _, b, _ in weight_details if k == 'layer')

    def gb(b): return b / (1024**3)
    def mb(b): return b / (1024*1024)

    print(f"  ┌── Weight VRAM ──────────────────────────────────────────┐")
    print(f"  │ {'Category':<20} {'Tensors':>8} {'Disk':>10} {'GPU':>10} {'Ratio':>8} │")
    print(f"  │ {'─'*20} {'─'*8} {'─'*10} {'─'*10} {'─'*8} │")

    # Group by quant type
    by_qt = {}
    for t in weight_details:
        by_qt.setdefault(t[1], []).append(t)

    # Show largest tensors individually, rest as summary
    BIG_THRESHOLD_MB = 500  # show individual tensors > 500 MB on disk
    shown = []
    other_disk = 0
    other_gpu = 0
    other_count = 0
    for t in sorted(weight_details, key=lambda x: -x[3]):  # sort by disk size
        if t[3] >= BIG_THRESHOLD_MB:
            qt_label = {1: 'F16→F32', 13: 'MQ4', 14: 'MQ8', 3: 'Q8'}.get(t[1], f'qt={t[1]}')
            shown.append((t[0][:50], qt_label, t[3], t[4], t[5]))
        else:
            other_disk += t[3]
            other_gpu += t[4]
            other_count += 1

    for name, qt_label, disk_mb, gpu_mb, ratio in shown[:20]:
        print(f"  │ {name:<50} {disk_mb:>8.1f}M {gpu_mb:>8.1f}M {ratio:>5.2f}x │")
    if other_count > 0:
        print(f"  │ ... {other_count} small tensors             {other_disk:>8.1f}M {other_gpu:>8.1f}M {'─':>7} │")

    print(f"  │ {'─'*20} {'─'*8} {'─'*10} {'─'*10} {'─'*8} │")
    print(f"  │ {'TOTAL WEIGHT VRAM':<55} {gb(total_weight_bytes):>8.3f} GB │")
    print(f"  └{'─'*58}┘")
    print()

    # Two scenarios
    print(f"  ┌── SCENARIO A: Calibration (kv_seq={args.kv_seq}, {args.kv_mode}) ──────┐")
    print(f"  │ {'Component':<35} {'MB':>10} {'GB':>8} {'%VRAM':>8} │")
    print(f"  │ {'─'*35} {'─'*10} {'─'*8} {'─'*8} │")

    rows = [
        ("Model weights", mb(total_weight_bytes), gb(total_weight_bytes)),
        (f"KV cache ({args.kv_mode}, seq={args.kv_seq})", mb(calib_kv), gb(calib_kv)),
        ("Scratch buffers", mb(calib_scratch), gb(calib_scratch)),
        ("DeltaNet state", mb(dn), gb(dn)),
        ("Driver/ROCm overhead", mb(driver_overhead), gb(driver_overhead)),
    ]

    cumulative = 0
    for label, mb_val, gb_val in rows:
        cumulative += gb_val
        pct = (gb_val / args.vram) * 100
        cum_pct = (cumulative / args.vram) * 100
        print(f"  │ {label:<35} {mb_val:>10.1f} {gb_val:>8.3f} {pct:>7.1f}% │")

    print(f"  │ {'─'*35} {'─'*10} {'─'*8} {'─'*8} │")
    pct = (calib_total / vram_bytes) * 100
    fits = "✓ FITS" if calib_total <= vram_bytes else "✗ OVERFLOW"
    print(f"  │ {'TOTAL':<35} {mb(calib_total):>10.1f} {gb(calib_total):>8.3f} {pct:>7.1f}% │")
    print(f"  │ {'':<35} {'':>10} {'':>8} {fits:>8} │")
    print(f"  │ {'Headroom':<35} {'':>10} {args.vram - gb(calib_total):>8.3f} {'':>8} │")
    print(f"  └{'─'*58}┘")
    print()

    print(f"  ┌── SCENARIO B: Daemon load WITHOUT sidecar (asym3, seq={full_seq}) ─┐")
    print(f"  │ {'Component':<35} {'MB':>10} {'GB':>8} {'%VRAM':>8} │")
    print(f"  │ {'─'*35} {'─'*10} {'─'*8} {'─'*8} │")

    rows_b = [
        ("Model weights", mb(total_weight_bytes), gb(total_weight_bytes)),
        (f"KV cache (asym3, seq={full_seq})", mb(full_kv), gb(full_kv)),
        ("Scratch buffers", mb(calib_scratch), gb(calib_scratch)),
        ("DeltaNet state", mb(dn), gb(dn)),
        ("Driver/ROCm overhead", mb(driver_overhead), gb(driver_overhead)),
    ]

    cumulative = 0
    for label, mb_val, gb_val in rows_b:
        cumulative += gb_val
        pct = (gb_val / args.vram) * 100
        cum_pct = (cumulative / args.vram) * 100
        print(f"  │ {label:<35} {mb_val:>10.1f} {gb_val:>8.3f} {pct:>7.1f}% │")

    print(f"  │ {'─'*35} {'─'*10} {'─'*8} {'─'*8} │")
    total_b = total_weight_bytes + full_kv + calib_scratch + dn + driver_overhead
    pct_b = (total_b / vram_bytes) * 100
    fits_b = "✓ FITS" if total_b <= vram_bytes else "✗ OVERFLOW"
    print(f"  │ {'TOTAL':<35} {mb(total_b):>10.1f} {gb(total_b):>8.3f} {pct_b:>7.1f}% │")
    print(f"  │ {'':<35} {'':>10} {'':>8} {fits_b:>8} │")
    print(f"  │ {'Headroom':<35} {'':>10} {args.vram - gb(total_b):>8.3f} {'':>8} │")
    print(f"  └{'─'*58}┘")
    print()

    # ── Verdict ──
    print(f"  {'─'*65}")
    if calib_total <= vram_bytes:
        print(f"  ▶ VERDICT (Calibration): YES — fits with {args.vram - gb(calib_total):.2f} GB headroom.")
        print(f"    The triattn_validate / cask_gen calibration binary will work on")
        print(f"    this GPU with kv_seq={args.kv_seq} in {args.kv_mode} mode.")
    else:
        over = gb(calib_total) - args.vram
        print(f"  ▶ VERDICT (Calibration): NO — exceeds VRAM by {over:.2f} GB.")
        print(f"    Options:")
        if args.kv_seq > 128:
            print(f"    • Reduce kv_seq (try --kv-seq 128)")
        if args.kv_mode == 'q8':
            print(f"    • The model weights alone may exceed VRAM — check quantization settings")

    if total_b <= vram_bytes:
        print(f"  ▶ VERDICT (Daemon): Surprisingly, daemon also fits on {args.vram:.0f} GB.")
        print(f"    The OOM may be caused by daemon-specific allocations not captured here")
        print(f"    (e.g., weight pager overhead, multiple graph instances, or memory fragmentation).")
    else:
        over_b = gb(total_b) - args.vram
        print(f"  ▶ VERDICT (Daemon): OVERFLOW by {over_b:.2f} GB — the KV cache at seq={full_seq}")
        print(f"    in asym3 mode is the primary cause. A sidecar would reduce KV to")
        print(f"    physical_cap ≈ budget+beta+safety (typically 512-1024 tokens), recovering")
        print(f"    {gb(full_kv - calib_kv):.2f} GB.")

    print(f"  {'─'*65}")

    # ── Optional: try reducing KV seq ──
    if calib_total > vram_bytes:
        print(f"\n  Trying reduced KV seq to find breakpoint:")
        for try_seq in [256, 128, 64, 32]:
            try_kv = kv_cache_bytes(n_layers, n_kv_heads, head_dim, try_seq, args.kv_mode)
            try_total = total_weight_bytes + try_kv + calib_scratch + dn + driver_overhead
            if try_total <= vram_bytes:
                print(f"    kv_seq={try_seq}: {gb(try_total):.3f} GB ({args.vram - gb(try_total):.2f} GB headroom) ✓")
                break
            print(f"    kv_seq={try_seq}: {gb(try_total):.3f} GB (over by {gb(try_total) - args.vram:.2f} GB)")


if __name__ == '__main__':
    main()
