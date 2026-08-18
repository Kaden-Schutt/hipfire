#!/usr/bin/env python3
"""Repack qt=13 (MQ4G256, 136 B/group) tensors to qt=45 (MQ4C pad, 136 B/group).

Pure header rewrite on a canonical HFQM container — no re-quantization:

    read  136 B: [f32 scale][f32 zero][128 B nibbles]
    write 136 B: [fp16 scale][fp16 zero][4 zero pad][128 B nibbles unchanged]

Canonical HFQM (32-byte header):
    magic b"HFQM"; u32 version, arch, tensor_count;
    u64 metadata_offset, data_offset.
Metadata JSON ends at its matching top-level brace; the packed index follows:
    u32 count; entries u16 name_len/name/u8 qt/u8 ndim/ndim*u32 shape/
    u32 group_size/u64 data_len. Tensor payloads are contiguous from data_offset.

Usage: mq4c_repack.py IN.mq4 OUT.mq4c [--limit-tensors N]
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, List, Optional, Sequence, Tuple

import numpy as np

GROUP_BYTES = 136
NIBBLE_BYTES = 128
QT_V1 = 13
QT_V15 = 45
HFQM_MAGIC = b"HFQM"
HEADER_BYTES = 32
# Refuse at the nibble re-fit boundary: drift >= 0.5 can flip a code.
DRIFT_REFUSE = 0.5
DRIFT_WARN = 0.25
# Stream groups in bounded chunks (bytes ≈ CHUNK_GROUPS * 136).
CHUNK_GROUPS = 4096


class HfqmError(Exception):
    """Fatal HFQM parse / validation / I/O error."""


@dataclass
class TensorEntry:
    name: str
    qt: int
    qt_pos: int
    ndim: int
    shape: Tuple[int, ...]
    group_size: int
    data_len: int
    data_off: int


@dataclass
class HfqmIndex:
    version: int
    arch: int
    tensor_count: int
    metadata_offset: int
    data_offset: int
    entries: List[TensorEntry]
    # Absolute end of the packed index (may be < data_offset due to alignment pad).
    index_end: int


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise HfqmError(msg)


def _json_top_level_end(buf: bytes, start: int, limit: int) -> int:
    """Return absolute offset one past the matching top-level '}' of JSON at start."""
    _require(start < limit, "metadata_offset past readable prefix")
    brace = 0
    in_str = False
    esc = False
    i = start
    while i < limit:
        b = buf[i]
        if esc:
            esc = False
            i += 1
            continue
        if b == 0x5C and in_str:  # backslash
            esc = True
            i += 1
            continue
        if b == 0x22:  # quote
            in_str = not in_str
            i += 1
            continue
        if not in_str:
            if b == 0x7B:  # {
                brace += 1
            elif b == 0x7D:  # }
                brace -= 1
                if brace == 0:
                    return i + 1
                _require(brace > 0, "metadata JSON brace underflow")
        i += 1
    raise HfqmError("metadata JSON missing matching top-level brace")


def parse_hfqm_index(path: Path) -> HfqmIndex:
    """Parse and bounds-check the canonical 32-byte HFQM header + index."""
    size = path.stat().st_size
    _require(size >= HEADER_BYTES, f"file shorter than HFQM header ({size} < {HEADER_BYTES})")

    with path.open("rb") as f:
        head = f.read(HEADER_BYTES)
    _require(len(head) == HEADER_BYTES, "truncated HFQM header")
    magic = head[0:4]
    _require(magic == HFQM_MAGIC, f"bad magic {magic!r}, expected {HFQM_MAGIC!r}")

    version, arch, tensor_count = struct.unpack_from("<III", head, 4)
    metadata_offset, data_offset = struct.unpack_from("<QQ", head, 16)
    _require(metadata_offset >= HEADER_BYTES, f"metadata_offset {metadata_offset} inside header")
    _require(metadata_offset <= data_offset, "metadata_offset past data_offset")
    _require(data_offset <= size, f"data_offset {data_offset} past EOF {size}")

    # Prefix holds header + metadata + index + optional alignment padding.
    with path.open("rb") as f:
        prefix = f.read(data_offset)
    _require(len(prefix) == data_offset, "truncated HFQM prefix")

    json_end = _json_top_level_end(prefix, metadata_offset, data_offset)
    pos = json_end
    _require(pos + 4 <= data_offset, "index count past data_offset")
    (idx_n,) = struct.unpack_from("<I", prefix, pos)
    pos += 4
    _require(
        idx_n == tensor_count,
        f"index count {idx_n} != header tensor_count {tensor_count}",
    )

    entries: List[TensorEntry] = []
    data_cur = data_offset
    for i in range(idx_n):
        _require(pos + 2 <= data_offset, f"tensor {i}: name_len past data_offset")
        (name_len,) = struct.unpack_from("<H", prefix, pos)
        pos += 2
        _require(pos + name_len <= data_offset, f"tensor {i}: name past data_offset")
        name = prefix[pos : pos + name_len].decode("utf-8")
        pos += name_len
        _require(pos + 2 <= data_offset, f"tensor {i}: qt/ndim past data_offset")
        qt_pos = pos
        qt = prefix[pos]
        pos += 1
        ndim = prefix[pos]
        pos += 1
        need = 4 * ndim + 4 + 8
        _require(pos + need <= data_offset, f"tensor {i}: shape/gs/data_len past data_offset")
        shape = struct.unpack_from("<" + "I" * ndim, prefix, pos) if ndim else ()
        pos += 4 * ndim
        (group_size,) = struct.unpack_from("<I", prefix, pos)
        pos += 4
        (data_len,) = struct.unpack_from("<Q", prefix, pos)
        pos += 8
        _require(data_cur + data_len <= size, f"tensor {i} ({name!r}): data past EOF")
        entries.append(
            TensorEntry(
                name=name,
                qt=qt,
                qt_pos=qt_pos,
                ndim=ndim,
                shape=tuple(shape),
                group_size=group_size,
                data_len=int(data_len),
                data_off=data_cur,
            )
        )
        data_cur += int(data_len)
    _require(
        data_cur == size,
        f"indexed tensor data ends at {data_cur}, but file size is {size}; "
        "refusing to drop or reinterpret trailing bytes",
    )

    return HfqmIndex(
        version=version,
        arch=arch,
        tensor_count=tensor_count,
        metadata_offset=metadata_offset,
        data_offset=data_offset,
        entries=entries,
        index_end=pos,
    )


def select_qt13(entries: Sequence[TensorEntry], limit: Optional[int]) -> List[TensorEntry]:
    """All qt=13 tensors, or only the first N qt=13 tensors when limit is set."""
    qt13 = [e for e in entries if e.qt == QT_V1]
    if limit is None:
        return list(qt13)
    if limit < 0:
        raise HfqmError(f"--limit-tensors must be >= 0, got {limit}")
    return qt13[:limit]


def _validate_chunk(raw: np.ndarray, tensor_name: str) -> Tuple[float, float]:
    """Validate one (n, 136) uint8 chunk. Returns (worst_drift, min_scale)."""
    n = raw.shape[0]
    if n == 0:
        return 0.0, float("inf")
    s32 = raw[:, 0:4].copy().view(np.float32).ravel()
    z32 = raw[:, 4:8].copy().view(np.float32).ravel()
    nib = raw[:, 8:GROUP_BYTES]

    if not np.isfinite(s32).all():
        raise HfqmError(f"tensor {tensor_name!r}: non-finite f32 scale")
    if not np.isfinite(z32).all():
        raise HfqmError(f"tensor {tensor_name!r}: non-finite f32 zero")
    if not (s32 > 0).all():
        raise HfqmError(f"tensor {tensor_name!r}: non-positive f32 scale")

    s16 = s32.astype(np.float16)
    z16 = z32.astype(np.float16)
    s16f = s16.astype(np.float32)
    z16f = z16.astype(np.float32)

    if not np.isfinite(s16f).all():
        raise HfqmError(f"tensor {tensor_name!r}: non-finite f16 scale")
    if not np.isfinite(z16f).all():
        raise HfqmError(f"tensor {tensor_name!r}: non-finite f16 zero")
    if not (s16f > 0).all():
        raise HfqmError(f"tensor {tensor_name!r}: non-positive f16 scale")

    # Drift of the v1 reconstruction onto the fp16 affine grid, in code steps.
    q = np.empty((n, 256), dtype=np.float32)
    q[:, 0::2] = (nib & 0x0F).astype(np.float32)
    q[:, 1::2] = (nib >> 4).astype(np.float32)
    d = (z32 - z16f)[:, None] + q * (s32 - s16f)[:, None]
    ad = np.abs(d / s16f[:, None])
    worst = float(ad.max()) if ad.size else 0.0
    if worst >= DRIFT_REFUSE:
        raise HfqmError(
            f"tensor {tensor_name!r}: drift {worst:.4f} steps at/over {DRIFT_REFUSE} guard; "
            f"pure header rewrite is not equivalent to a re-fit"
        )
    return worst, float(s32.min())


def _transform_chunk(raw: np.ndarray) -> bytes:
    """v1 group rows -> pad layout rows (same 136 B stride)."""
    n = raw.shape[0]
    s32 = raw[:, 0:4].copy().view(np.float32).ravel()
    z32 = raw[:, 4:8].copy().view(np.float32).ravel()
    nib = raw[:, 8:GROUP_BYTES]
    s16 = s32.astype(np.float16)
    z16 = z32.astype(np.float16)
    hdr = np.empty((n, 4), dtype=np.uint8)
    hdr[:, 0:2] = np.frombuffer(np.ascontiguousarray(s16).tobytes(), dtype=np.uint8).reshape(n, 2)
    hdr[:, 2:4] = np.frombuffer(np.ascontiguousarray(z16).tobytes(), dtype=np.uint8).reshape(n, 2)
    pad = np.zeros((n, 4), dtype=np.uint8)
    interleaved = np.concatenate([hdr, pad, nib], axis=1)
    return interleaved.tobytes()


def _iter_group_chunks(fi: BinaryIO, data_off: int, data_len: int) -> Iterable[np.ndarray]:
    _require(data_len % GROUP_BYTES == 0, f"data_len {data_len} not divisible by {GROUP_BYTES}")
    n_groups = data_len // GROUP_BYTES
    fi.seek(data_off)
    remaining = n_groups
    while remaining:
        take = min(CHUNK_GROUPS, remaining)
        blob = fi.read(take * GROUP_BYTES)
        _require(len(blob) == take * GROUP_BYTES, "truncated tensor payload")
        yield np.frombuffer(blob, dtype=np.uint8).reshape(take, GROUP_BYTES).copy()
        remaining -= take


def validate_selected(src: Path, selected: Sequence[TensorEntry]) -> Tuple[float, float, int]:
    """Stream-validate every selected group before any destination write."""
    worst_drift = 0.0
    min_scale = float("inf")
    groups = 0
    with src.open("rb") as fi:
        for e in selected:
            _require(
                e.data_len % GROUP_BYTES == 0,
                f"tensor {e.name!r}: data_len {e.data_len} not divisible by {GROUP_BYTES}",
            )
            for chunk in _iter_group_chunks(fi, e.data_off, e.data_len):
                d, s = _validate_chunk(chunk, e.name)
                worst_drift = max(worst_drift, d)
                min_scale = min(min_scale, s)
                groups += chunk.shape[0]
                if DRIFT_WARN <= d < DRIFT_REFUSE:
                    print(
                        f"  note: {e.name!r} chunk drifts {d:.4f} steps "
                        f"(no flip, boundary {DRIFT_REFUSE})"
                    )
    return worst_drift, min_scale, groups


def _copy_bytes(fi: BinaryIO, fo: BinaryIO, nbytes: int, bufsize: int = 8 * 1024 * 1024) -> None:
    left = nbytes
    while left:
        chunk = fi.read(min(bufsize, left))
        if not chunk:
            raise HfqmError("unexpected EOF while copying")
        fo.write(chunk)
        left -= len(chunk)


def repack(src: str | Path, dst: str | Path, limit: Optional[int] = None) -> int:
    src_path = Path(src)
    dst_path = Path(dst)
    idx = parse_hfqm_index(src_path)
    selected = select_qt13(idx.entries, limit)
    selected_ids = {id(e) for e in selected}
    n_v1 = sum(1 for e in idx.entries if e.qt == QT_V1)
    print(f"{src_path}: {len(idx.entries)} tensors, {n_v1} at qt={QT_V1}, converting {len(selected)}")
    if not selected:
        print("nothing to repack")
        return 1

    # Full validation before creating/replacing the destination.
    worst_drift, min_scale, groups_done = validate_selected(src_path, selected)

    # Build patched prefix: header/metadata/index/alignment byte-identical except qt bytes.
    with src_path.open("rb") as fi:
        prefix = bytearray(fi.read(idx.data_offset))
    _require(len(prefix) == idx.data_offset, "truncated prefix on write pass")
    for e in selected:
        prefix[e.qt_pos] = QT_V15

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    # Temp lives in the destination directory so os.replace stays atomic on same FS.
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{dst_path.name}.",
        suffix=".tmp",
        dir=str(dst_path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fo, src_path.open("rb") as fi:
            fo.write(prefix)
            # Preserve any gap is already in prefix (alignment to data_offset).
            for e in idx.entries:
                if id(e) in selected_ids:
                    for chunk in _iter_group_chunks(fi, e.data_off, e.data_len):
                        fo.write(_transform_chunk(chunk))
                else:
                    fi.seek(e.data_off)
                    _copy_bytes(fi, fo, e.data_len)
            fo.flush()
            os.fsync(fo.fileno())
        os.replace(tmp_path, dst_path)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise

    a, b = src_path.stat().st_size, dst_path.stat().st_size
    print(f"groups repacked : {groups_done:,}")
    print(f"max drift       : {worst_drift:.6f} steps   (flip boundary {DRIFT_REFUSE})")
    print(f"min f32 scale   : {min_scale:.3e}")
    print(f"bytes           : {a:,} -> {b:,}")
    print("OK: pure header rewrite, every nibble preserved (pad layout).")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path, help="input HFQM (.mq4) path")
    parser.add_argument("dst", type=Path, help="output HFQM (.mq4c) path")
    parser.add_argument(
        "--limit-tensors",
        type=int,
        default=None,
        metavar="N",
        help="convert only the first N qt=13 tensors (default: all)",
    )
    args = parser.parse_args(argv)
    try:
        return repack(args.src, args.dst, args.limit_tensors)
    except HfqmError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
