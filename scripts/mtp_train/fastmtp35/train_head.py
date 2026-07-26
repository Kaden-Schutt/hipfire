#!/usr/bin/env python3
"""Train Qwen3.6 A3B's MTP head from exact Hipfire MQ4R hidden-state shards.

The trunk is intentionally absent from this process. Feature generation and
head optimization are separate phases so the 18 GB deployed trunk never has
to coexist with BF16 MTP parameters, gradients, and Adam state on a 32 GB
R9700. Launch with torchrun; one process owns one R9700.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.distributed as dist
import torch.nn.functional as F
import xxhash
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoConfig

from mtp_module import Qwen35MtpBlock, load_mtp_from_safetensors

MAGIC = b"HFMTPF01"
MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_RECORD_BYTES = 2 * 1024 * 1024 * 1024
RUNTIME_SHIFTED_ALIGNMENT = "runtime-shifted-v1"
LEGACY_SAME_POSITION_ALIGNMENT = "legacy-same-position-v0"
ALIGNMENTS = (RUNTIME_SHIFTED_ALIGNMENT, LEGACY_SAME_POSITION_ALIGNMENT)
TEACHER_FORCED_RECURRENCE = "teacher-forced-v1"
SELF_ROLLOUT_RECURRENCE = "self-rollout-v0"
RECURRENCE_INPUTS = (TEACHER_FORCED_RECURRENCE, SELF_ROLLOUT_RECURRENCE)


@dataclass
class FeatureRecord:
    source_ordinal: int
    absolute_start: int
    hidden_rows: int
    tokens: torch.Tensor
    hidden: torch.Tensor


class FeatureShard:
    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as handle:
            if handle.read(8) != MAGIC:
                raise ValueError(f"{path}: bad feature magic")
            (header_len,) = struct.unpack("<I", handle.read(4))
            if header_len > MAX_HEADER_BYTES:
                raise ValueError(f"{path}: oversized header")
            self.header = json.loads(handle.read(header_len))
        if self.header["schema_version"] != 1:
            raise ValueError(f"{path}: unsupported schema")
        if self.header["hidden_dtype"] != "bf16-le":
            raise ValueError(f"{path}: expected bf16-le hidden rows")
        if self.header["record_checksum"] != "xxh3-64":
            raise ValueError(f"{path}: expected xxh3-64 record checksums")

    def records(self) -> Iterator[FeatureRecord]:
        dim = int(self.header["hidden_dim"])
        k = int(self.header["recursive_steps"])
        with self.path.open("rb") as handle:
            handle.seek(8)
            (header_len,) = struct.unpack("<I", handle.read(4))
            handle.seek(header_len, os.SEEK_CUR)
            while True:
                prefix = handle.read(16)
                if not prefix:
                    return
                if len(prefix) != 16:
                    raise ValueError(f"{self.path}: truncated record prefix")
                payload_len, checksum = struct.unpack("<QQ", prefix)
                if payload_len > MAX_RECORD_BYTES:
                    raise ValueError(f"{self.path}: oversized record")
                payload = handle.read(payload_len)
                if len(payload) != payload_len:
                    raise ValueError(f"{self.path}: truncated record")
                if xxhash.xxh3_64_intdigest(payload) != checksum:
                    raise ValueError(f"{self.path}: checksum mismatch")
                id_len, source_ordinal, absolute_start, hidden_rows, token_count = (
                    struct.unpack_from("<IQIII", payload, 0)
                )
                offset = struct.calcsize("<IQIII") + id_len
                if token_count != hidden_rows + k:
                    raise ValueError(f"{self.path}: token/K contract mismatch")
                hidden_count = hidden_rows * dim
                expected = offset + token_count * 4 + hidden_count * 2
                if expected != payload_len:
                    raise ValueError(f"{self.path}: payload shape mismatch")
                tokens = torch.frombuffer(
                    bytearray(payload[offset : offset + token_count * 4]),
                    dtype=torch.int32,
                    count=token_count,
                ).clone().to(torch.long)
                offset += token_count * 4
                hidden = torch.frombuffer(
                    bytearray(payload[offset:]),
                    dtype=torch.bfloat16,
                    count=hidden_count,
                ).clone().reshape(hidden_rows, dim)
                yield FeatureRecord(
                    source_ordinal=source_ordinal,
                    absolute_start=absolute_start,
                    hidden_rows=hidden_rows,
                    tokens=tokens,
                    hidden=hidden,
                )


def snapshot_dir(path: Path) -> Path:
    snapshots = sorted((path / "snapshots").glob("*"))
    return snapshots[0] if snapshots else path


def find_tensor(model_dir: Path, names: list[str]) -> torch.Tensor:
    root = snapshot_dir(model_dir)
    files = sorted(root.glob("*.safetensors"))
    for name in names:
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as handle:
                if name in handle.keys():
                    return handle.get_tensor(name)
    raise KeyError(f"none of {names} found under {root}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def causal_mask(lengths: torch.Tensor, width: int, dtype: torch.dtype) -> torch.Tensor:
    # HF eager attention accepts [B, 1, Q, K] additive masks. Mask both causal
    # future keys and padded keys; padded query rows are excluded from loss.
    device = lengths.device
    q = torch.arange(width, device=device)[:, None]
    key = torch.arange(width, device=device)[None, :]
    allowed = key <= q
    allowed = allowed[None, None, :, :] & (
        key[None, None, :, :] < lengths[:, None, None, None]
    )
    mask = torch.zeros((lengths.shape[0], 1, width, width), dtype=dtype, device=device)
    return mask.masked_fill(~allowed, torch.finfo(dtype).min)


def collate(
    records: list[FeatureRecord],
    dim: int,
    k: int,
    alignment: str = RUNTIME_SHIFTED_ALIGNMENT,
) -> dict[str, torch.Tensor]:
    if alignment not in ALIGNMENTS:
        raise ValueError(f"unsupported MTP alignment {alignment!r}")
    # Transformers' MTP contract shifts input tokens one position ahead of
    # the trunk hidden sequence. For trunk h[t], the head consumes token
    # x[t+1] at position t+1 and predicts x[t+2]. HFMTPF01 predates that
    # clarification and stores N hidden rows plus N+K tokens, so the corrected
    # view uses the first N-1 hidden rows and all N+K tokens. No expensive
    # feature regeneration is needed; only the final hidden row is unused.
    shifted = alignment == RUNTIME_SHIFTED_ALIGNMENT
    effective_lengths = [
        record.hidden_rows - 1 if shifted else record.hidden_rows for record in records
    ]
    if any(length <= 0 for length in effective_lengths):
        raise ValueError("MTP feature record is too short for the requested alignment")
    width = max(effective_lengths)
    batch = len(records)
    # Preserve K trailing trunk rows when present so distribution training can
    # score the draft against the exact trunk LM distribution for each
    # recursive depth. HFMTPF01 stores only one guaranteed trailing hidden row;
    # deeper tail positions remain zero-padded and are excluded by the
    # per-depth teacher mask below.
    hidden = torch.zeros((batch, width + k, dim), dtype=torch.bfloat16)
    token_tail = k + 1 if shifted else k
    tokens = torch.zeros((batch, width + token_tail), dtype=torch.long)
    lengths = torch.tensor(effective_lengths, dtype=torch.long)
    positions = torch.zeros((batch, width), dtype=torch.long)
    for row, (record, n) in enumerate(zip(records, effective_lengths, strict=True)):
        hidden[row, : record.hidden_rows].copy_(record.hidden)
        tokens[row, : n + token_tail].copy_(record.tokens[: n + token_tail])
        position_start = record.absolute_start + (1 if shifted else 0)
        positions[row, :n] = torch.arange(
            position_start, position_start + n, dtype=torch.long
        )
    return {"hidden": hidden, "tokens": tokens, "lengths": lengths, "positions": positions}


def owned_shards(shard_paths: list[Path], rank: int, world: int) -> list[Path]:
    marker = f"-p{rank:03}-of{world:03}-"
    partitioned = [path for path in shard_paths if marker in path.name]
    return partitioned if partitioned else list(shard_paths[rank::world])


def record_stream(
    shard_paths: list[Path], rank: int, world: int, seed: int, epoch: int
) -> Iterator[FeatureRecord]:
    owned = owned_shards(shard_paths, rank, world)
    random.Random(seed + epoch).shuffle(owned)
    for path in owned:
        yield from FeatureShard(path).records()


def batches(
    shard_paths: list[Path],
    rank: int,
    world: int,
    seed: int,
    epoch: int,
    micro_batch: int,
    max_batches: int | None = None,
) -> Iterator[list[FeatureRecord]]:
    pending: list[FeatureRecord] = []
    emitted = 0
    for record in record_stream(shard_paths, rank, world, seed, epoch):
        pending.append(record)
        if len(pending) == micro_batch:
            yield pending
            pending = []
            emitted += 1
            if max_batches is not None and emitted >= max_batches:
                return


def load_vocab_map(path: Path, full_vocab: int) -> tuple[torch.Tensor, torch.Tensor]:
    body = json.loads(path.read_text())
    values = torch.tensor(body["draft_to_full"], dtype=torch.long)
    if len(values) != int(body["compressed_vocab_size"]):
        raise ValueError("vocab map length does not match compressed_vocab_size")
    if values.min().item() < 0 or values.max().item() >= full_vocab:
        raise ValueError("vocab map contains a token outside full vocabulary")
    if values.unique().numel() != values.numel():
        raise ValueError("vocab map contains duplicate full-vocab ids")
    inverse = torch.full((full_vocab,), -100, dtype=torch.long)
    inverse[values] = torch.arange(values.numel(), dtype=torch.long)
    return values, inverse


def unwrap(model: torch.nn.Module) -> Qwen35MtpBlock:
    return model.module if isinstance(model, DistributedDataParallel) else model


def save_checkpoint(
    output: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    epoch: int,
    rank: int,
    final: bool = False,
) -> None:
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        name = "final" if final else f"step-{step:07}"
        tensor_path = output / f"{name}.safetensors"
        tensor_tmp = tensor_path.with_suffix(".safetensors.partial")
        state = {
            f"mtp.{key}": value.detach().cpu().contiguous()
            for key, value in unwrap(model).state_dict().items()
        }
        save_file(state, str(tensor_tmp))
        tensor_tmp.replace(tensor_path)
        if not final:
            state_path = output / f"{name}.optimizer.pt"
            state_tmp = state_path.with_suffix(".pt.partial")
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "epoch": epoch,
                },
                state_tmp,
            )
            state_tmp.replace(state_path)
    if dist.is_initialized():
        dist.barrier()


def load_checkpoint_weights(model: Qwen35MtpBlock, path: Path) -> None:
    with safe_open(path, framework="pt", device="cpu") as handle:
        state = {key: handle.get_tensor(key) for key in handle.keys()}
    missing, unexpected = model.load_pretrained_(state)
    if missing or unexpected:
        raise ValueError(
            f"resume checkpoint mismatch: missing={missing[:8]} unexpected={unexpected[:8]}"
        )


def train_microbatch(
    model: torch.nn.Module,
    records: list[FeatureRecord],
    embed_weight: torch.Tensor,
    lm_weight: torch.Tensor,
    vocab_map: torch.Tensor,
    inverse_vocab: torch.Tensor,
    loss_weights: torch.Tensor,
    device: torch.device,
    dim: int,
    k: int,
    alignment: str,
    recurrence_input: str,
    soft_target_weight: float,
    soft_target_topk: int,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[float]]:
    batch = collate(records, dim, k, alignment)
    trunk_hidden = batch["hidden"].to(device, non_blocking=True)
    tokens = batch["tokens"].to(device, non_blocking=True)
    lengths = batch["lengths"].to(device, non_blocking=True)
    positions = batch["positions"].to(device, non_blocking=True)
    width = positions.shape[1]
    mask = causal_mask(lengths, width, trunk_hidden.dtype)
    current_hidden = trunk_hidden[:, :width]
    token_offset = 1 if alignment == RUNTIME_SHIFTED_ALIGNMENT else 0
    target_offset = token_offset + 1
    current_tokens = tokens[:, token_offset : token_offset + width]
    valid_rows = torch.arange(width, device=device)[None, :] < lengths[:, None]
    losses = []
    soft_losses = []
    coverage = []
    for depth in range(k):
        if recurrence_input == TEACHER_FORCED_RECURRENCE:
            current_tokens = tokens[
                :, token_offset + depth : token_offset + depth + width
            ]
        elif recurrence_input != SELF_ROLLOUT_RECURRENCE:
            raise ValueError(f"unsupported recurrence input {recurrence_input!r}")
        current_emb = F.embedding(current_tokens, embed_weight)
        current_hidden = model(
            current_emb,
            current_hidden,
            positions + depth,
            attention_mask=mask,
        )
        logits = F.linear(current_hidden, lm_weight)
        target_full = tokens[:, target_offset + depth : target_offset + depth + width]
        target = inverse_vocab[target_full]
        keep = valid_rows & (target >= 0)
        if not keep.any():
            raise RuntimeError("microbatch has no targets inside compressed vocabulary")
        # ROCm cross_entropy applies its FP32 opmath internally for BF16 input.
        # Avoid materializing a second FP32 logits tensor: at larger batches it
        # costs multiple GiB per rank without a meaningful gradient benefit.
        hard_loss = F.cross_entropy(logits[keep], target[keep])
        if soft_target_weight:
            # For MTP depth d at trunk h[t], the exact teacher distribution is
            # LM(h[t+d+1]). Feature records contain sequential post-final-norm
            # trunk rows, so this target requires neither a second trunk
            # forward nor regenerated Stage 2 features. Near the record tail,
            # only rows with a retained future hidden state participate.
            soft_keep = (
                torch.arange(width, device=device)[None, :]
                < (lengths - depth).clamp_min(0)[:, None]
            )
            with torch.no_grad():
                teacher_logits = F.linear(
                    trunk_hidden[:, depth + 1 : depth + 1 + width],
                    lm_weight,
                )
                topk = min(soft_target_topk, teacher_logits.shape[-1])
                teacher_values, teacher_indices = teacher_logits[soft_keep].topk(
                    topk, dim=-1
                )
                teacher_log_norm = torch.logsumexp(
                    teacher_logits[soft_keep].float(), dim=-1, keepdim=True
                )
                teacher_probs = (teacher_values.float() - teacher_log_norm).exp()
                teacher_tail = (1.0 - teacher_probs.sum(-1)).clamp_min(0.0)
                del teacher_logits, teacher_values, teacher_log_norm
            draft_logits = logits[soft_keep]
            draft_log_norm = torch.logsumexp(draft_logits.float(), dim=-1, keepdim=True)
            draft_top_logprob = (
                draft_logits.gather(1, teacher_indices).float() - draft_log_norm
            )
            draft_tail_logprob = torch.log1p(
                -draft_top_logprob.exp().sum(-1).clamp(max=1.0 - 1e-7)
            )
            soft_loss = -(
                (teacher_probs * draft_top_logprob).sum(-1)
                + teacher_tail * draft_tail_logprob
            ).mean()
            del (
                draft_logits,
                teacher_indices,
                teacher_probs,
                teacher_tail,
                draft_log_norm,
                draft_top_logprob,
                draft_tail_logprob,
            )
        else:
            soft_loss = hard_loss.new_zeros(())
        losses.append(
            hard_loss * (1.0 - soft_target_weight) + soft_loss * soft_target_weight
        )
        soft_losses.append(soft_loss)
        coverage.append(float(keep.sum()) / float(valid_rows.sum()))
        if recurrence_input == SELF_ROLLOUT_RECURRENCE:
            with torch.no_grad():
                current_tokens = vocab_map[logits.argmax(-1)]
    loss = sum(loss_weights[index] * losses[index] for index in range(k))
    return loss, losses, soft_losses, coverage


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    embed_weight: torch.Tensor,
    lm_weight: torch.Tensor,
    vocab_map: torch.Tensor,
    inverse_vocab: torch.Tensor,
    shard_paths: list[Path],
    rank: int,
    world: int,
    seed: int,
    micro_batch: int,
    dim: int,
    k: int,
    max_batches: int,
    alignment: str,
    recurrence_input: str,
) -> list[dict[str, float]]:
    model.eval()
    totals = [
        {"loss": 0.0, "correct": 0.0, "covered": 0.0, "tokens": 0.0} for _ in range(k)
    ]
    for batch_index, records in enumerate(
        batches(shard_paths, rank, world, seed, 0, micro_batch, max_batches)
    ):
        batch = collate(records, dim, k, alignment)
        trunk_hidden = batch["hidden"].cuda(non_blocking=True)
        tokens = batch["tokens"].cuda(non_blocking=True)
        lengths = batch["lengths"].cuda(non_blocking=True)
        positions = batch["positions"].cuda(non_blocking=True)
        width = positions.shape[1]
        mask = causal_mask(lengths, width, trunk_hidden.dtype)
        current_hidden = trunk_hidden[:, :width]
        token_offset = 1 if alignment == RUNTIME_SHIFTED_ALIGNMENT else 0
        target_offset = token_offset + 1
        current_tokens = tokens[:, token_offset : token_offset + width]
        valid_rows = (
            torch.arange(width, device=trunk_hidden.device)[None, :] < lengths[:, None]
        )
        for depth in range(k):
            if recurrence_input == TEACHER_FORCED_RECURRENCE:
                current_tokens = tokens[
                    :, token_offset + depth : token_offset + depth + width
                ]
            elif recurrence_input != SELF_ROLLOUT_RECURRENCE:
                raise ValueError(f"unsupported recurrence input {recurrence_input!r}")
            current_emb = F.embedding(current_tokens, embed_weight)
            current_hidden = model(
                current_emb,
                current_hidden,
                positions + depth,
                attention_mask=mask,
            )
            logits = F.linear(current_hidden, lm_weight)
            target_full = tokens[:, target_offset + depth : target_offset + depth + width]
            target = inverse_vocab[target_full]
            keep = valid_rows & (target >= 0)
            prediction = logits.argmax(-1)
            if keep.any():
                loss = F.cross_entropy(logits[keep].float(), target[keep], reduction="sum")
                totals[depth]["loss"] += float(loss)
                totals[depth]["correct"] += float((prediction[keep] == target[keep]).sum())
                totals[depth]["covered"] += float(keep.sum())
            totals[depth]["tokens"] += float(valid_rows.sum())
            if recurrence_input == SELF_ROLLOUT_RECURRENCE:
                current_tokens = vocab_map[prediction]
    packed = torch.tensor(
        [
            value
            for depth in totals
            for value in (
                depth["loss"],
                depth["correct"],
                depth["covered"],
                depth["tokens"],
            )
        ],
        dtype=torch.float64,
        device="cuda",
    )
    if world > 1:
        dist.all_reduce(packed)
    result = []
    for depth in range(k):
        loss, correct, covered, tokens = packed[depth * 4 : depth * 4 + 4].tolist()
        result.append(
            {
                "loss": loss / max(covered, 1.0),
                "accuracy": correct / max(covered, 1.0),
                "coverage": covered / max(tokens, 1.0),
                "tokens": tokens,
            }
        )
    model.train()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--validation-features", type=Path, required=True)
    parser.add_argument("--hf-model", type=Path, required=True)
    parser.add_argument("--vocab-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument(
        "--soft-target-weight",
        type=float,
        default=0.0,
        help="blend weight for exact-trunk distribution cross entropy",
    )
    parser.add_argument(
        "--soft-target-topk",
        type=int,
        default=256,
        help="teacher support retained per row for distribution cross entropy",
    )
    parser.add_argument("--resume-optimizer", type=Path)
    parser.add_argument("--resume-weights", type=Path)
    parser.add_argument(
        "--alignment",
        choices=ALIGNMENTS,
        default=RUNTIME_SHIFTED_ALIGNMENT,
        help="token/hidden contract; runtime-shifted-v1 matches Transformers and Hipfire serving",
    )
    parser.add_argument(
        "--recurrence-input",
        choices=RECURRENCE_INPUTS,
        default=TEACHER_FORCED_RECURRENCE,
        help="recursive-token source; teacher-forced-v1 matches FastMTP training",
    )
    args = parser.parse_args()
    if not 0.0 <= args.soft_target_weight <= 1.0:
        raise SystemExit("--soft-target-weight must be between zero and one")
    if args.soft_target_topk <= 0:
        raise SystemExit("--soft-target-topk must be positive")
    if (
        args.soft_target_weight
        and args.alignment != RUNTIME_SHIFTED_ALIGNMENT
    ):
        raise SystemExit(
            "soft targets require runtime-shifted-v1 feature alignment"
        )

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    train_shards = sorted(args.features.glob("*.rwf"))
    validation_shards = sorted(args.validation_features.glob("*.rwf"))
    if not train_shards or not validation_shards:
        raise SystemExit("both training and validation feature directories need .rwf shards")
    header = FeatureShard(train_shards[0]).header
    dim = int(header["hidden_dim"])
    k = int(header["recursive_steps"])
    for path in train_shards + validation_shards:
        candidate = FeatureShard(path).header
        for field in (
            "schema_version",
            "architecture",
            "trunk_sha256",
            "hidden_dim",
            "recursive_steps",
            "hidden_dtype",
        ):
            if candidate[field] != header[field]:
                raise ValueError(f"{path}: header field {field} differs from training contract")

    config = AutoConfig.from_pretrained(snapshot_dir(args.hf_model), trust_remote_code=True)
    text_config = config.text_config if hasattr(config, "text_config") else config
    if int(text_config.hidden_size) != dim:
        raise ValueError(f"feature dim {dim} != model hidden size {text_config.hidden_size}")
    full_vocab = int(text_config.vocab_size)
    vocab_map_cpu, inverse_cpu = load_vocab_map(args.vocab_map, full_vocab)

    if rank == 0:
        print(
            json.dumps(
                {
                    "event": "load",
                    "world": world,
                    "train_shards": len(train_shards),
                    "validation_shards": len(validation_shards),
                    "hidden_dim": dim,
                    "recursive_steps": k,
                    "compressed_vocab": len(vocab_map_cpu),
                }
            ),
            flush=True,
        )
    mtp = Qwen35MtpBlock(text_config)
    missing, unexpected = mtp.load_pretrained_(
        load_mtp_from_safetensors(str(args.hf_model))
    )
    if missing or unexpected:
        raise ValueError(
            f"official MTP warm start mismatch: missing={missing[:8]} "
            f"unexpected={unexpected[:8]}"
        )
    if args.resume_weights:
        load_checkpoint_weights(mtp, args.resume_weights)
    mtp = mtp.to(device=device, dtype=torch.bfloat16)

    embed_cpu = find_tensor(
        args.hf_model,
        [
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
            "embed_tokens.weight",
        ],
    ).to(torch.bfloat16)
    lm_cpu = find_tensor(
        args.hf_model,
        [
            "lm_head.weight",
            "model.lm_head.weight",
            "model.language_model.lm_head.weight",
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
            "embed_tokens.weight",
        ],
    ).to(torch.bfloat16)
    embed_weight = embed_cpu.to(device)
    lm_weight = lm_cpu[vocab_map_cpu].contiguous().to(device)
    del embed_cpu, lm_cpu
    vocab_map = vocab_map_cpu.to(device)
    inverse_vocab = inverse_cpu.to(device)

    if world > 1:
        mtp = DistributedDataParallel(
            mtp,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
    optimizer = torch.optim.AdamW(
        mtp.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=True,
    )
    accumulation = args.global_batch_size // (world * args.micro_batch_size)
    if accumulation == 0 or accumulation * world * args.micro_batch_size != args.global_batch_size:
        raise ValueError("global batch must divide exactly by world * micro batch")

    # State files contain exact record counts. Cap every rank to the smallest
    # partition so DDP executes the same number of collectives even if final
    # trajectory lengths caused minor producer skew.
    state_files = sorted(args.features.glob("*.state.json"))
    marker = f"-p{rank:03}-of{world:03}.state.json"
    owned_states = [path for path in state_files if path.name.endswith(marker)]
    if owned_states:
        local_records = sum(
            json.loads(path.read_text())["feature_records"] for path in owned_states
        )
    else:
        local_records = sum(
            json.loads(path.read_text())["feature_records"] for path in state_files[rank::world]
        )
    local_records_tensor = torch.tensor(local_records, dtype=torch.long, device=device)
    if world > 1:
        dist.all_reduce(local_records_tensor, op=dist.ReduceOp.MIN)
    records_per_rank = int(local_records_tensor)
    if records_per_rank == 0:
        raise ValueError("feature state manifests report zero records")
    local_micro_batches = records_per_rank // args.micro_batch_size
    steps_per_epoch = local_micro_batches // accumulation
    planned_steps = steps_per_epoch * args.epochs
    stop_step = min(planned_steps, args.max_steps) if args.max_steps else planned_steps
    warmup_steps = max(1, int(planned_steps * args.warmup_ratio))

    def lr_scale(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, planned_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    start_step = 0
    start_epoch = 0
    if args.resume_optimizer:
        if not args.resume_weights:
            raise ValueError("--resume-optimizer requires --resume-weights")
        resume = torch.load(args.resume_optimizer, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        start_step = int(resume["step"])
        start_epoch = start_step // steps_per_epoch

    weights = torch.tensor(
        [0.6**depth for depth in range(k)], dtype=torch.float32, device=device
    )
    weights /= weights.sum()
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    micro_step = 0
    started = time.monotonic()
    mtp.train()

    for epoch in range(start_epoch, args.epochs):
        completed_steps_this_epoch = (
            start_step % steps_per_epoch if epoch == start_epoch else 0
        )
        skip_micro_batches = completed_steps_this_epoch * accumulation
        for records in batches(
            train_shards,
            rank,
            world,
            args.seed,
            epoch,
            args.micro_batch_size,
            steps_per_epoch * accumulation,
        ):
            if skip_micro_batches:
                skip_micro_batches -= 1
                continue
            sync_now = (micro_step + 1) % accumulation == 0
            sync_context = (
                contextlib.nullcontext()
                if sync_now or not isinstance(mtp, DistributedDataParallel)
                else mtp.no_sync()
            )
            with sync_context:
                loss, losses, soft_losses, coverage = train_microbatch(
                    mtp,
                    records,
                    embed_weight,
                    lm_weight,
                    vocab_map,
                    inverse_vocab,
                    weights,
                    device,
                    dim,
                    k,
                    args.alignment,
                    args.recurrence_input,
                    args.soft_target_weight,
                    args.soft_target_topk,
                )
                (loss / accumulation).backward()
            micro_step += 1
            if not sync_now:
                continue
            torch.nn.utils.clip_grad_norm_(mtp.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            if rank == 0 and (step == 1 or step % 10 == 0):
                print(
                    json.dumps(
                        {
                            "event": "train",
                            "epoch": epoch,
                            "step": step,
                            "planned_steps": planned_steps,
                            "stop_step": stop_step,
                            "loss": float(loss.detach()),
                            "step_losses": [
                                float(value.detach()) for value in losses
                            ],
                            "soft_target_losses": [
                                float(value.detach()) for value in soft_losses
                            ],
                            "coverage": coverage,
                            "lr": scheduler.get_last_lr()[0],
                            "elapsed_s": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
            if args.eval_every and step % args.eval_every == 0:
                metrics = evaluate(
                    mtp,
                    embed_weight,
                    lm_weight,
                    vocab_map,
                    inverse_vocab,
                    validation_shards,
                    rank,
                    world,
                    args.seed,
                    args.micro_batch_size,
                    dim,
                    k,
                    args.eval_batches,
                    args.alignment,
                    args.recurrence_input,
                )
                if rank == 0:
                    print(json.dumps({"event": "validation", "step": step, "metrics": metrics}), flush=True)
            if args.checkpoint_every and step % args.checkpoint_every == 0:
                save_checkpoint(args.output, mtp, optimizer, scheduler, step, epoch, rank)
            if step >= stop_step:
                break
        if step >= stop_step:
            break

    metrics = evaluate(
        mtp,
        embed_weight,
        lm_weight,
        vocab_map,
        inverse_vocab,
        validation_shards,
        rank,
        world,
        args.seed,
        args.micro_batch_size,
        dim,
        k,
        args.eval_batches,
        args.alignment,
        args.recurrence_input,
    )
    save_checkpoint(args.output, mtp, optimizer, scheduler, step, args.epochs, rank, final=True)
    if rank == 0:
        manifest = {
            "schema_version": 1,
            "feature_header": header,
            "vocab_map": str(args.vocab_map),
            "vocab_map_sha256": sha256(args.vocab_map),
            "hf_model": str(args.hf_model),
            "world_size": world,
            "global_batch_size": args.global_batch_size,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation": accumulation,
            "epochs": args.epochs,
            "steps": step,
            "planned_steps": planned_steps,
            "stop_step": stop_step,
            "loss_weights": weights.cpu().tolist(),
            "alignment": args.alignment,
            "recurrence_input": args.recurrence_input,
            "soft_target_weight": args.soft_target_weight,
            "soft_target_topk": args.soft_target_topk,
            "validation": metrics,
            "output": "final.safetensors",
        }
        (args.output / "training-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        print(json.dumps({"event": "complete", **manifest}), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
