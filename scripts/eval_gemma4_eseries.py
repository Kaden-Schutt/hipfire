#!/usr/bin/env python3
"""Run reproducible Gemma4 E-series LongBench or long-decode validation."""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
import queue
import re
import signal
import statistics
import subprocess
import threading
import time
from pathlib import Path


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_hash(text: str, algorithm: str = "sha256") -> str:
    return hashlib.new(algorithm, text.encode()).hexdigest()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def metric_stats(rows: list[dict], key: str) -> dict:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return {
        "n": len(values),
        "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p90": percentile(values, 0.9),
        "max": max(values) if values else None,
    }


def extract_choice(text: str) -> str | None:
    patterns = (
        r"the\s+correct\s+answer\s+is\s*\(?\s*([ABCD])(?![A-Za-z])\s*\)?",
        r"(?:答案|选项)\s*(?:是|为|：|:)\s*\(?\s*([ABCD])(?![A-Za-z])\s*\)?",
        r"\(([ABCD])\)",
        r"\b([ABCD])\b",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    return None


GSM8K_FEWSHOT = (
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the "
        "grove today. After they are done, there will be 21 trees. How many trees "
        "did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more "
        "were planted. So there must have been 21 - 15 = 6. The answer is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many "
        "cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many "
        "pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they "
        "had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 "
        "lollipops. How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "
        "So he gave Denny 20 - 12 = 8. The answer is 8.",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and "
        "dad. How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then "
        "that is 4 more toys. 5 + 4 = 9. The answer is 9.",
    ),
    (
        "There were nine computers in the server room. Five more computers were "
        "installed each day, from monday to thursday. How many computers are now in "
        "the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more computers "
        "were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer "
        "is 29.",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, "
        "he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he had "
        "58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The "
        "answer is 33.",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does she "
        "have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 "
        "dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    ),
)


def gsm8k_prompt(question: str) -> str:
    turns = [f"Q: {question}\n\nA: {answer}" for question, answer in GSM8K_FEWSHOT]
    turns.append(f"Q: {question}\n\nA:")
    return "\n\n".join(turns)


def normalize_number(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return None
    if number == number.to_integral_value():
        return str(number.quantize(Decimal(1)))
    return format(number.normalize(), "f")


def extract_gsm8k_answer(text: str) -> tuple[str | None, str | None]:
    patterns = (
        ("hash", r"####\s*(-?\$?[0-9][0-9,]*(?:\.[0-9]+)?)"),
        ("answer", r"the\s+answer\s+is\s*:?[ \t]*(-?\$?[0-9][0-9,]*(?:\.[0-9]+)?)"),
        ("boxed", r"\\boxed\{\s*(-?\$?[0-9][0-9,]*(?:\.[0-9]+)?)\s*\}"),
        ("flexible", r"(-?\$?[0-9][0-9,]*(?:\.[0-9]+)?)"),
    )
    for source, pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return normalize_number(matches[-1]), source
    return None, None


class Daemon:
    def __init__(
        self,
        binary: Path,
        stderr_path: Path,
        physical_gpu: str,
        prefill_batch: int,
        q8_fused_prefill: bool,
        ple_batched_prefill: bool,
        ple_branch_batched_prefill: bool,
        runtime_home: Path | None,
    ):
        env = os.environ.copy()
        if runtime_home is not None:
            runtime_home.mkdir(parents=True, exist_ok=True)
            (runtime_home / ".hipfire").mkdir(exist_ok=True)
            env["HOME"] = str(runtime_home.resolve())
        env["HIP_VISIBLE_DEVICES"] = physical_gpu
        env["HIPFIRE_GEMMA4_GRAPH"] = "0"
        env["HIPFIRE_GEMMA4_EAGLE"] = "0"
        env["HIPFIRE_GEMMA4_PREFILL_BATCH"] = str(prefill_batch)
        env["HIPFIRE_GEMMA4_Q8_FUSED_PREFILL"] = "1" if q8_fused_prefill else "0"
        env["HIPFIRE_GEMMA4_PLE_BATCHED_PREFILL"] = "1" if ple_batched_prefill else "0"
        env["HIPFIRE_GEMMA4_PLE_BRANCH_BATCHED_PREFILL"] = (
            "1" if ple_branch_batched_prefill else "0"
        )
        self.stderr_stream = stderr_path.open("w", buffering=1)
        self.proc = subprocess.Popen(
            [str(binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_stream,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=True,
        )
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.events: queue.Queue[dict] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for raw in self.proc.stdout:
            raw = raw.strip()
            if not raw:
                continue
            try:
                self.events.put(json.loads(raw))
            except json.JSONDecodeError:
                self.events.put({"type": "protocol_error", "line": raw})

    def send(self, payload: dict) -> None:
        if self.proc.poll() is not None:
            raise RuntimeError(f"daemon exited with {self.proc.returncode}")
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()

    def request(self, payload: dict, terminal: set[str], timeout: float) -> tuple[dict, list[dict]]:
        self.send(payload)
        deadline = time.monotonic() + timeout
        events: list[dict] = []
        while time.monotonic() < deadline:
            try:
                event = self.events.get(timeout=min(1.0, deadline - time.monotonic()))
            except queue.Empty:
                if self.proc.poll() is not None:
                    break
                continue
            events.append(event)
            if event.get("type") == "commit_ready":
                self.send({
                    "type": "commit",
                    "id": event.get("id"),
                    "attempt_id": event.get("attempt_id"),
                })
            if event.get("type") in terminal:
                return event, events
        raise TimeoutError(f"waiting for {sorted(terminal)}; rc={self.proc.poll()} tail={events[-5:]}")

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                self.request({"type": "unload"}, {"unloaded", "error"}, 120)
        except Exception:
            pass
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=10)
        self.stderr_stream.close()


def collect_text(events: list[dict]) -> tuple[str, str]:
    visible = "".join(
        str(event.get("text", "")) for event in events if event.get("type") == "token"
    )
    if not visible:
        visible = "".join(
            str(event.get("text", ""))
            for event in events
            if event.get("type") == "committed" and isinstance(event.get("text"), str)
        )
    reasoning = "".join(
        str(event.get("text", "")) for event in events if event.get("type") == "reasoning"
    )
    return visible, reasoning


def load_longbench(dataset: Path, manifest: Path) -> tuple[list[dict], dict]:
    manifest_data = json.loads(manifest.read_text())
    expected = manifest_data.get("sample_sha256")
    actual = file_hash(dataset)
    if expected and expected != actual:
        raise ValueError(f"LongBench SHA mismatch: {actual} != {expected}")
    rows = [json.loads(line) for line in dataset.read_text().split("\n") if line.strip()]
    tasks = []
    for row in rows:
        prompt = row["prompt_no_think"]
        expected_md5 = row.get("prompt_no_think_md5")
        if expected_md5 and text_hash(prompt, "md5") != expected_md5:
            raise ValueError(f"prompt MD5 mismatch at ordinal {row['ordinal']}")
        tasks.append({
            "id": f"longbench-{int(row['ordinal']):02d}",
            "ordinal": int(row["ordinal"]),
            "category": row.get("domain"),
            "prompt": prompt,
            "gold": str(row.get("answer", "")).upper(),
        })
    return tasks, manifest_data


def load_gsm8k(dataset: Path) -> tuple[list[dict], dict]:
    rows = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    tasks = []
    for ordinal, row in enumerate(rows):
        gold = normalize_number(str(row["answer"]).rsplit("####", 1)[-1])
        if gold is None:
            raise ValueError(f"invalid GSM8K gold answer at ordinal {ordinal}")
        tasks.append({
            "id": f"gsm8k-{ordinal:04d}",
            "ordinal": ordinal,
            "category": "gsm8k",
            "prompt": gsm8k_prompt(str(row["question"])),
            "question": str(row["question"]),
            "gold": gold,
            "answer": str(row["answer"]),
        })
    return tasks, {
        "dataset": "openai/grade-school-math test.jsonl",
        "dataset_sha256": file_hash(dataset),
        "dataset_rows": len(rows),
        "prompt": "lm-evaluation-harness gsm8k-cot 8-shot (first_n fixed samples)",
        "scoring": "last strict/flexible numeric extraction after normalization",
    }


def load_tasks(args: argparse.Namespace) -> tuple[list[dict], dict | None]:
    if args.suite == "longbench":
        return load_longbench(args.dataset, args.manifest)
    if args.suite == "gsm8k":
        return load_gsm8k(args.dataset)
    return json.loads(args.tasks.read_text()), None


def summarize(rows: list[dict], config: dict) -> dict:
    valid = [row for row in rows if not row.get("error")]
    scored = [row for row in valid if row.get("gold")]
    return {
        "completed": len(rows),
        "valid": len(valid),
        "errors": len(rows) - len(valid),
        "accuracy": (
            sum(bool(row.get("correct")) for row in scored) / len(scored) if scored else None
        ),
        "prefill_tok_s": metric_stats(valid, "prefill_tok_s"),
        "decode_tok_s": metric_stats(valid, "decode_tok_s"),
        "ttft_ms": metric_stats(valid, "ttft_ms"),
        "wall_s": metric_stats(valid, "wall_s"),
        "generated_tokens": metric_stats(valid, "tokens"),
        "config": config,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daemon", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument(
        "--suite", choices=("longbench", "longdecode", "gsm8k"), required=True
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", default="1")
    parser.add_argument("--max-seq", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--category")
    parser.add_argument("--task-id")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--prefill-batch", type=int, default=8)
    parser.add_argument("--q8-fused-prefill", action="store_true")
    parser.add_argument("--ple-batched-prefill", action="store_true")
    parser.add_argument("--ple-branch-batched-prefill", action="store_true")
    parser.add_argument("--runtime-home", type=Path)
    args = parser.parse_args()
    if args.suite == "longbench" and (not args.dataset or not args.manifest):
        parser.error("longbench requires --dataset and --manifest")
    if args.suite == "gsm8k" and not args.dataset:
        parser.error("gsm8k requires --dataset")
    if args.suite == "longdecode" and not args.tasks:
        parser.error("longdecode requires --tasks")

    tasks, manifest_data = load_tasks(args)
    if args.category:
        tasks = [task for task in tasks if task.get("category") == args.category]
    if args.task_id:
        tasks = [task for task in tasks if task.get("id") == args.task_id]
    if args.limit:
        tasks = tasks[: args.limit]
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if not 1 <= args.prefill_batch <= 64:
        parser.error("--prefill-batch must be between 1 and 64")
    if args.repeats > 1:
        tasks = [
            {**task, "source_id": task["id"], "id": f"{task['id']}-repeat-{repeat + 1}"}
            for task in tasks
            for repeat in range(args.repeats)
        ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "results.jsonl"
    summary_path = args.out_dir / "summary.json"
    events_dir = args.out_dir / "events"
    outputs_dir = args.out_dir / "outputs"
    events_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)
    existing = [
        json.loads(line) for line in results_path.read_text().split("\n") if line.strip()
    ] if results_path.exists() else []
    completed = {row["id"] for row in existing if not row.get("error")}
    config = {
        "suite": args.suite,
        "model_label": args.model_label,
        "model": str(args.model.resolve()),
        "model_sha256": file_hash(args.model),
        "daemon": str(args.daemon.resolve()),
        "daemon_sha256": file_hash(args.daemon),
        "physical_gpu": args.physical_gpu,
        "max_seq": args.max_seq,
        "default_max_tokens": args.max_tokens,
        "temperature": 0.0,
        "kv_mode": "q8",
        "prefill_batch": args.prefill_batch,
        "q8_fused_prefill": args.q8_fused_prefill,
        "ple_batched_prefill": args.ple_batched_prefill,
        "ple_branch_batched_prefill": args.ple_branch_batched_prefill,
        "repeats": args.repeats,
        "manifest": manifest_data,
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")

    daemon = Daemon(
        args.daemon.resolve(),
        args.out_dir / "daemon.stderr.log",
        args.physical_gpu,
        args.prefill_batch,
        args.q8_fused_prefill,
        args.ple_batched_prefill,
        args.ple_branch_batched_prefill,
        args.runtime_home,
    )
    rows = list(existing)
    try:
        loaded, load_events = daemon.request(
            {
                "type": "load",
                "model": str(args.model.resolve()),
                "params": {
                    "max_seq": args.max_seq,
                    "kv_mode": "q8",
                    "dflash_mode": "off",
                    "mtp_mode": "off",
                    "tp": 1,
                    "pp": 1,
                },
            },
            {"loaded", "error"},
            args.timeout,
        )
        (events_dir / "load.jsonl").write_text(
            "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in load_events)
        )
        if loaded.get("type") != "loaded":
            raise RuntimeError(f"load failed: {loaded}")
        daemon.request({"type": "reset", "attempt_id": 1}, {"reset", "error"}, 60)
        warm, _ = daemon.request(
            {
                "type": "generate", "id": "warmup", "attempt_id": 1,
                "prompt": "Reply with exactly: ready", "temperature": 0.0, "max_tokens": 8,
            },
            {"done", "error", "aborted"}, args.timeout,
        )
        if warm.get("type") != "done":
            raise RuntimeError(f"warmup failed: {warm}")

        for index, task in enumerate(tasks):
            if task["id"] in completed:
                continue
            attempt_id = index + 10
            daemon.request({"type": "reset", "attempt_id": attempt_id}, {"reset", "error"}, 60)
            prompt = task["prompt"]
            max_tokens = int(task.get("max_tokens", args.max_tokens))
            request = {
                "type": "generate",
                "id": task["id"],
                "attempt_id": attempt_id,
                "prompt": prompt,
                "temperature": 0.0,
                "top_p": 1.0,
                "repeat_penalty": 1.0,
                "max_tokens": max_tokens,
            }
            started = time.monotonic()
            try:
                done, events = daemon.request(request, {"done", "error", "aborted"}, args.timeout)
                wall_s = time.monotonic() - started
                visible, reasoning = collect_text(events)
                if args.suite == "gsm8k":
                    pred, pred_source = extract_gsm8k_answer(visible)
                else:
                    pred = extract_choice(visible) if task.get("gold") else None
                    pred_source = None
                row = {
                    "id": task["id"],
                    "source_id": task.get("source_id", task["id"]),
                    "ordinal": task.get("ordinal"),
                    "category": task.get("category"),
                    "prompt_md5": text_hash(prompt, "md5"),
                    "prompt_chars": len(prompt),
                    "max_tokens": max_tokens,
                    "prediction": visible,
                    "prediction_sha256": text_hash(visible),
                    "prediction_chars": len(visible),
                    "reasoning": reasoning,
                    "gold": task.get("gold"),
                    "pred": pred,
                    "pred_source": pred_source,
                    "correct": pred == task.get("gold") if task.get("gold") else None,
                    "wall_s": wall_s,
                    **{key: done.get(key) for key in (
                        "type", "finish_reason", "tokens", "tok_s", "prefill_tokens",
                        "prefill_ms", "prefill_tok_s", "decode_tok_s", "ttft_ms", "cached_tokens",
                    )},
                }
                if done.get("type") != "done":
                    row["error"] = done
            except Exception as exc:
                events = []
                visible = ""
                row = {
                    "id": task["id"], "ordinal": task.get("ordinal"),
                    "source_id": task.get("source_id", task["id"]),
                    "category": task.get("category"), "gold": task.get("gold"),
                    "error": repr(exc), "wall_s": time.monotonic() - started,
                }
            (events_dir / f"{task['id']}.jsonl").write_text(
                "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events)
            )
            (outputs_dir / f"{task['id']}.md").write_text(visible)
            with results_path.open("a") as output:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            summary_path.write_text(json.dumps(summarize(rows, config), indent=2, ensure_ascii=False) + "\n")
            print(
                f"[{len(rows):02d}/{len(tasks):02d}] {task['id']} type={row.get('type')} "
                f"prefill={row.get('prefill_tok_s')} decode={row.get('decode_tok_s')} "
                f"tokens={row.get('tokens')} correct={row.get('correct')}",
                flush=True,
            )
    finally:
        daemon.close()

    final = summarize(rows, config)
    summary_path.write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(final, indent=2, ensure_ascii=False))
    return 0 if final["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
