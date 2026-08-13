#!/usr/bin/env python3
"""Run isolated headless startup and control-path checks on generated Pygame code."""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Callable


OUTPUT_LIMIT = 64 * 1024


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def drain_output(stream, output: bytearray) -> None:
    """Drain a pipe completely while retaining only a bounded prefix."""
    try:
        while chunk := stream.read(4096):
            remaining = OUTPUT_LIMIT - len(output)
            if remaining > 0:
                output.extend(chunk[:remaining])
    finally:
        stream.close()


def startup_worker(path: Path, quit_delay_s: float) -> int:
    """Import one artifact, enter its real loop, post QUIT, and require clean exit."""
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    import pygame

    module = load_module(path, "generated_startup_worker")
    if hasattr(module, "main"):
        run: Callable[[], object] = module.main
    elif hasattr(module, "SnakeGame"):
        run = module.SnakeGame().run
    elif hasattr(module, "TetrisGame"):
        run = module.TetrisGame().run
    else:
        print(json.dumps({"ok": False, "error": "no runnable entry point"}))
        return 1

    posted = threading.Event()
    consumed = threading.Event()
    stop_monitor = threading.Event()
    state: dict[str, object] = {"display_ready": False, "post_error": None}
    original_event_get = pygame.event.get

    def tracked_event_get(*args, **kwargs):
        events = original_event_get(*args, **kwargs)
        if any(event.type == pygame.QUIT for event in events):
            consumed.set()
        return events

    pygame.event.get = tracked_event_get

    def request_quit() -> None:
        try:
            deadline = time.monotonic() + quit_delay_s
            while not stop_monitor.is_set() and time.monotonic() < deadline:
                if pygame.display.get_init() and pygame.display.get_surface() is not None:
                    state["display_ready"] = True
                    pygame.event.post(pygame.event.Event(pygame.QUIT))
                    posted.set()
                    return
                time.sleep(0.01)
        except Exception as exc:  # Generated program lifecycle is under test.
            state["post_error"] = f"{type(exc).__name__}: {exc}"

    monitor = threading.Thread(target=request_quit, daemon=True)
    monitor.start()
    run_error = None
    try:
        run()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            run_error = f"SystemExit: {exc.code}"
    except Exception as exc:  # Generated program is the object under test.
        run_error = f"{type(exc).__name__}: {exc}"
    finally:
        stop_monitor.set()
        monitor.join(timeout=1.0)
        pygame.event.get = original_event_get
        pygame.quit()

    result = {
        "ok": bool(posted.is_set() and consumed.is_set() and state["display_ready"] and not state["post_error"] and not run_error),
        "quit_event_posted": posted.is_set(),
        "quit_event_consumed": consumed.is_set(),
        "display_ready": state["display_ready"],
        "post_error": state["post_error"],
        "run_error": run_error,
    }
    print(json.dumps(result))
    return 0 if result["ok"] else 1


def startup_smoke(path: Path, timeout_s: float) -> dict:
    env = os.environ.copy()
    env.update({
        "SDL_VIDEODRIVER": "dummy",
        "SDL_AUDIODRIVER": "dummy",
        "PYGAME_HIDE_SUPPORT_PROMPT": "1",
    })
    init_wait_s = max(0.5, timeout_s - 1.0)
    with tempfile.TemporaryDirectory(prefix="gemma4-pygame-startup-") as temp_dir:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(path), str(init_wait_s)],
            cwd=temp_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        assert process.stdout is not None and process.stderr is not None
        stdout_bytes = bytearray()
        stderr_bytes = bytearray()
        readers = [
            threading.Thread(target=drain_output, args=(process.stdout, stdout_bytes), daemon=True),
            threading.Thread(target=drain_output, args=(process.stderr, stderr_bytes), daemon=True),
        ]
        for reader in readers:
            reader.start()
        timed_out = False
        try:
            returncode = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            returncode = process.wait(timeout=2.0)
        finally:
            # The worker may have spawned descendants before exiting normally.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for reader in readers:
            reader.join(timeout=2.0)
        if any(reader.is_alive() for reader in readers):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            for reader in readers:
                reader.join(timeout=1.0)
        stdout_text = stdout_bytes.decode(errors="replace").strip()
        stderr_text = stderr_bytes.decode(errors="replace").strip()

    worker_result = None
    for line in reversed(stdout_text.splitlines()):
        try:
            worker_result = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    return {
        "ok": bool(not timed_out and returncode == 0 and worker_result and worker_result.get("ok")),
        "result": "timeout" if timed_out else f"exited_{returncode}",
        "timeout_s": timeout_s,
        "worker": worker_result,
        "stdout_tail": stdout_text,
        "stderr_tail": stderr_text,
    }


def check_snake(path: Path, label: str) -> dict:
    import pygame

    module = load_module(path, f"generated_{label.replace('-', '_')}_snake")
    game = module.SnakeGame()
    source = path.read_text()
    result: dict[str, object] = {}
    try:
        if label.startswith("e2b"):
            pygame.event.clear()
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
            game.handle_input()
            result["pause"] = game.game_state == module.STATE_PAUSED
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
            game.handle_input()
            result["resume"] = game.game_state == module.STATE_RUNNING
            if result["resume"]:
                old_head = game.snake[0]
                game.update_game_logic()
                result["movement_after_resume"] = game.snake[0] != old_head
            else:
                result["movement_after_resume"] = False
            game.game_state = module.STATE_GAME_OVER
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
            game.handle_input()
            result["restart"] = game.game_state == module.STATE_RUNNING
            result["speed_control_source_hint"] = "clock.tick(self.speed_level" in source
            result["high_score_source_hint"] = "high_score" in source
        else:
            game.handle_input(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
            result["pause"] = game.game_state == module.GameState.PAUSED
            game.handle_input(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
            result["resume"] = game.game_state == module.GameState.RUNNING
            old_head = game.snake[0]
            game.update_game_logic()
            result["movement_after_resume"] = game.snake[0] != old_head
            game.game_state = module.GameState.GAME_OVER
            game.handle_input(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
            result["restart"] = game.game_state == module.GameState.RUNNING
            result["speed_control_source_hint"] = "clock.tick(self.speed)" in source
            result["high_score_update_source_hint"] = (
                "self.high_score = max" in source or "self.save_high_score()" in source
            )
        result["resizable_window_source_hint"] = "pygame.RESIZABLE" in source
    finally:
        pygame.quit()
    core = ("pause", "resume", "movement_after_resume", "restart")
    result["core_ok"] = all(bool(result[key]) for key in core)
    result["ok"] = bool(result["core_ok"])
    return result


def check_tetris_e2b(module: ModuleType, source: str) -> dict:
    game = module.TetrisGame()
    x0 = game.current_piece.x
    horizontal_move = bool(game.move_piece(-1, 0)) and game.current_piece.x != x0
    y0 = game.current_piece.y
    game.drop_piece()
    manual_drop = game.current_piece.y != y0
    rotation0 = game.current_piece.rotation
    game.rotate_piece()
    rotation = game.current_piece.rotation != rotation0
    return {
        "horizontal_move": horizontal_move,
        "manual_drop": manual_drop,
        "rotation": rotation,
        "hold_supported": callable(getattr(game, "hold_piece", None)),
        "seven_bag_source_hint": "bag" in source.lower(),
        "ghost_piece_source_hint": "ghost" in source.lower(),
    }


def check_tetris_e4b(module: ModuleType, source: str) -> dict:
    game = module.Game()
    x0 = game.current_piece.x
    game.move_piece(-1)
    result: dict[str, object] = {"horizontal_move": game.current_piece.x != x0}
    y0 = game.current_piece.y
    game.update(game.fall_speed)
    result["automatic_fall"] = game.current_piece.y != y0
    rotation0 = game.current_piece.rotation
    try:
        game.current_piece.rotate()
        result["rotation"] = game.current_piece.rotation != rotation0
    except Exception as exc:
        result["rotation"] = False
        result["rotation_error"] = f"{type(exc).__name__}: {exc}"
    hold = getattr(game, "hold_piece", None)
    try:
        if not callable(hold):
            raise TypeError(f"hold_piece is {type(hold).__name__}, not callable")
        hold()
        result["hold_supported"] = True
    except Exception as exc:
        result["hold_supported"] = False
        result["hold_error"] = f"{type(exc).__name__}: {exc}"
    result["seven_bag_source_hint"] = "bag" in source.lower()
    result["ghost_piece_source_hint"] = "ghost" in source.lower()
    return result


def check_tetris(path: Path, label: str) -> dict:
    import pygame

    source = path.read_text()
    try:
        ast.parse(source)
    except SyntaxError as exc:
        return {"ok": False, "syntax_ok": False, "error": f"{exc.msg} at line {exc.lineno}"}
    module = load_module(path, f"generated_{label.replace('-', '_')}_tetris")
    try:
        result = check_tetris_e2b(module, source) if label.startswith("e2b") else check_tetris_e4b(module, source)
    finally:
        pygame.quit()
    result["syntax_ok"] = True
    required = ("horizontal_move", "rotation", "hold_supported")
    result["ok"] = all(bool(result.get(key)) for key in required)
    return result


def safe_check(checker: Callable[[Path, str], dict], path: Path, label: str) -> dict:
    try:
        with tempfile.TemporaryDirectory(prefix="gemma4-pygame-controls-") as temp_dir:
            with contextlib.chdir(temp_dir):
                return checker(path, label)
    except BaseException as exc:  # Generated code may call sys.exit().
        return {"ok": False, "harness_error": f"{type(exc).__name__}: {exc}"}
    finally:
        try:
            import pygame

            pygame.quit()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--startup-seconds", type=float, default=5.0)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.report = args.report.resolve()

    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

    report: dict[str, dict] = {}
    for label in ("e2b-longdecode", "e4b-longdecode"):
        model: dict[str, object] = {}
        for task, checker in (("snake-python", check_snake), ("tetris-python", check_tetris)):
            path = (args.root / "extracted-code" / f"{label}-{task}.py").resolve()
            try:
                ast.parse(path.read_text())
                model[f"{task}-startup"] = startup_smoke(path, args.startup_seconds)
            except SyntaxError as exc:
                model[f"{task}-startup"] = {
                    "ok": False,
                    "result": "syntax_error",
                    "error": f"{exc.msg} at line {exc.lineno}",
                }
            model[f"{task}-controls"] = safe_check(checker, path, label)
        report[label] = model

    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if all(item.get("ok", False) for model in report.values() for item in model.values()) else 1


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        raise SystemExit(startup_worker(Path(sys.argv[2]), float(sys.argv[3])))
    raise SystemExit(main())
