#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${HIPFIRE_DIFFUSION_SMOKE_MODEL:-/tmp/hipfire-tiny-sd-diffusion.hfq}"
WIDTH="${HIPFIRE_DIFFUSION_SMOKE_WIDTH:-64}"
HEIGHT="${HIPFIRE_DIFFUSION_SMOKE_HEIGHT:-64}"
STEPS="${HIPFIRE_DIFFUSION_SMOKE_STEPS:-1}"
CFG_SCALE="${HIPFIRE_DIFFUSION_SMOKE_CFG_SCALE:-1.0}"
SEED="${HIPFIRE_DIFFUSION_SMOKE_SEED:-501}"
PROMPT="${HIPFIRE_DIFFUSION_SMOKE_PROMPT:-hipfire SDAPI diffusion smoke test}"
BATCH_SIZE="${HIPFIRE_DIFFUSION_SMOKE_BATCH_SIZE:-1}"
N_ITER="${HIPFIRE_DIFFUSION_SMOKE_N_ITER:-1}"
ROCM_DEVICE_ID="${HIPFIRE_DIFFUSION_SMOKE_ROCM_DEVICE_ID:-}"
HIGHRES="${HIPFIRE_DIFFUSION_SMOKE_HIGHRES:-0}"
HIGHRES_SCALE="${HIPFIRE_DIFFUSION_SMOKE_HIGHRES_SCALE:-1.0}"
HIGHRES_WIDTH="${HIPFIRE_DIFFUSION_SMOKE_HIGHRES_WIDTH:-}"
HIGHRES_HEIGHT="${HIPFIRE_DIFFUSION_SMOKE_HIGHRES_HEIGHT:-}"
SERVER_SMOKE_LOCK="${HIPFIRE_SERVER_SMOKE_LOCK:-${TMPDIR:-/tmp}/hipfire-server-smoke.lock}"
SERVER_SMOKE_LOCK_WAIT="${HIPFIRE_SERVER_SMOKE_LOCK_WAIT:-300}"

exec 9>"$SERVER_SMOKE_LOCK"
if ! flock -w "$SERVER_SMOKE_LOCK_WAIT" 9; then
  echo "timed out waiting for server smoke lock: $SERVER_SMOKE_LOCK" >&2
  exit 2
fi

if [[ ! -f "$MODEL" ]]; then
  echo "missing diffusion HFQ model: $MODEL" >&2
  echo "set HIPFIRE_DIFFUSION_SMOKE_MODEL to a runnable diffusion .hfq artifact" >&2
  exit 2
fi

python3 - "$ROOT" "$MODEL" "$WIDTH" "$HEIGHT" "$STEPS" "$CFG_SCALE" "$SEED" "$PROMPT" "$BATCH_SIZE" "$N_ITER" "$ROCM_DEVICE_ID" "$HIGHRES" "$HIGHRES_SCALE" "$HIGHRES_WIDTH" "$HIGHRES_HEIGHT" <<'PY'
import base64
import json
import math
import os
import shlex
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zlib
from typing import Any

(
    root,
    model,
    width_s,
    height_s,
    steps_s,
    cfg_scale_s,
    seed_s,
    prompt,
    batch_size_s,
    n_iter_s,
    rocm_device_id_s,
    highres_s,
    highres_scale_s,
    highres_width_s,
    highres_height_s,
) = sys.argv[1:]
width = int(width_s)
height = int(height_s)
steps = int(steps_s)
cfg_scale = float(cfg_scale_s)
seed = int(seed_s)
batch_size = int(batch_size_s)
n_iter = int(n_iter_s)
expected_images = batch_size * n_iter
rocm_device_id = int(rocm_device_id_s) if rocm_device_id_s else None
highres_enabled = highres_s.lower() in ("1", "true", "yes", "on")
highres_scale = float(highres_scale_s)
highres_width = int(highres_width_s) if highres_width_s else None
highres_height = int(highres_height_s) if highres_height_s else None
request_timeout = float(os.environ.get("HIPFIRE_DIFFUSION_SDAPI_SMOKE_REQUEST_TIMEOUT", "420"))

if batch_size < 1:
    raise RuntimeError(f"batch size must be positive, got {batch_size}")
if n_iter < 1:
    raise RuntimeError(f"n_iter must be positive, got {n_iter}")
if highres_enabled and (not math.isfinite(highres_scale) or highres_scale <= 0.0):
    raise RuntimeError(f"highres scale must be positive, got {highres_scale}")


def pick_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def fetch_json(url: str, body: dict[str, Any] | None = None, timeout: float = 30.0) -> dict[str, Any]:
    data = None
    method = "GET"
    headers = {"Content-Type": "application/json"}
    if body is not None:
        data = json.dumps(body, separators=(",", ":")).encode("utf-8")
        method = "POST"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_health(base_url: str, proc: subprocess.Popen[str], log_path: str) -> dict[str, Any]:
    deadline = time.time() + 120.0
    last_err: Exception | None = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}; log={log_path}")
        try:
            health = fetch_json(f"{base_url}/health", timeout=2.0)
            if health.get("status") == "ok":
                return health
        except Exception as err:
            last_err = err
        time.sleep(0.25)
    raise RuntimeError(f"server did not become healthy; last_err={last_err}; log={log_path}")


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(kind)
    crc = zlib.crc32(payload, crc)
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc & 0xFFFF_FFFF)


def make_half_mask_png_b64(width: int, height: int) -> str:
    raw = bytearray()
    for _y in range(height):
        raw.append(0)
        for x in range(width):
            value = 255 if x >= width // 2 else 0
            raw.extend((value, value, value))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(bytes(raw)))
        + png_chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode("ascii")


def decode_image_payload(image: str) -> bytes:
    payload = image.split(",", 1)[1] if image.startswith("data:image/") and "," in image else image
    return base64.b64decode(payload, validate=True)


def validate_png(image: str, expected_width: int, expected_height: int, label: str) -> bytes:
    png = decode_image_payload(image)
    if not png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError(f"{label}: response image is not a PNG")
    if png[12:16] != b"IHDR":
        raise RuntimeError(f"{label}: PNG missing first IHDR chunk")
    actual_width, actual_height = struct.unpack(">II", png[16:24])
    if (actual_width, actual_height) != (expected_width, expected_height):
        raise RuntimeError(
            f"{label}: PNG dimensions {actual_width}x{actual_height} != {expected_width}x{expected_height}"
        )
    return png


def expected_seeds(first_seed: int, count: int = expected_images) -> list[int]:
    return [first_seed + idx for idx in range(count)]


def scaled_highres_dimension(dimension: int, scale: float) -> int:
    return max(1, int(math.floor((dimension * scale) + 0.5)))


def aspect_scaled_dimension(target: int, source_num: int, source_den: int) -> int:
    return max(1, int(math.floor(((target * source_num) / source_den) + 0.5)))


def expected_highres_dimensions() -> tuple[int, int]:
    if highres_width is not None and highres_height is not None:
        return highres_width, highres_height
    if highres_width is not None:
        return highres_width, aspect_scaled_dimension(highres_width, height, width)
    if highres_height is not None:
        return aspect_scaled_dimension(highres_height, width, height), highres_height
    return (
        scaled_highres_dimension(width, highres_scale),
        scaled_highres_dimension(height, highres_scale),
    )


def sdapi_request(
    base_url: str,
    route: str,
    body: dict[str, Any],
    label: str,
    *,
    expected_count: int = expected_images,
    expected_width: int = width,
    expected_height: int = height,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        out = fetch_json(f"{base_url}{route}", body, timeout=request_timeout)
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{label}: HTTP {err.code}: {detail}") from err
    if "error" in out:
        raise RuntimeError(f"{label}: response error: {out['error']}")
    images = out.get("images")
    if not isinstance(images, list) or len(images) != expected_count:
        raise RuntimeError(f"{label}: expected {expected_count} image(s), got {out}")
    for idx, image in enumerate(images):
        validate_png(image, expected_width, expected_height, f"{label}[{idx}]")
    info_raw = out.get("info")
    if not isinstance(info_raw, str):
        raise RuntimeError(f"{label}: missing SDAPI info string: {out}")
    info = json.loads(info_raw)
    if info.get("backend") != "hipfire-diffusion-hfq":
        raise RuntimeError(f"{label}: unexpected backend info: {info}")
    if info.get("batch_size") != expected_count:
        raise RuntimeError(f"{label}: expected merged batch_size {expected_count}, got {info}")
    if rocm_device_id is not None and info.get("runtime") != "rocm-hybrid-reference":
        raise RuntimeError(f"{label}: expected rocm-hybrid-reference runtime, got {info}")
    return out, info


port = pick_port()
base_url = f"http://127.0.0.1:{port}"
log_file = tempfile.NamedTemporaryFile("w", prefix="hipfire-diffusion-sdapi-smoke-", suffix=".log", delete=False)
log_path = log_file.name
env = os.environ.copy()
env.setdefault("HIPFIRE_NO_PID_FILE", "1")

custom_cmd = os.environ.get("HIPFIRE_DIFFUSION_SDAPI_SMOKE_CMD")
cargo_features = os.environ.get("HIPFIRE_DIFFUSION_SDAPI_SMOKE_CARGO_FEATURES", "").strip()
if rocm_device_id is not None and not cargo_features and not custom_cmd:
    cargo_features = "rocm"
release_bin = os.path.join(root, "target", "release", "hipfire")
if custom_cmd:
    cmd = shlex.split(custom_cmd)
else:
    if cargo_features:
        cmd = [
            "cargo",
            "run",
            "-q",
            "--release",
            "-p",
            "hipfire-cli",
            "--features",
            cargo_features,
            "--",
        ]
    elif os.path.exists(release_bin) and os.access(release_bin, os.X_OK):
        cmd = [release_bin]
    else:
        cmd = ["cargo", "run", "-q", "--release", "-p", "hipfire-cli", "--"]
    cmd += ["serve", "--host", "127.0.0.1", "--port", str(port), "--model", model]
proc = subprocess.Popen(
    cmd,
    cwd=root,
    stdin=subprocess.DEVNULL,
    stdout=log_file,
    stderr=log_file,
    text=True,
    env=env,
)

try:
    wait_health(base_url, proc, log_path)
    options = fetch_json(f"{base_url}/sdapi/v1/options", timeout=10.0)
    if options.get("sd_model_checkpoint") != model:
        raise RuntimeError(f"options did not report default model {model!r}: {options}")
    samplers = fetch_json(f"{base_url}/sdapi/v1/samplers", timeout=10.0)
    if not isinstance(samplers, list) or not samplers:
        raise RuntimeError(f"samplers endpoint returned no samplers: {samplers}")
    loras = fetch_json(f"{base_url}/sdapi/v1/loras", timeout=10.0)
    if not isinstance(loras, list):
        raise RuntimeError(f"loras endpoint returned non-list: {loras}")
    refresh_loras = fetch_json(f"{base_url}/sdapi/v1/refresh-loras", {}, timeout=10.0)
    if not isinstance(refresh_loras, dict):
        raise RuntimeError(f"refresh-loras endpoint returned non-object: {refresh_loras}")
    for command in ("server-kill", "server-restart", "server-stop"):
        command_response = fetch_json(f"{base_url}/sdapi/v1/{command}", {}, timeout=10.0)
        if (
            not isinstance(command_response, dict)
            or command_response.get("success") is not False
            or command_response.get("command") != command
            or command_response.get("server_command_supported") is not False
        ):
            raise RuntimeError(f"{command} endpoint returned unexpected response: {command_response}")
    skip_response = fetch_json(f"{base_url}/sdapi/v1/skip", {}, timeout=10.0)
    if skip_response != {}:
        raise RuntimeError(f"skip endpoint returned unexpected response: {skip_response}")
    skipped_progress = fetch_json(f"{base_url}/sdapi/v1/progress", timeout=10.0)
    if (
        skipped_progress.get("state", {}).get("skipped") is not True
        or skipped_progress.get("state", {}).get("interrupted") is not False
    ):
        raise RuntimeError(f"skip endpoint did not mark skipped state without interrupting: {skipped_progress}")

    txt_body = {
        "model": model,
        "prompt": prompt,
        "negative_prompt": "",
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "sampler_name": "Euler",
        "seed": seed,
        "seed_resize_from_w": width,
        "seed_resize_from_h": height,
        "batch_size": batch_size,
        "n_iter": n_iter,
        "send_images": True,
        "save_images": False,
    }
    if rocm_device_id is not None:
        txt_body["hipfire_rocm_device_id"] = rocm_device_id
    txt, txt_info = sdapi_request(base_url, "/sdapi/v1/txt2img", txt_body, "txt2img")
    if txt_info.get("pipeline") != "StableDiffusionPipeline":
        raise RuntimeError(f"txt2img did not use StableDiffusionPipeline: {txt_info}")
    if txt_info.get("seeds") != expected_seeds(seed):
        raise RuntimeError(f"txt2img seeds wrong: {txt_info}")
    if txt_info.get("seed_resize_from_w") != width or txt_info.get("seed_resize_from_h") != height:
        raise RuntimeError(f"txt2img seed resize info wrong: {txt_info}")
    png_info = fetch_json(f"{base_url}/sdapi/v1/png-info", {"image": txt["images"][0]}, timeout=10.0)
    if prompt not in str(png_info.get("info", "")):
        raise RuntimeError(f"png-info did not include prompt: {png_info}")

    highres_info = None
    if highres_enabled:
        highres_body = dict(txt_body)
        highres_body.update({
            "prompt": f"{prompt} highres",
            "seed": seed + 3,
            "n_iter": 1,
            "enable_hr": True,
            "hr_scale": highres_scale,
            "hr_second_pass_steps": steps,
            "denoising_strength": 1.0,
        })
        if highres_width is not None:
            highres_body["hr_resize_x"] = highres_width
        if highres_height is not None:
            highres_body["hr_resize_y"] = highres_height
        expected_highres_width, expected_highres_height = expected_highres_dimensions()
        _highres, highres_info = sdapi_request(
            base_url,
            "/sdapi/v1/txt2img",
            highres_body,
            "txt2img-hires",
            expected_count=batch_size,
            expected_width=expected_highres_width,
            expected_height=expected_highres_height,
        )
        if highres_info.get("mode") != "txt2img-hires" or highres_info.get("highres") is not True:
            raise RuntimeError(f"highres txt2img mode/info wrong: {highres_info}")
        if highres_info.get("seeds") != expected_seeds(seed + 3, batch_size):
            raise RuntimeError(f"highres txt2img seeds wrong: {highres_info}")
        if highres_info.get("hr_second_pass_steps") != steps:
            raise RuntimeError(f"highres txt2img second-pass steps wrong: {highres_info}")

    img_body = dict(txt_body)
    img_body.update({
        "prompt": f"{prompt} img2img",
        "seed": seed + 1,
        "init_images": txt["images"][:batch_size],
        "denoising_strength": 0.5,
    })
    _img, img_info = sdapi_request(base_url, "/sdapi/v1/img2img", img_body, "img2img")
    if img_info.get("mode") != "img2img" or img_info.get("masked") is not False:
        raise RuntimeError(f"img2img mode/masked info wrong: {img_info}")
    if img_info.get("seeds") != expected_seeds(seed + 1):
        raise RuntimeError(f"img2img seeds wrong: {img_info}")

    masked_body = dict(img_body)
    masked_body.update({
        "prompt": f"{prompt} masked",
        "seed": seed + 2,
        "mask": make_half_mask_png_b64(width, height),
        "mask_blur": 1,
        "mask_round": True,
        "inpainting_fill": 2,
    })
    _masked, masked_info = sdapi_request(base_url, "/sdapi/v1/img2img", masked_body, "masked-img2img")
    if masked_info.get("mode") != "img2img" or masked_info.get("masked") is not True:
        raise RuntimeError(f"masked img2img mode/masked info wrong: {masked_info}")
    if masked_info.get("seeds") != expected_seeds(seed + 2):
        raise RuntimeError(f"masked img2img seeds wrong: {masked_info}")
    if masked_info.get("inpainting_fill") != 2 or masked_info.get("masked_content") != "latent noise":
        raise RuntimeError(f"masked img2img inpainting_fill info wrong: {masked_info}")
    if masked_info.get("inpaint_full_res") is not True:
        raise RuntimeError(f"masked img2img inpaint_full_res info wrong: {masked_info}")

    progress = fetch_json(f"{base_url}/sdapi/v1/progress", timeout=10.0)
    if progress.get("state", {}).get("interrupted"):
        raise RuntimeError(f"progress endpoint reports interrupted after smoke: {progress}")
    if progress.get("state", {}).get("skipped"):
        raise RuntimeError(f"progress endpoint kept stale skipped state after generation: {progress}")

    log_text = open(log_path, "r", encoding="utf-8", errors="replace").read()
    if "pre-warm load failed" in log_text and "tokenizer not found" in log_text:
        raise RuntimeError(
            "diffusion model was routed through the LLM prewarm path; "
            f"see server log {log_path}"
        )
    if "diffusion warm-up complete" not in log_text:
        raise RuntimeError(f"server log did not record diffusion warm-up; see {log_path}")

    print(json.dumps({
        "status": "pass",
        "base_url": base_url,
        "model": model,
        "batch_size": batch_size,
        "n_iter": n_iter,
        "images_per_route": expected_images,
        "rocm_device_id": rocm_device_id,
        "loras": len(loras),
        "server_command_noops": 3,
        "skip_noop": True,
        "txt2img": {"backend": txt_info.get("backend"), "runtime": txt_info.get("runtime")},
        "txt2img_highres": (
            None
            if highres_info is None
            else {
                "mode": highres_info.get("mode"),
                "width": highres_info.get("width"),
                "height": highres_info.get("height"),
            }
        ),
        "img2img": {"masked": img_info.get("masked")},
        "masked_img2img": {"masked": masked_info.get("masked")},
        "log": log_path,
    }, indent=2))
finally:
    proc.terminate()
    try:
        proc.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10.0)
    log_file.close()
PY
