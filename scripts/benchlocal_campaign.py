#!/usr/bin/env python3
# Copyright (c) Kaden Schutt
"""Reproduce the BenchLocal nine-pack quality campaign.

This is developer-only orchestration.  It deliberately keeps every command in
one standalone, stdlib-only file so archived runs remain reproducible.

Operational landmines preserved from the 2026-07-31 campaigns:
* Hermes ``--all`` can fail while fetching after HA-03 or HA-16; use
  ``recover-hermes`` rather than discarding the completed scenarios.
* HA-09 can exhaust GPU memory.  An isolated retry is still canonical evidence.
* Hermes' pinned OpenAI SDK rejects ``top_k`` before making HTTP requests, so
  sampling is forced by the HTTP proxy rather than passed to that runner.
* Retry only after removing old verifier containers: Docker bind collisions are
  infrastructure failures, not model-quality failures.
* A 96-token smoke cap exposes open thinking spans.  The campaign health smoke
  must use ``max_tokens=1152`` (the manifest records this contract).
"""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "tools" / "benchlocal" / "manifest.json"
PROXY_SCRIPT = REPO_ROOT / "tools" / "benchlocal" / "sampling_proxy.mjs"
HERMES_RESULT_RE = re.compile(
    r"^\[(PASS|PARTIAL|FAIL)\]\s+(HA-\d{2})\s+score=([0-9]+(?:\.[0-9]+)?)\b",
    re.MULTILINE,
)


class CampaignError(RuntimeError):
    """An actionable campaign configuration or execution error."""


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_manifest(path):
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot load manifest {path}: {error}") from error
    if manifest.get("schema_version") != 1:
        raise CampaignError(
            f"unsupported manifest schema_version {manifest.get('schema_version')!r}"
        )
    if not isinstance(manifest.get("routes"), list) or not isinstance(
        manifest.get("packs"), list
    ):
        raise CampaignError("manifest must contain routes and packs arrays")
    return manifest


def route_map(manifest):
    return {route["slug"]: route for route in manifest["routes"]}


def require_route(manifest, slug):
    route = route_map(manifest).get(slug)
    if route is None:
        known = ", ".join(route_map(manifest))
        raise CampaignError(f"unknown route {slug!r}; expected one of: {known}")
    return route


def ordered_packs(manifest):
    return sorted(manifest["packs"], key=lambda pack: pack["order"])

def select_packs(manifest, requested):
    packs = ordered_packs(manifest)
    if not requested:
        return packs
    requested_ids = {
        pack_id.strip()
        for value in requested
        for pack_id in value.split(",")
        if pack_id.strip()
    }
    valid_ids = [pack["id"] for pack in packs]
    if not requested_ids:
        raise CampaignError(
            f"--pack requires at least one pack id; valid pack ids: {', '.join(valid_ids)}"
        )
    unknown = sorted(requested_ids.difference(valid_ids))
    if unknown:
        raise CampaignError(
            f"unknown pack id(s): {', '.join(unknown)}; valid pack ids: {', '.join(valid_ids)}"
        )
    return [pack for pack in packs if pack["id"] in requested_ids]


def absolute_campaign_root(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise CampaignError("--campaign-root must be an absolute path")
    return path


def _number(value):
    return str(value)


def _shell_command(cwd, env, argv):
    # The archive contract intentionally uses a plain, single-space rendering;
    # its paths and values contain no whitespace or shell metacharacters.
    pieces = [f"{key}={value}" for key, value in env.items()] + list(argv)
    return f"cd {cwd} && {' '.join(pieces)}"


def build_pack_command(manifest, route, pack, campaign_root):
    model = route["model_path"]
    endpoints = route["endpoints"]
    cwd = campaign_root / "packs" / pack["dir"]
    kind = pack["runner"]["kind"]
    env = {}

    if kind == "benchlocal-cli":
        env = {
            "LLAMACPP_HOST": endpoints["non_hermes_base"],
            "LLM_MODELS": f"llamacpp:{model}",
            "MODEL_REQUEST_TIMEOUT_SECONDS": str(
                manifest["sampling"]["request_timeout_seconds"]
            ),
        }
        sidecar = pack.get("sidecar")
        if sidecar:
            sidecar_url = f"http://127.0.0.1:{sidecar['host_port']}"
            env[sidecar["url_env"]] = sidecar_url
        sampling = manifest["sampling"]["base"]
        argv = list(pack["runner"]["argv"])
        argv += [
            "--model",
            f"llamacpp:{model}",
            "--temperature",
            _number(sampling["temperature"]),
            "--top-p",
            _number(sampling["top_p"]),
            "--top-k",
            _number(sampling["top_k"]),
            "--min-p",
            _number(sampling["min_p"]),
            "--timeout",
            str(manifest["sampling"]["request_timeout_seconds"]),
        ]
        if sidecar:
            argv += [sidecar["cli_flag"], sidecar_url]
        if pack.get("tool_pack"):
            argv += [
                "--repetition-penalty",
                _number(sampling["repetition_penalty"]),
                "--tools-format",
                "default",
            ]
        argv += ["--show-raw", "--json"]
    elif kind == "cli40":
        runner = pack["runner"]
        argv = list(runner["argv"])
        argv += [
            "--model",
            model,
            "--provider",
            runner["provider"],
            "--base-url",
            f"{endpoints['non_hermes_base']}/v1",
            "--docker-base-url",
            endpoints["cli40_docker_base"],
            "--auth-mode",
            runner["auth_mode"],
            "--image",
            runner["docker_image"],
            "--json",
            "--verbose",
        ]
    elif kind == "hermes":
        runner = pack["runner"]
        argv = list(runner["argv"])
        argv += [
            "--model",
            model,
            "--provider-model",
            model,
            "--provider",
            runner["provider"],
            "--base-url",
            endpoints["hermes_docker_base"],
            "--auth-mode",
            runner["auth_mode"],
            "--api-key",
            runner["api_key"],
            "--image",
            runner["docker_image"],
            "--json",
            "--verbose",
        ]
    else:
        raise CampaignError(f"unsupported runner kind {kind!r} for {pack['id']}")

    return {
        "cwd": str(cwd),
        "env": env,
        "argv": argv,
        "command": _shell_command(cwd, env, argv),
    }


def build_plan(manifest, route, campaign_root, thinking):
    # Thinking is enforced by sampling_proxy.mjs at run time.  It therefore
    # does not alter any pack argv/env field in this archived command plan.
    if thinking not in manifest["sampling"]["thinking_variants"]:
        raise CampaignError(f"manifest does not define thinking variant {thinking!r}")
    family = route["family"]
    presence = manifest["sampling"]["presence_penalty_by_family"][family]
    if route["proxy_scope"] == "hermes-only":
        presence = None
    sampling = manifest["sampling"]["base"]
    timeout = manifest["sampling"]["request_timeout_seconds"]
    shared_env = {
        "LLAMACPP_HOST": route["endpoints"]["non_hermes_base"],
        "LLM_MODELS": f"llamacpp:{route['model_path']}",
        "MODEL_REQUEST_TIMEOUT_SECONDS": str(timeout),
    }
    return {
        "assembled_at": utc_now(),
        "campaign_id": campaign_root.name,
        "route": route["slug"],
        "model": route["model_path"],
        "results_root": str(campaign_root / "results" / route["slug"]),
        "shared_env_non_hermes": shared_env,
        "sampling_contract": {
            "temperature": sampling["temperature"],
            "top_p": sampling["top_p"],
            "top_k": sampling["top_k"],
            "min_p": sampling["min_p"],
            "repetition_penalty": sampling["repetition_penalty"],
            "presence_penalty": presence,
            "timeout": timeout,
        },
        "packs": {
            pack["id"]: build_pack_command(manifest, route, pack, campaign_root)
            for pack in ordered_packs(manifest)
        },
    }


def command_plan(args, manifest):
    root = absolute_campaign_root(args.campaign_root)
    routes = route_map(manifest)
    if args.route == "all":
        selected = list(routes.values())
    else:
        selected = [require_route(manifest, args.route)]
    plans = {
        route["slug"]: build_plan(manifest, route, root, args.thinking)
        for route in selected
    }
    if args.out:
        out = Path(args.out).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        for slug, plan in plans.items():
            with open(out / f"{slug}.runner-commands.json", "w", encoding="utf-8") as handle:
                json.dump(plan, handle, indent=2)
                handle.write("\n")
        return
    payload = plans if args.route == "all" else next(iter(plans.values()))
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")


def run_checked(argv, **kwargs):
    try:
        return subprocess.run(argv, check=True, **kwargs)
    except FileNotFoundError as error:
        raise CampaignError(f"required executable not found: {argv[0]}") from error
    except subprocess.CalledProcessError as error:
        detail = ""
        if getattr(error, "stderr", None):
            detail = f": {error.stderr.strip()}"
        raise CampaignError(f"command failed ({error.returncode}): {shlex.join(argv)}{detail}") from error


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_verify(args, manifest):
    root = absolute_campaign_root(args.campaign_root)
    packs_root = root / "packs"
    failures = []
    for pack in ordered_packs(manifest):
        destination = packs_root / pack["dir"]
        try:
            if destination.exists():
                if not (destination / ".git").exists():
                    raise CampaignError(f"{destination} exists but is not a git clone")
                result = run_checked(
                    ["git", "-C", str(destination), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                )
                head = result.stdout.strip()
                if head != pack["revision"]:
                    raise CampaignError(
                        f"{pack['id']}: HEAD {head} does not match pinned {pack['revision']}"
                    )
            elif args.skip_clone:
                raise CampaignError(f"{pack['id']}: missing clone {destination} (--skip-clone)")
            else:
                packs_root.mkdir(parents=True, exist_ok=True)
                run_checked(["git", "clone", pack["repo"], str(destination)])
                run_checked(
                    ["git", "-C", str(destination), "fetch", "origin", pack["revision"]]
                )
                run_checked(
                    [
                        "git",
                        "-C",
                        str(destination),
                        "checkout",
                        "--detach",
                        "--force",
                        pack["revision"],
                    ]
                )

            metadata_path = destination / "benchlocal.pack.json"
            with open(metadata_path, encoding="utf-8") as handle:
                metadata = json.load(handle)
            if metadata.get("version") != pack["version"]:
                raise CampaignError(
                    f"{pack['id']}: metadata version {metadata.get('version')!r} "
                    f"does not match manifest {pack['version']!r}"
                )
            # Pack scenario data is executable TypeScript/MJS rather than a
            # stable JSON inventory in these pinned revisions.  Importing it
            # would violate the stdlib-only/offline verifier boundary, and
            # counting source regex matches is not reliable.  Hermes alone has
            # an explicit inventory in the manifest, so all packs intentionally
            # report the count as not independently verified here.
            print(
                f"pack {pack['id']}: revision={pack['revision']} version={pack['version']} "
                f"scenario_count={pack['scenario_count']} (manifest only; source inventory is executable)"
            )
        except (OSError, json.JSONDecodeError, CampaignError) as error:
            failures.append(str(error))
            print(f"pack {pack['id']}: ERROR {error}", file=sys.stderr)

    digest_cache = {}
    for route in manifest["routes"]:
        model = Path(route["model_path"]).expanduser()
        if not model.exists():
            print(f"model {route['slug']}: absent {model}")
            continue
        try:
            actual = digest_cache.setdefault(str(model), sha256_file(model))
        except OSError as error:
            failures.append(f"{route['slug']}: cannot hash {model}: {error}")
            print(f"model {route['slug']}: ERROR {error}", file=sys.stderr)
            continue
        if actual == route["model_sha256"]:
            print(f"model {route['slug']}: present sha256={actual} OK")
        else:
            message = (
                f"{route['slug']}: model sha256 mismatch: expected "
                f"{route['model_sha256']}, got {actual}"
            )
            failures.append(message)
            print(f"model {route['slug']}: MISMATCH expected={route['model_sha256']} actual={actual}")
    if failures:
        raise CampaignError(f"verification failed with {len(failures)} error(s)")


def proxy_spec(manifest, route, root, thinking):
    results = root / "results" / route["slug"]
    hermes_only = route["proxy_scope"] == "hermes-only"
    attest_name = (
        "hermes-sampling-attestations.jsonl"
        if hermes_only
        else "sampling-attestations.jsonl"
    )
    log_name = "hermes-sampling-proxy.log" if hermes_only else "sampling-proxy.log"
    env = {
        "LISTEN_HOST": manifest["topology"]["proxy_listen_host"],
        "LISTEN_PORT": str(route["endpoints"]["proxy_port"]),
        "TARGET_ORIGIN": f"http://127.0.0.1:{route['endpoints']['serve_port']}",
        "MODEL_SLUG": route["slug"] + ("-hermes" if hermes_only else ""),
        "ATTESTATION_LOG": str(results / attest_name),
        "PRESENCE_PENALTY": _number(
            manifest["sampling"]["presence_penalty_by_family"][route["family"]]
        ),
        "THINKING_MODE": thinking,
    }
    return env, ["node", str(PROXY_SCRIPT)], results / log_name


def _wait_proxy(route, process, timeout=30):
    url = (
        f"http://127.0.0.1:{route['endpoints']['proxy_port']}"
        "/__sampling_proxy/health"
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise CampaignError(f"sampling proxy exited early with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if 200 <= response.status < 300:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.25)
    raise CampaignError(f"sampling proxy did not become healthy at {url}")


def start_proxy(manifest, route, root, thinking):
    env_delta, argv, log_path = proxy_spec(manifest, route, root, thinking)
    log_handle = open(log_path, "ab")
    try:
        process = subprocess.Popen(
            argv,
            cwd=REPO_ROOT,
            env={**os.environ, **env_delta},
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as error:
        log_handle.close()
        raise CampaignError(f"cannot launch sampling proxy: {error}") from error
    process._benchlocal_log_handle = log_handle
    _wait_proxy(route, process)
    return process


def terminate_process(process):
    if process is None:
        return
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    handle = getattr(process, "_benchlocal_log_handle", None)
    if handle:
        handle.close()

def wait_url(url, label, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if 200 <= response.status < 300:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.25)
    raise CampaignError(f"{label} did not become healthy at {url}")


def ensure_docker():
    if not shutil.which("docker"):
        raise CampaignError("Docker is required; install/start Docker or pass --skip-sidecars")
    result = subprocess.run(["docker", "info"], capture_output=True, text=True)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise CampaignError(
            f"Docker is unavailable; start the daemon or pass --skip-sidecars: {detail}"
        )


def docker_build(pack, root):
    sidecar = pack["sidecar"]
    context = (root / "packs" / pack["dir"] / sidecar["build_context"]).resolve()
    runner = pack["runner"]
    if runner["kind"] in ("cli40", "hermes"):
        image = runner["docker_image"]
    else:
        image = f"benchlocal-{pack['id']}-verifier"
    run_checked(["docker", "build", "-t", image, str(context)])
    return image


def docker_start_sidecar(pack, image, campaign_id, route_slug):
    sidecar = pack["sidecar"]
    if "host_port" not in sidecar:
        return None
    safe_campaign = re.sub(r"[^A-Za-z0-9_.-]", "-", campaign_id)
    name = f"benchlocal-{safe_campaign}-{route_slug}-{pack['id']}"
    # Deterministic names let retries remove only their own stale container;
    # otherwise Docker bind collisions can masquerade as benchmark failures.
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    run_checked(
        [
            "docker",
            "run",
            "--rm",
            "-d",
            "--name",
            name,
            "-p",
            f"{sidecar['host_port']}:{sidecar['container_port']}",
            image,
        ]
    )
    try:
        wait_url(
            f"http://127.0.0.1:{sidecar['host_port']}/health",
            f"{pack['id']} verifier sidecar",
        )
    except Exception:
        docker_stop(name)
        raise
    return name


def docker_stop(name):
    if name:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def default_remote_root(manifest, campaign_root):
    for campaign in manifest.get("provenance", {}).get("campaigns", {}).values():
        if campaign.get("campaign_id") == campaign_root.name:
            return campaign["remote_root"]
    return f"/home/kaden/{campaign_root.name}"


def remote_launch_spec(manifest, route, campaign_root, args):
    remote_root = args.remote_root or default_remote_root(manifest, campaign_root)
    home = f"{remote_root}/homes/gpu{route['gpu']}"
    mode = route["mode"]
    environment = {
        "HOME": home,
        "HIP_VISIBLE_DEVICES": str(route["gpu"]),
        "HIPFIRE_DAEMON_BIN": f"{remote_root}/bin/daemon",
        "HIPFIRE_MODEL": route["model_path"],
        "HIPFIRE_KV_MODE": mode["kv"],
        "HIPFIRE_DFLASH_MODE": mode.get("dflash", "off"),
    }
    if "dflash_ctx_cap" in mode:
        environment["HIPFIRE_DFLASH_CTX_CAP"] = str(mode["dflash_ctx_cap"])
    if mode.get("mtp") == "on":
        environment.update(
            HIPFIRE_QWEN_MTP="1",
            HIPFIRE_MTP_SAMPLED="1",
            HIPFIRE_MTP_PREFIX_CACHE="1",
        )
    serve_argv = [
        f"{remote_root}/bin/hipfire",
        "serve",
        "127.0.0.1",
        str(route["endpoints"]["serve_port"]),
        route["model_path"],
    ]
    env_words = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in environment.items()
    )
    remote_command = (
        f"mkdir -p {shlex.quote(home + '/.hipfire')} && exec env {env_words} "
        f"{shlex.join(serve_argv)}"
    )
    daemon_ssh = ["ssh", args.remote_host, remote_command]

    proxy_port = route["endpoints"]["proxy_port"]
    local_ports = {route["endpoints"]["serve_port"]}
    for endpoint in route["endpoints"].values():
        if not isinstance(endpoint, str) or not endpoint.startswith("http"):
            continue
        parsed = urllib.parse.urlparse(endpoint)
        if parsed.port and parsed.port != proxy_port:
            local_ports.add(parsed.port)
    tunnel_ssh = ["ssh", "-N", "-o", "ExitOnForwardFailure=yes"]
    for local_port in sorted(local_ports):
        tunnel_ssh += [
            "-L",
            f"0.0.0.0:{local_port}:127.0.0.1:{route['endpoints']['serve_port']}",
        ]
    tunnel_ssh.append(args.remote_host)
    return daemon_ssh, tunnel_ssh

def wait_serve(route):
    wait_url(
        f"http://127.0.0.1:{route['endpoints']['serve_port']}/health",
        f"route {route['slug']} serve endpoint",
        timeout=300,
    )


def start_remote(manifest, route, root, args):
    daemon_argv, tunnel_argv = remote_launch_spec(manifest, route, root, args)
    try:
        daemon = subprocess.Popen(daemon_argv, start_new_session=True)
        time.sleep(1)
        if daemon.poll() is not None:
            raise CampaignError(f"remote daemon launch failed with code {daemon.returncode}")
        tunnel = subprocess.Popen(tunnel_argv, start_new_session=True)
        time.sleep(1)
        if tunnel.poll() is not None:
            raise CampaignError(f"SSH tunnel launch failed with code {tunnel.returncode}")
        return daemon, tunnel
    except Exception:
        if "daemon" in locals():
            terminate_process(daemon)
        raise



def print_dry_run(manifest, route, root, args, plan, packs):
    print(f"write-json {root / 'results' / route['slug'] / (route['slug'] + '.runner-commands.json')}")
    if args.endpoint_mode == "remote" and not args.attach_only:
        daemon, tunnel = remote_launch_spec(manifest, route, root, args)
        print(f"$ {shlex.join(daemon)}")
        print(f"$ {shlex.join(tunnel)}")
    print(f"wait http://127.0.0.1:{route['endpoints']['serve_port']}/health")
    env, argv, _ = proxy_spec(manifest, route, root, args.thinking)
    print(f"$ {_shell_command(REPO_ROOT, env, argv)}")
    print(
        f"wait http://127.0.0.1:{route['endpoints']['proxy_port']}"
        "/__sampling_proxy/health"
    )
    for pack in packs:
        if not args.skip_sidecars and pack.get("sidecar"):
            sidecar = pack["sidecar"]
            context = root / "packs" / pack["dir"] / sidecar["build_context"]
            image = pack["runner"].get(
                "docker_image", f"benchlocal-{pack['id']}-verifier"
            )
            print(f"$ {shlex.join(['docker', 'build', '-t', image, str(context)])}")
            if "host_port" in sidecar:
                print(
                    "$ "
                    + shlex.join(
                        [
                            "docker",
                            "run",
                            "--rm",
                            "-d",
                            "-p",
                            f"{sidecar['host_port']}:{sidecar['container_port']}",
                            image,
                        ]
                    )
                )
        print(f"$ {plan['packs'][pack['id']]['command']}")


def command_run(args, manifest):
    root = absolute_campaign_root(args.campaign_root)
    route = require_route(manifest, args.route)
    packs = select_packs(manifest, args.pack)
    plan = build_plan(manifest, route, root, args.thinking)
    if args.dry_run:
        print_dry_run(manifest, route, root, args, plan, packs)
        return

    for pack in packs:
        if not (root / "packs" / pack["dir"]).is_dir():
            raise CampaignError(
                f"missing pack {root / 'packs' / pack['dir']}; run verify first"
            )
    if not args.skip_sidecars and any(pack.get("sidecar") for pack in packs):
        ensure_docker()
    results = root / "results" / route["slug"]
    results.mkdir(parents=True, exist_ok=True)
    plan_path = results / f"{route['slug']}.runner-commands.json"
    with open(plan_path, "w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2)
        handle.write("\n")

    remote_processes = ()
    proxy = None
    try:
        if args.endpoint_mode == "remote" and not args.attach_only:
            remote_processes = start_remote(manifest, route, root, args)
        wait_serve(route)
        proxy = start_proxy(manifest, route, root, args.thinking)
        for pack in packs:
            container = None
            try:
                if not args.skip_sidecars and pack.get("sidecar"):
                    image = docker_build(pack, root)
                    container = docker_start_sidecar(
                        pack, image, root.name, route["slug"]
                    )
                command = plan["packs"][pack["id"]]
                stdout_path = results / f"{pack['id']}.stdout.log"
                stderr_path = results / f"{pack['id']}.stderr.log"
                try:
                    with open(stdout_path, "wb") as stdout, open(
                        stderr_path, "wb"
                    ) as stderr:
                        completed = subprocess.run(
                            command["argv"],
                            cwd=command["cwd"],
                            env={**os.environ, **command["env"]},
                            stdout=stdout,
                            stderr=stderr,
                        )
                    exit_code = completed.returncode
                except OSError as error:
                    with open(stderr_path, "ab") as stderr:
                        stderr.write(f"benchlocal_campaign: {error}\n".encode())
                    exit_code = 127
                with open(results / f"{pack['id']}.exit", "w", encoding="ascii") as handle:
                    handle.write(f"{exit_code}\n")
                print(f"{pack['id']}: exit {exit_code}")
            finally:
                docker_stop(container)
    finally:
        terminate_process(proxy)
        for process in reversed(remote_processes):
            terminate_process(process)


def split_hermes_results(text):
    matches = list(HERMES_RESULT_RE.finditer(text))
    rows = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        chunk = text[match.start() : end]
        chunk = re.sub(
            r"\ncompleted=\d+\s+pass=\d+\s+partial=\d+\s+fail=\d+\s+averageScore=[0-9.]+\s*$",
            "\n",
            chunk,
        )
        rows[match.group(2)] = {
            "status": match.group(1),
            "score": float(match.group(3)),
            "chunk": chunk.rstrip() + "\n",
        }
    return rows


def line_count(path):
    try:
        with open(path, "rb") as handle:
            return sum(1 for _ in handle)
    except FileNotFoundError:
        return 0


def average_value(values):
    value = sum(values) / len(values)
    return int(value) if value.is_integer() else value


def command_recover_hermes(args, manifest):
    root = absolute_campaign_root(args.campaign_root)
    route = require_route(manifest, args.route)
    results = root / "results" / route["slug"]
    canonical_stdout = results / "hermesagent-20.stdout.log"
    canonical_stderr = results / "hermesagent-20.stderr.log"
    canonical_exit = results / "hermesagent-20.exit"
    try:
        original_text = canonical_stdout.read_text(encoding="utf-8")
    except OSError as error:
        raise CampaignError(f"cannot read canonical Hermes output {canonical_stdout}: {error}") from error
    parsed = split_hermes_results(original_text)
    hermes_pack = next(
        pack for pack in manifest["packs"] if pack["runner"]["kind"] == "hermes"
    )
    scenario_ids = hermes_pack["runner"]["scenario_ids"]
    missing = [scenario_id for scenario_id in scenario_ids if scenario_id not in parsed]
    plan = build_plan(manifest, route, root, args.thinking)
    base_command = plan["packs"][hermes_pack["id"]]
    attest_name = (
        "hermes-sampling-attestations.jsonl"
        if route["proxy_scope"] == "hermes-only"
        else "sampling-attestations.jsonl"
    )
    attest_path = results / attest_name
    isolated_dir = results / "hermesagent-20-isolated"

    if args.dry_run:
        for scenario_id in missing:
            argv = list(base_command["argv"])
            argv[argv.index("--all")] = hermes_pack["runner"]["scenario_flag"]
            argv.insert(argv.index(hermes_pack["runner"]["scenario_flag"]) + 1, scenario_id)
            print(f"$ {_shell_command(base_command['cwd'], base_command['env'], argv)}")
        return

    if not args.skip_sidecars:
        ensure_docker()
        docker_build(hermes_pack, root)
    isolated_dir.mkdir(parents=True, exist_ok=True)
    canonical_attest_end = line_count(attest_path)
    provenance = {}
    for scenario_id, row in parsed.items():
        isolated = isolated_dir / f"{scenario_id}.stdout.log"
        isolated.write_text(row["chunk"], encoding="utf-8")
        provenance[scenario_id] = {
            "status": row["status"],
            "score": row["score"],
            "source_artifact": canonical_stdout.name,
            "source_kind": "official_pack_partial_before_fetch_failed",
            "attest_range": f"0-{canonical_attest_end}",
            "isolated_copy": str(isolated.relative_to(results)),
        }

    remote_processes = ()
    proxy = None
    try:
        if missing:
            if args.endpoint_mode == "remote" and not args.attach_only:
                remote_processes = start_remote(manifest, route, root, args)
            wait_serve(route)
            proxy = start_proxy(manifest, route, root, args.thinking)
        for scenario_id in missing:
            argv = list(base_command["argv"])
            all_index = argv.index("--all")
            argv[all_index : all_index + 1] = [
                hermes_pack["runner"]["scenario_flag"],
                scenario_id,
            ]
            stdout_path = isolated_dir / f"{scenario_id}.stdout.log"
            stderr_path = isolated_dir / f"{scenario_id}.stderr.log"
            before = line_count(attest_path)
            with open(stdout_path, "wb") as stdout, open(stderr_path, "wb") as stderr:
                completed = subprocess.run(
                    argv,
                    cwd=base_command["cwd"],
                    env={**os.environ, **base_command["env"]},
                    stdout=stdout,
                    stderr=stderr,
                )
            after = line_count(attest_path)
            text = stdout_path.read_text(encoding="utf-8", errors="replace")
            recovered = split_hermes_results(text).get(scenario_id)
            if recovered is None:
                raise CampaignError(
                    f"isolated {scenario_id} run exited {completed.returncode} without a parseable result; "
                    f"preserved {stdout_path} and {stderr_path}"
                )
            parsed[scenario_id] = recovered
            provenance[scenario_id] = {
                "status": recovered["status"],
                "score": recovered["score"],
                "source_artifact": str(stdout_path.relative_to(results)),
                "source_kind": "official_isolated_scenario_run",
                "attest_range": f"{before}-{after}",
                "isolated_copy": str(stdout_path.relative_to(results)),
            }
    finally:
        terminate_process(proxy)
        for process in reversed(remote_processes):
            terminate_process(process)

    still_missing = [scenario_id for scenario_id in scenario_ids if scenario_id not in parsed]
    if still_missing:
        raise CampaignError(f"Hermes recovery incomplete; missing: {', '.join(still_missing)}")
    statuses = [parsed[scenario_id]["status"] for scenario_id in scenario_ids]
    scores = [parsed[scenario_id]["score"] for scenario_id in scenario_ids]
    summary = {
        "completed": len(scenario_ids),
        "pass": statuses.count("PASS"),
        "partial": statuses.count("PARTIAL"),
        "fail": statuses.count("FAIL"),
        "averageScore": average_value(scores),
        "exit": 0 if all(status == "PASS" for status in statuses) else 1,
    }
    stitched = "".join(parsed[scenario_id]["chunk"] for scenario_id in scenario_ids)
    stitched += (
        f"\ncompleted={summary['completed']} pass={summary['pass']} "
        f"partial={summary['partial']} fail={summary['fail']} "
        f"averageScore={summary['averageScore']}\n"
    )
    canonical_stdout.write_text(stitched, encoding="utf-8")
    canonical_stderr.touch(exist_ok=True)
    canonical_exit.write_text(f"{summary['exit']}\n", encoding="ascii")
    record = {
        "assembled_at": utc_now(),
        "method": "scenario-isolated-stitch",
        "runner_restored": True,
        "canonical": {
            "stdout": str(canonical_stdout),
            "stderr": str(canonical_stderr),
            "exit": str(canonical_exit),
        },
        "summary": summary,
        "scenarios": {scenario_id: provenance[scenario_id] for scenario_id in scenario_ids},
    }
    provenance_path = results / "hermesagent-20-provenance.json"
    with open(provenance_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print(
        f"Hermes restored: completed={summary['completed']} pass={summary['pass']} "
        f"partial={summary['partial']} fail={summary['fail']} averageScore={summary['averageScore']}"
    )


def add_common(parser, route=False):
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--campaign-root", required=True)
    if route:
        parser.add_argument("--route", required=True)
    parser.add_argument(
        "--thinking", choices=("disabled", "medium"), default="disabled"
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    plan = subparsers.add_parser("plan", help="emit canonical runner commands")
    add_common(plan, route=True)
    plan.add_argument("--out")

    verify = subparsers.add_parser("verify", help="verify pinned packs and models")
    verify.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    verify.add_argument("--campaign-root", required=True)
    verify.add_argument("--skip-clone", action="store_true")

    run = subparsers.add_parser("run", help="execute one route's nine packs")
    add_common(run, route=True)
    run.add_argument("--endpoint-mode", choices=("local", "remote"), default="local")
    run.add_argument("--remote-host", default="hiptrx")
    run.add_argument("--remote-root")
    run.add_argument("--attach-only", action="store_true")
    run.add_argument("--skip-sidecars", action="store_true")
    run.add_argument(
        "--pack",
        action="append",
        help="run one pack id; repeat the flag or pass comma-separated ids",
    )
    run.add_argument("--dry-run", action="store_true")

    recover = subparsers.add_parser(
        "recover-hermes", help="rerun missing Hermes scenarios and stitch results"
    )
    add_common(recover, route=True)
    recover.add_argument("--endpoint-mode", choices=("local", "remote"), default="local")
    recover.add_argument("--remote-host", default="hiptrx")
    recover.add_argument("--remote-root")
    recover.add_argument("--attach-only", action="store_true")
    recover.add_argument("--skip-sidecars", action="store_true")
    recover.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.subcommand == "plan":
            command_plan(args, manifest)
        elif args.subcommand == "verify":
            command_verify(args, manifest)
        elif args.subcommand == "run":
            command_run(args, manifest)
        elif args.subcommand == "recover-hermes":
            command_recover_hermes(args, manifest)
        else:
            parser.error(f"unknown subcommand {args.subcommand}")
    except CampaignError as error:
        print(f"benchlocal_campaign: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
