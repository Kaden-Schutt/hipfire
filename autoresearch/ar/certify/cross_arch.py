# Copyright (c) Kaden Schutt
"""cross_arch — per-arch perf isolation via preprocessor invariance (ported from cross_arch_guard.sh).

Policy: a win tuned for arch X must NOT change ANY other arch's compiled code ("−any% cross-arch is
unacceptable"). We prove isolation by byte-exact PREPROCESSOR INVARIANCE, not "within noise": preprocess
the kernel's DEVICE translation unit (``hipcc --cuda-device-only -E --offload-arch=<other>``) from both
the baseline and the variant; if the normalized TU changes for any other arch, the edit touches that
arch's codegen → it must be arch-gated (``#if defined(__gfxNNNN__)``) or moved to a ``.gfxNNNN`` file.

Device-pass (``--cuda-device-only``) is required so ``__gfxNNNN__`` macros are actually defined; a
host-pass ``-E`` leaves them unset. Same preprocessed TU ⇒ identical codegen (the engine caches kernels
by exactly this determinism).

``check_cross_arch(kernel_file, arch, other_archs, repo) -> list[str]`` returns the OTHER archs whose
device TU changed (empty = isolated). The hipcc invocation is behind an injectable ``preprocess`` seam
so the diff logic is no-GPU-unit-testable; without hipcc on the box it SKIPs (returns []).
"""
import os
import re
import shutil
import subprocess

_ARCH_SUFFIX = re.compile(r"\.gfx[0-9][^.]*\.hip$")


def _hipcc_preprocess(arch, kernel_file, variant, repo, base_sha):
    """Normalized device TU for `kernel_file` at `arch`. When `variant` is False, preprocess the
    BASELINE version (``git show <base_sha>:<file>`` into a temp), else the on-disk variant. Returns
    the normalized TU string (``#`` line markers / blanks / trailing ws stripped), or "" on failure."""
    hipinc = os.path.join(os.environ.get("HIP_PATH", "/opt/rocm"), "include")
    ksrcdir = os.environ.get("KSRC_DIR", "kernels/src")
    if variant:
        src = kernel_file
        cleanup = None
    else:
        base = subprocess.run(["git", "show", f"{base_sha}:{kernel_file}"], cwd=repo,
                              capture_output=True)
        if base.returncode != 0:
            return ""
        src = os.path.join("/tmp", f"cross_arch_base_{arch}_{os.path.basename(kernel_file)}")
        with open(src, "wb") as f:
            f.write(base.stdout)
        cleanup = src
    # per-kernel magic-comment flags (mirror compiler.rs `// HIPFIRE_COMPILER_FLAGS:`)
    pkf = ""
    try:
        with open(kernel_file, errors="ignore") as f:
            for line in f:
                m = re.search(r"//\s*HIPFIRE_COMPILER_FLAGS:\s*(.*)", line)
                if m:
                    pkf += " " + m.group(1).strip()
    except OSError:
        pass
    cmd = (f"hipcc --cuda-device-only -E --offload-arch={arch} -O3 -I{hipinc} -I{ksrcdir} {pkf} "
           f"'{src}' 2>/dev/null")
    try:
        r = subprocess.run(cmd, shell=True, cwd=repo, capture_output=True, text=True)
        lines = [ln.rstrip() for ln in r.stdout.splitlines()
                 if ln and not ln.startswith("#") and ln.strip()]
        return "\n".join(lines)
    finally:
        if cleanup and os.path.exists(cleanup):
            os.remove(cleanup)


def check_cross_arch(kernel_file, arch, other_archs, repo, base_sha=None, preprocess=None):
    """Return the OTHER archs whose device TU changes between baseline and variant of `kernel_file`.

    Empty list = isolated (OK). A non-empty list = the edit touches those archs' codegen → reject /
    arch-gate. Arch-suffixed files (``foo.gfx1201.hip``) are already dispatch-isolated → skip (return
    []). If hipcc is unavailable (and no seam injected) → SKIP (return []); we never fabricate a
    CROSS_ARCH. An arch whose BASELINE won't preprocess (empty TU) is unattributable → skipped.
    """
    base = os.path.basename(kernel_file)
    if _ARCH_SUFFIX.search(base):
        return []
    pp = preprocess or _hipcc_preprocess
    if pp is _hipcc_preprocess and not shutil.which("hipcc"):
        return []
    changed = []
    for oa in other_archs:
        base_tu = pp(oa, kernel_file, False, repo, base_sha)
        if not base_tu or not base_tu.strip():
            continue                      # can't attribute (baseline won't preprocess for this arch)
        var_tu = pp(oa, kernel_file, True, repo, base_sha)
        if base_tu != var_tu:
            changed.append(oa)
    return changed
