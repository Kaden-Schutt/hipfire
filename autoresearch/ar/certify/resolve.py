# Copyright (c) Kaden Schutt
"""resolve — symbol→file resolution + DEAD_FILE / NO_OP target guards.

Ported from the ``ab_certify_v2p.sh`` "Bug-2 guard": kernels are EMBEDDED via ``include_str!`` (there
is NO runtime disk-by-name loader), and the BOD names __global__ SYMBOLS whose source often lives in a
differently-named base file. A swap into a file that is NOT compiled-in — or a variant byte-identical
to the baseline — is a NO-OP recompile whose A/B "win" is a pure measurement phantom (the overnight
"26 wins" inflation). This resolves the symbol to its compiled-in source, else rejects.

Resolution (mirrors the bash, word-anchored so ``k8`` != ``k8_indexed``):
  1. If ``kernels/src/<kernel>.hip`` is itself referenced from ``crates/*/src`` (include_str!'d) →
     that file IS the target.
  2. Else grep ``kernels/src/*.hip`` for a ``__global__ ... <kernel>(`` definition, keep only the
     matches whose basename is include_str!'d. Exactly ONE → that file. Zero / ambiguous → DeadFile.
  3. If a variant + base_sha are supplied and the variant is byte-identical to the baseline version of
     the resolved file → NoOp.
"""
import os
import re
import subprocess


class DeadFile(Exception):
    """The kernel symbol has no unique compiled-in (include_str!'d) source file — un-targetable."""


class NoOp(Exception):
    """The variant is byte-identical to the baseline version of the resolved file — no real edit."""


def _compiled_in(basename, repo):
    """True iff `basename` (e.g. 'gemv.hip') is referenced from crates/*/src (include_str!'d)."""
    r = subprocess.run(f'grep -rqF "{basename}" crates/*/src/', shell=True, cwd=repo,
                       capture_output=True)
    return r.returncode == 0


def _symbol_files(kernel, repo):
    """kernels/src/*.hip files whose text defines `__global__ ... <kernel>(` (word-anchored)."""
    ksrc = os.path.join(repo, "kernels", "src")
    if not os.path.isdir(ksrc):
        return []
    pat = re.compile(r"__global__.*\b" + re.escape(kernel) + r"\s*\(")
    out = []
    for name in sorted(os.listdir(ksrc)):
        if not name.endswith(".hip"):
            continue
        try:
            with open(os.path.join(ksrc, name), "r", errors="ignore") as f:
                txt = f.read()
        except OSError:
            continue
        if pat.search(txt):
            out.append(name)
    return out


def _git_show_bytes(repo, ref_path):
    """Bytes of `ref_path` (e.g. 'HEAD:kernels/src/x.hip') at a git ref, or None if unavailable."""
    r = subprocess.run(["git", "show", ref_path], cwd=repo, capture_output=True)
    return r.stdout if r.returncode == 0 else None


def resolve_kernel_file(kernel, repo, variant=None, base_sha=None, _base_reader=_git_show_bytes):
    """Resolve a kernel SYMBOL to its unique compiled-in ``kernels/src/*.hip`` source.

    Returns the repo-relative path (e.g. ``kernels/src/gemv.hip``). Raises:
      * ``DeadFile`` — no compiled-in file / ambiguous symbol→file mapping (un-targetable via swap).
      * ``NoOp``     — ``variant`` supplied and byte-identical to the ``base_sha`` version of the file.

    ``_base_reader(repo, "<sha>:<path>") -> bytes|None`` is injectable for no-GPU testing.
    """
    ksrc_name = f"{kernel}.hip"
    if _compiled_in(ksrc_name, repo):
        resolved = f"kernels/src/{ksrc_name}"
    else:
        matches = [f for f in _symbol_files(kernel, repo) if _compiled_in(f, repo)]
        if len(matches) == 1:
            resolved = f"kernels/src/{matches[0]}"
        else:
            raise DeadFile(
                f"kernels/src/{ksrc_name} not compiled in (include_str!) and symbol->file is "
                f"{len(matches)}-way — un-targetable via file-swap")

    if variant is not None and base_sha is not None:
        base_bytes = _base_reader(repo, f"{base_sha}:{resolved}")
        if base_bytes is not None:
            with open(variant, "rb") as f:
                if f.read() == base_bytes:
                    raise NoOp(f"variant byte-identical to baseline {resolved} — no real edit")
    return resolved
