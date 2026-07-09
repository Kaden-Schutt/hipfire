#!/usr/bin/env bash
# cross_arch_guard.sh <variant_src_file> <baseline_src_file> [archs...]
#
# PER-ARCH PERF ISOLATION (policy: a win tuned for arch X must NOT change ANY other arch's compiled code —
# "-any% cross-arch is unacceptable"). Proves it by byte-exact PREPROCESSOR INVARIANCE, not "within noise":
# preprocess the kernel's DEVICE translation unit (hipcc --cuda-device-only -E --offload-arch=<other>) from
# both the baseline and the variant; if the normalized TU changes for any other arch, the edit touches that
# arch's codegen -> the change must be arch-gated (`#if defined(__gfxNNNN__)`) or moved to a .gfxNNNN file.
#
# Same preprocessed TU  => identical codegen (the engine itself caches kernels by this determinism). Device-pass
# (--cuda-device-only) is required so __gfxNNNN__ macros are actually defined; host-pass -E leaves them unset.
#
# Exit 0 + "OK" (or "SKIP <why>")  = isolated / not applicable.
# Exit 1 + "CROSS_ARCH:<archs>"    = the win changes those archs' device TU -> reject / arch-gate.
#
# env: HIP_PATH (/opt/rocm), KSRC_DIR (kernels/src) for #include resolution.
set -u
VARIANT="${1:?variant src required}"; BASE_SRC="${2:?baseline src required}"; shift 2
ARCHS="${*:-gfx1100 gfx1151 gfx1030}"
KBASE=$(basename "$VARIANT")
command -v hipcc >/dev/null 2>&1 || { echo "SKIP no-hipcc"; exit 0; }
# arch-suffixed kernels (foo.gfx1201.hip, foo.gfx12.hip, ...) are already isolated by naming — the dispatch
# only compiles them on their own arch, so they can never touch another arch. Nothing to check.
case "$KBASE" in *.gfx[0-9]*.hip) echo "SKIP arch-suffixed"; exit 0;; esac
HIPINC="${HIP_PATH:-/opt/rocm}/include"; KSRCDIR="${KSRC_DIR:-kernels/src}"
# per-kernel magic-comment flags (mirror compiler.rs per_kernel_flags: `// HIPFIRE_COMPILER_FLAGS: ...`),
# so a kernel whose #if regions depend on a -D still preprocesses the same way the engine compiles it.
PKF=$(grep -hoE '//[[:space:]]*HIPFIRE_COMPILER_FLAGS:.*' "$VARIANT" "$BASE_SRC" 2>/dev/null \
      | sed -E 's|.*HIPFIRE_COMPILER_FLAGS:[[:space:]]*||' | tr '\n' ' ' | sort -u | tr '\n' ' ')
dev_tu(){ # $1=arch $2=srcfile -> normalized device translation unit (strip #line markers, blanks, trailing ws)
  hipcc --cuda-device-only -E --offload-arch="$1" -O3 -I"$HIPINC" -I"$KSRCDIR" $PKF "$2" 2>/dev/null \
    | grep -vE '^#' | sed 's/[[:space:]]*$//' | grep -v '^[[:space:]]*$'
}
TB=$(mktemp); TV=$(mktemp); trap 'rm -f "$TB" "$TV"' EXIT
CROSS=""; CHECKED=""
for oa in $ARCHS; do
  dev_tu "$oa" "$BASE_SRC" > "$TB"; dev_tu "$oa" "$VARIANT" > "$TV"
  [ -s "$TB" ] || continue        # baseline won't preprocess for this arch (pre-existing) -> can't attribute
  CHECKED="$CHECKED $oa"
  cmp -s "$TB" "$TV" || CROSS="$CROSS $oa"
done
if [ -n "$CROSS" ]; then echo "CROSS_ARCH:${CROSS# }"; exit 1; fi
[ -n "$CHECKED" ] && echo "OK checked:${CHECKED# }" || echo "SKIP unverifiable(no-arch-preprocessed)"
exit 0
