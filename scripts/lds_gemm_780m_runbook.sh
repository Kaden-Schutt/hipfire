#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-/tmp/hipfire-lds-tail-snop-780m}"
LOCAL_SUMMARY="${LOCAL_SUMMARY:-}"
LOCAL_KEDGE_SUMMARY="${LOCAL_KEDGE_SUMMARY:-}"

cat <<EOF
# gfx1103 / 780M LDS tail-snop repro runbook

# 1. Optional safe codegen/metadata preflight. This compiles and captures
#    codegen artifacts but does not launch the repro kernel.
BUILD_ONLY=1 OUT=$OUT-buildonly tests/gfx1103-lds-tail-snop-repro.sh

# 2. Risky repro. This is expected to exercise the HIP-719/reset path on
#    affected gfx1103 stacks. It writes report.tsv, summary.txt, and
#    artifact-summary.tsv/.md under $OUT.
OUT=$OUT tests/gfx1103-lds-tail-snop-repro.sh

# 3. Optional sharper K-edge repro.
PROFILE=kedge OUT=$OUT-kedge tests/gfx1103-lds-tail-snop-repro.sh

# 4. Rebuild summaries manually if needed.
scripts/lds_gemm_artifact_summary.sh $OUT
scripts/lds_gemm_artifact_summary.sh $OUT-kedge

# 5. Compare against a local/known summary TSV after copying it beside this
#    checkout or setting LOCAL_SUMMARY.
EOF

if [[ -n "$LOCAL_SUMMARY" || -n "$LOCAL_KEDGE_SUMMARY" ]]; then
    local_repro="${LOCAL_SUMMARY:-/path/to/local-artifact-summary.tsv}"
    local_kedge="${LOCAL_KEDGE_SUMMARY:-/path/to/local-kedge-summary.tsv}"
    cat <<EOF
scripts/lds_gemm_summary_compare.sh $local_repro $OUT/artifact-summary.tsv
scripts/lds_gemm_summary_compare.sh $local_kedge $OUT-kedge/artifact-summary.tsv
EOF
else
    cat <<EOF
scripts/lds_gemm_summary_compare.sh /path/to/local-artifact-summary.tsv $OUT/artifact-summary.tsv
scripts/lds_gemm_summary_compare.sh /path/to/local-kedge-summary.tsv $OUT-kedge/artifact-summary.tsv
EOF
fi

cat <<'EOF'

# Interpretation:
# - source-drift: compare runs only after aligning repo revision/source.
# - codegen-drift: same source compiled differently; compare selected_isa first,
#   then whole normalized ISA, then ROCm/HIP/tool env.
# - same-codegen-runtime-diff: same source/object/ISA but different outcome;
#   this points at runtime/driver/device state.
# - codegen-metadata-drift: normalized ISA matches, but raw object/disassembly
#   metadata differs, usually from build path or object metadata noise.
# - same: summary rows match on the fields the comparator tracks.
EOF
