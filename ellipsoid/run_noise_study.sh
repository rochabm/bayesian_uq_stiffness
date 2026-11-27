#!/usr/bin/env bash
# =============================================================================
# run_noise_study.sh -- MAP noise study for Cases 1-2 (Table 3, Fig. 3)
#
# ex06_map_comparison for
#   cases   : linear (Case 1), fibrosis (Case 2), each with its own prior
#   noise   : 0 %, 1 %, 3 % of RMS(Fd - I)   (likelihood std = data-noise std;
#             for 0 % a fixed sigma = 1e-3)
#   methods : Newton-CG first (all levels and cases), then L-BFGS-B
# then builds the LaTeX tables with make_latex_tables.
#
# Run from the repository root (dolfinx 0.8). Extra arguments go to ex06, e.g.
#   bash ellipsoid/run_noise_study.sh --gtol 1e-6 --ftol 1e-12 --maxiter 1000
#
# Output: $OUT/<method>/noise_XXX/<case>/   (default OUT=results/ellipsoid/noise_study)
#
# Runs are sequential on purpose (FFCx JIT cache).
# =============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1   # print line by line through the pipe to tee

OUT=${OUT:-results/ellipsoid/noise_study}
LEVELS=(0 0.01 0.03)
PHASES=${PHASES:-"newtoncg lbfgsb"}
CASES=${CASES:-"linear fibrosis"}
PYTHON=${PYTHON:-python}

# ---- case-specific priors (gamma, delta of the BiLaplacian on m = log C) ----
PRIOR_LINEAR=${PRIOR_LINEAR:-"--gamma 6.22 --delta 9.85e-3"}
PRIOR_FIBROSIS=${PRIOR_FIBROSIS:-"--gamma 1.457 --delta 1.441e-2"}

prior_for() {
    case "$1" in
        linear)   echo "$PRIOR_LINEAR" ;;
        fibrosis) echo "$PRIOR_FIBROSIS" ;;
        *) echo "unknown case $1" >&2; exit 1 ;;
    esac
}

mkdir -p "$OUT"
export OUTPUT_DIR="$OUT/forward"   # files written by ex03 during data generation
echo "Noise study -> $OUT"
echo "  phases : $PHASES"
echo "  cases  : $CASES"
echo "  prior linear   : $PRIOR_LINEAR"
echo "  prior fibrosis : $PRIOR_FIBROSIS"
echo "  extra ex06 args: $*"

for method in $PHASES; do
    echo
    echo "######################################################################"
    echo "# PHASE: $method"
    echo "######################################################################"
    for lvl in "${LEVELS[@]}"; do
        pct=$("$PYTHON" -c "print(f'{round($lvl*100):03d}')")
        dir="$OUT/$method/noise_${pct}"
        mkdir -p "$dir"
        for case in $CASES; do
            prior=$(prior_for "$case")
            echo
            echo "======================================================================"
            echo " $method | noise ${lvl} | $case | prior: $prior"
            echo "======================================================================"
            start=$(date +%s)
            # shellcheck disable=SC2086   (prior is intentionally word-split)
            "$PYTHON" -u -m ellipsoid.map_comparison \
                --methods "$method" \
                --cases "$case" \
                $prior \
                --data-noise "$lvl" \
                --data-noise-seed 1234 \
                --match-noise-std \
                --output-dir "$dir" \
                "$@" 2>&1 | tee "$dir/run_${case}.log"
            echo " done in $(( $(date +%s) - start )) s"
        done
    done
    # tables after each phase (after phase 1 they show Newton-CG only)
    echo
    echo "Building LaTeX tables from what is available ..."
    "$PYTHON" -m ellipsoid.make_latex_tables --root "$OUT" --out "$OUT/tables.tex" \
        --standalone "$OUT/tables_standalone.tex"
    "$PYTHON" -m ellipsoid.make_latex_tables --root "$OUT" --paper --out "$OUT/table_paper.tex" \
        --standalone "$OUT/table_paper_standalone.tex"
done

echo
echo "All phases done."
