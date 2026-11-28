#!/usr/bin/env bash
# =============================================================================
# run_uq_study.sh -- Laplace-approximation UQ for Cases 1-2 (Section 3.2)
#
# Run from the repository root (dolfinx 0.8):
#   bash ellipsoid/run_uq_study.sh run       # simulations (UQ runs), ~3 h
#   bash ellipsoid/run_uq_study.sh figures   # all figures and tables, from the runs
#   bash ellipsoid/run_uq_study.sh all       # both
#
# run:      UQ (uq.py), 0 % noise
#             64 points, Case 1 and Case 2        -> results/ellipsoid/uq
#             32 points, Case 2                   -> results/ellipsoid/uq_s32
#             16 points, Case 2                   -> results/ellipsoid/uq_s16
# figures:  Figs. 4, 5  eigenvalues; truth, MAP, posterior samples   (plot_uq.py)
#           Fig. 6      regional stiffness, 1000 samples             (uq_qoi.py)
#           Fig. 7      truth, P(C > 4 kPa), std reduction           (uq_regions.py)
#           Fig. 8      16 measurement points                        (plot_scarce.py)
#           Table 4     64 / 32 / 16 points                          (table_sensors.py)
#           Fig. 3      MAP maps with the colours of Fig. 5          (plot_maps.py;
#                       needs the runs of run_noise_study.sh)
# =============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1
PYTHON=${PYTHON:-python}

R=results/ellipsoid
UQ64=$R/uq
UQ32=$R/uq_s32
UQ16=$R/uq_s16
COMMON="--data-noise 0"          # 0 % noise; likelihood std 1e-3 (as Step 3 at 0 %)

run_uq() {   # run_uq <case> <num_nodes> <k_eig> <output_dir>
  mkdir -p "$4"
  "$PYTHON" -u -m ellipsoid.uq --case "$1" --num-nodes "$2" --k-eig "$3" $COMMON \
      --output-dir "$4" 2>&1 | tee "$4/run_$1.log"
}

simulations() {
  run_uq linear   64 500 $UQ64
  run_uq fibrosis 64 500 $UQ64
  run_uq fibrosis 32 250 $UQ32
  run_uq fibrosis 16 250 $UQ16
}

figures() {
  # Figs. 4 and 5
  "$PYTHON" -m ellipsoid.plot_uq --root $UQ64 --eig
  "$PYTHON" -m ellipsoid.plot_uq --root $UQ64 --case linear
  "$PYTHON" -m ellipsoid.plot_uq --root $UQ64 --case fibrosis
  # Fig. 6 (64 points) and the regional stiffness for Table 4
  for d in $UQ64 $UQ32 $UQ16; do
    "$PYTHON" -m ellipsoid.uq_qoi --root $d --case fibrosis --n-samples 1000
  done
  # Fig. 7 (64 points) and the regional statistics for Table 4
  "$PYTHON" -m ellipsoid.uq_regions --root $UQ64 --case fibrosis --num-nodes 64 --vr-clim 0 1
  "$PYTHON" -m ellipsoid.uq_regions --root $UQ32 --case fibrosis --num-nodes 32 --vr-clim 0 1
  "$PYTHON" -m ellipsoid.uq_regions --root $UQ16 --case fibrosis --num-nodes 16 --vr-clim 0 1
  # Fig. 8 (16 points)
  "$PYTHON" -m ellipsoid.plot_scarce --root $UQ16 --n-sensors 16 --out $R/fig_scarce_s16
  # Table 4
  "$PYTHON" -m ellipsoid.table_sensors --runs $UQ64:64 $UQ32:32 $UQ16:16 \
      --out $R/table_sensors
  # Fig. 3 with the colour range of Fig. 5
  "$PYTHON" -m ellipsoid.plot_maps --root $R/noise_study --clim-from-uq $UQ64 \
      --out $R/fig_maps.pdf
}

case "${1:-}" in
  run)     simulations ;;
  figures) figures ;;
  all)     simulations; figures ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
