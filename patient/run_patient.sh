#!/usr/bin/env bash
# =============================================================================
# run_patient.sh -- Case 3, patient LV with segmented fibrosis (Section 3.4)
#
# Run from the repository root (dolfinx 0.8):
#   bash patient/run_patient.sh run       # MAP + UQ runs (several hours, serial)
#   bash patient/run_patient.sh figures   # Figs. 9-11 and Table 5, from the runs
#   bash patient/run_patient.sh all       # both
#
# run:      MAP by inexact Newton-CG       (map_estimate.py) -> results/patient/map
#           Laplace approximation          (uq.py)           -> results/patient/uq
#           128 measurement points, p_endo = 3 kPa, 0 % noise, prior with k = 4
#           (gamma, delta from Step 1, calib_case3); uq.py reuses the MAP point
# figures:  Fig. 9   truth and MAP, five views         (plot_lv_views.py)
#           Fig. 10  truth, MAP, three posterior samples (plot_posteriors_lv_patient_v2.py)
#           Fig. 11  posterior summary                  (plot_uq_lv_patient.py --bounds)
#           Table 5  MAP cost and errors                (table_patient.py)
# =============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1
PYTHON=${PYTHON:-python}

R=results/patient
MAP=$R/map
UQ=$R/uq
VTU=data/Patient_7_lv_tissue.vtu
PRIOR="--order 4 --gamma 2.038 --delta 0.1234"
COMMON="--vtu $VTU --num-nodes 128 --load -3 $PRIOR"
AZ=${AZ:-0}            # camera angle around the long axis for Figs. 10 and 11

simulations() {
  mkdir -p $MAP $UQ
  "$PYTHON" -u -m patient.map_estimate $COMMON --output-dir $MAP 2>&1 | tee $MAP/run.log
  "$PYTHON" -u -m patient.uq $COMMON --k-eig 1100 --map-from $MAP/m_map_lv_patient.npy \
      --output-dir $UQ 2>&1 | tee $UQ/run.log
}

figures() {
  # Fig. 9: ground truth (top) and MAP (bottom), five views
  "$PYTHON" -m patient.plot_lv_views --h5 $MAP/map_lv_patient.h5 --out $R/fig_lv_views
  # Fig. 10: ground truth, MAP and three posterior samples
  "$PYTHON" -m patient.plot_posteriors_lv_patient_v2 --root $UQ --n-samples 3 \
      --azimuth $AZ --out $R/fig_posteriors
  # Fig. 11: truth, MAP, 5th/95th percentiles; P(C > 4 kPa), std reduction, spectrum
  "$PYTHON" -m patient.plot_uq_lv_patient --root $UQ --bounds --azimuth $AZ \
      --out $R/fig_uq_panel
  # Table 5
  "$PYTHON" -m patient.table_patient --root $MAP --out $R/table_patient
}

case "${1:-}" in
  run)     simulations ;;
  figures) figures ;;
  all)     simulations; figures ;;
  *) sed -n '2,18p' "$0"; exit 1 ;;
esac
