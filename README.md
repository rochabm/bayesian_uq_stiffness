# Bayesian inversion and uncertainty quantification of myocardial stiffness

Code to reproduce the results of the paper

> J. Rocha et al., *Bayesian framework for patient-specific cardiac ... (2026)*

The stiffness is inferred as the log-stiffness m = ln C, with a Gaussian prior
of BiLaplacian type, a MAP estimate by inexact Newton-CG, and a Laplace
approximation of the posterior.

> **Status:** this README is written step by step. So far it covers only the
> **prior calibration (Table 2 and Fig. 1)**.

---

## Repository layout

```
repo_final/
├── README.md
├── paper_style.py               # common matplotlib style of all paper figures
├── data/
│   └── Patient_7_lv_tissue.vtu  # patient LV mesh with the "tissue" labels (Case 3)
└── priors/
    ├── prior_calibration.py     # calibrates (gamma, delta) of the prior -> Table 2
    └── plot_prior_calibration.py  # figures of the calibration           -> Fig. 1
```

**Run every command from the repository root.** The scripts are called as
modules (`python -m priors.<script>`) so that `paper_style.py` is found, and
the ellipsoid mesh is created once in `lv_ellipsoid/` at the root, where the
later steps also look for it.

## Requirements

| Package | Needed for |
|---|---|
| numpy, scipy, matplotlib, meshio, h5py | everything |
| [SciencePlots](https://github.com/garrettj403/SciencePlots) (optional) | paper style; without it the same look is set from plain rcParams |
| dolfinx and [cardiac-geometriesx](https://github.com/ComputationalPhysiology/cardiac-geometriesx) | ellipsoid mesh of Cases 1–2 |

The patient mesh (Case 3) is read with meshio only, so it doesn't need dolfinx.

---

## Step 1 — Prior calibration (Table 2, Fig. 1)

The prior on m = ln C has covariance

    Gamma_prior = (R^-1 M)^(k-1) R^-1,   R = gamma K + delta M + beta M_boundary,
    beta = sqrt(gamma delta) / 1.42,

with k = 2 (BiLaplacian) or k = 4 (smoother samples). The textbook (Matérn)
formulas give (gamma, delta) for a target standard deviation sigma and
correlation length rho, but they assume an unbounded domain. In the thin LV
wall the actual values differ, so `prior_calibration.py` measures them:

- the exact pointwise prior std,
- the correlation function, from exact covariance columns,

and returns a **calibrated** pair (`gamma_calibrated`, `delta_calibrated`):
- `--calibrate-rho` (implied by `--match-truth`) adjusts the input rho until
  the measured correlation length equals the target;
- gamma and delta are then scaled by one common factor so that the mean prior
  std equals sigma exactly.

With `--match-truth` the targets come from the ground truth:
- sigma = max|ln C_true − m0| / 2, so that the truth lies on the edge of the
  pointwise 95 % prior band;
- rho = correlation length of the ground-truth field.

All cases use the prior mean m0 = ln 2 (C = 2 kPa, healthy tissue).

### Commands

```bash
# Case 2 (ellipsoid, spherical fibrosis): targets from the ground truth   (~1 min)
python -m priors.prior_calibration --mesh ellipsoid --order 2 --log --m0 0.693147 \
    --truth fibrosis --match-truth --output-dir results/priors/calib_case2

# Case 3 (patient LV, segmented fibrosis): k = 4, targets from the ground truth   (~10 min)
python -m priors.prior_calibration --mesh data/Patient_7_lv_tissue.vtu --order 4 --log \
    --m0 0.6931471805599453 --truth tissue --match-truth --centre-truth --n-ref 30 \
    --output-dir results/priors/calib_case3

# Case 1 (ellipsoid, linear truth): sigma from the amplitude, rho = ventricle size   (~1 min)
python -m priors.prior_calibration --mesh ellipsoid --order 2 --log --m0 0.693147 \
    --truth linear --sigma 0.2027326 --rho 50 --output-dir results/priors/calib_case1

# Fig. 1: prior of Case 2 (m0 +- 2 sigma, prior samples, ground truth)
python -m priors.plot_prior_calibration results/priors/calib_case2 \
    --plane-point -5 -2 -9 --plane-normal 0 1 0
```

Each run writes to its `--output-dir`:

| File | Content |
|---|---|
| `calib_summary.txt` | targets, achieved std and correlation length, **`gamma_calibrated`, `delta_calibrated`** |
| `calib_corr.txt` | measured correlation vs distance, and the Matérn curve |
| `calib_fields.npz` | mesh, prior std, prior samples, truth (input of the plot script) |
| `calib_fields.xdmf/.h5`, `calib_fields.vtu` | the same fields for ParaView |

`plot_prior_calibration.py` writes into the same folder:

| File | Content |
|---|---|
| **`fig_prior_3d.pdf/.png`** | **Fig. 1** of the paper |
| `fig_prior_calibration.pdf/.png` | measured correlation vs the Matérn curve, histogram of the prior std |
| `fig_prior_samples.pdf/.png` | the same fields on a planar cut |

### Expected results (Table 2)

| Case | k | sigma (`target_sigma`) | rho [mm] (`target_rho`) | gamma (`gamma_calibrated`) | delta (`delta_calibrated`) | beta |
|---|---|---|---|---|---|---|
| 1 (linear) | 2 | 0.203 | 50* | 6.22 | 9.85 × 10⁻³ | 0.174 |
| 2 (fibrosis) | 2 | 0.549 | 11.6 (measured 11.5) | 1.457 | 1.441 × 10⁻² | 0.102 |
| 3 (patient) | 4 | 0.549 | 17.0 | 2.038 | 0.1234 | 0.353 |

\* nominal value (input of the Matérn formula); see the note on Case 1 below.

beta = sqrt(gamma delta) / 1.42 is not written to `calib_summary.txt`; compute
it from the calibrated pair.

**Notes**

- **Case 2:** reproduces Table 2 exactly (`gamma_calibrated 1.4570`,
  `delta_calibrated 0.014408`, `empirical_rho 11.5`).
- **Case 3:** the original run gave `target_rho 16.987`, `gamma_calibrated
  2.038235`, `delta_calibrated 0.12337`. `--centre-truth` and `--n-ref 30`
  were not recorded in its `calib_summary.txt`; they are the options that
  reproduce `target_rho ≈ 17`. Without `--centre-truth` the correlation of
  the truth is estimated around a non-zero mean and comes out at about 64 mm.
- **Case 1:** a linear field has no finite correlation length, so rho was set
  by hand to the ventricle size (input 50 mm, measured about 24.5 mm in the
  wall). The values used in the paper and in the Case 1 runs
  (gamma = 6.22, delta = 9.85 × 10⁻³) were set by hand from the Matérn formula,
  not produced by this script. With the same targets the script gives
  gamma = 6.61 and delta = 1.058 × 10⁻², which correspond to a prior std about
  6 % smaller.
- The prior samples are reproducible: they use a fixed seed (`--seed 0`).
