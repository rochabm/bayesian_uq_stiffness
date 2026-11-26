# Bayesian inversion and uncertainty quantification of myocardial stiffness

Code to reproduce the results of the paper

> *Bayesian framework for patient-specific cardiac ... (2026)*

The stiffness is inferred as the log-stiffness m = ln C, with a Gaussian prior of BiLaplacian type, a MAP estimate by inexact Newton-CG, and a Laplace approximation of the posterior.

> **Status:** this README is written step by step. So far it covers the
> **prior calibration (Table 2, Fig. 1)** and the **Taylor test (Fig. 2)**.

---

## Repository layout

```
bayesian_uq_stiffness/
├── README.md
├── paper_style.py               # common matplotlib style of all paper figures
├── cardiac_utils.py             # measurement-point selection, F at the points, output
├── hessian_ucq.py               # Hessian-vector product (incremental forward/adjoint)
├── data/
│   └── Patient_7_lv_tissue.vtu  # patient LV mesh with the "tissue" labels (Case 3)
├── priors/
│   ├── prior_calibration.py     # calibrates (gamma, delta) of the prior -> Table 2
│   └── plot_prior_calibration.py  # figures of the calibration           -> Fig. 1
└── verification/
    └── taylor_test.py           # Taylor remainder test of gradient and Hessian -> Fig. 2
```

**Run every command from the repository root.** The scripts are called as
modules (`python -m <folder>.<script>`) so that the modules at the root
(`paper_style.py`, `cardiac_utils.py`, `hessian_ucq.py`) are found, and the
ellipsoid mesh is created once in `lv_ellipsoid/` at the root, where the later
steps also look for it.

## Requirements

| Package | Needed for |
|---|---|
| numpy, scipy, matplotlib, meshio, h5py | everything |
| [SciencePlots](https://github.com/garrettj403/SciencePlots) (optional) | paper style; without it the same look is set from plain rcParams |
| dolfinx (**0.8**) and [cardiac-geometriesx](https://github.com/ComputationalPhysiology/cardiac-geometriesx) | ellipsoid mesh of Cases 1–2, forward and inverse problems |

The patient mesh (Case 3) is read with meshio only, so the prior calibration
of Case 3 doesn't need dolfinx.

**Use dolfinx 0.8.** The forward solver uses `dolfinx.fem.petsc.NonlinearProblem`
with `dolfinx.nls.petsc.NewtonSolver`. This API changed in dolfinx 0.10, where
the scripts stop with
`TypeError: NonlinearProblem.__init__() takes 3 positional arguments but 4 were given`.
The node ordering also differs between dolfinx versions, which changes the
selected measurement points; the results of the paper were computed with
dolfinx 0.8.

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
# Case 1 (ellipsoid, linear truth): sigma from the amplitude, rho = ventricle size   (~1 min)
# NOTE: the paper uses gamma = 6.22, delta = 9.85e-3, set by hand for these targets;
#       this command gives the calibrated pair gamma = 6.61, delta = 1.058e-2.
python -m priors.prior_calibration --mesh ellipsoid --order 2 --log --m0 0.693147 \
    --truth linear --sigma 0.2027326 --rho 50 --output-dir results/priors/calib_case1

# Case 2 (ellipsoid, spherical fibrosis): targets from the ground truth   (~1 min)
python -m priors.prior_calibration --mesh ellipsoid --order 2 --log --m0 0.693147 \
    --truth fibrosis --match-truth --output-dir results/priors/calib_case2

# Fig. 1: prior of Case 2 (m0 +- 2 sigma, prior samples, ground truth) -> fig_prior_3d.pdf
python -m priors.plot_prior_calibration results/priors/calib_case2 \
    --plane-point -5 -2 -9 --plane-normal 0 1 0

# Case 3 (patient LV, segmented fibrosis): k = 4, targets from the ground truth   (~10 min)
python -m priors.prior_calibration --mesh data/Patient_7_lv_tissue.vtu --order 4 --log \
    --m0 0.6931471805599453 --truth tissue --match-truth --centre-truth \
    --output-dir results/priors/calib_case3

# Optional: the same figures for Case 3 (not in the paper)
python -m priors.plot_prior_calibration results/priors/calib_case3
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
| **`fig_prior_3d.pdf/.png`** | **Fig. 1** of the paper (from `calib_case2`) |
| `fig_prior_calibration.pdf/.png` | measured correlation vs the Matérn curve, histogram of the prior std |
| `fig_prior_samples.pdf/.png` | the same fields on a planar cut |

The calibrated pairs in `calib_summary.txt` are the gamma and delta of
Table 2 (beta = sqrt(gamma delta) / 1.42), and the values passed to the
inverse problems with `--gamma` / `--delta`.

---

## Step 2 — Gradient and Hessian verification (Fig. 2)

### Command

```bash
python -m verification.taylor_test_log --output-dir results/verification-log
```

It prints a table of R_grad and R_hess with the rate between consecutive eps,
and the average rates.

### Expected result

```
Average rate R_grad: 2.000  (expected 2)  ✓ PASS
Average rate R_hess: 3.002  (expected 3)  ✓ PASS
```

| File | Content |
|---|---|
| **`fig_taylor.pdf/.png`** | **Fig. 2** of the paper (paper style from `paper_style.py`) |
| `taylor_remainders.txt` | eps, R_grad, R_hess |

