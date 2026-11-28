# Bayesian inversion and uncertainty quantification of myocardial stiffness

Code to reproduce the results of the paper

> *Bayesian framework for patient-specific cardiac ... (2026)*

The stiffness is inferred as the log-stiffness m = ln C, with a Gaussian prior of BiLaplacian type, a MAP estimate by inexact Newton-CG, and a Laplace approximation of the posterior.

> **Status:** this README is written step by step. So far it covers the
> **prior calibration (Table 2, Fig. 1)**, the **Taylor test (Fig. 2)** and the
> **MAP estimation of Cases 1–2 (Table 3, Fig. 3)** and the **UQ of Cases 1–2
> (Figs. 4–8, Table 4)**.

---

## Repository layout

```
bayesian_uq_stiffness/
├── README.md
├── paper_style.py               # common matplotlib style of all paper figures
├── cardiac_utils.py             # measurement-point selection, F at the points, output
├── hessian_ucq.py               # Hessian-vector product (incremental forward/adjoint)
├── newton_cg_solver.py          # inexact Newton-CG with bounds (Eisenstat-Walker)
├── data/
│   └── Patient_7_lv_tissue.vtu  # patient LV mesh with the "tissue" labels (Case 3)
├── priors/
│   ├── prior_calibration.py     # calibrates (gamma, delta) of the prior -> Table 2
│   └── plot_prior_calibration.py  # figures of the calibration           -> Fig. 1
├── verification/
│   ├── taylor_test_log.py       # Taylor test of the MAP functional in m = ln C -> Fig. 2
│   └── taylor_test.py           # the same test in the earlier parameterization in C
└── ellipsoid/                   # Cases 1 (linear) and 2 (fibrosis), ellipsoid LV
    ├── forward.py                # forward problem, synthetic data
    ├── map_comparison.py         # MAP: L-BFGS-B vs inexact Newton-CG
    ├── run_noise_study.sh       # map_comparison for both cases, 0/1/3 % noise -> Table 3
    ├── make_latex_tables.py     # LaTeX tables from the MAP metrics  -> Table 3
    ├── plot_maps.py             # ground truth and MAP maps          -> Fig. 3
    ├── uq.py                    # Laplace approximation: MAP, eigenpairs, variance, samples
    ├── run_uq_study.sh          # UQ runs (64/32/16 points), then all UQ figures and tables
    ├── plot_uq.py               # eigenvalue spectra; truth, MAP, posterior samples -> Figs. 4, 5
    ├── uq_qoi.py                # regional stiffness from 1000 samples              -> Fig. 6
    ├── uq_regions.py            # truth, P(C > 4 kPa), std reduction; regional stats -> Fig. 7
    ├── plot_scarce.py           # 16 measurement points                              -> Fig. 8
    └── table_sensors.py         # 64 / 32 / 16 points                                -> Table 4
```

**Run every command from the repository root.** The scripts are called as
modules (`python -m <folder>.<script>`) so that the modules at the root
(`paper_style.py`, `cardiac_utils.py`, `hessian_ucq.py`, `newton_cg_solver.py`)
are found, and the
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

Taylor remainder test of the adjoint gradient G and the Hessian action H of
the cost functional J, at a base point m along a random unit direction h:

    R_grad(eps) = | J(m + eps h) - J(m) - eps <G, h> |                    = O(eps^2)
    R_hess(eps) = | J(m + eps h) - J(m) - eps <G, h> - eps^2/2 <h, H h> | = O(eps^3)

Setup (as in the paper): ellipsoid mesh, synthetic data from the linear ground
truth of Case 1, homogeneous base point C = 3 kPa, eps from 5e-3 to 5e-1. Every
evaluation solves the forward problem from zero displacement with the same
load steps (p_endo = -2 kPa in 2 steps), a Newton tolerance of 1e-12 and a
direct solver (MUMPS), so that the remainders are reproducible down to
round-off.

### Command

```bash
python -m verification.taylor_test      # about 1 min, serial, dolfinx 0.8
```

It prints a table of R_grad and R_hess with the rate between consecutive eps,
and the average rates.

### Expected result

```
     eps          R_grad          R_hess   rate_grad   rate_hess
 5.0e-01      4.9146e-07      9.3445e-13           -           -
 2.0e-01      7.8633e-08      6.0670e-14        2.00        2.98
 1.0e-01      1.9658e-08      7.6201e-15        2.00        2.99
 5.0e-02      4.9146e-09      9.5476e-16        2.00        3.00
 2.0e-02      7.8633e-10      6.1361e-17        2.00        3.00
 1.0e-02      1.9658e-10      7.5951e-18        2.00        3.01
 5.0e-03      4.9146e-11      8.9112e-19        2.00        3.09

Average rate R_grad: 2.000  (expected 2)  ✓ PASS
Average rate R_hess: 2.992  (expected 3)  ✓ PASS
```

The average is taken over the first four rates (the last two are closest to
round-off). The paper reports 2.00 and 2.98.

Outputs in `results/verification/` (change with `--output-dir`):

| File | Content |
|---|---|
| **`fig_taylor.pdf/.png`** | **Fig. 2** of the paper (paper style from `paper_style.py`) |
| `taylor_remainders.txt` | eps, R_grad, R_hess |

**Notes**

- The test differentiates with respect to C itself, with the regularized
  functional J = J_data + alpha J_smooth (alpha = 1e-3), not the MAP functional
  in m = ln C used for the results. `HessianOperator` (`hessian_ucq.py`) is the
  same code in both cases (UFL differentiates exp(m) automatically), so the
  test checks the same implementation.
- The ellipsoid mesh is read from `lv_ellipsoid/` (created in Step 1); if the
  folder does not exist, the script creates it.

---

## Step 3 — MAP estimation, Cases 1–2: L-BFGS-B vs inexact Newton-CG (Table 3, Fig. 3)

`map_comparison.py` computes the MAP point of m = ln C with both
optimizers, from the same initial guess (C ~ U(2, 10) kPa, seed 57), with the
same bounds (C in [1, 16] kPa) and the same stopping criteria. The synthetic
data are generated by `forward.py` (64 measurement
points, endocardial pressure 10 kPa); Gaussian noise of 0, 1 or 3 % of
RMS(F_d − I) is added to each component of F. For noisy data the likelihood
std equals the noise std; for clean data a fixed sigma = 1e-3 is used.

`run_noise_study.sh` runs `map_comparison.py` for both cases and the three noise levels, each
case with its own prior from Step 1 (Case 1: gamma = 6.22, delta = 9.85e-3;
Case 2: gamma = 1.457, delta = 1.441e-2), first Newton-CG and then L-BFGS-B,
and builds the tables.

### Command

```bash
bash ellipsoid/run_noise_study.sh --gtol 1e-6 --ftol 1e-12 --maxiter 1000    # ~1 h, serial
```

`--maxiter 1000` is needed: L-BFGS-B takes more than the default of 200
iterations for Case 2 at 0 % noise. To run only part of the study:

```bash
CASES=fibrosis PHASES=newtoncg bash ellipsoid/run_noise_study.sh --gtol 1e-6 --ftol 1e-12 --maxiter 1000
```

### Outputs

In `results/ellipsoid/noise_study/` (change with `OUT=...`):

| File | Content |
|---|---|
| **`table_paper.tex`** | **Table 3** of the paper (`table_paper_standalone.tex` compiles on its own) |
| `tables.tex` | four detailed tables: data and noise, cost, accuracy, agreement of the two methods |
| `<method>/noise_XXX/<case>/metrics_<case>.json` | all settings, costs and errors of one run |
| `<method>/noise_XXX/<case>/map_comparison_<case>.xdmf/.h5` | fields (truth, initial guess, MAP, errors) for ParaView and for Fig. 3 |
| `<method>/noise_XXX/<case>/m_map_<method>_<case>.npy` | the MAP point |
| `<method>/noise_XXX/run_<case>.log` | console output |
| `forward/` | files written by `forward.py` during data generation |

### Expected results (Table 3)

| Problem | Noise | L-BFGS-B iter. / fwd | Newton-CG iter. / fwd / Hm | e_C^L2 [%] | e_C^core [%] |
|---|---|---|---|---|---|
| Case 1 | 0 % | 93 / 104 | 10 / 11 / 112 | 0.21 | – |
| Case 1 | 1 % | 81 / 87 | 9 / 10 / 106 | 2.15 | – |
| Case 1 | 3 % | 129 / 136 | 9 / 10 / 171 | 3.12 | – |
| Case 2 | 0 % | 243 / 267 | 12 / 13 / 285 | 1.44 | 2.15 |
| Case 2 | 1 % | 119 / 133 | 11 / 12 / 158 | 5.23 | 4.93 |
| Case 2 | 3 % | 67 / 73 | 9 / 10 / 95 | 8.74 | 8.65 |

The Newton-CG counts (iterations, forward solves, Hessian-vector products) and
the errors are reproduced exactly. The L-BFGS-B counts can differ by a few
iterations between machines or library builds: near the optimum the objective is
flat, so round-off changes which line-search steps are accepted. A rerun gave
128 / 135 (Case 1, 3 %), 249 / 274 (Case 2, 0 %) and 120 / 132 (Case 2, 1 %),
with the same MAP points and errors. Wall times and speed-ups depend on the
machine. Both methods reach the same MAP point (max nodal |Δm| < 4 × 10⁻⁶).

**Note:** `forward.py` also writes `dof_coords.csv/.vtk/.xyz` (the measurement points)
into the folder the command is run from.

### Fig. 3 — ground truth and MAP maps

Drawn from the runs above; nothing is recomputed. One row per case:
ground truth, MAP at 0 %, 1 % and 3 % noise (Newton-CG; L-BFGS-B gives the same
MAP points). Each heart is rendered off-screen with PyVista (matplotlib 3D if
PyVista is not installed).

```bash
python -m ellipsoid.plot_maps --root results/ellipsoid/noise_study \
    --out results/ellipsoid/fig_maps.pdf
```

Options: `--errors` also writes |C_MAP − C_true| (`fig_maps_errors.pdf`);
`--clim-linear a b` / `--clim-fibrosis a b` fix the colour range of a row;
`--view`, `--zoom` change the camera.

**Colour scale:** by default each row uses the range of its ground truth
(2–3 kPa and 2–6 kPa). In the paper the two rows share the colour range of the
posterior-sample figure (Fig. 5), so that Figs. 3 and 5 can be compared
directly. That needs the UQ results of Step 4; then:

```bash
python -m ellipsoid.plot_maps --root results/ellipsoid/noise_study \
    --clim-from-uq results/ellipsoid/uq --out results/ellipsoid/fig_maps.pdf
```

---

## Step 4 — Uncertainty quantification, Cases 1–2 (Figs. 4–8, Table 4)

`uq.py` computes the MAP point (inexact Newton-CG), the leading eigenpairs of
the prior-preconditioned data-misfit Hessian (randomized double pass), the
pointwise posterior variance of the Laplace approximation and posterior
samples. All runs use 0 % noise and the priors of Step 1.

`run_uq_study.sh` has two parts:

```bash
bash ellipsoid/run_uq_study.sh run       # the 4 UQ runs (64 points ×2 cases, 32 and 16 points)
bash ellipsoid/run_uq_study.sh figures   # Figs. 3–8 and Table 4, in one go
bash ellipsoid/run_uq_study.sh all       # both
```

| Part | What | Output in `results/ellipsoid/` |
|---|---|---|
| `run` | 64 points, Cases 1 and 2 (500 eigenpairs) | `uq/<case>/` |
| `run` | 32 and 16 points, Case 2 (250 eigenpairs) | `uq_s32/fibrosis/`, `uq_s16/fibrosis/` |
| `figures` | **Fig. 4** eigenvalue spectra | `uq/eigenvalues.pdf` |
| `figures` | **Fig. 5** truth, MAP, posterior samples | `uq/posterior_samples_<case>.pdf` |
| `figures` | **Fig. 6** regional stiffness | `uq/fibrosis/qoi_fibrosis.pdf` |
| `figures` | **Fig. 7** truth, P(C > 4 kPa), std reduction | `uq/fibrosis/uq_regions_fibrosis.pdf` |
| `figures` | **Fig. 8** 16 measurement points | `fig_scarce_s16.pdf` |
| `figures` | **Table 4** 64 / 32 / 16 points | `table_sensors.tex` |
| `figures` | **Fig. 3** with the colour range of Fig. 5 (needs Step 3) | `fig_maps.pdf` |
