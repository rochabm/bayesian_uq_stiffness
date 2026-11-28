"""
ex07_uq_lv_patient.py
---------------------
Case 3: Bayesian UQ (Laplace approximation) on the patient-specific LV mesh
(Patient 7, real fibrosis), same pipeline as ex07_uq.py (Cases 1-2):

  1. MAP point (default: loaded from ex06_map_lv_patient, --map-from), or
     solved here by inexact Newton-CG if no MAP file is given
  2. generalized eigenproblem  H_misfit v = lambda H_prior v  at the MAP
     (randomized double pass, k + p Hessian applies, twice)
  3. Laplace approximation  Gamma_post ~ Gamma_prior - V diag(lambda/(1+lambda)) V^T
     -> pointwise posterior std, variance and std reduction, P(C > C_th)
  4. posterior samples (fields) and regional quantities of interest from
     --n-qoi-samples samples: geometric-mean stiffness of fibrosis / healthy
     tissue and their ratio, with 95% credible intervals

Data, forward model, measurement points and prior as ex06_map_lv_patient.py
(same defaults: p_endo = 10 kPa, gamma = 1.254, delta = 1.429e-2).
Regions from the tissue labels (P1 nodes): fibrosis = all adjacent cells
fibrotic, border = mixed, healthy = all healthy.

Large mesh: the exact prior variance diag(Gamma_prior) is computed in blocks
with a sparse LU of R (no dense n x n matrix).

Usage:
    python ex07_uq_lv_patient.py                         # MAP from out_ex06_lv_patient
    python ex07_uq_lv_patient.py --k-eig 300             # quicker first test
    python ex07_uq_lv_patient.py --map-from none         # solve the MAP here

OUTPUTS in <output-dir>/ :
    uq_lv_patient.xdmf/.h5   truth, MAP, prior/posterior std, variance and std
                             reduction, P(C > C_th), credible z, regions, tissue,
                             eigenvectors, prior/posterior samples
    uq_lv_patient.npz        eigenvalues, eigenvectors, variances, MAP, truth
    eigenvalues_lv_patient.txt/.png
    regions_lv_patient.json, table_regions_lv_patient.tex   regional summary
    qoi_lv_patient.json, qoi_lv_patient.tex                 regional stiffness
    metrics_uq_lv_patient.json
Serial only.
"""

from mpi4py import MPI
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import petsc4py.PETSc as PETSc
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.linalg import solve_triangular
from scipy.spatial import cKDTree
from scipy.special import ndtr

import ufl
import dolfinx
import dolfinx.fem.petsc
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile

from ex03_lv_vtu_forward import (load_lv_geometry, prob_ventricle_passive_filling,
                                 stiffness_field)
from lv_geometry import FIBROSIS_LABELS
from cardiac_utils import select_distributed_nodes, save_selected_nodes
from hessian_ucq import HessianOperator
from newton_cg_solver import inexact_newton_cg

if MPI.COMM_WORLD.size > 1:
    raise RuntimeError("ex07_uq_lv_patient is serial-only.")

TAG = "lv_patient"

# =============================================================================
# command-line arguments
# =============================================================================

parser = argparse.ArgumentParser(description="ex07 (Case 3, patient LV) -- Laplace UQ")
parser.add_argument("--vtu", default="Patient_7_lv_tissue.vtu")
parser.add_argument("--fiber-vtu", default=None)
parser.add_argument("--c-healthy", type=float, default=2.0)
parser.add_argument("--fibrosis-factor", type=float, default=3.0)
parser.add_argument("--truth-space", choices=["P1", "DG0"], default="P1")
parser.add_argument("--num-nodes", type=int, default=64)
parser.add_argument("--min-dist", type=float, default=4.0)
# prior (Case 3 calibration)
parser.add_argument("--gamma", type=float, default=1.254)
parser.add_argument("--delta", type=float, default=1.429e-2)
parser.add_argument("--order", type=int, default=2, choices=[2, 4],
                    help="Prior covariance A^-order: 2 = BiLaplacian (Matern nu=1/2 in 3D), "
                         "4 = smoother (nu=5/2); gamma/delta from prior_calibration.py --order 4")
parser.add_argument("--prior-fiber-ratio", type=float, default=1.0)
parser.add_argument("--C0", type=float, default=2.0)
# noise / likelihood (as ex06_map_lv_patient)
parser.add_argument("--noise-std", type=float, default=1e-3)
parser.add_argument("--data-noise", type=float, default=0.0)
parser.add_argument("--data-noise-seed", type=int, default=1234)
parser.add_argument("--match-noise-std", action="store_true")
# forward
parser.add_argument("--load", type=float, default=-10.0)
parser.add_argument("--kappa", type=float, default=1e2)
parser.add_argument("--load-steps", type=int, default=10)
parser.add_argument("--newton-maxit", type=int, default=30)
parser.add_argument("--min-dlam", type=float, default=1e-4)
# MAP
parser.add_argument("--map-from", default="out_ex06_lv_patient/m_map_lv_patient.npy",
                    help="MAP from ex06_map_lv_patient ('none' = solve it here)")
parser.add_argument("--gtol", type=float, default=1e-6)
parser.add_argument("--ftol", type=float, default=1e-12)
parser.add_argument("--maxiter", type=int, default=100)
parser.add_argument("--cg-maxiter", type=int, default=200)
parser.add_argument("--cg-tol", type=float, default=0.1)
parser.add_argument("--C-bounds", type=float, nargs=2, default=[1.0, 16.0])
parser.add_argument("--C-init-range", type=float, nargs=2, default=[2.0, 10.0])
parser.add_argument("--init-seed", type=int, default=57)
# eigenproblem / UQ
parser.add_argument("--k-eig", type=int, default=550,
                    help="Eigenpairs (rank of the data <= ~8 x num-nodes = 512 for 64 points)")
parser.add_argument("--oversampling", type=int, default=20)
parser.add_argument("--eig-seed", type=int, default=0)
parser.add_argument("--check-eig", type=int, default=5)
parser.add_argument("--eigvec-ids", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
parser.add_argument("--n-samples", type=int, default=4, help="Sample fields written to XDMF")
parser.add_argument("--sample-seed", type=int, default=99)
parser.add_argument("--n-qoi-samples", type=int, default=1000,
                    help="Samples for the regional stiffness intervals (0 = skip)")
parser.add_argument("--qoi-seed", type=int, default=2024)
parser.add_argument("--c-th", type=float, default=4.0, help="Threshold for P(C > C_th) [kPa]")
parser.add_argument("--var-block", type=int, default=256,
                    help="Right-hand sides per block for the exact prior variance")
parser.add_argument("--verbose-forward", action="store_true")
parser.add_argument("--output-dir", default="out_ex07_lv_patient")
args = parser.parse_args()

if args.map_from and args.map_from.lower() != "none" and not Path(args.map_from).exists():
    found = sorted(Path(".").rglob("m_map_lv_patient*.npy"))
    raise SystemExit(f"MAP file not found: {args.map_from}\n"
                     + (("  found: " + ", ".join(map(str, found)) + "\n  -> pass one with --map-from")
                        if found else "  no m_map_lv_patient*.npy under this folder; "
                        "run ex06_map_lv_patient.py first, or use --map-from none"))

OUT = Path(args.output_dir)
OUT.mkdir(parents=True, exist_ok=True)
LO, HI = np.log(args.C_bounds[0]), np.log(args.C_bounds[1])
M0 = float(np.log(args.C0))


# =============================================================================
# BiLaplacian prior (as ex07; exact variance by blocked sparse LU)
# =============================================================================

class BiLaplacianPrior:
    """Gamma_prior = (R^-1 M_L)^(order-1) R^-1; order 2 = BiLaplacian, order 4 = smoother."""

    def __init__(self, Va, gamma, delta, dx, domain, theta=None, order=2):
        self.order = int(order)
        self.ndofs = Va.dofmap.index_map.size_local * Va.dofmap.index_map_bs
        self.gamma, self.delta = float(gamma), float(delta)
        self.beta = np.sqrt(self.gamma * self.delta) / 1.42
        q, dq = ufl.TrialFunction(Va), ufl.TestFunction(Va)
        ds_all = ufl.Measure("ds", domain=domain)
        grad_q = ufl.grad(q) if theta is None else theta * ufl.grad(q)
        R_form = (self.gamma * ufl.inner(grad_q, ufl.grad(dq)) * dx
                  + self.delta * ufl.inner(q, dq) * dx
                  + self.beta * ufl.inner(q, dq) * ds_all)
        self._R = fem.petsc.assemble_matrix(fem.form(R_form))
        self._R.assemble()
        M = fem.petsc.assemble_matrix(fem.form(ufl.inner(q, dq) * dx))
        M.assemble()
        self._x = self._R.createVecRight()
        self._b = self._R.createVecRight()
        self._b.array[:] = 1.0
        M.mult(self._b, self._x)
        self.M_lump = np.maximum(self._x.array.copy(), 1e-30)
        M.destroy()
        # scipy copy of R for blocked solves (variance, many samples)
        ai, aj, av = self._R.getValuesCSR()
        self._lu = sla.splu(sp.csr_matrix((av, aj, ai), shape=(self.ndofs,) * 2).tocsc())

    def _R_mult(self, v):
        self._b.array[:] = v
        self._R.mult(self._b, self._x)
        return self._x.array.copy()

    def _R_solve(self, v):
        return self._lu.solve(np.asarray(v, dtype=float))

    def mult(self, v):
        """H_prior v = R (M_L^{-1} R)^(order-1) v."""
        y = self._R_mult(v)
        for _ in range(self.order - 1):
            y = self._R_mult(y / self.M_lump)
        return y

    def solve(self, v):
        """Gamma_prior v = (R^{-1} M_L)^(order-1) R^{-1} v."""
        y = self._R_solve(v)
        for _ in range(self.order - 1):
            y = self._R_solve(self.M_lump * y)
        return y

    def sample(self, rng):
        """z = (R^-1 M_L)^(order/2-1) R^-1 M_L^{1/2} w,  w ~ N(0, I)."""
        y = self._R_solve(np.sqrt(self.M_lump) * rng.standard_normal(self.ndofs))
        for _ in range(self.order // 2 - 1):
            y = self._R_solve(self.M_lump * y)
        return y

    def pointwise_variance(self, block=256):
        """Exact diag(Gamma_prior)_i = y_i^T M_L y_i, y_i = R^{-1} (M_L R^{-1})^(order/2-1) e_i."""
        n = self.ndofs
        d = np.empty(n)
        for s0 in range(0, n, block):
            idx = np.arange(s0, min(s0 + block, n))
            E = np.zeros((n, len(idx)))
            E[idx, np.arange(len(idx))] = 1.0
            Y = self._lu.solve(E)
            for _ in range(self.order // 2 - 1):
                Y = self._lu.solve(self.M_lump[:, None] * Y)
            d[idx] = np.einsum("ij,ij->j", Y, self.M_lump[:, None] * Y)
        return d


# =============================================================================
# geometry, forward model, data, likelihood (as ex06_map_lv_patient)
# =============================================================================

geo = load_lv_geometry(args.vtu, fiber_source="vtu" if args.fiber_vtu else "rule",
                       fiber_vtu=args.fiber_vtu, alpha_endo=40.0, alpha_epi=-50.0)
domain = geo.mesh
dim = domain.geometry.dim
BASE, ENDO = geo.markers["BASE"][0], geo.markers["ENDO"][0]
facet_tags = geo.ffun
V = fem.functionspace(domain, ("Lagrange", 1, (dim,)))
Va = fem.functionspace(domain, ("Lagrange", 1))
du, v = ufl.TrialFunction(V), ufl.TestFunction(V)
uh = fem.Function(V)
base_dofs = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(BASE))
bcs = [fem.dirichletbc(np.zeros(dim, dtype=default_scalar_type), base_dofs, V)]

dof_coords = V.tabulate_dof_coordinates().reshape(-1, dim)
selected_dofs, selected_coords = select_distributed_nodes(
    dof_coords, N=args.num_nodes, min_dist=args.min_dist, seed=42)
selected_dofs = np.asarray(selected_dofs, dtype=np.int64)
Npoints = len(selected_dofs)
save_selected_nodes(selected_coords, selected_dofs)
if not np.allclose(dof_coords[selected_dofs], domain.geometry.x[selected_dofs, :dim]):
    raise RuntimeError("P1 dof ordering differs from the geometry node ordering.")

I = ufl.variable(ufl.Identity(dim))
F = ufl.variable(I + ufl.grad(uh))
J = ufl.variable(ufl.det(F))
e1, e2, e3 = geo.f0, geo.s0, geo.n0
bf, bt, bfs = default_scalar_type(6.6), default_scalar_type(4.0), default_scalar_type(2.6)
kappa = fem.Constant(domain, float(args.kappa))
m_fun = fem.Function(Va, name="m")
Es = 0.5 * (J ** (-2 / 3) * F.T * F - I)
E = [[ufl.inner(Es * a, b) for b in (e1, e2, e3)] for a in (e1, e2, e3)]
Q = (bf * E[0][0] ** 2
     + bt * (E[1][1] ** 2 + E[2][2] ** 2 + E[1][2] ** 2 + E[2][1] ** 2)
     + bfs * (E[0][1] ** 2 + E[1][0] ** 2 + E[0][2] ** 2 + E[2][0] ** 2))
strain_energy = ufl.exp(m_fun) / 2.0 * (ufl.exp(Q) - 1) + kappa * (J * ufl.ln(J) - J + 1)
P = ufl.diff(strain_energy, F)
p_endo = fem.Constant(domain, 0.0)
metadata = {"quadrature_degree": 4}
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags, metadata=metadata)
dx = ufl.Measure("dx", domain=domain, metadata=metadata)
N_fac = ufl.FacetNormal(domain)
Fun = (ufl.inner(P, ufl.grad(v)) * dx
       - p_endo * ufl.inner(v, J * ufl.transpose(ufl.inv(F)) * N_fac) * ds(ENDO))


def solve_nl_prob(u_func):
    problem = NonlinearProblem(Fun, u_func, bcs)
    solver = NewtonSolver(domain.comm, problem)
    solver.atol = 1e-8
    solver.rtol = 1e-8
    solver.max_it = args.newton_maxit
    ksp = solver.krylov_solver
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    lam = 0.0
    for lam_target in np.linspace(0.0, 1.0, args.load_steps + 1)[1:]:
        dlam = lam_target - lam
        while lam < lam_target - 1e-12:
            dlam = min(dlam, lam_target - lam)
            u_prev = u_func.x.array.copy()
            p_endo.value = (lam + dlam) * args.load
            try:
                its, converged = solver.solve(u_func)
            except RuntimeError:
                converged = False
            if converged:
                lam += dlam
            else:
                u_func.x.array[:] = u_prev
                u_func.x.scatter_forward()
                dlam *= 0.5
                if dlam < args.min_dlam:
                    raise RuntimeError(f"forward Newton failed at load factor {lam:.4f}")


node_to_cells = {}
for cell_id in range(domain.topology.index_map(domain.topology.dim).size_local):
    for node in domain.geometry.dofmap[cell_id]:
        node_to_cells.setdefault(int(node), []).append(cell_id)
sensor_cells = [node_to_cells[int(d)][0] for d in selected_dofs]
if len(set(sensor_cells)) != len(sensor_cells):
    raise RuntimeError("some measurement points share a cell; reduce --num-nodes.")

TT = fem.functionspace(domain, ("DG", 0, (dim, dim)))
W0 = fem.functionspace(domain, ("DG", 0))
Fd_func = fem.Function(TT, name="Fd_observed")
bs = TT.dofmap.bs
cell_vol = fem.Function(W0)
fem.assemble_vector(cell_vol.x.array, fem.form(ufl.TestFunction(W0) * dx))
sensor_weight = fem.Function(W0)
for cell in sensor_cells:
    c = W0.dofmap.cell_dofs(cell)
    sensor_weight.x.array[c] = 1.0 / cell_vol.x.array[c]
sensor_weight.x.scatter_forward()
noise_prec = fem.Constant(domain, 1.0)
Phi = 0.5 * noise_prec * sensor_weight * ufl.inner(F - Fd_func, F - Fd_func) * dx
Phi_c = fem.form(Phi)

if args.prior_fiber_ratio != 1.0:
    _f0n = geo.f0 / ufl.sqrt(ufl.inner(geo.f0, geo.f0))
    Theta = ufl.Identity(dim) + (args.prior_fiber_ratio ** 2 - 1.0) * ufl.outer(_f0n, _f0n)
else:
    Theta = None
prior = BiLaplacianPrior(Va, args.gamma, args.delta, dx, domain, theta=Theta,
                         order=args.order)
n = prior.ndofs
m0_vec = np.full(n, M0)

lu_opts = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
lmbda = fem.Function(V)
adj_problem = LinearProblem(ufl.adjoint(ufl.derivative(Fun, uh, du)),
                            -ufl.derivative(Phi, uh, v), bcs=bcs, petsc_options=lu_opts)
dFdm_c = fem.form(ufl.action(ufl.adjoint(ufl.derivative(Fun, m_fun, ufl.TrialFunction(Va))),
                             lmbda))
grad_misfit = fem.Function(Va)


class MAPObjective:
    def __init__(self):
        self.x, self.n_forward, self.n_hessvec = None, 0, 0

    def _update(self, x):
        if self.x is not None and np.array_equal(x, self.x):
            return
        self.x = None
        m_fun.x.array[:] = x
        m_fun.x.scatter_forward()
        uh.x.array[:] = 0.0
        solve_nl_prob(uh)
        self.n_forward += 1
        self.Jd = fem.assemble_scalar(Phi_c)
        lam = adj_problem.solve()
        lmbda.x.array[:] = lam.x.array
        lmbda.x.scatter_forward()
        grad_misfit.x.array[:] = 0.0
        fem.assemble_vector(grad_misfit.x.array, dFdm_c)
        dm = x - m0_vec
        Hp_dm = prior.mult(dm)
        self.Jp = 0.5 * float(dm @ Hp_dm)
        self.J = self.Jd + self.Jp
        self.g = grad_misfit.x.array.copy() + Hp_dm
        self.x = x.copy()

    def eval_J(self, x):
        self._update(x)
        return self.J

    def eval_grad(self, x):
        self._update(x)
        return self.g.copy()

    def misfit_hessian(self):
        return HessianOperator(Fun, Phi, uh, m_fun, lmbda, V, Va, facet_tags, domain)

    def build_hop(self):
        Hm, obj = self.misfit_hessian(), self

        class _Full:
            ndofs_m = Hm.ndofs_m

            def mult(self, x):
                obj.n_hessvec += 1
                return Hm.mult(x) + prior.mult(x)
        return _Full()


obj = MAPObjective()


# =============================================================================
# regions from the tissue labels (as ex06_map_lv_patient)
# =============================================================================

data = geo.data
is_fib = np.isin(data.tissue, FIBROSIS_LABELS).astype(float)
npts = len(data.points)
frac_fib = (np.bincount(data.cells.ravel(), weights=np.repeat(is_fib, 4), minlength=npts)
            / np.maximum(np.bincount(data.cells.ravel(), minlength=npts), 1))
dist, k_pt = cKDTree(data.points).query(Va.tabulate_dof_coordinates()[:, :3])
assert dist.max() < 1e-6 * np.ptp(data.points, axis=0).max(), "dof/vtu node matching failed"
frac_dof = frac_fib[k_pt]
regions = {"fibrosis": frac_dof >= 1.0 - 1e-12,
           "border": (frac_dof > 1e-12) & (frac_dof < 1.0 - 1e-12),
           "healthy": frac_dof <= 1e-12,
           "all": np.ones(n, bool)}
REGION_LABEL = {"fibrosis": "Fibrosis", "border": "Border", "healthy": "Healthy",
                "all": "Whole LV"}
region_code = np.select([regions["healthy"], regions["border"], regions["fibrosis"]],
                        [0.0, 1.0, 2.0])
w_node = prior.M_lump                      # lumped volume weights, as the prior


# =============================================================================
# eigensolver (as ex07)
# =============================================================================

def b_orthonormalize(Y, B_mult):
    for _ in range(2):
        BY = np.column_stack([B_mult(Y[:, i]) for i in range(Y.shape[1])])
        G = 0.5 * (Y.T @ BY + (Y.T @ BY).T)
        L = np.linalg.cholesky(G + 1e-14 * np.trace(G) / len(G) * np.eye(len(G)))
        Y = solve_triangular(L, Y.T, lower=True).T
    return Y


def double_pass_generalized(A_mult, B_mult, Binv, n, k, p, rng):
    Omega = rng.standard_normal((n, k + p))
    Y = np.empty_like(Omega)
    t0 = time.perf_counter()
    for i in range(k + p):
        Y[:, i] = Binv(A_mult(Omega[:, i]))
        if (i + 1) % 20 == 0:
            el = time.perf_counter() - t0
            print(f"    pass 1: {i + 1}/{k + p}  ({el:.0f} s, ~{el / (i + 1) * 2 * (k + p) - el:.0f} s left)")
    Qm = b_orthonormalize(Y, B_mult)
    AQ = np.empty_like(Qm)
    for i in range(k + p):
        AQ[:, i] = A_mult(Qm[:, i])
        if (i + 1) % 20 == 0:
            print(f"    pass 2: {i + 1}/{k + p}")
    T = Qm.T @ AQ
    T = 0.5 * (T + T.T)
    lam, W = np.linalg.eigh(T)
    order = np.argsort(lam)[::-1][:k]
    return lam[order], Qm @ W[:, order]


def va_function(name, arr):
    f = fem.Function(Va, name=name)
    f.x.array[:] = arr
    f.x.scatter_forward()
    return f


def wmean(x, mask):
    w = w_node[mask]
    return float(w @ x[mask] / w.sum())


# =============================================================================
# 1. data + MAP
# =============================================================================

t_all = time.perf_counter()
print("\n" + "=" * 70)
print(f" ex07 UQ  |  Case 3, patient LV ({args.vtu}), {n} parameter dofs")
print(f"  prior: gamma={prior.gamma:.4g}, delta={prior.delta:.4g}, beta={prior.beta:.4g}, "
      f"m0=log({args.C0}), order {prior.order};  p_endo={-args.load:g} kPa;  {Npoints} measurement points")
print(f"  regions (P1 nodes): " + ", ".join(f"{k} {int(v.sum())}" for k, v in regions.items()
                                          if k != "all"))
print("=" * 70)

os.environ["OUTPUT_DIR"] = str(OUT / "forward")
ud_arr, Fd_arr, CC = prob_ventricle_passive_filling(
    geo, ndofs_data=args.num_nodes, c_healthy=args.c_healthy,
    fibrosis_factor=args.fibrosis_factor, target_load=args.load, kappa_bulk=args.kappa,
    param_space=args.truth_space, load_steps=args.load_steps, min_dist=args.min_dist,
    compute_stress_output=False)
Fd_clean = np.asarray(Fd_arr, dtype=float)
if Fd_clean.shape[0] != Npoints:
    raise RuntimeError("ex03 and ex07 selected a different number of measurement points.")
C_true = (CC if args.truth_space == "P1"
          else stiffness_field(geo, args.c_healthy, args.fibrosis_factor, space="P1")).x.array.copy()
m_true = np.log(C_true)

F_scale = float(np.sqrt(np.mean((Fd_clean - np.eye(dim)) ** 2)))
sigma_data = args.data_noise * F_scale
eps = np.random.default_rng(args.data_noise_seed).standard_normal(Fd_clean.shape)
Fd_obs = Fd_clean + sigma_data * eps
sigma_lik = sigma_data if (args.match_noise_std and sigma_data > 0) else args.noise_std
noise_prec.value = 1.0 / sigma_lik ** 2
Fd_func.x.array[:] = 0.0
for j, cell in enumerate(sensor_cells):
    c = TT.dofmap.cell_dofs(cell)[0]
    Fd_func.x.array[c * bs:(c + 1) * bs] = Fd_obs[j].flatten()
Fd_func.x.scatter_forward()
print(f"  data noise {100 * args.data_noise:g}% -> std {sigma_data:.3e}; likelihood std "
      f"{sigma_lik:.3e}; N_obs = {Fd_obs.size}")

if args.map_from and args.map_from.lower() != "none":
    m_map = np.load(args.map_from)
    if m_map.shape != (n,):
        raise RuntimeError(f"--map-from has shape {m_map.shape}, expected ({n},)")
    print(f"  MAP loaded from {args.map_from}")
    map_info = dict(source=args.map_from)
else:
    print("\n--- MAP by inexact Newton-CG ---")
    x0 = np.clip(np.log(np.random.RandomState(args.init_seed).uniform(*args.C_init_range, n)),
                 LO, HI)
    t0 = time.perf_counter()
    m_map, nhist = inexact_newton_cg(
        obj.eval_J, obj.eval_grad, obj.build_hop, x0=x0,
        max_outer_iter=args.maxiter, grad_tol=args.gtol, ftol=args.ftol,
        cg_tol=args.cg_tol, cg_maxiter=args.cg_maxiter,
        bounds=(LO, HI), eisenstat_walker=True, verbose=False)
    map_info = dict(source="newtoncg", wall_time_s=time.perf_counter() - t0,
                    n_forward=obj.n_forward, n_hessvec=obj.n_hessvec,
                    stop_reason=nhist[-1].get("stopped_reason", "max_outer_iter"))
np.save(OUT / f"m_map_{TAG}.npy", m_map)

J_map = obj.eval_J(m_map)               # sets uh, m_fun, lmbda at the MAP
g_map = obj.eval_grad(m_map)
n_bound = int(np.sum((m_map <= LO + 1e-10) | (m_map >= HI - 1e-10)))
print(f"  J(MAP) = {J_map:.4e} (Phi = {obj.Jd:.4e}), ||g||_inf = {np.abs(g_map).max():.2e}, "
      f"nodes at bounds: {n_bound}")
if np.abs(g_map).max() > 1e-4:
    print("  WARNING: gradient at the loaded MAP is not small -- is it the MAP of THIS problem "
          "(same prior, load, measurement points, noise)?")

# =============================================================================
# 2. eigenproblem at the MAP
# =============================================================================

print(f"\n--- generalized eigenproblem (k = {args.k_eig}, p = {args.oversampling}) ---")
Hm = obj.misfit_hessian()
n_hv = [0]


def A_mult(x):
    n_hv[0] += 1
    return Hm.mult(x)


t0 = time.perf_counter()
k = min(args.k_eig, n - args.oversampling)
lam, U = double_pass_generalized(A_mult, prior.mult, prior.solve, n, k, args.oversampling,
                                 np.random.default_rng(args.eig_seed))
t_eig = time.perf_counter() - t0
n_neg = int(np.sum(lam < 0))
lam = np.maximum(lam, 0.0)
print(f"  done in {t_eig:.0f} s, {n_hv[0]} Hessian applies;  lambda_1 = {lam[0]:.3e}, "
      f"lambda_k = {lam[-1]:.3e},  # lambda > 1: {int(np.sum(lam > 1))}")
if lam[-1] > 0.1:
    print(f"  WARNING: lambda_k = {lam[-1]:.2f} > 0.1 -- increase --k-eig.")
eig_check = []
for i in range(min(args.check_eig, k)):
    Au, Bu = A_mult(U[:, i]), prior.mult(U[:, i])
    eig_check.append(float(np.linalg.norm(Au - lam[i] * Bu)
                           / max(np.linalg.norm(lam[i] * Bu), 1e-300)))
print("  eigen-residuals (first 5): " + ", ".join(f"{r:.1e}" for r in eig_check))
np.savetxt(OUT / f"eigenvalues_{TAG}.txt", lam, header="generalized eigenvalues, descending")

fig, ax = plt.subplots(figsize=(4.0, 3.2))
pos = lam > 0
ax.semilogy(np.arange(1, len(lam) + 1)[pos], lam[pos], "-", lw=1.0)
ax.axhline(1.0, color="k", lw=0.8, ls="--")
ax.set(xlabel="index $i$", ylabel=r"$\lambda_i$", title="Case 3 (patient LV)")
fig.tight_layout()
fig.savefig(OUT / f"eigenvalues_{TAG}.png", dpi=300)
plt.close(fig)

# =============================================================================
# 3. Laplace approximation: pointwise quantities
# =============================================================================

print("\n--- posterior (Laplace approximation) ---")
t0 = time.perf_counter()
prior_var = prior.pointwise_variance(args.var_block)
print(f"  exact prior variance ({n} dofs) in {time.perf_counter() - t0:.0f} s")
post_var = np.maximum(prior_var - (U ** 2) @ (lam / (1.0 + lam)), 0.0)
prior_std, post_std = np.sqrt(prior_var), np.sqrt(post_var)
var_red = 1.0 - post_var / prior_var
std_red = 1.0 - post_std / prior_std
z_post = np.abs(m_true - m_map) / np.maximum(post_std, 1e-12)
z_prior = np.abs(m_true - M0) / prior_std
prob = ndtr((m_map - np.log(args.c_th)) / np.maximum(post_std, 1e-12))
C_map = np.exp(m_map)

stats = {}
for name, mask in regions.items():
    if not mask.any():
        continue
    dC = C_map[mask] - C_true[mask]
    w = w_node[mask]
    stats[name] = dict(
        n_nodes=int(mask.sum()),
        C_rel_L2=float(np.sqrt(np.sum(w * dC ** 2) / np.sum(w * C_true[mask] ** 2))),
        prior_std=wmean(prior_std, mask), post_std=wmean(post_std, mask),
        var_reduction=wmean(var_red, mask), std_reduction=wmean(std_red, mask),
        coverage95=wmean((z_post <= 1.96).astype(float), mask),
        coverage95_prior=wmean((z_prior <= 1.96).astype(float), mask),
        mean_P_exceed=wmean(prob, mask))
print(f"  {'region':<10s} {'nodes':>6s} {'eL2[%]':>7s} {'s_prior':>8s} {'s_post':>7s} "
      f"{'VR':>6s} {'SR':>6s} {'cov[%]':>7s} {'covPr[%]':>8s} {'P(C>th)':>8s}")
for name, s in stats.items():
    print(f"  {REGION_LABEL[name]:<10s} {s['n_nodes']:6d} {100 * s['C_rel_L2']:7.2f} "
          f"{s['prior_std']:8.3f} {s['post_std']:7.3f} {s['var_reduction']:6.3f} "
          f"{s['std_reduction']:6.3f} {100 * s['coverage95']:7.1f} "
          f"{100 * s['coverage95_prior']:8.1f} {s['mean_P_exceed']:8.3f}")
print(f"  (volume-weighted; SR = std reduction 1 - s_post/s_prior; P with C_th = {args.c_th:g} kPa)")

rows = []
for name in ["fibrosis", "border", "healthy", "all"]:
    if name in stats:
        s = stats[name]
        if name == "all":
            rows.append(r"\midrule")
        rows.append(f"{REGION_LABEL[name]:<9s} & ${100 * s['C_rel_L2']:.2f}$ & "
                    f"${s['post_std']:.3f}$ & ${s['std_reduction']:.3f}$ & "
                    f"${100 * s['coverage95']:.1f}$ \\\\")
(OUT / f"table_regions_{TAG}.tex").write_text(
    "% Regional summary, Case 3 -- requires \\usepackage{booktabs}\n"
    "\\begin{tabular}{lcccc}\n\\toprule\n"
    "Region & $e_C^{L^2}$ [\\%] & $\\bar s_{\\mathrm{post}}$ & Std.\\ red. & Coverage [\\%] \\\\\n"
    "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
(OUT / f"regions_{TAG}.json").write_text(json.dumps(dict(c_th=args.c_th, stats=stats), indent=2))

# =============================================================================
# 4. samples: fields + regional quantities of interest
# =============================================================================

dp = 1.0 - 1.0 / np.sqrt(1.0 + lam)


def posterior_from(z):
    return m_map + z - U @ (dp * (U.T @ prior.mult(z)))


rng_s = np.random.default_rng(args.sample_seed)
prior_samples, post_samples = [], []
for s in range(args.n_samples):
    z = prior.sample(rng_s)
    prior_samples.append(m0_vec + z)
    post_samples.append(posterior_from(z))

qoi = None
fib, hea = regions["fibrosis"], regions["healthy"]


def geo_means(m_rows):
    m_rows = np.atleast_2d(m_rows)
    qf = np.exp(m_rows[:, fib] @ w_node[fib] / w_node[fib].sum())
    qh = np.exp(m_rows[:, hea] @ w_node[hea] / w_node[hea].sum())
    return {"fibrosis": qf, "healthy": qh, "ratio": qf / qh}


if args.n_qoi_samples > 0 and fib.any() and hea.any():
    print(f"\n--- regional geometric-mean stiffness ({args.n_qoi_samples} samples) ---")
    rng_q = np.random.default_rng(args.qoi_seed)
    q_post = {k_: [] for k_ in ("fibrosis", "healthy", "ratio")}
    q_prior = {k_: [] for k_ in ("fibrosis", "healthy", "ratio")}
    for _ in range(args.n_qoi_samples):
        z = prior.sample(rng_q)
        for k_, val in geo_means(posterior_from(z)).items():
            q_post[k_].append(float(val[0]))
        for k_, val in geo_means(m0_vec + z).items():
            q_prior[k_].append(float(val[0]))
    q_true = {k_: float(v_[0]) for k_, v_ in geo_means(m_true).items()}
    q_map = {k_: float(v_[0]) for k_, v_ in geo_means(m_map).items()}
    qoi = {}
    print(f"  {'QoI':<9s} {'truth':>7s} {'MAP':>7s} {'mean':>7s} {'post 95% CI':>18s} "
          f"{'prior 95% CI':>18s} {'in CI':>6s}")
    for k_ in ("fibrosis", "healthy", "ratio"):
        a_ = np.array(q_post[k_])
        b_ = np.array(q_prior[k_])
        lo_, hi_ = np.percentile(a_, [2.5, 97.5])
        plo, phi = np.percentile(b_, [2.5, 97.5])
        inside = bool(lo_ <= q_true[k_] <= hi_)
        qoi[k_] = dict(truth=q_true[k_], map=q_map[k_], mean=float(a_.mean()),
                       q025=float(lo_), q975=float(hi_), prior_q025=float(plo),
                       prior_q975=float(phi), truth_in_95CI=inside)
        print(f"  {k_:<9s} {q_true[k_]:7.3f} {q_map[k_]:7.3f} {a_.mean():7.3f} "
              f"[{lo_:6.3f}, {hi_:6.3f}] [{plo:6.3f}, {phi:6.3f}] {'yes' if inside else 'NO':>6s}")
    (OUT / f"qoi_{TAG}.json").write_text(json.dumps(dict(n_samples=args.n_qoi_samples,
                                                         qoi=qoi), indent=2))
    tex = {"fibrosis": r"\bar C_{\mathrm{fib}}", "healthy": r"\bar C_{\mathrm{healthy}}",
           "ratio": r"\bar C_{\mathrm{fib}}/\bar C_{\mathrm{healthy}}"}
    (OUT / f"qoi_{TAG}.tex").write_text("\n".join(
        f"${tex[k_]} = {q['mean']:.2f}\\;[{q['q025']:.2f},\\,{q['q975']:.2f}]$ "
        f"(truth ${q['truth']:.2f}$)" for k_, q in qoi.items()) + "\n")

# =============================================================================
# outputs
# =============================================================================

tissue_p1 = va_function("fibrosis_fraction", frac_dof)
fields = [va_function("m_true", m_true), va_function("C_true", C_true),
          va_function("m_map", m_map), va_function("C_map", C_map),
          va_function("C_rel_error", np.abs(C_map - C_true) / C_true),
          va_function("m_prior_std", prior_std), va_function("m_post_std", post_std),
          va_function("variance_reduction", var_red), va_function("std_reduction", std_red),
          va_function("prob_C_above_threshold", prob), va_function("credible_z", z_post),
          va_function("region", region_code), tissue_p1]
for i in args.eigvec_ids:
    if 1 <= i <= k:
        u_ = U[:, i - 1]
        fields.append(va_function(f"eigvec_{i:03d}", u_ / np.abs(u_).max()))
for s, (mp, mq) in enumerate(zip(prior_samples, post_samples), start=1):
    fields += [va_function(f"m_prior_sample_{s}", mp), va_function(f"C_prior_sample_{s}", np.exp(mp)),
               va_function(f"m_post_sample_{s}", mq), va_function(f"C_post_sample_{s}", np.exp(mq))]
with XDMFFile(MPI.COMM_WORLD, str(OUT / f"uq_{TAG}.xdmf"), "w") as xf:
    xf.write_mesh(domain)
    for f_ in fields:
        xf.write_function(f_)
    xf.write_function(geo.tissue)

np.savez(OUT / f"uq_{TAG}.npz", eigvals=lam, eigvecs=U, prior_var=prior_var,
         post_var=post_var, m_map=m_map, m_true=m_true, region_code=region_code,
         w_node=w_node, sensor_xyz=np.asarray(selected_coords, float),
         prior_samples=np.array(prior_samples),
         post_samples=np.array(post_samples))

metrics = dict(
    case=TAG, settings=vars(args),
    prior=dict(gamma=prior.gamma, delta=prior.delta, beta=prior.beta, m0=M0,
               order=prior.order),
    data=dict(sigma_data=sigma_data, sigma_lik=sigma_lik, n_obs=int(Fd_obs.size),
              n_measurement_points=Npoints),
    map=map_info | dict(J=float(J_map), Phi=float(obj.Jd), grad_inf=float(np.abs(g_map).max()),
                        n_at_bounds=n_bound),
    eig=dict(k=int(k), p=args.oversampling, n_hessian_applies=n_hv[0], wall_time_s=t_eig,
             lambda_first=float(lam[0]), lambda_last=float(lam[-1]),
             n_gt_1=int(np.sum(lam > 1)), n_gt_0p1=int(np.sum(lam > 0.1)),
             eff_dof=float(np.sum(lam / (1 + lam))), n_negative_clipped=n_neg,
             residuals=eig_check),
    regions=stats, qoi=qoi,
    total_wall_time_s=time.perf_counter() - t_all)
with open(OUT / f"metrics_uq_{TAG}.json", "w") as f_:
    json.dump(metrics, f_, indent=2, default=float)

print(f"\nDone in {metrics['total_wall_time_s']:.0f} s. Outputs in {OUT.resolve()}")
