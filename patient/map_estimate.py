"""
ex06_map_lv_patient.py
----------------------
Case 3: MAP estimate of the log-stiffness m = log C on the patient-specific LV
mesh (Patient 7) with the real fibrosis region, inexact Newton-CG only.

Same inverse problem as ex06_map_comparison.py (Cases 1-2):
    m_MAP = argmin_m  J(m) = Phi(m) + 0.5 || A (m - m0) ||^2
    Phi(m) = 1/(2 sigma^2) sum_j || F_j(m) - Fd_j ||_F^2      (measurement points j)
    BiLaplacian prior  R = gamma K + delta M + beta M_bndry,  m0 = log C0

What changes with respect to ex06
---------------------------------
- geometry, facet markers and fibres from the .vtu via ex03_lv_vtu_forward
  (load_lv_geometry): 5 = base (fixed), 6 = endocardium (pressure)
- synthetic data from ex03_lv_vtu_forward.prob_ventricle_passive_filling:
  C = c_healthy in healthy tissue, fibrosis_factor * c_healthy in the
  labelled fibrosis (tissue 1, 2)
      --truth-space P1  (default) truth = nodal average of the cell values,
                        i.e. representable by the P1 parameter (as Cases 1-2)
      --truth-space DG0 data from the sharp cell-wise field (no inverse crime
                        in the parameter); errors are measured against the
                        P1 nodal average
- measurement points selected exactly as ex03_lv_vtu_forward does
  (P1 dof coordinates, --min-dist, seed 42), with a consistency check
- forward solve with automatic load-step cutting (as ex03_lv_vtu_forward)
- regions from the tissue labels (not a sphere):
      fibrosis = nodes whose cells are all fibrotic
      border   = nodes touching both tissues
      healthy  = nodes whose cells are all healthy

OUTPUTS in <output-dir>/ :
    map_lv_patient.xdmf/.h5     m and C (true, init, MAP), errors, displacements,
                                tissue labels, regions, measurement indicator
    metrics_lv_patient.json     errors (whole LV and per region) + costs
    history_newtoncg_lv_patient.txt, convergence_lv_patient.png
    m_map_lv_patient.npy        (reusable by ex07_uq_lv_patient --map-from)
    forward/                    ex03 outputs of the data generation

Serial only.

Usage:
    python ex06_map_lv_patient.py
    python ex06_map_lv_patient.py --vtu Patient_7_lv_tissue.vtu --fiber-vtu Patient_7_UVC.vtu
    python ex06_map_lv_patient.py --data-noise 0.01 --match-noise-std
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
from scipy.spatial import cKDTree

import ufl
import dolfinx
import dolfinx.fem.petsc
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile

from ex03_lv_vtu_forward import (load_lv_geometry, prob_ventricle_passive_filling,
                                 stiffness_field)
from lv_geometry import FIBROSIS_LABELS, cell_volumes
from cardiac_utils import select_distributed_nodes, compute_Fh, save_selected_nodes
from hessian_ucq import HessianOperator
from newton_cg_solver import inexact_newton_cg

if MPI.COMM_WORLD.size > 1:
    raise RuntimeError("ex06_map_lv_patient is serial-only.")

TAG = "lv_patient"

# =============================================================================
# command-line arguments
# =============================================================================

parser = argparse.ArgumentParser(description="ex06 (Case 3, patient LV) -- MAP, m = log C, Newton-CG")
# geometry
parser.add_argument("--vtu", default="Patient_7_lv_tissue.vtu")
parser.add_argument("--fiber-vtu", default=None,
                    help="Patient fibres from this .vtu (e.g. Patient_7_UVC.vtu); "
                         "default: rule-based, +40/-50 deg as ex03")
# true field
parser.add_argument("--c-healthy", type=float, default=2.0, help="C in healthy tissue [kPa]")
parser.add_argument("--fibrosis-factor", type=float, default=3.0,
                    help="C in fibrosis = factor * c_healthy (default 3 -> 6 kPa)")
parser.add_argument("--truth-space", choices=["P1", "DG0"], default="P1",
                    help="Field used to generate the data (default P1, as Cases 1-2)")
# measurement points
parser.add_argument("--num-nodes", type=int, default=64)
parser.add_argument("--min-dist", type=float, default=4.0,
                    help="Minimum distance between measurement points (as ex03_lv)")
# prior (defaults: Case 2 calibration -- check against the lesion size printed below)
parser.add_argument("--gamma", type=float, default=1.457)
parser.add_argument("--delta", type=float, default=1.441e-2)
parser.add_argument("--order", type=int, default=2, choices=[2, 4],
                    help="Prior covariance A^-order: 2 = BiLaplacian (Matern nu=1/2 in 3D), "
                         "4 = smoother (nu=5/2); gamma/delta from prior_calibration.py --order 4")
parser.add_argument("--prior-fiber-ratio", type=float, default=1.0)
parser.add_argument("--C0", type=float, default=2.0, help="Prior mean in kPa (m0 = log C0)")
# likelihood / data noise (as ex06)
parser.add_argument("--noise-std", type=float, default=1e-3)
parser.add_argument("--data-noise", type=float, default=0.0)
parser.add_argument("--data-noise-seed", type=int, default=1234)
parser.add_argument("--match-noise-std", action="store_true")
# forward problem (defaults of ex03_lv_vtu_forward)
parser.add_argument("--load", type=float, default=-3.0, help="Endocardial pressure [kPa]")
parser.add_argument("--kappa", type=float, default=1e2)
parser.add_argument("--load-steps", type=int, default=10)
parser.add_argument("--newton-maxit", type=int, default=30)
parser.add_argument("--min-dlam", type=float, default=1e-4,
                    help="Smallest load-factor step before the forward solve gives up")
# optimization (Newton-CG)
parser.add_argument("--gtol", type=float, default=1e-6)
parser.add_argument("--ftol", type=float, default=1e-12)
parser.add_argument("--maxiter", type=int, default=100)
parser.add_argument("--cg-maxiter", type=int, default=200)
parser.add_argument("--cg-tol", type=float, default=0.1)
parser.add_argument("--C-bounds", type=float, nargs=2, default=[1.0, 16.0])
parser.add_argument("--C-init-range", type=float, nargs=2, default=[2.0, 10.0])
parser.add_argument("--init-seed", type=int, default=57)
parser.add_argument("--no-prior-std", action="store_true",
                    help="Skip the exact prior pointwise std (n back-solves, dense n x n)")
parser.add_argument("--stress", action="store_true",
                    help="Also write the stress post-processing of the data generation")
parser.add_argument("--verbose-forward", action="store_true")
parser.add_argument("--output-dir", type=str, default="out_ex06_lv_patient")
args = parser.parse_args()

OUT = Path(args.output_dir)
OUT.mkdir(parents=True, exist_ok=True)
LO, HI = np.log(args.C_bounds[0]), np.log(args.C_bounds[1])
M0 = float(np.log(args.C0))


# =============================================================================
# BiLaplacian prior on m (identical to ex06)
# =============================================================================

class BiLaplacianPrior:
    """H_prior = R (M_L^{-1} R)^(order-1),  R = gamma K_Theta + delta M + beta M_bndry.
    order 2: BiLaplacian (Gamma = R^-1 M_L R^-1); order 4: Gamma = (R^-1 M_L)^3 R^-1."""

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
        self._ksp = PETSc.KSP().create(domain.comm)
        self._ksp.setOperators(self._R)
        self._ksp.setType("preonly")
        self._ksp.getPC().setType("lu")
        self._ksp.getPC().setFactorSolverType("mumps")
        self._ksp.setUp()

    def _R_mult(self, v):
        self._b.array[:] = v
        self._R.mult(self._b, self._x)
        return self._x.array.copy()

    def _R_solve(self, v):
        self._b.array[:] = v
        self._ksp.solve(self._b, self._x)
        return self._x.array.copy()

    def mult(self, v):
        y = self._R_mult(v)
        for _ in range(self.order - 1):
            y = self._R_mult(y / self.M_lump)
        return y

    def pointwise_std(self, max_dofs=6000):
        n = self.ndofs
        if n > max_dofs:
            print(f"  prior pointwise std skipped ({n} dofs > {max_dofs})")
            return None
        Y = np.empty((n, n))
        e = np.zeros(n)
        for j in range(n):
            e[j] = 1.0
            y = self._R_solve(e)
            for _ in range(self.order // 2 - 1):      # y = R^-1 (M_L R^-1)^(order/2-1) e_j
                y = self._R_solve(self.M_lump * y)
            Y[:, j] = y
            e[j] = 0.0
        return np.sqrt(np.abs((Y ** 2) @ self.M_lump))


# =============================================================================
# patient geometry
# =============================================================================

geo = load_lv_geometry(args.vtu,
                       fiber_source="vtu" if args.fiber_vtu else "rule",
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

# =============================================================================
# measurement points: same selection as ex03_lv_vtu_forward
# =============================================================================

dof_coords = V.tabulate_dof_coordinates().reshape(-1, dim)
selected_dofs, selected_coords = select_distributed_nodes(
    dof_coords, N=args.num_nodes, min_dist=args.min_dist, seed=42)
selected_dofs = np.asarray(selected_dofs, dtype=np.int64)
Npoints = len(selected_dofs)
save_selected_nodes(selected_coords, selected_dofs)

# the measurement cells are found through the geometry dofmap, so the selected
# P1 dof indices must also be geometry node indices (same ordering)
if not np.allclose(dof_coords[selected_dofs], domain.geometry.x[selected_dofs, :dim]):
    raise RuntimeError("P1 dof ordering differs from the geometry node ordering: the data "
                       "would be compared in the wrong cells.")

# =============================================================================
# forward model, parameterized by m = log C (as ex06)
# =============================================================================

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
    """Load stepping with automatic step cutting (as ex03_lv_vtu_forward)."""
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
        if args.verbose_forward:
            print(f"  load factor {lam:.3f}: p_endo = {lam * args.load:.3f}")


# =============================================================================
# measurement cells, data container, likelihood
# =============================================================================

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
indicator = fem.Function(W0, name="measurement_indicator")
cell_vol = fem.Function(W0)
fem.assemble_vector(cell_vol.x.array, fem.form(ufl.TestFunction(W0) * dx))
sensor_weight = fem.Function(W0)
for cell in sensor_cells:
    c = W0.dofmap.cell_dofs(cell)
    sensor_weight.x.array[c] = 1.0 / cell_vol.x.array[c]
    indicator.x.array[c] = 1.0
sensor_weight.x.scatter_forward()
indicator.x.scatter_forward()

noise_prec = fem.Constant(domain, 1.0 / args.noise_std ** 2)
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


# =============================================================================
# objective (as ex06)
# =============================================================================

class MAPObjective:
    def __init__(self):
        self.reset()

    def reset(self):
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

    def build_hop(self):
        Hm, obj = HessianOperator(Fun, Phi, uh, m_fun, lmbda, V, Va, facet_tags, domain), self

        class _Full:
            ndofs_m = Hm.ndofs_m

            def mult(self, x):
                obj.n_hessvec += 1
                return Hm.mult(x) + prior.mult(x)
        return _Full()


obj = MAPObjective()


def proj_grad_inf(x, g):
    return float(np.abs(np.clip(x - g, LO, HI) - x).max())


# =============================================================================
# regions from the tissue labels (nodal), mapped to the P1 dofs
# =============================================================================

data = geo.data
is_fib = np.isin(data.tissue, FIBROSIS_LABELS).astype(float)
npts = len(data.points)
frac_fib = (np.bincount(data.cells.ravel(), weights=np.repeat(is_fib, 4), minlength=npts)
            / np.maximum(np.bincount(data.cells.ravel(), minlength=npts), 1))
dof_xyz = Va.tabulate_dof_coordinates()[:, :3]
dist, k_pt = cKDTree(data.points).query(dof_xyz)
assert dist.max() < 1e-6 * np.ptp(data.points, axis=0).max(), "dof/vtu node matching failed"
frac_dof = frac_fib[k_pt]
regions = {"fibrosis": frac_dof >= 1.0 - 1e-12,
           "border": (frac_dof > 1e-12) & (frac_dof < 1.0 - 1e-12),
           "healthy": frac_dof <= 1e-12}
region_code = np.select([regions["healthy"], regions["border"], regions["fibrosis"]],
                        [0.0, 1.0, 2.0])

# geometric scales, to judge the prior correlation length
vols = cell_volumes(data.points, data.cells)
V_fib = float(vols[is_fib > 0].sum())
r_eq = (3.0 * V_fib / (4.0 * np.pi)) ** (1.0 / 3.0)
bbox = np.ptp(data.points, axis=0)


# =============================================================================
# metrics and output helpers
# =============================================================================

def l2_rel(err, ref, mask=None):
    """Relative L2 with the lumped mass (same weights as the prior)."""
    w = prior.M_lump if mask is None else prior.M_lump * mask
    return float(np.sqrt(np.sum(w * err ** 2) / max(np.sum(w * ref ** 2), 1e-300)))


def param_metrics(m_est, m_true):
    C_est, C_true = np.exp(m_est), np.exp(m_true)
    dC = C_est - C_true
    rel = np.abs(dC) / C_true
    out = dict(C_rel_L2=l2_rel(dC, C_true), C_max_rel=float(rel.max()),
               C_mean_rel=float(rel.mean()), m_max_abs=float(np.abs(m_est - m_true).max()),
               n_at_lower_bound=int(np.sum(m_est <= LO + 1e-10)),
               n_at_upper_bound=int(np.sum(m_est >= HI - 1e-10)))
    for name, mask in regions.items():
        if mask.any():
            w = prior.M_lump[mask]
            out[f"C_rel_L2_{name}"] = l2_rel(dC, C_true, mask.astype(float))
            out[f"C_mean_est_{name}"] = float(w @ C_est[mask] / w.sum())
            out[f"C_mean_true_{name}"] = float(w @ C_true[mask] / w.sum())
    return out


def state_metrics(u_est, u_ref, Fd_ref):
    du_ = u_est.x.array - u_ref.x.array
    u_rel = np.linalg.norm(du_) / max(np.linalg.norm(u_ref.x.array), 1e-300)
    eF = compute_Fh(domain, u_est, selected_dofs) - Fd_ref
    return dict(u_rel_nodal=float(u_rel),
                F_rel_frob=float(np.linalg.norm(eF) / np.linalg.norm(Fd_ref)))


def evaluate_point(x, label):
    J_val = obj.eval_J(x)
    g = obj.eval_grad(x)
    u_copy = fem.Function(V, name=f"displacement_{label}")
    u_copy.x.array[:] = uh.x.array
    return dict(J=float(J_val), Phi=float(obj.Jd), J_prior=float(obj.Jp),
                proj_grad_inf=proj_grad_inf(x, g)), u_copy


def va_function(name, arr):
    f = fem.Function(Va, name=name)
    f.x.array[:] = arr
    f.x.scatter_forward()
    return f


def param_fields(tag, m_arr, m_true=None):
    C_arr = np.exp(m_arr)
    out = [va_function(f"m_{tag}", m_arr), va_function(f"C_{tag}", C_arr)]
    if m_true is not None:
        C_true = np.exp(m_true)
        out += [va_function(f"C_abs_error_{tag}", np.abs(C_arr - C_true)),
                va_function(f"C_rel_error_{tag}", np.abs(C_arr - C_true) / C_true)]
    return out


def plot_convergence(path, hist):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    it = [h["iter"] for h in hist]
    axes[0].semilogy(it, [h["J"] for h in hist], "o-", ms=4)
    axes[1].semilogy(it, [max(h["pgrad"], 1e-30) for h in hist], "o-", ms=4)
    axes[0].set(xlabel="Newton iteration", ylabel=r"$\mathcal{J}(m)$")
    axes[1].set(xlabel="Newton iteration", ylabel=r"$\|P(m-g)-m\|_\infty$")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("MAP convergence, Case 3 (patient LV), Newton-CG")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# 1. synthetic data on the patient mesh
# =============================================================================

print("\n" + "=" * 70)
print(f" ex06 MAP  |  Case 3, patient LV ({args.vtu})")
print(f"  mesh: {n} P1 dofs, {len(data.cells)} cells, bounding box "
      f"{bbox[0]:.1f} x {bbox[1]:.1f} x {bbox[2]:.1f}")
print(f"  fibrosis: {int(is_fib.sum())} cells, volume {V_fib:.1f} "
      f"({100 * V_fib / vols.sum():.1f}% of the LV), equivalent radius {r_eq:.1f}")
print(f"  regions (P1 nodes): fibrosis {regions['fibrosis'].sum()}, "
      f"border {regions['border'].sum()}, healthy {regions['healthy'].sum()}")
print(f"  prior: gamma={prior.gamma:.4g}, delta={prior.delta:.4g}, beta={prior.beta:.4g}, "
      f"m0=log({args.C0}), order {prior.order}  (Case 2 calibration: rho ~ 11.5; compare with the lesion size)")
print(f"  measurement points: {Npoints} (min dist {args.min_dist})")
print("=" * 70)

os.environ["OUTPUT_DIR"] = str(OUT / "forward")        # ex03 writes its files here
ud_arr, Fd_arr, CC = prob_ventricle_passive_filling(
    geo, ndofs_data=args.num_nodes, c_healthy=args.c_healthy,
    fibrosis_factor=args.fibrosis_factor, target_load=args.load, kappa_bulk=args.kappa,
    param_space=args.truth_space, load_steps=args.load_steps, min_dist=args.min_dist,
    compute_stress_output=args.stress)
Fd_clean = np.asarray(Fd_arr, dtype=float)
if Fd_clean.shape[0] != Npoints:
    raise RuntimeError(f"ex03 returned {Fd_clean.shape[0]} F samples, {Npoints} points "
                       "selected here -- node selection must match.")

# truth on the P1 parameter space (nodal average of the cell values)
C_true = (CC if args.truth_space == "P1"
          else stiffness_field(geo, args.c_healthy, args.fibrosis_factor, space="P1")).x.array.copy()
m_true = np.log(C_true)
print(f"  C_true (P1) in [{C_true.min():.3f}, {C_true.max():.3f}] kPa; "
      f"data generated from the {args.truth_space} field")

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
print(f"  data noise {100 * args.data_noise:g}% of RMS(Fd-I) = {F_scale:.3e} -> std "
      f"{sigma_data:.3e}; likelihood std {sigma_lik:.3e}; N_obs = {Fd_obs.size}")

# =============================================================================
# 2. MAP by inexact Newton-CG
# =============================================================================

prior_std = None if args.no_prior_std else prior.pointwise_std()
x0 = np.clip(np.log(np.random.RandomState(args.init_seed).uniform(*args.C_init_range, n)),
             LO, HI)

obj.reset()
true_obj, u_true = evaluate_point(m_true, "true")
init_obj, u_init = evaluate_point(x0, "init")

print("\n--- MAP by inexact Newton-CG ---")
obj.reset()
snapshots = []


def grad_logged(x):
    g = obj.eval_grad(x)
    snapshots.append((obj.n_forward, obj.n_hessvec, proj_grad_inf(x, g)))
    return g


t0 = time.perf_counter()
m_map, nhist = inexact_newton_cg(
    obj.eval_J, grad_logged, obj.build_hop, x0=x0.copy(),
    max_outer_iter=args.maxiter, grad_tol=args.gtol, ftol=args.ftol,
    cg_tol=args.cg_tol, cg_maxiter=args.cg_maxiter,
    bounds=(LO, HI), eisenstat_walker=True, verbose=True)
wall = time.perf_counter() - t0
stop = nhist[-1].get("stopped_reason", "max_outer_iter") if nhist else "none"
info = dict(n_iter=sum(1 for h in nhist if "stopped_reason" not in h),
            n_forward=obj.n_forward, n_hessvec=obj.n_hessvec, wall_time_s=wall,
            stop_reason=stop)
hist = [dict(iter=h["iter"], J=h["J"], pgrad=pg, n_forward=nf, n_hessvec=nh,
             cg_iters=h["cg_iters"]) for h, (nf, nh, pg) in zip(nhist, snapshots)]
np.save(OUT / f"m_map_{TAG}.npy", m_map)

map_obj, u_map = evaluate_point(m_map, "map")

# =============================================================================
# outputs
# =============================================================================

metrics = dict(
    case=TAG,
    settings=vars(args) | {"m0": M0, "beta": prior.beta, "n_param_dofs": n,
                           "n_measurement_points": Npoints, "n_obs": int(Fd_obs.size),
                           "fibrosis_volume": V_fib, "fibrosis_equiv_radius": r_eq,
                           "n_nodes_per_region": {k: int(v.sum()) for k, v in regions.items()}},
    data_noise=dict(level=args.data_noise, F_scale_rms=F_scale, sigma_data=sigma_data,
                    noise_std_likelihood=sigma_lik),
    true=true_obj,
    init=init_obj | param_metrics(x0, m_true) | state_metrics(u_init, u_true, Fd_clean),
    newtoncg=info | map_obj | param_metrics(m_map, m_true) | state_metrics(u_map, u_true, Fd_clean),
)
with open(OUT / f"metrics_{TAG}.json", "w") as f:
    json.dump(metrics, f, indent=2, default=float)
np.savetxt(OUT / f"history_newtoncg_{TAG}.txt",
           np.array([[h["iter"], h["J"], h["pgrad"], h["n_forward"], h["n_hessvec"],
                      h["cg_iters"]] for h in hist]),
           header="iter J proj_grad_inf n_forward n_hessvec cg_iters")
plot_convergence(OUT / f"convergence_{TAG}.png", hist)

tissue_p1 = va_function("fibrosis_fraction", frac_dof)
fields = (param_fields("true", m_true) + param_fields("init", x0, m_true)
          + param_fields("map", m_map, m_true)
          + [va_function("region", region_code), tissue_p1, geo.tissue,
             u_true, u_map, indicator, Fd_func])
if prior_std is not None:
    fields.append(va_function("m_prior_std", prior_std))
with XDMFFile(MPI.COMM_WORLD, str(OUT / f"map_{TAG}.xdmf"), "w") as xf:
    xf.write_mesh(domain)
    for fn in fields:
        xf.write_function(fn)

m = metrics["newtoncg"]
print("\n--- Case 3 (patient LV): summary ---")
print(f"  Newton-CG: {info['n_iter']} iterations, {info['n_forward']} forward solves, "
      f"{info['n_hessvec']} Hessian applies, {wall:.0f} s, stop: {stop}")
print(f"  J(MAP) = {m['J']:.4e} (Phi {m['Phi']:.3e}, prior {m['J_prior']:.3e});  "
      f"J(truth) = {true_obj['J']:.4e}")
print(f"  C rel. L2 error: whole LV {100 * m['C_rel_L2']:.2f}%  |  "
      + "  ".join(f"{k} {100 * m[f'C_rel_L2_{k}']:.2f}%" for k in regions if f"C_rel_L2_{k}" in m))
print("  mean C [kPa] MAP / true: "
      + "  ".join(f"{k} {m[f'C_mean_est_{k}']:.2f}/{m[f'C_mean_true_{k}']:.2f}"
                  for k in regions if f"C_mean_est_{k}" in m))
print(f"  F rel. misfit {m['F_rel_frob']:.2e};  nodes at bounds: "
      f"{m['n_at_lower_bound']} lower, {m['n_at_upper_bound']} upper")
print(f"\nDone. Outputs in {OUT.resolve()}")
