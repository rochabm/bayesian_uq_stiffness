"""
taylor_test_log.py
------------------
Taylor remainder test of the gradient and the Hessian of the MAP functional
in the log-stiffness m = ln C, i.e. the functional actually minimized for the
results (ex06/ex07):

    J(m) = Phi(m) + 0.5 (m - m0)^T H_prior (m - m0)
    Phi(m) = 1/(2 sigma^2) sum_j || F_j(u(m)) - Fd_j ||_F^2      (measurement points j)
    W = exp(m)/2 (exp(Q) - 1) + kappa (J ln J - J + 1)

    gradient : adjoint of Phi + H_prior (m - m0)                   (as ex06)
    Hessian  : HessianOperator (hessian_ucq.py) on Phi + H_prior   (as ex06/ex07)

    R_grad(eps) = |J(m + eps h) - J(m) - eps g^T h|                  O(eps^2)
    R_hess(eps) = |J(m + eps h) - J(m) - eps g^T h - eps^2/2 h^T H h|  O(eps^3)

Setup: ellipsoid mesh, clean synthetic data from the linear ground truth of
Case 1, Case 1 prior (gamma = 6.22, delta = 9.85e-3, m0 = ln 2), base point
m = ln 3 (C = 3 kPa), random unit direction h. Every evaluation solves the
forward problem from zero displacement with the same load steps, a Newton
tolerance of 1e-12 and a direct solver (MUMPS).

taylor_test.py does the same test for the earlier parameterization in C.

Usage (from the repository root, dolfinx 0.8):
    python -m verification.taylor_test_log
    python -m verification.taylor_test_log --output-dir results/verification

Writes <output-dir>/fig_taylor_log.pdf/.png and taylor_remainders_log.txt.
Serial only.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpi4py import MPI
import ufl
import dolfinx
import dolfinx.fem.petsc  # noqa: F401  (makes fem.petsc available)
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
import cardiac_geometries

from cardiac_utils import select_distributed_nodes
from hessian_ucq import HessianOperator
from paper_style import apply_paper_style, save_figure, COLORS, FIG_SINGLE

if MPI.COMM_WORLD.size > 1:
    raise RuntimeError("taylor_test_log is serial-only.")

parser = argparse.ArgumentParser(description="Taylor remainder test in m = ln C (MAP functional)")
parser.add_argument("--num-nodes", type=int, default=64, help="Measurement points (as Case 1)")
parser.add_argument("--gamma", type=float, default=6.22, help="Prior gamma (Case 1)")
parser.add_argument("--delta", type=float, default=9.85e-3, help="Prior delta (Case 1)")
parser.add_argument("--C0", type=float, default=2.0, help="Prior mean in kPa (m0 = ln C0)")
parser.add_argument("--C-base", type=float, default=3.0, help="Base point in kPa (m = ln C_base)")
parser.add_argument("--noise-std", type=float, default=1e-3, help="Likelihood std sigma")
parser.add_argument("--load", type=float, default=-2.0, help="Endocardial pressure [kPa]")
parser.add_argument("--load-steps", type=int, default=2)
parser.add_argument("--newton-tol", type=float, default=1e-12)
parser.add_argument("--seed", type=int, default=42, help="Seed of the direction h")
parser.add_argument("--output-dir", default="results/verification")
parser.add_argument("--use-tex", action="store_true")
args = parser.parse_args()
OUT = Path(args.output_dir)
OUT.mkdir(parents=True, exist_ok=True)


def _interp_pts(space):
    ip = space.element.interpolation_points
    return ip() if callable(ip) else ip


# =============================================================================
# geometry, spaces, BCs
# =============================================================================

geodir = Path("lv_ellipsoid")
try:
    geo = cardiac_geometries.geometry.Geometry.from_folder(geodir)
except Exception:
    geo = cardiac_geometries.mesh.lv_ellipsoid(
        outdir=geodir, create_fibers=True, fiber_space="DG_0",
        psize_ref=3, r_short_epi=10, fiber_angle_endo=40.0, fiber_angle_epi=-50.0)

domain = geo.mesh
dim = domain.geometry.dim
facet_tags = geo.ffun
V = fem.functionspace(domain, ("Lagrange", 1, (dim,)))
Va = fem.functionspace(domain, ("Lagrange", 1))
TT = fem.functionspace(domain, ("DG", 0, (dim, dim)))
W0 = fem.functionspace(domain, ("DG", 0))
du, v = ufl.TrialFunction(V), ufl.TestFunction(V)
uh = fem.Function(V)
base_dofs = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(5))
bcs = [fem.dirichletbc(np.zeros(dim, dtype=default_scalar_type), base_dofs, V)]

# =============================================================================
# forward model in m = ln C (as ex06)
# =============================================================================

m_fun = fem.Function(Va, name="m")
I = ufl.variable(ufl.Identity(dim))
F = ufl.variable(I + ufl.grad(uh))
J = ufl.variable(ufl.det(F))
e1, e2, e3 = geo.f0, geo.s0, geo.n0
bf, bt, bfs = default_scalar_type(6.6), default_scalar_type(4.0), default_scalar_type(2.6)
kappa = fem.Constant(domain, 1e2)
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
       - p_endo * ufl.inner(v, J * ufl.transpose(ufl.inv(F)) * N_fac) * ds(6))


def solve_fwd(m_arr):
    """Forward solve from zero displacement, same load steps every time."""
    m_fun.x.array[:] = m_arr
    m_fun.x.scatter_forward()
    uh.x.array[:] = 0.0
    uh.x.scatter_forward()
    solver = NewtonSolver(domain.comm, NonlinearProblem(Fun, uh, bcs))
    solver.atol = args.newton_tol
    solver.rtol = args.newton_tol
    solver.max_it = 50
    ksp = solver.krylov_solver
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    for load in np.linspace(args.load / args.load_steps, args.load, args.load_steps):
        p_endo.value = load
        n_it, conv = solver.solve(uh)
        if not conv:
            raise RuntimeError(f"Newton did not converge at load {load}")
    return uh.x.array.copy()


# =============================================================================
# measurement points and likelihood (pointwise sum over the points, as ex06)
# =============================================================================

selected_dofs, _ = select_distributed_nodes(domain.geometry.x, N=args.num_nodes, seed=42)
node_to_cells = {}
for cell_id in range(domain.topology.index_map(domain.topology.dim).size_local):
    for node in domain.geometry.dofmap[cell_id]:
        node_to_cells.setdefault(int(node), []).append(cell_id)
sensor_cells = [node_to_cells[int(d)][0] for d in selected_dofs]
if len(set(sensor_cells)) != len(sensor_cells):
    raise RuntimeError("some measurement points share a cell; reduce --num-nodes.")

cell_vol = fem.Function(W0)
fem.assemble_vector(cell_vol.x.array, fem.form(ufl.TestFunction(W0) * dx))
sensor_weight = fem.Function(W0)          # 1/|K_j| on the sensor cells -> sum over j
for cell in sensor_cells:
    c = W0.dofmap.cell_dofs(cell)
    sensor_weight.x.array[c] = 1.0 / cell_vol.x.array[c]
sensor_weight.x.scatter_forward()

Fd_func = fem.Function(TT, name="Fd")
noise_prec = fem.Constant(domain, 1.0 / args.noise_std ** 2)
Phi = 0.5 * noise_prec * sensor_weight * ufl.inner(F - Fd_func, F - Fd_func) * dx
Phi_c = fem.form(Phi)

# synthetic data: F of the forward solution for the Case 1 truth, in the sensor cells
xx = Va.tabulate_dof_coordinates()[:, 0]
C_true = 2.0 + (xx - xx.min()) / (xx.max() - xx.min())           # Case 1 (linear)
solve_fwd(np.log(C_true))
F_all = fem.Function(TT)
F_all.interpolate(fem.Expression(ufl.Identity(dim) + ufl.grad(uh), _interp_pts(TT)))
bs = TT.dofmap.bs
Fd_func.x.array[:] = 0.0
for cell in sensor_cells:
    b = TT.dofmap.cell_dofs(cell)[0]
    Fd_func.x.array[b * bs:(b + 1) * bs] = F_all.x.array[b * bs:(b + 1) * bs]
Fd_func.x.scatter_forward()

# =============================================================================
# BiLaplacian prior on m (as ex06)
# =============================================================================

class BiLaplacianPrior:
    """H_prior = R M_L^{-1} R,  R = gamma K + delta M + beta M_bndry."""

    def __init__(self, gamma, delta):
        self.beta = np.sqrt(gamma * delta) / 1.42
        q, dq = ufl.TrialFunction(Va), ufl.TestFunction(Va)
        ds_all = ufl.Measure("ds", domain=domain)
        R_form = (gamma * ufl.inner(ufl.grad(q), ufl.grad(dq)) * dx
                  + delta * ufl.inner(q, dq) * dx
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

    def _R_mult(self, x):
        self._b.array[:] = x
        self._R.mult(self._b, self._x)
        return self._x.array.copy()

    def mult(self, x):
        return self._R_mult(self._R_mult(x) / self.M_lump)


prior = BiLaplacianPrior(args.gamma, args.delta)
n = Va.dofmap.index_map.size_local
m0_vec = np.full(n, np.log(args.C0))

# =============================================================================
# objective, adjoint gradient, Hessian (as ex06 MAPObjective / FullHessian)
# =============================================================================

lu_opts = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
lmbda = fem.Function(V)
adj_problem = LinearProblem(ufl.adjoint(ufl.derivative(Fun, uh, du)),
                            -ufl.derivative(Phi, uh, v), bcs=bcs, petsc_options=lu_opts)
dFdm_c = fem.form(ufl.action(ufl.adjoint(ufl.derivative(Fun, m_fun, ufl.TrialFunction(Va))),
                             lmbda))
grad_misfit = fem.Function(Va)


def eval_J(m_arr):
    solve_fwd(m_arr)
    dm = m_arr - m0_vec
    return float(fem.assemble_scalar(Phi_c)) + 0.5 * float(dm @ prior.mult(dm))


def eval_J_and_g(m_arr):
    """J and its gradient; leaves uh, m_fun, lmbda at m_arr (for the Hessian)."""
    J_val = eval_J(m_arr)
    lam = adj_problem.solve()
    lmbda.x.array[:] = lam.x.array
    lmbda.x.scatter_forward()
    grad_misfit.x.array[:] = 0.0
    fem.assemble_vector(grad_misfit.x.array, dFdm_c)
    return J_val, grad_misfit.x.array.copy() + prior.mult(m_arr - m0_vec)


# =============================================================================
# Taylor remainder test
# =============================================================================

print("\n" + "=" * 65)
print("  Taylor remainder test in m = ln C  (MAP functional, Case 1 prior)")
print(f"  base point C = {args.C_base} kPa, {len(sensor_cells)} measurement points, "
      f"sigma = {args.noise_std}, load = {args.load} in {args.load_steps} steps")
print("=" * 65)

m_base = np.full(n, np.log(args.C_base))
rng = np.random.default_rng(args.seed)
h_dir = rng.standard_normal(n)
h_dir /= np.linalg.norm(h_dir)
epsilons = [5e-1, 2e-1, 1e-1, 5e-2, 2e-2, 1e-2, 5e-3]

J0, g0 = eval_J_and_g(m_base)
dJ = float(g0 @ h_dir)
Hm = HessianOperator(Fun, Phi, uh, m_fun, lmbda, V, Va, facet_tags, domain)  # state at m_base
Hh = Hm.mult(h_dir) + prior.mult(h_dir)
d2J = float(h_dir @ Hh)
J_prior0 = 0.5 * float((m_base - m0_vec) @ prior.mult(m_base - m0_vec))

print(f"\n  J(m0)        = {J0:.6e}   (Phi {J0 - J_prior0:.4e}, prior {J_prior0:.4e})")
print(f"  g^T h        = {dJ:.6e}")
print(f"  h^T H h      = {d2J:.6e}")
print(f"  ||H*h||      = {np.linalg.norm(Hh):.6e}")

R_grad, R_hess = [], []
for eps in epsilons:
    dJfd = eval_J(m_base + eps * h_dir) - J0
    R_grad.append(abs(dJfd - eps * dJ))
    R_hess.append(abs(dJfd - eps * dJ - 0.5 * eps ** 2 * d2J))


def rate(a, b, ea, eb):
    return np.log(a / b) / np.log(ea / eb) if a > 0 and b > 0 else float("nan")


rates_g = [rate(R_grad[i], R_grad[i - 1], epsilons[i], epsilons[i - 1])
           for i in range(1, len(epsilons))]
rates_h = [rate(R_hess[i], R_hess[i - 1], epsilons[i], epsilons[i - 1])
           for i in range(1, len(epsilons))]

print(f"\n  {'eps':>8}  {'R_grad':>14}  {'R_hess':>14}  {'rate_grad':>10}  {'rate_hess':>10}")
print("  " + "-" * 65)
for i, eps in enumerate(epsilons):
    rg = f"{rates_g[i - 1]:.2f}" if i > 0 else "-"
    rh = f"{rates_h[i - 1]:.2f}" if i > 0 else "-"
    print(f"  {eps:>8.1e}  {R_grad[i]:>14.4e}  {R_hess[i]:>14.4e}  {rg:>10}  {rh:>10}")

# average over the first rates (the last two are closest to round-off), as taylor_test.py
i_end = max(len(rates_g) - 2, len(rates_g) // 2)
avg_rg = np.nanmean(rates_g[:i_end])
avg_rh = np.nanmean(rates_h[:i_end])
grad_pass = abs(avg_rg - 2.0) < 0.25
hess_pass = abs(avg_rh - 3.0) < 0.5
print(f"\n  Average rate R_grad: {avg_rg:.3f}  (expected 2)  {'✓ PASS' if grad_pass else '✗ FAIL'}")
print(f"  Average rate R_hess: {avg_rh:.3f}  (expected 3)  {'✓ PASS' if hess_pass else '✗ FAIL'}")

# =============================================================================
# outputs (paper style)
# =============================================================================

eps_arr = np.array(epsilons)
np.savetxt(OUT / "taylor_remainders_log.txt", np.c_[eps_arr, R_grad, R_hess],
           header="eps R_grad R_hess   (m = ln C, MAP functional)")

apply_paper_style(use_tex=args.use_tex)
fig, ax = plt.subplots(figsize=FIG_SINGLE)
ax.loglog(eps_arr, R_grad, "o-", color=COLORS["blue"], markerfacecolor="white",
          markeredgewidth=0.8, label=r"$R_{\mathrm{grad}}$")
ax.loglog(eps_arr, R_hess, "s-", color=COLORS["orange"], markerfacecolor="white",
          markeredgewidth=0.8, label=r"$R_{\mathrm{hess}}$")
for R, p, txt in [(R_grad, 2, r"$\mathcal{O}(\epsilon^2)$"),
                  (R_hess, 3, r"$\mathcal{O}(\epsilon^3)$")]:
    ref = 0.1 * R[0] * (eps_arr / eps_arr[0]) ** p      # one decade below the data
    ax.loglog(eps_arr, ref, "--", color=COLORS["gray"], lw=0.8)
    k = len(eps_arr) // 2
    ax.annotate(txt, (eps_arr[k], ref[k]), textcoords="offset points", xytext=(4, -10),
                color=COLORS["gray"])
ax.set_xlabel(r"Perturbation size $\epsilon$ [$-$]")
ax.set_ylabel(r"Taylor remainder [$-$]")
ax.legend(loc="upper left")
save_figure(fig, str(OUT / "fig_taylor_log"))
plt.close(fig)

print("\n" + "=" * 65)
print(f"  VERDICT: {'ALL PASSED ✓' if grad_pass and hess_pass else 'SEE ABOVE'}")
print("=" * 65)
