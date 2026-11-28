"""
Passive filling forward problem on the patient-specific LV mesh
(Patient_7_lv_tissue.vtu), based on ex03_ventricle_discrete_forward.py.

Differences with respect to ex03
--------------------------------
- geometry read from the .vtu instead of cardiac_geometries
- boundary markers computed from the surface (lv_geometry.py), same ids as
  ex03: 5 = base (fixed), 6 = endocardium (pressure), 7 = epicardium
- fibres: rule-based with the ex03 angles (+40 endo, -50 epi), piecewise
  constant per cell (DG0). Optional: copy the patient fibres f_0/s_0/n_0
  from Patient_7_UVC.vtu (fiber_source="vtu").
- stiffness from the "tissue" field:
      tissue 0      -> C = c_healthy
      tissue 1, 2   -> C = fibrosis_factor * c_healthy   (3x by default)
- direct solver (MUMPS) + automatic load-step cutting if Newton fails
- stress at the final load (compute_stress):
      stress.xdmf           : Cauchy stress tensor + von Mises, fibre
                              stress, hydrostatic pressure, fibre stretch
      stress_by_tissue.csv  : healthy vs fibrosis statistics (also without
                              the basal 10%, where the fixed base distorts
                              the stress)

Constitutive law (bf, bt, bfs, C, volumetric term kappa), base fixed,
pressure sign convention, quadrature degree, node selection and outputs
are the same as in ex03.

Run:
    python ex03_lv_vtu_forward.py Patient_7_lv_tissue.vtu
    python ex03_lv_vtu_forward.py Patient_7_lv_tissue.vtu Patient_7_UVC.vtu
        (second argument -> use the patient fibres from that file)
    mpirun -n 4 python ex03_lv_vtu_forward.py Patient_7_lv_tissue.vtu
"""

from pathlib import Path
from types import SimpleNamespace
import os
import sys
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from scipy.spatial import cKDTree
import basix.ufl
import dolfinx
from dolfinx import fem, default_scalar_type
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import ufl
from ufl import inner

from cardiac_utils import (
    compute_Fh,
    save_results_xdmf,
    save_selected_nodes,
    select_distributed_nodes,
)
from lv_geometry import (
    MARKERS,
    FIBROSIS_LABELS,
    cell_volumes,
    format_stats,
    load_lv_vtu,
    region_stats,
    stiffness_per_cell,
    summary,
    write_stats_csv,
)


# =============================================================================
# geometry from the .vtu
# =============================================================================

def _create_mesh(comm, cells, x):
    coord_el = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))
    ufl_domain = ufl.Mesh(coord_el)
    try:                                   # dolfinx 0.8 / 0.9
        return dolfinx.mesh.create_mesh(comm, cells, x, ufl_domain)
    except TypeError:                      # dolfinx >= 0.10 argument order
        return dolfinx.mesh.create_mesh(comm, cells, ufl_domain, x)


def load_lv_geometry(vtu_path, fiber_source="rule", fiber_vtu=None,
                     alpha_endo=40.0, alpha_epi=-50.0, comm=MPI.COMM_WORLD):
    """
    Returns an object with the attributes ex03 uses from cardiac_geometries
    (mesh, ffun, markers, f0, s0, n0) plus tissue, data, cell_map, ncells.
    """
    data = load_lv_vtu(vtu_path, fiber_source=fiber_source,
                       fiber_vtu=fiber_vtu,
                       alpha_endo=alpha_endo, alpha_epi=alpha_epi)
    if comm.rank == 0:
        print(summary(data))

    if comm.rank == 0:
        cells, x = data.cells, data.points
    else:
        cells = np.empty((0, 4), dtype=np.int64)
        x = np.empty((0, 3), dtype=np.float64)
    domain = _create_mesh(comm, cells, x)

    tdim = domain.topology.dim
    fdim = tdim - 1
    tol = 1e-6 * np.ptp(data.points, axis=0).max()

    # ---- cell data: match local cells to VTU cells by centroid ------------
    imap = domain.topology.index_map(tdim)
    ncells = imap.size_local + imap.num_ghosts
    mids = dolfinx.mesh.compute_midpoints(
        domain, tdim, np.arange(ncells, dtype=np.int32))
    dist, cell_map = cKDTree(data.cell_centroids).query(mids)
    assert dist.max() < tol, "cell matching with the .vtu failed"

    W = fem.functionspace(domain, ("DG", 0, (3,)))
    Q0 = fem.functionspace(domain, ("DG", 0))

    def dg0_vector(values, name):
        fn = fem.Function(W, name=name)
        fn.x.array.reshape(-1, 3)[W.dofmap.list[:ncells, 0]] = values[cell_map]
        fn.x.scatter_forward()
        return fn

    def dg0_scalar(values, name):
        fn = fem.Function(Q0, name=name)
        fn.x.array[Q0.dofmap.list[:ncells, 0]] = values[cell_map]
        fn.x.scatter_forward()
        return fn

    f0 = dg0_vector(data.f0, "f0")
    s0 = dg0_vector(data.s0, "s0")
    n0 = dg0_vector(data.n0, "n0")
    tissue = dg0_scalar(data.tissue.astype(np.float64), "tissue")

    # ---- facet tags --------------------------------------------------------
    domain.topology.create_connectivity(fdim, tdim)
    ext = dolfinx.mesh.exterior_facet_indices(domain.topology)
    fmids = dolfinx.mesh.compute_midpoints(domain, fdim, ext)
    dist, j = cKDTree(data.bnd_midpoints).query(fmids)
    assert dist.size == 0 or dist.max() < tol, "facet matching failed"
    ffun = dolfinx.mesh.meshtags(domain, fdim, ext,
                                 data.bnd_markers[j].astype(np.int32))

    return SimpleNamespace(mesh=domain, ffun=ffun, markers=MARKERS,
                           f0=f0, s0=s0, n0=n0, tissue=tissue,
                           data=data, cell_map=cell_map, ncells=ncells)


def stiffness_field(geo, c_healthy=2.0, fibrosis_factor=3.0, space="DG0"):
    """
    True parameter field C.
      "DG0": cell-wise, sharp jump at the fibrosis border (exact labels)
      "P1" : nodal average of the cell values (smooth border), e.g. if the
             inverse problem (ex04) estimates C in P1
    """
    domain, data = geo.mesh, geo.data
    C_cells = stiffness_per_cell(data.tissue, c_healthy, fibrosis_factor)

    if space == "DG0":
        Va = fem.functionspace(domain, ("DG", 0))
        CC = fem.Function(Va, name="C")
        CC.x.array[Va.dofmap.list[:geo.ncells, 0]] = C_cells[geo.cell_map]
    elif space == "P1":
        npts = len(data.points)
        vsum = np.bincount(data.cells.ravel(),
                           weights=np.repeat(C_cells, 4), minlength=npts)
        vcnt = np.bincount(data.cells.ravel(), minlength=npts)
        C_nodes = vsum / np.maximum(vcnt, 1)

        Va = fem.functionspace(domain, ("Lagrange", 1))
        CC = fem.Function(Va, name="C")
        coords = Va.tabulate_dof_coordinates()[:, :3]
        dist, k = cKDTree(data.points).query(coords)
        assert dist.max() < 1e-6 * np.ptp(data.points, axis=0).max()
        CC.x.array[:] = C_nodes[k]
    else:
        raise ValueError("space must be 'DG0' or 'P1'")

    CC.x.scatter_forward()
    return CC


# =============================================================================
# stress post-processing
# =============================================================================

def _interpolation_points(V):
    ip = V.element.interpolation_points
    return ip() if callable(ip) else ip          # dolfinx 0.8 vs 0.9+


def _apicobasal_height(data):
    """Nodal height: 1 on the base plane, 0 at the apex (geometric)."""
    h = getattr(data, "height", None)
    if h is not None:
        return h
    pts = data.points
    bf = data.bnd_facets[data.bnd_markers == MARKERS["BASE"][0]]
    x = pts[bf]
    nb = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]).sum(axis=0)
    nb /= np.linalg.norm(nb)
    h = (pts - pts[np.unique(bf)].mean(axis=0)) @ nb
    h = h * np.sign(-h[np.argmax(np.abs(h))])      # apex at negative h
    return 1.0 + h / np.abs(h.min())


def compute_stress(geo, F, J, P, I, output_dir, base_cut=0.9):
    """
    Cauchy stress and scalar measures, cell-wise (DG0, evaluated at the
    cell midpoint), plus statistics per tissue region.

        sigma     = (1/J) P F^T                   Cauchy stress
        sigma_vm  = sqrt(3/2 dev(sigma):dev(sigma))   von Mises
        sigma_ff  = f . sigma f,  f = F f0/|F f0|      fibre stress
        p_hyd     = -tr(sigma)/3                  hydrostatic pressure
        lambda_f  = |F f0|                        fibre stretch

    Writes stress.xdmf and stress_by_tissue.csv, prints the table.
    """
    domain = geo.mesh
    comm = domain.comm
    Q0 = geo.tissue.function_space
    T0 = fem.functionspace(domain, ("DG", 0, (3, 3)))

    sigma = (1.0 / J) * P * F.T
    dev = sigma - ufl.tr(sigma) / 3.0 * I
    Ff0 = F * geo.f0
    lam_f = ufl.sqrt(ufl.inner(Ff0, Ff0))
    f_cur = Ff0 / lam_f

    exprs = {
        "sigma_vm": ufl.sqrt(1.5 * ufl.inner(dev, dev)),
        "sigma_ff": ufl.inner(sigma * f_cur, f_cur),
        "p_hyd":    -ufl.tr(sigma) / 3.0,
        "lambda_f": lam_f,
    }

    sig_fn = fem.Function(T0, name="sigma")
    sig_fn.interpolate(fem.Expression(sigma, _interpolation_points(T0)))

    funcs = {}
    for name, e in exprs.items():
        fn = fem.Function(Q0, name=name)
        fn.interpolate(fem.Expression(e, _interpolation_points(Q0)))
        fn.x.scatter_forward()
        funcs[name] = fn

    with XDMFFile(comm, str(output_dir / "stress.xdmf"), "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(sig_fn)
        for fn in funcs.values():
            xdmf.write_function(fn)
        xdmf.write_function(geo.tissue)

    # ---- statistics per region (owned cells, gathered on rank 0) ----------
    data = geo.data
    nloc = domain.topology.index_map(domain.topology.dim).size_local
    dofs = Q0.dofmap.list[:nloc, 0]
    cm = geo.cell_map[:nloc]

    local = {name: fn.x.array[dofs].copy() for name, fn in funcs.items()}
    local["_vol"] = cell_volumes(data.points, data.cells)[cm]
    local["_tissue"] = data.tissue[cm]
    local["_height"] = _apicobasal_height(data)[data.cells[cm]].mean(axis=1)

    gathered = comm.gather(local, root=0)
    rows = None
    if comm.rank == 0:
        allv = {k: np.concatenate([g[k] for g in gathered]) for k in local}
        fields = {k: v for k, v in allv.items() if not k.startswith("_")}
        rows = region_stats(fields, allv["_vol"], allv["_tissue"],
                            allv["_height"], base_cut=base_cut)
        write_stats_csv(rows, output_dir / "stress_by_tissue.csv")
        print("\nstress by tissue region (volume-weighted mean, "
              "ratio = mean / healthy mean)")
        print(format_stats(rows))
    return funcs, rows


# =============================================================================
# forward problem
# =============================================================================

def prob_ventricle_passive_filling(geo, ndofs_data=32,
                                   c_healthy=2.0, fibrosis_factor=3.0,
                                   target_load=-3.0, kappa_bulk=1e2,
                                   param_space="DG0", load_steps=10,
                                   min_dist=4.0, compute_stress_output=True):
    """
    Solves the passive filling forward problem and returns synthetic data.

    Returns
    -------
    u_mat : (Npoints, 3)    displacement at selected nodes
    Fd    : (Npoints, 3, 3) deformation gradient at selected nodes
    CC    : dolfinx.fem.Function, true parameter field
    """
    comm = geo.mesh.comm
    rank0 = comm.rank == 0
    if rank0:
        print("\nsolving forward problem to generate synthetic data")

    output_dir = Path(os.environ.get("OUTPUT_DIR", "."))
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # mesh
    # -------------------------------------------------------------------------

    domain     = geo.mesh
    facet_tags = geo.ffun
    markers    = geo.markers
    BASE       = markers["BASE"][0]
    ENDO       = markers["ENDO"][0]

    if rank0:
        print(markers)

    # -------------------------------------------------------------------------
    # true parameter field from the tissue labels
    # -------------------------------------------------------------------------

    CC = stiffness_field(geo, c_healthy, fibrosis_factor, space=param_space)
    if rank0:
        nfib = int(np.isin(geo.data.tissue, FIBROSIS_LABELS).sum())
        print(f"C healthy = {c_healthy}, C fibrosis = "
              f"{c_healthy * fibrosis_factor} ({nfib} fibrotic cells, "
              f"tissue 1 and 2), space = {param_space}")

    with XDMFFile(comm, str(output_dir / "fiber.xdmf"), "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_meshtags(facet_tags, domain.geometry)
        xdmf.write_function(geo.f0)
        xdmf.write_function(geo.tissue)
        xdmf.write_function(CC)

    # -------------------------------------------------------------------------
    # function spaces
    # -------------------------------------------------------------------------

    V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    v = ufl.TestFunction(V)
    u = fem.Function(V)

    # -------------------------------------------------------------------------
    # boundary conditions (base fixed)
    # -------------------------------------------------------------------------

    u_bc      = np.array((0,) * domain.geometry.dim, dtype=default_scalar_type)
    base_dofs = fem.locate_dofs_topological(V, facet_tags.dim,
                                            facet_tags.find(BASE))
    bcs       = [fem.dirichletbc(u_bc, base_dofs, V)]

    # -------------------------------------------------------------------------
    # nodes selected for data measurement
    # -------------------------------------------------------------------------

    dim        = domain.geometry.dim
    dof_coords = V.tabulate_dof_coordinates().reshape(-1, dim)

    selected_dofs, selected_coords = select_distributed_nodes(
        dof_coords, N=ndofs_data, min_dist=min_dist, seed=42
    )

    Npoints = len(selected_dofs)
    print(f"Número de nodes: {Npoints}")
    print("Coordenadas nodes selecionados:")
    print(selected_coords)

    save_selected_nodes(selected_coords, selected_dofs)

    # -------------------------------------------------------------------------
    # kinematics
    # -------------------------------------------------------------------------

    d = len(u)
    I = ufl.variable(ufl.Identity(d))
    F = ufl.variable(I + ufl.grad(u))
    J = ufl.variable(ufl.det(F))

    f0 = geo.f0
    s0 = geo.s0
    n0 = geo.n0

    # -------------------------------------------------------------------------
    # constitutive model (same as ex03)
    # -------------------------------------------------------------------------
    bf    = default_scalar_type(6.6)
    bt    = default_scalar_type(4.0)
    bfs   = default_scalar_type(2.6)
    kappa = fem.Constant(domain, default_scalar_type(kappa_bulk))

    e1, e2, e3 = f0, s0, n0
    Cs  = J**(-2/3) * F.T * F
    Es  = 0.5 * (Cs - I)

    E11 = ufl.inner(Es*e1, e1)
    E12 = ufl.inner(Es*e1, e2)
    E13 = ufl.inner(Es*e1, e3)
    E21 = ufl.inner(Es*e2, e1)
    E22 = ufl.inner(Es*e2, e2)
    E23 = ufl.inner(Es*e2, e3)
    E31 = ufl.inner(Es*e3, e1)
    E32 = ufl.inner(Es*e3, e2)
    E33 = ufl.inner(Es*e3, e3)

    Q = (
        bf  * E11**2
        + bt  * (E22**2 + E33**2 + E23**2 + E32**2)
        + bfs * (E12**2 + E21**2 + E13**2 + E31**2)
    )

    Wpassive      = CC / 2.0 * (ufl.exp(Q) - 1)
    Wvolume       = kappa * (J * ufl.ln(J) - J + 1)
    strain_energy = Wpassive + Wvolume
    P             = ufl.diff(strain_energy, F)

    # -------------------------------------------------------------------------
    # variational form (internal + endocardial pressure)
    # -------------------------------------------------------------------------

    p_endo   = fem.Constant(domain, default_scalar_type(0.0))
    metadata = {"quadrature_degree": 4}
    ds       = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags,
                           metadata=metadata)
    dx       = ufl.Measure("dx", domain=domain, metadata=metadata)

    N     = ufl.FacetNormal(domain)
    Gendo = -p_endo * inner(v, J * ufl.transpose(ufl.inv(F)) * N) * ds(ENDO)
    R     = inner(P, ufl.grad(v)) * dx + Gendo

    # -------------------------------------------------------------------------
    # solver (direct LU with MUMPS)
    # -------------------------------------------------------------------------

    problem       = NonlinearProblem(R, u, bcs)
    solver        = NewtonSolver(comm, problem)
    solver.atol   = 1e-8
    solver.rtol   = 1e-8
    solver.max_it = 30

    ksp    = solver.krylov_solver
    opts   = PETSc.Options()
    prefix = ksp.getOptionsPrefix()
    opts[f"{prefix}ksp_type"] = "preonly"
    opts[f"{prefix}pc_type"] = "lu"
    opts[f"{prefix}pc_factor_mat_solver_type"] = "mumps"
    ksp.setFromOptions()

    # -------------------------------------------------------------------------
    # step-wise loading (with step cutting) + XDMF output
    # -------------------------------------------------------------------------

    filename   = output_dir / "out_ex03_ventricle_u"
    xdmf       = XDMFFile(comm, f"{filename}.xdmf", "w")
    xdmf.write_mesh(domain)
    u_out      = fem.Function(V)
    u_out.name = "u"

    lam, min_dlam = 0.0, 1e-4
    lam_targets = np.linspace(0.0, 1.0, load_steps)

    for step, lam_target in enumerate(lam_targets):
        dlam = lam_target - lam
        while lam < lam_target - 1e-12:
            dlam = min(dlam, lam_target - lam)
            u_prev = u.x.array.copy()
            p_endo.value = (lam + dlam) * target_load
            try:
                num_its, converged = solver.solve(u)
            except RuntimeError:
                converged = False
            if converged:
                lam += dlam
                u.x.scatter_forward()
            else:
                u.x.array[:] = u_prev
                u.x.scatter_forward()
                dlam *= 0.5
                if rank0:
                    print(f"  Newton failed, cutting step to dlam = {dlam:.2e}")
                if dlam < min_dlam:
                    raise RuntimeError(f"no convergence at load factor {lam}")
        if rank0:
            print(f"step {step}: p_endo = {lam * target_load:.3f}")
        u_out.interpolate(u)
        xdmf.write_function(u_out, step)

    xdmf.close()
    u.x.scatter_forward()

    # -------------------------------------------------------------------------
    # stress at the final load (healthy vs fibrotic)
    # -------------------------------------------------------------------------

    if compute_stress_output:
        compute_stress(geo, F, J, P, I, output_dir)

    # -------------------------------------------------------------------------
    # extract solution at selected DOFs
    # -------------------------------------------------------------------------

    save_results_xdmf(domain, {"displacement": u, "parameter": CC},
                      filename=str(output_dir / "out_ex03_novo.xdmf"))

    u_values = u.x.array.reshape(-1, dim)
    u_mat    = np.zeros((Npoints, 3))
    for j, dof in enumerate(selected_dofs):
        u_mat[j, :] = u_values[dof, :]

    Fd = compute_Fh(domain, u, selected_dofs)

    if rank0:
        umax = np.linalg.norm(u_values, axis=1).max()
        print(f"max |u| (rank 0) = {umax:.3f}")
        print("end of synthetic data generation")

    return u_mat, Fd, CC


# =============================================================================
# entry point
# =============================================================================

if __name__ == "__main__":

    vtu_path = sys.argv[1] if len(sys.argv) > 1 else "Patient_7_lv_tissue.vtu"
    fiber_vtu = sys.argv[2] if len(sys.argv) > 2 else None

    geo = load_lv_geometry(
        vtu_path,
        fiber_source = "vtu" if fiber_vtu else "rule",
        fiber_vtu    = fiber_vtu,
        alpha_endo   = 40.0,      # same as ex03
        alpha_epi    = -50.0,     # same as ex03
    )

    u, Fd, c = prob_ventricle_passive_filling(
        geo,
        c_healthy       = 2.0,
        fibrosis_factor = 3.0,
        target_load     = -3.0,
        param_space     = "DG0",
    )
