"""
Preprocessing of the patient-specific LV mesh (Patient_7_lv_tissue.vtu).

Pure numpy / scipy / meshio -- no dolfinx needed, so it can be checked on
its own. The dolfinx part is in ex03_lv_vtu_forward.py.

The .vtu only has tetrahedra + the cell field "tissue":
    0 -> healthy tissue
    1 -> fibrosis
    2 -> fibrosis (border zone, also treated as fibrosis)

Everything else is built here:

1. Boundary markers (same ids as ex03 / cardiac_geometries):
       5 = BASE  (flat cut plane)
       6 = ENDO
       7 = EPI
   The base is the largest set of coplanar boundary facets. Removing it
   splits the surface into two closed pieces; the one with the smaller
   area is the endocardium.

2. Fibres (rule-based, Bayer et al. 2012 style, LV only):
       phi : Laplace, endo = 0, epi = 1           -> transmural direction
       psi : Laplace, apex = 0, base = 1          -> apico-basal direction
       e_t = grad(phi)                            (endo -> epi)
       e_l = grad(psi) orthogonalised to e_t       (apex -> base)
       e_c = e_l x e_t                            (circumferential)
       alpha(phi) = alpha_endo + (alpha_epi - alpha_endo) * phi
       f0 = cos(alpha) e_c + sin(alpha) e_l
       s0 = e_t ,  n0 = f0 x s0
   Default angles are those of ex03 (+40 endo, -50 epi).
   Alternatively the fibres can be copied (nearest cell) from the
   biventricular Patient_7_UVC.vtu (fields f_0, s_0, n_0).
"""

from dataclasses import dataclass
import numpy as np
import meshio
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

MARKERS = {
    "BASE": (5, 2),
    "ENDO": (6, 2),
    "EPI":  (7, 2),
}

TISSUE_HEALTHY = 0
FIBROSIS_LABELS = (1, 2)


@dataclass
class LVData:
    points: np.ndarray          # (Np, 3)
    cells: np.ndarray           # (Nc, 4)
    cell_centroids: np.ndarray  # (Nc, 3)
    tissue: np.ndarray          # (Nc,) int
    bnd_facets: np.ndarray      # (Nf, 3)
    bnd_markers: np.ndarray     # (Nf,)
    bnd_midpoints: np.ndarray   # (Nf, 3)
    f0: np.ndarray = None       # (Nc, 3)
    s0: np.ndarray = None
    n0: np.ndarray = None
    phi: np.ndarray = None      # transmural coordinate at nodes (0 endo, 1 epi)
    psi: np.ndarray = None      # apico-basal Laplace solution (for fibres)
    height: np.ndarray = None   # geometric height at nodes (0 apex, 1 base)
    frame: dict = None          # e_t, e_l, e_c per cell


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _normalize(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def _orient_tets(points, cells):
    """Make every tet positively oriented (det > 0)."""
    x = points[cells]
    det = np.einsum("ij,ij->i", np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]),
                    x[:, 3] - x[:, 0])
    cells = cells.copy()
    neg = det < 0
    cells[neg] = cells[neg][:, [0, 2, 1, 3]]
    return cells


def boundary_facets(cells):
    faces = np.vstack([cells[:, [1, 2, 3]], cells[:, [0, 2, 3]],
                       cells[:, [0, 1, 3]], cells[:, [0, 1, 2]]])
    faces = np.sort(faces, axis=1)
    uniq, counts = np.unique(faces, axis=0, return_counts=True)
    return uniq[counts == 1]


def _facet_components(facets):
    nf = len(facets)
    edges = np.sort(np.vstack([facets[:, [0, 1]], facets[:, [1, 2]],
                               facets[:, [0, 2]]]), axis=1)
    fid = np.tile(np.arange(nf), 3)
    key = edges[:, 0] * (int(facets.max()) + 1) + edges[:, 1]
    order = np.argsort(key, kind="stable")
    key, fid = key[order], fid[order]
    same = key[1:] == key[:-1]
    a, b = fid[:-1][same], fid[1:][same]
    adj = coo_matrix((np.ones(len(a)), (a, b)), shape=(nf, nf))
    return connected_components(adj, directed=False)


def _facet_normals_areas(points, facets):
    x = points[facets]
    c = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
    a = 0.5 * np.linalg.norm(c, axis=1)
    return _normalize(c), a


# -----------------------------------------------------------------------------
# boundary classification
# -----------------------------------------------------------------------------

def classify_boundary(points, facets, plane_tol=None, normal_tol=0.999):
    """
    Base = largest planar patch of the boundary (the cut plane).
    Then: 2 remaining components, smaller area -> endo, larger -> epi.
    """
    normals, areas = _facet_normals_areas(points, facets)
    mids = points[facets].mean(axis=1)
    if plane_tol is None:
        plane_tol = 1e-3 * np.ptp(points, axis=0).max()

    # candidate plane normals: area-weighted vote among facet normals
    # (sign-independent). Test the normal of every facet against the
    # 200 largest facets and keep the best plane.
    order = np.argsort(-areas)[:200]
    best, best_area = None, -1.0
    for i in order:
        n, d = normals[i], normals[i] @ mids[i]
        on = (np.abs(normals @ n) > normal_tol) & \
             (np.abs(points[facets] @ n - d).max(axis=1) < plane_tol)
        a = areas[on].sum()
        if a > best_area:
            best, best_area = on, a
    base = best

    markers = np.full(len(facets), MARKERS["BASE"][0], dtype=np.int32)
    idx = np.where(~base)[0]
    ncomp, lab = _facet_components(facets[idx])
    if ncomp != 2:
        raise RuntimeError(f"expected 2 surfaces (endo/epi) after removing "
                           f"the base, got {ncomp}")
    comp_area = np.array([areas[idx[lab == c]].sum() for c in range(2)])
    endo = int(np.argmin(comp_area))
    markers[idx] = np.where(lab == endo, MARKERS["ENDO"][0],
                            MARKERS["EPI"][0])
    return markers


# -----------------------------------------------------------------------------
# P1 Laplace solver (numpy / scipy)
# -----------------------------------------------------------------------------

def _p1_gradients(points, cells):
    """Gradients of the 4 barycentric functions per tet + volumes."""
    x = points[cells]
    M = np.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]],
                 axis=1)                                 # (Nc,3,3) rows
    Minv = np.linalg.inv(M)                              # cols = grad l1..l3
    G = np.empty((len(cells), 4, 3))
    G[:, 1:, :] = np.transpose(Minv, (0, 2, 1))
    G[:, 0, :] = -G[:, 1:, :].sum(axis=1)
    vol = np.abs(np.linalg.det(M)) / 6.0
    return G, vol


def laplace(points, cells, dirichlet_nodes, dirichlet_vals, G=None, vol=None):
    if G is None:
        G, vol = _p1_gradients(points, cells)
    Ke = vol[:, None, None] * np.einsum("eik,ejk->eij", G, G)
    rows = np.repeat(cells, 4, axis=1).ravel()
    cols = np.tile(cells, (1, 4)).ravel()
    n = len(points)
    K = coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()

    u = np.zeros(n)
    u[dirichlet_nodes] = dirichlet_vals
    free = np.ones(n, bool)
    free[dirichlet_nodes] = False
    rhs = -K[free][:, ~free] @ u[~free]
    u[free] = spsolve(K[free][:, free].tocsc(), rhs)
    return u


# -----------------------------------------------------------------------------
# fibres
# -----------------------------------------------------------------------------

def rule_based_fibers(points, cells, facets, markers,
                      alpha_endo=40.0, alpha_epi=-50.0, apex_radius=2.0):
    G, vol = _p1_gradients(points, cells)

    endo_n = np.unique(facets[markers == MARKERS["ENDO"][0]])
    epi_n = np.unique(facets[markers == MARKERS["EPI"][0]])
    base_n = np.unique(facets[markers == MARKERS["BASE"][0]])
    # nodes on the base rim belong to base + endo/epi; keep them for phi
    epi_only = np.setdiff1d(epi_n, endo_n)

    # transmural: endo = 0, epi = 1
    phi = laplace(points, cells,
                  np.concatenate([endo_n, epi_only]),
                  np.concatenate([np.zeros(len(endo_n)),
                                  np.ones(len(epi_only))]), G, vol)

    # apex = epi node farthest from the base plane
    bn, _ = _facet_normals_areas(points, facets[markers == MARKERS["BASE"][0]])
    nb = _normalize(bn.mean(axis=0))
    base_c = points[base_n].mean(axis=0)
    h = (points[epi_n] - base_c) @ nb
    apex_node = epi_n[np.argmax(np.abs(h))]
    apex = points[apex_node]
    apex_nodes = np.where(np.linalg.norm(points - apex, axis=1)
                          < apex_radius)[0]
    apex_nodes = np.intersect1d(apex_nodes, epi_n)
    if len(apex_nodes) == 0:
        apex_nodes = np.array([apex_node])
    psi = laplace(points, cells,
                  np.concatenate([apex_nodes, base_n]),
                  np.concatenate([np.zeros(len(apex_nodes)),
                                  np.ones(len(base_n))]), G, vol)

    # per-cell frame
    grad_phi = np.einsum("ei,eik->ek", phi[cells], G)
    grad_psi = np.einsum("ei,eik->ek", psi[cells], G)
    long_axis = _normalize(base_c - apex)

    e_t = _normalize(grad_phi)
    e_l = grad_psi - np.einsum("ij,ij->i", grad_psi, e_t)[:, None] * e_t
    weak = np.linalg.norm(e_l, axis=1) < 1e-3 * np.linalg.norm(grad_psi,
                                                                axis=1).max()
    fallback = long_axis - (e_t @ long_axis)[:, None] * e_t
    e_l[weak] = fallback[weak]
    e_l = _normalize(e_l)
    e_c = _normalize(np.cross(e_l, e_t))

    a = np.deg2rad(alpha_endo + (alpha_epi - alpha_endo) * phi[cells].mean(1))
    f0 = _normalize(np.cos(a)[:, None] * e_c + np.sin(a)[:, None] * e_l)
    s0 = e_t
    n0 = _normalize(np.cross(f0, s0))
    frame = dict(e_t=e_t, e_l=e_l, e_c=e_c)
    return f0, s0, n0, phi, psi, frame


def transfer_fibers(centroids, vtu_path):
    """Nearest-cell copy of f_0 / s_0 / n_0 from another .vtu."""
    m = meshio.read(vtu_path)
    P = m.points
    T = m.cells_dict["tetra"]
    tree = cKDTree(P[T].mean(axis=1))
    dist, j = tree.query(centroids)
    out = []
    for name in ("f_0", "s_0", "n_0"):
        out.append(_normalize(np.asarray(m.cell_data_dict[name]["tetra"])[j]))
    return (*out, dist)


def helix_angle(f, frame):
    """Helix angle in degrees, in [-90, 90] (fibre sign is irrelevant)."""
    fc = np.einsum("ij,ij->i", f, frame["e_c"])
    fl = np.einsum("ij,ij->i", f, frame["e_l"])
    a = np.degrees(np.arctan2(fl, fc))
    a = np.where(a > 90, a - 180, a)
    a = np.where(a < -90, a + 180, a)
    return a


# -----------------------------------------------------------------------------
# public API
# -----------------------------------------------------------------------------

def load_lv_vtu(path, fiber_source="rule", fiber_vtu=None,
                alpha_endo=40.0, alpha_epi=-50.0):
    """
    fiber_source = "rule" : rule-based fibres (alpha_endo / alpha_epi)
                   "vtu"  : copy f_0/s_0/n_0 from fiber_vtu (nearest cell)
    """
    m = meshio.read(path)
    pts = np.asarray(m.points, dtype=np.float64)
    cells = _orient_tets(pts, np.asarray(m.cells_dict["tetra"], np.int64))
    tissue = np.rint(np.asarray(m.cell_data_dict["tissue"]["tetra"])
                     ).astype(np.int32)

    facets = boundary_facets(cells)
    markers = classify_boundary(pts, facets)

    d = LVData(points=pts, cells=cells, cell_centroids=pts[cells].mean(1),
               tissue=tissue, bnd_facets=facets, bnd_markers=markers,
               bnd_midpoints=pts[facets].mean(1))

    f0, s0, n0, phi, psi, frame = rule_based_fibers(
        pts, cells, facets, markers, alpha_endo, alpha_epi)
    d.phi, d.psi, d.frame = phi, psi, frame

    # geometric apico-basal height: 1 on the base plane, 0 at the apex
    bf = facets[markers == MARKERS["BASE"][0]]
    bn, _ = _facet_normals_areas(pts, bf)
    nb = _normalize(bn.mean(axis=0))
    h = (pts - pts[np.unique(bf)].mean(axis=0)) @ nb
    h = h * np.sign(-h[np.argmax(np.abs(h))])      # apex at negative h
    d.height = 1.0 + h / np.abs(h.min())

    if fiber_source == "rule":
        d.f0, d.s0, d.n0 = f0, s0, n0
    elif fiber_source == "vtu":
        if fiber_vtu is None:
            raise ValueError("fiber_source='vtu' needs fiber_vtu=<path>")
        d.f0, d.s0, d.n0, dist = transfer_fibers(d.cell_centroids, fiber_vtu)
        print(f"fibres copied from {fiber_vtu}: nearest-cell distance "
              f"median {np.median(dist):.2f}, max {dist.max():.2f}")
    else:
        raise ValueError("fiber_source must be 'rule' or 'vtu'")
    return d


def stiffness_per_cell(tissue, c_healthy=2.0, fibrosis_factor=3.0):
    """C = c_healthy in healthy tissue, c_healthy*factor in tissue 1 and 2."""
    C = np.full(tissue.shape, c_healthy, dtype=np.float64)
    C[np.isin(tissue, FIBROSIS_LABELS)] *= fibrosis_factor
    return C


def cell_volumes(points, cells):
    x = points[cells]
    return np.abs(np.einsum("ij,ij->i",
                            np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]),
                            x[:, 3] - x[:, 0])) / 6.0


def region_stats(fields, vol, tissue, height_cell, base_cut=0.9):
    """
    Statistics of cell-wise scalar fields per tissue region.

    fields   : dict name -> (Nc,) values
    vol      : (Nc,) cell volumes (weights for the mean)
    tissue   : (Nc,) labels 0 / 1 / 2
    height_cell : (Nc,) apico-basal height (0 apex, 1 base)
    base_cut : cells with height > base_cut are also reported separately
               excluded ("no base"), since the fixed base creates
               artificial stress concentrations.

    Returns a list of dict rows (one per field x region x subset).
    """
    fib = np.isin(tissue, FIBROSIS_LABELS)
    regions = {
        "healthy (0)":    tissue == TISSUE_HEALTHY,
        "fibrosis (1+2)": fib,
        "fibrosis core (1)": tissue == 1,
        "fibrosis border (2)": tissue == 2,
    }
    subsets = {"all": np.ones_like(fib), "no base": height_cell <= base_cut}

    rows = []
    for fname, val in fields.items():
        for sname, smask in subsets.items():
            ref = None
            for rname, rmask in regions.items():
                m = rmask & smask
                if not m.any():
                    continue
                w = vol[m]
                mean = float((val[m] * w).sum() / w.sum())
                if rname.startswith("healthy"):
                    ref = mean
                rows.append(dict(
                    field=fname, subset=sname, region=rname,
                    n_cells=int(m.sum()), volume=float(w.sum()),
                    mean=mean,
                    median=float(np.median(val[m])),
                    p95=float(np.percentile(val[m], 95)),
                    max=float(val[m].max()),
                    ratio_to_healthy=(mean / ref if ref not in (None, 0.0)
                                      else np.nan),
                ))
    return rows


def format_stats(rows):
    head = (f"{'field':<10} {'subset':<8} {'region':<20} {'n':>6} "
            f"{'mean':>10} {'median':>10} {'p95':>10} {'ratio':>7}")
    out = [head, "-" * len(head)]
    for r in rows:
        out.append(f"{r['field']:<10} {r['subset']:<8} {r['region']:<20} "
                   f"{r['n_cells']:>6} {r['mean']:>10.4g} {r['median']:>10.4g} "
                   f"{r['p95']:>10.4g} {r['ratio_to_healthy']:>7.2f}")
    return "\n".join(out)


def write_stats_csv(rows, path):
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summary(d: LVData):
    names = {v[0]: k for k, v in MARKERS.items()}
    lines = [f"points: {len(d.points)}, tets: {len(d.cells)}"]
    for t, n in zip(*np.unique(d.tissue, return_counts=True)):
        kind = "fibrosis" if t in FIBROSIS_LABELS else "healthy"
        lines.append(f"tissue {t} ({kind}): {n} cells")
    for mk, n in zip(*np.unique(d.bnd_markers, return_counts=True)):
        lines.append(f"facet marker {mk} ({names[mk]}): {n} facets")
    return "\n".join(lines)


def write_check_vtu(d: LVData, path):
    """Volume file with tissue, C, fibres, phi/psi for ParaView."""
    meshio.write(path, meshio.Mesh(
        d.points, [("tetra", d.cells)],
        point_data={"phi_transmural": d.phi, "psi_apicobasal": d.psi},
        cell_data={"tissue": [d.tissue.astype(float)],
                   "C": [stiffness_per_cell(d.tissue)],
                   "f0": [d.f0], "s0": [d.s0], "n0": [d.n0],
                   "helix_angle": [helix_angle(d.f0, d.frame)]}))


def write_boundary_vtu(d: LVData, path):
    meshio.write(path, meshio.Mesh(
        d.points, [("triangle", d.bnd_facets)],
        cell_data={"marker": [d.bnd_markers.astype(float)]}))
