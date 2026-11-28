"""
plot_lv_views.py
----------------
Patient LV (Case 3) seen from several directions around its long axis, one
row of panels + one colour bar (same rendering and "Fast" colour map as
plot_maps.py / plot_uq.py). Standalone; uses paper_style.py if it is in the folder.

Field sources
-------------
  --h5 out_ex06_lv_patient/map_lv_patient.h5      (default rows: C_true, C_map)
                                         top row ground truth, bottom row MAP
  --vtu Patient_7_lv_tissue.vtu          ground truth only, from the tissue labels
                                         (2 kPa healthy, 6 kPa in tissue 1, 2;
                                         nodal average = the P1 truth of ex06/ex07)
  --vtu ... --cells                      sharp cell-wise labels instead
  --h5 out_ex07_lv_patient/uq_lv_patient.h5 --field C_map
                                         any P1 field written by ex06/ex07
                                         (C_true, C_map, C_post_sample_1, m_post_std ...)

Views: --n-views (default 5) cameras equally spaced around the long axis,
base at the top. The long axis and the base side are found from the mesh;
--azimuth0 rotates the whole set (e.g. to start at the septum).

Usage:
    python plot_lv_views.py --vtu Patient_7_lv_tissue.vtu
    python plot_lv_views.py --h5 out_ex07_lv_patient/uq_lv_patient.h5 --field C_map
    python plot_lv_views.py --vtu Patient_7_lv_tissue.vtu --n-views 4 --azimuth0 30
Writes <out>.pdf/.png  (default lv_views_<field>).
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, cm

from matplotlib.colors import LinearSegmentedColormap

try:                                   # same fonts as the other paper figures, if available
    from paper_style import apply_paper_style, save_figure
except ImportError:
    def apply_paper_style(use_tex=False):
        plt.rcParams.update({"font.family": "serif",
                             "font.serif": ["Times New Roman", "Times", "Nimbus Roman",
                                            "TeX Gyre Termes", "DejaVu Serif"],
                             "mathtext.fontset": "stix", "font.size": 10})

    def save_figure(fig, stem):
        for ext in ("pdf", "png"):
            fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
        print(f"Saved: {stem}.pdf, {stem}.png")


# ParaView "Fast" colour map (as plot_maps.py)
_FAST = [(0.0000, (0.0564, 0.0564, 0.4700)), (0.1716, (0.2430, 0.4604, 0.8100)),
         (0.2985, (0.3568, 0.7450, 0.9544)), (0.4321, (0.6882, 0.9300, 0.9179)),
         (0.5000, (0.8995, 0.9446, 0.7687)), (0.5882, (0.9571, 0.8338, 0.5089)),
         (0.7061, (0.9275, 0.6214, 0.3154)), (0.8476, (0.8000, 0.3520, 0.1600)),
         (1.0000, (0.5900, 0.0767, 0.1195))]


def get_cmap(name):
    if name.lower() in ("fast", "paraview"):
        return LinearSegmentedColormap.from_list("fast", _FAST, N=256)
    return matplotlib.colormaps[name]


def crop_white(img, pad=6, thr=248):
    mask = (img < thr).any(axis=2)
    if not mask.any():
        return img
    r, c = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
    return img[max(r[0] - pad, 0):r[-1] + pad + 1, max(c[0] - pad, 0):c[-1] + pad + 1]


def read_case(h5file, fields):
    """(points, tets, {field: nodal values}) from a dolfinx XDMF/HDF5 file."""
    import h5py

    def first(g):
        for key in g:
            obj = g[key]
            if isinstance(obj, h5py.Dataset):
                return obj[()]
            sub = first(obj)
            if sub is not None:
                return sub
        return None

    with h5py.File(h5file, "r") as f:
        mesh = f["Mesh"][next(iter(f["Mesh"]))]
        pts, tets = mesh["geometry"][()], mesh["topology"][()].astype(np.int64)
        vals = {}
        for name in fields:
            if name not in f["Function"]:
                raise KeyError(f"'{name}' not in {h5file}; available: {list(f['Function'].keys())}")
            vals[name] = np.asarray(first(f["Function"][name])).reshape(-1)
    return pts, tets, vals


# =============================================================================
# data
# =============================================================================

def from_vtu(path, cells_only=False, c_healthy=2.0, factor=3.0):
    import pyvista as pv
    g = pv.read(path)
    tet = g.cells.reshape(-1, 5)[:, 1:]
    tissue = np.asarray(g.cell_data["tissue"])
    Cc = np.where(np.isin(np.rint(tissue), (1, 2)), c_healthy * factor, c_healthy)
    pts = np.asarray(g.points, float)
    if cells_only:
        return pts, tet, Cc, "cell"
    cnt = np.bincount(tet.ravel(), minlength=len(pts))
    Cn = np.bincount(tet.ravel(), weights=np.repeat(Cc, 4), minlength=len(pts)) / np.maximum(cnt, 1)
    return pts, tet, Cn, "point"


def long_axis(pts, tet):
    """Unit long axis (apex -> base) and centre. The base is the flat cut of the LV:
    the largest set of boundary faces sharing one outward normal; the axis is that
    normal. Falls back to the principal direction if no flat base is found."""
    f = np.sort(np.vstack([tet[:, [0, 1, 2]], tet[:, [0, 1, 3]],
                           tet[:, [0, 2, 3]], tet[:, [1, 2, 3]]]), axis=1)
    u, cnt = np.unique(f, axis=0, return_counts=True)
    bf = u[cnt == 1]
    x = pts[bf]
    nrm = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
    area = 0.5 * np.linalg.norm(nrm, axis=1)
    nrm /= np.maximum(2 * area[:, None], 1e-30)
    c = pts.mean(axis=0)
    fc = x.mean(axis=1)
    nrm *= np.sign(np.einsum("ij,ij->i", nrm, fc - c))[:, None]      # outward
    rng = np.random.default_rng(0)
    cand = nrm[rng.choice(len(nrm), min(400, len(nrm)), replace=False)]
    score = [(area * (nrm @ n > 0.995)).sum() for n in cand]
    best = cand[int(np.argmax(score))]
    flat = nrm @ best > 0.995
    if area[flat].sum() < 0.03 * area.sum():                          # no flat base
        _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
        return Vt[0], c
    axis = (area[flat, None] * nrm[flat]).sum(0)
    return axis / np.linalg.norm(axis), c


# =============================================================================
# rendering
# =============================================================================

def render_views(pts, tet, rows, assoc, cmap, clim, n_views, az0, elev, zoom, size=(700, 800)):
    """rows: list of value arrays on the same mesh -> list of rows of images.
    elev > 0 lifts the camera above the base plane, so the cavity and the
    endocardium are seen through the open base."""
    import pyvista as pv
    cells = np.hstack([np.full((len(tet), 1), 4), tet]).ravel()
    grid = pv.UnstructuredGrid(cells, np.full(len(tet), pv.CellType.TETRA), pts)
    surf = grid.extract_surface(algorithm=None)
    ids = surf.point_data["vtkOriginalPointIds"] if assoc == "point" else None
    cids = surf.cell_data["vtkOriginalCellIds"] if assoc == "cell" else None
    axis, centre = long_axis(pts, tet)
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = ref - (ref @ axis) * axis
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    dist = 3.0 * np.ptp(pts, axis=0).max()
    el = np.deg2rad(elev)
    angles = [(az0 + 360.0 * k / n_views) % 360 for k in range(n_views)]
    out = []
    for vals in rows:
        g = surf.copy()
        if assoc == "point":
            g.point_data["f"] = vals[ids]
        else:
            g.cell_data["f"] = vals[cids]
        imgs = []
        for ang_deg in angles:
            ang = np.deg2rad(ang_deg)
            d = np.cos(el) * (np.cos(ang) * e1 + np.sin(ang) * e2) + np.sin(el) * axis
            up = axis - (axis @ d) * d                    # base towards the top of the image
            pl = pv.Plotter(off_screen=True, window_size=size)
            pl.set_background("white")
            pl.add_mesh(g, scalars="f", cmap=cmap, clim=clim,
                        smooth_shading=assoc == "point", show_scalar_bar=False)
            pl.camera.focal_point = centre
            pl.camera.position = centre + dist * d
            pl.camera.up = up / np.linalg.norm(up)
            pl.reset_camera()
            pl.camera.zoom(zoom)
            imgs.append(crop_white(pl.screenshot(return_img=True)))
            pl.close()
        out.append(imgs)
    return out, angles


def make_figure(rows, titles, row_labels, cmap, clim, label, out):
    nr, n = len(rows), len(rows[0])
    aspect = np.median([im.shape[0] / im.shape[1] for r in rows for im in r])
    left = 0.05 if any(row_labels) else 0.01
    w_panel = (7.0 - 0.8) / n
    fig = plt.figure(figsize=(7.0, nr * (w_panel * aspect + 0.25) + 0.15))
    gs = fig.add_gridspec(nr, n + 1, width_ratios=[1] * n + [0.22], wspace=0.04, hspace=0.06,
                          left=left, right=0.93, top=0.92, bottom=0.02)
    for i, (imgs, rl) in enumerate(zip(rows, row_labels)):
        for j, img in enumerate(imgs):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(img, interpolation="lanczos")
            ax.set_axis_off()
            if i == 0:
                ax.set_title(titles[j], pad=2)
            if j == 0 and rl:
                ax.text(-0.06, 0.5, rl, transform=ax.transAxes, rotation=90,
                        ha="center", va="center")
    slot = fig.add_subplot(gs[:, n])
    slot.set_axis_off()
    pos = slot.get_position()
    h = pos.height * (0.70 if nr == 1 else 0.55)
    cax = fig.add_axes([pos.x0 + 0.10 * pos.width, pos.y0 + (pos.height - h) / 2,
                        0.30 * pos.width, h])
    cb = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(*clim), cmap=cmap), cax=cax)
    cb.set_label(label)
    cb.ax.tick_params(direction="out", length=3, width=0.6, right=True, left=False)
    cb.ax.minorticks_off()
    cb.outline.set_linewidth(0.8)
    save_figure(fig, str(out))
    plt.close(fig)


# =============================================================================
# main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vtu", default=None, help="Patient .vtu with the 'tissue' cell field "
                                                "(one row: ground truth)")
    ap.add_argument("--cells", action="store_true", help="With --vtu: sharp cell-wise labels")
    ap.add_argument("--h5", default=None, help=".h5 written by ex06/ex07 (P1 fields)")
    ap.add_argument("--fields", nargs="+", default=["C_true", "C_map"],
                    help="Fields of --h5, one row each (default: C_true C_map)")
    ap.add_argument("--row-labels", nargs="+", default=None,
                    help="Row labels (default: Ground truth / MAP / field names)")
    ap.add_argument("--label", default=None, help="Colour-bar label")
    ap.add_argument("--clim", type=float, nargs=2, default=None,
                    help="Colour range, shared by all rows (default: min/max over all rows)")
    ap.add_argument("--n-views", type=int, default=5)
    ap.add_argument("--azimuth0", type=float, default=0.0, help="Angle of the first view [deg]")
    ap.add_argument("--elevation", type=float, default=35.0,
                    help="Camera height above the base plane [deg]; >0 shows the endocardium "
                         "through the open base (0 = side views)")
    ap.add_argument("--titles", nargs="+", default=None,
                    help="Column titles (default: view angles)")
    ap.add_argument("--zoom", type=float, default=1.3)
    ap.add_argument("--cmap", default="fast")
    ap.add_argument("--use-tex", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    default_labels = {"C_true": "Ground truth", "C_map": "MAP"}
    if a.h5:
        pts, tet, vals = read_case(a.h5, a.fields)
        rows, assoc, names = [vals[f] for f in a.fields], "point", a.fields
    elif a.vtu:
        pts, tet, v, assoc = from_vtu(a.vtu, a.cells)
        rows, names = [v], ["C_true"]
    else:
        ap.error("give --vtu or --h5")
    row_labels = a.row_labels or ([default_labels.get(f, f.replace("_", " ")) for f in names]
                                  if len(rows) > 1 else [""])
    if len(row_labels) != len(rows):
        ap.error(f"--row-labels needs {len(rows)} entries")

    stiffness = all(f.startswith("C_") for f in names)
    # one colour range for all rows: the full range of everything shown, so the same
    # stiffness has the same colour in the truth and the MAP rows (truth alone: 2-6 kPa)
    if a.clim:
        clim = tuple(a.clim)
    elif len(rows) == 1 and stiffness and a.vtu:
        clim = (2.0, 6.0)
    else:
        clim = (min(float(r.min()) for r in rows), max(float(r.max()) for r in rows))
    label = a.label or (r"$C$ [kPa]" if stiffness else names[0].replace("_", " "))
    apply_paper_style(use_tex=a.use_tex)
    cmap = get_cmap(a.cmap)
    imgs, angles = render_views(pts, tet, rows, assoc, cmap, clim, a.n_views, a.azimuth0,
                                a.elevation, a.zoom)
    titles = a.titles or [f"{ang:.0f}$^\\circ$" for ang in angles]
    if len(titles) != a.n_views:
        ap.error(f"--titles needs {a.n_views} entries")
    out = a.out or ("lv_views_" + "_".join(names))
    make_figure(imgs, titles, row_labels, cmap, clim, label, out)
    print(f"  {len(rows)} row(s) x {a.n_views} views, elevation {a.elevation:g} deg, "
          f"colour range [{clim[0]:.3g}, {clim[1]:.3g}]")


if __name__ == "__main__":
    main()
