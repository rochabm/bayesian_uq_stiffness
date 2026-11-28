"""
plot_uq_lv_patient.py
---------------------
Case 3 (patient LV) UQ figure, same layout as the Case 2 figure (plot_scarce.py):

  top    : ground truth | MAP | posterior samples 1..4      (one colour scale, C [kPa])
           --bounds: ground truth | MAP | 5th | 95th percentile of C (pointwise 90%
           credible interval, exp(m_MAP -/+ 1.645 s_post); --ci-level) [| samples]
           --ci-width: ground truth | MAP | width of the 90% credible interval [kPa]
  bottom : P(C > C_th) | std reduction 1 - s_post/s_prior (0-1, viridis, measurement
           points as red dots) | eigenvalue spectrum

One camera for all hearts: raised --elevation degrees above the base plane so the
endocardium is seen through the open base; --azimuth turns it around the long axis
(use plot_lv_views.py to pick the angle that shows the most fibrosis).

Reads the ex07_uq_lv_patient outputs (no dolfinx):
    <root>/uq_lv_patient.h5, eigenvalues_lv_patient.txt, uq_lv_patient.npz (dots)
Needs plot_lv_views.py in the same folder (mesh reading, long axis, colour map).

Usage:
    python plot_uq_lv_patient.py
    python plot_uq_lv_patient.py --azimuth 216 --elevation 35
Writes <out>.pdf/.png (default <root>/uq_lv_patient_panel).
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from scipy.special import ndtri

from plot_lv_views import (read_case, crop_white, get_cmap, long_axis,
                           apply_paper_style, save_figure)


def renderer(pts, tet, azimuth, elev, zoom, size=(700, 800)):
    import pyvista as pv
    cells = np.hstack([np.full((len(tet), 1), 4), tet]).ravel()
    surf = pv.UnstructuredGrid(cells, np.full(len(tet), pv.CellType.TETRA),
                               pts).extract_surface(algorithm=None)
    ids = surf.point_data["vtkOriginalPointIds"]
    axis, centre = long_axis(pts, tet)
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = ref - (ref @ axis) * axis
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    a, el = np.deg2rad(azimuth), np.deg2rad(elev)
    d = np.cos(el) * (np.cos(a) * e1 + np.sin(a) * e2) + np.sin(el) * axis
    up = axis - (axis @ d) * d
    dist = 3.0 * np.ptp(pts, axis=0).max()

    def render(vals, cmap, clim, points=None):
        g = surf.copy()
        g.point_data["f"] = vals[ids]
        pl = pv.Plotter(off_screen=True, window_size=size)
        pl.set_background("white")
        pl.add_mesh(g, scalars="f", cmap=cmap, clim=clim, smooth_shading=True,
                    show_scalar_bar=False)
        if points is not None and len(points):
            pl.add_points(np.asarray(points), color="red", point_size=9,
                          render_points_as_spheres=True)
        pl.camera.focal_point = centre
        pl.camera.position = centre + dist * d
        pl.camera.up = up / np.linalg.norm(up)
        pl.reset_camera()
        pl.camera.zoom(zoom)
        img = crop_white(pl.screenshot(return_img=True))
        pl.close()
        return img
    return render


def add_colorbar(fig, slot_spec, clim, cmap, label):
    slot = fig.add_subplot(slot_spec)
    slot.set_axis_off()
    pos = slot.get_position()
    h = pos.height * 0.75
    cax = fig.add_axes([pos.x0 + 0.02 * pos.width, pos.y0 + (pos.height - h) / 2,
                        0.13 * pos.width, h])
    cax.set_zorder(5)
    cb = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(*clim), cmap=cmap), cax=cax)
    cb.set_label(label)
    cb.ax.tick_params(direction="out", length=3, width=0.6, right=True, left=False)
    cb.ax.minorticks_off()
    cb.outline.set_linewidth(0.8)


def show(ax, img, title):
    ax.imshow(img, interpolation="lanczos")
    ax.set_axis_off()
    ax.set_title(title, pad=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="out_ex07_lv_patient")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--bounds", action="store_true",
                    help="Top row: truth | MAP | lower and upper percentile of the pointwise "
                         "--ci-level credible interval of C | --bound-samples samples")
    ap.add_argument("--ci-width", action="store_true",
                    help="Top row: truth | MAP | width of the pointwise --ci-level credible "
                         "interval of C in kPa (one panel, own colour bar)")
    ap.add_argument("--ci-level", type=float, default=90.0,
                    help="Credible level in %% for --bounds / --ci-width (default 90)")
    ap.add_argument("--ci-cmap", default="fast",
                    help="Colour map of the CI-width panel (default: same as C)")
    ap.add_argument("--bound-samples", type=int, default=0,
                    help="With --bounds: posterior samples appended to the top row (default 0)")
    ap.add_argument("--c-th", type=float, default=4.0, help="Only for the panel title")
    ap.add_argument("--clim", type=float, nargs=2, default=None,
                    help="C range of the top row, e.g. 1.5 8 (default: percentiles, see --clip-pct)")
    ap.add_argument("--clip-pct", type=float, nargs=2, default=[1.0, 99.0],
                    help="Without --clim: C range = these percentiles of the top-row values "
                         "(default 1 99); values outside are shown in the end colours")
    ap.add_argument("--sr-clim", type=float, nargs=2, default=[0.0, 1.0])
    ap.add_argument("--azimuth", type=float, default=0.0)
    ap.add_argument("--elevation", type=float, default=35.0)
    ap.add_argument("--zoom", type=float, default=1.3)
    ap.add_argument("--no-dots", action="store_true")
    ap.add_argument("--cmap", default="fast")
    ap.add_argument("--sr-cmap", default="viridis")
    ap.add_argument("--use-tex", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = Path(a.root)
    if a.bounds and a.ci_width:
        ap.error("use either --bounds or --ci-width")
    if a.bounds:                      # truth | MAP | lower / upper percentile | [samples]
        samples = [f"C_post_sample_{s}" for s in range(1, a.bound_samples + 1)]
        extra = ["m_map", "m_post_std"]
    elif a.ci_width:                  # truth | MAP | CI width
        samples, extra = [], ["m_map", "m_post_std"]
    else:
        samples = [f"C_post_sample_{s}" for s in range(1, a.n_samples + 1)]
        extra = []
    pts, tet, v = read_case(root / "uq_lv_patient.h5",
                            ["C_true", "C_map", "prob_C_above_threshold", "std_reduction"]
                            + samples + extra)
    p_lo = (100.0 - a.ci_level) / 2.0                     # e.g. 5 and 95 for 90 %
    p_hi = 100.0 - p_lo
    if a.bounds or a.ci_width:
        zq = float(ndtri(p_hi / 100.0))                   # 1.645 for 90 %, 1.960 for 95 %
        v["C_lo"] = np.exp(v["m_map"] - zq * v["m_post_std"])
        v["C_hi"] = np.exp(v["m_map"] + zq * v["m_post_std"])
        v["C_ciw"] = v["C_hi"] - v["C_lo"]
        print(f"  {a.ci_level:g}% credible interval: percentiles {p_lo:g}/{p_hi:g}, z = {zq:.3f}; "
              f"width [{v['C_ciw'].min():.2f}, {v['C_ciw'].max():.2f}] kPa")
    dots = None
    npz = root / "uq_lv_patient.npz"
    if not a.no_dots and npz.exists():
        z = np.load(npz)
        if "sensor_xyz" in z.files:
            dots = z["sensor_xyz"]
        else:
            print("  uq_lv_patient.npz has no sensor_xyz (older ex07): no dots")

    apply_paper_style(use_tex=a.use_tex)
    cmap, sr_cmap = get_cmap(a.cmap), get_cmap(a.sr_cmap)
    render = renderer(pts, tet, a.azimuth, a.elevation, a.zoom)

    pct = lambda q: f"{q:g}th percentile"
    if a.bounds:
        top = ["C_true", "C_map", "C_lo", "C_hi"] + samples
        top_titles = (["Ground truth", "MAP", pct(p_lo), pct(p_hi)]
                      + [f"Sample {s}" for s in range(1, a.bound_samples + 1)])
    elif a.ci_width:
        top = ["C_true", "C_map"]
        top_titles = ["Ground truth", "MAP"]
    else:
        top = ["C_true", "C_map"] + samples
        top_titles = ["Ground truth", "MAP"] + [f"Sample {s}" for s in range(1, a.n_samples + 1)]
    if a.clim:
        clim = tuple(a.clim)
    else:                    # robust range: a few extreme sample values would wash out the rest
        allv = np.concatenate([v[f] for f in top])
        clim = tuple(float(x) for x in np.percentile(allv, a.clip_pct))
    print(f"  top row: C colour range [{clim[0]:.2f}, {clim[1]:.2f}] kPa")
    top_imgs = [render(v[f], cmap, clim) for f in top]
    p_img = render(v["prob_C_above_threshold"], cmap, (0.0, 1.0))
    sr_img = render(v["std_reduction"], sr_cmap, tuple(a.sr_clim), points=dots)

    nt = len(top_imgs) if not a.ci_width else 5
    aspect = np.median([im.shape[0] / im.shape[1] for im in top_imgs + [p_img, sr_img]])
    w_fig = 7.2
    w_panel = (w_fig - 0.6) / nt
    h_row = w_panel * aspect + 0.30
    fig = plt.figure(figsize=(w_fig, 2 * h_row + 0.55))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.12], hspace=0.12,
                             left=0.01, right=0.99, top=0.95, bottom=0.08)
    if a.ci_width:
        # same column layout as the bottom row: two hearts + bar | width panel + bar | blank
        w_clim = clim                         # same limits and colours as truth / MAP
        w_img = render(v["C_ciw"], get_cmap(a.ci_cmap), w_clim)
        gt = outer[0].subgridspec(1, 6, width_ratios=[1, 1, 0.5, 1, 0.5, 1.2], wspace=0.04)
        show(fig.add_subplot(gt[0, 0]), top_imgs[0], top_titles[0])
        show(fig.add_subplot(gt[0, 1]), top_imgs[1], top_titles[1])
        add_colorbar(fig, gt[0, 2], clim, cmap, r"$C$ [kPa]")
        show(fig.add_subplot(gt[0, 3]), w_img, f"{a.ci_level:g}% CI width")
        add_colorbar(fig, gt[0, 4], w_clim, get_cmap(a.ci_cmap), "width [kPa]")
    else:
        gt = outer[0].subgridspec(1, nt + 1, width_ratios=[1] * nt + [0.45], wspace=0.04)
        for j, (img, t) in enumerate(zip(top_imgs, top_titles)):
            show(fig.add_subplot(gt[0, j]), img, t)
        add_colorbar(fig, gt[0, nt], clim, cmap, r"$C$ [kPa]")

    gb = outer[1].subgridspec(1, 5, width_ratios=[1, 0.5, 1, 0.5, 2.2], wspace=0.04)
    show(fig.add_subplot(gb[0, 0]), p_img, rf"$P(C>{a.c_th:g}$ kPa$)$")
    add_colorbar(fig, gb[0, 1], (0.0, 1.0), cmap, "probability")
    show(fig.add_subplot(gb[0, 2]), sr_img, "Std. reduction")
    add_colorbar(fig, gb[0, 3], tuple(a.sr_clim), sr_cmap,
                 r"$1 - s_{\mathrm{post}}/s_{\mathrm{prior}}$")

    slot = fig.add_subplot(gb[0, 4])
    slot.set_axis_off()
    pos = slot.get_position()
    ax = fig.add_axes([pos.x0 + 0.22 * pos.width, pos.y0 + 0.16 * pos.height,
                       0.62 * pos.width, 0.72 * pos.height])
    f = root / "eigenvalues_lv_patient.txt"
    if f.exists():
        lam = np.loadtxt(f)
        lam = lam[lam > 0]
        ax.semilogy(np.arange(1, len(lam) + 1), lam, lw=1.0, marker="o",
                    markevery=max(1, len(lam) // 8), markersize=3, markerfacecolor="white",
                    markeredgewidth=0.7, color="#0072BD")
        ax.set_xlim(0, len(lam) + 1)
        print(f"  {int(np.sum(lam > 1))} eigenvalues > 1 (of {len(lam)})")
    ax.axhline(1.0, color="black", lw=0.7, ls="--")
    ax.set_xlabel(r"number")
    ax.set_ylabel(r"$\lambda_i$")
    ax.set_title("Spectrum", pad=2)

    out = a.out or str(root / "uq_lv_patient_panel")
    save_figure(fig, out)
    plt.close(fig)


if __name__ == "__main__":
    main()
