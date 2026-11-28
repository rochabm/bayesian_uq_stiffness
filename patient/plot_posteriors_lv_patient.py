"""
plot_posteriors_lv_patient.py
-----------------------------
Look at the posterior of Case 3 (patient LV) in more detail, from the
ex07_uq_lv_patient outputs (no dolfinx, nothing recomputed):

1) samples figure (<out>_samples.pdf/.png)
       top    : ground truth | prior samples 1..N        (C = exp(m0 + z))
       bottom : MAP          | posterior samples 1..N    (the SAME z, conditioned on the data)
   Same camera and colour scale for all panels, so you see what the data change
   in each prior draw. N = number of samples written by ex07 (--n-samples there).

2) pointwise distributions (<out>_nodes.pdf/.png)
   Prior and posterior density of C at a few nodes (Laplace approximation:
   log-normal, m ~ N(m_MAP, s_post^2), prior m ~ N(m0, s_prior^2)), with the
   truth and the MAP marked:
       "near point"  : the node closest to a measurement point inside the fibrosis
       "fibrosis"    : the fibrotic node farthest from all measurement points
       "healthy"     : the healthy node farthest from all measurement points
       "border"      : a border node close to a measurement point
   or your own nodes with --nodes i j k.

Usage:
    python plot_posteriors_lv_patient.py --root out_ex07_lv_patient_n128
    python plot_posteriors_lv_patient.py --root out_ex07_lv_patient_n128 --azimuth 216
Needs plot_lv_views.py and plot_uq_lv_patient.py in the same folder.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from plot_lv_views import read_case, get_cmap, apply_paper_style, save_figure
from plot_uq_lv_patient import renderer, add_colorbar, show


def lognormal_pdf(c, mu, s):
    return np.exp(-(np.log(c) - mu) ** 2 / (2 * s ** 2)) / (c * s * np.sqrt(2 * np.pi))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="out_ex07_lv_patient")
    ap.add_argument("--n-samples", type=int, default=None,
                    help="Samples to show (default: all found in the .h5)")
    ap.add_argument("--clim", type=float, nargs=2, default=[1.5, 8.0])
    ap.add_argument("--nodes", type=int, nargs="+", default=None,
                    help="Node indices for the distribution plot (default: picked automatically)")
    ap.add_argument("--C0", type=float, default=2.0, help="Prior mean in kPa (as ex07)")
    ap.add_argument("--azimuth", type=float, default=0.0)
    ap.add_argument("--elevation", type=float, default=35.0)
    ap.add_argument("--zoom", type=float, default=1.3)
    ap.add_argument("--cmap", default="fast")
    ap.add_argument("--use-tex", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = Path(a.root)
    h5 = root / "uq_lv_patient.h5"
    import h5py
    with h5py.File(h5, "r") as f:
        names = list(f["Function"].keys())
    n_found = sum(1 for k in names if k.startswith("C_post_sample_"))
    ns = min(a.n_samples or n_found, n_found)
    if ns == 0:
        raise SystemExit(f"no C_post_sample_* in {h5}")
    fields = (["C_true", "C_map", "m_true", "m_map", "m_prior_std", "m_post_std", "region"]
              + [f"C_prior_sample_{s}" for s in range(1, ns + 1)]
              + [f"C_post_sample_{s}" for s in range(1, ns + 1)])
    pts, tet, v = read_case(h5, fields)
    out = a.out or str(root / "posteriors_lv_patient")
    apply_paper_style(use_tex=a.use_tex)
    cmap = get_cmap(a.cmap)
    clim = tuple(a.clim)

    # ------------------------------------------------------------------ 1) samples
    render = renderer(pts, tet, a.azimuth, a.elevation, a.zoom)
    top = [render(v["C_true"], cmap, clim)] + \
          [render(v[f"C_prior_sample_{s}"], cmap, clim) for s in range(1, ns + 1)]
    bot = [render(v["C_map"], cmap, clim)] + \
          [render(v[f"C_post_sample_{s}"], cmap, clim) for s in range(1, ns + 1)]
    nc = ns + 1
    aspect = np.median([im.shape[0] / im.shape[1] for im in top + bot])
    w_panel = (7.2 - 0.6) / nc
    fig = plt.figure(figsize=(7.2, 2 * (w_panel * aspect + 0.3) + 0.2))
    gs = fig.add_gridspec(2, nc + 1, width_ratios=[1] * nc + [0.45], wspace=0.04, hspace=0.12,
                          left=0.05, right=0.99, top=0.93, bottom=0.02)
    for i, (row, first, lab) in enumerate([(top, "Ground truth", "Prior"),
                                           (bot, "MAP", "Posterior")]):
        for j, img in enumerate(row):
            ax = fig.add_subplot(gs[i, j])
            show(ax, img, first if j == 0 else f"{lab} sample {j}")
            if j == 0:
                ax.text(-0.08, 0.5, lab, transform=ax.transAxes, rotation=90,
                        ha="center", va="center")
    add_colorbar(fig, gs[:, nc], clim, cmap, r"$C$ [kPa]")
    save_figure(fig, out + "_samples")
    plt.close(fig)

    # ------------------------------------------------------------------ 2) nodes
    region = np.rint(v["region"]).astype(int)            # 0 healthy, 1 border, 2 fibrosis
    sensors = None
    npz = root / "uq_lv_patient.npz"
    if npz.exists():
        z = np.load(npz)
        if "sensor_xyz" in z.files:
            sensors = z["sensor_xyz"]
    if a.nodes:
        picks = [(f"node {k}", k) for k in a.nodes]
    else:
        if sensors is None:
            raise SystemExit("no sensor_xyz in uq_lv_patient.npz: give --nodes i j k")
        d_s, _ = cKDTree(sensors).query(pts)             # distance to the nearest point
        picks = []
        for lab, reg, near in [("fibrosis, near a point", 2, True),
                               ("fibrosis, far from points", 2, False),
                               ("border, near a point", 1, True),
                               ("healthy, far from points", 0, False)]:
            idx = np.where(region == reg)[0]
            if len(idx):
                k = idx[np.argmin(d_s[idx])] if near else idx[np.argmax(d_s[idx])]
                picks.append((f"{lab} ({d_s[k]:.0f} mm)", int(k)))
    m0 = np.log(a.C0)
    fig, axes = plt.subplots(1, len(picks), figsize=(7.2, 2.0), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, (lab, k) in zip(axes, picks):
        sp, sq = v["m_prior_std"][k], v["m_post_std"][k]
        hi = float(np.exp(max(m0 + 3 * sp, v["m_map"][k] + 3 * sq, v["m_true"][k] + 0.3)))
        c = np.linspace(0.05, hi, 400)
        ax.plot(c, lognormal_pdf(c, m0, sp), color="0.6", lw=1.0, label="prior")
        ax.plot(c, lognormal_pdf(c, v["m_map"][k], sq), color="#0072BD", lw=1.4, label="posterior")
        ax.axvline(v["C_true"][k], color="black", ls="--", lw=0.9, label="truth")
        ax.axvline(v["C_map"][k], color="#D95319", lw=1.0, label="MAP")
        ax.set_title(lab, fontsize=8, pad=2)
        ax.set_xlabel(r"$C$ [kPa]")
        ax.set_yticks([])
        print(f"  node {k}: {lab}: truth {v['C_true'][k]:.2f}, MAP {v['C_map'][k]:.2f} kPa, "
              f"s_prior {sp:.3f}, s_post {sq:.3f} (std reduction {1 - sq / sp:.2f})")
    axes[0].set_ylabel("density")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=0.6)
    save_figure(fig, out + "_nodes")
    plt.close(fig)


if __name__ == "__main__":
    main()
