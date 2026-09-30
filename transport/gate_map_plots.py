#!/usr/bin/env python3
"""Figures + numbers for the gate-voltage map report (reads gate_map*.npz)."""
import json
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else "report"
d = np.load("gate_map.npz")
o = np.load("gate_map_out.npz")
T, Vgs, Es = d["T"], d["Vgs"], d["Es"]
Vsds, G, I = o["Vsds"], o["G"], o["I"]

# --- resonance lines: E_res = c_n - alpha_eff*Vg -> group peaks by c = E + Vg
pts = []
for i, Vg in enumerate(Vgs):
    t = T[i]
    for j in range(1, len(Es) - 1):
        if t[j] > t[j - 1] and t[j] >= t[j + 1] and t[j] > 0.3:
            pts.append((Vg, Es[j]))
pts = np.array(pts)
c = pts[:, 0] + pts[:, 1]
order = np.argsort(c)
groups, cur = [], [order[0]]
for k in order[1:]:
    if c[k] - c[cur[-1]] < 0.05:
        cur.append(k)
    else:
        groups.append(cur)
        cur = [k]
groups.append(cur)
levels = []
for g in groups:
    if len(g) < 15:
        continue
    P = pts[g]
    slope, icpt = np.polyfit(P[:, 0], P[:, 1], 1)
    levels.append(dict(c=float(icpt), slope=float(slope), n=len(g)))
levels.sort(key=lambda x: x["c"])
cs = np.array([l["c"] for l in levels])
slopes = np.array([l["slope"] for l in levels])

# --- diamond-edge check: on the G map, ridge |Vsd| vs Vg for one level
ridge = []
for l in levels:
    for Vg in Vgs:
        Eres = l["c"] + l["slope"] * Vg
        if 0.15 < abs(Eres) < 0.7:
            i = np.argmin(abs(Vgs - Vg))
            row = G[i]
            pos = Vsds > 0
            ridge.append((Vg, Eres, Vsds[pos][np.argmax(row[pos])]))
ridge = np.array(ridge)
edge_ratio = np.median(ridge[:, 2] / (2 * abs(ridge[:, 1])))

zb = G[:, np.argmin(abs(Vsds))]
zb_peaks = [float(Vgs[i]) for i in range(1, len(Vgs) - 1)
            if zb[i] > zb[i - 1] and zb[i] > zb[i + 1] and zb[i] > 0.3]

stats = dict(n_levels=len(levels), level_c=cs.tolist(), spacing=np.diff(cs).tolist(),
             mean_slope=float(slopes.mean()), std_slope=float(slopes.std()),
             edge_ratio_median=float(edge_ratio), zero_bias_peaks_Vg=zb_peaks,
             G_max=float(G.max()), I_max=float(I.max()))
json.dump(stats, open(f"{out}/stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))

ext = [Vgs[0], Vgs[-1], Vsds[0], Vsds[-1]]
# Fig 1: T(E, Vg)
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(np.log10(np.clip(T.T, 1e-6, 1)), origin="lower", aspect="auto",
               extent=[Vgs[0], Vgs[-1], Es[0], Es[-1]], cmap="viridis", vmin=-6, vmax=0)
ax.axhline(0, color="w", lw=0.6, ls=":")
ax.set(xlabel="gate voltage $V_g$ [t]", ylabel="energy $E$ [t]",
       title="Transmission $\\log_{10}T(E, V_g)$")
fig.colorbar(im, label="$\\log_{10} T$")
fig.tight_layout()
fig.savefig(f"{out}/fig1_T_map.png", dpi=150)

# Fig 2: dI/dV map with analytic edges
fig, ax = plt.subplots(figsize=(6, 4.2))
im = ax.imshow(G.T, origin="lower", aspect="auto", extent=ext, cmap="magma",
               vmin=0, vmax=np.percentile(G, 99.5))
for l in levels:
    e = l["c"] + l["slope"] * Vgs
    ax.plot(Vgs, 2 * e, "c--", lw=0.7)
    ax.plot(Vgs, -2 * e, "c--", lw=0.7)
ax.set(xlim=ext[:2], ylim=ext[2:], xlabel="gate voltage $V_g$ [t]",
       ylabel="bias $V_{sd}$ [t/e]", title="Differential conductance $dI/dV_{sd}$ [e$^2$/h]")
fig.colorbar(im, label="$G$ [e$^2$/h]")
fig.tight_layout()
fig.savefig(f"{out}/fig2_diamond_map.png", dpi=150)

# Fig 3: cuts
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
axs[0].plot(Vgs, zb)
axs[0].set(xlabel="$V_g$ [t]", ylabel="$G(V_{sd}\\!=\\!0)$ [e$^2$/h]",
           title="Zero-bias conductance peaks")
for Vg0 in (Vgs[len(Vgs) // 4], Vgs[len(Vgs) // 2], Vgs[3 * len(Vgs) // 4]):
    i = np.argmin(abs(Vgs - Vg0))
    axs[1].plot(Vsds, I[i], label=f"$V_g$={Vgs[i]:.2f}")
axs[1].set(xlabel="$V_{sd}$ [t/e]", ylabel="$I$ [e t/h]", title="I–V curves")
axs[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{out}/fig3_cuts.png", dpi=150)
