"""2D TCAD (DEVSIM) of a charge-trap NAND flash cell: Id-Vg for erased /
neutral / programmed states.

    y=0   gate (n+ poly, contact)
          blocking oxide 8 nm | SiN trap 6 nm | tunnel oxide 4 nm
    y=18  p-Si body (Na=1e17), n+ source/drain boxes (Lg = 120 nm stack,
          ~80 nm metallurgical channel), body contact at the bottom.

Stored charge in the SiN layer is a fixed volume charge (+ holes = erased,
- electrons = programmed).  Drift-diffusion with constant mobility and SRH.

Run: python3 nand_cell_2d.py [Lch_nm]   (needs devsim, numpy, matplotlib)
"""
import csv
import os
import sys

import numpy as np
import devsim as ds
from devsim.python_packages.model_create import (
    CreateNodeModel, CreateSolution, CreateContactNodeModel)
from devsim.python_packages.simple_physics import (
    SetOxideParameters, SetSiliconParameters, CreateSiliconPotentialOnly,
    CreateSiliconPotentialOnlyContact, CreateOxidePotentialOnly, CreateOxideContact,
    CreateSiliconOxideInterface, CreateSiliconDriftDiffusion,
    CreateSiliconDriftDiffusionAtContact, GetContactBiasName)

HERE = os.path.dirname(os.path.abspath(__file__))
nm = 1e-7
T, NI, Q, EPS0 = 300.0, 1.0e10, 1.602e-19, 8.854e-14
NA, ND = 1e17, 5e19
PHI_GATE = 0.56
EPS_SIN = 7.5
VD = 0.05                                  # linear-region drain bias [V]

# geometry [nm]: metallurgical channel length LCH (argv[1], default 80); the gate
# stack overlaps each n+ S/D junction by 20 nm; n+ S/D for x<XJ0 or x>XJ1
LCH = float(sys.argv[1]) if len(sys.argv) > 1 else 80.0
XG0 = 40.0
XJ0 = XG0 + 20.0
XJ1 = XJ0 + LCH
XG1 = XJ1 + 20.0
W = XG1 + 40.0
TAG = "" if LCH == 80.0 else f"_L{int(LCH)}"
Y_BLK, Y_TRAP, Y_TUN, Y_SI0, Y_SIB, Y_JD = 8, 14, 18, 18, 118, 58
T_TRAP = (Y_TRAP - Y_BLK) * nm

dev, mesh = "nand2d", "nand2d_mesh"


def axis(keys):
    """Non-uniform 1D grid [nm] from (position, spacing) keys."""
    pts = [keys[0][0]]
    for (p0, s0), (p1, s1) in zip(keys[:-1], keys[1:]):
        n = max(1, int(round((p1 - p0) / (0.5 * (s0 + s1)))))
        pts += [p0 + (p1 - p0) * k / n for k in range(1, n + 1)]
    return pts


XS = axis([(0, 10), (XG0, 5), (XJ0 - 15, 3), (XJ0 + 15, 3), (0.5 * (XJ0 + XJ1), 8), (XJ1 - 15, 3),
           (XJ1 + 15, 3), (XG1, 5), (W, 10)])
YS = axis([(0, 3), (Y_BLK, 2), (Y_TRAP, 2), (Y_TUN, 1), (28, 2), (Y_JD, 4), (Y_SIB, 20)])


def region_of(xc, yc):
    if Y_SI0 <= yc <= Y_SIB:
        return "si"
    if XG0 <= xc <= XG1:
        for name, y0, y1 in (("blk", 0, Y_BLK), ("trap", Y_BLK, Y_TRAP), ("tun", Y_TRAP, Y_TUN)):
            if y0 <= yc <= y1:
                return name
    return None


def write_msh(path):
    """Right-triangle mesh of the tensor grid in gmsh 2 format (cm)."""
    nid, nodes, tris, lines = {}, [], {}, {}

    def node(i, j):
        if (i, j) not in nid:
            nid[(i, j)] = len(nodes) + 1
            nodes.append((XS[i] * nm, YS[j] * nm))
        return nid[(i, j)]

    for i in range(len(XS) - 1):
        for j in range(len(YS) - 1):
            r = region_of(0.5 * (XS[i] + XS[i + 1]), 0.5 * (YS[j] + YS[j + 1]))
            if r is None:
                continue
            a, b, c, d = node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)
            tris.setdefault(r, []).extend([(a, b, c), (a, c, d)])

    def hline(name, y, x0, x1):
        for i in range(len(XS) - 1):
            if x0 <= XS[i] and XS[i + 1] <= x1:
                lines.setdefault(name, []).append((nid[(i, YS.index(y))], nid[(i + 1, YS.index(y))]))

    def vline(name, x, y0, y1):
        for j in range(len(YS) - 1):
            if y0 <= YS[j] and YS[j + 1] <= y1:
                lines.setdefault(name, []).append((nid[(XS.index(x), j)], nid[(XS.index(x), j + 1)]))

    hline("gate", 0, XG0, XG1)
    vline("source", 0, Y_SI0, Y_JD)
    vline("drain", W, Y_SI0, Y_JD)
    hline("body", Y_SIB, 0, W)
    hline("if1", Y_BLK, XG0, XG1)
    hline("if2", Y_TRAP, XG0, XG1)
    hline("if3", Y_TUN, XG0, XG1)

    groups = [(n, 2) for n in tris] + [(n, 1) for n in lines]
    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n%d\n" % len(groups))
        for k, (n, dim) in enumerate(groups, 1):
            f.write('%d %d "%s"\n' % (dim, k, n))
        f.write("$EndPhysicalNames\n$Nodes\n%d\n" % len(nodes))
        for k, (x, y) in enumerate(nodes, 1):
            f.write("%d %.10e %.10e 0\n" % (k, x, y))
        f.write("$EndNodes\n")
        els = []
        for k, (n, dim) in enumerate(groups, 1):
            for e in (tris[n] if dim == 2 else lines[n]):
                els.append((2 if dim == 2 else 1, k, e))
        f.write("$Elements\n%d\n" % len(els))
        for k, (t, g, e) in enumerate(els, 1):
            f.write("%d %d 2 %d %d %s\n" % (k, t, g, g, " ".join(map(str, e))))
        f.write("$EndElements\n")


MSH = os.path.join(os.environ.get("TMPDIR", "/tmp"), "nand2d.msh")
write_msh(MSH)
ds.create_gmsh_mesh(mesh=mesh, file=MSH)
for r, mat in (("blk", "Oxide"), ("trap", "Nitride"), ("tun", "Oxide"), ("si", "Silicon")):
    ds.add_gmsh_region(mesh=mesh, gmsh_name=r, region=r, material=mat)
for c, r in (("gate", "blk"), ("source", "si"), ("drain", "si"), ("body", "si")):
    ds.add_gmsh_contact(mesh=mesh, gmsh_name=c, region=r, name=c, material="metal")
for name, r0, r1 in (("if1", "blk", "trap"), ("if2", "trap", "tun"), ("if3", "tun", "si")):
    ds.add_gmsh_interface(mesh=mesh, gmsh_name=name, region0=r0, region1=r1, name=name)
ds.finalize_mesh(mesh=mesh)
ds.create_device(mesh=mesh, device=dev)

# ---- physics ---------------------------------------------------------------
for r in ("blk", "trap", "tun"):
    SetOxideParameters(dev, r, T)
ds.set_parameter(device=dev, region="trap", name="Permittivity", value=EPS_SIN * EPS0)
SetSiliconParameters(dev, "si", T)
ds.set_parameter(device=dev, region="si", name="n_i", value=NI)
ds.set_parameter(device=dev, region="si", name="mu_n", value=300.0)
ds.set_parameter(device=dev, region="si", name="mu_p", value=150.0)

lg = lambda a, w=3 * nm: f"(1/(1+exp(({a})/{w})))"
sd = f"({lg(f'x-{XJ0*nm}')}+{lg(f'{XJ1*nm}-x')})*{lg(f'y-{Y_JD*nm}')}"
CreateNodeModel(dev, "si", "NetDoping", f"{ND}*{sd} - {NA}*(1-{sd}*0)")
# (n+ boxes are ND - NA where sd~1; sd*ND already dominates, keep -NA everywhere)

CreateSiliconPotentialOnly(dev, "si")
for r in ("blk", "trap", "tun"):
    CreateOxidePotentialOnly(dev, r, "log_damp")

ds.set_parameter(device=dev, name="Nfix", value=0.0)
CreateNodeModel(dev, "trap", "TrapNodeCharge", "-ElectronCharge*Nfix")
ds.equation(device=dev, region="trap", name="PotentialEquation",
            variable_name="Potential", node_model="TrapNodeCharge",
            edge_model="PotentialEdgeFlux", variable_update="log_damp")

for i in (1, 2, 3):
    CreateSiliconOxideInterface(dev, f"if{i}")
CreateOxideContact(dev, "blk", "gate")
for c in ("source", "drain", "body"):
    CreateSiliconPotentialOnlyContact(dev, "si", c)
    ds.set_parameter(device=dev, name=GetContactBiasName(c), value=0.0)


def vg_set(vg):
    ds.set_parameter(device=dev, name=GetContactBiasName("gate"), value=vg + PHI_GATE)


def newton(maxit=60):
    ds.solve(type="dc", absolute_error=1e10, relative_error=1e-8, maximum_iterations=maxit)


def try_step(setter, target, cur, tol=1e-4):
    """Move a bias/charge from cur to target, halving the step on failure."""
    step = target - cur
    while abs(target - cur) > tol:
        nxt = cur + np.clip(target - cur, -abs(step), abs(step))
        setter(nxt)
        try:
            newton()
            cur = nxt
            step = min(abs(step) * 1.5, abs(target - cur) if target != cur else 1) * \
                np.sign(target - cur or 1)
        except ds.error:
            setter(cur)
            step /= 2
            if abs(step) < 1e-4:
                raise
            newton()
    return cur


# 1) equilibrium (potential only)
vg_set(0.0)
newton()
# 2) switch on drift-diffusion
for n in ("Electrons", "Holes"):
    CreateSolution(dev, "si", n)
ds.set_node_values(device=dev, region="si", name="Electrons", init_from="IntrinsicElectrons")
ds.set_node_values(device=dev, region="si", name="Holes", init_from="IntrinsicHoles")
CreateSiliconDriftDiffusion(dev, "si")
for c in ("source", "drain", "body"):
    CreateSiliconDriftDiffusionAtContact(dev, "si", c)
newton()


def drain_current():
    """|Id| in A/um (DEVSIM 2D current is per cm of depth)."""
    ie = ds.get_contact_current(device=dev, contact="drain", equation="ElectronContinuityEquation")
    ih = ds.get_contact_current(device=dev, contact="drain", equation="HoleContinuityEquation")
    return abs(ie + ih) * 1e-4


def set_vd(v):
    ds.set_parameter(device=dev, name=GetContactBiasName("drain"), value=v)


def set_nfix(sheet):
    ds.set_parameter(device=dev, name="Nfix", value=sheet / T_TRAP)


def id_vg(sheet, vgs):
    """Set stored charge at Vg=0, then sweep Vg upward at Vd=VD."""
    global cur_vg, cur_q
    cur_vg = try_step(vg_set, 0.0, cur_vg)
    cur_q = try_step(set_nfix, sheet, cur_q, tol=1e6) if abs(sheet - cur_q) > 1e6 else cur_q
    rows = []
    for vg in vgs:
        cur_vg = try_step(vg_set, vg, cur_vg)
        rows.append((vg, drain_current()))
        sys.stdout.write("."); sys.stdout.flush()
    print()
    return np.array(rows)


def vth_cc(rows, icc=1e-7):
    """Constant-current Vth: log-linear interpolation at Id = icc [A/um]."""
    return float(np.interp(np.log(icc), np.log(rows[:, 1]), rows[:, 0]))


def vth_gm(rows):
    """Linear extrapolation at max gm, corrected by VD/2."""
    vg, i = rows[:, 0], rows[:, 1]
    gm = np.gradient(i, vg)
    k = int(np.argmax(gm))
    return float(vg[k] - i[k] / gm[k] - VD / 2)


if __name__ == "__main__":
    cur_vg, cur_q = 0.0, 0.0
    cur_vd = try_step(set_vd, VD, 0.0)
    states = {"erased": +3e12, "neutral": 0.0, "programmed": -3e12}
    vgs = np.arange(-3.0, 6.001, 0.25)
    res, vth = {}, {}
    for name, ns in states.items():
        print(f"{name}: stored {ns:+.1e} q/cm2", end=" ")
        res[name] = id_vg(ns, vgs)
        vth[name] = (vth_cc(res[name]), vth_gm(res[name]))
        print(f"  Vth(cc 1e-7 A/um) = {vth[name][0]:.3f} V,  Vth(gm-max) = {vth[name][1]:.3f} V")
    print("memory window (cc) = %.3f V" % (vth["programmed"][0] - vth["erased"][0]))

    with open(os.path.join(HERE, f"idvg_2d{TAG}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state", "Vg[V]", f"Id[A/um] at Vd={VD}V"])
        for n, r in res.items():
            for row in r:
                w.writerow([n, *row])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"erased": "#1f77b4", "neutral": "#7f7f7f", "programmed": "#d62728"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for n, r in res.items():
        ax[0].semilogy(r[:, 0], r[:, 1], "-o", ms=3, color=col[n],
                       label=f"{n}  (Vth(gm)={vth[n][1]:.2f} V)")
        ax[1].plot(r[:, 0], r[:, 1] * 1e6, "-", color=col[n], label=n)
    ax[0].axhline(1e-7, color="0.6", ls=":", lw=1)
    ax[0].set(xlabel="Vg [V]", ylabel="Id [A/µm]", ylim=(1e-15, 1e-3),
              title=f"Id–Vg (Vd = {VD} V), log, Lch = {LCH:.0f} nm")
    ax[1].set(xlabel="Vg [V]", ylabel="Id [µA/µm]", title="Id–Vg, linear")
    for a in ax:
        a.grid(alpha=.3); a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, f"nand_cell_2d_idvg{TAG}.png"), dpi=130)
    print(f"wrote nand_cell_2d_idvg{TAG}.png, idvg_2d{TAG}.csv")
