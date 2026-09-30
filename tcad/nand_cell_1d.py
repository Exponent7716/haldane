"""1D TCAD of a charge-trap NAND flash cell gate stack (DEVSIM).

    gate(n+ poly) | blocking oxide | SiN trap layer | tunnel oxide | p-Si

Stored charge (electrons = programmed, holes = erased) is a fixed volume charge
in the SiN layer.  For each state the gate voltage is swept, the Poisson-
Boltzmann problem is solved in quasi-equilibrium, and the surface potential,
inversion charge and threshold voltage (psi_s = 2*phi_F) are extracted.

Run:  python3 nand_cell_1d.py      (needs devsim, numpy, matplotlib)
"""
import csv
import os

import numpy as np
import devsim as ds
from devsim.python_packages.model_create import CreateNodeModel
from devsim.python_packages.simple_physics import (
    SetOxideParameters, SetSiliconParameters, CreateSiliconPotentialOnly,
    CreateSiliconPotentialOnlyContact, CreateOxidePotentialOnly,
    CreateOxideContact, CreateSiliconOxideInterface, GetContactBiasName)

HERE = os.path.dirname(os.path.abspath(__file__))
ds.set_parameter(name="debug_level", value="warning")

# ---- geometry [cm], doping, constants -------------------------------------
T_BLK, T_TRAP, T_TUN, T_SI = 8e-7, 6e-7, 4e-7, 300e-7   # 8/6/4 nm oxide/SiN/oxide
NA = 1e17                        # p-body doping [cm^-3]
EPS_SIN = 7.5
PHI_GATE = 0.56                  # n+ poly Fermi level above Ei [V]
T, NI, Q, EPS0 = 300.0, 1.0e10, 1.602e-19, 8.854e-14
VT = 8.617e-5 * T
PHI_F = VT * np.log(NA / NI)

X1 = T_BLK
X2 = X1 + T_TRAP
X3 = X2 + T_TUN
X4 = X3 + T_SI

dev, mesh = "nand", "nand_mesh"
ds.create_1d_mesh(mesh=mesh)
for pos, ps, tag in [(0, 5e-8, "gate"), (X1, 2e-8, "i1"), (X2, 2e-8, "i2"),
                     (X3, 1e-8, "i3"), (X3 + 20e-7, 5e-8, "m"), (X4, 2e-6, "body")]:
    ds.add_1d_mesh_line(mesh=mesh, pos=pos, ps=ps, tag=tag)
ds.add_1d_region(mesh=mesh, material="Oxide", region="blk", tag1="gate", tag2="i1")
ds.add_1d_region(mesh=mesh, material="Nitride", region="trap", tag1="i1", tag2="i2")
ds.add_1d_region(mesh=mesh, material="Oxide", region="tun", tag1="i2", tag2="i3")
ds.add_1d_region(mesh=mesh, material="Silicon", region="si", tag1="i3", tag2="body")
ds.add_1d_contact(mesh=mesh, name="gate", tag="gate", material="metal")
ds.add_1d_contact(mesh=mesh, name="body", tag="body", material="metal")
for i in (1, 2, 3):
    ds.add_1d_interface(mesh=mesh, name=f"if{i}", tag=f"i{i}")
ds.finalize_mesh(mesh=mesh)
ds.create_device(mesh=mesh, device=dev)

# ---- physics --------------------------------------------------------------
for r in ("blk", "trap", "tun"):
    SetOxideParameters(dev, r, T)
ds.set_parameter(device=dev, region="trap", name="Permittivity", value=EPS_SIN * EPS0)
SetSiliconParameters(dev, "si", T)
ds.set_parameter(device=dev, region="si", name="n_i", value=NI)
CreateNodeModel(dev, "si", "NetDoping", f"-{NA}")

CreateSiliconPotentialOnly(dev, "si")
CreateOxidePotentialOnly(dev, "blk", "log_damp")
CreateOxidePotentialOnly(dev, "tun", "log_damp")
CreateOxidePotentialOnly(dev, "trap", "log_damp")

# fixed charge in SiN: rho = q*Nfix [Nfix>0: holes/erased, <0: electrons/programmed]
ds.set_parameter(device=dev, name="Nfix", value=0.0)
CreateNodeModel(dev, "trap", "TrapNodeCharge", "-ElectronCharge*Nfix")
ds.equation(device=dev, region="trap", name="PotentialEquation",
            variable_name="Potential", node_model="TrapNodeCharge",
            edge_model="PotentialEdgeFlux", variable_update="log_damp")

for i in (1, 2, 3):
    CreateSiliconOxideInterface(dev, f"if{i}")
CreateOxideContact(dev, "blk", "gate")          # psi_gate = Vg + PHI_GATE
CreateSiliconPotentialOnlyContact(dev, "si", "body")
ds.set_parameter(device=dev, name=GetContactBiasName("body"), value=0.0)


def solve(vg):
    ds.set_parameter(device=dev, name=GetContactBiasName("gate"), value=vg + PHI_GATE)
    ds.solve(type="dc", absolute_error=1e-2, relative_error=1e-10, maximum_iterations=80)


def nodes(region, name):
    x = np.array(ds.get_node_model_values(device=dev, region=region, name="x"))
    v = np.array(ds.get_node_model_values(device=dev, region=region, name=name))
    o = np.argsort(x)
    return x[o], v[o]


def observe():
    x, psi = nodes("si", "Potential")
    n = NI * np.exp(psi / VT)
    qinv = Q * np.trapezoid(n - NI ** 2 / NA, x)      # C/cm^2
    return psi[0], qinv


def sweep(nfix_sheet, vgs):
    """nfix_sheet: stored sheet charge density [q/cm^2], + = holes, - = electrons."""
    ds.set_parameter(device=dev, name="Nfix", value=nfix_sheet / T_TRAP)
    rows = []
    for vg in vgs:
        solve(vg)
        psi_s, qinv = observe()
        rows.append((vg, psi_s, qinv))
    return np.array(rows)


def vth_of(rows):
    """Vg at which band bending psi_s - psi_bulk = 2*phi_F (linear interpolation).

    psi is referenced to Ei with the Fermi level at 0, so psi_bulk = -phi_F and
    the criterion in absolute potential is psi_s = +phi_F.
    """
    o = np.argsort(rows[:, 1])
    return float(np.interp(PHI_F, rows[o, 1], rows[o, 0]))


def band_diagram(vg):
    """Return x[nm] and psi[V] across the whole stack at gate bias vg."""
    solve(vg)
    xs, ps = [], []
    for r in ("blk", "trap", "tun", "si"):
        x, p = nodes(r, "Potential")
        xs.append(x); ps.append(p)
    return np.concatenate(xs) * 1e7, np.concatenate(ps)


if __name__ == "__main__":
    print(f"phi_F = {PHI_F:.3f} V  (2phi_F = {2*PHI_F:.3f} V)")
    solve(0.0)
    states = {"erased (+3e12 /cm2)": +3e12, "neutral": 0.0, "programmed (-3e12 /cm2)": -3e12}
    vgs = np.arange(-2.0, 9.01, 0.25)
    results, vth = {}, {}
    for name, ns in states.items():
        ds.set_parameter(device=dev, name="Nfix", value=0.0); solve(0.0)  # restart
        # ramp the trap charge in a few steps for robust convergence
        for f in np.linspace(0, 1, 6)[1:]:
            ds.set_parameter(device=dev, name="Nfix", value=f * ns / T_TRAP); solve(0.0)
        results[name] = sweep(ns, vgs)
        vth[name] = vth_of(results[name])
        print(f"{name:28s} Vth = {vth[name]:6.3f} V")
    v0 = vth["neutral"]
    print("memory window (erased -> programmed) = "
          f"{vth['programmed (-3e12 /cm2)'] - vth['erased (+3e12 /cm2)']:.3f} V")

    with open(os.path.join(HERE, "vth_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state", "Vg[V]", "psi_s[V]", "Qinv[C/cm2]"])
        for name, r in results.items():
            for row in r:
                w.writerow([name, *row])

    # Vth vs stored charge
    qs = np.linspace(-4e12, 4e12, 9)
    vq = []
    for ns in qs:
        ds.set_parameter(device=dev, name="Nfix", value=0.0); solve(0.0)
        for fr in np.linspace(0, 1, 6)[1:]:
            ds.set_parameter(device=dev, name="Nfix", value=fr * ns / T_TRAP); solve(0.0)
        vq.append(vth_of(sweep(ns, vgs)))

    # band-diagram-like potential profiles at read bias
    prof = {}
    for name, ns in states.items():
        ds.set_parameter(device=dev, name="Nfix", value=0.0); solve(0.0)
        for fr in np.linspace(0, 1, 6)[1:]:
            ds.set_parameter(device=dev, name="Nfix", value=fr * ns / T_TRAP); solve(0.0)
        prof[name] = band_diagram(3.0)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for name, r in results.items():
        ax[0].semilogy(r[:, 0], np.maximum(np.abs(r[:, 2]), 1e-20), label=name)
    ax[0].set(xlabel="Vg [V]", ylabel="|Q_inv| [C/cm$^2$] (bulk-subtracted)",
              title="Inversion charge vs gate voltage", ylim=(1e-12, 1e-5))
    ax[0].legend(fontsize=8)
    ax[1].plot(qs / 1e12, vq, "o-")
    ax[1].set(xlabel="stored sheet charge [10$^{12}$ q/cm$^2$]  (+ holes, - electrons)",
              ylabel="Vth [V]", title="Threshold voltage vs stored charge")
    ax[1].grid(alpha=.3)
    for name, (x, p) in prof.items():
        ax[2].plot(x, p, label=name)
    for xb in np.array([X1, X2, X3]) * 1e7:
        ax[2].axvline(xb, color="0.7", ls=":")
    ax[2].set(xlim=(0, 60), xlabel="x [nm]", ylabel="electrostatic potential [V]",
              title="Potential profile at Vg = 3 V")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "nand_cell_1d.png"), dpi=130)
    print("wrote nand_cell_1d.png, vth_results.csv")
