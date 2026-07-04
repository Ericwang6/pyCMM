import pytest
import torch
import sys
import openmm.app as app

sys.path.insert(0, "/pscratch/sd/a/asa/software/pyCMM")
from cmm.bonded import (
    computeTorsionFromVecs, computeTorsion,
    computePeriodicTorsionEnergy,
    computeTorsionBondCoupling,
    computeTorsionAngleAngleCoupling,
    computeBond, computeAngle, computeAngleFromVecs,
)
from cmm.ffxml import ForceFieldXML
from cmm.topology import Topology
from cmm.units import BOHR2NM

PDB_PATH = "/pscratch/sd/a/asa/software/pyCMM/tests/data/reference_for_mchem/ethane_196_gaff_opt.pdb"
FF_PATH  = "/pscratch/sd/a/asa/software/pyCMM/tests/data/reference_for_mchem/cmm.xml"

# Atom layout per ethane: 0=C1, 1=C2, 2=H11, 3=H12, 4=H13, 5=H21, 6=H22, 7=H23
C1_OFF  = 0
C2_OFF  = 1
H1_OFFS = [2, 3, 4]
H2_OFFS = [5, 6, 7]

# Periodic torsion parameters (h.eth-c.eth-c.eth-h.eth)
PER   = torch.tensor([1.0, 2.0, 3.0, 4.0])
PHASE = torch.tensor([0.0, 3.14159265, 0.0, 3.14159265])
K_TOR = torch.tensor([0.00193656625, 0.00081527625, 0.000248115974, -9.84806839e-05])

# TorsionBond coupling
K_TB_1 = torch.tensor([-0.000431551244,  0.000510807173, -0.000481548426, -0.000145481971])  # ctype=1: C-C
K_TB_2 = torch.tensor([ 0.000836148074, -0.000940610181,  4.05631832e-05,  0.000158809328])  # ctype=2: C-H l-side
K_TB_3 = torch.tensor([ 0.000169256436, -0.000362116342,  6.28576852e-05,  0.000131336638])  # ctype=3: C-H i-side
R_EQ_CC = 2.8846949
R_EQ_CH = 2.06716744

# TorsionAngle coupling
# ctype=1: j-k-l angle (C1-C2-H2), theta_eq = THETA_EQ_CCH
# ctype=2: i-j-k angle (H1-C1-C2),  theta_eq = THETA_EQ_CCH
# ctype=3: h-c-h angle on i-side carbon, theta_eq = THETA_EQ_HCH
K_TA_1 = torch.tensor([-7.11543042e-05, -0.00139752683, -0.000204159274, -0.000858753727])
K_TA_2 = torch.tensor([ 0.00119055861, -0.000626278509, -0.000117989999, -0.000377726923])
K_TA_3 = torch.tensor([ 8.66193987e-05, -0.000905123301,  7.68286527e-05, -9.87639062e-05])
THETA_EQ_CCH = 1.94202996
THETA_EQ_HCH = 1.87821616


def build_torsion_indices(n_molecules: int, device):
    """9 H-C-C-H torsion indices per ethane: [H1, C1, C2, H2]"""
    rows = []
    for m in range(n_molecules):
        base = m * 8
        for h1 in H1_OFFS:
            for h2 in H2_OFFS:
                rows.append([base+h1, base+C1_OFF, base+C2_OFF, base+h2])
    return torch.tensor(rows, device=device, dtype=torch.int32)


def build_torsion_angle_indices(n_molecules: int, device):
    """
    Build 7-column torsion-angle indices: [i, j, k, l, a1, vertex, a2]
    The torsion is cols 0:4, the angle is cols 4:7 (used with computeAngle).

    ctype=1: angle = j-k-l  -> [H1, C1, C2, H2,  C1, C2, H2]  (C1-C2-H2 angle, vertex=C2)
    ctype=2: angle = i-j-k  -> [H1, C1, C2, H2,  H1, C1, C2]  (H1-C1-C2 angle, vertex=C1)
    ctype=3: angle = h-c-h  -> [H1, C1, C2, H2,  Ha, C1, Hb]  (H-C1-H angle,   vertex=C1)
    """
    rows_ctype1, rows_ctype2, rows_ctype3 = [], [], []

    for m in range(n_molecules):
        base = m * 8
        for h1 in H1_OFFS:
            for h2 in H2_OFFS:
                i = base + h1
                j = base + C1_OFF
                k = base + C2_OFF
                l = base + h2  # noqa: E741

                # ctype=1: C1-C2-H2 angle (vertex = C2 = k)
                rows_ctype1.append([i, j, k, l,  j, k, l])

                # ctype=2: H1-C1-C2 angle (vertex = C1 = j)
                rows_ctype2.append([i, j, k, l,  i, j, k])

                # ctype=3: H-C1-H angle using the OTHER two H atoms on C1 (vertex = C1 = j)
                other_hs = [base + h for h in H1_OFFS if h != h1]
                rows_ctype3.append([i, j, k, l,  other_hs[0], j, other_hs[1]])

    return (
        torch.tensor(rows_ctype1, device=device, dtype=torch.int32),
        torch.tensor(rows_ctype2, device=device, dtype=torch.int32),
        torch.tensor(rows_ctype3, device=device, dtype=torch.int32),
    )


def compute_all_torsion_energies(coords, n_molecules, device, dtype):
    """Compute all torsion + torsionbond + torsionangle energies."""

    tor_idx = build_torsion_indices(n_molecules, device)
    n_tors  = tor_idx.shape[0]

    per   = PER.to(device=device, dtype=dtype)
    phase = PHASE.to(device=device, dtype=dtype)

    # Expand to (n_tors, 4) so computePeriodicTorsionEnergy takes the 2D branch
    per_2d   = per.unsqueeze(0).expand(n_tors, -1)    # (n_tors, 4)
    phase_2d = phase.unsqueeze(0).expand(n_tors, -1)  # (n_tors, 4)

    # ---- Periodic torsion ----
    phi      = computeTorsion(coords, tor_idx, box=None, boxInv=None)
    k_tor_2d = K_TOR.to(device=device, dtype=dtype).unsqueeze(0).expand(n_tors, -1)
    e_tor    = computePeriodicTorsionEnergy(phi, per_2d, phase_2d, k_tor_2d).sum()

    # ---- TorsionBond ctype=1: central C-C bond (j-k) ----
    cc_idx   = torch.stack([tor_idx[:, 1], tor_idx[:, 2]], dim=1)
    r_cc     = computeBond(coords, cc_idx, box=None, boxInv=None)
    req_cc   = torch.full((n_tors,), R_EQ_CC, device=device, dtype=dtype)
    k_tb1_2d = K_TB_1.to(device=device, dtype=dtype).unsqueeze(0).expand(n_tors, -1)
    e_tb1    = computeTorsionBondCoupling(phi, r_cc, per_2d, phase_2d, k_tb1_2d, req_cc).sum()

    # ---- TorsionBond ctype=2: l-side C-H bond (k-l) ----
    chl_idx  = torch.stack([tor_idx[:, 2], tor_idx[:, 3]], dim=1)
    r_chl    = computeBond(coords, chl_idx, box=None, boxInv=None)
    req_ch   = torch.full((n_tors,), R_EQ_CH, device=device, dtype=dtype)
    k_tb2_2d = K_TB_2.to(device=device, dtype=dtype).unsqueeze(0).expand(n_tors, -1)
    e_tb2    = computeTorsionBondCoupling(phi, r_chl, per_2d, phase_2d, k_tb2_2d, req_ch).sum()

    # ---- TorsionBond ctype=3: i-side C-H bond (j-i) ----
    chi_idx  = torch.stack([tor_idx[:, 1], tor_idx[:, 0]], dim=1)
    r_chi    = computeBond(coords, chi_idx, box=None, boxInv=None)
    k_tb3_2d = K_TB_3.to(device=device, dtype=dtype).unsqueeze(0).expand(n_tors, -1)
    e_tb3    = computeTorsionBondCoupling(phi, r_chi, per_2d, phase_2d, k_tb3_2d, req_ch).sum()

    # ---- TorsionAngle: computeTorsionBondCoupling(torsion, angle, per, phase, k, theta_eq) ----
    ta_idx1, ta_idx2, ta_idx3 = build_torsion_angle_indices(n_molecules, device)

    phi_ta1 = computeTorsion(coords, ta_idx1[:, :4], box=None, boxInv=None)
    phi_ta2 = computeTorsion(coords, ta_idx2[:, :4], box=None, boxInv=None)
    phi_ta3 = computeTorsion(coords, ta_idx3[:, :4], box=None, boxInv=None)

    theta1 = computeAngle(coords, ta_idx1[:, 4:], box=None, boxInv=None)  # C1-C2-H2
    theta2 = computeAngle(coords, ta_idx2[:, 4:], box=None, boxInv=None)  # H1-C1-C2
    theta3 = computeAngle(coords, ta_idx3[:, 4:], box=None, boxInv=None)  # H-C1-H

    teq_cch  = torch.full((n_tors,), THETA_EQ_CCH, device=device, dtype=dtype)
    teq_hch  = torch.full((n_tors,), THETA_EQ_HCH, device=device, dtype=dtype)

    k_ta1_2d = K_TA_1.to(device=device, dtype=dtype).unsqueeze(0).expand(n_tors, -1)
    k_ta2_2d = K_TA_2.to(device=device, dtype=dtype).unsqueeze(0).expand(n_tors, -1)
    k_ta3_2d = K_TA_3.to(device=device, dtype=dtype).unsqueeze(0).expand(n_tors, -1)

    e_ta1 = computeTorsionBondCoupling(phi_ta1, theta1, per_2d, phase_2d, k_ta1_2d, teq_cch).sum()
    e_ta2 = computeTorsionBondCoupling(phi_ta2, theta2, per_2d, phase_2d, k_ta2_2d, teq_cch).sum()
    e_ta3 = computeTorsionBondCoupling(phi_ta3, theta3, per_2d, phase_2d, k_ta3_2d, teq_hch).sum()

    return {
        'phi':   phi,
        'e_tor': e_tor,
        'e_tb1': e_tb1, 'e_tb2': e_tb2, 'e_tb3': e_tb3,
        'e_ta1': e_ta1, 'e_ta2': e_ta2, 'e_ta3': e_ta3,
        'total': e_tor + e_tb1 + e_tb2 + e_tb3 + e_ta1 + e_ta2 + e_ta3,
    }


# ---------------------------------------------------------------------------
# Single molecule reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float32),
    ('cuda', torch.float64),
])
def test_torsion_reference_single(device, dtype):
    """Reference energy and forces for all torsion terms - single ethane."""

    pdb    = app.PDBFile(PDB_PATH)
    pos    = pdb.getPositions(asNumpy=True)._value / BOHR2NM
    coords = torch.tensor(pos[:8], device=device, dtype=dtype, requires_grad=True)

    results      = compute_all_torsion_energies(coords, 1, device, dtype)
    total_energy = results['total']

    print(f"\n{'='*60}")
    print(f"Torsion Reference - Single Ethane ({dtype})")
    print(f"{'='*60}")
    for i, p in enumerate(torch.rad2deg(results['phi'])):
        print(f"  phi[{i}] = {p.item():.10f} deg")
    print(f"\nEnergy breakdown:")
    print(f"  Periodic torsion:       {results['e_tor'].item():.17f}")
    print(f"  TorsionBond (C-C):      {results['e_tb1'].item():.17f}")
    print(f"  TorsionBond (C-H l):    {results['e_tb2'].item():.17f}")
    print(f"  TorsionBond (C-H i):    {results['e_tb3'].item():.17f}")
    print(f"  TorsionAngle (C1-C2-H): {results['e_ta1'].item():.17f}")
    print(f"  TorsionAngle (H-C1-C2): {results['e_ta2'].item():.17f}")
    print(f"  TorsionAngle (H-C1-H):  {results['e_ta3'].item():.17f}")
    print(f"  Total:                  {total_energy.item():.17f}")

    total_energy.backward()
    forces = -coords.grad

    labels = ['C1', 'C2', 'H11', 'H12', 'H13', 'H21', 'H22', 'H23']
    print(f"\nForces:")
    for i, lbl in enumerate(labels):
        print(f"  {lbl}: [{forces[i,0].item():20.15f}, {forces[i,1].item():20.15f}, {forces[i,2].item():20.15f}]")
    print(f"  Sum: [{forces.sum(0)[0].item():20.15f}, {forces.sum(0)[1].item():20.15f}, {forces.sum(0)[2].item():20.15f}]")
    print(f"{'='*60}\n")

    assert torch.allclose(forces.sum(0), torch.zeros(3, device=device, dtype=dtype), atol=1e-5)


# ---------------------------------------------------------------------------
# 196-molecule FF test (ground truth) — writes all dat files
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float64),
])
def test_torsion_ethane_196(device, dtype):
    """Reference energy, forces, and virial - 196 ethanes via ForceFieldXML."""

    pdb = app.PDBFile(PDB_PATH)
    top = Topology.fromOpenmm(pdb.topology, device)

    coords = torch.tensor(
        pdb.getPositions(asNumpy=True)._value / BOHR2NM,
        device=device, dtype=dtype, requires_grad=True
    )
    box_vectors = pdb.topology.getPeriodicBoxVectors()
    box = torch.tensor(
        [[vec.x / BOHR2NM, vec.y / BOHR2NM, vec.z / BOHR2NM] for vec in box_vectors],
        device=device, dtype=dtype, requires_grad=True
    )

    ff     = ForceFieldXML(FF_PATH, device=device)
    system = ff.parametrize(
        top, use_fd_morse=False, use_polarization=False,
        use_hardness_change=False, cutoff_sr=9.0,
        use_switch=True, use_customized_ops=False
    )

    energies = system.getEnergy(coords, box)
    e_tor = energies["torsion"]
    e_tb  = energies["torsion_bond"]
    e_ta  = energies["torsion_angle"]
    e_taa = energies["torsion_angle_angle"]  # adjust key if needed
    total = e_tor + e_tb + e_ta + e_taa

    print(f"\n{'='*80}")
    print(f"Torsion Reference Data - 196 Ethanes ({dtype})")
    print(f"{'='*80}")
    print(f"Atoms: {coords.shape[0]}  |  Ethanes: {coords.shape[0]//8}  |  Box: {box.diagonal()}")
    print(f"\nEnergy breakdown:")
    print(f"  Periodic torsion:  {e_tor.item():.17f}")
    print(f"  TorsionBond:       {e_tb.item():.17f}")
    print(f"  TorsionAngle:      {e_ta.item():.17f}")
    print(f"  TorsionAngleAngle: {e_taa.item():.17f}")
    print(f"  Total:             {total.item():.17f}")

    # Disable donated buffers so retain_graph=True works across multiple backward calls
    torch._functorch.config.donated_buffer = False

    def write_term_files(energy, term_name):
        """Backward through a single term and write energy, force, virial dat files."""
        if coords.grad is not None:
            coords.grad.zero_()

        energy.backward(retain_graph=True)
        forces = -coords.grad.clone()
        virial = -torch.einsum('ij,ik->jk', coords, forces)

        with open(f'ene_{term_name}.dat', 'w') as f:
            f.write(f'# reference {term_name} energy\n')
            f.write(f'{energy.item():.12f}\n')

        with open(f'ref_grad_{term_name}.dat', 'w') as f:
            f.write(f'# reference {term_name} forces\n')
            fn = forces.detach().cpu().numpy()
            for i in range(fn.shape[0]):
                f.write(f'{i} {fn[i,0]:.13f} {fn[i,1]:.13f} {fn[i,2]:.13f}\n')

        with open(f'virial_{term_name}_ref.dat', 'w') as f:
            f.write('# xx xy xz yx yy yz zx zy zz\n')
            v = virial.detach().cpu().numpy()
            f.write(f'{v[0,0]:.14f} {v[0,1]:.14f} {v[0,2]:.14f} '
                    f'{v[1,0]:.14f} {v[1,1]:.14f} {v[1,2]:.14f} '
                    f'{v[2,0]:.14f} {v[2,1]:.14f} {v[2,2]:.14f}\n')

        fs = forces.sum(0)
        print(f"\n  [{term_name}]")
        print(f"  Energy: {energy.item():.17f}")
        print(f"  Force sum: [{fs[0].item():.6e}, {fs[1].item():.6e}, {fs[2].item():.6e}]")
        print(f"  Virial trace: {virial.trace().item():.17f}")
        print(f"  Written: ene_{term_name}.dat  ref_grad_{term_name}.dat  virial_{term_name}_ref.dat")

        assert torch.allclose(forces.sum(0), torch.zeros(3, device=device, dtype=dtype), atol=1e-6), \
            f"Force conservation failed for {term_name}"

    print(f"\n--- Writing per-term dat files ---")
    write_term_files(e_tor,  'torsion')
    write_term_files(e_tb,   'torsion_bond')
    write_term_files(e_ta,   'torsion_angle')
    write_term_files(e_taa,  'torsion_angle_angle')
    write_term_files(total,  'torsion_all')

    print(f"\n{'='*80}\n")


# ---------------------------------------------------------------------------
# 196-molecule manual test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float64),
])
def test_torsion_ethane_196_manual(device, dtype):
    """Manual torsion calculation for 196 ethanes without ForceFieldXML."""

    pdb    = app.PDBFile(PDB_PATH)
    coords = torch.tensor(
        pdb.getPositions(asNumpy=True)._value / BOHR2NM,
        device=device, dtype=dtype, requires_grad=True
    )

    n_ethanes    = coords.shape[0] // 8
    results      = compute_all_torsion_energies(coords, n_ethanes, device, dtype)
    total_energy = results['total']

    print(f"\n{'='*80}")
    print(f"Manual Torsion - 196 Ethanes ({dtype})")
    print(f"{'='*80}")
    print(f"Torsions: {results['phi'].shape[0]}  ({n_ethanes} mol x 9)")
    print(f"Mean phi: {torch.rad2deg(results['phi']).mean().item():.6f} deg")
    print(f"\nEnergy breakdown:")
    print(f"  Periodic torsion:       {results['e_tor'].item():.17f}")
    print(f"  TorsionBond (C-C):      {results['e_tb1'].item():.17f}")
    print(f"  TorsionBond (C-H l):    {results['e_tb2'].item():.17f}")
    print(f"  TorsionBond (C-H i):    {results['e_tb3'].item():.17f}")
    print(f"  TorsionAngle (C1-C2-H): {results['e_ta1'].item():.17f}")
    print(f"  TorsionAngle (H-C1-C2): {results['e_ta2'].item():.17f}")
    print(f"  TorsionAngle (H-C1-H):  {results['e_ta3'].item():.17f}")
    print(f"  Total:                  {total_energy.item():.17f}")

    total_energy.backward()
    forces = -coords.grad.clone()
    virial = -torch.einsum('ij,ik->jk', coords, forces)

    labels = ['C1', 'C2', 'H11', 'H12', 'H13', 'H21', 'H22', 'H23']
    print(f"\nForces (first ethane):")
    for i, lbl in enumerate(labels):
        print(f"  {lbl}: [{forces[i,0].item():20.15f}, {forces[i,1].item():20.15f}, {forces[i,2].item():20.15f}]")
    fs = forces.sum(0)
    print(f"\nForce sum: [{fs[0].item():.6e}, {fs[1].item():.6e}, {fs[2].item():.6e}]")
    print(f"\nVirial:")
    for i in range(3):
        print(f"  [{virial[i,0].item():20.15f}, {virial[i,1].item():20.15f}, {virial[i,2].item():20.15f}]")
    print(f"{'='*80}\n")

    return {
        'energy': total_energy.item(),
        'forces': forces.detach().cpu().numpy(),
        'virial': virial.detach().cpu().numpy(),
    }
