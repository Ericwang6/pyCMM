import pytest
import torch
import sys
import openmm.app as app

sys.path.insert(0, "/pscratch/sd/a/asa/software/pyCMM")
from cmm.bonded import computeBondBondCoupling, computeAngle
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

# AngleAngle parameters from XML (ethane terms only)
# atomIndices are 6-wide: [a1, vertex1, a2,  b1, vertex2, b2]
# ctype=1: same-carbon H-C-H/H-C-C pairs, ctype=2: cross-carbon pairs

# c.eth-c.eth-h.eth / c.eth-c.eth-h.eth  ctype=1
AA_THETA_EQ_1_T1 = 1.94202996
AA_THETA_EQ_2_T1 = 1.94202996
AA_K_T1          = -0.0264544521

# c.eth-c.eth-h.eth / c.eth-c.eth-h.eth  ctype=2
AA_THETA_EQ_1_T2 = 1.94202996
AA_THETA_EQ_2_T2 = 1.94202996
AA_K_T2          =  0.0268238046

# h.eth-c.eth-h.eth / c.eth-c.eth-h.eth  ctype=1
AA_THETA_EQ_1_T3 = 1.87821616
AA_THETA_EQ_2_T3 = 1.94202996
AA_K_T3          = -0.00187590313

# h.eth-c.eth-h.eth / h.eth-c.eth-h.eth  ctype=1
AA_THETA_EQ_1_T4 = 1.87821616
AA_THETA_EQ_2_T4 = 1.87821616
AA_K_T4          =  0.0133855157


def build_angle_angle_indices(n_molecules: int, device):
    """
    Build 6-column angle-angle indices: [a1, vertex1, a2,  b1, vertex2, b2]
    cols [:3] = angle1, cols [-3:] = angle2, matching system.py convention.

    For ethane we have four coupling types per carbon center:
      T1 (ctype=1): H-C-C / H-C-C  (two different H-C-C angles on same carbon, ctype=1)
      T2 (ctype=2): H-C-C / H-C-C  (same pair, ctype=2 cross term)
      T3 (ctype=1): H-C-H / H-C-C
      T4 (ctype=1): H-C-H / H-C-H

    Each carbon has 3 H atoms giving 3 H-C-C angles and 3 H-C-H angles.
    We generate all unique pairs for each type.
    """
    rows_t1, rows_t2, rows_t3, rows_t4 = [], [], [], []

    for m in range(n_molecules):
        base = m * 8

        # Process each carbon center separately
        for c_off, h_offs, other_c_off in [
            (C1_OFF, H1_OFFS, C2_OFF),
            (C2_OFF, H2_OFFS, C1_OFF),
        ]:
            c      = base + c_off
            other_c = base + other_c_off
            hs     = [base + h for h in h_offs]

            # H-C-C angles: [H, C, other_C]
            hcc_angles = [[h, c, other_c] for h in hs]

            # H-C-H angles: [Ha, C, Hb] for all unique pairs
            hch_angles = []
            for idx_a in range(len(hs)):
                for idx_b in range(idx_a + 1, len(hs)):
                    hch_angles.append([hs[idx_a], c, hs[idx_b]])

            # T1/T2: H-C-C / H-C-C  (unique pairs of distinct H-C-C angles)
            for idx_a in range(len(hcc_angles)):
                for idx_b in range(idx_a + 1, len(hcc_angles)):
                    a1, v1, a2 = hcc_angles[idx_a]
                    b1, v2, b2 = hcc_angles[idx_b]
                    rows_t1.append([a1, v1, a2,  b1, v2, b2])
                    rows_t2.append([a1, v1, a2,  b1, v2, b2])

            # T3: H-C-H / H-C-C
            for hch in hch_angles:
                for hcc in hcc_angles:
                    a1, v1, a2 = hch
                    b1, v2, b2 = hcc
                    rows_t3.append([a1, v1, a2,  b1, v2, b2])

            # T4: H-C-H / H-C-H  (unique pairs)
            for idx_a in range(len(hch_angles)):
                for idx_b in range(idx_a + 1, len(hch_angles)):
                    a1, v1, a2 = hch_angles[idx_a]
                    b1, v2, b2 = hch_angles[idx_b]
                    rows_t4.append([a1, v1, a2,  b1, v2, b2])

    def to_tensor(rows):
        return torch.tensor(rows, device=device, dtype=torch.int32)

    return to_tensor(rows_t1), to_tensor(rows_t2), to_tensor(rows_t3), to_tensor(rows_t4)


def compute_angle_angle_energy(coords, n_molecules, device, dtype):
    """Compute angle-angle coupling energy matching system.py exactly."""

    aa_idx_t1, aa_idx_t2, aa_idx_t3, aa_idx_t4 = build_angle_angle_indices(n_molecules, device)

    def aa_energy(aa_idx, theta_eq_1_val, theta_eq_2_val, k_val):
        angles1 = computeAngle(coords, aa_idx[:, :3], box=None, boxInv=None)
        angles2 = computeAngle(coords, aa_idx[:, 3:], box=None, boxInv=None)
        n       = aa_idx.shape[0]
        teq1    = torch.full((n,), theta_eq_1_val, device=device, dtype=dtype)
        teq2    = torch.full((n,), theta_eq_2_val, device=device, dtype=dtype)
        k       = torch.full((n,), k_val,           device=device, dtype=dtype)
        return computeBondBondCoupling(
            torch.cos(angles1), torch.cos(angles2),
            torch.cos(teq1),    torch.cos(teq2),
            k
        ).sum()

    e_t1 = aa_energy(aa_idx_t1, AA_THETA_EQ_1_T1, AA_THETA_EQ_2_T1, AA_K_T1)
    e_t2 = aa_energy(aa_idx_t2, AA_THETA_EQ_1_T2, AA_THETA_EQ_2_T2, AA_K_T2)
    e_t3 = aa_energy(aa_idx_t3, AA_THETA_EQ_1_T3, AA_THETA_EQ_2_T3, AA_K_T3)
    e_t4 = aa_energy(aa_idx_t4, AA_THETA_EQ_1_T4, AA_THETA_EQ_2_T4, AA_K_T4)

    return {
        'e_t1': e_t1, 'e_t2': e_t2, 'e_t3': e_t3, 'e_t4': e_t4,
        'total': e_t1 + e_t2 + e_t3 + e_t4,
        'n_t1': aa_idx_t1.shape[0], 'n_t2': aa_idx_t2.shape[0],
        'n_t3': aa_idx_t3.shape[0], 'n_t4': aa_idx_t4.shape[0],
    }


# ---------------------------------------------------------------------------
# Single molecule reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float32),
    ('cuda', torch.float64),
])
def test_angle_angle_reference_single(device, dtype):
    """Reference energy and forces for angle-angle coupling - single ethane."""

    pdb    = app.PDBFile(PDB_PATH)
    pos    = pdb.getPositions(asNumpy=True)._value / BOHR2NM
    coords = torch.tensor(pos[:8], device=device, dtype=dtype, requires_grad=True)

    results      = compute_angle_angle_energy(coords, 1, device, dtype)
    total_energy = results['total']

    print(f"\n{'='*60}")
    print(f"Angle-Angle Coupling - Single Ethane ({dtype})")
    print(f"{'='*60}")
    print(f"  Pairs — T1(HCC/HCC ctype1): {results['n_t1']}  "
          f"T2(HCC/HCC ctype2): {results['n_t2']}  "
          f"T3(HCH/HCC): {results['n_t3']}  "
          f"T4(HCH/HCH): {results['n_t4']}")
    print(f"\nEnergy breakdown:")
    print(f"  T1 (HCC/HCC ctype1): {results['e_t1'].item():.17f}")
    print(f"  T2 (HCC/HCC ctype2): {results['e_t2'].item():.17f}")
    print(f"  T3 (HCH/HCC):        {results['e_t3'].item():.17f}")
    print(f"  T4 (HCH/HCH):        {results['e_t4'].item():.17f}")
    print(f"  Total:               {total_energy.item():.17f}")

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
def test_angle_angle_ethane_196(device, dtype):
    """Reference energy, forces, and virial for angle-angle coupling - 196 ethanes."""

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

    energies  = system.getEnergy(coords, box)
    aa_energy = energies["angle_angle"]  # adjust key to match your API

    print(f"\n{'='*80}")
    print(f"Angle-Angle Coupling Reference - 196 Ethanes ({dtype})")
    print(f"{'='*80}")
    print(f"Atoms: {coords.shape[0]}  |  Ethanes: {coords.shape[0]//8}  |  Box: {box.diagonal()}")
    print(f"\nTotal Angle-Angle Energy: {aa_energy.item():.17f}")

    torch._functorch.config.donated_buffer = False

    if coords.grad is not None:
        coords.grad.zero_()
    aa_energy.backward()
    forces = -coords.grad.clone()
    virial = -torch.einsum('ij,ik->jk', coords, forces)

    labels = ['C1', 'C2', 'H11', 'H12', 'H13', 'H21', 'H22', 'H23']
    print(f"\nForces (first ethane):")
    for i, lbl in enumerate(labels):
        print(f"  {lbl}: [{forces[i,0].item():20.15f}, "
              f"{forces[i,1].item():20.15f}, "
              f"{forces[i,2].item():20.15f}]")
    fs = forces.sum(0)
    print(f"\nForce sum: [{fs[0].item():.6e}, {fs[1].item():.6e}, {fs[2].item():.6e}]")
    print(f"\nVirial:")
    for i in range(3):
        print(f"  [{virial[i,0].item():20.15f}, "
              f"{virial[i,1].item():20.15f}, "
              f"{virial[i,2].item():20.15f}]")
    print(f"Virial trace: {virial.trace().item():.17f}")
    print(f"{'='*80}\n")

    assert torch.allclose(forces.sum(0), torch.zeros(3, device=device, dtype=dtype), atol=1e-6)

    with open('ene_angle_angle.dat', 'w') as f:
        f.write('# reference angle-angle coupling energy\n')
        f.write(f'{aa_energy.item():.12f}\n')

    with open('ref_grad_angle_angle.dat', 'w') as f:
        f.write('# reference angle-angle coupling forces\n')
        fn = forces.detach().cpu().numpy()
        for i in range(fn.shape[0]):
            f.write(f'{i} {fn[i,0]:.13f} {fn[i,1]:.13f} {fn[i,2]:.13f}\n')

    with open('virial_angle_angle_ref.dat', 'w') as f:
        f.write('# xx xy xz yx yy yz zx zy zz\n')
        v = virial.detach().cpu().numpy()
        f.write(f'{v[0,0]:.14f} {v[0,1]:.14f} {v[0,2]:.14f} '
                f'{v[1,0]:.14f} {v[1,1]:.14f} {v[1,2]:.14f} '
                f'{v[2,0]:.14f} {v[2,1]:.14f} {v[2,2]:.14f}\n')

    print("Written: ene_angle_angle.dat  ref_grad_angle_angle.dat  virial_angle_angle_ref.dat")


# ---------------------------------------------------------------------------
# 196-molecule manual test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float64),
])
def test_angle_angle_ethane_196_manual(device, dtype):
    """Manual angle-angle coupling for 196 ethanes without ForceFieldXML."""

    pdb    = app.PDBFile(PDB_PATH)
    coords = torch.tensor(
        pdb.getPositions(asNumpy=True)._value / BOHR2NM,
        device=device, dtype=dtype, requires_grad=True
    )

    n_ethanes    = coords.shape[0] // 8
    results      = compute_angle_angle_energy(coords, n_ethanes, device, dtype)
    total_energy = results['total']

    print(f"\n{'='*80}")
    print(f"Manual Angle-Angle Coupling - 196 Ethanes ({dtype})")
    print(f"{'='*80}")
    print(f"Ethanes: {n_ethanes}")
    print(f"  Pairs — T1: {results['n_t1']}  T2: {results['n_t2']}  "
          f"T3: {results['n_t3']}  T4: {results['n_t4']}")
    print(f"\nEnergy breakdown:")
    print(f"  T1 (HCC/HCC ctype1): {results['e_t1'].item():.17f}")
    print(f"  T2 (HCC/HCC ctype2): {results['e_t2'].item():.17f}")
    print(f"  T3 (HCH/HCC):        {results['e_t3'].item():.17f}")
    print(f"  T4 (HCH/HCH):        {results['e_t4'].item():.17f}")
    print(f"  Total:               {total_energy.item():.17f}")

    total_energy.backward()
    forces = -coords.grad.clone()
    virial = -torch.einsum('ij,ik->jk', coords, forces)

    labels = ['C1', 'C2', 'H11', 'H12', 'H13', 'H21', 'H22', 'H23']
    print(f"\nForces (first ethane):")
    for i, lbl in enumerate(labels):
        print(f"  {lbl}: [{forces[i,0].item():20.15f}, "
              f"{forces[i,1].item():20.15f}, "
              f"{forces[i,2].item():20.15f}]")
    fs = forces.sum(0)
    print(f"\nForce sum: [{fs[0].item():.6e}, {fs[1].item():.6e}, {fs[2].item():.6e}]")
    print(f"\nVirial:")
    for i in range(3):
        print(f"  [{virial[i,0].item():20.15f}, "
              f"{virial[i,1].item():20.15f}, "
              f"{virial[i,2].item():20.15f}]")
    print(f"Virial trace: {virial.trace().item():.17f}")
    print(f"{'='*80}\n")

    return {
        'energy': total_energy.item(),
        'forces': forces.detach().cpu().numpy(),
        'virial': virial.detach().cpu().numpy(),
    }
@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float64),
])
def test_angle_angle_analytic_forces(device, dtype):
    """Compare autograd vs analytic forces for angle-angle coupling - 196 ethanes."""

    pdb    = app.PDBFile(PDB_PATH)
    coords = torch.tensor(
        pdb.getPositions(asNumpy=True)._value / BOHR2NM,
        device=device, dtype=dtype, requires_grad=True
    )

    n_ethanes = coords.shape[0] // 8

    # --- Autograd forces ---
    results      = compute_angle_angle_energy(coords, n_ethanes, device, dtype)
    total_energy = results['total']
    total_energy.backward()
    forces_autograd = -coords.grad.clone()

    # --- Analytic forces ---
    # E = K * (cos(theta) - cos(theta0)) * (cos(theta') - cos(theta0'))
    #
    # dE/d(cos_theta)  = K * (cos(theta') - cos(theta0'))  → Wilson grad on angle1 atoms
    # dE/d(cos_theta') = K * (cos(theta)  - cos(theta0))   → Wilson grad on angle2 atoms
    #
    # Wilson gradient of cos(theta) for angle [A, vertex, C]:
    #   d(cos)/dr_A      = (uBC - cos*uBA) / |rBA|
    #   d(cos)/dr_C      = (uBA - cos*uBC) / |rBC|
    #   d(cos)/dr_vertex = -(d/dr_A + d/dr_C)
    #
    # F_i = -dE/dr_i

    coords_d = coords.detach()
    device_  = coords_d.device
    dtype_   = coords_d.dtype

    aa_idx_t1, aa_idx_t2, aa_idx_t3, aa_idx_t4 = build_angle_angle_indices(n_ethanes, device_)

    forces_analytic = torch.zeros_like(coords_d)

    def wilson_grad(aa_idx, k_val, teq1_val, teq2_val):
        """Accumulate analytic angle-angle forces for one coupling type."""
        n   = aa_idx.shape[0]
        idx1 = aa_idx[:, :3]   # [A, vertex1, C]  — angle 1
        idx2 = aa_idx[:, 3:]   # [B, vertex2, D]  — angle 2

        # Angle 1 geometry
        rBA1      = coords_d[idx1[:, 0]] - coords_d[idx1[:, 1]]  # A - vertex1
        rBC1      = coords_d[idx1[:, 2]] - coords_d[idx1[:, 1]]  # C - vertex1
        rBA1_norm = torch.norm(rBA1, dim=-1, keepdim=True)
        rBC1_norm = torch.norm(rBC1, dim=-1, keepdim=True)
        uBA1      = rBA1 / rBA1_norm
        uBC1      = rBC1 / rBC1_norm
        cos1      = torch.sum(uBA1 * uBC1, dim=-1, keepdim=True)  # (N,1)

        # Angle 2 geometry
        rBA2      = coords_d[idx2[:, 0]] - coords_d[idx2[:, 1]]  # B - vertex2
        rBC2      = coords_d[idx2[:, 2]] - coords_d[idx2[:, 1]]  # D - vertex2
        rBA2_norm = torch.norm(rBA2, dim=-1, keepdim=True)
        rBC2_norm = torch.norm(rBC2, dim=-1, keepdim=True)
        uBA2      = rBA2 / rBA2_norm
        uBC2      = rBC2 / rBC2_norm
        cos2      = torch.sum(uBA2 * uBC2, dim=-1, keepdim=True)  # (N,1)

        cos_eq1   = torch.cos(torch.tensor(teq1_val, device=device_, dtype=dtype_))
        cos_eq2   = torch.cos(torch.tensor(teq2_val, device=device_, dtype=dtype_))
        dcos1     = cos1 - cos_eq1   # (N,1)
        dcos2     = cos2 - cos_eq2

        # dE/d(cos1) = K * dcos2,  dE/d(cos2) = K * dcos1
        dE_dcos1  = k_val * dcos2   # (N,1)
        dE_dcos2  = k_val * dcos1

        # Wilson gradients — no sin
        dcos1_drA      = (uBC1 - cos1 * uBA1) / rBA1_norm   # (N,3)
        dcos1_drC      = (uBA1 - cos1 * uBC1) / rBC1_norm
        dcos1_drVertex = -(dcos1_drA + dcos1_drC)

        dcos2_drB      = (uBC2 - cos2 * uBA2) / rBA2_norm
        dcos2_drD      = (uBA2 - cos2 * uBC2) / rBC2_norm
        dcos2_drVertex = -(dcos2_drB + dcos2_drD)

        # F = -dE/dr = -dE/d(cos) * d(cos)/dr
        F_A       = -dE_dcos1 * dcos1_drA
        F_C       = -dE_dcos1 * dcos1_drC
        F_V1      = -dE_dcos1 * dcos1_drVertex
        F_B       = -dE_dcos2 * dcos2_drB
        F_D       = -dE_dcos2 * dcos2_drD
        F_V2      = -dE_dcos2 * dcos2_drVertex

        forces_analytic.index_add_(0, idx1[:, 0].long(), F_A)
        forces_analytic.index_add_(0, idx1[:, 2].long(), F_C)
        forces_analytic.index_add_(0, idx1[:, 1].long(), F_V1)
        forces_analytic.index_add_(0, idx2[:, 0].long(), F_B)
        forces_analytic.index_add_(0, idx2[:, 2].long(), F_D)
        forces_analytic.index_add_(0, idx2[:, 1].long(), F_V2)

    wilson_grad(aa_idx_t1, AA_K_T1, AA_THETA_EQ_1_T1, AA_THETA_EQ_2_T1)
    wilson_grad(aa_idx_t2, AA_K_T2, AA_THETA_EQ_1_T2, AA_THETA_EQ_2_T2)
    wilson_grad(aa_idx_t3, AA_K_T3, AA_THETA_EQ_1_T3, AA_THETA_EQ_2_T3)
    wilson_grad(aa_idx_t4, AA_K_T4, AA_THETA_EQ_1_T4, AA_THETA_EQ_2_T4)

    # --- Compare ---
    max_diff = (forces_autograd - forces_analytic).abs().max().item()

    labels = ['C1', 'C2', 'H11', 'H12', 'H13', 'H21', 'H22', 'H23']
    print(f"\n{'='*80}")
    print(f"Angle-Angle Analytic Force Test - 196 Ethanes ({dtype})")
    print(f"{'='*80}")
    print(f"\nAutograd forces (first ethane):")
    for i, lbl in enumerate(labels):
        print(f"  {lbl}: [{forces_autograd[i,0].item():20.15f}, "
              f"{forces_autograd[i,1].item():20.15f}, "
              f"{forces_autograd[i,2].item():20.15f}]")
    print(f"\nAnalytic forces (first ethane):")
    for i, lbl in enumerate(labels):
        print(f"  {lbl}: [{forces_analytic[i,0].item():20.15f}, "
              f"{forces_analytic[i,1].item():20.15f}, "
              f"{forces_analytic[i,2].item():20.15f}]")
    print(f"\nMax force difference: {max_diff:.6e}")
    fs_auto = forces_autograd.sum(0)
    fs_anal = forces_analytic.sum(0)
    print(f"Force sum (autograd): [{fs_auto[0].item():.6e}, {fs_auto[1].item():.6e}, {fs_auto[2].item():.6e}]")
    print(f"Force sum (analytic): [{fs_anal[0].item():.6e}, {fs_anal[1].item():.6e}, {fs_anal[2].item():.6e}]")
    print(f"{'='*80}\n")

    assert torch.allclose(forces_autograd, forces_analytic, atol=1e-10), \
        f"Analytic and autograd forces differ by max {max_diff:.6e}"
