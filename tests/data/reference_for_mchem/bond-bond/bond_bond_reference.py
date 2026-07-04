import pytest
import torch
import sys
import openmm.app as app

sys.path.insert(0, "/pscratch/sd/a/asa/software/pyCMM")
from cmm.bonded import computeBondBondCoupling, computeBond
from cmm.ffxml import ForceFieldXML
from cmm.topology import Topology
from cmm.units import BOHR2NM
from cmm.units import BOHR2ANG


@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float32),
    ('cuda', torch.float64),
])
def test_bond_bond_coupling_reference(device, dtype):
    """Generate reference energy and forces for bond-bond coupling - single water"""

    # Same single water molecule coords as other tests
    #coords = torch.tensor([
    #    [-2.722762, 15.350993, -0.920138],  # O
    #    [-2.603766, 14.476160, -1.273219],  # H1
    #    [-3.543810, 15.582853, -1.332255],  # H2
    #], device=device, dtype=dtype, requires_grad=True)
    ANG2BOHR = 1.0 / BOHR2ANG

    coords = torch.tensor([
        [-2.722762, 15.350993, -0.920138],
        [-2.603766, 14.476160, -1.273219],
        [-3.543810, 15.582853, -1.332255],
    ], device=device, dtype=dtype) * ANG2BOHR
    coords = coords.requires_grad_(True)

    # O-H1 and O-H2 bonds
    bonds1 = torch.tensor([[0, 1]], device=device, dtype=torch.int32)  # O-H1
    bonds2 = torch.tensor([[0, 2]], device=device, dtype=torch.int32)  # O-H2

    # Parameters from XML — req is shared for both O-H bonds
    req  = torch.tensor(1.81211318, device=device, dtype=dtype)
    # !! UPDATE k_bb to your actual bond-bond coupling force constant from XML !!
    k_bb = torch.tensor(-0.0065212683916988, device=device, dtype=dtype)  # placeholder

    r1 = computeBond(coords, bonds1, box=None, boxInv=None)
    r2 = computeBond(coords, bonds2, box=None, boxInv=None)

    energy = computeBondBondCoupling(r1, r2, req, req, k_bb)
    total_energy = energy.sum()
    print("COORDS")
    print(coords)

    print(f"\n{'='*60}")
    print(f"Bond-Bond Coupling Test Results ({dtype})")
    print(f"{'='*60}")
    print(f"r1 (O-H1): {r1.item():.15f}")
    print(f"r2 (O-H2): {r2.item():.15f}")
    print(f"req:       {req.item():.15f}")
    print(f"dr1:       {(r1 - req).item():.15f}")
    print(f"dr2:       {(r2 - req).item():.15f}")
    print(f"Total energy: {total_energy.item():.17f}")

    total_energy.backward()
    forces = -coords.grad
    virial = -torch.einsum('ij,ik->jk', coords, forces)

    print(f"\nForces on atoms:")
    print(f"O  force: [{forces[0,0].item():20.15f}, {forces[0,1].item():20.15f}, {forces[0,2].item():20.15f}]")
    print(f"H1 force: [{forces[1,0].item():20.15f}, {forces[1,1].item():20.15f}, {forces[1,2].item():20.15f}]")
    print(f"H2 force: [{forces[2,0].item():20.15f}, {forces[2,1].item():20.15f}, {forces[2,2].item():20.15f}]")
    print(f"Force sum: [{forces.sum(0)[0].item():20.15f}, {forces.sum(0)[1].item():20.15f}, {forces.sum(0)[2].item():20.15f}]")
    print(f"{'='*60}\n")

    assert torch.allclose(forces.sum(0), torch.zeros(3, device=device, dtype=dtype), atol=1e-6)
    print(f"\nVirial Tensor:")
    for i in range(3):
        print(f"[{virial[i,0].item():20.15f}, {virial[i,1].item():20.15f}, {virial[i,2].item():20.15f}]")
    print(f"Virial trace: {virial.trace().item():.17f}")

    return {
        'r1': r1.item(), 'r2': r2.item(),
        'energy': total_energy.item(),
        'forces': forces.detach().cpu().numpy(),
    }


@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float32),
    ('cuda', torch.float64),
])
def test_bond_bond_coupling_at_equilibrium(device, dtype):
    """Verify zero energy and forces when both bonds are at equilibrium"""

    req_val = 1.81211318
    coords = torch.tensor([
        [0.0,      0.0, 0.0],   # O
        [req_val,  0.0, 0.0],   # H1 — exactly at req along x
        [0.0,  req_val, 0.0],   # H2 — exactly at req along y
    ], device=device, dtype=dtype, requires_grad=True)

    bonds1 = torch.tensor([[0, 1]], device=device, dtype=torch.int32)
    bonds2 = torch.tensor([[0, 2]], device=device, dtype=torch.int32)

    req  = torch.tensor(req_val, device=device, dtype=dtype)
    k_bb = torch.tensor(-0.0065212683916988, device=device, dtype=dtype)  # nonzero k to test properly

    r1 = computeBond(coords, bonds1, box=None, boxInv=None)
    r2 = computeBond(coords, bonds2, box=None, boxInv=None)

    energy = computeBondBondCoupling(r1, r2, req, req, k_bb).sum()

    print(f"\nAt equilibrium:")
    print(f"r1={r1.item():.10f}  r2={r2.item():.10f}  req={req.item():.10f}")
    print(f"Energy: {energy.item():.10f} (should be ~0)")

    assert torch.allclose(energy, torch.zeros_like(energy), atol=1e-6)

    energy.backward()
    forces = -coords.grad
    assert torch.allclose(forces, torch.zeros_like(forces), atol=1e-5)


@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float64),
])
def test_bond_bond_coupling_minv_clamp(device, dtype):
    """Verify the minv floor is applied when energy would be negative"""

    # Place H atoms closer than req on both bonds → (r-req) negative on both
    # → product positive → no clamp. Stretch one, compress other → negative product → clamp.
    req_val = 1.81211318
    coords = torch.tensor([
        [0.0,       0.0, 0.0],   # O
        [req_val * 1.1, 0.0, 0.0],  # H1 stretched  (r1 > req  → dr1 > 0)
        [0.0, req_val * 0.9, 0.0],  # H2 compressed (r2 < req  → dr2 < 0)
    ], device=device, dtype=dtype, requires_grad=True)

    bonds1 = torch.tensor([[0, 1]], device=device, dtype=torch.int32)
    bonds2 = torch.tensor([[0, 2]], device=device, dtype=torch.int32)

    req  = torch.tensor(req_val, device=device, dtype=dtype)
    k_bb = torch.tensor(1.0, device=device, dtype=dtype)  # large k to guarantee clamp triggers
    minv = -0.002

    r1 = computeBond(coords, bonds1, box=None, boxInv=None)
    r2 = computeBond(coords, bonds2, box=None, boxInv=None)

    raw_energy = k_bb * (r1 - req) * (r2 - req)
    energy = computeBondBondCoupling(r1, r2, req, req, k_bb)

    print(f"\nClamp test:")
    print(f"r1={r1.item():.8f}  r2={r2.item():.8f}")
    print(f"Raw energy:    {raw_energy.item():.10f}")
    print(f"Clamped energy:{energy.item():.10f}  (minv={minv})")

    if raw_energy.item() < minv:
        assert torch.allclose(energy, torch.tensor(minv, device=device, dtype=dtype), atol=1e-10)
        print("Clamp was applied correctly.")
    else:
        assert torch.allclose(energy, raw_energy, atol=1e-10)
        print("No clamp needed.")


@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float64),
])
def test_bond_bond_coupling_water_216(device, dtype):
    """Generate reference energy, forces, and virial for bond-bond coupling - 216 waters"""

    pdb_path = "/pscratch/sd/a/asa/software/pyCMM/tests/data/water_216.pdb"
    ff_path  = "/pscratch/sd/a/asa/software/pyCMM/tests/data/water-aalim.xml"

    pdb = app.PDBFile(pdb_path)
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

    ff = ForceFieldXML(ff_path, device=device)
    system = ff.parametrize(
        top,
        use_fd_morse=False,
        use_polarization=False,
        use_hardness_change=False,
        cutoff_sr=9.0,
        use_switch=True,
        use_customized_ops=False
    )

    energies = system.getEnergy(coords, box)
    bb_energy = energies["bond_bond"]  # adjust key to match your API

    print(f"\n{'='*80}")
    print(f"Bond-Bond Coupling Reference Data - 216 Waters ({dtype})")
    print(f"{'='*80}")
    print(f"Number of atoms:  {coords.shape[0]}")
    print(f"Number of waters: {coords.shape[0] // 3}")
    print(f"Box dimensions:   {box.diagonal()}")
    print(f"\nTotal Bond-Bond Coupling Energy: {bb_energy.item():.17f}")

    bb_energy.backward()
    forces = -coords.grad.clone()
    virial = -torch.einsum('ij,ik->jk', coords, forces)

    print(f"\nForces (first 3 atoms - first water):")
    for i in range(3):
        print(f"Atom {i}: [{forces[i,0].item():20.15f}, "
              f"{forces[i,1].item():20.15f}, "
              f"{forces[i,2].item():20.15f}]")

    print(f"\nForce sum (should be ~0):")
    fs = forces.sum(0)
    print(f"[{fs[0].item():20.15f}, {fs[1].item():20.15f}, {fs[2].item():20.15f}]")

    print(f"\nVirial Tensor (3x3):")
    for i in range(3):
        print(f"[{virial[i,0].item():20.15f}, "
              f"{virial[i,1].item():20.15f}, "
              f"{virial[i,2].item():20.15f}]")
    print(f"Virial trace: {virial.trace().item():.17f}")
    print(f"{'='*80}\n")

    assert torch.allclose(forces.sum(0), torch.zeros(3, device=device, dtype=dtype), atol=1e-6)

    with open('ene_bb_coupling_test.dat', 'w') as f:
        f.write('# reference bond-bond coupling energy\n')
        f.write(f'{bb_energy.item():.12f}\n')

    with open('ref_grad_bb_coupling.dat', 'w') as f:
        f.write('# reference bond-bond coupling forces\n')
        forces_np = forces.detach().cpu().numpy()
        for i in range(forces_np.shape[0]):
            f.write(f'{i} {forces_np[i,0]:.13f} {forces_np[i,1]:.13f} {forces_np[i,2]:.13f}\n')

    with open('virial_bb_coupling_ref.dat', 'w') as f:
        f.write('# xx xy xz yx yy yz zx zy zz\n')
        v = virial.detach().cpu().numpy()
        f.write(f'{v[0,0]:.14f} {v[0,1]:.14f} {v[0,2]:.14f} '
                f'{v[1,0]:.14f} {v[1,1]:.14f} {v[1,2]:.14f} '
                f'{v[2,0]:.14f} {v[2,1]:.14f} {v[2,2]:.14f}\n')

    print("Written: ene_bb_coupling_test.dat, ref_grad_bb_coupling.dat, virial_bb_coupling_ref.dat")

    return {
        'energy': bb_energy.item(),
        'forces': forces.detach().cpu().numpy(),
        'virial': virial.detach().cpu().numpy(),
    }


@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float64),
])
def test_bond_bond_coupling_water_216_manual(device, dtype):
    """Manual bond-bond coupling calculation for 216 waters without ForceFieldXML"""

    pdb_path = "/pscratch/sd/a/asa/software/pyCMM/tests/data/water_216.pdb"

    pdb = app.PDBFile(pdb_path)
    coords = torch.tensor(
        pdb.getPositions(asNumpy=True)._value / BOHR2NM,
        device=device, dtype=dtype, requires_grad=True
    )

    n_waters = coords.shape[0] // 3

    # One O-H1/O-H2 pair per water
    bonds1_idx = torch.tensor([[i*3, i*3+1] for i in range(n_waters)], device=device, dtype=torch.int32)
    bonds2_idx = torch.tensor([[i*3, i*3+2] for i in range(n_waters)], device=device, dtype=torch.int32)

    req  = torch.tensor(1.81211318, device=device, dtype=dtype)
    # !! UPDATE k_bb to your actual value from XML !!
    k_bb = torch.tensor(-0.0065212683916988, device=device, dtype=dtype)  # placeholder

    r1 = computeBond(coords, bonds1_idx, box=None, boxInv=None)
    r2 = computeBond(coords, bonds2_idx, box=None, boxInv=None)

    energies = computeBondBondCoupling(r1, r2, req, req, k_bb)
    total_energy = energies.sum()

    print(f"\n{'='*80}")
    print(f"Manual Bond-Bond Coupling - 216 Waters ({dtype})")
    print(f"{'='*80}")
    print(f"Number of bond pairs: {n_waters}")
    print(f"Mean r1: {r1.mean().item():.6f}  Mean r2: {r2.mean().item():.6f}  req: {req.item():.6f}")
    print(f"Total energy: {total_energy.item():.17f}")

    total_energy.backward()
    forces = -coords.grad
    virial = -torch.einsum('ij,ik->jk', coords, forces)

    print(f"\nVirial Tensor:")
    for i in range(3):
        print(f"[{virial[i,0].item():20.15f}, "
              f"{virial[i,1].item():20.15f}, "
              f"{virial[i,2].item():20.15f}]")
    print(f"{'='*80}\n")

    return {
        'energy': total_energy.item(),
        'forces': forces.detach().cpu().numpy(),
        'virial': virial.detach().cpu().numpy(),
    }


@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float64),
])
def test_bond_bond_coupling_analytic_forces(device, dtype):
    """Compare autograd forces against analytic forces for bond-bond coupling."""

    pdb_path = "/pscratch/sd/a/asa/software/pyCMM/tests/data/water_216.pdb"

    pdb    = app.PDBFile(pdb_path)
    coords = torch.tensor(
        pdb.getPositions(asNumpy=True)._value / BOHR2NM,
        device=device, dtype=dtype, requires_grad=True
    )

    n_waters   = coords.shape[0] // 3
    bonds1_idx = torch.tensor([[i*3, i*3+1] for i in range(n_waters)], device=device, dtype=torch.int32)
    bonds2_idx = torch.tensor([[i*3, i*3+2] for i in range(n_waters)], device=device, dtype=torch.int32)

    req  = torch.tensor(1.81211318, device=device, dtype=dtype)
    k_bb = torch.tensor(-0.0065212683916988, device=device, dtype=dtype)

    # --- Autograd forces ---
    r1 = computeBond(coords, bonds1_idx, box=None, boxInv=None)
    r2 = computeBond(coords, bonds2_idx, box=None, boxInv=None)
    total_energy = computeBondBondCoupling(r1, r2, req, req, k_bb).sum()
    total_energy.backward()
    forces_autograd = -coords.grad.clone()

    # --- Analytic forces ---
    # E = k * (r1 - req) * (r2 - req)
    # dE/d(atom_i) = k * (r2 - req) * d(r1)/d(atom_i)
    #              + k * (r1 - req) * d(r2)/d(atom_i)
    #
    # d(r)/d(atom_j) = +unit_vec  (away from other atom, for the j-th atom)
    # d(r)/d(atom_i) = -unit_vec  (toward other atom,    for the i-th atom)
    #
    # Bond 1: O(i*3) -> H1(i*3+1),  vec = H1 - O
    # Bond 2: O(i*3) -> H2(i*3+2),  vec = H2 - O

    coords_detached = coords.detach()

    vec1     = coords_detached[bonds1_idx[:, 1]] - coords_detached[bonds1_idx[:, 0]]  # H1 - O
    vec2     = coords_detached[bonds2_idx[:, 1]] - coords_detached[bonds2_idx[:, 0]]  # H2 - O
    r1_d     = torch.norm(vec1, dim=-1, keepdim=True)
    r2_d     = torch.norm(vec2, dim=-1, keepdim=True)
    uv1      = vec1 / r1_d   # unit vec along bond1
    uv2      = vec2 / r2_d   # unit vec along bond2

    dr1      = (r1 - req).detach()   # (n_waters,)
    dr2      = (r2 - req).detach()

    # clamp mask: where energy was clamped to minv, gradient is zero
    raw_e    = k_bb * dr1 * dr2
    minv     = -0.002
    clamped  = raw_e < minv           # (n_waters,) bool

    # dE/d(bond1) = k * dr2  [scalar per bond pair]
    dE_dr1   = torch.where(clamped, torch.zeros_like(dr1), k_bb * dr2)  # (n_waters,)
    dE_dr2   = torch.where(clamped, torch.zeros_like(dr2), k_bb * dr1)

    forces_analytic = torch.zeros_like(coords_detached)

    # Bond 1 contribution: O gets -dE_dr1 * uv1, H1 gets +dE_dr1 * uv1
    dE_dr1_3d = dE_dr1.unsqueeze(1)   # (n_waters, 1)
    forces_analytic.index_add_(0, bonds1_idx[:, 0].long(), -dE_dr1_3d * uv1)  # O
    forces_analytic.index_add_(0, bonds1_idx[:, 1].long(),  dE_dr1_3d * uv1)  # H1

    # Bond 2 contribution: O gets -dE_dr2 * uv2, H2 gets +dE_dr2 * uv2
    dE_dr2_3d = dE_dr2.unsqueeze(1)
    forces_analytic.index_add_(0, bonds2_idx[:, 0].long(), -dE_dr2_3d * uv2)  # O
    forces_analytic.index_add_(0, bonds2_idx[:, 1].long(),  dE_dr2_3d * uv2)  # H2
    # F = -dE/dq, so negate the whole thing
    forces_analytic = -forces_analytic

    print(f"\n{'='*80}")
    print(f"Bond-Bond Analytic Force Test - 216 Waters ({dtype})")
    print(f"{'='*80}")
    print(f"\nAutograd forces (first water):")
    labels = ['O', 'H1', 'H2']
    for i, lbl in enumerate(labels):
        print(f"  {lbl}: [{forces_autograd[i,0].item():20.15f}, "
              f"{forces_autograd[i,1].item():20.15f}, "
              f"{forces_autograd[i,2].item():20.15f}]")

    print(f"\nAnalytic forces (first water):")
    for i, lbl in enumerate(labels):
        print(f"  {lbl}: [{forces_analytic[i,0].item():20.15f}, "
              f"{forces_analytic[i,1].item():20.15f}, "
              f"{forces_analytic[i,2].item():20.15f}]")

    max_diff = (forces_autograd - forces_analytic).abs().max().item()
    print(f"\nMax force difference: {max_diff:.6e}")
    print(f"Force sum (autograd): [{forces_autograd.sum(0)[0].item():.6e}, "
          f"{forces_autograd.sum(0)[1].item():.6e}, "
          f"{forces_autograd.sum(0)[2].item():.6e}]")
    print(f"Force sum (analytic): [{forces_analytic.sum(0)[0].item():.6e}, "
          f"{forces_analytic.sum(0)[1].item():.6e}, "
          f"{forces_analytic.sum(0)[2].item():.6e}]")
    print(f"{'='*80}\n")

    assert torch.allclose(forces_autograd, forces_analytic, atol=1e-10), \
        f"Analytic and autograd forces differ by max {max_diff:.6e}"

