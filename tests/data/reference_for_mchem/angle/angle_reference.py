import pytest
import torch
import sys
import openmm.app as app

sys.path.insert(0, "/pscratch/sd/a/asa/software/pyCMM")
from cmm.bonded import computeAngleFromVecs, computeCosAnglePotential
from cmm.ffxml import ForceFieldXML
from cmm.topology import Topology
from cmm.units import BOHR2NM
from cmm.units import BOHR2ANG


@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float32),
    ('cuda', torch.float64),
])
def test_cos_angle_reference(device, dtype):
    """Generate reference energy and forces for cosine angle potential - single water"""
    ANG2BOHR = 1.0 / BOHR2ANG

    coords = torch.tensor([
        [-2.722762, 15.350993, -0.920138],
        [-2.603766, 14.476160, -1.273219],
        [-3.543810, 15.582853, -1.332255],
    ], device=device, dtype=dtype) * ANG2BOHR
    coords = coords.requires_grad_(True)

    # [atom1, vertex, atom2] — matches computeAngle convention
    angle_indices = torch.tensor([[1, 0, 2]], device=device, dtype=torch.int32)

    thetaeq = torch.tensor(1.822532146, device=device, dtype=dtype)  # ~104.5 deg in radians
    k_theta = torch.tensor(0.1722274, device=device, dtype=dtype)

    # Bond vectors from vertex (O) outward to each H
    bondVec1 = coords[angle_indices[:, 0]] - coords[angle_indices[:, 1]]  # H1 - O
    bondVec2 = coords[angle_indices[:, 2]] - coords[angle_indices[:, 1]]  # H2 - O
    theta = computeAngleFromVecs(bondVec1, bondVec2)

    # k/2 * (cos(theta) - cos(thetaeq))^2
    energy = computeCosAnglePotential(theta, thetaeq, k_theta)
    total_energy = energy.sum()

    print(f"\n{'='*60}")
    print(f"Cosine Angle Test Results ({dtype})")
    print(f"{'='*60}")
    print(f"H1-O-H2 angle:  {theta.item():.15f} rad  ({torch.rad2deg(theta).item():.10f} deg)")
    print(f"Equilibrium:    {thetaeq.item():.15f} rad  ({torch.rad2deg(thetaeq).item():.10f} deg)")
    print(f"cos(theta):     {torch.cos(theta).item():.15f}")
    print(f"cos(thetaeq):   {torch.cos(thetaeq).item():.15f}")
    print(f"Total energy:   {total_energy.item():.17f}")

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
    print(f"{'='*60}\n")

    return {
        'angle': theta.item(),
        'energy': total_energy.item(),
        'forces': forces.detach().cpu().numpy(),
    }


@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float32),
    ('cuda', torch.float64),
])
def test_cos_angle_at_equilibrium(device, dtype):
    """Verify zero energy and forces when angle is exactly at equilibrium"""

    thetaeq_val = 1.8238
    half_angle = thetaeq_val / 2.0
    r_OH = 1.81211318

    coords = torch.tensor([
        [0.0, 0.0, 0.0],
        [ r_OH * torch.cos(torch.tensor(half_angle)).item(),
          r_OH * torch.sin(torch.tensor(half_angle)).item(), 0.0],
        [ r_OH * torch.cos(torch.tensor(half_angle)).item(),
         -r_OH * torch.sin(torch.tensor(half_angle)).item(), 0.0],
    ], device=device, dtype=dtype, requires_grad=True)

    thetaeq = torch.tensor(thetaeq_val, device=device, dtype=dtype)
    k_theta = torch.tensor(0.6282, device=device, dtype=dtype)

    bondVec1 = (coords[1] - coords[0]).unsqueeze(0)
    bondVec2 = (coords[2] - coords[0]).unsqueeze(0)
    theta = computeAngleFromVecs(bondVec1, bondVec2)

    energy = computeCosAnglePotential(theta, thetaeq, k_theta).sum()

    print(f"\nAt equilibrium:")
    print(f"Angle:  {theta.item():.10f} rad  (thetaeq = {thetaeq.item():.10f} rad)")
    print(f"Energy: {energy.item():.10f} (should be ~0)")

    assert torch.allclose(energy, torch.zeros_like(energy), atol=1e-6)

    energy.backward()
    forces = -coords.grad
    assert torch.allclose(forces, torch.zeros_like(forces), atol=1e-5)


@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float64),
])
def test_cos_angle_water_216(device, dtype):
    """Generate reference energy, forces, and virial for cosine angle potential - 216 waters"""

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
    angle_energy = energies["angle"]  # adjust key to match your API

    print(f"\n{'='*80}")
    print(f"Cosine Angle Reference Data - 216 Waters ({dtype})")
    print(f"{'='*80}")
    print(f"Number of atoms:  {coords.shape[0]}")
    print(f"Number of waters: {coords.shape[0] // 3}")
    print(f"Box dimensions:   {box.diagonal()}")
    print(f"\nTotal Cosine Angle Energy: {angle_energy.item():.17f}")

    angle_energy.backward()
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

    with open('ene_cos_angle_test.dat', 'w') as f:
        f.write('# reference cosine angle energy\n')
        f.write(f'{angle_energy.item():.12f}\n')

    with open('ref_grad_cos_angle.dat', 'w') as f:
        f.write('# reference cosine angle forces\n')
        forces_np = forces.detach().cpu().numpy()
        for i in range(forces_np.shape[0]):
            f.write(f'{i} {forces_np[i,0]:.13f} {forces_np[i,1]:.13f} {forces_np[i,2]:.13f}\n')

    with open('virial_cos_angle_ref.dat', 'w') as f:
        f.write('# xx xy xz yx yy yz zx zy zz\n')
        v = virial.detach().cpu().numpy()
        f.write(f'{v[0,0]:.14f} {v[0,1]:.14f} {v[0,2]:.14f} '
                f'{v[1,0]:.14f} {v[1,1]:.14f} {v[1,2]:.14f} '
                f'{v[2,0]:.14f} {v[2,1]:.14f} {v[2,2]:.14f}\n')

    print("Written: ene_cos_angle_test.dat, ref_grad_cos_angle.dat, virial_cos_angle_ref.dat")

    return {
        'energy': angle_energy.item(),
        'forces': forces.detach().cpu().numpy(),
        'virial': virial.detach().cpu().numpy(),
    }


@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float64),
])
def test_cos_angle_water_216_manual(device, dtype):
    """Manual cosine angle calculation for 216 waters without ForceFieldXML"""

    pdb_path = "/pscratch/sd/a/asa/software/pyCMM/tests/data/water_216.pdb"

    pdb = app.PDBFile(pdb_path)
    coords = torch.tensor(
        pdb.getPositions(asNumpy=True)._value / BOHR2NM,
        device=device, dtype=dtype, requires_grad=True
    )

    n_waters = coords.shape[0] // 3

    # H1-O-H2 for each water, central atom O at index 1
    angle_indices = torch.tensor(
        [[i*3 + 1, i*3, i*3 + 2] for i in range(n_waters)],  # [H1, O, H2]
        device=device, dtype=torch.int32
    )

    thetaeq = torch.tensor(1.8238, device=device, dtype=dtype)
    k_theta  = torch.tensor(0.6282, device=device, dtype=dtype)

    bondVec1 = coords[angle_indices[:, 0]] - coords[angle_indices[:, 1]]  # H1 - O
    bondVec2 = coords[angle_indices[:, 2]] - coords[angle_indices[:, 1]]  # H2 - O

    thetas = computeAngleFromVecs(bondVec1, bondVec2)
    energies = computeCosAnglePotential(thetas, thetaeq, k_theta)
    total_energy = energies.sum()

    print(f"\n{'='*80}")
    print(f"Manual Cosine Angle Calculation - 216 Waters ({dtype})")
    print(f"{'='*80}")
    print(f"Number of angles: {angle_indices.shape[0]}")
    print(f"Mean angle: {torch.rad2deg(thetas).mean().item():.6f} deg")
    print(f"Min/Max:    {torch.rad2deg(thetas).min().item():.6f} / {torch.rad2deg(thetas).max().item():.6f} deg")
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
def test_cos_angle_analytic_forces(device, dtype):
    """Compare autograd vs analytic forces for cosine angle potential - no sin."""

    pdb_path = "/pscratch/sd/a/asa/software/pyCMM/tests/data/water_216.pdb"

    pdb    = app.PDBFile(pdb_path)
    coords = torch.tensor(
        pdb.getPositions(asNumpy=True)._value / BOHR2NM,
        device=device, dtype=dtype, requires_grad=True
    )

    n_waters = coords.shape[0] // 3

    # [H1, O, H2] — O is vertex (index 1)
    angle_indices = torch.tensor(
        [[i*3+1, i*3, i*3+2] for i in range(n_waters)],
        device=device, dtype=torch.int32
    )

    thetaeq = torch.tensor(1.8238, device=device, dtype=dtype)
    k_theta = torch.tensor(0.6282, device=device, dtype=dtype)

    # --- Autograd forces ---
    bondVec1 = coords[angle_indices[:, 0]] - coords[angle_indices[:, 1]]  # H1 - O
    bondVec2 = coords[angle_indices[:, 2]] - coords[angle_indices[:, 1]]  # H2 - O
    thetas       = computeAngleFromVecs(bondVec1, bondVec2)
    energies     = computeCosAnglePotential(thetas, thetaeq, k_theta)
    total_energy = energies.sum()
    total_energy.backward()
    forces_autograd = -coords.grad.clone()

    # --- Analytic forces ---
    # V = K * (cos(theta) - cos(theta_0))^2
    # dV/dr = dV/d(cos_theta) * d(cos_theta)/dr
    # dV/d(cos_theta) = 2*K*(cos(theta) - cos(theta_0))
    #
    # cos(theta) = (rBA . rBC) / (|rBA| * |rBC|)
    #
    # d(cos_theta)/dr_A = (uBC - cos(theta)*uBA) / |rBA|
    # d(cos_theta)/dr_C = (uBA - cos(theta)*uBC) / |rBC|
    # d(cos_theta)/dr_B = -(d(cos_theta)/dr_A + d(cos_theta)/dr_C)
    #
    # F_i = -dV/dr_i

    coords_d = coords.detach()

    rBA = coords_d[angle_indices[:, 0]] - coords_d[angle_indices[:, 1]]  # O->H1
    rBC = coords_d[angle_indices[:, 2]] - coords_d[angle_indices[:, 1]]  # O->H2

    rBA_norm = torch.norm(rBA, dim=-1, keepdim=True)  # (N,1)
    rBC_norm = torch.norm(rBC, dim=-1, keepdim=True)

    uBA = rBA / rBA_norm  # unit vec O->H1
    uBC = rBC / rBC_norm  # unit vec O->H2

    cos_theta    = torch.sum(uBA * uBC, dim=-1, keepdim=True)  # (N,1)
    cos_eq       = torch.cos(thetaeq)

    # dV/d(cos_theta) = 2*K*(cos(theta) - cos(theta_0))
    dV_dcos = k_theta * (cos_theta - cos_eq)  # (N,1)

    # d(cos_theta)/dr — no sin anywhere
    dcos_drA = (uBC - cos_theta * uBA) / rBA_norm   # (N,3)
    dcos_drC = (uBA - cos_theta * uBC) / rBC_norm
    dcos_drB = -(dcos_drA + dcos_drC)

    # F = -dV/dr
    F_H1 = -dV_dcos * dcos_drA   # (N,3)
    F_H2 = -dV_dcos * dcos_drC
    F_O  = -dV_dcos * dcos_drB

    forces_analytic = torch.zeros_like(coords_d)
    forces_analytic.index_add_(0, angle_indices[:, 0].long(), F_H1)
    forces_analytic.index_add_(0, angle_indices[:, 1].long(), F_O)
    forces_analytic.index_add_(0, angle_indices[:, 2].long(), F_H2)

    # --- Compare ---
    max_diff = (forces_autograd - forces_analytic).abs().max().item()

    print(f"\n{'='*80}")
    print(f"Cosine Angle Analytic Force Test - 216 Waters ({dtype})")
    print(f"{'='*80}")
    labels = ['O', 'H1', 'H2']
    print(f"\nAutograd forces (first water):")
    for i, lbl in enumerate(labels):
        print(f"  {lbl}: [{forces_autograd[i,0].item():20.15f}, "
              f"{forces_autograd[i,1].item():20.15f}, "
              f"{forces_autograd[i,2].item():20.15f}]")
    print(f"\nAnalytic forces (first water):")
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
