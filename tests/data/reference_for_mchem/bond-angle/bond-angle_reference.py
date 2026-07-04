import pytest
import torch
import sys
import openmm.app as app

sys.path.insert(0, "/pscratch/sd/a/asa/software/pyCMM")
from cmm.bonded import computeBondAngleCoupling, computeBond, computeAngleFromVecs
from cmm.ffxml import ForceFieldXML
from cmm.topology import Topology
from cmm.units import BOHR2NM
from cmm.units import BOHR2ANG


# Parameters directly from XML
REQ     = 1.81211318
THETAEQ = 1.822532146
K_BA_1  = -0.0322254957761867  # O-H1 bond coupled to H1-O-H2 angle
K_BA_2  = -0.0322254957761867  # O-H2 bond coupled to H1-O-H2 angle


@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float32),
    ('cuda', torch.float64),
])
def test_bond_angle_coupling_reference(device, dtype):
    """Generate reference energy and forces for bond-angle coupling - single water.
    
    Water has TWO bond-angle coupling terms per molecule:
      term 1: k_ba_1 * (r_OH1 - req) * (cos(theta) - cos(thetaeq))
      term 2: k_ba_2 * (r_OH2 - req) * (cos(theta) - cos(thetaeq))
    """

    ANG2BOHR = 1.0/ BOHR2ANG
    print(f"ANG2BOHR: {ANG2BOHR:.17f}")
    coords = torch.tensor([
        [-2.722762, 15.350993, -0.920138],
        [-2.603766, 14.476160, -1.273219],
        [-3.543810, 15.582853, -1.332255],
    ], device=device, dtype=dtype) * ANG2BOHR
    coords = coords.requires_grad_(True)

    bonds1 = torch.tensor([[0, 1]], device=device, dtype=torch.int32)  # O-H1
    bonds2 = torch.tensor([[0, 2]], device=device, dtype=torch.int32)  # O-H2

    req     = torch.tensor(REQ,     device=device, dtype=dtype)
    thetaeq = torch.tensor(THETAEQ, device=device, dtype=dtype)
    k_ba_1  = torch.tensor(K_BA_1,  device=device, dtype=dtype)
    k_ba_2  = torch.tensor(K_BA_2,  device=device, dtype=dtype)

    # Bond distances
    r1 = computeBond(coords, bonds1, box=None, boxInv=None)
    r2 = computeBond(coords, bonds2, box=None, boxInv=None)

    # Angle H1-O-H2 (bond vecs from vertex O outward)
    bondVec1 = coords[1] - coords[0]  # O -> H1
    bondVec2 = coords[2] - coords[0]  # O -> H2
    theta = computeAngleFromVecs(bondVec1.unsqueeze(0), bondVec2.unsqueeze(0))

    # Two coupling terms: one per O-H bond
    e1 = computeBondAngleCoupling(r1, req, theta, thetaeq, k_ba_1)
    e2 = computeBondAngleCoupling(r2, req, theta, thetaeq, k_ba_2)
    total_energy = (e1 + e2).sum()

    print(f"\n{'='*60}")
    print(f"Bond-Angle Coupling Test Results ({dtype})")
    print(f"{'='*60}")
    print(f"r1 (O-H1):     {r1.item():.15f}")
    print(f"r2 (O-H2):     {r2.item():.15f}")
    print(f"req:           {req.item():.15f}")
    print(f"theta:         {theta.item():.15f} rad  ({torch.rad2deg(theta).item():.10f} deg)")
    print(f"thetaeq:       {thetaeq.item():.15f} rad  ({torch.rad2deg(thetaeq).item():.10f} deg)")
    print(f"cos(theta):    {torch.cos(theta).item():.15f}")
    print(f"cos(thetaeq):  {torch.cos(thetaeq).item():.15f}")
    print(f"e1 (r1 term):  {e1.item():.17f}")
    print(f"e2 (r2 term):  {e2.item():.17f}")
    print(f"Total energy:  {total_energy.item():.17f}")

    total_energy.backward()
    forces = -coords.grad
    virial = -torch.einsum('ij,ik->jk', coords, forces)

    print(f"\nForces on atoms:")
    print(f"O  force: [{forces[0,0].item():20.15f}, {forces[0,1].item():20.15f}, {forces[0,2].item():20.15f}]")
    print(f"H1 force: [{forces[1,0].item():20.15f}, {forces[1,1].item():20.15f}, {forces[1,2].item():20.15f}]")
    print(f"H2 force: [{forces[2,0].item():20.15f}, {forces[2,1].item():20.15f}, {forces[2,2].item():20.15f}]")
    print(f"Force sum: [{forces.sum(0)[0].item():20.15f}, {forces.sum(0)[1].item():20.15f}, {forces.sum(0)[2].item():20.15f}]")
    print(f"{'='*60}\n")
    print(f"\nVirial Tensor:")
    for i in range(3):
        print(f"[{virial[i,0].item():20.15f}, {virial[i,1].item():20.15f}, {virial[i,2].item():20.15f}]")
    print(f"Virial trace: {virial.trace().item():.17f}")
    print(f"{'='*60}\n")

    assert torch.allclose(forces.sum(0), torch.zeros(3, device=device, dtype=dtype), atol=1e-6)

    return {
        'r1': r1.item(), 'r2': r2.item(), 'theta': theta.item(),
        'energy': total_energy.item(),
        'forces': forces.detach().cpu().numpy(),
    }


@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float32),
    ('cuda', torch.float64),
])
def test_bond_angle_coupling_at_equilibrium(device, dtype):
    """Verify zero energy and forces when bond and angle are both at equilibrium"""

    half_angle = THETAEQ / 2.0
    coords = torch.tensor([
        [0.0, 0.0, 0.0],
        [ REQ * torch.cos(torch.tensor(half_angle)).item(),
          REQ * torch.sin(torch.tensor(half_angle)).item(), 0.0],
        [ REQ * torch.cos(torch.tensor(half_angle)).item(),
         -REQ * torch.sin(torch.tensor(half_angle)).item(), 0.0],
    ], device=device, dtype=dtype, requires_grad=True)

    req     = torch.tensor(REQ,     device=device, dtype=dtype)
    thetaeq = torch.tensor(THETAEQ, device=device, dtype=dtype)
    k_ba    = torch.tensor(K_BA_1,  device=device, dtype=dtype)

    bonds1 = torch.tensor([[0, 1]], device=device, dtype=torch.int32)
    bonds2 = torch.tensor([[0, 2]], device=device, dtype=torch.int32)

    r1 = computeBond(coords, bonds1, box=None, boxInv=None)
    r2 = computeBond(coords, bonds2, box=None, boxInv=None)

    bondVec1 = (coords[1] - coords[0]).unsqueeze(0)
    bondVec2 = (coords[2] - coords[0]).unsqueeze(0)
    theta = computeAngleFromVecs(bondVec1, bondVec2)

    e1 = computeBondAngleCoupling(r1, req, theta, thetaeq, k_ba)
    e2 = computeBondAngleCoupling(r2, req, theta, thetaeq, k_ba)
    energy = (e1 + e2).sum()

    print(f"\nAt equilibrium:")
    print(f"r1={r1.item():.10f}  r2={r2.item():.10f}  req={req.item():.10f}")
    print(f"theta={theta.item():.10f}  thetaeq={thetaeq.item():.10f}")
    print(f"Energy: {energy.item():.10f} (should be ~0)")

    assert torch.allclose(energy, torch.zeros_like(energy), atol=1e-6)

    energy.backward()
    forces = -coords.grad
    assert torch.allclose(forces, torch.zeros_like(forces), atol=1e-5)


@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float64),
])
def test_bond_angle_coupling_minv_clamp(device, dtype):
    """Verify minv floor: stretch bond (dr > 0) with angle > thetaeq (dcos < 0) → negative → clamp"""

    req_val     = REQ
    thetaeq_val = THETAEQ

    # Stretch both bonds and open the angle beyond thetaeq
    # k_ba < 0, dr > 0, dcos < 0  →  k*dr*dcos > 0  (no clamp in this config)
    # To force clamp: use large positive k with dr > 0 and dcos < 0
    half_angle = (thetaeq_val + 0.3) / 2.0  # open angle beyond eq
    coords = torch.tensor([
        [0.0, 0.0, 0.0],
        [ req_val * 1.1 * torch.cos(torch.tensor(half_angle)).item(),
          req_val * 1.1 * torch.sin(torch.tensor(half_angle)).item(), 0.0],
        [ req_val * 1.1 * torch.cos(torch.tensor(half_angle)).item(),
         -req_val * 1.1 * torch.sin(torch.tensor(half_angle)).item(), 0.0],
    ], device=device, dtype=dtype, requires_grad=True)

    req     = torch.tensor(req_val,     device=device, dtype=dtype)
    thetaeq = torch.tensor(thetaeq_val, device=device, dtype=dtype)
    k_ba    = torch.tensor(1.0,         device=device, dtype=dtype)  # large positive k
    minv    = -0.002

    bonds1 = torch.tensor([[0, 1]], device=device, dtype=torch.int32)
    r1 = computeBond(coords, bonds1, box=None, boxInv=None)

    bondVec1 = (coords[1] - coords[0]).unsqueeze(0)
    bondVec2 = (coords[2] - coords[0]).unsqueeze(0)
    theta = computeAngleFromVecs(bondVec1, bondVec2)

    raw_energy = k_ba * (r1 - req) * (torch.cos(theta) - torch.cos(thetaeq))
    energy = computeBondAngleCoupling(r1, req, theta, thetaeq, k_ba)

    print(f"\nClamp test:")
    print(f"r1={r1.item():.8f}  theta={torch.rad2deg(theta).item():.6f} deg")
    print(f"dr={( r1 - req).item():.8f}  dcos={(torch.cos(theta) - torch.cos(thetaeq)).item():.8f}")
    print(f"Raw energy:     {raw_energy.item():.10f}")
    print(f"Clamped energy: {energy.item():.10f}  (minv={minv})")

    if raw_energy.item() < minv:
        assert torch.allclose(energy, torch.tensor(minv, device=device, dtype=dtype), atol=1e-10)
        print("Clamp was applied correctly.")
    else:
        assert torch.allclose(energy, raw_energy, atol=1e-10)
        print("No clamp needed — adjust geometry if you want to test the clamp branch.")


@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float64),
])
def test_bond_angle_coupling_water_216(device, dtype):
    """Generate reference energy, forces, and virial for bond-angle coupling - 216 waters"""

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
    ba_energy = energies["bond_angle"]  # adjust key to match your API

    print(f"\n{'='*80}")
    print(f"Bond-Angle Coupling Reference Data - 216 Waters ({dtype})")
    print(f"{'='*80}")
    print(f"Number of atoms:  {coords.shape[0]}")
    print(f"Number of waters: {coords.shape[0] // 3}")
    print(f"Box dimensions:   {box.diagonal()}")
    print(f"\nTotal Bond-Angle Coupling Energy: {ba_energy.item():.17f}")

    ba_energy.backward()
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

    with open('ene_ba_coupling_test.dat', 'w') as f:
        f.write('# reference bond-angle coupling energy\n')
        f.write(f'{ba_energy.item():.12f}\n')

    with open('ref_grad_ba_coupling.dat', 'w') as f:
        f.write('# reference bond-angle coupling forces\n')
        forces_np = forces.detach().cpu().numpy()
        for i in range(forces_np.shape[0]):
            f.write(f'{i} {forces_np[i,0]:.13f} {forces_np[i,1]:.13f} {forces_np[i,2]:.13f}\n')

    with open('virial_ba_coupling_ref.dat', 'w') as f:
        f.write('# xx xy xz yx yy yz zx zy zz\n')
        v = virial.detach().cpu().numpy()
        f.write(f'{v[0,0]:.14f} {v[0,1]:.14f} {v[0,2]:.14f} '
                f'{v[1,0]:.14f} {v[1,1]:.14f} {v[1,2]:.14f} '
                f'{v[2,0]:.14f} {v[2,1]:.14f} {v[2,2]:.14f}\n')

    print("Written: ene_ba_coupling_test.dat, ref_grad_ba_coupling.dat, virial_ba_coupling_ref.dat")

    return {
        'energy': ba_energy.item(),
        'forces': forces.detach().cpu().numpy(),
        'virial': virial.detach().cpu().numpy(),
    }


@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float64),
])
def test_bond_angle_coupling_water_216_manual(device, dtype):
    """Manual bond-angle coupling calculation for 216 waters without ForceFieldXML"""

    pdb_path = "/pscratch/sd/a/asa/software/pyCMM/tests/data/water_216.pdb"

    pdb = app.PDBFile(pdb_path)
    coords = torch.tensor(
        pdb.getPositions(asNumpy=True)._value / BOHR2NM,
        device=device, dtype=dtype, requires_grad=True
    )

    n_waters = coords.shape[0] // 3

    bonds1_idx = torch.tensor([[i*3, i*3+1] for i in range(n_waters)], device=device, dtype=torch.int32)  # O-H1
    bonds2_idx = torch.tensor([[i*3, i*3+2] for i in range(n_waters)], device=device, dtype=torch.int32)  # O-H2
    # Angle indices: [H1, O, H2] — O is vertex
    angle_idx  = torch.tensor([[i*3+1, i*3, i*3+2] for i in range(n_waters)], device=device, dtype=torch.int32)

    req     = torch.tensor(REQ,     device=device, dtype=dtype)
    thetaeq = torch.tensor(THETAEQ, device=device, dtype=dtype)
    k_ba_1  = torch.tensor(K_BA_1,  device=device, dtype=dtype)
    k_ba_2  = torch.tensor(K_BA_2,  device=device, dtype=dtype)

    r1 = computeBond(coords, bonds1_idx, box=None, boxInv=None)
    r2 = computeBond(coords, bonds2_idx, box=None, boxInv=None)

    bondVec1 = coords[angle_idx[:, 0]] - coords[angle_idx[:, 1]]  # H1 - O
    bondVec2 = coords[angle_idx[:, 2]] - coords[angle_idx[:, 1]]  # H2 - O
    thetas = computeAngleFromVecs(bondVec1, bondVec2)

    # Two terms per water: one for each O-H bond coupled to the H-O-H angle
    e1 = computeBondAngleCoupling(r1, req, thetas, thetaeq, k_ba_1)
    e2 = computeBondAngleCoupling(r2, req, thetas, thetaeq, k_ba_2)
    total_energy = (e1 + e2).sum()

    print(f"\n{'='*80}")
    print(f"Manual Bond-Angle Coupling - 216 Waters ({dtype})")
    print(f"{'='*80}")
    print(f"Number of waters: {n_waters}")
    print(f"Mean r1: {r1.mean().item():.6f}  Mean r2: {r2.mean().item():.6f}  req: {req.item():.6f}")
    print(f"Mean theta: {torch.rad2deg(thetas).mean().item():.6f} deg  thetaeq: {torch.rad2deg(thetaeq).item():.6f} deg")
    print(f"Total energy: {total_energy.item():.17f}")

    total_energy.backward()
    forces = -coords.grad
    virial = -torch.einsum('ij,ik->jk', coords, forces)

    print(f"\nForces (first 3 atoms - first water):")
    for i in range(3):
        print(f"Atom {i}: [{forces[i,0].item():20.15f}, "
              f"{forces[i,1].item():20.15f}, "
              f"{forces[i,2].item():20.15f}]")

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
def test_bond_angle_coupling_analytic_forces(device, dtype):
    """Compare autograd vs analytic forces for bond-angle coupling."""

    pdb_path = "/pscratch/sd/a/asa/software/pyCMM/tests/data/water_216.pdb"

    pdb    = app.PDBFile(pdb_path)
    coords = torch.tensor(
        pdb.getPositions(asNumpy=True)._value / BOHR2NM,
        device=device, dtype=dtype, requires_grad=True
    )

    n_waters   = coords.shape[0] // 3
    bonds1_idx = torch.tensor([[i*3, i*3+1] for i in range(n_waters)], device=device, dtype=torch.int32)
    bonds2_idx = torch.tensor([[i*3, i*3+2] for i in range(n_waters)], device=device, dtype=torch.int32)
    angle_idx  = torch.tensor([[i*3+1, i*3, i*3+2] for i in range(n_waters)], device=device, dtype=torch.int32)

    req     = torch.tensor(REQ,     device=device, dtype=dtype)
    thetaeq = torch.tensor(THETAEQ, device=device, dtype=dtype)
    k_ba_1  = torch.tensor(K_BA_1,  device=device, dtype=dtype)
    k_ba_2  = torch.tensor(K_BA_2,  device=device, dtype=dtype)

    # --- Autograd forces ---
    r1       = computeBond(coords, bonds1_idx, box=None, boxInv=None)
    r2       = computeBond(coords, bonds2_idx, box=None, boxInv=None)
    bondVec1 = coords[angle_idx[:, 0]] - coords[angle_idx[:, 1]]
    bondVec2 = coords[angle_idx[:, 2]] - coords[angle_idx[:, 1]]
    thetas   = computeAngleFromVecs(bondVec1, bondVec2)
    e1       = computeBondAngleCoupling(r1, req, thetas, thetaeq, k_ba_1)
    e2       = computeBondAngleCoupling(r2, req, thetas, thetaeq, k_ba_2)
    total_energy = (e1 + e2).sum()
    total_energy.backward()
    forces_autograd = -coords.grad.clone()

    # --- Analytic forces ---
    # E = k*(r-req)*(cos(theta)-cos(thetaeq))   [with minv clamp]
    # Bond part:  dE/dr   = k*(cos(theta)-cos(thetaeq))
    # Angle part: dE/dcos = k*(r-req)
    # F = -dE/dr (total, summed over both terms)

    coords_d  = coords.detach()
    minv      = -0.002

    # Bond vectors and unit vecs
    vec1      = coords_d[bonds1_idx[:, 1]] - coords_d[bonds1_idx[:, 0]]  # H1-O
    vec2      = coords_d[bonds2_idx[:, 1]] - coords_d[bonds2_idx[:, 0]]  # H2-O
    r1_d      = torch.norm(vec1, dim=-1)
    r2_d      = torch.norm(vec2, dim=-1)
    uv1       = vec1 / r1_d.unsqueeze(1)
    uv2       = vec2 / r2_d.unsqueeze(1)

    # Angle geometry
    rBA       = coords_d[angle_idx[:, 0]] - coords_d[angle_idx[:, 1]]  # H1-O
    rBC       = coords_d[angle_idx[:, 2]] - coords_d[angle_idx[:, 1]]  # H2-O
    rBA_norm  = torch.norm(rBA, dim=-1, keepdim=True)
    rBC_norm  = torch.norm(rBC, dim=-1, keepdim=True)
    uBA       = rBA / rBA_norm
    uBC       = rBC / rBC_norm
    cos_theta = torch.sum(uBA * uBC, dim=-1)
    cos_eq    = torch.cos(thetaeq)

    dr1   = r1_d - req
    dr2   = r2_d - req
    dcos  = cos_theta - cos_eq

    # Clamp masks
    clamp1 = (k_ba_1 * dr1 * dcos) < minv
    clamp2 = (k_ba_2 * dr2 * dcos) < minv

    # Bond gradient: dE/dr (zero where clamped)
    dE1_dr1 = torch.where(clamp1, torch.zeros_like(dr1), k_ba_1 * dcos).unsqueeze(1)
    dE2_dr2 = torch.where(clamp2, torch.zeros_like(dr2), k_ba_2 * dcos).unsqueeze(1)

    # Angle gradient: dE/d(cos_theta) summed over both terms (zero where clamped)
    dE1_dcos = torch.where(clamp1, torch.zeros_like(dr1), k_ba_1 * dr1)
    dE2_dcos = torch.where(clamp2, torch.zeros_like(dr2), k_ba_2 * dr2)
    dE_dcos  = (dE1_dcos + dE2_dcos).unsqueeze(1)

    # Wilson gradient of cos(theta) — no sin
    cos_theta_3d = cos_theta.unsqueeze(1)
    dcos_drH1    = (uBC - cos_theta_3d * uBA) / rBA_norm  # (N,3)
    dcos_drH2    = (uBA - cos_theta_3d * uBC) / rBC_norm
    dcos_drO     = -(dcos_drH1 + dcos_drH2)

    # F = -dE/dq accumulated per atom
    # Convention: F_autograd = -grad(E), so we match that sign
    forces_analytic = torch.zeros_like(coords_d)

    # Bond1 contribution: r1 = |H1-O|, dr1/dO = -uv1, dr1/dH1 = +uv1
    # F_O  from bond1 = -dE1/dr1 * (-uv1) = +dE1/dr1 * uv1
    # F_H1 from bond1 = -dE1/dr1 * (+uv1) = -dE1/dr1 * uv1
    forces_analytic.index_add_(0, bonds1_idx[:, 0].long(),  dE1_dr1 * uv1)  # O
    forces_analytic.index_add_(0, bonds1_idx[:, 1].long(), -dE1_dr1 * uv1)  # H1

    # Bond2 contribution
    forces_analytic.index_add_(0, bonds2_idx[:, 0].long(),  dE2_dr2 * uv2)  # O
    forces_analytic.index_add_(0, bonds2_idx[:, 1].long(), -dE2_dr2 * uv2)  # H2

    # Angle contribution: F_i = -dE/d(cos)*d(cos)/dr_i
    forces_analytic.index_add_(0, angle_idx[:, 0].long(), -dE_dcos * dcos_drH1)  # H1
    forces_analytic.index_add_(0, angle_idx[:, 2].long(), -dE_dcos * dcos_drH2)  # H2
    forces_analytic.index_add_(0, angle_idx[:, 1].long(), -dE_dcos * dcos_drO)   # O

    # --- Compare ---
    max_diff = (forces_autograd - forces_analytic).abs().max().item()

    print(f"\n{'='*80}")
    print(f"Bond-Angle Coupling Analytic Force Test - 216 Waters ({dtype})")
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
