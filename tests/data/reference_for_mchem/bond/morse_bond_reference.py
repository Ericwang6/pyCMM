import pytest
import torch
import torch.nn as nn
import sys
import openmm.app as app
# Assuming these are imported from your module
sys.path.insert(0,"/pscratch/sd/a/asa/software/pyCMM")
from cmm.bonded import computeMorseBondPotential, computeBond 
from cmm.ffxml import ForceFieldXML
from cmm.topology import Topology
from cmm.units import BOHR2NM
from cmm.units import BOHR2ANG

@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float32),
    ('cuda', torch.float64),
])
def test_morse_bond_reference(device, dtype):
    """Test to generate reference energy and forces for Morse bond"""
    
    ANG2BOHR = 1.0/ BOHR2ANG
    print(f"ANG2BOHR: {ANG2BOHR:.17f}")
    coords = torch.tensor([
        [-2.722762, 15.350993, -0.920138],
        [-2.603766, 14.476160, -1.273219],
        [-3.543810, 15.582853, -1.332255],
    ], device=device, dtype=dtype) * ANG2BOHR
    coords = coords.requires_grad_(True)
    
    # Bonds: O-H1 and O-H2
    bonds = torch.tensor([[0, 1], [0, 2]], device=device, dtype=torch.int32)
    
    # Morse potential parameters
    D = torch.tensor(0.19968199, device=device, dtype=dtype)
    req = torch.tensor(1.81211318, device=device, dtype=dtype)
    kb = torch.tensor(0.54375456, device=device, dtype=dtype)
    
    # Calculate beta (a in the function) from kb and D
    # From your morse_bond.cuh: beta = sqrt(kb / 2 / d)
    beta = torch.sqrt(kb / (2 * D))
    
    # Compute bond distances
    r1 = computeBond(coords, bonds[0:1], box=None, boxInv=None)
    r2 = computeBond(coords, bonds[1:2], box=None, boxInv=None)
    
    # Compute energies for each bond
    e1 = computeMorseBondPotential(r1, req, D, beta)
    e2 = computeMorseBondPotential(r2, req, D, beta)
    # virial for bond 1 only
    e1.backward(retain_graph=True)
    forces_1 = -coords.grad.clone()
    virial = -torch.einsum('ij,ik->jk', coords, forces_1)
    coords.grad.zero_()
    
    # Total energy
    total_energy = e1 + e2
    
    print(f"\n{'='*60}")
    print(f"Morse Bond Test Results ({dtype})")
    print(f"{'='*60}")
    print(f"Bond 1 (O-H1) distance: {r1.item():.15f} Bohr")
    print(f"Bond 2 (O-H2) distance: {r2.item():.15f} Bohr")
    print(f"Precision: {dtype} | Bond 1 energy: {e1.item():.17f}")
    print(f"Bond 2 energy: {e2.item():.17f}")
    print(f"Total energy: {total_energy.item():.17f}")
    
    # Compute forces via autograd
    total_energy.backward()
    forces = -coords.grad
    
    print(f"\nForces on atoms:")
    print(f"O force:  [{forces[0, 0].item():20.15f}, {forces[0, 1].item():20.15f}, {forces[0, 2].item():20.15f}]")
    print(f"Precision: {dtype}  | H1 force: [{forces[1, 0].item():20.15f}, {forces[1, 1].item():20.15f}, {forces[1, 2].item():20.15f}]")
    print(f"H2 force: [{forces[2, 0].item():20.15f}, {forces[2, 1].item():20.15f}, {forces[2, 2].item():20.15f}]")
    print(f"Force sum: [{forces.sum(0)[0].item():20.15f}, {forces.sum(0)[1].item():20.15f}, {forces.sum(0)[2].item():20.15f}]")
    print(f"{'='*60}\n")
    
    # Verify force conservation (sum should be ~0)
    assert torch.allclose(forces.sum(0), torch.zeros(3, device=device, dtype=dtype), atol=1e-6)
    print(f"\nVirial Tensor for bond 1:")
    for i in range(3):
        print(f"[{virial[i,0].item():20.15f}, {virial[i,1].item():20.15f}, {virial[i,2].item():20.15f}]")
    print(f"Virial trace: {virial.trace().item():.17f}")
    print(f"{'='*60}\n")
    
    # Store reference values for comparison
    return {
        'energy': total_energy.item(),
        'forces': forces.detach().cpu().numpy(),
        'bond_distances': (r1.item(), r2.item())
    }


@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float32),
    ('cuda', torch.float64),
])
def test_morse_bond_components(device, dtype):
    """Test individual components of Morse potential"""
    
    # Simple test: single bond at equilibrium
    coords = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.81211318, 0.0, 0.0],  # Exactly at req
    ], device=device, dtype=dtype, requires_grad=True)
    
    bonds = torch.tensor([[0, 1]], device=device, dtype=torch.int32)
    
    D = torch.tensor(0.19968199, device=device, dtype=dtype)
    req = torch.tensor(1.81211318, device=device, dtype=dtype)
    kb = torch.tensor(0.54375456, device=device, dtype=dtype)
    beta = torch.sqrt(kb / (2 * D))
    
    r = computeBond(coords, bonds, box=None, boxInv=None)
    energy = computeMorseBondPotential(r, req, D, beta)
    
    print(f"\nAt equilibrium distance:")
    print(f"Distance: {r.item():.8f} (req = {req.item():.8f})")
    print(f"Energy: {energy.item():.10f} (should be ~0)")
    
    # At equilibrium, energy should be ~0
    assert torch.allclose(energy, torch.zeros_like(energy), atol=1e-6)
    
    # Forces should also be ~0
    energy.backward()
    forces = -coords.grad
    print(f"Forces: {forces}")
    assert torch.allclose(forces, torch.zeros_like(forces), atol=1e-5)
@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float64),
])
def test_morse_bond_water_216(device, dtype):
    """Generate reference energy, forces, and virial for 216 water molecules"""

    # Paths to your files
    pdb_path = "/pscratch/sd/a/asa/software/pyCMM/tests/data/water_216.pdb"  # Update this path
    ff_path = "/pscratch/sd/a/asa/software/pyCMM/tests/data/water-aalim.xml"   # Update this path

    # Load PDB and create topology
    pdb = app.PDBFile(pdb_path)
    top = Topology.fromOpenmm(pdb.topology, device)

    # Get coordinates and box
    coords = torch.tensor(
        pdb.getPositions(asNumpy=True)._value / BOHR2NM,
        device=device,
        dtype=dtype,
        requires_grad=True
    )

    box_vectors = pdb.topology.getPeriodicBoxVectors()
    box = torch.tensor(
        [[vec.x / BOHR2NM, vec.y / BOHR2NM, vec.z / BOHR2NM] for vec in box_vectors],
        device=device,
        dtype=dtype,
        requires_grad=True
    )

    # Parametrize system with only Morse bonds (disable other terms)
    ff = ForceFieldXML(ff_path, device=device)
    system = ff.parametrize(
        top,
        use_fd_morse=False,
        use_polarization=False,      # Disable for pure Morse test
        use_hardness_change=False,
        cutoff_sr=9.0,
        use_switch=True,
        use_customized_ops=False
    )

    # Get only bonded energy (assuming system separates bonded/nonbonded)
    # You may need to adjust this depending on your system's API
    energies = system.getEnergy(coords, box)

    # Extract just the Morse bond energy
    morse_energy = energies["bond"]

    print(f"\n{'='*80}")
    print(f"Morse Bond Reference Data - 216 Waters ({dtype})")
    print(f"{'='*80}")
    print(f"Number of atoms: {coords.shape[0]}")
    print(f"Number of waters: {coords.shape[0] // 3}")
    print(f"Box dimensions: {box.diagonal()}")
    print(f"\nTotal Morse Bond Energy: {morse_energy.item():.17f}")

    # Compute forces via autograd
    morse_energy.backward(retain_graph=True)
    forces = -coords.grad.clone()

    # Compute virial tensor
    # Virial = -sum_i (r_i ⊗ F_i) where ⊗ is outer product
    virial = -torch.einsum('ij,ik->jk', coords, forces)

    print(f"\nForces (first 3 atoms - first water):")
    for i in range(3):
        print(f"Atom {i}: [{forces[i, 0].item():20.15f}, "
              f"{forces[i, 1].item():20.15f}, "
              f"{forces[i, 2].item():20.15f}]")

    print(f"\nForce sum (should be ~0):")
    force_sum = forces.sum(0)
    print(f"[{force_sum[0].item():20.15f}, "
          f"{force_sum[1].item():20.15f}, "
          f"{force_sum[2].item():20.15f}]")

    print(f"\nVirial Tensor (3x3):")
    for i in range(3):
        print(f"[{virial[i, 0].item():20.15f}, "
              f"{virial[i, 1].item():20.15f}, "
              f"{virial[i, 2].item():20.15f}]")

    print(f"\nVirial trace (related to pressure): {virial.trace().item():.17f}")
    print(f"{'='*80}\n")

    # Verify force conservation
    assert torch.allclose(forces.sum(0), torch.zeros(3, device=device, dtype=dtype), atol=1e-6)

    # 1. Write energy file
    with open('ene_test.dat', 'w') as f:
        f.write('# reference energy\n')
        f.write(f'{morse_energy.item():.12f}\n')
    print(f"\nEnergy written to 'ene_test.dat'")

    # 2. Write forces file
    with open('ref_grad_bond.dat', 'w') as f:
        f.write('# reference bond stretch force\n')
        forces_np = forces.detach().cpu().numpy()
        for i in range(forces_np.shape[0]):
            f.write(f'{i} {forces_np[i, 0]:.13f} {forces_np[i, 1]:.13f} {forces_np[i, 2]:.13f}\n')
    print(f"Forces written to 'ref_grad_bond.dat' ({forces.shape[0]} atoms)")

    # 3. Write virial file
    with open('virial_ref.dat', 'w') as f:
        f.write('# xx xy xz yx yy yz zx zy zz\n')
        virial_np = virial.detach().cpu().numpy()
        # Write as single line: xx xy xz yx yy yz zx zy zz
        f.write(f'{virial_np[0,0]:.14f} {virial_np[0,1]:.14f} {virial_np[0,2]:.14f} ')
        f.write(f'{virial_np[1,0]:.14f} {virial_np[1,1]:.14f} {virial_np[1,2]:.14f} ')
        f.write(f'{virial_np[2,0]:.14f} {virial_np[2,1]:.14f} {virial_np[2,2]:.14f}\n')
    print(f"Virial written to 'virial_ref.dat'")

    # Return reference data
    return {
        'energy': morse_energy.item(),
        'forces': forces.detach().cpu().numpy(),
        'virial': virial.detach().cpu().numpy(),
        'coords': coords.detach().cpu().numpy(),
        'box': box.detach().cpu().numpy(),
    }


@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float64),
])
def test_morse_bond_manual_calculation(device, dtype):
    """Alternative: Manual Morse bond calculation for 216 waters"""

    pdb_path = "/path/to/water_216.pdb"

    # Load coordinates
    pdb = app.PDBFile(pdb_path)
    coords = torch.tensor(
        pdb.getPositions(asNumpy=True)._value / BOHR2NM,
        device=device,
        dtype=dtype,
        requires_grad=True
    )

    n_waters = coords.shape[0] // 3

    # Create bond list for all waters (O-H1 and O-H2 for each)
    bonds = []
    for i in range(n_waters):
        o_idx = i * 3
        h1_idx = i * 3 + 1
        h2_idx = i * 3 + 2
        bonds.append([o_idx, h1_idx])
        bonds.append([o_idx, h2_idx])

    bonds = torch.tensor(bonds, device=device, dtype=torch.int32)

    # Morse parameters
    D = torch.tensor(0.19968199, device=device, dtype=dtype)
    req = torch.tensor(1.81211318, device=device, dtype=dtype)
    kb = torch.tensor(0.54375456, device=device, dtype=dtype)
    beta = torch.sqrt(kb / (2 * D))

    # Compute all bond distances
    bond_vecs = coords[bonds[:, 1]] - coords[bonds[:, 0]]
    r = torch.norm(bond_vecs, dim=1)

    # Compute energies
    energies = computeMorseBondPotential(r, req, D, beta)
    total_energy = energies.sum()

    print(f"\n{'='*80}")
    print(f"Manual Morse Bond Calculation - 216 Waters ({dtype})")
    print(f"{'='*80}")
    print(f"Number of bonds: {bonds.shape[0]}")
    print(f"Total energy: {total_energy.item():.17f}")

    # Compute forces
    total_energy.backward()
    forces = -coords.grad

    # Compute virial
    virial = -torch.einsum('ij,ik->jk', coords, forces)

    print(f"\nVirial Tensor:")
    for i in range(3):
        print(f"[{virial[i, 0].item():20.15f}, "
              f"{virial[i, 1].item():20.15f}, "
              f"{virial[i, 2].item():20.15f}]")

    print(f"{'='*80}\n")

    return {
        'energy': total_energy.item(),
        'forces': forces.detach().cpu().numpy(),
        'virial': virial.detach().cpu().numpy(),
    }
