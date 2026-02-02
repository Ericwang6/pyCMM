import torch
import os
import sys
os.environ["TORCH_COMPILE_DISABLE"] = "1"
sys.path.insert(0, "/pscratch/sd/a/asa/software/pyCMM")
import numpy as np

import openmm.app as app
import cmm
from cmm.ffxml import ForceFieldXML
from cmm.topology import Topology
from cmm.units import BOHR2NM
torch.set_printoptions(precision=8)
torch.set_default_dtype(torch.float64)

from pprint import pprint as pp


def finite_difference_forces(system, coords, box, epsilon=1e-5):
    """
    Calculate forces using finite difference method.
    F = -dE/dx ≈ -(E(x+ε) - E(x-ε)) / (2ε)
    
    Args:
        system: The force field system
        coords: Coordinates tensor (N, 3)
        box: Box vectors tensor (3, 3)
        epsilon: Step size for finite differences
    
    Returns:
        forces: Force tensor (N, 3)
    """
    forces = torch.zeros_like(coords)
    
    # Loop over all atoms and all dimensions
    for i in range(coords.shape[0]):
        for j in range(3):  # x, y, z
            # Create perturbed coordinates
            coords_plus = coords.clone()
            coords_minus = coords.clone()
            
            coords_plus[i, j] += epsilon
            coords_minus[i, j] -= epsilon
            
            # Calculate energies at perturbed positions
            with torch.no_grad():
                energy_plus = system.getEnergy(coords_plus, box)['total']
                energy_minus = system.getEnergy(coords_minus, box)['total']
            
            # Central difference formula: F = -dE/dx
            forces[i, j] = -(energy_plus - energy_minus) / (2 * epsilon)
    
    return forces


def test_customized_ops():
    ff_path = "/pscratch/sd/a/asa/software/pyCMM/tests/data/water_refit_nofdmorse.xml"
    pdb_path = "/pscratch/sd/a/asa/software/pyCMM/tests/data/water_216.pdb"

    device = 'cuda'
    ff = ForceFieldXML(ff_path, device=device)
    pdb = app.PDBFile(pdb_path)
    top = Topology.fromOpenmm(pdb.topology, device)

    coords = torch.tensor(pdb.getPositions(asNumpy=True)._value / BOHR2NM, device=device, requires_grad=True)
    box = torch.tensor([[vec.x / BOHR2NM, vec.y / BOHR2NM, vec.z / BOHR2NM] for vec in pdb.topology.getPeriodicBoxVectors()], device=device, requires_grad=True)
    
    #########################################
    # 1) Forces using energies.backward() - PyTorch autograd
    system = ff.parametrize(top, use_fd_morse=True, use_polarization=False, 
                           polarization_tolerance=1e-5, use_hardness_change=False, 
                           cutoff_sr=9.0, use_customized_ops=True, use_pme=True, 
                           use_switch=True)
    
    energies = system.getEnergy(coords, box)
    energies['total'].backward()
    
    # Forces are negative gradient
    autograd_forces = -coords.grad.clone()
    
    print('===== Autograd Forces (from .backward()) =====')
    pp(energies)
    print("Forces (first 3 atoms):")
    pp(autograd_forces[:3].cpu().numpy())
    
    ##########
    # 2) Forces using numerical Finite Difference
    print('\n===== Finite Difference Forces =====')
    
    # Detach coords for FD calculation
    coords_fd = coords.detach().clone()
    coords_fd.requires_grad = False
    
    fd_forces = finite_difference_forces(system, coords_fd, box, epsilon=1e-5)
    
    print("Forces (first 3 atoms):")
    pp(fd_forces[:3].cpu().numpy())
    
    ##########
    # 3) Compare the two methods
    print('\n===== Comparison =====')
    
    autograd_forces_np = autograd_forces.cpu().numpy()
    fd_forces_np = fd_forces.cpu().numpy()
    
    # Calculate differences
    abs_diff = np.abs(autograd_forces_np - fd_forces_np)
    rel_diff = abs_diff / (np.abs(autograd_forces_np) + 1e-10)
    
    print(f"Max absolute difference: {np.max(abs_diff):.6e}")
    print(f"Mean absolute difference: {np.mean(abs_diff):.6e}")
    print(f"Max relative difference: {np.max(rel_diff):.6e}")
    print(f"Mean relative difference: {np.mean(rel_diff):.6e}")
    
    # Check if they match within tolerance
    tolerance = 1e-4
    is_close = np.allclose(autograd_forces_np, fd_forces_np, rtol=tolerance, atol=tolerance)
    print(f"\nForces match within tolerance ({tolerance})? {is_close}")
    
    # Show worst mismatches
    worst_indices = np.argsort(abs_diff.flatten())[-5:]
    print("\nWorst 5 mismatches:")
    for idx in reversed(worst_indices):
        atom_idx = idx // 3
        dim_idx = idx % 3
        dim_name = ['x', 'y', 'z'][dim_idx]
        print(f"  Atom {atom_idx}, {dim_name}: autograd={autograd_forces_np.flatten()[idx]:.6e}, "
              f"FD={fd_forces_np.flatten()[idx]:.6e}, diff={abs_diff.flatten()[idx]:.6e}")


if __name__ == "__main__":
    test_customized_ops()
