import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torch.utils.benchmark as benchmark

import numpy as np
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import cmm

from cmm.units import HARTREE2KCAL, BOHR2ANG, AMU2ELECTRON_MASS, FS2AU, KB_EhPerK
from cmm.misc_utils import read_xyz_tinker
from cmm.coordinate_manager import CoordinateManager
from cmm.topology import Topology
from cmm.parameters import Parameterizer
from cmm.force_field import CMM
from cmm.interfaces import CMM_ASE
from cmm.dynamics import VelocityVerlet
from ase.optimize import LBFGS
from ase.data import atomic_masses, atomic_numbers
import ase

def get_water_box_coords(requires_grad=True, device="cpu"):
    labels, atom_types, coords, bonds = read_xyz_tinker(os.path.join(os.path.dirname(__file__), "../tests/data/water_216.xyz"))
    permutation = np.argsort(bonds[0], kind='stable') # Make sure sort is stable so equivalent indices don't get swapped.
    bonds[0] = bonds[0][permutation]
    bonds[1] = bonds[1][permutation]
    atom_types = torch.tensor(atom_types, dtype=torch.long, requires_grad=False, device=device) - 1
    coords = torch.tensor(coords / BOHR2ANG, requires_grad=requires_grad, device=device)
    return labels, coords, atom_types, bonds

def profile_md():
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    labels, coords, atom_types, bonds = get_water_box_coords(requires_grad=True, device=device)
    masses = [atomic_masses[atomic_numbers[labels[i]]] for i in range(len(labels))]
    masses = torch.as_tensor(masses, device=device) * AMU2ELECTRON_MASS
    
    # Normally, the parser should enforce just returning the names of atom types
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, requires_grad=True, device=device)
    cm = CoordinateManager(coords, box, 10.0 / BOHR2ANG, 1024)
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = CMM(use_ewald=True)
    with torch.no_grad():
        topology = Topology(bonds, cm.neighbor_list, coords.size(0))
        parameters = Parameterizer(
            atom_type_names, pairs, topology.angle_atoms,
            ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
        )

    # Set up velocity verlet integrator #
    kT = torch.tensor(KB_EhPerK * 300, device=device) # in Hartree
    stds = torch.sqrt(kT.expand(masses.size(0)) / masses).repeat_interleave(3, 0)
    velocities = torch.normal(torch.zeros(3 * masses.size(0), device=device), stds).reshape(-1, 3)
    integrator = VelocityVerlet(
        velocities, masses,
        timestep=0.5 * FS2AU
    )

    with profile(activities=[ProfilerActivity.CUDA], profile_memory=True, record_shapes=True, with_stack=True) as prof:
        with record_function("CMM_5_steps_nve_water_box_216"):
            energies_ff = ff.evaluate(cm, topology, parameters)
            energies_ff['tot'].backward()
            for _ in range(5):
                new_coords = integrator.first_half_step(cm.coords)
                cm.update_coordinates(new_coords)
                energies_ff = ff.evaluate(cm, topology, parameters)
                energies_ff['tot'].backward()
                integrator.second_half_step()
                
                # Synchronization after each step to get kernel info
                torch.cuda.synchronize()
    
        # Final synchronization
        torch.cuda.synchronize()
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    #prof.export_chrome_trace("trace.json")

def profile_optimization():
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    labels, coords, atom_types, bonds = get_water_box_coords(requires_grad=True, device=device)
    
    # Normally, the parser should enforce just returning the names of atom types
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, requires_grad=True, device=device)
    cm = CoordinateManager(coords, box, 10.0 / BOHR2ANG, 1024)
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = CMM(use_ewald=True)
    with torch.no_grad():
        topology = Topology(bonds, cm.neighbor_list, coords.size(0))
        parameters = Parameterizer(
            atom_type_names, pairs, topology.angle_atoms,
            ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
        )
    ff_ase = CMM_ASE(ff, cm, topology, parameters)
    ff_ase.calculate()
    dyn = LBFGS(ff_ase.atoms)
    dyn.run(fmax=1e-6)

    #def run_model_forward_and_backward(ff, cm, topology, parameters):
    #    energies = ff.evaluate(cm, topology, parameters)
    #    energies['tot'].backward(retain_graph=True)
    #    return energies
    
    #t = benchmark.Timer(
    #   stmt = 'run_model_forward_and_backward(ff, cm, topology, parameters)',
    #   globals={
    #       'run_model_forward_and_backward': run_model_forward_and_backward,
    #       'ff': ff, 'cm': cm, 'topology': topology, 'parameters': parameters
    #    },
    #   label="Average Inference Duration",
    #)

    #with profile(activities=[ProfilerActivity.CUDA], profile_memory=True, record_shapes=False) as prof:
    #    with record_function("CMM_evaluate_water_box_216"):
    #        energies = ff.evaluate(cm, topology, parameters)
    #        energies['tot'].backward()
    #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

if __name__ == "__main__":
    profile_md()
    #profile_optimization()
    