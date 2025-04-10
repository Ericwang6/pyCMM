from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.units import Bohr, Hartree

from cmm.parameters import Parameterizer
from cmm.coordinate_manager import CoordinateManager
from cmm.topology import Topology
from cmm.force_field import CMM
from cmm.units import BOHR2ANG
import numpy as np
import torch
import os
from typing import Optional

class CMM_ASE(Calculator):
    def __init__(self, ff: CMM, cm: CoordinateManager, topology: Topology, params: Parameterizer,
                 output_folder: str=""):
        super().__init__()
        self._ff = ff
        self._cm = cm
        self._topology = topology
        self._params = params
        if output_folder == "":
            self.output_folder = "."
        else:
            self.output_folder = output_folder

        os.makedirs(os.path.abspath(self.output_folder), exist_ok=True)

        self._checkpoint_counter = 0
        
        self.atoms = Atoms(
            positions=cm.coords.detach().cpu().numpy() * Bohr,
            cell=cm.box.detach().cpu().numpy() * Bohr,
            pbc=[ff.use_ewald, ff.use_ewald, ff.use_ewald],
            symbols=cm.labels
        )

        # Set ourselves as the calculator
        self.atoms.calc = self

        self.implemented_properties = ['energy', 'forces', 'stress']
    
        self._energies = {}

        self.results = {
            'energy': 0.0,
            'forces': np.zeros((len(self.atoms), 3)),
            'stress': np.zeros((3, 3))
        }
    
    def save_state(self, filename: str):
        """
        Save the state of the CMM_ASE calculator for restart purposes.

        Args:
            filename (str): The name of the file to save the state to.
        """
        import json, os

        # Create a dictionary with all the necessary information
        state = {
            'positions': self._cm.coords.detach().cpu().numpy().tolist(),
            'velocities': self.atoms.get_velocities().tolist(),  # Save velocities
            'cell': self._cm.box.detach().cpu().numpy().tolist(),
            'atom_labels': self._cm.labels,
            'atom_type_names': [self._params._atom_type_names[i] for i in range(len(self._params._atom_type_names))],
            'bonds': self._topology.bonded_atoms.detach().cpu().numpy().tolist(),
            'cutoff': float(self._cm.cutoff.item()),
            'requires_grad': self._cm._need_coordindate_grads,
            'device': str(self._cm.coords.device),
            'torch_dtype': str(self._cm.coords.dtype),
            'output_folder': str(self.output_folder),
            'solve_tolerance': self._ff.solve_tolerance.detach().cpu().numpy().tolist(),
            'ewald_tolerance': self._ff.ewald_tolerance.detach().cpu().numpy().tolist(),
            'use_ewald': self._ff.use_ewald,
            'use_polarization': self._ff.use_polarization,
        }

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

        # Save the dictionary to a JSON file
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load_state(cls, filename, ff=None):
        """
        Load a CMM_ASE calculator from a saved state file.

        Args:
            filename (str): The name of the file to load the state from.
            ff (CMM, optional): The force field to use. If None, a new CMM
                               instance will be created with default parameters.

        Returns:
            CMM_ASE: A reconstructed CMM_ASE calculator.
        """
        import json
        import torch
        from cmm.coordinate_manager import CoordinateManager
        from cmm.topology import Topology
        from cmm.parameters import Parameterizer
        from cmm.force_field import CMM
        import os
        import numpy as np
        from ase.io import read as ase_read

        # Load the state from the JSON file
        with open(filename, 'r') as f:
            state = json.load(f)

        # Check for trajectory file to load velocities
        traj_filename = None
        if filename.endswith('.json'):
            traj_filename = filename.replace('.json', '.traj')
        else:
            traj_filename = filename + '.traj'

        has_traj = os.path.exists(traj_filename)

        # Determine device and dtype
        device = state.get('device', 'cpu')
        if device == 'cpu':
            device = torch.device('cpu')
        else:
            # Handle CUDA devices
            device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Determine torch dtype
        dtype_str = state.get('torch_dtype', 'torch.float64')
        if dtype_str == 'torch.float64':
            dtype = torch.float64
        elif dtype_str == 'torch.float32':
            dtype = torch.float32
        else:
            dtype = torch.float64  # Default to float64

        # Convert positions and cell to tensors
        positions = torch.tensor(state['positions'], dtype=dtype, 
                                requires_grad=state['requires_grad'], 
                                device=device)
        box = torch.tensor(state['cell'], dtype=dtype, 
                          requires_grad=state['requires_grad'], 
                          device=device)
        bonds = torch.tensor(state['bonds'], device=device)
        cm = CoordinateManager(positions, box, state['cutoff'], 
                             labels=state['atom_labels'])
        topology = Topology(bonds, cm.neighbor_list, positions.size(0))
        if ff is None:
            use_ewald = state['use_ewald']
            use_polarization = state['use_polarization']
            solve_tolerance = torch.tensor(state['solve_tolerance'], device=device, dtype=dtype)
            ewald_tolerance = torch.tensor(state['ewald_tolerance'], device=device, dtype=dtype)
            ff = CMM(use_ewald=use_ewald, use_polarization=use_polarization, solve_tolerance=solve_tolerance, ewald_tolerance=ewald_tolerance)

        # Create Parameterizer
        pairs, _, _ = cm.get_distances_vectors_and_pairs()
        parameters = Parameterizer(
            state['atom_type_names'], pairs, topology.angle_atoms,
            ff.atomic_params, ff.pair_params, ff.pair_pair_params, 
            ff.pair_angle_params, ff.angle_params
        )


        # Create CMM_ASE calculator
        calculator = cls(ff, cm, topology, parameters, output_folder=state['output_folder'])
        calculator.atoms.set_velocities(np.array(state['velocities']))
        calculator.atoms.set_pbc([use_ewald, use_ewald, use_ewald])

        return calculator
    
    def create_checkpoint(self, filename_prefix='checkpoint'):
        """
        Create a checkpoint that can be used to restart the simulation.

        Args:
            filename_prefix (str): Prefix for the checkpoint files.

        Returns:
            str: Name of the checkpoint file.
        """
        import os, time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        json_filename = os.path.join(self.output_folder, f"{filename_prefix}_{timestamp}.json")
        self.save_state(json_filename)
        return json_filename
    
    def _evaluate_ff(self):
        self._energies = self._ff.evaluate(self._cm, self._topology, self._params, reset_grads=True)
        self._energies['total'].backward()

        self.results['energy'] = float(self._energies['total'].detach().cpu()) * Hartree
        self.results['forces'] = -self._cm.coords.grad.detach().cpu().numpy() * (Hartree / Bohr)
        if self._cm.box.grad is not None:
            self.results['stress'] = ((
                torch.matmul(self._cm.coords.grad.T, self._cm.coords) + torch.matmul(self._cm.box.grad.T, self._cm.box)
             ) / self._cm.box_volume).detach().cpu().numpy() * (Hartree / Bohr**3)

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=['positions']):
        # Call the parent implementation first 
        Calculator.calculate(self, atoms, properties, system_changes)
    
        # If atoms is provided and it's different from our internal one,
        # update our reference
        if atoms is not None and atoms is not self.atoms:
            self.atoms = atoms
            self.atoms.calc = self

        # Only update coordinates if positions have changed
        if 'positions' in system_changes:
            positions_tensor = torch.from_numpy(self.atoms.get_positions() / Bohr).to(self._cm.coords.device)
            self._cm.update_coordinates(positions_tensor)

        # Only update box if cell has changed
        if 'cell' in system_changes:
            box_tensor = torch.from_numpy(self.atoms.get_cell().array / Bohr).to(self._cm.coords.device)
            self._cm.update_box(box_tensor)

        # Ensure coordinates have gradients enabled
        if not self._cm.coords.requires_grad:
            self._cm.coords.requires_grad_(True)

        # Calculate forces and energy
        self._evaluate_ff()

    def get_potential_energy(self, atoms=None, force_consistent=False):
        """Get potential energy for current atomic configuration"""
        if atoms is not None:
            self.calculate(atoms, ['energy'], ['positions', 'cell', 'numbers', 'pbc'])
        else:
            self.calculate(self.atoms, ['energy'], ['positions', 'cell', 'numbers', 'pbc'])
        return self.results['energy']
    
    def get_forces(self, atoms=None):
        """Get forces for current atomic configuration"""
        if atoms is not None:
            self.calculate(atoms, ['forces'], ['positions', 'cell', 'numbers', 'pbc'])
        else:
            self.calculate(self.atoms, ['forces'], ['positions', 'cell', 'numbers', 'pbc'])
        return self.results['forces']

    def get_stress(self, atoms=None, include_ideal_gas=True):
        """Get stress for current atomic configuration"""
        if atoms is not None:
            self.calculate(atoms, ['stress'], ['positions', 'cell', 'numbers', 'pbc'])
        else:
            self.calculate(self.atoms, ['stress'], ['positions', 'cell', 'numbers', 'pbc'])
        if include_ideal_gas:
            return self.results['stress'] + self.atoms.get_kinetic_stress(voigt=False)
        return self.results['stress']
    
    def get_dipole_moment(self, include_induced_moments: bool = True):
        dipole_moment = self._ff.get_dipole_moment(self._cm.coords, include_induced_moments=include_induced_moments)
        return dipole_moment.detach().cpu().numpy()