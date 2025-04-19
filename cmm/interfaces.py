from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Bohr, Hartree
from ase.stress import full_3x3_to_voigt_6_stress, voigt_6_to_full_3x3_stress

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
    implemented_properties = ['energy', 'forces', 'stress']
    calculate_numerical_stress = False
    calculate_numerical_forces = False

    def __init__(self, ff: CMM, cm: CoordinateManager, topology: Topology, params: Parameterizer,
                 output_folder: str="", use_cache=True):
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
        self._last_atoms_hash = None
        self._last_positions = None

        # Set ourselves as the calculator
        self.atoms.calc = self

        self._energies = {}
        self.results = {
            'energy': 0.0,
            'forces': np.zeros((len(self.atoms), 3)),
            'stress': np.zeros((3, 3))
        }
    
    def save_state(self, filename: str):
        """
        Save the state of the CMM_ASE calculator for restart purposes.
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
            'cutoff_max': float(self._cm.cutoff.item()),
            'cutoff_ewald': float(self._ff.cutoff_ewald.item()),
            'cutoff_short_range': float(self._ff.cutoff_sr.item()),
            'require_coord_grads': self._cm._need_coordinate_grads,
            'require_box_grads': self._cm._need_box_grads,
            'max_neighbors': self._cm.max_neighbors,
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
                                requires_grad=state['require_coord_grads'],
                                device=device)
        box = torch.tensor(state['cell'], dtype=dtype,
                          requires_grad=state['require_box_grads'],
                          device=device)
        bonds = torch.tensor(state['bonds'], device=device)
        cm = CoordinateManager(positions, box, state['cutoff_max'],
                             labels=state['atom_labels'], max_neighbors=state['max_neighbors'])
        topology = Topology(bonds, cm.neighbor_list, positions.size(0))
        if ff is None:
            use_ewald = state['use_ewald']
            use_polarization = state['use_polarization']
            solve_tolerance = torch.tensor(state['solve_tolerance'], device=device, dtype=dtype)
            ewald_tolerance = torch.tensor(state['ewald_tolerance'], device=device, dtype=dtype)
            cutoff_ewald = torch.tensor(state['cutoff_ewald'], device=device, dtype=dtype)
            cutoff_short_range = torch.tensor(state['cutoff_short_range'], device=device, dtype=dtype)
            ff = CMM(
                cutoff_ewald=cutoff_ewald, cutoff_short_range=cutoff_short_range,
                use_ewald=use_ewald, use_polarization=use_polarization,
                solve_tolerance=solve_tolerance, ewald_tolerance=ewald_tolerance
            )

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
    
    @classmethod
    def load_last_state(cls, directory: str, ff=None):
        """
        Load a CMM_ASE calculator from the most recent saved state file in a directory.

        This method searches the specified directory for checkpoint files and loads
        the most recent one based on the timestamp in the filename.
        """
        import os
        import re
        import glob

        # Pattern to match checkpoint files with timestamps
        # Expected format: checkpoint_YYYYMMDD_HHMMSS.json
        checkpoint_pattern = os.path.join(directory, "checkpoint_*.json")
        checkpoint_files = glob.glob(checkpoint_pattern)

        if not checkpoint_files:
            # Also try to match any .json file that might be a checkpoint
            alternative_pattern = os.path.join(directory, "*.json")
            checkpoint_files = glob.glob(alternative_pattern)

        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {directory}")

        # Extract timestamps from filenames
        timestamp_pattern = re.compile(r'.*_(\d{8}_\d{6})\.json$')

        # Try to find files with timestamps (like checkpoint_20220101_120000.json)
        timestamped_files = []
        for filepath in checkpoint_files:
            match = timestamp_pattern.match(filepath)
            if match:
                timestamp = match.group(1)
                timestamped_files.append((filepath, timestamp))

        if timestamped_files:
            # Sort by timestamp (most recent last)
            timestamped_files.sort(key=lambda x: x[1])
            latest_file = timestamped_files[-1][0]
        else:
            # If no files match the timestamp pattern, use file modification time
            checkpoint_files.sort(key=os.path.getmtime)
            latest_file = checkpoint_files[-1]

        return cls.load_state(latest_file, ff=ff)
    
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
        self.results['stress'] = (
            torch.matmul(self._cm.coords.grad.T, self._cm.coords) / self._cm.box_volume
        ).detach().cpu().numpy() * (Hartree / Bohr**3)
        if self._cm.box.grad is not None:
            self.results['stress'] = self.results['stress'] + ((
                torch.matmul(self._cm.box.grad.T, self._cm.box)
             ) / self._cm.box_volume).detach().cpu().numpy() * (Hartree / Bohr**3)

    def calculate(self, atoms=None, properties=None, system_changes=['positions', 'cell']):
        if properties is None:
            properties = self.implemented_properties
        super().calculate(atoms, properties, system_changes)
    
        if atoms is not None and atoms is not self.atoms:
            self.atoms = atoms
            self.atoms.calc = self
        
        current_hash = self._get_configuration_hash(self.atoms)
        if self._last_atoms_hash == current_hash:
            return
        if self._last_positions is not None:
            # NOTE(JOE): I am not sure if this is right or not...
            # We might just have to eat the two evaluations per step with NPT.
            pos_ratio = self.atoms.get_positions() / self._last_positions
            if np.max(pos_ratio - pos_ratio[0]) < 1e-12:
                return
        positions_tensor = torch.from_numpy(self.atoms.get_positions() / Bohr).to(self._cm.coords.device)
        box_tensor = torch.from_numpy(self.atoms.get_cell().array / Bohr).to(self._cm.coords.device)
        self._cm.update_coordinates(positions_tensor)
        self._cm.update_box(box_tensor)

        # Calculate forces, energy, and stress then store hash for this configuration #
        self._evaluate_ff()
        self._last_positions = self.atoms.get_positions()
        self._last_atoms_hash = current_hash

    def get_potential_energy(self, atoms=None, force_consistent=False, apply_constraint=False):
        """Get potential energy for current atomic configuration"""
        self.calculate(self.atoms, properties=['energy'])
        return self.results['energy']
    
    def get_forces(self, atoms=None):
        """Get forces for current atomic configuration"""
        self.calculate(self.atoms, properties=['forces'])
        return self.results['forces']

    def get_stress(self, voigt=False, include_ideal_gas=True):
        """Get stress for current atomic configuration"""
        self.calculate(self.atoms, properties=['stress'])
        stress = self.results['stress']
        if voigt:
            stress = full_3x3_to_voigt_6_stress(stress)
        if include_ideal_gas:
            return stress - self.atoms.get_kinetic_stress(voigt=voigt)
        return stress
    
    def get_dipole_moment(self, include_induced_moments: bool = True):
        dipole_moment = self._ff.get_dipole_moment(self._cm.coords, include_induced_moments=include_induced_moments)
        return dipole_moment.detach().cpu().numpy()
    
    def _get_configuration_hash(self, atoms):
        """Generate a hash that uniquely identifies the atomic configuration"""
        positions_hash = hash(np.array2string(atoms.positions, precision=10))
        if atoms.cell is not None and np.any(atoms.cell != 0.0):
            cell_hash = hash(np.array2string(atoms.cell, precision=10))
            return hash((positions_hash, cell_hash))
        else:
            return positions_hash