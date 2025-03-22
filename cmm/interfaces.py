import ase
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

class CMM_ASE(Calculator):
    def __init__(self, ff: CMM, cm: CoordinateManager, topology: Topology, params: Parameterizer, output_folder: str = "."):
        super().__init__()
        self._ff = ff
        self._cm = cm
        self._topology = topology
        self._params = params
        if output_folder == "":
            self.output_folder = "."
        else:
            self.output_folder = output_folder

        self._checkpoint_counter = 0
        
        self.atoms = Atoms(
            positions=cm.coords.detach().cpu().numpy() * Bohr,
            cell=cm.box.detach().cpu().numpy() * Bohr,
            pbc=[1, 1, 1],
            symbols=cm.labels,
            calculator=self
        )

        self.implemented_properties = ['energy', 'forces'] # TODO: add stress and dipole
    
        self._energies = {}

        self.results = {'energy': 0.0,
                'forces': np.zeros((len(self.atoms), 3)),
                'stress': np.zeros(6),
                'dipole': np.zeros(3),
                'charges': np.zeros(len(self.atoms)),
                'magmom': 0.0,
                'magmoms': np.zeros(len(self.atoms))}
        
        #self.calculate(self.atoms) # Calculate at beginning to seed forces
    
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
            'cell': self._cm.box.detach().cpu().numpy().tolist(),
            'atom_labels': self._cm.labels,
            'atom_type_names': [self._params._atom_type_names[i] for i in range(len(self._params._atom_type_names))],
            'bonds': self._topology.bonded_atoms.detach().cpu().numpy().tolist(),
            'pbc': [True, True, True],  # Usually true for CMM
            'cutoff': float(self._cm.cutoff.item()),
            'requires_grad': self._cm._need_coordindate_grads,
            'device': str(self._cm.coords.device),
            'torch_dtype': str(self._cm.coords.dtype),
        }

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

        # Save the dictionary to a JSON file
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)

        # Optionally, save the current atomic positions as an ASE trajectory
        # This is redundant but can be useful for visualization
        if filename.endswith('.json'):
            traj_filename = filename.replace('.json', '.traj')
        else:
            traj_filename = filename + '.traj'

        from ase.io import write
        write(traj_filename, self.atoms)

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

        # Load the state from the JSON file
        with open(filename, 'r') as f:
            state = json.load(f)

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

        # Convert bonds to tensor
        bonds = torch.tensor(state['bonds'], device=device)

        # Create CoordinateManager
        cm = CoordinateManager(positions, box, state['cutoff'], 
                             labels=state['atom_labels'])

        # Create Topology
        topology = Topology(bonds, cm.neighbor_list, positions.size(0))

        # Create or use provided force field
        if ff is None:
            ff = CMM()

        # Create Parameterizer
        pairs, _, _ = cm.get_distances_vectors_and_pairs()
        parameters = Parameterizer(
            state['atom_type_names'], pairs, topology.angle_atoms,
            ff.atomic_params, ff.pair_params, ff.pair_pair_params, 
            ff.pair_angle_params, ff.angle_params
        )

        # Create and return CMM_ASE calculator
        return cls(ff, cm, topology, parameters)
    
    def create_checkpoint(self, filename_prefix='checkpoint'):
        """
        Create a checkpoint that can be used to restart the simulation.

        Args:
            filename_prefix (str): Prefix for the checkpoint files.

        Returns:
            str: Name of the checkpoint file.
        """
        import os
    
        # Alternate between checkpoint_0 and checkpoint_1
        suffix = self._checkpoint_counter % 2
        self._checkpoint_counter += 1

        # Create filename
        filename = os.path.join(self.output_folder, f"{filename_prefix}_{suffix}.json")

        # Save the state
        self.save_state(filename)

        return filename

    def write_trajectory(self, filename: str, mode: str='a', properties=None):
        """
        Write the current state to a trajectory file.

        Args:
            filename (str): Name of the trajectory file.
            mode (str): Write mode, 'a' for append, 'w' for write.
            properties (list): Properties to include in the trajectory.
        """
        import os
        from ase.io import write

        # Create absolute path using output folder
        full_path = os.path.join(self.output_folder, filename)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(full_path)), exist_ok=True)

        # Write atomic configuration to trajectory
        write(full_path, self.atoms, mode=mode, properties=properties)

        # Also save the topology info to enable proper restart
        # Only save this information if writing a new file or appending first frame
        if mode == 'w' or (mode == 'a' and not os.path.exists(full_path)):
            # Save topology file alongside trajectory
            topo_filename = full_path + '.topology.json'
            self.save_state(topo_filename)

    def get_potential_energy(self, atoms=None, force_consistent=False, apply_constraint=True):
        self.calculate(atoms=atoms)
        return self.results['energy']
    
    #def get_forces(self, atoms=None, apply_constraint=True, md=False):
    #    return self.results['forces']

    # TODO: ASE does not always populate these properties or system_changes arrays
    # which makes it difficult to know how to respond to a call to calculate.
    # Currently I have removed the get_forces method since having this present
    # just seems to result in calculations happening twice. I should basically
    # keep track of when we last updated the positions, last evaluated the force
    # field and last back-propagated through so that I don't have to rely on ASE
    # to figure out what needs to be calculated. Instead, I can just do the minimal
    # work required for whatever function ASE is calling.
    def calculate(self,
        atoms=None,
        properties=['energy', 'forces', 'stress', 'dipole'],
        system_changes=[]
    ):
        if atoms:
            # Update positions on GPU #
            self.atoms = atoms
        self._cm.update_coordinates(torch.from_numpy(self.atoms.positions / Bohr).to(self._cm.coords.device))

        self._energies = self._ff.evaluate(self._cm, self._topology, self._params)
        self._energies['tot'].backward()

        self.results['energy'] = float(self._energies['tot'].detach().cpu()) * Hartree
        self.results['forces'] = -self._cm.coords.grad.detach().cpu().numpy() * (Hartree / Bohr)