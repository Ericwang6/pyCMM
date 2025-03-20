from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.units import Bohr, Hartree

from cmm.parameters import Parameterizer
from cmm.coordinate_manager import CoordinateManager
from cmm.topology import Topology
from cmm.force_field import CMM
import numpy as np
import torch

class CMM_ASE(Calculator):
    def __init__(self, ff: CMM, cm: CoordinateManager, topology: Topology, params: Parameterizer):
        super().__init__()
        self._ff = ff
        self._cm = cm
        self._topology = topology
        self._params = params
        
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
    
    @classmethod
    def from_atoms(cls, atoms):
        atomic_numbers_to_names = {8: "O_water", 1: "H_water"}
        atom_type_names = [atomic_numbers_to_names[atomic_number] for atomic_number in atoms.get_atomic_numbers()]
        #box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, dtype=torch.float64, requires_grad=True, device=device)
        #cm = CoordinateManager(coords, box, 10.0 / BOHR2ANG, labels=labels, max_neighbors=1024)
        #topology = Topology(bonds, cm.neighbor_list, coords.size(0))
        #pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
        #ff = CMM()
        #parameters = Parameterizer(
        #    atom_type_names, pairs, topology.angle_atoms,
        #    ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
        #)

    def _evaluate_ff(self):
            self._energies = self._ff.evaluate(self._cm, self._topology, self._params)
            self._energies['tot'].backward()

            self.results['energy'] = float(self._energies['tot'].detach().cpu()) * Hartree
            self.results['forces'] = -self._cm.coords.grad.detach().cpu().numpy() * (Hartree / Bohr)

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
        
        self._evaluate_ff()