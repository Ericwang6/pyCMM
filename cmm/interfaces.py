from ase import Atoms
from ase.calculators.calculator import Calculator

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
            positions=cm.coords.detach().cpu().numpy(),
            cell=cm.box.detach().cpu().numpy(),
            pbc=[1, 1, 1],
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
    
    def _evaluate_ff(self):
            self._energies = self._ff.evaluate(self._cm, self._topology, self._params)
            self._energies['tot'].backward()

            self.results['energy'] = float(self._energies['tot'].detach().cpu())
            self.results['forces'] = -self._cm.coords.grad.detach().cpu().numpy()

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
            self._cm.update_coordinates(torch.from_numpy(self.atoms.positions).to(self._cm.coords.device))
        
        self._evaluate_ff()