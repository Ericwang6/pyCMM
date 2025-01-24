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
        self.atoms = Atoms(positions=cm.coords.detach().cpu().numpy(), cell=cm.box.detach().cpu().numpy(), pbc=[1, 1, 1])
        self._ff = ff
        self._cm = cm
        self._topology = topology
        self._params = params
    
        self.results = {'energy': 0.0,
                'forces': np.zeros((len(self.atoms), 3)),
                'stress': np.zeros(6),
                'dipole': np.zeros(3),
                'charges': np.zeros(len(self.atoms)),
                'magmom': 0.0,
                'magmoms': np.zeros(len(self.atoms))}
    
    def _evaluate_ff(self, evaluate_gradients: bool):
        energies = self._ff.evaluate(self._cm, self._topology, self._params)
        self.results['energy'] = float(energies['tot'].detach().cpu())
        if evaluate_gradients:
            energies['tot'].backward()
            self.results['forces'] = self._cm.coords.grad.detach().cpu().numpy()

    def calculate(self,
        properties=['energy', 'forces', 'stress', 'dipole'],
        system_changes=[]
    ):
        if 'positions' in system_changes:
            # Update positions on GPU #
            self._cm.update_coordinates(torch.from_numpy(self.atoms.positions).to(self._cm.coords.device))
            print("Positions changed")
        
        self._evaluate_ff('forces' in properties)
        print(self.results['energy'])
        print(self.results['forces'])