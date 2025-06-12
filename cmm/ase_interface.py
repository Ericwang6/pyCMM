from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.units import Bohr, Hartree
from ase.stress import full_3x3_to_voigt_6_stress, voigt_6_to_full_3x3_stress

from cmm.system import System
from cmm.force_fields.ff import FF
import numpy as np
import torch
import os

class ASE_Interface(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']
    calculate_numerical_stress = False
    calculate_numerical_forces = False

    def __init__(self, ff: FF, system: System):
        super().__init__()
        self._ff = ff
        self._system = system
        use_pbc = system.settings.get_long_range_electrostatics_settings().use_long_range
        
        self.atoms = Atoms(
            positions=self._system.coords.cpu().detach().numpy() * Bohr,
            cell=self._system.box.cpu().detach().numpy() * Bohr,
            pbc=[use_pbc, use_pbc, use_pbc],
            symbols=system.labels
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
    
    def _evaluate_ff(self):
        self._ff.forward(self._system)
        self._energies = self._ff.energies
        if self._system._need_coordinate_grads:
            self._energies['V_total'].backward()

        # Store results so that ASE can access them #
        self.results['energy'] = self._energies['V_total'].item() * Hartree
        if self._system.coords.grad is not None:
            self.results['forces'] = -self._system.coords.grad.cpu().numpy() * (Hartree / Bohr)
            self.results['stress'] = (
                torch.matmul(self._system.coords.grad.T, self._system.coords) / torch.det(self._system.box)
            ).cpu().detach().numpy() * (Hartree / Bohr**3)
            if self._system.box.grad is not None:
                self.results['stress'] = self.results['stress'] + ((
                    torch.matmul(self._system.box.grad.T, self._system.box)
                 ) / torch.det(self._system.box)).cpu().detach().numpy() * (Hartree / Bohr**3)

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
        positions_tensor = torch.from_numpy(self.atoms.get_positions() / Bohr).to(self._system.device)
        box_tensor = torch.from_numpy(self.atoms.get_cell().array / Bohr).to(self._system.device)
        self._system.update_coordinates(positions_tensor)
        self._system.update_box(box_tensor)

        # Calculate forces, energy, and stress then store hash for this configuration #
        self._evaluate_ff()
        self._last_positions = self.atoms.get_positions()
        self._last_atoms_hash = current_hash

    def get_potential_energy(self, atoms=None, force_consistent=False, apply_constraint=False):
        self.calculate(self.atoms, properties=['energy'])
        return self.results['energy']
    
    def get_forces(self, atoms=None):
        self.calculate(self.atoms, properties=['forces'])
        return self.results['forces']

    def get_stress(self, voigt=False, include_ideal_gas=True):
        # Note: The ideal gas part is added internally by ASE.
        # By turning off all interaction terms, I have validated that
        # we reproduce the volume predicted by the ideal gas law for
        # particular choices of N,P, and T.
        self.calculate(self.atoms, properties=['stress'])
        stress = self.results['stress']
        if voigt:
            stress = full_3x3_to_voigt_6_stress(stress)
        return stress
    
    def _get_configuration_hash(self, atoms):
        """Generate a hash that uniquely identifies the atomic configuration"""
        positions_hash = hash(np.array2string(atoms.positions, precision=10))
        if atoms.cell is not None and np.any(atoms.cell != 0.0):
            cell_hash = hash(np.array2string(atoms.cell, precision=10))
            return hash((positions_hash, cell_hash))
        else:
            return positions_hash