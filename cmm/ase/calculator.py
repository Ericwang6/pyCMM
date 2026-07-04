from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Bohr, Hartree
from ase.stress import (
    full_3x3_to_voigt_6_stress, voigt_6_to_full_3x3_stress
)
from collections import defaultdict
# from cmm.parameters import Parameterizer
# from cmm.coordinate_manager import CoordinateManager
# from cmm.topology import Topology
# from cmm.system import System
# from cmm.units import BOHR2ANG
import time
import numpy as np
import torch
import os
from typing import Optional


class CMMCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']
    calculate_numerical_stress = False
    calculate_numerical_forces = False

    def __init__(
        self, 
        system, 
        topology,
        coords: torch.Tensor,
        box: torch.Tensor,
        output_folder: os.PathLike = ".",
        use_cache=True,
        profile=False,
        use_cuda_graph=False,
        rebuild_nblist_interval=25
    ):
        super().__init__()
        self._system = system
        self._topology = topology
        self.coords = torch.zeros_like(coords, requires_grad=True)
        self.box = torch.zeros_like(box, requires_grad=True)

        with torch.no_grad():
            self.coords.copy_(coords)
            self.box.copy_(box)
        
        self.output_folder = output_folder
        os.makedirs(os.path.abspath(self.output_folder), exist_ok=True)
        
        self.atoms = Atoms(
            positions=coords.numpy(force=True) * Bohr,
            cell=box.numpy(force=True) * Bohr,
            pbc=[system.use_ewald, system.use_ewald, system.use_ewald],
            symbols=topology.atomSymbols
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
        self._run_profile = profile
        self._profiler = defaultdict(float)
        self._profiler_count = 0

        # run the system immediately to initalize Ewald, induced_dipoles etc
        self.cutoff_verlet = self._system.cutoff_lr + 3.0 # in bohr
        self.use_cuda_graph = use_cuda_graph
        
        if self.use_cuda_graph:
            assert self._system.use_customized_ops, 'CUDA graph has to be used with customized ops'
        
        with torch.no_grad():
            self.pairs = self._system.rebuild_nblist(self.coords, self.cutoff_verlet, self.box)
            # self.pairs = torch.full((int(pairs.shape[0]*1.5), 2), -1, device=pairs.device, dtype=pairs.dtype)
            # self.pairs[:pairs.shape[0]].copy_(pairs)

        self.step_counter = 0
        self._minimization = True
        self._rebuild_nblist_interval = rebuild_nblist_interval
        self._start_graph = 100
        self._cuda_graph = torch.cuda.CUDAGraph()
        self._evaluate_ff()
    
    def set_minimization(self):
        self._minimization = True
    
    def unset_minimization(self):
        self._minimization = False
        
    def reset_profiler(self):
        for key in self._profiler:
            self._profiler[key] = 0.0
        self._profiler_count = 0
    
    def print_profiler(self):
        for key in self._profiler:
            print(f"{key}: {self._profiler[key] / self._profiler_count * 1000:.4f} ms")
    
    def _evaluate_ff(self):
        if (not self.use_cuda_graph) or (self.step_counter <= self._start_graph):
            if self.coords.grad is not None:
                self.coords.grad.zero_()
            if self.box.grad is not None:
                self.box.grad.zero_()
            if self._run_profile:
                torch.cuda.synchronize()
                start = time.perf_counter()
                torch.cuda.nvtx.range_push("CMM-Forward")
            self._energies = self._system.getEnergy(self.coords, self.box)
            if self._run_profile:
                torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
                end = time.perf_counter()
                self._profiler['forward'] += end - start
                self._profiler_count += 1

            if self.coords.requires_grad:
                if self._run_profile:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    torch.cuda.nvtx.range_push("CMM-Backward")
                self._energies['total'].backward()
                if self._run_profile:
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                    self._profiler['backward'] += end - start
        else:
            if self._run_profile:
                torch.cuda.synchronize()
                start = time.perf_counter()
                torch.cuda.nvtx.range_push("CMM-GRAPH")
            self._cuda_graph.replay()
            if self._run_profile:
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
                end = time.perf_counter()
                self._profiler['graph'] += end - start
                self._profiler_count += 1
            
        
        # Store results so that ASE can access them #
        self.results['energy'] = self._energies['total'].item() * Hartree
        if self.coords.grad is not None:
            self.results['forces'] = -self.coords.grad.cpu().numpy() * (Hartree / Bohr)
            self.results['stress'] = (
                torch.matmul(self.coords.grad.T, self.coords) / torch.det(self.box)
            ).cpu().detach().numpy() * (Hartree / Bohr**3)
            if self.box.grad is not None:
                self.results['stress'] = self.results['stress'] + ((
                    torch.matmul(self.box.grad.T, self.box)
                 ) / torch.det(self.box)).cpu().detach().numpy() * (Hartree / Bohr**3)
        
        if not self._minimization:
            self.step_counter += 1

        # rebuild_nblist = self._minimization or (self.step_counter % self._rebuild_nblist_interval == 0)
        # if rebuild_nblist:
        #     self.pairs = self._system.rebuild_nblist(self.coords, self.cutoff_verlet, self.box)
        # record graph
        if self.use_cuda_graph and ((self.step_counter == self._start_graph) or (self.step_counter > self._start_graph and rebuild_nblist)):
            if self._run_profile: torch.cuda.nvtx.range_push("CMM-setup stream")
            if not hasattr(self, '_graph_stream'):
                self._graph_stream = torch.cuda.Stream()
            self._cuda_graph = torch.cuda.CUDAGraph()
            self._energies = {}
            if self.step_counter == self._start_graph:
                new_coords = torch.zeros_like(self.coords, requires_grad=True)
                new_box = torch.zeros_like(self.box, requires_grad=True)
                with torch.no_grad():
                    new_coords.copy_(self.coords)
                    new_box.copy_(self.box)
                self.coords = new_coords
                self.box = new_box
            torch.cuda.synchronize()
            self._graph_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._graph_stream):
                self.coords.grad = None
                self.box.grad = None
                self._energies = self._system.getEnergy(self.coords, self.box, self.pairs)
                self._energies['total'].backward()
            torch.cuda.current_stream().wait_stream(self._graph_stream)

            self.coords.grad = None
            self.box.grad = None
            if self._run_profile: 
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push("CMM-build graph")
            with torch.cuda.graph(self._cuda_graph, stream=self._graph_stream):
                self._energies = self._system.getEnergy(self.coords, self.box, self.pairs)
                self._energies['total'].backward()
            if self._run_profile:
                torch.cuda.nvtx.range_pop()
    
    def _update_coords(self, new_coords: np.ndarray):
        with torch.no_grad():
            coords_tensor_cpu = torch.from_numpy(new_coords).to(
                dtype=self.coords.dtype, non_blocking=True).pin_memory()
            self.coords.copy_(coords_tensor_cpu, non_blocking=True)
    
    def _update_box(self, new_box: np.ndarray):
        with torch.no_grad():
            box_tensor_cpu = torch.from_numpy(new_box).to(
                dtype=self.coords.dtype, non_blocking=True).pin_memory()
            self.box.copy_(box_tensor_cpu, non_blocking=True)

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
        
        self._update_coords(self.atoms.get_positions() / Bohr)
        self._update_box(self.atoms.get_cell().array / Bohr)
        
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
        # Note: The ideal gas part is added internally by ASE.
        # By turning off all interaction terms, I have validated that
        # we reproduce the volume predicted by the ideal gas law for
        # particular choices of N,P, and T.
        self.calculate(self.atoms, properties=['stress'])
        stress = self.results['stress']
        if voigt:
            stress = full_3x3_to_voigt_6_stress(stress)
        return stress
    
    def get_dipole_moment(self, include_induced_moments: bool = True):
        dipole_moment = self._ff.get_dipole_moment(self.coords, include_induced_moments=include_induced_moments)
        return dipole_moment.cpu().detach().numpy()
    
    def _get_configuration_hash(self, atoms):
        """Generate a hash that uniquely identifies the atomic configuration"""
        positions_hash = hash(np.array2string(atoms.positions, precision=10))
        if atoms.cell is not None and np.any(atoms.cell != 0.0):
            cell_hash = hash(np.array2string(atoms.cell, precision=10))
            return hash((positions_hash, cell_hash))
        else:
            return positions_hash