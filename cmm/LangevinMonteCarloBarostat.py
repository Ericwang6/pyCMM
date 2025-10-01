""" Monte Carlo Barostat + Langevin Integrator
Based on:

OpenMM implementation
https://github.com/openmm/openmm/blob/master/openmmapi/src/MonteCarloBarostatImpl.cpp

and 

R. Eppenga & D. Frenkel (1984) Monte Carlo study of the isotropic and nematic phases of infinitely thin hard platelets, Molecular Physics, 52:6, 1303-1334, DOI: 10.1080/00268978400101951
https://doi.org/10.1080/00268978400101951
"""

import numpy as np
from ase import units
from ase.md.langevin import Langevin
from ase.parallel import world


class NPTLangevinMonteCarloBarostat(Langevin):
    def __init__(
        self,
        atoms,
        timestep,
        pressure_au=1.01325 * units.bar,
        bsinterval=25,
        volume_scale=None,
        temperature=None,
        friction=None,
        fixcm=True,
        *,
        temperature_K=None,
        trajectory=None,
        logfile=None,
        loginterval=1,
        communicator=world,
        rng=None,
        append_trajectory=False
    ):
        super().__init__(
            atoms,
            timestep,
            temperature,
            friction,
            fixcm,
            temperature_K=temperature_K,
            trajectory=trajectory,
            logfile=logfile,
            loginterval=loginterval,
            communicator=communicator,
            rng=rng,
            append_trajectory=append_trajectory,
        )
        self.pressure = pressure_au
        # scales random move
        self.volume_scale = atoms.get_volume() * 0.01 if volume_scale is None else volume_scale  
        self.bsinterval = bsinterval  # monte carlo move interval

        self.num_attempted = 0
        self.num_accepted = 0

    def step(self, forces=None):
        # Langevin integrator step
        forces = super().step(forces)
        
        if self.get_number_of_steps() == 0:
            return forces

        # Monte Carlo Barostat
        if self.get_number_of_steps() % self.bsinterval == 0:
            natoms = len(self.atoms)

            # get current
            old_cell = self.atoms.get_cell()
            old_volume = self.atoms.get_volume()
            old_energy = self.atoms.get_potential_energy()

            # propose a change of volume
            dV = self.volume_scale * self.rng.uniform(-1, 1)
            new_volume = old_volume + dV

            # scale box and get new energy
            length_scale = np.power(new_volume / old_volume, 1.0 / 3.0)
            new_cell = length_scale * old_cell
            self.atoms.set_cell(new_cell, scale_atoms=True)
            new_energy = self.atoms.get_potential_energy()

            # accept or reject
            dE = new_energy - old_energy
            pdV = self.pressure * dV #* 1e30 * units.J
            kT = self.temp  # kT
            w = dE + pdV - natoms * kT * np.log(new_volume / old_volume)
            if (w > 0) and (self.rng.uniform() > np.exp(-w / kT)):
                # reject
                self.atoms.set_cell(old_cell, scale_atoms=True)
            else:
                # accept
                self.num_accepted += 1
            self.num_attempted += 1

            # move rescaling (for parity with OpenMM MonteCarloBarostat)
            if self.num_attempted >= 10:
                if self.num_accepted < 0.25 * self.num_attempted:
                    self.volume_scale /= 1.1
                    self.num_attempted = 0
                    self.num_accepted = 0
                elif self.num_accepted > 0.75 * self.num_attempted:
                    self.volume_scale = min(self.volume_scale * 1.1, old_volume * 0.3)
                    self.num_attempted = 0
                    self.num_accepted = 0
        
        return forces