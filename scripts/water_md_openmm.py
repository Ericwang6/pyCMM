import torch
torch.set_printoptions(precision=8)
import torch.nn as nn
import os
import numpy as np
from functools import partial


import openmm.app as app
import openmm as mm
from openmmtorch import TorchForce

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cmm.ffxml import ForceFieldXML
from cmm.topology import Topology
from cmm.units import BOHR2NM, BOHR2ANG, HARTREE2KJ


class CMMTorchForce(nn.Module):
    def __init__(self, ff_path: str, top: app.Topology, device: str, **kwargs):
        super().__init__()
        ff = ForceFieldXML(ff_path, device=device, float_dtype=torch.float64)
        cmm_top = Topology.fromOpenmm(top, device)
        self.cmm = ff.parametrize(cmm_top, **kwargs)
        self.bohr2nm = BOHR2NM
        self.hartree2kj = HARTREE2KJ
    
    def forward(self, positions: torch.Tensor, box: torch.Tensor):
        positions_bohr = positions / self.bohr2nm
        box_bohr = box / self.bohr2nm
        energies = self.cmm.getEnergy(positions_bohr, box_bohr)
        ene = energies['total'] * self.hartree2kj
        return ene


class MyMinReporter(mm.MinimizationReporter):
    def __init__(self, reportInterval=1):
        super().__init__()
        self.reportInterval = reportInterval

    def report(self, iteration, x, grad, args):
        """
        Called every iteration of the minimizer.
        - `iteration` is the current iteration count.
        - `x` are the flattened particle coordinates.
        - `grad` is the flattened gradient of the objective function.
        Return False to stop minimization early.
        """
        # You can also pull energies from the context if you have it in scope,
        # or compute stats from x/grad directly.
        print(f"Iter {iteration} {np.linalg.norm(np.array(grad).reshape(-1, 3), axis=1).max():.4f} {args['system energy']}")
        return False   # return False if you want to abort
        

if __name__ == '__main__':
    device = 'cuda'
    torch.set_default_dtype(torch.float64)

    ff_path = 'water.xml'
    pdb_path = 'water_216.pdb'

    pdb = app.PDBFile(pdb_path)
    model = CMMTorchForce(ff_path, pdb.topology, device, use_fd_morse=True, use_polarization=True)

    coords = torch.tensor(pdb.getPositions(asNumpy=True)._value, device=device, requires_grad=True)
    box = torch.tensor([[vec.x, vec.y, vec.z] for vec in pdb.topology.getPeriodicBoxVectors()], device=device, requires_grad=False)

    # energies = model(coords, box)

    # print(energies)
    # energies['pol'].backward()

    # np.savetxt('pol_grad', coords.grad.numpy(force=True))

    # model_jit = torch.jit.trace(model, example_inputs=(coords, box))
    model_jit = torch.jit.script(model.to(device))
    # ene = model_jit(coords, box)
    # ene.backward()
    # print(coords.grad/HARTREE2KJ*BOHR2NM)
    # exit(0)

    tforce = TorchForce(model_jit)
    tforce.setOutputsForces(False)
    tforce.setUsesPeriodicBoundaryConditions(True)

    system = mm.System()
    for atom in pdb.topology.atoms():
        system.addParticle(atom.element.mass)
    
    system.addForce(tforce)


    # system = app.ForceField('tip3p.xml').createSystem(pdb.topology, app.PME, nonbondedCutoff=0.7)

    integrator = mm.LangevinIntegrator(10, 1.0, 0.001)
    simulation = app.Simulation(
        pdb.topology, 
        system, 
        integrator, 
        mm.Platform.getPlatformByName('CUDA'), 
        {'CudaPrecision': 'double'}
    )
    simulation.context.setPositions(pdb.getPositions())
    simulation.context.setPeriodicBoxVectors(*pdb.getTopology().getPeriodicBoxVectors())

    state = simulation.context.getState(getEnergy=True, getForces=True)
    print(state.getPotentialEnergy())
    print(state.getForces(asNumpy=True)._value)

    simulation.minimizeEnergy(
        # maxIterations=1, 
        # tolerance=100
    )

    app.PDBFile.writeFile(pdb.topology, simulation.context.getState(getPositions=True).getPositions(), 'water_216_opt.pdb')
    print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
    print("Finished!")

    # simulation.reporters.append(app.StateDataReporter(sys.stdout, reportInterval=50, step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, density=True, speed=True))
    # simulation.context.setVelocitiesToTemperature(1000)
    # simulation.step(1000)





    
    
    