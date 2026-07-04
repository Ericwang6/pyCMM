import openmm.app as app
import openmm as mm
import numpy as np

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

    ff_path = 'water.xml'
    pdb_path = 'water_216.pdb'

    pdb = app.PDBFile(pdb_path)

    system = app.ForceField('tip3p.xml').createSystem(pdb.topology, app.PME, nonbondedCutoff=0.7, constraints=None, rigidWater=False)

    integrator = mm.LangevinIntegrator(298.15, 1.0, 0.001)
    simulation = app.Simulation(pdb.topology, system, integrator, mm.Platform.getPlatformByName('CUDA'))
    simulation.context.setPositions(pdb.getPositions())
    simulation.context.setPeriodicBoxVectors(*pdb.getTopology().getPeriodicBoxVectors())

    print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
    simulation.minimizeEnergy(reporter=MyMinReporter())

    app.PDBFile.writeFile(pdb.topology, simulation.context.getState(getPositions=True).getPositions(), 'water_216_opt.pdb')



    
    
    