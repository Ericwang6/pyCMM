import os
import torch
import openmm as mm
import openmm.unit as unit
import openmm.app as app

from cmm.ffxml import ForceFieldXML
from cmm.topology import Topology
from cmm.units import HARTREE2KJ, BOHR2NM

def test_methanol():
    with open(os.path.join(os.path.dirname(__file__), 'data/methanol_qforce.xml')) as f:
        xmlstr = f.read()
    system = mm.XmlSerializer.deserialize(xmlstr)

    names = []
    for idx in range(system.getNumForces()):
        force = system.getForce(idx)
        force.setForceGroup(idx)
        names.append(force.getName())
    
    pdb = app.PDBFile(os.path.join(os.path.dirname(__file__), 'data/methanol.pdb'))
    ctx = mm.Context(system, mm.LangevinIntegrator(300, 1, 0.5))
    ctx.setPositions(pdb.positions)
    ctx.reinitialize(preserveState=True)

    energies = {}
    for idx in range(system.getNumForces()):
        state = ctx.getState(getEnergy=True, groups={idx})
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        energies[names[idx]] = energy
    
    device = 'cpu'
    dtype = torch.float64
    torch.set_default_dtype(dtype)

    ff = ForceFieldXML(os.path.join(os.path.dirname(__file__), 'data/water_methanol.xml'), device=device)
    top = Topology.fromOpenmm(pdb.topology, device=device)
    coords = torch.tensor(pdb.getPositions(asNumpy=True)._value / BOHR2NM, device=device, dtype=dtype)
    system = ff.parametrize(top, use_fd_morse=True)
    energies_cmm = system.getEnergy(coords)

    for key in energies_cmm:
        val = energies_cmm[key]
        if torch.is_tensor(val):
            val = val.item()
        val *= HARTREE2KJ
        if val == 0.0:
            continue
        key_qforce = ''.join([x.replace('torsion', 'Dihedral').capitalize() for x in key.split('_')])
        if key_qforce == 'Dihedral':
            key_qforce = 'PeriodicDihedral'
        if key_qforce == 'Total':
            continue
        assert abs(val - energies[key_qforce]) < 0.002, f'{key_qforce} energy not same'

    print(energies)
    print(energies_cmm)


    