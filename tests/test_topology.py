import os
from cmm.topology import Topology

def test_top_init():
    top = Topology(device='cpu')
    top.addAtom('O', 'HOH')
    top.addAtom('H1', 'HOH')
    top.addAtom('H2', 'HOH')
    top.addAtom('O', 'HOH')
    top.addAtom('H1', 'HOH')
    top.addAtom('NA', 'NA')
    top.addAtom('H2', 'HOH')
    top.addBond(0, 1)
    top.addBond(0, 2)
    top.addBond(3, 4)
    top.addBond(3, 6)
    top.finalize()

    assert top.natoms == 7
    assert len(top.polarization_groups.unbind()) == 3
    assert top.getBonds(asTensor=True).shape == (4, 2)
    assert top.getAngles(asTensor=True).shape == (2, 3)
    assert top.pol_group_indices_a.shape == (7,)
    assert top.pol_group_lengths_g.shape == (3,)
    assert top.n_pol_groups == 3
    assert top.pol_group_segment_indices.shape == (4,)


def test_read_pdb():
    top = Topology.fromPDB(os.path.join(os.path.dirname(__file__), 'data/water_dimer_2.pdb'))
    assert top.natoms == 6
    assert len(top.getBonds(asTensor=False)) == 4


def test_top_functions():
    top = Topology('cpu')
    top.addAtom('C', 'MOH')
    top.addAtom('O', 'MOH')
    top.addAtom('H1', 'MOH')
    top.addAtom('H2', 'MOH')
    top.addAtom('H3', 'MOH')
    top.addAtom('HO', 'MOH')
    top.addBond(0, 1)
    top.addBond(0, 2)
    top.addBond(0, 3)
    top.addBond(0, 4)
    top.addBond(1, 5)
    top.finalize()

    assert len(top.getAnglesByAtom(0)) == len(top.getAngles(asTensor=False))
    assert len(top.getAnglesByBond(0, 1)) == 4
    assert tuple(sorted(top.getNeighborAtoms(0))) == (1, 2, 3, 4)