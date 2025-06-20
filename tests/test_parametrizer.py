import itertools
from cmm.topology import Topology
from cmm.ffxml.parametrizer import (
    BondParametrizer, AngleParametrizer, TorsionParametrizer,
    AngleAngleParametrizer, TorsionAngleParametrizer, TorsionBondParametrizer,
    MultipoleParametrizer, PairParametrizer
)
import torch


def methanol_top():
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
    top._atom_types = ['c.moh', 'o.moh', 'hc.moh', 'hc.moh', 'hc.moh', 'ho.moh']
    return top


def methanol_water_top():
    top = Topology('cpu')
    top.addAtom('C', 'MOH')
    top.addAtom('O', 'MOH')
    top.addAtom('H1', 'MOH')
    top.addAtom('H2', 'MOH')
    top.addAtom('H3', 'MOH')
    top.addAtom('HO', 'MOH')
    top.addAtom('O', 'HOH')
    top.addAtom('H1', 'HOH')
    top.addAtom('H2', 'HOH')
    top.addBond(0, 1)
    top.addBond(0, 2)
    top.addBond(0, 3)
    top.addBond(0, 4)
    top.addBond(1, 5)
    top.addBond(6, 7)
    top.addBond(6, 8)
    top.finalize()
    top._atom_types = ['c.moh', 'o.moh', 'hc.moh', 'hc.moh', 'hc.moh', 'ho.moh', 'ow', 'hw', 'hw']
    return top


def test_bond_parametrizer():
    parametrizer = BondParametrizer(
        types=[('c.moh', 'o.moh'), ('c.moh', 'hc.moh'), ('o.moh', 'ho.moh')],
        top=methanol_top(),
        name='Bond'
    )
    parametrizer.registerParameters('foo', torch.tensor([1.0, 2.0, 3.0]))
    parametrizer.expandParameters()
    assert torch.allclose(
        parametrizer.getExpandParameters('foo'),
        torch.tensor([1.0, 2.0, 2.0, 2.0, 3.0])
    )


def test_angle_parametrizer():
    parametrizer = AngleParametrizer(
        types=[('hc.moh', 'c.moh', 'hc.moh'), ('hc.moh', 'c.moh', 'o.moh'), ('c.moh', 'o.moh', 'ho.moh')],
        top=methanol_top(),
        name='Angle'
    )
    parametrizer.registerParameters('foo', torch.tensor([1.0, 2.0, 3.0]))
    parametrizer.expandParameters()
    assert torch.all(parametrizer.getExpandParameters('atomIndices') == torch.tensor([
        [2, 0, 1], [4, 0, 1], [3, 0, 1],
        [2, 0, 4],
        [0, 1, 5],
        [2, 0, 3], [3, 0, 4]]))
    assert torch.allclose(
        parametrizer.getExpandParameters('foo'), 
        torch.tensor([2., 2., 2., 1., 3., 1., 1.])
    )


def test_torsion_parametrizer():
    parametrizer = TorsionParametrizer(
        types=[('hc.moh', 'c.moh', 'o.moh', 'ho.moh')],
        top=methanol_top(),
        name='Torsion'
    )
    parametrizer.registerParameters('foo', torch.tensor([1.0]))
    parametrizer.expandParameters()
    assert torch.all(parametrizer.getExpandParameters('atomIndices') == torch.tensor([[4, 0, 1, 5],
        [3, 0, 1, 5],
        [2, 0, 1, 5]]))
    assert torch.allclose(parametrizer.getParameters('foo'), torch.tensor([1., 1., 1.]))


def test_angle_angle_parametrizer():
    parametrizer = AngleAngleParametrizer(
        types=[
            ('o.moh', 'c.moh', 'hc.moh', 'ho.moh', 'o.moh', 'c.moh', '2'), 
            ('hc.moh', 'c.moh', 'hc.moh', 'hc.moh', 'c.moh', 'hc.moh', '1'),
            ('o.moh', 'c.moh', 'hc.moh', 'o.moh', 'c.moh', 'hc.moh', '1'),
            ('o.moh', 'c.moh', 'hc.moh', 'hc.moh', 'c.moh', 'hc.moh', '1')
        ],
        top=methanol_top(),
        name='AngleAngle'
    )
    parametrizer.registerParameters('foo', torch.tensor([1.0, 2.0, 3.0, 4.0]))
    parametrizer.expandParameters()
    assert parametrizer.getExpandParameters('atomIndices').shape[0] == 15


def test_torsion_bond_parametrizer():
    parametrizer = TorsionBondParametrizer(
        types=[
            ('hc.moh', 'c.moh', 'o.moh', 'ho.moh', 'c.moh', 'o.moh', '1'), 
            ('hc.moh', 'c.moh', 'o.moh', 'ho.moh', 'c.moh', 'hc.moh', '2'), 
            ('hc.moh', 'c.moh', 'o.moh', 'ho.moh', 'o.moh', 'ho.moh', '2'), 
            ('hc.moh', 'c.moh', 'o.moh', 'ho.moh', 'c.moh', 'hc.moh', '3'), 
        ],
        top=methanol_top(),
        name='TorsionBond'
    )
    parametrizer.registerParameters('foo', torch.tensor([1.0, 2.0, 3.0, 4.0]))
    parametrizer.expandParameters()
    assert parametrizer.getExpandParameters('atomIndices').shape[0] == 15


def test_torsion_angle_parametrizer():
    parametrizer = TorsionAngleParametrizer(
        types=[
            ('hc.moh', 'c.moh', 'o.moh', 'ho.moh', 'hc.moh', 'c.moh', 'o.moh', '1'), 
            ('hc.moh', 'c.moh', 'o.moh', 'ho.moh', 'c.moh', 'o.moh', 'ho.moh', '1'), 
            ('hc.moh', 'c.moh', 'o.moh', 'ho.moh', 'o.moh', 'c.moh', 'hc.moh', '2'), 
            ('hc.moh', 'c.moh', 'o.moh', 'ho.moh', 'hc.moh', 'c.moh', 'hc.moh', '3'), 
        ],
        top=methanol_top(),
        name='TorsionBond'
    )
    parametrizer.registerParameters('foo', torch.tensor([1.0, 2.0, 3.0, 4.0]))
    parametrizer.expandParameters()
    assert parametrizer.getExpandParameters('atomIndices').shape[0] == 18


def test_pair_parametrizer():
    pairs = torch.tensor(list(itertools.product([0, 1, 2, 3, 4, 5], [6, 7, 8])))
    foo = torch.arange(1, 7, dtype=torch.float32)
    bar = torch.zeros(6, dtype=torch.float32)
    random = torch.randn(6, dtype=torch.float32)

    parametrizer = PairParametrizer(
        types=['ow', 'hw', 'c.moh', 'hc.moh', 'o.moh', 'ho.moh'],
        top=methanol_water_top(),
        name='Pair'
    )
    parametrizer.registerPairwiseParameters('foo', foo)
    parametrizer.registerPairwiseParameters(
        'bar', bar, 
        [('ow', 'hw'), ('o.moh', 'ho.moh'), ('ow', 'ho.moh'), ('o.moh', 'hw')], 
        torch.arange(1, 5, dtype=torch.float32)
    )
    parametrizer.registerParameters('random', random)
    parametrizer.expandParameters()

    assert torch.allclose(
        parametrizer.getExpandParameters('foo', pairs),
        parametrizer.combination_rule(foo[parametrizer.paramIndices[pairs][:, 0]], foo[parametrizer.paramIndices[pairs][:, 1]]))
    
    assert torch.allclose(
        parametrizer.getExpandParameters('bar', pairs),
        torch.tensor([0., 0., 0., 0., 4., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3., 0., 0.])
    )

    assert torch.allclose(
        parametrizer.getExpandParameters('random'),
        random[parametrizer.paramIndices]
    )


def test_multipole_parametrizer():
    parametrizer = MultipoleParametrizer(
        types=[
            ('c.moh', 'o.moh', 'ho.moh', ''),
            ('o.moh', 'c.moh', 'ho.moh', ''),
            ('hc.moh', 'c.moh', 'o.moh', ''),
            ('ho.moh', 'o.moh', 'c.moh', ''),
            ('ow', 'hw', 'hw', ''),
            ('hw', 'ow', 'hw', '')
        ],
        top=methanol_water_top(),
        name='Mutlipole'
    )
    data = {p: torch.randn(6) for p in ['mono', 'dx', 'dy', 'dz', 'q20', 'q21c', 'q21s', 'q22c', 'q22s']}
    for k, v in data.items():
        parametrizer.registerParameters(k, v)
    parametrizer.expandParameters()
    assert torch.all(parametrizer.getExpandParameters('paramIndices') == torch.tensor([0, 1, 2, 2, 2, 3, 4, 5, 5]))
    assert torch.all(parametrizer.getExpandParameters('kzIndices') == torch.tensor([1, 0, 0, 0, 0, 1, 7, 6, 6]))
    assert torch.all(parametrizer.getExpandParameters('kxIndices') == torch.tensor([5, 5, 1, 1, 1, 0, 8, 8, 7]))
    assert torch.all(parametrizer.getExpandParameters('kyIndices') == -1)
    assert torch.allclose(
        parametrizer.getExpandParameters('dipo'),
        torch.vstack((data['dx'], data['dy'], data['dz'])).T[parametrizer.paramIndices]
    )
    
    