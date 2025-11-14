import os
import numpy as np
import torch
import openmm.app as app
import xml.etree.ElementTree as ET

from cmm.ffxml import ItemTable, ParameterSet, ForceFieldXML
from cmm.topology import Topology
from cmm.units import BOHR2NM, HARTREE2KJ
from cmm.interfaces import CMMCalculator, Hartree

from pprint import pprint

def test_export_table_to_xml():
    table = ItemTable({
        "k": [1.0, 2.0],
        "b": [2.0, 3.0]
    }, name='Bond')
    ref = '''<Bond k="1.0" b="2.0"/>
    <Bond k="2.0" b="3.0"/>'''
    ref = ''.join(ref.split())
    string = ''.join(table.to_xml_str().split())
    assert ref == string


def test_parse_table_from_element():
    ref = '''<Bonds><Bond k="1.0" b="2.0"/><Bond k="2.0" b="3.0"/><Pair foo="a" bar="b"/><Pair foo="a" bar="b"/></Bonds>'''
    tables = ItemTable.parseElement(ET.fromstring(ref))
    assert np.allclose(tables['Bond'][['k', 'b']].values, [[1.0, 2.0], [2.0, 3.0]])
    assert tables['Pair'][['foo', 'bar']].values.tolist() == [['a', 'b'], ['a', 'b']]


def test_export_table_to_tensor():
    device = 'cpu'
    float_dtype = torch.float64
    ref = '''<Bonds><Bond k="1.0" b="2.0"/><Bond k="2.0" b="3.0"/><Pair foo="a" bar="1.0" int="2"/><Pair foo="b" bar="2.0" int="1"/></Bonds>'''
    tables = ItemTable.parseElement(ET.fromstring(ref))
    tensors = tables['Bond'].to_tensors(device=device, float_dtype=float_dtype)
    assert torch.allclose(tensors['k'], torch.tensor([1.0, 2.0], dtype=float_dtype))
    assert torch.allclose(tensors['b'], torch.tensor([2.0, 3.0], dtype=float_dtype))

    tensors2 = tables['Pair'].to_tensors(device=device, float_dtype=float_dtype)
    assert tensors2['foo'] == ['a', 'b']
    assert torch.allclose(tensors2['bar'], torch.tensor([1.0, 2.0], dtype=float_dtype))
    assert torch.allclose(tensors2['int'], torch.tensor([2, 1]))


def test_parse_paramset_from_element():
    device = 'cpu'
    float_dtype = torch.float64
    ref = '''<ForceField><Bonds><Bond k="1.0" b="2.0"/><Bond k="2.0" b="3.0"/><Pair foo="a" bar="1.0" int="2"/><Pair foo="b" bar="2.0" int="1"/></Bonds></ForceField>'''
    pset = ParameterSet.parseElement(ET.fromstring(ref), device=device, float_dtype=float_dtype)
    test = '<ForceField>' + pset.to_xml_str() + '</ForceField>'
    assert ''.join(ref.split()) == ''.join(test.split())


def test_parse_ffxml():
    fpath = os.path.join(os.path.dirname(__file__), 'data/water.xml')
    ff = ForceFieldXML(fpath)
    xmlstr = ff.save()


def test_parametrize():
    device = 'cpu'
    float_dtype = torch.float64
    torch.set_default_dtype(float_dtype)

    pdb = app.PDBFile(os.path.join(os.path.dirname(__file__), 'data/water_dimer_2.pdb'))
    coords = torch.tensor((pdb.getPositions(asNumpy=True)._value / BOHR2NM).tolist(), device=device, dtype=float_dtype)
    box = torch.tensor([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]], device=device, dtype=float_dtype)

    ff = ForceFieldXML(os.path.join(os.path.dirname(__file__), 'data/water.xml'), device=device, float_dtype=torch.float64)
    top = Topology.fromOpenmm(pdb.topology, device)
    system = ff.parametrize(top)


def test_parametrize_with_class():
    device = 'cpu'
    float_dtype = torch.float64
    torch.set_default_dtype(float_dtype)

    pdb = app.PDBFile(os.path.join(os.path.dirname(__file__), 'data/water_dimer_2.pdb'))
    coords = torch.tensor((pdb.getPositions(asNumpy=True)._value / BOHR2NM).tolist(), device=device, dtype=float_dtype)
    box = torch.tensor([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]], device=device, dtype=float_dtype)

    ff = ForceFieldXML(os.path.join(os.path.dirname(__file__), 'data/water_class.xml'), device=device, float_dtype=torch.float64)
    top = Topology.fromOpenmm(pdb.topology, device)
    system = ff.parametrize(top)


def test_ase_interface():
    device = 'cpu'
    float_dtype = torch.float64
    torch.set_default_dtype(float_dtype)

    pdb = app.PDBFile(os.path.join(os.path.dirname(__file__), 'data/water_216.pdb'))
    box = torch.tensor([[v.x/BOHR2NM, v.y/BOHR2NM, v.z/BOHR2NM] for v in pdb.topology.getPeriodicBoxVectors()], device=device, dtype=float_dtype)
    print(box)

    coords = torch.tensor((pdb.getPositions(asNumpy=True)._value / BOHR2NM).tolist(), device=device, dtype=float_dtype)

    ff = ForceFieldXML(os.path.join(os.path.dirname(__file__), 'data/water.xml'), device=device, float_dtype=torch.float64)
    top = Topology.fromOpenmm(pdb.topology, device)
    system = ff.parametrize(top)

    energies = system.getEnergy(coords, box)
    pprint(energies)
    
    # ASE
    calculator = CMMCalculator(system, top, coords, box)
    atoms = calculator.atoms
    calculator.calculate(atoms)
    
    assert np.allclose(energies['total'].item() * Hartree, calculator.results['energy'])


