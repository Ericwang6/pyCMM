import os
import warnings
from typing import Dict, List, Any, Iterable, Optional, Union
from collections import defaultdict
import xml.dom.minidom
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype
import torch

from ..topology import Topology
from ..system import System
from ..batched_system import BatchedSystem
from ..multipole import AxisTypes
from .parametrizer import (
    BondParametrizer, 
    AngleAngleParametrizer, AngleParametrizer, 
    TorsionAngleParametrizer, TorsionBondParametrizer, TorsionParametrizer, 
    MultipoleParametrizer, PairParametrizer, PolarizationParametrizer
)


AxisTypesAsDict = {name: member.value for name, member in AxisTypes.__members__.items()}


def prettify_xml(xmlstr: str):
    pretxml = xml.dom.minidom.parseString(xmlstr)
    pretstr = pretxml.toprettyxml()
    pretstr = '\n'.join([x for x in pretstr.split('\n')[1:] if x.strip()])
    return pretstr


class ItemTable(pd.DataFrame):

    def __init__(self, *args, **kwargs):
        infer_dtypes = kwargs.pop('infer_dtypes', True)
        name = kwargs.pop('name', '')
        assert name, 'Must provide a name'
        super().__init__(*args, **kwargs)
        self.attrs['name'] = name
        if infer_dtypes:
            for col in self.columns:
                if col.startswith('type') or col.endswith('type') or col in ['kz', 'kx', 'ky']:
                    continue
                try:
                    self[col] = pd.to_numeric(self[col])
                except:
                    continue

    def to_xml_str(self, attr_cols=None, encoding='utf-8') -> str:
        if attr_cols is None:
            attr_cols = self.columns.tolist()

        xmlstr = super().to_xml(
            path_or_buffer=None,
            index=False,
            root_name="Root",
            row_name=self.attrs['name'],
            attr_cols=attr_cols,
            encoding=encoding,
            xml_declaration=False,
            pretty_print=True,
        )
        return '\n'.join(xmlstr.split('\n')[1:-1])
    
    def to_tensors(self, float_dtype=torch.float64, integer_dtype=torch.long, device=None, requires_grad=False):
        data = {}
        for col in self.columns:
            if is_float_dtype(self[col]):
                data[col] = torch.tensor(self[col].tolist(), dtype=float_dtype, device=device, requires_grad=requires_grad)
            elif is_integer_dtype(self[col]):
                # integer always doesn't need grads
                data[col] = torch.tensor(self[col].tolist(), dtype=integer_dtype, device=device, requires_grad=False)
            else:
                data[col] = self[col].tolist()
        return data

    @staticmethod
    def parseElement(element: ET.Element):
        tables = {}
        for item in element:
            name = item.tag
            table = tables.get(name, defaultdict(list))
            for k, v in item.attrib.items():
                table[k].append(v)
            tables[name] = table
        tables = {key: ItemTable(tables[key], name=key) for key in tables}
        return tables


class ParameterSet:
    def __init__(
        self, 
        data: Dict[str, Dict[str, Union[Dict[str, Iterable], ItemTable]]],
        float_dtype=torch.float64, 
        integer_dtype=torch.long, 
        device=None, 
        requires_grad=False
    ):
        """
        data: {"HarmonicBond": {"Bond": {"k": [1.0, 1.0], "b0": [1.0, 1.0]}}}
        """
        
        self.data: Dict[str, Dict[str, Dict[str, Union[torch.Tensor, List]]]] = {}

        for force_name in data.keys():
            self.data[force_name] = {}
            for item_name in data[force_name].keys():
                self.data[force_name][item_name] = {}
                if not isinstance(data[force_name][item_name], ItemTable):
                    table = ItemTable(data[force_name][item_name], name=item_name)
                else:
                    table = data[force_name][item_name]
                params = table.to_tensors(float_dtype, integer_dtype, device, requires_grad)
                for param_name, param in params.items():
                    self.data[force_name][item_name][param_name] = param
        
        self.float_dtype = float_dtype
        self.integer_dtype = integer_dtype
        self.device = device
        self.requires_grad = requires_grad
    
    def getData(self, asDataFrame=False, asNumpy=False):
        if asDataFrame:
            data = {}
            for force_name in self.data.keys():
                data[force_name] = {}
                for item_name in self.data[force_name].keys():
                    data[force_name][item_name] = ItemTable(self.data[force_name][item_name], name=item_name)
            return data
        
        if asNumpy:
            data = {}
            for force_name in self.data.keys():
                data[force_name] = {}
                for item_name in self.data[force_name].keys():
                    data[force_name][item_name] = {}
                    for param_name in self.data[force_name][item_name].keys():
                        if torch.is_tensor(self.data[force_name][item_name][param_name]):
                            data[force_name][item_name][param_name] = self.data[force_name][item_name][param_name].detach().cpu().numpy()
                        else:
                            data[force_name][item_name][param_name] = np.array(self.data[force_name][item_name][param_name])
            return data
        
        return self.data
            
    def find(self, path: str, default: Any = None):
        tmp = tuple(path.split('/'))
        assert len(tmp) == 3, f'Invalid path: {path}'
        if default is None:
            return self.data[tmp[0]][tmp[1]][tmp[2]]
        else:
            try:
                return self.data[tmp[0]][tmp[1]][tmp[2]]
            except:
                return default
    
    def findall(self, only_tensor=True, requires_grad=True):
        if requires_grad and (not only_tensor):
            warnings.warn('requires_grad is True but only_tensor is False. Will return all parameters.')
            only_tensor = False
            requires_grad = False

        for force_name in self.data.keys():
            for item_name in self.data[force_name].keys():
                for param_name in self.data[force_name].keys():
                    is_tensor = torch.is_tensor(self.data[force_name][item_name][param_name])
                    has_grad = is_tensor and self.data[force_name][item_name][param_name].requires_grad
                    if only_tensor and not is_tensor:
                        continue
                    if requires_grad and not has_grad:
                        continue
                    yield f'{force_name}/{item_name}/{param_name}', self.data[force_name][item_name][param_name]
    
    def to_xml_str(self):
        buffer = []
        for force_name in self.data.keys():
            buffer.append(f'<{force_name}>\n')
            for item_name in self.data[force_name].keys():
                data_cpu = {}
                for k, v in self.data[force_name][item_name].items():
                    if torch.is_tensor(v):
                        data_cpu[k] = v.detach().cpu().numpy()
                    else:
                        data_cpu[k] = v
                table = ItemTable(data_cpu, name=f'{item_name}')
                buffer.append(table.to_xml_str())
            buffer.append(f'</{force_name}>\n')
        xmlstr = ''.join(buffer)
        return xmlstr
    
    @classmethod
    def parseElement(
        cls, 
        element: ET.Element,
        float_dtype=torch.float64, 
        integer_dtype=torch.long, 
        device=None, 
        requires_grad=False
    ):
        data = {}
        for force in element:
            if force.tag in ['Residues', 'AtomTypes']:
                continue
            data[force.tag] = ItemTable.parseElement(force)
        return cls(data, float_dtype, integer_dtype, device, requires_grad)
    
    def __add__(self, other):
        assert self.device == other.device, f'Not same device: {self.device} vs {other.device}'
        assert self.float_dtype == other.float_dtype, f'Not same float dtype: {self.float_dtype} vs {other.float_dtype}'
        assert self.integer_dtype == other.integer_dtype, f'Not same integer dtype: {self.integer_dtype} vs {other.integer_dtype}'
        assert self.requires_grad == other.requires_grad, f'Not same requires_grad: {self.requires_grad} vs {other.requires_grad}'

        data_1 = self.getData(asNumpy=True)
        data_2 = self.getData(asNumpy=True)
        
        for force_name in data_2.keys():
            if force_name not in data_1:
                data_1[force_name] = data_2[force_name]
                continue
            for item_name in data_2[force_name].keys():
                if item_name not in data_1[force_name]:
                    data_1[force_name][item_name] = data_2[force_name][item_name]
                    continue
                for param_name in data_2[force_name][item_name].keys():
                    data_1[force_name][item_name][param_name] = np.concatenate((
                        data_1[force_name][item_name][param_name],
                        data_2[force_name][item_name][param_name]
                    ))
        return ParameterSet(data_1, self.float_dtype, self.integer_dtype, self.device, self.requires_grad)


class ForceFieldXML:
    def __init__(
        self, *files, 
        float_dtype=torch.float64, 
        integer_dtype=torch.long, 
        device=None, 
        requires_grad=False
    ):
        self.files = self.processFileNames(files)
        self.trees = [ET.parse(f) for f in self.files]

        self.float_dtype = float_dtype
        self.integer_dtype = integer_dtype
        self.device = device
        self.requires_grad = requires_grad

        self.atomTypeDefs: Dict[str, Dict[str, str]] = defaultdict(dict)
        self.atomClassDefs: Dict[str, Dict[str, str]] = defaultdict(dict)
        self.pset: ParameterSet = None

        self.loadAtomTypeDefs()
        self.loadParameterSet()

        self._empty_tensor = torch.tensor([], device=device)
    
    def processFileNames(self, files):
        dirname = os.path.dirname(__file__)
        files = list(files) if isinstance(files, tuple) else [files]
        for i in range(len(files)):
            if not os.path.exists(files[i]):
                trial = os.path.join(dirname, files[i])
                if not os.path.isfile(trial):
                    raise FileNotFoundError()
                else:
                    files[i] = trial
        return files
    
    def loadAtomTypeDefs(self):
        for tree in self.trees:
            residues = tree.getroot().find("Residues")
            for res in residues.findall("Residue"):
                resname = res.get("name")
                assert resname not in self.atomTypeDefs, f"Multiple definitions for residue {resname}"
                for atom in res.findall("Atom"):
                    name = atom.get("name")
                    self.atomTypeDefs[resname][name] = atom.get('type')
                    self.atomClassDefs[resname][name] = atom.get('class')

    def exportAtomTypeDefs(self):
        residues = ET.Element('Residues')
        for resname in self.atomTypeDefs:
            residue = ET.SubElement(residues, 'Residue', {"name": resname})
            for name, atype in self.atomTypeDefs[resname].items():
                ele = ET.SubElement(residue, 'Atom', {"name": name, "type": atype})
        return ET.tostring(residues, encoding='unicode')
    
    def loadParameterSet(self):
        pset = ParameterSet.parseElement(
            self.trees[0].getroot(),
            self.float_dtype, self.integer_dtype,
            self.device, self.requires_grad
        )
        for tree in self.trees[1:]:
            pset = pset + ParameterSet.parseElement(
                tree.getroot(),
                self.float_dtype, self.integer_dtype,
                self.device, self.requires_grad
            )
        self.pset = pset
    
    def exportParameterSet(self):
        return self.pset.to_xml_str()
    
    def assignAtomTypes(self, top: Topology):
        top.atomTypes.clear()
        for resname, name in top.atomSigs:
            top.atomTypes.append(self.atomTypeDefs[resname][name])
            top.atomClasses.append(self.atomClassDefs[resname][name])
    
    def save(self, fname: os.PathLike = '') -> str:
        xmlstr = [
            '<ForceField>',
            self.exportAtomTypeDefs(),
            self.exportParameterSet(),
            '</ForceField>'
        ]
        xmlstr = prettify_xml(''.join(xmlstr))
        if fname:
            with open(fname, 'w') as f:
                f.write(xmlstr)
        return xmlstr
    
    def createParametrizers(self, top: Topology):
        parametrizers = {}

        # assign atom types
        self.assignAtomTypes(top)
        
        # bond
        bondTypes = list(zip(self.pset.find('Bonds/Bond/type1', []), self.pset.find('Bonds/Bond/type2', [])))
        use_aclass = False
        if len(bondTypes) == 0:
            bondTypes = list(zip(self.pset.find('Bonds/Bond/class1', []), self.pset.find('Bonds/Bond/class2', [])))
            use_aclass = len(bondTypes) > 0
        bondParametrizer = BondParametrizer(types=bondTypes, top=top, name='Bond', use_atom_class=use_aclass)
        bondParams = [
            'r_eq', 'D', 'k_b', 'j_cf', 'j_cf_pauli', 'k_hardness_b', 
            'dip_deriv_1', 'dip_deriv_2', 'ct_slope_1', 'ct_slope_2'
        ]
        for p in bondParams:
            bondParametrizer.registerParameters(p, self.pset.find(f'Bonds/Bond/{p}', self._empty_tensor))
        parametrizers['Bond'] = bondParametrizer
        
        # angle
        angleTypes = list(zip(
            self.pset.find('Angles/Angle/type1', []), 
            self.pset.find('Angles/Angle/type2', []),
            self.pset.find('Angles/Angle/type3', [])
        ))
        use_aclass = False
        if len(angleTypes) == 0:
            angleTypes = list(zip(
                self.pset.find('Angles/Angle/class1', []), 
                self.pset.find('Angles/Angle/class2', []),
                self.pset.find('Angles/Angle/class3', [])
            ))
            use_aclass = len(angleTypes) > 0
        angleParametrizer = AngleParametrizer(angleTypes, top, 'Angle', use_atom_class=use_aclass)
        angleParams = [
            'theta_eq', 'k_theta', 'r_eq_1', 'r_eq_2', 'k_bb', 'k_ba_1', 'k_ba_2',
            'j_cf_angle', 'k_hardness_angle', 'j_cf_bb', 'k_hardness_bb'
        ]
        for p in angleParams:
            angleParametrizer.registerParameters(p, self.pset.find(f'Angles/Angle/{p}', self._empty_tensor))
        parametrizers['Angle'] = angleParametrizer
        
        # torsion
        torsionTypes = list(zip(
            self.pset.find('Torsions/Torsion/type1', []), self.pset.find('Torsions/Torsion/type2', []),
            self.pset.find('Torsions/Torsion/type3', []), self.pset.find('Torsions/Torsion/type4', [])
        ))
        use_aclass = False
        if len(torsionTypes) == 0:
            torsionTypes = list(zip(
                self.pset.find('Torsions/Torsion/class1', []), self.pset.find('Torsions/Torsion/class2', []),
                self.pset.find('Torsions/Torsion/class3', []), self.pset.find('Torsions/Torsion/class4', [])
            ))
            use_aclass = len(torsionTypes) > 0
        torsionParametrizer = TorsionParametrizer(torsionTypes, top, 'Torsion', use_atom_class=use_aclass)
        torsionParams = [
            'per1', 'phase1', 'k1', 'per2', 'phase2', 'k2',
            'per3', 'phase3', 'k3', 'per4', 'phase4', 'k4',
            'theta_eq_1', 'theta_eq_2', 
            'k_taa_1', 'k_taa_2', 'k_taa_3', 'k_taa_4',
        ]
        for p in torsionParams:
            torsionParametrizer.registerParameters(p, self.pset.find(f'Torsions/Torsion/{p}', self._empty_tensor))
        parametrizers['Torsion'] = torsionParametrizer
        
        # angle-angle
        aaTypes = list(zip(
            self.pset.find('AngleAngleCoupling/AngleAngle/type1', []), self.pset.find('AngleAngleCoupling/AngleAngle/type2', []),
            self.pset.find('AngleAngleCoupling/AngleAngle/type3', []), self.pset.find('AngleAngleCoupling/AngleAngle/type4', []), 
            self.pset.find('AngleAngleCoupling/AngleAngle/type5', []), self.pset.find('AngleAngleCoupling/AngleAngle/type6', []),
            self.pset.find('AngleAngleCoupling/AngleAngle/ctype', []) # coupling type
        ))
        use_aclass = False
        if len(aaTypes) == 0:
            aaTypes = list(zip(
                self.pset.find('AngleAngleCoupling/AngleAngle/class1', []), self.pset.find('AngleAngleCoupling/AngleAngle/class2', []),
                self.pset.find('AngleAngleCoupling/AngleAngle/class3', []), self.pset.find('AngleAngleCoupling/AngleAngle/class4', []), 
                self.pset.find('AngleAngleCoupling/AngleAngle/class5', []), self.pset.find('AngleAngleCoupling/AngleAngle/class6', []),
                self.pset.find('AngleAngleCoupling/AngleAngle/ctype', []) # coupling type
            ))
            use_aclass = len(aaTypes) > 0
        aaParametrizer = AngleAngleParametrizer(aaTypes, top, 'AngleAngle', use_atom_class=use_aclass)
        for p in ['theta_eq_1', 'theta_eq_2', 'k_aa']:
            aaParametrizer.registerParameters(p, self.pset.find(f'AngleAngleCoupling/AngleAngle/{p}', self._empty_tensor))
        parametrizers['AngleAngle'] = aaParametrizer
        
        # torsion-bond
        tbTypes = list(zip(
            self.pset.find('TorsionBondCoupling/TorsionBond/type1', []), 
            self.pset.find('TorsionBondCoupling/TorsionBond/type2', []),
            self.pset.find('TorsionBondCoupling/TorsionBond/type3', []), 
            self.pset.find('TorsionBondCoupling/TorsionBond/type4', []), 
            self.pset.find('TorsionBondCoupling/TorsionBond/type5', []), 
            self.pset.find('TorsionBondCoupling/TorsionBond/type6', []),
            self.pset.find('TorsionBondCoupling/TorsionBond/ctype', []) # coupling type
        ))
        use_aclass = False
        if len(tbTypes) == 0:
            tbTypes = list(zip(
                self.pset.find('TorsionBondCoupling/TorsionBond/class1', []), 
                self.pset.find('TorsionBondCoupling/TorsionBond/class2', []),
                self.pset.find('TorsionBondCoupling/TorsionBond/class3', []), 
                self.pset.find('TorsionBondCoupling/TorsionBond/class4', []), 
                self.pset.find('TorsionBondCoupling/TorsionBond/class5', []), 
                self.pset.find('TorsionBondCoupling/TorsionBond/class6', []),
                self.pset.find('TorsionBondCoupling/TorsionBond/ctype', []) # coupling type
            ))
            use_aclass = len(tbTypes) > 0
        tbParametrizer = TorsionBondParametrizer(tbTypes, top, 'TorsionBond', use_atom_class=use_aclass)
        tbParams = [
            'per1', 'phase1', 'per2', 'phase2',
            'per3', 'phase3', 'per4', 'phase4',
            'r_eq', 'k_tb_1', 'k_tb_2', 'k_tb_3', 'k_tb_4',
        ]
        for p in tbParams:
            tbParametrizer.registerParameters(p, self.pset.find(f'TorsionBondCoupling/TorsionBond/{p}', self._empty_tensor))
        parametrizers['TorsionBond'] = tbParametrizer
        
        # torsion-angle
        taTypes = list(zip(
            self.pset.find('TorsionAngleCoupling/TorsionAngle/type1', []), 
            self.pset.find('TorsionAngleCoupling/TorsionAngle/type2', []),
            self.pset.find('TorsionAngleCoupling/TorsionAngle/type3', []), 
            self.pset.find('TorsionAngleCoupling/TorsionAngle/type4', []), 
            self.pset.find('TorsionAngleCoupling/TorsionAngle/type5', []), 
            self.pset.find('TorsionAngleCoupling/TorsionAngle/type6', []),
            self.pset.find('TorsionAngleCoupling/TorsionAngle/type7', []),
            self.pset.find('TorsionAngleCoupling/TorsionAngle/ctype', []) # coupling type
        ))
        use_aclass = False
        if len(taTypes) == 0:
            taTypes = list(zip(
                self.pset.find('TorsionAngleCoupling/TorsionAngle/class1', []), 
                self.pset.find('TorsionAngleCoupling/TorsionAngle/class2', []),
                self.pset.find('TorsionAngleCoupling/TorsionAngle/class3', []), 
                self.pset.find('TorsionAngleCoupling/TorsionAngle/class4', []), 
                self.pset.find('TorsionAngleCoupling/TorsionAngle/class5', []), 
                self.pset.find('TorsionAngleCoupling/TorsionAngle/class6', []),
                self.pset.find('TorsionAngleCoupling/TorsionAngle/class7', []),
                self.pset.find('TorsionAngleCoupling/TorsionAngle/ctype', []) # coupling type
            ))
            use_aclass = len(taTypes) > 0
        taParametrizer = TorsionAngleParametrizer(taTypes, top, 'TorsionAngle', use_atom_class=use_aclass)
        taParams = [
            'per1', 'phase1', 'per2', 'phase2',
            'per3', 'phase3', 'per4', 'phase4',
            'theta_eq', 'k_ta_1', 'k_ta_2', 'k_ta_3', 'k_ta_4',
        ]
        for p in taParams:
            taParametrizer.registerParameters(p, self.pset.find(f'TorsionAngleCoupling/TorsionAngle/{p}', self._empty_tensor))
        parametrizers['TorsionAngle'] = taParametrizer
        
        # multipoles
        try:
            mpoleTypes = list(zip(
                self.pset.find('Multipoles/Multipole/type'),
                self.pset.find('Multipoles/Multipole/kz'),
                self.pset.find('Multipoles/Multipole/kx'),
                self.pset.find('Multipoles/Multipole/ky'),
            ))
            use_aclass = False
        except:
            mpoleTypes = list(zip(
                self.pset.find('Multipoles/Multipole/class'),
                self.pset.find('Multipoles/Multipole/kz'),
                self.pset.find('Multipoles/Multipole/kx'),
                self.pset.find('Multipoles/Multipole/ky'),
            ))
            use_aclass = True
        mpoleParametrizer = MultipoleParametrizer(mpoleTypes, top, 'Multipole', use_atom_class=use_aclass)
        mpoleParametrizer.registerParameters(
            'axistype',
            torch.tensor([AxisTypesAsDict[t] for t in self.pset.find('Multipoles/Multipole/axistype')], device=top.device)
        )
        mpoleParametrizer.registerParameters('mono', self.pset.find('Multipoles/Multipole/c0'))
        for p in ['dx', 'dy', 'dz', 'q20', 'q21c', 'q21s', 'q22c', 'q22s']:
            mpoleParametrizer.registerParameters(p, self.pset.find(f'Multipoles/Multipole/{p}'))
        parametrizers['Multipoles'] = mpoleParametrizer
        
        # charge penetration
        try:
            cpTypes = self.pset.find("ChargePenetration/CP/type")
            use_aclass = False
        except:
            cpTypes = self.pset.find("ChargePenetration/CP/class")
            use_aclass = True
        keyword = 'class' if use_aclass else 'type'
        cpParametrizer = PairParametrizer(cpTypes, top, name='ChargePenetration', use_atom_class=use_aclass)
        cpParametrizer.registerParameters('Z', self.pset.find('ChargePenetration/CP/Z'))
        cpParametrizer.registerPairwiseParameters(
            'b_elec', 
            self.pset.find('ChargePenetration/CP/b_elec'),
            specific_pair_types=list(zip(
                self.pset.find(f'ChargePenetration/Pair/{keyword}1', list()),
                self.pset.find(f'ChargePenetration/Pair/{keyword}2', list())
            )),
            specific_pair_params=self.pset.find('ChargePenetration/Pair/b_elec', list())
        )
        parametrizers['ChargePenetration'] = cpParametrizer

        # Pauli
        try:
            pauliTypes = self.pset.find('PauliRepulsion/Pauli/type')
            use_aclass = False
        except:
            pauliTypes = self.pset.find('PauliRepulsion/Pauli/class')
            use_aclass = True
        keyword = 'class' if use_aclass else 'type'
        pauliParametrizer = PairParametrizer(pauliTypes, top, name='Pauli', use_atom_class=use_aclass)
        for p in ['q_pauli', 'Kdipo_pauli', 'Kquad_pauli']:
            pauliParametrizer.registerParameters(p, self.pset.find(f'PauliRepulsion/Pauli/{p}'))
        pauliParametrizer.registerPairwiseParameters(
            'b_pauli', 
            self.pset.find('PauliRepulsion/Pauli/b_pauli'),
            specific_pair_types=list(zip(
                self.pset.find(f'PauliRepulsion/Pair/{keyword}1', list()),
                self.pset.find(f'PauliRepulsion/Pair/{keyword}2', list())
            )),
            specific_pair_params=self.pset.find('PauliRepulsion/Pair/b_pauli', list())
        )
        parametrizers['Pauli'] = pauliParametrizer

        # Exchange-Pol
        try:
            xpolTypes = self.pset.find('ExchangePolarization/Xpol/type')
            use_aclass = False
        except:
            xpolTypes = self.pset.find('ExchangePolarization/Xpol/class')
            use_aclass = True
        keyword = 'class' if use_aclass else 'type'
        xpolParametrizer = PairParametrizer(xpolTypes, top, name='ExchangePolarization', use_atom_class=use_aclass)
        for p in ['q_xpol', 'Kdipo_xpol', 'Kquad_xpol']:
            xpolParametrizer.registerParameters(p, self.pset.find(f'ExchangePolarization/Xpol/{p}'))
        xpolParametrizer.registerPairwiseParameters(
            'b_xpol', 
            self.pset.find('ExchangePolarization/Xpol/b_xpol'),
            specific_pair_types=list(zip(
                self.pset.find(f'ExchangePolarization/Pair/{keyword}1', list()),
                self.pset.find(f'ExchangePolarization/Pair/{keyword}2', list())
            )),
            specific_pair_params=self.pset.find('ExchangePolarization/Pair/b_xpol', list())
        )
        parametrizers['ExchangePolarization'] = xpolParametrizer

        # Dispersion
        try:
            dispTypes = self.pset.find("Dispersion/Disp/type")
            use_aclass = False
        except KeyError:
            dispTypes = self.pset.find("Dispersion/Disp/class")
            use_aclass = True
        
        dispParametrizer = PairParametrizer(dispTypes, top, name='Dispersion', use_atom_class=use_aclass)
        keyword = 'class' if use_aclass else 'type'
        for p in ['C6_disp', 'b_disp']:
            pair_params = self.pset.find(f'Dispersion/Pair/{p}', list())
            dispParametrizer.registerPairwiseParameters(
                p,
                self.pset.find(f'Dispersion/Disp/{p}'),
                specific_pair_types=list(zip(
                    self.pset.find(f'Dispersion/Pair/{keyword}1', list()),
                    self.pset.find(f'Dispersion/Pair/{keyword}2', list())
                )),
                specific_pair_params=pair_params
            )
        parametrizers['Dispersion'] = dispParametrizer
        
        # ChargeTransfer
        try:
            ctTypes = self.pset.find("ChargeTransfer/Direct/type")
            use_aclass = False
        except:
            ctTypes = self.pset.find("ChargeTransfer/Direct/class")
            use_aclass = True
        keyword = 'class' if use_aclass else 'type'
        ctParameterizer = PairParametrizer(ctTypes, top, name='ChargeTransfer', use_atom_class=use_aclass)
        for p in ['q_ct_acc', 'q_ct_don', 'Kdipo_ct_acc', 'Kdipo_ct_don', 'Kquad_ct_acc', 'Kquad_ct_don']:
            ctParameterizer.registerParameters(p, self.pset.find(f"ChargeTransfer/Direct/{p}"))
        
        ctParameterizer.registerPairwiseParameters(
            "b_ct",
            self.pset.find("ChargeTransfer/Direct/b_ct"),
            specific_pair_types=list(zip(
                self.pset.find(f'ChargeTransfer/Pair/{keyword}1', list()),
                self.pset.find(f'ChargeTransfer/Pair/{keyword}2', list())
            )),
            specific_pair_params=self.pset.find('ChargeTransfer/Pair/b_ct', list())
        )
        ctParameterizer.registerPairwiseParameters(
            "eps_ct",
            torch.zeros_like(ctParameterizer.getParameters('q_ct_acc')),
            specific_pair_types=list(zip(
                self.pset.find(f'ChargeTransfer/Indirect/{keyword}1'),
                self.pset.find(f'ChargeTransfer/Indirect/{keyword}2')
            )),
            specific_pair_params=self.pset.find('ChargeTransfer/Indirect/eps_ct')
        )
        parametrizers['ChargeTransfer'] = ctParameterizer

        # Polarization
        try:
            polTypes = self.pset.find("Polarization/Pol/type")
            use_aclass = False
        except KeyError:
            polTypes = self.pset.find("Polarization/Pol/class")
            use_aclass = True
        keyword = 'class' if use_aclass else 'type'
        polParametrizer = PolarizationParametrizer(polTypes, top, name='Polarization', use_atom_class=use_aclass)
        for p in ['eta', 'alpha_xx', 'alpha_yy', 'alpha_zz', 'alpha_damp_exponent', 'alpha_damp_max']:
            polParametrizer.registerParameters(p, self.pset.find(f'Polarization/Pol/{p}'))
        parametrizers['Polarization'] = polParametrizer
        
        return parametrizers
    
    def parametrize(self, top: Topology, batch: bool = False, **kwargs):
        parametrizers = self.createParametrizers(top)
        if batch:
            system = BatchedSystem(top, parametrizers, **kwargs)
        else:
            system = System(top, parametrizers, **kwargs)
        return system