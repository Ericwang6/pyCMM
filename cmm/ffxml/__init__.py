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


def prettify_xml(xmlstr: str):
    pretxml = xml.dom.minidom.parseString(xmlstr)
    pretstr = pretxml.toprettyxml()
    return pretstr


class ItemTable(pd.DataFrame):

    def __init__(self, *args, **kwargs):
        infer_dtypes = kwargs.pop('infer_dtypes', True)
        name = kwargs.pop('name', '')
        assert name, 'Must provide a name'
        self.attrs['name'] = name
        super().__init__(*args, **kwargs)
        if infer_dtypes:
            self.convert_dtypes()

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
        if device is not None:
            device = device
        elif torch.cuda.is_available():
            device = 'cuda:0'
        elif torch.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

        data = {}
        for col in self.columns:
            if is_float_dtype(self[col]):
                data[col] = torch.tensor(self[col].tolist(), dtype=float_dtype, device=device, requires_grad=requires_grad)
            elif is_integer_dtype(self[col]):
                # integer always doesn't need grads
                data[col] = torch.tensor(self[col].tolist(), dtype=integer_dtype, device=device, requires_grad=False)
            else:
                data[col] = self[col].data
        return data

    @staticmethod
    def parseElement(element: ET.Element):
        tables = defaultdict(defaultdict(list))
        for item in element:
            name = f'{element.tag}/{item.tag}'
            for k, v in item.attrib.items():
                tables[name][k].append(v)
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
        
        if device is not None:
            device = device
        elif torch.cuda.is_available():
            device = 'cuda:0'
        elif torch.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        
        self.data: Dict[str, Dict[str, Dict[str, Union[torch.Tensor, List]]]] = {}

        for force_name in data.keys():
            self.data[force_name] = {}
            for item_name in self.data[force_name].keys():
                if not isinstance(self.data[force_name][item_name]):
                    table = ItemTable(self.data[force_name][item_name], name=item_name)
                else:
                    table = self.data[force_name][item_name]
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
            
    def find(self, path: str):
        tmp = tuple(path.split('/'))
        assert len(tmp) == 3, f'Invalid path: {path}'
        return self.data[tmp[0]][tmp[1]][tmp[2]]
    
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
    
    def to_xml_str(self, pretty=True):
        buffer = []
        for force_name in self.data.keys():
            buffer.append(f'<{force_name}>\n')
            for item_name in self.data[force_name].keys():
                table = ItemTable(self.data[force_name][item_name], name=f'{item_name}')
                buffer.append(table.to_xml_str())
            buffer.append(f'</{force_name}>\n')
        xmlstr = ''.join(buffer)
        if pretty:
            xmlstr = prettify_xml(xmlstr)
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
            data[force] = ItemTable.parseElement(force)
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

        self.loadAtomTypeDefs()
        self.loadParameterSet()
    
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
        self.atomTypeDefs = Dict[str, Dict[str, str]] = defaultdict(dict)
        for tree in self.trees:
            residues = tree.getroot().find("Residues")
            for res in residues.findall("Residue"):
                resname = res.get("name")
                assert resname not in self.atomTypeDefs, f"Multiple definitions for residue {resname}"
                for atom in res.findall("Atom"):
                    name = atom.get("name")
                    atype = atom.get('type')
                    self.atomTypeDefs[resname][name] = atype

    def exportAtomTypeDefs(self):
        residues = ET.Element(self.root, 'Residues')
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
        return self.pset.to_xml_str(False)
    
    def assignAtomTypes(self, top: Topology):
        for resname, name in top.atomSigs:
            top.atomTypes.append(self.atomTypeDefs[resname][name])
    
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