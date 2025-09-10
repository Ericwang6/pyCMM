import os
from pathlib import Path
from typing import Dict
import networkx as nx

import torch
import numpy as np
import openmm.app as app

from ..bonded import computeBondFromVecs, computeAngleFromVecs
from .qchem import QChemReader
from .xyz import XYZReader
from ..units import BOHR2ANG
from ..multipole import computeSphericalQuadrupoles


AxisTypes = {
    "Z-then-X": 'ZThenX',
    "3-Fold": "ThreeFold",
    "Z-Bisect": "ZBisect",
    "Z-Only": "ZOnly",
}


# def extract_bond_angle_eq(f_pdb: os.PathLike, atype_def: Dict[str, str], xyz: os.PathLike = ""):
#     pdb = app.PDBFile(f_pdb)
    
#     msg = 'Not vaild atom type definitions for {}'.format(list(pdb.topology.residues())[0].name)
#     assert pdb.topology.getNumAtoms() == len(atype_def), msg
    
#     atom_types = []
#     for atom in pdb.topology.atoms():
#         if atom.name not in atype_def:
#             raise RuntimeError(msg + f": {atom.name} no corresponding type")
#         atom_types.append(atype_def[atom.name])

#     if not xyz:
#         coords = torch.from_numpy(pdb.getPositions(asNumpy=True)._value) * 10

#     suffix = Path(xyz).suffix
#     if suffix == '.out':
#         coords = torch.from_numpy(QChemReader.read_out(xyz).coords)
#     elif suffix == '.xyz':
#         coords = torch.from_numpy(XYZReader(xyz).read()[0].coords)

#     coords /= BOHR2ANG
    
#     topdata = TopologyData(pdb.topology)

#     # bonds
#     bonds = computeBondFromVecs(coords[topdata.bonds[:, 1]] - coords[topdata.bonds[:, 0]])
#     bond_types = {}
#     for index, (i, j) in enumerate(zip(topdata.bonds[:, 0], topdata.bonds[:, 1])):
#         if (atom_types[j], atom_types[i]) in bond_types:
#             btyp = (atom_types[j], atom_types[i])
#         else:
#             btyp = (atom_types[i], atom_types[j])
        
#         tmp = bond_types.get(btyp, [])
#         tmp.append(bonds[index])
#         bond_types[btyp] = tmp
    
#     for key in bond_types:
#         bond_types[key] = np.mean(bond_types[key])

#     # angles
#     angles = computeAngleFromVecs(
#         coords[topdata.angles[:, 0]] - coords[topdata.angles[:, 1]],
#         coords[topdata.angles[:, 2]] - coords[topdata.angles[:, 1]]
#     )

#     angle_types = {}
#     for index, (i, j, k) in enumerate(zip(topdata.angles[:, 0], topdata.angles[:, 1], topdata.angles[:, 2])):
#         if (atom_types[k], atom_types[j], atom_types[i]) in angle_types:
#             atyp = (atom_types[k], atom_types[j], atom_types[i])
#         else:
#             atyp = (atom_types[i], atom_types[j], atom_types[k])
        
#         tmp = angle_types.get(atyp, [])
#         tmp.append(angles[index])
#         angle_types[atyp] = tmp
    
#     for key in angle_types:
#         angle_types[key] = np.mean(angle_types[key])

#     return bond_types, angle_types



def extract_multipoles(f_pdb: os.PathLike, atype_def: Dict[str, str], poledit_out: os.PathLike):
    pdb = app.PDBFile(f_pdb)
    
    msg = 'Not vaild atom type definitions for {}'.format(list(pdb.topology.residues())[0].name)
    assert pdb.topology.getNumAtoms() == len(atype_def), msg
    
    atom_types = []
    for atom in pdb.topology.atoms():
        if atom.name not in atype_def:
            raise RuntimeError(msg + f": {atom.name} no corresponding type")
        atom_types.append(atype_def[atom.name])

    multipoles = []

    g = nx.Graph()
    for i in range(pdb.topology.getNumAtoms()):
        g.add_node(i)

    with open(poledit_out) as f:
        for line in f:

            if line.startswith(' Equivalent Atoms Assigned the Same Atom Type :'):
                f.readline()
                while True:
                    next_line = f.readline().strip()
                    if not next_line:
                        break
                    content = next_line.split()
                    i, j = int(content[1]), int(content[3])
                    g.add_edge(i-1, j-1)


            if line.startswith(' Final Atomic Multipole Moments after Regularization :'):
                f.readline()
                noneq = [list(subg)[0] for subg in nx.connected_components(g)]
                for i in range(pdb.topology.getNumAtoms()):
                    atom_line = f.readline()
                    f.readline()
                    frame_line = f.readline().strip().split()
                    axistype = frame_line[-4]
                    kz, kx, ky = tuple(map(int, frame_line[-3:]))
                    f.readline()
                    mono = float(f.readline().strip().split()[-1])
                    dipo = list(map(float, f.readline().strip().split()[-3:]))
                    qxx = float(f.readline().strip().split()[-1])
                    qxy, qyy = tuple(map(float, f.readline().strip().split()))
                    qxz, qyz, qzz = tuple(map(float, f.readline().strip().split()))
                    f.readline()
                    if i not in noneq:
                        continue
                    multipoles.append({
                        "type": atom_types[i],
                        "axis_type": AxisTypes.get(axistype, axistype),
                        "z_atom": atom_types[kz - 1] if kz != 0 else "",
                        "x_atom": atom_types[kx - 1] if kx != 0 else "",
                        "y_atom": atom_types[ky - 1] if ky != 0 else "",
                        "mono": mono,
                        "dipo": dipo,
                        "quad_s": computeSphericalQuadrupoles(torch.tensor([[qxx, qxy, qxz, qyy, qyz, qzz]])).numpy(force=True)[0].tolist()
                    })
    
    return multipoles



