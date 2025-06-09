import os
from collections import defaultdict
from typing import List, Tuple

import networkx as nx
import torch
import openmm.app as app

    
class Topology:
    def __init__(
        self, 
        device: str | None = None,
        cutoff: int = 3
    ):
        
        if device is not None:
            device = device
        elif torch.cuda.is_available():
            device = 'cuda:0'
        elif torch.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        self.device = device
        self.cutoff = cutoff
        self.natoms = 0

        self._bonds: List[Tuple[int, int]] = []
        self._angles: List[Tuple[int, int, int]] = []
        self._dihedrals: List[Tuple[int, int, int, int]] = []

        self._atom_types: List[str] = []
        self._atom_sigs: List[Tuple[str, str]] = []
        self._graph = nx.Graph()
    
    @classmethod
    def fromOpenmm(cls, omm_top: app.Topology, device: str | None = None, cutoff: int = 3):
        top = cls(device, cutoff)
        for atom in omm_top.atoms():
            top.addAtom(atom.name, atom.residue.name)
        for bond in omm_top.bonds():
            top.addBond(bond.atom1.index, bond.atom2.index)
        top.finalize()
        return top
    
    @classmethod
    def fromPDB(cls, pdb_file: os.PathLike, device: str | None = None, cutoff: int = 3):
        return cls.fromOpenmm(app.PDBFile(pdb_file).topology, device, cutoff)

    def finalize(self):
        self._build()
        self._find_polarization_groups_and_scatter_indices()
    
    @property
    def nbonds(self) -> int:
        return len(self._bonds)
    
    @property
    def nangles(self) -> int:
        return len(self._angles)
    
    @property
    def ndihedrals(self) -> int:
        return len(self._dihedrals)

    @property
    def atomTypes(self) -> List[str]:
        return self._atom_types
    
    @property
    def atomSigs(self) -> List[Tuple[str, str]]:
        return self._atom_sigs
    
    def addAtom(self, name: str = '', residue: str = ''):
        self._atom_sigs.append((residue, name))
        self._graph.add_node(self.natoms)
        self.natoms += 1
    
    def addBond(self, atom1: int, atom2: int):
        assert atom1 < self.natoms, f'Atom {atom1} exceeds the number of atoms {self.natoms}'
        assert atom2 < self.natoms, f'Atom {atom2} exceeds the number of atoms {self.natoms}'
        assert atom1 != atom2, 'Two atoms are same'
        if atom1 < atom2:
            self._bonds.append((atom1, atom2))
        else:
            self._bonds.append((atom2, atom1))
        self._graph.add_edge(atom1, atom2)
        
    def getAnglesByAtom(self, idx: int) -> List[Tuple[int, int, int]]:
        return self._angles_by_atoms[idx]
    
    def getAnglesByBond(self, idx1: int, idx2: int) -> List[Tuple[int, int, int]]:
        angles1 = self._angles_by_atoms[idx1]
        angles2 = self._angles_by_atoms[idx2]
        results = [ang for ang in angles1 if ang in angles2]
        return results
    
    def getNeighborAtoms(self, idx: int) -> List[int]:
        return list(self._graph.neighbors(idx))
    
    def getBonds(self, asTensor: bool = True, transpose: bool = False):
        if asTensor:
            return self._bonds_tensor if not transpose else self._bonds_tensor.T.contiguous()
        else:
            return self._bonds if not transpose else self._bonds_tensor.T.detach().cpu().numpy().tolist()

    def getAngles(self, asTensor: bool = True, transpose: bool = False):
        if asTensor:
            return self._angles_tensor if not transpose else self._angles_tensor.T.contiguous()
        else:
            return self._angles if not transpose else self._angles_tensor.T.detach().cpu().numpy().tolist()
    
    def getDihedrals(self, asTensor: bool = True, transpose: bool = False):
        if asTensor:
            return self._dihedrals_tensor if not transpose else self._dihedrals_tensor.T.contiguous()
        else:
            return self._dihedrals if not transpose else self._dihedrals_tensor.T.detach().cpu().numpy().tolist()

    def _build(self):
        connect_data = {i+1: set() for i in range(self.cutoff)}
        connect_data[1] = set(self._bonds)
        
        for i in range(2, self.cutoff + 1):
            for path in connect_data[i - 1]:
                for nei in self.getNeighborAtoms(path[0]):
                    # nei in path - form a loop
                    # nei > pair[-1] - make sure the indices strictly incresing, i.e. path[-1] > path[0]
                    if nei in path or nei > path[-1]:
                        continue
                    connect_data[i].add(tuple([nei] + list(path)))
                for nei in self.getNeighborAtoms(path[-1]):
                    # nei in path - form a loop
                    # nei < pair[0] - make sure the indices strictly incresing, i.e. path[-1] > path[0]
                    if nei in path or nei < path[0]:
                        continue
                    connect_data[i].add(tuple(list(path) + [nei]))
        
        self._connect_data = connect_data
        self._angles = list(connect_data[2])
        self._dihedrals = list(connect_data[3])

        self._bonds_tensor = torch.tensor(self._bonds, device=self.device)
        self._angles_tensor = torch.tensor(self._angles, device=self.device)
        self._dihedrals_tensor = torch.tensor(self._dihedrals, device=self.device)
        
        self._angles_by_atoms = defaultdict(list)
        for angle in connect_data[2]:
            for i in angle:
                self._angles_by_atoms[i].append(angle)
        
        # Here we have to use `set` because in ring systems like cyclopropane, bonded atoms and angle atoms may overlap
        intra_pairs = set()
        for dist, paths in connect_data.items():
            for path in paths:
                intra_pairs.add(tuple(sorted((path[0], path[-1]))))
        
        self._intra_pairs = list(intra_pairs)
        self._intra_pairs_tensor = torch.tensor(self._intra_pairs, device=self.device)

    def _find_polarization_groups_and_scatter_indices(self):
        """
        Find polarization groups. A polarization group is a whole molecule (e.g. water, ions) whose
        total net charge needed to be constrained when calculating the induced charge

        The following attributes is set up in this function:

        n_pol_groups: int
            Number of polarization groups
        polarization_groups: torch.nested_tensor
            Atom indices of each group
        pol_group_indices_a: torch.Tensor
            A flat view of polairzation groups, shape (natoms,)
        pol_group_lengths_g: torch.Tensor
            The number of atoms in each group, shape (n_pol_groups,)
        pol_group_segment_indices: torch.Tensor
            The offset of each group, i.e. the cumsum (or inclusive scan) of `pol_group_lengths_g`,
            shape (n_pol_groups+1,)
        """
        subgraphs = list(nx.connected_components(self._graph))
        
        groups = []
        for gidx, subgraph in enumerate(subgraphs):
            group = torch.tensor(list(sorted(list(subgraph))), device=self.device, requires_grad=False, dtype=torch.long)
            groups.append(group)
        
        self.polarization_groups = torch.nested.nested_tensor(
            groups,
            dtype=torch.long,
            device=self.device,
            requires_grad=False
        )
        self.pol_group_indices_a = torch.cat(self.polarization_groups.unbind())
        self.pol_group_lengths_g = torch.tensor(
            [g.size(0) for g in self.polarization_groups.unbind()], 
            dtype=torch.long, device=self.device
        )
        self.n_pol_groups = self.pol_group_lengths_g.size(0)
        self.pol_group_segment_indices = torch.zeros(self.n_pol_groups + 1, 
            dtype=torch.long, device=self.device, requires_grad=False
        )
        self.pol_group_segment_indices[1:] = torch.cumsum(self.pol_group_lengths_g, dim=0)