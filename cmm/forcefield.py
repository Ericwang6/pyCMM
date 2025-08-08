import os
from pathlib import Path
import json
import toml
from typing import Dict, Any, Tuple
from collections import defaultdict
import itertools
import torch
from torch_scatter import scatter
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
import openmm.app as app

from .multipole import (
    AxisTypes,
    computeCartesianQuadrupoles, 
    computeLocal2GlobalRotationMatrixBatch,
    rotateMultipoles,
    rotateQuadrupoles
)
from .bonded import (
    computeChargeFluxBond, computeChargeFluxAngle, computeChargeFluxBondBond,
    computeBondFromVecs, computeAngleFromVecs
)
from .short_range import computeShortRangeEnergyFromPairs, scaleMultipoles, computePairwiseChargeTransfer
from .dispersion import computeDispersionFromPairs
from .electrostatics import computePermElecAndPolarizationEnergy, computePermanentElectricPotentialExpansion
from .units import HARTREE2KCAL


class TopologyData():
    def __init__(self, top: Any, max_connect: int = 5):
        self.max_connect = max_connect
        if isinstance(top, app.Topology):
            self.build_from_openmm_topology(top)
        else:
            raise NotImplementedError(f'Unsupported type: {type(top)}')
    
    @property
    def bonds(self):
        return torch.tensor([list(x) for x in self.connect_data[1]])
    
    @property
    def angles(self):
        return torch.tensor([list(x) for x in self.connect_data[2]])
    
    @property
    def dihedrals(self):
        return torch.tensor([list(x) for x in self.connect_data[3]])

    def build_from_openmm_topology(self, top: app.Topology):
        neighbors = defaultdict(list)
        connect_data = {i+1: set() for i in range(self.max_connect)}

        for bond in top.bonds():
            if bond.atom1.index < bond.atom2.index:
                atom1, atom2 = bond.atom1.index, bond.atom2.index
            else:
                atom2, atom1 = bond.atom1.index, bond.atom2.index
            connect_data[1].add((atom1, atom2))
            neighbors[atom1].append(atom2)
            neighbors[atom2].append(atom1)
        self.neighbors = neighbors
        
        for i in range(2, self.max_connect + 1):
            for path in connect_data[i - 1]:
                for nei in neighbors[path[0]]:
                    # nei in path - form a loop
                    # nei > pair[-1] - make sure the indices strictly incresing, i.e. path[-1] > path[0]
                    if nei in path or nei > path[-1]:
                        continue
                    connect_data[i].add(tuple([nei] + list(path)))
                for nei in neighbors[path[-1]]:
                    # nei in path - form a loop
                    # nei < pair[0] - make sure the indices strictly incresing, i.e. path[-1] > path[0]
                    if nei in path or nei < path[0]:
                        continue
                    connect_data[i].add(tuple(list(path) + [nei]))
        self.connect_data = connect_data
        # (atom_i, atom_j) -> bond_idx
        self.bond_indices_map = {x: i for i, x in enumerate(self.connect_data[1])}
        
        # build connectivity matrix, maybe useful in the future
        tmp = {}
        for dist in range(1, self.max_connect + 1):
            for path in connect_data[dist]:
                pair = (path[0], path[-1])
                tmp[pair] = tmp.get(pair, dist)
        
        atoms1, atoms2, dists = [], [], []
        for pair, dist in tmp.items():
            atoms1.append(pair[0])
            atoms2.append(pair[1])
            atoms1.append(pair[1])
            atoms2.append(pair[0])
            dists.append(dist)
            dists.append(dist)
        
        n_atoms = top.getNumAtoms()
        self.n_atoms = n_atoms
        self.connect_matrix = coo_matrix((dists, (atoms1, atoms2)), shape=(n_atoms, n_atoms), dtype=np.int8)
    
    def get_bond_index(self, atom_i: int, atom_j: int):
        if (atom_i, atom_j) in self.bond_indices_map:
            return self.bond_indices_map[(atom_i, atom_j)]
        elif (atom_j, atom_i) in self.bond_indices_map:
            return self.bond_indices_map[(atom_j, atom_i)]
        else:
            raise KeyError(f'Bond ({atom_i}, {atom_j}) not exist')
        
    def get_angles_as_bond_indices(self) -> torch.Tensor:
        '''
        Get angles as bond indices

        Return
        ------

        angles_as_bond_indices: torch.Tensor
            Tensor (2, n_angles). Each column consists two indices of the bonds that form the angle.
        '''
        angles_as_bond_indices = []
        for angle in self.connect_data[2]:
            bond_idx_1 = self.get_bond_index(angle[0], angle[1])
            bond_idx_2 = self.get_bond_index(angle[2], angle[1])
            angles_as_bond_indices.append([bond_idx_1, bond_idx_2])
        angles_as_bond_indices =  torch.tensor(angles_as_bond_indices, dtype=torch.long).T
        return angles_as_bond_indices
    
    def construct_nb_pairs(self):
        graph = nx.Graph()
        for atom in range(self.n_atoms):
            graph.add_node(atom)
        for bond in self.bonds:
            graph.add_edge(int(bond[0]), int(bond[1]))
        subgraphs = list(nx.connected_components(graph))
        pairs = []
        for i in range(len(subgraphs)):
            for j in range(i + 1, len(subgraphs)):
                subgraph_i, subgraph_j = subgraphs[i], subgraphs[j]
                for atom_i in subgraph_i:
                    for atom_j in subgraph_j:
                        # bi-direction
                        pairs.append([atom_i, atom_j])
                        pairs.append([atom_j, atom_i])
        pairs = torch.tensor(pairs, dtype=torch.long).T

        groups_indices = [None for _ in range(self.n_atoms)]
        groups = []
        for gidx, subgraph in enumerate(subgraphs):
            groups.append(sorted(list(subgraph)))
            for atom in subgraph:
                groups_indices[atom] = gidx
        groups_indices = torch.tensor(groups_indices, dtype=torch.long)
        return pairs, groups, groups_indices


class SystemNoCutoff:
    '''
    Only used for parametrization, don't use it for simulation
    '''
    def __init__(self, topdata: TopologyData, ff_param, param_expand_indices, **kwargs):
        self.topdata = topdata
        self.ff_param = ff_param
        self.param_expand_indices = param_expand_indices
        self.z_atoms = torch.tensor(kwargs['z_atoms'], dtype=torch.long)
        self.y_atoms = torch.tensor(kwargs['y_atoms'], dtype=torch.long)
        self.x_atoms = torch.tensor(kwargs['x_atoms'], dtype=torch.long)
        self.bond_pairs = kwargs['bond_pairs']
        self.angle_pairs = kwargs['angle_pairs']
        self.angle_as_bond_indices = self.topdata.get_angles_as_bond_indices()
        self.has_angles = self.angle_as_bond_indices.shape[0] > 0
        # NOTE: in condensed phase, this should be replaced by a NeighborList
        self.nbpairs, self.groups, self.group_indices = self.topdata.construct_nb_pairs()
    
    def update_ff_param(self, new_ff_param):
        self.ff_param = new_ff_param
        self.expand_params()

    def expand_params(self):
        self.param = {}
        num_atomic_params = len(self.ff_param['atomic_params']['type'])
        num_pair_params = num_atomic_params * (num_atomic_params + 1)
        for param_name in self.ff_param:
            self.param[param_name] = {}                
            expand_indices = self.param_expand_indices[param_name]
            for key in self.ff_param[param_name]:
                if not isinstance(self.ff_param[param_name][key], torch.Tensor):
                    continue
                # special treatment for pairwise params
                if param_name == 'pair_params':
                    self.param[param_name][key] = torch.zeros((num_pair_params,))
                    self.param[param_name][key][expand_indices] += self.ff_param[param_name][key]
                    continue

                self.param[param_name][key] = self.ff_param[param_name][key][expand_indices]
            
            if param_name == 'atomic_params':
                self.param[param_name]['q_shell'] = (self.ff_param[param_name]['mono'] - self.ff_param[param_name]['Z'])[expand_indices]
                self.param[param_name]['quad'] = computeCartesianQuadrupoles(self.ff_param[param_name]['quad_s'])[expand_indices]
                
                # original alpha params contains only axx, axy, axz, ayy, ayz, azz columns
                alpha_dense = self.ff_param[param_name]['alpha']
                self.param[param_name]['alpha'] = alpha_dense[:, [0, 1, 2, 1, 3, 4, 2, 4, 5]].reshape(-1, 3, 3)[expand_indices]
        
        self.group_charges = scatter(self.param['atomic_params']['mono'], self.group_indices)
        self.group_charges = torch.zeros_like(self.group_charges)
    
    def evaluate_bonded_terms(self, coords):
        bond_vecs = coords[self.bond_pairs[1]] - coords[self.bond_pairs[0]]
        bond_dists = computeBondFromVecs(bond_vecs)
        angle_thetas = computeAngleFromVecs(
            bond_vecs[self.angle_as_bond_indices[0]],
            bond_vecs[self.angle_as_bond_indices[1]]
        )
        return bond_dists, angle_thetas

    
    def evaluate_electric_properties(self, coords: torch.Tensor, grid: torch.Tensor | None = None):

        self.expand_params()
        
        atomic_params = self.param['atomic_params']
        bond_params = self.param['bond_params']
        angle_params = self.param['angle_params']

        # Rotation matrix
        rot_mats = computeLocal2GlobalRotationMatrixBatch(
            coords, 
            self.z_atoms, 
            self.x_atoms, 
            self.y_atoms, 
            atomic_params['axis_type']
        )

        # Electric Multipoles
        q_shell: torch.Tensor = atomic_params['q_shell']
        
        bond_vecs = coords[self.bond_pairs[1]] - coords[self.bond_pairs[0]]
        bond_dists = computeBondFromVecs(bond_vecs)
        bond_cf_i, bond_cf_j = computeChargeFluxBond(
            bond_dists,
            bond_params['r_eq'],
            bond_params['j_cf']
        )
        q_shell.scatter_add_(0, self.bond_pairs[0], bond_cf_i)
        q_shell.scatter_add_(0, self.bond_pairs[1], bond_cf_j)

        if self.has_angles:
            bb_cf_1_i, bb_cf_1_j, bb_cf_2_i, bb_cf_2_j = computeChargeFluxBondBond(
                bond_dists[self.angle_as_bond_indices[0]],
                bond_dists[self.angle_as_bond_indices[1]],
                bond_params['r_eq'][self.angle_as_bond_indices[0]],
                bond_params['r_eq'][self.angle_as_bond_indices[1]],
                angle_params['j_cf_bb'],
                angle_params['j_cf_bb']
            )

            angle_thetas = computeAngleFromVecs(
                bond_vecs[self.angle_as_bond_indices[0]],
                bond_vecs[self.angle_as_bond_indices[1]]
            )
            
            angle_cf_i, angle_cf_j, angle_cf_k = computeChargeFluxAngle(
                angle_thetas,
                angle_params['theta_eq'],
                angle_params['j_cf_angle']
            )

            q_shell.scatter_add_(0, self.angle_pairs[0], angle_cf_i)
            q_shell.scatter_add_(0, self.angle_pairs[1], angle_cf_j)
            q_shell.scatter_add_(0, self.angle_pairs[2], angle_cf_k)
            q_shell.scatter_add_(0, self.angle_pairs[1], bb_cf_1_i)
            q_shell.scatter_add_(0, self.angle_pairs[0], bb_cf_1_j)
            q_shell.scatter_add_(0, self.angle_pairs[1], bb_cf_2_i)
            q_shell.scatter_add_(0, self.angle_pairs[2], bb_cf_2_j)

        multipoles = rotateMultipoles(
            q_shell,
            atomic_params['dipo'],
            atomic_params['quad'],
            rot_mats
        ) * torch.tensor([1, 1, 1, 1, 1/3, 2/3, 2/3, 1/3, 2/3, 1/3])

        if grid is not None:
            ngrids = grid.shape[0]
            pairs = torch.tensor(list(itertools.product(range(coords.shape[0]), range(ngrids)))).T
            ePot, eField, eField_grad = computePermanentElectricPotentialExpansion(
                ngrids,
                grid[pairs[1]] - coords[pairs[0]],
                pairs,
                multipoles,
                atomic_params['Z'],
                atomic_params['b_elec']
            )
        else:
            ePot = None

        polarizability = torch.sum(rotateQuadrupoles(atomic_params['alpha'], rot_mats), dim=0)
        eta = 1 / (2 * atomic_params['eta'])
        tmp_a = torch.sum(eta.view(-1, 1, 1) * torch.einsum('ni,nj->nij', coords, coords), dim=0)
        
        weighted_coords = torch.sum(eta.view(-1, 1) * coords, dim=0)
        tmp_b = torch.outer(weighted_coords, weighted_coords) / torch.sum(eta)
        polarizability += tmp_a - tmp_b

        multipoles[:, 0] += atomic_params['Z']
        dipole = torch.sum(multipoles[:, [0]] * coords + multipoles[:,1:4], dim=0)

        res = {
            "multipoles": multipoles,
            "charge": torch.sum(multipoles[:, 0]),
            "dipole": dipole, # molecular dipoleta
            "polarizability": polarizability
        }
        if ePot is not None:
            res['esp'] = ePot
        return res
    
    def evaluate(self, coords: torch.Tensor):
        self.expand_params()
        atomic_params = self.param['atomic_params']
        pair_params = self.param['pair_params']
        bond_params = self.param['bond_params']
        angle_params = self.param['angle_params']
        
        # Rotation matrix
        rot_mats = computeLocal2GlobalRotationMatrixBatch(
            coords, 
            self.z_atoms, 
            self.x_atoms, 
            self.y_atoms, 
            atomic_params['axis_type']
        )
        
        # Electric Multipoles
        q_shell: torch.Tensor = atomic_params['q_shell']
        
        # Electric Charge Flux
        bond_vecs = coords[self.bond_pairs[1]] - coords[self.bond_pairs[0]]
        bond_dists = computeBondFromVecs(bond_vecs)
        bond_cf_i, bond_cf_j = computeChargeFluxBond(
            bond_dists,
            bond_params['r_eq'],
            bond_params['j_cf']
        )
        q_shell.scatter_add_(0, self.bond_pairs[0], bond_cf_i)
        q_shell.scatter_add_(0, self.bond_pairs[1], bond_cf_j)

        if self.has_angles:
            bb_cf_1_i, bb_cf_1_j, bb_cf_2_i, bb_cf_2_j = computeChargeFluxBondBond(
                bond_dists[self.angle_as_bond_indices[0]],
                bond_dists[self.angle_as_bond_indices[1]],
                bond_params['r_eq'][self.angle_as_bond_indices[0]],
                bond_params['r_eq'][self.angle_as_bond_indices[1]],
                angle_params['j_cf_bb'],
                angle_params['j_cf_bb']
            )

            angle_thetas = computeAngleFromVecs(
                bond_vecs[self.angle_as_bond_indices[0]],
                bond_vecs[self.angle_as_bond_indices[1]]
            )
            angle_cf_i, angle_cf_j, angle_cf_k = computeChargeFluxAngle(
                angle_thetas,
                angle_params['theta_eq'],
                angle_params['j_cf_angle']
            )

            q_shell.scatter_add_(0, self.angle_pairs[0], angle_cf_i)
            q_shell.scatter_add_(0, self.angle_pairs[1], angle_cf_j)
            q_shell.scatter_add_(0, self.angle_pairs[2], angle_cf_k)
            q_shell.scatter_add_(0, self.angle_pairs[1], bb_cf_1_i)
            q_shell.scatter_add_(0, self.angle_pairs[0], bb_cf_1_j)
            q_shell.scatter_add_(0, self.angle_pairs[1], bb_cf_2_i)
            q_shell.scatter_add_(0, self.angle_pairs[2], bb_cf_2_j)

        multipoles = rotateMultipoles(
            q_shell,
            atomic_params['dipo'],
            atomic_params['quad'],
            rot_mats
        ) * torch.tensor([1, 1, 1, 1, 1/3, 2/3, 2/3, 1/3, 2/3, 1/3])

        # nonbonded dist vectors
        pairs = self.nbpairs
        drVec = coords[pairs[1]] - coords[pairs[0]]
        dr = torch.norm(drVec, dim=1)

        # Polarizability
        polarizabilities = rotateQuadrupoles(atomic_params['alpha'], rot_mats)

        # Charge-Transfer
        b_ct = atomic_params['b_ct']

        atomic_param_indices = self.param_expand_indices['atomic_params']
        atom_type_indices_pairs = torch.vstack((atomic_param_indices[pairs[0]], atomic_param_indices[pairs[1]])).T
        pair_indices = symmetric_pairing_function(atom_type_indices_pairs)
        eps_ct = pair_params['eps_ct'][pair_indices]
        # print("Pairs:", pairs)
        # print("pair_indices:", pair_indices)
        # print("eps_ct:", eps_ct)

        mPoles_ct_acc = scaleMultipoles(
            multipoles, 
            atomic_params['q_ct_acc'], 
            atomic_params['Kdipo_ct_acc'], 
            atomic_params['Kquad_ct_acc']
        )
        mPoles_ct_don = scaleMultipoles(
            multipoles, 
            atomic_params['q_ct_don'], 
            atomic_params['Kdipo_ct_don'], 
            atomic_params['Kquad_ct_don']
        )
        
        ct_direct_pairwise, dq_pairwise = computePairwiseChargeTransfer(
            drVec,
            mPoles_ct_acc[pairs[0]], mPoles_ct_acc[pairs[1]],
            mPoles_ct_don[pairs[0]], mPoles_ct_don[pairs[1]],
            b_ct[pairs[0]], b_ct[pairs[1]],
            eps_ct
        )
        ene_ct_direct = torch.sum(ct_direct_pairwise) / 2 * HARTREE2KCAL
        dq = scatter(dq_pairwise, pairs[1])
        dq_groups = scatter(dq, self.group_indices)

        ene_perm_elec, ene_pol = computePermElecAndPolarizationEnergy(
            coords,
            self.groups,
            multipoles,
            atomic_params['Z'],
            atomic_params['b_elec'],
            True,
            polarizabilities,
            atomic_params['eta'] * 2, # can we move this '2' in to param definition?
            self.group_charges,
        )

        _, ene_pol_ct = computePermElecAndPolarizationEnergy(
            coords,
            self.groups,
            multipoles,
            atomic_params['Z'],
            atomic_params['b_elec'],
            True,
            polarizabilities,
            atomic_params['eta'] * 2,
            self.group_charges + dq_groups,
        )
        ene_perm_elec *= HARTREE2KCAL
        ene_pol *= HARTREE2KCAL
        ene_pol_ct *= HARTREE2KCAL
        ene_ct_indirect = ene_pol_ct - ene_pol

        # Pauli repulsion
        q_pauli: torch.Tensor = atomic_params['q_pauli']
        bond_cf_pauli_i, bond_cf_pauli_j = computeChargeFluxBond(
            bond_dists,
            bond_params['r_eq'],
            bond_params['j_cf_pauli']
        )
        q_pauli.scatter_add_(0, self.bond_pairs[0], bond_cf_pauli_i)
        q_pauli.scatter_add_(0, self.bond_pairs[1], bond_cf_pauli_j)

        multipoles_pauli = scaleMultipoles(
            multipoles, 
            q_pauli, 
            atomic_params['Kdipo_pauli'], 
            atomic_params['Kquad_pauli']
        )
        b_pauli = atomic_params['b_pauli']
        pauli_pairwise = computeShortRangeEnergyFromPairs(
            dr, drVec,
            multipoles_pauli[pairs[0]], multipoles_pauli[pairs[1]],
            torch.sqrt(b_pauli[pairs[0]] * b_pauli[pairs[1]]),
            True
        )
        ene_pauli = torch.sum(pauli_pairwise) / 2 * HARTREE2KCAL

        # Dispersion
        C6_disp = atomic_params['C6_disp']
        b_disp = atomic_params['b_disp']
        # print(C6_disp, b_disp)
        disp_pairwise = computeDispersionFromPairs(
            dr,
            torch.sqrt(C6_disp[pairs[0]] * C6_disp[pairs[1]]),
            torch.sqrt(b_disp[pairs[0]] * b_disp[pairs[1]])
        )
        ene_disp = torch.sum(disp_pairwise) / 2 * HARTREE2KCAL

        # Exchange-polarization
        multipoles_xpol = scaleMultipoles(
            multipoles, 
            atomic_params['q_xpol'], 
            atomic_params['Kdipo_xpol'], 
            atomic_params['Kquad_xpol']
        )
        b_xpol = atomic_params['b_xpol']
        xpol_pairwise = computeShortRangeEnergyFromPairs(
            dr, drVec,
            multipoles_xpol[pairs[0]], multipoles_xpol[pairs[1]],
            torch.sqrt(b_xpol[pairs[0]] * b_xpol[pairs[1]]),
            False
        )
        ene_xpol = torch.sum(xpol_pairwise) / 2 * HARTREE2KCAL

        ene_tot = ene_perm_elec + ene_pauli + ene_disp + ene_pol + ene_ct_direct + ene_ct_indirect + ene_xpol
        ene = {
            "perm_elec": ene_perm_elec,
            "pauli": ene_pauli,
            "disp": ene_disp,
            "elec_pol": ene_pol,
            "xpol": ene_xpol,
            "pol": ene_xpol + ene_pol,
            "pol_ct": ene_pol_ct,
            "ct": ene_ct_direct + ene_ct_indirect,
            "ct_direct": ene_ct_direct,
            "ct_indirect": ene_ct_indirect,
            "total": ene_tot
        }
        return ene
    
    def batch_evalute(self, coords: torch.Tensor):
        enes_cmm_raw = [self.evaluate(coord) for coord in coords]
        enes_cmm = {key: [] for key in enes_cmm_raw[0].keys()}
        for e in enes_cmm_raw:
            for key in enes_cmm:
                enes_cmm[key].append(e[key])
        enes_cmm = {key: torch.hstack(enes_cmm[key]) for key in enes_cmm}
        return enes_cmm
    
    def batch_evaluate_electric_properties(self, coords, grids=None):
        if grids is None:
            grids = [None for _ in range(len(coords))]
        
        results = []
        for coord, grid in zip(coords, grids):
            res = self.evaluate_electric_properties(coord, grid)
            results.append(res)

        results_as_dict = {key: [] for key in results[0].keys()}
        for res in results:
            for key in results_as_dict:
                results_as_dict[key].append(res[key])
        
        results = {key: torch.vstack(results_as_dict[key]) for key in results_as_dict}
        return results



def symmetric_pairing_function(pairs: torch.Tensor):
    return torch.floor_divide(torch.square(torch.sum(pairs, dim=1) + 1) - torch.remainder((torch.sum(pairs, dim=1) + 1), 2), 4) + torch.min(pairs, dim=1).values


class CMMForceField:
    def __init__(self, param: os.PathLike | Dict[str, Any]):
        if not isinstance(param, dict):
            param = self.parse_file(param)
        
        # Definition of atom types, e.g. {'HOH': {'O': 'ow', 'H1': 'hw', 'H2': 'hw'}}
        self.atypes_def: Dict[str, Dict[str, Any]] = {}
        # Parameters, e.g. {'atomic_params': {'Z': ...}, 'pair_params': {'D': ...}}
        self.params: Dict[str, Dict[str, Any]] = {}
        # Parameters type indices mapping, e.g. {'atomic_params': {'ow': 0}}
        self.params_types_indices_map: Dict[str, Dict[str, Any]] = {}

        self.from_raw_dict(param)
    
    def from_raw_dict(self, raw_dict: Dict[str, Any]):
        self.atypes_def = raw_dict.pop('atomtypes')

        for param_name, items in raw_dict.items():
            tmp = defaultdict(list)
            for item in items:
                for key, value in item.items():
                    tmp[key].append(value)
            
            param = {}
            type_indices_map = {}
            for key in tmp:
                if key == 'axis_type':
                    param[key] = torch.tensor([getattr(AxisTypes, v).value for v in tmp[key]])
                else:
                    try:
                        param[key] = torch.tensor(tmp[key])
                    except:
                        param[key] = tmp[key]
            
                # record the indices of each atom type
                for i, type in enumerate(param['type']):
                    if isinstance(type, str):
                        type_indices_map[type] = i
                    else:
                        type_indices_map[tuple(type)] = i

            self.params[param_name] = param
            self.params_types_indices_map[param_name] = type_indices_map
        
        # Special treatment to pair_params because they are expanded on-the-fly
        atom_type_indices_pairs = []
        for type in self.params['pair_params']['type']:
            atom_type_indices_pairs.append(
                [
                    self.params_types_indices_map['atomic_params'][type[0]], 
                    self.params_types_indices_map['atomic_params'][type[1]]
                ]
            )
        atom_type_indices_pairs = torch.tensor(atom_type_indices_pairs, dtype=torch.long)
        self.pair_param_indices_after_paring_func = symmetric_pairing_function(atom_type_indices_pairs)

    def to_raw_dict(self):
        raw_dict = {}
        raw_dict.update({'atomtypes': self.atypes_def})
        
        for param_name in self.params:
            param_as_list = {}
            for key in self.params[param_name]:
                if isinstance(self.params[param_name][key], torch.Tensor):
                    param_as_list[key] = self.params[param_name][key].tolist()
                else:
                    param_as_list[key] = self.params[param_name][key]
            num_params = len(param_as_list[key])

            raw_param = []
            for i in range(num_params):
                raw_param.append({key: param_as_list[key][i] for key in param_as_list})
            raw_dict[param_name] = raw_param
        
        axisTypesInt2Str = {member.value: member.name for member in AxisTypes}
        for param in raw_dict['atomic_params']:
            param['axis_type'] = axisTypesInt2Str[param['axis_type']]
        
        return raw_dict
    
    def save(self, fname: os.PathLike):
        fpath = Path(fname)
        if fpath.suffix == '.json':
            with open(fname, 'w') as f:
                json.dump(self.to_raw_dict(), f, indent=4)
        else:
            raise NotImplementedError()

    def parse_file(self, fname: os.PathLike):
        fpath = Path(fname)
        if not fpath.is_file():
            raise IOError(f"{fname} does not exist or no a file")
        suffix = fpath.suffix
        if suffix == '.json':
            return self.parse_json(fpath)
        elif suffix == '.toml':
            return self.parse_toml(fpath)
        else:
            raise NotImplementedError(f'Unsupported parameter file format: {suffix}')
    
    def parse_json(self, fname: os.PathLike):
        with open(fname) as f:
            data = json.load(f)
        return data
    
    def parse_toml(self, fname: os.PathLike):
        with open(fname) as f:
            data = toml.load(f)
        return data
    
    def assign_atom_types(self, topology: app.Topology):
        atypes = []
        for residue in topology.residues():
            for atom in residue.atoms():
                atype = self.atypes_def.get(residue.name, {}).get(atom.name, None)
                assert atype is not None, f'No atom type defined for {atom.name} in {residue.name}'
                atypes.append(atype)
        return atypes
    
    def parametrize(self, topology: app.Topology):
        atypes = self.assign_atom_types(topology)
        param_expand_indices = {}
        param_expand_indices['atomic_params'] = torch.tensor(
            [self.params_types_indices_map['atomic_params'][a] for a in atypes]
        )
        param_expand_indices['pair_params'] = self.pair_param_indices_after_paring_func
        
        data = TopologyData(topology)

        # bond pairs
        param_expand_indices['bond_params'] = []
        bond_pairs = []
        for bond in data.bonds:
            atom1, atom2 = int(bond[0]), int(bond[1])
            if (atypes[atom1], atypes[atom2]) in self.params_types_indices_map['bond_params']:
                bond_pairs.append([atom1, atom2])
                param_expand_indices['bond_params'].append(
                    self.params_types_indices_map['bond_params'][(atypes[atom1], atypes[atom2])]
                )
            elif (atypes[atom2], atypes[atom1]) in self.params_types_indices_map['bond_params']:
                bond_pairs.append([atom2, atom1])
                param_expand_indices['bond_params'].append(
                    self.params_types_indices_map['bond_params'][(atypes[atom2], atypes[atom1])]
                )
            else:
                raise ValueError(f'Bond bewteen atom {atom1} and {atom2} does not have a parameter')
        param_expand_indices['bond_params'] = torch.tensor(param_expand_indices['bond_params'], dtype=torch.long)
        bond_pairs = torch.tensor(bond_pairs, dtype=torch.long).T
        
        # angles
        param_expand_indices['angle_params'] = []
        angle_pairs = []
        for angle in data.angles:
            atom1, atom2, atom3 = int(angle[0]), int(angle[1]), int(angle[2])
            atyp1, atyp2, atyp3 = atypes[atom1], atypes[atom2], atypes[atom3]
            if (atyp1, atyp2, atyp3) in self.params_types_indices_map['angle_params']:
                angle_pairs.append([atom1, atom2, atom3])
                param_expand_indices['angle_params'].append(
                    self.params_types_indices_map['angle_params'][(atyp1, atyp2, atyp3)]
                )
            elif (atyp3, atyp2, atyp1) in self.params_types_indices_map['angle_params']:
                angle_pairs.append([atom3, atom2, atom1])
                angle_pairs['angle_params'].append(
                    self.params_types_indices_map['angle_params'][(atyp3, atyp2, atyp1)]
                )
            else:
                raise ValueError(f'Angle bewteen atoms ({atom1}, {atom2}, {atom3}) does not have a parameter')
        param_expand_indices['angle_params'] = torch.tensor(param_expand_indices['angle_params'], dtype=torch.long)
        angle_pairs = torch.tensor(angle_pairs, dtype=torch.long).T


        # electrostatics - find atoms for constructing local frame
        z_atoms, x_atoms, y_atoms = [], [], []
        for atom_index in range(data.n_atoms):
            kz, kx, ky = -1, -1, -1
            atype = atypes[atom_index]
            atype_index = self.params_types_indices_map['atomic_params'][atype]

            z_atype = self.params['atomic_params']['z_atom'][atype_index]
            x_atype = self.params['atomic_params']['x_atom'][atype_index]
            y_atype = self.params['atomic_params']['y_atom'][atype_index]
            
            # z-atom
            if z_atype:
                for nei_idx in data.neighbors[atom_index]:
                    if atypes[nei_idx] == z_atype:
                        kz = nei_idx
                        break
                # raise an error if not found Z-atom in its first neighbors
                assert (kz != -1), f'Cannot determine Z-axis for atom {atom_index}'
            
            # x-atom
            if x_atype:
                for nei_idx in data.neighbors[atom_index]:
                    if nei_idx != kz and atypes[nei_idx] == x_atype:
                        kx = nei_idx
                        break
                # x-axis is determined by its second neighbor, like H in water
                if kx == -1:
                    for nnei_idx in data.neighbors[kz]:
                        if nnei_idx != atom_index and atypes[nnei_idx] == x_atype:
                            kx = nnei_idx
                            break
                assert kx != -1, f'Cannot determine X-axis for atom {atom_index}'

            # y-atom
            if y_atype:
                for nei_idx in data.neighbors[atom_index]:
                    if nei_idx != kz and nei_idx != kx and atypes[nei_idx] == y_atype:
                        ky = nei_idx
                        break
                # y-axis is determined by its second neighbor, like H in water
                if ky == -1:
                    nei_idx = data.neighbors[atom_index][0]
                    for nnei_idx in data.neighbors[nei_idx]:
                        if nnei_idx != atom_index and nnei_idx != kx and atypes[nnei_idx] == y_atype:
                            ky = nnei_idx
                            break
                assert ky != -1, f'Cannot determine Y-axis for atom {atom_index}'
            
            z_atoms.append(kz)
            x_atoms.append(kx)
            y_atoms.append(ky)
        
        system = SystemNoCutoff(
            topdata=data,
            ff_param=self.params,
            param_expand_indices=param_expand_indices,
            z_atoms=z_atoms,
            x_atoms=x_atoms,
            y_atoms=y_atoms,
            bond_pairs=bond_pairs,
            angle_pairs=angle_pairs
        )
        return system
