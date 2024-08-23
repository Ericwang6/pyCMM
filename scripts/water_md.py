import itertools
from pprint import pprint
import math
import time

import numpy as np

import torch
torch.set_printoptions(precision=8)
import torch.nn as nn
from torch_scatter import scatter

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import cmm



def read_tinker_xyz(fname):
    atoms = []
    coords = []
    with open(fname) as f:
        num_atoms = int(f.readline().strip().split()[0])
        for _ in range(num_atoms):
            line = f.readline().strip().split()
            atoms.append(line[1])
            coords.append([float(x) for x in line[2: 5]])
    return atoms, coords


class CMMWater(nn.Module):
    def __init__(self, num_waters: int, rcut: float = 10, use_pme: bool = False, do_polarization: bool = True, seperate_indirect_ct: bool = True):
        Z = torch.tensor([3.61565, 0.93619])
        mono = torch.tensor([-0.390896, 0.195448])
        qShell = mono - Z
        dipo = torch.tensor([
            [0.0,       0.0, -0.094298],
            [0.0910288, 0.0, -0.207851]
        ])
    
        quad_s = torch.tensor([
            # Q20,       Q21c,      Q21s, Q22c,       Q22s
            [-0.330685,  0.0,       0.0,  0.869923,   0.0],
            [-0.0739388, 0.0929482, 0.0,  0.00532425, 0.0]
        ])

        self.nb_params_raw = {
            # elec
            "Z": Z,
            "q_shell": qShell,
            "mono": mono,
            "dipo": dipo,
            "quad_s": quad_s,
            "quad": cmm.computeCartesianQuadrupoles(quad_s),
            "b": torch.tensor([2.13358, 2.33322]),
            # Pauli repulsion
            "b_pauli": torch.tensor([2.1975, 1.96474]),
            "Kmono_pauli": torch.tensor([6.50923, 0.527804]) / qShell,
            "Kdipo_pauli": torch.tensor([-5.61925, -0.515584]),
            "Kquad_pauli": torch.tensor([-1.56567, -0.440164]),
            # Dispersion
            "C6_disp": torch.tensor([35.8289, 1.98954]),
            "b_disp": torch.tensor([1.84302, 1.30993]),
            # Polarization
            "alpha": torch.tensor([
                [[4.45992, 0.0, 0.0], [0.0, 6.07259, 0.0], [0.0, 0.0, 4.55391]],
                [[2.22001, 0.0, 0.0], [0.0, 1.66835, 0.0], [0.0, 0.0, 0.183855]]
            ]),
            "eta": torch.tensor([6.18699e-6, 0.561535]) * 2,          
            # Exchange-polarization
            "b_xpol": torch.tensor([2.73582, 2.04028]),
            "Kmono_xpol": torch.tensor([1.26592, 0.200089]) / qShell,
            "Kdipo_xpol": torch.zeros((2,)),
            "Kquad_xpol": torch.zeros((2,)),
            # Charge Transfer
            "b_ct": torch.tensor([1.89485, 2.36763]),
            "Kmono_ct_acc": torch.tensor([-0.67857, 1.36735]) / qShell,
            "Kdipo_ct_acc": torch.tensor([0.0, 0.0]),
            "Kquad_ct_acc": torch.tensor([0.0, 0.0]),
            "Kmono_ct_don": torch.tensor([0.757752, 0.00888982]) / qShell,
            "Kdipo_ct_don": torch.tensor([-0.512036, -0.0511668]),
            "Kquad_ct_don": torch.tensor([-0.208186, 0.0568152]),
            "eps": torch.tensor([[1e15, 0.380979], [0.380979, 1e15]])
        }
        
        # expand parameters to atoms/pairs
        paramIndices = torch.tensor([0, 1, 1] * num_waters, dtype=torch.long)
        self.nb_params = {}
        for key in self.nb_params_raw:
            if key == 'eps':
                self.nb_params[key] = self.nb_params_raw[key][torch.meshgrid(paramIndices, paramIndices, indexing='xy')]
            else:
                self.nb_params[key] = self.nb_params_raw[key][paramIndices]

        # compute ref atoms for rotate multipoles
        zatoms, xatoms, yatoms, axistypes = [], [], [], []
        for i in range(num_waters * 3):
            if i % 3 == 0:
                zatoms.append(i + 1)
                xatoms.append(i + 2)
                axistypes.append(1)
            elif i % 3 == 1:
                zatoms.append(i - 1)
                xatoms.append(i + 1)
                axistypes.append(0)
            else:
                zatoms.append(i - 2)
                xatoms.append(i - 1)
                axistypes.append(0)
            yatoms.append(-1)

        self.nb_params.update({
            "zatoms": torch.tensor(zatoms, dtype=torch.long),
            "xatoms": torch.tensor(xatoms, dtype=torch.long),
            "yatoms": torch.tensor(yatoms, dtype=torch.long),
            "axistypes": torch.tensor(axistypes, dtype=torch.long),
            "groups": [[i, i + 1, i + 2] for i in range(0, num_waters * 3, 3)],
            "groups_scatter": torch.concat([torch.tensor([i, i, i], dtype=torch.long) for i in range(num_waters)]),
            "groupCharges": torch.zeros(num_waters),  
        })

        self.bonds = []
        self.bbs = torch.tensor([[i, i+1] for i in range(0, num_waters * 2, 2)], dtype=torch.long).T
        self.bas = torch.vstack((
            torch.arange(num_waters * 2, dtype=torch.long),
            torch.concat([torch.tensor([i, i], dtype=torch.long) for i in range(num_waters)])
        ))

        for i in range(0, num_waters * 3, 3):
            self.bonds.append([i, i+1])
            self.bonds.append([i, i+2])
        self.bonds = torch.tensor(self.bonds, dtype=torch.long).T

        self.bonded_params_raw = {
            "D": torch.tensor([524.265 / cmm.HARTREE2KJ]),
            "k_b": torch.tensor([5098.15 / cmm.HARTREE2KJ * cmm.BOHR2ANG * cmm.BOHR2ANG]),
            "b_eq": torch.tensor([0.958413 / cmm.BOHR2ANG]),
            "k_bb": torch.tensor([-61.1423 / cmm.HARTREE2KJ * cmm.BOHR2ANG * cmm.BOHR2ANG]),
            "k_ba": torch.tensor([-159.886 / cmm.HARTREE2KJ * cmm.BOHR2ANG]),
            "theta_eq": torch.tensor([104.4234 * math.pi / 180.00]),
            "k_theta": torch.tensor([452.183 / cmm.HARTREE2KJ])
        }
        self.bonded_params_raw['beta'] = torch.sqrt(self.bonded_params_raw['k_b'] / 2 / self.bonded_params_raw['D'])

        # expand bonded parameters
        self.bonded_params = {}
        for key in self.bonded_params_raw:
            if key in ['d_oh', 'k_b', 'b_eq', 'beta', 'k_ba', 'D']:
                self.bonded_params[key] = self.bonded_params_raw[key][torch.zeros(num_waters * 2, dtype=torch.long)]
            else:
                self.bonded_params[key] = self.bonded_params_raw[key][torch.zeros(num_waters, dtype=torch.long)]
        
        self.all_pairs = cmm.getPairsFromGroups(self.nb_params['groups'])
        self.rcut = rcut / cmm.BOHR2ANG

        self.use_pme = use_pme
        self.do_polarization = do_polarization
        self.seperate_indirect_ct = seperate_indirect_ct

    def computeEnergy(self, coords: torch.Tensor, box: torch.Tensor):
        boxInv = torch.linalg.inv(box)
        bondVecs = cmm.applyPBC(coords[self.bonds[1]] - coords[self.bonds[0]], box, boxInv)
        bonds = torch.norm(bondVecs, dim=1)
        # morse-bond
        ene_bond_list = cmm.computeMorseBondPotential(bonds, self.bonded_params['b_eq'], self.bonded_params['D'], self.bonded_params['beta'])
        ene_bonds = torch.sum(ene_bond_list)

        # bond-bond couplings
        ene_bbs_list = cmm.computeBondBondCoupling(
            bonds[self.bbs[0]], bonds[self.bbs[1]],
            self.bonded_params['b_eq'][self.bbs[0]], self.bonded_params['b_eq'][self.bbs[1]],
            self.bonded_params['k_bb']
        )
        ene_bbs = torch.sum(ene_bbs_list)

        # angles
        angles = cmm.computeAngleFromVecs(bondVecs[self.bbs[0]], bondVecs[self.bbs[1]])
        ene_angles_list = cmm.computeCosAnglePotential(
            angles, self.bonded_params['theta_eq'], self.bonded_params['k_theta']
        )
        ene_angles = torch.sum(ene_angles_list)

        # bond-angle couplings
        ene_bas_list = cmm.computeBondAngleCoupling(
            bonds[self.bas[0]], self.bonded_params['b_eq'][self.bas[0]],
            angles[self.bas[1]], self.bonded_params['theta_eq'][self.bas[1]],
            self.bonded_params['k_ba']
        )
        ene_bas = torch.sum(ene_bas_list)

        # non-bonded interactions
        rotMatrix = cmm.computeLocal2GlobalRotationMatrix(
            coords, 
            coords[self.nb_params['zatoms']],
            coords[self.nb_params['xatoms']],
            coords[self.nb_params['yatoms']],
            self.nb_params['axistypes'],
            box,
            boxInv
        )

        mPoles = cmm.rotateMultipoles(
            self.nb_params['q_shell'],
            self.nb_params['dipo'],
            self.nb_params['quad'],
            rotMatrix
        ) * torch.tensor([1, 1, 1, 1, 1/3, 2/3, 2/3, 1/3, 2/3, 1/3])

        polarizabilities = cmm.rotateQuadrupoles(self.nb_params['alpha'], rotMatrix)

        pairs, drVecs = self.computeNeighborList(coords, box, boxInv)

        # direct charge-transfer
        mPoles_ct_acc = cmm.scaleMultipoles(mPoles, self.nb_params['Kmono_ct_acc'], self.nb_params['Kdipo_ct_acc'], self.nb_params['Kquad_ct_acc'])
        mPoles_ct_don = cmm.scaleMultipoles(mPoles, self.nb_params['Kmono_ct_don'], self.nb_params['Kdipo_ct_don'], self.nb_params['Kquad_ct_don'])

        ct_direct_pairwise, dq_pairwise = cmm.computePairwiseChargeTransfer(
            drVecs,
            mPoles_ct_acc[pairs[0]], mPoles_ct_acc[pairs[1]],
            mPoles_ct_don[pairs[0]], mPoles_ct_don[pairs[1]],
            self.nb_params['b_ct'][pairs[0]], self.nb_params['b_ct'][pairs[1]],
            self.nb_params['eps'][pairs[0], pairs[1]]
        )
        ene_ct_direct = torch.sum(ct_direct_pairwise) / 2
        dq = scatter(dq_pairwise, pairs[1])

        dq_groups = scatter(dq, self.nb_params['groups_scatter'])

        # elec, pol and charge-transfer
        if self.seperate_indirect_ct:
            groupCharges = self.nb_params['groupCharges']
            groupChargesCT = self.nb_params['groupCharges'] + dq_groups
        else:
            groupCharges = self.nb_params['groupCharges'] + dq_groups
            groupCharges = None
        ene_perm_elec, ene_pol, ene_ct_indirect = cmm.computePermElecAndPolarizationEnergy(
            coords,
            self.nb_params['groups'],
            mPoles,
            self.nb_params['Z'],
            self.nb_params['b'],
            self.do_polarization,
            polarizabilities,
            self.nb_params['eta'],
            groupCharges,
            groupChargesCT,
            pairs = pairs
        )
        
        # Pauli repulsion
        mPoles_pauli = cmm.scaleMultipoles(mPoles, self.nb_params['Kmono_pauli'], self.nb_params['Kdipo_pauli'], self.nb_params['Kquad_pauli'])
        pauli_pairwise = cmm.computeShortRangeEnergy(
            drVecs,
            mPoles_pauli[pairs[0]], mPoles_pauli[pairs[1]],
            self.nb_params['b_pauli'][pairs[0]], self.nb_params['b_pauli'][pairs[1]]
        )
        ene_pauli = torch.sum(pauli_pairwise) / 2

        # dispersion
        disp_pairwise = cmm.computeDispersion(
            drVecs, 
            self.nb_params['C6_disp'][pairs[0]], self.nb_params['C6_disp'][pairs[1]],
            self.nb_params['b_disp'][pairs[0]], self.nb_params['b_disp'][pairs[1]]
        )
        ene_disp = torch.sum(disp_pairwise) / 2

        # exchange-polarization
        mPoles_xpol = cmm.scaleMultipoles(mPoles, self.nb_params['Kmono_xpol'], self.nb_params['Kdipo_xpol'], self.nb_params['Kquad_xpol'])
        xpol_pairwise = cmm.computeShortRangeEnergy(
            drVecs,
            mPoles_xpol[pairs[0]], mPoles_xpol[pairs[1]],
            self.nb_params['b_xpol'][pairs[0]], self.nb_params['b_xpol'][pairs[1]],
            False
        )
        ene_xpol = torch.sum(xpol_pairwise) / 2

        ene_tot = ene_perm_elec + ene_pol + ene_xpol + ene_pauli + ene_disp + ene_ct_direct + ene_bonds + ene_angles + ene_bas + ene_bbs + ene_ct_indirect
        energies = {
            "perm_elec": ene_perm_elec,
            "pol": ene_pol,
            "ct_direct": ene_ct_direct,
            "ct_indirect": ene_ct_indirect,
            "xpol": ene_xpol,
            "pauli": ene_pauli,
            "disp": ene_disp,
            "bond": ene_bonds,
            "angle": ene_angles,
            "bond_bond": ene_bbs,
            "bond_angle": ene_angles,
            "tot": ene_tot
        }
        return energies

        
    def computeNeighborList(self, coords: torch.Tensor, box: torch.Tensor, boxInv: torch.Tensor):
        drVecs = cmm.pbc.applyPBC(coords[self.all_pairs[1]] - coords[self.all_pairs[0]], box, boxInv)
        mask = torch.norm(drVecs, dim=1) < self.rcut
        pairs = self.all_pairs[:, mask]
        drVecs = drVecs[mask]
        return pairs, drVecs


if __name__ == '__main__':
    model = CMMWater(2, do_polarization=True)
    coords = torch.tensor(np.array([
        [ 1.5165013870,  -0.0000008497,   0.1168590962],
        [ 0.5714469342,   0.0000007688,  -0.0477240756],
        [ 1.9206469769,   0.0000030303,  -0.7531309382],
        [-1.3965797657,   0.0000005579,  -0.1058991347],
        [-1.7503737705,  -0.7612400781,   0.3583839434],
        [-1.7503754020,   0.7612382608,   0.3583875091]
    ]) / cmm.BOHR2ANG, dtype=torch.float32, requires_grad=True)
    box = torch.tensor(np.eye(3) * 100, dtype=torch.float32, requires_grad=True)

    energies = model.computeEnergy(coords, box)
    energies['tot'].backward()
    grad = coords.grad

    for key in energies:
        energies[key] *= cmm.HARTREE2KCAL
    pprint(energies)

    with torch.no_grad():
        grad_numerical = np.zeros_like(coords.detach().numpy())
        for i in range(coords.shape[0]):
            for j in range(coords.shape[1]):
                h = 0.01
                coords[i, j] += h
                ene_u = model.computeEnergy(coords, box)['tot']
                coords[i, j] -= 2 * h
                ene_d = model.computeEnergy(coords, box)['tot']
                grad_numerical[i, j] += (ene_u.detach().item() - ene_d.detach().item()) / (2 * h)
                coords[i, j] += h

    print(grad, grad_numerical, sep='\n')    
