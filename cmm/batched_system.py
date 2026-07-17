import warnings
from typing import Dict, List
import torch
import torch.nn as nn

from .ffxml.parametrizer import Parametrizer
from .topology import Topology
from .bonded import (
    computeBondBatch, computeBondFromVecs, 
    computeAngleBatch, computeAngleFromVecs, computeCosAnglePotential, 
    computeBondBondCoupling,
    computeTorsionBatch, computeTorsionFromVecs, computePeriodicTorsionEnergy,
    computeTorsionBondCoupling, computeTorsionAngleAngleCoupling, 
    computeChargeFluxBond, computeChargeFluxAngle, computeChargeFluxBondBond,
    computeHardnessChangeBond, computeHardnessChangeAngle, computeHardnessChangeBondBond,
    computeBondAngleCoupling, computeFieldDependentMorseParams, computeMorseBondPotential
)
from .multipole import (
    computeLocal2GlobalRotationMatrixBatch,
    convertMultipolesToPolytensor,
    scaleMultipoles,
    rotateDipoles, rotateQuadrupoles,
    computeInteractionTensor
)
from .short_range import (
    computeShortRangeEnergyFromPairs, 
    computeShortRangeOneCenterDampFactors,
    computeShortRangeTwoCenterDampFactors,
    computeShortRangePolarizationDampFactors
)
from .dispersion import computeDispersionFromPairs
from .dipole_saturation import (
    unit_field_axis, saturation_energy, saturation_secant_matrix
)
from .electrostatics import computePermanentElectricPotentialExpansion
from .units import HARTREE2KCAL


class BatchedSystem(nn.Module):

    def __init__(
        self,
        top: Topology,
        parametrizers: Dict[str, Parametrizer],
        use_fd_morse: bool = True,
        use_polarization: bool = True,
        use_hardness_change: bool = False,
        use_dipole_saturation: bool = True,
        sat_max_iter: int = 10,
        sat_tol: float = 1e-9,
        **kwargs
    ):
        super().__init__()

        # the parameters in kwargs are not used
        if len(kwargs) > 0:
            warnings.warn(f'The following parameters are not used: {",".join(kwargs.keys())}')

        self.top = top
        self.natoms = self.top.natoms
        self.nbonds = self.top.nbonds
        self.nangles = self.top.nangles
        self.ndihedrals = self.top.ndihedrals

        self._has_torsions = self.ndihedrals > 0

        self.use_fd_morse = use_fd_morse
        self.use_hardness_change = use_hardness_change

        # polarization settings
        self.use_polarization = use_polarization
        self.use_dipole_saturation = use_dipole_saturation
        self.sat_max_iter = sat_max_iter
        self.sat_tol = sat_tol
        self.n_pol_groups = self.top.n_pol_groups
        self.pol_group_indices_a = self.top.pol_group_indices_a
        self.pol_group_segment_indices = self.top.pol_group_segment_indices
        self.pol_group_lengths_g = self.top.pol_group_lengths_g

        # self.parametrizers = parametrizers
        self.parametrizers = nn.ModuleDict(parametrizers)

        self.all_pairs = self.top.getIncludePairs(bidirection=False)

        # indices used to construct polarization tensor
        self.atom_indices = torch.arange(self.natoms, device=self.top.device)
        self._row_indices_1x1 = self.atom_indices * 4

        _col_indices_4x4 = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], device=self.top.device)
        _row_indices_4x4 = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], device=self.top.device)
        
        self._has_nb = self.all_pairs.numel() > 0
        if not self._has_nb:
            self.all_pairs_i = None
            self.all_pairs_j = None
        else:
            self.all_pairs_i = self.all_pairs[:, 0]
            self.all_pairs_j = self.all_pairs[:, 1]
        
            self._row_indices_4x4 = torch.flatten(_row_indices_4x4.expand(self.all_pairs.shape[0], -1) + self.all_pairs_j.reshape(-1, 1) * 4)
            self._col_indices_4x4 = torch.flatten(_col_indices_4x4.expand(self.all_pairs.shape[0], -1) + self.all_pairs_i.reshape(-1, 1) * 4)
    
            self._row_indices_4x4_transpose = torch.flatten(_row_indices_4x4.expand(self.all_pairs.shape[0], -1) + self.all_pairs_i.reshape(-1, 1) * 4)
            self._col_indices_4x4_transpose = torch.flatten(_col_indices_4x4.expand(self.all_pairs.shape[0], -1) + self.all_pairs_j.reshape(-1, 1) * 4)
    
            _row_indices_3x3 = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], device=self.top.device)
            _col_indices_3x3 = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2], device=self.top.device)
    
            self._row_indices_3x3 = torch.flatten(_row_indices_3x3.expand(self.natoms, -1) + self._row_indices_1x1.reshape(-1, 1) + 1)
            self._col_indices_3x3 = torch.flatten(_col_indices_3x3.expand(self.natoms, -1) + self._row_indices_1x1.reshape(-1, 1) + 1)
    
            self._row_indices_constraint = self.top.atom_indices_to_group_indices + self.natoms * 4
            self._col_indices_constraint = self._row_indices_1x1
    
            self._fill_bvec_epot_indices = self._row_indices_1x1
            self._fill_bvec_efield_indices = torch.flatten(torch.tensor([1, 2, 3], device=self.top.device).expand(self.natoms, -1) + self._row_indices_1x1.reshape(-1, 1))

        self.total_keys = [
            "bond", "angle", "torsion", "bond_bond", "bond_angle", "angle_angle", 
            "torsion_bond", "torsion_angle", "torsion_angle_angle",
            "perm_elec", "pauli", "disp", "pol", "ct"
        ]


    def expandParametrizers(self):
        for _, parametrizer in self.parametrizers.items():
            parametrizer.expandParameters()
    
    def getEnergy(self, coords: torch.Tensor, grid: List[torch.Tensor] = list(), energy_in_kcal: bool = False, include_bonded: bool = True, ext_field: torch.Tensor = None):
        '''
        Parameters
        ----------
        coords: torch.Tensor
            Shape (n_bz, n_atoms, 3)
        ext_field: torch.Tensor, optional
            Uniform external electric field in a.u., shape (3,) or (n_bz, 3).
            Added to the permanent field (and potential) driving the
            polarization solve; used for finite-field polarizabilities.
        '''
        self.expandParametrizers()

        nbz = coords.shape[0]
        device = coords.device
        dtype = coords.dtype

        # Charge flux
        charge_flux = torch.zeros((nbz, self.natoms), device=device, dtype=dtype)
        charge_flux_pauli = torch.zeros((nbz, self.natoms), device=device, dtype=dtype)
        hardness_change = torch.ones((nbz, self.natoms), device=device, dtype=dtype)
        hardness_flux = torch.zeros((nbz, self.natoms), device=device, dtype=dtype)
        
        if not self.parametrizers['Bond'].is_empty:
            bondIndices = self.parametrizers['Bond'].getExpandParameters("atomIndices")
            bondVecs = coords[:, bondIndices[:, 1]] - coords[:, bondIndices[:, 0]]
            bonds = computeBondFromVecs(bondVecs)
            j_cf = self.parametrizers['Bond'].getExpandParameters("j_cf")
            j_cf_pauli = self.parametrizers['Bond'].getExpandParameters("j_cf_pauli")
            r_eq = self.parametrizers['Bond'].getExpandParameters("r_eq")
            flux_bond = computeChargeFluxBond(bonds, r_eq, j_cf)
            flux_pauli_bond = computeChargeFluxBond(bonds, r_eq, j_cf_pauli)
            charge_flux.scatter_add_(1, bondIndices[:, 0].expand(nbz, -1), flux_bond[0])
            charge_flux.scatter_add_(1, bondIndices[:, 1].expand(nbz, -1), flux_bond[1])
            charge_flux_pauli.scatter_add_(1, bondIndices[:, 0].expand(nbz, -1), flux_pauli_bond[0])
            charge_flux_pauli.scatter_add_(1, bondIndices[:, 1].expand(nbz, -1), flux_pauli_bond[1])
            
            if self.use_hardness_change:
                k_hardness_b = self.parametrizers['Bond'].getExpandParameters("k_hardness_b")
                hardness_change_bond = computeHardnessChangeBond(bonds, r_eq, k_hardness_b)
                # NOTE(Eric): here can we use in-place operations?
                hardness_change = hardness_change.scatter_reduce(1, bondIndices[:, 1], hardness_change_bond, 'prod')
        
        if not self.parametrizers['Angle'].is_empty:
            angleIndices = self.parametrizers['Angle'].getExpandParameters("atomIndices")
            r_eq_1 = self.parametrizers['Angle'].getExpandParameters("r_eq_1")
            r_eq_2 = self.parametrizers['Angle'].getExpandParameters("r_eq_2")
            theta_eq = self.parametrizers['Angle'].getExpandParameters("theta_eq")
            j_cf_bb = self.parametrizers['Angle'].getExpandParameters("j_cf_bb")
            j_cf_angle = self.parametrizers['Angle'].getExpandParameters("j_cf_angle")
            k_th = self.parametrizers['Angle'].getExpandParameters("k_theta")
            k_bb = self.parametrizers['Angle'].getExpandParameters("k_bb")
            k_ba_1 = self.parametrizers['Angle'].getExpandParameters("k_ba_1")
            k_ba_2 = self.parametrizers['Angle'].getExpandParameters("k_ba_2")
            
            bondVecs_ij = coords[:, angleIndices[:, 0]] - coords[:, angleIndices[:, 1]]
            bondVecs_kj = coords[:, angleIndices[:, 2]] - coords[:, angleIndices[:, 1]]
            r1, r2 = computeBondFromVecs(bondVecs_ij), computeBondFromVecs(bondVecs_kj)
            theta = computeAngleFromVecs(bondVecs_ij, bondVecs_kj)
            
            flux_angle = computeChargeFluxAngle(theta, theta_eq, j_cf_angle)
            flux_bb = computeChargeFluxBondBond(r1, r2, r_eq_1, r_eq_2, j_cf_bb, j_cf_bb)

            charge_flux.scatter_add_(1, angleIndices[:, 0].expand(nbz, -1), flux_angle[0])
            charge_flux.scatter_add_(1, angleIndices[:, 1].expand(nbz, -1), flux_angle[1])
            charge_flux.scatter_add_(1, angleIndices[:, 2].expand(nbz, -1), flux_angle[2])

            charge_flux.scatter_add_(1, angleIndices[:, 0].expand(nbz, -1), flux_bb[0])
            charge_flux.scatter_add_(1, angleIndices[:, 1].expand(nbz, -1), flux_bb[1])
            charge_flux.scatter_add_(1, angleIndices[:, 2].expand(nbz, -1), flux_bb[2])
            charge_flux.scatter_add_(1, angleIndices[:, 1].expand(nbz, -1), flux_bb[3])

            if self.use_hardness_change:
                k_hardness_bb = self.parametrizers['Angle'].getExpandParameters("k_hardness_bb")
                k_hardness_angle = self.parametrizers['Angle'].getExpandParameters("k_hardness_angle")
                hardness_change_bb_1, hardness_change_bb_2 = computeHardnessChangeBondBond(r1, r2, r_eq_1, r_eq_2, k_hardness_bb, k_hardness_bb)
                hardness_flux_angle = computeHardnessChangeAngle(theta, theta_eq, k_hardness_angle)
                hardness_flux.scatter_add_(1, angleIndices[:, 0].expand(nbz, -1), hardness_flux_angle)
                hardness_flux.scatter_add_(1, angleIndices[:, 2].expand(nbz, -1), hardness_flux_angle)
                # NOTE(Eric): again, can we do in-place operations
                hardness_change.scatter_reduce_(1, angleIndices[:, 0].expand(nbz, -1), hardness_change_bb_1, 'prod')
                hardness_change.scatter_reduce_(1, angleIndices[:, 2].expand(nbz, -1), hardness_change_bb_2, 'prod')

            ene_angle = torch.sum(computeCosAnglePotential(theta, theta_eq, k_th), dim=1)
            ene_bb = torch.sum(computeBondBondCoupling(r1, r2, r_eq_1, r_eq_2, k_bb), dim=1)
            ene_ba = torch.sum(computeBondAngleCoupling(r1, r_eq_1, theta, theta_eq, k_ba_1) + computeBondAngleCoupling(r2, r_eq_2, theta, theta_eq, k_ba_2), dim=1)
        else:
            ene_angle = torch.zeros(nbz, device=device, dtype=dtype)
            ene_bb = torch.zeros(nbz, device=device, dtype=dtype)
            ene_ba = torch.zeros(nbz, device=device, dtype=dtype)

        # Torsion
        ene_torsion = torch.zeros(nbz, dtype=dtype, device=device)
        ene_torsion_angle_angle = torch.zeros(nbz, dtype=dtype, device=device)
        if not self.parametrizers['Torsion'].is_empty:
            torsionIndices = self.parametrizers['Torsion'].getExpandParameters("atomIndices")
            teq1 = self.parametrizers['Torsion'].getExpandParameters("theta_eq_1")
            teq2 = self.parametrizers['Torsion'].getExpandParameters("theta_eq_2")

            bondVecs_ij = coords[:, torsionIndices[:, 1]] - coords[:, torsionIndices[:, 0]]
            bondVecs_jk = coords[:, torsionIndices[:, 2]] - coords[:, torsionIndices[:, 1]]
            bondVecs_kl = coords[:, torsionIndices[:, 3]] - coords[:, torsionIndices[:, 2]]

            torsions = computeTorsionFromVecs(bondVecs_ij, bondVecs_jk, bondVecs_kl)
            angles1 = computeAngleFromVecs(-bondVecs_ij, bondVecs_jk)
            angles2 = computeAngleFromVecs(-bondVecs_jk, bondVecs_kl)

            for i in range(4):
                per = self.parametrizers['Torsion'].getExpandParameters(f'per{i+1}')
                phase = self.parametrizers['Torsion'].getExpandParameters(f'phase{i+1}')
                k = self.parametrizers['Torsion'].getExpandParameters(f'k{i+1}')
                k_taa = self.parametrizers['Torsion'].getExpandParameters(f'k_taa_{i+1}')
                ene_torsion += torch.sum(computePeriodicTorsionEnergy(torsions, per, phase, k), dim=1)
                ene_torsion_angle_angle += torch.sum(computeTorsionAngleAngleCoupling(torsions, angles1, angles2, per, phase, k_taa, teq1, teq2).reshape(nbz, -1), dim=1)

        # Torsion-bond coupling
        ene_torsion_bond = torch.zeros(nbz, device=device, dtype=dtype)
        if not self.parametrizers['TorsionBond'].is_empty:
            torsionBondIndices = self.parametrizers['TorsionBond'].getExpandParameters("atomIndices")
            req = self.parametrizers['TorsionBond'].getExpandParameters("r_eq")
            torsions_tb = computeTorsionBatch(coords, torsionBondIndices)
            bonds_tb = computeBondBatch(coords, torsionBondIndices[:, 4:])

            for i in range(4):
                per_tb = self.parametrizers['TorsionBond'].getExpandParameters(f"per{i+1}")
                phase_tb = self.parametrizers['TorsionBond'].getExpandParameters(f"phase{i+1}")
                k_tb = self.parametrizers['TorsionBond'].getExpandParameters(f"k_tb_{i+1}")
                ene_torsion_bond += torch.sum(computeTorsionBondCoupling(torsions_tb, bonds_tb, per_tb, phase_tb, k_tb, req), dim=1)


        # Torsion-angle coupling
        ene_torsion_angle = torch.zeros(nbz, device=device, dtype=dtype)
        if not self.parametrizers['TorsionAngle'].is_empty:
            torsionAngleIndices = self.parametrizers['TorsionAngle'].getExpandParameters("atomIndices")
            teq = self.parametrizers['TorsionAngle'].getExpandParameters("theta_eq")
            torsions_ta = computeTorsionBatch(coords, torsionAngleIndices)
            angles_ta = computeAngleBatch(coords, torsionAngleIndices[:, 4:])
            for i in range(4):
                per_ta = self.parametrizers['TorsionAngle'].getExpandParameters(f"per{i+1}")
                phase_ta = self.parametrizers['TorsionAngle'].getExpandParameters(f"phase{i+1}")
                k_ta = self.parametrizers['TorsionAngle'].getExpandParameters(f"k_ta_{i+1}")
                ene_torsion_angle += torch.sum(computeTorsionBondCoupling(torsions_ta, angles_ta, per_ta, phase_ta, k_ta, teq), dim=1)

        # angle-angle coupling
        if not self.parametrizers['AngleAngle'].is_empty:
            aaIndices = self.parametrizers['AngleAngle'].getExpandParameters('atomIndices')
            theta_eq_1 = self.parametrizers['AngleAngle'].getExpandParameters("theta_eq_1")
            theta_eq_2 = self.parametrizers['AngleAngle'].getExpandParameters("theta_eq_2")
            k_aa = self.parametrizers['AngleAngle'].getExpandParameters("k_aa")
            angles1 = computeAngleBatch(coords, aaIndices[:, :3])
            angles2 = computeAngleBatch(coords, aaIndices[:, -3:])
            ene_angle_angle = torch.sum(computeBondBondCoupling(
                torch.cos(angles1), torch.cos(angles2),
                torch.cos(theta_eq_1), torch.cos(theta_eq_2),
                k_aa
            ), dim=1)
        else:
            ene_angle_angle = torch.zeros(nbz, device=device, dtype=dtype)


        # Nonbonded interactions from this point
        
        ene_elec = torch.zeros(nbz, device=device, dtype=dtype)
        ene_pol = torch.zeros(nbz, device=device, dtype=dtype)
        ene_ct_direct = torch.zeros(nbz, device=device, dtype=dtype)
        ene_ct_indirect = torch.zeros(nbz, device=device, dtype=dtype)
        ene_xpol = torch.zeros(nbz, device=device, dtype=dtype)
        ene_pauli = torch.zeros(nbz, device=device, dtype=dtype)
        ene_disp = torch.zeros(nbz, device=device, dtype=dtype)
        induced_molecular_dipole = torch.zeros((nbz, 3), device=device, dtype=dtype)
        # these two variables are used in evaluate fd-morse
        efield = torch.zeros((self.natoms*nbz, 3), device=device, dtype=dtype)
        dq_a = torch.zeros(self.natoms*nbz, device=device, dtype=dtype)
        dq_groups = torch.zeros((nbz, self.n_pol_groups), device=device, dtype=dtype)


        coords_flatten = coords.reshape(-1, 3)
        expand_to_batch_indices = self.atom_indices.expand(nbz, -1).flatten()
        offset = torch.arange(nbz, device=device) * self.natoms
        offset_indices = (torch.arange(nbz) * self.natoms).repeat_interleave(self.natoms)

        axistypes = self.parametrizers['Multipoles'].getExpandParameters('axistype')[expand_to_batch_indices]
        kzIndices = self.parametrizers['Multipoles'].getExpandParameters('kzIndices')[expand_to_batch_indices] + offset_indices
        kxIndices = self.parametrizers['Multipoles'].getExpandParameters('kxIndices')[expand_to_batch_indices] + offset_indices
        kyIndices = self.parametrizers['Multipoles'].getExpandParameters('kyIndices')[expand_to_batch_indices] + offset_indices

        rotMatrices = computeLocal2GlobalRotationMatrixBatch(coords_flatten, kzIndices, kxIndices, kyIndices, axistypes, None, None)

        mono_raw = self.parametrizers['Multipoles'].getExpandParameters('mono')[expand_to_batch_indices]
        mono = mono_raw + charge_flux.flatten()
        Z = self.parametrizers['ChargePenetration'].getExpandParameters("Z")[expand_to_batch_indices]
        b_elec = self.parametrizers['ChargePenetration'].getExpandParameters("b_elec")[expand_to_batch_indices]

        dipo_raw = self.parametrizers['Multipoles'].getExpandParameters('dipo')[expand_to_batch_indices]
        dipo = rotateDipoles(dipo_raw, rotMatrices).squeeze(1)

        quad_raw = self.parametrizers['Multipoles'].getExpandParameters('quad')[expand_to_batch_indices]
        quad = rotateQuadrupoles(quad_raw, rotMatrices)

        multipoles = convertMultipolesToPolytensor(mono, dipo, quad)

        charges = torch.sum(mono.reshape(nbz, -1), dim=1)
        permanent_dipoles = torch.sum(torch.reshape(mono.reshape(-1, 1) * coords_flatten + dipo, (nbz, -1, 3)), dim=1)

        if self.use_polarization:
            eta = self.parametrizers['Polarization'].getExpandParameters("eta")[expand_to_batch_indices]
            if self.use_hardness_change:
                eta = eta * hardness_change.flatten() + hardness_flux.flatten()
            eta_times_2 = eta * 2
            inv_eta = 1 / eta_times_2

            alpha = self.parametrizers['Polarization'].getExpandParameters("alpha")[expand_to_batch_indices]
            # free-ion polarizabilities: left untouched, saturation enters the solve instead
            polarizabilities = rotateQuadrupoles(alpha, rotMatrices)
            sat_c_iso = self.parametrizers['Polarization'].getExpandParameters("sat_c_iso")[expand_to_batch_indices]
            sat_c_ani = self.parametrizers['Polarization'].getExpandParameters("sat_c_ani")[expand_to_batch_indices]
            sat_w = self.parametrizers['Polarization'].getExpandParameters("sat_w")[expand_to_batch_indices]
            sat_e0 = self.parametrizers['Polarization'].getExpandParameters("sat_e0")[expand_to_batch_indices]
            alpha_iso = torch.diagonal(polarizabilities, dim1=-2, dim2=-1).mean(dim=-1)
            use_sat = self.use_dipole_saturation and bool(
                torch.any(sat_c_iso != 0) or torch.any(sat_c_ani != 0)
            )
            # shape nbz,3,3
            molecular_polarizability = torch.sum(polarizabilities.reshape(nbz, -1, 3, 3), dim=1)
            # shape nbz, 3, 3
            tmp_a = torch.sum((inv_eta.view(-1, 1, 1) * torch.einsum('ni,nj->nij', coords_flatten, coords_flatten)).reshape(nbz, -1, 3, 3), dim=1)
            # shape nbz, 3
            weighted_coords = torch.sum((inv_eta.view(-1, 1) * coords_flatten).reshape(nbz, -1, 3), dim=1)
            tmp_b = torch.einsum('ni,nj->nij', weighted_coords, weighted_coords) / torch.sum(inv_eta.reshape(nbz, -1), dim=1).view(-1, 1, 1)
            molecular_polarizability += tmp_a - tmp_b
        else:
            molecular_polarizability = torch.zeros((nbz, 3, 3), device=device, dtype=dtype)
        
        # electrostatic potential
        grid_epot, grid_efield, grid_efield_grad = [], [], []
        if len(grid) > 0:
            mpoles_cp = convertMultipolesToPolytensor(mono - Z, dipo, quad)
            for index, g in enumerate(grid):
                pairs = torch.cartesian_prod(torch.arange(self.natoms, device=device), torch.arange(g.shape[0], device=device))
                grid_drvecs = g[pairs[:, 1]] - coords[index][pairs[:, 0]]
                slicing = slice(index*nbz*self.natoms, (index+1)*nbz*self.natoms)
                g_epot, g_efield, g_efield_grad = computePermanentElectricPotentialExpansion(
                    g.shape[0], grid_drvecs, pairs, 
                    mpoles_cp[slicing],
                    Z[slicing], b_elec[slicing]
                )
                grid_epot.append(g_epot)
                grid_efield.append(g_efield)
                grid_efield_grad.append(g_efield_grad)
        
        if self._has_nb:
            # indices
            expand_to_batch_indices_pairs = torch.arange(self.all_pairs.shape[0], device=device).expand(nbz, -1).flatten()

            pairs = (self.all_pairs.expand(nbz, -1, -1) + offset.view(-1, 1, 1)).reshape(-1, 2)
            pairs_i = pairs[:, 0]
            pairs_j = pairs[:, 1]

            distVecs = coords_flatten[pairs_j] - coords_flatten[pairs_i]
            dists = torch.norm(distVecs, dim=-1)
            dists_inv = 1 / dists
            switch = torch.ones(dists.shape[0], device=device, dtype=dtype)            

            # Pauli
            pauli_mpoles = scaleMultipoles(
                multipoles,
                self.parametrizers['Pauli'].getExpandParameters("q_pauli")[expand_to_batch_indices] + charge_flux_pauli.flatten(),
                self.parametrizers['Pauli'].getExpandParameters("Kdipo_pauli")[expand_to_batch_indices],
                self.parametrizers['Pauli'].getExpandParameters("Kquad_pauli")[expand_to_batch_indices]
            )
            b_pauli_ij = self.parametrizers['Pauli'].getExpandParameters("b_pauli", self.all_pairs)[expand_to_batch_indices_pairs]
            pauli_pairwise = computeShortRangeEnergyFromPairs(
                dists, distVecs, 
                pauli_mpoles[pairs_i], pauli_mpoles[pairs_j], b_pauli_ij, 
                switch, True, dists_inv
            )
            ene_pauli = torch.sum(pauli_pairwise.reshape(nbz, -1), dim=1)

            # XPol
            xpol_mpoles = scaleMultipoles(
                multipoles,
                self.parametrizers['ExchangePolarization'].getExpandParameters("q_xpol")[expand_to_batch_indices],
                self.parametrizers['ExchangePolarization'].getExpandParameters("Kdipo_xpol")[expand_to_batch_indices],
                self.parametrizers['ExchangePolarization'].getExpandParameters("Kquad_xpol")[expand_to_batch_indices]
            )
            b_xpol_ij = self.parametrizers['ExchangePolarization'].getExpandParameters("b_xpol", self.all_pairs)[expand_to_batch_indices_pairs]
            xpol_pairwise = computeShortRangeEnergyFromPairs(
                dists, distVecs, 
                xpol_mpoles[pairs_i], xpol_mpoles[pairs_j], b_xpol_ij, 
                switch, False, dists_inv
            )
            ene_xpol = torch.sum(xpol_pairwise.reshape(nbz, -1), dim=1)

            # Dispersion
            c6_disp_ij = self.parametrizers['Dispersion'].getExpandParameters("C6_disp", self.all_pairs)[expand_to_batch_indices_pairs]
            b_disp_ij = self.parametrizers['Dispersion'].getExpandParameters("b_disp", self.all_pairs)[expand_to_batch_indices_pairs]
            disp_pairwise = computeDispersionFromPairs(dists, c6_disp_ij, b_disp_ij)
            ene_disp = torch.sum(disp_pairwise.reshape(nbz, -1), dim=1)

            # Charge Transfer
            eps_ct = self.parametrizers['ChargeTransfer'].getExpandParameters("eps_ct", self.all_pairs)[expand_to_batch_indices_pairs]
            ct_acc_mpoles = scaleMultipoles(
                multipoles,
                self.parametrizers['ChargeTransfer'].getExpandParameters("q_ct_acc")[expand_to_batch_indices],
                self.parametrizers['ChargeTransfer'].getExpandParameters("Kdipo_ct_acc")[expand_to_batch_indices],
                self.parametrizers['ChargeTransfer'].getExpandParameters("Kquad_ct_acc")[expand_to_batch_indices]
            )
            ct_don_mpoles = scaleMultipoles(
                multipoles,
                self.parametrizers['ChargeTransfer'].getExpandParameters("q_ct_don")[expand_to_batch_indices],
                self.parametrizers['ChargeTransfer'].getExpandParameters("Kdipo_ct_don")[expand_to_batch_indices],
                self.parametrizers['ChargeTransfer'].getExpandParameters("Kquad_ct_don")[expand_to_batch_indices]
            )
            b_ct_ij = self.parametrizers['ChargeTransfer'].getExpandParameters("b_ct", self.all_pairs)[expand_to_batch_indices_pairs]
            ct_tensor = computeInteractionTensor(
                distVecs,
                -computeShortRangeTwoCenterDampFactors(dists, b_ct_ij),
                dists_inv
            )
            ct_direct_pairwise_ij = torch.bmm(ct_don_mpoles[pairs_j].unsqueeze(1), torch.bmm(ct_tensor, ct_acc_mpoles[pairs_i].unsqueeze(2))).flatten()
            ct_direct_pairwise_ji = torch.bmm(ct_acc_mpoles[pairs_j].unsqueeze(1), torch.bmm(ct_tensor, ct_don_mpoles[pairs_i].unsqueeze(2))).flatten()
            ene_ct_direct = torch.sum((ct_direct_pairwise_ij + ct_direct_pairwise_ji).reshape(nbz, -1), dim=1)

            # compute transferred charges
            drInvDamp_ct = ct_tensor[:, 0, 0].flatten()
            dq_pairwise = (ct_don_mpoles[pairs_i, 0] * ct_acc_mpoles[pairs_j, 0] - ct_acc_mpoles[pairs_i, 0] * ct_don_mpoles[pairs_j, 0]) * drInvDamp_ct * eps_ct
            dq_a.scatter_add_(0, pairs_j, dq_pairwise)
            dq_a.scatter_add_(0, pairs_i, -dq_pairwise)
            dq_groups.scatter_add_(1, self.top.atom_indices_to_group_indices.expand(nbz, -1), dq_a.reshape(nbz, -1))

            # Charge penetration
            b_elec_ij = self.parametrizers['ChargePenetration'].getExpandParameters("b_elec", self.all_pairs)[expand_to_batch_indices_pairs]
            cp_mpoles = convertMultipolesToPolytensor(mono - Z, dipo, quad)

            # shell-shell damped
            cp_mpoles_i, cp_mpoles_j = cp_mpoles[pairs_i], cp_mpoles[pairs_j]
            elec_cp_ss_pairwise = computeShortRangeEnergyFromPairs(
                dists, distVecs, cp_mpoles_i, cp_mpoles_j, b_elec_ij,
                switch, False, dists_inv
            )
            
            # core-shell
            cp_damps_i = -computeShortRangeOneCenterDampFactors(dists, b_elec[pairs_i])
            cp_damps_j = -computeShortRangeOneCenterDampFactors(dists, b_elec[pairs_j])
            cp_tensor_i = computeInteractionTensor(distVecs, cp_damps_i, dists_inv, 2)
            cp_tensor_j = computeInteractionTensor(-distVecs, cp_damps_j, dists_inv, 2)

            edata_cs_pairwise_ij = torch.bmm(cp_tensor_i, cp_mpoles_i.unsqueeze(2)).squeeze(2)
            edata_cs_pairwise_ji = torch.bmm(cp_tensor_j, cp_mpoles_j.unsqueeze(2)).squeeze(2)
            elec_cp_cs_pairwise = edata_cs_pairwise_ij[:, 0] * Z[pairs_j] + edata_cs_pairwise_ji[:, 0] * Z[pairs_i]

            # Multipolar
            realSpaceTensor_ij = computeInteractionTensor(distVecs, [], dists_inv, 2)
            realSpaceTensor_ji = realSpaceTensor_ij.permute(0, 2, 1)
            edata_pairwise_real_ij = torch.bmm(realSpaceTensor_ij, multipoles[pairs_i].unsqueeze(2))
            edata_pairwise_real_ji = torch.bmm(realSpaceTensor_ji, multipoles[pairs_j].unsqueeze(2))
            ene_perm_elec_real_pairwise = torch.bmm(multipoles[pairs_j].unsqueeze(1), edata_pairwise_real_ij).flatten()

            edata = torch.zeros(nbz * self.natoms, 10, device=device, dtype=dtype)
            edata.scatter_add_(0, pairs_j.unsqueeze(1).expand(-1, 10), edata_cs_pairwise_ij)
            edata.scatter_add_(0, pairs_i.unsqueeze(1).expand(-1, 10), edata_cs_pairwise_ji)

            edata.scatter_add_(0, pairs_j.unsqueeze(1).expand(-1, 10), edata_pairwise_real_ij.squeeze(2))
            edata.scatter_add_(0, pairs_i.unsqueeze(1).expand(-1, 10), edata_pairwise_real_ji.squeeze(2))

            epot = edata[:, 0]
            efield = -edata[:, 1:4]

            ene_elec = torch.sum((elec_cp_ss_pairwise + elec_cp_cs_pairwise + ene_perm_elec_real_pairwise).reshape(nbz, -1), dim=1)

            # Polarization
            if self.use_polarization:
                pol_tensor = computeInteractionTensor(
                    distVecs,
                    1-computeShortRangePolarizationDampFactors(dists, b_elec_ij),
                    dists_inv,
                    1
                )

                # optional uniform external field (finite-field polarizabilities)
                efield_tot = efield
                epot_tot = epot
                if ext_field is not None:
                    e_ext = torch.as_tensor(ext_field, device=device, dtype=dtype).reshape(-1, 3)
                    e_ext_a = e_ext.expand(nbz, 3).unsqueeze(1).expand(nbz, self.natoms, 3).reshape(-1, 3)
                    efield_tot = efield + e_ext_a
                    epot_tot = epot - torch.sum(coords_flatten * e_ext_a, dim=-1)

                # fill b-vector
                b_vector = torch.zeros((nbz, self.natoms*4+self.n_pol_groups), device=device, dtype=dtype)
                b_vector[:, self._fill_bvec_epot_indices] = -epot_tot.reshape(nbz, -1)
                b_vector[:, self._fill_bvec_efield_indices] = efield_tot.reshape(nbz, -1)

                # fill A-matrix (free-ion dipole self-block)
                inverse_polarizabilities = torch.inverse(polarizabilities)
                A_matrix = torch.zeros((nbz, self.n_pol_groups+self.natoms*4, self.n_pol_groups+self.natoms*4), device=device, dtype=dtype)
                A_matrix[:, self._row_indices_1x1, self._row_indices_1x1] = eta_times_2.reshape(nbz, -1)
                A_matrix[:, self._row_indices_3x3, self._col_indices_3x3] = inverse_polarizabilities.reshape(nbz, -1)
                A_matrix[:, self._row_indices_4x4, self._col_indices_4x4] = pol_tensor.reshape(nbz, -1)
                A_matrix[:, self._row_indices_4x4_transpose, self._col_indices_4x4_transpose] = pol_tensor.permute(0, 2, 1).reshape(nbz, -1)
                A_matrix[:, self._row_indices_constraint, self._col_indices_constraint] = torch.ones((nbz, self.natoms), dtype=dtype, device=device)
                A_matrix[:, self._col_indices_constraint, self._row_indices_constraint] = torch.ones((nbz, self.natoms), dtype=dtype, device=device)

                sat_axis = unit_field_axis(efield_tot) if use_sat else None

                def solve_polarization(bv):
                    """Solve the (possibly saturated) polarization equations.

                    Returns the solution vector and the variational energy
                    1/2 x^T A0 x - b^T x + sum_i U_sat,i(mu_i), where A0 keeps
                    the free-ion dipole block. With saturation, the dipole
                    self-block is updated with the secant matrix K(mu) until
                    self-consistency, so the fixed point satisfies
                    alpha^-1 mu + grad U_sat(mu) = E_tot exactly.
                    """
                    x = torch.linalg.solve(A_matrix, bv)
                    e_sat = torch.zeros(nbz, device=device, dtype=dtype)
                    if use_sat:
                        for _ in range(self.sat_max_iter):
                            mu = x[:, self._fill_bvec_efield_indices].reshape(-1, 3)
                            K = saturation_secant_matrix(mu, alpha_iso, sat_axis, sat_c_iso, sat_c_ani, sat_w, field_scale=sat_e0)
                            A_sat = A_matrix.clone()
                            A_sat[:, self._row_indices_3x3, self._col_indices_3x3] = (inverse_polarizabilities + K).reshape(nbz, -1)
                            x_new = torch.linalg.solve(A_sat, bv)
                            dmu = torch.max(torch.abs(
                                x_new[:, self._fill_bvec_efield_indices] - x[:, self._fill_bvec_efield_indices]
                            ))
                            x = x_new
                            if dmu < self.sat_tol:
                                break
                        mu = x[:, self._fill_bvec_efield_indices].reshape(-1, 3)
                        e_sat = saturation_energy(mu, alpha_iso, sat_axis, sat_c_iso, sat_c_ani, sat_w, field_scale=sat_e0).reshape(nbz, -1).sum(dim=1)
                    ene = torch.bmm(x.unsqueeze(1), 0.5 * torch.bmm(A_matrix, x.unsqueeze(2)) - bv.unsqueeze(2)).squeeze() + e_sat
                    return x, ene

                solutions, ene_pol = solve_polarization(b_vector)

                b_vector_ct = torch.zeros_like(b_vector)
                b_vector_ct.copy_(b_vector)
                b_vector_ct[:, -self.n_pol_groups:] = dq_groups
                solutions_ct, ene_pol_ct = solve_polarization(b_vector_ct)
                ene_ct_indirect = ene_pol_ct - ene_pol

                # induced molecular dipole P = sum_i (q_i r_i + mu_i) from the
                # converged polarization solution (finite-field polarizability)
                q_ind = solutions[:, self._fill_bvec_epot_indices]
                mu_ind = solutions[:, self._fill_bvec_efield_indices].reshape(nbz, self.natoms, 3)
                induced_molecular_dipole = torch.sum(q_ind.unsqueeze(-1) * coords, dim=1) + torch.sum(mu_ind, dim=1)
        
        # Field-dependent morse 
        if not self.parametrizers['Bond'].is_empty:
            
            bondIndices = self.parametrizers['Bond'].getExpandParameters("atomIndices")
            bondVecs = coords[:, bondIndices[:, 1]] - coords[:, bondIndices[:, 0]]
            bonds = computeBondFromVecs(bondVecs)
                                
            r_eq = self.parametrizers['Bond'].getExpandParameters('r_eq')
            k_b = self.parametrizers['Bond'].getExpandParameters('k_b')
            D = self.parametrizers['Bond'].getExpandParameters('D')

            if self.use_fd_morse:
                dip_deriv_1 = self.parametrizers['Bond'].getExpandParameters('dip_deriv_1')
                dip_deriv_2 = self.parametrizers['Bond'].getExpandParameters('dip_deriv_2')
                ct_slope_1 = self.parametrizers['Bond'].getExpandParameters('ct_slope_1')
                ct_slope_2 = self.parametrizers['Bond'].getExpandParameters('ct_slope_2')
                r_eq_fd, beta_fd = computeFieldDependentMorseParams(
                    bonds, bondVecs,
                    k_b, D, r_eq, dip_deriv_1, dip_deriv_2,
                    ct_slope_1, ct_slope_2,
                    efield.reshape(nbz, -1, 3)[:, bondIndices[:, 1]],
                    dq_a.reshape(nbz, -1)[:, bondIndices[:, 1]]
                )
                ene_bond_list = computeMorseBondPotential(bonds, r_eq_fd, D, beta_fd)
            else:
                beta = torch.sqrt(k_b / 2 / D)
                ene_bond_list = computeMorseBondPotential(bonds, r_eq, D, beta)
            
            ene_bond = torch.sum(ene_bond_list.reshape(nbz, -1), dim=1)
        else:
            ene_bond = torch.zeros(nbz, dtype=dtype, device=device)

        energies = {
            "perm_elec": ene_elec,
            "elec_pol": ene_pol,
            "ct_direct": ene_ct_direct,
            "ct_indirect": ene_ct_indirect,
            "xpol": ene_xpol,
            "pauli": ene_pauli,
            "disp": ene_disp,
            "pol": ene_pol + ene_xpol,
            "ct": ene_ct_indirect + ene_ct_direct,
        }

        if include_bonded:
            energies.update({"bond": ene_bond,
            "angle": ene_angle,
            "torsion": ene_torsion,
            "bond_bond": ene_bb,
            "bond_angle": ene_ba,
            "angle_angle": ene_angle_angle,
            "torsion_bond": ene_torsion_bond,
            "torsion_angle": ene_torsion_angle,
            "torsion_angle_angle": ene_torsion_angle_angle})

        if energy_in_kcal:
            for k in energies:
                energies[k] *= HARTREE2KCAL

        ene_total = torch.zeros(nbz, device=device, dtype=dtype)
        for key in self.total_keys:
            if key in energies:
                ene_total = ene_total + energies[key]
        energies['total'] = ene_total

        energies.update({"charges": charges,
            "dipoles": permanent_dipoles,
            "induced_molecular_dipole": induced_molecular_dipole,
            "grid_epot": grid_epot,
            "grid_efield": grid_efield,
            "grid_efield_grad": grid_efield_grad,
            "polarizability": molecular_polarizability})

        return energies
