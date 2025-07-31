import math
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch_scatter import segment_csr

from .units import BOHR2ANG, HARTREE2KCAL
from .neighbor_list import NeighborList, VerletList
# from .ffxml import ParameterSet
from .ffxml.parametrizer import Parametrizer
from .topology import Topology
from .pbc import applyPBC
from .bonded import (
    computeBond, computeBondFromVecs, 
    computeAngle, computeAngleFromVecs, computeCosAnglePotential, 
    computeBondBondCoupling,
    computeTorsion, computeTorsionFromVecs, computePeriodicTorsionEnergy,
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
from .dispersion import computeDispersionFromPairs, computeLongRangeDispersionCorrection
from .ewald import long_range_potential, long_range_potential_rank_1
from .electrostatics import computeDampFactorsErfc, computeDampFactorsErf
from .polarization import (
    get_field_dependent_polarizabilities, 
    compute_product_with_polarization_matrix, 
    direct_polarization_guess,
    direct_field_induced_dipole_guess
)
from .switching_functions import SwitchFunction
from .polarization_solver import cg_solve, CG, CMMPolarization


class System(nn.Module):

    def __init__(
        self,
        top: Topology,
        parametrizers: Dict[str, Parametrizer],
        expand_parametrizers_during_init: bool = True,
        periodic: bool = True,
        use_fd_morse: bool = True,
        use_ewald: bool = True,
        use_lr_dispersion: bool = True,
        use_polarization: bool = True,
        use_switch: bool = True,
        switch_buffer: float = 2.0,
        ewald_tolerance: float = 1e-6,
        polarization_solver: str = 'cg',
        polarization_max_iteration: int = 400,
        polarization_tolerance: float = 1e-7,
        use_cutoff: bool = True,
        cutoff_vdw: float = 9.0,
        cutoff_ewald: float = 9.0,
        cutoff_sr: float = 5.0,
    ):
        super().__init__()

        self.top = top
        self.natoms = self.top.natoms
        self.nbonds = self.top.nbonds
        self.nangles = self.top.nangles
        self.ndihedrals = self.top.ndihedrals

        self._has_torsions = self.ndihedrals > 0
        
        # base settings
        self.periodic = periodic
        
        if self.periodic:
            self.use_ewald = use_ewald
            self.use_lr_dispersion = use_lr_dispersion
        else:
            self.use_ewald = False
            self.use_lr_dispersion = False

        self.use_fd_morse = use_fd_morse

        # polarization settings
        self.use_polarization = use_polarization
        self.n_pol_groups = self.top.n_pol_groups
        self.pol_group_indices_a = self.top.pol_group_indices_a
        self.pol_group_segment_indices = self.top.pol_group_segment_indices
        self.pol_group_lengths_g = self.top.pol_group_lengths_g

        self.polarization_solver = CMMPolarization(
            self.natoms, 
            self.top.pol_group_indices_a, self.top.pol_group_segment_indices, self.top.pol_group_lengths_g, 
            rtol=polarization_tolerance, atol=0, maxiter=polarization_max_iteration, 
            verbose=False, use_lr=self.use_ewald
        )
        self.polarization_max_iteration = polarization_max_iteration
        self.polarization_tolerance = polarization_tolerance
        
        # cutoff settings
        self.use_cutoff = use_cutoff
        if self.use_cutoff == False:
            # Set a really large cutoff to make sure we capture all interactions
            self.cutoff_vdw = 1000.0
            self.cutoff_ewald = 1000.0
            self.cutoff_sr = 1000.0
            self.cutoff_max = max(self.cutoff_vdw, self.cutoff_sr, self.cutoff_ewald)
        self.cutoff_vdw = cutoff_vdw / BOHR2ANG
        self.cutoff_ewald = cutoff_ewald / BOHR2ANG
        self.cutoff_sr = cutoff_sr / BOHR2ANG
        self.cutoff_max = max(self.cutoff_vdw, self.cutoff_sr, self.cutoff_ewald)

        # switch functions for interactions with cutoff
        self.use_switch = use_switch if self.use_cutoff else False
        self.switch_buf = switch_buffer
        self.switch_func = SwitchFunction(self.use_switch, self.cutoff_sr, self.switch_buf)

        # ewald settings
        self.ewald_tolerance = ewald_tolerance
        self.alpha_ewald = 0.0
        self.k_max = 0
        self._set_ewald = False

        # self.parametrizers = parametrizers
        self.parametrizers = nn.ModuleDict(parametrizers)

        # self.nblist: NeighborList = ...
        self.all_pairs = self.top.getIncludePairs()

        self._has_nb = self.all_pairs.numel() > 0
        if self._has_nb:
            self.all_pairs_i = self.all_pairs[:, 0]
            self.all_pairs_j = self.all_pairs[:, 1]
        else:
            self.all_pairs_i = None
            self.all_pairs_j = None

        # Pairs to exclude
        self.pairs_excl: torch.Tensor = self.top.getExclusionPairs(bidirection=True)
        self.pairs_excl_i = self.pairs_excl[:, 0]
        self.pairs_excl_j = self.pairs_excl[:, 1]

        self.last_induced_multipoles = torch.zeros(0, device=self.top.device)
        self.last_perm_multipoles = torch.zeros(0, device=self.top.device)

        self._expand_parametrizers_during_init = expand_parametrizers_during_init
        if self._expand_parametrizers_during_init:
            self.expandParametrizers()
        else:
            raise NotImplementedError('expand_parametrizers_during_init=False not supported yet')
        
    def expandParametrizers(self):
        for _, parametrizer in self.parametrizers.items():
            parametrizer.expandParameters()
    
    def setEwaldParameters(self, box: torch.Tensor):
        if self._set_ewald:
            return
        maxBoxLen = torch.max(torch.norm(box, dim=1)).item()
        # Find appropriate ewald parameters. This should really be done by the CM.
        self.alpha_ewald = math.sqrt(-math.log10(2 * self.ewald_tolerance)) / self.cutoff_ewald
        self.k_max = 50
        for i in range(2, 50):
            error_estimate = (i * math.sqrt(maxBoxLen * self.alpha_ewald) / 20.0) * math.exp(-torch.pi * torch.pi * i * i / (maxBoxLen * self.alpha_ewald * maxBoxLen * self.alpha_ewald))
            if error_estimate < self.ewald_tolerance:
                self.k_max = i
                break
        if hasattr(self, 'polarization_solver'):
            self.polarization_solver.alpha_ewald = self.alpha_ewald
            self.polarization_solver.k_max = self.k_max
        self._set_ewald = True
    
    def getEnergy(self, coords: torch.Tensor, box: torch.Tensor | None = None):

        # if not self._expand_parametrizers_during_init:
        #     self.expandParametrizers()

        
        boxInv = None if box is None else torch.linalg.inv(box)

        # Charge flux
        charge_flux = torch.zeros(self.natoms, device=coords.device, dtype=coords.dtype)
        charge_flux_pauli = torch.zeros(self.natoms, device=coords.device, dtype=coords.dtype)
        hardness_change = torch.ones(self.natoms, device=coords.device, dtype=coords.dtype)
        hardness_flux = torch.zeros(self.natoms, device=coords.device, dtype=coords.dtype)
        
        if not self.parametrizers['Bond'].is_empty:
            bondIndices = self.parametrizers['Bond'].getExpandParameters("atomIndices")
            bondVecs = applyPBC(coords[bondIndices[:, 1]] - coords[bondIndices[:, 0]], box, boxInv)
            bonds = computeBondFromVecs(bondVecs)
            j_cf = self.parametrizers['Bond'].getExpandParameters("j_cf")
            j_cf_pauli = self.parametrizers['Bond'].getExpandParameters("j_cf_pauli")
            r_eq = self.parametrizers['Bond'].getExpandParameters("r_eq")
            flux_bond = computeChargeFluxBond(bonds, r_eq, j_cf)
            flux_pauli_bond = computeChargeFluxBond(bonds, r_eq, j_cf_pauli)
            charge_flux.scatter_add_(0, bondIndices[:, 0], flux_bond[0])
            charge_flux.scatter_add_(0, bondIndices[:, 1], flux_bond[1])
            charge_flux_pauli.scatter_add_(0, bondIndices[:, 0], flux_pauli_bond[0])
            charge_flux_pauli.scatter_add_(0, bondIndices[:, 1], flux_pauli_bond[1])
            
            k_hardness_b = self.parametrizers['Bond'].getExpandParameters("k_hardness_b")
            hardness_change_bond = computeHardnessChangeBond(bonds, r_eq, k_hardness_b)
            # NOTE(Eric): here can we use in-place operations?
            hardness_change = hardness_change.scatter_reduce(0, bondIndices[:, 1], hardness_change_bond, 'prod')
        
        if not self.parametrizers['Angle'].is_empty:
            angleIndices = self.parametrizers['Angle'].getExpandParameters("atomIndices")
            r_eq_1 = self.parametrizers['Angle'].getExpandParameters("r_eq_1")
            r_eq_2 = self.parametrizers['Angle'].getExpandParameters("r_eq_2")
            theta_eq = self.parametrizers['Angle'].getExpandParameters("theta_eq")
            j_cf_bb = self.parametrizers['Angle'].getExpandParameters("j_cf_bb")
            j_cf_angle = self.parametrizers['Angle'].getExpandParameters("j_cf_angle")
            k_th = self.parametrizers['Angle'].getExpandParameters("k_theta")
            k_bb = self.parametrizers['Angle'].getExpandParameters("k_bb")
            k_hardness_bb = self.parametrizers['Angle'].getExpandParameters("k_hardness_bb")
            k_hardness_angle = self.parametrizers['Angle'].getExpandParameters("k_hardness_angle")
            k_ba_1 = self.parametrizers['Angle'].getExpandParameters("k_ba_1")
            k_ba_2 = self.parametrizers['Angle'].getExpandParameters("k_ba_2")
            
            bondVecs_ij = applyPBC(coords[angleIndices[:, 0]] - coords[angleIndices[:, 1]], box, boxInv)
            bondVecs_kj = applyPBC(coords[angleIndices[:, 2]] - coords[angleIndices[:, 1]], box, boxInv)
            r1, r2 = computeBondFromVecs(bondVecs_ij), computeBondFromVecs(bondVecs_kj)
            theta = computeAngleFromVecs(bondVecs_ij, bondVecs_kj)
            
            flux_angle = computeChargeFluxAngle(theta, theta_eq, j_cf_angle)
            flux_bb = computeChargeFluxBondBond(r1, r2, r_eq_1, r_eq_2, j_cf_bb, j_cf_bb)

            charge_flux.scatter_add_(0, angleIndices[:, 0], flux_angle[0])
            charge_flux.scatter_add_(0, angleIndices[:, 1], flux_angle[1])
            charge_flux.scatter_add_(0, angleIndices[:, 2], flux_angle[2])

            charge_flux.scatter_add_(0, angleIndices[:, 0], flux_bb[0])
            charge_flux.scatter_add_(0, angleIndices[:, 1], flux_bb[1])
            charge_flux.scatter_add_(0, angleIndices[:, 2], flux_bb[2])
            charge_flux.scatter_add_(0, angleIndices[:, 1], flux_bb[3])

            hardness_change_bb_1, hardness_change_bb_2 = computeHardnessChangeBondBond(r1, r2, r_eq_1, r_eq_2, k_hardness_bb, k_hardness_bb)
            hardness_flux_angle = computeHardnessChangeAngle(theta, theta_eq, k_hardness_angle)

            # NOTE(Eric): again, can we do in-place operations
            hardness_change.scatter_reduce(0, angleIndices[:, 0], hardness_change_bb_1, 'prod')
            hardness_change.scatter_reduce(0, angleIndices[:, 2], hardness_change_bb_2, 'prod')

            hardness_flux.scatter_add_(0, angleIndices[:, 0], hardness_flux_angle)
            hardness_flux.scatter_add_(0, angleIndices[:, 2], hardness_flux_angle)

            ene_angle = torch.sum(computeCosAnglePotential(theta, theta_eq, k_th))
            ene_bb = torch.sum(computeBondBondCoupling(r1, r2, r_eq_1, r_eq_2, k_bb))
            ene_ba = torch.sum(computeBondAngleCoupling(r1, r_eq_1, theta, theta_eq, k_ba_1) + computeBondAngleCoupling(r2, r_eq_2, theta, theta_eq, k_ba_2))
        else:
            ene_angle = torch.tensor(0.0, device=coords.device)
            ene_bb = torch.tensor(0.0, device=coords.device)
            ene_ba = torch.tensor(0.0, device=coords.device)

        # Torsion
        if not self.parametrizers['Torsion'].is_empty:
            torsionIndices = self.parametrizers['Torsion'].getExpandParameters("atomIndices")
            # per = self.parametrizers['Torsion'].getExpandParameters("periodicity")
            # phase = self.parametrizers['Torsion'].getExpandParameters("phase")
            # k = self.parametrizers['Torsion'].getExpandParameters("k")
            teq1 = self.parametrizers['Torsion'].getExpandParameters("theta_eq_1")
            teq2 = self.parametrizers['Torsion'].getExpandParameters("theta_eq_2")
            # k_taa = self.parametrizers['Torsion'].getExpandParameters("k_taa")

            bondVecs_ij = applyPBC(coords[torsionIndices[:, 1]] - coords[torsionIndices[:, 0]], box, boxInv)
            bondVecs_jk = applyPBC(coords[torsionIndices[:, 2]] - coords[torsionIndices[:, 1]], box, boxInv)
            bondVecs_kl = applyPBC(coords[torsionIndices[:, 3]] - coords[torsionIndices[:, 2]], box, boxInv)
            torsions = computeTorsionFromVecs(bondVecs_ij, bondVecs_jk, bondVecs_kl)
            angles1 = computeAngleFromVecs(-bondVecs_ij, bondVecs_jk)
            angles2 = computeAngleFromVecs(-bondVecs_jk, bondVecs_kl)

            ene_torsion = torch.zeros(torsionIndices.shape[0], dtype=coords.dtype, device=coords.device)
            ene_torsion_angle_angle = torch.zeros(torsionIndices.shape[0], dtype=coords.dtype, device=coords.device)

            for i in range(4):
                per = self.parametrizers['Torsion'].getExpandParameters(f'per{i+1}')
                phase = self.parametrizers['Torsion'].getExpandParameters(f'phase{i+1}')
                k = self.parametrizers['Torsion'].getExpandParameters(f'k{i+1}')
                k_taa = self.parametrizers['Torsion'].getExpandParameters(f'k_taa_{i+1}')
                ene_torsion += computePeriodicTorsionEnergy(torsions, per, phase, k)
                ene_torsion_angle_angle += computeTorsionAngleAngleCoupling(torsions, angles1, angles2, per, phase, k_taa, teq1, teq2)
            
            ene_torsion = torch.sum(ene_torsion)
            ene_torsion_angle_angle = torch.sum(ene_torsion_angle_angle)
        else:
            ene_torsion = torch.tensor(0.0, device=coords.device)
            ene_torsion_angle_angle = torch.tensor(0.0, device=coords.device)

        # Torsion-bond coupling
        if not self.parametrizers['TorsionBond'].is_empty:
            torsionBondIndices = self.parametrizers['TorsionBond'].getExpandParameters("atomIndices")
            req = self.parametrizers['TorsionBond'].getExpandParameters("r_eq")
            torsions_tb = computeTorsion(coords, torsionBondIndices, box, boxInv)
            bonds_tb = computeBond(coords, torsionBondIndices[:, 4:], box, boxInv)

            ene_torsion_bond = torch.zeros(torsionBondIndices.shape[0], device=coords.device, dtype=coords.dtype)
            for i in range(4):
                per_tb = self.parametrizers['TorsionBond'].getExpandParameters(f"per{i+1}")
                phase_tb = self.parametrizers['TorsionBond'].getExpandParameters(f"phase{i+1}")
                k_tb = self.parametrizers['TorsionBond'].getExpandParameters(f"k_tb_{i+1}")
                ene_torsion_bond += computeTorsionBondCoupling(torsions_tb, bonds_tb, per_tb, phase_tb, k_tb, req)
            ene_torsion_bond = torch.sum(ene_torsion_bond)
        else:
            ene_torsion_bond = torch.tensor(0.0, device=coords.device)

        # Torsion-angle coupling
        if not self.parametrizers['TorsionAngle'].is_empty:
            torsionAngleIndices = self.parametrizers['TorsionAngle'].getExpandParameters("atomIndices")
            teq = self.parametrizers['TorsionAngle'].getExpandParameters("theta_eq")
            torsions_ta = computeTorsion(coords, torsionAngleIndices, box, boxInv)
            angles_ta = computeAngle(coords, torsionAngleIndices[:, 4:], box, boxInv)
            ene_torsion_angle = torch.zeros(torsionAngleIndices.shape[0], device=coords.device, dtype=coords.dtype)
            for i in range(4):
                per_ta = self.parametrizers['TorsionAngle'].getExpandParameters(f"per{i+1}")
                phase_ta = self.parametrizers['TorsionAngle'].getExpandParameters(f"phase{i+1}")
                k_ta = self.parametrizers['TorsionAngle'].getExpandParameters(f"k_ta_{i+1}")
                ene_torsion_angle += computeTorsionBondCoupling(torsions_ta, angles_ta, per_ta, phase_ta, k_ta, teq)
            ene_torsion_angle = torch.sum(ene_torsion_angle)
        else:
            ene_torsion_angle = torch.tensor(0.0, device=coords.device)

        # angle-angle coupling
        if not self.parametrizers['AngleAngle'].is_empty:
            aaIndices = self.parametrizers['AngleAngle'].getExpandParameters('atomIndices')
            theta_eq_1 = self.parametrizers['AngleAngle'].getExpandParameters("theta_eq_1")
            theta_eq_2 = self.parametrizers['AngleAngle'].getExpandParameters("theta_eq_2")
            k_aa = self.parametrizers['AngleAngle'].getExpandParameters("k_aa")
            angles1 = computeAngle(coords, aaIndices[:, :3], box, boxInv)
            angles2 = computeAngle(coords, aaIndices[:, -3:], box, boxInv)
            ene_angle_angle = torch.sum(computeBondBondCoupling(
                torch.cos(angles1), torch.cos(angles2),
                torch.cos(theta_eq_1), torch.cos(theta_eq_2),
                k_aa
            ))
        else:
            ene_angle_angle = torch.tensor(0.0, device=coords.device)

        # Nonbonded interactions from this point
        if not self._has_nb:
            ene_elec = torch.tensor(0.0, device=coords.device)
            ene_pol = torch.tensor(0.0, device=coords.device)
            ene_ct_direct = torch.tensor(0.0, device=coords.device)
            ene_xpol = torch.tensor(0.0, device=coords.device)
            ene_pauli = torch.tensor(0.0, device=coords.device)
            ene_disp = torch.tensor(0.0, device=coords.device)
            # these two variables are used in evaluate fd-morse
            efield = torch.zeros((self.natoms, 3), device=coords.device, dtype=coords.dtype)
            dq_a = torch.zeros(self.natoms, device=coords.device, dtype=coords.dtype)
        else:
            # pairs
            all_distVecs = applyPBC(coords[self.all_pairs_j] - coords[self.all_pairs_i], box, boxInv)
            all_dists = torch.norm(all_distVecs, dim=1)

            if self.use_cutoff:
                mask = all_dists < self.cutoff_max
            else:
                mask = torch.ones_like(all_dists, device=all_dists.device, dtype=torch.bool)

            pairs = self.all_pairs[mask]
            pairs_i = pairs[:, 0]
            pairs_j = pairs[:, 1]

            distVecs = all_distVecs[mask]
            dists = all_dists[mask]
            dists_inv = 1 / dists

            # masks
            mask_nb = torch.ones_like(dists, device=dists.device, dtype=torch.bool)

            mask_sr = torch.logical_and(dists < self.cutoff_sr, mask_nb)
            mask_vdw = torch.logical_and(dists < self.cutoff_vdw, mask_nb)
            mask_ewald = torch.logical_and(dists < self.cutoff_ewald, mask_nb)
            switch_sr = self.switch_func(dists) * mask_nb

            axistypes = self.parametrizers['Multipoles'].getExpandParameters('axistype')
            kzIndices = self.parametrizers['Multipoles'].getExpandParameters('kzIndices')
            kxIndices = self.parametrizers['Multipoles'].getExpandParameters('kxIndices')
            kyIndices = self.parametrizers['Multipoles'].getExpandParameters('kxIndices')
            rotMatrices = computeLocal2GlobalRotationMatrixBatch(coords, kzIndices, kxIndices, kyIndices, axistypes, box, boxInv)

            mono = self.parametrizers['Multipoles'].getExpandParameters('mono') + charge_flux
            dipo = rotateDipoles(self.parametrizers['Multipoles'].getExpandParameters('dipo'), rotMatrices).squeeze(1)
            quad = rotateQuadrupoles(self.parametrizers['Multipoles'].getExpandParameters('quad'), rotMatrices)

            multipoles = convertMultipolesToPolytensor(mono, dipo, quad)
            self.last_perm_multipoles = multipoles

            # Pauli
            pauli_mpoles = scaleMultipoles(
                multipoles,
                self.parametrizers['Pauli'].getExpandParameters("q_pauli") + charge_flux_pauli,
                self.parametrizers['Pauli'].getExpandParameters("Kdipo_pauli"),
                self.parametrizers['Pauli'].getExpandParameters("Kquad_pauli")
            )
            b_pauli_ij = self.parametrizers['Pauli'].getExpandParameters("b_pauli", pairs)
            pauli_pairwise = computeShortRangeEnergyFromPairs(
                dists, distVecs, 
                pauli_mpoles[pairs_i], pauli_mpoles[pairs_j], b_pauli_ij, 
                switch_sr, True, dists_inv
            )
            ene_pauli = 0.5 * torch.sum(pauli_pairwise)

            # ene_elec = torch.tensor(0.0, device=coords.device)
            # ene_xpol = torch.tensor(0.0, device=coords.device)
            ene_pol = torch.tensor(0.0, device=coords.device)
            # ene_ct_direct = torch.tensor(0.0, device=coords.device)
            # ene_disp = torch.tensor(0.0, device=coords.device)
            # efield = torch.zeros((self.natoms, 3), device=coords.device, dtype=coords.dtype)
            # dq_a = torch.zeros(self.natoms, device=coords.device, dtype=coords.dtype)

            # XPol
            xpol_mpoles = scaleMultipoles(
                multipoles,
                self.parametrizers['ExchangePolarization'].getExpandParameters("q_xpol"),
                self.parametrizers['ExchangePolarization'].getExpandParameters("Kdipo_xpol"),
                self.parametrizers['ExchangePolarization'].getExpandParameters("Kquad_xpol")
            )
            b_xpol_ij = self.parametrizers['ExchangePolarization'].getExpandParameters("b_xpol", pairs)
            xpol_pairwise = computeShortRangeEnergyFromPairs(
                dists, distVecs, 
                xpol_mpoles[pairs_i], xpol_mpoles[pairs_j], b_xpol_ij, 
                switch_sr, False, dists_inv
            )
            ene_xpol = 0.5 * torch.sum(xpol_pairwise)

            # Dispersion
            c6_disp_ij = self.parametrizers['Dispersion'].getExpandParameters("C6_disp", pairs)
            b_disp_ij = self.parametrizers['Dispersion'].getExpandParameters("b_disp", pairs)
            disp_pairwise = computeDispersionFromPairs(dists, c6_disp_ij, b_disp_ij)
            ene_disp = 0.5 * torch.sum(disp_pairwise * mask_vdw)

            if self.use_lr_dispersion and box is not None:
                boxV = torch.linalg.det(box)
                ene_disp = ene_disp + computeLongRangeDispersionCorrection(c6_disp_ij, self.cutoff_vdw, self.natoms, boxV)
            
            # Charge penetration parameters
            Z = self.parametrizers['ChargePenetration'].getExpandParameters("Z")
            b_elec = self.parametrizers['ChargePenetration'].getExpandParameters("b_elec")
            b_elec_ij = self.parametrizers['ChargePenetration'].getExpandParameters("b_elec", pairs)
            cp_mpoles = convertMultipolesToPolytensor(mono - Z, dipo, quad)

            # Polarization parameters
            eta = self.parametrizers['Polarization'].getExpandParameters("eta") * hardness_change + hardness_flux
            eta_times_2 = eta * 2

            alpha = self.parametrizers['Polarization'].getExpandParameters("alpha")
            alpha_damp_exponent = self.parametrizers['Polarization'].getExpandParameters("alpha_damp_exponent")
            alpha_damp_max = self.parametrizers['Polarization'].getExpandParameters("alpha_damp_max")

            # Charge Transfer
            eps_ct = self.parametrizers['ChargeTransfer'].getExpandParameters("eps_ct", pairs)
            ct_acc_mpoles = scaleMultipoles(
                multipoles,
                self.parametrizers['ChargeTransfer'].getExpandParameters("q_ct_acc"),
                self.parametrizers['ChargeTransfer'].getExpandParameters("Kdipo_ct_acc"),
                self.parametrizers['ChargeTransfer'].getExpandParameters("Kquad_ct_acc")
            )
            ct_don_mpoles = scaleMultipoles(
                multipoles,
                self.parametrizers['ChargeTransfer'].getExpandParameters("q_ct_don"),
                self.parametrizers['ChargeTransfer'].getExpandParameters("Kdipo_ct_don"),
                self.parametrizers['ChargeTransfer'].getExpandParameters("Kquad_ct_don")
            )
            b_ct_ij = self.parametrizers['ChargeTransfer'].getExpandParameters("b_ct", pairs)
            ct_tensor = computeInteractionTensor(
                distVecs,
                -computeShortRangeTwoCenterDampFactors(dists, b_ct_ij),
                dists_inv
            )
            ct_direct_pairwise_ij = torch.bmm(ct_don_mpoles[pairs_j].unsqueeze(1), torch.bmm(ct_tensor, ct_acc_mpoles[pairs_i].unsqueeze(2))).flatten()
            ct_direct_pairwise_ji = torch.bmm(ct_acc_mpoles[pairs_j].unsqueeze(1), torch.bmm(ct_tensor, ct_don_mpoles[pairs_i].unsqueeze(2))).flatten()
            ene_ct_direct = 0.5 * torch.sum((ct_direct_pairwise_ij + ct_direct_pairwise_ji) * switch_sr)

            # compute transferred charges
            drInvDamp_ct = ct_tensor[:, 0, 0].flatten()
            dq_forward = ct_don_mpoles[pairs_i][:, 0] * ct_acc_mpoles[pairs_j][:, 0] * drInvDamp_ct * eps_ct
            dq_backward = ct_acc_mpoles[pairs_i][:, 0] * ct_don_mpoles[pairs_j][:, 0] * drInvDamp_ct * eps_ct
            dq_pairwise = (dq_forward - dq_backward) * switch_sr
            dq_a = torch.zeros(self.natoms, device=pairs.device, dtype=coords.dtype)
            # dq_groups = torch.zeros(self.n_pol_groups, device=pairs.device, dtype=coords.dtype)
            dq_a = dq_a.scatter_add(0, pairs_j, dq_pairwise)
            dq_groups = segment_csr(dq_a[self.pol_group_indices_a], self.pol_group_segment_indices, reduce='sum')

            # Charge penetration
            # shell-shell
            cp_mpoles_i, cp_mpoles_j = cp_mpoles[pairs_i], cp_mpoles[pairs_j]
            elec_cp_ss_pairwise = computeShortRangeEnergyFromPairs(
                dists, distVecs, cp_mpoles_i, cp_mpoles_j, b_elec_ij,
                switch_sr, False, dists_inv
            )
            
            # core-shell
            # TODO(Eric): there might be some duplicated calculation if the pairs are bi-directional
            cp_damps_i = -computeShortRangeOneCenterDampFactors(dists, b_elec[pairs_i])
            cp_damps_j = -computeShortRangeOneCenterDampFactors(dists, b_elec[pairs_j])
            cp_tensor_i = computeInteractionTensor(distVecs, cp_damps_i, dists_inv, 2)
            cp_tensor_j = computeInteractionTensor(distVecs, cp_damps_j, dists_inv, 2)

            Z_mpoles = torch.zeros_like(multipoles)
            Z_mpoles[:, 0] += Z

            edata_cs_pairwise_ij = torch.bmm(cp_tensor_i, cp_mpoles_i.unsqueeze(2)) * mask_nb[:, None, None]
            elec_cs_pairwise_ji = torch.bmm(cp_mpoles_j.unsqueeze(1), torch.bmm(cp_tensor_j, Z_mpoles[pairs_i].unsqueeze(2))).flatten()
            elec_cs_pairwise_ij = torch.bmm(Z_mpoles[pairs_j].unsqueeze(1), edata_cs_pairwise_ij).flatten()
            elec_cp_pairwise = elec_cp_ss_pairwise + elec_cs_pairwise_ji + elec_cs_pairwise_ij
            ene_elec_cp = 0.5 * torch.sum(elec_cp_pairwise * switch_sr)

            if self.use_ewald and box is not None:
                # Recip-space electrostatics
                self.setEwaldParameters(box)
                ewald_potential, ewald_field, ewald_field_gradient = long_range_potential(coords, mono, dipo, quad, box, self.alpha_ewald, self.k_max)
                ene_perm_elec_recip_raw = 0.5 * (
                    torch.einsum("n,n->", mono, ewald_potential) -
                    torch.einsum("ni,ni->", dipo, ewald_field) -
                    torch.einsum("nij,nij->", quad, ewald_field_gradient) / 3
                )
                # substract excluded pairs
                drVecs_excl = applyPBC(coords[self.pairs_excl_j] - coords[self.pairs_excl_i], box, boxInv)
                dr_excl = torch.norm(drVecs_excl, dim=1)
                erfDamps = -computeDampFactorsErf(dr_excl, self.alpha_ewald)
                realSpaceTensor_excl = computeInteractionTensor(drVecs_excl, erfDamps)
                realSpaceTensor_excl_pol = realSpaceTensor_excl[:, :4, :4]
                edata_pairwise_excl = torch.bmm(realSpaceTensor_excl, multipoles[self.pairs_excl_i].unsqueeze(2))
                ene_perm_elec_recip_excl = 0.5 * torch.sum(torch.bmm(multipoles[self.pairs_excl_j].unsqueeze(1), edata_pairwise_excl).flatten())
                ene_perm_elec_recip = ene_perm_elec_recip_raw + ene_perm_elec_recip_excl
            else:
                realSpaceTensor_excl_pol = None

            # Real-space electrostatics
            erfcDamps = computeDampFactorsErfc(dists, self.alpha_ewald)
            realSpaceTensor = computeInteractionTensor(distVecs, erfcDamps, dists_inv, 2)
            edata_pairwise_real = torch.bmm(realSpaceTensor, multipoles[pairs_i].unsqueeze(2)) * mask_ewald[:, None, None]
            ene_perm_elec_real = 0.5 * torch.sum(torch.bmm(multipoles[pairs_j].unsqueeze(1), edata_pairwise_real).flatten())

            # accumulate electric potential and field
            edata = torch.zeros(self.natoms, 10, device=dists.device, dtype=dists.dtype)
            index = pairs_j.unsqueeze(1).expand(-1, 10)
            edata = edata.scatter_add(0, index, edata_pairwise_real.squeeze(2))
            edata = edata.scatter_add(0, index, edata_cs_pairwise_ij.squeeze(2))
            
            if self.use_ewald:
                edata = edata.scatter_add(0, self.pairs_excl_j.unsqueeze(1).expand(-1, 10), edata_pairwise_excl.squeeze(2))
            epot = edata[:, 0]
            efield = -edata[:, 1:4]
            if self.use_ewald:
                epot = epot + ewald_potential
                efield = efield + ewald_field

            ene_elec = ene_elec_cp + ene_perm_elec_real
            if self.use_ewald:
                ene_elec = ene_elec + ene_perm_elec_recip

            if self.use_polarization:
                polarizabilities = rotateQuadrupoles(alpha, rotMatrices)
                polarizabilities = get_field_dependent_polarizabilities(polarizabilities, efield, alpha_damp_exponent, alpha_damp_max)
                inverse_polarizabilities = torch.linalg.inv(polarizabilities)

                pol_tensor = computeInteractionTensor(
                    distVecs,
                    -computeShortRangePolarizationDampFactors(dists, b_elec_ij),
                    dists_inv,
                    1
                )
                
                b_vector = torch.hstack((-epot, efield.flatten(), dq_groups))
                with torch.no_grad():
                    # Evaluate the initial guess #
                    if self.last_induced_multipoles.numel() == 0:
                        self.last_induced_multipoles = direct_field_induced_dipole_guess(self.natoms, self.n_pol_groups, polarizabilities, efield)
                    
                    self.last_induced_multipoles = self.polarization_solver(
                        coords,
                        box,
                        b_vector,
                        self.last_induced_multipoles,
                        pairs_i[mask_ewald], pairs_j[mask_ewald],
                        pairs_i[mask_sr], pairs_j[mask_sr],
                        self.pairs_excl_i, self.pairs_excl_j,
                        realSpaceTensor[mask_ewald, :4, :4],
                        pol_tensor[mask_sr],
                        realSpaceTensor_excl_pol,
                        eta_times_2,
                        polarizabilities,
                        inverse_polarizabilities
                    )
                
                tmp = self.polarization_solver.compute_product_with_polarization_matrix(
                        coords, box,
                        self.last_induced_multipoles,
                        pairs_i[mask_ewald], pairs_j[mask_ewald],
                        pairs_i[mask_sr], pairs_j[mask_sr],
                        self.pairs_excl_i, self.pairs_excl_j,
                        realSpaceTensor[mask_ewald, :4, :4],
                        pol_tensor[mask_sr],
                        realSpaceTensor_excl_pol,
                        eta_times_2,
                        inverse_polarizabilities
                )
                ene_pol = torch.dot(self.last_induced_multipoles, (0.5 * tmp - b_vector))
                self.last_induced_multipoles = self.last_induced_multipoles.detach().clone()
            else:
                ene_pol = torch.tensor(0.0, device=coords.device)
            #else:
            #    raise NotImplementedError('No ewald is not supported')
            #    eTensor = computeInteractionTensor(distVecs, None, dists_inv, 2)
            #    edata_pairwise_mpoles = torch.bmm(eTensor, multipoles[pairs_i].unsqueeze(2))
            #    ene_perm_elec_mpoles = 0.5 * torch.sum(torch.bmm(multipoles[pairs_j].unsqueeze(1), edata_pairwise_mpoles).flatten())
            #    ene_elec = ene_elec_cp + ene_perm_elec_mpoles
            #    if self.use_polarization:
            #            # accumulate electric potential and field
            #        edata = torch.zeros(self.natoms, 10, device=dists.device, dtype=dists.dtype, requires_grad=True)
            #        index = pairs_j.unsqueeze(1).expand(-1, 10)
            #        edata = edata.scatter_add(0, index, edata_pairwise_mpoles.squeeze(2))
            #        edata = edata.scatter_add(0, index, edata_cs_pairwise_ij.squeeze(2))
            #        epot = edata[:, 0]
            #        efield = -edata[:, 1:4]
            #        b_vector = torch.hstack((-epot, efield.flatten(), dq_groups))
    #
            #        polDamps_i = 1 - computeShortRangePolarizationDampFactors(dists, b_elec_ij)
            #        polTensor = computeInteractionTensor(distVecs, polDamps_i, rank=1)
    #
            #        dimA = self.top.natoms * 4 + self.top.n_pol_groups
            #        A_matrix = torch.zeros((dimA, dimA), dtype=dists.dtype, device=dists.device)
            #        arange = torch.arange(self.top.natoms, device=A_matrix.device)
            #        A_matrix[arange, arange] += eta_times_2
    #
            #        polarizabilities = rotateQuadrupoles(alpha, rotMatrices)
            #        polarizabilities = get_field_dependent_polarizabilities(polarizabilities, efield, alpha_damp_exponent, alpha_damp_max)
            #        inverse_polarizabilities = torch.linalg.inv(polarizabilities)
            #        A_matrix[self.top.natoms:self.top.natoms*4, self.top.natoms:self.top.natoms*4] = torch.block_diag(*inverse_polarizabilities.unbind(0))
    #
            #        # for i, (ai, aj) in enumerate(zip(pairs[0], pairs[1])):
            #        #     # dipo-dipo 
            #        #     matA[aj*3+offset: (aj+1)*3+offset, ai*3+offset: (ai+1)*3+offset] += polTensor[i, -3:, -3:]
            #        #     # charge-charge
            #        #     matA[aj, ai] += polTensor[i, 0, 0]
            #        #     # charge-dipo
            #        #     matA[aj, ai*3+offset:(ai+1)*3+offset] += polTensor[i, 0, -3:]
            #        #     matA[aj*3+offset:(aj+1)*3+offset, ai] += polTensor[i, -3:, 0]
            #else:
            #    ene_pol = torch.tensor(0.0, device=coords.device)
        

        # Field-dependent morse
        if not self.parametrizers['Bond'].is_empty:
            bondIndices = self.parametrizers['Bond'].getExpandParameters("atomIndices")
            bondVecs = applyPBC(coords[bondIndices[:, 1]] - coords[bondIndices[:, 0]], box, boxInv)
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
                    efield[bondIndices[:, 1]],
                    dq_a[bondIndices[:, 1]]
                )
                ene_bond_list = computeMorseBondPotential(bonds, r_eq_fd, D, beta_fd)
            else:
                beta = torch.sqrt(k_b / 2 / D)
                ene_bond_list = computeMorseBondPotential(bonds, r_eq, D, beta)
            
            ene_bond = torch.sum(ene_bond_list)
        else:
            ene_bond = torch.tensor(0.0, device=coords.device)


        energies = {
            "bond": ene_bond,
            "angle": ene_angle,
            "torsion": ene_torsion,
            "bond_bond": ene_bb,
            "bond_angle": ene_ba,
            "angle_angle": ene_angle_angle,
            "torsion_bond": ene_torsion_bond,
            "torsion_angle": ene_torsion_angle,
            "torsion_angle_angle": ene_torsion_angle_angle,
            "perm_elec": ene_elec,
            "pol": ene_pol,
            "ct_direct": ene_ct_direct,
            "xpol": ene_xpol,
            "pauli": ene_pauli,
            "disp": ene_disp
        }

        ene_total = torch.tensor(0.0, device=coords.device)
        for key in energies.values():
            ene_total = ene_total + key
        energies['total'] = ene_total

        return energies