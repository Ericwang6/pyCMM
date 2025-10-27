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
    computeInteractionTensor,
)
from .short_range import (
    computeShortRangeEnergyFromPairs, 
    computeShortRangeOneCenterDampFactors,
    computeShortRangeTwoCenterDampFactors,
    computeShortRangePolarizationDampFactors
)
from .dispersion import computeDispersionFromPairs, computeLongRangeDispersionCorrection
from .ewald import Ewald
from .electrostatics import computeDampFactorsErfc, computeDampFactorsErf
from .polarization import get_field_dependent_polarizabilities, CMMPolarization
from .switching_functions import SwitchFunction

import time, os
from contextlib import contextmanager

PROFILE = int(os.environ.get('CMM_PROFILE', 0))

if PROFILE:
    @contextmanager
    def timer(name: str = ''):
        torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        torch.cuda.synchronize()
        elapsed = 1000 * (time.perf_counter() - start)
        if name:
            print(f"[{name}] elapsed: {elapsed:.6f} ms")
        else:
            print(f"Elapsed: {elapsed:.6f} ms")
else:
    @contextmanager
    def timer(name: str = ''):
        yield


try:
    import torchff
    import torchff_cmm
    import torchff_multipoles
except ImportError:
    pass


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
        use_hardness_change: bool = True,
        switch_buffer: float = 2.0,
        ewald_tolerance: float = 1e-6,
        polarization_solver: str = 'cg',
        polarization_max_iteration: int = 400,
        polarization_tolerance: float = 1e-7,
        use_cutoff: bool = True,
        cutoff_lr: float = 9.0,
        cutoff_sr: float = 5.0,
        use_customized_ops: bool = False
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
        self.use_hardness_change = use_hardness_change

        # cutoff settings
        self.use_cutoff = use_cutoff
        self.cutoff_lr = cutoff_lr / BOHR2ANG
        self.cutoff_sr = cutoff_sr / BOHR2ANG
        self.cutoff_max = max(self.cutoff_lr, self.cutoff_sr)

        # polarization settings
        self.use_polarization = use_polarization
        self.n_pol_groups = self.top.n_pol_groups
        self.pol_group_indices_a = self.top.pol_group_indices_a
        self.pol_group_segment_indices = self.top.pol_group_segment_indices
        self.pol_group_lengths_g = self.top.pol_group_lengths_g

        self.polarization_solver = CMMPolarization(
            self.natoms, 
            self.top.pol_group_indices_a, self.top.pol_group_segment_indices, self.top.pol_group_lengths_g, 
            rcut_sr=self.cutoff_sr, rcut_lr=self.cutoff_lr,
            rtol=polarization_tolerance, atol=0, maxiter=polarization_max_iteration, 
            verbose=PROFILE, use_lr=True, use_customized_ops=use_customized_ops
        )
        self.polarization_solver.to(device=top.device, dtype=torch.get_default_dtype())
        self.polarization_max_iteration = polarization_max_iteration
        self.polarization_tolerance = polarization_tolerance

        # switch functions for interactions with cutoff
        self.use_switch = use_switch if self.use_cutoff else False
        self.switch_buf = switch_buffer
        self.switch_func_sr = SwitchFunction(self.use_switch, self.cutoff_sr, self.switch_buf)
        self.switch_func_lr = SwitchFunction(self.use_switch, self.cutoff_lr, self.switch_buf)

        # ewald settings
        self.ewald_tolerance = ewald_tolerance
        self.alpha_ewald = 0.0
        self.k_max = 0
        self._set_ewald = False

        # self.parametrizers = parametrizers
        self.parametrizers = nn.ModuleDict(parametrizers)

        # self.nblist: NeighborList = ...
        self.all_pairs = self.top.getIncludePairs(bidirection=False)
        
        self._has_nb = self.all_pairs.numel() > 0
        if self._has_nb:
            self.all_pairs_i = self.all_pairs[:, 0]
            self.all_pairs_j = self.all_pairs[:, 1]
        else:
            self.all_pairs_i = None
            self.all_pairs_j = None

        # Pairs to exclude
        self.pairs_excl: torch.Tensor = self.top.getExclusionPairs(bidirection=False)
        self.pairs_i_excl = self.pairs_excl[:, 0]
        self.pairs_j_excl = self.pairs_excl[:, 1]
        self.pairs_excl_bidir: torch.Tensor = self.top.getExclusionPairs(bidirection=True)
        self.pairs_i_excl_bidir = self.pairs_excl_bidir[:, 0]
        self.pairs_j_excl_bidir = self.pairs_excl_bidir[:, 1]

        self.last_induced_multipoles = torch.zeros(0, device=self.top.device)
        self.last_perm_multipoles = torch.zeros(0, device=self.top.device)

        self._expand_parametrizers_during_init = expand_parametrizers_during_init
        if self._expand_parametrizers_during_init:
            self.expandParametrizers()
        else:
            raise NotImplementedError('expand_parametrizers_during_init=False not supported yet')
        
        # Long-Range dispersion correction
        if self.use_lr_dispersion and self._has_nb:
            self.c6_mean = torch.mean(self.parametrizers['Dispersion'].getExpandParameters("C6_disp", self.all_pairs))
        
        self.use_customized_ops = use_customized_ops
        if self.use_customized_ops:
            assert self.use_ewald, "Must use ewald when using customized ops"
        
    def expandParametrizers(self):
        for _, parametrizer in self.parametrizers.items():
            parametrizer.expandParameters()
    
    def setEwaldParameters(self, box: torch.Tensor):
        if self._set_ewald:
            return
        maxBoxLen = torch.max(torch.norm(box, dim=1)).item()
        # Find appropriate ewald parameters. This should really be done by the CM.
        self.alpha_ewald = math.sqrt(-math.log10(2 * self.ewald_tolerance)) / self.cutoff_lr
        self.k_max = 50
        for i in range(2, 50):
            error_estimate = (i * math.sqrt(maxBoxLen * self.alpha_ewald) / 20.0) * math.exp(-torch.pi * torch.pi * i * i / (maxBoxLen * self.alpha_ewald * maxBoxLen * self.alpha_ewald))
            if error_estimate < self.ewald_tolerance:
                self.k_max = i
                break
        
        self.ewald = Ewald(self.alpha_ewald, self.k_max, 2, self.use_customized_ops)
        self.ewald.to(device=box.device, dtype=box.dtype)
        if hasattr(self, 'polarization_solver'):
            self.polarization_solver.set_ewald(self.alpha_ewald, self.k_max, box.device, box.dtype)
        
        self._set_ewald = True
    
    def getEnergy(self, coords: torch.Tensor, box: torch.Tensor | None = None):
        
        boxInv = None if box is None else torch.linalg.inv(box)

        # Charge flux
        charge_flux = torch.zeros(self.natoms, device=coords.device, dtype=coords.dtype)
        charge_flux_pauli = torch.zeros(self.natoms, device=coords.device, dtype=coords.dtype)
        hardness_change = torch.ones(self.natoms, device=coords.device, dtype=coords.dtype)
        hardness_flux = torch.zeros(self.natoms, device=coords.device, dtype=coords.dtype)
        
        with timer("Bonded"):
            with timer("  Bond"):
                if not self.parametrizers['Bond'].is_empty:
                    bondIndices = self.parametrizers['Bond'].getExpandParameters("atomIndices")
                    j_cf = self.parametrizers['Bond'].getExpandParameters("j_cf")
                    j_cf_pauli = self.parametrizers['Bond'].getExpandParameters("j_cf_pauli")
                    r_eq = self.parametrizers['Bond'].getExpandParameters("r_eq")
                    if not self.use_customized_ops:
                        bondVecs = applyPBC(coords[bondIndices[:, 1]] - coords[bondIndices[:, 0]], box, boxInv)
                        bonds = computeBondFromVecs(bondVecs)
                        flux_bond = computeChargeFluxBond(bonds, r_eq, j_cf)
                        flux_pauli_bond = computeChargeFluxBond(bonds, r_eq, j_cf_pauli)
                        charge_flux.scatter_add_(0, bondIndices[:, 0], flux_bond[0])
                        charge_flux.scatter_add_(0, bondIndices[:, 1], flux_bond[1])
                        charge_flux_pauli.scatter_add_(0, bondIndices[:, 0], flux_pauli_bond[0])
                        charge_flux_pauli.scatter_add_(0, bondIndices[:, 1], flux_pauli_bond[1])
                        
                        if self.use_hardness_change:
                            k_hardness_b = self.parametrizers['Bond'].getExpandParameters("k_hardness_b")
                            hardness_change_bond = computeHardnessChangeBond(bonds, r_eq, k_hardness_b)
                            # NOTE(Eric): here can we use in-place operations?
                            hardness_change = hardness_change.scatter_reduce(0, bondIndices[:, 1], hardness_change_bond, 'prod')
                    else:
                        bond_cf, bond_cf_pauli = torch.ops.torchff.cmm_bond_charge_flux(coords, bondIndices.to(torch.int32), r_eq, j_cf, j_cf_pauli)
                        charge_flux.add_(bond_cf)
                        charge_flux_pauli.add_(bond_cf_pauli)
                        if self.use_hardness_change:
                            raise NotImplementedError()
            
            with timer("  Angle"):
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

                    if not self.use_customized_ops:
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

                        if self.use_hardness_change:
                            k_hardness_bb = self.parametrizers['Angle'].getExpandParameters("k_hardness_bb")
                            k_hardness_angle = self.parametrizers['Angle'].getExpandParameters("k_hardness_angle")
                            hardness_change_bb_1, hardness_change_bb_2 = computeHardnessChangeBondBond(r1, r2, r_eq_1, r_eq_2, k_hardness_bb, k_hardness_bb)
                            hardness_flux_angle = computeHardnessChangeAngle(theta, theta_eq, k_hardness_angle)
                            hardness_flux.scatter_add_(0, angleIndices[:, 0], hardness_flux_angle)
                            hardness_flux.scatter_add_(0, angleIndices[:, 2], hardness_flux_angle)
                            # NOTE(Eric): again, can we do in-place operations
                            hardness_change.scatter_reduce(0, angleIndices[:, 0], hardness_change_bb_1, 'prod')
                            hardness_change.scatter_reduce(0, angleIndices[:, 2], hardness_change_bb_2, 'prod')

                        ene_angle = torch.sum(computeCosAnglePotential(theta, theta_eq, k_th))
                        ene_bb = torch.sum(computeBondBondCoupling(r1, r2, r_eq_1, r_eq_2, k_bb))
                        ene_ba = torch.sum(computeBondAngleCoupling(r1, r_eq_1, theta, theta_eq, k_ba_1) + computeBondAngleCoupling(r2, r_eq_2, theta, theta_eq, k_ba_2))
                    else:
                        ene_angle, charge_flux_angle = torch.ops.torchff.cmm_angles(
                            coords, angleIndices.to(torch.int32), theta_eq, k_th, r_eq_1, r_eq_2, 
                            k_bb, k_ba_1, k_ba_2, j_cf_bb, j_cf_angle, -0.002
                        )
                        ene_bb = torch.tensor(0.0, device=coords.device)
                        ene_ba = torch.tensor(0.0, device=coords.device)
                        charge_flux.add_(charge_flux_angle)
                        if self.use_hardness_change:
                            raise NotImplementedError()
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
            with timer("Pairs"):
                # pairs
                all_distVecs = applyPBC(coords[self.all_pairs_j] - coords[self.all_pairs_i], box, boxInv)
                all_dists = torch.norm(all_distVecs, dim=1)

                mask_lr = all_dists < self.cutoff_lr
                pairs_lr = self.all_pairs[mask_lr]
                pairs_i_lr, pairs_j_lr = pairs_lr[:, 0], pairs_lr[:, 1]
                distVecs_lr = all_distVecs[mask_lr]
                dists_lr = all_dists[mask_lr]
                dists_inv_lr = 1 / dists_lr

                mask_sr = dists_lr < self.cutoff_sr
                pairs_sr = pairs_lr[mask_sr]
                pairs_i_sr, pairs_j_sr = pairs_sr[:, 0], pairs_sr[:, 1]
                distVecs_sr = distVecs_lr[mask_sr]
                dists_sr = dists_lr[mask_sr]
                dists_inv_sr = dists_inv_lr[mask_sr]
                switch_sr = self.switch_func_sr(dists_sr)
                switch_lr = self.switch_func_lr(dists_lr)
            
            with timer("Prep multipoles"):
                axistypes = self.parametrizers['Multipoles'].getExpandParameters('axistype')
                kzIndices = self.parametrizers['Multipoles'].getExpandParameters('kzIndices')
                kxIndices = self.parametrizers['Multipoles'].getExpandParameters('kxIndices')
                kyIndices = self.parametrizers['Multipoles'].getExpandParameters('kyIndices')
                if not self.use_customized_ops:
                    rotMatrices = computeLocal2GlobalRotationMatrixBatch(coords, kzIndices, kxIndices, kyIndices, axistypes, box, boxInv)
                else:
                    rotMatrices = torch.ops.torchff.compute_rotation_matrices(
                        coords, kzIndices.to(torch.int32), kxIndices.to(torch.int32), kyIndices.to(torch.int32), 
                        axistypes.to(torch.int32)
                    )
                mono = self.parametrizers['Multipoles'].getExpandParameters('mono') + charge_flux
                dipo = rotateDipoles(self.parametrizers['Multipoles'].getExpandParameters('dipo'), rotMatrices).squeeze(1)
                quad = rotateQuadrupoles(self.parametrizers['Multipoles'].getExpandParameters('quad'), rotMatrices)
                multipoles = convertMultipolesToPolytensor(mono, dipo, quad)
                self.last_perm_multipoles = multipoles
            
            ene_disp = torch.tensor(0.0, device=coords.device)
            if self.use_lr_dispersion:
                with timer("Dispersion-LR"):
                    if self.use_lr_dispersion and box is not None:
                        boxV = torch.linalg.det(box)
                        # ene_disp += computeLongRangeDispersionCorrection(c6_disp_ij, self.cutoff_lr, self.natoms, boxV)
                        ene_disp += -(2 / 3) * torch.pi * self.natoms * self.natoms * self.c6_mean / (self.cutoff_lr**3 * boxV)
            
            if not self.use_customized_ops:
                # Dispersion      
                with timer("Dispersion"):
                    c6_disp_ij = self.parametrizers['Dispersion'].getExpandParameters("C6_disp", pairs_lr)
                    b_disp_ij = self.parametrizers['Dispersion'].getExpandParameters("b_disp", pairs_lr)
                    disp_pairwise = computeDispersionFromPairs(dists_lr, c6_disp_ij, b_disp_ij)
                    ene_disp += torch.sum(disp_pairwise * switch_lr)
                
                # Pauli
                with timer("Pauli"):
                    pauli_mpoles = scaleMultipoles(
                        multipoles,
                        self.parametrizers['Pauli'].getExpandParameters("q_pauli") + charge_flux_pauli,
                        self.parametrizers['Pauli'].getExpandParameters("Kdipo_pauli"),
                        self.parametrizers['Pauli'].getExpandParameters("Kquad_pauli")
                    )
                    b_pauli_ij = self.parametrizers['Pauli'].getExpandParameters("b_pauli", pairs_sr)
                    pauli_pairwise = computeShortRangeEnergyFromPairs(
                        dists_sr, distVecs_sr, 
                        pauli_mpoles[pairs_i_sr], pauli_mpoles[pairs_j_sr], b_pauli_ij, 
                        switch_sr, True, dists_inv_sr
                    )
                    ene_pauli = torch.sum(pauli_pairwise)

                # XPol
                with timer("XPol"):
                    xpol_mpoles = scaleMultipoles(
                        multipoles,
                        self.parametrizers['ExchangePolarization'].getExpandParameters("q_xpol"),
                        self.parametrizers['ExchangePolarization'].getExpandParameters("Kdipo_xpol"),
                        self.parametrizers['ExchangePolarization'].getExpandParameters("Kquad_xpol")
                    )
                    b_xpol_ij = self.parametrizers['ExchangePolarization'].getExpandParameters("b_xpol", pairs_sr)
                    xpol_pairwise = computeShortRangeEnergyFromPairs(
                        dists_sr, distVecs_sr, 
                        xpol_mpoles[pairs_i_sr], xpol_mpoles[pairs_j_sr], b_xpol_ij, 
                        switch_sr, False, dists_inv_sr
                    )
                    ene_xpol = torch.sum(xpol_pairwise)

                # Charge Transfer
                with timer("CT"):
                    eps_ct = self.parametrizers['ChargeTransfer'].getExpandParameters("eps_ct", pairs_sr)
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
                    b_ct_ij = self.parametrizers['ChargeTransfer'].getExpandParameters("b_ct", pairs_sr)
                    ct_tensor = computeInteractionTensor(
                        distVecs_sr,
                        -computeShortRangeTwoCenterDampFactors(dists_sr, b_ct_ij),
                        dists_inv_sr
                    )
                    ct_direct_pairwise_ij = torch.bmm(ct_don_mpoles[pairs_j_sr].unsqueeze(1), torch.bmm(ct_tensor, ct_acc_mpoles[pairs_i_sr].unsqueeze(2))).flatten()
                    ct_direct_pairwise_ji = torch.bmm(ct_acc_mpoles[pairs_j_sr].unsqueeze(1), torch.bmm(ct_tensor, ct_don_mpoles[pairs_i_sr].unsqueeze(2))).flatten()
                    ene_ct_direct = torch.sum((ct_direct_pairwise_ij + ct_direct_pairwise_ji) * switch_sr)

                    drInvDamp_ct = ct_tensor[:, 0, 0].flatten()

                    dq_pairwise = (ct_don_mpoles[pairs_i_sr, 0] * ct_acc_mpoles[pairs_j_sr, 0] - ct_acc_mpoles[pairs_i_sr, 0] * ct_don_mpoles[pairs_j_sr, 0]) * drInvDamp_ct * eps_ct * switch_sr
                    dq_a = torch.zeros(self.natoms, device=pairs_sr.device, dtype=coords.dtype)
                    dq_a.scatter_add_(0, pairs_j_sr, dq_pairwise)
                    dq_a.scatter_add_(0, pairs_i_sr, -dq_pairwise)
                    dq_groups = segment_csr(dq_a[self.pol_group_indices_a], self.pol_group_segment_indices, reduce='sum')
                
                # Charge penetration
                with timer("CP"):
                    Z = self.parametrizers['ChargePenetration'].getExpandParameters("Z")
                    b_elec = self.parametrizers['ChargePenetration'].getExpandParameters("b_elec")
                    b_elec_ij = self.parametrizers['ChargePenetration'].getExpandParameters("b_elec", pairs_sr)
                    cp_mpoles = convertMultipolesToPolytensor(mono - Z, dipo, quad)

                    # shell-shell
                    cp_mpoles_i, cp_mpoles_j = cp_mpoles[pairs_i_sr], cp_mpoles[pairs_j_sr]
                    elec_cp_ss_pairwise = computeShortRangeEnergyFromPairs(
                        dists_sr, distVecs_sr, cp_mpoles_i, cp_mpoles_j, b_elec_ij,
                        switch_sr, False, dists_inv_sr
                    )
                    elec_cp_ss = torch.sum(elec_cp_ss_pairwise)
                    
                    # core-shell
                    cp_damps_i = -computeShortRangeOneCenterDampFactors(dists_sr, b_elec[pairs_i_sr])
                    cp_damps_j = -computeShortRangeOneCenterDampFactors(dists_sr, b_elec[pairs_j_sr])
                    cp_tensor_i = computeInteractionTensor(distVecs_sr, cp_damps_i, dists_inv_sr, 2)
                    cp_tensor_j = computeInteractionTensor(-distVecs_sr, cp_damps_j, dists_inv_sr, 2)

                    edata_cs_pairwise_ij = torch.bmm(cp_tensor_i, cp_mpoles_i.unsqueeze(2)).squeeze(2)
                    edata_cs_pairwise_ji = torch.bmm(cp_tensor_j, cp_mpoles_j.unsqueeze(2)).squeeze(2)

                    elec_cp_cs = torch.sum((edata_cs_pairwise_ij[:, 0] * Z[pairs_j_sr] + edata_cs_pairwise_ji[:, 0] * Z[pairs_i_sr]) * switch_sr)
                    ene_elec_cp = elec_cp_ss + elec_cp_cs

                # substract excluded pairs
                with timer("Ewald (excl)"):
                    self.setEwaldParameters(box)
                    drVecs_excl = applyPBC(coords[self.pairs_j_excl] - coords[self.pairs_i_excl], box, boxInv)
                    dr_excl = torch.norm(drVecs_excl, dim=1)
                    erfDamps = -computeDampFactorsErf(dr_excl, self.alpha_ewald)
                    realSpaceTensor_excl_ij = computeInteractionTensor(drVecs_excl, erfDamps)
                    realSpaceTensor_excl_ji = realSpaceTensor_excl_ij.permute(0, 2, 1)

                    edata_pairwise_excl_ij = torch.bmm(realSpaceTensor_excl_ij, multipoles[self.pairs_i_excl].unsqueeze(2))
                    edata_pairwise_excl_ji = torch.bmm(realSpaceTensor_excl_ji, multipoles[self.pairs_j_excl].unsqueeze(2))

                    ene_perm_elec_recip_excl = torch.sum(torch.bmm(multipoles[self.pairs_j_excl].unsqueeze(1), edata_pairwise_excl_ij).flatten())
                
                # Real-space electrostatics
                with timer("Ewald (real)"):
                    erfcDamps = computeDampFactorsErfc(dists_lr, self.alpha_ewald)
                    realSpaceTensor_ij = computeInteractionTensor(distVecs_lr, erfcDamps, dists_inv_lr, 2)
                    realSpaceTensor_ji = realSpaceTensor_ij.permute(0, 2, 1)
                    edata_pairwise_real_ij = torch.bmm(realSpaceTensor_ij, multipoles[pairs_i_lr].unsqueeze(2))
                    edata_pairwise_real_ji = torch.bmm(realSpaceTensor_ji, multipoles[pairs_j_lr].unsqueeze(2))
                    ene_perm_elec_ewald_real = torch.sum(torch.bmm(multipoles[pairs_j_lr].unsqueeze(1), edata_pairwise_real_ij).flatten())

                # accumulate electric potential and field
                with timer("Ewald (accumulate field)"):
                    edata = torch.zeros(self.natoms, 10, device=dists_lr.device, dtype=dists_lr.dtype)
                    edata.scatter_add_(0, pairs_j_sr.unsqueeze(1).expand(-1, 10), edata_cs_pairwise_ij)
                    edata.scatter_add_(0, pairs_i_sr.unsqueeze(1).expand(-1, 10), edata_cs_pairwise_ji)

                    edata.scatter_add_(0, pairs_j_lr.unsqueeze(1).expand(-1, 10), edata_pairwise_real_ij.squeeze(2))
                    edata.scatter_add_(0, pairs_i_lr.unsqueeze(1).expand(-1, 10), edata_pairwise_real_ji.squeeze(2))

                    edata.scatter_add_(0, self.pairs_j_excl.unsqueeze(1).expand(-1, 10), edata_pairwise_excl_ij.squeeze(2))
                    edata.scatter_add_(0, self.pairs_i_excl.unsqueeze(1).expand(-1, 10), edata_pairwise_excl_ji.squeeze(2))

                    epot_real = edata[:, 0]
                    efield_real = -edata[:, 1:4]

                    ene_perm_elec_real = ene_elec_cp + ene_perm_elec_ewald_real + ene_perm_elec_recip_excl
            else:
                # zero = torch.zeros(self.natoms, device=pairs_lr.device, dtype=coords.dtype)
                with timer('Pauli+XPOL+CT+Disp'):
                    c6_disp_ij = self.parametrizers['Dispersion'].getExpandParameters("C6_disp", pairs_lr)
                    b_disp_ij = self.parametrizers['Dispersion'].getExpandParameters("b_disp", pairs_lr)
                    ene_pauli, dq_a = torch.ops.torchff.cmm_non_elec_nonbonded_interaction_from_pairs(
                        distVecs_lr, pairs_lr.to(torch.int32), multipoles,
                        self.parametrizers['Pauli'].getExpandParameters("q_pauli") + charge_flux_pauli,
                        self.parametrizers['Pauli'].getExpandParameters("Kdipo_pauli"),
                        self.parametrizers['Pauli'].getExpandParameters("Kquad_pauli"),
                        # zero, zero, zero,
                        self.parametrizers['Pauli'].getExpandParameters("b_pauli", pairs_lr),
                        self.parametrizers['ExchangePolarization'].getExpandParameters("q_xpol"),
                        self.parametrizers['ExchangePolarization'].getExpandParameters("Kdipo_xpol"),
                        self.parametrizers['ExchangePolarization'].getExpandParameters("Kquad_xpol"),
                        # zero, zero, zero,
                        self.parametrizers['ExchangePolarization'].getExpandParameters("b_xpol", pairs_lr),
                        self.parametrizers['ChargeTransfer'].getExpandParameters("q_ct_don"),
                        self.parametrizers['ChargeTransfer'].getExpandParameters("Kdipo_ct_don"),
                        self.parametrizers['ChargeTransfer'].getExpandParameters("Kquad_ct_don"),
                        self.parametrizers['ChargeTransfer'].getExpandParameters("q_ct_acc"),
                        self.parametrizers['ChargeTransfer'].getExpandParameters("Kdipo_ct_acc"),
                        self.parametrizers['ChargeTransfer'].getExpandParameters("Kquad_ct_acc"),
                        # zero, zero, zero, zero, zero, zero,
                        self.parametrizers['ChargeTransfer'].getExpandParameters("b_ct", pairs_lr),
                        self.parametrizers['ChargeTransfer'].getExpandParameters("eps_ct", pairs_lr),
                        c6_disp_ij,
                        # torch.zeros_like(b_disp_ij, dtype=b_disp_ij.dtype, device=b_disp_ij.device),
                        b_disp_ij,
                        self.cutoff_sr, self.cutoff_lr, self.switch_buf
                    )
                with timer("CP+Ewald (real)+Ewald (excl)"):
                    self.setEwaldParameters(box)
                    b_elec_ij = self.parametrizers['ChargePenetration'].getExpandParameters("b_elec", pairs_lr)
                    drVecs_excl = applyPBC(coords[self.pairs_j_excl] - coords[self.pairs_i_excl], box, boxInv)
                    ene_perm_elec_real, epot_real, efield_real = torch.ops.torchff.cmm_elec_from_pairs(
                        distVecs_lr, pairs_lr.to(torch.int32), 
                        drVecs_excl, self.pairs_excl.to(torch.int32),
                        multipoles, 
                        self.parametrizers['ChargePenetration'].getExpandParameters("Z"),
                        b_elec_ij,
                        self.parametrizers['ChargePenetration'].getExpandParameters("b_elec"),
                        self.alpha_ewald,
                        self.cutoff_sr,
                        self.cutoff_lr,
                        self.switch_buf
                    )
                
                # Prepare variables for polarization
                # if self.use_polarization:
                #     with timer("POL-Tensors"):
                #         drVecs_excl = applyPBC(coords[self.pairs_j_excl] - coords[self.pairs_i_excl], box, boxInv)
                #         dr_excl = torch.norm(drVecs_excl, dim=1)
                #         erfDamps = -computeDampFactorsErf(dr_excl, self.alpha_ewald)
                #         realSpaceTensor_excl_ij = computeInteractionTensor(drVecs_excl, erfDamps)
                #         realSpaceTensor_excl_ji = realSpaceTensor_excl_ij.permute(0, 2, 1)
                #         erfcDamps = computeDampFactorsErfc(dists_lr, self.alpha_ewald)
                #         realSpaceTensor_ij = computeInteractionTensor(distVecs_lr, erfcDamps, dists_inv_lr, 2)
                #         realSpaceTensor_ji = realSpaceTensor_ij.permute(0, 2, 1)

                ene_ct_direct = torch.tensor(0.0, device=coords.device)
                ene_xpol = torch.tensor(0.0, device=coords.device)
                dq_groups = segment_csr(dq_a[self.pol_group_indices_a], self.pol_group_segment_indices, reduce='sum')
            
            with timer("Ewald (recip)"):
                # ewald_potential, ewald_field, ewald_field_gradient = long_range_potential(coords, mono, dipo, quad, box, self.alpha_ewald, self.k_max)
                ewald_potential, ewald_field, ewald_field_gradient = self.ewald(coords, box, mono, dipo, quad)
                ene_ewald = 0.5 * (
                    torch.einsum("n,n->", mono, ewald_potential) -
                    torch.einsum("ni,ni->", dipo, ewald_field) -
                    torch.einsum("nij,nij->", quad, ewald_field_gradient) / 3
                )

            ene_elec = ene_ewald + ene_perm_elec_real
            epot = epot_real + ewald_potential
            efield = efield_real + ewald_field

            if self.use_polarization:
                with timer("Polarization"):
                    b_vector = torch.hstack((-epot, efield.flatten(), dq_groups))
                    with timer("  Polarization-Parameters"):
                        # Polarization parameters
                        if self.use_hardness_change:
                            eta = self.parametrizers['Polarization'].getExpandParameters("eta") * hardness_change + hardness_flux
                        else:
                            eta = self.parametrizers['Polarization'].getExpandParameters("eta")
                        eta_times_2 = eta * 2

                        alpha = self.parametrizers['Polarization'].getExpandParameters("alpha")
                        alpha_damp_exponent = self.parametrizers['Polarization'].getExpandParameters("alpha_damp_exponent")
                        alpha_damp_max = self.parametrizers['Polarization'].getExpandParameters("alpha_damp_max")

                        polarizabilities = rotateQuadrupoles(alpha, rotMatrices)
                        polarizabilities = get_field_dependent_polarizabilities(polarizabilities, efield, alpha_damp_exponent, alpha_damp_max)
                        
                    # Evaluate initial guess #
                    if self.last_induced_multipoles.numel() == 0:
                        self.last_induced_multipoles = self.polarization_solver.direct_polarization_guess_without_charge(polarizabilities, efield)
                        
                    if not self.use_customized_ops:
                        with timer("  Polarization-tensors"):
                            pol_tensor = computeInteractionTensor(
                                distVecs_sr,
                                -computeShortRangePolarizationDampFactors(dists_sr, b_elec_ij),
                                dists_inv_sr,
                                1
                            )
                            pol_tensor = torch.vstack((pol_tensor, pol_tensor.permute(0, 2, 1)))
                            pairs_lr_bidir = torch.vstack((pairs_lr, pairs_lr[:, [1, 0]]))
                            pairs_sr_bidir = torch.vstack((pairs_sr, pairs_sr[:, [1, 0]]))
                            realSpaceTensor = torch.vstack((realSpaceTensor_ij[:, :4, :4], realSpaceTensor_ji[:, :4, :4]))
                            realSpaceTensor_excl = torch.vstack((realSpaceTensor_excl_ij[:, :4, :4], realSpaceTensor_excl_ji[:, :4, :4]))
                        
                        with timer("  Polarization-compute"):
                            ene_pol, induced_multipoles = self.polarization_solver(
                                coords,
                                box,
                                b_vector,
                                self.last_induced_multipoles,
                                eta_times_2,
                                polarizabilities,
                                pairs_lr_i_a=pairs_lr_bidir[:, 0], pairs_lr_j_a=pairs_lr_bidir[:, 1],
                                pairs_sr_i_a=pairs_sr_bidir[:, 0], pairs_sr_j_a=pairs_sr_bidir[:, 1],
                                pairs_excl_i_a=self.pairs_i_excl_bidir, pairs_excl_j_a=self.pairs_j_excl_bidir,
                                direct_field_tensor_lr=realSpaceTensor,
                                pol_interaction_tensor_sr=pol_tensor,
                                direct_field_tensor_excl=realSpaceTensor_excl,
                            )
                    else:
                        with timer("  Polarization-compute"):
                            ene_pol, induced_multipoles = self.polarization_solver(
                                coords,
                                box,
                                b_vector,
                                self.last_induced_multipoles,
                                eta_times_2,
                                polarizabilities,
                                pairs=pairs_lr.to(torch.int32),
                                pairs_excl=self.pairs_excl.to(torch.int32),
                                b_elec_ij=b_elec_ij,
                                dist_vecs=distVecs_lr,
                                dist_vecs_excl=drVecs_excl
                            )
                    self.last_induced_multipoles = induced_multipoles.detach().clone()
            else:
                ene_pol = torch.tensor(0.0, device=coords.device)

        # Field-dependent morse
        with timer("FDMorse"):
            if not self.parametrizers['Bond'].is_empty:
                bondIndices = self.parametrizers['Bond'].getExpandParameters("atomIndices")
                r_eq = self.parametrizers['Bond'].getExpandParameters('r_eq')
                k_b = self.parametrizers['Bond'].getExpandParameters('k_b')
                D = self.parametrizers['Bond'].getExpandParameters('D')

                if not self.use_customized_ops:
                    bondVecs = applyPBC(coords[bondIndices[:, 1]] - coords[bondIndices[:, 0]], None, None)
                    bonds = computeBondFromVecs(bondVecs)

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
                        ene_bond = torch.sum(computeMorseBondPotential(bonds, r_eq_fd, D, beta_fd))
                    else:
                        beta = torch.sqrt(k_b / 2 / D)
                        ene_bond = torch.sum(computeMorseBondPotential(bonds, r_eq, D, beta))
                else:
                    if self.use_fd_morse:
                        dip_deriv_1 = self.parametrizers['Bond'].getExpandParameters('dip_deriv_1')
                        dip_deriv_2 = self.parametrizers['Bond'].getExpandParameters('dip_deriv_2')
                        ene_bond = torch.ops.torchff.cmm_field_dependent_morse_bond(
                            coords, bondIndices.to(torch.int32), r_eq, k_b, D, dip_deriv_1, dip_deriv_2, efield
                        )
                    else:
                        raise NotImplementedError()
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