import torch

__all__ = [
    "computeBondFromVecs", "computeAngleFromVecs", "computeMorseBondPotential",
    "computeBondBondCoupling", "computeCosAnglePotential", "computeBondAngleCoupling",
    "computeChargeFluxBond", "computeChargeFluxBondBond", "computeChargeFluxAngle",
    "computeHardnessChangeBond", "computeHardnessChangeBondBond", "computeHardnessChangeAngle",
    "computeFieldDependentMorseParams", "computeHarmonicBondPotential", "computeHarmonicAnglePotential"
]

def computeBondFromVecs(drVecs):
    return torch.norm(drVecs, dim=1)

def computeAngleFromVecs(drVecs1, drVecs2):
    cosVal = torch.sum(drVecs1 * drVecs2, dim=1) / torch.norm(drVecs1, dim=1) / torch.norm(drVecs2, dim=1)
    return torch.arccos(cosVal)

def computeChargeFluxBond(r: torch.Tensor, req: torch.Tensor, j_cf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (-j_cf * (r - req), j_cf * (r - req))

def computeChargeFluxBondBond(
        r1: torch.Tensor, r2: torch.Tensor,
        req1: torch.Tensor, req2: torch.Tensor,
        j_bb_cf_1: torch.Tensor, j_bb_cf_2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
         j_bb_cf_1 * (r2 - req2),
        -j_bb_cf_1 * (r2 - req2),
         j_bb_cf_2 * (r1 - req1),
        -j_bb_cf_2 * (r1 - req1),
    )

def computeChargeFluxAngle(
        theta: torch.Tensor,
        thetaeq: torch.Tensor,
        theta_cf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (theta_cf * (theta - thetaeq), -2 * theta_cf * (theta - thetaeq), theta_cf * (theta - thetaeq))

def computeHardnessChangeBond(r: torch.Tensor, req: torch.Tensor, k_hardness: torch.Tensor) -> torch.Tensor:
    return torch.pow(req / r, k_hardness)

def computeHardnessChangeBondBond(
        r1: torch.Tensor, r2: torch.Tensor,
        req1: torch.Tensor, req2: torch.Tensor,
        k_bb_hardness_1: torch.Tensor, k_bb_hardness_2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
         torch.pow(req2 / r2, k_bb_hardness_1),
         torch.pow(req1 / r1, k_bb_hardness_2)
    )

def computeHardnessChangeAngle(
        theta: torch.Tensor,
        thetaeq: torch.Tensor,
        theta_hardness: torch.Tensor) -> torch.Tensor:
    return theta_hardness * (theta - thetaeq)

def computeHarmonicBondPotential(r: torch.Tensor, req: torch.Tensor, k_b: torch.Tensor):
    return 0.5 * k_b * (r - req)**2

def computeMorseBondPotential(r: torch.Tensor, req: torch.Tensor, d: torch.Tensor, a: torch.Tensor):
    return d * (1 - torch.exp(-a * (r - req))) ** 2


def computeBondBondCoupling(r1: torch.Tensor, r2: torch.Tensor, req1: torch.Tensor, req2: torch.Tensor, k: torch.Tensor):
    return k * (r1 - req1) * (r2 - req2)

def computeHarmonicAnglePotential(theta: torch.Tensor, thetaeq: torch.Tensor, k: torch.Tensor):
    return 0.5 * k * (theta - thetaeq)**2

def computeCosAnglePotential(theta: torch.Tensor, thetaeq: torch.Tensor, k: torch.Tensor):
    return k / 2 * (torch.cos(theta) - torch.cos(thetaeq)) ** 2


def computeBondAngleCoupling(r: torch.Tensor, req: torch.Tensor, theta: torch.Tensor, thetaeq: torch.Tensor, k: torch.Tensor):
    return k * (r - req) * (torch.cos(theta) - torch.cos(thetaeq))

def computeFieldDependentMorseParams(
        bond_dists_p: torch.Tensor, bond_vecs_p: torch.Tensor,
        k_e_p: torch.Tensor, D_e_p: torch.Tensor, r_e_p: torch.Tensor,
        dipole_1_p: torch.Tensor, dipole_2_p: torch.Tensor,
        ct_slope_1_p: torch.Tensor, ct_slope_2_p: torch.Tensor,
        E_field_p: torch.Tensor, dQ_ct_p: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluates the field-dependent force constants and equilibrium distances
    used in the CMM Morse potential. All of these tensors should have N_fd_bond
    entries where N_fd_bond is the number of bonds which are field-dependent.
    Note that dR_vec gets dotted with E, so you need to ensure that the distance
    vectors are computed in the right direction.
    """
    
    # We are making an assumption here which will have to be enforced by the topology
    # builder. The field is considered only for the second atom of the bond vector.
    # For water, for instance, this means we consider the field at the H atom.
    # In general, the specific atom will depend on the bond in question, so the
    # topology builder will have to look at the specific bond and force field terms
    # requested so that it can set up the bond indices appropriately. -Joe

    # NOTE(JOE): The charge transfer parameters are broken for ions right now so I
    # turned off that contribution to the FD morse. That term is kind of kludgy anyways
    # and we should probably consider removing it.
    E_proj_p = torch.func.vmap(torch.dot)(bond_vecs_p, E_field_p) / bond_dists_p
    dr_e_p = E_proj_p * dipole_1_p / (k_e_p - E_proj_p * dipole_2_p) #+ ct_slope_1_p * dQ_ct_p * dQ_ct_p
    k_e_fd = k_e_p - (3 * k_e_p * torch.sqrt(0.5 * k_e_p / D_e_p) * dr_e_p + E_proj_p * dipole_2_p) #+ ct_slope_2_p * dQ_ct_p * dQ_ct_p
    
    # Ideally this will never happen but this is how I implemented it originally
    # to avoid the possiblity of taking a sqrt of a negative force constant
    # during the energy evaluation. Really hitting this branch indicates
    # the field is too strong for this model to be reasonable or that the
    # parameters determining the change in force constant are unrealistic.
    k_e_fd = torch.clamp(k_e_fd, 0.4 * k_e_p)
    beta_fd = torch.sqrt(k_e_fd / 2 / D_e_p)
    return (r_e_p + dr_e_p, beta_fd)