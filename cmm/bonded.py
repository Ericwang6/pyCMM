import torch
from .pbc import applyPBC

__all__ = [
    "computeBondFromVecs", "computeAngleFromVecs", "computeMorseBondPotential",
    "computeBondBondCoupling", "computeCosAnglePotential", "computeBondAngleCoupling",
    "computeChargeFluxBond", "computeChargeFluxBondBond", "computeChargeFluxAngle",
    "computeHardnessChangeBond", "computeHardnessChangeBondBond", "computeHardnessChangeAngle",
    "computeFieldDependentMorseParams", "computeHarmonicBondPotential", "computeHarmonicAnglePotential",
    "computeFieldDependentCosAngleParams"
]

def computeBondFromVecs(drVecs):
    return torch.norm(drVecs, dim=-1)

def computeBond(coords: torch.Tensor, bondIndices: torch.Tensor, box: torch.Tensor | None = None, boxInv: torch.Tensor | None = None):
    return computeBondFromVecs(
        applyPBC(coords[bondIndices[:, 1]] - coords[bondIndices[:, 0]], box, boxInv)
    )


def computeBondBatch(coords: torch.Tensor, bondIndices: torch.Tensor):
    return computeBondFromVecs(coords[:, bondIndices[:, 1]] - coords[:, bondIndices[:, 0]])


@torch.compile
def computeAngleFromVecs(drVecs1, drVecs2):
    cosVal = torch.sum(drVecs1 * drVecs2, dim=-1) / torch.norm(drVecs1, dim=-1) / torch.norm(drVecs2, dim=-1)
    return torch.arccos(cosVal)

@torch.compile
def computeAngle(coords: torch.Tensor, angleIndices: torch.Tensor, box: torch.Tensor | None = None, boxInv: torch.Tensor | None = None):
    bondVec1 = applyPBC(coords[angleIndices[:, 0]] - coords[angleIndices[:, 1]], box, boxInv)
    bondVec2 = applyPBC(coords[angleIndices[:, 2]] - coords[angleIndices[:, 1]], box, boxInv)
    return computeAngleFromVecs(bondVec1, bondVec2)


def computeAngleBatch(coords: torch.Tensor, angleIndices: torch.Tensor):
    bondVec1 = coords[:, angleIndices[:, 0]] - coords[:, angleIndices[:, 1]]
    bondVec2 = coords[:, angleIndices[:, 2]] - coords[:, angleIndices[:, 1]]
    return computeAngleFromVecs(bondVec1, bondVec2)


@torch.compile
def computeChargeFluxBond(r: torch.Tensor, req: torch.Tensor, j_cf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flux = j_cf * (r - req)
    return (-flux, flux)

@torch.compile
def computeChargeFluxBondBond(
    r1: torch.Tensor, r2: torch.Tensor,
    req1: torch.Tensor, req2: torch.Tensor,
    j_bb_cf_1: torch.Tensor, j_bb_cf_2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flux_1 = j_bb_cf_1 * (r2 - req2)
    flux_2 = j_bb_cf_2 * (r1 - req1)
    return (flux_1, -flux_1, flux_2, -flux_2)

@torch.compile
def computeChargeFluxAngle(
    theta: torch.Tensor,
    thetaeq: torch.Tensor,
    theta_cf: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flux = theta_cf * (theta - thetaeq)
    return (flux, -2*flux, flux)

@torch.compile
def computeHardnessChangeBond(r: torch.Tensor, req: torch.Tensor, k_hardness: torch.Tensor) -> torch.Tensor:
    return torch.pow(req / r, k_hardness)

@torch.compile
def computeHardnessChangeBondBond(
        r1: torch.Tensor, r2: torch.Tensor,
        req1: torch.Tensor, req2: torch.Tensor,
        k_bb_hardness_1: torch.Tensor, k_bb_hardness_2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
         torch.pow(req2 / r2, k_bb_hardness_1),
         torch.pow(req1 / r1, k_bb_hardness_2)
    )

@torch.compile
def computeHardnessChangeAngle(
        theta: torch.Tensor,
        thetaeq: torch.Tensor,
        theta_hardness: torch.Tensor) -> torch.Tensor:
    return theta_hardness * (theta - thetaeq)

def computeHarmonicBondPotential(r: torch.Tensor, req: torch.Tensor, k_b: torch.Tensor):
    return 0.5 * k_b * (r - req)**2

@torch.compile
def computeMorseBondPotential(r: torch.Tensor, req: torch.Tensor, d: torch.Tensor, a: torch.Tensor):
    return d * (1 - torch.exp(-a * (r - req))) ** 2

@torch.compile
def computeBondBondCoupling(r1: torch.Tensor, r2: torch.Tensor, req1: torch.Tensor, req2: torch.Tensor, k: torch.Tensor, minv: float = -0.002):
    energy = k * (r1 - req1) * (r2 - req2)
    return torch.where(energy > minv, energy, minv)

def computeHarmonicAnglePotential(theta: torch.Tensor, thetaeq: torch.Tensor, k: torch.Tensor):
    return 0.5 * k * (theta - thetaeq)**2

@torch.compile
def computeCosAnglePotential(theta: torch.Tensor, thetaeq: torch.Tensor, k: torch.Tensor):
    return k / 2 * (torch.cos(theta) - torch.cos(thetaeq)) ** 2

@torch.compile
def computeBondAngleCoupling(r: torch.Tensor, req: torch.Tensor, theta: torch.Tensor, thetaeq: torch.Tensor, k: torch.Tensor, minv: float = -0.002):
    energy = k * (r - req) * (torch.cos(theta) - torch.cos(thetaeq))
    return torch.where(energy > minv, energy, minv)

@torch.compile
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
    E_proj_p = torch.sum(bond_vecs_p * E_field_p, dim=-1) / bond_dists_p
    #uncomment ct lines to use Morse CT expression
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


def computeTorsionFromVecs(drVecs1: torch.Tensor, drVecs2: torch.Tensor, drVecs3: torch.Tensor):
    n1 = torch.cross(drVecs1, drVecs2, dim=-1)
    n2 = torch.cross(drVecs2, drVecs3, dim=-1)
    norm_n1 = torch.norm(n1, dim=-1)
    norm_n2 = torch.norm(n2, dim=-1)
    cosval = torch.clamp(
        torch.sum(n1 * n2, dim=-1) / (norm_n1 * norm_n2), 
        -0.999999999, 0.999999999
    )
    phi = torch.acos(cosval) * torch.sign(torch.sum(n1 * drVecs3, dim=-1))
    return phi

def computeTorsion(coords: torch.Tensor, torsionIndices: torch.Tensor, box: torch.Tensor | None = None, boxInv: torch.Tensor | None = None):
    bondVecs_ij = applyPBC(coords[torsionIndices[:, 1]] - coords[torsionIndices[:, 0]], box, boxInv)
    bondVecs_jk = applyPBC(coords[torsionIndices[:, 2]] - coords[torsionIndices[:, 1]], box, boxInv)
    bondVecs_kl = applyPBC(coords[torsionIndices[:, 3]] - coords[torsionIndices[:, 2]], box, boxInv)
    torsions = computeTorsionFromVecs(bondVecs_ij, bondVecs_jk, bondVecs_kl)
    return torsions


def computeTorsionBatch(coords: torch.Tensor, torsionIndices: torch.Tensor):
    bondVecs_ij = applyPBC(coords[:, torsionIndices[:, 1]] - coords[:, torsionIndices[:, 0]])
    bondVecs_jk = applyPBC(coords[:, torsionIndices[:, 2]] - coords[:, torsionIndices[:, 1]])
    bondVecs_kl = applyPBC(coords[:, torsionIndices[:, 3]] - coords[:, torsionIndices[:, 2]])
    torsions = computeTorsionFromVecs(bondVecs_ij, bondVecs_jk, bondVecs_kl)
    return torsions


def computePeriodicTorsionEnergy(torsions: torch.Tensor, per: torch.Tensor, phase: torch.Tensor, k: torch.Tensor):
    if len(per.shape) == 2:
        return k * (1 + torch.cos(torsions.reshape(-1, 1) * per - phase))
    else:
        return k * (1 + torch.cos(torsions * per - phase))

def computeTorsionBondCoupling(
    torsions: torch.Tensor, bonds: torch.Tensor,
    per: torch.Tensor, phase: torch.Tensor, k: torch.Tensor,
    req: torch.Tensor,
    minv: float = -0.002
):
    if len(per.shape) == 2:
        energy = k * torch.reshape(bonds - req, (-1, 1)) * (1 + torch.cos(torsions.reshape(-1, 1) * per - phase))
    else:
        energy = k * (bonds - req) * (1 + torch.cos(torsions * per - phase))
    return torch.where(energy > minv, energy, minv)
    

def computeTorsionAngleAngleCoupling(
    torsions: torch.Tensor, angles1: torch.Tensor, angles2: torch.Tensor,
    per: torch.Tensor, phase: torch.Tensor, k: torch.Tensor,
    theta_eq_1: torch.Tensor, theta_eq_2: torch.Tensor,
    minv: float = -0.002
):
    if len(per.shape) == 2:
        energy = k * torch.reshape((angles2 - theta_eq_2) * (angles1 - theta_eq_1), (-1, 1)) * (1 + torch.cos(torsions.reshape(-1, 1) * per - phase))
    else:
        energy = k * (angles2 - theta_eq_2) * (angles1 - theta_eq_1) * (1 + torch.cos(torsions * per - phase))
    return torch.where(energy > minv, energy, minv)

@torch.compile
def computeFieldDependentCosAngleParams(
        bond_vecs_1_p: torch.Tensor,    # vec from B→A (or bisector direction)
        bond_vecs_2_p: torch.Tensor,    # vec from B→C
        theta_p: torch.Tensor,          # current angle θ (radians)
        theta_eq_p: torch.Tensor,       # equilibrium angle θ₀
        k_p: torch.Tensor,              # force constant k
        dmu_dtheta_p: torch.Tensor,     # |∂μ/∂θ| at θ₀ (charge × length)
        d2mu_dtheta2_p: torch.Tensor,   # |∂²μ/∂θ²| at θ₀ (charge × length; pass zeros to ignore)
        E_field_p: torch.Tensor,        # electric field vector at atom B
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluates the field-dependent force constant and equilibrium angle for the
    cosine-harmonic angle potential.

    Works analogously to computeFieldDependentMorseParams: the field is
    projected onto the angle bisector direction at atom B, then used to
    compute cosine-space coupling constants lambda1 and lambda2, which shift the
    effective equilibrium and soften/stiffen the force constant.

    All input tensors should have N_fd_angle entries.

    Returns:
        k_eff       — field-modified force constant
        cos_theta0_eff — field-modified cos(theta_0)
        theta0_eff  — arccos of the above (radians)
    """
    # Bisector direction: normalize the sum of the two unit bond vectors.
    norm1 = torch.norm(bond_vecs_1_p, dim=-1, keepdim=True)
    norm2 = torch.norm(bond_vecs_2_p, dim=-1, keepdim=True)
    bisector = bond_vecs_1_p / norm1 + bond_vecs_2_p / norm2
    bisector_norm = torch.norm(bisector, dim=-1, keepdim=True)
    bisector_hat = bisector / bisector_norm          # unit bisector d_hat

    # Project field onto the bisector at B (scalar, one per angle)
    eps = torch.sum(E_field_p * bisector_hat, dim=-1)   # epsilon = field * d_hat

    # Angular dipole couplings at theta0 (Eq. 4 of the PDF)
    mu_prime  = dmu_dtheta_p  * eps   # mu' =  |dmu/dheta| * epsilon
    mu_dprime = d2mu_dtheta2_p * eps  # mu'' = |d2mu/dtheta2| * epsilon

    sin_t0  = torch.sin(theta_eq_p)
    cos_t0  = torch.cos(theta_eq_p)
    sin2_t0 = sin_t0 * sin_t0

    # Cosine-space coupling constants (Eqs. 6–7)
    lambda1 = -mu_prime / sin_t0
    lambda2 = (mu_dprime - (cos_t0 / sin_t0) * mu_prime) / sin2_t0

    # Field-dependent parameters (Eqs. 9–11)
    k_eff = k_p - lambda2

    # Guard against unphysical softening (mirrors the clamp in the Morse function)
    k_eff = torch.clamp(k_eff, 0.4 * k_p)

    cos_theta0_eff = cos_t0 + lambda1 / k_eff
    # Clamp to valid arccos domain before inverting
    cos_theta0_eff = torch.clamp(cos_theta0_eff, -1.0 + 1e-6, 1.0 - 1e-6)
    theta0_eff = torch.arccos(cos_theta0_eff)

    return (k_eff, cos_theta0_eff, theta0_eff)
