import torch

def computeDispersion(
    drVec: torch.Tensor, 
    c6_i: torch.Tensor, c6_j: torch.Tensor,
    b_i: torch.Tensor, b_j: torch.Tensor
):
    c6_ij = torch.sqrt(c6_i * c6_j)
    b_ij = torch.sqrt(b_i * b_j)
    dr = torch.norm(drVec, dim=1)
    u = b_ij * dr
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    u5 = u4 * u
    u6 = u5 * u
    exp_u = torch.exp(-u)
    damp = 1 - exp_u * (1 + u + u2 / 2 + u3 / 6 + u4 / 24 + u5 / 120 + u6 / 720)
    return -damp * c6_ij / torch.pow(dr, 6)

def computeDispersionFromPairs(
    dists_p: torch.Tensor, 
    c6_ij_p: torch.Tensor,b_ij_p: torch.Tensor
):
    u = b_ij_p * dists_p
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    u5 = u4 * u
    u6 = u5 * u
    exp_u = torch.exp(-u)
    damp = 1 - exp_u * (1 + u + u2 / 2 + u3 / 6 + u4 / 24 + u5 / 120 + u6 / 720)
    return -damp * c6_ij_p / torch.pow(dists_p, 6)

def computeDispersionFromPairsWithSwitching(
    dists_p: torch.Tensor, 
    c6_ij_p: torch.Tensor,b_ij_p: torch.Tensor,
    switching_values: torch.Tensor
):
    u = b_ij_p * dists_p
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    u5 = u4 * u
    u6 = u5 * u
    exp_u = torch.exp(-u)
    damp = 1 - exp_u * (1 + u + u2 / 2 + u3 / 6 + u4 / 24 + u5 / 120 + u6 / 720)
    return -damp * c6_ij_p * switching_values / torch.pow(dists_p, 6)

def compute_long_range_dispersion_correction(
    C6_ij_p: torch.Tensor, cutoff_vdw: torch.Tensor,
    natoms: torch.Tensor, box_volume: torch.Tensor
):
    # NOTE(JOE): There is a chance that this correction should be a factor of 2 larger.
    # But I'm not really sure. Leaving as is since this is what is in the Gromacs manual:
    # https://manual.gromacs.org/current/reference-manual/functions/long-range-vdw.html
    C6_average = torch.mean(C6_ij_p)
    return -(2 / 3) * torch.pi * natoms * natoms * C6_average / (torch.pow(cutoff_vdw, 3) * box_volume)

def computeLennardJonesFromPairs(
    dists_p: torch.Tensor, 
    sigma_ij_p: torch.Tensor, eps_ij_p: torch.Tensor
):
    u6  = (sigma_ij_p / dists_p)**6
    u12 = u6**2
    return 4 * eps_ij_p * (u12 - u6)

def compute_long_range_lennard_jones_correction(
    sigma_ij_p: torch.Tensor, eps_ij_p: torch.Tensor, cutoff_vdw: torch.Tensor,
    natoms: torch.Tensor, box_volume: torch.Tensor
):
    # See: http://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#lennard-jones-interaction
    # or DOI: 10.1021/jp0735987 J. Phys. Chem. B 2007, 111, 13052-13063
    eu12 = torch.mean(eps_ij_p * sigma_ij_p**12)
    eu6 = torch.mean(eps_ij_p * sigma_ij_p**6)
    return 8 * torch.pi * natoms * natoms * (eu12 / (9 * cutoff_vdw**9) - eu6 / (3 * cutoff_vdw**3)) / box_volume