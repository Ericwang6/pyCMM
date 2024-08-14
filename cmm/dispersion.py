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