import torch


def computeBondFromVecs(drVecs):
    return torch.norm(drVecs, dim=1)


def computeAngleFromVecs(drVecs1, drVecs2):
    cosVal = torch.sum(drVecs1 * drVecs2, dim=1) / torch.norm(drVecs1, dim=1) / torch.norm(drVecs2, dim=1)
    return torch.arccos(cosVal)


def computeMorseBondPotential(r: torch.Tensor, req: torch.Tensor, d: torch.Tensor, a: torch.Tensor):
    return d * (1 - torch.exp(-a * (r - req))) ** 2


def computeBondBondCoupling(r1: torch.Tensor, r2: torch.Tensor, req1: torch.Tensor, req2: torch.Tensor, k: torch.Tensor):
    return k * (r1 - req1) * (r2 - req2)


def computeCosAnglePotential(theta: torch.Tensor, thetaeq: torch.Tensor, k: torch.Tensor):
    return k / 2 * (torch.cos(theta) - torch.cos(thetaeq)) ** 2


def computeBondAngleCoupling(r: torch.Tensor, req: torch.Tensor, theta: torch.Tensor, thetaeq: torch.Tensor, k: torch.Tensor):
    return k * (r - req) * (torch.cos(theta) - torch.cos(thetaeq))

