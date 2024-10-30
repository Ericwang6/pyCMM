import torch

__all__ = [
    "computeBondFromVecs", "computeAngleFromVecs", "computeMorseBondPotential",
    "computeBondBondCoupling", "computeCosAnglePotential", "computeBondAngleCoupling",
    "computeChargeFluxBond", "computeChargeFluxBondBond", "computeChargeFluxAngle"
]

def computeBondFromVecs(drVecs):
    return torch.norm(drVecs, dim=1)

def computeAngleFromVecs(drVecs1, drVecs2):
    cosVal = torch.sum(drVecs1 * drVecs2, dim=1) / torch.norm(drVecs1, dim=1) / torch.norm(drVecs2, dim=1)
    return torch.arccos(cosVal)

def computeChargeFluxBond(r: torch.Tensor, req: torch.Tensor, j_cf: torch.Tensor):
    return torch.stack((-j_cf * (r - req), j_cf * (r - req)))

def computeChargeFluxBondBond(
        r1: torch.Tensor, r2: torch.Tensor,
        req1: torch.Tensor, req2: torch.Tensor,
        j_bb_cf_1: torch.Tensor, j_bb_cf_2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return (j_bb_cf_1 * (r2 - req2), j_bb_cf_2 * (r1 - req1))

def computeChargeFluxAngle(theta: torch.Tensor, thetaeq: torch.Tensor, theta_cf: torch.Tensor) -> torch.Tensor:
    return theta_cf * (theta - thetaeq)
    

def computeMorseBondPotential(r: torch.Tensor, req: torch.Tensor, d: torch.Tensor, a: torch.Tensor):
    return d * (1 - torch.exp(-a * (r - req))) ** 2


def computeBondBondCoupling(r1: torch.Tensor, r2: torch.Tensor, req1: torch.Tensor, req2: torch.Tensor, k: torch.Tensor):
    return k * (r1 - req1) * (r2 - req2)


def computeCosAnglePotential(theta: torch.Tensor, thetaeq: torch.Tensor, k: torch.Tensor):
    return k / 2 * (torch.cos(theta) - torch.cos(thetaeq)) ** 2


def computeBondAngleCoupling(r: torch.Tensor, req: torch.Tensor, theta: torch.Tensor, thetaeq: torch.Tensor, k: torch.Tensor):
    return k * (r - req) * (torch.cos(theta) - torch.cos(thetaeq))

