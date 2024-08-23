import math
from typing import List, Optional
import torch

from .pbc import applyPBC

HALF_SQRT3 = math.sqrt(3) / 2


def normVec(vec):
    return vec / torch.norm(vec, dim=1, keepdim=True)


def computeLocal2GlobalRotationMatrix(pos, pos1, pos2, pos3, axisTypes, box=None, boxInv=None):
    """
    Compute local to global rotation matrix
    """
    # ZThenX
    if (box is not None) and (boxInv is not None):
        zvec = normVec(applyPBC(pos1 - pos, box, boxInv))
        xvec = normVec(applyPBC(pos2 - pos, box, boxInv))
    else:
        zvec = normVec(pos1 - pos)
        xvec = normVec(pos2 - pos)
    # Bisector  
    zvec += xvec * (axisTypes == 1).unsqueeze(1)
    zvec = normVec(zvec)
    
    xvec = xvec - torch.sum(zvec * xvec, dim=1, keepdim=True) * zvec
    xvec = normVec(xvec)
    yvec = torch.cross(zvec, xvec)
    rotMatrix = torch.hstack((xvec, yvec, zvec)).reshape(-1, 3, 3)
    return rotMatrix


def rotateDipoles(dipo: torch.Tensor, rotMatrix: torch.Tensor):
    return torch.bmm(dipo.unsqueeze(1), rotMatrix)


def rotateQuadrupoles(quad: torch.Tensor, rotMatrix: torch.Tensor):
    return torch.bmm(torch.bmm(rotMatrix.permute(0, 2, 1), quad), rotMatrix)


def rotateMultipoles(mono: torch.Tensor, dipo: torch.Tensor, quad: torch.Tensor, rotMatrix: torch.Tensor):
    """
    Rotate multipoles

    Parameters
    ----------
    mono: torch.Tensor
        Monopoles, shape (N,)
    dipo: torch.Tensor
        Dipoles, shape (N, 3)
    quad: torch.Tensor
        Quadrupoles, shape (N, 3, 3)
    
    Returns
    -------
    mPoles: torch.Tensor
        Multipoles [q, ux, uy, uz, Qxx, Qxy, Qxz, Qyy, Qyz, Qzz], shape (N, 10)
    """
    mono = mono.unsqueeze(1)
    dipo = rotateDipoles(dipo, rotMatrix).squeeze(1)
    quad = rotateQuadrupoles(quad, rotMatrix)[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]
    return torch.hstack((mono, dipo, quad))


def computeCartesianQuadrupoles(quad_s: torch.Tensor):
    """
    Compute cartesian quadrupoles from spheric-harmonics quadrupoles

    Parameters
    ----------
    quad_s: torch.Tensor
        Quadrupoles in spherical harmonics form (Q20, Q21c, Q21s, Q22c, Q22s), shape (N, 5).

    Returns
    -------
    quad: torch.Tensor
        Quadrupoles in cartesian form, shape N x 3 x 3
    """
    qxx = quad_s[:, 3] * HALF_SQRT3 - quad_s[:, 0] / 2
    qxy = quad_s[:, 4] * HALF_SQRT3
    qxz = quad_s[:, 1] * HALF_SQRT3
    qyy = -quad_s[:, 3] * HALF_SQRT3 - quad_s[:, 0] / 2
    qyz = quad_s[:, 2] * HALF_SQRT3
    qzz = quad_s[:, 0]
    quad = torch.vstack((qxx, qxy, qxz, qxy, qyy, qyz, qxz, qyz, qzz)).T.reshape(-1, 3, 3)
    return quad


def computeInteractionTensor(drVec: torch.Tensor, dampFactors: Optional[List[torch.Tensor]] = None, drInv: Optional[torch.Tensor] = None, rank: int = 2):
    """
    drVec: N x 3
    mPoles: N x 10
    dampFactors: 5 x N

    eData: N x 
    """
    if drInv is None:
        drInv = 1 / torch.norm(drVec, dim=1)
    
    # calculate inversions
    if rank > 0:
        drInv2 = torch.pow(drInv, 2)
        drInv3 = drInv2 * drInv
        drInv5 = drInv3 * drInv2

        drVec2 = torch.pow(drVec, 2)
        x, y, z = drVec[:, 0], drVec[:, 1], drVec[:, 2]
        x2, y2, z2 = drVec2[:, 0], drVec2[:, 1], drVec2[:, 2]
        xy, xz, yz = x * y, x * z, y * z
    if rank > 1:
        drInv7 = drInv5 * drInv2
        drInv9 = drInv7 * drInv2
    

    if dampFactors is not None:
        drInv = drInv * dampFactors[0]
    if rank > 0:
        if dampFactors is not None:
            drInv3 = drInv3 * dampFactors[1]
            drInv5 = drInv5 * dampFactors[2]
        tx, ty, tz = -x * drInv3, -y * drInv3, -z * drInv3
        
        txx = 3 * x2 * drInv5 - drInv3
        txy = 3 * xy * drInv5
        txz = 3 * xz * drInv5
        tyy = 3 * y2 * drInv5 - drInv3
        tyz = 3 * yz * drInv5
        tzz = 3 * z2 * drInv5 - drInv3

    if rank > 1:
        if dampFactors is not None:
            drInv7 = drInv7 * dampFactors[3]
            drInv9 = drInv9 * dampFactors[4]

        txxx = -15 * x2 * x * drInv7 + 9 * x * drInv5
        txxy = -15 * x2 * y * drInv7 + 3 * y * drInv5
        txxz = -15 * x2 * z * drInv7 + 3 * z * drInv5
        tyyy = -15 * y2 * y * drInv7 + 9 * y * drInv5
        tyyx = -15 * y2 * x * drInv7 + 3 * x * drInv5
        tyyz = -15 * y2 * z * drInv7 + 3 * z * drInv5
        tzzz = -15 * z2 * z * drInv7 + 9 * z * drInv5
        tzzx = -15 * z2 * x * drInv7 + 3 * x * drInv5
        tzzy = -15 * z2 * y * drInv7 + 3 * y * drInv5
        txyz = -15 * x * y * z * drInv7

        txxxx = 105 * x2 * x2 * drInv9 - 90 * x2 * drInv7 + 9 * drInv5
        txxxy = 105 * x2 * xy * drInv9 - 45 * xy * drInv7
        txxxz = 105 * x2 * xz * drInv9 - 45 * xz * drInv7
        txxyy = 105 * x2 * y2 * drInv9 - 15 * (x2 + y2) * drInv7 + 3 * drInv5
        txxzz = 105 * x2 * z2 * drInv9 - 15 * (x2 + z2) * drInv7 + 3 * drInv5
        txxyz = 105 * x2 * yz * drInv9 - 15 * yz * drInv7

        tyyyy = 105 * y2 * y2 * drInv9 - 90 * y2 * drInv7 + 9 * drInv5
        tyyyx = 105 * y2 * xy * drInv9 - 45 * xy * drInv7
        tyyyz = 105 * y2 * yz * drInv9 - 45 * yz * drInv7
        tyyzz = 105 * y2 * z2 * drInv9 - 15 * (y2 + z2) * drInv7 + 3 * drInv5
        tyyxz = 105 * y2 * xz * drInv9 - 15 * xz * drInv7

        tzzzz = 105 * z2 * z2 * drInv9 - 90 * z2 * drInv7 + 9 * drInv5
        tzzzx = 105 * z2 * xz * drInv9 - 45 * xz * drInv7
        tzzzy = 105 * z2 * yz * drInv9 - 45 * yz * drInv7
        tzzxy = 105 * z2 * xy * drInv9 - 15 * xy * drInv7

    
    if rank == 0:
        iTensor = drInv
    elif rank == 1:
        iTensor = torch.vstack((
            drInv, -tx,   -ty,   -tz,   
            tx,    -txx,  -txy,  -txz,  
            ty,    -txy,  -tyy,  -tyz,  
            tz,    -txz,  -tyz,  -tzz,  
        )).T.reshape(-1, 4, 4)
    elif rank == 2:
        iTensor = torch.vstack((
            drInv, -tx,   -ty,   -tz,   txx,   txy,   txz,   tyy,   tyz,   tzz,
            tx,    -txx,  -txy,  -txz,  txxx,  txxy,  txxz,  tyyx,  txyz,  tzzx,
            ty,    -txy,  -tyy,  -tyz,  txxy,  tyyx,  txyz,  tyyy,  tyyz,  tzzy,
            tz,    -txz,  -tyz,  -tzz,  txxz,  txyz,  tzzx,  tyyz,  tzzy,  tzzz,
            txx,   -txxx, -txxy, -txxz, txxxx, txxxy, txxxz, txxyy, txxyz, txxzz,
            txy,   -txxy, -tyyx, -txyz, txxxy, txxyy, txxyz, tyyyx, tyyxz, tzzxy,
            txz,   -txxz, -txyz, -tzzx, txxxz, txxyz, txxzz, tyyxz, tzzxy, tzzzx,
            tyy,   -tyyx, -tyyy, -tyyz, txxyy, tyyyx, tyyxz, tyyyy, tyyyz, tyyzz,
            tyz,   -txyz, -tyyz, -tzzy, txxyz, tyyxz, tzzxy, tyyyz, tyyzz, tzzzy,
            tzz,   -tzzx, -tzzy, -tzzz, txxzz, tzzxy, tzzzx, tyyzz, tzzzy, tzzzz
        )).T.reshape(-1, 10, 10)
    else:
        raise NotImplementedError(f"Rank >= {rank} not supported")
    
    return iTensor


def computePairwisePermElecEnergyNoDamp(drVec: torch.Tensor, mPoles_i: torch.Tensor, mPoles_j: torch.Tensor, rank: int = 2):
    """
    Compute permanent electrostatic energy without damping between site i-s and site j-s

    Parameters
    ----------
    drVec: torch.Tensor
        Coordinate vectors from i to j, i.e. coords[j] - coords[i], shape (N x 3)
    mPoles_i: torch.Tensor
        Multipoles of site i, shape (N x 10)
    mPoles_j: torch.Tensor
        Multipoles of site j, shape (N x 10)
    """
    iTensor = computeInteractionTensor(drVec, rank=rank)
    if rank == 0:
        mPoles_i = mPoles_i.flatten()
        mPoles_j = mPoles_j.flatten()
        energies = mPoles_i * mPoles_j * iTensor
    else:
        energies = torch.bmm(mPoles_j.unsqueeze(1), torch.bmm(iTensor, mPoles_i.unsqueeze(2))).flatten()
    return energies


# def computeEletrostaticData(drVec: torch.Tensor, mPoles: torch.Tensor, dampFactors: Optional[List[torch.Tensor]] = None, drInv: Optional[torch.Tensor] = None):
#     """
#     drVec: N x 3
#     mPoles: N x 10
#     dampFactors: 5 x N

#     eData: N x 
#     """
#     if drInv is None:
#         drInv = 1 / torch.norm(drVec, dim=1)
    
#     drInv2 = torch.pow(drInv, 2)
#     drInv3 = drInv2 * drInv
#     drInv5 = drInv3 * drInv2
#     drInv7 = drInv5 * drInv2
#     drInv9 = drInv7 * drInv2

#     if dampFactors is not None:
#         drInv  = drInv  * dampFactors[0]
#         drInv3 = drInv3 * dampFactors[1]
#         drInv5 = drInv5 * dampFactors[2]
#         drInv7 = drInv7 * dampFactors[3]
#         drInv9 = drInv9 * dampFactors[4]

#     drVec2 = torch.pow(drVec, 2)

#     x, y, z = drVec[:, 0], drVec[:, 1], drVec[:, 2]
#     x2, y2, z2 = drVec2[:, 0], drVec2[:, 1], drVec2[:, 2]
#     xy, xz, yz = x * y, x * z, y * z

#     # Interaction Tensors
#     tx, ty, tz = -x * drInv3, -y * drInv3, -z * drInv3
    
#     txx = 3 * x2 * drInv5 - drInv3
#     txy = 3 * xy * drInv5
#     txz = 3 * xz * drInv5
#     tyy = 3 * y2 * drInv5 - drInv3
#     tyz = 3 * yz * drInv5
#     tzz = 3 * z2 * drInv5 - drInv3

#     txxx = -15 * x2 * x * drInv7 + 9 * x * drInv5
#     txxy = -15 * x2 * y * drInv7 + 3 * y * drInv5
#     txxz = -15 * x2 * z * drInv7 + 3 * z * drInv5
#     tyyy = -15 * y2 * y * drInv7 + 9 * y * drInv5
#     tyyx = -15 * y2 * x * drInv7 + 3 * x * drInv5
#     tyyz = -15 * y2 * z * drInv7 + 3 * z * drInv5
#     tzzz = -15 * z2 * z * drInv7 + 9 * z * drInv5
#     tzzx = -15 * z2 * x * drInv7 + 3 * x * drInv5
#     tzzy = -15 * z2 * y * drInv7 + 3 * y * drInv5
#     txyz = -15 * x * y * z * drInv7

#     txxxx = 105 * x2 * x2 * drInv9 - 90 * x2 * drInv7 + 9 * drInv5
#     txxxy = 105 * x2 * xy * drInv9 - 45 * xy * drInv7
#     txxxz = 105 * x2 * xz * drInv9 - 45 * xz * drInv7
#     txxyy = 105 * x2 * y2 * drInv9 - 15 * (x2 + y2) * drInv7 + 3 * drInv5
#     txxzz = 105 * x2 * z2 * drInv9 - 15 * (x2 + z2) * drInv7 + 3 * drInv5
#     txxyz = 105 * x2 * yz * drInv9 - 15 * yz * drInv7

#     tyyyy = 105 * y2 * y2 * drInv9 - 90 * y2 * drInv7 + 9 * drInv5
#     tyyyx = 105 * y2 * xy * drInv9 - 45 * xy * drInv7
#     tyyyz = 105 * y2 * yz * drInv9 - 45 * yz * drInv7
#     tyyzz = 105 * y2 * z2 * drInv9 - 15 * (y2 + z2) * drInv7 + 3 * drInv5
#     tyyxz = 105 * y2 * xz * drInv9 - 15 * xz * drInv7

#     tzzzz = 105 * z2 * z2 * drInv9 - 90 * z2 * drInv7 + 9 * drInv5
#     tzzzx = 105 * z2 * xz * drInv9 - 45 * xz * drInv7
#     tzzzy = 105 * z2 * yz * drInv9 - 45 * yz * drInv7
#     tzzxy = 105 * z2 * xy * drInv9 - 15 * xy * drInv7

#     mPolesScaled = mPoles * torch.tensor([1, 1, 1, 1, 1/3, 2/3, 2/3, 1/3, 2/3, 1/3])
#     iTensor = torch.vstack((
#         drInv, -tx,   -ty,   -tz,   txx,   txy,   txz,   tyy,   tyz,   tzz,
#         tx,    -txx,  -txy,  -txz,  txxx,  txxy,  txxz,  tyyx,  txyz,  tzzx,
#         ty,    -txy,  -tyy,  -tyz,  txxy,  tyyx,  txyz,  tyyy,  tyyz,  tzzy,
#         tz,    -txz,  -tyz,  -tzz,  txxz,  txyz,  tzzx,  tyyz,  tzzy,  tzzz,
#         txx,   -txxx, -txxy, -txxz, txxxx, txxxy, txxxz, txxyy, txxyz, txxzz,
#         txy,   -txxy, -tyyx, -txyz, txxxy, txxyy, txxyz, tyyyx, tyyxz, tzzxy,
#         txz,   -txxz, -txyz, -tzzx, txxxz, txxyz, txxzz, tyyxz, tzzxy, tzzzx,
#         tyy,   -tyyx, -tyyy, -tyyz, txxyy, tyyyx, tyyxz, tyyyy, tyyyz, tyyzz,
#         tyz,   -txyz, -tyyz, -tzzy, txxyz, tyyxz, tzzxy, tyyyz, tyyzz, tzzzy,
#         tzz,   -tzzx, -tzzy, -tzzz, txxzz, tzzxy, tzzzx, tyyzz, tzzzy, tzzzz
#     )).T.reshape(-1, 10, 10)
#     eData = torch.bmm(iTensor, mPolesScaled.unsqueeze(2))
#     return eData


# def computePermElectEnergyDamped(
#     mPoles_i: torch.Tensor, mPoles_j: torch.Tensor, 
#     Z_i: torch.Tensor, Z_j: torch.Tensor, 
#     b_i: torch.Tensor, b_j: torch.Tensor, 
#     drVec: torch.Tensor
# ):
#     """
#     Compute damped permanent electrostatic energy
#     """
#     b_ij = torch.sqrt(b_i * b_j)
#     dr = torch.norm(drVec, dim=1)
#     drInv = 1 / dr
#     oneCenterDamps_i = computePermElecOneCenterDampFactors(dr, b_i)
#     oneCenterDamps_j = computePermElecOneCenterDampFactors(dr, b_j)
#     twoCenterDamps = computePermElecTwoCenterDampFactors(dr, b_ij)

#     mPoles_j_scaled = mPoles_j * torch.tensor([1, 1, 1, 1, 1/3, 2/3, 2/3, 1/3, 2/3, 1/3])
#     # core-core interactions
#     ccEnergies = Z_i * Z_j * drInv
#     # shell-shell interactions
#     eDataTwoCenter = computeEletrostaticData(drVec, mPoles_i, twoCenterDamps, drInv)
#     ssEnergies = torch.bmm(mPoles_j_scaled.unsqueeze(1), eDataTwoCenter).flatten()
#     # core-shell interactions
#     ePot_i = computeEletrostaticData(drVec,  mPoles_i, oneCenterDamps_i, drInv)[:, 0].flatten() # elec potential caused by site i-s
#     ePot_j = computeEletrostaticData(-drVec, mPoles_j, oneCenterDamps_j, drInv)[:, 0].flatten() # elec potential caused by site j-s
#     scEnergies = ePot_i * Z_j + ePot_j * Z_i
#     energies = ccEnergies + ssEnergies + scEnergies
#     return energies, eDataTwoCenter

    
if __name__ == '__main__':
    torch.manual_seed(42)
    N = 10
    # drVec = torch.rand(N, 3)
    # mPoles = torch.rand(N, 10)
    # print(mPoles.unsqueeze(1).shape)
    # eData = computeEletrostaticData(drVec, mPoles)
    # print(eData)
    # print(eData[:, 0].flatten())
    # enes = computePermElecEnergyNoDamp(mPoles, mPoles * 2, drVec)
    # print(enes)
    # coords = torch.rand(8, 3)
    # print(computeLocal2GlobalRotationMatrix(
    #     coords[:2, :],
    #     coords[2:4, :],
    #     coords[4:6, :],
    #     coords[6:, :],
    #     axisTypes=torch.LongTensor([0, 1])
    # ))
    # quad_s = torch.rand(N, 5)
    # print(quad_s)
    # print(computeCartesianQuadrupoles(quad_s))
