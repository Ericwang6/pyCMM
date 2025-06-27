import math
from typing import List, Optional
import torch
from enum import IntEnum
from .pbc import applyPBC


class AxisTypes(IntEnum):
    ZThenX            = 0
    Bisector          = 1
    ZBisect           = 2
    ThreeFold         = 3
    ZOnly             = 4
    NoAxisType        = 5
    LastAxisTypeIndex = 6


HALF_SQRT3 = math.sqrt(3) / 2


def normVec(vec):
    return vec / torch.norm(vec, dim=1, keepdim=True)


def computeLocal2GlobalRotationMatrix(pos: torch.Tensor, pos1: torch.Tensor, pos2: torch.Tensor, pos3: torch.Tensor, axisTypes: torch.Tensor, box: torch.Tensor=None, boxInv: torch.Tensor=None):
    """
    Compute local to global rotation matrix.
    Axis types are specified as follows:
    0 - Identity
    1 - Z-Then-X
    2 - Bisector
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
    yvec = torch.linalg.cross(zvec, xvec)
    rotMatrix = torch.hstack((xvec, yvec, zvec)).reshape(-1, 3, 3)
    return rotMatrix


def computeLocal2GlobalRotationMatrixBatch(positions, zAtoms, xAtoms, yAtoms, axisTypes, box=None, boxInv=None):
    """
    Compute local to global rotation matrix for a set of atoms

    Parameters
    ----------
    positions: torch.Tensor
        Atom positions, shape (N, 3)
    zAtoms: torch.Tensor[int]
        Atomic indices specifying Z-axis, shape (N,)
    xAtoms: torch.Tensor[int]
        Atomic indices specifying X-axis, shape (N,)
    yAtoms: torch.Tensor[int]
        Atomic indices specifying Y-axis, shape (N,)
    axisTypes: torch.Tensor[int]
        Integers specifying local axis types, shape (N,)
    box: torch.Tensor
        Peroidic box, shape (3, 3), optional
    """

    zVec = applyPBC(positions[zAtoms] - positions, box, boxInv)
    zVec = normVec(zVec)
    xVec = torch.zeros_like(zVec)
    yVec = torch.zeros_like(zVec)

    # Z-Only
    filterZOnly = torch.logical_or(axisTypes == AxisTypes.ZOnly.value, axisTypes == AxisTypes.NoAxisType.value)
    xVecNotZOnly = applyPBC(positions[xAtoms][~filterZOnly] - positions[~filterZOnly], box, boxInv)
    xVec[~filterZOnly] += normVec(xVecNotZOnly)
    xVec[filterZOnly, 0] += 1 - zVec[filterZOnly, 0]
    xVec[filterZOnly, 1] += zVec[filterZOnly, 0]

    # Bisector
    filterBisector = (axisTypes == AxisTypes.Bisector.value)
    if torch.any(filterBisector):
        zVec[filterBisector] += xVec[filterBisector]
        zVec = normVec(zVec)
    
    # Z-Bisect
    filterZBisect = (axisTypes == AxisTypes.ZBisect.value)
    if torch.any(filterZBisect):
        yVecZBisect = applyPBC(positions[yAtoms][filterZBisect] - positions[filterZBisect], box, boxInv)
        yVecZBisect = normVec(yVecZBisect)
        xVecZBisect = normVec(xVec[filterZBisect] + yVecZBisect)
        xVec[filterZBisect] = xVecZBisect
    
    # Threefold
    filterThreeFold = (axisTypes == AxisTypes.ThreeFold.value)
    if torch.any(filterThreeFold):
        yVecThreeFold = applyPBC(positions[yAtoms][filterThreeFold] - positions[filterThreeFold], box, boxInv)
        yVecThreeFold = normVec(yVecThreeFold)
        xVecThreeFold = xVec[filterThreeFold]
        zVecThreeFold = zVec[filterThreeFold]
        zVec[filterThreeFold] = normVec(zVecThreeFold + xVecThreeFold + yVecThreeFold)

    xVec = normVec(xVec - zVec * torch.sum(zVec * xVec, dim=1, keepdim=True))
    yVec = torch.linalg.cross(zVec, xVec)

    # No axis
    filterNoAxis = (axisTypes == AxisTypes.NoAxisType.value)
    if torch.any(filterNoAxis):
        filterNoAxis = filterNoAxis.view(-1, 1)
        zVec = torch.where(filterNoAxis, torch.tensor([0.0, 0.0, 1.0], dtype=zVec.dtype, device=zVec.device), zVec)
        xVec = torch.where(filterNoAxis, torch.tensor([1.0, 0.0, 0.0], dtype=xVec.dtype, device=xVec.device), xVec)
        yVec = torch.where(filterNoAxis, torch.tensor([0.0, 1.0, 0.0], dtype=yVec.dtype, device=yVec.device), yVec)

    rotMatrix = torch.hstack((xVec, yVec, zVec)).reshape(-1, 3, 3)
    return rotMatrix


def scaleMultipoles(
    mPoles: torch.Tensor, 
    monoScales: torch.Tensor, dipoScales: torch.Tensor, quadScales: torch.Tensor,
):
    # The monopoles are set directly from the parameter list while the
    # multipoles are directly scaled versions of the electric multipoles.
    mPolesScaled = torch.zeros_like(mPoles)
    mPolesScaled[:, 0]   += monoScales
    mPolesScaled[:, 1:4] += mPoles[:, 1:4] * dipoScales.unsqueeze(1)
    mPolesScaled[:, 4:]  += mPoles[:, 4:] * quadScales.unsqueeze(1)
    return mPolesScaled


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

def convertMultipolesToPolytensor(mono: torch.Tensor, dipo: torch.Tensor, quad: torch.Tensor):
    """
    Takes already-rotated multipoles and flattens to (N, 10) polytensor with quadrupole
    entries appropriately scaled so that symmetry-equivalent operations are avoided.

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
    return torch.hstack((mono.unsqueeze(1), dipo, quad[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]])) * torch.tensor([1, 1, 1, 1, 1/3, 2/3, 2/3, 1/3, 2/3, 1/3], device=mono.device)


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

def computeSphericalQuadrupoles(quad_c: torch.Tensor):
    # Conversion factors come from Table E.1 of Anthony Stone book
    Q_20  = quad_c[:, 2, 2]
    Q_21c = quad_c[:, 0, 2] / HALF_SQRT3
    Q_21s = quad_c[:, 1, 2] / HALF_SQRT3
    Q_22c = (quad_c[:, 0, 0] - quad_c[:, 1, 1]) / HALF_SQRT3 / 2
    Q_22s = quad_c[:, 0, 1] / HALF_SQRT3
    return torch.vstack((Q_20, Q_21c, Q_21s, Q_22c, Q_22s)).T.reshape(-1, 5)

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

def computeUndampedInteractionTensorBlocks(dist_vecs: torch.Tensor, dists: torch.Tensor):
    XX = torch.zeros_like(dists)
    t_r1 = 1 / dists
    drInv2 = torch.pow(t_r1, 2)
    drInv3 = drInv2 * t_r1
    drInv5 = drInv3 * drInv2
    drInv7 = drInv5 * drInv2
    drInv9 = drInv7 * drInv2

    x, y, z = dist_vecs[:, 0], dist_vecs[:, 1], dist_vecs[:, 2]
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    
    tx_r3, ty_r3, tz_r3 = -x * drInv3, -y * drInv3, -z * drInv3
        
    txx_r5 = 3 * x2 * drInv5
    txy_r5 = 3 * xy * drInv5
    txz_r5 = 3 * xz * drInv5
    tyy_r5 = 3 * y2 * drInv5
    tyz_r5 = 3 * yz * drInv5
    tzz_r5 = 3 * z2 * drInv5
    
    txx_r3 = -drInv3
    tyy_r3 = -drInv3
    tzz_r3 = -drInv3

    txxx_r7 = -15 * x2 * x * drInv7
    txxy_r7 = -15 * x2 * y * drInv7
    txxz_r7 = -15 * x2 * z * drInv7
    tyyy_r7 = -15 * y2 * y * drInv7
    tyyx_r7 = -15 * y2 * x * drInv7
    tyyz_r7 = -15 * y2 * z * drInv7
    tzzz_r7 = -15 * z2 * z * drInv7
    tzzx_r7 = -15 * z2 * x * drInv7
    tzzy_r7 = -15 * z2 * y * drInv7
    txyz_r7 = -15 * x * y * z * drInv7

    txxx_r5 = 9 * x * drInv5
    txxy_r5 = 3 * y * drInv5
    txxz_r5 = 3 * z * drInv5
    tyyy_r5 = 9 * y * drInv5
    tyyx_r5 = 3 * x * drInv5
    tyyz_r5 = 3 * z * drInv5
    tzzz_r5 = 9 * z * drInv5
    tzzx_r5 = 3 * x * drInv5
    tzzy_r5 = 3 * y * drInv5

    txxxx_r9 = 105 * x2 * x2 * drInv9
    txxxy_r9 = 105 * x2 * xy * drInv9
    txxxz_r9 = 105 * x2 * xz * drInv9
    txxyy_r9 = 105 * x2 * y2 * drInv9
    txxzz_r9 = 105 * x2 * z2 * drInv9
    txxyz_r9 = 105 * x2 * yz * drInv9
    tyyyy_r9 = 105 * y2 * y2 * drInv9
    tyyyx_r9 = 105 * y2 * xy * drInv9
    tyyyz_r9 = 105 * y2 * yz * drInv9
    tyyzz_r9 = 105 * y2 * z2 * drInv9
    tyyxz_r9 = 105 * y2 * xz * drInv9
    tzzzz_r9 = 105 * z2 * z2 * drInv9
    tzzzx_r9 = 105 * z2 * xz * drInv9
    tzzzy_r9 = 105 * z2 * yz * drInv9
    tzzxy_r9 = 105 * z2 * xy * drInv9

    txxxx_r7 = -90 * x2 * drInv7
    txxxy_r7 = -45 * xy * drInv7
    txxxz_r7 = -45 * xz * drInv7
    txxyy_r7 = -15 * (x2 + y2) * drInv7
    txxzz_r7 = -15 * (x2 + z2) * drInv7
    txxyz_r7 = -15 * yz * drInv7
    tyyyy_r7 = -90 * y2 * drInv7
    tyyyx_r7 = -45 * xy * drInv7
    tyyyz_r7 = -45 * yz * drInv7
    tyyzz_r7 = -15 * (y2 + z2) * drInv7
    tyyxz_r7 = -15 * xz * drInv7
    tzzzz_r7 = -90 * z2 * drInv7
    tzzzx_r7 = -45 * xz * drInv7
    tzzzy_r7 = -45 * yz * drInv7                
    tzzxy_r7 = -15 * xy * drInv7

    txxxx_r5 = 9 * drInv5
    txxyy_r5 = 3 * drInv5
    txxzz_r5 = 3 * drInv5
    tyyyy_r5 = 9 * drInv5
    tyyzz_r5 = 3 * drInv5
    tzzzz_r5 = 9 * drInv5


    interaction_tensor_13579 = torch.vstack((
        t_r1,     -tx_r3,   -ty_r3,   -tz_r3,   txx_r5,   txy_r5,   txz_r5,   tyy_r5,   tyz_r5,   tzz_r5,
        tx_r3,    -txx_r5,  -txy_r5,  -txz_r5,  txxx_r7,  txxy_r7,  txxz_r7,  tyyx_r7,  txyz_r7,  tzzx_r7,
        ty_r3,    -txy_r5,  -tyy_r5,  -tyz_r5,  txxy_r7,  tyyx_r7,  txyz_r7,  tyyy_r7,  tyyz_r7,  tzzy_r7,
        tz_r3,    -txz_r5,  -tyz_r5,  -tzz_r5,  txxz_r7,  txyz_r7,  tzzx_r7,  tyyz_r7,  tzzy_r7,  tzzz_r7,
        txx_r5,   -txxx_r7, -txxy_r7, -txxz_r7, txxxx_r9, txxxy_r9, txxxz_r9, txxyy_r9, txxyz_r9, txxzz_r9,
        txy_r5,   -txxy_r7, -tyyx_r7, -txyz_r7, txxxy_r9, txxyy_r9, txxyz_r9, tyyyx_r9, tyyxz_r9, tzzxy_r9,
        txz_r5,   -txxz_r7, -txyz_r7, -tzzx_r7, txxxz_r9, txxyz_r9, txxzz_r9, tyyxz_r9, tzzxy_r9, tzzzx_r9,
        tyy_r5,   -tyyx_r7, -tyyy_r7, -tyyz_r7, txxyy_r9, tyyyx_r9, tyyxz_r9, tyyyy_r9, tyyyz_r9, tyyzz_r9,
        tyz_r5,   -txyz_r7, -tyyz_r7, -tzzy_r7, txxyz_r9, tyyxz_r9, tzzxy_r9, tyyyz_r9, tyyzz_r9, tzzzy_r9,
        tzz_r5,   -tzzx_r7, -tzzy_r7, -tzzz_r7, txxzz_r9, tzzxy_r9, tzzzx_r9, tyyzz_r9, tzzzy_r9, tzzzz_r9
    )).T.reshape(-1, 10, 10)

    interaction_tensor_00357 = torch.vstack((
        XX,             XX,       XX,       XX,   txx_r3,       XX,       XX,   tyy_r3,       XX,   tzz_r3,
        XX,        -txx_r3,       XX,       XX,  txxx_r5,  txxy_r5,  txxz_r5,  tyyx_r5,       XX,  tzzx_r5,
        XX,             XX,  -tyy_r3,       XX,  txxy_r5,  tyyx_r5,       XX,  tyyy_r5,  tyyz_r5,  tzzy_r5,
        XX,             XX,       XX,  -tzz_r3,  txxz_r5,       XX,  tzzx_r5,  tyyz_r5,  tzzy_r5,  tzzz_r5,
        txx_r3,   -txxx_r5, -txxy_r5, -txxz_r5, txxxx_r7, txxxy_r7, txxxz_r7, txxyy_r7, txxyz_r7, txxzz_r7,
        XX,       -txxy_r5, -tyyx_r5,       XX, txxxy_r7, txxyy_r7, txxyz_r7, tyyyx_r7, tyyxz_r7, tzzxy_r7,
        XX,       -txxz_r5,       XX, -tzzx_r5, txxxz_r7, txxyz_r7, txxzz_r7, tyyxz_r7, tzzxy_r7, tzzzx_r7,
        tyy_r3,   -tyyx_r5, -tyyy_r5, -tyyz_r5, txxyy_r7, tyyyx_r7, tyyxz_r7, tyyyy_r7, tyyyz_r7, tyyzz_r7,
        XX,             XX, -tyyz_r5, -tzzy_r5, txxyz_r7, tyyxz_r7, tzzxy_r7, tyyyz_r7, tyyzz_r7, tzzzy_r7,
        tzz_r3,   -tzzx_r5, -tzzy_r5, -tzzz_r5, txxzz_r7, tzzxy_r7, tzzzx_r7, tyyzz_r7, tzzzy_r7, tzzzz_r7
    )).T.reshape(-1, 10, 10)

    interaction_tensor_00005 = torch.vstack((
        XX, XX, XX, XX,       XX,       XX,       XX,       XX,       XX,       XX,
        XX, XX, XX, XX,       XX,       XX,       XX,       XX,       XX,       XX,
        XX, XX, XX, XX,       XX,       XX,       XX,       XX,       XX,       XX,
        XX, XX, XX, XX,       XX,       XX,       XX,       XX,       XX,       XX,
        XX, XX, XX, XX, txxxx_r5,       XX,       XX, txxyy_r5,       XX, txxzz_r5,
        XX, XX, XX, XX,       XX, txxyy_r5,       XX,       XX,       XX,       XX,
        XX, XX, XX, XX,       XX,       XX, txxzz_r5,       XX,       XX,       XX,
        XX, XX, XX, XX, txxyy_r5,       XX,       XX, tyyyy_r5,       XX, tyyzz_r5,
        XX, XX, XX, XX,       XX,       XX,       XX,       XX, tyyzz_r5,       XX,
        XX, XX, XX, XX, txxzz_r5,       XX,       XX, tyyzz_r5,       XX, tzzzz_r5
    )).T.reshape(-1, 10, 10)
    
    return interaction_tensor_13579, interaction_tensor_00357, interaction_tensor_00005

def formDampingFactorBlocksRank1(damp_factors: torch.Tensor):
    f1 = damp_factors[0]
    f3 = damp_factors[1]
    f5 = damp_factors[2]
    XX = torch.zeros_like(f1)

    damp_135 = torch.vstack((
        f1, f3, f3, f3,
        f3, f5, f5, f5,
        f3, f5, f5, f5,
        f3, f5, f5, f5,
    )).T.reshape(-1, 4, 4)

    damp_003 = torch.vstack((
        XX, XX, XX, XX,
        XX, f3, XX, XX,
        XX, XX, f3, XX,
        XX, XX, XX, f3,
    )).T.reshape(-1, 4, 4)
    
    return damp_135, damp_003

def formDampingFactorBlocksRank2(damp_factors: torch.Tensor):
    f1 = damp_factors[0]
    f3 = damp_factors[1]
    f5 = damp_factors[2]
    f7 = damp_factors[3]
    f9 = damp_factors[4]
    XX = torch.zeros_like(f1)

    damp_13579 = torch.vstack((
        f1, f3, f3, f3, f5, f5, f5, f5, f5, f5,
        f3, f5, f5, f5, f7, f7, f7, f7, f7, f7,
        f3, f5, f5, f5, f7, f7, f7, f7, f7, f7,
        f3, f5, f5, f5, f7, f7, f7, f7, f7, f7,
        f5, f7, f7, f7, f9, f9, f9, f9, f9, f9,
        f5, f7, f7, f7, f9, f9, f9, f9, f9, f9,
        f5, f7, f7, f7, f9, f9, f9, f9, f9, f9,
        f5, f7, f7, f7, f9, f9, f9, f9, f9, f9,
        f5, f7, f7, f7, f9, f9, f9, f9, f9, f9,
        f5, f7, f7, f7, f9, f9, f9, f9, f9, f9
    )).T.reshape(-1, 10, 10)

    damp_00357 = torch.vstack((
        XX, XX, XX, XX, f3, XX, XX, f3, XX, f3,
        XX, f3, XX, XX, f5, f5, f5, f5, XX, f5,
        XX, XX, f3, XX, f5, f5, XX, f5, f5, f5,
        XX, XX, XX, f3, f5, XX, f5, f5, f5, f5,
        f3, f5, f5, f5, f7, f7, f7, f7, f7, f7,
        XX, f5, f5, XX, f7, f7, f7, f7, f7, f7,
        XX, f5, XX, f5, f7, f7, f7, f7, f7, f7,
        f3, f5, f5, f5, f7, f7, f7, f7, f7, f7,
        XX, XX, f5, f5, f7, f7, f7, f7, f7, f7,
        f3, f5, f5, f5, f7, f7, f7, f7, f7, f7
    )).T.reshape(-1, 10, 10)

    damp_00005 = torch.vstack((
        XX, XX, XX, XX, XX, XX, XX, XX, XX, XX,
        XX, XX, XX, XX, XX, XX, XX, XX, XX, XX,
        XX, XX, XX, XX, XX, XX, XX, XX, XX, XX,
        XX, XX, XX, XX, XX, XX, XX, XX, XX, XX,
        XX, XX, XX, XX, f5, XX, XX, f5, XX, f5,
        XX, XX, XX, XX, XX, f5, XX, XX, XX, XX,
        XX, XX, XX, XX, XX, XX, f5, XX, XX, XX,
        XX, XX, XX, XX, f5, XX, XX, f5, XX, f5,
        XX, XX, XX, XX, XX, XX, XX, XX, f5, XX,
        XX, XX, XX, XX, f5, XX, XX, f5, XX, f5
    )).T.reshape(-1, 10, 10)
    
    return damp_13579, damp_00357, damp_00005

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
