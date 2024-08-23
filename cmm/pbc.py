import torch


def applyPBC(drVecs: torch.Tensor, box: torch.Tensor, boxInv: torch.Tensor):
    """
    Apply peroidic boundary conditions to a set of vectors

    Parameters
    ----------
    drVecs: torch.Tensor
        Real space vectors in Cartesian, with shape (N, 3)
    box: torch.Tensor
        Simulation box, with axes arranged in rows, shape (3, 3)
    boxInv: torch.Tensor
        Inverse of the simulation box matrix, with axes arranged in rows, shape (3, 3)
    """
    dsVecs = torch.matmul(drVecs, boxInv)
    dsVecsPBC = dsVecs - torch.floor(dsVecs + 0.5)
    drVecsPBC = torch.matmul(dsVecsPBC, box)
    return drVecsPBC