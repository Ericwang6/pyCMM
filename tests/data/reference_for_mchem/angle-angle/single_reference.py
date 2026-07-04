import pytest
import torch
import sys
import openmm.app as app

sys.path.insert(0, "/pscratch/sd/a/asa/software/pyCMM")
from cmm.bonded import computeAngle, computeBondBondCoupling
from cmm.units import BOHR2NM
from scipy import constants

BOHR2ANG = constants.value("atomic unit of length") * 1e10
ANG2BOHR = 1.0 / BOHR2ANG

# Single ethane coordinates in Angstroms (from PDB)
COORDS_ANG = [
    [19.706, 15.064, 10.638],  # 0: C1
    [19.560, 15.608,  9.208],  # 1: C2
    [19.053, 15.610, 11.324],  # 2: H11
    [20.737, 15.173, 10.981],  # 3: H12
    [19.438, 14.006, 10.674],  # 4: H13
    [18.534, 15.481,  8.856],  # 5: H21
    [19.810, 16.670,  9.176],  # 6: H22
    [20.227, 15.075,  8.527],  # 7: H23
]

# T1 parameters: H-C-C / H-C-C (ctype=1)
AA_THETA_EQ_1_T1 = 1.94202996  # rad
AA_THETA_EQ_2_T1 = 1.94202996  # rad
AA_K_T1          = -0.0264544521

# Atoms for this term:
# Angle 1: H11(2) - C1(0) - C2(1)  →  indices [2, 0, 1]
# Angle 2: H12(3) - C1(0) - C2(1)  →  indices [3, 0, 1]

@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float32),
    ('cuda', torch.float64),
])
def test_angle_angle_reference_single(device, dtype):
    """Reference energy, forces, and virial for one angle-angle T1 term - single ethane.

    Term: H11-C1-C2 / H12-C1-C2  (T1 ctype=1)
    Atoms (0-indexed): angle1=[2,0,1], angle2=[3,0,1]
    """

    coords = torch.tensor(COORDS_ANG, device=device, dtype=dtype) * ANG2BOHR
    coords = coords.requires_grad_(True)

    # Angle indices
    idx1 = torch.tensor([[2, 0, 1]], device=device, dtype=torch.int32)  # H11-C1-C2
    idx2 = torch.tensor([[3, 0, 1]], device=device, dtype=torch.int32)  # H12-C1-C2

    thetaeq1 = torch.tensor(AA_THETA_EQ_1_T1, device=device, dtype=dtype)
    thetaeq2 = torch.tensor(AA_THETA_EQ_2_T1, device=device, dtype=dtype)
    k        = torch.tensor(AA_K_T1,           device=device, dtype=dtype)

    # Compute angles
    theta1 = computeAngle(coords, idx1, box=None, boxInv=None)
    theta2 = computeAngle(coords, idx2, box=None, boxInv=None)

    # E = K * (cos(theta1) - cos(theta1_eq)) * (cos(theta2) - cos(theta2_eq))
    energy = computeBondBondCoupling(
        torch.cos(theta1), torch.cos(theta2),
        torch.cos(thetaeq1), torch.cos(thetaeq2),
        k
    ).sum()

    print(f"\n{'='*60}")
    print(f"Angle-Angle Reference - Single Ethane ({dtype})")
    print(f"{'='*60}")
    print(f"Angle 1 (H11-C1-C2): {theta1.item():.15f} rad  ({torch.rad2deg(theta1).item():.8f} deg)")
    print(f"Angle 2 (H12-C1-C2): {theta2.item():.15f} rad  ({torch.rad2deg(theta2).item():.8f} deg)")
    print(f"theta_eq:            {AA_THETA_EQ_1_T1:.15f} rad  ({torch.rad2deg(thetaeq1).item():.8f} deg)")
    print(f"cos(theta1):         {torch.cos(theta1).item():.15f}")
    print(f"cos(theta2):         {torch.cos(theta2).item():.15f}")
    print(f"cos(thetaeq):        {torch.cos(thetaeq1).item():.15f}")
    print(f"K:                   {AA_K_T1:.15f}")
    print(f"Energy:              {energy.item():.17f}")

    energy.backward()
    forces = -coords.grad.clone()
    virial = -torch.einsum('ij,ik->jk', coords, forces)

    labels = ['C1', 'C2', 'H11', 'H12', 'H13', 'H21', 'H22', 'H23']
    print(f"\nForces on atoms:")
    for i, lbl in enumerate(labels):
        print(f"  {lbl}: [{forces[i,0].item():20.15f}, "
              f"{forces[i,1].item():20.15f}, "
              f"{forces[i,2].item():20.15f}]")
    fs = forces.sum(0)
    print(f"  Sum: [{fs[0].item():.6e}, {fs[1].item():.6e}, {fs[2].item():.6e}]")

    print(f"\nVirial Tensor:")
    for i in range(3):
        print(f"  [{virial[i,0].item():20.15f}, "
              f"{virial[i,1].item():20.15f}, "
              f"{virial[i,2].item():20.15f}]")
    print(f"Virial trace: {virial.trace().item():.17f}")
    print(f"{'='*60}\n")

    assert torch.allclose(forces.sum(0), torch.zeros(3, device=device, dtype=dtype), atol=1e-5)

    return {
        'energy': energy.item(),
        'forces': forces.detach().cpu().numpy(),
        'virial': virial.detach().cpu().numpy(),
    }
