import pytest
import torch
import sys

sys.path.insert(0, "/pscratch/sd/a/asa/software/pyCMM")
from cmm.bonded import computeTorsion, computeAngle
from cmm.units import BOHR2ANG
ANG2BOHR = 1.0 / BOHR2ANG

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

# h.eth - c.eth - c.eth - h.eth TAA parameters
# V = K * (theta1 - theta_eq1) * (theta2 - theta_eq2) * cos(phi)
K_TAA          = -0.0172299875
TAA_THETA_EQ_1 = 1.94202996  # rad, angle at j (C1) end
TAA_THETA_EQ_2 = 1.94202996  # rad, angle at k (C2) end

# Torsion: H11(2) - C1(0) - C2(1) - H21(5)
TORSION_INDICES = [2, 0, 1, 5]
# Angle 1: H11(2) - C1(0) - C2(1)  (flanking torsion at j=C1)
ANGLE1_INDICES  = [2, 0, 1]
# Angle 2: C1(0) - C2(1) - H21(5)  (flanking torsion at k=C2)
ANGLE2_INDICES  = [0, 1, 5]

@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float32),
    ('cuda', torch.float64),
])
def test_torsion_angle_angle_reference_single(device, dtype):
    """Reference energy, forces, and virial for one TAA term - single ethane.

    Term: H11-C1-C2-H21
    Torsion indices: [2, 0, 1, 5]
    Angle1 (at C1):  [2, 0, 1]
    Angle2 (at C2):  [0, 1, 5]
    V = K * (theta1 - theta_eq1) * (theta2 - theta_eq2) * cos(phi)
    All units: Bohr, Hartree, radians
    """

    coords = torch.tensor(COORDS_ANG, device=device, dtype=dtype) * ANG2BOHR
    coords = coords.requires_grad_(True)

    tors_idx   = torch.tensor([TORSION_INDICES], device=device, dtype=torch.int32)
    angle1_idx = torch.tensor([ANGLE1_INDICES],  device=device, dtype=torch.int32)
    angle2_idx = torch.tensor([ANGLE2_INDICES],  device=device, dtype=torch.int32)

    phi    = computeTorsion(coords, tors_idx,   box=None, boxInv=None)
    theta1 = computeAngle(coords,  angle1_idx,  box=None, boxInv=None)
    theta2 = computeAngle(coords,  angle2_idx,  box=None, boxInv=None)

    theta_eq1 = torch.tensor(TAA_THETA_EQ_1, device=device, dtype=dtype)
    theta_eq2 = torch.tensor(TAA_THETA_EQ_2, device=device, dtype=dtype)
    k         = torch.tensor(K_TAA,          device=device, dtype=dtype)

    # V = K * (theta1 - theta_eq1) * (theta2 - theta_eq2) * 1+cos(phi)
    energy = k * (theta1 - theta_eq1) * (theta2 - theta_eq2) * (1+torch.cos(phi))
    energy = energy.sum()

    print(f"\n{'='*60}")
    print(f"Torsion-Angle-Angle Reference - Single Ethane ({dtype})")
    print(f"{'='*60}")
    print(f"Phi    (H11-C1-C2-H21): {phi.item():.15f} rad")
    print(f"Theta1 (H11-C1-C2):     {theta1.item():.15f} rad")
    print(f"Theta2 (C1-C2-H21):     {theta2.item():.15f} rad")
    print(f"theta_eq1:              {TAA_THETA_EQ_1:.15f} rad")
    print(f"theta_eq2:              {TAA_THETA_EQ_2:.15f} rad")
    print(f"(theta1 - eq1):         {(theta1 - theta_eq1).item():.15f}")
    print(f"(theta2 - eq2):         {(theta2 - theta_eq2).item():.15f}")
    print(f"cos(phi):               {torch.cos(phi).item():.15f}")
    print(f"K_TAA:                  {K_TAA:.15f}")
    print(f"Energy: {energy.item():.17f} Hartree")
    dV_dcosphi = K_TAA * (theta1-theta_eq1).item() * (theta2-theta_eq2).item()
    dV_dtheta1 = K_TAA * (theta2-theta_eq2).item() * (1+torch.cos(phi)).item()
    dV_dtheta2 = K_TAA * (theta1-theta_eq1).item() * (1+torch.cos(phi)).item()

    theta1.backward(retain_graph=True)
    dtheta1_di = coords.grad[2].clone()
    dtheta1_dj = coords.grad[0].clone()
    dtheta1_dk = coords.grad[1].clone()
    print("dtheta1/dr_i (H11):", dtheta1_di)
    print("dtheta1/dr_j (C1):", dtheta1_dj)
    print("dtheta1/dr_k (C2):", dtheta1_dk)
    coords.grad.zero_()

    theta2.backward(retain_graph=True)
    dtheta2_dj = coords.grad[0].clone()
    dtheta2_dk = coords.grad[1].clone()
    dtheta2_dl = coords.grad[5].clone()
    print("dtheta2/dr_j (C1):", dtheta2_dj)
    print("dtheta2/dr_k (C2):", dtheta2_dk)
    print("dtheta2/dr_l (H21):", dtheta2_dl)
    coords.grad.zero_()

    # need dcosphi/dr_j — get from torsion backward
    torch.cos(phi).backward(retain_graph=True)
    dcosphi_dj = coords.grad[0].clone()
    coords.grad.zero_()

    contrib_cosphi_j = dV_dcosphi * dcosphi_dj
    contrib_theta1_j = dV_dtheta1 * dtheta1_dj
    contrib_theta2_j = dV_dtheta2 * dtheta2_dj
    print(f"contrib_cosphi_j: {contrib_cosphi_j}")
    print(f"contrib_theta1_j: {contrib_theta1_j}")
    print(f"contrib_theta2_j: {contrib_theta2_j}")
    print(f"force_j expected: {-(contrib_cosphi_j + contrib_theta1_j + contrib_theta2_j)}")

    energy.backward()
    forces = -coords.grad.clone()
    virial = -torch.einsum('ij,ik->jk', coords, forces)

    labels = ['C1', 'C2', 'H11', 'H12', 'H13', 'H21', 'H22', 'H23']
    print(f"\nForces on atoms (Hartree/Bohr):")
    for i, lbl in enumerate(labels):
        print(f"  {lbl}: [{forces[i,0].item():20.15f}, "
              f"{forces[i,1].item():20.15f}, "
              f"{forces[i,2].item():20.15f}]")
    fs = forces.sum(0)
    print(f"  Sum: [{fs[0].item():.6e}, {fs[1].item():.6e}, {fs[2].item():.6e}]")

    print(f"\nVirial Tensor (Hartree):")
    for i in range(3):
        print(f"  [{virial[i,0].item():20.15f}, "
              f"{virial[i,1].item():20.15f}, "
              f"{virial[i,2].item():20.15f}]")
    print(f"Virial trace: {virial.trace().item():.17f}")
    print(f"{'='*60}\n")

    assert torch.allclose(
        forces.sum(0),
        torch.zeros(3, device=device, dtype=dtype),
        atol=1e-5
    ), "Forces do not sum to zero — momentum conservation violated"

    return {
        'energy':  energy.item(),
        'forces':  forces.detach().cpu().numpy(),
        'virial':  virial.detach().cpu().numpy(),
    }
