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

# h.eth - c.eth - c.eth - h.eth, ctype=1
# V = sum_n K_n * (cos(theta) - cos(theta_eq)) * (1 + cos(n*phi - phase_n))
TA_PARAMS = [
    # (n, phase, k_ta)
    (1, 0.0,         -7.11543042e-05),
    (2, 3.14159265,  -0.00139752683),
    (3, 0.0,         -0.000204159274),
    (4, 3.14159265,  -0.000858753727),
]
TA_THETA_EQ = 1.94202996  # rad

# Torsion: H11(2) - C1(0) - C2(1) - H21(5)
TORSION_INDICES = [2, 0, 1, 5]
# Coupled angle (ctype=1): C1(0) - C2(1) - H21(5)  (at l end)
ANGLE_INDICES   = [0, 1, 5]

@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float32),
    ('cuda', torch.float64),
])
def test_torsion_angle_reference_single(device, dtype):
    """Reference energy, forces, and virial for one TA term - single ethane.

    Term: H11-C1-C2-H21, ctype=1
    Torsion indices: [2, 0, 1, 5]
    Angle (at C2):   [0, 1, 5]  = C1-C2-H21
    V = sum_n K_n * (cos(theta) - cos(theta_eq)) * (1 + cos(n*phi - phase_n))
    All units: Bohr, Hartree, radians
    """

    coords = torch.tensor(COORDS_ANG, device=device, dtype=dtype) * ANG2BOHR
    coords = coords.requires_grad_(True)

    tors_idx  = torch.tensor([TORSION_INDICES], device=device, dtype=torch.int32)
    angle_idx = torch.tensor([ANGLE_INDICES],   device=device, dtype=torch.int32)

    phi   = computeTorsion(coords, tors_idx,  box=None, boxInv=None)
    theta = computeAngle(coords,   angle_idx, box=None, boxInv=None)

    theta_eq = torch.tensor(TA_THETA_EQ, device=device, dtype=dtype)

    cos_theta    = torch.cos(theta)
    cos_theta_eq = torch.cos(theta_eq)

    energy = torch.zeros(1, device=device, dtype=dtype)
    for (n, phase, k_ta) in TA_PARAMS:
        k_t     = torch.tensor(k_ta,  device=device, dtype=dtype)
        phase_t = torch.tensor(phase, device=device, dtype=dtype)
        energy  = energy + k_t * (cos_theta - cos_theta_eq) \
                               * (1.0 + torch.cos(n * phi - phase_t))
    energy = energy.sum()

    print(f"\n{'='*60}")
    print(f"Torsion-Angle Reference - Single Ethane ({dtype})")
    print(f"{'='*60}")
    print(f"Phi   (H11-C1-C2-H21): {phi.item():.15f} rad")
    print(f"Theta (C1-C2-H21):     {theta.item():.15f} rad")
    print(f"theta_eq:              {TA_THETA_EQ:.15f} rad")
    print(f"cos(theta):            {cos_theta.item():.15f}")
    print(f"cos(theta_eq):         {cos_theta_eq.item():.15f}")
    print(f"cos(theta)-cos(eq):    {(cos_theta - cos_theta_eq).item():.15f}")
    for (n, phase, k_ta) in TA_PARAMS:
        contrib = k_ta * (cos_theta - cos_theta_eq).item() \
                       * (1.0 + torch.cos(torch.tensor(n * phi.item() - phase)).item())
        print(f"  n={n}: k={k_ta:+.12f} phase={phase:.8f} contrib={contrib:.15f}")
    print(f"Energy: {energy.item():.17f} Hartree")

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
