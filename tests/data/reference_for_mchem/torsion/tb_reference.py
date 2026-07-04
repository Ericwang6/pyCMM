import pytest
import torch
import sys

sys.path.insert(0, "/pscratch/sd/a/asa/software/pyCMM")
from cmm.bonded import computeTorsion, computeBond
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
# V = sum_n K_n * (b - b_eq) * (1 + cos(n*phi - phase_n))
TB_PARAMS = [
    # (n, phase, k_tb)
    (1, 0.0,         -0.000431551244),
    (2, 3.14159265,   0.000510807173),
    (3, 0.0,         -0.000481548426),
    (4, 3.14159265,  -0.000145481971),
]
TB_R_EQ = 2.8846949  # Bohr, C1-C2 bond

# Torsion: H11(2) - C1(0) - C2(1) - H21(5)
TORSION_INDICES = [2, 0, 1, 5]
# Coupled bond (ctype=1): C1(0) - C2(1)
BOND_INDICES    = [0, 1]

@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float32),
    ('cuda', torch.float64),
])
def test_torsion_bond_reference_single(device, dtype):
    """Reference energy, forces, and virial for one TB term - single ethane.

    Term: H11-C1-C2-H21, ctype=1
    Torsion indices: [2, 0, 1, 5]
    Bond (C1-C2):    [0, 1]
    V = sum_n K_n * (b - b_eq) * (1 + cos(n*phi - phase_n))
    All units: Bohr, Hartree, radians
    """

    coords = torch.tensor(COORDS_ANG, device=device, dtype=dtype) * ANG2BOHR
    coords = coords.requires_grad_(True)

    tors_idx = torch.tensor([TORSION_INDICES], device=device, dtype=torch.int32)
    bond_idx = torch.tensor([BOND_INDICES],    device=device, dtype=torch.int32)

    phi  = computeTorsion(coords, tors_idx, box=None, boxInv=None)
    bond = computeBond(coords,   bond_idx,  box=None, boxInv=None)

    r_eq = torch.tensor(TB_R_EQ, device=device, dtype=dtype)

    db = bond - r_eq

    energy = torch.zeros(1, device=device, dtype=dtype)
    for (n, phase, k_tb) in TB_PARAMS:
        k_t     = torch.tensor(k_tb,  device=device, dtype=dtype)
        phase_t = torch.tensor(phase, device=device, dtype=dtype)
        energy  = energy + k_t * db * (1.0 + torch.cos(n * phi - phase_t))
    energy = energy.sum()

    print(f"\n{'='*60}")
    print(f"Torsion-Bond Reference - Single Ethane ({dtype})")
    print(f"{'='*60}")
    print(f"Phi  (H11-C1-C2-H21): {phi.item():.15f} rad")
    print(f"Bond (C1-C2):         {bond.item():.15f} Bohr")
    print(f"r_eq:                 {TB_R_EQ:.15f} Bohr")
    print(f"(b - b_eq):           {db.item():.15f}")
    for (n, phase, k_tb) in TB_PARAMS:
        contrib = k_tb * db.item() \
                       * (1.0 + torch.cos(torch.tensor(n * phi.item() - phase)).item())
        print(f"  n={n}: k={k_tb:+.12f} phase={phase:.8f} contrib={contrib:.15f}")
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
