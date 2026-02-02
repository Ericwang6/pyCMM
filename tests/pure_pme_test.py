import torch
import os
import sys
import numpy as np
from pprint import pprint as pp

# --- PATH SETUP ---
sys.path.insert(0, "/pscratch/sd/a/asa/software/pyCMM")

import openmm.app as app
from cmm.ffxml import ForceFieldXML
from cmm.topology import Topology
from cmm.units import BOHR2NM

from cmm.pme_helper import PME
from cmm.ewald import Ewald

torch.set_printoptions(precision=8)
torch.set_default_dtype(torch.float64)

def test_pure_pme_consistency(coords, box, q, p, t, alpha, K, rank):
    """
    Runs the exact comparison logic for PME.
    Inputs (q, p, t) are expected to be standard Cartesian tensors.
    """
    print(f"\n> RUNNING PURE PME KERNEL CHECK (N={q.shape[0]})...")
    print(f"  alpha={alpha}, K={K}, rank={rank}")

#    # 1. Reference (Python)
    coords_ref = coords.detach().clone().requires_grad_(True)

    # Initialize PME Module (Python mode)
    # The PME class will handle Cartesian -> Spherical conversion internally
    pme_py = PME(alpha=alpha, max_hkl=K, rank=rank, use_customized_ops=False).to(coords.device)

    # Compute Output
    out_ref = pme_py(coords_ref, box, q, p, t)
    print(f"(PY) POTENTIAL  ATOM 0 {out_ref[0][0]}")
    print(f"(PY) FIELD      ATOM 0 {out_ref[1][0]}")
    print(f"(PY) FIELD GRAD ATOM 0 {out_ref[2][0]}")

    # Unpack based on your PME signature: (phi, E, EG, energy, forces)
    U_ref = out_ref[3]

    # If U_ref is not a scalar sum yet, sum it
    if U_ref.dim() > 0:
        U_ref = U_ref.sum()

    # Compute Reference Forces via Autograd
    if coords_ref.grad is not None: coords_ref.grad.zero_()
    U_ref.backward()
    F_ref = -coords_ref.grad.detach()
    print(f"(PY) FORCE ATOM 0 {F_ref[0]}")
    print(f"(PY) ENERGY       {U_ref}")

    # 2. CUDA Custom Op
    pme_cu = PME(alpha=alpha, max_hkl=K, rank=rank, use_customized_ops=True).to(coords.device)

    # Forward Pass
    out_cu = pme_cu(coords, box, q, p, t)
    U_cu = out_cu[3]
    F_cu = out_cu[4]
    print(f"(CUDA) POTENTIAL  ATOM 0 {out_cu[0][0]}")
    print(f"(CUDA) FIELD      ATOM 0 {out_cu[1][0]}")
    print(f"(CUDA) FIELD GRAD ATOM 0 {out_cu[2][0]}")
    print(f"(CUDA) FORCE      ATOM 0 {out_cu[4][0]}")
    print(f"(CUDA) ENERGY            {out_cu[3]}")
    # 3. Compare
    e_diff = (U_ref - U_cu).abs().item()
    f_diff = (F_ref - F_cu).abs().max().item()

    print(f"  [Energy] Ref: {U_ref.item():.5f} | CUDA: {U_cu.item():.5f} | Diff: {e_diff:.2e}")
    print(f"  [Forces] Max Diff: {f_diff:.2e}")

    if e_diff < 1e-6:
        print("  >>> SUCCESS: PME Energy Match!")
    else:
        print("  >>> FAIL: PME Energy Mismatch.")



def run_test(rank_val, no_monopoles=False, no_dipoles=False, no_quadrupoles=False):
    # Setup Paths
    #data_dir = os.path.join(os.path.dirname(__file__), 'data')
    #ff_path = os.path.join(data_dir, 'water_test.xml')
    #pdb_path = os.path.join(data_dir, 'water_216.pdb')
    #pdb_path = os.path.join(data_dir, 'water_1.pdb')
    ff_path = "/pscratch/sd/a/asa/software/pyCMM/tests/data/water_refit.xml"
    pdb_path = "/pscratch/sd/a/asa/software/pyCMM/tests/data/water_216.pdb"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")

    # Load System
    ff = ForceFieldXML(ff_path, device=device)
    pdb = app.PDBFile(pdb_path)
    top = Topology.fromOpenmm(pdb.topology, device)

    # Prepare inputs
    pos_np = pdb.getPositions(asNumpy=True)._value / BOHR2NM
    box_np = np.array([[v.x, v.y, v.z] for v in pdb.topology.getPeriodicBoxVectors()]) / BOHR2NM

    # --- 1. REFERENCE PREP ---
    print("Preparing System (getting multipoles)...")
    coords_ref = torch.tensor(pos_np, device=device, requires_grad=True)
    box_ref = torch.tensor(box_np, device=device, requires_grad=True)

    # Parametrize to get multipoles
    sys_ref = ff.parametrize(top, use_fd_morse=True, use_polarization=False, cutoff_sr=9.0, use_customized_ops=False)
    _ = sys_ref.getEnergy(coords_ref, box_ref)

    # --- 2. EXTRACT CARTESIAN MULTIPOLES ---
    print("> Extracting Multipoles...")
    multipoles = sys_ref.last_perm_multipoles.detach()
    print(multipoles[0])

    N_atoms = multipoles.shape[0]
    dim = multipoles.shape[1]

    # q (N)
    if not no_monopoles:
        q_test = multipoles[:, 0].contiguous()
    else:
        print("SETTING MONOPOLES TO 0")
        q_test = torch.zeros(N_atoms, device=device, dtype=multipoles.dtype)
    # p (N, 3)
    if not no_dipoles:
        p_test = multipoles[:, 1:4].contiguous()
    else:
        print("SETTING DIPOLES TO 0")
        p_test = torch.zeros(N_atoms, 3, device=device, dtype=multipoles.dtype)
    if not no_quadrupoles:
        t_test = multipoles[:,4:10].contiguous()
    else:
        print("SETTING QUADRUPOLES TO 0")
        t_test = torch.zeros(N_atoms, 6, device=device, dtype=multipoles.dtype)


    # --- 3. RUN PME TEST ---
    alpha_val = 2.8
    K_val = 64

    test_pure_pme_consistency(
        coords_ref,
        box_ref,
        q_test,
        p_test,
        t_test,
        alpha=alpha_val,
        K=K_val,
        rank=rank_val
    )

if __name__ == "__main__":
    #print("MONOPOLES ONLY TEST")
    #run_test(0)
    #print("\n\n\nDIPOLES ONLY TEST")
    #run_test(1, True)
    #print("\n\n\nQUADRUPOLES ONLY TEST")
    #run_test(2,True,True)
    #print("\n\n\nMONOPOLES + DIPOLES")
    #run_test(1,False,False,True)
    print("\n\n\nALL MULTIPOLES")
    run_test(2)

