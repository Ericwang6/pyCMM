import math
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
from cmm.sph_pme_helper import PME_Spherical

torch.set_printoptions(precision=8)
torch.set_default_dtype(torch.float64)

def spherical_fg_to_cartesian(sph_fg: torch.Tensor):
    """
    Converts the 5-component Spherical Field Gradient back to
    the 3x3 Cartesian Field Gradient tensor for verification.

    Input: sph_fg (N, 5) [Axial, TiltX, TiltY, PlanarCos, PlanarSin]
    Output: cart_fg (N, 3, 3) [[Vxx, Vxy, Vxz], [Vxy, Vyy, Vyz], [Vxz, Vyz, Vzz]]
    """
    # Unpack components
    # Note: Depending on your specific harmonics code, 1 and 2 might be swapped.
    # Based on your Field output (Z, X, Y), we assume:
    # 0=ZZ, 1=XZ, 2=YZ, 3=XX-YY, 4=XY

    fg_0  = sph_fg[:, 0] # Axial
    fg_1c = sph_fg[:, 1] # Tilt X
    fg_1s = sph_fg[:, 2] # Tilt Y
    fg_2c = sph_fg[:, 3] # Planar Diag
    fg_2s = sph_fg[:, 4] # Planar Off-Diag

    rt3 = math.sqrt(3)

    # 1. Recover V_zz
    # Factor ~2.0 observed in data
    v_zz = fg_0 / 2.0

    # 2. Recover Off-Diagonals
    # Factor 2*sqrt(3) observed in data
    v_xz = fg_1c / (2.0 * rt3)
    v_yz = fg_1s / (2.0 * rt3)
    v_xy = fg_2s / (2.0 * rt3)

    # 3. Recover Planar Terms (V_xx, V_yy)
    # Using Trace = 0 constraint: V_xx + V_yy = -V_zz
    # And Planar Diff: V_xx - V_yy = (sqrt(3)/2) * fg_2c
    v_sum  = -v_zz
    v_diff = fg_2c * (rt3 / 2.0)

    v_xx = 0.5 * (v_sum + v_diff)
    v_yy = 0.5 * (v_sum - v_diff)

    # 4. Construct 3x3 Matrices
    # Shape (N, 3, 3)
    out = torch.zeros(sph_fg.shape[0], 3, 3, device=sph_fg.device)

    # Fill Diagonal
    out[:, 0, 0] = v_xx
    out[:, 1, 1] = v_yy
    out[:, 2, 2] = v_zz

    # Fill Off-Diagonal (Symmetric)
    out[:, 0, 1] = v_xy; out[:, 1, 0] = v_xy
    out[:, 0, 2] = v_xz; out[:, 2, 0] = v_xz
    out[:, 1, 2] = v_yz; out[:, 2, 1] = v_yz

    return out
def test_pure_pme_consistency(coords, box, q, p, t, alpha, K, rank):
    print(f"\n{'='*60}")
    print(f" PME CONSISTENCY CHECK (N={q.shape[0]})")
    print(f" Alpha={alpha}, Grid={K}, Rank={rank}")
    print(f"{'='*60}")
    print(f"ORIGINAL QUAD TENSOR 1st ATOM: {t[0]}")

    device = coords.device

    # ---------------------------------------------------------
    # 1. TEST TARGET: CARTESIAN PME
    # ---------------------------------------------------------
    print("\n> Running Target: CARTESIAN PME...")
    coords_cart = coords.detach().clone().requires_grad_(True)

    pme_cart = PME(alpha=alpha, max_hkl=K, rank=rank, use_customized_ops=False).to(device)
    out_cart = pme_cart(coords_cart, box, q, p, t)

    # Extract Energies
    U_cart_manual = out_cart[3]  # The one using your sums (term_q + term_p + term_t)
    E_k_cart_gold = out_cart[4]  # The Structure Factor Energy (The Truth)

    # --- Backward Pass 1: Manual Sum ---
    if coords_cart.grad is not None: coords_cart.grad.zero_()
    # IMPORTANT: retain_graph=True so we can backprop E_k afterwards
    U_cart_manual.sum().backward(retain_graph=True)
    F_cart_manual = -coords_cart.grad.detach().clone()

    # --- Backward Pass 2: Structure Factor ---
    coords_cart.grad.zero_() # CRITICAL: Zero out gradients from previous step
    E_k_cart_gold.backward()
    F_cart_gold = -coords_cart.grad.detach().clone()

    # ---------------------------------------------------------
    # 2. REFERENCE: SPHERICAL PME
    # ---------------------------------------------------------
    print("> Running Reference: SPHERICAL PME...")
    coords_sph = coords.detach().clone().requires_grad_(True)

    pme_sph = PME_Spherical(alpha=alpha, max_hkl=K, rank=rank, use_customized_ops=False).to(device)
    out_sph = pme_sph(coords_sph, box, q, p, t)
    t_spherical_in_cart = spherical_fg_to_cartesian(out_sph[2])

    # Extract Energies
    U_sph_manual = out_sph[3]
    E_k_sph_gold = out_sph[4]

    # --- Backward Pass 1: Manual Sum ---
    if coords_sph.grad is not None: coords_sph.grad.zero_()
    U_sph_manual.sum().backward(retain_graph=True)
    F_sph_manual = -coords_sph.grad.detach().clone()

    # --- Backward Pass 2: Structure Factor ---
    coords_sph.grad.zero_()
    E_k_sph_gold.backward()
    F_sph_gold = -coords_sph.grad.detach().clone()

    # ---------------------------------------------------------
    # 3. ANALYSIS & DIAGNOSTICS
    # ---------------------------------------------------------
    print("\n" + "-"*30)
    print(" COMPARISON RESULTS")
    print("-" * 30)
    # Potentials, fields, Field Gradients
    print(f"SPHERICAL PME POTENTIAL     : {out_sph[0][0]}")
    print(f"SPHERICAL PME FIELD         : {out_sph[1][0]}")
    print(f"SPHERICAL PME FG            : {out_sph[2][0]}")
    print(f"SPHERICAL PME FG(CARTESIAN) : {t_spherical_in_cart[0]}")
    print(f"CARTESIAN PME POTENTIAL     : {out_cart[0][0]}")
    print(f"CARTESIAN PME FIELD         : {out_cart[1][0]}")
    print(f"CARTESIAN PME FG            : {out_cart[2][0]}")
    print("\n" + "-"*30)

    print(" INTERNAL CONSISTENCY (Does Manual Sum == Structure Factor?)")
    print("-" * 30)

    # CARTESIAN INTERNAL CHECK
    diff_cart = (F_cart_manual - F_cart_gold).abs().max().item()
    print(f"CARTESIAN Force Discrepancy : {diff_cart:.2e}")
    if diff_cart > 1e-3:
        ratio = F_cart_gold.norm() / F_cart_manual.norm()
        print(f" -> WARNING: Cartesian Manual Sum does not match Structure Factor!")
        print(f" -> Likely wrong pre-factor. Ratio needed: {ratio:.4f}")
    else:
        print(" -> Cartesian Implementation is CONSISTENT.")

    # SPHERICAL INTERNAL CHECK
    diff_sph = (F_sph_manual - F_sph_gold).abs().max().item()
    print(f"SPHERICAL Force Discrepancy : {diff_sph:.2e}")
    if diff_sph > 1e-3:
        ratio = F_sph_gold.norm() / F_sph_manual.norm()
        print(f" -> WARNING: Spherical Manual Sum does not match Structure Factor!")
        print(f" -> Likely wrong pre-factor. Ratio needed: {ratio:.4f}")
    else:
        print(" -> Spherical Implementation is CONSISTENT.")

    print("\n" + "-"*30)
    print(" CROSS-IMPLEMENTATION CHECK (Cartesian vs Spherical)")
    print("-" * 30)

    # Check Gold vs Gold (The ultimate test)
    e_diff_gold = abs(E_k_cart_gold.item() - E_k_sph_gold.item())
    print(f"Energy (Structure Factor) Diff : {e_diff_gold:.2e}")
    #Check sum energy vs sum energy
    print(f"Energy Manual Sum Spherical: {U_sph_manual}")
    print(f"Energy Manual Sum Cartesian: {U_cart_manual}")
    e_diff_sum = abs(U_cart_manual.item() - U_sph_manual.item())
    print(f"Energy (Manual Sum) Diff : {e_diff_gold:.2e}")

    f_diff_gold = (F_cart_gold - F_sph_gold).norm().item()
    print(f"Force (Structure Factor) Norm Diff : {f_diff_gold:.2e}")

    # Check Manual vs Manual
    f_diff_manual = (F_cart_manual - F_sph_manual).norm().item()
    print(f"Force (Manual Sum) Norm Diff       : {f_diff_manual:.2e}")

    # Single Atom Values for Debugging
    print(f"\nAtom 0 Force  (Gold Standard):")
    print(f"Cart: {F_cart_gold[0]} | Sph: {F_sph_gold[0]}")
    print(f"\nAtom 0 Force  (From Sum:")
    print(f"Cart: {F_cart_manual[0]} | Sph: {F_sph_manual[0]}")

def run_test(rank_val, no_monopoles=False, no_dipoles=False):
    # Setup Paths
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    ff_path = os.path.join(data_dir, 'water_test.xml')
    pdb_path = os.path.join(data_dir, 'water_216.pdb')

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
    print("FIRST ATOMS MULTIPOLES")
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

    # t (Quadrupoles) - Extract as standard 3x3 Cartesian
    q_cols = dim - 4
    t = multipoles[:,4:10]

    if q_cols == 6:
        print(f"  > Detected 6-component Quadrupoles (Row-Major: xx, xy, xz, yy, yz, zz)")
        print("Creating full 3x3 matrix of Quadrupoles")
        
        # Raw data: [Qxx, Qxy, Qxz, Qyy, Qyz, Qzz]
        # Indices:    0    1    2    3    4    5
        raw_sym = multipoles[:, 4:10]
        
        t_cart = torch.zeros(N_atoms, 3, 3, device=device, dtype=multipoles.dtype)
        
        # --- DIAGONALS ---
        t_cart[:, 0, 0] = raw_sym[:, 0]  # Qxx (Index 0)
        t_cart[:, 1, 1] = raw_sym[:, 3]  # Qyy (Index 3)
        t_cart[:, 2, 2] = raw_sym[:, 5]  # Qzz (Index 5)

        # --- OFF-DIAGONALS (Symmetric) ---
        # Qxy (Index 1)
        t_cart[:, 0, 1] = raw_sym[:, 1]
        t_cart[:, 1, 0] = raw_sym[:, 1]
        
        # Qxz (Index 2)
        t_cart[:, 0, 2] = raw_sym[:, 2]
        t_cart[:, 2, 0] = raw_sym[:, 2]
        
        # Qyz (Index 4)
        t_cart[:, 1, 2] = raw_sym[:, 4]
        t_cart[:, 2, 1] = raw_sym[:, 4]

    elif q_cols == 9:
        # Full 3x3 stored flat (If this format is used)
        print(f"  > Detected 9-component Quadrupoles (Full)")
        t_cart = multipoles[:, 4:13].reshape(N_atoms, 3, 3).contiguous()
    else:
        # Default fallback
        print(f"  > No Quadrupoles detected. Using Zeros.")
        t_cart = torch.zeros(N_atoms, 3, 3, device=device, dtype=multipoles.dtype)

    # --- 3. RUN PME TEST ---
    # We pass t_cart (3x3) directly. The PME class converts it.
    alpha_val = 2.8
    K_val = 64

    test_pure_pme_consistency(
        coords_ref,
        box_ref,
        q_test,
        p_test,
        t_cart,
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
    #run_test(1)
    print("\n\n\n ALL MULTIPOLES")
    run_test(2)

~                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
~                                                                                                                                                                           
