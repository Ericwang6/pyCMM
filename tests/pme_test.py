import torch
import numpy as np
import os
import sys
import math
from pprint import pprint

# --- CONFIGURATION ---
PYCMM_PATH = "/pscratch/sd/a/asa/software/pyCMM"
FF_PATH = f"{PYCMM_PATH}/tests/data/water_refit.xml"
PDB_PATH = f"{PYCMM_PATH}/tests/data/water_216.pdb"
OUTPUT_NPZ = "water_debug_data.npz"

sys.path.insert(0, PYCMM_PATH)

# Import pyCMM components
import openmm.app as app
from cmm.ffxml import ForceFieldXML
from cmm.topology import Topology
from cmm.units import BOHR2NM
from cmm.pme_helper import PME as CMM_PME  

from cmm.pme import compute_pme  # pme.py script

torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_unified_test():
    # --- 1. DATA EXTRACTION (REAL WATER) ---
    print(f"--- Step 1: Extracting real water data from {PDB_PATH} ---")
    ff = ForceFieldXML(FF_PATH, device=device)
    pdb = app.PDBFile(PDB_PATH)
    top = Topology.fromOpenmm(pdb.topology, device)

    coords_raw = torch.tensor(pdb.getPositions(asNumpy=True)._value / BOHR2NM, device=device)
    box_raw = torch.tensor(np.array([[v.x, v.y, v.z] for v in pdb.topology.getPeriodicBoxVectors()]) / BOHR2NM, device=device)

    # Parametrize to get "Real" Multipoles
    sys_ref = ff.parametrize(top, use_fd_morse=True, use_polarization=False, cutoff_sr=9.0)
    _ = sys_ref.getEnergy(coords_raw, box_raw)
    
    # Extract [q, px, py, pz, qxx, qxy, qxz, qyy, qyz, qzz]
    mpoles = sys_ref.last_perm_multipoles.detach()
    print(f"mpoles[0] = {mpoles[0]}")
    
    q = mpoles[:, 0].contiguous()
    p = mpoles[:, 1:4].contiguous()
    t = mpoles[:, 4:10].contiguous()

    # --- 2. SAVE FOR COLLEAGUE ---
    print(f"--- Step 2: Saving data to {OUTPUT_NPZ} ---")
    np.savez(OUTPUT_NPZ,
             coords=coords_raw.cpu().numpy(),
             box=box_raw.cpu().numpy(),
             q=q.cpu().numpy(),
             p=p.cpu().numpy(),
             t=t.cpu().numpy(),
             alpha=2.8,
             K=64)

    # --- 3. REFERENCE CALCULATION (pme.py) ---
    print("--- Step 3: Running Reference Python PME ---")
    coords_ref = coords_raw.clone().requires_grad_(True)
    
    # Instantiate CMM_PME with use_customized_ops=False to force Python path
    pme_py_obj = CMM_PME(alpha=2.8, max_hkl=64, rank=2, use_customized_ops=False).to(device)
    # CMM_PME returns (phi, E, EG, energy)
    py_phi, py_E, py_EG, py_energy = pme_py_obj(coords_ref, box_raw, q, p, t)
    
    # Calculate Energy for reference
    # E = 0.5 * sum(q*phi + p*E + trace(theta*EG))
    # We use the energy returned by your pme_py if available, 
    # otherwise we calculate it from potentials.
    
    # --- 4. CUDA CALCULATION (torchff-lib) ---
    print("--- Step 4: Running CUDA Custom Op PME ---")
    # Assuming your PME class handles the dispatch
    # Use the same rank and alpha
    from torchff.pme import PME as CUDAPME 
    pme_cu_op = CUDAPME(alpha=2.8, max_hkl=64, rank=2, use_customized_ops=True).to(device)
    
    # Forward Pass
    # Result: (phi, E, EG, energy, forces)
    cu_phi, cu_E, cu_EG, cu_energy, cu_forces = pme_cu_op(coords_raw, box_raw, q, p, t)

    # --- 5. PARITY ANALYSIS ---
    print("\n" + "="*50)
    print("FINAL PARITY RESULTS")
    print("="*50)
    
    phi_diff = torch.abs(py_phi - cu_phi).max().item()
    e_field_diff = torch.abs(py_E - cu_E).max().item()
    eg_diff = torch.abs(py_EG - cu_EG).max().item()
    
    print(f"Potential Max Diff:  {phi_diff:.2e}")
    print(f"Field Max Diff:      {e_field_diff:.2e}")
    print(f"Field Grad Max Diff: {eg_diff:.2e}")
    
    # Check Atom 0 specifically
    print("\n--- Atom 0 Comparison ---")
    print(f"PY  Potential: {py_phi[0].item():.8f}")
    print(f"CUDA Potential: {cu_phi[0].item():.8f}")

    print(f"PY ENERGY : {py_energy.item():.8f}")
    print(f"CUDA ENERGY: {cu_energy.item():.8f}")
    
    if phi_diff < 1e-7:
        print("\n>>> SUCCESS: Python and CUDA match on real water!")
    else:
        print("\n>>> FAIL: Numerical divergence detected.")
    print("="*50)

if __name__ == "__main__":
    run_unified_test()
