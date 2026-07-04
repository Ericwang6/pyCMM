import torch
import numpy as np
import os
import sys

# --- CONFIGURATION ---
PYCMM_PATH = "/pscratch/sd/a/asa/software/pyCMM"
FF_PATH = f"{PYCMM_PATH}/tests/data/water_refit.xml"
PDB_PATH = f"{PYCMM_PATH}/tests/data/water_216.pdb"
OUTPUT_NPZ = "water_debug_data.npz"

sys.path.insert(0, PYCMM_PATH)

import openmm.app as app
from cmm.ffxml import ForceFieldXML
from cmm.topology import Topology
from cmm.units import BOHR2NM
from cmm.pme_helper import PME as CMM_PME
from torchff.pme import PME as CUDAPME

torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_exporter():
    """Step 1: Get real data and REFERENCE results from pyCMM/Python."""
    print(f"--- Step 1: Exporting Reference Data ---")
    ff = ForceFieldXML(FF_PATH, device=device)
    pdb = app.PDBFile(PDB_PATH)
    top = Topology.fromOpenmm(pdb.topology, device)

    coords = torch.tensor(pdb.getPositions(asNumpy=True)._value / BOHR2NM, device=device)
    box = torch.tensor(np.array([[v.x, v.y, v.z] for v in pdb.topology.getPeriodicBoxVectors()]) / BOHR2NM, device=device)

    # Parametrize to get Real Multipoles
    sys_ref = ff.parametrize(top, use_fd_morse=True, use_polarization=False, cutoff_sr=9.0)
    _ = sys_ref.getEnergy(coords, box)
    mpoles = sys_ref.last_perm_multipoles.detach()

    q = mpoles[:, 0].contiguous()
    p = mpoles[:, 1:4].contiguous()
    t = mpoles[:, 4:10].contiguous()

    # Run Python Reference
    pme_py_obj = CMM_PME(alpha=2.8, max_hkl=64, rank=2, use_customized_ops=False).to(device)
    py_phi, py_E, py_EG, py_energy = pme_py_obj(coords, box, q, p, t)

    # Save everything including the Python results
    np.savez(OUTPUT_NPZ,
             coords=coords.cpu().numpy(),
             box=box.cpu().numpy(),
             q=q.cpu().numpy(),
             p=p.cpu().numpy(),
             t=t.cpu().numpy(),
             alpha=2.8,
             K=64,
             ref_energy=py_energy.item(),
             ref_phi=py_phi.cpu().numpy())
    print(f"Done! Saved input data and reference energy ({py_energy.item():.6f}) to {OUTPUT_NPZ}\n")

def run_loader_verification():
    """Step 2: Load the NPZ and verify the CUDA library matches the saved reference."""
    print(f"--- Step 2: Verification (Loading from {OUTPUT_NPZ}) ---")
    data = np.load(OUTPUT_NPZ)
    
    # Reconstruct Tensors
    coords = torch.from_numpy(data['coords']).to(device)
    box = torch.from_numpy(data['box']).to(device)
    q = torch.from_numpy(data['q']).to(device)
    p = torch.from_numpy(data['p']).to(device)
    t = torch.from_numpy(data['t']).to(device)
    
    ref_energy = float(data['ref_energy'])
    ref_phi = torch.from_numpy(data['ref_phi']).to(device)
    
    alpha = float(data['alpha'])
    K = int(data['K'])

    # Run CUDA library (The Standalone Test)
    pme_cu_op = CUDAPME(alpha=alpha, max_hkl=K, rank=2, use_customized_ops=True).to(device)
    cu_phi, cu_E, cu_EG, cu_energy, cu_forces = pme_cu_op(coords, box, q, p, t)

    # FINAL COMPARISON
    e_diff = abs(ref_energy - cu_energy.item())
    phi_diff = torch.abs(ref_phi - cu_phi).max().item()

    print("="*50)
    print(f"SAVED REF ENERGY: {ref_energy:.10f}")
    print(f"CUDA LOAD ENERGY: {cu_energy.item():.10f}")
    print(f"ENERGY DIFFERENCE: {e_diff:.2e}")
    print(f"POTENTIAL MAX DIFF: {phi_diff:.2e}")
    print("="*50)

    if e_diff < 1e-7:
        print(">>> SUCCESS: Standalone CUDA loader matches Python Reference!")
    else:
        print(">>> FAIL: Still seeing a difference in standalone mode.")

if __name__ == "__main__":
    run_exporter()
    run_loader_verification()
