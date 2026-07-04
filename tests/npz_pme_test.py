import torch
import numpy as np
from torchff.pme import PME

def test_load_and_run(npz_path="water_debug_data.npz"):
    # --- 1. LOAD FROM NPZ ---
    print(f"Loading data from {npz_path}...")
    data = np.load(npz_path)
    
    device = 'cuda'
    dtype = torch.float64

    # Reconstruct tensors
    coords = torch.from_numpy(data['coords']).to(device).to(dtype)
    box = torch.from_numpy(data['box']).to(device).to(dtype)
    q = torch.from_numpy(data['q']).to(device).to(dtype)
    p = torch.from_numpy(data['p']).to(device).to(dtype)
    t = torch.from_numpy(data['t']).to(device).to(dtype)
    
    alpha = float(data['alpha'])
    K = int(data['K'])

    # --- 2. RUN TORCHFF-LIB CODE ---
    print(f"Running torchff-lib PME (alpha={alpha}, K={K})...")
    pme_obj = PME(alpha=alpha, max_hkl=K, rank=2, use_customized_ops=True).to(device)
    
    # Forward Pass
    phi, E, EG, energy, forces = pme_obj(coords, box, q, p, t)

    # --- 3. VERIFY ---
    print("\nStandalone Results:")
    print(f"Energy: {energy.item():.10f}")
    print(f"Atom 0 Potential: {phi[0].item():.10f}")
    
    return energy

if __name__ == "__main__":
    test_load_and_run()
