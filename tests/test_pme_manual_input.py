import torch
import sys
import os
torch.set_default_dtype(torch.float64)

# Adjusting path for PME helper
sys.path.insert(0, "/pscratch/sd/a/asa/software/pyCMM")
from cmm.pme_helper import PME  # Changed from Ewald to PME

# -----------------------------
def make_trace_free(Q):
    """Make a symmetric, trace-free version of Q (N,3,3)."""
    # symmetrize
    Qs = 0.5 * (Q + Q.transpose(-1, -2))
    # remove trace
    tr = torch.einsum('nii->n', Qs)  # (N,)
    I = torch.eye(3, dtype=Qs.dtype, device=Qs.device).expand_as(Qs)
    Qtf = Qs - (tr.view(-1, 1, 1) / 3.0) * I
    return Qtf

def autograd_reference_forces(coords, box, q, p, t, alpha, K, rank, device="cuda"):
    """
    Returns (U_autograd, F_autograd) using the PME Python path.
    """
    coords_req = coords.detach().clone().requires_grad_(True)

    # Instantiate PME with the Python path
    pme_py = PME(alpha=alpha, max_hkl=K, rank=rank, use_customized_ops=False).to(device)

    # PME typically returns (phi, E, dE, energy) in Python mode
    out = pme_py(coords_req, box, q, p, t)
    
    phi, E, dE = out[:3]
    U = out[3] # PME usually computes total reciprocal + self energy directly

    print("PYCMM PHI             : ", phi[0])
    print("PYCMM FIELD           : ", E[0] if E is not None else "None")
    print("PYCMM FIELD GRADIENT  : ", dE[0] if dE is not None else "None")
    
    # Differentiate energy wrt coords
    coords_req.grad = None
    U.backward()
    F_ad = -coords_req.grad
    return U.detach(), F_ad.detach()

def compare_cuda_vs_autograd(coords, box, q, p, t, alpha, K, rank, device="cuda",
                             rtol=1e-10, atol=1e-10):
    # 1) Autograd reference (Python)
    U_ref, F_ref = autograd_reference_forces(coords, box, q, p, t, alpha, K, rank, device)

    # 2) CUDA path 
    pme_cu = PME(alpha=alpha, max_hkl=K, rank=rank, use_customized_ops=True).to(device)
    
    # CUDA returns (phi, E, dE, energy, forces)
    phi_cu, E_cu, EFG_cu, U_cu, F_cu = pme_cu(coords, box, q, p, t)
    
    print("CUDA PHI:            ", phi_cu[0])
    print("CUDA FIELD:          ", E_cu[0] if E_cu is not None else "None")
    print("CUDA FIELD GRADIENT: ", EFG_cu[0] if EFG_cu is not None else "None")
    
    # 3) Compare
    def mdiff(a, b): return (a - b).abs().max().item()
    print("PYCMM Energy: ", U_ref)
    print("CUDA  Energy: ", U_cu)
    
    print("\n--- FORCE COMPARISON ---")
    print("PYCMM Forces (Atom 0): ", F_ref[0])
    print("CUDA  Forces (Atom 0): ", F_cu[0])
    
    e_diff = (U_ref - U_cu).item()
    f_max_diff = mdiff(F_ref, F_cu)
    
    print(f"\nU allclose: {torch.allclose(U_ref, U_cu, rtol=rtol, atol=atol)} | Δ={e_diff:.2e}")
    print(f"F allclose: {torch.allclose(F_ref, F_cu, rtol=rtol, atol=atol)} | max|Δ|={f_max_diff:.2e}")
    
    return U_ref, F_ref, U_cu, F_cu


if __name__ == "__main__":
    device = "cuda"
    L = 10.0
    box = torch.tensor([[L, 0, 0], [0, L, 0], [0, 0, L]], device=device)
    
    # Coordinates for 3 atoms
    coords = torch.tensor([
        [1.3, 2.1, 3.7],
        [4.2, 5.8, 1.1],
        [8.6, 7.5, 6.4]
    ], device=device)
    
    # Monopoles, Dipoles, and Quadrupoles
    q = torch.tensor([0.3, -0.5, 0.2], device=device)
    p = torch.tensor([
        [0.10, -0.20, 0.05],
        [-0.03, 0.06, -0.02],
        [0.07, 0.01, 0.04]
    ], device=device)
    
    raw = torch.tensor([
        [[0.20, 0.03, -0.02], [0.03, -0.05, 0.01], [-0.02, 0.01, 0.00]],
        [[-0.10, 0.02, 0.01], [0.02, 0.06, -0.03], [0.01, -0.03, 0.04]],
        [[0.08, -0.01, 0.00], [-0.01, 0.02, 0.02], [0.00, 0.02, -0.06]],
    ], device=device)
    
    # Quadrupoles in 6-element format [xx, xy, xz, yy, yz, zz] for PME kernels
    t_tf = make_trace_free(raw)
    t = torch.stack([
        t_tf[:, 0, 0], t_tf[:, 0, 1], t_tf[:, 0, 2],
        t_tf[:, 1, 1], t_tf[:, 1, 2], t_tf[:, 2, 2]
    ], dim=1)

    # Run the test
    compare_cuda_vs_autograd(coords, box, q, p, t, alpha=0.35, K=64, rank=2, device=device)
