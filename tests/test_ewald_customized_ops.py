import torch
import sys
import os
torch.set_default_dtype(torch.float64)

sys.path.insert(0, "/pscratch/sd/a/asa/software/pyCMM/cmm")
from ewald import Ewald

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
    Returns (U_autograd, F_autograd) using the pure Python path (no custom CUDA),
    by differentiating the exact energy expression wrt coords.
    """
    coords_req = coords.detach().clone().requires_grad_(True)

    # Pure Python/diff path
    ewald_py = Ewald(alpha=alpha, max_hkl=K, rank=rank, use_customized_ops=False).to(device)

    # Unpack by rank
    if rank == 0:
        phi = ewald_py(coords_req, box, q, torch.zeros_like(p), torch.zeros_like(t))
        E = dE = None
        U = 0.5 * (q @ phi)

    elif rank == 1:
        phi, E = ewald_py(coords_req, box, q, p, torch.zeros_like(t))
        dE = None
        print()
        U = 0.5 * (q @ phi) - 0.5 * torch.einsum('ni,ni->', p, E)

    else:  # rank >= 2
        out = ewald_py(coords_req, box, q, p, t)
        phi, E, dE = out[:3]
        U = 0.5 * (q @ phi) - 0.5 * torch.einsum('ni,ni->', p, E) - (1.0/6.0) * torch.einsum('nij,nij->', t, dE)

    print("PYCMM PHI            : ", phi)
    print("PYCMM FIELD          : ", E)
    print("PYCMM FIELD GRADIENT : ", dE)
    # Differentiate energy wrt coords
    coords_req.grad = None
    U.backward()
    F_ad = -coords_req.grad
    return U.detach(), F_ad.detach()

def compare_cuda_vs_autograd(coords, box, q, p, t, alpha, K, rank, device="cuda",
                             rtol=1e-10, atol=1e-10):
    # 1) Autograd reference
    U_ref, F_ref = autograd_reference_forces(coords, box, q, p, t, alpha, K, rank, device)

    # 2) CUDA path 
    ewald_cu = Ewald(alpha=alpha, max_hkl=K, rank=rank, use_customized_ops=True).to(device)
    out_cu = ewald_cu(coords, box, q, p, t)  # returns (phi, E, dE, energy, forces)
    # handle rank-specific tuple length; last two are energy, forces
    phi_cu, E_cu, EFG_cu, U_cu, F_cu = out_cu
    print("CUDA PHI:            ", phi_cu)
    print("CUDA FIELD:          ", E_cu)
    print("CUDA FIELD GRADIENT: ", EFG_cu)
    # 3) Compare
    def mdiff(a, b): return (a - b).abs().max().item()
    print("PYCMM Forces: ",  F_ref)
    print("CUDA  Forces: ",  F_cu)
    print(f"U   allclose: {torch.allclose(U_ref, U_cu, rtol=rtol, atol=atol)}  Δ={float((U_ref - U_cu).item())}")
    print(f"F   allclose: {torch.allclose(F_ref, F_cu, rtol=rtol, atol=atol)}  max|Δ|={mdiff(F_ref, F_cu)}  rms={(torch.mean((F_ref-F_cu)**2).sqrt().item())}")
    return U_ref, F_ref, U_cu, F_cu


if __name__ == "__main__":
    # tiny test
    device="cuda"
    L=10.0
    box = torch.tensor([[L,0,0],[0,L,0],[0,0,L]], device=device)
    coords = torch.tensor([[1.3,2.1,3.7],[4.2,5.8,1.1],[8.6,7.5,6.4]], device=device)
    q = torch.tensor([0.3,-0.5,0.2], device=device)
    p = torch.tensor([[ 0.10, -0.20,  0.05],                      [-0.03,  0.06, -0.02],    [ 0.07,  0.01,  0.04]], device=device)
    raw = torch.tensor([        [[ 0.20,  0.03, -0.02], [ 0.03, -0.05,  0.01], [-0.02,  0.01,  0.00]], [[-0.10,  0.02,  0.01], [ 0.02,  0.06, -0.03], [ 0.01, -0.03,  0.04]], [[ 0.08, -0.01,  0.00], [-0.01,  0.02,  0.02], [ 0.00,  0.02, -0.06]], ], device=device)
    #q = torch.zeros(3, device=device)
    #p = torch.zeros((3,3), device=device)         # start with monopoles only
    #raw = torch.zeros((3,3,3), device=device)
    t = make_trace_free(raw.clone())

    compare_cuda_vs_autograd(coords, box, q, p, t, alpha=0.35, K=4, rank=2, device=device)



