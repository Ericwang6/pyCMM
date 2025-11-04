import torch
import torch.nn as nn
from typing import Optional

try:
    import torchff
    import torchff_ewald
except Exception as e:
    pass


class Ewald(nn.Module):
    def __init__(self, alpha: float, max_hkl: int, rank: int, use_customized_ops: bool = False):
        super().__init__()
        sym_factors = []
        all_hkl = []
        for h in range(0, max_hkl+1):
            for k in range(-max_hkl, max_hkl+1):
                for l in range(-max_hkl, max_hkl+1):
                    if h == 0 and k == 0 and l == 0:
                        continue
                    all_hkl.append([float(h), float(k), float(l)])
                    sym_factors.append(2.0 if h > 0 else 1.0)
                        
        self.register_buffer('sym_factors', torch.tensor(sym_factors).unsqueeze(1))
        self.register_buffer('all_hkl', torch.tensor(all_hkl))
        self.max_hkl = max_hkl
        self.alpha = alpha
        self.alpha2 = alpha * alpha
        self.alpha_over_root_pi = self.alpha / torch.sqrt(torch.tensor(torch.pi))
        self.rank = rank
        self.use_customized_ops = use_customized_ops
    
    def forward(self, coords: torch.Tensor, box: torch.Tensor, q: torch.Tensor, p: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None):
        if self.use_customized_ops:
            return self._forward_cpp(coords, box, q, p, t)
        else:
            return self._forward_python(coords, box, q, p, t)
    
    def _forward_cpp(self, coords, box, q, p, t):
        dev = coords.device
        dtype = torch.float64
        coords = coords.to(device=dev, dtype=dtype).contiguous()
        box    = box.to(device=dev, dtype=dtype).contiguous()
        q      = q.to(device=dev, dtype=dtype).contiguous()
        N = q.shape[0]
            # Provide valid tensors even if None
        if p is None:
            p = torch.zeros((N, 3), device=dev, dtype=dtype)
        else:
            p = p.to(device=dev, dtype=dtype).contiguous()

        if t is None:
            t = torch.zeros((N, 3, 3), device=dev, dtype=dtype)
        else:
            t = t.to(device=dev, dtype=dtype).contiguous()
        pot, fld, grad, energy, forces = torch.ops.torchff.ewald_long_range(
            coords, box, q, p, t, self.max_hkl, self.rank, float(self.alpha)
        )
        # res = {potential, field, field_grad, energy, forces}
        # Return structure identical to _forward_python
        return pot, fld, grad, energy, forces
    def _forward_python(
        self,
        coords: torch.Tensor,
        box: torch.Tensor,
        q: torch.Tensor,
        p: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        ):
        # ---------------- DEBUG TOGGLES ----------------
        DBG = False          # turn prints on/off
        DBG_KMAX = 6        # print first K entries
        DBG_NIDX = 1        # which atom n to trace in per-k prints
        # ------------------------------------------------

        device = coords.device
        dtype = coords.dtype

        box_inv = torch.inverse(box)
        V = torch.det(box)

        # Convert h,k,l indices to reciprocal space vectors: (M1,3)
        kvectors = torch.matmul(self.all_hkl, box_inv)        # (M1,3)
        M1 = kvectors.shape[0]
        N  = coords.shape[0]

        # Apply spherical cutoff based on k^2
        k_squared = torch.einsum('ij,ij->i', kvectors, kvectors)  # (M1,)
        gaussian_factors = torch.exp(-torch.pi * torch.pi * k_squared / self.alpha2) / k_squared  # (M1,)

        # Calculating phases
        k_dot_r = torch.matmul(kvectors, coords.T)                 # (M1,N)
        cos_k_dot_r = torch.cos(2 * torch.pi * k_dot_r)            # (M1,N)
        sin_k_dot_r = torch.sin(2 * torch.pi * k_dot_r)            # (M1,N)

        # exact rank-0 S(k) using your cos/sin arrays
        Sr0 = torch.einsum('kn,n->k', cos_k_dot_r, q)
        Si0 = torch.einsum('kn,n->k', sin_k_dot_r, q)
        print("PY rank-0 S[0..5] explicit:", list(zip(Sr0[:6].tolist(), Si0[:6].tolist())))


        # Structure-factor ingredients
        if self.rank == 2:
            F_l_real = q.expand(kvectors.size(0), -1) - torch.einsum(
                'kj,nij,ki->kn', kvectors, t, kvectors
            ) * (2 * torch.pi) * (2 * torch.pi) / 3
            F_l_imag = torch.matmul(kvectors, p.T) * 2 * torch.pi
        elif self.rank == 1:
            F_l_real = q.expand(kvectors.size(0), -1)
            F_l_imag = torch.matmul(kvectors, p.T) * 2 * torch.pi
        else:
            F_l_real = q.expand(kvectors.size(0), -1)
            F_l_imag = torch.zeros(kvectors.size(0), q.size(0), device=q.device, dtype=q.dtype)

        exp_k_dot_r = torch.complex(cos_k_dot_r, sin_k_dot_r)      # (M1,N), e^{+iθ}
        exp_minus_k_dot_r = torch.complex(cos_k_dot_r, -sin_k_dot_r)  # (M1,N), e^{-iθ}

        F_2 = torch.complex(F_l_real, F_l_imag)   # (M1,N), (F_r + i F_i)
        structure_factors = torch.sum(F_2 * exp_k_dot_r, dim=1)    # (M1,), S(k)

        # ---------------- DEBUG PRINTS (match CUDA style) ----------------
        if DBG:
            # Basic scalars
            print(f"[PY] rank = {self.rank}")
            print(f"[PY] N={N}  M1={M1}")
            inv_piV = float(1.0 / (torch.pi * V))
            m2_over_V = float(-2.0 / V)
            print(f"[PY] Volume V={float(V)}  1/(πV)={inv_piV}  -2/V={m2_over_V}")

            # First few k rows: kvec, g, sym
            sym = self.sym_factors.squeeze(1)  # (M1,)
            print(f"[PY] sum(sym) = {float(sym.sum())}")
            Kshow = min(DBG_KMAX, M1)
            for k in range(Kshow):
                kx, ky, kz = kvectors[k, 0].item(), kvectors[k, 1].item(), kvectors[k, 2].item()
                gk  = gaussian_factors[k].item()
                sk  = float(sym[k])
                print(f"[PY] k[{k}] = ({kx},{ky},{kz})  g={gk}  sym={sk}")

            # First few cos/sin for atom n = DBG_NIDX
            n = min(DBG_NIDX, N-1)
            print(f"--- PY cos/sin for n={n} (first {Kshow} k) ---")
            for k in range(Kshow):
                c = cos_k_dot_r[k, n].item()
                s = sin_k_dot_r[k, n].item()
                print(f"k={k}  cos={c}  sin={s}")
            print("PY shapes:", cos_k_dot_r.shape, sin_k_dot_r.shape, q.shape)
            print("PY strides:", cos_k_dot_r.stride(), sin_k_dot_r.stride())
            # Force clean contiguous layout in case they are transposed views
            coskr, sinkr = cos_k_dot_r, sin_k_dot_r
            coskr = coskr.contiguous()
            sinkr = sinkr.contiguous()

            # Rebuild S directly from phases and q
            Sr = torch.einsum('kn,n->k', coskr, q)      # sum_n q[n]*cos
            Si = torch.einsum('kn,n->k', sinkr, q)      # sum_n q[n]*sin

            print("PY recomputed S[0..5]:", [(Sr[k].item(), Si[k].item()) for k in range(6)])

            # Also print per-atom contributions for k=0 to line up with CUDA's [SF.DBG]
            k = 0
            for n in range(N):
                re_add = (q[n]*coskr[k,n]).item()
                im_add = (q[n]*sinkr[k,n]).item()
                print(f"PY k=0 n={n}  c={coskr[k,n].item():+.12f}  s={sinkr[k,n].item():+.12f} "
                  f"q={q[n].item():+.12f}  re_add={re_add:+.12f}  im_add={im_add:+.12f}")


            # First few S(k)
            Sr = structure_factors.real
            Si = structure_factors.imag
            print(f"--- PY S(k) (first {Kshow}) ---")
            for k in range(Kshow):
                print(f"[PY] S[{k}] = ({float(Sr[k])}, {float(Si[k])})")

            # Per-k breakdown for atom n (matches CUDA ACC prints)
            c = cos_k_dot_r[:, n]
            s = sin_k_dot_r[:, n]
            re = Sr * c + Si * s
            im = Si * c - Sr * s
            kx = kvectors[:, 0]
            ky = kvectors[:, 1]
            kz = kvectors[:, 2]

            print(f"--- PY per-k (first {Kshow}) for n={n} ---")
            for k in range(Kshow):
                print(
                    f"k={k:2d} Sr={float(Sr[k]): .8e} Si={float(Si[k]): .8e} "
                    f"c={float(c[k]): .8e} s={float(s[k]): .8e} "
                    f"re={float(re[k]): .8e} im={float(im[k]): .8e} "
                    f"g={float(gaussian_factors[k]): .8e} sym={int(sym[k])} "
                    f"kx={float(kx[k]): .8e} ky={float(ky[k]): .8e} kz={float(kz[k]): .8e}"
                )

            # Pre-prefactor totals (these should match CUDA [ACC] totals)
            phi_k = gaussian_factors * sym * re
            Fx_k  = gaussian_factors * sym * im * kx
            Fy_k  = gaussian_factors * sym * im * ky
            Fz_k  = gaussian_factors * sym * im * kz
            print("--- PY totals (pre-prefactor) ---")
            print("pot_sum =", float(phi_k.sum()))
            print("Fx,Fy,Fz =", float(Fx_k.sum()), float(Fy_k.sum()), float(Fz_k.sum()))
        # -----------------------------------------------------------------

        # Apply symmetry factors to each k-vector contribution and sum over k
        # phi_expanded: (M1,N) complex
        phi_expanded = (
            gaussian_factors.unsqueeze(1) * structure_factors.unsqueeze(1) * self.sym_factors
        ) * exp_minus_k_dot_r

        potential = torch.sum(phi_expanded.real, dim=0) / (torch.pi * V)  # (N,)
        potential = potential - 2 * self.alpha_over_root_pi * q           # self term

        if DBG:
            # Final outputs (like CUDA tail)
            if potential.numel() >= 3:
                print(
                    "PY potential[0..2]:",
                    float(potential[0]), float(potential[1]), float(potential[2])
                )

        if self.rank == 0:
            return potential

        # Field
        field = 2 * (
            torch.matmul(phi_expanded.T, torch.complex(torch.zeros_like(kvectors), kvectors)).real
        ) / V
        field = field + self.alpha_over_root_pi * (4 * self.alpha2 / 3) * p

        if DBG and self.rank >= 1 and N >= 2:
            print(
                "PY field[1]: (",
                float(field[1,0]), ", ",
                float(field[1,1]), ", ",
                float(field[1,2]), ")",
                sep=""
            )

        if self.rank == 1:
            return potential, field

        # Field gradient (rank 2 path)
        k_outer = torch.einsum('bi,bj->bij', kvectors, kvectors).reshape(-1, 9)  # (M1,9)
        field_grad = 4 * torch.pi * (
            torch.matmul(phi_expanded.T, torch.complex(k_outer, torch.zeros_like(k_outer))).real.reshape(-1, 3, 3)
        ) / V
        field_grad = field_grad + self.alpha_over_root_pi * (16 * self.alpha2 * self.alpha2 / 5) * t / 3

        if self.rank == 2:
            return potential, field, field_grad
 
#    def _forward_python(self, coords: torch.Tensor, box: torch.Tensor, q: torch.Tensor, p: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None):
#        box_inv = torch.inverse(box)
#        V = torch.det(box)
#
#        # Convert h,k,l indices to reciprocal space vectors
#        kvectors = torch.matmul(self.all_hkl, box_inv)
#    
#        # Apply spherical cutoff based on k^2
#        k_squared = torch.einsum('ij,ij->i', kvectors, kvectors)
#        gaussian_factors = torch.exp(-torch.pi * torch.pi * k_squared / self.alpha2) / k_squared
#
#        # Calculating all structure factors
#        k_dot_r = torch.matmul(kvectors, coords.T)
#        cos_k_dot_r = torch.cos(2 * torch.pi * k_dot_r)
#        sin_k_dot_r = torch.sin(2 * torch.pi * k_dot_r)
#
#        if self.rank == 2:
#            F_l_real = q.expand(kvectors.size(0), -1) - torch.einsum('kj,nij,ki->kn', kvectors, t, kvectors) * (2 * torch.pi) * (2 * torch.pi) / 3
#            F_l_imag = torch.matmul(kvectors, p.T) * 2 * torch.pi
#        elif self.rank == 1:
#            F_l_real = q.expand(kvectors.size(0), -1)
#            F_l_imag = torch.matmul(kvectors, p.T) * 2 * torch.pi
#        else:
#            F_l_real = q.expand(kvectors.size(0), -1)
#            F_l_imag = torch.zeros(kvectors.size(0), q.size(0), device=q.device, dtype=q.dtype)
#        
#        exp_k_dot_r = torch.complex(cos_k_dot_r, sin_k_dot_r)
#        exp_minus_k_dot_r = torch.complex(cos_k_dot_r, -sin_k_dot_r)
#        F_2 = torch.complex(F_l_real, F_l_imag)
#        structure_factors = torch.sum(F_2 * exp_k_dot_r, dim=1)
#        ##DEBUGGING###
#        n = 1
#        S  = structure_factors
#        Sr = S.real; Si = S.imag
#        c  = cos_k_dot_r[:, n]; s = sin_k_dot_r[:, n]
#        re = Sr * c + Si * s
#        im = Si * c - Sr * s
#        g  = gaussian_factors
#        sym = self.sym_factors.squeeze(1)
#        kx, ky, kz = kvectors[:,0], kvectors[:,1], kvectors[:,2]
#
#        print("--- PY per-k (first 3) for n=1 ---")
#        for k in range(3):
#            print(f"k={k:2d} Sr={Sr[k]: .8e} Si={Si[k]: .8e} c={c[k]: .8e} s={s[k]: .8e} "
#                  f"re={re[k]: .8e} im={im[k]: .8e} g={g[k]: .8e} sym={sym[k]:.0f} "
#                  f"kx={kx[k]: .8e} ky={ky[k]: .8e} kz={kz[k]: .8e}")
#
#        phi_k = g * sym * re
#        Fx_k  = g * sym * im * kx
#        Fy_k  = g * sym * im * ky
#        Fz_k  = g * sym * im * kz
#        print("--- PY totals (pre-prefactor) for n=1 ---")
#        print("pot_sum =", float(phi_k.sum()))
#        print("Fx,Fy,Fz =", float(Fx_k.sum()), float(Fy_k.sum()), float(Fz_k.sum()))
#        # Apply symmetry factors to each k-vector contribution
#        phi_expanded = (gaussian_factors.unsqueeze(1) * structure_factors.unsqueeze(1) * self.sym_factors) * exp_minus_k_dot_r
#        
#        potential = torch.sum(phi_expanded.real, dim=0) / (torch.pi * V)
#        potential = potential - 2 * self.alpha_over_root_pi * q  # self contributions
#        if self.rank == 0:
#            return potential
#        
#        field = 2 * (
#            torch.matmul(phi_expanded.T, torch.complex(torch.zeros_like(kvectors), kvectors)).real
#        ) / V
#        field = field + self.alpha_over_root_pi * (4 * self.alpha2 / 3) * p
#
#        if self.rank == 1:
#            return potential, field
#        
#        k_outer = torch.einsum('bi,bj->bij', kvectors, kvectors).reshape(-1, 9)
#        field_grad = 4 * torch.pi * (
#            torch.matmul(phi_expanded.T, torch.complex(k_outer, torch.zeros_like(k_outer))).real.reshape(-1, 3, 3)
#        ) / V
#        field_grad = field_grad + self.alpha_over_root_pi * (16 * self.alpha2 * self.alpha2 / 5) * t / 3
#
#        if self.rank == 2:
#            return potential, field, field_grad
