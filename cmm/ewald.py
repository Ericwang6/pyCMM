import torch
import math

def long_range(coords, q, p, t, box, alpha, kcutoff):
    V = torch.det(box) #compute volume of box
    kx = (2*math.pi)/ V * torch.linalg.cross(box[1],box[2])
    ky = (2*math.pi)/ V * torch.linalg.cross(box[2],box[0])
    kz = (2*math.pi)/ V * torch.linalg.cross(box[0],box[1])
    k_sq_max = kcutoff**2
    kvectors = []
    max_hkl = 29
    hkl_range = torch.arange(-max_hkl, max_hkl + 1)
    print(max_hkl)
    #print(f"K_SQ_MAX: {k_sq_max}, KCUTOFF: {kcutoff}, alpha: {alpha}, V: {V}, kx: {kx}, ky: {ky}, kz: {kz}")
    for h in hkl_range:
        for k in hkl_range:
            for l in hkl_range:
                if h == 0 and k == 0 and l == 0:
                    continue
                kvec = h*kx + k*ky +l*kz
                if torch.norm(kvec) < k_sq_max and torch.norm(kvec)>0:
                    kvectors.append(kvec)
    kvectors = torch.stack(kvectors)
    #Precalculating gaussian factors
    k_squared = torch.sum(kvectors ** 2, dim=1)
    gaussian_factor = torch.exp(-k_squared /(4*alpha**2)) / k_squared
    #Calculating all structure factors
    sk_cos = torch.zeros(len(kvectors))
    sk_sin = torch.zeros(len(kvectors))
    sk_dipole_cos = torch.zeros(len(kvectors))
    sk_dipole_sin = torch.zeros(len(kvectors))
    sk_quad_cos = torch.zeros(len(kvectors))
    sk_quad_sin = torch.zeros(len(kvectors))
    for i, kvec in enumerate(kvectors):
        k_dot_r = torch.sum(kvec * coords, dim=1)
        #k_dot_p = torch.sum(p * kvec, dim=1)
        #k_outer = torch.outer(kvec, kvec)         # h_a h_b
        #h_contract_theta = torch.sum(k_outer * t, dim = (1,2))  # h_a h_b theta_iab
        sk_cos[i] = torch.sum(q * torch.cos(k_dot_r))
        sk_sin[i] = torch.sum(q * torch.sin(k_dot_r))
        #sk_dipole_cos[i] = torch.sum(k_dot_p * torch.cos(k_dot_r))
        #sk_dipole_sin[i] = torch.sum(k_dot_p * torch.sin(k_dot_r))
        #sk_quad_cos[i] = torch.sum(h_contract_theta * torch.cos(k_dot_r))
        #sk_quad_sin[i] = torch.sum(h_contract_theta * torch.sin(k_dot_r))
    ###Charge–charge### 
    cc_ene = (4*math.pi/ V)* torch.sum(gaussian_factor * (sk_cos**2 + sk_sin**2))
    ###Charge–dipole### 
    #cd_ene = (8 * math.pi / V) * torch.sum(gaussian_factor* (sk_sin * sk_dipole_cos - sk_cos * sk_dipole_sin))
    ####Dipole-dipole### 
    #dd_ene = (4 * math.pi / V) *torch.sum(gaussian_factor * (sk_dipole_cos**2 + sk_dipole_sin**2))
    ####charge-quadrupole###
    #ct_ene = -(8*math.pi/V) * torch.sum(gaussian_factor * (sk_cos *sk_quad_cos + sk_sin * sk_quad_sin)/3)
    ####dipole-quadrupole###
    #dt_ene = (8*math.pi/V) * torch.sum(gaussian_factor*(sk_dipole_sin * sk_quad_cos - sk_dipole_cos * sk_quad_sin)/3)
    ####quadrupole-quadrupole###
    #tt_ene = (4*math.pi/V) * torch.sum(gaussian_factor*(sk_quad_cos**2 + sk_quad_sin**2)/9)
    #####Final reciprocal energy#####
    U_l = cc_ene #+ cd_ene + dd_ene + ct_ene + dt_ene + tt_ene
    #print("KVECTORS: \n",kvectors)
    #print("k_squared values: ", k_squared)
    #print("GAUSSIAN FACTOR: \n", gaussian_factor)
    #print("CHARGE CHARGE ENERGY: ", cc_ene)
    #print("CHARGE DIPOLE ENERGY: ", cd_ene)
    #print("DIPOLE DIPOLE ENERGY: ", dd_ene)
    #print("CHARGE QUADRUPOLE ENERGY: ", ct_ene)
    #print("DIPOLE QUADRUPOLE ENERGY: ", dt_ene)
    #print("QUADRUPOLE QUADRUPOLE ENERGY: ", tt_ene)
    return U_l

def long_range_vectorized(coords: torch.Tensor, q: torch.Tensor, p: torch.Tensor, t: torch.Tensor, box: torch.Tensor, alpha: torch.Tensor, max_hkl: torch.NumberType):

    # Reciporcal lattice vectors #
    V = torch.det(box) # volume of box
    reciprocal_box = torch.stack((
        torch.linalg.cross(box[1], box[2]),
        torch.linalg.cross(box[2], box[0]),
        torch.linalg.cross(box[0], box[1])
    )) / V

    hkl_range = torch.arange(-max_hkl, max_hkl + 1)
    all_hkl = torch.cartesian_prod(hkl_range, hkl_range, hkl_range).to(torch.float64)
    all_hkl = all_hkl[torch.norm(all_hkl, dim=1) != 0.0]
    kvectors = torch.matmul(all_hkl, reciprocal_box) # Can also be expressed with torch.bmm if faster
    
    # Precalculating gaussian factors
    k_squared = torch.einsum('ij,ij->i', kvectors, kvectors) # @SPEED: This is a row-wise dot product and can likely be done more efficiently.
    
    gaussian_factors = torch.exp(-torch.pi * torch.pi * k_squared / (alpha * alpha)) / k_squared

    # Calculating all structure factors
    k_dot_r = torch.matmul(kvectors, coords.T)
    q_cos_k_dot_r = torch.matmul(torch.cos(2 * torch.pi * k_dot_r), q)
    q_sin_k_dot_r = torch.matmul(torch.sin(2 * torch.pi * k_dot_r), q)
    structure_factor_cos = torch.square(q_cos_k_dot_r)
    structure_factor_sin = torch.square(q_sin_k_dot_r)

    ###Charge–charge###
    cc_ene = (1 / (2 * torch.pi * V)) * torch.dot(gaussian_factors, (structure_factor_cos + structure_factor_sin))
    ###Charge–dipole### 
    #cd_ene = (8 * math.pi / V) * torch.sum(gaussian_factor* (sk_sin * sk_dipole_cos - sk_cos * sk_dipole_sin))
    ####Dipole-dipole### 
    #dd_ene = (4 * math.pi / V) *torch.sum(gaussian_factor * (sk_dipole_cos**2 + sk_dipole_sin**2))
    ####charge-quadrupole###
    #ct_ene = -(8*math.pi/V) * torch.sum(gaussian_factor * (sk_cos *sk_quad_cos + sk_sin * sk_quad_sin)/3)
    ####dipole-quadrupole###
    #dt_ene = (8*math.pi/V) * torch.sum(gaussian_factor*(sk_dipole_sin * sk_quad_cos - sk_dipole_cos * sk_quad_sin)/3)
    ####quadrupole-quadrupole###
    #tt_ene = (4*math.pi/V) * torch.sum(gaussian_factor*(sk_quad_cos**2 + sk_quad_sin**2)/9)
    #####Final reciprocal energy#####
    U_l = cc_ene #+ cd_ene + dd_ene + ct_ene + dt_ene + tt_ene
    #print("KVECTORS: \n",kvectors)
    #print("k_squared values: ", k_squared)
    #print("GAUSSIAN FACTOR: \n", gaussian_factor)
    #print("CHARGE CHARGE ENERGY: ", cc_ene)
    #print("CHARGE DIPOLE ENERGY: ", cd_ene)
    #print("DIPOLE DIPOLE ENERGY: ", dd_ene)
    #print("CHARGE QUADRUPOLE ENERGY: ", ct_ene)
    #print("DIPOLE QUADRUPOLE ENERGY: ", dt_ene)
    #print("QUADRUPOLE QUADRUPOLE ENERGY: ", tt_ene)
    return U_l

def long_range_potential(coords: torch.Tensor, q: torch.Tensor, p: torch.Tensor, t: torch.Tensor, box: torch.Tensor, alpha: torch.Tensor, max_hkl: torch.NumberType):

    # Reciporcal lattice vectors #
    V = torch.det(box) # volume of box
    reciprocal_box = torch.stack((
        torch.linalg.cross(box[1], box[2]),
        torch.linalg.cross(box[2], box[0]),
        torch.linalg.cross(box[0], box[1])
    )) / V

    # @SPEED: Two optimizations here.
    # 1) only do calculation over positive k vectors and just multiply by 2 (or whatever the factor is)
    # 2) Take this rectangular range and make it spherical by making the cutoff on k^2 not k_max.
    hkl_range = torch.arange(-max_hkl, max_hkl + 1)
    all_hkl = torch.cartesian_prod(hkl_range, hkl_range, hkl_range).to(torch.float64)
    all_hkl = all_hkl[torch.norm(all_hkl, dim=1) != 0.0]
    kvectors = torch.matmul(all_hkl, reciprocal_box)
    
    # Precalculating gaussian factors
    k_squared = torch.einsum('ij,ij->i', kvectors, kvectors)
    gaussian_factors = torch.exp(-torch.pi * torch.pi * k_squared / (alpha * alpha)) / k_squared

    # Calculating all structure factors
    k_dot_r = torch.matmul(kvectors, coords.T)
    cos_k_dot_r = torch.cos(2 * torch.pi * k_dot_r)
    sin_k_dot_r = torch.sin(2 * torch.pi * k_dot_r)

    # NOTE(JOE): I am confident that the potential, field, and field gradient expressions here are correct for charges.
    # So, the only unchecked source of errors would be in the fourier transform of the multipoles below.
    # If we run into problems with multipoles, the below two lines are the first place to look.
    # See: A coherent derivation of the Ewald summation for arbitrary orders of multipoles
    # for relevant expressions.
    F_l_real = q.expand(kvectors.size(0), -1) - torch.einsum('kj,nij,ki->kn', kvectors, t, kvectors) * (2 * torch.pi) * (2 * torch.pi)
    F_l_imag = torch.matmul(kvectors, p.T) * 2 * torch.pi
    
    exp_k_dot_r = torch.complex(cos_k_dot_r, sin_k_dot_r)
    exp_minus_k_dot_r = torch.complex(cos_k_dot_r, -sin_k_dot_r)
    F_2 = torch.complex(F_l_real, F_l_imag)
    structure_factors = torch.sum(F_2 * exp_k_dot_r, dim=1)
    phi_expanded = (gaussian_factors * structure_factors).unsqueeze(1) * exp_minus_k_dot_r
    
    potential = torch.sum(phi_expanded.real, dim=0) / (torch.pi * V) # can take .real inside sum since .imag sums to zero.
    field = 2 * (
        torch.matmul(phi_expanded.T, torch.complex(torch.zeros_like(kvectors), kvectors)).real
    ) / V
    k_outer = torch.vmap(torch.outer)(kvectors, kvectors).reshape(-1, 9)
    field_grad = 4 * torch.pi  * (
        torch.matmul(phi_expanded.T, torch.complex(k_outer, torch.zeros_like(k_outer))).real.reshape(-1, 3, 3)
    ) / V

    # Now add in the self contributions to potential, field, and field gradient #
    alpha_over_root_pi = alpha / torch.sqrt(torch.tensor(torch.pi))
    potential = potential - 2 * alpha_over_root_pi * q
    field = field + alpha_over_root_pi * (4 * alpha * alpha / 3) * p
    field_grad = field_grad + alpha_over_root_pi * (16 * alpha * alpha * alpha * alpha / 5) * t / 3

    return potential, field, field_grad

def long_range_potential_rank_1(coords: torch.Tensor, q: torch.Tensor, p: torch.Tensor, box: torch.Tensor, alpha: torch.Tensor, max_hkl: torch.NumberType):

    # Reciporcal lattice vectors #
    V = torch.det(box) # volume of box
    reciprocal_box = torch.stack((
        torch.linalg.cross(box[1], box[2]),
        torch.linalg.cross(box[2], box[0]),
        torch.linalg.cross(box[0], box[1])
    )) / V

    # @SPEED: Two optimizations here.
    # 1) only do calculation over positive k vectors and just multiply by 2 (or whatever the factor is)
    # 2) Take this rectangular range and make it spherical by making the cutoff on k^2 not k_max.
    hkl_range = torch.arange(-max_hkl, max_hkl + 1)
    all_hkl = torch.cartesian_prod(hkl_range, hkl_range, hkl_range).to(torch.float64)
    all_hkl = all_hkl[torch.norm(all_hkl, dim=1) != 0.0]
    kvectors = torch.matmul(all_hkl, reciprocal_box)
    
    # Precalculating gaussian factors
    k_squared = torch.einsum('ij,ij->i', kvectors, kvectors)
    gaussian_factors = torch.exp(-torch.pi * torch.pi * k_squared / (alpha * alpha)) / k_squared

    # Calculating all structure factors
    k_dot_r = torch.matmul(kvectors, coords.T)
    cos_k_dot_r = torch.cos(2 * torch.pi * k_dot_r)
    sin_k_dot_r = torch.sin(2 * torch.pi * k_dot_r)

    # NOTE(JOE): I am confident that the potential, field, and field gradient expressions here are correct for charges.
    # So, the only unchecked source of errors would be in the fourier transform of the multipoles below.
    # If we run into problems with multipoles, the below two lines are the first place to look.
    # See: A coherent derivation of the Ewald summation for arbitrary orders of multipoles
    # for relevant expressions.
    F_l_real = q.expand(kvectors.size(0), -1)
    F_l_imag = torch.matmul(kvectors, p.T) * 2 * torch.pi
    
    exp_k_dot_r = torch.complex(cos_k_dot_r, sin_k_dot_r)
    exp_minus_k_dot_r = torch.complex(cos_k_dot_r, -sin_k_dot_r)
    F_2 = torch.complex(F_l_real, F_l_imag)
    structure_factors = torch.sum(F_2 * exp_k_dot_r, dim=1)
    phi_expanded = (gaussian_factors * structure_factors).unsqueeze(1) * exp_minus_k_dot_r
    potential = torch.sum(phi_expanded.real, dim=0) / (torch.pi * V) # can take .real inside sum since .imag sums to zero.
    field = 2 * (
        torch.matmul(phi_expanded.T, torch.complex(torch.zeros_like(kvectors), kvectors)).real
    ) / V

    # Now add in the self contributions to potential, field, and field gradient #
    alpha_over_root_pi = alpha / torch.sqrt(torch.tensor(torch.pi))
    potential = potential - 2 * alpha_over_root_pi * q
    field = field + alpha_over_root_pi * (4 * alpha * alpha / 3) * p

    return potential, field

def self_interaction(coords, q , p, t, alpha):
    #Self interaction energy. Subtracted from total Ewald energy.
    #U_q, U_d, U_cq, U_t = 0, 0, 0, 0
    #N = len(coords)
    #for i in range(N):
    #    #monopole
    #    U_q += q[i]**2
    #    #dipole
    #    U_d += torch.sum(p[i]**2)
    #    #charge-quadrupole
    #    U_cq += q[i] * torch.trace(t[i])
    #    #quadrupole
    #    U_t += torch.sum(t[i]*t[i])
    #multiplying with correct constants
    U_q = -alpha * torch.dot(q, q) / math.sqrt(math.pi)
    #U_d *= 2 * alpha**3 /(3 * math.sqrt(math.pi))
    #U_cq *= 2*alpha**3/(3 * math.pi)
    #U_t *= 8*alpha**5/(45*math.pi)
    #total self-interaction energy
    U_self = U_q #+ U_d + U_cq + U_t
    #U_self = constant * U_self
    #print("U_MONO: ", U_q)
    #print("U_DIPOLE: ", U_d)
    #print("U_CHARGE_QUADRUPOLE: ", U_cq)
    #print("U_QUADRUPOLE: ", U_t)
    return U_self


def compute_ewald(coords, q ,p, t,box, alpha, rcutoff, kcutoff):
    #Total energy = short range + long range - self interaction
    #print("CHARGES: ", q)
    #print("DIPOLES: ", p)
    #print("QUADRUPOLES: ", t)
    U_s = short_range(coords, q,p,t, box, alpha, rcutoff) 
    U_l =  long_range(coords, q,p,t, box, alpha, kcutoff)  
    U_self = self_interaction(coords, q,p,t, alpha)
    U_ewald = U_s + U_l - U_self
    print(f"FINAL ENERGIES: U_s = {U_s} , U_l = {U_l} , U_self = {U_self}, Total Coulombic Energy = {U_ewald}")
    return U_ewald



