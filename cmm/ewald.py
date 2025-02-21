import torch
from scipy import special
from torch.special import erfc
import math
from . import units, pbc
from .units import EPSILON0 , ELE_CHG 
from .pbc import applyPBC
from .electrostatics import computeInteractionTensor
import numpy as np
torch.set_printoptions(profile="full")

def short_range_ewald_pairwise(
    natoms: torch.NumberType,
    pairs_i_a: torch.Tensor,
    pairs_j_a: torch.Tensor,
    dists_p: torch.Tensor,
    dist_vecs_p: torch.Tensor,
    mPoles_a: torch.Tensor,
    Z_a: torch.Tensor
):
    mPoles_i_p = mPoles_a[pairs_i_a]
    mPoles_j_p = mPoles_a[pairs_j_a]
    Z_i_p = Z_a[pairs_i_a]
    Z_j_p = Z_a[pairs_j_a]

    drInv = 1 / dists_p
    drInv3 = torch.pow(drInv, 3)
    drInv5 = torch.pow(drInv, 5)

    # Core-Core interactions #
    ePotCore = Z_i_p * drInv
    eFieldCore = dist_vecs_p * (Z_i_p * drInv3).unsqueeze(-1)
    eFieldGradCore_1 = torch.vmap(torch.mul)(torch.vmap(torch.outer)(dist_vecs_p, dist_vecs_p), (3 * Z_i_p * drInv5))
    I = torch.eye(3, device=Z_a.device)
    I = I.reshape((1, 3, 3))
    I = I.repeat((Z_i_p.size(0), 1, 1))
    eFieldGradCore_2 = torch.vmap(torch.mul)(I, (Z_i_p * drInv3))
    eFieldGradCore = (eFieldGradCore_2 - eFieldGradCore_1)
    eFieldGradCore = eFieldGradCore.view(-1, 9)

    # damping factors
    oneCenterDamps_i = computeOneCenterDampFactorsSlater(dists_p, b_i_p)
    twoCenterDamps = computeTwoCenterDampFactorsSlater(dists_p, b_ij_p)

    # interaction tensors
    cs_tensor_ij = computeInteractionTensor(dist_vecs_p, oneCenterDamps_i, drInv)
    ss_tensor_ij = computeInteractionTensor(dist_vecs_p, twoCenterDamps, drInv)

    # core-shell interactions
    eData_i = torch.bmm(cs_tensor_ij, mPoles_i_p.unsqueeze(2))
    ePot_i = eData_i[:, 0].flatten()
    eField_i = eData_i[:, 1:4].reshape(-1, 3)
    eFieldGrad_i = eData_i[:, 4:].reshape(-1, 6)
    scPairwiseEnergies = ePot_i * Z_j_p

    # shell-shell interactions
    ss_edata = torch.bmm(ss_tensor_ij, mPoles_i_p.unsqueeze(2))
    ssPairwiseEnergies = torch.bmm(mPoles_j_p.unsqueeze(1), ss_edata).flatten()

    # Accumulate fields #
    E_potentials = torch.zeros(natoms, device=Z_a.device) # N
    E_fields = torch.zeros(natoms, 3, device=Z_a.device) # Nx3
    E_field_grads = torch.zeros(natoms, 6, device=Z_a.device) # Nx6 because only store upper triangle

    # How do I do this in a way that doesn't copy?
    E_potentials.scatter_add_(0, pairs_j_a, ePot_i + ePotCore)
    E_fields[:, 0].scatter_add_(0, pairs_j_a, eFieldCore[:, 0] - eField_i[:, 0])
    E_fields[:, 1].scatter_add_(0, pairs_j_a, eFieldCore[:, 1] - eField_i[:, 1])
    E_fields[:, 2].scatter_add_(0, pairs_j_a, eFieldCore[:, 2] - eField_i[:, 2])
    # Yes, I am doing it like this. Please help.
    E_field_grads[:, 0].scatter_add_(0, pairs_j_a, eFieldGradCore[:, 0] - eFieldGrad_i[:, 0])
    E_field_grads[:, 1].scatter_add_(0, pairs_j_a, eFieldGradCore[:, 1] - eFieldGrad_i[:, 1])
    E_field_grads[:, 2].scatter_add_(0, pairs_j_a, eFieldGradCore[:, 2] - eFieldGrad_i[:, 2])
    E_field_grads[:, 3].scatter_add_(0, pairs_j_a, eFieldGradCore[:, 4] - eFieldGrad_i[:, 3])
    E_field_grads[:, 4].scatter_add_(0, pairs_j_a, eFieldGradCore[:, 5] - eFieldGrad_i[:, 4])
    E_field_grads[:, 5].scatter_add_(0, pairs_j_a, eFieldGradCore[:, 8] - eFieldGrad_i[:, 5])

    elecPairwiseEnergies = scPairwiseEnergies + ssPairwiseEnergies
    ene_elec = 0.5 * (torch.sum(Z_a * E_potentials) + torch.sum(elecPairwiseEnergies))
    return ene_elec, E_potentials, E_fields, E_field_grads

# This file contains the Ewald Summation for computing Long range interactions
def short_range(coords, q , p, t, box, alpha, rcutoff):
    U_cc, U_cd, U_dd, U_ct, U_dt, U_tt = 0, 0, 0, 0, 0, 0
    constant = 1/(4*math.pi * EPSILON0) 
    N = len(coords) #atoms
    boxInv = torch.linalg.inv(box)  # Inverse of the simulation box matrix
    for i in range(N):
        for j in range(i+1,N):
            drVec = coords[j] - coords[i]
            r_ = applyPBC(drVec.unsqueeze(0), box, boxInv).squeeze(0)
            r = torch.norm(r_) #Get distance between two points.
            if r < 3.0: # TEMPORARY HACK TO AVOID INTERNAL PAIRS
                continue
            rhat = r_ / r
            if r < rcutoff:
                r2 = r**2
                r3 =r2*r
                r4 = r3*r
                r5 = r3*r2
                r7 = r5*r2
                r9 = r7*r2
                alpha2 = alpha*alpha
                alpha3 = alpha2*alpha
                delta = torch.eye(3) #kronecker delta
                ########|||||Ewald screening functions are represented by f_ |||||#######
                f_0 = erfc(r * alpha)
                f_1 = (2*alpha*r/math.pi * torch.exp(-alpha2 * r2)) + f_0
                f_2 = 4 * alpha3/math.sqrt(math.pi) * torch.exp(-alpha2 * r2)/r2
                f_3 = r3 * f_2
                f_4 = 8 * alpha/math.pi * (alpha2 * r2 + 1)/r4 * torch.exp(-alpha2 * r2)
                f_5 = r3 * f_4 - 3 * r * f_2
                f_6 = 2*(alpha2 + 1/r2) * f_4 + 4*f_2/r4
                # HERE: Need to get proper reference cause I'm pretty sure this is all wrong.
                #print(f_0.item(), f_1.item(), f_2.item(), f_3.item(), f_4.item(), f_5.item(), f_6.item())
                #######|||||Multipole tensors are represented by T_. Also implemented are the D_ tensors|||||#######
                T_ = 1/r
                T_a = -rhat/r3
                T_ab = (3 * torch.outer(r_,r_) - r2 * delta)/r5
                #################################################################
                term1_abg = 15 * torch.einsum("i,j,k->ijk", r_,r_,r_)
                term2_abg = -3 * r2 *(
                            torch.einsum("i,jk->ijk", r_, delta)+
                            torch.einsum("j,ik->ijk", r_, delta)+
                            torch.einsum("k,ij->ijk", r_, delta)
                )
                T_abg = -(term1_abg + term2_abg) / r7
                #################################################################
                term1_abgd = 105 * torch.einsum("i,j,k,l -> ijkl", r_,r_,r_,r_)
                term1 = torch.einsum("i,j,kl->ijkl", r_, r_, delta)  # R_ij,a R_ij,b k_gd
                term2 = torch.einsum("i,k,jl->ijkl", r_, r_, delta)  # R_ij,a R_ij,g k_bd
                term3 = torch.einsum("i,l,jk->ijkl", r_, r_, delta)  # R_ij,a R_ij,d k_bg
                term4 = torch.einsum("j,k,il->ijkl", r_, r_, delta)  # R_ij,b R_ij,g k_ad
                term5 = torch.einsum("j,l,ik->ijkl", r_, r_, delta)  # R_ij,b R_ij,d k_ag
                term6 = torch.einsum("k,l,ij->ijkl", r_, r_, delta)  # R_ij,g R_ij,d k_ab
                term2_abgd = -15 * r2 * (term1 + term2 + term3 + term4 + term5 + term6)
                term3_abgd = 3 * r4 * (
                  torch.einsum("ij,kl->ijkl", delta, delta) +
                  torch.einsum("ik,jl->ijkl", delta, delta) +
                  torch.einsum("il,jk->ijkl", delta, delta)
                )
                T_abgd = (term1_abgd +term2_abgd + term3_abgd)/r9 #final term
                D_ab = torch.einsum("i,j->ij", r_,r_)
                D_abgd = torch.einsum("ij,k,l -> ijkl", delta, r_, r_) + torch.einsum("ik,j,l->ijkl", delta, r_, r_) + torch.einsum("il,j,k->ijkl", delta, r_, r_)
                ###charge-charge###
                U_cc += (q[i] * q[j])*f_0*T_    
                ###charge-dipole###
                U_cd += -f_1*(q[i]*torch.dot(T_a,p[j]) - q[j]*torch.dot(T_a,p[i]))
                ###dipole-dipole###
                U_dd += -(f_1* torch.dot(p[i],torch.matmul(T_ab,p[j])) + f_2 *torch.dot(p[i],torch.matmul(D_ab,p[j])))
                ###charge-quadrupole###
                U_ct += f_1*(q[i]*torch.sum(T_ab * t[j]) + q[j]* torch.sum(T_ab*t[i]))  + f_2*(q[i] * torch.sum(D_ab * t[j]) + q[j] * torch.sum(D_ab * t[i]))
                ###dipole-quadrupole###
                term1_dt = f_1 * (torch.einsum("i,ijk,ik->",p[i], T_abg,t[j]) - torch.einsum("i,ijk,ik->",p[j], T_abg,t[i   ]))
                term2_dt = f_2 * (torch.einsum("i,j,ij->", p[i],r_,t[j]) - torch.einsum("i,j,ij->", p[j],r_,t[i]))
                term3_dt = -f_3 * (torch.einsum("i,i,jk,jk->",p[i],r_,T_ab,t[j]) - torch.einsum("i,i,jk,jk->",p[j],r_,T_ab,t[i]))
                term4_dt = -f_4 * (torch.einsum("i,i,jk,jk->",p[i],r_,D_ab,t[j]) - torch.einsum("i,i,jk,jk->",p[j],r_,D_ab,t[i])) 
                U_dt += term1_dt + term2_dt + term3_dt + term4_dt
                ###quadrupole-quadrupole###
                term1_tt = f_1 * torch.einsum("ij,ijkl,kl->",t[i],T_abgd,t[j])
                term2_tt = 2 * f_2 * torch.einsum("ij,ij->",t[i],t[j])
                term3_tt = f_3 * (torch.einsum("ij,ij,kl,kl->",t[i],delta,T_ab,t[j]) + torch.einsum("ij,ij,kl,kl",t[i],T_ab,delta,t[j]))/2
                term4_tt = -f_3 * (torch.einsum("ij,i,jkl,kl->",t[i],r_,T_abg,t[j]) + torch.einsum("ij,i,jkl,kl->",t[j],r_,T_abg,t[i]))
                term5_tt = -f_4 * (torch.einsum("ij,ijkl,kl->",t[i],D_abgd,t[j]) + torch.einsum("ij,klij,kl->",t[i],D_abgd,t[j]))/2 
                term6_tt = -2 * f_4 * torch.einsum("i,ij,jk,k->",r_,t[i],t[j],r_)
                term7_tt = f_5 * 1/2 * (torch.einsum("ij,ij->",D_ab,t[i]) * torch.einsum("ij,ij->", T_ab,t[j])+
                           torch.einsum("ij,ij->",D_ab,t[j]) * torch.einsum("ij,ij->", T_ab,t[i])
                           ) 
                term8_tt = f_6 * torch.einsum("ij,ij->",D_ab,t[i]) * torch.einsum("ij,ij->",D_ab,t[j]) 
                U_tt = term1_tt + term2_tt + term3_tt + term4_tt + term5_tt + term6_tt + term7_tt + term8_tt

                #print(f"Atom {i}-{j}: r = {r}, q[i] = {q[i]}, q[j] = {q[j]},"
                #      f" p[i] = {p[i]}, p[j] = {p[j]} U_cc = {U_cc},"
                #      f"U_cd = {U_cd} , U_dd = {U_dd}, U_ct = {U_ct},"
                #      f"U_dt = {U_dt}, U_tt = {U_tt}")
    U_ct *= 1/3
    U_dt *= 1/3
    U_tt *= 1/9
    U_s = U_cc + U_cd + U_dd + U_ct + U_dt + U_tt
    return U_s

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

def long_range_potential_vectorized(coords: torch.Tensor, q: torch.Tensor, p: torch.Tensor, t: torch.Tensor, box: torch.Tensor, alpha: torch.Tensor, max_hkl: torch.NumberType):

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
    cos_k_dot_r = torch.cos(2 * torch.pi * k_dot_r)
    sin_k_dot_r = torch.sin(2 * torch.pi * k_dot_r)
    F_l_real = q.expand(kvectors.size(0), -1) - torch.einsum('kj,nij,ki->kn', kvectors, t, kvectors) * (2 * torch.pi) * (2 * torch.pi)
    F_l_imag = torch.matmul(kvectors, p.T) * 2 * torch.pi
    exp_k_dot_r = torch.complex(cos_k_dot_r, sin_k_dot_r)
    exp_minus_k_dot_r = torch.complex(cos_k_dot_r, -sin_k_dot_r)
    F_2 = torch.complex(F_l_real, F_l_imag)
    structure_factors = torch.sum(F_2 * exp_k_dot_r, dim=1)
    phi_expanded = (gaussian_factors * structure_factors).unsqueeze(1) * exp_minus_k_dot_r
    phi = torch.sum(phi_expanded.real, dim=0) / (torch.pi * V) # can take .real inside sum since .imag sums to zero.
    field = 2 * (
        torch.matmul(phi_expanded.T, torch.complex(torch.zeros_like(kvectors), kvectors)).real
    ) / V

    return phi, field

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



