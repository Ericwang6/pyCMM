import torch
import numpy as np

from cmm.units import BOHR2NM, HARTREE2KJ, HARTREE2KCAL, BOHR2ANG
from cmm.multipole import computeLocal2GlobalRotationMatrix, rotateMultipoles, rotateQuadrupoles, computeCartesianQuadrupoles
from cmm.short_range import computeShortRangeEnergy, scaleMultipoles, computePairwiseChargeTransfer
from cmm.dispersion import computeDispersion
from cmm.electrostatics import computePermElecAndPolarizationEnergy
from cmm.cmm_water import CMMWater

def test_geometry_dependent_params():
    torch.set_default_dtype(torch.float64)
    model = CMMWater(2, do_polarization=False)
    coords = torch.tensor(np.array([
         [ 0.0031771858,  1.4710501499, -0.0034222052],
         [ 0.0981342864,  0.5090994249, -0.0041139499],
         [ 0.8976520298,  1.8147098331,  0.0031728568],
         [-0.0036002768, -1.3547622039,  0.0027150961],
         [-0.492886242,  -1.6733692175,  0.7647713563],
         [-0.4948459831, -1.6611969865, -0.763123154 ]
    ]) / BOHR2ANG, dtype=torch.float64, requires_grad=True)
    box = torch.tensor(np.eye(3) * 100, dtype=torch.float64, requires_grad=True)

    energies = model.computeEnergy(coords, box)
    energies['tot'].backward()
    grad = coords.grad

    j_OH = -0.024794
    j_OH_bb = -0.0332338
    j_HOH = 0.0220891
    q_O = -0.390896
    O_Z = 3.61565
    H_Z = 0.93619

    O_hardness = 6.18699e-6
    H_hardness = 0.561535
    k_OH_bb_hardness = 0.958157
    k_OH_hardness = 2.32191
    k_OH_θ_hardness = -0.0991956

    O1_Z_ref, O1_q_shell_ref = 3.61565, -4.006514108368438
    H1_Z_ref, H1_q_shell_ref = 0.93619, -0.7406910974925237
    H2_Z_ref, H2_q_shell_ref = 0.93619, -0.7408247941390381
    O2_Z_ref, O2_q_shell_ref = 3.61565, -4.007096982573129
    H3_Z_ref, H3_q_shell_ref = 0.93619, -0.740466890458911
    H4_Z_ref, H4_q_shell_ref = 0.93619, -0.7404661269679597

    O1_hardness_ref = 6.18699e-6
    H1_hardness_ref = 0.5499299397234746
    H2_hardness_ref = 0.5565209286526765
    O2_hardness_ref = 6.18699e-6
    H3_hardness_ref = 0.5576284661496946
    H4_hardness_ref = 0.557590426519481

#def test_field_dependent_morse():
#    torch.set_default_dtype(torch.float64)
#
#    coords = get_water_dimer_coords()
#    print(coords)
#
#    def get_perm_elec_fields(coords: torch.Tensor):
#        pairs, param_elec, param_pauli, param_disp, param_pol, param_xpol, param_ct = water_data(coords)
#
#        # Perm elec and polarization parameters
#        Z, mPoles, b = param_elec
#        alpha, eta, groupCharges = param_pol
#        perm_elec, pol = computePermElecAndPolarizationEnergy(
#            coords,
#            [[0, 1, 2], [3, 4, 5]],
#            mPoles,
#            Z,
#            b,
#            True,
#            alpha,
#            eta,
#            groupCharges,
#        )
#    
#    get_perm_elec_fields(coords)
#
#    # permanent fields
#    # core potential ref.
#    # -0.018479444438618625
#    # -0.04177127228520493
#    # -0.015155958241297148
#    #  0.026296788643028986
#    #  0.014373349856660637
#    #  0.014373361809481053
#    # E_field_core reference
#    #[-0.007080152596418484, 1.5502342969799704e-8, -0.0021537177257822614]
#    #[-0.020038994986737075, 2.1921807617141182e-8, -0.0029933276979837334]
#    #[-0.004908679507711978, 8.277222945898368e-10, 0.00016868336622703387]
#    #[-0.013794464558691594, -1.8833728761974327e-8, 0.0016810526562941485]
#    #[-0.005330512573582459, -0.0030473781495875137, 0.0030252192820030385]
#    #[-0.0053305191443680875, 0.0030473416888460144, 0.003025232274093582]
#
#    assert False