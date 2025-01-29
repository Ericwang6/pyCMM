import torch, math
from torch_scatter import scatter
from .multipole import computeCartesianQuadrupoles, rotateMultipoles, rotateQuadrupoles
from .electrostatics import computePermanentElectricPotentialExpansionAndEnergyFromPairs, computePolarizationEnergyAndInducedMultipolesFromPairs, computeInducedElectricPotentialAndFieldsFromPairs
from .polarization import direct_field_induced_dipole_guess, solvePolarizationByCG, computeProductWithPolarizationMatrix
from .short_range import scaleMultipoles, computePairwiseChargeTransfer, computeShortRangeEnergyFromPairs
from .dispersion import computeDispersionFromPairs
from .coordinate_manager import CoordinateManager
from .parameters import Parameterizer
from .topology import Topology
from .terms import *
from .units import *

from copy import copy

# NOTE(JOE): The design of this object is still up in the air. I think that we could
# allow inheritance for the purpose of making it really trivial to set
# up a force field. This just saves the user having to call the appropriate
# set up functions manually I guess? Custom force fields can just work with
# the base class I think. Ultimately all that this object does is hold
# onto a list of functions we need to call and all of the parameters
# needed to pass to the Parameterizer to populate the parameter arrays.
# Also, in the future, we will add the option to specify parameters
# from a file or dictionary or something.

# TODO: The axis types are specified as follows:
# 0 = Identity
# 1 = z-then-x
# 2 = bisector
# That's all we have for now. The axis type should really be specified by the
# force field by using a mapping from the atom type to the axis type. Don't have that yet.

class ForceField(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._atomic_params = {}
        self._pair_params = {}

class CMM(ForceField):
    def __init__(self) -> None:
        super().__init__()
        # These are local indices for the atom types, not the actual
        # atom type indices which are decided by the Parameterizer.
        self._types_to_index = {
            "O_water": 0, "H_water": 1,
            2: "F-", 3: "Cl-", 4: "Br-", 5: "I-",
            6: "Li+", 7: "Na+", 8: "K+", 9: "Rb+", 10: "Cs+",
            11: "Mg2+", 12: "Ca2+"
        }

        self._build()
    
    def _build(self):
        # NOTE: The purpose of this function is to produce the atomic_params, bond_params,
        # angle_params, and pair_params dictionaries. These are what will ultimately be
        # used by the parameterizer to produce the full-sized parameter
        # arrays which are used when evaluating the force field.
        # Everything else used in this function is just for convenience
        # when building atomic_params, bond_params, and pair_params.

        # Electrostatic raw params #
        Z = torch.tensor([
            3.61565, 0.93619,
            4.693, 12.1239, 18.9726, 35.5833,
            -0.895467, 3.5489, 7.73324, 12.2026, 11.5038,
            2.83412, 4.93631
        ])
        mono = torch.tensor([
            -0.390896, 0.195448, # Water
            -1.0, -1.0, -1.0, -1.0, # Halides
            1.0, 1.0, 1.0, 1.0, 1.0, # Alkali
            2.0, 2.0 # Mg2+, Ca2+
        ])
        dipo = torch.tensor([
            [0.0,       0.0, -0.094298], # O_water
            [0.0910288, 0.0, -0.207851], # H_water
            [0.0,       0.0,  0.0],      # F-
            [0.0,       0.0,  0.0],      # Cl-
            [0.0,       0.0,  0.0],      # Br-
            [0.0,       0.0,  0.0],      # I-
            [0.0,       0.0,  0.0],      # Li+
            [0.0,       0.0,  0.0],      # Na+
            [0.0,       0.0,  0.0],      # K+
            [0.0,       0.0,  0.0],      # Rb+
            [0.0,       0.0,  0.0],      # Cs+
            [0.0,       0.0,  0.0],      # Mg2+
            [0.0,       0.0,  0.0],      # Ca2+
        ])
        quad_s = torch.tensor([
            # Q20,       Q21c,      Q21s, Q22c,       Q22s
            [-0.330685,  0.0,       0.0,  0.869923,   0.0], # O_water
            [-0.0739388, 0.0929482, 0.0,  0.00532425, 0.0], # H_water
            [0.0,        0.0,       0.0,  0.0,        0.0], # F-
            [0.0,        0.0,       0.0,  0.0,        0.0], # Cl-
            [0.0,        0.0,       0.0,  0.0,        0.0], # Br-
            [0.0,        0.0,       0.0,  0.0,        0.0], # I-
            [0.0,        0.0,       0.0,  0.0,        0.0], # Li+
            [0.0,        0.0,       0.0,  0.0,        0.0], # Na+
            [0.0,        0.0,       0.0,  0.0,        0.0], # K+
            [0.0,        0.0,       0.0,  0.0,        0.0], # Rb+
            [0.0,        0.0,       0.0,  0.0,        0.0], # Cs+
            [0.0,        0.0,       0.0,  0.0,        0.0], # Mg2+
            [0.0,        0.0,       0.0,  0.0,        0.0], # Ca2+
        ])
        
        b_elec = torch.tensor([
            2.13358, 2.33322, # Water
            2.42894, 1.77558, 1.73844, 1.70583, # Halides
            4.44984, 2.59626, 2.39879, 2.38187, 2.03392, # Alkali
            1.92445, 2.11191, # Mg2+, Ca2+
        ])

        b_pauli = torch.tensor([
            2.1975, 1.96474, # Water
            1.70352, 1.4384, 1.38496, 1.35031, # Halides
            2.6441, 2.54145, 2.2465, 2.27644, 2.0059, # Alkali
            2.66576, 2.21834, # Mg2+, Ca2+
        ])

        b_disp = torch.tensor([
            1.84302, 1.30993, # Water
            1.21488, 1.07019, 0.978881, 1.30013, # Halides
            2.23422, 1.99839, 1.95926, 4.01118, 4.01118, # Alkali
            1.60887, 1.63789, # Mg2+, Ca2+
        ])

        b_ct = torch.tensor([
            1.89485, 2.36763, # Water
            1.37059, 0.948365, 0.882003, 0.841482, # Halides
            1.65423, 1.867647, 2.04256, 2.02393, 1.94083, # Alkali
            1.5,     1.6, # Mg2+, Ca2+
        ])

        b_xpol = torch.tensor([
            2.73582, 2.04028, # Water
            1.90554, 1.60669, 1.4814, 1.38744, # Halides
            2.6441, 2.54145, 2.2465, 2.27644, 2.0059, # Alkali
            5.14456, 3.67375, # Mg2+, Ca2+
        ])

        C6_disp = torch.tensor([
            35.8289, 1.98954, # Water
            146.12, 661.859, 1115.92, 1358.97, # Halides
            0.609382, 5.4421, 45.6395, 63.085, 170.628, # Alkali
            3.70387, 26.5808, # Mg2+, Ca2+
        ])

        q_pauli = torch.tensor([
            6.50923, 0.527804, # Water
            3.70166, 5.80737, 7.17975, 10.7103, # Halides
            1.54888, 4.17489, 10.3705, 19.0684, 20.9948, # Alkali
            4.25833, 8.06718, # Mg2+, Ca2+
        ])

        Kdipo_pauli = torch.tensor([
            -5.61925, -0.515584, # Water
            0.0, 0.0, 0.0, 0.0, # Halides
            0.0, 0.0, 0.0, 0.0, 0.0, # Alkali
            0.0, 0.0, # Mg2+, Ca2+
        ])

        Kquad_pauli = torch.tensor([
            -1.56567, -0.440164, # Water
            0.0, 0.0, 0.0, 0.0, # Halides
            0.0, 0.0, 0.0, 0.0, 0.0, # Alkali
            0.0, 0.0, # Mg2+, Ca2+
        ])

        q_ct_acc = torch.tensor([
            -0.67857, 1.36735, # Water
            0.122194, -1.51565, -1.65479, -1.253, # Halides
            0.913977, 1.03203, 7.34713, 12.6713, 25.443, # Alkali
            3.5885, 7.30099 # Mg2+, Ca2+
        ])

        Kdipo_ct_acc = torch.tensor([
            0.0, 0.0, # Water
            0.0, 0.0, 0.0, 0.0, # Halides
            0.0, 0.0, 0.0, 0.0, 0.0, # Alkali
            0.0, 0.0, # Mg2+, Ca2+
        ])

        Kquad_ct_acc = torch.tensor([
            0.0, 0.0, # Water
            0.0, 0.0, 0.0, 0.0, # Halides
            0.0, 0.0, 0.0, 0.0, 0.0, # Alkali
            0.0, 0.0, # Mg2+, Ca2+
        ])

        q_ct_don = torch.tensor([
            0.757752, 0.00888982, # Water
            0.586295, 0.94665, 1.08484, 1.37576, # Halides
            -0.120616, 0.170082, 0.669004, 1.81764, 3.42462, # Alkali
            -0.499793, 0.655079, # Mg2+, Ca2+
        ])

        Kdipo_ct_don = torch.tensor([
            -0.512036, -0.0511668, # Water
            0.0, 0.0, 0.0, 0.0, # Halides
            0.0, 0.0, 0.0, 0.0, 0.0, # Alkali
            0.0, 0.0, # Mg2+, Ca2+
        ])

        Kquad_ct_don = torch.tensor([
            -0.208186, 0.0568152, # Water
            0.0, 0.0, 0.0, 0.0, # Halides
            0.0, 0.0, 0.0, 0.0, 0.0, # Alkali
            0.0, 0.0, # Mg2+, Ca2+
        ])

        q_xpol = torch.tensor([
            1.26592, 0.200089, # Water
            -0.107239, -0.622035, -0.708194, -0.883374, # Halides
            -4.655, -5.18609, -2.06504, 0.485645, 9.65725, # Alkali
            -448.263, -220.293, # Mg2+, Ca2+
        ])

        eta = torch.tensor([
            6.18699e-6, 0.561535, # Water
            0.0, 0.0, 0.0, 0.0, # Halides
            0.0, 0.0, 0.0, 0.0, 0.0, # Alkali
            0.0, 0.0, # Mg2+, Ca2+
        ])

        axistypes = torch.tensor([
            2, 1, # Water
            0, 0, 0, 0, # Halide
            0, 0, 0, 0, 0, # Alkali
            0, 0, # Divalent cations
        ])

        alpha = torch.tensor([
            torch.diag([4.45992, 6.07259, 4.55391]), # O_water
            torch.diag([2.22001, 1.66835, 0.183855]), # H_water
            torch.diag([11.7270176, 11.7270176, 11.7270176]), # F-
            torch.diag([32.2880907, 32.2880907, 32.2880907]), # Cl-
            torch.diag([42.7172275, 42.7172275, 42.7172275]), # Br-
            torch.diag([64.1111144, 64.1111144, 64.1111144]), # I-
            torch.diag([0.1586152, 0.1586152, 0.1586152]), # Li+
            torch.diag([0.9542199, 0.9542199, 0.9542199]), # Na+
            torch.diag([5.5376271, 5.5376271, 5.5376271]), # K+
            torch.diag([8.6857518, 8.6857518, 8.6857518]), # Rb+
            torch.diag([15.7177865, 15.7177865, 15.7177865]), # Cs+
            torch.diag([0.4822524, 0.4822524, 0.4822524]), # Mg2+
            torch.diag([3.2809409, 3.2809409, 3.2809409]), # Ca2+
        ])

        alpha_damp_exponent = torch.tensor([
            0.0, 0.0, # Water
            241.724, 428.717, 484.249, 599.029, # Halides
            0.0, 0.0, 0.0, 0.0, 0.0, # Alkali
            0.0, 0.0, # Mg2+, Ca2+
        ])

        alpha_damp_max = torch.tensor([
            0.0, 0.0, # Water
            0.75, 0.75, 0.75, 0.75, # Halides
            0.0, 0.0, 0.0, 0.0, 0.0, # Alkali
            0.0, 0.0, # Mg2+, Ca2+
        ])

        # TODO: Add all the ion-ion pair-specific parameters!

        self._raw_atomic_params = {
            # elec
            "Z": Z,
            "q_shell": mono - Z,
            "dipo": dipo,
            "quad": computeCartesianQuadrupoles(quad_s),
            "b_elec": b_elec,
            # Pauli repulsion
            "b_pauli": b_pauli,
            "q_pauli": q_pauli,
            "Kdipo_pauli": Kdipo_pauli,
            "Kquad_pauli": Kquad_pauli,
            # Dispersion
            "C6_disp": C6_disp,
            "b_disp": b_disp,
            # Polarization
            "alpha": alpha,
            "alpha_damp_exponent": alpha_damp_exponent,
            "alpha_damp_max": alpha_damp_max,
            "eta": eta,
            # Exchange-polarization
            "b_xpol": b_xpol,
            "q_xpol": q_xpol,
            "Kdipo_xpol": torch.zeros((len(self._types_to_index),)),
            "Kquad_xpol": torch.zeros((len(self._types_to_index),)),
            # Charge Transfer
            "b_ct": b_ct,
            "q_ct_acc": q_ct_acc,
            "Kdipo_ct_acc": Kdipo_ct_acc,
            "Kquad_ct_acc": Kquad_ct_acc,
            "q_ct_don": q_ct_don,
            "Kdipo_ct_don": Kdipo_ct_don,
            "Kquad_ct_don": Kquad_ct_don,
            "axistypes": axistypes,
        }

        self.pair_params = {
            ("O_water", "H_water"): {
                "D": torch.tensor([524.265 / HARTREE2KJ]),
                "k_b": torch.tensor([5098.15 / HARTREE2KJ * BOHR2ANG * BOHR2ANG]),
                "r_eq": torch.tensor([0.958929 / BOHR2ANG]),
                "j_cf_pauli": torch.tensor([0.0911036]),
                "j_cf": torch.tensor([-0.024794]),
                "k_hardness_b": torch.tensor([2.32191]),
                "dip_deriv_1": torch.tensor([0.1654220912271531]),
                "dip_deriv_2": torch.tensor([-0.012458400000000472]),
                "ct_slope_1": torch.tensor([65.0]),
                "ct_slope_2": torch.tensor([13.7812]),
                "eps": torch.tensor([1.0 / 0.380979]),
            },
            ("H_water", "F-"): {"eps": torch.tensor([1.0 / 1.78074]),},
            ("H_water", "Cl-"): {"eps": torch.tensor([1.0 / 0.929684]),},
            ("H_water", "Br-"): {"eps": torch.tensor([1.0 / 0.894156]),},
            ("H_water", "I-"): {"eps": torch.tensor([1.0 / 0.655324]),},
            ("O_water", "Li+"): {"eps": torch.tensor([1.0 / 0.964901]),},
            ("O_water", "Na+"): {"eps": torch.tensor([1.0 / 0.80]),},
            ("O_water", "K+"): {"eps": torch.tensor([1.0 / 0.70]),},
            ("O_water", "Rb+"): {"eps": torch.tensor([1.0 / 0.684706]),},
            ("O_water", "Cs+"): {"eps": torch.tensor([1.0 / 0.584055]),},
            ("O_water", "Mg2+"): {"eps": torch.tensor([1.0 / 0.638288]),},
            ("O_water", "Ca2+"): {"eps": torch.tensor([1.0 / 2.4784]),},
        }

        self.pair_pair_params = {
            (("O_water", "H_water"), ("O_water", "H_water")): {
                "j_cf_bb": torch.tensor([-0.0332338]),
                "k_hardness_bb": torch.tensor([0.958157]),
                "k_bb": torch.tensor([-61.1423 / HARTREE2KJ * BOHR2ANG * BOHR2ANG]),
            },
        }

        self.pair_angle_params = {
            (("O_water", "H_water"), ("H_water", "O_water", "H_water")): {
                "k_ba": torch.tensor([-159.886 / HARTREE2KJ * BOHR2ANG]),
            },
        }

        self.angle_params = {
            ("H_water", "O_water", "H_water"): {
                "theta_eq": torch.tensor([104.4234 * math.pi / 180.0]),
                "k_theta": torch.tensor([452.183 / HARTREE2KJ]),
                "j_cf_angle": torch.tensor([0.0220891]),
                "k_hardness_angle": torch.tensor([-0.0991956]),
            }
        }

        with torch.no_grad():
            self.atomic_params = {}
            for type_key in self._types_to_index.keys():
                these_atomic_params = {}
                for param_key in self._raw_atomic_params.keys():
                    these_atomic_params[param_key] = self._raw_atomic_params[param_key][self._types_to_index[type_key]]
                    self.atomic_params[type_key] = these_atomic_params
            
            # Symmetrize the parameter dictionaries for convenience when making parameter arrays #
            for key in list(self.pair_params.keys()):
                self.pair_params[(key[1], key[0])] = self.pair_params[key]
            for key in list(self.angle_params.keys()):
                self.angle_params[(key[2], key[1], key[0])] = self.angle_params[key]

    #@torch.compile
    def evaluate(self, cm: CoordinateManager, topology: Topology, params: Parameterizer):
        pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()

        # TODO: Should check if the parameters need to be updated here before actually doing anything!! #

        angles = computeAngleFromVecs(dist_vecs[topology.angle_pairs[0]], dist_vecs[topology.angle_pairs[1]])

        # All pairs forming an angle #
        pairs_angles_a = topology.angle_pairs.T.flatten()

        # Intermolecular pairs #
        pairs_inter_i_a = pairs[:, 0][topology.all_intermolecular_pairs]
        pairs_inter_j_a = pairs[:, 1][topology.all_intermolecular_pairs]

        dists_inter = dists[topology.all_intermolecular_pairs]
        dist_vecs_inter = dist_vecs[topology.all_intermolecular_pairs]

        # Electric Multipoles #
        Z = params.get_atomic_parameters('Z')
        q_shell = params.get_atomic_parameters('q_shell')
        dipo = params.get_atomic_parameters('dipo')
        quad = params.get_atomic_parameters('quad')
        natoms = torch.tensor(Z.size(0), device=pairs.device)

        # Polarizability #
        alpha = params.get_atomic_parameters('alpha')

        # Pauli Multipoles #
        q_pauli = params.get_atomic_parameters('q_pauli')
        Kdipo_pauli = params.get_atomic_parameters('Kdipo_pauli')
        Kquad_pauli = params.get_atomic_parameters('Kquad_pauli')

        # Exchange Polarization #
        q_xpol = params.get_atomic_parameters('q_xpol')
        Kdipo_xpol = params.get_atomic_parameters('Kdipo_xpol')
        Kquad_xpol = params.get_atomic_parameters('Kquad_xpol')

        # Charge Transfer Multipoles #
        q_ct_acc = params.get_atomic_parameters('q_ct_acc')
        Kdipo_ct_acc = params.get_atomic_parameters('Kdipo_ct_acc')
        Kquad_ct_acc = params.get_atomic_parameters('Kquad_ct_acc')
        q_ct_don = params.get_atomic_parameters('q_ct_don')
        Kdipo_ct_don = params.get_atomic_parameters('Kdipo_ct_don')
        Kquad_ct_don = params.get_atomic_parameters('Kquad_ct_don')

        # Dispersion Multipoles #
        C6_disp = params.get_atomic_parameters('C6_disp')

        # Atomic widths #
        b_elec = params.get_atomic_parameters('b_elec')
        b_pauli = params.get_atomic_parameters('b_pauli')
        b_disp = params.get_atomic_parameters('b_disp')
        b_xpol = params.get_atomic_parameters('b_xpol')
        b_ct = params.get_atomic_parameters('b_ct')
        eta = params.get_atomic_parameters('eta')

        eps = params.get_pair_parameters('eps', topology.all_intermolecular_pairs)
        r_eq = params.get_pair_parameters('r_eq', topology.bonded_pairs)
        k_b_p = params.get_pair_parameters('k_b', topology.bonded_pairs)
        D_p = params.get_pair_parameters('D', topology.bonded_pairs)
        dip_deriv_1_p = params.get_pair_parameters('dip_deriv_1', topology.bonded_pairs)
        dip_deriv_2_p = params.get_pair_parameters('dip_deriv_2', topology.bonded_pairs)
        ct_slope_1_p = params.get_pair_parameters('ct_slope_1', topology.bonded_pairs)
        ct_slope_2_p = params.get_pair_parameters('ct_slope_2', topology.bonded_pairs)
        j_cf = params.get_pair_parameters('j_cf', topology.bonded_pairs)
        j_cf_pauli = params.get_pair_parameters('j_cf_pauli', topology.bonded_pairs)
        k_hardness_b = params.get_pair_parameters('k_hardness_b', topology.bonded_pairs)
        
        # NOTE(JOE): Need to test that we get the right bond-bond parameters for non-symmetric angles.
        # Currently, we don't have parameters for a non-symmetric angle but they will come up with
        # organic molecules.
        r_eq_bb_1 = params.get_pair_parameters('r_eq', topology.angle_pairs[0])
        r_eq_bb_2 = params.get_pair_parameters('r_eq', topology.angle_pairs[1])
        r_eq_ba = torch.stack((r_eq_bb_1, r_eq_bb_2), dim=1).flatten()

        k_bb = params.get_pair_pair_parameters('k_bb', topology.angle_pairs[0], topology.angle_pairs[1])
        j_cf_bb_1 = params.get_pair_pair_parameters('j_cf_bb', topology.angle_pairs[0], topology.angle_pairs[1])
        j_cf_bb_2 = params.get_pair_pair_parameters('j_cf_bb', topology.angle_pairs[1], topology.angle_pairs[0])
        k_hardness_bb_1 = params.get_pair_pair_parameters('k_hardness_bb', topology.angle_pairs[0], topology.angle_pairs[1])
        k_hardness_bb_2 = params.get_pair_pair_parameters('k_hardness_bb', topology.angle_pairs[1], topology.angle_pairs[0])

        theta_eq = params.get_angle_parameters('theta_eq', topology.angle_atoms)
        k_theta = params.get_angle_parameters('k_theta', topology.angle_atoms)
        j_cf_angle = params.get_angle_parameters('j_cf_angle', topology.angle_atoms)
        k_hardness_angle = params.get_angle_parameters('k_hardness_angle', topology.angle_atoms)
        k_ba = params.get_pair_angle_parameters('k_ba', topology.angle_pairs, topology.angle_atoms)

        # Pauli charge flux #
        evaluate_bond_charge_flux(pairs, dists, topology.bonded_pairs, q_pauli, r_eq, j_cf_pauli)

        # Electrostatic charge flux #
        evaluate_bond_and_angle_charge_flux(
            pairs, dists, angles,
            topology.bonded_pairs, topology.angle_pairs, topology.angle_atoms,
            q_shell, r_eq, theta_eq, j_cf, j_cf_angle,
            r_eq_bb_1, r_eq_bb_2, j_cf_bb_1, j_cf_bb_2
        )
        
        # Hardness change #
        evaluate_hardness_change(
            pairs, dists, angles,
            topology.bonded_pairs, topology.angle_pairs, topology.angle_atoms,
            eta, r_eq, theta_eq, k_hardness_b, k_hardness_angle,
            r_eq_bb_1, r_eq_bb_2, k_hardness_bb_1, k_hardness_bb_2
        )
        eta_times_2 = 2 * eta

        # Rotation Matrices #
        rotation_matrices = cm.compute_rotation_matrices(params.get_atomic_parameters("axistypes"))
        
        multipoles = rotateMultipoles(
            q_shell, dipo, quad, rotation_matrices
        ) * torch.tensor([1, 1, 1, 1, 1/3, 2/3, 2/3, 1/3, 2/3, 1/3], device=pairs.device)
        polarizabilities = rotateQuadrupoles(alpha, rotation_matrices)
        inverse_polarizabilities = torch.linalg.inv(polarizabilities)

        # SPEED: Makes copies. Might be unavoidable but could maybe be done more efficiently.
        multipoles_ct_acc = scaleMultipoles(multipoles, q_ct_acc, Kdipo_ct_acc, Kquad_ct_acc)
        multipoles_ct_don = scaleMultipoles(multipoles, q_ct_don, Kdipo_ct_don, Kquad_ct_don)
        multipoles_pauli = scaleMultipoles(multipoles, q_pauli, Kdipo_pauli, Kquad_pauli)
        multipoles_xpol = scaleMultipoles(multipoles, q_xpol, Kdipo_xpol, Kquad_xpol)

        ct_direct_pairwise, dq_pairwise = computePairwiseChargeTransfer(
            dist_vecs_inter,
            multipoles_ct_acc[pairs_inter_i_a], multipoles_ct_acc[pairs_inter_j_a],
            multipoles_ct_don[pairs_inter_i_a], multipoles_ct_don[pairs_inter_j_a],
            b_ct[pairs_inter_i_a], b_ct[pairs_inter_j_a],
            eps
        )
        ene_ct_direct = torch.sum(ct_direct_pairwise) / 2
        dq_a = torch.zeros(natoms, device=pairs.device)
        dq_a = dq_a.scatter_add(0, pairs_inter_j_a, dq_pairwise)
        group_indices = torch.repeat_interleave(torch.arange(q_shell.size(0) // 3, device=pairs.device), 3)
        # ^^^ This is just a hack to get things working for water. Ultimately, we will
        # need a more general approach which will be provided by the topology. This
        # way of defining groups is somewhat troublesome. It means that we chop up molecules
        # into non-overlapping groups and some of the atoms then won't polarize charge to their
        # direct neighbors (not acceptable) OR we define a separate group for each atom which
        # overlaps with other groups. These would just be all of the 1-2, 1-3, and 1-4 neighbors
        # for a given atom. This is physically acceptable but increases the number of lagrange multipliers
        # considerably. Obviously in that case, we need to collapse identical constraints into one
        # so that we avoid linear dependencies.
        # In the medium-term I think we would be better off switching to a different charge polarization
        # model based on pair-parameters. Basically, there should be an electrostatic component to the
        # charge flux model which moves around charge based on the potential difference between pairs
        # of atoms. We can actually parameterize the pairwise model to reproduce the variational EEM
        # model.
        ngroups = natoms // 3
        dq_groups = torch.zeros(ngroups, device=pairs.device)
        dq_groups = dq_groups.scatter_add(0, group_indices, dq_a)

        # Get electrostatic energy, electric potential, field, and field gradients
        b_i_elec_p = b_elec[pairs_inter_i_a]
        b_j_elec_p = b_elec[pairs_inter_j_a]
        b_ij_elec_p = torch.sqrt(b_i_elec_p * b_j_elec_p)
        ene_perm_elec, elec_potential, elec_field, elec_field_grad = computePermanentElectricPotentialExpansionAndEnergyFromPairs(
            natoms,
            pairs_inter_i_a,
            pairs_inter_j_a,
            dists_inter, dist_vecs_inter,
            b_i_elec_p, b_ij_elec_p,
            multipoles, Z
        )

        # TODO: Need the topology to determine the polarization groups. I really hate this
        # polarization group concept. The polarization group, I suppose, is the mask
        # which specifies all intramolecular atoms (including that atom itself) for each
        # atom. The charge constraints are then enforced over those groups.
        #
        # The below is again a hack to work for water.

        # Solve Polarization Equations #
        groups = torch.stack((
            torch.arange(0, natoms, 3, device=pairs.device),
            torch.arange(1, natoms, 3, device=pairs.device),
            torch.arange(2, natoms, 3, device=pairs.device)), dim=1)

        b_vec = torch.hstack((-elec_potential, elec_field.flatten(), dq_groups)) # TODO: This should actually add in the "groupCharges" to dq_groups which are zero for water but nonzero for ions.
        with torch.no_grad():
            induced_multipoles_and_lagrange_muls = direct_field_induced_dipole_guess(natoms, natoms, ngroups, polarizabilities, elec_field)
            induced_multipoles_and_lagrange_muls_out = solvePolarizationByCG(
                induced_multipoles_and_lagrange_muls,
                b_vec,
                natoms,
                pairs_inter_i_a,
                pairs_inter_j_a,
                dists_inter, dist_vecs_inter,
                b_ij_elec_p,
                eta_times_2,
                inverse_polarizabilities,
                group_indices, groups
            )
        
        TM, elec_potential_induced, elec_field_induced  = computeProductWithPolarizationMatrix(
            induced_multipoles_and_lagrange_muls_out, natoms,
            pairs_inter_i_a, pairs_inter_j_a,
            dists_inter, dist_vecs_inter,
            b_ij_elec_p, eta_times_2, inverse_polarizabilities,
            group_indices, groups, True
        )
        ene_pol = torch.dot(induced_multipoles_and_lagrange_muls_out, (0.5 * TM - b_vec))

        # NOTE(JOE): There is a problem with the gradients here when induced
        # fields are included. Basically, the partial derivatives of the induced
        # multipoles with respect to the cartesian coordinates are needed for the
        # FD morse derivatives. Unfortunately, if gradient tracking is on when the
        # polarization equations are solved, then things become very slow and
        # use a lot of memory (but the gradients are right!). If we have gradient
        # tracking off then everything is much more efficient but the FD morse
        # gradients are wrong. So, we need to compute the field gradients
        # due to the induced multipoles and properly incorporate them into the
        # pytorch computational graph. This is possible, but I am going to
        # figure that out once we are in a better position to actually run MD.
        re_fd_p, beta_fd_p = computeFieldDependentMorseParams(
            dists[topology.bonded_pairs], dist_vecs[topology.bonded_pairs],
            k_b_p, D_p, r_eq, dip_deriv_1_p, dip_deriv_2_p,
            ct_slope_1_p, ct_slope_2_p,
            #(elec_field + elec_field_induced)[topology.bonded_atoms[1]],
            (elec_field)[topology.bonded_atoms[1]],
            dq_a[topology.bonded_atoms[1]]
        )

        # morse-bond
        ene_bond_list = computeMorseBondPotential(dists[topology.bonded_pairs], re_fd_p, D_p, beta_fd_p)
        ene_bonds = torch.sum(ene_bond_list)
        
        # bond-bond couplings
        ene_bbs_list = computeBondBondCoupling(
            dists[topology.angle_pairs[0]], dists[topology.angle_pairs[1]],
            r_eq_bb_1, r_eq_bb_2, k_bb
        )
        ene_bbs = torch.sum(ene_bbs_list)

        # angles
        ene_angles_list = computeCosAnglePotential(
            angles, theta_eq, k_theta
        )
        ene_angles = torch.sum(ene_angles_list)

        ## bond-angle couplings
        ene_bas_list = computeBondAngleCoupling(
            dists[pairs_angles_a], r_eq_ba,
            angles.repeat_interleave(2), theta_eq.repeat_interleave(2),
            k_ba
        )
        ene_bas = torch.sum(ene_bas_list)

        ## Pauli repulsion
        pauli_pairwise = computeShortRangeEnergyFromPairs(
            dists_inter, dist_vecs_inter,
            multipoles_pauli[pairs_inter_i_a], multipoles_pauli[pairs_inter_j_a],
            torch.sqrt(b_pauli[pairs_inter_i_a] * b_pauli[pairs_inter_j_a])
        )
        ene_pauli = torch.sum(pauli_pairwise) / 2

        # dispersion
        disp_pairwise = computeDispersionFromPairs(
            dists_inter,
            torch.sqrt(C6_disp[pairs_inter_i_a] * C6_disp[pairs_inter_j_a]),
            torch.sqrt(b_disp[pairs_inter_i_a] * b_disp[pairs_inter_j_a])
        )
        ene_disp = torch.sum(disp_pairwise) / 2

        # exchange-polarization
        xpol_pairwise = computeShortRangeEnergyFromPairs(
            dists_inter, dist_vecs_inter,
            multipoles_xpol[pairs_inter_i_a], multipoles_xpol[pairs_inter_j_a],
            torch.sqrt(b_xpol[pairs_inter_i_a] * b_xpol[pairs_inter_j_a]),
            False
        )
        ene_xpol = torch.sum(xpol_pairwise) / 2

        ene_tot = ene_perm_elec + ene_pol + ene_xpol + ene_pauli + ene_disp + ene_ct_direct + ene_bonds + ene_angles + ene_bas + ene_bbs
        energies = {
            "perm_elec": ene_perm_elec,
            "pol": ene_pol,
            "ct_direct": ene_ct_direct,
            "xpol": ene_xpol,
            "pauli": ene_pauli,
            "disp": ene_disp,
            "deformation": ene_bonds + ene_angles + ene_bbs + ene_bas,
            "bond": ene_bonds,
            "angle": ene_angles,
            "bond_bond": ene_bbs,
            "bond_angle": ene_bas,
            "tot": ene_tot
        }
        return energies