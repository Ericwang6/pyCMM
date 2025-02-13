import torch, math
from torch_scatter import segment_csr
from .multipole import computeCartesianQuadrupoles, rotateMultipoles, rotateQuadrupoles, computeLocal2GlobalRotationMatrix
from .electrostatics import computePermanentElectricPotentialExpansionAndEnergyFromPairs, computePermanentElectricPotentialExpansionAndEnergyFromPairsEwald
from .ewald import long_range_vectorized, self_interaction
from .polarization import direct_field_induced_dipole_guess, solvePolarizationByCG, computeProductWithPolarizationMatrix, get_field_dependent_polarizabilities
from .short_range import scaleMultipoles, computePairwiseChargeTransfer, computeShortRangeEnergyFromPairs
from .dispersion import computeDispersionFromPairs
from .coordinate_manager import CoordinateManager
from .axis_types import AxisTypes
from .parameters import Parameterizer
from .topology import Topology
from .terms import *
from .units import *
from .switching_functions import switch_543

from copy import copy

class ForceField(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._atomic_params = {}
        self._pair_params = {}

class CMM(ForceField):
    def __init__(self,
                 cutoff_short_range: torch.Tensor=torch.tensor(5.0 / BOHR2ANG),
                 cutoff_ewald: torch.Tensor=torch.tensor(10.0 / BOHR2ANG),
                 ewald_tolerance: torch.Tensor=torch.tensor(1e-6),
                 use_ewald: bool=False) -> None:
        super().__init__()
        # These are local indices for the atom types, not the actual
        # atom type indices which are decided by the Parameterizer.
        self._types_to_index = {
            "O_water": 0, "H_water": 1,
            "F-": 2, "Cl-": 3, "Br-": 4, "I-": 5,
            "Li+": 6, "Na+": 7, "K+": 8, "Rb+": 9, "Cs+": 10,
            "Mg2+": 11, "Ca2+": 12
        }
        self.cutoff_sr = cutoff_short_range
        self.use_ewald = use_ewald
        self.cutoff_ewald = cutoff_ewald
        self.ewald_tolerance = ewald_tolerance

        self._build()
    
    def _build(self):
        # NOTE: The purpose of this function is to produce the atomic_params, bond_params,
        # angle_params, and pair_params dictionaries. These are what will ultimately be
        # used by the parameterizer to produce the full-sized parameter
        # arrays which are used when evaluating the force field.
        # Everything else used in this function is just for convenience
        # when building atomic_params, bond_params, and pair_params.

        # Electrostatic raw params #
        self.Z = torch.tensor([
            3.61565, 0.93619, # Water
            4.693, 12.1239, 18.9726, 35.5833, # Halides
            -0.895467, 3.5489, 7.73324, 12.2026, 11.5038, # Alkali
            2.83412, 4.93631 # Mg2+, Ca2+
        ])
        self.mono = torch.tensor([
            -0.390896, 0.195448, # Water
            -1.0, -1.0, -1.0, -1.0, # Halides
            1.0, 1.0, 1.0, 1.0, 1.0, # Alkali
            2.0, 2.0 # Mg2+, Ca2+
        ])
        self.dipo = torch.tensor([
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
        self.quad_s = torch.tensor([
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
        
        self.b_elec = torch.tensor([
            2.13358, 2.33322, # Water
            2.42894, 1.77558, 1.73844, 1.70583, # Halides
            4.44984, 2.59626, 2.39879, 2.38187, 2.03392, # Alkali
            1.92445, 2.11191, # Mg2+, Ca2+
        ])

        self.b_disp = torch.tensor([
            1.84302, 1.30993, # Water
            1.21488, 1.07019, 0.978881, 1.30013, # Halides
            2.23422, 1.99839, 1.95926, 4.01118, 4.01118, # Alkali
            1.60887, 1.63789, # Mg2+, Ca2+
        ])

        self.b_ct = torch.tensor([
            1.89485, 2.36763, # Water
            1.39081, 0.96508, 0.897324, 0.865887, # Halides
            1.69562, 1.876471, 2.0527, 2.04252, 1.97119, # Alkali
            1.5,     1.6, # Mg2+, Ca2+
        ])

        self.C6_disp = torch.tensor([
            35.8289, 1.98954, # Water
            146.12, 661.859, 1115.92, 1358.97, # Halides
            0.609382, 5.4421, 45.6395, 63.085, 170.628, # Alkali
            3.70387, 26.5808, # Mg2+, Ca2+
        ])

        self.b_pauli = torch.tensor([
            2.1975, 1.96474, # Water
            1.6851, 1.39256, 1.33717, 1.30314, # Halides
            2.82412, 2.9209, 2.38994, 2.34565, 2.06684, # Alkali
            2.75822, 2.21105, # Mg2+, Ca2+
        ])

        self.q_pauli = torch.tensor([
            6.50923, 0.527804, # Water
            3.61413, 5.22659, 6.36105, 9.36718, # Halides
            1.91402, 7.34252, 13.6855, 22.0153, 24.1029, # Alkali
            4.63567, 7.89129, # Mg2+, Ca2+
        ])

        self.Kdipo_pauli = torch.tensor([
            -5.61925, -0.515584, # Water
            0.0, 0.0, 0.0, 0.0, # Halides
            0.0, 0.0, 0.0, 0.0, 0.0, # Alkali
            0.0, 0.0, # Mg2+, Ca2+
        ])

        self.Kquad_pauli = torch.tensor([
            -1.56567, -0.440164, # Water
            0.0, 0.0, 0.0, 0.0, # Halides
            0.0, 0.0, 0.0, 0.0, 0.0, # Alkali
            0.0, 0.0, # Mg2+, Ca2+
        ])

        self.q_ct_acc = torch.tensor([
            -0.67857, 1.36735, # Water
            0.271625, -1.49937, -1.65442, -1.23716, # Halides
            1.01809, 1.07641, 7.67781, 13.5199, 27.9703, # Alkali
            3.5885, 7.30099 # Mg2+, Ca2+
        ])

        self.Kdipo_ct_acc = torch.tensor([
            0.0, 0.0, # Water
            0.0, 0.0, 0.0, 0.0, # Halides
            0.0, 0.0, 0.0, 0.0, 0.0, # Alkali
            0.0, 0.0, # Mg2+, Ca2+
        ])

        self.Kquad_ct_acc = torch.tensor([
            0.0, 0.0, # Water
            0.0, 0.0, 0.0, 0.0, # Halides
            0.0, 0.0, 0.0, 0.0, 0.0, # Alkali
            0.0, 0.0, # Mg2+, Ca2+
        ])

        self.q_ct_don = torch.tensor([
            0.757752, 0.00888982, # Water
            0.601589, 0.990161, 1.13917, 1.50009, # Halides
            -0.12094, 0.167905, 0.670336, 1.89103, 3.63343, # Alkali
            -0.499793, 0.655079, # Mg2+, Ca2+
        ])

        self.Kdipo_ct_don = torch.tensor([
            -0.512036, -0.0511668, # Water
            0.0, 0.0, 0.0, 0.0, # Halides
            0.0, 0.0, 0.0, 0.0, 0.0, # Alkali
            0.0, 0.0, # Mg2+, Ca2+
        ])

        self.Kquad_ct_don = torch.tensor([
            -0.208186, 0.0568152, # Water
            0.0, 0.0, 0.0, 0.0, # Halides
            0.0, 0.0, 0.0, 0.0, 0.0, # Alkali
            0.0, 0.0, # Mg2+, Ca2+
        ])

        self.b_xpol = torch.tensor([
            2.73582, 2.04028, # Water
            1.90554, 1.60669, 1.4814, 1.38744, # Halides
            2.6441, 2.54145, 2.2465, 2.27644, 2.0059, # Alkali
            5.14456, 3.67375, # Mg2+, Ca2+
        ])

        self.q_xpol = torch.tensor([
            1.26592, 0.200089, # Water
            -0.0914759, -1.11363, -1.46248, -2.28003, # Halides
            -4.68943, -5.33155, -3.61961, -3.15899, 5.43047, # Alkali
            -445.336, -220.273, # Mg2+, Ca2+
        ])

        self.eta = torch.tensor([
            6.18699e-6, 0.561535, # Water
            0.0, 0.0, 0.0, 0.0, # Halides
            0.0, 0.0, 0.0, 0.0, 0.0, # Alkali
            0.0, 0.0, # Mg2+, Ca2+
        ])

        notype = AxisTypes.NoAxisType.value
        self.axistypes = torch.tensor([
            AxisTypes.Bisector.value, AxisTypes.ZThenX.value, # Water
            notype, notype, notype, notype, # Halide
            notype, notype, notype, notype, notype, # Alkali
            notype, notype, # Divalent cations
        ])

        self.alpha = torch.stack((
            torch.diag(torch.tensor([4.45992, 6.07259, 4.55391])), # O_water
            torch.diag(torch.tensor([2.22001, 1.66835, 0.183855])), # H_water
            torch.diag(torch.tensor([11.7270176, 11.7270176, 11.7270176])), # F-
            torch.diag(torch.tensor([32.2880907, 32.2880907, 32.2880907])), # Cl-
            torch.diag(torch.tensor([42.7172275, 42.7172275, 42.7172275])), # Br-
            torch.diag(torch.tensor([64.1111144, 64.1111144, 64.1111144])), # I-
            torch.diag(torch.tensor([0.1586152, 0.1586152, 0.1586152])), # Li+
            torch.diag(torch.tensor([0.9542199, 0.9542199, 0.9542199])), # Na+
            torch.diag(torch.tensor([5.5376271, 5.5376271, 5.5376271])), # K+
            torch.diag(torch.tensor([8.6857518, 8.6857518, 8.6857518])), # Rb+
            torch.diag(torch.tensor([15.7177865, 15.7177865, 15.7177865])), # Cs+
            torch.diag(torch.tensor([0.4822524, 0.4822524, 0.4822524])), # Mg2+
            torch.diag(torch.tensor([3.2809409, 3.2809409, 3.2809409])), # Ca2+
        ))

        self.alpha_damp_exponent = torch.tensor([
            0.0, 0.0, # Water
            241.724, 428.717, 484.249, 599.029, # Halides
            0.0, 0.0, 0.0, 0.0, 0.0, # Alkali
            0.0, 0.0, # Mg2+, Ca2+
        ])

        self.alpha_damp_max = torch.tensor([
            0.0, 0.0, # Water
            0.75, 0.75, 0.75, 0.75, # Halides
            0.0, 0.0, 0.0, 0.0, 0.0, # Alkali
            0.0, 0.0, # Mg2+, Ca2+
        ])

        # TODO: Add all the ion-ion pair-specific parameters!

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
            ("Na+", "Cl-"): {"eps": torch.tensor([1.0 / 1e15]),}, # PLACEHOLDER VALUE FOR TESTING!!
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
    
        self.rebuild_atomic_params()
        self.parameters_have_changed = False

    def rebuild_atomic_params(self):
        self._raw_atomic_params = {
            # elec
            "Z": self.Z,
            "q_shell": self.mono - self.Z,
            "dipo": self.dipo,
            "quad": computeCartesianQuadrupoles(self.quad_s),
            "b_elec": self.b_elec,
            # Pauli repulsion
            "b_pauli": self.b_pauli,
            "q_pauli": self.q_pauli,
            "Kdipo_pauli": self.Kdipo_pauli,
            "Kquad_pauli": self.Kquad_pauli,
            # Dispersion
            "C6_disp": self.C6_disp,
            "b_disp": self.b_disp,
            # Polarization
            "alpha": self.alpha,
            "alpha_damp_exponent": self.alpha_damp_exponent,
            "alpha_damp_max": self.alpha_damp_max,
            "eta": self.eta,
            # Exchange-polarization
            "b_xpol": self.b_xpol,
            "q_xpol": self.q_xpol,
            "Kdipo_xpol": torch.zeros((len(self._types_to_index),)),
            "Kquad_xpol": torch.zeros((len(self._types_to_index),)),
            # Charge Transfer
            "b_ct": self.b_ct,
            "q_ct_acc": self.q_ct_acc,
            "Kdipo_ct_acc": self.Kdipo_ct_acc,
            "Kquad_ct_acc": self.Kquad_ct_acc,
            "q_ct_don": self.q_ct_don,
            "Kdipo_ct_don": self.Kdipo_ct_don,
            "Kquad_ct_don": self.Kquad_ct_don,
            "axistypes": self.axistypes,
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
        
            self.parameters_have_changed = True

    #@torch.compile
    def evaluate(self, cm: CoordinateManager, topology: Topology, params: Parameterizer):
        # Get all intermolecular and intramolecular pairs, dists, and vectors inside long-range cutoff #
        pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs(reset_gradients=True)
        
        if self.parameters_have_changed:
            # SPEED: Can of course do this per parameter type so that not everything is rebuilt
            # each time this is called. Currently would SOMETIMES NOT WORK for pair params since
            # we symmetrize the pair parameters w.r.t. a specific choice of the atom types.
            # This would be a reason to add setter functions. In addition to a way to set the status bool.
            params.rebuild(self.atomic_params, self.pair_params, self.pair_pair_params, self.pair_angle_params, self.angle_params)
            self.parameters_have_changed = False
        
        # Get pairs, dists, and vectors for long-range nonbonded potential #
        pairs_lr = pairs[topology.all_intermolecular_pairs, :]
        pairs_lr_i_a = pairs_lr[:, 0]
        pairs_lr_j_a = pairs_lr[:, 1]
        dists_lr = dists[topology.all_intermolecular_pairs]
        dist_vecs_lr = dist_vecs[topology.all_intermolecular_pairs]

        # Get switching function values for long-range nonbonded potential #
        cutoff_lr = cm.cutoff
        switch_start_lr = cutoff_lr - 2.0
        switch_start_lr = switch_start_lr if switch_start_lr > 0.0 else 0.0
        switch_lr = switch_543(dists_lr, switch_start_lr, cutoff_lr)

        # Get pairs, dists, and vectors for short-range nonbonded potential #
        indices_lr_to_sr = torch.where(dists_lr <= self.cutoff_sr, torch.arange(dists_lr.size(0), dtype=torch.long), torch.tensor(-1, dtype=torch.long))
        indices_lr_to_sr = indices_lr_to_sr[indices_lr_to_sr >= 0]
        all_intermolecular_pairs_sr = topology.all_intermolecular_pairs[indices_lr_to_sr]

        pairs_sr = pairs_lr[indices_lr_to_sr, :]
        pairs_sr_i_a = pairs_sr[:, 0]
        pairs_sr_j_a = pairs_sr[:, 1]
        dists_sr = dists_lr[indices_lr_to_sr]
        dist_vecs_sr = dist_vecs_lr[indices_lr_to_sr]

        # Get switching function values for short-range nonbonded potential #
        switch_start_sr = self.cutoff_sr - 2.0
        switch_start_sr = switch_start_sr if switch_start_sr > 0.0 else 0.0
        switch_sr = switch_543(dists_sr, switch_start_sr, self.cutoff_sr)

        if topology.angle_pairs.size(0) > 0:
            angles = computeAngleFromVecs(dist_vecs[topology.angle_pairs[0]], dist_vecs[topology.angle_pairs[1]])

        # All pairs forming an angle #
        pairs_angles_p = topology.angle_pairs.T.flatten()

        # Electric Multipoles #
        Z = params.get_atomic_parameters('Z')
        natoms = torch.tensor(Z.size(0), device=pairs.device)
        q_shell = params.get_atomic_parameters('q_shell')
        dipo = params.get_atomic_parameters('dipo')
        quad = params.get_atomic_parameters('quad')
        axis_types = params.get_atomic_parameters("axistypes")

        # Polarizability #
        alpha = params.get_atomic_parameters("alpha")
        alpha_damp_exponent = params.get_atomic_parameters("alpha_damp_exponent")
        alpha_damp_max = params.get_atomic_parameters("alpha_damp_max")

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

        eps = params.get_pair_parameters('eps', all_intermolecular_pairs_sr)

        if topology.bonded_pairs.size(0) > 0:
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
        if topology.angle_pairs.size(0) > 0:
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

        # Rotation Matrices #
        rotation_matrices = cm.compute_rotation_matrices(topology.zatoms, topology.xatoms, topology.yatoms, axis_types)
        if self.use_ewald:
            # Find appropraiate ewald parameters. This should really be done by the CM.
            if self.use_ewald:
                self.alpha_ewald = torch.sqrt(-torch.log10(2 * self.ewald_tolerance)) / self.cutoff_ewald
                self.k_max = 50
                for i in range(2, 50):
                    error_estimate = (i * torch.sqrt(cm.box_lengths[0] * self.alpha_ewald) / 20.0) * torch.exp(-torch.pi * torch.pi * i * i / (cm.box_lengths[0] * self.alpha_ewald * cm.box_lengths[0] * self.alpha_ewald))
                    if error_estimate < self.ewald_tolerance:
                        self.k_max = i
                        break

            monopoles = (q_shell + Z).detach().clone()
            dipo_2 = dipo.detach().clone()
            quad_2 = quad.detach().clone()
            multipoles_2 = rotateMultipoles(
                monopoles, dipo_2, quad_2, rotation_matrices
            ) * torch.tensor([1, 1, 1, 1, 1/3, 2/3, 2/3, 1/3, 2/3, 1/3], device=pairs.device)

            ene_ewald_direct, elec_potential, elec_field, elec_field_grad = computePermanentElectricPotentialExpansionAndEnergyFromPairsEwald(
                natoms,
                pairs_lr_i_a,
                pairs_lr_j_a,
                dists_lr, dist_vecs_lr,
                self.alpha_ewald, multipoles_2
            )
            ene_ewald_long_range = long_range_vectorized(cm.coords, monopoles, dipo_2, quad_2, cm.box, self.alpha_ewald, self.k_max)
            ene_ewald_self = self_interaction(cm.coords, monopoles, dipo_2, quad_2, self.alpha_ewald)
            ene_ewald = ene_ewald_direct + ene_ewald_long_range + ene_ewald_self

        # Pauli charge flux #
        if topology.bonded_pairs.size(0):
            evaluate_bond_charge_flux(pairs, dists, topology.bonded_pairs, q_pauli, r_eq, j_cf_pauli)

        # Electrostatic charge flux #
        if topology.angle_pairs.size(0) > 0:
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
        
        multipoles = rotateMultipoles(
            q_shell, dipo, quad, rotation_matrices
        ) * torch.tensor([1, 1, 1, 1, 1/3, 2/3, 2/3, 1/3, 2/3, 1/3], device=pairs.device)

        multipoles_ct_acc = scaleMultipoles(multipoles, q_ct_acc, Kdipo_ct_acc, Kquad_ct_acc)
        multipoles_ct_don = scaleMultipoles(multipoles, q_ct_don, Kdipo_ct_don, Kquad_ct_don)
        multipoles_pauli = scaleMultipoles(multipoles, q_pauli, Kdipo_pauli, Kquad_pauli)
        multipoles_xpol = scaleMultipoles(multipoles, q_xpol, Kdipo_xpol, Kquad_xpol)

        ct_direct_pairwise, dq_pairwise = computePairwiseChargeTransfer(
            dist_vecs_sr,
            multipoles_ct_acc[pairs_sr_i_a], multipoles_ct_acc[pairs_sr_j_a],
            multipoles_ct_don[pairs_sr_i_a], multipoles_ct_don[pairs_sr_j_a],
            b_ct[pairs_sr_i_a], b_ct[pairs_sr_j_a],
            eps, switch_sr
        )
        ene_ct_direct = torch.sum(ct_direct_pairwise) / 2
        
        # Find total charges in each polarization group to use as constraints
        dq_a = torch.zeros(natoms, device=pairs.device)
        dq_groups = torch.zeros(topology.n_pol_groups, device=pairs.device)
        dq_a = dq_a.scatter_add(0, pairs_sr_j_a, dq_pairwise)
        dq_groups = segment_csr(dq_a[topology.pol_group_indices_a], topology.pol_group_segment_indices, reduce='sum')

        # Get electrostatic energy, electric potential, field, and field gradients
        b_i_elec_p = b_elec[pairs_lr_i_a]
        b_j_elec_p = b_elec[pairs_lr_j_a]
        b_ij_elec_p = torch.sqrt(b_i_elec_p * b_j_elec_p)
        ene_perm_elec, elec_potential, elec_field, elec_field_grad = computePermanentElectricPotentialExpansionAndEnergyFromPairs(
            natoms,
            pairs_lr_i_a,
            pairs_lr_j_a,
            dists_lr, dist_vecs_lr,
            b_i_elec_p, b_ij_elec_p,
            multipoles, Z
        )

        polarizabilities = rotateQuadrupoles(alpha, rotation_matrices)
        polarizabilities = get_field_dependent_polarizabilities(polarizabilities, elec_field, alpha_damp_exponent, alpha_damp_max)
        inverse_polarizabilities = torch.linalg.inv(polarizabilities)

        #b_vec = torch.hstack((-elec_potential, elec_field.flatten(), dq_groups))
        #with torch.no_grad():
        #    guess_solution = direct_field_induced_dipole_guess(natoms, natoms, topology.n_pol_groups, polarizabilities, elec_field)
        #
        #    induced_multipoles_and_lagrange_muls_out = solvePolarizationByCG(
        #        guess_solution,
        #        b_vec,
        #        natoms,
        #        pairs_lr_i_a,
        #        pairs_lr_j_a,
        #        dists_lr, dist_vecs_lr,
        #        b_ij_elec_p,
        #        eta_times_2,
        #        inverse_polarizabilities,
        #        topology.pol_group_indices_a,
        #        topology.pol_group_segment_indices,
        #        topology.pol_group_lengths_g
        #    )
        #
        #TM, elec_potential_induced, elec_field_induced  = computeProductWithPolarizationMatrix(
        #    induced_multipoles_and_lagrange_muls_out, natoms,
        #    pairs_lr_i_a, pairs_lr_j_a,
        #    dists_lr, dist_vecs_lr,
        #    b_ij_elec_p, eta_times_2, inverse_polarizabilities,
        #    topology.pol_group_indices_a,
        #    topology.pol_group_segment_indices,
        #    topology.pol_group_lengths_g
        #)
        #ene_pol = torch.dot(induced_multipoles_and_lagrange_muls_out, (0.5 * TM - b_vec))
        ene_pol = torch.tensor(0.0)

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
        ene_bonds = torch.zeros(1, dtype=dists.dtype, device=dists.device)
        ene_bbs = torch.zeros(1, dtype=dists.dtype, device=dists.device)
        if topology.bonded_pairs.size(0) > 0:
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

        ene_angles = torch.zeros(1, dtype=dists.dtype, device=dists.device)
        ene_bas = torch.zeros(1, dtype=dists.dtype, device=dists.device)
        if topology.angle_atoms.size(0) > 0:
            # angles
            ene_angles_list = computeCosAnglePotential(
                angles, theta_eq, k_theta
            )
            ene_angles = torch.sum(ene_angles_list)

            ## bond-angle couplings
            ene_bas_list = computeBondAngleCoupling(
                dists[pairs_angles_p], r_eq_ba,
                angles.repeat_interleave(2), theta_eq.repeat_interleave(2),
                k_ba
            )
            ene_bas = torch.sum(ene_bas_list)

        ## Pauli repulsion
        pauli_pairwise = computeShortRangeEnergyFromPairs(
            dists_sr, dist_vecs_sr,
            multipoles_pauli[pairs_sr_i_a], multipoles_pauli[pairs_sr_j_a],
            torch.sqrt(b_pauli[pairs_sr_i_a] * b_pauli[pairs_sr_j_a]),
            switch_sr
        )
        ene_pauli = torch.sum(pauli_pairwise) / 2

        # dispersion
        disp_pairwise = computeDispersionFromPairs(
            dists_lr,
            torch.sqrt(C6_disp[pairs_lr_i_a] * C6_disp[pairs_lr_j_a]),
            torch.sqrt(b_disp[pairs_lr_i_a] * b_disp[pairs_lr_j_a]),
            switch_lr
        )
        ene_disp = torch.sum(disp_pairwise) / 2

        # exchange-polarization
        xpol_pairwise = computeShortRangeEnergyFromPairs(
            dists_sr, dist_vecs_sr,
            multipoles_xpol[pairs_sr_i_a], multipoles_xpol[pairs_sr_j_a],
            torch.sqrt(b_xpol[pairs_sr_i_a] * b_xpol[pairs_sr_j_a]),
            switch_sr, False
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

        if self.use_ewald:
            energies["ewald_direct"] = ene_ewald_direct
            energies["ewald_long_range"] = ene_ewald_long_range
            energies["ewald_self"] = ene_ewald_self
            energies["ewald"] = ene_ewald
            energies["tot"] = ene_tot + ene_ewald

        return energies