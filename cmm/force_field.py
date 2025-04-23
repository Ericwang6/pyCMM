import torch, math
from torch_scatter import segment_csr
from .multipole import computeCartesianQuadrupoles, convertMultipolesToPolytensor, rotateDipoles, rotateQuadrupoles, computeUndampedInteractionTensorBlocks, formDampingFactorBlocksRank1, formDampingFactorBlocksRank2
from .electrostatics import computeDampFactorsErfc, computeDampFactorsErf
from .ewald import long_range_potential, long_range_potential_rank_1
from .polarization import direct_field_induced_dipole_guess, get_field_dependent_polarizabilities, direct_polarization_guess, compute_product_with_polarization_matrix
from .short_range import scaleMultipoles, computeShortRangeOneCenterDampFactors, computeShortRangeTwoCenterDampFactors, computeShortRangePolarizationDampFactors
from .dispersion import computeDispersionFromPairs
from .coordinate_manager import CoordinateManager
from .axis_types import AxisTypes
from .parameters import Parameterizer
from .topology import Topology
from .terms import *
from .units import *
from .switching_functions import switch_543
from.polarization_solver import PolarizationSolver, cg_solve

from copy import copy

class ForceField(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._atomic_params = {}
        self._pair_params = {}

class CMM(ForceField):
    def __init__(self,
                 cutoff_short_range: torch.Tensor=torch.tensor(5.0 / BOHR2ANG, dtype=torch.float64),
                 cutoff_ewald: torch.Tensor=torch.tensor(9.0 / BOHR2ANG, dtype=torch.float64),
                 ewald_tolerance: torch.Tensor=torch.tensor(1e-6, dtype=torch.float64),
                 use_ewald: bool=False, use_polarization: bool=True,
                 pol_solver_type="conjugate_gradient", max_iterations=400,
                 solve_tolerance: torch.Tensor=torch.tensor(1e-7, dtype=torch.float64)) -> None:
        super().__init__()
        # These are local indices for the atom types, not the actual
        # atom type indices which are decided by the Parameterizer.
        self._types_to_index = {
            "O_water": 0, "H_water": 1,
            "F-": 2, "Cl-": 3, "Br-": 4, "I-": 5,
            "Li+": 6, "Na+": 7, "K+": 8, "Rb+": 9, "Cs+": 10,
            "Mg2+": 11, "Ca2+": 12
        }
        self.cutoff_sr = cutoff_short_range if torch.is_tensor(cutoff_short_range) else torch.tensor(cutoff_short_range, dtype=torch.float64)
        self.ewald_tolerance = ewald_tolerance if torch.is_tensor(ewald_tolerance) else torch.tensor(ewald_tolerance, dtype=torch.float64)
        self.cutoff_ewald = cutoff_ewald if torch.is_tensor(cutoff_ewald) else torch.tensor(cutoff_ewald, dtype=torch.float64)
        self.cutoff_vdw = cutoff_ewald if torch.is_tensor(cutoff_ewald) else torch.tensor(cutoff_ewald, dtype=torch.float64)
        self.use_ewald = use_ewald

        self.use_polarization = use_polarization
        self.polarization_solver = None
        if self.use_polarization:
            self.solver_type = pol_solver_type
            self.max_iterations = max_iterations
            self.solve_tolerance = solve_tolerance if torch.is_tensor(solve_tolerance) else torch.tensor(solve_tolerance, dtype=torch.float64)
            self.polarization_solver = PolarizationSolver(
                max_iter=self.max_iterations, tol=self.solve_tolerance, solver_type=self.solver_type
            )
        
        self.last_induced_multipoles = None
        self.last_permanent_multipoles = None
        
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

        # NOTE(JOE): The oxygen eta value should be exactly 0.0 by symmetry.
        # I am leaving it at this small value since that is what was used
        # when fitting the model. Changing it to 0.0 does not introduce any
        # problems or change energies/forces meaningfully since this is
        # actually the inverse hardness, rather than the hardness itself.
        # Once the code is more solid, we should change it to exactly 0.0.
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
            ("Li+", "F-"): {
                "b_pauli": torch.tensor([2.0662]),
                "b_elec": torch.tensor([3.29393]),
                "b_disp": torch.tensor([1.66054]),
                "b_xpol": torch.tensor([2.42899]),
                "b_ct": torch.tensor([1.68936]),
                "eps": torch.tensor([1.0 / 4.16245e6]),
            },
            ("Li+", "Cl-"): {
                "b_pauli": torch.tensor([1.81696]),
                "b_elec": torch.tensor([2.53683]),
                "b_disp": torch.tensor([1.50876]),
                "b_xpol": torch.tensor([21.7689]),
                "b_ct": torch.tensor([1.14099]),
                "eps": torch.tensor([1.0 / 0.622129]),
            },
            ("Li+", "Br-"): {
                "b_pauli": torch.tensor([1.78426]),
                "b_elec": torch.tensor([2.4508]),
                "b_disp": torch.tensor([1.42241]),
                "b_xpol": torch.tensor([3.43206]),
                "b_ct": torch.tensor([1.09604]),
                "eps": torch.tensor([1.0 / 0.556197]),
            },
            ("Li+", "I-"): {
                "b_pauli": torch.tensor([1.76695]),
                "b_elec": torch.tensor([2.34386]),
                "b_disp": torch.tensor([2.0152]),
                "b_xpol": torch.tensor([2.6963]),
                "b_ct": torch.tensor([0.911407]),
                "eps": torch.tensor([1.0 / 0.493998]),
            },
            ("Na+", "F-"): {
                "b_pauli": torch.tensor([2.23085]),
                "b_elec": torch.tensor([2.47042]),
                "b_disp": torch.tensor([1.52165]),
                "b_xpol": torch.tensor([24.3618]),
                "b_ct": torch.tensor([2.12257]),
                "eps": torch.tensor([1.0 / 5.59222e6]),
            },
            ("Na+", "Cl-"): {
                "b_pauli": torch.tensor([1.96851]),
                "b_elec": torch.tensor([2.09285]),
                "b_disp": torch.tensor([1.37519]),
                "b_xpol": torch.tensor([13.9896]),
                "b_ct": torch.tensor([0.981402]),
                "eps": torch.tensor([1.0 / 0.92137]),
            },
            ("Na+", "Br-"): {
                "b_pauli": torch.tensor([1.90723]),
                "b_elec": torch.tensor([2.06585]),
                "b_disp": torch.tensor([1.32684]),
                "b_xpol": torch.tensor([13.7141]),
                "b_ct": torch.tensor([0.948933]),
                "eps": torch.tensor([1.0 / 0.834803]),
            },
            ("Na+", "I-"): {
                "b_pauli": torch.tensor([1.87522]),
                "b_elec": torch.tensor([2.0484]),
                "b_disp": torch.tensor([1.63812]),
                "b_xpol": torch.tensor([13.5543]),
                "b_ct": torch.tensor([0.911423]),
                "eps": torch.tensor([1.0 / 0.634873]),
            },
            ("K+", "F-"): {
                "b_pauli": torch.tensor([2.0344]),
                "b_elec": torch.tensor([2.27313]),
                "b_disp": torch.tensor([1.39697]),
                "b_xpol": torch.tensor([2.77048]),
                "b_ct": torch.tensor([1.92339]),
                "eps": torch.tensor([1.0 / 1.30686e7]),
            },
            ("K+", "Cl-"): {
                "b_pauli": torch.tensor([1.79779]),
                "b_elec": torch.tensor([2.01972]),
                "b_disp": torch.tensor([1.28869]),
                "b_xpol": torch.tensor([13.5339]),
                "b_ct": torch.tensor([1.56964]),
                "eps": torch.tensor([1.0 / 1.41644]),
            },
            ("K+", "Br-"): {
                "b_pauli": torch.tensor([1.74843]),
                "b_elec": torch.tensor([1.99904]),
                "b_disp": torch.tensor([1.24628]),
                "b_xpol": torch.tensor([13.2437]),
                "b_ct": torch.tensor([1.47289]),
                "eps": torch.tensor([1.0 / 0.975592]),
            },
            ("K+", "I-"): {
                "b_pauli": torch.tensor([1.70111]),
                "b_elec": torch.tensor([1.9751]),
                "b_disp": torch.tensor([1.66612]),
                "b_xpol": torch.tensor([12.9938]),
                "b_ct": torch.tensor([1.34706]),
                "eps": torch.tensor([1.0 / 0.621612]),
            },
            ("Rb+", "F-"): {
                "b_pauli": torch.tensor([2.07856]),
                "b_elec": torch.tensor([2.289]),
                "b_disp": torch.tensor([0.989019]),
                "b_xpol": torch.tensor([15.3797]),
                "b_ct": torch.tensor([1.97104]),
                "eps": torch.tensor([1.0 / 4.07063]),
            },
            ("Rb+", "Cl-"): {
                "b_pauli": torch.tensor([1.79968]),
                "b_elec": torch.tensor([2.0295]),
                "b_disp": torch.tensor([1.58976]),
                "b_xpol": torch.tensor([13.8017]),
                "b_ct": torch.tensor([1.43009]),
                "eps": torch.tensor([1.0 / 0.997574]),
            },
            ("Rb+", "Br-"): {
                "b_pauli": torch.tensor([1.74034]),
                "b_elec": torch.tensor([1.9988]),
                "b_disp": torch.tensor([1.62445]),
                "b_xpol": torch.tensor([13.8468]),
                "b_ct": torch.tensor([1.34474]),
                "eps": torch.tensor([1.0 / 0.833069]),
            },
            ("Rb+", "I-"): {
                "b_pauli": torch.tensor([1.71241]),
                "b_elec": torch.tensor([1.99082]),
                "b_disp": torch.tensor([3.90536]),
                "b_xpol": torch.tensor([12.3616]),
                "b_ct": torch.tensor([1.29478]),
                "eps": torch.tensor([1.0 / 0.626737]),
            },
            ("Cs+", "F-"): {
                "b_pauli": torch.tensor([1.94985]),
                "b_elec": torch.tensor([2.05036]),
                "b_disp": torch.tensor([1.17374]),
                "b_xpol": torch.tensor([14.339]),
                "b_ct": torch.tensor([1.80732]),
                "eps": torch.tensor([1.0 / 1.10774]),
            },
            ("Cs+", "Cl-"): {
                "b_pauli": torch.tensor([1.70545]),
                "b_elec": torch.tensor([1.90651]),
                "b_disp": torch.tensor([1.49986]),
                "b_xpol": torch.tensor([12.9929]),
                "b_ct": torch.tensor([1.39542]),
                "eps": torch.tensor([1.0 / 0.819386]),
            },
            ("Cs+", "Br-"): {
                "b_pauli": torch.tensor([1.64889]),
                "b_elec": torch.tensor([1.89059]),
                "b_disp": torch.tensor([1.47483]),
                "b_xpol": torch.tensor([12.7337]),
                "b_ct": torch.tensor([1.33849]),
                "eps": torch.tensor([1.0 / 0.748004]),
            },
            ("Cs+", "I-"): {
                "b_pauli": torch.tensor([1.64008]),
                "b_elec": torch.tensor([1.90755]),
                "b_disp": torch.tensor([2.21299]),
                "b_xpol": torch.tensor([12.6147]),
                "b_ct": torch.tensor([1.31385]),
                "eps": torch.tensor([1.0 / 0.577828]),
            },
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

        self.combination_rules = {
            "b_pauli": torch.sqrt,
            "b_elec": torch.sqrt,
            "b_disp": torch.sqrt,
            "b_xpol": torch.sqrt,
            "b_ct": torch.sqrt,
            "C6_disp": torch.sqrt,
        }
    
        self.rebuild_atomic_params()
        self.parameters_have_changed = False

    def rebuild_atomic_params(self):
        self._raw_atomic_params = {
            # elec
            "Z": self.Z,
            "q_shell": self.mono - self.Z,
            "mono": self.mono,
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

    def get_dipole_moment(self, coords: torch.Tensor, include_induced_moments: bool = True):
        natoms = self.last_permanent_multipoles.size(0)
        dipole_moment = torch.matmul(coords.T, self.last_permanent_multipoles[:, 0])
        dipole_moment = dipole_moment + torch.sum(self.last_permanent_multipoles[:, 1:4], dim=0)
        if self.last_induced_multipoles is not None and include_induced_moments:
            dipole_moment = dipole_moment + torch.matmul(coords.T, self.last_induced_multipoles[0:natoms])
            dipole_moment = dipole_moment + torch.sum(self.last_induced_multipoles[natoms:4*natoms].view(-1, 3), dim=0)
        
        return dipole_moment

    #@torch.compile
    def evaluate(self, cm: CoordinateManager, topology: Topology, params: Parameterizer, reset_grads: bool=False):
        # Get all intermolecular and intramolecular pairs, dists, and vectors inside long-range cutoff #
        pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs(topology, reset_grads=reset_grads)
        self.cutoff_vdw = cm.cutoff-0.05

        if True: #self.parameters_have_changed:
            # SPEED: Can of course do this per parameter type so that not everything is rebuilt
            # each time this is called. Currently would SOMETIMES NOT WORK for pair params since
            # we symmetrize the pair parameters w.r.t. a specific choice of the atom types.
            # This would be a reason to add setter functions. In addition to a way to set the status bool.
            params.rebuild(pairs, self.atomic_params, self.pair_params, self.pair_pair_params, self.pair_angle_params, self.angle_params)
            self.parameters_have_changed = False
        
        # Get pairs, dists, and vectors for exclusion list (needed to remove their contribution from long-range interactions) #
        pairs_excl = pairs[topology.all_intramolecular_pairs, :]
        pairs_excl_i_a = pairs_excl[:, 0]
        pairs_excl_j_a = pairs_excl[:, 1]
        dists_excl = dists[topology.all_intramolecular_pairs]
        dist_vecs_excl = dist_vecs[topology.all_intramolecular_pairs]

        # Get pairs, dists, and vectors for vdw potential #
        pairs_vdw = pairs[topology.all_intermolecular_pairs, :]
        pairs_vdw_i_a = pairs_vdw[:, 0]
        pairs_vdw_j_a = pairs_vdw[:, 1]
        dists_vdw = dists[topology.all_intermolecular_pairs]
        dist_vecs_vdw = dist_vecs[topology.all_intermolecular_pairs]

        # Get switching function values for long-range nonbonded potential #
        switch_start_vdw = self.cutoff_vdw - 3.0
        switch_start_vdw = switch_start_vdw if switch_start_vdw > 0.0 else 0.0
        switch_vdw = switch_543(dists_vdw, switch_start_vdw, self.cutoff_vdw)

        # Get pairs, dists, and vectors for long-range nonbonded potential #
        indices_vdw_to_lr = torch.where(dists_vdw <= self.cutoff_ewald, torch.arange(dists_vdw.size(0), dtype=torch.long, device=dists_vdw.device), torch.tensor(-1, dtype=torch.long, device=dists_vdw.device))
        indices_vdw_to_lr = indices_vdw_to_lr[indices_vdw_to_lr >= 0]

        pairs_lr = pairs_vdw[indices_vdw_to_lr, :]
        pairs_lr_i_a = pairs_lr[:, 0]
        pairs_lr_j_a = pairs_lr[:, 1]
        dists_lr = dists_vdw[indices_vdw_to_lr]
        dist_vecs_lr = dist_vecs_vdw[indices_vdw_to_lr]

        # Get pairs, dists, and vectors for short-range nonbonded potential #
        indices_vdw_to_sr = torch.where(dists_vdw <= self.cutoff_sr, torch.arange(dists_vdw.size(0), dtype=torch.long, device=dists_vdw.device), torch.tensor(-1, dtype=torch.long, device=dists_vdw.device))
        indices_vdw_to_sr = indices_vdw_to_sr[indices_vdw_to_sr >= 0]
        all_intermolecular_pairs_sr = topology.all_intermolecular_pairs[indices_vdw_to_sr]

        pairs_sr = pairs_vdw[indices_vdw_to_sr, :]
        pairs_sr_i_a = pairs_sr[:, 0]
        pairs_sr_j_a = pairs_sr[:, 1]
        dists_sr = dists_vdw[indices_vdw_to_sr]
        dist_vecs_sr = dist_vecs_vdw[indices_vdw_to_sr]

        # Get switching function values for short-range nonbonded potential #
        switch_start_sr = self.cutoff_sr - 2.0
        switch_start_sr = switch_start_sr if switch_start_sr > 0.0 else 0.0
        switch_sr = switch_543(dists_sr, switch_start_sr, self.cutoff_sr)

        if topology.angle_pairs.numel() > 0:
            angles = computeAngleFromVecs(dist_vecs[topology.angle_pairs[0]], dist_vecs[topology.angle_pairs[1]])

        # All pairs forming an angle #
        pairs_angles_p = topology.angle_pairs.T.flatten()

        # Electric Multipoles #
        Z = params.get_atomic_parameters('Z')
        natoms = torch.tensor(Z.size(0), device=pairs.device)
        mono = params.get_atomic_parameters('mono')
        dipo = params.get_atomic_parameters('dipo')
        quad = params.get_atomic_parameters('quad')
        axis_types = params.get_atomic_parameters("axistypes")

        # Polarizability #
        eta = params.get_atomic_parameters("eta")
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

        # Electric Multipoles #
        rotation_matrices = cm.compute_rotation_matrices(topology.zatoms, topology.xatoms, topology.yatoms, axis_types)

        # Atomic widths #
        b_elec = params.get_atomic_parameters('b_elec')
        b_ij_cp_sr_p = params.get_pair_parameters_with_optional_combination_rule(
            'b_elec', all_intermolecular_pairs_sr, pairs_sr
        )
        b_ij_pauli_sr_p = params.get_pair_parameters_with_optional_combination_rule(
            'b_pauli', all_intermolecular_pairs_sr, pairs_sr
        )
        b_ij_xpol_sr_p = params.get_pair_parameters_with_optional_combination_rule(
            'b_xpol', all_intermolecular_pairs_sr, pairs_sr
        )
        b_ij_ct_sr_p = params.get_pair_parameters_with_optional_combination_rule(
            'b_ct', all_intermolecular_pairs_sr, pairs_sr
        )
        b_ij_disp_vdw_p = params.get_pair_parameters_with_optional_combination_rule(
            'b_disp', topology.all_intermolecular_pairs, pairs_vdw
        )
        C6_ij_disp_vdw_p = params.get_pair_parameters_with_optional_combination_rule(
            'C6_disp', topology.all_intermolecular_pairs, pairs_vdw
        )
        eps = params.get_pair_parameters_with_optional_combination_rule('eps', all_intermolecular_pairs_sr, pairs[all_intermolecular_pairs_sr])

        if topology.bonded_pairs.numel() > 0:
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
        if topology.angle_pairs.numel() > 0:
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
        
        # Find appropriate ewald parameters. This should really be done by the CM.
        if self.use_ewald:
            self.alpha_ewald = torch.sqrt(-torch.log10(2 * self.ewald_tolerance)) / self.cutoff_ewald
            self.k_max = 50
            for i in range(2, 50):
                error_estimate = (i * torch.sqrt(cm.box_lengths[0] * self.alpha_ewald) / 20.0) * torch.exp(-torch.pi * torch.pi * i * i / (cm.box_lengths[0] * self.alpha_ewald * cm.box_lengths[0] * self.alpha_ewald))
                if error_estimate < self.ewald_tolerance:
                    self.k_max = i
                    break
            erfc_damps = computeDampFactorsErfc(dists_lr, self.alpha_ewald) # direct space
            erf_damps = -computeDampFactorsErf(dists_excl, self.alpha_ewald)
            # ^^^ for removing excluded interactions that are implicitly included in long-range summation
            # The reciprocal space calculation uses an erf(alpha*r) damping so the above is -erf(alpha*r)
        else:
            # This is hard-coded to 5 since we always compute damping factors up to quad-quad interactions.
            # In the (distant) future, we should enable automatic detection of multipole rank and try
            # to dispatch batches to kernels which consider the smallest maximum rank allowable. In that
            # case, this 5 would not be hard-coded. The code is going to be so different at that point this
            # comment is hardly worth writing, but at least now you know why there is a 5 here.
            erfc_damps = torch.ones((5, dists_lr.size(0)))
            erf_damps = torch.zeros((5, dists_excl.size(0)))

        b_i_elec_p = b_elec[pairs_sr_i_a]
        b_j_elec_p = b_elec[pairs_sr_j_a]
        cp_damps_sr_1c_i = -computeShortRangeOneCenterDampFactors(dists_sr, b_i_elec_p)
        cp_damps_sr_1c_j = -computeShortRangeOneCenterDampFactors(dists_sr, b_j_elec_p)
        cp_damps_sr_2c = -computeShortRangeTwoCenterDampFactors(dists_sr, b_ij_cp_sr_p)
        pauli_damps_sr_2c = computeShortRangeTwoCenterDampFactors(dists_sr, b_ij_pauli_sr_p)
        xpol_damps_sr_2c = -computeShortRangeTwoCenterDampFactors(dists_sr, b_ij_xpol_sr_p)
        ct_damps_sr_2c = -computeShortRangeTwoCenterDampFactors(dists_sr, b_ij_ct_sr_p)
        pol_damps_sr_2c = -computeShortRangePolarizationDampFactors(dists_sr, b_ij_cp_sr_p)

        # Get all undamped and damped interactions needed for multipolar interactions #
        undamped_tensor_1_lr, undamped_tensor_2_lr, undamped_tensor_3_lr = computeUndampedInteractionTensorBlocks(dist_vecs_lr, dists_lr)
        undamped_tensor_1_sr, undamped_tensor_2_sr, undamped_tensor_3_sr = computeUndampedInteractionTensorBlocks(dist_vecs_sr, dists_sr)
        undamped_tensor_1_excl, undamped_tensor_2_excl, undamped_tensor_3_excl = computeUndampedInteractionTensorBlocks(dist_vecs_excl, dists_excl)
        undamped_tensor_1_pol_sr = undamped_tensor_1_sr[:, :4, :4]
        undamped_tensor_2_pol_sr = undamped_tensor_2_sr[:, :4, :4]

        ewald_damps_lr_1, ewald_damps_lr_2, ewald_damps_lr_3 = formDampingFactorBlocksRank2(erfc_damps)
        ewald_damps_excl_1, ewald_damps_excl_2, ewald_damps_excl_3 = formDampingFactorBlocksRank2(erf_damps)
        cp_damps_sr_1c_1_i, cp_damps_sr_1c_2_i, cp_damps_sr_1c_3_i = formDampingFactorBlocksRank2(cp_damps_sr_1c_i)
        cp_damps_sr_1c_1_j, cp_damps_sr_1c_2_j, cp_damps_sr_1c_3_j = formDampingFactorBlocksRank2(cp_damps_sr_1c_j)
        cp_damps_sr_2c_1, cp_damps_sr_2c_2, cp_damps_sr_2c_3 = formDampingFactorBlocksRank2(cp_damps_sr_2c)
        pauli_damps_sr_2c_1, pauli_damps_sr_2c_2, pauli_damps_sr_2c_3 = formDampingFactorBlocksRank2(pauli_damps_sr_2c)
        xpol_damps_sr_2c_1, xpol_damps_sr_2c_2, xpol_damps_sr_2c_3 = formDampingFactorBlocksRank2(xpol_damps_sr_2c)
        ct_damps_sr_2c_1, ct_damps_sr_2c_2, ct_damps_sr_2c_3 = formDampingFactorBlocksRank2(ct_damps_sr_2c)
        pol_damps_sr_2c_1, pol_damps_sr_2c_2 = formDampingFactorBlocksRank1(pol_damps_sr_2c)

        direct_field_tensor_lr = torch.mul(undamped_tensor_1_lr, ewald_damps_lr_1) + torch.mul(undamped_tensor_2_lr, ewald_damps_lr_2) + torch.mul(undamped_tensor_3_lr, ewald_damps_lr_3)
        direct_field_tensor_excl = torch.mul(undamped_tensor_1_excl, ewald_damps_excl_1) + torch.mul(undamped_tensor_2_excl, ewald_damps_excl_2) + torch.mul(undamped_tensor_3_excl, ewald_damps_excl_3)
        direct_field_tensor_rank_1_lr = direct_field_tensor_lr[:, :4, :4]
        direct_field_tensor_excl_rank_1 = direct_field_tensor_excl[:, :4, :4]
        # ^^^^ Gets just the entries needed for charges and dipoles (for polarization)
        
        cp_field_tensor_sr_i = torch.mul(undamped_tensor_1_sr, cp_damps_sr_1c_1_i) + torch.mul(undamped_tensor_2_sr, cp_damps_sr_1c_2_i) + torch.mul(undamped_tensor_3_sr, cp_damps_sr_1c_3_i)
        cp_field_tensor_sr_j = torch.mul(undamped_tensor_1_sr, cp_damps_sr_1c_1_j) + torch.mul(undamped_tensor_2_sr, cp_damps_sr_1c_2_j) + torch.mul(undamped_tensor_3_sr, cp_damps_sr_1c_3_j)
        cp_interaction_tensor_sr = torch.mul(undamped_tensor_1_sr, cp_damps_sr_2c_1) + torch.mul(undamped_tensor_2_sr, cp_damps_sr_2c_2) + torch.mul(undamped_tensor_3_sr, cp_damps_sr_2c_3)
        pauli_interaction_tensor_sr = torch.mul(undamped_tensor_1_sr, pauli_damps_sr_2c_1) + torch.mul(undamped_tensor_2_sr, pauli_damps_sr_2c_2) + torch.mul(undamped_tensor_3_sr, pauli_damps_sr_2c_3)
        xpol_interaction_tensor_sr = torch.mul(undamped_tensor_1_sr, xpol_damps_sr_2c_1) + torch.mul(undamped_tensor_2_sr, xpol_damps_sr_2c_2) + torch.mul(undamped_tensor_3_sr, xpol_damps_sr_2c_3)
        ct_interaction_tensor_sr = torch.mul(undamped_tensor_1_sr, ct_damps_sr_2c_1) + torch.mul(undamped_tensor_2_sr, ct_damps_sr_2c_2) + torch.mul(undamped_tensor_3_sr, ct_damps_sr_2c_3)
        pol_interaction_tensor_sr = torch.mul(undamped_tensor_1_pol_sr, pol_damps_sr_2c_1) + torch.mul(undamped_tensor_2_pol_sr, pol_damps_sr_2c_2)

        # Pauli charge flux #
        if topology.bonded_pairs.numel():
            evaluate_bond_charge_flux(pairs, dists, topology.bonded_pairs, q_pauli, r_eq, j_cf_pauli)

        # Electrostatic charge flux #
        if topology.angle_pairs.numel() > 0:
            evaluate_bond_and_angle_charge_flux(
                pairs, dists, angles,
                topology.bonded_pairs, topology.angle_pairs, topology.angle_atoms,
                mono, r_eq, theta_eq, j_cf, j_cf_angle,
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

        mono_lr = mono
        dipo_lr = rotateDipoles(dipo, rotation_matrices).squeeze(1)
        quad_lr = rotateQuadrupoles(quad, rotation_matrices)

        multipoles_real = convertMultipolesToPolytensor(
            mono_lr, dipo_lr, quad_lr
        )
        multipoles_cp = convertMultipolesToPolytensor(
            mono_lr - Z, dipo_lr, quad_lr
        )
        self.last_permanent_multipoles = multipoles_real.clone().detach().requires_grad_(False)

        # @SPEED: Z_mpoles is all zeros besides the charge. Can certainly avoid allocating the
        # multipolar array entries and thereby eliminate the multiplications by zero.
        # Happens in the short-range index space so will not be a particularly large optimization.
        Z_mpoles = torch.zeros_like(multipoles_real)
        Z_mpoles[:, 0] += Z
        multipoles_ct_acc = scaleMultipoles(multipoles_real, q_ct_acc, Kdipo_ct_acc, Kquad_ct_acc)
        multipoles_ct_don = scaleMultipoles(multipoles_real, q_ct_don, Kdipo_ct_don, Kquad_ct_don)
        multipoles_pauli = scaleMultipoles(multipoles_real, q_pauli, Kdipo_pauli, Kquad_pauli)
        multipoles_xpol = scaleMultipoles(multipoles_real, q_xpol, Kdipo_xpol, Kquad_xpol)

        # Distribute multipoles over appropriate interaction pairs #
        multipoles_ct_acc_i_p = multipoles_ct_acc[pairs_sr_i_a]
        multipoles_ct_acc_j_p = multipoles_ct_acc[pairs_sr_j_a]
        multipoles_ct_don_i_p = multipoles_ct_don[pairs_sr_i_a]
        multipoles_ct_don_j_p = multipoles_ct_don[pairs_sr_j_a]
        multipoles_pauli_i_p = multipoles_pauli[pairs_sr_i_a]
        multipoles_pauli_j_p = multipoles_pauli[pairs_sr_j_a]
        multipoles_xpol_i_p = multipoles_xpol[pairs_sr_i_a]
        multipoles_xpol_j_p = multipoles_xpol[pairs_sr_j_a]
        multipoles_cp_i_p = multipoles_cp[pairs_sr_i_a]
        multipoles_cp_j_p = multipoles_cp[pairs_sr_j_a]
        multipoles_real_i_p = multipoles_real[pairs_lr_i_a]
        multipoles_real_j_p = multipoles_real[pairs_lr_j_a]
        multipoles_excl_i_p = multipoles_real[pairs_excl_i_a]
        multipoles_excl_j_p = multipoles_real[pairs_excl_j_a]
        Z_mpoles_i_p = Z_mpoles[pairs_sr_i_a]
        Z_mpoles_j_p = Z_mpoles[pairs_sr_j_a]

        ### After this point, the final values of all geometry-dependent params have been established
        ### and we just evaluate the intermolecular energy functions. After polarization is done we have
        ### to get field-dependent parameters for bonding.

        if self.use_ewald:
            # Get reciprocal space and self contributions to field variables
            # and corresponding electrostatic interactions.
            ewald_potential, ewald_field, ewald_field_gradient = long_range_potential(cm.coords, mono_lr, dipo_lr, quad_lr, cm.box, self.alpha_ewald, self.k_max)
            ene_ewald = 0.5 * (
                torch.einsum("n,n->", mono_lr, ewald_potential) -
                torch.einsum("ni,ni->", dipo_lr, ewald_field) -
                torch.einsum("nij,nij->", quad_lr, ewald_field_gradient) / 3
            )

        # All multipolar interaction contributions #
        ct_pairwise_ij = torch.bmm(multipoles_ct_don_j_p.unsqueeze(1), torch.bmm(ct_interaction_tensor_sr, multipoles_ct_acc_i_p.unsqueeze(2))).flatten()
        ct_pairwise_ji = torch.bmm(multipoles_ct_acc_j_p.unsqueeze(1), torch.bmm(ct_interaction_tensor_sr, multipoles_ct_don_i_p.unsqueeze(2))).flatten()
        pauli_pairwise = torch.bmm(multipoles_pauli_j_p.unsqueeze(1), torch.bmm(pauli_interaction_tensor_sr, multipoles_pauli_i_p.unsqueeze(2))).flatten()
        xpol_pairwise = torch.bmm(multipoles_xpol_j_p.unsqueeze(1), torch.bmm(xpol_interaction_tensor_sr, multipoles_xpol_i_p.unsqueeze(2))).flatten()
        elec_ss_pairwise = torch.bmm(multipoles_cp_j_p.unsqueeze(1), torch.bmm(cp_interaction_tensor_sr, multipoles_cp_i_p.unsqueeze(2))).flatten()
        elec_cs_pairwise_ji = torch.bmm(multipoles_cp_j_p.unsqueeze(1), torch.bmm(cp_field_tensor_sr_j, Z_mpoles_i_p.unsqueeze(2))).flatten()
        
        # Get real space field data #
        edata_point_pairwise = torch.bmm(direct_field_tensor_lr, multipoles_real_i_p.unsqueeze(2))
        edata_cs_pairwise_ij = torch.bmm(cp_field_tensor_sr_i, multipoles_cp_i_p.unsqueeze(2))
        
        # Real Space Electrostatic Interactions #
        elec_point_pairwise = torch.bmm(multipoles_real_j_p.unsqueeze(1), edata_point_pairwise).flatten()
        elec_cs_pairwise_ij = torch.bmm(Z_mpoles_j_p.unsqueeze(1), edata_cs_pairwise_ij).flatten()

        # Accumulate the total potentials, fields, and field gradients.
        # One contribution is accumulated over all long-range pairs, while the other
        # accumulates just the penetration contribution.
        all_field_data = torch.zeros(natoms, 10, device=dists.device, dtype=dists.dtype, requires_grad=True)
        all_field_data = all_field_data.scatter_add(0, pairs_lr_j_a.unsqueeze(1).expand(-1, 10), edata_point_pairwise.squeeze(2))
        all_field_data = all_field_data.scatter_add(0, pairs_sr_j_a.unsqueeze(1).expand(-1, 10), edata_cs_pairwise_ij.squeeze(2))

        if self.use_ewald:
            edata_point_excl_pairwise = torch.bmm(direct_field_tensor_excl, multipoles_excl_i_p.unsqueeze(2))
            elec_point_excl_pairwise = torch.bmm(multipoles_excl_j_p.unsqueeze(1), edata_point_excl_pairwise).flatten()
            all_field_data = all_field_data.scatter_add(0, pairs_excl_j_a.unsqueeze(1).expand(-1, 10), edata_point_excl_pairwise.squeeze(2))

        all_field_data = all_field_data.mul(torch.tensor([1, -1, -1, -1, -1, -1, -1, -1, -1, -1], device=pairs.device).reshape(1, -1))
        elec_potential = all_field_data[:, 0]
        elec_field = all_field_data[:, 1:4]
        if self.use_ewald:
            elec_potential = elec_potential + ewald_potential
            elec_field = elec_field + ewald_field
        
        ene_ct_direct = 0.5 * torch.sum((ct_pairwise_ij + ct_pairwise_ji) * switch_sr)
        ene_pauli = 0.5 * torch.sum(pauli_pairwise * switch_sr)
        ene_xpol = 0.5 * torch.sum(xpol_pairwise * switch_sr)
        ene_perm_elec = 0.5 * (
            torch.sum(elec_point_pairwise) +
            torch.sum((elec_cs_pairwise_ij + elec_cs_pairwise_ji + elec_ss_pairwise) * switch_sr)
        )
        if self.use_ewald:
            ene_perm_elec = ene_perm_elec + 0.5 * torch.sum(elec_point_excl_pairwise)

        # Find total charges in each polarization group to use as constraints
        drInvDamp_ct = ct_interaction_tensor_sr[:, 0, 0].flatten()
        dq_forward = multipoles_ct_don_i_p[:, 0] * multipoles_ct_acc_j_p[:, 0] * drInvDamp_ct * eps
        dq_backward = multipoles_ct_acc_i_p[:, 0] * multipoles_ct_don_j_p[:, 0] * drInvDamp_ct * eps
        dq_pairwise = (dq_forward - dq_backward) * switch_sr
        dq_a = torch.zeros(natoms, device=pairs.device, requires_grad=True)
        dq_groups = torch.zeros(topology.n_pol_groups, device=pairs.device, requires_grad=True)
        dq_a = dq_a.scatter_add(0, pairs_sr_j_a, dq_pairwise)
        dq_groups = segment_csr(dq_a[topology.pol_group_indices_a], topology.pol_group_segment_indices, reduce='sum')

        polarizabilities = rotateQuadrupoles(alpha, rotation_matrices)
        polarizabilities = get_field_dependent_polarizabilities(polarizabilities, elec_field, alpha_damp_exponent, alpha_damp_max)
        inverse_polarizabilities = torch.linalg.inv(polarizabilities)

        long_range_induced_potential_function = None
        if self.use_ewald:
            long_range_induced_potential_function = lambda charges, dipoles : long_range_potential_rank_1(cm.coords, charges, dipoles, cm.box, self.alpha_ewald, self.k_max)
        
        # TODO: To accelerate convergence of the polarization calculation, try the following:
        # Implement local iterations using a 4 angstrom cutoff. Use that as the preconditioner
        # as described in https://pubs.acs.org/doi/10.1021/acs.jctc.3c00226
        # Plus, implement some of the other tricks there.

        def A_mm(x: torch.Tensor):
            return compute_product_with_polarization_matrix(
                x,
                natoms,
                pairs_lr_i_a, pairs_lr_j_a, pairs_sr_i_a, pairs_sr_j_a,
                pairs_excl_i_a, pairs_excl_j_a, direct_field_tensor_rank_1_lr,
                pol_interaction_tensor_sr, direct_field_tensor_excl_rank_1,
                eta_times_2, inverse_polarizabilities, topology.pol_group_indices_a,
                topology.pol_group_segment_indices, topology.pol_group_lengths_g,
                long_range_potential_function=long_range_induced_potential_function
            )

        def M_mm(x: torch.Tensor):
            return direct_polarization_guess(
                x, natoms, topology.n_pol_groups, polarizabilities
            )

        # TODO: Implement least-squares extrapolation for generating induced dipole guess.
        # Solve polarization equations by preconditioned conjugate gradient #
        ene_pol = torch.tensor(0.0)
        if self.use_polarization:
            b_vector = torch.hstack((-elec_potential, elec_field.flatten(), dq_groups))
            with torch.no_grad():
                # Evaluate the initial guess #
                if self.last_induced_multipoles is None:
                    self.last_induced_multipoles = direct_field_induced_dipole_guess(natoms, topology.n_pol_groups, polarizabilities, elec_field)
                self.last_induced_multipoles, info = cg_solve(
                    A_mm, b_vector,
                    X0=self.last_induced_multipoles, M_mm=M_mm,
                    atol=self.solve_tolerance, rtol=self.solve_tolerance
                )
                #print(f"Solved polarization in {info['niter']} iterations")
            ene_pol = torch.dot(self.last_induced_multipoles, (0.5 * A_mm(self.last_induced_multipoles) - b_vector))
            self.last_induced_multipoles = self.last_induced_multipoles.detach().clone()

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
        if topology.bonded_pairs.numel() > 0:
            re_fd_p, beta_fd_p = computeFieldDependentMorseParams(
                dists[topology.bonded_pairs], dist_vecs[topology.bonded_pairs],
                k_b_p, D_p, r_eq, dip_deriv_1_p, dip_deriv_2_p,
                ct_slope_1_p, ct_slope_2_p,
                #(elec_field + induced_field)[topology.bonded_atoms[1]],
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
        if topology.angle_atoms.numel() > 0:
            # angles
            ene_angles_list = computeCosAnglePotential(
                angles, theta_eq, k_theta
            )
            ene_angles = torch.sum(ene_angles_list)

            # bond-angle couplings
            ene_bas_list = computeBondAngleCoupling(
                dists[pairs_angles_p], r_eq_ba,
                angles.repeat_interleave(2), theta_eq.repeat_interleave(2),
                k_ba
            )
            ene_bas = torch.sum(ene_bas_list)

        # dispersion
        disp_pairwise = computeDispersionFromPairs(
            dists_vdw,
            C6_ij_disp_vdw_p, b_ij_disp_vdw_p,
            switch_vdw
        )
        ene_disp = torch.sum(disp_pairwise) / 2

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
            "total": ene_tot
        }

        if self.use_ewald:
            energies["ewald"] = ene_ewald
            energies["total"] = ene_tot + ene_ewald

        return energies