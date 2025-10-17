import torch, math
from .ff import FF
from ..terms.bonded import *
from ..terms.nonbonded import *
from ..terms.parameter import *

from ..system import System
from ..parameters import Parameterizer2
from ..units import HARTREE2KCAL, BOHR2ANG, HARTREE2KJ

import torch, math
from ..multipole import computeCartesianQuadrupoles
from ..legacy.axis_types import AxisTypes
from ..polarization_solver import cg_solve, CG

class CMM2(FF):
    def __init__(self, system: System, dtype: torch.dtype=torch.float64, device: torch.DeviceObjType=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), requires_param_grads: bool=False) -> None:
        super().__init__(system, dtype, device)
        # TODO: The force field should take in only the settings (not the system) and it should
        # fill out some default settings. Users can then override these settings after the force field
        # has been constructed? Details to be worked out.
        lr_elec_settings = system.settings.get_long_range_electrostatics_settings()
        lr_disp_settings = system.settings.get_long_range_dispersion_settings()
        pol_settings = system.settings.get('polarization')
        solver = CG(None, None,
            rtol=torch.tensor(pol_settings.tolerance, dtype=torch.float64),
            atol=0,
            maxiter=pol_settings.max_iterations,
            n_extrapolate_from=pol_settings.n_extrapolate_from,
            verbose=pol_settings.verbose
        )
        
        self.add_term(StoreIndices())
        self.add_term(StoreChargeFluxCMM())
        self.add_term(StoreMultipolesCMM())
        self.add_term(StoreInteractionTensorsCMM())
        self.add_term(StoreSwitchingValues())

        self.setup_long_range_interactions(system)
        self.add_term(VariableHardness())
        self.add_term(TTDispersionC6(lr_disp_settings.use_switching, lr_disp_settings.switching_start_before_cutoff))
        self.add_term(ExchangePolarizationCMM())
        self.add_term(MultipolarElectrostatics2())
        self.add_term(MultipolarChargePenetration())
        self.add_term(ExcludedMultipolarElectrostatics2())
        self.add_term(MultipolarPauli())
        self.add_term(MultipolarChargeTransfer())
        self.add_term(ManyBodyChargeTransfer())
        self.add_term(MultipolarPolarization1(solver, lr_elec_settings.alpha, lr_elec_settings.k_max))

        self.add_term(CosineAngle())
        self.add_term(FieldDependentMorseParams())
        self.add_term(MorseBond())
        self.add_term(BondedCouplingCMM())
        self._build()

    def forward(self, system: System):
        # TODO: Make a better API for filling out the parameter arrays and getting the params.
        # There should be a simple way to specify which indices are needed for each term and
        # which parameters. Those should then get filled in all in one call so that we can
        # basically call one setup function which fills in the parameter arrays and then have
        # a static run through the force field. Some parameters depend on the outcome of other
        # terms so with some force fields there have to be multiple stages to evaluation, but
        # we will cross that bridge when we get there.
        pairs, dists, distance_vecs = system.get_distances_vectors_and_pairs()
        system.parameterizer.update(pairs, self.atomic_params, self.pair_params, self.angle_params, self.pair_pair_params, self.pair_angle_params, angle_atoms=system.topology.angle_atoms)
        
        V_total = torch.tensor(0.0, device=self._device, dtype=self._dtype)
        for term in self.terms:
            output_dict = term.forward(pairs, dists, distance_vecs, system)
            for key in output_dict.keys():
                self.energies[key] = output_dict[key]
                V_total = V_total + output_dict[key]
        
        self.energies["V_total"] = V_total

    def _build(self):
        self._types_to_index = {
            "O_water": 0, "H_water": 1,
            "F-": 2, "Cl-": 3, "Br-": 4, "I-": 5,
            "Li+": 6, "Na+": 7, "K+": 8, "Rb+": 9, "Cs+": 10,
            "Mg2+": 11, "Ca2+": 12
        }

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

    def rebuild_atomic_params(self):
        self._raw_atomic_params = {
            # elec
            "Z": self.Z,
            "q_shell": self.mono - self.Z,
            "q": self.mono,
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