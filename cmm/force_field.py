import torch, math
from .multipole import computeCartesianQuadrupoles, rotateMultipoles, rotateQuadrupoles
from .short_range import scaleMultipoles
from .coordinate_manager import CoordinateManager
from .parameters import Parameterizer
from .topology import Topology
from .terms import *
from .units import *

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

class ForceField:
    def __init__(self) -> None:
        self._atomic_params = {}
        self._pair_params = {}

class CMM(ForceField):
    def __init__(self) -> None:
        super().__init__()
        # These are local indices for the atom types, not the actual
        # atom type indices which are decided by the Parameterizer.
        self._types_to_index = {
            "O_water": 0,
            "H_water": 1,
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
        Z = torch.tensor([3.61565, 0.93619], requires_grad=False)
        mono = torch.tensor([-0.390896, 0.195448], requires_grad=False)
        dipo = torch.tensor([
            [0.0,       0.0, -0.094298], # O_water
            [0.0910288, 0.0, -0.207851]  # H_water
        ])
        quad_s = torch.tensor([
            # Q20,       Q21c,      Q21s, Q22c,       Q22s
            [-0.330685,  0.0,       0.0,  0.869923,   0.0], # O_water
            [-0.0739388, 0.0929482, 0.0,  0.00532425, 0.0]  # H_water
        ])
        
        self._raw_atomic_params = {
            # elec
            "Z": Z,
            "q_shell": mono - Z,
            "dipo": dipo,
            "quad": computeCartesianQuadrupoles(quad_s),
            "b": torch.tensor([2.13358, 2.33322]),
            # Pauli repulsion
            "b_pauli": torch.tensor([2.1975, 1.96474]),
            "q_pauli": torch.tensor([6.50923, 0.527804]),
            "Kdipo_pauli": torch.tensor([-5.61925, -0.515584]),
            "Kquad_pauli": torch.tensor([-1.56567, -0.440164]),
            # Dispersion
            "C6_disp": torch.tensor([35.8289, 1.98954]),
            "b_disp": torch.tensor([1.84302, 1.30993]),
            # Polarization
            "alpha": torch.tensor([
                [[4.45992, 0.0, 0.0], [0.0, 6.07259, 0.0], [0.0, 0.0, 4.55391]],
                [[2.22001, 0.0, 0.0], [0.0, 1.66835, 0.0], [0.0, 0.0, 0.183855]]
            ]),
            "eta": torch.tensor([6.18699e-6, 0.561535]),
            # Exchange-polarization
            "b_xpol": torch.tensor([2.73582, 2.04028]),
            "q_xpol": torch.tensor([1.26592, 0.200089]),
            "Kdipo_xpol": torch.zeros((2,)),
            "Kquad_xpol": torch.zeros((2,)),
            # Charge Transfer
            "b_ct": torch.tensor([1.89485, 2.36763]),
            "q_ct_acc": torch.tensor([-0.67857, 1.36735]),
            "Kdipo_ct_acc": torch.tensor([0.0, 0.0]),
            "Kquad_ct_acc": torch.tensor([0.0, 0.0]),
            "q_ct_don": torch.tensor([0.757752, 0.00888982]),
            "Kdipo_ct_don": torch.tensor([-0.512036, -0.0511668]),
            "Kquad_ct_don": torch.tensor([-0.208186, 0.0568152]),
            "axistypes": torch.tensor([2, 1])
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
                "eps":torch.tensor([1.0 / 0.380979]),
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

    def evaluate(self, cm: CoordinateManager, topology: Topology, params: Parameterizer):
        # TODO: Now do the evaluation of the distances, vectors, and stuff
        # which should internally update the neighbor list as needed.
        # Also pull out the topological indices to be used for evaluating the FF.
        pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
        angles = computeAngleFromVecs(dist_vecs[topology.angle_pairs[0]], dist_vecs[topology.angle_pairs[1]])

        # Electric Multipoles #
        q_shell = params.get_atomic_parameters('q_shell')
        dipo = params.get_atomic_parameters('dipo')
        quad = params.get_atomic_parameters('quad')
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

        b = params.get_atomic_parameters('b')
        b_pauli = params.get_atomic_parameters('b_pauli')
        b_disp = params.get_atomic_parameters('b_disp')
        b_xpol = params.get_atomic_parameters('b_xpol')
        b_ct = params.get_atomic_parameters('b_ct')
        eta = params.get_atomic_parameters('eta')

        # HERE: I think the problem is that below produces a 3 for [1,1], which is fine. The problem is that
        # the storage space (of zeros) is not large enough so that when we try to pull from the array we index
        # out of bounds.
        print(params._symmetric_pairing_function(torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])))
        #eps = params.get_pair_parameters('eps', topology.all_intermolecular_pairs) # Need to get intermolecular pairs?
        r_eq = params.get_pair_parameters('r_eq', topology.bonded_pairs)
        j_cf = params.get_pair_parameters('j_cf', topology.bonded_pairs)
        j_cf_pauli = params.get_pair_parameters('j_cf_pauli', topology.bonded_pairs)
        k_hardness_b = params.get_pair_parameters('k_hardness_b', topology.bonded_pairs)
        
        # NOTE(JOE): Need to test that we get the right bond-bond parameters for non-symmetric angles.
        # Currently, we don't have parameters for a non-symmetric angle but they will come up with
        # organic molecules.
        r_eq_bb_1 = params.get_pair_parameters('r_eq', topology.angle_pairs[0])
        r_eq_bb_2 = params.get_pair_parameters('r_eq', topology.angle_pairs[1])
        j_cf_bb_1 = params.get_pair_pair_parameters('j_cf_bb', topology.angle_pairs[0], topology.angle_pairs[1])
        j_cf_bb_2 = params.get_pair_pair_parameters('j_cf_bb', topology.angle_pairs[1], topology.angle_pairs[0])
        k_hardness_bb_1 = params.get_pair_pair_parameters('k_hardness_bb', topology.angle_pairs[0], topology.angle_pairs[1])
        k_hardness_bb_2 = params.get_pair_pair_parameters('k_hardness_bb', topology.angle_pairs[1], topology.angle_pairs[0])

        theta_eq = params.get_angle_parameters('theta_eq', pairs, topology.angle_pairs)
        j_cf_angle = params.get_angle_parameters('j_cf_angle', pairs, topology.angle_pairs)
        k_hardness_angle = params.get_angle_parameters('k_hardness_angle', pairs, topology.angle_pairs)

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

        # Rotation Matrices #
        rotation_matrices = cm.compute_rotation_matrices(params.get_atomic_parameters("axistypes"))
        
        multipoles = rotateMultipoles(
            q_shell, dipo, quad, rotation_matrices
        ) * torch.tensor([1, 1, 1, 1, 1/3, 2/3, 2/3, 1/3, 2/3, 1/3])
        polarizabilities = rotateQuadrupoles(alpha, rotation_matrices)

        # SPEED: Makes copies. Might be unavoidable but could maybe be done more efficiently.
        multipoles_ct_acc = scaleMultipoles(multipoles, q_ct_acc, Kdipo_ct_acc, Kquad_ct_acc)
        multipoles_ct_don = scaleMultipoles(multipoles, q_ct_don, Kdipo_ct_don, Kquad_ct_don)
        multipoles_pauli = scaleMultipoles(multipoles, q_pauli, Kdipo_pauli, Kquad_pauli)

        #ct_direct_pairwise, dq_pairwise = computePairwiseChargeTransfer(
        #    dist_vecs,
        #    multipoles_ct_acc[pairs[0]], multipoles_ct_acc[pairs[1]],
        #    multipoles_ct_don[pairs[0]], multipoles_ct_don[pairs[1]],
        #    b_ct[pairs[0]], b_ct[pairs[1]],
        #    self.nb_params['eps'][pairs[0], pairs[1]]
        #)

