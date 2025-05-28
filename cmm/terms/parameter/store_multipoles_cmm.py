import torch
from ..term import Term
from ...system import System
from ...multipole import computeCartesianQuadrupoles, convertMultipolesToPolytensor, rotateDipoles, rotateQuadrupoles, computeUndampedInteractionTensorBlocks, formDampingFactorBlocksRank1, formDampingFactorBlocksRank2
from ...short_range import scaleMultipoles, computeShortRangeOneCenterDampFactors, computeShortRangeTwoCenterDampFactors, computeShortRangePolarizationDampFactors

class StoreMultipolesCMM(Term):
    def __init__(self):
        super().__init__()
    
    @property
    def param_data(self):
        return []
    
    @property
    def outputs(self):
        return []

    def forward(self, pairs: torch.Tensor, dists: torch.Tensor, distance_vecs: torch.Tensor, system: System):
        
        # Electric Multipoles #
        q = system.parameterizer.get_atomic_parameters('q')
        Z = system.parameterizer.get_atomic_parameters('Z')
        dipo = system.parameterizer.get_atomic_parameters('dipo')
        quad = system.parameterizer.get_atomic_parameters('quad')
        axis_types = system.parameterizer.get_atomic_parameters('axistypes')
        
        # Pauli Multipoles #
        q_pauli = system.parameterizer.get_atomic_parameters('q_pauli') + system.storage.get('q_flux_pauli')
        Kdipo_pauli = system.parameterizer.get_atomic_parameters('Kdipo_pauli')
        Kquad_pauli = system.parameterizer.get_atomic_parameters('Kquad_pauli')

        # Exchange Polarization #
        q_xpol = system.parameterizer.get_atomic_parameters('q_xpol')
        Kdipo_xpol = system.parameterizer.get_atomic_parameters('Kdipo_xpol')
        Kquad_xpol = system.parameterizer.get_atomic_parameters('Kquad_xpol')

        # Charge Transfer Multipoles #
        q_ct_acc = system.parameterizer.get_atomic_parameters('q_ct_acc')
        Kdipo_ct_acc = system.parameterizer.get_atomic_parameters('Kdipo_ct_acc')
        Kquad_ct_acc = system.parameterizer.get_atomic_parameters('Kquad_ct_acc')
        q_ct_don = system.parameterizer.get_atomic_parameters('q_ct_don')
        Kdipo_ct_don = system.parameterizer.get_atomic_parameters('Kdipo_ct_don')
        Kquad_ct_don = system.parameterizer.get_atomic_parameters('Kquad_ct_don')
        
        # Polarizability #
        eta = system.parameterizer.get_atomic_parameters("eta")
        alpha = system.parameterizer.get_atomic_parameters("alpha")
        alpha_damp_exponent = system.parameterizer.get_atomic_parameters("alpha_damp_exponent")
        alpha_damp_max = system.parameterizer.get_atomic_parameters("alpha_damp_max")

        rotation_matrices = system.compute_rotation_matrices(system.topology.zatoms, system.topology.xatoms, system.topology.yatoms, axis_types)

        polarizabilities = rotateQuadrupoles(alpha, rotation_matrices)

        dipo_lr = rotateDipoles(dipo, rotation_matrices).squeeze(1)
        quad_lr = rotateQuadrupoles(quad, rotation_matrices)
        q_total = q + system.storage.get('q_flux')
        
        multipoles_real = convertMultipolesToPolytensor(
            q_total, dipo_lr, quad_lr
        )
        multipoles_cp = convertMultipolesToPolytensor(
            q_total - Z, dipo_lr, quad_lr
        )
        multipoles_Z = torch.zeros_like(multipoles_real)
        multipoles_Z[:, 0] += Z
        multipoles_ct_acc = scaleMultipoles(multipoles_real, q_ct_acc, Kdipo_ct_acc, Kquad_ct_acc)
        multipoles_ct_don = scaleMultipoles(multipoles_real, q_ct_don, Kdipo_ct_don, Kquad_ct_don)
        multipoles_pauli = scaleMultipoles(multipoles_real, q_pauli, Kdipo_pauli, Kquad_pauli)
        multipoles_xpol = scaleMultipoles(multipoles_real, q_xpol, Kdipo_xpol, Kquad_xpol)
        
        # Storage for field data #
        electric_field_data = torch.zeros(system.neighbor_list.natoms, 10, device=dists.device, dtype=dists.dtype, requires_grad=True)
        # NOTE(JOE): The 10 here is for the potential (1), field (3), and field gradients (6)
        # If we used spherical harmonics there would be 5 field gradient components
        # but we can only reduce cartesian quadrupoles to 6 components using symmetry.
        # In any case, the code for dealing with multipoles and all related variables could be more general
        # and likely more efficient.

        # Store all multipoles for later use #
        system.storage.add('q', q_total)
        system.storage.add('dipo', dipo_lr)
        system.storage.add('quad', quad_lr)
        system.storage.add('multipoles_real', multipoles_real)
        system.storage.add('multipoles_cp', multipoles_cp)
        system.storage.add('multipoles_Z', multipoles_Z)
        system.storage.add('multipoles_ct_acc', multipoles_ct_acc)
        system.storage.add('multipoles_ct_don', multipoles_ct_don)
        system.storage.add('multipoles_pauli', multipoles_pauli)
        system.storage.add('multipoles_xpol', multipoles_xpol)
        system.storage.add('electric_field_data', electric_field_data)
        system.storage.add('eta', eta)
        system.storage.add('polarizabilities', polarizabilities)
        system.storage.add('alpha_damp_exponent', alpha_damp_exponent)
        system.storage.add('alpha_damp_max', alpha_damp_max)

        return {}