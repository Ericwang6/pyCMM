import torch
from ..term import Term, CutoffType, ParameterType
from ...system import System
from ...ewald import long_range_potential_rank_1
from ...polarization import direct_polarization_guess, compute_product_with_polarization_matrix, direct_field_induced_dipole_guess

class MultipolarPolarization1(Term):
    def __init__(self, solver, alpha: float, k_max: int, get_fields: bool=False):
        super().__init__()
        self.solver = solver
        self.alpha = alpha
        self.k_max = k_max
        self.get_fields = get_fields
        self.last_induced_multipoles = None
    
    @property
    def param_data(self):
        return [('induced_multipoles_real', ParameterType.Atomic, CutoffType.NB_Medium)]
    
    @property
    def outputs(self):
        return ['V_polarization']

    def forward(self, pairs: torch.Tensor, dists: torch.Tensor, distance_vecs: torch.Tensor, system: System):
        lr_settings = system.settings.get_long_range_electrostatics_settings()
        long_range_induced_potential_function = None
        if lr_settings.use_long_range:
            long_range_induced_potential_function = lambda charges, dipoles : long_range_potential_rank_1(system.coords, charges, dipoles, system.box, self.alpha, self.k_max)

        #pairs_medium = system.storage.get('pairs_medium')
        pairs_medium_pol = system.storage.get('pairs_medium_pol')
        #pairs_short = system.storage.get('pairs_short')
        pairs_pol = system.storage.get('pairs_pol')
        #pairs_excl = system.storage.get('pairs_excl')
        pairs_excl_pol = system.storage.get('pairs_excl_pol')
        pairs_medium_pol_i_a = pairs_medium_pol[:, 0]
        pairs_medium_pol_j_a = pairs_medium_pol[:, 1]
        #pairs_short_i_a = pairs_short[:, 0]
        #pairs_short_j_a = pairs_short[:, 1]
        pairs_pol_i_a = pairs_pol[:, 0]
        pairs_pol_j_a = pairs_pol[:, 1]
        pairs_excl_pol_i_a = pairs_excl_pol[:, 0]
        pairs_excl_pol_j_a = pairs_excl_pol[:, 1]

        direct_field_tensor_rank_1_medium = system.storage.get('direct_field_tensor_medium_pol')
        direct_field_tensor_excl_rank_1 = system.storage.get('direct_field_tensor_excl_pol')
        #pol_interaction_tensor_short = system.storage.get('pol_interaction_tensor_short')
        pol_interaction_tensor_pol = system.storage.get('pol_interaction_tensor_pol')

        eta = system.storage.get('eta')
        polarizabilities = system.storage.get('polarizabilities')
        inverse_polarizabilities = system.storage.get('inverse_polarizabilities')
        dq_groups = system.storage.get('dq_groups')

        # Form the total potential and field from stored components #
        electric_field_data = system.storage.get('electric_field_data')
        elec_potential = electric_field_data[:, 0]
        elec_field = electric_field_data[:, 1:4].mul(torch.tensor([-1, -1, -1], device=system.device).reshape(1, -1))
        if lr_settings.use_long_range:
            elec_potential = elec_potential + system.storage.get('ewald_elec_potential')
            elec_field = elec_field + system.storage.get('ewald_elec_field')
        system.storage.add('electric_potential_permanent', elec_potential)
        system.storage.add('electric_field_permanent', elec_field)

        def A_mm(x: torch.Tensor):
            return compute_product_with_polarization_matrix(
                x,
                system.topology.natoms,
                pairs_medium_pol_i_a, pairs_medium_pol_j_a, pairs_pol_i_a, pairs_pol_j_a, #pairs_short_i_a, pairs_short_j_a,
                pairs_excl_pol_i_a, pairs_excl_pol_j_a, direct_field_tensor_rank_1_medium,
                pol_interaction_tensor_pol, #pol_interaction_tensor_short,
                direct_field_tensor_excl_rank_1,
                eta, inverse_polarizabilities, system.topology.pol_group_indices_a,
                system.topology.pol_group_segment_indices, system.topology.pol_group_lengths_g,
                long_range_potential_function=long_range_induced_potential_function
            )

        def M_mm_direct(x: torch.Tensor):
            return direct_polarization_guess(
                x, system.topology.natoms, system.topology.n_pol_groups, polarizabilities
            )

        b_vector = torch.hstack((-elec_potential, elec_field.flatten(), dq_groups))
        with torch.no_grad():
            # Evaluate the initial guess #
            # TODO: This only applies on the first step (before there is a history), so we should really just move it into the solver initialization.
            if self.last_induced_multipoles is None:
                self.last_induced_multipoles = direct_field_induced_dipole_guess(system.topology.natoms, system.topology.n_pol_groups, polarizabilities, elec_field)
            
            self.solver.A_mm = A_mm
            self.solver.M_mm = M_mm_direct
            self.last_induced_multipoles = self.solver.solve(B=b_vector, X0=self.last_induced_multipoles)
        
        V_pol = torch.dot(self.last_induced_multipoles, (0.5 * A_mm(self.last_induced_multipoles) - b_vector))
        return {'V_polarization': V_pol}