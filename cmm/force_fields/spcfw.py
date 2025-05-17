import torch, math
from .ff import FF
from ..terms.bonded.harmonic_bond import HarmonicBond

from ..system import System
from ..parameters import Parameterizer2
from ..units import HARTREE2KCAL, BOHR2ANG

class SPCFW(FF):
    def __init__(self, dtype: torch.dtype=torch.float64, device: torch.DeviceObjType=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), requires_param_grads: bool=False) -> None:
        super().__init__(dtype, device)

        self.add_term(HarmonicBond())

        self.atomic_params = {
            "O_water": {
                "q": torch.tensor(-0.82, dtype=self._dtype, device=self._device)
            },
            "H_water": {
                "q": torch.tensor(0.41, dtype=self._dtype, device=self._device)
            }
        }

        self.pair_params = {
            ("O_water", "H_water"): {
                "k_bond": torch.tensor([1059.162 / HARTREE2KCAL * BOHR2ANG * BOHR2ANG], dtype=self._dtype, device=self._device, requires_grad=requires_param_grads),
                "r_eq": torch.tensor([1.012 / BOHR2ANG], dtype=self._dtype, device=self._device, requires_grad=requires_param_grads),
            },
            ("O_water", "O_water"): {
                "eps_lj": torch.tensor([0.1554253 / HARTREE2KCAL], dtype=self._dtype, device=self._device, requires_grad=requires_param_grads),
                "sigma_lj": torch.tensor([3.165492 / BOHR2ANG], dtype=self._dtype, device=self._device, requires_grad=requires_param_grads),
            },
        }

        self.angle_params = {
            ("H_water", "O_water", "H_water"): {
                "k_theta": torch.tensor([75.90 / HARTREE2KCAL], dtype=self._dtype, device=self._device, requires_grad=requires_param_grads),
                "theta_eq": torch.tensor([113.24 * math.pi / 180.0], dtype=self._dtype, device=self._device, requires_grad=requires_param_grads),
            }
        }

        with torch.no_grad():
            # Symmetrize the parameter dictionaries for convenience when making parameter arrays #
            for key in list(self.pair_params.keys()):
                self.pair_params[(key[1], key[0])] = self.pair_params[key]
            for key in list(self.angle_params.keys()):
                self.angle_params[(key[2], key[1], key[0])] = self.angle_params[key]

    def forward(self, system: System):
        # TODO: Make a better API for filling out the parameter arrays and getting the params.
        # There should be a simple way to specify which indices are needed for each term and
        # which parameters. Those should then get filled in all in one call so that we can
        # basically call one setup function which fills in the parameter arrays and then have
        # a static run through the force field. Some parameters depend on the outcome of other
        # terms so with some force fields there have to be multiple stages to evaluation, but
        # we will cross that bridge when we get there.
        pairs, dists, distance_vecs = system.get_distances_vectors_and_pairs()
        system.parameterizer.update(pairs, self.atomic_params, self.pair_params, self.angle_params, angle_atoms=system.topology.angle_atoms)

        #excluded_pair_indices = system.neighbor_list.get_pair_indices(system.neighbor_list.excluded_pairs)
        #included_pair_indices = system.neighbor_list.get_pair_indices(system.neighbor_list.included_pairs)
        #angle_pair_indices_ij = system.neighbor_list.get_pair_indices(system.topology.angle_atoms[:, 0:2])
        #angle_pair_indices_jk = system.neighbor_list.get_pair_indices(system.topology.angle_atoms[:, 1:].flip(1))
        #theta_eq = system.parameterizer.get_angle_parameters('theta_eq', system.topology.angle_atoms)
        #k_theta = system.parameterizer.get_angle_parameters('k_theta', system.topology.angle_atoms)
        
        for term in self.terms:
            output_dict = term.forward(pairs, dists, distance_vecs, system)
            for key in output_dict.keys():
                self.energies[key] = output_dict[key]
        
        print(self.energies)
