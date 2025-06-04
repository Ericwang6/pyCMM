import torch
from scipy.optimize import minimize

from typing import List
from cmm.system import System
from cmm.force_fields.cmm import CMM2
from cmm.force_fields.spcfw import SPCFW
# ^^^ Figure out how to do this using the __init__.py or whatever so you can do:
# from cmm.force_fields import *

# NOTE(JOE): I am still not sure what the very top-level design of this package
# should look like. We want to handle many different types of calculations
# seamlessly. The basics like single-point calculations, MD, MC, and optimizations
# are needed. As are less common uses like property calculations that happen in post.
# We also want to support use-cases like force field parameterization (this may
# end up being the main use of the package for all I know). The latter of these
# and other use cases also mean we should be able to efficiently run calculations
# over many systems in a batch and return the energies for each system separately
# while maintaining the computational graph for each calculation.
# In the short-term this will just mean running a whole bunch of separate calculations 
# but in the future we can accelerate this tremendously by changing the internals of
# the code to assign separate energies to disjoint sets of pairs which I
# think should be called "segments" of the pairs. We already calculate energies in
# a pairwise manner so this only requires a small change to the topology and how
# we accumulate the energies at the end of Term.
# Anyways, that's getting off track. What I am going to do for now is implement
# a drivers for the various use-cases I have and see what patterns emerge. Maybe
# there is no need for an abstraction and we just provide some top-level drivers
# and dispatch to them based on user-provided settings or maybe we find some
# common pattern that allows us to unify the calculation types in a nice way.
# This file is where the preliminary driver implementations go.

class BatchSinglePointDriver:
    def __init__(self, ff_type: str) -> None:
        # TODO: Eventually, this should also just take the settings and the details
        # about which force field we are using will be in there.
        # Since the FF uses the system settings when it is constructed,
        # we need to figure out the most natural way to specify all the settings
        # but for now this is fine. We use look at the string and build the appropriate
        # FF for each system.
        self._ff_type = ff_type
        self.outputs = []
        self._ff_constructor = None
        self._get_ff_constructor()

    def _get_ff_constructor(self):
        # Dictionary mapping strings to classes
        ff_mapping = {
            "CMM": CMM2,
            "SPCfw": SPCFW,
        }
        
        if self._ff_type in ff_mapping:
            self._ff_constructor = ff_mapping[self._ff_type]
        else:
            raise ValueError(f"You requested ff_type {self._ff_type}, which we do not recognize as a valid force field name.")

    def run(self, systems: List[System], reset=True):
        if reset:
            self.outputs = []
        for i in range(len(systems)):
            ff = self._ff_constructor(systems[i])
            ff.forward(systems[i])
            self.outputs.append(ff.energies)

class OptimizationDriver:
    def __init__(self,
                 ff_type: str,
                 method="BFGS",
                 tolerance: float=1e-6
        ) -> None:
        self.method = method
        self.tolerance = tolerance
        self._ff_type = ff_type
        self._ff_constructor = None
        self._get_ff_constructor()

    def _get_ff_constructor(self):
        ff_mapping = {
            "CMM": CMM2,
            "SPCfw": SPCFW,
        }
        
        if self._ff_type in ff_mapping:
            self._ff_constructor = ff_mapping[self._ff_type]
        else:
            raise ValueError(f"You requested ff_type {self._ff_type}, which we do not recognize as a valid force field name.")

    def energy_and_gradient(self, coords_flat: torch.Tensor):
        """Objective function and gradient for scipy"""
        self.system.coords = torch.tensor(coords_flat.reshape((-1, 3)), dtype=self.system.dtype, requires_grad=True)

        self.ff.forward(self.system)
        self.ff.energies['V_total'].backward()
        return self.ff.energies['V_total'].item(), self.system.coords.grad.numpy().flatten()

    def run(self, system: System):
        self.ff = self._ff_constructor(system)
        self.system = System.from_instance(system)

        result = minimize(self.energy_and_gradient, self.system.coords.detach().numpy().flatten(), method=self.method, tol=self.tolerance, jac=True, options={'disp': True, 'maxiter': 1000})
        return result

            