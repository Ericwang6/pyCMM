import warnings
from abc import ABC, abstractmethod
import itertools
from typing import List, Dict, Any, Iterable, Callable, Tuple, Optional
import torch
import torch.nn as nn
from torch_scatter import scatter
from ..topology import Topology
from ..multipole import AxisTypes, computeCartesianQuadrupoles


def format_types(types: Iterable, check=False):
    if isinstance(types, str):
        if check:
            assert '/' not in types, f'A valide string "/" in {types}'
        return types
    else:
        if check:
            for t in types:
                assert '/' not in t, f'A valide string "/" in {t}'
        return '/'.join(str(t) for t in types)


class Parametrizer(ABC, nn.Module):
    def __init__(
        self, 
        types: Iterable, 
        top: Topology, 
        name: str = '',
        handle_unmatched: str = 'default',
        use_atom_class: bool = False
    ):
        super().__init__()
        self.top = top
        assert top.atomTypes, 'Atom types are not assigned'

        self.atomTypes = self.top.atomTypes if not use_atom_class else self.top.atomClasses
        self.typesAsDict = {}
        
        if len(types) > 0:
            self._num = 1 if isinstance(types[0], str) else len(types[0])
            for i, t in enumerate(types):
                # typ = t if isinstance(t, str) else tuple(str(x) for x in t)
                typ = format_types(t, True)
                n = 1 if isinstance(t, str) else len(t)
                assert self._num == n, "Number of types not consistent"
                self.typesAsDict[typ] = i
        
        self.name = name if name else self.__class__.__name__[:-12]
        self.params = {}
        self.params_expand = {}
        self.device = top.device

        if handle_unmatched == 'default':
            self.setDefaultHandleUnmatched()
        elif handle_unmatched == 'error':
            self.raise_error = True
            self.raise_warning = False
        elif handle_unmatched == 'warning':
            self.raise_error = False
            self.raise_warning = True
        elif not handle_unmatched:
            self.raise_error = False
            self.raise_warning = False
        else:
            raise ValueError(f"Invalid value: '{handle_unmatched}' (valid values: 'default', 'error', 'warning' or empty string)")
        
        self._param_is_indices = {}
        atomIndices, paramIndices = self.initIndices()
        self.registerIndices('atomIndices', atomIndices)
        self.registerIndices('paramIndices', paramIndices)
        self.is_empty: bool = self.params['atomIndices'].numel() == 0
    
    def expandParameters(self):
        for name in self.params:
            if self._param_is_indices[name]:
                self.params_expand[name] = self.params[name].clone()
            elif self.params['paramIndices'].numel() > 0:
                self.params_expand[name] = self.params[name][self.params['paramIndices']]

    def setDefaultHandleUnmatched(self):
        self.raise_error = False
        self.raise_warning = False

    def getParameters(self, name: str):
        return self.params[name]
    
    def getExpandParameters(self, name: str):
        return self.params_expand[name]

    @abstractmethod
    def initIndices(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    def registerParameters(self, name: str, params: torch.Tensor):
        assert params.shape[0] == len(self.typesAsDict), \
                f"Length of input parameters not correct, should be {len(self.typesAsDict)}, but found {params.shape[0]}"
        # When adding an empty tensor, by default they all point to the same stoarge 
        # and this will make torch.jit.script fails
        if params.numel() == 0:
            self.params[name] = torch.empty(1, device=self.device)[:0]
        else:
            self.params[name] = params
        self._param_is_indices[name] = False
    
    def registerIndices(self, name: str, indices: torch.Tensor):
        # When adding an empty tensor, by default they all point to the same stoarge 
        # and this will make torch.jit.script fails
        if indices.numel() == 0:
            self.params[name] = torch.empty(1, device=self.device)[:0]
        else:
            self.params[name] = indices
        self._param_is_indices[name] = True
        
    def raiseException(self, msg):
        if self.raise_error:
            raise RuntimeError(msg)
        if self.raise_warning:
            warnings.warn(msg)
    
    def raiseUnmatchExcpetion(self, atoms: List[int]):
        self.raiseException(f"Atoms {'-'.join(str(x) for x in atoms)} does not match any {self.name}")


class AtomicParametrizer(Parametrizer):

    def setDefaultHandleUnmatched(self):
        self.raise_error = True
        self.raise_warning = False

    def initIndices(self):
        paramIndices = []
        atomIndices = []
        for i, atype in enumerate(self.atomTypes):
            paramIndices.append(self.typesAsDict[atype])
            atomIndices.append(i)
        atomIndices = torch.tensor(atomIndices, device=self.device)
        paramIndices = torch.tensor(paramIndices, device=self.device)
        return atomIndices, paramIndices
    
    def expandParameters(self):
        return super().expandParameters()
    

class BondParametrizer(Parametrizer):

    def setDefaultHandleUnmatched(self):
        self.raise_error = True
        self.raise_warning = False

    def initIndices(self):
        terms = self.top.getBonds(asTensor=False)
        paramIndices = []
        atomIndices = []
        for term in terms:
            types = tuple(self.atomTypes[t] for t in term)
            typ1 = format_types(types, True)
            typ2 = format_types(list(reversed(types)), True)
            if typ1 in self.typesAsDict:
                paramIndices.append(self.typesAsDict[typ1])
                atomIndices.append(term)
            elif typ2 in self.typesAsDict:
                paramIndices.append(self.typesAsDict[typ2])
                atomIndices.append(list(reversed(term)))
            else:
                self.raiseUnmatchExcpetion(term)

        return torch.tensor(atomIndices, device=self.device), torch.tensor(paramIndices, device=self.device)
    

class AngleParametrizer(Parametrizer):
    """
    Class for assigning parameters to angles
    """
    
    def setDefaultHandleUnmatched(self):
        self.raise_error = True
        self.raise_warning = False
    
    def initIndices(self):
        terms = self.top.getAngles(asTensor=False)
        paramIndices = []
        atomIndices = []
        for term in terms:
            types = tuple(self.atomTypes[t] for t in term)
            typ1 = format_types(types, True)
            typ2 = format_types(list(reversed(types)), True)
            if typ1 in self.typesAsDict:
                paramIndices.append(self.typesAsDict[typ1])
                atomIndices.append(term)
            elif typ2 in self.typesAsDict:
                paramIndices.append(self.typesAsDict[typ2])
                atomIndices.append(list(reversed(term)))
            else:
                self.raiseUnmatchExcpetion(term)
        return torch.tensor(atomIndices, device=self.device), torch.tensor(paramIndices, device=self.device)


class AngleAngleParametrizer(Parametrizer):
    """
    Class for assigning parameters to angle-angle couplings
    """

    def setDefaultHandleUnmatched(self):
        self.raise_error = False
        self.raise_warning = True
    
    def initIndices(self):
        angles = self.top.getAngles(asTensor=False)
        paramIndices = []
        atomIndices = []
        for angle in angles:
            angles2 = self.top.getAnglesByBond(angle[0], angle[1]) + self.top.getAnglesByBond(angle[1], angle[2])
            for angle2 in angles2:
                # avoid double counting and self-self coupling
                if hash(tuple(angle)) >= hash(tuple(angle2)):
                    continue
                # type 1: i-j-k/i-j-k'
                # type 2: i-j-k/j-i-k'
                couple_type = "1" if angle[1] == angle2[1] else "2"
                trials = [
                    angle+angle2, angle+angle2[::-1], angle[::-1]+angle2, angle[::-1]+angle2,
                    angle2+angle, angle2+angle[::-1], angle2[::-1]+angle, angle2[::-1]+angle,
                ]
                for trial in trials:
                    # key = tuple([self.atomTypes[t] for t in trial] + [couple_type])
                    key = format_types([self.atomTypes[t] for t in trial] + [couple_type])
                    if key in self.typesAsDict:
                        paramIndices.append(self.typesAsDict[key])
                        atomIndices.append(trial)
                        break
                else:
                    self.raiseUnmatchExcpetion(trials[0])

        return torch.tensor(atomIndices, device=self.device), torch.tensor(paramIndices, device=self.device)


class TorsionBondParametrizer(Parametrizer):
    """
    Class for assigning parameters to torsion-bond couplings
    """
    def setDefaultHandleUnmatched(self):
        self.raise_error = False
        self.raise_warning = True

    def initIndices(self):
        dihes = self.top.getDihedrals(asTensor=False)
        paramIndices = []
        atomIndices = []
        for dihe in dihes:
            # type 1: i-j-k-l/j-k
            # type 2: i-j-k-l/i-j
            # type 3: i-j-k-l/j-k'
            # type 3: i-j-k-l/i-k'
            bonds = [dihe[1:3], dihe[:2], dihe[-2:]]
            couple_types = ["1", "2", "2"]
            for i in dihe[1:3]:
                for j in self.top.getNeighborAtoms(i):
                    if j in dihe:
                        continue
                    bonds.append((i, j))
                    couple_types.append("3")
            
            for i in [dihe[0], dihe[-1]]:
                for j in self.top.getNeighborAtoms(i):
                    if j in dihe:
                        continue
                    bonds.append((i, j))
                    couple_types.append("4")

            for bo, couple_type in zip(bonds, couple_types, strict=True):
                trials = [dihe+bo, dihe+bo[::-1], dihe[::-1]+bo, dihe[::-1]+bo[::-1]]
                for trial in trials:
                    # key = tuple([self.atomTypes[t] for t in trial] + [couple_type])
                    key = format_types([self.atomTypes[t] for t in trial] + [couple_type])
                    if key in self.typesAsDict:
                        paramIndices.append(self.typesAsDict[key])
                        atomIndices.append(trial)
                        break
                else:
                    self.raiseUnmatchExcpetion(trials[0])

        return torch.tensor(atomIndices, device=self.device), torch.tensor(paramIndices, device=self.device)


class TorsionAngleParametrizer(Parametrizer):
    """
    Class for assigning parameters to torsion-angle couplings
    """
    def setDefaultHandleUnmatched(self):
        self.raise_error = False
        self.raise_warning = True

    def initIndices(self):
        dihes = self.top.getDihedrals(asTensor=False)
        paramIndices = []
        atomIndices = []
        for dihe in dihes:
            # type 1: i-j-k-l/i-j-k
            # type 2: i-j-k-l/j-k-l'
            # type 3: i-j-k-l/i-j-k'
            angles = [dihe[:3], dihe[-3:]]
            couple_types = ["1", "1"]

            for j in self.top.getNeighborAtoms(dihe[1]):
                if j in dihe:
                    continue
                angles.append((dihe[2], dihe[1], j))
                couple_types.append("2")
                angles.append((dihe[0], dihe[1], j))
                couple_types.append("3")

            for j in self.top.getNeighborAtoms(dihe[2]):
                if j in dihe:
                    continue
                angles.append((dihe[1], dihe[2], j))
                couple_types.append("2")
                angles.append((dihe[3], dihe[2], j))
                couple_types.append("3")

            for angle, couple_type in zip(angles, couple_types, strict=True):
                trials = [dihe+angle, dihe+angle[::-1], dihe[::-1]+angle, dihe[::-1]+angle[::-1]]
                for trial in trials:
                    # key = tuple([self.atomTypes[t] for t in trial] + [couple_type])
                    key = format_types([self.atomTypes[t] for t in trial] + [couple_type])
                    if key in self.typesAsDict:
                        paramIndices.append(self.typesAsDict[key])
                        atomIndices.append(trial)
                        break
                else:
                    self.raiseUnmatchExcpetion(trials[0])

        return torch.tensor(atomIndices, device=self.device), torch.tensor(paramIndices, device=self.device)


class TorsionParametrizer(Parametrizer):

    def setDefaultHandleUnmatched(self):
        self.raise_error = True
        self.raise_warning = False

    def initIndices(self):
        terms = self.top.getDihedrals(asTensor=False)
        paramIndices = []
        atomIndices = []
        for term in terms:
            types = tuple(self.atomTypes[t] for t in term)
            typ1 = format_types(types, True)
            typ2 = format_types(list(reversed(types)), True)
            if typ1 in self.typesAsDict:
                paramIndices.append(self.typesAsDict[typ1])
                atomIndices.append(term)
            elif typ2 in self.typesAsDict:
                paramIndices.append(self.typesAsDict[typ2])
                atomIndices.append(list(reversed(term)))
            else:
                self.raiseUnmatchExcpetion(term)
        return torch.tensor(atomIndices, device=self.device), torch.tensor(paramIndices, device=self.device)


class MultipoleParametrizer(AtomicParametrizer):

    def initIndices(self):
        paramIndices = []
        atomIndices = []
        kzIndices = []
        kxIndices = []
        kyIndices = []
        for i, atype in enumerate(self.atomTypes):
            neighbors = self.top.getNeighborAtoms(i)
            trials = [[i, -1, -1, -1]]
            for kz in neighbors:
                trials.append([i, kz, -1, -1])
            for kz, kx in itertools.permutations(neighbors, 2):
                trials.append([i, kz, kx, -1])
            for kz, kx, ky in itertools.permutations(neighbors, 3):
                trials.append([i, kz, kx, ky])
            
            for kz in neighbors:
                for kx in self.top.getNeighborAtoms(kz):
                    if kz != kx and kx != i:
                        trials.append([i, kz, kx, -1])
            
            # ZBisect ky (HN in methylamine)
            for kz in neighbors:
                for kx in self.top.getNeighborAtoms(kz):
                    if kz != kx and kx != i:
                        for ky in self.top.getNeighborAtoms(kz):
                            if ky != kx and ky != i:
                                trials.append([i, kz, kx, ky])
            
            for trial in trials:
                key = []
                for t in trial:
                    key.append('' if t == -1 else self.atomTypes[t])
                # key = tuple(key)
                key = format_types(key)
                if key in self.typesAsDict:
                    paramIndices.append(self.typesAsDict[key])
                    atomIndices.append(i)
                    kzIndices.append(trial[1])
                    kxIndices.append(trial[2])
                    kyIndices.append(trial[3])
                    break
            else:
                self.raiseUnmatchExcpetion([i])

        self.kzIndices = torch.tensor(kzIndices, device=self.device)
        self.kxIndices = torch.tensor(kxIndices, device=self.device)
        self.kyIndices = torch.tensor(kyIndices, device=self.device)
        self.registerIndices('kzIndices', self.kzIndices)
        self.registerIndices('kxIndices', self.kxIndices)
        self.registerIndices('kyIndices', self.kyIndices)
        return torch.tensor(atomIndices, device=self.device), torch.tensor(paramIndices, device=self.device)
    
    def expandParameters(self):
        dipo = torch.vstack((self.params['dx'], self.params['dy'], self.params['dz'])).T.contiguous()
        self.params['dipo'] = dipo
        self._param_is_indices['dipo'] = False

        quad_s = torch.vstack([self.params[p] for p in ['q20', 'q21c', 'q21s', 'q22c', 'q22s']]).T.contiguous()
        quad = computeCartesianQuadrupoles(quad_s)
        self.params['quad'] = quad
        self._param_is_indices['quad'] = False

        for name in self.params:
            if self._param_is_indices[name]:
                self.params_expand[name] = self.params[name]
            elif self.params['paramIndices'].numel() > 0:
                self.params_expand[name] = self.params[name][self.params['paramIndices']]


class PolarizationParametrizer(AtomicParametrizer):

    def expandParameters(self):
        # alpha = torch.vmap(torch.diag)(torch.vstack([self.params['alpha_xx'], self.params['alpha_yy'], self.params['alpha_zz']]).T)
        alpha = torch.diag_embed(torch.vstack([self.params['alpha_xx'], self.params['alpha_yy'], self.params['alpha_zz']]).T)
        self.params['alpha'] = alpha
        self._param_is_indices['alpha'] = False
        for name in self.params:
            if self._param_is_indices[name]:
                self.params_expand[name] = self.params[name].clone()
            elif self.params['paramIndices'].numel() > 0:
                self.params_expand[name] = self.params[name][self.params['paramIndices']]

@torch.compile
def symmetric_pairing_function(pairs: torch.Tensor) -> torch.Tensor:
    """
    Given two positive indices, (i,j), this function generates a unique index k.
    The particular pairing function chosen here is described in equation (13) in 
    https://arxiv.org/pdf/2105.10752.

    The important features of this pairing function are that it is symmetric (i.e. the order of
    indices does not matter. Most pairing functions intentionally don't have this property).
    Additionally, if we have N atom types, the largest index is close to N^2. We can check
    the actual value and use it to pre-allocate the arrays we index into. In the case that
    the number of atom types is very large, the arrays will become sparse and we can just
    revisit this solution at that point. Likely, just using a sparse arrays is sufficient.
    """
    tmp = torch.sum(pairs, dim=1) + 1
    k = torch.floor_divide(torch.square(tmp) - torch.remainder(tmp, 2), 4) + torch.min(pairs, dim=1).values
    return k


class PairParametrizer(AtomicParametrizer):
    def __init__(
        self,
        types: Iterable, 
        top: Topology, 
        name: str,
        combination_rule: str = 'geometric',
        use_atom_class: bool = False
    ):
        super().__init__(types, top, name, use_atom_class=use_atom_class)
        self.combination_rule = combination_rule
        assert combination_rule == 'geometric', f'Combination rule {combination_rule} not supported'

        self.specific_pair_param_indices: Dict[str, torch.Tensor] = {'_dummy_': torch.Tensor([])}
        self.specific_pair_params: Dict[str, torch.Tensor] = {'_dummy_': torch.Tensor([])}

        atom_type_pairs = []
        for i in range(len(self.typesAsDict)):
            for j in range(i, len(self.typesAsDict)):
                atom_type_pairs.append([i, j])
        self.atom_type_pairs = torch.tensor(atom_type_pairs, device=self.device)
        self.atom_type_pairs_after_pairing_func = symmetric_pairing_function(self.atom_type_pairs)

        self._param_is_pairwise = {}
    
    def registerParameters(self, name, params):
        super().registerParameters(name, params)
        self._param_is_pairwise[name] = False

    def registerPairwiseParameters(
        self, 
        name: str,
        params: torch.Tensor, 
        specific_pair_types: List[Tuple[str, str]] = list(),
        specific_pair_params: torch.Tensor | None = None
    ):
        
        super().registerParameters(name, params)
        self._param_is_pairwise[name] = True

        if len(specific_pair_params) != 0:
            assert len(specific_pair_params.shape) == 1, "Input specific pair parameter must be a 1-D tensor"
            assert specific_pair_params.shape[0] == len(specific_pair_types), \
                f"Length of input parameters not correct, should be {len(specific_pair_types)}, but found {specific_pair_params.shape[0]}"
            atom_type_pairs = torch.tensor([[self.typesAsDict[t] for t in typ] for typ in specific_pair_types], device=params.device)
            self.specific_pair_param_indices[name] = symmetric_pairing_function(atom_type_pairs)
            self.specific_pair_params[name] = specific_pair_params
    
    def expandParameters(self):
        for name in self.params:
            if self._param_is_indices[name]:
                self.params_expand[name] = self.params[name].clone()
            else:
                # we also want to keep track of the non-paired values
                self.params_expand[name] = self.params[name][self.params['paramIndices']]
                if self._param_is_pairwise[name]:
                    if self.combination_rule == 'geometric':
                        pairwise_params_as_types = torch.sqrt(self.params[name][self.atom_type_pairs[:, 0]] * self.params[name][self.atom_type_pairs[:, 1]])
                    else:
                        pairwise_params_as_types = (self.params[name][self.atom_type_pairs[:, 0]] + self.params[name][self.atom_type_pairs[:, 1]]) / 2
                    pairwise_param = scatter(
                        pairwise_params_as_types,
                        self.atom_type_pairs_after_pairing_func
                    )
                    if name in self.specific_pair_param_indices:
                        pairwise_param[self.specific_pair_param_indices[name]] = self.specific_pair_params[name]
                    self.params_expand[name+'_ij'] = pairwise_param

    def getExpandParameters(self, name: str, pairs: torch.Tensor | None = None):
        if pairs is None:
            return self.params_expand[name]
        else:
            return self.params_expand[name+'_ij'][symmetric_pairing_function(self.params['paramIndices'][pairs])]
