"""
alchemy.py

Lambda-dependent (alchemical) parameter scaling for CMM hydration free
energy calculations.

TWO-STAGE PROTOCOL (required): scaling all parameters together in one
stage is unsafe -- at intermediate lambda the repulsive wall (q_pauli) can
be weak while attraction (mono/alpha/eta) is still strong, letting a water
H punch through and collapse onto the ion. This module always uses two
independent methods, set_electrostatic_lambda() and set_vdw_lambda(),
rather than one combined lambda.

Two window generators implement different published protocols:
- two_stage_windows(): discharge (electrostatic bucket) first, then
  decouple (vdw bucket).
- mbucb_style_windows(): MB-UCB's growth (vdw first) then charging
  (electrostatic second).
- gsc_two_step_windows(): Li & Nam's GSC-based two-step annihilation --
  see that function's docstring for the corrected eq 11 derivation.

alpha/eta and q_pauli are floored near (not at) zero by default to avoid a
singular polarizability matrix and a Pauli 1/r divergence respectively;
see disable_polarization_floor()/disable_vdw_floor() to remove each once
validated.

Term bucketing:

    Term              Parametrizer            Parameter(s)         Bucket
    ----              ------------             ------------         -----
    Electrostatics    Multipoles               mono                 electrostatic
    Charge Pen.       ChargePenetration        Z                    electrostatic
    Polarization      Polarization             alpha_xx/yy/zz, eta  electrostatic
    Charge Transfer   ChargeTransfer           q_ct_acc, q_ct_don   electrostatic
    Charge Transfer   ChargeTransfer           eps_ct (override)    electrostatic
    XPol               ExchangePolarization     q_xpol               electrostatic
    Pauli             Pauli                    q_pauli              vdw
    Dispersion        Dispersion               C6_disp (override)  vdw

Dispersion and eps_ct use an override mechanism instead of atomic scaling:
PairParametrizer combines atomic values geometrically (sqrt(C6_i*C6_j)), so
scaling the ion's atomic value alone gives sqrt(lambda) decoupling, not
lambda. This module instead writes lambda*C6_ij (or lambda*eps_ct) directly
into the pairwise override table, bypassing the combining rule.

Call expand_affected() after every set_*_lambda() call, before running MD.

GSC (Gaussian soft-core, optional): q_pauli has a genuine 1/r divergence at
short range. enable_gsc() adds a bounded additive bump to protect against
this. Requires a small addition to System.getEnergy() -- see gsc.py.
"""

from typing import Dict, List, Tuple
import torch

from .ffxml.parametrizer import Parametrizer, symmetric_pairing_function
from .gsc import gsc_alpha_schedule


# q_xpol is grouped with the electrostatic bucket (MB-UCB convention), not
# with Pauli -- a deliberate choice to match their published protocol.
_ELECTROSTATIC_ATOMIC_SPECS = [
    ('Multipoles',           'mono'),
    ('ChargePenetration',    'Z'),
    ('ChargeTransfer',       'q_ct_acc'),
    ('ChargeTransfer',       'q_ct_don'),
    ('Polarization',         'eta'),
    ('Polarization',         'alpha_xx'),
    ('Polarization',         'alpha_yy'),
    ('Polarization',         'alpha_zz'),
    ('ExchangePolarization', 'q_xpol'),
]
_VDW_ATOMIC_SPECS = [
    ('Pauli', 'q_pauli'),
]

_AFFECTED_PARAMETRIZERS = [
    'Multipoles', 'ChargePenetration', 'Pauli', 'ExchangePolarization',
    'ChargeTransfer', 'Polarization', 'Dispersion',
]

# Keeps the polarizability matrix from becoming singular at lambda=0.
_ALPHA_ETA_LAMBDA_FLOOR = 1e-3

# Keeps a minimal steric wall so nothing can diffuse onto the ion at
# vdw_lambda=0. Not needed once GSC is enabled and validated.
_VDW_LAMBDA_FLOOR = 1e-3


def _linspace(a: float, b: float, n: int) -> List[float]:
    if n <= 1:
        return [a]
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def two_stage_windows(n_discharge: int = 10, n_decouple: int = 10) -> List[Tuple[str, float]]:
    """
    (stage, lambda) tuples: discharge (electrostatic, 1->0) then decouple
    (vdw, 1->0). Use mbucb_style_windows() for MB-UCB's own growth/charging
    order instead.
    """
    windows = [('discharge', lam) for lam in _linspace(1.0, 0.0, n_discharge)]
    windows += [('decouple', lam) for lam in _linspace(1.0, 0.0, n_decouple)]
    return windows


def mbucb_style_windows(increment: float = 0.1) -> List[Tuple[str, float]]:
    """
    MB-UCB's published growth/charging protocol: vdw bucket grown 0->1
    first (electrostatic held at 0), then electrostatic bucket charged 0->1
    second (vdw held at 1).

    Steps by `increment` from 0.0 to 1.0 inclusive (11 points for the
    default 0.1) -- "10 windows" in the paper means 10 intervals, not 10
    points.

    Caller must call set_electrostatic_lambda(0.0) once before growth
    starts, and must not call set_vdw_lambda again during charging.
    """
    n_points = int(round(1.0 / increment)) + 1
    lambdas = [round(i * increment, 10) for i in range(n_points)]
    windows = [('growth', lam) for lam in lambdas]
    windows += [('charging', lam) for lam in lambdas]
    return windows


def gsc_two_step_windows(n_decouple: int = 11, n_remove_gsc: int = 11) -> List[Tuple[str, float, float | None]]:
    """
    Li & Nam (2020)'s two-step GSC annihilation protocol (Figure 1b).

    Working out eq 11 for the annihilation case (state 1 = real physics, no
    bump; state 0 = bump only, no real physics):

        U(lambda) = (1-lambda) * alpha_0 * bump(r)  +  lambda * U_real(r)

    The bump's coefficient is (1-lambda) -- the SAME lambda driving the
    real interactions, just flipped. It must ramp in smoothly as real
    physics ramps out: zero at lambda=1, full alpha_max only as lambda->0.
    This is exactly the DEFAULT automatic coupling (gsc_alpha_schedule via
    set_vdw_lambda) -- so step 1 needs NO set_gsc_lambda() call at all.

    An earlier, incorrect version called set_gsc_lambda(1.0) once at the
    start of step 1, pinning the bump at full strength for the whole stage
    including lambda=1.0 (where it should be ~0). This created a real,
    empirically-confirmed artificial energy trap. Do not reintroduce it.

    Step 1 ("decouple"): electrostatic_lambda and vdw_lambda move together,
    1.0 -> 0.0, with the bump auto-tracking (1-lambda).

    Step 2 ("remove_gsc"): electrostatic_lambda/vdw_lambda fixed at 0.0
    (safe by construction -- q_pauli(lambda)=0 means Pauli energy is
    identically zero at every r). Only gsc_lambda sweeps 1.0->0.0, via
    set_gsc_lambda() (manual-control mode is correct here).

    Yields
    ------
    List of (stage, combined_lambda, gsc_lambda) tuples.
        'decouple': call set_electrostatic_lambda/set_vdw_lambda(combined_lambda),
        do NOT call set_gsc_lambda (gsc_lambda is None here on purpose).
        'remove_gsc': only call set_gsc_lambda(gsc_lambda); electrostatic/vdw
        lambda are already 0.0 from the end of step 1.
    """
    windows = [('decouple', lam, None) for lam in _linspace(1.0, 0.0, n_decouple)]
    windows += [('remove_gsc', 0.0, lam) for lam in _linspace(1.0, 0.0, n_remove_gsc)]
    return windows


def three_stage_gsc_windows(
    n_discharge: int = 6, n_decouple_gsc: int = 6, n_remove_gsc: int = 2
) -> List[Tuple[str, float]]:
    """
    Three-stage discharge/decouple-with-GSC/remove-GSC protocol. Matches
    the standard "decharge-vdW-recharge" staged strategy used for RBFE
    mutations, adapted for annihilation (no final species to recharge --
    the ion just vanishes) with GSC providing softcore protection during
    vdW removal instead of CSC/Beutler-style reparameterization.

    Stage 1 ("discharge"): electrostatic_lambda 1->0. vdw_lambda held at
    1.0 (full physical) the entire stage -- never call set_vdw_lambda here.
    GSC's automatic coupling naturally gives alpha_gsc=alpha_max*(1-1.0)=0
    throughout, correctly inert (the real wall doesn't need help yet).

    Stage 2 ("decouple_gsc"): electrostatic_lambda held at 0.0 (from the
    end of stage 1 -- never call set_electrostatic_lambda here). vdw_lambda
    1->0. GSC's DEFAULT automatic coupling (alpha_max*(1-vdw_lambda), via
    set_vdw_lambda -- do NOT call set_gsc_lambda here) handles the bump.
    Mechanically identical to mbucb_style_windows()'s "growth" stage, with
    the addition of GSC as extra protection -- electrostatics is already
    exactly zero throughout, so there is no attractive pull toward the
    weakening wall, only ordinary thermal diffusion.

    Stage 3 ("remove_gsc"): electrostatic_lambda and vdw_lambda both fixed
    at 0.0. Only gsc_lambda sweeps 1->0, via set_gsc_lambda() (manual
    control -- correct here, identical to gsc_two_step_windows()'s own
    stage 2). Real Pauli is already IDENTICALLY zero at every r by this
    point (not just weak), so there's no divergence left to protect
    against -- this stage should converge with very few windows.

    Yields
    ------
    List of (stage, lambda) tuples.
        'discharge': call set_electrostatic_lambda(lambda). Do not touch
        vdw_lambda or call set_gsc_lambda.
        'decouple_gsc': call set_vdw_lambda(lambda). Do not touch
        electrostatic_lambda or call set_gsc_lambda.
        'remove_gsc': call set_gsc_lambda(lambda). Do not touch
        electrostatic_lambda or vdw_lambda.
    Caller must call set_electrostatic_lambda(0.0) once after stage 1
    finishes and before stage 2 begins is NOT needed -- stage 1's last
    window already leaves electrostatic_lambda at 0.0.
    """
    windows = [('discharge', lam) for lam in _linspace(1.0, 0.0, n_discharge)]
    windows += [('decouple_gsc', lam) for lam in _linspace(1.0, 0.0, n_decouple_gsc)]
    windows += [('remove_gsc', lam) for lam in _linspace(1.0, 0.0, n_remove_gsc)]
    return windows


class IonAlchemicalScaler:
    """
    Scales a single ion's CMM parameters for alchemical hydration free
    energy calculations, via two independent buckets (electrostatic, vdw)
    rather than one combined lambda.
    """

    def __init__(
        self,
        parametrizers: Dict[str, Parametrizer],
        ion_type: str,
        solvent_types: List[str],
    ):
        self.p = parametrizers
        self.ion_type = ion_type
        self.solvent_types = list(solvent_types)
        self._electrostatic_lambda = 1.0
        self._vdw_lambda = 1.0

        def _cache_atomic(specs):
            base = {}
            for pname, param in specs:
                parametrizer = self.p[pname]
                idx = self._type_index(parametrizer, self.ion_type)
                base[(pname, param)] = parametrizer.params[param][idx].clone()
            return base

        self._base_electrostatic_atomic = _cache_atomic(_ELECTROSTATIC_ATOMIC_SPECS)
        self._base_vdw_atomic = _cache_atomic(_VDW_ATOMIC_SPECS)

        # Not every (ion, solvent_type) pair has an eps_ct entry; missing
        # pairs mean zero contribution and are skipped, not an error.
        ct = self.p['ChargeTransfer']
        self._base_eps_ct = {}
        for stype in self.solvent_types:
            try:
                self._base_eps_ct[stype] = self._lookup_specific_pair(
                    ct, 'eps_ct', self.ion_type, stype
                ).clone()
            except KeyError:
                print(
                    f"[IonAlchemicalScaler] No eps_ct entry for "
                    f"('{self.ion_type}', '{stype}') -- treating as zero, "
                    f"nothing to scale for this pair."
                )

        # No C6_disp override exists by default, so compute the physical
        # pairwise value ourselves via the same geometric-mean rule.
        disp = self.p['Dispersion']
        c6_atomic = disp.params['C6_disp']
        ion_idx = self._type_index(disp, self.ion_type)
        self._base_c6_ij = {
            stype: torch.sqrt(c6_atomic[ion_idx] * c6_atomic[self._type_index(disp, stype)]).clone()
            for stype in self.solvent_types
        }

        self._gsc_enabled = False
        self._vdw_floor_enabled = True
        self._polarization_floor_enabled = True

    @staticmethod
    def find_ion_atom_index(top, ion_type: str) -> int:
        """Flat atom index of a monatomic ion in a Topology. Raises unless exactly one match."""
        matches = [i for i, t in enumerate(top.atomTypes) if t == ion_type]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one atom of type '{ion_type}' in the topology, "
                f"found {len(matches)}: {matches}. IonAlchemicalScaler/GSC assume "
                f"a single ion; if you have multiple, this needs a different design."
            )
        return matches[0]

    def enable_gsc(
        self,
        top,
        alpha_max: float,
        beta: float,
        r_min: float,
        X: int = 4,
        ion_atom_index: int = None,
    ):
        """
        Enable the GSC protective bump for this ion's Pauli decoupling.
        alpha_max in Hartree, sized against real Pauli energy at
        practically-reachable close distances. beta/r_min fixed
        (lambda-independent) per Li & Nam's PI-architecture guidance.
        """
        self._gsc_enabled = True
        self._gsc_natoms = top.natoms
        self._gsc_ion_atom_index = (
            ion_atom_index if ion_atom_index is not None
            else self.find_ion_atom_index(top, self.ion_type)
        )
        self._gsc_alpha_max = float(alpha_max)
        self._gsc_beta = float(beta)
        self._gsc_r_min = float(r_min)
        self._gsc_X = int(X)
        self._gsc_current_alpha = gsc_alpha_schedule(self._vdw_lambda, self._gsc_alpha_max)
        self._gsc_manual_control = False

        reference_tensor = self.p['Pauli'].params['q_pauli']
        self._gsc_device = reference_tensor.device
        self._gsc_dtype = reference_tensor.dtype

    def set_gsc_lambda(self, lam: float):
        """
        Manually set the bump's own lambda (1.0=full alpha_max, 0.0=off),
        decoupling it from vdw_lambda's automatic (1-vdw_lambda) schedule.

        Use for gsc_two_step_windows()'s Step 2 ONLY. Calling this during
        Step 1 pins the bump at full strength even at lambda=1.0 (where it
        should be ~0) and creates an artificial energy trap -- see
        gsc_two_step_windows()'s docstring for the full derivation.

        Requires enable_gsc() first.
        """
        if not self._gsc_enabled:
            raise RuntimeError("Call enable_gsc() before set_gsc_lambda().")
        self._gsc_manual_control = True
        # Direct mapping (lam=1.0 -> alpha_max), NOT gsc_alpha_schedule's
        # inverted formula -- that's for the automatic vdw_lambda coupling only.
        self._gsc_current_alpha = float(lam) * self._gsc_alpha_max

    def disable_vdw_floor(self):
        """Remove _VDW_LAMBDA_FLOOR, allowing q_pauli to reach exact zero. Only after GSC is validated."""
        self._vdw_floor_enabled = False

    def disable_polarization_floor(self):
        """
        Remove _ALPHA_ETA_LAMBDA_FLOOR, allowing alpha/eta to reach exact
        zero. Different failure mode than the vdw floor (singular matrix,
        not a geometric collapse) -- GSC does not protect against this.
        Requires polarization.py's Tikhonov-regularization patch first.
        """
        self._polarization_floor_enabled = False

    @staticmethod
    def _type_index(parametrizer: Parametrizer, type_str: str) -> int:
        """
        Row index for `type_str`. Some parametrizers key by composite
        "type/kz/kx/ky" strings; this falls back to matching the bare type
        before the first '/'.
        """
        if type_str in parametrizer.typesAsDict:
            return parametrizer.typesAsDict[type_str]
        matches = [
            idx for key, idx in parametrizer.typesAsDict.items()
            if key.split('/')[0] == type_str
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise KeyError(
                f"Ambiguous type '{type_str}' in parametrizer '{parametrizer.name}': "
                f"matches multiple entries ({matches}). For a monatomic ion "
                f"this shouldn't happen -- check for duplicate registrations."
            )
        raise KeyError(
            f"Type '{type_str}' not found in parametrizer '{parametrizer.name}' "
            f"typesAsDict keys: {list(parametrizer.typesAsDict.keys())}"
        )

    def _pair_key(self, parametrizer: Parametrizer, type1: str, type2: str) -> torch.Tensor:
        i = self._type_index(parametrizer, type1)
        j = self._type_index(parametrizer, type2)
        pair = torch.tensor([[i, j]], device=parametrizer.device)
        return symmetric_pairing_function(pair)

    def _lookup_specific_pair(self, parametrizer: Parametrizer, name: str, type1: str, type2: str) -> torch.Tensor:
        key = self._pair_key(parametrizer, type1, type2)
        indices = torch.atleast_1d(parametrizer.specific_pair_param_indices[name])
        values = torch.atleast_1d(parametrizer.specific_pair_params[name])
        mask = indices == key
        if not mask.any():
            raise KeyError(
                f"No specific-pair entry for '{name}' between types "
                f"'{type1}' and '{type2}' in parametrizer '{parametrizer.name}'."
            )
        return values[mask]

    def _write_specific_pair(self, parametrizer: Parametrizer, name: str, type1: str, type2: str, value: torch.Tensor):
        key = torch.atleast_1d(self._pair_key(parametrizer, type1, type2))
        value = torch.atleast_1d(value)
        if name in parametrizer.specific_pair_param_indices:
            indices = torch.atleast_1d(parametrizer.specific_pair_param_indices[name])
            values = torch.atleast_1d(parametrizer.specific_pair_params[name])
            mask = indices == key
            if mask.any():
                values[mask] = value
                parametrizer.specific_pair_params[name] = values
                parametrizer.specific_pair_param_indices[name] = indices
                return
            parametrizer.specific_pair_param_indices[name] = torch.cat([indices, key])
            parametrizer.specific_pair_params[name] = torch.cat([values, value])
        else:
            parametrizer.specific_pair_param_indices[name] = key
            parametrizer.specific_pair_params[name] = value

    def set_electrostatic_lambda(self, lam: float):
        """
        Set electrostatics/polarization/CT/xpol to `lam` (0=off, 1=full).
        Does not touch q_pauli/C6_disp. alpha/eta floored near zero unless
        disable_polarization_floor() was called.
        """
        self._electrostatic_lambda = float(lam)
        if self._polarization_floor_enabled:
            alpha_eta_lam = max(self._electrostatic_lambda, _ALPHA_ETA_LAMBDA_FLOOR)
        else:
            alpha_eta_lam = self._electrostatic_lambda

        for (pname, param), base_val in self._base_electrostatic_atomic.items():
            parametrizer = self.p[pname]
            idx = self._type_index(parametrizer, self.ion_type)
            if pname == 'Polarization':
                parametrizer.params[param][idx] = alpha_eta_lam * base_val
            else:
                parametrizer.params[param][idx] = self._electrostatic_lambda * base_val

        ct = self.p['ChargeTransfer']
        for stype, base_val in self._base_eps_ct.items():
            self._write_specific_pair(ct, 'eps_ct', self.ion_type, stype, self._electrostatic_lambda * base_val)

    def set_vdw_lambda(self, lam: float):
        """
        Set Pauli/dispersion to `lam` (0=off, 1=full). q_pauli floored near
        zero unless disable_vdw_floor() was called.

        If GSC is enabled and set_gsc_lambda() has never been called, the
        bump auto-tracks this via gsc_alpha_schedule(). Once
        set_gsc_lambda() is called, that coupling is disabled permanently.
        """
        self._vdw_lambda = float(lam)
        if self._vdw_floor_enabled:
            repulsive_lam = max(self._vdw_lambda, _VDW_LAMBDA_FLOOR)
        else:
            repulsive_lam = self._vdw_lambda

        for (pname, param), base_val in self._base_vdw_atomic.items():
            parametrizer = self.p[pname]
            idx = self._type_index(parametrizer, self.ion_type)
            parametrizer.params[param][idx] = repulsive_lam * base_val

        if self._gsc_enabled and not self._gsc_manual_control:
            self._gsc_current_alpha = gsc_alpha_schedule(self._vdw_lambda, self._gsc_alpha_max)

        disp = self.p['Dispersion']
        for stype, base_val in self._base_c6_ij.items():
            self._write_specific_pair(disp, 'C6_disp', self.ion_type, stype, self._vdw_lambda * base_val)

    def set_discharge_lambda(self, lam: float):
        """Alias for set_electrostatic_lambda()."""
        self.set_electrostatic_lambda(lam)

    def set_decouple_lambda(self, lam: float):
        """Alias for set_vdw_lambda()."""
        self.set_vdw_lambda(lam)

    @property
    def electrostatic_lambda(self) -> float:
        return self._electrostatic_lambda

    @property
    def vdw_lambda(self) -> float:
        return self._vdw_lambda

    @property
    def discharge_lambda(self) -> float:
        return self._electrostatic_lambda

    @property
    def decouple_lambda(self) -> float:
        return self._vdw_lambda

    @property
    def affected_parametrizers(self) -> List[str]:
        """Parametrizer names needing expandParameters() re-run after a lambda change."""
        return list(_AFFECTED_PARAMETRIZERS)

    def expand_affected(self, system=None):
        """
        Re-runs expandParameters() on affected parametrizers. If `system`
        is given, also refreshes c6_mean and (if GSC enabled) pushes
        gsc_alpha_atomic/beta/r_min/X onto it for System.getEnergy().
        """
        for name in self.affected_parametrizers:
            self.p[name].expandParameters()

        if system is not None and getattr(system, 'use_lr_dispersion', False) and system._has_nb:
            system.c6_mean = torch.mean(
                system.parametrizers['Dispersion'].getExpandParameters("C6_disp", system.all_pairs)
            )

        if system is not None and self._gsc_enabled:
            if (not hasattr(system, 'gsc_alpha_atomic')
                    or system.gsc_alpha_atomic.shape[0] != self._gsc_natoms):
                system.gsc_alpha_atomic = torch.zeros(
                    self._gsc_natoms, device=self._gsc_device, dtype=self._gsc_dtype
                )
            else:
                system.gsc_alpha_atomic.zero_()
            system.gsc_alpha_atomic[self._gsc_ion_atom_index] = self._gsc_current_alpha
            system.gsc_beta = self._gsc_beta
            system.gsc_r_min = self._gsc_r_min
            system.gsc_X = self._gsc_X