import os
import torch
import pytest

torch.set_default_dtype(torch.float64)

from cmm.dipole_saturation import (
    unit_field_axis,
    saturation_energy,
    saturation_gradient,
    saturation_hessian,
    saturation_secant_coeffs,
    saturation_secant_matrix,
)

DATA = os.path.join(os.path.dirname(__file__), 'data')


def _random_inputs(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(n, 3, generator=g) * 0.8
    alpha = torch.rand(n, generator=g) * 30 + 1.0
    n_axis = unit_field_axis(torch.randn(n, 3, generator=g))
    c_iso = torch.rand(n, generator=g) * 3
    c_ani = torch.rand(n, generator=g) * 0.2
    w = torch.rand(n, generator=g) * 1.5 + 0.5
    return mu, alpha, n_axis, c_iso, c_ani, w


def test_gradient_matches_energy():
    mu, alpha, n_axis, c_iso, c_ani, w = _random_inputs()
    grad = saturation_gradient(mu, alpha, n_axis, c_iso, c_ani, w)
    h = 1e-6
    for k in range(3):
        d = torch.zeros(3)
        d[k] = h
        ep = saturation_energy(mu + d, alpha, n_axis, c_iso, c_ani, w)
        em = saturation_energy(mu - d, alpha, n_axis, c_iso, c_ani, w)
        fd = (ep - em) / (2 * h)
        assert torch.allclose(grad[:, k], fd, atol=1e-8), f"component {k}"


def test_hessian_matches_gradient():
    mu, alpha, n_axis, c_iso, c_ani, w = _random_inputs(seed=1)
    hess = saturation_hessian(mu, alpha, n_axis, c_iso, c_ani, w)
    h = 1e-6
    for k in range(3):
        d = torch.zeros(3)
        d[k] = h
        gp = saturation_gradient(mu + d, alpha, n_axis, c_iso, c_ani, w)
        gm = saturation_gradient(mu - d, alpha, n_axis, c_iso, c_ani, w)
        fd = (gp - gm) / (2 * h)
        assert torch.allclose(hess[:, :, k], fd, atol=1e-7), f"column {k}"


def test_secant_identity():
    # grad U_sat(mu) == K(mu) mu by construction
    mu, alpha, n_axis, c_iso, c_ani, w = _random_inputs(seed=2)
    grad = saturation_gradient(mu, alpha, n_axis, c_iso, c_ani, w)
    K = saturation_secant_matrix(mu, alpha, n_axis, c_iso, c_ani, w)
    assert torch.allclose(grad, torch.bmm(K, mu.unsqueeze(-1)).squeeze(-1), atol=1e-12)


def test_zero_dipole_limit():
    # U_sat, its gradient, and its full Hessian vanish as mu -> 0, so the
    # free-ion linear response mu = alpha E is recovered exactly.
    _, alpha, n_axis, c_iso, c_ani, w = _random_inputs(seed=3)
    mu = torch.zeros(6, 3)
    assert torch.allclose(saturation_energy(mu, alpha, n_axis, c_iso, c_ani, w), torch.zeros(6))
    assert torch.allclose(saturation_gradient(mu, alpha, n_axis, c_iso, c_ani, w), mu)
    hess = saturation_hessian(mu, alpha, n_axis, c_iso, c_ani, w)
    assert torch.allclose(hess, torch.zeros(6, 3, 3), atol=1e-12)


def test_parallel_perpendicular_signs():
    # Perpendicular stiffness (U_iso'/m) is added monotonically (damping);
    # the axial term softens at small dipole (U_ani'' < 0) and hardens at
    # large dipole (U_ani'' > 0).
    alpha = torch.tensor([30.0])
    n_axis = torch.tensor([[1.0, 0.0, 0.0]])
    c_iso = torch.tensor([2.0])
    c_ani = torch.tensor([0.2])
    w = torch.tensor([1.0])
    m0 = alpha * 0.05

    # U_ani'' < 0 for v/w < 0.90 (softening) and > 0 for 0.90 < v/w < 1.92
    small = (0.3 * m0).view(1, 1) * n_axis
    large = (1.3 * m0).view(1, 1) * n_axis

    h_small = saturation_hessian(small, alpha, n_axis, c_iso, c_ani, w)
    h_large = saturation_hessian(large, alpha, n_axis, c_iso, c_ani, w)

    # perpendicular (yy) component grows with dipole and is >= 0
    assert h_small[0, 1, 1] >= 0
    assert h_large[0, 1, 1] > h_small[0, 1, 1]

    # isolate the anisotropic axial contribution
    h_small_iso_only = saturation_hessian(small, alpha, n_axis, c_iso, torch.zeros(1), w)
    h_large_iso_only = saturation_hessian(large, alpha, n_axis, c_iso, torch.zeros(1), w)
    ani_small = h_small[0, 0, 0] - h_small_iso_only[0, 0, 0]
    ani_large = h_large[0, 0, 0] - h_large_iso_only[0, 0, 0]
    assert ani_small < 0  # parallel polarizability rises at intermediate field
    assert ani_large > 0  # then falls at large field


def test_spd_floor():
    # even for aggressive c_ani, the secant parallel eigenvalue stays positive
    mu, alpha, n_axis, _, _, w = _random_inputs(seed=4)
    c_iso = torch.zeros(6)
    c_ani = torch.full((6,), 50.0)
    k_iso, k_ani = saturation_secant_coeffs(mu * 0.01, alpha, n_axis, c_iso, c_ani, w)
    par_eig = 1.0 / alpha + k_iso + k_ani
    assert torch.all(par_eig > 0)


def test_batched_zero_saturation_regression():
    # with sat_c_iso = sat_c_ani = 0 the solver must reproduce the plain
    # (undamped) linear-response energies, and the use_dipole_saturation flag
    # must have no effect
    app = pytest.importorskip('openmm.app')
    from cmm.ffxml import ForceFieldXML
    from cmm.topology import Topology
    import cmm.units as units

    pdb = app.PDBFile(os.path.join(DATA, 'water_dimer.pdb'))
    coords = torch.tensor(
        (pdb.getPositions(asNumpy=True)._value / units.BOHR2NM).tolist()
    ).unsqueeze(0)

    # water.xml has no sat_* attributes -> defaults (0, 0, 1)
    ff = ForceFieldXML(os.path.join(DATA, 'water.xml'), device='cpu', float_dtype=torch.float64)
    top = Topology.fromOpenmm(pdb.topology, 'cpu')

    e_on = ff.parametrize(top, batch=True, use_fd_morse=False).getEnergy(coords, energy_in_kcal=True)
    e_off = ff.parametrize(top, batch=True, use_fd_morse=False, use_dipole_saturation=False).getEnergy(coords, energy_in_kcal=True)
    assert torch.allclose(e_on['pol'], e_off['pol'], atol=1e-12)
    assert torch.allclose(e_on['total'], e_off['total'], atol=1e-12)


def test_uniform_field_saturation_sublinear():
    # a single saturating ion in a uniform external field develops a dipole
    # that grows sublinearly, while the zero-saturation ion stays linear
    app = pytest.importorskip('openmm.app')
    from cmm.ffxml import ForceFieldXML
    from cmm.topology import Topology

    scans = os.environ.get(
        'CMM_DATA', os.path.join(os.path.dirname(__file__), '..', '..', 'CMM_Data')
    )
    pdb_file = os.path.join(scans, 'ion_water', 'ion_ion_scans', 'na_cl_scan.pdb')
    if not os.path.exists(pdb_file):
        pytest.skip('CMM_Data not available')

    pdb = app.PDBFile(pdb_file)
    ff = ForceFieldXML(
        os.path.join(os.path.dirname(__file__), '..', 'scripts', 'ion_water_refit.xml'),
        device='cpu', float_dtype=torch.float64
    )
    top = Topology.fromOpenmm(pdb.topology, 'cpu')
    system = ff.parametrize(top, batch=True)

    # ions far apart so the response is dominated by the external field
    coords = torch.tensor([[[0.0, 0.0, 0.0], [80.0, 0.0, 0.0]]])

    def induced(scale):
        res = system.getEnergy(coords, ext_field=torch.tensor([0.0, 0.0, scale]))
        return res['induced_molecular_dipole'][0, 2].item()

    e1, e2 = 0.02, 0.04
    p1, p2 = induced(e1), induced(e2)
    assert p1 > 0
    assert p2 / p1 < 2.0 * 0.999  # sublinear growth for the saturating Cl-
