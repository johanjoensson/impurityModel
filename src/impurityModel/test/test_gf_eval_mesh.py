r"""The Green's-function convergence monitor must test ``G`` where the caller evaluates it.

`_make_gf_convergence_monitor`'s original mesh came from `_gf_converged_mesh`: the resolved Ritz
band, on the line :math:`\omega + i\delta`. It therefore converged the **real-axis** resolvent at
broadening ``delta`` whether or not a real-axis mesh had been requested, and over the whole band
rather than the window the caller asked about. A Matsubara point :math:`i\omega_n` sits a distance
:math:`\sqrt{E_k^2 + \omega_n^2}` from every pole while a real-axis point can come within ``delta``
of one, so a Matsubara-only self-energy was paying for a resolvent it never evaluates.

`_gf_eval_meshes` reproduces exactly what `calc_G` will be handed --
:math:`\omega_P = \pm\omega + i(\pm\delta) + e` per thermal eigenstate, ``delta = 0`` on the
Matsubara axis -- and the monitor tests there instead.

Two properties these tests exist to protect, both of which a naive implementation gets wrong:

* **Cost.** The production real mesh is ~2000 points and the Matsubara one ~375. The monitor's
  continued-fraction rebuild is the single largest cost of the recurrence (~53% for reort=NONE),
  so the meshes must be subsampled to `_GF_MONITOR_POINTS` per axis, not passed verbatim.
* **One axis must not mask the other.** The measure is relative,
  :math:`\max|\Delta G| / \max|G|`. At :math:`T = 0.002`, :math:`|G| \sim 1/\pi T \approx 159` on
  Matsubara against :math:`1/\delta = 5` on the real axis, so a single concatenated mesh would
  divide the real-axis change by the Matsubara peak and declare the real axis converged more than
  an order of magnitude early. The axes are kept separate and the monitor takes the max.
"""

import numpy as np
import scipy.linalg as la

from impurityModel.ed.BlockLanczosArray import Reort, block_lanczos_array
from impurityModel.ed.greens_function import (
    _GF_MONITOR_POINTS,
    _GF_REL_TOL_FLOOR,
    _gf_eval_meshes,
    _gf_rel_tol,
    _make_gf_convergence_monitor,
    build_qr,
    calc_G,
)

_DELTA = 0.2
_TAU = 0.002


def _matsubara(n, tau=_TAU):
    return 1j * (2 * np.arange(n) + 1) * np.pi * tau


# --------------------------------------------------------------------------------------
# _gf_eval_meshes reproduces calc_G's frame exactly
# --------------------------------------------------------------------------------------


def test_eval_mesh_matches_calc_G_for_the_addition_side():
    """`calc_G(..., mesh, e, delta)` forms ``mesh + 1j*delta + e``; side 0 passes ``(+w, +delta)``."""
    w = np.linspace(-1.5, 1.5, 7)
    (mesh,) = _gf_eval_meshes(None, w, side_i=0, delta=_DELTA, es=[3.5])
    np.testing.assert_allclose(mesh, w + 1j * _DELTA + 3.5)


def test_eval_mesh_matches_calc_G_for_the_removal_side():
    """Removal passes ``(-w, -delta)``, so both the mesh and the broadening flip sign."""
    w = np.linspace(-1.5, 1.5, 7)
    (mesh,) = _gf_eval_meshes(None, w, side_i=1, delta=_DELTA, es=[3.5])
    np.testing.assert_allclose(mesh, -w - 1j * _DELTA + 3.5)


def test_matsubara_axis_carries_no_broadening():
    """`calc_thermally_averaged_G` is called with ``delta = 0`` on the Matsubara axis."""
    iw = _matsubara(5)
    (mesh,) = _gf_eval_meshes(iw, None, side_i=0, delta=_DELTA, es=[0.0])
    np.testing.assert_allclose(mesh, iw)
    assert np.all(mesh.imag > 0), "Im(z) must keep the sign of the unit's signed delta"

    (mesh_rem,) = _gf_eval_meshes(iw, None, side_i=1, delta=_DELTA, es=[0.0])
    np.testing.assert_allclose(mesh_rem, -iw)
    assert np.all(mesh_rem.imag < 0)


def test_every_stacked_eigenstate_is_shifted_in():
    """A unit stacks several thermal states; each shifts the mesh by its own energy."""
    w = np.linspace(-1.0, 1.0, 4)
    es = [1.0, 2.0, 3.0]
    (mesh,) = _gf_eval_meshes(None, w, side_i=0, delta=_DELTA, es=es)
    assert mesh.size == len(w) * len(es)
    for k, e in enumerate(es):
        np.testing.assert_allclose(mesh[k * len(w) : (k + 1) * len(w)], w + 1j * _DELTA + e)


def test_no_mesh_falls_back_to_the_spectral_edge_monitor():
    assert _gf_eval_meshes(None, None, side_i=0, delta=_DELTA, es=[0.0]) is None


# --------------------------------------------------------------------------------------
# The two properties a naive implementation loses
# --------------------------------------------------------------------------------------


def test_the_callers_mesh_is_subsampled_not_passed_verbatim():
    """A 2000-point real mesh must not become a 2000-point continued-fraction evaluation."""
    w = np.linspace(-1.83, 1.83, 2000)
    iw = _matsubara(375)
    meshes = _gf_eval_meshes(iw, w, side_i=0, delta=_DELTA, es=[0.0])
    assert len(meshes) == 2, "one array per requested axis"
    for mesh in meshes:
        assert mesh.size <= _GF_MONITOR_POINTS, f"{mesh.size} points would dominate the recurrence"


def test_axes_stay_separate_so_neither_can_mask_the_other():
    """Concatenating them would divide the real-axis change by the Matsubara peak."""
    w = np.linspace(-1.83, 1.83, 32)
    iw = _matsubara(32)
    meshes = _gf_eval_meshes(iw, w, side_i=0, delta=_DELTA, es=[0.0])
    assert len(meshes) == 2
    mats, real = meshes
    # |G| ~ 1/|Im z| sets each axis's scale. They differ by well over an order of magnitude, which
    # is exactly how much a shared denominator would under-converge the real axis.
    mats_scale = 1.0 / np.min(np.abs(mats.imag))
    real_scale = 1.0 / np.min(np.abs(real.imag))
    assert np.allclose(np.abs(real.imag), _DELTA)
    assert mats_scale > 10.0 * real_scale, f"|G| scales {mats_scale:.1f} vs {real_scale:.1f}"


def test_a_matsubara_only_run_stops_far_earlier_and_stays_exact():
    """The production effect, in miniature: a Matsubara-only run must not pay for the real axis.

    The spectrum lives on ``[1, 50]``. Every Matsubara point ``i*w_n`` is then at least distance 1
    from every pole, so ``G(i w_n)`` is smooth and converges in a handful of blocks. A real-axis
    point at broadening ``delta = 0.2`` can come within ``0.2`` of a pole, so the band-wide monitor
    keeps building. The saving must be real *and* free: ``G`` on the Matsubara mesh has to match a
    fully-converged reference, or the early stop is premature convergence wearing a mesh.
    """
    rng = np.random.default_rng(4)
    n = 300
    d = np.linspace(1.0, 50.0, n)
    u = la.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))[0]
    h = (u * d) @ np.conj(u.T)
    h = 0.5 * (h + np.conj(h.T))
    seed = rng.standard_normal((n, 2)) + 1j * rng.standard_normal((n, 2))
    q0, r = build_qr(seed)

    iw = _matsubara(32)
    eval_meshes = _gf_eval_meshes(iw, None, side_i=0, delta=_DELTA, es=[0.0])

    def run(monitor):
        a, b, _q, _widths = block_lanczos_array(
            psi0=q0, h_op=h, converged=monitor, reort=Reort.FULL, verbose=False, return_widths=True
        )
        return a, b

    matsubara_aware, _f1, _d1, _dg1 = _make_gf_convergence_monitor(_DELTA, 0.0, eval_meshes=eval_meshes)
    band_wide, _f2, _d2, _dg2 = _make_gf_convergence_monitor(_DELTA, 0.0)

    a_m, b_m = run(matsubara_aware)
    a_w, _b_w = run(band_wide)
    a_exact, b_exact = run(lambda *args, **kwargs: False)  # exhaust the Krylov space

    assert len(a_m) < len(a_w), (
        f"the Matsubara-aware monitor used {len(a_m)} blocks against the band-wide monitor's "
        f"{len(a_w)}; it is no longer saving anything"
    )

    # The Matsubara Green's function it stopped on is the converged one. `delta = 0` on this axis.
    g_early = calc_G(a_m, b_m, r, iw, 0.0, 0.0)
    g_exact = calc_G(a_exact, b_exact, r, iw, 0.0, 0.0)
    err = np.max(np.abs(g_early - g_exact)) / np.max(np.abs(g_exact))
    assert err < 100 * _gf_rel_tol(0.0), f"premature convergence: G(iw) differs from exact by {err:.3e}"


def test_the_declared_tolerance_is_the_one_that_is_applied():
    """`_gf_rel_tol` is the single source of truth for the monitor and the diagnostic summary.

    It is `max(slaterWeightMin**2, _GF_REL_TOL_FLOOR)`, so the floor -- not the cutoff -- governs
    every production slaterWeightMin (1e-5 gives 1e-10, well under the floor). Only a cutoff
    looser than `sqrt(_GF_REL_TOL_FLOOR)` ever overrides it.
    """
    assert _gf_rel_tol(0.0) == _GF_REL_TOL_FLOOR
    assert _gf_rel_tol(1e-5) == _GF_REL_TOL_FLOOR  # 1e-10 < floor
    assert _gf_rel_tol(1e-12) == _GF_REL_TOL_FLOOR
    # A cutoff above sqrt(floor) does override it: the basis truncation, not the monitor, is then
    # the accuracy limit.
    loose = 10 * np.sqrt(_GF_REL_TOL_FLOOR)
    assert _gf_rel_tol(loose) == loose**2 > _GF_REL_TOL_FLOOR
