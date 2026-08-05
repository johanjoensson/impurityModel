"""The double counting must enter the solved Hamiltonian exactly once.

RSPt hands the solver the raw Kohn-Sham block plus ``sig_dc`` and expects back the *pure
interaction* self-energy; it applies the double counting to the lattice itself, once, in
``double_counting_apply`` (``green_double_counting.F90:596``). The external CTHYB path makes
the convention explicit: RSPt subtracts ``sig_dc`` from ``hlda`` before calling that solver
(``green_solver.F90:578-582``) and *still* subtracts ``sig_dc`` from ``sig`` afterwards.

impurityModel solves ``H = h0 - dc + U`` (``solver_basis.prepare_solver_basis``). The
self-energy it returns is therefore only the pure interaction self-energy if it is extracted
against the non-interacting part of *that* Hamiltonian, i.e. against ``h0 - dc`` -- not
against the raw ``h0``. The two tests below pin that from opposite directions:

* :func:`test_sigma_is_invariant_under_moving_a_shift_between_h0_and_dc` -- two models with
  the *same* ``h0 - dc`` are the same physical problem and must give the same ``Sigma``.
* :func:`test_sigma_static_carries_no_double_counting` -- the high-frequency limit of
  ``Sigma`` is the Hartree-Fock self-energy of the interaction alone, so it cannot depend on
  ``dc`` at all.

Both fail by exactly the double-counting shift when ``Sigma`` is defined against the raw
``h0``.
"""

import contextlib
import io

import numpy as np
import pytest

from impurityModel.ed import atomic_physics
from impurityModel.ed.model import BasisOptions, ImpurityModel, Meshes, SolverOptions
from impurityModel.ed.selfenergy import calc_selfenergy

EPS_D, EPS_B, V, U = -0.7, 1.5, 0.3, 3.0


def _u4_operator():
    u4 = np.zeros((2, 2, 2, 2))
    u4[0, 1, 0, 1] = u4[1, 0, 1, 0] = U
    return atomic_physics.getUop_from_rspt_u4(u4)


def _model(*, imp_shift=0.0, dc_shift=None):
    """The end-to-end SIAM of ``test_selfenergy_end_to_end``, with the impurity level shifted
    by ``imp_shift`` and a uniform double counting of ``dc_shift`` on the impurity block."""
    h0 = {
        ((0, "c"), (0, "a")): EPS_D + imp_shift,
        ((1, "c"), (1, "a")): EPS_D + imp_shift,
        ((2, "c"), (2, "a")): EPS_B,
        ((3, "c"), (3, "a")): EPS_B,
        ((0, "c"), (2, "a")): V,
        ((2, "c"), (0, "a")): V,
        ((1, "c"), (3, "a")): V,
        ((3, "c"), (1, "a")): V,
    }
    dc = None
    if dc_shift is not None:
        dc = {((0, "c"), (0, "a")): dc_shift, ((1, "c"), (1, "a")): dc_shift}
    return ImpurityModel(
        h0=h0,
        dc=dc,
        u4=_u4_operator(),
        impurity_orbitals={0: [0, 1]},
        rot_to_spherical=np.eye(2),
    )


def _meshes():
    iw = 1j * np.pi * 0.5 * (2 * np.arange(200) + 1)
    w = np.linspace(-4.0, 4.0, 21)
    return Meshes(iw=iw, w=w, delta=0.2)


def _run(model):
    basis = BasisOptions(
        nominal_occ={0: 1},
        mixed_valence=False,
        dN=None,
        truncation_threshold=np.inf,
        chain_restrict=False,
        spin_flip_dj=False,
        occ_cutoff=1e-10,
        slater_weight_min=1e-10,
        tau=0.05,
    )
    solver = SolverOptions(reort=None, dense_cutoff=500, sparse_green=True, gf_method="lanczos")
    with contextlib.redirect_stdout(io.StringIO()):
        return calc_selfenergy(model, _meshes(), basis, solver, comm=None, verbosity=0)


@pytest.mark.parametrize("c", [0.0, 1.25])
def test_sigma_is_invariant_under_moving_a_shift_between_h0_and_dc(c):
    """``(h0, dc=0)`` and ``(h0 + c on the impurity, dc = c)`` are the *same* Hamiltonian.

    ``H = h0 - dc + U`` is bit-identical between the two, so every eigenvalue, the ground
    state, the Green's function and hence the self-energy must agree. A ``Sigma`` extracted
    against the raw ``h0`` instead differs by exactly ``c`` on the impurity block -- the
    double counting leaking into the returned self-energy, which RSPt then subtracts a second
    time.
    """
    reference = _run(_model())
    shifted = _run(_model(imp_shift=c, dc_shift=c))

    for key in ("sigma", "sigma_real", "sigma_static"):
        np.testing.assert_allclose(
            shifted[key],
            reference[key],
            atol=1e-8,
            err_msg=f"{key} moved when a uniform shift c={c} was moved from h0 into dc",
        )


def test_sigma_static_carries_no_double_counting():
    """``Sigma(omega -> inf)`` is the Hartree-Fock self-energy of the *interaction*.

    It is a contraction of ``u4`` with the impurity density matrix and contains no one-body
    term, so it cannot depend on ``dc``. Two models differing only in ``dc`` (with ``h0``
    compensated so the solved Hamiltonian is unchanged) must return the same ``sigma_static``;
    with the double counting leaking into ``Sigma`` they differ by the shift.
    """
    c = 0.8
    no_dc = _run(_model())["sigma_static"]
    with_dc = _run(_model(imp_shift=c, dc_shift=c))["sigma_static"]

    offset = np.asarray(with_dc) - np.asarray(no_dc)
    assert np.max(np.abs(offset)) < 1e-8, (
        "sigma_static shifted by "
        f"{np.diagonal(offset, axis1=-2, axis2=-1)} when the double counting changed by {c}; "
        "the high-frequency self-energy must be the pure Hartree-Fock interaction term."
    )
