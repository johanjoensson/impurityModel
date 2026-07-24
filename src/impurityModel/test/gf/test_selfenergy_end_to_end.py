"""End-to-end (unmocked) tests for calc_selfenergy (selfenergy.py:310).

test_selfenergy.py's orchestration tests all mock calc_gs / get_Greens_function /
get_sigma, so only the plumbing is exercised -- none of them runs a real many-body
solve. This runs calc_selfenergy for real on a minimal interacting SIAM (one impurity
orbital pair with a Hubbard U, hybridising with one bath level per spin), checking two
properties that only an unmocked run can exercise: causality (Im Sigma(z) <= 0 on both
the real and Matsubara axes, for a physical self-energy) and that the returned
high-frequency moments (sigma_static, sigma_moment_1, sigma_moment_2) genuinely
reconstruct the tail of the frequency-resolved sigma the solver actually computed, not
just an internally-consistent but disconnected pair of numbers.
"""

import contextlib
import io

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.model import BasisOptions, ImpurityModel, Meshes, SolverOptions
from impurityModel.ed.selfenergy import calc_selfenergy


def _siam_model():
    """One impurity orbital pair (Hubbard U) + one bath level per spin -- small enough to
    solve exactly (truncation_threshold=inf, dN=None), but with genuine hybridization so
    the self-energy is dynamic (not just the static Hartree-Fock term)."""
    eps_d, eps_b, v, U = -0.7, 1.5, 0.3, 3.0
    h0 = {
        ((0, "c"), (0, "a")): eps_d,
        ((1, "c"), (1, "a")): eps_d,
        ((2, "c"), (2, "a")): eps_b,
        ((3, "c"), (3, "a")): eps_b,
        ((0, "c"), (2, "a")): v,
        ((2, "c"), (0, "a")): v,
        ((1, "c"), (3, "a")): v,
        ((3, "c"), (1, "a")): v,
    }
    u4 = np.zeros((2, 2, 2, 2))
    u4[0, 1, 0, 1] = u4[1, 0, 1, 0] = U
    return ImpurityModel(h0=h0, u4=u4, impurity_orbitals={0: [0, 1]}, rot_to_spherical=np.eye(2))


def _meshes():
    iw = 1j * np.pi * 0.5 * (2 * np.arange(400) + 1)  # extends far enough for the tail check below
    w = np.linspace(-4.0, 4.0, 41)
    return Meshes(iw=iw, w=w, delta=0.2)


def _basis_and_solver():
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
    return basis, solver


def _run(comm):
    basis, solver = _basis_and_solver()
    with contextlib.redirect_stdout(io.StringIO()):
        return calc_selfenergy(_siam_model(), _meshes(), basis, solver, comm=comm, verbosity=0)


def _assert_causal_and_tail_consistent(res, iw):
    # Causality: Im Sigma <= 0 on both axes (physical retarded self-energy convention).
    assert np.all(np.diagonal(res["sigma_real"], axis1=1, axis2=2).imag <= 1e-9)
    assert np.all(np.diagonal(res["sigma"], axis1=1, axis2=2).imag <= 1e-9)

    # High-frequency tail: Sigma(z) ~ sigma_static + sigma_moment_1/z + sigma_moment_2/z^2
    # at large |z|. Checked against the highest Matsubara point actually computed by the
    # resolvent solve -- an independent numerical route from the Lehmann-moment formula
    # that produced sigma_static/sigma_moment_1/sigma_moment_2, not a re-derivation of it.
    z = iw[-1]
    reconstructed_tail = res["sigma_static"] + res["sigma_moment_1"] / z + res["sigma_moment_2"] / z**2
    np.testing.assert_allclose(reconstructed_tail, res["sigma"][-1], atol=1e-6)


def test_calc_selfenergy_end_to_end_causal_and_tail_consistent():
    res = _run(MPI.COMM_SELF)
    _assert_causal_and_tail_consistent(res, _meshes().iw)


@pytest.mark.mpi
def test_calc_selfenergy_end_to_end_mpi_matches_serial():
    """The distributed run must reproduce the serial (COMM_SELF) answer exactly -- this
    model is the one production, unmocked consumer of get_greens_function_moments
    (selfenergy.py:564), so it exercises the whole distributed GF/moments pipeline
    together, not just one function in isolation."""
    comm = MPI.COMM_WORLD
    res_distributed = _run(comm)

    # sigma/sigma_real/the moments are root-only under real distribution (calc_selfenergy
    # inherits this from get_Greens_function's root-only gs_matsubara/gs_realaxis), so both
    # the physicality/tail checks and the serial comparison only make sense on rank 0.
    if comm.rank == 0:
        _assert_causal_and_tail_consistent(res_distributed, _meshes().iw)
        res_serial = _run(MPI.COMM_SELF)
        np.testing.assert_allclose(res_distributed["sigma"], res_serial["sigma"], atol=1e-10)
        np.testing.assert_allclose(res_distributed["sigma_real"], res_serial["sigma_real"], atol=1e-10)
        np.testing.assert_allclose(res_distributed["sigma_static"], res_serial["sigma_static"], atol=1e-10)
        np.testing.assert_allclose(res_distributed["sigma_moment_1"], res_serial["sigma_moment_1"], atol=1e-10)
        np.testing.assert_allclose(res_distributed["sigma_moment_2"], res_serial["sigma_moment_2"], atol=1e-10)
