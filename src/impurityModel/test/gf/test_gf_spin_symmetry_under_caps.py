"""G_up == G_down when the Green's-function support caps bind (review step 7', GF-side SYM measurement).

The ground state of a paramagnetic SU(2) model is a singlet, so the impurity Green's function is spin-diagonal
and equal in the two spins. The support caps cut each unit separately: with the spins in separate blocks the
(c^dagger_up, psi) and (c^dagger_dn, psi) units are spin images of each other, may sit in different MPI
colours, and each is cut by its own ``freeze_at_cap`` (tie-break: ``collective_first_keys``, determinant-key
order, which a spin flip does not preserve). This pins the measured outcome: they stay equal to roundoff.
If a cut ever breaks the pairing this fails; the fix is review step 10c.
"""

import numpy as np
import pytest

from impurityModel.ed.greens_function import get_Greens_function
from impurityModel.ed.ManyBodyUtils import ManyBodyState
from impurityModel.test.support.aim_fixtures import build_semicircle_siam
from impurityModel.test.support.comms import COMMS

NB = 7  # 16 spin-orbitals, a 4,900-determinant singlet; the removal/addition sectors hold 3,920
CAP = 2500  # below the 3,920 each unit reaches: every unit freezes, and G moves ~5e-5 (3000 moves it 8e-11)


def _green(comm, cap, method):
    aim = build_semicircle_siam(NB)
    basis = aim.basis(sorted(aim.gs.keys()), truncation_threshold=cap, comm=comm)
    gs = ManyBodyState(dict(aim.gs.items()) if comm is None or comm.rank == 0 else {}, width=1)
    if comm is not None:
        gs = basis.redistribute_psis(gs)[0]
    mats = 1j * np.pi * 0.4 * (2 * np.arange(6) + 1)
    omega = np.linspace(-1, 1, 7) + 0.05j
    mat, real, _ = get_Greens_function(
        matsubara_mesh=mats,
        omega_mesh=omega,
        psis=[gs],
        es=[aim.e0],
        tau=1e-3,
        basis=basis,
        hOp=aim.hOp,
        delta=0.05,
        blocks=[[0], [1]],
        verbose=False,
        verbose_extra=False,
        reort=True if method == "lanczos" else None,
        dN=None,
        occ_cutoff=0.0,
        slaterWeightMin=0.0,
        sparse=True,
        gf_method=method,
    )
    return mat, real


@pytest.mark.parametrize("comm", COMMS)
@pytest.mark.parametrize("method", ["lanczos"])
def test_spin_partner_units_cut_at_the_cap_give_equal_green_functions(comm, method):
    mat, real = _green(comm, CAP, method)
    full, _ = _green(comm, np.inf, method)  # collective: every rank, before any early return
    # Only the ranks that assemble the answer hold it; on those, a missing block is a failure.
    if comm is not None and comm.rank != 0:
        return
    assert mat is not None and mat[0] is not None and mat[1] is not None
    tol = 1e-7 * max(np.abs(mat[0]).max(), 1e-300)
    for up, dn in ((mat[0], mat[1]), (real[0], real[1])):
        assert np.abs(up[:, 0, 0] - dn[:, 0, 0]).max() < tol
    # The test is only sensitive if the cut moves G far above the tolerance: a cut of one spin alone would show.
    assert max(np.abs(mat[s][:, 0, 0] - full[s][:, 0, 0]).max() for s in (0, 1)) > 100 * tol
