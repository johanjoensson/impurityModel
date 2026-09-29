"""The GF must not leave its excited-sector mask on the caller's Hamiltonian (ledger row C3).

``ManyBodyOperator.set_restrictions`` is sticky on the operator object. ``_block_green_group``
installs each unit's excited window on the shared ``hOp`` and never restores it, and
``calc_selfenergy`` then hands the *same* ``h`` to ``get_greens_function_moments``
(``selfenergy.py``), whose moments are documented as exact. With a confining window the
moments are computed with ``P H P`` instead of ``H``: ``M_1`` survives (one application of H on
the seed stays in the window) but ``M_2``, ``M_3`` -- and so ``sigma_moment_1``/``_2`` -- are
truncated.
"""

import contextlib
import io

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.greens_function import get_Greens_function, get_greens_function_moments
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyState
from impurityModel.test.support.gf_branch_oracle import _det, two_orbital_model

C3 = "C3: sticky excited-sector mask left on hOp (doc/reviews/gf_review.md)"


def _setup():
    hOp, imp, baths, n_orb, _n0, blocks = two_orbital_model(1)
    (val,) = baths[0][0]
    # One determinant: impurity half filled on orbitals 1 and 3, valence bath (4 and 6, which
    # hybridize with the *empty* impurity orbitals 0 and 2) filled, conduction empty. The excited
    # window derived from it (dN=1) confines the impurity to 1..3 electrons, so on the seed
    # c_0^dag|psi> (3 impurity electrons) a single hop 6 -> 2 already leaves the window: the mask
    # binds at the first H application, which the second moment sees.
    occupied = [1, 3, *val]
    psi = ManyBodyState({_det(occupied, n_orb): 1.0})
    basis = Basis(imp, baths, initial_basis=list(psi.keys()), comm=MPI.COMM_SELF, verbose=False)
    return hOp, basis, [psi], blocks


def _moments(hOp, basis, psis):
    return get_greens_function_moments(psis, [0.0], 1.0, basis, hOp, [0, 1, 2, 3], max_order=3)


@pytest.mark.xfail(strict=True, reason=C3)
def test_moments_after_a_windowed_gf_equal_moments_before_it():
    hOp, basis, psis, blocks = _setup()
    before = _moments(hOp, basis, psis)
    with contextlib.redirect_stdout(io.StringIO()):
        get_Greens_function(
            matsubara_mesh=1j * np.pi * (2 * np.arange(4) + 1),
            omega_mesh=None,
            psis=psis,
            es=[0.0],
            tau=1.0,
            basis=basis,
            hOp=hOp,
            delta=0.1,
            blocks=blocks,
            verbose=False,
            verbose_extra=False,
            reort=None,
            dN=1,
            occ_cutoff=1e-12,
            slaterWeightMin=0.0,
            sparse=True,
        )
    after = _moments(hOp, basis, psis)
    np.testing.assert_allclose(after[1], before[1], atol=1e-12)  # one H application: unaffected
    np.testing.assert_allclose(after, before, atol=1e-12)
