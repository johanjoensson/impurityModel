r"""The unpruned growth of a sparse GF unit: with no cutoff it closes the sector, at a cap it freezes.

(The support settle that used to stop a trickling unit early is gone: pruned growth, ``gf_prune_tol``, is the
default depth filter, and an unpruned unit grows until it closes or hits the cap.)
"""

import itertools

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.greens_function import calc_G
from impurityModel.test.basis.test_sparse_matrix_build import _det
from impurityModel.test.support.gf_chain_fixtures import (
    DELTA,
    N_ORB,
    OMEGA,
    SECTOR,
    exact_chain_g,
    run_chain,
)


@pytest.fixture(autouse=True)
def _hermetic_knobs(monkeypatch):
    for name in ("GF_FROZEN_CSR", "APPLY_ROW_CHUNKS", "MATVEC_PRUNE"):
        monkeypatch.delenv(name, raising=False)


def _g(result):
    alphas, betas, r, _ = result
    return calc_G(alphas, betas, r, OMEGA, 0.0, DELTA)


def _exact_sector_g(seed=5):
    return exact_chain_g([_det(list(c)) for c in itertools.combinations(range(N_ORB), 7)], seed=seed)


def test_with_no_cutoff_the_unit_grows_to_the_closed_sector_and_gives_the_exact_g():
    result = run_chain()
    assert result[3]["retained_size"] == SECTOR
    np.testing.assert_allclose(_g(result), _exact_sector_g(seed=3), atol=1e-8)


def test_a_cap_below_the_sector_freezes_the_unit_and_says_so():
    info = run_chain(cap=2600)[3]
    assert info["cap_hit"] and info["retained_size"] <= 2600


@pytest.mark.mpi
def test_the_rank_count_moves_neither_the_support_nor_g():
    comm = MPI.COMM_WORLD
    result = run_chain(comm=comm)
    serial = run_chain()
    assert result[3]["retained_size"] == serial[3]["retained_size"] == SECTOR
    mine = (result[3]["retained_size"], len(result[0]))
    assert len(set(comm.allgather(mine))) == 1
    np.testing.assert_allclose(_g(result), _g(serial), atol=1e-6)
