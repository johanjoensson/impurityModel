"""TRLM refuses Ritz vectors that cannot be an orthonormal block, and says which input was damaged."""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import trlm
from impurityModel.ed.trlm import CorruptRitzVectors, _ritz_vectors


def _inputs(n=12, d=6, k=3, seed=0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(n, d)) + 1j * rng.normal(size=(n, d)))
    y, _ = np.linalg.qr(rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)))
    return np.ascontiguousarray(q), np.ascontiguousarray(y[:, :k])


def test_sound_vectors_pass_through_unchanged():
    q, y = _inputs()
    assert np.array_equal(_ritz_vectors(q, y, "extract", None), q @ y)


def test_an_empty_rank_is_sound():
    q, y = _inputs()
    assert _ritz_vectors(q[:0], y, "extract", None).shape == (0, 3)


@pytest.mark.parametrize("value", [np.nan, np.inf, 1e200, 3.0])
def test_a_wrong_product_from_sound_inputs_blames_the_product(monkeypatch, value):
    q, y = _inputs()

    def damaged(Q, Y, cutoff):
        out = Q @ Y
        out[1, 2] = value
        return out

    monkeypatch.setattr(trlm, "block_combine", damaged)
    with pytest.raises(CorruptRitzVectors, match="the matrix product returned wrong values"):
        _ritz_vectors(q, y, "extract", None)


def test_a_damaged_basis_is_named():
    q, y = _inputs()
    q[4, 1] = 1e180
    with pytest.raises(CorruptRitzVectors, match="rows of the Lanczos basis were already damaged"):
        _ritz_vectors(q, y, "extract", None)


def test_a_damaged_rotation_is_named():
    q, y = _inputs()
    y[2, 0] = np.nan
    with pytest.raises(CorruptRitzVectors, match="copy of the rotation was already damaged"):
        _ritz_vectors(q, y, "extract", None)


@pytest.mark.mpi
def test_damage_on_one_rank_raises_on_every_rank(monkeypatch):
    comm = MPI.COMM_WORLD
    q, y = _inputs(seed=comm.rank)
    if comm.rank == comm.size - 1:
        monkeypatch.setattr(trlm, "block_combine", lambda Q, Y, cutoff: np.full((Q.shape[0], Y.shape[1]), np.inf + 0j))
    expected = (
        "damaged Ritz vectors on rank" if comm.rank == comm.size - 1 else "another rank holds damaged Ritz vectors"
    )
    with pytest.raises(CorruptRitzVectors, match=expected):
        _ritz_vectors(q, y, "extract", comm)
