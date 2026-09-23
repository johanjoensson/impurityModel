"""``GF_APPLY_ROW_CHUNKS``: the row-chunked matvec of the sparse block-Lanczos GF recurrence
(``_lanczos_step.pxi``'s ``wp = h_op.apply_block(q_curr, ...)``).

Mirrors ``GS_APPLY_ROW_CHUNKS`` (``CIPSISolver._apply_block_and_redistribute``,
``test_cipsi_apply_chunking.py``) at the GF unit's own matvec: bound the raw apply output,
packed send buffer and receive buffer to one chunk of ``q_curr``'s rows at a time instead of
the whole block (``doc/plans/dc_smo_memory.md``, "GF unit memory").

Unlike the CIPSI selection round, this sum feeds a block-Lanczos recurrence directly, so the
oracle here is not bit-identical agreement with the one-shot path but the same *strong* oracle
``test_gf_truncation.py`` already holds the one-shot path to: a capped recurrence's continued
fraction must equal the exact dense resolvent of ``H`` projected onto whatever it actually
retained. That invariant does not care which row order a candidate is discovered in, so it is
the right test for a change that is explicitly "exact up to summation order." A run whose
recurrence never binds the cap (so no boundary tie-break exists at all) is additionally checked
for tight numerical agreement against the unchunked run.

The knob is registered under ``group="units"`` and its unset default (4, chunked -- on since
2026-09-14) is covered generically by ``test_config.py``'s ``test_defaults_when_unset``; this
file's job is the chunked *behaviour*, comparing explicitly against ``n_chunks=1`` (the
one-shot path) as the baseline throughout rather than relying on "unset" to mean one-shot.
"""

import numpy as np
import pytest

from impurityModel.ed import config
from impurityModel.ed.basis_transcription import build_dense_matrix
from impurityModel.ed.gf_solvers import block_Green_sparse
from impurityModel.ed.greens_function import calc_G
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant

DELTA = 0.1
OMEGA = np.linspace(-8.0, 8.0, 41)

_IMP = {0: [[0, 1]]}
_BATHS = ({0: [[2, 3]]}, {0: [[4, 5]]})


@pytest.fixture(autouse=True)
def _knob_unset(monkeypatch):
    monkeypatch.delenv("GF_APPLY_ROW_CHUNKS", raising=False)


def _det(occupied):
    """Determinant with the given orbitals occupied (MSB-first: orbital i = bit 7-i)."""
    b = 0
    for i in occupied:
        b |= 1 << (7 - i)
    return SlaterDeterminant.from_bytes(bytes([b]))


def _siam_6():
    """Single-impurity Anderson model, 6 spin-orbitals (0,1 imp; 2,3 val; 4,5 cond) -- the
    same toy model test_gf_truncation.py's oracle tests use."""
    ed_, u, ev, ec, v = -1.0, 4.0, -3.0, 3.0, 0.5
    terms = {}
    for o in (0, 1):
        terms[((o, "c"), (o, "a"))] = ed_
    for o in (2, 3):
        terms[((o, "c"), (o, "a"))] = ev
    for o in (4, 5):
        terms[((o, "c"), (o, "a"))] = ec
    terms[((0, "c"), (1, "c"), (1, "a"), (0, "a"))] = u
    for a, b in ((0, 2), (1, 3), (0, 4), (1, 5)):
        terms[((a, "c"), (b, "a"))] = v
        terms[((b, "c"), (a, "a"))] = v
    return ManyBodyOperator(terms)


def _seeds():
    """Two seed columns in the N=3 sector (the reachable space is the full sector, 20 dets)."""
    return [
        ManyBodyState({_det([0, 2, 3]): 1.0 + 0j, _det([1, 2, 3]): 0.5 + 0j}),
        ManyBodyState({_det([0, 1, 2]): 1.0 + 0j}),
    ]


def _excited_basis(cap):
    seed_support = sorted({state for s in _seeds() for state in s})
    return Basis(_IMP, _BATHS, initial_basis=seed_support, truncation_threshold=cap, verbose=False)


def _run(cap, n_chunks, monkeypatch):
    """``n_chunks=None`` leaves the knob unset (today's default, 4 -- chunked); pass ``1``
    explicitly for the one-shot baseline rather than relying on "unset" to mean that."""
    if n_chunks is None:
        monkeypatch.delenv("GF_APPLY_ROW_CHUNKS", raising=False)
        assert config.GF_APPLY_ROW_CHUNKS.get() == 4, "unset must be the measured default of 4 chunks"
    else:
        monkeypatch.setenv("GF_APPLY_ROW_CHUNKS", str(n_chunks))
        assert config.GF_APPLY_ROW_CHUNKS.get() == n_chunks
    basis = _excited_basis(cap)
    seeds = [ManyBodyState.from_states([s]).to_states()[0] for s in _seeds()]
    info = {}
    alphas, betas, r = block_Green_sparse(_siam_6(), seeds, basis, DELTA, verbose=False, cap_info=info)
    return calc_G(alphas, betas, r, OMEGA, 0.0, DELTA), info


def _dense_reference_on(retained_keys):
    """G(w) = V^dag ((w + i*delta) - H)^{-1} V on the space spanned by retained_keys."""
    basis = Basis(_IMP, _BATHS, initial_basis=sorted(retained_keys), verbose=False)
    H = np.asarray(build_dense_matrix(basis, _siam_6()))
    index = {det: i for i, det in enumerate(sorted(retained_keys))}
    V = np.zeros((len(index), len(_seeds())), dtype=complex)
    for j, seed in enumerate(_seeds()):
        for det, amp in seed.items():
            V[index[det], j] = amp[0]
    G = np.empty((len(OMEGA), V.shape[1], V.shape[1]), dtype=complex)
    for k, w in enumerate(OMEGA):
        G[k] = V.conj().T @ np.linalg.solve((w + 1j * DELTA) * np.eye(len(index)) - H, V)
    return G


@pytest.mark.parametrize("n_chunks", [2, 3, 4, 8])
def test_chunked_matches_one_shot_above_the_reachable_space(n_chunks, monkeypatch):
    """A cap the recurrence never reaches has no admission boundary to perturb: chunking
    must reproduce the one-shot result to numerical precision, not just the same physics."""
    g_one_shot, info_one_shot = _run(1000, 1, monkeypatch)
    g_chunked, info_chunked = _run(1000, n_chunks, monkeypatch)
    assert not info_one_shot["cap_hit"] and not info_chunked["cap_hit"]
    np.testing.assert_allclose(g_chunked, g_one_shot, rtol=1e-10, atol=1e-12)


def test_unset_default_is_four_chunks_and_matches_explicit_four(monkeypatch):
    """The registry default (4, on since 2026-09-14) must actually be what an unset run gets,
    not just what the Knob declares -- pin it against the explicit spelling too."""
    g_unset, info_unset = _run(1000, None, monkeypatch)
    g_explicit, info_explicit = _run(1000, 4, monkeypatch)
    assert not info_unset["cap_hit"] and not info_explicit["cap_hit"]
    np.testing.assert_array_equal(g_unset, g_explicit)


@pytest.mark.parametrize("cap", [6, 12, 17])
@pytest.mark.parametrize("n_chunks", [2, 4])
def test_capped_gf_equals_dense_php_resolvent_with_row_chunking(cap, n_chunks, monkeypatch):
    """The strong oracle (test_gf_truncation.py) must survive chunking under a binding cap:
    whatever the chunked matvec's summation order admits, the result must still be the exact
    GF of H projected on the retained set -- the boundary tie-break may differ from the
    one-shot path, but the recurrence's exactness on whatever it retained may not.

    No reort axis: chunking changes only how the matvec is summed, and the reort x cap grid is
    already the oracle in test_gf_truncation.py::test_capped_gf_equals_dense_php_resolvent."""
    g, info = _run(cap, n_chunks, monkeypatch)
    assert info["cap_hit"]
    assert info["retained_size"] <= cap
    retained = info["proxy"].retained_keys()
    assert len(retained) == info["retained_size"]
    np.testing.assert_allclose(g, _dense_reference_on(retained), atol=1e-9)
    assert np.all(np.diagonal(g.imag, axis1=1, axis2=2) <= 1e-12)  # causality


def test_more_chunks_than_rows_does_not_crash(monkeypatch):
    """A chunk count larger than the row count (empty chunks) must be handled, not desync."""
    g_one_shot, _ = _run(1000, 1, monkeypatch)
    g_many_chunks, _ = _run(1000, 64, monkeypatch)
    np.testing.assert_allclose(g_many_chunks, g_one_shot, rtol=1e-10, atol=1e-12)
