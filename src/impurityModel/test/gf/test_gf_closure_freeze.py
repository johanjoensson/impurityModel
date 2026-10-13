r"""``GF_CLOSURE_FREEZE``: switch a unit to the frozen ``P H P`` CSR once a whole window admits no new row.

Only a unit that hit its cap used to switch; one whose support stalled below the cap ran the sparse kernel
to the end (CrI3 at ``GS_MEMORY_BUDGET_SAFETY=0.75``: six units stalled at ~5.14M under a 6.69M cap, 2.0-2.7
ks each, against 230-300 s on the CSR). The contract these tests pin:

* on by default, and it freezes on the first window that admits nothing -- not earlier (a window that
  still grew), not later;
* the window is ``_CLOSURE_WINDOW`` (``GF_STAGNATION_WINDOW`` only under ``GF_STAGNATION_FREEZE``);
* a stall is not closure: with a cutoff it can freeze short of the sector, and the frozen ``G`` is then no
  farther from the exact one than the unfrozen sparse ``G`` (which cuts sub-cutoff amplitudes inside ``P``
  too);
* it is taken near the unit's tolerance too (no near-end veto), with ``reort='none'`` only;
* a CSR that does not fit is declined once and leaves the result untouched; ``0`` turns it off;
* alongside ``GF_STAGNATION_FREEZE``, whichever fires first wins and is reported as such;
* every rank takes the same decision (``-n 2``/``-n 3``).
"""

import functools
import itertools
import re

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import gf_solvers
from impurityModel.ed.basis_transcription import build_dense_matrix
from impurityModel.ed.greens_function import calc_G
from impurityModel.test.basis.test_sparse_matrix_build import _basis, _det
from impurityModel.test.gf.test_gf_stagnation_freeze import DELTA, N_ORB, OMEGA, _operator, _run, _seeds

SECTOR = 3432  # the fixtures' whole sector: 7 electrons in 14 orbitals


@pytest.fixture(autouse=True)
def _hermetic_knobs(monkeypatch):
    for name in (
        "GF_CLOSURE_FREEZE",
        "GF_STAGNATION_FREEZE",
        "GF_STAGNATION_WINDOW",
        "GF_FROZEN_CSR",
        "GF_APPLY_ROW_CHUNKS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(gf_solvers, "_CLOSURE_WINDOW", 4)  # the fixtures stall within a few dozen blocks


def _g(result):
    alphas, betas, r, _ = result
    return calc_G(alphas, betas, r, OMEGA, 0.0, DELTA)


@functools.lru_cache(maxsize=None)
def _spectrum(keys, seed):
    """Eigen-decomposition of ``H`` on the span of ``keys`` (a tuple of determinant bytes) and the seeds there."""
    from impurityModel.ed.ManyBodyUtils import SlaterDeterminant

    basis = _basis(sorted(SlaterDeterminant.from_bytes(k) for k in keys), None, n_orb=N_ORB)
    h = np.asarray(build_dense_matrix(basis, _operator(N_ORB, seed=seed)))
    index = {bytes(d.to_bytearray()[:2]): i for i, d in enumerate(basis.local_basis)}
    v = np.zeros((h.shape[0], len(_seeds())), dtype=complex)
    for j, s in enumerate(_seeds()):
        for det, amp in s.items():
            v[index[bytes(det.to_bytearray()[:2])], j] = amp[0]
    # The fixture's H is real: a real eigh is ~10x cheaper than the complex one (6 s against 63 s here).
    assert not np.any(h.imag), "premise: a real Hamiltonian"
    e, u = np.linalg.eigh(h.real)
    return e, u.conj().T @ v


def _dense_g(keys, seed=3):
    """G(w) = V^dag ((w + i delta) - H)^{-1} V on the span of ``keys``, by one eigh (not one solve per w)."""
    e, uv = _spectrum(tuple(sorted(keys)), seed)
    return np.einsum("ki,wk,kj->wij", uv.conj(), 1.0 / ((OMEGA[:, None] + 1j * DELTA) - e[None, :]), uv)


def _key_bytes(proxy):
    return [bytes(k.to_bytearray()[:2]) for k in proxy.retained_keys()]


def _sector():
    return [bytes(_det(list(c)).to_bytearray()[:2]) for c in itertools.combinations(range(N_ORB), 7)]


def _support_lines(out):
    """``(blocks, growth %)`` per logged window, and the index of the window the closure freeze followed."""
    windows, frozen_after = [], None
    for line in out.splitlines():
        m = re.match(r"GF support: [0-9,]+ determinants after (\d+) blocks \(\+([0-9.]+)%", line)
        if m:
            windows.append((int(m.group(1)), float(m.group(2))))
        elif "on closure" in line and frozen_after is None:
            frozen_after = len(windows) - 1
    return windows, frozen_after


def test_a_closed_support_freezes_by_default_and_gives_the_dense_g():
    frozen = _run()
    info = frozen[3]
    assert info["closure_frozen"] and info["csr_fallback"], "on by default"
    assert not info["cap_hit"] and not info["stagnation_frozen"]
    assert info["retained_size"] == SECTOR
    np.testing.assert_allclose(_g(frozen), _dense_g(_key_bytes(info["proxy"])), atol=1e-8)
    assert "GF_CLOSURE_FREEZE" in info["proxy"].freeze_message()


def test_it_freezes_on_the_first_window_that_admits_nothing(capsys):
    """Not on a window that still grew (by however few rows), and not a window later."""
    _run(verbose=True)
    windows, frozen_after = _support_lines(capsys.readouterr().out)
    assert frozen_after is not None, "the closure freeze was not logged"
    assert windows[frozen_after][1] == 0.0, "it froze after a window that still admitted rows"
    assert all(g > 0.0 for _, g in windows[:frozen_after]), "an earlier window admitted nothing and was passed over"


def test_the_window_is_the_closure_window(monkeypatch, capsys):
    monkeypatch.setattr(gf_solvers, "_CLOSURE_WINDOW", 7)
    _run(verbose=True)
    windows, _ = _support_lines(capsys.readouterr().out)
    assert windows and all(b % 7 == 0 for b, _ in windows), windows


def test_it_matches_the_sparse_run_it_replaces(monkeypatch):
    frozen = _g(_run())
    monkeypatch.setenv("GF_CLOSURE_FREEZE", "0")
    plain = _run()
    assert not plain[3]["closure_frozen"] and not plain[3]["csr_fallback"]
    np.testing.assert_allclose(frozen, _g(plain), atol=1e-8)


def test_a_stall_is_not_closure_but_costs_no_more_than_the_cutoff_does(monkeypatch):
    """With a cutoff the support can stand still for a window and then grow again: the 16-block window
    freezes a few determinants short of the sector. The sparse kernel cuts sub-cutoff amplitudes inside
    ``P`` as well, so the frozen ``G`` is no farther from the exact one than the unfrozen sparse ``G``."""
    monkeypatch.setattr(gf_solvers, "_CLOSURE_WINDOW", 16)
    op = _operator(N_ORB, seed=5)
    frozen = _run(op=op, slater_min=1e-4)
    info = frozen[3]
    assert info["closure_frozen"]
    assert info["retained_size"] < SECTOR, "premise: the support stalled before closing"
    monkeypatch.setenv("GF_CLOSURE_FREEZE", "0")
    plain = _run(op=op, slater_min=1e-4)
    exact = _dense_g(_sector(), seed=5)
    err_frozen = np.abs(_g(frozen) - exact).max()
    err_plain = np.abs(_g(plain) - exact).max()
    assert err_plain > 1e-7, "premise: the cutoff costs the sparse kernel something"
    assert err_frozen <= 1.5 * err_plain, (err_frozen, err_plain)


def test_it_is_taken_near_the_tolerance_too(monkeypatch):
    monkeypatch.setattr(gf_solvers, "_STAGNATION_NEAR_END", 1e30)
    assert _run()[3]["closure_frozen"]


def test_off_switch(monkeypatch):
    monkeypatch.setenv("GF_CLOSURE_FREEZE", "0")
    info = _run()[3]
    assert not info["closure_frozen"] and not info["csr_fallback"]


def test_a_declined_csr_leaves_the_result_untouched_and_is_asked_once(monkeypatch):
    monkeypatch.setenv("GF_CLOSURE_FREEZE", "0")
    plain = _run()
    monkeypatch.delenv("GF_CLOSURE_FREEZE")
    asked = []

    def declining(*args, **kwargs):
        asked.append(1)
        return False

    monkeypatch.setattr(gf_solvers, "_frozen_csr_fits", declining)
    declined = _run()
    assert len(asked) == 1, "a declined switch is not retried window after window"
    assert not declined[3]["closure_frozen"] and not declined[3]["csr_fallback"]
    np.testing.assert_array_equal(plain[0], declined[0])
    np.testing.assert_array_equal(plain[1], declined[1])


def test_it_needs_reort_none():
    assert not _run(reort="full")[3]["closure_frozen"]


def test_with_the_stagnation_freeze_whichever_fires_first_wins(monkeypatch):
    monkeypatch.setenv("GF_STAGNATION_WINDOW", "4")
    # A threshold no window meets: only a window that admits nothing can freeze, and closure needs one.
    monkeypatch.setenv("GF_STAGNATION_FREEZE", "1e-300")
    info = _run()[3]
    assert info["closure_frozen"] and not info["stagnation_frozen"]
    # A threshold every window meets: stagnation freezes after two windows, while the support still grows.
    monkeypatch.setenv("GF_STAGNATION_FREEZE", "2")
    info = _run()[3]
    assert info["stagnation_frozen"] and not info["closure_frozen"] and info["retained_size"] < SECTOR


@pytest.mark.mpi
def test_every_rank_takes_the_same_decision_and_matches_the_dense_oracle():
    comm = MPI.COMM_WORLD
    result = _run(comm=comm)
    info = result[3]
    assert info["closure_frozen"] and info["csr_fallback"]
    assert len(set(comm.allgather(info["retained_size"]))) == 1
    all_keys = [k for rank_keys in comm.allgather(_key_bytes(info["proxy"])) for k in rank_keys]
    assert len(all_keys) == info["retained_size"]
    np.testing.assert_allclose(_g(result), _dense_g(all_keys), atol=1e-8)
