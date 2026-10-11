r"""``GF_STAGNATION_FREEZE``: switch a unit to the frozen ``P H P`` CSR when the weight reaching new rows dies away.

A unit that converges below its determinant cap never freezes, so it ran the sparse kernel to the end
(SrMnO3 cubic, unit 10: 11.4M determinants, 509 blocks, 14 ks on 3 ranks). The knob measures, per window of
blocks, the fraction of each matvec's squared norm that lands outside the retained set -- what a freeze
drops -- and freezes once it has stayed below the threshold for two windows. It counts weight, not
determinants. The contract these tests pin:

* running the recurrence a window at a time changes nothing (bit-identical to one round);
* a stagnation freeze is exactly the cap freeze with ``P`` = the support reached: ``G`` equals the dense
  resolvent of ``H`` projected on the retained determinants;
* the criterion is the weight: a support that keeps growing by rows carrying ~1e-10 of the amplitude
  freezes, and the number of determinants alone does not decide;
* two quiet windows are needed, and a unit within two decades of its tolerance is left alone;
* a CSR that does not fit is declined, asked once, and leaves the result untouched;
* it applies to ``reort='none'`` only, and says so when it cannot apply;
* every rank takes the same decision, empty ranks included (``-n 3``).
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import gf_solvers
from impurityModel.ed.basis_transcription import build_dense_matrix
from impurityModel.ed.gf_diagnostics import Severity, check_basis_truncation, check_stagnation_freeze
from impurityModel.ed.gf_solvers import block_Green_sparse
from impurityModel.ed.greens_function import calc_G
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState
from impurityModel.test.basis.test_sparse_matrix_build import _basis, _det
from impurityModel.test.gf.test_gf_truncation import _redistribute_as_width1

N_ORB = 14
DELTA = 0.1
OMEGA = np.linspace(-6.0, 6.0, 31)
CAP = 10**9  # finite, so the capped proxy is installed; far above anything these recurrences reach


@pytest.fixture(autouse=True)
def _hermetic_knobs(monkeypatch):
    for name in ("GF_STAGNATION_FREEZE", "GF_STAGNATION_WINDOW", "GF_FROZEN_CSR", "GF_APPLY_ROW_CHUNKS"):
        monkeypatch.delenv(name, raising=False)


def _operator(n_orb=N_ORB, seed=3, strong=None, weak=1e-5):
    """A chain with nearest-neighbour hopping and an on-site interaction: the Krylov support spreads a
    shell or two per block, so it is still growing after several windows (a dense random hopping matrix
    saturates its whole sector in eight blocks). With ``strong`` set, orbitals ``0..strong-1`` hop at the
    full strength and the bond into the rest has strength ``weak``: the support keeps growing through the
    weak bond, by rows that carry ~``weak**2`` of the amplitude."""
    rng = np.random.default_rng(seed)
    terms = {((o, "c"), (o, "a")): float(rng.normal()) for o in range(n_orb)}
    for a in range(n_orb - 1):
        v = 0.4 + 0.1 * float(rng.normal())
        if strong is not None and a == strong - 1:
            v = weak
        terms[((a, "c"), (a + 1, "a"))] = v
        terms[((a + 1, "c"), (a, "a"))] = v
    for a in range(0, n_orb - 1, 2):
        terms[((a, "c"), (a + 1, "c"), (a + 1, "a"), (a, "a"))] = 2.0
    return ManyBodyOperator(terms)


def _seeds():
    return [
        ManyBodyState({_det([0, 1, 2, 3, 4, 5, 6]): 1.0 + 0j, _det([0, 1, 2, 3, 4, 5, 7]): 0.5 + 0j}),
        ManyBodyState({_det([1, 2, 3, 4, 5, 6, 7]): 1.0 + 0j}),
    ]


def _run(comm=None, op=None, reort=None, verbose=False, cap=CAP, slater_min=0.0):
    basis = _basis(sorted({d for s in _seeds() for d in s}), comm, n_orb=N_ORB)
    basis.truncation_threshold = cap
    full = _seeds()
    seeds = _redistribute_as_width1(basis, full if comm is None or comm.rank == 0 else None, len(full))
    info = {}
    alphas, betas, r = block_Green_sparse(
        _operator(N_ORB) if op is None else op,
        seeds,
        basis,
        DELTA,
        reort=reort,
        verbose=verbose,
        cap_info=info,
        slaterWeightMin=slater_min,
    )
    return alphas, betas, r, info


def _dense_g(retained_keys, comm=None, op=None):
    """G(w) = V^dag ((w + i delta) - H)^{-1} V on the span of ``retained_keys``."""
    basis = _basis(sorted(retained_keys), comm, n_orb=N_ORB)
    h = np.asarray(build_dense_matrix(basis, _operator(N_ORB) if op is None else op))
    index = {bytes(d.to_bytearray()[:2]): i for i, d in enumerate(basis.local_basis)}
    v = np.zeros((h.shape[0], len(_seeds())), dtype=complex)
    for j, seed in enumerate(_seeds()):
        for det, amp in seed.items():
            v[index[bytes(det.to_bytearray()[:2])], j] = amp[0]
    g = np.empty((len(OMEGA), v.shape[1], v.shape[1]), dtype=complex)
    for k, w in enumerate(OMEGA):
        g[k] = v.conj().T @ np.linalg.solve((w + 1j * DELTA) * np.eye(h.shape[0]) - h, v)
    return g


def test_a_windowed_recurrence_is_bit_identical_to_one_round(monkeypatch):
    plain = _run()
    monkeypatch.setenv("GF_STAGNATION_FREEZE", "0")  # windowed, logs, never triggers
    monkeypatch.setenv("GF_STAGNATION_WINDOW", "4")
    windowed = _run()
    assert not windowed[3]["stagnation_frozen"]
    np.testing.assert_array_equal(plain[0], windowed[0])
    np.testing.assert_array_equal(plain[1], windowed[1])
    np.testing.assert_array_equal(plain[2], windowed[2])
    assert plain[3]["retained_size"] == windowed[3]["retained_size"]


def test_a_stagnation_freeze_is_the_php_resolvent_on_the_support_reached(monkeypatch):
    plain_blocks = len(_run()[0])
    monkeypatch.setenv("GF_STAGNATION_FREEZE", "2")  # a fraction of the weight is below 2: always quiet
    monkeypatch.setenv("GF_STAGNATION_WINDOW", "4")
    alphas, betas, r, info = _run()
    assert info["stagnation_frozen"] and info["csr_fallback"] and not info["cap_hit"]
    assert 0 < info["retained_size"] < 3432
    g = calc_G(alphas, betas, r, OMEGA, 0.0, DELTA)
    keys = info["proxy"].retained_keys()
    assert len(keys) == info["retained_size"]
    np.testing.assert_allclose(g, _dense_g(keys), atol=1e-8)
    assert len(alphas) > 4, "the restart runs the recurrence to convergence, not to the freeze"
    assert plain_blocks > 4, "the fixture must outlive two windows for the freeze to have happened mid-run"


def test_a_zero_threshold_only_reads_the_boundary_weight(monkeypatch, capsys):
    monkeypatch.setenv("GF_STAGNATION_FREEZE", "0")
    monkeypatch.setenv("GF_STAGNATION_WINDOW", "4")
    info = _run(verbose=True)[3]
    assert not info["stagnation_frozen"] and not info["csr_fallback"]
    out = capsys.readouterr().out
    assert "GF support:" in out and "weight on new rows" in out, "each window logs the support and the weight"


def test_a_saturated_support_has_no_boundary_weight_and_freezes_for_any_positive_threshold(monkeypatch):
    """The fixture's sector saturates, so nothing reaches a new row: the weight is exactly zero."""
    monkeypatch.setenv("GF_STAGNATION_FREEZE", "1e-30")
    monkeypatch.setenv("GF_STAGNATION_WINDOW", "4")
    assert _run()[3]["stagnation_frozen"]


def test_a_declined_csr_leaves_the_result_untouched_and_is_asked_once(monkeypatch):
    plain = _run()
    asked = []

    def declining(*args, **kwargs):
        asked.append(1)
        return False

    monkeypatch.setattr(gf_solvers, "_frozen_csr_fits", declining)
    monkeypatch.setenv("GF_STAGNATION_FREEZE", "2")
    monkeypatch.setenv("GF_STAGNATION_WINDOW", "4")
    declined = _run()
    assert len(asked) == 1, "a declined switch is not retried window after window"
    assert not declined[3]["stagnation_frozen"] and not declined[3]["csr_fallback"]
    np.testing.assert_array_equal(plain[0], declined[0])
    np.testing.assert_array_equal(plain[1], declined[1])


def test_the_knob_needs_the_csr_fallback(monkeypatch):
    monkeypatch.setenv("GF_FROZEN_CSR", "0")
    monkeypatch.setenv("GF_STAGNATION_FREEZE", "2")
    monkeypatch.setenv("GF_STAGNATION_WINDOW", "4")
    assert not _run()[3]["stagnation_frozen"]


@pytest.mark.mpi
def test_every_rank_takes_the_same_decision_and_matches_the_dense_oracle(monkeypatch):
    comm = MPI.COMM_WORLD
    monkeypatch.setenv("GF_STAGNATION_FREEZE", "2")
    monkeypatch.setenv("GF_STAGNATION_WINDOW", "4")
    alphas, betas, r, info = _run(comm=comm)
    assert info["stagnation_frozen"] and info["csr_fallback"]
    sizes = comm.allgather(info["retained_size"])
    assert len(set(sizes)) == 1, "the retained count is replicated"
    g = calc_G(alphas, betas, r, OMEGA, 0.0, DELTA)
    keys = [bytes(k.to_bytearray()[:2]) for k in info["proxy"].retained_keys()]
    all_keys = [k for rank_keys in comm.allgather(keys) for k in rank_keys]
    assert len(all_keys) == sizes[0]
    from impurityModel.ed.ManyBodyUtils import SlaterDeterminant

    # The oracle is independent of how the run was distributed: every rank builds it serially.
    np.testing.assert_allclose(g, _dense_g([SlaterDeterminant.from_bytes(k) for k in all_keys]), atol=1e-8)


def _weakly_coupled():
    """Ten strongly coupled orbitals and four more behind a 1e-5 bond: the support keeps growing, by rows
    that carry a vanishing share of the amplitude."""
    return _operator(N_ORB, strong=10, weak=1e-5)


def test_the_weight_decides_not_the_determinant_count(monkeypatch):
    """The count grows 125%, 71%, 39% per window while the weight on new rows falls from 3e-13 to 3e-23.

    A count-based trigger (growth below 0.2% per window) would wait for the sector to saturate, at 3432
    determinants; the weight-based one freezes at a fraction of that, and G does not move by more than the
    weight it dropped.
    """
    op = _weakly_coupled()
    plain = _run(op=op)
    monkeypatch.setenv("GF_STAGNATION_WINDOW", "4")
    monkeypatch.setenv("GF_STAGNATION_FREEZE", "1e-6")
    frozen = _run(op=op)
    info = frozen[3]
    assert info["stagnation_frozen"] and info["csr_fallback"]
    assert plain[3]["retained_size"] == 3432, "without the knob the support runs to the whole sector"
    assert info["retained_size"] < 0.7 * 3432, "the weight-based trigger freezes while the count is still growing"
    assert info["stagnation_leakage"] < 1e-6
    g_plain = calc_G(plain[0], plain[1], plain[2], OMEGA, 0.0, DELTA)
    g_frozen = calc_G(frozen[0], frozen[1], frozen[2], OMEGA, 0.0, DELTA)
    np.testing.assert_allclose(g_frozen, g_plain, atol=1e-6)
    np.testing.assert_allclose(g_frozen, _dense_g(info["proxy"].retained_keys(), op=op), atol=1e-8)


def _logged_weights(capsys):
    import re

    return [float(w) for w in re.findall(r"weight on new rows ([0-9.e+-]+)", capsys.readouterr().out)]


def test_the_weight_has_a_floor_set_by_the_slater_cutoff(monkeypatch, capsys):
    """Every row the apply returns has |amp| >= slaterWeightMin, so it carries >= slaterWeightMin**2.

    Without the cutoff the weight on new rows falls without bound (here to 1e-65, then exactly 0); with the
    production cutoff it plateaus at the cutoff's own scale and is otherwise exactly 0 -- which is why no
    threshold can be read off a cutoff-free run.
    """
    op = _weakly_coupled()
    monkeypatch.setenv("GF_STAGNATION_WINDOW", "4")
    monkeypatch.setenv("GF_STAGNATION_FREEZE", "0")
    _run(op=op, verbose=True)
    free = _logged_weights(capsys)
    _run(op=op, verbose=True, slater_min=1.4901161193847656e-08)
    cut = _logged_weights(capsys)
    assert min(w for w in free if w > 0.0) < 1e-40, "no cutoff: the weight keeps falling"
    nonzero = [w for w in cut if w > 0.0]
    assert nonzero and min(nonzero) > 1e-18, "with the cutoff it never goes below its own scale"
    assert min(nonzero) >= 1.4901161193847656e-08**2 / 100.0, "...which is slaterWeightMin**2 over the matvec's norm"


def test_a_threshold_below_the_weight_waits_for_the_support_to_saturate(monkeypatch):
    op = _weakly_coupled()
    monkeypatch.setenv("GF_STAGNATION_WINDOW", "4")
    monkeypatch.setenv("GF_STAGNATION_FREEZE", "1e-70")  # the last non-zero window carries 5e-65
    info = _run(op=op)[3]
    assert info["stagnation_frozen"] and info["retained_size"] == 3432 and info["stagnation_leakage"] == 0.0


def test_two_quiet_windows_are_needed_not_one(monkeypatch):
    op = _weakly_coupled()
    monkeypatch.setenv("GF_STAGNATION_WINDOW", "4")
    monkeypatch.setenv("GF_STAGNATION_FREEZE", "1e-6")
    two = _run(op=op)[3]["retained_size"]
    monkeypatch.setattr(gf_solvers, "_STAGNATION_QUIET_WINDOWS", 1)
    one = _run(op=op)[3]["retained_size"]
    assert one < two, "one quiet window freezes a window earlier, on a smaller support"


def test_a_unit_close_to_its_tolerance_is_left_alone(monkeypatch):
    """The restart recomputes every block from the seeds, so near the end it loses; the unit is not frozen."""
    op = _weakly_coupled()
    plain = _run(op=op)
    monkeypatch.setenv("GF_STAGNATION_WINDOW", "4")
    monkeypatch.setenv("GF_STAGNATION_FREEZE", "1e-6")
    monkeypatch.setattr(gf_solvers, "_STAGNATION_NEAR_END", 1e30)
    left = _run(op=op)
    assert not left[3]["stagnation_frozen"] and not left[3]["csr_fallback"]
    np.testing.assert_array_equal(plain[0], left[0])
    np.testing.assert_array_equal(plain[1], left[1])


@pytest.mark.parametrize(
    "kwargs,phrase",
    [
        (dict(reort="full"), "reort='none' only"),
        (dict(cap=np.inf), "finite determinant cap"),
    ],
)
def test_a_unit_it_cannot_apply_to_says_why_and_runs_unchanged(monkeypatch, capsys, kwargs, phrase):
    plain = _run(**kwargs)
    monkeypatch.setenv("GF_STAGNATION_WINDOW", "4")
    monkeypatch.setenv("GF_STAGNATION_FREEZE", "2")
    ignored = _run(verbose=True, **kwargs)
    assert not ignored[3]["stagnation_frozen"]
    assert phrase in capsys.readouterr().out
    np.testing.assert_array_equal(plain[0], ignored[0])


def test_the_diagnostic_carries_its_number_and_names_its_knob():
    d = check_stagnation_freeze(1_234_567, 3.2e-9, 1e-8)
    assert d.name == "stagnation_freeze" and d.severity == Severity.WARN
    assert "3.2e-09" in d.message and "1,234,567" in d.message and "GF_STAGNATION_FREEZE" in d.message
    assert d.value == 3.2e-9 and d.threshold == 1e-8
    assert not getattr(d, "needs_more_iterations", False) and not getattr(d, "needs_more_states", False)


def test_the_cap_diagnostic_is_unchanged_by_the_new_check():
    assert check_basis_truncation(False, 10, 1e9).severity == Severity.OK
    assert check_basis_truncation(True, 10, 10).severity == Severity.WARN
