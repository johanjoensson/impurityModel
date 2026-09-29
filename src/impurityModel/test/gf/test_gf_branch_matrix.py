"""Branch matrix of the Green's-function execution paths against an exact Lehmann oracle.

Every switch that selects a different code path through the GF engine is a factor here; a
pairwise covering array of the factors (every pair of factor values appears in at least one
cell) runs in the test gate, and the full cartesian product runs behind
``-m branch_matrix_full``. Each cell is compared against
:class:`~impurityModel.test.support.gf_branch_oracle.LehmannOracle`, a sector-wise dense
diagonalization that shares no code with the Krylov / continued-fraction pipeline.

The cells here are all *exact*: no occupation window (``dN=None``, ``chain_restrict=False``)
and no determinant cap, so any disagreement beyond the solver tolerance is a bug. Truncated
configurations (restrictions, caps) are judged by their own tests, which need a tolerance.

Factors
-------
``method``   lanczos / bicgstab / sliced / cipsi (``gf_method``)
``sparse``   ManyBodyState kernel vs CSR/dense array kernel (lanczos only)
``group``    ``GF_EIGENSTATE_GROUP`` 1 or 2 (stack thermal states into one recurrence)
``split``    ``GF_OPERATOR_SPLIT`` (pairwise scalar fractions; lanczos only)
``reort``    none / partial / full (lanczos only)
``mesh``     Matsubara only / real axis only / both
``model``    ``nb1`` (small sectors: the dense <500 array branch) / ``nb2`` (792-det
             excited sectors: the array *operator* branch)
``comm``     ``self`` (serial engine path) / ``world`` (the split + redistribute path)
"""

import contextlib
import io
import itertools
import os

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.greens_function import calc_Greens_function_with_offdiag, get_Greens_function
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.spectra import calc_spectra
from impurityModel.test.support.gf_branch_oracle import cached_oracle, distribute

TAU = 0.3
N_STATES = 3
DELTA = 0.2
IW = 1j * np.pi * TAU * (2 * np.arange(10) + 1)
W = np.linspace(-6.0, 6.0, 17)

FACTORS = {
    "method": ["lanczos", "bicgstab", "sliced", "cipsi"],
    "sparse": [True, False],
    "group": [1, 2],
    "split": [False, True],
    "reort": [None, "partial", "full"],
    "mesh": ["iw", "w", "both"],
    "model": [1, 2],
    "comm": ["self", "world"],
}


def _valid(cell):
    """Exclude combinations the code rejects or silently ignores (they would be duplicates)."""
    # sparse / split / reort are documented as ignored off the Lanczos path.
    lanczos_only = cell.get("sparse") is False or cell.get("split") or cell.get("reort") is not None
    if cell.get("method", "lanczos") != "lanczos" and lanczos_only:
        return False
    # Split and grouping are mutually exclusive; the split takes precedence (a duplicate cell).
    return not (cell.get("split") and cell.get("group") == 2)


def _pairwise_cells(factors, valid):
    """Greedy all-pairs covering array over ``factors`` restricted to ``valid`` cells."""
    names = list(factors)
    uncovered = {
        ((a, va), (b, vb))
        for a, b in itertools.combinations(names, 2)
        for va in factors[a]
        for vb in factors[b]
        if valid({a: va, b: vb})
    }
    cells = []
    all_cells = [dict(zip(names, vals)) for vals in itertools.product(*(factors[n] for n in names))]
    all_cells = [c for c in all_cells if valid(c)]
    while uncovered:
        best, best_gain = None, -1
        for c in all_cells:
            gain = sum(((a, c[a]), (b, c[b])) in uncovered for a, b in itertools.combinations(names, 2))
            if gain > best_gain:
                best, best_gain = c, gain
        if best_gain <= 0:
            break
        cells.append(best)
        for a, b in itertools.combinations(names, 2):
            uncovered.discard(((a, best[a]), (b, best[b])))
    return cells


def _cell_id(cell):
    return "-".join(f"{k}={v}" for k, v in cell.items())


PAIRWISE = _pairwise_cells(FACTORS, _valid)
FULL = [
    c
    for c in (dict(zip(FACTORS, vals)) for vals in itertools.product(*FACTORS.values()))
    if _valid(c) and c not in PAIRWISE
]


@contextlib.contextmanager
def _env(**values):
    old = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _setup(model, comm):
    oracle, (hOp, imp, baths, _n_orb, _n0, blocks) = cached_oracle(model)
    idx, es = oracle.thermal_states(N_STATES)
    basis = Basis(imp, baths, initial_basis=oracle.gs.dets, comm=comm, verbose=False)
    psis = distribute(basis, oracle.psis(idx))
    return oracle, hOp, basis, psis, idx, es, blocks


def _comm(kind):
    return MPI.COMM_SELF if kind == "self" else MPI.COMM_WORLD


def _known_failure(cell):
    """Reason string for a cell pinned to a ledger item (``doc/reviews/gf_review.md``), else None.

    Each is a strict xfail so the fix that resolves it flips the cell to XPASS -> failure,
    forcing the pin to be removed in the same commit.
    """
    lanczos_array = cell["method"] == "lanczos" and not cell["sparse"]
    if lanczos_array and cell["model"] == 2 and cell["comm"] == "world" and MPI.COMM_WORLD.size > 1:
        return "M1: array operator branch on a >1-rank color (test_block_green_array_multirank)"
    if lanczos_array and cell["model"] == 1 and cell["group"] == 2 and cell["mesh"] != "iw":
        # The Matsubara-only cells truncate too, but below the cell tolerance.
        return "C11: dense array kernel max_iter=ceil(N/p) ignores deflation -> truncated fraction"
    if cell["method"] == "sliced" and cell["model"] == 1 and cell["mesh"] != "iw":
        return "sliced: slice-seed/partition error above tolerance (path retired in Phase 1)"
    return None


def _run_cell(cell):
    comm = _comm(cell["comm"])
    oracle, hOp, basis, psis, idx, es, blocks = _setup(cell["model"], comm)
    iw = IW if cell["mesh"] in ("iw", "both") else None
    w = W if cell["mesh"] in ("w", "both") else None
    with (
        _env(GF_EIGENSTATE_GROUP=cell["group"], GF_OPERATOR_SPLIT=int(cell["split"])),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        gs_iw, gs_w, report = get_Greens_function(
            matsubara_mesh=iw,
            omega_mesh=w,
            psis=psis,
            es=list(es),
            tau=TAU,
            basis=basis,
            hOp=hOp,
            delta=DELTA,
            blocks=blocks,
            verbose=False,
            verbose_extra=False,
            reort=cell["reort"],
            dN=None,
            occ_cutoff=1e-12,
            slaterWeightMin=0.0,
            sparse=cell["sparse"],
            num_wanted=N_STATES,
            gf_method=cell["method"],
        )

    def check():
        for mesh, gs, z in ((iw, gs_iw, iw), (w, gs_w, None if w is None else w + 1j * DELTA)):
            if mesh is None:
                assert gs is None
                continue
            for block, g in zip(blocks, gs):
                ref = oracle.greens_function(block, idx, TAU, z)
                np.testing.assert_allclose(g, ref, atol=2e-6 * np.max(np.abs(ref)), rtol=0, err_msg=_cell_id(cell))
        assert report is not None

    _root_verdict(comm, check)


def _root_verdict(comm, check):
    """Run ``check`` on root (where the gathered G lives) and raise on EVERY rank if it fails.

    A root-only assertion would let the other ranks pass, which turns a strict xfail into an
    XPASS on those ranks.
    """
    message = None
    if comm.rank == 0:
        try:
            check()
        except AssertionError as err:
            message = str(err)
    message = comm.bcast(message, root=0)
    if message is not None:
        raise AssertionError(message)


def _cell_param(cell, extra_marks=()):
    marks = list(extra_marks)
    reason = _known_failure(cell)
    if reason is not None:
        # M1 fires only when the packing gives some color >1 rank, which depends on the unit
        # weights and the rank count -- not a property of the cell -- so it cannot be strict.
        marks.append(pytest.mark.xfail(strict=not reason.startswith("M1"), reason=reason))
    if cell["comm"] == "world":
        marks.append(pytest.mark.mpi)
    return pytest.param(cell, id=_cell_id(cell), marks=marks)


@pytest.mark.parametrize("cell", [_cell_param(c) for c in PAIRWISE])
def test_gf_branch_matrix_pairwise(cell):
    _run_cell(cell)


@pytest.mark.branch_matrix_full
@pytest.mark.parametrize("cell", [_cell_param(c) for c in FULL])
def test_gf_branch_matrix_full(cell):
    _run_cell(cell)


def test_pairwise_covering_array_covers_every_valid_pair():
    names = list(FACTORS)
    for a, b in itertools.combinations(names, 2):
        for va in FACTORS[a]:
            for vb in FACTORS[b]:
                if _valid({a: va, b: vb}):
                    assert any(c[a] == va and c[b] == vb for c in PAIRWISE), (a, va, b, vb)


# --- Consumers other than the self-energy driver ---------------------------------------------


@pytest.mark.parametrize("comm_kind", ["self", pytest.param("world", marks=pytest.mark.mpi)])
@pytest.mark.parametrize("side", ["add", "remove"])
def test_calc_spectra_matches_oracle(side, comm_kind):
    """PES/IPS path: ``calc_spectra`` on single-orbital transition operators, both sides."""
    comm = _comm(comm_kind)
    oracle, hOp, basis, psis, idx, es, _blocks = _setup(1, comm)
    char, shift = ("c", +1) if side == "add" else ("a", -1)
    tOps = [ManyBodyOperator({((o, char),): 1.0}) for o in (0, 2)]
    sgn = 1 if side == "add" else -1
    with contextlib.redirect_stdout(io.StringIO()):
        gs = calc_spectra(
            hOp, tOps, psis, np.asarray(es), TAU, sgn * W, basis, sgn * DELTA, 0.0, False, 1e-12, None, None, None
        )

    def check():
        for i, t in enumerate(tOps):
            ref = oracle.transition_tensor([t], shift, idx, TAU, sgn * W + 1j * sgn * DELTA)[:, 0, 0]
            np.testing.assert_allclose(gs[:, i], ref, atol=2e-6 * np.max(np.abs(ref)), rtol=0)

    _root_verdict(comm, check)


@pytest.mark.xfail(strict=True, reason="C11: default sparse=False runs the dense array kernel, truncated by deflation")
@pytest.mark.parametrize("comm_kind", ["self", pytest.param("world", marks=pytest.mark.mpi)])
def test_calc_greens_function_with_offdiag_matches_oracle(comm_kind):
    """The tensor path's engine entry: one width-m block over mixed transition operators."""
    from impurityModel.ed.gf_primitives import calc_thermally_averaged_G

    comm = _comm(comm_kind)
    oracle, hOp, basis, psis, idx, es, _blocks = _setup(1, comm)
    tOps = [
        ManyBodyOperator({((0, "c"),): 1.0, ((1, "c"),): 0.5j}),
        ManyBodyOperator({((2, "c"),): 1.0}),
        ManyBodyOperator({((3, "c"),): 0.7, ((0, "c"),): -0.2}),
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        alphas, betas, r = calc_Greens_function_with_offdiag(hOp, tOps, psis, list(es), basis, DELTA, verbose=False)

    def check():
        e0 = np.min(es)
        Z = np.sum(np.exp(-(np.asarray(es) - e0) / TAU))
        chi = calc_thermally_averaged_G(alphas, betas, r, W, es, e0, TAU, DELTA) / Z
        ref = oracle.transition_tensor(tOps, +1, idx, TAU, W + 1j * DELTA)
        np.testing.assert_allclose(chi, ref, atol=2e-6 * np.max(np.abs(ref)), rtol=0)

    _root_verdict(comm, check)
