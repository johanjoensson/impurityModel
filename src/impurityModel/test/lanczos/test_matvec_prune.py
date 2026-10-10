"""``MATVEC_PRUNE``: a sparse matvec whose output is redistributed cuts ``slaterWeightMin`` on each
row's summed amplitude (``apply_and_redistribute``, ``block_apply``), not on the per-rank partials.

The oracle is a serial ``apply_block(v, cutoff)`` over the whole block, which sees every row's
whole sum. The multi-rank tests split the input rows over the ranks by position, not by owner, so
determinants reached from rows on different ranks arrive as partial sums -- the case the knob is
about. ``test_gf_apply_row_chunking.py`` covers the block-Lanczos step itself.
"""

import itertools

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.BiCGSTAB import _make_matmat, block_bicgstab
from impurityModel.ed.BlockLanczosArray import Reort
from impurityModel.ed.BlockLanczosCore import apply_and_redistribute, block_apply, matvec_cut_after_sum
from impurityModel.ed.chebyshev_filter import chebyshev_apply, partition_of_unity
from impurityModel.ed.gf_admission import start_set
from impurityModel.ed.gf_primitives import _CappedBasisProxy
from impurityModel.ed.gf_solvers import block_Green
from impurityModel.ed.gf_units import enumerate_gf_units
from impurityModel.ed.greens_function import calc_G
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant

_IMP = {0: [[0, 1]]}
_BATHS = ({0: [[2, 3]]}, {0: [[4, 5]]})

# Cutoffs at which this model's H v has rows whose per-rank partials fall below the cutoff while
# their sums do not, at 2, 3 and 4 ranks and in both forms (before_sum then loses them; checked below).
_CUTOFFS = (0.4, 0.7)


@pytest.fixture(autouse=True)
def _knob_unset(monkeypatch):
    monkeypatch.delenv("MATVEC_PRUNE", raising=False)


def _det(occupied):
    """Determinant with the given orbitals occupied (MSB-first: orbital i = bit 7-i)."""
    b = 0
    for i in occupied:
        b |= 1 << (7 - i)
    return SlaterDeterminant.from_bytes(bytes([b]))


def _siam_6():
    """Single-impurity Anderson model, 6 spin-orbitals (0,1 imp; 2,3 val; 4,5 cond)."""
    terms = {}
    for o, e in ((0, -1.0), (1, -1.0), (2, -3.0), (3, -3.0), (4, 3.0), (5, 3.0)):
        terms[((o, "c"), (o, "a"))] = e
    terms[((0, "c"), (1, "c"), (1, "a"), (0, "a"))] = 4.0
    for a, b in ((0, 2), (1, 3), (0, 4), (1, 5)):
        terms[((a, "c"), (b, "a"))] = 0.5
        terms[((b, "c"), (a, "a"))] = 0.5
    return ManyBodyOperator(terms)


def _states():
    """Two columns over the N=3 sector, unequal amplitudes so the cut lands between rows."""
    rng = np.random.default_rng(7)
    dets = [_det(o) for o in ((0, 2, 3), (1, 2, 3), (0, 1, 2), (0, 1, 3), (0, 2, 4), (1, 3, 5), (2, 3, 4), (2, 3, 5))]
    return [ManyBodyState({d: complex(a) for d, a in zip(dets, rng.uniform(0.1, 1.0, len(dets)))}) for _ in range(2)]


def _rows(block):
    """{determinant: (amplitude per column)} of a shared-support block, every row it holds -- an
    all-zero row included, which a to_states() round trip would hide."""
    return {k: tuple(complex(a) for a in row) for k, row in block.items()}


def _gathered(rows, comm):
    out = {}
    for part in comm.allgather(rows):
        assert not set(out) & set(part), "a row on two ranks"
        out.update(part)
    return out


def _my_share(states, comm):
    """This rank's slice of every column's rows, split by position (not by owner)."""
    keys = sorted({k for s in states for k in s.keys()})
    mine = keys[comm.rank :: comm.size]
    return [ManyBodyState({k: a for k, a in s.items() if k in mine}) for s in states]


def _assert_rows_match(got, want):
    assert set(got) == set(want)
    for k in want:
        np.testing.assert_allclose(got[k], want[k], rtol=1e-12, atol=1e-14)


def test_the_cut_is_after_the_sum_whenever_something_is_summed(monkeypatch):
    assert matvec_cut_after_sum(True, 0.1)
    assert matvec_cut_after_sum(True, 0.0)  # the exact-zero cut: rows whose partials cancel
    assert not matvec_cut_after_sum(False, 0.1)  # no redistribute: the apply sees whole sums
    monkeypatch.setenv("MATVEC_PRUNE", "before_sum")
    assert not matvec_cut_after_sum(True, 0.1)


def test_matvec_prune_rejects_an_unknown_value(monkeypatch):
    monkeypatch.setenv("MATVEC_PRUNE", "sometimes")
    with pytest.raises(ValueError, match="MATVEC_PRUNE"):
        matvec_cut_after_sum(True, 0.1)


def test_no_redistribute_is_the_plain_apply():
    H, block = _siam_6(), ManyBodyState.from_states(_states())
    _assert_rows_match(_rows(apply_and_redistribute(H, block, None, 0.3, False)), _rows(H.apply_block(block, 0.3)))


def _distributed(comm):
    keys = sorted({k for s in _states() for k in s.keys()})
    return Basis(_IMP, _BATHS, initial_basis=keys, comm=comm, verbose=False)


def _block_reference(cutoff):
    """One serial apply over the whole block: a row survives if any column's whole sum does."""
    return _rows(_siam_6().apply_block(ManyBodyState.from_states(_states()), cutoff))


def _list_reference(cutoff):
    """The state-list form cuts each state on its own (apply_multi's per-state cut)."""
    return _rows(
        ManyBodyState.from_states(
            [_siam_6().apply_block(ManyBodyState.from_states([st]), cutoff).to_states()[0] for st in _states()]
        )
    )


def _block_result(cutoff, comm):
    out = apply_and_redistribute(
        _siam_6(), ManyBodyState.from_states(_my_share(_states(), comm)), _distributed(comm), cutoff, True
    )
    return _gathered(_rows(out), comm)


def _list_result(cutoff, comm):
    out = block_apply(_siam_6(), _my_share(_states(), comm), _distributed(comm), True, cutoff)
    return _gathered(_rows(ManyBodyState.from_states(list(out))), comm)


@pytest.mark.mpi
@pytest.mark.skipif(MPI.COMM_WORLD.size == 1, reason="needs input rows on more than one rank")
@pytest.mark.parametrize("cutoff", _CUTOFFS)
@pytest.mark.parametrize(
    "result, reference",
    [(_block_result, _block_reference), (_list_result, _list_reference)],
    ids=["block", "state_list"],
)
def test_a_redistributed_matvec_cuts_the_summed_row(result, reference, cutoff):
    _assert_rows_match(result(cutoff, MPI.COMM_WORLD), reference(cutoff))


@pytest.mark.mpi
@pytest.mark.skipif(MPI.COMM_WORLD.size == 1, reason="needs input rows on more than one rank")
@pytest.mark.parametrize("cutoff", _CUTOFFS)
@pytest.mark.parametrize(
    "result, reference",
    [(_block_result, _block_reference), (_list_result, _list_reference)],
    ids=["block", "state_list"],
)
def test_before_sum_cuts_each_ranks_partial(result, reference, cutoff, monkeypatch):
    """The fallback keeps the old order, and these cases are ones where the order matters -- so the
    after_sum test above discriminates."""
    monkeypatch.setenv("MATVEC_PRUNE", "before_sum")
    with pytest.raises(AssertionError):
        _assert_rows_match(result(cutoff, MPI.COMM_WORLD), reference(cutoff))


def _capped_matvec(cutoff, cap):
    """One serial matvec through a capping proxy (redistribute=True, as BiCGSTAB/GMRES and the
    Chebyshev filter call it on a capped unit). Serially there are no partial sums, so the cut order
    must change nothing -- including what the proxy admits."""
    keys = sorted({k for s in _states() for k in s.keys()})
    proxy = _CappedBasisProxy(Basis(_IMP, _BATHS, initial_basis=keys[:3], verbose=False), cap=cap)
    out = apply_and_redistribute(_siam_6(), ManyBodyState.from_states(_states()), proxy, cutoff, True)
    return _rows(out), proxy.frozen, proxy.cap_hit, proxy.retained_size


@pytest.mark.parametrize("cutoff", _CUTOFFS)
@pytest.mark.parametrize("cap", [10, 1000])
def test_a_capping_proxy_admits_on_the_cut_rows(cutoff, cap, monkeypatch):
    """The proxy admits inside redistribute_block; admitting before the cut counted rows the cut then
    dropped toward the cap (cap 10, cutoff 0.7: frozen at 10 retained from 7 surviving rows)."""
    monkeypatch.setenv("MATVEC_PRUNE", "before_sum")
    rows, *state = _capped_matvec(cutoff, cap)
    monkeypatch.setenv("MATVEC_PRUNE", "after_sum")
    rows_after, *state_after = _capped_matvec(cutoff, cap)
    _assert_rows_match(rows_after, rows)
    assert state_after == state


# --- The other call sites, end to end or at the site (each goes red with that site reverted to the
# before-sum cut at -n 2, 3 and 4; the cutoffs are ones that discriminate there). ---------------

_multirank = pytest.mark.skipif(MPI.COMM_WORLD.size == 1, reason="needs input rows on more than one rank")


@pytest.mark.mpi
@_multirank
@pytest.mark.parametrize("cutoff", _CUTOFFS)
def test_block_apply_cuts_the_summed_row(cutoff):
    comm = MPI.COMM_WORLD
    out = block_apply(
        _siam_6(), ManyBodyState.from_states(_my_share(_states(), comm)), _distributed(comm), True, cutoff
    )
    _assert_rows_match(_gathered(_rows(out), comm), _block_reference(cutoff))


@pytest.mark.mpi
@_multirank
@pytest.mark.parametrize("cutoff", _CUTOFFS)
def test_the_bicgstab_matvec_cuts_the_summed_row(cutoff):
    comm = MPI.COMM_WORLD
    matmat = _make_matmat(_siam_6(), _distributed(comm), cutoff, False, True)
    out = matmat(ManyBodyState.from_states(_my_share(_states(), comm)))
    _assert_rows_match(_gathered(_rows(out), comm), _block_reference(cutoff))


def _sector(comm):
    """The whole N=3 sector of the 6-orbital model (20 determinants)."""
    return Basis(
        _IMP, _BATHS, initial_basis=[_det(c) for c in itertools.combinations(range(6), 3)], comm=comm, verbose=False
    )


def _owned_seeds(basis, comm):
    """Every column of _states(), supplied by rank 0 and routed to the owners."""
    if comm is None:
        return _states()
    seeds = [ManyBodyState.from_states([s]) if comm.rank == 0 else ManyBodyState(width=1) for s in _states()]
    return [b.to_states()[0] for b in basis.redistribute_psis(*seeds)]


def _columns(states, comm):
    """{determinant: amplitude} per column, gathered over the ranks when ``comm`` is given."""
    cols = [{k: complex(a[0]) for k, a in st.items()} for st in states]
    return [_gathered(c, comm) for c in cols] if comm is not None else cols


def _assert_columns_match(got, want, atol):
    for g, w in zip(got, want):
        assert set(g) == set(w)
        np.testing.assert_allclose([g[k] for k in w], [w[k] for k in w], rtol=0, atol=atol)


def _bicgstab(comm, cutoff):
    basis = _sector(comm)
    rhs = ManyBodyState.from_states(_owned_seeds(basis, comm))
    x = block_bicgstab(_siam_6(), ManyBodyState(width=rhs.width), rhs, basis, cutoff, atol=1e-10, max_iter=8)
    return _columns(x.to_states(), comm)


@pytest.mark.mpi
@_multirank
def test_bicgstab_does_not_depend_on_the_rank_count():
    _assert_columns_match(_bicgstab(MPI.COMM_WORLD, 0.05), _bicgstab(None, 0.05), 1e-9)


def _chebyshev(comm, cutoff):
    basis = _sector(comm)
    bounds = (-12.0, 12.0)
    coefficients, _, _ = partition_of_unity(bounds, np.array([-3.0, 0.0, 3.0]), degree=6)
    out = chebyshev_apply(_siam_6(), basis, _owned_seeds(basis, comm), coefficients, cutoff, bounds)
    return _columns([col for window in out for col in window], comm)


@pytest.mark.mpi
@_multirank
@pytest.mark.parametrize("cutoff", (0.2, 0.4))
def test_the_chebyshev_filter_does_not_depend_on_the_rank_count(cutoff):
    _assert_columns_match(_chebyshev(MPI.COMM_WORLD, cutoff), _chebyshev(None, cutoff), 1e-12)


def _det12(occupied):
    """Two-byte determinant of the 12-orbital model below."""
    b = [0, 0]
    for i in occupied:
        b[i // 8] |= 1 << (7 - i % 8)
    return SlaterDeterminant.from_bytes(bytes(b))


def _model12():
    """12 spin-orbitals, two impurity orbitals hybridizing with ten bath orbitals: large enough that
    block_Green's basis-expansion probe stops short of the sector at these cutoffs (the 6-orbital
    model's probe reaches it whole and cannot discriminate)."""
    terms = {((o, "c"), (o, "a")): -1.0 + 0.3 * o for o in range(12)}
    for a in (0, 1):
        for b in range(2, 12):
            terms[((a, "c"), (b, "a"))] = 0.4
            terms[((b, "c"), (a, "a"))] = 0.4
    terms[((0, "c"), (1, "c"), (1, "a"), (0, "a"))] = 3.0
    return ManyBodyOperator(terms)


def _block_green(comm, cutoff):
    full = [
        ManyBodyState({_det12([0, 2, 3, 4, 5, 6]): 1.0 + 0j, _det12([1, 2, 3, 4, 5, 6]): 0.5 + 0j}),
        ManyBodyState({_det12([0, 1, 2, 3, 4, 5]): 1.0 + 0j}),
    ]
    basis = Basis(
        {0: [[0, 1]]},
        ({0: [list(range(2, 7))]}, {0: [list(range(7, 12))]}),
        initial_basis=sorted({k for s in full for k in s.keys()}),
        comm=comm,
        verbose=False,
    )
    seeds = [ManyBodyState(dict(s.items()) if comm is None or comm.rank == 0 else {}, width=1) for s in full]
    if comm is not None:
        seeds = basis.redistribute_psis(*seeds)
    omega = np.linspace(-6.0, 6.0, 25)
    alphas, betas, r = block_Green(_model12(), seeds, basis, 0.2, Reort.NONE, slaterWeightMin=cutoff, verbose=False)
    return calc_G(alphas, betas, r, omega, 0.0, 0.2), basis.size


@pytest.mark.mpi
@_multirank
@pytest.mark.parametrize("cutoff", (0.2, 0.4))
def test_the_block_green_probe_does_not_depend_on_the_rank_count(cutoff):
    g, size = _block_green(MPI.COMM_WORLD, cutoff)
    g_serial, size_serial = _block_green(None, cutoff)
    assert size == size_serial
    np.testing.assert_allclose(g, g_serial, rtol=1e-10, atol=1e-12)


# --- GF/spectra seeds: a many-term transition operator (an XAS dipole, a NIXS or rotated-orbital
# operator) sends determinants on different ranks to the same seed row. ---------------------------


def _many_term_op():
    return ManyBodyOperator({((o, "a"),): a for o, a in ((0, 0.9), (1, 0.6), (2, 0.5), (3, 0.4), (4, 0.3), (5, 0.2))})


def _seed_columns(comm, cutoff):
    """Every unit's seed columns, summed over the ranks (so a before_sum run's partials add up too)."""
    psis, basis = (_states(), None) if comm is None else (_my_share(_states(), comm), _distributed(comm))
    _units, unit_seeds, _windows = enumerate_gf_units(
        [([_many_term_op()], 0.1)], psis, [None], None, cutoff, basis=basis
    )
    cols = [{k: complex(a[0]) for k, a in col.items()} for seeds in unit_seeds for col in seeds]
    if comm is None:
        return cols
    summed = []
    for col in cols:
        total = {}
        for part in comm.allgather(col):
            for k, v in part.items():
                total[k] = total.get(k, 0) + v
        summed.append(total)
    return summed


@pytest.mark.mpi
@_multirank
@pytest.mark.parametrize("cutoff", _CUTOFFS)
def test_transition_operator_seeds_are_cut_on_the_summed_row(cutoff):
    _assert_columns_match(_seed_columns(MPI.COMM_WORLD, cutoff), _seed_columns(None, cutoff), 1e-13)


@pytest.mark.mpi
@_multirank
@pytest.mark.parametrize("cutoff", _CUTOFFS)
def test_before_sum_cuts_each_ranks_seed_partials(cutoff, monkeypatch):
    monkeypatch.setenv("MATVEC_PRUNE", "before_sum")
    with pytest.raises(AssertionError):
        _assert_columns_match(_seed_columns(MPI.COMM_WORLD, cutoff), _seed_columns(None, cutoff), 1e-13)


# --- Exact cancellation: a determinant reached from rows on two ranks whose partials cancel. A serial
# apply never emits it; after the sum it is dropped too, at any cutoff including 0. ----------------


def _cancelling_seed():
    """``|A> - |B>`` with ``<D|H|A> == <D|H|B>`` for some ``D``: H takes the seed's rows to ``D`` with
    amplitudes that cancel exactly. Returns the two rows and the cancelled determinants."""
    h, dets = _siam_6(), [_det(c) for c in itertools.combinations(range(6), 3)]
    for a, b in itertools.combinations(dets, 2):
        ha = {k: complex(v[0]) for k, v in h.apply_block(ManyBodyState.from_states([ManyBodyState({a: 1.0})])).items()}
        hb = {k: complex(v[0]) for k, v in h.apply_block(ManyBodyState.from_states([ManyBodyState({b: 1.0})])).items()}
        cancelled = [k for k in ha if k in hb and k not in (a, b) and ha[k] == hb[k]]
        if cancelled:
            return (a, 1.0 + 0j), (b, -1.0 + 0j), cancelled
    raise AssertionError("the model has no exactly cancelling pair")


def _split_seed(comm, sign=-1.0):
    """``|A> + sign |B>`` (the cancelling pair) with its two rows on ranks 0 and 1 (serial: both rows).
    At ``sign = -1`` the shared images cancel; at ``+1`` they add up."""
    row_a, (b, amp_b), _ = _cancelling_seed()
    row_b = (b, -sign * amp_b)
    rows = [row for owner, row in ((0, row_a), (1, row_b)) if comm is None or owner == comm.rank]
    return ManyBodyState.from_states([ManyBodyState(dict(rows), width=1)])


@pytest.mark.mpi
@_multirank
def test_a_row_whose_partials_cancel_is_dropped_after_the_sum():
    comm = MPI.COMM_WORLD
    *_, cancelled = _cancelling_seed()
    reference = _rows(_siam_6().apply_block(_split_seed(None), 0.0))
    assert not set(cancelled) & set(reference)
    got = _gathered(_rows(apply_and_redistribute(_siam_6(), _split_seed(comm), _sector(comm), 0.0, True)), comm)
    _assert_rows_match(got, reference)


def _start_keys(comm, seeds_block, shell_tol):
    basis = _sector(comm) if comm is not None else None
    seeds = seeds_block.to_states()
    x0 = [ManyBodyState(width=1) for _ in seeds]
    keys = start_set(_siam_6(), seeds_block, seeds, x0, 1e-3, comm, len(seeds), shell_tol, basis=basis)
    if comm is None:
        return set(keys)
    out = set()
    for part in comm.allgather(set(keys)):
        out |= part
    return out


@pytest.mark.mpi
@_multirank
def test_the_admission_start_set_cuts_the_summed_shell():
    """GF_BICGSTAB_ADMISSION=outer with GF_ADMIT_FIRST_SHELL_TOL: the shared image of ``|A> + |B>``
    gets 0.5 / sqrt(2) = 0.35 of the seed norm from each rank and 0.71 in total, so a 0.5 cut keeps
    it only on the sum."""
    comm = MPI.COMM_WORLD
    *_, shared = _cancelling_seed()
    serial = _start_keys(None, _split_seed(None, sign=+1.0), 0.5)
    assert set(shared) <= serial
    assert _start_keys(comm, _split_seed(comm, sign=+1.0), 0.5) == serial


@pytest.mark.mpi
@_multirank
def test_the_admission_start_set_drops_a_cancelled_row():
    comm = MPI.COMM_WORLD
    *_, cancelled = _cancelling_seed()
    serial = _start_keys(None, _split_seed(None), 0.0)
    assert not set(cancelled) & serial
    assert _start_keys(comm, _split_seed(comm), 0.0) == serial
