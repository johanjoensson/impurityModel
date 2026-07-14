"""Single-precision Krylov storage (Phase 2 of ``doc/plans/blocklanczos_reort_memory.md``).

The retained Krylov basis is the dominant allocation of a reorthogonalized Green's-function
run — ``16 * p * n_blocks`` bytes per retained determinant, ~30x everything else at the
FCC-Ni operating point. Storing it in ``complex64`` halves that, and only the *stored*
basis narrows: the recurrence blocks, the overlaps and the residual stay complex128.

The catch this module pins down: a complex64 basis can only be projected against to
``u32 ~ 6e-8``, which is *above* the ``REORT_TOL = sqrt(EPS) ~ 1.5e-8`` semi-orthogonality
target that ``PARTIAL``/``SELECTIVE`` steer to. The target is unreachable, and their block
selection threshold ``BAD_BLOCK_TOL = EPS**0.75 ~ 1.8e-12`` sits five orders below the fp32
noise floor, so every block would be flagged: PARTIAL degenerates into FULL while delivering
worse orthogonality than FULL at complex128. Those modes reject the combination outright.
``FULL``/``PERIODIC`` hold no estimator and simply settle at ~6e-8.

Measured on the fixture below (4 bath levels, 200-determinant sector, 82 blocks):

===============  ==========  ============  ==============
mode             buffer      ||Q^H Q - I||  max\\|G - G_ref\\|
===============  ==========  ============  ==============
FULL c128        786 432 B   1.1e-15       7.7e-15
FULL c64         393 216 B   6.0e-08       1.8e-07
PARTIAL c128     786 432 B   1.9e-09       3.7e-14
===============  ==========  ============  ==============

So complex64 costs ~7 orders of Green's-function accuracy and buys exactly 2x — a good
trade only because 1.8e-07 is still far below the physical broadening (``delta = 0.1``) and
below the recurrence's own convergence tolerance. Which of complex64-FULL and
complex128-PARTIAL is the more orthogonal basis is workload dependent (on a 5-bath variant
PARTIAL drifts to 4.4e-07, above its own sqrt(EPS) target), so this module asserts absolute
bounds, not an ordering between them.
"""

import numpy as np
import pytest

from impurityModel.ed.basis_transcription import build_dense_matrix
from impurityModel.ed.BlockLanczos import block_lanczos_cy
from impurityModel.ed.BlockLanczosArray import REORT_TOL, Reort
from impurityModel.ed.greens_function import _make_gf_convergence_monitor, calc_G
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant, SparseKrylovDense

DELTA = 0.1
OMEGA = np.linspace(-10.0, 10.0, 41)
N_BATH = 4
ED, U, V = -1.5, 5.0, 0.7
# Bath levels on both sides of E_F: the FCC-Ni pathology (a bath state above E_F blows the
# reachable Krylov space up) in miniature.
EPS_BATH = np.array([-4.0, -1.2, 0.9, 3.1])
N_SO = 2 + 2 * N_BATH


def _det(occupied):
    b = bytearray((N_SO + 7) // 8)
    for i in occupied:
        b[i // 8] |= 1 << (7 - (i % 8))
    return SlaterDeterminant.from_bytes(bytes(b))


def _hamiltonian():
    """imp spin-orbitals 0 (dn), 1 (up); bath level k occupies 2+2k (dn), 3+2k (up)."""
    t = {}
    for o in (0, 1):
        t[((o, "c"), (o, "a"))] = ED
    t[((0, "c"), (1, "c"), (1, "a"), (0, "a"))] = U
    for k, e in enumerate(EPS_BATH):
        for s in (0, 1):
            b = 2 + 2 * k + s
            t[((b, "c"), (b, "a"))] = float(e)
            t[((s, "c"), (b, "a"))] = V
            t[((b, "c"), (s, "a"))] = V
    return ManyBodyOperator(t)


def _imp_bath():
    val = [2 + 2 * k + s for k in range(N_BATH) if EPS_BATH[k] < 0 for s in (0, 1)]
    con = [2 + 2 * k + s for k in range(N_BATH) if EPS_BATH[k] >= 0 for s in (0, 1)]
    return {0: [[0, 1]]}, ({0: [val]}, {0: [con]})


def _seeds(n_elec=5):
    return [
        ManyBodyState({_det(range(n_elec)): 1.0 + 0j}),
        ManyBodyState({_det(list(range(n_elec - 1)) + [n_elec]): 1.0 + 0j}),
    ]


@pytest.fixture(scope="module")
def siam():
    """(hOp, imp, baths, seeds, dense reference G) on the fully expanded sector."""
    hOp, (imp, baths) = _hamiltonian(), _imp_bath()
    psi = _seeds()
    support = sorted({d for s in psi for d in s})
    full = Basis(imp, baths, initial_basis=support, verbose=False)
    while True:
        before = full.size
        full.expand(hOp)
        if full.size == before:
            break
    H = np.asarray(build_dense_matrix(full, hOp))
    index = {d: i for i, d in enumerate(sorted(full.local_basis))}
    Vm = np.zeros((len(index), len(psi)), dtype=complex)
    for j, s in enumerate(psi):
        for d, a in s.items():
            Vm[index[d], j] = a
    g_ref = np.array([Vm.conj().T @ np.linalg.solve((w + 1j * DELTA) * np.eye(len(index)) - H, Vm) for w in OMEGA])
    return hOp, imp, baths, psi, support, full.size, g_ref


def _run(siam, reort, dtype):
    hOp, imp, baths, psi, support, sector, _ = siam
    basis = Basis(imp, baths, initial_basis=support, verbose=False)
    converged, _flag, _tol, _dg = _make_gf_convergence_monitor(DELTA, 0.0)
    alphas, betas, q, _w, _widths, status = block_lanczos_cy(
        [s.copy() for s in psi],
        hOp,
        basis,
        converged,
        verbose=False,
        reort=reort,
        max_iter=sector // len(psi),
        return_widths=True,
        return_status=True,
        store_krylov=reort != Reort.NONE,
        krylov_dtype=dtype,
    )
    return alphas, betas, q, status


def _orthonormality(store):
    """||Q^H Q - I||_max of the materialized (widened) store columns."""
    from impurityModel.ed.ManyBodyUtils import inner_multi

    cols = list(store)
    return np.max(np.abs(inner_multi(cols, cols) - np.eye(len(cols))))


def test_store_dtype_halves_the_buffer_and_round_trips():
    rng = np.random.default_rng(3)
    dets = [_det([i]) for i in range(N_SO)]
    cols = []
    for _ in range(8):
        st = ManyBodyState()
        for d in dets:
            st[d] = rng.standard_normal() + 1j * rng.standard_normal()
        cols.append(st)

    wide, narrow = SparseKrylovDense(), SparseKrylovDense(np.complex64)
    for store in (wide, narrow):
        store.reserve_rows(N_SO)
        store.append(cols)

    assert wide.dtype == np.dtype(np.complex128)
    assert narrow.dtype == np.dtype(np.complex64)
    assert narrow.stats()["buffer_bytes"] * 2 == wide.stats()["buffer_bytes"]

    for a, b in zip(list(wide), cols):
        assert a == b  # complex128 round-trips bit-exactly
    for a, b in zip(list(narrow), cols):
        rel = np.sqrt((a - b).norm2()) / np.sqrt(b.norm2())
        assert rel < 1e-6  # complex64 round-trips to fp32 roundoff


def test_bad_dtype_rejected():
    with pytest.raises(ValueError, match="complex64 or complex128"):
        SparseKrylovDense(np.float64)


@pytest.mark.parametrize("reort", [Reort.PARTIAL, Reort.SELECTIVE])
def test_complex64_rejected_by_estimator_modes(siam, reort):
    """A basis known to ~6e-8 cannot support a sqrt(EPS) semi-orthogonality target."""
    with pytest.raises(ValueError, match="incompatible with reort"):
        _run(siam, reort, np.complex64)


def test_full_complex64_halves_the_store_and_keeps_the_greens_function(siam):
    """FULL + complex64: exactly half the store, orthogonality at fp32 roundoff, GF intact.

    The complex128 FULL run is the accuracy reference. See the module docstring for the
    measured trade; the assertions here are absolute bounds, since the ordering against
    PARTIAL is workload dependent.
    """
    _, _, _, psi, _, _, g_ref = siam
    r = np.eye(len(psi), dtype=complex)

    results = {}
    for label, reort, dtype in (("c128", Reort.FULL, None), ("c64", Reort.FULL, np.complex64)):
        alphas, betas, store, status = _run(siam, reort, dtype)
        assert status == "converged", f"{label}: {status}"
        results[label] = {
            "bytes": store.stats()["buffer_bytes"],
            "orth": _orthonormality(store),
            "err": np.max(np.abs(calc_G(alphas, betas, r, OMEGA, 0.0, DELTA) - g_ref)),
        }

    c128, c64 = results["c128"], results["c64"]

    # The coefficient buffer is the whole point: it must halve.
    assert 1.8 < c128["bytes"] / c64["bytes"] <= 2.0
    # complex128 FULL is machine-orthogonal ...
    assert c128["orth"] < 1e-12
    # ... complex64 FULL settles at fp32 roundoff. The lower bound proves the narrow store
    # is actually in use and the projection is limited by *it*, not by anything else.
    assert 1e-9 < c64["orth"] < 1e-6
    # The Green's function still tracks the dense resolvent far below the broadening
    # (delta = 0.1) and below the recurrence's own convergence tolerance.
    assert c64["err"] < 1e-5


def test_block_green_sparse_defaults_to_complex128():
    """complex64 is opt-in through ``block_Green_sparse``, never a silent default.

    Defaulting it on would halve the store for free, but it would also cost ~7 orders of
    Green's-function accuracy — quietly breaking the guarantee that a capped recurrence
    reproduces the dense ``P H P`` resolvent exactly (``test_gf_truncation``).
    """
    from impurityModel.ed.greens_function import block_Green_sparse

    hOp, (imp, baths) = _hamiltonian(), _imp_bath()
    psi = _seeds()
    support = sorted({d for s in psi for d in s})

    for reort in (Reort.FULL, Reort.PARTIAL):
        basis = Basis(imp, baths, initial_basis=support, verbose=False)
        alphas, _betas, _r = block_Green_sparse(
            hOp, basis.redistribute_psis([s.copy() for s in psi]), basis, DELTA, reort=reort, verbose=False
        )
        assert len(alphas) > 1  # complex128 for both: no mode is auto-narrowed

    # An explicit illegal request is rejected, not silently widened back.
    basis = Basis(imp, baths, initial_basis=support, verbose=False)
    with pytest.raises(ValueError, match="incompatible with reort"):
        block_Green_sparse(
            hOp,
            basis.redistribute_psis([s.copy() for s in psi]),
            basis,
            DELTA,
            reort=Reort.PARTIAL,
            verbose=False,
            krylov_dtype=np.complex64,
        )

    # An explicit legal request reaches the store.
    basis = Basis(imp, baths, initial_basis=support, verbose=False)
    alphas, _betas, _r = block_Green_sparse(
        hOp,
        basis.redistribute_psis([s.copy() for s in psi]),
        basis,
        DELTA,
        reort=Reort.FULL,
        verbose=False,
        krylov_dtype=np.complex64,
    )
    assert len(alphas) > 1


def test_semi_orthogonality_target_is_documented_constant():
    """The guard's premise: fp32 roundoff sits above the semi-orthogonality target."""
    assert np.finfo(np.complex64).eps > REORT_TOL
