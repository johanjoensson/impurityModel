"""The operator-family seam of :func:`get_Greens_function` (the self-energy estimator hook).

A family ``X`` resolved through ``operator_families`` must give the full ``len(X) x len(X)``
Green's function ``G_ab(z) = <X_a (z - H + E)^-1 X_b^dag> + <X_b^dag (z + H - E)^-1 X_a>``,
checked against the exact Lehmann oracle, whichever kernel runs it.
"""

import contextlib
import functools
import io
import types

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.greens_function import get_Greens_function, get_greens_function_moments, impurity_operator_family
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.sigma import get_hcorr_v_hbath, get_Sigma_moments
from impurityModel.test.support.gf_branch_oracle import LehmannOracle, cached_oracle, distribute, two_orbital_model

TAU = 0.3
N_STATES = 3
DELTA = 0.2
IW = 1j * np.pi * TAU * (2 * np.arange(10) + 1)
W = np.linspace(-6.0, 6.0, 17)
#: A bath orbital of the nb1 model that hybridizes with impurity orbital 0 (block [0, 1]).
BATH = 4


def family_reference(oracle, removal_ops, idx, z):
    """Oracle G of the family ``X = removal_ops``: addition side minus the transposed removal side."""
    addition_ops = [op.adjoint() for op in removal_ops]
    g_add = oracle.transition_tensor(addition_ops, +1, idx, TAU, z)
    g_rem = oracle.transition_tensor(removal_ops, -1, idx, TAU, -z)
    return g_add - np.transpose(g_rem, (0, 2, 1))


def run_gf(comm, families, *, gf_method="lanczos", sparse=True, model=1):
    oracle, (hOp, imp, baths, _n_orb, _n0, blocks) = cached_oracle(model)
    idx, es = oracle.thermal_states(N_STATES)
    basis = Basis(imp, baths, initial_basis=oracle.gs.dets, comm=comm, verbose=False)
    psis = distribute(basis, oracle.psis(idx))
    with contextlib.redirect_stdout(io.StringIO()):
        gs_iw, gs_w, report = get_Greens_function(
            matsubara_mesh=IW,
            omega_mesh=W,
            psis=psis,
            es=list(es),
            tau=TAU,
            basis=basis,
            hOp=hOp,
            delta=DELTA,
            blocks=blocks,
            verbose=False,
            verbose_extra=False,
            reort=None,
            dN=None,
            occ_cutoff=1e-12,
            slaterWeightMin=0.0,
            sparse=sparse,
            num_wanted=N_STATES,
            gf_method=gf_method,
            operator_families=families,
        )
    return oracle, idx, blocks, gs_iw, gs_w, report


def root_verdict(comm, check):
    """Run ``check`` on root, where the gathered G lives, and raise on every rank if it fails."""
    message = None
    if comm.rank == 0:
        try:
            check()
        except AssertionError as err:
            message = str(err)
    message = comm.bcast(message, root=0)
    if message is not None:
        raise AssertionError(message)


def _with_bath(block):
    """The plain family of ``block`` plus one bath orbital: a non-trivial, still canonical, width."""
    add, rem = impurity_operator_family(block)
    return (
        add + [ManyBodyOperator({((BATH, "c"),): 1})],
        rem + [ManyBodyOperator({((BATH, "a"),): 1})],
    )


def test_the_family_reference_is_the_oracle_greens_function_for_the_plain_family():
    """The oracle composition used below reduces to the oracle's own G for plain ``c``."""
    oracle, (_h, _i, _b, _n, _n0, blocks) = cached_oracle(1)
    idx, _es = oracle.thermal_states(N_STATES)
    for block in blocks:
        ref = family_reference(oracle, impurity_operator_family(block)[1], idx, IW)
        np.testing.assert_allclose(ref, oracle.greens_function(block, idx, TAU, IW), rtol=0, atol=1e-12)


@pytest.mark.parametrize("comm_kind", ["self", pytest.param("world", marks=pytest.mark.mpi)])
def test_an_explicit_plain_family_is_bitwise_the_default(comm_kind):
    comm = MPI.COMM_SELF if comm_kind == "self" else MPI.COMM_WORLD
    *_, gs_iw, gs_w, _ = run_gf(comm, None)
    *_, fs_iw, fs_w, _ = run_gf(comm, impurity_operator_family)

    def check():
        for a, b in zip(gs_iw + gs_w, fs_iw + fs_w):
            np.testing.assert_array_equal(a, b)

    root_verdict(comm, check)


KERNELS = [("lanczos", True), ("lanczos", False), ("bicgstab", True)]


# The array cells also guard ledger C12: block [2, 3]'s bath column deflates inside the growing
# basis, which the expansion probe used to miss.
@pytest.mark.parametrize("gf_method, sparse", KERNELS)
@pytest.mark.parametrize("comm_kind", ["self", pytest.param("world", marks=pytest.mark.mpi)])
def test_a_wider_family_resolves_its_full_greens_function(gf_method, sparse, comm_kind):
    comm = MPI.COMM_SELF if comm_kind == "self" else MPI.COMM_WORLD
    oracle, idx, blocks, gs_iw, gs_w, report = run_gf(comm, _with_bath, gf_method=gf_method, sparse=sparse)

    def check():
        for block, g_iw, g_w in zip(blocks, gs_iw, gs_w):
            removal = _with_bath(block)[1]
            assert g_iw.shape == (len(IW), len(block) + 1, len(block) + 1)
            for g, z in ((g_iw, IW), (g_w, W + 1j * DELTA)):
                ref = family_reference(oracle, removal, idx, z)
                np.testing.assert_allclose(g, ref, rtol=0, atol=2e-6 * np.max(np.abs(ref)))
        # The sum rule is checked on the plain c part only; the bath column must not break it.
        # (The per-frequency path records no sum rule: it has no seed factors.)
        sum_rules = [d for d in report.diagnostics if d.name == "sum_rule"]
        if gf_method == "lanczos":
            assert len(sum_rules) == len(blocks)
        assert all(d.severity.name == "OK" for d in sum_rules), [d.value for d in sum_rules]

    root_verdict(comm, check)


def test_a_family_narrower_than_its_block_is_rejected():
    def narrow(block):
        add, rem = impurity_operator_family(block)
        return add[:1], rem[:1]

    with pytest.raises(ValueError, match="at least len"):
        run_gf(MPI.COMM_SELF, narrow)


# --------------------------------------------------------------------------------------------
# The improved-estimator family through the seam (test-only stub; the estimator itself is a
# later campaign). X = [c, q] with q_m = [c_m, H_int]: the removal side is 2n wide.
# --------------------------------------------------------------------------------------------


class ImprovedEstimatorStub:
    """The operator family of the symmetric improved estimator, and nothing else."""

    name = "improved-stub"

    def operator_families(self, block, solver_basis):
        c = [ManyBodyOperator({((orb, "a"),): 1}) for orb in block]
        q = [c_m.commutator(solver_basis.h_int) for c_m in c]
        removal = c + q
        return [op.adjoint() for op in removal], removal


def _split_model(u):
    """``two_orbital_model``'s ``H`` at interaction ``u``, with its one-body part ``h0`` and ``H_int``."""
    hOp, imp, baths, n_orb, n0, blocks = two_orbital_model(1, u=u)
    h0 = two_orbital_model(1, u=0.0)[0]
    return hOp, h0, hOp - h0, imp, baths, n_orb, n0, blocks


@functools.lru_cache(maxsize=None)
def _stub_oracle(u):
    hOp, h0, h_int, imp, baths, n_orb, n0, blocks = _split_model(u)
    return LehmannOracle(hOp, imp, baths, n_orb, n0), (hOp, h0, h_int, imp, baths, n_orb, n0, blocks)


def _run_stub(comm, u, iw, w=None, gf_method="lanczos", sparse=True):
    oracle, (hOp, _h0, h_int, imp, baths, _n, _n0, blocks) = _stub_oracle(u)
    stub = ImprovedEstimatorStub()
    solver_basis = types.SimpleNamespace(h_int=h_int)
    idx, es = oracle.thermal_states(N_STATES)
    basis = Basis(imp, baths, initial_basis=oracle.gs.dets, comm=comm, verbose=False)
    psis = distribute(basis, oracle.psis(idx))
    with contextlib.redirect_stdout(io.StringIO()):
        gs_iw, gs_w, _report = get_Greens_function(
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
            reort=None,
            dN=None,
            occ_cutoff=1e-12,
            slaterWeightMin=0.0,
            sparse=sparse,
            num_wanted=N_STATES,
            gf_method=gf_method,
            operator_families=lambda block: stub.operator_families(block, solver_basis),
        )
    families = [stub.operator_families(block, solver_basis)[1] for block in blocks]
    return oracle, idx, blocks, families, basis, psis, es, gs_iw, gs_w


@pytest.mark.parametrize("u", [2.5, 0.0])
@pytest.mark.parametrize(
    "gf_method, sparse",
    [("lanczos", True), ("lanczos", False), ("bicgstab", True)],
)
@pytest.mark.parametrize("comm_kind", ["self", pytest.param("world", marks=pytest.mark.mpi)])
def test_the_improved_estimator_family_resolves_its_full_greens_function(u, gf_method, sparse, comm_kind):
    """All four ``n x n`` blocks of the ``[c, q]`` Green's function match the oracle.

    Comparing only the ``c`` block would pass with ``q`` of the wrong sign or adjoint; the cross
    blocks are what pins them. At ``u = 0`` every ``q`` is zero, half of each seed deflates, and
    the ``q`` rows and columns must come back zero -- with an empty half on some rank at ``-n 3``.
    """
    comm = MPI.COMM_SELF if comm_kind == "self" else MPI.COMM_WORLD
    oracle, idx, blocks, families, *_, gs_iw, gs_w = _run_stub(comm, u, IW, W, gf_method=gf_method, sparse=sparse)

    def check():
        for block, removal, g_iw, g_w in zip(blocks, families, gs_iw, gs_w):
            n = len(block)
            assert g_iw.shape == (len(IW), 2 * n, 2 * n)
            for g, z in ((g_iw, IW), (g_w, W + 1j * DELTA)):
                ref = family_reference(oracle, removal, idx, z)
                np.testing.assert_allclose(g, ref, rtol=0, atol=2e-6 * np.max(np.abs(ref)))
                np.testing.assert_allclose(g[:, :n, :n], oracle.greens_function(block, idx, TAU, z), rtol=0, atol=1e-5)
                if u == 0.0:
                    assert np.max(np.abs(g[:, n:, :])) == 0 and np.max(np.abs(g[:, :, n:])) == 0

    root_verdict(comm, check)


@pytest.mark.parametrize("comm_kind", ["self", pytest.param("world", marks=pytest.mark.mpi)])
def test_the_improved_family_seed_gram_cross_block_is_the_static_self_energy(comm_kind):
    """``<{q_a, c_b^dag}>`` -- the ``1/z`` tail of the ``(q, c)`` block -- is ``sigma_static``.

    The reference is production's own static self-energy, ``M1 - hcorr`` from the exact spectral
    moments (:func:`get_greens_function_moments`, :func:`get_Sigma_moments`), in the solver basis
    and unmasked. Holding by construction once ``H_int = H - h0``, this pins the wiring: the
    ``H_int`` the family is built from is the interaction the Dyson estimator subtracts ``h0`` for.
    """
    comm = MPI.COMM_SELF if comm_kind == "self" else MPI.COMM_WORLD
    z = np.array([1e6j])
    _oracle, _idx, blocks, _f, basis, psis, es, gs_iw, _ = _run_stub(comm, 2.5, z)
    _o, (hOp, h0, _hi, imp, baths, _n, _n0, _b) = _stub_oracle(2.5)
    n_imp = sum(len(b) for b in imp[0])
    n_bath = sum(len(b) for side in baths for b in side[0])
    M = get_greens_function_moments(psis, list(es), TAU, basis, hOp, list(range(n_imp)))
    hcorr, v, _vd, h_bath = get_hcorr_v_hbath(h0, {0: n_imp}, {0: n_bath})
    sigma_inf = get_Sigma_moments(M, hcorr, v, h_bath)[0]

    def check():
        assert np.max(np.abs(sigma_inf)) > 0.1  # the interaction is on, so this is not 0 == 0
        for block, g in zip(blocks, gs_iw):
            n = len(block)
            gram = z[0] * g[0]  # z G(z) = M0 + O(1/z)
            np.testing.assert_allclose(gram[:n, :n], np.eye(n), atol=1e-4)
            np.testing.assert_allclose(gram[n:, :n], sigma_inf[np.ix_(block, block)], atol=1e-4)

    root_verdict(comm, check)
