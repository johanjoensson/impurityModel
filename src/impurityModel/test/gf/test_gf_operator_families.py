"""The operator-family seam of :func:`get_Greens_function` (the self-energy estimator hook).

A family ``X`` resolved through ``operator_families`` must give the full ``len(X) x len(X)``
Green's function ``G_ab(z) = <X_a (z - H + E)^-1 X_b^dag> + <X_b^dag (z + H - E)^-1 X_a>``,
checked against the exact Lehmann oracle, whichever kernel runs it.
"""

import contextlib
import io

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.greens_function import get_Greens_function, impurity_operator_family
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.test.support.gf_branch_oracle import cached_oracle, distribute

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


#: Ledger C12 (doc/reviews/gf_review.md): the array path's basis-expansion probe starts from the
#: last Lanczos vector only, so a column whose chain closes inside the still-incomplete basis
#: deflates and its missing determinants are never added. The bath column of block [2, 3] is such
#: a column. Strict serially; on a world communicator M1 can pre-empt it, which depends on packing.
C12 = "C12: array-path expansion probe misses the determinants of a deflated column"
KERNELS = [("lanczos", True), ("lanczos", False), ("bicgstab", True)]


def _kernel_params(comm_kind):
    params = []
    for gf_method, sparse in KERNELS:
        marks = []
        if gf_method == "lanczos" and not sparse:
            marks.append(pytest.mark.xfail(strict=comm_kind == "self", reason=C12))
        params.append(pytest.param(gf_method, sparse, comm_kind, marks=marks, id=f"{comm_kind}-{gf_method}-{sparse}"))
    return params


@pytest.mark.parametrize(
    "gf_method, sparse, comm_kind",
    _kernel_params("self")
    + [pytest.param(*p.values, marks=[*p.marks, pytest.mark.mpi], id=p.id) for p in _kernel_params("world")],
)
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
