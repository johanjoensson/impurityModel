"""Polarization spectral tensor (B2b): ``I_eps = eps^dagger chi eps`` equals the
per-polarization scalar spectrum.

A one-body transition operator is linear in the polarization,
``T_eps = sum_alpha eps_alpha T_alpha``, so the spectrum for any polarization is a
contraction of the single Hermitian tensor ``chi_{alpha beta}(w) = <g|T_alpha^dagger (w-H)^-1
T_beta|g>``. :func:`spectra.calc_spectra_tensor` computes and returns ``chi`` itself (one
block-Lanczos); :func:`impurityModel.ed.polarization.contract_spectra_tensor` then contracts
it -- this must reproduce :func:`spectra.calc_spectra` run on the combined operator
``T_eps`` for every polarization -- including complex / circular ones.
"""

from itertools import combinations

import numpy as np
from mpi4py import MPI

from impurityModel.ed import polarization, spectra
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant, applyOp, inner
from impurityModel.ed.symmetries import ComponentReduction

N_ORB = 4  # 0,1 core doublet; 2,3 impurity doublet.


def _model():
    ec, ei, t, u = -2.0, 0.3, 0.4, 1.5
    terms = {
        ((0, "c"), (0, "a")): ec,
        ((1, "c"), (1, "a")): ec,
        ((2, "c"), (2, "a")): ei,
        ((3, "c"), (3, "a")): ei,
        ((2, "c"), (3, "a")): t,
        ((3, "c"), (2, "a")): t,
        ((2, "c"), (3, "c"), (3, "a"), (2, "a")): u,
    }
    return ManyBodyOperator(terms)


def _bytes(occ):
    b = bytearray(1)
    for o in occ:
        b[0] |= 1 << (7 - o)
    return bytes(b)


def _ground_state(op, n_elec):
    dets = [_bytes(occ) for occ in combinations(range(N_ORB), n_elec)]
    states = [ManyBodyState({SlaterDeterminant.from_bytes(d): 1.0}) for d in dets]
    n = len(states)
    h = np.zeros((n, n), dtype=complex)
    for j, sj in enumerate(states):
        col = applyOp(op, sj)
        for i, si in enumerate(states):
            h[i, j] = inner(si, col)
    ev, evec = np.linalg.eigh(h)
    gs = ManyBodyState({SlaterDeterminant.from_bytes(dets[i]): evec[i, 0] for i in range(n) if abs(evec[i, 0]) > 1e-14})
    return gs, ev[0], dets


def _basis(dets):
    return Basis(
        impurity_orbitals={0: [list(range(N_ORB))]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=list(dets),
        verbose=False,
        comm=MPI.COMM_SELF,
    )


# Three "Cartesian" component operators: core -> impurity one-body operators.
def _components():
    return [
        ManyBodyOperator({((2, "c"), (0, "a")): 1.0, ((3, "c"), (1, "a")): 1.0}),
        ManyBodyOperator({((2, "c"), (1, "a")): 1.0, ((3, "c"), (0, "a")): 1.0}),
        ManyBodyOperator({((2, "c"), (0, "a")): 1.0, ((3, "c"), (1, "a")): -1.0}),
    ]


_DN = dict(dN_imp={0: (1, 1)}, dN_val={0: (1, 1)}, dN_con={0: (1, 1)})


def test_tensor_contraction_matches_per_polarization():
    op = _model()
    gs, e0, dets = _ground_state(op, 2)
    comps = _components()
    w = np.linspace(-5.0, 5.0, 61)

    polarizations = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 1j, 0.0],  # circular
        [0.3, -0.7, 0.5j],
    ]

    chi = spectra.calc_spectra_tensor(
        op,
        comps,
        [gs],
        [e0],
        tau=0.01,
        w=w,
        basis=_basis(dets),
        delta=0.2,
        slaterWeightMin=0.0,
        verbose=False,
        occ_cutoff=1e-12,
        reduction=None,
        **_DN,
    )
    tensor = polarization.contract_spectra_tensor(chi, polarizations)

    for p, eps in enumerate(polarizations):
        combined = spectra._combine_component_ops(comps, eps)
        ref = spectra.calc_spectra(
            op,
            [combined],
            [gs],
            [e0],
            tau=0.01,
            w=w,
            basis=_basis(dets),
            delta=0.2,
            slaterWeightMin=0.0,
            verbose=False,
            occ_cutoff=1e-12,
            **_DN,
        )
        np.testing.assert_allclose(tensor[:, p], ref[:, 0], atol=1e-8, err_msg=f"polarization {eps}")


def _decoupled_model():
    """Two independent core<->impurity channels (0<->2 and 1<->3) with no cross term, so the
    cross tensor chi_{01} vanishes and chi is diagonal."""
    ec, ei = -2.0, 0.3
    return ManyBodyOperator(
        {
            ((0, "c"), (0, "a")): ec,
            ((1, "c"), (1, "a")): ec,
            ((2, "c"), (2, "a")): ei,
            ((3, "c"), (3, "a")): ei,
            ((0, "c"), (2, "a")): 0.5,
            ((2, "c"), (0, "a")): 0.5,
            ((1, "c"), (3, "a")): 0.5,
            ((3, "c"), (1, "a")): 0.5,
        }
    )


def test_diagonal_reconstruction_branch_matches_full():
    """Diagonalizable reduction (Q known, singleton groups) reproduces the full-tensor result
    when chi is diagonal -- exercising the diagonal-extraction + Q reconstruction branch."""
    op = _decoupled_model()
    gs, e0, dets = _ground_state(op, 2)
    comps = [
        ManyBodyOperator({((2, "c"), (0, "a")): 1.0}),
        ManyBodyOperator({((3, "c"), (1, "a")): 1.0}),
    ]
    w = np.linspace(-6.0, 6.0, 41)
    pols = [[1.0, 0.0], [0.0, 1.0], [1.0, 1j]]

    reduction = ComponentReduction(np.eye(2, dtype=complex), [0, 1], [0, 1], True)
    diag_chi = spectra.calc_spectra_tensor(
        op,
        comps,
        [gs],
        [e0],
        tau=0.01,
        w=w,
        basis=_basis(dets),
        delta=0.3,
        slaterWeightMin=0.0,
        verbose=False,
        occ_cutoff=1e-12,
        reduction=reduction,
        **_DN,
    )
    full_chi = spectra.calc_spectra_tensor(
        op,
        comps,
        [gs],
        [e0],
        tau=0.01,
        w=w,
        basis=_basis(dets),
        delta=0.3,
        slaterWeightMin=0.0,
        verbose=False,
        occ_cutoff=1e-12,
        reduction=None,
        **_DN,
    )
    diag = polarization.contract_spectra_tensor(diag_chi, pols)
    full = polarization.contract_spectra_tensor(full_chi, pols)
    np.testing.assert_allclose(diag, full, atol=1e-9)


def test_moment_check_falls_back_on_incorrect_reduction():
    """If a reduction wrongly groups two components with different seed moments, the safety net
    detects it and falls back to the full tensor (result stays correct)."""
    op = _model()
    gs, e0, dets = _ground_state(op, 2)
    # Two components with clearly different seed norms -> not symmetry-equivalent.
    comps = [
        ManyBodyOperator({((2, "c"), (0, "a")): 1.0, ((3, "c"), (1, "a")): 1.0}),
        ManyBodyOperator({((2, "c"), (1, "a")): 2.0, ((3, "c"), (0, "a")): 2.0}),
    ]
    w = np.linspace(-5.0, 5.0, 41)
    pols = [[1.0, 0.0], [0.0, 1.0], [1.0, 1j]]

    wrong = ComponentReduction(np.eye(2, dtype=complex), [0], [0, 0], True)  # claims 0 == 1
    chi = spectra.calc_spectra_tensor(
        op,
        comps,
        [gs],
        [e0],
        tau=0.01,
        w=w,
        basis=_basis(dets),
        delta=0.2,
        slaterWeightMin=0.0,
        verbose=False,
        occ_cutoff=1e-12,
        reduction=wrong,
        **_DN,
    )
    got = polarization.contract_spectra_tensor(chi, pols)
    for p, eps in enumerate(pols):
        combined = spectra._combine_component_ops(comps, eps)
        ref = spectra.calc_spectra(
            op,
            [combined],
            [gs],
            [e0],
            tau=0.01,
            w=w,
            basis=_basis(dets),
            delta=0.2,
            slaterWeightMin=0.0,
            verbose=False,
            occ_cutoff=1e-12,
            **_DN,
        )
        np.testing.assert_allclose(got[:, p], ref[:, 0], atol=1e-8, err_msg=f"polarization {eps}")
