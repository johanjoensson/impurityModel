"""End-to-end correctness of conserved-charge sector confinement in getSpectra_new (B1).

Confining each transition-operator Lanczos to the conserved-charge sector of ``tOp|gs>`` only
removes determinants that are symmetry-unreachable from the seed, so it must leave the spectrum
numerically unchanged while (by construction) pruning the excited basis. We check that the
confined spectrum equals the spectrum with confinement disabled, on a 4-orbital Anderson model
whose per-shell occupation window is deliberately loose enough to admit out-of-sector states.
"""

from itertools import combinations

import numpy as np
from mpi4py import MPI

import impurityModel.ed.product_state_representation as psr
from impurityModel.ed import spectra
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant


def _anderson_4():
    """0=imp_up, 1=imp_dn, 2=bath_up, 3=bath_dn. Spin-diagonal hopping + on-site U."""
    eps_i, eps_b, v, u = -1.0, 0.5, 0.7, 3.0
    terms = {
        ((0, "c"), (0, "a")): eps_i,
        ((1, "c"), (1, "a")): eps_i,
        ((2, "c"), (2, "a")): eps_b,
        ((3, "c"), (3, "a")): eps_b,
    }
    for a, b in ((0, 2), (1, 3)):
        terms[((a, "c"), (b, "a"))] = v
        terms[((b, "c"), (a, "a"))] = v
    terms[((0, "c"), (1, "c"), (1, "a"), (0, "a"))] = u
    return ManyBodyOperator(terms)


def _bytes(occ):
    b = bytearray(1)
    for o in occ:
        b[0] |= 1 << (7 - o)
    return bytes(b)


def _ground_state_half_filled(op):
    """Lowest eigenstate in the (N_up=1, N_dn=1) sector, as (ManyBodyState, energy)."""
    # up in {0,2}, down in {1,3}: one electron each.
    sector = [(u, d) for u in (0, 2) for d in (1, 3)]
    dets = [_bytes(occ) for occ in sector]
    states = [ManyBodyState({SlaterDeterminant.from_bytes(d): 1.0}) for d in dets]
    n = len(states)
    h = np.zeros((n, n), dtype=complex)
    from impurityModel.ed.ManyBodyUtils import applyOp, inner

    for j, sj in enumerate(states):
        col = applyOp(op, sj)
        for i, si in enumerate(states):
            h[i, j] = inner(si, col)
    ev, evec = np.linalg.eigh(h)
    gs = ManyBodyState(
        {SlaterDeterminant.from_bytes(dets[i]): evec[i, 0] for i in range(n) if abs(evec[i, 0]) > 1e-14}
    )
    return gs, ev[0], dets


def _run(hOp, tOps, gs, e0, basis, confine, monkeypatch):
    w = np.linspace(-6.0, 6.0, 41)
    if not confine:
        monkeypatch.setattr(spectra, "_sector_restrictions_per_top", lambda *a, **k: None)
    return spectra.getSpectra_new(
        hOp,
        tOps,
        [gs],
        [e0],
        tau=0.01,
        w=w,
        basis=basis,
        delta=0.2,
        slaterWeightMin=0.0,
        verbose=False,
        occ_cutoff=1e-12,
        # Deliberately loose windows: admit a hole AND a particle in every shell, so the
        # unconfined excited basis wanders into wrong-Sz / wrong-N sectors that confinement prunes.
        dN_imp={0: (1, 1)},
        dN_val={0: (1, 1)},
        dN_con={0: (1, 1)},
    )


def _fresh_basis(dets):
    return Basis(
        impurity_orbitals={0: [[0, 1]]},
        bath_states=({0: [[2, 3]]}, {0: [[]]}),
        initial_basis=list(dets),
        verbose=False,
        comm=MPI.COMM_SELF,
    )


def test_sector_confinement_preserves_pes_spectrum(monkeypatch):
    op = _anderson_4()
    gs, e0, dets = _ground_state_half_filled(op)

    # Removal (PES-like) of the impurity up electron: definite sector shift N_up -> 0.
    tOps = [ManyBodyOperator({((0, "a"),): 1.0})]

    confined = _run(op, tOps, gs, e0, _fresh_basis(dets), confine=True, monkeypatch=monkeypatch)
    unconfined = _run(op, tOps, gs, e0, _fresh_basis(dets), confine=False, monkeypatch=monkeypatch)

    np.testing.assert_allclose(confined, unconfined, atol=1e-10)


def test_sector_confinement_preserves_ips_spectrum(monkeypatch):
    op = _anderson_4()
    gs, e0, dets = _ground_state_half_filled(op)

    # Addition (IPS-like) of an impurity down electron: definite sector shift N_dn -> 2.
    tOps = [ManyBodyOperator({((1, "c"),): 1.0})]

    confined = _run(op, tOps, gs, e0, _fresh_basis(dets), confine=True, monkeypatch=monkeypatch)
    unconfined = _run(op, tOps, gs, e0, _fresh_basis(dets), confine=False, monkeypatch=monkeypatch)

    np.testing.assert_allclose(confined, unconfined, atol=1e-10)


def test_sector_restrictions_engage_and_confine():
    """The mechanism actually fires: a definite-sector operator yields a tightening restriction."""
    op = _anderson_4()
    gs, e0, dets = _ground_state_half_filled(op)
    basis = _fresh_basis(dets)
    tOps = [ManyBodyOperator({((0, "a"),): 1.0})]

    sector = spectra._sector_restrictions_per_top(op, tOps, [gs], basis)
    assert sector is not None
    restr = sector[0]
    assert restr is not None
    # c_0 removes a spin-up electron: N_up = {0,2} pinned to 0, N_dn = {1,3} pinned to 1.
    assert restr[frozenset({0, 2})] == (0, 0)
    assert restr[frozenset({1, 3})] == (1, 1)
