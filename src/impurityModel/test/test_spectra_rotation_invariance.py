"""Rotation-invariance of spectra + degeneracy dedup (A + B2a).

A scalar spectrum ``I(w) = <gs|T^dagger (w - H)^{-1} T|gs>`` is invariant under a unitary
single-particle basis change: rotating ``H``, the transition operator ``T`` and the ground
state consistently leaves ``I(w)`` unchanged. That invariance is what makes it safe to *solve*
spectra in the cheaper symmetry-adapted basis. These tests verify:

* one-body (dipole-like) transition spectrum is identical in the spherical and the
  symmetry-adapted basis (:func:`impurity_symmetry_rotation`);
* the *summed* PES spectrum (trace of the spectral function) is basis-invariant, and equals the
  degeneracy-weighted sum over the inequivalent orbital classes -- the B2a dedup identity.
"""

from itertools import combinations

import numpy as np
from mpi4py import MPI

from impurityModel.ed import spectra
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant, applyOp, inner
from impurityModel.ed.symmetries import (
    impurity_block_structure,
    impurity_symmetry_rotation,
    rotate_hamiltonian,
)

N_ORB = 4  # 0,1 = core doublet; 2,3 = degenerate impurity doublet.


def _model():
    """Core doublet + a split impurity doublet + on-site U. Impurity block is non-diagonal, so
    impurity_symmetry_rotation returns a genuinely non-trivial rotation of orbitals 2,3."""
    ec, ei, t, u = -2.0, 0.3, 0.4, 1.5
    terms = {
        ((0, "c"), (0, "a")): ec,
        ((1, "c"), (1, "a")): ec,
        ((2, "c"), (2, "a")): ei,
        ((3, "c"), (3, "a")): ei,
        ((2, "c"), (3, "a")): t,  # impurity off-diagonal -> non-trivial eigenbasis
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
    """Lowest eigenstate at fixed particle number, as (ManyBodyState, energy, det-bytes list)."""
    dets = [_bytes(occ) for occ in combinations(range(N_ORB), n_elec)]
    states = [ManyBodyState({SlaterDeterminant.from_bytes(d): 1.0}) for d in dets]
    n = len(states)
    h = np.zeros((n, n), dtype=complex)
    for j, sj in enumerate(states):
        col = applyOp(op, sj)
        for i, si in enumerate(states):
            h[i, j] = inner(si, col)
    ev, evec = np.linalg.eigh(h)
    gs = ManyBodyState(
        {SlaterDeterminant.from_bytes(dets[i]): evec[i, 0] for i in range(n) if abs(evec[i, 0]) > 1e-14}
    )
    return gs, ev[0], dets


def _basis(dets):
    return Basis(
        impurity_orbitals={0: [list(range(N_ORB))]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=list(dets),
        verbose=False,
        comm=MPI.COMM_SELF,
    )


def _spectrum(hOp, tOps, gs, e0, dets):
    w = np.linspace(-5.0, 5.0, 51)
    return spectra.getSpectra_new(
        hOp,
        tOps,
        [gs],
        [e0],
        tau=0.01,
        w=w,
        basis=_basis(dets),
        delta=0.2,
        slaterWeightMin=0.0,
        verbose=False,
        occ_cutoff=1e-12,
        dN_imp={0: (1, 1)},
        dN_val={0: (1, 1)},
        dN_con={0: (1, 1)},
    )


def test_one_body_spectrum_is_rotation_invariant():
    op = _model()
    n_elec = 2

    # Spherical basis.
    gs, e0, dets = _ground_state(op, n_elec)
    # Dipole-like: core -> impurity (number-conserving one-body operator).
    tOp = ManyBodyOperator({((2, "c"), (0, "a")): 1.0, ((3, "c"), (1, "a")): 1.0})
    spec_sph = _spectrum(op, [tOp], gs, e0, dets)

    # Symmetry-adapted basis: rotate H, the ground state (re-solved) and the operator by the SAME W.
    W, _ = impurity_symmetry_rotation(op, [2, 3], n_orb=N_ORB)
    op_rot = rotate_hamiltonian(op, W)
    tOp_rot = rotate_hamiltonian(tOp, W)
    gs_rot, e0_rot, dets_rot = _ground_state(op_rot, n_elec)

    assert np.isclose(e0, e0_rot, atol=1e-10)  # eigenvalues are basis-invariant
    spec_rot = _spectrum(op_rot, [tOp_rot], gs_rot, e0_rot, dets_rot)
    np.testing.assert_allclose(spec_sph, spec_rot, atol=1e-9)


def test_summed_pes_spectrum_invariant_and_dedup_identity():
    op = _model()
    n_elec = 2
    gs, e0, dets = _ground_state(op, n_elec)

    # PES = removal from every impurity orbital (2,3) in the spherical basis.
    pes_sph = [ManyBodyOperator({((i, "a"),): 1.0}) for i in (2, 3)]
    spec_sph = _spectrum(op, pes_sph, gs, e0, dets)
    summed_sph = spec_sph.sum(axis=1)

    # Symmetry-adapted basis: PES operators are the rotated-orbital removals, which split into
    # inequivalent classes by impurity_block_structure. The trace (sum over all orbitals) is
    # invariant, and equals the degeneracy-weighted sum over one representative per class.
    W, _ = impurity_symmetry_rotation(op, [2, 3], n_orb=N_ORB)
    op_rot = rotate_hamiltonian(op, W)
    gs_rot, e0_rot, dets_rot = _ground_state(op_rot, n_elec)

    pes_rot = [ManyBodyOperator({((i, "a"),): 1.0}) for i in (2, 3)]
    spec_rot = _spectrum(op_rot, pes_rot, gs_rot, e0_rot, dets_rot)
    summed_rot = spec_rot.sum(axis=1)

    # Trace of the spectral function is basis-invariant.
    np.testing.assert_allclose(summed_sph, summed_rot, atol=1e-9)

    # Dedup identity: sum over inequivalent classes weighted by class size reproduces the total.
    bs = impurity_block_structure(op_rot, [2, 3], n_orb=N_ORB)
    orbital_of_block = {b: blk[0] for b, blk in enumerate(bs.blocks)}
    summed_dedup = np.zeros_like(summed_rot)
    for rep in bs.inequivalent_blocks:
        members = next(cls for cls in bs.identical_blocks if rep in cls)
        rep_orb = orbital_of_block[rep]
        col = rep_orb - 2  # impurity orbitals 2,3 -> operator columns 0,1
        summed_dedup += len(members) * spec_rot[:, col]
    np.testing.assert_allclose(summed_rot, summed_dedup, atol=1e-9)


def _swap_symmetric_model():
    """Two impurity orbitals (0,1) symmetric under 0<->1, each hopping to a shared bath (2).
    G_00 = G_11 exactly, so PES on 0 and 1 are one equivalence class."""
    ei, eb, v, u = -0.5, 0.8, 0.6, 1.2
    terms = {
        ((0, "c"), (0, "a")): ei,
        ((1, "c"), (1, "a")): ei,
        ((2, "c"), (2, "a")): eb,
        ((0, "c"), (1, "c"), (1, "a"), (0, "a")): u,
    }
    for a in (0, 1):
        terms[((a, "c"), (2, "a"))] = v
        terms[((2, "c"), (a, "a"))] = v
    return ManyBodyOperator(terms)


def test_equivalence_groups_dedup_matches_full():
    """getSpectra_new with equivalence_groups (compute one rep, broadcast) reproduces the
    per-operator computation for genuinely degenerate operators."""
    global N_ORB
    saved = N_ORB
    N_ORB = 3
    try:
        op = _swap_symmetric_model()
        gs, e0, dets = _ground_state(op, 2)
        pes = [ManyBodyOperator({((i, "a"),): 1.0}) for i in (0, 1)]

        full = _spectrum(op, pes, gs, e0, dets)
        # 0 and 1 are symmetry-equivalent -> one class.
        deduped = spectra.getSpectra_new(
            op,
            pes,
            [gs],
            [e0],
            tau=0.01,
            w=np.linspace(-5.0, 5.0, 51),
            basis=_basis(dets),
            delta=0.2,
            slaterWeightMin=0.0,
            verbose=False,
            occ_cutoff=1e-12,
            dN_imp={0: (1, 1)},
            dN_val={0: (1, 1)},
            dN_con={0: (1, 1)},
            equivalence_groups=[0, 0],
        )
        # The two columns are genuinely equal (degeneracy), and dedup reproduces the full result.
        np.testing.assert_allclose(full[:, 0], full[:, 1], atol=1e-9)
        np.testing.assert_allclose(full, deduped, atol=1e-12)
    finally:
        N_ORB = saved
