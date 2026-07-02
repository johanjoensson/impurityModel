"""RIXS correctness backbone (T0 for the RIXS tensor refactor).

:func:`spectra.getRIXSmap_new` implements the Kramers-Heisenberg map

    A_{ij}(w_in, w_loss) = sum_g (weight_g / Z)
        <g| Tin_i^dagger R1(w_in) Tout_j^dagger R2(w_loss) Tout_j R1(w_in) Tin_i |g>,

with R1 = (w_in + i d1 + E_g - H)^-1 and R2 = (w_loss + i d2 + E_g - H)^-1.

These tests pin that behaviour with an independent dense reference (so the in-progress
tensor refactor is checked against physics, not just against itself) and verify that the
map's component sum is invariant under a single-particle basis rotation.
"""

from itertools import combinations

import numpy as np
from mpi4py import MPI

from impurityModel.ed import spectra
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant, applyOp, inner
from impurityModel.ed.symmetries import impurity_symmetry_rotation, rotate_hamiltonian

# orbitals 0,1 = impurity "3d" (block key 2); orbital 2 = core "2p" (block key 1)
N_ORB = 3
TAU = 0.02
D1, D2 = 0.4, 0.2
WIN = np.array([-7.0, -6.5])
WLOSS = np.linspace(-1.0, 3.0, 9)


def _model():
    ei, ec, t, u = 0.5, -8.0, 0.3, 2.0
    return ManyBodyOperator(
        {
            ((0, "c"), (0, "a")): ei,
            ((1, "c"), (1, "a")): ei + 0.2,
            ((2, "c"), (2, "a")): ec,
            ((0, "c"), (1, "a")): t,
            ((1, "c"), (0, "a")): t,
            ((0, "c"), (1, "c"), (1, "a"), (0, "a")): u,
        }
    )


def _bytes(occ):
    b = bytearray(1)
    for o in occ:
        b[0] |= 1 << (7 - o)
    return bytes(b)


def _dets(ne):
    return [_bytes(o) for o in combinations(range(N_ORB), ne)]


def _states(dets):
    return [ManyBodyState({SlaterDeterminant.from_bytes(d): 1.0}) for d in dets]


def _matrix(op, states):
    n = len(states)
    m = np.zeros((n, n), dtype=complex)
    for j, sj in enumerate(states):
        col = applyOp(op, sj)
        for i, si in enumerate(states):
            m[i, j] = inner(si, col)
    return m


def _thermal_states(op, ne):
    dets = _dets(ne)
    states = _states(dets)
    h = _matrix(op, states)
    ev, vec = np.linalg.eigh(h)
    psis = [
        ManyBodyState(
            {SlaterDeterminant.from_bytes(dets[i]): vec[i, k] for i in range(len(dets)) if abs(vec[i, k]) > 1e-14}
        )
        for k in range(len(ev))
    ]
    return psis, list(ev), dets, states, vec


def _basis(dets):
    return Basis(
        impurity_orbitals={2: [[0, 1]], 1: [[2]]},
        bath_states=({2: [[]], 1: [[]]}, {2: [[]], 1: [[]]}),
        initial_basis=list(dets),
        verbose=False,
        comm=MPI.COMM_SELF,
    )


def _dense_rixs(op, ne, tin, tout, es, vecs, states):
    """Independent dense Kramers-Heisenberg map, thermally averaged."""
    n = len(states)
    eye = np.eye(n, dtype=complex)
    H = _matrix(op, states)
    Tin = [_matrix(t, states) for t in tin]
    Tout = [_matrix(t, states) for t in tout]
    e0 = min(es)
    Z = float(np.sum(np.exp(-(np.asarray(es) - e0) / TAU)))
    out = np.zeros((len(tin), len(tout), len(WIN), len(WLOSS)), dtype=complex)
    for g, eg in enumerate(es):
        gvec = vecs[:, g]
        wg = np.exp(-(eg - e0) / TAU)
        for i in range(len(tin)):
            for ki, win in enumerate(WIN):
                psi2 = np.linalg.solve((win + 1j * D1 + eg) * eye - H, Tin[i] @ gvec)
                for j in range(len(tout)):
                    psi3 = Tout[j] @ psi2
                    for kl, wl in enumerate(WLOSS):
                        r2 = np.linalg.solve((wl + 1j * D2 + eg) * eye - H, psi3)
                        out[i, j, ki, kl] += wg * (psi3.conj() @ r2)
    return out / Z


def _tin_tout():
    tin = [ManyBodyOperator({((0, "c"), (2, "a")): 1.0}), ManyBodyOperator({((1, "c"), (2, "a")): 1.0})]
    tout = [ManyBodyOperator({((2, "c"), (0, "a")): 1.0}), ManyBodyOperator({((2, "c"), (1, "a")): 1.0})]
    return tin, tout


def _run_rixs(op, psis, es, tin, tout, dets):
    return spectra.getRIXSmap_new(
        op, tin, tout, psis, es, tau=TAU, wIns=WIN, wLoss=WLOSS,
        delta1=D1, delta2=D2, basis=_basis(dets), verbose=False, slaterWeightMin=0.0,
    )


def test_rixs_matches_dense_reference():
    op = _model()
    psis, es, dets, states, vecs = _thermal_states(op, 2)
    tin, tout = _tin_tout()
    got = _run_rixs(op, psis, es, tin, tout, dets)
    ref = _dense_rixs(op, 2, tin, tout, es, vecs, states)
    np.testing.assert_allclose(got, ref, atol=1e-9)


def test_rixs_component_sum_is_rotation_invariant():
    op = _model()
    psis, es, dets, states, vecs = _thermal_states(op, 2)
    tin, tout = _tin_tout()
    base = _run_rixs(op, psis, es, tin, tout, dets).sum(axis=(0, 1))

    # Rotate the impurity block (0,1); rotate H, the operators and re-solve the states.
    W, _ = impurity_symmetry_rotation(op, [0, 1], n_orb=N_ORB)
    op_rot = rotate_hamiltonian(op, W)
    tin_rot = [ManyBodyOperator(spectra._rotate_op_dict(t.to_dict(), W)) for t in tin]
    tout_rot = [ManyBodyOperator(spectra._rotate_op_dict(t.to_dict(), W)) for t in tout]
    psis_rot, es_rot, dets_rot, _, _ = _thermal_states(op_rot, 2)
    rot = _run_rixs(op_rot, psis_rot, es_rot, tin_rot, tout_rot, dets_rot).sum(axis=(0, 1))

    np.testing.assert_allclose(base, rot, atol=1e-8)


# --- tests for the full rank-4 polarization tensor (getRIXSmap_tensor) ---

# In/out polarization vectors (length = #components = 2), including non-axis and circular ones.
EPS_IN = [[1.0, 0.0], [0.0, 1.0], [1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), 1j / np.sqrt(2)]]
EPS_OUT = [[1.0, 0.0], [0.0, 1.0], [1 / np.sqrt(2), -1 / np.sqrt(2)], [1 / np.sqrt(2), 1j / np.sqrt(2)]]


def _dense_rixs_pol(op, tin_comp, tout_comp, epsIn, epsOut, es, vecs, states):
    """Independent dense Kramers-Heisenberg map contracted with arbitrary in/out polarizations.

    T_in(e_in)  = sum_a e_in[a]      Tin_a      (in operators carry no dagger)
    T_out(e_out)= sum_b e_out[b]^*   Tout_b     (out operators are the daggered dipole components)
    A(e_in, e_out) = sum_g (w_g / Z) <g| T_in^dag R1 T_out^dag R2 T_out R1 T_in |g>.
    """
    n = len(states)
    eye = np.eye(n, dtype=complex)
    H = _matrix(op, states)
    Tin = [_matrix(t, states) for t in tin_comp]
    Tout = [_matrix(t, states) for t in tout_comp]
    nin, nout = len(Tin), len(Tout)
    ein = np.asarray(epsIn, dtype=complex)
    eout = np.asarray(epsOut, dtype=complex)
    e0 = min(es)
    Z = float(np.sum(np.exp(-(np.asarray(es) - e0) / TAU)))
    out = np.zeros((len(epsIn), len(epsOut), len(WIN), len(WLOSS)), dtype=complex)
    for g, eg in enumerate(es):
        gvec = vecs[:, g]
        wg = np.exp(-(eg - e0) / TAU)
        for ki, win in enumerate(WIN):
            psi2 = [np.linalg.solve((win + 1j * D1 + eg) * eye - H, Tin[a] @ gvec) for a in range(nin)]
            for pin, e_in in enumerate(ein):
                psi2_in = sum(e_in[a] * psi2[a] for a in range(nin))
                for pout, e_out in enumerate(eout):
                    psi3 = sum(np.conj(e_out[b]) * (Tout[b] @ psi2_in) for b in range(nout))
                    for kl, wl in enumerate(WLOSS):
                        r2 = np.linalg.solve((wl + 1j * D2 + eg) * eye - H, psi3)
                        out[pin, pout, ki, kl] += wg * (psi3.conj() @ r2)
    return out / Z


def _run_rixs_tensor(op, psis, es, tin, tout, dets, epsIn, epsOut):
    return spectra.getRIXSmap_tensor(
        op, tin, tout, epsIn, epsOut, psis, es, tau=TAU, wIns=WIN, wLoss=WLOSS,
        delta1=D1, delta2=D2, basis=_basis(dets), verbose=False, slaterWeightMin=0.0,
    )


def test_rixs_tensor_matches_dense_reference():
    """Full tensor contracted with arbitrary/circular polarizations vs an independent dense KH."""
    op = _model()
    psis, es, dets, states, vecs = _thermal_states(op, 2)
    tin, tout = _tin_tout()
    got = _run_rixs_tensor(op, psis, es, tin, tout, dets, EPS_IN, EPS_OUT)
    ref = _dense_rixs_pol(op, tin, tout, EPS_IN, EPS_OUT, es, vecs, states)
    np.testing.assert_allclose(got, ref, atol=1e-8)


def test_rixs_tensor_matches_getRIXSmap_new():
    """Contracting the tensor reproduces the validated per-pair getRIXSmap_new for the same
    polarizations (built as the linear combinations of the component operators)."""
    op = _model()
    psis, es, dets, states, vecs = _thermal_states(op, 2)
    tin, tout = _tin_tout()
    # Per-pair reference: T_in(e) = sum e_a Tin_a; T_out(e) = sum e_b^* Tout_b (daggered dipole).
    ref_in = [spectra._combine_component_ops(tin, e) for e in EPS_IN]
    ref_out = [spectra._combine_component_ops(tout, np.conj(e)) for e in EPS_OUT]
    ref = _run_rixs(op, psis, es, ref_in, ref_out, dets)
    got = _run_rixs_tensor(op, psis, es, tin, tout, dets, EPS_IN, EPS_OUT)
    np.testing.assert_allclose(got, ref, atol=1e-8)


def test_rixs_tensor_is_rotation_invariant():
    """Summed over a complete orthonormal polarization basis, the tensor map (a trace over the
    component span) is invariant under a single-particle basis rotation."""
    ident = np.eye(2, dtype=complex)
    op = _model()
    psis, es, dets, states, vecs = _thermal_states(op, 2)
    tin, tout = _tin_tout()
    base = _run_rixs_tensor(op, psis, es, tin, tout, dets, ident, ident).sum(axis=(0, 1))

    W, _ = impurity_symmetry_rotation(op, [0, 1], n_orb=N_ORB)
    op_rot = rotate_hamiltonian(op, W)
    tin_rot = [ManyBodyOperator(spectra._rotate_op_dict(t.to_dict(), W)) for t in tin]
    tout_rot = [ManyBodyOperator(spectra._rotate_op_dict(t.to_dict(), W)) for t in tout]
    psis_rot, es_rot, dets_rot, _, _ = _thermal_states(op_rot, 2)
    rot = _run_rixs_tensor(op_rot, psis_rot, es_rot, tin_rot, tout_rot, dets_rot, ident, ident).sum(axis=(0, 1))

    np.testing.assert_allclose(base, rot, atol=1e-8)
