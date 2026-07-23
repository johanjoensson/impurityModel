"""Tests for the dynamical susceptibility driver (impurityModel.ed.susceptibility).

The Lanczos-resolvent driver is validated against exact Lehmann sums from a dense
diagonalization of a 4-spin-orbital Anderson model (both on the real axis and on the
bosonic Matsubara mesh), and against closed forms on an atomic-limit doublet where the
degenerate-manifold projection removes the entire response (pure Curie).
"""

from itertools import combinations

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import susceptibility
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import (
    ManyBodyBlockState,
    ManyBodyOperator,
    ManyBodyState,
    SlaterDeterminant,
    applyOp,
    inner,
)

N_ORB = 4  # 0 = imp_dn, 1 = imp_up, 2 = bath_dn, 3 = bath_up


def _bytes(occ):
    b = bytearray(1)
    for o in occ:
        b[0] |= 1 << (7 - o)
    return bytes(b)


def _all_determinants():
    dets = []
    for k in range(N_ORB + 1):
        for occ in combinations(range(N_ORB), k):
            dets.append(_bytes(occ))
    return dets


def _dense(op, dets):
    """Matrix of a ManyBodyOperator in the determinant basis ``dets``."""
    states = [ManyBodyState({SlaterDeterminant.from_bytes(d): 1.0}) for d in dets]
    n = len(states)
    m = np.zeros((n, n), dtype=complex)
    for j, sj in enumerate(states):
        col = applyOp(op, sj)
        for i, si in enumerate(states):
            m[i, j] = inner(si, col)
    return m


def _anderson():
    """imp (0 dn, 1 up) hybridized with one bath level (2 dn, 3 up), U on the impurity."""
    eps_i, eps_b, v, u = -1.0, 0.4, 0.7, 3.0
    terms = {
        ((0, "c"), (0, "a")): eps_i,
        ((1, "c"), (1, "a")): eps_i,
        ((2, "c"), (2, "a")): eps_b,
        ((3, "c"), (3, "a")): eps_b,
    }
    for a, b in ((0, 2), (1, 3)):
        terms[((a, "c"), (b, "a"))] = v
        terms[((b, "c"), (a, "a"))] = v
    terms[((1, "c"), (0, "c"), (0, "a"), (1, "a"))] = u
    return ManyBodyOperator(terms)


def _operators():
    s_z = ManyBodyOperator({((1, "c"), (1, "a")): 0.5, ((0, "c"), (0, "a")): -0.5})
    s_plus = ManyBodyOperator({((1, "c"), (0, "a")): 1.0})
    s_minus = ManyBodyOperator({((0, "c"), (1, "a")): 1.0})
    n_imp = ManyBodyOperator({((0, "c"), (0, "a")): 1.0, ((1, "c"), (1, "a")): 1.0})
    return {"spin_z": (s_z, s_z), "charge": (n_imp, n_imp), "transverse": (s_plus, s_minus)}


def _retained(E, V, dets, tol=1e-6):
    """The lowest (near-)degenerate manifold as ManyBodyStates + energies."""
    keep = np.where(E - E[0] <= tol)[0]
    psis = []
    for k in keep:
        psis.append(
            ManyBodyState(
                {SlaterDeterminant.from_bytes(dets[i]): V[i, k] for i in range(len(dets)) if abs(V[i, k]) > 1e-14}
            )
        )
    return psis, E[keep]


def _lehmann_chi(zs, a_plus, a_minus, E, retained_idx, weights, tol=1e-6):
    """Exact regular susceptibility: Lehmann sum excluding the degenerate manifold."""
    chi = np.zeros(len(zs), dtype=complex)
    for w_n, n in zip(weights, retained_idx):
        for m in range(len(E)):
            if abs(E[m] - E[n]) <= tol:
                continue
            wmn = E[m] - E[n]
            chi += w_n * (abs(a_plus[m, n]) ** 2 / (zs - wmn) - abs(a_minus[m, n]) ** 2 / (zs + wmn))
    return chi


def _curie_ref(a_plus, E, retained_idx, weights, tol=1e-6):
    c2 = 0.0
    diag = 0.0
    for w_n, n in zip(weights, retained_idx):
        for m in retained_idx:
            if abs(E[m] - E[n]) <= tol:
                c2 += w_n * abs(a_plus[m, n]) ** 2
        diag += w_n * a_plus[n, n]
    return c2 - abs(diag) ** 2


def _basis(dets):
    return Basis(
        impurity_orbitals={0: [[0, 1]]},
        bath_states=({0: [[2, 3]]}, {0: [[]]}),
        initial_basis=list(dets),
        verbose=False,
        comm=MPI.COMM_SELF,
    )


def test_susceptibility_matches_dense_lehmann():
    """Driver chi (real axis + Matsubara) == exact Lehmann sums from dense eigenstates."""
    hOp = _anderson()
    dets = _all_determinants()
    h = _dense(hOp, dets)
    E, V = np.linalg.eigh(h)
    psis, es = _retained(E, V, dets)
    tau = 0.02
    weights = np.exp(-(es - es[0]) / tau)
    weights /= weights.sum()

    w = np.linspace(-6.0, 6.0, 61)
    delta = 0.1
    nu = susceptibility.bosonic_matsubara_mesh(tau, 6)
    ops = _operators()
    res = susceptibility.calc_susceptibility(
        hOp, psis, es, tau, _basis(dets), w, delta, matsubara_mesh=nu, operators=ops
    )
    assert res is not None

    retained_idx = list(range(len(es)))  # dense indices 0..len-1 are the retained states
    for name, (a_p_op, a_m_op) in ops.items():
        # Lehmann sums need the operator matrices in the eigenbasis.
        a_p = V.conj().T @ _dense(a_p_op, dets) @ V
        a_m = V.conj().T @ _dense(a_m_op, dets) @ V
        chi_w_ref = _lehmann_chi(w + 1j * delta, a_p, a_m, E, retained_idx, weights)
        chi_nu_ref = _lehmann_chi(1j * nu, a_p, a_m, E, retained_idx, weights)
        entry = res["operators"][name]
        np.testing.assert_allclose(entry["realaxis"], chi_w_ref, atol=1e-8, err_msg=name)
        np.testing.assert_allclose(entry["matsubara"], chi_nu_ref, atol=1e-8, err_msg=name)
        assert entry["curie_coefficient"] == pytest.approx(_curie_ref(a_p, E, retained_idx, weights), abs=1e-10)
        # The nu = 0 point is the (regular) Van Vleck susceptibility: real and finite.
        assert abs(entry["matsubara"][0].imag) < 1e-10
        assert np.isfinite(entry["matsubara"][0].real)


def test_susceptibility_su2_transverse_is_twice_longitudinal():
    """SU(2) symmetry: chi_+-(z) = 2 chi_zz(z) (regular parts and Curie weights alike)."""
    hOp = _anderson()
    dets = _all_determinants()
    E, V = np.linalg.eigh(_dense(hOp, dets))
    psis, es = _retained(E, V, dets)
    tau = 0.02
    w = np.linspace(-5.0, 5.0, 41)
    nu = susceptibility.bosonic_matsubara_mesh(tau, 4)
    ops = _operators()
    res = susceptibility.calc_susceptibility(
        hOp, psis, es, tau, _basis(dets), w, 0.08, matsubara_mesh=nu, operators=ops
    )
    zz = res["operators"]["spin_z"]
    pm = res["operators"]["transverse"]
    np.testing.assert_allclose(pm["realaxis"], 2.0 * zz["realaxis"], atol=1e-8)
    np.testing.assert_allclose(pm["matsubara"], 2.0 * zz["matsubara"], atol=1e-8)
    assert pm["curie_coefficient"] == pytest.approx(2.0 * zz["curie_coefficient"], abs=1e-10)


def test_susceptibility_atomic_doublet_is_pure_curie():
    """Atomic limit (no hybridization), N = 1 doublet: the entire spin response is elastic.

    chi_reg vanishes identically for Sz and S+- (all matrix elements stay inside the
    degenerate manifold, which the seed projection removes); the Curie coefficients carry
    the free moment: C_zz = 1/4 and C_+- = sum_n w_n <n|S-S+|n> = 1/2 (= 2 C_zz, the
    SU(2) relation: only |dn> can absorb S+). The charge is frozen: C_N = 0, chi_N = 0.
    """
    eps = -1.0
    hOp = ManyBodyOperator(
        {
            ((0, "c"), (0, "a")): eps,
            ((1, "c"), (1, "a")): eps,
            ((1, "c"), (0, "c"), (0, "a"), (1, "a")): 5.0,
        }
    )
    dets = [_bytes(occ) for k in range(3) for occ in combinations(range(2), k)]
    E, V = np.linalg.eigh(_dense(hOp, dets))
    psis, es = _retained(E, V, dets)
    assert len(psis) == 2  # the |dn>, |up> doublet
    tau = 0.03
    w = np.linspace(-2.0, 2.0, 21)
    nu = susceptibility.bosonic_matsubara_mesh(tau, 3)
    basis = Basis(
        impurity_orbitals={0: [[0, 1]]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=list(dets),
        verbose=False,
        comm=MPI.COMM_SELF,
    )
    res = susceptibility.calc_susceptibility(
        hOp, psis, es, tau, basis, w, 0.05, matsubara_mesh=nu, operators=_operators()
    )
    for name, curie in (("spin_z", 0.25), ("transverse", 0.5), ("charge", 0.0)):
        entry = res["operators"][name]
        np.testing.assert_allclose(entry["realaxis"], 0.0, atol=1e-10, err_msg=name)
        np.testing.assert_allclose(entry["matsubara"], 0.0, atol=1e-10, err_msg=name)
        assert entry["curie_coefficient"] == pytest.approx(curie, abs=1e-10), name


def test_bosonic_matsubara_mesh():
    nu = susceptibility.bosonic_matsubara_mesh(0.5, 3)
    np.testing.assert_allclose(nu, [0.0, np.pi, 2 * np.pi])


@pytest.mark.mpi
def test_susceptibility_distributed_matches_dense_lehmann():
    """The distributed driver reproduces the exact Lehmann sums at any rank count.

    States are fed on rank 0 ONLY and distributed with redistribute_psis (each
    determinant owned by exactly one rank); feeding replicated copies on every rank
    would double-count amplitudes.
    """
    comm = MPI.COMM_WORLD
    hOp = _anderson()
    dets = _all_determinants()
    E, V = np.linalg.eigh(_dense(hOp, dets))
    keep = np.where(E - E[0] <= 1e-6)[0]
    es = E[keep]
    tau = 0.02
    weights = np.exp(-(es - es[0]) / tau)
    weights /= weights.sum()

    basis = Basis(
        impurity_orbitals={0: [[0, 1]]},
        bath_states=({0: [[2, 3]]}, {0: [[]]}),
        initial_basis=list(dets),
        verbose=False,
        comm=comm,
    )
    if comm.rank == 0:
        psis = [
            ManyBodyState(
                {SlaterDeterminant.from_bytes(dets[i]): V[i, k] for i in range(len(dets)) if abs(V[i, k]) > 1e-14}
            )
            for k in keep
        ]
    else:
        psis = [ManyBodyState({}) for _ in keep]
    # Each seed goes through its own explicit width-1 block rather than a bare
    # ManyBodyState({}) placeholder on the non-owning rank: once the flat and block
    # classes merge (Phase 7 step 3), a bare placeholder is the width-0 polymorphic
    # zero, an asymmetric mismatch against the owning rank's populated (eventually
    # width-1) seeds that would deadlock redistribute_psis' collective.
    psi_blocks = [ManyBodyBlockState.from_states([psi]) for psi in psis]
    psis = [blk.to_states()[0] for blk in basis.redistribute_psis(psi_blocks)]

    w = np.linspace(-6.0, 6.0, 31)
    delta = 0.1
    nu = susceptibility.bosonic_matsubara_mesh(tau, 4)
    ops = _operators()
    res = susceptibility.calc_susceptibility(hOp, psis, es, tau, basis, w, delta, matsubara_mesh=nu, operators=ops)

    if comm.rank == 0:
        retained_idx = list(range(len(es)))
        for name, (a_p_op, a_m_op) in ops.items():
            a_p = V.conj().T @ _dense(a_p_op, dets) @ V
            a_m = V.conj().T @ _dense(a_m_op, dets) @ V
            chi_w_ref = _lehmann_chi(w + 1j * delta, a_p, a_m, E, retained_idx, weights)
            chi_nu_ref = _lehmann_chi(1j * nu, a_p, a_m, E, retained_idx, weights)
            entry = res["operators"][name]
            np.testing.assert_allclose(entry["realaxis"], chi_w_ref, atol=1e-8, err_msg=name)
            np.testing.assert_allclose(entry["matsubara"], chi_nu_ref, atol=1e-8, err_msg=name)
            assert entry["curie_coefficient"] == pytest.approx(
                _curie_ref(a_p, E, retained_idx, weights), abs=1e-10
            ), name
    else:
        assert res is None


def test_save_and_summary_roundtrip(tmp_path, capsys):
    """chi.h5 layout round-trips and the rank-0 summary prints the decomposition."""
    h5py = pytest.importorskip("h5py")

    tau = 0.02
    w = np.linspace(-2.0, 2.0, 5)
    nu = susceptibility.bosonic_matsubara_mesh(tau, 3)
    result = {
        "w": w,
        "matsubara": nu,
        "tau": tau,
        "delta": 0.05,
        "skipped": {"transverse": "no trustworthy (dn, up) spin labelling"},
        "operators": {
            "spin_z": {
                "realaxis": np.linspace(0, 1, 5) - 0.1j,
                "matsubara": np.array([0.3 + 0j, 0.2 + 0j, 0.1 + 0j]),
                "curie_coefficient": 0.25,
                "expectation": 0.0 + 0j,
            }
        },
    }
    path = tmp_path / "chi.h5"
    susceptibility.save_susceptibility(result, path)
    with h5py.File(path, "r") as h5f:
        np.testing.assert_allclose(h5f["w"][()], w)
        np.testing.assert_allclose(h5f["matsubara_mesh"][()], nu)
        np.testing.assert_allclose(h5f["chi/spin_z/realaxis"][()], result["operators"]["spin_z"]["realaxis"])
        np.testing.assert_allclose(h5f["chi/spin_z/matsubara"][()], result["operators"]["spin_z"]["matsubara"])
        assert h5f["chi/spin_z"].attrs["curie_coefficient"] == pytest.approx(0.25)
        assert h5f.attrs["tau"] == pytest.approx(tau)
        assert "no trustworthy" in h5f["chi"].attrs["skipped_transverse"]

    susceptibility.print_susceptibility_summary(result)
    out = capsys.readouterr().out
    assert "Impurity susceptibilities" in out
    assert "chi(0) = Curie/tau + VanVleck" in out
    # chi(0) = 0.25/0.02 + Re chi(i nu = 0) = 12.5 + 0.3.
    line = next(ln for ln in out.splitlines() if ln.lstrip().startswith("spin_z"))
    assert "12.8" in line
    assert "transverse: skipped" in out


def test_build_operators_and_auto_path():
    """Operator auto-build on the SU(2) SIAM: all four channels, chi_+- = 2 chi_zz.

    The impurity (0 dn, 1 up) is a spin-doubled l=0 shell: the Casimir build supplies
    Sz and an (identically zero) Lz, and the validated down-then-up pairing supplies
    S+/S-, so orb_z must come out as exactly zero response and transverse must obey
    the SU(2) relation against spin_z.
    """
    hOp = _anderson()
    dets = _all_determinants()
    E, V = np.linalg.eigh(_dense(hOp, dets))
    psis, es = _retained(E, V, dets)
    basis = _basis(dets)

    ops, skipped = susceptibility.build_susceptibility_operators(hOp, basis, np.eye(2, dtype=complex))
    assert set(ops) == {"charge", "spin_z", "orb_z", "transverse"}
    assert skipped == {}

    tau = 0.02
    w = np.linspace(-4.0, 4.0, 17)
    res = susceptibility.calc_susceptibility(
        hOp, psis, es, tau, basis, w, 0.1, rot_to_spherical=np.eye(2, dtype=complex)
    )
    np.testing.assert_allclose(res["operators"]["orb_z"]["realaxis"], 0.0, atol=1e-12)
    assert res["operators"]["orb_z"]["curie_coefficient"] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(
        res["operators"]["transverse"]["realaxis"], 2.0 * res["operators"]["spin_z"]["realaxis"], atol=1e-8
    )


def test_polarized_bath_transverse_matches_dense_lehmann():
    """Collinear spin-polarized bath (RSPt-style): chi_+- from the auto-built impurity
    S+/S- is exact and matches the dense Lehmann sum (only the impurity pairing enters)."""
    eps_i, v = -1.0, 0.5
    terms = {
        ((0, "c"), (0, "a")): eps_i,
        ((1, "c"), (1, "a")): eps_i,
        ((2, "c"), (2, "a")): -0.8,  # spin-split bath energies
        ((3, "c"), (3, "a")): 0.6,
        ((1, "c"), (0, "c"), (0, "a"), (1, "a")): 3.0,
    }
    for a, b, vv in ((0, 2, v), (1, 3, 0.4)):  # spin-split hoppings
        terms[((a, "c"), (b, "a"))] = vv
        terms[((b, "c"), (a, "a"))] = vv
    hOp = ManyBodyOperator(terms)
    dets = _all_determinants()
    E, V = np.linalg.eigh(_dense(hOp, dets))
    psis, es = _retained(E, V, dets)
    tau = 0.02
    weights = np.exp(-(es - es[0]) / tau)
    weights /= weights.sum()
    basis = _basis(dets)

    ops, skipped = susceptibility.build_susceptibility_operators(hOp, basis, np.eye(2, dtype=complex))
    assert "transverse" in ops and "transverse" not in skipped

    w = np.linspace(-4.0, 4.0, 33)
    delta = 0.1
    res = susceptibility.calc_susceptibility(
        hOp, psis, es, tau, basis, w, delta, operators={"transverse": ops["transverse"]}
    )
    a_p = V.conj().T @ _dense(ops["transverse"][0], dets) @ V
    a_m = V.conj().T @ _dense(ops["transverse"][1], dets) @ V
    retained_idx = list(range(len(es)))
    chi_ref = _lehmann_chi(w + 1j * delta, a_p, a_m, E, retained_idx, weights)
    np.testing.assert_allclose(res["operators"]["transverse"]["realaxis"], chi_ref, atol=1e-8)
