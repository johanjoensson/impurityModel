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
        psis = [ManyBodyState(width=1) for _ in keep]
    # Each seed goes through its own explicit width-1 block rather than a bare
    # ManyBodyState({}) placeholder on the non-owning rank: once the flat and block
    # classes merge (Phase 7 step 3), a bare placeholder is the width-0 polymorphic
    # zero, an asymmetric mismatch against the owning rank's populated (eventually
    # width-1) seeds that would deadlock redistribute_psis' collective.
    psi_blocks = [ManyBodyState.from_states([psi]) for psi in psis]
    psis = [blk.to_states()[0] for blk in basis.redistribute_psis(*psi_blocks)]

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

    ops, skipped, _notes = susceptibility.build_susceptibility_operators(hOp, basis, np.eye(2, dtype=complex))
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

    ops, skipped, _notes = susceptibility.build_susceptibility_operators(hOp, basis, np.eye(2, dtype=complex))
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


def test_the_workflow_solves_saves_and_keeps_su2(tmp_path):
    """calc_susceptibility_workflow -- what the API, the susceptibility CLI and `run` all call --
    had never run. It is glue (solver basis -> calc_gs -> calc_susceptibility -> save), so this
    checks the glue: the meshes, tau and delta reach the result, the file it writes round-trips,
    and a spin-symmetric model keeps chi_+- = 2 chi_zz through the whole pipeline. Genuinely
    distributed at -n 2/-n 3; the result and the file are rank 0's."""
    h5py = pytest.importorskip("h5py")
    from impurityModel.ed.model import Meshes

    from .test_selfenergy_end_to_end import _basis_and_solver, _siam_model

    basis, solver = _basis_and_solver()
    w = np.linspace(-3.0, 3.0, 13)
    meshes = Meshes(iw=None, w=w, delta=0.1)
    output = tmp_path / "chi.h5"
    comm = MPI.COMM_WORLD
    result = susceptibility.calc_susceptibility_workflow(
        _siam_model(), meshes, basis, solver, comm=comm, n_matsubara=4, output_filename=str(output)
    )
    if comm.rank != 0:
        return
    np.testing.assert_allclose(result["w"], w)
    np.testing.assert_allclose(result["matsubara"], susceptibility.bosonic_matsubara_mesh(basis.tau, 4))
    assert result["tau"] == basis.tau and result["delta"] == 0.1

    zz, pm = result["operators"]["spin_z"], result["operators"]["transverse"]
    assert np.abs(zz["realaxis"]).max() > 0
    np.testing.assert_allclose(pm["realaxis"], 2.0 * zz["realaxis"], atol=1e-8)
    np.testing.assert_allclose(pm["matsubara"], 2.0 * zz["matsubara"], atol=1e-8)

    with h5py.File(output, "r") as h5f:
        np.testing.assert_allclose(h5f["w"][()], w)
        np.testing.assert_allclose(h5f["chi/spin_z/realaxis"][()], zz["realaxis"])
        assert h5f.attrs["tau"] == pytest.approx(basis.tau)


def test_chi_solves_the_ground_state_under_the_basis_options_like_sigma(monkeypatch):
    """P3 A/B: chi's ground state is decided by the same ``BasisOptions`` values as the self-energy's (it hard-coded
    chain_restrict=False and ignored the budgets and tolerances). The recorded baseline is the pre-unification Curie
    coefficient of this fixture, where no truncation binds, so the two must agree to 1e-8; and the spec the driver
    hands ``calc_gs`` carries the options."""
    import dataclasses

    from impurityModel.ed import gs_calc
    from impurityModel.ed.model import Meshes

    from .test_selfenergy_end_to_end import _basis_and_solver, _siam_model

    basis, solver = _basis_and_solver()
    basis = dataclasses.replace(basis, chain_restrict=True, e_pt2_tol=1e-7, de2_min=1e-10)
    seen = {}
    real_calc_gs = gs_calc.calc_gs

    def spy(*args, **kwargs):
        seen["spec"] = kwargs["spec"]
        return real_calc_gs(*args, **kwargs)

    monkeypatch.setattr(gs_calc, "calc_gs", spy)
    meshes = Meshes(iw=None, w=np.linspace(-3.0, 3.0, 13), delta=0.1)
    result = susceptibility.calc_susceptibility_workflow(
        _siam_model(), meshes, basis, solver, comm=MPI.COMM_WORLD, n_matsubara=0
    )
    spec = seen["spec"]
    assert spec.chain_restrict is True and spec.e_pt2_tol == 1e-7 and spec.de2_min == 1e-10
    assert spec.cost_budget == basis.cost_budget
    if MPI.COMM_WORLD.rank == 0:
        assert result["operators"]["spin_z"]["curie_coefficient"] == pytest.approx(0.2412685545777933, abs=1e-8)


def _fake_result(expectation, elastic2, realaxis):
    """A calc_susceptibility-shaped result for one member (one operator, no Matsubara)."""
    return {
        "w": np.zeros(2),
        "operators": {
            "charge": {
                "realaxis": np.asarray(realaxis, dtype=complex),
                "matsubara": None,
                "curie_coefficient": float(elastic2 - abs(expectation) ** 2),
                "expectation": complex(expectation),
                "elastic2": float(elastic2),
            }
        },
    }


def test_the_sector_merge_forms_the_curie_coefficient_after_the_weighted_sum():
    """chi = sum_s c_s chi_s is linear, the Curie coefficient <A^2> - |<A>|^2 is not: the between-sector variance of
    <A> must survive.  Two sectors with a sharp charge (2 and 3, no fluctuation inside either) have Curie 0 each but
    c(1-c) in the ensemble."""
    a = _fake_result(2.0, 4.0, [1.0, 2.0])
    b = _fake_result(3.0, 9.0, [3.0, 6.0])
    merged = susceptibility.merge_susceptibility_results([a, b], [0.25, 0.75])
    entry = merged["operators"]["charge"]
    assert entry["expectation"] == pytest.approx(2.75)
    assert entry["curie_coefficient"] == pytest.approx(0.25 * 0.75)  # the variance of N over the two sectors
    np.testing.assert_allclose(entry["realaxis"], [0.25 * 1.0 + 0.75 * 3.0, 0.25 * 2.0 + 0.75 * 6.0])
    # the weights are renormalised, and a one-member merge is the member itself
    assert susceptibility.merge_susceptibility_results([a, b], [1.0, 3.0])["operators"]["charge"][
        "curie_coefficient"
    ] == pytest.approx(0.25 * 0.75)
    one = susceptibility.merge_susceptibility_results([a], [0.4])["operators"]["charge"]
    assert one["curie_coefficient"] == a["operators"]["charge"]["curie_coefficient"]
    np.testing.assert_array_equal(one["realaxis"], a["operators"]["charge"]["realaxis"])
    assert susceptibility.merge_susceptibility_results([None, None], [0.5, 0.5]) is None


def _workflow(mixed_valence, tau, n_matsubara=2):
    import dataclasses

    from impurityModel.ed.model import Meshes

    from .test_selfenergy_end_to_end import _basis_and_solver, _siam_model

    basis, solver = _basis_and_solver()
    basis = dataclasses.replace(basis, tau=tau, mixed_valence=mixed_valence)
    meshes = Meshes(iw=None, w=np.linspace(-3.0, 3.0, 9), delta=0.1)
    return susceptibility.calc_susceptibility_workflow(
        _siam_model(), meshes, basis, solver, comm=MPI.COMM_WORLD, n_matsubara=n_matsubara
    )


def test_one_member_auto_chi_is_the_off_chi():
    """At low temperature the ensemble has one member and takes the single-sector path: bit-identical."""
    off, auto = _workflow("off", 0.05), _workflow("auto", 0.05)
    if MPI.COMM_WORLD.rank != 0:
        return
    for name, entry in off["operators"].items():
        for key in ("realaxis", "matsubara", "curie_coefficient", "expectation"):
            np.testing.assert_array_equal(auto["operators"][name][key], entry[key])


def test_the_chi_ensemble_merges_the_members_with_their_weights(monkeypatch):
    """Hot enough for several charge sectors: the driver hands the members' own results and the ensemble weights to
    the merge, and the merged charge response carries the between-sector variance of N."""
    parts_seen = {}
    real = susceptibility.merge_susceptibility_results

    def spy(parts, weights):
        parts_seen["parts"], parts_seen["weights"] = parts, list(weights)
        return real(parts, weights)

    monkeypatch.setattr(susceptibility, "merge_susceptibility_results", spy)
    result = _workflow("on", 1.0)
    if MPI.COMM_WORLD.rank != 0:
        return
    parts, c = parts_seen["parts"], np.asarray(parts_seen["weights"])
    assert len(parts) >= 2 and c.sum() == pytest.approx(1.0, abs=1e-12)
    for name, entry in result["operators"].items():
        n_mean = sum(ci * p["operators"][name]["expectation"] for ci, p in zip(c, parts))
        n2 = sum(ci * p["operators"][name]["elastic2"] for ci, p in zip(c, parts))
        assert entry["expectation"] == pytest.approx(n_mean)
        assert entry["curie_coefficient"] == pytest.approx(n2 - abs(n_mean) ** 2)
        np.testing.assert_allclose(
            entry["realaxis"], sum(ci * p["operators"][name]["realaxis"] for ci, p in zip(c, parts))
        )
    charge = result["operators"]["charge"]
    within = sum(ci * p["operators"]["charge"]["curie_coefficient"] for ci, p in zip(c, parts))
    assert charge["curie_coefficient"] > within + 1e-3  # the sectors have different <N>


def test_the_chi_ensemble_charge_response_is_the_exact_grand_canonical_one_and_the_winner_alone_is_not():
    """Dense ED over every electron count of the minimal SIAM (the model of ``test_selfenergy_ensemble``): at
    tau = 0.2 the ensemble <N_imp> and the elastic (inside the retained thermal states) part of the charge Curie
    coefficient are the Boltzmann averages; the winner sector alone gets <N_imp> off by 1e-2 and a Curie coefficient
    of zero. Weights that were wrong (uniform, or unshifted by the sector energies) land far outside 1e-4."""
    from .test_selfenergy_ensemble import _dense_siam

    tau = 0.2
    h, cd = _dense_siam()
    e, v = np.linalg.eigh(h)
    nn = v.T @ (cd[0] @ cd[0].T + cd[1] @ cd[1].T) @ v
    p = np.exp(-(e - e.min()) / tau)
    p /= p.sum()
    keep = p > 1e-4  # the thermal states the ensemble retains (the dropped tail carries < 1e-3 of the weight)
    pk = p * keep / (p * keep).sum()
    mean = pk @ np.diag(nn)
    elastic = sum(pk[n] * sum(nn[m, n] ** 2 for m in range(len(e)) if keep[m]) for n in range(len(e)))
    on, off = _workflow("on", tau), _workflow("off", tau)
    if MPI.COMM_WORLD.rank != 0:
        return
    got, single = on["operators"]["charge"], off["operators"]["charge"]
    assert got["expectation"].real == pytest.approx(mean, abs=1e-4)
    assert got["curie_coefficient"] == pytest.approx(elastic - mean**2, abs=1e-4)
    assert abs(single["expectation"].real - mean) > 5e-3  # mutation: the winner alone is not the ensemble
    assert abs(single["curie_coefficient"] - (elastic - mean**2)) > 5e-3


def test_a_failing_rank0_chi_merge_is_raised_on_every_rank(monkeypatch):
    """A failure of the rank-0 ensemble merge is raised on every rank, not left as a hang in the next collective."""

    def boom(parts, weights):
        if MPI.COMM_WORLD.rank == 0:
            raise ValueError("injected merge failure")

    monkeypatch.setattr(susceptibility, "merge_susceptibility_results", boom)
    with pytest.raises((ValueError, RuntimeError)):
        _workflow("on", 1.0)
    assert MPI.COMM_WORLD.allreduce(1, op=MPI.SUM) == MPI.COMM_WORLD.size
