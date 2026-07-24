from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from impurityModel.ed import gf_diagnostics as gd
from impurityModel.ed import selfenergy
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant, inner
from impurityModel.ed.model import BasisOptions, ImpurityModel, Meshes, SolverOptions
from impurityModel.ed.selfenergy import UnphysicalGreensFunctionError


def _selfenergy_args(**overrides):
    """Grouped ``calc_selfenergy`` arguments for the single-orbital mocked-solver unit tests.

    The orchestration tests below mock ``calc_gs`` / ``get_Greens_function`` / ``get_sigma``,
    so the actual physics values are placeholders; only the plumbing matters. Override any
    field by name (e.g. ``verbosity=1``); returns the ``model, meshes, basis, solver`` kwargs
    (minus ``comm``, which each test passes explicitly).
    """
    v = dict(
        h0={((0, "c"), (0, "a")): 1.0},
        u4=np.zeros((1, 1, 1, 1)),
        iw=np.array([1j]),
        w=np.array([0.0]),
        delta=0.1,
        nominal_occ={0: 1},
        mixed_valence=False,
        impurity_orbitals={0: [0]},
        tau=0.1,
        verbosity=2,
        rot_to_spherical=np.eye(1),
        cluster_label="test",
        reort=None,
        dense_cutoff=100,
        spin_flip_dj=False,
        chain_restrict=False,
        occ_cutoff=1e-12,
        truncation_threshold=100,
        slaterWeightMin=1e-12,
        dN=None,
        sparse_green=False,
    )
    v.update(overrides)
    model = ImpurityModel(
        h0=v["h0"], u4=v["u4"], impurity_orbitals=v["impurity_orbitals"], rot_to_spherical=v["rot_to_spherical"]
    )
    meshes = Meshes(iw=v["iw"], w=v["w"], delta=v["delta"])
    basis = BasisOptions(
        nominal_occ=v["nominal_occ"],
        mixed_valence=v["mixed_valence"],
        dN=v["dN"],
        truncation_threshold=v["truncation_threshold"],
        chain_restrict=v["chain_restrict"],
        spin_flip_dj=v["spin_flip_dj"],
        occ_cutoff=v["occ_cutoff"],
        slater_weight_min=v["slaterWeightMin"],
        tau=v["tau"],
    )
    solver = SolverOptions(reort=v["reort"], dense_cutoff=v["dense_cutoff"], sparse_green=v["sparse_green"])
    return dict(
        model=model,
        meshes=meshes,
        basis=basis,
        solver=solver,
        verbosity=v["verbosity"],
        cluster_label=v["cluster_label"],
    )


def test_check_greens_function_valid():
    G = np.zeros((1, 2, 2), dtype=complex)
    G[0] = np.array([[-1j, 0], [0, -1j]])
    # Should not raise
    selfenergy.check_greens_function(G)


def test_check_greens_function_invalid():
    G = np.zeros((1, 2, 2), dtype=complex)
    G[0] = np.array([[1j, 0], [0, -1j]])
    with pytest.raises(UnphysicalGreensFunctionError):
        selfenergy.check_greens_function(G)


def test_raise_together_serial():
    selfenergy._raise_together(None, None)  # clean verdict: must not raise
    with pytest.raises(UnphysicalGreensFunctionError, match="boom"):
        selfenergy._raise_together(None, "boom")


@pytest.mark.mpi
def test_raise_together_is_collective():
    """An unphysical Green's function must raise on *every* rank, not just rank 0.

    ``get_Greens_function`` gathers to global rank 0, so the physicality verdict exists only
    there. Raising it on rank 0 alone left the other ranks to march into the next collective
    (``log_peak_vs_predicted``'s ``Allreduce``) and hang, turning an error into a deadlock.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("needs at least 2 ranks")

    # The verdict lives on rank 0 only, exactly as `gs_matsubara` does.
    message = "Matsubara self-energy:\nDiagonal term has positive imaginary part." if comm.rank == 0 else None
    with pytest.raises(UnphysicalGreensFunctionError, match="positive imaginary part"):
        selfenergy._raise_together(comm, message)

    # A clean verdict must leave every rank running, so the next collective completes. If any
    # rank had bailed out above, this Allreduce would hang (or the count would be short).
    selfenergy._raise_together(comm, None)
    assert comm.allreduce(1, op=MPI.SUM) == comm.size


def test_get_hcorr_v_hbath():
    # impurity_orbitals dict mapping cluster index to number of impurity orbitals
    impurity_orbitals = {0: 2}
    sum_bath_states = {0: 2}

    # h0op is a dict mapping ((i, "c"), (j, "a")) to val
    h0op = {
        ((0, "c"), (0, "a")): 1.0,
        ((1, "c"), (1, "a")): 2.0,
        ((0, "c"), (2, "a")): 0.5,
        ((2, "c"), (0, "a")): 0.5,
        ((2, "c"), (2, "a")): -1.0,
        ((3, "c"), (3, "a")): -2.0,
    }

    hcorr, v, v_dagger, h_bath = selfenergy.get_hcorr_v_hbath(h0op, impurity_orbitals, sum_bath_states)

    assert hcorr.shape == (2, 2)
    assert hcorr[0, 0] == 1.0
    assert hcorr[1, 1] == 2.0

    assert h_bath.shape == (2, 2)
    assert h_bath[0, 0] == -1.0
    assert h_bath[1, 1] == -2.0

    assert v_dagger.shape == (2, 2)
    assert v.shape == (2, 2)
    assert v[0, 0] == 0.5
    assert v_dagger[0, 0] == 0.5


def test_hyb():
    ws = np.array([0.0])
    delta = 0.1
    v = np.array([[1.0], [1.0]])
    hbath = np.array([[1.0, 0], [0, -1.0]])

    # Δ = v^dag [(ws+i*delta)I - hbath]^-1 V
    res = selfenergy.hyb(ws, v, hbath, delta)
    assert res.shape == (1, 1, 1)


def test_get_Sigma_static():
    """Hartree-Fock values for a two-orbital U/J model in RSPt's u4 convention.

    u4[i,j,k,l] = <ij|V|kl> (pairs (i,k),(j,l)): the direct element is
    u4[0,1,0,1] = U and the exchange element u4[0,1,1,0] = J, giving
    Sigma[0,0] = (U - J) n_1 and Sigma[1,1] = (U - J) n_0.
    """
    U, J = 2.3, 0.4
    U4 = np.zeros((2, 2, 2, 2))
    U4[0, 1, 0, 1] = U4[1, 0, 1, 0] = U  # direct
    U4[0, 1, 1, 0] = U4[1, 0, 0, 1] = J  # exchange
    n0, n1 = 1.0, 0.25
    rho = np.diag([n0, n1]).astype(complex)

    sigma = selfenergy.get_Sigma_static(U4, rho)
    assert sigma.shape == (2, 2)
    assert np.allclose(sigma, np.diag([(U - J) * n1, (U - J) * n0]))


def test_get_Sigma_static_consistent_with_operator():
    """For a single determinant D: <D|U_op|D> = 1/2 Tr[Sigma_static(rho_D) rho_D].

    Ties get_Sigma_static and getUop_from_rspt_u4 to the same u4 convention.
    """
    from itertools import combinations

    from impurityModel.ed.atomic_physics import getUop_from_rspt_u4

    n = 4
    rng = np.random.default_rng(1)
    r = rng.standard_normal((n, n, n, n)) + 1j * rng.standard_normal((n, n, n, n))
    r = r + r.transpose((1, 0, 3, 2))  # exchange symmetry
    U4 = r + np.conj(r.transpose((2, 3, 0, 1)))  # hermiticity
    u_op = ManyBodyOperator(getUop_from_rspt_u4(U4))

    for occupied in combinations(range(n), 2):
        data = bytearray((n + 7) // 8)
        for orb in occupied:
            data[orb // 8] |= 1 << (7 - orb % 8)
        det = ManyBodyState({SlaterDeterminant.from_bytes(bytes(data)): 1.0})
        rho = np.zeros((n, n), dtype=complex)
        for orb in occupied:
            rho[orb, orb] = 1.0
        e_op = inner(det, u_op(det, 0))
        e_hf = 0.5 * np.trace(selfenergy.get_Sigma_static(U4, rho) @ rho)
        assert np.isclose(e_op, e_hf, atol=1e-12)


def test_get_hcorr_v_hbath_reversed():
    """Anti-normal-ordered input is normal-ordered first: c_i c^dag_j = delta_ij - c^dag_j c_i.

    The Hamiltonian is always built normal-ordered in production, but ``ManyBodyOperator``
    accepts either order and canonicalizes, so the extraction must agree with the algebra
    rather than with the old hand-rolled ``1 - val`` convention (which folded the identity
    into the matrix element and was not even linear in ``val``).
    """
    impurity_orbitals = {0: 2}
    sum_bath_states = {0: 2}

    # 0.2 * c_0 c^dag_0 = 0.2 - 0.2 n_0 and 0.3 * c_1 c^dag_2 = -0.3 c^dag_2 c_1.
    h0op = {
        ((0, "a"), (0, "c")): 0.2,
        ((1, "a"), (2, "c")): 0.3,
    }

    for op in (h0op, ManyBodyOperator(h0op)):
        hcorr, v, v_dagger, _h_bath = selfenergy.get_hcorr_v_hbath(op, impurity_orbitals, sum_bath_states)
        assert hcorr[0, 0] == -0.2  # the constant +0.2 is dropped, not folded in here
        assert hcorr[1, 1] == 0.0  # not set
        # h0Matrix is 4x4 and n_corr == 2, so entry [2, 1] lands in v = h0Matrix[2:, 0:2].
        assert v[0, 1] == -0.3
        assert not np.any(v_dagger)


def test_get_hcorr_v_hbath_ignores_constant():
    """A ``()`` identity term (e.g. from ``z - hOp``) is dropped, not a crash."""
    impurity_orbitals = {0: 2}
    sum_bath_states = {0: 2}
    h0op = ManyBodyOperator({((0, "c"), (0, "a")): 1.0, ((3, "c"), (3, "a")): -2.0}) + 7.5

    assert h0op.constant == 7.5
    hcorr, _v, _v_dagger, h_bath = selfenergy.get_hcorr_v_hbath(h0op, impurity_orbitals, sum_bath_states)
    assert hcorr[0, 0] == 1.0
    assert h_bath[1, 1] == -2.0


def test_get_sigma():
    omega_mesh = np.array([0.0, 1.0])
    delta = 0.1
    impurity_orbitals = {0: 2}
    nBaths = {0: 2}
    h0op = {
        ((0, "c"), (0, "a")): 1.0,
        ((1, "c"), (1, "a")): 2.0,
        ((2, "c"), (2, "a")): -1.0,
        ((3, "c"), (3, "a")): -2.0,
        ((0, "c"), (2, "a")): 0.5,
        ((1, "c"), (3, "a")): 0.5,
    }
    blocks = [[0, 1]]

    # gs has shape (len(omega_mesh), len(block), len(block))
    # Let's provide a dummy non-interacting GS: g = g0
    # Then self-energy should be zero.

    hcorr, v_full, _, h_bath = selfenergy.get_hcorr_v_hbath(h0op, impurity_orbitals, nBaths)
    block_ix = np.ix_(blocks[0], blocks[0])
    wIs = (omega_mesh + 1j * delta)[:, np.newaxis, np.newaxis] * np.eye(2)[np.newaxis, :, :]

    hyb_func = selfenergy.hyb(omega_mesh, v_full[:, blocks[0]], h_bath, delta)
    g0_inv = wIs - hcorr[block_ix] - hyb_func
    g0 = np.linalg.inv(g0_inv)

    sigma = selfenergy.get_sigma(omega_mesh, impurity_orbitals, nBaths, [g0], h0op, delta, blocks, clustername="test")

    assert len(sigma) == 1
    assert sigma[0].shape == (2, 2, 2)
    np.testing.assert_allclose(sigma[0], np.zeros((2, 2, 2), dtype=complex), atol=1e-12)


def _identity_moments(n_corr=1, max_order=3):
    """A stand-in for get_greens_function_moments in the mocked orchestration tests."""
    M = np.zeros((max_order + 1, n_corr, n_corr), dtype=complex)
    M[0] = np.eye(n_corr)
    return M


@patch("impurityModel.ed.selfenergy.get_greens_function_moments")
@patch("impurityModel.ed.selfenergy.calc_gs")
@patch("impurityModel.ed.selfenergy.get_Greens_function")
def test_calc_selfenergy(mock_get_gf, mock_calc_gs, mock_moments):
    # Setup mocks
    mock_calc_gs.return_value = (
        [np.array([1.0])],  # psis
        [0.0],  # es
        MagicMock(restrictions=None, impurity_orbitals={0: [[0]]}),  # ground_state_basis
        np.array([[1.0]]),  # thermal_rho
        {"rhos": [np.array([[1.0]])]},  # gs_info
    )

    mock_get_gf.return_value = (
        [np.array([[[-1j]]])],  # gs_matsubara: one (n_omega, len(block), len(block)) per inequivalent block
        [np.array([[[-1j]]])],  # gs_realaxis
        None,
    )
    mock_moments.return_value = _identity_moments()

    res = selfenergy.calc_selfenergy(**_selfenergy_args(), comm=None)

    assert res["sigma"] is not None
    assert res["sigma_real"] is not None
    assert res["sigma_moment_1"] is not None
    assert res["sigma_moment_2"] is not None
    assert res["sigma_moment_1"].shape == res["sigma_static"].shape
    assert mock_calc_gs.called
    assert mock_get_gf.called
    assert mock_moments.called


@patch("impurityModel.ed.selfenergy.get_greens_function_moments")
@patch("impurityModel.ed.selfenergy.calc_gs")
@patch("impurityModel.ed.selfenergy.get_Greens_function")
def test_calc_selfenergy_retries_on_truncated_ensemble(mock_get_gf, mock_calc_gs, mock_moments):
    """A diagnostics report flagging a truncated thermal ensemble triggers exactly one
    auto-retry of calc_gs with a larger num_wanted; a clean report stops the loop."""
    mock_calc_gs.return_value = (
        [np.array([1.0])],
        [0.0],
        MagicMock(restrictions=None, impurity_orbitals={0: [[0]]}),
        np.array([[1.0]]),
        {"rhos": [np.array([[1.0]])]},
    )

    truncated = gd.DiagnosticReport()
    truncated.add("[0]", gd.check_thermal_weight_cutoff([0.0, 0.05], 0.0, 0.1, n_returned=10, num_wanted=10))
    assert truncated.needs_more_states  # sanity: first report does ask for more states
    clean = gd.DiagnosticReport()
    gf = ([np.array([[[-1j]]])], [np.array([[[-1j]]])], None)
    # First call returns the truncated report, second the clean one.
    mock_get_gf.side_effect = [gf[:-1] + (truncated,), gf[:-1] + (clean,)]
    mock_moments.return_value = _identity_moments()

    selfenergy.calc_selfenergy(**_selfenergy_args(verbosity=1), comm=None)

    # Exactly one retry: calc_gs called twice, the second time with a larger num_wanted.
    assert mock_calc_gs.call_count == 2
    assert mock_get_gf.call_count == 2
    first_nw = mock_calc_gs.call_args_list[0].kwargs["num_wanted"]
    second_nw = mock_calc_gs.call_args_list[1].kwargs["num_wanted"]
    assert second_nw > first_nw


# fixed_peak_dc and fixed_occupation_dc are covered end-to-end in test_fixed_dc.py.


@patch("impurityModel.ed.selfenergy.get_greens_function_moments")
@patch("impurityModel.ed.selfenergy.calc_gs")
@patch("impurityModel.ed.selfenergy.get_Greens_function")
def test_calc_selfenergy_no_matsubara(mock_get_gf, mock_calc_gs, mock_moments):
    mock_calc_gs.return_value = (
        [np.array([1.0])],
        [0.0],
        MagicMock(restrictions=None, impurity_orbitals={0: [[0]]}),
        np.array([[1.0]]),
        {"rhos": [np.array([[1.0]])]},
    )
    mock_get_gf.return_value = (None, [np.array([[[-1j]]])], None)  # gs_matsubara=None, gs_realaxis=one block
    mock_moments.return_value = _identity_moments()
    res = selfenergy.calc_selfenergy(**_selfenergy_args(), comm=None)
    assert res["sigma"] is None
    assert res["sigma_real"] is not None
    assert res["sigma_moment_1"] is not None
    assert res["sigma_moment_2"] is not None


@patch("impurityModel.ed.selfenergy.calc_gs")
@patch("impurityModel.ed.selfenergy.get_Greens_function")
def test_calc_selfenergy_exceptions(mock_get_gf, mock_calc_gs):
    # Test Unphysical greens function in matsubara
    mock_calc_gs.return_value = (
        [np.array([1.0])],
        [0.0],
        MagicMock(restrictions={((0,),): (0, 1)}, impurity_orbitals={0: [[0]]}),
        np.array([[1.0]]),
        {"rhos": [np.array([[1.0]])]},
    )
    mock_get_gf.return_value = ([np.array([[[1j]]])], None, None)  # Diagonal has positive imag part -> unphysical

    with pytest.raises(UnphysicalGreensFunctionError):
        selfenergy.calc_selfenergy(**_selfenergy_args(), comm=None)

    # Test Unphysical greens function in realaxis
    mock_get_gf.return_value = (None, [np.array([[[1j]]])], None)
    with pytest.raises(UnphysicalGreensFunctionError):
        selfenergy.calc_selfenergy(**_selfenergy_args(), comm=None)


@patch("impurityModel.ed.selfenergy.save_Greens_function")
@patch("impurityModel.ed.selfenergy.calc_gs")
@patch("impurityModel.ed.selfenergy.get_Greens_function")
@patch("impurityModel.ed.selfenergy.get_sigma")
def test_calc_selfenergy_sigma_exceptions(mock_get_sigma, mock_get_gf, mock_calc_gs, mock_save_gf):
    mock_calc_gs.return_value = (
        [np.array([1.0])],
        [0.0],
        MagicMock(restrictions=None, impurity_orbitals={0: [[0]]}),
        np.array([[1.0]]),
        {"rhos": [np.array([[1.0]])]},
    )

    # Test unphysical sigma_real
    mock_get_gf.return_value = (None, [np.array([[[-1j]]])], None)  # Valid GS
    mock_get_sigma.return_value = [np.array([[[1j]]])]  # Invalid sigma

    with pytest.raises(UnphysicalGreensFunctionError):
        selfenergy.calc_selfenergy(**_selfenergy_args(), comm=None)

    # Test unphysical sigma matsubara
    mock_get_gf.return_value = ([np.array([[[-1j]]])], None, None)  # Valid GS
    mock_get_sigma.return_value = [np.array([[[1j]]])]  # Invalid sigma

    with pytest.raises(UnphysicalGreensFunctionError):
        selfenergy.calc_selfenergy(**_selfenergy_args(), comm=None)


# --- Self-energy moments (Sigma_1, Sigma_2) --------------------------------------------------


def test_get_Sigma_moments_formulas():
    """get_Sigma_moments reproduces the three high-frequency coefficients and is Hermitian.

    Uses hand-built Green's-function moments and hybridization blocks; checks the closed forms
    Sigma_inf = M1 - hcorr, Sigma_1 = M2 - M1^2 - V^dag V,
    Sigma_2 = M3 - M1 M2 - M2 M1 + M1^3 - V^dag hbath V.
    """
    rng = np.random.default_rng(0)

    def herm(n):
        a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        return a + a.conj().T

    n_corr, n_bath = 2, 3
    M = np.stack([np.eye(n_corr).astype(complex), herm(n_corr), herm(n_corr), herm(n_corr)])
    hcorr = herm(n_corr)
    hbath = herm(n_bath)
    v = rng.standard_normal((n_bath, n_corr)) + 1j * rng.standard_normal((n_bath, n_corr))

    sigma_inf, sigma_1, sigma_2 = selfenergy.get_Sigma_moments(M, hcorr, v, hbath)

    m1, m2, m3 = M[1], M[2], M[3]
    np.testing.assert_allclose(sigma_inf, m1 - hcorr, atol=1e-12)
    np.testing.assert_allclose(sigma_1, m2 - m1 @ m1 - v.conj().T @ v, atol=1e-12)
    np.testing.assert_allclose(sigma_2, m3 - m1 @ m2 - m2 @ m1 + m1 @ m1 @ m1 - v.conj().T @ hbath @ v, atol=1e-12)
    # Sigma_1, Sigma_2 are Hermitian for Hermitian inputs.
    np.testing.assert_allclose(sigma_1, sigma_1.conj().T, atol=1e-12)
    np.testing.assert_allclose(sigma_2, sigma_2.conj().T, atol=1e-12)


def _serial_basis(impurity_orbitals, bath_states, initial_basis):
    from mpi4py import MPI

    from impurityModel.ed.manybody_basis import Basis

    return Basis(
        impurity_orbitals=impurity_orbitals,
        bath_states=bath_states,
        initial_basis=initial_basis,
        comm=MPI.COMM_SELF,
    )


def test_greens_function_moments_noninteracting():
    """For a non-interacting impurity+bath the exact GF moments are M_n = (h^n) restricted to
    the impurity, and both dynamic self-energy moments vanish.

    One impurity spin-orbital (0) hybridising with one bath spin-orbital (1); the many-body
    ground state is the vacuum (a valid Slater determinant), for which the impurity GF is the
    single-particle resolvent [(z - h)^-1]_00 independent of filling.
    """
    from impurityModel.ed.greens_function import get_greens_function_moments

    eps_d, eps_b, vhop = -0.4, 1.1, 0.7
    hOp = ManyBodyOperator(
        {
            ((0, "c"), (0, "a")): eps_d,
            ((1, "c"), (1, "a")): eps_b,
            ((0, "c"), (1, "a")): vhop,
            ((1, "c"), (0, "a")): vhop,
        }
    )
    h = np.array([[eps_d, vhop], [vhop, eps_b]], dtype=complex)

    basis = _serial_basis({0: [[0]]}, ({0: [[1]]}, {0: [[]]}), [b"\x00"])
    vac = ManyBodyState({SlaterDeterminant.from_bytes(b"\x00"): 1.0})

    M = get_greens_function_moments([vac], [0.0], tau=1.0, basis=basis, hOp=hOp, impurity_indices=[0])

    for n in range(4):
        np.testing.assert_allclose(M[n], np.linalg.matrix_power(h, n)[0:1, 0:1], atol=1e-10)

    sigma_inf, sigma_1, sigma_2 = selfenergy.get_Sigma_moments(
        M, np.array([[eps_d]], dtype=complex), np.array([[vhop]], dtype=complex), np.array([[eps_b]], dtype=complex)
    )
    np.testing.assert_allclose(sigma_inf, 0.0, atol=1e-10)
    np.testing.assert_allclose(sigma_1, 0.0, atol=1e-10)
    np.testing.assert_allclose(sigma_2, 0.0, atol=1e-10)


def test_greens_function_moments_hubbard_atom():
    """Half-filled single-orbital Hubbard atom: the thermally averaged impurity GF is a
    two-pole function with poles {eps, eps+U} and equal weights, giving the atomic self-energy
    Sigma_inf = U/2, Sigma_1 = U^2/4, Sigma_2 = (U^2/4)(eps + U/2)."""
    from impurityModel.ed.atomic_physics import getUop_from_rspt_u4
    from impurityModel.ed.greens_function import get_greens_function_moments

    eps, U = -0.7, 3.0
    U4 = np.zeros((2, 2, 2, 2))
    U4[0, 1, 0, 1] = U4[1, 0, 1, 0] = U  # density-density U n_up n_dn
    hOp = ManyBodyOperator({((0, "c"), (0, "a")): eps, ((1, "c"), (1, "a")): eps}) + ManyBodyOperator(
        getUop_from_rspt_u4(U4)
    )

    # The degenerate half-filled ground manifold: |up> (orbital 0) and |dn> (orbital 1), E = eps.
    basis = _serial_basis({0: [[0, 1]]}, ({0: [[]]}, {0: [[]]}), [b"\x80", b"\x40"])
    psi_up = ManyBodyState({SlaterDeterminant.from_bytes(b"\x80"): 1.0})
    psi_dn = ManyBodyState({SlaterDeterminant.from_bytes(b"\x40"): 1.0})

    M = get_greens_function_moments(
        [psi_up, psi_dn], [eps, eps], tau=1.0, basis=basis, hOp=hOp, impurity_indices=[0, 1]
    )

    # Reference: two-pole G per spin, poles {eps, eps+U}, weights {1/2, 1/2}; no spin mixing.
    poles = np.array([eps, eps + U])
    weights = np.array([0.5, 0.5])
    for n in range(4):
        m_n = np.sum(weights * poles**n)
        np.testing.assert_allclose(M[n], m_n * np.eye(2), atol=1e-10)

    hcorr = eps * np.eye(2, dtype=complex)
    empty_v = np.zeros((0, 2), dtype=complex)
    empty_hbath = np.zeros((0, 0), dtype=complex)
    sigma_inf, sigma_1, sigma_2 = selfenergy.get_Sigma_moments(M, hcorr, empty_v, empty_hbath)
    np.testing.assert_allclose(sigma_inf, (U / 2) * np.eye(2), atol=1e-10)
    np.testing.assert_allclose(sigma_1, (U**2 / 4) * np.eye(2), atol=1e-10)
    np.testing.assert_allclose(sigma_2, (U**2 / 4) * (eps + U / 2) * np.eye(2), atol=1e-10)
