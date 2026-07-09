import numpy as np
import pytest
from impurityModel.ed import gf_diagnostics as gd
from impurityModel.ed import selfenergy
from impurityModel.ed.selfenergy import UnphysicalGreensFunctionError


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
    from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant, inner

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
    impurity_orbitals = {0: 2}
    sum_bath_states = {0: 2}

    # Test opj == "c" and opi == "a" (lines 510-514)
    h0op = {
        ((0, "a"), (0, "c")): 0.2,  # i==j case: h0Matrix[0, 0] = 1 - 0.2 = 0.8
        ((1, "a"), (2, "c")): 0.3,  # i!=j case: h0Matrix[1, 2] = -0.3
    }

    hcorr, v, v_dagger, h_bath = selfenergy.get_hcorr_v_hbath(h0op, impurity_orbitals, sum_bath_states)

    assert hcorr[0, 0] == 0.8
    assert hcorr[1, 1] == 0.0  # not set
    assert (
        v_dagger[1, 0] == -0.3
    )  # n_corr=2, so j=2 corresponds to col 0 of v_dagger. wait, h0Matrix[1, 2] is hcorr vs v_dagger?
    # let's be careful. h0Matrix is 4x4.
    # [1, 2] is row 1, col 2. n_corr=2. so it's in v_dagger.
    # v_dagger = h0Matrix[0:2, 2:4].
    # v_dagger[1, 0] = h0Matrix[1, 2] = -0.3.
    assert v_dagger[1, 0] == -0.3


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


from unittest.mock import patch, MagicMock


@patch("impurityModel.ed.selfenergy.calc_gs")
@patch("impurityModel.ed.selfenergy.get_Greens_function")
def test_calc_selfenergy(mock_get_gf, mock_calc_gs):
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

    h0 = {((0, "c"), (0, "a")): 1.0}
    u4 = np.zeros((1, 1, 1, 1))
    iw = np.array([1j])
    w = np.array([0.0])
    delta = 0.1
    nominal_occ = {0: 1}
    mixed_valence = False
    impurity_orbitals = {0: [0]}
    tau = 0.1
    verbosity = 2
    rot_to_spherical = np.eye(1)

    res = selfenergy.calc_selfenergy(
        h0=h0,
        u4=u4,
        iw=iw,
        w=w,
        delta=delta,
        nominal_occ=nominal_occ,
        mixed_valence=mixed_valence,
        impurity_orbitals=impurity_orbitals,
        tau=tau,
        verbosity=verbosity,
        rot_to_spherical=rot_to_spherical,
        cluster_label="test",
        reort=None,
        dense_cutoff=100,
        spin_flip_dj=False,
        comm=None,
        chain_restrict=False,
        occ_cutoff=1e-12,
        truncation_threshold=100,
        slaterWeightMin=1e-12,
        dN=None,
        sparse_green=False,
    )

    assert res["sigma"] is not None
    assert res["sigma_real"] is not None
    assert mock_calc_gs.called
    assert mock_get_gf.called


@patch("impurityModel.ed.selfenergy.calc_gs")
@patch("impurityModel.ed.selfenergy.get_Greens_function")
def test_calc_selfenergy_retries_on_truncated_ensemble(mock_get_gf, mock_calc_gs):
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

    selfenergy.calc_selfenergy(
        h0={((0, "c"), (0, "a")): 1.0},
        u4=np.zeros((1, 1, 1, 1)),
        iw=np.array([1j]),
        w=np.array([0.0]),
        delta=0.1,
        nominal_occ={0: 1},
        mixed_valence=False,
        impurity_orbitals={0: [0]},
        tau=0.1,
        verbosity=1,
        rot_to_spherical=np.eye(1),
        cluster_label="test",
        reort=None,
        dense_cutoff=100,
        spin_flip_dj=False,
        comm=None,
        chain_restrict=False,
        occ_cutoff=1e-12,
        truncation_threshold=100,
        slaterWeightMin=1e-12,
        dN=None,
        sparse_green=False,
    )

    # Exactly one retry: calc_gs called twice, the second time with a larger num_wanted.
    assert mock_calc_gs.call_count == 2
    assert mock_get_gf.call_count == 2
    first_nw = mock_calc_gs.call_args_list[0].kwargs["num_wanted"]
    second_nw = mock_calc_gs.call_args_list[1].kwargs["num_wanted"]
    assert second_nw > first_nw


# fixed_peak_dc and fixed_occupation_dc are covered end-to-end in test_fixed_dc.py.


@patch("impurityModel.ed.selfenergy.calc_gs")
@patch("impurityModel.ed.selfenergy.get_Greens_function")
def test_calc_selfenergy_no_matsubara(mock_get_gf, mock_calc_gs):
    mock_calc_gs.return_value = (
        [np.array([1.0])],
        [0.0],
        MagicMock(restrictions=None, impurity_orbitals={0: [[0]]}),
        np.array([[1.0]]),
        {"rhos": [np.array([[1.0]])]},
    )
    mock_get_gf.return_value = (None, [np.array([[[-1j]]])], None)  # gs_matsubara=None, gs_realaxis=one block
    res = selfenergy.calc_selfenergy(
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
        comm=None,
        chain_restrict=False,
        occ_cutoff=1e-12,
        truncation_threshold=100,
        slaterWeightMin=1e-12,
        dN=None,
        sparse_green=False,
    )
    assert res["sigma"] is None
    assert res["sigma_real"] is not None


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
        selfenergy.calc_selfenergy(
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
            comm=None,
            chain_restrict=False,
            occ_cutoff=1e-12,
            truncation_threshold=100,
            slaterWeightMin=1e-12,
            dN=None,
            sparse_green=False,
        )

    # Test Unphysical greens function in realaxis
    mock_get_gf.return_value = (None, [np.array([[[1j]]])], None)
    with pytest.raises(UnphysicalGreensFunctionError):
        selfenergy.calc_selfenergy(
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
            comm=None,
            chain_restrict=False,
            occ_cutoff=1e-12,
            truncation_threshold=100,
            slaterWeightMin=1e-12,
            dN=None,
            sparse_green=False,
        )


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
        selfenergy.calc_selfenergy(
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
            comm=None,
            chain_restrict=False,
            occ_cutoff=1e-12,
            truncation_threshold=100,
            slaterWeightMin=1e-12,
            dN=None,
            sparse_green=False,
        )

    # Test unphysical sigma matsubara
    mock_get_gf.return_value = ([np.array([[[-1j]]])], None, None)  # Valid GS
    mock_get_sigma.return_value = [np.array([[[1j]]])]  # Invalid sigma

    with pytest.raises(UnphysicalGreensFunctionError):
        selfenergy.calc_selfenergy(
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
            comm=None,
            chain_restrict=False,
            occ_cutoff=1e-12,
            truncation_threshold=100,
            slaterWeightMin=1e-12,
            dN=None,
            sparse_green=False,
        )


@patch("impurityModel.ed.selfenergy.calc_selfenergy")
@patch("impurityModel.ed.selfenergy.get_noninteracting_hamiltonian_operator")
@patch("impurityModel.ed.selfenergy.atomic_physics.getUop")
def test_get_selfenergy(mock_getUop, mock_get_h0, mock_calc):
    from mpi4py import MPI

    if MPI.COMM_WORLD.rank != 0:
        return
    mock_getUop.return_value = {(((0, 0, 0), "c"), ((0, 0, 0), "a"), ((0, 0, 0), "c"), ((0, 0, 0), "a")): 1.0}
    mock_get_h0.return_value = {(((0, 0, 0), "c"), ((0, 0, 0), "a")): 1.0}
    mock_calc.return_value = {"sigma": None, "sigma_real": None, "sigma_static": None}

    result = selfenergy.get_selfenergy(
        clustername="test",
        h0_filename="dummy.h0",
        ls=0,
        nBaths=2,
        nValBaths=1,
        n0imps=1,
        dnTols=1,
        dnValBaths=1,
        dnConBaths=1,
        Fdd=[0.0],
        xi=0.0,
        chargeTransferCorrection=0.0,
        hField=(0.0, 0.0, 0.0),
        nPsiMax=1,
        nPrintSlaterWeights=1,
        tau=0.1,
        energy_cut=1.0,
        delta=0.1,
        verbose=True,
    )
    # get_selfenergy forwards calc_selfenergy's result dict (no bogus tuple-unpacking).
    assert result == {"sigma": None, "sigma_real": None, "sigma_static": None}
    assert mock_calc.called


@patch("impurityModel.ed.selfenergy.calc_selfenergy")
@patch("impurityModel.ed.selfenergy.get_noninteracting_hamiltonian_operator")
@patch("impurityModel.ed.selfenergy.atomic_physics.getUop")
def test_get_selfenergy_exception(mock_getUop, mock_get_h0, mock_calc):
    from mpi4py import MPI

    if MPI.COMM_WORLD.rank != 0:
        return
    # Pass an invalid spinOrb to trigger Exception in c2i
    mock_getUop.return_value = {
        (((99, 99, 99), "c"), ((99, 99, 99), "a"), ((99, 99, 99), "c"), ((99, 99, 99), "a")): 1.0
    }
    mock_get_h0.return_value = {(((99, 99, 99), "c"), ((99, 99, 99), "a")): 1.0}
    mock_calc.return_value = (None, None, None)

    with pytest.raises(Exception):
        selfenergy.get_selfenergy(
            clustername="test",
            h0_filename="dummy.h0",
            ls=0,
            nBaths=2,
            nValBaths=1,
            n0imps=1,
            dnTols=1,
            dnValBaths=1,
            dnConBaths=1,
            Fdd=[0.0],
            xi=0.0,
            chargeTransferCorrection=0.0,
            hField=(0.0, 0.0, 0.0),
            nPsiMax=1,
            nPrintSlaterWeights=1,
            tau=0.1,
            energy_cut=1.0,
            delta=0.1,
            verbose=True,
        )
