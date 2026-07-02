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
    U4 = np.zeros((2, 2, 2, 2))
    U4[0, 1, 0, 1] = 1.0
    rho = np.array([[1.0, 0], [0, 1.0]])

    sigma = selfenergy.get_Sigma_static(U4, rho)
    assert sigma.shape == (2, 2)


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


@patch("impurityModel.ed.selfenergy.finite.thermal_average_scale_indep")
@patch("impurityModel.ed.selfenergy.CIPSISolver")
@patch("impurityModel.ed.selfenergy.Basis")
@patch("impurityModel.ed.selfenergy.finite.getUop_from_rspt_u4")
def test_fixed_peak_dc(mock_getUop, mock_Basis, mock_CIPSISolver, mock_thermal_avg):
    mock_getUop.return_value = {}

    mock_solver_instance = MagicMock()
    mock_CIPSISolver.return_value = mock_solver_instance
    mock_solver_instance.get_eigenvectors.return_value = ([0.0], [np.array([1.0])])

    mock_basis_instance = MagicMock()
    mock_Basis.return_value = mock_basis_instance
    mock_basis_instance.tau = 0.1
    mock_basis_instance.build_density_matrices.return_value = np.array([[1.0]])

    mock_thermal_avg.side_effect = [np.array([[1.0]]), np.array([[2.0]])] * 10

    dc_guess = np.array([[1.0]])
    u4 = np.zeros((1, 1, 1, 1))

    dc = selfenergy.fixed_peak_dc(
        h0_op={},
        N0={0: 1},
        mixed_valence=False,
        impurity_orbitals={0: [[0]]},
        bath_states=({0: []}, {0: []}),
        u4=u4,
        peak_position=0.1,
        dc_guess=dc_guess,
        spin_flip_dj=False,
        tau=0.1,
        rank=0,
        verbose=False,
        dense_cutoff=100,
        slaterWeightMin=1e-12,
        truncation_threshold=100,
    )

    assert dc is not None
    assert dc.shape == (1, 1)


@patch("impurityModel.ed.selfenergy.finite.thermal_average_scale_indep")
@patch("impurityModel.ed.selfenergy.CIPSISolver")
@patch("impurityModel.ed.selfenergy.Basis")
@patch("impurityModel.ed.selfenergy.finite.getUop_from_rspt_u4")
def test_fixed_peak_dc_negative(mock_getUop, mock_Basis, mock_CIPSISolver, mock_thermal_avg):
    mock_getUop.return_value = {}
    mock_solver_instance = MagicMock()
    mock_CIPSISolver.return_value = mock_solver_instance
    mock_solver_instance.get_eigenvectors.return_value = ([0.0], [np.array([1.0])])
    mock_basis_instance = MagicMock()
    mock_Basis.return_value = mock_basis_instance
    mock_basis_instance.tau = 0.0
    mock_basis_instance.build_density_matrices.return_value = np.array([[1.0]])
    mock_thermal_avg.side_effect = [np.array([[1.0]]), np.array([[2.0]])] * 10

    dc = selfenergy.fixed_peak_dc(
        h0_op={},
        N0={0: 1},
        mixed_valence=False,
        impurity_orbitals={0: [[0]]},
        bath_states=({0: []}, {0: []}),
        u4=np.zeros((1, 1, 1, 1)),
        peak_position=-0.1,
        dc_guess=np.array([[1.0]]),
        spin_flip_dj=False,
        tau=-1.0,
        rank=0,
        verbose=True,
        dense_cutoff=100,
        slaterWeightMin=1e-12,
        truncation_threshold=100,
    )
    assert dc is not None


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
@patch("impurityModel.ed.get_spectra.get_noninteracting_hamiltonian_operator")
@patch("impurityModel.ed.selfenergy.finite.getUop")
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
@patch("impurityModel.ed.get_spectra.get_noninteracting_hamiltonian_operator")
@patch("impurityModel.ed.selfenergy.finite.getUop")
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
