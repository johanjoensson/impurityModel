import pytest
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
import math
from collections import OrderedDict
import scipy.integrate as si
from scipy.special import spherical_jn

from impurityModel.ed import spectra
from impurityModel.ed import hamiltonian_io
from impurityModel.ed import transition_operators
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator

# --- tests for spectra.py ---


def test_sph_harm():
    # Test m=0, n=0 -> should be 1 / sqrt(4*pi)
    val = spectra.sph_harm(0, 0, 0, 0)
    expected = 1.0 / np.sqrt(4 * np.pi)
    assert np.isclose(val, expected)


def test_inverse_photoemission_operators():
    nBaths = OrderedDict([(2, 10)])
    ops = spectra.inverse_photoemission_operators(nBaths, l=2)
    # l=2 => 5 m values (-2, -1, 0, 1, 2)
    # s=2 => 2 spins
    # Total 10 operators
    assert len(ops) == 10
    assert list(ops[0].values())[0] == 1


def test_photoemission_operators():
    nBaths = OrderedDict([(2, 10)])
    ops = spectra.photoemission_operators(nBaths, l=2)
    assert len(ops) == 10
    assert list(ops[0].values())[0] == 1


# The transition-operator builders live in transition_operators.py; patch and call them there
# (their internal cross-calls -- dipole_operators->dipole_operator, nixs_operators->
# nixs_operator -- resolve inside that module).
@patch("impurityModel.ed.transition_operators.gauntC")
def test_dipole_operator(mock_gauntC):
    nBaths = OrderedDict([(2, 10), (1, 6)])
    mock_gauntC.return_value = 1.0
    n = [1, 0, 0]
    op = transition_operators.dipole_operator(nBaths, n)
    assert isinstance(op, dict)
    assert len(op) > 0


@patch("impurityModel.ed.transition_operators.dipole_operator")
def test_dipole_operators(mock_dipole_operator):
    mock_dipole_operator.return_value = {"mock": 1}
    ops = transition_operators.dipole_operators(OrderedDict(), [[1, 0, 0], [0, 1, 0]])
    assert len(ops) == 2
    assert ops[0] == {"mock": 1}


@patch("impurityModel.ed.transition_operators.dipole_operator")
@patch("impurityModel.ed.transition_operators.daggerOp")
def test_daggered_dipole_operators(mock_daggerOp, mock_dipole_operator):
    mock_dipole_operator.return_value = {"mock": 1}
    mock_daggerOp.return_value = {"mock_dag": 1}
    ops = transition_operators.daggered_dipole_operators(OrderedDict(), [[1, 0, 0]])
    assert len(ops) == 1
    assert ops[0] == {"mock_dag": 1}


@patch("impurityModel.ed.transition_operators.si.simpson")
@patch("impurityModel.ed.transition_operators.spherical_jn")
def test_nixs_operator(mock_jn, mock_simpson):
    mock_simpson.return_value = 1.0
    mock_jn.return_value = 1.0
    nBaths = OrderedDict([(2, 10)])
    op = transition_operators.nixs_operator(
        nBaths, [1, 1, 1], 2, 2, np.array([1.0]), np.array([1.0]), np.array([1.0]), kmin=1
    )
    assert isinstance(op, dict)


def test_nixs_operators():
    with patch("impurityModel.ed.transition_operators.nixs_operator") as mock_nixs:
        mock_nixs.return_value = {"op": 1}
        ops = transition_operators.nixs_operators(OrderedDict(), [[1, 1, 1]], 2, 2, [1], [1], [1], 1)
        assert len(ops) == 1
        assert ops[0] == {"op": 1}


@patch("impurityModel.ed.spectra._sector_restrictions_per_top")
@patch("impurityModel.ed.spectra.gf._build_excited_restrictions")
@patch("impurityModel.ed.spectra.gf.enumerate_gf_units")
@patch("impurityModel.ed.spectra.gf.unit_cost_weights")
@patch("impurityModel.ed.spectra.gf.run_units_distributed")
@patch("impurityModel.ed.spectra.gf.calc_thermally_averaged_G")
def test_calc_spectra(mock_therm_G, mock_run, mock_weights, mock_enum, mock_build, mock_sector):
    from impurityModel.ed.gf_units import GFUnit

    mock_sector.return_value = None
    mock_build.return_value = (None, None)
    # One (tOp, eigenstate) unit whose kernel result is a single width-1 r column.
    mock_enum.return_value = ([GFUnit(0, (0,), 1, 0.1)], [[MagicMock()]], [None])
    mock_weights.return_value = np.array([1.0])
    mock_run.return_value = [(None, None, [np.zeros((1, 1), dtype=complex)])]
    mock_therm_G.return_value = np.zeros((10, 1, 1), dtype=complex)

    basis = MagicMock()
    basis.comm = None

    gs = spectra.calc_spectra(
        MagicMock(spec=ManyBodyOperator),
        [MagicMock(spec=ManyBodyOperator)],
        [MagicMock()],
        [0.0],
        0.01,
        np.linspace(-1, 1, 10),
        basis,
        0.1,
        1e-4,
        False,
        0,
        {1: (0, 0)},
        {1: (0, 0)},
        {1: (0, 0)},
    )
    assert gs.shape == (10, 1)


@patch("impurityModel.ed.spectra.inverse_photoemission_operators")
@patch("impurityModel.ed.spectra.photoemission_operators")
@patch("impurityModel.ed.spectra.calc_spectra")
@patch("impurityModel.ed.spectra.calc_spectra_tensor")
@patch("impurityModel.ed.spectra.component_symmetry_reduction")
@patch("impurityModel.ed.spectra.extract_tensors")
@patch("impurityModel.ed.spectra.dipole_operators")
@patch("impurityModel.ed.spectra.nixs_operators")
def test_simulate_spectra(
    mock_getNIXS,
    mock_getDipole,
    mock_extract,
    mock_reduction,
    mock_calc_spectra_tensor,
    mock_calc_spectra,
    mock_getPS,
    mock_getIPS,
):
    mock_calc_spectra.return_value = np.zeros((10, 1), dtype=complex)
    # XAS now goes through the polarization-tensor path (unprojected dipole): calc_spectra_tensor
    # returns the (n_w, m, m) chi tensor directly (no contraction inside simulate_spectra).
    mock_calc_spectra_tensor.return_value = np.zeros((10, 3, 3), dtype=complex)
    mock_extract.return_value = (np.zeros((1, 1), dtype=complex), None, 0)
    mock_reduction.return_value = None
    mock_getIPS.return_value = [{((0, "c"),): 1}]
    mock_getPS.return_value = [{((0, "a"),): 1}]
    mock_getDipole.return_value = [{((0, "a"),): 1}]
    mock_getNIXS.return_value = [{((0, "a"),): 1}]

    hOp = MagicMock(spec=ManyBodyOperator)
    h5f = MagicMock()
    spectra.simulate_spectra(
        es=[0.0],
        psis=[MagicMock()],
        hOp=hOp,
        tau=0.01,
        w=np.linspace(-1, 1, 10),
        delta=0.1,
        epsilons=[[1, 0, 0]],
        wLoss=np.linspace(-1, 1, 10),
        deltaNIXS=0.1,
        qsNIXS=[[1, 1, 1]],
        liNIXS=2,
        ljNIXS=2,
        RiNIXS=np.array([1.0]),
        RjNIXS=np.array([1.0]),
        radialMesh=np.array([1.0]),
        wIn=[],
        deltaRIXS=0.1,
        epsilonsRIXSin=[],
        epsilonsRIXSout=[],
        restrictions={},
        h5f=h5f,
        nBaths=OrderedDict([(2, 10), (1, 6)]),
        XAS_projectors=None,
        RIXS_projectors=None,
        basis=MagicMock(),
        occ_cutoff=0,
        dN={},
        slaterWeightMin=1e-4,
        verbose=False,
    )
    # The solvers are collective, so the mocks are called on every rank; the h5 datasets and
    # the file close, however, are written on the root rank only (simulate_spectra gates them
    # on ``spectra.rank == 0``). Under ``mpiexec -n N`` this unmarked test runs on every rank,
    # so the write assertions must be guarded or they fail on the non-root ranks.
    assert mock_calc_spectra.called
    assert mock_calc_spectra_tensor.called
    if spectra.rank == 0:
        written = {c.args[0] for c in h5f.create_dataset.call_args_list}
        assert {"PS/spectra", "XPS/spectra", "NIXS/spectra", "XAS/tensor"} <= written
        assert not any(name.endswith("thermal") for name in written)  # no legacy dataset names
        h5f.close.assert_called_once()


# --- tests for hamiltonian_io.py ---


def test_gethHfieldop():
    from impurityModel.ed import atomic_physics

    op = atomic_physics.gethHfieldop(1.0, 0.0, 0.0, l=2)
    assert len(op) > 0


def test_read_tuple():
    s = "((0, c), (1, a))"
    t = hamiltonian_io.read_tuple(s)
    assert t == ((0, "c"), (1, "a"))


def test_read_h0_dict():
    file_content = "((0, 'c'), (0, 'a')) 1.0\n"
    with patch("impurityModel.ed.op_parser.parse_file") as mock_parse:
        mock_parse.return_value = {0: {((0, "c"), (0, "a")): 1.0}}
        d = hamiltonian_io.read_h0_dict("dummy.dat")
        assert d == {((0, "c"), (0, "a")): 1.0}


def test_read_pickled_file():
    with patch("builtins.open", mock_open()) as m:
        with patch("pickle.load") as mock_load:
            mock_load.return_value = "data"
            res = hamiltonian_io.read_pickled_file("dummy.pickle")
            assert res == "data"


def test_read_h0_CF_file():
    json_data = '{"e_imp": -1.0, "e_deltaO_imp": 0.5}'
    with patch("builtins.open", mock_open(read_data=json_data)):
        res = hamiltonian_io.read_h0_CF_file("dummy.json")
        assert res[0] == -1.0
        assert res[1] == 0.5


@patch("impurityModel.ed.hamiltonian_io.read_h0_CF_file")
def test_get_CF_hamiltonian(mock_read_cf):
    mock_read_cf.return_value = (-1.0, 0.5, -4.0, -6.0, 3.0, 2.0, 1.0, 1.0, 0.5, 0.5)
    nBaths = {2: 10}
    nValBaths = {2: 10}
    h0 = hamiltonian_io.get_CF_hamiltonian(nBaths, nValBaths, "dummy.json")
    assert isinstance(h0, dict)
    assert len(h0) > 0


@patch("impurityModel.ed.hamiltonian_io.read_h0_operator")
def test_get_noninteracting_hamiltonian_operator(mock_read_h0):
    mock_read_h0.return_value = {}
    h0 = hamiltonian_io.get_noninteracting_hamiltonian_operator(
        {2: 10, 1: 6}, {2: 10, 1: 6}, (0.1, 0.1), (0.0, 0.0, 0.0), "dummy.pickle", 0, False
    )
    assert isinstance(h0, dict)


@patch("impurityModel.ed.hamiltonian_io.get_noninteracting_hamiltonian_operator")
def test_get_hamiltonian_operator(mock_h0):
    mock_h0.return_value = {}
    hOp = hamiltonian_io.get_hamiltonian_operator(
        {2: 10, 1: 6},
        {2: 10, 1: 6},
        ([1.0] * 5, [1.0] * 3, [1.0] * 3, [1.0] * 4),
        (0.1, 0.1),
        ({2: 8, 1: 6}, 1.0),
        (0.0, 0.0, 0.0),
        "dummy.pickle",
        0,
        False,
    )
    assert isinstance(hOp, dict)


def test_calc_spectra_extra_meshes_match_direct_calls():
    """extra_meshes evaluates the same Lanczos coefficients on more meshes.

    The list return must reproduce (i) the plain single-mesh call on the primary mesh and
    (ii) a plain call on the extra mesh, including a purely imaginary (Matsubara-style)
    mesh with delta = 0.
    """
    from mpi4py import MPI

    from impurityModel.ed.manybody_basis import Basis
    from impurityModel.ed.ManyBodyUtils import ManyBodyState, SlaterDeterminant

    # 0=imp_up, 1=imp_dn, 2=bath_up, 3=bath_dn; U on the impurity.
    eps_i, eps_b, v, u = -1.0, 0.5, 0.7, 3.0
    terms = {((o, "c"), (o, "a")): eps_i if o < 2 else eps_b for o in range(4)}
    for a, b in ((0, 2), (1, 3)):
        terms[((a, "c"), (b, "a"))] = v
        terms[((b, "c"), (a, "a"))] = v
    terms[((0, "c"), (1, "c"), (1, "a"), (0, "a"))] = u
    hOp = ManyBodyOperator(terms)

    def _b(occ):
        data = bytearray(1)
        for o in occ:
            data[0] |= 1 << (7 - o)
        return bytes(data)

    dets = [_b((u_, d_)) for u_ in (0, 2) for d_ in (1, 3)]
    gs = ManyBodyState({SlaterDeterminant.from_bytes(dets[0]): 1.0})

    def fresh_basis():
        return Basis(
            impurity_orbitals={0: [[0, 1]]},
            bath_states=({0: [[2, 3]]}, {0: [[]]}),
            initial_basis=list(dets),
            verbose=False,
            comm=MPI.COMM_SELF,
        )

    tOps = [ManyBodyOperator({((0, "a"),): 1.0})]
    w = np.linspace(-4.0, 4.0, 21)
    w_matsubara = 1j * np.linspace(0.1, 3.0, 7)
    delta = 0.15
    common = dict(
        psis=[gs],
        es=[float(eps_i)],
        tau=0.01,
        delta=delta,
        slaterWeightMin=0.0,
        verbose=False,
        occ_cutoff=1e-12,
        dN_imp={0: (1, 1)},
        dN_val={0: (1, 1)},
        dN_con={0: (0, 0)},
    )
    combined = spectra.calc_spectra(hOp, tOps, w=w, basis=fresh_basis(), extra_meshes=[(w_matsubara, 0.0)], **common)
    assert isinstance(combined, list) and len(combined) == 2
    direct_w = spectra.calc_spectra(hOp, tOps, w=w, basis=fresh_basis(), **common)
    direct_iw = spectra.calc_spectra(hOp, tOps, w=w_matsubara, basis=fresh_basis(), **{**common, "delta": 0.0})
    np.testing.assert_allclose(combined[0], direct_w, atol=1e-12)
    np.testing.assert_allclose(combined[1], direct_iw, atol=1e-12)
