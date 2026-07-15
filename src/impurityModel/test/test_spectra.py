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


def test_getInversePhotoEmissionOperators():
    nBaths = OrderedDict([(2, 10)])
    ops = spectra.getInversePhotoEmissionOperators(nBaths, l=2)
    # l=2 => 5 m values (-2, -1, 0, 1, 2)
    # s=2 => 2 spins
    # Total 10 operators
    assert len(ops) == 10
    assert list(ops[0].values())[0] == 1


def test_getPhotoEmissionOperators():
    nBaths = OrderedDict([(2, 10)])
    ops = spectra.getPhotoEmissionOperators(nBaths, l=2)
    assert len(ops) == 10
    assert list(ops[0].values())[0] == 1


# The transition-operator builders live in transition_operators.py; patch and call them there
# (their internal cross-calls -- getDipoleOperators->getDipoleOperator, getNIXSOperators->
# getNIXSOperator -- resolve inside that module).
@patch("impurityModel.ed.transition_operators.gauntC")
def test_getDipoleOperator(mock_gauntC):
    nBaths = OrderedDict([(2, 10), (1, 6)])
    mock_gauntC.return_value = 1.0
    n = [1, 0, 0]
    op = transition_operators.getDipoleOperator(nBaths, n)
    assert isinstance(op, dict)
    assert len(op) > 0


@patch("impurityModel.ed.transition_operators.getDipoleOperator")
def test_getDipoleOperators(mock_getDipoleOperator):
    mock_getDipoleOperator.return_value = {"mock": 1}
    ops = transition_operators.getDipoleOperators(OrderedDict(), [[1, 0, 0], [0, 1, 0]])
    assert len(ops) == 2
    assert ops[0] == {"mock": 1}


@patch("impurityModel.ed.transition_operators.getDipoleOperator")
@patch("impurityModel.ed.transition_operators.daggerOp")
def test_getDaggeredDipoleOperators(mock_daggerOp, mock_getDipoleOperator):
    mock_getDipoleOperator.return_value = {"mock": 1}
    mock_daggerOp.return_value = {"mock_dag": 1}
    ops = transition_operators.getDaggeredDipoleOperators(OrderedDict(), [[1, 0, 0]])
    assert len(ops) == 1
    assert ops[0] == {"mock_dag": 1}


@patch("impurityModel.ed.transition_operators.si.simpson")
@patch("impurityModel.ed.transition_operators.spherical_jn")
def test_getNIXSOperator(mock_jn, mock_simpson):
    mock_simpson.return_value = 1.0
    mock_jn.return_value = 1.0
    nBaths = OrderedDict([(2, 10)])
    op = transition_operators.getNIXSOperator(
        nBaths, [1, 1, 1], 2, 2, np.array([1.0]), np.array([1.0]), np.array([1.0]), kmin=1
    )
    assert isinstance(op, dict)


def test_getNIXSOperators():
    with patch("impurityModel.ed.transition_operators.getNIXSOperator") as mock_nixs:
        mock_nixs.return_value = {"op": 1}
        ops = transition_operators.getNIXSOperators(OrderedDict(), [[1, 1, 1]], 2, 2, [1], [1], [1], 1)
        assert len(ops) == 1
        assert ops[0] == {"op": 1}


@patch("impurityModel.ed.spectra._sector_restrictions_per_top")
@patch("impurityModel.ed.spectra.gf._build_excited_restrictions")
@patch("impurityModel.ed.spectra.gf.enumerate_gf_units")
@patch("impurityModel.ed.spectra.gf.unit_cost_weights")
@patch("impurityModel.ed.spectra.gf.run_units_distributed")
@patch("impurityModel.ed.spectra.gf.calc_thermally_averaged_G")
def test_getSpectra_new(mock_therm_G, mock_run, mock_weights, mock_enum, mock_build, mock_sector):
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

    gs = spectra.getSpectra_new(
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


@patch("impurityModel.ed.spectra.getInversePhotoEmissionOperators")
@patch("impurityModel.ed.spectra.getPhotoEmissionOperators")
@patch("impurityModel.ed.spectra.getSpectra_new")
@patch("impurityModel.ed.spectra.getSpectra_tensor")
@patch("impurityModel.ed.spectra.component_symmetry_reduction")
@patch("impurityModel.ed.spectra.extract_tensors")
@patch("impurityModel.ed.spectra.getDipoleOperators")
@patch("impurityModel.ed.spectra.getNIXSOperators")
def test_simulate_spectra(
    mock_getNIXS,
    mock_getDipole,
    mock_extract,
    mock_reduction,
    mock_getSpectra_tensor,
    mock_getSpectra_new,
    mock_getPS,
    mock_getIPS,
):
    mock_getSpectra_new.return_value = np.zeros((10, 1), dtype=complex)
    # XAS now goes through the polarization-tensor path (unprojected dipole): getSpectra_tensor
    # returns the (n_w, m, m) chi tensor directly (no contraction inside simulate_spectra).
    mock_getSpectra_tensor.return_value = np.zeros((10, 3, 3), dtype=complex)
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
    assert mock_getSpectra_new.called
    assert mock_getSpectra_tensor.called
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
