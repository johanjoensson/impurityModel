from collections import OrderedDict
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

from impurityModel.ed import hamiltonian_io, spectra, spectra_techniques, transition_operators
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
    assert next(iter(ops[0].values())) == 1


def test_photoemission_operators():
    nBaths = OrderedDict([(2, 10)])
    ops = spectra.photoemission_operators(nBaths, l=2)
    assert len(ops) == 10
    assert next(iter(ops[0].values())) == 1


# The transition-operator builders live in transition_operators.py; patch and call them there
# (their internal cross-calls -- dipole_operators->dipole_operator, nixs_operators->
# nixs_operator -- resolve inside that module).
@patch("impurityModel.ed.transition_operators.gauntC")
def test_dipole_operator(mock_gauntC):
    nBaths = OrderedDict([(2, 10), (1, 6)])
    mock_gauntC.return_value = 1.0
    n = [1, 0, 0]
    op = transition_operators.dipole_operator(nBaths, n, 1, 2)
    assert isinstance(op, dict)
    assert len(op) > 0


@patch("impurityModel.ed.transition_operators.dipole_operator")
def test_dipole_operators(mock_dipole_operator):
    mock_dipole_operator.return_value = {"mock": 1}
    ops = transition_operators.dipole_operators(OrderedDict(), [[1, 0, 0], [0, 1, 0]], 1, 2)
    assert len(ops) == 2
    assert ops[0] == {"mock": 1}


@patch("impurityModel.ed.transition_operators.dipole_operator")
@patch("impurityModel.ed.transition_operators.daggerOp")
def test_daggered_dipole_operators(mock_daggerOp, mock_dipole_operator):
    mock_dipole_operator.return_value = {"mock": 1}
    mock_daggerOp.return_value = {"mock_dag": 1}
    ops = transition_operators.daggered_dipole_operators(OrderedDict(), [[1, 0, 0]], 1, 2)
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
    # (alphas, betas, r_slices, cap_stats) -- the kernel reports its unit's basis size
    # alongside the Lanczos coefficients (see calc_spectra's per-unit basis report).
    # (alphas, betas, r per stacked state, cap record, convergence record) -- gf_engine's contract.
    coeffs = np.zeros((1, 1, 1), dtype=complex)
    mock_run.return_value = [
        (coeffs, coeffs, [np.zeros((1, 1), dtype=complex)], {"retained_size": 7, "cap_hit": False}, {})
    ]
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


@patch("impurityModel.ed.spectra_techniques.inverse_photoemission_operators")
@patch("impurityModel.ed.spectra_techniques.photoemission_operators")
@patch("impurityModel.ed.spectra_techniques.calc_spectra")
@patch("impurityModel.ed.spectra_techniques.calc_spectra_tensor")
@patch("impurityModel.ed.spectra_techniques.component_symmetry_reduction")
@patch("impurityModel.ed.spectra_techniques.extract_tensors")
@patch("impurityModel.ed.spectra_techniques.dipole_operators")
@patch("impurityModel.ed.spectra_techniques.nixs_operators")
def test_run_techniques(
    mock_getNIXS,
    mock_getDipole,
    mock_extract,
    mock_reduction,
    mock_calc_spectra_tensor,
    mock_calc_spectra,
    mock_getPS,
    mock_getIPS,
):
    from impurityModel.ed.options import NIXSOptions, PESOptions, XASOptions, XPSOptions
    from impurityModel.ed.spectra_techniques import (
        LabelledOperators,
        ShellLayout,
        SpectraTarget,
        run_techniques,
    )

    mock_calc_spectra.return_value = np.zeros((10, 1), dtype=complex)
    # XAS goes through the polarization-tensor path (unprojected dipole): calc_spectra_tensor
    # returns the (n_w, m, m) chi tensor directly (no contraction inside the technique).
    mock_calc_spectra_tensor.return_value = np.zeros((10, 3, 3), dtype=complex)
    mock_extract.return_value = (np.zeros((1, 1), dtype=complex), None, 0)
    mock_reduction.return_value = None
    mock_getIPS.return_value = [{((0, "c"),): 1}]
    mock_getPS.return_value = [{((0, "a"),): 1}]
    mock_getDipole.return_value = [{((0, "a"),): 1}]
    mock_getNIXS.return_value = [{((0, "a"),): 1}]

    basis = MagicMock()
    basis.comm.rank = spectra.rank
    target = SpectraTarget(
        hOp=MagicMock(spec=ManyBodyOperator),
        psis=[MagicMock()],
        es=[0.0],
        tau=0.01,
        basis=basis,
        occ_cutoff=0,
        slater_weight_min=1e-4,
        verbose=False,
        gf_pt2_tol=None,
        layout=ShellLayout(valence_key=2, core_key=1, operators=LabelledOperators(OrderedDict([(2, 10), (1, 6)]))),
    )
    w = np.linspace(-1, 1, 10)
    datasets = run_techniques(
        target,
        [
            XASOptions(w=w, delta=0.1),  # listed out of order on purpose
            NIXSOptions(radial=(np.array([1.0]), np.array([1.0]), np.array([1.0])), w_loss=w, qs=[[1, 1, 1]]),
            XPSOptions(w=w, delta=0.1),
            PESOptions(w=w, delta=0.1),
        ],
    )
    # The solvers are collective, so the mocks are called on every rank; the datasets are
    # returned on the root rank only. Under ``mpiexec -n N`` this unmarked test runs on every rank,
    # so the assertions must be guarded or they fail on the non-root ranks.
    assert mock_calc_spectra.called
    assert mock_calc_spectra_tensor.called
    if spectra.rank == 0:
        written = set(datasets)
        assert {"PS/spectra", "XPS/spectra", "NIXS/spectra", "XAS/tensor"} <= written
        assert list(datasets) == ["PS/spectra", "XPS/spectra", "NIXS/spectra", "XAS/tensor"], "execution order"
        assert not any(name.endswith("thermal") for name in written)  # no legacy dataset names
    else:
        assert datasets == {}


def test_a_core_technique_without_a_core_shell_raises_before_computing_anything():
    from impurityModel.ed.options import PESOptions, XASOptions
    from impurityModel.ed.spectra_techniques import LabelledOperators, ShellLayout, SpectraTarget, run_techniques

    target = SpectraTarget(
        hOp=MagicMock(spec=ManyBodyOperator),
        psis=[],
        es=[],
        tau=0.01,
        basis=MagicMock(),
        occ_cutoff=0,
        slater_weight_min=1e-4,
        verbose=False,
        gf_pt2_tol=None,
        layout=ShellLayout(valence_key=2, core_key=None, operators=LabelledOperators(OrderedDict([(2, 10)]))),
    )
    with patch("impurityModel.ed.spectra_techniques.calc_spectra") as calc:
        with pytest.raises(ValueError, match="core hole"):
            run_techniques(target, [PESOptions(), XASOptions()])
    assert not calc.called, "PES must not run before the missing core shell is reported"


def test_a_duplicate_request_is_refused():
    from impurityModel.ed.options import PESOptions
    from impurityModel.ed.spectra_techniques import run_techniques

    with pytest.raises(ValueError, match="more than one PES"):
        run_techniques(MagicMock(), [PESOptions(), PESOptions()])


def test_a_non_request_is_refused():
    from impurityModel.ed.spectra_techniques import run_techniques

    with pytest.raises(TypeError, match="not a spectrum request"):
        run_techniques(MagicMock(), [object()])


# --- tests for hamiltonian_io.py ---


def test_gethHfieldop():
    from impurityModel.ed import atomic_physics

    op = atomic_physics.gethHfieldop(1.0, 0.0, 0.0, l=2)
    assert len(op) > 0


def test_read_h0_dict():
    with patch("impurityModel.ed.op_parser.parse_file") as mock_parse:
        mock_parse.return_value = {0: {((0, "c"), (0, "a")): 1.0}}
        d = hamiltonian_io.read_h0_dict("dummy.dat")
        assert d == {((0, "c"), (0, "a")): 1.0}


def test_read_pickled_file():
    with patch("builtins.open", mock_open()), patch("pickle.load") as mock_load:
        mock_load.return_value = "data"
        res = hamiltonian_io.read_pickled_file("dummy.pickle")
        assert res == "data"


def test_read_h0_CF_file():
    json_data = '{"e_imp": -1.0, "e_deltaO_imp": 0.5}'
    with patch("builtins.open", mock_open(read_data=json_data)):
        e_imp, deltas, e_val, e_con, v_val, v_con = hamiltonian_io.read_h0_CF_file("dummy.json", l=2)
        assert e_imp == -1.0
        # One splitting for a d shell; an f shell would return two, which is the whole reason
        # this is a tuple rather than a scalar.
        assert deltas == (0.5,)
        # The rest fall back to the Ni-in-NiO defaults, keyed by irrep rather than positional.
        assert set(e_val) == set(e_con) == set(v_val) == set(v_con) == {"eg", "t2g"}
        assert v_val["eg"] == 1.883


@patch("impurityModel.ed.hamiltonian_io.read_h0_CF_file")
def test_get_CF_hamiltonian(mock_read_cf):
    mock_read_cf.return_value = (
        -1.0,
        (0.5,),
        {"eg": -4.0, "t2g": -6.0},
        {"eg": 3.0, "t2g": 2.0},
        {"eg": 1.0, "t2g": 1.0},
        {"eg": 0.5, "t2g": 0.5},
    )
    nBaths = {2: 10}
    nValBaths = {2: 10}
    h0 = hamiltonian_io.get_CF_hamiltonian(nBaths, nValBaths, "dummy.json", l=2)
    assert isinstance(h0, dict)
    assert len(h0) > 0


@patch("impurityModel.ed.hamiltonian_io.read_h0_operator")
def test_get_noninteracting_hamiltonian_operator(mock_read_h0):
    mock_read_h0.return_value = {}
    h0 = hamiltonian_io.get_noninteracting_hamiltonian_operator(
        {2: 10, 1: 6},
        {2: 10, 1: 6},
        "dummy.pickle",
        0,
        False,
        valence_l=2,
        xi_valence=0.1,
        hField=(0.0, 0.0, 0.0),
        core_l=1,
        xi_core=0.1,
    )
    assert isinstance(h0, dict)


@patch("impurityModel.ed.hamiltonian_io.get_noninteracting_hamiltonian_operator")
def test_get_hamiltonian_operator(mock_h0):
    mock_h0.return_value = {}
    hOp = hamiltonian_io.get_hamiltonian_operator(
        {2: 10, 1: 6},
        {2: 10, 1: 6},
        ([1.0] * 5, [1.0] * 3, [1.0] * 3, [1.0] * 4),
        ({2: 8, 1: 6}, 1.0),
        "dummy.pickle",
        0,
        False,
        valence_l=2,
        xi_valence=0.1,
        hField=(0.0, 0.0, 0.0),
        core_l=1,
        xi_core=0.1,
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


def test_merge_spectra_datasets_is_the_weighted_sum_and_keeps_dtypes():
    a = {"PS/spectra": np.array([[1.0 + 1j]]), "RIXS/tensor": np.array([2.0 + 0j], dtype=np.complex64), "P/x": "proj"}
    b = {"PS/spectra": np.array([[3.0 - 1j]]), "RIXS/tensor": np.array([4.0 + 0j], dtype=np.complex64), "P/x": "other"}
    m = spectra_techniques.merge_spectra_datasets([a, b], [1.0, 3.0])  # weights are renormalised
    np.testing.assert_allclose(m["PS/spectra"], [[0.25 * (1 + 1j) + 0.75 * (3 - 1j)]])
    assert m["RIXS/tensor"].dtype == np.complex64
    np.testing.assert_allclose(m["RIXS/tensor"], [3.5])
    assert m["P/x"] == "proj"
    assert spectra_techniques.merge_spectra_datasets([a], [0.3])["PS/spectra"][0, 0] == a["PS/spectra"][0, 0]


def test_a_blocked_transition_window_falls_back_to_the_shared_window():
    from impurityModel.ed.basis_restrictions import intersect_sector_window

    shared = {frozenset({0, 1}): (0, 2)}
    assert intersect_sector_window(shared, {frozenset({0, 1}): (1, 2)}) == {frozenset({0, 1}): (1, 2)}
    assert intersect_sector_window(shared, {frozenset({0, 1}): (3, 3)}) == shared  # infeasible: zero seeds anyway
    assert intersect_sector_window(shared, {frozenset({0, 1}): (3, 3)}) is not shared


def test_a_window_that_is_merely_too_narrow_still_raises():
    import pytest

    from impurityModel.ed.basis_restrictions import intersect_sector_window

    shared = {frozenset({0, 1}): (0, 1)}
    with pytest.raises(ValueError, match="admit no determinant"):
        intersect_sector_window(shared, {frozenset({0, 1}): (2, 2)})  # fits two orbitals: not blocked, just too narrow
