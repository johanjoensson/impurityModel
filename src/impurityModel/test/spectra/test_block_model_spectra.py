"""Spectra of a block model (no ``(l, s, m)`` labels) against the labelled model they describe.

A K-edge model (1s core + 2p valence + two bath orbitals) is solved twice: once as the
``from_shells`` model the TOML path builds, and once as a *block model* -- the same operator, but
shells found through ``rot_to_spherical`` / ``shell_ls`` and transition operators built from flat
indices. The labelled run is the oracle. The block model is then rotated, given an explicit dipole
matrix and split into ``h0 + u4``; each must leave the spectra where they were.
"""

import dataclasses
import itertools

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.model import ImpurityModel
from impurityModel.ed.options import NIXSOptions, PESOptions, XASOptions, XPSOptions
from impurityModel.ed.spectra import CARTESIAN
from impurityModel.ed.spectroscopy import calc_spectroscopy, spectra_layout
from impurityModel.inputformat.build import build
from impurityModel.inputformat.reader import load_input
from impurityModel.test.inputformat.test_k_edge_end_to_end import K_EDGE_RUN, small_p_shell_h0  # noqa: F401

COMM = MPI.COMM_WORLD
# MPI runs agree with themselves only to the iterative solves' tolerance.
ATOL = 1e-10 if COMM.size == 1 else 1e-4


@pytest.fixture
def labelled(small_p_shell_h0, tmp_path):  # noqa: F811
    path = tmp_path / "k_edge_block.toml"
    path.write_text(K_EDGE_RUN.format(h0=small_p_shell_h0.name))
    built = build(load_input(path))
    basis = dataclasses.replace(built.basis, mixed_valence="off", tau=0.01)
    return built.model, basis


def _block_twin(model, **changes):
    """The labelled K-edge model as a block model: shells 0 (1s, l=0) and 1 (2p, l=1), identity rotation."""
    n_imp = 8
    fields = dict(
        h0=model.h0,
        impurity_orbitals={0: list(range(n_imp))},
        rot_to_spherical=np.eye(n_imp, dtype=complex),
        bath_states=({0: [8, 9]}, {0: []}),
        n_spin_orbitals=model.n_spin_orbitals,
        shell_ls=(0, 1),
        valence_shell=1,
        core_shell=0,
    )
    fields.update(changes)
    return ImpurityModel(**fields)


def _run(model, requests, basis):
    return calc_spectroscopy(model, requests, basis, comm=COMM, verbosity=0, cluster_label="block_test")


def _requests():
    w = np.linspace(-6, 6, 41)
    return [PESOptions(w=w, delta=0.3), XPSOptions(w=w, delta=0.3), XASOptions(w=w, delta=0.3)]


def _assert_same(result, reference, keys, *, atol=ATOL):
    if COMM.rank != 0:
        return
    for key in keys:
        np.testing.assert_allclose(result["datasets"][key], reference["datasets"][key], atol=atol, err_msg=key)


def test_a_block_model_of_the_same_operator_reproduces_the_labelled_spectra(labelled):
    model, basis = labelled
    reference = _run(model, _requests(), basis)
    result = _run(_block_twin(model), _requests(), basis)
    _assert_same(result, reference, ("PS/spectra", "XPS/spectra", "XAS/tensor"))


def _rotate_valence(h0, rows, unitary):
    """``h0`` with the orbitals ``rows`` re-expressed in the basis of ``unitary``: c^+_a = sum_i U[i,a] c^+_i."""
    position = {o: k for k, o in enumerate(rows)}

    def expand(index, action):
        if index not in position:
            return [((index, action), 1.0)]
        a = position[index]
        if action == "c":
            return [((rows[i], "c"), unitary[i, a]) for i in range(len(rows))]
        return [((rows[i], "a"), np.conj(unitary[i, a])) for i in range(len(rows))]

    out = {}
    for process, value in h0.items():
        for choice in itertools.product(*(expand(i, act) for i, act in process)):
            coefficient = value
            for _, c in choice:
                coefficient = coefficient * c
            key = tuple(op for op, _ in choice)
            out[key] = out.get(key, 0) + coefficient
    return {k: v for k, v in out.items() if abs(v) > 1e-13}


def test_rotating_the_valence_shell_leaves_the_invariant_spectra_alone(labelled):
    model, basis = labelled
    reference = _run(model, _requests(), basis)
    rows = list(range(2, 8))
    rng = np.random.default_rng(3)
    unitary = np.linalg.qr(rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6)))[0]
    rot = np.eye(8, dtype=complex)
    rot[2:8, 2:8] = unitary
    rotated = _block_twin(model, h0=_rotate_valence(model.h0, rows, unitary), rot_to_spherical=rot)
    result = _run(rotated, _requests(), basis)
    _assert_same(result, reference, ("XAS/tensor", "XPS/spectra"))
    if COMM.rank == 0:
        np.testing.assert_allclose(
            result["datasets"]["PS/spectra"].sum(axis=1), reference["datasets"]["PS/spectra"].sum(axis=1), atol=ATOL
        )


def _atomic_dipole(model):
    """The atomic dipole as a ``(3, n_imp, n_imp)`` array in the model's input basis."""
    ops = spectra_layout(model, [XASOptions(w=np.linspace(-1, 1, 3))])[0].operators
    d = np.zeros((3, 8, 8), dtype=complex)
    for k, op in enumerate(ops.dipole(CARTESIAN, 0, 1)):
        for ((i, _), (j, _)), value in op.items():
            d[k, i, j] = value
    return d


def test_an_explicit_dipole_equal_to_the_atomic_one_changes_nothing_and_scales_quadratically(labelled):
    model, basis = labelled
    block = _block_twin(model)
    w = np.linspace(-6, 6, 41)
    atomic = _run(block, [XASOptions(w=w, delta=0.3)], basis)
    d = _atomic_dipole(block)
    explicit = _run(block, [XASOptions(w=w, delta=0.3, dipole=d)], basis)
    scaled = _run(block, [XASOptions(w=w, delta=0.3, dipole=3.0 * d)], basis)
    _assert_same(explicit, atomic, ("XAS/tensor",))
    if COMM.rank == 0:
        assert np.max(np.abs(atomic["datasets"]["XAS/tensor"])) > 0
        np.testing.assert_allclose(
            scaled["datasets"]["XAS/tensor"], 9.0 * atomic["datasets"]["XAS/tensor"], atol=ATOL * 9
        )


def test_a_dipole_carried_by_the_model_equals_the_xas_option_and_the_option_overrides_it(labelled):
    from impurityModel.ed.local_operators import LocalOperators

    model, basis = labelled
    block = _block_twin(model)
    w = np.linspace(-6, 6, 41)
    d = _atomic_dipole(block)
    via_option = _run(block, [XASOptions(w=w, delta=0.3, dipole=3.0 * d)], basis)
    ops = LocalOperators(sz=np.zeros((8, 8)), dipole=3.0 * d, source="rspt", frame="global")
    carried = _run(_block_twin(model, operators=ops), [XASOptions(w=w, delta=0.3)], basis)
    _assert_same(carried, via_option, ("XAS/tensor",))
    # the run option wins over what the model carries
    overridden = _run(_block_twin(model, operators=ops), [XASOptions(w=w, delta=0.3, dipole=d)], basis)
    atomic = _run(block, [XASOptions(w=w, delta=0.3)], basis)
    _assert_same(overridden, atomic, ("XAS/tensor",))


def test_a_dipole_with_the_valence_and_core_blocks_swapped_is_not_the_atomic_spectrum(labelled):
    """The equalities above must be able to fail: a transposed (core <- valence) dipole gives another spectrum."""
    model, basis = labelled
    block = _block_twin(model)
    w = np.linspace(-6, 6, 41)
    atomic = _run(block, [XASOptions(w=w, delta=0.3)], basis)
    d = _atomic_dipole(block)
    # An operator that cannot move an electron from the core into the valence shell carries no XAS weight.
    wrong = _run(block, [XASOptions(w=w, delta=0.3, dipole=np.swapaxes(d, 1, 2))], basis)
    if COMM.rank == 0:
        assert not np.allclose(wrong["datasets"]["XAS/tensor"], atomic["datasets"]["XAS/tensor"], atol=1e-6)


def test_h0_plus_u4_is_the_full_operator(labelled):
    model, basis = labelled
    one_body = {k: v for k, v in model.h0.items() if len(k) == 2}
    two_body = {k: v for k, v in model.h0.items() if len(k) == 4}
    assert one_body and two_body
    reference = _run(_block_twin(model), _requests(), basis)
    split = _run(_block_twin(model, h0=one_body, u4=two_body), _requests(), basis)
    _assert_same(split, reference, ("PS/spectra", "XPS/spectra", "XAS/tensor"))


def test_missing_shell_roles_and_nixs_raise_on_every_rank(labelled):
    model, basis = labelled
    with pytest.raises(ValueError, match="valence_shell"):
        _run(_block_twin(model, valence_shell=None, core_shell=None), _requests(), basis)
    with pytest.raises(ValueError, match="core"):
        _run(_block_twin(model, core_shell=None), _requests(), basis)
    nixs = NIXSOptions(radial=(np.linspace(0.01, 5, 20), np.ones(20), np.ones(20)), l_final=1, l_initial=1)
    with pytest.raises(ValueError, match="NIXS"):
        _run(_block_twin(model), [nixs], basis)
