"""Unit tests for impurityModel.ed.gs_statistics.

These exercise the (serial) statistics math directly: configuration bucketing,
marginal occupation distributions, participation / entropy, natural-orbital
occupations, top determinants, and the JSON round-trip. The MPI reduction path is a
thin wrapper around the same partial results (see ``compute_gs_statistics``).
"""

import json
import math

import numpy as np
import pytest

from impurityModel.ed.gs_statistics import (
    _det_participation,
    _marginal,
    _participation,
    _sorted_weight_rows,
    compute_gs_statistics,
    save_gs_statistics,
)
from impurityModel.ed.ManyBodyUtils import ManyBodyState, SlaterDeterminant

N_ORBS = 8
IMPURITY = [0, 1]
VALENCE = [2, 3]
CONDUCTION = [4, 5]


class _FakeBasis:
    """Minimal stand-in for Basis exposing only what compute_gs_statistics needs."""

    num_spin_orbitals = N_ORBS
    impurity_spin_orbital_indices = IMPURITY
    valence_spin_orbital_indices = VALENCE
    conduction_spin_orbital_indices = CONDUCTION
    is_distributed = False
    comm = None


def _det(occupied):
    """SlaterDeterminant with the given occupied orbitals (MSB-first bit convention)."""
    data = bytearray((N_ORBS + 7) // 8)
    for orb in occupied:
        data[orb // 8] |= 1 << (7 - orb % 8)
    return SlaterDeterminant.from_bytes(bytes(data))


def _make_psis():
    """Two states with known configuration / determinant weights.

    state 0: |[0,1,2,4]>*sqrt(.6) + |[0,2,3,4,5]>*sqrt(.4)  -> configs (2,1,1),(1,2,2)
    state 1: |[0,2,4]>                                       -> config  (1,1,1)
    """
    psi0 = ManyBodyState({_det([0, 1, 2, 4]): math.sqrt(0.6), _det([0, 2, 3, 4, 5]): math.sqrt(0.4)})
    psi1 = ManyBodyState({_det([0, 2, 4]): 1.0})
    return [psi0, psi1]


def _stats():
    # es equal -> Boltzmann weights 0.5 / 0.5, giving exact, hand-checkable numbers.
    thermal_rho = np.zeros((N_ORBS, N_ORBS), dtype=complex)
    thermal_rho[np.ix_(IMPURITY, IMPURITY)] = np.array([[0.7, 0.1], [0.1, 0.3]], dtype=complex)
    return compute_gs_statistics(
        _FakeBasis(),
        _make_psis(),
        es=np.array([0.0, 0.0]),
        tau=1.0,
        thermal_rho=thermal_rho,
        impurity_indices=IMPURITY,
    )


def test_thermal_config_weights_sum_to_one():
    stats = _stats()
    rows = stats["thermal_config_weights"]["rows"]
    assert stats["thermal_config_weights"]["remaining"] is None
    total_fraction = sum(r["fraction"] for r in rows)
    assert total_fraction == pytest.approx(1.0)
    # Rows are sorted by descending weight and the cumulative column reaches 1.
    weights = [r["weight"] for r in rows]
    assert weights == sorted(weights, reverse=True)
    assert rows[-1]["cumulative"] == pytest.approx(1.0)
    by_config = {r["config"]: r["weight"] for r in rows}
    assert by_config == {
        (1, 1, 1): pytest.approx(0.5),
        (2, 1, 1): pytest.approx(0.3),
        (1, 2, 2): pytest.approx(0.2),
    }


def test_marginals_and_fluctuations():
    stats = _stats()
    imp = stats["marginals"]["impurity"]
    assert dict(imp["distribution"]) == {1: pytest.approx(0.7), 2: pytest.approx(0.3)}
    assert imp["mean"] == pytest.approx(1.3)
    assert imp["variance"] == pytest.approx(0.21)
    val = stats["marginals"]["valence"]
    assert val["mean"] == pytest.approx(1.2)
    assert val["variance"] == pytest.approx(0.16)


def test_participation_and_entropy():
    stats = _stats()
    probs = [0.5, 0.3, 0.2]
    expected_neff = 1.0 / sum(p * p for p in probs)
    expected_entropy = -sum(p * math.log(p) for p in probs)
    cfg = stats["participation"]["configurations"]
    det = stats["participation"]["determinants"]
    assert cfg["effective_number"] == pytest.approx(expected_neff)
    assert cfg["entropy"] == pytest.approx(expected_entropy)
    # The three determinants happen to carry the same weight set here.
    assert det["effective_number"] == pytest.approx(expected_neff)
    assert det["entropy"] == pytest.approx(expected_entropy)
    assert stats["num_determinants"] == 3


def test_natural_orbital_occupations_match_eigvalsh():
    stats = _stats()
    expected = np.sort(np.linalg.eigvalsh(np.array([[0.7, 0.1], [0.1, 0.3]])))[::-1]
    assert stats["natural_orbital_occupations"] == pytest.approx(list(expected))


def test_one_body_entropy_and_natural_vectors():
    stats = _stats()
    occ = np.array(stats["natural_orbital_occupations"])
    # S_1b = -sum [n ln n + (1-n) ln(1-n)] over the natural occupations.
    n = np.clip(occ, 1e-12, 1 - 1e-12)
    expected = float(-np.sum(n * np.log(n) + (1 - n) * np.log(1 - n)))
    assert stats["one_body_entropy"] == pytest.approx(expected)
    # Vectors diagonalize the impurity block, columns ordered like the occupations.
    imp_block = np.array([[0.7, 0.1], [0.1, 0.3]], dtype=complex)
    vecs = np.asarray(stats["natural_orbital_vectors"])
    assert np.allclose(vecs.conj().T @ imp_block @ vecs, np.diag(occ), atol=1e-12)
    assert stats["natural_orbital_basis"] == IMPURITY

    # Trivial occupations (0/1) -> zero one-body entropy.
    thermal_rho = np.zeros((N_ORBS, N_ORBS), dtype=complex)
    thermal_rho[0, 0] = 1.0
    stats0 = compute_gs_statistics(
        _FakeBasis(), _make_psis(), es=np.array([0.0, 0.0]), tau=1.0, thermal_rho=thermal_rho, impurity_indices=IMPURITY
    )
    assert stats0["one_body_entropy"] == pytest.approx(0.0, abs=1e-9)


def test_json_round_trip_complex_vectors(tmp_path):
    """Complex natural-orbital vectors serialize as [re, im] pairs."""
    stats = _stats()
    path = tmp_path / "stats_cplx.json"
    save_gs_statistics(stats, path)
    loaded = json.loads(path.read_text())
    vec00 = loaded["natural_orbital_vectors"][0][0]
    assert isinstance(vec00, list) and len(vec00) == 2  # [re, im]
    reconstructed = np.array([[complex(re, im) for re, im in row] for row in loaded["natural_orbital_vectors"]])
    assert np.allclose(reconstructed, np.asarray(stats["natural_orbital_vectors"]), atol=1e-12)


def test_top_determinants_sorted_and_labelled():
    stats = _stats()
    top = stats["top_determinants"]
    weights = [d["weight"] for d in top]
    assert weights == sorted(weights, reverse=True)
    assert weights[0] == pytest.approx(0.5)
    # Dominant determinant is |[0,2,4]> -> imp[0], val[2], con[4].
    assert top[0]["impurity_occupied"] == [0]
    assert top[0]["valence_occupied"] == [2]
    assert top[0]["conduction_occupied"] == [4]
    assert top[0]["config"] == (1, 1, 1)


def test_print_top_determinants_hole_notation(capsys):
    """Default printing shows valence holes / conduction particles; verbose>=2 full lists."""
    from impurityModel.ed.gs_statistics import print_gs_statistics

    stats = _stats()
    print_gs_statistics(stats)
    out = capsys.readouterr().out
    # Dominant determinant |[0,2,4]>: valence holes = {2,3} - {2} = [3], conduction particle [4].
    assert "imp occupied | val holes | con particles" in out
    assert "imp [0] | val holes [3] | con [4]" in out

    print_gs_statistics(stats, verbose=2)
    out = capsys.readouterr().out
    assert "occupied (imp|val|con)" in out
    assert "[0]|[2]|[4]" in out


def test_json_round_trip(tmp_path):
    stats = _stats()
    path = tmp_path / "stats.json"
    save_gs_statistics(stats, path)
    loaded = json.loads(path.read_text())
    assert loaded["num_determinants"] == 3
    assert loaded["marginals"]["impurity"]["mean"] == pytest.approx(1.3)
    # config keys round-trip as lists through JSON.
    assert loaded["thermal_config_weights"]["rows"][0]["config"] == [1, 1, 1]


def test_entanglement_entropy_analytic():
    """Impurity-bath entanglement: product state -> 0, Bell states -> ln 2."""
    from impurityModel.ed.gs_statistics import compute_entanglement_entropy

    tau = 1e-3
    # Product state: one impurity electron, one bath electron -> no entanglement.
    product = ManyBodyState({_det([0, 2]): 1.0})
    ent = compute_entanglement_entropy(_FakeBasis(), [product], np.array([0.0]), tau)
    assert ent["per_state_entropy"][0] == pytest.approx(0.0, abs=1e-12)
    assert ent["per_state_norm"][0] == pytest.approx(1.0)
    assert ent["n_imp_sectors"] == [1]

    # Bell state within one impurity sector: S = ln 2.
    bell = ManyBodyState({_det([1, 2]): math.sqrt(0.5), _det([0, 3]): -math.sqrt(0.5)})
    ent = compute_entanglement_entropy(_FakeBasis(), [bell], np.array([0.0]), tau)
    assert ent["per_state_entropy"][0] == pytest.approx(math.log(2))
    assert ent["spectrum_top"][0][:2] == pytest.approx([0.5, 0.5])

    # Bell state across impurity sectors (valence fluctuation): S = ln 2, two sectors.
    mixed_valence = ManyBodyState({_det([0, 1]): math.sqrt(0.5), _det([0, 2]): math.sqrt(0.5)})
    ent = compute_entanglement_entropy(_FakeBasis(), [mixed_valence], np.array([0.0]), tau)
    assert ent["per_state_entropy"][0] == pytest.approx(math.log(2))
    assert ent["n_imp_sectors"] == [1, 2]
    assert ent["sector_weights"][0] == {1: pytest.approx(0.5), 2: pytest.approx(0.5)}

    # Superposition on the impurity over the SAME bath configuration stays pure: S = 0.
    coherent = ManyBodyState({_det([0, 2]): math.sqrt(0.5), _det([1, 2]): math.sqrt(0.5)})
    ent = compute_entanglement_entropy(_FakeBasis(), [coherent], np.array([0.0]), tau)
    assert ent["per_state_entropy"][0] == pytest.approx(0.0, abs=1e-12)


def test_entanglement_thermal_vs_mixture():
    """Two orthogonal product eigenstates: thermal entropy = pure mixture entropy."""
    from impurityModel.ed.gs_statistics import compute_entanglement_entropy

    psis = [ManyBodyState({_det([0, 2]): 1.0}), ManyBodyState({_det([1, 3]): 1.0})]
    es = np.array([0.0, 0.5])
    tau = 1.0
    ent = compute_entanglement_entropy(_FakeBasis(), psis, es, tau)
    assert ent["per_state_entropy"] == pytest.approx([0.0, 0.0], abs=1e-12)
    assert ent["thermal_entropy"] == pytest.approx(ent["mixture_entropy"])


def test_entanglement_sector_weights_match_config_marginal():
    """Tr rho_imp[N] reproduces the per-state impurity-occupation weights."""
    from impurityModel.ed.gs_statistics import compute_entanglement_entropy

    ent = compute_entanglement_entropy(_FakeBasis(), _make_psis(), np.array([0.0, 0.0]), 1.0)
    # psi0: config (2,1,1) with weight 0.6 and (1,2,2) with weight 0.4.
    assert ent["sector_weights"][0] == {1: pytest.approx(0.4), 2: pytest.approx(0.6)}
    assert ent["sector_weights"][1] == {1: pytest.approx(1.0), 2: pytest.approx(0.0)}


def test_entanglement_memory_guard():
    from impurityModel.ed.gs_statistics import compute_entanglement_entropy

    ent = compute_entanglement_entropy(_FakeBasis(), _make_psis(), np.array([0.0, 0.0]), 1.0, max_bytes=1)
    assert ent is None


@pytest.mark.mpi
def test_entanglement_entropy_distributed_matches_serial():
    """Distributed impurity-RDM path reproduces the serial analytic entropies.

    States are fed on rank 0 ONLY and distributed with redistribute_psis (each
    determinant owned by exactly one rank); feeding replicated copies on every rank
    would double-count amplitudes.
    """
    from mpi4py import MPI

    from impurityModel.ed.gs_statistics import compute_entanglement_entropy
    from impurityModel.ed.manybody_basis import Basis

    comm = MPI.COMM_WORLD
    dets = [_det(occ) for occ in ([1, 2], [0, 3], [0, 1], [0, 2])]
    basis = Basis(
        impurity_orbitals={0: [[0, 1]]},
        bath_states=({0: [[2, 3]]}, {0: [[4, 5]]}),
        initial_basis=dets,
        comm=comm,
        verbose=False,
    )
    if comm.rank == 0:
        bell = ManyBodyState({_det([1, 2]): math.sqrt(0.5), _det([0, 3]): -math.sqrt(0.5)})
        mixed_valence = ManyBodyState({_det([0, 1]): math.sqrt(0.5), _det([0, 2]): math.sqrt(0.5)})
    else:
        bell = ManyBodyState({})
        mixed_valence = ManyBodyState({})
    psis = basis.redistribute_psis([bell, mixed_valence])

    ent = compute_entanglement_entropy(basis, psis, np.array([0.0, 1.0]), tau=0.5)
    assert ent["per_state_entropy"] == pytest.approx([math.log(2), math.log(2)])
    assert ent["per_state_norm"] == pytest.approx([1.0, 1.0])
    assert ent["n_imp_sectors"] == [1, 2]
    assert ent["sector_weights"][1] == {1: pytest.approx(0.5), 2: pytest.approx(0.5)}


def test_participation_helpers_on_known_distributions():
    # Uniform over k entries -> N_eff = k, entropy = ln k.
    k = 4
    uniform = dict.fromkeys(range(k), 1.0)
    part = _participation(uniform, total=float(k))
    assert part["effective_number"] == pytest.approx(k)
    assert part["entropy"] == pytest.approx(math.log(k))
    # Delta distribution -> N_eff = 1, entropy = 0.
    delta = _participation({0: 1.0}, total=1.0)
    assert delta["effective_number"] == pytest.approx(1.0)
    assert delta["entropy"] == pytest.approx(0.0)
    # Determinant scalar-partial form agrees with the dict form (weights 0.3,0.2,0.5).
    w = np.array([0.3, 0.2, 0.5])
    det = _det_participation(w.sum(), np.sum(w**2), np.sum(w * np.log(w)))
    ref = _participation(dict(enumerate(w)), total=float(w.sum()))
    assert det["effective_number"] == pytest.approx(ref["effective_number"])
    assert det["entropy"] == pytest.approx(ref["entropy"])


def test_sorted_weight_rows_cutoff_collapses_small_configs():
    weight_map = {(2, 0, 0): 0.6, (1, 1, 0): 0.39, (0, 2, 0): 0.01}
    rows, remaining = _sorted_weight_rows(weight_map, total=1.0, weight_cutoff=0.05)
    assert [r["config"] for r in rows] == [(2, 0, 0), (1, 1, 0)]
    assert remaining is not None
    assert remaining["count"] == 1
    assert remaining["weight"] == pytest.approx(0.01)


def test_marginal_axis_selects_channel():
    thermal_config = {(2, 1, 1): 0.3, (1, 2, 2): 0.2, (1, 1, 1): 0.5}
    con = _marginal(thermal_config, axis=2)
    assert dict(con["distribution"]) == {1: pytest.approx(0.8), 2: pytest.approx(0.2)}
