"""``SeedWindow``: a bath-less shell's *seed* occupation is a window, not a pin (review J1, step 11.0b)."""

import pytest

from impurityModel.ed.basis_generation import DEEP_SEED_WIDTH, SeedWindow, generate_initial_basis
from impurityModel.test.basis.test_sector_key import N_ORB, _model

NOMINAL = {0: 2, 1: 2}


def _seed(frozen, nominal=NOMINAL):
    _h, impurity_orbitals, bath_states = _model()
    states, _n = generate_initial_basis(
        impurity_orbitals=impurity_orbitals,
        bath_states=bath_states,
        delta_valence_occ=dict.fromkeys(nominal, 0),
        delta_conduction_occ=dict.fromkeys(nominal, 0),
        delta_impurity_occ=dict.fromkeys(nominal, 0),
        nominal_impurity_occ=nominal,
        n_bytes=(N_ORB + 7) // 8,
        verbose=False,
        frozen_occupations=frozen,
    )
    return {bytes(s.to_bytearray()) for s in states}


def _occ(seed, orbs):
    """Occupations of ``orbs`` over a seed (orbital -> bit is MSB-first within a byte)."""
    out = set()
    for det in seed:
        out.add(sum((det[o // 8] >> (7 - o % 8)) & 1 for o in orbs))
    return out


def test_window_spans_width_and_keeps_total_electrons():
    nominal = {0: 1, 1: 1}  # total impurity charge 2
    narrow = _seed(SeedWindow({0}, width=0), nominal)
    wide = _seed(SeedWindow({0}, width=1), nominal)
    assert _occ(narrow, (0, 1)) == {1}
    assert _occ(wide, (0, 1)) == {0, 1, 2}
    assert narrow < wide
    # Charge is conserved: every determinant holds the same number of electrons.
    assert len({sum(bin(b).count("1") for b in det) for det in wide}) == 1


def test_width_zero_is_a_total_pin():
    seed = _seed(SeedWindow({0}, width=0))
    assert _occ(seed, (0, 1)) == {2}


def test_window_contains_the_pinned_seed():
    pinned = _seed({0})
    window = _seed(SeedWindow({0}, width=1))
    assert pinned <= window <= _seed(None)


def test_plain_set_path_is_unchanged():
    assert _seed({0}) == _seed(set({0}))
    assert _occ(_seed({0}), (0, 1)) == {2}


def test_default_width():
    assert SeedWindow({0}).width == DEEP_SEED_WIDTH == 1


def _deep_p_plus_d_sb(nominal=None):
    from impurityModel.ed.lie_algebra import tensors_to_operator
    from impurityModel.ed.solver_basis import prepare_solver_basis

    h = [[0.0] * 20 for _ in range(20)]
    import numpy as np

    h = np.zeros((20, 20), dtype=complex)
    for o in range(6):
        h[o, o] = -60.0
    for o in range(6, 16):
        h[o, o] = 1.0
    for b, e in zip(range(16, 20), (-2.0, -2.0, 2.0, 2.0)):
        h[b, b] = e
    for o, b in ((6, 16), (11, 17), (7, 18), (12, 19)):
        h[o, b] = h[b, o] = 0.3
    return prepare_solver_basis(
        tensors_to_operator(h).to_dict(),
        {},
        {},
        {0: list(range(16))},
        nominal or {0: 6, 1: 2},
        rot_to_spherical=np.eye(16, dtype=complex),
        verbosity=0,
        shell_ls=(1, 2),
    )


def test_prepare_solver_basis_names_the_bath_less_groups_as_the_seed_window():
    sb = _deep_p_plus_d_sb()
    assert isinstance(sb.seed_window, SeedWindow) and sb.seed_window.width == DEEP_SEED_WIDTH
    for k in sb.impurity_orbitals:
        orb = next(o for block in sb.impurity_orbitals[k] for o in block)
        assert (k in sb.seed_window) == (sb.impurity_shell_index[orb] == 0)


def test_seed_window_bounds_the_seed_of_the_deep_shell():
    sb = _deep_p_plus_d_sb()

    def seed(frozen):
        states, _ = generate_initial_basis(
            sb.impurity_orbitals,
            sb.bath_states,
            dict.fromkeys(sb.impurity_orbitals, 0),
            dict.fromkeys(sb.impurity_orbitals, 0),
            dict.fromkeys(sb.impurity_orbitals, 0),
            sb.nominal_occ,
            (20 + 7) // 8,
            False,
            frozen_occupations=frozen,
        )
        return states

    free, windowed = seed(None), seed(sb.seed_window)
    assert len(windowed) < len(free) and len(windowed) < 1000
    for det in windowed:
        n_p = sum(1 for o in range(6) if (bytes(det.to_bytearray())[o // 8] >> (7 - o % 8)) & 1)
        assert abs(n_p - 6) <= DEEP_SEED_WIDTH


def test_calc_gs_accepts_the_seed_window_and_returns_the_full_deep_shell(tmp_path):
    """A plumbing check, not a discriminating one: the -60 level fills the p shell whatever the window does
    (review J1 measured the same with nothing frozen). What the window changes is the seed, which
    ``test_seed_window_bounds_the_seed_of_the_deep_shell`` asserts."""
    import numpy as np

    from impurityModel.ed.groundstate import calc_gs

    sb = _deep_p_plus_d_sb()
    basis_information = {
        "impurity_orbitals": sb.impurity_orbitals,
        "bath_states": sb.bath_states,
        "N0": sb.nominal_occ,
        "tau": 0.002,
        "chain_restrict": False,
        "dense_cutoff": 1000,
        "comm": None,
        "spin_layout": sb.spin_layout,
        "truncation_threshold": np.inf,
        "frozen_occupations": sb.seed_window,
    }
    _psis, es, _basis, rho, _info = calc_gs(
        sb.h,
        basis_information,
        sb.block_structure,
        sb.rot_to_spherical,
        0,
        stats_path=str(tmp_path / "gs.json"),
        shell_ls=sb.shell_ls,
    )
    assert np.isfinite(es[0])
    n_p = np.trace(np.asarray(rho)[:6, :6]).real
    assert n_p == pytest.approx(6.0, abs=1e-6)


def test_hf_that_refills_the_deep_shell_is_reported(tmp_path, capsys):
    """Nominal p5 d3 against an energy that fills p6: the window is centred on HF's count, and says so."""
    import numpy as np

    from impurityModel.ed.groundstate import calc_gs

    sb = _deep_p_plus_d_sb({0: 5, 1: 3})
    assert sb.seed_window is not None
    basis_information = {
        "impurity_orbitals": sb.impurity_orbitals,
        "bath_states": sb.bath_states,
        "N0": sb.nominal_occ,
        "tau": 0.002,
        "chain_restrict": False,
        "dense_cutoff": 1000,
        "comm": None,
        "spin_layout": sb.spin_layout,
        "truncation_threshold": np.inf,
        "frozen_occupations": sb.seed_window,
    }
    calc_gs(
        sb.h, basis_information, sb.block_structure, sb.rot_to_spherical, 0,
        stats_path=str(tmp_path / "gs.json"), shell_ls=sb.shell_ls,
    )  # fmt: skip
    assert "Hartree-Fock fills the bath-less shell(s)" in capsys.readouterr().out


def test_hard_frozen_is_empty_for_a_window_and_the_set_itself_otherwise():
    from impurityModel.ed.groundstate import _hard_frozen

    assert _hard_frozen(SeedWindow({0, 3})) == set()
    assert _hard_frozen({0, 3}) == {0, 3}
    assert _hard_frozen(None) == set()


@pytest.mark.parametrize("as_window", [True, False])
def test_hartree_fock_is_pinned_by_a_plain_freeze_and_not_by_a_window(tmp_path, monkeypatch, as_window):
    import numpy as np

    from impurityModel.ed import groundstate

    sb = _deep_p_plus_d_sb()
    groups = set(sb.seed_window)
    frozen = sb.seed_window if as_window else set(groups)
    seen = []
    real = groundstate.hartree_fock_seed_occupation

    def spy(*args, **kwargs):
        seen.append(set(kwargs["frozen_occupations"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(groundstate, "hartree_fock_seed_occupation", spy)
    basis_information = {
        "impurity_orbitals": sb.impurity_orbitals,
        "bath_states": sb.bath_states,
        "N0": sb.nominal_occ,
        "tau": 0.002,
        "chain_restrict": False,
        "dense_cutoff": 1000,
        "comm": None,
        "spin_layout": sb.spin_layout,
        "truncation_threshold": np.inf,
        "frozen_occupations": frozen,
    }
    groundstate.calc_gs(
        sb.h, basis_information, sb.block_structure, sb.rot_to_spherical, 0,
        stats_path=str(tmp_path / "gs.json"), shell_ls=sb.shell_ls,
    )  # fmt: skip
    assert seen and all(s == (set() if as_window else groups) for s in seen), seen
