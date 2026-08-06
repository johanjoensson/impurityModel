"""``get_spectra.build_spectra_model`` / ``ImpurityModel.from_shells``: the multi-shell (2p
core + 3d correlated) spectra model, built from either a labelled or a flat ``.h0`` file.
"""

import pickle
from collections import OrderedDict
from unittest.mock import patch

import numpy as np
import pytest

from impurityModel.ed import h0_format
from impurityModel.ed.get_spectra import build_spectra_model
from impurityModel.ed.model import ImpurityModel, load_model
from impurityModel.ed.operator_algebra import i2c

FDD = [7.5, 0.0, 9.9, 0.0, 6.6]
FPP = [0.0, 0.0, 0.0]
FPD = [8.9, 0.0, 6.8]
GPD = [0.0, 5.0, 0.0, 3.0]
XI_2P, XI_3D = 11.629, 0.096
CTC = 1.5
H_FIELD = (0.0, 0.0, 1e-4)


def _labelled_d_shell(tmp_path, n_bath, *, name="d_shell.pickle"):
    """A d-only (l=2) labelled h0 pickle, matching every real spectra workload's shape."""
    l = 2
    n_imp = 2 * (2 * l + 1)
    n = n_imp + n_bath
    rng = np.random.default_rng(0)
    h = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    h = 0.5 * (h + h.conj().T)

    shell = OrderedDict({l: n_bath})
    labelled = {
        ((i2c(shell, i), "c"), (i2c(shell, j), "a")): complex(h[i, j])
        for i in range(n)
        for j in range(n)
        if h[i, j] != 0
    }
    path = tmp_path / name
    with open(path, "wb") as f:
        pickle.dump(labelled, f)
    return path, h


def _flat_d_shell(tmp_path, n_bath, *, name="d_shell.h0", **header_extra):
    l = 2
    n_imp = 2 * (2 * l + 1)
    n = n_imp + n_bath
    rng = np.random.default_rng(0)
    h = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    h = 0.5 * (h + h.conj().T)

    header_extra.setdefault("basis", "spherical")
    header_extra.setdefault("spin_ordering", "down_first")
    header_extra.setdefault("energy_reference", "fermi")
    header_extra.setdefault("impurity_l", l)
    # This is random noise, not physical SOC -- declare that truthfully so the
    # contains_soc-absent-is-unknown guard (model.py) doesn't refuse these fixtures.
    header_extra.setdefault("contains_soc", False)
    path = tmp_path / name
    h0_format.write_h0_file(path, h, impurity_orbitals={0: list(range(n_imp))}, **header_extra)
    return path, h


def test_from_shells_layout_matches_c2i_order(tmp_path):
    """Regression for the interleaved-offset bug: with a non-zero core-shell bath count, the
    old per-shell interleaved loop put the 3d impurity block at 12-21; c2i puts it at 6-15.
    """
    with patch("impurityModel.ed.model.get_hamiltonian_operator", return_value={}):
        model = ImpurityModel.from_shells(
            "dummy.pickle",
            OrderedDict({1: 6, 2: 10}),
            OrderedDict({1: 6, 2: 10}),
            OrderedDict({1: 6, 2: 8}),
            (FDD, FPP, FPD, GPD),
            (XI_2P, XI_3D),
            CTC,
            verbose=False,
        )

    assert model.impurity_orbitals == {1: [[0, 1, 2, 3, 4, 5]], 2: [[6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]}
    valence, conduction = model.bath_states
    assert valence == {1: [[16, 17, 18, 19, 20, 21]], 2: [[22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]}
    assert conduction == {1: [[]], 2: [[]]}
    assert model.n_spin_orbitals == 32


def test_from_shells_rejects_a_non_2p3d_shell():
    with pytest.raises(ValueError, match="2p/3d"):
        ImpurityModel.from_shells(
            "dummy.pickle",
            OrderedDict({3: 6}),
            OrderedDict({3: 6}),
            OrderedDict({3: 8}),
            (FDD, FPP, FPD, GPD),
            (XI_2P, XI_3D),
            CTC,
        )


def test_from_shells_rejects_mismatched_shells_and_val_shells():
    with pytest.raises(ValueError, match="same angular momenta"):
        ImpurityModel.from_shells(
            "dummy.pickle",
            OrderedDict({1: 0, 2: 10}),
            OrderedDict({2: 10}),
            OrderedDict({1: 6, 2: 8}),
            (FDD, FPP, FPD, GPD),
            (XI_2P, XI_3D),
            CTC,
        )


def test_build_spectra_model_equals_from_shells(tmp_path):
    """The CLI-facing wrapper must build exactly the same model as the classmethod."""
    path, _ = _labelled_d_shell(tmp_path, n_bath=4)

    wrapped = build_spectra_model(
        str(path), (1, 2), (0, 4), (0, 4), (6, 8), FDD, FPP, FPD, GPD, XI_2P, XI_3D, CTC, H_FIELD, verbose=False
    )
    direct = ImpurityModel.from_shells(
        str(path),
        OrderedDict({1: 0, 2: 4}),
        OrderedDict({1: 0, 2: 4}),
        OrderedDict({1: 6, 2: 8}),
        (FDD, FPP, FPD, GPD),
        (XI_2P, XI_3D),
        CTC,
        h_field=H_FIELD,
        verbose=False,
    )

    assert wrapped.h0 == direct.h0
    assert wrapped.impurity_orbitals == direct.impurity_orbitals
    assert wrapped.bath_states == direct.bath_states


def test_labelled_and_flat_h0_build_the_same_spectra_model(tmp_path):
    """The equivalence proof: a labelled pickle and a flat .h0 of the *same* physical
    Hamiltonian must produce identical interacting operators once SOC/field/Coulomb/DC are
    folded in. A wrong spin ordering or a wrong c2i mapping would break this even though both
    files parse cleanly.
    """
    n_bath = 4
    # Both helpers seed the same RNG and build the same shape, so they hold the same matrix --
    # verified below -- letting the comparison isolate the reader rather than the data.
    labelled_path, h_labelled = _labelled_d_shell(tmp_path, n_bath)
    flat_path, h_flat = _flat_d_shell(tmp_path, n_bath)
    np.testing.assert_allclose(h_labelled, h_flat)

    shells = OrderedDict({1: 0, 2: n_bath})
    kwargs = dict(
        shells=shells,
        val_shells=OrderedDict({1: 0, 2: n_bath}),
        n0imps=OrderedDict({1: 6, 2: 8}),
        slater_condon=(FDD, FPP, FPD, GPD),
        socs=(XI_2P, XI_3D),
        charge_transfer_correction=CTC,
        h_field=H_FIELD,
        verbose=False,
    )

    labelled_model = load_model(str(labelled_path), **kwargs)
    flat_model = load_model(str(flat_path), **kwargs)

    assert set(labelled_model.h0) == set(flat_model.h0)
    for key in labelled_model.h0:
        assert flat_model.h0[key] == pytest.approx(labelled_model.h0[key], abs=1e-10)


def test_from_shells_flat_rejects_a_non_spherical_header(tmp_path):
    path, _ = _flat_d_shell(tmp_path, n_bath=4, basis="cubic")
    with pytest.raises(RuntimeError, match="basis"):
        load_model(
            str(path),
            shells=OrderedDict({1: 0, 2: 4}),
            val_shells=OrderedDict({1: 0, 2: 4}),
            n0imps=OrderedDict({1: 6, 2: 8}),
            slater_condon=(FDD, FPP, FPD, GPD),
            socs=(XI_2P, XI_3D),
            charge_transfer_correction=CTC,
            verbose=False,
        )


def test_from_shells_flat_rejects_a_bath_count_that_disagrees_with_the_header(tmp_path):
    path, _ = _flat_d_shell(tmp_path, n_bath=4)
    with pytest.raises(RuntimeError, match="bath orbitals"):
        load_model(
            str(path),
            shells=OrderedDict({1: 0, 2: 10}),
            val_shells=OrderedDict({1: 0, 2: 10}),
            n0imps=OrderedDict({1: 6, 2: 8}),
            slater_condon=(FDD, FPP, FPD, GPD),
            socs=(XI_2P, XI_3D),
            charge_transfer_correction=CTC,
            verbose=False,
        )


def test_from_shells_flat_rejects_an_impurity_l_the_shells_dict_does_not_have(tmp_path):
    path, _ = _flat_d_shell(tmp_path, n_bath=4, impurity_l=3)
    with pytest.raises(RuntimeError, match="impurity_l"):
        load_model(
            str(path),
            shells=OrderedDict({1: 0, 2: 4}),
            val_shells=OrderedDict({1: 0, 2: 4}),
            n0imps=OrderedDict({1: 6, 2: 8}),
            slater_condon=(FDD, FPP, FPD, GPD),
            socs=(XI_2P, XI_3D),
            charge_transfer_correction=CTC,
            verbose=False,
        )


def test_from_shells_rejects_contains_soc_with_a_nonzero_xi_3d(tmp_path):
    path, _ = _flat_d_shell(tmp_path, n_bath=4, contains_soc=True)
    with pytest.raises(ValueError, match="double-count"):
        load_model(
            str(path),
            shells=OrderedDict({1: 0, 2: 4}),
            val_shells=OrderedDict({1: 0, 2: 4}),
            n0imps=OrderedDict({1: 6, 2: 8}),
            slater_condon=(FDD, FPP, FPD, GPD),
            socs=(XI_2P, XI_3D),
            charge_transfer_correction=CTC,
            verbose=False,
        )


def test_from_shells_rejects_an_absent_contains_soc_with_a_nonzero_xi_3d(tmp_path):
    """Omitting the key is *unknown*, not *no* -- refused exactly like an explicit ``true``.

    This is the regression case: for a while ``build_h0`` never wrote ``contains_soc`` at all,
    which silently disabled this guard for every file it produced (see
    doc/h0_file_format.md, Interaction).
    """
    path, _ = _flat_d_shell(tmp_path, n_bath=4, contains_soc=None)
    with pytest.raises(ValueError, match="does not declare 'contains_soc'"):
        load_model(
            str(path),
            shells=OrderedDict({1: 0, 2: 4}),
            val_shells=OrderedDict({1: 0, 2: 4}),
            n0imps=OrderedDict({1: 6, 2: 8}),
            slater_condon=(FDD, FPP, FPD, GPD),
            socs=(XI_2P, XI_3D),
            charge_transfer_correction=CTC,
            verbose=False,
        )


def test_from_shells_allows_contains_soc_with_a_zero_xi_3d(tmp_path):
    """SOC already baked into the file is fine as long as the CLI isn't asking to add more."""
    path, _ = _flat_d_shell(tmp_path, n_bath=4, contains_soc=True)
    with patch("impurityModel.ed.model.get_hamiltonian_operator", return_value={}):
        load_model(
            str(path),
            shells=OrderedDict({1: 0, 2: 4}),
            val_shells=OrderedDict({1: 0, 2: 4}),
            n0imps=OrderedDict({1: 6, 2: 8}),
            slater_condon=(FDD, FPP, FPD, GPD),
            socs=(XI_2P, 0.0),
            charge_transfer_correction=CTC,
            verbose=False,
        )


def test_from_shells_honors_a_noncontiguous_header_valence_split(tmp_path):
    """The header's valence_bath/conduction_bath need not be a contiguous prefix of the bath
    block; from_shells must place them at the header's own positions, not the first n_val.
    """
    n_bath = 4
    n_imp = 10
    # File-local bath indices n_imp..n_imp+3 == 10,11,12,13; declare the *odd* ones valence.
    path, _ = _flat_d_shell(
        tmp_path,
        n_bath=n_bath,
        valence_bath=[11, 13],
        conduction_bath=[10, 12],
    )
    with patch("impurityModel.ed.model.get_hamiltonian_operator", return_value={}):
        model = ImpurityModel.from_shells(
            str(path),
            OrderedDict({1: 0, 2: n_bath}),
            OrderedDict({1: 0, 2: 2}),
            OrderedDict({1: 6, 2: 8}),
            (FDD, FPP, FPD, GPD),
            (XI_2P, XI_3D),
            CTC,
            verbose=False,
        )

    valence, conduction = model.bath_states
    # bath block starts right after the 2p (0-5) + 3d (6-15) impurity blocks, at offset 16.
    assert valence[2] == [[16 + (11 - n_imp), 16 + (13 - n_imp)]]
    assert conduction[2] == [[16 + (10 - n_imp), 16 + (12 - n_imp)]]


def test_from_shells_rejects_a_header_valence_count_mismatch(tmp_path):
    path, _ = _flat_d_shell(tmp_path, n_bath=4, valence_bath=[10, 11], conduction_bath=[12, 13])
    with pytest.raises(ValueError, match="valence_bath"):
        load_model(
            str(path),
            shells=OrderedDict({1: 0, 2: 4}),
            # val_shells says 3 valence baths; the header only lists 2.
            val_shells=OrderedDict({1: 0, 2: 3}),
            n0imps=OrderedDict({1: 6, 2: 8}),
            slater_condon=(FDD, FPP, FPD, GPD),
            socs=(XI_2P, XI_3D),
            charge_transfer_correction=CTC,
            verbose=False,
        )
