"""``ImpurityModel.from_h0_text``: the flat-index path from a ``.h0`` file to a model."""

import json
from pathlib import Path

import numpy as np
import pytest

from impurityModel.ed import h0_format
from impurityModel.ed.model import ImpurityModel, atomic_u4

FIXTURES = Path(__file__).parent
INDEX_FIXTURE = FIXTURES / "golden_h0_v1_index.h0"

D_SHELL_SLATER = [7.5, 0.0, 9.9, 0.0, 6.6]


def _d_shell_file(tmp_path, name="d_shell.h0", **header_extra):
    """A 10-impurity-orbital (d) model with 4 bath orbitals, written through the writer.

    Every impurity orbital couples to every bath orbital, as in a real hybridization fit --
    a bath that reaches only part of the impurity block would leave the trailing sub-block
    narrow-banded and (correctly) trip the layout check in the legacy reader.
    """
    n_imp, n_bath = 10, 4
    n = n_imp + n_bath
    h = np.zeros((n, n), dtype=complex)
    h[:n_imp, :n_imp] = np.diag(np.linspace(-1.5, -1.0, n_imp)).astype(complex)
    for b in range(n_imp, n):
        h[b, b] = -3.0 + 0.5 * (b - n_imp)
        for i in range(n_imp):
            h[b, i] = h[i, b] = 0.4 - 0.01 * i

    out = tmp_path / name
    h0_format.write_h0_file(out, h, impurity_orbitals={2: list(range(n_imp))}, **header_extra)
    return out, h


def test_from_h0_text_matches_from_solver_matrix(tmp_path):
    """The file path and the in-memory path must build the same model."""
    path, h = _d_shell_file(tmp_path)

    from_file = ImpurityModel.from_h0_text(path, l=2, slater=D_SHELL_SLATER)
    from_memory = ImpurityModel.from_solver_matrix(
        h, 10, u4=atomic_u4(2, D_SHELL_SLATER), rot_to_spherical=np.eye(10, dtype=complex)
    )

    assert from_file.h0 == from_memory.h0
    assert from_file.u4 == from_memory.u4
    assert from_file.impurity_orbitals == from_memory.impurity_orbitals
    assert from_file.n_spin_orbitals == from_memory.n_spin_orbitals == 14


def test_from_h0_text_takes_the_interaction_from_the_header(tmp_path):
    path, _ = _d_shell_file(tmp_path, interaction={"kind": "slater", "l": 2, "F": D_SHELL_SLATER}, impurity_l=2)

    from_header = ImpurityModel.from_h0_text(path)
    from_arguments = ImpurityModel.from_h0_text(path, l=2, slater=D_SHELL_SLATER)

    assert from_header.u4 is not None
    assert from_header.u4 == from_arguments.u4


def test_from_h0_text_without_an_interaction_raises(tmp_path):
    """u4=None runs a silently non-interacting solve, so it must not be the default."""
    path, _ = _d_shell_file(tmp_path)

    with pytest.raises(ValueError, match="non-interacting"):
        ImpurityModel.from_h0_text(path)

    model = ImpurityModel.from_h0_text(path, allow_noninteracting=True)
    assert model.u4 is None


def test_from_h0_text_rejects_a_shell_that_contradicts_the_file(tmp_path):
    path, _ = _d_shell_file(tmp_path)
    with pytest.raises(ValueError, match="implies"):
        ImpurityModel.from_h0_text(path, l=3, slater=D_SHELL_SLATER)


@pytest.mark.parametrize("kwargs", [{"xi": 0.1}, {"h_field": (0.0, 0.0, 1e-4)}])
def test_from_h0_text_refuses_label_space_dressing(tmp_path, kwargs):
    """SOC and the magnetic field are built in (l,s,m); guessing the spin order is worse."""
    path, _ = _d_shell_file(tmp_path)
    with pytest.raises(NotImplementedError, match="doc/h0_file_format.md"):
        ImpurityModel.from_h0_text(path, l=2, slater=D_SHELL_SLATER, **kwargs)


def test_from_h0_text_carries_the_bath_split_and_rotation(tmp_path):
    model = ImpurityModel.from_h0_text(INDEX_FIXTURE, allow_noninteracting=True)

    assert model.impurity_orbitals == {0: [0, 1, 2, 3]}
    assert model.bath_states == ({0: [4, 5]}, {0: [6, 7]})
    assert model.n_spin_orbitals == 8
    np.testing.assert_array_equal(model.rot_to_spherical, np.eye(4, dtype=complex))


def test_from_h0_text_reads_the_legacy_format_when_given_n_imp(tmp_path):
    path, h = _d_shell_file(tmp_path)
    legacy = tmp_path / "legacy_h0_op.dict"
    legacy.write_text(
        "\n".join(
            f"{i} {j} {complex(h[i, j]).real!r} {complex(h[i, j]).imag!r}"
            for i in range(h.shape[0])
            for j in range(h.shape[0])
            if h[i, j] != 0
        )
        + "\n"
    )

    with pytest.deprecated_call():
        model = ImpurityModel.from_h0_text(legacy, l=2, slater=D_SHELL_SLATER, n_impurity_orbitals=10)
    assert model.n_spin_orbitals == 14
    assert model.impurity_orbitals == {0: list(range(10))}


def test_from_h0_text_legacy_without_n_imp_raises(tmp_path):
    legacy = tmp_path / "legacy_h0_op.dict"
    legacy.write_text("0 0 1.0 0.0\n")
    with pytest.raises(ValueError, match="n_impurity_orbitals"):
        ImpurityModel.from_h0_text(legacy)


def test_a_renamed_legacy_file_is_still_routed_to_the_legacy_reader(tmp_path):
    """Extension is not the discriminator; the magic line is."""
    disguised = tmp_path / "actually_legacy.h0"
    disguised.write_text("  0   0 -4.125093021114729  0.000000000000000\n")
    assert not h0_format.is_h0_format(disguised)

    with pytest.raises(ValueError, match="n_impurity_orbitals"):
        ImpurityModel.from_h0_text(disguised)


def test_header_impurity_group_key_survives_json(tmp_path):
    """JSON object keys are strings; the group label must come back as the shell's l."""
    path, _ = _d_shell_file(tmp_path)
    assert json.loads(path.read_text().splitlines()[1])["impurity_orbitals"] == {"2": list(range(10))}
    assert h0_format.read_h0_file(path).impurity_orbitals == {2: tuple(range(10))}
