"""A K-edge (1s -> 2p) spectroscopy input, from TOML file to assembled Hamiltonian.

The unit tests elsewhere check each generalised piece against its own reference. This one
checks that they compose: a file describing an edge the solver could not run a week ago is
parsed, passes the capability gate, and comes out the far end as an ``ImpurityModel`` whose
orbital layout is the one its shells imply.

``l_core = 0`` is the deliberate choice. It is the case that a length-derived core angular
momentum silently gets wrong -- an omitted core array used to be replaced by a ``(0, 0, 0)``
placeholder, and every array length is an assertion about ``l_core``.
"""

from pathlib import Path

import numpy as np
import pytest

from impurityModel.ed import h0_format
from impurityModel.inputformat.build import build
from impurityModel.inputformat.reader import InputError, load_input

K_EDGE = """
[format]
version = [1, 0]

[units]
energy = "eV"

[hamiltonian.file]
path = "{h0}"

[[shell]]
l = 0
role = "core"
nominal_occupation = 2
soc = 0.0

[[shell]]
l = 1
role = "valence"
n_bath = 4
n_valence_bath = 4
nominal_occupation = 3
soc = 0.05

[interaction.slater]
F_vv = [6.0, 0.0, 4.0]
F_cc = [0.0]
F_cv = [5.0]
G_cv = [0.0, 2.0]

[double_counting.mlft]
c = 1.0

[spectroscopy]

[spectroscopy.xas]
enabled = true

[spectroscopy.rixs]
enabled = false
"""


@pytest.fixture
def p_shell_h0(tmp_path):
    """A flat ``.h0`` for an l=1 valence shell with four bath orbitals."""
    l, n_bath = 1, 4
    n_imp = 2 * (2 * l + 1)
    n = n_imp + n_bath
    rng = np.random.default_rng(7)
    h = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    h = 0.5 * (h + h.conj().T)
    # All four bath levels below E_F, so the deduced valence/conduction split is 4/0 and the
    # counts written in the file are the ones the reader cross-checks against.
    for o in range(n_imp, n):
        h[o, o] = -1.0 - 0.1 * (o - n_imp)
    path = tmp_path / "p_shell.h0"
    h0_format.write_h0_file(
        path,
        h,
        impurity_orbitals={0: list(range(n_imp))},
        basis="spherical",
        spin_ordering="down_first",
        energy_reference="fermi",
        impurity_l=l,
        contains_soc=False,
    )
    return path


@pytest.fixture
def write(tmp_path):
    def _write(text, name="k_edge.toml"):
        path = tmp_path / name
        path.write_text(text)
        return path

    return _write


def test_a_k_edge_input_builds_a_model(p_shell_h0, write):
    """The whole chain, on an edge the capability gate used to refuse."""
    built = build(load_input(write(K_EDGE.format(h0=p_shell_h0.name))))

    # 1s impurity block (2 spin-orbitals), then 2p (6), then the 2p bath (4).
    assert built.model.impurity_orbitals == {0: [[0, 1]], 1: [[2, 3, 4, 5, 6, 7]]}
    valence, conduction = built.model.bath_states
    assert valence == {0: [[]], 1: [[8, 9, 10, 11]]}
    assert conduction == {0: [[]], 1: [[]]}
    assert built.model.n_spin_orbitals == 12
    assert built.model.h0


def test_the_k_edge_hamiltonian_lives_only_on_its_own_shells(p_shell_h0, write):
    """No term may reach an orbital the declared shells do not have."""
    built = build(load_input(write(K_EDGE.format(h0=p_shell_h0.name))))
    indices = {i for process in built.model.h0 for i, _ in process}
    assert indices
    assert max(indices) < built.model.n_spin_orbitals


def test_a_k_edge_rejects_the_l23_slater_array_lengths(p_shell_h0, write):
    """The lengths follow the declared l, so an L2,3-shaped file is caught, not accepted.

    This is the failure that a zero-filled placeholder used to hide: `F_cv = [8.9, 0, 6.8]` is
    the right length for `l_core = 1` and the wrong one for the `l_core = 0` declared here.
    """
    text = K_EDGE.format(h0=p_shell_h0.name).replace("F_cv = [5.0]", "F_cv = [8.9, 0, 6.8]")
    with pytest.raises(InputError, match="F_cv"):
        build(load_input(write(text)))


def test_a_crystal_field_shell_must_use_its_own_irreps(write):
    """The keys follow the valence shell's O_h levels, so d-shaped keys on a p shell are caught.

    The crystal field is no longer d-only -- it follows whatever level structure the valence
    shell has. What cannot be inherited is the *parametrisation*: an l=1 shell is a single
    t_1u level, so it has no 10Dq and no e_g/t_2g bath rows, and a file that supplies them is
    describing a different shell than the one it declared.
    """
    text = K_EDGE.replace(
        '[hamiltonian.file]\npath = "{h0}"',
        "[hamiltonian.crystal_field]\n"
        "e_imp = -1.0\ne_deltaO_imp = 0.6\n"
        "e_val_eg = -4.4\ne_val_t2g = -6.5\ne_con_eg = 3.0\ne_con_t2g = 2.0\n"
        "v_val_eg = 1.9\nv_val_t2g = 1.4\nv_con_eg = 0.6\nv_con_t2g = 0.4",
    ).replace("n_bath = 4\nn_valence_bath = 4", "n_bath = 6\nn_valence_bath = 6")
    with pytest.raises(InputError) as excinfo:
        build(load_input(write(text)))
    message = str(excinfo.value)
    assert "l=1 valence shell" in message
    assert "t1u" in message
    # Both halves of the mismatch are named, not just the first one hit.
    assert "Missing: e_val_t1u" in message
    assert "e_deltaO_imp" in message


def test_a_p_shell_crystal_field_builds_with_its_own_keys(write):
    """The same p shell, described in its own irrep: one t_1u level, no splitting parameter."""
    text = K_EDGE.replace(
        '[hamiltonian.file]\npath = "{h0}"',
        "[hamiltonian.crystal_field]\n" "e_imp = -1.0\n" "e_val_t1u = -4.4\nv_val_t1u = 1.9",
    ).replace("n_bath = 4\nn_valence_bath = 4", "n_bath = 6\nn_valence_bath = 6")
    built = build(load_input(write(text)))
    assert built.model.impurity_orbitals == {0: [[0, 1]], 1: [[2, 3, 4, 5, 6, 7]]}
    assert built.model.h0


K_EDGE_RUN = """
[format]
version = [1, 0]

[units]
energy = "eV"

[hamiltonian.file]
path = "{h0}"

[[shell]]
l = 0
role = "core"
nominal_occupation = 2
soc = 0.0

[[shell]]
l = 1
role = "valence"
n_bath = 2
n_valence_bath = 2
nominal_occupation = 3
soc = 0.05

[interaction.slater]
F_vv = [6.0, 0.0, 4.0]
F_cc = [0.0]
F_cv = [5.0]
G_cv = [0.0, 2.0]

[double_counting.mlft]
c = 1.0

[spectroscopy]
w = {{min = -8.0, max = 8.0, n = 61}}
w_loss = {{min = -1.0, max = 8.0, n = 41}}

[spectroscopy.pes]
enabled = true

[spectroscopy.xps]
enabled = true

[spectroscopy.xas]
enabled = true

[spectroscopy.rixs]
enabled = true
w_in = {{min = 0.0, max = 20.0, n = 5}}
"""


@pytest.fixture
def small_p_shell_h0(tmp_path):
    """The same ``.h0``, with two bath orbitals: small enough to solve inside a test.

    Written per rank into that rank's own ``tmp_path``. The construction is deterministic, so
    every rank builds the identical model without needing a shared file or a barrier.
    """
    l, n_bath = 1, 2
    n_imp = 2 * (2 * l + 1)
    n = n_imp + n_bath
    rng = np.random.default_rng(7)
    h = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    h = 0.5 * (h + h.conj().T)
    for o in range(n_imp, n):
        h[o, o] = -1.0 - 0.1 * (o - n_imp)
    path = tmp_path / "p_shell_small.h0"
    h0_format.write_h0_file(
        path,
        h,
        impurity_orbitals={0: list(range(n_imp))},
        basis="spherical",
        spin_ordering="down_first",
        energy_reference="fermi",
        impurity_l=l,
        contains_soc=False,
    )
    return path


def test_a_k_edge_actually_solves_and_produces_weight(small_p_shell_h0, write, tmp_path):
    """PES, XPS, XAS and RIXS at 1s -> 2p, run for real.

    The other tests here stop at ``build``; the assemblers below it are what a non-L2,3 edge
    reaches for the first time -- ``slater_condon_Uop(lv=1, lc=0)`` with its length-1 ``F_cc``
    and ``F_cv``, ``dc_MLFT``'s filled-core gate at ``2*(2*l_c+1) = 2``, ``getSOCop(l=0)``,
    the ``dN`` windows keyed on ``l_core = 0``, the core-shell generation pin, and the dipole
    operator. A build that succeeds proves none of them raised; only a solve proves they
    produce a spectrum.

    The assertion is deliberately weak on values -- this ``h0`` is a random Hermitian matrix,
    not a material -- and strong on the two things that would signal a wrong generalisation:
    every spectrum must be finite, and each must carry non-zero weight. A silently mislabelled
    shell gives an identically zero XAS, which is exactly what the L2,3-only code did when its
    two shells arrived in the wrong dict order.
    """
    from mpi4py import MPI

    import h5py

    from impurityModel.ed.get_spectra import run_spectra

    built = build(load_input(write(K_EDGE_RUN.format(h0=small_p_shell_h0.name), "k_edge_run.toml")))
    assert built.model.n_spin_orbitals == 10

    output = tmp_path / "k_edge.h5"
    run_spectra(built.model, built.spectra, built.basis, MPI.COMM_WORLD, verbosity=0, output_filename=str(output))

    if MPI.COMM_WORLD.rank != 0:
        return
    with h5py.File(output, "r") as f:
        for name in ("PS/spectra", "XPS/spectra", "XAS/tensor", "RIXS/tensor"):
            values = np.asarray(f[name])
            assert np.all(np.isfinite(values)), f"{name} is not finite"
            assert np.max(np.abs(values)) > 0, f"{name} has no weight at the K edge"
