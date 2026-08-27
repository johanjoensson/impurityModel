"""A bath-less valence shell (Hubbard-I), and how far the core/valence roles reach.

Two claims, both of which the code got wrong.

*Hubbard-I.* A correlated shell with no bath at all is a model, not a degenerate input: the
impurity is diagonalised with no hybridization. The crystal-field builder used to identify
"the shell it describes" as "the shell that has a bath", which cannot name a shell that has
none, and it required bath parameters that such a model has nowhere to put.

*Roles.* ``role = "core"`` and ``role = "valence"`` say which shell carries the core spin-orbit
term and which the field, the valence spin-orbit term and the Coulomb double counting. That is
meaningful for **any** pair of shells. Only core-level spectroscopy builds a transition
operator between them, and only that is bound by |l_core - l_valence| = 1 -- so the rule is
enforced where an operator is built and where a run asking for XPS/XAS/RIXS is validated, not
in the model constructor.
"""

from collections import OrderedDict

import numpy as np
import pytest

from impurityModel.ed import hamiltonian_io
from impurityModel.ed.atomic_physics import direct_array_length, exchange_array_length, inter_orbital_k_values
from impurityModel.inputformat.build import build
from impurityModel.inputformat.reader import InputError, load_input

HUBBARD_I = """
[format]
version = [1, 0]

[units]
energy = "eV"

[hamiltonian.crystal_field]
e_imp = -1.31796
e_deltaO_imp = 0.60422

[[shell]]
l = 2
role = "valence"
n_bath = 0
n_valence_bath = 0
nominal_occupation = 8
soc = 0.096

[interaction.slater]
F_vv = [7.5, 0, 9.9, 0, 6.6]

[double_counting.mlft]
c = 1.5

[spectroscopy]
w = {min = -12.0, max = 12.0, n = 41}

[spectroscopy.pes]
enabled = true

[spectroscopy.xps]
enabled = false

[spectroscopy.xas]
enabled = false

[spectroscopy.rixs]
enabled = false
"""


@pytest.fixture
def write(tmp_path):
    def _write(text, name="input.toml"):
        path = tmp_path / name
        path.write_text(text)
        return path

    return _write


@pytest.mark.parametrize(
    "l, parameters",
    [
        (1, {"e_imp": -0.7}),
        (2, {"e_imp": -1.3, "e_deltaO_imp": 0.6}),
        (3, {"e_imp": -1.1, "e_deltaO_imp": 0.9, "e_delta6_imp": 0.21}),
    ],
)
def test_a_bath_less_shell_gives_an_impurity_only_hamiltonian(l, parameters):
    """Hubbard-I at the operator level: the crystal field, and nothing else."""
    h0 = hamiltonian_io.get_CF_hamiltonian({l: 0}, {l: 0}, parameters, l=l)
    labels = {label for process in h0 for label, _ in process}
    assert labels, "a bath-less shell still has its own crystal-field splitting"
    # (l, s, m) is an impurity spin-orbital; (l, b) is a bath state. There must be no bath.
    assert all(len(label) == 3 for label in labels)
    assert {label[0] for label in labels} == {l}


def test_the_shell_is_named_not_inferred_from_where_the_baths_are():
    """ "The shell with a bath" cannot name a shell that has none."""
    parameters = {"e_imp": -1.1, "e_deltaO_imp": 0.9, "e_delta6_imp": 0.21}
    shells = OrderedDict({2: 0, 3: 0})  # a bath-less 3d core placeholder + a bath-less 4f valence
    h0 = hamiltonian_io.get_CF_hamiltonian(shells, shells, parameters, l=3)
    assert {label[0] for process in h0 for label, _ in process} == {3}

    # And `l` is now required outright -- there is no inference path left to get wrong.
    with pytest.raises(TypeError, match="l"):
        hamiltonian_io.get_CF_hamiltonian(shells, shells, parameters)


def test_a_half_filled_bath_block_is_refused_rather_than_halved():
    """Each impurity spin-orbital gets one partner per block, so a block is all or nothing."""
    with pytest.raises(ValueError, match="holds either 0 or 10 states -- not 7"):
        hamiltonian_io.get_CF_hamiltonian({2: 7}, {2: 7}, {"e_imp": -1.3, "e_deltaO_imp": 0.6}, l=2)


def test_a_bath_less_shell_needs_no_bath_parameters():
    """Requiring e_val_* / v_val_* of a model with nowhere to put them is noise."""
    assert hamiltonian_io.cf_parameter_names(2, hamiltonian_io.CFBathBlocks(valence=False, conduction=False)) == (
        "e_imp",
        "e_deltaO_imp",
    )
    assert hamiltonian_io.cf_parameter_names(3, hamiltonian_io.CFBathBlocks(valence=False, conduction=False)) == (
        "e_imp",
        "e_deltaO_imp",
        "e_delta6_imp",
    )
    # A valence bath but no conduction bath: half the rows, not all of them.
    assert hamiltonian_io.cf_parameter_names(2, hamiltonian_io.CFBathBlocks(valence=True, conduction=False)) == (
        "e_imp",
        "e_deltaO_imp",
        "e_val_eg",
        "e_val_t2g",
        "v_val_eg",
        "v_val_t2g",
    )


def test_a_hubbard_i_input_builds_from_two_parameters(write):
    built = build(load_input(write(HUBBARD_I)))
    assert built.model.n_spin_orbitals == 10
    assert built.model.impurity_orbitals == {2: [list(range(10))]}
    assert built.model.bath_states == ({2: [[]]}, {2: [[]]})
    assert built.model.h0


def test_a_hubbard_i_input_reports_bath_parameters_it_cannot_use(write):
    """Accepted but reported, rather than silently doing nothing.

    Refusing would be defensible, but the shipped NiO/CoO/FeO/MnO files all set `*_con_*`
    while asking for no conduction bath -- those values have never done anything, and saying
    so is more useful than breaking four working inputs.
    """
    text = HUBBARD_I.replace("e_deltaO_imp = 0.60422", "e_deltaO_imp = 0.60422\ne_val_eg = -4.4")
    built = build(load_input(write(text)))
    assert any("e_val_eg" in note and "Ignored" in note for note in built.notes), built.notes
    assert built.model.h0


def test_a_key_belonging_to_another_shell_is_still_an_error(write):
    """Distinct from the above: this one is not inert, it describes a different shell."""
    text = HUBBARD_I.replace("e_deltaO_imp = 0.60422", "e_deltaO_imp = 0.60422\ne_delta6_imp = 0.2")
    with pytest.raises(InputError, match="Not a key for this shell: e_delta6_imp"):
        build(load_input(write(text)))


def test_a_hubbard_i_model_solves(write, tmp_path):
    """The cheapest complete calculation this package can do: no bath, PES only."""
    from mpi4py import MPI

    import h5py

    from impurityModel.ed.get_spectra import run_spectra

    built = build(load_input(write(HUBBARD_I)))
    output = tmp_path / "hubbard_i.h5"
    run_spectra(built.model, built.spectra, built.basis, MPI.COMM_WORLD, verbosity=0, output_filename=str(output))

    if MPI.COMM_WORLD.rank != 0:
        return
    with h5py.File(output, "r") as f:
        spectrum = np.asarray(f["PS/spectra"])
    assert spectrum.shape[1] == 10
    assert np.all(np.isfinite(spectrum))
    assert np.max(np.abs(spectrum)) > 0


# --- The roles, and how far they reach -----------------------------------------------------


@pytest.mark.parametrize("lv, lc", [(2, 1), (1, 0), (3, 2), (4, 3)])
def test_a_dipole_allowed_edge_keeps_its_historical_array_lengths(lv, lc):
    """The generalisation must not move any shipped edge's arrays."""
    assert direct_array_length(lv, lc) == 2 * lc + 1
    assert exchange_array_length(lv, lc) == 2 * lc + 2


@pytest.mark.parametrize("lv, lc", [(2, 0), (3, 1), (3, 0), (2, 1), (3, 2)])
def test_the_exchange_array_reaches_every_k_the_sum_uses(lv, lc):
    """`2*l_c + 2` is only long enough when the shells are one apart.

    A 1s core under a 3d valence shell needs G^2, and the old length reached k=1 -- so the
    double-counting sum walked off the end of the array with an IndexError. That pair is a
    perfectly good Hamiltonian; it is only a *transition* between the two that is forbidden.
    """
    assert max(inter_orbital_k_values(lv, lc)) < exchange_array_length(lv, lc)


@pytest.mark.parametrize("lv, lc", [(2, 0), (1, 2), (3, 1)])
def test_the_coulomb_assembly_handles_any_pair_of_roles(lv, lc):
    """Including l_core above l_valence, which only core-level spectroscopy rules out.

    ``capabilities.check`` refuses ``l_core > l_valence`` when XPS/XAS/RIXS are asked for,
    since a core level lies below the shell it is excited into and the pair is mislabelled.
    A PES-only, self-energy or susceptibility run makes no such claim, so the assembly has to
    cope -- and the array lengths follow ``min``/``sum`` of the two, not ``l_core`` alone.
    """
    from impurityModel.ed import atomic_physics
    from impurityModel.ed.operator_algebra import daggerOp

    n_direct = direct_array_length(lv, lc)
    n_exchange = exchange_array_length(lv, lc)
    slater = dict(
        Fvv=[5.0] + [0.0] * (2 * lv),
        Fcc=[1.0] + [0.0] * (2 * lc),
        Fcv=[4.0] + [0.0] * (n_direct - 1),
        Gcv=[0.5] * n_exchange,
    )
    u_operator = atomic_physics.slater_condon_Uop(lv, lc, **slater)
    assert u_operator and daggerOp(u_operator) == u_operator

    dc = atomic_physics.dc_MLFT(
        lv,
        2,
        1.0,
        slater["Fvv"],
        lc=lc,
        n_core_i=2 * (2 * lc + 1),
        Fcv=slater["Fcv"],
        Gcv=slater["Gcv"],
    )
    assert sorted(dc) == sorted({lv, lc})


def test_two_shells_cannot_share_an_angular_momentum(write):
    """`l` is the identity of a shell everywhere below the input file.

    Two ``[[shell]]`` tables with the same ``l`` collapse into one key in ``nBaths``,
    ``impurity_orbitals`` and the operator labels. Before this was refused, the collision
    surfaced as a ``dc_MLFT`` complaint that the *core* shell's occupation was the valence
    shell's -- an error naming neither the file nor the cause. It was unreachable while only
    the {1, 2} pair was accepted; with any pair allowed it is one typo away.
    """
    text = HUBBARD_I.replace(
        '[[shell]]\nl = 2\nrole = "valence"',
        '[[shell]]\nl = 2\nrole = "core"\nnominal_occupation = 10\nsoc = 1.0\n\n' '[[shell]]\nl = 2\nrole = "valence"',
    )
    with pytest.raises(InputError, match="share l=2"):
        build(load_input(write(text)))
