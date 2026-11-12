"""The ground-state report does not depend on how the spin-orbitals are numbered.

A SIAM is solved in its natural numbering and under permutations of every spin-orbital label (impurity
scattered among the bath, dn/up exchanged). ``H``, ``impurity_orbitals``, ``bath_states`` and the *rows* of
``rot_to_spherical`` (rows are orbitals, columns are spherical harmonics) are relabelled together; every
physical number of ``gs_info["report"]`` must come out the same, and the spin pairing must be the relabelled one.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.block_structure import BlockStructure
from impurityModel.ed.gs_calc import calc_gs
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator

#: old index -> new index
PERMUTATIONS = {
    "identity": [0, 1, 2, 3, 4, 5],
    "impurity scattered in the bath": [3, 5, 0, 1, 2, 4],
    "dn and up exchanged": [5, 3, 1, 0, 4, 2],
}


def _terms():
    ed, U, ev, ec, V = -2.0, 6.0, -4.0, 4.0, 1.0
    terms = {((o, "c"), (o, "a")): ed for o in (0, 1)}
    terms.update({((o, "c"), (o, "a")): ev for o in (2, 3)})
    terms.update({((o, "c"), (o, "a")): ec for o in (4, 5)})
    terms[((0, "c"), (1, "c"), (1, "a"), (0, "a"))] = U
    for a, b in ((0, 2), (1, 3), (0, 4), (1, 5)):
        terms[((a, "c"), (b, "a"))] = V
        terms[((b, "c"), (a, "a"))] = V
    return terms


def _report(perm, comm):
    """``gs_info["report"]`` as JSON for the SIAM relabelled by ``perm`` (old -> new)."""
    terms = {tuple((perm[o], k) for o, k in process): amp for process, amp in _terms().items()}
    imp = sorted(perm[o] for o in (0, 1))
    val = [perm[o] for o in (2, 3)]
    con = [perm[o] for o in (4, 5)]
    # rows of the rotation follow the sorted impurity list: row r is the orbital imp[r]
    rot = np.zeros((2, 2), dtype=complex)
    for old_orbital, spherical in ((0, 0), (1, 1)):  # old orbital 0 is dn (column 0), 1 is up (column 1)
        rot[imp.index(perm[old_orbital]), spherical] = 1.0
    block_structure = BlockStructure(
        blocks=[[0, 1]],
        identical_blocks=[[0]],
        transposed_blocks=[[]],
        particle_hole_blocks=[[]],
        particle_hole_transposed_blocks=[[]],
        inequivalent_blocks=[0],
    )
    setup = dict(
        impurity_orbitals={0: [imp]},
        bath_states=({0: [sorted(val)]}, {0: [sorted(con)]}),
        N0={0: 1},
        tau=0.01,
        dense_cutoff=1000,
        comm=comm,
        truncation_threshold=100000,
    )
    *_, gs_info = calc_gs(ManyBodyOperator(terms), setup, block_structure, rot, verbose=0, slaterWeightMin=1e-12)
    return gs_info["report"].to_json()


def _physical(report):
    """The label-free content of the report."""
    levels = sorted(
        (lv["eps_dn"], lv["n_dn"], lv["n_up"], lv["v_dn"], lv["sisb"]) for lv in report["screening"]["levels"]
    )
    manifolds = [
        {k: m[k] for k in ("energy", "g", "weight", "two_sz", "s", "l", "j", "s_sharp", "l_sharp", "j_sharp")}
        for m in report["manifolds"]
    ]
    return {
        "e0": report["e0"],
        "thermal": report["thermal"],
        "magnetic": {k: v for k, v in report["magnetic"].items() if k != "term"},
        "term": report["magnetic"]["term"],
        "manifolds": manifolds,
        "docc": report["correlation"]["docc_open_shell_total"],
        "sz2": report["correlation"]["sz2_thermal"],
        "energy_decomposition": report["energy_decomposition"],
        "grade": report["spin_layout"]["grade"],
        "levels": levels,
    }


def _assert_close(a, b, path=""):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            _assert_close(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_close(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        assert b == pytest.approx(a, abs=1e-8), path
    else:
        assert a == b, path


@pytest.mark.mpi
@pytest.mark.parametrize("name", [k for k in PERMUTATIONS if k != "identity"])
def test_report_is_invariant_under_relabelling_of_the_spin_orbitals(name):
    comm = MPI.COMM_WORLD
    reference = _report(PERMUTATIONS["identity"], comm)
    relabelled = _report(PERMUTATIONS[name], comm)
    if comm.rank != 0:  # off the root the report model is assembled incompletely (gs_stats exists on rank 0 only)
        return
    _assert_close(_physical(reference), _physical(relabelled))
    perm = PERMUTATIONS[name]
    pairs = sorted(tuple(sorted((perm[dn], perm[up]))) for dn, up in reference["spin_layout"]["imp_pairs"])
    assert sorted(tuple(sorted(p)) for p in relabelled["spin_layout"]["imp_pairs"]) == pairs
