"""Rank discipline: rank 0 parses, everyone agrees, and a bad file errors rather than hangs.

The failure this guards against is not a wrong answer, it is a hang. If every rank parsed for
itself and only some could read the file -- a networked or node-local filesystem, a launcher
that gives ranks different working directories -- the readers would raise and the rest would
block in the next collective. So these tests check the *contract* (the parse happens once, on
rank 0, and the verdict is broadcast) rather than trying to reproduce a divergent filesystem,
which a single-node test cannot do.
"""

import pathlib

import pytest
from mpi4py import MPI

from impurityModel.inputformat.reader import InputError, load_input

from .conftest import MINIMAL_SPECTROSCOPY


def _shared_input(tmp_path, text=MINIMAL_SPECTROSCOPY):
    """Write the file on rank 0 and broadcast its path (each rank has its own tmp_path)."""
    comm = MPI.COMM_WORLD
    path = None
    if comm.rank == 0:
        (tmp_path / "h0.pickle").write_bytes(b"")
        target = tmp_path / "in.toml"
        target.write_text(text)
        path = str(target)
    return comm.bcast(path, root=0)


@pytest.mark.mpi
def test_every_rank_resolves_the_same_values(tmp_path):
    comm = MPI.COMM_WORLD
    resolved = load_input(_shared_input(tmp_path), comm=comm)
    gathered = comm.allgather((resolved.tables, resolved.shells, resolved.calculation))
    assert all(entry == gathered[0] for entry in gathered)


@pytest.mark.mpi
def test_a_malformed_file_raises_on_every_rank(tmp_path):
    """An error on rank 0 alone would be a deadlock, not an error."""
    comm = MPI.COMM_WORLD
    path = _shared_input(tmp_path, MINIMAL_SPECTROSCOPY.replace("soc = 0.096", "sock = 0.096"))
    raised = False
    try:
        load_input(path, comm=comm)
    except InputError:
        raised = True
    assert all(comm.allgather(raised)), "the raise must be collective"


@pytest.mark.mpi
def test_only_rank_zero_touches_the_filesystem(tmp_path, monkeypatch):
    """The contract that makes the divergent-filesystem hang impossible.

    Made observable by breaking ``read_text`` everywhere except rank 0: if any other rank
    tried to open the file, this would raise there and the allgather below would deadlock or
    report the failure.
    """
    comm = MPI.COMM_WORLD
    path = _shared_input(tmp_path)
    if comm.rank != 0:
        original = pathlib.Path.read_text

        def forbidden(self, *args, **kwargs):
            raise AssertionError(f"rank {comm.rank} read {self}, but only rank 0 may parse")

        monkeypatch.setattr(pathlib.Path, "read_text", forbidden)

    resolved = load_input(path, comm=comm)
    assert resolved.calculation == "spectroscopy"
    assert all(comm.allgather(True))


@pytest.mark.mpi
def test_a_refusal_keeps_its_type_across_the_broadcast(tmp_path):
    """A caller must still be able to tell "not yet" from "your file is wrong" on every rank.

    The two refusals are broadcast as themselves, not collapsed into a generic error. The
    example here is a selection-rule violation rather than an unsupported edge, because the
    solver now runs every edge -- but the distinction the broadcast has to preserve is the
    same one, and ``UnsupportedCalculation`` is still reachable (a core shell on a
    single-shell calculation).
    """
    from impurityModel.inputformat.capabilities import InvalidShellCombination

    comm = MPI.COMM_WORLD
    # l_core = 2, l_valence = 4: |l_c - l_v| = 2, zero by the Gaunt selection rule.
    text = MINIMAL_SPECTROSCOPY.replace('l = 1\nrole = "core"', 'l = 2\nrole = "core"')
    text = text.replace('l = 2\nrole = "valence"', 'l = 4\nrole = "valence"')
    text = text.replace("F_vv = [7.5, 0, 9.9, 0, 6.6]", "F_vv = [7.5, 0, 9.9, 0, 6.6, 0, 5.0, 0, 3.0, 0]")
    text = text.replace("F_cc = [0, 0, 0]", "F_cc = [0, 0, 0, 0, 0]")
    text = text.replace("F_cv = [8.9, 0, 6.8]", "F_cv = [8.9, 0, 6.8, 0, 4.1]")
    text = text.replace("G_cv = [0, 5.0, 0, 2.8]", "G_cv = [0, 5.0, 0, 2.8, 0, 1.4]")
    path = _shared_input(tmp_path, text)
    kind = None
    try:
        load_input(path, comm=comm)
    except InvalidShellCombination:
        kind = "invalid_shells"
    except InputError:
        kind = "invalid"
    assert all(k == "invalid_shells" for k in comm.allgather(kind))
