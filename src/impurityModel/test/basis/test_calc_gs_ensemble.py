"""``calc_gs`` under ``mixed_valence``: the merged thermal density matrix is the grand-canonical one (MV 6)."""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.block_structure import BlockStructure
from impurityModel.ed.groundstate import calc_gs
from impurityModel.test.support.comms import COMMS
from impurityModel.test.support.gs_oracle import exact_spectra, pinned_model

TAU = 0.15


def _setup(index, comm, mixed_valence):
    h_op, imp, bath, N0, n_orb, n_val, _ = pinned_model(20261016, index)
    n_imp = len(imp[0][0])
    block_structure = BlockStructure(
        blocks=[list(range(n_imp))],
        identical_blocks=[[0]],
        transposed_blocks=[[]],
        particle_hole_blocks=[[]],
        particle_hole_transposed_blocks=[[]],
        inequivalent_blocks=[0],
    )
    setup = {
        "impurity_orbitals": imp,
        "bath_states": bath,
        "N0": N0,
        "tau": TAU,
        "chain_restrict": False,
        "dense_cutoff": 1000,
        "comm": comm,
        "truncation_threshold": np.inf,
        "mixed_valence": mixed_valence,
    }
    return h_op, setup, block_structure, np.eye(n_imp, dtype=complex), n_orb, n_val


def _run(index, comm, mixed_valence, tmp_path):
    h_op, setup, bs, rot, n_orb, n_val = _setup(index, comm, mixed_valence)
    out = calc_gs(h_op, setup, bs, rot, 0, slaterWeightMin=1e-12, stats_path=str(tmp_path / "gs.json"))
    return out, n_orb, n_val


@pytest.mark.mpi
@pytest.mark.parametrize("comm", COMMS)
def test_a_single_member_ensemble_is_bit_identical_to_off(comm, tmp_path):
    (psis0, es0, _, rho0, info0), _, _ = _run(20, comm, "off", tmp_path)
    (psis1, es1, _, rho1, info1), _, _ = _run(20, comm, "auto", tmp_path)
    assert np.array_equal(es0, es1) and np.array_equal(rho0, rho1)
    assert len(info1["ensemble"]["members"]) == 1 and info1["ensemble"]["members"][0]["weight"] == 1.0


@pytest.mark.mpi
@pytest.mark.parametrize("comm", COMMS)
@pytest.mark.parametrize("index, members", [(3, [1, 2]), (33, [0, 1])])
def test_the_merged_density_matrix_carries_the_grand_canonical_electron_count(comm, index, members, tmp_path):
    (psis, es, basis, rho, info), n_orb, n_val = _run(index, comm, "on", tmp_path)
    ens = info["ensemble"]
    assert sorted(m["total"] for m in ens["members"]) == members
    h_op, *_ = _setup(index, comm, "on")
    spectra = exact_spectra(h_op, n_orb)
    z = {t: np.exp(-spectra[n_val + t] / TAU).sum() for t in members}
    exact = sum(z[t] * (n_val + t) for t in members) / sum(z.values())
    assert np.trace(rho).real == pytest.approx(exact, abs=5e-3)
    # the dominant member's own states are what the tuple returns; the winner alone would be an integer count
    dominant = ens["members"][0]
    assert len(es) == dominant["n_states"] and len(psis) == len(es)
    assert abs((n_val + dominant["total"]) - exact) > 0.05
    assert sum(m["weight"] for m in ens["members"]) == pytest.approx(1.0)
    assert len(ens["state_sector"]) == sum(m["n_states"] for m in ens["members"])
    assert (comm or MPI.COMM_SELF).allgather(float(np.trace(rho).real)) == [float(np.trace(rho).real)] * (
        comm.size if comm else 1
    )
