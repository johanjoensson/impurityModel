"""The freeze-replay rig (``test/support/freeze_replay.py``): its grading functions always, the replay opt-in.

The smoke tests pin the metric definitions on synthetic self-energies, so the table functions the write-up
quotes cannot rot unnoticed. The replay itself needs an archive: ``WORKLOAD_H5=...``.
"""

import os

import numpy as np
import pytest

from impurityModel.test.support import freeze_replay as fr
from impurityModel.test.support.bench_opts import bench_opt


def _sigma(n_w=40, n=2, scale=1.0):
    w = np.linspace(-1, 1, n_w)
    im = -scale * (0.2 + 0.1 * np.cos(3 * w))[:, None] * np.ones((1, n))
    s = np.zeros((n_w, n, n), dtype=complex)
    for i in range(n):
        s[:, i, i] = 0.3 * w + 1j * im[:, i]
    return s


def test_headroom_is_the_guards_room():
    s = _sigma()
    h = fr.headroom(s, 1e-3)
    im = np.imag(np.diagonal(s, axis1=1, axis2=2))
    assert np.allclose(h, -im + 1e-3 * np.max(np.abs(im), axis=0))
    assert np.all(h > 0)


def test_grade_of_identical_runs_is_zero():
    s, m = _sigma(), _sigma(8)
    ref = {"sigma": m, "sigma_real": s, "gs_matsubara": m + 1, "gs_realaxis": s + 1}
    g = fr.grade(ref, dict(ref))
    assert g["mats_abs"] == 0 and g["real_abs"] == 0 and g["real_vs_headroom"] == 0 and g["g_real_rel"] == 0


def test_grade_binds_on_the_orbital_with_least_headroom():
    s = _sigma()
    new = s.copy()
    d = 1e-4
    new[7, 1, 1] += d
    ref = {"sigma": _sigma(8), "sigma_real": s, "gs_matsubara": _sigma(8) + 1, "gs_realaxis": s + 1}
    out = fr.grade(ref, {**ref, "sigma_real": new})
    assert out["real_binding"] == (7, 1)
    h = fr.headroom(s, 1e-3)[7, 1]
    assert out["real_vs_headroom"] == pytest.approx(d / h)
    assert "(B)" in fr.format_grade(out)


def test_a_reference_point_already_acausal_is_counted_not_gated_to_nan():
    s = _sigma()
    s[3, 0, 0] = s[3, 0, 0].real + 5e-2j  # Im Sigma > 0 in the reference
    ref = {"sigma": _sigma(8), "sigma_real": s, "gs_matsubara": _sigma(8) + 1, "gs_realaxis": s + 1}
    out = fr.grade(ref, dict(ref))
    assert np.isfinite(out["real_abs"]) and out["artefact_points"] >= 1




@pytest.mark.benchmark
def test_replay_runs_and_writes_arrays(tmp_path):
    from mpi4py import MPI

    h5 = bench_opt("WORKLOAD_H5")
    assert h5, "WORKLOAD_H5 must point to an impurityModel_data.h5 archive"
    out = str(tmp_path / "replay")
    res = fr.run(
        h5, out, iteration=int(bench_opt("ITERATION", "1")), freeze="on", comm=MPI.COMM_WORLD, verbosity=0
    )
    if MPI.COMM_WORLD.rank == 0:
        assert "sigma_real" in res and os.path.exists(out + ".npz")
