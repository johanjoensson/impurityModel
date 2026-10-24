r"""Replay one archived solver call with the Green's-function support freeze (settle) switched, and grade it.

The measurement rig behind ``doc/plans/gf_support_freeze_redesign.md``. One run = one (archive, cluster,
iteration, ``slaterWeightMin``, freeze mode) point: the archived ``calc_selfenergy`` call is rebuilt through
:mod:`real_workload` (same options, ``gf_tol``/``gf_real_tol`` from the archive), the self-energy causality guard
is left non-fatal (``SIGMA_CAUSALITY_TOL``) so the Sigma comes out whatever it is, and the arrays are written
to an ``.npz``. :func:`grade` then compares two such runs against the acceptance rule of the plan:

* (A) ``max |dSigma_mats|`` on the Matsubara mesh;
* (B) per orbital, pointwise on the real mesh, ``|dSigma_ii(w)| / h_i(w)`` with the causality guard's own headroom
  ``h_i(w) = -Im Sigma_ref,ii(w) + tol * max_w |Im Sigma_ref,ii|`` (``tol`` = ``SIGMA_CAUSALITY_TOL``);
* ``max |dG| / max |G|`` per axis, reported and not gated.

Run (one process per point; ``-n 3`` is what the numbers in the write-up used)::

    mpiexec -n 3 python -m impurityModel.test.support.freeze_replay run ARCHIVE.h5 --iteration 1 \
        --swm 1e-8 --freeze off --out debug/gf_freeze/swm8_nofreeze
    python -m impurityModel.test.support.freeze_replay grade debug/gf_freeze/swm8.npz \
        debug/gf_freeze/swm8_nofreeze.npz

``--freeze`` is ``on`` (the code's default), ``off``, or a number (a settle ratio, once the rule takes one).
The archive path is always an argument: nothing here hard-codes a workload.
"""

import argparse
import os
import sys
import time

# One thread per rank: set before numpy loads (OpenBLAS reads these once, at import), or a 3-rank replay
# runs ~30 threads and oversubscribes the machine. Overridable from the environment.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

#: Archive attributes the replay may skip: ``warm_start`` / ``warm_tol`` describe the bath fit that produced
#: ``H solver``; ``spin_flip_dj`` is a retired option (recorded False by every archive that has it).
_BATH_FIT_OR_RETIRED = frozenset({"warm_start", "warm_tol", "spin_flip_dj"})

#: Per-orbital quantities :func:`grade` reports, in print order.
GRADE_FIELDS = ("mats_abs", "real_abs", "real_vs_headroom", "g_real_rel", "g_mats_rel", "worst_im_ref", "worst_im_new")


def headroom(sigma_ref, tol):
    r"""``h_i(w) = -Im Sigma_ref,ii(w) + tol * max_w |Im Sigma_ref,ii|`` -- the room the causality guard leaves.

    ``sigma_ref`` has shape ``(n_w, n, n)``; the result ``(n_w, n)``. Points with ``h < tol * max`` already
    violate causality in the reference (the known-issues artefact) and are not a freeze effect.
    """
    im = np.imag(np.diagonal(np.asarray(sigma_ref), axis1=1, axis2=2))
    return -im + tol * np.max(np.abs(im), axis=0, keepdims=True)


def grade(ref, new, tol=1e-3):
    """Compare two result dicts (``sigma``, ``sigma_real``, ``gs_matsubara``, ``gs_realaxis``) per the plan's rule.

    Returns a dict with ``mats_abs`` (A), ``real_abs``, ``real_vs_headroom`` (B: the largest
    ``|dSigma_ii| / h_i`` over orbitals and frequencies, with ``(w, orbital)`` of the binding point in
    ``real_binding``), the relative G changes per axis and the worst ``Im Sigma_ii`` of both runs.
    """
    out = {}
    out["mats_abs"] = float(np.max(np.abs(new["sigma"] - ref["sigma"])))
    d_real = np.abs(np.diagonal(new["sigma_real"] - ref["sigma_real"], axis1=1, axis2=2))
    out["real_abs"] = float(np.max(np.abs(new["sigma_real"] - ref["sigma_real"])))
    h = headroom(ref["sigma_real"], tol)
    ratio = d_real / np.where(h > 0, h, np.nan)
    ratio = np.where(np.isfinite(ratio), ratio, np.inf)
    i = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
    out["real_vs_headroom"] = float(ratio[i])
    out["real_binding"] = (int(i[0]), int(i[1]))
    out["artefact_points"] = int(
        np.sum(
            h
            < tol
            * np.max(np.abs(np.imag(np.diagonal(ref["sigma_real"], axis1=1, axis2=2))), axis=0, keepdims=True)
            * (1 - 1e-12)
        )
    )
    for key, name in (("gs_realaxis", "g_real_rel"), ("gs_matsubara", "g_mats_rel")):
        out[name] = float(np.max(np.abs(new[key] - ref[key])) / max(np.max(np.abs(ref[key])), np.finfo(float).tiny))
    for label, res in (("ref", ref), ("new", new)):
        im = np.imag(np.diagonal(res["sigma_real"], axis1=1, axis2=2))
        out[f"worst_im_{label}"] = float(np.max(im))
    return out


def format_grade(g):
    """One block of text for :func:`grade`'s result."""
    lines = [
        f"(A) max|dSigma_mats|           {g['mats_abs']:.3e}",
        f"    max|dSigma_real|           {g['real_abs']:.3e}",
        f"(B) max |dSigma_ii|/headroom   {g['real_vs_headroom']:.3e} at mesh point {g['real_binding'][0]}, "
        f"orbital {g['real_binding'][1]}   (reference points already inside the guard: {g['artefact_points']})",
        f"    max|dG|/max|G| real, mats  {g['g_real_rel']:.3e}, {g['g_mats_rel']:.3e}   (context, not gated)",
        f"    max Im Sigma_ii real       ref {g['worst_im_ref']:+.3e}   new {g['worst_im_new']:+.3e}",
    ]
    return "\n".join(lines)


def load_result(path):
    """An ``.npz`` written by :func:`run` as a plain dict of arrays."""
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def set_freeze(mode):
    """Switch the support freeze: ``"on"``, ``"off"`` or a settle ratio. Returns what it set, for the log."""
    from impurityModel.ed import gf_solvers

    if mode == "on":
        return "on"
    if mode == "off":
        gf_solvers._SUPPORT_FREEZE = False
        return "off"
    ratio = float(mode)
    if not hasattr(gf_solvers, "_SETTLE_RATIO_OVERRIDE"):
        raise SystemExit("this tree has no settle ratio yet; use --freeze on|off")
    gf_solvers._SETTLE_RATIO_OVERRIDE = ratio
    return f"ratio {ratio:g}"


def run(h5_path, out, iteration=1, cluster=None, swm=None, freeze="on", comm=None, verbosity=1):
    """Replay one archived call and write ``out + '.npz'`` on rank 0. Returns the result dict (rank 0)."""
    os.environ.setdefault("SIGMA_CAUSALITY_TOL", "1e9")
    from impurityModel.test.support import real_workload as rw

    mode = set_freeze(freeze)
    wl = rw.load_workload(h5_path, cluster=cluster, iteration=iteration)
    if wl["ignored_attrs"]:
        # warm_start / warm_tol describe the bath fit that produced H solver, not the solver call.
        extra = set(wl["ignored_attrs"]) - _BATH_FIT_OR_RETIRED
        if extra:
            raise SystemExit(f"archive records options this replay does not consume: {sorted(extra)}")
    if swm is not None:
        wl["slaterWeightMin"] = float(swm)
    t0 = time.perf_counter()
    res = rw.run_selfenergy(wl, comm=comm, verbosity=verbosity, gf_tol="archive", gf_real_tol="archive")
    wall = time.perf_counter() - t0
    if comm is not None and comm.rank != 0:
        return None
    arrays = {}
    for key, value in res.items():
        try:
            arr = np.asarray(value)
        except Exception:  # noqa: BLE001 -- non-array result entries are simply not saved
            continue
        if arr.dtype != object:
            arrays[key] = arr
    np.savez(
        out + ".npz",
        w=wl["w_mesh"],
        freeze=np.array(mode),
        swm=np.array(wl["slaterWeightMin"]),
        wall=np.array(wall),
        **arrays,
    )
    print(f"RESULT {out}: swm {wl['slaterWeightMin']:g} freeze {mode} wall {wall:.1f} s", flush=True)
    return arrays


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("archive")
    r.add_argument("--iteration", type=int, default=1)
    r.add_argument("--cluster", default=None)
    r.add_argument("--swm", type=float, default=None)
    r.add_argument("--freeze", default="on")
    r.add_argument("--out", required=True)
    r.add_argument("--verbosity", type=int, default=1)
    g = sub.add_parser("grade")
    g.add_argument("ref")
    g.add_argument("new")
    g.add_argument("--tol", type=float, default=1e-3)
    a = ap.parse_args(argv)
    if a.cmd == "grade":
        print(format_grade(grade(load_result(a.ref), load_result(a.new), a.tol)))
        return 0
    from mpi4py import MPI

    run(a.archive, a.out, a.iteration, a.cluster, a.swm, a.freeze, MPI.COMM_WORLD, a.verbosity)
    return 0


if __name__ == "__main__":
    sys.exit(main())
