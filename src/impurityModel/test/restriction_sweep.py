"""Phase-2 cutoff-frontier sweep: accuracy vs. basis size as the restriction knobs tighten.

For a workload, run ``calc_selfenergy`` under a ladder of restriction settings and compare
each against a *looser-than-default* reference. The measured deviation answers the campaign
question directly: how far can each knob be tightened before the self-energy / Green's
function moves by more than the physical broadening (~1e-3 relative)?

Sweepable knobs
---------------
* ``slater_weight_min`` -- determinant amplitude cutoff (``BasisOptions``).
* ``occ_cutoff`` -- filled/empty bath classification cutoff (``BasisOptions``).
* ``dN`` -- excited-sector occupation window width (``BasisOptions``).
* ``coupling_cutoff`` / ``min_dist`` -- freeze-eligibility threshold, overridden through the
  :mod:`basis_restrictions` module constants (``COUPLING_CUTOFF_DEFAULT`` / ``MIN_DIST_DEFAULT``).
* ``chain_hole_fraction`` / ``chain_electron_fraction`` -- chain-window widths, overridden
  through ``basis_restrictions.CHAIN_FILLED_HOLE_FRACTION`` / ``CHAIN_EMPTY_ELECTRON_FRACTION``.

Metrics per config: ground-state energy shift ``dE0``; max relative self-energy / Green's
function deviation on the evaluation mesh; causality (max ``Im`` diagonal, must stay <= 0);
ground-state basis size (from a paired Phase-1 build); peak RSS.

Opt-in, one workload/config ladder per process for honest ``VmHWM``::

    RUN_RESTRICTION_SWEEP=1 RESTRICTION_SWEEP_WORKLOAD=fcc_ni_5 \\
        pytest -s -m benchmark src/impurityModel/test/restriction_sweep.py

or directly::

    python -m impurityModel.test.restriction_sweep fcc_ni_5
"""

import os
from contextlib import contextmanager

import numpy as np

from impurityModel.ed import basis_restrictions
from impurityModel.test.restriction_diagnostics import WORKLOADS, build_ground_state

# Sentinel for the absolute-cap overrides, whose ``None`` is a meaningful value ("disable the
# cap"), distinct from "leave the module default untouched".
_UNSET = object()


@contextmanager
def _restriction_overrides(
    coupling_cutoff=None,
    min_dist=None,
    hole_fraction=None,
    electron_fraction=None,
    max_holes=_UNSET,
    max_electrons=_UNSET,
    graded=None,
    intermediate_max_excitations=None,
):
    """Temporarily override the :mod:`basis_restrictions` tunables (restored on exit).

    ``max_holes`` / ``max_electrons`` accept ``None`` as a *meaningful* value (disable the
    absolute per-chain cap), so they use a distinct ``_UNSET`` sentinel to mean "leave the
    module default"; the fraction knobs keep their ``None``-means-leave convention. ``graded``
    toggles the three-zone hopping-derived restriction (``CHAIN_GRADED_RESTRICT``).
    """
    saved = (
        basis_restrictions.COUPLING_CUTOFF_DEFAULT,
        basis_restrictions.MIN_DIST_DEFAULT,
        basis_restrictions.CHAIN_FILLED_HOLE_FRACTION,
        basis_restrictions.CHAIN_EMPTY_ELECTRON_FRACTION,
        basis_restrictions.CHAIN_MAX_HOLES,
        basis_restrictions.CHAIN_MAX_ELECTRONS,
        basis_restrictions.CHAIN_GRADED_RESTRICT,
        basis_restrictions.CHAIN_INTERMEDIATE_MAX_EXCITATIONS,
    )
    if coupling_cutoff is not None:
        basis_restrictions.COUPLING_CUTOFF_DEFAULT = coupling_cutoff
    if min_dist is not None:
        basis_restrictions.MIN_DIST_DEFAULT = min_dist
    if hole_fraction is not None:
        basis_restrictions.CHAIN_FILLED_HOLE_FRACTION = hole_fraction
    if electron_fraction is not None:
        basis_restrictions.CHAIN_EMPTY_ELECTRON_FRACTION = electron_fraction
    if max_holes is not _UNSET:
        basis_restrictions.CHAIN_MAX_HOLES = max_holes
    if max_electrons is not _UNSET:
        basis_restrictions.CHAIN_MAX_ELECTRONS = max_electrons
    if graded is not None:
        basis_restrictions.CHAIN_GRADED_RESTRICT = graded
    if intermediate_max_excitations is not None:
        basis_restrictions.CHAIN_INTERMEDIATE_MAX_EXCITATIONS = intermediate_max_excitations
    try:
        yield
    finally:
        (
            basis_restrictions.COUPLING_CUTOFF_DEFAULT,
            basis_restrictions.MIN_DIST_DEFAULT,
            basis_restrictions.CHAIN_FILLED_HOLE_FRACTION,
            basis_restrictions.CHAIN_EMPTY_ELECTRON_FRACTION,
            basis_restrictions.CHAIN_MAX_HOLES,
            basis_restrictions.CHAIN_MAX_ELECTRONS,
            basis_restrictions.CHAIN_GRADED_RESTRICT,
            basis_restrictions.CHAIN_INTERMEDIATE_MAX_EXCITATIONS,
        ) = saved


# A config = knob overrides. ``None`` leaves a knob at the production default. The first
# config in a ladder is the reference: the *production default* settings. The campaign
# question is whether each knob can be tightened FROM the default without moving the
# spectra, so every later config only tightens (higher amplitude cutoff, narrower chain
# window, more aggressive freeze) and is measured against the default reference. (A
# looser-than-default reference would build an uncapped basis that does not fit the test
# machine; the truncation_reliability campaign already established the default as the
# accurate production baseline.)
_DEFAULT_SWMIN = float(np.sqrt(np.finfo(float).eps))


def _default_ladder():
    """Production-default reference + one-knob-at-a-time tightenings, workload-independent."""
    base = {
        "slater_weight_min": _DEFAULT_SWMIN,
        "hole_fraction": 0.5,
        "electron_fraction": 0.5,
        "coupling_cutoff": 1e-3,
    }
    return [
        {"name": "default(ref)", **base},
        {"name": "swmin=1e-6", **{**base, "slater_weight_min": 1e-6}},
        {"name": "swmin=1e-5", **{**base, "slater_weight_min": 1e-5}},
        {"name": "swmin=1e-4", **{**base, "slater_weight_min": 1e-4}},
        {"name": "chain=0.34", **{**base, "hole_fraction": 1 / 3, "electron_fraction": 1 / 3}},
        {"name": "chain=0.25", **{**base, "hole_fraction": 0.25, "electron_fraction": 0.25}},
        {"name": "couple=1e-2", **{**base, "coupling_cutoff": 1e-2}},
    ]


def _max_rel_dev(a, ref):
    """Max |a - ref| / max|ref| over the array (0 if the reference is uniformly ~0)."""
    a = np.asarray(a)
    ref = np.asarray(ref)
    if a.shape != ref.shape:
        return float("nan")
    scale = np.max(np.abs(ref))
    if scale < 1e-30:
        return float(np.max(np.abs(a - ref)))
    return float(np.max(np.abs(a - ref)) / scale)


def _run_config(workload_key, cfg, comm=None, n_iw=0, n_w=60, verbosity=0):
    """One config: returns metrics dict (self-energy/GF arrays + E0 + gs_size + peak RSS)."""
    from impurityModel.ed.memory_estimate import peak_rss_bytes
    from impurityModel.ed.model import BasisOptions, ImpurityModel, Meshes, SolverOptions
    from impurityModel.ed.selfenergy import calc_selfenergy
    from impurityModel.test.real_workload import _subsample, load_workload

    wl = load_workload(WORKLOADS[workload_key])
    with _restriction_overrides(
        coupling_cutoff=cfg.get("coupling_cutoff"),
        min_dist=cfg.get("min_dist"),
        hole_fraction=cfg.get("hole_fraction"),
        electron_fraction=cfg.get("electron_fraction"),
        max_holes=cfg.get("max_holes", _UNSET),
        max_electrons=cfg.get("max_electrons", _UNSET),
    ):
        # Ground-state basis size (Phase-1 build; cheap relative to the GF, responds directly
        # to the amplitude cutoff and chain-window knobs).
        gs = build_ground_state(
            workload_key,
            comm=comm,
            verbosity=0,
            truncation_threshold=cfg.get("truncation_threshold", 300000),
            excitation_budget=cfg.get("excitation_budget"),
            slater_weight_min=cfg.get("slater_weight_min"),
        )
        gs_size = gs["gs_basis"].size
        del gs

        model = ImpurityModel(
            h0=wl["h0"],
            u4=wl["u4"],
            impurity_orbitals=wl["impurity_orbitals"],
            rot_to_spherical=wl["rot_to_spherical"],
        )
        w = _subsample(wl["w_mesh"], n_w)
        iw = _subsample(wl["iw_mesh"], n_iw)
        meshes = Meshes(iw=1j * iw if iw is not None else None, w=w, delta=wl["delta"])
        basis = BasisOptions(
            nominal_occ=wl["nominal_occ"],
            mixed_valence=wl["mixed_valence"],
            dN=cfg.get("dN", wl["dN"]),
            # Generous finite cap: bounds memory on a small test machine without binding the
            # natural basis of these workloads (so knob effects on size are not confounded by
            # the cap). A binding cap would itself shrink the basis; see truncation_reliability.
            truncation_threshold=cfg.get("truncation_threshold", 300000),
            chain_restrict=wl["chain_restrict"],
            spin_flip_dj=wl["spin_flip_dj"],
            occ_cutoff=cfg.get("occ_cutoff", wl["occ_cutoff"]),
            slater_weight_min=cfg.get("slater_weight_min", wl["slaterWeightMin"]),
            excitation_budget=cfg.get("excitation_budget"),
            tau=wl["tau"],
        )
        solver = SolverOptions(
            reort=wl["reort"],
            dense_cutoff=wl["dense_cutoff"],
            sparse_green=wl["sparse_green"],
            gf_method="lanczos",
        )
        result = calc_selfenergy(
            model, meshes, basis, solver, comm=comm, verbosity=verbosity, cluster_label=f"{workload_key}:{cfg['name']}"
        )

    peak = peak_rss_bytes()
    if comm is not None:
        from mpi4py import MPI

        peak = comm.allreduce(peak, op=MPI.MAX)
    if comm is not None and comm.rank != 0:
        return None
    sig = result["sigma_real"]
    gf = result["gs_realaxis"]
    return {
        "name": cfg["name"],
        "E0": float(np.min(result["gs_energies"])),
        "sigma": np.asarray(sig),
        "gf": np.asarray(gf),
        "causality": float(np.max(np.diagonal(np.asarray(gf).imag, axis1=1, axis2=2))),
        "gs_size": gs_size,
        "peak_rss_gib": peak / 2**30,
    }


def sweep(workload_key, ladder=None, comm=None, n_iw=0, n_w=60, verbosity=0):
    """Run a config ladder for a workload; return per-config metrics vs the reference (config 0)."""
    ladder = ladder or _default_ladder()
    rank = comm.rank if comm is not None else 0
    rows = []
    ref = None
    for cfg in ladder:
        m = _run_config(workload_key, cfg, comm=comm, n_iw=n_iw, n_w=n_w, verbosity=verbosity)
        if rank != 0:
            continue
        if ref is None:
            ref = m
        rows.append(
            {
                "name": m["name"],
                "dE0": m["E0"] - ref["E0"],
                "dSigma_rel": _max_rel_dev(m["sigma"], ref["sigma"]),
                "dGF_rel": _max_rel_dev(m["gf"], ref["gf"]),
                "causality": m["causality"],
                "gs_size": m["gs_size"],
                "gs_ratio": ref["gs_size"] / m["gs_size"] if m["gs_size"] else float("nan"),
                "peak_rss_gib": m["peak_rss_gib"],
                "max_imG": float(np.max(np.abs(ref["gf"].imag))),
            }
        )
    return rows


def render(workload_key, rows):
    lines = [f"=== restriction cutoff sweep: {workload_key} (reference = row 0) ==="]
    lines.append(
        f"{'config':<18}{'dE0':>12}{'dSigma_rel':>12}{'dGF_rel':>12}{'causality':>12}"
        f"{'gs_size':>10}{'gs_x':>7}{'RSS/GiB':>9}"
    )
    for r in rows:
        lines.append(
            f"{r['name']:<18}{r['dE0']:>12.2e}{r['dSigma_rel']:>12.2e}{r['dGF_rel']:>12.2e}"
            f"{r['causality']:>12.2e}{r['gs_size']:>10d}{r['gs_ratio']:>7.2f}{r['peak_rss_gib']:>9.3f}"
        )
    if rows:
        lines.append(f"(reference max|Im G| = {rows[0]['max_imG']:.3e}; trust deviations only if > 0)")
    return "\n".join(lines)


@contextmanager
def _force_gs_slater_weight_min(gs_cutoff):
    """Force ``calc_gs`` to use ``gs_cutoff`` regardless of the driver's ``slaterWeightMin``.

    ``calc_selfenergy`` passes a single ``slaterWeightMin`` to both the ground-state build and
    the Green's function. This wraps the ``calc_gs`` name in the ``selfenergy`` module so the
    *ground state* uses ``gs_cutoff`` while the GF keeps the driver value -- the split-cutoff
    hypothesis: a looser GS cutoff shrinks the GS basis; does it move the spectra?
    """
    import impurityModel.ed.selfenergy as se

    orig = se.calc_gs

    def patched(*args, **kwargs):
        kwargs["slaterWeightMin"] = gs_cutoff
        return orig(*args, **kwargs)

    se.calc_gs = patched
    try:
        yield
    finally:
        se.calc_gs = orig


def _selfenergy_once(workload_key, gs_cutoff, gf_cutoff, comm=None, n_iw=0, n_w=8, verbosity=0):
    """Run calc_selfenergy with a (possibly split) GS/GF amplitude cutoff; return sigma/gf/E0."""
    from impurityModel.ed.model import BasisOptions, ImpurityModel, Meshes, SolverOptions
    from impurityModel.ed.selfenergy import calc_selfenergy
    from impurityModel.test.real_workload import _subsample, load_workload

    wl = load_workload(WORKLOADS[workload_key])
    model = ImpurityModel(
        h0=wl["h0"], u4=wl["u4"], impurity_orbitals=wl["impurity_orbitals"], rot_to_spherical=wl["rot_to_spherical"]
    )
    w = _subsample(wl["w_mesh"], n_w)
    iw = _subsample(wl["iw_mesh"], n_iw)
    meshes = Meshes(iw=1j * iw if iw is not None else None, w=w, delta=wl["delta"])
    basis = BasisOptions(
        nominal_occ=wl["nominal_occ"],
        mixed_valence=wl["mixed_valence"],
        dN=wl["dN"],
        truncation_threshold=300000,
        chain_restrict=wl["chain_restrict"],
        spin_flip_dj=wl["spin_flip_dj"],
        occ_cutoff=wl["occ_cutoff"],
        slater_weight_min=gf_cutoff,
        tau=wl["tau"],
    )
    solver = SolverOptions(
        reort=wl["reort"], dense_cutoff=wl["dense_cutoff"], sparse_green=wl["sparse_green"], gf_method="lanczos"
    )
    with _force_gs_slater_weight_min(gs_cutoff):
        result = calc_selfenergy(
            model,
            meshes,
            basis,
            solver,
            comm=comm,
            verbosity=verbosity,
            cluster_label=f"{workload_key}:gs{gs_cutoff:.0e}/gf{gf_cutoff:.0e}",
        )
    if comm is not None and comm.rank != 0:
        return None
    gf = np.asarray(result["gs_realaxis"])
    return {
        "sigma": np.asarray(result["sigma_real"]),
        "gf": gf,
        "E0": float(np.min(result["gs_energies"])),
        "causality": float(np.max(np.diagonal(gf.imag, axis1=1, axis2=2))),
    }


def split_cutoff_confirm(workload_key, gs_cutoffs=(1e-4, 1e-3), gf_cutoff=None, comm=None, n_w=8, verbosity=0):
    """Confirm the split-cutoff recommendation: loosen the GS cutoff, keep the GF cutoff tight.

    Reference: GS and GF both at the tight default (``gf_cutoff``, default √ε) -- the current
    production behavior. Each trial loosens *only* the GS cutoff. If the spectra move by less
    than the physical broadening (~1e-3 relative), the GS basis can be built with the looser
    cutoff (a memory saving on large-GS workloads) without touching the Green's function.
    """
    gf_cutoff = float(np.sqrt(np.finfo(float).eps)) if gf_cutoff is None else gf_cutoff
    rank = comm.rank if comm is not None else 0
    ref = _selfenergy_once(workload_key, gf_cutoff, gf_cutoff, comm=comm, n_w=n_w, verbosity=verbosity)
    rows = []
    for gc in gs_cutoffs:
        m = _selfenergy_once(workload_key, gc, gf_cutoff, comm=comm, n_w=n_w, verbosity=verbosity)
        if rank == 0:
            rows.append(
                {
                    "gs_cutoff": gc,
                    "dE0": m["E0"] - ref["E0"],
                    "dSigma_rel": _max_rel_dev(m["sigma"], ref["sigma"]),
                    "dGF_rel": _max_rel_dev(m["gf"], ref["gf"]),
                    "causality": m["causality"],
                }
            )
    if rank == 0:
        print(f"\n=== split-cutoff confirmation: {workload_key} (GF cutoff fixed at {gf_cutoff:.1e}) ===")
        print(f"reference: GS & GF both {gf_cutoff:.1e}; ref max|Im G| = {np.max(np.abs(ref['gf'].imag)):.3e}")
        print(f"{'GS cutoff':>12}{'dSigma_rel':>13}{'dGF_rel':>13}{'dE0':>13}{'causality':>13}")
        for r in rows:
            print(
                f"{r['gs_cutoff']:>12.1e}{r['dSigma_rel']:>13.2e}{r['dGF_rel']:>13.2e}"
                f"{r['dE0']:>13.2e}{r['causality']:>13.2e}"
            )
    return rows


RUN = os.environ.get("RUN_RESTRICTION_SWEEP") == "1"


def test_restriction_sweep():
    import pytest

    if not RUN:
        pytest.skip("Set RUN_RESTRICTION_SWEEP=1 to run the cutoff-frontier sweep.")
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    key = os.environ.get("RESTRICTION_SWEEP_WORKLOAD", "nio_20")
    rows = sweep(key, comm=comm, n_w=int(os.environ.get("RESTRICTION_SWEEP_NW", "60")))
    if comm.rank == 0:
        print("\n" + render(key, rows))


test_restriction_sweep.benchmark = True


if __name__ == "__main__":
    import sys

    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD if MPI.COMM_WORLD.size > 1 else None
    except ImportError:
        comm = None
    key = sys.argv[1] if len(sys.argv) > 1 else "nio_20"
    if len(sys.argv) > 2 and sys.argv[2] == "split":
        gs_cutoffs = [float(x) for x in sys.argv[3:]] or [1e-4, 1e-3]
        split_cutoff_confirm(key, gs_cutoffs=gs_cutoffs, comm=comm)
    else:
        rows = sweep(key, comm=comm)
        if comm is None or comm.rank == 0:
            print("\n" + render(key, rows))
