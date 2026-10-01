"""Run one Green's-function method on a fixture and score it against the exact reference.

The unit of the basis-size comparison is a *cell*: one ``(method, cap or threshold)`` run of the
production driver ``get_Greens_function`` on an :class:`~aim_fixtures.AIM`, reported as

* the **basis size** it needed -- the largest retained determinant count over its work units, read
  from the ``gf_unit_basis`` trace notes (never from a basis object the solver was handed: an
  uncapped Lanczos recurrence does not track its support, and the per-point bicgstab maxima are
  the only honest size of a rebuilt-and-discarded basis);
* the **error** against the exact dense reference, on ``G`` and on ``Sigma`` -- ``Sigma`` is what the
  calculation is for, and it amplifies a small ``G`` error wherever ``|G|`` is small;
* what the solver itself reports (convergence, cap hit), so a truncation error is never confused
  with a solver that simply stopped.

Methods (``METHODS``): the two admission-free baselines (``lanczos-cap``, ``bicgstab-cap``), the two
importance-pruned variants (``lanczos-pruned``, ``bicgstab-outer``) and the existing amplitude cutoff
inside the matvec (``bicgstab-swm``, the control the pruned rules must beat). Per-frequency methods
run **cold** (warm-start history 0): a warm start carries its extrapolation's support into each
point's basis, which would measure the sweep, not the point.
"""

import contextlib
import io
import os
import time
from contextlib import contextmanager

import numpy as np
from mpi4py import MPI

from impurityModel.ed import solver_trace
from impurityModel.ed.greens_function import get_Greens_function
from impurityModel.test.support.aim_fixtures import free_G_inverse, reference_G, self_energy

METHODS = ("lanczos-cap", "lanczos-pruned", "bicgstab-cap", "bicgstab-outer", "bicgstab-swm")
NON_BINDING = 10**9  # a finite cap that never binds: installs the tracking proxy without truncating
BICGSTAB_ATOL = "1e-11"


@contextmanager
def env(**values):
    """Set environment variables for the duration, restoring what was there (knobs are read lazily)."""
    old = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class Mesh:
    """Frequency meshes of a comparison: the full ones, and the points the per-frequency methods solve.

    The per-frequency methods cost one solve per point, so they run on ``sel_*`` (the points of
    greatest spectral weight plus an even spread); the Lanczos family solves the whole mesh at once
    and is scored on the same points, so every method is compared where the others are.
    """

    def __init__(self, aim, omega, matsubara, delta, n_real=16, n_mats=8, truth=None):
        """``n_real = 0`` or ``n_mats = 0`` drops that axis: the Matsubara axis does not depend on the
        broadening, so a sweep over ``delta`` runs it once and the real axis per ``delta``.

        ``truth`` is an AIM whose reference is the *physical* answer -- the exact ground state in any
        basis, since ``G`` is basis-invariant. With a truncated ground state the cell then also reports
        ``dSigma_truth_*``, the end-to-end error: what the truncated ground state *and* the solver lose
        together. ``dSigma_*`` stays the error against the cell's own seeds, the solver's error alone."""
        self.truth = truth
        self.omega, self.delta = np.asarray(omega, dtype=float), float(delta)
        self.matsubara = np.asarray(matsubara, dtype=complex)
        self.z_real = self.omega + 1j * self.delta
        self.has_real, self.has_mats = n_real > 0, n_mats > 0
        if self.has_real:
            self.ref_real = reference_G(aim, self.z_real)
            self.truth_real = reference_G(truth, self.z_real) if truth is not None else None
            weight = np.linalg.norm(self.ref_real, axis=(1, 2))
            top = np.argsort(-weight)[: n_real // 2]
            spread = np.linspace(0, len(self.omega) - 1, n_real - len(top)).astype(int)
            self.sel_real = np.unique(np.concatenate([top, spread]))
        if self.has_mats:
            self.ref_mats = reference_G(aim, self.matsubara)
            self.truth_mats = reference_G(truth, self.matsubara) if truth is not None else None
            self.sel_mats = np.unique(
                np.linspace(0, len(self.matsubara) - 1, min(n_mats, len(self.matsubara))).astype(int)
            )


def _knobs(method, cap, eta, shell_tol=0.0):
    """Environment and driver arguments that realize ``method`` at ``cap`` / threshold ``eta``."""
    base = {"GF_BICGSTAB_ATOL": BICGSTAB_ATOL, "GF_BICGSTAB_RESIDUAL_CHECK": 0}
    if shell_tol:
        base["GF_ADMIT_FIRST_SHELL_TOL"] = repr(float(shell_tol))
    if method == "lanczos-cap":
        return dict(gf_method="lanczos", reort="full", slater=0.0, env=base)
    if method == "lanczos-pruned":
        return dict(
            gf_method="lanczos",
            reort="full",
            slater=0.0,
            env={**base, "GF_LANCZOS_ADMIT_TOL": repr(float(eta)), "GF_APPLY_ROW_CHUNKS": 1},
        )
    cold = {**base, "GF_BICGSTAB_WARM_HISTORY": 0}
    if method == "bicgstab-cap":
        return dict(gf_method="bicgstab", reort=None, slater=0.0, env=cold)
    if method == "bicgstab-outer":
        return dict(
            gf_method="bicgstab",
            reort=None,
            slater=0.0,
            env={**cold, "GF_BICGSTAB_ADMISSION": "outer", "GF_BICGSTAB_ADMIT_TOL_AMP": repr(float(eta))},
        )
    if method == "bicgstab-swm":
        return dict(gf_method="bicgstab", reort=None, slater=float(eta), env=cold)
    raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")


def _units(notes):
    """``(max retained size, max seed size, per-unit sizes, cap_hit, n_blocks, per-point profile)``."""
    sizes = [n["retained_size"] for n in notes if n["retained_size"] is not None]
    profile = [p for n in notes for p in n.get("points", [])]
    seeds = [n["seed_size"] for n in notes if n.get("seed_size") is not None] + [p["seed_size"] for p in profile]
    blocks = [n["n_blocks"] for n in notes if n.get("n_blocks") is not None]
    return {
        "size": max(sizes) if sizes else None,
        "seed_size": max(seeds) if seeds else None,
        "unit_sizes": sizes,
        "cap_hit": any(n["cap_hit"] for n in notes),
        "n_blocks": max(blocks) if blocks else None,
        "rebuild_size": max((p["rebuild_size"] for p in profile), default=None),
        "unconverged": sum(1 for p in profile if not p["converged"]),
        # Per-solve bases of a per-frequency method as [axis, k, size] (axis 0 = Matsubara when both
        # are present): the maximum is what a run must hold, the rest shows how many points needed it.
        "point_sizes": [[p["axis"], p["k"], p["solve_size"]] for p in profile],
    }


def _fro(a):
    return np.linalg.norm(a, axis=(-2, -1))


def _errors(aim, mesh, G_mats, G_real, U):
    """``G`` and ``Sigma`` errors on the selected points, and the worst acausality of the real-axis ``Sigma``."""
    out = {
        "dG_mats": None,
        "dSigma_mats": None,
        "dG_real": None,
        "dSigma_real": None,
        "acausal_real": None,
        "dSigma_truth_mats": None,
        "dSigma_truth_real": None,
    }
    axes = []
    if mesh.has_mats:
        axes.append(("mats", G_mats, mesh.ref_mats, mesh.matsubara, mesh.sel_mats, mesh.truth_mats))
    if mesh.has_real:
        axes.append(("real", G_real, mesh.ref_real, mesh.z_real, mesh.sel_real, mesh.truth_real))
    for name, G, ref, z, sel, truth in axes:
        Gs, Rs, zs = G[sel], ref[sel], z[sel]
        out[f"dG_{name}"] = float(np.max(_fro(Gs - Rs)) / np.max(_fro(Rs)))
        G0_inv = free_G_inverse(aim, zs)
        SR = self_energy(Rs, G0_inv)
        try:
            Ss = self_energy(Gs, G0_inv)
        except np.linalg.LinAlgError:
            # A method that loses a seed column (an over-aggressive amplitude cutoff) returns a
            # singular G: that is a failed cell, not a crash, and it must not read as a small error.
            out[f"dSigma_{name}"] = np.inf
            if truth is not None:
                out[f"dSigma_truth_{name}"] = np.inf
            if name == "real":
                out["acausal_real"] = np.inf
            continue
        out[f"dSigma_{name}"] = float(np.max(_fro(Ss - SR) / np.maximum(_fro(SR), 0.1 * U)))
        if truth is not None:
            ST = self_energy(truth[sel], G0_inv)
            out[f"dSigma_truth_{name}"] = float(np.max(_fro(Ss - ST) / np.maximum(_fro(ST), 0.1 * U)))
        if name == "real":
            anti = (Ss - np.conj(np.transpose(Ss, (0, 2, 1)))) / 2j
            out["acausal_real"] = float(max(0.0, np.max(np.linalg.eigvalsh(anti))))
    return out


def run_cell(aim, mesh, method, cap=NON_BINDING, eta=0.0, shell_tol=0.0, comm=None):
    """One ``(method, cap, eta)`` cell: the basis it needed and how wrong it was.

    ``cap`` is the determinant budget (a finite value always: see ``NON_BINDING``); ``eta`` the
    admission threshold of the pruned methods or the ``slaterWeightMin`` of ``bicgstab-swm``;
    ``shell_tol`` > 0 lets the pruned methods drop weak rows of the seeds' first H-shell too
    (``GF_ADMIT_FIRST_SHELL_TOL``; 0 keeps it whole, which is what holds the ``Sigma`` tail exact).
    """
    spec = _knobs(method, cap, eta, shell_tol)
    per_point = spec["gf_method"] == "bicgstab"
    omega = (mesh.omega[mesh.sel_real] if per_point else mesh.omega) if mesh.has_real else None
    mats = (mesh.matsubara[mesh.sel_mats] if per_point else mesh.matsubara) if mesh.has_mats else None
    basis = aim.basis(sorted(aim.gs.keys()), truncation_threshold=cap, comm=comm if comm is not None else MPI.COMM_SELF)
    t0 = time.perf_counter()
    with env(**spec["env"]), solver_trace.tracing() as trace, contextlib.redirect_stdout(io.StringIO()):
        G_mats, G_real, report = get_Greens_function(
            matsubara_mesh=mats,
            omega_mesh=omega,
            psis=[aim.gs],
            es=[aim.e0],
            tau=1e-3,
            basis=basis,
            hOp=aim.hOp,
            delta=mesh.delta,
            blocks=[aim.imp],
            verbose=False,
            verbose_extra=False,
            reort=spec["reort"],
            dN=None,
            occ_cutoff=0.0,
            slaterWeightMin=spec["slater"],
            sparse=True,
            gf_method=spec["gf_method"],
        )
    wall = time.perf_counter() - t0
    notes = trace.of_kind("gf_unit_basis")
    # The per-frequency result is on the selected points already; the Lanczos family's is on the full
    # mesh. Put both on the full mesh's indexing so one scorer serves them.
    full_m = full_r = None
    if mesh.has_mats:
        if per_point:
            full_m = np.full((len(mesh.matsubara),) + G_mats[0].shape[1:], np.nan, dtype=complex)
            full_m[mesh.sel_mats] = G_mats[0]
        else:
            full_m = G_mats[0]
    if mesh.has_real:
        if per_point:
            full_r = np.full((len(mesh.omega),) + G_real[0].shape[1:], np.nan, dtype=complex)
            full_r[mesh.sel_real] = G_real[0]
        else:
            full_r = G_real[0]
    cell = {"method": method, "cap": int(cap), "eta": float(eta), "shell_tol": float(shell_tol), "wall": wall}
    cell.update(_units(notes))
    cell.update(_errors(aim, mesh, full_m, full_r, aim.params.get("U", 8.0)))
    present = [cell[k] for k in ("dSigma_mats", "dSigma_real") if cell[k] is not None]
    cell["dSigma"] = max(present)
    cell["report_severity"] = int(report.worst_severity) if report is not None else None
    return cell


def closure_sizes(aim, mesh, methods=("lanczos-cap", "bicgstab-cap")):
    """The uncapped basis each method needs, measured under a non-binding cap. ``{method: cell}``."""
    return {m: run_cell(aim, mesh, m) for m in methods}


def pareto(cells, metric="dSigma"):
    """Lower envelope of ``(size, error)`` per method: the smallest error reached at or below each size.

    Returns ``{method: [(size, error), ...]}`` sorted by size, keeping only points that improve on
    every smaller one -- the frontier a method can actually reach. Cells that were not converged, or
    that hit no size, are left out rather than plotted as if they were measurements.
    """
    out = {}
    for method in sorted({c["method"] for c in cells}):
        pts = sorted(
            (c["size"], c[metric])
            for c in cells
            if c["method"] == method and c["size"] is not None and c["unconverged"] == 0 and c[metric] is not None
        )
        best, front = np.inf, []
        for size, err in pts:
            if err < best:
                best = err
                front.append((size, err))
        out[method] = front
    return out


def smallest_size_below(front, tol):
    """The smallest size on a frontier whose error is at most ``tol`` (``None`` if none reaches it)."""
    for size, err in front:
        if err <= tol:
            return size
    return None


DEFAULT_CAP_FRACTIONS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.55, 0.7, 0.85)
DEFAULT_ETAS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
DEFAULT_SWM_ETAS = (1e-3, 1e-4, 1e-5, 1e-6)
TOLERANCES = (1e-1, 1e-2, 1e-3, 1e-4)


def run_grid(
    aim,
    mesh,
    cap_fractions=DEFAULT_CAP_FRACTIONS,
    etas=DEFAULT_ETAS,
    swm_etas=DEFAULT_SWM_ETAS,
    tie_shell=True,
    methods=METHODS,
    progress=None,
):
    """The full comparison on one fixture: closure sizes, the cap ladder, and the threshold ladders.

    Caps are fractions of the **measured** uncapped Lanczos basis, so the ladder always spans the
    range that matters. Pruned methods run under a non-binding cap; with ``tie_shell`` their first-shell
    cut equals their threshold (a permanent ban makes the first-shell cut a floor on the error of the
    pruned Lanczos recurrence, so a ladder that moves only ``eta`` would not move the answer).
    ``progress`` is called with each finished cell, for long runs.
    """
    closure = closure_sizes(aim, mesh)
    full = closure["lanczos-cap"]["size"]
    cells = list(closure.values())
    caps = sorted({max(int(f * full), 1) for f in cap_fractions})

    def add(method, **kw):
        if method in methods:
            cell = run_cell(aim, mesh, method, **kw)
            cells.append(cell)
            if progress is not None:
                progress(cell)

    for cap in caps:
        add("lanczos-cap", cap=cap)
        add("bicgstab-cap", cap=cap)
    for eta in etas:
        shell = eta if tie_shell else 0.0
        add("lanczos-pruned", eta=eta, shell_tol=shell)
        add("bicgstab-outer", eta=eta, shell_tol=shell)
    for eta in swm_etas:
        add("bicgstab-swm", eta=eta)
    return {
        "closure": {m: c["size"] for m, c in closure.items()},
        "seed_size": closure["lanczos-cap"]["seed_size"],
        "caps": caps,
        "cells": cells,
        "pareto": pareto(cells),
        "pareto_by": {
            m: pareto(cells, m) for m in ("dSigma_mats", "dSigma_real") if any(c[m] is not None for c in cells)
        },
    }


def _size_table(result, metric, tolerances):
    methods = sorted(result["pareto_by"][metric])
    lines = [
        f"   {metric}: smallest basis (determinants) reaching the tolerance",
        "   " + "tol".ljust(8) + "".join(m.rjust(16) for m in methods),
    ]
    for tol in tolerances:
        row = []
        for m in methods:
            size = smallest_size_below(result["pareto_by"][metric][m], tol)
            row.append(("-" if size is None else f"{size:,}").rjust(16))
        lines.append("   " + f"{tol:.0e}".ljust(8) + "".join(row))
    return lines


def format_report(title, result, tolerances=TOLERANCES):
    """Readable tables: the smallest basis each method needs to reach each Sigma tolerance, per axis.

    The axes are kept apart because they disagree: a method can be the best on the Matsubara axis and
    the worst on the real axis at the same basis size. ``-`` means no cell of that method reached the
    tolerance; cells that did not converge are not counted (``unconverged`` is reported below).
    """
    lines = [
        f"== {title}",
        f"   uncapped basis: {result['closure']}   seed support: {result['seed_size']}",
    ]
    for metric in ("dSigma_mats", "dSigma_real"):
        if metric in result["pareto_by"]:
            lines += _size_table(result, metric, tolerances)
    bad = sorted({c["method"] for c in result["cells"] if c["unconverged"]})
    if bad:
        lines.append(f"   (cells with unconverged per-frequency solves, excluded above: {', '.join(bad)})")
    return "\n".join(lines)
