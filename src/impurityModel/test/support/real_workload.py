"""Reconstruct a production ``calc_selfenergy`` call from an ``impurityModel_data.h5`` archive.

``impurityModel_interface.lib`` writes one group per (cluster, DMFT iteration) into
``impurityModel_data.h5`` immediately before its ``calc_selfenergy`` call: the full
one-particle solver Hamiltonian (``H solver``, impurity + fitted bath, CF basis), the
Coulomb tensor ``U``, both frequency meshes, the orbital index splits, the
``Rot to spherical`` matrix and every solver option as group attributes. That makes the
archive self-contained for reproducing the exact solver run -- which is what this module
does (see ``doc/plans/bicgstab_per_frequency_gf.md``, Phase 3a-quinquies: benchmarks must
run on these real workloads, never on synthetic anchors whose Green's function is constant
on the evaluation mesh).

Typical use (opt-in benchmarks, e.g. ``test_bicgstab_gf_real_workload.py``)::

    wl = load_workload("impmod_tests/FCC_Ni/impmod/.../impurityModel_data.h5")
    result = run_selfenergy(wl, gf_method="bicgstab", n_iw=64, n_w=0, comm=comm)

The archives live outside this repository (``impmod_tests``); everything here fails with a
clear message when the file is missing, so the callers can skip cleanly.
"""

import numpy as np

from impurityModel.ed.selfenergy import calc_selfenergy


def _attr(attrs, key, default=None):
    """Group attribute with the interface's ``None``-as-string convention undone."""
    value = attrs.get(key, default)
    if isinstance(value, (str, bytes)) and str(value) == "None":
        return None
    return value


#: Archive attributes this module reads and passes to the solver. Anything the archive records that is in
#: neither this set nor ``_BATH_FIT_ONLY`` is reported as ``ignored_attrs``: a replay that silently drops an
#: option solves a different problem (the excitation budget, 8 in the archive and 4 by library default, once
#: restricted a whole replay's ground state to 169k instead of 277k determinants).
_CONSUMED_ATTRS = frozenset(
    {
        "nominal occupation",
        "mv",
        "tau",
        "delta",
        "reort",
        "dense_cutoff",
        "chain_restrict",
        "occ_cutoff",
        "truncation_threshold",
        "slater_min",
        "dN",
        "sparse_green",
        "gf_tol",
        "gf_real_tol",
        "gf_method",
        "gf_admission",
        "gf_admit_tol",
        "gf_min_weight",
        "excitation_budget",
        "e_pt2_tol",
        "de2_min",
    }
)
#: Attributes that describe the bath fit that produced ``H solver`` (or the run's provenance), not the solver
#: call: the fitted result is the dataset, so a replay does not need them.
_BATH_FIT_ONLY = frozenset(
    {
        "bath_geometry",
        "collapse_chains",
        "fit_unocc",
        "freeze_bath_energies",
        "gamma",
        "n_baths",
        "weight",
        "weight_function",
        "weight_w0",
        "solver line",
        "DC damping alpha",
        "DC ground state sector",
        "DC guess fingerprint",
        "impurityModel version",
        "impurityModel_interface version",
        "rspt2spectra version",
    }
)


def _optional(attrs, key, cast):
    """``attrs[key]`` cast, or ``None`` when absent or recorded as the string ``"None"``."""
    value = _attr(attrs, key)
    return None if value is None else cast(value)


def load_workload(h5_path, cluster=None, iteration=None):
    """Load one (cluster, iteration) group of an ``impurityModel_data.h5`` archive.

    Parameters
    ----------
    h5_path : str or Path
        Path to the archive.
    cluster : str, optional
        Cluster label (e.g. ``"Ni"``); defaults to the label of the first group.
    iteration : int, optional
        DMFT iteration; defaults to the archive's ``last iteration`` attribute.

    Returns
    -------
    dict
        Keyword arguments for :func:`run_selfenergy` /
        :func:`impurityModel.ed.selfenergy.calc_selfenergy`: the solver Hamiltonian as an
        operator dict (``h0``), ``u4``, the raw meshes (``iw_mesh`` real-valued as stored,
        ``w_mesh``), ``rot_to_spherical``, ``impurity_orbitals``, and every solver option
        the interface recorded (``tau``, ``delta``, ``nominal_occ``, ``reort``, ``dN``,
        ``chain_restrict``, ...). ``label`` names the group it came from.
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        if cluster is None or iteration is None:
            labels = sorted(f.keys())
            if not labels:
                raise ValueError(f"{h5_path}: archive holds no cluster groups")
        if iteration is None:
            iteration = int(f.attrs.get("last iteration", 1))
        if cluster is None:
            cluster = labels[0].rsplit(" ", 1)[0]
        name = f"{cluster} {iteration}"
        if name not in f:
            raise ValueError(f"{h5_path}: no group {name!r}; available: {sorted(f.keys())}")
        g = f[name]
        attrs = dict(g.attrs)

        h_solver = np.asarray(g["H solver"])
        # "DC" is a newer archive dataset (mirrors model.py's _read_archive_group for the same
        # impurityModel_data.h5 schema); older archives lack it and dc stays None (no behavior
        # change). Assumes "H solver" does NOT already have the DC subtracted -- unverified
        # against a real interface-written archive, since none is available in this repo.
        dc = np.asarray(g["DC"]) if "DC" in g else None
        u4 = np.asarray(g["U"])
        iw_mesh = np.asarray(g["Matsubara frequency mesh"])
        w_mesh = np.asarray(g["Real frequency mesh"])
        rot_to_spherical = np.asarray(g["Rot to spherical"])
        impurity_indices = [int(i) for i in np.asarray(g["Impurity orbitals"])]

    # The interface hands calc_selfenergy the Hamiltonian as a second-quantized operator.
    h0 = {}
    for i, j in zip(*np.nonzero(h_solver)):
        h0[((int(i), "c"), (int(j), "a"))] = complex(h_solver[i, j])

    truncation_threshold = _attr(attrs, "truncation_threshold")
    if truncation_threshold is not None:
        truncation_threshold = float(truncation_threshold)
    dN = _attr(attrs, "dN")
    if dN is not None:
        dN = int(dN)
    gf_tol = _attr(attrs, "gf_tol")
    gf_real_tol = _attr(attrs, "gf_real_tol")

    return {
        # What the archive recorded and this module does not consume (see _CONSUMED_ATTRS): empty for a
        # faithful replay. Callers that must not solve a different problem should refuse a non-empty list.
        "ignored_attrs": sorted(set(attrs) - _CONSUMED_ATTRS - _BATH_FIT_ONLY),
        "excitation_budget": _optional(attrs, "excitation_budget", int),
        "e_pt2_tol": _optional(attrs, "e_pt2_tol", float),
        "de2_min": _optional(attrs, "de2_min", float),
        "gf_admission": _optional(attrs, "gf_admission", str),
        "gf_admit_tol": _optional(attrs, "gf_admit_tol", float),
        "gf_min_weight": _optional(attrs, "gf_min_weight", float),
        "gf_method": _optional(attrs, "gf_method", str),
        "label": name,
        "h0": h0,
        "dc": dc,
        "u4": u4,
        "iw_mesh": iw_mesh,
        "w_mesh": w_mesh,
        "rot_to_spherical": rot_to_spherical,
        "impurity_orbitals": {0: impurity_indices},
        "nominal_occ": {0: int(attrs["nominal occupation"])},
        "mixed_valence": _attr(attrs, "mv"),
        "tau": float(attrs["tau"]),
        "delta": float(attrs["delta"]),
        "reort": _attr(attrs, "reort"),
        "dense_cutoff": int(_attr(attrs, "dense_cutoff", 1000)),
        "chain_restrict": bool(_attr(attrs, "chain_restrict", False)),
        "occ_cutoff": float(_attr(attrs, "occ_cutoff", 1e-6)),
        "truncation_threshold": truncation_threshold,
        "slaterWeightMin": float(_attr(attrs, "slater_min", 0.0)),
        "dN": dN,
        "sparse_green": bool(_attr(attrs, "sparse_green", True)),
        "gf_tol": None if gf_tol is None else float(gf_tol),
        "gf_real_tol": None if gf_real_tol is None else float(gf_real_tol),
    }


def _subsample(mesh, n):
    """``n`` evenly spaced points of ``mesh`` (``0`` -> None/axis off, ``None`` -> full mesh)."""
    if n == 0:
        return None
    if n is None or n >= len(mesh):
        return mesh
    return mesh[np.linspace(0, len(mesh) - 1, n).astype(int)]


def build_options(
    workload,
    gf_method=None,
    reort="archive",
    truncation_threshold="archive",
    dN="archive",
    n_iw=None,
    n_w=None,
    gf_min_weight=None,
    gf_tol=None,
    gf_real_tol=None,
):
    """The ``(model, meshes, basis_options, solver_options)`` a replay of ``workload`` hands to ``calc_selfenergy``.

    Split out of :func:`run_selfenergy` so a driver can print exactly what it is about to solve before it
    starts (and diff that against the production run's option table). Every solver-relevant attribute the
    archive recorded is passed on, the excitation budget included; an attribute the archive does not record
    is left to the library default. Arguments as in :func:`run_selfenergy`; ``gf_min_weight=None`` takes the
    archive's value (``None`` when unrecorded), and so does ``gf_method=None`` (``"lanczos"`` when unrecorded):
    an archive written by a ``bicgstab`` run is not silently replayed as Lanczos.
    """
    if gf_method is None:
        gf_method = workload.get("gf_method") or "lanczos"
    from impurityModel.ed import atomic_physics
    from impurityModel.ed.lie_algebra import tensors_to_operator
    from impurityModel.ed.model import BasisOptions, ImpurityModel, Meshes, SolverOptions

    iw = _subsample(workload["iw_mesh"], n_iw)
    w = _subsample(workload["w_mesh"], n_w)
    dc = workload["dc"]
    model = ImpurityModel(
        h0=workload["h0"],
        dc=tensors_to_operator(np.asarray(dc, dtype=complex)).to_dict() if dc is not None else None,
        u4=atomic_physics.getUop_from_rspt_u4(workload["u4"]),
        impurity_orbitals=workload["impurity_orbitals"],
        rot_to_spherical=workload["rot_to_spherical"],
    )
    meshes = Meshes(iw=1j * iw if iw is not None else None, w=w, delta=workload["delta"])
    basis_kwargs = dict(
        nominal_occ=workload["nominal_occ"],
        mixed_valence=workload["mixed_valence"],
        dN=workload["dN"] if dN == "archive" else dN,
        truncation_threshold=(
            workload["truncation_threshold"] if truncation_threshold == "archive" else truncation_threshold
        ),
        chain_restrict=workload["chain_restrict"],
        occ_cutoff=workload["occ_cutoff"],
        slater_weight_min=workload["slaterWeightMin"],
        tau=workload["tau"],
    )
    # Only what the archive recorded: BasisOptions' own defaults (excitation budget 4, residual-PT2 tolerance)
    # apply to a field it does not carry, exactly as for a run that never set it.
    for key in ("excitation_budget", "e_pt2_tol", "de2_min"):
        if workload.get(key) is not None:
            basis_kwargs[key] = workload[key]
    basis = BasisOptions(**basis_kwargs)
    solver_kwargs = dict(
        reort=workload["reort"] if reort == "archive" else reort,
        dense_cutoff=workload["dense_cutoff"],
        sparse_green=workload["sparse_green"],
        gf_method=gf_method,
        gf_min_weight=workload.get("gf_min_weight") if gf_min_weight is None else gf_min_weight,
        gf_tol=workload["gf_tol"] if gf_tol == "archive" else gf_tol,
        gf_real_tol=workload["gf_real_tol"] if gf_real_tol == "archive" else gf_real_tol,
    )
    for key in ("gf_admission", "gf_admit_tol"):
        if workload.get(key) is not None:
            solver_kwargs[key] = workload[key]
    return model, meshes, basis, SolverOptions(**solver_kwargs)


def run_selfenergy(
    workload,
    comm=None,
    gf_method=None,
    reort="archive",
    truncation_threshold="archive",
    dN="archive",
    n_iw=None,
    n_w=None,
    verbosity=0,
    gf_min_weight=None,
    gf_tol=None,
    gf_real_tol=None,
):
    """Re-run ``calc_selfenergy`` on a loaded workload, with benchmark-friendly overrides.

    Parameters
    ----------
    workload : dict
        From :func:`load_workload`.
    gf_method : str, optional
        ``"lanczos"`` or ``"bicgstab"``; ``None`` (default) takes the archive's recorded kernel.
    reort, truncation_threshold, dN
        ``"archive"`` keeps the recorded production setting; anything else overrides it
        (``dN`` bounds the excited-sector occupation window -- FCC Ni production runs
        record ``dN=None``, i.e. no window at all).
    n_iw, n_w : int, optional
        Subsample the Matsubara / real mesh to this many points (``0`` drops the axis
        entirely, ``None`` keeps the full mesh). Point counts scale the per-frequency
        method's wall time ~linearly, so benchmarks usually subsample; the per-point
        *memory* is mesh-size independent.
    gf_min_weight : float, optional
        ``SolverOptions.gf_min_weight``: drop thermal manifolds below this Boltzmann weight.
        ``None`` takes the archive's recorded value (``None``, keep every state, when unrecorded).
    gf_tol, gf_real_tol : float or ``"archive"``, optional
        The Lanczos convergence tolerances (``SolverOptions.gf_tol`` / ``gf_real_tol``). ``None``
        (default) leaves them to the knobs and defaults, as every earlier caller got;
        ``"archive"`` takes the recorded production value (the real-axis tolerance sets the Lanczos
        depth of a production run, so a faithful replay needs it).

    Returns
    -------
    dict
        The ``calc_selfenergy`` result dict (rank 0; empty-ish on other ranks).

    Notes
    -----
    The excitation budget, residual-PT2 tolerances and GF admission options the archive recorded are passed
    on (see :func:`build_options`); before this was done a replay used the library default budget of 4 where
    a production run had used 8, and solved a ground state 0.61x the size with an E0 1.2 meV too high.
    """
    model, meshes, basis, solver = build_options(
        workload,
        gf_method=gf_method,
        reort=reort,
        truncation_threshold=truncation_threshold,
        dN=dN,
        n_iw=n_iw,
        n_w=n_w,
        gf_min_weight=gf_min_weight,
        gf_tol=gf_tol,
        gf_real_tol=gf_real_tol,
    )
    return calc_selfenergy(
        model,
        meshes,
        basis,
        solver,
        comm=comm,
        verbosity=verbosity,
        cluster_label=workload["label"],
    )
