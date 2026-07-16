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

    return {
        "label": name,
        "h0": h0,
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
        "spin_flip_dj": bool(_attr(attrs, "spin_flip_dj", False)),
        "chain_restrict": bool(_attr(attrs, "chain_restrict", False)),
        "occ_cutoff": float(_attr(attrs, "occ_cutoff", 1e-6)),
        "truncation_threshold": truncation_threshold,
        "slaterWeightMin": float(_attr(attrs, "slater_min", 0.0)),
        "dN": dN,
        "sparse_green": bool(_attr(attrs, "sparse_green", True)),
    }


def _subsample(mesh, n):
    """``n`` evenly spaced points of ``mesh`` (``0`` -> None/axis off, ``None`` -> full mesh)."""
    if n == 0:
        return None
    if n is None or n >= len(mesh):
        return mesh
    return mesh[np.linspace(0, len(mesh) - 1, n).astype(int)]


def run_selfenergy(
    workload,
    comm=None,
    gf_method="lanczos",
    reort="archive",
    truncation_threshold="archive",
    dN="archive",
    n_iw=None,
    n_w=None,
    verbosity=0,
):
    """Re-run ``calc_selfenergy`` on a loaded workload, with benchmark-friendly overrides.

    Parameters
    ----------
    workload : dict
        From :func:`load_workload`.
    gf_method : str
        ``"lanczos"`` or ``"bicgstab"``.
    reort, truncation_threshold, dN
        ``"archive"`` keeps the recorded production setting; anything else overrides it
        (``dN`` bounds the excited-sector occupation window -- FCC Ni production runs
        record ``dN=None``, i.e. no window at all).
    n_iw, n_w : int, optional
        Subsample the Matsubara / real mesh to this many points (``0`` drops the axis
        entirely, ``None`` keeps the full mesh). Point counts scale the per-frequency
        method's wall time ~linearly, so benchmarks usually subsample; the per-point
        *memory* is mesh-size independent.

    Returns
    -------
    dict
        The ``calc_selfenergy`` result dict (rank 0; empty-ish on other ranks).
    """
    from impurityModel.ed.model import BasisOptions, ImpurityModel, Meshes, SolverOptions

    iw = _subsample(workload["iw_mesh"], n_iw)
    w = _subsample(workload["w_mesh"], n_w)
    model = ImpurityModel(
        h0=workload["h0"],
        u4=workload["u4"],
        impurity_orbitals=workload["impurity_orbitals"],
        rot_to_spherical=workload["rot_to_spherical"],
    )
    meshes = Meshes(iw=1j * iw if iw is not None else None, w=w, delta=workload["delta"])
    basis = BasisOptions(
        nominal_occ=workload["nominal_occ"],
        mixed_valence=workload["mixed_valence"],
        dN=workload["dN"] if dN == "archive" else dN,
        truncation_threshold=(
            workload["truncation_threshold"] if truncation_threshold == "archive" else truncation_threshold
        ),
        chain_restrict=workload["chain_restrict"],
        spin_flip_dj=workload["spin_flip_dj"],
        occ_cutoff=workload["occ_cutoff"],
        slater_weight_min=workload["slaterWeightMin"],
        tau=workload["tau"],
    )
    solver = SolverOptions(
        reort=workload["reort"] if reort == "archive" else reort,
        dense_cutoff=workload["dense_cutoff"],
        sparse_green=workload["sparse_green"],
        gf_method=gf_method,
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
