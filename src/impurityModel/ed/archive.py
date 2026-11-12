"""The ``impurityModel_data.h5`` archive: the solver problem of one (cluster, iteration), written and read here.

An embedding code (the RSPt interface) calls :func:`write_selfenergy_group` once per DMFT iteration;
:func:`load_selfenergy_archive` rebuilds the matching ``calc_selfenergy`` call offline.  The writer and
the reader live together so they cannot drift apart, and the group carries :data:`ARCHIVE_VERSION`: a
group written by another version is refused rather than guessed at.  Iteration bookkeeping that is the
embedding code's own (fit reuse, double-counting damping) is not part of this format; the caller adds
its datasets and attributes to the same group.
"""

import numpy as np

from impurityModel.ed import atomic_physics
from impurityModel.ed.lie_algebra import tensors_to_operator
from impurityModel.ed.model import ImpurityModel, _operator_from_matrix
from impurityModel.ed.options import BasisOptions, Meshes, SolverOptions, options_from_attrs, options_to_attrs

__all__ = ["ARCHIVE_VERSION", "load_selfenergy_archive", "read_archive_group", "write_selfenergy_group"]

#: Bumped when the datasets or attributes of a group change meaning.
ARCHIVE_VERSION = 1

_VERSION_ATTR = "archive version"


def _put(group, name, data):
    """Create the dataset ``name`` in ``group``, replacing an existing one (a group can be re-entered)."""
    if name in group:
        del group[name]
    group.create_dataset(name, data=data)


def write_selfenergy_group(
    group,
    *,
    basis,
    solver,
    delta,
    w,
    iw,
    h_solver,
    u4,
    dc,
    rot_to_spherical,
    impurity_orbitals,
    valence_orbitals,
    conduction_orbitals,
    h_dft,
    h_bath,
    v,
):
    """Write the solver problem into an open h5py group.

    Parameters
    ----------
    group : h5py.Group
        The ``"<label> <iteration>"`` group; existing datasets of the same names are replaced.
    basis, solver : BasisOptions, SolverOptions
        The options of the run.  ``basis.nominal_occ`` is ``{0: n}`` for one shell.
    delta : float
        Real-axis broadening.
    w, iw : numpy.ndarray
        Real and Matsubara frequency meshes (``iw`` real: ``z = i * iw``).
    h_solver, u4, dc : numpy.ndarray
        Solver-basis one-particle Hamiltonian (impurity + bath), RSPt-layout Coulomb tensor and
        double counting.
    rot_to_spherical : numpy.ndarray
        Solver basis to spherical harmonics.
    impurity_orbitals, valence_orbitals, conduction_orbitals : sequence of int
        Orbital layout of ``h_solver``.
    h_dft, h_bath, v : numpy.ndarray
        The blocks ``h_solver`` was assembled from (kept for inspection).
    """
    group.attrs[_VERSION_ATTR] = ARCHIVE_VERSION
    group.attrs["tau"] = basis.tau
    group.attrs["delta"] = delta
    group.attrs["nominal occupation"] = basis.nominal_occ[0]
    # h5py cannot store None: an unset option is an absent attribute.
    for key, value in options_to_attrs(basis, solver).items():
        if value is not None:
            group.attrs[key] = value
        elif key in group.attrs:
            del group.attrs[key]
    _put(group, "Real frequency mesh", w)
    _put(group, "Matsubara frequency mesh", iw)
    _put(group, "Rot to spherical", rot_to_spherical)
    _put(group, "Impurity orbitals", impurity_orbitals)
    _put(group, "Valence orbitals", valence_orbitals)
    _put(group, "Conduction orbitals", conduction_orbitals)
    _put(group, "H DFT", h_dft)
    _put(group, "H bath", h_bath)
    _put(group, "V", v)
    _put(group, "DC", dc)
    _put(group, "U", u4)
    _put(group, "H solver", h_solver)


def read_archive_group(path, cluster=None, iteration=None) -> dict:
    """Parse one ``(cluster, iteration)`` group of an archive.

    Returns the raw physics arrays (``h0`` operator dict, ``u4``, ``dc``, ``impurity_orbitals``,
    ``rot_to_spherical``, ``n_spin_orbitals``), both frequency meshes, the recorded option
    keyword arguments (``basis_kwargs``, ``solver_kwargs``) and the group ``label``.

    Raises
    ------
    ValueError
        The file has no such group, or the group was not written by this archive version.
    """
    import h5py

    with h5py.File(path, "r") as f:
        if iteration is None:
            iteration = int(f.attrs.get("last iteration", 1))
        if cluster is None:
            labels = sorted(f.keys())
            if not labels:
                raise ValueError(f"{path}: archive holds no cluster groups")
            cluster = labels[0].rsplit(" ", 1)[0]
        name = f"{cluster} {iteration}"
        if name not in f:
            raise ValueError(f"{path}: no group {name!r}; available: {sorted(f.keys())}")
        g = f[name]
        attrs = dict(g.attrs)
        version = attrs.get(_VERSION_ATTR)
        if version is None or int(version) != ARCHIVE_VERSION:
            raise ValueError(
                f"{path}: group {name!r} has archive version {version!r}, this impurityModel reads version "
                f"{ARCHIVE_VERSION}. Re-run the calculation to write the archive again."
            )
        h_solver = np.asarray(g["H solver"])
        u4_rspt = np.asarray(g["U"])
        dc_matrix = np.asarray(g["DC"]) if "DC" in g else None
        iw_mesh = np.asarray(g["Matsubara frequency mesh"])
        w_mesh = np.asarray(g["Real frequency mesh"])
        rot_to_spherical = np.asarray(g["Rot to spherical"])
        impurity_indices = [int(i) for i in np.asarray(g["Impurity orbitals"])]
        # Multi-shell archives only; absent means one shell inferred from the column count.
        shell_source = g["Spherical shell l"] if "Spherical shell l" in g else attrs.get("Spherical shell l")
        shell_ls = tuple(int(l) for l in np.atleast_1d(np.asarray(shell_source))) if shell_source is not None else None
        occ_source = (
            g["nominal occupation per shell"]
            if "nominal occupation per shell" in g
            else attrs.get("nominal occupation per shell")
        )
        nominal_occ = (
            {a: int(n) for a, n in enumerate(np.atleast_1d(np.asarray(occ_source)))}
            if occ_source is not None
            else {0: int(attrs["nominal occupation"])}
        )

    basis_kwargs, solver_kwargs = options_from_attrs(attrs)
    basis_kwargs["nominal_occ"] = nominal_occ
    basis_kwargs["tau"] = float(attrs["tau"])
    return {
        "label": name,
        "h0": _operator_from_matrix(h_solver),
        "u4": atomic_physics.getUop_from_rspt_u4(u4_rspt),
        "dc": tensors_to_operator(dc_matrix).to_dict() if dc_matrix is not None else None,
        "n_spin_orbitals": int(h_solver.shape[0]),
        "impurity_orbitals": {0: impurity_indices},
        "rot_to_spherical": rot_to_spherical,
        # Meshes are stored real-valued; the Matsubara mesh is z = i * nu (see Meshes.iw).
        "iw_mesh": iw_mesh if iw_mesh.size else None,
        "w_mesh": w_mesh if w_mesh.size else None,
        "shell_ls": shell_ls,
        "delta": float(attrs["delta"]),
        "basis_kwargs": basis_kwargs,
        "solver_kwargs": solver_kwargs,
    }


def load_selfenergy_archive(path, cluster=None, iteration=None):
    """Reconstruct a full ``calc_selfenergy`` call from an ``impurityModel_data.h5`` archive.

    Returns ``(model, meshes, basis, solver, cluster_label)`` -- the :class:`ImpurityModel` together
    with the frequency meshes and the recorded basis/solver options, so an archived DFT-embedded
    self-energy run can be reproduced offline (see :meth:`ImpurityModel.from_hdf5` for the
    physics-only view).

    Parameters
    ----------
    path : str or path-like
        Path to the archive.
    cluster : str, optional
        Cluster label; defaults to the first group's label.
    iteration : int, optional
        DMFT iteration; defaults to the archive's ``last iteration`` attribute.

    Returns
    -------
    (ImpurityModel, Meshes, BasisOptions, SolverOptions, str)
    """
    raw = read_archive_group(path, cluster, iteration)
    model = ImpurityModel(
        h0=raw["h0"],
        dc=raw["dc"],
        u4=raw["u4"],
        impurity_orbitals=raw["impurity_orbitals"],
        rot_to_spherical=raw["rot_to_spherical"],
        n_spin_orbitals=raw["n_spin_orbitals"],
        shell_ls=raw["shell_ls"],
    )
    meshes = Meshes(
        iw=1j * raw["iw_mesh"] if raw["iw_mesh"] is not None else None,
        w=raw["w_mesh"],
        delta=raw["delta"],
    )
    return model, meshes, BasisOptions(**raw["basis_kwargs"]), SolverOptions(**raw["solver_kwargs"]), raw["label"]
