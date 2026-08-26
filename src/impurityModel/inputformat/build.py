"""Turn a resolved input file into the arguments each driver actually takes.

This is the only module of the package that sits above the solver: it imports
:mod:`impurityModel.ed.model` and the calculation drivers, so everything else stays
importable from outside (notably from ``impurityModel_interface``).

It builds nothing itself. Every model is constructed through the existing single dispatch
point, :func:`impurityModel.ed.model.load_model`, or one of ``ImpurityModel``'s classmethods,
so the TOML front-end cannot drift from the argparse one in what it produces -- only in how
it is spelled.

The drivers do **not** take a uniform argument tuple, and pretending otherwise loses things:
``run_spectra`` takes no ``Meshes`` and no ``SolverOptions``, and
``calc_susceptibility_workflow`` takes its Matsubara count and eigenstate count as separate
scalars. So this returns a per-calculation bundle shaped like the driver it feeds.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from impurityModel.ed import h0_format
from impurityModel.inputformat.reader import InputError

__all__ = ["Built", "build", "deduce_bath_counts", "read_h0_header"]


@dataclass
class Built:
    """Everything a driver needs, plus what had to be deduced to get there.

    Attributes
    ----------
    model : impurityModel.ed.model.ImpurityModel
        The impurity problem.
    basis : impurityModel.ed.model.BasisOptions
    solver : impurityModel.ed.model.SolverOptions
    spectra : impurityModel.ed.model.SpectraOptions or None
        Only for a spectroscopy run.
    meshes : impurityModel.ed.model.Meshes or None
        Only for a self-energy or susceptibility run.
    extra : dict
        Driver-specific scalars that are not part of any option group (cluster label, output
        filename, Matsubara counts, eigenstate count).
    notes : list of str
        What was deduced rather than stated, and where from. Reported on rank 0, and written
        into the provenance record, so no deduction is silent.
    """

    model: Any
    basis: Any
    solver: Any
    spectra: Optional[Any] = None
    meshes: Optional[Any] = None
    extra: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def read_h0_header(resolved):
    """Parse the ``.h0`` header when the Hamiltonian is one, else return ``None``.

    Only the self-describing flat format carries a header. A legacy ``.pickle``/``.json``/
    ``.dat`` records no layout at all, which is why every count it needs must be written out
    by hand.
    """
    if resolved.hamiltonian_source != "file":
        return None
    path = resolved.tables["hamiltonian.file"]["path"]
    if not h0_format.is_h0_format(path):
        return None
    return h0_format.read_h0_file(path)


def _bath_indices(header):
    """Bath spin-orbital indices: everything the header does not call an impurity orbital."""
    impurity = {index for orbitals in header.impurity_orbitals.values() for index in orbitals}
    return [index for index in range(header.n_orb) if index not in impurity]


def _valence_from_onsite_energies(header, bath):
    """Classify bath orbitals by the sign of their on-site energy.

    Mirrors the rule :func:`impurityModel.ed.solver_basis.classify_bath_occupation` applies --
    below the Fermi level (``h[o, o] < 0``) is valence, i.e. initially occupied. Read straight
    off the operator's diagonal rather than by densifying, since only a count is wanted here;
    the self-energy path re-derives the authoritative split for itself regardless.
    """
    return [index for index in bath if np.real(header.h0.get(((index, "c"), (index, "a")), 0.0)) < 0.0]


def deduce_bath_counts(resolved, header, notes):
    """Fill in each shell's ``n_bath`` / ``n_valence_bath``, reporting how.

    The ladder, in order:

    1. The header names *which* shell it describes (``impurity_l``). That shell owns the
       stored bath; **every other shell gets zero**, because a shell the file does not
       describe has no fitted bath. For a core shell that is essentially always right -- every
       shipped run script passes ``--nBaths 0 <n>`` -- and it stays settable for the
       exceptional case where core bath states really are wanted.
    2. The valence/conduction split comes from the header's ``valence_bath`` /
       ``conduction_bath`` lists when present. A real hybridization fit is not necessarily a
       contiguous prefix, so the lists beat any positional assumption.
    3. When they are absent -- producers omit them for a non-star geometry, where each site is
       a Lanczos combination of star modes and the labels would mislead -- fall back to the
       on-site energy sign, and *say so*.

    An explicitly written count that disagrees with the deduction is an error, not an
    override: silently preferring one would make the file and the model disagree about the
    same number.
    """
    shells = resolved.shells
    valence_shell = next(shell for shell in shells if shell["role"] == "valence")

    if header is None:
        # No header to deduce from. The correlated shell's counts have to be written out --
        # a legacy labelled file, a crystal-field parametrisation and a bare matrix all record
        # no bath layout. Every OTHER shell still defaults to zero: a shell whose Hamiltonian
        # was never read has no fitted bath, which for a core shell is the normal case (every
        # shipped run script passes `--nBaths 0 <n>`). Writing a non-zero count stays possible
        # for the exceptional case where core bath states really are wanted.
        for shell in shells:
            if shell is valence_shell:
                for key in ("n_bath", "n_valence_bath"):
                    if shell.get(key) is None:
                        raise InputError(
                            f'[[shell]] l={shell["l"]}: {key} is required here. '
                            f"[hamiltonian.{resolved.hamiltonian_source}] records no bath "
                            "layout, so there is nothing to deduce it from; only a "
                            "self-describing .h0 carries one."
                        )
                continue
            for key in ("n_bath", "n_valence_bath"):
                if shell.get(key) is None:
                    shell[key] = 0
            if shell["n_bath"] == 0:
                notes.append(
                    f'l={shell["l"]} (role {shell["role"]}): no bath states -- no Hamiltonian '
                    "was read for this shell."
                )
        return

    bath = _bath_indices(header)
    described = header.impurity_l if header.impurity_l is not None else valence_shell["l"]
    if header.impurity_l is None:
        notes.append(
            f"The .h0 header does not record impurity_l; assuming its {len(bath)} bath "
            f"orbitals belong to the valence shell (l={valence_shell['l']})."
        )

    if header.valence_bath is not None:
        valence = list(header.valence_bath)
        source = "the header's valence_bath list"
    else:
        valence = _valence_from_onsite_energies(header, bath)
        geometry = header.bath_geometry or "unrecorded"
        source = (
            f"the sign of the bath on-site energies (the header records no valence_bath, "
            f"which producers omit for a non-star geometry; this file records {geometry})"
        )

    for shell in shells:
        described_here = shell["l"] == described
        deduced = {
            "n_bath": len(bath) if described_here else 0,
            "n_valence_bath": len(valence) if described_here else 0,
        }
        for key, value in deduced.items():
            written = shell.get(key)
            if written is None:
                shell[key] = value
            elif written != value:
                raise InputError(
                    f'[[shell]] l={shell["l"]}: {key} = {written}, but the .h0 file gives '
                    f"{value}. A written count that disagrees with the file is an error, not "
                    "an override -- remove it, or use a Hamiltonian that matches."
                )
        if described_here:
            notes.append(
                f"l={shell['l']}: {shell['n_bath']} bath orbitals from the .h0 header, "
                f"{shell['n_valence_bath']} of them valence, from {source}."
            )
        elif deduced["n_bath"] == 0 and shell.get("n_bath") == 0:
            notes.append(f"l={shell['l']}: no bath states (the .h0 file does not describe this shell).")


def check_header_agreement(resolved, header, notes):
    """Cross-check what the file *says* against what the header *guarantees*.

    A disagreement is an error, never an override. The alternative -- letting the written
    value win -- leaves the input file and the Hamiltonian describing different models while
    both look right, which is precisely the class of mistake a self-describing header exists
    to make impossible. This extends to every header key the three cross-checks already inside
    ``from_shells`` cover only a subset of.
    """
    table = resolved.tables.get("hamiltonian.file", {})

    if table.get("unit") is not None and header is None:
        raise InputError(
            "[hamiltonian.file].unit is only meaningful for a self-describing .h0, and nothing "
            "in the reader scales a legacy pickle/.dat/.json amplitude. Convert the file to "
            ".h0 (which records its own unit) rather than declaring one here."
        )
    if header is None:
        return

    declared_unit = table.get("unit")
    if declared_unit is not None and declared_unit != header.header.get("unit"):
        raise InputError(
            f"[hamiltonian.file].unit says {declared_unit!r} but the .h0 header says "
            f"{header.header.get('unit')!r}. The header wins and the disagreement is an error, "
            "never a silent override."
        )

    declared_soc = table.get("contains_soc")
    if declared_soc is not None and header.contains_soc is not None and declared_soc != header.contains_soc:
        raise InputError(
            f"[hamiltonian.file].contains_soc says {declared_soc} but the .h0 header says " f"{header.contains_soc}."
        )

    declared_reference = table.get("energy_reference")
    if declared_reference is not None and declared_reference != header.energy_reference:
        raise InputError(
            f"[hamiltonian.file].energy_reference says {declared_reference!r} but the header "
            f"says {header.energy_reference!r}."
        )
    if header.energy_reference != "fermi" and (
        resolved.dc_scheme not in ("none", "mlft") or resolved.calculation != "spectroscopy"
    ):
        raise InputError(
            f"The .h0 header records energy_reference = {header.energy_reference!r}, but this "
            "run needs a Fermi-referenced Hamiltonian: the bath valence/conduction split is "
            "taken from the sign of the on-site energies and the reference filling from "
            "mu_chem = 0, so an offset zero silently re-partitions the bath into a different "
            "model rather than shifting the same one."
        )

    valence_shell = next(shell for shell in resolved.shells if shell["role"] == "valence")
    if header.impurity_l is not None and header.impurity_l != valence_shell["l"]:
        raise InputError(
            f"The .h0 header describes an l={header.impurity_l} shell, but the [[shell]] with "
            f'role = "valence" is l={valence_shell["l"]}.'
        )
    if header.fermi_energy is not None:
        notes.append(f"Hamiltonian is referenced to a Fermi level of {header.fermi_energy:.6g} eV.")
    if header.producer:
        notes.append(f"Hamiltonian written by {header.producer}.")


def _mesh_array(mesh):
    """Materialise a normalised mesh declaration as a numpy array."""
    if mesh["kind"] == "uniform":
        return np.linspace(mesh["min"], mesh["max"], mesh["n"])
    if mesh["kind"] == "values":
        return np.asarray(mesh["values"], dtype=float)
    return np.loadtxt(mesh["file"]).reshape(-1)


def _matrix_array(spec, size=None):
    """Materialise a normalised matrix declaration as a complex numpy array."""
    kind = spec["kind"]
    if kind == "dense":
        real = np.asarray(spec["real"], dtype=float)
        imag = np.zeros_like(real) if spec["imag"] is None else np.asarray(spec["imag"], dtype=float)
        return real + 1j * imag
    if kind == "identity":
        if size is None:
            raise InputError("identity = true needs a size, which this position does not imply")
        return np.eye(size, dtype=complex)
    if kind == "diagonal":
        return np.diag(np.asarray(spec["diagonal"], dtype=complex))
    if kind == "scalar":
        if size is None:
            raise InputError("scalar needs a size, which this position does not imply")
        return spec["scalar"] * np.eye(size, dtype=complex)
    path = spec["path"]
    if str(path).endswith(".npy"):
        return np.asarray(np.load(path), dtype=complex)
    raw = np.loadtxt(path)
    if spec.get("columns") == "re im" and raw.ndim == 2 and raw.shape[1] == 2:
        flat = raw[:, 0] + 1j * raw[:, 1]
        side = int(round(np.sqrt(flat.size)))
        return flat.reshape(side, side)
    return np.asarray(raw, dtype=complex)


def _slater_arrays(resolved, core_l, valence_l):
    """Slater-Condon arrays, with the lengths their angular momenta imply.

    The expected lengths are ``slater_condon_Uop``'s own contract (``2*l+1`` for a same-shell
    F, ``2*l_c+2`` for the core-valence G), derived here from the declared shells rather than
    restated in the file -- so a mismatch is caught with a message naming both numbers instead
    of surfacing as an assertion deep inside the Coulomb assembly.
    """
    if resolved.interaction_kind != "slater":
        return None
    table = resolved.tables["interaction.slater"]
    expected = {"F_vv": 2 * valence_l + 1}
    if core_l is not None:
        expected.update({"F_cc": 2 * core_l + 1, "F_cv": 2 * core_l + 1, "G_cv": 2 * core_l + 2})
    for name, length in expected.items():
        value = table.get(name)
        if value is None:
            continue
        if len(value) != length:
            raise InputError(
                f"[interaction.slater].{name} has {len(value)} entries, but an "
                f"l_core={core_l} / l_valence={valence_l} pair needs {length}."
            )
    return table


#: The symmetry-breaking nudge the spectroscopy command line has always applied to the
#: correlated shell. It is not ``from_shells``' own default -- that is a zero field -- so it
#: has to be supplied here for a TOML run to reproduce the equivalent command line.
SPECTROSCOPY_DEFAULT_ZEEMAN = (0.0, 0.0, 0.0001)


def _spectroscopy_zeeman(valence, notes):
    """Zeeman splitting for a spectroscopy run, defaulting to the historical nudge.

    Omitting the key means "whatever this path has always done", which here is a tiny field
    that lifts the spin degeneracy so the solver picks a definite state rather than an
    arbitrary mixture. An explicit ``[0, 0, 0]`` is a different request and is honoured as
    written.
    """
    if valence["zeeman_splitting"] is None:
        notes.append(
            f"No zeeman_splitting given; using the spectroscopy default "
            f"{SPECTROSCOPY_DEFAULT_ZEEMAN} (a symmetry-breaking nudge, in eV)."
        )
        return SPECTROSCOPY_DEFAULT_ZEEMAN
    return tuple(valence["zeeman_splitting"])


def _build_model(resolved, header, notes, rank, verbose):
    """Construct the ImpurityModel through the existing dispatch points, never around them."""
    from impurityModel.ed.get_spectra import build_spectra_model
    from impurityModel.ed.model import ImpurityModel, load_model, load_selfenergy_archive

    shells = resolved.shells
    valence = next(shell for shell in shells if shell["role"] == "valence")
    core = next((shell for shell in shells if shell["role"] == "core"), None)
    slater = _slater_arrays(resolved, None if core is None else core["l"], valence["l"])

    source = resolved.hamiltonian_source
    if source == "archive":
        table = resolved.tables["hamiltonian.archive"]
        model, meshes, basis, solver, cluster = load_selfenergy_archive(
            table["path"], cluster=table["cluster"], iteration=table["iteration"]
        )
        notes.append(f"Model, meshes and recorded options taken from the archive (cluster {cluster!r}).")
        return model, {"archive": (meshes, basis, solver, cluster)}

    if source in ("blocks", "matrix"):
        if source == "blocks":
            table = resolved.tables["hamiltonian.blocks"]
            h_imp = _matrix_array(table["h_imp"])
            model = ImpurityModel.from_blocks(
                h_imp,
                _matrix_array(table["v"]),
                _matrix_array(table["h_bath"]),
                rot_to_spherical=np.eye(h_imp.shape[0], dtype=complex),
            )
        else:
            table = resolved.tables["hamiltonian.matrix"]
            n_imp = table["n_impurity_orbitals"]
            model = ImpurityModel.from_solver_matrix(
                _matrix_array(table["h"]),
                n_imp,
                rot_to_spherical=np.eye(n_imp, dtype=complex),
            )
        return model, {}

    if resolved.calculation == "spectroscopy":
        if source != "file":
            raise InputError(
                f"[hamiltonian.{source}] cannot drive a spectroscopy run: the multi-shell "
                "interacting assembly reads its Hamiltonian from a file."
            )
        # Core first, then valence -- the order get_hamiltonian_operator's own c2i indexing
        # assumes, and the order the equivalent command line spells as `--ls 1 2`.
        ordered = [shell for shell in (core, valence) if shell is not None]
        return (
            build_spectra_model(
                resolved.tables["hamiltonian.file"]["path"],
                tuple(shell["l"] for shell in ordered),
                tuple(shell["n_bath"] for shell in ordered),
                tuple(shell["n_valence_bath"] for shell in ordered),
                tuple(shell["nominal_occupation"] for shell in ordered),
                tuple(slater["F_vv"]),
                tuple(slater["F_cc"] or (0.0, 0.0, 0.0)),
                tuple(slater["F_cv"] or (0.0, 0.0, 0.0)),
                tuple(slater["G_cv"] or (0.0, 0.0, 0.0, 0.0)),
                core["soc"] if core is not None else 0.0,
                valence["soc"],
                resolved.tables.get("double_counting.mlft", {}).get("c", 0.0),
                _spectroscopy_zeeman(valence, notes),
                rank=rank,
                verbose=verbose,
            ),
            {},
        )

    if core is not None:
        raise InputError(
            f"A {resolved.calculation} run builds a single correlated shell, but a "
            f'[[shell]] with role = "core" is declared.'
        )
    zeeman = valence["zeeman_splitting"]
    return (
        load_model(
            resolved.tables["hamiltonian.file"]["path"],
            l=valence["l"],
            n_baths=valence["n_bath"],
            slater=None if slater is None else slater["F_vv"],
            xi=valence["soc"],
            # None, not (0, 0, 0): the sentinel selects each format's own default -- no field
            # for a flat .h0, a symmetry-breaking nudge for the labelled ones -- and an
            # explicit zero is a third thing again, since it skips the dressing step.
            h_field=None if zeeman is None else tuple(zeeman),
            n_val_baths=valence["n_valence_bath"],
            n_impurity_orbitals=resolved.tables["hamiltonian.file"]["n_impurity_orbitals"],
            allow_noninteracting=resolved.interaction_kind == "none",
            rank=rank,
            verbose=verbose,
        ),
        {},
    )


def _build_basis(resolved):
    from impurityModel.ed.model import BasisOptions, resolve_excitation_budget

    table = resolved.tables["many_body_basis"]
    shells = resolved.shells
    valence = next(shell for shell in shells if shell["role"] == "valence")

    if resolved.calculation == "spectroscopy":
        nominal = OrderedDict((shell["l"], shell["nominal_occupation"]) for shell in shells)
    else:
        nominal = {valence["l"]: valence["nominal_occupation"]}

    mixed = table["mixed_valence"]
    if mixed is not None and resolved.calculation != "spectroscopy":
        mixed = {valence["l"]: mixed}

    budget = table["excitation_budget"]
    threshold = table["truncation_threshold"]
    kwargs = dict(
        nominal_occ=nominal,
        mixed_valence=mixed,
        dN=table["dN"],
        # "auto" -> None, so the RAM-derived cap stays at its collective call site; "none" ->
        # infinity, which disables capping. The two are NOT the same thing.
        truncation_threshold=None if threshold == "auto" else (np.inf if threshold == "none" else threshold),
        chain_restrict=table["chain_restrict"],
        spin_flip_dj=table["spin_flip_dj"],
        tau=resolved.tables["temperature"]["tau"],
        excitation_budget=resolve_excitation_budget(None if budget == "auto" else (-1 if budget == "none" else budget)),
    )
    if table["occ_cutoff"] is not None:
        kwargs["occ_cutoff"] = table["occ_cutoff"]
    if table["slater_weight_min"] is not None:
        kwargs["slater_weight_min"] = table["slater_weight_min"]
    return BasisOptions(**kwargs)


def _build_solver(resolved):
    from impurityModel.ed.model import SolverOptions

    table = resolved.tables["solver"]
    return SolverOptions(
        # "auto" -> None, which is NOT one mode: it means NONE on the Green's-function path
        # and PARTIAL on the eigensolver path, and it also selects the memory model used to
        # derive the determinant budget.
        reort=None if table["reort"] == "auto" else table["reort"],
        dense_cutoff=table["dense_cutoff"],
        sparse_green=table["sparse_green"],
        gf_method=table["gf_method"],
    )


def _build_spectra_options(resolved, notes):
    from impurityModel.ed.model import SpectraOptions

    table = resolved.tables["spectroscopy"]
    rixs = resolved.tables["spectroscopy.rixs"]
    nixs = resolved.tables["spectroscopy.nixs"]

    radial = None
    if nixs["enabled"]:
        mesh, values = np.loadtxt(nixs["radial_file"]).T
        radial = (mesh, values, np.copy(values))

    # RIXS off still needs a mesh object of the right shape; an empty one is what the driver
    # reads as "no incoming energies", and the switch above is what actually decides.
    w_in = _mesh_array(rixs["w_in"]) if rixs["enabled"] else np.array([])

    return SpectraOptions(
        w=_mesh_array(table["w"]),
        delta=table["core_hole_broadening"],
        wLoss=_mesh_array(table["w_loss"]),
        wIn=w_in,
        deltaRIXS=rixs["final_state_broadening"],
        deltaNIXS=nixs["broadening"],
        qsNIXS=None if not nixs["q"] else [np.asarray(q, dtype=float) for q in nixs["q"]],
        liNIXS=nixs["l_final"],
        ljNIXS=nixs["l_initial"],
        radial=radial,
        auto_block_structure=resolved.tables["solver"]["auto_block_structure"],
        pes=resolved.tables["spectroscopy.pes"]["enabled"],
        xps=resolved.tables["spectroscopy.xps"]["enabled"],
        xas=resolved.tables["spectroscopy.xas"]["enabled"],
        rixs=rixs["enabled"],
        nixs=nixs["enabled"],
    )


def _fermionic_matsubara(tau, n_points):
    """``i*nu_n`` with ``nu_n = (2n+1)*pi*tau`` -- already multiplied by i, as Meshes.iw wants."""
    return 1j * (2 * np.arange(n_points) + 1) * np.pi * tau


def _build_meshes(resolved):
    """Meshes for the self-energy path.

    The susceptibility path deliberately does not go through here: its Matsubara mesh is
    bosonic, real-valued, includes ``nu = 0`` (the Van Vleck term) and is not carried in
    ``Meshes`` at all, so folding both into one object would quietly drop that point.
    """
    from impurityModel.ed.model import Meshes

    prefix = resolved.calculation
    real_axis = resolved.tables[f"{prefix}.real_axis"]
    w = _mesh_array(real_axis["mesh"]) if real_axis["enabled"] else None

    iw = None
    if prefix == "selfenergy":
        matsubara = resolved.tables["selfenergy.matsubara"]
        if matsubara["enabled"] and matsubara["n_points"] > 0:
            iw = _fermionic_matsubara(resolved.tables["temperature"]["tau"], matsubara["n_points"])
    if w is None and iw is None:
        raise InputError(
            f"[{prefix}]: both the real axis and the Matsubara output are disabled, so there " "is nothing to compute."
        )
    return Meshes(iw=iw, w=w, delta=real_axis["broadening"])


def apply_double_counting(resolved, model, basis, solver, notes, comm, verbosity):
    """Compute the double counting and attach it to the model.

    ``mlft`` never reaches here: it is a scalar folded into ``h0`` by the spectroscopy model
    builder with a ``+`` sign, not a matrix that gets subtracted, and the reader refuses it on
    any other calculation.

    The searching schemes cost one full collective ground-state solve per trial shift -- a
    dozen or more -- and their collectives run on ``MPI.COMM_WORLD`` regardless of the ``comm``
    passed to them. So this is entered unconditionally on every rank: never behind a
    rank-local test, and any decision about the outcome comes from a broadcast value.
    """
    from dataclasses import replace

    from impurityModel.ed import dc_criteria, dc_static
    from impurityModel.ed.dc_search import DoubleCountingUnreachable
    from impurityModel.ed.lie_algebra import tensors_to_operator

    scheme = resolved.dc_scheme
    if scheme in ("none", "mlft"):
        return model

    table = resolved.tables[f"double_counting.{scheme}"]
    tau = resolved.tables["temperature"]["tau"]

    if scheme in ("fll", "amf", "nominal", "sigma_inf"):
        if scheme == "fll":
            matrix = dc_static.fll_dc(model, tau=tau, u=table["u"], j=table["j"])
        elif scheme == "amf":
            matrix = dc_static.amf_dc(model, tau=tau)
        elif scheme == "sigma_inf":
            matrix = dc_static.sigma_inf_dc(model, tau=tau)
        else:
            nominal = sum(shell["nominal_occupation"] for shell in resolved.shells if shell["role"] == "valence")
            matrix = dc_static.nominal_dc(model, nominal, u=table["u"], j=table["j"])
        notes.append(f"Double counting from the {scheme} scheme (no ground-state solve).")
        return replace(model, dc=tensors_to_operator(np.asarray(matrix, dtype=complex)).to_dict())

    search = {
        "fixed_occupation": dc_criteria.fixed_occupation_dc,
        "fixed_peak": dc_criteria.fixed_peak_dc,
        "fixed_gap": dc_criteria.fixed_gap_dc,
    }[scheme]
    target = {
        "fixed_occupation": {"occupation": table.get("occupation")},
        "fixed_peak": {"peak_position": table.get("peak_position")},
        "fixed_gap": {"offset": table.get("offset")},
    }[scheme]
    if scheme == "fixed_occupation":
        target.update(occ_tol=table["occ_tol"], initial_step=table["initial_step"], max_shift=table["max_shift"])

    guess = table["guess"]
    seeded = model if guess == 0.0 else replace(model, dc=_uniform_dc(model, guess))
    try:
        matrix = search(seeded, basis, solver, comm=comm, verbosity=verbosity, **target)
    except DoubleCountingUnreachable:
        if table["on_unreachable"] != "keep_guess":
            raise
        notes.append(
            f"The {scheme} target has no solution (a plateau, or a target the observable steps "
            "across at a charge-sector boundary -- the expected outcome for a charge-transfer "
            'insulator). Keeping the guess, as on_unreachable = "keep_guess" asked.'
        )
        return seeded

    damping = table["damping"]
    if damping != 1.0:
        previous = np.asarray(_dense_dc(seeded, matrix.shape[0]), dtype=complex)
        matrix = previous + damping * (np.asarray(matrix) - previous)
        notes.append(f"Double counting damped by {damping} against the guess.")
    notes.append(f"Double counting from the {scheme} search.")
    return replace(model, dc=tensors_to_operator(np.asarray(matrix, dtype=complex)).to_dict())


def _uniform_dc(model, value):
    """``value * identity`` on the impurity block, in the operator form ``model.dc`` holds."""
    from impurityModel.ed.lie_algebra import tensors_to_operator

    n_imp = len(model.impurity_indices)
    return tensors_to_operator(value * np.eye(n_imp, dtype=complex)).to_dict()


def _dense_dc(model, n_imp):
    """Dense impurity-block view of ``model.dc`` (zeros when there is none)."""
    dense = np.zeros((n_imp, n_imp), dtype=complex)
    for (created, annihilated), value in (model.dc or {}).items():
        i, j = created[0], annihilated[0]
        if i < n_imp and j < n_imp:
            dense[i, j] = value
    return dense


def build(resolved, comm=None, verbosity=0):
    """Turn a resolved input file into everything its driver needs.

    Parameters
    ----------
    resolved : impurityModel.inputformat.reader.ResolvedInput
    comm : MPI communicator, optional
    verbosity : int, optional

    Returns
    -------
    Built
    """
    rank = 0 if comm is None else comm.rank
    notes = list(resolved.warnings)

    header = read_h0_header(resolved)
    check_header_agreement(resolved, header, notes)
    deduce_bath_counts(resolved, header, notes)

    model, archived = _build_model(resolved, header, notes, rank, verbosity > 0)

    if archived:
        meshes, basis, solver, cluster = archived["archive"]
        solver = _build_solver(resolved)
        extra = {"cluster_label": cluster}
    else:
        basis = _build_basis(resolved)
        solver = _build_solver(resolved)
        meshes = None
        extra = {}

    spectra_options = None
    if resolved.calculation == "spectroscopy":
        spectra_options = _build_spectra_options(resolved, notes)
        extra.setdefault("cluster_label", resolved.tables["spectroscopy"]["cluster"])
        extra["output"] = resolved.tables["spectroscopy"]["output"]
    else:
        if meshes is None:
            meshes = _build_meshes(resolved)
        extra.setdefault("cluster_label", resolved.tables[resolved.calculation]["cluster"])
        extra["output"] = resolved.tables[resolved.calculation]["output"]
        if resolved.calculation == "susceptibility":
            matsubara = resolved.tables["susceptibility.matsubara"]
            extra["n_matsubara"] = matsubara["n_points"] if matsubara["enabled"] else 0
            extra["num_wanted"] = resolved.tables["susceptibility"]["n_psi_max"]

    model = apply_double_counting(resolved, model, basis, solver, notes, comm, verbosity)

    return Built(
        model=model,
        basis=basis,
        solver=solver,
        spectra=spectra_options,
        meshes=meshes,
        extra=extra,
        notes=notes,
    )
