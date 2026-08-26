"""``impurityModel run input.toml`` -- drive any calculation from a single input file.

Also hosts the two verbs that make the format discoverable without reading the source:
``impurityModel init`` writes a commented starter file, and ``impurityModel schema`` prints
the reference table. Both are generated from the schema declarations, so neither can drift.

Three separate verbs replace what started as one ``--dry-run`` flag, because it wanted to be
three incompatible things at once. ``--check`` is cheap enough for CI to run on every example;
``--show-resolved`` prints what the file actually means; and the fully-resolved record --
including values only the solver can know -- is written *after* the run, into the output
archive, where those values exist.
"""

import json
import os
from pathlib import Path

from impurityModel.inputformat import schema
from impurityModel.inputformat.reader import apply_environment, load_input
from impurityModel.scripts._verbosity import add_verbosity_argument, resolve_verbosity


def add_arguments(parser):
    """Register the ``run`` sub-command arguments on ``parser``."""
    parser.add_argument("input", type=str, help="TOML input file describing the calculation.")
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Validate the file and exit without solving: parsing, cross-checks and the "
            "capability gate, but no MPI work. Cheap enough to run on every input in CI."
        ),
    )
    parser.add_argument(
        "--show-resolved",
        dest="show_resolved",
        action="store_true",
        help=(
            "Validate, then print every value as the solver will see it -- energies in eV, "
            "each with the source it came from -- plus every effective tuning knob, and exit."
        ),
    )
    add_verbosity_argument(parser)


def _source_of(resolved, table, key):
    """Where a resolved value came from, for ``--show-resolved``."""
    spec = schema.TABLES.get(table)
    if spec is None:
        return "file"
    declared = {k.name: k for k in spec.keys}
    if key not in declared:
        return "file"
    overrides = schema.TABLES[resolved.calculation].overrides.get(table, {})
    if key in overrides:
        return f"default for [{resolved.calculation}]"
    return "default" if declared[key].default is not schema.UNSET else "deduced"


def _print_resolved(resolved, built=None):
    from impurityModel.ed import config

    print(f"# {resolved.path}")
    print(f"# format {resolved.version[0]}.{resolved.version[1]}, calculation: {resolved.calculation}")
    print(f"# energies below are in eV (the file declares {resolved.tables['units']['energy']})")
    for table in sorted(resolved.tables):
        values = resolved.tables[table]
        if not values:
            continue
        print(f"\n[{table}]")
        for key in sorted(values):
            print(f"  {key} = {values[key]!r}    # {_source_of(resolved, table, key)}")
    for index, shell in enumerate(resolved.shells):
        print(f"\n[[shell]]  # {index}")
        for key in sorted(shell):
            print(f"  {key} = {shell[key]!r}")
    print("\n[environment]  # effective values of every tuning knob")
    for name, knob in sorted(config.KNOBS.items()):
        chosen = resolved.environment.get(name)
        if chosen is not None:
            print(f"  {name} = {chosen}    # from this file")
        elif name in os.environ:
            print(f"  {name} = {os.environ[name]}    # from the environment")
        else:
            default = "derived at the call site" if knob.default is None else repr(knob.default)
            print(f"  # {name} = {default}")
    if built is not None and built.notes:
        print("\n# deduced:")
        for note in built.notes:
            print(f"#   - {note}")


def _provenance(resolved, built):
    """What the run was, in a form another program can read back.

    Stored as JSON rather than TOML only because writing TOML would mean a new dependency for
    a record nothing hand-edits; the content is the resolved input, not a re-quoted copy of
    the file, which is kept verbatim beside it.

    The determinant cap is *not* in here when it was left on ``"auto"``: it is derived inside
    the solver from the memory available on each rank, so this layer genuinely does not know
    it. Recording ``"auto"`` and saying where the real value is decided is honest; inventing a
    number here would not be.
    """
    from impurityModel.ed import config

    return {
        "input_file": resolved.path,
        "format_version": list(resolved.version),
        "calculation": resolved.calculation,
        "declared_energy_unit": resolved.tables["units"]["energy"],
        "resolved_eV": {table: _jsonable(values) for table, values in resolved.tables.items()},
        "shells": [_jsonable(shell) for shell in resolved.shells],
        "deduced": list(built.notes),
        "knobs": {name: knob.get() for name, knob in sorted(config.KNOBS.items()) if knob.get() is not None},
        "notes_on_completeness": (
            "truncation_threshold = 'auto' is resolved inside the solver from per-rank memory "
            "and is therefore not recorded here."
        ),
    }


def _jsonable(mapping):
    """Coerce a resolved table to something ``json`` will take."""
    import numpy as np

    out = {}
    for key, value in mapping.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def _write_provenance(path, resolved, built):
    """Append the run record to an output archive, on rank 0, after the run."""
    import h5py

    if not Path(path).is_file():
        return
    with h5py.File(path, "a") as handle:
        group = handle.require_group("provenance")
        for name in ("input_toml", "resolved"):
            if name in group:
                del group[name]
        group.create_dataset("input_toml", data=resolved.raw_text)
        group.create_dataset("resolved", data=json.dumps(_provenance(resolved, built), indent=1, default=str))


def _dispatch(resolved, built, comm, verbosity, outdir):
    """Call the driver this input file selected, with the arguments it actually takes."""
    output = built.extra.get("output")
    cluster = built.extra.get("cluster_label", "cluster")

    if resolved.calculation == "spectroscopy":
        from impurityModel.ed.get_spectra import run_spectra

        target = str(Path(outdir, output or "spectra.h5"))
        run_spectra(built.model, built.spectra, built.basis, comm, verbosity=verbosity, output_filename=target)
        return target

    if resolved.calculation == "selfenergy":
        from impurityModel.ed.selfenergy import calc_selfenergy
        from impurityModel.scripts.selfenergy import _save_results

        result = calc_selfenergy(
            built.model, built.meshes, built.basis, built.solver, comm=comm, verbosity=verbosity, cluster_label=cluster
        )
        target = str(Path(outdir, output or f"selfenergy-{cluster}.h5"))
        if (comm is None or comm.rank == 0) and result is not None:
            _save_results(result, built.meshes, cluster, target)
        return target

    from impurityModel.ed.susceptibility import calc_susceptibility_workflow

    target = str(Path(outdir, output or "chi.h5"))
    calc_susceptibility_workflow(
        built.model,
        built.meshes,
        built.basis,
        built.solver,
        comm=comm,
        verbosity=verbosity,
        cluster_label=cluster,
        num_wanted=built.extra.get("num_wanted", 5),
        n_matsubara=built.extra.get("n_matsubara", 0),
        output_filename=target,
    )
    return target


def run(args):
    """Load ``args.input``, then validate, explain or execute it."""
    from mpi4py import MPI

    from impurityModel.inputformat.build import build

    comm = MPI.COMM_WORLD
    verbosity = resolve_verbosity(args)
    resolved = load_input(args.input, comm=comm)

    if comm.rank == 0:
        for warning in resolved.warnings:
            print(f"warning: {warning}")

    if args.check:
        # Validation only: no model is constructed, so this stays cheap enough for CI.
        if comm.rank == 0:
            print(f"{args.input}: OK ({resolved.calculation}, format {resolved.version[0]}.{resolved.version[1]})")
        return 0

    # Tuning knobs are read lazily on every access, so setting them here -- before any solver
    # call, on every rank -- is early enough. They are restored on the way out, which matters
    # for anything that calls this more than once in a process.
    with apply_environment(resolved.environment):
        built = build(resolved, comm=comm, verbosity=verbosity)

        if args.show_resolved:
            if comm.rank == 0:
                _print_resolved(resolved, built)
            return 0

        if comm.rank == 0:
            for note in built.notes:
                print(f"note: {note}")

        outdir = resolved.tables["run"]["outdir"]
        if comm.rank == 0:
            Path(outdir).mkdir(parents=True, exist_ok=True)
        comm.Barrier()

        target = _dispatch(resolved, built, comm, verbosity, outdir)

    if comm.rank == 0:
        _write_provenance(target, resolved, built)
        print(f"Wrote {target} (with a provenance record of this input file).")
    return 0


# --------------------------------------------------------------------------------------
# Discoverability: `impurityModel init` and `impurityModel schema`
# --------------------------------------------------------------------------------------


def add_init_arguments(parser):
    parser.add_argument(
        "--calculation",
        choices=schema.CALCULATIONS,
        default="spectroscopy",
        help="Which calculation the starter file should describe.",
    )


def init(args):
    """Print a commented starter input file for ``args.calculation``.

    Required keys appear with a plausible value; optional ones appear commented out with their
    default and one-line description, so the file itself is the reference. Generated from the
    same declarations as the documentation, so it cannot fall behind the reader.
    """
    print(render_template(args.calculation))
    return 0


def render_template(calculation):
    """The starter file for ``calculation``, as text."""
    lines = [
        f"# impurityModel input file -- {calculation}",
        f"# Generated by `impurityModel init --calculation {calculation}`.",
        "# Commented lines show the default; uncomment to change one.",
        "",
        "[format]",
        f"version = [{schema.SPEC_VERSION[0]}, {schema.SPEC_VERSION[1]}]",
        "",
        "[units]",
        "# REQUIRED, no default: the command line defaults to eV and RSPt writes Rydberg, so a",
        "# default here would let the two disagree by 13.6057x in silence.",
        'energy = "eV"',
        "",
        "[hamiltonian.file]",
        "# A self-describing .h0 also supplies the bath layout, so the [[shell]] counts below",
        "# can be omitted with it. A legacy .pickle/.json/.dat records none, so they cannot.",
        'path = "h0.h0"',
        "",
        "[[shell]]",
        "l = 2",
        'role = "valence"        # core | valence -- declared, never inferred from `l`',
        "nominal_occupation = 8",
        "# soc = 0.0",
        "# n_bath = 10           # deduced from a .h0 header; required for other sources",
        "# n_valence_bath = 10",
        "",
        "[interaction.slater]",
        "F_vv = [7.5, 0.0, 9.9, 0.0, 6.6]   # length 2*l_valence + 1",
        "",
    ]
    if calculation == "spectroscopy":
        lines += [
            "# A core shell is required for XPS/XAS/RIXS, and must satisfy",
            "# |l_core - l_valence| = 1 or the transition is zero by selection rule.",
            "[[shell]]",
            "l = 1",
            'role = "core"',
            "nominal_occupation = 6",
            "soc = 11.629",
            "",
            "[double_counting.mlft]",
            "c = 1.5",
            "",
        ]
    lines += [f"[{calculation}]"]
    for path in [calculation] + sorted(p for p in schema.TABLES if p.startswith(f"{calculation}.")):
        table = schema.TABLES[path]
        if path != calculation:
            lines += ["", f"[{path}]"]
        for key in table.keys:
            summary = " ".join(key.doc.split())[:88]
            rendered = None if key.default is schema.UNSET else toml_value(key.default)
            if rendered is None:
                # No default, or one TOML cannot express: show the key, not a broken value.
                lines.append(f"# {key.name} =    # {summary}")
                continue
            # `enabled` is written live, so the file states plainly which spectra it wants;
            # everything else is commented out at its default.
            prefix = "" if key.name == "enabled" else "# "
            lines.append(f"{prefix}{key.name} = {rendered}    # {summary}")
    lines += [
        "",
        "# [environment]        # any impurityModel.ed.config knob, by name",
        "# GF_BICGSTAB_ATOL = 1e-9",
    ]
    return "\n".join(lines)


def toml_value(value):
    """Render a Python value as TOML, or return ``None`` when TOML cannot express it.

    ``None`` is the case worth naming: TOML has no null, so a key whose default is "unset"
    cannot be written with a value at all and has to be emitted as a comment instead.
    Writing Python's ``repr`` would produce ``None``, ``True`` and ``'single quotes'`` -- a
    file that does not parse, which is how the first version of this generator was caught.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (list, tuple)):
        rendered = [toml_value(item) for item in value]
        return None if any(item is None for item in rendered) else "[" + ", ".join(rendered) + "]"
    if isinstance(value, dict):
        rendered = {key: toml_value(item) for key, item in value.items()}
        if any(item is None for item in rendered.values()):
            return None
        return "{ " + ", ".join(f"{key} = {item}" for key, item in rendered.items()) + " }"
    return None


def add_schema_arguments(parser):
    parser.add_argument("table", nargs="?", default=None, help="Print only this table (e.g. spectroscopy.rixs).")


def show_schema(args):
    """Print the generated key reference, for one table or all of them."""
    if args.table is None:
        print(schema.dump())
        return 0
    if args.table not in schema.TABLES:
        import difflib

        close = difflib.get_close_matches(args.table, sorted(schema.TABLES), n=1)
        hint = f" Did you mean {close[0]!r}?" if close else ""
        raise SystemExit(f"No table {args.table!r}.{hint} Run `impurityModel schema` for the full list.")
    table = schema.TABLES[args.table]
    heading = f"[[{args.table}]]" if table.repeatable else f"[{args.table}]"
    print(f"{heading}\n\n{' '.join(table.doc.split())}\n")
    for key in table.keys:
        default = (
            "*deduced*" if key.deduced_from else ("**required**" if key.default is schema.UNSET else repr(key.default))
        )
        print(f"  {key.name}  ({key.kind.value}, default {default})")
        print(f"      {' '.join(key.doc.split())}")
        if key.choices:
            print(f"      choices: {', '.join(repr(c) for c in key.choices)}")
    return 0


# --------------------------------------------------------------------------------------
# The migration bridge: an old command line, written out as the input file that replaces it
# --------------------------------------------------------------------------------------


def _line(key, value, comment=None):
    rendered = toml_value(value)
    if rendered is None:
        return f"# {key} =" + (f"    # {comment}" if comment else "")
    return f"{key} = {rendered}" + (f"    # {comment}" if comment else "")


def emit_toml(command, args):
    """Render an argparse ``Namespace`` as the input file that would do the same thing.

    Called *before* the unit conversion, so the emitted file declares the unit the user typed
    and keeps their numbers as they typed them, rather than silently rewriting everything into
    eV. That makes the output a translation of the command line rather than a normalisation
    of it.

    This is a bridge, not an oracle. It can only emit the subset the flags can express: the
    double-counting schemes, the NIXS momentum transfer, the per-spectrum switches and the
    tuning knobs have no flag to translate.
    """
    out = [
        f"# Generated by `impurityModel {command} ... --emit-toml`.",
        "# A translation of the command line, in the unit it was typed in.",
        "",
        "[format]",
        f"version = [{schema.SPEC_VERSION[0]}, {schema.SPEC_VERSION[1]}]",
        "",
        "[units]",
        f'energy = "{args.unit}"',
        "",
        "[hamiltonian.file]",
        _line("path", args.h0_filename or "SET_ME"),
    ]
    if getattr(args, "n_impurity_orbitals", None) is not None:
        out.append(_line("n_impurity_orbitals", args.n_impurity_orbitals))

    if command == "spectra":
        shells = list(zip(args.ls, args.nBaths, args.nValBaths, args.n0imps))
        for angular, n_bath, n_valence, occupation in shells:
            role = "valence" if angular == max(args.ls) else "core"
            soc = args.xi_3d if role == "valence" else args.xi_2p
            out += [
                "",
                "[[shell]]",
                _line("l", angular),
                _line("role", role),
                _line("n_bath", n_bath),
                _line("n_valence_bath", n_valence),
                _line("nominal_occupation", occupation),
                _line("soc", soc),
            ]
            if role == "valence" and any(args.hField):
                out.append(_line("zeeman_splitting", list(args.hField)))
        out += [
            "",
            "[interaction.slater]",
            _line("F_vv", list(args.Fdd)),
            _line("F_cc", list(args.Fpp)),
            _line("F_cv", list(args.Fpd)),
            _line("G_cv", list(args.Gpd)),
            "",
            "[double_counting.mlft]",
            _line("c", args.chargeTransferCorrection),
            "",
            "[temperature]",
            _line("kelvin", args.T),
            "",
            "[spectroscopy]",
            _line("core_hole_broadening", args.delta),
            "",
            "[spectroscopy.pes]",
            _line("enabled", True),
            "",
            "[spectroscopy.xps]",
            _line("enabled", True),
            "",
            "[spectroscopy.xas]",
            _line("enabled", True),
            "",
            "[spectroscopy.rixs]",
            _line("enabled", args.deltaRIXS > 0, "was: deltaRIXS > 0"),
            _line("final_state_broadening", args.deltaRIXS),
            "",
            "[spectroscopy.nixs]",
            _line("enabled", bool(args.radial_filename), "was: a radial file was supplied"),
            _line("radial_file", args.radial_filename or "SET_ME"),
            _line("broadening", args.deltaNIXS),
        ]
    else:
        out += [
            "",
            "[[shell]]",
            _line("l", args.ls),
            _line("role", "valence"),
            _line("n_bath", args.nBaths),
            _line("nominal_occupation", args.n0imps),
            _line("soc", args.xi),
        ]
        if args.hField is not None:
            out.append(_line("zeeman_splitting", list(args.hField)))
        out += [
            "",
            "[interaction.slater]",
            _line("F_vv", list(args.Fdd)),
            "",
            "[temperature]",
            _line("tau", args.tau),
            "",
            f"[{command}]",
            _line("cluster", args.clustername),
            "",
            f"[{command}.real_axis]",
            _line("enabled", getattr(args, "realaxis", True)),
            f"mesh = {{ min = {args.w_min!r}, max = {args.w_max!r}, n = {args.w_n} }}",
            _line("broadening", args.delta),
            "",
            f"[{command}.matsubara]",
            _line("enabled", args.n_matsubara > 0),
            _line("n_points", args.n_matsubara),
        ]

    out += [
        "",
        "# Not expressible as a command-line flag, and therefore not emitted here:",
        "#   [double_counting.*] beyond mlft, [spectroscopy.nixs].q, and [environment].",
        "# Run `impurityModel schema` to see them.",
    ]
    return "\n".join(out)
