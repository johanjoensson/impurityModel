"""Read, validate and resolve a TOML input file.

Rank discipline
---------------
**Rank 0 parses; the result is broadcast; the raise verdict is collective.** The tempting
alternative -- every rank parses the same bytes, so the result must agree -- is wrong in the
way that produces hangs rather than errors: parsing is a pure function only of the bytes *a
rank managed to read*. On a networked or node-local filesystem a subset of ranks can get
``FileNotFoundError``, which is not a uniform raise but an asymmetric exit, with the raisers
dying and the survivors blocking in the first collective. Relative paths resolve against a
per-rank working directory under some launchers, too. This repository has already written
down the counter-argument, in ``ed/dc_search.py``: *"environment variables are uniform under
mpiexec in every normal invocation, and 'in every normal invocation' is how this repo's
deadlocks got in."* One broadcast of a few kilobytes against a multi-minute solve is not a
cost worth reasoning about.

Nothing is derived here
-----------------------
``"auto"`` resolves to the dataclass sentinel, never to a number. Values derived from
available memory or communicator size stay at their existing collective call sites, so this
module never needs MPI beyond the single broadcast and never has to be rank-aware.

A leaf module: :mod:`~impurityModel.inputformat.schema`,
:mod:`~impurityModel.inputformat.capabilities`, :mod:`impurityModel.ed.h0_format` and
:mod:`impurityModel.ed.average` (for Boltzmann's constant). No solver imports, no argparse,
no ``mpi4py`` import -- the communicator arrives as an argument.
"""

import difflib
import os
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from impurityModel.ed import config
from impurityModel.ed.average import k_B
from impurityModel.ed.h0_format import ENERGY_UNITS
from impurityModel.inputformat import capabilities, schema
from impurityModel.inputformat.schema import UNSET, Kind

__all__ = [
    "InputError",
    "ResolvedInput",
    "load_input",
    "load_environment",
    "apply_environment",
    "find_environment_file",
    "ENVIRONMENT_FILENAME",
    "ENVIRONMENT_PATH_VAR",
]


class InputError(ValueError):
    """A malformed, contradictory or unrecognised input file.

    Carries the dotted location so the message points at a key rather than at a line of TOML
    the user then has to map back to a key themselves.
    """


def _suggest(name, candidates):
    """``" Did you mean 'x'?"`` when a close match exists, else an empty string.

    The candidate list is always sorted before it reaches :mod:`difflib`, because an
    unordered set would make the suggestion depend on ``PYTHONHASHSEED`` -- harmless for a
    single process, but this message can be produced on any rank.
    """
    matches = difflib.get_close_matches(name, sorted(candidates), n=1, cutoff=0.6)
    return f" Did you mean {matches[0]!r}?" if matches else ""


def _fail(where, message, candidates=()):
    raise InputError(f"[{where}]: {message}{_suggest(where.rsplit('.', 1)[-1], candidates) if candidates else ''}")


# --------------------------------------------------------------------------------------
# Value coercion, driven entirely by the declared kind
# --------------------------------------------------------------------------------------


def _as_float(where, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"[{where}]: expected a number, got {value!r}")
    return float(value)


def _as_int(where, value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputError(f"[{where}]: expected an integer, got {value!r}")
    return int(value)


def _as_sequence(where, value):
    if not isinstance(value, (list, tuple)):
        raise InputError(f"[{where}]: expected a list, got {value!r}")
    return list(value)


def _coerce_mesh(where, value):
    """Normalise a mesh to ``{"kind": ..., ...}``; energies are converted by the caller.

    Three declared shapes -- ``{min, max, n}``, ``{values = [...]}`` and ``{file = "..."}``.
    Declaring the *shape* now, rather than only the uniform case, is what lets the
    non-uniform meshes the solver already builds internally acquire an input representation
    later without a format break.
    """
    if not isinstance(value, dict):
        raise InputError(f"[{where}]: a mesh must be a table, e.g. {{ min = -10, max = 10, n = 2001 }}")
    keys = set(value)
    if keys == {"min", "max", "n"}:
        n = _as_int(f"{where}.n", value["n"])
        if n < 1:
            raise InputError(f"[{where}.n]: a mesh needs at least one point, got {n}")
        lo, hi = _as_float(f"{where}.min", value["min"]), _as_float(f"{where}.max", value["max"])
        if hi < lo:
            raise InputError(f"[{where}]: max ({hi}) is below min ({lo})")
        return {"kind": "uniform", "min": lo, "max": hi, "n": n}
    if keys == {"values"}:
        values = [_as_float(f"{where}.values", v) for v in _as_sequence(f"{where}.values", value["values"])]
        if not values:
            raise InputError(f"[{where}.values]: an explicit mesh must have at least one point")
        return {"kind": "values", "values": values}
    if keys == {"file"}:
        return {"kind": "file", "file": str(value["file"])}
    raise InputError(
        f"[{where}]: unrecognised mesh {sorted(keys)}. Use exactly one of "
        "{min, max, n}, {values = [...]} or {file = '...'}."
    )


def _coerce_matrix(where, value):
    """Normalise a matrix declaration.

    The encoding is ``shape`` plus row-major ``real`` (and optional ``imag``) nested arrays,
    one TOML array per matrix row, rather than the ``.h0`` header's interleaved ``[re, im]``
    pairs. Interleaving is fine for a file one program writes and another reads; it is
    miserable to eyeball, diff or hand-edit, and matrices here are things a person may well
    want to inspect. Shorthands exist because they are what people actually write.
    """
    if not isinstance(value, dict):
        raise InputError(f"[{where}]: a matrix must be a table (shape/real/imag, identity, diagonal, scalar or path)")
    keys = set(value)
    if "path" in keys:
        return {"kind": "path", "path": str(value["path"]), "columns": value.get("columns", "re im")}
    if keys == {"identity"} and value["identity"] is True:
        return {"kind": "identity"}
    if keys == {"diagonal"}:
        return {"kind": "diagonal", "diagonal": [_as_float(f"{where}.diagonal", v) for v in value["diagonal"]]}
    if keys == {"scalar"}:
        return {"kind": "scalar", "scalar": _as_float(f"{where}.scalar", value["scalar"])}
    if "real" in keys:
        unknown = keys - {"real", "imag", "shape"}
        if unknown:
            raise InputError(f"[{where}]: unexpected matrix keys {sorted(unknown)}")
        real = [[_as_float(f"{where}.real", v) for v in _as_sequence(f"{where}.real", row)] for row in value["real"]]
        rows, cols = len(real), len(real[0]) if real else 0
        if any(len(row) != cols for row in real):
            raise InputError(f"[{where}.real]: rows have unequal lengths")
        imag = None
        if "imag" in keys:
            imag = [
                [_as_float(f"{where}.imag", v) for v in _as_sequence(f"{where}.imag", row)] for row in value["imag"]
            ]
            if len(imag) != rows or any(len(row) != cols for row in imag):
                raise InputError(f"[{where}.imag]: shape does not match `real` ({rows}x{cols})")
        if "shape" in keys:
            shape = [_as_int(f"{where}.shape", v) for v in _as_sequence(f"{where}.shape", value["shape"])]
            if shape != [rows, cols]:
                raise InputError(f"[{where}.shape]: declared {shape} but the data is {[rows, cols]}")
        if rows * cols > INLINE_MATRIX_LIMIT:
            raise InputError(
                f"[{where}]: {rows}x{cols} is too large to write inline "
                f"(limit {INLINE_MATRIX_LIMIT} numbers). Use path = 'matrix.npy', or a "
                "two-column text file with columns = 're im'."
            )
        return {"kind": "dense", "shape": [rows, cols], "real": real, "imag": imag}
    raise InputError(
        f"[{where}]: unrecognised matrix {sorted(keys)}. Use shape/real[/imag], "
        "identity = true, diagonal = [...], scalar = x, or path = '...'."
    )


#: Largest inline matrix, in numbers. Above this the reader insists on an out-of-line file:
#: a thousand numbers pasted into a TOML table is not something anyone reads or reviews, and
#: the failure mode of letting it through is a file nobody can diff.
INLINE_MATRIX_LIMIT = 200


def _normalize_default(where, key, value, base_dir):
    """Put a *default* into the same shape a written value ends up in, without converting it.

    Structure and units are separate concerns here. A default is already stated in internal
    units (the code's own eV literals), so it must not be scaled -- but a mesh default still
    has to acquire the same normalised shape a written mesh gets, or consumers would need two
    code paths for "the user wrote it" and "the user did not".
    """
    if value is None or key.kind not in (Kind.MESH, Kind.MATRIX):
        return value
    identity = {"energy": "eV", "temperature": "K"}
    return _coerce(where, key, value, identity, base_dir)


def _coerce(where, key, value, units, base_dir):
    """Convert one raw TOML value to its resolved form, according to ``key.kind``.

    This is the single place ``[units].energy`` is applied, and it is applied *by kind*. A
    dimensionless multiplier (``energy_cut``), an electron count (``occupation``) and an
    inverse length (the NIXS ``q``) therefore cannot be scaled by 13.6 no matter what the
    file's unit is -- which the per-sub-command ``_ENERGY_FIELDS`` tuples could not promise,
    since a key added later was simply absent from them.
    """
    kind = key.kind
    factor = ENERGY_UNITS[units["energy"]]

    if kind is Kind.ENERGY:
        return _as_float(where, value) * factor
    if kind is Kind.ENERGY_LIST:
        return [_as_float(where, v) * factor for v in _as_sequence(where, value)]
    if kind is Kind.ENERGY_VECTOR:
        vector = [_as_float(where, v) * factor for v in _as_sequence(where, value)]
        if len(vector) != 3:
            raise InputError(f"[{where}]: expected 3 components (x, y, z), got {len(vector)}")
        return vector
    if kind is Kind.TEMPERATURE:
        # Kelvin becomes the fundamental temperature k_B*T in eV; an energy is converted like
        # any other energy. Two governances, which is why they are separate keys.
        raw = _as_float(where, value)
        return raw * k_B if units["temperature"] == "K" else raw * factor
    if kind is Kind.COUNT:
        number = _as_int(where, value)
        if key.minimum is not None and number < key.minimum:
            raise InputError(f"[{where}]: must be >= {key.minimum}, got {number}")
        return number
    if kind in (Kind.DIMENSIONLESS, Kind.INVERSE_LENGTH):
        number = _as_float(where, value)
        if key.minimum is not None and number < key.minimum:
            raise InputError(f"[{where}]: must be >= {key.minimum}, got {number}")
        return number
    if kind is Kind.BOOL:
        if not isinstance(value, bool):
            raise InputError(f"[{where}]: expected true or false, got {value!r}")
        return value
    if kind is Kind.PATH:
        # Rebased onto the input file's directory: an input is part of what the file
        # describes, so the file and its data travel together (and the launcher scripts'
        # "${DIR}/.." pattern keeps working).
        return str(Path(base_dir, str(value)))
    if kind is Kind.OUTPUT_PATH:
        # Deliberately NOT rebased. Where results go belongs to the invocation, not to the
        # model. Rebasing would also make an explicit `outdir = "."` mean something different
        # from the identical default, which is the kind of split nobody would ever guess.
        return str(value)
    if kind is Kind.STRING:
        return str(value)
    if kind is Kind.STRING_LIST:
        return [str(v) for v in _as_sequence(where, value)]
    if kind is Kind.VERSION:
        parts = [_as_int(where, v) for v in _as_sequence(where, value)]
        if len(parts) != 2:
            raise InputError(f"[{where}]: expected [major, minor], got {value!r}")
        return tuple(parts)
    if kind in (Kind.ENUM, Kind.AUTO_ENUM, Kind.AUTO_COUNT):
        if kind is Kind.AUTO_COUNT and isinstance(value, int) and not isinstance(value, bool):
            if key.minimum is not None and value < key.minimum:
                raise InputError(f"[{where}]: must be >= {key.minimum}, got {value}")
            return int(value)
        text = str(value)
        if key.choices and text not in key.choices:
            allowed = ", ".join(repr(c) for c in key.choices)
            extra = " (or an integer)" if kind is Kind.AUTO_COUNT else ""
            raise InputError(f"[{where}]: {value!r} is not one of {allowed}{extra}.{_suggest(text, key.choices)}")
        return text
    if kind is Kind.MESH:
        mesh = _coerce_mesh(where, value)
        if mesh["kind"] == "uniform":
            mesh["min"] *= factor
            mesh["max"] *= factor
        elif mesh["kind"] == "values":
            mesh["values"] = [v * factor for v in mesh["values"]]
        else:
            mesh["file"] = str(Path(base_dir, mesh["file"]))
        return mesh
    if kind is Kind.MATRIX:
        matrix = _coerce_matrix(where, value)
        if matrix["kind"] == "path":
            matrix["path"] = str(Path(base_dir, matrix["path"]))
        return matrix
    if kind is Kind.VECTOR_LIST:
        vectors = []
        for entry in _as_sequence(where, value):
            vector = [_as_float(where, v) for v in _as_sequence(where, entry)]
            if len(vector) != 3:
                raise InputError(f"[{where}]: each entry needs 3 components, got {len(vector)}")
            vectors.append(vector)
        return vectors
    raise InputError(f"[{where}]: no coercion is defined for kind {kind}")


#: Semantics a reader must understand to interpret a file correctly. An entry in a file's
#: ``required_features`` that is not here is a hard error -- the same contract the ``.h0``
#: header uses, and the reason unknown *keys* can afford to be lenient while this cannot.
SUPPORTED_FEATURES = frozenset(
    {
        "units",
        "shell_roles",
        "spectroscopy_switches",
        "bath_deduction",
        "double_counting",
        "environment",
    }
)


@dataclass
class ResolvedInput:
    """A validated input file with every value in eV and every path absolute.

    Attributes
    ----------
    tables : dict
        ``{table_path: {key: value}}`` for every declared table, present or defaulted.
    provided : dict
        ``{table_path: set_of_key_names}`` the *file* supplied, as opposed to keys that took a
        default. Without this, "where did this value come from?" cannot be answered, and
        answering it is the whole point of ``--show-resolved``.
    shells : list of dict
        The ``[[shell]]`` entries, in file order.
    environment : dict
        ``{knob_name: value}`` from ``[environment]``, already validated and clamped.
    calculation : str
        Which of :data:`schema.CALCULATIONS` this file selects.
    hamiltonian_source, interaction_kind, dc_scheme : str
        The selected variant of each tagged union (``dc_scheme`` is ``"none"`` if absent).
    version : tuple of int
        ``(major, minor)`` as declared.
    path : str
        Absolute path of the input file.
    raw_text : str
        The file verbatim, for the provenance record.
    warnings : list of str
        Non-fatal notes (unknown future keys, deduction fallbacks) to report on rank 0.
    """

    tables: dict
    provided: dict
    shells: list
    environment: dict
    calculation: str
    hamiltonian_source: str
    interaction_kind: str
    dc_scheme: str
    version: tuple
    path: str
    raw_text: str
    warnings: list = field(default_factory=list)

    def get(self, table, key, default=None):
        """Value of ``key`` in ``table``, or ``default`` when the table is absent."""
        return self.tables.get(table, {}).get(key, default)


def _check_version(raw, warnings):
    """Refuse a future major; return the declared ``(major, minor)``."""
    declared = raw.get("format", {}).get("version")
    if declared is None:
        raise InputError(
            "[format].version is required, as [major, minor]. Write `version = "
            f"[{schema.SPEC_VERSION[0]}, {schema.SPEC_VERSION[1]}]`."
        )
    version = _coerce("format.version", schema.TABLES["format"].keys[0], declared, {"energy": "eV"}, ".")
    if version[0] > schema.SPEC_VERSION[0]:
        raise InputError(
            f"[format].version says {version[0]}.{version[1]}, but this reader understands "
            f"major version {schema.SPEC_VERSION[0]} at most. Upgrade impurityModel."
        )
    return version


def _check_required_features(raw):
    for feature in raw.get("format", {}).get("required_features", []):
        if feature not in SUPPORTED_FEATURES:
            raise InputError(
                f"[format].required_features lists {feature!r}, which this reader does not "
                "understand. The file needs a semantic this version cannot provide, so "
                "reading it would silently misinterpret it." + _suggest(str(feature), SUPPORTED_FEATURES)
            )


def _unknown_key(where, name, declared, version, warnings):
    """Apply the minor-version rule to an undeclared key.

    A ``.h0`` file is written by one program for another, so the risk there is an old reader
    meeting a new producer and unknown keys are ignored. A TOML file is written by a person,
    so the risk here is a typo silently doing nothing for six hours. The minor version tells
    the two apart: at or below ours, an unknown key cannot be a future key, so it is a typo.
    """
    if version[1] > schema.SPEC_VERSION[1]:
        warnings.append(
            f"[{where}]: ignoring unknown key {name!r} -- the file declares format "
            f"{version[0]}.{version[1]}, newer than this reader's "
            f"{schema.SPEC_VERSION[0]}.{schema.SPEC_VERSION[1]}, so it may be a later addition."
        )
        return
    raise InputError(f"[{where}]: unknown key {name!r}.{_suggest(name, declared)}")


def _resolve_table(path, raw_table, units, base_dir, version, warnings, overrides=None, provided=None):
    """Coerce one declared table's keys and fill in defaults.

    ``provided`` collects the key names the file actually wrote, so a later report can tell a
    written value from a defaulted one.
    """
    table = schema.TABLES[path]
    declared = {key.name: key for key in table.keys}
    resolved = {}
    if provided is not None:
        provided[path] = {name for name in raw_table if name in declared}
    for name, value in raw_table.items():
        if name in declared:
            resolved[name] = _coerce(f"{path}.{name}", declared[name], value, units, base_dir)
        elif f"{path}.{name}" in schema.TABLES:
            continue  # a declared sub-table; walked by the caller
        else:
            sub_tables = [p.rsplit(".", 1)[1] for p in schema.TABLES if p.startswith(f"{path}.")]
            _unknown_key(path, name, list(declared) + sub_tables, version, warnings)
    for name, key in declared.items():
        if name in resolved:
            continue
        override = (overrides or {}).get(name, UNSET)
        if override is not UNSET:
            resolved[name] = _normalize_default(f"{path}.{name}", key, override, base_dir)
        elif key.default is not UNSET:
            resolved[name] = _normalize_default(f"{path}.{name}", key, key.default, base_dir)
        elif key.deduced_from is None:
            raise InputError(f"[{path}].{name} is required. {key.doc.splitlines()[0]}")
    return resolved


def _select_variant(root, raw, version, warnings):
    """Resolve a tagged union by which sub-table is present."""
    variants = schema.variants_of(root)
    names = [v.rsplit(".", 1)[1] for v in variants]
    raw_root = raw.get(root, {})
    if not isinstance(raw_root, dict):
        raise InputError(f"[{root}]: expected a table with one of the sub-tables {names}")
    present = [name for name in names if isinstance(raw_root.get(name), dict)]
    stray = [k for k, v in raw_root.items() if not isinstance(v, dict)]
    if stray:
        raise InputError(
            f"[{root}]: keys {sorted(stray)} sit directly in [{root}], but this section is "
            f"selected by which sub-table is present. Write [{root}.{names[0]}] and put them there."
        )
    if len(present) > 1:
        raise InputError(f"[{root}]: {sorted(present)} are mutually exclusive; give exactly one.")
    return present[0] if present else None


#: Tables that always apply, whichever calculation is selected.
_GLOBAL_TABLES = ("format", "run", "units", "rotation_to_spherical", "temperature", "many_body_basis", "solver")


def _resolve_environment(raw_env, warnings):
    """Validate ``[environment]`` against the knob registry, parsing and clamping each value.

    Validating the *name* is not enough. ``Knob.get`` clamps silently, and its boolean parser
    treats every string except a short false-list as true -- so ``"no"`` would mean ``True``.
    Both are fine for an environment variable a user sets deliberately; neither is acceptable
    for a value written in a file that claims to describe the run.
    """
    knobs = dict(config.KNOBS)
    resolved = {}
    for name, value in raw_env.items():
        if name not in knobs:
            hint = _suggest(name, knobs)
            if not hint and name.isupper():
                hint = (
                    " Only knobs in impurityModel.ed.config can be set here; variables read by "
                    "other libraries (OMP_NUM_THREADS, MKL_NUM_THREADS, ...) are consumed long "
                    "before this runs, so setting them here would silently do nothing."
                )
            raise InputError(f"[environment]: {name!r} is not a tuning knob.{hint}")
        knob = knobs[name]
        text = str(value).lower() if isinstance(value, bool) else str(value)
        try:
            parsed = config._PARSERS[knob.kind](text)
        except (TypeError, ValueError) as exc:
            raise InputError(f"[environment].{name}: {value!r} is not a valid {knob.kind} ({exc})") from exc
        if knob.kind == "bool" and text not in ("0", "1", "true", "false", "True", "False"):
            raise InputError(
                f"[environment].{name}: {value!r} is ambiguous as a boolean. Write true or false; "
                "the underlying parser treats every other string as true, including 'no'."
            )
        if knob.minimum is not None and parsed < knob.minimum:
            raise InputError(
                f"[environment].{name}: {parsed} is below the knob's minimum {knob.minimum} and "
                "would be silently clamped. Write a value at or above it."
            )
        resolved[name] = text
    return resolved


def _resolve(raw, input_path, raw_text):
    """Validate and resolve an already-parsed TOML mapping. Pure; no I/O, no MPI."""
    warnings = []
    base_dir = str(Path(input_path).resolve().parent)
    version = _check_version(raw, warnings)
    _check_required_features(raw)

    if "units" not in raw or "energy" not in raw.get("units", {}):
        raise InputError(
            "[units].energy is required, with no default: the command-line interface defaults "
            "to eV while RSPt writes Rydberg, so a default here would let two front-ends of "
            f"the same code disagree by {ENERGY_UNITS['Ry']:.4f}x in silence. "
            f"Choose one of {', '.join(repr(u) for u in ENERGY_UNITS)}."
        )
    provided = {}
    units = _resolve_table(
        "units", raw["units"], {"energy": "eV", "temperature": "K"}, base_dir, version, warnings, provided=provided
    )

    present = [name for name in schema.CALCULATIONS if name in raw]
    if len(present) != 1:
        raise InputError(
            f"Exactly one of {list(schema.CALCULATIONS)} must be present; found {present or 'none'}. "
            "The calculation is chosen by which table you write, not by a `type` string."
        )
    calculation = present[0]
    overrides = schema.TABLES[calculation].overrides

    roots = ("hamiltonian", "interaction", "double_counting")
    sources = {root: _select_variant(root, raw, version, warnings) for root in roots}
    if sources["hamiltonian"] is None:
        raise InputError(
            "[hamiltonian] is required; give exactly one of "
            f"{[f'hamiltonian.{s}' for s in schema.HAMILTONIAN_SOURCES]}."
        )
    interaction_kind = sources["interaction"] or "none"
    dc_scheme = sources["double_counting"] or "none"

    wanted = list(_GLOBAL_TABLES)
    wanted += [
        f"hamiltonian.{sources['hamiltonian']}",
        f"interaction.{interaction_kind}",
        f"double_counting.{dc_scheme}",
    ]
    wanted += [calculation] + [p for p in schema.TABLES if p.startswith(f"{calculation}.")]

    tables = {}
    for path in wanted:
        raw_table = raw
        for part in path.split("."):
            raw_table = raw_table.get(part, {}) if isinstance(raw_table, dict) else {}
        tables[path] = _resolve_table(
            path, raw_table, units, base_dir, version, warnings, overrides.get(path), provided=provided
        )

    shells = []
    for index, raw_shell in enumerate(raw.get("shell", [])):
        if not isinstance(raw_shell, dict):
            raise InputError(f"[[shell]] entry {index} is not a table")
        shells.append(_resolve_table("shell", raw_shell, units, base_dir, version, warnings))

    environment = _resolve_environment(raw.get("environment", {}), warnings)

    known_top = (
        set(_GLOBAL_TABLES)
        | set(schema.CALCULATIONS)
        | {"hamiltonian", "interaction", "double_counting", "shell", "environment"}
    )
    for name in raw:
        if name not in known_top:
            if version[1] > schema.SPEC_VERSION[1]:
                warnings.append(f"Ignoring unknown table [{name}] from a newer format minor version.")
            else:
                warnings.append(f"Ignoring unknown table [{name}].{_suggest(name, known_top)}")

    return ResolvedInput(
        tables=tables,
        provided=provided,
        shells=shells,
        environment=environment,
        calculation=calculation,
        hamiltonian_source=sources["hamiltonian"],
        interaction_kind=interaction_kind,
        dc_scheme=dc_scheme,
        version=version,
        path=str(Path(input_path).resolve()),
        raw_text=raw_text,
        warnings=warnings,
    )


#: Double-counting schemes each calculation may use. ``mlft`` is folded into ``h0`` by the
#: spectroscopy model builder and is meaningless elsewhere; the matrix schemes are read only
#: by the self-energy and susceptibility drivers, so one on a spectroscopy run would be a
#: silent no-op -- the driver never looks at ``model.dc``.
_DC_APPLICABILITY = {
    "spectroscopy": ("mlft", "none"),
    "selfenergy": tuple(s for s in schema.DC_SCHEMES if s != "mlft"),
    "susceptibility": tuple(s for s in schema.DC_SCHEMES if s != "mlft"),
}

#: Tables an archive supplies itself. Writing one alongside ``[hamiltonian.archive]`` would
#: be silently ignored: the loader restores the recorded model, meshes and options and the
#: driver honours only the Green's-function kernel and excitation budget on top.
_ARCHIVE_SUPPLIES = ("many_body_basis", "temperature", "shell", "interaction", "double_counting")


def _resolve_tau(resolved, raw):
    """Collapse ``[temperature]`` to the one number the solver wants: ``tau = k_B*T`` in eV.

    The table offers two keys because users think in both, but they are governed by
    *different* units -- Kelvin for one, ``[units].energy`` for the other -- and a table
    carrying two unit governances is how a ``tau = 0.002`` written under ``energy = "Ry"``
    becomes a silent 13.6x temperature error. So exactly one may be given, and the choice is
    read from the *file*, not from the resolved table: the per-calculation default fills the
    other key in, and counting those would reject a legitimate file.

    Defaults are stated in each key's own unit -- 300 for ``kelvin``, an eV value for ``tau``
    -- so the Kelvin default is converted here while the energy one is already internal.
    """
    written = [name for name in ("kelvin", "tau") if name in raw.get("temperature", {})]
    if len(written) > 1:
        raise InputError(
            "[temperature]: give exactly one of `kelvin` and `tau`. They are governed by "
            "different units, and carrying both in one table is how a `tau` written under "
            "[units].energy = 'Ry' becomes a silent 13.6x temperature error."
        )
    if written:
        # Already converted by _coerce: TEMPERATURE applied k_B, ENERGY applied the unit factor.
        return resolved.tables["temperature"][written[0]]
    override = schema.TABLES[resolved.calculation].overrides.get("temperature", {})
    if "tau" in override:
        return override["tau"]
    if "kelvin" in override:
        return override["kelvin"] * k_B
    raise InputError(f"[temperature]: no default for a {resolved.calculation} run; give `kelvin` or `tau`.")


def _cross_check(resolved, raw):
    """Checks that need more than one table, run after everything is resolved."""
    warnings = resolved.warnings

    resolved.tables["temperature"]["tau"] = _resolve_tau(resolved, raw)

    if resolved.dc_scheme not in _DC_APPLICABILITY[resolved.calculation]:
        reason = (
            "the spectroscopy driver never reads model.dc, so this would be a silent no-op"
            if resolved.calculation == "spectroscopy"
            else "mlft is folded into h0 by the spectroscopy model builder and has no meaning here"
        )
        raise InputError(
            f"[double_counting.{resolved.dc_scheme}] cannot be used with [{resolved.calculation}]: "
            f"{reason}. Allowed here: {list(_DC_APPLICABILITY[resolved.calculation])}."
        )

    if resolved.hamiltonian_source == "archive":
        clashing = [name for name in _ARCHIVE_SUPPLIES if name in raw]
        if clashing:
            raise InputError(
                f"[hamiltonian.archive] supplies the model, meshes and recorded options itself, "
                f"so {sorted(clashing)} would be read from the archive and this file's values "
                "ignored. Remove them, or use a different Hamiltonian source."
            )

    roles = [shell["role"] for shell in resolved.shells]
    if roles.count("valence") != 1:
        raise InputError(
            f"Exactly one [[shell]] must have role = \"valence\"; found {roles.count('valence')}. "
            "The role is declared rather than inferred from `l`, because inferring it is the "
            "2p/3d assumption this format has to outlive."
        )
    if roles.count("core") > 1:
        raise InputError(f'At most one [[shell]] may have role = "core"; found {roles.count("core")}.')

    # Every shell dict below this point -- nBaths, nValBaths, impurity_orbitals, the operator
    # labels themselves -- is keyed by `l`, so two shells sharing one silently collapse into
    # the last one written. That used to be unreachable while the solver accepted only the
    # {1, 2} pair; now that any pair of angular momenta is allowed it is a typo away, and it
    # surfaces far downstream as a nominal_occupation that belongs to the other shell.
    seen = [shell["l"] for shell in resolved.shells]
    duplicates = sorted({x for x in seen if seen.count(x) > 1})
    if duplicates:
        raise InputError(
            f"Two [[shell]] tables share l={duplicates[0]}. A shell is identified by its "
            "angular momentum everywhere below this file -- bath counts, orbital layout and "
            "the Hamiltonian's own labels are all keyed by it -- so two shells cannot have "
            "the same l. Give them different l, or describe them as one shell."
        )

    for shell in resolved.shells:
        n_bath, n_valence = shell.get("n_bath"), shell.get("n_valence_bath")
        if n_bath is not None and n_valence is not None and n_valence > n_bath:
            raise InputError(f"[[shell]] l={shell['l']}: n_valence_bath ({n_valence}) exceeds n_bath ({n_bath}).")
        max_occupation = 2 * (2 * shell["l"] + 1)
        if shell["nominal_occupation"] > max_occupation:
            raise InputError(
                f"[[shell]] l={shell['l']}: nominal_occupation {shell['nominal_occupation']} "
                f"exceeds the {max_occupation} spin-orbitals an l={shell['l']} shell has."
            )

    techniques = ()
    if resolved.calculation == "spectroscopy":
        techniques = tuple(
            name for name in ("pes", "xps", "xas", "rixs", "nixs") if resolved.tables[f"spectroscopy.{name}"]["enabled"]
        )
        if not techniques:
            raise InputError("[spectroscopy]: every technique is disabled, so there is nothing to compute.")
        nixs = resolved.tables["spectroscopy.nixs"]
        if nixs["enabled"] and not nixs["radial_file"]:
            raise InputError(
                "[spectroscopy.nixs] is enabled but has no radial_file. NIXS needs the radial "
                "part of the correlated orbitals; it used to be switched on merely by supplying "
                "one, which is why this is now stated explicitly rather than inferred."
            )

    valence = next(shell for shell in resolved.shells if shell["role"] == "valence")
    core = next((shell for shell in resolved.shells if shell["role"] == "core"), None)
    capabilities.check(
        resolved.calculation,
        core_l=None if core is None else core["l"],
        valence_l=valence["l"],
        techniques=techniques,
    )
    return resolved


def _parse(path):
    """Read and fully resolve one file. Runs on rank 0 only."""
    text = Path(path).read_text()
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise InputError(f"{path}: {exc}") from exc
    return _cross_check(_resolve(raw, path, text), raw)


#: Exceptions that survive the broadcast with their type intact, so a caller can still
#: distinguish "your file is wrong" from "the solver cannot do this yet".
_RAISEABLE = {
    "InputError": InputError,
    "UnsupportedCalculation": capabilities.UnsupportedCalculation,
    "InvalidShellCombination": capabilities.InvalidShellCombination,
}


def load_input(path, comm=None):
    """Load, validate and resolve an input file, identically on every rank.

    Rank 0 does all the I/O and validation; the outcome -- a resolved input *or* an error --
    is then broadcast, and every rank acts on the broadcast verdict. No rank ever branches on
    something only it can see, which is the property that keeps a bad file from turning into
    a hang instead of an error message.

    Parameters
    ----------
    path : str or pathlib.Path
        The input file.
    comm : MPI communicator, optional
        When given, rank 0 parses and broadcasts. When ``None`` (a serial run, or a caller
        that has already broadcast the path) this rank parses for itself.

    Returns
    -------
    ResolvedInput

    Raises
    ------
    InputError
        Malformed, contradictory or unrecognised input. Raised on *every* rank.
    capabilities.UnsupportedCalculation
        Valid input the current solver cannot run.
    """
    if comm is None or comm.rank == 0:
        try:
            payload = ("ok", _parse(path))
        except Exception as exc:  # re-raised below, on every rank
            payload = ("error", type(exc).__name__, f"{exc}")
    else:
        payload = None

    if comm is not None:
        payload = comm.bcast(payload, root=0)

    if payload[0] == "error":
        _, name, message = payload
        raise _RAISEABLE.get(name, InputError)(message)
    return payload[1]


# --------------------------------------------------------------------------------------
# [environment] on its own, for callers that are not the impurityModel CLI
# --------------------------------------------------------------------------------------

#: Conventional filename looked for in the working directory. RSPt drives the solver through
#: an ``@ffi.def_extern`` callback whose whole configuration surface is two fixed-width
#: strings and a label -- there is no argument to thread a path through, and RSPt's own
#: sources are not ours to change -- so a convention is the only channel available.
ENVIRONMENT_FILENAME = "impurityModel.toml"

#: Environment variable overriding :data:`ENVIRONMENT_FILENAME`.
ENVIRONMENT_PATH_VAR = "IMPURITYMODEL_INPUT"


def find_environment_file(directory="."):
    """Path of the conventional input file, or ``None`` if there is not one."""
    override = os.environ.get(ENVIRONMENT_PATH_VAR)
    if override:
        return override if Path(override).is_file() else None
    candidate = Path(directory, ENVIRONMENT_FILENAME)
    return str(candidate) if candidate.is_file() else None


def load_environment(path, comm=None):
    """Return just the ``[environment]`` table of ``path``, validated against the registry.

    Only that table is read. The rest of the file describes a model, and on the RSPt path
    RSPt supplies the model itself -- reading more would create two sources of truth for the
    same physics.
    """
    if comm is None or comm.rank == 0:
        try:
            raw = tomllib.loads(Path(path).read_text())
            payload = ("ok", _resolve_environment(raw.get("environment", {}), []))
        except Exception as exc:
            payload = ("error", type(exc).__name__, f"{exc}")
    else:
        payload = None
    if comm is not None:
        payload = comm.bcast(payload, root=0)
    if payload[0] == "error":
        raise _RAISEABLE.get(payload[1], InputError)(payload[2])
    return payload[1]


@contextmanager
def apply_environment(mapping, override=True):
    """Set tuning knobs in ``os.environ`` for the duration of the block, then restore.

    Restoring matters on the RSPt path, where the callback runs once per cluster label per
    self-consistency iteration plus once more for the double-counting pass: without a
    restore, iteration two silently inherits iteration one's knobs.

    Parameters
    ----------
    mapping : dict
        ``{knob_name: string_value}``, e.g. from :func:`load_environment`.
    override : bool
        When False, a variable already set in the environment wins and is reported as
        skipped. That is the rule ``impurityModel_interface`` already follows for
        ``OMP_NUM_THREADS``: a value the user set in their shell should not be quietly
        overridden by a file.

    Yields
    ------
    list of str
        Names that were skipped because they were already set (empty when ``override``).
    """
    previous = {}
    skipped = []
    try:
        for name, value in mapping.items():
            if not override and name in os.environ:
                skipped.append(name)
                continue
            previous[name] = os.environ.get(name)
            os.environ[name] = str(value)
        yield skipped
    finally:
        for name, old in previous.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old
