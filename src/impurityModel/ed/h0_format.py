"""
Reader and writer for the ``.h0`` interchange format (see ``doc/h0_file_format.md``).

The non-interacting impurity Hamiltonian as produced by upstream tooling (``rspt2spectra``'s
``build_h0``): a JSON header carrying the orbital layout, units, energy reference and basis,
followed by one ``i j re im`` line per stored matrix element.

Indices are **flat** single-particle spin-orbital indices with the impurity block first --
the convention :class:`~impurityModel.ed.model.ImpurityModel` holds internally -- *not* the
``(l, s, m)`` / ``(l, b)`` labels that :mod:`impurityModel.ed.op_parser` reads and ``c2i``
maps. The two orderings differ, so a file in one format read as the other is silently wrong
rather than an error; that is the mistake this module exists to make impossible.

A leaf module: numpy and the standard library only, no MPI, no collectives. The reader is a
pure function of the file bytes and therefore rank-invariant by construction.
"""

import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

#: Highest format version this module understands.
SPEC_VERSION = 1

#: The magic first line. Also the discriminator against the legacy bare-integer format --
#: the file extension is not, since legacy files renamed to .dat/.h0 are common.
MAGIC_RE = re.compile(r"^#\s*impurityModel-h0\s+v(\d+)\s*$")

#: Default relative threshold below which a matrix element pair is dropped. Matches the
#: solver's own noise floor (BREAKDOWN_TOL, rotate_hamiltonian's tol).
DEFAULT_DROP_TOLERANCE = 1e-12

#: Hermiticity check tolerance, relative to max|H|. Two ULP of a float64 mantissa.
HERMITICITY_RTOL = 2e-15

#: Bath one-body blocks are diagonal (star) or tridiagonal (chain); used to validate a
#: caller-supplied n_imp for the legacy format.
MAX_BATH_BANDWIDTH = 1

RY_TO_EV = 13.605693122994232

_KNOWN_UNITS = {"eV": 1.0, "Ry": RY_TO_EV}

#: Header keys whose meaning a reader must understand to interpret the term list. Anything
#: listed in a file's "required_features" that is not here makes the reader raise.
_SUPPORTED_FEATURES = frozenset(
    {
        "unit",
        "energy_reference",
        "index_convention",
        "spin_ordering",
        "shell_layout",
        "storage",
    }
)

_SUPPORTED_INDEX_CONVENTIONS = frozenset({"impurity-block-first"})
_SUPPORTED_STORAGE = frozenset({"full"})


class H0FormatError(ValueError):
    """Raised for any malformed or unsupported ``.h0`` input.

    Subclasses :class:`ValueError` so callers that only care that parsing failed need not
    know about this module.
    """


@dataclass(frozen=True, eq=False)
class H0File:
    """Parsed contents of an ``.h0`` (or legacy) Hamiltonian file.

    Attributes
    ----------
    h0 : dict
        Non-interacting Hamiltonian in single-index operator form,
        ``{((i, "c"), (j, "a")): amplitude}``, in eV.
    n_orb : int
        Total number of spin-orbitals (impurity + bath).
    impurity_orbitals : dict
        ``{group: tuple_of_indices}``. A map, so multi-shell layouts are representable.
    valence_bath, conduction_bath : tuple of int or None
        Advisory bath classification; ``None`` when the producer did not record one.
    rot_to_spherical : numpy.ndarray or None
        ``n_imp x n_imp`` rotation from the impurity basis to spherical harmonics.
    unit : str
        Always ``"eV"`` -- amplitudes are converted on read.
    energy_reference : str
        ``"fermi"`` or ``"absolute"``. See ``doc/h0_file_format.md``.
    basis : str
        Impurity orbital basis: ``"spherical"``, ``"cubic"`` or ``"unknown"``.
    contains_soc : bool
        Whether spin-orbit coupling is already present in the amplitudes.
    interaction : dict or None
        Optional interaction spec, e.g. ``{"kind": "slater", "l": 2, "F": [...]}``.
    header : dict
        The raw header, for keys this dataclass does not promote.
    """

    h0: dict
    n_orb: int
    impurity_orbitals: dict
    valence_bath: Optional[tuple] = None
    conduction_bath: Optional[tuple] = None
    rot_to_spherical: Optional[np.ndarray] = None
    unit: str = "eV"
    energy_reference: str = "fermi"
    basis: str = "unknown"
    contains_soc: bool = False
    interaction: Optional[dict] = None
    header: dict = field(default_factory=dict)

    @property
    def impurity_indices(self) -> list:
        """Sorted flat list of impurity spin-orbital indices across all groups."""
        return sorted(orb for orbs in self.impurity_orbitals.values() for orb in orbs)

    @property
    def n_imp(self) -> int:
        """Number of impurity spin-orbitals."""
        return len(self.impurity_indices)

    def to_matrix(self) -> np.ndarray:
        """Dense ``(n_orb, n_orb)`` Hamiltonian."""
        h = np.zeros((self.n_orb, self.n_orb), dtype=complex)
        for process, value in self.h0.items():
            (i, _), (j, _) = process
            h[i, j] = value
        return h


def _fail(path, lineno, expected, line=None):
    """Raise :class:`H0FormatError` with the file, the 1-based line and what was expected."""
    where = f"{path}:{lineno}: " if lineno is not None else f"{path}: "
    got = f", got {line!r}" if line is not None else ""
    raise H0FormatError(f"{where}{expected}{got}")


def parse_flat_terms(lines, *, path="<string>", start_lineno=1):
    """Parse ``i j re im`` lines into ``{(i, j): amplitude}``.

    Blank lines and lines whose first non-blank character is ``#`` are skipped. A repeated
    ``(i, j)`` is an error rather than an accumulation: floating addition is not associative,
    so summing duplicates makes the result order-dependent and can break hermiticity
    non-deterministically when ``(i, j)`` and ``(j, i)`` are listed in different orders.

    Parameters
    ----------
    lines : iterable of str
        The term lines.
    path : str, optional
        Used only in error messages.
    start_lineno : int, optional
        1-based line number of ``lines[0]`` in ``path``.

    Returns
    -------
    dict
        ``{(i, j): complex}``.

    Raises
    ------
    H0FormatError
        On a malformed line or a duplicated index pair.
    """
    terms = {}
    seen_at = {}
    for offset, raw in enumerate(lines):
        lineno = start_lineno + offset
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 4:
            _fail(path, lineno, f"expected 4 whitespace-separated fields 'i j re im', got {len(fields)}", line)
        try:
            i, j = int(fields[0]), int(fields[1])
        except ValueError:
            _fail(path, lineno, "expected integer orbital indices in 'i j re im'", line)
        try:
            re_part, im_part = float(fields[2]), float(fields[3])
        except ValueError:
            _fail(path, lineno, "expected real amplitudes in 'i j re im'", line)
        if not (np.isfinite(re_part) and np.isfinite(im_part)):
            _fail(path, lineno, "amplitudes must be finite", line)
        if (i, j) in seen_at:
            _fail(path, lineno, f"duplicate index pair ({i}, {j}), first seen on line {seen_at[(i, j)]}", line)
        seen_at[(i, j)] = lineno
        terms[(i, j)] = complex(re_part, im_part)
    return terms


def check_hermitian(terms, *, path, atol=None):
    """Raise unless ``terms`` is Hermitian, naming the offending element.

    Deliberately *not* :func:`operator_algebra.assert_hermitian`, which is a bare ``assert``
    on exact dict equality: it is removed under ``python -O``, it compares complex floats
    exactly, and its failure message is the source line rather than the offending element.

    Parameters
    ----------
    terms : dict
        ``{(i, j): complex}``.
    path : str
        Used in the error message.
    atol : float, optional
        Absolute tolerance. Defaults to ``HERMITICITY_RTOL * max|H|``.

    Raises
    ------
    H0FormatError
        If any element deviates from its conjugate partner by more than ``atol``, including
        the case where the partner is absent entirely.
    """
    if not terms:
        return
    scale = max(abs(v) for v in terms.values())
    if atol is None:
        atol = HERMITICITY_RTOL * max(scale, 1.0)
    for (i, j), value in terms.items():
        partner = terms.get((j, i), 0.0)
        deviation = abs(value - np.conj(partner))
        if deviation > atol:
            missing = " (conjugate partner absent)" if (j, i) not in terms else ""
            raise H0FormatError(
                f"{path}: not Hermitian: |H[{i},{j}] - conj(H[{j},{i}])| = {deviation:.3e} > {atol:.1e}{missing}"
            )


def _terms_to_operator(terms):
    """``{(i, j): v}`` -> the single-index operator form ``{((i, "c"), (j, "a")): v}``."""
    return {((i, "c"), (j, "a")): value for (i, j), value in terms.items()}


def _split_header(text, path):
    """Return ``(version, header_dict, rest_text, terms_start_lineno)``.

    The header's extent comes from JSON's own grammar rather than from a ``--`` sentinel
    line, which would break on CRLF, on trailing whitespace and on ``--`` inside a string.
    """
    lines = text.splitlines()
    first_idx = next((k for k, line in enumerate(lines) if line.strip()), None)
    if first_idx is None:
        _fail(path, None, "file is empty")
    match = MAGIC_RE.match(lines[first_idx].strip())
    if match is None:
        _fail(path, first_idx + 1, "expected the magic line '# impurityModel-h0 v1'", lines[first_idx])
    version = int(match.group(1))
    if version > SPEC_VERSION:
        _fail(
            path,
            first_idx + 1,
            f"file is format v{version} but this impurityModel understands at most v{SPEC_VERSION};"
            " upgrade impurityModel or regenerate the file. See doc/h0_file_format.md",
        )

    brace = text.find("{", sum(len(line) + 1 for line in lines[: first_idx + 1]))
    if brace < 0:
        _fail(path, first_idx + 2, "expected a JSON header object after the magic line")
    try:
        header, end = json.JSONDecoder().raw_decode(text, brace)
    except json.JSONDecodeError as exc:
        _fail(path, first_idx + 2, f"malformed JSON header ({exc.msg})")
    if not isinstance(header, dict):
        _fail(path, first_idx + 2, "the JSON header must be an object")
    return version, header, text[end:], text.count("\n", 0, end) + 1


def _validate_header(header, path):
    """Check the declared features and conventions; raise on anything unrecognised."""
    unknown = sorted(set(header.get("required_features", [])) - _SUPPORTED_FEATURES)
    if unknown:
        _fail(
            path,
            None,
            f"file requires header features this impurityModel does not implement: {unknown}. "
            f"Producer {header.get('producer')}, spec v{SPEC_VERSION}. See doc/h0_file_format.md",
        )
    convention = header.get("index_convention", "impurity-block-first")
    if convention not in _SUPPORTED_INDEX_CONVENTIONS:
        _fail(
            path,
            None,
            f"unsupported index_convention {convention!r}; expected one of {sorted(_SUPPORTED_INDEX_CONVENTIONS)}",
        )
    storage = header.get("storage", "full")
    if storage not in _SUPPORTED_STORAGE:
        _fail(path, None, f"unsupported storage {storage!r}; expected one of {sorted(_SUPPORTED_STORAGE)}")
    if "unit" not in header:
        _fail(path, None, "header is missing the mandatory 'unit' key")
    if header["unit"] not in _KNOWN_UNITS:
        _fail(path, None, f"unknown unit {header['unit']!r}; expected one of {sorted(_KNOWN_UNITS)}")
    if "energy_reference" not in header:
        _fail(path, None, "header is missing the mandatory 'energy_reference' key")


def _rot_from_header(header):
    """Decode ``rot_to_spherical`` from nested ``[re, im]`` pairs."""
    raw = header.get("rot_to_spherical")
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=float)
    if arr.ndim != 3 or arr.shape[-1] != 2:
        raise H0FormatError(f"rot_to_spherical must be a (n, n, 2) nest of [re, im] pairs, got shape {arr.shape}")
    return arr[..., 0] + 1j * arr[..., 1]


def read_h0_file(path):
    """Read a ``.h0`` file.

    Parameters
    ----------
    path : str or pathlib.Path
        The file to read.

    Returns
    -------
    H0File
        Amplitudes converted to eV.

    Raises
    ------
    H0FormatError
        On a malformed header or term list, an unsupported version or feature, an index
        outside ``n_orb``, or a non-Hermitian Hamiltonian.
    """
    path = str(path)
    raw = Path(path).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    text = raw.decode("utf-8")

    _version, header, rest, terms_start = _split_header(text, path)
    _validate_header(header, path)

    # The `--` separator is a human-facing convenience, not a delimiter -- the header's
    # extent already came from JSON's grammar. Consume it if present, blanking rather than
    # removing the line so reported line numbers still match the file.
    term_lines = rest.splitlines()
    for k, line in enumerate(term_lines):
        if not line.strip():
            continue
        if line.strip() == "--":
            term_lines[k] = ""
        break

    terms = parse_flat_terms(term_lines, path=path, start_lineno=terms_start)

    n_orb = header.get("n_orb")
    if n_orb is None:
        n_orb = (max(max(k) for k in terms) + 1) if terms else 0
    n_orb = int(n_orb)
    for i, j in terms:
        if not (0 <= i < n_orb and 0 <= j < n_orb):
            raise H0FormatError(f"{path}: index pair ({i}, {j}) is outside n_orb = {n_orb}")

    check_hermitian(terms, path=path)

    scale = _KNOWN_UNITS[header["unit"]]
    if scale != 1.0:
        terms = {key: value * scale for key, value in terms.items()}

    # JSON object keys are always strings; restore integer group labels so the map matches
    # ImpurityModel.impurity_orbitals, whose groups are keyed by the shell's l.
    impurity_orbitals = {
        (int(key) if str(key).lstrip("-").isdigit() else key): tuple(int(o) for o in orbs)
        for key, orbs in header.get("impurity_orbitals", {}).items()
    }
    if not impurity_orbitals:
        _fail(path, None, "header is missing 'impurity_orbitals'")

    return H0File(
        h0=_terms_to_operator(terms),
        n_orb=n_orb,
        impurity_orbitals=impurity_orbitals,
        valence_bath=_as_index_tuple(header.get("valence_bath")),
        conduction_bath=_as_index_tuple(header.get("conduction_bath")),
        rot_to_spherical=_rot_from_header(header),
        unit="eV",
        energy_reference=header["energy_reference"],
        basis=header.get("basis", "unknown"),
        contains_soc=bool(header.get("contains_soc", False)),
        interaction=header.get("interaction"),
        header=header,
    )


def _as_index_tuple(value):
    """``None`` stays ``None``; anything else becomes a tuple of ints."""
    return None if value is None else tuple(int(v) for v in value)


def infer_n_impurity_orbitals(terms, n_orb):
    """Smallest ``k`` whose trailing block ``H[k:, k:]`` has bandwidth <= 1.

    The impurity block is dense and hybridizes to every bath orbital, while a star bath block
    is diagonal and a chain bath block tridiagonal -- so the first split that leaves a
    narrow-banded remainder is the impurity/bath boundary. Over the stored legacy workloads
    this recovers ``n_imp`` correctly in every case, with a wide margin.

    Used to *check* a caller-supplied ``n_imp``, never to infer one silently: the bandwidth
    assumption is verified for star and chain geometries only.

    Parameters
    ----------
    terms : dict
        ``{(i, j): complex}``.
    n_orb : int
        Total spin-orbitals.

    Returns
    -------
    int or None
        The recovered boundary, or ``None`` if no split qualifies.
    """
    present = [(i, j) for (i, j), value in terms.items() if value != 0]
    for k in range(1, n_orb):
        bandwidth = max((abs(i - j) for i, j in present if i >= k and j >= k), default=0)
        if bandwidth <= MAX_BATH_BANDWIDTH:
            return k
    return None


def read_legacy_flat_h0(path, n_imp, *, verify_layout=True):
    """Read the legacy bare-integer ``i j re im`` file written by older ``build_h0``.

    The legacy file carries no header, so ``n_imp`` must come from the caller. It is
    validated against the sparsity pattern (see :func:`infer_n_impurity_orbitals`) rather
    than trusted. Both triangles are stored and were filtered elementwise by the producer, so
    the result is symmetrized on load.

    Parameters
    ----------
    path : str or pathlib.Path
        The file to read.
    n_imp : int
        Number of impurity spin-orbitals (the leading block dimension).
    verify_layout : bool, optional
        Check ``n_imp`` against the sparsity pattern. Default True.

    Returns
    -------
    H0File
        ``unit`` and ``energy_reference`` are unknown for this format and are reported as
        given by the producer's convention, i.e. not validated.

    Raises
    ------
    H0FormatError
        On a malformed line, or when ``n_imp`` contradicts the sparsity pattern.
    """
    path = str(path)
    text = Path(path).read_text(encoding="utf-8")
    terms = parse_flat_terms(text.splitlines(), path=path)
    if not terms:
        _fail(path, None, "file contains no terms")

    n_orb = max(max(k) for k in terms) + 1
    n_imp = int(n_imp)
    if not 0 < n_imp <= n_orb:
        raise H0FormatError(f"{path}: n_imp = {n_imp} is not in 1..{n_orb}")

    if verify_layout:
        recovered = infer_n_impurity_orbitals(terms, n_orb)
        if recovered is not None and recovered != n_imp:
            raise H0FormatError(
                f"{path}: n_imp = {n_imp} disagrees with the sparsity pattern. The bath block "
                f"H[{n_imp}:, {n_imp}:] is not diagonal/tridiagonal; the smallest consistent value is "
                f"n_imp = {recovered}. Pass n_imp={recovered} if that is correct, or verify_layout=False "
                f"for a bath geometry this check does not cover."
            )

    warnings.warn(
        f"{path} is in the legacy bare-integer h0 format, which records no unit, energy "
        f"reference, basis or orbital layout. Regenerate it with build_h0 to get a .h0 file; "
        f"see doc/h0_file_format.md.",
        DeprecationWarning,
        stacklevel=2,
    )

    # The producer applied its zero-filter elementwise to a matrix that is not bitwise
    # Hermitian, so the two triangles can disagree -- one shipped file has an element whose
    # conjugate partner was dropped entirely. Symmetrize rather than reject.
    symmetric = {}
    for i, j in set(terms) | {(j, i) for i, j in terms}:
        symmetric[(i, j)] = 0.5 * (terms.get((i, j), 0.0) + np.conj(terms.get((j, i), 0.0)))

    return H0File(
        h0=_terms_to_operator({key: value for key, value in symmetric.items() if value != 0}),
        n_orb=n_orb,
        impurity_orbitals={0: tuple(range(n_imp))},
        unit="unknown",
        energy_reference="unknown",
        header={},
    )


def write_h0_file(path, h_matrix, *, impurity_orbitals, unit="eV", energy_reference="fermi", **header_extra):
    """Write a dense Hamiltonian as a ``.h0`` file.

    Symmetrizes (``(H + H^dagger)/2``, bitwise Hermitian in float64), forces the diagonal
    real, drops negligible element *pairs*, and writes round-trip-exact floats. The exact
    inverse of :func:`read_h0_file` for a file already in eV.

    Parameters
    ----------
    path : str or pathlib.Path
        Output file.
    h_matrix : numpy.ndarray
        Dense ``(n, n)`` Hamiltonian, impurity block first.
    impurity_orbitals : dict
        ``{group: indices}``.
    unit : str, optional
        Energy unit of ``h_matrix``. Recorded verbatim; no conversion is performed here.
    energy_reference : str, optional
        ``"fermi"`` or ``"absolute"``.
    **header_extra
        Any further header keys (``basis``, ``rot_to_spherical``, ``interaction``, ...).

    Raises
    ------
    H0FormatError
        If ``h_matrix`` holds a non-finite value.
    """
    h = np.asarray(h_matrix, dtype=complex)
    if not np.isfinite(h).all():
        raise H0FormatError("h_matrix holds non-finite values, which the .h0 format forbids")
    h = 0.5 * (h + h.conj().T)
    np.fill_diagonal(h, h.diagonal().real)

    n_orb = h.shape[0]
    drop_tolerance = float(header_extra.pop("drop_tolerance", DEFAULT_DROP_TOLERANCE))
    scale = np.abs(h).max() if h.size else 0.0
    cutoff = drop_tolerance * scale

    header = {
        "version": SPEC_VERSION,
        "required_features": ["unit", "energy_reference", "index_convention", "storage"],
        "unit": unit,
        "energy_reference": energy_reference,
        "n_orb": int(n_orb),
        "index_convention": "impurity-block-first",
        "storage": "full",
        "impurity_orbitals": {str(k): [int(o) for o in orbs] for k, orbs in impurity_orbitals.items()},
        "drop_tolerance": drop_tolerance,
    }
    rot = header_extra.pop("rot_to_spherical", None)
    if rot is not None:
        rot = np.asarray(rot, dtype=complex)
        header["rot_to_spherical"] = [[[float(v.real), float(v.imag)] for v in row] for row in rot]
    header.update(header_extra)

    lines = [f"# impurityModel-h0 v{SPEC_VERSION}", json.dumps(header, allow_nan=False), "--"]
    for i in range(n_orb):
        for j in range(n_orb):
            # Pairwise, so the two triangles can never end up with different key sets.
            if max(abs(h[i, j]), abs(h[j, i])) <= cutoff:
                continue
            # complex() first: repr of a numpy scalar is "np.float64(...)", which is not a
            # number as far as any reader of this format is concerned.
            value = complex(h[i, j])
            lines.append(f"{i} {j} {value.real!r} {value.imag!r}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
