"""Declarative schema for the TOML input format.

Every key is declared exactly once here, with its value *kind*, default, and the rationale
for that default -- the same single-declaration discipline :mod:`impurityModel.ed.config`
uses for the environment knobs, and for the same reason: a default that lives in two places
drifts. Both the reader and ``doc/input_format.md`` are generated from these declarations,
so documentation cannot fall behind the code.

The **kind** is what makes unit conversion safe. ``[units].energy`` converts every key of
kind :attr:`Kind.ENERGY` (and its list/vector forms) and nothing else, so a dimensionless
multiplier such as ``energy_cut`` or an inverse length such as the NIXS ``q`` cannot be
silently scaled by 13.6. The hand-maintained ``_ENERGY_FIELDS`` tuples in
:mod:`impurityModel.scripts.spectra` are exactly the rot this replaces.

A leaf module: standard library only. No numpy, no MPI, no solver imports -- the knob
registry that ``[environment]`` is validated against is consulted by the reader, not here.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

__all__ = [
    "SPEC_VERSION",
    "Kind",
    "Key",
    "Table",
    "TABLES",
    "CALCULATIONS",
    "HAMILTONIAN_SOURCES",
    "INTERACTION_KINDS",
    "DC_SCHEMES",
    "UNSET",
    "dump",
]

#: Format version written as ``[format].version = [major, minor]``. A file whose *major*
#: exceeds this is refused. The *minor* discriminates unknown keys: at or below ours an
#: unknown key can only be a typo (error); above ours it may be a legitimate future key
#: (warn and ignore). See :func:`impurityModel.inputformat.reader.load_input`.
SPEC_VERSION = (1, 0)


class _Unset:
    """Sentinel for "no default"; distinct from ``None``, which is a meaningful default."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "UNSET"

    def __bool__(self):
        return False


#: The "no default, must be supplied" marker.
UNSET = _Unset()


class Kind(Enum):
    """What a value *is*, which decides how (and whether) it is converted and validated.

    Only :attr:`ENERGY`, :attr:`ENERGY_LIST` and :attr:`ENERGY_VECTOR` are touched by
    ``[units].energy``. :attr:`TEMPERATURE` is governed by ``[units].temperature`` instead,
    and everything else is converted by nothing at all -- which is the point.
    """

    #: A scalar energy, converted from ``[units].energy`` to eV on read.
    ENERGY = "energy"
    #: A list of energies (e.g. Slater-Condon parameters), converted elementwise.
    ENERGY_LIST = "energy list"
    #: A 3-vector of energies (the Zeeman splitting), converted elementwise.
    ENERGY_VECTOR = "energy vector"
    #: A temperature, governed by ``[units].temperature`` (Kelvin or an energy).
    TEMPERATURE = "temperature"
    #: A non-negative integer count. Never converted.
    COUNT = "count"
    #: A pure number: a ratio, a tolerance, a multiplier. Never converted. ``energy_cut``
    #: lives here despite its name -- it is a multiple of ``k_B * T``.
    DIMENSIONLESS = "dimensionless"
    #: An inverse length, reciprocal to the radial mesh it is used with (NIXS ``q``).
    INVERSE_LENGTH = "inverse length"
    #: A filesystem path, resolved relative to the directory holding the input file.
    PATH = "path"
    #: One of a fixed set of strings (``choices``).
    ENUM = "enum"
    #: A boolean.
    BOOL = "bool"
    #: A frequency mesh: ``{min, max, n}``, ``{values = [...]}`` or ``{file = "..."}``.
    MESH = "mesh"
    #: A matrix: ``{shape, real, imag}``, ``{identity}``, ``{diagonal}``, ``{scalar}`` or
    #: ``{path}``. See the reader for the encoding.
    MATRIX = "matrix"
    #: A list of 3-vectors that are not energies (NIXS momentum transfers).
    VECTOR_LIST = "vector list"
    #: Free text.
    STRING = "string"
    #: ``"auto"`` (derive at the call site), ``"none"`` (disable), or an integer.
    AUTO_COUNT = "auto/count"
    #: ``"auto"`` (the solver's own default) or a string mode.
    AUTO_ENUM = "auto/enum"
    #: A ``[major, minor]`` format version.
    VERSION = "version"
    #: A list of free-text strings (``required_features``).
    STRING_LIST = "string list"


@dataclass(frozen=True)
class Key:
    """One declared key.

    Parameters
    ----------
    name : str
        The key as it appears in the TOML table.
    kind : Kind
        Value kind; decides conversion and validation.
    default : Any, optional
        Value when the key is absent. :data:`UNSET` means the key must be supplied (or
        deduced -- see ``deduced_from``).
    doc : str
        What the key does, and why its default is what it is.
    choices : tuple of str, optional
        Allowed values for :attr:`Kind.ENUM` / :attr:`Kind.AUTO_ENUM`.
    deduced_from : str, optional
        Human-readable description of where the value comes from when it is absent and has
        no static default (e.g. the ``.h0`` header). Rendered in the generated docs and in
        the error raised when the deduction is not available.
    minimum : int or float, optional
        Lower bound, checked after parsing.
    """

    name: str
    kind: Kind
    default: Any = UNSET
    doc: str = ""
    choices: Optional[tuple] = None
    deduced_from: Optional[str] = None
    minimum: Any = None

    @property
    def required(self) -> bool:
        """Whether a file must supply this key outright (no default, no deduction)."""
        return self.default is UNSET and self.deduced_from is None


@dataclass(frozen=True)
class Table:
    """One declared table.

    Parameters
    ----------
    path : str
        Dotted table path, e.g. ``"hamiltonian.file"``.
    doc : str
        What the table configures.
    keys : tuple of Key
        Declared keys.
    repeatable : bool
        True for a TOML array of tables (``[[shell]]``).
    variant_of : str, optional
        When set, this table is one arm of a tagged union rooted at that path: the *presence
        of the sub-table* selects the variant, so a stale key from another arm cannot sit
        around being silently ignored (which a ``source = "..."`` string tag would allow).
    """

    path: str
    doc: str
    keys: tuple = ()
    repeatable: bool = False
    variant_of: Optional[str] = None
    overrides: dict = field(default_factory=dict)


#: The three calculations. Exactly one of these tables must be present; its *presence* is
#: what selects the calculation (no ``type = "..."`` string -- same reasoning as `variant_of`).
CALCULATIONS = ("spectroscopy", "selfenergy", "susceptibility")

#: Hamiltonian source variants, as sub-tables of ``[hamiltonian]``.
HAMILTONIAN_SOURCES = ("file", "crystal_field", "archive", "blocks", "matrix")

#: Interaction variants, as sub-tables of ``[interaction]``.
INTERACTION_KINDS = ("slater", "u4_file", "none")

#: Double-counting variants, as sub-tables of ``[double_counting]``. ``mlft`` is not a
#: double-counting *matrix* at all -- it is RSPt's charge-transfer correction ``c``, folded
#: into ``h0`` with a ``+`` sign by ``ImpurityModel.from_shells`` -- which is exactly why it
#: is its own tag instead of sharing a ``value`` key with the others.
DC_SCHEMES = (
    "mlft",
    "fll",
    "amf",
    "nominal",
    "sigma_inf",
    "fixed_occupation",
    "fixed_peak",
    "fixed_gap",
    "none",
)

#: Schemes that run a search: each costs 1-15 full collective ground-state solves, and their
#: collectives run on ``MPI.COMM_WORLD`` regardless of any ``comm`` argument
#: (``ed/dc_criteria.py:675``).
DC_SEARCH_SCHEMES = ("fixed_occupation", "fixed_peak", "fixed_gap")

#: Keys shared by every searching double-counting variant.
_DC_SEARCH_KEYS = (
    Key("guess", Kind.ENERGY, 0.0, "Starting double counting for the search."),
    Key(
        "on_unreachable",
        Kind.ENUM,
        "abort",
        "What to do when the target has no solution -- a plateau, or a target the observable "
        "steps across at a charge-sector boundary. This is the *expected* outcome of a "
        "fixed-occupation search on a charge-transfer insulator, so it is a modelling verdict, "
        "not necessarily a bug: 'keep_guess' proceeds loudly with the guess, 'abort' stops.",
        choices=("abort", "keep_guess"),
    ),
    Key(
        "damping",
        Kind.DIMENSIONLESS,
        1.0,
        "Mixing against the previous answer, dc = dc_prev + damping * (dc_found - dc_prev). "
        "RSPt's 'alpha' on the double-counting line, where it defaults to 0.5 because an "
        "undamped Newton step on a target that moves each CSC iteration is a limit-cycle "
        "generator. 1.0 (no damping) here, since a standalone run has no outer loop.",
        minimum=0.0,
    ),
    Key("occ_tol", Kind.DIMENSIONLESS, 1e-2, "Occupation convergence tolerance.", minimum=0.0),
    Key("initial_step", Kind.ENERGY, 0.25, "First trial step of the shift search."),
    Key("max_shift", Kind.ENERGY, 20.0, "Largest |mu| the search will try before giving up."),
)

_TABLE_LIST = [
    Table(
        "format",
        "Format version and forward-compatibility declarations.",
        (
            Key(
                "version",
                Kind.VERSION,
                list(SPEC_VERSION),
                "[major, minor]. A major above the reader's is refused outright. The minor "
                "decides how an unknown key is treated: at or below ours it can only be a "
                "typo (error), above ours it may be a future key (warn and ignore).",
            ),
            Key(
                "required_features",
                Kind.STRING_LIST,
                [],
                "Semantics a reader must understand to interpret this file correctly. An "
                "entry this reader does not recognise is a hard error -- the same contract "
                "as the .h0 header (doc/h0_file_format.md), and the reason unknown keys can "
                "safely be lenient while this is strict.",
            ),
        ),
    ),
    Table(
        "run",
        "Where output goes and how much of it there is.",
        (
            Key("outdir", Kind.PATH, ".", "Directory for the output archive."),
            Key("verbosity", Kind.COUNT, 0, "0-3; the CLI's -v/-vv/-vvv overrides this.", minimum=0),
        ),
    ),
    Table(
        "units",
        "How to read the numbers in THIS file. Never describes the Hamiltonian file, which "
        "carries its own unit in its header.",
        (
            Key(
                "energy",
                Kind.ENUM,
                UNSET,
                "REQUIRED, deliberately with no default. Governs every key of kind 'energy'. "
                "The argparse CLI defaults to eV and RSPt writes Rydberg, so any default here "
                "would let two front-ends of the same code disagree by 13.6057x with nothing "
                "but a heuristic warning to catch it. A default can be added in a later "
                "version; it can never be removed.",
                choices=("eV", "Ry", "Ha"),
            ),
            Key(
                "temperature",
                Kind.ENUM,
                "K",
                "Whether [temperature].kelvin is Kelvin, or an energy in the unit above.",
                choices=("K", "energy"),
            ),
        ),
    ),
]

_TABLE_LIST += [
    Table(
        "hamiltonian.file",
        "Read the one-particle Hamiltonian from a file: a self-describing flat `.h0`, or a "
        "legacy labelled `.pickle`/`.json`/`.dat`. Which one is decided by the file's own "
        "content, not its extension (see ed.model.load_model).",
        (
            Key("path", Kind.PATH, UNSET, "The Hamiltonian file, relative to this input file."),
            Key(
                "unit",
                Kind.ENUM,
                None,
                "ERROR on a legacy format: nothing in the reader scales a pickle/.dat/.json "
                "amplitude, every shipped legacy file is already eV-scale, and anyone holding "
                "a Rydberg Hamiltonian is on .h0, which records its own unit. Convert to .h0 "
                "instead. On a .h0 this may only restate the header's unit; disagreeing is an "
                "error, never a silent override.",
                choices=("eV", "Ry", "Ha"),
            ),
            Key(
                "n_impurity_orbitals",
                Kind.COUNT,
                None,
                "Impurity block size, for the legacy bare-integer format only -- it records no "
                "orbital layout. Validated against the file's sparsity pattern.",
                minimum=1,
            ),
            Key(
                "contains_soc",
                Kind.BOOL,
                None,
                "Cross-check against a .h0 header, never an override. The header treats an "
                "absent value as *unknown*, not false, and requesting a non-zero shell `soc` "
                "against an unknown or true value is a hard error -- this exact SOC "
                "double-counting has shipped once already.",
            ),
            Key(
                "energy_reference",
                Kind.ENUM,
                None,
                "Cross-check against the header. 'absolute' is refused for any double-counting "
                "scheme, sector walk or Fermi-centred mesh: the bath valence/conduction split "
                "is taken from sign(h[o,o]) and the DFT reference filling from mu_chem = 0, so "
                "an offset zero silently re-partitions the bath into a different model.",
                choices=("fermi", "absolute"),
            ),
        ),
        variant_of="hamiltonian",
    ),
    Table(
        "hamiltonian.crystal_field",
        "Build the Hamiltonian from crystal-field parameters. ALL TEN are required: the "
        "underlying reader fills each absent key from a hard-coded Ni-in-NiO value, so the "
        "shipped CoO/FeO/MnO files (which set six) silently run with Ni's conduction bath.",
        (
            Key("e_imp", Kind.ENERGY, UNSET, "Average valence-shell on-site energy."),
            Key("e_deltaO_imp", Kind.ENERGY, UNSET, "Cubic (10Dq) splitting of the valence shell."),
            Key("e_val_eg", Kind.ENERGY, UNSET, "Valence bath level coupled to the eg orbitals."),
            Key("e_val_t2g", Kind.ENERGY, UNSET, "Valence bath level coupled to the t2g orbitals."),
            Key("e_con_eg", Kind.ENERGY, UNSET, "Conduction bath level coupled to the eg orbitals."),
            Key("e_con_t2g", Kind.ENERGY, UNSET, "Conduction bath level coupled to the t2g orbitals."),
            Key("v_val_eg", Kind.ENERGY, UNSET, "Valence hybridization with the eg orbitals."),
            Key("v_val_t2g", Kind.ENERGY, UNSET, "Valence hybridization with the t2g orbitals."),
            Key("v_con_eg", Kind.ENERGY, UNSET, "Conduction hybridization with the eg orbitals."),
            Key("v_con_t2g", Kind.ENERGY, UNSET, "Conduction hybridization with the t2g orbitals."),
            Key(
                "bath_state_basis",
                Kind.ENUM,
                "spherical",
                "Basis the bath states are expressed in. Reachable from no CLI today.",
                choices=("spherical", "cubic"),
            ),
        ),
        variant_of="hamiltonian",
    ),
    Table(
        "hamiltonian.archive",
        "Reconstruct the model from an impurityModel_data.h5 archive written by the RSPt "
        "interface. The archive supplies the model, both frequency meshes and the recorded "
        "basis/solver options, so tables it covers must not also appear in this file.",
        (
            Key("path", Kind.PATH, UNSET, "Archive file."),
            Key("cluster", Kind.STRING, None, "Cluster label; default is the first group."),
            Key("iteration", Kind.COUNT, None, "DMFT iteration; default is the last."),
        ),
        variant_of="hamiltonian",
    ),
    Table(
        "hamiltonian.blocks",
        "Build from the impurity / hybridization / bath blocks, H = [[H_imp, V^dag], [V, H_bath]].",
        (
            Key("h_imp", Kind.MATRIX, UNSET, "Effective impurity block (n_imp, n_imp)."),
            Key("v", Kind.MATRIX, UNSET, "Impurity-bath hopping (n_bath, n_imp)."),
            Key("h_bath", Kind.MATRIX, UNSET, "Bath block (n_bath, n_bath)."),
        ),
        variant_of="hamiltonian",
    ),
    Table(
        "hamiltonian.matrix",
        "Build from the full one-particle solver matrix, impurity block first.",
        (
            Key("h", Kind.MATRIX, UNSET, "Full (n, n) one-particle Hamiltonian."),
            Key("n_impurity_orbitals", Kind.COUNT, UNSET, "Leading impurity block dimension.", minimum=1),
        ),
        variant_of="hamiltonian",
    ),
]

_TABLE_LIST += [
    Table(
        "shell",
        "One correlated or core shell. An array of tables, so a shell's angular momentum is "
        "tied to ITS OWN bath count and occupation -- unlike the CLI's four order-coupled "
        "lists (--ls / --nBaths / --nValBaths / --n0imps), where only list position relates "
        "them and only equal lengths are checked.",
        (
            Key(
                "l",
                Kind.COUNT,
                UNSET,
                "Angular momentum. UNRESTRICTED by this schema: the format must be able to "
                "express any (core l, valence l) pair before the solver can execute it, or it "
                "needs replacing the day the 2p/3d restriction lifts. What the solver can "
                "actually do is checked separately -- see inputformat.capabilities.",
                minimum=0,
            ),
            Key(
                "role",
                Kind.ENUM,
                UNSET,
                "REQUIRED and never inferred from `l`. The inference 'l=1 means core, l=2 "
                "means valence' is precisely the hardcoding this format has to outlive.",
                choices=("core", "valence"),
            ),
            Key(
                "n_bath",
                Kind.COUNT,
                UNSET,
                "Total bath states for this shell.",
                deduced_from=(
                    "the .h0 header (n_orb minus the impurity block) for the shell the file "
                    "describes; 0 for every other shell, since a shell with no Hamiltonian has "
                    "no fitted bath -- the normal case for a core shell. Required for any "
                    "non-.h0 source, none of which records a bath layout."
                ),
                minimum=0,
            ),
            Key(
                "n_valence_bath",
                Kind.COUNT,
                UNSET,
                "Bath states that start occupied. Must not exceed n_bath.",
                deduced_from=(
                    "the .h0 header's valence_bath/conduction_bath lists when present; "
                    "otherwise from the bath on-site energies, h[o,o] < 0 being valence -- the "
                    "same rule solver_basis.classify_bath_occupation already applies. 0 for a "
                    "shell the file does not describe."
                ),
                minimum=0,
            ),
            Key("nominal_occupation", Kind.COUNT, UNSET, "Nominal electron count on this shell.", minimum=0),
            Key(
                "soc",
                Kind.ENERGY,
                0.0,
                "Spin-orbit coupling to add. Only added when the Hamiltonian does not already "
                "contain it; a non-zero value against a .h0 whose header says contains_soc is "
                "true, or does not say at all, is a hard error.",
            ),
            Key(
                "zeeman_splitting",
                Kind.ENERGY_VECTOR,
                None,
                "Zeeman ENERGY (hx, hy, hz) -- a spin-only splitting with no Bohr magneton, no "
                "g-factor and no orbital term, so it is not 'a magnetic field'. OMIT for the "
                "format-dependent default: no field for a flat .h0, a (0, 0, 1e-4) "
                "symmetry-breaking nudge for the labelled formats. Writing [0, 0, 0] is a "
                "third thing again -- it skips the dressing step entirely.",
            ),
        ),
        repeatable=True,
    ),
    Table(
        "interaction.slater",
        "Slater-Condon parameters. Array lengths are DERIVED from the shells' angular momenta "
        "(2*l_v+1, 2*l_c+1, 2*l_c+1, 2*l_c+2) and checked, rather than restated as l_core / "
        "l_valence keys -- one source of truth per angular momentum.",
        (
            Key("F_vv", Kind.ENERGY_LIST, UNSET, "Valence-valence F^k (was Fdd). Length 2*l_v + 1."),
            Key("F_cc", Kind.ENERGY_LIST, None, "Core-core F^k (was Fpp). Length 2*l_c + 1."),
            Key("F_cv", Kind.ENERGY_LIST, None, "Core-valence direct F^k (was Fpd). Length 2*l_c + 1."),
            Key("G_cv", Kind.ENERGY_LIST, None, "Core-valence exchange G^k (was Gpd). Length 2*l_c + 2."),
        ),
        variant_of="interaction",
    ),
    Table(
        "interaction.u4_file",
        "Read the four-index Coulomb tensor from a file. Out-of-line only: nobody hand-writes "
        "n_imp^4 numbers, and the RSPt index convention must be named at the reference site.",
        (Key("path", Kind.PATH, UNSET, "A .npy holding the rank-4 tensor in RSPt convention."),),
        variant_of="interaction",
    ),
    Table(
        "interaction.none",
        "No interaction: a non-interacting reference calculation.",
        (),
        variant_of="interaction",
    ),
]

_TABLE_LIST += [
    Table(
        "double_counting.mlft",
        "RSPt's charge-transfer correction `c`. SPECTROSCOPY ONLY, and not a double-counting "
        "matrix: it enters H with a `+` sign folded into h0 by ImpurityModel.from_shells and "
        "takes a different value per shell, whereas every scheme below produces a matrix that "
        "is SUBTRACTED. Sharing one `value` key between the two would be a sign error waiting "
        "to happen, which is why this has its own tag.",
        (Key("c", Kind.ENERGY, 1.5, "The charge-transfer correction."),),
        variant_of="double_counting",
    ),
    Table(
        "double_counting.fll",
        "Fully Localized Limit, dc = [U(N - 1/2) - (J/2)(N - 1)] I, at the DFT reference "
        "occupation. Needs U and J: derived from the Coulomb tensor when the model has one, "
        "otherwise supply them here.",
        (
            Key("u", Kind.ENERGY, None, "Average Coulomb repulsion; derived from u4 when absent."),
            Key("j", Kind.ENERGY, None, "Average exchange; derived from u4 when absent."),
        ),
        variant_of="double_counting",
    ),
    Table(
        "double_counting.amf",
        "Around Mean Field. Requires the model to carry a Coulomb tensor -- there is no "
        "explicit u/j escape hatch, so it cannot run on a spectroscopy model (u4 is None there).",
        (),
        variant_of="double_counting",
    ),
    Table(
        "double_counting.nominal",
        "FLL evaluated at the NOMINAL integer occupation rather than the DFT reference. Needs "
        "no reference filling, so it cannot saturate on a coarse bath fit -- the natural first "
        "guess, and a reference to check a converged fixed-occupation answer against.",
        (
            Key("u", Kind.ENERGY, None, "Average Coulomb repulsion; derived from u4 when absent."),
            Key("j", Kind.ENERGY, None, "Average exchange; derived from u4 when absent."),
        ),
        variant_of="double_counting",
    ),
    Table(
        "double_counting.sigma_inf",
        "The static (high-frequency) limit of the self-energy. Requires a Coulomb tensor.",
        (),
        variant_of="double_counting",
    ),
    Table(
        "double_counting.fixed_occupation",
        "Choose dc so the interacting thermal impurity occupation hits a target. Karolak's "
        "Eq. 2 -- the right criterion for METALS. Inside a gap the occupation is flat in mu, "
        "so a whole interval satisfies it and none is picked out; use fixed_gap there. Runs a "
        "search: 1-15 full collective ground-state solves.",
        (
            Key(
                "occupation",
                Kind.DIMENSIONLESS,
                None,
                "Target impurity occupation in electrons -- NOT an energy, so [units].energy "
                "does not touch it. Absent means the DFT reference filling of the raw h0.",
                minimum=0.0,
            ),
        )
        + _DC_SEARCH_KEYS,
        variant_of="double_counting",
    ),
    Table(
        "double_counting.fixed_peak",
        "Choose dc so a peak in the impurity spectral function lands at a given energy. "
        "Positive places an electron-addition peak, negative a removal peak. Runs a search.",
        (Key("peak_position", Kind.ENERGY, UNSET, "Where to put the peak, relative to E_F."),) + _DC_SEARCH_KEYS,
        variant_of="double_counting",
    ),
    Table(
        "double_counting.fixed_gap",
        "Centre dc in the charge gap (Karolak's insulator prescription): put the midpoint of "
        "the removal and addition excitations at `offset`. RECOMMENDED FOR CHARGE-TRANSFER "
        "INSULATORS, where the fixed-occupation condition breaks down. Note what is actually "
        "measured is the gap of the whole cluster, not of the impurity; the criterion reports "
        "its own exposure per edge. Runs a search.",
        (Key("offset", Kind.ENERGY, 0.0, "Where to centre the gap; 0 is the Fermi level."),) + _DC_SEARCH_KEYS,
        variant_of="double_counting",
    ),
    Table("double_counting.none", "No double counting.", (), variant_of="double_counting"),
    Table(
        "rotation_to_spherical",
        "Rotation from the impurity basis to spherical harmonics. Used for L/S/J OBSERVABLE "
        "REPORTING ONLY -- it does not rotate the Hamiltonian into a spherical representation, "
        "and the solver composes its own rotation independently. Stored per shell, so a "
        "per-shell override is a sub-table.",
        (
            Key(
                "from_h0",
                Kind.BOOL,
                True,
                "Take the rotation from the .h0 header, falling back to the identity. Set "
                "false to require an explicit per-shell matrix.",
            ),
        ),
    ),
    Table(
        "temperature",
        "The thermal occupation. Give exactly one of these: they are governed by different "
        "units, and one table carrying two unit governances is how a `tau = 0.002` under "
        "[units].energy = 'Ry' becomes a silent 13.6x temperature error.",
        (
            Key("kelvin", Kind.TEMPERATURE, None, "Temperature; Kelvin unless [units].temperature says otherwise."),
            Key("tau", Kind.ENERGY, None, "Fundamental temperature k_B*T directly, as an energy."),
        ),
    ),
]

_TABLE_LIST += [
    Table(
        "many_body_basis",
        "How the many-body determinant basis is built. Named for the determinant basis "
        "specifically: 'basis' alone means both the single-particle orbital basis (a .h0 "
        "header declares one) and this, and both appear in one input file.",
        (
            Key(
                "truncation_threshold",
                Kind.AUTO_COUNT,
                "auto",
                "Cap on determinants per basis. 'auto' derives it from available per-rank "
                "memory at the (collective) call site; 'none' disables capping. The two are "
                "NOT interchangeable even though the underlying code currently collapses both "
                "to infinity in one place.",
                choices=("auto", "none"),
            ),
            Key(
                "excitation_budget",
                Kind.AUTO_COUNT,
                "auto",
                "Maximum total bath excitations per determinant. 'auto' takes the solver's "
                "measured-lossless default; 'none' disables it. Prefer omitting to writing the "
                "number: the default is documented as the tightest MEASURED value and is "
                "expected to be re-measured, so a copy here would freeze a stale one.",
                choices=("auto", "none"),
            ),
            Key("chain_restrict", Kind.BOOL, True, "Apply chain occupation restrictions."),
            Key("spin_flip_dj", Kind.BOOL, False, "Generate spin-flipped determinants."),
            Key(
                "occ_cutoff",
                Kind.DIMENSIONLESS,
                None,
                "Occupation cutoff deciding filled/partial/empty bath classification, i.e. the "
                "variational space -- not cosmetic. Per-calculation default.",
                minimum=0.0,
            ),
            Key("slater_weight_min", Kind.DIMENSIONLESS, None, "Minimum determinant weight retained.", minimum=0.0),
            Key(
                "dN",
                Kind.COUNT,
                None,
                "Impurity occupation window (+-dN) for the excited bases. Note the sentinel "
                "means different things per driver: the spectroscopy path substitutes 2, the "
                "Green's-function path treats absent as NO window at all.",
                minimum=0,
            ),
            Key("mixed_valence", Kind.DIMENSIONLESS, None, "Mixed-valence scalar, forwarded per group."),
        ),
    ),
    Table(
        "solver",
        "Green's-function kernel and eigensolver settings.",
        (
            Key(
                "gf_method",
                Kind.ENUM,
                "lanczos",
                "Green's-function kernel.",
                choices=("lanczos", "bicgstab", "sliced", "cipsi"),
            ),
            Key(
                "reort",
                Kind.AUTO_ENUM,
                "auto",
                "Block-Lanczos reorthogonalization. 'auto' is the solver's own default, which "
                "is NOT one mode: it means NONE on the Green's-function path and PARTIAL on "
                "the eigensolver path. Writing a mode also moves the derived determinant "
                "budget, since retention switches the memory model to its worst case.",
                choices=("auto", "none", "partial", "periodic", "selective", "full"),
            ),
            Key("dense_cutoff", Kind.COUNT, 500, "Use a dense eigensolver below this matrix size.", minimum=1),
            Key("sparse_green", Kind.BOOL, True, "Use the sparse block-Lanczos Green's-function path."),
            Key(
                "auto_block_structure",
                Kind.BOOL,
                True,
                "Derive the block structure and symmetry-adapted solver basis from the "
                "hybridization-dressed impurity matrix instead of the hand-coded 2p/3d one. A "
                "solver-basis decision (it replaces the Hamiltonian operator the solve runs "
                "on), which is why it lives here and not under a spectroscopy table.",
            ),
        ),
    ),
    Table(
        "environment",
        "Runtime tuning knobs, by their registry name in impurityModel.ed.config. Free-form: "
        "every key is validated against that registry, so an unknown name gets an exact "
        "closest-match suggestion rather than a guess. Reachable from the RSPt interface too, "
        "which is why it is a table of its own rather than CLI flags.",
        (),
    ),
]

_MESH_W = {"min": -25.0, "max": 25.0, "n": 3001}
_MESH_WLOSS = {"min": -2.0, "max": 12.0, "n": 4000}

_TABLE_LIST += [
    Table(
        "spectroscopy",
        "PES / XPS / XAS / RIXS / NIXS. The meshes and the core-hole broadening live HERE, "
        "not under a technique, because the code genuinely shares them: one `delta` is both "
        "the PES/XPS/XAS lineshape and RIXS's intermediate-state broadening, and NIXS is "
        "evaluated on RIXS's energy-loss mesh. Filing either under one technique would mean "
        "switching that technique off changed another one.",
        (
            Key("w", Kind.MESH, _MESH_W, "PES / XPS / XAS evaluation mesh, relative to E_F."),
            Key("w_loss", Kind.MESH, _MESH_WLOSS, "Energy-loss mesh, shared by RIXS and NIXS."),
            Key(
                "core_hole_broadening",
                Kind.ENERGY,
                0.2,
                "HWHM above the real axis. Sets the PES/XPS/XAS lineshape AND the RIXS "
                "INTERMEDIATE-state resolvent broadening -- one number, two roles, which is "
                "why it is not named per technique.",
                minimum=0.0,
            ),
            Key("cluster", Kind.STRING, "cluster", "Label used in the output."),
            Key("output", Kind.PATH, "spectra.h5", "Output archive, relative to [run].outdir."),
        ),
        overrides={"many_body_basis": {"occ_cutoff": 1e-6, "dN": 2}, "temperature": {"kelvin": 300.0}},
    ),
    Table(
        "spectroscopy.pes",
        "Valence photoemission and inverse photoemission.",
        (Key("enabled", Kind.BOOL, True, "Compute it. Today this is unconditional and cannot be switched off."),),
    ),
    Table(
        "spectroscopy.xps",
        "Core-level photoemission.",
        (Key("enabled", Kind.BOOL, True, "Compute it. Today this is unconditional and cannot be switched off."),),
    ),
    Table(
        "spectroscopy.xas",
        "X-ray absorption. Uses the shared core_hole_broadening.",
        (Key("enabled", Kind.BOOL, True, "Compute it."),),
    ),
    Table(
        "spectroscopy.rixs",
        "Resonant inelastic x-ray scattering, on the shared w_loss mesh.",
        (
            Key(
                "enabled",
                Kind.BOOL,
                False,
                "Compute it. THE ONLY SWITCH: a non-positive broadening and an empty incoming "
                "mesh used to disable RIXS as side effects, which meant two independent "
                "switches with no stated precedence and a broadening doubling as a feature "
                "flag. Both are now validation errors instead.",
            ),
            Key("w_in", Kind.MESH, {"min": -10.0, "max": 20.0, "n": 50}, "Incoming photon energies."),
            Key(
                "final_state_broadening",
                Kind.ENERGY,
                0.05,
                "HWHM of the FINAL state. The intermediate-state half of the lineshape is the "
                "shared core_hole_broadening.",
                minimum=0.0,
            ),
        ),
    ),
    Table(
        "spectroscopy.nixs",
        "Non-resonant inelastic x-ray scattering, on the shared w_loss mesh.",
        (
            Key(
                "enabled",
                Kind.BOOL,
                False,
                "Compute it. Previously implied by supplying a radial file; now explicit, and "
                "the radial file is required when this is on.",
            ),
            Key(
                "radial_file",
                Kind.PATH,
                None,
                "Two-column radial wavefunction of the correlated orbitals. Its length unit is "
                "what makes `q` meaningful -- they are reciprocal.",
            ),
            Key("broadening", Kind.ENERGY, 0.1, "HWHM for NIXS.", minimum=0.0),
            Key(
                "q",
                Kind.VECTOR_LIST,
                None,
                "Momentum transfers, reciprocal to the radial mesh's length unit -- an inverse "
                "length, so [units].energy does not touch it. NOTE: a q exactly along z "
                "currently yields NaN in the transition operator; use a tilted q until that is "
                "fixed.",
            ),
            Key("l_final", Kind.COUNT, 2, "Angular momentum of the final orbitals (was liNIXS).", minimum=0),
            Key("l_initial", Kind.COUNT, 2, "Angular momentum of the initial orbitals (was ljNIXS).", minimum=0),
        ),
    ),
]

_TABLE_LIST += [
    Table(
        "selfenergy",
        "Impurity self-energy Sigma(w) / Sigma(i nu) and the impurity Green's function.",
        (
            Key("cluster", Kind.STRING, "cluster", "Cluster label used in the output filenames."),
            Key("output", Kind.PATH, None, "Output archive; default selfenergy-<cluster>.h5."),
        ),
        overrides={"temperature": {"tau": 0.002}, "many_body_basis": {"mixed_valence": 0}},
    ),
    Table(
        "selfenergy.real_axis",
        "Real-frequency output.",
        (
            Key("enabled", Kind.BOOL, True, "Compute it. An explicit switch, not an empty mesh."),
            Key("mesh", Kind.MESH, {"min": -10.0, "max": 10.0, "n": 2001}, "Real frequencies, relative to E_F."),
            Key("broadening", Kind.ENERGY, 0.1, "Distance above the real axis.", minimum=0.0),
        ),
    ),
    Table(
        "selfenergy.matsubara",
        "FERMIONIC Matsubara output: i*nu_n with nu_n = (2n+1)*pi*tau.",
        (
            Key("enabled", Kind.BOOL, False, "Compute it. An explicit switch, not a zero count."),
            Key("n_points", Kind.COUNT, 0, "Number of fermionic Matsubara frequencies.", minimum=0),
        ),
    ),
    Table(
        "susceptibility",
        "Dynamical impurity susceptibilities chi(w) / chi(i nu).",
        (
            Key("cluster", Kind.STRING, "cluster", "Cluster label used in the output."),
            Key("output", Kind.PATH, "chi.h5", "Output archive, relative to [run].outdir."),
            Key(
                "n_psi_max",
                Kind.COUNT,
                5,
                "Eigenstates to solve for. Configurable on THIS path only: the spectroscopy "
                "driver ignores it and the self-energy driver hardcodes its own count.",
                minimum=1,
            ),
            Key(
                "energy_cut",
                Kind.DIMENSIONLESS,
                10.0,
                "Thermal window in multiples of k_B*T -- a MULTIPLIER, not an energy, despite "
                "the name; [units].energy must not touch it.",
                minimum=0.0,
            ),
        ),
        overrides={"temperature": {"tau": 0.002}, "many_body_basis": {"mixed_valence": 0}},
    ),
    Table(
        "susceptibility.real_axis",
        "Real-frequency output.",
        (
            Key("enabled", Kind.BOOL, True, "Compute it."),
            Key("mesh", Kind.MESH, {"min": -5.0, "max": 5.0, "n": 501}, "Real frequencies."),
            Key("broadening", Kind.ENERGY, 0.01, "Distance above the real axis.", minimum=0.0),
        ),
    ),
    Table(
        "susceptibility.matsubara",
        "BOSONIC Matsubara output. Distinct from the self-energy's in both statistics and "
        "convention (this mesh is real-valued and includes nu = 0, which carries the Van "
        "Vleck term), which is why the two are separate tables rather than one shared key.",
        (
            Key("enabled", Kind.BOOL, True, "Compute it."),
            Key("n_points", Kind.COUNT, 64, "Number of bosonic Matsubara frequencies.", minimum=0),
        ),
    ),
]

#: Every declared table, keyed by dotted path.
TABLES = {t.path: t for t in _TABLE_LIST}


def variants_of(root: str) -> tuple:
    """Return the sub-table paths forming the tagged union rooted at ``root``.

    The *presence* of one of these sub-tables selects the variant. A string tag
    (``source = "file"``) cannot do this job: switching the tag leaves the previous arm's
    keys behind as known-and-ignored, which is the same silent-staleness bug the format
    exists to remove.
    """
    return tuple(sorted(path for path, table in TABLES.items() if table.variant_of == root))


def dump() -> str:
    """Render the whole schema as the Markdown reference table.

    ``doc/input_format.md`` is generated from this, mirroring how ``doc/configuration.md``
    is generated from :func:`impurityModel.ed.config.dump` -- so a new key is documented by
    declaring it, and the docs cannot drift from the reader.
    """
    lines = [
        "# Input file reference",
        "",
        "> Generated from `impurityModel/inputformat/schema.py`; edit the `Key` declarations "
        "there, not this file. Regenerate with",
        "> `python -m impurityModel.inputformat.schema > doc/input_format.md`.",
        "",
        f"Format version {SPEC_VERSION[0]}.{SPEC_VERSION[1]}.",
        "",
    ]
    for path in sorted(TABLES):
        table = TABLES[path]
        heading = f"[[{path}]]" if table.repeatable else f"[{path}]"
        lines += [f"## `{heading}`", "", table.doc, ""]
        if not table.keys:
            lines += ["*No declared keys.*", ""]
            continue
        lines += ["| Key | Kind | Default | Description |", "| --- | --- | --- | --- |"]
        for key in table.keys:
            if key.default is UNSET:
                default = "*deduced*" if key.deduced_from else "**required**"
            else:
                default = f"`{key.default!r}`"
            doc = key.doc
            if key.choices:
                doc += " Choices: " + ", ".join(f"`{c}`" for c in key.choices) + "."
            if key.deduced_from:
                doc += f" Deduced from {key.deduced_from}"
            lines.append(f"| `{key.name}` | {key.kind.value} | {default} | {doc.replace(chr(10), ' ')} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    print(dump())
