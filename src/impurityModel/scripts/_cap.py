"""The ``--truncation_threshold`` option, shared by every CLI that sizes a many-body basis."""

from argparse import ArgumentTypeError

from impurityModel.ed.memory_estimate import parse_truncation_threshold

#: argparse default telling "not given" apart from an explicit ``auto`` (which parses to ``None``):
#: under ``--from-archive`` only a cap actually passed may override the archived one.
CAP_NOT_GIVEN = object()

HELP = (
    "Determinant cap per basis: auto (default; sized from available memory, separately for the ground state "
    "and the Green's-function units, and may be held lower at run time if measured memory runs short), "
    "unlimited (or inf), or a positive integer such as 2e6 (final: never lowered, you get a warning if memory "
    "runs short). With --from-archive the archived value is used unless this flag is given."
)


def cap_argument(text):
    """argparse ``type`` for ``--truncation_threshold`` (see ``parse_truncation_threshold``)."""
    try:
        return parse_truncation_threshold(text)
    except ValueError as err:
        raise ArgumentTypeError(str(err)) from None


def add_cap_argument(parser):
    """Add ``--truncation_threshold`` (default :data:`CAP_NOT_GIVEN`) to ``parser``."""
    parser.add_argument("--truncation_threshold", type=cap_argument, default=CAP_NOT_GIVEN, help=HELP)


def requested_cap(args):
    """The cap the user asked for: ``None`` (auto) when the flag was not given."""
    return None if args.truncation_threshold is CAP_NOT_GIVEN else args.truncation_threshold
