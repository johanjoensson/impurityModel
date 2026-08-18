"""Shared ``-v``/``-vv``/``-vvv`` verbosity flag for ``spectra``/``selfenergy``/``susceptibility``.

Verbosity is rank-uniform: argparse runs identically on every rank, so ``args.verbose`` (and
the ``resolve_verbosity`` result below) is the same integer everywhere -- never zero it out on
non-root ranks. Drivers gate *printing* on rank instead (see :class:`impurityModel.ed.utils.Reporter`).
"""

from impurityModel.ed.utils import V_DEBUG


def add_verbosity_argument(parser):
    """Register the ``-v``/``-vv``/``-vvv`` count flag on ``parser``."""
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase output detail (-v stage summaries, -vv per-restart/round, -vvv per-iteration).",
    )


def resolve_verbosity(args) -> int:
    """Clamp ``args.verbose`` to the ``V_RESULT``..``V_DEBUG`` range."""
    return min(args.verbose, V_DEBUG)
