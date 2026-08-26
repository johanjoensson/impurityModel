"""Umbrella command-line interface: ``impurityModel <subcommand> ...``.

``run`` drives any calculation from a TOML input file and is the intended entry point;
``init`` and ``schema`` make that format discoverable without reading the source. The
per-calculation sub-commands (``spectra``, ``selfenergy``, ``susceptibility``) remain as the
argparse interface they always were, and the plot sub-commands (``plot-spectra``,
``plot-rixs``) are delegated to the existing plot mains. Also reachable as
``python -m impurityModel``.

MPI note: argument parsing runs identically on every rank (no rank-gated collectives here);
the sub-command ``run`` functions own all MPI work, keeping collectives unconditional.
"""

import argparse
import sys

from impurityModel.scripts import run_cmd
from impurityModel.scripts import selfenergy as selfenergy_cmd
from impurityModel.scripts import spectra as spectra_cmd
from impurityModel.scripts import susceptibility as susceptibility_cmd

# name -> (add_arguments, run, one-line help)
_SUBCOMMANDS = {
    "run": (run_cmd.add_arguments, run_cmd.run, "Run any calculation from a TOML input file."),
    "init": (run_cmd.add_init_arguments, run_cmd.init, "Print a commented starter input file."),
    "schema": (run_cmd.add_schema_arguments, run_cmd.show_schema, "Print the input-file key reference."),
    "spectra": (spectra_cmd.add_arguments, spectra_cmd.run, "Calculate PS/XPS/NIXS/XAS/RIXS spectra."),
    "selfenergy": (selfenergy_cmd.add_arguments, selfenergy_cmd.run, "Calculate the impurity self-energy."),
    "susceptibility": (
        susceptibility_cmd.add_arguments,
        susceptibility_cmd.run,
        "Calculate dynamical impurity susceptibilities.",
    ),
}


def _plot_delegates():
    """Lazily import the plot mains (they pull in matplotlib); returns ``name -> main``."""
    from impurityModel.scripts import plot_RIXS, plot_spectra

    return {"plot-spectra": plot_spectra.main, "plot-rixs": plot_RIXS.main}


def main(argv=None):
    """Parse ``argv`` (default ``sys.argv[1:]``) and dispatch to the selected sub-command."""
    argv = list(sys.argv[1:] if argv is None else argv)

    # The plot sub-commands run their own argparse; intercept them before our parser so their
    # positional/option handling is untouched (and `--help` reaches the plot parser).
    if argv and argv[0] in ("plot-spectra", "plot-rixs"):
        name = argv[0]
        sys.argv = [f"impurityModel {name}", *argv[1:]]
        return _plot_delegates()[name]()

    parser = argparse.ArgumentParser(
        prog="impurityModel", description="impurityModel exact-diagonalization solver CLI."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<subcommand>")
    for name, (add_arguments, run, help_text) in _SUBCOMMANDS.items():
        sub = subparsers.add_parser(name, help=help_text, description=help_text)
        add_arguments(sub)
        sub.set_defaults(_run=run)
    # Register the plot sub-commands for help/listing only; they are dispatched above.
    subparsers.add_parser("plot-spectra", help="Plot spectra from a spectra.h5 file.", add_help=False)
    subparsers.add_parser("plot-rixs", help="Plot a RIXS map from a spectra.h5 file.", add_help=False)

    args = parser.parse_args(argv)
    return args._run(args)


if __name__ == "__main__":
    main()
