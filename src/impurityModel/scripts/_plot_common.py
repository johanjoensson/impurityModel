"""
Shared command line options and figure handling for the impurityModel plot CLIs.

Ported from ``pyRSPthon.cli._common`` (the RSPt plotting utilities) to keep the same
conventions across both packages.
"""

import h5py
import matplotlib
import numpy as np


def add_plot_arguments(parser):
    """
    Add the options shared by all plot CLIs: input file, output file and basic figure
    styling.
    """
    parser.add_argument(
        "--filename",
        "-f",
        default="spectra.h5",
        type=str,
        help="HDF5 file written by impurityModel.ed.spectra.simulate_spectra.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        type=str,
        help="Save the figure(s) to this file instead of showing them " "(multiple figures get numbered suffixes).",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=None,
        metavar=("W", "H"),
        help="Figure size in inches.",
    )
    parser.add_argument("--dpi", type=float, default=None, help="Figure DPI.")
    parser.add_argument("--font-size", type=float, default=None, help="Base font size.")
    parser.add_argument(
        "--export",
        default=None,
        type=str,
        metavar="PREFIX",
        help="Also write the plotted (contracted) curves to PREFIX-<name>.dat text files.",
    )


def apply_plot_style(args):
    """
    Apply the shared style options. Must run before pyplot figures are made.
    """
    if args.output is not None:
        matplotlib.use("Agg")
    # Deferred: pyplot must not be imported anywhere before matplotlib.use() runs above, or
    # the backend is already locked to the default interactive one.
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if args.figsize is not None:
        plt.rcParams["figure.figsize"] = args.figsize
    if args.dpi is not None:
        plt.rcParams["figure.dpi"] = args.dpi
        plt.rcParams["savefig.dpi"] = args.dpi
    if args.font_size is not None:
        plt.rcParams["font.size"] = args.font_size


def finish_plots(args):
    """
    Show the figures, or save them to args.output. With several open figures the output
    name gets a numbered suffix per figure.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415 -- see apply_plot_style

    if args.output is None:
        plt.show()
        return
    fignums = plt.get_fignums()
    if len(fignums) == 1:
        plt.figure(fignums[0])
        plt.savefig(args.output, bbox_inches="tight")
        print(f"Wrote {args.output}")
        return
    stem, dot, ext = args.output.rpartition(".")
    if not dot:
        stem, ext = args.output, "png"
    for i, num in enumerate(fignums, start=1):
        fname = f"{stem}-{i}.{ext}"
        plt.figure(num)
        plt.savefig(fname, bbox_inches="tight")
        print(f"Wrote {fname}")


def export_curves(prefix, name, x, x_label, curves):
    """
    Write a set of curves sharing one x-axis to ``PREFIX-<name>.dat``.

    Parameters
    ----------
    prefix : str
        Filename prefix (see ``--export``).
    name : str
        Short name for this set of curves (e.g. ``"XAS"``), used in the filename.
    x : ndarray
        Shared x-axis values (length ``n``).
    x_label : str
        Column header for the x-axis.
    curves : dict
        Mapping of column label -> array of length ``n``.
    """
    labels = list(curves.keys())
    data = np.column_stack([x] + [curves[label] for label in labels])
    header = "  ".join([x_label] + labels)
    filename = f"{prefix}-{name}.dat"
    np.savetxt(filename, data, fmt="%.6e", header=header)
    print(f"Wrote {filename}")


def parse_orbital_selection(spec, norb):
    """
    Parse an orbital selection like "0,2,4", "0-4" or "0+1,3-5" into a list of index
    groups. Comma-separated entries plot separately (a range gives one entry per index);
    '+'-joined indices/ranges form one summed group. With spec None every orbital is its
    own group.
    """
    if spec is None:
        return [[i] for i in range(norb)]

    def expand(token):
        token = token.strip()
        if "-" in token and not token.startswith("-"):
            lo, hi = (int(s) for s in token.split("-", 1))
            if hi < lo:
                raise ValueError(f"empty range {token!r}")
            return list(range(lo, hi + 1))
        return [int(token)]

    picked = []
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        try:
            if "+" in part:
                group = []
                for sub in part.split("+"):
                    if sub.strip():
                        group.extend(expand(sub))
                if group:
                    picked.append(group)
            else:
                picked.extend([i] for i in expand(part))
        except ValueError:
            raise SystemExit(
                f'Malformed orbital selection {part!r} (expected e.g. "0,2,4", "0-4" or "0+1+2")'
            ) from None
    bad = sorted({i for group in picked for i in group if i < 0 or i >= norb})
    if bad:
        raise SystemExit(f"Orbital indices {bad} out of range (0..{norb - 1})")
    return picked


def load_spectra_h5(filename):
    """
    Read a ``spectra.h5`` file into a flat dict of numpy arrays.

    Only the datasets present in the file are returned; downstream plot functions check
    for the keys they need. Keys are dataset paths, e.g. ``"XAS/tensor"``.
    """
    data = {}
    with h5py.File(filename, "r") as h5f:
        for key in ("E", "w", "wIn", "wLoss", "qsNIXS", "r", "RiNIXS", "RjNIXS"):
            if key in h5f:
                data[key] = np.array(h5f[key])
        for group in ("PS", "XPS", "NIXS"):
            if group in h5f and "spectra" in h5f[group]:
                data[f"{group}/spectra"] = np.array(h5f[group]["spectra"])
        for group in ("XAS", "RIXS"):
            if group not in h5f:
                continue
            for member in ("tensor", "projected"):
                if member in h5f[group]:
                    data[f"{group}/{member}"] = np.array(h5f[group][member])
    return data
