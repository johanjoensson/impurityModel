"""
Plot script for the RIXS Kramers-Heisenberg tensor written by
:func:`impurityModel.ed.spectra.simulate_spectra`.

The tensor is contracted with the requested in/out polarization pairs at plot time (see
:mod:`impurityModel.ed.polarization`), so different polarizations, dichroism, or an
emission-energy view are all cheap post-processing -- no re-run of the solver needed.
"""

import argparse
import os.path

import numpy as np

from impurityModel.ed import polarization as pol
from impurityModel.scripts._plot_common import (
    add_plot_arguments,
    apply_plot_style,
    export_curves,
    finish_plots,
    load_spectra_h5,
)


def _load_map(filename):
    """Return ``(wIn, wLoss, C_or_map, is_tensor)``.

    ``is_tensor`` is True when the file holds the rank-4 Cartesian tensor
    (``RIXS/tensor``, contract at plot time) and False for the legacy per-operator map
    (``RIXS/projected``, already polarization-resolved).
    """
    data = load_spectra_h5(filename)
    if "wIn" not in data or "wLoss" not in data:
        raise SystemExit(f"{filename} has no RIXS data (missing wIn/wLoss).")
    if "RIXS/tensor" in data:
        return data["wIn"], data["wLoss"], data["RIXS/tensor"], True
    if "RIXS/projected" in data:
        return data["wIn"], data["wLoss"], data["RIXS/projected"], False
    raise SystemExit(f"{filename} has no RIXS/tensor or RIXS/projected dataset.")


def _resolve_maps(C_or_map, is_tensor, pols_in, pols_out):
    """Contract (if needed) into a real intensity map of shape (n_pin, n_pout, wIn, wLoss)."""
    if is_tensor:
        return pol.intensity(pol.contract_rixs_tensor(C_or_map, pols_in, pols_out))
    return pol.intensity(C_or_map)


def _plot_map(ax, fig, wIn, wLoss, intensity, cutoff, title):
    tmp = np.copy(intensity.T)
    tmp[tmp < cutoff] = np.nan
    dx = wIn[1] - wIn[0] if len(wIn) > 1 else 1.0
    dy = wLoss[1] - wLoss[0] if len(wLoss) > 1 else 1.0
    extent = (wIn[0] - dx / 2, wIn[-1] + dx / 2, wLoss[0] - dy / 2, wLoss[-1] + dy / 2)
    if not np.any(np.isfinite(tmp)):
        # Every point is below cutoff (e.g. the mesh has no weight in this window, see the
        # "NiO workload's GF has no weight in the window" note) -- a log-normed colorbar over
        # an all-NaN image raises inside matplotlib; report it instead of crashing the plot.
        print(f"warning: no intensity above cutoff={cutoff:g} in this map -- nothing to plot.")
        ax.set_xlabel(r"$\omega_{in}$   (eV)")
        ax.set_ylabel(r"$\omega_{loss}$   (eV)")
        if title:
            ax.set_title(title)
        return
    cs = ax.imshow(tmp, origin="lower", extent=extent, aspect="auto", norm="log")
    cbar = fig.colorbar(cs, ax=ax)
    cbar.ax.set_ylabel("RIXS intensity")
    ax.set_xlabel(r"$\omega_{in}$   (eV)")
    ax.set_ylabel(r"$\omega_{loss}$   (eV)")
    if title:
        ax.set_title(title)


def run(args) -> None:
    wIn, wLoss, C_or_map, is_tensor = _load_map(args.filename)
    cutoff = args.cutoff

    # wIn stays the x-axis for the 2D maps regardless of --emission; --emission only relabels
    # the --cuts line-plot x-axis to wOut = wIn - wLoss further down.
    x_label = r"$\omega_{in}$   (eV)"

    if not is_tensor and (args.pol_in or args.pol_out or args.mcd):
        print("warning: this file stores the legacy per-operator RIXS map; --pol-in/--pol-out/--mcd are ignored.")

    if args.mcd:
        if not is_tensor:
            raise SystemExit("--mcd requires a RIXS/tensor dataset (legacy per-operator maps can't be re-polarized).")
        i_cl = pol.intensity(pol.contract_rixs_tensor(C_or_map, ["cl"], args.pol_out or ["x", "y", "z"]))
        i_cr = pol.intensity(pol.contract_rixs_tensor(C_or_map, ["cr"], args.pol_out or ["x", "y", "z"]))
        mcd_map = np.sum(i_cl - i_cr, axis=(0, 1))  # (wIn, wLoss)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        tmp = mcd_map.T
        vmax = np.max(np.abs(tmp))
        cs = ax.imshow(
            tmp,
            origin="lower",
            extent=(wIn[0], wIn[-1], wLoss[0], wLoss[-1]),
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        cbar = fig.colorbar(cs, ax=ax)
        cbar.ax.set_ylabel(r"$I_{cl} - I_{cr}$")
        ax.set_xlabel(x_label)
        ax.set_ylabel(r"$\omega_{loss}$   (eV)")
        ax.set_title("RIXS-MCD")
        plt.tight_layout()

    pols_in = args.pol_in if args.pol_in else ["x", "y", "z"]
    pols_out = args.pol_out if args.pol_out else ["x", "y", "z"]
    maps = _resolve_maps(C_or_map, is_tensor, pols_in, pols_out)  # (n_pin, n_pout, wIn, wLoss)
    n_pin, n_pout = maps.shape[:2]

    import matplotlib.pyplot as plt

    summed = np.sum(maps, axis=(0, 1))
    fig, ax = plt.subplots()
    _plot_map(ax, fig, wIn, wLoss, summed, cutoff, None)
    plt.tight_layout()

    if args.export is not None:
        export_curves(args.export, "RIXS-wLoss-sum", wIn, "wIn", {"RIXS_sum": np.sum(summed, axis=1)})

    # The per-pair grid needs polarization labels that actually match `maps`' axes, which is
    # only guaranteed when we did the contracting (is_tensor); a legacy per-operator map's
    # stored axes are whatever the run requested, unrelated to --pol-in/--pol-out here (and
    # already warned about above).
    if is_tensor and (args.pol_in or args.pol_out):
        fig, axes = plt.subplots(nrows=n_pin, ncols=n_pout, sharex=True, sharey=True, squeeze=False)
        for i in range(n_pin):
            for j in range(n_pout):
                _plot_map(axes[i, j], fig, wIn, wLoss, maps[i, j], cutoff, f"in={pols_in[i]}, out={pols_out[j]}")
        plt.tight_layout()

    if args.fy:
        dwLoss = wLoss[1] - wLoss[0]
        fy = dwLoss * np.sum(summed, axis=1)
        mask = wLoss < 0.2
        fy_qe = dwLoss * np.sum(summed[:, mask], axis=1)
        plt.figure()
        plt.plot(wIn, fy, "-r", label="FY")
        plt.plot(wIn, fy_qe, "-b", label="quasi-elastic FY")
        plt.legend()
        plt.xlabel(x_label)
        plt.ylabel("Intensity")
        plt.title("Fluorescence yield")
        plt.tight_layout()
        if args.export is not None:
            export_curves(args.export, "RIXS-FY", wIn, "wIn", {"FY": fy, "FY_quasi_elastic": fy_qe})

    if args.cuts:
        plt.figure()
        for e in args.cuts:
            i = int(np.argmin(np.abs(wIn - e)))
            if args.emission:
                x = wIn[i] - wLoss
                xlabel = r"$\omega_{out}$   (eV)"
            else:
                x = wLoss
                xlabel = r"$\omega_{loss}$   (eV)"
            plt.plot(x, summed[i, :], label=rf"$\omega_{{in}}$={wIn[i]:.3g}")
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel("Intensity")
        plt.title("Energy-loss cuts")
        plt.tight_layout()

    finish_plots(args)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="plot_RIXS",
        description="Plot RIXS spectra calculated by impurityModel (spectra.h5).",
    )
    add_plot_arguments(parser)
    parser.add_argument(
        "--cutoff",
        type=float,
        default=1e-6,
        help="Intensities below this value are masked (NaN) in the 2D maps.",
    )
    parser.add_argument(
        "--pol-in",
        nargs="+",
        default=None,
        metavar="POL",
        help='In-going polarizations to contract with (named "x"/"y"/"z"/"cl"/"cr" or '
        "comma-separated components). Given together with --pol-out, plots a grid of "
        "per-pair maps in addition to the polarization-summed map. Default: x, y, z.",
    )
    parser.add_argument(
        "--pol-out",
        nargs="+",
        default=None,
        metavar="POL",
        help="Out-going polarizations to contract with. See --pol-in.",
    )
    parser.add_argument(
        "--mcd",
        action="store_true",
        help="Also plot the RIXS circular-dichroism map (incoming cl - cr, summed over --pol-out).",
    )
    parser.add_argument("--fy", action="store_true", help="Also plot the fluorescence-yield curve(s) vs wIn.")
    parser.add_argument(
        "--emission",
        action="store_true",
        help="Label --cuts against the emitted photon energy wOut = wIn - wLoss instead of the energy loss.",
    )
    parser.add_argument(
        "--cuts",
        nargs="+",
        type=float,
        default=None,
        metavar="E",
        help="Plot energy-loss line cuts at the wIn value(s) nearest to each E.",
    )
    args = parser.parse_args()
    apply_plot_style(args)
    if not os.path.isfile(args.filename):
        raise SystemExit("Data file does not exist: " + args.filename)
    run(args)


if __name__ == "__main__":
    main()
