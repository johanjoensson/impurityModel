"""
Plot script for PS/XPS/NIXS/XAS(/RIXS fluorescence-yield) spectra written by
:func:`impurityModel.ed.spectra.simulate_spectra`.

XAS (and RIXS, for the fluorescence-yield overlay) are stored as polarization *tensors*;
this script contracts them with the requested polarizations at plot time (see
:mod:`impurityModel.ed.polarization`), so no re-run of the solver is needed to look at a
different polarization, dichroism, or an isotropic average.
"""

import argparse
import os.path
from math import pi

import numpy as np

from impurityModel.ed import polarization as pol
from impurityModel.scripts._plot_common import (
    add_plot_arguments,
    apply_plot_style,
    export_curves,
    finish_plots,
    load_spectra_h5,
    parse_orbital_selection,
)


def _grouped_curves(spectra_arr, orbital_spec):
    """Intensity ``-Im`` of ``spectra_arr`` (n_w, n_orb), grouped/summed per --orbitals."""
    intensity = pol.intensity(spectra_arr)  # (n_w, n_orb)
    n_orb = intensity.shape[1]
    groups = parse_orbital_selection(orbital_spec, n_orb)
    curves = {}
    for group in groups:
        label = "+".join(str(i) for i in group)
        curves[label] = np.sum(intensity[:, group], axis=1)
    return curves


def _plot_orbital_spectrum(w, spectra_arr, title, xlabel, orbital_spec, export_prefix, export_name):
    import matplotlib.pyplot as plt

    curves = _grouped_curves(spectra_arr, orbital_spec)
    total = np.sum(pol.intensity(spectra_arr), axis=1)

    plt.figure()
    plt.plot(w, total, "-k", label="total")
    if len(curves) > 1:
        for label, y in curves.items():
            plt.plot(w, y, label=label)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("Intensity")
    plt.title(title)
    plt.tight_layout()

    if export_prefix is not None:
        export_curves(export_prefix, export_name, w, xlabel, {"total": total, **curves})


def _plot_nixs(wLoss, nixs, qs, export_prefix):
    import matplotlib.pyplot as plt

    intensity = pol.intensity(nixs)  # (n_w, n_q)
    if qs is not None:
        labels = ["|q|={:3.1f}".format(np.linalg.norm(q)) + r" A$^{-1}$" for q in qs]
    else:
        labels = [str(i) for i in range(intensity.shape[1])]

    plt.figure()
    curves = {}
    for i, label in enumerate(labels):
        plt.plot(wLoss, intensity[:, i], label=label)
        curves[label] = intensity[:, i]
    plt.legend()
    plt.xlabel(r"$\omega_{loss}$   (eV)")
    plt.ylabel("Intensity")
    plt.title("NIXS")
    plt.tight_layout()

    if export_prefix is not None:
        export_curves(export_prefix, "NIXS", wLoss, "wLoss", curves)


def _plot_xas_tensor(w, chi, args):
    import matplotlib.pyplot as plt

    pols = args.pol if args.pol else ["x", "y", "z"]
    contracted = pol.intensity(pol.contract_spectra_tensor(chi, pols))  # (n_w, n_pol)
    iso = pol.intensity(pol.isotropic(chi))

    plt.figure()
    for p, label in enumerate(pols):
        plt.plot(w, contracted[:, p], label=str(label))
    plt.plot(w, iso, "--k", label="isotropic")
    plt.legend()
    plt.xlabel(r"$\omega$   (eV)")
    plt.ylabel("Intensity")
    plt.title("XAS")
    plt.tight_layout()

    export_curves_dict = {str(p): contracted[:, k] for k, p in enumerate(pols)}
    export_curves_dict["isotropic"] = iso

    if args.xmcd:
        xmcd = pol.circular_dichroism(chi)
        plt.figure()
        plt.plot(w, xmcd, "-k")
        plt.axhline(0, color="gray", lw=0.5)
        plt.xlabel(r"$\omega$   (eV)")
        plt.ylabel(r"$I_{cl} - I_{cr}$")
        plt.title("XMCD")
        plt.tight_layout()
        export_curves_dict["xmcd"] = xmcd

    if args.xld:
        xld = pol.linear_dichroism(chi)
        plt.figure()
        plt.plot(w, xld, "-k")
        plt.axhline(0, color="gray", lw=0.5)
        plt.xlabel(r"$\omega$   (eV)")
        plt.ylabel(r"$I_z - I_x$")
        plt.title("XLD")
        plt.tight_layout()
        export_curves_dict["xld"] = xld

    if args.export is not None:
        export_curves(args.export, "XAS", w, "w", export_curves_dict)

    if args.tensor_components:
        m = chi.shape[1]
        fig, axes = plt.subplots(nrows=2, ncols=m, sharex=True, squeeze=False)
        comp_labels = ["x", "y", "z"][:m] if m <= 3 else [str(i) for i in range(m)]
        for a in range(m):
            axes[0, a].plot(w, -np.imag(chi[:, a, a]), "-k")
            axes[0, a].set_title(rf"$-\mathrm{{Im}}\,\chi_{{{comp_labels[a]}{comp_labels[a]}}}$")
            axes[1, a].plot(w, np.real(chi[:, a, a]), "-k")
            axes[1, a].set_title(rf"$\mathrm{{Re}}\,\chi_{{{comp_labels[a]}{comp_labels[a]}}}$")
            axes[1, a].set_xlabel(r"$\omega$   (eV)")
        fig.suptitle("XAS spectral tensor (diagonal)")
        plt.tight_layout()


def _plot_xas_projected(w, gs, export_prefix):
    _plot_orbital_spectrum(w, gs, "XAS (projected)", r"$\omega$   (eV)", None, export_prefix, "XAS")


def _fluorescence_yield(wIn, wLoss, rixs_intensity, quasi_elastic_cutoff=0.2):
    """Fluorescence-yield and quasi-elastic-FY curves vs wIn from a real RIXS intensity map
    of shape (..., n_wIn, n_wLoss) (leading axes summed over)."""
    axes = tuple(range(rixs_intensity.ndim - 2)) + (rixs_intensity.ndim - 1,)
    scale = 1.0 / (pi * np.prod([rixs_intensity.shape[a] for a in range(rixs_intensity.ndim - 2)]))
    dwLoss = wLoss[1] - wLoss[0]
    fy = dwLoss * np.sum(rixs_intensity, axis=axes) * scale
    mask = wLoss < quasi_elastic_cutoff
    fy_qe = dwLoss * np.sum(rixs_intensity[..., mask], axis=axes) * scale
    return fy, fy_qe


def plot_spectra_in_file(filename, args=None):
    """
    Plot the spectra/tensors stored in ``filename`` (an
    :func:`impurityModel.ed.spectra.simulate_spectra` output).

    Parameters
    ----------
    filename : str
    args : argparse.Namespace, optional
        Parsed CLI arguments (polarizations, orbital grouping, export prefix, ...).
        A default namespace is used when omitted (e.g. when called from a notebook).
    """
    if args is None:
        args = argparse.Namespace(
            pol=None, xmcd=False, xld=False, tensor_components=False, orbitals=None, export=None, output=None
        )

    print("Read data from file: ", filename)
    data = load_spectra_h5(filename)
    print("data-sets:", sorted(data.keys()))

    if "PS/spectra" in data:
        print("Photo-emission spectroscopy (PS) spectrum")
        _plot_orbital_spectrum(
            data["w"], data["PS/spectra"], "PS", r"$\omega$   (eV)", args.orbitals, args.export, "PS"
        )

    if "XPS/spectra" in data:
        print("X-ray photo-emission spectroscopy (XPS) spectrum")
        _plot_orbital_spectrum(
            data["w"], data["XPS/spectra"], "XPS", r"$\omega$   (eV)", args.orbitals, args.export, "XPS"
        )

    if "NIXS/spectra" in data:
        print("NIXS spectrum")
        _plot_nixs(data["wLoss"], data["NIXS/spectra"], data.get("qsNIXS"), args.export)

    if "XAS/tensor" in data:
        print("XAS spectrum (tensor)")
        _plot_xas_tensor(data["w"], data["XAS/tensor"], args)
    elif "XAS/projected" in data:
        print("XAS spectrum (projected)")
        _plot_xas_projected(data["w"], data["XAS/projected"], args.export)

    rixs_key = "RIXS/tensor" if "RIXS/tensor" in data else "RIXS/projected" if "RIXS/projected" in data else None
    if rixs_key is not None and ("XAS/tensor" in data or "XAS/projected" in data):
        import matplotlib.pyplot as plt

        wIn, wLoss = data["wIn"], data["wLoss"]
        if rixs_key == "RIXS/tensor":
            pols = args.pol if args.pol else ["x", "y", "z"]
            rixs = pol.intensity(pol.contract_rixs_tensor(data[rixs_key], pols, pols))
            rixs = np.sum(rixs, axis=(0, 1))  # sum requested (in, out) diagonal pairs -> (wIn, wLoss)
        else:
            rixs = np.sum(pol.intensity(data[rixs_key]), axis=(0, 1))
        fy, fy_qe = _fluorescence_yield(wIn, wLoss, rixs)

        w = data["w"]
        xas_key = "XAS/tensor" if "XAS/tensor" in data else "XAS/projected"
        if xas_key == "XAS/tensor":
            xas_total = np.sum(pol.intensity(pol.contract_spectra_tensor(data[xas_key], ["x", "y", "z"])), axis=1)
        else:
            xas_total = np.sum(pol.intensity(data[xas_key]), axis=1)

        plt.figure()
        plt.plot(w, xas_total, "-k", label="XAS")
        plt.plot(wIn, fy, "-r", label="FY")
        plt.plot(wIn, fy_qe, "-b", label="quasi-elastic FY")
        plt.legend()
        plt.xlabel(r"$\omega_{in}$   (eV)")
        plt.ylabel("Intensity")
        plt.title("XAS vs fluorescence yield")
        plt.tight_layout()

        if args.export is not None:
            export_curves(args.export, "FY", wIn, "wIn", {"FY": fy, "FY_quasi_elastic": fy_qe})

    finish_plots(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PS/XPS/NIXS/XAS spectra from a spectra.h5 file.")
    add_plot_arguments(parser)
    parser.add_argument(
        "--pol",
        nargs="+",
        default=None,
        metavar="POL",
        help='XAS/RIXS polarizations to contract with (named "x"/"y"/"z"/"cl"/"cr" or '
        'comma-separated components, e.g. "1,1j,0"). Default: x, y, z.',
    )
    parser.add_argument("--xmcd", action="store_true", help="Also plot the XAS circular dichroism (I_cl - I_cr).")
    parser.add_argument("--xld", action="store_true", help="Also plot the XAS linear dichroism (I_z - I_x).")
    parser.add_argument(
        "--tensor-components",
        action="store_true",
        help="Also plot a grid of the XAS spectral tensor's diagonal components (Re and -Im).",
    )
    parser.add_argument(
        "--orbitals",
        default=None,
        metavar="SPEC",
        help='Group PS/XPS spin-orbitals for plotting, e.g. "0-4,5+6" (default: one curve per orbital).',
    )
    args = parser.parse_args()
    apply_plot_style(args)
    if not os.path.isfile(args.filename):
        raise SystemExit("Data file does not exist: " + args.filename)
    plot_spectra_in_file(args.filename, args)


if __name__ == "__main__":
    main()
