from collections.abc import Iterable
from rspt2spectra.readfile import parse_matrices
from rspt2spectra.h2imp import write_to_file
from rspt2spectra.hyb_fit import fit_hyb
from rspt2spectra.energies import get_mu
from rspt2spectra.weight_functions import weight_functions
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from pyRSPthon.read import extract_dat
from impurityModel.ed.finite import matrixToIOp
from impurityModel.ed.greens_function import (
    block_diagonalize_hyb,
)
from impurityModel.ed.block_structure import build_block_structure, get_blocks, build_matrix
from impurityModel.ed.edchain import (
    build_H_bath_v,
    build_full_bath,
    build_imp_bath_blocks,
)
from impurityModel.ed.utils import matrix_print
import pickle

from mpi4py import MPI


def partition_index(l: Iterable, pred=bool):
    yes, no = [], []

    for idx, item in enumerate(l):
        if pred(item):
            yes.append(idx)
        else:
            no.append(idx)
    return yes, no


def filter_and_shift(ebs, vs, w_min, w_max, block_structure):
    filtered_ebs_star, filtered_vs_star = ([], [])
    shifts = [
        np.zeros((len(block_structure.blocks[i_block]), len(block_structure.blocks[i_block])), dtype=complex)
        for i_block in block_structure.inequivalent_blocks
    ]
    for ebs, vs, shift in zip(ebs, vs, shifts):
        f = np.logical_or(ebs < w_min, ebs > w_max)
        shift += np.sum(np.conj(np.transpose(vs[f], (0, 2, 1))) @ vs[f] / ebs[f, None, None], axis=0)
        filtered_ebs_star.append(ebs[np.logical_not(f)].copy())
        filtered_vs_star.append(vs[np.logical_not(f)].copy())
    return build_matrix(shifts, block_structure), filtered_ebs_star, filtered_vs_star


def run(
    cluster: str,
    bath_geometry: str,
    eim: float,
    bath_states_per_orbital: int,
    gamma: float,
    fit_unocc: bool,
    fit_imag: bool,
    prefix: str,
    verbose: bool,
    plot: bool,
    *kwargs,
):
    comm = MPI.COMM_WORLD

    if comm is not None and comm.rank != 0:
        verbose = False

    hyb_dat = extract_dat("hyb", cluster, prefix)
    hs = parse_matrices(out_file="out", search_phrase="Local hamiltonian", prefix=prefix)
    ps = parse_matrices(out_file="out", search_phrase="Projection matrix", prefix=prefix)
    qs = parse_matrices(
        out_file="out",
        search_phrase="Transformation to the local cf basis:",
        prefix=prefix,
    )
    if cluster not in hs:
        raise RuntimeError(f"Could not extract local hamiltonian for cluster {cluster} from file {prefix}/out.")
    H_dft = hs[cluster]
    mu = get_mu()
    hyb = hyb_dat.orbitals
    w = hyb_dat.w

    # If transformations to the CF basis were found, use them
    T = np.eye(H_dft.shape[0], dtype=complex)
    if cluster in qs:
        T = qs[cluster]
    hyb_cf = np.conjugate(T.T)[None] @ hyb @ T[None]
    H_dft = np.conjugate(T.T) @ H_dft @ T

    phase_hyb, Q = block_diagonalize_hyb(hyb_cf)

    block_structure = build_block_structure(phase_hyb, tol=1e-15)

    ebs_star, vs_star = fit_hyb(
        w,
        eim,
        phase_hyb,
        bath_states_per_orbital,
        block_structure,
        gamma,
        (w[0], 0) if not fit_unocc else None,
        verbose,
        comm,
        regularization="L2",
        weight_fun=weight_functions["sqrtgauss"](0, 2.0),
    )
    for ebss, vss in zip(ebs_star, vs_star):
        if len(ebss) == 0:
            continue
        sorted_indices = np.argsort(ebss, kind="stable")
        ebss[:] = ebss[sorted_indices]
        vss[:] = vss[sorted_indices]
    if verbose:
        print("Star bath energies and hopping parameters:")
        for bi, (eb, vb) in enumerate(zip(ebs_star, vs_star)):
            for eb_i, vb_i in zip(eb, vb):
                matrix_print(vb_i, f"Energy {eb_i: 9.6f} :")
            print()
        print("=" * 80)

    w_min = w[0]
    w_max = w[-1]
    if not fit_unocc:
        w_max = 0
    original_ebs_star = ebs_star
    original_vs_star = vs_star
    H_shift, ebs_star, vs_star = filter_and_shift(ebs_star, vs_star, w_min, w_max, block_structure)
    if verbose:
        matrix_print(H_shift, r"Shift of $\Delta(\omega=0)$")

    original_H_baths, original_vs_star = build_H_bath_v(
        np.conj(Q.T) @ H_dft @ Q - H_shift,
        original_ebs_star,
        original_vs_star,
        bath_geometry.lower(),
        block_structure,
        verbose,
        False,
    )
    original_H_bath, original_v = build_full_bath(original_H_baths, original_vs_star, block_structure)
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, original_H_bath, op=MPI.SUM)
        original_H_bath /= comm.size
        comm.Allreduce(MPI.IN_PLACE, original_v, op=MPI.SUM)
        original_v /= comm.size
    H_baths, vs = build_H_bath_v(
        np.conj(Q.T) @ H_dft @ Q - H_shift,
        ebs_star,
        vs_star,
        bath_geometry.lower(),
        block_structure,
        verbose,
        False,
    )
    H_bath, v = build_full_bath(H_baths, vs, block_structure)
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, H_bath, op=MPI.SUM)
        H_bath /= comm.size
        comm.Allreduce(MPI.IN_PLACE, v, op=MPI.SUM)
        v /= comm.size

    if plot:
        from itertools import product
        import matplotlib.pyplot as plt

        wn = w + 1j * eim
        I = np.eye(original_H_bath.shape[0])
        original_hyb = (
            Q[None]
            @ np.conj(original_v.T)[None, ...]
            @ np.linalg.solve(I[None, ...] * wn[:, None, None] - original_H_bath[None, ...], original_v[None, ...])
            @ np.conj(Q.T)[None]
        )
        I = np.eye(H_bath.shape[0])
        hyb = (
            Q[None]
            @ np.conj(v.T)[None, ...]
            @ np.linalg.solve(I[None, ...] * wn[:, None, None] - H_bath[None, ...], v[None, ...])
            @ np.conj(Q.T)[None]
        )
        phase_hyb = Q[None] @ phase_hyb @ np.conj(Q.T)[None]
        blocks = get_blocks(hyb, tol=0)
        # blocks = [list(range(10))]
        for block in blocks:
            fig, ax = plt.subplots(nrows=len(block), ncols=len(block), squeeze=False, sharex="all", sharey="all")
            for (i, orb_i), (j, orb_j) in product(enumerate(block), repeat=2):
                ax[i, j].fill_between(
                    wn.real,
                    phase_hyb[:, orb_i, orb_j].real,
                    0,
                    alpha=0.3,
                    color="tab:blue",
                )
                ax[i, j].plot(
                    wn.real,
                    original_hyb[:, orb_i, orb_j].real,
                    color="tab:orange",
                    linestyle="--",
                    alpha=0.5,
                    label="Full fit",
                )
                ax[i, j].axhline(
                    H_shift[orb_i, orb_j].real,
                    color="black",
                    linestyle="--",
                    alpha=0.5,
                    label=r"$\Delta(\omega=0)$ shift",
                )
                ax[i, j].plot(wn.real, hyb[:, orb_i, orb_j].real, color="tab:blue", label="Resulting fit")
            ax[0, 0].set_ylim(bottom=np.min(phase_hyb.real), top=np.max(phase_hyb.real))
            fig.suptitle(r"Re$\left\{\Delta_{fit}(\omega)\right\}$")
            fig, ax = plt.subplots(nrows=len(block), ncols=len(block), squeeze=False, sharex="all", sharey="all")
            for (i, orb_i), (j, orb_j) in product(enumerate(block), repeat=2):
                ax[i, j].fill_between(
                    wn.real,
                    phase_hyb[:, orb_i, orb_j].imag,
                    0,
                    alpha=0.3,
                    color="tab:blue",
                )
                ax[i, j].plot(
                    wn.real,
                    original_hyb[:, orb_i, orb_j].imag,
                    color="tab:orange",
                    linestyle="--",
                    alpha=0.8,
                    label="Full fit",
                )
                ax[i, j].axhline(
                    H_shift[orb_i, orb_j].imag,
                    color="black",
                    linestyle="--",
                    alpha=0.5,
                    label=r"$\Delta(\omega=0)$ shift",
                )
                ax[i, j].plot(wn.real, hyb[:, orb_i, orb_j].imag, color="tab:blue", label="Resulting fit")
            ax[0, 0].set_ylim(bottom=np.min(phase_hyb.imag), top=np.max(phase_hyb.imag))
            fig.suptitle(r"Im$\left\{\Delta_{fit}(\omega)\right\}$")
        plt.show()

    occupied_indices, positive_indices = partition_index(np.diag(H_bath), pred=lambda x: x < 0)
    sorted_bath_indices = np.array(occupied_indices + positive_indices, dtype=int)
    H_bath = H_bath[np.ix_(sorted_bath_indices, sorted_bath_indices)]
    v = v[sorted_bath_indices, :]
    n_orb = H_dft.shape[0]
    H = np.zeros((n_orb + H_bath.shape[0], n_orb + H_bath.shape[0]), dtype=complex)

    # Transform from block diagonal -> CF -> correlated basis
    H[:n_orb, :n_orb] = T @ (H_dft - Q @ H_shift @ np.conj(Q.T)) @ np.conj(T.T)
    H[n_orb:, n_orb:] = H_bath
    H[n_orb:, :n_orb] = v @ np.conj(Q.T) @ np.conj(T.T)
    H[:n_orb, n_orb:] = np.conj(H[n_orb:, :n_orb].T)

    impurity_indices, valence_bath_indices, conduction_bath_indices, block_structure = build_imp_bath_blocks(H, n_orb)

    if verbose:
        print(f"eigvals(H) :\n{np.linalg.eigvalsh(H)}")

    h_op = matrixToIOp(H)
    write_to_file(h_op, f"{cluster}_h0_op", save_as_dict=True)


def main():
    parser = ArgumentParser(
        prog="build_h0",
        description="Create local hamiltonians by reading RSPt out files and fitting hybridization functions.",
    )
    parser.add_argument("cluster", type=str)
    parser.add_argument("bath_states_per_orbital", type=int)
    parser.add_argument("-bg", "--bath-geometry", type=str, default="Star")
    parser.add_argument("--eim", type=float, default=0.010)
    parser.add_argument("--gamma", type=float, default=0.100)
    parser.add_argument("--fit-unocc", action="store_true", dest="fit_unocc")
    parser.add_argument("-i", "--imag-only", action="store_true", dest="fit_imag")
    parser.add_argument("-d", "--directory", type=str, default=".", dest="prefix")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--plot", action="store_true")
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
