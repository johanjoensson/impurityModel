from collections.abc import Iterable
from rspt2spectra.readfile import parse_matrices
from rspt2spectra.h2imp import write_to_file
from rspt2spectra.hyb_fit import fit_hyb
from rspt2spectra.energies import get_mu
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from rspt_utils.read import extract_dat
from impurityModel.ed.finite import matrixToIOp
from impurityModel.ed.greens_function import (
    block_diagonalize_hyb,
)
from impurityModel.ed.block_structure import (
    build_block_structure, get_blocks
)
from impurityModel.ed.edchain import (
    build_H_bath_v,
    build_full_bath,
    build_imp_bath_blocks,
)
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


def run(
    cluster: str,
    bath_geometry: str,
    eim: float,
    bath_states_per_orbital: int,
    gamma: float,
    fit_unocc: bool,
    fit_real: bool,
    prefix: str,
    verbose: bool,
    plot: bool,
    *kwargs,
):
    comm = MPI.COMM_WORLD
    
    if comm is not None and comm.rank != 0:
        verbose = False

    hyb_dat = extract_dat("hyb", cluster, prefix)
    hs = parse_matrices(
        out_file="out", search_phrase="Local hamiltonian", prefix=prefix
    )
    ps = parse_matrices(
        out_file="out", search_phrase="Projection matrix", prefix=prefix
    )
    qs = parse_matrices(
        out_file="out",
        search_phrase="Transformation to the local cf basis:",
        prefix=prefix,
    )
    if cluster not in hs:
        raise RuntimeError(
            f"Could not extract local hamiltonian for cluster {cluster} from file {prefix}/out."
        )
    H_dft = hs[cluster]
    mu = get_mu()
    hyb = hyb_dat.orbitals
    w = hyb_dat.w

    if cluster in qs:
        T = qs[cluster]
        hyb_cf = np.conjugate(T.T)[None, :, :] @ hyb @ T[None, :, :]
        H_dft = np.conjugate(T.T) @ H_dft @ T
    else:
        hyb_cf = hyb
    phase_hyb, Q = block_diagonalize_hyb(hyb_cf)

    block_structure = build_block_structure(phase_hyb, tol=1e-6)

    ebs_star, vs_star = fit_hyb(
        w,
        eim,
        phase_hyb,
        bath_states_per_orbital,
        block_structure,
        gamma,
        not fit_real,
        (w[0], 0) if fit_unocc else None,
        verbose,
        comm,
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
            print(
                f"Energy   :  Hopping  (impurity orbitals {block_structure.blocks[block_structure.inequivalent_blocks[bi]]})"
            )
            for eb_i, vb_i in zip(eb, vb):
                print(f"{eb_i: 9.6f}:  ", "  ".join(f"{val: 9.6f}" for val in vb_i))
            print("")
        print("=" * 80)

    print(f"{bath_geometry=}")
    H_baths, vs = build_H_bath_v(
        H_dft,
        ebs_star,
        vs_star,
        bath_geometry.lower(),
        block_structure,
        verbose,
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
        beta = 10.99
        iwn = np.arange(-128, 129, 1)
        iwn = np.array([np.pi/beta*(2*n - 1)*1j for n in iwn])
        I = np.eye(H_bath.shape[0])
        V = v @ np.conj(Q.T)
        hyb = np.conj(V.T)[None, ...] @ np.linalg.solve(I[None, ...]*iwn[:, None, None] - H_bath[None, ...], V[None, ...])
        blocks = get_blocks(hyb)
        print("V = ")
        print("\n".join(["  ".join([f"{el.real: 5.3f} {el.imag:+5.3f}" for el in row]) for row in V]))
        print("H_bath = ")
        print("\n".join(["  ".join([f"{el.real: 5.3f} {el.imag:+5.3f}" for el in row]) for row in H_bath]))
        print(f"{blocks=}")
        for block in get_blocks(hyb):
            fig, ax = plt.subplots(nrows=len(block), ncols=len(block), squeeze=False)
            for (i, orb_i), (j, orb_j) in product(enumerate(block), repeat=2):
                ax[i,j].plot(iwn.imag, hyb[:, orb_i,orb_j].real, '-')
                ax[i,j].plot(iwn.imag, hyb[:, orb_i,orb_j].imag, '--')
            fig.suptitle(r"$\Delta_{fit}(\omega)$")
        plt.show()

    occupied_indices, positive_indices = partition_index(
        np.diag(H_bath), pred=lambda x: x < 0
    )
    sorted_bath_indices = np.array(occupied_indices + positive_indices, dtype=int)
    # sorted_bath_indices = np.argsort(np.diag(H_bath))
    H_bath = H_bath[np.ix_(sorted_bath_indices, sorted_bath_indices)]
    v = v[sorted_bath_indices, :]
    n_orb = H_dft.shape[0]
    H = np.zeros((n_orb + H_bath.shape[0], n_orb + H_bath.shape[0]), dtype=complex)
    H[:n_orb, :n_orb] = H_dft
    H[n_orb:, n_orb:] = H_bath
    H[n_orb:, :n_orb] = v @ np.conj(Q.T)
    H[:n_orb, n_orb:] = np.conj(H[n_orb:, :n_orb].T)

    impurity_indices, valence_bath_indices, conduction_bath_indices, block_structure = (
        build_imp_bath_blocks(H, n_orb)
    )

    if verbose:
        print("H = ")
        print(
            "\n".join(
                [
                    " ".join(
                        [f"{np.real(elem): .6f} {np.imag(elem):+.6f}i" for elem in row]
                    )
                    for row in H
                ]
            )
        )
        print(f"Diag(H) :\n{np.diag(H)}")

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
    parser.add_argument("--eim", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=0.001)
    parser.add_argument("--fit-unocc", action="store_true", dest="fit_unocc")
    parser.add_argument("-i", "--imag-only", action="store_true", dest="fit_real")
    parser.add_argument("-d", "--directory", type=str, default=".", dest="prefix")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--plot", action="store_true")
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
