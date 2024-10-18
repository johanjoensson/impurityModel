from os import devnull, remove
from collections import namedtuple
import traceback
import sys
import pickle
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from impmod_ed import ffi
from mpi4py import MPI
from rspt2spectra import offdiagonal, orbitals, h2imp, energies
import rspt2spectra.hyb_fit as hf

# hf.get_block_structure, hf.get_identical_blocks, hf.get_transposed_blocks, hf.fit_hyb
from impurityModel.ed.greens_function import (
    save_Greens_function,
    get_block_structure,
    get_identical_blocks,
    get_transposed_blocks,
    get_particle_hole_blocks,
    get_particle_hole_and_transpose_blocks,
    block_diagonalize_hyb,
)
from impurityModel.ed import finite
from impurityModel.ed.lanczos import Reort
from impurityModel.ed.greens_function import rotate_Greens_function, rotate_matrix, rotate_4index_U
from impurityModel.ed.manybody_basis import CIPSI_Basis
from impurityModel.ed.selfenergy import fixed_peak_dc
from impurityModel.ed.edchain import tridiagonalize, edchains, haverkort_chain

BlockStructure = namedtuple(
    "BlockStructure",
    [
        "inequivalent_blocks",
        "blocks",
        "identical_blocks",
        "transposed_blocks",
        "particle_hole_blocks",
        "particle_hole_transposed_blocks",
    ],
)


def get_hyb_chain(w, V0, H_bath):
    n_orb = V0.shape[1]
    assert H_bath.shape[0] % V0.shape[0] == 0
    I = np.identity(n_orb, dtype=complex)
    wI = w[:, np.newaxis, np.newaxis] * I[np.newaxis, :, :]
    hyb = wI - H_bath[np.newaxis, -n_orb:, -n_orb:]
    for i in range(H_bath.shape[0] // n_orb - 1, 0, -1):
        hyb[:] = (
            wI
            - H_bath[np.newaxis, (i - 1) * n_orb : i * n_orb, (i - 1) * n_orb : i * n_orb]
            - H_bath[np.newaxis, (i - 1) * n_orb : i * n_orb, i * n_orb : (i + 1) * n_orb]
            @ np.linalg.solve(hyb, H_bath[np.newaxis, i * n_orb : (i + 1) * n_orb, (i - 1) * n_orb : i * n_orb])
        )
    return np.conj(V0.T)[np.newaxis, :, :] @ np.linalg.solve(hyb, V0[np.newaxis, :, :])


def kth_diag_indices(m, k):
    rows, cols = np.diag_indices_from(m)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols


def matrix_print(matrix, label=None):
    if label is not None:
        print(label)
    print("\n".join([" ".join([f"{np.real(el): .6f} {np.imag(el):+.6f}j" for el in row]) for row in matrix]))


class ImpModCluster:
    def __init__(
        self,
        label,
        h_dft,
        hyb,
        u4,
        nominal_occ,
        delta_occ,
        impurity_orbitals,
        bath_states,
        sig,
        sig_real,
        sig_static,
        sig_dc,
        corr_to_spherical,
        corr_to_cf,
        blocked,
        spin_flip_dj,
        occ_restrict,
        chain_restrict,
        truncation_threshold,
    ):
        self.label = label
        self.h_dft = h_dft
        self.u4 = u4
        self.hyb = hyb
        self.impurity_orbitals = impurity_orbitals
        self.bath_states = bath_states
        self.nominal_occ = nominal_occ
        self.delta_occ = delta_occ
        self.sig = sig
        self.sig_real = sig_real
        self.sig_static = sig_static
        self.sig_dc = sig_dc
        self.corr_to_spherical = corr_to_spherical
        self.corr_to_cf = corr_to_cf
        self.spin_flip_dj = spin_flip_dj
        self.occ_restrict = occ_restrict
        self.chain_restrict = chain_restrict
        self.truncation_threshold = truncation_threshold

        if blocked:
            self.blocks = get_block_structure(
                self.hyb,
                h_dft,
            )
            self.identical_blocks = get_identical_blocks(
                self.blocks,
                self.hyb,
                h_dft,
            )
            self.transposed_blocks = get_transposed_blocks(
                self.blocks,
                self.hyb,
                h_dft,
            )
        else:
            # Use only one nxn block
            self.blocks = [[i for i in range(hyb.shape[1])]]
            self.identical_blocks = [[0]]
            self.transposed_blocks = [[]]

        self.inequivalent_blocks = []
        for blocks in self.identical_blocks:
            if len(blocks) == 0:
                continue
            unique = True
            for transpose in self.transposed_blocks:
                if blocks[0] in transpose[1:]:
                    unique = False
                    break
            if unique:
                self.inequivalent_blocks.append(blocks[0])


class dcStruct:
    def __init__(
        self,
        nominal_occ,
        delta_occ,
        impurity_orbitals,
        bath_states,
        u4,
        peak_position,
        dc_guess,
        spin_flip_dj,
        tau,
    ):
        self.nominal_occ = nominal_occ
        self.delta_occ = delta_occ
        self.impurity_orbitals = impurity_orbitals
        self.bath_states = bath_states
        self.u4 = u4
        self.peak_position = peak_position
        self.dc_guess = dc_guess
        self.spin_flip_dj = spin_flip_dj
        self.tau = tau

    def __repr__(self):
        return (
            f"dcStruct( nominal_occ = {self.nominal_occ},\n"
            f"          delta_occ = {self.delta_occ},\n"
            f"          num_spin_orbitals = {self.num_spin_orbitals},\n"
            f"          bath_states = {self.bath_states},\n"
            f"          peak_position = {self.peak_position})"
            f"          dc_guess = {self.dc_guess})"
        )


def parse_solver_line(solver_line):
    """
    N0 dN dVal dCon Nbath [[pro, full] [dense_cutoff 50] [no_block], [fit_unocc] [weight 2]]
    """
    # Remove comments from the solver line
    solver_line = solver_line.split("!")[0]
    solver_line = solver_line.split("#")[0]
    solver_array = solver_line.strip().split()
    assert (
        len(solver_array) >= 5
    ), "The impurityModel ED solver requires at least 5 arguments; N0 dN dValence dConduction nBaths"
    try:
        nominal_occ = int(solver_array[0])
        delta_occ = (int(solver_array[1]), int(solver_array[2]), int(solver_array[3]))
        nBaths = int(solver_array[4])
    except Exception as e:
        raise RuntimeError(
            f"{e}\n"
            f"--->N0 {solver_array[0]}\n"
            f"--->dN {solver_array[1]}\n"
            f"--->dValence {solver_array[2]}\n"
            f"--->dConduction {solver_array[3]}\n"
            f"--->Nbaths {solver_array[4]}\n"
            f"--->Other params {solver_array[5:]}"
        )
    options = {
        "dense_cutoff": 50,
        "reort": Reort.NONE,
        "blocked": True,
        "fit_unocc": False,
        "weight_function": "gaussian",
        "weight": 2,
        "spin_flip_dj": False,
        "bath_geometry": "star",
        "occ_restrict": True,
        "chain_restrict": True,
        "truncation_threshold": int(1e9),
    }
    if len(solver_array) > 5:
        skip_next = False
        for i in range(5, len(solver_array)):
            if skip_next:
                skip_next = False
                continue
            arg = solver_array[i]
            if arg.lower() in {"pro", "full"}:
                if arg.lower() == "pro":
                    options["reort"] = Reort.PARTIAL
                elif arg.lower() == "full":
                    options["reort"] = Reort.FULL
            elif arg.lower() in {"star", "chain", "haver"}:
                options["bath_geometry"] = arg.lower()
            elif arg.lower() == "fit_unocc":
                options["fit_unocc"] = True
            elif arg.lower() == "fit_occ":
                options["fit_unocc"] = False
            elif arg.lower() == "dense_cutoff":
                options["dense_cutoff"] = int(solver_array[i + 1])
                skip_next = True
            elif arg.lower() == "no_block":
                options["blocked"] = False
            elif arg.lower() in {"gaussian", "rspt", "exponential"}:
                options["weight_function"] = arg.lower()
            elif arg.lower() == "weight":
                options["weight"] = float(solver_array[i + 1])
                skip_next = True
            elif arg.lower() == "spin_flip_dj":
                options["spin_flip_dj"] = True
            elif arg.lower() == "no_restrict":
                options["occ_restrict"] = False
            elif arg.lower() == "no_chain_restrict":
                options["chain_restrict"] = False
            elif arg.lower() == "truncation_threshold":
                options["truncation_threshold"] = int(solver_array[i + 1])
                skip_next = True
            else:
                raise RuntimeError(f"Unknown solver parameter {arg}.\n" f"--->Other solver params {solver_array[5:]}")
    if options["bath_geometry"] == "star":
        options["chain_restrict"] = False

    print(
        f"Nominal imp. occupation  |> {nominal_occ}\n"
        f"Delta occupation         |> {delta_occ}\n"
        f"# bath states / imp. orb.|> {nBaths}\n"
        f"Bath geometry            |> {options['bath_geometry']}\n"
        f"Fit unoccupied states    |> {options['fit_unocc']}\n"
        f"Generate spin fliped Djs |> {options['spin_flip_dj']}\n"
        f"Use block structure      |> {options['blocked']}\n"
        f"Reorthogonalizaion mode  |> {options['reort']}\n"
        f"Dense matrix size cutoff |> {options['dense_cutoff']}\n"
        f"Fitting weight function  |> {options['weight_function']}\n"
        f"Fitting weight factor    |> {options['weight']}\n"
        f"Occupation restrictions  |> {options['occ_restrict']}\n"
        f"Chain occ. restrictions  |> {options['chain_restrict']}\n"
    )
    return nominal_occ, delta_occ, nBaths, options


def get_weight_function(weight_function_name, w0, e):
    if weight_function_name.lower() == "gaussian":
        return lambda w: np.exp(-e * np.abs(w - w0))
    elif weight_function_name.lower() == "exponential":
        return lambda w: np.exp(-e / 2 * np.abs(w - w0) ** 2)
    elif weight_function_name.lower() == "rspt":
        return lambda w: np.abs(w - w0) / (1 + e * np.abs(w - w0)) ** 3
    else:
        raise RuntimeError(f"Unknown weight function {weight_function_name}")
    return None


@ffi.def_extern()
def run_impmod_ed(
    rspt_label,
    rspt_solver_line,
    rspt_dc_line,
    rspt_dc_flag,
    rspt_u4,
    rspt_hyb,
    rspt_h_dft,
    rspt_sig,
    rspt_sig_real,
    rspt_sig_static,
    rspt_sig_dc,
    rspt_iw,
    rspt_w,
    rspt_corr_to_spherical,
    rspt_corr_to_cf,
    n_orb,
    n_rot_cols,
    n_orb_full,
    n_iw,
    n_w,
    eim,
    tau,
    verbosity,
    size_real,
    size_complex,
):
    comm = MPI.COMM_WORLD
    rank = comm.rank

    label = ffi.string(rspt_label, 18).decode("ascii")
    solver_line = ffi.string(rspt_solver_line, 100).decode("ascii")

    h_dft = np.ndarray(
        buffer=ffi.buffer(rspt_h_dft, n_orb * n_orb * size_complex), shape=(n_orb, n_orb), order="F", dtype=complex
    )
    u4 = np.ndarray(
        buffer=ffi.buffer(rspt_u4, n_orb * n_orb * n_orb * n_orb * size_complex),
        shape=(n_orb, n_orb, n_orb, n_orb),
        order="F",
        dtype=complex,
    )
    hyb = np.ndarray(
        buffer=ffi.buffer(rspt_hyb, n_w * n_orb * n_orb * size_complex),
        shape=(n_orb, n_orb, n_w),
        order="F",
        dtype=complex,
    )
    iw = np.ndarray(buffer=ffi.buffer(rspt_iw, n_iw * size_real), shape=(n_iw,), dtype=float)
    w = np.ndarray(buffer=ffi.buffer(rspt_w, n_w * size_real), shape=(n_w,), dtype=float)
    sig = np.ndarray(
        buffer=ffi.buffer(rspt_sig, n_iw * n_orb * n_orb * size_complex),
        shape=(n_orb, n_orb, n_iw),
        order="F",
        dtype=complex,
    )
    sig_real = np.ndarray(
        buffer=ffi.buffer(rspt_sig_real, n_w * n_orb * n_orb * size_complex),
        shape=(n_orb, n_orb, n_w),
        order="F",
        dtype=complex,
    )
    sig_static = np.ndarray(
        buffer=ffi.buffer(rspt_sig_static, n_orb * n_orb * size_complex), shape=(n_orb, n_orb), order="F", dtype=complex
    )
    sig_dc = np.ndarray(
        buffer=ffi.buffer(rspt_sig_dc, n_orb * n_orb * size_complex), shape=(n_orb, n_orb), order="F", dtype=complex
    )
    rspt_corr_to_spherical_arr = np.ndarray(
        buffer=ffi.buffer(rspt_corr_to_spherical, n_orb * n_orb_full * size_complex),
        shape=(n_orb, n_orb_full),
        order="F",
        dtype=complex,
    )
    rspt_corr_to_cf_arr = np.ndarray(
        buffer=ffi.buffer(rspt_corr_to_cf, n_orb * n_rot_cols * size_complex),
        shape=(n_orb, n_rot_cols),
        order="F",
        dtype=complex,
    )

    if n_rot_cols == n_orb_full and n_orb == n_orb_full:
        corr_to_spherical = rspt_corr_to_spherical_arr
        corr_to_cf = rspt_corr_to_cf_arr
    else:
        corr_to_spherical = np.empty((n_orb, 2 * n_orb_full), dtype=complex)
        corr_to_cf = np.empty((n_orb, n_orb), dtype=complex)
        corr_to_spherical[:, :n_orb_full] = rspt_corr_to_spherical_arr
        corr_to_spherical[:, n_orb_full:] = np.roll(rspt_corr_to_spherical_arr, n_orb_full, axis=0)
        corr_to_cf[:, :n_rot_cols] = rspt_corr_to_cf_arr
        corr_to_cf[:, n_rot_cols:] = np.roll(rspt_corr_to_cf_arr, n_rot_cols, axis=0)
    # Rotate the U-matrix to the CF basis
    u4 = rotate_4index_U(u4, corr_to_cf)
    # impurityModel uses a weird convention for the U-matrix
    u4 = np.moveaxis(u4, 1, 0)

    # For python, it makes more sense to put the frequency index first, instead of last
    sig_python = np.moveaxis(sig, -1, 0)
    sig_real_python = np.moveaxis(sig_real, -1, 0)
    hyb = np.moveaxis(hyb, -1, 0)

    # Rotate hybridization function and DFT hamiltonian to the CF basis
    hyb = rotate_Greens_function(hyb, corr_to_cf)
    h_dft = rotate_matrix(h_dft, corr_to_cf)

    stdout_save = sys.stdout
    if rank == 0:
        sys.stdout = open(f"impurityModel-{label.strip()}{'-dc' if rspt_dc_flag == 1 else ''}.out", "w")
    elif verbosity > 0:
        sys.stdout = open(f"impurityModel-{label.strip()}{'-dc' if rspt_dc_flag == 1 else ''}-{rank}.out", "w")
    else:
        sys.stdout = open(devnull, "w")

    (nominal_occ, delta_occ, bath_states_per_orbital, options) = parse_solver_line(solver_line)
    nominal_occ = {0: nominal_occ}
    delta_occ = ({0: delta_occ[0]}, {0: delta_occ[1]}, {0: delta_occ[2]})
    if any(n0 > n_orb for n0 in nominal_occ.values()) or any(n0 < 0 for n0 in nominal_occ.values()):
        raise RuntimeError(f"Nominal impurity occupation {nominal_occ} out of bounds [0, {n_orb}]")

    h_op, imp_bath_blocks = get_ed_h0(
        h_dft,
        hyb,
        corr_to_cf,
        bath_states_per_orbital,
        w,
        eim,
        gamma=0.001,
        weight_function=options["weight_function"],
        exp_weight=options["weight"],
        imag_only=False,
        valence_bath_only=not options["fit_unocc"],
        bath_geometry=options["bath_geometry"],
        label=label.strip(),
        save_baths_and_hopping=rspt_dc_flag == 1,
        verbose=(verbosity >= 0 or rspt_dc_flag == 1),
        comm=comm,
    )
    if rank == 0:
        with open(f"Ham-op-{label.strip()}.pickle", "wb") as f:
            pickle.dump(h_op, f)

    if rspt_dc_flag == 1:
        dc_line = ffi.string(rspt_dc_line, 100).decode("ascii")
        dc_line = dc_line.split("!")[0]
        dc_line = dc_line.split("#")[0]
        dc_array = dc_line.strip().split()
        assert len(dc_array) == 1
        peak_position = float(dc_array[0])

        dc_struct = dcStruct(
            nominal_occ=nominal_occ,
            delta_occ=delta_occ,
            impurity_orbitals={0: [block[0] for block in imp_bath_blocks]},
            bath_states=(
                {0: [block[1] for block in imp_bath_blocks]},
                {0: [block[2] for block in imp_bath_blocks]},
                {0: [block[3] for block in imp_bath_blocks]},
            ),
            u4=u4,
            peak_position=peak_position,
            dc_guess=sig_dc,
            spin_flip_dj=options["spin_flip_dj"],
            tau=tau,
        )

        try:
            sig_dc[:, :] = fixed_peak_dc(
                h_op, dc_struct, rank=rank, verbose=verbosity > 0, dense_cutoff=options["dense_cutoff"]
            )
            er = 0
        except Exception as e:
            print("!" * 100)
            print(f"Exception {repr(e)} caught on rank {rank}!")
            print(traceback.format_exc())
            print("Adding positive infinity to the imaginary part of the DC selfenergy.", flush=True)
            print("!" * 100)
            sig_dc[:, :] = np.inf + 1j * np.inf
            er = -1
            comm.Abort(er)

        sys.stdout.close()
        sys.stdout = stdout_save
        return er

    cluster = ImpModCluster(
        label=label.strip(),
        h_dft=h_dft,
        hyb=hyb,
        u4=u4,
        impurity_orbitals={0: [block[0] for block in imp_bath_blocks]},
        bath_states=(
            {0: [block[1] for block in imp_bath_blocks]},
            {0: [block[2] for block in imp_bath_blocks]},
            {0: [block[3] for block in imp_bath_blocks]},
        ),
        nominal_occ=nominal_occ,
        delta_occ=delta_occ,
        sig=sig_python,
        sig_real=sig_real_python,
        sig_static=sig_static,
        sig_dc=sig_dc,
        corr_to_spherical=corr_to_spherical,
        corr_to_cf=corr_to_cf,
        blocked=options["blocked"],
        spin_flip_dj=options["spin_flip_dj"],
        occ_restrict=options["occ_restrict"],
        chain_restrict=options["chain_restrict"],
        truncation_threshold=options["truncation_threshold"],
    )

    from impurityModel.ed import selfenergy

    try:
        selfenergy.run(
            cluster,
            h_op,
            1j * iw,
            w,
            eim,
            tau,
            verbosity,
            reort=options["reort"],
            dense_cutoff=options["dense_cutoff"],
            comm=comm,
        )

        # Rotate self energy from CF basis to RSPt's corr basis
        u = np.conj(corr_to_cf.T)
        cluster.sig[:, :, :] = rotate_Greens_function(cluster.sig, u)
        cluster.sig_real[:, :, :] = rotate_Greens_function(cluster.sig_real, u)
        cluster.sig_static[:, :] = rotate_matrix(cluster.sig_static, u)

        comm.Bcast(sig_static, root=0)
        comm.Bcast(sig_real, root=0)
        comm.Bcast(sig, root=0)
        er = 0
    except Exception as e:
        print("!" * 100)
        print(f"Exception {repr(e)} caught on rank {rank}!")
        print(traceback.format_exc())
        print(
            "Adding positive infinity to the imaginary part of the selfenergy at the last matsubara frequency.",
            flush=True,
        )
        print("!" * 100)
        cluster.sig[:, :, -1] += 1j * np.inf
        er = -1
        comm.Abort(er)
    else:
        print("Self energy calculated! impurityModel shutting down.", flush=True)

    sys.stdout.close()
    sys.stdout = stdout_save
    return er


def get_ed_h0(
    H_dft,
    hyb,
    corr_to_cf,
    bath_states_per_orbital,
    w,
    eim,
    gamma=0.001,
    exp_weight=2,
    weight_function="Gaussian",
    weight_w0=0,
    imag_only=False,
    valence_bath_only=True,
    bath_geometry="star",
    label=None,
    save_baths_and_hopping=False,
    verbose=True,
    comm=None,
):
    """
    Calculate the non-interacting hamiltonian, h0, for use in exact diagonalization.
    Bath states are fitted to the real frequency hybridization function.
    In block form h0 can be written
    [ h_dft  V^+ ]
    [  V     Eb ],
    where h_dft is the dft hamiltonian projected onto the correlated orbitals, V is
    the hopping amplitudes between the impurity and the bath, and Eb is a diagonal
    matrix with energies of the bath states along the diagonal.
    Parameters:
    hyb           -- The real frequency hybridiaztion function. Used to fit the bath states.
    hdft          -- The DFT hamiltonian, projected onto the impurity orbitals.
    bath_states   -- Number of bath states to fit per impurity orbital.
    rot_spherical -- Transformation matrix to transform to spherical harmonics basis.
    w             -- Real frequency mesh.
    eim           -- All real frequency quantities are evaluated i*eim above the real frequency axis.
    gamma         -- Regularization parameter.
    imag_only     -- Only fit the imaginary part of the hybridization function, default: False.
    valence_bath_only -- Only fit bath stated in the valence band, default: True.
    label          -- Label for the cluster, used for saving a copy of the Hamiltonian that can be plugged into the Matsubara
    ED solver in RSPr, default: None,

    Returns:
    h0   -- The non-interacting impurity hamiltonian in operator form.
    eb   -- The bath states used for fitting the hybridization function.
    """

    # We do the fitting by first transforming the hyridization function into a basis
    # where each block is (hopefully) close to diagonal
    # np.conj(Q.T) @ cf_hyb @ Q is the transformation performed
    phase_hyb, Q = block_diagonalize_hyb(hyb)

    block_structure = build_block_structure(phase_hyb, tol=1e-6)

    vs_star = None
    ebs_star = None
    if comm.rank == 0:
        # Check to see if we have already done a fit
        vs_star = []
        ebs_star = []
        try:
            with open(f"impurityModel_bath_energies_and_hopping_parameters_{label}.npy", "rb") as f:
                n_block = np.load(f)
                for _ in range(n_block):
                    vs_star.append(np.load(f))
                    ebs_star.append(np.load(f))
            remove(f"impurityModel_bath_energies_and_hopping_parameters_{label}.npy")
        except FileNotFoundError:
            vs_star = None
            ebs_star = None
        except ValueError:
            remove(f"impurityModel_bath_energies_and_hopping_parameters_{label}.npy")
            vs_star = None
            ebs_star = None
    if ebs_star is not None and verbose:
        print(
            f"Read bath energies and hopping parameters from impurityModel_bath_energies_and_hopping_parameters_{label}.npy"
        )

    ebs_star, vs_star = fit_hyb(
        w,
        eim,
        phase_hyb,
        bath_states_per_orbital,
        block_structure,
        gamma=gamma,
        imag_only=imag_only,
        x_lim=(w[0], 0 if valence_bath_only else w[-1]),
        verbose=verbose,
        comm=comm,
        weight_fun=get_weight_function(weight_function, weight_w0, exp_weight),
        ebs_guess=ebs_star,
        vs_guess=vs_star,
    )
    for ebss, vss in zip(ebs_star, vs_star):
        if len(ebss) == 0:
            continue
        sorted_indices = np.argsort(ebss, kind="stable")
        ebss[:] = ebss[sorted_indices]
        vss[:] = vss[sorted_indices]
    assert len(vs_star) == len(block_structure.inequivalent_blocks), "Number of inequivalent blocks is inconsitent"
    n_occ_block = [np.sum(eb < -1e-2) for i, eb in enumerate(ebs_star)]
    n_zero_block = [np.sum(np.abs(eb) <= 1e-2) for i, eb in enumerate(ebs_star)]
    n_empty_block = [np.sum(eb > 1e-2) for i, eb in enumerate(ebs_star)]
    H_bath_star, v_star = build_full_bath([np.diag(eb) for eb in ebs_star], vs_star, block_structure)

    if verbose:
        print(f"Star bath energies:")
        for eb in ebs_star:
            print(eb)
        print(f"Star hopping parameters:")
        for vb in vs_star:
            matrix_print(vb)
            print("")
    if bath_geometry == "star":
        H_bath, v = H_bath_star, v_star
    elif bath_geometry == "chain":
        H_baths = []
        vs = []
        for v, ebs in zip(vs_star, ebs_star):
            (H_bath_occ, v_occ), (H_bath_unocc, v_unocc) = edchains(v, ebs)
            H_baths.append(sp.linalg.block_diag(H_bath_occ, H_bath_unocc))
            vs.append(np.vstack((v_occ, v_unocc)))
        if verbose:
            print(f"Chain baths")
            for Hb in H_baths:
                matrix_print(Hb)
                print("")
            print(f"Hopping parameters")
            for vb in vs:
                matrix_print(vb)
                print("")
        H_baths, vs = build_full_bath(H_baths, vs, block_structure)
        H_bath = sp.linalg.block_diag(H_baths)
        v = np.vstack(tuple(vs))
    elif bath_geometry == "haver":
        H_baths = []
        vs = []
        for i_b, (vss, ebss) in enumerate(zip(vs_star, ebs_star)):
            sorted_indices = np.argsort(ebss)
            ebss = ebss[sorted_indices]
            vss = vss[sorted_indices]
            ebs_chain, tns_chain, v0 = tridiagonalize(ebss, vss)
            block_ix = block_structure.inequivalent_blocks[i_b]
            block_orbs = block_structure.blocks[block_ix]
            b_ix = np.ix_(block_orbs, block_orbs)
            vh, Hh = haverkort_chain(H_dft[b_ix], np.append(v0, tns_chain[:-1]), ebs_chain)
            H_baths.append(Hh)
            vs.append(vh)
        if verbose:
            print("Haverkort chain baths")
            for Hb in H_baths:
                matrix_print(Hb)
                print("")
            print("Hopping parameters")
            for vb in vs:
                matrix_print(vb)
                print("")
        H_bath, v = build_full_bath(H_baths, vs, block_structure)

    if save_baths_and_hopping or True:
        if comm is None or comm.rank == 0:
            with open(f"impurityModel_bath_energies_and_hopping_parameters_{label}.npy", "wb") as f:
                np.save(f, len(vs_star))
                for i in range(len(vs_star)):
                    np.save(f, vs_star[i])
                    np.save(f, ebs_star[i])
            with open(f"impurityModel_bath_energies_and_hopping_parameters_{label}_bak.npy", "wb") as f:
                np.save(f, len(vs_star))
                for i in range(len(vs_star)):
                    np.save(f, vs_star[i])
                    np.save(f, ebs_star[i])

    if verbose:
        print("DFT hamiltonian in correlated basis")
        matrix_print(rotate_matrix(H_dft, np.conj(corr_to_cf.T)))
        print("DFT hamiltonian in CF basis")
        matrix_print(H_dft)
        print("Hopping parameters in CF basis")
        matrix_print(v)
        print("Bath state energies")
        print(np.array_str(np.diag(H_bath), max_line_width=1000, precision=4, suppress_small=False))

    n_orb = H_dft.shape[0]
    H = np.zeros((n_orb + H_bath.shape[0], n_orb + H_bath.shape[0]), dtype=complex)
    H[:n_orb, :n_orb] = H_dft
    H[n_orb:, n_orb:] = H_bath
    H[n_orb:, :n_orb] = v @ np.conj(Q.T)
    H[:n_orb, n_orb:] = np.conj(H[n_orb:, :n_orb].T)

    if verbose:
        print("DFT hamiltonian, with baths, in CF basis")
        matrix_print(H)

        if comm is None or comm.rank == 0:
            hyb_star = np.conj((v_star @ np.conj(Q.T)).T) @ np.linalg.solve(
                (w + 1j * eim)[:, None, None] * np.identity(H_bath_star.shape[0], dtype=complex)[None, :, :]
                - H_bath_star[None, :, :],
                (v_star @ np.conj(Q.T))[None, :, :],
            )
            hyb = np.conj(v @ np.conj(Q.T)).T @ np.linalg.solve(
                (w + 1j * eim)[:, None, None] * np.identity(H_bath.shape[0], dtype=complex)[None, :, :] - H_bath,
                (v @ np.conj(Q.T))[None, :, :],
            )
            save_Greens_function(rotate_Greens_function(hyb_star, np.conj(corr_to_cf.T)), w, f"hyb-star-fit-{label}")
            save_Greens_function(rotate_Greens_function(hyb, np.conj(corr_to_cf.T)), w, f"hyb-fit-{label}")

        H_tmp = np.zeros((n_orb + H_bath_star.shape[0], n_orb + H_bath_star.shape[0]), dtype=complex)
        H_tmp[:n_orb, :n_orb] = corr_to_cf @ H_dft @ np.conj(corr_to_cf).T
        H_tmp[n_orb:, n_orb:] = H_bath_star
        H_tmp[n_orb:, :n_orb] = v_star @ np.conj(Q.T) @ np.conj(corr_to_cf).T
        H_tmp[:n_orb, n_orb:] = np.conj(H_tmp[n_orb:, :n_orb].T)
        print("DFT hamiltonian, with star geometry baths, in correlated basis")
        matrix_print(H_tmp)
        with open(f"Ham-{label}{'-dc' if save_baths_and_hopping else ''}.inp", "w") as f:
            for i in range(H_tmp.shape[0]):
                for j in range(H_tmp.shape[1]):
                    f.write(f" 0 0 0 {i+1} {j+1} {np.real(H_tmp[i, j])} {np.imag(H_tmp[i, j])}\n")
        assert np.allclose(np.linalg.eigvalsh(H), np.linalg.eigvalsh(H_tmp))

    occupied_indices = [None] * len(block_structure.blocks)
    zero_indices = [None] * len(block_structure.blocks)
    unoccupied_indices = [None] * len(block_structure.blocks)
    for inequiv_i, block_i in enumerate(block_structure.inequivalent_blocks):
        for identical_block in block_structure.identical_blocks[block_i]:
            occupied_indices[identical_block] = list(range(n_occ_block[inequiv_i]))
            zero_indices[identical_block] = list(range(n_zero_block[inequiv_i]))
            unoccupied_indices[identical_block] = list(range(n_empty_block[inequiv_i]))
        for transposed_block in block_structure.transposed_blocks[block_i]:
            occupied_indices[transposed_block] = list(range(n_occ_block[inequiv_i]))
            zero_indices[transposed_block] = list(range(n_zero_block[inequiv_i]))
            unoccupied_indices[transposed_block] = list(range(n_empty_block[inequiv_i]))
        for particle_hole_block in block_structure.particle_hole_blocks[block_i]:
            occupied_indices[particle_hole_block] = list(range(n_occ_block[inequiv_i]))
            zero_indices[particle_hole_block] = list(range(n_zero_block[inequiv_i]))
            unoccupied_indices[particle_hole_block] = list(range(n_empty_block[inequiv_i]))
        for particle_hole_transpose_block in block_structure.particle_hole_transposed_blocks[block_i]:
            occupied_indices[particle_hole_transpose_block] = list(range(n_occ_block[inequiv_i]))
            zero_indices[particle_hole_transpose_block] = list(range(n_zero_block[inequiv_i]))
            unoccupied_indices[particle_hole_transpose_block] = list(range(n_empty_block[inequiv_i]))
    offset = n_orb
    for i in range(len(block_structure.blocks)):
        occupied_indices[i] = [index + offset for index in occupied_indices[i]]
        offset += len(occupied_indices[i])
        zero_indices[i] = [index + offset for index in zero_indices[i]]
        offset += len(zero_indices[i])
        unoccupied_indices[i] = [index + offset for index in unoccupied_indices[i]]
        offset += len(unoccupied_indices[i])
    occupied_indices = {i for block in occupied_indices for i in block}
    zero_indices = {i for block in zero_indices for i in block}
    unoccupied_indices = {i for block in unoccupied_indices for i in block}

    imp_bath_mask = np.abs(H) > 1e-6

    n_blocks, block_idxs = connected_components(csgraph=csr_matrix(imp_bath_mask), directed=False, return_labels=True)
    blocks = [[] for _ in range(n_blocks)]
    for orb_i, block_i in enumerate(block_idxs):
        blocks[block_i].append(orb_i)
    imp_bath_blocks = [None] * n_blocks
    for block_i, bath_block in enumerate(blocks):
        imp_orbs = [i for i in bath_block if i in range(n_orb)]
        occ_baths = [i for i in bath_block if i in occupied_indices]
        zero_baths = [i for i in bath_block if i in zero_indices]
        empty_baths = [i for i in bath_block if i in unoccupied_indices]
        imp_bath_blocks[block_i] = (imp_orbs, occ_baths, zero_baths, empty_baths)

    h_op = finite.matrixToIOp(H)
    return h_op, imp_bath_blocks


def fit_hyb(
    w,
    delta,
    hyb,
    bath_states_per_orbital,
    block_structure,
    gamma,
    imag_only,
    x_lim=None,
    tol=1e-6,
    verbose=True,
    comm=None,
    weight_fun=lambda w, w0, e: np.ones_like(w),
    ebs_guess=None,
    vs_guess=None,
):
    """
    Calculate the bath energies and hopping parameters for fitting the
    hybridization function.

    Parameters:
    w           -- Real frequency mesh
    delta       -- All quantities will be evaluated i*delta above the real
                   frequency line.
    hyb         -- Hybridization function
    bath_states_per_orbital --Number of bath states to fit for each orbital
    w_lim       -- (w_min, w_max) Only fit for frequencies w_min <= w <= w_max.
                   If not set, fit for all w.
    Returns:
    eb          -- Bath energies
    v           -- Hopping parameters
    """
    if bath_states_per_orbital == 0:
        return [np.empty((0,), dtype=float) for ib in block_structure.inequivalent_blocks], [
            np.empty((0, len(block_structure.blocks[ib])), dtype=complex) for ib in block_structure.inequivalent_blocks
        ]
    if x_lim is not None:
        mask = np.logical_and(w >= x_lim[0], w <= x_lim[1])
    else:
        mask = np.array([True] * len(w))

    if verbose:
        print(block_structure)

    ebs_star = [np.empty((0,), dtype=float) for ib in block_structure.inequivalent_blocks]
    vs_star = [
        np.empty((0, len(block_structure.blocks[ib])), dtype=complex) for ib in block_structure.inequivalent_blocks
    ]
    if verbose:
        print(f"inequivalent blocks = {block_structure.inequivalent_blocks}")
    states_per_inequivalent_block = get_state_per_inequivalent_block(
        block_structure,
        bath_states_per_orbital,
        hyb[mask, :, :],
        w[mask],
        weight_fun,
    )

    #### Do the fit
    for inequivalent_block_i, block_i in enumerate(block_structure.inequivalent_blocks):
        if states_per_inequivalent_block[inequivalent_block_i] == 0:
            continue
        block = block_structure.blocks[block_i]
        idx = np.ix_(range(hyb.shape[0]), block, block)
        block_hyb = hyb[idx]
        realvalue_v = np.all(np.abs(block_hyb - np.transpose(block_hyb, (0, 2, 1))) < 1e-6)
        block_eb_star, block_vs_star = hf.fit_block(
            block_hyb[mask, :, :],
            w[mask],
            delta,
            states_per_inequivalent_block[inequivalent_block_i],
            gamma=gamma,
            imag_only=imag_only,
            realvalue_v=realvalue_v,
            comm=comm,
            verbose=verbose,
            weight_fun=weight_fun,
            bath_guess=ebs_guess[inequivalent_block_i] if ebs_guess is not None else None,
            v_guess=vs_guess[inequivalent_block_i] if vs_guess is not None else None,
        )
        # Remove states with negligleble hopping
        bath_mask = []
        for group_i in range(0, block_vs_star.shape[0], len(block)):
            if np.any(np.all(np.abs(block_vs_star[group_i : group_i + len(block)]) ** 2 < 1e-10, axis=1)):
                bath_mask.extend([False] * len(block))
            else:
                bath_mask.extend([True] * len(block))
        block_vs_star = block_vs_star[bath_mask]
        block_eb_star = block_eb_star[bath_mask]

        vs_star[inequivalent_block_i] = block_vs_star
        ebs_star[inequivalent_block_i] = block_eb_star

    return ebs_star, vs_star


def get_inequivalent_blocks(
    identical_blocks,
    transposed_blocks,
    particle_hole_blocks,
    particle_hole_and_transpose_blocks,
):
    inequivalent_blocks = []
    for blocks in identical_blocks:
        if len(blocks) == 0:
            continue
        unique = True
        for transpose in transposed_blocks:
            if blocks[0] in transpose[1:]:
                unique = False
                break
        for particle_hole in particle_hole_blocks:
            if blocks[0] in particle_hole[1:]:
                unique = False
                break
        for particle_hole_and_transpose in particle_hole_and_transpose_blocks:
            if blocks[0] in particle_hole_and_transpose[1:]:
                unique = False
                break
        if unique:
            inequivalent_blocks.append(blocks[0])
    return inequivalent_blocks


def get_state_per_inequivalent_block(
    block_structure,
    bath_states_per_orbital,
    hyb,
    w,
    weight_fun,
):
    (
        inequivalent_blocks,
        blocks,
        identical_blocks,
        transposed_blocks,
        particle_hole_blocks,
        particle_hole_and_transpose_blocks,
    ) = block_structure

    orbitals_per_inequivalent_block = [0] * len(inequivalent_blocks)
    weight_per_inequivalent_block = np.zeros((len(inequivalent_blocks)), dtype=float)
    for inequivalent_block_i, block_i in enumerate(inequivalent_blocks):
        block = blocks[block_i]
        block_multiplicity = (
            len(identical_blocks[block_i])
            + len(transposed_blocks[block_i])
            + len(particle_hole_blocks[block_i])
            + len(particle_hole_and_transpose_blocks[block_i])
        )
        orbitals_per_inequivalent_block[inequivalent_block_i] = len(block) * block_multiplicity
        idx = np.ix_(range(hyb.shape[0]), block, block)
        block_hyb = hyb[idx]
        weight_per_inequivalent_block[inequivalent_block_i] = (
            np.trapz(
                -np.imag(np.sum(np.diagonal(block_hyb, axis1=1, axis2=2), axis=1)) * weight_fun(w),
                w,
            )
            * block_multiplicity
        )
    states_per_inequivalent_block = np.round(
        weight_per_inequivalent_block
        / np.sum(weight_per_inequivalent_block)
        * np.sum(orbitals_per_inequivalent_block)
        * bath_states_per_orbital
        / orbitals_per_inequivalent_block
    ).astype(int)
    states_per_inequivalent_block[states_per_inequivalent_block < 0] = 0
    return states_per_inequivalent_block


def build_block_structure(hyb, tol):
    blocks = get_block_structure(hyb, tol=tol)
    identical_blocks = get_identical_blocks(blocks, hyb, tol=tol)
    transposed_blocks = get_transposed_blocks(blocks, hyb, tol=tol)
    particle_hole_blocks = get_particle_hole_blocks(blocks, hyb, tol=tol)
    particle_hole_and_transpose_blocks = get_particle_hole_and_transpose_blocks(blocks, hyb, tol=tol)
    inequivalent_blocks = get_inequivalent_blocks(
        identical_blocks,
        transposed_blocks,
        particle_hole_blocks,
        particle_hole_and_transpose_blocks,
    )

    return BlockStructure(
        inequivalent_blocks,
        blocks,
        identical_blocks,
        transposed_blocks,
        particle_hole_blocks,
        particle_hole_and_transpose_blocks,
    )


def build_full_bath(H_bath_inequiv, v_inequiv, block_structure: BlockStructure):
    (
        inequivalent_blocks,
        blocks,
        identical_blocks,
        transposed_blocks,
        particle_hole_blocks,
        particle_hole_and_transposed_blocks,
    ) = block_structure
    n_orb = sum(len(b) for b in blocks)
    H_baths = [None] * len(blocks)
    vs = [None] * len(blocks)
    for i, block_i in enumerate(inequivalent_blocks):
        H_bath = H_bath_inequiv[i]
        v = v_inequiv[i]
        for b in identical_blocks[block_i]:
            v_tmp = np.zeros((v.shape[0], n_orb), dtype=complex)
            H_baths[b] = H_bath.copy()
            v_tmp[:, blocks[b]] = v
            vs[b] = v_tmp
        for b in transposed_blocks[block_i]:
            v_tmp = np.zeros((v.shape[0], n_orb), dtype=complex)
            H_baths[b] = H_bath.copy().T
            v_tmp[:, blocks[b]] = v
            vs[b] = v_tmp
        for b in particle_hole_blocks[block_i]:
            v_tmp = np.zeros((v.shape[0], n_orb), dtype=complex)
            H_baths[b] = H_bath.copy() @ (-np.identity(H_bath.shape[0]))
            v_tmp[:, blocks[b]] = v
            vs[b] = v_tmp
        for b in particle_hole_and_transposed_blocks[block_i]:
            v_tmp = np.zeros((v.shape[0], n_orb), dtype=complex)
            H_baths[b] = H_bath.copy().T @ (-np.identity(H_bath.shape[0]))
            v_tmp[:, blocks[b]] = v
            vs[b] = v_tmp
    return sp.linalg.block_diag(*H_baths), np.vstack(vs)
