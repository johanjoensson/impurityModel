from os import devnull, remove
import traceback
import sys
import pickle
import numpy as np
import scipy as sp
from impmod_ed import ffi
from mpi4py import MPI
from rspt2spectra import offdiagonal, orbitals, h2imp
import rspt2spectra.hyb_fit as hf

# hf.get_block_structure, hf.get_identical_blocks, hf.get_transposed_blocks, hf.fit_hyb
from rspt2spectra import h2imp, energies
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
from impurityModel.ed.edchain import tridiagonalize, edchains


def kth_diag_indices(m, k):
    rows, cols = np.diag_indices_from(m)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols


def matrix_print(matrix):
    print("\n".join([" ".join([f"{np.real(el): .6f} {np.imag(el):+.6f}j" for el in row]) for row in matrix]))


class ImpModCluster:
    def __init__(
        self,
        label,
        h_dft,
        hyb,
        h_star_bath,
        v_star,
        u4,
        nominal_occ,
        delta_occ,
        n_bath_states,
        sig,
        sig_real,
        sig_static,
        sig_dc,
        corr_to_spherical,
        corr_to_cf,
        blocked,
        spin_flip_dj,
    ):
        self.label = label
        self.h_dft = h_dft
        self.u4 = u4
        self.hyb = hyb
        self.h_star_bath = h_star_bath
        self.v_star = v_star
        self.bath_states = n_bath_states
        self.nominal_occ = nominal_occ
        self.delta_occ = delta_occ
        self.sig = sig
        self.sig_real = sig_real
        self.sig_static = sig_static
        self.sig_dc = sig_dc
        self.corr_to_spherical = corr_to_spherical
        self.corr_to_cf = corr_to_cf
        self.spin_flip_dj = spin_flip_dj

        valence_baths, conduction_baths = self.bath_states
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
        nominal_occ = (int(solver_array[0]), 0, 0)
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
    dense_cutoff = 50
    reort = Reort.NONE
    blocked = True
    fit_unocc = False
    weight = 2
    spin_flip_dj = False
    if len(solver_array) > 5:
        skip_next = False
        for i in range(5, len(solver_array)):
            if skip_next:
                skip_next = False
                continue
            arg = solver_array[i]
            if arg.lower() == "pro":
                reort = Reort.PARTIAL
            elif arg.lower() == "full":
                reort = Reort.FULL
            elif arg.lower() == "dense_cutoff":
                dense_cutoff = int(solver_array[i + 1])
                skip_next = True
            elif arg.lower() == "no_block":
                blocked = False
            elif arg.lower() == "fit_unocc":
                fit_unocc = True
            elif arg.lower() == "weight":
                weight = float(solver_array[i + 1])
                skip_next = True
            elif arg.lower() == "spin_flip_dj":
                spin_flip_dj = True
            else:
                raise RuntimeError(f"Unknown solver parameter {arg}.\n" f"--->Other solver params {solver_array[5:]}")

    print(
        f"Nominal imp. occupation  |> {nominal_occ[0]}\n"
        f"Delta occupation         |> {delta_occ}\n"
        f"# bath states / imp. orb.|> {nBaths}\n"
        f"Reorthogonalizaion mode  |> {reort}\n"
        f"Dense matrix size cutoff |> {dense_cutoff}\n"
        f"Use block structure      |> {blocked}\n"
        f"Fit unoccupied states    |> {fit_unocc}\n"
        f"Fitting weight factor    |> {weight}\n"
        f"Generate spin fliped Djs |> {spin_flip_dj}\n"
    )
    return nominal_occ, delta_occ, nBaths, reort, dense_cutoff, blocked, fit_unocc, weight, spin_flip_dj


def exp_weight(w, w0, e):
    return np.exp(-e * np.abs(w - w0))


def gauss_weight(w, w0, e):
    return np.exp(-e / 2 * np.abs(w - w0) ** 2)


def get_weight_function(weight_function_name, w0, e):
    if weight_function_name == "Gaussian":
        return lambda w: gauss_weight(w, w0, e)
    elif weight_function_name == "Exponential":
        return lambda w: exp_weight(w, w0, e)
    elif weight_function_name == "RSPt":
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
    # slater_from_rspt = np.ndarray(buffer=ffi.buffer(rspt_slater, 4 * size_real), shape=(4,), dtype=float)

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

    # l = (n_orb // 2 - 1) // 2

    # slater = [0] * (2 * l + 1)
    # for i in range(l + 1):
    #     slater[2 * i] = slater_from_rspt[i]

    stdout_save = sys.stdout
    if rank == 0:
        sys.stdout = open(f"impurityModel-{label.strip()}{'-dc' if rspt_dc_flag == 1 else ''}.out", "w")
    elif verbosity > 0:
        sys.stdout = open(f"impurityModel-{label.strip()}{'-dc' if rspt_dc_flag == 1 else ''}-{rank}.out", "w")
    else:
        sys.stdout = open(devnull, "w")

    (
        nominal_occ,
        delta_occ,
        bath_states_per_orbital,
        reort,
        dense_cutoff,
        blocked,
        fit_unocc,
        weight,
        spin_flip_dj,
    ) = parse_solver_line(solver_line)
    nominal_occ = ({0: nominal_occ[0]}, {0: nominal_occ[1]}, {0: nominal_occ[2]})
    delta_occ = ({0: delta_occ[0]}, {0: delta_occ[1]}, {0: delta_occ[2]})
    if any(n0 > n_orb for n0 in nominal_occ[0].values()) or any(n0 < 0 for n0 in nominal_occ[0].values()):
        raise RuntimeError(f"Nominal impurity occupation {nominal_occ[0]} out of bounds [0, {n_orb}]")

    h_op, e_baths, v_star, H_star_bath = get_ed_h0(
        h_dft,
        hyb,
        corr_to_cf,
        bath_states_per_orbital,
        w,
        eim,
        gamma=0.001,
        exp_weight=weight,
        imag_only=False,
        valence_bath_only=not fit_unocc,
        label=label.strip(),
        save_baths_and_hopping=rspt_dc_flag == 1,
        verbose=(verbosity >= 0 or rspt_dc_flag == 1),
        comm=comm,
    )
    if rank == 0:
        with open(f"Ham-op-{label.strip()}.pickle", "wb") as f:
            pickle.dump(h_op, f)

    h_op = comm.bcast(h_op, root=0)
    e_baths = comm.bcast(e_baths, root=0)

    n_bath_states = ({0: len(e_baths[e_baths <= 0])}, {0: len(e_baths[e_baths > 0])})
    nominal_occ = (nominal_occ[0], {0: len(e_baths[e_baths <= 0])}, nominal_occ[2])

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
            impurity_orbitals={0: n_orb},
            bath_states=({0: sum(e_baths < 0)}, {0: sum(e_baths >= 0)}),
            u4=u4,
            peak_position=peak_position,
            dc_guess=sig_dc,
            spin_flip_dj=spin_flip_dj,
            tau=tau,
        )

        try:
            sig_dc[:, :] = fixed_peak_dc(h_op, dc_struct, rank=rank, verbose=verbosity > 0, dense_cutoff=dense_cutoff)
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
        h_star_bath=H_star_bath,
        v_star=v_star,
        u4=u4,
        n_bath_states=n_bath_states,
        nominal_occ=nominal_occ,
        delta_occ=delta_occ,
        sig=sig_python,
        sig_real=sig_real_python,
        sig_static=sig_static,
        sig_dc=sig_dc,
        corr_to_spherical=corr_to_spherical,
        corr_to_cf=corr_to_cf,
        blocked=blocked,
        spin_flip_dj=spin_flip_dj,
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
            reort=reort,
            dense_cutoff=dense_cutoff,
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
    imag_only=False,
    valence_bath_only=True,
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

    if comm.rank == 0:
        # Check to see if we have already done a fit
        try:
            with open(f"impurityModel_bath_energies_and_hopping_parameters_{label}.npy", "rb") as f:
                v = np.load(f)
                H_bath = np.load(f)
                v_star = np.load(f)
                H_star_bath = np.load(f)
            remove(f"impurityModel_bath_energies_and_hopping_parameters_{label}.npy")
        except FileNotFoundError:
            v = None
            H_bath = None
            v_star = None
            H_star_bath = None
        except ValueError:
            remove(f"impurityModel_bath_energies_and_hopping_parameters_{label}.npy")
            H_bath = None
            v = None
            H_star_bath = None
            v_star = None
    H_bath = comm.bcast(H_bath if comm.rank == 0 else None)
    v = comm.bcast(v if comm.rank == 0 else None)
    H_star_bath = comm.bcast(H_bath if comm.rank == 0 else None)
    v_star = comm.bcast(v if comm.rank == 0 else None)
    if H_bath is not None and verbose:
        print(
            f"Read bath energies and hopping parameters from impurityModel_bath_energies_and_hopping_parameters_{label}.npy"
        )
        eb = np.diag(H_bath)

    # We haven't already done a fit, so we do one now
    if H_bath is None and v is None:
        v, H_bath, v_star, H_star_bath = fit_hyb(
            w,
            eim,
            hyb,
            bath_states_per_orbital,
            gamma=gamma,
            imag_only=imag_only,
            x_lim=(w[0], 0 if valence_bath_only else w[-1]),
            verbose=verbose,
            comm=comm,
            weight_fun=get_weight_function("Gaussian", 0, exp_weight),
        )
        eb = np.diag(H_bath)
    if verbose:
        fitted_hyb = np.moveaxis(offdiagonal.get_hyb(w + eim * 1j, np.diag(H_star_bath), v_star), -1, 0)
        save_Greens_function(rotate_Greens_function(fitted_hyb, np.conj(corr_to_cf.T)), w, f"{label}-hyb-fit")

    if save_baths_and_hopping:
        if comm is not None and comm.rank == 0:
            with open(f"impurityModel_bath_energies_and_hopping_parameters_{label}.npy", "wb") as f:
                np.save(f, v)
                np.save(f, H_bath)
                np.save(f, v_star)
                np.save(f, H_star_bath)

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
    H[:n_orb, n_orb:] = np.conj(v.T)
    H[n_orb:, :n_orb] = v

    if verbose:
        H_tmp = np.zeros((n_orb + H_star_bath.shape[0], n_orb + H_star_bath.shape[0]), dtype=complex)
        H_tmp[:n_orb, :n_orb] = H_dft
        H_tmp[n_orb:, n_orb:] = H_star_bath
        H_tmp[:n_orb, n_orb:] = np.conj(v_star.T)
        H_tmp[n_orb:, :n_orb] = v_star
        u = np.identity(H_tmp.shape[0], dtype=complex)
        u[:n_orb, :n_orb] = np.conj(corr_to_cf.T)
        H_tmp = rotate_matrix(H_tmp, u)  # np.conj(u.T) @ H @ u
        print("DFT (star) hamiltonian, with baths")
        matrix_print(H_tmp)
        with open(f"Ham-{label}{'-dc' if save_baths_and_hopping else ''}.inp", "w") as f:
            for i in range(H_tmp.shape[0]):
                for j in range(H_tmp.shape[1]):
                    f.write(f" 0 0 0 {i+1} {j+1} {np.real(H_tmp[i, j])} {np.imag(H_tmp[i, j])}\n")

    h_op = finite.matrixToIOp(H)
    return h_op, eb, v_star, H_star_bath


def fit_hyb(
    w,
    delta,
    hyb,
    bath_states_per_orbital,
    gamma,
    imag_only,
    x_lim=None,
    tol=1e-6,
    verbose=True,
    comm=None,
    weight_fun=lambda w, w0, e: np.ones_like(w),
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
        return np.empty((0,), dtype=float), np.empty((0, hyb.shape[1]), dtype=complex)
    if x_lim is not None:
        mask = np.logical_and(w >= x_lim[0], w <= x_lim[1])
    else:
        mask = np.array([True] * len(w))

    # We do the fitting by first transforming the hyridization function into a basis
    # where each block is (hopefully) close to diagonal
    # np.conj(Q.T) @ cf_hyb @ Q is the transformation performed
    phase_hyb, Q = block_diagonalize_hyb(hyb)

    phase_blocks = get_block_structure(phase_hyb, tol=tol)
    phase_identical_blocks = get_identical_blocks(phase_blocks, phase_hyb, tol=tol)
    phase_transposed_blocks = get_transposed_blocks(phase_blocks, phase_hyb, tol=tol)
    phase_particle_hole_blocks = get_particle_hole_blocks(phase_blocks, phase_hyb, tol=tol)
    phase_particle_hole_and_transpose_blocks = get_particle_hole_and_transpose_blocks(phase_blocks, phase_hyb, tol=tol)

    if verbose:
        print(f"block structure : {phase_blocks}")
        print(f"identical blocks : {phase_identical_blocks}")
        print(f"transposed blocks : {phase_transposed_blocks}")
        print(f"particle hole blocks : {phase_particle_hole_blocks}")
        print(f"particle hole and transpose blocks : { phase_particle_hole_and_transpose_blocks}")

    n_orb = sum(len(block) for block in phase_blocks)

    H_baths = [np.empty((0, 0), dtype=complex)] * len(phase_blocks)
    vs = [np.empty((0, n_orb), dtype=complex)] * len(phase_blocks)
    H_star_baths = [np.empty((0, 0), dtype=complex)] * len(phase_blocks)
    vs_star = [np.empty((0, n_orb), dtype=complex)] * len(phase_blocks)
    inequivalent_blocks = get_inequivalent_blocks(
        phase_identical_blocks,
        phase_transposed_blocks,
        phase_particle_hole_blocks,
        phase_particle_hole_and_transpose_blocks,
    )
    if verbose:
        print(f"inequivalent blocks = {inequivalent_blocks}")
    states_per_inequivalent_block = get_state_per_inequivalent_block(
        inequivalent_blocks,
        phase_identical_blocks,
        phase_transposed_blocks,
        phase_particle_hole_blocks,
        phase_particle_hole_and_transpose_blocks,
        phase_blocks,
        bath_states_per_orbital,
        phase_hyb[mask, :, :],
        w[mask],
        weight_fun,
    )

    #### Do the fit, and copy to symmetrically equivalent blocks
    for inequivalent_block_i, block_i in enumerate(inequivalent_blocks):
        if states_per_inequivalent_block[inequivalent_block_i] == 0:
            continue
        block = phase_blocks[block_i]
        idx = np.ix_(range(phase_hyb.shape[0]), block, block)
        block_hyb = phase_hyb[idx]
        realvalue_v = np.all(np.abs(block_hyb - np.transpose(block_hyb, (0, 2, 1))) < 1e-6)
        block_star_eb, block_star_v = hf.fit_block(
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
        )
        n_block_orb = len(block)
        ##### Remove states with negligleble hopping
        bath_mask = []
        for block_i in range(0, len(block_star_v), n_block_orb):
            if np.all(np.abs(block_star_v[block_i : block_i + n_block_orb]) ** 2 < 1e-10):
                bath_mask.extend([False] * n_block_orb)
            else:
                bath_mask.extend([True] * n_block_orb)
        block_star_v = block_star_v[bath_mask]
        block_star_eb = block_star_eb[bath_mask]
        copy_to_symmetrically_equivalent_blocks(
            np.diag(block_star_eb),
            block_star_v,
            H_star_baths,
            vs_star,
            n_orb,
            phase_identical_blocks[inequivalent_block_i],
            phase_transposed_blocks[inequivalent_block_i],
            phase_particle_hole_blocks[inequivalent_block_i],
            phase_particle_hole_and_transpose_blocks[inequivalent_block_i],
            phase_blocks,
        )
        if False:
            H_bath_block = np.diag(block_star_eb)
            block_v = block_star_v
        else:
            block_v, H_bath_block = edchains(block_star_v, block_star_eb)

        if verbose:
            print(f"--> eb {np.diag(H_bath_block)}")
            print(f"--> v  {block_v}", flush=True)

        #### Copy to symmetrically equivalent blocks
        copy_to_symmetrically_equivalent_blocks(
            H_bath_block,
            block_v,
            H_baths,
            vs,
            n_orb,
            phase_identical_blocks[inequivalent_block_i],
            phase_transposed_blocks[inequivalent_block_i],
            phase_particle_hole_blocks[inequivalent_block_i],
            phase_particle_hole_and_transpose_blocks[inequivalent_block_i],
            phase_blocks,
        )

    print(f"{H_baths=}")
    H_bath = sp.linalg.block_diag(*H_baths)
    H_star_bath = sp.linalg.block_diag(*H_star_baths)
    print(f"{H_bath=}")
    # eb = np.concatenate(ebs, axis=0)
    v = np.vstack(vs)
    v_star = np.vstack(vs_star)

    # Transform hopping parameters back from the (close to) diagonal
    # basis to the original basis
    v = v @ np.conj(Q.T)
    v_star = v_star @ np.conj(Q.T)
    # Sort bath states, it is important for impurityModel that all unoccupied states come after the occupied states
    sort_indices = np.argsort(np.diag(H_bath), kind="stable")
    # sort_indices = np.argsort(eb, kind="stable")
    # eb = eb[sort_indices]
    H_bath = H_bath[np.ix_(sort_indices, sort_indices)]
    v = v[sort_indices]
    print(f"{H_bath=}")
    print(f"{np.diag(H_bath)=}")

    return v, H_bath, v_star, H_star_bath


def get_inequivalent_blocks(
    identical_blocks,
    transposed_blocks,
    particle_hole_blocks,
    particle_hole_and_transpose_blocks,
):
    inequivalent_blocks = []
    for blocks in identical_blocks:
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
    inequivalent_blocks,
    identical_blocks,
    transposed_blocks,
    particle_hole_blocks,
    particle_hole_and_transpose_blocks,
    blocks,
    bath_states_per_orbital,
    hyb,
    w,
    weight_fun,
):
    orbitals_per_inequivalent_block = [0] * len(inequivalent_blocks)
    weight_per_inequivalent_block = np.zeros((len(inequivalent_blocks)), dtype=float)
    for inequivalent_block_i, block_i in enumerate(inequivalent_blocks):
        block = blocks[block_i]
        block_multiplicity = (
            len(identical_blocks[inequivalent_block_i])
            + len(transposed_blocks[inequivalent_block_i])
            + len(particle_hole_blocks[inequivalent_block_i])
            + len(particle_hole_and_transpose_blocks[inequivalent_block_i])
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


def copy_to_symmetrically_equivalent_blocks(
    H_bath,
    v,
    H_baths,
    vs,
    n_orb,
    identical_blocks,
    transposed_blocks,
    particle_hole_blocks,
    particle_hole_and_transpose_blocks,
    blocks,
):
    n_block_orb = v.shape[1]
    n_eb = H_bath.shape[0]
    for b in identical_blocks:
        H_baths[b] = H_bath
        for i_orb, orb in enumerate(blocks[b]):
            # ebs[orb] = np.append(ebs[orb], [eb[i_orb::n_block_orb]])
            v_tmp = np.zeros((n_eb // n_block_orb, n_orb), dtype=complex)
            v_tmp[:, blocks[b]] = v[i_orb::n_block_orb, :]
            vs[orb] = np.append(vs[orb], v_tmp, axis=0)
    for b in transposed_blocks:
        H_baths[b] = H_bath.T
        for i_orb, orb in enumerate(blocks[b]):
            # ebs[orb] = np.append(ebs[orb], [eb[i_orb::n_block_orb]])
            v_tmp = np.zeros((n_eb // n_block_orb, n_orb), dtype=complex)
            v_tmp[:, blocks[b]] = np.conj(v[i_orb::n_block_orb, :])
            vs[orb] = np.append(vs[orb], v_tmp, axis=0)
    for b in particle_hole_blocks:
        H_baths[b] = H_bath @ (-np.identity(H_bath.shape[0]))
        for i_orb, orb in enumerate(blocks[b]):
            # ebs[orb] = np.append(ebs[orb], [-eb[i_orb::n_block_orb]])
            v_tmp = np.zeros((n_eb // n_block_orb, n_orb), dtype=complex)
            v_tmp[:, blocks[b]] = v[i_orb::n_block_orb, :]
            vs[orb] = np.append(vs[orb], v_tmp, axis=0)
    for b in particle_hole_and_transpose_blocks:
        H_baths[b] = H_bath.T @ (-np.identity(H_bath.shape[0]))
        for i_orb, orb in enumerate(blocks[b]):
            # ebs[orb] = np.append(ebs[orb], [-eb[i_orb::n_block_orb]])
            v_tmp = np.zeros((n_eb // n_block_orb, n_orb), dtype=complex)
            v_tmp[:, blocks[b]] = np.conj(v[i_orb::n_block_orb, :])
            vs[orb] = np.append(vs[orb], v_tmp, axis=0)
