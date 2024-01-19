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
from impurityModel.ed.greens_function import save_Greens_function
from impurityModel.ed import finite
from impurityModel.ed.lanczos import Reort
from impurityModel.ed.greens_function import rotate_Greens_function, rotate_matrix, rotate_4index_U
from impurityModel.ed.manybody_basis import CIPSI_Basis


def matrix_print(matrix):
    print("\n".join([" ".join([f"{np.real(el): .6f} {np.imag(el):+.6f}j" for el in row]) for row in matrix]))


class ImpModCluster:
    def __init__(
        self,
        label,
        h_dft,
        hyb,
        u4,
        slater,
        nominal_occ,
        delta_occ,
        n_bath_states,
        sig,
        sig_real,
        sig_static,
        sig_dc,
        corr_to_cf,
        corr_to_spherical,
        blocked,
    ):
        self.label = label
        self.h_dft = h_dft
        self.u4 = u4
        self.hyb = hyb
        self.slater = slater
        self.bath_states = n_bath_states
        self.nominal_occ = nominal_occ
        self.delta_occ = delta_occ
        self.sig = sig
        self.sig_real = sig_real
        self.sig_static = sig_static
        self.sig_dc = sig_dc
        self.corr_to_cf = corr_to_cf
        self.corr_to_spherical = corr_to_spherical

        if blocked:
            self.blocks = hf.get_block_structure(
                self.hyb,
                h_dft,
            )
            self.identical_blocks = hf.get_identical_blocks(
                self.blocks,
                self.hyb,
                h_dft,
            )
            self.transposed_blocks = hf.get_transposed_blocks(
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
        self, nominal_occ, delta_occ, num_spin_orbitals, bath_states, u4, slater_params, peak_position, dc_guess
    ):
        self.nominal_occ = nominal_occ
        self.delta_occ = delta_occ
        self.num_spin_orbitals = num_spin_orbitals
        self.bath_states = bath_states
        self.u4 = u4
        self.slater_params = slater_params
        self.peak_position = peak_position
        self.dc_guess = dc_guess

    def __repr__(self):
        return (
            f"dcStruct( nominal_occ = {self.nominal_occ},\n"
            f"          delta_occ = {self.delta_occ},\n"
            f"          num_spin_orbitals = {self.num_spin_orbitals},\n"
            f"          bath_states = {self.bath_states},\n"
            f"          slater_params = {self.slater_params},\n"
            f"          peak_position = {self.peak_position})"
            f"          dc_guess = {self.dc_guess})"
        )


def parse_solver_line(solver_line):
    """
    N0 dN dVal dCon Nbath [[pro, full] [dense_cutoff 50] [no_block], [fit_unocc]]
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
            else:
                raise RuntimeError(f"Unknown solver parameter {arg}.\n" f"--->Other solver params {solver_array[5:]}")

    print(
        f"Nominal imp. occupation  +> {nominal_occ[0]}\n"
        f"Delta occupation         +> {delta_occ}\n"
        f"# bath states / imp. orb.+> {nBaths}\n"
        f"Reorthogonalizaion mode  +> {reort}\n"
        f"Dense matrix size cutoff +> {dense_cutoff}\n"
        f"Use block structure      +> {blocked}\n"
        f"Fit unoccupied states    +> {fit_unocc}\n"
    )
    return nominal_occ, delta_occ, nBaths, reort, dense_cutoff, blocked, fit_unocc


@ffi.def_extern()
def run_impmod_ed(
    rspt_label,
    rspt_solver_line,
    rspt_dc_line,
    rspt_dc_flag,
    rspt_slater,
    rspt_u4,
    rspt_hyb,
    rspt_h_dft,
    rspt_sig,
    rspt_sig_real,
    rspt_sig_static,
    rspt_sig_dc,
    rspt_iw,
    rspt_w,
    rspt_corr_to_cf,
    rspt_rot_spherical,
    n_orb,
    n_iw,
    n_w,
    n_rot_cols,
    eim,
    tau,
    verbosity,
    size_real,
    size_complex,
):
    from mpi4py import MPI

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
    rspt_rot_spherical_arr = np.ndarray(
        buffer=ffi.buffer(rspt_rot_spherical, n_orb * n_rot_cols * size_complex),
        shape=(n_orb, n_rot_cols),
        order="F",
        dtype=complex,
    )
    rspt_corr_to_cf_arr = np.ndarray(
        buffer=ffi.buffer(rspt_corr_to_cf, n_orb * n_orb * size_complex),
        shape=(n_orb, n_rot_cols),
        order="F",
        dtype=complex,
    )
    slater_from_rspt = np.ndarray(buffer=ffi.buffer(rspt_slater, 4 * size_real), shape=(4,), dtype=float)

    if n_rot_cols == n_orb:
        corr_to_cf = rspt_corr_to_cf_arr
        corr_to_spherical = rspt_rot_spherical_arr
    else:
        corr_to_cf = np.empty((n_orb, n_orb), dtype=complex)
        corr_to_cf[:, :n_rot_cols] = rspt_corr_to_cf_arr
        corr_to_cf[:, n_rot_cols:] = np.roll(rspt_corr_to_cf_arr, n_rot_cols, axis=0)
        corr_to_spherical = np.empty((n_orb, n_orb), dtype=complex)
        corr_to_spherical[:, :n_rot_cols] = rspt_rot_spherical_arr
        corr_to_spherical[:, n_rot_cols:] = np.roll(rspt_rot_spherical_arr, n_rot_cols, axis=0)
    # Rotate the U-matrix to the CF basis
    u4 = rotate_4index_U(u4, corr_to_cf)
    # impurityModel uses a weird convention for the U-matrix
    u4 = np.moveaxis(u4, 1, 0)
    # Rotate hybridization function and DFT hamiltonian to the CF basis
    hyb = rotate_Greens_function(hyb, corr_to_cf)
    h_dft = rotate_matrix(h_dft, corr_to_cf)

    l = (n_orb // 2 - 1) // 2

    slater = [0] * (2 * l + 1)
    for i in range(l + 1):
        slater[2 * i] = slater_from_rspt[i]

    stdout_save = sys.stdout
    if rank == 0:
        sys.stdout = open(f"impurityModel-{label.strip()}{'-dc' if rspt_dc_flag == 1 else ''}.out", "w")
    elif True or verbosity > 0:
        sys.stdout = open(f"impurityModel-{label.strip()}{'-dc' if rspt_dc_flag == 1 else ''}-{rank}.out", "w")
    else:
        sys.stdout = open(devnull, "w")

    nominal_occ, delta_occ, bath_states_per_orbital, reort, dense_cutoff, blocked, fit_unocc = parse_solver_line(
        solver_line
    )
    nominal_occ = ({l: nominal_occ[0]}, {l: nominal_occ[1]}, {l: nominal_occ[2]})
    delta_occ = ({l: delta_occ[0]}, {l: delta_occ[1]}, {l: delta_occ[2]})

    h_op, e_baths = get_ed_h0(
        h_dft,
        hyb,
        corr_to_cf,
        bath_states_per_orbital,
        w,
        eim,
        gamma=0.01,
        imag_only=False,
        valence_bath_only=not fit_unocc,
        label=label.strip(),
        save_baths_and_hopping=rspt_dc_flag == 1,
        verbose=(verbosity >= 2 or rspt_dc_flag == 1) and rank == 0,
        comm=comm,
    )
    if rank == 0:
        with open(f"Ham-op-{label.strip()}.pickle", "wb") as f:
            pickle.dump(h_op, f)

    h_op = comm.bcast(h_op, root=0)
    e_baths = comm.bcast(e_baths, root=0)

    n_bath_states = ({l: len(e_baths[e_baths <= 0])}, {l: len(e_baths[e_baths > 0])})
    nominal_occ = (nominal_occ[0], {l: len(e_baths[e_baths <= 0])}, nominal_occ[2])

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
            num_spin_orbitals=n_orb + len(e_baths),
            bath_states=({l: sum(e_baths < 0)}, {l: sum(e_baths >= 0)}),
            u4=u4,
            slater_params=slater,
            peak_position=peak_position,
            dc_guess=sig_dc[0, 0],
        )

        try:
            sig_dc[:, :] = fixed_peak_dc(h_op, dc_struct, rank=rank, verbose=rank == 0, dense_cutoff=dense_cutoff)
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
        slater=slater,
        n_bath_states=n_bath_states,
        nominal_occ=nominal_occ,
        delta_occ=delta_occ,
        sig=sig,
        sig_real=sig_real,
        sig_static=sig_static,
        sig_dc=sig_dc,
        corr_to_cf=corr_to_cf,
        corr_to_spherical=corr_to_spherical,
        blocked=blocked,
    )

    from impurityModel.ed import selfenergy

    try:
        selfenergy.run(
            cluster, h_op, 1j * iw, w, eim, tau, verbosity if rank == 0 else 0, reort=reort, dense_cutoff=dense_cutoff
        )

        # Rotate self energy from CF basis to RSPt's corr basis
        u = np.conj(corr_to_cf.T)
        cluster.sig[:, :, :] = rotate_Greens_function(cluster.sig, u)
        cluster.sig_real[:, :, :] = rotate_Greens_function(cluster.sig_real, u)
        cluster.sig_static[:, :] = rotate_matrix(cluster.sig_static, u)
        er = 0
    except Exception as e:
        print("!" * 100)
        print(f"Exception {repr(e)} caught on rank {rank}!")
        print(traceback.format_exc())
        print(
            "Adding positive infinity to the imaginaty part of the selfenergy at the last matsubara frequency.",
            flush=True,
        )
        print("!" * 100)
        cluster.sig[:, :, -1] += 1j * np.inf
        er = -1
        comm.Abort(er)
    else:
        if rank == 0:
            print("Self energy calculated! impurityModel shutting down.", flush=True)

    sys.stdout.close()
    sys.stdout = stdout_save
    return er


def fixed_peak_dc(h0_op, dc_struct, rank, verbose, dense_cutoff):
    N0 = dc_struct.nominal_occ
    delta_impurity_occ, delta_valence_occ, delta_conduction_occ = dc_struct.delta_occ
    peak_position = dc_struct.peak_position
    num_valence_bath_states, num_conduction_bath_states = dc_struct.bath_states
    sum_bath_states = {l: num_valence_bath_states[l] + num_conduction_bath_states[l] for l in num_valence_bath_states}
    l = list(lv for lv in N0[0])[0]
    u = finite.getUop_from_rspt_u4(dc_struct.u4)

    Np = ({l: N0[0][l] + 1 for l in N0[0]}, N0[1], N0[2])
    Nm = ({l: N0[0][l] - 1 for l in N0[0]}, N0[1], N0[2])
    if peak_position >= 0:
        basis_upper = CIPSI_Basis(
            ls=[l for l in N0[0]],
            valence_baths=num_valence_bath_states,
            conduction_baths=num_conduction_bath_states,
            delta_valence_occ=delta_valence_occ,
            delta_conduction_occ=delta_conduction_occ,
            delta_impurity_occ=delta_impurity_occ,
            nominal_impurity_occ=Np[0],
            verbose=False and verbose,
            comm=MPI.COMM_WORLD,
        )
        basis_lower = CIPSI_Basis(
            ls=[l for l in N0[0]],
            valence_baths=num_valence_bath_states,
            conduction_baths=num_conduction_bath_states,
            delta_valence_occ=delta_valence_occ,
            delta_conduction_occ=delta_conduction_occ,
            delta_impurity_occ=delta_impurity_occ,
            nominal_impurity_occ=N0[0],
            verbose=False and verbose,
            comm=MPI.COMM_WORLD,
        )
    else:
        basis_upper = CIPSI_Basis(
            ls=[l for l in N0[0]],
            valence_baths=num_valence_bath_states,
            conduction_baths=num_conduction_bath_states,
            delta_valence_occ=delta_valence_occ,
            delta_conduction_occ=delta_conduction_occ,
            delta_impurity_occ=delta_impurity_occ,
            nominal_impurity_occ=N0[0],
            verbose=False and verbose,
            comm=MPI.COMM_WORLD,
        )
        basis_lower = CIPSI_Basis(
            ls=[l for l in N0[0]],
            valence_baths=num_valence_bath_states,
            conduction_baths=num_conduction_bath_states,
            delta_valence_occ=delta_valence_occ,
            delta_conduction_occ=delta_conduction_occ,
            delta_impurity_occ=delta_impurity_occ,
            nominal_impurity_occ=Nm[0],
            verbose=False and verbose,
            comm=MPI.COMM_WORLD,
        )


    def F(dc_trial):
        bu = basis_upper.copy()
        bl = basis_lower.copy()
        dc_op = {(((l, s, m), "c"), ((l, s, m), "a")): -dc_trial for m in range(-l, l + 1) for s in range(2)}
        h_op_c = finite.addOps([h0_op, u, dc_op])
        h_op_i = finite.c2i_op(sum_bath_states, h_op_c)
        h_dict = bu.expand(h_op_i, dense_cutoff=dense_cutoff, de2_min=1e-5, slaterWeightMin=0)
        h = (
            basis_upper.build_sparse_matrix(h_op_i, h_dict)
            if basis_upper.size > dense_cutoff
            else basis_upper.build_dense_matrix(h_op_i, h_dict)
        )
        e_upper = finite.eigensystem_new(
            h,
            0,
            k=1,
            eigenValueTol=0,
            verbose=verbose,
            dense_cutoff=dense_cutoff,
            return_eigvecs=False,
        )
        h_dict = bl.expand(h_op_i, dense_cutoff=dense_cutoff, de2_min=1e-5, slaterWeightMin=0)
        h = (
            basis_lower.build_sparse_matrix(h_op_i, h_dict)
            if basis_lower.size > dense_cutoff
            else basis_lower.build_dense_matrix(h_op_i, h_dict)
        )
        e_lower = finite.eigensystem_new(
            h,
            0,
            k=1,
            eigenValueTol=0,
            verbose=verbose,
            dense_cutoff=dense_cutoff,
            return_eigvecs=False,
        )
        return e_upper[0] - e_lower[0] - peak_position

    res = sp.optimize.root_scalar(F, x0=dc_struct.dc_guess, x1=dc_struct.dc_guess + F(dc_struct.dc_guess))
    dc = res.root
    if verbose:
        print(f"dc found : {dc}")

    return dc * np.identity(2 * (2 * l + 1), dtype=complex)


def get_ed_h0(
    h_dft,
    hyb,
    corr_to_cf,
    bath_states_per_orbital,
    w,
    eim,
    gamma=0.001,
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
        with open(f"hyb-in-{label}.npy", "wb") as f:
            np.save(f, hyb)

    if comm.rank == 0:
        try:
            with open(f"impurityModel_bath_energies_and_hopping_parameters_{label}.npy", "rb") as f:
                eb = np.load(f)
                v = np.load(f)
            remove(f"impurityModel_bath_energies_and_hopping_parameters_{label}.npy")
        except FileNotFoundError:
            eb = None
            v = None
    eb = comm.bcast(eb if comm.rank == 0 else None)
    v = comm.bcast(v if comm.rank == 0 else None)
    if eb is not None and verbose:
        print(
            f"Read bath energies and hopping parameters from impurityModel_bath_energies_and_hopping_parameters_{label}.npy"
        )
    if eb is None and v is None and bath_states_per_orbital > 0:
        eb, v = hf.fit_hyb(
            w,
            eim,
            hyb,
            bath_states_per_orbital,
            gamma=gamma,
            imag_only=imag_only,
            x_lim=(w[0], 0 if valence_bath_only else w[-1]),
            verbose=verbose,
            comm=comm,
            new_v=True,
        )
        sort_indices = np.argsort(eb, kind="stable")
        eb = eb[sort_indices]
        v = v[sort_indices]

    if save_baths_and_hopping:
        if comm is not None and comm.rank == 0:
            with open(f"impurityModel_bath_energies_and_hopping_parameters_{label}.npy", "wb") as f:
                np.save(f, eb)
                np.save(f, v)
    if verbose:
        fit_hyb = offdiagonal.get_hyb(w + eim * 1j, eb, v)
        save_Greens_function(rotate_Greens_function(fit_hyb, np.conj(corr_to_cf.T)), w, f"{label}-hyb-fit")
        with open(f"{label}-hyb-fit.npy", "wb") as f:
            fit_hyb = offdiagonal.get_hyb(w + eim * 1j, eb, v)
            np.save(f, fit_hyb)

    if verbose:
        print("DFT hamiltonian in correlated basis")
        matrix_print(rotate_matrix(h_dft, np.conj(corr_to_cf.T)))
        print("DFT hamiltonian in CF basis")
        matrix_print(h_dft)
        print("Hopping parameters in CF basis")
        matrix_print(v)
        print("Bath state energies")
        print(np.array_str(eb, max_line_width=1000, precision=4, suppress_small=False))

    n_orb = v.shape[1]
    h = np.zeros((n_orb + len(eb), n_orb + len(eb)), dtype=complex)
    h[:n_orb, :n_orb] = h_dft
    h[:n_orb, n_orb:] = np.conj(v.T)
    h[n_orb:, :n_orb] = v
    np.fill_diagonal(h[n_orb:, n_orb:], eb)

    if verbose:
        u = np.identity(h.shape[0], dtype=complex)
        u[:n_orb, :n_orb] = np.conj(corr_to_cf.T)
        h_tmp = rotate_matrix(h, u)  # np.conj(u.T) @ h @ u
        print("DFT hamiltonian, with baths")
        matrix_print(h_tmp)
        with open(f"Ham-{label}{'-dc' if save_baths_and_hopping else ''}.inp", "w") as f:
            for i in range(h_tmp.shape[0]):
                for j in range(h_tmp.shape[1]):
                    f.write(f" 0 0 0 {i+1} {j+1} {np.real(h_tmp[i, j])} {np.imag(h_tmp[i, j])}\n")

    h_op = h2imp.get_H_operator_from_dense_rspt_H_matrix(h, ang=(n_orb // 2 - 1) // 2)
    return h_op, eb
