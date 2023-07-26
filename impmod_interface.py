from impmod_ed import ffi
import numpy as np
import scipy as sp
import traceback
import sys
import pickle
from impurityModel.ed.lanczos import Reort



def matrix_print(matrix):
    print("\n".join([" ".join([f"{np.real(el): .6f} {np.imag(el):+.6f}j" for el in row]) for row in matrix]))


class impModCluster:
    def __init__(
        self,
        label,
        h_dft,
        hyb,
        slater,
        nominal_occ,
        delta_occ,
        n_bath_states,
        blocks,
        sig,
        sig_real,
        sig_static,
        sig_dc,
        rot_spherical,
    ):
        self.label = label
        self.h_dft = h_dft
        self.hyb = hyb
        self.slater = slater
        self.bath_states = n_bath_states
        self.nominal_occ = nominal_occ
        self.delta_occ = delta_occ
        self.blocks = blocks
        self.sig = sig
        self.sig_real = sig_real
        self.sig_static = sig_static
        self.sig_dc = sig_dc
        self.rot_spherical = rot_spherical


class dcStruct:
    def __init__(self, nominal_occ, delta_occ, num_spin_orbitals, bath_states, slater_params, peak_position):
        self.nominal_occ = nominal_occ
        self.delta_occ = delta_occ
        self.num_spin_orbitals = num_spin_orbitals
        self.bath_states = bath_states
        self.slater_params = slater_params
        self.peak_position = peak_position

    def __repr__(self):
        return (
            f"dcStruct( nominal_occ = {self.nominal_occ},\n"
            f"          delta_occ = {self.delta_occ},\n"
            f"          num_spin_orbitals = {self.num_spin_orbitals},\n"
            f"          bath_states = {self.bath_states},\n"
            f"          slater_params = {self.slater_params},\n"
            f"          peak_position = {self.peak_position})"
        )


def parse_solver_line(solver_line):
    """
    N0 dN dVal dCon Nbath [pro] [dense_cutoff 5000]
    """
    # Remove comments from the solver line
    solver_line = solver_line.split("!")[0]
    solver_line = solver_line.split("#")[0]
    solver_array = solver_line.strip().split()
    assert len(solver_array) >= 5, "The impurityModel ED solver requires at least 5 arguments; N0 dN dValence dConduction nBaths"
    try:
        nominal_occ = (int(solver_array[0]), 0, 0)
        delta_occ = (int(solver_array[1]), int(solver_array[2]), int(solver_array[3]))
        nBaths = int(solver_array[4])
    except Exception as e:
        raise RuntimeError(f"{e}\n"
                           f"--->N0 {solver_array[0]}\n"
                           f"--->dN {solver_array[1]}\n"
                           f"--->dValence {solver_array[2]}\n"
                           f"--->dConduction {solver_array[3]}\n"
                           f"--->Nbaths {solver_array[4]}")
    partial_reort = False
    dense_cutoff = 5000
    reort = Reort.NONE
    if len(solver_array) > 5:
        skip_next = False
        for i in range(len(solver_array)):
            if i < 5:
                continue
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
    return nominal_occ, delta_occ, nBaths, reort, dense_cutoff

@ffi.def_extern()
def run_impmod_ed(
    rspt_label,
    rspt_solver_line,
    rspt_dc_line,
    rspt_dc_flag,
    rspt_slater,
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
    comm.barrier()

    label = ffi.string(rspt_label, 18).decode("ascii")
    solver_line = ffi.string(rspt_solver_line, 100).decode("ascii")

    h_dft = np.ndarray(
        buffer=ffi.buffer(rspt_h_dft, n_orb * n_orb * size_complex), shape=(n_orb, n_orb), order="F", dtype=complex
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
        rot_spherical = rspt_rot_spherical_arr
        corr_to_cf = rspt_corr_to_cf_arr
    else:
        rot_spherical = np.empty((n_orb, n_orb), dtype=complex)
        rot_spherical[:, :n_rot_cols] = rspt_rot_spherical_arr
        rot_spherical[:, n_rot_cols:] = np.roll(rspt_rot_spherical_arr, n_rot_cols, axis=0)
        corr_to_cf = np.empty((n_orb, n_orb), dtype=complex)
        corr_to_cf[:, :n_rot_cols] = rspt_corr_to_cf_arr
        corr_to_cf[:, n_rot_cols:] = np.roll(rspt_corr_to_cf_arr, n_rot_cols, axis=0)

    l = (n_orb // 2 - 1) // 2

    slater = [0] * (2 * l + 1)
    for i in range(l + 1):
        slater[2 * i] = slater_from_rspt[i]

    stdout_save = sys.stdout
    # sys.stdout = open(f"impurityModel-{label.strip()}-{rank}.out", "a")

    nominal_occ, delta_occ, bath_states_per_orbital, reort, dense_cutoff = parse_solver_line(solver_line)
    nominal_occ = ({l: nominal_occ[0]}, {l: nominal_occ[1]}, {l:nominal_occ[2]})
    delta_occ = ({l: delta_occ[0]}, {l: delta_occ[1]}, {l:delta_occ[2]})

    h_op, e_baths = get_ed_h0(
        h_dft,
        hyb,
        corr_to_cf,
        rot_spherical,
        bath_states_per_orbital,
        w,
        eim,
        gamma=0.01,
        imag_only=False,
        valence_bath_only = delta_occ[2][l] == 0,
        label = label.strip(),
        save_baths_and_hopping = rspt_dc_flag == 1,
        verbose = verbosity >= 2 and rank == 0,
        comm = comm,
    )
    if rank == 0:
        with open(f"Ham-op-{label.strip()}.pickle", 'wb') as f:
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
            slater_params=slater,
            peak_position=peak_position,
        )

        try:
            sig_dc[:, :] = fixed_peak_dc(h_op, dc_struct, rank=rank, verbose  = rank == 0 and verbosity >= 1, dense_cutoff = dense_cutoff)
        except Exception as e:
            print(f"!"*100)
            print(f"Exception {repr(e)} caught on rank {rank}!")
            print (traceback.format_exc())
            print (f"Adding positive infinity to the imaginaty part of the DC selfenergy.", flush = True)
            print(f"!"*100)
            sig_dc[:, :] = 1j*np.inf

        # sys.stdout.close()
        sys.stdout = stdout_save
        return

    from rspt2spectra.hyb_fit import get_block_structure

    cluster = impModCluster(
        label=label.strip(),
        h_dft=h_dft,
        hyb=hyb,
        slater=slater,
        n_bath_states=n_bath_states,
        nominal_occ=nominal_occ,
        delta_occ=delta_occ,
        blocks=get_block_structure( np.moveaxis(np.conj(rot_spherical.T)[np.newaxis, :, :] @ 
                                                np.moveaxis(hyb, -1, 0) @ 
                                                rot_spherical[np.newaxis, :, :], 0, -1),
                                   np.conj(rot_spherical.T) @ h_dft @ rot_spherical
                                   ),
        sig=sig,
        sig_real=sig_real,
        sig_static=sig_static,
        sig_dc=sig_dc,
        rot_spherical=rot_spherical,
    )

    from impurityModel.ed import selfenergy

    # Python exceptions are ignored by the outside code, so we need to capture all exceptions,
    # print an error message and make sure that the outside code becomes aware that something
    # has gone terribly wrong.
    try:
        selfenergy.run(cluster, h_op, 1j * iw, w, eim, tau, verbosity if rank == 0 else 0, reort = reort, dense_cutoff = dense_cutoff)

        # Rotate self energy from spherical harmonics basis to RSPt's corr basis
        u = cluster.rot_spherical
        cluster.sig[:, :, :] = np.moveaxis(
            u[np.newaxis, :, :] @ np.moveaxis(cluster.sig, -1, 0) @ np.conj(u.T)[np.newaxis, :, :], 0, -1
        )
        cluster.sig_real[:, :, :] = np.moveaxis(
            u[np.newaxis, :, :] @ np.moveaxis(cluster.sig_real, -1, 0) @ np.conj(u.T)[np.newaxis, :, :], 0, -1
        )
        cluster.sig_static[:, :] = u @ cluster.sig_static @ np.conj(u.T)
    except Exception as e:
        print(f"!"*100)
        print(f"Exception {repr(e)} caught on rank {rank}!")
        print (traceback.format_exc())
        print (f"Adding positive infinity to the imaginaty part of the selfenergy at the last matsubara frequency.", flush = True)
        print(f"!"*100)
        cluster.sig[:,:, -1] += 1j*np.inf
    else:
        if rank == 0:
            print (f"Self energy calculated! impurityModel shutting down.", flush = True)

    sys.stdout.close()
    sys.stdout = stdout_save
    return

def symmetrize_sigma(sigma, blocks, equivalent_blocks):
    symmetrized_sigma = np.zeros_like(sigma)
    for equivalent_block in equivalent_blocks:
        for block_i in equivalent_block:
            block_idx = np.ix_(blocks[block_i], blocks[block_i])
            for block in equivalent_block:
                idx = np.ix_(blocks[block], blocks[block])
                symmetrized_sigma[idx] += sigma[block_idx] / len(equivalent_block)
    return symmetrized_sigma


def fixed_peak_dc(h0_op, dc_struct, rank, verbose, dense_cutoff):
    from impurityModel.ed import finite
    from rspt2spectra import h2imp
    from impurityModel.ed.manybody_basis import Basis, CIPSI_Basis
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    N0 = dc_struct.nominal_occ
    delta_impurity_occ, delta_valence_occ, delta_conduction_occ = dc_struct.delta_occ
    peak_position = dc_struct.peak_position
    num_spin_orbitals = dc_struct.num_spin_orbitals
    num_valence_bath_states, num_conduction_bath_states = dc_struct.bath_states
    sum_bath_states = {l: num_valence_bath_states[l] + num_conduction_bath_states[l] for l in num_valence_bath_states}
    l = [lv for lv in N0[0]][0]
    u = finite.getUop(l, l, l, l, dc_struct.slater_params)

    Np = ({l: N0[0][l] + 1 for l in N0[0]}, N0[1], N0[2])
    Nm = ({l: N0[0][l] - 1 for l in N0[0]}, N0[1], N0[2])
    if peak_position >= 0:
        basis_upper = CIPSI_Basis(
                valence_baths        = num_valence_bath_states,
                conduction_baths     = num_conduction_bath_states,
                delta_valence_occ    = delta_valence_occ,
                delta_conduction_occ = delta_conduction_occ,
                delta_impurity_occ   = delta_impurity_occ,
                nominal_impurity_occ = Np[0],
                verbose = False,
                comm = MPI.COMM_WORLD
                )
        basis_lower = CIPSI_Basis(
                valence_baths        = num_valence_bath_states,
                conduction_baths     = num_conduction_bath_states,
                delta_valence_occ    = delta_valence_occ,
                delta_conduction_occ = delta_conduction_occ,
                delta_impurity_occ   = delta_impurity_occ,
                nominal_impurity_occ = N0[0],
                verbose = False,
                comm = MPI.COMM_WORLD
                )
    else:
        basis_upper = Basis(
                valence_baths        = num_valence_bath_states,
                conduction_baths     = num_conduction_bath_states,
                delta_valence_occ    = delta_valence_occ,
                delta_conduction_occ = delta_conduction_occ,
                delta_impurity_occ   = delta_impurity_occ,
                nominal_impurity_occ = N0[0],
                verbose = False,
                comm = MPI.COMM_WORLD
                )
        basis_lower = Basis(
                valence_baths        = num_valence_bath_states,
                conduction_baths     = num_conduction_bath_states,
                delta_valence_occ    = delta_valence_occ,
                delta_conduction_occ = delta_conduction_occ,
                delta_impurity_occ   = delta_impurity_occ,
                nominal_impurity_occ = Nm[0],
                verbose = False,
                comm = MPI.COMM_WORLD
                )
    def F(dc_trial):
        dc_op = {(((l, s, m), "c"), ((l, s, m), "a")): -dc_trial for m in range(-l, l + 1) for s in range(2)}
        h_op_c = finite.addOps([h0_op, u, dc_op])
        h_op_i = finite.c2i_op(sum_bath_states, h_op_c)
        h_dict = basis_upper.expand(h_op_i, dense_cutoff = dense_cutoff)
        h_sparse = basis_upper.build_sparse_operator(h_op_i, h_dict)
        e_upper, _ = finite.eigensystem_new(h_sparse, basis_upper, 0, k = 1, dk = 2, eigenValueTol = 0, verbose = False, dense_cutoff = dense_cutoff)
        h_dict = basis_lower.expand(h_op_i, dense_cutoff = dense_cutoff)
        h_sparse = basis_lower.build_sparse_operator(h_op_i, h_dict)
        e_lower, _ = finite.eigensystem_new(h_sparse, basis_lower, 0, k = 1, dk = 2, eigenValueTol = 0, verbose = False, dense_cutoff = dense_cutoff)
        return e_upper[0] - e_lower[0] - peak_position

    res = sp.optimize.root_scalar(F, x0 = 0, x1 = F(0))
    dc = res.root
    if verbose:
        print(f"dc found : {dc}")

    return dc * np.identity(2 * (2 * l + 1), dtype=complex)


def get_ed_h0(
    h_dft,
    hyb,
    corr_to_cf,
    rot_spherical,
    bath_states_per_orbital,
    w,
    eim,
    gamma=0.001,
    imag_only=False,
    valence_bath_only=True,
    label = None,
    save_baths_and_hopping = False,
    verbose = True,
    comm = None,
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
    from rspt2spectra import orbitals
    from rspt2spectra import offdiagonal
    from rspt2spectra.hyb_fit import fit_hyb
    from rspt2spectra import energies
    from rspt2spectra import h2imp
    from impurityModel.ed.greens_function import save_Greens_function

    if comm.rank == 0:
        with open(f"hyb-in-{label}.npy", 'wb') as f:
            np.save(f, hyb)
    if save_baths_and_hopping:
        eb, v = fit_hyb(
            w,
            eim,
            hyb,
            corr_to_cf,
            rot_spherical,
            bath_states_per_orbital,
            gamma=gamma,
            imag_only=imag_only,
            x_lim=(w[0], 0 if valence_bath_only else w[-1]),
            verbose = verbose,
            comm = comm,
            new_v = True,
        )
        if comm is not None and comm.rank == 0:
            with open(f"impurityModel_bath_energies_and_hopping_parameters_{label}.npy", 'wb') as f:
                np.save(f, eb)
                np.save(f, v)
    else:
        try:
            with open(f"impurityModel_bath_energies_and_hopping_parameters_{label}.npy", 'rb') as f:
                eb = np.load(f)
                v = np.load(f)
            if verbose:
                print (f"Read bath energies and hopping parameters from impurityModel_bath_energies_and_hopping_parameters_{label}.npy")
        except:
            eb, v = fit_hyb(
                w,
                eim,
                hyb,
                corr_to_cf,
                rot_spherical,
                bath_states_per_orbital,
                gamma=gamma,
                imag_only=imag_only,
                x_lim=(w[0], 0 if valence_bath_only else w[-1]),
                verbose = verbose,
                comm = comm,
                new_v = True,
            )
    if verbose:
        fit_hyb = offdiagonal.get_hyb(w + eim * 1j, eb, v)
        save_Greens_function(fit_hyb, w, f"{label}-hyb-fit")
        with open(f"{label}-hyb-fit.npy", 'wb') as f:
            fit_hyb = offdiagonal.get_hyb(w + eim * 1j, eb, v @ np.conj(rot_spherical.T))
            np.save(f, fit_hyb)

    if verbose:
        print(f"DFT hamiltonian")
        matrix_print(h_dft)
        print("Hopping parameters")
        matrix_print(v)
        print("Bath state energies")
        print(np.array_str(eb, max_line_width=1000, precision=4, suppress_small=False))

    n_orb = v.shape[1]
    h = np.zeros((n_orb + len(eb), n_orb + len(eb)), dtype= complex)
    h[:n_orb, :n_orb] = np.conj(rot_spherical.T) @ h_dft @ rot_spherical
    h[:n_orb, n_orb:] = np.conj(v.T)
    h[n_orb:, :n_orb] = v
    np.fill_diagonal(h[n_orb:, n_orb:], eb)

    if verbose:
        with open(f"Ham-{label}.inp", "w") as f:
            for i in range(h.shape[0]):
                for j in range(h.shape[1]):
                    f.write(f" 0 0 0 {i+1} {j+1} {np.real(h[i,j])} {np.imag(h[i,j])}\n")

    h_op = h2imp.get_H_operator_from_dense_rspt_H_matrix(h, ang=(n_orb // 2 - 1) // 2)
    return h_op, eb
