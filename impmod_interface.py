from impmod_ed import ffi
import numpy as np
import scipy as sp

def matrix_print(matrix):
    print ("\n".join([" ".join([f"{np.real(el): .6f} {np.imag(el):+.6f}j" for el in row]) for row in matrix]))

class impModCluster:
    def __init__(self, label, h_dft, hyb, slater, nominal_occ, delta_occ,
                 n_bath_states, blocks,
                 sig, sig_real, sig_static, sig_dc, 
                 rot_spherical):
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

@ffi.def_extern()
def run_impmod_ed(rspt_label, rspt_solver_line, rspt_slater,
                  rspt_hyb, rspt_h_dft, rspt_sig, 
                  rspt_sig_real, rspt_sig_static, 
                  rspt_sig_dc, rspt_iw, rspt_w, 
                  rspt_rot_spherical,
                  n_orb, n_iw, n_w, n_rot_rows, eim, tau,
                  verbosity, size_real, size_complex):

    from rspt2spectra.hyb_fit import get_block_structure, get_equivalent_blocks
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    comm.barrier()

    label = ffi.string(rspt_label, 18).decode('ascii')
    solver_line = ffi.string(rspt_solver_line, 100).decode('ascii')

    h_dft = np.ndarray(buffer = ffi.buffer(rspt_h_dft, n_orb*n_orb*size_complex), shape = (n_orb, n_orb), order = 'F', dtype = complex)
    hyb = np.ndarray(buffer = ffi.buffer(rspt_hyb, n_w*n_orb*n_orb*size_complex), shape = (n_orb, n_orb, n_w), order = 'F', dtype = complex)
    iw = np.ndarray(buffer = ffi.buffer(rspt_iw, n_iw*size_real), shape = (n_iw, ), dtype = float)
    w = np.ndarray(buffer = ffi.buffer(rspt_w, n_w*size_real), shape = (n_w, ), dtype = float)
    sig = np.ndarray(buffer = ffi.buffer(rspt_sig, n_iw*n_orb*n_orb*size_complex), shape = (n_orb, n_orb, n_iw), order = 'F', dtype = complex)
    sig_real = np.ndarray(buffer = ffi.buffer(rspt_sig_real, n_w*n_orb*n_orb*size_complex), shape = (n_orb, n_orb, n_w), order = 'F', dtype = complex)
    sig_static = np.ndarray(buffer = ffi.buffer(rspt_sig_static, n_orb*n_orb*size_complex), shape = (n_orb, n_orb), order = 'F', dtype = complex)
    sig_dc = np.ndarray(buffer = ffi.buffer(rspt_sig_dc, n_orb*n_orb*size_complex), shape = (n_orb, n_orb), order = 'F', dtype = complex)
    rspt_rot_spherical = np.ndarray(buffer = ffi.buffer(rspt_rot_spherical, n_orb*n_rot_rows*size_complex), shape = (n_orb, n_rot_rows), order = 'F', dtype = complex)
    slater_from_rspt = np.ndarray(buffer = ffi.buffer(rspt_slater, 4*size_real), shape = (4,), dtype = float)

    if n_rot_rows == n_orb:
        if rank == 0:
            print (f"Just using rot_spherical")
        rot_spherical = rspt_rot_spherical
    else:
        rot_spherical = np.empty((n_orb, n_orb), dtype = complex)
        rot_spherical[:, :n_rot_rows] = rspt_rot_spherical
        rot_spherical[:, n_rot_rows:] = np.roll(rspt_rot_spherical, n_rot_rows, axis = 0)

    l = (n_orb//2 - 1)//2

    slater = [0]*(2*l + 1)
    for i in range(l + 1):
        slater[2*i] = slater_from_rspt[i]

    # Remove comments from the solver line
    solver_line = solver_line.split("!")[0]
    solver_line = solver_line.split("#")[0]
    solver_array = solver_line.strip().split()
    assert(len(solver_array) == 5)
    nominal_occ = ({l: int(solver_array[0])}, {l: 0}, {l: 0})
    delta_occ = ({l: int(solver_array[1])}, {l: int(solver_array[2])}, {l: int(solver_array[3])})

    bath_states_per_orbital = int(solver_array[4])
    h_op = None
    e_baths = None
    blocks = None
    spherical_blocks = None
    equivalent_blocks = None
    if rank == 0:
        h_op, e_baths = get_ed_h0(h_dft, hyb, rot_spherical, bath_states_per_orbital, w, eim, gamma = 0.01, imag_only = False)
        blocks = get_block_structure(hyb, hamiltonian = h_dft)
        equivalent_blocks = get_equivalent_blocks(blocks, hyb, hamiltonian = h_dft)
        spherical_hyb = np.moveaxis(
                        np.conj(rot_spherical.T)[np.newaxis, :, :] 
                        @ np.moveaxis(hyb, -1, 0) 
                        @ rot_spherical[np.newaxis, :, :]
                        , 0, -1)
        spherical_blocks = get_block_structure(spherical_hyb, hamiltonian = np.conj(rot_spherical.T) @ h_dft @ rot_spherical)
    h_op = comm.bcast(h_op, root = 0)
    e_baths = comm.bcast(e_baths, root = 0)
    blocks = comm.bcast(blocks, root = 0)
    spherical_blocks = comm.bcast(spherical_blocks, root = 0)
    equivalent_blocks = comm.bcast(equivalent_blocks, root = 0)

    from rspt2spectra.hyb_fit import get_block_structure

    n_bath_states = ({l: len(e_baths[e_baths <= 0]) }, {l: len(e_baths[e_baths > 0])})
    nominal_occ = (nominal_occ[0], {l: len(e_baths[e_baths <= 0])}, nominal_occ[2])
    if rank == 0:
        print (f"Block structure: {blocks}")
        print (f"Equivalent blocks: {equivalent_blocks}")
        print (f"Nominal occupation: {nominal_occ}")
        print (f"Bath states: {n_bath_states}")

    cluster = impModCluster(
            label = label.strip(),
            h_dft = h_dft,
            hyb = hyb,
            slater = slater,
            n_bath_states = n_bath_states,
            nominal_occ = nominal_occ,
            delta_occ = delta_occ,
            # blocks = spherical_blocks,
            blocks = None,
            sig = sig,
            sig_real = sig_real,
            sig_static = sig_static,
            sig_dc = sig_dc,
            rot_spherical = rot_spherical
            )


    from impurityModel.ed import selfenergy
    selfenergy.run(cluster, h_op, 1j*iw, w, eim, tau, verbosity)

    # Rotate self energy from spherical harmonics basis to RSPt's corr basis
    u = cluster.rot_spherical
    cluster.sig[:, :, :] = np.moveaxis(u[np.newaxis, :, :]
                        @ np.moveaxis(cluster.sig, -1, 0)
                        @ np.conj(u.T)[np.newaxis, :, :],
                        0, -1)
    # cluster.sig[:, :, :] = symmetrize_sigma(
    #         np.moveaxis(u[np.newaxis, :, :]
    #                     @ np.moveaxis(cluster.sig, -1, 0)
    #                     @ np.conj(u.T)[np.newaxis, :, :],
    #                     0, -1),
    #         blocks,
    #         equivalent_blocks
    #         )
    # max_val = np.max(np.abs(cluster.sig))
    # mask = np.all(np.abs(cluster.sig) < 5e-5, axis = 2)
    # cluster.sig[mask, :] = 0
    cluster.sig_real[:, :, :] = np.moveaxis(u[np.newaxis, :, :]
                        @ np.moveaxis(cluster.sig_real, -1, 0)
                        @ np.conj(u.T)[np.newaxis, :, :],
                        0, -1)
    # cluster.sig_real[:, :, :] = symmetrize_sigma(
    #         np.moveaxis(u[np.newaxis, :, :]
    #                     @ np.moveaxis(cluster.sig_real, -1, 0)
    #                     @ np.conj(u.T)[np.newaxis, :, :],
    #                     0, -1),
    #         blocks,
    #         equivalent_blocks
    #         )
    # max_val = np.max(np.abs(cluster.sig_real))
    # mask = np.all(np.abs(cluster.sig_real) < 5e-5, axis = 2)
    # cluster.sig_real[mask, :] = 0
    # for i in range(cluster.sig_real.shape[2]):
    #     cluster.sig_real[:, :, i] = u @ cluster.sig_real[:, :, i] @ np.conj(u.T)
    cluster.sig_static[:, :] = u @ cluster.sig_static @ np.conj(u.T)
    # cluster.sig_static[:, :] = symmetrize_sigma(
    #         u @ cluster.sig_static @ np.conj(u.T),
    #         blocks,
    #         equivalent_blocks
    #         )
    # max_val = np.max(np.abs(cluster.sig_static))
    # mask = np.abs(cluster.sig_static) < 1e-6
    # cluster.sig_static[mask] = 0

def symmetrize_sigma(sigma, blocks, equivalent_blocks):
    symmetrized_sigma = np.zeros_like(sigma)
    for equivalent_block in equivalent_blocks:
        for block_i in equivalent_block:
            block_idx = np.ix_(blocks[block_i], blocks[block_i])
            for block in equivalent_block:
                idx = np.ix_(blocks[block], blocks[block])
                symmetrized_sigma[idx] += sigma[block_idx]/len(equivalent_block)
    return symmetrized_sigma

def get_ed_h0(h_dft, hyb, rot_spherical, bath_states_per_orbital, w, eim, w_sparse = 1, gamma = 0.001, imag_only = False):
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
    w_sparse      -- Use every w_sparse frequency in w.
    gamma         -- Regularization parameter.

    Returns:
    h0   -- The non-interacting impurity hamiltonian in operator form.
    """
    from rspt2spectra import orbitals
    from rspt2spectra import offdiagonal
    from rspt2spectra.hyb_fit import fit_hyb, get_block_structure, get_equivalent_blocks
    from rspt2spectra import energies
    from rspt2spectra import h2imp


    # eb, v = fit_hyb(w, eim, hyb, rot_spherical, bath_states_per_orbital, gamma = gamma, imag_only = imag_only, x_lim = (w[0], 0))

    n_orb = 10
    h_rspt = np.loadtxt("h0_RSPt_real.dat") + 1j*np.loadtxt("h0_RSPt_imag.dat")
    v = h_rspt[n_orb:, :n_orb]
    eb = np.diag(h_rspt[n_orb:, n_orb:])

    print (f"DFT hamiltonian")
    matrix_print(h_dft)
    print('Hopping parameters')
    matrix_print(v)
    print('Bath state energies')
    print(np.array_str(eb, max_line_width=1000, precision=4, suppress_small=False))
    print('Shape of bath state energies:', np.shape(eb))
    print('Shape of hopping parameters:', np.shape(v))

    n_orb = v.shape[1]
    h = np.zeros((n_orb+len(eb),n_orb+len(eb)), dtype=np.complex)
    h[:n_orb, :n_orb] = h_dft
    # h[:n_orb, :n_orb] = np.conj(rot_spherical.T) @ h_dft @ rot_spherical
    h[:n_orb, n_orb:] = np.conj(v.T)
    h[n_orb:, :n_orb] = v
    np.fill_diagonal(h[n_orb:, n_orb:], eb)
    assert np.all(np.abs(h - np.conj(h.T))) < 1e-12

    u = np.identity(h.shape[0], dtype = complex)
    u[:n_orb, :n_orb] = rot_spherical
    h = np.conj(u.T) @ h @ u
    assert np.all(np.abs(h - np.conj(h.T))) < 1e-12

    h_op = h2imp.get_H_operator_from_dense_rspt_H_matrix(h,
                                                         ang=(n_orb//2-1)//2)
    return h_op, eb
