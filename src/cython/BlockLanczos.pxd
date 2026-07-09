# distutils: language = c++
"""
BlockLanczos.pxd
================
Public Cython-level interface declarations for ``BlockLanczos.pyx``.

All public functions are pure-Python callable; this file exposes
the signatures so other Cython modules can ``cimport`` them without
going through the Python import machinery.
"""

# All public symbols are Python-callable functions defined in
# BlockLanczos.pyx.  We declare them as `def` (not `cpdef`) because
# the implementation is pure-Python mode with optional ManyBodyState
# arguments that cannot be statically typed in the .pxd without full
# cimport of ManyBodyUtils.


def block_lanczos_step_cy(
    h_op,
    q_prev,
    q_curr,
    Q_basis,
    alphas,
    betas,
    it,
    reort_mode,
    W,
    mpi,
    comm,
    basis,
    slaterWeightMin=0.0,
    truncation_threshold=0,
    reort_period=5,
    start_it=0,
): ...


def block_lanczos_cy(
    psi0,
    h_op,
    basis,
    converged_fn,
    verbose=False,
    reort="full",
    max_iter=None,
    slaterWeightMin=0.0,
    truncation_threshold=0,
    comm=None,
    reort_period=5,
    alphas_init=None,
    betas_init=None,
    Q_init=None,
    W_init=None,
    return_widths=False,
    return_status=False,
    block_widths_init=None,
    locked=None,
    locked_evals=None,
    locked_res=0.0,
    locked_reort="full",
    store_krylov=True,
    krylov_dtype=None,
): ...


def thick_restart_block_lanczos_cy(
    psi0,
    h_op,
    basis,
    num_wanted,
    max_subspace_blocks,
    tol=1e-8,
    max_restarts=100,
    verbose=True,
    slaterWeightMin=0.0,
    truncation_threshold=0,
    reort="partial",
    comm=None,
): ...


def implicitly_restarted_block_lanczos_cy(
    psi0,
    h_op,
    basis,
    num_wanted,
    max_subspace_blocks,
    tol=1e-8,
    max_restarts=100,
    verbose=True,
    slaterWeightMin=0.0,
    truncation_threshold=0,
    reort="partial",
    comm=None,
): ...
