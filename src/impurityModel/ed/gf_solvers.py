r"""Per-unit Green's-function resolvent kernels (block-Lanczos and per-frequency BiCGSTAB).

This is the *kernel* half of the Green's-function machinery: given one already-built,
already-seeded unit basis, these functions compute its block Green's function. The
block-Lanczos recurrence (:func:`block_green_impl` / :func:`block_Green_sparse`, wrapped by
:func:`block_Green`) serves the whole frequency mesh from one recurrence; the per-frequency
driver (:func:`block_Green_bicgstab`, on top of :func:`solve_shifted_block`) rebuilds and
discards a basis per shift. The distribution engine that partitions work into units and calls
these kernels lives in :mod:`impurityModel.ed.gf_units`; the top-level assembly drivers live
in :mod:`impurityModel.ed.greens_function`.
"""

import itertools
from typing import Optional

import numpy as np
import scipy as sp
from mpi4py import MPI

from impurityModel.ed import config
from impurityModel.ed.basis_transcription import build_dense_matrix, build_sparse_matrix, build_state, build_vector
from impurityModel.ed.BlockLanczos import block_lanczos_cy
from impurityModel.ed.BlockLanczosArray import Reort, block_lanczos_array, resolve_reort
from impurityModel.ed.TSQR import DEFLATE_TOL_SEEDS
from impurityModel.ed.cg import block_bicgstab
from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.gf_convergence import _make_gf_convergence_monitor
from impurityModel.ed.gf_primitives import (
    _CappedBasisProxy,
    _distributed_seed_qr,
    _sanitize_continued_fraction,
    _trim_blocks,
    build_qr,
    calc_G,
)
from impurityModel.ed.gmres import block_gmres
from impurityModel.ed.manybody_basis import collective_amplitude_cutoff
from impurityModel.ed.ManyBodyUtils import ManyBodyBlockState, ManyBodyOperator, ManyBodyState, block_inner_cy

comm = MPI.COMM_WORLD
rank = comm.rank


def block_Green(
    hOp,
    psi_arr,
    basis,
    delta,
    reort,
    slaterWeightMin=0,
    verbose=True,
    eval_meshes=None,
    info=None,
):
    """
    Calculate one block of the Greens function. This function builds the many body basis
    iteratively, reducing memory requirements.

    ``eval_meshes`` is the caller's evaluation mesh per axis (see :func:`_gf_eval_meshes`); ``None``
    leaves the convergence monitor on its spectral-edge fallback.

    ``info`` (optional dict) is filled with the last :func:`block_green_impl` call's
    ``{"converged", "d_g", "n_blocks"}`` (diagnostics; e.g. the RIXS R2 solve summary
    aggregates it across every call). A caller-supplied dict is mutated in place so a
    unit's cumulative counters keep accumulating across a whole run.
    """

    n = len(psi_arr)

    # alphas/betas stay padded (k, P, P) here so the cross-expansion elementwise
    # diff below has matching shapes; they are trimmed to true block widths before
    # any continued-fraction evaluation and at the final return.
    alphas, betas, r, last_q, widths = block_green_impl(
        basis, hOp, basis.redistribute_psis(psi_arr), delta, reort, slaterWeightMin, verbose, eval_meshes, info
    )
    done = False
    while not done:
        old_size = basis.size
        # Reachability probe: repeatedly apply H to the residual block to discover new
        # determinants. The block shares its support, so the block matvec applies here too.
        # The truncation_threshold check sits INSIDE the probe loop so the basis can
        # overshoot the cap by at most one H-application batch (checking only after all
        # five rounds used to blow past it by the full five-fold fanout). basis.size is
        # replicated by add_states, so the break is collective-consistent.
        probe = ManyBodyBlockState.from_states(list(last_q))
        capped = False
        for _i in range(5):
            probe = hOp.apply_block(probe, slaterWeightMin)
            basis.add_states(
                {state for state in probe.support_keys(0.0) if state not in basis.local_basis},
            )
            if basis.size > basis.truncation_threshold:
                capped = True
                break
        if basis.size == old_size or capped:
            break
        if verbose:
            print(f"    expanded basis contains {basis.size} states")
        alphas_prev = alphas
        betas_prev = betas
        widths_prev = widths
        alphas, betas, r, last_q, widths = block_green_impl(
            basis, hOp, basis.redistribute_psis(psi_arr), delta, reort, slaterWeightMin, verbose, eval_meshes, info
        )

        n_test = min(alphas.shape[0], alphas_prev.shape[0])
        # relatively large changes in alpha and/or betas means we have not converged
        if np.any(np.abs(alphas[:n_test] - alphas_prev[:n_test]) > 1e-12) or np.any(
            np.abs(betas[:n_test] - betas_prev[:n_test]) > 1e-12
        ):
            done = False
            continue

        # alphas seem decently converged, check the Greens function to be sure
        a_t, b_t = _trim_blocks(alphas, betas, widths)
        ap_t, bp_t = _trim_blocks(alphas_prev, betas_prev, widths_prev)
        ws = np.concatenate([np.diagonal(a) for a in a_t])[: n_test * n] if a_t else np.zeros(0, dtype=complex)
        G_prev = calc_G(ap_t, bp_t, np.identity(n), ws, 0, delta)
        G = calc_G(a_t, b_t, np.identity(n), ws, 0, delta)
        done = (
            np.all(np.diagonal(G.imag, axis1=1, axis2=2) * np.sign(delta) <= 0) and np.max(np.abs(G - G_prev)) < 1e-12
        )
    return _trim_blocks(alphas, betas, widths) + (r,)


# --- Per-frequency BiCGSTAB Green's function (gf_method="bicgstab") -------------------------
# The tunable parameters (atol, iteration bound, restarts, the GMRES fallback's restart
# lengths) are declared in `ed/config.py` and read at call time -- an import-time constant
# cannot be set by a caller that has already imported this module (which silently voided a
# slicing test once).
#
# Solutions retained for the warm start: quadratic extrapolation in z through the last three
# is the measured optimum (doc/plans/bicgstab_per_frequency_gf.md Phase 3a; cubic amplifies the
# atol-level noise it extrapolates through, and each retained block costs live memory).
_GF_BICGSTAB_WARM_HISTORY = 3


# A restart must shrink the reported residual by at least this factor to earn the next one, so
# a genuinely stuck point stops early and is reported rather than looping.
_GF_BICGSTAB_RESTART_PROGRESS = 0.5


def block_green_impl(basis, hOp, psi_arr, delta, reort, slaterWeightMin, verbose, eval_meshes=None, info=None):
    """
    Internal block Green's function implementation.

    Parameters
    ----------
    basis : Basis
        The many-body basis.
    hOp : dict
        Hamiltonian operator.
    psi_arr : list of ManyBodyState
        Input state vectors.
    delta : float or ndarray
        Imaginary part/mesh info.
    reort : Reort
        Reorthogonalization method.
    slaterWeightMin : float
        Slater determinant cutoff weight.
    verbose : bool
        Whether to print verbose output.
    info : dict, optional
        Filled with ``{"converged", "d_g", "n_blocks"}`` from the convergence monitor
        (diagnostics/tests; e.g. the RIXS R2 solve summary).

    Returns
    -------
    gs_matsubara : ndarray
        Matsubara Green's function.
    gs_realaxis : ndarray
        Real axis Green's function.
    r : ndarray
        R matrix from QR.
    psi_arr : list
        Resulting states.
    """
    n = len(psi_arr)

    comm = basis.comm
    rank = comm.rank if comm is not None else 0

    dense = len(basis) < 500
    if dense:
        psi_dense = build_vector(basis, psi_arr, slaterWeightMin=0).T
        psi_dense_local, r = build_qr(psi_dense)
    else:
        # 0, not `slaterWeightMin`: this branch has always built its seed block unpruned
        # (unlike block_Green_sparse/KrylovShiftedResolvent, which prune it) -- preserved
        # as-is here since this is a mechanical extraction, not a behaviour change.
        psi_dense_local, r = _distributed_seed_qr(basis, psi_arr, 0)

    if psi_dense_local.shape[1] == 0:
        return np.zeros((0, n, n), dtype=complex), np.zeros((0, n, n), dtype=complex), r, psi_arr, []

    converged, converged_flag, delta_min, last_dg = _make_gf_convergence_monitor(delta, slaterWeightMin, eval_meshes)

    # The continued fraction only consumes alphas/betas plus the final residual block
    # (q_last below), so with reort NONE skip the full Krylov-basis retention.
    resolved_reort = resolve_reort(reort if reort is not None else Reort.NONE)

    if dense:
        H = build_dense_matrix(basis, hOp)
        alphas, betas, Q_list, widths, status = block_lanczos_array(
            psi0=psi_dense_local,
            h_op=H,
            converged=converged,
            verbose=False and verbose,  # noqa: SIM223  (force-off toggle; keep verbose wiring)
            reort=resolved_reort,
            build_krylov_basis=resolved_reort != Reort.NONE,
            return_widths=True,
            return_status=True,
            # ceil (not floor): spanning an N-dim (possibly closed) sector with a width-w block
            # needs ceil(N/w) blocks; floor truncates the final, deflating block and leaves up to
            # w-1 dimensions of the sector unresolved -- a systematic resolvent error that grows
            # with the block width (the RIXS tensor floor). The final block simply deflates.
            max_iter=-(-H.shape[0] // psi_dense_local.shape[1]),
            # The seed block is the stacked transition operators of this unit; its
            # symmetry-dependent components are what deflation has to remove, and they are
            # zero only to their construction rounding. See DEFLATE_TOL_SEEDS in TSQR.pyx.
            deflate_tol=DEFLATE_TOL_SEEDS,
        )
    else:
        h_local = build_sparse_matrix(basis, hOp)[:, basis.local_indices]

        def matmat(v):
            """
            Perform matrix-matrix multiplication with the local Hamiltonian.

            Applies the local Hamiltonian to a set of state vectors and performs
            an MPI reduction across MPI processes to accumulate the results.

            Parameters
            ----------
            v : ndarray
                Input vectors to multiply.

            Returns
            -------
            res : ndarray
                The resulting matrix product after MPI reduction.
            """
            res = h_local @ v
            if comm is not None:
                comm.Reduce(MPI.IN_PLACE if rank == 0 else res, res, op=MPI.SUM, root=0)
            return res.reshape(h_local.shape[0], v.shape[1])

        H = sp.sparse.linalg.LinearOperator(
            (len(basis), len(basis.local_indices)),
            matvec=matmat,
            rmatvec=matmat,
            matmat=matmat,
            rmatmat=matmat,
            dtype=complex,
        )

        # Run Lanczos on psi0^T* [wI - j*delta - H]^-1 psi0
        alphas, betas, Q_list, widths, status = block_lanczos_array(
            psi0=psi_dense_local,
            h_op=H,
            converged=converged,
            reort=resolved_reort,
            build_krylov_basis=resolved_reort != Reort.NONE,
            verbose=False and verbose,  # noqa: SIM223  (force-off toggle; keep verbose wiring)
            comm=comm,
            return_widths=True,
            return_status=True,
            # ceil (not floor): spanning an N-dim (possibly closed) sector with a width-w block
            # needs ceil(N/w) blocks; floor truncates the final, deflating block and leaves up to
            # w-1 dimensions of the sector unresolved -- a systematic resolvent error that grows
            # with the block width (the RIXS tensor floor). The final block simply deflates.
            max_iter=-(-H.shape[0] // psi_dense_local.shape[1]),
            # The seed block is the stacked transition operators of this unit; its
            # symmetry-dependent components are what deflation has to remove, and they are
            # zero only to their construction rounding. See DEFLATE_TOL_SEEDS in TSQR.pyx.
            deflate_tol=DEFLATE_TOL_SEEDS,
        )
    # An invariant subspace closes the Krylov space under H, so the continued fraction is
    # exact: treat it as converged (same semantics as the sparse path) so it does not trip
    # the non-convergence warning below.
    if status == "invariant_subspace":
        converged_flag[0] = True
    if not converged_flag[0] and verbose and rank == 0:
        print(
            f"warning: block Green's function did not reach the convergence tolerance "
            f"{delta_min:.1e} in {len(alphas)} block(s). The continued fraction uses the "
            f"subspace built so far.",
            flush=True,
        )
    if info is not None:
        info["converged"] = converged_flag[0]
        info["d_g"] = last_dg[0]
        info["n_blocks"] = len(alphas)
    # Keep alphas/betas padded (k, P, P) for the caller's elementwise cross-expansion diff;
    # only drop a corrupted trailing tail (whole blocks + widths) so it never reaches the
    # continued fraction. Norms of padded blocks equal those of the true blocks (zeros add
    # nothing), so the scan is valid on the padded arrays.
    keep = len(_sanitize_continued_fraction(list(alphas), list(betas), verbose=verbose, rank=rank)[0])
    if keep < len(alphas):
        alphas, betas, widths = alphas[:keep], betas[:keep], widths[:keep]
    q_last = Q_list[:, -1:]
    return alphas, betas, r, build_state(basis, q_last.T, slaterWeightMin=slaterWeightMin), widths


def block_Green_sparse(
    hOp,
    psi_arr,
    basis,
    delta,
    reort: Optional[Reort] = None,
    slaterWeightMin=0,
    verbose=True,
    cap_info=None,
    krylov_dtype=None,
    eval_meshes=None,
):
    """
    Calculate one block of the Greens function. This function builds the many body basis
    iteratively, reducing memory requirements.

    ``basis.truncation_threshold`` caps the number of Slater determinants the
    recurrence may touch (see :class:`_CappedBasisProxy`); ``np.inf`` (the ``Basis``
    default) leaves the growth bounded only by ``slaterWeightMin`` and the
    restrictions. Pass a dict as ``cap_info`` to receive ``{"cap_hit",
    "retained_size", "proxy"}`` back (diagnostics/tests).

    ``krylov_dtype`` sets the storage precision of the retained Krylov basis, which is the
    dominant allocation of a reorthogonalized run (``16 * p * n_blocks`` bytes per retained
    determinant, ~30x everything else at the FCC-Ni operating point). ``complex64`` halves
    it, at the cost of an orthogonality (and Green's function) floor at fp32 roundoff,
    ~6e-8. It is **opt-in**, not the default, for two reasons: it is rejected outright by
    ``PARTIAL``/``SELECTIVE``, whose Paige-Simon estimator steers to ``sqrt(EPS) ~ 1.5e-8``
    and cannot be fed a basis known only to ~6e-8; and it would silently break the exactness
    guarantee that a capped recurrence reproduces the dense ``P H P`` resolvent (see
    ``test_gf_truncation``). Only the *stored* basis narrows -- the recurrence, the overlaps
    and the residual stay complex128. See ``doc/plans/blocklanczos_reort_memory.md``.

    ``eval_meshes`` is the caller's evaluation mesh per axis (see :func:`_gf_eval_meshes`), which
    the convergence monitor tests ``G`` on. ``None`` leaves it on the spectral-edge fallback, which
    converges the real-axis resolvent whether or not a real-axis mesh was asked for.
    """
    comm = basis.comm
    rank = comm.rank if comm is not None else 0

    N = len(basis)
    n = len(psi_arr)

    if N == 0 or n == 0:
        return np.empty((0, n, n), dtype=complex), np.empty((0, n, n), dtype=complex), np.zeros((n, n), dtype=complex)
    psi_dense_local, r = _distributed_seed_qr(basis, psi_arr, slaterWeightMin)
    psi_arr = build_state(basis, psi_dense_local.T, slaterWeightMin=0)
    if len(psi_arr) == 0:
        return np.empty((0, n, n), dtype=complex), np.empty((0, n, n), dtype=complex), r

    converged, converged_flag, delta_min, _last_dg = _make_gf_convergence_monitor(delta, slaterWeightMin, eval_meshes)

    # The block-Lanczos matvec (h_op.apply_multi) discovers new Slater determinants as the
    # recurrence proceeds, so the reachable Krylov dimension is *not* bounded by the initial
    # excited-basis size: convergence can require many more blocks than basis.size // n. Rather
    # than guess one large cap (which either cuts the recurrence off early or wastes work),
    # resume the recurrence in growing chunks until either the Green's function converges or
    # the recurrence terminates on its own (invariant subspace / rank-deficient residual), at
    # which point the continued fraction is already exact on the space built so far. A round
    # returns fewer than `budget` new blocks exactly when the kernel stopped early; otherwise
    # it used the whole budget and there may be more spectrum to resolve, so we extend it.
    alphas = betas = Q = W = widths = None
    budget = max(int(getattr(basis, "size", 0)) // max(n, 1), 1)
    # Enforce the determinant cap on the recurrence: the proxy persists across the
    # resume rounds below, so the retained set (and a freeze) carries over.
    cap = getattr(basis, "truncation_threshold", np.inf)
    lanczos_basis = _CappedBasisProxy(basis, cap) if np.isfinite(cap) else basis
    # With reort NONE the kernel never projects against the accumulated Krylov basis and
    # the resume protocol reads only the two-block tail, so skip the full retention.
    resolved_reort = resolve_reort(reort if reort is not None else Reort.NONE)
    while True:
        alphas, betas, Q, W, widths, status = block_lanczos_cy(
            psi_arr,
            hOp,
            lanczos_basis,
            converged,
            verbose=verbose,
            reort=resolved_reort,
            slaterWeightMin=slaterWeightMin,
            max_iter=budget,
            return_widths=True,
            return_status=True,
            alphas_init=alphas,
            betas_init=betas,
            Q_init=Q,
            W_init=W,
            block_widths_init=widths,
            store_krylov=resolved_reort != Reort.NONE,
            krylov_dtype=krylov_dtype,
            # Transition-operator seed block: see DEFLATE_TOL_SEEDS in TSQR.pyx.
            deflate_tol=DEFLATE_TOL_SEEDS,
        )
        # The kernel reports exactly why it stopped (see block_lanczos_cy):
        #   * "converged"          -- the GF convergence monitor was satisfied.
        #   * "invariant_subspace" -- the block-Krylov space is closed under H (within the
        #                             excited-sector restrictions), so the continued fraction
        #                             is *exact*: this is a converged result.
        #   * "diverged"           -- the divergence guard truncated a corrupted tail; not
        #                             converged, and no further blocks can be built.
        #   * "max_iter"           -- the budget was exhausted while the matvec was still
        #                             reaching new determinants; grow the budget and resume.
        if status in ("converged", "invariant_subspace"):
            converged_flag[0] = True
            break
        if status == "diverged":
            break
        budget *= 2

    if isinstance(lanczos_basis, _CappedBasisProxy):
        if lanczos_basis.cap_hit and verbose and rank == 0:
            print(lanczos_basis.freeze_message(), flush=True)
        if cap_info is not None:
            cap_info["cap_hit"] = lanczos_basis.cap_hit
            cap_info["retained_size"] = lanczos_basis.retained_size
            cap_info["proxy"] = lanczos_basis
    elif cap_info is not None:
        cap_info["cap_hit"] = False
        cap_info["retained_size"] = None
        cap_info["proxy"] = None

    if not converged_flag[0] and rank == 0:
        print(
            f"warning: block Green's function did not reach the convergence tolerance "
            f"{delta_min:.1e}; the block-Lanczos recurrence was truncated "
            f"after {len(alphas)} block(s) (divergent tail). The continued fraction uses the "
            f"subspace built so far.",
            flush=True,
        )

    alphas, betas = _trim_blocks(alphas, betas, widths)
    alphas, betas = _sanitize_continued_fraction(alphas, betas, verbose=verbose, rank=rank)
    return alphas, betas, r


def _warm_start_extrapolation(zs, sols, z_new, n_cols):
    r"""Warm-start guess at ``z_new``: Lagrange extrapolation through the retained solutions.

    ``zs``/``sols`` hold the last (up to :data:`_GF_BICGSTAB_WARM_HISTORY`) frequencies and
    solution blocks of the sweep, oldest first. Zero, one and two retained solutions give the
    cold start, the previous solution and linear extrapolation respectively; three gives the
    quadratic optimum. The coefficients sum to 1 (an extrapolation, not a fit), so a solution
    that is locally polynomial in ``z`` is reproduced exactly.
    """
    if not sols:
        return [ManyBodyState() for _ in range(n_cols)]
    coeffs = []
    for k, zk in enumerate(zs):
        c = 1.0 + 0j
        for j, zj in enumerate(zs):
            if j != k:
                c *= (z_new - zj) / (zk - zj)
        coeffs.append(c)
    return [sum((sol[col] * c for c, sol in zip(coeffs, sols)), ManyBodyState()) for col in range(n_cols)]


def _bicgstab_sweep_order(z_shifted):
    r"""Sweep indices from the easiest frequency toward the hardest.

    Distance to the spectrum is governed by ``|Im z|``: a point far from the real axis is
    nearly diagonal-dominant and converges in a couple of iterations, so sweeping from large
    ``|Im z|`` down builds the warm-start chain on cheap solves before it reaches the hard
    region. On a fixed-broadening real-axis mesh all ``|Im z|`` are equal and the stable sort
    leaves the caller's (monotone-in-``omega``) order unchanged -- exactly the contiguous
    sweep the warm start wants there.
    """
    return np.argsort(-np.abs(np.imag(z_shifted)), kind="stable")


def solve_shifted_block(A_op, x0, rhs, basis, slaterWeightMin, atol, rtol=0.0, max_iter=None, info=None):
    r"""Restart-while-progressing BiCGSTAB, escalated to ``block_gmres`` on stagnation.

    Shared by every per-frequency resolvent solve on this branch (:func:`block_Green_bicgstab`,
    the RIXS R1 fallback in ``rixs._rixs_map_flat``): runs up to ``1 + config.GF_BICGSTAB_RESTARTS``
    :func:`~impurityModel.ed.cg.block_bicgstab` attempts, restarting with the current iterate as
    long as each attempt still makes at least ``_GF_BICGSTAB_RESTART_PROGRESS`` progress over the
    previous residual (each restart re-deflates ``Y - A x0`` and picks a fresh shadow residual,
    which is what cures near-pole stagnation -- a plain re-solve from the same iterate would not).
    If still unconverged after the restarts, escalates to :func:`~impurityModel.ed.gmres.block_gmres`,
    warm-started from BiCGSTAB's last iterate, before that iterate can poison a warm-start chain
    downstream.

    Every field of ``info`` derives from allreduce'd norms (``block_bicgstab``/``block_gmres`` are
    collective), so this restart loop is collective-consistent: every rank takes the same branch.

    Parameters
    ----------
    A_op, x0, rhs, basis, slaterWeightMin, atol
        Forwarded to ``block_bicgstab``/``block_gmres`` (``x0`` is the warm start; ``rhs`` is
        the right-hand side block, ``y`` in their signature).
    rtol : float, optional
        BiCGSTAB-only relative tolerance floor (some callers pin the RIXS R1 solve to one
        additionally); 0 (default) omits it and uses ``block_bicgstab``'s own default.
    max_iter : int, optional
        BiCGSTAB-only per-attempt iteration bound; ``None`` uses ``block_bicgstab``'s default.
    info : dict, optional
        Filled (created if not supplied) with ``converged``, ``rel_residual`` (both as reported
        by whichever solver ran last), cumulative ``iterations`` across every attempt and any
        GMRES escalation, ``gmres_used`` and ``gmres_iterations``.

    Returns
    -------
    list of ManyBodyState
        The solution block.
    """
    if info is None:
        info = {}
    bicgstab_kwargs = {"atol": atol, "info": info}
    if rtol:
        bicgstab_kwargs["rtol"] = rtol
    if max_iter is not None:
        bicgstab_kwargs["max_iter"] = max_iter

    iterations = 0
    X = x0
    prev_residual = np.inf
    for _attempt in range(1 + config.GF_BICGSTAB_RESTARTS.get()):
        X = block_bicgstab(A_op, X, rhs, basis, slaterWeightMin, **bicgstab_kwargs)
        iterations += info["iterations"]
        if info["converged"] or info["rel_residual"] > _GF_BICGSTAB_RESTART_PROGRESS * prev_residual:
            break
        prev_residual = info["rel_residual"]

    gmres_used = False
    gmres_iterations = 0
    if not info["converged"]:
        X = block_gmres(
            A_op,
            X,
            rhs,
            basis,
            slaterWeightMin,
            atol=atol,
            restart=config.GF_GMRES_RESTART.get(),
            max_restarts=config.GF_GMRES_MAX_RESTARTS.get(),
            info=info,
        )
        iterations += info["iterations"]
        gmres_used = True
        gmres_iterations = info["iterations"]

    info["iterations"] = iterations
    info["gmres_used"] = gmres_used
    info["gmres_iterations"] = gmres_iterations
    return X


def block_Green_bicgstab(
    hOp,
    psi_arr,
    basis,
    es,
    n_ops,
    z_axes,
    slaterWeightMin=0,
    atol=None,
    max_iter=None,
    verbose=False,
    excited_restrictions=None,
    excited_weighted_restrictions=None,
    bra_seeds=None,
):
    r"""Per-frequency BiCGSTAB Green's function for one work unit (memory-first path).

    For every stacked eigenstate ``e`` and every frequency ``z`` of every requested axis this
    solves the resolvent linear system

    .. math:: (z + E_e - H)\, X = \text{seeds}_e, \qquad
              G_e[i, j](z) = \langle \text{seed}_i | X_j \rangle

    instead of running one block-Lanczos recurrence for the whole mesh. The memory contract is
    the point: the excited basis is **rebuilt from the current seed + warm-start support and
    discarded at every frequency point** (the RIXS resolvent's ``tmp_basis`` pattern), so the
    retained footprint is the largest *single-point* support, not the union over the mesh that
    a Lanczos recurrence accumulates -- and a finite ``basis.truncation_threshold`` caps even
    that via :class:`_CappedBasisProxy` (freeze-growth, exact on the retained subspace). No
    Krylov store exists on this path and there is no orthogonality to lose, so accuracy is set
    by ``atol`` alone.

    Parameters
    ----------
    hOp : ManyBodyOperator
        The Hamiltonian. Each point solves against a fresh ``z*I - hOp`` operator (the RIXS
        identity-operator construction), whose occupation restrictions come from the rebuilt
        basis and whose weighted restrictions are set from ``excited_weighted_restrictions``.
    psi_arr : list of ManyBodyState
        Flat seed columns in ``(eigenstate, operator)`` order -- ``len(es) * n_ops`` entries,
        exactly the unit-seed convention of :func:`enumerate_gf_units`.
    basis : Basis
        The unit's (split) basis; carries the communicator, the clone template and
        ``truncation_threshold``.
    es : sequence of float
        Energies of the stacked eigenstates. Each eigenstate is solved separately: the shift
        enters the operator, so solves cannot be stacked across eigenstates the way one
        Lanczos recurrence serves them all.
    n_ops : int
        Seed columns per eigenstate (the Green's-function block width).
    z_axes : list of ndarray
        Complex frequency axes from :func:`_gf_signed_axes` -- *before* the ``E_e`` shift,
        which is applied here per eigenstate.
    atol : float, optional
        Per-solve residual tolerance relative to the seed norm; defaults to
        :data:`config.GF_BICGSTAB_ATOL`.
    max_iter : int, optional
        Per-point iteration bound; defaults to :data:`config.GF_BICGSTAB_MAX_ITER`.
    bra_seeds : list of ManyBodyState, optional
        Cross-element mode (the spectrum-slicing driver): a second flat block in the same
        ``(eigenstate, operator)`` order whose columns form the *bra* of the Gram,
        ``G_e[i, j] = <bra_i | X_j>`` -- e.g. the unfiltered seeds against a filtered
        right-hand side, computing ``<v| (z-H)^{-1} p_s(H) |v>``. ``None`` (default) uses
        the seeds themselves (the symmetric element).

    Returns
    -------
    tuple
        ``(G_axes, stats)``: ``G_axes[ax][p, k]`` is the ``n_ops x n_ops`` block of eigenstate
        ``p`` at frequency ``k`` of axis ``ax`` (caller's mesh order), and ``stats`` is the
        reliability/memory record consumed by the diagnostics -- solver convergence
        (``n_points``, ``n_unconverged``, ``max_rel_residual``, ``iterations``), the cap state
        (``cap``, ``cap_hit``, ``retained_size``, ``seed_overflow``) and the measured
        per-point support (``max_solve_basis``, ``max_rebuild_basis`` -- the numbers that
        decide whether this path's memory promise holds on a given workload).
    """
    atol = config.GF_BICGSTAB_ATOL.get() if atol is None else atol
    max_iter = config.GF_BICGSTAB_MAX_ITER.get() if max_iter is None else max_iter
    n_e = len(es)
    sub_comm = basis.comm
    cap = getattr(basis, "truncation_threshold", np.inf)
    # One clone (and one cloned communicator) per unit; the per-point rebuild is
    # clear() + add_states, never a re-clone. Freed collectively below -- every rank of the
    # color runs the identical unit list, so this stays in lock-step.
    tmp_basis = basis.clone(
        initial_basis=[],
        restrictions=excited_restrictions,
        weighted_restrictions=excited_weighted_restrictions,
        verbose=False,
        comm=sub_comm.Clone() if sub_comm is not None else None,
    )

    G_axes = [np.zeros((n_e, len(z_axis), n_ops, n_ops), dtype=complex) for z_axis in z_axes]
    stats = {
        "n_points": 0,
        "n_unconverged": 0,
        "max_rel_residual": 0.0,
        "iterations": 0,
        "gmres_points": 0,
        "gmres_iterations": 0,
        "atol": atol,
        "cap": cap,
        "cap_hit": False,
        "retained_size": None,
        "seed_overflow": False,
        "max_solve_basis": 0,
        "max_rebuild_basis": 0,
    }

    for p in range(n_e):
        seeds = list(psi_arr[p * n_ops : (p + 1) * n_ops])
        # Cross-element mode (spectrum slicing): the bra of the Gram is a separate block
        # (the unfiltered seeds) riding along through every per-point redistribution.
        bras = list(bra_seeds[p * n_ops : (p + 1) * n_ops]) if bra_seeds is not None else None
        for ax, z_axis in enumerate(z_axes):
            z_shifted = z_axis + es[p]
            # Fresh warm-start chain per (eigenstate, axis): extrapolating across axes (or
            # across eigenstates) would extrapolate through a discontinuous z-path.
            hist_z: list[complex] = []
            hist_x: list[list[ManyBodyState]] = []
            for k in _bicgstab_sweep_order(z_shifted):
                z = complex(z_shifted[k])
                x0 = _warm_start_extrapolation(hist_z, hist_x, z, n_ops)
                if slaterWeightMin > 0:
                    for x in x0:
                        x.prune(slaterWeightMin)
                # Rebuild-and-discard: the basis holds only this point's seed + warm-start
                # support; redistribute_psis aligns the amplitudes to the fresh ownership
                # layout (the solver assumes its states are distributed per `basis`).
                #
                # The bras are redistributed but deliberately NOT added to the basis. They
                # enter only the closing Gram, and block_inner_cy merge-joins the two key
                # vectors, so a determinant in supp(bra)\supp(X) contributes nothing;
                # ownership is by determinant hash, which is basis-independent, so the
                # merge-join stays MPI-consistent. Admitting them would pin every basis to
                # the *unfiltered* seed support -- exactly the quantity spectrum slicing
                # exists to avoid paying (on FCC Ni the unfiltered seeds saturate the cap,
                # so it would have silently capped every slice at the union support).
                carried = seeds + x0 + (bras if bras is not None else [])
                tmp_basis.clear()
                tmp_basis.add_states(sorted({state for psi in seeds + x0 for state in psi.keys()}))
                redistributed = tmp_basis.redistribute_psis(carried)
                seeds = list(redistributed[:n_ops])
                x0 = list(redistributed[n_ops : 2 * n_ops])
                if bras is not None:
                    bras = list(redistributed[2 * n_ops :])
                stats["max_rebuild_basis"] = max(stats["max_rebuild_basis"], int(tmp_basis.size))

                solve_basis = tmp_basis
                if np.isfinite(cap):
                    if tmp_basis.size > cap:
                        # The seed/warm-start support alone exceeds the cap. Never truncate
                        # the right-hand side silently: solve on it frozen (exact on that
                        # subspace) and flag it for the diagnostics.
                        stats["seed_overflow"] = True
                    solve_basis = _CappedBasisProxy(tmp_basis, cap)

                # A fresh operator per point: block_bicgstab sets its occupation
                # restrictions from the basis; the weighted restrictions are set here
                # (unconditionally, so a None clears any stale mask -- the Basis.expand
                # convention).
                A_op = z - hOp
                A_op.set_weighted_restrictions(excited_weighted_restrictions)

                # Solve, restarting while unconverged and still making progress and
                # escalating to GMRES on stagnation (block_Green_bicgstab's own warm-start
                # chain is separate from the RIXS one but shares the same solver policy).
                info = {}
                X = solve_shifted_block(
                    A_op, x0, seeds, solve_basis, slaterWeightMin, atol, max_iter=max_iter, info=info
                )

                stats["n_points"] += 1
                stats["iterations"] += info["iterations"]
                stats["max_rel_residual"] = max(stats["max_rel_residual"], info["rel_residual"])
                if info["gmres_used"]:
                    stats["gmres_points"] += 1
                    stats["gmres_iterations"] += info["gmres_iterations"]
                if not info["converged"]:
                    stats["n_unconverged"] += 1
                stats["max_solve_basis"] = max(stats["max_solve_basis"], int(tmp_basis.size))
                if isinstance(solve_basis, _CappedBasisProxy) and solve_basis.cap_hit:
                    stats["cap_hit"] = True
                    retained = solve_basis.retained_size
                    if stats["retained_size"] is None or retained < stats["retained_size"]:
                        stats["retained_size"] = retained

                # G_e[i, j] = <bra_i | X_j> (bra = seeds unless the caller supplied a
                # separate bra block); both blocks live on tmp_basis's layout, so the
                # local Gram + Allreduce is the whole inner product (no state-vector gather).
                gram = block_inner_cy(
                    ManyBodyBlockState.from_states(bras if bras is not None else seeds),
                    ManyBodyBlockState.from_states(list(X)),
                )
                if sub_comm is not None:
                    sub_comm.Allreduce(MPI.IN_PLACE, gram, op=MPI.SUM)
                G_axes[ax][p, k] = gram

                hist_z.append(z)
                hist_x.append(list(X))
                if len(hist_z) > _GF_BICGSTAB_WARM_HISTORY:
                    hist_z.pop(0)
                    hist_x.pop(0)
            if verbose:
                print(
                    f"    axis {ax}, eigenstate {p}: {len(z_shifted)} solves, "
                    f"{stats['iterations']} cumulative iterations, "
                    f"max per-point basis {stats['max_solve_basis']}",
                    flush=True,
                )

    if sub_comm is not None:
        tmp_basis.free_comm()
    return G_axes, stats


def _fit_warm_start_to_budget(tmp_basis, seeds, x0, budget):
    r"""Shrink a rebuilt per-point basis to ``budget`` by discarding warm-start-only support.

    The seeds are the right-hand side and are never truncated (the bicgstab driver's
    seed-overflow contract): every determinant carrying seed amplitude is retained, and when
    the seed support alone reaches the budget the caller solves frozen on the full rebuilt
    support with ``seed_overflow`` flagged. Otherwise the warm-start-only remainder is kept
    top-K by max column :math:`|amp|^2` through the collective amplitude bisection (ties
    under-admitted), the basis is rebuilt in place from the retained set and the warm-start
    columns are pruned to it. ``seeds``/``x0`` must be redistributed per ``tmp_basis``, so
    each determinant is counted once on its hash-owner rank. Collective on ``tmp_basis.comm``.

    Returns ``(x0, seed_overflow)``; the seeds are not touched.
    """
    seed_keys = {state for s in seeds for state in s.keys()}
    n_seed = len(seed_keys)
    if tmp_basis.is_distributed:
        n_seed = tmp_basis.comm.allreduce(n_seed, op=MPI.SUM)
    if n_seed >= budget:
        return x0, True

    keys, norms2 = ManyBodyBlockState.from_states(list(x0)).row_max_norms2()
    extra_mask = np.array([key not in seed_keys for key in keys], dtype=bool)
    comm = tmp_basis.comm if tmp_basis.is_distributed else None
    cutoff2 = collective_amplitude_cutoff(norms2[extra_mask], int(budget) - n_seed, comm)
    kept = seed_keys | set(itertools.compress(keys, extra_mask & (norms2 > cutoff2)))
    tmp_basis.clear()
    tmp_basis.add_states(sorted(kept))
    x0 = [ManyBodyState({state: amp for state, amp in x.items() if state in kept}) for x in x0]
    return x0, False


def block_Green_cipsi(
    hOp,
    psi_arr,
    basis,
    es,
    n_ops,
    z_axes,
    slaterWeightMin=0,
    atol=None,
    max_iter=None,
    verbose=False,
    excited_restrictions=None,
    excited_weighted_restrictions=None,
):
    r"""Per-frequency Green's function with resolvent-targeted CIPSI basis selection.

    Same resolvent systems and unit contract as :func:`block_Green_bicgstab` -- for every
    stacked eigenstate and frequency solve :math:`(z + E_e - H) X = \text{seeds}_e` and close
    the Gram -- but the per-point basis is grown by *importance* instead of connectivity:
    every solve runs **frozen** on the current basis :math:`P` (exact BiCGSTAB/GMRES of
    :math:`PHP`, the :class:`_CappedBasisProxy` contract), then
    :meth:`~impurityModel.ed.cipsi_solver.CIPSISolver.select_at` scores the out-of-basis
    boundary of the iterate -- :math:`\sum_i |\langle D|H|X_i\rangle|^2 / |z - E_D|^2`, the
    leading-order weight of :math:`D` in the exact solution -- and only the top candidates
    are admitted before the next round. The loop stops when the boundary residual (the true
    residual outside :math:`P`, *the* measure freeze-growth never sees) drops below
    ``GF_CIPSI_BOUNDARY_TOL``, no candidates remain, or the ``GF_CIPSI_BUDGET`` /
    ``GF_CIPSI_MAX_ROUNDS`` budgets are exhausted. Optionally the discarded boundary is
    folded back at second order (``GF_CIPSI_PT2``); its magnitude is recorded either way as
    the per-point truncation-error bar.

    Every ``GF_CIPSI_*`` knob is read from :mod:`~impurityModel.ed.config`; the solver
    tolerances reuse the bicgstab knobs (``GF_BICGSTAB_ATOL`` etc.). Parameters and the
    ``(G_axes, stats)`` return follow :func:`block_Green_bicgstab` exactly (no ``bra_seeds``
    mode); ``stats`` adds ``rounds``, ``max_boundary_rel``, ``boundary_tol`` and
    ``pt2_max_correction``.
    """
    atol = config.GF_BICGSTAB_ATOL.get() if atol is None else atol
    max_iter = config.GF_BICGSTAB_MAX_ITER.get() if max_iter is None else max_iter
    budget = config.GF_CIPSI_BUDGET.get()
    if budget is None:
        budget = getattr(basis, "truncation_threshold", np.inf)
    max_new_cfg = config.GF_CIPSI_MAX_NEW.get()
    de2_min = config.GF_CIPSI_DE2_MIN.get()
    max_rounds = config.GF_CIPSI_MAX_ROUNDS.get()
    boundary_tol = config.GF_CIPSI_BOUNDARY_TOL.get()
    if boundary_tol is None:
        boundary_tol = atol
    scorer = config.GF_CIPSI_SCORER.get()
    use_pt2 = config.GF_CIPSI_PT2.get()

    n_e = len(es)
    sub_comm = basis.comm
    tmp_basis = basis.clone(
        initial_basis=[],
        restrictions=excited_restrictions,
        weighted_restrictions=excited_weighted_restrictions,
        verbose=False,
        comm=sub_comm.Clone() if sub_comm is not None else None,
    )
    selector = CIPSISolver(tmp_basis)
    # Candidate generation must respect the excited windows: determinants outside them are
    # never admitted. Set on the Hamiltonian unconditionally (Basis.expand's convention --
    # a None clears any stale mask left on the shared operator object).
    hOp.set_restrictions(tmp_basis.restrictions)
    hOp.set_weighted_restrictions(excited_weighted_restrictions)

    G_axes = [np.zeros((n_e, len(z_axis), n_ops, n_ops), dtype=complex) for z_axis in z_axes]
    stats = {
        "n_points": 0,
        "n_unconverged": 0,
        "max_rel_residual": 0.0,
        "iterations": 0,
        "gmres_points": 0,
        "gmres_iterations": 0,
        "atol": atol,
        "cap": float(budget),
        "cap_hit": False,
        "retained_size": None,
        "seed_overflow": False,
        "max_solve_basis": 0,
        "max_rebuild_basis": 0,
        "rounds": 0,
        "max_boundary_rel": 0.0,
        "boundary_tol": boundary_tol,
        "pt2_max_correction": 0.0,
    }

    for p in range(n_e):
        seeds = list(psi_arr[p * n_ops : (p + 1) * n_ops])
        for ax, z_axis in enumerate(z_axes):
            z_shifted = z_axis + es[p]
            # Fresh warm-start chain per (eigenstate, axis), as in block_Green_bicgstab.
            hist_z: list[complex] = []
            hist_x: list[list[ManyBodyState]] = []
            for k in _bicgstab_sweep_order(z_shifted):
                z = complex(z_shifted[k])
                x0 = _warm_start_extrapolation(hist_z, hist_x, z, n_ops)
                if slaterWeightMin > 0:
                    for x in x0:
                        x.prune(slaterWeightMin)
                # Rebuild-and-discard from the seed + warm-start support; redistribute_psis
                # aligns the amplitudes to the fresh ownership layout.
                tmp_basis.clear()
                tmp_basis.add_states(sorted({state for psi in seeds + x0 for state in psi.keys()}))
                redistributed = tmp_basis.redistribute_psis(seeds + x0)
                seeds = list(redistributed[:n_ops])
                x0 = list(redistributed[n_ops:])
                stats["max_rebuild_basis"] = max(stats["max_rebuild_basis"], int(tmp_basis.size))
                if tmp_basis.size > budget:
                    x0, overflowed = _fit_warm_start_to_budget(tmp_basis, seeds, x0, budget)
                    stats["seed_overflow"] = stats["seed_overflow"] or overflowed

                # Per-column seed norms close the *relative* boundary residual; global, so
                # every rank takes the same stop decision.
                seed_norms2 = np.array([s.norm2() for s in seeds], dtype=float)
                if sub_comm is not None:
                    sub_comm.Allreduce(MPI.IN_PLACE, seed_norms2, op=MPI.SUM)
                nonzero = seed_norms2 > 0.0

                X = x0
                sel = None
                for _round in range(max_rounds):
                    stats["rounds"] += 1
                    # Solve exactly on the frozen current basis: a cap at the current size
                    # makes _CappedBasisProxy freeze immediately, so the solve is an exact
                    # BiCGSTAB/GMRES of P H P -- growth belongs to the selection, not the
                    # solver's connectivity closure.
                    frozen = _CappedBasisProxy(tmp_basis, max(int(tmp_basis.size), 1))
                    A_op = z - hOp
                    A_op.set_weighted_restrictions(excited_weighted_restrictions)
                    info = {}
                    X = solve_shifted_block(A_op, X, seeds, frozen, slaterWeightMin, atol, max_iter=max_iter, info=info)
                    stats["iterations"] += info["iterations"]
                    stats["max_rel_residual"] = max(stats["max_rel_residual"], info["rel_residual"])
                    if info["gmres_used"]:
                        stats["gmres_points"] += 1
                        stats["gmres_iterations"] += info["gmres_iterations"]

                    # Selection round (collective): score the out-of-basis boundary of the
                    # iterate. Runs even with an exhausted budget (max_new=0) -- the boundary
                    # residual and the PT2 ingredients come from the same pass.
                    remaining = budget - tmp_basis.size
                    if max_new_cfg is not None:
                        remaining = min(remaining, max_new_cfg)
                    max_new = None if np.isinf(remaining) else max(int(remaining), 0)
                    new_Dj, sel = selector.select_at(
                        z, list(X), hOp, de2_min=de2_min, max_new=max_new, scorer=scorer, slater_cutoff=slaterWeightMin
                    )
                    boundary_rel = (
                        float(np.max(np.sqrt(sel["boundary_norms2"][nonzero] / seed_norms2[nonzero])))
                        if np.any(nonzero)
                        else 0.0
                    )
                    # Stop without admitting on the last allowed round too: states added
                    # here would never be re-solved, and the PT2 closure below assumes
                    # every remaining candidate is still boundary.
                    if boundary_rel <= boundary_tol or sel["n_admitted"] == 0 or _round == max_rounds - 1:
                        break
                    tmp_basis.add_states(sorted(new_Dj))
                    # Determinant ownership is hash-based and basis-independent, so the
                    # grown basis needs no re-redistribution of seeds/X; the admitted
                    # candidates simply become in-basis rows on their owner ranks.

                stats["n_points"] += 1
                if not (info["converged"] and boundary_rel <= boundary_tol):
                    stats["n_unconverged"] += 1
                stats["max_boundary_rel"] = max(stats["max_boundary_rel"], boundary_rel)
                stats["max_solve_basis"] = max(stats["max_solve_basis"], int(tmp_basis.size))
                if np.isfinite(budget) and boundary_rel > boundary_tol and tmp_basis.size >= budget:
                    stats["cap_hit"] = True
                    retained = int(tmp_basis.size)
                    if stats["retained_size"] is None or retained < stats["retained_size"]:
                        stats["retained_size"] = retained

                gram = block_inner_cy(
                    ManyBodyBlockState.from_states(seeds),
                    ManyBodyBlockState.from_states(list(X)),
                )
                if sub_comm is not None:
                    sub_comm.Allreduce(MPI.IN_PLACE, gram, op=MPI.SUM)

                # Second-order (Loewdin downfolding) closure of the discarded boundary,
                # dG_ij = sum_D <D|H|X_i> (z - E_D)^{-1} <D|H|X_j> (complex-symmetric
                # approximation: the bra solve at conj(z) is taken as conj(X), exact for a
                # real Hamiltonian matrix). Computed always -- it is the per-point
                # truncation-error bar -- added to G only when GF_CIPSI_PT2 is set. The
                # final round admitted nothing, so every remaining candidate is boundary.
                ov = sel["overlaps"]
                dG = (ov / (z - sel["e_Dj"])[None, :]) @ ov.T if ov.shape[1] else np.zeros((n_ops, n_ops), complex)
                dG = np.ascontiguousarray(dG, dtype=complex)
                if sub_comm is not None:
                    sub_comm.Allreduce(MPI.IN_PLACE, dG, op=MPI.SUM)
                stats["pt2_max_correction"] = max(stats["pt2_max_correction"], float(np.max(np.abs(dG))))
                if use_pt2:
                    gram = gram + dG

                G_axes[ax][p, k] = gram

                hist_z.append(z)
                hist_x.append(list(X))
                if len(hist_z) > _GF_BICGSTAB_WARM_HISTORY:
                    hist_z.pop(0)
                    hist_x.pop(0)
            if verbose:
                print(
                    f"    axis {ax}, eigenstate {p}: {len(z_shifted)} points, "
                    f"{stats['rounds']} cumulative selection rounds, "
                    f"max per-point basis {stats['max_solve_basis']}, "
                    f"max boundary residual {stats['max_boundary_rel']:.1e}",
                    flush=True,
                )

    if sub_comm is not None:
        tmp_basis.free_comm()
    return G_axes, stats
