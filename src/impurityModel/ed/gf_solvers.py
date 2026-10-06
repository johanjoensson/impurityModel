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
import time
from typing import Optional

import numpy as np
from mpi4py import MPI

from impurityModel.ed import config
from impurityModel.ed import basis_transcription
from impurityModel.ed.basis_transcription import (
    build_dense_matrix,
    build_sparse_matrix,
    build_state,
    build_vector,
    iter_local_operator_images,
)
from impurityModel.ed.BlockLanczos import block_lanczos_cy
from impurityModel.ed.BlockLanczosArray import Reort, block_lanczos_array, resolve_reort
from impurityModel.ed.cg import block_bicgstab
from impurityModel.ed.gf_admission import solve_point_outer
from impurityModel.ed.gf_convergence import _gf_monitor_tol, _make_gf_convergence_monitor
from impurityModel.ed.gf_primitives import (
    _allreduced_col_norms2,
    _CappedBasisProxy,
    _distributed_seed_qr,
    _PrunedBasisProxy,
    _sanitize_continued_fraction,
    _trim_blocks,
    build_qr,
    calc_G,
    guarded_proxy,
    real_up_to_phase,
    residual_split,
    resolvent_error_bound,
)
from impurityModel.ed.gmres import block_gmres
from impurityModel.ed.ManyBodyUtils import ManyBodyState, block_inner_cy
from impurityModel.ed.memory_estimate import current_rss_bytes
from impurityModel.ed.TSQR import DEFLATE_TOL_SEEDS

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
        basis, hOp, basis.redistribute_psis(*psi_arr), delta, reort, slaterWeightMin, verbose, eval_meshes, info
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
        # `last_q` is already the block `build_state` returned; `list()` on it would
        # iterate determinant keys, not columns.
        probe = last_q
        capped = False
        for _i in range(5):
            probe = hOp.apply_block(probe, slaterWeightMin)
            basis.add_states(
                {state for state in probe.support_keys(0.0) if not basis.contains_local(state)},
            )
            if basis.size > basis.truncation_threshold:
                capped = True
                break
        if basis.size == old_size or capped:
            break
        if verbose and (basis.comm is None or basis.comm.rank == 0):
            print(f"    expanded basis contains {basis.size} states")
        alphas_prev = alphas
        betas_prev = betas
        widths_prev = widths
        alphas, betas, r, last_q, widths = block_green_impl(
            basis, hOp, basis.redistribute_psis(*psi_arr), delta, reort, slaterWeightMin, verbose, eval_meshes, info
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


#: The array Green's-function kernel builds the dense sector matrix below this many
#: determinants and a CSR operator from here up. Deliberately *not* ``SolverOptions.dense_cutoff``:
#: that one picks the ground-state eigensolver (dense ``eigh`` vs Lanczos), a different trade --
#: a dense matvec is cheaper than CSR only for small sectors, while a dense eigensolve pays off
#: much later. Named here so it is no longer an unexplained literal (review ledger C8).
_GF_ARRAY_DENSE_MAX = 500


def _gf_reort(reort):
    """Resolve the GF ``reort`` argument; ``None`` is ``Reort.NONE``.

    ``resolve_reort`` returns any non-string unchanged, so a float (which ``SolverOptions`` used
    to document as allowed) reached the kernels untranslated, where no ``reort_mode ==`` branch
    matches it (review ledger C8). Reject it here instead.
    """
    resolved = resolve_reort(reort if reort is not None else Reort.NONE)
    if not isinstance(resolved, Reort):
        raise TypeError(f"reort must be None, a Reort member or one of its names, got {reort!r}")
    return resolved


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

    dense = len(basis) < _GF_ARRAY_DENSE_MAX
    if dense:
        psi_dense = build_vector(basis, psi_arr, slaterWeightMin=0).T
        psi_dense_local, r = build_qr(psi_dense)
    else:
        # 0, not `slaterWeightMin`: this branch has always built its seed block unpruned
        # (unlike block_Green_sparse/KrylovShiftedResolvent, which prune it) -- preserved
        # as-is here since this is a mechanical extraction, not a behaviour change.
        psi_dense_local, r = _distributed_seed_qr(basis, psi_arr, 0)

    if psi_dense_local.shape[1] == 0:
        # A deflated-to-nothing seed block never runs the Lanczos recurrence at all --
        # trivially converged (there is no continued fraction to be wrong), and set here
        # so `info` never leaves this call with a missing key.
        if info is not None:
            info["converged"] = True
            info["d_g"] = float("nan")
            info["n_blocks"] = 0
            info["tol"] = _gf_monitor_tol(slaterWeightMin, eval_meshes)
        # `last_q` must be a block on EVERY return: `block_Green` feeds it straight into
        # `apply_block`. `psi_arr` here is `redistribute_psis(*psi_arr)`, i.e. a list, so it
        # needs wrapping -- only the other return was migrated.
        return (
            np.zeros((0, n, n), dtype=complex),
            np.zeros((0, n, n), dtype=complex),
            r,
            psi_arr if isinstance(psi_arr, ManyBodyState) else ManyBodyState.from_states(list(psi_arr)),
            [],
        )

    # The continued fraction only consumes alphas/betas plus the final residual block
    # (q_last below), so with reort NONE skip the full Krylov-basis retention.
    resolved_reort = _gf_reort(reort)
    alphas, betas, Q_list, widths = _array_block_lanczos(
        basis, hOp, psi_dense_local, dense, delta, resolved_reort, slaterWeightMin, eval_meshes, info
    )
    probe = _expansion_probe_columns(Q_list, widths, tail_only=resolved_reort == Reort.NONE)
    return alphas, betas, r, build_state(basis, probe.T, slaterWeightMin=slaterWeightMin), widths


def _array_block_lanczos(
    basis, hOp, psi_dense_local, dense, delta, resolved_reort, slaterWeightMin, eval_meshes=None, info=None
):
    """The array-kernel block-Lanczos recurrence of ``hOp`` on ``basis`` from ``psi_dense_local``.

    ``psi_dense_local`` is the orthonormal seed block's rows on this rank (``basis`` order);
    ``dense`` picks the dense sector matrix over the distributed CSR. Returns the padded
    ``(alphas, betas, Q_list, widths)`` with a corrupted tail dropped, and fills ``info`` like
    :func:`block_green_impl`. Shared by :func:`block_green_impl` and the frozen-basis fallback
    of :func:`block_Green_sparse`.
    """
    comm = basis.comm
    rank = comm.rank if comm is not None else 0

    if dense:
        H = build_dense_matrix(basis, hOp)
        kernel_comm = None
    else:
        # The (global_N, N_local) CSR goes to the kernel as-is, like the CIPSI ground state's:
        # `local_indices` is the rank-contiguous range `offset + arange(N_local)`, which is the
        # row layout the kernel's distributed matvec reduce-scatters into each rank's own rows.
        # It used to be wrapped in a LinearOperator whose matmat returned all global_N rows
        # (reduced to rank 0), which the kernel cannot store in its N_local-row buffer --
        # "could not broadcast (970,9) into (273,9)" on every colour spanning two or more ranks
        # (review ledger M1; RIXS R3 at 128 ranks on the cluster).
        H = build_sparse_matrix(basis, hOp)
        if comm is not None:
            H = H[:, basis.local_indices]
        kernel_comm = comm

    # Run Lanczos on psi0^T* [wI - j*delta - H]^-1 psi0 until the continued fraction converges or
    # the Krylov space closes. ceil(N/p) blocks span an N-dim sector only while every block keeps
    # its full width p; a deflating block (stacked eigenstates, rank-deficient seeds) is narrower,
    # so ceil(N/p) can stop at "max_iter" with the sector unspanned and the fraction silently
    # truncated (review ledger C11: widths [4,4,2,2,2,2,2] span 20 of 28 dims, G off by 0.07).
    # The kernel cannot resume without a stored Krylov basis (reort NONE keeps none) and it
    # preallocates its coefficient buffers at max_iter, so the bound is not simply raised to N:
    # the budget doubles, capped at N (at width >= 1, N blocks always close the sector), and the
    # recurrence reruns. That happens only when blocks deflated, and costs at most ~2x the final
    # run. The convergence monitor is stateful, so every attempt gets a fresh one.
    n_dim = H.shape[0]
    max_iter = -(-n_dim // psi_dense_local.shape[1])
    while True:
        converged, converged_flag, delta_min, last_dg = _make_gf_convergence_monitor(
            delta, slaterWeightMin, eval_meshes
        )
        alphas, betas, Q_list, widths, status = block_lanczos_array(
            psi0=psi_dense_local,
            h_op=H,
            converged=converged,
            reort=resolved_reort,
            build_krylov_basis=resolved_reort != Reort.NONE,
            # The kernel's per-iteration print is off; the unit memory line reports instead.
            verbose=False,
            comm=kernel_comm,
            return_widths=True,
            return_status=True,
            max_iter=max_iter,
            # The seed block is the stacked transition operators of this unit; its
            # symmetry-dependent components are what deflation has to remove, and they are
            # zero only to their construction rounding. See DEFLATE_TOL_SEEDS in TSQR.pyx.
            deflate_tol=DEFLATE_TOL_SEEDS,
        )
        if status != "max_iter" or max_iter >= n_dim:
            break
        max_iter = min(2 * max_iter, n_dim)
    # An invariant subspace closes the Krylov space under H, so the continued fraction is
    # exact: treat it as converged (same semantics as the sparse path) so it does not trip
    # the non-convergence warning below.
    if status == "invariant_subspace":
        converged_flag[0] = True
    if not converged_flag[0] and rank == 0:
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
        info["tol"] = delta_min
    # Keep alphas/betas padded (k, P, P) for the caller's elementwise cross-expansion diff;
    # only drop a corrupted trailing tail (whole blocks + widths) so it never reaches the
    # continued fraction. Norms of padded blocks equal those of the true blocks (zeros add
    # nothing), so the scan is valid on the padded arrays.
    keep = len(_sanitize_continued_fraction(list(alphas), list(betas), rank=rank)[0])
    if keep < len(alphas):
        alphas, betas, widths = alphas[:keep], betas[:keep], widths[:keep]
    return alphas, betas, Q_list, widths


def _expansion_probe_columns(Q, widths, *, tail_only):
    """The Lanczos vectors :func:`block_Green` grows its basis from.

    The recurrence runs on H restricted to the current basis, so a column's chain can close
    only because the basis truncates H: it deflates, and the determinants that chain was
    missing lie next to its last vectors -- the block just before the width drop -- not next to
    the final block. Probing from the final block alone (it used to be its last column only)
    stopped the growth early and returned a silently wrong G (review ledger C12). So probe from
    the final block and from every block that preceded a narrowing.

    With the Krylov basis retained, ``Q`` holds the blocks in order (widths ``widths``, plus
    possibly one trailing residual block) and they are sliced out. In tail-only mode the kernel
    returns exactly those blocks already, pre-narrowing ones first.
    """
    if tail_only:
        return Q
    offsets = np.concatenate(([0], np.cumsum(widths, dtype=int)))
    columns = [np.arange(offsets[i], offsets[i + 1]) for i in range(len(widths) - 1) if widths[i + 1] < widths[i]]
    last = offsets[len(widths) - 1] if len(widths) > 0 else 0
    columns.append(np.arange(last, Q.shape[1]))
    return Q[:, np.concatenate(columns)]


#: Local determinants whose ``H`` images estimate the frozen CSR's fan-out for the memory check.
_CSR_FANOUT_SAMPLE = 256
#: Upper bound on the bytes one matrix element costs during and after the CSR build: the COO
#: triplet (8 + 8 + 16), the CSC it becomes (8 + 16) and the rank's column slice of it (8 + 16).
_CSR_BYTES_PER_ELEMENT = 80
#: Upper bound on the bytes one element of a build batch holds before its lookup: the bra's key
#: object plus its column and value.
_CSR_BATCH_BYTES_PER_ELEMENT = 160


def _frozen_csr_fits(frozen_basis, hOp, memory_budget, comm):
    """Collective: does the frozen ``P H P`` CSR fit the GF memory budget on every rank?

    No budget (the guard is off) always fits. Otherwise the per-rank cost is estimated from the
    mean ``H`` image size of a sample of local determinants -- an upper bound on the in-``P``
    nonzeros -- and compared with the budget left above the current RSS. Every rank enters the
    one reduction whatever its own count (an empty rank estimates 0).
    """
    if memory_budget is None:
        return True
    n_local = len(frozen_basis.local_basis)
    sampled = elements = 0
    for image in itertools.islice(iter_local_operator_images(frozen_basis, hOp, 0), _CSR_FANOUT_SAMPLE):
        sampled += 1
        elements += len(image)
    fanout = elements / sampled if sampled else 0.0
    batch = min(basis_transcription._SPARSE_BUILD_BATCH, n_local * fanout)
    need = n_local * fanout * _CSR_BYTES_PER_ELEMENT + batch * _CSR_BATCH_BYTES_PER_ELEMENT
    over = current_rss_bytes() + need > memory_budget
    if comm is not None and comm.size > 1:
        over = comm.allreduce(bool(over), op=MPI.LOR)
    return not over


def _frozen_csr_green(proxy, basis, hOp, seeds, delta, reort, slaterWeightMin, eval_meshes, info, verbose):
    """The capped GF as an array-kernel recurrence on the frozen ``P H P`` CSR, or ``None``.

    Once ``proxy`` has frozen, the whole capped recurrence -- pre-freeze steps included -- is the
    exact block Lanczos of ``P H P`` from the seeds (see :class:`_CappedBasisProxy`), so restarting
    it here on the retained set ``P`` reproduces the sparse kernel's continued fraction to
    rounding, while every step costs one SpMV instead of an apply whose out-of-``P`` image is
    discarded. Declines (returns ``None``, collectively) when the freeze came from the memory
    guard -- RSS is already at budget then -- or when the CSR would not fit the budget; the caller
    then resumes the sparse recurrence. Returns ``(alphas, betas, r, build_seconds)`` otherwise,
    with ``info`` filled by the array kernel's own monitor.
    """
    comm = basis.comm
    root = comm is None or comm.rank == 0
    if proxy.memory_frozen:
        return None
    frozen_basis = basis.clone_from_keys(proxy.retained_mask)
    if not _frozen_csr_fits(frozen_basis, hOp, proxy.memory_budget, comm):
        if verbose and root:
            print(
                f"GF basis frozen at {frozen_basis.size:,} determinants: the P H P matrix would not fit the "
                "memory budget, so the recurrence continues on the sparse kernel.",
                flush=True,
            )
        return None
    t0 = time.perf_counter()
    psi_dense_local, r = _distributed_seed_qr(frozen_basis, seeds, slaterWeightMin)
    alphas, betas, _, widths = _array_block_lanczos(
        frozen_basis, hOp, psi_dense_local, False, delta, reort, slaterWeightMin, eval_meshes, info
    )
    seconds = time.perf_counter() - t0
    if verbose and root:
        print(
            f"GF basis frozen at {frozen_basis.size:,} determinants: recurrence restarted on the P H P matrix "
            f"({len(alphas)} block(s), {seconds:.1f} s).",
            flush=True,
        )
    alphas, betas = _trim_blocks(alphas, betas, widths)
    alphas, betas = _sanitize_continued_fraction(alphas, betas, rank=comm.rank if comm is not None else 0)
    return alphas, betas, r, seconds


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
    info=None,
    memory_budget=None,
    memory_policy="tighten",
):
    """
    Calculate one block of the Greens function. This function builds the many body basis
    iteratively, reducing memory requirements.

    ``memory_budget``/``memory_policy`` switch on :class:`_CappedBasisProxy`'s measured memory
    guard (off by default; only meaningful with a finite cap).

    **Frozen-basis CSR fallback** (``GF_FROZEN_CSR``, on by default): when the cap freezes the
    support, the sparse recurrence stops at that step and the whole recurrence restarts from the
    seeds as an array-kernel SpMV on the ``P H P`` CSR of the retained set (see
    :func:`_frozen_csr_green`) -- exact on ``P`` like the sparse kernel, without applying ``H`` to
    every retained row and discarding the image outside ``P``. Not taken for a freeze by the
    memory guard, a CSR that would not fit the budget, or a ``krylov_dtype`` store.
    ``cap_info["csr_fallback"]`` says whether it ran.

    ``basis.truncation_threshold`` caps the number of Slater determinants the
    recurrence may touch (see :class:`_CappedBasisProxy`); ``np.inf`` (the ``Basis``
    default) leaves the growth bounded only by ``slaterWeightMin`` and the
    restrictions. Pass a dict as ``cap_info`` to receive ``{"cap_hit",
    "retained_size", "proxy"}`` back (diagnostics/tests). ``retained_size`` is the global
    determinant count the recurrence ran on, ``0`` when there was no seed to run one, and
    ``None`` when the support was not tracked -- only ``_CappedBasisProxy`` counts the
    determinants the matvec discovers, and it is installed only under a finite cap.

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

    ``info`` (optional dict) is filled with ``{"converged", "d_g", "n_blocks", "tol"}`` -- the
    runtime monitor's own verdict, mirroring the ``block_green_impl``/``block_Green`` contract.
    A caller-supplied dict is mutated in place. Filled with a trivially-converged default
    immediately on entry so both early-return paths below (an empty basis/seed, or a seed block
    that deflates to nothing) still leave every key present.
    """
    comm = basis.comm
    rank = comm.rank if comm is not None else 0

    N = len(basis)
    n = len(psi_arr)

    if info is not None:
        info["converged"] = True
        info["d_g"] = float("nan")
        info["n_blocks"] = 0
        info["tol"] = _gf_monitor_tol(slaterWeightMin, eval_meshes)
    if cap_info is not None:
        # Same "set every key on entry" discipline as `info` above, for the same reason: both
        # early returns below leave a caller reading `cap_info` with a fully-populated dict.
        # `retained_size` None means "the support was not tracked" (no proxy, see below), which
        # is NOT what an empty seed means -- that one ran no recurrence at all, so the two early
        # returns say 0 rather than leaving the caller to report an unmeasured quantity.
        cap_info["cap_hit"] = False
        cap_info["retained_size"] = None
        cap_info["proxy"] = None
        cap_info["csr_fallback"] = False
        cap_info["csr_seconds"] = 0.0

    if N == 0 or n == 0:
        if cap_info is not None:
            cap_info["retained_size"] = 0
        return np.empty((0, n, n), dtype=complex), np.empty((0, n, n), dtype=complex), np.zeros((n, n), dtype=complex)
    seeds = psi_arr
    psi_dense_local, r = _distributed_seed_qr(basis, psi_arr, slaterWeightMin)
    psi_arr = build_state(basis, psi_dense_local.T, slaterWeightMin=0)
    # `.width`, not `len()`: len() is the rank-local row count, so an empty-rank early
    # return here would skip the collectives the other ranks are entering.
    if psi_arr.width == 0:
        if cap_info is not None:
            cap_info["retained_size"] = 0
        return np.empty((0, n, n), dtype=complex), np.empty((0, n, n), dtype=complex), r

    converged, converged_flag, delta_min, last_dg = _make_gf_convergence_monitor(delta, slaterWeightMin, eval_meshes)

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
    # With a memory budget the proxy is installed even without a finite cap (`unlimited`): its
    # measured guard is the only thing standing between an uncapped recurrence and an OOM kill.
    # The count cap is then effectively infinite. (This routes an unlimited serial run through
    # the capped, row-chunked path, which is not bit-identical to the unproxied one.)
    prune_tol = config.GF_LANCZOS_ADMIT_TOL.get()
    if prune_tol > 0.0:
        # The ban argument needs the full, cutoff-0 step output on one rank-consistent block.
        chunks = config.GF_APPLY_ROW_CHUNKS.get()
        if chunks is not None and chunks > 1:
            raise ValueError(
                f"GF_LANCZOS_ADMIT_TOL needs GF_APPLY_ROW_CHUNKS=1 (got {chunks}): a chunked matvec hands "
                "the proxy partial sums, so a row would be ranked -- and banned -- on incomplete amplitudes"
            )
        if slaterWeightMin > 0.0:
            raise ValueError(
                f"GF_LANCZOS_ADMIT_TOL needs slaterWeightMin=0 (got {slaterWeightMin}): a row dropped inside "
                "the apply is invisible to the proxy and cannot be banned"
            )
        # The memory guard composes with pruning exactly as with the plain cap: an explicit budget
        # wins, else the one the GF stage put on this basis.
        guard_budget = memory_budget if memory_budget is not None else getattr(basis, "gf_memory_budget", None)
        guard_policy = (
            memory_policy if memory_budget is not None else (getattr(basis, "gf_memory_policy", None) or "tighten")
        )
        budget_kwargs = {"memory_budget": guard_budget, "memory_policy": guard_policy}
        lanczos_basis = _PrunedBasisProxy(
            basis,
            cap if np.isfinite(cap) else 2**62,
            prune_tol,
            first_shell_tol=config.GF_ADMIT_FIRST_SHELL_TOL.get(),
            **budget_kwargs,
        )
    elif memory_budget is None:
        # Not handed one explicitly: the guard the GF stage configured on this basis, if any
        # (clones carry it), exactly as every other capped GF kernel reads it.
        lanczos_basis = guarded_proxy(basis, cap)
    else:
        lanczos_basis = _CappedBasisProxy(
            basis, cap if np.isfinite(cap) else 2**62, memory_budget=memory_budget, memory_policy=memory_policy
        )
    # With reort NONE the kernel never projects against the accumulated Krylov basis and
    # the resume protocol reads only the two-block tail, so skip the full retention.
    resolved_reort = _gf_reort(reort)
    # The frozen-basis CSR fallback: hand control back at the freeze. A krylov_dtype store stays
    # on the sparse kernel (the array kernel keeps its Krylov basis in complex128).
    csr_candidate = config.GF_FROZEN_CSR.get() and krylov_dtype is None and isinstance(lanczos_basis, _CappedBasisProxy)
    csr = None

    def _try_csr():
        return _frozen_csr_green(
            lanczos_basis, basis, hOp, seeds, delta, resolved_reort, slaterWeightMin, eval_meshes, info, verbose
        )

    if csr_candidate:
        lanczos_basis.stop_on_freeze = True
        if lanczos_basis.frozen:
            # The seed support alone reached the cap: nothing for the sparse kernel to discover.
            # Declined: the sparse kernel runs it frozen, and must not stop again on a freeze
            # it starts in.
            lanczos_basis.stop_on_freeze = False
            csr = _try_csr()
    while csr is None:
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
        #   * "frozen"             -- the support froze this step (stop_on_freeze): restart on
        #                             the P H P CSR, or, if that declines, resume right here.
        if status == "frozen":
            lanczos_basis.stop_on_freeze = False
            csr = _try_csr()
            continue
        if status in ("converged", "invariant_subspace"):
            converged_flag[0] = True
            break
        if status == "diverged":
            converged_flag[0] = False
            break
        budget *= 2

    if isinstance(lanczos_basis, _CappedBasisProxy):
        if lanczos_basis.cap_hit and verbose and rank == 0:
            print(lanczos_basis.freeze_message(), flush=True)
        if cap_info is not None:
            cap_info["cap_hit"] = lanczos_basis.cap_hit
            cap_info["retained_size"] = lanczos_basis.retained_size
            cap_info["memory_frozen"] = lanczos_basis.memory_frozen
            cap_info["proxy"] = lanczos_basis
            cap_info["csr_fallback"] = csr is not None
            if csr is not None:
                cap_info["csr_seconds"] = csr[3]
        if csr is not None:
            # The array kernel's monitor filled `info` and warned about non-convergence itself.
            return csr[0], csr[1], csr[2]
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
    if info is not None:
        info["converged"] = converged_flag[0]
        info["d_g"] = last_dg[0]
        info["n_blocks"] = len(alphas)
        info["tol"] = delta_min

    alphas, betas = _trim_blocks(alphas, betas, widths)
    alphas, betas = _sanitize_continued_fraction(alphas, betas, rank=rank)
    return alphas, betas, r


def _warm_start_extrapolation(zs, sols, z_new, n_cols):
    r"""Warm-start guess at ``z_new``: Lagrange extrapolation through the retained solutions.

    ``zs``/``sols`` hold the last (up to :data:`config.GF_BICGSTAB_WARM_HISTORY`) frequencies and
    solution blocks of the sweep, oldest first. Zero, one and two retained solutions give the
    cold start, the previous solution and linear extrapolation respectively; three gives the
    quadratic optimum. The coefficients sum to 1 (an extrapolation, not a fit), so a solution
    that is locally polynomial in ``z`` is reproduced exactly.
    """
    if not sols:
        # width=1 per column: the cold-start return is a real list element (it flows
        # into redistribute_psis/from_states alongside genuinely-populated seeds), not
        # a bare additive-identity accumulator like the sum() below -- it must not be
        # the width-0 polymorphic zero.
        return [ManyBodyState(width=1) for _ in range(n_cols)]
    coeffs = []
    for k, zk in enumerate(zs):
        c = 1.0 + 0j
        for j, zj in enumerate(zs):
            if j != k:
                c *= (z_new - zj) / (zk - zj)
        coeffs.append(c)
    return [sum((sol[col] * c for c, sol in zip(coeffs, sols)), ManyBodyState()) for col in range(n_cols)]


def _global_seed_support(seeds, comm):
    """Distinct determinants over the seed columns, summed over the communicator.

    Every determinant has one owner once the seeds are redistributed, so the local key sets are
    disjoint. A key set rather than a block row count: a rank owning none of the seeds may hold a
    width-0 state, which ``from_states`` rejects. Collective; call from the same point on every rank.
    """
    n_local = np.array([len({key for psi in seeds for key in psi.keys()})], dtype=np.int64)
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, n_local, op=MPI.SUM)
    return int(n_local[0])


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
        the right-hand side block, ``y`` in their signature). ``x0``/``rhs`` are
        ``ManyBodyState`` on the sparse path; the restart loop below carries the same
        block ``X`` through every attempt (and into the GMRES escalation) with no
        list round trip in between.
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
    ManyBodyState
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
    admission=None,
    admit_tol=None,
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
    admission : {"all", "outer"}, optional
        Basis-growth policy; ``None`` takes :data:`config.GF_BICGSTAB_ADMISSION`. An explicit value
        wins over the environment knob. ``"outer"`` also switches the measured error bound on
        unless :data:`config.GF_BICGSTAB_RESIDUAL_CHECK` forces it off.
    admit_tol : float, optional
        Admission threshold of ``"outer"``; ``None`` takes the knob of the selected scorer.

    Returns
    -------
    tuple
        ``(G_axes, stats)``: ``G_axes[ax][p, k]`` is the ``n_ops x n_ops`` block of eigenstate
        ``p`` at frequency ``k`` of axis ``ax`` (caller's mesh order), and ``stats`` is the
        reliability/memory record consumed by the diagnostics -- solver convergence
        (``n_points``, ``n_unconverged``, ``max_rel_residual``, ``iterations``), the cap state
        (``cap``, ``cap_hit``, ``retained_size``, ``seed_overflow``) and the measured
        per-point support (``max_solve_basis``, ``max_rebuild_basis`` -- the numbers that
        decide whether this path's memory promise holds on a given workload). ``points`` is
        the per-point record behind those maxima, one dict per solve in sweep order:
        ``eigenstate``, ``axis``, ``k`` (mesh index), ``z``, ``seed_size`` (global seed
        support), ``rebuild_size`` (seed + warm-start support, before the solve grows it),
        ``solve_size`` (after), ``cap_hit``, ``converged``, ``rel_residual``, ``iterations``,
        ``gmres_used`` (and, under :data:`config.GF_BICGSTAB_RESIDUAL_CHECK`, ``r_inside``,
        ``boundary``, ``second_order`` and ``dG_bound`` -- the measured residual split and the
        elementwise bound on ``|G - G_exact|``, with ``max_dG_bound``/``max_boundary`` their
        maxima over the unit). ``rebuild_size - seed_size`` is what the warm start carried in: set
        :data:`config.GF_BICGSTAB_WARM_HISTORY` to 0 to measure per-point support cold.
    """
    atol = config.GF_BICGSTAB_ATOL.get() if atol is None else atol
    max_iter = config.GF_BICGSTAB_MAX_ITER.get() if max_iter is None else max_iter
    warm_history = config.GF_BICGSTAB_WARM_HISTORY.get()
    if admission is None:
        admission = config.GF_BICGSTAB_ADMISSION.get()
        if admission not in config.GF_ADMISSIONS:
            raise ValueError(f"GF_BICGSTAB_ADMISSION={admission!r}: expected one of {config.GF_ADMISSIONS}")
    elif admission not in config.GF_ADMISSIONS:
        raise ValueError(f"admission={admission!r}: expected one of {config.GF_ADMISSIONS}")
    # Unset, the measured error bound follows the policy: outer admission trades basis size for an
    # error, and an error nobody can read is not a trade. 1/0 in the environment force it.
    forced = config.GF_BICGSTAB_RESIDUAL_CHECK.get()
    check_residual = (admission == "outer") if forced is None else forced
    # The second-order bound needs real H (then the adjoint solve is the conjugate of the forward
    # one); a property of the operator alone, so decided once per unit.
    h_is_real = check_residual and all(np.imag(amp) == 0 for _term, amp in hOp.items())
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
        # Measured truncation error bar (GF_BICGSTAB_RESIDUAL_CHECK); None = not measured.
        "max_dG_bound": None,
        "max_boundary": None,
        "points": [],
    }

    # Freed in `finally` so a solve that raises on every rank does not leak the cloned
    # communicator (review ledger M4); collective, every rank of the color runs this unit.
    try:
        for p in range(n_e):
            seeds = list(psi_arr[p * n_ops : (p + 1) * n_ops])
            # Global seed support (distinct determinants over the eigenstate's columns): the floor
            # below which no per-point basis can go. Counted at this eigenstate's first point,
            # after the rebuild's redistribute_psis has given every determinant exactly one owner.
            seed_size = None
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
                    # A fresh operator per point: block_bicgstab sets its occupation
                    # restrictions from the basis; the weighted restrictions are set here
                    # (unconditionally, so a None clears any stale mask -- the Basis.expand
                    # convention).
                    A_op = z - hOp
                    A_op.set_weighted_restrictions(excited_weighted_restrictions)

                    admission_record = None
                    if admission == "outer":
                        # Importance-admitted basis: rebuilds and solves inside (gf_admission).
                        X, seeds, info, admission_record, solve_basis = solve_point_outer(
                            A_op,
                            hOp,
                            z,
                            seeds,
                            x0,
                            tmp_basis,
                            cap,
                            slaterWeightMin,
                            atol,
                            max_iter,
                            sub_comm,
                            n_ops,
                            solve_shifted_block,
                            eta_override=admit_tol,
                        )
                        solve_basis.cap_hit = admission_record["cap_hit"]
                        rebuild_size = admission_record["start_size"]
                        if seed_size is None:
                            seed_size = _global_seed_support(seeds, sub_comm)
                        stats["max_rebuild_basis"] = max(stats["max_rebuild_basis"], rebuild_size)
                        if np.isfinite(cap) and rebuild_size > cap:
                            stats["seed_overflow"] = True
                    else:
                        # Rebuild-and-discard: the basis holds only this point's seed + warm-start
                        # support; redistribute_psis aligns the amplitudes to the fresh ownership
                        # layout (the solver assumes its states are distributed per `basis`).
                        carried = seeds + x0
                        tmp_basis.clear()
                        tmp_basis.add_states(sorted({state for psi in seeds + x0 for state in psi.keys()}))
                        redistributed = tmp_basis.redistribute_psis(*carried)
                        seeds = list(redistributed[:n_ops])
                        x0 = list(redistributed[n_ops : 2 * n_ops])
                        if seed_size is None:
                            seed_size = _global_seed_support(seeds, sub_comm)
                        rebuild_size = int(tmp_basis.size)
                        stats["max_rebuild_basis"] = max(stats["max_rebuild_basis"], rebuild_size)

                        if np.isfinite(cap) and tmp_basis.size > cap:
                            # The seed/warm-start support alone exceeds the cap. Never truncate the
                            # right-hand side silently: solve on it frozen (exact on that subspace) and
                            # flag it for the diagnostics.
                            stats["seed_overflow"] = True
                        solve_basis = guarded_proxy(tmp_basis, cap)

                        # Solve, restarting while unconverged and still making progress and
                        # escalating to GMRES on stagnation (block_Green_bicgstab's own warm-start
                        # chain is separate from the RIXS one but shares the same solver policy).
                        # seeds/x0 are wrapped into blocks once here, at the solver boundary; the
                        # restart loop inside solve_shifted_block then carries X as a block with no
                        # further round trip.
                        info = {}
                        X = solve_shifted_block(
                            A_op,
                            ManyBodyState.from_states(list(x0)),
                            ManyBodyState.from_states(list(seeds)),
                            solve_basis,
                            slaterWeightMin,
                            atol,
                            max_iter=max_iter,
                            info=info,
                        )

                    stats["n_points"] += 1
                    stats["iterations"] += info["iterations"]
                    stats["max_rel_residual"] = max(stats["max_rel_residual"], info["rel_residual"])
                    if info["gmres_used"]:
                        stats["gmres_points"] += 1
                        stats["gmres_iterations"] += info["gmres_iterations"]
                    if not info["converged"]:
                        stats["n_unconverged"] += 1
                    solve_size = int(tmp_basis.size)
                    stats["max_solve_basis"] = max(stats["max_solve_basis"], solve_size)
                    point_cap_hit = isinstance(solve_basis, _CappedBasisProxy) and solve_basis.cap_hit
                    if point_cap_hit:
                        stats["cap_hit"] = True
                        retained = solve_basis.retained_size
                        if stats["retained_size"] is None or retained < stats["retained_size"]:
                            stats["retained_size"] = retained
                    record = {}
                    if check_residual:
                        # Y and the solve's own A_op, on the RAW basis (the proxy would keep_rows the
                        # boundary away); the retained set is the proxy's mask when there is one,
                        # else the basis itself (nothing was excluded).
                        mask = (
                            solve_basis.retained_mask
                            if isinstance(solve_basis, _CappedBasisProxy)
                            else ManyBodyState.from_keys(tmp_basis.local_basis)
                        )
                        seed_block = ManyBodyState.from_states(seeds)
                        r_p, b = residual_split(A_op, X, seed_block, tmp_basis, mask, n_ops, sub_comm)
                        s_norm = np.sqrt(_allreduced_col_norms2(seed_block, n_ops, sub_comm))
                        symmetric = (
                            real_up_to_phase(seed_block, n_ops, sub_comm) if h_is_real else np.zeros(n_ops, dtype=bool)
                        )
                        dG_bound = resolvent_error_bound(s_norm, r_p, b, abs(z.imag), symmetric)
                        stats["max_dG_bound"] = max(stats["max_dG_bound"] or 0.0, float(np.max(dG_bound)))
                        stats["max_boundary"] = max(stats["max_boundary"] or 0.0, float(np.max(b)))
                        record = {
                            "r_inside": r_p,
                            "boundary": b,
                            "second_order": symmetric,
                            "dG_bound": dG_bound,
                        }
                    stats["points"].append(
                        {
                            "eigenstate": p,
                            "axis": ax,
                            "k": int(k),
                            "z": z,
                            "seed_size": seed_size,
                            "rebuild_size": rebuild_size,
                            "solve_size": solve_size,
                            "cap_hit": bool(point_cap_hit),
                            "converged": bool(info["converged"]),
                            "rel_residual": float(info["rel_residual"]),
                            "iterations": int(info["iterations"]),
                            "gmres_used": bool(info["gmres_used"]),
                            **({"admission": admission_record} if admission_record is not None else {}),
                            **record,
                        }
                    )

                    # G_e[i, j] = <seed_i | X_j>; both blocks live on tmp_basis's layout, so the
                    # local Gram + Allreduce is the whole inner product (no state-vector gather).
                    gram = block_inner_cy(ManyBodyState.from_states(seeds), X)
                    if sub_comm is not None:
                        sub_comm.Allreduce(MPI.IN_PLACE, gram, op=MPI.SUM)
                    G_axes[ax][p, k] = gram

                    if warm_history > 0:
                        hist_z.append(z)
                        hist_x.append(X.to_states())
                        if len(hist_z) > warm_history:
                            hist_z.pop(0)
                            hist_x.pop(0)
                if verbose and (sub_comm is None or sub_comm.rank == 0):
                    print(
                        f"    axis {ax}, eigenstate {p}: {len(z_shifted)} solves, "
                        f"{stats['iterations']} cumulative iterations, "
                        f"max per-point basis {stats['max_solve_basis']}",
                        flush=True,
                    )
    finally:
        if sub_comm is not None:
            tmp_basis.free_comm()
    return G_axes, stats
