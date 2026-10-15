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

import time
from typing import Optional

import numpy as np
from mpi4py import MPI

from impurityModel.ed import basis_transcription, config
from impurityModel.ed.basis_transcription import (
    build_dense_matrix,
    build_sparse_matrix,
    build_state,
    build_vector,
    iter_local_operator_images,
)
from impurityModel.ed.BlockLanczos import block_lanczos_cy
from impurityModel.ed.BlockLanczosArray import Reort, block_lanczos_array, resolve_reort
from impurityModel.ed.BlockLanczosCore import apply_and_redistribute
from impurityModel.ed.cg import block_bicgstab
from impurityModel.ed.DistributedSparseMatrix import DistributedSparseMatrix
from impurityModel.ed.gf_admission import solve_point_outer
from impurityModel.ed.gf_convergence import _gf_monitor_tol, _make_gf_convergence_monitor
from impurityModel.ed.gf_primitives import (
    FrozenColumnMatvec,
    _allreduced_col_norms2,
    _CappedBasisProxy,
    _distributed_seed_qr,
    _sanitize_continued_fraction,
    _trim_blocks,
    block_to_local_dense,
    boundary_residual,
    build_qr,
    calc_G,
    guarded_proxy,
    local_dense_to_block,
    real_up_to_phase,
    residual_split,
    resolvent_error_bound,
)
from impurityModel.ed.gmres import block_gmres
from impurityModel.ed.ManyBodyUtils import ManyBodyState, block_inner_cy
from impurityModel.ed.memory_estimate import current_rss_bytes, format_bytes
from impurityModel.ed.mpi_comm import matvec_column_chunk_width
from impurityModel.ed.TSQR import DEFLATE_TOL_SEEDS
from impurityModel.ed.work_queue import queue_progress

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
            # Cut on the summed rows, so the discovered set does not depend on the rank count.
            probe = apply_and_redistribute(hOp, probe, basis, slaterWeightMin, basis.is_distributed)
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


def _array_operator(basis, hOp, dense):
    """``(H, kernel_comm)`` for the array kernel on ``basis``: the small sector's matrix gathered onto every
    rank and held undistributed, or this rank's columns of the distributed ``P H P``."""
    if dense:
        return DistributedSparseMatrix.from_dense(build_dense_matrix(basis, hOp)), None
    # The (global_N, N_local) CSR goes to the kernel as-is, like the CIPSI ground state's:
    # `local_indices` is the rank-contiguous range `offset + arange(N_local)`, which is the
    # row layout the kernel's distributed matvec reduce-scatters into each rank's own rows.
    # It used to be wrapped in a LinearOperator whose matmat returned all global_N rows
    # (reduced to rank 0), which the kernel cannot store in its N_local-row buffer --
    # "could not broadcast (970,9) into (273,9)" on every colour spanning two or more ranks
    # (review ledger M1; RIXS R3 at 128 ranks on the cluster).
    # Only this rank's columns are ever built, and kept as the build's column blocks
    # (DistributedSparseMatrix): ~20 B per element instead of a CSR's ~50 at its peak, and no N + 1
    # row pointer on every rank. The kernel multiplies by scattering columns (see
    # DistributedSparseMatrix.pyx); _frozen_csr_fits budgets exactly this.
    return build_sparse_matrix(basis, hOp), basis.comm


def _finish_array_run(alphas, betas, widths, status, converged_flag, delta_min, last_dg, info, rank):
    """Close an array-kernel run: its verdict, the non-convergence warning, ``info``, and the corrupted tail
    dropped. Returns the padded ``(alphas, betas, widths)``."""
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
    return alphas, betas, widths


def _array_block_lanczos(
    basis,
    hOp,
    psi_dense_local,
    dense,
    delta,
    resolved_reort,
    slaterWeightMin,
    eval_meshes=None,
    info=None,
    initial_blocks=None,
):
    """The array-kernel block-Lanczos recurrence of ``hOp`` on ``basis`` from ``psi_dense_local``.

    ``psi_dense_local`` is the orthonormal seed block's rows on this rank (``basis`` order);
    ``dense`` picks the dense sector matrix over the distributed CSR. Returns the padded
    ``(alphas, betas, Q_list, widths)`` with a corrupted tail dropped, and fills ``info`` like
    :func:`block_green_impl`. Shared by :func:`block_green_impl` and the frozen-basis fallback
    of :func:`block_Green_sparse`.

    ``initial_blocks`` (default: enough to span the sector, ``ceil(N/p)``) is the first attempt's
    block budget. The kernel preallocates its coefficient buffers at the budget, ``~32 N p`` bytes
    at ``ceil(N/p)`` -- GiBs on a multi-million determinant frozen basis that converges in a few
    hundred blocks -- so the fallback starts small and lets the doubling below raise it.
    """
    comm = basis.comm
    rank = comm.rank if comm is not None else 0
    H, kernel_comm = _array_operator(basis, hOp, dense)

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
    if initial_blocks is not None:
        max_iter = min(max_iter, max(1, int(initial_blocks)))
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
    alphas, betas, widths = _finish_array_run(
        alphas, betas, widths, status, converged_flag, delta_min, last_dg, info, rank
    )
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
#: Upper bound on the bytes one stored matrix element of the frozen ``P H P`` costs, held as the
#: build's column blocks (``DistributedSparseMatrix``: a 16 B value and a 4 B row). Measured by
#: ``test_the_column_operator_peak_stays_below_its_budget`` (20.5 B resident, 22.3 at the serial
#: build's peak); a CSR cost ~50 at its peak.
_CSR_BYTES_PER_ELEMENT = 24
#: Upper bound on the bytes one element of a build batch holds before its lookup: the bra's key
#: object plus its column and value. Measured ~104 on the distributed build.
_CSR_BATCH_BYTES_PER_ELEMENT = 160
#: Lanczos blocks of ``N_local x width`` the array kernel holds at once at reort 'none' (the
#: current and previous block, the product, the next block).
_CSR_KERNEL_VECTORS = 4
#: GF_SUPPORT_FREEZE's window: the blocks over which the weight reaching new rows is weighed against what the
#: slaterWeightMin prunes dropped. Long enough to span a gap between the layered shells a support grows in.
_SUPPORT_WINDOW = 16
#: Floor of the frozen-basis recurrence's first block budget (the measured SrMnO3 units converged
#: in 174-429 blocks); the array kernel's doubling raises it when a unit needs more.
_CSR_INITIAL_BLOCKS = 256


def _frozen_matrix_need(n_local, fanout, n_global, n_ranks, width, work_blocks=_CSR_KERNEL_VECTORS):
    """Upper bound on the bytes this rank needs for the frozen ``P H P`` recurrence.

    The matrix (:data:`_CSR_BYTES_PER_ELEMENT` per element, ``n_local * fanout`` elements, plus its
    int64 column pointers), and on top of it the larger of the two phases that follow each other:
    the build's batch transient, or the solver's working set -- ``work_blocks`` dense blocks
    (:data:`_CSR_KERNEL_VECTORS` for the Lanczos recurrence; :func:`_bicgstab_work_blocks` for the
    frozen BiCGSTAB/GMRES solves) and, on more than one rank, the graph exchange's send, receive and self buffers.
    Those are row-sized, not element-sized: up to every other rank's rows go out (``n_global -
    n_local``) and ``n_local`` rows come back from each other rank, ``w_c`` columns at a time, and
    ``w_c`` is at least one column whatever ``GS_MATVEC_EXCHANGE_BYTES`` says
    (``mpi_comm.matvec_column_chunk_width``). On SrMnO3's 4-rank colours that is ~0.45 GiB, more
    than the matrix's row pointer the column blocks removed.
    """
    elements = n_local * fanout
    matrix = elements * _CSR_BYTES_PER_ELEMENT + 8 * (n_local + 1)
    batch = min(basis_transcription._SPARSE_BUILD_BATCH, elements) * _CSR_BATCH_BYTES_PER_ELEMENT
    working = work_blocks * 16 * n_local * width
    if n_ranks > 1:
        send_rows = max(n_global - n_local, 0)
        recv_rows = n_local * (n_ranks - 1)
        w_c = matvec_column_chunk_width(max(send_rows, recv_rows), width, config.GS_MATVEC_EXCHANGE_BYTES.get())
        working += 16 * w_c * (send_rows + recv_rows + n_local)
    return matrix + max(batch, working)


def _frozen_csr_fits(frozen_basis, hOp, memory_budget, comm, report=None, width=1, work_blocks=_CSR_KERNEL_VECTORS):
    """Collective: does the frozen ``P H P`` CSR fit the GF memory budget on every rank?

    No budget (the guard is off) always fits. Otherwise the per-rank cost
    (:func:`_frozen_matrix_need`, for a seed block ``width`` wide) is estimated from the mean ``H``
    image size of a strided sample of local determinants -- an upper bound on the in-``P``
    nonzeros -- and compared with the budget left above the current RSS. Every rank enters the
    reductions whatever its own count (an empty rank estimates 0).

    ``report``, if a dict, is filled with the numbers the decision used, reduced over ``comm`` (the
    maxima: the decision is taken on the worst rank): ``n_local``, ``fanout``, ``need``, ``rss``
    and ``budget``. It adds one reduction, so it is passed or not identically on every rank.
    """
    if memory_budget is None:
        return True
    n_local = len(frozen_basis.local_basis)
    # A stride over the sorted local basis, not its first states: those share their leading-orbital
    # occupation, so their fan-out is not the basis's.
    positions = ()
    if n_local:
        positions = np.unique(np.linspace(0, n_local - 1, min(_CSR_FANOUT_SAMPLE, n_local)).astype(np.int64))
    sampled = elements = 0
    for image in iter_local_operator_images(frozen_basis, hOp, 0, indices=positions):
        sampled += 1
        elements += len(image)
    fanout = elements / sampled if sampled else 0.0
    n_ranks = comm.size if comm is not None else 1
    need = _frozen_matrix_need(n_local, fanout, frozen_basis.size, n_ranks, width, work_blocks)
    rss = current_rss_bytes()
    over = rss + need > memory_budget
    if comm is not None and comm.size > 1:
        over = comm.allreduce(bool(over), op=MPI.LOR)
        if report is not None:
            worst = np.array([n_local, fanout, need, rss], dtype=np.float64)
            comm.Allreduce(MPI.IN_PLACE, worst, op=MPI.MAX)
            n_local, fanout, need, rss = int(worst[0]), float(worst[1]), float(worst[2]), float(worst[3])
    if report is not None:
        report.update(n_local=n_local, fanout=fanout, need=need, rss=rss, budget=memory_budget)
    return not over


def _frozen_csr_basis(proxy, basis, hOp, verbose, reason="", width=1):
    """Collective: the frozen retained set as a ``Basis`` if the ``P H P`` fallback should run, else ``None``.

    Declines when the matrix would not fit the budget on top of the current RSS; the caller then
    resumes the sparse recurrence. A freeze by the measured-memory guard is offered too: RSS is at
    the guard's budget then, but the decision has its own, looser one and counts that RSS, so it
    accepts only what still fits. (It used to decline every guard freeze outright, and on CrI3,
    whose sparse estimate under-predicts, that is where capped units end up.) ``reason`` is
    appended to the decision line (``" by the support freeze"``); it only reads the retained mask.

    The budget is the GF stage's frozen-CSR budget on ``basis`` (``GF_CSR_MEMORY_SAFETY``, looser
    than the guard's), else -- for a basis that did not come through the GF stage -- the guard's.
    """
    comm = basis.comm
    frozen_basis = basis.clone_from_keys(proxy.retained_mask)
    report: dict = {}
    budget = proxy.memory_budget
    if budget is not None and getattr(basis, "gf_csr_memory_budget", None) is not None:
        budget = max(budget, basis.gf_csr_memory_budget)
    fits = _frozen_csr_fits(frozen_basis, hOp, budget, comm, report=report, width=width)
    if verbose and (comm is None or comm.rank == 0):
        # What the decision used, so a decline can be checked against the measured peak instead of
        # backed out of VmHWM afterwards (the iteration-2 declines at 6-7 ranks could not be).
        used = (
            f" [worst rank: {report['n_local']:,} local determinants x fan-out {report['fanout']:.1f} -> "
            f"need {format_bytes(report['need'])} on top of RSS {format_bytes(report['rss'])}; "
            f"budget {format_bytes(report['budget'])}]"
            if report
            else ""
        )
        if fits:
            print(
                f"GF basis frozen at {frozen_basis.size:,} determinants{reason}: the P H P matrix fits{used}.",
                flush=True,
            )
        else:
            print(
                f"GF basis frozen at {frozen_basis.size:,} determinants{reason}: the P H P matrix would not fit "
                f"the memory budget, so the recurrence continues on the sparse kernel{used}.",
                flush=True,
            )
    return frozen_basis if fits else None


def _frozen_csr_green(frozen_basis, hOp, seeds, delta, reort, slaterWeightMin, eval_meshes, info, verbose, blocks_hint):
    """The capped GF as an array-kernel recurrence on the frozen ``P H P`` CSR.

    Once the proxy has frozen, the whole capped recurrence -- pre-freeze steps included -- is the
    exact block Lanczos of ``P H P`` from the seeds (see :class:`_CappedBasisProxy`), so restarting
    it here on the retained set ``P`` reproduces the sparse kernel's continued fraction to
    rounding, while every step costs one SpMV instead of an apply whose out-of-``P`` image is
    discarded. ``blocks_hint`` is how many blocks the sparse kernel had built when it stopped; the
    first attempt's budget is a few times that (at least :data:`_CSR_INITIAL_BLOCKS`), never the
    ``ceil(N/p)`` the kernel would preallocate for. Returns ``(alphas, betas, r, seconds)``, with
    ``info`` filled by the array kernel's own monitor.
    """
    comm = frozen_basis.comm
    t0 = time.perf_counter()
    psi_dense_local, r = _distributed_seed_qr(frozen_basis, seeds, slaterWeightMin)
    alphas, betas, _, widths = _array_block_lanczos(
        frozen_basis,
        hOp,
        psi_dense_local,
        False,
        delta,
        reort,
        slaterWeightMin,
        eval_meshes,
        info,
        initial_blocks=max(_CSR_INITIAL_BLOCKS, 4 * blocks_hint),
    )
    seconds = time.perf_counter() - t0
    if verbose and (comm is None or comm.rank == 0):
        print(
            f"GF basis frozen at {frozen_basis.size:,} determinants: recurrence restarted on the P H P matrix "
            f"({len(alphas)} block(s), {seconds:.1f} s).",
            flush=True,
        )
    alphas, betas = _trim_blocks(alphas, betas, widths)
    alphas, betas = _sanitize_continued_fraction(alphas, betas, rank=comm.rank if comm is not None else 0)
    return alphas, betas, r, seconds


def _csr_continue_budget(start):
    """First block budget of a continuation after ``start`` sparse blocks: the kernel preallocates its
    coefficient buffers at it, and a run that spends it continues from its tail at twice the budget."""
    return max(_CSR_INITIAL_BLOCKS, 2 * start)


def _sparse_tail_to_local_dense(frozen_basis, mask, tail, widths):
    """The sparse kernel's two-block tail as this rank's dense rows on ``frozen_basis``. Collective.

    ``tail`` is the flat list of width-1 states a tail-only run returns (``_pack_result``): the previous
    block's columns, then the current block's. It is split at the replicated width ``widths[-1]`` of the
    previous block and the replicated column count, never at a rank-local width (a rank holding none of a
    block's rows still holds its columns). Every row must lie in the retained set ``mask``: a frozen
    proxy projected the freezing step, and every earlier step was admitted whole. One unconditional
    reduction checks that, and the widths, on every rank, and every rank raises together if either fails.
    """
    comm = frozen_basis.comm
    w_prev = int(widths[len(widths) - 1])
    blocks = (ManyBodyState.from_states(tail[:w_prev]), ManyBodyState.from_states(tail[w_prev:]))
    w_curr = len(tail) - w_prev
    outside = sum(len(b) - b.count_rows_in(mask) for b in blocks)
    bad_width = any(len(b) and b.width != w for b, w in zip(blocks, (w_prev, w_curr)))
    check = np.array([outside > 0, bad_width, len(tail), -len(tail)], dtype=np.int64)
    if comm is not None and comm.size > 1:
        comm.Allreduce(MPI.IN_PLACE, check, op=MPI.MAX)
    if check[0] or check[1] or check[2] != -check[3] or w_curr < 1:
        raise RuntimeError(
            f"the frozen recurrence's tail cannot continue on the P H P matrix: rows outside P {bool(check[0])}, "
            f"width mismatch {bool(check[1])}, columns {-check[3]}..{check[2]} (previous block {w_prev} wide)"
        )
    return tuple(block_to_local_dense(frozen_basis, b, w) for b, w in zip(blocks, (w_prev, w_curr)))


def _frozen_csr_continue(frozen_basis, hOp, alphas, betas, widths, tail, monitor, r, info, verbose):
    """The frozen sparse recurrence carried on as the array-kernel recurrence of ``P H P``, from where it stopped.

    Every block the sparse kernel built before the freeze had its whole ``H`` image admitted (so inside the
    final ``P``) and the freezing step was projected onto ``P``: up to the ``slaterWeightMin`` cut the sparse
    recurrence so far IS the block Lanczos of ``P H P``, and its next block is ``P H P`` applied to the tail.
    So it continues here instead of restarting from the seeds (:func:`_frozen_csr_green`, which recomputes
    every pre-freeze block -- 573 of them on one SrMnO3 unit). At ``slaterWeightMin = 0`` the two agree to
    rounding; above it the pre-freeze coefficients carry the sparse kernel's cut, which the restart would
    not. reort 'none' only: a two-block tail cannot continue a reorthogonalized recurrence.

    ``tail`` is the pair of dense blocks from :func:`_sparse_tail_to_local_dense`; ``monitor`` the sparse
    run's ``(converged, converged_flag, delta_min, last_dg)``, which goes on judging the same continued
    fraction (a fresh one would restart its own history); ``r`` the sparse run's seed factor. A run whose
    first budget is spent continues from its own tail, so no block is ever computed twice. Returns
    ``(alphas, betas, r, seconds)``, with ``info`` filled.
    """
    comm = frozen_basis.comm
    rank = comm.rank if comm is not None else 0
    t0 = time.perf_counter()
    converged, converged_flag, delta_min, last_dg = monitor
    start = len(alphas)
    alphas, betas, widths = np.array(alphas), np.array(betas), list(widths)
    H, kernel_comm = _array_operator(frozen_basis, hOp, False)
    n_dim = H.shape[0]
    budget = min(_csr_continue_budget(start), n_dim)
    while True:
        alphas, betas, _, widths, status, tail = block_lanczos_array(
            psi0=None,
            h_op=H,
            converged=converged,
            reort=Reort.NONE,
            build_krylov_basis=False,
            verbose=False,
            comm=kernel_comm,
            alphas=alphas,
            betas=betas,
            tail=tail,
            block_widths_init=widths,
            max_iter=budget,
            return_widths=True,
            return_status=True,
            return_tail=True,
            deflate_tol=DEFLATE_TOL_SEEDS,
        )
        # The kernel's buffers are sized to this attempt; the next one copies what it continues from.
        if status != "max_iter" or len(alphas) - start >= n_dim:
            break
        budget = min(2 * budget, n_dim - (len(alphas) - start))
    alphas, betas, widths = _finish_array_run(
        alphas, betas, widths, status, converged_flag, delta_min, last_dg, info, rank
    )
    seconds = time.perf_counter() - t0
    if verbose and rank == 0:
        print(
            f"GF basis frozen at {frozen_basis.size:,} determinants: recurrence continued on the P H P matrix "
            f"({len(alphas) - start} block(s) after {start} sparse, {seconds:.1f} s).",
            flush=True,
        )
    alphas, betas = _trim_blocks(alphas, betas, widths)
    alphas, betas = _sanitize_continued_fraction(alphas, betas, rank=rank)
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
    support, the sparse recurrence stops at that step and carries on as an array-kernel SpMV on the
    ``P H P`` CSR of the retained set -- from its last two blocks at reort 'none'
    (:func:`_frozen_csr_continue`), else restarted from the seeds (:func:`_frozen_csr_green`) -- exact
    on ``P`` like the sparse kernel, without applying ``H`` to every retained row and discarding the
    image outside ``P``. Not taken for a CSR that would not fit the budget, or a ``krylov_dtype`` store.
    ``cap_info["csr_fallback"]`` says whether it ran. The same switch is taken when growing the
    support below the cap no longer buys more than the ``slaterWeightMin`` cut costs
    (``GF_SUPPORT_FREEZE``, on by default; ``cap_info["support_frozen"]``, with the per-step weights
    it decided on in ``cap_info["support_history"]``).

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
    ~6e-8. It is **opt-in**, not the default, for two reasons: it would silently break the exactness
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
        cap_info["support_frozen"] = False
        cap_info["support_history"] = []

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
    alphas = betas = Q = widths = None
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
        # The memory guard composes with pruning exactly as with the plain cap: an explicit budget
        # wins, else the one the GF stage put on this basis.
        guard_budget = memory_budget if memory_budget is not None else getattr(basis, "gf_memory_budget", None)
        guard_policy = (
            memory_policy if memory_budget is not None else (getattr(basis, "gf_memory_policy", None) or "tighten")
        )
        budget_kwargs = {"memory_budget": guard_budget, "memory_policy": guard_policy}
        lanczos_basis = _CappedBasisProxy(
            basis,
            cap if np.isfinite(cap) else 2**62,
            eta=prune_tol,
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
    # Rank 0 reads the knobs and broadcasts them, so a malformed value fails on rank 0 alone and every rank
    # takes the same branch below (this bcast is unconditional, after every rank-invariant early return).
    knobs = (
        (
            config.GF_FROZEN_CSR.get(),
            config.GF_SUPPORT_FREEZE.get(),
        )
        if rank == 0
        else None
    )
    if comm is not None:
        knobs = comm.bcast(knobs, root=0)
    assert knobs is not None  # rank 0 read them; every other rank has them from the broadcast
    frozen_csr, support = knobs
    csr_candidate = frozen_csr and krylov_dtype is None and getattr(lanczos_basis, "tracks_support", False)
    csr = None

    def _csr_green(frozen_basis, blocks_hint=0, stopped=None):
        # `stopped` = the sparse run's (alphas, betas, widths, tail) at the freeze, passed at reort 'none'
        # only: the recurrence carries on from it. Otherwise (or with nothing built yet) it restarts from
        # the seeds, with a first budget of a few times `blocks_hint`.
        if stopped is not None and len(stopped[0]) > 0:
            s_alphas, s_betas, s_widths, s_tail = stopped
            dense_tail = _sparse_tail_to_local_dense(frozen_basis, lanczos_basis.retained_mask, s_tail, s_widths)
            return _frozen_csr_continue(
                frozen_basis,
                hOp,
                s_alphas,
                s_betas,
                s_widths,
                dense_tail,
                (converged, converged_flag, delta_min, last_dg),
                r,
                info,
                verbose,
            )
        return _frozen_csr_green(
            frozen_basis,
            hOp,
            seeds,
            delta,
            resolved_reort,
            slaterWeightMin,
            eval_meshes,
            info,
            verbose,
            blocks_hint,
        )

    # GF_SUPPORT_FREEZE: the proxy freezes the support itself, on the step its window decides, and the kernel
    # hands control back there (stop_on_freeze). reort 'none' only: that is where the frozen recurrence
    # carries on from its tail, and a reorthogonalized one would recompute every block to drop weight.
    if (
        support
        and csr_candidate
        and resolved_reort == Reort.NONE
        and getattr(lanczos_basis, "tracks_support", False)
        and not lanczos_basis.prunes
    ):
        lanczos_basis.enable_support_freeze(_SUPPORT_WINDOW, log=verbose and rank == 0)
    if csr_candidate:
        lanczos_basis.stop_on_freeze = True
        if lanczos_basis.frozen:
            # The seed support alone reached the cap: nothing for the sparse kernel to discover.
            # If declined, the sparse kernel runs it frozen, and must not stop again on a freeze
            # it starts in.
            lanczos_basis.stop_on_freeze = False
            frozen_basis = _frozen_csr_basis(lanczos_basis, basis, hOp, verbose, width=psi_arr.width)
            if frozen_basis is not None:
                csr = _csr_green(frozen_basis)
    while csr is None:
        alphas, betas, Q, _, widths, status = block_lanczos_cy(
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
        #   * "frozen"             -- the support froze this step (stop_on_freeze): carry on on
        #                             the P H P CSR, or, if that declines, resume right here.
        if status == "frozen":
            lanczos_basis.stop_on_freeze = False
            frozen_basis = _frozen_csr_basis(
                lanczos_basis,
                basis,
                hOp,
                verbose,
                reason=" by the support freeze" if lanczos_basis.support_frozen else "",
                width=psi_arr.width,
            )
            if frozen_basis is not None:
                # At reort 'none' the stopped state (its two-block tail) is where the CSR carries on from;
                # otherwise it is only needed to resume a declined switch: release its Krylov store before
                # the restart builds its own.
                stopped = (alphas, betas, widths, Q) if resolved_reort == Reort.NONE else None
                Q = None
                csr = _csr_green(frozen_basis, len(alphas), stopped)
            elif lanczos_basis.support_frozen:
                # A support freeze that buys no speed is not kept: carry on sparse and unfrozen, and still hand
                # a later cap or memory-guard freeze to the CSR.
                lanczos_basis.unfreeze_support()
                lanczos_basis.stop_on_freeze = True
            continue
        if status in ("converged", "invariant_subspace"):
            converged_flag[0] = True
            break
        if status == "diverged":
            converged_flag[0] = False
            break
        budget *= 2

    if getattr(lanczos_basis, "tracks_support", False):
        if lanczos_basis.support_frozen and csr is None:
            # The support froze on the step the recurrence ended on anyway (converged, closed or diverged): the
            # freeze dropped nothing, so it is not reported as one.
            lanczos_basis.unfreeze_support()
        frozen_early = lanczos_basis.support_frozen
        if (lanczos_basis.cap_hit or frozen_early) and verbose and rank == 0:
            print(lanczos_basis.freeze_message(), flush=True)
        if cap_info is not None:
            cap_info["cap_hit"] = lanczos_basis.cap_hit
            cap_info["retained_size"] = lanczos_basis.retained_size
            cap_info["memory_frozen"] = lanczos_basis.memory_frozen
            cap_info["proxy"] = lanczos_basis
            cap_info["csr_fallback"] = csr is not None
            cap_info["support_frozen"] = lanczos_basis.support_frozen
            cap_info["support_history"] = list(lanczos_basis.support_history)
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


class _ShiftedFrozenOperator:
    """``z - P H P`` on a frozen basis's local rows: what ``block_bicgstab``/``block_gmres`` take as an
    array operator with its own distributed matvec (``BiCGSTAB._make_matmat``)."""

    def __init__(self, z, matvec):
        self.z = complex(z)
        self._matvec = matvec

    def matmat(self, V):
        V = np.asarray(V)
        return self.z * V - self._matvec.apply(V)


class _FrozenShiftedSystem:
    r"""The per-frequency resolvent ``(z - P H P) X = Y`` on a frozen support ``P``, for every remaining
    shift of one right-hand side.

    The BiCGSTAB drivers (:func:`block_Green_bicgstab`, the RIXS R1 fallback) solve one point at a time
    on a sparse basis rebuilt per point, so every BiCGSTAB matvec applies ``H`` to every row of the
    support. Their right-hand side is the same for every shift of a unit, so once a point's solve
    admits no new determinant (or the cap freezes it), the support is the one the next points will
    reach too, and the frozen ``P H P`` -- built once, the CSR the Lanczos fallback uses -- serves them
    at one SpMV per matvec. Each solution is checked against the full ``H`` by :meth:`boundary`, the
    rows the sparse solve would have admitted; the caller falls back to the sparse solve when that is
    not below its tolerance, so a support the next shift outgrows is caught, not assumed away.

    Collective on ``frozen_basis.comm`` (construction, :meth:`solve`, :meth:`gram`, :meth:`boundary`).
    """

    def __init__(self, frozen_basis, hOp, seeds, n_cols):
        self.basis = frozen_basis
        self.n_cols = int(n_cols)
        self.size = int(frozen_basis.size)
        self.comm = frozen_basis.comm
        self.matvec = FrozenColumnMatvec(build_sparse_matrix(frozen_basis, hOp), self.comm, self.n_cols)
        self.seeds = block_to_local_dense(frozen_basis, ManyBodyState.from_states(list(seeds)), self.n_cols)
        norms2 = np.sum(np.abs(self.seeds) ** 2, axis=0)
        if self.comm is not None:
            self.comm.Allreduce(MPI.IN_PLACE, norms2, op=MPI.SUM)
            # Rank 0's bits everywhere: :meth:`boundary` divides the (broadcast) leak by these, and the verdict
            # on the ratio picks the next collectives, which an Allreduce does not promise to keep identical.
            if self.comm.size > 1:
                self.comm.Bcast(norms2, root=0)
        self.seed_norms = np.sqrt(norms2)

    def to_dense(self, states):
        return block_to_local_dense(self.basis, ManyBodyState.from_states(list(states)), self.n_cols)

    def to_block(self, dense):
        return local_dense_to_block(self.basis, dense)

    def solve(self, z, x0, atol, max_iter, info, rtol=0.0):
        """``X`` (local dense rows) with ``(z - P H P) X = Y``; ``info`` as :func:`solve_shifted_block`'s.
        Bounded like the sparse path, which stops at ``iterations * rank >= |support|`` with ``rank`` the
        deflated rank of the initial residual -- at least 1, so ``|P|`` iterations at most."""
        bound = self.size + 1
        bound = bound if max_iter is None else min(bound, int(max_iter))
        with self.matvec.session():
            return solve_shifted_block(
                _ShiftedFrozenOperator(z, self.matvec),
                x0,
                self.seeds,
                self.basis,
                0.0,
                atol,
                rtol=rtol,
                max_iter=bound,
                info=info,
            )

    def gram(self, X):
        """``G[i, j] = <seed_i | X_j>``, summed over the ranks."""
        gram = np.ascontiguousarray(self.seeds.conj().T @ X)
        if self.comm is not None:
            self.comm.Allreduce(MPI.IN_PLACE, gram, op=MPI.SUM)
        return gram

    def boundary(self, A_op, X_block, slaterWeightMin):
        """Per column, ``||(1 - P) A X||`` relative to ``||Y||``, over the rows the sparse solve would have
        admitted. That solve runs on the normalized system (unit right-hand-side columns), so its
        ``slaterWeightMin`` is relative to ``||Y||``, and so is the cut here: an absolute cut is lax for
        small seeds and strict for large ones (measured: seeds scaled by 1e-3 and 1e3 froze 0 and 9 of
        the same points)."""
        if hasattr(A_op, "set_restrictions"):
            A_op.set_restrictions(self.basis.restrictions)
        cutoff = slaterWeightMin * float(np.max(self.seed_norms)) if len(self.seed_norms) else 0.0
        b = boundary_residual(A_op, X_block, self.basis, self.n_cols, cutoff, self.comm)
        return b / np.maximum(self.seed_norms, np.finfo(float).tiny)


def _bicgstab_work_blocks():
    """Dense ``(n_local, width)`` blocks a frozen solve holds: BiCGSTAB's iterates and the restart
    loop's copies (measured 14.1), or, on a GMRES escalation, its Krylov basis of ``GF_GMRES_RESTART``
    blocks plus its own (measured 49.3 at the default 40)."""
    return max(15, config.GF_GMRES_RESTART.get() + 10)


def _freeze_bicgstab_support(tmp_basis, hOp, seeds, n_cols, verbose):
    """Collective: a :class:`_FrozenShiftedSystem` on ``tmp_basis``'s current support, or ``None`` when the
    frozen matrix would not fit its budget or the exchange cannot run it.

    ``tmp_basis`` is cleared either way: the next point either solves frozen or rebuilds it, so its
    copy of the support is dropped before the matrix is sized (and is not held next to the frozen
    basis's)."""
    comm = tmp_basis.comm
    frozen_basis = tmp_basis.clone_from_keys(tmp_basis._keys)
    tmp_basis.clear()
    budget = getattr(tmp_basis, "gf_csr_memory_budget", None)
    if budget is None:
        budget = getattr(tmp_basis, "gf_memory_budget", None)
    report = {}
    fits = _frozen_csr_fits(
        frozen_basis, hOp, budget, comm, report=report, width=n_cols, work_blocks=_bicgstab_work_blocks()
    )
    if verbose and (comm is None or comm.rank == 0):
        verdict = "fits" if fits else "would not fit; the solves stay sparse"
        print(
            f"BiCGSTAB support stalled at {frozen_basis.size:,} determinants: the P H P matrix {verdict}.", flush=True
        )
    if not fits:
        return None
    return _FrozenShiftedSystem(frozen_basis, hOp, seeds, n_cols)


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
    # GF_SUPPORT_FREEZE: once a point's solve admits no new determinant, the remaining points of the
    # eigenstate solve on the frozen P H P (_FrozenShiftedSystem). Rank 0 reads the knob and broadcasts it,
    # so every rank takes the same branch.
    closure = config.GF_SUPPORT_FREEZE.get() if sub_comm is None or sub_comm.rank == 0 else None
    if sub_comm is not None:
        closure = sub_comm.bcast(closure, root=0)
    closure = bool(closure) and admission != "outer"
    stats["frozen_points"] = 0
    stats["frozen_fallbacks"] = 0
    stats["frozen_size"] = None
    try:
        for p in range(n_e):
            seeds = list(psi_arr[p * n_ops : (p + 1) * n_ops])
            # The frozen system serves every axis of this eigenstate (same seeds); a fallback ends it.
            frozen = None
            freeze_ok = closure
            points_left = sum(len(z_axis) for z_axis in z_axes)  # this eigenstate's points still to solve
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
                    # Once per frequency point: a GF unit-queue host lets pending fetches through
                    # (work_queue; the Lanczos kernels do it once per block in the monitor).
                    queue_progress()
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
                    frozen_used = False
                    if frozen is not None:
                        info = {}
                        X_dense = frozen.solve(z, frozen.to_dense(x0), atol, max_iter, info)
                        X = frozen.to_block(X_dense)
                        leak = frozen.boundary(A_op, X, slaterWeightMin)
                        if np.all(leak <= atol):
                            frozen_used = True
                            solve_basis = frozen.basis
                            rebuild_size = frozen.size
                            stats["frozen_points"] += 1
                        else:
                            # The support this shift needs is larger than the frozen one: solve it
                            # sparse from here on, warm-started from the frozen solution.
                            if verbose and (sub_comm is None or sub_comm.rank == 0):
                                print(
                                    f"BiCGSTAB frozen support left {float(np.max(leak)):.1e} of the right-hand "
                                    f"side outside it at z = {z:.4g} (tolerance {atol:.1e}): back to the sparse "
                                    "solve for this eigenstate.",
                                    flush=True,
                                )
                            stats["frozen_fallbacks"] += 1
                            x0 = X.to_states()
                            frozen = None
                            freeze_ok = False
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
                    elif not frozen_used:
                        # Rebuild-and-discard: the basis holds only this point's seed + warm-start
                        # support; redistribute_psis aligns the amplitudes to the fresh ownership
                        # layout (the solver assumes its states are distributed per `basis`).
                        # Summed over the ranks before their keys enter the basis (routing is by
                        # owner, not by membership), so a row whose partials cancel is not kept.
                        tmp_basis.clear()
                        redistributed = tmp_basis.redistribute_psis(*(seeds + x0))
                        tmp_basis.add_states(sorted({state for psi in redistributed for state in psi.keys()}))
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
                    solve_size = frozen.size if frozen_used else int(tmp_basis.size)
                    stats["max_solve_basis"] = max(stats["max_solve_basis"], solve_size)
                    point_cap_hit = getattr(solve_basis, "tracks_support", False) and solve_basis.cap_hit
                    points_left -= 1
                    # This point's solve admitted nothing beyond its starting support: freeze it for
                    # the eigenstate's remaining points (at the end of this point; the residual check
                    # below still reads tmp_basis). All replicated. Not after a cap hit: the capped
                    # support always leaks against the full H, so the frozen check would send every
                    # point back (measured: 0 frozen, 1 fallback per eigenstate).
                    freeze_now = (
                        freeze_ok
                        and frozen is None
                        and points_left > 0
                        and not point_cap_hit
                        and solve_size <= rebuild_size
                    )
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
                        if frozen_used:
                            res_basis, mask = solve_basis, solve_basis._keys
                        else:
                            res_basis = tmp_basis
                            mask = (
                                solve_basis.retained_mask
                                if getattr(solve_basis, "tracks_support", False)
                                else ManyBodyState.from_keys(tmp_basis.local_basis)
                            )
                        seed_block = ManyBodyState.from_states(seeds)
                        r_p, b = residual_split(A_op, X, seed_block, res_basis, mask, n_ops, sub_comm)
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
                            "frozen": bool(frozen_used),
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
                    if frozen_used:
                        gram = frozen.gram(X_dense)
                    else:
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
                    if freeze_now:
                        frozen = _freeze_bicgstab_support(tmp_basis, hOp, seeds, n_ops, verbose)
                        freeze_ok = frozen is not None
                        if frozen is not None:
                            stats["frozen_size"] = frozen.size
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
