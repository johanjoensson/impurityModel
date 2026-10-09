"""Importance-ranked basis admission for one per-frequency point of the BiCGSTAB Green's function.

The default per-point basis grows by admitting every determinant the solver produces. This module
replaces that with an *outer loop* around frozen-basis solves::

    P0 = seeds + first H-shell + the determinants of the warm-start guess that still matter
    repeat:  solve (z - H) X = Y exactly on P      (a frozen basis: every matvec is projected)
             measure the residual outside P         (residual_blocks; one cutoff-0 matvec)
             score each outside determinant, admit those above the threshold (budget permitting)
    final:   solve once more to the requested tolerance on the last P

Admission happens only *between* solves. Dropping rows inside a BiCGSTAB recurrence breaks its
recursively updated residual (and with it the restart gate and the GMRES escalation), whereas
every solve here is an exact solve of ``P H P`` -- the same contract as ``_CappedBasisProxy`` after
its freeze, so the error bound of :func:`gf_primitives.resolvent_error_bound` applies unchanged.

The seeds and their first H-shell are always in ``P0`` and never pruned: that keeps the moments of
G through ``H^2`` (hence the ``Sigma`` high-frequency tail) exact.

Every decision below is taken on allreduced quantities, so all ranks of the communicator branch
identically; ``Basis.add_states`` is collective when distributed and is called unconditionally.
"""

import numpy as np
from mpi4py import MPI

from impurityModel.ed import config
from impurityModel.ed.gf_primitives import _allreduced_col_norms2, _CappedBasisProxy, residual_blocks
from impurityModel.ed.manybody_basis import collective_first_keys, collective_top_k_bounds
from impurityModel.ed.ManyBodyUtils import ManyBodyState, block_inner_cy

SCORERS = ("amplitude", "jacobi")


def _inverse_norms(norms2):
    """``1/sqrt(norms2)`` per column, 0 where the column vanishes (an annihilated orbital)."""
    norms = np.sqrt(norms2)
    out = np.zeros_like(norms)
    nonzero = norms > 0.0
    out[nonzero] = 1.0 / norms[nonzero]
    return out


def _row_scores2(block, col_weights, denominators):
    """Squared importance of every row: ``max_j |block[r, j] * col_weights[j]|^2 / |denominators[r]|^2``."""
    if len(block) == 0:
        return np.empty(0, dtype=float)
    amps = np.asarray(block)
    scores2 = np.max(np.abs(amps * col_weights[None, :]) ** 2, axis=1)
    del amps  # release the exported buffer view before anything mutates the block
    if denominators is not None:
        scores2 = scores2 / np.abs(denominators) ** 2
    return scores2


def _select_rows(scores2, eta, slots, comm, key_of):
    """Local indices of the rows to admit: score above ``eta``, and among the global top ``slots``.

    ``slots=None`` means no budget. Under a budget the top ``slots`` are taken by
    :func:`~impurityModel.ed.manybody_basis.collective_top_k_bounds` -- whole near-tie groups, then the
    boundary group in determinant-key order (``key_of(i)``: row ``i``'s key as ``bytes``) -- so every
    rank admits its share of the same set whatever the layout, and the budget is filled exactly."""
    above_eta = scores2 > eta * eta
    if slots is None:
        return np.nonzero(above_eta)[0]
    candidates = np.nonzero(above_eta)[0]
    above, boundary, n_fill = collective_top_k_bounds(scores2[candidates], max(int(slots), 0), comm)
    picked = candidates[scores2[candidates] > above]
    group = candidates[(scores2[candidates] > boundary) & (scores2[candidates] <= above)]
    chosen = collective_first_keys([key_of(int(i)) for i in group] if n_fill > 0 else [], n_fill, comm)
    if chosen:
        picked = np.concatenate([picked, np.array([i for i in group if key_of(int(i)) in chosen], dtype=int)])
    return np.sort(picked)


def _global_sum(value, comm):
    out = np.array([value], dtype=np.int64)
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, out, op=MPI.SUM)
    return int(out[0])


def _frozen_proxy(tmp_basis):
    """``tmp_basis`` frozen at its current content: every matvec through it is projected onto P."""
    proxy = _CappedBasisProxy(tmp_basis, int(tmp_basis.size))
    proxy.cap_hit = False
    return proxy


def start_set(A_op, seeds_block, seeds, x0, carry_tol, comm, n_ops, shell_tol=0.0):
    """Determinants of the point's starting basis: the seeds, their first H-shell, and the warm-start
    determinants with ``|x0_D| / ||x0|| >= carry_tol`` (scored at this point's own ``x0``, so a
    determinant that mattered at the previous frequency but not this one is dropped).

    The first shell is kept whole unless ``shell_tol`` > 0, which drops its rows with amplitude below
    ``shell_tol * ||seed_j||`` (``GF_ADMIT_FIRST_SHELL_TOL``)."""
    keys = {key for psi in seeds for key in psi.keys()}
    shell = A_op.apply_block(seeds_block, 0.0)
    if shell_tol > 0.0:
        inv_seed = _inverse_norms(_allreduced_col_norms2(seeds_block, n_ops, comm))
        scaled_shell = shell.combine_columns(np.diag(inv_seed).astype(complex))
        shell = scaled_shell.keys_new_above(ManyBodyState.from_keys([]), shell_tol * shell_tol)
    keys.update(shell.keys())
    carried = ManyBodyState.from_states(list(x0))
    inv = _inverse_norms(_allreduced_col_norms2(carried, n_ops, comm))
    if np.any(inv):
        scaled = carried.combine_columns(np.diag(inv).astype(complex))
        keys.update(scaled.keys_new_above(ManyBodyState.from_keys([]), carry_tol * carry_tol).keys())
    return keys


def solve_point_outer(
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
    comm,
    n_ops,
    solver,
    excited_weighted_restrictions=None,
    eta_override=None,
):
    r"""Solve ``(z - H) X = seeds`` at one frequency on an importance-admitted basis.

    Parameters
    ----------
    A_op : ManyBodyOperator
        ``z - H`` for this point, weighted restrictions already set by the caller.
    hOp : ManyBodyOperator
        ``H`` (for the diagonal ``H_DD`` of the Jacobi score and of the Epstein-Nesbet estimate).
    z : complex
        The shifted frequency ``z + E_e`` this point solves at.
    seeds, x0 : list of ManyBodyState
        Seed columns and the warm-start guess (cold start: empty states), as in the default path.
    tmp_basis : Basis
        The unit's per-point basis; cleared and rebuilt here. Holds ``P`` on return.
    cap : float
        Determinant budget (``inf`` = none). A start set over the cap is solved frozen, as the
        default path does, and reported as ``"budget"``.
    solver : callable
        ``gf_solvers.solve_shifted_block`` (passed in: this module sits below it).
    eta_override : float, optional
        Admission threshold of the selected scorer; ``None`` reads its ``GF_BICGSTAB_ADMIT_TOL_*``
        knob. An explicit value wins over the environment.

    Returns
    -------
    X : ManyBodyState
        The solution block, distributed per ``tmp_basis``.
    seeds : list of ManyBodyState
        The seeds redistributed onto ``tmp_basis``'s layout (what the caller's Gram needs).
    info : dict
        The last solve's ``solve_shifted_block`` info, with ``iterations`` / ``gmres_*`` summed over
        every solve of the point.
    record : dict
        ``rounds``, ``exit_reason``, ``n_solves``, ``start_size``, ``final_size``, ``cap_hit``.
    proxy : _CappedBasisProxy
        Frozen at the final ``P``: the object whose ``retained_mask`` / ``retained_size`` the
        caller's bound and statistics read.
    """
    scorer = config.GF_BICGSTAB_ADMIT_SCORER.get()
    if scorer not in SCORERS:
        raise ValueError(f"GF_BICGSTAB_ADMIT_SCORER={scorer!r}: expected one of {SCORERS}")
    eta = eta_override
    if eta is None:
        eta = (config.GF_BICGSTAB_ADMIT_TOL_AMP if scorer == "amplitude" else config.GF_BICGSTAB_ADMIT_TOL_JACOBI).get()
    rounds = config.GF_BICGSTAB_ADMIT_ROUNDS.get()
    shells = config.GF_BICGSTAB_ADMIT_SHELLS.get()
    carry_tol = config.GF_BICGSTAB_ADMIT_CARRY_TOL.get()
    en_tol = config.GF_BICGSTAB_ADMIT_EN_TOL.get()
    # Intermediate solves only need to resolve what the threshold can see; the last one goes to atol.
    loose = max(atol, 0.1 * eta)

    # --- start set -------------------------------------------------------------------------
    A_op.set_restrictions(tmp_basis.restrictions)
    keys = start_set(
        A_op,
        ManyBodyState.from_states(list(seeds)),
        seeds,
        x0,
        carry_tol,
        comm,
        n_ops,
        config.GF_ADMIT_FIRST_SHELL_TOL.get(),
    )
    tmp_basis.clear()
    tmp_basis.add_states(sorted(keys))
    redistributed = tmp_basis.redistribute_psis(*(list(seeds) + list(x0)))
    seeds = list(redistributed[:n_ops])
    Y = ManyBodyState.from_states(seeds)
    X = ManyBodyState.from_states(list(redistributed[n_ops : 2 * n_ops]))
    X.keep_rows(ManyBodyState.from_keys(tmp_basis.local_basis))
    y_inv = _inverse_norms(_allreduced_col_norms2(Y, n_ops, comm))

    record = {"start_size": int(tmp_basis.size), "rounds": 0, "exit_reason": None, "n_solves": 0, "cap_hit": False}
    total = {"iterations": 0, "gmres_iterations": 0}
    gmres_used = False
    info = {}

    def solve(tol):
        nonlocal X, gmres_used, info
        proxy = _frozen_proxy(tmp_basis)
        info = {}
        X = solver(A_op, X, Y, proxy, slaterWeightMin, tol, max_iter=max_iter, info=info)
        record["n_solves"] += 1
        total["iterations"] += info["iterations"]
        total["gmres_iterations"] += info["gmres_iterations"]
        gmres_used = gmres_used or info["gmres_used"]
        return proxy

    # --- rounds ----------------------------------------------------------------------------
    tight_done = False
    while True:
        proxy = solve(loose)
        tight_done = loose <= atol
        if record["rounds"] >= rounds:
            record["exit_reason"] = "rounds"
            break
        slots = None if not np.isfinite(cap) else int(cap) - int(tmp_basis.size)
        if slots is not None and slots <= 0:
            record["exit_reason"] = "budget"
            record["cap_hit"] = True
            break

        _inside, boundary = residual_blocks(A_op, X, Y, tmp_basis, proxy.retained_mask, n_ops)
        if en_tol > 0.0 and _en_converged(boundary, hOp, z, Y, X, en_tol, comm, n_ops):
            record["exit_reason"] = "estimate"
            break

        x_inv = _inverse_norms(_allreduced_col_norms2(X, n_ops, comm))
        admitted = 0
        b = boundary
        for shell in range(shells):
            denominators = (z - hOp.diagonal(b)) if scorer == "jacobi" and len(b) else None
            scores2 = _row_scores2(b, y_inv if scorer == "amplitude" else x_inv, denominators)
            rows = _select_rows(scores2, eta, slots, comm, lambda i, b=b: bytes(b.key_at(i).to_bytearray()))
            n_admit = _global_sum(len(rows), comm)
            chosen = [b.key_at(int(i)) for i in rows]
            tmp_basis.add_states(chosen)  # collective; empty on a rank with nothing to add
            if n_admit == 0:
                break
            admitted += n_admit
            if slots is not None:
                slots -= n_admit
                if slots <= 0:
                    record["cap_hit"] = True
                    break
            if shell + 1 < shells:
                b = _next_shell(A_op, hOp, z, b, chosen, tmp_basis, n_ops)
        if admitted == 0:
            record["exit_reason"] = "converged"
            break
        record["rounds"] += 1

    if not tight_done:
        proxy = solve(atol)
    info["iterations"] = total["iterations"]
    info["gmres_iterations"] = total["gmres_iterations"]
    info["gmres_used"] = gmres_used
    record["final_size"] = int(tmp_basis.size)
    return X, seeds, info, record, proxy


def _next_shell(A_op, hOp, z, b, chosen, tmp_basis, n_ops):
    """Boundary of the boundary: the rows a Jacobi-extended correction on ``chosen`` reaches.

    ``b`` holds the residual rows just admitted (``chosen`` are their keys). The correction they
    would receive is ``b_D / (z - H_DD)``; applying the operator to it gives the residual on the next
    shell, restricted to the determinants still outside the (now larger) basis."""
    ext = b.copy()
    ext.keep_rows(ManyBodyState.from_keys(chosen))
    if len(ext):
        scale = 1.0 / (z - hOp.diagonal(ext))
        view = np.asarray(ext)
        view *= scale[:, None]
        del view
    nxt = tmp_basis.redistribute_block(A_op.apply_block(ext, 0.0))
    mask = ManyBodyState.from_keys(tmp_basis.local_basis)
    nxt.keep_rows(nxt.keys_new_above(mask, 0.0))
    return nxt


def _en_converged(boundary, hOp, z, Y, X, en_tol, comm, n_ops):
    """Whether the Epstein-Nesbet estimate of the remaining error is below ``en_tol * max|G_jj|``."""
    est = np.zeros((n_ops, n_ops), dtype=complex)
    if len(boundary):
        amps = np.asarray(boundary)
        weights = 1.0 / (z - hOp.diagonal(boundary))
        est = (amps.conj().T * weights[None, :]) @ amps
        del amps
    gram = block_inner_cy(Y, X)
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, est, op=MPI.SUM)
        comm.Allreduce(MPI.IN_PLACE, gram, op=MPI.SUM)
    scale = float(np.max(np.abs(np.diagonal(gram)))) if gram.size else 0.0
    return scale > 0.0 and float(np.max(np.abs(est))) <= en_tol * scale
