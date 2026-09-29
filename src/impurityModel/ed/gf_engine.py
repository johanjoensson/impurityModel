"""The Green's-function engine: the pieces every GF driver shares.

Every driver -- the self-energy's :func:`greens_function.get_Greens_function`, the spectra
tensor path's :func:`greens_function.calc_Greens_function_with_offdiag` and
:func:`spectra.calc_spectra` -- runs the same pipeline: enumerate work units (operator group x
spectral side x eigenstate chunk, :func:`gf_units.enumerate_gf_units`), distribute them
(:func:`gf_units.run_units_distributed`), solve each with a block-Lanczos recurrence, then
reassemble the per-unit coefficients by eigenstate and thermally average. They used to carry
their own copies of the unit kernel, the reassembly and the side combination; this module is
the one copy. ``gf_units`` owns distribution (units, weights, caps, the split); this module owns
what a unit computes and how the results are put back together.

Layering: below ``greens_function``, ``spectra``, ``rixs`` and ``susceptibility``; above
``gf_units``, ``gf_solvers`` and the basis layer.
"""

import sys

import numpy as np
from mpi4py import MPI

from impurityModel.ed.gf_solvers import block_Green, block_Green_sparse
from impurityModel.ed.memory_estimate import format_bytes, peak_rss_bytes, reset_peak_rss
from impurityModel.ed.solver_trace import note as _trace_note


def _block_green_group(
    split_basis,
    hOp,
    group_seed_states,
    reort,
    delta,
    slaterWeightMin,
    sparse,
    verbose,
    excited_restrictions,
    excited_weighted_restrictions,
    eval_meshes=None,
    conv_stats=None,
    unit_label=None,
):
    """Run one (possibly wide) block-Lanczos Green's function for a group of stacked seeds.

    ``group_seed_states`` is the flat list of seed columns for an eigenstate group (length
    ``len(group) * n_ops`` in ``(eigenstate, operator)`` order). Builds the excited basis from
    their union, points ``hOp`` at its restrictions, and runs the sparse or dense block-Green
    kernel. Returns ``(alphas, betas, r, n_basis, cap_stats)``; the caller slices ``r``'s columns
    per eigenstate (``r[:, p*n_ops:(p+1)*n_ops]``) since ``(alphas, betas)`` are shared by the
    group. ``cap_stats`` is ``{"cap_hit", "retained_size", "cap"}`` describing whether this
    solve froze at ``truncation_threshold`` (feeds the basis_cap diagnostic).
    ``retained_size`` is the global determinant count the recurrence ran on -- this unit's
    maximum basis size -- or ``None`` on the sparse path with no finite cap, where nothing
    tracks the support the matvec discovers (see the comment at the sparse branch).

    ``eval_meshes`` (:func:`_gf_eval_meshes`) tells the convergence monitor which frequencies this
    unit's ``G`` will be evaluated on. Passing ``None`` -- the default, and what the spectra/RIXS
    callers do -- leaves it converging the real-axis resolvent over the resolved Ritz band.

    ``conv_stats`` (optional dict) is filled with the solver's own convergence verdict --
    ``{"converged", "d_g", "n_blocks", "tol"}``, forwarded verbatim from
    :func:`block_Green_sparse`/:func:`block_Green` (see their ``info`` parameter). ``None``
    (the default) skips it -- only the caller assembling the ``lanczos`` diagnostic needs it.
    This function always reads ``n_blocks`` back for its own memory report below, regardless
    of whether the caller wants the rest of ``conv_stats``.

    ``unit_label`` (optional) identifies this unit in the memory report below -- a plain
    string or index, not interpreted otherwise.

    Every call -- one per work unit, this being the per-unit body ``run_units_distributed``
    calls into -- reports its own peak: the GS phase prints ``VmHWM`` on every CIPSI cycle,
    but before this the GF phase printed only the two split-time *predictions* and then
    nothing until it finished or was OOM-killed (the SrMnO3 crash this closes needed
    arithmetic to reconstruct, instead of a log line -- see ``doc/plans/dc_smo_memory.md``,
    "GF unit memory"). ``peak_rss_bytes()`` is reset on entry (this call's own transient,
    not a run-wide high-water mark that never resets) and MAX-allreduced over
    ``split_basis.comm`` -- the color this unit ran on, not the job -- so the binding rank's
    peak is what gets reported, matching how the memory model itself is sized (see
    :func:`memory_estimate._routing_skew_factor`). Collective, called unconditionally
    whether or not the print fires; the note is recorded even with no active
    :func:`solver_trace.tracing` block (a no-op then).
    """
    # Capture whether the reset actually worked, the way cipsi_solver does: on a kernel or
    # container where /proc/self/clear_refs is unwritable, `peak_rss_bytes()` below is a
    # run-wide cumulative mark, not this unit's transient, and reporting it as the latter is
    # exactly the staleness that misled an earlier debugging session (round 6).
    peak_is_own = reset_peak_rss()
    excited_basis = split_basis.clone(
        initial_basis={state for p in group_seed_states for state in p},
        restrictions=excited_restrictions,
        weighted_restrictions=excited_weighted_restrictions,
        verbose=False,
    )
    # Unconditional: masks are sticky on the operator object and cannot be read back, so every
    # consumer states the mask it needs (None clears). A conditional set left the previous
    # unit's -- or the ground state's -- window in force (review ledger C3).
    hOp.set_restrictions(excited_basis.restrictions)
    hOp.set_weighted_restrictions(excited_basis.weighted_restrictions)
    cap = getattr(excited_basis, "truncation_threshold", np.inf)
    # The seed support: the union of the unit's seed columns, before any recurrence step. When it
    # alone reaches the cap the solve is frozen at its seeds (gf_diagnostics.check_basis_truncation).
    seed_size = int(excited_basis.size)
    # `conv_stats` stays None to the caller that didn't ask for it; this function still wants
    # `n_blocks` for its own report, so it reads back through its own dict either way.
    info = {} if conv_stats is None else conv_stats
    if sparse:
        cap_info = {}
        alphas, betas, r = block_Green_sparse(
            reort=reort,
            hOp=hOp,
            psi_arr=excited_basis.redistribute_psis(*group_seed_states),
            basis=excited_basis,
            delta=delta,
            slaterWeightMin=slaterWeightMin,
            verbose=verbose,
            cap_info=cap_info,
            eval_meshes=eval_meshes,
            info=info,
            # Set per color by gf_units.run_units_distributed (the GF memory guard); absent on a
            # basis that did not come through it, which leaves the guard off.
            memory_budget=getattr(split_basis, "gf_memory_budget", None),
            memory_policy=getattr(split_basis, "gf_memory_policy", "tighten"),
        )
        # `retained_size` stays None when the cap is infinite, and that is not a formatting
        # gap to paper over: the sparse recurrence's support is tracked *only* by
        # `_CappedBasisProxy.redistribute_block`, which `block_Green_sparse` installs only when
        # the cap is finite. `excited_basis` is the clone of the seed support and the matvec
        # never adds to it (`Basis.redistribute_block` routes rows, it does not register them),
        # so `len(excited_basis)` here is the SEED size, not the Krylov support -- measured 1 vs
        # 15 determinants for identical physics, uncapped vs a non-binding cap. See
        # block_Green_sparse's own note that "the reachable Krylov dimension is *not* bounded by
        # the initial excited-basis size". The dense branch below is the opposite case: the
        # array kernel cannot leave its basis, so there `len(excited_basis)` IS the support.
        cap_stats = {
            "cap_hit": bool(cap_info.get("cap_hit", False)),
            "retained_size": cap_info.get("retained_size"),
            "cap": cap,
            "seed_size": seed_size,
            "memory_frozen": bool(cap_info.get("memory_frozen", False)),
        }
    else:
        alphas, betas, r = block_Green(
            reort=reort,
            hOp=hOp,
            psi_arr=excited_basis.redistribute_psis(*group_seed_states),
            basis=excited_basis,
            delta=delta,
            slaterWeightMin=slaterWeightMin,
            verbose=verbose,
            eval_meshes=eval_meshes,
            info=info,
        )
        # The array path stops expanding when the basis crosses the cap (never removes).
        cap_stats = {
            "cap_hit": bool(np.isfinite(cap) and excited_basis.size > cap),
            "retained_size": len(excited_basis),
            "cap": cap,
            "seed_size": seed_size,
        }
    comm = split_basis.comm
    peak = peak_rss_bytes()
    if comm is not None:
        peak = comm.allreduce(peak, op=MPI.MAX)
    # A cumulative mark is not this unit's peak; say so rather than quietly overstating it.
    peak_kind = "unit" if peak_is_own else "cumulative"
    retained_size = cap_stats["retained_size"]
    n_blocks = info.get("n_blocks")
    _trace_note(
        "gf_unit_memory",
        unit=unit_label,
        retained_size=int(retained_size) if retained_size is not None else None,
        cap=float(cap) if np.isfinite(cap) else None,
        cap_hit=bool(cap_stats["cap_hit"]),
        n_blocks=int(n_blocks) if n_blocks is not None else None,
        peak_rss_bytes=int(peak),
        peak_is_unit_transient=bool(peak_is_own),
    )
    if verbose and (comm is None or comm.rank == 0):
        label = f"unit {unit_label}" if unit_label is not None else "unit"
        retained_display = f"{retained_size:,}" if retained_size is not None else "n/a"
        print(
            f"  {label}: excited basis {retained_display} determinants "
            f"(cap={cap:,.0f}, cap_hit={cap_stats['cap_hit']}) n_blocks={n_blocks} "
            f"color MAX VmHWM={format_bytes(peak)} ({peak_kind})",
            flush=True,
        )
    return alphas, betas, r, cap_stats


def lanczos_unit_kernel(
    units,
    hOp,
    unit_windows,
    weighted_window,
    *,
    reort,
    sparse,
    slaterWeightMin,
    solver_verbose=False,
    print_windows=False,
    print_size=False,
    eval_meshes_for=None,
):
    """The ``run_units_distributed`` kernel for block-Lanczos Green's-function units.

    Returns ``kernel(split_basis, u, seeds) -> (alphas, betas, r_per_state, cap_stats,
    conv_stats)``: the unit's shared block-tridiagonal coefficients, its seed projection split
    into one ``n_ops``-column slice per stacked eigenstate (``r[:, p*n_ops:(p+1)*n_ops]``), and
    the cap and convergence records :func:`_block_green_group` fills.

    Parameters
    ----------
    units : list of GFUnit
        From :func:`gf_units.enumerate_gf_units`.
    hOp : ManyBodyOperator
        The Hamiltonian.
    unit_windows : list
        The excited occupation window per unit (``None`` = unrestricted).
    weighted_window
        The (widened) weighted restrictions shared by every unit.
    reort, sparse, slaterWeightMin
        Passed to :func:`_block_green_group`.
    solver_verbose : bool
        Its ``verbose`` (the per-unit memory line and the solver's own prints).
    print_windows : bool
        Print the unit's windows before solving (the self-energy driver at ``-v``).
    print_size : bool
        Print the unit's excited basis size after solving.
    eval_meshes_for : callable, optional
        ``eval_meshes_for(unit)`` gives the frequencies the convergence monitor tests (see
        :func:`gf_convergence._gf_eval_meshes`); ``None`` leaves it on the Ritz-band fallback.
    """

    def kernel(split_basis, u, seeds):
        unit = units[u]
        unit_rank0 = split_basis.comm is None or split_basis.comm.rank == 0
        if print_windows and unit_rank0 and unit_windows[u] is not None:
            print("Excited restrictions:")
            for indices, occ_rest in unit_windows[u].items():
                print(f"{sorted(indices)}: {occ_rest}")
            sys.stdout.flush()
        if print_windows and unit_rank0 and weighted_window is not None:
            print("weight restrictions:")
            for weights, sum_rest in weighted_window:
                print(f"{weights}: {sum_rest}")
            sys.stdout.flush()
        conv_stats = {}
        alphas, betas, r, cap_stats = _block_green_group(
            split_basis,
            hOp,
            seeds,
            reort,
            unit.delta,
            slaterWeightMin,
            sparse,
            solver_verbose,
            unit_windows[u],
            weighted_window,
            eval_meshes=eval_meshes_for(unit) if eval_meshes_for is not None else None,
            conv_stats=conv_stats,
            unit_label=u,
        )
        if print_size and unit_rank0:
            print(f"Expanded excited state basis contains {cap_stats['retained_size']} elements.")
        # The coefficients, cap and convergence records ride back together: only the ranks of the
        # color that ran this unit see them locally, and the gather of the kernel's return value
        # is the one path that carries every unit's result to rank 0.
        return (
            alphas,
            betas,
            [r[:, p * unit.n_ops : (p + 1) * unit.n_ops] for p in range(len(unit.chunk))],
            cap_stats,
            conv_stats,
        )

    return kernel


def states_by_group(units, results, n_groups, n_states):
    """Reassemble Lanczos unit results by operator group and eigenstate.

    Returns ``[(alphas_list, betas_list, r_list)]`` per operator group, each list indexed by
    eigenstate: stacked eigenstates share their unit's ``alphas``/``betas`` and keep their own
    ``r`` slice (see :func:`lanczos_unit_kernel`). Every (group, eigenstate) pair must be covered.
    """
    acc = [([None] * n_states, [None] * n_states, [None] * n_states) for _ in range(n_groups)]
    for unit, (alphas, betas, r_slices, *_records) in zip(units, results):
        a_list, b_list, r_list = acc[unit.group_i]
        for p, ei in enumerate(unit.chunk):
            a_list[ei], b_list[ei], r_list[ei] = alphas, betas, r_slices[p]
    for g, (a_list, _b, _r) in enumerate(acc):
        missing = [ei for ei, a in enumerate(a_list) if a is None]
        if missing:
            raise RuntimeError(f"operator group {g}: no unit covered eigenstate(s) {missing}")
    return acc


def combine_sides(g_add, g_rem, Z):
    r"""The one-particle Green's function from its two spectral sides: :math:`(G^+ - G^{-T})/Z`.

    ``g_add`` is the thermally summed addition resolvent (seeds :math:`c^\dagger|\psi\rangle`),
    ``g_rem`` the removal one (seeds :math:`c|\psi\rangle`, evaluated at :math:`-z`); both
    ``(n_z, n, n)`` and not yet divided by ``Z``. The removal side enters transposed because its
    seed indices are the bra/ket-swapped pair of the anticommutator's second term.
    """
    return (g_add - np.transpose(g_rem, (0, 2, 1))) / Z
