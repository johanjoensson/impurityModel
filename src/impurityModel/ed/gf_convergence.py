"""Green's-function convergence monitoring: the runtime block-Lanczos convergence gate and its
post-hoc diagnostic counterpart, plus the shared frequency-mesh helpers both use.

Split out of :mod:`greens_function` (which re-exports everything here for backwards
compatibility). Depends only on :mod:`gf_primitives` (the continued-fraction evaluation
``_trim_blocks``/``_block_cf_inverse``) -- nothing here imports from :mod:`greens_function` or
:mod:`gf_shift_recycling`.
"""

import os

import numpy as np

from impurityModel.ed.gf_primitives import _block_cf_inverse, _trim_blocks

# Relative-change convergence floor for the block-Lanczos Green's function, shared by the
# runtime monitor (_make_gf_convergence_monitor) and the post-hoc diagnostic summary
# (_lanczos_convergence_summary) so the two can never disagree -- single source of truth.
#
# 1e-9, not the historical 1e-6. While the monitor sampled the whole Ritz band on the real axis it
# converged a resolvent the caller often never evaluated, and so *over*-delivered: a declared 1e-6
# returned sigma accurate to 1e-13..1e-15. Now that it tests G on the caller's own mesh the
# tolerance means what it says, and leaving the floor at 1e-6 would have quietly turned that into
# ~5e-8. Measured end to end on the 20-bath NiO self-energy, sigma against a deeply-converged
# reference, with the number of Lanczos blocks the run actually produced:
#
#     monitor        floor    Matsubara            real axis            blocks
#     band-wide      1e-6     3.2e-15              2.3e-13                 336
#     caller's mesh  1e-6     2.0e-08              5.2e-08                 158
#     caller's mesh  1e-9     3.5e-11              2.4e-12                 180 / 214
#     caller's mesh  1e-10    3.8e-14              2.4e-12                 214
#
# So 1e-9 is a deliberate trade, not a restoration: against the old *accidental* accuracy it gives
# up ~4 orders on the Matsubara axis and ~10x on the real axis, both still comfortably inside the
# tolerance it now honestly declares, and it needs 1.9x fewer blocks (and 1.9x less retained Q).
# Callers who want the old numbers back should set 1e-10, which costs 1.57x instead.
#
# CAVEAT: that table was measured on `_nio_workload`, which defaults to
# `chargeTransferCorrection=None`. Without a double counting the addition-GF poles sit ~14688 eV
# above E0 while the meshes span |z| <= 4.7, so `G` is *constant* on the frequencies it is evaluated
# at (relative variation 5.0e-08 across the whole Matsubara mesh). Its block counts mean nothing.
#
# Re-measured on the real workloads (the RSPt Hamiltonians in `impmod_tests/*/impurityModel_data.h5`,
# which have `max|Im G| = 19.9` in the window). NiO, 1 bath/orbital, 200 mesh points, reort=partial,
# blocks summed over all (block, side, eigenstate) units:
#
#     axis             band-wide      caller's mesh     sigma agreement
#     Matsubara only   3496 blocks    360 blocks        5.4e-13
#     real axis only   3496 blocks    3479 blocks       3.5e-11
#
# So ~10x on a Matsubara-only self-energy and ~nothing on the real axis -- the shape Phase 3a-bis
# predicted. On antiferromagnetic NiO (matrix-valued G, block widths to 4) the caller's-mesh monitor
# finishes all 240 units in 5.2 s; the band-wide one manages 45 of them in 300 s and is still going
# after two hours, all of it inside `_block_cf_inverse`, this monitor's own O(k^2) rebuild.
#
# The 1e-9 floor stands: it is strictly tighter than the old 1e-6, so nothing can silently degrade.
# See doc/plans/bicgstab_per_frequency_gf.md, Phases 3a-quater and 3a-quinquies.
#
# Note `_gf_rel_tol` takes max(slaterWeightMin**2, this), so this floor -- not the cutoff --
# governs every production slaterWeightMin (1e-5 gives 1e-10, far below it). Only a cutoff looser
# than sqrt(floor) ever overrides it, and then basis truncation is the accuracy limit anyway.
_GF_REL_TOL_FLOOR = 1e-9
# Minimum blocks before the convergence mesh may be frozen (let the extremal Ritz values start
# to settle before we commit to a sampling window).
_GF_MESH_FREEZE_BLOCKS = 3
# Mesh padding fraction AND the per-step edge-growth threshold below which the spectral edges
# (the alpha-diagonal range) count as "settled": we only freeze the mesh once a new block grows
# the range by less than the margin we pad with, and we re-extend the mesh if a later block's
# range escapes the padded window. One constant ties the two so they stay consistent.
_GF_MESH_MARGIN_REL = 0.05
# Consecutive sub-tolerance steps required before declaring convergence -- guards against a
# single accidental small relative change tripping convergence prematurely.
_GF_CONSEC_CONVERGED = 2
# Adaptive convergence-test sampling (see _make_gf_convergence_monitor). The resolvent-change
# test rebuilds the O(k)-level block continued fraction each call -- the single largest cost of
# the block-Lanczos Green's function (~53% of runtime for reort=NONE; measured). During the long
# approach (relative change still far above tolerance, convergence impossible) it is sampled only
# every _GF_CHECK_EVERY blocks; once a check lands within _GF_NEAR_FACTOR x tol it switches to
# every block so the exact convergence point is caught with no added Lanczos steps. The measure
# and tolerance are unchanged, so the converged Green's function is unchanged.
_GF_CHECK_EVERY = int(os.environ.get("GF_CHECK_EVERY", 8))  # set to 1 to disable (check every block)
# Switch from sparse to per-block sampling once the relative change is within this factor of the
# tolerance (i.e. convergence is imminent and must not be sampled coarsely). Kept small: the
# relative change typically sits on a long noisy plateau a decade or two above tolerance before
# the final descent, and that plateau must stay in the sparse regime for the sampling to pay off.
_GF_NEAR_FACTOR = float(os.environ.get("GF_NEAR_FACTOR", 2.0))
# Sample points per requested axis, per eigenstate, when the caller's evaluation mesh is known
# (see _gf_eval_meshes). The production real-axis mesh is ~2000 points and the Matsubara one ~375;
# feeding either to the monitor verbatim would multiply the continued-fraction cost above -- which
# is already the largest single cost of the recurrence -- by more than an order of magnitude. The
# monitor only has to decide *whether* G has stopped moving, so it subsamples. Matches the 64
# points _gf_sample_mesh uses for its spectral-edge fallback, so the cost is unchanged.
_GF_MONITOR_POINTS = 64


def _gf_signed_axes(matsubara_mesh, omega_mesh, side_i, delta):
    r"""The requested frequency axes in the resolvent frame, *before* the thermal-energy shift.

    One complex array per requested axis (Matsubara first): ``sign*mesh + i*sign*axis_delta``,
    where ``sign`` selects addition (+, ``side_i = 0``) vs removal (-, ``side_i = 1``) and the
    broadening applies to the real axis only (a Matsubara mesh already carries its imaginary
    part). The sign multiplies the mesh *and* the broadening so ``Im(z)`` keeps the sign of the
    unit's signed delta. Adding an eigenstate energy ``e`` to any returned axis gives exactly the
    ``omegaP`` frame :func:`calc_G` evaluates in -- the single source of that frame, shared by
    the convergence monitor (:func:`_gf_eval_meshes`, subsampled) and the per-frequency BiCGSTAB
    driver (full axes). Empty list when no mesh was requested.
    """
    sign = 1.0 if side_i == 0 else -1.0
    axes = []
    if matsubara_mesh is not None:
        axes.append((sign * np.asarray(matsubara_mesh)).astype(complex))
    if omega_mesh is not None:
        axes.append((sign * np.asarray(omega_mesh) + 1j * sign * delta).astype(complex))
    return axes


def _gf_eval_meshes(matsubara_mesh, omega_mesh, side_i, delta, es, n_points=_GF_MONITOR_POINTS):
    r"""The frequencies the caller will actually evaluate ``G`` at, in the ``alphas`` frame.

    ``calc_G`` forms :math:`\omega_P = \omega + i\delta + e`, and ``get_Greens_function`` calls it
    with ``(+mesh, +delta)`` for the addition side and ``(-mesh, -delta)`` for removal, with
    ``delta = 0`` on the Matsubara axis. Reproduce exactly that, for every thermal eigenstate the
    unit stacks.

    Returned as one array **per axis**, not one concatenated array, because the convergence measure
    is relative: :math:`\max|\Delta G| / \max|G|` over a mesh spanning both axes would divide the
    real-axis change by the Matsubara peak. At :math:`T = 0.002` and :math:`\delta = 0.2` those
    scales are :math:`1/\pi T \approx 159` and :math:`1/\delta = 5`, so the real axis would be
    declared converged more than an order of magnitude early. The monitor takes the max of the
    per-axis relative changes instead.

    Returns ``None`` when the caller asked for no mesh at all, which sends the monitor back to its
    spectral-edge fallback.
    """
    axes = _gf_signed_axes(matsubara_mesh, omega_mesh, side_i, delta)
    if not axes:
        return None

    per_axis = max(2, n_points // max(1, len(es)))

    meshes = []
    for axis_mesh in axes:
        # Subsampling by index commutes with the (already applied) affine sign/broadening map,
        # so this is exactly the old subsample-then-shift construction.
        sub = (
            axis_mesh
            if len(axis_mesh) <= per_axis
            else axis_mesh[np.linspace(0, len(axis_mesh) - 1, per_axis).astype(int)]
        )
        meshes.append(np.concatenate([sub + e for e in es]))
    return meshes


def _gf_rel_tol(slaterWeightMin):
    """Relative-change convergence tolerance: the basis-truncation floor ``slaterWeightMin**2``
    but never below :data:`_GF_REL_TOL_FLOOR`. Used by both the runtime monitor and the
    diagnostic summary so they apply identical thresholds."""
    return max(slaterWeightMin**2, _GF_REL_TOL_FLOOR)


def _make_gf_convergence_monitor(delta, slaterWeightMin, eval_meshes=None):
    r"""Relative-change convergence monitor for the block-Lanczos Green's function.

    Shared by both GF kernels (``block_green_impl``, ``block_Green_sparse``). Returns
    ``(converged_fn, converged_flag, delta_min, last_dg)`` where ``delta_min`` is the
    convergence tolerance actually used (the single source of truth for the warning
    messages, so they never drift from this declaration): ``converged_fn(alphas, betas,
    verbose, block_widths)`` estimates ``G`` and reports convergence only after
    :data:`_GF_CONSEC_CONVERGED` *consecutive* steps whose relative change
    (:func:`_greens_function_change`, with the cross-step ``gs_cache``) stays below
    ``max(slaterWeightMin**2, 1e-6)``.  Requiring the tolerance to hold for several
    steps in a row guards against a single fluke step.  ``converged_flag[0]`` records
    whether convergence was actually declared, for the non-convergence warning.
    ``last_dg[0]`` is the most recently measured relative change (``None`` if the monitor
    never got to compute one), for callers that report it (e.g. the RIXS R2 summary).

    ``eval_meshes`` (from :func:`_gf_eval_meshes`) is the list of frequency arrays the caller will
    actually evaluate ``G`` on -- one per requested axis, already shifted into the ``alphas`` frame.
    Given it, the monitor tests convergence *there*, and takes the **max** of the per-axis relative
    changes so neither axis can mask the other.

    Without it the monitor falls back to an *adaptively* frozen mesh spanning the resolved Ritz
    band on the line :math:`\omega + i\delta` (:func:`_gf_converged_mesh`) -- frozen once the
    spectral edges settle and re-extended if a later block escapes the window. That fallback
    converges the **real-axis** resolvent at broadening ``delta`` whether or not a real-axis mesh
    was requested, and a Matsubara point :math:`i\omega_n` sits a distance
    :math:`\sqrt{E_k^2 + \omega_n^2}` from every pole while a real-axis point can come within
    ``delta`` of one. So a Matsubara-only self-energy was being charged for a resolvent it never
    evaluates: measured 3.6-4.1x more blocks than it needs, against 1.2-1.4x when the real axis is
    also requested. It remains the right behaviour for a caller that supplies no mesh.
    """
    delta_min = _gf_rel_tol(slaterWeightMin)
    converged_flag = [False]
    mesh_cache = [None, -1]  # [mesh, frozen_block_count]; frozen_block_count detects (re)freezes
    gs_cache = [None, 0]
    consec = [0]  # consecutive sub-tolerance steps on the current (stable) mesh
    step = [0]  # block count since the mesh froze, for the adaptive sampling gate
    last_dg = [None]  # most recent relative change, to decide sparse vs dense sampling
    # One continued-fraction cache per axis: the caller's meshes never move, so the cross-step
    # reuse in _greens_function_change is always valid and never needs the freeze bookkeeping.
    axis_caches = [[None, 0] for _ in (eval_meshes or ())]

    def converged_on_eval_meshes(alphas, betas, verbose, block_widths):
        d_g = 0.0
        for mesh, cache in zip(eval_meshes, axis_caches):
            d = _greens_function_change(alphas, betas, block_widths, delta, omegaP=mesh, cache=cache)
            if d is None:  # spurious (wrong-sign) imaginary part on this axis -> not converged
                return None
            d_g = max(d_g, d)
        return d_g

    def converged(alphas, betas, verbose=False, block_widths=None, **kwargs):
        if len(alphas) <= 1:
            return False
        # B6 adaptive sampling. The resolvent-change test rebuilds an O(k)-level block continued
        # fraction every call, so running it on every block is O(k^2) per GF invocation and is the
        # single largest cost for reort=NONE (~53% measured). But the test also *terminates* the
        # recurrence, so simply running it less often delays convergence and adds (more expensive)
        # Lanczos steps that cancel the saving. Instead: during the long approach, where the change
        # is still far above tolerance and convergence is impossible, sample only every
        # _GF_CHECK_EVERY blocks; once a check lands within _GF_NEAR_FACTOR x tol, switch to every
        # block so the precise convergence point and the _GF_CONSEC_CONVERGED gate are detected with
        # no delay. Same convergence measure/tolerance -> same converged Green's function.
        step[0] += 1
        near = last_dg[0] is not None and last_dg[0] < _GF_NEAR_FACTOR * delta_min
        if not near and (step[0] % _GF_CHECK_EVERY) != 0:
            return False
        if eval_meshes is not None:
            # The caller's mesh is fixed from the start, so there is nothing to freeze or
            # re-extend: no spectral-edge warm-up, no cache resets.
            d_g = converged_on_eval_meshes(alphas, betas, verbose, block_widths)
        else:
            A_trim, _ = (
                _trim_blocks(alphas, betas, block_widths)
                if (block_widths is not None and len(block_widths) == len(alphas))
                else ([np.asarray(a) for a in alphas], None)
            )
            res = _gf_converged_mesh(A_trim, delta)
            if res is None:  # spectral edges have not settled yet -> keep building
                return False
            mesh, frozen = res
            if frozen != mesh_cache[1]:
                # First freeze or a re-extension changed the mesh: the cross-step resolvent cache
                # and the consecutive-step count are measured against the old window, so reset both
                # and re-confirm convergence on the new one.
                mesh_cache[0], mesh_cache[1] = mesh, frozen
                gs_cache[0], gs_cache[1] = None, 0
                consec[0] = 0
                last_dg[0] = None
            d_g = _greens_function_change(alphas, betas, block_widths, delta, omegaP=mesh_cache[0], cache=gs_cache)
        if d_g is None:  # spurious (wrong-sign) imaginary part -> not converged
            consec[0] = 0
            last_dg[0] = None
            return False
        last_dg[0] = d_g
        if verbose:
            print(rf"$\delta$ = {d_g}", flush=True)
        consec[0] = consec[0] + 1 if d_g < delta_min else 0
        is_conv = consec[0] >= _GF_CONSEC_CONVERGED
        converged_flag[0] = converged_flag[0] or is_conv
        return is_conv

    return converged, converged_flag, delta_min, last_dg


def _greens_function_change(alphas, betas, block_widths, delta, omegaP=None, cache=None):
    r"""Relative change in the block resolvent when the last Lanczos block is added.

    Block-Lanczos convergence monitor for the Green's function.  It compares the
    seed-block resolvent :math:`G = (G^{-1}_0)^{-1}` built from all ``k`` blocks against
    the one from the first ``k-1`` blocks, on sample frequencies drawn from the (trimmed)
    diagonal blocks (the broadened Ritz values — where the spectral weight sits).  The
    measure is the *relative* change

    .. math::

        d_g = \frac{\max_\omega \lVert G_k(\omega) - G_{k-1}(\omega)\rVert}
                   {\max_\omega \lVert G_k(\omega)\rVert},

    so it is scale-invariant and reflects the spectral function the self-energy actually
    needs — unlike the absolute change in :math:`G^{-1}_0`, whose leading
    :math:`\omega I - \alpha_0` term (:math:`\sim\lvert\omega\rvert`) is identical between
    the two and which therefore never decays to a tight absolute tolerance even when the
    spectrum is fully resolved.  Shrinking-block deflation is handled by trimming to
    ``block_widths`` first, so no fixed block dimension is assumed.

    The optional ``cache`` (a ``[gs_value, n_blocks]`` list) lets the convergence loop reuse
    work: this step's ``G^{-1}`` over ``k-1`` blocks is *identical* to the previous step's
    over ``k-1`` blocks on the same frozen mesh, so the ``gs_prev`` continued fraction is taken
    from the cache instead of rebuilt — halving the per-step continued-fraction cost. Exact
    (no behavior change); pass the same list each step.

    Returns:
        float or None: the relative change, or ``None`` if the freshly added block yields
        a wrong-sign spectral weight (not yet stabilized).
    """
    if block_widths is not None and len(block_widths) == len(alphas):
        A, B = _trim_blocks(alphas, betas, block_widths)
    else:
        A = [np.asarray(alphas[i]) for i in range(len(alphas))]
        B = [np.asarray(betas[i]) for i in range(len(betas))]
    if omegaP is None:
        # Default (back-compat / standalone use): sample at the current Ritz values. The
        # convergence loop should instead pass a *frozen* mesh (see _gf_sample_mesh): the
        # Ritz set grows every step, so each new block adds a pole at a fresh sample point
        # and the change never decays — measuring on a fixed mesh is what converges.
        n0 = A[0].shape[0]
        ws = np.concatenate([np.diagonal(a) for a in A])[: 15 * n0]
        omegaP = ws.real + delta * 1j
    gs_new = _block_cf_inverse(A, B, omegaP)
    if cache is not None and cache[0] is not None and cache[1] == len(A) - 1:
        gs_prev = cache[0]  # == previous step's gs_new (CF over the same k-1 blocks/mesh)
    else:
        gs_prev = _block_cf_inverse(A[:-1], B[:-1], omegaP)
    if cache is not None:
        cache[0], cache[1] = gs_new, len(A)
    if np.any(np.diagonal(gs_new.imag, axis1=1, axis2=2) * np.sign(delta) < 0):
        return None
    # Compare the resolvents G = (G^{-1})^{-1}, not their inverses: the broadening
    # (Im omega = delta) keeps G^{-1} non-singular, so the inverse is well defined.
    G_new = np.linalg.inv(gs_new)
    G_prev = np.linalg.inv(gs_prev)
    scale = np.max(np.abs(G_new))
    return np.max(np.abs(G_new - G_prev)) / max(scale, np.finfo(float).tiny)


def _lanczos_convergence_summary(alphas_list, betas_list, delta, tol=_GF_REL_TOL_FLOOR):
    r"""Post-hoc block-Lanczos convergence summary over the per-thermal-state coefficients.

    Avoids threading run-time monitor state out of ``block_Green_sparse``: for each thermal
    state's trimmed ``(alphas, betas)`` it re-evaluates the final relative resolvent change via
    :func:`_greens_function_change`.  To agree with the run-time monitor's verdict (and not
    raise spurious "not converged" warnings) it mirrors that monitor exactly:

    * **Same frozen mesh.**  The monitor freezes its sample mesh after the first
      ``_GF_MESH_FREEZE_BLOCKS`` blocks; here we rebuild it from those same leading blocks
      (``A[:_GF_MESH_FREEZE_BLOCKS]``), not from the full final Ritz range -- otherwise the
      last block's spectral-edge contribution registers as a large change on a wider mesh the
      monitor never used.
    * **Invariant subspace == exact.**  When the recurrence terminated on an invariant subspace
      the trailing coupling block vanished (``betas[-1] ~ 0``); the continued fraction is then
      exact regardless of the relative-change value, so that state counts as converged.  (A run
      that stopped on the tolerance instead has a normal, nonzero trailing ``beta``.)

    Args:
        alphas_list, betas_list: Per-thermal-state lists of trimmed Lanczos blocks.
        delta: Broadening used to place the frozen sample mesh off the real axis.
        tol: Relative-change threshold below which a state counts as converged. Defaults to
            :data:`_GF_REL_TOL_FLOOR`; pass :func:`_gf_rel_tol` (slaterWeightMin) to match the
            monitor's basis-truncation floor.

    Returns:
        tuple[bool, float, int]: ``(all_converged, worst_final_change, max_blocks)``.
    """
    worst = 0.0
    max_blocks = 0
    all_converged = True
    for A, B in zip(alphas_list, betas_list):
        A = list(A)
        B = list(B)
        max_blocks = max(max_blocks, len(A))
        if len(A) < _GF_MESH_FREEZE_BLOCKS:  # invariant subspace reached almost immediately -> exact
            continue
        # Invariant subspace: trailing coupling vanished -> exact (matches the kernel's
        # "invariant_subspace" status, which the run-time monitor treats as converged).
        tail = np.asarray(B[-1]) if len(B) else np.zeros(0)
        scale = max((float(np.linalg.norm(np.asarray(a), 2)) for a in A), default=1.0)
        if tail.size == 0 or float(np.linalg.norm(tail, 2)) <= _GF_REL_TOL_FLOOR * max(scale, 1.0):
            continue
        # Use the *same* adaptively-frozen mesh the run-time monitor settled on (pure function of
        # the coefficients, so the two agree). If the edges never settled the run is genuinely
        # under-resolved.
        res = _gf_converged_mesh(A, delta if delta else 1.0)
        if res is None:
            all_converged = False
            continue
        mesh, _ = res
        d_g = _greens_function_change(A, B, None, delta if delta else 1.0, omegaP=mesh)
        if d_g is None:
            all_converged = False
            continue
        worst = max(worst, float(d_g))
        if d_g >= tol:
            all_converged = False
    return all_converged, worst, max_blocks


def _gf_sample_mesh(alphas, delta, n_points=64):
    r"""Frozen real-frequency mesh for the block-Lanczos Green's-function convergence test.

    Spans the current Ritz range (the diagonal entries of the ``alphas`` blocks — Lanczos
    resolves the spectral *edges* within a few blocks) padded by a margin, on the line
    :math:`\omega + i\,\mathrm{sign}(\delta)\,\lvert\delta\rvert`.  The caller builds this
    once and reuses it, so the convergence measure is evaluated at *fixed* frequencies and
    actually decays as the spectrum fills in.
    """
    ws = np.concatenate([np.real(np.diagonal(np.asarray(a))) for a in alphas])
    mesh, _, _ = _gf_mesh_from_range(float(np.min(ws)), float(np.max(ws)), delta, n_points)
    return mesh


def _gf_mesh_from_range(lo, hi, delta, n_points=64):
    r"""Build the convergence sample mesh covering ``[lo, hi]`` padded by the margin, on the line
    :math:`\omega + i\,\delta`.  Returns ``(mesh, ext_lo, ext_hi)`` where ``[ext_lo, ext_hi]`` is
    the padded extent actually sampled (used by :func:`_gf_converged_mesh` to detect a later
    block whose spectral edge escapes the window)."""
    span = hi - lo
    margin = _GF_MESH_MARGIN_REL * span + 10.0 * abs(delta)
    ext_lo, ext_hi = lo - margin, hi + margin
    return np.linspace(ext_lo, ext_hi, n_points) + 1j * delta, ext_lo, ext_hi


def _gf_diag_range(alpha):
    """Min/max of the real alpha-block diagonal (the per-direction Rayleigh quotients ~ the
    spectral edges resolved so far)."""
    d = np.real(np.diagonal(np.asarray(alpha)))
    return float(np.min(d)), float(np.max(d))


def _gf_converged_mesh(alphas, delta):
    r"""Adaptively-frozen convergence mesh, as a *pure* function of the (trimmed) alpha blocks so
    the runtime monitor and the post-hoc summary settle on exactly the same window.

    Block-Lanczos resolves the spectral edges from the inside out, so a mesh frozen after a fixed
    handful of blocks can be too narrow.  Instead we scan the blocks, tracking the cumulative
    alpha-diagonal range ``[lo, hi]`` (the resolved edges), and:

    * **Freeze** once at least :data:`_GF_MESH_FREEZE_BLOCKS` blocks are in *and* the range grew
      by less than :data:`_GF_MESH_MARGIN_REL` in the last block (edges settled within the margin
      we pad with).
    * **Re-extend** afterwards if a later block's range escapes the padded window, rebuilding the
      mesh around the new, wider range.

    Returns ``(mesh, frozen_blocks)`` -- ``frozen_blocks`` is the block count at the last (re)freeze,
    so a caller can detect when the mesh changed -- or ``None`` if the edges have not settled yet.
    """
    if len(alphas) < _GF_MESH_FREEZE_BLOCKS:
        return None
    lo = hi = None
    prev_span = None
    mesh = None
    ext_lo = ext_hi = None
    frozen = 0
    for k, alpha in enumerate(alphas):
        cur_lo, cur_hi = _gf_diag_range(alpha)
        lo = cur_lo if lo is None else min(lo, cur_lo)
        hi = cur_hi if hi is None else max(hi, cur_hi)
        span = hi - lo
        if mesh is None:
            if k + 1 >= _GF_MESH_FREEZE_BLOCKS and prev_span is not None:
                growth = (span - prev_span) / max(span, np.finfo(float).tiny)
                if growth < _GF_MESH_MARGIN_REL:
                    mesh, ext_lo, ext_hi = _gf_mesh_from_range(lo, hi, delta)
                    frozen = k + 1
            prev_span = span
        elif lo < ext_lo or hi > ext_hi:  # a later edge escaped the sampled window -> re-extend
            mesh, ext_lo, ext_hi = _gf_mesh_from_range(lo, hi, delta)
            frozen = k + 1
    if mesh is None:
        return None
    return mesh, frozen
