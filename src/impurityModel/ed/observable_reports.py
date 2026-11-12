"""Ground-state observable reports: the ``compute_*`` summaries and ``print_*`` tables built on
:mod:`impurityModel.ed.observables`, which keeps the ρ-based values and the operator builders."""

import numpy as np

from impurityModel.ed.block_structure import get_equivalent_blocks
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.observables import (
    block_group_labels,
    casimir_to_quantum_number,
    compute_shell_observables,
    gated_term_symbol,
    get_Sz_from_rho_pairs,
    lande_g_and_moments,
    make_spin_operators,
    manifold_observable_values,
    spin_correlation_operator,
    term_symbol,
    thermal_observable_value,
)


def print_impurity_orbital_groups(equivalent_blocks, block_structure, impurity_orbitals=None, file=None):
    """Print the legend mapping each ``N(...)`` column label to its global orbital indices.

    The labels are the ``impurity_orbitals`` group keys (see :func:`block_group_labels`), the
    same names the occupation search uses, so this line and
    :func:`solver_basis.prepare_solver_basis`'s legend describe the same groups.

    A no-op when there is only one group (nothing to disambiguate).
    """
    _, legend = block_group_labels(equivalent_blocks, block_structure, impurity_orbitals)
    if len(legend) <= 1:
        return
    entries = [f"{label}: {orbs}" for label, orbs in legend]
    print(f"Impurity orbital groups:  {'  '.join(entries)}", file=file)


def print_expectation_values(
    rhos,
    es,
    rot_to_spherical,
    block_structure,
    s_values=None,
    l_values=None,
    j_values=None,
    sisb_values=None,
    sisb_z_values=None,
    impurity_orbitals=None,
    shell_ls=None,
):
    """
    print several expectation values, e.g. E, N, L^2.

    If ``s_values`` / ``l_values`` / ``j_values`` are given (one impurity ``S`` / ``L``
    / ``J`` quantum number per eigenstate, e.g. from :func:`manifold_observable_values`
    with :func:`make_impurity_casimir_operators` + :func:`casimir_to_quantum_number`),
    the corresponding columns are appended; ``sisb_z_values`` (the longitudinal
    ``<Sz_imp Sz_bath>``, reported for a collinear spin-polarized bath) adds a
    ``Szi.Szb`` column. When all are ``None`` the output is identical to before (used
    when the eigenstates are not available, e.g. on non-root ranks).

    ``impurity_orbitals`` (``Basis.impurity_orbitals``, optional) makes the ``N``/``Lz``/
    ``Sz``/``L.S`` columns shell-aware via :func:`compute_shell_observables`: for a
    multi-shell impurity (e.g. a 2p+3d core-hole model) each shell's own contribution is
    evaluated separately and summed, instead of misinferring a single ``l`` from the
    concatenated block. ``None`` (the default) reproduces today's single-shell behaviour
    exactly.
    """
    equivalent_blocks = get_equivalent_blocks(block_structure)
    print(f"E0 = {es[0]:9.6f}")
    block_labels, _ = block_group_labels(equivalent_blocks, block_structure, impurity_orbitals)
    block_N_string = [f"N({label})" for label in block_labels]
    # Each block-occupation column is right-aligned to a width that fits both its header
    # and the 7-8 char ``.5f`` value below it, so header and numbers line up.
    block_N_widths = [max(len(Ns), 8) for Ns in block_N_string]
    block_N_string_formatted = [f"{Ns:>{w}s}" for Ns, w in zip(block_N_string, block_N_widths)]
    extra = [
        (name, vals)
        for name, vals in (
            ("S", s_values),
            ("L", l_values),
            ("J", j_values),
            ("Si.Sb", sisb_values),
            ("Szi.Szb", sisb_z_values),
        )
        if vals is not None
    ]
    # Lz/Sz/L.S and the S/L/J/Si.Sb columns are printed with the space-flag format
    # ``{x: 8.6f}``, which is 9 characters wide (the sign column sits on top of the 8),
    # so their headers must be 9 wide to line up with the numbers below.
    extra_header = "".join(f"  {name:>9s}" for name, _ in extra)
    print(
        f"{'i':>3s}  {'E-E0':>11s}  {'N':>8s}  {'N(Dn)':>8s}  {'N(Up)':>8s}  "
        f"{'  '.join(block_N_string_formatted)}  {'Lz':>9s}  {'Sz':>9s}  {'L.S':>9s}{extra_header}"
    )
    for i, (e, rho) in enumerate(zip(es - es[0], rhos)):
        block_occs = [
            np.sum(np.diag(rho)[[orb for block in blocks for orb in block_structure.blocks[block]]])
            for blocks in equivalent_blocks
        ]
        block_occ_string_formatted = ["" for _ in block_occs]
        for ib, b_occ in enumerate(block_occs):
            block_occ_string_formatted[ib] = f"{np.real(b_occ):>{block_N_widths[ib]}.5f}"
        totals = compute_shell_observables(rho, rot_to_spherical, impurity_orbitals, shell_ls=shell_ls)["total"]
        N, Ndn, Nup = totals["n"], totals["n_dn"], totals["n_up"]
        Lz, Sz, LS = totals["lz"], totals["sz"], totals["l_dot_s"]
        extra_fields = "".join(f"  {vals[i]: 8.6f}" for _, vals in extra)
        print(
            f"{i:>3d}  {e:11.8f}  {N:8.5f}  {Ndn:8.5f}  {Nup:8.5f}  "
            f"{'  '.join(block_occ_string_formatted)}  {Lz: 8.6f}  {Sz: 8.6f}  {LS: 8.6f}{extra_fields}"
        )
    print("\n")


#: An orbital pair whose both occupations are within this of 0 (empty) or 1 (filled) carries no correlation.
OPEN_SHELL_TOL = 1e-3


def compute_correlation_diagnostics(
    psis, es, tau, thermal_rho, imp_pairs, comm=None, redistribute=None, keep=None, values=None, pairing_verified=True
):
    r"""Correlation-strength diagnostics of the impurity (Mott/Hund physics).

    Per impurity spatial orbital ``a`` (one ``(dn, up)`` pair each): the double occupancy
    :math:`d_a = \langle n_{a\uparrow} n_{a\downarrow}\rangle` and the local moment
    :math:`\langle m_z^2\rangle_a = (\langle n_\uparrow\rangle + \langle n_\downarrow\rangle
    - 2 d_a)/4` (using :math:`n^2 = n`). For the whole impurity: :math:`\langle S_z^2\rangle`,
    the static longitudinal susceptibility
    :math:`\chi_{zz} = (\langle S_z^2\rangle - \langle S_z\rangle^2)/\tau`, and the
    inter-orbital Hund matrix :math:`\langle \mathbf S_a\cdot\mathbf S_b\rangle` (diagonal
    :math:`\langle S_a^2\rangle = 3\langle m_z^2\rangle_a`).

    All two-body values are evaluated per state with :func:`manifold_observable_values`
    (manifold-safe) and thermally averaged. Collective when ``comm`` is given.

    Parameters
    ----------
    psis : ManyBodyState
        Eigenstates as one shared-support block (rank-local when distributed), one
        column per state.
    es, tau
        Energies, thermal energy scale.
    thermal_rho : np.ndarray
        Full thermally-averaged one-particle density matrix (replicated).
    imp_pairs : list of (int, int)
        Validated impurity ``(dn, up)`` spin-orbital pairs.
    comm, redistribute
        As in :func:`manifold_observable_values`.
    keep : KeyFilter, optional
        Support filter of ``psis`` (:func:`~impurityModel.ed.basis_transcription.comm_support_filter`):
        each ``O|n>`` is generated only on the eigenstates' support, which is all that is read.

    values : callable, optional
        A stand-in for :func:`manifold_observable_values` (same call signature): a replay of values measured
        while the states were alive, so this function can run without them. ``None`` evaluates them here.
    pairing_verified : bool, optional
        Whether the spin layout grade (``su2`` / ``collinear``) establishes that ``imp_pairs`` is the physical dn/up
        pairing. Under ``sz_only`` it is only known that ``H`` conserves ``S_z`` with this pairing; the
        double occupancies, ``<m_z^2>`` and the Hund matrix then depend on an unverified assumption. Stored in the
        result as ``pairing_verified`` and flagged by :func:`print_correlation_diagnostics`.

    Returns
    -------
    dict
    """
    if values is None:

        def values(op):
            return manifold_observable_values(psis, es, op, comm=comm, redistribute=redistribute)

    n_sp = len(imp_pairs)
    docc_values = np.empty((n_sp, len(es)))
    for a, (dn, up) in enumerate(imp_pairs):
        op = ManyBodyOperator({((up, "c"), (dn, "c"), (dn, "a"), (up, "a")): 1.0})
        docc_values[a] = values(lambda blk, _op=op: _op.apply_block(blk, 0, keep))
    docc_thermal = np.array([np.real(thermal_observable_value(docc_values[a], es, tau)) for a in range(n_sp)])
    n_dn = np.array([np.real(thermal_rho[dn, dn]) for dn, _ in imp_pairs])
    n_up = np.array([np.real(thermal_rho[up, up]) for _, up in imp_pairs])
    mz2 = (n_dn + n_up - 2.0 * docc_thermal) / 4.0

    ops_imp = make_spin_operators(imp_pairs)
    sz2_op = spin_correlation_operator(ops_imp, ops_imp, z_only=True)
    sz2_values = values(lambda blk: sz2_op.apply_block(blk, 0, keep))
    sz2_thermal = float(np.real(thermal_observable_value(sz2_values, es, tau)))
    sz_thermal = get_Sz_from_rho_pairs(thermal_rho, imp_pairs)
    filled = (n_dn > 1.0 - OPEN_SHELL_TOL) & (n_up > 1.0 - OPEN_SHELL_TOL)
    empty = (n_dn < OPEN_SHELL_TOL) & (n_up < OPEN_SHELL_TOL)
    open_shell = ~(filled | empty)

    hund = np.diag(3.0 * mz2)
    orbital_ops = [make_spin_operators([pair]) for pair in imp_pairs]
    for i in range(n_sp):
        for j in range(i + 1, n_sp):
            corr_op = spin_correlation_operator(orbital_ops[i], orbital_ops[j])
            vals = values(lambda blk, _op=corr_op: _op.apply_block(blk, 0, keep))
            hund[i, j] = hund[j, i] = float(np.real(thermal_observable_value(vals, es, tau)))

    return {
        "pairs": [(int(dn), int(up)) for dn, up in imp_pairs],
        "n_dn": n_dn,
        "n_up": n_up,
        "docc": docc_thermal,
        "docc_values": np.real(docc_values),
        "open_shell": [bool(x) for x in open_shell],
        "filled": [int(a) for a in np.flatnonzero(filled)],
        "empty": [int(a) for a in np.flatnonzero(empty)],
        "docc_open_shell_total": float(np.sum(docc_thermal[open_shell])),
        "local_moment_z2": mz2,
        "sz2_values": np.real(sz2_values),
        "sz2_thermal": sz2_thermal,
        "sz_thermal": sz_thermal,
        "tau": float(tau),
        "hund": hund,
        "pairing_verified": bool(pairing_verified),
    }


def print_correlation_diagnostics(corr, file=None):
    """Pretty-print :func:`compute_correlation_diagnostics`: open-shell orbitals in full, the rest folded."""
    print("Impurity correlation diagnostics (thermal):", file=file)
    if not corr.get("pairing_verified", True):
        print("  (dn,up) pairing NOT verified by H (grade S_z only): the values below assume it.", file=file)
    print(f"  {'orbital(dn,up)':>16s} {'n_dn':>8s} {'n_up':>8s} {'<n_up*n_dn>':>12s} {'<m_z^2>':>9s}", file=file)
    open_shell = corr["open_shell"]
    for a, (dn, up) in enumerate(corr["pairs"]):
        if not open_shell[a]:
            continue
        pair_label = f"({dn},{up})"
        print(
            f"  {pair_label:>16s} {corr['n_dn'][a]:>8.5f} {corr['n_up'][a]:>8.5f} "
            f"{corr['docc'][a]:>12.6f} {corr['local_moment_z2'][a]:>9.5f}",
            file=file,
        )
    for name in ("filled", "empty"):
        if corr[name]:
            labels = ", ".join(f"({corr['pairs'][a][0]},{corr['pairs'][a][1]})" for a in corr[name])
            print(f"  {len(corr[name])} {name} orbitals (no correlation): {labels}", file=file)
    print(f"  open-shell double occupancy D = {corr['docc_open_shell_total']:.6f}", file=file)
    print(f"  <Sz_imp^2> = {corr['sz2_thermal']:.6f}   (tau = {corr['tau']:.4g})", file=file)
    keep = [a for a, o in enumerate(open_shell) if o]
    print("  Inter-orbital <S_i.S_j> (open-shell impurity orbitals, labelled by their (dn,up) pairs):", file=file)
    for i in keep:
        label = f"({corr['pairs'][i][0]},{corr['pairs'][i][1]})"
        print(f"  {label:>10s} " + " ".join(f"{np.real(corr['hund'][i][j]): 9.5f}" for j in keep), file=file)


def compute_screening_diagnostics(
    psis,
    es,
    tau,
    thermal_rho,
    imp_pairs,
    bath_pairs,
    h1,
    imp_groups=None,
    z_only=False,
    comm=None,
    redistribute=None,
    max_bath_correlation_levels=200,
    keep=None,
    values=None,
):
    r"""Channel- and bath-level-resolved impurity-bath spin correlations (Kondo screening).

    Two resolutions of :math:`\langle\mathbf S_\mathrm{imp}\cdot\mathbf S_\mathrm{bath}\rangle`:

    - **per impurity channel**: ``imp_groups`` maps a label (e.g. the equivalent-block
      group of the ``N(...)`` columns) to a subset of impurity pairs; one correlation per
      group tells *which* orbital channel is screened;
    - **per bath level**: one row per bath ``(dn, up)`` pair with its on-site energies,
      thermal occupations, hybridization strengths and (when the level count does not
      exceed ``max_bath_correlation_levels``) its :math:`\langle\mathbf S_\mathrm{imp}
      \cdot\mathbf S_b\rangle` — the screening cloud resolved over the bath spectrum.

    With ``z_only=True`` (collinear spin-polarized bath) the longitudinal part
    :math:`\langle S^z_\mathrm{imp} S^z_b\rangle` is evaluated instead — exact under the
    label-only validation, consistent with the flagged full ``<S_imp.S_bath>``.

    Parameters
    ----------
    h1 : np.ndarray
        One-body Hamiltonian matrix (``extract_tensors(..., two_body=False)``).
    imp_groups : dict or None
        ``{label: [pair, ...]}``; ``None`` skips the channel resolution.
    keep : KeyFilter, optional
        Support filter of ``psis``, as in :func:`compute_correlation_diagnostics`.

    values : callable, optional
        A stand-in for :func:`manifold_observable_values` (same call signature): a replay of values measured
        while the states were alive, so this function can run without them. ``None`` evaluates them here.

    Returns
    -------
    dict
    """
    if values is None:

        def values(op):
            return manifold_observable_values(psis, es, op, comm=comm, redistribute=redistribute)

    def corr_op(ops_left, ops_right):
        return spin_correlation_operator(ops_left, ops_right, z_only=z_only)

    imp_orbs = sorted(orb for pair in imp_pairs for orb in pair)

    channels = []
    if imp_groups:
        ops_bath = make_spin_operators(bath_pairs)
        for label, pairs in imp_groups.items():
            if not pairs:
                continue
            op_g = corr_op(make_spin_operators(pairs), ops_bath)
            vals = values(lambda blk, _op=op_g: _op.apply_block(blk, 0, keep))
            channels.append((label, float(np.real(thermal_observable_value(vals, es, tau)))))

    with_correlation = len(bath_pairs) <= max_bath_correlation_levels
    ops_imp = make_spin_operators(imp_pairs)
    levels = []
    for dn, up in bath_pairs:
        row = {
            "pair": (int(dn), int(up)),
            "eps_dn": float(np.real(h1[dn, dn])),
            "eps_up": float(np.real(h1[up, up])),
            "n_dn": float(np.real(thermal_rho[dn, dn])),
            "n_up": float(np.real(thermal_rho[up, up])),
            "v_dn": float(np.linalg.norm(h1[imp_orbs, dn])),
            "v_up": float(np.linalg.norm(h1[imp_orbs, up])),
        }
        if with_correlation:
            op_b = corr_op(ops_imp, make_spin_operators([(dn, up)]))
            vals = values(lambda blk, _op=op_b: _op.apply_block(blk, 0, keep))
            row["sisb"] = float(np.real(thermal_observable_value(vals, es, tau)))
        levels.append(row)
    levels.sort(key=lambda r: 0.5 * (r["eps_dn"] + r["eps_up"]))

    return {"z_only": bool(z_only), "channels": channels, "levels": levels, "with_correlation": with_correlation}


def collapse_screening_levels(levels, tol=1e-6):
    """Bath rows that agree in every number (to ``tol``) fold into one ``(first_row, count)``, energy order kept.

    Symmetry-equivalent bath levels (the t2g triplet, the eg pair) carry identical numbers; a class whose
    members disagree in any column stays expanded, which is how a broken symmetry shows.
    """
    out = {}
    for row in levels:
        key = tuple(
            round(float(row[k]) / tol)
            for k in ("eps_dn", "eps_up", "n_dn", "n_up", "v_dn", "v_up")
            + (("sisb",) if row.get("sisb") is not None else ())
        )
        if key in out:
            out[key][1] += 1
        else:
            out[key] = [row, 1]
    return [(row, count) for row, count in out.values()]


def print_screening_diagnostics(scr, file=None):
    """Pretty-print the dict from :func:`compute_screening_diagnostics`."""
    corr_label = "Sz_imp.Sz_b" if scr["z_only"] else "S_imp.S_b"
    print("Screening channels (thermal):", file=file)
    if scr["z_only"]:
        print("  (spin-polarized bath: longitudinal z-parts only, exact: only S_z is needed)", file=file)
    for label, value in scr["channels"]:
        name = f"<Sz_imp({label}).Sz_bath>" if scr["z_only"] else f"<S_imp({label}).S_bath>"
        print(f"  {name} = {value: .6f}", file=file)
    header = (
        f"  {'bath(dn,up)':>14s} {'eps_dn':>9s} {'eps_up':>9s} {'n_dn':>8s} {'n_up':>8s} {'|V|_dn':>8s} {'|V|_up':>8s}"
    )
    if scr["with_correlation"]:
        header += f" {f'<{corr_label}>':>13s}"
    print("  Bath levels (sorted by energy):", file=file)
    print(header, file=file)
    uncoupled = [row for row in scr["levels"] if max(row["v_dn"], row["v_up"]) < 1e-8]
    classes = collapse_screening_levels([row for row in scr["levels"] if row not in uncoupled])
    for row, count in classes:
        pair_label = "({},{})".format(*row["pair"])
        line = (
            f"  {pair_label:>14s} {row['eps_dn']:>9.4f} {row['eps_up']:>9.4f} "
            f"{row['n_dn']:>8.5f} {row['n_up']:>8.5f} {row['v_dn']:>8.4f} {row['v_up']:>8.4f}"
        )
        if scr["with_correlation"]:
            line += f" {row['sisb']:>13.6f}"
        print(line + (f"   x{count}" if count > 1 else ""), file=file)
    if uncoupled:
        print(f"  {len(uncoupled)} bath level(s) with |V| = 0 (not coupled to the impurity) not listed", file=file)


def compute_magnetic_summary(
    rho_imp, rot_to_spherical, s2=None, l2=None, j2=None, impurity_orbitals=None, j_sharp=True, shell_ls=None
):
    r"""JSON-able summary of the magnetism/multiplet observables (A-bundle).

    Same underlying helpers as :func:`print_thermal_expectation_values`; used to persist
    the values into the ground-state statistics JSON.

    Parameters
    ----------
    rho_imp : np.ndarray
        Impurity block of the (thermal) density matrix, computational basis.
    rot_to_spherical : np.ndarray
        Spherical -> computational rotation for the impurity block.
    s2, l2, j2 : float, optional
        Thermal Casimir expectation values; when all are given the term symbol and
        Landé/effective moments are included.
    impurity_orbitals : dict, optional
        ``Basis.impurity_orbitals``; makes ``lz``/``sz``/etc shell-aware via
        :func:`compute_shell_observables` -- see :func:`print_expectation_values`.
    j_sharp : bool, default True
        From :func:`is_j_sharp`; gates ``term``/``g_j``/``mu_eff`` via
        :func:`gated_term_symbol`, matching the printed report -- see
        :func:`print_thermal_expectation_values`. The persisted JSON must agree with what
        is printed, so this defaults to ``True`` (ungated) only for callers that have not
        computed ``j_sharp``, not as a "usually fine" assumption.

    Returns
    -------
    dict
    """
    totals = compute_shell_observables(rho_imp, rot_to_spherical, impurity_orbitals, shell_ls=shell_ls)["total"]
    out = {
        "lz": float(totals["lz"]),
        "sz": float(totals["sz"]),
        "m_z": float(totals["m_z"]),
        "j_z": float(totals["j_z"]),
        "t_z": float(totals["t_z"]),
        "l_dot_s": float(totals["l_dot_s"]),
    }
    if s2 is not None and l2 is not None and j2 is not None:
        _, _, mu_spin = lande_g_and_moments(s2, l2, j2)
        term, g_j, mu_eff = gated_term_symbol(s2, l2, j2, j_sharp)
        out.update(
            {
                "s2": float(np.real(s2)),
                "l2": float(np.real(l2)),
                "j2": float(np.real(j2)),
                "term": term,
                "g_j": g_j,
                "mu_eff": mu_eff,
                "mu_spin_only": mu_spin,
            }
        )
    return out


def compute_state_summary(
    rhos,
    es,
    rot_to_spherical,
    s_values=None,
    l_values=None,
    j_values=None,
    entanglement=None,
    impurity_orbitals=None,
    shell_ls=None,
):
    r"""Compact per-eigenstate summary rows (term symbol, moments, entanglement).

    One dict per state with ``energy_rel``, ``m_z`` (:math:`\langle L_z+2S_z\rangle`),
    ``j_z``, and — when available — ``term`` (from the manifold-resolved S/L/J values)
    and ``s_ent`` (impurity-bath entanglement entropy). Shared by the printed table and
    the statistics JSON.

    Note the usual degenerate-manifold caveat: ``m_z``/``j_z`` are single-particle
    values on one arbitrary member of a degenerate manifold; ``term`` and ``s_ent``
    are manifold-safe.

    Parameters
    ----------
    rhos : np.ndarray
        Per-state impurity density matrices (computational basis), shape (n, d, d).
    es : array_like
        Eigen-energies.
    rot_to_spherical : np.ndarray
        Spherical -> computational rotation for the impurity block.
    s_values, l_values, j_values : array_like or None
        Per-state quantum numbers (from the Casimirs); all three needed for ``term``.
    entanglement : dict or None
        The dict from :func:`gs_statistics.compute_entanglement_entropy`.
    impurity_orbitals : dict, optional
        ``Basis.impurity_orbitals``; makes ``m_z``/``j_z`` shell-aware (summed over
        shells) via :func:`compute_shell_observables` -- see
        :func:`print_expectation_values`.

    Returns
    -------
    list of dict
    """
    es = np.asarray(es, dtype=float)
    e0 = float(np.min(es)) if es.size else 0.0
    have_term = s_values is not None and l_values is not None and j_values is not None
    ent_values = entanglement["per_state_entropy"] if entanglement is not None else None
    rows = []
    for i, (e, rho) in enumerate(zip(es, rhos)):
        totals = compute_shell_observables(rho, rot_to_spherical, impurity_orbitals, shell_ls=shell_ls)["total"]
        m_z, j_z = totals["m_z"], totals["j_z"]
        row = {"state": i, "energy_rel": float(e - e0), "m_z": m_z, "j_z": j_z}
        if have_term:
            row["term"] = term_symbol(s_values[i], l_values[i], j_values[i])
        if ent_values is not None and i < len(ent_values):
            row["s_ent"] = float(ent_values[i])
        rows.append(row)
    return rows


def print_state_summary(summary, file=None):
    """Pretty-print the rows from :func:`compute_state_summary`."""
    have_term = any("term" in row for row in summary)
    have_ent = any("s_ent" in row for row in summary)
    header = f"{'i':>3s}  {'E-E0':>11s}  {'Lz+2Sz':>9s}  {'Jz':>9s}"
    if have_term:
        header += f"  {'term':>8s}"
    if have_ent:
        header += f"  {'S_ent':>8s}"
    print("Per-state summary (m_z/Jz are arbitrary within a degenerate manifold; term/S_ent are not):", file=file)
    print(header, file=file)
    for row in summary:
        line = f"{row['state']:>3d}  {row['energy_rel']:11.8f}  {row['m_z']: 8.5f}  {row['j_z']: 8.5f}"
        if have_term:
            line += f"  {row.get('term', ''):>8s}"
        if have_ent:
            line += f"  {row.get('s_ent', float('nan')):>8.4f}"
        print(line, file=file)


def compute_energy_decomposition(thermal_rho, h1, impurity_indices, e_thermal):
    r"""Decompose the thermal energy into one-body blocks and the Coulomb remainder.

    With the density-matrix convention :math:`\rho_{ij} = \langle c_j^\dagger c_i\rangle`
    the one-body energy is :math:`\langle H_{1b}\rangle = \mathrm{Tr}(h_1\rho)`, split into
    the impurity block, the bath block and the impurity-bath (hybridization) cross terms;
    the interaction energy follows as
    :math:`\langle H_U\rangle = \langle E\rangle - \langle H_{1b}\rangle`.

    Returns
    -------
    dict with keys ``e_imp_1b``, ``e_bath``, ``e_hyb``, ``e_one_body``, ``e_coulomb``,
    ``e_total``.
    """
    n_orb = thermal_rho.shape[0]
    imp = sorted(impurity_indices)
    bath = sorted(set(range(n_orb)) - set(imp))
    e_one_body = float(np.real(np.trace(h1 @ thermal_rho)))
    e_imp = float(np.real(np.trace(h1[np.ix_(imp, imp)] @ thermal_rho[np.ix_(imp, imp)])))
    e_bath = float(np.real(np.trace(h1[np.ix_(bath, bath)] @ thermal_rho[np.ix_(bath, bath)])))
    e_hyb = e_one_body - e_imp - e_bath
    return {
        "e_imp_1b": e_imp,
        "e_bath": e_bath,
        "e_hyb": e_hyb,
        "e_one_body": e_one_body,
        "e_coulomb": float(np.real(e_thermal) - e_one_body),
        "e_total": float(np.real(e_thermal)),
    }


def _print_shell_observable_table(shells, total, shell_casimir, j_sharp=True):
    r"""Print the *Per-shell impurity observables* table (one row per shell + a total row).

    ``shells`` is ``compute_shell_observables(...)["shells"]`` (one dict per shell, in
    ascending global-orbital order); ``total`` is that same call's ``["total"]`` -- passed
    in rather than re-summed here, so the table's total row is guaranteed to equal the
    scalar charge/magnetism rows printed above it, not an independently computed number
    that happens to agree. ``shell_casimir`` is ``{partition: {"l", "s2_thermal",
    "l2_thermal", "j2_thermal"}}`` from ``calc_gs``'s per-shell Casimir evaluation, joined
    here by ``shell["partition"]`` -- **not** ``shell["l"]``, since two distinct shells can
    share the same inferred ``l`` (e.g. two independent p-shells). A shell absent from
    ``shell_casimir`` (Casimirs unavailable/skipped) prints its N/Lz/Sz/L.S/T_z columns
    with the ``<S^2>``/``<L^2>``/``<J^2>``/``term`` columns left blank.

    The total row deliberately excludes the Casimir columns, since ``S^2``/``L^2``/``J^2``
    do not simply add across shells (a many-body Casimir of a combined system is not the
    sum of its parts' Casimirs).
    """
    header = (
        f"  {'shell':>5s}  {'N':>8s}  {'N(Dn)':>8s}  {'N(Up)':>8s}  {'Lz':>9s}  {'Sz':>9s}  "
        f"{'L.S':>9s}  {'T_z':>9s}  {'<S^2>':>7s}  {'<L^2>':>7s}  {'<J^2>':>7s}  {'term':>6s}"
    )
    print("Per-shell impurity observables:")
    print(header)
    for shell in shells:
        shell_label = f"l={shell['l']}" if shell["l"] is not None else "?"
        casimir = shell_casimir.get(shell["partition"], {}) if shell_casimir else {}
        line = (
            f"  {shell_label:>5s}  {shell['n']:8.5f}  {shell['n_dn']:8.5f}  {shell['n_up']:8.5f}  "
            f"{shell['lz']: 9.5f}  {shell['sz']: 9.5f}  {shell['l_dot_s']: 9.5f}  {shell['t_z']: 9.5f}"
        )
        if all(k in casimir for k in ("s2_thermal", "l2_thermal", "j2_thermal")):
            s2, l2, j2 = (np.real(casimir[k]) for k in ("s2_thermal", "l2_thermal", "j2_thermal"))
            term, _, _ = gated_term_symbol(s2, l2, j2, j_sharp)
            line += f"  {s2:7.4f}  {l2:7.4f}  {j2:7.4f}  {term:>6s}"
        print(line)
    total_line = (
        f"  {'total':>5s}  {total['n']:8.5f}  {total['n_dn']:8.5f}  {total['n_up']:8.5f}  "
        f"{total['lz']: 9.5f}  {total['sz']: 9.5f}  {total['l_dot_s']: 9.5f}  {total['t_z']: 9.5f}"
    )
    print(total_line)


def print_thermal_expectation_values(
    rho_thermal,
    e_thermal,
    rot_to_spherical,
    block_structure,
    s_thermal=None,
    l_thermal=None,
    j_thermal=None,
    sisb_thermal=None,
    sisb_z_thermal=None,
    sisb_z_connected=None,
    sisb_pairing_note=None,
    spin_pairing=None,
    extra_groups=None,
    impurity_orbitals=None,
    shell_casimir=None,
    j_sharp=True,
    shell_ls=None,
):
    """
    print several thermal expectation values, e.g. E, N, Sz, Lz.

    The table is printed in titled sub-blocks (*Charge & occupations*, *Magnetism &
    multiplets*, *Spin correlations*) with the ``=`` signs aligned across all blocks.
    ``extra_groups`` (optional) is a list of ``(title, rows)`` sub-blocks appended after
    the built-in ones (e.g. the energy decomposition), where each row is a
    ``(label, value, suffix)`` tuple; a ``None`` value prints the suffix as text.

    If ``s_thermal`` / ``l_thermal`` / ``j_thermal`` are given (the thermally-averaged
    impurity ``S(S+1)`` / ``L(L+1)`` / ``J(J+1)``), the corresponding ``<S^2>`` /
    ``<L^2>`` / ``<J^2>`` lines (with the quantum number) are appended. When all are
    ``None`` the output is identical to before.

    For a collinear spin-polarized bath the longitudinal correlation and its connected
    (covariance) form can be passed as ``sisb_z_thermal`` / ``sisb_z_connected`` — they
    print as ``<Sz_imp.Sz_bath>`` / ``cov(Sz_imp,Sz_bath)`` — and
    ``sisb_pairing_note`` (a string, from :meth:`SpinLayout.sisb_note`) marks the full
    ``<S_imp.S_bath>`` line as depending on a dn/up pairing that ``H`` does not verify;
    ``spin_pairing`` (:meth:`SpinLayout.pairs_source`) is printed as a ``spin pairing`` row when the
    pairs do not come from the plain index convention.

    ``impurity_orbitals`` (``Basis.impurity_orbitals``, optional) makes the charge/
    magnetism rows shell-aware via :func:`compute_shell_observables` -- see
    :func:`print_expectation_values`. ``None`` reproduces today's single-shell behaviour.
    When it resolves to more than one shell, a *Per-shell impurity observables* table is
    additionally printed (see :func:`_print_shell_observable_table`); ``shell_casimir``
    (from ``calc_gs``'s per-shell Casimir evaluation) supplies that table's ``<S^2>``/
    ``<L^2>``/``<J^2>``/``term`` columns when given.

    ``j_sharp`` (default ``True``, from :func:`is_j_sharp`) gates the scalar ``term``/
    ``g_J``/``mu_eff`` rows *and* the per-shell table's ``term`` column through the same
    :func:`gated_term_symbol` decision -- without spin-orbit coupling ``J`` is generically
    not a good quantum number, so reporting one value for it is misleading; when
    ``j_sharp=False`` the term omits the ``J`` subscript and ``g_J``/``mu_eff`` are
    suppressed (``mu_spin_only`` is unaffected). The default ``True`` reproduces today's
    (ungated) behaviour for callers that have not computed ``j_sharp``.
    """
    equivalent_blocks = get_equivalent_blocks(block_structure)
    block_labels, _ = block_group_labels(equivalent_blocks, block_structure, impurity_orbitals)
    shell_observables = compute_shell_observables(rho_thermal, rot_to_spherical, impurity_orbitals, shell_ls=shell_ls)
    totals = shell_observables["total"]
    N, Ndn, Nup = totals["n"], totals["n_dn"], totals["n_up"]

    # Collect (label, value, suffix) rows into titled sub-blocks, then print with the '='
    # signs aligned across all blocks and the numbers right-aligned (sign-padded), so the
    # whole section reads as one tidy table.
    charge_rows = [
        ("<E>", e_thermal, ""),
        ("<N>", N, ""),
        ("<N(Dn)>", Ndn, ""),
        ("<N(Up)>", Nup, ""),
    ]
    for blocks, label in zip(equivalent_blocks, block_labels):
        occ = np.sum(np.diag(rho_thermal)[[orb for block in blocks for orb in block_structure.blocks[block]]]).real
        charge_rows.append((f"<N({label})>", occ, ""))
    magnetism_rows = [
        ("<Lz>", totals["lz"], ""),
        ("<Sz>", totals["sz"], ""),
    ]
    magnetism_rows.append(("<Lz+2Sz>", totals["m_z"], "(saturation moment m_z, in mu_B)"))
    magnetism_rows.append(("<Jz>", totals["j_z"], ""))
    magnetism_rows.append(("<L.S>", totals["l_dot_s"], ""))
    magnetism_rows.append(("<T_z>", totals["t_z"], "(magnetic dipole, XMCD spin sum rule)"))
    for label, value in (("S", s_thermal), ("L", l_thermal), ("J", j_thermal)):
        if value is not None:
            magnetism_rows.append(
                (f"<{label}^2>", np.real(value), f"({label} = {casimir_to_quantum_number(value): 6.4f})")
            )
    if s_thermal is not None and l_thermal is not None and j_thermal is not None:
        _, _, mu_spin = lande_g_and_moments(s_thermal, l_thermal, j_thermal)
        term, g_j, mu_eff = gated_term_symbol(s_thermal, l_thermal, j_thermal, j_sharp)
        magnetism_rows.append(("term", None, term))
        if g_j is not None:
            magnetism_rows.append(("g_J", g_j, ""))
            magnetism_rows.append(("mu_eff", mu_eff, "(g_J sqrt(<J^2>), in mu_B)"))
        magnetism_rows.append(("mu_spin_only", mu_spin, "(2 sqrt(<S^2>), in mu_B)"))
    correlation_rows = []
    if spin_pairing:
        correlation_rows.append(("spin pairing", None, spin_pairing))
    if sisb_thermal is not None:
        suffix = sisb_pairing_note or ""
        correlation_rows.append(("<S_imp.S_bath>", np.real(sisb_thermal), suffix))
    if sisb_z_thermal is not None:
        correlation_rows.append(("<Sz_imp.Sz_bath>", np.real(sisb_z_thermal), ""))
    if sisb_z_connected is not None:
        correlation_rows.append(("cov(Sz_imp,Sz_bath)", np.real(sisb_z_connected), ""))

    groups = [
        ("Charge & occupations", charge_rows),
        ("Magnetism & multiplets", magnetism_rows),
        ("Spin correlations", correlation_rows),
    ]
    if extra_groups:
        groups.extend(extra_groups)
    groups = [(title, rows) for title, rows in groups if rows]

    label_width = max(len(label) for _, rows in groups for label, _, _ in rows)
    for gi, (title, rows) in enumerate(groups):
        if gi > 0:
            print()
        print(f"{title}:")
        for label, value, suffix in rows:
            # Text-only rows (e.g. the term symbol) carry their content in the suffix.
            line = (
                f"  {label:<{label_width}} = {suffix}"
                if value is None
                else f"  {label:<{label_width}} = {value: 12.7f}"
            )
            if suffix and value is not None:
                line += f"  {suffix}"
            print(line)

    if len(shell_observables["shells"]) > 1:
        print()
        _print_shell_observable_table(shell_observables["shells"], totals, shell_casimir, j_sharp)
