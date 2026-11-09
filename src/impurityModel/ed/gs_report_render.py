"""Text renderers of the ground-state report model.

Pure functions from a :class:`~impurityModel.ed.gs_report_model.GroundStateReport` (plus a few replicated
extras) to a list of lines; no collectives, no printing. ``render_summary`` is the compact ``-v`` view.
"""

import numpy as np

from impurityModel.ed.utils import V_RESULT  # noqa: F401  (documents the tier the summary belongs to)

RULE = "=" * 80


def fmt(x, prec=3, signed=False):
    """Fixed-point ``x`` with ``prec`` decimals; magnitudes below ``10**-prec`` print as zero (never ``-0.000``)."""
    if x is None:
        return "-"
    x = float(np.real(x))
    if abs(x) < 10.0**-prec:
        x = 0.0
    return f"{x:+.{prec}f}" if signed else f"{x:.{prec}f}"


def _two_sz_text(two_sz):
    if two_sz is None:
        return "-"
    return ", ".join(f"{v:+d}" if v else "0" for v in sorted(two_sz))


def _group_occupations(naming, diag):
    """``[(label, n_orbitals, occupation)]`` over the impurity orbital groups, from ``diag`` (index -> n)."""
    groups = {}
    for o in naming.impurity_indices:
        label = naming.infos[o].group_label
        n, k = groups.get(label, (0.0, 0))
        groups[label] = (n + float(diag[o]), k + 1)
    return [(label, k, n) for label, (n, k) in groups.items()]


def render_summary(report, diag=None, configurations=None, one_body_entropy=None, unit=None, files=None):
    """The compact ``-v`` summary: about 20 lines, one screen.

    Parameters
    ----------
    report : GroundStateReport
    diag : mapping, optional
        Thermal one-body occupation per spin-orbital index (diagonal of the thermal density matrix).
    configurations : list, optional
        ``[((n_imp, n_val, n_con), weight)]`` in descending weight.
    one_body_entropy : float, optional
        ``S_1b`` of the natural occupations.
    unit : str, optional
        Name of the unit (impurity) the report belongs to.
    files : tuple of str, optional
        ``(text_path, json_path)`` to point at.
    """
    th = report.thermal
    lines = [RULE, "  Ground state" + (f": {unit}" if unit else ""), RULE]
    window = report.manifolds[-1].energy - report.manifolds[0].energy if report.manifolds else 0.0
    lines.append(
        f"E0 = {report.e0:.6f}   tau = {report.tau:.4g}   "
        f"{report.n_states} state{'s' if report.n_states != 1 else ''} in {len(report.manifolds)} "
        f"manifold{'s' if len(report.manifolds) != 1 else ''} (window {window:.4f})"
    )
    lines.append(f"basis: {report.n_determinants:,} determinants")
    layout = report.spin_layout
    if layout:
        grade = layout.get("grade") if isinstance(layout, dict) else None
        lines.append(f"spin layout: {grade if grade else 'not resolved'}")
    if report.ensemble:
        lines.append(
            "charge-sector ensemble: "
            + ", ".join(f"N_imp = {m['total']} (weight {m['weight']:.4f})" for m in report.ensemble["members"])
        )
    for rec in report.excitation_bounds:
        tag = "" if rec.get("n_imp") is None or len(report.excitation_bounds) == 1 else f"N_imp = {rec['n_imp']}: "
        lines.append(
            f"{tag}bath excitations: holes <= {rec['max_holes']}, electrons <= {rec['max_electrons']}, "
            f"total <= {rec['max_total']}" + (" [cap-bound]" if rec.get("cap_bound") else "")
        )
    mag = report.magnetic or {}
    lines.append("manifold   E-E0         g   weight    S       L      term    2Sz")
    for m in report.manifolds:
        s = None if m.s is None else np.mean(m.s)
        l_ = None if m.l is None else np.mean(m.l)
        term = mag.get("term") if m.index == 0 else None
        lines.append(
            f"  {m.index:<7d}{m.energy - report.e0:>10.7f}  {m.g:>3d}  {m.weight:>7.4f}  "
            f"{fmt(s) if s is not None else '-':>6}  {fmt(l_) if l_ is not None else '-':>6}  "
            f"{term or '-':<6}  {_two_sz_text(m.two_sz)}"
        )
    if diag is not None and report.naming is not None:
        occ = _group_occupations(report.naming, diag)
        n_imp = sum(n for _, _, n in occ)
        parts = "   ".join(f"{label} {n:.3f}/{k}" for label, k, n in occ)
        lines.append(f"<N_imp> = {n_imp:.4f}   by group (occupation/orbitals): {parts}")
    if configurations:
        lines.append(
            "configurations (imp,val,con): "
            + "  ".join(f"({a},{b},{c}) {100 * w:.1f}%" for (a, b, c), w in configurations[:3])
        )
    if mag:
        moments = [f"<Sz> = {fmt(mag.get('sz'))}", f"<Lz> = {fmt(mag.get('lz'))}"]
        if "s2" in mag:
            moments = [f"<S^2> = {fmt(mag['s2'])}", f"<L^2> = {fmt(mag['l2'])}"] + moments
            moments.append(f"mu_spin_only = {fmt(mag.get('mu_spin_only'), 2)} mu_B")
        lines.append("moments: " + "   ".join(moments))
    corr = report.correlation
    if corr:
        lines.append(
            f"correlation: open-shell double occupancy D = {fmt(corr.get('docc_open_shell_total'))}"
            + (f"   chi_spin_zz (Curie) = {fmt(th['chi_spin_zz'], 1)}" if th.get("chi_spin_zz") is not None else "")
        )
    if th.get("sisb") is not None:
        lines.append(f"screening: <S_imp.S_bath> = {fmt(th['sisb'], 4, signed=True)}")
    elif th.get("sisb_z") is not None:
        lines.append(f"screening: <Sz_imp.Sz_bath> = {fmt(th['sisb_z'], 4, signed=True)}")
    if one_body_entropy is not None:
        lines.append(f"entanglement: S_1b = {fmt(one_body_entropy)}")
    from impurityModel.ed.gs_report import convergence_summary  # local: the model layer does not import gs_report

    lines.append(convergence_summary(report.convergence))
    for reason in report.warnings:
        lines.append(f"WARNING: {reason}")
    for skip in report.skips:
        lines.append(f"not reported: {skip}")
    if files:
        lines.append(f"full report: {files[0]}   data: {files[1]}")
    return lines
