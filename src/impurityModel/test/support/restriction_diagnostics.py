"""Phase-1 restriction diagnostics: where does the many-body basis carry dead weight?

This module answers, on a real workload, two questions that decide whether the existing
occupation restrictions can be tightened or a new *graded* restriction can be added:

1. **Window slack** -- the enforced occupation windows
   (``Basis.restrictions`` / ``Basis.weighted_restrictions``) vs. the windows the
   converged basis *actually* uses (:func:`basis_restrictions.get_effective_restrictions`).
   Slack that is never occupied is a free tightening.
2. **Excitation-weight profile** -- the thermally-weighted determinant weight
   :math:`\sum_k p_k |\langle D | \psi_k\rangle|^2` resolved by (a) the number of bath
   excitations a determinant carries, (b) the coupling-distance shell of its deepest
   excitation (reusing :func:`basis_restrictions._impurity_coupling_distance`), and
   (c) the bath on-site energy :math:`|\varepsilon_o - \mu|` of its deepest excitation.
   A steep decay in (a)/(b) means a graded excitation budget (Phase 3a) can prune the
   basis; a mismatch between (b) and (c) motivates a perturbation-aware metric (Phase 3b).

The heavy lifting (ground state) runs through the production path
(:func:`selfenergy._prepare_solver_basis` + :func:`groundstate.calc_gs`), so the measured
basis is exactly what a real ``calc_selfenergy`` run would build. The per-determinant
reductions are MPI collectives (fixed bin edges on every rank, unconditional
``Allreduce``), matching the CLAUDE.md rules.

Run as an opt-in pytest module (one workload per process for honest ``VmHWM``)::

    RUN_RESTRICTION_DIAG=1 RESTRICTION_DIAG_WORKLOAD=fcc_ni_5 \\
        pytest -s -m benchmark src/impurityModel/test/restriction_diagnostics.py

or directly::

    python -m impurityModel.test.support.restriction_diagnostics fcc_ni_5
"""

import os

import numpy as np

# Loadable ``impurityModel_data.h5`` workloads (metal / insulator / off-diagonal / cheap
# exact-reference). AFM_NiO has no archive with an ``H solver`` group, so SMO stands in for
# the off-diagonal tier; NiO-verify (nso=20) is the cheap synthetic-style exact reference.
_IMPMOD_ROOT = "/home/johan/Programming/impmod_tests"
WORKLOADS = {
    "fcc_ni_5": f"{_IMPMOD_ROOT}/FCC_Ni/impmod/"
    "5_BathStates_HaverGeometry_noneReorthonormalization/impurityModel_data.h5",
    "fcc_ni_15": f"{_IMPMOD_ROOT}/FCC_Ni/impmod/"
    "15_BathStates_HaverGeometry_partialReorthonormalization/impurityModel_data.h5",
    "nio_20": f"{_IMPMOD_ROOT}/NiO/impmod/verify_fixes/impurityModel_data.h5",
    "nio_15chain": f"{_IMPMOD_ROOT}/NiO/impmod/"
    "15_BathStates_linked_chainGeometry_noneReorthonormalization_6_processors_/impurityModel_data.h5",
    "nio_25chain": f"{_IMPMOD_ROOT}/NiO/impmod/"
    "25_BathStates_linked_chainGeometry_noneReorthonormalization_6_processors_/impurityModel_data.h5",
    "smo": f"{_IMPMOD_ROOT}/SMO/cubic/impmod/impurityModel_data.h5",
}


def _boltzmann_weights(es, tau):
    """Normalised Boltzmann weights ``p_k`` for state energies ``es`` at scale ``tau``."""
    es = np.asarray(es, dtype=float)
    w = np.exp(-(es - es.min()) / tau)
    return w / w.sum()


def build_ground_state(
    workload_key, comm=None, verbosity=0, truncation_threshold=np.inf, excitation_budget=None, slater_weight_min=None
):
    """Run the production GS path for a workload; return the diagnostics inputs.

    Returns a dict with the solver-basis operator ``h``, the ``ground_state_basis``, the
    eigenstates ``psis`` and energies ``es``, the impurity/valence/conduction orbital
    layout, and ``tau`` -- everything :func:`measure` needs. ``truncation_threshold`` caps
    the determinant count (``np.inf`` = natural basis; a finite value bounds memory on a
    small machine). ``excitation_budget`` (int), when set, injects a Phase-3a
    :func:`basis_restrictions.excitation_budget_restriction` bounding the total bath
    excitation -- the gating experiment for whether a budget shrinks the GS basis at fixed E0.
    """
    from impurityModel.ed.basis_restrictions import excitation_budget_restriction
    from impurityModel.ed.groundstate import calc_gs
    from impurityModel.ed.selfenergy import _prepare_solver_basis
    from impurityModel.test.support.real_workload import load_workload

    wl = load_workload(WORKLOADS[workload_key])
    sb = _prepare_solver_basis(
        wl["h0"],
        wl["u4"],
        wl["impurity_orbitals"],
        wl["nominal_occ"],
        wl["mixed_valence"],
        wl["rot_to_spherical"],
        verbosity,
    )
    weighted_restrictions = None
    if excitation_budget is not None:
        rest = excitation_budget_restriction(sb.bath_states, budget=excitation_budget)
        weighted_restrictions = [rest] if rest is not None else None
    basis_information = {
        "impurity_orbitals": sb.impurity_orbitals,
        "bath_states": sb.bath_states,
        "N0": sb.nominal_occ,
        "mixed_valence": sb.mixed_valence,
        "tau": wl["tau"],
        "chain_restrict": wl["chain_restrict"],
        "dense_cutoff": wl["dense_cutoff"],
        "spin_flip_dj": wl["spin_flip_dj"],
        "rank": comm.rank if comm is not None else 0,
        "comm": comm,
        "truncation_threshold": truncation_threshold,
        "weighted_restrictions": weighted_restrictions,
    }
    swmin = wl["slaterWeightMin"] if slater_weight_min is None else slater_weight_min
    psis, es, gs_basis, _thermal_rho, _gs_info = calc_gs(
        sb.h,
        basis_information,
        sb.block_structure,
        sb.rot_to_spherical,
        verbosity >= 2,
        slaterWeightMin=swmin,
    )
    return {
        "label": wl["label"],
        "workload": workload_key,
        "h": sb.h,
        "gs_basis": gs_basis,
        "psis": psis,
        "es": es,
        "impurity_orbitals": sb.impurity_orbitals,
        "bath_states": sb.bath_states,
        "n_spin_orbitals": sb.n_spin_orbitals,
        "tau": wl["tau"],
        "slaterWeightMin": wl["slaterWeightMin"],
    }


def _observed_subset_range(basis, subset):
    """Observed (min, max) of ``sum_{o in subset} n_o`` across the distributed basis."""
    from mpi4py import MPI

    from impurityModel.ed import product_state_representation as psr

    idx = sorted(subset)
    lo, hi = len(idx), 0
    for state in basis.local_basis:
        bits = psr.bytes2bitarray(bytes(state.to_bytearray()), basis.num_spin_orbitals)
        n = sum(bits[o] for o in idx)
        lo, hi = min(lo, n), max(hi, n)
    if basis.is_distributed:
        lo = basis.comm.allreduce(lo, op=MPI.MIN)
        hi = basis.comm.allreduce(hi, op=MPI.MAX)
    return lo, hi


def _observed_weighted_range(basis, weights):
    """Observed (min, max) of ``sum_o weights[o] * n_o`` across the distributed basis."""
    from mpi4py import MPI

    from impurityModel.ed import product_state_representation as psr

    lo, hi = None, None
    for state in basis.local_basis:
        bits = psr.bytes2bitarray(bytes(state.to_bytearray()), basis.num_spin_orbitals)
        q = sum(int(w) * bits[o] for o, w in weights.items())
        lo = q if lo is None else min(lo, q)
        hi = q if hi is None else max(hi, q)
    if lo is None:
        lo, hi = 0, 0
    if basis.is_distributed:
        lo = basis.comm.allreduce(lo, op=MPI.MIN)
        hi = basis.comm.allreduce(hi, op=MPI.MAX)
    return lo, hi


def _window_slack(gs):
    """Enforced occupation windows vs. the range the converged basis actually occupies.

    The enforced restrictions (``Basis.restrictions`` / ``weighted_restrictions``) and the
    convention-based :func:`get_effective_restrictions` partition the orbitals differently,
    so a key-by-key comparison is meaningless. Instead, for each *enforced* subset we
    measure the occupation range the basis actually spans on that exact orbital set and
    report the never-used margin (``slack``). Positive slack on either side is a candidate
    tightening. ``get_effective_restrictions`` is reported separately as the observed
    per-group windows.
    """
    from impurityModel.ed.basis_restrictions import get_effective_restrictions

    basis = gs["gs_basis"]
    enforced = basis.restrictions or {}
    rows = []
    for key in sorted(enforced, key=lambda s: (len(s), sorted(s))):
        e_lo, e_hi = enforced[key]
        o_lo, o_hi = _observed_subset_range(basis, key)
        rows.append(
            {
                "orbitals": sorted(key),
                "size": len(key),
                "enforced": (e_lo, e_hi),
                "observed": (o_lo, o_hi),
                "slack_low": o_lo - e_lo,
                "slack_high": e_hi - o_hi,
            }
        )
    weighted_rows = []
    for weights, (q_lo, q_hi) in basis.weighted_restrictions or []:
        o_lo, o_hi = _observed_weighted_range(basis, weights)
        weighted_rows.append(
            {
                "n_orbitals": len(weights),
                "enforced": (q_lo, q_hi),
                "observed": (o_lo, o_hi),
                "slack_low": o_lo - q_lo,
                "slack_high": q_hi - o_hi,
            }
        )
    return {
        "windows": rows,
        "weighted_windows": weighted_rows,
        "effective": get_effective_restrictions(basis),
    }


def _excitation_profiles(gs, coupling_cutoff=1e-3, min_dist=4, n_depth_bins=8, n_energy_bins=8):
    """Thermally-weighted determinant weight by excitation order / coupling depth / bath energy.

    Reference occupation: valence baths nominally filled, conduction baths nominally empty
    (the ``get_effective_restrictions`` convention). A determinant's bath *excitations* are
    holes in filled-valence orbitals and electrons in empty-conduction orbitals. Each such
    orbital contributes at its coupling distance from the impurity and its ``|h[o,o]|`` bath
    energy; a determinant is placed in the bin of its *deepest* (largest-distance /
    largest-energy) excitation.
    """
    from mpi4py import MPI

    from impurityModel.ed import product_state_representation as psr
    from impurityModel.ed.basis_restrictions import _impurity_coupling_distance
    from impurityModel.ed.ManyBodyUtils import ManyBodyOperator

    basis = gs["gs_basis"]
    comm = basis.comm if basis.is_distributed else None
    valence_baths, conduction_baths = gs["bath_states"]

    all_impurity = [o for blocks in gs["impurity_orbitals"].values() for blk in blocks for o in blk]
    valence_orbs = [o for blocks in valence_baths.values() for blk in blocks for o in blk]
    conduction_orbs = [o for blocks in conduction_baths.values() for blk in blocks for o in blk]
    tot_orb = gs["n_spin_orbitals"]

    op = gs["h"] if isinstance(gs["h"], ManyBodyOperator) else ManyBodyOperator(gs["h"])
    dist_matrix, dist_cutoff = _impurity_coupling_distance(op, tot_orb, all_impurity, coupling_cutoff, min_dist)
    # dist_matrix rows are ordered as all_impurity; column o = distance to orbital o.
    # depth(o) = min over impurity orbitals; +inf orbitals (disconnected) clip to the max finite.
    depth = {o: float(np.min(dist_matrix[:, o])) for o in valence_orbs + conduction_orbs}
    finite = [d for d in depth.values() if np.isfinite(d)]
    d_max = max(finite) if finite else 1.0
    depth = {o: (d if np.isfinite(d) else d_max) for o, d in depth.items()}

    # Bath on-site energy |eps_o - mu|; mu = 0 in the solver basis (Fermi level, see
    # classify_bath_occupation). Read the one-body diagonal off the operator (ManyBodyOperator
    # supports ``key in op`` / ``op[key]`` but not ``.get``).
    def _diag(o):
        key = ((o, "c"), (o, "a"))
        return abs(op[key]) if key in op else 0.0

    h_diag = {o: _diag(o) for o in valence_orbs + conduction_orbs}
    e_max = max(h_diag.values()) if h_diag else 1.0

    depth_edges = np.linspace(0.0, dist_cutoff * 2 if dist_cutoff > 0 else d_max, n_depth_bins + 1)
    depth_edges[-1] = max(depth_edges[-1], d_max) + 1e-9
    energy_edges = np.linspace(0.0, e_max + 1e-12, n_energy_bins + 1)

    p_k = _boltzmann_weights(gs["es"], gs["tau"])
    max_order = 0
    order_w = {}
    depth_w = np.zeros(n_depth_bins)
    energy_w = np.zeros(n_energy_bins)
    total_w = 0.0

    for k, psi in enumerate(gs["psis"]):
        for det, amp in psi.items():
            wgt = p_k[k] * (abs(amp) ** 2)
            if wgt == 0.0:
                continue
            total_w += wgt
            bits = psr.bytes2bitarray(bytes(det.to_bytearray()), tot_orb)
            excited = [o for o in valence_orbs if bits[o] == 0]  # holes in filled valence
            excited += [o for o in conduction_orbs if bits[o] == 1]  # electrons in empty conduction
            order = len(excited)
            order_w[order] = order_w.get(order, 0.0) + wgt
            max_order = max(max_order, order)
            if excited:
                dd = max(depth[o] for o in excited)
                ee = max(h_diag[o] for o in excited)
                depth_w[min(np.searchsorted(depth_edges, dd, side="right") - 1, n_depth_bins - 1)] += wgt
                ee_bin = min(np.searchsorted(energy_edges, ee, side="right") - 1, n_energy_bins - 1)
                energy_w[max(ee_bin, 0)] += wgt
            else:
                depth_w[0] += wgt  # reference config carries no excitation

    if comm is not None:
        total_w = comm.allreduce(total_w, op=MPI.SUM)
        depth_w = comm.allreduce(depth_w, op=MPI.SUM)
        energy_w = comm.allreduce(energy_w, op=MPI.SUM)
        max_order = comm.allreduce(max_order, op=MPI.MAX)
        order_vec = np.zeros(max_order + 1)
        for o, w in order_w.items():
            order_vec[o] += w
        order_vec = comm.allreduce(order_vec, op=MPI.SUM)
        order_w = {o: order_vec[o] for o in range(max_order + 1)}

    return {
        "total_weight": total_w,
        "order_weight": order_w,
        "depth_edges": depth_edges,
        "depth_weight": depth_w,
        "dist_cutoff": dist_cutoff,
        "energy_edges": energy_edges,
        "energy_weight": energy_w,
    }


def measure(gs, **kwargs):
    """Full Phase-1 diagnostic bundle for a ground state from :func:`build_ground_state`."""
    return {
        "label": gs["label"],
        "workload": gs["workload"],
        "gs_size": gs["gs_basis"].size,
        "n_states": len(gs["psis"]),
        "slack": _window_slack(gs),
        "profiles": _excitation_profiles(gs, **kwargs),
    }


def render(result):
    """Human-readable report of a :func:`measure` bundle."""
    lines = []
    lines.append(f"=== restriction diagnostics: {result['workload']} ({result['label']}) ===")
    lines.append(f"ground-state basis: {result['gs_size']} determinants, {result['n_states']} thermal states")
    lines.append("")
    lines.append("-- enforced occupation-window slack (slack = enforced margin the basis never uses) --")
    if not result["slack"]["windows"]:
        lines.append("  (no subset restrictions enforced on this basis)")
    else:
        lines.append(f"{'orbitals':<28} {'enforced':>12} {'observed':>12} {'slack lo/hi':>12}")
        for row in result["slack"]["windows"]:
            orbs = str(row["orbitals"])
            orbs = orbs if len(orbs) <= 26 else orbs[:23] + "..."
            lines.append(
                f"{orbs:<28} {row['enforced']!s:>12} {row['observed']!s:>12} "
                f"{str(row['slack_low']) + '/' + str(row['slack_high']):>12}"
            )
    wr = result["slack"]["weighted_windows"]
    if wr:
        lines.append("weighted restrictions (enforced vs observed weighted-sum range):")
        for row in wr:
            lines.append(
                f"  {row['n_orbitals']} orbs: enforced {row['enforced']} observed {row['observed']} "
                f"slack {row['slack_low']}/{row['slack_high']}"
            )
    else:
        lines.append("weighted restrictions enforced: none")
    lines.append("")

    prof = result["profiles"]
    tot = prof["total_weight"] or 1.0
    lines.append(f"-- thermal weight by bath-excitation order (total={prof['total_weight']:.4f}) --")
    cum = 0.0
    for order in sorted(prof["order_weight"]):
        w = prof["order_weight"][order]
        cum += w
        lines.append(f"  order {order:>2}: {w / tot:8.4%}   (cumulative {cum / tot:8.4%})")
    lines.append("")
    lines.append(
        f"-- thermal weight by deepest-excitation coupling distance (freeze cutoff={prof['dist_cutoff']:.3f}) --"
    )
    edges = prof["depth_edges"]
    for i, w in enumerate(prof["depth_weight"]):
        lines.append(f"  [{edges[i]:6.3f}, {edges[i + 1]:6.3f}): {w / tot:8.4%}")
    lines.append("")
    lines.append("-- thermal weight by deepest-excitation bath energy |eps-mu| --")
    edges = prof["energy_edges"]
    for i, w in enumerate(prof["energy_weight"]):
        lines.append(f"  [{edges[i]:6.3f}, {edges[i + 1]:6.3f}): {w / tot:8.4%}")
    return "\n".join(lines)


def _run(workload_key, comm=None, verbosity=0):
    gs = build_ground_state(workload_key, comm=comm, verbosity=verbosity)
    result = measure(gs)
    rank = comm.rank if comm is not None else 0
    if rank == 0:
        print(render(result))
    return result


def budget_experiment(workload_key, budgets, comm=None, truncation_threshold=np.inf, verbosity=0):
    """Phase-3a gate: does an excitation budget shrink the GS basis at fixed E0?

    Builds the ground state with no budget (reference), then with each budget in ``budgets``,
    and reports E0 shift and basis-size ratio. A budget "passes" where it shrinks the basis
    meaningfully while ``|dE0|`` stays at round-off (the correlation energy is carried by the
    low-excitation determinants the budget keeps).
    """
    rank = comm.rank if comm is not None else 0
    ref = build_ground_state(workload_key, comm=comm, verbosity=0, truncation_threshold=truncation_threshold)
    ref_size = ref["gs_basis"].size
    ref_e0 = float(np.min(ref["es"]))
    del ref
    rows = []
    for b in budgets:
        gs = build_ground_state(
            workload_key, comm=comm, verbosity=0, truncation_threshold=truncation_threshold, excitation_budget=b
        )
        size = gs["gs_basis"].size
        e0 = float(np.min(gs["es"]))
        del gs
        if rank == 0:
            rows.append(
                {
                    "budget": b,
                    "gs_size": size,
                    "size_ratio": ref_size / size if size else float("nan"),
                    "dE0": e0 - ref_e0,
                }
            )
    if rank == 0:
        print(f"\n=== excitation-budget GS experiment: {workload_key} (reference: no budget) ===")
        print(f"reference: {ref_size} determinants, E0 = {ref_e0:.6f}")
        print(f"{'budget':>8}{'gs_size':>10}{'ref/size':>10}{'dE0':>14}")
        for r in rows:
            print(f"{r['budget']:>8}{r['gs_size']:>10d}{r['size_ratio']:>10.2f}{r['dE0']:>14.2e}")
    return rows


def _manifold_fidelity(ref_psis, res_psis):
    """Singular values of the cross-overlap between two low-energy eigenvector manifolds.

    Amplitudes are keyed by determinant bytes so the two independently-built bases can be
    compared on their shared determinants (the restricted basis is a subset, so its states
    carry zero amplitude on the pruned determinants). With orthonormal input states the
    singular values are the cosines of the principal angles between the subspaces; the
    smallest is the worst-case fidelity, and ``1 - min`` is the largest manifold rotation.
    """

    def _keyed(psis):
        return [{bytes(d.to_bytearray()): a for d, a in p.items()} for p in psis]

    ref, res = _keyed(ref_psis), _keyed(res_psis)
    overlap = np.zeros((len(ref), len(res)), dtype=complex)
    for i, r in enumerate(ref):
        for j, s in enumerate(res):
            shared = r.keys() & s.keys()
            overlap[i, j] = sum(np.conj(r[k]) * s[k] for k in shared)
    return np.linalg.svd(overlap, compute_uv=False)


def eigenvector_overlap_experiment(workload_key, budgets, comm=None, truncation_threshold=np.inf, verbosity=0):
    """Accuracy gate: does an excitation budget rotate the ground-state manifold?

    The self-energy / Green's function is a deterministic function of the eigenstates
    ``psis`` and energies ``es`` (the campaign's rigorous cheap proxy -- no GF solve, which
    avoids the FCC Ni over-convergence pathology). Builds the reference GS (no budget) and
    each budget-restricted GS, and reports the worst-case fidelity of the restricted
    low-energy manifold within the reference manifold (smallest cross-overlap singular value)
    alongside ``dE0``. A budget is lossless where ``1 - min_fidelity`` stays below the
    physical broadening (~1e-3); ``dE0`` alone is a MISLEADING proxy (see the campaign verdict).
    """
    rank = comm.rank if comm is not None else 0
    ref = build_ground_state(workload_key, comm=comm, verbosity=0, truncation_threshold=truncation_threshold)
    ref_e0 = float(np.min(ref["es"]))
    ref_psis = ref["psis"]
    rows = []
    for b in budgets:
        gs = build_ground_state(
            workload_key, comm=comm, verbosity=0, truncation_threshold=truncation_threshold, excitation_budget=b
        )
        sv = _manifold_fidelity(ref_psis, gs["psis"])
        e0 = float(np.min(gs["es"]))
        size = gs["gs_basis"].size
        del gs
        if rank == 0:
            rows.append({"budget": b, "gs_size": size, "min_fidelity": float(np.min(sv)), "dE0": e0 - ref_e0})
    if rank == 0:
        print(f"\n=== excitation-budget eigenvector-overlap gate: {workload_key} (reference: no budget) ===")
        print(f"reference: {ref['gs_basis'].size} determinants, E0 = {ref_e0:.6f}, {len(ref_psis)} states")
        print(f"{'budget':>8}{'gs_size':>10}{'1-fidelity':>13}{'dE0':>13}")
        for r in rows:
            print(f"{r['budget']:>8}{r['gs_size']:>10d}{1 - r['min_fidelity']:>13.2e}{r['dE0']:>13.2e}")
    return rows


# ---- opt-in pytest entry -------------------------------------------------------------

RUN = os.environ.get("RUN_RESTRICTION_DIAG") == "1"


def test_restriction_diagnostics():
    import pytest

    if not RUN:
        pytest.skip("Set RUN_RESTRICTION_DIAG=1 to run the restriction diagnostics.")
    from mpi4py import MPI

    key = os.environ.get("RESTRICTION_DIAG_WORKLOAD", "nio_20")
    _run(key, comm=MPI.COMM_WORLD, verbosity=int(os.environ.get("RESTRICTION_DIAG_VERBOSITY", "0")))


test_restriction_diagnostics.benchmark = True  # type: ignore[attr-defined]  # pytest-benchmark marker


if __name__ == "__main__":
    import sys

    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD if MPI.COMM_WORLD.size > 1 else None
    except ImportError:
        comm = None
    key = sys.argv[1] if len(sys.argv) > 1 else "nio_20"
    if len(sys.argv) > 2 and sys.argv[2] == "budget":
        budgets = [int(b) for b in sys.argv[3:]] or [8, 6, 5, 4, 3, 2, 1]
        budget_experiment(key, budgets, comm=comm)
    else:
        _run(key, comm=comm, verbosity=1)
