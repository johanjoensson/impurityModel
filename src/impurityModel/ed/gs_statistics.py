"""Ground-state configuration / determinant statistics.

This module owns the *occupation-weight* statistics of the (thermal) ground state:
how the weight of the many-body wavefunction is distributed over impurity / valence /
conduction configurations and over individual Slater determinants, plus a handful of
derived diagnostics (charge fluctuations, participation ratio / entropy, natural-orbital
occupations).

The density-matrix-based expectation values (``<N>``, ``<Lz>``, ``<L.S>``, ``<S^2>`` ...)
live in :mod:`impurityModel.ed.observables`; this module is complementary and consumes the
already-gathered eigenstates (``full_psis`` on rank 0 of :func:`calc_gs`).
"""

import json
from collections import defaultdict

import numpy as np

from impurityModel.ed import product_state_representation as psr


def _boltzmann_weights(es, tau):
    """Normalised Boltzmann weights ``w_n = exp(-(E_n - E0)/tau) / Z``."""
    es = np.asarray(es, dtype=float)
    if es.size == 0:
        return es
    weights = np.exp(-(es - np.min(es)) / tau)
    return weights / np.sum(weights)


def _channel_occupied(bits, indices):
    """Occupied spin-orbital indices (from ``indices``) in a determinant ``bits``."""
    return [orb for orb in indices if bits[orb]]


def _sorted_weight_rows(weight_map, total, weight_cutoff):
    """Turn ``{config: weight}`` into weight-sorted rows with fraction + cumulative.

    Rows with ``fraction < weight_cutoff`` are collapsed into a single ``remaining``
    row carrying the summed weight and the number of configurations it represents.
    """
    rows = []
    cumulative = 0.0
    remaining_weight = 0.0
    remaining_count = 0
    for config, weight in sorted(weight_map.items(), key=lambda kv: kv[1], reverse=True):
        fraction = weight / total if total > 0 else 0.0
        if fraction < weight_cutoff:
            remaining_weight += weight
            remaining_count += 1
            continue
        cumulative += fraction
        rows.append(
            {
                "config": tuple(int(n) for n in config),
                "weight": float(weight),
                "fraction": float(fraction),
                "cumulative": float(cumulative),
            }
        )
    remaining = None
    if remaining_count > 0:
        remaining = {
            "weight": float(remaining_weight),
            "fraction": float(remaining_weight / total) if total > 0 else 0.0,
            "count": int(remaining_count),
        }
    return rows, remaining


def _marginal(thermal_config, axis):
    """Marginal distribution + mean/variance of one occupation channel.

    ``axis`` selects the channel: 0 = impurity, 1 = valence, 2 = conduction.
    """
    dist = defaultdict(float)
    total = 0.0
    for config, weight in thermal_config.items():
        dist[int(config[axis])] += weight
        total += weight
    if total > 0:
        for n in dist:
            dist[n] /= total
    mean = sum(n * p for n, p in dist.items())
    variance = sum(n * n * p for n, p in dist.items()) - mean * mean
    return {
        "distribution": [(int(n), float(dist[n])) for n in sorted(dist)],
        "mean": float(mean),
        "variance": float(variance),
    }


def _participation(weight_map, total):
    """Effective number of entries ``1/Σ p²`` and Shannon entropy ``-Σ p ln p``."""
    if total <= 0:
        return {"effective_number": 0.0, "entropy": 0.0}
    probs = np.array([w / total for w in weight_map.values()], dtype=float)
    probs = probs[probs > 0]
    inv_participation = float(1.0 / np.sum(probs**2)) if probs.size else 0.0
    entropy = float(-np.sum(probs * np.log(probs))) if probs.size else 0.0
    return {"effective_number": inv_participation, "entropy": entropy}


def _local_partials(basis, psis, weights):
    """Single local pass over the rank-local states.

    Returns per-state config buckets, the thermally-weighted config buckets, and the
    rank-local determinant thermal-weights (+ occupation info). After
    ``redistribute_psis`` each determinant is owned by exactly one rank, so the local
    determinant weights are globally complete for the determinants this rank owns.
    """
    n_orb = basis.num_spin_orbitals
    imp_idx = basis.impurity_spin_orbital_indices
    val_idx = basis.valence_spin_orbital_indices
    con_idx = basis.conduction_spin_orbital_indices

    state_configs = [defaultdict(float) for _ in psis]
    thermal_config = defaultdict(float)
    local_det = {}  # det-bytes -> thermal weight
    local_det_info = {}  # det-bytes -> (config, occupied-per-channel)

    for n, psi in enumerate(psis):
        w_n = weights[n]
        for state, amp in psi.items():
            key = bytes(state.to_bytearray())
            p = abs(amp) ** 2
            bits = psr.bytes2bitarray(key, n_orb)
            config = (bits[imp_idx].count(), bits[val_idx].count(), bits[con_idx].count())
            state_configs[n][config] += p
            thermal_config[config] += w_n * p
            local_det[key] = local_det.get(key, 0.0) + w_n * p
            if key not in local_det_info:
                local_det_info[key] = (
                    config,
                    _channel_occupied(bits, imp_idx),
                    _channel_occupied(bits, val_idx),
                    _channel_occupied(bits, con_idx),
                )
    return state_configs, thermal_config, local_det, local_det_info


def compute_gs_statistics(
    basis,
    psis,
    es,
    tau,
    thermal_rho,
    impurity_indices,
    top_k=10,
    weight_cutoff=1e-3,
):
    """Compute occupation-weight statistics for the thermal ground state.

    Computed in a distributed fashion: each rank makes a single pass over its local
    states and the small per-rank partial results are reduced/gathered on rank 0 (the
    config space is tiny, and only each rank's top-``top_k`` determinant candidates plus
    scalar participation partials are exchanged). The full state vectors are **not**
    gathered. This is a collective call.

    Parameters
    ----------
    basis : Basis
        The ground-state basis (provides the flat orbital-index lists,
        ``num_spin_orbitals``, ``comm`` and ``is_distributed``).
    psis : list of ManyBodyState
        The rank-local low-energy eigenstates (as redistributed onto ``basis``).
    es : array_like
        The corresponding eigen-energies.
    tau : float
        Thermal energy scale used for the Boltzmann weights.
    thermal_rho : np.ndarray
        Thermally-averaged single-particle density matrix (computational basis),
        replicated on every rank.
    impurity_indices : list of int
        Flat list of impurity spin-orbital indices into ``thermal_rho``.
    top_k : int, default 10
        Number of rows to keep in the per-state and top-determinant tables.
    weight_cutoff : float, default 1e-3
        Fractional-weight threshold below which configurations are collapsed into a
        ``remaining`` row in the thermal configuration table.

    Returns
    -------
    stats : dict or None
        Structured statistics on rank 0 (JSON-serialisable via
        :func:`save_gs_statistics`); ``None`` on the other ranks.
    """
    weights = _boltzmann_weights(es, tau)
    state_configs, thermal_config, local_det, local_det_info = _local_partials(basis, psis, weights)

    # Determinant participation/entropy from scalar partials. Each determinant is owned
    # by a single rank, so summing these scalars across ranks is exact.
    lw = np.fromiter(local_det.values(), dtype=float, count=len(local_det))
    local_sum_w = float(lw.sum())
    local_sum_w2 = float(np.sum(lw**2))
    local_sum_wlogw = float(np.sum(lw * np.log(lw))) if lw.size else 0.0
    local_ranked = sorted(local_det.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    local_top = [(w, local_det_info[key]) for key, w in local_ranked]
    local_n_dets = len(local_det)

    if basis.is_distributed:
        comm = basis.comm
        gathered_state_configs = comm.gather([dict(c) for c in state_configs], root=0)
        gathered_thermal = comm.gather(dict(thermal_config), root=0)
        gathered_top = comm.gather(local_top, root=0)
        sum_w = comm.reduce(local_sum_w, root=0)
        sum_w2 = comm.reduce(local_sum_w2, root=0)
        sum_wlogw = comm.reduce(local_sum_wlogw, root=0)
        n_dets = comm.reduce(local_n_dets, root=0)
        if comm.rank != 0:
            return None
        merged_state_configs = [defaultdict(float) for _ in psis]
        for rank_configs in gathered_state_configs:
            for n, c in enumerate(rank_configs):
                for cfg, w in c.items():
                    merged_state_configs[n][cfg] += w
        merged_thermal = defaultdict(float)
        for d in gathered_thermal:
            for cfg, w in d.items():
                merged_thermal[cfg] += w
        top_candidates = [item for sub in gathered_top for item in sub]
    else:
        merged_state_configs = state_configs
        merged_thermal = thermal_config
        top_candidates = local_top
        sum_w, sum_w2, sum_wlogw, n_dets = local_sum_w, local_sum_w2, local_sum_wlogw, local_n_dets

    return _assemble_stats(
        merged_thermal,
        merged_state_configs,
        top_candidates,
        sum_w,
        sum_w2,
        sum_wlogw,
        n_dets,
        es,
        weights,
        tau,
        thermal_rho,
        impurity_indices,
        top_k,
        weight_cutoff,
    )


def _det_participation(sum_w, sum_w2, sum_wlogw):
    """Determinant effective-number / entropy from global scalar partials.

    With ``p = w / T`` and ``T = sum_w``:  ``1/Σ p² = T²/Σ w²`` and
    ``-Σ p ln p = -(Σ w ln w)/T + ln T``.
    """
    if sum_w <= 0:
        return {"effective_number": 0.0, "entropy": 0.0}
    return {
        "effective_number": float(sum_w**2 / sum_w2) if sum_w2 > 0 else 0.0,
        "entropy": float(-sum_wlogw / sum_w + np.log(sum_w)),
    }


def _assemble_stats(
    thermal_config,
    state_configs,
    top_candidates,
    sum_w,
    sum_w2,
    sum_wlogw,
    n_dets,
    es,
    weights,
    tau,
    thermal_rho,
    impurity_indices,
    top_k,
    weight_cutoff,
):
    """Build the final statistics dict from the merged (global) partial results."""
    config_total = sum(thermal_config.values())
    thermal_rows, thermal_remaining = _sorted_weight_rows(thermal_config, config_total, weight_cutoff)

    per_state = []
    es_arr = np.asarray(es, dtype=float)
    e0 = float(np.min(es_arr)) if es_arr.size else 0.0
    for n, configs in enumerate(state_configs):
        state_total = sum(configs.values())
        rows = []
        cumulative = 0.0
        for config, weight in sorted(configs.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
            fraction = weight / state_total if state_total > 0 else 0.0
            cumulative += fraction
            rows.append(
                {
                    "config": tuple(int(x) for x in config),
                    "weight": float(weight),
                    "fraction": float(fraction),
                    "cumulative": float(cumulative),
                }
            )
        per_state.append(
            {
                "energy_rel": float(es_arr[n] - e0),
                "boltzmann_weight": float(weights[n]),
                "rows": rows,
            }
        )

    marginals = {
        "impurity": _marginal(thermal_config, 0),
        "valence": _marginal(thermal_config, 1),
        "conduction": _marginal(thermal_config, 2),
    }

    participation = {
        "configurations": _participation(thermal_config, config_total),
        "determinants": _det_participation(sum_w, sum_w2, sum_wlogw),
    }

    imp_block = np.asarray(thermal_rho)[np.ix_(impurity_indices, impurity_indices)]
    nat_occ = np.sort(np.real(np.linalg.eigvalsh(imp_block)))[::-1]

    top_dets = []
    for weight, info in sorted(top_candidates, key=lambda kv: kv[0], reverse=True)[:top_k]:
        config, imp_occ, val_occ, con_occ = info
        top_dets.append(
            {
                "config": tuple(int(x) for x in config),
                "weight": float(weight),
                "fraction": float(weight / sum_w) if sum_w > 0 else 0.0,
                "impurity_occupied": list(imp_occ),
                "valence_occupied": list(val_occ),
                "conduction_occupied": list(con_occ),
            }
        )

    return {
        "tau": float(tau),
        "num_determinants": int(n_dets),
        "thermal_config_weights": {"rows": thermal_rows, "remaining": thermal_remaining},
        "per_state": per_state,
        "marginals": marginals,
        "participation": participation,
        "natural_orbital_occupations": [float(x) for x in nat_occ],
        "top_determinants": top_dets,
    }


def _print_config_table(rows, remaining, file, indent=""):
    """Print a weight-sorted (imp, val, con) configuration table."""
    print(
        f"{indent}{'Imp':>5s} {'Val':>5s} {'Con':>5s} {'Weight':>12s} {'%':>8s} {'Cumul%':>8s}",
        file=file,
    )
    for row in rows:
        n_imp, n_val, n_con = row["config"]
        print(
            f"{indent}{n_imp:>5d} {n_val:>5d} {n_con:>5d} "
            f"{row['weight']:>12.6f} {100 * row['fraction']:>8.3f} {100 * row['cumulative']:>8.3f}",
            file=file,
        )
    if remaining is not None:
        print(
            f"{indent}{'...':>5s} {'':>5s} {'':>5s} "
            f"{remaining['weight']:>12.6f} {100 * remaining['fraction']:>8.3f}"
            f"   ({remaining['count']} more configs below cutoff)",
            file=file,
        )


def print_gs_statistics(stats, file=None):
    """Pretty-print the statistics returned by :func:`compute_gs_statistics`."""
    sep = "=" * 80
    print(sep, file=file)
    print("Thermal ground-state configuration weights (Impurity, Valence, Conduction)", file=file)
    print(f"({stats['num_determinants']} Slater determinants, tau = {stats['tau']:.4g})", file=file)
    _print_config_table(stats["thermal_config_weights"]["rows"], stats["thermal_config_weights"]["remaining"], file)

    print("\nMarginal occupation distributions (thermal):", file=file)
    for name in ("impurity", "valence", "conduction"):
        m = stats["marginals"][name]
        dist = "  ".join(f"P({n})={p:.4f}" for n, p in m["distribution"])
        print(f"  {name:>11s}: <N>={m['mean']:7.4f}  Var={m['variance']:7.4f}  | {dist}", file=file)

    cfg = stats["participation"]["configurations"]
    det = stats["participation"]["determinants"]
    print("\nParticipation / entropy (thermal):", file=file)
    print(
        f"  configurations: N_eff={cfg['effective_number']:8.3f}  S={cfg['entropy']:8.4f}",
        file=file,
    )
    print(
        f"  determinants  : N_eff={det['effective_number']:8.3f}  S={det['entropy']:8.4f}",
        file=file,
    )

    nat = stats["natural_orbital_occupations"]
    print("\nImpurity natural-orbital occupations (thermal):", file=file)
    print("  " + "  ".join(f"{x:6.4f}" for x in nat), file=file)

    print("\nTop Slater determinants (thermal weight):", file=file)
    print(f"{'Imp':>5s} {'Val':>5s} {'Con':>5s} {'Weight':>12s} {'%':>8s}  occupied (imp|val|con)", file=file)
    for d in stats["top_determinants"]:
        n_imp, n_val, n_con = d["config"]
        occ = f"{d['impurity_occupied']}|{d['valence_occupied']}|{d['conduction_occupied']}"
        print(
            f"{n_imp:>5d} {n_val:>5d} {n_con:>5d} " f"{d['weight']:>12.6f} {100 * d['fraction']:>8.3f}  {occ}",
            file=file,
        )

    print("\nPer-eigenstate configuration weights (top entries):", file=file)
    for i, ps in enumerate(stats["per_state"]):
        print(
            f"  state {i:<3d} E-E0 = {ps['energy_rel']: .6f}   Boltzmann weight = {ps['boltzmann_weight']:.4f}",
            file=file,
        )
        _print_config_table(ps["rows"], None, file, indent="    ")
        print("", file=file)
    print(sep, file=file)


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars / arrays."""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_gs_statistics(stats, path):
    """Write ``stats`` to ``path`` as JSON (rank-0 caller's responsibility)."""
    with open(path, "w") as fh:
        json.dump(stats, fh, indent=2, cls=_NumpyEncoder)
