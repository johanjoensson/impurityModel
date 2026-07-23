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
import zlib
from collections import defaultdict
from math import comb

import numpy as np
from mpi4py import MPI

from impurityModel.ed import product_state_representation as psr
from impurityModel.ed.ManyBodyUtils import ManyBodyState
from impurityModel.ed.mpi_comm import graph_alltoall


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
    ``redistribute_psis``/``redistribute_block`` each determinant is owned by exactly
    one rank, so the local determinant weights are globally complete for the
    determinants this rank owns.

    ``psis`` is a ``ManyBodyState`` (one shared-support block, one column per
    state; a plain ``list[ManyBodyState]`` is accepted too and converted once). Pure
    support iteration with no inner products, so this is one pass over the block's
    rows instead of ``width`` separate per-state dict traversals -- a genuine
    algorithmic win, not just a container change (Phase 6a of the state-unification
    refactor; see doc/plans/manybodystate_block_unification.md).
    """
    if not isinstance(psis, ManyBodyState):
        psis = ManyBodyState.from_states(list(psis))
    n_orb = basis.num_spin_orbitals
    imp_idx = basis.impurity_spin_orbital_indices
    val_idx = basis.valence_spin_orbital_indices
    con_idx = basis.conduction_spin_orbital_indices

    state_configs = [defaultdict(float) for _ in range(psis.width)]
    thermal_config = defaultdict(float)
    local_det = {}  # det-bytes -> thermal weight
    local_det_info = {}  # det-bytes -> (config, occupied-per-channel)

    for state, row in psis.items():
        key = bytes(state.to_bytearray())
        bits = psr.bytes2bitarray(key, n_orb)
        config = (bits[imp_idx].count(), bits[val_idx].count(), bits[con_idx].count())
        local_det_info[key] = (
            config,
            _channel_occupied(bits, imp_idx),
            _channel_occupied(bits, val_idx),
            _channel_occupied(bits, con_idx),
        )
        det_weight = 0.0
        for n in range(psis.width):
            p = abs(row[n]) ** 2
            state_configs[n][config] += p
            thermal_config[config] += weights[n] * p
            det_weight += weights[n] * p
        local_det[key] = det_weight
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
    psis : ManyBodyState or list of ManyBodyState
        The rank-local low-energy eigenstates (as redistributed onto ``basis``), one
        column/element per state.
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
        merged_state_configs = [defaultdict(float) for _ in state_configs]
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
        channel_indices={
            "impurity": [int(i) for i in basis.impurity_spin_orbital_indices],
            "valence": [int(i) for i in basis.valence_spin_orbital_indices],
            "conduction": [int(i) for i in basis.conduction_spin_orbital_indices],
        },
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
    channel_indices=None,
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
    nat_eigvals, nat_eigvecs = np.linalg.eigh(imp_block)
    order = np.argsort(np.real(nat_eigvals))[::-1]
    nat_occ = np.real(nat_eigvals)[order]
    nat_vecs = nat_eigvecs[:, order]  # column k = natural orbital k, in impurity_indices order
    # One-body entanglement entropy of the free-fermion state with these occupations:
    # S_1b = -sum_k [n ln n + (1-n) ln(1-n)]. Zero iff every occupation is 0 or 1.
    n_clip = np.clip(nat_occ, 1e-12, 1 - 1e-12)
    one_body_entropy = float(-np.sum(n_clip * np.log(n_clip) + (1 - n_clip) * np.log(1 - n_clip)))

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
        "natural_orbital_vectors": nat_vecs,
        "natural_orbital_basis": [int(i) for i in impurity_indices],
        "one_body_entropy": one_body_entropy,
        "top_determinants": top_dets,
        "channel_indices": channel_indices,
    }


def _combination_rank(positions):
    """Colex rank of a k-combination given its ascending member positions.

    Bijection from k-subsets of ``{0..n-1}`` to ``[0, C(n,k))`` that needs no lookup
    table, so every MPI rank indexes impurity configurations identically.
    """
    return sum(comb(p, i + 1) for i, p in enumerate(positions))


def compute_impurity_rdm(basis, psis, max_bytes=256 * 1024**2):
    r"""Many-body impurity reduced density matrices :math:`\rho_\mathrm{imp} =
    \mathrm{Tr}_\mathrm{bath}|\psi_n\rangle\langle\psi_n|`, one per eigenstate.

    Each :math:`\rho_\mathrm{imp}` is block-diagonal in the impurity electron count
    :math:`N_\mathrm{imp}` (the eigenstates have definite total ``N`` and the bath count
    is fixed within a bath configuration), so only the observed ``N`` blocks are built,
    each of dimension :math:`\binom{n_\mathrm{imp}}{N}` with impurity configurations
    indexed by their colex combination rank (:func:`_combination_rank`).

    Distributed evaluation (collective): each rank splits its local determinants into an
    impurity configuration and a bath-bytes key, the groups are re-shuffled with
    :func:`mpi_comm.graph_alltoall` keyed on ``crc32(bath_key) % size`` (deterministic
    across processes, unlike Python's salted ``hash``) so every bath configuration is
    completed on exactly one rank, the per-group outer products are accumulated locally
    and the small blocks are ``Allreduce``-d. No state-vector gather.

    Parameters
    ----------
    basis : Basis
        Provides ``num_spin_orbitals``, ``impurity_spin_orbital_indices``, ``comm``,
        ``is_distributed``. The ``psis`` must be redistributed onto this basis (each
        determinant owned by exactly one rank).
    psis : ManyBodyState or list of ManyBodyState
        Rank-local eigenstates, one column/element per state.
    max_bytes : int, optional
        Memory guard: if the dense blocks for all states would exceed this, return
        ``None`` instead (e.g. a half-filled f shell with a wide occupation window).

    Returns
    -------
    state_blocks : list of dict or None
        ``state_blocks[n][N]`` is the :math:`N_\mathrm{imp}=N` block of state ``n``'s
        impurity RDM (identical on every rank), or ``None`` if the memory guard tripped.
    """
    if not isinstance(psis, ManyBodyState):
        psis = ManyBodyState.from_states(list(psis))
    width = psis.width
    n_orb = basis.num_spin_orbitals
    imp_idx = sorted(basis.impurity_spin_orbital_indices)
    n_imp = len(imp_idx)
    imp_set = set(imp_idx)
    bath_idx = [i for i in range(n_orb) if i not in imp_set]
    comm = basis.comm if basis.is_distributed else None

    # Local pass: bath configuration -> [(state, N_imp, imp-config rank, amplitude)].
    # Each determinant (row) is visited once regardless of width: its impurity
    # configuration / bath key depend only on its own bit pattern, not on which
    # column supplied it (Phase 6a of the state-unification refactor). A column's
    # missing determinants are exact zeros in the block's shared support (documented
    # ManyBodyState.from_states contract); skipping them here reproduces the old
    # per-state sparse traversal exactly (a zero-amplitude outer product contributes
    # nothing to the RDM either way) while avoiding shipping p-fold more entries
    # through the alltoall below than the sparse states actually held.
    local_groups = defaultdict(list)
    observed_n = set()
    for state, row in psis.items():
        bits = psr.bytes2bitarray(bytes(state.to_bytearray()), n_orb)
        imp_positions = [k for k, orb in enumerate(imp_idx) if bits[orb]]
        n_e = len(imp_positions)
        bath_key = bits[bath_idx].tobytes()
        m = _combination_rank(imp_positions)
        for n in range(width):
            amp = row[n]
            if amp != 0:
                local_groups[bath_key].append((n, n_e, m, complex(amp)))
                observed_n.add(n_e)

    if comm is not None and comm.size > 1:
        send = [defaultdict(list) for _ in range(comm.size)]
        for bath_key, entries in local_groups.items():
            send[zlib.crc32(bath_key) % comm.size][bath_key] = entries
        received = graph_alltoall([dict(s) for s in send], comm)
        groups = defaultdict(list)
        for d in received:
            for bath_key, entries in (d or {}).items():
                groups[bath_key].extend(entries)
        observed_n = set().union(*comm.allgather(observed_n)) if comm.size > 1 else observed_n
    else:
        groups = local_groups

    # Memory guard on the dense blocks (identical decision on every rank: observed_n is
    # the allgathered union).
    dims = {n_e: comb(n_imp, n_e) for n_e in observed_n}
    if sum(d * d for d in dims.values()) * 16 * max(width, 1) > max_bytes:
        return None

    state_blocks = [{n_e: np.zeros((dims[n_e], dims[n_e]), dtype=complex) for n_e in observed_n} for _ in range(width)]
    for entries in groups.values():
        per_state = defaultdict(list)
        for n, n_e, m, amp in entries:
            per_state[(n, n_e)].append((m, amp))
        for (n, n_e), lst in per_state.items():
            idx = [m for m, _ in lst]
            amps = np.array([a for _, a in lst], dtype=complex)
            state_blocks[n][n_e][np.ix_(idx, idx)] += np.outer(amps, amps.conj())

    if comm is not None and comm.size > 1:
        for n in range(width):
            for n_e in sorted(observed_n):
                comm.Allreduce(MPI.IN_PLACE, state_blocks[n][n_e], op=MPI.SUM)
    return state_blocks


def compute_entanglement_entropy(basis, psis, es, tau, max_bytes=256 * 1024**2, top_k=8):
    r"""Impurity-bath entanglement from the many-body impurity RDM (collective call).

    For each **pure** eigenstate the von Neumann entropy of
    :math:`\rho_\mathrm{imp} = \mathrm{Tr}_\mathrm{bath}|\psi_n\rangle\langle\psi_n|`
    is the genuine impurity-bath entanglement entropy
    :math:`S_n = -\sum_i \lambda_i \ln\lambda_i` (0 for a product state,
    :math:`\ln 2` for an impurity-bath Bell singlet). Additionally the entropy of the
    *thermal* impurity RDM :math:`\sum_n w_n \rho^{(n)}_\mathrm{imp}` is reported — for
    a mixed state this is the impurity entropy (entanglement + thermal mixing), not an
    entanglement measure; the mixture entropy :math:`-\sum_n w_n \ln w_n` is included
    for comparison.

    Returns
    -------
    dict or None
        ``None`` when the RDM memory guard tripped. Keys: ``per_state_entropy``,
        ``per_state_norm`` (:math:`\sum\lambda`, should be 1), ``spectrum_top`` (largest
        RDM eigenvalues per state), ``sector_weights`` (per state,
        :math:`\mathrm{Tr}\,\rho[N]` — matches the impurity marginal),
        ``thermal_entropy``, ``mixture_entropy``, ``n_imp_sectors``.
    """
    state_blocks = compute_impurity_rdm(basis, psis, max_bytes=max_bytes)
    if state_blocks is None:
        return None
    weights = _boltzmann_weights(es, tau)

    def _entropy(eigvals):
        lam = eigvals[eigvals > 1e-14]
        return float(max(-np.sum(lam * np.log(lam)), 0.0))

    per_state_entropy = []
    per_state_norm = []
    spectrum_top = []
    sector_weights = []
    sectors = sorted(state_blocks[0].keys()) if state_blocks else []
    for blocks in state_blocks:
        eigvals = np.concatenate([np.linalg.eigvalsh(blocks[n_e]) for n_e in sectors])
        eigvals = np.clip(np.real(eigvals), 0.0, None)
        per_state_entropy.append(_entropy(eigvals))
        per_state_norm.append(float(np.sum(eigvals)))
        spectrum_top.append([float(x) for x in np.sort(eigvals)[::-1][:top_k]])
        sector_weights.append({int(n_e): float(np.real(np.trace(blocks[n_e]))) for n_e in sectors})

    thermal_blocks = {n_e: sum(w * blocks[n_e] for w, blocks in zip(weights, state_blocks)) for n_e in sectors}
    thermal_eigvals = np.clip(
        np.real(np.concatenate([np.linalg.eigvalsh(thermal_blocks[n_e]) for n_e in sectors])), 0.0, None
    )
    w_pos = weights[weights > 1e-14]
    return {
        "per_state_entropy": per_state_entropy,
        "per_state_norm": per_state_norm,
        "spectrum_top": spectrum_top,
        "sector_weights": sector_weights,
        "thermal_entropy": _entropy(thermal_eigvals),
        "mixture_entropy": float(max(-np.sum(w_pos * np.log(w_pos)), 0.0)),
        "n_imp_sectors": [int(x) for x in sectors],
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


def print_gs_statistics(stats, file=None, verbose=1):
    """Pretty-print the statistics returned by :func:`compute_gs_statistics`.

    ``verbose >= 2`` prints the top Slater determinants with their full per-channel
    occupation lists; the default prints the compact hole/particle notation (valence
    holes and conduction particles relative to the filled valence sea).
    """
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

    ent = stats.get("entanglement")
    if ent is not None:
        print("\nImpurity-bath entanglement (many-body impurity RDM):", file=file)
        print(
            "  per-state S_ent = " + "  ".join(f"{s:.4f}" for s in ent["per_state_entropy"]),
            file=file,
        )
        print(
            f"  thermal impurity entropy = {ent['thermal_entropy']:.4f}  "
            f"(mixture entropy = {ent['mixture_entropy']:.4f}; the thermal value includes "
            "classical mixing, only the per-state values are pure entanglement)",
            file=file,
        )
        print(
            "  impurity RDM spectrum (state 0, largest): " + "  ".join(f"{x:.4f}" for x in ent["spectrum_top"][0]),
            file=file,
        )
    elif "entanglement" in stats:
        print("\nImpurity-bath entanglement: skipped (impurity RDM exceeds the memory guard)", file=file)

    nat = stats["natural_orbital_occupations"]
    print("\nImpurity natural-orbital occupations (thermal):", file=file)
    print("  " + "  ".join(f"{x:6.4f}" for x in nat), file=file)
    print(f"  one-body entanglement entropy S_1b = {stats['one_body_entropy']:.4f}", file=file)
    vecs = np.asarray(stats["natural_orbital_vectors"])
    orbital_labels = stats["natural_orbital_basis"]
    print("  composition (top components, |orb> = computational spin-orbital):", file=file)
    for k, occ in enumerate(nat):
        weights = np.abs(vecs[:, k]) ** 2
        top = np.argsort(weights)[::-1]
        parts = []
        covered = 0.0
        for idx in top[:3]:
            if weights[idx] < 0.01:
                break
            parts.append(f"{np.abs(vecs[idx, k]):.2f}|{orbital_labels[idx]}>")
            covered += weights[idx]
            if covered >= 0.95:
                break
        print(f"    n={occ:6.4f}: " + " + ".join(parts), file=file)

    print("\nTop Slater determinants (thermal weight):", file=file)
    channels = stats.get("channel_indices")
    compact = verbose < 2 and channels is not None
    occ_header = "imp occupied | val holes | con particles" if compact else "occupied (imp|val|con)"
    print(f"{'Imp':>5s} {'Val':>5s} {'Con':>5s} {'Weight':>12s} {'%':>8s}  {occ_header}", file=file)
    for d in stats["top_determinants"]:
        n_imp, n_val, n_con = d["config"]
        if compact:
            # Valence states are near-filled and conduction states near-empty, so holes /
            # particles relative to the filled valence sea are the short description.
            val_holes = sorted(set(channels["valence"]) - set(d["valence_occupied"]))
            occ = (
                f"imp {sorted(d['impurity_occupied'])} | val holes {val_holes} | con {sorted(d['conduction_occupied'])}"
            )
        else:
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


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars / arrays and complex values.

    Complex numbers are stored as ``[re, im]`` pairs (applied element-wise when a
    complex ndarray is expanded via ``tolist()``).
    """

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, (complex, np.complexfloating)):
            return [float(np.real(o)), float(np.imag(o))]
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_gs_statistics(stats, path):
    """Write ``stats`` to ``path`` as JSON (rank-0 caller's responsibility)."""
    with open(path, "w") as fh:
        json.dump(stats, fh, indent=2, cls=_NumpyEncoder)
