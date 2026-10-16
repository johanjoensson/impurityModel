r"""Single-particle density matrices of many-body states.

``rho_n[i, j] = <psi_n| c_j^dagger c_i |psi_n> = <chi_j|phi_i>`` with ``|phi_k> = c_k |psi_n>``.

Both factors live on the (N-1)-electron determinant ``d' = d - k``, so the only data movement
needed is to bring every ``(d', k, amplitude)`` triple to one rank. :func:`build_density_matrices`
does exactly that in one sparse all-to-all and finishes with a sparse product per rank; the
``phi_k`` are never materialised as states, so nothing is ever dense over the union of their
supports (the previous implementation fused them into one ``rows x n_orb`` block and ran out of
memory on large states).

Both public functions are collective on ``basis.comm`` when the basis is distributed.

Conventions (verified against ``ManyBodyOperator`` by the tests): a determinant key is a byte
string, orbital ``i`` is bit ``7 - i % 8`` of byte ``i // 8``; ``c_k`` carries the sign
``(-1)^(number of occupied orbitals with index < k)``.
"""

from math import ceil

import numpy as np
import scipy.sparse as sp
from mpi4py import MPI

from impurityModel.ed.ManyBodyUtils import ManyBodyState
from impurityModel.ed.mpi_comm import graph_alltoall

_DEFAULT_MAX_BYTES = 256 * 1024**2
_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def _as_block(psis):
    if isinstance(psis, ManyBodyState):
        return psis
    return ManyBodyState.from_states(list(psis))


def _bit_matrix(block, n_orb):
    """Key bytes ``(rows, n_bytes)`` and occupation bits ``(rows, n_orb)`` of a block's determinants."""
    keys = block.keys()
    if not keys:
        return np.zeros((0, 0), dtype=np.uint8), np.zeros((0, n_orb), dtype=np.uint8)
    key_bytes = np.frombuffer(b"".join(bytes(k.to_bytearray()) for k in keys), dtype=np.uint8).reshape(len(keys), -1)
    bits = np.unpackbits(key_bytes, axis=1)[:, :n_orb]
    return key_bytes, bits


def _amplitudes(block, rows, width):
    amps = np.asarray(block)
    if amps.ndim == 1:
        amps = amps[:, np.newaxis]
    if rows == 0:
        return np.zeros((0, width), dtype=complex)
    return amps.astype(complex, copy=False)


def _global_width(block, comm):
    """The block width, checked identical on every rank that owns rows (decided collectively)."""
    width = int(block.width)
    if comm is None or comm.size == 1:
        return width
    w_max = comm.allreduce(width, op=MPI.MAX)
    w_seen = comm.allreduce(width if len(block) else w_max, op=MPI.MIN)
    if w_seen != w_max:
        raise ValueError(f"ranks disagree on the number of states ({w_seen} vs {w_max})")
    return w_max


def _check_orbitals(orbitals, n_orb):
    orbitals = [int(o) for o in orbitals]
    if any(o < 0 or o >= n_orb for o in orbitals):
        raise ValueError(f"orbital index out of range [0, {n_orb}): {orbitals}")
    return orbitals


def occupations(basis, psis, orbitals=None):
    r"""``<psi_n| n_i |psi_n>`` for each state ``n`` and orbital ``i`` in ``orbitals`` (collective).

    Returns an array of shape ``(width, len(orbitals))``. No density matrix is built, and nothing but
    one ``Allreduce`` is communicated after the block is brought to its owner ranks.
    """
    n_orb = basis.num_spin_orbitals
    orbitals = _check_orbitals(range(n_orb) if orbitals is None else orbitals, n_orb)
    block = _as_block(psis)
    comm = basis.comm if basis.is_distributed else None
    width = _global_width(block, comm)
    if comm is not None:
        # A determinant's amplitude can be split across ranks (a freshly applied state); |a|^2 of a
        # split amplitude is not the sum of the |a_r|^2, so make every determinant whole first.
        block = basis.redistribute_block(block)
    _, bits = _bit_matrix(block, n_orb)
    amps = _amplitudes(block, bits.shape[0], width)
    weights = amps.real**2 + amps.imag**2  # (rows, width)
    occ = weights.T @ bits[:, orbitals].astype(float) if bits.shape[0] else np.zeros((width, len(orbitals)))
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, occ, op=MPI.SUM)
    return occ


def _splitmix(h):
    h = (h ^ (h >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    h = (h ^ (h >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return h ^ (h >> np.uint64(31))


def build_density_matrices(
    basis, psis, orbital_indices_left=None, orbital_indices_right=None, max_bytes=_DEFAULT_MAX_BYTES, rounds=None
):
    r"""Single-particle density matrices ``rho[n, i, j] = <psi_n| c_{right_j}^dagger c_{left_i} |psi_n>``.

    Parameters
    ----------
    basis : Basis
    psis : ManyBodyState or list of ManyBodyState
        Rank-local states; a determinant's amplitude may be split across ranks (the pieces add).
    orbital_indices_left, orbital_indices_right : sequence of int, optional
        Each defaults independently to every spin-orbital. Any order, repeats allowed.
    max_bytes : int
        Budget for the transient per-rank entry arrays of one round; the scatter is cut into
        hash-bucket rounds of the (N-1)-determinant so that a round fits.
    rounds : int, optional
        Force the number of rounds (tests); default is derived from ``max_bytes``.

    Returns
    -------
    ndarray of complex, shape ``(width, len(left), len(right))``, identical on every rank.
    """
    n_orb = basis.num_spin_orbitals
    left = _check_orbitals(range(n_orb) if orbital_indices_left is None else orbital_indices_left, n_orb)
    right = _check_orbitals(range(n_orb) if orbital_indices_right is None else orbital_indices_right, n_orb)
    block = _as_block(psis)
    comm = basis.comm if basis.is_distributed else None
    size = comm.size if comm is not None else 1
    width = _global_width(block, comm)

    used = sorted(set(left) | set(right))
    pos = {o: p for p, o in enumerate(used)}
    n_used = len(used)
    used_arr = np.asarray(used, dtype=np.int64)
    rho_used = np.zeros((width, n_used, n_used), dtype=complex)

    key_bytes, bits = _bit_matrix(block, n_orb)
    n_rows = bits.shape[0]
    amps = _amplitudes(block, n_rows, width)

    if n_rows:
        # (-1)^(occupied orbitals below k), per row and used orbital.
        below = np.cumsum(bits, axis=1, dtype=np.int16) - bits
        sign = 1.0 - 2.0 * (below[:, used_arr] & 1)
        occ_used = bits[:, used_arr].astype(bool)
        weights = np.random.default_rng(20261016).integers(0, 2**63, size=n_orb, dtype=np.uint64) * np.uint64(2) + 1
        row_hash = (bits.astype(np.uint64) * weights).sum(axis=1, dtype=np.uint64)
        ent_row, ent_col = np.nonzero(occ_used)
        del below
        # The hash of d' = d - k is the hash of d minus k's weight: no key is built to route it.
        ent_hash = _splitmix(row_hash[ent_row] - weights[used_arr[ent_col]])
    else:
        sign = np.zeros((0, n_used))
        ent_row = ent_col = np.zeros(0, dtype=np.int64)
        ent_hash = np.zeros(0, dtype=np.uint64)

    entry_bytes = (key_bytes.shape[1] if n_rows else 1) + 2 + 16 * width
    if rounds is None:
        n_entries = len(ent_row)
        if comm is not None:
            n_entries = comm.allreduce(n_entries, op=MPI.MAX)
        # x4: the send copy, the received copy, the concatenation and the sparse assembly.
        rounds = max(1, ceil(4 * n_entries * entry_bytes / max(int(max_bytes), 1)))
    rounds = int(rounds)
    bucket = ((ent_hash >> np.uint64(32)) % np.uint64(rounds)).astype(np.int64)
    dest = (ent_hash % np.uint64(size)).astype(np.int64)

    for r in range(rounds):
        sel = np.nonzero(bucket == r)[0]
        rows_r, cols_r = ent_row[sel], ent_col[sel]
        keys_r = key_bytes[rows_r].copy() if len(sel) else np.zeros((0, 1), dtype=np.uint8)
        if len(sel):
            orb = used_arr[cols_r]
            keys_r[np.arange(len(sel)), orb // 8] &= ~(np.uint8(1) << (7 - orb % 8).astype(np.uint8))
        payload = {
            "key": keys_r,
            "k": cols_r.astype(np.int16),
            "amp": amps[rows_r] * sign[rows_r, cols_r][:, None] if len(sel) else np.zeros((0, width), dtype=complex),
        }
        if comm is not None:
            d_r = dest[sel]
            send = [{name: arr[d_r == p] for name, arr in payload.items()} for p in range(size)]
            del payload
            received = [c for c in graph_alltoall(send, comm) if c and len(c["k"])]
            del send
            if received:
                payload = {name: np.concatenate([c[name] for c in received]) for name in ("key", "k", "amp")}
            else:
                payload = {
                    "key": np.zeros((0, 1), dtype=np.uint8),
                    "k": np.zeros(0, np.int16),
                    "amp": payload["amp"][:0],
                }
            del received
        n_ent = len(payload["k"])
        if n_ent == 0:
            continue
        _, group = np.unique(payload["key"], axis=0, return_inverse=True)
        group = np.asarray(group).reshape(-1)
        n_groups = int(group.max()) + 1
        for n in range(width):
            # COO -> CSR sums duplicate (group, orbital) pairs: split rows of one determinant add.
            a = sp.coo_matrix((payload["amp"][:, n], (group, payload["k"])), shape=(n_groups, n_used)).tocsr()
            rho_used[n] += (a.T @ a.conj()).toarray()
        del payload

    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, rho_used, op=MPI.SUM)

    li = np.asarray([pos[o] for o in left], dtype=np.int64)
    ri = np.asarray([pos[o] for o in right], dtype=np.int64)
    return rho_used[:, li][:, :, ri]
