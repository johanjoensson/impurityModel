"""The two routing hashes (``SlaterDeterminant::routing_hash_nibble`` / ``routing_hash_orbital``).

The nibble hash gives every determinant with the same per-nibble occupation counts the same value, so an
LMO-shaped basis (10 impurity + 22 bath orbitals, up to 3 valence holes) collapses onto a few hundred hash
values and the busiest rank holds 2-3x the mean. The per-orbital hash gives every occupation pattern its own
weight and is balanced to sampling noise.
"""

import itertools
import os
import subprocess
import sys

import numpy as np
import pytest

from impurityModel.ed.ManyBodyUtils import SlaterDeterminant

N_IMP, N_BATH, MAX_HOLES, IMP_N = 10, 22, 3, 2
N_ORB = N_IMP + N_BATH


def _set_bits(occ):
    b = bytearray((N_ORB + 7) // 8)
    for o in occ:
        b[o // 8] |= 1 << (7 - o % 8)  # MSB-first, the repo's orbital -> bit convention
    return bytes(b)


@pytest.fixture(scope="module")
def lmo_shaped():
    """Occupation sets of an LMO-shaped basis: the impurity gains one electron per bath valence hole."""
    occs = []
    for h in range(MAX_HOLES + 1):
        for holes in itertools.combinations(range(N_BATH), h):
            bath = {N_IMP + i for i in range(N_BATH) if i not in holes}
            for imp in itertools.combinations(range(N_IMP), IMP_N + h):
                occs.append(frozenset(imp) | bath)
    return occs


@pytest.fixture(scope="module")
def hashes(lmo_shaped):
    sds = [SlaterDeterminant.from_bytes(_set_bits(o)) for o in lmo_shaped]
    nibble = np.array([s.routing_hash_nibble() for s in sds], dtype=np.uint64)
    orbital = np.array([s.routing_hash_orbital() for s in sds], dtype=np.uint64)
    return nibble, orbital


def _skew(h, ranks):
    counts = np.bincount((h % np.uint64(ranks)).astype(np.int64), minlength=ranks)
    return counts.max() / counts.mean()


def test_the_nibble_hash_collapses_the_basis_onto_a_few_hundred_values(hashes):
    nibble, orbital = hashes
    assert len(np.unique(nibble)) < len(nibble) // 100  # measured: 800 of 439,275
    assert len(np.unique(orbital)) == len(orbital)  # every determinant its own value


@pytest.mark.parametrize("ranks", [128, 240])
def test_the_nibble_hash_skews_the_partition_at_ground_state_rank_counts(hashes, ranks):
    # Pins the mechanism behind the logged local[min,max] = [1,156, 12,224] (max/mean 2.57) at 240 ranks.
    assert _skew(hashes[0], ranks) > 2.0


@pytest.mark.parametrize("ranks", [8, 19, 40, 64, 128, 240])
def test_the_orbital_hash_balances_the_partition_at_every_rank_count(hashes, ranks):
    assert _skew(hashes[1], ranks) < 1.1


@pytest.mark.parametrize("mode", ["routing_hash_nibble", "routing_hash_orbital"])
def test_a_hop_shifts_the_hash_by_a_determinant_independent_constant(lmo_shaped, mode):
    """MatvecExchangePlan's sparsity rests on this: moving an electron from j to i changes the hash by one
    constant per (i, j), whatever else is occupied -- so the set of peers a rank talks to is bounded by the
    number of distinct hops. (Orbital hash: W_i - W_j. Nibble hash: constant within a nibble pair.)"""
    # i and j in the same nibble (bits 0-3 of byte 1 = orbitals 8..11 here, MSB-first).
    i, j = 8, 9
    deltas = set()
    for occ in lmo_shaped[::97]:
        if j in occ and i not in occ:
            moved = (occ - {j}) | {i}
            before = getattr(SlaterDeterminant.from_bytes(_set_bits(occ)), mode)()
            after = getattr(SlaterDeterminant.from_bytes(_set_bits(moved)), mode)()
            deltas.add((int(after) - int(before)) % 2**64)
    assert len(deltas) == 1


def _hash_in_subprocess(mode_env):
    env = {k: v for k, v in os.environ.items() if k != "IMPURITYMODEL_ROUTING_HASH"}
    if mode_env is not None:
        env["IMPURITYMODEL_ROUTING_HASH"] = mode_env
    code = (
        "from impurityModel.ed.ManyBodyUtils import SlaterDeterminant as S\n"
        "d = S.from_bytes(b'\\xa5\\x0f\\x33\\x01')\n"
        "print(d.routing_hash() == d.routing_hash_nibble(), d.routing_hash() == d.routing_hash_orbital())\n"
    )
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("mode, expected", [(None, "True False"), ("nibble", "True False"), ("orbital", "False True")])
def test_routing_hash_dispatches_on_the_environment_knob(mode, expected):
    done = _hash_in_subprocess(mode)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == expected


def test_an_unknown_mode_aborts_instead_of_falling_back():
    # A rank that silently fell back would disagree with the others about who owns what.
    done = _hash_in_subprocess("bogus")
    assert done.returncode != 0 and "expected 'nibble' or 'orbital'" in done.stderr
