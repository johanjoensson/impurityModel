"""Sector-resolved exact-diagonalization oracle for the Green's-function branch matrix.

Every consumer of the GF engine (``get_Greens_function``, ``calc_spectra``,
``calc_Greens_function_with_offdiag``, ``calc_selfenergy``) is compared against the
Lehmann sum built here, which shares no code with the Krylov/continued-fraction pipeline:
each particle-number sector is diagonalized densely *once* (see
[[dense-lehmann-oracle-verification-technique]]: never mix two ``eigh`` calls on one
matrix), the transition operators are tabulated between sector eigenbases, and

.. math::

    G_{ab}(z) = \\frac{1}{Z}\\sum_e w_e \\Big[
        \\sum_m \\frac{\\langle e|A_a^\\dagger|m\\rangle\\langle m|A_b|e\\rangle}{z-(E_m-E_e)}
      + \\sum_m \\frac{\\langle e|A_b|m\\rangle\\langle m|A_a^\\dagger|e\\rangle}{z+(E_m-E_e)}
    \\Big]

with ``A = c^\\dagger`` for the one-particle Green's function (the second, "removal" term is
the anticommutator partner) or an arbitrary transition operator ``T`` for a spectrum (addition
term only: ``G_T(z) = <T^dagger (z - H + E)^{-1} T>``).

Row order note: ``Basis`` sorts its ``initial_basis``; every vector here is indexed by
``basis.local_basis`` order, never by construction order.
"""

import functools
import itertools

import numpy as np
from mpi4py import MPI

from impurityModel.ed.basis_transcription import build_dense_matrix
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant


def _det(occupied, n_orb):
    """Determinant over ``n_orb`` orbitals (MSB-first per byte: orbital i = bit 7 - i % 8)."""
    raw = bytearray((n_orb + 7) // 8)
    for i in occupied:
        raw[i // 8] |= 1 << (7 - i % 8)
    return SlaterDeterminant.from_bytes(bytes(raw))


class Sector:
    """One particle-number sector: its determinants (``Basis`` order), eigenvalues, eigenvectors."""

    def __init__(self, hOp, impurity_orbitals, bath_states, n_orb, n):
        dets = [_det(c, n_orb) for c in itertools.combinations(range(n_orb), n)]
        basis = Basis(impurity_orbitals, bath_states, initial_basis=dets, comm=MPI.COMM_SELF, verbose=False)
        self.n = n
        self.dets = list(basis.local_basis)
        self.index = {d: i for i, d in enumerate(self.dets)}
        H = np.asarray(build_dense_matrix(basis, hOp))
        assert np.allclose(H, H.conj().T), "oracle Hamiltonian is not Hermitian"
        self.e, self.v = np.linalg.eigh(H)

    def __len__(self):
        return len(self.dets)

    def state(self, m, prune=1e-14):
        """Eigenvector ``m`` as a ManyBodyState (amplitudes below ``prune`` dropped)."""
        return ManyBodyState({d: complex(a) for d, a in zip(self.dets, self.v[:, m]) if abs(a) > prune})


def _op_matrix(op, src, dst):
    """Dense ``<dst det| op |src det>`` (dst x src), then rotated to the two eigenbases."""
    M = np.zeros((len(dst), len(src)), dtype=complex)
    for j, d in enumerate(src.dets):
        out = op(ManyBodyState({d: 1.0}), 0)
        for det, amp in out.to_dict().items():
            M[dst.index[det], j] += complex(np.ravel(amp)[0])
    return dst.v.conj().T @ M @ src.v


class LehmannOracle:
    """Exact thermal Green's functions of ``hOp`` from sectors ``n0 - 1 .. n0 + 1``."""

    def __init__(self, hOp, impurity_orbitals, bath_states, n_orb, n0):
        self.hOp = hOp
        self.n_orb = n_orb
        self.sectors = {
            n: Sector(hOp, impurity_orbitals, bath_states, n_orb, n) for n in (n0 - 1, n0, n0 + 1) if 0 <= n <= n_orb
        }
        self.n0 = n0

    @property
    def gs(self):
        return self.sectors[self.n0]

    def thermal_states(self, count):
        """Indices (in the ``n0`` sector) and energies of the ``count`` lowest eigenstates."""
        idx = list(range(count))
        return idx, self.gs.e[idx]

    def psis(self, idx):
        return [self.gs.state(m) for m in idx]

    def _weights(self, idx, tau):
        es = self.gs.e[idx]
        w = np.exp(-(es - es.min()) / tau)
        return w / w.sum()

    def greens_function(self, orbitals, idx, tau, z):
        """One-particle ``G_ab(z)`` (``a, b`` in ``orbitals``), thermal over ``idx``."""
        z = np.asarray(z)
        w = self._weights(idx, tau)
        n0 = self.n0
        plus, minus = self.sectors.get(n0 + 1), self.sectors.get(n0 - 1)
        cdag = {o: _op_matrix(ManyBodyOperator({((o, "c"),): 1.0}), self.gs, plus) for o in orbitals} if plus else {}
        c = {o: _op_matrix(ManyBodyOperator({((o, "a"),): 1.0}), self.gs, minus) for o in orbitals} if minus else {}
        G = np.zeros((len(z), len(orbitals), len(orbitals)), dtype=complex)
        for wi, e in zip(w, idx):
            E = self.gs.e[e]
            if plus is not None:
                den = 1.0 / (z[:, None] - (plus.e[None, :] - E))  # (nz, m)
                for ai, a in enumerate(orbitals):
                    for bi, b in enumerate(orbitals):
                        # <e|c_a|m><m|c_b^dag|e> = conj(<m|c_a^dag|e>) <m|c_b^dag|e>
                        G[:, ai, bi] += wi * den @ (cdag[a][:, e].conj() * cdag[b][:, e])
            if minus is not None:
                den = 1.0 / (z[:, None] + (minus.e[None, :] - E))
                for ai, a in enumerate(orbitals):
                    for bi, b in enumerate(orbitals):
                        # <e|c_b^dag|m><m|c_a|e> = conj(<m|c_b|e>) <m|c_a|e>
                        G[:, ai, bi] += wi * den @ (c[b][:, e].conj() * c[a][:, e])
        return G

    def transition_tensor(self, tOps, n_shift, idx, tau, z, sign=+1):
        """``chi_ab(z) = <T_a^dag (z - sign (H - E))^{-1} T_b>`` thermal over ``idx``.

        Every ``T`` changes the particle number by ``n_shift``. ``sign=-1`` with ``z -> -z`` is
        the removal-side convention the spectra drivers use.
        """
        z = np.asarray(z)
        w = self._weights(idx, tau)
        dst = self.sectors[self.n0 + n_shift]
        Ts = [_op_matrix(t, self.gs, dst) for t in tOps]
        out = np.zeros((len(z), len(tOps), len(tOps)), dtype=complex)
        for wi, e in zip(w, idx):
            den = 1.0 / (z[:, None] - sign * (dst.e[None, :] - self.gs.e[e]))
            for a, Ta in enumerate(Ts):
                for b, Tb in enumerate(Ts):
                    out[:, a, b] += wi * den @ (Ta[:, e].conj() * Tb[:, e])
        return out


def distribute(basis, psis):
    """Hash-distribute replicated states onto ``basis`` (rank 0 owns the amplitudes)."""
    comm = basis.comm
    if comm is None or comm.size == 1:
        return list(psis)
    blocks = [ManyBodyState.from_states([p]) if comm.rank == 0 else ManyBodyState(width=1) for p in psis]
    return [blk.to_states()[0] for blk in basis.redistribute_psis(*blocks)]


# --- Models ---------------------------------------------------------------------------------


def two_orbital_model(n_bath_per_orb=1, u=2.5, t=0.2 + 0.25j):
    """Two impurity spin-orbital pairs with intra-pair hopping ``t`` and density-density ``u``.

    Impurity orbitals 0,1 (block A) and 2,3 (block B) -- blocks [[0, 1], [2, 3]] are exact
    one-body symmetry blocks. Each impurity orbital hybridizes with ``n_bath_per_orb`` bath
    orbitals (a short chain), valence below / conduction above the Fermi level, so both
    spectral sides carry weight and the Krylov spaces do not close trivially.

    Returns ``(hOp, impurity_orbitals, bath_states, n_orb, n0, blocks)``.
    """
    eps = [-0.6, -0.4, -0.5, -0.3]
    terms = {}
    for o, e in enumerate(eps):
        terms[((o, "c"), (o, "a"))] = e
    for a, b in ((0, 1), (2, 3)):
        # Complex hopping: G_ab != G_ba, so a dropped/misplaced transpose is visible.
        terms[((a, "c"), (b, "a"))] = t
        terms[((b, "c"), (a, "a"))] = np.conj(t)
    for a, b in itertools.combinations(range(4), 2):
        terms[((a, "c"), (b, "c"), (b, "a"), (a, "a"))] = u
    val, con = [], []
    nxt = 4
    for o in range(4):
        for k in range(n_bath_per_orb):
            e_b = (-2.0 if (k + o) % 2 == 0 else 1.8) + 0.3 * k + 0.05 * o
            v = 0.45 / (1 + k)
            terms[((nxt, "c"), (nxt, "a"))] = e_b
            src = o if k < 2 else nxt - 4
            terms[((src, "c"), (nxt, "a"))] = v
            terms[((nxt, "c"), (src, "a"))] = v
            (val if e_b < 0 else con).append(nxt)
            nxt += 1
    n_orb = nxt
    impurity_orbitals = {0: [[0, 1, 2, 3]]}
    bath_states = ({0: [val]}, {0: [con]})
    # Half filling of the impurity (2) plus the valence bath.
    n0 = 2 + len(val)
    return ManyBodyOperator(terms), impurity_orbitals, bath_states, n_orb, n0, [[0, 1], [2, 3]]


@functools.lru_cache(maxsize=None)
def cached_oracle(n_bath_per_orb):
    hOp, imp, baths, n_orb, n0, blocks = two_orbital_model(n_bath_per_orb)
    return LehmannOracle(hOp, imp, baths, n_orb, n0), (hOp, imp, baths, n_orb, n0, blocks)
