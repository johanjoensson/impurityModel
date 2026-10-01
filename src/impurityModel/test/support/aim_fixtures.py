"""Small Anderson-impurity fixtures with an exact dense reference, for the basis-size comparison.

``doc/plans/gf_basis_size_comparison.md`` asks how large a basis each Green's-function method needs
for a given error. That question needs models where (a) the answer is known exactly, (b) the
support is large enough for a cap ladder to have room, and (c) a physical knob -- the hybridization
-- moves the answer. This module builds those and nothing else: no solver logic lives here.

**F-NiO** (``build_nio_like``): two e_g-like orbitals (four spin-orbitals) with Kanamori ``U, J``,
each impurity spin-orbital hybridized with its own star of ``n_b`` filled bath levels. Nominally
``d8`` (two impurity electrons), so the ground state is a few-hole problem like the production NiO
workload: the reachable support is the 2-3-hole space of the bath, a few thousand determinants at
``n_b = 9``. ``V_eff`` (the per-spin-orbital hybridization strength) is **calibrated to the weight
of the charge-transfer configuration** ``d9L``, not quoted as a nominal ``V/Delta``: with a bath
as wide as the charge-transfer energy a nominal ratio means little.

The exact reference is one dense eigendecomposition per sector (``reference_G``), evaluated at any
number of frequencies in ``O(N nz)``. It follows the driver's own convention,
``G = (G_add - G_rem^T) / Z`` (``gf_engine.combine_sides``)::

    G_ij(z) = <psi| c_i (z - H + E0)^-1 c_j^+ |psi>  +  <psi| c_j^+ (z + H - E0)^-1 c_i |psi>
"""

import itertools
from dataclasses import dataclass, field

import numpy as np

from impurityModel.ed.basis_transcription import build_dense_matrix
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant

N_IMP = 4  # two orbitals x two spins; index = 2 * spin + orbital, spin 0 = up


def _det(occupied, n_orb):
    """Determinant with the given orbitals occupied.

    MSB-first: orbital ``i`` is bit ``7 - i % 8`` of byte ``i // 8``.
    """
    buf = bytearray((n_orb + 7) // 8)
    for i in occupied:
        buf[i // 8] |= 1 << (7 - i % 8)
    return SlaterDeterminant.from_bytes(bytes(buf))


def spin_of(orbital, n_b, n_imp=N_IMP):
    """0 = up, 1 = down, for an impurity or bath spin-orbital of the layout below.

    Impurity spin-orbital ``m`` is up for ``m < n_imp / 2`` and down after; its bath star follows the
    impurity block, ``n_b`` levels per impurity spin-orbital, and carries its spin."""
    m = orbital if orbital < n_imp else (orbital - n_imp) // n_b
    return 0 if m < n_imp // 2 else 1


@dataclass
class AIM:
    """A built model: the operator, its layout, the ground state and its exact sector spectra."""

    hOp: ManyBodyOperator
    n_b: int
    params: dict
    gs: ManyBodyState = None
    e0: float = 0.0
    weights: dict = field(default_factory=dict)
    _sector_cache: dict = field(default_factory=dict)
    # Layout and bath data: what the reference and the free Green's function need, so that they work
    # for any model built on this layout rather than only the one that was written first.
    n_imp: int = N_IMP
    n_electrons: int = 0
    sector0: tuple = None  # (n_up, n_dn) of the ground-state sector
    eps_d: float = 0.0
    levels: np.ndarray = None  # one impurity spin-orbital's bath levels
    couplings: np.ndarray = None  # ... and their hybridization amplitudes
    exact_e0: float = None  # the exact sector minimum when the ground state was truncated

    @property
    def n_orb(self):
        return self.n_imp * (1 + self.n_b)

    @property
    def imp(self):
        return list(range(self.n_imp))

    @property
    def bath(self):
        return list(range(self.n_imp, self.n_orb))

    def basis(self, initial, truncation_threshold=np.inf, comm=None):
        """A ``Basis`` over this model's orbital layout (impurity block, filled valence bath)."""
        return Basis(
            impurity_orbitals={0: [self.imp]},
            bath_states=({0: [self.bath]}, {0: [[]]}),
            initial_basis=initial,
            truncation_threshold=truncation_threshold,
            comm=comm,
            verbose=False,
        )

    def sector(self, n_up, n_dn):
        """All determinants with the given spin populations, sorted."""
        up = [i for i in range(self.n_orb) if spin_of(i, self.n_b, self.n_imp) == 0]
        dn = [i for i in range(self.n_orb) if spin_of(i, self.n_b, self.n_imp) == 1]
        dets = [
            _det(tuple(a) + tuple(b), self.n_orb)
            for a in itertools.combinations(up, n_up)
            for b in itertools.combinations(dn, n_dn)
        ]
        return sorted(dets)

    def dense(self, dets):
        """``(H, index)``: the dense Hamiltonian on ``dets`` and its determinant -> row map.

        Returned real when it is: a real symmetric ``eigh`` is several times faster than a complex
        Hermitian one at these sizes, and the diagonalizations are the cost of the whole fixture."""
        basis = self.basis(dets)
        H = np.asarray(build_dense_matrix(basis, self.hOp))
        if not np.iscomplexobj(H) or np.max(np.abs(H.imag)) == 0.0:
            H = np.ascontiguousarray(H.real)
        return H, {bytes(d.to_bytearray()): i for i, d in enumerate(dets)}


def kanamori_terms(U, J, n_b):
    """Kanamori interaction on the four impurity spin-orbitals, as ``c^+_i c^+_j c_l c_k`` terms.

    ``U`` intra-orbital, ``U' = U - 2J`` inter-orbital opposite spin, ``U' - J`` inter-orbital same
    spin, plus spin flip and pair hop with amplitude ``J``.
    """
    up = lambda a: a  # noqa: E731
    dn = lambda a: 2 + a  # noqa: E731
    Up = U - 2.0 * J
    terms = {}

    def add(i, j, k, l_, coef):
        """``coef * c^+_i c^+_j c_l c_k``"""
        key = ((i, "c"), (j, "c"), (l_, "a"), (k, "a"))
        terms[key] = terms.get(key, 0.0) + coef

    def density(p, q, coef):  # coef * n_p n_q = coef * c^+_p c^+_q c_q c_p
        add(p, q, p, q, coef)

    for a in (0, 1):
        density(up(a), dn(a), U)
    for a, b in ((0, 1), (1, 0)):
        density(up(a), dn(b), Up)
    density(up(0), up(1), Up - J)
    density(dn(0), dn(1), Up - J)
    for a, b in ((0, 1), (1, 0)):
        # spin flip  -J c^+_{a up} c_{a dn} c^+_{b dn} c_{b up} = -J c^+_{a up} c^+_{b dn} c_{b up} c_{a dn}
        add(up(a), dn(b), dn(a), up(b), -J)
        # pair hop    J c^+_{a up} c^+_{a dn} c_{b dn} c_{b up}
        add(up(a), dn(a), up(b), dn(b), +J)
    return terms


def interaction_increment(U, J):
    """``E_int(d9) - E_int(d8)``: lowest 3-electron minus lowest 2-electron interaction energy of the
    isolated Kanamori impurity (both at ``eps_d = 0``), by exact diagonalization of its 4 spin-orbitals.

    The charge-transfer energy of the full model at ``V = 0`` is ``eps_d - eps_b + this``, so this is
    what places the impurity level for a requested charge-transfer energy."""
    terms = kanamori_terms(U, J, 0)
    basis_orbitals = {0: [list(range(N_IMP))]}
    lowest = {}
    for n in (2, 3):
        dets = sorted(_det(c, N_IMP) for c in itertools.combinations(range(N_IMP), n))
        basis = Basis(
            impurity_orbitals=basis_orbitals,
            bath_states=({0: [[]]}, {0: [[]]}),
            initial_basis=dets,
            comm=None,
            verbose=False,
        )
        lowest[n] = float(np.linalg.eigvalsh(np.asarray(build_dense_matrix(basis, ManyBodyOperator(terms))))[0])
    return lowest[3] - lowest[2]


SPECTATOR_COUPLING = 1e-5
SPECTATOR_LEVEL = -15.0


def bath_star(n_b, bath_center, width, v_eff, n_spectator):
    """``(levels, couplings)`` of one impurity spin-orbital's bath: ``n_b`` filled levels in all.

    The first ``n_b - n_spectator`` are the physical bath, evenly spread over ``width`` around
    ``bath_center`` with total hybridization ``v_eff`` (``V_k = v_eff / sqrt(n_main)``). The last
    ``n_spectator`` sit far below at ``SPECTATOR_LEVEL - k`` with a coupling of ``1e-5``: real
    connectivity, negligible weight -- the determinants the closure holds and G does not need.
    """
    n_main = n_b - n_spectator
    main = bath_center + width * np.linspace(-0.5, 0.5, n_main) if n_main > 1 else np.array([bath_center])
    spec = SPECTATOR_LEVEL - np.arange(n_spectator, dtype=float)
    levels = np.concatenate([main, spec])
    couplings = np.concatenate([np.full(n_main, v_eff / np.sqrt(n_main)), np.full(n_spectator, SPECTATOR_COUPLING)])
    return levels, couplings


def impurity_level(U, J, delta_ct, bath_center):
    """``eps_d`` such that ``E(d9L) - E(d8) = delta_ct`` at ``V = 0`` with the bath centred at ``bath_center``."""
    return delta_ct + bath_center - interaction_increment(U, J)


def nio_like_operator(n_b, U=8.0, J=1.0, delta_ct=4.0, bath_center=0.0, width=3.0, v_eff=1.0, n_spectator=0):
    """Hamiltonian of the F-NiO fixture: Kanamori impurity plus one filled star bath per spin-orbital."""
    eps_d = impurity_level(U, J, delta_ct, bath_center)
    terms = kanamori_terms(U, J, n_b)
    levels, couplings = bath_star(n_b, bath_center, width, v_eff, n_spectator)
    for m in range(N_IMP):
        terms[((m, "c"), (m, "a"))] = eps_d
        for k, (eps, v) in enumerate(zip(levels, couplings)):
            b = N_IMP + m * n_b + k
            terms[((b, "c"), (b, "a"))] = float(eps)
            terms[((m, "c"), (b, "a"))] = float(v)
            terms[((b, "c"), (m, "a"))] = float(v)
    return ManyBodyOperator(terms)


def _n_imp_weights(aim, vec, dets):
    """Probability of each impurity electron count in a ground-state vector over ``dets``."""
    counts = {}
    for det, amp in zip(dets, vec):
        bits = bytes(det.to_bytearray())
        n = sum(1 for i in aim.imp if bits[i // 8] & (1 << (7 - i % 8)))
        counts[n] = counts.get(n, 0.0) + float(abs(amp) ** 2)
    return counts


def ground_state(aim, keep=None):
    """Fill ``aim.gs``, ``aim.e0`` and ``aim.weights`` from the ground-state sector ``aim.sector0``.

    With ``keep`` set the state is **truncated**, the way a CIPSI ground state is: the lowest eigenstate
    of ``H`` restricted to the ``keep`` determinants of largest amplitude in the exact one. It is not an
    eigenstate of ``H``, but ``G_ij(z) = <s_i|(z - H + E)^-1|s_j>`` is well defined for any seeds and
    energy, so the exact reference stays exact -- and the seed support stays small next to the
    closure, which is the regime of the production calculations (an exact ground state in a metal
    spreads over the whole sector, and the seeds alone then fill it).
    """
    n_up, n_dn = aim.sector0
    dets = aim.sector(n_up, n_dn)
    H, index = aim.dense(dets)
    evals, evecs = np.linalg.eigh(H)
    vec = evecs[:, 0]
    aim.exact_e0 = float(evals[0])
    aim.e0 = float(evals[0])
    if keep is not None and keep < len(dets):
        top = np.sort(np.argsort(-np.abs(vec))[:keep])
        sub_evals, sub_evecs = np.linalg.eigh(H[np.ix_(top, top)])
        vec = np.zeros(len(dets))
        vec[top] = sub_evecs[:, 0]
        aim.e0 = float(sub_evals[0])
    aim.gs = ManyBodyState({det: complex(vec[i]) for i, det in enumerate(dets) if abs(vec[i]) > 0.0})
    aim.weights = _n_imp_weights(aim, vec, dets)
    aim._sector_cache[(n_up, n_dn)] = (H, index, evals, evecs, dets)
    return aim


def charge_transfer_weight(aim):
    """Weight of the ``d9L`` configuration (three impurity electrons) in the ground state."""
    return aim.weights.get(3, 0.0)


def build_nio_like(n_b=9, target_d9L=None, v_eff=1.0, **kwargs):
    """The F-NiO fixture. With ``target_d9L`` set, ``v_eff`` is solved for that ``d9L`` weight."""
    params = dict(kwargs)

    def make(v):
        p = dict(params, v_eff=v)
        center = p.get("bath_center", 0.0)
        levels, couplings = bath_star(n_b, center, p.get("width", 3.0), v, p.get("n_spectator", 0))
        n_up0 = 2 + 2 * n_b  # both impurity electrons up: the up shell is full
        n_electrons = N_IMP // 2 + N_IMP * n_b  # d8 nominal: two impurity electrons, filled bath
        aim = AIM(
            nio_like_operator(n_b, v_eff=v, **params),
            n_b,
            p,
            n_electrons=n_electrons,
            sector0=(n_up0, n_electrons - n_up0),
            eps_d=impurity_level(p.get("U", 8.0), p.get("J", 1.0), p.get("delta_ct", 4.0), center),
            levels=levels,
            couplings=couplings,
        )
        return ground_state(aim)

    if target_d9L is None:
        return make(v_eff)
    lo, hi = 1e-3, 6.0
    for _ in range(40):  # the weight is monotone in V over this range
        mid = 0.5 * (lo + hi)
        if charge_transfer_weight(make(mid)) < target_d9L:
            lo = mid
        else:
            hi = mid
    return make(0.5 * (lo + hi))


def semicircle_star(n_b, D, v):
    """``(levels, couplings)`` of a ``n_b``-level star discretizing a semicircular hybridization.

    Gauss-Chebyshev quadrature of the second kind: nodes ``D cos(k pi / (n_b + 1))`` and weights
    ``2 sin^2(k pi / (n_b + 1)) / (n_b + 1)``, which sum to one, so ``sum_k V_k^2 = v^2`` and
    ``Delta(w) -> v^2 / w`` at large ``w`` like the continuum ``2 v^2 / D^2 (w - sqrt(w^2 - D^2))``.
    The levels are symmetric about zero (with one at zero for odd ``n_b``), as a particle-hole
    symmetric model needs.
    """
    theta = np.arange(1, n_b + 1) * np.pi / (n_b + 1)
    return D * np.cos(theta), v * np.sqrt(2.0 * np.sin(theta) ** 2 / (n_b + 1))


def hubbard_terms(U):
    """Single-orbital Hubbard interaction ``U n_up n_dn`` on impurity spin-orbitals 0 (up) and 1 (down)."""
    return {((0, "c"), (1, "c"), (1, "a"), (0, "a")): U}


def semicircle_operator(n_b, U, D, v):
    """Single-impurity Anderson model with a semicircular star bath, at its particle-hole symmetric point.

    The impurity level sits at ``-U/2`` (the ``[double_counting.nominal]`` shift of
    ``examples/semicircular_siam``), so the half-filled ground state has ``<n_imp> = 1`` and
    ``Re Sigma(i w_n) = U / 2`` exactly.
    """
    n_imp = 2
    levels, couplings = semicircle_star(n_b, D, v)
    terms = hubbard_terms(U)
    for m in range(n_imp):
        terms[((m, "c"), (m, "a"))] = -U / 2
        for k, (eps, w) in enumerate(zip(levels, couplings)):
            b = n_imp + m * n_b + k
            terms[((b, "c"), (b, "a"))] = float(eps)
            terms[((m, "c"), (b, "a"))] = float(w)
            terms[((b, "c"), (m, "a"))] = float(w)
    return ManyBodyOperator(terms)


def build_semicircle_siam(n_b=7, U=0.5, D=0.5, v=0.5, gs_keep=None):
    """The F-metal fixture: a half-filled SIAM on a semicircular star, a scaled-down ``semicircular_siam``.

    ``n_b`` (odd) levels per spin, so ``2 (1 + n_b)`` spin-orbitals and ``1 + n_b`` electrons in the
    ``(n_up, n_dn) = ((1 + n_b) / 2, (1 + n_b) / 2)`` singlet sector. ``gs_keep`` truncates the ground
    state to that many determinants (see :func:`ground_state`).
    """
    if n_b % 2 == 0:
        raise ValueError("n_b must be odd: the half-filled sector needs (1 + n_b) / 2 electrons per spin")
    levels, couplings = semicircle_star(n_b, D, v)
    n_up0 = (1 + n_b) // 2
    aim = AIM(
        semicircle_operator(n_b, U, D, v),
        n_b,
        {"U": U, "D": D, "v_eff": v},
        n_imp=2,
        n_electrons=1 + n_b,
        sector0=(n_up0, n_up0),
        eps_d=-U / 2,
        levels=levels,
        couplings=couplings,
    )
    return ground_state(aim, keep=gs_keep)


def _seeds_for(aim, side):
    """Seed columns ``c^+_i|gs>`` (``side = 0``) or ``c_i|gs>`` (``side = 1``) for the impurity block."""
    kind = "c" if side == 0 else "a"
    out = []
    for i in aim.imp:
        op = ManyBodyOperator({((i, kind),): 1.0})
        out.append(op.apply_block(ManyBodyState.from_states([aim.gs]), 0.0).to_states()[0])
    return out


def reference_G(aim, z):
    """Exact ``G(z)`` of the impurity block, shape ``(len(z), n_imp, n_imp)``, by dense diagonalization.

    One ``eigh`` per reachable sector (cached on ``aim``), then ``O(N nz)`` per frequency.
    """
    z = np.atleast_1d(np.asarray(z, dtype=complex))
    n_up0, n_dn0 = aim.sector0
    n_imp = aim.n_imp
    out = np.zeros((len(z), n_imp, n_imp), dtype=complex)
    for side, sectors in ((0, [(n_up0 + 1, n_dn0), (n_up0, n_dn0 + 1)]), (1, [(n_up0 - 1, n_dn0), (n_up0, n_dn0 - 1)])):
        seeds = _seeds_for(aim, side)
        for sector in sectors:
            n_up, n_dn = sector
            if n_up < 0 or n_dn < 0:
                continue
            if sector not in aim._sector_cache:
                dets = aim.sector(n_up, n_dn)
                if not dets:
                    continue
                H, index = aim.dense(dets)
                evals, evecs = np.linalg.eigh(H)
                aim._sector_cache[sector] = (H, index, evals, evecs, dets)
            _H, index, evals, evecs, dets = aim._sector_cache[sector]
            V = np.zeros((len(dets), n_imp), dtype=complex)
            for j, s in enumerate(seeds):
                for det, amp in s.items():
                    key = bytes(det.to_bytearray())
                    if key in index:
                        V[index[key], j] = amp[0]
            W = evecs.conj().T @ V  # (n, n_imp): <n| seed_j>
            if side == 0:
                # G_add[i, j] = sum_n conj(W_in) W_jn / (z - lam_n + E0)
                denom = z[:, None] - evals[None, :] + aim.e0
                out += np.einsum("ni,zn,nj->zij", W.conj(), 1.0 / denom, W)
            else:
                # term[i, j] = sum_n W_in conj(W_jn) / (z + lam_n - E0)
                denom = z[:, None] + evals[None, :] - aim.e0
                out += np.einsum("ni,zn,nj->zij", W, 1.0 / denom, W.conj())
    return out


def free_G_inverse(aim, z):
    """``G0^-1(z) = z - eps_d - Delta(z)`` of the non-interacting impurity, shape ``(len(z), n_imp, n_imp)``.

    Diagonal: every impurity spin-orbital has its own star, identical for all of them."""
    z = np.atleast_1d(np.asarray(z, dtype=complex))
    delta = np.sum(aim.couplings[None, :] ** 2 / (z[:, None] - aim.levels[None, :]), axis=1)
    out = np.zeros((len(z), aim.n_imp, aim.n_imp), dtype=complex)
    for m in range(aim.n_imp):
        out[:, m, m] = z - aim.eps_d - delta
    return out


def self_energy(G, G0_inv):
    """``Sigma = G0^-1 - G^-1`` per frequency."""
    return G0_inv - np.linalg.inv(G)
