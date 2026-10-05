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
    h1: np.ndarray = None  # the one-body matrix (real symmetric): impurity block, coupling, bath block
    interaction: dict = None  # the two-body terms, which live on the impurity only
    basis_name: str = "star"
    keep: int = None  # determinants kept in the ground state (None = exact)
    fermi: float = 0.0  # the chemical potential: bath levels below it are filled (used by the linked chain)

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


def operator_from(h1, interaction):
    """``ManyBodyOperator`` of a one-body matrix ``h1`` (``sum_ij h1[i, j] c^+_i c_j``) plus two-body terms."""
    terms = dict(interaction)
    for i, j in zip(*np.nonzero(h1)):
        terms[((int(i), "c"), (int(j), "a"))] = float(h1[i, j])
    return ManyBodyOperator(terms)


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


def nio_like_h1(n_b, U=8.0, J=1.0, delta_ct=4.0, bath_center=0.0, width=3.0, v_eff=1.0, n_spectator=0):
    """``(h1, interaction)`` of the F-NiO fixture: Kanamori impurity plus one filled star bath per spin-orbital."""
    eps_d = impurity_level(U, J, delta_ct, bath_center)
    levels, couplings = bath_star(n_b, bath_center, width, v_eff, n_spectator)
    h1 = np.zeros((N_IMP * (1 + n_b),) * 2)
    for m in range(N_IMP):
        h1[m, m] = eps_d
        for k, (eps, v) in enumerate(zip(levels, couplings)):
            b = N_IMP + m * n_b + k
            h1[b, b] = eps
            h1[m, b] = h1[b, m] = v
    return h1, kanamori_terms(U, J, n_b)


def nio_like_operator(n_b, **kwargs):
    """Hamiltonian of the F-NiO fixture."""
    return operator_from(*nio_like_h1(n_b, **kwargs))


def _n_imp_weights(aim, vec, dets):
    """Probability of each impurity electron count in a ground-state vector over ``dets``."""
    counts = {}
    for det, amp in zip(dets, vec):
        bits = bytes(det.to_bytearray())
        n = sum(1 for i in aim.imp if bits[i // 8] & (1 << (7 - i % 8)))
        counts[n] = counts.get(n, 0.0) + float(abs(amp) ** 2)
    return counts


def ground_state(aim, keep=None, lost_weight=None):
    """Fill ``aim.gs``, ``aim.e0`` and ``aim.weights`` from the ground-state sector ``aim.sector0``.

    With ``keep`` set the state is **truncated**, the way a CIPSI ground state is: the lowest eigenstate
    of ``H`` restricted to the ``keep`` determinants of largest amplitude in the exact one. It is not an
    eigenstate of ``H``, but ``G_ij(z) = <s_i|(z - H + E)^-1|s_j>`` is well defined for any seeds and
    energy, so the exact reference stays exact -- and the seed support stays small next to the
    closure, which is the regime of the production calculations (an exact ground state in a metal
    spreads over the whole sector, and the seeds alone then fill it).

    ``lost_weight`` sizes the truncation by accuracy instead: the smallest ``keep`` whose discarded
    weight is at most this, which is how a CIPSI state is sized and which depends strongly on the
    orbital basis.
    """
    n_up, n_dn = aim.sector0
    dets = aim.sector(n_up, n_dn)
    H, index = aim.dense(dets)
    evals, evecs = np.linalg.eigh(H)
    vec = evecs[:, 0]
    aim.exact_e0 = float(evals[0])
    aim.e0 = float(evals[0])
    if lost_weight is not None:
        keep = _keep_for_weight(vec, lost_weight)
    aim.keep = keep
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
        h1, interaction = nio_like_h1(n_b, v_eff=v, **params)
        aim = AIM(
            operator_from(h1, interaction),
            n_b,
            p,
            n_electrons=n_electrons,
            sector0=(n_up0, n_electrons - n_up0),
            eps_d=impurity_level(p.get("U", 8.0), p.get("J", 1.0), p.get("delta_ct", 4.0), center),
            levels=levels,
            couplings=couplings,
            h1=h1,
            interaction=interaction,
            # the bath is filled to its top by construction, so the Fermi level sits just above it
            fermi=center + p.get("width", 3.0) / 2 + 0.5,
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


def semicircle_h1(n_b, U, D, v):
    """``(h1, interaction)`` of the single-impurity Anderson model on a semicircular star, at its
    particle-hole symmetric point.

    The impurity level sits at ``-U/2`` (the ``[double_counting.nominal]`` shift of
    ``examples/semicircular_siam``), so the half-filled ground state has ``<n_imp> = 1`` and
    ``Re Sigma(i w_n) = U / 2`` exactly.
    """
    n_imp = 2
    levels, couplings = semicircle_star(n_b, D, v)
    h1 = np.zeros((n_imp * (1 + n_b),) * 2)
    for m in range(n_imp):
        h1[m, m] = -U / 2
        for k, (eps, w) in enumerate(zip(levels, couplings)):
            b = n_imp + m * n_b + k
            h1[b, b] = eps
            h1[m, b] = h1[b, m] = w
    return h1, hubbard_terms(U)


def semicircle_operator(n_b, U, D, v):
    return operator_from(*semicircle_h1(n_b, U, D, v))


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
    h1, interaction = semicircle_h1(n_b, U, D, v)
    aim = AIM(
        operator_from(h1, interaction),
        n_b,
        {"U": U, "D": D, "v_eff": v},
        h1=h1,
        interaction=interaction,
        n_imp=2,
        n_electrons=1 + n_b,
        sector0=(n_up0, n_up0),
        eps_d=-U / 2,
        levels=levels,
        couplings=couplings,
    )
    return ground_state(aim, keep=gs_keep)


# --- exact changes of the bath basis -------------------------------------------------------------------
#
# A unitary change of the bath orbitals (impurity fixed, spin preserved) leaves H unitarily equivalent and
# the impurity Green's function untouched, so with an exact ground state every basis has the SAME G. What
# changes is the determinant structure -- the support of the ground state, the seeds, the closure -- and
# that is what the basis-size comparison needs to vary. The interaction lives on the impurity, so it is
# not rotated.

from impurityModel.ed.ManyBodyUtils import block_inner_cy  # noqa: E402


def block_krylov_basis(h, start, tol=1e-10):
    """Orthogonal ``Q`` (``n x n``) whose first columns are the block-Lanczos chain of ``h`` from ``start``.

    ``Q^T h Q`` is block tridiagonal with the first block coupled to ``start`` (one chain per column of
    ``start``). Full reorthogonalization, so degeneracies and tiny couplings do not lose orthogonality; a
    Krylov space that closes early (a disconnected part of the bath) is completed with the remaining
    unit vectors, deterministically.
    """
    n = h.shape[0]
    cols = []

    def accept(block):
        new = []
        for v in np.atleast_2d(block.T):
            w = np.array(v, dtype=float)
            for _ in range(2):
                for q in cols + new:
                    w = w - q * (q @ w)
            norm = np.linalg.norm(w)
            if norm > tol:
                new.append(w / norm)
        return new

    current = accept(np.asarray(start, dtype=float).reshape(n, -1))
    cols += current
    while current and len(cols) < n:
        current = accept(h @ np.array(current).T)
        cols += current
    unit = 0
    while len(cols) < n:
        e = np.zeros(n)
        e[unit] = 1.0
        unit += 1
        cols += accept(e[:, None])
    return np.array(cols).T


def _bath_by_spin(aim):
    """``{spin: [positions within the bath block]}`` -- the bath orbitals of each spin, in index order."""
    out = {0: [], 1: []}
    for position, orbital in enumerate(aim.bath):
        out[spin_of(orbital, aim.n_b, aim.n_imp)].append(position)
    return out


def _impurity_by_spin(aim):
    return {s: [m for m in aim.imp if spin_of(m, aim.n_b, aim.n_imp) == s] for s in (0, 1)}


def chain_rotation(aim):
    """Per-channel chains: each impurity spin-orbital's star becomes a chain with the impurity at its end.

    The bath orbitals are rotated within each channel (``n_b`` consecutive bath orbitals) by the Lanczos
    tridiagonalization of that channel's levels from its coupling vector, so the impurity couples to the
    first chain site alone and the bath block is tridiagonal.
    """
    n = aim.n_imp
    U = np.eye(aim.n_orb - n)
    for m in aim.imp:
        sites = np.arange(m * aim.n_b, (m + 1) * aim.n_b)
        h = aim.h1[n:, n:][np.ix_(sites, sites)]
        v = aim.h1[m, n:][sites]
        if np.linalg.norm(v) > 0:
            U[np.ix_(sites, sites)] = block_krylov_basis(h, v[:, None])
    return U


def one_body_density(aim):
    """``rho_ij = <psi|c^+_i c_j|psi>`` over the bath orbitals of the EXACT ground state (shape ``n_bath^2``).

    Taken from the cached sector eigenvector, never from a truncated state: natural orbitals are the
    physical object, and a CIPSI run would get them from a first approximate density anyway.
    """
    _H, _index, _evals, evecs, dets = aim._sector_cache[aim.sector0]
    psi = ManyBodyState.from_states(
        [ManyBodyState({d: complex(evecs[k, 0]) for k, d in enumerate(dets) if abs(evecs[k, 0]) > 0})]
    )
    n_bath = len(aim.bath)
    rho = np.zeros((n_bath, n_bath))
    by_spin = _bath_by_spin(aim)
    for positions in by_spin.values():
        for i in positions:
            for j in positions:
                op = ManyBodyOperator({((aim.n_imp + i, "c"), (aim.n_imp + j, "a")): 1.0})
                rho[i, j] = float(np.real(block_inner_cy(psi, op.apply_block(psi, 0.0))[0, 0]))
    return rho


HALF_FILLED_TOL = 1e-8


def is_filled(occupations):
    """Which natural orbitals belong to the valence (filled) chain: occupation above 1/2.

    A particle-hole symmetric bath with an odd number of levels per spin has one natural orbital at
    exactly 1/2, which roundoff puts at ``0.5 +- 1e-16`` with a compiler-dependent sign. The band makes
    that tie go to the empty chain on every build, and the fixture and its tests share this one rule
    instead of each breaking the tie on its own.
    """
    return np.asarray(occupations) > 0.5 + HALF_FILLED_TOL


def natural_orbital_rotation(aim, chains=False):
    """Rotate each spin's bath to its natural orbitals (eigenvectors of the bath density matrix).

    Orbitals are ordered by occupation, filled first. With ``chains`` the filled (:func:`is_filled`) and the
    empty natural orbitals are each re-tridiagonalized from the impurity coupling -- the valence and
    conduction chains of the Haverkort construction -- so the impurity couples only to the head of each.
    Returns ``(U, occupations)``.
    """
    n = aim.n_imp
    rho = one_body_density(aim)
    h_bath = aim.h1[n:, n:]
    U = np.eye(len(aim.bath))
    occupations = np.zeros(len(aim.bath))
    for spin, positions in _bath_by_spin(aim).items():
        sub = rho[np.ix_(positions, positions)]
        occ, vecs = np.linalg.eigh(sub)
        order = np.argsort(-occ)
        occ, vecs = occ[order], vecs[:, order]
        occupations[positions] = occ
        if not chains:
            U[np.ix_(positions, positions)] = vecs
            continue
        imp = _impurity_by_spin(aim)[spin]
        coupling = aim.h1[np.ix_(imp, [n + p for p in positions])]  # (n_imp_spin, n_spin_bath)
        h_spin = h_bath[np.ix_(positions, positions)]
        blocks = []
        filled = is_filled(occ)
        for group in (filled, ~filled):
            if not np.any(group):
                continue
            Ug = vecs[:, group]
            Q = block_krylov_basis(Ug.T @ h_spin @ Ug, (coupling @ Ug).T)
            blocks.append(Ug @ Q)
        U[np.ix_(positions, positions)] = np.hstack(blocks)
    return U, occupations


def rotate_bath(aim, U, name, keep=None, lost_weight=None):
    """The same model in the bath basis given by the columns of ``U`` (orthogonal, spin-preserving).

    ``keep`` truncates the new ground state as in :func:`ground_state`: determinants are then the ones
    of the new basis, so a basis in which the ground state is compact loses less to the same ``keep``.
    """
    n = aim.n_imp
    spins = [spin_of(o, aim.n_b, aim.n_imp) for o in aim.bath]
    for i, si in enumerate(spins):
        for j, sj in enumerate(spins):
            if si != sj and abs(U[i, j]) > 1e-12:
                raise ValueError("a bath rotation must not mix spins: the sector structure would be lost")
    if not np.allclose(U.T @ U, np.eye(U.shape[0]), atol=1e-10):
        raise ValueError("U must be orthogonal")
    R = np.eye(aim.n_orb)
    R[n:, n:] = U
    h1 = R.T @ aim.h1 @ R
    return _aim_with_h1(aim, h1, name, keep, lost_weight)


def _aim_with_h1(aim, h1, name, keep=None, lost_weight=None):
    """The same model with a different one-body matrix (interaction, electrons and sector unchanged)."""
    h1 = np.array(h1)
    h1[np.abs(h1) < 1e-14] = 0.0
    new = AIM(
        operator_from(h1, aim.interaction),
        aim.n_b,
        dict(aim.params),
        n_imp=aim.n_imp,
        n_electrons=aim.n_electrons,
        sector0=aim.sector0,
        eps_d=aim.eps_d,
        h1=h1,
        interaction=aim.interaction,
        basis_name=name,
        fermi=aim.fermi,
    )
    return ground_state(new, keep=keep, lost_weight=lost_weight)


def linked_chain_h1(aim):
    """The one-body matrix of ``aim`` with each spin's star replaced by its linked double chain.

    Uses ``rspt2spectra.edchain.linked_double_chain`` (an optional dependency, imported here): the one-body
    eigenstates of the non-interacting impurity + star are split by the sign of their energy relative to
    ``aim.fermi`` into an occupied and an unoccupied part, each made a chain, and the impurity character
    is restored by an SVD -- no many-body input, unlike natural orbitals. The algorithm assumes the Fermi
    level at zero, so energies are measured from ``aim.fermi`` and put back afterwards (a constant shift
    commutes with the construction). Requires a star bath; keeps the impurity block as it was.
    """
    from rspt2spectra.edchain import linked_double_chain

    n = aim.n_imp
    h1 = aim.h1.copy()
    for spin, positions in _bath_by_spin(aim).items():
        cols = [n + p for p in positions]
        imp = _impurity_by_spin(aim)[spin]
        h_bath = aim.h1[np.ix_(cols, cols)]
        if np.max(np.abs(h_bath - np.diag(np.diag(h_bath)))) > 1e-12:
            raise ValueError("the linked chain is built from a star bath; this bath is not diagonal")
        eye_imp = np.eye(len(imp))
        v, hb = linked_double_chain(
            aim.h1[np.ix_(imp, imp)] - aim.fermi * eye_imp,
            aim.h1[np.ix_(cols, imp)],
            np.diag(h_bath) - aim.fermi,
            verbose=False,
        )
        hb = hb + aim.fermi * np.eye(len(cols))
        if np.iscomplexobj(hb) or np.iscomplexobj(v):
            if np.max(np.abs(np.imag(hb))) > 1e-10 or np.max(np.abs(np.imag(v))) > 1e-10:
                raise ValueError("the linked chain came out complex for a real model")
            hb, v = np.real(hb), np.real(v)
        h1[np.ix_(cols, cols)] = hb
        h1[np.ix_(cols, imp)] = v
        h1[np.ix_(imp, cols)] = v.T
    return h1


def _keep_for_weight(vec, lost_weight):
    weight = np.sort(vec**2)[::-1]
    tail = 1.0 - np.cumsum(weight)  # weight lost keeping the first k+1
    return int(np.argmax(tail <= lost_weight)) + 1 if np.any(tail <= lost_weight) else len(weight)


def keep_for_weight(aim, lost_weight):
    """The smallest ``keep`` whose top-``keep`` determinants of the exact ground state lose at most ``lost_weight``.

    This is how a CIPSI ground state is sized: by the accuracy wanted, not by a fixed count -- and the
    count it needs depends strongly on the orbital basis (see :func:`geometry_variants`).
    """
    return _keep_for_weight(aim._sector_cache[aim.sector0][3][:, 0], lost_weight)


BASES = ("star", "chain", "natural", "natural-chains")  # the self-contained bath bases
ALL_BASES = BASES + ("linked-chain",)  # + the one that needs rspt2spectra


def geometry_variants(aim, keep=None, lost_weight=None, names=BASES):
    """``{name: AIM}`` -- the model in the requested bath bases (default: star, chain, natural orbitals,
    natural orbitals re-chained; ``"linked-chain"`` additionally needs ``rspt2spectra``).

    ``aim`` must carry its exact ground state (built with ``keep=None``). Each variant is rebuilt with
    the truncation given by ``keep`` (a fixed count) or ``lost_weight`` (the count each basis needs for
    that accuracy), so the truncation acts on that basis's own determinants: a basis in which the ground
    state is compact needs far fewer of them.
    """
    n_bath = len(aim.bath)
    wanted = list(names)
    unknown = set(wanted) - set(ALL_BASES)
    if unknown:
        raise ValueError(f"unknown bath basis {sorted(unknown)}; expected a subset of {ALL_BASES}")
    out = {}
    for name in wanted:
        if name == "linked-chain":
            out[name] = _aim_with_h1(aim, linked_chain_h1(aim), name, keep=keep, lost_weight=lost_weight)
            continue
        if name == "star":
            U = np.eye(n_bath)
        elif name == "chain":
            U = chain_rotation(aim)
        elif name == "natural":
            U = natural_orbital_rotation(aim)[0]
        else:
            U = natural_orbital_rotation(aim, chains=True)[0]
        out[name] = rotate_bath(aim, U, name, keep=keep, lost_weight=lost_weight)
    return out


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
    """``G0^-1(z) = z - h_imp - V (z - h_bath)^-1 V^T`` of the non-interacting impurity.

    Written from the one-body matrix, so it is the same for every basis of the bath: a unitary change
    of the bath leaves ``V (z - h_bath)^-1 V^T`` -- the hybridization function -- untouched.
    """
    z = np.atleast_1d(np.asarray(z, dtype=complex))
    n = aim.n_imp
    h_imp, V, h_bath = aim.h1[:n, :n], aim.h1[:n, n:], aim.h1[n:, n:]
    eye_b = np.eye(h_bath.shape[0])
    out = np.empty((len(z), n, n), dtype=complex)
    for k, zk in enumerate(z):
        out[k] = zk * np.eye(n) - h_imp - V @ np.linalg.solve(zk * eye_b - h_bath, V.T)
    return out


def self_energy(G, G0_inv):
    """``Sigma = G0^-1 - G^-1`` per frequency."""
    return G0_inv - np.linalg.inv(G)
