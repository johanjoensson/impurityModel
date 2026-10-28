"""The ground-state memory model's workload inputs, measured on this calculation's own operator.

The ground-state cap (:func:`impurityModel.ed.memory_estimate.suggest_gs_truncation_threshold`) prices
three things that depend on the model and on nothing a user sets: the Hamiltonian's stored elements per
determinant, and the rows a CIPSI selection round's ``H|psi_ref>`` block reaches per determinant. Both were
constants measured once, on SrMnO3 (100 and 40), and applied to every model: 2.5-10x too high on NiO
(raw connectivity 16.7, block fan-out 4.3) and, on a model with a denser interaction, just as likely too
low. :func:`measure_gs_connectivity` measures them, once per calculation, the way the Green's-function cap
measures its fan-out (:func:`impurityModel.ed.gf_units.measure_gf_row_fanout`).

The probe is the breadth-first closure of the nominal-occupation seed under ``H`` up to a few thousand
determinants. It never looks at an energy, so it is invariant under the chemical-potential shift
``H -> H - mu N_imp`` that a double-counting search applies: the DC search and the self-energy resolve the
same cap from the same numbers.
"""

import time
from dataclasses import dataclass

import numpy as np

from impurityModel.ed.ManyBodyUtils import ManyBodyState

#: Determinants the breadth-first closure grows to. The fan-out of a block *falls* with its size
#: (SrMnO3: 13.3, 10.7, 8.1 at 5k, 20k, 100k determinants), so a probe smaller than the production basis
#: reads high: the conservative side for a memory estimate.
PROBE_DETERMINANTS = 15_000

#: Determinants whose individual ``H|d>`` row count is averaged for the stored-element estimate.
_RAW_SAMPLE = 2_000


@dataclass(frozen=True)
class GsConnectivity:
    """What the ground-state memory model needs to know about the operator.

    ``nnz_per_state`` is the mean number of distinct rows ``H|d>`` reaches, an upper bound on the stored
    elements per determinant at any basis size (restriction and truncation only remove rows).
    ``selection_fanout`` is the de-duplicated rows of one block ``H|psi_ref>`` over the whole probe, per
    determinant: the quantity a selection round's candidate arrays scale with. ``window_states`` is the
    degeneracy of the lowest level of ``H`` on the seed: how many states the window solve holds at least
    (0: not counted, which prices the window phase off). ``measured`` is False for
    :meth:`unmeasured`, which carries the constants measured on SrMnO3.
    """

    nnz_per_state: float
    selection_fanout: float
    probe_dets: int = 0
    seed_dets: int = 0
    seconds: float = 0.0
    measured: bool = True
    window_states: int = 0

    @classmethod
    def unmeasured(cls):
        """The SrMnO3 constants, for callers with no operator to measure."""
        from impurityModel.ed.memory_estimate import _UNMEASURED_NNZ_PER_STATE, _UNMEASURED_SELECTION_FANOUT

        return cls(_UNMEASURED_NNZ_PER_STATE, _UNMEASURED_SELECTION_FANOUT, measured=False)

    def describe(self):
        """One line for the memory-budget log."""
        if not self.measured:
            return (
                f"UNMEASURED (SrMnO3 constants): raw connectivity {self.nnz_per_state:g} rows/det, "
                f"selection fan-out {self.selection_fanout:g}; no operator was given to measure"
            )
        return (
            f"measured on this model: raw connectivity {self.nnz_per_state:.1f} rows/det, selection fan-out "
            f"{self.selection_fanout:.1f}, {self.window_states} states in the lowest level "
            f"(probe {self.probe_dets:,} dets from a {self.seed_dets:,}-det seed, {self.seconds:.1f} s)"
        )


def measure_gs_connectivity(h_op, impurity_orbitals, bath_states, nominal_occ, comm=None, *, target=PROBE_DETERMINANTS):
    """Measure :class:`GsConnectivity` of ``h_op`` on the nominal-occupation sector. Collective on ``comm``.

    Rank 0 builds the seed the ground state starts from (``build_basis_and_solver`` at ``nominal_occ``, no
    cap), closes it breadth-first under ``h_op`` to ``target`` determinants (hash-ordered, so the set does
    not depend on dictionary order), and broadcasts the numbers. The seed's sector has a fixed impurity
    occupation, so its energy order is the same at every chemical-potential shift.
    """
    result = None
    if comm is None or comm.rank == 0:
        result = _probe(h_op, impurity_orbitals, bath_states, nominal_occ, target)
    return comm.bcast(result, root=0) if comm is not None and comm.size > 1 else result


def _probe(h_op, impurity_orbitals, bath_states, nominal_occ, target):
    from impurityModel.ed.eigensolvers import Lowest
    from impurityModel.ed.groundstate import build_probe_seed

    start = time.perf_counter()
    basis, _solver = build_probe_seed(
        h_op, impurity_orbitals, bath_states, nominal_occ, None, 0, True, None, None, False, None, None, 0,
        manifold=Lowest(),
    )  # fmt: skip
    h_op.set_restrictions(basis.restrictions)
    try:
        seed = list(basis.local_basis)
        window = _lowest_level_degeneracy(h_op, basis)
        members = set(seed)
        frontier = seed
        while frontier and len(members) < target:
            image = h_op.apply_block(ManyBodyState(dict.fromkeys(frontier, 1.0)), 0.0)
            new = sorted((k for k in image.keys() if k not in members), key=lambda k: k.get_hash())
            new = new[: target - len(members)]
            members.update(new)
            frontier = new
        dets = sorted(members, key=lambda k: k.get_hash())
        sample = dets[:: max(1, len(dets) // _RAW_SAMPLE)]
        raw = np.array([len(h_op.apply_block(ManyBodyState({k: 1.0}), 0.0)) for k in sample], dtype=float)
        block = len(h_op.apply_block(ManyBodyState(dict.fromkeys(dets, 1.0)), 0.0)) / len(dets)
    finally:
        h_op.set_restrictions(None)  # the solves set their own; the operator is shared
    return GsConnectivity(
        nnz_per_state=float(raw.mean()),
        selection_fanout=float(block),
        probe_dets=len(dets),
        seed_dets=len(seed),
        seconds=time.perf_counter() - start,
        window_states=window,
    )


#: Seeds larger than this are not diagonalized densely to count the lowest level.
_LEVEL_COUNT_MAX_SEED = 3000

#: States this close above the lowest energy are one level: a symmetry-degenerate multiplet is not
#: split by roundoff. Tight on purpose -- the seed's *excited* levels are not converged, so a thermal
#: window measured on it counts states the converged spectrum does not have (NiO: 21 against 3).
_LEVEL_DEGENERACY_TOL = 1e-6


def _lowest_level_degeneracy(h_op, basis):
    """States of ``h_op`` on the seed at its lowest energy: the multiplet the window solve must hold.

    Exact by symmetry, so a lower bound on the window's width (a thermal window may add levels above it).
    Counts 0 when the seed is too large to diagonalize densely, which prices the window phase off.
    """
    from impurityModel.ed.basis_transcription import build_dense_matrix

    if basis.size > _LEVEL_COUNT_MAX_SEED:
        return 0
    energies = np.linalg.eigvalsh(build_dense_matrix(basis, h_op))
    return int(np.sum(energies - energies[0] <= _LEVEL_DEGENERACY_TOL))


def solver_basis_connectivity(sb, comm=None):
    """A zero-argument callable measuring ``sb``'s connectivity, for :func:`memory_estimate.resolve_cap_policy`.

    Every driver builds the same :class:`~impurityModel.ed.solver_basis.SolverBasis` from the same
    inputs, so this is the one place the double-counting search, the self-energy and the susceptibility
    take their cap's inputs from.
    """
    return lambda: measure_gs_connectivity(sb.h, sb.impurity_orbitals, sb.bath_states, sb.nominal_occ, comm)
