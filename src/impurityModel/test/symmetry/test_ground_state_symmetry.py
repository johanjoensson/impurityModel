"""G5: the generators handed to the closure are symmetries, and the ground state has them.

Two things were asserted nowhere before. First, that the operators the CIPSI closure completes
orbits with actually commute with the Hamiltonian -- the discovery routine works from the
*one-body* block, whose commutant is far larger than the interacting Hamiltonian's, so it
over-produces by construction. Second, that a ground state which comes back from ``calc_gs``
carries the quantum numbers it should: the existing symmetry tests either check ``[h, g] = 0``
against the one-body ``h`` only (``test_symmetries.py``) or build a Casimir by hand and never
touch a produced state (``test_nonabelian_casimir.py``).

The model here is a cubic d-shell -- an ``eg``/``t2g`` crystal field on 5 spatial orbitals in the
down-then-up spin layout, with a genuine Slater-Condon interaction and a hybridising bath. Its
continuous symmetries are total charge and total spin: the octahedral point group is discrete,
so the orbital part has no Lie generators at all, and everything the one-body commutant offers
beyond spin is broken by ``U``.
"""

import numpy as np

from impurityModel.ed.cipsi_solver import CIPSISolver, _commutes_with
from impurityModel.ed.lie_algebra import tensors_to_operator
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState
from impurityModel.ed.observables import (
    casimir_operator,
    casimir_to_quantum_number,
    make_spin_operators,
    manifold_observable_values,
)
from impurityModel.ed.solver_basis import get_symmetry_generators
from impurityModel.ed.spin_pairs import bath_spin_pairs, impurity_spin_pairs
from impurityModel.test.support.cubic_d_shell import cubic_d_shell


def _as_operator(generator):
    return generator if isinstance(generator, ManyBodyOperator) else tensors_to_operator(generator, tol=1e-12)


def test_every_generator_offered_is_a_symmetry_of_the_full_hamiltonian():
    """The property the closure's correctness rests on.

    A generator that does not commute with ``H`` still enlarges the candidate set, and under a
    binding ``truncation_threshold`` its images displace determinants that carry real weight --
    which is how a "symmetry-adapted" run came back with a *higher* variational energy.
    """
    h_op, h_full, impurity_orbitals, bath_states = cubic_d_shell()
    generators = get_symmetry_generators(h_op, impurity_orbitals, bath_states)

    assert generators, "no symmetry generators offered for a cubic d-shell with SU(2) spin symmetry"
    for i, generator in enumerate(generators):
        assert _commutes_with(
            h_full, _as_operator(generator)
        ), f"generator {i} does not commute with the full Hamiltonian (two-body terms included)"


def test_spin_orbit_coupling_withdraws_the_spin_generators():
    """``S_+-`` is genuinely not a symmetry once spin is mixed, and must not be offered."""
    h_op, h_full, impurity_orbitals, bath_states = cubic_d_shell(soc=0.15)
    generators = get_symmetry_generators(h_op, impurity_orbitals, bath_states)

    for i, generator in enumerate(generators):
        assert _commutes_with(h_full, _as_operator(generator)), f"generator {i} survives a spin-mixing term"


def test_the_ground_state_is_a_spin_eigenstate():
    """The produced state, not the operators: ``S_z`` and ``S^2`` on a ``calc_gs``-style solve.

    ``H`` commutes with total spin, so its ground state carries good ``S_z`` and ``S^2``. On this
    two-electron sector the ground state is the ``S = 1`` triplet: ``S_z = -1, 0, +1``.
    """
    # One bath set, not two: the full 30-orbital layout takes ~60 s here, and the property under
    # test (good Sz / S^2 on the produced state) does not depend on having a conduction bath.
    _h_op, h_full, impurity_orbitals, bath_states = cubic_d_shell(n_bath_sets=1)
    pairs = impurity_spin_pairs(impurity_orbitals) + bath_spin_pairs(bath_states)
    s_plus, s_minus, s_z = make_spin_operators(pairs)

    basis = Basis(
        impurity_orbitals,
        bath_states,
        nominal_impurity_occ={0: 2},
        mixed_valence={0: 0},
        tau=1e-3,
        # Capped, which is both faster and closer to production: spin purity has to survive a
        # budget, not just the exact limit. Uncapped this model reaches 3589 determinants and
        # |S^2 - 2| ~ 5e-15; at this cap it is ~1e-8, still a clean quantum number.
        truncation_threshold=800,
        verbose=False,
        comm=None,
    )
    solver = CIPSISolver(basis)
    solver.expand(h_full, dense_cutoff=2000, de2_min=1e-10, slaterWeightMin=1e-12)
    # Six, not four: the ground state is a 3-fold triplet with a second triplet ~1e-5 above it.
    # `manifold_observable_values` diagonalises the observable inside each degenerate group, so a
    # request that cuts a group in half yields the eigenvalues of a *truncated* block -- which is
    # not a quantum number and reads as a spurious failure (measured <Sz> = -0.88).
    es, psis = solver.get_eigenvectors(h_full, num_wanted=6, dense_cutoff=2000, slaterWeightMin=1e-12)

    # A degenerate block is only defined up to a rotation inside the manifold, so the physical
    # values come from diagonalising the observable on it -- the same route calc_gs's own
    # Casimir reporting takes (groundstate.py, manifold_observable_values).
    psis_blk = ManyBodyState.from_states(psis)
    s2_op = casimir_operator(s_plus, s_minus, s_z)
    sz_values = np.real(manifold_observable_values(psis_blk, es, lambda blk: s_z.apply_block(blk, 0)))
    s2_values = np.real(manifold_observable_values(psis_blk, es, lambda blk: s2_op.apply_block(blk, 0)))

    ground = np.abs(np.asarray(es) - es[0]) < 1e-6
    assert ground.sum() == 3, f"expected a 3-fold ground manifold, got {ground.sum()} (energies {es})"

    # Good quantum numbers: Sz must be a half-integer and S(S+1) must come from a half-integer S.
    for value in sz_values[ground]:
        assert abs(2 * value - round(2 * value)) < 1e-6, f"<Sz> = {value} is not a half-integer"
    np.testing.assert_allclose(sorted(np.round(sz_values[ground], 9)), [-1.0, 0.0, 1.0], atol=1e-6)
    for value in s2_values[ground]:
        spin = casimir_to_quantum_number(value)
        assert abs(2 * spin - round(2 * spin)) < 1e-6, f"<S^2> = {value} does not correspond to a half-integer S"
        assert abs(spin - 1.0) < 1e-6, f"expected an S = 1 ground triplet, got S = {spin}"


def test_the_closure_trades_energy_for_spin_purity_under_a_cap():
    """Why ``SYMMETRY_CLOSURE_DEFAULT`` is off, pinned to numbers rather than an opinion.

    With correct generators the closure does what it promises -- uncapped it drives the ground
    triplet's ``|S^2 - 2|`` from ~1e-11 to ~1e-15 -- but at roughly double the determinants. Under
    a binding budget it is a net loss on the variational energy, because symmetry partners are not
    generally the highest-de2 candidates and displace ones that are. Both goals are legitimate;
    they conflict, so the caller chooses.
    """
    h_op, h_full, impurity_orbitals, bath_states = cubic_d_shell(n_bath_sets=1)

    def run(generators, threshold):
        basis = Basis(
            impurity_orbitals,
            bath_states,
            nominal_impurity_occ={0: 2},
            mixed_valence={0: 0},
            tau=1e-3,
            truncation_threshold=threshold,
            verbose=False,
            comm=None,
        )
        solver = CIPSISolver(basis)
        solver.expand(h_full, dense_cutoff=2000, de2_min=1e-10, slaterWeightMin=1e-12, symmetry_generators=generators)
        es, _psis = solver.get_eigenvectors(h_full, num_wanted=6, dense_cutoff=2000, slaterWeightMin=1e-12)
        return float(es[0]), basis.size

    auto = get_symmetry_generators(h_op, impurity_orbitals, bath_states)
    e_on, size_on = run(auto, 400)
    e_off, size_off = run([], 400)

    assert size_on == size_off == 400, (size_on, size_off)
    assert e_off <= e_on + 1e-12, (
        f"the closure improved the capped energy ({e_on:.10f} vs {e_off:.10f}); if that is now "
        "reproducible, revisit SYMMETRY_CLOSURE_DEFAULT -- the measurement it encodes has changed"
    )
