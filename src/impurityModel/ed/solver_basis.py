import functools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional, cast

import numpy as np

from impurityModel.ed.basis_generation import SeedWindow, check_mixed_valence
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.observables import resolve_shell_ls, shell_index_of_orbitals
from impurityModel.ed.spin_pairs import spin_layout as resolve_spin_layout
from impurityModel.ed.symmetries import (
    classify_bath_occupation,
    discover_one_body_symmetries,
    extract_tensors,
    group_orbitals_by_blocks,
    impurity_block_structure,
    impurity_symmetry_rotation,
    rotate_hamiltonian,
)
from impurityModel.ed.utils import format_index_ranges

# Adaptive symmetry-adapted-basis rotation (calc_selfenergy): drop rotated operator terms below
# this magnitude (eV; removes rotation round-off fill), and rotate into the symmetry-adapted
# basis only if it keeps the operator term count within this factor of the input basis.
_ROTATION_TRIM_TOL = 1e-8
_MAX_ROTATION_FILL = 2.0


@dataclass(frozen=True)
class SolverBasis:
    """Solver-basis Hamiltonian and derived orbital/block layout for a self-energy run.

    Produced by :func:`_prepare_solver_basis`: the (optionally symmetry-adapted) solver
    Hamiltonian ``h`` and the matching non-interacting operator ``h0_solve``, the impurity/bath
    orbital grouping and block structure derived from it, the per-group occupation windows, and
    the rotations (``rotation_full`` full-space, ``u_imp`` impurity block) that carry results
    back to the caller's input basis.
    """

    h: object
    h0_solve: ManyBodyOperator
    n_spin_orbitals: int
    block_structure: object
    impurity_orbitals: dict
    bath_states: tuple
    nominal_occ: dict
    rotation_full: "np.ndarray"
    u_imp: "np.ndarray"
    rot_to_spherical: "np.ndarray"
    total_impurity_orbitals: dict
    sum_bath_states: dict
    #: The (dn, up) pairing of the solver-basis impurity and bath, resolved once per run; see
    #: :func:`~impurityModel.ed.spin_pairs.spin_layout`.
    spin_layout: object = None
    #: Multi-shell models only (``None`` for one shell, where everything below is the trivial single entry):
    #: the ``l`` of each spherical column block of ``rot_to_spherical``, the shell of every impurity orbital
    #: (solver-basis index -> shell), whether each shell has bath orbitals, and the impurity orbitals of the
    #: shells that do (the correlated, active ones).
    shell_ls: Optional[tuple] = None
    impurity_shell_index: Optional[dict] = None
    shell_has_bath: Optional[dict] = None
    active_impurity_indices: Optional[list] = None
    #: Multi-shell models with a bath-less shell only (``None`` otherwise, so a single shell is untouched): the
    #: impurity groups of the shells without bath, as a :class:`~basis_generation.SeedWindow`. Pass it as
    #: ``frozen_occupations`` to seed those shells as a window rather than let the seed span them freely.
    seed_window: Optional[SeedWindow] = None

    @functools.cached_property
    def h_int(self) -> ManyBodyOperator:
        """The interaction ``h - h0_solve``, built on first use.

        Defined against ``h0_solve``, the operator the Dyson estimator extracts the self-energy
        against, so an estimator resolving ``[c, h_int]`` measures the same self-energy. Lazy
        because only such an estimator needs it: on a production d shell it is a full copy of the
        Coulomb operator, which the Dyson path never pays for.
        """
        h = self.h if isinstance(self.h, ManyBodyOperator) else ManyBodyOperator(dict(cast(Mapping, self.h)))
        return h - self.h0_solve


def _per_group_occupation(nominal_occ, impurity_orbitals, h=None, impurity_shell_index=None):
    """Map ``nominal_occ`` onto the derived orbital-symmetry groups.

    Accepts a dict already keyed by the group indices (used as-is), or any other dict / a
    scalar interpreted as the *total* impurity occupation. When the one-body Hamiltonian ``h``
    is supplied, the total is distributed by **energetic filling** — the lowest on-site-energy
    impurity spin-orbitals (``h[o, o]``) are occupied first — so e.g. a cubic d-shell fills the
    lower ``t2g`` manifold before ``eg`` (giving ``t2g=6``, ``eg=2`` for ``d8``) and the split
    is spin-symmetric. Without ``h`` it falls back to a size-proportional split (remainder to
    the largest groups). The prescan refines the per-group split, so this only needs to be a
    sensible starting point.

    With ``impurity_shell_index`` (orbital -> shell, multi-shell models) a dict keyed by the shell indices is
    a per-shell nominal occupation: each shell's electrons are distributed over that shell's groups alone.
    """
    keys = list(impurity_orbitals)
    if impurity_shell_index is not None and isinstance(nominal_occ, dict):
        shell_of_group = {
            k: impurity_shell_index[next(o for block in impurity_orbitals[k] for o in block)] for k in keys
        }
        # In a multi-shell model a dict covering exactly the shells is shell-keyed, even when its keys also
        # happen to be the group keys (group k is not shell k: groups follow orbital index).
        if set(nominal_occ) == set(shell_of_group.values()):
            out = {}
            for a, n_a in nominal_occ.items():
                sub = {k: impurity_orbitals[k] for k in keys if shell_of_group[k] == a}
                out.update(_per_group_occupation(int(n_a), sub, h))
            return {k: out[k] for k in keys}
    if isinstance(nominal_occ, dict) and set(nominal_occ) == set(keys):
        return {k: int(nominal_occ[k]) for k in keys}
    total = int(sum(nominal_occ.values()) if isinstance(nominal_occ, dict) else nominal_occ)

    if h is not None:
        # Energetic filling: occupy the lowest on-site-energy impurity spin-orbitals first and
        # count how many land in each group. Ties broken by orbital index for determinism.
        orb_to_group = {orb: k for k in keys for block in impurity_orbitals[k] for orb in block}
        ordered = sorted(orb_to_group, key=lambda o: (np.real(h[o, o]), o))
        alloc = dict.fromkeys(keys, 0)
        for orb in ordered[: max(0, min(total, len(ordered)))]:
            alloc[orb_to_group[orb]] += 1
        return alloc

    sizes = {k: sum(len(block) for block in impurity_orbitals[k]) for k in keys}
    tot_size = sum(sizes.values()) or 1
    alloc = {k: int(total * sizes[k] // tot_size) for k in keys}
    remainder = total - sum(alloc.values())
    for k in sorted(keys, key=lambda k: sizes[k], reverse=True):
        if remainder <= 0:
            break
        alloc[k] += 1
        remainder -= 1
    return alloc


def prepare_solver_basis(
    h0,
    dc,
    u,
    impurity_orbitals,
    nominal_occ,
    *,
    mixed_valence=None,
    rot_to_spherical,
    verbosity,
    rank=0,
    comm=None,
    shell_ls=None,
):
    """Build the solver-basis Hamiltonian and derive its orbital/block layout.

    Assembles the interacting Hamiltonian ``H = h0 - DC + U(u4)`` in the caller's input basis, then
    adaptively rotates into the impurity-diagonalising basis when that does not densify the
    Coulomb tensor (fill ratio ``<= _MAX_ROTATION_FILL``; the input basis is kept otherwise).
    Derives the bath valence/conduction split, the Green's-function block structure, and the
    per-group impurity/bath orbital grouping and occupation windows -- all in whichever basis is
    solved in. Returns a :class:`SolverBasis`.

    ``rank`` gates the report only: every rank computes the identical layout, so without the gate the
    legend prints once per rank. The one collective is the broadcast of the resolved
    :attr:`SolverBasis.spin_layout`, made when ``comm`` is given (unconditionally, on every rank of
    it); without a ``comm`` the layout is resolved rank-locally.

    ``mixed_valence`` only has to be off (``None``/``False``/``0``): see
    :func:`~impurityModel.ed.basis_generation.check_mixed_valence`. It is checked first, before any collective.

    ``shell_ls`` (``l`` of each column block of an ndarray ``rot_to_spherical``) marks a multi-shell impurity.
    The symmetry-adapted rotation is then kept only if it maps every orbital into a single shell (otherwise
    the input basis is solved in), every orbital group must lie in one shell, and ``nominal_occ`` may be keyed
    by shell. ``None`` (or one shell) is the single-shell path, unchanged.
    """
    check_mixed_valence(mixed_valence)
    # construct local, interacting, hamiltonian (in the caller's input/correlated basis B)
    h_input = ManyBodyOperator(h0) - ManyBodyOperator(dc) + ManyBodyOperator(u)
    # The *non-interacting part of the Hamiltonian actually diagonalised* -- h0 with the double
    # counting already removed, not the raw KS block. This is what the self-energy is extracted
    # against (`get_sigma`'s `h0op`, via `SolverBasis.h0_solve`), and the two must agree or the
    # returned Sigma carries a spurious `-dc`: RSPt hands us the raw block plus `sig_dc` and
    # expects the *pure interaction* self-energy back, applying the double counting to the
    # lattice itself exactly once in `double_counting_apply` (green_double_counting.F90:596).
    # The external CTHYB path shows the same contract from the other side -- RSPt subtracts
    # `sig_dc` from `hlda` before that solver is called (green_solver.F90:578-582) and still
    # subtracts it from `sig` afterwards.
    h0_input = ManyBodyOperator(h0) - ManyBodyOperator(dc)

    # Flatten the impurity orbital dict (dict[int, list[int]]) into a plain spin-orbital index
    # list; the total orbital count is inferred from the Hamiltonian (impurity + bath). The bath
    # orbitals and their valence/conduction split are derived below, not passed in.
    impurity_indices = sorted(o for orbs in impurity_orbitals.values() for o in orbs)
    h_input_matrix = extract_tensors(h_input, two_body=False)[0]
    n_spin_orbitals = h_input_matrix.shape[0]

    # Adaptive symmetry-adapted basis: diagonalising the impurity one-body block collapses the
    # Green's-function block structure to its finest form (e.g. 1x1 eg/t2g blocks) BUT can
    # express the Coulomb interaction more densely. h0 and u4 are in the caller's "correlated"
    # input basis (NOT assumed spherical); the fill test below is measured *relative to that
    # input basis*, so we rotate only when it does not densify the operator (fill <= threshold)
    # and keep the input basis otherwise (e.g. a j,m_j eigenbasis under spin-orbit coupling
    # densifies the Coulomb tensor). Every output is rotated back to the input basis B before
    # returning; nothing here presumes a spherical-harmonic input.
    rotation_full, u_imp = impurity_symmetry_rotation(
        h_input, impurity_indices, n_orb=n_spin_orbitals, h0_matrix=h_input_matrix
    )
    h_rotated = rotate_hamiltonian(h_input, rotation_full, tol=_ROTATION_TRIM_TOL)
    n_terms_input = sum(1 for v in h_input.values() if abs(v) > _ROTATION_TRIM_TOL)
    fill_ratio = len(h_rotated) / max(n_terms_input, 1)

    rotate = fill_ratio <= _MAX_ROTATION_FILL
    multishell = shell_ls is not None and len(resolve_shell_ls(rot_to_spherical, shell_ls)) > 1
    if rotate and multishell:
        # A rotation that mixes shells would make "the occupation of shell a" meaningless: keep the input basis.
        try:
            shell_index_of_orbitals(u_imp.conj().T @ np.asarray(rot_to_spherical, dtype=complex), shell_ls)
        except ValueError:
            rotate = False
    if rotate:
        h = h_rotated
        h0_solve = rotate_hamiltonian(h0_input, rotation_full, tol=_ROTATION_TRIM_TOL)
        # Observable rotation for the solve (spherical -> S): compose the caller's input rotation
        # R_in (spherical -> B) with W^dag (B -> S). On the impurity block, R = u_imp^dag @ R_in.
        rot_to_spherical = u_imp.conj().T @ np.asarray(rot_to_spherical, dtype=complex)
    else:
        # Stay in the input basis; make the output rotation below a no-op.
        h = h_input
        h0_solve = h0_input
        rotation_full = np.eye(n_spin_orbitals, dtype=complex)
        u_imp = np.eye(len(impurity_indices), dtype=complex)

    # One-body matrix of the solver-basis Hamiltonian, extracted once and shared by the
    # classification/grouping helpers below (each would otherwise re-walk the full operator
    # and allocate its own dense n_orb x n_orb copy).
    h_matrix = extract_tensors(h, n_orb=n_spin_orbitals, two_body=False)[0] if rotate else h_input_matrix

    # Derive the bath orbitals (complement of the impurity set) and their initial occupation:
    # baths below the Fermi level (h[o, o] < 0) are valence (initially occupied), the rest are
    # conduction (initially empty). The bath one-body diagonal is unchanged by the impurity-only
    # rotation, so this is consistent whether measured in the input or solver basis.
    valence_flat, conduction_flat = classify_bath_occupation(
        h, impurity_indices, n_orb=n_spin_orbitals, h0_matrix=h_matrix
    )

    # GF block structure from the hybridization-dressed impurity matrix (h[imp,imp] + V^dag V),
    # in whichever basis we solve in (fixes bath-mediated coupling; 1x1 blocks when rotated).
    # Derived from h *after* any rotation, so the blocks label the sectors of the solver basis.
    block_structure = impurity_block_structure(h, impurity_indices, h0_matrix=h_matrix)

    # Group the flat orbital lists into orbital-symmetry manifolds (the inequivalent blocks and
    # their spin-degenerate partners, e.g. eg / t2g) **in the solver basis** h. Grouping by the
    # block structure keeps both spins of a manifold in one group, so the many-body basis spans
    # all S_z sectors (spin multiplets stay degenerate); the impurity occupation window is tied
    # across groups by the restriction machinery, not pinned per group.
    merged_groups: list = []
    impurity_orbitals, bath_states = group_orbitals_by_blocks(
        h,
        impurity_indices,
        valence_flat,
        conduction_flat,
        block_structure,
        n_orb=n_spin_orbitals,
        h0_matrix=h_matrix,
        merged=merged_groups,
    )
    if merged_groups and rank == 0:
        # Printed at every verbosity: it changes what the per-group occupation windows mean.
        for parts in merged_groups:
            print(
                "Impurity orbital groups "
                + " and ".join(format_index_ranges(orbs) for orbs in parts)
                + " merged: one bath component hops to both, so no window can be set per group.",
                flush=True,
            )
    impurity_shell_index = shell_has_bath = active_impurity_indices = None
    if multishell:
        local = shell_index_of_orbitals(rot_to_spherical, shell_ls)
        impurity_shell_index = {int(o): int(a) for o, a in zip(impurity_indices, local)}
        for k, blocks in impurity_orbitals.items():
            shells = {impurity_shell_index[o] for block in blocks for o in block}
            if len(shells) > 1:
                raise ValueError(
                    f"impurity orbital group {k} spans shells {sorted(shells)}: a bath component hops to more than "
                    "one shell, so no per-shell occupation window can be set."
                )
    nominal_occ = _per_group_occupation(nominal_occ, impurity_orbitals, h_matrix, impurity_shell_index)

    valence_baths, conduction_baths = bath_states
    total_impurity_orbitals = {i: sum(len(orbs) for orbs in impurity_orbitals[i]) for i in impurity_orbitals}
    sum_bath_states = {
        i: sum(len(orbs) for orbs in valence_baths[i]) + sum(len(orbs) for orbs in conduction_baths[i])
        for i in valence_baths
    }

    if verbosity > 0 and rank == 0:
        # The group keys below are the labels every later occupation report speaks in (the
        # occupation search, the basis-generation windows, the frozen-shell notes). Printed
        # once, here, where the grouping is derived -- without this legend an integer key says
        # nothing about which orbitals it stands for.
        basis_note = "symmetry-adapted basis" if rotate else "input basis"
        print(
            f"Block structure: {len(block_structure.blocks)} blocks, "
            f"solving in the {basis_note} (operator fill {fill_ratio:.1f}x)"
        )
        print("Impurity orbital groups:")
        print(f"  {'group':>5}  {'orbitals':<20} {'imp':>4} {'val':>4} {'con':>4}" f" {'nominal':>7}")
        for i in sorted(impurity_orbitals):
            orbs = sorted(orb for block in impurity_orbitals[i] for orb in block)
            n_val = sum(len(block) for block in valence_baths[i])
            n_con = sum(len(block) for block in conduction_baths[i])
            print(
                f"  {i:>5}  {format_index_ranges(orbs):<20} {len(orbs):>4} {n_val:>4} {n_con:>4}"
                f" {nominal_occ[i]:>7}"
            )
    if multishell:
        shell_has_bath = dict.fromkeys(range(len(shell_ls)), False)
        for k in impurity_orbitals:
            if sum_bath_states.get(k, 0) > 0:
                shell_has_bath[impurity_shell_index[next(o for block in impurity_orbitals[k] for o in block)]] = True
        active_impurity_indices = sorted(o for o, a in impurity_shell_index.items() if shell_has_bath[a])
    seed_window = None
    if multishell:
        # Per *shell*, not per group: a hybridising shell has bath-less groups too (the d orbitals no bath hops to).
        deep_groups = [
            k
            for k in impurity_orbitals
            if not shell_has_bath[impurity_shell_index[next(o for block in impurity_orbitals[k] for o in block)]]
        ]
        if deep_groups and len(deep_groups) < len(impurity_orbitals):
            seed_window = SeedWindow(deep_groups)
    spin_layout = resolve_spin_layout(
        h, impurity_orbitals, bath_states, rot_to_spherical, n_spin_orbitals, comm=comm, shell_ls=shell_ls
    )
    return SolverBasis(
        h=h,
        h0_solve=h0_solve,
        n_spin_orbitals=n_spin_orbitals,
        block_structure=block_structure,
        impurity_orbitals=impurity_orbitals,
        bath_states=bath_states,
        nominal_occ=nominal_occ,
        rotation_full=rotation_full,
        u_imp=u_imp,
        rot_to_spherical=rot_to_spherical,
        total_impurity_orbitals=total_impurity_orbitals,
        sum_bath_states=sum_bath_states,
        spin_layout=spin_layout,
        shell_ls=tuple(resolve_shell_ls(rot_to_spherical, shell_ls)) if multishell else None,
        impurity_shell_index=impurity_shell_index,
        shell_has_bath=shell_has_bath,
        active_impurity_indices=active_impurity_indices,
        seed_window=seed_window,
    )


def _one_body_commutant_generators(h_op, impurity_orbitals, bath_states):
    """The continuous commutant of the *one-body* block, propagated along the bath chain.

    Correct exactly when ``H`` has no two-body part: then the commutant of ``h`` really is a
    symmetry algebra. With a Coulomb interaction it is far too large -- a degenerate manifold
    yields the whole of ``u(n)``, arbitrary orbital and spin rotations inside it, and a
    Slater-Condon ``U`` breaks essentially all of them (measured on a cubic d-shell: 36 of 36
    fail ``[H, g] = 0`` with residuals 0.86 to 2.57). Kept because it is the only source of
    *orbital* generators, and gated by the consumer against the full ``H``, which is what makes
    the over-production harmless.

    .. note:: Two known limitations, both of which make this return fewer generators than it
       should rather than wrong ones (the consumer's gate covers the rest):

       * the bath site map is built per bath index ``k`` while looping over the valence and
         conduction dicts in turn, so on a model with **both** the conduction entries overwrite
         the valence ones and every generator then fails the commutator test -- i.e. ``[]``;
       * ``imp_map`` keys on ``(group, index-within-block)``, so a group holding several blocks
         (``eg`` and ``t2g``: precisely a cubic d-shell) collides -- 10 orbitals collapse onto 6
         keys -- and the propagation pairs impurity orbitals with the wrong bath partners.
    """
    imp_orbs = []
    if impurity_orbitals:
        for orbs in impurity_orbitals.values():
            for o in orbs:
                imp_orbs.extend(o)

    h, _, _ = extract_tensors(h_op, two_body=False)

    if imp_orbs:
        imp = sorted(set(imp_orbs))
        h_imp = h[np.ix_(imp, imp)]
        imp_generators = discover_one_body_symmetries(h_imp)

        imp_map = {}
        for group, imp_blocks in impurity_orbitals.items():
            for imp_blk in imp_blocks:
                for idx_in_grp, o in enumerate(imp_blk):
                    imp_map[o] = (group, idx_in_grp)

        generators = []
        for g_imp in imp_generators:
            g_full = np.zeros_like(h, dtype=complex)
            for i, oi in enumerate(imp):
                for j, oj in enumerate(imp):
                    g_full[oi, oj] = g_imp[i, j]

            n_bath = 0
            for bath_dict in bath_states:
                if bath_dict:
                    n_bath = max(n_bath, max((len(blks) for blks in bath_dict.values()), default=0))

            site_maps = [{i: imp[i] for i in range(len(imp))}]

            for k in range(n_bath):
                site_k_map = {}
                for bath_dict in bath_states:
                    if not bath_dict:
                        continue
                    for i, o_imp in enumerate(imp):
                        if o_imp not in imp_map:
                            continue
                        group, idx_in_grp = imp_map[o_imp]
                        if group in bath_dict and k < len(bath_dict[group]) and idx_in_grp < len(bath_dict[group][k]):
                            site_k_map[i] = bath_dict[group][k][idx_in_grp]
                site_maps.append(site_k_map)

                for i in range(len(imp)):
                    if i not in site_k_map:
                        continue
                    oi = site_k_map[i]
                    for j in range(len(imp)):
                        if j not in site_k_map:
                            continue
                        oj = site_k_map[j]

                        propagated = False
                        for prev_map in reversed(site_maps[:-1]):
                            if i not in prev_map or j not in prev_map:
                                continue
                            prev_oi = prev_map[i]
                            prev_oj = prev_map[j]

                            t_j = h[oj, prev_oj]
                            t_i = h[oi, prev_oi]

                            # Chain hoppings decay geometrically along the fit, so an absolute gap
                            # test (e.g. < 1e-6) treats t_i=1e-9, t_j=1e-10 -- a 10x mismatch -- as
                            # equal; scale by the larger hopping.
                            if (
                                abs(t_j) > 1e-12
                                and abs(t_i) > 1e-12
                                and abs(abs(t_i) - abs(t_j)) < 1e-6 * max(abs(t_i), abs(t_j))
                            ):
                                g_full[oi, oj] = g_full[prev_oi, prev_oj] * (t_j / t_i)
                                propagated = True
                                break
                        if not propagated:
                            g_full[oi, oj] = 0.0

            # Absolute commutator tolerance is scale-dependent: a core level at -6000 eV vs
            # bath hoppings fit to ~1e-6 eV precision means the same 1e-9 cut either accepts
            # noise (large ||h||) or rejects true symmetries (small ||[h,g]|| relative to the
            # fit residual). Normalize by ||h||*||g|| so the cut is scale-free.
            norm_h = np.linalg.norm(h)
            norm_g = np.linalg.norm(g_full)
            if norm_h > 0 and norm_g > 0 and np.linalg.norm(h @ g_full - g_full @ h) / (norm_h * norm_g) < 1e-9:
                generators.append(g_full)
    else:
        generators = discover_one_body_symmetries(h)
    return generators


def get_symmetry_generators(
    h_op, impurity_orbitals, bath_states, n_orb=None, tol=1e-6, rot_to_spherical=None, spin_layout=None
):
    r"""Operators the CIPSI closure may use to complete a determinant's symmetry orbit.

    Returns the **total** spin ladder operators :math:`\hat S_+ = \sum_a c^\dagger_{a\uparrow}
    c_{a\downarrow}` and :math:`\hat S_-`, summed over impurity *and* bath, or ``[]`` when the
    spin labelling cannot be trusted.

    **Why these and not the commutant of the one-body block.** The previous implementation
    handed back ``discover_one_body_symmetries(h_imp)`` -- the continuous (Lie) commutant of the
    impurity one-body block, propagated to the bath. For a degenerate manifold that is the whole
    of ``u(n)``: arbitrary orbital *and* spin rotations inside it. A Slater-Condon ``U`` breaks
    essentially all of them. Measured on a cubic d-shell (``eg``/``t2g`` crystal field,
    ``F0,F2,F4 = 9,8,6``, bath propagation included): all 36 generators fail ``[H, g] = 0`` with
    residuals of 0.86 to 2.57 -- O(1), not roundoff. They are simply not symmetries.

    Nor could they be. The orbital symmetry of a cubic system is the octahedral point group
    ``O_h``, which is **discrete** and has no continuous generators, so it cannot be expressed as
    a Lie commutant at all. With spin-orbit coupling off, the continuous symmetries of a cubic
    shell with a Slater interaction are total charge and total spin -- and on that same model
    ``N_total`` and ``S_z_total`` both commute with the full ``H`` to exactly ``0``.

    Of those, only the ladder operators are *useful here*: the closure completes orbits by
    applying an operator to a determinant, and ``N`` and ``S_z`` are diagonal -- they return the
    determinant they were given and generate nothing. ``S_\pm`` flip one spin, so their orbit is
    the spin multiplet, which is what makes the retained space spin-adapted and the multiplet
    degeneracy exact rather than approximate.

    The pairing is the run's :class:`~impurityModel.ed.spin_pairs.SpinLayout` (``spin_layout``;
    resolved here from ``rot_to_spherical`` when ``None``), and the ladder operators are offered only
    at grade ``"su2"``, where ``[H, S_+] = 0`` was verified on the full operator. The consumer
    (:meth:`cipsi_solver.CIPSISolver.expand`) re-checks it all the same.

    The one-body commutant is still offered alongside them, via
    :func:`_one_body_commutant_generators`: it is the only source of *orbital* generators, and
    it is exactly right when ``H`` has no two-body part. Over-production there is harmless
    precisely because the consumer gates on the full ``H``.

    Returns
    -------
    list
        ``[S_+, S_-]`` (``ManyBodyOperator``) followed by any one-body commutant generators
        (plain matrices). Empty when neither source yields anything trustworthy.
    """
    from impurityModel.ed.observables import make_spin_operators

    generators = []
    layout = spin_layout
    if layout is None:
        if n_orb is None:
            n_orb = extract_tensors(h_op, two_body=False)[0].shape[0]
        layout = resolve_spin_layout(h_op, impurity_orbitals, bath_states, rot_to_spherical, n_orb, tol)
    if layout.su2:
        s_plus, s_minus, _s_z = make_spin_operators(layout.pairs)
        generators += [s_plus, s_minus]
    # Otherwise: spin-orbit coupling or a spin-polarized bath. S_+- is then not a symmetry --
    # better no closure than a wrong one.
    return generators + _one_body_commutant_generators(h_op, impurity_orbitals, bath_states)
