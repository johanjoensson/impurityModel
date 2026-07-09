"""Shared NiO d-shell workload builder for the self-energy benchmarks / golden tests.

Factored out of ``test_selfenergy_perf.py`` so the performance harness, the Phase 0
golden-baseline regression, and the Phase 5 symmetric-vs-spherical benchmark all build
the *same* ``calc_selfenergy`` inputs from the ``h0/h0_NiO_<n>bath.pickle`` files.

Not a test module (leading underscore): it exposes helpers only.
"""

import os
from collections import OrderedDict

import numpy as np

# Repo root: this file is src/impurityModel/test/<this>; the h0 pickles live in <root>/h0.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def build_selfenergy_inputs(
    ls=2,
    nBaths=10,
    nValBaths=None,
    n0imp=8,
    n_omega=2000,
    dense_cutoff=100000,
    # Explicit (not None): the golden baselines were generated at this cap; the
    # RAM-derived default would make them machine-dependent.
    truncation_threshold=1000,
    *,
    rank=0,
    verbose=False,
    xi=0.0,
    hField=(0.0, 0.0, 0.0001),
    delta=0.2,
    tau=0.002,
    reort=None,
    occ_cutoff=1e-12,
    slaterWeightMin=1e-12,
    dN=None,
    rot_to_spherical=None,
    chargeTransferCorrection=None,
    n0imp_p=6,
    Fpd=(8.9, 0.0, 6.8),
    Gpd=(0.0, 5.0, 0.0, 2.8),
):
    """Construct the ``calc_selfenergy`` keyword arguments for the NiO d-shell workload.

    Mirrors the input construction inside ``selfenergy.get_selfenergy`` but calls
    ``get_noninteracting_hamiltonian_operator`` with the correct keyword arguments
    (``get_selfenergy`` currently mis-calls it with a stale positional order).

    Parameters
    ----------
    ls : int
        Impurity angular momentum (2 for a d-shell).
    nBaths, nValBaths : int
        Total / valence bath spin-orbitals. ``nValBaths`` defaults to ``nBaths``.
    n0imp : int
        Nominal impurity occupation (NiO d8 -> 8).
    n_omega : int
        Real-axis mesh size.
    dense_cutoff, truncation_threshold : int
        Solver knobs forwarded to ``calc_selfenergy``.
    rot_to_spherical : np.ndarray, optional
        Override the default identity rotation (e.g. a symmetry-adapting unitary).

    Returns
    -------
    dict
        Keyword arguments for ``selfenergy.calc_selfenergy`` (minus ``comm``).
    """
    from impurityModel.ed import atomic_physics, operator_algebra
    from impurityModel.ed.hamiltonian_io import get_noninteracting_hamiltonian_operator

    if nValBaths is None:
        nValBaths = nBaths

    Fdd = [7.5, 0, 9.9, 0, 6.6]

    sum_baths = OrderedDict({ls: nBaths})
    nValBaths_d = OrderedDict({ls: nValBaths})
    n_imp = 2 * (2 * ls + 1)

    # Coulomb U as a rank-4 tensor in the impurity spin-orbital index space.
    u4 = np.zeros((n_imp, n_imp, n_imp, n_imp), dtype=complex)
    uOp = atomic_physics.getUop(l1=ls, l2=ls, l3=ls, l4=ls, R=Fdd)
    nBaths_for_c2i = OrderedDict({ls: 0})
    for process, val in uOp.items():
        i = operator_algebra.c2i(nBaths_for_c2i, process[0][0])
        j = operator_algebra.c2i(nBaths_for_c2i, process[1][0])
        k = operator_algebra.c2i(nBaths_for_c2i, process[2][0])
        m = operator_algebra.c2i(nBaths_for_c2i, process[3][0])
        # RSPt convention: u4[i,j,k,l] multiplies c^dag_i c^dag_j c_l c_k, so
        # the process c^dag_i c^dag_j c_k c_m is stored with k and m swapped.
        u4[i, j, m, k] = 2.0 * val

    # Flat impurity spin-orbital index list (dict[int, list[int]]); calc_selfenergy re-groups the
    # orbitals and derives the bath orbitals + valence/conduction split from the Hamiltonian.
    impurity_orbitals = {ls: list(range(n_imp))}
    mixed_valence = {ls: 0}
    nominal_occ = {ls: n0imp}

    if rot_to_spherical is None:
        rot_to_spherical = np.eye(n_imp, dtype=complex)

    h0_filename = os.path.join(REPO_ROOT, "h0", f"h0_NiO_{nBaths}bath.pickle")
    hOp = get_noninteracting_hamiltonian_operator(
        nBaths=sum_baths,
        nValBaths=nValBaths_d,
        SOCs=[0, xi],
        hField=hField,
        h0_filename=h0_filename,
        rank=rank,
        verbose=verbose,
    )

    # Multiplet ligand-field-theory double counting (opt-in via ``chargeTransferCorrection``;
    # ``None`` -> no DC, the historical SOC-free d6 workload used by the perf / driver-glue
    # anchors). Without a DC the full Coulomb (u4) is double-counted against h0's mean-field d
    # level (d8 sits ~180 eV above d2), so the impurity empties and only the occupation window
    # keeps it near nominal. We remove the d-d double counting with the **d-only** MLFT DC,
    # dc[2] = Udd*n3d - c, applied as a one-body level shift -dc[2] on every d spin-orbital ->
    # nominal d occupation becomes the genuine energetic minimum (d ~ 8.16, physical NiO Ni(2+)
    # with ligand->d covalency; interior to the window, so the DC, not the window, sets it).
    #
    # NOTE: this is a d-ONLY valence model (h0_NiO pickle has only l=2; no explicit 2p core). The
    # full 2p3d form dc[2] = Udd*n3d + Upd*n2p - c (the get_spectra path, using n0imp_p / Fpd /
    # Gpd) includes the p-d mean field precisely because that model carries the explicit 2p-3d
    # Coulomb which cancels it; adding Upd*n2p here (no such interaction to cancel) over-subtracts
    # ~50 eV and fills the impurity to d10. The 2p3d parameters (n0imp_p, Fpd, Gpd) and the 2p
    # spin-orbit xi_2p are therefore retained only to document the physical NiO model / a future
    # 2p3d benchmark; the d-only self-energy uses just Fdd, c and the valence xi_3d SOC (``xi``).
    if chargeTransferCorrection is not None:
        dc = atomic_physics.dc_MLFT(n3d_i=n0imp, c=chargeTransferCorrection, Fdd=Fdd)
        eDCOperator = {(((ls, s, m), "c"), ((ls, s, m), "a")): -dc[ls] for s in range(2) for m in range(-ls, ls + 1)}
        hOp = operator_algebra.addOps([hOp, eDCOperator])

    # Map (l,s,m) / (l,b) labels to single integer indices. Drop identically-zero terms
    # first: get_noninteracting_hamiltonian_operator unconditionally adds a 2p (l=1) SOC
    # operator whose terms are all 0.0 when xi_2p=0; those carry unmappable l=1 labels and
    # contribute nothing to H.
    hOp_int = {}
    for process, value in hOp.items():
        if abs(value) == 0:
            continue
        hOp_int[tuple((operator_algebra.c2i(sum_baths, spinOrb), action) for spinOrb, action in process)] = value

    omega_mesh = np.linspace(-1.83, 1.83, n_omega)

    return dict(
        h0=hOp_int,
        u4=u4,
        iw=None,
        w=omega_mesh,
        delta=delta,
        nominal_occ=nominal_occ,
        mixed_valence=mixed_valence,
        impurity_orbitals=impurity_orbitals,
        tau=tau,
        verbosity=2 if verbose else 0,
        rot_to_spherical=rot_to_spherical,
        cluster_label="bench",
        reort=reort,
        dense_cutoff=dense_cutoff,
        spin_flip_dj=False,
        chain_restrict=False,
        occ_cutoff=occ_cutoff,
        truncation_threshold=truncation_threshold,
        slaterWeightMin=slaterWeightMin,
        dN=dN,
        sparse_green=True,
    )


def build_ground_state_workload(
    nBaths=10,
    mixed_valence=1,
    truncation_threshold=30000,
    dense_cutoff=50,
    slater_weight_min=1e-12,
    de2_min=1e-6,
    comm=None,
    verbose=False,
):
    """Build the NiO d-shell Hamiltonian and its converged CIPSI ground-state basis.

    Runs the same preamble as ``selfenergy.calc_selfenergy``: symmetry rotation of the
    Hamiltonian, bath valence/conduction classification, orbital grouping by symmetry
    block, then ``find_ground_state_basis`` plus one ``CIPSISolver.expand``.

    Parameters
    ----------
    nBaths : int
        Which ``h0/h0_NiO_<n>bath.pickle`` to load.
    mixed_valence : int
        Impurity occupation slack per orbital group. The 10-bath anchor's zero window
        pins every group's occupation and collapses the ground-state sector to a
        trivial basis, so benchmarks want at least 1.
    truncation_threshold : int
        CIPSI basis-size cap.
    dense_cutoff : int
        Basis size above which the restarted (TRLM) eigensolver is used.
    de2_min : float
        CIPSI Epstein-Nesbet selection threshold. This, more than ``nBaths``, sets the
        ground-state basis size: at the default the NiO anchor selects only a few hundred
        determinants regardless of the bath count.

    Returns
    -------
    dict
        ``{"h", "basis", "solver", "tau", "comm"}``. ``h``'s restrictions are already
        pointed at ``basis``.
    """
    from impurityModel.ed import atomic_physics
    from impurityModel.ed.BlockLanczosArray import Reort
    from impurityModel.ed.cipsi_solver import CIPSISolver
    from impurityModel.ed.groundstate import find_ground_state_basis
    from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
    from impurityModel.ed.selfenergy import (
        _MAX_ROTATION_FILL,
        _ROTATION_TRIM_TOL,
        _per_group_occupation,
        _per_group_scalar,
    )
    from impurityModel.ed.symmetries import (
        classify_bath_occupation,
        extract_tensors,
        group_orbitals_by_blocks,
        impurity_block_structure,
        impurity_symmetry_rotation,
        rotate_hamiltonian,
    )

    inputs = build_selfenergy_inputs(nBaths=nBaths, truncation_threshold=truncation_threshold, verbose=verbose)

    u = atomic_physics.getUop_from_rspt_u4(inputs["u4"])
    h_input = ManyBodyOperator(inputs["h0"]) + ManyBodyOperator(u)
    impurity_indices = sorted(o for orbs in inputs["impurity_orbitals"].values() for o in orbs)
    h_input_matrix = extract_tensors(h_input, two_body=False)[0]
    n_orb = h_input_matrix.shape[0]

    rotation_full, _u_imp = impurity_symmetry_rotation(h_input, impurity_indices, n_orb=n_orb, h0_matrix=h_input_matrix)
    h_rotated = rotate_hamiltonian(h_input, rotation_full, tol=_ROTATION_TRIM_TOL)
    n_terms_input = sum(1 for v in h_input.values() if abs(v) > _ROTATION_TRIM_TOL)
    if len(h_rotated) / max(n_terms_input, 1) <= _MAX_ROTATION_FILL:
        h = h_rotated
        h_matrix = extract_tensors(h, n_orb=n_orb, two_body=False)[0]
    else:
        h = h_input
        h_matrix = h_input_matrix

    valence_flat, conduction_flat = classify_bath_occupation(h, impurity_indices, n_orb=n_orb, h0_matrix=h_matrix)
    block_structure = impurity_block_structure(h, impurity_indices, h0_matrix=h_matrix)
    impurity_orbitals, bath_states = group_orbitals_by_blocks(
        h, impurity_indices, valence_flat, conduction_flat, block_structure, n_orb=n_orb, h0_matrix=h_matrix
    )
    nominal_occ = _per_group_occupation(inputs["nominal_occ"], impurity_orbitals, h_matrix)
    # _per_group_scalar maps a dict keyed by the derived group indices through unchanged;
    # anything else collapses to the default -- so key the window by group explicitly.
    mv = _per_group_scalar(dict.fromkeys(impurity_orbitals, mixed_valence), impurity_orbitals, default=0)

    tau = inputs["tau"]
    basis = find_ground_state_basis(
        h,
        impurity_orbitals,
        bath_states,
        nominal_occ,
        mixed_valence=mv,
        tau=tau / 100,  # calc_gs runs the occupation search at tau/100
        chain_restrict=False,
        dense_cutoff=dense_cutoff,
        spin_flip_dj=False,
        comm=comm,
        truncation_threshold=truncation_threshold,
        verbose=verbose,
        slaterWeightMin=np.sqrt(slater_weight_min),
        cipsi_solver_method="trlm",
    )
    basis.tau = tau
    solver = CIPSISolver(basis)
    solver.expand(
        h,
        dense_cutoff=dense_cutoff,
        de2_min=de2_min,
        slaterWeightMin=slater_weight_min,
        solver="trlm",
        reort=Reort.PARTIAL,
    )
    if basis.restrictions is not None:
        h.set_restrictions(basis.restrictions)
    return {"h": h, "basis": basis, "solver": solver, "tau": tau, "comm": comm}
