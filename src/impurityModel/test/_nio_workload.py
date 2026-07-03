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
        u4[i, j, k, m] = 2.0 * val

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
