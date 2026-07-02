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
    from impurityModel.ed import finite
    from impurityModel.ed.get_spectra import get_noninteracting_hamiltonian_operator

    if nValBaths is None:
        nValBaths = nBaths

    Fdd = [7.5, 0, 9.9, 0, 6.6]

    sum_baths = OrderedDict({ls: nBaths})
    nValBaths_d = OrderedDict({ls: nValBaths})
    n_imp = 2 * (2 * ls + 1)

    # Coulomb U as a rank-4 tensor in the impurity spin-orbital index space.
    u4 = np.zeros((n_imp, n_imp, n_imp, n_imp), dtype=complex)
    uOp = finite.getUop(l1=ls, l2=ls, l3=ls, l4=ls, R=Fdd)
    nBaths_for_c2i = OrderedDict({ls: 0})
    for process, val in uOp.items():
        i = finite.c2i(nBaths_for_c2i, process[0][0])
        j = finite.c2i(nBaths_for_c2i, process[1][0])
        k = finite.c2i(nBaths_for_c2i, process[2][0])
        m = finite.c2i(nBaths_for_c2i, process[3][0])
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
    # Map (l,s,m) / (l,b) labels to single integer indices. Drop identically-zero terms
    # first: get_noninteracting_hamiltonian_operator unconditionally adds a 2p (l=1) SOC
    # operator whose terms are all 0.0 when xi_2p=0; those carry unmappable l=1 labels and
    # contribute nothing to H.
    hOp_int = {}
    for process, value in hOp.items():
        if abs(value) == 0:
            continue
        hOp_int[tuple((finite.c2i(sum_baths, spinOrb), action) for spinOrb, action in process)] = value

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
