r"""Dynamical local susceptibilities of the impurity (spin / orbital / charge / transverse).

For an impurity operator :math:`\hat A` the retarded local susceptibility is evaluated in
its Lehmann form over the thermal ground manifold,

.. math::
    \chi_A(z) = \sum_n w_n \sum_m \left[
        \frac{|\langle m|\hat A_+|n\rangle|^2}{z - (E_m - E_n)}
      - \frac{|\langle m|\hat A_-|n\rangle|^2}{z + (E_m - E_n)} \right],

with :math:`\hat A_+ = \hat A_- = \hat A` for the Hermitian operators (impurity
:math:`S_z`, :math:`L_z`, :math:`N`) and :math:`(\hat A_+, \hat A_-) = (S_+, S_-)` for
the transverse spin response. The two branches are computed with the block-Lanczos
resolvent machinery of :func:`impurityModel.ed.spectra.calc_spectra` (charge-conserving
seeds, so the excited sector equals the ground-state sector), evaluated on a real mesh
:math:`z = \omega + i\delta` and, from the same Lanczos coefficients, on the bosonic
Matsubara mesh :math:`z = i\nu_k = 2\pi i k \tau`.

**Elastic/Curie separation.** Each seed :math:`\hat A|n\rangle` is projected orthogonal
to the retained (near-)degenerate manifold of :math:`|n\rangle` before the Lanczos runs.
The resolvent part (the *regular* susceptibility) is then analytic at :math:`z = 0` —
in particular the bosonic :math:`\nu = 0` point is well defined and equals the Van Vleck
susceptibility. The projected-out weight is returned as the **Curie coefficient**

.. math::
    C_A = \sum_n w_n \sum_{m \in \mathrm{manifold}(n)} |\langle m|\hat A_+|n\rangle|^2
          - \Big|\sum_n w_n \langle n|\hat A_+|n\rangle\Big|^2,

the free-moment weight whose static contribution is :math:`C_A/\tau` (a quasi-elastic
:math:`\omega = 0` peak on the real axis). The full isothermal static susceptibility is
:math:`\chi_A(0) = C_A/\tau + \chi_A^\mathrm{VV}`. Comparing the spin, orbital and
charge results is the Hund's-metal diagnostic: a large spin Curie weight screened at a
low energy scale, next to an orbital response screened at a much higher scale and a
suppressed charge response.

This module is a *driver* on top of ``groundstate`` and ``spectra`` (see the layering rule in
``doc/architecture_overview.md``); the command-line interface lives in
:mod:`impurityModel.scripts.susceptibility`. Run it as

.. code-block:: bash

    impurityModel susceptibility h0_and_two_body.h5 ...

The results are written to a ``chi.h5`` file (one group per operator with ``realaxis`` /
``matsubara`` datasets, the Curie coefficient and the meshes).
"""

import numpy as np

from impurityModel.ed import spectra
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, inner
from impurityModel.ed.observables import make_impurity_casimir_operators, make_spin_operators
from impurityModel.ed.spin_pairs import resolve_spin_pairs
from impurityModel.ed.utils import report_banner

DEGENERACY_TOL = 1e-6


def bosonic_matsubara_mesh(tau, n_points):
    r"""Bosonic Matsubara frequencies :math:`\nu_k = 2\pi k \tau`, ``k = 0 .. n_points-1``.

    Real-valued; the driver evaluates the resolvent at :math:`z = i\nu_k`. The
    :math:`\nu = 0` point is included — it is regular for the manifold-projected seeds
    and equals the Van Vleck susceptibility.
    """
    return 2.0 * np.pi * tau * np.arange(n_points)


def _group_manifolds(es, degeneracy_tol):
    """Member-index list of the (near-)degenerate manifold of each state.

    Successive grouping on the (sorted) energies, matching
    :func:`observables.manifold_observable_values`.
    """
    es = np.asarray(es, dtype=float)
    order = np.argsort(es)
    groups = []
    current = [int(order[0])] if order.size else []
    for idx in order[1:]:
        if es[idx] - es[current[-1]] <= degeneracy_tol:
            current.append(int(idx))
        else:
            groups.append(current)
            current = [int(idx)]
    if current:
        groups.append(current)
    manifold_of = {}
    for group in groups:
        for idx in group:
            manifold_of[idx] = group
    return [manifold_of[i] for i in range(len(es))]


def build_susceptibility_operators(hOp, basis, rot_to_spherical):
    """Impurity susceptibility operators ``name -> (A_plus, A_minus)``.

    ``charge`` (:math:`N_\\mathrm{imp}`) always works. ``spin_z`` / ``orb_z`` use the
    many-body :math:`S_z` / :math:`L_z` from the Casimir build (same aggregate-and-retry
    degrade path as ``calc_gs``); ``spin_z`` falls back to the validated spin pairing
    when the shell is not spin-doubled. ``transverse`` (:math:`S_+, S_-`) needs the
    validated *impurity* pairing (exact also for a collinear spin-polarized bath, where
    only the bath pairing is a modelling choice).

    Returns
    -------
    (ops, skipped) : (dict, dict)
        ``ops[name] = (A_plus, A_minus)`` and ``skipped[name] = reason`` for the rest.
    """
    impurity_indices = sorted(orb for blocks in basis.impurity_orbitals.values() for block in blocks for orb in block)
    ops = {}
    skipped = {}
    n_op = ManyBodyOperator({((int(i), "c"), (int(i), "a")): 1.0 for i in impurity_indices})
    ops["charge"] = (n_op, n_op)

    l_ops = None
    try:
        l_ops, s_ops, _ = make_impurity_casimir_operators(basis.impurity_orbitals, rot_to_spherical)
    except ValueError:
        if not isinstance(rot_to_spherical, dict):
            try:
                l_ops, s_ops, _ = make_impurity_casimir_operators({0: [impurity_indices]}, rot_to_spherical)
            except ValueError:
                l_ops = None

    resolved = resolve_spin_pairs(
        hOp, basis.impurity_orbitals, basis.bath_states, rot_to_spherical, basis.num_spin_orbitals
    )
    if l_ops is not None:
        ops["spin_z"] = (s_ops[2], s_ops[2])
        ops["orb_z"] = (l_ops[2], l_ops[2])
    else:
        skipped["orb_z"] = "impurity is not a spin-doubled l-shell (L operator unavailable)"
        if resolved is not None:
            pair_sz = make_spin_operators(resolved[0])[2]
            ops["spin_z"] = (pair_sz, pair_sz)
        else:
            skipped["spin_z"] = "no l-shell rotation and no validated spin pairing"
    if resolved is not None:
        s_plus, s_minus, _ = make_spin_operators(resolved[0])
        ops["transverse"] = (s_plus, s_minus)
    else:
        skipped["transverse"] = "no trustworthy (dn, up) spin labelling (see resolve_spin_pairs)"
    return ops, skipped


def calc_susceptibility(
    hOp,
    psis,
    es,
    tau,
    basis,
    w,
    delta,
    matsubara_mesh=None,
    operators=None,
    rot_to_spherical=None,
    occ_cutoff=1e-12,
    slaterWeightMin=0.0,
    verbose=False,
    degeneracy_tol=DEGENERACY_TOL,
):
    r"""Compute the dynamical impurity susceptibilities ``chi(w + i delta)`` (+ Matsubara).

    Collective over ``basis.comm``. This is a driver on top of
    :func:`spectra.calc_spectra`: two resolvent branches per operator,
    :math:`\chi(z) = G_+(z) + G_-(-z)` (see the module docstring), with each Lanczos
    seed projected out of its state's degenerate manifold so the regular part is
    analytic at :math:`z = 0`; the projected weight becomes the Curie coefficient.

    Parameters
    ----------
    hOp : ManyBodyOperator
        The Hamiltonian.
    psis, es, tau
        Thermal eigenstates (rank-local when distributed, aligned with ``basis``),
        energies, thermal energy scale.
    basis : Basis
        The ground-state basis the states live on.
    w : ndarray
        Real frequency mesh (eV).
    delta : float
        Broadening above the real axis.
    matsubara_mesh : ndarray, optional
        Bosonic frequencies :math:`\nu_k \ge 0` (see :func:`bosonic_matsubara_mesh`);
        the resolvent is evaluated at :math:`z = i\nu_k` from the same Lanczos
        coefficients. ``None`` skips the Matsubara output.
    operators : dict, optional
        ``name -> (A_plus, A_minus)``; defaults to
        :func:`build_susceptibility_operators` (which needs ``rot_to_spherical``).
    rot_to_spherical : ndarray or dict, optional
        Needed only when ``operators`` is None.
    occ_cutoff, slaterWeightMin, verbose
        Passed to :func:`spectra.calc_spectra`.
    degeneracy_tol : float, default 1e-6
        Energy window for the degenerate-manifold seed projection.

    Returns
    -------
    dict or None
        On rank 0: ``{"w": w, "matsubara": nu | None, "operators": {name: {"realaxis":
        chi(w), "matsubara": chi(i nu) | None, "curie_coefficient": C, "expectation":
        <A_+>_th}}, "skipped": {name: reason}}``. ``None`` on other ranks.
    """
    comm = basis.comm
    rank = comm.rank if comm is not None else 0
    skipped = {}
    if operators is None:
        operators, skipped = build_susceptibility_operators(hOp, basis, rot_to_spherical)

    es = np.asarray(es, dtype=float)
    e0 = float(np.min(es))
    boltzmann = np.exp(-(es - e0) / tau)
    boltzmann /= np.sum(boltzmann)
    manifolds = _group_manifolds(es, degeneracy_tol)
    redistribute = basis.redistribute_psis if basis.is_distributed else (lambda state: (state,))

    # (branch, i_op, ei) -> projection coefficients onto the manifold members of ei.
    # Filled by the seed transforms identically on every rank (the projection runs
    # before the work-unit split), so the Curie coefficients need no gather.
    records = {}

    def make_transform(branch):
        def transform(ei, i_op, seed):
            # Align the apply-local seed with the bra partition, then project out the
            # degenerate manifold of state ei: seed -= sum_m |m><m|seed>.
            seed = redistribute(seed)[0]
            coeffs = np.zeros(len(manifolds[ei]), dtype=complex)
            for k, m in enumerate(manifolds[ei]):
                c = inner(psis[m], seed)
                if comm is not None:
                    c = comm.allreduce(c)
                coeffs[k] = c
                if c != 0.0:
                    seed = seed - c * psis[m]
            records[(branch, i_op, ei)] = coeffs
            return seed

        return transform

    dn_zero = dict.fromkeys(basis.impurity_orbitals, (0, 0))
    common = dict(
        psis=psis,
        es=es,
        tau=tau,
        basis=basis,
        slaterWeightMin=slaterWeightMin,
        verbose=verbose,
        occ_cutoff=occ_cutoff,
        dN_imp=dn_zero,
        dN_val=dn_zero,
        dN_con=dn_zero,
    )
    names = list(operators)
    tops_plus = [operators[name][0] for name in names]
    tops_minus = [operators[name][1] for name in names]
    extra_plus = [(1j * np.asarray(matsubara_mesh), 0.0)] if matsubara_mesh is not None else None
    extra_minus = [(-1j * np.asarray(matsubara_mesh), 0.0)] if matsubara_mesh is not None else None

    # chi(z) = G_+(z) + G_-(-z): the resonant branch on (w, delta) and the anti-resonant
    # branch on (-w, -delta) — the PES/IPS mesh-negation pattern of simulate_spectra.
    res_plus = spectra.calc_spectra(
        hOp, tops_plus, w=w, delta=delta, extra_meshes=extra_plus, seed_transform=make_transform("+"), **common
    )
    res_minus = spectra.calc_spectra(
        hOp,
        tops_minus,
        w=-np.asarray(w),
        delta=-delta,
        extra_meshes=extra_minus,
        seed_transform=make_transform("-"),
        **common,
    )
    if extra_plus is None:
        res_plus, res_minus = [res_plus], [res_minus]

    result = None
    if rank == 0:
        result = {
            "w": np.asarray(w),
            "matsubara": None if matsubara_mesh is None else np.asarray(matsubara_mesh),
            "operators": {},
            "skipped": skipped,
            "tau": float(tau),
            "delta": float(delta),
        }
        for i, name in enumerate(names):
            # Curie coefficient from the resonant-branch projections:
            # C = sum_n w_n sum_m |<m|A|n>|^2 - |<A>|^2 over each state's manifold.
            elastic2 = 0.0
            expectation = 0.0
            for ei in range(len(psis)):
                coeffs = records[("+", i, ei)]
                elastic2 += boltzmann[ei] * float(np.sum(np.abs(coeffs) ** 2))
                diag_k = manifolds[ei].index(ei)
                expectation += boltzmann[ei] * coeffs[diag_k]
            curie = elastic2 - abs(expectation) ** 2
            entry = {
                "realaxis": res_plus[0][:, i] + res_minus[0][:, i],
                "matsubara": None,
                "curie_coefficient": float(curie),
                "expectation": complex(expectation),
            }
            if matsubara_mesh is not None:
                entry["matsubara"] = res_plus[1][:, i] + res_minus[1][:, i]
            result["operators"][name] = entry
    return result


def save_susceptibility(result, filename):
    """Write a :func:`calc_susceptibility` result dict to ``filename`` (HDF5).

    Layout: ``w`` / ``matsubara_mesh`` datasets, ``tau`` / ``delta`` attrs, and one
    ``chi/<name>`` group per operator with ``realaxis`` / ``matsubara`` complex datasets
    and ``curie_coefficient`` / ``expectation`` attrs. Skipped operators are recorded as
    ``chi`` group attrs ``skipped_<name>``.
    """
    import h5py

    with h5py.File(filename, "w") as h5f:
        h5f.create_dataset("w", data=np.asarray(result["w"]))
        if result["matsubara"] is not None:
            h5f.create_dataset("matsubara_mesh", data=np.asarray(result["matsubara"]))
        h5f.attrs["tau"] = result["tau"]
        h5f.attrs["delta"] = result["delta"]
        chi_group = h5f.create_group("chi")
        for name, entry in result["operators"].items():
            grp = chi_group.create_group(name)
            grp.create_dataset("realaxis", data=entry["realaxis"])
            if entry["matsubara"] is not None:
                grp.create_dataset("matsubara", data=entry["matsubara"])
            grp.attrs["curie_coefficient"] = entry["curie_coefficient"]
            grp.attrs["expectation"] = entry["expectation"]
        for name, reason in result["skipped"].items():
            chi_group.attrs[f"skipped_{name}"] = reason


def print_susceptibility_summary(result):
    """Rank-0 console summary: static decomposition and screening scales per operator."""
    report_banner("Impurity susceptibilities")
    w = np.asarray(result["w"])
    tau = result["tau"]
    print(f"chi(0) = Curie/tau + VanVleck   (tau = {tau:.4g}, Curie = free-moment weight of the retained manifold)")
    rows = []
    for name, entry in result["operators"].items():
        curie = entry["curie_coefficient"]
        van_vleck = None
        if entry["matsubara"] is not None and result["matsubara"] is not None and result["matsubara"][0] == 0.0:
            van_vleck = float(np.real(entry["matsubara"][0]))
        chi0 = curie / tau + (van_vleck or 0.0)
        # Dominant inelastic response on the positive real axis: the screening scale.
        pos = w > 0
        w_peak = None
        if np.any(pos):
            spectral = -np.imag(entry["realaxis"][pos])
            if np.max(np.abs(spectral)) > 0:
                w_peak = float(w[pos][int(np.argmax(spectral))])
        rows.append((name, curie, van_vleck, chi0, w_peak))
    print(f"  {'operator':>10s} {'Curie':>12s} {'VanVleck':>12s} {'chi(0)':>12s} {'peak(-Im chi)':>14s}")
    for name, curie, vv, chi0, w_peak in rows:
        vv_s = f"{vv:12.6f}" if vv is not None else f"{'-':>12s}"
        peak_s = f"{w_peak:14.4f}" if w_peak is not None else f"{'-':>14s}"
        print(f"  {name:>10s} {curie:12.6f} {vv_s} {chi0:12.6f} {peak_s}")
    by_name = {r[0]: r for r in rows}
    if "spin_z" in by_name and "orb_z" in by_name:
        s_peak, o_peak = by_name["spin_z"][4], by_name["orb_z"][4]
        if s_peak is not None and o_peak is not None:
            print(
                f"  spin-orbital separation: spin response peaks at {s_peak:.4f}, orbital at {o_peak:.4f} "
                "(a spin scale well below the orbital scale, with a large spin Curie weight and a "
                "suppressed charge response, is the Hund's-metal fingerprint)"
            )
    for name, reason in result["skipped"].items():
        print(f"  {name}: skipped ({reason})")


def calc_susceptibility_workflow(
    model,
    meshes,
    basis,
    solver,
    *,
    comm,
    verbosity=0,
    cluster_label="cluster",
    num_wanted=5,
    n_matsubara=64,
    output_filename="chi.h5",
):
    """Solve the ground states, compute the dynamical susceptibilities, and save/print them.

    Mirrors ``calc_selfenergy``'s grouped signature and ground-state stage (solver-basis
    preparation + ``calc_gs``), then runs :func:`calc_susceptibility` and, on rank 0, writes
    ``output_filename`` and prints the summary. Returns the result dict (rank 0) /
    ``None`` (other ranks).

    Parameters
    ----------
    model : impurityModel.ed.model.ImpurityModel
        The impurity problem (``h0``, ``u4``, ``impurity_orbitals``, ``rot_to_spherical``).
    meshes : impurityModel.ed.model.Meshes
        Real frequency mesh ``w`` and the real-axis smearing ``delta``. The bosonic Matsubara
        mesh is built from ``basis.tau`` and ``n_matsubara`` (see :func:`bosonic_matsubara_mesh`).
    basis : impurityModel.ed.model.BasisOptions
        Nominal occupation, mixed valence, temperature ``tau``, occupation cutoff, minimum
        Slater weight and the determinant budget.
    solver : impurityModel.ed.model.SolverOptions
        Provides the dense-eigensolver cutoff.
    comm : MPI.Comm or None
        MPI communicator.
    verbosity : int, optional
        Verbosity level.
    cluster_label : str, optional
        Label for the cluster (used in the output message).
    num_wanted : int, optional
        Number of ground-state eigenstates to solve for.
    n_matsubara : int, optional
        Number of bosonic Matsubara points (0 disables the Matsubara output).
    output_filename : str or None, optional
        HDF5 output path; ``None`` skips saving.
    """
    # Imported here (not at module top) to keep the module importable without pulling in
    # the whole self-energy stack when only the calc_susceptibility driver is used.
    from impurityModel.ed.groundstate import calc_gs
    from impurityModel.ed.memory_estimate import suggest_truncation_threshold
    from impurityModel.ed.selfenergy import _prepare_solver_basis

    # Unpack the grouped parameters into the local names used throughout the body.
    h0 = model.h0
    u4 = model.u4
    impurity_orbitals = model.impurity_orbitals
    rot_to_spherical = model.rot_to_spherical
    nominal_occ = basis.nominal_occ
    mixed_valence = basis.mixed_valence
    tau = basis.tau
    occ_cutoff = basis.occ_cutoff
    slaterWeightMin = basis.slater_weight_min
    truncation_threshold = basis.truncation_threshold
    w = meshes.w
    delta = meshes.delta
    dense_cutoff = solver.dense_cutoff

    rank = comm.rank if comm is not None else 0
    sb = _prepare_solver_basis(h0, u4, impurity_orbitals, nominal_occ, mixed_valence, rot_to_spherical, verbosity)
    gf_block_width = max(4, *(len(block) for block in sb.block_structure.blocks))
    if truncation_threshold is None:
        truncation_threshold = suggest_truncation_threshold(
            sb.n_spin_orbitals, comm=comm, block_width=gf_block_width, reort=None, method="lanczos"
        )
    basis_information = {
        "impurity_orbitals": sb.impurity_orbitals,
        "bath_states": sb.bath_states,
        "N0": sb.nominal_occ,
        "mixed_valence": sb.mixed_valence,
        "tau": tau,
        "chain_restrict": False,
        "dense_cutoff": dense_cutoff,
        "spin_flip_dj": False,
        "rank": rank,
        "comm": comm,
        "truncation_threshold": truncation_threshold,
    }
    psis, es, ground_state_basis, _thermal_rho, _gs_info = calc_gs(
        sb.h,
        basis_information,
        sb.block_structure,
        sb.rot_to_spherical,
        verbosity >= 2,
        slaterWeightMin=slaterWeightMin,
        num_wanted=num_wanted,
    )
    matsubara_mesh = bosonic_matsubara_mesh(tau, n_matsubara) if n_matsubara > 0 else None
    result = calc_susceptibility(
        sb.h,
        psis,
        es,
        tau,
        ground_state_basis,
        w,
        delta,
        matsubara_mesh=matsubara_mesh,
        rot_to_spherical=sb.rot_to_spherical,
        occ_cutoff=occ_cutoff,
        slaterWeightMin=slaterWeightMin,
        verbose=verbosity >= 1,
    )
    if rank == 0 and result is not None:
        print_susceptibility_summary(result)
        if output_filename is not None:
            save_susceptibility(result, output_filename)
            print(f"\nSusceptibilities written to {output_filename} (cluster '{cluster_label}').")
    return result
