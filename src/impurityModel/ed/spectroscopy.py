"""The spectroscopy driver: solve the ground state once, calculate any combination of spectra.

* :func:`calc_spectroscopy` takes a model and a list of per-technique option objects
  (:class:`PESOptions`, :class:`XPSOptions`, :class:`NIXSOptions`, :class:`XASOptions`,
  :class:`RIXSOptions`) and returns the datasets (rank 0), the way :func:`selfenergy.calc_selfenergy` does;
  :func:`calc_xas`, :func:`calc_rixs`, ... are its one-spectrum forms.
* :func:`save_spectra` writes the ``spectra.h5`` the plot scripts read; :func:`run_spectra` is both.
* :func:`build_spectra_model` assembles the full interacting :class:`ImpurityModel` from a
  non-interacting ``h0`` file.

The techniques themselves are in :mod:`impurityModel.ed.spectra_techniques`.
"""

import os
from collections import OrderedDict

import h5py
import numpy as np

from impurityModel.ed import config
from impurityModel.ed.block_structure import BlockStructure
from impurityModel.ed.groundstate import GS_DE2_MIN, GS_E_PT2_TOL
from impurityModel.ed.gs_calc import GroundStateSpec, calc_gs
from impurityModel.ed.gs_report import ReportFiles
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.model import load_model
from impurityModel.ed.observables import shell_column_slices, shell_index_of_orbitals

# Local stuff
from impurityModel.ed.sector_ensemble import resolve_mixed_valence
from impurityModel.ed.solver_basis import _flatten_bath_split, assemble_interacting_operator
from impurityModel.ed.spectra_techniques import (
    IndexedOperators,
    LabelledOperators,
    ShellLayout,
    SpectraTarget,
    merge_spectra_datasets,
    ordered_requests,
    run_techniques,
)
from impurityModel.ed.symmetries import (
    classify_bath_occupation,
    extract_tensors,
    impurity_block_structure,
)
from impurityModel.ed.utils import V_RESULT, Reporter


def build_spectra_model(
    h0_filename,
    ls,
    nBaths,
    nValBaths,
    n0imps,
    F_vv,
    F_cc,
    F_cv,
    G_cv,
    xi_core,
    xi_valence,
    chargeTransferCorrection,
    hField,
    rank=0,
    verbose=True,
    *,
    valence_l,
    core_l=None,
    valence_u4=None,
):
    """Assemble the full interacting spectra model from a non-interacting ``h0`` file.

    Thin wrapper over :func:`model.load_model`'s multi-shell path
    (:meth:`ImpurityModel.from_shells`), kept so callers and tests do not need to build ``OrderedDict``s themselves. ``h0_filename`` accepts either a
    labelled ``.pickle``/``.json``/``.dat`` file or a self-describing flat ``.h0`` (see
    :func:`hamiltonian_io.read_h0_operator`).

    Unlike the self-energy path (``h0`` + separate ``u4``), the spectra driver works with the
    *full* single-index interacting operator that :func:`hamiltonian_io.get_hamiltonian_operator`
    builds (core + correlated shells, SOC, magnetic field, atomic Coulomb and double counting all
    folded in). It is carried as :attr:`ImpurityModel.h0` with ``u4=None``; the explicit
    multi-shell ``(valence_baths, conduction_baths)`` partition is stored in
    :attr:`ImpurityModel.bath_states`.

    Parameters
    ----------
    h0_filename : str
        Non-interacting Hamiltonian file.
    ls : sequence of int
        Angular momenta of the shells (e.g. ``(1, 2)`` for a 2p core + 3d valence shell).
        This fixes the ``c2i`` index layout only; which shell is the core one is said by
        ``core_l``, never by this sequence's order.
    nBaths, nValBaths : sequence of int
        Total / valence bath-state counts, one per shell in ``ls``.
    n0imps : sequence of int
        Nominal impurity occupation per shell (used for the double-counting term).
    F_vv, F_cc, F_cv, G_cv : sequence of float
        Slater-Condon parameters: valence-valence direct, core-core direct, core-valence
        direct and core-valence exchange. The last three are ignored when ``core_l`` is
        ``None``.
    xi_core, xi_valence : float
        Spin-orbit couplings of the core and valence shells.
    chargeTransferCorrection : float
        Double-counting parameter.
    hField : sequence of float
        Magnetic field ``(hx, hy, hz)``, applied to the valence shell.
    rank : int
        MPI rank, forwarded to the reader for rank-0 logging.
    verbose : bool
        Whether the reader logs on rank 0.
    valence_l : int
        Angular momentum of the valence (correlated) shell.
    core_l : int, optional
        Angular momentum of the core shell; ``None`` when the model has no core shell.

    Returns
    -------
    ImpurityModel
        With ``h0`` = the full interacting operator, ``u4=None``, ``impurity_orbitals`` the
        per-shell block lists, and ``bath_states = (valence_baths, conduction_baths)``.
    """
    return load_model(
        h0_filename,
        shells=OrderedDict(zip(ls, nBaths)),
        val_shells=OrderedDict(zip(ls, nValBaths)),
        n0imps=OrderedDict(zip(ls, n0imps)),
        slater_condon=(F_vv, F_cc, F_cv, G_cv),
        socs=(xi_core, xi_valence),
        charge_transfer_correction=chargeTransferCorrection,
        h_field=hField,
        valence_l=valence_l,
        core_l=core_l,
        u4=valence_u4,
        rank=rank,
        verbose=verbose,
    )


def _resolve_shell_roles(nBaths):
    """Return ``(l_valence, l_core)`` for a spectra model, from the bath counts.

    A last resort, for a model that does not carry :attr:`ImpurityModel.valence_l` -- one
    built from an archive or a bare matrix rather than through
    :meth:`ImpurityModel.from_shells`, which is told the roles outright. The rule is that the
    core shell is the one with no bath of its own, which is what every RSPt-produced workload
    looks like (one hybridization file per orbital group, and a core group has none).

    It is a guess, and a bath-less *valence* shell -- the Hubbard-I approximation -- is exactly
    the case it would get backwards. So a layout it cannot read unambiguously raises and says
    to carry the roles instead.

    Parameters
    ----------
    nBaths : Mapping
        ``{l: n_bath}`` for every shell in the model.

    Returns
    -------
    tuple
        ``(l_valence, l_core)``; ``l_core`` is ``None`` when the model has no core shell.

    Raises
    ------
    ValueError
        If the shells do not resolve to exactly one valence shell and at most one core shell.
    """
    with_bath = [l for l, n in nBaths.items() if n > 0]
    without_bath = [l for l, n in nBaths.items() if n == 0]

    if len(nBaths) == 1:
        return next(iter(nBaths)), None
    if len(with_bath) == 1 and len(without_bath) == 1:
        return with_bath[0], without_bath[0]
    raise ValueError(
        f"Cannot tell which shell is the core one from the bath counts {dict(nBaths)}. Read this "
        "way, a spectra model needs exactly one shell with bath states (the valence shell) and at "
        "most one without (the core shell) -- which a bath-less valence shell (Hubbard-I) "
        "alongside a core shell is not. Build the model through ImpurityModel.from_shells, which "
        "is told valence_l/core_l outright, rather than leaving them to be inferred."
    )


def _group_of(technique):
    """The ``spectra.h5`` group a technique's datasets live under."""
    return {"pes": "PS", "xps": "XPS", "nixs": "NIXS", "xas": "XAS", "rixs": "RIXS"}[technique]


def _request_meshes(technique, request):
    """The meshes of one request, keyed by the dataset name they are stored under."""
    if technique in ("pes", "xps", "xas"):
        return {"w": np.asarray(request.w)}
    if technique == "rixs":
        return {"wIn": np.asarray(request.w_in), "wLoss": np.asarray(request.w_loss)}
    radial_mesh, r_final, r_initial = request.radial
    return {
        "wLoss": np.asarray(request.w_loss),
        "qs": np.asarray(request.resolved_qs),
        "r": np.asarray(radial_mesh),
        "Ri": np.asarray(r_final),
        "Rj": np.asarray(r_initial),
    }


def _is_block_model(model):
    """``True`` for a model built from blocks (ndarray ``rot_to_spherical``), not through ``from_shells``."""
    return not isinstance(model.rot_to_spherical, dict)


#: Largest residual of a site-frame fit (:func:`~impurityModel.ed.local_operators.site_frames`) that is
#: accepted as "the supplied L is a rotation of the spherical one". RSPt prints its matrices with eight
#: decimals and computes them to roundoff, so a genuine rotation fits far below this.
SITE_FRAME_TOL = 1e-6


def _block_model_inputs(model, hOp):
    """Shell-keyed solver inputs and the index-addressed operator provider of a block model.

    Shells are the column blocks of ``rot_to_spherical`` (``model.shell_ls``); an impurity orbital belongs
    to the shell holding its weight. The bath is attached to the valence shell. Keys are shell indices.
    """
    if model.valence_shell is None:
        raise ValueError(
            "a block model needs to know which shell is the valence shell for spectroscopy: pass valence_shell= "
            "(and core_shell= when core-level techniques are wanted) to ImpurityModel.from_blocks / from_solver_matrix."
        )
    shell_ls = model.resolved_shell_ls
    for name, shell in (("valence_shell", model.valence_shell), ("core_shell", model.core_shell)):
        if shell is not None and not 0 <= shell < len(shell_ls):
            raise ValueError(f"{name}={shell} is not an index into shell_ls={shell_ls}")
    if model.core_shell is not None and model.core_shell == model.valence_shell:
        raise ValueError("valence_shell and core_shell must name different shells")
    rot = np.asarray(model.rot_to_spherical)
    impurity_indices = model.impurity_indices
    shell_of = shell_index_of_orbitals(rot, shell_ls)
    slices = shell_column_slices(shell_ls)
    rows = {a: [o for o, sa in zip(impurity_indices, shell_of) if sa == a] for a in range(len(shell_ls))}
    rows = {a: r for a, r in rows.items() if r}
    n_orb = model.n_spin_orbitals
    if model.bath_states is not None:
        valence_flat, conduction_flat = _flatten_bath_split(model.bath_states, impurity_indices, n_orb)
    else:
        h_matrix = extract_tensors(hOp, n_orb=n_orb, two_body=False)[0]
        valence_flat, conduction_flat = classify_bath_occupation(hOp, impurity_indices, n_orb=n_orb, h0_matrix=h_matrix)
    impurity_orbitals = {a: [list(r)] for a, r in rows.items()}
    valence_baths = {a: [[] for _ in impurity_orbitals[a]] for a in rows}
    conduction_baths = {a: [[] for _ in impurity_orbitals[a]] for a in rows}
    if valence_flat:
        valence_baths[model.valence_shell] = [list(valence_flat)]
    if conduction_flat:
        conduction_baths[model.valence_shell] = [list(conduction_flat)]
    local_row = {o: k for k, o in enumerate(impurity_indices)}
    rotations = {a: rot[[local_row[o] for o in r]][:, slices[a]] for a, r in rows.items()}
    supplied = model.operators.dipole if model.operators is not None else None
    # With operators in the global (crystal) frame, polarisation vectors are crystal coordinates for the Gaunt
    # dipole too: each shell's local frame is fitted against the supplied L.
    frames = model.site_frames() or [None] * len(shell_ls)
    for a, frame in enumerate(frames):
        if frame is not None and frame.residual > SITE_FRAME_TOL:
            raise ValueError(
                f"shell {a}: the supplied orbital operators are not a rotation of the spherical ones in the basis "
                f"rot_to_spherical describes (residual {frame.residual:.2e}); the crystal frame of the polarisation "
                "vectors cannot be determined"
            )
    operators = IndexedOperators(
        rows,
        rotations,
        {a: shell_ls[a] for a in rows},
        default_dipole=supplied,
        frames={a: frames[a].rotation for a in rows if frames[a] is not None},
    )
    return impurity_orbitals, valence_baths, conduction_baths, dict(rotations), operators


def _model_inputs(model, hOp):
    """``(impurity_orbitals, valence_baths, conduction_baths, rot_to_spherical dict)`` for the ground state."""
    if _is_block_model(model):
        return _block_model_inputs(model, hOp)[:4]
    valence_baths, conduction_baths = model.bath_states
    return model.impurity_orbitals, valence_baths, conduction_baths, dict(model.rot_to_spherical)


def _model_hamiltonian(model):
    """The full interacting operator: ``h0`` as is when ``dc``/``u4`` are absent, else ``h0 - dc + U(u4)``."""
    if model.dc is None and model.u4 is None:
        return ManyBodyOperator(model.h0)
    return assemble_interacting_operator(model.h0, model.dc, model.u4)


def spectra_layout(model, requests, *, auto_block_structure=True, hOp=None, report=None):
    """The :class:`ShellLayout` (and the block structure the ground state is solved in) of ``model``.

    Parameters
    ----------
    model : ImpurityModel
        A model built through :meth:`ImpurityModel.from_shells` (or any model carrying ``valence_l``).
    requests : sequence
        The spectrum requests; only used to validate that core-level techniques have a core shell.
    auto_block_structure : bool
        Derive the block structure from the hybridization-dressed impurity matrix instead of one
        block per shell.

    Returns
    -------
    layout : ShellLayout
    block_structure : BlockStructure
        For the ground-state solve.
    nBaths : OrderedDict
        ``{l: n_bath}`` per shell.
    """
    hOp = _model_hamiltonian(model) if hOp is None else hOp
    block_model = _is_block_model(model)
    if block_model:
        impurity_orbitals, valence_baths, conduction_baths, _rot, block_operators = _block_model_inputs(model, hOp)
    else:
        impurity_orbitals = model.impurity_orbitals
        valence_baths, conduction_baths = model.bath_states
    ls = list(impurity_orbitals.keys())
    nBaths = OrderedDict(
        (l, sum(len(b) for b in valence_baths[l]) + sum(len(b) for b in conduction_baths[l])) for l in ls
    )
    n_spin_orbitals = model.n_spin_orbitals

    # Fall-back block structure -- one block per impurity shell -- used when
    # auto_block_structure is off. Derived from impurity_orbitals rather than written out as
    # [range(6), range(6, 16)], which was the 2p + 3d spin-orbital counts spelled as literals.
    shell_blocks = [sorted(orb for block in impurity_orbitals[l] for orb in block) for l in ls]
    n_shell_blocks = len(shell_blocks)
    block_structure = BlockStructure(
        blocks=shell_blocks,
        identical_blocks=[[i] for i in range(n_shell_blocks)],
        transposed_blocks=[[] for _ in range(n_shell_blocks)],
        particle_hole_blocks=[[] for _ in range(n_shell_blocks)],
        particle_hole_transposed_blocks=[[] for _ in range(n_shell_blocks)],
        inequivalent_blocks=list(range(n_shell_blocks)),
    )

    # The solver works in the basis the model is given; nothing here rotates it. Choosing a good basis
    # is the caller's job (rspt2spectra / the interface). Which shell is which is carried by the model,
    # because from_shells was told. Only a model built some other way (an archive, a bare matrix) falls
    # back to reading the roles off the bath counts.
    if block_model:
        l_valence, l_core = model.valence_shell, model.core_shell
    elif model.valence_l is not None:
        l_valence, l_core = model.valence_l, model.core_l
    else:
        l_valence, l_core = _resolve_shell_roles(nBaths)

    # Default: derive the block structure from the hybridization-dressed impurity matrix
    # (impurity_block_structure) rather than the hand-coded one. It matches or strictly refines
    # the manual structure (e.g. SOC / crystal field splits each shell into sub-blocks) and
    # fixes bath-mediated coupling.
    correlated_block_structure = None
    if auto_block_structure:
        impurity_indices = sorted(orb for blocks in impurity_orbitals.values() for block in blocks for orb in block)
        h_matrix = extract_tensors(hOp, n_orb=n_spin_orbitals, two_body=False)[0]
        block_structure = impurity_block_structure(hOp, impurity_indices, h0_matrix=h_matrix)
        if report is not None:
            report(f"Auto-derived block structure: {len(block_structure.blocks)} blocks", level=V_RESULT)
        if l_valence in impurity_orbitals:
            d_indices = sorted(orb for block in impurity_orbitals[l_valence] for orb in block)
            correlated_block_structure = impurity_block_structure(hOp, d_indices, h0_matrix=h_matrix)
            if report is not None:
                report(
                    f"{len(correlated_block_structure.inequivalent_blocks)} inequivalent PES/IPS classes "
                    f"for the l={l_valence} shell.",
                    level=V_RESULT,
                )
    if block_model and any(type(r).__name__ == "NIXSOptions" for r in requests):
        raise ValueError(
            "NIXS needs the radial wave functions and an (l, s, m)-labelled model, so it is not available on a "
            "block model (ImpurityModel.from_blocks / from_solver_matrix); build the model with from_shells."
        )
    layout = ShellLayout(
        valence_key=l_valence,
        core_key=l_core,
        operators=block_operators if block_model else LabelledOperators(nBaths),
        correlated_block_structure=correlated_block_structure,
    )
    ordered_requests(requests, layout)
    if block_model and report is not None:
        _report_dipole_frame(model, block_operators, l_valence, l_core, report)
    return layout, block_structure, nBaths


def _report_dipole_frame(model, operators, valence, core, report):
    """Say which frame polarisation vectors are in and, for a supplied core->valence dipole, how it compares
    with the frame-rotated Gaunt one (B4 of the Dxyz plan)."""
    frames = model.site_frames() or []
    rotated = [(a, f) for a, f in enumerate(frames) if f is not None]
    if rotated:
        report("Polarisation vectors are crystal (global) coordinates.", level=V_RESULT)
        for a, f in rotated:
            report(f"  shell {a} (l={model.resolved_shell_ls[a]}): {f.describe()}", level=V_RESULT)
    else:
        report("Polarisation vectors are in the shells' local frame (rot_to_spherical).", level=V_RESULT)
    if core is None:
        return
    from impurityModel.ed.spectra import CARTESIAN

    operators.dipole_checks.clear()
    operators.dipole(CARTESIAN, core, valence)
    if not operators.dipole_checks:
        source = "supplied dipole has no core->valence block, " if operators.default_dipole is not None else ""
        report(f"Dipole: {source}atomic (Gaunt) matrix elements, unit radial integral.", level=V_RESULT)
        return
    parts = ", ".join(
        f"{axis}: overlap {ov:.3f}, phase {np.degrees(ph):+.1f} deg"
        for axis, (_, _, k) in zip("xyz", sorted(operators.dipole_checks))
        for ov, ph in [operators.dipole_checks[(core, valence, k)]]
    )
    report(
        "Dipole: supplied core->valence block (RSPt's Dxyz). Against the frame-rotated Gaunt block: "
        f"{parts} (a momentum-type matrix is +-90 deg; overlap < 1 is crystal-field mixing or a frame error).",
        level=V_RESULT,
    )


def calc_spectroscopy(
    model,
    requests,
    basis,
    solver=None,
    *,
    comm,
    verbosity=V_RESULT,
    cluster_label="cluster",
    report_dir=None,
    auto_block_structure=True,
):
    """Solve the ground state of ``model`` once and calculate the requested spectra.

    Parameters
    ----------
    model : ImpurityModel
        The full interacting model from :func:`build_spectra_model` (``bath_states`` set).
    requests : sequence
        Spectrum requests: at most one :class:`PESOptions`, :class:`XPSOptions`, :class:`NIXSOptions`,
        :class:`XASOptions` or :class:`RIXSOptions` each, in any order (they run in the order PES, XPS,
        NIXS, XAS, RIXS). Each carries its own mesh and broadening.
    basis : BasisOptions
        Nominal occupation, determinant budget and ``tau = k_B * T``.
    solver : SolverOptions, optional
        Supplies ``dense_cutoff`` and ``gf_pt2_tol``. ``None`` uses ``config.GS_DENSE_CUTOFF`` and the
        amplitude rule.
    comm : mpi4py communicator
        The communicator every collective runs on.
    verbosity : int, optional
        Printing level (rank-uniform; printing itself is gated on rank downstream).
    cluster_label, report_dir : str, optional
        Name and directory of the ground-state report files (``ground_state-<label>.txt/.json``).
    auto_block_structure : bool, optional
        See :func:`spectra_layout`.

    Returns
    -------
    dict or None
        On rank 0: ``{"E", "datasets", "meshes", "ensemble", "gs_info"}`` where ``datasets`` maps dataset
        name (``PS/spectra``, ``XAS/tensor``, ...) to array and ``meshes`` maps technique name to its
        meshes. ``None`` on the other ranks. Write it with :func:`save_spectra`.
    """
    config.assert_uniform(comm)  # collective: a rank-divergent env knob is an error, not a deadlock
    # Replicated input: a malformed value raises on every rank before any collective.  The default is on.
    mixed_valence, _mv_note = resolve_mixed_valence(basis.mixed_valence, driver_default="on")
    requests = list(requests)
    if not requests:
        raise ValueError("no spectrum was requested, so there is nothing to compute")
    rank = comm.rank if comm is not None else 0
    # The stage one-liners below are the provenance heartbeat users pipe to logs; they stay
    # ungated (level V_RESULT) regardless of verbosity, so `report` here is only for the
    # shared rank gate and formatting, not for hiding anything at the default level.
    report = Reporter(verbosity, rank)

    hOp = _model_hamiltonian(model)
    impurity_orbitals, valence_baths, conduction_baths, rot_to_spherical = _model_inputs(model, hOp)

    report(f"Number of spin-orbitals: {model.n_spin_orbitals}", level=V_RESULT)
    layout, block_structure, _nBaths = spectra_layout(
        model, requests, auto_block_structure=auto_block_structure, hOp=hOp, report=report
    )
    l_core = layout.core_key
    # Measure how many physical processes the Hamiltonian contains.
    report(f"Hamiltonian contains {len(hOp)} terms.", level=V_RESULT)
    # Many body basis for the ground state
    report("Creating the many-body basis ...", level=V_RESULT, flush=True)
    tau = basis.tau
    # What decides which state is found, explicit (A1). Spectra solves at the calc_gs default Slater weight
    # (0) but builds its restrictions at basis.slater_weight_min. dense_cutoff and the CIPSI tolerances take
    # the values the self-energy takes (P2); chain_restrict stays False on purpose: the chain restriction
    # truncates the initial multiplet and spectra needs it complete (symmetry over energy).
    gs_spec = GroundStateSpec(
        tau=tau,
        chain_restrict=False,
        dense_cutoff=config.GS_DENSE_CUTOFF if solver is None else solver.dense_cutoff,
        e_pt2_tol=GS_E_PT2_TOL if basis.e_pt2_tol is None else basis.e_pt2_tol,
        de2_min=GS_DE2_MIN if basis.de2_min is None else basis.de2_min,
        # None = "as many determinants as fit in RAM", resolved against the per-rank available memory inside
        # find_ground_state_sector (see memory_estimate).
        truncation_threshold=basis.truncation_threshold,
        # No excitation budget: solve_member appends the bounds observed on the converged ground state; the
        # XAS/PES excited bases inherit them (widened) via spectra/greens_function; RIXS attaches them (rixs.py).
        cost_budget=basis.cost_budget,
        restriction_slater_weight_min=basis.slater_weight_min,
        mixed_valence=mixed_valence,
    )
    basis_setup = {
        "impurity_orbital": impurity_orbitals,
        "bath_states": (valence_baths, conduction_baths),
        "nominal_impurity_occ": basis.nominal_occ,
        # Pin the core shell at generation time so it cannot drain into the valence shell (a
        # 2p^4 d^10 configuration otherwise wins by ~23 eV and zeroes every core-level
        # spectrum -- see doc/basis_and_restrictions.md). Keyed on the shell the model says is
        # the core one, not on "the shell with no bath": those agreed only while both were the
        # same inference, and a core shell that does carry bath states would be left unpinned.
        "frozen_occupations": set() if l_core is None else {l_core},
        "comm": comm,
    }

    def spectra_of(member_es, member_psis, member_basis):
        """One member's spectra; the one-sector run and every ensemble member share it."""
        target = SpectraTarget(
            hOp=hOp,
            psis=member_psis,
            es=member_es,
            tau=tau,
            basis=member_basis,
            occ_cutoff=basis.occ_cutoff,
            slater_weight_min=basis.slater_weight_min,
            verbose=verbosity >= 1,
            gf_pt2_tol=None if solver is None else solver.gf_pt2_tol,
            layout=layout,
        )
        return run_techniques(target, requests)

    class _SpectraVisitor:
        def visit(self, _member, state):
            # Collective, run while the member's basis is alive and after its solver is released.
            return spectra_of(state.es, state.states, state.basis)

    _psis, es, _basis, _rho, gs_info = calc_gs(
        hOp,
        basis_setup,
        block_structure,
        rot_to_spherical,
        verbosity,
        spec=gs_spec,
        visitor=_SpectraVisitor(),
        report_files=ReportFiles.for_label(cluster_label, report_dir),
    )
    ensemble = gs_info["ensemble"]

    report.banner("Spectra", level=V_RESULT)
    report(f"Considering {len(es)} eigenstate(s) for the spectra.", level=V_RESULT)
    report("Calculating spectra ...", level=V_RESULT, flush=True)
    failure = None  # a rank-0 verdict: raising on rank 0 alone would leave the others in the Barrier below
    if len(gs_info["member_results"]) == 1:  # one member: its spectra are the file
        datasets = gs_info["member_results"][0]
    else:
        datasets = {}
        if rank == 0:
            try:
                weights = [m["weight"] for m in ensemble["members"]]
                datasets = merge_spectra_datasets(gs_info["member_results"], weights)
                if len(weights) > 1:  # one member: the file is the one-sector file, bit for bit
                    datasets["ensemble/weights"] = np.asarray(weights)
                    datasets["ensemble/e0"] = np.asarray([m["e0"] for m in ensemble["members"]])
                    datasets["ensemble/total"] = np.asarray([m["total"] for m in ensemble["members"]])
            except Exception as exc:
                failure = f"merging the sector-ensemble spectra failed: {exc!r}"

    from impurityModel.ed.selfenergy import _raise_together

    _raise_together(comm, failure)
    if comm is not None:
        comm.Barrier()
    report("\nDone.", level=V_RESULT)
    if rank != 0:
        return None
    meshes = {name: _request_meshes(name, request) for name, request in ordered_requests(requests)}
    return {"E": es, "datasets": datasets, "meshes": meshes, "ensemble": ensemble, "gs_info": gs_info}


def _single(request_type_name):
    def calc(model, request, basis, solver=None, **kwargs):
        return calc_spectroscopy(model, [request], basis, solver, **kwargs)

    calc.__name__ = f"calc_{request_type_name}"
    calc.__doc__ = (
        f"Calculate the {request_type_name.upper()} spectrum alone: :func:`calc_spectroscopy` with one request.\n\n"
        "Takes the same keyword arguments (``comm`` is required) and returns the same result."
    )
    return calc


calc_pes = _single("pes")
calc_xps = _single("xps")
calc_nixs = _single("nixs")
calc_xas = _single("xas")
calc_rixs = _single("rixs")


def save_spectra(result, filename):
    """Write a :func:`calc_spectroscopy` result as the ``spectra.h5`` the plot scripts read.

    Collective-free: call it on every rank; only the rank holding ``result`` (rank 0) writes.

    The file holds ``E``, the datasets, and the mesh each spectrum was computed on stored with it
    (``PS/w``, ``XPS/w``, ``XAS/w``, ``RIXS/wIn``, ``RIXS/wLoss``, ``NIXS/wLoss``, ...). The plot
    scripts read those. The historical top-level ``w``, ``wIn``, ``wLoss`` and ``qsNIXS`` are kept
    for older readers, but only where they are unambiguous: ``w`` when every requested PES/XPS/XAS
    shares one mesh, ``wLoss`` when RIXS and NIXS do. They are never filled with a default for a
    technique that was not run, and never hold one technique's mesh when another differs.
    """
    if result is None:
        return
    meshes = result["meshes"]

    def shared(techniques, name):
        found = [np.asarray(meshes[t][name]) for t in techniques if t in meshes]
        if found and all(m.shape == found[0].shape and np.array_equal(m, found[0]) for m in found):
            return found[0]
        return None

    legacy = {
        "w": shared(("pes", "xps", "xas"), "w"),
        "wIn": shared(("rixs",), "wIn"),
        "wLoss": shared(("rixs", "nixs"), "wLoss"),
    }
    nixs = meshes.get("nixs")
    with h5py.File(filename, "w") as h5f:
        h5f.create_dataset("E", data=result["E"])
        for name, mesh in legacy.items():
            if mesh is not None:
                h5f.create_dataset(name, data=mesh)
        if nixs:
            h5f.create_dataset("qsNIXS", data=nixs["qs"])
            h5f.create_dataset("r", data=nixs["r"])
            h5f.create_dataset("RiNIXS", data=nixs["Ri"])
            h5f.create_dataset("RjNIXS", data=nixs["Rj"])
        for technique, technique_meshes in meshes.items():
            for name, mesh in technique_meshes.items():
                h5f.create_dataset(f"{_group_of(technique)}/{name}", data=mesh)
        for name, data in result["datasets"].items():
            h5f.create_dataset(name, data=data)


def run_spectra(
    model,
    requests,
    basis,
    comm,
    *,
    verbosity=None,
    output_filename="spectra.h5",
    solver=None,
    auto_block_structure=True,
):
    """:func:`calc_spectroscopy` followed by :func:`save_spectra` to ``output_filename``.

    The ground-state report files are named after ``output_filename`` (``ground_state-<stem>.txt``,
    next to it).
    """
    result = calc_spectroscopy(
        model,
        requests,
        basis,
        solver,
        comm=comm,
        verbosity=V_RESULT if verbosity is None else verbosity,
        cluster_label=os.path.splitext(os.path.basename(output_filename))[0],
        report_dir=os.path.dirname(output_filename) or None,
        auto_block_structure=auto_block_structure,
    )
    save_spectra(result, output_filename)
    return result
