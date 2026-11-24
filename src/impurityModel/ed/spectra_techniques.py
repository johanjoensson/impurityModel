"""One function per spectrum.

``simulate_spectra`` used to compute PES, XPS, NIXS, XAS and RIXS in one body. Each is now a
function of a :class:`SpectraTarget` (one ground-state member plus what every technique needs) and
that technique's own options object, returning its datasets (``{}`` off the root rank):

* :func:`pes_spectrum`  -> ``PS/spectra``
* :func:`xps_spectrum`  -> ``XPS/spectra``
* :func:`nixs_spectrum` -> ``NIXS/spectra``
* :func:`xas_spectrum`  -> ``XAS/tensor`` or ``XAS/projected``
* :func:`rixs_spectrum` -> ``RIXS/tensor`` or ``RIXS/projected``

:data:`TECHNIQUES` lists them in the fixed execution order (PES, XPS, NIXS, XAS, RIXS); a driver always
runs requests in that order, so floating-point summation and printed output do not depend on the
order the caller listed them in.

Which operators a technique applies is the :class:`ShellLayout`'s business: it carries an operator
provider (:class:`LabelledOperators` for the ``(l, s, m)``-labelled models of
:meth:`ImpurityModel.from_shells`) and the shell keys the kernels group by.
"""

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.operator_algebra import arrayOp2Dict, c2i, combineOp, daggerOp
from impurityModel.ed.options import NIXSOptions, PESOptions, RIXSOptions, XASOptions, XPSOptions
from impurityModel.ed.rixs import calc_map, calc_tensor_map
from impurityModel.ed.spectra import (
    CARTESIAN,
    _pes_ips_equivalence_groups,
    _shell_windows,
    calc_spectra,
    calc_spectra_tensor,
)
from impurityModel.ed.symmetries import component_symmetry_reduction, extract_tensors
from impurityModel.ed.transition_operators import (
    daggered_dipole_operators,
    dipole_operators,
    inverse_photoemission_operators,
    nixs_operators,
    photoemission_operators,
)

__all__ = [
    "TECHNIQUES",
    "LabelledOperators",
    "ShellLayout",
    "SpectraTarget",
    "merge_spectra_datasets",
    "nixs_spectrum",
    "pes_spectrum",
    "rixs_spectrum",
    "xas_spectrum",
    "xps_spectrum",
]


class LabelledOperators:
    """Transition operators of an ``(l, s, m)``-labelled model, addressed through ``c2i(nBaths, ...)``.

    Parameters
    ----------
    nBaths : Mapping
        ``{l: n_bath}`` for every shell: fixes the ``c2i`` index layout.
    """

    def __init__(self, nBaths):
        self.nBaths = nBaths

    def photoemission(self, key):
        return photoemission_operators(self.nBaths, l=key)

    def inverse_photoemission(self, key):
        return inverse_photoemission_operators(self.nBaths, l=key)

    def dipole(self, vectors, core, valence, explicit=None):
        if explicit is not None:
            raise ValueError(
                "explicit dipole matrices are only supported for a block model (ImpurityModel.from_blocks / "
                "from_solver_matrix); an (l, s, m)-labelled model builds the atomic dipole itself."
            )
        return dipole_operators(self.nBaths, vectors, core, valence)

    def daggered_dipole(self, vectors, core, valence, explicit=None):
        if explicit is not None:
            raise ValueError("explicit dipole matrices need a block model; see LabelledOperators.dipole")
        return daggered_dipole_operators(self.nBaths, vectors, core, valence)

    def nixs(self, qs, l_final, l_initial, r_final, r_initial, radial_mesh):
        return nixs_operators(self.nBaths, qs, l_final, l_initial, r_final, r_initial, radial_mesh)

    def projector_operators(self, projectors):
        """The projector dicts, as operators on this model's index layout."""
        return arrayOp2Dict(self.nBaths, projectors.values())

    def combine(self, left, right):
        return combineOp(self.nBaths, left, right)

    def pes_equivalence_groups(self, key, block_structure):
        return _pes_ips_equivalence_groups(self.nBaths, key, block_structure)


class IndexedOperators:
    """Transition operators of a block model, addressed through flat orbital indices.

    The model carries no ``(l, s, m)`` labels: shells are given by the rows of the impurity block each
    one owns and by its column block of ``rot_to_spherical`` (``O_input = R O_spherical R^dagger``). The
    atomic dipole is built in the spherical basis and rotated, ``T_input = R_valence D R_core^dagger``;
    a caller-supplied ``(3, n_imp, n_imp)`` matrix (RSPt's ``Dxyz``, already in the input basis) replaces
    it. Photoemission operators are the annihilators of the shell's input-basis orbitals, in row order.

    Parameters
    ----------
    rows : Mapping
        ``{shell key: sorted impurity spin-orbital indices of that shell}``.
    rotations : Mapping
        ``{shell key: R_a}``, the ``(len(rows[a]), 2(2 l_a + 1))`` block of ``rot_to_spherical``.
    ls : Mapping
        ``{shell key: l}``, the angular momentum of each shell.
    """

    def __init__(self, rows, rotations, ls):
        self.rows = {k: list(v) for k, v in rows.items()}
        self.rotations = {k: np.asarray(v) for k, v in rotations.items()}
        self.ls = dict(ls)

    def photoemission(self, key):
        return [{((int(i), "a"),): 1} for i in self.rows[key]]

    def inverse_photoemission(self, key):
        return [{((int(i), "c"),): 1} for i in self.rows[key]]

    def _spherical_dipole(self, n, core, valence):
        l_core, l_valence = self.ls[core], self.ls[valence]
        local = OrderedDict([(l_core, 0), (l_valence, 0)])
        if l_core == l_valence:
            raise ValueError("a dipole transition needs two shells of different l")
        t = dipole_operators(local, [n], l_core, l_valence)[0]
        offset_valence = c2i(local, (l_valence, 0, -l_valence))
        offset_core = c2i(local, (l_core, 0, -l_core))
        d = np.zeros((2 * (2 * l_valence + 1), 2 * (2 * l_core + 1)), dtype=complex)
        for ((i, _), (j, _)), value in t.items():
            d[i - offset_valence, j - offset_core] = value
        return d

    def _matrix(self, n, core, valence):
        d = self._spherical_dipole(n, core, valence)
        return self.rotations[valence] @ d @ self.rotations[core].conj().T

    def _dict(self, matrix, core, valence):
        out = {}
        for a, i in enumerate(self.rows[valence]):
            for b, j in enumerate(self.rows[core]):
                if abs(matrix[a, b]) > 1e-14:
                    out[((int(i), "c"), (int(j), "a"))] = complex(matrix[a, b])
        return out

    def _explicit(self, n, explicit, core, valence):
        d = np.asarray(explicit)
        block = np.tensordot(np.asarray(n, dtype=complex), d, axes=(0, 0))
        return block[np.ix_(self.rows[valence], self.rows[core])]

    def dipole(self, vectors, core, valence, explicit=None):
        out = []
        for n in vectors:
            m = self._matrix(n, core, valence) if explicit is None else self._explicit(n, explicit, core, valence)
            out.append(self._dict(m, core, valence))
        return out

    def daggered_dipole(self, vectors, core, valence, explicit=None):
        return [daggerOp(t) for t in self.dipole(vectors, core, valence, explicit=explicit)]

    def nixs(self, *args, **kwargs):
        raise NotImplementedError(
            "NIXS needs the radial wave functions and an (l, s, m)-labelled model; it is not available on a "
            "block model (ImpurityModel.from_blocks)."
        )

    def projector_operators(self, projectors):
        raise NotImplementedError("projected spectra are not available on a block model")

    def combine(self, left, right):
        raise NotImplementedError("combine is only available on an (l, s, m)-labelled model")

    def pes_equivalence_groups(self, key, block_structure):
        """Symmetry-equivalence label of each shell orbital, from the shell's block structure.

        ``block_structure`` indexes the shell's orbitals locally (position in ``rows[key]``).
        """
        class_of_block = {}
        for cls_id, cls in enumerate(block_structure.identical_blocks):
            for b in cls:
                class_of_block[b] = cls_id
        label_of_local = {}
        for b, block in enumerate(block_structure.blocks):
            for position, local in enumerate(block):
                label_of_local[local] = (class_of_block.get(b, b), position)
        return [label_of_local[k] for k in range(len(self.rows[key]))]


@dataclass(frozen=True)
class ShellLayout:
    """Which shell is which, and how its transition operators are built.

    Attributes
    ----------
    valence_key, core_key : hashable
        The shell keys the kernels group by (``l`` for a labelled model). ``core_key`` is ``None`` for a
        model without a core shell, which then supports only PES and NIXS.
    correlated_block_structure : BlockStructure or None
        Impurity block structure of the valence shell; when given, degenerate PES operators are
        deduplicated.
    operators : LabelledOperators
        The operator provider.
    """

    valence_key: Any
    core_key: Any
    operators: Any
    correlated_block_structure: Any = None


@dataclass(frozen=True)
class SpectraTarget:
    """One ground-state member plus everything every technique needs.

    Attributes
    ----------
    hOp : ManyBodyOperator
        The full interacting Hamiltonian.
    psis, es : sequence
        The member's eigenstates and energies.
    tau : float
        ``k_B * T``.
    basis : object
        The member's many-body basis (carries the communicator).
    occ_cutoff, slater_weight_min : float
        Basis-expansion cutoffs.
    verbose : bool
        Forwarded to the kernels.
    gf_pt2_tol : float or None
        PT2 growth tolerance of the Green's-function kernel.
    layout : ShellLayout
    """

    hOp: ManyBodyOperator
    psis: Any
    es: Any
    tau: float
    basis: Any
    occ_cutoff: float
    slater_weight_min: float
    verbose: bool
    gf_pt2_tol: Optional[float]
    layout: ShellLayout

    @property
    def is_root(self):
        return self.basis.comm.rank == 0


def _prep_one_body(tOp_dicts):
    return [ManyBodyOperator(t) for t in tOp_dicts]


def _require_core(layout, name):
    if layout.core_key is None:
        raise ValueError(
            f"{name} needs a core hole, but no core shell was given (the layout has no core shell). "
            "Disable it, or build a model with a core shell."
        )


def pes_spectrum(target: SpectraTarget, options: PESOptions):
    """Valence photoemission + inverse photoemission: ``{"PS/spectra": (n_w, n_orb)}`` on the root rank."""
    layout, ops = target.layout, target.layout.operators
    l_valence, l_core = layout.valence_key, layout.core_key
    root = target.is_root
    datasets = {}
    if layout.correlated_block_structure is not None:
        correlated_groups = ops.pes_equivalence_groups(l_valence, layout.correlated_block_structure)
    else:
        correlated_groups = None
    t0 = time.perf_counter()
    w, delta = options.w, options.delta
    if root:
        print(f"Create l={l_valence} inverse photoemission and photoemission spectra...", flush=True)
    # Transition operators
    tOpsIPS = ops.inverse_photoemission(l_valence)
    tOpsPS = ops.photoemission(l_valence)
    if root:
        print("Inverse photoemission Green's function..", flush=True)
    assert isinstance(target.hOp, ManyBodyOperator)
    gsIPS = calc_spectra(
        target.hOp,
        [ManyBodyOperator(t) for t in tOpsIPS],
        target.psis,
        target.es,
        target.tau,
        w,
        target.basis,
        delta,
        target.slater_weight_min,
        target.verbose,
        target.occ_cutoff,
        **_shell_windows(l_valence, l_core, imp=(0, 1), val=(1, 0), con=(0, 1)),
        equivalence_groups=correlated_groups,
        unit_report_label=f"inverse photoemission, l={l_valence} (create)",
        gf_pt2_tol=target.gf_pt2_tol,
    )
    if root:
        print("Photoemission Green's function..", flush=True)
    gsPS = calc_spectra(
        target.hOp,
        [ManyBodyOperator(t) for t in tOpsPS],
        target.psis,
        target.es,
        target.tau,
        -w,
        target.basis,
        -delta,
        target.slater_weight_min,
        target.verbose,
        target.occ_cutoff,
        **_shell_windows(l_valence, l_core, imp=(1, 0), val=(1, 0), con=(0, 1)),
        equivalence_groups=correlated_groups,
        unit_report_label=f"photoemission, l={l_valence} (annihilate)",
        gf_pt2_tol=target.gf_pt2_tol,
    )
    gsPS *= -1
    gs = gsPS + gsIPS
    if root:
        print("#spin orbitals = {:d}".format(np.shape(gs)[1]))
        print("#mesh points = {:d}".format(np.shape(gs)[0]))
        datasets["PS/spectra"] = gs
        print("time(PS) = {:.2f} seconds \n".format(time.perf_counter() - t0), flush=True)
    return datasets


def xps_spectrum(target: SpectraTarget, options: XPSOptions):
    """Core-level photoemission: ``{"XPS/spectra": (n_w, n_orb)}`` on the root rank."""
    layout, ops = target.layout, target.layout.operators
    _require_core(layout, "XPS")
    l_valence, l_core = layout.valence_key, layout.core_key
    root = target.is_root
    datasets = {}
    t0 = time.perf_counter()
    w, delta = options.w, options.delta
    if root:
        print(f"Create core l={l_core} x-ray photoemission spectra (XPS) ...")
    # Transition operators
    tOpsPS = ops.photoemission(l_core)
    # Photoemission Green's function
    gs = calc_spectra(
        target.hOp,
        [ManyBodyOperator(t) for t in tOpsPS],
        target.psis,
        target.es,
        target.tau,
        -w,
        target.basis,
        -delta,
        target.slater_weight_min,
        target.verbose,
        target.occ_cutoff,
        **_shell_windows(l_valence, l_core, imp=(1, 1), val=(1, 0), con=(0, 1), core_imp=(1, 0)),
        unit_report_label=f"x-ray photoemission, core l={l_core} (annihilate)",
        gf_pt2_tol=target.gf_pt2_tol,
    )
    gs *= -1
    if root:
        print("#spin orbitals = {:d}".format(np.shape(gs)[1]))
        print("#mesh points = {:d}".format(np.shape(gs)[0]))
        datasets["XPS/spectra"] = gs
        print("time(XPS) = {:.2f} seconds \n".format(time.perf_counter() - t0), flush=True)
    return datasets


def nixs_spectrum(target: SpectraTarget, options: NIXSOptions):
    """Non-resonant inelastic x-ray scattering: ``{"NIXS/spectra": (n_w, n_q)}`` on the root rank."""
    ops = target.layout.operators
    root = target.is_root
    datasets = {}
    t0 = time.perf_counter()
    radial_mesh, R_final, R_initial = options.radial
    li, lj = options.l_final, options.l_initial
    if root:
        print("Create NIXS spectra...", flush=True)
    # Transition operator: exp(iq*r)
    tOps = ops.nixs(options.resolved_qs, li, lj, R_final, R_initial, radial_mesh)
    # Green's function
    gs = calc_spectra(
        target.hOp,
        _prep_one_body(tOps),
        target.psis,
        target.es,
        target.tau,
        options.w_loss,
        target.basis,
        options.delta,
        target.slater_weight_min,
        target.verbose,
        target.occ_cutoff,
        dN_imp={li: (1, 1), lj: (1, 1)},
        dN_val={li: (1, 0), lj: (1, 0)},
        dN_con={li: (0, 1), lj: (0, 1)},
        unit_report_label="NIXS",
        gf_pt2_tol=target.gf_pt2_tol,
    )
    if root:
        print("#q-points = {:d}".format(np.shape(gs)[1]))
        print("#mesh points = {:d}".format(np.shape(gs)[0]))
        datasets["NIXS/spectra"] = gs
        print("time(NIXS) = {:.2f} seconds \n".format(time.perf_counter() - t0), flush=True)
    return datasets


def xas_spectrum(target: SpectraTarget, options: XASOptions):
    """X-ray absorption: ``{"XAS/tensor": (n_w, 3, 3)}`` or ``{"XAS/projected": ...}`` on the root rank."""
    layout, ops = target.layout, target.layout.operators
    _require_core(layout, "XAS")
    l_valence, l_core = layout.valence_key, layout.core_key
    root = target.is_root
    datasets = {}
    t0 = time.perf_counter()
    w, delta = options.w, options.delta
    if root:
        print("Create XAS spectra...", flush=True)
    dN_XAS = _shell_windows(l_valence, l_core, imp=(0, 1), val=(1, 0), con=(0, 1), core_imp=(1, 0))
    if options.projectors:
        # Projected operators are not a plain Cartesian linear combination -> keep the
        # per-operator path.
        epsilons = options.epsilons if options.epsilons is not None else CARTESIAN
        tOps = ops.dipole(epsilons, l_core, l_valence, explicit=options.dipole)
        iBasisProjectors = ops.projector_operators(options.projectors)
        projectedTOps = []
        for proj in iBasisProjectors:
            for op in tOps:
                projectedTOps.append(ops.combine(proj, op))
        gs = calc_spectra(
            target.hOp,
            _prep_one_body(projectedTOps),
            target.psis,
            target.es,
            target.tau,
            w,
            target.basis,
            delta,
            target.slater_weight_min,
            target.verbose,
            target.occ_cutoff,
            **dN_XAS,
            unit_report_label=f"XAS, core l={l_core} (projected operators)",
            gf_pt2_tol=target.gf_pt2_tol,
        )
        if root:
            print("#projected operators = {:d}".format(np.shape(gs)[1]))
            print("#mesh points = {:d}".format(np.shape(gs)[0]))
            datasets["XAS/projected"] = gs
    else:
        # Dipole is linear in the polarization: compute and store the spectral tensor over the
        # 3 Cartesian components once (symmetry-reduced). Polarization contraction (arbitrary /
        # circular, dichroism, ...) is left to post-processing (impurityModel.ed.polarization),
        # so it never requires re-running the solve.
        cartesian_ops = _prep_one_body(ops.dipole(CARTESIAN, l_core, l_valence, explicit=options.dipole))
        n_orb = target.basis.num_spin_orbitals
        h_onebody = extract_tensors(target.hOp, n_orb=n_orb, two_body=False)[0]
        reduction = component_symmetry_reduction(cartesian_ops, h_onebody, n_orb=n_orb)
        chi = calc_spectra_tensor(
            target.hOp,
            cartesian_ops,
            target.psis,
            target.es,
            target.tau,
            w,
            target.basis,
            delta,
            target.slater_weight_min,
            target.verbose,
            target.occ_cutoff,
            reduction=reduction,
            **dN_XAS,
            unit_report_label=f"XAS, core l={l_core} (Cartesian tensor)",
            gf_pt2_tol=target.gf_pt2_tol,
        )
        if root:
            print("#Cartesian components = {:d}".format(chi.shape[1]))
            print("#mesh points = {:d}".format(chi.shape[0]))
            datasets["XAS/tensor"] = chi
    if root:
        print("time(XAS) = {:.2f} seconds \n".format(time.perf_counter() - t0), flush=True)
    return datasets


def rixs_spectrum(target: SpectraTarget, options: RIXSOptions):
    """Resonant inelastic x-ray scattering: ``{"RIXS/tensor": ...}`` or ``{"RIXS/projected": ...}``."""
    layout, ops = target.layout, target.layout.operators
    _require_core(layout, "RIXS")
    l_valence, l_core = layout.valence_key, layout.core_key
    root = target.is_root
    datasets = {}
    t0 = time.perf_counter()
    wIn, wLoss = options.w_in, options.w_loss
    delta, deltaRIXS = options.delta_intermediate, options.delta_final
    if root:
        print("Create RIXS spectra...", flush=True)

    if options.projectors:
        # Projected operators are not a plain Cartesian linear combination -> keep the
        # per-operator Kramers-Heisenberg path.
        epsilons_in = options.epsilons_in if options.epsilons_in is not None else CARTESIAN
        epsilons_out = options.epsilons_out if options.epsilons_out is not None else CARTESIAN
        tOpsIn = ops.dipole(epsilons_in, l_core, l_valence, explicit=options.dipole)
        tOpsOut = ops.daggered_dipole(epsilons_out, l_core, l_valence, explicit=options.dipole)
        iBasisProjectors = ops.projector_operators(options.projectors)
        projectedTOpsIn = []
        projectedTOpsOut = []
        for proj in iBasisProjectors:
            for opIn in tOpsIn:
                projectedTOpsIn.append(ops.combine(proj, opIn))
            for opOut in tOpsOut:
                projectedTOpsOut.append(ops.combine(opOut, proj))
        gs = calc_map(
            target.hOp,
            _prep_one_body(projectedTOpsIn),
            _prep_one_body(projectedTOpsOut),
            target.psis,
            target.es,
            target.tau,
            wIn,
            wLoss,
            delta,
            deltaRIXS,
            target.basis,
            target.verbose,
            slaterWeightMin=target.slater_weight_min,
            l_core=l_core,
            l_valence=l_valence,
            occ_cutoff=target.occ_cutoff,
        )
        if root:
            print("RIXS projectors = {}".format(options.projectors.keys()))
            print(f"shape(gs) = {np.shape(gs)}")
            datasets["RIXS/projected"] = gs
            for key, proj in options.projectors.items():
                datasets[f"RIXSprojectors/{key}"] = str(proj)
    else:
        # Dipole is linear in the polarization: compute and store the full rank-4
        # Kramers-Heisenberg tensor over the 3 Cartesian in/out components once (R4 -- the
        # RIXS analogue of B2b). Polarization contraction (arbitrary/circular, dichroism,
        # ...) is left to post-processing (impurityModel.ed.polarization).
        in_component_ops = _prep_one_body(ops.dipole(CARTESIAN, l_core, l_valence, explicit=options.dipole))
        out_component_ops = _prep_one_body(ops.daggered_dipole(CARTESIAN, l_core, l_valence, explicit=options.dipole))
        C = calc_tensor_map(
            target.hOp,
            in_component_ops,
            out_component_ops,
            target.psis,
            target.es,
            target.tau,
            wIn,
            wLoss,
            delta,
            deltaRIXS,
            target.basis,
            target.verbose,
            slaterWeightMin=target.slater_weight_min,
            l_core=l_core,
            l_valence=l_valence,
            occ_cutoff=target.occ_cutoff,
            adaptive_wIn_tol=options.adaptive_tol,
        )
        if root:
            print(f"shape(C) = {np.shape(C)}")
            print("#in-components = {:d}".format(C.shape[0]))
            print("#out-components = {:d}".format(C.shape[1]))
            print("#mesh points of input energy = {:d}".format(C.shape[4]))
            print("#mesh points of energy loss = {:d}".format(C.shape[5]))
            # complex64: ~1e-7 relative precision, below the R1 solve tolerance (1e-6);
            # halves the storage of the default-mesh tensor (~260 -> ~130 MB).
            datasets["RIXS/tensor"] = C.astype(np.complex64)

    if root:
        print("time(RIXS) = {:.2f} seconds \n".format(time.perf_counter() - t0), flush=True)
    return datasets


#: Execution order (the order the old monolith ran them in): ``name -> (options class, function)``.
TECHNIQUES = {
    "pes": (PESOptions, pes_spectrum),
    "xps": (XPSOptions, xps_spectrum),
    "nixs": (NIXSOptions, nixs_spectrum),
    "xas": (XASOptions, xas_spectrum),
    "rixs": (RIXSOptions, rixs_spectrum),
}


def technique_of(options):
    """The :data:`TECHNIQUES` name of an options object; raises ``TypeError`` for anything else."""
    for name, (cls, _function) in TECHNIQUES.items():
        if type(options) is cls:
            return name
    expected = [cls.__name__ for cls, _function in TECHNIQUES.values()]
    raise TypeError(f"{type(options).__name__} is not a spectrum request (expected one of {expected})")


def ordered_requests(requests, layout=None):
    """``requests`` as ``[(name, options)]`` in execution order, validated.

    Replicated input, so every rank raises the same error before any collective is entered: an
    object that is not a spectrum request, a technique asked for twice, or (given ``layout``) a
    core-level technique on a model without a core shell.
    """
    by_name = {}
    for request in requests:
        name = technique_of(request)
        if name in by_name:
            raise ValueError(f"more than one {name.upper()} request; give each technique at most once")
        by_name[name] = request
    if layout is not None and layout.core_key is None and {"xps", "xas", "rixs"} & set(by_name):
        raise ValueError(
            "XPS, XAS and RIXS all need a core hole, but no core shell was given "
            "(the layout has no core shell). Disable them, or build a model with a core shell."
        )
    return [(name, by_name[name]) for name in TECHNIQUES if name in by_name]


def run_techniques(target, requests):
    """Run ``requests`` (one options object per technique) on ``target`` in execution order; merge the datasets."""
    datasets = {}
    for name, request in ordered_requests(requests, target.layout):
        datasets.update(TECHNIQUES[name][1](target, request))
    return datasets


def merge_spectra_datasets(parts, weights):
    r"""Merge per-member :func:`run_techniques` outputs into the charge-sector ensemble's spectra.

    Every spectrum (PS, XPS, NIXS, XAS and RIXS, projected or tensor) is a thermal average
    :math:`\sum_i p_i \langle \psi_i | \dots | \psi_i \rangle` that :func:`run_techniques` normalises
    within the member it is given, so it is linear in the members and the ensemble's spectrum is
    :math:`\sum_s c_s S_s`. Excitations never connect two members (the transition operators act on a
    determinant of one member and the resolvent stays inside the excited sector), so there are no cross terms.
    Non-numeric entries (the RIXS projector descriptions) are taken from the first member.

    Parameters
    ----------
    parts : list of dict
        One :func:`run_techniques` return per member (empty dicts on non-root ranks).
    weights : sequence of float
        The members' ensemble weights ``c_s`` (renormalised here).

    Returns
    -------
    dict
        Dataset name -> merged array, with the dtype of the first member's array (RIXS tensors stay complex64).
    """
    c = np.asarray(weights, dtype=float)
    c = c / c.sum()
    merged = {}
    for name, first in parts[0].items():
        if isinstance(first, str):
            merged[name] = first
            continue
        total = sum(ci * np.asarray(part[name], dtype=complex) for ci, part in zip(c, parts))
        merged[name] = total.astype(first.dtype, copy=False) if np.iscomplexobj(first) else total.real
    return merged
