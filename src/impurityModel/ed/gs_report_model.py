"""The ground-state report's data model: what every spin-orbital index *is*.

Every number the report prints rests on one map, global spin-orbital index -> (impurity or bath, group, shell,
spin, block, equivalence class). :class:`OrbitalNaming` builds it once from the solver basis, the spin layout and
the block structure, validates it, and is the single place a renderer reads a label from.
"""

from dataclasses import asdict, dataclass, field, is_dataclass

import numpy as np

from impurityModel.ed.block_structure import blocks_as_positions, get_equivalent_blocks
from impurityModel.ed.degeneracy import MANIFOLD_TOL, degenerate_groups
from impurityModel.ed.gs_statistics import _boltzmann_weights
from impurityModel.ed.observables import block_group_labels

#: Hopping amplitudes below this do not count as a coupling between an impurity orbital and a bath orbital.
COUPLING_TOL = 1e-8


@dataclass(frozen=True)
class OrbitalInfo:
    """One spin-orbital.

    ``role`` is ``"impurity"``, ``"valence"`` or ``"conduction"``; ``group_key`` the ``impurity_orbitals`` key it
    belongs to and ``group_label`` its reader-facing name (``"group 0"``, or ``"group 1.a"`` when the block
    structure splits the group). ``shell`` indexes ``shell_ls`` (impurity, when given), ``spin`` is ``"dn"`` /
    ``"up"`` from the layout pairs (``None`` when unpaired), ``partner`` the other member of the pair. ``block``
    and ``equivalence_class`` are positions in the block structure (impurity only). ``one_body_level`` is
    ``h1[i, i]`` and ``coupled_to`` the impurity orbitals this bath orbital hops to (``h1`` given only).
    """

    index: int
    role: str
    group_key: object
    group_label: str
    shell: object = None
    spin: object = None
    partner: object = None
    block: object = None
    equivalence_class: object = None
    one_body_level: object = None
    coupled_to: tuple = ()


class OrbitalNaming:
    """Validated map from every spin-orbital index to an :class:`OrbitalInfo`.

    Build with :meth:`build`; a violated invariant raises ``ValueError``. All inputs are replicated, so the
    raise is rank-uniform. ``block_structure`` is stored with its blocks as positions into the sorted impurity
    list (:func:`~impurityModel.ed.block_structure.blocks_as_positions`).
    """

    def __init__(self, infos, impurity_indices, block_structure, class_labels):
        self.infos = dict(infos)
        self.impurity_indices = list(impurity_indices)
        self.block_structure = block_structure
        self.class_labels = list(class_labels)

    def __getitem__(self, index):
        return self.infos[int(index)]

    def __len__(self):
        return len(self.infos)

    def indices(self, role=None, group_key=None):
        """Sorted indices with the given role and/or group."""
        return sorted(
            i
            for i, o in self.infos.items()
            if (role is None or o.role == role) and (group_key is None or o.group_key == group_key)
        )

    @classmethod
    def build(
        cls,
        impurity_orbitals,
        bath_states,
        block_structure,
        layout=None,
        h1=None,
        rot_to_spherical=None,
        shell_ls=None,
    ):
        """Assemble and validate the table.

        Parameters
        ----------
        impurity_orbitals : dict
            ``Basis.impurity_orbitals``: group key -> list of blocks of spin-orbital indices.
        bath_states : tuple of dict
            ``Basis.bath_states``: ``(valence, conduction)``, each group key -> list of blocks.
        block_structure : BlockStructure
            Blocks as positions into the sorted impurity list, or as the impurity indices themselves.
        layout : SpinLayout, optional
            Supplies the ``(dn, up)`` pairs.
        h1 : ndarray, optional
            One-body matrix over all spin-orbitals.
        rot_to_spherical : ndarray or dict, optional
            Only its row count is checked (against the impurity size) when it is an ndarray.
        shell_ls : sequence of int, optional
            ``l`` of each column block of ``rot_to_spherical``; must tile the impurity in blocks of ``2(2l+1)``.
        """
        imp_by_group = {k: [list(map(int, b)) for b in blocks] for k, blocks in impurity_orbitals.items()}
        impurity = sorted(o for blocks in imp_by_group.values() for b in blocks for o in b)
        if len(impurity) != len(set(impurity)):
            raise ValueError("orbital naming: an impurity spin-orbital appears in more than one group")
        role_of = {o: "impurity" for o in impurity}
        group_of = {o: k for k, blocks in imp_by_group.items() for b in blocks for o in b}
        valence, conduction = bath_states
        for role, states in (("valence", valence), ("conduction", conduction)):
            for key, blocks in states.items():
                if key in imp_by_group and len(blocks) != len(imp_by_group[key]):
                    raise ValueError(
                        f"orbital naming: group {key!r} has {len(imp_by_group[key])} impurity blocks "
                        f"but {len(blocks)} {role} bath blocks"
                    )
                for b in blocks:
                    for o in map(int, b):
                        if o in role_of:
                            raise ValueError(f"orbital naming: spin-orbital {o} is both {role_of[o]} and {role}")
                        role_of[o] = role
                        group_of[o] = key
        n_imp = len(impurity)
        if isinstance(rot_to_spherical, np.ndarray) and rot_to_spherical.shape[0] != n_imp:
            raise ValueError(
                f"orbital naming: rot_to_spherical has {rot_to_spherical.shape[0]} rows, the impurity {n_imp} orbitals"
            )
        shell_of = {}
        if shell_ls is not None:
            if sum(2 * (2 * int(l) + 1) for l in shell_ls) != n_imp:
                raise ValueError(
                    f"orbital naming: shells l={list(shell_ls)} do not tile the {n_imp} impurity spin-orbitals "
                    "(each shell is 2(2l+1) of them)"
                )
            start = 0
            for s, l in enumerate(shell_ls):
                size = 2 * (2 * int(l) + 1)
                for o in impurity[start : start + size]:
                    shell_of[o] = s
                start += size
        bs = blocks_as_positions(block_structure, impurity)
        equivalent = get_equivalent_blocks(bs)
        seen = sorted(b for blocks in equivalent for b in blocks)
        if seen != list(range(len(bs.blocks))):
            raise ValueError("orbital naming: the block equivalence classes do not partition the blocks")
        class_labels, _ = block_group_labels(equivalent, bs, impurity_orbitals)
        block_of, class_of = {}, {}
        for c, blocks in enumerate(equivalent):
            for b in blocks:
                for pos in bs.blocks[b]:
                    block_of[impurity[pos]] = b
                    class_of[impurity[pos]] = c
        spin_of, partner_of = {}, {}
        if layout is not None and layout.labelled:
            for dn, up in layout.pairs:
                for o in (dn, up):
                    if o not in role_of:
                        raise ValueError(f"orbital naming: spin pair ({dn}, {up}) names the unknown spin-orbital {o}")
                spin_of[dn], spin_of[up] = "dn", "up"
                partner_of[dn], partner_of[up] = up, dn
        infos = {}
        for o, role in role_of.items():
            label = class_labels[class_of[o]] if role == "impurity" else f"group {group_of[o]}"
            level = coupled = None
            if h1 is not None:
                level = float(np.real(h1[o, o]))
                coupled = (
                    ()
                    if role == "impurity"
                    else tuple(i for i in impurity if abs(h1[i, o]) > COUPLING_TOL or abs(h1[o, i]) > COUPLING_TOL)
                )
            infos[o] = OrbitalInfo(
                index=o,
                role=role,
                group_key=group_of[o],
                group_label=label,
                shell=shell_of.get(o),
                spin=spin_of.get(o),
                partner=partner_of.get(o),
                block=block_of.get(o),
                equivalence_class=class_of.get(o),
                one_body_level=level,
                coupled_to=coupled or (),
            )
        return cls(infos, impurity, bs, class_labels)


#: Version of the ``report`` block of the JSON (the statistics keep their own ``schema_version``).
REPORT_SCHEMA_VERSION = 4

#: Per-manifold spread of a label below this makes it sharp (it is a Casimir quantum number, rounded already).
LABEL_SPREAD_TOL = 1e-6


@dataclass
class ManifoldRow:
    """One degenerate manifold: a unit of the spectrum.

    ``energy`` is the manifold mean and ``spread`` its width. ``weight`` is the Boltzmann weight of the whole
    manifold, ``states`` the indices of its members in the report's ascending-energy state order. ``two_sz``,
    ``s``, ``l``, ``j`` are per member (``None`` where not measured); ``*_sharp`` says the label is the same for
    every member, which a rotated manifold has for the labels that were diagonalised.
    """

    index: int
    energy: float
    spread: float
    g: int
    weight: float
    states: list
    two_sz: object = None
    s: object = None
    l: object = None
    j: object = None
    s_sharp: object = None
    l_sharp: object = None
    j_sharp: object = None
    sectors: object = None


@dataclass
class MemberRow:
    """One state, in the (possibly rotated) basis the report measured it in."""

    state: int
    manifold: int
    e_minus_e0: float
    weight: float
    two_sz: object = None
    n_imp_sector: object = None
    s: object = None
    l: object = None
    j: object = None
    sisb: object = None
    sisb_z: object = None


@dataclass
class GroundStateReport:
    """Everything the ground-state report says, as data. Renderers and the JSON read this and nothing else."""

    label: object
    e0: float
    tau: float
    n_states: int
    n_determinants: int
    degeneracy_tol: float
    spin_layout: object
    naming: object
    manifolds: list
    members: list
    thermal: dict
    magnetic: object = None
    correlation: object = None
    screening: object = None
    energy_decomposition: object = None
    excitation_bounds: list = field(default_factory=list)
    convergence: object = None
    ensemble: object = None
    warnings: list = field(default_factory=list)
    skips: list = field(default_factory=list)
    schema_version: int = REPORT_SCHEMA_VERSION

    def to_json(self):
        """JSON-ready dict (numpy values left to the encoder of :func:`gs_statistics.save_gs_statistics`)."""
        out = asdict(self)
        out["naming"] = (
            None
            if self.naming is None
            else {str(i): asdict(info) for i, info in sorted(self.naming.infos.items())}
        )
        return out


def _column(values, order):
    return None if values is None else [float(x) for x in np.asarray(values, dtype=float)[order]]


def assemble_report(
    es,
    tau,
    n_determinants,
    degeneracy_tol=MANIFOLD_TOL,
    naming=None,
    layout=None,
    two_sz=None,
    sectors=None,
    s_values=None,
    l_values=None,
    j_values=None,
    sisb_values=None,
    sisb_z_values=None,
    thermal=None,
    magnetic=None,
    correlation=None,
    screening=None,
    energy_decomposition=None,
    excitation_bounds=None,
    convergence=None,
    ensemble=None,
    warnings=None,
    skips=None,
    label=None,
):
    """Pure assembly of a :class:`GroundStateReport` from replicated per-state arrays. No collectives.

    ``es`` are the states' energies in the order every per-state array uses. States are re-ordered by
    ascending energy (stable), and manifolds are :func:`~impurityModel.ed.degeneracy.degenerate_groups` of
    that order at ``degeneracy_tol``.
    """
    es = np.asarray(es, dtype=float)
    order = np.argsort(es, kind="stable")
    e_sorted = es[order]
    w = _boltzmann_weights(es, tau)[order]
    e0 = float(e_sorted[0]) if len(es) else 0.0
    blocks = degenerate_groups(e_sorted, degeneracy_tol, anchor="first") if len(es) else []
    two = None if two_sz is None else np.asarray(two_sz)[order]
    sec = None if sectors is None else np.asarray(sectors, dtype=object)[order]
    cols = {
        "s": _column(s_values, order),
        "l": _column(l_values, order),
        "j": _column(j_values, order),
    }

    def sharp(col, idx):
        return None if col is None else bool(np.ptp(np.asarray(col)[idx]) < LABEL_SPREAD_TOL)

    manifolds, members = [], []
    for m, blk in enumerate(blocks):
        idx = np.asarray(blk, dtype=int)
        manifolds.append(
            ManifoldRow(
                index=m,
                energy=float(np.mean(e_sorted[idx])),
                spread=float(np.ptp(e_sorted[idx])),
                g=len(idx),
                weight=float(np.sum(w[idx])),
                states=[int(i) for i in idx],
                two_sz=None if two is None else [int(x) for x in two[idx]],
                s=None if cols["s"] is None else [cols["s"][i] for i in idx],
                l=None if cols["l"] is None else [cols["l"][i] for i in idx],
                j=None if cols["j"] is None else [cols["j"][i] for i in idx],
                s_sharp=sharp(cols["s"], idx),
                l_sharp=sharp(cols["l"], idx),
                j_sharp=sharp(cols["j"], idx),
                sectors=None if sec is None else sorted({int(x) for x in sec[idx] if x is not None}),
            )
        )
        sisb = _column(sisb_values, order)
        sisb_z = _column(sisb_z_values, order)
        for i in idx:
            members.append(
                MemberRow(
                    state=int(i),
                    manifold=m,
                    e_minus_e0=float(e_sorted[i] - e0),
                    weight=float(w[i]),
                    two_sz=None if two is None else int(two[i]),
                    n_imp_sector=None if sec is None or sec[i] is None else int(sec[i]),
                    s=None if cols["s"] is None else cols["s"][i],
                    l=None if cols["l"] is None else cols["l"][i],
                    j=None if cols["j"] is None else cols["j"][i],
                    sisb=None if sisb is None else sisb[i],
                    sisb_z=None if sisb_z is None else sisb_z[i],
                )
            )
    return GroundStateReport(
        label=label,
        e0=e0,
        tau=float(tau),
        n_states=len(es),
        n_determinants=int(n_determinants),
        degeneracy_tol=float(degeneracy_tol),
        spin_layout=None if layout is None else asdict(layout) if is_dataclass(layout) else dict(layout),
        naming=naming,
        manifolds=manifolds,
        members=members,
        thermal=dict(thermal or {}),
        magnetic=magnetic,
        correlation=correlation,
        screening=screening,
        energy_decomposition=energy_decomposition,
        excitation_bounds=list(excitation_bounds or []),
        convergence=convergence,
        ensemble=ensemble,
        warnings=list(warnings or []),
        skips=list(skips or []),
    )
