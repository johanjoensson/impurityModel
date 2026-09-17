"""The per-unit maximum-basis-size report a completed Green's-function calculation prints.

A finished calculation states, for every work unit -- one ``(orbital block, create/annihilate)``
pair -- the largest basis that unit's recurrence ran on. That is the number the next run's
``truncation_threshold`` has to accommodate, so it is printed unconditionally rather than at
``-vv``.

The defect these tests exist to pin: the sparse block-Lanczos recurrence's support is tracked
**only** by :class:`gf_primitives._CappedBasisProxy`, which ``block_Green_sparse`` installs only
under a finite cap. ``_block_green_group``'s ``excited_basis`` is the clone of the *seed* support
and the matvec never adds to it, so falling back to ``len(excited_basis)`` on the uncapped path
reports the seed size while claiming to report the Krylov support -- measured 1 vs 15
determinants on the fixture below, identical physics. An absent number invites a question; a
plausible wrong one ends the investigation, so the uncapped path must say it does not know.
"""

import numpy as np

from impurityModel.ed.greens_function import _report_max_unit_basis, _unit_basis_rows, get_Greens_function
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant

# Six impurity orbitals, nearest-neighbour hopping: the Krylov space provably leaves the
# single-determinant seed support, which is what makes the seed-vs-support distinction visible.
_TERMS = {((i, "c"), (i, "a")): 0.1 * (i + 1) for i in range(6)}
for _i in range(5):
    _TERMS[((_i, "c"), (_i + 1, "a"))] = 0.4
    _TERMS[((_i + 1, "c"), (_i, "a"))] = 0.4
HOP = ManyBodyOperator(_TERMS)
GROUND = b"\xe0"  # orbitals 0, 1, 2 occupied (MSB-first: orbital i is bit 7 - i)
# Blocks 0 and 1 are occupied, so their *addition* seeds vanish and only the removal side
# reports -- deliberate: it also covers a work unit that contributes no row.
BLOCKS = [[0], [1]]
NON_BINDING_CAP = 100_000  # far above anything this fixture can reach, so it never freezes


def _basis(cap):
    return Basis(
        impurity_orbitals={0: [[0, 1, 2, 3, 4, 5]]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=[GROUND],
        comm=None,
        truncation_threshold=cap,
    )


def _run(cap, gf_method="lanczos"):
    get_Greens_function(
        matsubara_mesh=None,
        omega_mesh=np.linspace(-3.0, 3.0, 21),
        psis=[ManyBodyState({SlaterDeterminant.from_bytes(GROUND): 1.0})],
        es=[0.0],
        tau=1.0,
        basis=_basis(cap),
        hOp=HOP,
        delta=0.1,
        blocks=BLOCKS,
        verbose=False,
        verbose_extra=False,
        reort=None,
        dN=3,
        occ_cutoff=1e-9,
        slaterWeightMin=0.0,
        sparse=True,
        gf_method=gf_method,
    )


def _report_lines(out):
    """The indented rows of the single report in ``out`` (heading and summary excluded)."""
    lines = out.splitlines()
    heads = [i for i, line in enumerate(lines) if line.startswith("Maximum ") and line.endswith(":")]
    assert len(heads) == 1, f"expected exactly one per-unit basis report, got {len(heads)}:\n{out}"
    rows, summary = [], None
    for line in lines[heads[0] + 1 :]:
        if not line.startswith("  "):
            break
        if line.strip().startswith("maximum over"):
            summary = line.strip()
            break
        rows.append(line.strip())
    return rows, summary


def _sizes(rows):
    """``determinants`` counts off the reported rows, keyed by the row's label text."""
    out = {}
    for row in rows:
        assert "not tracked" not in row, f"row reports no size: {row}"
        label, _, count = row.rpartition("  ")
        out[label.strip()] = int(count.split()[0].replace(",", ""))
    return out


def _seed_support_size():
    """Determinants in the removal seed ``c_0|psi>`` -- what the buggy fallback reported."""
    psi = ManyBodyState({SlaterDeterminant.from_bytes(GROUND): 1.0})
    seed = ManyBodyOperator({((0, "a"),): 1.0}).apply_block(ManyBodyState.from_states([psi]), 0.0)
    return len(seed.to_states()[0])


def test_uncapped_sparse_path_reports_no_size_rather_than_the_seed_size(capsys):
    """With no determinant cap nothing tracks the support the matvec discovers, so the rows for
    the units that actually ran a recurrence must say so. This is the regression: the fallback
    printed ``len(excited_basis)`` here, which is the seed support -- a plausible number, an
    order of magnitude low, that would send someone sizing ``truncation_threshold`` into an OOM.
    """
    _run(np.inf)
    rows, summary = _report_lines(capsys.readouterr().out)
    ran = [row for row in rows if "annihilate" in row]
    assert ran, rows
    assert all("not tracked (no determinant cap set)" in row for row in ran), ran
    # Nothing that could be mistaken for a measured size appears on those rows.
    assert not any(char.isdigit() for row in ran for char in row.rpartition("  ")[2])
    # The empty addition units are a different thing and still report truthfully.
    assert all(row.endswith("0 determinants") for row in rows if "create" in row), rows
    assert summary is not None and "2 of 4 units not tracked and not counted here" in summary
    assert summary.startswith("maximum over the tracked units:")


def test_a_non_binding_cap_reports_the_krylov_support_not_the_seed(capsys):
    """The tracked number must exceed the seed support -- the property the seed-size fallback
    violated by construction (it reported exactly the seed size). The cap is far above anything
    this fixture reaches, so it only switches tracking on; it never binds."""
    _run(NON_BINDING_CAP)
    rows, summary = _report_lines(capsys.readouterr().out)
    sizes = _sizes(rows)
    seed = _seed_support_size()
    assert seed >= 1
    ran = {label: size for label, size in sizes.items() if "annihilate" in label}
    assert ran, sizes
    assert all(size > seed for size in ran.values()), (ran, seed)
    assert all(size < NON_BINDING_CAP for size in ran.values()), ran
    assert summary is not None and "lower bound" not in summary
    assert int(summary.split(":")[1].split()[0].replace(",", "")) == max(sizes.values())


def test_a_binding_cap_is_marked_so_the_number_is_not_mistaken_for_a_demand(capsys):
    """When a unit freezes at the cap the number reported *is* the cap, and says nothing about
    how large the unit wanted to be. Both the row and the summary must say so, or a reader
    raises the cap, reads it back, and loops."""
    _run(4)
    rows, summary = _report_lines(capsys.readouterr().out)
    assert any("frozen at the cap" in row for row in rows), rows
    assert all(size <= 4 for size in _sizes(rows).values())
    assert summary is not None and "lower bound" in summary


def test_the_two_drivers_report_under_different_headings(capsys):
    """The block-Lanczos excited basis and the per-frequency rebuilt solve basis are different
    constructions. Reporting both as "excited basis size" would invite comparing them."""
    _run(NON_BINDING_CAP, gf_method="lanczos")
    lanczos = capsys.readouterr().out
    _run(NON_BINDING_CAP, gf_method="bicgstab")
    bicgstab = capsys.readouterr().out
    assert "Maximum excited basis size per unit:" in lanczos
    assert "Maximum per-frequency solve basis size per unit:" in bicgstab
    assert "Maximum excited basis size per unit:" not in bicgstab


def test_a_unit_with_no_transition_weight_reports_zero_rather_than_nothing(capsys):
    """Blocks 0 and 1 are occupied, so ``c_dagger`` annihilates the state and the addition units
    have empty seeds. They ran no recurrence, which is a measured 0 -- distinct from "not
    tracked", and worth a row: a silently missing side reads as a reporting gap."""
    _run(NON_BINDING_CAP)
    rows, _ = _report_lines(capsys.readouterr().out)
    assert len(rows) == 2 * len(BLOCKS)
    assert [row.endswith("0 determinants") for row in rows] == [True, False, True, False], rows


def test_a_block_with_no_units_at_all_contributes_no_row_and_no_indentation():
    """The row builder pads to the units that actually reported, so one huge unreported block
    cannot over-indent the rest."""
    rows = _unit_basis_rows([[0], [1, 2, 3, 4, 5]], {(0, 1): (7, False)})
    assert len(rows) == 1
    assert rows[0][0] == "block [0]  annihilate (removal)"


# --- the printer itself, independent of any driver ------------------------------------


def test_printer_formats_sizes_untracked_rows_and_the_maximum(capsys):
    _report_max_unit_basis("Maximum excited basis size per unit", [("a", 3456, False), ("b", 1, False)])
    out = capsys.readouterr().out
    assert "a  3,456 determinants" in out
    assert "b      1 determinant\n" in out  # singular, right-aligned to the widest count
    assert "maximum over all units: 3,456 determinants" in out
    assert "lower bound" not in out


def test_printer_reports_a_maximum_only_over_tracked_rows(capsys):
    _report_max_unit_basis("Maximum excited basis size per unit", [("a", None, False), ("b", 12, False)])
    out = capsys.readouterr().out
    assert "a  not tracked (no determinant cap set)" in out
    assert "maximum over the tracked units: 12 determinants (1 of 2 units not tracked" in out

    _report_max_unit_basis("Maximum excited basis size per unit", [("a", None, False)])
    assert "maximum over" not in capsys.readouterr().out


def test_printer_prints_nothing_when_there_are_no_units(capsys):
    _report_max_unit_basis("Maximum excited basis size per unit", [])
    assert capsys.readouterr().out == ""


# --- the Cartesian-tensor / off-diagonal driver (XAS, NIXS tensor) ----------------------


def test_offdiag_driver_reports_its_one_transition_block(capsys):
    """``calc_Greens_function_with_offdiag`` enumerates a single operator group -- the whole
    transition block shares one recurrence -- so its eigenstate chunks collapse to one row."""
    from impurityModel.ed.greens_function import calc_Greens_function_with_offdiag

    calc_Greens_function_with_offdiag(
        HOP,
        [ManyBodyOperator({((0, "a"),): 1.0}), ManyBodyOperator({((1, "a"),): 1.0})],
        [ManyBodyState({SlaterDeterminant.from_bytes(GROUND): 1.0})],
        [0.0],
        _basis(NON_BINDING_CAP),
        0.1,
        occ_cutoff=1e-9,
        slaterWeightMin=0.0,
        verbose=False,
        sparse=True,
        dN_imp={0: (3, 3)},
        dN_val={0: (0, 0)},
        dN_con={0: (0, 0)},
        unit_report_label="XAS, core l=1 (Cartesian tensor)",
    )
    out = capsys.readouterr().out
    assert "Maximum excited basis size per unit -- XAS, core l=1 (Cartesian tensor):" in out
    rows, _ = _report_lines(out)
    assert len(rows) == 1 and rows[0].startswith("transition block")
    assert _sizes(rows)["transition block"] > _seed_support_size()
